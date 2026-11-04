"""Quick Edit tab: mask-based lightweight edits. Every save is an *Edited Variant* or *Transparent Cutout*."""
from __future__ import annotations

import time

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox, QFileDialog,
                               QGridLayout, QHBoxLayout, QLabel, QMessageBox, QPushButton, QScrollArea, QSlider,
                               QSplitter, QVBoxLayout, QWidget)

from ..core import ops as cops
from ..core.engine import SourceRecord, Studio
from ..core.imageio import SUPPORTED_EXTS
from ..core.recipes import Recipe
from ..quickedit.edits import MaskDoc, recolor, remove_object, replace_background
from ..quickedit.matting import Matting, adjust_edge, decontaminate
from .common import mask_overlay, open_folder, run_bg, section_label, to_qimage, wrap_label
from .widgets.compare_view import CompareView


def _slider(lo, hi, val):
    s = QSlider(Qt.Orientation.Horizontal)
    s.setRange(lo, hi)
    s.setValue(val)
    return s


class QuickEditPanel(QWidget):
    status = Signal(str)
    exported = Signal()

    def __init__(self, studio: Studio, parent=None):
        super().__init__(parent)
        self.studio = studio
        self.src: SourceRecord | None = None
        self.rgb: np.ndarray | None = None
        self.alpha: np.ndarray | None = None
        self.mask: MaskDoc | None = None
        self.history: list[tuple[np.ndarray, np.ndarray | None, str]] = []
        self.future: list[tuple[np.ndarray, np.ndarray | None, str]] = []
        self.edits: list[str] = []
        self.generative = False
        self.recolor_color = QColor(47, 111, 214)
        self.bg_color = QColor(255, 255, 255)
        self._build()

    # ------------------------------------------------------------------ layout
    def _build(self):
        root = QHBoxLayout(self)
        split = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        tb = QHBoxLayout()
        b = QPushButton("Open…")
        b.clicked.connect(self.open_dialog)
        tb.addWidget(b)
        self.tools = QButtonGroup(self)
        for i, (lab, t) in enumerate((("Pan/Zoom", "view"), ("Brush +", "add"), ("Brush −", "sub"),
                                      ("Magic select", "magic"), ("Select rect", "rect"))):
            bt = QPushButton(lab, checkable=True, checked=(t == "view"))
            bt.clicked.connect(lambda _=False, tt=t: self._set_tool(tt))
            self.tools.addButton(bt, i)
            tb.addWidget(bt)
        tb.addWidget(QLabel("Size"))
        self.brush = _slider(1, 200, 24)
        self.brush.valueChanged.connect(lambda v: setattr(self.view, "brush_radius_src", float(v)))
        tb.addWidget(self.brush)
        self.show_mask = QCheckBox("Show mask")
        self.show_mask.setChecked(True)
        self.show_mask.toggled.connect(self._refresh)
        tb.addWidget(self.show_mask)
        tb.addStretch(1)
        ll.addLayout(tb)
        self.view = CompareView()
        self.view.labels = ("Before · Source", "After · Edited Variant (unsaved)")
        self.view.empty_text = "Drop an image to edit (edits are saved as labelled derivatives)"
        self.view.filesDropped.connect(lambda fs: self.open_path(fs[0]))
        self.view.imageClicked.connect(self._clicked)
        self.view.imageDragged.connect(self._dragged)
        self.view.regionSelected.connect(self._rect_selected)
        self.view.brush_radius_src = 24.0
        self.view.mode = "after"
        ll.addWidget(self.view, 1)
        vm = QHBoxLayout()
        for lab, m in (("Edited", "after"), ("Wipe", "wipe"), ("Original", "before"), ("Side by side", "side")):
            bt = QPushButton(lab)
            bt.clicked.connect(lambda _=False, mm=m: self.view.set_mode(mm))
            vm.addWidget(bt)
        vm.addStretch(1)
        ll.addLayout(vm)
        split.addWidget(left)

        side = QWidget()
        sl = QVBoxLayout(side)
        sl.addWidget(section_label("Selection mask"))
        g = QGridLayout()
        btns = [("Auto person", lambda: self._auto_mask("u2net_human_seg")),
                ("Auto subject", lambda: self._auto_mask("isnet_general")), ("Invert", lambda: self._mask_op("invert")),
                ("Clear", lambda: self._mask_op("clear")), ("Grow", lambda: self._mask_op("grow", 3)),
                ("Shrink", lambda: self._mask_op("grow", -3)), ("Feather", lambda: self._mask_op("feather", 2.0)),
                ("Undo mask", lambda: self._mask_op("undo")), ("Redo mask", lambda: self._mask_op("redo"))]
        for i, (lab, fn) in enumerate(btns):
            b = QPushButton(lab)
            b.clicked.connect(fn)
            g.addWidget(b, i // 3, i % 3)
        sl.addLayout(g)
        row = QHBoxLayout()
        row.addWidget(QLabel("Magic tolerance"))
        self.tol = _slider(2, 60, 14)
        row.addWidget(self.tol)
        sl.addLayout(row)
        self.mask_info = wrap_label("")
        sl.addWidget(self.mask_info)

        sl.addWidget(section_label("Extract person / remove background"))
        row = QHBoxLayout()
        row.addWidget(QLabel("Edge feather"))
        self.feather = QDoubleSpinBox()
        self.feather.setRange(0, 20)
        self.feather.setSingleStep(0.5)
        row.addWidget(self.feather)
        row.addWidget(QLabel("Edge shift"))
        self.shift = QDoubleSpinBox()
        self.shift.setRange(-10, 10)
        self.shift.setDecimals(0)
        row.addWidget(self.shift)
        sl.addLayout(row)
        self.decon = QCheckBox("Reduce edge colour fringing")
        self.decon.setChecked(True)
        sl.addWidget(self.decon)
        row = QHBoxLayout()
        b = QPushButton("Cut out (mask → transparency)")
        b.clicked.connect(self.cutout)
        row.addWidget(b)
        sl.addLayout(row)
        row = QHBoxLayout()
        self.bg_mode = QComboBox()
        self.bg_mode.addItem("Solid colour", "color")
        self.bg_mode.addItem("Blurred original", "blur")
        self.bg_mode.addItem("Image…", "image")
        row.addWidget(self.bg_mode)
        b = QPushButton("Colour…")
        b.clicked.connect(lambda: self._pick("bg"))
        row.addWidget(b)
        b = QPushButton("Replace background")
        b.clicked.connect(self.replace_bg)
        row.addWidget(b)
        sl.addLayout(row)

        sl.addWidget(section_label("Recolour clothing / item (inside mask)"))
        row = QHBoxLayout()
        self.color_btn = QPushButton("Target colour…")
        self.color_btn.clicked.connect(lambda: self._pick("recolor"))
        row.addWidget(self.color_btn)
        row.addWidget(QLabel("Strength"))
        self.rc_strength = _slider(0, 100, 100)
        row.addWidget(self.rc_strength)
        sl.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("Match lightness"))
        self.rc_light = _slider(0, 100, 0)
        row.addWidget(self.rc_light)
        b = QPushButton("Apply recolour")
        b.clicked.connect(self.do_recolor)
        row.addWidget(b)
        sl.addLayout(row)
        sl.addWidget(wrap_label("Hue/chroma move to the target; per-pixel shading, folds and printed patterns "
                                "are preserved (lightness variation is kept)."))

        sl.addWidget(section_label("Remove small object (inside mask)"))
        row = QHBoxLayout()
        self.rm_method = QComboBox()
        self.rm_method.addItem("LaMa – generative fill (labelled)", "lama")
        self.rm_method.addItem("Classical fill (Telea)", "telea")
        row.addWidget(self.rm_method, 1)
        b = QPushButton("Remove")
        b.clicked.connect(self.do_remove)
        row.addWidget(b)
        sl.addLayout(row)

        sl.addWidget(section_label("Crop / rotate / straighten"))
        row = QHBoxLayout()
        for lab, k in (("⟲ 90°", 3), ("⟳ 90°", 1)):
            b = QPushButton(lab)
            b.clicked.connect(lambda _=False, kk=k: self._geometry(rotate90=kk))
            row.addWidget(b)
        b = QPushButton("Flip")
        b.clicked.connect(lambda: self._geometry(flip=True))
        row.addWidget(b)
        b = QPushButton("Crop to rect")
        b.clicked.connect(self._crop_rect)
        row.addWidget(b)
        sl.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("Angle °"))
        self.angle = QDoubleSpinBox()
        self.angle.setRange(-45, 45)
        self.angle.setSingleStep(0.1)
        row.addWidget(self.angle)
        b = QPushButton("Detect")
        b.clicked.connect(self._detect_angle)
        row.addWidget(b)
        b = QPushButton("Straighten")
        b.clicked.connect(lambda: self._geometry(angle=self.angle.value()))
        row.addWidget(b)
        sl.addLayout(row)

        sl.addWidget(section_label("History & save"))
        row = QHBoxLayout()
        for lab, fn in (("Undo edit", self.undo), ("Redo edit", self.redo), ("Reset to original", self.reset)):
            b = QPushButton(lab)
            b.clicked.connect(fn)
            row.addWidget(b)
        sl.addLayout(row)
        self.hist_info = wrap_label("")
        sl.addWidget(self.hist_info)
        row = QHBoxLayout()
        b = QPushButton("Save Edited Variant")
        b.clicked.connect(lambda: self.save("edited"))
        row.addWidget(b)
        b = QPushButton("Save Transparent Cutout (PNG)")
        b.clicked.connect(lambda: self.save("cutout"))
        row.addWidget(b)
        sl.addLayout(row)
        self.save_info = wrap_label("")
        self.save_info.setTextFormat(Qt.TextFormat.RichText)
        sl.addWidget(self.save_info)
        self.open_btn = QPushButton("Open folder")
        self.open_btn.setEnabled(False)
        self.open_btn.clicked.connect(lambda: open_folder(self._last))
        sl.addWidget(self.open_btn)
        sl.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidget(side)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(400)
        split.addWidget(scroll)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)
        root.addWidget(split)
        self._last = None
        self._rect = None
        for seq, fn in (("Ctrl+Z", self.undo), ("Ctrl+Y", self.redo), ("Ctrl+Shift+Z", self.redo),
                        ("B", lambda: self._set_tool("add")), ("E", lambda: self._set_tool("sub")),
                        ("W", lambda: self._set_tool("magic")), ("V", lambda: self._set_tool("view")),
                        ("Ctrl+I", lambda: self._mask_op("invert"))):
            sc = QShortcut(QKeySequence(seq), self)
            sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            sc.activated.connect(fn)

    # ------------------------------------------------------------------ load
    def open_dialog(self):
        exts = " ".join(f"*{e}" for e in sorted(SUPPORTED_EXTS))
        f, _ = QFileDialog.getOpenFileName(self, "Open image", "", f"Images ({exts})")
        if f:
            self.open_path(f)

    def open_path(self, path: str):
        run_bg(lambda progress, cancel: self.studio.inspect(path), on_done=self._loaded, on_error=self._error)

    def load_array(self, src: SourceRecord, rgb: np.ndarray | None = None):
        self._loaded(src, rgb)

    def _loaded(self, src: SourceRecord, rgb: np.ndarray | None = None):
        self.src = src
        self.rgb = (rgb if rgb is not None else src.image.rgb).copy()
        self.alpha = src.image.alpha.copy() if src.image.alpha is not None else None
        self.mask = MaskDoc(self.rgb.shape[0], self.rgb.shape[1])
        self.history.clear()
        self.future.clear()
        self.edits.clear()
        self.generative = False
        self._orig_q = to_qimage(src.image.rgb, src.image.icc_profile, src.image.alpha)
        self._refresh()
        self.status.emit(f"Quick Edit: {src.path.name}")

    # ------------------------------------------------------------------ view
    def _refresh(self):
        if self.src is None:
            return
        after = to_qimage(self.rgb, self.src.image.icc_profile, self.alpha)
        same_geom = self.rgb.shape[:2] == self.src.image.rgb.shape[:2]
        self.view.set_images(self._orig_q if same_geom else after, after, keep_view=True)
        if self.mask is not None and self.show_mask.isChecked() and self.mask.mask.any():
            self.view.set_overlay(mask_overlay(self.mask.mask))
        else:
            self.view.set_overlay(None)
        m = self.mask.mask if self.mask is not None else None
        self.mask_info.setText(f"Mask covers {float(m.mean()) * 100:.1f}% of the image." if m is not None else "")
        self.hist_info.setText(("Edits: " + " → ".join(self.edits)) if self.edits else "No edits yet.")

    def _set_tool(self, t: str):
        self._tool = t
        self.view.tool = {"view": "view", "add": "paint", "sub": "paint", "magic": "click", "rect": "select"}[t]
        for b in self.tools.buttons():
            b.setChecked(b.text().lower().startswith({"view": "pan", "add": "brush +", "sub": "brush −",
                                                      "magic": "magic", "rect": "select"}[t]))
        self.view.update()

    def _clicked(self, x: float, y: float, buttons: int):
        if self.mask is None:
            return
        t = getattr(self, "_tool", "view")
        if t in ("add", "sub"):
            self.mask.begin_stroke()
            self.mask.brush(x, y, self.brush.value(), add=(t == "add") and buttons != 2)
            self._refresh()
        elif t == "magic":
            self.mask.magic_select(self.rgb, int(x), int(y), tolerance=float(self.tol.value()))
            self._refresh()

    def _dragged(self, x0, y0, x1, y1, buttons):
        t = getattr(self, "_tool", "view")
        if self.mask is not None and t in ("add", "sub"):
            self.mask.brush(x0, y0, self.brush.value(), add=(t == "add") and buttons != 2, x2=x1, y2=y1)
            self._refresh()

    def _rect_selected(self, r):
        self._rect = [int(r.left()), int(r.top()), int(r.right()), int(r.bottom())]
        if self.mask is not None:
            m = np.zeros_like(self.mask.mask)
            x0, y0, x1, y1 = self._rect
            m[max(0, y0):y1, max(0, x0):x1] = 1
            self.mask.set(np.maximum(self.mask.mask, m))
            self._refresh()

    # ------------------------------------------------------------------ mask ops
    def _mask_op(self, op: str, arg=None):
        if self.mask is None:
            return
        if op == "invert":
            self.mask.invert()
        elif op == "clear":
            self.mask.clear()
        elif op == "grow":
            self.mask.grow(int(arg))
        elif op == "feather":
            self.mask.feather(float(arg))
        elif op == "undo":
            self.mask.undo()
        elif op == "redo":
            self.mask.redo()
        self._refresh()

    def _auto_mask(self, model_id: str):
        if self.rgb is None:
            return
        rgb = self.rgb
        self.status.emit("Computing mask…")
        run_bg(lambda progress, cancel: Matting.person_mask(rgb, model_id),
               on_done=lambda m: (self.mask.set(m), self._refresh(), self.status.emit("Mask ready (editable).")),
               on_error=self._error)

    # ------------------------------------------------------------------ edits
    def _push(self, label: str):
        self.history.append((self.rgb.copy(), None if self.alpha is None else self.alpha.copy(), label))
        self.history = self.history[-12:]
        self.future.clear()

    def _need_mask(self) -> bool:
        if self.mask is None or not self.mask.mask.any():
            QMessageBox.information(self, "No selection", "Create a mask first (Auto person, brush or magic select).")
            return False
        return True

    def cutout(self):
        if self.rgb is None or not self._need_mask():
            return
        self._push("cutout")
        a = adjust_edge(self.mask.mask, self.feather.value(), int(self.shift.value()))
        if self.alpha is not None:
            a = np.minimum(a, self.alpha)
        if self.decon.isChecked():
            self.rgb = decontaminate(self.rgb, a)
        self.alpha = a
        self.edits.append(f"cutout(feather {self.feather.value():g}, shift {self.shift.value():g})")
        self._refresh()

    def replace_bg(self):
        if self.rgb is None:
            return
        a = self.alpha if self.alpha is not None else (self.mask.mask if self.mask.mask.any() else None)
        if a is None:
            QMessageBox.information(self, "No subject", "Cut out the subject or create a mask first.")
            return
        mode = self.bg_mode.currentData()
        bg = None
        if mode == "image":
            f, _ = QFileDialog.getOpenFileName(self, "Background image", "", "Images (*.jpg *.jpeg *.png *.tif *.tiff)")
            if not f:
                return
            from ..core.imageio import load_image
            bg = load_image(f).rgb
        self._push("background")
        c = self.bg_color
        self.rgb = replace_background(self.rgb, a, mode, (c.redF(), c.greenF(), c.blueF()), background=bg)
        self.alpha = None
        self.edits.append(f"background:{mode}")
        self._refresh()

    def do_recolor(self):
        if self.rgb is None or not self._need_mask():
            return
        self._push("recolor")
        c = self.recolor_color
        self.rgb = recolor(self.rgb, self.mask.mask, (c.redF(), c.greenF(), c.blueF()),
                           strength=self.rc_strength.value() / 100, match_lightness=self.rc_light.value() / 100)
        self.edits.append(f"recolor:{c.name()}")
        self._refresh()

    def do_remove(self):
        if self.rgb is None or not self._need_mask():
            return
        rgb, m, method = self.rgb, self.mask.mask.copy(), self.rm_method.currentData()

        def done(res):
            out, used = res
            self._push("remove")
            self.rgb = out
            self.edits.append(f"remove[{used}]")
            if "LaMa" in used:
                self.generative = True
            self.mask.clear()
            self._refresh()
            self.status.emit(f"Object removed with {used}.")
        run_bg(lambda progress, cancel: remove_object(rgb, m, method), on_done=done, on_error=self._error)

    def _geometry(self, **kw):
        if self.rgb is None:
            return
        self._push("geometry")
        p = {"angle": 0.0, "crop": None, "rotate90": 0, "flip": False, **kw}
        self.rgb = cops.crop_rotate(self.rgb, **p)
        if self.alpha is not None:
            self.alpha = cops.crop_rotate(self.alpha[..., None], **p)[..., 0]
        self.mask = MaskDoc(self.rgb.shape[0], self.rgb.shape[1])
        self.edits.append("geometry(" + ", ".join(f"{k}={v}" for k, v in kw.items()) + ")")
        self.view.fit = True
        self._refresh()

    def _crop_rect(self):
        if self._rect is None:
            QMessageBox.information(self, "Crop", "Use 'Select rect' and drag a rectangle first.")
            return
        self._geometry(crop=self._rect)
        self._rect = None

    def _detect_angle(self):
        if self.rgb is not None:
            a = cops.estimate_straighten_angle(self.rgb)
            if a is None:
                self.status.emit("No dominant horizontal/vertical lines found.")
            else:
                self.angle.setValue(a)
                self.status.emit(f"Suggested straighten angle {a:+.2f}° (not applied yet).")

    def _pick(self, which: str):
        c = QColorDialog.getColor(self.recolor_color if which == "recolor" else self.bg_color, self)
        if c.isValid():
            if which == "recolor":
                self.recolor_color = c
                self.color_btn.setStyleSheet(f"background:{c.name()}")
            else:
                self.bg_color = c

    def undo(self):
        if self.history:
            self.future.append((self.rgb, self.alpha, self.edits[-1] if self.edits else ""))
            self.rgb, self.alpha, _ = self.history.pop()
            if self.edits:
                self.edits.pop()
            if self.mask is None or self.mask.mask.shape != self.rgb.shape[:2]:
                self.mask = MaskDoc(self.rgb.shape[0], self.rgb.shape[1])
            self._refresh()

    def redo(self):
        if self.future:
            self.history.append((self.rgb, self.alpha, ""))
            self.rgb, self.alpha, lab = self.future.pop()
            self.edits.append(lab)
            if self.mask is None or self.mask.mask.shape != self.rgb.shape[:2]:
                self.mask = MaskDoc(self.rgb.shape[0], self.rgb.shape[1])
            self._refresh()

    def reset(self):
        if self.src is not None:
            self._loaded(self.src)

    # ------------------------------------------------------------------ save
    def save(self, kind: str):
        if self.rgb is None or self.src is None:
            return
        if kind == "cutout" and self.alpha is None:
            QMessageBox.information(self, "No transparency", "Use 'Cut out' first to create transparency.")
            return
        r = Recipe(created_by="quickedit", rationale=["Quick Edit: " + " → ".join(self.edits or ["(no edits)"])])
        r.ops = []
        label = "quickedit:" + ";".join(self.edits)[:200]
        o = self.studio.export(self.src, self.rgb, self.alpha, r, "cutout" if kind == "cutout" else "edited",
                               label=label, edited=True, generative=self.generative,
                               review_flags=(["Contains generative fill (LaMa)"] if self.generative else []))
        self._last = o.path
        self.open_btn.setEnabled(True)
        a = o.audit
        self.save_info.setText(f"Saved <b>{'Transparent Cutout' if kind == 'cutout' else 'Edited Variant'}</b> "
                               f"→ {o.path.name}<br>metadata {'verified' if a.get('verified') else 'NOT verified'}; "
                               f"labelled as edited{' + generative' if self.generative else ''} in XMP.")
        self.exported.emit()

    def _error(self, msg: str):
        QMessageBox.warning(self, "Quick Edit failed", msg[:3000])

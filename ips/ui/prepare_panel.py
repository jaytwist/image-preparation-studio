"""Prepare tab: one image -> Smart Auto -> compare -> Manual -> export."""
from __future__ import annotations

import html
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QDialog, QFileDialog, QHBoxLayout, QInputDialog,
                               QLabel, QListWidget, QListWidgetItem, QMessageBox, QProgressBar, QPushButton,
                               QScrollArea, QSplitter, QVBoxLayout, QWidget)

from ..core.engine import SourceRecord, Studio
from ..core.imageio import SUPPORTED_EXTS, icc_description
from ..core.outputs import KIND_LABELS
from ..core.pipeline import ProcessResult
from ..core.recipes import Recipe
from .common import Collapsible, open_folder, run_bg, section_label, to_qimage, wrap_label
from .recipe_editor import RecipeEditor
from .widgets.compare_view import CompareView


@dataclass
class Variant:
    label: str
    recipe: Recipe
    result: ProcessResult
    seconds: float
    load_seconds: float
    qimage: Any = None
    created: float = field(default_factory=time.time)


def _sev_color(level: str) -> str:
    return {"strong": "#e57373", "moderate": "#ffb74d", "mild": "#c5e1a5", "trace": "#9e9e9e"}.get(level, "#bbb")


class PreparePanel(QWidget):
    status = Signal(str)
    exported = Signal()

    def __init__(self, studio: Studio, parent=None):
        super().__init__(parent)
        self.studio = studio
        self.src: SourceRecord | None = None
        self.src_q = None
        self.variants: list[Variant] = []
        self.worker = None
        self.batch_job: tuple[Any, int] | None = None  # (JobManager, job_id) when opened from Batch
        self._build()

    # ------------------------------------------------------------------ layout
    def _build(self):
        root = QVBoxLayout(self)
        tb = QHBoxLayout()
        b = QPushButton("Open…")
        b.clicked.connect(self.open_dialog)
        tb.addWidget(b)
        self.view_group = QButtonGroup(self)
        for i, (lab, m) in enumerate((("Wipe", "wipe"), ("Before", "before"), ("After", "after"), ("Side by side", "side"))):
            bt = QPushButton(lab, checkable=True, checked=(m == "wipe"))
            bt.clicked.connect(lambda _=False, mm=m: self.view.set_mode(mm))
            self.view_group.addButton(bt, i)
            tb.addWidget(bt)
        tb.addSpacing(12)
        for lab, pct in (("Fit", None), ("100%", 100), ("200%", 200), ("400%", 400)):
            bt = QPushButton(lab)
            bt.clicked.connect(lambda _=False, p=pct: self.view.set_zoom_percent(p))
            tb.addWidget(bt)
        self.faces_combo = QComboBox()
        self.faces_combo.addItem("Detail crop…")
        self.faces_combo.activated.connect(self._focus_face)
        tb.addWidget(self.faces_combo)
        tb.addSpacing(12)
        tb.addWidget(QLabel("Left"))
        self.left_sel = QComboBox()
        self.left_sel.currentIndexChanged.connect(self._refresh_view)
        tb.addWidget(self.left_sel)
        tb.addWidget(QLabel("Right"))
        self.right_sel = QComboBox()
        self.right_sel.currentIndexChanged.connect(self._refresh_view)
        tb.addWidget(self.right_sel)
        tb.addStretch(1)
        fs = QPushButton("Full screen")
        fs.clicked.connect(self._fullscreen)
        tb.addWidget(fs)
        root.addLayout(tb)

        split = QSplitter(Qt.Orientation.Horizontal)
        self.view = CompareView()
        self.view.filesDropped.connect(lambda fs: self.open_path(fs[0]))
        self.view.regionSelected.connect(self._crop_preview)
        split.addWidget(self.view)

        side = QWidget()
        sl = QVBoxLayout(side)
        sl.setContentsMargins(6, 0, 6, 0)
        self.src_info = wrap_label("No image loaded.")
        sl.addWidget(section_label("Source"))
        sl.addWidget(self.src_info)
        sl.addWidget(section_label("Smart Auto"))
        row = QHBoxLayout()
        self.mode = QComboBox()
        self.mode.addItem("Faithful (default)", "faithful")
        self.mode.addItem("Balanced (GAN texture – can smooth skin/alter faces; always reviewed)", "balanced")
        self.mode.addItem("Generative Recovery (not available in v1)", "generative")
        self.mode.model().item(2).setEnabled(False)
        row.addWidget(self.mode, 2)
        self.scale = QComboBox()
        for lab, v in (("Scale: Auto", "auto"), ("×1", 1), ("×2", 2), ("×4", 4)):
            self.scale.addItem(lab, v)
        row.addWidget(self.scale, 1)
        sl.addLayout(row)
        self.look = QComboBox()
        for lab, v in [("Look: Enhanced (visible, non-generative)", "enhanced"), ("Look: Vivid (strong)", "vivid"),
                         ("Look: Natural (repair defects only)", "natural")]:
            self.look.addItem(lab, v)
        self.look.setCurrentIndex(max(0, self.look.findData(getattr(self.studio.settings, "default_look", "enhanced"))))
        self.look.setToolTip("Enhanced/Vivid add tone, local contrast, vibrance (skin protected) and halo-free "
                             "sharpening. No generative model; faces are identity-checked.")
        self.look.activated.connect(lambda _i: self.src is not None and self.run_auto())
        sl.addWidget(self.look)
        self.override = QCheckBox("Quality Override (allow jobs outside the measured 15 s Fast Profile)")
        self.override.setChecked(self.studio.settings.quality_override_default)
        sl.addWidget(self.override)
        row = QHBoxLayout()
        self.btn_auto = QPushButton("Auto Prepare")
        self.btn_auto.setStyleSheet("font-weight:600; padding:6px;")
        self.btn_auto.clicked.connect(self.run_auto)
        row.addWidget(self.btn_auto, 2)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel)
        row.addWidget(self.btn_cancel, 1)
        sl.addLayout(row)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(True)
        self.progress.setFormat("idle")
        sl.addWidget(self.progress)
        self.diag = wrap_label("")
        sl.addWidget(self.diag)
        self.why = wrap_label("")
        self.why.setTextFormat(Qt.TextFormat.RichText)
        sl.addWidget(Collapsible("Why this recipe?", self.why, expanded=True))
        self.flags = wrap_label("")
        self.flags.setTextFormat(Qt.TextFormat.RichText)
        sl.addWidget(self.flags)

        man = QWidget()
        ml = QVBoxLayout(man)
        ml.setContentsMargins(0, 0, 0, 0)
        self.editor = RecipeEditor()
        ml.addWidget(self.editor)
        row = QHBoxLayout()
        b = QPushButton("Run Manual")
        b.clicked.connect(self.run_manual)
        row.addWidget(b)
        b = QPushButton("Reset to Auto")
        b.clicked.connect(self.editor.reset_to_auto)
        row.addWidget(b)
        ml.addLayout(row)
        row = QHBoxLayout()
        self.presets = QComboBox()
        self._load_presets()
        row.addWidget(self.presets, 1)
        b = QPushButton("Load")
        b.clicked.connect(self._apply_preset)
        row.addWidget(b)
        b = QPushButton("Save preset…")
        b.clicked.connect(self._save_preset)
        row.addWidget(b)
        ml.addLayout(row)
        tip = QLabel("Tip: Shift+drag on the image runs the current recipe on that region only (crop preview).")
        tip.setWordWrap(True)
        tip.setStyleSheet("color:#8a8f98; font-size:11px")
        ml.addWidget(tip)
        self.batch_btn = QPushButton("Use this recipe for the batch item")
        self.batch_btn.setVisible(False)
        self.batch_btn.clicked.connect(self._apply_batch_override)
        ml.addWidget(self.batch_btn)
        sl.addWidget(Collapsible("Manual / Advanced", man, expanded=False))

        sl.addWidget(section_label("Variants (history)"))
        self.hist = QListWidget()
        self.hist.setMaximumHeight(130)
        self.hist.itemDoubleClicked.connect(self._hist_to_editor)
        sl.addWidget(self.hist)

        sl.addWidget(section_label("Export (right-hand variant)"))
        self.exp_checks = {}
        for key, lab in (("archive", "Lossless Archive Master (TIFF/PNG, source ICC, full metadata)"),
                         ("ai_ready", "AI / Video Ready (sRGB PNG, fit long edge, no crop)"),
                         ("share", "Share / Web (sRGB JPEG)")):
            c = QCheckBox(lab)
            c.setChecked(key == "archive")
            self.exp_checks[key] = c
            sl.addWidget(c)
        self.strip_gps = QCheckBox("Remove location metadata (Share / Web)")
        self.strip_gps.setChecked(self.studio.settings.share_strip_location)
        sl.addWidget(self.strip_gps)
        row = QHBoxLayout()
        b = QPushButton("Export")
        b.clicked.connect(self.export)
        row.addWidget(b)
        self.open_out = QPushButton("Open folder")
        self.open_out.setEnabled(False)
        self.open_out.clicked.connect(lambda: open_folder(self._last_export))
        row.addWidget(self.open_out)
        sl.addLayout(row)
        self.export_info = wrap_label("")
        self.export_info.setTextFormat(Qt.TextFormat.RichText)
        sl.addWidget(self.export_info)
        sl.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidget(side)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(400)
        split.addWidget(scroll)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)
        root.addWidget(split, 1)
        self._last_export = None
        for seq, fn in (("Ctrl+O", self.open_dialog), ("Ctrl+Return", self.run_auto), ("Ctrl+E", self.export),
                        ("Ctrl+Z", self.editor.undo), ("Ctrl+Shift+Z", self.editor.redo), ("Ctrl+Y", self.editor.redo),
                        ("F", lambda: self.view.set_zoom_percent(None)), ("1", lambda: self.view.set_zoom_percent(100)),
                        ("2", lambda: self.view.set_zoom_percent(200)), ("4", lambda: self.view.set_zoom_percent(400)),
                        ("Space", self._toggle_ab), ("F11", self._fullscreen), ("Escape", self._cancel)):
            sc = QShortcut(QKeySequence(seq), self)
            sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            sc.activated.connect(fn)

    # ------------------------------------------------------------------ loading
    def open_dialog(self):
        exts = " ".join(f"*{e}" for e in sorted(SUPPORTED_EXTS))
        f, _ = QFileDialog.getOpenFileName(self, "Open image", "", f"Images ({exts});;All files (*)")
        if f:
            self.open_path(f)

    def open_path(self, path: str, batch_job: tuple[Any, int] | None = None, rel_dir: str = ""):
        self.batch_job = batch_job
        self.batch_btn.setVisible(batch_job is not None)
        self._busy(True, f"Reading {Path(path).name}…")
        self.worker = run_bg(lambda progress, cancel: self.studio.inspect(path, rel_dir=rel_dir),
                             on_done=self._loaded, on_error=self._error)

    def _loaded(self, src: SourceRecord):
        self._busy(False)
        self.src = src
        self.variants.clear()
        self.hist.clear()
        self.src_q = to_qimage(src.image.rgb, src.image.icc_profile, src.image.alpha)
        im, s = src.image, src.summary
        warn = []
        warn += src.image.decode_notes
        warn += [f"metadata: {e}" for e in s.get("errors", [])]
        if s.get("exif_dimension_mismatch"):
            warn.append("EXIF dimensions differ from pixels (resized/cropped earlier)")
        lines = [f"<b>{html.escape(src.path.name)}</b>",
                 f"{im.source_format} · {im.width}×{im.height} · {im.bit_depth}-bit"
                 f"{' · alpha' if im.alpha is not None else ''} · ICC: {html.escape(icc_description(im.icc_profile))}",
                 f"Camera: {html.escape(str(s.get('camera_make') or '—'))} {html.escape(str(s.get('camera_model') or ''))}"
                 f" · ISO {s.get('iso') or '—'} · Taken {html.escape(str(s.get('date_taken') or '—'))}",
                 f"GPS: {'yes' if s.get('has_gps') else 'no'} · Software: {html.escape(str(s.get('software') or '—'))}",
                 f"SHA-256 {src.sha256[:16]}…"]
        if warn:
            lines.append("<span style='color:#ffb74d'>⚠ " + "<br>⚠ ".join(html.escape(w) for w in warn) + "</span>")
        self.src_info.setTextFormat(Qt.TextFormat.RichText)
        self.src_info.setText("<br>".join(lines))
        self.faces_combo.clear()
        self.faces_combo.addItem("Detail crop…")
        for i, f in enumerate(src.diag.faces):
            self.faces_combo.addItem(f"Face {i + 1} ({int(f.size)} px)", i)
        self.view.labels = ("Before · Source", "After")
        self.view.set_images(self.src_q, None)
        self._update_selectors()
        auto = self.studio.auto_recipe(src, self._request())
        self.editor.set_recipe(auto, auto=auto)
        self._show_assessment(auto)
        self.status.emit(f"Loaded {src.path.name}")
        if getattr(self.studio.settings, "auto_run_on_open", True):
            self._run(auto, "Auto")

    def _request(self):
        return self.studio.request(mode=self.mode.currentData(), scale=self.scale.currentData(),
                                   quality_override=self.override.isChecked(), look=self.look.currentData())

    def _show_assessment(self, r: Recipe):
        a = self.src.assessment
        parts = [f"<b>{a.content_class}</b> (confidence {a.content_confidence:.2f}) · diagnose "
                 f"{self.src.timings_ms.get('diagnose', 0):.0f} ms"]
        for i in a.issues:
            parts.append(f"<span style='color:{_sev_color(i.level)}'>● {i.code} — {i.level}</span> "
                         f"<span style='color:#8a8f98'>{html.escape(i.evidence)} (conf {i.confidence:.2f})</span>")
        if a.intents:
            parts.append("Preserve: " + ", ".join(a.intents))
        if not a.issues:
            parts.append("No degradations measured.")
        self.diag.setTextFormat(Qt.TextFormat.RichText)
        self.diag.setText("<br>".join(parts))
        why = [html.escape(x) for x in r.rationale]
        why += [f"<b>{html.escape(o.kind)}</b>: {html.escape(o.why)}" for o in r.ops if o.enabled and o.why]
        why.append(f"<b>Recipe</b> {r.hash}: {html.escape(r.summary())}")
        why.append(f"Prior time estimate: {r.predicted_seconds} s on {html.escape(self.studio.device.name)} "
                   f"({'certified' if self.studio.profile.certified(self.studio.device.name) else 'not certified'})")
        self.why.setText("<br>".join("• " + w for w in why))
        self._show_flags(r.flags, None)

    def _show_flags(self, flags: list[str], identity: dict | None):
        out = []
        for f in flags:
            out.append(f"<span style='color:#ffb74d'>⚑ {html.escape(f)}</span>")
        if identity:
            for fd in identity.get("faces", []):
                es = fd.get("embedding_similarity")
                out.append(f"Face {fd['index'] + 1}: similarity {es:.3f} · landmark drift "
                           f"{(fd.get('landmark_drift_iod') or 0):.3f} IOD · SSIM {(fd.get('region_ssim') or 0):.3f} · "
                           f"texture ×{(fd.get('fine_texture_ratio') or 0):.2f}/{(fd.get('mid_texture_ratio') or 0):.2f}"
                           if es is not None else f"Face {fd['index'] + 1}: {', '.join(fd.get('flags', [])) or 'ok'}")
            for n in identity.get("notes", []):
                out.append(f"<span style='color:#90a4ae'>ⓘ {html.escape(n)}</span>")
            out.append(f"<span style='color:#8a8f98'>{html.escape(identity.get('note', ''))}</span>")
        self.flags.setText("<br>".join(out) if out else "<span style='color:#81c784'>No review flags.</span>")

    # ------------------------------------------------------------------ running
    def run_auto(self):
        if not self.src:
            return
        auto = self.studio.auto_recipe(self.src, self._request())
        self.editor.set_recipe(auto, auto=auto, push=True)
        self._show_assessment(auto)
        self._run(auto, "Auto")

    def run_manual(self):
        if not self.src:
            return
        r = self.editor.recipe.clone()
        r.created_by = "manual" if (self.editor.auto_recipe is None or r.hash != self.editor.auto_recipe.hash) else "auto"
        self._run(r, "Manual" if r.created_by == "manual" else "Auto")

    def _run(self, recipe: Recipe, label: str):
        if recipe.outside_fast_profile and not recipe.quality_override and not self.override.isChecked() \
                and self.studio.profile.certified(self.studio.device.name):
            QMessageBox.information(self, "Outside Fast Profile",
                                    "This route/size is not certified ≤15 s on this GPU.\n\n" + "\n".join(recipe.flags)
                                    + "\n\nEnable Quality Override to run it anyway (it will be labelled), or choose "
                                      "a smaller scale / lighter recipe in Manual.")
            return
        if self.override.isChecked():
            recipe.quality_override = True
        self._busy(True, f"{label}: starting…")
        src = self.src
        t0 = time.perf_counter()

        def job(progress, cancel):
            res = self.studio.process(src, recipe, cancel=cancel, progress=progress)
            return recipe, res, time.perf_counter() - t0

        self.worker = run_bg(job, on_done=lambda r: self._processed(r, label), on_error=self._error,
                             on_progress=self._on_progress)

    def _processed(self, payload, label: str):
        recipe, res, secs = payload
        self._busy(False)
        load = sum(res.model_load_ms.values()) / 1000
        v = Variant(f"{label} {len(self.variants) + 1}", recipe, res, secs, load)
        v.qimage = to_qimage(res.rgb, self.src.image.icc_profile, res.alpha)
        self.variants.append(v)
        it = QListWidgetItem(f"{v.label} · {secs:.1f} s (load {load:.1f} s) · {recipe.summary()}")
        it.setToolTip(recipe.summary())
        self.hist.addItem(it)
        self._update_selectors(right=len(self.variants))
        self._show_flags(res.review_flags, res.identity)
        self.status.emit(f"{v.label}: {secs:.1f} s total ({load:.1f} s model load) · {recipe.summary()}")

    def _on_progress(self, stage: str, frac: float):
        self.progress.setValue(int(frac * 1000))
        self.progress.setFormat(f"{stage} — %p%")

    def _cancel(self):
        if self.worker:
            self.worker.cancel()
            self.progress.setFormat("cancelling…")

    def _busy(self, on: bool, text: str = ""):
        self.btn_auto.setEnabled(not on)
        self.btn_cancel.setEnabled(on)
        if on:
            self.progress.setValue(0)
            self.progress.setFormat(text)
        else:
            self.progress.setValue(1000 if self.variants else 0)
            self.progress.setFormat("done" if self.variants else "idle")
            self.worker = None

    def _error(self, msg: str):
        self._busy(False)
        if "Cancelled" in msg.split(":")[0]:
            self.status.emit("Cancelled.")
            return
        QMessageBox.warning(self, "Processing failed", msg[:3000])
        self.status.emit("Error: " + msg.splitlines()[0])

    # ------------------------------------------------------------------ comparison
    def _update_selectors(self, right: int | None = None):
        for combo in (self.left_sel, self.right_sel):
            combo.blockSignals(True)
            cur = combo.currentIndex()
            combo.clear()
            combo.addItem("Source", -1)
            for i, v in enumerate(self.variants):
                combo.addItem(v.label, i)
            combo.setCurrentIndex(max(0, cur) if cur < combo.count() else 0)
            combo.blockSignals(False)
        self.left_sel.setCurrentIndex(0) if right is not None else None
        if right is not None:
            self.right_sel.setCurrentIndex(right)
        self._refresh_view()

    def _img_for(self, idx: int):
        if idx < 0:
            return self.src_q, "Source"
        v = self.variants[idx]
        return v.qimage, f"{v.label} · Restored Master (unsaved)"

    def _refresh_view(self):
        if not self.src:
            return
        li = self.left_sel.currentData()
        ri = self.right_sel.currentData()
        a, la = self._img_for(li if li is not None else -1)
        b, lb = self._img_for(ri if ri is not None else -1)
        self.view.labels = (f"Before · {la}", f"After · {lb}")
        self.view.set_images(a, b if (ri is not None and ri >= 0) or li not in (None, -1) else None, keep_view=True)
        if ri is not None and ri >= 0:
            v = self.variants[ri]
            self._show_flags(v.result.review_flags, v.result.identity)

    def _toggle_ab(self):
        self.view.set_mode("before" if self.view.mode != "before" else "wipe")

    def _focus_face(self, i: int):
        if not self.src or i <= 0:
            return
        f = self.src.diag.faces[self.faces_combo.itemData(i)]
        self.view.focus_source_rect(*f.box)

    def _fullscreen(self):
        w = self.window()
        w.showNormal() if w.isFullScreen() else w.showFullScreen()

    def _crop_preview(self, r: QRectF):
        if not self.src:
            return
        box = (int(max(0, r.left())), int(max(0, r.top())), int(min(self.src.image.width, r.right())),
               int(min(self.src.image.height, r.bottom())))
        if box[2] - box[0] < 8 or box[3] - box[1] < 8:
            return
        recipe = self.editor.recipe.clone()
        self._busy(True, "Crop preview…")
        src = self.src

        def job(progress, cancel):
            t0 = time.perf_counter()
            a, b = self.studio.pipeline.preview_region(src.image.rgb, recipe, box, src.diag.faces, cancel=cancel,
                                                       progress=progress)
            return a, b, time.perf_counter() - t0

        def done(res):
            self._busy(False)
            a, b, secs = res
            dlg = QDialog(self)
            dlg.setWindowTitle(f"Crop preview {box} · {secs:.1f} s · {recipe.summary()}")
            lay = QVBoxLayout(dlg)
            cv = CompareView()
            cv.labels = ("Before · Source crop", "After · current recipe")
            cv.set_images(to_qimage(a, src.image.icc_profile), to_qimage(b, src.image.icc_profile))
            lay.addWidget(cv)
            dlg.resize(900, 700)
            dlg.show()
        self.worker = run_bg(job, on_done=done, on_error=self._error, on_progress=self._on_progress)

    def _hist_to_editor(self, item):
        v = self.variants[self.hist.row(item)]
        self.editor.set_recipe(v.recipe, push=True)

    # ------------------------------------------------------------------ presets
    def _load_presets(self):
        self.presets.clear()
        self.presets.addItem("Presets…", None)
        for row in self.studio.db.query("SELECT name FROM presets ORDER BY name"):
            self.presets.addItem(row["name"], row["name"])
        from ..paths import PROJECT_ROOT
        for f in sorted((PROJECT_ROOT / "recipes").glob("*.json")):
            self.presets.addItem(f"(sample) {f.stem}", str(f))

    def _apply_preset(self):
        key = self.presets.currentData()
        if not key:
            return
        import json
        if str(key).endswith(".json"):
            d = json.loads(Path(key).read_text("utf-8"))
        else:
            d = json.loads(self.studio.db.one("SELECT recipe_json FROM presets WHERE name=?", (key,))["recipe_json"])
        r = Recipe.from_json(d)
        r.created_by = f"preset:{Path(str(key)).stem}"
        self.editor.set_recipe(r, push=True)

    def _save_preset(self):
        name, ok = QInputDialog.getText(self, "Save preset", "Preset name:")
        if ok and name.strip():
            import json
            self.studio.db.execute("INSERT OR REPLACE INTO presets(name, recipe_json, created) VALUES(?,?,?)",
                                   (name.strip(), json.dumps(self.editor.recipe.to_json()), time.time()))
            self._load_presets()

    def _apply_batch_override(self):
        if self.batch_job:
            jm, jid = self.batch_job
            jm.set_override(jid, self.editor.recipe.clone())
            self.status.emit(f"Batch job {jid} will use this recipe (requeued).")

    # ------------------------------------------------------------------ export
    def export(self):
        if not self.src:
            return
        ri = self.right_sel.currentData()
        if ri is None or ri < 0:
            if not self.variants:
                QMessageBox.information(self, "Nothing to export", "Run Auto or Manual first.")
                return
            ri = len(self.variants) - 1
        v = self.variants[ri]
        profiles = [k for k, c in self.exp_checks.items() if c.isChecked()]
        if not profiles:
            return
        src, auto = self.src, self.editor.auto_recipe
        strip = self.strip_gps.isChecked()

        def job(progress, cancel):
            outs = []
            for i, pk in enumerate(profiles):
                progress(f"export {pk}", i / len(profiles))
                ov = {"strip_location": strip} if pk == "share" else None
                outs.append(self.studio.export(src, v.result.rgb, v.result.alpha, v.recipe, pk,
                                               review_flags=v.result.review_flags, elapsed_ms=v.seconds * 1000,
                                               overrides=ov, label=v.label))
            if auto is not None:
                choice = "accepted_auto" if v.recipe.hash == auto.hash else "manual_export"
                self.studio.record_correction(src, auto, v.recipe, choice)
            return outs

        def done(outs):
            self._busy(False)
            lines = []
            for o in outs:
                a = o.audit
                ok = a.get("verified")
                lines.append(f"<b>{KIND_LABELS.get(o.kind, o.kind)}</b> {o.width}×{o.height} {o.bits}-bit → "
                             f"{html.escape(o.path.name)}<br><span style='color:{'#81c784' if ok else '#ffb74d'}'>"
                             f"metadata {'verified' if ok else 'NOT fully verified'}: {len(a.get('transferred', []))} "
                             f"kept, {len(a.get('updated', {}))} updated, {len(a.get('dropped', {}))} not transferable, "
                             f"{len(a.get('failed', {}))} failed, {len(a.get('stripped_location', []))} location tags "
                             f"removed · ICC {html.escape(str(a.get('output_icc')))}</span>")
                for n in a.get("notes", []) + a.get("profile_notes", []):
                    lines.append(f"<span style='color:#8a8f98'>· {html.escape(n)}</span>")
            self.export_info.setText("<br>".join(lines))
            self._last_export = outs[0].path if outs else None
            self.open_out.setEnabled(bool(outs))
            self.exported.emit()
            self.status.emit(f"Exported {len(outs)} file(s); source unchanged.")
        self._busy(True, "Exporting…")
        self.worker = run_bg(job, on_done=done, on_error=self._error, on_progress=self._on_progress)

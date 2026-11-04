"""Manual / Advanced recipe editor.

Shows every operation Auto chose with its *actual* parameters. Each row states whether the strength
slider drives a native model/algorithm parameter or a labelled post-blend. Edits are undoable.
"""
from __future__ import annotations

import json

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QFrame, QHBoxLayout, QLabel,
                               QPushButton, QSlider, QVBoxLayout, QWidget)

from ..backends.registry import MODELS
from ..core.recipes import CLASSICAL_PARAMS, METHODS, OP_KINDS, Operation, Recipe, default_op, map_strength

EXTRA_METHODS: dict[str, list[str]] = {}   # e.g. {"upscale": ["comfy:upscale:<model file>"]} when ComfyUI is up

MASKS = [("none", None), ("protect faces", "protect_faces"), ("faces only", "faces_only"),
         ("protect text/UI", "protect_text")]


def method_label(m: str) -> str:
    if m.startswith("comfy:"):
        return f"ComfyUI: {m.split(':', 2)[2]} · external (character unverified)"
    if m.startswith("model:"):
        info = MODELS.get(m.split(":", 1)[1])
        if not info:
            return m
        tag = {"adopt": "", "prototype": " [prototype]", "baseline": " [baseline]", "reject": " [rejected]"}[info.status]
        ch = {"faithful": "faithful", "balanced": "GAN", "generative": "generative"}[info.character]
        miss = "" if info.installed else " (not installed)"
        return f"{info.name} · {ch}{tag}{miss}"
    return m.split(":", 1)[1].replace("_", " ") + " · classical"


class OpRow(QFrame):
    changed = Signal()
    removed = Signal(object)

    def __init__(self, op: Operation, parent=None):
        super().__init__(parent)
        self.op = op
        self.setFrameShape(QFrame.Shape.StyledPanel)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 4)
        top = QHBoxLayout()
        self.enabled = QCheckBox(op.kind.replace("_", " ").title())
        self.enabled.setChecked(op.enabled)
        self.enabled.setStyleSheet("font-weight:600")
        self.enabled.toggled.connect(self._on_enable)
        top.addWidget(self.enabled)
        top.addStretch(1)
        rm = QPushButton("✕")
        rm.setFixedWidth(26)
        rm.setToolTip("Remove operation")
        rm.clicked.connect(lambda: self.removed.emit(self))
        top.addWidget(rm)
        lay.addLayout(top)
        self.method = QComboBox()
        for m in METHODS.get(op.kind, [op.method]) + EXTRA_METHODS.get(op.kind, []):
            if m.startswith("model:") and MODELS.get(m.split(":", 1)[1]) and \
                    MODELS[m.split(":", 1)[1]].status == "reject":
                continue
            self.method.addItem(method_label(m), m)
        i = self.method.findData(op.method)
        if i < 0:
            self.method.addItem(method_label(op.method), op.method)
            i = self.method.count() - 1
        self.method.setCurrentIndex(i)
        self.method.currentIndexChanged.connect(self._on_method)
        lay.addWidget(self.method)
        srow = QHBoxLayout()
        srow.addWidget(QLabel("Strength"))
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(int(op.strength if op.strength is not None else round(op.blend * 100)))
        self.slider.sliderReleased.connect(self._on_strength)
        self.slider.valueChanged.connect(lambda v: self.sval.setText(str(v)))
        srow.addWidget(self.slider, 1)
        self.sval = QLabel(str(self.slider.value()))
        self.sval.setFixedWidth(28)
        srow.addWidget(self.sval)
        lay.addLayout(srow)
        self.mapping = QLabel()
        self.mapping.setWordWrap(True)
        self.mapping.setStyleSheet("color:#8a8f98; font-size:11px")
        lay.addWidget(self.mapping)
        self.form = QFormLayout()
        self.form.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(self.form)
        mrow = QHBoxLayout()
        mrow.addWidget(QLabel("Region"))
        self.mask = QComboBox()
        for lab, val in MASKS:
            self.mask.addItem(lab, val)
        mi = self.mask.findData(op.mask)
        self.mask.setCurrentIndex(max(0, mi))
        self.mask.currentIndexChanged.connect(self._on_mask)
        mrow.addWidget(self.mask, 1)
        lay.addLayout(mrow)
        self.why = QLabel(op.why)
        self.why.setWordWrap(True)
        self.why.setStyleSheet("color:#6f8fb0; font-size:11px")
        lay.addWidget(self.why)
        self._build_params()

    def _build_params(self):
        while self.form.rowCount():
            self.form.removeRow(0)
        op = self.op
        if op.model_id:
            info = MODELS.get(op.model_id)
            specs = [(p.name, p.label, p.minimum, p.maximum, p.step, p.help) for p in (info.native_params if info else ())]
            if op.params.get("face_protect") is not None:
                specs.append(("face_protect", "Face protection (blend toward input)", 0, 1, 0.05,
                              "Post-blend inside detected face regions; not a model parameter."))
            if op.kind == "upscale" and info:
                self.mapping.setText(f"Model scale ×{info.scale}. "
                                     + ("" if info.native_params else "No native strength input: slider = post-blend."))
            else:
                self.mapping.setText(map_strength(op.method, self.slider.value())[2])
        else:
            specs = []
            for k, v in op.params.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    rng = {"ev": (-3, 3, 0.05), "gamma": (0.3, 3, 0.05), "black": (0, 0.5, 0.005),
                           "white": (0.5, 1, 0.005), "scale": (0.25, 8, 0.25), "angle": (-45, 45, 0.1),
                           "rotate90": (0, 3, 1), "contrast": (-0.5, 0.5, 0.01), "saturation": (-1, 1, 0.01),
                           "vibrance": (-1, 1, 0.01)}.get(k, (0, 50 if k in ("h", "h_color") else 2, 0.01))
                    specs.append((k, k, rng[0], rng[1], rng[2], "Native parameter of the classical operation."))
            if op.params.get("gains") is not None:
                g = op.params["gains"]
                self.form.addRow(QLabel("WB gains"), QLabel(f"R×{g[0]:.3f}  G×{g[1]:.3f}  B×{g[2]:.3f} "
                                                            "(set from consensus illuminant)"))
            self.mapping.setText(map_strength(op.method, self.slider.value())[2])
        for name, label, lo, hi, step, helptext in specs:
            sb = QDoubleSpinBox()
            sb.setRange(lo, hi)
            sb.setSingleStep(step)
            sb.setDecimals(3 if step < 0.1 else 1 if step < 1 else 0)
            sb.setValue(float(op.params.get(name, lo) or 0))
            sb.setToolTip(helptext)
            sb.valueChanged.connect(lambda v, n=name: self._on_param(n, v))
            self.form.addRow(label, sb)
        if op.blend < 1.0 or (op.model_id and not (MODELS.get(op.model_id) and MODELS[op.model_id].native_params)):
            sb = QDoubleSpinBox()
            sb.setRange(0, 1)
            sb.setSingleStep(0.05)
            sb.setValue(op.blend)
            sb.setToolTip("Post-blend of this operation's output with its input (explicitly not a model parameter).")
            sb.valueChanged.connect(self._on_blend)
            self.form.addRow("Post-blend", sb)

    def _on_enable(self, on):
        self.op.enabled = on
        self.changed.emit()

    def _on_method(self):
        m = self.method.currentData()
        new = default_op(self.op.kind, m)
        self.op.method, self.op.params, self.op.blend = new.method, new.params, new.blend
        self.op.why = "manual"
        self._build_params()
        self.changed.emit()

    def _on_strength(self):
        v = self.slider.value()
        params, blend, why = map_strength(self.op.method, v)
        if self.op.model_id or blend < 1.0:
            self.op.params.update(params)
            self.op.blend = blend
        else:
            self.op.params.update(params)
        self.op.strength = v
        self.op.why = "manual"
        self._build_params()
        self.changed.emit()

    def _on_param(self, name, v):
        self.op.params[name] = v
        self.op.why = "manual"
        self.changed.emit()

    def _on_blend(self, v):
        self.op.blend = float(v)
        self.changed.emit()

    def _on_mask(self):
        self.op.mask = self.mask.currentData()
        self.changed.emit()


class RecipeEditor(QWidget):
    recipeChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.recipe = Recipe()
        self.auto_recipe: Recipe | None = None
        self._state = ""
        self._undo: list[str] = []
        self._redo: list[str] = []
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        row.addWidget(QLabel("Mode"))
        self.mode = QComboBox()
        self.mode.addItem("Faithful (default)", "faithful")
        self.mode.addItem("Balanced (GAN texture)", "balanced")
        self.mode.currentIndexChanged.connect(self._on_mode)
        row.addWidget(self.mode, 1)
        row.addWidget(QLabel("Scale ×"))
        self.scale = QDoubleSpinBox()
        self.scale.setRange(1, 8)
        self.scale.setSingleStep(0.5)
        self.scale.valueChanged.connect(self._on_scale)
        row.addWidget(self.scale)
        lay.addLayout(row)
        self.rows_box = QVBoxLayout()
        lay.addLayout(self.rows_box)
        add = QHBoxLayout()
        self.add_kind = QComboBox()
        for k in OP_KINDS:
            if k != "geometry":
                self.add_kind.addItem(k.replace("_", " "), k)
        add.addWidget(self.add_kind, 1)
        b = QPushButton("Add operation")
        b.clicked.connect(self._add)
        add.addWidget(b)
        lay.addLayout(add)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet("color:#8a8f98; font-size:11px")
        lay.addWidget(self.summary)

    def set_recipe(self, r: Recipe, auto: Recipe | None = None, push: bool = False):
        """Load *r*. With push=True the previous recipe is kept on the undo stack."""
        prev = getattr(self, "_state", None)
        self.recipe = r.clone()
        if auto is not None:
            self.auto_recipe = auto.clone()
        if push and prev:
            self._undo.append(prev)
            self._redo.clear()
        self._state = self._dump()
        self._rebuild()

    def _dump(self) -> str:
        return json.dumps(self.recipe.canonical(), sort_keys=True) + "\n" + json.dumps(self.recipe.to_json())

    def _load(self, state: str):
        self.recipe = Recipe.from_json(json.loads(state.split("\n", 1)[1]))
        self._state = state
        self._rebuild()
        self.recipeChanged.emit()

    def _commit(self):
        """Call after any mutation: records the previous state for undo."""
        self.recipe.created_by = "manual"
        self.recipe.no_op = not any(o.enabled for o in self.recipe.ops) and self.recipe.target_scale == 1
        new = self._dump()
        if new.split("\n", 1)[0] != getattr(self, "_state", "").split("\n", 1)[0]:
            self._undo.append(self._state)
            self._undo = self._undo[-50:]
            self._redo.clear()
        self._state = new
        self._update_summary()
        self.recipeChanged.emit()

    def _rebuild(self):
        while self.rows_box.count():
            w = self.rows_box.takeAt(0).widget()
            if w:
                w.deleteLater()
        for op in self.recipe.ops:
            row = OpRow(op)
            row.changed.connect(self._commit)
            row.removed.connect(self._remove)
            self.rows_box.addWidget(row)
        self.mode.blockSignals(True)
        self.mode.setCurrentIndex(max(0, self.mode.findData(self.recipe.mode)))
        self.mode.blockSignals(False)
        self.scale.blockSignals(True)
        self.scale.setValue(self.recipe.target_scale)
        self.scale.blockSignals(False)
        self._update_summary()

    def _update_summary(self):
        diff = ""
        if self.auto_recipe is not None:
            diff = " (= Auto)" if self.recipe.hash == self.auto_recipe.hash else " (modified from Auto)"
        self.summary.setText(f"Recipe {self.recipe.hash}{diff}: {self.recipe.summary()}")

    def _remove(self, row: OpRow):
        self.recipe.ops = [o for o in self.recipe.ops if o is not row.op]
        self._rebuild()
        self._commit()

    def _add(self):
        op = default_op(self.add_kind.currentData())
        op.why = "manual"
        self.recipe.ops.append(op)
        self.recipe.ops = self.recipe.sorted_ops()
        self._rebuild()
        self._commit()

    def _on_mode(self):
        self.recipe.mode = self.mode.currentData()
        self._commit()

    def _on_scale(self, v):
        self.recipe.target_scale = float(v)
        if v > 1 and not any(o.kind == "upscale" for o in self.recipe.ops):
            op = default_op("upscale", "classical:lanczos")
            op.params["scale"] = float(v)
            op.why = "manual"
            self.recipe.ops.append(op)
            self.recipe.ops = self.recipe.sorted_ops()
            self._rebuild()
        self._commit()

    def undo(self) -> bool:
        if not self._undo:
            return False
        self._redo.append(self._state)
        self._load(self._undo.pop())
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        self._undo.append(self._state)
        self._load(self._redo.pop())
        return True

    def reset_to_auto(self):
        if self.auto_recipe is not None:
            self.recipe = self.auto_recipe.clone()
            self._rebuild()
            self._commit()
            self.recipe.created_by = "auto"

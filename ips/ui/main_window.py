"""Main window: Prepare / Quick Edit / Batch / Outputs, header status, Settings."""
from __future__ import annotations

import logging

from PySide6.QtCore import QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout,
                               QLabel, QLineEdit, QMainWindow, QPushButton, QSizePolicy, QSpinBox, QTabWidget,
                               QToolBar, QWidget)

from .. import APP_NAME, __version__
from ..backends.registry import MODELS
from ..core.engine import Studio
from .batch_panel import BatchPanel
from .common import run_bg
from .gallery_panel import GalleryPanel
from .prepare_panel import PreparePanel
from .quickedit_panel import QuickEditPanel

log = logging.getLogger("ips.ui")


class SettingsDialog(QDialog):
    def __init__(self, studio: Studio, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.s = studio.settings
        f = QFormLayout(self)
        self.out = QLineEdit(self.s.output_root)
        b = QPushButton("…")
        b.clicked.connect(self._pick)
        row = QHBoxLayout()
        row.addWidget(self.out)
        row.addWidget(b)
        w = QWidget()
        w.setLayout(row)
        f.addRow("Output root", w)
        self.device = QComboBox()
        for d in ("auto", "cuda", "cpu"):
            self.device.addItem(d)
        self.device.setCurrentText(self.s.device)
        f.addRow("Inference device (restart)", self.device)
        self.tile = QSpinBox()
        self.tile.setRange(0, 2048)
        self.tile.setSingleStep(64)
        self.tile.setValue(self.s.tile)
        self.tile.setSpecialValueText("per-model default")
        f.addRow("Tile size (restart)", self.tile)
        self.mode = QComboBox()
        self.mode.addItems(["faithful", "balanced"])
        self.mode.setCurrentText(self.s.default_mode)
        f.addRow("Default mode", self.mode)
        self.look = QComboBox()
        self.look.addItems(["enhanced", "vivid", "natural"])
        self.look.setCurrentText(getattr(self.s, "default_look", "enhanced"))
        f.addRow("Default look", self.look)
        self.lowres = QCheckBox("Auto ×2 for sources with long side < 1024 px")
        self.lowres.setChecked(self.s.upscale_low_res_auto)
        f.addRow("", self.lowres)
        self.autorun = QCheckBox("Run Smart Auto immediately when an image is opened")
        self.autorun.setChecked(self.s.auto_run_on_open)
        f.addRow("", self.autorun)
        self.ident = QCheckBox("Identity drift check on faces")
        self.ident.setChecked(self.s.identity_check)
        f.addRow("", self.ident)
        self.guard = QCheckBox("Faithful fidelity guard (blend flagged faces toward conservative rendering)")
        self.guard.setChecked(self.s.fidelity_guard)
        f.addRow("", self.guard)
        self.afmt = QComboBox()
        self.afmt.addItems(["TIFF", "PNG"])
        self.afmt.setCurrentText(self.s.archive_format)
        f.addRow("Archive format", self.afmt)
        self.abits = QComboBox()
        self.abits.addItems(["source", "16"])
        self.abits.setCurrentText(self.s.archive_bits)
        f.addRow("Archive bit depth", self.abits)
        self.ai_edge = QSpinBox()
        self.ai_edge.setRange(256, 16384)
        self.ai_edge.setValue(self.s.ai_ready_long_edge)
        f.addRow("AI/Video Ready long edge", self.ai_edge)
        self.ai_mult = QSpinBox()
        self.ai_mult.setRange(1, 128)
        self.ai_mult.setValue(self.s.ai_ready_multiple_of)
        f.addRow("AI/Video Ready pad to multiple of (1 = off)", self.ai_mult)
        self.share_edge = QSpinBox()
        self.share_edge.setRange(0, 16384)
        self.share_edge.setValue(self.s.share_long_edge)
        self.share_edge.setSpecialValueText("keep size")
        f.addRow("Share long edge", self.share_edge)
        self.share_q = QSpinBox()
        self.share_q.setRange(60, 100)
        self.share_q.setValue(self.s.share_quality)
        f.addRow("Share JPEG quality", self.share_q)
        self.strip = QCheckBox("Remove location metadata from Share exports by default")
        self.strip.setChecked(self.s.share_strip_location)
        f.addRow("", self.strip)
        self.comfy = QCheckBox("Use local ComfyUI (optional routes only)")
        self.comfy.setChecked(self.s.comfy_enabled)
        f.addRow("", self.comfy)
        self.comfy_url = QLineEdit(self.s.comfy_url)
        f.addRow("ComfyUI URL (localhost only)", self.comfy_url)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self._save)
        bb.rejected.connect(self.reject)
        f.addRow(bb)

    def _pick(self):
        d = QFileDialog.getExistingDirectory(self, "Output root", self.out.text())
        if d:
            self.out.setText(d)

    def _save(self):
        s = self.s
        s.output_root = self.out.text()
        s.device = self.device.currentText()
        s.tile = self.tile.value()
        s.default_mode = self.mode.currentText()
        s.default_look = self.look.currentText()
        s.upscale_low_res_auto = self.lowres.isChecked()
        s.auto_run_on_open = self.autorun.isChecked()
        s.identity_check = self.ident.isChecked()
        s.fidelity_guard = self.guard.isChecked()
        s.archive_format = self.afmt.currentText()
        s.archive_bits = self.abits.currentText()
        s.ai_ready_long_edge = self.ai_edge.value()
        s.ai_ready_multiple_of = self.ai_mult.value()
        s.share_long_edge = self.share_edge.value()
        s.share_quality = self.share_q.value()
        s.share_strip_location = self.strip.isChecked()
        s.comfy_enabled = self.comfy.isChecked()
        s.comfy_url = self.comfy_url.text().strip()
        s.save()
        self.accept()


class MainWindow(QMainWindow):
    def __init__(self, studio: Studio):
        super().__init__()
        self.studio = studio
        self.setWindowTitle(f"{APP_NAME} {__version__}")
        self.resize(1500, 950)
        tb = QToolBar("Status")
        tb.setMovable(False)
        self.addToolBar(tb)
        self.backend_lbl = QLabel("Detecting backend…")
        self.backend_lbl.setStyleSheet("padding: 0 10px;")
        tb.addWidget(self.backend_lbl)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        tb.addWidget(spacer)
        act = QAction("Settings", self)
        act.setShortcut(QKeySequence("Ctrl+,"))
        act.triggered.connect(self._settings)
        tb.addAction(act)
        self.tabs = QTabWidget()
        self.prepare = PreparePanel(studio)
        self.quick = QuickEditPanel(studio)
        self.batch = BatchPanel(studio)
        self.gallery = GalleryPanel(studio)
        self.tabs.addTab(self.prepare, "Prepare")
        self.tabs.addTab(self.quick, "Quick Edit")
        self.tabs.addTab(self.batch, "Batch")
        self.tabs.addTab(self.gallery, "Outputs")
        self.setCentralWidget(self.tabs)
        for p in (self.prepare, self.quick, self.batch, self.gallery):
            p.status.connect(self.statusBar().showMessage)
        self.prepare.exported.connect(self.gallery.reload_filters)
        self.quick.exported.connect(self.gallery.reload_filters)
        self.batch.openInPrepare.connect(self._open_from_batch)
        self.gallery.openInPrepare.connect(lambda p: (self.tabs.setCurrentWidget(self.prepare),
                                                      self.prepare.open_path(p)))
        self.tabs.currentChanged.connect(lambda i: self.tabs.widget(i) is self.gallery and self.gallery.refresh())
        edit_act = QAction("Send current image to Quick Edit", self)
        edit_act.setShortcut(QKeySequence("Ctrl+Shift+E"))
        edit_act.triggered.connect(self._send_to_quick)
        self.addAction(edit_act)
        self.statusBar().showMessage("Ready. Drop an image on Prepare, or add folders in Batch.")
        QTimer.singleShot(50, self._probe_backend)
        self._comfy_timer = QTimer(self)
        self._comfy_timer.timeout.connect(self._probe_comfy)
        self._comfy_timer.start(15000)
        self._idle_timer = QTimer(self)
        self._idle_timer.timeout.connect(self._unload_idle)
        self._idle_timer.start(30000)

    def _unload_idle(self):
        mins = self.studio.settings.unload_idle_minutes
        if not mins or not self.studio._models_ok:
            return
        from ..backends import direct
        if direct._BACKEND is not None and direct._BACKEND.unload_if_idle(mins * 60):
            self.statusBar().showMessage(f"Models released from memory after {mins} idle minutes.")

    def _open_from_batch(self, path, jobref, rel):
        self.tabs.setCurrentWidget(self.prepare)
        self.prepare.open_path(path, batch_job=jobref, rel_dir=rel)

    def _send_to_quick(self):
        if self.prepare.src is not None:
            ri = self.prepare.right_sel.currentData()
            rgb = self.prepare.variants[ri].result.rgb if ri is not None and ri >= 0 else None
            if rgb is not None and rgb.shape[:2] != self.prepare.src.image.rgb.shape[:2]:
                rgb = None  # Quick Edit works on source geometry
            self.quick.load_array(self.prepare.src, rgb)
            self.tabs.setCurrentWidget(self.quick)

    def _probe_backend(self):
        def probe(progress, cancel):
            ok = self.studio.models_ok
            d = self.studio.device
            return ok, d
        run_bg(probe, on_done=self._show_backend, on_error=lambda m: self.backend_lbl.setText("Backend error: " + m[:120]))

    def _show_backend(self, res):
        ok, d = res
        inst = sum(1 for m in MODELS.values() if m.installed and m.status != "reject")
        tot = sum(1 for m in MODELS.values() if m.status != "reject")
        vram = f", {d.total_vram_mb // 1024} GB VRAM" if d.total_vram_mb else ""
        cert = "Fast Profile certified" if self.studio.profile.certified(d.name) else "Fast Profile not measured"
        self._base_status = (f"Device: {d.name} ({d.kind}{vram}) · PyTorch {'ok' if ok else 'unavailable'} · "
                             f"Models {inst}/{tot} installed · {cert}")
        self.backend_lbl.setText(self._base_status + " · ComfyUI: …")
        self._probe_comfy()

    def _probe_comfy(self):
        s = self.studio.settings
        if not s.comfy_enabled:
            self.backend_lbl.setText(getattr(self, "_base_status", "") + " · ComfyUI: disabled")
            for panel in (self.prepare, getattr(self, "batch", None)):
                if panel is not None and hasattr(panel, "set_generative_available"):
                    panel.set_generative_available(False, "Enable ComfyUI in Settings to use Qwen-Image 2.1.")
            return

        def probe(progress, cancel):
            from ..backends.comfy.client import ComfyClient
            c = ComfyClient(s.comfy_url, timeout=2)
            st = c.status()
            st.models = c.upscale_models() if st.available else []  # type: ignore[attr-defined]
            st.qwen = (None, "ComfyUI not reachable")  # type: ignore[attr-defined]
            if st.available:
                from ..backends.comfy import qwen
                m, why = qwen.detect(c, turbo=False)
                mt, _ = qwen.detect(c, turbo=True)
                found = [x.unet for x in (m, mt) if x]
                st.qwen = (bool(found), ", ".join(found) if found else why)  # type: ignore[attr-defined]
            return st

        def done(st):
            from . import recipe_editor
            recipe_editor.EXTRA_METHODS["upscale"] = [f"comfy:upscale:{m}" for m in getattr(st, "models", [])]
            ok, why = getattr(st, "qwen", (None, ""))
            for panel in (self.prepare, self.batch):
                if hasattr(panel, "set_generative_available"):
                    panel.set_generative_available(bool(ok), why)
            txt = (f"ComfyUI {st.version or ''} connected (queue {st.queue_running}+{st.queue_pending}, "
                   f"free {st.vram_free_mb} MB)" if st.available else "ComfyUI: not reachable")
            self.backend_lbl.setText(getattr(self, "_base_status", "") + " · " + txt)
        run_bg(probe, on_done=done, on_error=lambda m: None)

    def _settings(self):
        if SettingsDialog(self.studio, self).exec():
            self.statusBar().showMessage("Settings saved (device/tile changes apply after restart).")
            self._probe_comfy()

    def closeEvent(self, e):
        self.batch.shutdown()
        super().closeEvent(e)


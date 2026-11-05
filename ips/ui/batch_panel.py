"""Batch tab: queue, per-file state/time/issues/recipe/exception, pause/resume/cancel/retry, review filter."""
from __future__ import annotations

import json
import time
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QHeaderView,
                               QLabel, QLineEdit, QListWidget, QMessageBox, QPushButton, QSplitter, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from ..core.engine import Studio
from ..core.jobs import JobManager, enumerate_inputs
from ..core.recipes import Recipe
from .common import STATE_COLORS, open_folder, run_bg, to_qimage, wrap_label
from .widgets.compare_view import CompareView

FILTERS = [("All", None), ("Needs review", ("review", "needs_override")), ("Failed", ("failed", "cancelled")),
           ("Done", ("done",)), ("Queued/running", ("queued", "running")), ("Duplicates", ("duplicate",))]


class BatchPanel(QWidget):
    status = Signal(str)
    openInPrepare = Signal(str, object, str)   # path, (jm, job_id), rel_dir

    def __init__(self, studio: Studio, parent=None):
        super().__init__(parent)
        self.studio = studio
        self.jm = JobManager(studio)
        self.batch_id: int | None = None
        self.pending: list[str] = []
        self._thumbs: dict[str, QIcon] = {}
        self._build()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(1000)
        self._load_batches()

    def _build(self):
        root = QVBoxLayout(self)
        row = QHBoxLayout()
        b = QPushButton("Add files…")
        b.clicked.connect(self._add_files)
        row.addWidget(b)
        b = QPushButton("Add folder…")
        b.clicked.connect(self._add_folder)
        row.addWidget(b)
        self.recursive = QCheckBox("Include subfolders")
        self.recursive.setChecked(self.studio.settings.recursive_import)
        row.addWidget(self.recursive)
        row.addWidget(QLabel("Batch name"))
        self.name = QLineEdit(time.strftime("%Y-%m-%d_%H%M"))
        row.addWidget(self.name)
        row.addWidget(QLabel("Mode"))
        self.mode = QComboBox()
        self.mode.addItem("Faithful", "faithful")
        self.mode.addItem("Balanced (GAN)", "balanced")
        self.mode.addItem("Generative · Qwen-Image 2.1 (ComfyUI; prompt/strength from Prepare)", "generative")
        self.set_generative_available(False, "Enable ComfyUI in Settings; Qwen-Image 2.1 models are detected there.")
        row.addWidget(self.mode)
        self.scale = QComboBox()
        for lab, v in (("Scale: Auto per image", "auto"), ("×1", 1), ("×2", 2), ("×4", 4)):
            self.scale.addItem(lab, v)
        row.addWidget(self.scale)
        self.look = QComboBox()
        for lab, v in [("Look: Enhanced (visible, non-generative)", "enhanced"), ("Look: Vivid (strong)", "vivid"),
                         ("Look: Natural (repair defects only)", "natural")]:
            self.look.addItem(lab, v)
        self.look.setCurrentIndex(max(0, self.look.findData(getattr(self.studio.settings, "default_look", "enhanced"))))
        row.addWidget(self.look)
        row.addStretch(1)
        root.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("Outputs:"))
        self.profiles = {}
        for k, lab in (("archive", "Archive Master"), ("ai_ready", "AI/Video Ready"), ("share", "Share/Web")):
            c = QCheckBox(lab)
            c.setChecked(k == "archive")
            self.profiles[k] = c
            row.addWidget(c)
        self.override = QCheckBox("Quality Override")
        row.addWidget(self.override)
        row.addStretch(1)
        self.pending_lbl = QLabel("0 files queued for a new batch")
        row.addWidget(self.pending_lbl)
        b = QPushButton("Clear list")
        b.clicked.connect(self._clear_pending)
        row.addWidget(b)
        root.addLayout(row)
        row = QHBoxLayout()
        self.btn_start = QPushButton("Start new batch")
        self.btn_start.setStyleSheet("font-weight:600")
        self.btn_start.clicked.connect(self.start_new)
        row.addWidget(self.btn_start)
        row.addWidget(QLabel("or existing:"))
        self.batches = QComboBox()
        self.batches.currentIndexChanged.connect(self._select_batch)
        row.addWidget(self.batches, 1)
        for lab, fn in (("Resume", self.resume_batch), ("Pause", self.jm.pause), ("Continue", self.jm.resume),
                        ("Cancel", self.jm.cancel), ("Retry failed", self.retry_failed),
                        ("Retry selected", self.retry_selected), ("Report", self.report)):
            b = QPushButton(lab)
            b.clicked.connect(fn)
            row.addWidget(b)
        root.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("Show"))
        self.filter = QComboBox()
        for lab, v in FILTERS:
            self.filter.addItem(lab, v)
        self.filter.currentIndexChanged.connect(self.refresh)
        row.addWidget(self.filter)
        self.stats = QLabel("")
        row.addWidget(self.stats, 1)
        root.addLayout(row)
        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels(["", "File", "State", "Time (s)", "Content", "Diagnosed issues",
                                              "Recipe", "Review / exception", "Job"])
        self.table.setIconSize(QPixmap(56, 56).size())
        self.table.verticalHeader().setDefaultSectionSize(60)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        h = self.table.horizontalHeader()
        h.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        h.setStretchLastSection(False)
        for i, w in enumerate((64, 220, 90, 70, 90, 260, 320, 360, 50)):
            self.table.setColumnWidth(i, w)
        self.table.cellDoubleClicked.connect(self._open_row)
        self.table.setToolTip("Double-click a row to open it in Prepare (per-image override).")
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self.table)
        pv = QWidget()
        pl = QVBoxLayout(pv)
        pl.setContentsMargins(4, 0, 0, 0)
        prow = QHBoxLayout()
        self.follow = QCheckBox("Follow latest result")
        self.follow.setChecked(True)
        prow.addWidget(self.follow)
        for lab, m in (("Wipe", "wipe"), ("Before", "before"), ("After", "after"), ("Side", "side")):
            b = QPushButton(lab)
            b.clicked.connect(lambda _=False, mm=m: self.preview.set_mode(mm))
            prow.addWidget(b)
        prow.addStretch(1)
        pl.addLayout(prow)
        self.preview = CompareView()
        self.preview.empty_text = "Select a row (or start a batch) to preview Before / After"
        pl.addWidget(self.preview, 1)
        self.preview_info = wrap_label("")
        self.preview_info.setMaximumHeight(90)
        pl.addWidget(self.preview_info)
        split.addWidget(pv)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        self.table.itemSelectionChanged.connect(self._preview_selected)
        root.addWidget(split, 1)
        self._preview_job = None
        self._preview_seen = set()
        self.similar = QListWidget()
        self.similar.setMaximumHeight(70)
        root.addWidget(QLabel("Visually similar pairs (suggestions only — nothing is deleted):"))
        root.addWidget(self.similar)

    # ------------------------------------------------------------------ inputs
    def add_paths(self, paths: list[str]):
        self.pending.extend(paths)
        n = len(enumerate_inputs(self.pending, self.recursive.isChecked()))
        self.pending_lbl.setText(f"{n} files queued for a new batch")

    def _add_files(self):
        fs, _ = QFileDialog.getOpenFileNames(self, "Add images")
        if fs:
            self.add_paths(fs)

    def _add_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Add folder")
        if d:
            self.add_paths([d])

    def _clear_pending(self):
        self.pending.clear()
        self.pending_lbl.setText("0 files queued for a new batch")

    # ------------------------------------------------------------------ control
    def start_new(self):
        if not self.pending:
            QMessageBox.information(self, "Batch", "Add files or a folder first.")
            return
        if self.jm.running:
            QMessageBox.information(self, "Batch", "A batch is already running.")
            return
        profiles = tuple(k for k, c in self.profiles.items() if c.isChecked()) or ("archive",)
        self.batch_id = self.jm.create_batch(self.name.text().strip() or "batch", self.pending,
                                             recursive=self.recursive.isChecked(), profiles=profiles,
                                             mode=self.mode.currentData(), scale=self.scale.currentData(),
                                             look=self.look.currentData(),
                                             quality_override=self.override.isChecked())
        self._clear_pending()
        self._load_batches(select=self.batch_id)
        self.jm.start(self.batch_id)
        self.status.emit(f"Batch {self.batch_id} started.")

    def resume_batch(self):
        if self.batch_id is None or self.jm.running:
            return
        bad = self.jm.verify_completed(self.batch_id)
        self.jm.start(self.batch_id)
        self.status.emit(f"Batch {self.batch_id} resumed ({len(bad)} job(s) requeued after output verification).")

    def retry_failed(self):
        if self.batch_id is not None:
            n = self.jm.retry_failed(self.batch_id)
            self.status.emit(f"{n} job(s) requeued; press Resume.")

    def retry_selected(self):
        ids = sorted({int(self.table.item(i.row(), 8).text()) for i in self.table.selectedIndexes()})
        if ids:
            self.jm.retry(ids)
            self.status.emit(f"{len(ids)} job(s) requeued; press Resume.")

    def report(self):
        if self.batch_id is None:
            return
        jp, cp = self.jm.write_report(self.batch_id)
        open_folder(jp)
        self.status.emit(f"Report written: {jp}")

    def _load_batches(self, select: int | None = None):
        self.batches.blockSignals(True)
        self.batches.clear()
        for b in self.studio.db.query("SELECT b.id, b.name, b.state, COUNT(j.id) n, "
                                      "SUM(j.state='queued') q FROM batches b LEFT JOIN jobs j ON j.batch_id=b.id "
                                      "GROUP BY b.id ORDER BY b.id DESC LIMIT 50"):
            self.batches.addItem(f"#{b['id']} {b['name']} · {b['n']} files · {b['state']}"
                                 f"{' · ' + str(b['q']) + ' queued' if b['q'] else ''}", b["id"])
        if select is not None:
            self.batches.setCurrentIndex(max(0, self.batches.findData(select)))
        self.batches.blockSignals(False)
        self.batch_id = self.batches.currentData()

    def _select_batch(self):
        self.batch_id = self.batches.currentData()
        self._thumbs.clear()
        self.refresh()

    # ------------------------------------------------------------------ table
    def _thumb(self, path: str) -> QIcon:
        if path not in self._thumbs:
            pm = QPixmap(path)
            self._thumbs[path] = QIcon(pm.scaled(56, 56, Qt.AspectRatioMode.KeepAspectRatio,
                                                 Qt.TransformationMode.SmoothTransformation)) if not pm.isNull() else QIcon()
        return self._thumbs[path]

    def refresh(self):
        if self.batch_id is None or not self.isVisible():
            return
        if getattr(self, "_was_running", False) and not self.jm.running:
            self._load_batches(select=self.batch_id)
        self._was_running = self.jm.running
        filt = self.filter.currentData()
        jobs = self.jm.jobs(self.batch_id)
        shown = [j for j in jobs if filt is None or j["state"] in filt]
        self.table.setRowCount(len(shown))
        for r, j in enumerate(shown):
            rv = json.loads(j["review_json"]) if j["review_json"] else {}
            t = json.loads(j["timings_json"]) if j["timings_json"] else {}
            rc = Recipe.from_json(json.loads(j["recipe_json"])).summary() if j["recipe_json"] else ""
            secs = f"{t['total_job_ms'] / 1000:.1f}" if isinstance(t.get("total_job_ms"), (int, float)) else ""
            msg = j["error"] or " | ".join(rv.get("flags", []))
            vals = ["", Path(j["source_path"]).name, j["state"], secs, rv.get("content_class", ""),
                    ", ".join(rv.get("issues", [])), rc, msg, str(j["id"])]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if c == 2:
                    it.setForeground(QColor(STATE_COLORS.get(v, "#999")))
                if c in (5, 6, 7):
                    it.setToolTip(v)
                if c == 1:
                    it.setToolTip(j["source_path"])
                self.table.setItem(r, c, it)
            ic = QTableWidgetItem()
            ic.setIcon(self._thumb(j["source_path"]))
            self.table.setItem(r, 0, ic)
        self._follow_latest(jobs)
        st = self.jm.stats(self.batch_id)
        s = st["states"]
        tt = st["total_s"]
        self.stats.setText(" · ".join(f"{k}: {v}" for k, v in sorted(s.items()))
                           + (f" · median {tt['median']} s, p95 {tt['p95']} s, max {tt['max']} s" if tt["median"] else "")
                           + (f" · {'running' if self.jm.running else 'idle'}{' (paused)' if self.jm.paused else ''}"))
        self.similar.clear()
        for a, b, d in st["similar_pairs"][:50]:
            self.similar.addItem(f"job {a} ≈ job {b} (dHash distance {d})")

    # ------------------------------------------------------------------ live preview
    def _preview_selected(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        if rows:
            it = self.table.item(rows[0], 8)
            if it:
                self.show_preview(int(it.text()))

    def show_preview(self, job_id: int):
        """Load source + restored master for *job_id* off the UI thread and show them in the compare view."""
        if job_id == self._preview_job:
            return
        self._preview_job = job_id
        j = self.studio.db.one("SELECT * FROM jobs WHERE id=?", (job_id,))
        if not j:
            return
        out = self.studio.db.one("SELECT path, kind FROM outputs WHERE job_id=? ORDER BY "
                                 "(kind='restored_master') DESC, id DESC LIMIT 1", (job_id,))

        def load(progress, cancel):
            from ..core.imageio import load_image
            src = load_image(j["source_path"])
            res = load_image(out["path"]) if out and Path(out["path"]).is_file() else None
            return (to_qimage(src.rgb, src.icc_profile, src.alpha),
                    to_qimage(res.rgb, res.icc_profile, res.alpha) if res else None)

        def done(imgs):
            if self._preview_job != job_id:
                return
            a, b = imgs
            self.preview.labels = ("Before · Source", "After · Restored Master" if b is not None else "After · (pending)")
            self.preview.set_images(a, b)
            rv = json.loads(j["review_json"]) if j["review_json"] else {}
            rc = Recipe.from_json(json.loads(j["recipe_json"])).summary() if j["recipe_json"] else "—"
            t = json.loads(j["timings_json"]) if j["timings_json"] else {}
            secs = f"{t['total_job_ms'] / 1000:.1f} s" if isinstance(t.get("total_job_ms"), (int, float)) else ""
            flags = rv.get("flags", [])
            self.preview_info.setText(f"{Path(j['source_path']).name} · {j['state']} {secs}\n{rc}"
                                      + (("\n⚑ " + " | ".join(flags[:3])) if flags else "")
                                      + (f"\n{j['error']}" if j["error"] else ""))

        run_bg(load, on_done=done, on_error=lambda m: self.preview_info.setText("Preview failed: " + m[:200]))

    def _follow_latest(self, jobs: list[dict]):
        if not self.follow.isChecked():
            return
        finished = [j for j in jobs if j["state"] in ("done", "review") and j["id"] not in self._preview_seen]
        if finished:
            latest = max(finished, key=lambda j: j["finished"] or 0)
            for j in finished:
                self._preview_seen.add(j["id"])
            self.show_preview(latest["id"])

    def _open_row(self, row: int, col: int):
        jid = int(self.table.item(row, 8).text())
        j = self.studio.db.one("SELECT * FROM jobs WHERE id=?", (jid,))
        if j:
            self.openInPrepare.emit(j["source_path"], (self.jm, jid), j["rel_path"] or "")

    def shutdown(self):
        self.timer.stop()
        self.jm.shutdown()

    def set_generative_available(self, ok: bool, why: str = "") -> None:
        item = self.mode.model().item(2)
        item.setEnabled(bool(ok))
        item.setToolTip(why)
        if not ok and self.mode.currentData() == "generative":
            self.mode.setCurrentIndex(0)

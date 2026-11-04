"""Outputs / Gallery tab: search, filter, sort, approve/reject, open folder, export selected."""
from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (QComboBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QListView, QListWidget,
                               QListWidgetItem, QPushButton, QSplitter, QTextBrowser, QVBoxLayout, QWidget)

from ..core.engine import Studio
from ..core.imageio import load_image, write_bytes_no_clobber
from ..core.outputs import KIND_LABELS
from .common import open_folder, to_qimage

THUMB = 160


class GalleryPanel(QWidget):
    status = Signal(str)
    openInPrepare = Signal(str)

    def __init__(self, studio: Studio, parent=None):
        super().__init__(parent)
        self.studio = studio
        self._thumbs: dict[str, QIcon] = {}
        root = QVBoxLayout(self)
        row = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search file name, recipe, model, label…")
        self.search.textChanged.connect(self.refresh)
        row.addWidget(self.search, 2)
        self.kind = QComboBox()
        self.kind.addItem("All kinds", None)
        for k, lab in KIND_LABELS.items():
            if k != "source":
                self.kind.addItem(lab, k)
        self.kind.currentIndexChanged.connect(self.refresh)
        row.addWidget(self.kind)
        self.approval = QComboBox()
        for lab, v in (("Any approval", None), ("Needs review", "pending"), ("Approved", "approved"),
                       ("Rejected", "rejected"), ("Auto (no flags)", "auto")):
            self.approval.addItem(lab, v)
        self.approval.currentIndexChanged.connect(self.refresh)
        row.addWidget(self.approval)
        self.batch = QComboBox()
        self.batch.currentIndexChanged.connect(self.refresh)
        row.addWidget(self.batch)
        self.model = QComboBox()
        self.model.currentIndexChanged.connect(self.refresh)
        row.addWidget(self.model)
        self.sort = QComboBox()
        for lab, v in (("Newest first", "created DESC"), ("Oldest first", "created ASC"), ("Name", "path ASC"),
                       ("Source folder", "path ASC"), ("Slowest", "elapsed_ms DESC")):
            self.sort.addItem(lab, v)
        self.sort.currentIndexChanged.connect(self.refresh)
        row.addWidget(self.sort)
        b = QPushButton("Refresh")
        b.clicked.connect(self.reload_filters)
        row.addWidget(b)
        root.addLayout(row)
        split = QSplitter(Qt.Orientation.Horizontal)
        self.grid = QListWidget()
        self.grid.setViewMode(QListView.ViewMode.IconMode)
        self.grid.setIconSize(QSize(THUMB, THUMB))
        self.grid.setGridSize(QSize(THUMB + 30, THUMB + 46))
        self.grid.setResizeMode(QListView.ResizeMode.Adjust)
        self.grid.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.grid.setWordWrap(True)
        self.grid.itemSelectionChanged.connect(self._details)
        self.grid.itemDoubleClicked.connect(lambda it: open_folder(it.data(Qt.ItemDataRole.UserRole)["path"]))
        split.addWidget(self.grid)
        side = QWidget()
        sl = QVBoxLayout(side)
        self.info = QTextBrowser()
        self.info.setOpenLinks(False)
        self.info.setHtml("Select an output.")
        sl.addWidget(self.info, 1)
        for lab, fn in (("Approve", lambda: self._set_approval("approved")),
                        ("Reject", lambda: self._set_approval("rejected")),
                        ("Open containing folder", self._open), ("Export selected to…", self._export_selected),
                        ("Compare source in Prepare", self._compare)):
            b = QPushButton(lab)
            b.clicked.connect(fn)
            sl.addWidget(b)
        split.addWidget(side)
        side.setMinimumWidth(360)
        split.setStretchFactor(0, 3)
        root.addWidget(split, 1)
        self.count = QLabel("")
        root.addWidget(self.count)
        self.reload_filters()

    def reload_filters(self):
        for combo, sql, lab in ((self.batch, "SELECT id, name FROM batches ORDER BY id DESC", "All batches"),):
            combo.blockSignals(True)
            combo.clear()
            combo.addItem(lab, None)
            combo.addItem("Single images", -1)
            for r in self.studio.db.query(sql):
                combo.addItem(f"#{r['id']} {r['name']}", r["id"])
            combo.blockSignals(False)
        self.model.blockSignals(True)
        self.model.clear()
        self.model.addItem("Any model/recipe", None)
        ms = set()
        for r in self.studio.db.query("SELECT DISTINCT models FROM outputs"):
            ms.update(m for m in (r["models"] or "").split(",") if m)
        for m in sorted(ms):
            self.model.addItem(m, m)
        self.model.addItem("classical only / no-op", "")
        self.model.blockSignals(False)
        self.refresh()

    def refresh(self):
        q = "SELECT * FROM outputs WHERE 1=1"
        args: list = []
        if self.kind.currentData():
            q += " AND kind=?"
            args.append(self.kind.currentData())
        if self.approval.currentData():
            q += " AND approval=?"
            args.append(self.approval.currentData())
        b = self.batch.currentData()
        if b == -1:
            q += " AND batch_id IS NULL"
        elif b:
            q += " AND batch_id=?"
            args.append(b)
        m = self.model.currentData()
        if m == "":
            q += " AND (models IS NULL OR models='')"
        elif m:
            q += " AND (',' || models || ',') LIKE ?"
            args.append(f"%,{m},%")
        t = self.search.text().strip()
        if t:
            q += " AND (path LIKE ? OR recipe_summary LIKE ? OR label LIKE ? OR models LIKE ?)"
            args += [f"%{t}%"] * 4
        q += f" ORDER BY {self.sort.currentData() or 'created DESC'} LIMIT 2000"
        rows = self.studio.db.query(q, args)
        self.grid.clear()
        for r in rows:
            d = dict(r)
            it = QListWidgetItem(self._icon(d["path"]),
                                 f"{KIND_LABELS.get(d['kind'], d['kind'])}\n{Path(d['path']).name[:40]}")
            it.setData(Qt.ItemDataRole.UserRole, d)
            it.setToolTip(d["path"])
            self.grid.addItem(it)
        self.count.setText(f"{len(rows)} output(s)")

    def _icon(self, path: str) -> QIcon:
        if path in self._thumbs:
            return self._thumbs[path]
        icon = QIcon()
        if Path(path).is_file():
            pm = QPixmap(path)
            if pm.isNull():  # e.g. 16-bit TIFF: decode ourselves
                try:
                    im = load_image(path)
                    import cv2
                    s = THUMB / max(im.width, im.height)
                    small = cv2.resize(im.rgb, (max(1, int(im.width * s)), max(1, int(im.height * s))),
                                       interpolation=cv2.INTER_AREA)
                    a = cv2.resize(im.alpha, small.shape[1::-1]) if im.alpha is not None else None
                    pm = QPixmap.fromImage(to_qimage(small, im.icc_profile, a))
                except Exception:
                    pm = QPixmap()
            if not pm.isNull():
                icon = QIcon(pm.scaled(THUMB, THUMB, Qt.AspectRatioMode.KeepAspectRatio,
                                       Qt.TransformationMode.SmoothTransformation))
        self._thumbs[path] = icon
        return icon

    def _selected(self) -> list[dict]:
        return [it.data(Qt.ItemDataRole.UserRole) for it in self.grid.selectedItems()]

    def _details(self):
        sel = self._selected()
        if not sel:
            return
        d = sel[0]
        a = json.loads(d["audit_json"] or "{}")
        flags = json.loads(d["review_flags_json"] or "[]")
        src = self.studio.db.one("SELECT path, sha256 FROM sources WHERE id=?", (d["source_id"],)) if d["source_id"] else None
        self.info.setHtml(
            f"<b>{KIND_LABELS.get(d['kind'], d['kind'])}</b> · {d['format']} {d['width']}×{d['height']} "
            f"{d['bit_depth']}-bit<br>{d['path']}<br>Source: {src['path'] if src else '—'}<br>"
            f"Recipe {d['recipe_hash']}: {d['recipe_summary']}<br>Models: {d['models'] or 'none'}<br>"
            f"Approval: <b>{d['approval']}</b>{' · label ' + d['label'] if d['label'] else ''}<br>"
            f"Output SHA-256 {d['sha256'][:16]}…<br>Metadata: {'verified' if a.get('verified') else 'not verified'}"
            f", {len(a.get('dropped', {}))} non-transferable, {len(a.get('stripped_location', []))} location tags removed"
            + ("<br><span style='color:#ffb74d'>⚑ " + "<br>⚑ ".join(flags) + "</span>" if flags else ""))

    def _set_approval(self, v: str):
        for d in self._selected():
            self.studio.db.update("outputs", d["id"], {"approval": v})
        self.refresh()

    def _open(self):
        sel = self._selected()
        if sel:
            open_folder(sel[0]["path"])

    def _export_selected(self):
        sel = self._selected()
        if not sel:
            return
        d = QFileDialog.getExistingDirectory(self, "Export selected outputs to")
        if not d:
            return
        n = 0
        for o in sel:
            p = Path(o["path"])
            if p.is_file():
                write_bytes_no_clobber(p.read_bytes(), Path(d) / p.name)
                n += 1
        self.status.emit(f"Copied {n} file(s) to {d} (existing files were not overwritten).")

    def _compare(self):
        sel = self._selected()
        if sel and sel[0]["source_id"]:
            src = self.studio.db.one("SELECT path FROM sources WHERE id=?", (sel[0]["source_id"],))
            if src and Path(src["path"]).is_file():
                self.openInPrepare.emit(src["path"])


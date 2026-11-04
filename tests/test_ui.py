"""UI tests (offscreen Qt). Drive the real widgets; screenshots go to reports/ui/ for visual review.

These are code-level UI checks. Real Windows desktop validation (HiDPI, drag/drop from Explorer,
GPU responsiveness) is tracked separately in docs/TEST_RESULTS.md.
"""
import os
import time
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QPointF, QRectF, Qt

from tests.conftest import ROOT

SHOTS = ROOT / "reports" / "ui"


@pytest.fixture()
def win(qtbot, tmp_path):
    from ips.app import apply_theme
    from ips.core.db import DB
    from ips.core.engine import Studio
    from ips.settings import Settings
    from ips.ui.main_window import MainWindow
    from PySide6.QtWidgets import QApplication
    apply_theme(QApplication.instance())
    s = Settings()
    s.output_root = str(tmp_path / "out")
    s.device = "cpu"
    st = Studio(s, DB(tmp_path / "ui.sqlite"))
    w = MainWindow(st)
    w.resize(1500, 950)
    qtbot.addWidget(w)
    w.show()
    yield w
    w.batch.shutdown()


def _wait(qtbot, cond, timeout=600_000):
    qtbot.waitUntil(cond, timeout=timeout)


def shot(w, name):
    SHOTS.mkdir(parents=True, exist_ok=True)
    w.grab().save(str(SHOTS / f"{name}.png"))


def test_compare_view_alignment_math(qtbot):
    from ips.ui.common import to_qimage
    from ips.ui.widgets.compare_view import CompareView
    v = CompareView()
    qtbot.addWidget(v)
    v.resize(800, 600)
    v.show()
    src = np.random.default_rng(0).random((100, 150, 3)).astype(np.float32)
    out = np.random.default_rng(1).random((200, 300, 3)).astype(np.float32)
    v.set_images(to_qimage(src), to_qimage(out))
    r = v.target_rect()
    # both layers share one content rect -> source pixel (x,y) maps exactly onto output pixel (2x,2y)
    for sx, sy in ((0, 0), (75, 50), (149, 99)):
        wp = QPointF(r.x() + (sx * 2 + 1) * v.zoom, r.y() + (sy * 2 + 1) * v.zoom)
        s = v.widget_to_source(wp)
        assert abs(s.x() - (sx + 0.5)) < 1e-6 and abs(s.y() - (sy + 0.5)) < 1e-6
    assert v.source_scale() == 2.0
    # 100% means one image pixel per *device* pixel (HiDPI aware)
    v.set_zoom_percent(100)
    assert abs(v.zoom * v.devicePixelRatioF() - 1.0) < 1e-6
    # zoom around a point keeps that image point under the cursor
    anchor = QPointF(400, 300)
    before = v.widget_to_content(anchor)
    v._zoom_to(v.zoom * 2, anchor)
    after = v.widget_to_content(anchor)
    assert abs(before.x() - after.x()) < 1e-6 and abs(before.y() - after.y()) < 1e-6
    # wipe drag
    v.set_zoom_percent(None)
    r = v.target_rect()
    qtbot.mousePress(v, Qt.MouseButton.LeftButton, pos=QPointF(r.x() + r.width() * 0.5, 300).toPoint())
    qtbot.mouseMove(v, QPointF(r.x() + r.width() * 0.25, 300).toPoint())
    qtbot.mouseRelease(v, Qt.MouseButton.LeftButton, pos=QPointF(r.x() + r.width() * 0.25, 300).toPoint())
    assert abs(v.wipe - 0.25) < 0.02
    # face-crop focus maps a source box to fill the view
    v.focus_source_rect(50, 20, 100, 70)
    c = v.widget_to_source(QPointF(400, 300))
    assert abs(c.x() - 75) < 1 and abs(c.y() - 45) < 1


def test_prepare_auto_manual_export(win, qtbot, corpus):
    p = win.prepare
    p.open_path(str(corpus / "edited/print_scan_faded.jpg"))
    _wait(qtbot, lambda: len(p.variants) >= 1)
    v = p.variants[0]
    assert "white_balance" in {o.kind for o in v.recipe.ops}
    assert p.editor.auto_recipe is not None and p.editor.recipe.hash == p.editor.auto_recipe.hash
    assert p.view.after is not None and p.view.before is not None
    shot(win, "01_prepare_auto")
    # Auto -> Manual: disable white balance, run manual
    row = next(p.editor.rows_box.itemAt(i).widget() for i in range(p.editor.rows_box.count())
               if p.editor.rows_box.itemAt(i).widget().op.kind == "white_balance")
    row.enabled.setChecked(False)
    assert p.editor.recipe.hash != p.editor.auto_recipe.hash and "modified" in p.editor.summary.text()
    assert p.editor.undo() and p.editor.recipe.hash == p.editor.auto_recipe.hash
    assert p.editor.redo()
    p.run_manual()
    _wait(qtbot, lambda: len(p.variants) >= 2)
    assert p.variants[1].recipe.created_by == "manual"
    # A/B between the two variants
    p.left_sel.setCurrentIndex(1)
    p.right_sel.setCurrentIndex(2)
    assert "Auto 1" in p.view.labels[0] and "Manual 2" in p.view.labels[1]
    p.view.set_mode("side")
    shot(win, "02_prepare_variants_side")
    p.view.set_mode("wipe")
    for k, c in p.exp_checks.items():
        c.setChecked(True)
    p.export()
    _wait(qtbot, lambda: "verified" in p.export_info.text())
    assert p.export_info.text().count("verified") >= 3
    row = win.studio.db.one("SELECT COUNT(*) n FROM corrections")
    assert row["n"] == 1, "export of a manual variant must be recorded as a correction"
    shot(win, "03_prepare_exported")


def test_quick_edit_cutout_recolor(win, qtbot, corpus):
    q = win.quick
    win.tabs.setCurrentWidget(q)
    q.open_path(str(corpus / "cutout/children_texture.png"))
    _wait(qtbot, lambda: q.src is not None, 60_000)
    q._auto_mask("u2net_human_seg")
    _wait(qtbot, lambda: q.mask.mask.mean() > 0.05, 120_000)
    shot(win, "04_quickedit_mask")
    q.cutout()
    assert q.alpha is not None
    q.save("cutout")
    assert "Transparent Cutout" in q.save_info.text()
    shot(win, "05_quickedit_cutout")
    q.reset()
    q._set_tool("magic")
    q._clicked(300, 200, 1)
    assert q.mask.mask.any()
    q.do_recolor()
    assert q.edits and q.edits[-1].startswith("recolor")
    q.undo()
    assert not q.edits
    q.redo()
    assert q.edits
    q.save("edited")
    assert "Edited Variant" in q.save_info.text()
    shot(win, "06_quickedit_recolor")


def test_batch_and_gallery(win, qtbot, corpus, tmp_path):
    b = win.batch
    win.tabs.setCurrentWidget(b)
    b.add_paths([str(corpus / "broken")])
    b.scale.setCurrentIndex(1)  # ×1
    b.start_new()
    _wait(qtbot, lambda: not b.jm.running and b.jm.jobs(b.batch_id)[-1]["state"] != "queued", 600_000)
    b.refresh()
    assert b.table.rowCount() == 6
    states = {b.table.item(r, 2).text() for r in range(6)}
    assert "failed" in states and ({"done", "review"} & states)
    b.filter.setCurrentIndex(2)  # Failed
    b.refresh()
    assert b.table.rowCount() == 4
    shot(win, "07_batch")
    g = win.gallery
    win.tabs.setCurrentWidget(g)
    g.reload_filters()
    assert g.grid.count() >= 2
    g.grid.setCurrentRow(0)
    g._set_approval("approved")
    g.approval.setCurrentIndex(2)
    assert g.grid.count() >= 1
    shot(win, "08_gallery")

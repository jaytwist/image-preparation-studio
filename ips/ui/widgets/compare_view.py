"""Before/After comparison canvas.

Alignment: both images are mapped onto the same *content rectangle* (output pixel grid, W×H). The
source is lower resolution when upscaled; it is scaled for display only, so every source pixel lands
exactly over the output pixels it produced. Wipe position, zoom and pan are shared by both layers.

Zoom is in *physical* pixels: "100%" means one image pixel per device pixel on HiDPI screens.
Only the visible part of each image is drawn, so 50+ MP outputs stay responsive.
"""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

MODES = ("wipe", "before", "after", "side")


class CompareView(QWidget):
    viewChanged = Signal(float, float, float)          # zoom, center_x, center_y (content px)
    regionSelected = Signal(QRectF)                     # in *source* pixel coordinates
    imageClicked = Signal(float, float, int)            # source coords, mouse button
    imageDragged = Signal(float, float, float, float, int)  # from, to (source coords), buttons
    filesDropped = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(320, 240)
        self.setMouseTracking(True)
        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.before: QImage | None = None
        self.after: QImage | None = None
        self.overlay: QImage | None = None              # e.g. mask, drawn over 'after' in content space
        self.labels = ("Before · Source", "After")
        self.mode = "wipe"
        self.wipe = 0.5
        self.zoom = 1.0
        self.fit = True
        self.offset = QPointF(0, 0)
        self.smooth_below_100 = True
        self.tool = "view"                              # view | select | paint
        self.brush_radius_src = 0.0                     # for cursor preview in paint tool
        self._drag = None
        self._last = QPointF()
        self._sel_start: QPointF | None = None
        self._sel_rect: QRectF | None = None
        self._cursor = QPointF(-1, -1)
        self.empty_text = "Drop an image here, or use Open…"

    # ------------------------------------------------------------------ content
    def content_size(self) -> QSize:
        img = self.after or self.before
        return img.size() if img is not None else QSize(0, 0)

    def source_scale(self) -> float:
        """Content (output) pixels per source pixel."""
        if self.before is None or self.after is None or self.before.width() == 0:
            return 1.0
        return self.after.width() / self.before.width()

    def set_images(self, before: QImage | None, after: QImage | None, keep_view: bool = False):
        old = self.content_size()
        self.before, self.after = before, after
        if not keep_view or self.content_size() != old:
            self.fit = True
        self.update()

    def set_overlay(self, ov: QImage | None):
        self.overlay = ov
        self.update()

    def set_mode(self, m: str):
        self.mode = m
        self.update()

    # ------------------------------------------------------------------ view math
    def _dpr(self) -> float:
        return float(self.devicePixelRatioF() or 1.0)

    def _panes(self) -> int:
        return 2 if self.mode == "side" else 1

    def _pane_w(self) -> float:
        return self.width() / self._panes()

    def _fit_zoom(self) -> float:
        cs = self.content_size()
        if cs.isEmpty():
            return 1.0
        return min((self._pane_w() - 16) / cs.width(), (self.height() - 16) / cs.height())

    def _apply_fit(self):
        cs = self.content_size()
        self.zoom = self._fit_zoom()
        self.offset = QPointF((self._pane_w() - cs.width() * self.zoom) / 2, (self.height() - cs.height() * self.zoom) / 2)

    def target_rect(self, pane: int = 0) -> QRectF:
        if self.fit:
            self._apply_fit()
        cs = self.content_size()
        return QRectF(self.offset.x() + pane * self._pane_w(), self.offset.y(), cs.width() * self.zoom,
                      cs.height() * self.zoom)

    def widget_to_content(self, p: QPointF) -> QPointF:
        r = self.target_rect(0)
        x = p.x() % self._pane_w() if self.mode == "side" else p.x()
        return QPointF((x - r.x()) / self.zoom, (p.y() - r.y()) / self.zoom)

    def widget_to_source(self, p: QPointF) -> QPointF:
        c = self.widget_to_content(p)
        s = self.source_scale()
        return QPointF(c.x() / s, c.y() / s)

    def zoom_percent(self) -> float:
        return self.zoom * self._dpr() * 100

    def set_zoom_percent(self, pct: float | None, anchor: QPointF | None = None):
        if pct is None:
            self.fit = True
            self.update()
            self._emit_view()
            return
        anchor = anchor or QPointF(self._pane_w() / 2, self.height() / 2)
        self._zoom_to(pct / 100 / self._dpr(), anchor)

    def _zoom_to(self, z: float, anchor: QPointF):
        if self.fit:
            self._apply_fit()
        self.fit = False
        z = max(0.02, min(64.0, z))
        c = self.widget_to_content(anchor)
        ax = anchor.x() % self._pane_w() if self.mode == "side" else anchor.x()
        self.zoom = z
        self.offset = QPointF(ax - c.x() * z, anchor.y() - c.y() * z)
        self.update()
        self._emit_view()

    def focus_source_rect(self, x0: float, y0: float, x1: float, y1: float, margin: float = 0.35):
        """Crop preview: zoom so the given source-pixel box (e.g. a face) fills the view."""
        s = self.source_scale()
        x0, y0, x1, y1 = x0 * s, y0 * s, x1 * s, y1 * s
        w, h = (x1 - x0) * (1 + margin), (y1 - y0) * (1 + margin)
        self.fit = False
        self.zoom = min(self._pane_w() / max(w, 1), self.height() / max(h, 1))
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        self.offset = QPointF(self._pane_w() / 2 - cx * self.zoom, self.height() / 2 - cy * self.zoom)
        self.update()
        self._emit_view()

    def sync_from(self, zoom: float, cx: float, cy: float):
        self.fit = False
        self.zoom = zoom
        self.offset = QPointF(self._pane_w() / 2 - cx * zoom, self.height() / 2 - cy * zoom)
        self.update()

    def _emit_view(self):
        c = self.widget_to_content(QPointF(self._pane_w() / 2, self.height() / 2))
        self.viewChanged.emit(self.zoom, c.x(), c.y())

    # ------------------------------------------------------------------ painting
    def _draw_image(self, p: QPainter, img: QImage, target: QRectF, clip: QRectF):
        """Draw only the visible part of *img* mapped to *target*."""
        vis = target.intersected(clip)
        if vis.isEmpty():
            return
        sx = img.width() / target.width()
        sy = img.height() / target.height()
        src = QRectF((vis.x() - target.x()) * sx, (vis.y() - target.y()) * sy, vis.width() * sx, vis.height() * sy)
        p.drawImage(vis, img, src)

    def paintEvent(self, ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(38, 40, 44))
        if self.content_size().isEmpty():
            p.setPen(QColor(170, 170, 175))
            f = QFont(self.font())
            f.setPointSizeF(f.pointSizeF() * 1.3)
            p.setFont(f)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.empty_text)
            return
        smooth = self.smooth_below_100 and self.zoom * self._dpr() < 0.999
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, smooth)
        full = QRectF(self.rect())
        if self.mode == "side":
            for pane, (img, lab) in enumerate(((self.before, self.labels[0]), (self.after, self.labels[1]))):
                r = self.target_rect(pane)
                clip = QRectF(pane * self._pane_w(), 0, self._pane_w(), self.height())
                p.save()
                p.setClipRect(clip)
                if img is not None:
                    self._draw_image(p, img, r, clip)
                if pane == 1 and self.overlay is not None:
                    self._draw_image(p, self.overlay, r, clip)
                p.restore()
                self._label(p, lab, clip.x() + 8, 8)
            p.setPen(QPen(QColor(255, 255, 255, 120), 1))
            p.drawLine(int(self._pane_w()), 0, int(self._pane_w()), self.height())
        else:
            r = self.target_rect(0)
            if self.mode == "before" or (self.after is None):
                if self.before is not None:
                    self._draw_image(p, self.before, r, full)
                self._label(p, self.labels[0], 8, 8)
            elif self.mode == "after":
                self._draw_image(p, self.after, r, full)
                if self.overlay is not None:
                    self._draw_image(p, self.overlay, r, full)
                self._label(p, self.labels[1], 8, 8)
            else:
                x = r.x() + self.wipe * r.width()
                left = QRectF(full.x(), full.y(), max(0.0, x - full.x()), full.height())
                right = QRectF(x, full.y(), max(0.0, full.right() - x), full.height())
                if self.before is not None:
                    p.save()
                    p.setClipRect(left)
                    self._draw_image(p, self.before, r, left)
                    p.restore()
                p.save()
                p.setClipRect(right)
                self._draw_image(p, self.after, r, right)
                if self.overlay is not None:
                    self._draw_image(p, self.overlay, r, right)
                p.restore()
                p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
                p.setPen(QPen(QColor(255, 255, 255, 230), 2))
                p.drawLine(QPointF(x, 0), QPointF(x, self.height()))
                p.setBrush(QColor(255, 255, 255, 235))
                p.setPen(QPen(QColor(30, 30, 30, 200), 1))
                p.drawEllipse(QPointF(x, self.height() / 2), 11, 11)
                p.setPen(QPen(QColor(40, 40, 40), 2))
                cy = self.height() / 2
                p.drawLine(QPointF(x - 5, cy), QPointF(x - 2, cy - 3))
                p.drawLine(QPointF(x - 5, cy), QPointF(x - 2, cy + 3))
                p.drawLine(QPointF(x + 5, cy), QPointF(x + 2, cy - 3))
                p.drawLine(QPointF(x + 5, cy), QPointF(x + 2, cy + 3))
                self._label(p, self.labels[0], 8, 8)
                self._label(p, self.labels[1], self.width() - 8, 8, right_align=True)
        if self._sel_rect is not None:
            p.setPen(QPen(QColor(80, 180, 255), 1, Qt.PenStyle.DashLine))
            p.setBrush(QColor(80, 180, 255, 40))
            p.drawRect(self._sel_rect)
        if self.tool == "paint" and self.brush_radius_src > 0 and self._cursor.x() >= 0:
            rr = self.brush_radius_src * self.source_scale() * self.zoom
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(QColor(255, 255, 255, 200), 1))
            p.drawEllipse(self._cursor, rr, rr)
        self._label(p, f"{self.zoom_percent():.0f}%", 8, self.height() - 28)

    def _label(self, p: QPainter, text: str, x: float, y: float, right_align: bool = False):
        fm = p.fontMetrics()
        w = fm.horizontalAdvance(text) + 12
        h = fm.height() + 6
        if right_align:
            x -= w
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0, 0, 0, 150))
        p.drawRoundedRect(QRectF(x, y, w, h), 4, 4)
        p.setPen(QColor(240, 240, 240))
        p.drawText(QRectF(x, y, w, h), Qt.AlignmentFlag.AlignCenter, text)

    # ------------------------------------------------------------------ interaction
    def _near_wipe(self, pos: QPointF) -> bool:
        if self.mode != "wipe" or self.after is None:
            return False
        r = self.target_rect(0)
        return abs(pos.x() - (r.x() + self.wipe * r.width())) < 12

    def mousePressEvent(self, e):
        pos = e.position()
        self._last = pos
        if e.button() == Qt.MouseButton.LeftButton and self._near_wipe(pos) and self.tool != "paint":
            self._drag = "wipe"
        elif e.button() == Qt.MouseButton.LeftButton and (self.tool == "select" or
                                                          e.modifiers() & Qt.KeyboardModifier.ShiftModifier):
            self._drag = "select"
            self._sel_start = pos
            self._sel_rect = QRectF(pos, pos)
        elif e.button() == Qt.MouseButton.LeftButton and self.tool == "paint":
            self._drag = "paint"
            s = self.widget_to_source(pos)
            self.imageClicked.emit(s.x(), s.y(), int(e.buttons().value))
        elif e.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            self._drag = "pan"
            if self.tool == "click" and e.button() == Qt.MouseButton.LeftButton:
                s = self.widget_to_source(pos)
                self.imageClicked.emit(s.x(), s.y(), 1)
                self._drag = None
        elif e.button() == Qt.MouseButton.RightButton and self.tool == "paint":
            self._drag = "paint"
            s = self.widget_to_source(pos)
            self.imageClicked.emit(s.x(), s.y(), int(e.buttons().value))

    def mouseMoveEvent(self, e):
        pos = e.position()
        self._cursor = pos
        if self._drag == "wipe":
            r = self.target_rect(0)
            self.wipe = max(0.0, min(1.0, (pos.x() - r.x()) / max(1.0, r.width())))
        elif self._drag == "pan":
            if self.fit:
                self._apply_fit()
                self.fit = False
            self.offset += pos - self._last
            self._emit_view()
        elif self._drag == "select" and self._sel_start is not None:
            self._sel_rect = QRectF(self._sel_start, pos).normalized()
        elif self._drag == "paint":
            a, b = self.widget_to_source(self._last), self.widget_to_source(pos)
            self.imageDragged.emit(a.x(), a.y(), b.x(), b.y(), int(e.buttons().value))
        else:
            self.setCursor(Qt.CursorShape.SplitHCursor if self._near_wipe(pos) else Qt.CursorShape.ArrowCursor)
        self._last = pos
        self.update()

    def mouseReleaseEvent(self, e):
        if self._drag == "select" and self._sel_rect is not None and self._sel_rect.width() > 4:
            a = self.widget_to_source(self._sel_rect.topLeft())
            b = self.widget_to_source(self._sel_rect.bottomRight())
            self.regionSelected.emit(QRectF(a, b).normalized())
        if self._drag == "select":
            self._sel_rect = None
        self._drag = None
        self.update()

    def wheelEvent(self, e):
        d = e.angleDelta().y()
        if d == 0:
            return
        if self.fit:
            self._apply_fit()
        self._zoom_to(self.zoom * (1.25 if d > 0 else 0.8), e.position())

    def mouseDoubleClickEvent(self, e):
        if self.fit:
            self.set_zoom_percent(100, e.position())
        else:
            self.set_zoom_percent(None)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.update()

    def leaveEvent(self, e):
        self._cursor = QPointF(-1, -1)
        self.update()

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        files = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        if files:
            self.filesDropped.emit(files)

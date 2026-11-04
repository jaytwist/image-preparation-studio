"""Qt helpers: numpy<->QImage, background workers, small widgets."""
from __future__ import annotations

import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QFrame, QLabel, QSizePolicy, QToolButton, QVBoxLayout, QWidget

from ..core import imageio


def to_qimage(rgb: np.ndarray, icc: bytes | None = None, alpha: np.ndarray | None = None,
              checker: bool = True) -> QImage:
    """Display rendering: converts to sRGB for the screen (saved files are never affected)."""
    u8 = imageio.display_srgb_u8(rgb, icc)
    if alpha is not None:
        a = alpha[..., None]
        if checker:
            h, w = alpha.shape
            yy, xx = np.mgrid[:h, :w]
            ch = (((yy // 12) + (xx // 12)) % 2 * 40 + 190).astype(np.float32)[..., None] / 255
            comp = u8.astype(np.float32) / 255 * a + ch * (1 - a)
            u8 = (comp * 255 + 0.5).astype(np.uint8)
    u8 = np.ascontiguousarray(u8)
    h, w = u8.shape[:2]
    img = QImage(u8.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    return img.copy()  # own the buffer


def mask_overlay(mask: np.ndarray, color=(255, 40, 60), opacity: float = 0.45) -> QImage:
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = color
    rgba[..., 3] = (np.clip(mask, 0, 1) * 255 * opacity).astype(np.uint8)
    rgba = np.ascontiguousarray(rgba)
    return QImage(rgba.data, w, h, 4 * w, QImage.Format.Format_RGBA8888).copy()


class WorkerSignals(QObject):
    progress = Signal(str, float)
    done = Signal(object)
    error = Signal(str)


class Worker(QRunnable):
    """Run fn(progress=..., cancel=...) on the thread pool; results come back on the UI thread."""

    def __init__(self, fn: Callable[..., Any], *args, **kwargs):
        super().__init__()
        self.fn, self.args, self.kwargs = fn, args, kwargs
        self.signals = WorkerSignals()
        self.cancelled = False
        # Python owns the lifetime (see _ACTIVE): C++ must not delete the runnable under us.
        self.setAutoDelete(False)

    def cancel(self):
        self.cancelled = True

    def run(self):
        try:
            res = self.fn(*self.args, **self.kwargs,
                          progress=lambda s, f: self.signals.progress.emit(str(s), float(f)),
                          cancel=lambda: self.cancelled)
            self.signals.done.emit(res)
        except Exception as e:  # surfaced in UI, never swallowed
            self.signals.error.emit(f"{type(e).__name__}: {e}\n\n{traceback.format_exc(limit=6)}")


_ACTIVE: set[Worker] = set()


def run_bg(fn, *args, on_done=None, on_error=None, on_progress=None, **kwargs) -> Worker:
    """Start *fn* on the thread pool. The worker (and its signal object) is kept alive until its
    result has been delivered on the UI thread, so callers need not hold a reference."""
    w = Worker(fn, *args, **kwargs)
    if on_done:
        w.signals.done.connect(on_done)
    if on_error:
        w.signals.error.connect(on_error)
    if on_progress:
        w.signals.progress.connect(on_progress)
    w.signals.done.connect(lambda *_: _ACTIVE.discard(w))
    w.signals.error.connect(lambda *_: _ACTIVE.discard(w))
    _ACTIVE.add(w)
    QThreadPool.globalInstance().start(w)
    return w


class Collapsible(QWidget):
    def __init__(self, title: str, content: QWidget, expanded: bool = False, parent=None):
        super().__init__(parent)
        self.btn = QToolButton(text=title, checkable=True, checked=expanded)
        self.btn.setStyleSheet("QToolButton { border: none; font-weight: 600; padding: 4px 0; }")
        self.btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.btn.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self.btn.toggled.connect(self._toggle)
        self.content = content
        content.setVisible(expanded)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.btn)
        lay.addWidget(content)

    def _toggle(self, on: bool):
        self.btn.setArrowType(Qt.ArrowType.DownArrow if on else Qt.ArrowType.RightArrow)
        self.content.setVisible(on)


def section_label(text: str) -> QLabel:
    lb = QLabel(text)
    lb.setStyleSheet("font-weight: 600; margin-top: 6px;")
    return lb


def hline() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setFrameShadow(QFrame.Shadow.Sunken)
    return f


def wrap_label(text: str = "", selectable: bool = True) -> QLabel:
    lb = QLabel(text)
    lb.setWordWrap(True)
    lb.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
    if selectable:
        lb.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return lb


def open_folder(path: str | Path) -> None:
    p = Path(path)
    if p.is_file():
        if sys.platform == "win32":
            subprocess.Popen(["explorer", "/select,", str(p)])
            return
        p = p.parent
    if sys.platform == "win32":
        os.startfile(str(p))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(p)])
    else:
        subprocess.Popen(["xdg-open", str(p)])


STATE_COLORS = {"done": "#2e7d32", "review": "#b26a00", "needs_override": "#8e24aa", "failed": "#c62828",
                "cancelled": "#616161", "duplicate": "#546e7a", "running": "#1565c0", "queued": "#757575",
                "skipped": "#757575"}

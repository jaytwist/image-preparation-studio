"""Desktop entry point:  python -m ips.app   (or run.ps1 on Windows)."""
from __future__ import annotations

import logging
import logging.handlers
import os
import sys


def _setup_logging():
    from .paths import log_dir
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fh = logging.handlers.RotatingFileHandler(log_dir() / "studio.log", maxBytes=5_000_000, backupCount=5,
                                              encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(sh)


def apply_theme(app):
    from PySide6.QtGui import QColor, QPalette
    app.setStyle("Fusion")
    p = QPalette()
    base, alt, text = QColor(43, 45, 49), QColor(53, 55, 60), QColor(226, 228, 232)
    p.setColor(QPalette.ColorRole.Window, base)
    p.setColor(QPalette.ColorRole.WindowText, text)
    p.setColor(QPalette.ColorRole.Base, QColor(33, 35, 38))
    p.setColor(QPalette.ColorRole.AlternateBase, alt)
    p.setColor(QPalette.ColorRole.Text, text)
    p.setColor(QPalette.ColorRole.Button, alt)
    p.setColor(QPalette.ColorRole.ButtonText, text)
    p.setColor(QPalette.ColorRole.Highlight, QColor(64, 132, 214))
    p.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    p.setColor(QPalette.ColorRole.ToolTipBase, QColor(25, 25, 28))
    p.setColor(QPalette.ColorRole.ToolTipText, text)
    p.setColor(QPalette.ColorRole.PlaceholderText, QColor(140, 143, 150))
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(120, 122, 128))
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(120, 122, 128))
    app.setPalette(p)


def main(argv=None) -> int:
    _setup_logging()
    # Per-monitor HiDPI is the Qt 6 default; keep fractional scale factors exact (125%/150% Windows scaling).
    os.environ.setdefault("QT_SCALE_FACTOR_ROUNDING_POLICY", "PassThrough")
    from PySide6.QtWidgets import QApplication

    from . import APP_NAME
    from .core.engine import Studio
    from .ui.main_window import MainWindow
    app = QApplication(sys.argv if argv is None else argv)
    app.setApplicationName(APP_NAME)
    apply_theme(app)
    studio = Studio()
    w = MainWindow(studio)
    w.show()
    args = (sys.argv if argv is None else argv)[1:]
    if args and os.path.isfile(args[0]):
        w.prepare.open_path(args[0])
    smoke = float(os.environ.get("IPS_SMOKE_SECONDS", "0") or 0)
    if smoke > 0:  # installation check: window must come up, backend probe must run, then exit cleanly
        from PySide6.QtCore import QTimer

        def done():
            ok = w.isVisible() and "Device:" in w.backend_lbl.text()
            logging.getLogger("ips.app").info("smoke: visible=%s backend=%r", w.isVisible(), w.backend_lbl.text())
            print("SMOKE", "OK" if ok else "FAIL", w.backend_lbl.text(), flush=True)
            w.close()
            app.exit(0 if ok else 3)
        QTimer.singleShot(int(smoke * 1000), done)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

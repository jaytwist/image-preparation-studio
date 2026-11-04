"""Offline / no-telemetry enforcement. Imported first by the ``ips`` package.

Finding (2026-11-04, see docs/TEST_RESULTS.md): onnxruntime 1.31 ships a 1DS telemetry client that
connects to mobile.events.data.microsoft.com from a background thread as soon as a session is created,
and ``onnxruntime.disable_telemetry_events()`` alone does NOT stop it. The environment variable
``ORT_DISABLE_TELEMETRY=1`` (read at library load) does; it is forced here before any import.
"""
from __future__ import annotations

import os

FORCED_ENV = {
    "ORT_DISABLE_TELEMETRY": "1",        # onnxruntime 1DS telemetry (verified with strace: 0 connects)
    "HF_HUB_OFFLINE": "1",               # never reach Hugging Face at runtime
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "DO_NOT_TRACK": "1",
    "TORCH_HOME": "",                    # placeholder; set below to an app-local folder (no implicit hub dirs)
}


def enforce() -> None:
    for k, v in FORCED_ENV.items():
        if k == "TORCH_HOME":
            continue
        os.environ[k] = v
    try:
        from .paths import app_data_dir
        os.environ.setdefault("TORCH_HOME", str(app_data_dir() / "torch_home"))
    except Exception:
        pass


def disable_runtime_telemetry() -> None:
    """Belt and braces after import (covers Windows ETW telemetry in onnxruntime)."""
    try:
        import onnxruntime as ort
        ort.disable_telemetry_events()
    except Exception:
        pass


enforce()

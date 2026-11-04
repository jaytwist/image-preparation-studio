"""Filesystem locations. Everything the app writes lives under app-owned folders.

Nothing here points into a ComfyUI or NH Maker 2 installation.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def app_data_dir() -> Path:
    env = os.environ.get("IPS_DATA_DIR")
    if env:
        p = Path(env)
    elif sys.platform == "win32":
        p = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "ImagePreparationStudio"
    else:
        p = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "image-preparation-studio"
    p.mkdir(parents=True, exist_ok=True)
    return p


def models_dir() -> Path:
    env = os.environ.get("IPS_MODELS_DIR")
    p = Path(env) if env else PROJECT_ROOT / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p


def default_output_root() -> Path:
    env = os.environ.get("IPS_OUTPUT_ROOT")
    if env:
        return Path(env)
    if sys.platform == "win32":
        return Path.home() / "Pictures" / "Image Preparation Studio"
    return Path.home() / "ImagePreparationStudio"


def db_path() -> Path:
    return app_data_dir() / "studio.sqlite"


def log_dir() -> Path:
    p = app_data_dir() / "logs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def cache_dir() -> Path:
    p = app_data_dir() / "cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


def settings_path() -> Path:
    return app_data_dir() / "settings.json"

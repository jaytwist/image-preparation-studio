"""User settings (JSON in the app data folder)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .paths import default_output_root, settings_path


@dataclass
class Settings:
    output_root: str = field(default_factory=lambda: str(default_output_root()))
    device: str = "auto"                 # auto | cuda | cpu
    tile: int = 0                        # 0 = per-model default
    max_cached_models: int = 3
    unload_idle_minutes: int = 5         # release GPU memory for other apps when Studio is idle (0 = never)
    default_mode: str = "faithful"
    upscale_low_res_auto: bool = True
    identity_check: bool = True
    fidelity_guard: bool = True
    quality_override_default: bool = False
    archive_format: str = "TIFF"         # TIFF | PNG
    archive_bits: str = "source"         # source | 16
    ai_ready_long_edge: int = 2048
    ai_ready_multiple_of: int = 1        # 1 = off; >1 pads (never crops) to a multiple
    ai_ready_format: str = "PNG"
    share_long_edge: int = 2560          # 0 = keep
    share_quality: int = 92
    share_strip_location: bool = True
    comfy_enabled: bool = False
    comfy_url: str = "http://127.0.0.1:8188"
    comfy_vram_min_free_mb: int = 4000
    recursive_import: bool = True
    auto_run_on_open: bool = True        # Prepare tab runs Smart Auto immediately after import

    @staticmethod
    def load(path: Path | None = None) -> "Settings":
        p = path or settings_path()
        s = Settings()
        if p.is_file():
            try:
                data = json.loads(p.read_text("utf-8"))
                for k, v in data.items():
                    if hasattr(s, k):
                        setattr(s, k, v)
            except Exception:
                pass
        return s

    def save(self, path: Path | None = None) -> None:
        p = path or settings_path()
        p.write_text(json.dumps(asdict(self), indent=2), "utf-8")

"""Direct local inference (PyTorch + spandrel) with tiling, device/VRAM management and native params.

* One GPU worker at a time (``GPU_LOCK``) – incompatible heavy models never run concurrently.
* Model cache is LRU with a VRAM budget; models are not unloaded/reloaded needlessly.
* Tiles are blended with a linear ramp across the overlap to avoid seams.
"""
from __future__ import annotations

import gc
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .registry import MODELS, ModelInfo

log = logging.getLogger("ips.direct")
GPU_LOCK = threading.RLock()


class CancelledError(Exception):
    pass


class BackendUnavailable(Exception):
    pass


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


@dataclass
class DeviceInfo:
    kind: str                 # cuda | cpu
    name: str
    total_vram_mb: int | None = None
    free_vram_mb: int | None = None
    torch_version: str = ""
    cuda_version: str | None = None
    capability: str | None = None
    fp16: bool = False


def device_info(prefer: str = "auto") -> DeviceInfo:
    import torch
    if prefer in ("auto", "cuda") and torch.cuda.is_available():
        i = torch.cuda.current_device()
        free, total = torch.cuda.mem_get_info(i)
        cap = torch.cuda.get_device_capability(i)
        return DeviceInfo("cuda", torch.cuda.get_device_name(i), total // 2**20, free // 2**20, torch.__version__,
                          torch.version.cuda, f"sm_{cap[0]}{cap[1]}", fp16=True)
    import platform
    return DeviceInfo("cpu", platform.processor() or platform.machine(), torch_version=torch.__version__)


class DirectBackend:
    """Loads spandrel models and runs them tiled."""

    def __init__(self, device: str = "auto", tile: int | None = None, overlap: int = 32, max_cached: int = 3,
                 vram_reserve_mb: int = 1500, use_half: bool = True):
        import torch
        self.torch = torch
        self.dev_info = device_info(device)
        self.device = torch.device("cuda" if self.dev_info.kind == "cuda" else "cpu")
        self.tile_override = tile
        self.overlap = overlap
        self.max_cached = max_cached
        self.vram_reserve_mb = vram_reserve_mb
        self.use_half = use_half and self.dev_info.kind == "cuda"
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self.load_times_ms: dict[str, float] = {}
        self.last_used = time.time()
        if self.dev_info.kind == "cpu":
            torch.set_num_threads(max(1, torch.get_num_threads()))
        else:
            torch.backends.cudnn.benchmark = True  # fixed tile shapes -> autotuned kernels

    # ------------------------------------------------------------------ loading
    def _load_descriptor(self, info: ModelInfo, variant: dict[str, Any] | None = None):
        from spandrel import ModelLoader
        if not info.installed:
            raise BackendUnavailable(f"Model file missing: {info.file}. Run tools/fetch_models.py (user-directed).")
        loader = ModelLoader(device="cpu")
        if info.id == "realesr_general_x4v3" and variant and "denoise_strength" in variant:
            s = float(variant["denoise_strength"])
            wdn_file = info.extra_files[0][0]
            import torch
            from ..paths import models_dir
            sd_a = torch.load(str(info.path), map_location="cpu", weights_only=True)
            sd_b = torch.load(str(models_dir() / wdn_file), map_location="cpu", weights_only=True)
            ka = "params" if "params" in sd_a else None
            a = sd_a[ka] if ka else sd_a
            b = sd_b["params"] if "params" in sd_b else sd_b
            mixed = {k: s * a[k] + (1 - s) * b[k] for k in a}
            desc = loader.load_from_state_dict(mixed)
        else:
            desc = loader.load_from_file(str(info.path))
        desc.model.eval()
        return desc

    def _key(self, model_id: str, params: dict[str, Any]) -> str:
        if model_id == "realesr_general_x4v3":
            return f"{model_id}@dni={round(float(params.get('denoise_strength', 0.5)), 2)}"
        return model_id

    def get(self, model_id: str, params: dict[str, Any] | None = None):
        info = MODELS[model_id]
        key = self._key(model_id, params or {})
        self.last_used = time.time()
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        self._make_room()
        t0 = time.perf_counter()
        desc = self._load_descriptor(info, params)
        half = self.use_half and getattr(desc, "supports_half", False)
        desc.to(self.device)
        if half:
            desc.model.half()
        desc._ips_half = half  # type: ignore[attr-defined]
        self.load_times_ms[key] = (time.perf_counter() - t0) * 1e3
        self._cache[key] = desc
        return desc

    def _make_room(self) -> None:
        while len(self._cache) >= self.max_cached:
            self._evict()
        if self.dev_info.kind == "cuda":
            free = self.torch.cuda.mem_get_info()[0] // 2**20
            while self._cache and free < self.vram_reserve_mb + 1500:
                self._evict()
                free = self.torch.cuda.mem_get_info()[0] // 2**20

    def _evict(self) -> None:
        if not self._cache:
            return
        k, d = self._cache.popitem(last=False)
        del d
        gc.collect()
        if self.dev_info.kind == "cuda":
            self.torch.cuda.empty_cache()
        log.info("evicted %s", k)

    def unload_all(self) -> None:
        while self._cache:
            self._evict()

    def unload_if_idle(self, idle_s: float) -> bool:
        """Release VRAM for other local apps (ComfyUI / NH Maker 2) after *idle_s* seconds without use."""
        if self._cache and time.time() - self.last_used > idle_s and GPU_LOCK.acquire(blocking=False):
            try:
                self.unload_all()
                return True
            finally:
                GPU_LOCK.release()
        return False

    def is_loaded(self, model_id: str, params: dict[str, Any] | None = None) -> bool:
        return self._key(model_id, params or {}) in self._cache

    # ------------------------------------------------------------------ inference
    def _forward(self, desc, x, model_id: str, params: dict[str, Any]):
        torch = self.torch
        if model_id == "drunet_color":
            sigma = float(params.get("sigma", 10.0)) / 255.0
            nm = torch.full((x.shape[0], 1, x.shape[2], x.shape[3]), sigma, dtype=x.dtype, device=x.device)
            return desc.model(torch.cat([x, nm], dim=1))
        if model_id == "fbcnn_color":
            qf = params.get("qf")
            if qf is None:
                return desc.model(x)[0]
            qf_in = torch.tensor([[1.0 - float(qf) / 100.0]], dtype=x.dtype, device=x.device)
            return desc.model(x, qf_in)[0]
        return desc(x)

    def predict_jpeg_qf(self, rgb: np.ndarray) -> float | None:
        """FBCNN's own blind QF prediction on a centre crop (pixel-level check of metadata QF)."""
        try:
            desc = self.get("fbcnn_color")
        except BackendUnavailable:
            return None
        torch = self.torch
        h, w = rgb.shape[:2]
        s = 256
        y0, x0 = max(0, (h - s) // 2) // 8 * 8, max(0, (w - s) // 2) // 8 * 8
        crop = rgb[y0:y0 + s, x0:x0 + s]
        with GPU_LOCK, torch.inference_mode():
            x = torch.from_numpy(np.ascontiguousarray(crop.transpose(2, 0, 1)))[None].to(self.device)
            if getattr(desc, "_ips_half", False):
                x = x.half()
            _, qf = desc.model(x)
        return float((1 - qf.float().item()) * 100)

    def run(self, model_id: str, rgb: np.ndarray, params: dict[str, Any] | None = None,
            cancel: Callable[[], bool] | None = None, progress: Callable[[float], None] | None = None) -> np.ndarray:
        params = params or {}
        info = MODELS[model_id]
        torch = self.torch
        with GPU_LOCK:
            desc = self.get(model_id, params)
            scale = int(desc.scale)
            half = getattr(desc, "_ips_half", False)
            req = desc.size_requirements
            mult = max(1, int(getattr(req, "multiple_of", 1) or 1))
            minimum = int(getattr(req, "minimum", 0) or 0)
            tile = self.tile_override or info.tile
            if self.dev_info.kind == "cuda" and not self.tile_override:
                free = torch.cuda.mem_get_info()[0] // 2**20
                if free < 3000:
                    tile = min(tile, 256)
                elif free > 8000:
                    # larger tiles amortise overlap on 16 GB cards; transformer (SwinIR) kept smaller
                    tile = max(tile, 384 if model_id.startswith("swinir") else 512)
            h, w = rgb.shape[:2]
            out = np.zeros((h * scale, w * scale, 3), np.float32)
            weight = np.zeros((h * scale, w * scale, 1), np.float32)
            ov = self.overlap if (h > tile or w > tile) else 0
            step = max(16, tile - 2 * ov)
            ys = list(range(0, max(1, h - 2 * ov), step)) if h > tile else [0]
            xs = list(range(0, max(1, w - 2 * ov), step)) if w > tile else [0]
            total = len(ys) * len(xs)
            n = 0
            with torch.inference_mode():
                for ty in ys:
                    for tx in xs:
                        if cancel and cancel():
                            raise CancelledError()
                        y0, x0 = ty, tx
                        y1, x1 = min(h, ty + tile), min(w, tx + tile)
                        if h > tile and y1 - y0 < tile:
                            y0 = max(0, y1 - tile)
                        if w > tile and x1 - x0 < tile:
                            x0 = max(0, x1 - tile)
                        patch = rgb[y0:y1, x0:x1]
                        ph, pw = patch.shape[:2]
                        th = max(minimum, -(-ph // mult) * mult)
                        tw = max(minimum, -(-pw // mult) * mult)
                        if (th, tw) != (ph, pw):
                            patch = np.pad(patch, ((0, th - ph), (0, tw - pw), (0, 0)), mode="reflect")
                        x = torch.from_numpy(np.ascontiguousarray(patch.transpose(2, 0, 1)))[None].to(self.device)
                        if half:
                            x = x.half()
                        yt = self._forward(desc, x, model_id, params)
                        y = yt[0].float().clamp_(0, 1).cpu().numpy().transpose(1, 2, 0)
                        y = y[: ph * scale, : pw * scale]
                        wy = _ramp(ph, ov if y0 > 0 else 0, ov if y1 < h else 0)
                        wx = _ramp(pw, ov if x0 > 0 else 0, ov if x1 < w else 0)
                        wgt = np.outer(np.repeat(wy, scale), np.repeat(wx, scale))[..., None].astype(np.float32)
                        out[y0 * scale:y1 * scale, x0 * scale:x1 * scale] += y * wgt
                        weight[y0 * scale:y1 * scale, x0 * scale:x1 * scale] += wgt
                        n += 1
                        if progress:
                            progress(n / total)
            out /= np.maximum(weight, 1e-6)
            return np.clip(out, 0, 1)

    def peak_vram_mb(self) -> float | None:
        if self.dev_info.kind != "cuda":
            return None
        return self.torch.cuda.max_memory_allocated() / 2**20

    def reset_peak(self) -> None:
        if self.dev_info.kind == "cuda":
            self.torch.cuda.reset_peak_memory_stats()


def _ramp(n: int, lead: int, trail: int) -> np.ndarray:
    w = np.ones(n, np.float32)
    if lead > 0:
        w[:lead] = np.linspace(0.02, 1.0, lead, dtype=np.float32)
    if trail > 0:
        w[n - trail:] = np.minimum(w[n - trail:], np.linspace(1.0, 0.02, trail, dtype=np.float32))
    return w


_BACKEND: DirectBackend | None = None
_BACKEND_LOCK = threading.Lock()


def get_backend(**kwargs) -> DirectBackend:
    global _BACKEND
    with _BACKEND_LOCK:
        if _BACKEND is None:
            _BACKEND = DirectBackend(**kwargs)
        return _BACKEND


def reset_backend() -> None:
    global _BACKEND
    with _BACKEND_LOCK:
        if _BACKEND is not None:
            _BACKEND.unload_all()
        _BACKEND = None

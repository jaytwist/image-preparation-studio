"""Subject/person masks via local ONNX segmentation models + guided-filter edge refinement."""
from __future__ import annotations

import threading

import cv2
import numpy as np

from ..backends.registry import MODELS
from ..core import ops as cops

_SPECS = {
    "u2net_human_seg": {"size": 320, "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
    "isnet_general": {"size": 1024, "mean": (0.5, 0.5, 0.5), "std": (1.0, 1.0, 1.0)},
}


class Matting:
    _sessions: dict[str, object] = {}
    _lock = threading.Lock()

    @classmethod
    def session(cls, model_id: str):
        with cls._lock:
            if model_id not in cls._sessions:
                import onnxruntime as ort
                from ..privacy import disable_runtime_telemetry
                disable_runtime_telemetry()
                info = MODELS[model_id]
                if not info.installed:
                    raise FileNotFoundError(f"Matting model missing: {info.file}")
                from ..core.identity import _providers
                so = ort.SessionOptions()
                so.log_severity_level = 3
                cls._sessions[model_id] = ort.InferenceSession(str(info.path), so, providers=_providers())
            return cls._sessions[model_id]

    @classmethod
    def coarse(cls, rgb: np.ndarray, model_id: str = "u2net_human_seg") -> np.ndarray:
        spec = _SPECS[model_id]
        s = spec["size"]
        sess = cls.session(model_id)
        h, w = rgb.shape[:2]
        x = cv2.resize(rgb, (s, s), interpolation=cv2.INTER_AREA if max(h, w) > s else cv2.INTER_CUBIC)
        x = x / max(float(x.max()), 1e-6)
        x = (x - np.array(spec["mean"], np.float32)) / np.array(spec["std"], np.float32)
        blob = x.transpose(2, 0, 1)[None].astype(np.float32)
        out = sess.run(None, {sess.get_inputs()[0].name: blob})[0][0, 0]
        out = (out - out.min()) / max(1e-6, float(out.max() - out.min()))
        return np.clip(cv2.resize(out.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC), 0, 1)

    @classmethod
    def refine(cls, rgb: np.ndarray, alpha: np.ndarray, radius: int | None = None, eps: float = 1e-4) -> np.ndarray:
        """Guided-filter refinement: snaps soft edges (hair) to image structure."""
        h, w = alpha.shape
        r = radius or max(4, int(min(h, w) / 160))
        guide = cops.luma(rgb)
        ref = cops.guided_filter(guide, alpha, r=r, eps=eps)
        # keep confident interior/exterior, refine only the uncertain band
        band = (alpha > 0.03) & (alpha < 0.97)
        band = cv2.dilate(band.astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0
        outa = np.where(band, ref, alpha)
        return np.clip(outa, 0, 1).astype(np.float32)

    @classmethod
    def person_mask(cls, rgb: np.ndarray, model_id: str = "u2net_human_seg", refine: bool = True) -> np.ndarray:
        a = cls.coarse(rgb, model_id)
        return cls.refine(rgb, a) if refine else a


def decontaminate(rgb: np.ndarray, alpha: np.ndarray, radius: int = 6) -> np.ndarray:
    """Reduce background colour fringing in semi-transparent edge pixels.

    Edge colours are pulled toward a blur of confidently-foreground colours (normalised convolution).
    """
    fg = (alpha > 0.95).astype(np.float32)
    k = (0, 0)
    num = cv2.GaussianBlur(rgb * fg[..., None], k, radius)
    den = cv2.GaussianBlur(fg, k, radius)[..., None]
    est = num / np.maximum(den, 1e-4)
    edge = ((alpha > 0.02) & (alpha < 0.95)).astype(np.float32)[..., None]
    w = edge * (1 - alpha[..., None]) * (den > 1e-3)
    return np.clip(rgb * (1 - w) + est * w, 0, 1).astype(np.float32)


def adjust_edge(alpha: np.ndarray, feather_px: float = 0.0, shift_px: int = 0) -> np.ndarray:
    a = alpha.astype(np.float32)
    if shift_px:
        k = np.ones((2 * abs(shift_px) + 1, 2 * abs(shift_px) + 1), np.uint8)
        a = cv2.dilate(a, k) if shift_px > 0 else cv2.erode(a, k)
    if feather_px > 0:
        a = cops.feather(a, feather_px)
    return np.clip(a, 0, 1)

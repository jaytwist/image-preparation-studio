"""Mask document (brush/magic select/undo) and the edit operations themselves."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..core import ops as cops


@dataclass
class MaskDoc:
    """Editable selection mask with undo/redo (float32 HxW in [0,1])."""
    h: int
    w: int
    mask: np.ndarray = None  # type: ignore[assignment]
    _undo: list[np.ndarray] = field(default_factory=list)
    _redo: list[np.ndarray] = field(default_factory=list)
    max_history: int = 40

    def __post_init__(self):
        if self.mask is None:
            self.mask = np.zeros((self.h, self.w), np.float32)

    def _push(self):
        self._undo.append(self.mask.copy())
        if len(self._undo) > self.max_history:
            self._undo.pop(0)
        self._redo.clear()

    def set(self, m: np.ndarray):
        self._push()
        self.mask = np.clip(m.astype(np.float32), 0, 1)

    def begin_stroke(self):
        self._push()

    def brush(self, x: float, y: float, radius: float, add: bool = True, hardness: float = 0.7,
              x2: float | None = None, y2: float | None = None):
        """Paint a dab (or a segment to x2,y2) without pushing history (call begin_stroke first)."""
        r = max(1, int(radius))
        stamp = np.zeros_like(self.mask)
        if x2 is None:
            cv2.circle(stamp, (int(x), int(y)), r, 1.0, -1)
        else:
            cv2.line(stamp, (int(x), int(y)), (int(x2), int(y2)), 1.0, 2 * r)
        if hardness < 1:
            stamp = cops.feather(stamp, r * (1 - hardness) * 0.6)
        self.mask = np.maximum(self.mask, stamp) if add else np.minimum(self.mask, 1 - stamp)

    def magic_select(self, rgb: np.ndarray, x: int, y: int, tolerance: float = 12.0, contiguous: bool = True,
                     add: bool = True):
        """Select pixels similar in Lab colour to the seed (ΔE < tolerance)."""
        self._push()
        lab = cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2LAB)
        seed = lab[int(np.clip(y, 0, self.h - 1)), int(np.clip(x, 0, self.w - 1))]
        # chroma/hue-weighted distance so shading (L) varies more freely than colour – fabric folds
        d = np.sqrt(0.35 * (lab[..., 0] - seed[0]) ** 2 + (lab[..., 1] - seed[1]) ** 2 + (lab[..., 2] - seed[2]) ** 2)
        sel = (d < tolerance).astype(np.uint8)
        if contiguous:
            n, lbl = cv2.connectedComponents(sel, connectivity=8)
            k = lbl[int(y), int(x)]
            sel = (lbl == k).astype(np.uint8) if k > 0 else np.zeros_like(sel)
            sel = cv2.morphologyEx(sel, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        m = sel.astype(np.float32)
        self.mask = np.maximum(self.mask, m) if add else np.minimum(self.mask, 1 - m)

    def feather(self, px: float):
        self._push()
        self.mask = cops.feather(self.mask, px)

    def grow(self, px: int):
        self._push()
        k = np.ones((2 * abs(px) + 1, 2 * abs(px) + 1), np.uint8)
        self.mask = cv2.dilate(self.mask, k) if px > 0 else cv2.erode(self.mask, k)

    def invert(self):
        self._push()
        self.mask = 1 - self.mask

    def clear(self):
        self._push()
        self.mask = np.zeros_like(self.mask)

    def undo(self) -> bool:
        if not self._undo:
            return False
        self._redo.append(self.mask.copy())
        self.mask = self._undo.pop()
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        self._undo.append(self.mask.copy())
        self.mask = self._redo.pop()
        return True


# ----------------------------------------------------------------------------- recolor

def recolor(rgb: np.ndarray, mask: np.ndarray, target_rgb: tuple[float, float, float], strength: float = 1.0,
            match_lightness: float = 0.0) -> np.ndarray:
    """Recolour the masked region to *target_rgb* (0..1) in CIE LCh.

    Hue and chroma move to the target; per-pixel lightness *variation* (folds, shading, weave and
    printed patterns) is preserved. ``match_lightness`` (0..1) shifts the mean lightness toward the
    target colour while keeping the variation.
    """
    lab = cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2LAB)
    t = cv2.cvtColor(np.array([[target_rgb]], np.float32), cv2.COLOR_RGB2LAB)[0, 0]
    m = np.clip(mask, 0, 1).astype(np.float32)
    msum = float(m.sum()) + 1e-6
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    c_src = np.hypot(a, b)
    c_mean = float((c_src * m).sum() / msum)
    c_t = float(np.hypot(t[1], t[2]))
    h_t = float(np.arctan2(t[2], t[1]))
    # chroma: keep relative chroma variation (pattern contrast) but move its level to the target
    rel = c_src / max(c_mean, 1.0)
    c_new = np.clip(rel * c_t, 0, 128)
    a_new, b_new = c_new * np.cos(h_t), c_new * np.sin(h_t)
    L_new = L
    if match_lightness > 0:
        l_mean = float((L * m).sum() / msum)
        L_new = np.clip(L + match_lightness * (t[0] - l_mean), 0, 100)
    lab2 = np.dstack([L_new, a_new, b_new]).astype(np.float32)
    out = np.clip(cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB), 0, 1)
    w = m * float(strength)
    res = cops.blend(rgb, out, w)
    # pixels outside the (visible) selection stay bit-identical
    return np.where((w > 1.0 / 1024)[..., None], res, rgb).astype(np.float32)


# ----------------------------------------------------------------------------- remove / inpaint

def remove_object(rgb: np.ndarray, mask: np.ndarray, method: str = "lama") -> tuple[np.ndarray, str]:
    """Fill the masked region. Returns (image, method actually used)."""
    hard = (mask > 0.5).astype(np.uint8)
    if hard.sum() == 0:
        return rgb, "none (empty mask)"
    hard = cv2.dilate(hard, np.ones((7, 7), np.uint8))
    ys, xs = np.where(hard > 0)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    bh, bw = y1 - y0, x1 - x0
    pad = int(max(bh, bw) * 0.8) + 16
    H, W = rgb.shape[:2]
    cy0, cy1, cx0, cx1 = max(0, y0 - pad), min(H, y1 + pad), max(0, x0 - pad), min(W, x1 + pad)
    crop = rgb[cy0:cy1, cx0:cx1]
    mcrop = hard[cy0:cy1, cx0:cx1]
    used = method
    filled = None
    if method == "lama":
        try:
            filled = _lama(crop, mcrop)
            used = "LaMa (generative fill)"
        except Exception as e:
            used = f"OpenCV Telea (LaMa unavailable: {type(e).__name__})"
    if filled is None:
        u8 = (crop * 255 + 0.5).astype(np.uint8)
        filled = cv2.inpaint(u8, mcrop * 255, 5, cv2.INPAINT_TELEA).astype(np.float32) / 255.0
        if method != "lama":
            used = "OpenCV Telea (classical fill)"
    soft = cops.feather(mcrop.astype(np.float32), 2.0)
    out = rgb.copy()
    out[cy0:cy1, cx0:cx1] = cops.blend(crop, filled, soft)
    return out, used


def _lama(crop: np.ndarray, mask: np.ndarray) -> np.ndarray:
    import torch

    from ..backends.direct import GPU_LOCK, get_backend
    be = get_backend()
    desc = be.get("lama")
    h, w = crop.shape[:2]
    s = min(1.0, 768 / max(h, w))
    work = cv2.resize(crop, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else crop
    mw = cv2.resize(mask, (work.shape[1], work.shape[0]), interpolation=cv2.INTER_NEAREST) if s < 1 else mask
    ph, pw = (-work.shape[0]) % 8, (-work.shape[1]) % 8
    work_p = np.pad(work, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    m_p = np.pad(mw, ((0, ph), (0, pw)), mode="constant")
    with GPU_LOCK, torch.inference_mode():
        x = torch.from_numpy(work_p.transpose(2, 0, 1).copy())[None].to(be.device)
        m = torch.from_numpy(m_p.astype(np.float32))[None, None].to(be.device)
        y = desc(x, m)[0].float().clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
    y = y[: work.shape[0], : work.shape[1]]
    if s < 1:
        y = cv2.resize(y, (w, h), interpolation=cv2.INTER_CUBIC)
    # keep original pixels outside the mask exactly
    return np.where(mask[..., None] > 0, y, crop).astype(np.float32)


# ----------------------------------------------------------------------------- background

def replace_background(rgb: np.ndarray, alpha: np.ndarray, mode: str = "color",
                       color: tuple[float, float, float] = (1.0, 1.0, 1.0), blur_sigma: float = 18.0,
                       background: np.ndarray | None = None) -> np.ndarray:
    a = np.clip(alpha, 0, 1)[..., None]
    if mode == "blur":
        bg = cv2.GaussianBlur(rgb, (0, 0), blur_sigma)
    elif mode == "image" and background is not None:
        bg = cops.resize_to(background, rgb.shape[1], rgb.shape[0])
    else:
        bg = np.ones_like(rgb) * np.array(color, np.float32)
    return np.clip(rgb * a + bg * (1 - a), 0, 1).astype(np.float32)

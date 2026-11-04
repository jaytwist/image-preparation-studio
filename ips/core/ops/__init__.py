"""Classical (non-neural) image operations. All take/return float32 RGB in [0,1]."""
from __future__ import annotations

import math

import cv2
import numpy as np


# ----------------------------------------------------------------------------- colour helpers

def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


def luma(rgb: np.ndarray) -> np.ndarray:
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)


# ----------------------------------------------------------------------------- white balance

def neutral_cast(rgb: np.ndarray, strength: float = 0.7, gains: list[float] | None = None, **_) -> np.ndarray:
    """Von-Kries style correction in linear light, luminance-preserving.

    With explicit ``gains`` (set by Auto from the consensus illuminant, editable in Manual) they are used
    as-is; otherwise gains are estimated from near-neutral pixels and applied with ``strength``.
    """
    if gains is not None:
        g = np.array(gains, np.float32)
        lin = srgb_to_linear(rgb)
        y0 = 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]
        out = lin * g[None, None, :]
        y1 = 0.2126 * out[..., 0] + 0.7152 * out[..., 1] + 0.0722 * out[..., 2]
        out = out * (y0 / np.maximum(y1, 1e-6))[..., None]
        return linear_to_srgb(out)
    if strength <= 0:
        return rgb
    small = cv2.resize(rgb, (min(512, rgb.shape[1]), min(512, rgb.shape[0])), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB)
    chroma = np.hypot(lab[..., 1], lab[..., 2])
    m = (chroma < 18) & (lab[..., 0] > 15) & (lab[..., 0] < 95)
    lin = srgb_to_linear(small)
    ref = lin[m].mean(axis=0) if m.sum() > 200 else lin.reshape(-1, 3).mean(axis=0)
    ref = np.maximum(ref, 1e-4)
    y = float(0.2126 * ref[0] + 0.7152 * ref[1] + 0.0722 * ref[2])
    gains = (y / ref) ** float(strength)
    gains = np.clip(gains, 0.6, 1.6)
    out = linear_to_srgb(srgb_to_linear(rgb) * gains[None, None, :].astype(np.float32))
    return out


# ----------------------------------------------------------------------------- tone

def levels(rgb: np.ndarray, ev: float = 0.0, black: float = 0.0, white: float = 1.0, gamma: float = 1.0,
           **_) -> np.ndarray:
    out = rgb
    if ev:
        out = linear_to_srgb(srgb_to_linear(out) * (2.0 ** ev))
    if black != 0.0 or white != 1.0:
        out = np.clip((out - black) / max(1e-4, white - black), 0, 1)
    if gamma != 1.0:
        out = np.power(np.clip(out, 0, 1), 1.0 / gamma)
    return out.astype(np.float32)


def contrast_saturation(rgb: np.ndarray, contrast: float = 0.0, saturation: float = 0.0, vibrance: float = 0.0,
                        **_) -> np.ndarray:
    lab = cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2LAB)
    L, a, b = lab[..., 0] / 100.0, lab[..., 1], lab[..., 2]
    if contrast:
        # smooth S-curve around mid grey, endpoints fixed
        k = float(contrast)
        L = L + k * (L - 0.5) * (1 - np.abs(2 * L - 1))
    if saturation or vibrance:
        c = np.hypot(a, b)
        hue = (np.degrees(np.arctan2(b, a)) + 360) % 360
        skin = ((hue > 20) & (hue < 75)).astype(np.float32)  # protect skin tones from vibrance
        scale = 1.0 + float(saturation)
        if vibrance:
            scale = scale + float(vibrance) * (1 - np.clip(c / 60.0, 0, 1)) * (1 - 0.7 * skin)
        a, b = a * scale, b * scale
    out = cv2.cvtColor(np.dstack([np.clip(L, 0, 1) * 100.0, a, b]).astype(np.float32), cv2.COLOR_LAB2RGB)
    return np.clip(out, 0, 1).astype(np.float32)


# ----------------------------------------------------------------------------- detail

def halo_free_unsharp(rgb: np.ndarray, amount: float = 0.35, radius: float = 1.0, threshold: float = 0.01,
                      **_) -> np.ndarray:
    """Luminance unsharp mask with the result clamped to the local 3x3 min/max (no halos/overshoot)."""
    if amount <= 0:
        return rgb
    y = luma(rgb)
    blur = cv2.GaussianBlur(y, (0, 0), max(0.3, float(radius)))
    detail = y - blur
    detail = np.where(np.abs(detail) > threshold, detail, 0.0)
    y2 = y + float(amount) * detail
    k = np.ones((3, 3), np.uint8)
    y2 = np.clip(y2, cv2.erode(y, k), cv2.dilate(y, k))
    out = rgb + (y2 - y)[..., None]
    return np.clip(out, 0, 1).astype(np.float32)


def unsharp_deconv(rgb: np.ndarray, radius: float = 1.5, amount: float = 0.6, **_) -> np.ndarray:
    out = rgb
    for _ in range(2):
        out = halo_free_unsharp(out, amount=amount, radius=radius, threshold=0.004)
    return out


def nlmeans(rgb: np.ndarray, h: float = 6.0, h_color: float = 6.0, **_) -> np.ndarray:
    u8 = (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)
    den = cv2.fastNlMeansDenoisingColored(u8, None, float(h), float(h_color), 7, 21)
    # carry the 8-bit delta onto the float signal to keep 16-bit precision where unchanged
    delta = den.astype(np.float32) / 255.0 - u8.astype(np.float32) / 255.0
    return np.clip(rgb + delta, 0, 1).astype(np.float32)


# ----------------------------------------------------------------------------- resize

def resize_scale(rgb: np.ndarray, scale: float, method: str = "lanczos") -> np.ndarray:
    h, w = rgb.shape[:2]
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    if method == "nearest":
        interp = cv2.INTER_NEAREST
    elif scale < 1:
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_LANCZOS4
    return np.clip(cv2.resize(rgb, (nw, nh), interpolation=interp), 0, 1).astype(np.float32)


def resize_to(img: np.ndarray, w: int, h: int, method: str = "auto") -> np.ndarray:
    if img.shape[1] == w and img.shape[0] == h:
        return img
    if method == "nearest":
        interp = cv2.INTER_NEAREST
    elif w < img.shape[1]:
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_LANCZOS4
    out = cv2.resize(img, (int(w), int(h)), interpolation=interp)
    return np.clip(out, 0, 1).astype(np.float32)


# ----------------------------------------------------------------------------- masks

def face_mask(shape: tuple[int, int], boxes: list[tuple[float, float, float, float]], expand: float = 1.35,
              scale: float = 1.0) -> np.ndarray:
    h, w = shape
    m = np.zeros((h, w), np.float32)
    for x0, y0, x1, y1 in boxes:
        x0, y0, x1, y1 = x0 * scale, y0 * scale, x1 * scale, y1 * scale
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        ax, ay = (x1 - x0) / 2 * expand, (y1 - y0) / 2 * expand * 1.15
        cv2.ellipse(m, (int(cx), int(cy)), (max(1, int(ax)), max(1, int(ay))), 0, 0, 360, 1.0, -1)
        sigma = max(1.0, 0.12 * min(ax, ay))
        m = np.maximum(m, 0)
    if m.any():
        sig = max(1.0, 0.06 * math.sqrt(m.sum() / max(1, len(boxes))))
        m = cv2.GaussianBlur(m, (0, 0), sig)
        m = np.clip(m / max(1e-6, m.max()), 0, 1)
    return m


def text_mask(rgb: np.ndarray, dilate: int = 2) -> np.ndarray:
    """Mask of sharp, high-contrast fine structure (text, icons, UI lines) at native resolution."""
    y = (np.clip(luma(rgb), 0, 1) * 255).astype(np.uint8)
    grad = cv2.morphologyEx(y, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    _, bw = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    bw = cv2.dilate(bw, np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8))
    m = bw.astype(np.float32) / 255.0
    return cv2.GaussianBlur(m, (0, 0), 1.0)


def feather(mask: np.ndarray, px: float) -> np.ndarray:
    if px <= 0:
        return mask.astype(np.float32)
    return np.clip(cv2.GaussianBlur(mask.astype(np.float32), (0, 0), float(px)), 0, 1)


def blend(a: np.ndarray, b: np.ndarray, m: np.ndarray | float) -> np.ndarray:
    """a*(1-m) + b*m with broadcasting for HxW masks."""
    if isinstance(m, np.ndarray) and m.ndim == 2:
        m = m[..., None]
    return (a * (1 - m) + b * m).astype(np.float32)


def guided_filter(guide: np.ndarray, src: np.ndarray, r: int = 8, eps: float = 1e-4) -> np.ndarray:
    """He et al. guided filter (gray guide). Used for alpha matte edge refinement."""
    I = guide.astype(np.float32)
    p = src.astype(np.float32)
    k = (2 * r + 1, 2 * r + 1)
    mI = cv2.boxFilter(I, -1, k)
    mp = cv2.boxFilter(p, -1, k)
    cIp = cv2.boxFilter(I * p, -1, k) - mI * mp
    vI = cv2.boxFilter(I * I, -1, k) - mI * mI
    a = cIp / (vI + eps)
    b = mp - a * mI
    return (cv2.boxFilter(a, -1, k) * I + cv2.boxFilter(b, -1, k)).astype(np.float32)


# ----------------------------------------------------------------------------- geometry

def largest_inscribed(w: int, h: int, angle_deg: float) -> tuple[int, int]:
    """Largest axis-aligned rectangle inside a w×h rectangle rotated by angle."""
    if w <= 0 or h <= 0:
        return 0, 0
    a = math.radians(abs(angle_deg) % 180)
    if a > math.pi / 2:
        a = math.pi - a
    sin_a, cos_a = math.sin(a), math.cos(a)
    width_is_longer = w >= h
    side_long, side_short = (w, h) if width_is_longer else (h, w)
    if side_short <= 2.0 * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
        x = 0.5 * side_short
        wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
    else:
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr, hr = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a
    return int(wr), int(hr)


def crop_rotate(img: np.ndarray, angle: float = 0.0, crop: list[int] | None = None, rotate90: int = 0,
                flip: bool = False, autocrop: bool = True, **_) -> np.ndarray:
    out = img
    if rotate90:
        out = np.ascontiguousarray(np.rot90(out, -int(rotate90) % 4))
    if flip:
        out = np.ascontiguousarray(out[:, ::-1])
    if angle:
        h, w = out.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), -float(angle), 1.0)
        out = cv2.warpAffine(out, M, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
        if autocrop:
            cw, ch = largest_inscribed(w, h, angle)
            x0, y0 = (w - cw) // 2, (h - ch) // 2
            out = out[y0:y0 + ch, x0:x0 + cw]
        out = np.clip(out, 0, 1).astype(np.float32)
    if crop:
        x0, y0, x1, y1 = [int(v) for v in crop]
        h, w = out.shape[:2]
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        if x1 > x0 and y1 > y0:
            out = out[y0:y1, x0:x1]
    return np.ascontiguousarray(out)


def estimate_straighten_angle(rgb: np.ndarray, max_angle: float = 10.0) -> float | None:
    """Dominant deviation of near-horizontal/vertical lines (degrees, + = rotate clockwise)."""
    y = (np.clip(luma(rgb), 0, 1) * 255).astype(np.uint8)
    s = 1024 / max(y.shape)
    if s < 1:
        y = cv2.resize(y, (int(y.shape[1] * s), int(y.shape[0] * s)), interpolation=cv2.INTER_AREA)
    edges = cv2.Canny(y, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=80, minLineLength=max(40, min(y.shape) // 6),
                            maxLineGap=8)
    if lines is None:
        return None
    devs, weights = [], []
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        ang = math.degrees(math.atan2(y2 - y1, x2 - x1))
        for ref in (0, 90, -90, 180, -180):
            d = ang - ref
            if abs(d) <= max_angle:
                devs.append(d if ref in (0, 180, -180) else -d)
                weights.append(math.hypot(x2 - x1, y2 - y1))
                break
    if len(devs) < 3:
        return None
    order = np.argsort(devs)
    d, wts = np.array(devs)[order], np.array(weights)[order]
    c = np.cumsum(wts)
    med = float(d[np.searchsorted(c, c[-1] / 2)])
    return round(med, 2) if abs(med) >= 0.1 else 0.0

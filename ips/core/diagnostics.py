"""Pixel diagnostics. Fast (designed for < ~1 s on a 12 MP photo on CPU).

Global statistics use an analysis copy (long side 1024). Resolution-dependent measures (noise,
JPEG blocking, blur, halos) use native-resolution crops so they are not hidden by downscaling.
All values are measurements, not judgements; ``classify`` and ``router`` interpret them.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import cv2
import numpy as np

ANALYSIS_LONG_SIDE = 1024
CROP = 512


@dataclass
class Face:
    box: tuple[float, float, float, float]          # x0, y0, x1, y1 in source pixels
    score: float
    landmarks: list[tuple[float, float]] | None = None  # 5 points (eyes, nose, mouth corners)

    @property
    def size(self) -> float:
        return float(min(self.box[2] - self.box[0], self.box[3] - self.box[1]))


@dataclass
class Diagnostics:
    width: int
    height: int
    megapixels: float
    bit_depth: int
    has_alpha: bool
    features: dict[str, float] = field(default_factory=dict)
    faces: list[Face] = field(default_factory=list)
    face_detector: str = "none"
    timings_ms: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["faces"] = [asdict(f) for f in self.faces]
        return d


# ----------------------------------------------------------------------------- helpers

def _luma(rgb: np.ndarray) -> np.ndarray:
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)


def _analysis_copy(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    s = ANALYSIS_LONG_SIDE / max(h, w)
    if s >= 1:
        return rgb
    return cv2.resize(rgb, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)


def native_crops(img: np.ndarray, n: int = 4, size: int = CROP) -> list[np.ndarray]:
    """Up to n crops at native resolution; offsets on the 8-px grid so JPEG phase is preserved."""
    h, w = img.shape[:2]
    if h <= size or w <= size:
        return [img]
    ys = [int((h - size) * f) // 8 * 8 for f in (0.3, 0.6)]
    xs = [int((w - size) * f) // 8 * 8 for f in (0.3, 0.6)]
    crops = [img[y:y + size, x:x + size] for y in ys for x in xs]
    return crops[:n]


def _texture_mask(y: np.ndarray, q: float = 50) -> np.ndarray:
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    thr = np.percentile(mag, q)
    return mag <= max(thr, 1e-6)


# ----------------------------------------------------------------------------- measures

_IMMERKAER = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)


def noise_sigma(channel: np.ndarray) -> float:
    """Immerkær fast noise estimate on flat pixels, in 0..255 units."""
    c = channel.astype(np.float32) * 255.0
    if c.shape[0] < 8 or c.shape[1] < 8:
        return 0.0
    conv = np.abs(cv2.filter2D(c, -1, _IMMERKAER, borderType=cv2.BORDER_REFLECT))[1:-1, 1:-1]
    flat = _texture_mask(c, 40)[1:-1, 1:-1]
    vals = conv[flat]
    if vals.size < 64:
        vals = conv.ravel()
    return float(math.sqrt(math.pi / 2.0) * np.mean(vals) / 6.0)


def blockiness(y: np.ndarray) -> float:
    """Phase-robust 8x8 grid discontinuity. ~0 for clean images, > 0.25 visible blocking."""
    c = y.astype(np.float32) * 255.0
    if min(c.shape) < 32:
        return 0.0
    dh = np.abs(np.diff(c, axis=1))
    dv = np.abs(np.diff(c, axis=0))
    smooth_h = dh < 20  # ignore real edges
    smooth_v = dv < 20

    def phase_score(d, m, axis):
        means = []
        for p in range(8):
            if axis == 1:
                sl, ms = d[:, p::8], m[:, p::8]
            else:
                sl, ms = d[p::8, :], m[p::8, :]
            v = sl[ms]
            means.append(float(v.mean()) if v.size else 0.0)
        means = np.array(means)
        med = float(np.median(means)) + 1e-3
        return float((means.max() - med) / med)

    return 0.5 * (phase_score(dh, smooth_h, 1) + phase_score(dv, smooth_v, 0))


def crete_blur(y: np.ndarray) -> float:
    """Crete et al. 2007 no-reference blur metric in [0,1]; higher = blurrier."""
    f = y.astype(np.float32)
    if min(f.shape) < 16:
        return 0.0
    k = np.ones((1, 9), np.float32) / 9.0
    bv = cv2.filter2D(f, -1, k.T)
    bh = cv2.filter2D(f, -1, k)
    d_fv = np.abs(np.diff(f, axis=0))
    d_fh = np.abs(np.diff(f, axis=1))
    d_bv = np.abs(np.diff(bv, axis=0))
    d_bh = np.abs(np.diff(bh, axis=1))
    vv = np.maximum(0, d_fv - d_bv)
    vh = np.maximum(0, d_fh - d_bh)
    s_fv, s_fh = d_fv.sum() + 1e-6, d_fh.sum() + 1e-6
    b_v = (s_fv - vv.sum()) / s_fv
    b_h = (s_fh - vh.sum()) / s_fh
    return float(max(b_v, b_h))


def motion_anisotropy(y: np.ndarray) -> float:
    """Structure-tensor coherence of gradients over the whole crop (0 isotropic .. 1 one direction)."""
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
    jxx, jyy, jxy = float((gx * gx).mean()), float((gy * gy).mean()), float((gx * gy).mean())
    tr = jxx + jyy + 1e-9
    disc = math.sqrt(max(0.0, (jxx - jyy) ** 2 + 4 * jxy * jxy))
    return float(disc / tr)


def halo_index(y: np.ndarray) -> float:
    """Overshoot around strong edges relative to edge contrast (oversharpening / ringing)."""
    u8 = np.clip(y * 255, 0, 255).astype(np.uint8)
    edges = cv2.Canny(u8, 60, 160)
    if edges.mean() < 0.002 * 255:
        return 0.0
    band = cv2.dilate(edges, np.ones((5, 5), np.uint8)) > 0
    f = cv2.medianBlur(u8, 3).astype(np.float32) / 255.0
    base = cv2.GaussianBlur(f, (0, 0), 2.0)
    hi = cv2.dilate(base, np.ones((7, 7), np.uint8))
    lo = cv2.erode(base, np.ones((7, 7), np.uint8))
    over = np.maximum(f - hi, 0) + np.maximum(lo - f, 0)
    contrast = (hi - lo)[band].mean() + 1e-4
    return float(over[band].mean() / contrast)


def texture_ratio(y: np.ndarray) -> float:
    """Fine-detail energy vs. mid-frequency energy in non-edge areas (low = waxy/oversmoothed)."""
    f = y.astype(np.float32)
    lap = np.abs(cv2.Laplacian(f, cv2.CV_32F, ksize=1))
    dog = np.abs(cv2.GaussianBlur(f, (0, 0), 1.0) - cv2.GaussianBlur(f, (0, 0), 3.0))
    u8 = np.clip(f * 255, 0, 255).astype(np.uint8)
    edges = cv2.dilate(cv2.Canny(u8, 50, 150), np.ones((5, 5), np.uint8)) > 0
    m = ~edges
    if m.sum() < 100:
        m = np.ones_like(edges)
    return float(lap[m].mean() / (dog[m].mean() + 1e-5))


def screenshot_features(rgb_native: np.ndarray) -> dict[str, float]:
    """Flat-region fraction, axis-aligned edge share and colour-count, computed at native resolution."""
    crops = native_crops(rgb_native, n=4, size=384)
    flat_fr, axis_fr, uniq = [], [], []
    for c in crops:
        u8 = (np.clip(c, 0, 1) * 255 + 0.5).astype(np.uint8)
        g = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY).astype(np.int16)
        mx = cv2.dilate(g.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.int16)
        mn = cv2.erode(g.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.int16)
        flat_fr.append(float(((mx - mn) == 0).mean()))
        gx = cv2.Sobel(g.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        mag = np.abs(gx) + np.abs(gy)
        strong = mag > 80
        if strong.sum() > 20:
            ax = (np.minimum(np.abs(gx), np.abs(gy)) < 0.08 * np.maximum(np.abs(gx), np.abs(gy)))
            axis_fr.append(float(ax[strong].mean()))
        else:
            axis_fr.append(0.0)
        q = (u8[::2, ::2] // 4).reshape(-1, 3)
        keys = q[:, 0].astype(np.int32) * 4096 + q[:, 1].astype(np.int32) * 64 + q[:, 2]
        uniq.append(float(np.unique(keys).size) / max(1, keys.size))
    return {"flat_fraction": float(np.mean(flat_fr)), "axis_edge_fraction": float(np.mean(axis_fr)),
            "unique_color_ratio": float(np.mean(uniq))}


def text_density(y_analysis: np.ndarray) -> float:
    """Approximate density of small high-contrast glyph-like components (0..1)."""
    u8 = np.clip(y_analysis * 255, 0, 255).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    grad = cv2.morphologyEx(u8, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    _, bw = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k)
    n, _, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if n <= 1:
        return 0.0
    h = stats[1:, cv2.CC_STAT_HEIGHT]
    w = stats[1:, cv2.CC_STAT_WIDTH]
    area = stats[1:, cv2.CC_STAT_AREA]
    fill = area / np.maximum(1, w * h)
    glyphish = (h >= 5) & (h <= 40) & (w >= 2 * h) & (fill > 0.35)
    covered = float((w[glyphish] * h[glyphish]).sum()) / float(u8.size)
    return float(min(1.0, covered * 3.0))


def exposure_stats(y: np.ndarray) -> dict[str, float]:
    p1, p5, p50, p95, p99 = np.percentile(y, [1, 5, 50, 95, 99])
    return {"luma_mean": float(y.mean()), "luma_p1": float(p1), "luma_p5": float(p5), "luma_p50": float(p50),
            "luma_p95": float(p95), "luma_p99": float(p99),
            "clip_low": float((y <= 2 / 255).mean()), "clip_high": float((y >= 253 / 255).mean()),
            "dynamic_range": float(p99 - p1)}


def color_stats(rgb: np.ndarray) -> dict[str, float]:
    lab = cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2LAB)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    chroma = np.sqrt(a * a + b * b)
    neutral = (chroma < 18) & (L > 15) & (L < 95)
    frac = float(neutral.mean())
    if neutral.sum() > 200:
        ca, cb = float(a[neutral].mean()), float(b[neutral].mean())
    else:
        ca, cb = float(a.mean()) * 0.5, float(b.mean()) * 0.5
    hue = math.degrees(math.atan2(cb, ca)) % 360
    return {"ab_std": float(np.sqrt(a.var() + b.var())), "cast_a": ca, "cast_b": cb, "cast_magnitude": float(math.hypot(ca, cb)), "cast_hue_deg": hue,
            "neutral_fraction": frac, "chroma_mean": float(chroma.mean()),
            "chroma_p95": float(np.percentile(chroma, 95))}


def _srgb_lin(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def illuminant_estimates(rgb: np.ndarray) -> dict[str, float]:
    """Three independent illuminant estimates in log-chromaticity (log R/G, log B/G), linear light.

    gray-world on near-neutral pixels, gray-edge (p=6, σ=2; van de Weijer et al. 2007) and white-patch
    (top-3% luminance, unclipped). Agreement between them is required before Auto corrects a cast.
    Also reports how much of the scene is saturated in the cast's hue (a warm subject, not a warm light).
    """
    lin = _srgb_lin(rgb)
    out: dict[str, float] = {}

    def lc(e):
        e = np.maximum(np.asarray(e, np.float64), 1e-5)
        return float(np.log(e[0] / e[1])), float(np.log(e[2] / e[1]))
    lab = cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2LAB)
    chroma = np.hypot(lab[..., 1], lab[..., 2])
    neutral = (chroma < 12) & (lab[..., 0] > 15) & (lab[..., 0] < 95)
    gw = lin[neutral].mean(0) if neutral.sum() > 200 else lin.reshape(-1, 3).mean(0)
    ge = []
    for c in range(3):
        ch = cv2.GaussianBlur(lin[..., c], (0, 0), 2.0)
        gx, gy = cv2.Sobel(ch, cv2.CV_32F, 1, 0), cv2.Sobel(ch, cv2.CV_32F, 0, 1)
        ge.append(float((np.sqrt(gx * gx + gy * gy) ** 6).mean() ** (1 / 6)))
    y = lin.mean(2)
    thr = np.percentile(y, 97)
    m = (y >= thr) & (lin.max(2) < 0.98)
    wp = lin[m].mean(0) if m.sum() > 50 else lin.reshape(-1, 3).mean(0)
    vecs = {"gw": lc(gw), "ge": lc(ge), "wp": lc(wp)}
    for k, (r, b) in vecs.items():
        out[f"illum_{k}_r"], out[f"illum_{k}_b"] = round(r, 4), round(b, 4)
    V = np.array(list(vecs.values()))
    mags = np.linalg.norm(V, axis=1)
    unit = V / np.maximum(mags[:, None], 1e-6)
    cos = [float(unit[i] @ unit[j]) for i in range(3) for j in range(i + 1, 3)]
    out["illum_agreement"] = round(min(cos), 3)
    out["illum_min_mag"] = round(float(mags.min()), 4)
    cons = V.mean(0)
    out["illum_cons_r"], out["illum_cons_b"] = round(float(cons[0]), 4), round(float(cons[1]), 4)
    # dominance of saturated scene colour in the cast hue
    hue = np.degrees(np.arctan2(lab[..., 2], lab[..., 1])) % 360
    e = np.array([np.exp(cons[0]), 1.0, np.exp(cons[1])], np.float32)
    e = e / e.max() * 0.6
    e_srgb = np.where(e <= 0.0031308, e * 12.92, 1.055 * e ** (1 / 2.4) - 0.055).astype(np.float32)
    lab_e = cv2.cvtColor(e_srgb[None, None, :], cv2.COLOR_RGB2LAB)[0, 0]
    cast_hue = float(np.degrees(np.arctan2(lab_e[2], lab_e[1])) % 360)
    out["illum_hue_deg"] = round(cast_hue, 1)
    dh = np.abs(((hue - cast_hue) + 180) % 360 - 180)
    out["scene_hue_dominance"] = round(float(((chroma > 25) & (dh < 35)).mean()), 3)
    return out


def face_texture(y_native: np.ndarray, faces: list[Face]) -> float | None:
    vals = []
    for f in faces[:4]:
        x0, y0, x1, y1 = [int(round(v)) for v in f.box]
        x0, y0 = max(0, x0), max(0, y0)
        crop = y_native[y0:y1, x0:x1]
        if crop.size < 400:
            continue
        crop = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_AREA)
        vals.append(float(np.abs(cv2.Laplacian(crop, cv2.CV_32F, ksize=1)).mean() * 255))
    return float(np.mean(vals)) if vals else None


# ----------------------------------------------------------------------------- entry

def diagnose(rgb: np.ndarray, *, bit_depth: int = 8, has_alpha: bool = False,
             face_detector: Callable[[np.ndarray], list[Face]] | None = None,
             face_detector_name: str = "none") -> Diagnostics:
    import time
    t0 = time.perf_counter()
    h, w = rgb.shape[:2]
    d = Diagnostics(width=w, height=h, megapixels=round(w * h / 1e6, 3), bit_depth=bit_depth, has_alpha=has_alpha)
    ana = _analysis_copy(rgb)
    ya = _luma(ana)
    feats: dict[str, float] = {}
    feats.update(exposure_stats(ya))
    feats.update(color_stats(ana))
    feats.update(illuminant_estimates(ana))
    t1 = time.perf_counter()

    y_native = _luma(rgb)
    crops = native_crops(y_native)
    ycc = cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2YCrCb) if max(h, w) <= CROP else None
    sig_l, sig_c, blk, blur, aniso, halo, tex = [], [], [], [], [], [], []
    rgb_crops = native_crops(rgb)
    for yc, rc in zip(crops, rgb_crops):
        sig_l.append(noise_sigma(yc))
        cc = cv2.cvtColor(rc.astype(np.float32), cv2.COLOR_RGB2YCrCb) if ycc is None else ycc
        sig_c.append(0.5 * (noise_sigma(cc[..., 1]) + noise_sigma(cc[..., 2])))
        blk.append(blockiness(yc))
        blur.append(crete_blur(yc))
        aniso.append(motion_anisotropy(yc))
        halo.append(halo_index(yc))
        tex.append(texture_ratio(yc))
    feats["noise_sigma_luma"] = float(np.median(sig_l))
    feats["noise_sigma_chroma"] = float(np.median(sig_c))
    feats["blockiness"] = float(np.median(blk))
    feats["blur_crete"] = float(np.median(blur))
    feats["blur_crete_min"] = float(np.min(blur))
    feats["gradient_anisotropy"] = float(np.median(aniso))
    feats["halo_index"] = float(np.median(halo))
    feats["texture_ratio"] = float(np.median(tex))
    feats["laplacian_var"] = float(cv2.Laplacian(ya, cv2.CV_32F).var() * 255 * 255)
    t2 = time.perf_counter()
    feats.update(screenshot_features(rgb))
    feats["text_density"] = text_density(ya)
    feats["dims_multiple_of_64"] = float(w % 64 == 0 and h % 64 == 0)
    feats["dims_multiple_of_8"] = float(w % 8 == 0 and h % 8 == 0)
    t3 = time.perf_counter()
    if face_detector is not None:
        try:
            d.faces = face_detector(rgb)
            d.face_detector = face_detector_name
        except Exception as e:  # never let a detector failure break diagnosis
            d.face_detector = f"{face_detector_name} failed: {e}"
    feats["face_count"] = float(len(d.faces))
    feats["min_face_px"] = float(min((f.size for f in d.faces), default=0.0))
    feats["face_area_fraction"] = float(sum((f.box[2] - f.box[0]) * (f.box[3] - f.box[1]) for f in d.faces)
                                        / max(1, w * h))
    ft = face_texture(y_native, d.faces)
    if ft is not None:
        feats["face_texture"] = ft
    t4 = time.perf_counter()
    d.features = {k: round(float(v), 5) for k, v in feats.items()}
    d.timings_ms = {"global": (t1 - t0) * 1e3, "native": (t2 - t1) * 1e3, "structure": (t3 - t2) * 1e3,
                    "faces": (t4 - t3) * 1e3, "total": (t4 - t0) * 1e3}
    return d

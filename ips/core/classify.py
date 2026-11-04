"""Separate *content* classification from *degradation* assessment, combining metadata and pixels.

Every issue carries severity (0..1), confidence (0..1) and human-readable evidence citing the
metadata and/or pixel measurements that produced it. Metadata is never trusted alone: e.g. a low
JPEG quality table with no measurable blocking lowers the severity and the confidence.
Thresholds live in ``THRESHOLDS`` and are documented/calibrated in docs/AUTO_RECIPES.md.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .diagnostics import Diagnostics

THRESHOLDS: dict[str, float] = {
    # noise sigma (0-255 scale, Immerkær on flat regions at native resolution)
    "noise_mild": 2.2, "noise_moderate": 4.0, "noise_strong": 7.0,
    # 8x8 grid discontinuity
    "block_mild": 0.12, "block_moderate": 0.30, "block_strong": 0.70,
    # JPEG quality estimated from quantisation tables
    "jpegq_mild": 88, "jpegq_moderate": 70, "jpegq_strong": 45,
    # Crete blur (higher = blurrier)
    "blur_soft": 0.42, "blur_strong": 0.55,
    # illuminant log-chromaticity magnitude (smallest of three estimators) / agreement / ambiguity
    "cast_mag_mild": 0.08, "cast_mag_strong": 0.30, "cast_agree": 0.85, "cast_dominance_max": 0.35,
    # halo / oversharpening
    "halo_mild": 0.12, "halo_strong": 0.18,
    # oversmoothing (texture ratio)
    "smooth_low": 0.95,
    # screenshot
    "flat_screenshot": 0.30,
    # low resolution (long side px)
    "lowres_long_side": 1024,
    # fading (tone range); photos need far stronger evidence than scans
    "fade_dr_scan": 0.70, "fade_dr_photo": 0.50,
}


@dataclass
class Issue:
    code: str
    severity: float
    confidence: float
    evidence: str

    @property
    def level(self) -> str:
        s = self.severity
        return "strong" if s >= 0.75 else "moderate" if s >= 0.45 else "mild" if s >= 0.15 else "trace"


@dataclass
class Assessment:
    content_class: str
    content_scores: dict[str, float]
    content_confidence: float
    issues: list[Issue] = field(default_factory=list)
    intents: list[str] = field(default_factory=list)     # things to preserve
    evidence: list[str] = field(default_factory=list)
    metadata_summary: dict[str, Any] = field(default_factory=dict)
    features: dict[str, float] = field(default_factory=dict)
    jpeg_quality: int | None = None
    source_format: str = ""

    def issue(self, code: str) -> Issue | None:
        for i in self.issues:
            if i.code == code:
                return i
        return None

    def severity(self, code: str) -> float:
        i = self.issue(code)
        return i.severity if i else 0.0

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["issues"] = [dict(asdict(i), level=i.level) for i in self.issues]
        return d


def _ramp(x: float, lo: float, hi: float) -> float:
    if hi == lo:
        return 1.0 if x >= hi else 0.0
    return float(min(1.0, max(0.0, (x - lo) / (hi - lo))))


def assess(diag: Diagnostics, meta: dict[str, Any], *, source_format: str = "", jpeg_quality: int | None = None,
           png_text: dict[str, str] | None = None) -> Assessment:
    T = THRESHOLDS
    f = diag.features
    ev: list[str] = []
    fmt = (source_format or "").upper()

    # ------------------------------------------------------------- content
    s_shot = 0.0
    if f.get("flat_fraction", 0) > T["flat_screenshot"]:
        s_shot += 0.45 + 0.4 * _ramp(f["flat_fraction"], T["flat_screenshot"], 0.7)
    s_shot += 0.25 * _ramp(f.get("axis_edge_fraction", 0), 0.35, 0.75)
    s_shot += 0.15 * _ramp(f.get("text_density", 0), 0.05, 0.3)
    if fmt in ("PNG", "WEBP", "BMP"):
        s_shot += 0.1
    if meta.get("is_camera_capture"):
        s_shot -= 0.5
    sw = str(meta.get("software") or "").lower()
    if any(k in sw for k in ("snipping", "screenshot", "greenshot", "sharex", "screencapture", "gnome-screenshot")):
        s_shot += 0.5
        ev.append(f"Software tag '{meta.get('software')}' indicates a screen capture")

    ai_hints = meta.get("ai_generation_hints") or []
    s_ai = 0.0
    if ai_hints:
        s_ai += 0.75
        ev.append("AI-generation metadata: " + "; ".join(ai_hints[:3]))
    if not meta.get("is_camera_capture") and f.get("dims_multiple_of_64", 0) > 0:
        s_ai += 0.15
    if not meta.get("is_camera_capture") and f.get("noise_sigma_luma", 9) < 1.2 and f.get("flat_fraction", 0) < 0.2:
        s_ai += 0.1
    if meta.get("is_camera_capture"):
        s_ai -= 0.4

    s_scan = 0.6 if meta.get("is_scanner") else 0.0
    s_photo = 0.35
    if meta.get("is_camera_capture"):
        s_photo += 0.5
        ev.append(f"Camera capture: {meta.get('camera_make') or ''} {meta.get('camera_model') or ''}".strip())
    if fmt == "RAW":
        s_photo += 0.4
    if f.get("flat_fraction", 0) < 0.05:
        s_photo += 0.1
    scores = {"photo": round(max(0, s_photo), 3), "screenshot": round(max(0, s_shot), 3),
              "ai_generated": round(max(0, s_ai), 3), "scan": round(max(0, s_scan), 3)}
    ordered = sorted(scores.items(), key=lambda kv: -kv[1])
    cls, top = ordered[0]
    margin = top - ordered[1][1]
    conf = float(min(1.0, 0.45 + margin))
    if f.get("face_count", 0) and cls == "photo":
        ev.append(f"{int(f['face_count'])} face(s) detected ({diag.face_detector})")

    a = Assessment(content_class=cls, content_scores=scores, content_confidence=round(conf, 3), evidence=ev,
                   metadata_summary=meta, features=dict(f), jpeg_quality=jpeg_quality, source_format=fmt)

    # ------------------------------------------------------------- degradations
    long_side = max(diag.width, diag.height)
    blk = f.get("blockiness", 0.0)
    q = jpeg_quality
    pix_sev = _ramp(blk, T["block_mild"], T["block_strong"])
    if q is not None:
        meta_sev = _ramp(T["jpegq_mild"] - q, 0, T["jpegq_mild"] - T["jpegq_strong"] + 10)
        if pix_sev > 0.05 and meta_sev > 0.05:
            sev, conf_j = max(pix_sev, 0.6 * meta_sev + 0.4 * pix_sev), 0.9
            src = "metadata + pixels agree"
        elif meta_sev > 0.05:
            sev, conf_j = 0.5 * meta_sev, 0.5
            src = "metadata suggests compression; pixel blocking weak"
        else:
            sev, conf_j = pix_sev, 0.6
            src = "pixel blocking only"
        if sev >= 0.12:
            a.issues.append(Issue("jpeg_artifacts", round(sev, 3), conf_j,
                                  f"JPEG quality≈{q} (quant tables), blockiness {blk:.2f}: {src}"))
    elif pix_sev >= 0.15:
        a.issues.append(Issue("jpeg_artifacts", round(pix_sev, 3), 0.65,
                              f"8×8 blocking {blk:.2f} in a {fmt or 'non-JPEG'} file: previously JPEG-compressed"))

    nl, nc = f.get("noise_sigma_luma", 0.0), f.get("noise_sigma_chroma", 0.0)
    # noise estimates are inflated by JPEG blocking; discount when blocking is strong
    nl_eff = nl * (1 - 0.35 * pix_sev)
    n_sev = _ramp(nl_eff, T["noise_mild"], T["noise_strong"])
    iso = meta.get("iso")
    if n_sev > 0.05:
        conf_n = 0.7
        why = f"luma σ≈{nl:.1f}, chroma σ≈{nc:.1f}"
        if iso:
            why += f", ISO {iso}"
            if iso >= 1600:
                conf_n = 0.9
        a.issues.append(Issue("noise", round(n_sev, 3), conf_n, why))
        grain_like = nc < 0.45 * nl and (cls == "ai_generated" or meta.get("edit_software") or (iso and iso <= 400))
        if grain_like:
            a.intents.append("grain")
            a.evidence.append(f"Noise is luminance-dominant (chroma/luma {nc / max(nl, 1e-3):.2f}) with "
                              f"{'AI source' if cls == 'ai_generated' else 'low ISO/edit software'}: "
                              "possibly intentional grain")

    blur = f.get("blur_crete", 0.0)
    b_sev = _ramp(blur, T["blur_soft"], T["blur_strong"] + 0.1)
    if b_sev > 0.1 and cls != "screenshot":
        code = "blur_motion" if f.get("gradient_anisotropy", 0) > 0.55 else "blur_soft"
        a.issues.append(Issue(code, round(b_sev, 3), 0.55, f"Crete blur {blur:.2f}, gradient anisotropy "
                                                            f"{f.get('gradient_anisotropy', 0):.2f}"))
    if long_side < T["lowres_long_side"]:
        a.issues.append(Issue("low_resolution", round(_ramp(T["lowres_long_side"] - long_side, 0, 900), 3), 0.95,
                              f"long side {long_side}px"))

    # colour cast: three independent illuminant estimates must agree; a scene dominated by saturated
    # colour in the cast hue (orange cat, red wall) makes the estimate ambiguous -> not a cast.
    mag = f.get("illum_min_mag", 0.0)
    agree = f.get("illum_agreement", -1.0)
    dom = f.get("scene_hue_dominance", 0.0)
    toned = f.get("ab_std", 99) < 6.0 and f.get("chroma_mean", 0) > 3.0
    if toned and cls in ("photo", "scan"):
        a.intents.append("toned_monochrome")
        ev.append(f"Toned monochrome (sepia/cyanotype-like: a/b spread {f.get('ab_std', 0):.1f}): tone preserved")
    elif cls in ("photo", "scan") and mag >= T["cast_mag_mild"]:
        c_sev = _ramp(mag, T["cast_mag_mild"], T["cast_mag_strong"])
        conf_c = 0.85 if agree >= T["cast_agree"] else 0.35
        why = (f"illuminant estimates (gray-world/gray-edge/white-patch) agreement {agree:+.2f}, smallest "
               f"magnitude {mag:.3f}, hue {f.get('illum_hue_deg', 0):.0f}°, scene-hue dominance {dom:.2f}")
        if dom >= T["cast_dominance_max"]:
            conf_c = min(conf_c, 0.3)
            why += " — scene dominated by this hue: ambiguous (subject colour, not illuminant)"
        hue = f.get("illum_hue_deg", 0)
        wb = str(meta.get("white_balance_mode") or "")
        if wb.strip() in ("1", "Manual"):
            conf_c *= 0.7
            why += "; EXIF says manual white balance"
        degraded = any(i.code in ("jpeg_artifacts", "low_resolution") and i.severity >= 0.45 for i in a.issues)
        if 40 <= hue <= 80 and mag < T["cast_mag_strong"] and meta.get("is_camera_capture") and not degraded \
                and not meta.get("exif_dimension_mismatch"):
            a.intents.append("warm_light")
            conf_c *= 0.6
            why += " (warm, recent camera capture: may be intentional lighting)"
        a.issues.append(Issue("color_cast", round(c_sev, 3), round(conf_c, 2), why))

    p50, p99, p1 = f.get("luma_p50", 0.5), f.get("luma_p99", 1.0), f.get("luma_p1", 0.0)
    if cls != "screenshot":
        if p50 < 0.22 and p99 < 0.75:
            a.issues.append(Issue("underexposed", round(_ramp(0.22 - p50, 0, 0.15), 3), 0.7,
                                  f"median luma {p50:.2f}, p99 {p99:.2f}"))
        if f.get("clip_high", 0) > 0.02 and p50 > 0.65:
            a.issues.append(Issue("overexposed", round(_ramp(f["clip_high"], 0.02, 0.15), 3), 0.7,
                                  f"{f['clip_high']:.1%} clipped highlights (not recoverable)"))
        # Natural low-contrast scenes and faded prints have the same global tone statistics, so fade
        # correction needs scan evidence (metadata) or an extreme measurement.
        dr = f.get("dynamic_range", 1.0)
        lim_dr, lim_p1 = (T["fade_dr_scan"], 0.08) if cls == "scan" else (T["fade_dr_photo"], 0.12)
        if dr < lim_dr and p1 > lim_p1:
            a.issues.append(Issue("faded", round(_ramp(lim_dr - dr, 0, 0.25), 3), 0.7 if cls == "scan" else 0.5,
                                  f"dynamic range {dr:.2f}, lifted blacks p1={p1:.2f}"
                                  f"{' (scan)' if cls == 'scan' else ''}"))

    halo = f.get("halo_index", 0.0)
    if halo > T["halo_mild"] and cls != "screenshot":
        a.issues.append(Issue("oversharpened", round(_ramp(halo, T["halo_mild"], T["halo_strong"] + 0.05), 3), 0.55,
                              f"edge overshoot index {halo:.3f}"))
    tex = f.get("texture_ratio", 2.0)
    if cls == "photo" and tex < T["smooth_low"] and nl < T["noise_mild"]:
        a.issues.append(Issue("oversmoothed", round(_ramp(T["smooth_low"] - tex, 0, 0.35), 3), 0.45,
                              f"fine/mid texture ratio {tex:.2f} with very low noise: possibly smoothed earlier"))
    if meta.get("edit_software"):
        a.evidence.append(f"Previously edited with {meta['edit_software']}")
    if meta.get("exif_dimension_mismatch"):
        a.evidence.append("EXIF pixel dimensions differ from actual pixels: image was resized/cropped after capture")
    if cls == "ai_generated":
        a.intents.append("style")
    return a

"""Smart Auto router: Assessment -> Recipe (fewest necessary operations, ordered, parameterised).

Deterministic and evaluable. Every decision appends a rationale line citing the evidence. Learned
adjustments (from recorded manual corrections) are applied from ``router_adjustments.json`` – a
small, human-readable table produced by ``python -m ips.eval.auto_accuracy --learn``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..backends.registry import MODELS
from ..paths import app_data_dir
from .classify import THRESHOLDS, Assessment
from .perf import BUDGET_SECONDS, FastProfile, predict_seconds, route_key
from .recipes import Operation, Recipe, default_op


@dataclass
class RouteRequest:
    mode: str = "faithful"               # faithful | balanced | generative
    scale: float | str = "auto"          # "auto" | 1 | 2 | 4 | float
    target_long_edge: int | None = None  # e.g. AI/Video Ready profile; overrides scale when set
    upscale_low_res_auto: bool = True    # Auto: ×2 for sources with long side < lowres threshold
    quality_override: bool = False       # allow running outside the Fast Profile
    device_kind: str = "cpu"
    device_name: str = ""
    allow_models: bool = True            # False when torch/weights unavailable
    look: str = "enhanced"               # natural | enhanced | vivid  (visible, non-generative enhancement)


# Non-generative "look" enhancement applied on top of defect repair. Global photometric + local-contrast +
# halo-clamped sharpening only: no resampling of facial geometry, no texture synthesis.
LOOKS: dict[str, dict[str, float]] = {
    "natural": {},
    "enhanced": {"contrast": 0.20, "vibrance": 0.30, "saturation": 0.03, "clarity": 0.70, "clarity_clip": 2.5,
                 "sharpen": 0.85, "upscale_below": 1536, "face_protect": 0.35},
    "vivid": {"contrast": 0.32, "vibrance": 0.48, "saturation": 0.08, "clarity": 1.0, "clarity_clip": 3.2,
              "sharpen": 1.2, "upscale_below": 2048, "face_protect": 0.3},
}


def _installed(model_id: str) -> bool:
    m = MODELS.get(model_id)
    return bool(m and m.installed)


def load_adjustments() -> dict[str, Any]:
    p = app_data_dir() / "router_adjustments.json"
    if p.is_file():
        try:
            return json.loads(p.read_text("utf-8"))
        except Exception:
            return {}
    return {}


def choose_scale(a: Assessment, req: RouteRequest, rationale: list[str]) -> float:
    long_side = max(int(a.features.get("_w", 0)), int(a.features.get("_h", 0)))
    if req.target_long_edge:
        s = req.target_long_edge / max(1, long_side)
        if s <= 1.0:
            rationale.append(f"Target long edge {req.target_long_edge}px ≤ source {long_side}px: no upscale "
                             "(resize happens only at export, without cropping).")
            return 1.0
        rationale.append(f"Target long edge {req.target_long_edge}px needs ×{s:.2f}.")
        return round(s, 4)
    if req.scale != "auto":
        s = float(req.scale)
        rationale.append(f"Scale ×{s:g} requested explicitly.")
        return s
    below = int(LOOKS.get(req.look, {}).get("upscale_below", THRESHOLDS["lowres_long_side"]))
    below = max(below, int(THRESHOLDS["lowres_long_side"]))
    if req.upscale_low_res_auto and long_side and long_side < below:
        rationale.append(f"Source long side {long_side}px < {below}px ({req.look} look): Auto proposes ×2.")
        return 2.0
    rationale.append("Resolution adequate: no upscale (×1).")
    return 1.0


def route(a: Assessment, req: RouteRequest, width: int, height: int,
          profile: FastProfile | None = None) -> Recipe:
    a.features["_w"], a.features["_h"] = width, height
    rat: list[str] = []
    flags: list[str] = []
    ops: list[Operation] = []
    confs: list[float] = [a.content_confidence]
    cls = a.content_class
    adj = load_adjustments().get(cls, {})
    rat.append(f"Content: {cls} (confidence {a.content_confidence:.2f}; scores {a.content_scores}).")
    for e in a.evidence:
        rat.append("Evidence: " + e)
    models_ok = req.allow_models
    faces = int(a.features.get("face_count", 0))

    def suppressed(kind: str) -> bool:
        s = adj.get(kind)
        if s and s.get("suppress"):
            rat.append(f"Skipped {kind}: learned from {s.get('n', '?')} manual corrections for {cls} images "
                       f"(acceptance {s.get('acceptance', 0):.0%}).")
            return True
        return False

    # ---------------------------------------------------------------- JPEG
    j = a.issue("jpeg_artifacts")
    if j and j.severity >= 0.15 and not suppressed("jpeg_cleanup"):
        if models_ok and _installed("fbcnn_color"):
            q = a.jpeg_quality
            pred = a.features.get("fbcnn_qf_pred")
            if pred is not None:
                qf = int(round(pred))
                src_q = f"pixel-estimated qf≈{qf} (FBCNN blind prediction)"
                if q is not None and abs(q - qf) >= 15:
                    src_q += f"; container tables say q={q}: file was re-saved after earlier compression"
            else:
                qf = q if q is not None else int(round(80 - 60 * j.severity))
                if j.confidence < 0.8:
                    qf += 10  # metadata not confirmed by pixels: gentler
                src_q = f"qf from {'quant tables' if q is not None else 'blockiness'}"
            qf = int(max(5, min(90, qf)))
            op = default_op("jpeg_cleanup", "model:fbcnn_color")
            op.params["qf"] = qf
            op.why = f"JPEG artifacts {j.level} ({j.evidence}) → FBCNN native qf={qf} [{src_q}]"
            if cls == "screenshot":
                op.params["qf"] = min(90, qf + 10)
                flags.append("Text/UI image processed by JPEG cleanup: verify glyph legibility.")
            ops.append(op)
            confs.append(j.confidence)
        else:
            rat.append("JPEG artifacts found but FBCNN unavailable: not treated (no classical substitute).")

    # ---------------------------------------------------------------- noise
    n = a.issue("noise")
    if n and n.severity >= 0.12 and not suppressed("denoise"):
        if "grain" in a.intents and req.mode == "faithful" and n.severity < 0.75:
            rat.append("Noise looks like intentional grain: preserved (Faithful). Enable denoise manually if unwanted.")
        elif models_ok and cls == "photo" and _installed("scunet_real_psnr"):
            blend = round(0.55 + 0.45 * n.severity, 2)
            op = default_op("denoise", "model:scunet_real_psnr")
            op.blend = blend
            op.why = (f"Noise {n.level} ({n.evidence}) → SCUNet real-noise (blind, PSNR-trained); "
                      f"post-blend {blend:.0%} (SCUNet has no native strength)")
            if faces:
                op.params["face_protect"] = 0.5
                op.mask = "protect_faces"
                op.why += "; face regions blended 50% toward input to protect skin texture"
            ops.append(op)
            confs.append(n.confidence)
        elif models_ok and _installed("drunet_color"):
            base_sigma = a.features.get("noise_sigma_rgb", a.features.get("noise_sigma_luma", 3.0))
            # σ = 1.0 × measured: on the Gaussian-noise fixture 1.0× beat 0.8× on PSNR and SSIM (TEST_RESULTS)
            sigma = round(max(1.0, base_sigma), 1)
            op = default_op("denoise", "model:drunet_color")
            op.params["sigma"] = sigma
            op.why = f"Noise {n.level} ({n.evidence}) on {cls} → DRUNet with native σ={sigma} (conservative)"
            if faces:
                op.params["face_protect"] = 0.4
                op.mask = "protect_faces"
            ops.append(op)
            confs.append(n.confidence * 0.9)
        else:
            op = default_op("denoise", "classical:nlmeans", strength=int(30 + 40 * n.severity))
            op.why = f"Noise {n.level}: classical NL-means fallback (models unavailable)"
            ops.append(op)
            confs.append(0.5)

    # ---------------------------------------------------------------- blur
    for code in ("blur_soft", "blur_motion"):
        b = a.issue(code)
        if b and b.severity >= 0.5:
            flags.append(f"{'Motion' if code == 'blur_motion' else 'Defocus/soft'} blur ({b.evidence}): faithful "
                         "deblurring is not automatic; Manual offers classical halo-free deconvolution sharpening.")
            confs.append(0.6)

    # ---------------------------------------------------------------- colour
    c = a.issue("color_cast")
    if c and c.severity >= 0.2 and cls == "photo" and req.mode == "faithful":
        # Held-out evidence (docs/TEST_RESULTS.md §5b): automatic WB increased colour error vs ground truth on most
        # unseen photos (scene colour is removed together with any cast). Faithful Auto therefore only reports it.
        rat.append(f"Possible colour cast ({c.evidence}). Not corrected automatically for photos: illuminant "
                   "estimates cannot separate scene colour from a cast reliably. Try White balance in Manual.")
        flags.append("Possible colour cast detected — White balance available in Manual (not applied automatically).")
    elif c and c.severity >= 0.2 and cls in ("photo", "scan") and not suppressed("white_balance"):
        if "warm_light" in a.intents and c.severity < 0.7:
            rat.append(f"Warm cast kept: likely intentional lighting ({c.evidence}). Manual WB available.")
        elif c.confidence < 0.6:
            rat.append(f"Possible cast not corrected (low confidence {c.confidence:.2f}): {c.evidence}.")
        else:
            import math
            strength = round(min(0.8, 0.45 + 0.4 * c.severity) * (1 - a.features.get("scene_hue_dominance", 0)), 2)
            r_, b_ = a.features.get("illum_cons_r", 0.0), a.features.get("illum_cons_b", 0.0)
            gains = [round(math.exp(-r_ * strength), 4), 1.0, round(math.exp(-b_ * strength), 4)]
            op = default_op("white_balance", "classical:neutral_cast")
            op.params.update(strength=strength, gains=gains)
            op.strength = int(strength * 100)
            op.why = (f"Colour cast {c.level} ({c.evidence}) → remove {strength:.0%} of consensus illuminant "
                      f"(linear-light gains R×{gains[0]}, B×{gains[2]}, luminance preserved)")
            ops.append(op)
            confs.append(c.confidence)
    if "toned_monochrome" in a.intents:
        rat.append("Toned monochrome detected: tone/colour left as authored.")

    # ---------------------------------------------------------------- exposure / fade
    ex_params: dict[str, float] = {}
    u = a.issue("underexposed")
    if u and u.severity >= 0.2 and not suppressed("exposure"):
        p50 = a.features.get("luma_p50", 0.2)
        p99 = a.features.get("luma_p99", 0.7)
        import math
        ev = math.log2(max(1.0, min(0.36 / max(p50, 0.02), 0.95 / max(p99, 0.05))))
        ev = round(min(1.2, ev * 0.8), 2)
        if ev >= 0.15:
            ex_params["ev"] = ev
            rat.append(f"Underexposed ({u.evidence}) → +{ev} EV (limited so p99 stays below clipping).")
    fd = a.issue("faded")
    if fd and fd.severity >= 0.25 and cls in ("photo", "scan") and not suppressed("exposure"):
        black = round(max(0.0, a.features.get("luma_p1", 0) * 0.8), 3)
        white = round(min(1.0, max(a.features.get("luma_p99", 1.0) + 0.02, black + 0.4)), 3)
        ex_params.update(black=black, white=white)
        rat.append(f"Faded/low contrast ({fd.evidence}) → levels black {black}, white {white}.")
    if ex_params:
        op = default_op("exposure", "classical:levels")
        op.params.update(ex_params)
        op.why = "Exposure/levels: " + ", ".join(f"{k}={v}" for k, v in ex_params.items())
        ops.append(op)
        confs.append(0.7)
    o = a.issue("overexposed")
    if o:
        flags.append(f"Clipped highlights ({o.evidence}): detail there cannot be faithfully recovered.")

    # ---------------------------------------------------------------- look (visible enhancement)
    L = LOOKS.get(req.look, {})
    if L and cls != "screenshot" and "toned_monochrome" not in a.intents:
        tag = f"{req.look.title()} look"
        p1, p99, p50 = a.features.get("luma_p1", 0), a.features.get("luma_p99", 1), a.features.get("luma_p50", 0.5)
        if not any(o.kind == "exposure" for o in ops) and (p1 > 0.02 or p99 < 0.97):
            op = default_op("exposure", "classical:levels")
            op.params.update(black=round(min(0.06, p1 * 0.8), 3), white=round(max(0.88, min(1.0, p99 + 0.015)), 3),
                             gamma=round(1.06 if p50 < 0.35 else 0.97 if p50 > 0.65 else 1.0, 3))
            op.why = f"{tag}: tonal range stretch (black {op.params['black']}, white {op.params['white']})"
            ops.append(op)
        op = default_op("tone", "classical:contrast_saturation")
        op.params.update(contrast=L["contrast"], vibrance=L["vibrance"] * (0.85 if "style" in a.intents else 1.0),
                         saturation=L["saturation"])
        op.why = (f"{tag}: contrast +{L['contrast']}, vibrance +{op.params['vibrance']:.2f} (skin hues protected)")
        ops.append(op)
        amt = L["clarity"] * (0.5 if a.issue("oversharpened") else 1.0)
        op = default_op("tone", "classical:clarity")
        op.params.update(amount=round(amt, 2), clip=L["clarity_clip"])
        if faces:
            op.mask, op.params["face_protect"] = "protect_faces", L["face_protect"]
        op.why = f"{tag}: local contrast {amt:.2f} on lightness only" + (
            f"; faces {int(L['face_protect'] * 100)}% protected" if faces else "")
        ops.append(op)
        rat.append(f"{tag} requested: visible, non-generative enhancement (tone, local contrast, vibrance, "
                   "halo-free sharpening). Switch Look to Natural for defect repair only.")

    # ---------------------------------------------------------------- upscale
    scale = choose_scale(a, req, rat)
    if scale > 1.0:
        up = _choose_upscaler(a, req, scale, rat, flags, models_ok)
        ops.extend(up)

    # ---------------------------------------------------------------- sharpen (rarely)
    if a.issue("oversharpened"):
        rat.append(f"Previously oversharpened ({a.issue('oversharpened').evidence}): no sharpening added. Halos "
                   "cannot be removed faithfully.")
    elif L and cls != "screenshot":
        mp_out = width * height * scale * scale / 1e6
        op = default_op("sharpen", "classical:halo_free_unsharp")
        op.params.update(amount=round(L["sharpen"] + 0.3 * a.severity("blur_soft"), 2),
                         radius=1.0 if mp_out <= 4 else 1.4, threshold=0.012)
        if faces:
            op.mask, op.params["face_protect"] = "protect_faces", L["face_protect"]
        op.why = (f"{req.look.title()} look: halo-clamped detail sharpening amount {op.params['amount']} "
                  f"(cannot overshoot local min/max)")
        ops.append(op)
    elif cls in ("photo", "scan") and a.severity("blur_soft") >= 0.25 and not suppressed("sharpen"):
        op = default_op("sharpen", "classical:halo_free_unsharp", strength=int(20 + 20 * a.severity("blur_soft")))
        op.why = "Mild softness → halo-clamped luminance unsharp (cannot create overshoot)"
        ops.append(op)
    if a.issue("oversmoothed"):
        flags.append("Image appears previously smoothed: Faithful mode will not synthesise texture. "
                     "Balanced (GAN) can add plausible texture – labelled as such.")
    if "style" in a.intents and not L:
        rat.append("AI-generated/stylised source: colour, tone and grain left as authored.")

    r = Recipe(ops=ops, mode=req.mode, target_scale=scale, created_by="auto", rationale=rat, flags=flags,
               content_class=cls, quality_override=req.quality_override)
    r.ops = r.sorted_ops()
    if not r.ops:
        r.no_op = True
        r.rationale.append("No enhancement needed: no measured issue exceeded its threshold "
                           f"(noise σ {a.features.get('noise_sigma_luma', 0):.1f}, blocking "
                           f"{a.features.get('blockiness', 0):.2f}, cast {a.features.get('cast_magnitude', 0):.1f}). "
                           "Exports will only re-encode/convert as the output profile requires.")
    r.confidence = round(float(min(confs) if confs else 1.0), 3)
    if r.confidence < 0.55:
        r.flags.append(f"Low-confidence diagnosis ({r.confidence:.2f}): review recommended.")
    if faces and any(op.model_id for op in r.ops):
        r.flags.append(f"{faces} face(s) present: identity drift signals will be computed after processing.")
    _apply_perf(r, width, height, req, profile)
    return r


def _choose_upscaler(a: Assessment, req: RouteRequest, scale: float, rat: list[str], flags: list[str],
                     models_ok: bool) -> list[Operation]:
    cls = a.content_class
    ops: list[Operation] = []
    if cls == "screenshot":
        pixel_art = a.features.get("unique_color_ratio", 1) < 0.02 and a.features.get("flat_fraction", 0) > 0.6
        if pixel_art and float(scale).is_integer():
            op = default_op("upscale", "classical:nearest")
            op.params["scale"] = int(scale)
            op.why = "Flat-colour UI/pixel graphics → integer nearest-neighbour (exact pixels, no invented detail)"
        else:
            op = default_op("upscale", "classical:lanczos")
            op.params["scale"] = scale
            op.why = "Text/UI → Lanczos (no learned model may invent glyphs); halo-clamped sharpening not added"
        rat.append(op.why)
        return [op]
    if req.mode == "balanced" and models_ok:
        mid = "realesrgan_x4plus" if scale > 2 else "realesrgan_x2plus"
        if _installed(mid):
            op = default_op("upscale", f"model:{mid}")
            op.why = (f"Balanced mode → {MODELS[mid].name} (GAN: synthesises plausible texture; not a faithful "
                      "reconstruction)")
            rat.append(op.why)
            flags.append("Balanced (GAN) upscale: fine texture is synthesised; in the portrait benchmark Real-ESRGAN "
                         "visibly smoothed skin (beautification class). Review faces and text.")
            ops.append(op)
            if MODELS[mid].scale != scale:
                rat.append(f"Model ×{MODELS[mid].scale} result resized to exact ×{scale:g} with area/Lanczos.")
            return ops
    if models_ok and _installed("swinir_x2"):
        op = default_op("upscale", "model:swinir_x2")
        op.why = ("Faithful upscale → SwinIR classical ×2 (PSNR-trained; reconstructs, does not hallucinate "
                  "texture)")
        rat.append(op.why)
        ops.append(op)
        if scale > 2.0:
            op2 = default_op("upscale", "classical:lanczos")
            op2.params["scale"] = round(scale / 2.0, 4)
            op2.why = f"Remaining ×{scale / 2:g} by Lanczos (no second learned pass: avoids compounding artifacts)"
            rat.append(op2.why)
            ops.append(op2)
        elif scale < 2.0:
            rat.append(f"×2 model output resized to exact ×{scale:g}.")
        return ops
    op = default_op("upscale", "classical:lanczos")
    op.params["scale"] = scale
    op.why = "Models unavailable → Lanczos upscale"
    rat.append(op.why)
    return [op]


def _apply_perf(r: Recipe, width: int, height: int, req: RouteRequest, profile: FastProfile | None) -> None:
    profile = profile or FastProfile()
    r.predicted_seconds = predict_seconds(r, width, height, req.device_kind, req.device_name, profile)
    if r.no_op:
        return
    certified = bool(req.device_name) and profile.certified(req.device_name)
    if certified:
        sup = profile.supported(req.device_name, route_key(r), width * height / 1e6)
        if sup is not True:
            r.outside_fast_profile = True
            msg = ("Route/size not measured within the 15 s Fast Profile on this GPU" if sup is None else
                   "Measured p95 for this route/size exceeds 15 s on this GPU")
            if r.quality_override:
                r.flags.append(msg + ": running under explicit Quality Override.")
            else:
                r.flags.append(msg + ": requires Quality Override (resolution/quality are never reduced silently).")
    else:
        r.rationale.append(f"Device '{req.device_name or req.device_kind}' has no measured Fast Profile: timing is "
                           f"recorded but the 15 s contract is not certified here (prior estimate "
                           f"{r.predicted_seconds:.1f} s).")
        if r.predicted_seconds and r.predicted_seconds > BUDGET_SECONDS:
            r.rationale.append("Prior estimate exceeds 15 s on this device (informational; not a certified result).")

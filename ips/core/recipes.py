"""Recipes: ordered, fully explicit operation lists.

A recipe is plain data (JSON-serialisable) so that Auto decisions are inspectable, editable in Manual,
reproducible (hash), storable as presets and comparable in the Auto Accuracy Benchmark.

UI strength (0–100) is a *normalised convenience*. For every method, ``map_strength`` documents
whether strength drives a **native** model/algorithm parameter or a **post-blend** with the op input.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

RECIPE_SCHEMA = 2

# kind -> ordered list of methods. "model:<id>" methods run neural nets; others are classical.
OP_KINDS = ("geometry", "jpeg_cleanup", "denoise", "deblur", "white_balance", "exposure", "tone", "upscale",
            "sharpen")

CANONICAL_ORDER = {k: i for i, k in enumerate(OP_KINDS)}

METHODS: dict[str, list[str]] = {
    "jpeg_cleanup": ["model:fbcnn_color", "model:swinir_car_q30", "model:swinir_car_q10"],
    "denoise": ["model:scunet_real_psnr", "model:drunet_color", "classical:nlmeans", "model:swinir_dn15"],
    "deblur": ["classical:unsharp_deconv"],
    "white_balance": ["classical:neutral_cast"],
    "exposure": ["classical:levels"],
    "tone": ["classical:contrast_saturation", "classical:clarity"],
    "upscale": ["model:swinir_x2", "classical:lanczos", "classical:nearest", "model:realesrgan_x2plus",
                "model:realesrgan_x4plus", "model:realesr_general_x4v3", "model:realplksr_nomos2_x4"],
    "sharpen": ["classical:halo_free_unsharp"],
    "geometry": ["classical:crop_rotate"],
}

CLASSICAL_PARAMS: dict[str, dict[str, Any]] = {
    "classical:nlmeans": {"h": 6.0, "h_color": 6.0},
    "classical:unsharp_deconv": {"radius": 1.5, "amount": 0.6},
    "classical:neutral_cast": {"strength": 0.7},
    "classical:levels": {"ev": 0.0, "black": 0.0, "white": 1.0, "gamma": 1.0},
    "classical:contrast_saturation": {"contrast": 0.0, "saturation": 0.0, "vibrance": 0.0},
    "classical:clarity": {"amount": 0.5, "clip": 2.0, "tiles": 8},
    "classical:lanczos": {"scale": 2.0},
    "classical:nearest": {"scale": 2},
    "classical:halo_free_unsharp": {"amount": 0.35, "radius": 1.0, "threshold": 0.01},
    "classical:crop_rotate": {"angle": 0.0, "crop": None, "rotate90": 0, "flip": False},
}


@dataclass
class Operation:
    kind: str
    method: str
    params: dict[str, Any] = field(default_factory=dict)
    blend: float = 1.0                # post-blend with op input; 1.0 = full effect
    mask: str | None = None           # None | "protect_faces" | "protect_text" | "faces_only" | "region:x0,y0,x1,y1"
    enabled: bool = True
    strength: int | None = None       # last UI strength used to derive params (informational)
    why: str = ""                     # rationale line (Auto) or "manual"

    @property
    def model_id(self) -> str | None:
        return self.method.split(":", 1)[1] if self.method.startswith("model:") else None

    def label(self) -> str:
        if self.method.startswith("comfy:"):
            return "ComfyUI · " + self.method.split(":", 2)[2]
        if self.model_id:
            from ..backends.registry import MODELS
            m = MODELS.get(self.model_id)
            return m.name if m else self.model_id
        return self.method.split(":", 1)[1].replace("_", " ")


@dataclass
class Recipe:
    ops: list[Operation] = field(default_factory=list)
    mode: str = "faithful"            # faithful | balanced | generative
    target_scale: float = 1.0
    created_by: str = "auto"          # auto | manual | preset:<name>
    rationale: list[str] = field(default_factory=list)
    confidence: float = 1.0
    flags: list[str] = field(default_factory=list)
    no_op: bool = False
    predicted_seconds: float | None = None
    outside_fast_profile: bool = False
    quality_override: bool = False
    content_class: str = ""
    schema: int = RECIPE_SCHEMA

    # ----------------------------------------------------------------- identity
    def canonical(self) -> dict[str, Any]:
        """Fields that determine pixels (hash input)."""
        return {"schema": self.schema, "mode": self.mode, "target_scale": round(float(self.target_scale), 4),
                "ops": [{"kind": o.kind, "method": o.method, "params": o.params, "blend": round(o.blend, 4),
                         "mask": o.mask} for o in self.ops if o.enabled]}

    @property
    def hash(self) -> str:
        s = json.dumps(self.canonical(), sort_keys=True, default=str)
        return hashlib.sha256(s.encode()).hexdigest()[:16]

    @property
    def is_generative(self) -> bool:
        from ..backends.registry import MODELS
        return any(o.enabled and o.model_id and MODELS.get(o.model_id) and
                   MODELS[o.model_id].character == "generative" for o in self.ops)

    @property
    def uses_external(self) -> bool:
        """External (ComfyUI) models: character not benchmarked by Studio -> always reviewed."""
        return any(o.enabled and o.method.startswith("comfy:") for o in self.ops)

    @property
    def uses_gan(self) -> bool:
        from ..backends.registry import MODELS
        return any(o.enabled and o.model_id and MODELS.get(o.model_id) and
                   MODELS[o.model_id].character == "balanced" for o in self.ops)

    def summary(self) -> str:
        if self.no_op or not any(o.enabled for o in self.ops):
            return "No enhancement needed (no-op)"
        parts = []
        for o in self.ops:
            if not o.enabled:
                continue
            p = ",".join(f"{k}={v}" for k, v in o.params.items() if v is not None)
            parts.append(f"{o.kind}[{o.method}{('|' + p) if p else ''}{'|blend=%.2f' % o.blend if o.blend < 1 else ''}"
                         f"{('|' + o.mask) if o.mask else ''}]")
        return f"{self.mode} ×{self.target_scale:g}: " + " → ".join(parts)

    def sorted_ops(self) -> list[Operation]:
        return sorted(self.ops, key=lambda o: CANONICAL_ORDER.get(o.kind, 99))

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["hash"] = self.hash
        return d

    @staticmethod
    def from_json(d: dict[str, Any]) -> "Recipe":
        d = copy.deepcopy(d)
        d.pop("hash", None)
        ops = [Operation(**o) for o in d.pop("ops", [])]
        known = {f for f in Recipe.__dataclass_fields__}
        return Recipe(ops=ops, **{k: v for k, v in d.items() if k in known})

    def clone(self, **changes) -> "Recipe":
        r = Recipe.from_json(self.to_json())
        for k, v in changes.items():
            setattr(r, k, v)
        return r


# ----------------------------------------------------------------------------- strength mapping

def map_strength(method: str, strength: int) -> tuple[dict[str, Any], float, str]:
    """Map UI strength 0..100 -> (native params, blend, honest description)."""
    s = max(0, min(100, int(strength))) / 100.0
    mid = method.split(":", 1)[1]
    if method == "model:fbcnn_color":
        qf = int(round(95 - 85 * s))
        return {"qf": qf}, 1.0, f"native: FBCNN quality factor qf={qf} (strength {strength} → 95−85·s)"
    if method == "model:drunet_color":
        sigma = round(50 * s, 1)
        return {"sigma": sigma}, 1.0, f"native: DRUNet σ={sigma} (strength {strength} → 50·s)"
    if method == "model:realesr_general_x4v3":
        return {"denoise_strength": round(s, 2)}, 1.0, f"native: DNI weight {s:.2f} between general/wdn weights"
    if method.startswith("model:"):
        return {}, round(s, 3), (f"post-blend: {mid} has no native strength input; output mixed with its input "
                                 f"at {s:.0%}")
    base = dict(CLASSICAL_PARAMS.get(method, {}))
    if method == "classical:nlmeans":
        base.update(h=round(2 + 14 * s, 1), h_color=round(2 + 14 * s, 1))
        return base, 1.0, f"native: NL-means filter strength h={base['h']}"
    if method == "classical:neutral_cast":
        base["strength"] = round(s, 2)
        return base, 1.0, f"native: remove {s:.0%} of the measured neutral-axis cast"
    if method == "classical:halo_free_unsharp":
        base["amount"] = round(1.2 * s, 2)
        return base, 1.0, f"native: unsharp amount {base['amount']} with halo clamp"
    if method == "classical:unsharp_deconv":
        base["amount"] = round(1.5 * s, 2)
        return base, 1.0, f"native: deconvolution-style sharpening amount {base['amount']}"
    if method == "classical:contrast_saturation":
        base.update(contrast=round(0.3 * s, 3))
        return base, 1.0, f"native: contrast +{base['contrast']}"
    if method == "classical:clarity":
        base.update(amount=round(s, 2))
        return base, 1.0, f"native: local-contrast blend {base['amount']} (lightness only)"
    return base, round(s, 3), "post-blend"


def default_op(kind: str, method: str | None = None, strength: int | None = None) -> Operation:
    """Operation with native defaults. Without *strength*, model ops run at full effect (blend 1.0)."""
    method = method or METHODS[kind][0]
    if strength is None:
        params, blend, why = {}, 1.0, "native defaults"
    else:
        params, blend, why = map_strength(method, strength)
    if method.startswith("model:"):
        from ..backends.registry import MODELS
        m = MODELS.get(method.split(":", 1)[1])
        base = m.native_defaults() if m else {}
        base.update(params)
        params = base
    else:
        base = dict(CLASSICAL_PARAMS.get(method, {}))
        base.update(params)
        params = base
    return Operation(kind=kind, method=method, params=params, blend=blend, strength=strength, why=why)

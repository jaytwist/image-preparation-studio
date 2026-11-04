"""Latency prediction and the empirically-defined Fast Profile.

The Fast Profile is *empty* until ``tools/bench.py`` has measured this device. Entries are keyed by
device name and record, for each route (method) and input-megapixel bucket, measured warm p95 job
time. A route/size is "supported" only if a measured p95 ≤ budget exists for an equal-or-larger
bucket on the current device. Nothing is assumed from other GPUs or from marketing claims.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from ..paths import app_data_dir

BUDGET_SECONDS = 15.0
MP_BUCKETS = (0.5, 1, 2, 4, 8, 12, 16, 24, 36, 50)

# Rough *priors* (ms per output megapixel) used only for ordering and for warning before a run on an
# uncertified device. CPU values measured in the development container (4 vCPU); GPU values are
# placeholders replaced by bench measurements. They never certify a route.
PRIOR_MS_PER_MP = {
    "cpu": {"model:fbcnn_color": 9000, "model:scunet_real_psnr": 30000, "model:drunet_color": 9000,
            "model:swinir_x2": 60000, "model:realesrgan_x2plus": 12000, "model:realesrgan_x4plus": 6000,
            "model:realesr_general_x4v3": 900, "model:realplksr_nomos2_x4": 4000, "model:swinir_car_q10": 90000,
            "model:swinir_car_q30": 90000, "model:swinir_dn15": 90000, "model:restormer_real_dn": 60000,
            "model:restormer_defocus": 60000},
    "cuda": {"model:fbcnn_color": 300, "model:scunet_real_psnr": 900, "model:drunet_color": 250,
             "model:swinir_x2": 1500, "model:realesrgan_x2plus": 400, "model:realesrgan_x4plus": 150,
             "model:realesr_general_x4v3": 40, "model:realplksr_nomos2_x4": 150, "model:swinir_car_q10": 2500,
             "model:swinir_car_q30": 2500, "model:swinir_dn15": 2500, "model:restormer_real_dn": 1800,
             "model:restormer_defocus": 1800},
}
CLASSICAL_MS_PER_MP = 60.0


def bucket(mp: float) -> float:
    for b in MP_BUCKETS:
        if mp <= b + 1e-9:
            return b
    return MP_BUCKETS[-1]


class FastProfile:
    _lock = threading.Lock()

    def __init__(self, path: Path | None = None):
        self.path = path or (app_data_dir() / "fast_profile.json")
        self.data: dict[str, Any] = {}
        if self.path.is_file():
            try:
                self.data = json.loads(self.path.read_text("utf-8"))
            except Exception:
                self.data = {}

    def device_entry(self, device_name: str) -> dict[str, Any] | None:
        return self.data.get("devices", {}).get(device_name)

    def certified(self, device_name: str) -> bool:
        e = self.device_entry(device_name)
        return bool(e and e.get("routes"))

    def supported(self, device_name: str, route_key: str, in_mp: float, budget: float = BUDGET_SECONDS) -> bool | None:
        """True/False if measured, None if this device/route was never measured."""
        e = self.device_entry(device_name)
        if not e:
            return None
        r = e.get("routes", {}).get(route_key)
        if not r:
            return None
        b = bucket(in_mp)
        ok_buckets = [float(k) for k, v in r.items() if v.get("p95_s") is not None and v["p95_s"] <= budget]
        bigger = [k for k in ok_buckets if k >= b]
        return bool(bigger)

    def measured_ms_per_mp(self, device_name: str, method: str) -> float | None:
        e = self.device_entry(device_name)
        if not e:
            return None
        return e.get("ms_per_mp", {}).get(method)

    def record(self, device_name: str, routes: dict[str, Any], ms_per_mp: dict[str, float], meta: dict[str, Any]):
        with self._lock:
            dev = self.data.setdefault("devices", {}).setdefault(device_name, {})
            dev.setdefault("routes", {}).update(routes)
            dev.setdefault("ms_per_mp", {}).update(ms_per_mp)
            dev["meta"] = meta
            self.path.write_text(json.dumps(self.data, indent=2), "utf-8")


def route_key(recipe) -> str:
    """Stable key for a recipe *shape* (methods + scale), independent of parameter values."""
    meths = "+".join(o.method.split(":", 1)[1] for o in recipe.sorted_ops() if o.enabled) or "noop"
    return f"{meths}@x{recipe.target_scale:g}"


def predict_seconds(recipe, width: int, height: int, device_kind: str, device_name: str = "",
                    profile: FastProfile | None = None) -> float:
    mp_in = width * height / 1e6
    total_ms = 400.0 + mp_in * 120.0  # decode/diagnose/encode/metadata overhead prior
    cur_mp = mp_in
    for o in recipe.sorted_ops():
        if not o.enabled:
            continue
        out_mp = cur_mp
        if o.kind == "upscale":
            sc = float(o.params.get("scale", 0) or 0)
            if o.model_id:
                from ..backends.registry import MODELS
                sc = MODELS[o.model_id].scale
            out_mp = cur_mp * (sc or 1) ** 2
        rate = None
        if profile and device_name:
            rate = profile.measured_ms_per_mp(device_name, o.method)
        if rate is None:
            rate = PRIOR_MS_PER_MP.get(device_kind, PRIOR_MS_PER_MP["cpu"]).get(o.method, CLASSICAL_MS_PER_MP)
        total_ms += rate * out_mp
        cur_mp = out_mp
    final_mp = mp_in * recipe.target_scale ** 2
    total_ms += final_mp * 80.0  # encode cost prior
    return round(total_ms / 1000.0, 2)

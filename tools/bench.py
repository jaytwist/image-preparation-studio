"""Performance benchmark that *defines* the Fast Profile empirically on the current device.

For each route (recipe shape) and input size, it measures complete jobs through ``Studio.prepare``:
accepted execution -> diagnostics -> inference -> metadata -> encoded + committed archive output.
It records cold (first, with model load) and warm runs separately, p50/p95/max, peak VRAM, and
writes the results to the Fast Profile store, which the router then uses. Nothing is assumed: a
route/size that was not measured, or whose warm p95 exceeds 15 s, is not Fast-Profile-supported.

    python tools/bench.py                      # default grid
    python tools/bench.py --sizes 1,2,4,8,12 --repeats 5 --routes swinir_x2,fbcnn
    python tools/bench.py --quick              # small grid for a smoke check
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

import ips  # noqa: E402,F401
from ips.core.imageio import encode_image  # noqa: E402
from ips.core.perf import BUDGET_SECONDS, FastProfile, bucket, route_key  # noqa: E402
from ips.core.recipes import Operation, Recipe, default_op  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def routes() -> dict[str, Recipe]:
    def up(m, s=2.0):
        op = default_op("upscale", m)
        if not op.model_id:
            op.params["scale"] = s
        return op
    R = {
        "noop": Recipe(no_op=True),
        "fbcnn": Recipe(ops=[Operation("jpeg_cleanup", "model:fbcnn_color", {"qf": 40})]),
        "scunet": Recipe(ops=[default_op("denoise", "model:scunet_real_psnr")]),
        "drunet": Recipe(ops=[Operation("denoise", "model:drunet_color", {"sigma": 10})]),
        "fbcnn+scunet": Recipe(ops=[Operation("jpeg_cleanup", "model:fbcnn_color", {"qf": 40}),
                                    default_op("denoise", "model:scunet_real_psnr")]),
        "wb+levels": Recipe(ops=[Operation("white_balance", "classical:neutral_cast", {"strength": 0.5}),
                                 Operation("exposure", "classical:levels", {"black": 0.05, "white": 0.95})]),
        "lanczos_x2": Recipe(ops=[up("classical:lanczos")], target_scale=2),
        "swinir_x2": Recipe(ops=[up("model:swinir_x2")], target_scale=2),
        "fbcnn+swinir_x2": Recipe(ops=[Operation("jpeg_cleanup", "model:fbcnn_color", {"qf": 40}),
                                       up("model:swinir_x2")], target_scale=2),
        "realesrgan_x2plus": Recipe(ops=[up("model:realesrgan_x2plus")], target_scale=2),
        "realesrgan_x4plus": Recipe(ops=[up("model:realesrgan_x4plus")], target_scale=4),
        "realesr_general_x4v3": Recipe(ops=[up("model:realesr_general_x4v3")], target_scale=4),
        "realplksr_x4": Recipe(ops=[up("model:realplksr_nomos2_x4")], target_scale=4),
    }
    return R


def make_input(mp: float, tmp: Path) -> Path:
    """Natural-image content at the requested size (tiled sample photo + mild noise), saved as JPEG q90."""
    from skimage import data
    base = np.concatenate([data.astronaut()[:400], data.coffee()[:400, :512], data.astronaut()[::-1][:400]],
                          axis=1).astype(np.float32) / 255
    w = int(round(np.sqrt(mp * 1e6 * 4 / 3)))
    h = int(round(w * 3 / 4))
    reps = (h // base.shape[0] + 1, w // base.shape[1] + 1, 1)
    img = np.tile(base, reps)[:h, :w]
    img = np.clip(img + np.random.default_rng(0).normal(0, 0.01, img.shape), 0, 1).astype(np.float32)
    p = tmp / f"bench_{mp:g}mp.jpg"
    p.write_bytes(encode_image(img, None, "JPEG", jpeg_quality=90))
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="0.5,1,2,4,8,12,16,24")
    ap.add_argument("--routes", default="")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--max-predicted", type=float, default=120.0, help="skip combos predicted slower (s)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "reports"))
    ap.add_argument("--no-record", action="store_true", help="do not write the Fast Profile")
    a = ap.parse_args()
    if a.quick:
        a.sizes, a.repeats = "0.25,0.5", 2
        a.routes = a.routes or "noop,fbcnn,lanczos_x2,realesr_general_x4v3"
    from ips.core.db import DB
    from ips.core.engine import Studio
    from ips.core.perf import predict_seconds
    from ips.settings import Settings
    tmp = Path(tempfile.mkdtemp(prefix="ips_bench_"))
    s = Settings()
    s.output_root = str(tmp / "out")
    st = Studio(s, DB(tmp / "bench.sqlite"))
    dev = st.device
    print(f"device: {dev.name} ({dev.kind}) torch {dev.torch_version} cuda {dev.cuda_version} "
          f"cap {dev.capability} vram {dev.total_vram_mb} MB")
    R = routes()
    names = [n for n in (a.routes.split(",") if a.routes else R) if n in R]
    sizes = [float(x) for x in a.sizes.split(",")]
    results: dict[str, dict[str, dict]] = {}
    rows = []
    rate: dict[str, list[float]] = {}
    for mp in sizes:
        path = make_input(mp, tmp)
        src = st.inspect(path, record=False)
        for n in names:
            r = R[n].clone()
            pred = predict_seconds(r, src.image.width, src.image.height, dev.kind, dev.name, st.profile)
            if pred > a.max_predicted:
                print(f"skip {n} @ {mp} MP (prior estimate {pred:.0f} s > {a.max_predicted:.0f} s)")
                rows.append({"route": n, "mp": mp, "skipped": f"predicted {pred:.0f}s"})
                continue
            totals, warms, loads, vram = [], [], [], []
            for i in range(a.repeats + 1):
                rr = r.clone()
                rr.quality_override = True
                rr.created_by = f"bench{i}"
                rr.rationale = [f"bench {time.time()}"]
                # unique recipe per run so the master cache is never reused (measure real work)
                rr.ops = [Operation(**{**o.__dict__}) for o in rr.ops] or rr.ops
                if rr.ops:
                    rr.ops[0].params = {**rr.ops[0].params, "_bench_run": i}
                if st.device.kind == "cuda":
                    import torch
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                o = st.prepare(path, profiles=("archive",), recipe=rr, src=src)
                if st.device.kind == "cuda":
                    import torch
                    torch.cuda.synchronize()
                    vram.append(torch.cuda.max_memory_allocated() / 2**20)
                total = time.perf_counter() - t0 + src.timings_ms.get("diagnose", 0) / 1000 \
                    + src.timings_ms.get("decode", 0) / 1000
                load = o.timings_ms.get("model_load_total_ms", 0) / 1000
                (totals if i == 0 else warms).append(total)
                loads.append(load)
            warm = sorted(warms)
            p95 = warm[min(len(warm) - 1, int(round(0.95 * (len(warm) - 1))))]
            row = {"route": n, "route_key": route_key(r), "mp": mp, "in_size": [src.image.width, src.image.height],
                   "cold_s": round(totals[0], 3), "cold_load_s": round(loads[0], 3),
                   "warm_p50_s": round(statistics.median(warm), 3), "warm_p95_s": round(p95, 3),
                   "warm_max_s": round(max(warm), 3), "peak_vram_mb": round(max(vram), 1) if vram else None,
                   "within_budget": p95 <= BUDGET_SECONDS}
            rows.append(row)
            results.setdefault(route_key(r), {})[str(bucket(mp))] = {"p95_s": row["warm_p95_s"],
                                                                     "p50_s": row["warm_p50_s"],
                                                                     "max_s": row["warm_max_s"],
                                                                     "peak_vram_mb": row["peak_vram_mb"]}
            out_mp = mp * r.target_scale ** 2
            for op in r.ops:
                if op.model_id:
                    rate.setdefault(op.method, []).append(row["warm_p50_s"] * 1000 / max(out_mp, 1e-3))
            print(f"{n:22s} {mp:5.2f} MP  cold {row['cold_s']:7.2f} s  warm p50 {row['warm_p50_s']:6.2f}  "
                  f"p95 {row['warm_p95_s']:6.2f}  max {row['warm_max_s']:6.2f}  VRAM {row['peak_vram_mb']} MB  "
                  f"{'OK' if row['within_budget'] else 'OVER 15 s'}", flush=True)
    meta = {"platform": platform.platform(), "torch": dev.torch_version, "cuda": dev.cuda_version,
            "capability": dev.capability, "vram_mb": dev.total_vram_mb, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "repeats": a.repeats, "budget_s": BUDGET_SECONDS}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = "".join(ch if ch.isalnum() else "_" for ch in dev.name)[:40]
    rep = out / f"bench_{tag}.json"
    rep.write_text(json.dumps({"device": dev.__dict__, "meta": meta, "rows": rows}, indent=2), "utf-8")
    print(f"report: {rep}")
    if not a.no_record and dev.kind == "cuda":
        FastProfile().record(dev.name, results, {k: round(statistics.median(v), 2) for k, v in rate.items()}, meta)
        print("Fast Profile updated for", dev.name)
    elif dev.kind != "cuda":
        print("CPU device: results reported but NOT recorded as a Fast Profile (contract is GPU-specific).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

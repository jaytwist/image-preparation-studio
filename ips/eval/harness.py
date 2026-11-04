"""Evaluation harness.

    python -m ips.eval.harness compare  [--tasks upscale_x2,restore_jpeg] [--out reports/eval]
    python -m ips.eval.harness auto     [--out reports/eval]
    python -m ips.eval.harness learn    # derive router_adjustments.json from recorded corrections

Fairness: every candidate runs on the same inputs through the same pipeline with the identity guard
*off* (so model behaviour is measured, not Studio's mitigation). 4× models are resized to the task's
scale with area filtering. No candidate's ranking is presupposed.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..backends.registry import MODELS
from ..core import imageio
from ..core.pipeline import Pipeline
from ..core.recipes import Operation, Recipe, default_op
from ..paths import PROJECT_ROOT
from . import metrics as M

CORPUS = PROJECT_ROOT / "fixtures" / "corpus"


def up(method: str, scale: float, **params) -> Operation:
    op = default_op("upscale", method)
    if not op.model_id:
        op.params["scale"] = scale
    op.params.update(params)
    return op


def cand_upscale(scale: float, screenshot: bool) -> dict[str, Recipe]:
    c: dict[str, Recipe] = {"lanczos": Recipe(ops=[up("classical:lanczos", scale)], target_scale=scale)}
    if screenshot:
        c["nearest"] = Recipe(ops=[up("classical:nearest", int(scale))], target_scale=scale)
    if scale == 2:
        c["swinir_x2"] = Recipe(ops=[up("model:swinir_x2", 2)], target_scale=2)
        c["realesrgan_x2plus"] = Recipe(ops=[up("model:realesrgan_x2plus", 2)], target_scale=2)
    else:
        c["swinir_x2+lanczos"] = Recipe(ops=[up("model:swinir_x2", 2), up("classical:lanczos", scale / 2)],
                                        target_scale=scale)
    for mid in ("realesrgan_x4plus", "realesr_general_x4v3", "realplksr_nomos2_x4"):
        c[mid] = Recipe(ops=[up(f"model:{mid}", 4)], target_scale=scale)
    return {k: v for k, v in c.items() if all(not o.model_id or MODELS[o.model_id].installed for o in v.ops)}


def cand_denoise(sigma: float) -> dict[str, Recipe]:
    c = {"none": Recipe(no_op=True),
         "scunet_real_psnr": Recipe(ops=[default_op("denoise", "model:scunet_real_psnr")]),
         "drunet(σ=est)": Recipe(ops=[Operation("denoise", "model:drunet_color", {"sigma": round(sigma, 1)})]),
         "nlmeans": Recipe(ops=[default_op("denoise", "classical:nlmeans", strength=60)]),
         "swinir_dn15": Recipe(ops=[default_op("denoise", "model:swinir_dn15")])}
    return c


def cand_jpeg(qf: int) -> dict[str, Recipe]:
    return {"none": Recipe(no_op=True),
            f"fbcnn(qf={qf})": Recipe(ops=[Operation("jpeg_cleanup", "model:fbcnn_color", {"qf": qf})]),
            "fbcnn(blind)": Recipe(ops=[Operation("jpeg_cleanup", "model:fbcnn_color", {})]),
            "swinir_car_q10": Recipe(ops=[default_op("jpeg_cleanup", "model:swinir_car_q10")]),
            "swinir_car_q30": Recipe(ops=[default_op("jpeg_cleanup", "model:swinir_car_q30")])}


TASKS = {
    "upscale_x2": ["portrait_x2", "group_x2", "ui_x2"],
    "upscale_x4": ["portrait_x4", "group_x4"],
    "restore_noise": ["dslr_iso6400"],
    "restore_jpeg": ["oldphone_group", "ui_jpeg", "ai_portrait_jpeg"],
}


class Harness:
    def __init__(self, out: Path):
        from ..backends.direct import DirectBackend, device_info
        from ..core.identity import FaceAnalyzer
        self.out = out
        out.mkdir(parents=True, exist_ok=True)
        (out / "visual").mkdir(exist_ok=True)
        self.backend = DirectBackend(device="auto")
        self.fa = FaceAnalyzer.get()
        self.pipe = Pipeline(self.backend, self.fa)
        self.dev = device_info("auto")
        man = json.loads((CORPUS / "manifest.json").read_text())
        self.items = {i["id"]: i for i in man["items"]}

    def load(self, rel: str) -> np.ndarray:
        return imageio.load_image(CORPUS / rel).rgb

    def run_recipe(self, rgb: np.ndarray, r: Recipe) -> tuple[np.ndarray, float, float]:
        if r.no_op or not r.ops:
            return rgb, 0.0, 0.0
        loads_before = dict(self.backend.load_times_ms)
        t0 = time.perf_counter()
        res = self.pipe.run(rgb, None, r, [], identity_check=False, fidelity_guard=False)
        dt = time.perf_counter() - t0
        load = sum(v for k, v in self.backend.load_times_ms.items() if k not in loads_before) / 1000
        return res.rgb, dt - load, load

    # ------------------------------------------------------------------ model comparison
    def compare(self, tasks: list[str]) -> dict[str, Any]:
        from ..core import diagnostics
        results: dict[str, Any] = {"device": self.dev.__dict__, "platform": platform.platform(), "tasks": {}}
        for task in tasks:
            rows = []
            for iid in TASKS[task]:
                it = self.items.get(iid)
                if not it:
                    continue
                src = self.load(it["path"])
                gt = self.load(it["gt"])
                gt_faces = self.fa.detect(gt) if it.get("faces") else []
                if task.startswith("upscale"):
                    scale = 2.0 if task == "upscale_x2" else 4.0
                    cands = cand_upscale(scale, it["content"] == "screenshot")
                elif task == "restore_noise":
                    sig = diagnostics.diagnose(src).features["noise_sigma_luma"]
                    cands = cand_denoise(sig * 0.8)
                else:
                    qf = self.backend.predict_jpeg_qf(src) or 50
                    cands = cand_jpeg(int(round(qf)))
                outs = {}
                for name, r in cands.items():
                    o, secs, load = self.run_recipe(src, r)
                    m = M.full_reference(o, gt)
                    m.update(M.face_identity(o, gt, gt_faces, self.fa))
                    m["seconds_warm"] = round(secs, 3)
                    m["model_load_s"] = round(load, 3)
                    m["out_mp"] = round(o.shape[0] * o.shape[1] / 1e6, 3)
                    m["composite"] = M.composite(m)
                    rows.append({"item": iid, "candidate": name, **{k: (round(v, 4) if isinstance(v, float) else v)
                                                                   for k, v in m.items()}})
                    outs[name] = M.match(o, gt)
                    print(f"[{task}] {iid:18s} {name:24s} psnr {m['psnr']:.2f} ssim {m['ssim']:.4f} "
                          f"ΔE {m['delta_e']:.2f} tex {m['texture_ratio']:.2f} "
                          f"face {m.get('face_sim_min', float('nan')):.3f} t {secs:.1f}s", flush=True)
                self.contact_sheet(f"{task}__{iid}", gt, outs, gt_faces)
            results["tasks"][task] = rows
        return results

    def contact_sheet(self, name: str, gt: np.ndarray, outs: dict[str, np.ndarray], faces) -> None:
        """Side-by-side crops (face or centre) for human visual review."""
        h, w = gt.shape[:2]
        if faces:
            x0, y0, x1, y1 = [int(v) for v in faces[0].box]
            pad = int(0.25 * (x1 - x0))
            x0, y0, x1, y1 = max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)
        else:
            cw, ch = min(w, 256), min(h, 256)
            x0, y0 = (w - cw) // 2, (h - ch) // 2
            x1, y1 = x0 + cw, y0 + ch
        tiles = [("ground truth", gt)] + list(outs.items())
        size = 256
        cells = []
        for lab, im in tiles:
            c = cv2.resize(im[y0:y1, x0:x1], (size, int(size * (y1 - y0) / max(1, x1 - x0))),
                           interpolation=cv2.INTER_NEAREST)
            c = (np.clip(c, 0, 1) * 255).astype(np.uint8)
            bar = np.full((22, size, 3), 30, np.uint8)
            cv2.putText(bar, lab[:34], (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1, cv2.LINE_AA)
            cells.append(np.vstack([bar, c]))
        hmax = max(c.shape[0] for c in cells)
        cells = [np.pad(c, ((0, hmax - c.shape[0]), (0, 4), (0, 0))) for c in cells]
        sheet = np.hstack(cells)
        cv2.imwrite(str(self.out / "visual" / f"{name}.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))

    def real_world(self) -> list[dict[str, Any]]:
        """No ground truth: ×2 candidates on real images; signals vs *source* + sheets for human review."""
        rows = []
        for iid, it in self.items.items():
            if it["category"] != "real":
                continue
            src = self.load(it["path"])
            faces = self.fa.detect(src)
            cands = cand_upscale(2.0, it["content"] == "screenshot")
            outs = {}
            base_tex = None
            for name, r in cands.items():
                o, secs, load = self.run_recipe(src, r)
                back = M.match(o, src)
                yo = M.luma(o)
                tex = M.hf_energy(yo)
                if name == "lanczos":
                    base_tex = tex
                m = {"item": iid, "candidate": name, "consistency_ssim_vs_source": round(float(
                    M.structural_similarity(M.luma(src), M.luma(back), data_range=1.0)), 4),
                     "texture_vs_lanczos": round(tex / (base_tex or tex), 3), "seconds_warm": round(secs, 2)}
                fi = M.face_identity(o, src, faces, self.fa)
                m.update({k: (round(v, 4) if isinstance(v, float) else v) for k, v in fi.items()})
                rows.append(m)
                outs[name] = cv2.resize(o, (src.shape[1] * 2, src.shape[0] * 2), interpolation=cv2.INTER_AREA) \
                    if o.shape[1] != src.shape[1] * 2 else o
                print(f"[real] {iid:24s} {name:22s} cons {m['consistency_ssim_vs_source']:.3f} "
                      f"tex×{m['texture_vs_lanczos']:.2f} face {fi.get('face_sim_min', float('nan')):.3f}", flush=True)
            ref = cv2.resize(src, (src.shape[1] * 2, src.shape[0] * 2), interpolation=cv2.INTER_NEAREST)
            sf = [type(f)(box=tuple(v * 2 for v in f.box), score=f.score) for f in faces]
            self.contact_sheet(f"real__{iid}", ref, outs, sf)
        return rows


# ----------------------------------------------------------------------------- Auto accuracy

ISSUES = ("jpeg_artifacts", "noise", "color_cast", "faded", "oversharpened", "oversmoothed", "low_resolution")


def auto_accuracy(out: Path) -> dict[str, Any]:
    from ..core.db import DB
    from ..core.engine import Studio
    from ..settings import Settings
    s = Settings()
    s.output_root = str(out / "_tmp_outputs")
    st = Studio(s, DB(out / "_auto_eval.sqlite"))
    man = json.loads((CORPUS / "manifest.json").read_text())["items"]
    diag_rows, route_rows = [], []
    counts = {k: {"tp": 0, "fp": 0, "fn": 0} for k in ISSUES}
    cls_ok = cls_n = 0
    h = Harness(out)
    for it in man:
        if not it.get("synthetic", True) or it["category"] in ("batch", "cutout", "recolor", "metadata"):
            continue
        p = CORPUS / it["path"]
        if not p.is_file():
            continue
        src = st.inspect(p, record=False)
        pred = {i.code for i in src.assessment.issues if i.severity >= 0.15}
        truth = set(it.get("truth_issues", []))
        for k in ISSUES:
            if k in pred and k in truth:
                counts[k]["tp"] += 1
            elif k in pred:
                counts[k]["fp"] += 1
            elif k in truth:
                counts[k]["fn"] += 1
        cls_n += 1
        cls_ok += src.assessment.content_class == it["content"]
        r1 = st.auto_recipe(src, st.request(scale=1))
        exp = it.get("expect", {})
        checks = {}
        if exp.get("no_op"):
            checks["no_op"] = r1.no_op
        if exp.get("forbid_kinds"):
            checks["forbid_kinds"] = not ({o.kind for o in r1.ops} & set(exp["forbid_kinds"]))
        diag_rows.append({"item": it["id"], "truth_content": it["content"],
                          "pred_content": src.assessment.content_class, "truth_issues": sorted(truth),
                          "pred_issues": sorted(pred), "auto_x1": r1.summary(), "checks": checks})
        # routing regret at ×1 (restoration decisions) against GT
        if it.get("gt") and it.get("task") is None:
            gt = h.load(it["gt"])
            gt_faces = h.fa.detect(gt) if it.get("faces") else []
            neigh: dict[str, Recipe] = {"noop": Recipe(no_op=True), "auto": r1}
            for i, op in enumerate(r1.ops):
                ab = r1.clone()
                ab.ops = [o for j, o in enumerate(ab.ops) if j != i]
                neigh[f"auto − {op.kind}"] = ab
                from ..core.recipes import METHODS
                for alt in METHODS.get(op.kind, []):
                    if alt == op.method or (alt.startswith("model:") and (not MODELS[alt[6:]].installed or
                                                                           MODELS[alt[6:]].status == "reject")):
                        continue
                    if alt in ("model:swinir_dn15", "model:swinir_car_q10", "model:swinir_car_q30"):
                        continue  # slow transformer variants are covered by the model comparison
                    sub = r1.clone()
                    sub.ops[i] = default_op(op.kind, alt)
                    neigh[f"auto[{op.kind}→{alt.split(':', 1)[1]}]"] = sub
            scores = {}
            for name, r in neigh.items():
                if name == "auto" and r.no_op:
                    continue
                o, _, _ = h.run_recipe(src.image.rgb, r)
                m = M.full_reference(o, gt)
                m.update(M.face_identity(o, gt, gt_faces, h.fa))
                scores[name] = {"composite": round(M.composite(m), 4), "ssim": round(m["ssim"], 4),
                                "psnr": round(m["psnr"], 2), "delta_e": round(m["delta_e"], 2),
                                "face_sim_min": round(m.get("face_sim_min", float("nan")), 4)}
            if r1.no_op:
                scores["auto"] = dict(scores["noop"])
            best = max(scores, key=lambda k: scores[k]["composite"])
            route_rows.append({"item": it["id"], "auto": r1.summary(), "best": best,
                               "auto_rank": 1 + sum(v["composite"] > scores["auto"]["composite"] + 1e-9
                                                    for v in scores.values()),
                               "n_candidates": len(scores),
                               "regret": round(scores[best]["composite"] - scores["auto"]["composite"], 4),
                               "scores": scores})
            print(f"[auto] {it['id']:20s} auto rank {route_rows[-1]['auto_rank']}/{len(scores)} "
                  f"regret {route_rows[-1]['regret']:.4f} best={best}", flush=True)
    prec = {k: (v["tp"] / max(1, v["tp"] + v["fp"]), v["tp"] / max(1, v["tp"] + v["fn"])) for k, v in counts.items()}
    corr = [dict(r) for r in st.db.query("SELECT content_class, choice FROM corrections")]
    rep = {"diagnosis": {"per_issue": {k: {**counts[k], "precision": round(p, 3), "recall": round(r, 3)}
                                       for k, (p, r) in prec.items()},
                         "content_accuracy": round(cls_ok / max(1, cls_n), 3), "items": diag_rows},
           "routing": {"items": route_rows,
                       "top1_rate": round(sum(r["auto_rank"] == 1 for r in route_rows) / max(1, len(route_rows)), 3),
                       "mean_regret": round(float(np.mean([r["regret"] for r in route_rows])) if route_rows else 0, 4)},
           "user_corrections": {"n": len(corr), "note": "user preference records (real-world), separate from the "
                                                        "objective synthetic benchmark"}}
    return rep


def learn(min_n: int = 5, suppress_below: float = 0.3) -> dict[str, Any]:
    """Derive per-(content class, op kind) suppressions from recorded manual corrections."""
    from ..core.db import DB
    from ..paths import app_data_dir
    db = DB()
    stats: dict[str, dict[str, list[int]]] = {}
    for row in db.query("SELECT content_class, auto_recipe_json, final_recipe_json FROM corrections"):
        a = Recipe.from_json(json.loads(row["auto_recipe_json"]))
        f = Recipe.from_json(json.loads(row["final_recipe_json"]))
        fk = {o.kind for o in f.ops if o.enabled}
        for o in a.ops:
            stats.setdefault(row["content_class"], {}).setdefault(o.kind, []).append(int(o.kind in fk))
    adj: dict[str, Any] = {}
    for cls, kinds in stats.items():
        for k, v in kinds.items():
            acc = sum(v) / len(v)
            adj.setdefault(cls, {})[k] = {"n": len(v), "acceptance": round(acc, 3),
                                          "suppress": len(v) >= min_n and acc < suppress_below}
    (app_data_dir() / "router_adjustments.json").write_text(json.dumps(adj, indent=2), "utf-8")
    return adj


# ----------------------------------------------------------------------------- report writing

def write_compare_md(res: dict[str, Any], real: list[dict[str, Any]], path: Path) -> None:
    L = [f"# Model comparison ({res['device']['name']}, {res['device']['kind']})", "",
         "Objective full-reference metrics on **synthetic** fixtures (ground truth available). Columns are "
         "reported separately; `composite` is only a development ranking heuristic. Times are warm "
         "(model load excluded) on the device above — CPU numbers say nothing about the RTX 5070 Ti.", ""]
    cols = ["candidate", "psnr", "ssim", "delta_e", "texture_ratio", "structure_change", "blockiness",
            "face_sim_min", "face_texture_ratio", "landmark_drift_max", "seconds_warm", "composite"]
    for task, rows in res["tasks"].items():
        L.append(f"## {task}")
        for iid in dict.fromkeys(r["item"] for r in rows):
            L.append(f"\n**{iid}**\n")
            L.append("| " + " | ".join(cols) + " |")
            L.append("|" + "---|" * len(cols))
            for r in sorted((r for r in rows if r["item"] == iid), key=lambda r: -r["composite"]):
                L.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
        L.append("")
    if real:
        L += ["## Real-world images (no ground truth)", "",
              "Fidelity *assessment* only: consistency = SSIM of the downscaled output vs the source; texture vs "
              "Lanczos > 1 means added high-frequency content (which may be real detail or invention); face "
              "similarity is vs the source. Requires human review of `visual/real__*.png`.", "",
              "| item | candidate | consistency_ssim_vs_source | texture_vs_lanczos | face_sim_min | seconds_warm |",
              "|---|---|---|---|---|---|"]
        for r in real:
            L.append(f"| {r['item']} | {r['candidate']} | {r['consistency_ssim_vs_source']} | {r['texture_vs_lanczos']}"
                     f" | {r.get('face_sim_min', '')} | {r['seconds_warm']} |")
    path.write_text("\n".join(L), "utf-8")


def write_auto_md(rep: dict[str, Any], path: Path) -> None:
    d, r = rep["diagnosis"], rep["routing"]
    L = ["# Auto Accuracy Benchmark", "", "Synthetic fixtures with injected (known) degradations. Objective only; "
         "user preferences are tracked separately from recorded corrections.", "",
         f"Content-class accuracy: **{d['content_accuracy']:.0%}**", "",
         "| issue | TP | FP | FN | precision | recall |", "|---|---|---|---|---|---|"]
    for k, v in d["per_issue"].items():
        L.append(f"| {k} | {v['tp']} | {v['fp']} | {v['fn']} | {v['precision']} | {v['recall']} |")
    L += ["", "## Expectation checks (×1)", "", "| item | truth issues | predicted | Auto ×1 | checks |",
          "|---|---|---|---|---|"]
    for it in d["items"]:
        L.append(f"| {it['item']} | {', '.join(it['truth_issues'])} | {', '.join(it['pred_issues'])} | "
                 f"{it['auto_x1']} | {it['checks']} |")
    L += ["", "## Routing vs neighbourhood (ablations/substitutions), ×1", "",
          f"Auto ranked first in **{r['top1_rate']:.0%}** of items; mean composite regret **{r['mean_regret']}**.", "",
          "| item | Auto | best candidate | Auto rank | regret |", "|---|---|---|---|---|"]
    for it in r["items"]:
        L.append(f"| {it['item']} | {it['auto']} | {it['best']} | {it['auto_rank']}/{it['n_candidates']} | "
                 f"{it['regret']} |")
    L += ["", f"User correction records: {rep['user_corrections']['n']}"]
    path.write_text("\n".join(L), "utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["compare", "auto", "learn", "real"])
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "reports" / "eval"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.cmd == "compare":
        h = Harness(out)
        res = h.compare([t for t in a.tasks.split(",") if t])
        real = h.real_world()
        res["real_world"] = real
        (out / "model_comparison.json").write_text(json.dumps(res, indent=2, default=str), "utf-8")
        write_compare_md(res, real, out / "MODEL_COMPARISON.md")
    elif a.cmd == "real":
        h = Harness(out)
        real = h.real_world()
        (out / "real_world.json").write_text(json.dumps(real, indent=2), "utf-8")
    elif a.cmd == "auto":
        rep = auto_accuracy(out)
        (out / "auto_accuracy.json").write_text(json.dumps(rep, indent=2, default=str), "utf-8")
        write_auto_md(rep, out / "AUTO_ACCURACY.md")
    else:
        print(json.dumps(learn(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

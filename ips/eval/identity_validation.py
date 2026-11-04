"""Validate identity review thresholds on UNSEEN faces (CodeFormer cropped faces; never used for tuning).

For each face (512² ground truth): ×0.5 source → ×2 by each upscaler; beauty-filtered source; JPEG q30
source cleaned by FBCNN (legitimate restoration, must not be flagged). Reports Studio review-flag
rates per method using the *current* thresholds in ips/core/identity.py.

    python -m ips.eval.identity_validation
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..core import imageio
from ..core.identity import FaceAnalyzer, compare_faces
from ..core.ops import resize_scale
from ..paths import PROJECT_ROOT
from .fixtures import down, jpeg_roundtrip
from .holdout import beauty_filter

SRC = PROJECT_ROOT / "fixtures" / "_downloads" / "identity_val"
SRC2 = PROJECT_ROOT / "fixtures" / "_downloads" / "identity_val2"   # second, untouched face set


def main(src: Path = SRC, tag: str = "") -> int:
    from ..backends.direct import DirectBackend
    be = DirectBackend(device="auto")
    fa = FaceAnalyzer.get()
    methods = {
        "lanczos": lambda x: resize_scale(x, 2),
        "swinir_x2": lambda x: be.run("swinir_x2", x, {}),
        "realesrgan_x2plus": lambda x: be.run("realesrgan_x2plus", x, {}),
        "realesr_general_x4v3": lambda x: resize_scale(be.run("realesr_general_x4v3", x, {"denoise_strength": 0.5}), 0.5),
        "realplksr_nomos2_x4": lambda x: resize_scale(be.run("realplksr_nomos2_x4", x, {}), 0.5),
    }
    res: dict[str, list] = {k: [] for k in list(methods) + ["beauty_filter_1x", "fbcnn_on_q30_1x"]}
    for p in sorted(src.glob("*.png")):
        gt = imageio.load_image(p).rgb
        half = down(gt, 0.5)
        faces = fa.detect(half)
        if not faces:
            print("no face detected", p.name)
            continue
        for k, fn in methods.items():
            rep = compare_faces(half, fn(half), faces, fa)
            res[k].append({"face": p.stem, "flagged": bool(rep.flags), "sim": rep.worst_similarity, "flags": rep.flags})
        fgt = fa.detect(gt)
        rep = compare_faces(gt, beauty_filter(gt), fgt, fa)
        res["beauty_filter_1x"].append({"face": p.stem, "flagged": bool(rep.flags), "sim": rep.worst_similarity,
                                        "flags": rep.flags})
        jq = jpeg_roundtrip(gt, 30)
        fj = fa.detect(jq)
        rep = compare_faces(jq, be.run("fbcnn_color", jq, {"qf": 30}), fj, fa, artifact_removal=True)
        res["fbcnn_on_q30_1x"].append({"face": p.stem, "flagged": bool(rep.flags), "sim": rep.worst_similarity,
                                       "flags": rep.flags})
        print(p.name, {k: v[-1]["flagged"] for k, v in res.items() if v}, flush=True)
    summary = {k: {"n": len(v), "flag_rate": round(float(np.mean([x["flagged"] for x in v])), 3) if v else None,
                   "sim_median": round(float(np.median([x["sim"] for x in v if x["sim"] is not None])), 4) if v else None}
               for k, v in res.items()}
    out = PROJECT_ROOT / "reports" / "eval" / f"identity_validation{tag}.json"
    out.write_text(json.dumps({"summary": summary, "detail": res}, indent=2, default=float), "utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    import sys
    if "--set2" in sys.argv:
        raise SystemExit(main(SRC2, "_set2"))
    raise SystemExit(main())

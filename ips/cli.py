"""Command-line interface (also used by tests and the benchmark harness).

    python -m ips.cli inspect <file>
    python -m ips.cli prepare <files/dirs...> [--profiles archive,share,ai_ready] [--mode faithful] [--scale auto]
    python -m ips.cli batch <dirs/files...> --name NAME [--profiles ...] [--resume BATCH_ID]
    python -m ips.cli cutout <file> [--model u2net_human_seg]
    python -m ips.cli recolor <file> --x X --y Y --color "#3366cc" [--tolerance 14]
    python -m ips.cli env
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time


def _studio(args):
    from .core.engine import Studio
    from .settings import Settings
    s = Settings.load()
    if getattr(args, "output_root", None):
        s.output_root = args.output_root
    if getattr(args, "device", None):
        s.device = args.device
    return Studio(s)


def cmd_inspect(args) -> int:
    st = _studio(args)
    src = st.inspect(args.file, record=False)
    r = st.auto_recipe(src, st.request(mode=args.mode, scale=_scale(args.scale), look=args.look))
    out = {"file": str(src.path), "sha256": src.sha256, "format": src.image.source_format,
           "size": [src.image.width, src.image.height], "bit_depth": src.image.bit_depth,
           "metadata": src.summary, "assessment": src.assessment.to_json(), "timings_ms": src.timings_ms,
           "auto_recipe": r.to_json(), "summary": r.summary()}
    print(json.dumps(out, indent=2, default=str))
    return 0


def _scale(v):
    if v in (None, "auto"):
        return "auto"
    return float(v)


def cmd_prepare(args) -> int:
    from .core.jobs import enumerate_inputs
    st = _studio(args)
    profiles = tuple(p.strip() for p in args.profiles.split(",") if p.strip())
    req = st.request(mode=args.mode, scale=_scale(args.scale), target_long_edge=args.target_long_edge,
                     quality_override=args.quality_override, look=args.look)
    rc = 0
    for f, rel in enumerate_inputs(args.inputs, recursive=True):
        try:
            o = st.prepare(f, profiles=profiles, req=req, rel_dir=rel)
            print(json.dumps({"file": str(f), "state": o.state, "recipe": o.recipe.summary(),
                              "outputs": [str(x.path) for x in o.outputs],
                              "total_s": round(o.timings_ms["total_job_ms"] / 1000, 3),
                              "flags": o.review_flags}, default=str))
        except Exception as e:
            rc = 1
            print(json.dumps({"file": str(f), "state": "failed", "error": f"{type(e).__name__}: {e}"}))
    return rc


def cmd_batch(args) -> int:
    from .core.jobs import JobManager
    st = _studio(args)
    jm = JobManager(st, on_event=lambda jid, state, info: print(f"[job {jid}] {state} "
                                                                  f"{info.get('error', '')}", flush=True))
    if args.resume:
        bid = int(args.resume)
        jm.verify_completed(bid)
    else:
        profiles = tuple(p.strip() for p in args.profiles.split(",") if p.strip())
        bid = jm.create_batch(args.name or time.strftime("%Y%m%d-%H%M%S"), args.inputs, profiles=profiles,
                              mode=args.mode, scale=_scale(args.scale), target_long_edge=args.target_long_edge,
                              quality_override=args.quality_override, look=args.look)
    print(f"batch {bid}")
    jm.start(bid)
    try:
        jm.wait()
    except KeyboardInterrupt:
        jm.cancel()
        jm.wait(30)
    stats = jm.stats(bid)
    jp, cp = jm.write_report(bid)
    print(json.dumps({"batch": bid, "stats": stats, "report": str(jp), "csv": str(cp)}, indent=2, default=str))
    return 0 if not stats["states"].get("failed") else 2


def cmd_cutout(args) -> int:
    from .core.recipes import Recipe
    from .quickedit.matting import Matting, adjust_edge, decontaminate
    st = _studio(args)
    src = st.inspect(args.file)
    a = Matting.person_mask(src.image.rgb, args.model)
    a = adjust_edge(a, args.feather, args.shift)
    rgb = decontaminate(src.image.rgb, a)
    r = Recipe(created_by="quickedit:cutout", rationale=[f"cutout model {args.model}"])
    out = st.export(src, rgb, a, r, "cutout", label=f"cutout:{args.model}", edited=True, generative=False)
    print(json.dumps({"output": str(out.path), "audit_verified": out.audit.get("verified")}))
    return 0


def cmd_recolor(args) -> int:
    from .core.recipes import Recipe
    from .quickedit.edits import MaskDoc, recolor
    st = _studio(args)
    src = st.inspect(args.file)
    h = args.color.lstrip("#")
    target = tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    md = MaskDoc(src.image.height, src.image.width)
    md.magic_select(src.image.rgb, args.x, args.y, tolerance=args.tolerance)
    md.feather(1.5)
    out_rgb = recolor(src.image.rgb, md.mask, target, strength=1.0)
    r = Recipe(created_by="quickedit:recolor", rationale=[f"recolor to {args.color} at ({args.x},{args.y})"])
    o = st.export(src, out_rgb, src.image.alpha, r, "edited", label=f"recolor:{args.color}", edited=True)
    print(json.dumps({"output": str(o.path), "mask_fraction": float(md.mask.mean())}))
    return 0


def cmd_env(args) -> int:
    from tools.inspect_env import collect  # type: ignore
    print(json.dumps(collect(), indent=2, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="ips")
    ap.add_argument("--output-root")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--mode", default=None, choices=["faithful", "balanced"])
        p.add_argument("--scale", default="auto")
        p.add_argument("--target-long-edge", type=int, default=None)
        p.add_argument("--quality-override", action="store_true")
        p.add_argument("--look", choices=["natural", "enhanced", "vivid"], default=None)

    p = sub.add_parser("inspect")
    p.add_argument("file")
    common(p)
    p.set_defaults(fn=cmd_inspect)
    p = sub.add_parser("prepare")
    p.add_argument("inputs", nargs="+")
    p.add_argument("--profiles", default="archive")
    common(p)
    p.set_defaults(fn=cmd_prepare)
    p = sub.add_parser("batch")
    p.add_argument("inputs", nargs="*")
    p.add_argument("--name")
    p.add_argument("--resume")
    p.add_argument("--profiles", default="archive")
    common(p)
    p.set_defaults(fn=cmd_batch)
    p = sub.add_parser("cutout")
    p.add_argument("file")
    p.add_argument("--model", default="u2net_human_seg", choices=["u2net_human_seg", "isnet_general"])
    p.add_argument("--feather", type=float, default=0.0)
    p.add_argument("--shift", type=int, default=0)
    p.set_defaults(fn=cmd_cutout)
    p = sub.add_parser("recolor")
    p.add_argument("file")
    p.add_argument("--x", type=int, required=True)
    p.add_argument("--y", type=int, required=True)
    p.add_argument("--color", required=True)
    p.add_argument("--tolerance", type=float, default=14.0)
    p.set_defaults(fn=cmd_recolor)
    p = sub.add_parser("env")
    p.set_defaults(fn=cmd_env)
    args = ap.parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    sys.exit(main())

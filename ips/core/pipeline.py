"""Recipe execution with per-stage timing, masks, cancellation and the Faithful fidelity guard."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from . import ops as cops
from .diagnostics import Face
from .recipes import Operation, Recipe

log = logging.getLogger("ips.pipeline")

GLOBAL_COLOR_KINDS = ("white_balance", "exposure", "tone")


@dataclass
class ProcessResult:
    rgb: np.ndarray
    alpha: np.ndarray | None
    recipe: Recipe
    timings_ms: dict[str, float] = field(default_factory=dict)
    model_load_ms: dict[str, float] = field(default_factory=dict)
    identity: dict[str, Any] | None = None
    guard_actions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    peak_vram_mb: float | None = None

    @property
    def review_flags(self) -> list[str]:
        f = list(self.recipe.flags)
        if self.identity:
            f.extend(self.identity.get("flags", []))
        f.extend(self.guard_actions)
        return f


class Cancelled(Exception):
    pass


def _apply_classical(op: Operation, img: np.ndarray) -> np.ndarray:
    m = op.method
    p = op.params
    if m == "classical:neutral_cast":
        return cops.neutral_cast(img, **p)
    if m == "classical:levels":
        return cops.levels(img, **p)
    if m == "classical:contrast_saturation":
        return cops.contrast_saturation(img, **p)
    if m == "classical:halo_free_unsharp":
        return cops.halo_free_unsharp(img, **p)
    if m == "classical:unsharp_deconv":
        return cops.unsharp_deconv(img, **p)
    if m == "classical:nlmeans":
        return cops.nlmeans(img, **p)
    if m == "classical:lanczos":
        return cops.resize_scale(img, float(p.get("scale", 2.0)), "lanczos")
    if m == "classical:nearest":
        return cops.resize_scale(img, float(p.get("scale", 2)), "nearest")
    if m == "classical:crop_rotate":
        return cops.crop_rotate(img, **p)
    raise ValueError(f"Unknown classical method {m}")


def _region_mask(spec: str, h: int, w: int, scale: float) -> np.ndarray:
    x0, y0, x1, y1 = [float(v) * scale for v in spec.split(":", 1)[1].split(",")]
    m = np.zeros((h, w), np.float32)
    m[int(max(0, y0)):int(min(h, y1)), int(max(0, x0)):int(min(w, x1))] = 1.0
    return cops.feather(m, max(2.0, 0.02 * max(h, w)))


class Pipeline:
    def __init__(self, backend=None, face_analyzer=None):
        self._backend = backend
        self._fa = face_analyzer

    @property
    def backend(self):
        if self._backend is None:
            from ..backends.direct import get_backend
            self._backend = get_backend()
        return self._backend

    @property
    def face_analyzer(self):
        if self._fa is None:
            from .identity import FaceAnalyzer
            self._fa = FaceAnalyzer.get()
        return self._fa

    def _run_comfy(self, op: Operation, img: np.ndarray, cancel) -> np.ndarray:
        from ..backends.comfy.client import ComfyClient
        from ..settings import Settings
        st = Settings.load()
        if not st.comfy_enabled:
            raise RuntimeError("ComfyUI routes are disabled in Settings")
        kind, name = op.method.split(":", 2)[1:]
        if kind != "upscale":
            raise ValueError(f"Unsupported ComfyUI route {op.method}")
        from ..backends.direct import GPU_LOCK
        with GPU_LOCK:  # never run a Studio GPU model and a Studio ComfyUI job at the same time
            return ComfyClient(st.comfy_url).upscale_with_model(img, name, st.comfy_vram_min_free_mb, cancel=cancel)

    # ------------------------------------------------------------------ main
    def run(self, rgb: np.ndarray, alpha: np.ndarray | None, recipe: Recipe, faces: list[Face] | None = None, *,
            cancel: Callable[[], bool] | None = None, progress: Callable[[str, float], None] | None = None,
            identity_check: bool = True, fidelity_guard: bool = True) -> ProcessResult:
        faces = list(faces or [])
        t_all = time.perf_counter()
        timings: dict[str, float] = {}
        loads: dict[str, float] = {}
        warnings: list[str] = []
        src_for_identity = rgb
        cur = rgb
        scale = 1.0
        ops_list = [o for o in recipe.sorted_ops() if o.enabled]
        if hasattr(self.backend if any(o.model_id for o in ops_list) else None, "reset_peak"):
            self.backend.reset_peak()
        n_ops = max(1, len(ops_list))
        for i, op in enumerate(ops_list):
            if cancel and cancel():
                raise Cancelled()
            t0 = time.perf_counter()
            if progress:
                progress(f"{op.kind}: {op.label()}", i / n_ops)
            inp = cur
            if op.model_id:
                be = self.backend
                was_loaded = be.is_loaded(op.model_id, op.params)
                tl = time.perf_counter()
                if not was_loaded:
                    be.get(op.model_id, op.params)
                    loads[op.model_id] = (time.perf_counter() - tl) * 1e3
                try:
                    from ..backends.direct import CancelledError
                    out = be.run(op.model_id, inp, op.params, cancel=cancel,
                                 progress=(lambda f, k=op.kind, i=i: progress(k, (i + f) / n_ops)) if progress else None)
                except CancelledError as e:
                    raise Cancelled() from e
                if op.kind == "upscale":
                    from ..backends.registry import MODELS
                    want = float(op.params.get("scale", MODELS[op.model_id].scale) or MODELS[op.model_id].scale)
                    remaining = recipe.target_scale / scale
                    eff = min(MODELS[op.model_id].scale, remaining) if remaining > 1 else MODELS[op.model_id].scale
                    if abs(eff - MODELS[op.model_id].scale) > 1e-6:
                        h0, w0 = inp.shape[:2]
                        out = cops.resize_to(out, int(round(w0 * eff)), int(round(h0 * eff)))
                    _ = want
            elif op.method.startswith("comfy:"):
                out = self._run_comfy(op, inp, cancel)
                if op.kind == "upscale":
                    remaining = recipe.target_scale / scale
                    sc = out.shape[1] / inp.shape[1]
                    if remaining < sc - 1e-6:
                        h0, w0 = inp.shape[:2]
                        out = cops.resize_to(out, int(round(w0 * remaining)), int(round(h0 * remaining)))
                warnings.append(f"{op.method}: ran on local ComfyUI (8-bit PNG transfer; model character "
                                "not benchmarked by Studio)")
            else:
                out = _apply_classical(op, inp)
            if op.kind == "geometry":
                if alpha is not None:
                    alpha = cops.crop_rotate(alpha[..., None], **op.params)[..., 0]
                src_for_identity = out
                faces = self.face_analyzer.detect(out) if faces else faces
                cur = out
                timings[f"{i:02d}_{op.kind}"] = (time.perf_counter() - t0) * 1e3
                continue
            oh, ow = out.shape[:2]
            step_scale = ow / inp.shape[1]
            inp_res = inp if step_scale == 1.0 else cops.resize_to(inp, ow, oh)
            if op.blend < 1.0:
                out = cops.blend(inp_res, out, float(op.blend))
            new_scale = scale * step_scale
            if op.mask == "protect_faces" and faces:
                protect = float(op.params.get("face_protect", 0.5))
                m = cops.face_mask((oh, ow), [f.box for f in faces], scale=new_scale)
                out = cops.blend(out, inp_res, m * protect)
            elif op.mask == "faces_only" and faces:
                m = cops.face_mask((oh, ow), [f.box for f in faces], scale=new_scale)
                out = cops.blend(inp_res, out, m)
            elif op.mask == "protect_text":
                m = cops.text_mask(inp_res)
                out = cops.blend(out, inp_res, m)
            elif op.mask and op.mask.startswith("region:"):
                m = _region_mask(op.mask, oh, ow, new_scale)
                out = cops.blend(inp_res, out, m)
            cur = np.clip(out, 0, 1).astype(np.float32)
            scale = new_scale
            timings[f"{i:02d}_{op.kind}:{op.method.split(':', 1)[1]}"] = (time.perf_counter() - t0) * 1e3
        # exact final geometry for the requested scale
        th = int(round(src_for_identity.shape[0] * recipe.target_scale))
        tw = int(round(src_for_identity.shape[1] * recipe.target_scale))
        if (cur.shape[0], cur.shape[1]) != (th, tw):
            cur = cops.resize_to(cur, tw, th)
        if alpha is not None and (alpha.shape[0], alpha.shape[1]) != (th, tw):
            import cv2
            alpha = np.clip(cv2.resize(alpha, (tw, th), interpolation=cv2.INTER_CUBIC), 0, 1).astype(np.float32)
        res = ProcessResult(rgb=cur, alpha=alpha, recipe=recipe, timings_ms=timings, model_load_ms=loads,
                            warnings=warnings)
        # identity signals + fidelity guard
        uses_model = any(o.model_id for o in ops_list)
        if identity_check and faces and uses_model:
            t0 = time.perf_counter()
            from .identity import compare_faces
            denoised = any(o.kind == "denoise" for o in ops_list)
            rep = compare_faces(src_for_identity, cur, faces, self.face_analyzer, denoise_applied=denoised)
            if fidelity_guard and recipe.mode == "faithful" and rep.flags:
                bad = [fd for fd in rep.faces if fd.flags and not any("too small" in x for x in fd.flags)]
                if bad:
                    base = src_for_identity
                    for o in ops_list:
                        if o.kind in GLOBAL_COLOR_KINDS:
                            base = _apply_classical(o, base)
                    base = cops.resize_to(base, tw, th)
                    m = cops.face_mask((th, tw), [fd.box for fd in bad], scale=th / src_for_identity.shape[0])
                    cur = cops.blend(cur, base, m * 0.6)
                    res.rgb = cur
                    res.guard_actions.append(
                        f"Fidelity guard: {len(bad)} flagged face region(s) blended 60% toward a conservative "
                        "rendering (classical colour ops + Lanczos). Review recommended.")
                    rep2 = compare_faces(src_for_identity, cur, faces, self.face_analyzer, denoise_applied=denoised)
                    d = rep2.to_json()
                    d["before_guard"] = rep.to_json()
                    res.identity = d
                else:
                    res.identity = rep.to_json()
            else:
                res.identity = rep.to_json()
            timings["identity_check"] = (time.perf_counter() - t0) * 1e3
        res.timings_ms["pipeline_total"] = (time.perf_counter() - t_all) * 1e3
        if uses_model:
            res.peak_vram_mb = self.backend.peak_vram_mb()
        return res

    def preview_region(self, rgb: np.ndarray, recipe: Recipe, box: tuple[int, int, int, int],
                       faces: list[Face] | None = None, margin: int = 32, **kw) -> tuple[np.ndarray, np.ndarray]:
        """Process only *box* (+margin context). Returns (source_crop, processed_crop) aligned."""
        h, w = rgb.shape[:2]
        x0, y0, x1, y1 = box
        mx0, my0 = max(0, x0 - margin) // 8 * 8, max(0, y0 - margin) // 8 * 8
        mx1, my1 = min(w, x1 + margin), min(h, y1 + margin)
        crop = rgb[my0:my1, mx0:mx1]
        local_faces = []
        for f in faces or []:
            bx = (f.box[0] - mx0, f.box[1] - my0, f.box[2] - mx0, f.box[3] - my0)
            lm = [(a - mx0, b - my0) for a, b in f.landmarks] if f.landmarks else None
            local_faces.append(Face(box=bx, score=f.score, landmarks=lm))
        r = recipe.clone()
        r.ops = [o for o in r.ops if o.kind != "geometry"]
        res = self.run(crop, None, r, local_faces, **kw)
        s = r.target_scale
        ix0, iy0 = int((x0 - mx0) * s), int((y0 - my0) * s)
        ix1, iy1 = int((x1 - mx0) * s), int((y1 - my0) * s)
        return rgb[y0:y1, x0:x1], res.rgb[iy0:iy1, ix0:ix1]

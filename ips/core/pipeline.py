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
    provenance: list[dict[str, Any]] = field(default_factory=list)   # external/generative engine details

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
    if m == "classical:clarity":
        return cops.clarity(img, **p)
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


def face_reference(img: np.ndarray, faces: list[Face] | None, min_side: int = 512) -> np.ndarray | None:
    """Padded crop of the largest face (identity anchor for generative models), upsampled to >= *min_side*."""
    if not faces:
        return None
    f = max(faces, key=lambda x: (x.box[2] - x.box[0]) * (x.box[3] - x.box[1]))
    x0, y0, x1, y1 = (float(v) for v in f.box)
    h, w = img.shape[:2]
    cx, cy, side = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) * 1.6
    if side < 24:
        return None
    a0, b0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    a1, b1 = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
    crop = img[b0:b1, a0:a1]
    if crop.size == 0:
        return None
    k = max(1.0, min_side / max(1, min(crop.shape[:2])))
    if k > 1.0:
        crop = cops.resize_to(crop, int(round(crop.shape[1] * k)), int(round(crop.shape[0] * k)))
    return np.ascontiguousarray(crop)


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

    def _run_comfy(self, op: Operation, img: np.ndarray, cancel, faces: list[Face] | None = None,
                   provenance: list[dict[str, Any]] | None = None) -> np.ndarray:
        from ..backends.comfy.client import ComfyClient
        from ..settings import Settings
        st = Settings.load()
        if not st.comfy_enabled:
            raise RuntimeError("ComfyUI routes are disabled in Settings")
        kind, name = op.method.split(":", 2)[1:]
        from ..backends.direct import GPU_LOCK
        client = ComfyClient(st.comfy_url)
        if kind == "upscale":
            with GPU_LOCK:  # never run a Studio GPU model and a Studio ComfyUI job at the same time
                return client.upscale_with_model(img, name, st.comfy_vram_min_free_mb, cancel=cancel)
        if kind == "generative":
            from ..backends.comfy import qwen
            p = op.params
            face = face_reference(img, faces) if p.get("face_reference", True) and faces else None
            res = int(p.get("resolution") or qwen.resolution_for(img.shape[1], img.shape[0], 1.0))
            # release Studio's own GPU models first so the diffusion model has the VRAM (never touches others')
            from ..backends import direct
            if direct._BACKEND is not None:
                try:
                    direct._BACKEND.unload_all()
                except Exception:
                    pass
            with GPU_LOCK:
                out, prov = client.qwen_edit(img, p.get("prompt") or qwen.DEFAULT_PROMPT, face=face,
                                             turbo=bool(p.get("turbo")), strength=float(p.get("strength", 1.0)),
                                             resolution=res, steps=int(p.get("steps", 25)),
                                             cfg=float(p.get("cfg", 1.0)), negative=p.get("negative") or None,
                                             seed=p.get("seed"), min_free_mb=st.comfy_vram_min_free_mb,
                                             cancel=cancel)
            if provenance is not None:
                provenance.append(prov)
            return out
        raise ValueError(f"Unsupported ComfyUI route {op.method}")

    # ------------------------------------------------------------------ main
    def run(self, rgb: np.ndarray, alpha: np.ndarray | None, recipe: Recipe, faces: list[Face] | None = None, *,
            cancel: Callable[[], bool] | None = None, progress: Callable[[str, float], None] | None = None,
            identity_check: bool = True, fidelity_guard: bool = True) -> ProcessResult:
        faces = list(faces or [])
        t_all = time.perf_counter()
        timings: dict[str, float] = {}
        loads: dict[str, float] = {}
        warnings: list[str] = []
        prov: list[dict[str, Any]] = []
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
                out = self._run_comfy(op, inp, cancel, faces, prov)
                if op.kind == "generative":
                    # Qwen generates on a ×32 pixel budget: map back to the exact requested geometry
                    remaining = recipe.target_scale / scale
                    h0, w0 = inp.shape[:2]
                    tw_, th_ = int(round(w0 * remaining)), int(round(h0 * remaining))
                    ar_in, ar_out = w0 / h0, out.shape[1] / out.shape[0]
                    if abs(ar_out / ar_in - 1) > 0.02:
                        warnings.append(f"Generative output aspect {ar_out:.3f} vs source {ar_in:.3f}: resized to the "
                                        "source framing (check edges).")
                    if (out.shape[1], out.shape[0]) != (tw_, th_):
                        out = cops.resize_to(out, tw_, th_)
                elif op.kind == "upscale":
                    remaining = recipe.target_scale / scale
                    sc = out.shape[1] / inp.shape[1]
                    if remaining < sc - 1e-6:
                        h0, w0 = inp.shape[:2]
                        out = cops.resize_to(out, int(round(w0 * remaining)), int(round(h0 * remaining)))
                warnings.append(f"{op.method}: ran on local ComfyUI (8-bit PNG transfer; model character "
                                "not benchmarked by Studio)")
                if op.kind != "generative":
                    recipe.flags.append(f"External ComfyUI model {op.method.split(':', 2)[2]}: identity/texture "
                                        "behaviour not benchmarked — review faces and text.")
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
                            warnings=warnings, provenance=prov)
        # identity signals + fidelity guard
        uses_model = any(o.model_id for o in ops_list)
        if identity_check and faces and ops_list:   # any change to a face is checked, not only model ops
            t0 = time.perf_counter()
            from .identity import compare_faces
            denoised = any(o.kind == "denoise" for o in ops_list)
            artifact = any(o.kind in ("denoise", "jpeg_cleanup") for o in ops_list)
            sharpened = any(o.kind == "sharpen" or o.method == "classical:clarity" for o in ops_list)
            colour = any(o.kind in GLOBAL_COLOR_KINDS for o in ops_list)
            rep = compare_faces(src_for_identity, cur, faces, self.face_analyzer, denoise_applied=denoised,
                                color_changed=colour, artifact_removal=artifact, sharpened=sharpened)
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
                    rep2 = compare_faces(src_for_identity, cur, faces, self.face_analyzer, denoise_applied=denoised,
                                         color_changed=colour, artifact_removal=artifact, sharpened=sharpened)
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

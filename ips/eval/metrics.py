"""Evaluation metrics. Each dimension is reported separately; no single score is treated as truth.

Full-reference (synthetic fixtures with ground truth): PSNR, SSIM, ΔE2000 colour error, texture
ratio vs GT, structural change (edge-map SSIM), residual blockiness, face identity vs GT.
No-reference (real images): identity/texture drift vs *source*, blockiness, noise — fidelity signals only.
LPIPS is computed only if the ``lpips`` package and its backbone weights are installed locally.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from ..core import diagnostics as dg

_LPIPS = None


def _lpips():
    global _LPIPS
    if _LPIPS is None:
        try:
            import lpips  # type: ignore
            _LPIPS = lpips.LPIPS(net="alex", verbose=False)
        except Exception:
            _LPIPS = False
    return _LPIPS or None


def luma(x: np.ndarray) -> np.ndarray:
    return (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]).astype(np.float32)


def match(out: np.ndarray, gt: np.ndarray) -> np.ndarray:
    if out.shape[:2] != gt.shape[:2]:
        out = cv2.resize(out, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_AREA
                         if out.shape[1] > gt.shape[1] else cv2.INTER_CUBIC)
    return np.clip(out, 0, 1).astype(np.float32)


def hf_energy(y: np.ndarray) -> float:
    return float(np.abs(cv2.Laplacian(y, cv2.CV_32F, ksize=1)).mean())


def edge_map(y: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
    m = np.sqrt(gx * gx + gy * gy)
    return m / (m.max() + 1e-6)


def full_reference(out: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    o = match(out, gt)
    yo, yg = luma(o), luma(gt)
    r = {
        "psnr": float(peak_signal_noise_ratio(gt, o, data_range=1.0)),
        "ssim": float(structural_similarity(yg, yo, data_range=1.0)),
        "delta_e": float(np.mean(deltaE_ciede2000(rgb2lab(gt), rgb2lab(o)))),
        "texture_ratio": hf_energy(yo) / (hf_energy(yg) + 1e-6),
        "structure_change": float(1 - structural_similarity(edge_map(yg), edge_map(yo), data_range=1.0)),
        "blockiness": dg.blockiness(yo),
    }
    lp = _lpips()
    if lp is not None:
        import torch
        t = lambda a: torch.from_numpy(a.transpose(2, 0, 1)[None] * 2 - 1).float()
        with torch.no_grad():
            r["lpips"] = float(lp(t(gt), t(o)).item())
    return r


def face_identity(out: np.ndarray, ref: np.ndarray, faces, analyzer) -> dict[str, Any]:
    """Identity of output faces vs reference faces (GT for synthetic, source for real), aligned with the
    reference landmarks scaled to the comparison grid."""
    if not faces:
        return {}
    o = match(out, ref)
    sims, ftex, drift = [], [], []
    out_faces = analyzer.detect(o) if analyzer.det is not None else []
    for f in faces[:8]:
        if not f.landmarks or f.size < 24:
            continue
        er = analyzer.embed(analyzer.align(ref, f.landmarks))
        eo = analyzer.embed(analyzer.align(o, f.landmarks))
        if er is not None and eo is not None:
            sims.append(float(np.dot(er, eo)))
        x0, y0, x1, y1 = [int(v) for v in f.box]
        cr, co = luma(ref[y0:y1, x0:x1]), luma(o[y0:y1, x0:x1])
        if cr.size > 100:
            ftex.append(hf_energy(co) / (hf_energy(cr) + 1e-6))
        if out_faces:
            cx, cy = (f.box[0] + f.box[2]) / 2, (f.box[1] + f.box[3]) / 2
            g = min(out_faces, key=lambda g: ((g.box[0] + g.box[2]) / 2 - cx) ** 2 + ((g.box[1] + g.box[3]) / 2 - cy) ** 2)
            if g.landmarks:
                a, b = np.array(f.landmarks), np.array(g.landmarks)
                drift.append(float(np.linalg.norm(a - b, axis=1).mean() / (np.linalg.norm(a[0] - a[1]) + 1e-6)))
    r: dict[str, Any] = {"faces_evaluated": len(sims)}
    if sims:
        r.update(face_sim_min=float(min(sims)), face_sim_mean=float(np.mean(sims)))
    if ftex:
        r["face_texture_ratio"] = float(np.mean(ftex))
    if drift:
        r["landmark_drift_max"] = float(max(drift))
    r.update(face_regions(o, ref, faces))
    return r


def face_regions(o: np.ndarray, ref: np.ndarray, faces) -> dict[str, float]:
    """Face fidelity beyond embeddings: structure (SSIM), skin colour (ΔE2000 in the central face),
    fine and mid-band skin texture ratios. Worst face reported (min SSIM, max ΔE, texture furthest from 1)."""
    ss, de, fine, mid = [], [], [], []
    h, w = ref.shape[:2]
    for f in faces[:8]:
        if f.size < 24:
            continue
        x0, y0, x1, y1 = [int(round(v)) for v in f.box]
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        if x1 - x0 < 16 or y1 - y0 < 16:
            continue
        lr, lo = luma(ref[y0:y1, x0:x1]), luma(o[y0:y1, x0:x1])
        ss.append(float(structural_similarity(lr, lo, data_range=1.0)))
        cy0, cy1, cx0, cx1 = y0 + (y1 - y0) // 4, y1 - (y1 - y0) // 4, x0 + (x1 - x0) // 4, x1 - (x1 - x0) // 4
        de.append(float(np.mean(deltaE_ciede2000(rgb2lab(ref[cy0:cy1, cx0:cx1]), rgb2lab(o[cy0:cy1, cx0:cx1])))))
        fine.append(hf_energy(lo) / (hf_energy(lr) + 1e-6))
        dog = lambda x: float(np.abs(cv2.GaussianBlur(x, (0, 0), 1.0) - cv2.GaussianBlur(x, (0, 0), 3.0)).mean())
        mid.append(dog(lo) / (dog(lr) + 1e-6))
    if not ss:
        return {}
    far = lambda v: max(v, key=lambda x: abs(np.log(max(x, 1e-3))))
    return {"face_ssim_min": float(min(ss)), "skin_delta_e_max": float(max(de)),
            "face_fine_texture_worst": float(far(fine)), "face_mid_texture_worst": float(far(mid))}


def text_fidelity(out: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """Glyph fidelity on text/UI: SSIM inside the GT text mask and the fraction of glyph pixels whose
    binarised value differs (catches invented/garbled letters that global metrics average away)."""
    from ..core.ops import text_mask
    o = match(out, gt)
    m = text_mask(gt, dilate=1) > 0.5
    if m.sum() < 100:
        return {}
    yg, yo = luma(gt), luma(o)
    ys, xs = np.where(m)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    _, smap = structural_similarity(yg[y0:y1, x0:x1], yo[y0:y1, x0:x1], data_range=1.0, full=True)
    thr = 0.5 * (np.percentile(yg[m], 10) + np.percentile(yg[m], 90))
    bg, bo = yg < thr, yo < thr
    return {"text_ssim": float(smap[m[y0:y1, x0:x1]].mean()), "glyph_error": float((bg != bo)[m].mean())}


def composite(m: dict[str, Any]) -> float:
    """Development-time ranking heuristic (priority order: identity > fidelity > texture/colour).

    Used only to rank candidates in the Auto Accuracy Benchmark; reported alongside raw metrics.
    """
    s = 0.45 * m.get("ssim", 0) + 0.15 * min(1.0, m.get("psnr", 0) / 40)
    s += 0.15 * (1 - min(1.0, m.get("delta_e", 10) / 10))
    s += 0.15 * (1 - min(1.0, abs(np.log(max(1e-3, m.get("texture_ratio", 1.0))))))
    s += 0.10 * (1 - min(1.0, m.get("structure_change", 1)))
    if "glyph_error" in m:
        s -= 1.0 * m["glyph_error"]  # garbled glyphs dominate for text images
    if "face_ssim_min" in m:
        s += 0.10 * m["face_ssim_min"] - 0.01 * m.get("skin_delta_e_max", 0)
    if "face_sim_min" in m:
        s += 0.30 * m["face_sim_min"]
        if m["face_sim_min"] < 0.80:
            s -= 0.5  # identity veto
    return float(s)

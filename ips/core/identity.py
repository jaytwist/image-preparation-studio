"""Face detection and identity-drift *signals*.

Detector/recogniser: InsightFace ``buffalo_l`` ONNX models (SCRFD det_10g + ArcFace w600k_r50) run
directly with onnxruntime (the insightface Python package is not required). NOTE: the buffalo_l
pretrained weights are released by InsightFace for **non-commercial research use**; they are an
optional component (see docs/MODEL_MATRIX.md). Without them, OpenCV's bundled Haar cascade is used for
face *location* only and embedding checks are reported as unavailable.

Nothing here certifies identity. Scores trigger the Review Queue; humans decide.
"""
from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..paths import models_dir
from .diagnostics import Face

ARCFACE_DST = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                        [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


def _providers() -> list[str]:
    import onnxruntime as ort
    avail = ort.get_available_providers()
    order = ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
    return [p for p in order if p in avail] or avail


class FaceAnalyzer:
    _instance: "FaceAnalyzer | None" = None
    _lock = threading.Lock()

    def __init__(self, root: Path | None = None):
        import onnxruntime as ort
        root = root or (models_dir() / "buffalo_l")
        self.det = self.rec = None
        self.name = "haar-cascade"
        so = ort.SessionOptions()
        so.log_severity_level = 3
        det_p, rec_p = root / "det_10g.onnx", root / "w600k_r50.onnx"
        if det_p.is_file():
            self.det = ort.InferenceSession(str(det_p), so, providers=_providers())
            self.det_in = self.det.get_inputs()[0].name
            self.name = "scrfd-10g"
        if rec_p.is_file():
            self.rec = ort.InferenceSession(str(rec_p), so, providers=_providers())
            self.rec_in = self.rec.get_inputs()[0].name
        self._haar = None
        self._run_lock = threading.Lock()

    @classmethod
    def get(cls) -> "FaceAnalyzer":
        with cls._lock:
            if cls._instance is None:
                cls._instance = FaceAnalyzer()
            return cls._instance

    @property
    def has_embeddings(self) -> bool:
        return self.rec is not None

    # ------------------------------------------------------------------ detection
    def detect(self, rgb: np.ndarray, thresh: float = 0.5, max_faces: int = 12) -> list[Face]:
        if self.det is None:
            return self._detect_haar(rgb)
        h, w = rgb.shape[:2]
        size = 640
        s = size / max(h, w)
        nh, nw = int(round(h * s)), int(round(w * s))
        u8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        resized = cv2.resize(u8, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        canvas = np.zeros((size, size, 3), np.uint8)
        canvas[:nh, :nw] = resized
        blob = ((canvas.astype(np.float32) - 127.5) / 128.0).transpose(2, 0, 1)[None]
        with self._run_lock:
            outs = self.det.run(None, {self.det_in: blob})
        outs = [o[0] if o.ndim == 3 else o for o in outs]
        fmc = 3
        strides = (8, 16, 32)
        boxes, scores, kpss = [], [], []
        for i, st in enumerate(strides):
            sc = outs[i].reshape(-1)
            bb = outs[i + fmc].reshape(-1, 4) * st
            kp = outs[i + 2 * fmc].reshape(-1, 10) * st
            fh, fw = size // st, size // st
            centers = np.stack(np.mgrid[:fh, :fw][::-1], axis=-1).astype(np.float32).reshape(-1, 2) * st
            centers = np.repeat(centers, 2, axis=0)
            keep = np.where(sc >= thresh)[0]
            if keep.size == 0:
                continue
            c = centers[keep]
            b = bb[keep]
            boxes.append(np.stack([c[:, 0] - b[:, 0], c[:, 1] - b[:, 1], c[:, 0] + b[:, 2], c[:, 1] + b[:, 3]], 1))
            k = kp[keep].reshape(-1, 5, 2) + c[:, None, :]
            kpss.append(k)
            scores.append(sc[keep])
        if not boxes:
            return []
        B = np.concatenate(boxes) / s
        K = np.concatenate(kpss) / s
        S = np.concatenate(scores)
        idx = cv2.dnn.NMSBoxes([[float(x0), float(y0), float(x1 - x0), float(y1 - y0)] for x0, y0, x1, y1 in B],
                               S.tolist(), thresh, 0.4)
        idx = np.array(idx).reshape(-1)
        faces = []
        for i in idx[:max_faces]:
            x0, y0, x1, y1 = B[i]
            faces.append(Face(box=(float(max(0, x0)), float(max(0, y0)), float(min(w, x1)), float(min(h, y1))),
                              score=float(S[i]), landmarks=[(float(a), float(b)) for a, b in K[i]]))
        faces.sort(key=lambda f: -(f.box[2] - f.box[0]) * (f.box[3] - f.box[1]))
        return faces

    def _detect_haar(self, rgb: np.ndarray) -> list[Face]:
        if self._haar is None:
            self._haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        h, w = rgb.shape[:2]
        s = min(1.0, 1024 / max(h, w))
        g = cv2.cvtColor((np.clip(rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else g
        rects = self._haar.detectMultiScale(g, 1.1, 6, minSize=(24, 24))
        return [Face(box=(x / s, y / s, (x + rw) / s, (y + rh) / s), score=0.5) for (x, y, rw, rh) in rects]

    # ------------------------------------------------------------------ embedding
    def align(self, rgb: np.ndarray, landmarks: list[tuple[float, float]]) -> np.ndarray:
        src = np.array(landmarks, dtype=np.float32)
        M, _ = cv2.estimateAffinePartial2D(src, ARCFACE_DST, method=cv2.LMEDS)
        u8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        return cv2.warpAffine(u8, M, (112, 112), borderValue=0)

    def embed(self, aligned_u8: np.ndarray) -> np.ndarray | None:
        if self.rec is None:
            return None
        blob = ((aligned_u8.astype(np.float32) - 127.5) / 127.5).transpose(2, 0, 1)[None]
        with self._run_lock:
            v = self.rec.run(None, {self.rec_in: blob})[0][0]
        return v / (np.linalg.norm(v) + 1e-9)


# ----------------------------------------------------------------------------- comparison

@dataclass
class FaceDrift:
    index: int
    box: tuple[float, float, float, float]
    embedding_similarity: float | None = None
    landmark_drift_iod: float | None = None
    region_ssim: float | None = None
    fine_texture_ratio: float | None = None   # output/source Laplacian energy at source resolution
    mid_texture_ratio: float | None = None    # output/source DoG(1-3px) energy
    skin_delta_e: float | None = None
    flags: list[str] = field(default_factory=list)


@dataclass
class IdentityReport:
    available: bool
    method: str
    faces: list[FaceDrift] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    worst_similarity: float | None = None
    note: str = ("Signals only: embeddings/landmarks/texture ratios cannot prove identity preservation. "
                 "Flagged results go to the Review Queue for visual inspection.")

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


# thresholds calibrated on the synthetic/real portrait fixtures (see docs/AUTO_RECIPES.md)
SIM_REVIEW = 0.80
LANDMARK_REVIEW = 0.06
SMOOTHING_REVIEW = 0.60
FINE_SMOOTHING_REVIEW = 0.60
INVENTED_REVIEW = 1.80
SKIN_DE_REVIEW = 6.0


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    from skimage.metrics import structural_similarity
    return float(structural_similarity(a, b, data_range=1.0))


def compare_faces(src_rgb: np.ndarray, out_rgb: np.ndarray, faces: list[Face],
                  analyzer: FaceAnalyzer | None = None, denoise_applied: bool = False) -> IdentityReport:
    """Compare *faces* (detected on source) between source and output (any scale, same framing)."""
    if not faces:
        return IdentityReport(available=True, method="no faces detected")
    analyzer = analyzer or FaceAnalyzer.get()
    sh, sw = src_rgb.shape[:2]
    oh, ow = out_rgb.shape[:2]
    if (oh, ow) != (sh, sw):
        out_s = cv2.resize(out_rgb, (sw, sh), interpolation=cv2.INTER_AREA)
    else:
        out_s = out_rgb
    rep = IdentityReport(available=True, method=f"{analyzer.name}" + ("+arcface" if analyzer.has_embeddings else ""))
    out_faces = analyzer.detect(out_s) if analyzer.det is not None else []
    sims = []
    for i, f in enumerate(faces[:8]):
        fd = FaceDrift(index=i, box=f.box)
        x0, y0, x1, y1 = [int(round(v)) for v in f.box]
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(sw, x1), min(sh, y1)
        if x1 - x0 < 12 or y1 - y0 < 12:
            fd.flags.append("face too small to assess")
            rep.faces.append(fd)
            continue
        ys, yo = src_rgb[y0:y1, x0:x1], out_s[y0:y1, x0:x1]
        ls = cv2.cvtColor(ys, cv2.COLOR_RGB2GRAY)
        lo = cv2.cvtColor(yo, cv2.COLOR_RGB2GRAY)
        fd.region_ssim = _ssim(ls, lo)
        e_s = float(np.abs(cv2.Laplacian(ls, cv2.CV_32F, ksize=1)).mean()) + 1e-5
        e_o = float(np.abs(cv2.Laplacian(lo, cv2.CV_32F, ksize=1)).mean())
        fd.fine_texture_ratio = e_o / e_s

        def dog(x):
            return float(np.abs(cv2.GaussianBlur(x, (0, 0), 1.0) - cv2.GaussianBlur(x, (0, 0), 3.0)).mean())
        fd.mid_texture_ratio = dog(lo) / (dog(ls) + 1e-5)
        cy0, cy1 = y0 + (y1 - y0) // 4, y1 - (y1 - y0) // 4
        cx0, cx1 = x0 + (x1 - x0) // 4, x1 - (x1 - x0) // 4
        lab_s = cv2.cvtColor(src_rgb[cy0:cy1, cx0:cx1].astype(np.float32), cv2.COLOR_RGB2LAB).reshape(-1, 3).mean(0)
        lab_o = cv2.cvtColor(out_s[cy0:cy1, cx0:cx1].astype(np.float32), cv2.COLOR_RGB2LAB).reshape(-1, 3).mean(0)
        fd.skin_delta_e = float(np.linalg.norm(lab_s - lab_o))
        if f.landmarks and analyzer.has_embeddings:
            es = analyzer.embed(analyzer.align(src_rgb, f.landmarks))
            eo = analyzer.embed(analyzer.align(out_s, f.landmarks))
            if es is not None and eo is not None:
                fd.embedding_similarity = float(np.dot(es, eo))
                sims.append(fd.embedding_similarity)
        if f.landmarks and out_faces:
            cx, cy = (f.box[0] + f.box[2]) / 2, (f.box[1] + f.box[3]) / 2
            best = min(out_faces, key=lambda g: (((g.box[0] + g.box[2]) / 2 - cx) ** 2 + ((g.box[1] + g.box[3]) / 2 - cy) ** 2))
            if best.landmarks:
                a, b = np.array(f.landmarks), np.array(best.landmarks)
                iod = float(np.linalg.norm(a[0] - a[1])) + 1e-6
                fd.landmark_drift_iod = float(np.linalg.norm(a - b, axis=1).mean() / iod)
        # flags
        if fd.embedding_similarity is not None and fd.embedding_similarity < SIM_REVIEW:
            fd.flags.append(f"identity embedding similarity {fd.embedding_similarity:.3f} < {SIM_REVIEW}")
        if fd.landmark_drift_iod is not None and fd.landmark_drift_iod > LANDMARK_REVIEW:
            fd.flags.append(f"landmark geometry drift {fd.landmark_drift_iod:.3f} IOD > {LANDMARK_REVIEW}")
        # beautification removes fine skin texture first; denoising legitimately removes fine noise, so the
        # fine-texture threshold is relaxed when a denoise op ran (mid-band texture must still survive).
        fine_thr = FINE_SMOOTHING_REVIEW * (0.6 if denoise_applied else 1.0)
        if (fd.mid_texture_ratio is not None and fd.mid_texture_ratio < SMOOTHING_REVIEW) or \
                (fd.fine_texture_ratio is not None and fd.fine_texture_ratio < fine_thr):
            fd.flags.append(f"possible skin smoothing / beautification (fine-texture ×{fd.fine_texture_ratio:.2f}, "
                            f"mid-texture ×{fd.mid_texture_ratio:.2f})")
        if fd.fine_texture_ratio is not None and fd.fine_texture_ratio > INVENTED_REVIEW:
            fd.flags.append(f"possible invented fine detail (fine-texture ×{fd.fine_texture_ratio:.2f})")
        if fd.skin_delta_e is not None and fd.skin_delta_e > SKIN_DE_REVIEW:
            fd.flags.append(f"skin tone shift ΔE {fd.skin_delta_e:.1f} > {SKIN_DE_REVIEW}")
        rep.faces.append(fd)
        rep.flags.extend(f"face {i + 1}: {m}" for m in fd.flags)
    if sims:
        rep.worst_similarity = float(min(sims))
    if not analyzer.has_embeddings:
        rep.flags.append("identity embeddings unavailable (buffalo_l not installed): texture/SSIM signals only")
    return rep

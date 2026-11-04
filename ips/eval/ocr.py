"""Development-time text-identity check using the Tesseract CLI if installed (not an app dependency).

Character error rate (CER) of an output's OCR text against the OCR text of the ground truth, so that
OCR quirks common to both cancel. Detects invented/garbled glyphs that pixel metrics reward.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np


def available() -> bool:
    return shutil.which("tesseract") is not None


def ocr(rgb: np.ndarray, upscale_to_height: int = 1000) -> str:
    h = rgb.shape[0]
    s = max(1.0, upscale_to_height / h)
    img = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC) if s > 1 else rgb
    u8 = cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.png"
        cv2.imwrite(str(p), u8)
        out = subprocess.run(["tesseract", str(p), "-", "--psm", "6"], capture_output=True, text=True, timeout=120)
    return " ".join(out.stdout.split())


def _lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(out: np.ndarray, gt: np.ndarray) -> dict[str, float | str]:
    ref, hyp = ocr(gt), ocr(cv2.resize(out, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_AREA))
    return {"ocr_cer": _lev(hyp, ref) / max(1, len(ref)), "ocr_ref_chars": len(ref), "ocr_sample": hyp[:120]}

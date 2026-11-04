"""Content hashes (exact duplicates) and perceptual hashes (similar-image *suggestions*)."""
from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np


def sha256_file(path: str | Path, chunk: int = 4 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dhash(rgb: np.ndarray, size: int = 8) -> int:
    """64-bit difference hash of a float RGB image."""
    gray = cv2.cvtColor((np.clip(rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def hamming(a: int, b: int) -> int:
    return bin(int(a) ^ int(b)).count("1")


SIMILAR_THRESHOLD = 6  # <= 6 of 64 bits: visually similar (suggestion only, never auto-delete)

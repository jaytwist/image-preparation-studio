"""Export profiles and organised, collision-safe output layout.

Layout:  <output_root>/<collection>/<relative source subfolder>/<Kind folder>/<file>
Kinds:   Restored/  (restored master)   Edited/   (edited variants)   Share/   AI_Ready/   Cutouts/
         Recipes/   (recipe + diagnostics + audit JSON)   Reports/ (batch reports)
Names:   <stem>__<src sha6>__<kind>__<recipe hash8>.<ext>; never overwritten (``_v002`` …).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import ops as cops
from .imageio import encode_image, icc_is_srgb, srgb_icc_bytes, to_srgb, write_bytes_no_clobber
from .metadata import Metadata, WriteSpec, embed_metadata

KIND_FOLDERS = {"restored_master": "Restored", "edited_variant": "Edited", "share_export": "Share",
                "ai_ready": "AI_Ready", "cutout": "Cutouts"}
KIND_LABELS = {"source": "Source", "restored_master": "Restored Master", "edited_variant": "Edited Variant",
               "share_export": "Share Export", "ai_ready": "AI / Video Ready", "cutout": "Transparent Cutout"}
EXT = {"TIFF": ".tif", "PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp"}


@dataclass
class ExportProfile:
    key: str                    # archive | ai_ready | share | cutout | edited
    kind: str                   # output kind
    fmt: str = "TIFF"
    bits: str = "source"        # source | 8 | 16
    long_edge: int = 0          # 0 = keep size
    multiple_of: int = 1        # pad (never crop) to a multiple
    jpeg_quality: int = 95
    to_srgb: bool = False
    strip_location: bool = False
    keep_alpha: bool = True
    label: str = ""


def profiles_from_settings(s) -> dict[str, ExportProfile]:
    return {
        "archive": ExportProfile("archive", "restored_master", fmt=s.archive_format, bits=s.archive_bits,
                                 label="Lossless Archive Master"),
        "ai_ready": ExportProfile("ai_ready", "ai_ready", fmt=s.ai_ready_format, bits="8",
                                  long_edge=s.ai_ready_long_edge, multiple_of=s.ai_ready_multiple_of, to_srgb=True,
                                  label="AI / Video Ready"),
        "share": ExportProfile("share", "share_export", fmt="JPEG", bits="8", long_edge=s.share_long_edge,
                               jpeg_quality=s.share_quality, to_srgb=True, strip_location=s.share_strip_location,
                               keep_alpha=False, label="Share / Web"),
        "cutout": ExportProfile("cutout", "cutout", fmt="PNG", bits="source", label="Transparent Cutout"),
        "edited": ExportProfile("edited", "edited_variant", fmt="PNG", bits="source", label="Edited Variant"),
    }


def safe_stem(name: str) -> str:
    s = re.sub(r"[^\w\-. ]+", "_", Path(name).stem, flags=re.UNICODE).strip(" .")
    return s[:80] or "image"


def output_path(root: Path, collection: str, rel_dir: str, stem: str, src_sha: str, kind: str, recipe_hash: str,
                fmt: str) -> Path:
    rel = Path(rel_dir) if rel_dir and rel_dir not in (".", "") else Path()
    if rel.is_absolute() or ".." in rel.parts:
        rel = Path(*[p for p in rel.parts if p not in ("..",) and not Path(p).anchor])
    folder = Path(root) / collection / rel / KIND_FOLDERS.get(kind, "Other")
    tag = {"restored_master": "master", "edited_variant": "edit", "share_export": "share", "ai_ready": "aiready",
           "cutout": "cutout"}.get(kind, kind)
    return folder / f"{safe_stem(stem)}__{src_sha[:6]}__{tag}__{recipe_hash[:8]}{EXT[fmt]}"


def render_for_profile(rgb: np.ndarray, alpha: np.ndarray | None, icc: bytes | None, prof: ExportProfile,
                       source_bits: int) -> tuple[np.ndarray, np.ndarray | None, bytes | None, int, list[str]]:
    """Apply profile size/colour/bit-depth. Returns (rgb, alpha, icc, bits, notes). Never crops."""
    notes: list[str] = []
    out_icc = icc
    if prof.to_srgb and not icc_is_srgb(icc):
        rgb, note = to_srgb(rgb, icc)
        notes.append(note)
        out_icc = srgb_icc_bytes()
    elif prof.to_srgb:
        out_icc = icc or srgb_icc_bytes()
    h, w = rgb.shape[:2]
    if prof.long_edge and max(h, w) > prof.long_edge:
        s = prof.long_edge / max(h, w)
        nw, nh = max(1, round(w * s)), max(1, round(h * s))
        rgb = cops.resize_to(rgb, nw, nh)
        if alpha is not None:
            alpha = cops.resize_to(alpha[..., None].repeat(3, 2), nw, nh)[..., 0]
        notes.append(f"Resized to fit long edge {prof.long_edge}px ({w}×{h} → {nw}×{nh}); aspect kept, no crop.")
    if prof.multiple_of and prof.multiple_of > 1:
        m = prof.multiple_of
        h, w = rgb.shape[:2]
        ph, pw = (-h) % m, (-w) % m
        if ph or pw:
            rgb = np.pad(rgb, ((0, ph), (0, pw), (0, 0)), mode="edge")
            if alpha is not None:
                alpha = np.pad(alpha, ((0, ph), (0, pw)), mode="constant")
            notes.append(f"Padded by {pw}px right / {ph}px bottom (edge replicate) to a multiple of {m}; "
                         "original content uncropped.")
    if alpha is not None and (not prof.keep_alpha or prof.fmt == "JPEG"):
        rgb = rgb * alpha[..., None] + (1 - alpha[..., None])  # flatten on white
        alpha = None
        notes.append("Transparency flattened onto white (format/profile has no alpha).")
    bits = source_bits if prof.bits == "source" else int(prof.bits)
    if prof.fmt in ("JPEG", "WEBP"):
        bits = 8
    return rgb, alpha, out_icc, bits, notes


def write_output(rgb: np.ndarray, alpha: np.ndarray | None, icc: bytes | None, prof: ExportProfile,
                 src_md: Metadata, dest: Path, *, source_bits: int, source_sha: str, recipe_hash: str,
                 recipe_summary: str, scale: float, generative: bool, edited: bool,
                 png_text: dict[str, str] | None = None) -> tuple[Path, dict, tuple[int, int, int], bytes]:
    rgb2, alpha2, out_icc, bits, notes = render_for_profile(rgb, alpha, icc, prof, source_bits)
    h, w = rgb2.shape[:2]
    keep_text = {k: v for k, v in (png_text or {}).items() if k in ("prompt", "workflow", "parameters")}
    enc = encode_image(rgb2, alpha2, prof.fmt, bits=bits, icc=out_icc, jpeg_quality=prof.jpeg_quality,
                       png_text=keep_text if prof.fmt == "PNG" and bits == 8 else None)
    spec = WriteSpec(output_kind=prof.kind, width=w, height=h, scale=scale * (w / max(1, rgb.shape[1])),
                     output_icc=out_icc, output_is_srgb=icc_is_srgb(out_icc), strip_location=prof.strip_location,
                     source_sha256=source_sha, recipe_hash=recipe_hash, recipe_summary=recipe_summary,
                     generative=generative, edited=edited, fmt=prof.fmt)
    data, audit = embed_metadata(enc, src_md, spec)
    audit["profile_notes"] = notes
    if keep_text and prof.fmt == "PNG" and bits == 8:
        audit["notes"].append(f"Generation-metadata PNG chunks carried over: {sorted(keep_text)}")
    elif src_md.png_text and any(k in src_md.png_text for k in ("prompt", "workflow", "parameters")):
        audit["notes"].append("Source generation-metadata PNG chunks not representable in this format; "
                              "kept in the recipe sidecar only.")
        audit.setdefault("dropped", {})["PNG:generation_text"] = f"not representable in {prof.fmt} {bits}-bit"
    final = write_bytes_no_clobber(data, dest)
    return final, audit, (w, h, bits), data

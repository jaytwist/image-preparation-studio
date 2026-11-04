"""Decoding and encoding.

Design rules
- Source files are opened read-only and never written, renamed or moved.
- Pixels are carried as float32 RGB in [0, 1] (H, W, 3) plus an optional float32 alpha (H, W).
- Processing happens in the *source* colour space; the source ICC profile travels with the
  image and is embedded in archive outputs, so wide-gamut sources are not clipped to sRGB.
  Share/Web exports are converted to sRGB explicitly (see ``to_srgb``).
- 16-bit sources stay 16-bit through archive outputs.
- EXIF orientation is applied to pixels at decode; outputs are written with Orientation = 1.
- Camera RAW is *developed* (rawpy/LibRaw) to 16-bit RGB. A processed RAW is never written back
  as camera RAW.
"""
from __future__ import annotations

import io
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageCms

try:  # HEIC/HEIF support is optional
    import pillow_heif

    pillow_heif.register_heif_opener()
    HEIF_AVAILABLE = True
except Exception:  # pragma: no cover - depends on platform wheels
    HEIF_AVAILABLE = False

try:
    import rawpy

    RAW_AVAILABLE = True
except Exception:  # pragma: no cover
    RAW_AVAILABLE = False

import tifffile

Image.MAX_IMAGE_PIXELS = 400_000_000  # allow large scans, still guard decompression bombs

RAW_EXTS = {".cr2", ".cr3", ".nef", ".nrw", ".arw", ".srf", ".sr2", ".dng", ".raf", ".orf", ".rw2",
            ".pef", ".srw", ".3fr", ".erf", ".kdc", ".mrw", ".x3f", ".iiq", ".rwl"}
HEIF_EXTS = {".heic", ".heif", ".hif", ".avif"}
PIL_EXTS = {".jpg", ".jpeg", ".jpe", ".jfif", ".webp", ".bmp", ".gif", ".tga", ".jp2", ".ico"}
CV_EXTS = {".png", ".tif", ".tiff"}
SUPPORTED_EXTS = RAW_EXTS | HEIF_EXTS | PIL_EXTS | CV_EXTS


class DecodeError(Exception):
    """Raised for unreadable, truncated or unsupported files. Message is user-facing."""


@dataclass
class LoadedImage:
    rgb: np.ndarray                      # float32 HxWx3 in [0,1]
    alpha: np.ndarray | None = None      # float32 HxW in [0,1]
    bit_depth: int = 8                   # 8 or 16 (container precision of source)
    icc_profile: bytes | None = None
    source_format: str = ""              # e.g. JPEG, PNG, TIFF, HEIF, RAW
    source_path: str = ""
    was_grayscale: bool = False
    orientation_applied: int = 1         # EXIF orientation that was baked into pixels
    jpeg_quality_estimate: int | None = None
    png_text: dict[str, str] = field(default_factory=dict)
    decode_notes: list[str] = field(default_factory=list)
    raw_developed: bool = False

    @property
    def height(self) -> int:
        return int(self.rgb.shape[0])

    @property
    def width(self) -> int:
        return int(self.rgb.shape[1])

    @property
    def megapixels(self) -> float:
        return self.width * self.height / 1e6


# ----------------------------------------------------------------------------- helpers

def _read_bytes(path: Path) -> bytes:
    # Read-only, binary. np.fromfile/open both work with non-ASCII Windows paths.
    with open(path, "rb") as f:
        return f.read()


def apply_orientation(arr: np.ndarray, orientation: int) -> np.ndarray:
    """Apply EXIF orientation (1..8) to an HxW[xC] array."""
    o = int(orientation or 1)
    if o == 2:
        return arr[:, ::-1].copy()
    if o == 3:
        return arr[::-1, ::-1].copy()
    if o == 4:
        return arr[::-1].copy()
    if o == 5:
        return np.ascontiguousarray(np.swapaxes(arr, 0, 1))
    if o == 6:
        return np.ascontiguousarray(np.rot90(arr, -1))
    if o == 7:
        return np.ascontiguousarray(np.rot90(np.swapaxes(arr, 0, 1), 2))
    if o == 8:
        return np.ascontiguousarray(np.rot90(arr, 1))
    return arr


def estimate_jpeg_quality(pil_img: Image.Image) -> int | None:
    """Estimate IJG quality from the luminance quantisation table (metadata-level signal)."""
    q = getattr(pil_img, "quantization", None)
    if not q:
        return None
    # Standard IJG luminance table (zigzag order irrelevant for the mean ratio).
    std = np.array([16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
                    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113,
                    92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], dtype=np.float64)
    lum = np.array(q.get(0, list(q.values())[0]), dtype=np.float64)
    if lum.size != 64:
        return None
    s = float(np.mean(lum) / np.mean(std) * 100.0)  # IJG scale factor in percent
    if s <= 0:
        return 100
    quality = (200.0 - s) / 2.0 if s <= 100 else 5000.0 / s
    return int(np.clip(round(quality), 1, 100))


def _to_float(arr: np.ndarray) -> tuple[np.ndarray, int]:
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0, 8
    if arr.dtype == np.uint16:
        return arr.astype(np.float32) / 65535.0, 16
    if arr.dtype in (np.float32, np.float64, np.float16):
        a = arr.astype(np.float32)
        if a.max() > 1.0 + 1e-3:
            a = a / max(float(a.max()), 1e-6)
        return np.clip(a, 0, 1), 16
    if arr.dtype == np.int16 or arr.dtype == np.int32 or arr.dtype == np.uint32:
        a = arr.astype(np.float64)
        a = (a - a.min()) / max(float(a.max() - a.min()), 1.0)
        return a.astype(np.float32), 16
    if arr.dtype == bool:
        return arr.astype(np.float32), 8
    raise DecodeError(f"Unsupported sample type {arr.dtype}")


def _split_channels(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray | None, bool]:
    """Return (rgb, alpha, was_gray) from HxW, HxWx1..4 in RGB(A) order."""
    if arr.ndim == 2:
        return np.repeat(arr[:, :, None], 3, axis=2), None, True
    c = arr.shape[2]
    if c == 1:
        return np.repeat(arr, 3, axis=2), None, True
    if c == 2:  # gray + alpha
        return np.repeat(arr[:, :, :1], 3, axis=2), arr[:, :, 1], True
    if c == 3:
        return arr, None, False
    return arr[:, :, :3], arr[:, :, 3], False


# ----------------------------------------------------------------------------- decode

def load_image(path: str | os.PathLike, orientation: int | None = None) -> LoadedImage:
    """Decode *path* without modifying it.

    ``orientation`` may be supplied from metadata; otherwise it is read from EXIF via Pillow.
    """
    p = Path(path)
    if not p.is_file():
        raise DecodeError(f"File not found: {p}")
    ext = p.suffix.lower()
    if p.stat().st_size == 0:
        raise DecodeError("File is empty (0 bytes).")
    try:
        if ext in RAW_EXTS:
            return _load_raw(p)
        data = _read_bytes(p)
        if ext in CV_EXTS:
            return _load_cv(p, data, orientation)
        return _load_pil(p, data, orientation)
    except DecodeError:
        raise
    except Exception as e:  # pragma: no cover - library-specific failure modes
        raise DecodeError(f"Could not decode {p.name}: {type(e).__name__}: {e}") from e


def _pil_side_info(data: bytes) -> tuple[bytes | None, int, dict[str, str], int | None, str]:
    """ICC, orientation, PNG text chunks, JPEG quality, format — read lazily via Pillow."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            icc = im.info.get("icc_profile")
            try:
                orient = int(im.getexif().get(0x0112, 1) or 1)
            except Exception:
                orient = 1
            text = {k: v for k, v in im.info.items() if isinstance(v, str) and k not in ("icc_profile",)}
            jq = estimate_jpeg_quality(im) if im.format == "JPEG" else None
            return icc, orient, text, jq, im.format or ""
    except Exception:
        return None, 1, {}, None, ""


def _load_cv(p: Path, data: bytes, orientation: int | None) -> LoadedImage:
    notes: list[str] = []
    icc, orient, text, _, fmt = _pil_side_info(data)
    arr = None
    if p.suffix.lower() in (".tif", ".tiff"):
        try:
            arr = tifffile.imread(io.BytesIO(data))
            if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[2] not in (3, 4):
                arr = np.moveaxis(arr, 0, -1)  # planar -> interleaved
            if arr.ndim > 3:
                arr = arr[0]
                notes.append("Multi-page TIFF: only the first page was loaded.")
            # tifffile returns RGB(A) order already
        except Exception as e:
            notes.append(f"tifffile could not decode ({type(e).__name__}); fell back to OpenCV.")
            arr = None
    if arr is None:
        buf = np.frombuffer(data, dtype=np.uint8)
        arr = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED | cv2.IMREAD_IGNORE_ORIENTATION)
        if arr is None:
            # Pillow handles some palette / odd PNGs OpenCV refuses
            return _load_pil(p, data, orientation)
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA if arr.shape[2] == 4 else cv2.COLOR_BGR2RGB)
    f, depth = _to_float(arr)
    rgb, alpha, gray = _split_channels(f)
    o = orientation if orientation is not None else orient
    rgb = apply_orientation(rgb, o)
    if alpha is not None:
        alpha = apply_orientation(alpha, o)
    return LoadedImage(rgb=np.ascontiguousarray(rgb, dtype=np.float32), alpha=alpha, bit_depth=depth,
                       icc_profile=icc, source_format=fmt or p.suffix.upper().strip("."), source_path=str(p),
                       was_grayscale=gray, orientation_applied=o, png_text=text, decode_notes=notes)


def _load_pil(p: Path, data: bytes, orientation: int | None) -> LoadedImage:
    notes: list[str] = []
    ext = p.suffix.lower()
    if ext in HEIF_EXTS and not HEIF_AVAILABLE:
        raise DecodeError("HEIC/HEIF decoding is not available (pillow-heif not installed).")
    im = Image.open(io.BytesIO(data))
    im.load()
    fmt = im.format or ext.upper().strip(".")
    icc = im.info.get("icc_profile")
    try:
        orient = int(im.getexif().get(0x0112, 1) or 1)
    except Exception:
        orient = 1
    if getattr(im, "n_frames", 1) > 1:
        notes.append(f"{fmt} has {im.n_frames} frames: only the first frame was loaded.")
        im.seek(0)
    jq = estimate_jpeg_quality(im) if fmt == "JPEG" else None
    text = {k: v for k, v in im.info.items() if isinstance(v, str)}
    gray = im.mode in ("L", "LA", "1", "I", "I;16", "F")
    if im.mode == "CMYK":
        notes.append("CMYK source converted to RGB via its ICC profile when present.")
        if icc:
            try:
                src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
                im = ImageCms.profileToProfile(im, src, ImageCms.createProfile("sRGB"), outputMode="RGB")
                icc = None  # pixels are now sRGB
            except Exception:
                im = im.convert("RGB")
        else:
            im = im.convert("RGB")
    if im.mode in ("I;16", "I;16B", "I;16L", "I"):
        arr = np.asarray(im).astype(np.uint16) if im.mode != "I" else np.asarray(im)
        f, depth = _to_float(arr)
        rgb, alpha = np.repeat(f[:, :, None], 3, axis=2), None
    else:
        has_alpha = im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info)
        im = im.convert("RGBA" if has_alpha else "RGB")
        f, depth = _to_float(np.asarray(im))
        rgb, alpha, _ = _split_channels(f)
    if ext in HEIF_EXTS:
        notes.append("HEIF decoded at 8-bit via pillow-heif; HDR/10-bit gain maps are not preserved.")
    o = orientation if orientation is not None else orient
    rgb = apply_orientation(rgb, o)
    if alpha is not None:
        alpha = apply_orientation(alpha, o)
    return LoadedImage(rgb=np.ascontiguousarray(rgb, dtype=np.float32), alpha=alpha, bit_depth=depth, icc_profile=icc,
                       source_format=fmt, source_path=str(p), was_grayscale=gray, orientation_applied=o,
                       jpeg_quality_estimate=jq, png_text=text, decode_notes=notes)


def _load_raw(p: Path) -> LoadedImage:
    if not RAW_AVAILABLE:
        raise DecodeError("Camera RAW decoding is not available (rawpy not installed).")
    try:
        with rawpy.imread(str(p)) as raw:
            rgb16 = raw.postprocess(output_bps=16, use_camera_wb=True, no_auto_bright=True,
                                    output_color=rawpy.ColorSpace.sRGB, gamma=(2.222, 4.5))
    except Exception as e:
        raise DecodeError(f"RAW decode failed for {p.name}: {e}") from e
    f, _ = _to_float(rgb16)
    return LoadedImage(rgb=np.ascontiguousarray(f), bit_depth=16, icc_profile=None, source_format="RAW",
                       source_path=str(p), raw_developed=True,
                       decode_notes=["Camera RAW developed with LibRaw (camera white balance, no auto-brightness, "
                                     "sRGB primaries, 16-bit). Outputs are developed derivatives, not RAW files."])


# ----------------------------------------------------------------------------- colour

_SRGB_PROFILE_BYTES: bytes | None = None


def srgb_icc_bytes() -> bytes:
    global _SRGB_PROFILE_BYTES
    if _SRGB_PROFILE_BYTES is None:
        prof = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
        _SRGB_PROFILE_BYTES = prof.tobytes()
    return _SRGB_PROFILE_BYTES


def icc_description(icc: bytes | None) -> str:
    if not icc:
        return "none (assumed sRGB)"
    try:
        prof = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        return (ImageCms.getProfileDescription(prof) or "unnamed").strip()
    except Exception:
        return "unreadable ICC profile"


def _colorants(icc: bytes) -> np.ndarray | None:
    try:
        p = ImageCms.ImageCmsProfile(io.BytesIO(icc)).profile
        cs = [p.red_colorant, p.green_colorant, p.blue_colorant]
        if any(c is None for c in cs):
            return None
        return np.array([c[0] for c in cs], np.float64)  # XYZ of each primary
    except Exception:
        return None


_SRGB_COLORANTS: np.ndarray | None = None


def icc_is_srgb(icc: bytes | None) -> bool:
    """True if no profile (sRGB assumed) or the profile has sRGB primaries (name alone is not trusted)."""
    global _SRGB_COLORANTS
    if not icc:
        return True
    d = icc_description(icc).lower()
    if "srgb" in d or "iec61966" in d or "iec 61966" in d:
        return True
    c = _colorants(icc)
    if c is None:
        return False
    if _SRGB_COLORANTS is None:
        _SRGB_COLORANTS = _colorants(srgb_icc_bytes())
    return bool(_SRGB_COLORANTS is not None and np.abs(c - _SRGB_COLORANTS).max() < 0.003)


def to_srgb(rgb: np.ndarray, icc: bytes | None) -> tuple[np.ndarray, str]:
    """Convert float RGB from *icc* to sRGB (perceptual intent). Returns (rgb, note)."""
    if icc_is_srgb(icc):
        return rgb, "already sRGB"
    try:
        src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        dst = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
        # LittleCMS via Pillow operates on 8-bit RGB; use 16-bit-safe two-pass by splitting hi/lo is
        # overkill for share exports which are 8-bit anyway.
        u8 = float_to_uint(rgb, 8)
        im = Image.fromarray(u8, "RGB")
        out = ImageCms.profileToProfile(im, src, dst, renderingIntent=ImageCms.Intent.PERCEPTUAL, outputMode="RGB")
        return np.asarray(out).astype(np.float32) / 255.0, f"converted {icc_description(icc)} -> sRGB (perceptual)"
    except Exception as e:
        return rgb, f"ICC conversion failed ({e}); pixels left unconverted and flagged"


def display_srgb_u8(rgb: np.ndarray, icc: bytes | None) -> np.ndarray:
    """8-bit sRGB rendering for on-screen display only."""
    out, _ = to_srgb(rgb, icc)
    return float_to_uint(out, 8)


# ----------------------------------------------------------------------------- encode

def float_to_uint(a: np.ndarray, bits: int) -> np.ndarray:
    a = np.clip(a, 0.0, 1.0)
    if bits == 16:
        return (a * 65535.0 + 0.5).astype(np.uint16)
    return (a * 255.0 + 0.5).astype(np.uint8)


def _compose(rgb: np.ndarray, alpha: np.ndarray | None) -> np.ndarray:
    if alpha is None:
        return rgb
    return np.concatenate([rgb, alpha[:, :, None]], axis=2)


def encode_image(rgb: np.ndarray, alpha: np.ndarray | None, fmt: str, *, bits: int = 8,
                 icc: bytes | None = None, jpeg_quality: int = 95, png_text: dict[str, str] | None = None,
                 tiff_compression: str = "zlib") -> bytes:
    """Encode to bytes. fmt in {TIFF, PNG, JPEG, WEBP}."""
    fmt = fmt.upper()
    if fmt == "TIFF":
        arr = float_to_uint(_compose(rgb, alpha), bits)
        bio = io.BytesIO()
        extra = {"extrasamples": [2]} if alpha is not None else {}  # 2 = unassociated alpha
        kwargs: dict[str, Any] = dict(photometric="rgb", compression=tiff_compression, **extra)
        if icc:
            kwargs["iccprofile"] = icc
        if arr.nbytes > 3_500_000_000:
            kwargs["bigtiff"] = True
        tifffile.imwrite(bio, arr, **kwargs)
        return bio.getvalue()
    if fmt == "PNG":
        if bits == 16:
            arr = float_to_uint(_compose(rgb, alpha), 16)
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA if alpha is not None else cv2.COLOR_RGB2BGR)
            ok, buf = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, 6])
            if not ok:
                raise IOError("PNG encode failed")
            return buf.tobytes()  # ICC is embedded afterwards by metadata.write (iCCP via exiv2)
        from PIL import PngImagePlugin
        arr = float_to_uint(_compose(rgb, alpha), 8)
        im = Image.fromarray(arr, "RGBA" if alpha is not None else "RGB")
        info = PngImagePlugin.PngInfo()
        for k, v in (png_text or {}).items():
            try:
                info.add_itxt(k, v)
            except Exception:
                pass
        bio = io.BytesIO()
        im.save(bio, "PNG", icc_profile=icc, pnginfo=info, compress_level=6)
        return bio.getvalue()
    if fmt in ("JPEG", "JPG"):
        if alpha is not None:
            raise ValueError("JPEG cannot store transparency; flatten first or choose PNG/TIFF.")
        im = Image.fromarray(float_to_uint(rgb, 8), "RGB")
        bio = io.BytesIO()
        im.save(bio, "JPEG", quality=int(jpeg_quality), subsampling=0 if jpeg_quality >= 90 else 2,
                optimize=True, icc_profile=icc)
        return bio.getvalue()
    if fmt == "WEBP":
        im = Image.fromarray(float_to_uint(_compose(rgb, alpha), 8), "RGBA" if alpha is not None else "RGB")
        bio = io.BytesIO()
        im.save(bio, "WEBP", quality=int(jpeg_quality), method=5, icc_profile=icc)
        return bio.getvalue()
    raise ValueError(f"Unsupported output format {fmt}")


def write_bytes_no_clobber(data: bytes, dest: Path) -> Path:
    """Atomically write *data* to *dest* without ever replacing an existing file.

    If *dest* exists, ``_v002``, ``_v003`` ... suffixes are tried. Returns the final path.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".ips_", suffix=dest.suffix + ".part", dir=str(dest.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        stem, suffix = dest.stem, dest.suffix
        candidate = dest
        n = 1
        while True:
            try:
                os.link(tmp, candidate)  # fails if candidate exists (all platforms)
                break
            except FileExistsError:
                n += 1
                candidate = dest.with_name(f"{stem}_v{n:03d}{suffix}")
            except OSError:
                # Filesystems without hard links: exclusive-create then copy.
                try:
                    with open(candidate, "xb") as out:
                        out.write(data)
                    break
                except FileExistsError:
                    n += 1
                    candidate = dest.with_name(f"{stem}_v{n:03d}{suffix}")
        return candidate
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def resize(rgb: np.ndarray, width: int, height: int, method: str = "lanczos") -> np.ndarray:
    interp = {"lanczos": cv2.INTER_LANCZOS4, "cubic": cv2.INTER_CUBIC, "area": cv2.INTER_AREA,
              "nearest": cv2.INTER_NEAREST, "linear": cv2.INTER_LINEAR}[method]
    if width < rgb.shape[1] and method == "lanczos":
        interp = cv2.INTER_AREA  # downscale: area avoids aliasing
    out = cv2.resize(rgb, (int(width), int(height)), interpolation=interp)
    return np.clip(out, 0, 1).astype(np.float32) if out.dtype != np.uint8 else out

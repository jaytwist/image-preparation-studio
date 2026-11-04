"""Metadata reading, transfer and post-write audit.

* Every pyexiv2 call is serialised by ``EXIV2_LOCK`` (pyexiv2/exiv2 is not thread-safe).
* Metadata is written into the *encoded output bytes in memory* (``pyexiv2.ImageData``) before the
  file is committed, so a crash can never leave a half-written output on disk.
* Tags that describe the old container or old pixels (dimensions, orientation, compression,
  thumbnails, strip offsets, DNG/RAW-specific tags, MakerNotes, Camera Raw develop settings)
  are not copied; each is recorded in the audit with a reason.
* After writing, the output is re-read and compared with what was intended.
"""
from __future__ import annotations

import datetime as _dt
import io
import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from .. import __version__

try:
    import pyexiv2

    pyexiv2.set_log_level(4)  # mute
    # BMFF (HEIF/AVIF/CR3) support is always enabled in pyexiv2 >= 2.14
    EXIV2_AVAILABLE = True
except Exception:  # pragma: no cover
    pyexiv2 = None
    EXIV2_AVAILABLE = False

EXIV2_LOCK = threading.RLock()
IPS_NS = "http://ns.image-preparation-studio.local/1.0/"
_ns_registered = False
MAX_EXIV2_BYTES = 2_000_000_000  # larger files are read via Pillow fallback only

OUTPUT_KINDS = {"restored_master", "edited_variant", "share_export", "ai_ready", "cutout"}


def _ensure_ns() -> None:
    global _ns_registered
    if _ns_registered or not EXIV2_AVAILABLE:
        return
    try:
        pyexiv2.registerNs(IPS_NS, "ips")
    except Exception:
        pass  # already registered
    _ns_registered = True


@dataclass
class Metadata:
    exif: dict[str, Any] = field(default_factory=dict)
    iptc: dict[str, Any] = field(default_factory=dict)
    xmp: dict[str, Any] = field(default_factory=dict)
    icc: bytes | None = None
    png_text: dict[str, str] = field(default_factory=dict)
    comment: str | None = None
    reader: str = "none"
    errors: list[str] = field(default_factory=list)
    c2pa_present: bool = False

    def to_json(self) -> dict[str, Any]:
        def _clean(d: dict[str, Any]) -> dict[str, Any]:
            out = {}
            for k, v in d.items():
                s = v if isinstance(v, (str, int, float, list)) else str(v)
                if isinstance(s, str) and len(s) > 2000:
                    s = s[:2000] + "…(truncated)"
                out[k] = s
            return out
        return {"exif": _clean(self.exif), "iptc": _clean(self.iptc), "xmp": _clean(self.xmp),
                "icc_bytes": len(self.icc) if self.icc else 0, "png_text_keys": sorted(self.png_text),
                "comment": self.comment, "reader": self.reader, "errors": self.errors,
                "c2pa_present": self.c2pa_present}


# ----------------------------------------------------------------------------- read

def read_metadata(path: str | Path, data: bytes | None = None) -> Metadata:
    p = Path(path)
    md = Metadata()
    if data is None:
        try:
            if p.stat().st_size <= MAX_EXIV2_BYTES:
                data = p.read_bytes()
        except OSError as e:
            md.errors.append(f"read failed: {e}")
            return md
    if data is not None:
        md.c2pa_present = _detect_c2pa(data)
    if EXIV2_AVAILABLE and data is not None:
        with EXIV2_LOCK:
            img = None
            try:
                img = pyexiv2.ImageData(data)
                md.exif = _safe(img.read_exif)
                md.iptc = _safe(img.read_iptc)
                md.xmp = _safe(img.read_xmp)
                try:
                    icc = img.read_icc()
                    md.icc = bytes(icc) if icc else None
                except Exception:
                    md.icc = None
                try:
                    md.comment = img.read_comment() or None
                except Exception:
                    pass
                md.reader = "exiv2"
            except Exception as e:
                md.errors.append(f"exiv2: {type(e).__name__}: {e}")
            finally:
                if img is not None:
                    try:
                        img.close()
                    except Exception:
                        pass
    if md.reader == "none" or not md.exif:
        _pillow_fallback(p, data, md)
    # PNG text chunks (ComfyUI/A1111 generation params) via Pillow, independent of exiv2
    if data is not None and data[:8] == b"\x89PNG\r\n\x1a\n":
        try:
            with Image.open(io.BytesIO(data)) as im:
                md.png_text = {k: v for k, v in im.info.items() if isinstance(v, str)}
        except Exception:
            pass
    return md


def _safe(fn) -> dict[str, Any]:
    try:
        return dict(fn() or {})
    except Exception:
        return {}


def _pillow_fallback(p: Path, data: bytes | None, md: Metadata) -> None:
    try:
        src = io.BytesIO(data) if data is not None else p
        with Image.open(src) as im:
            ex = im.getexif()
            from PIL.ExifTags import GPSTAGS, TAGS
            for k, v in ex.items():
                md.exif.setdefault(f"Exif.Image.{TAGS.get(k, k)}", str(v))
            try:
                sub = ex.get_ifd(0x8769)
                for k, v in sub.items():
                    md.exif.setdefault(f"Exif.Photo.{TAGS.get(k, k)}", str(v))
                gps = ex.get_ifd(0x8825)
                for k, v in gps.items():
                    md.exif.setdefault(f"Exif.GPSInfo.{GPSTAGS.get(k, k)}", str(v))
            except Exception:
                pass
            if md.icc is None:
                md.icc = im.info.get("icc_profile")
        if md.reader == "none":
            md.reader = "pillow"
    except Exception as e:
        md.errors.append(f"pillow: {type(e).__name__}: {e}")


def _detect_c2pa(data: bytes) -> bool:
    head = data[: 4 * 1024 * 1024]
    return b"c2pa" in head and (b"jumb" in head or b"caBX" in head)


# ----------------------------------------------------------------------------- summary

AI_SOFTWARE_HINTS = ("comfyui", "stable diffusion", "automatic1111", "invokeai", "midjourney", "dall-e", "dall·e",
                     "firefly", "flux", "qwen", "grok", "imagen", "ideogram", "leonardo", "novelai", "fooocus",
                     "sdxl", "krea", "nh maker")


def summarize(md: Metadata, width: int, height: int) -> dict[str, Any]:
    e, x = md.exif, md.xmp

    def g(*keys):
        for k in keys:
            v = e.get(k) or x.get(k)
            if v not in (None, "", []):
                return v
        return None

    software = g("Exif.Image.Software", "Xmp.xmp.CreatorTool")
    dst = x.get("Xmp.Iptc4xmpExt.DigitalSourceType") or ""
    ai_hints: list[str] = []
    for k in ("prompt", "workflow", "parameters", "Comment", "Description", "Software"):
        if k in md.png_text:
            val = md.png_text[k].lower()
            if k in ("prompt", "workflow"):
                ai_hints.append(f"PNG '{k}' chunk (ComfyUI-style workflow metadata)")
            elif k == "parameters" and ("steps:" in val or "sampler" in val or "seed" in val):
                ai_hints.append("PNG 'parameters' chunk (A1111-style generation parameters)")
            elif any(h in val for h in AI_SOFTWARE_HINTS):
                ai_hints.append(f"PNG '{k}' mentions a generator")
    if software and any(h in str(software).lower() for h in AI_SOFTWARE_HINTS):
        ai_hints.append(f"Software tag: {software}")
    if "trainedalgorithmicmedia" in str(dst).lower() or "algorithmicmedia" in str(dst).lower():
        ai_hints.append(f"IPTC DigitalSourceType: {dst.rsplit('/', 1)[-1]}")
    if md.c2pa_present:
        ai_hints.append("C2PA content-credentials manifest present (not cryptographically verified)")
    edit_software = None
    if software and any(s in str(software).lower() for s in ("photoshop", "lightroom", "gimp", "snapseed", "vsco",
                                                             "affinity", "capture one", "luminar", "facetune",
                                                             "picsart", "meitu", "topaz", "remini")):
        edit_software = str(software)
    claimed_w = g("Exif.Photo.PixelXDimension")
    claimed_h = g("Exif.Photo.PixelYDimension")
    dim_mismatch = False
    try:
        if claimed_w and claimed_h:
            cw, ch = int(str(claimed_w)), int(str(claimed_h))
            dim_mismatch = {cw, ch} != {width, height}
    except ValueError:
        pass
    iso = g("Exif.Photo.ISOSpeedRatings", "Exif.Photo.PhotographicSensitivity")
    try:
        iso_v = int(str(iso).split()[0]) if iso is not None else None
    except ValueError:
        iso_v = None
    make = g("Exif.Image.Make")
    model = g("Exif.Image.Model")
    scanner = bool(make and any(s in str(make).lower() for s in ("epson", "plustek", "canoscan", "scan"))) or \
        bool(model and "scan" in str(model).lower())
    return {
        "camera_make": make, "camera_model": model,
        "lens": g("Exif.Photo.LensModel", "Xmp.aux.Lens"),
        "iso": iso_v, "exposure_time": g("Exif.Photo.ExposureTime"), "f_number": g("Exif.Photo.FNumber"),
        "date_taken": g("Exif.Photo.DateTimeOriginal", "Xmp.exif.DateTimeOriginal", "Xmp.photoshop.DateCreated"),
        "software": software, "edit_software": edit_software,
        "has_gps": any(k.startswith("Exif.GPSInfo.GPSLatitude") or k.startswith("Xmp.exif.GPSLatitude")
                       for k in list(e) + list(x)),
        "copyright": g("Exif.Image.Copyright", "Xmp.dc.rights", "Iptc.Application2.Copyright"),
        "keywords": md.iptc.get("Iptc.Application2.Keywords") or x.get("Xmp.dc.subject"),
        "white_balance_mode": g("Exif.Photo.WhiteBalance"),
        "scene_type": g("Exif.Photo.SceneCaptureType"),
        "icc": "embedded" if md.icc else "none",
        "ai_generation_hints": ai_hints,
        "is_camera_capture": bool(make or model) and not scanner,
        "is_scanner": scanner,
        "exif_dimension_mismatch": dim_mismatch,
        "digital_source_type": dst or None,
        "c2pa_present": md.c2pa_present,
        "reader": md.reader,
        "errors": md.errors,
    }


# ----------------------------------------------------------------------------- transfer rules

_EXIF_DROP_EXACT = {
    "Exif.Image.ImageWidth": "describes source container dimensions",
    "Exif.Image.ImageLength": "describes source container dimensions",
    "Exif.Image.BitsPerSample": "describes source encoding",
    "Exif.Image.Compression": "describes source encoding",
    "Exif.Image.PhotometricInterpretation": "describes source encoding",
    "Exif.Image.StripOffsets": "container structure",
    "Exif.Image.RowsPerStrip": "container structure",
    "Exif.Image.StripByteCounts": "container structure",
    "Exif.Image.PlanarConfiguration": "container structure",
    "Exif.Image.SamplesPerPixel": "container structure",
    "Exif.Image.TileWidth": "container structure", "Exif.Image.TileLength": "container structure",
    "Exif.Image.TileOffsets": "container structure", "Exif.Image.TileByteCounts": "container structure",
    "Exif.Image.JPEGInterchangeFormat": "container structure",
    "Exif.Image.JPEGInterchangeFormatLength": "container structure",
    "Exif.Image.SubIFDs": "RAW/TIFF sub-image structure",
    "Exif.Image.NewSubfileType": "container structure",
    "Exif.Image.XMLPacket": "XMP is transferred separately",
    "Exif.Image.InterColorProfile": "ICC is handled by colour management",
    "Exif.Image.IPTCNAA": "IPTC is transferred separately",
    "Exif.Image.ImageResources": "Photoshop resources describe old pixels",
    "Exif.Image.PrintImageMatching": "printer-specific, describes old pixels",
    "Exif.Photo.MakerNote": "vendor MakerNote offsets/signatures cannot be meaningfully carried to new pixels",
    "Exif.Photo.InteroperabilityTag": "pointer",
    "Exif.Image.ExifTag": "pointer", "Exif.Image.GPSTag": "pointer",
    "Exif.Photo.ComponentsConfiguration": "describes source encoding",
    "Exif.Photo.CompressedBitsPerPixel": "describes source encoding",
}
_EXIF_DROP_PREFIX = {
    "Exif.Thumbnail.": "embedded thumbnail shows the unprocessed image",
    "Exif.SubImage": "RAW sub-image structure", "Exif.Iop.": "interoperability index of the source",
    "Exif.MakerNote.": "vendor MakerNote", "Exif.Canon": "vendor MakerNote", "Exif.Nikon": "vendor MakerNote",
    "Exif.Sony": "vendor MakerNote", "Exif.Fujifilm": "vendor MakerNote", "Exif.Olympus": "vendor MakerNote",
    "Exif.OlympusCs": "vendor MakerNote", "Exif.Panasonic": "vendor MakerNote", "Exif.Pentax": "vendor MakerNote",
    "Exif.Samsung": "vendor MakerNote", "Exif.Sigma": "vendor MakerNote", "Exif.Minolta": "vendor MakerNote",
    "Exif.Casio": "vendor MakerNote", "Exif.Apple": "vendor MakerNote", "Exif.Leica": "vendor MakerNote",
    "Exif.Image2.": "secondary IFD of source", "Exif.Image3.": "secondary IFD of source",
}
_EXIF_DROP_REGEX = [(re.compile(r"^Exif\.Image\.(DNG|Camera|Calibration|ColorMatrix|ForwardMatrix|Analog|"
                                r"BaselineExposure|BaselineNoise|BaselineSharpness|LinearResponse|"
                                r"CFA|BlackLevel|WhiteLevel|DefaultCrop|DefaultScale|ActiveArea|"
                                r"Illuminant|AsShot|Profile|Opcode|RawData|Linearization|NoiseProfile|"
                                r"OriginalRaw|Shadow|BestQuality|ChromaBlur|AntiAlias|MaskedAreas|Bayer|"
                                r"ReductionMatrix|RawImage|PreviewApplication|PreviewSettings|PreviewColor|"
                                r"PreviewDateTime|RawToPreview|NoiseReduction|LensInfo$)"),
                     "RAW/DNG development data does not apply to developed pixels")]

_XMP_DROP_EXACT = {
    "Xmp.tiff.ImageWidth": "old dimensions", "Xmp.tiff.ImageLength": "old dimensions",
    "Xmp.tiff.BitsPerSample": "old encoding", "Xmp.tiff.Compression": "old encoding",
    "Xmp.tiff.PhotometricInterpretation": "old encoding", "Xmp.tiff.SamplesPerPixel": "old encoding",
    "Xmp.tiff.PlanarConfiguration": "old encoding", "Xmp.photoshop.ICCProfile": "replaced by actual output ICC",
    "Xmp.xmpMM.InstanceID": "new instance", "Xmp.exif.NativeDigest": "stale digest",
    "Xmp.tiff.NativeDigest": "stale digest",
}
_XMP_DROP_PREFIX = {
    "Xmp.crs.": "Camera Raw develop settings would be re-applied by Lightroom/ACR to already-processed pixels",
    "Xmp.xmpMM.History": "history entries are summarised in the IPS audit instead",
    "Xmp.photoshop.DocumentAncestors": "large ancestor list, describes source document",
    "Xmp.ips.": "previous IPS provenance replaced by current job",
}

LOCATION_EXIF_PREFIX = ("Exif.GPSInfo.",)
LOCATION_XMP_PREFIX = ("Xmp.exif.GPS", "Xmp.photoshop.City", "Xmp.photoshop.State", "Xmp.photoshop.Country",
                       "Xmp.iptc.Location", "Xmp.iptc.CountryCode", "Xmp.Iptc4xmpCore.Location",
                       "Xmp.Iptc4xmpCore.CountryCode", "Xmp.Iptc4xmpExt.LocationCreated",
                       "Xmp.Iptc4xmpExt.LocationShown")
LOCATION_IPTC = ("Iptc.Application2.City", "Iptc.Application2.SubLocation", "Iptc.Application2.ProvinceState",
                 "Iptc.Application2.CountryName", "Iptc.Application2.CountryCode", "Iptc.Application2.LocationCode",
                 "Iptc.Application2.LocationName")


def _drop_reason_exif(key: str) -> str | None:
    if key in _EXIF_DROP_EXACT:
        return _EXIF_DROP_EXACT[key]
    for pre, why in _EXIF_DROP_PREFIX.items():
        if key.startswith(pre):
            return why
    for rx, why in _EXIF_DROP_REGEX:
        if rx.match(key):
            return why
    return None


def _drop_reason_xmp(key: str) -> str | None:
    if key in _XMP_DROP_EXACT:
        return _XMP_DROP_EXACT[key]
    for pre, why in _XMP_DROP_PREFIX.items():
        if key.startswith(pre):
            return why
    return None


@dataclass
class WriteSpec:
    output_kind: str                 # one of OUTPUT_KINDS
    width: int
    height: int
    scale: float = 1.0
    output_icc: bytes | None = None  # profile actually embedded in output pixels
    output_is_srgb: bool = True
    strip_location: bool = False
    source_sha256: str = ""
    recipe_hash: str = ""
    recipe_summary: str = ""
    generative: bool = False
    edited: bool = False
    fmt: str = "TIFF"


def plan_transfer(src: Metadata, spec: WriteSpec) -> tuple[dict, dict, dict, dict[str, Any]]:
    """Return (exif, iptc, xmp, audit_plan)."""
    now = _dt.datetime.now().strftime("%Y:%m:%d %H:%M:%S")
    audit: dict[str, Any] = {"transferred": [], "updated": {}, "dropped": {}, "stripped_location": [], "notes": []}
    exif: dict[str, Any] = {}
    for k, v in src.exif.items():
        why = _drop_reason_exif(k)
        if why:
            audit["dropped"][k] = why
            continue
        if spec.strip_location and k.startswith(LOCATION_EXIF_PREFIX):
            audit["stripped_location"].append(k)
            continue
        exif[k] = v
        audit["transferred"].append(k)
    upd = {
        "Exif.Image.Orientation": "1",
        "Exif.Photo.PixelXDimension": str(spec.width),
        "Exif.Photo.PixelYDimension": str(spec.height),
        "Exif.Image.DateTime": now,
        "Exif.Image.Software": f"Image Preparation Studio {__version__}",
        "Exif.Photo.ColorSpace": "1" if spec.output_is_srgb else "65535",
    }
    if src.exif:
        for res_key in ("Exif.Image.XResolution", "Exif.Image.YResolution"):
            if res_key in exif and spec.scale and spec.scale != 1.0:
                try:
                    num, den = str(exif[res_key]).split("/") if "/" in str(exif[res_key]) else (exif[res_key], 1)
                    dpi = float(num) / float(den)
                    upd[res_key] = f"{int(round(dpi * spec.scale))}/1"
                    audit["notes"].append(f"{res_key} scaled ×{spec.scale:g} to keep physical print size")
                except Exception:
                    pass
    for k, v in upd.items():
        if exif.get(k) != v:
            audit["updated"][k] = {"from": exif.get(k), "to": v}
        exif[k] = v
        if k in audit["transferred"]:
            audit["transferred"].remove(k)

    iptc: dict[str, Any] = {}
    for k, v in src.iptc.items():
        if spec.strip_location and k in LOCATION_IPTC:
            audit["stripped_location"].append(k)
            continue
        iptc[k] = v
        audit["transferred"].append(k)

    xmp: dict[str, Any] = {}
    for k, v in src.xmp.items():
        why = _drop_reason_xmp(k)
        if why:
            audit["dropped"][k] = why
            continue
        if spec.strip_location and k.startswith(LOCATION_XMP_PREFIX):
            audit["stripped_location"].append(k)
            continue
        if isinstance(v, dict):  # lang-alt etc. are passed through as-is
            xmp[k] = v
        else:
            xmp[k] = v
        audit["transferred"].append(k)
    iso_now = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
    xupd: dict[str, Any] = {
        "Xmp.tiff.Orientation": "1",
        "Xmp.exif.PixelXDimension": str(spec.width),
        "Xmp.exif.PixelYDimension": str(spec.height),
        "Xmp.xmp.ModifyDate": iso_now,
        "Xmp.xmp.MetadataDate": iso_now,
        "Xmp.xmp.CreatorTool": f"Image Preparation Studio {__version__}",
        "Xmp.ips.OutputKind": spec.output_kind,
        "Xmp.ips.SourceSHA256": spec.source_sha256,
        "Xmp.ips.RecipeHash": spec.recipe_hash,
        "Xmp.ips.Recipe": spec.recipe_summary[:1500],
        "Xmp.ips.Generative": "True" if spec.generative else "False",
        "Xmp.ips.Edited": "True" if spec.edited else "False",
        "Xmp.ips.Version": __version__,
    }
    src_sw = src.exif.get("Exif.Image.Software") or src.xmp.get("Xmp.xmp.CreatorTool")
    if src_sw:
        xupd["Xmp.ips.SourceSoftware"] = str(src_sw)
    if spec.generative:
        xupd["Xmp.Iptc4xmpExt.DigitalSourceType"] = \
            "http://cv.iptc.org/newscodes/digitalsourcetype/compositeWithTrainedAlgorithmicMedia"
    if "Xmp.tiff.Orientation" not in src.xmp:
        xupd.pop("Xmp.tiff.Orientation")
    for k, v in xupd.items():
        if xmp.get(k) != v and not k.startswith("Xmp.ips."):
            audit["updated"][k] = {"from": xmp.get(k), "to": v}
        xmp[k] = v
        if k in audit["transferred"]:
            audit["transferred"].remove(k)
    if src.exif.get("Exif.Photo.MakerNote") or any(k.startswith(("Exif.Canon", "Exif.Nikon", "Exif.Sony"))
                                                   for k in src.exif):
        audit["notes"].append("MakerNote not transferred; standard EXIF lens/camera fields were kept.")
    if src.c2pa_present:
        audit["notes"].append("Source C2PA manifest cannot remain valid after pixel changes; not carried over.")
    return exif, iptc, xmp, audit


def _apply_dict(img, kind: str, data: dict[str, Any], failed: dict[str, str]) -> None:
    if not data:
        return
    fn = {"exif": img.modify_exif, "iptc": img.modify_iptc, "xmp": img.modify_xmp}[kind]
    try:
        fn(data)
        return
    except Exception:
        pass
    for k, v in data.items():  # isolate failing keys
        try:
            fn({k: v})
        except Exception as e:
            failed[k] = f"{type(e).__name__}: {str(e)[:160]}"


def embed_metadata(encoded: bytes, src: Metadata, spec: WriteSpec) -> tuple[bytes, dict[str, Any]]:
    """Return (bytes_with_metadata, audit). Never raises for metadata problems: they are audited."""
    exif, iptc, xmp, audit = plan_transfer(src, spec)
    audit["failed"] = {}
    audit["output_format"] = spec.fmt
    audit["source_counts"] = {"exif": len(src.exif), "iptc": len(src.iptc), "xmp": len(src.xmp),
                              "icc": bool(src.icc), "png_text": sorted(src.png_text)}
    if not EXIV2_AVAILABLE:
        audit["notes"].append("exiv2 unavailable: only pixel data and embedded ICC were written.")
        audit["verified"] = False
        return encoded, audit
    _ensure_ns()
    with EXIV2_LOCK:
        img = None
        try:
            img = pyexiv2.ImageData(encoded)
            _apply_dict(img, "exif", exif, audit["failed"])
            _apply_dict(img, "iptc", iptc, audit["failed"])
            _apply_dict(img, "xmp", xmp, audit["failed"])
            if spec.output_icc and spec.fmt == "PNG":
                try:
                    cur = img.read_icc()
                except Exception:
                    cur = None
                if not cur:
                    try:
                        img.modify_icc(spec.output_icc)
                    except Exception as e:
                        audit["failed"]["ICC"] = str(e)
            if src.comment:
                try:
                    img.modify_comment(src.comment)
                except Exception:
                    pass
            out = img.get_bytes()
        except Exception as e:
            audit["notes"].append(f"exiv2 could not write metadata to {spec.fmt}: {type(e).__name__}: {e}")
            audit["verified"] = False
            return encoded, audit
        finally:
            if img is not None:
                try:
                    img.close()
                except Exception:
                    pass
    audit.update(verify_output(out, exif, iptc, xmp, spec))
    return out, audit


def _norm(v: Any) -> str:
    if isinstance(v, list):
        return "|".join(_norm(x) for x in v)
    if isinstance(v, dict):
        return json.dumps(v, sort_keys=True)
    return str(v).strip()


def verify_output(data: bytes, exif: dict, iptc: dict, xmp: dict, spec: WriteSpec) -> dict[str, Any]:
    """Re-read written bytes and compare with what was intended."""
    md = read_metadata("<memory>", data)
    mism: dict[str, Any] = {}
    for intended, actual in ((exif, md.exif), (iptc, md.iptc), (xmp, md.xmp)):
        for k, v in intended.items():
            if k not in actual or (_norm(actual[k]) == "" and _norm(v) != ""):
                mism[k] = "missing after write"
            elif _norm(actual[k]) != _norm(v) and not k.endswith("Date"):
                # exiv2 may normalise representation (e.g. rationals); keep as a soft mismatch
                mism[k] = {"intended": _norm(v)[:120], "actual": _norm(actual[k])[:120]}
    leaked = []
    if spec.strip_location:
        leaked = [k for k in list(md.exif) + list(md.xmp) + list(md.iptc)
                  if k.startswith(LOCATION_EXIF_PREFIX) or k.startswith(LOCATION_XMP_PREFIX) or k in LOCATION_IPTC]
    hard = {k: v for k, v in mism.items() if v == "missing after write"}
    from .imageio import icc_description
    return {"verified": not hard and not leaked, "verify_mismatches": mism, "location_leaks": leaked,
            "output_icc": icc_description(md.icc) if md.icc else "none",
            "output_counts": {"exif": len(md.exif), "iptc": len(md.iptc), "xmp": len(md.xmp)}}

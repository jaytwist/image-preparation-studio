"""Benchmark corpus generator.

Two clearly separated groups:
* **synthetic** – clean reference images with *known, injected* degradations and the clean ground
  truth. These support objective full-reference metrics (PSNR/SSIM/ΔE/…) and known truth labels for
  the Auto Accuracy Benchmark.
* **real** – real-world images without ground truth (old photos, phone crops, real RAW/EXIF samples).
  These support only no-reference measurements, identity-drift signals and human review. Results on
  them are *fidelity assessments / preferences*, never objective accuracy.

AI-generated category: no genuine Qwen/FLUX/Grok outputs are available offline here, so the fixtures
are **stand-ins** (real images carrying generator-style metadata). They test metadata routing only.

Sources: scikit-image bundled sample data (public domain / permissive), and – when present – images
copied into fixtures/_downloads (SwinIR/Real-ESRGAN/InsightFace test images, exif-samples, rawpy test
NEF, Compact-ICC-Profiles). Generated files live in fixtures/corpus (git-ignored).

    python -m ips.eval.fixtures [--out fixtures/corpus]
"""
from __future__ import annotations

import io
import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin

from ..paths import PROJECT_ROOT

DL = PROJECT_ROOT / "fixtures" / "_downloads"
RNG = np.random.default_rng(1234)


# ----------------------------------------------------------------------------- degradations

def f32(u8: np.ndarray) -> np.ndarray:
    return u8.astype(np.float32) / 255.0


def u8(f: np.ndarray) -> np.ndarray:
    return (np.clip(f, 0, 1) * 255 + 0.5).astype(np.uint8)


def down(img: np.ndarray, s: float) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.resize(img, (max(8, int(round(w * s))), max(8, int(round(h * s)))), interpolation=cv2.INTER_AREA)


def jpeg_bytes(img: np.ndarray, q: int, exif: bytes | None = None, icc: bytes | None = None) -> bytes:
    bio = io.BytesIO()
    kw: dict[str, Any] = {"quality": q, "subsampling": 2}
    if exif:
        kw["exif"] = exif
    if icc:
        kw["icc_profile"] = icc
    Image.fromarray(u8(img)).save(bio, "JPEG", **kw)
    return bio.getvalue()


def jpeg_roundtrip(img: np.ndarray, q: int) -> np.ndarray:
    return f32(np.asarray(Image.open(io.BytesIO(jpeg_bytes(img, q))).convert("RGB")))


def noise(img: np.ndarray, sigma: float, chroma: float = 0.5) -> np.ndarray:
    n_l = RNG.normal(0, sigma / 255, img.shape[:2])[..., None]
    n_c = RNG.normal(0, sigma * chroma / 255, img.shape)
    return np.clip(img + n_l + n_c, 0, 1).astype(np.float32)


def cast(img: np.ndarray, gains: tuple[float, float, float]) -> np.ndarray:
    lin = np.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4)
    lin = lin * np.array(gains, np.float32)
    lin = np.clip(lin, 0, 1)
    return np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055).astype(np.float32)


def oversharpen(img: np.ndarray, amount: float = 2.2, radius: float = 2.0) -> np.ndarray:
    b = cv2.GaussianBlur(img, (0, 0), radius)
    return np.clip(img + amount * (img - b), 0, 1).astype(np.float32)


def oversmooth(img: np.ndarray) -> np.ndarray:
    x = u8(img)
    for _ in range(3):
        x = cv2.bilateralFilter(x, 9, 40, 9)
    return f32(x)


def hdr_look(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(u8(img), cv2.COLOR_RGB2LAB)
    cl = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    lab[..., 0] = cl.apply(lab[..., 0])
    out = f32(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))
    hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.35, 0, 1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def fade(img: np.ndarray) -> np.ndarray:
    out = 0.12 + img * 0.68
    return cast(out, (1.08, 1.0, 0.8))


# ----------------------------------------------------------------------------- sources

def src_images() -> dict[str, np.ndarray]:
    from skimage import data
    s: dict[str, np.ndarray] = {
        "astronaut": f32(data.astronaut()),
        "coffee": f32(data.coffee()),
        "chelsea": f32(data.chelsea()),
        "rocket": f32(data.rocket()),
    }
    if (DL / "t1.jpg").is_file():
        s["t1"] = f32(np.asarray(Image.open(DL / "t1.jpg").convert("RGB")))
    for i in (1, 5, 9):
        p = DL / "mcmaster" / f"{i}.tif"
        if p.is_file():
            s[f"mcm{i}"] = f32(np.asarray(Image.open(p).convert("RGB")))
    return s


def ui_screenshot(w: int = 800, h: int = 500) -> np.ndarray:
    im = Image.new("RGB", (w, h), (246, 247, 249))
    d = ImageDraw.Draw(im)
    fonts = {}
    for sz in (9, 11, 13, 16, 22):
        try:
            fonts[sz] = ImageFont.truetype("DejaVuSans.ttf", sz)
        except OSError:
            try:
                fonts[sz] = ImageFont.truetype("arial.ttf", sz)
            except OSError:
                fonts[sz] = ImageFont.load_default()
    d.rectangle([0, 0, w, 44], fill=(32, 45, 70))
    d.text((16, 11), "Project Dashboard — Q3 Review", fill=(255, 255, 255), font=fonts[22])
    d.rectangle([0, 44, 180, h], fill=(229, 233, 240))
    for i, t in enumerate(["Overview", "Reports", "Settings", "Team", "Billing", "Help"]):
        y = 64 + i * 30
        d.rectangle([12, y, 24, y + 12], outline=(80, 90, 110), width=1)
        d.text((32, y - 2), t, fill=(30, 35, 45), font=fonts[13])
    d.line([180, 44, 180, h], fill=(190, 196, 206), width=1)
    for r in range(8):
        y = 70 + r * 22
        d.line([200, y + 18, w - 20, y + 18], fill=(220, 224, 230), width=1)
        d.text((204, y), f"Invoice #{1042 + r}   2026-0{r % 9 + 1}-1{r}   $1,{r}49.00   Approved",
               fill=(35, 40, 50), font=fonts[11])
    d.text((204, 260), "Small print: terms apply. Version 4.12.7 build 20261103 (rc) — ID 7f3a-9c21",
           fill=(90, 95, 105), font=fonts[9])
    for i, col in enumerate([(220, 60, 60), (60, 170, 90), (50, 110, 210), (240, 170, 30)]):
        x = 210 + i * 120
        d.rounded_rectangle([x, 300, x + 100, 330], 6, fill=col)
        d.text((x + 14, 307), ["Delete", "Save", "Share", "Export"][i], fill=(255, 255, 255), font=fonts[13])
    for i in range(10):
        d.ellipse([210 + i * 30, 360, 226 + i * 30, 376], outline=(60, 60, 60), width=2)
        d.line([210 + i * 30, 400, 226 + i * 30, 416], fill=(60, 60, 60), width=2)
    d.text((204, 440), "The quick brown fox jumps over the lazy dog 0123456789", fill=(20, 20, 20), font=fonts[16])
    return f32(np.asarray(im))


# ----------------------------------------------------------------------------- writing helpers

class Corpus:
    def __init__(self, root: Path):
        self.root = root
        self.items: list[dict[str, Any]] = []
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)

    def path(self, rel: str) -> Path:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def png(self, rel: str, img: np.ndarray, text: dict[str, str] | None = None, bits: int = 8,
            alpha: np.ndarray | None = None) -> str:
        p = self.path(rel)
        if bits == 16:
            arr = (np.clip(img, 0, 1) * 65535 + 0.5).astype(np.uint16)
            if alpha is not None:
                arr = np.dstack([arr, (alpha * 65535).astype(np.uint16)])
                cv2.imwrite(str(p), cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA))
            else:
                cv2.imwrite(str(p), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
            return rel
        info = PngImagePlugin.PngInfo()
        for k, v in (text or {}).items():
            info.add_text(k, v)
        mode_img = u8(img) if alpha is None else np.dstack([u8(img), u8(alpha)])
        Image.fromarray(mode_img).save(p, "PNG", pnginfo=info)
        return rel

    def jpg(self, rel: str, img: np.ndarray, q: int = 92, exif: bytes | None = None, icc: bytes | None = None) -> str:
        self.path(rel).write_bytes(jpeg_bytes(img, q, exif, icc))
        return rel

    def tif16(self, rel: str, img: np.ndarray, icc: bytes | None = None) -> str:
        import tifffile
        kw = {"photometric": "rgb", "compression": "zlib"}
        if icc:
            kw["iccprofile"] = icc
        tifffile.imwrite(self.path(rel), (np.clip(img, 0, 1) * 65535 + 0.5).astype(np.uint16), **kw)
        return rel

    def add(self, **kw: Any) -> None:
        kw.setdefault("synthetic", True)
        kw.setdefault("truth_issues", [])
        kw.setdefault("expect", {})
        self.items.append(kw)


def exif_bytes(make="Canon", model="EOS R6", iso=100, dt="2019:07:14 18:22:05", orientation=1, software=None,
               gps=None, copyright_=None) -> bytes:
    ex = Image.Exif()
    ex[0x010F] = make
    ex[0x0110] = model
    ex[0x0112] = orientation
    if software:
        ex[0x0131] = software
    if copyright_:
        ex[0x8298] = copyright_
    sub = ex.get_ifd(0x8769)
    sub[0x8827] = iso
    sub[0x9003] = dt
    from PIL.TiffImagePlugin import IFDRational
    sub[0x829A] = IFDRational(1, 125)
    sub[0x829D] = IFDRational(28, 10)
    if gps:
        g = ex.get_ifd(0x8825)
        lat, lon = gps
        g[1] = "N" if lat >= 0 else "S"
        g[2] = _dms(abs(lat))
        g[3] = "E" if lon >= 0 else "W"
        g[4] = _dms(abs(lon))
    return ex.tobytes()


def _dms(v: float):
    d = int(v)
    m = int((v - d) * 60)
    s = round(((v - d) * 60 - m) * 60, 2)
    return (float(d), float(m), float(s))


# ----------------------------------------------------------------------------- build

def build(out: Path | None = None) -> Path:
    out = out or PROJECT_ROOT / "fixtures" / "corpus"
    C = Corpus(out)
    S = src_images()
    icc = {n: (DL / f"{n}.icc").read_bytes() for n in ("AdobeCompat-v4", "DisplayP3-v4", "ProPhoto-v4")
           if (DL / f"{n}.icc").is_file()}

    # 1. old phone ------------------------------------------------------------------------------------
    a = S["astronaut"]
    gt1 = down(a, 320 / 512)
    deg = jpeg_roundtrip(noise(cast(gt1, (1.12, 0.94, 0.82)), 4.0), 35)
    C.add(id="oldphone_astronaut", category="old_phone", path=C.jpg("old_phone/astronaut_oldphone.jpg", deg, q=92,
                                                                   exif=exif_bytes("Nokia", "6600", 400)),
          gt=C.png("gt/oldphone_astronaut_gt.png", gt1), gt_x2=C.png("gt/oldphone_astronaut_gt_x2.png", down(a, 640 / 512)),
          content="photo", truth_issues=["jpeg_artifacts", "color_cast", "noise", "low_resolution"], faces=1,
          notes="JPEG q35 baked in (re-saved q92 container), warm/magenta cast, σ4 noise, 320px")
    if "t1" in S:
        t1 = S["t1"]
        g = down(t1, 480 / max(t1.shape[:2]))
        d2 = jpeg_roundtrip(g, 28)
        C.add(id="oldphone_group", category="old_phone", path=C.jpg("old_phone/group_q28.jpg", d2, q=28),
              gt=C.png("gt/oldphone_group_gt.png", g), content="photo", truth_issues=["jpeg_artifacts", "low_resolution"],
              faces=6, notes="group photo, JPEG q28 (quant tables present), 480px")

    # 2. DSLR / RAW / high-res ------------------------------------------------------------------------
    c = S["coffee"]
    C.add(id="dslr_iso6400", category="dslr", path=C.jpg("dslr/coffee_iso6400.jpg", noise(c, 12.0, 0.8), q=95,
                                                         exif=exif_bytes("Canon", "EOS R6", 6400)),
          gt=C.png("gt/dslr_coffee_gt.png", c), content="photo", truth_issues=["noise"], faces=0,
          notes="σ12 luma + chroma noise, ISO 6400 in EXIF, q95")
    if "mcm1" in S and "DisplayP3-v4" in icc:
        C.add(id="dslr_16bit_p3", category="dslr", path=C.tif16("dslr/mcm1_16bit_p3_noisy.tif", noise(S["mcm1"], 8.0),
                                                                icc["DisplayP3-v4"]),
              gt=C.png("gt/dslr_mcm1_gt.png", S["mcm1"]), content="photo", truth_issues=["noise"],
              notes="16-bit TIFF, Display P3 ICC, σ8 noise", icc="Display P3")
    if (DL / "iss030e122639.NEF").is_file():
        shutil.copy(DL / "iss030e122639.NEF", C.path("dslr/iss030e122639.NEF"))
        C.add(id="raw_nef", category="dslr", path="dslr/iss030e122639.NEF", synthetic=False, content="photo",
              notes="Real Nikon NEF (NASA ISS image, public domain) – decode/metadata test, no ground truth")
    C.add(id="dslr_clean", category="dslr", path=C.jpg("dslr/chelsea_clean.jpg", S["chelsea"], q=97,
                                                       exif=exif_bytes("Sony", "ILCE-7M3", 100)),
          gt=C.png("gt/dslr_chelsea_gt.png", S["chelsea"]), content="photo", truth_issues=[],
          expect={"no_op": True}, notes="Clean q97 photo: correct Auto outcome is no change")

    # 3. portrait identity ----------------------------------------------------------------------------
    C.add(id="portrait_clean", category="portrait", path=C.png("portrait/astronaut_clean.png", a),
          gt=C.png("gt/portrait_astronaut_gt.png", a), content="photo", truth_issues=[], faces=1,
          expect={"no_op": True}, notes="clean portrait, 512px: no change expected at ×1")
    C.add(id="portrait_x2", category="portrait", path=C.png("portrait/astronaut_half.png", down(a, 0.5)),
          gt="gt/portrait_astronaut_gt.png", content="photo", truth_issues=["low_resolution"], faces=1, task="x2",
          notes="bicubic-area ×0.5; upscale ×2 identity/beautification test")
    C.add(id="portrait_x4", category="portrait", path=C.png("portrait/astronaut_quarter.png", down(a, 0.25)),
          gt="gt/portrait_astronaut_gt.png", content="photo", truth_issues=["low_resolution"], faces=1, task="x4",
          notes="×0.25; ×4 upscale identity/beautification test (all ×4 candidates incl. RealPLKSR)")
    if "t1" in S:
        g = down(S["t1"], 0.5)
        C.add(id="group_x2", category="portrait", path=C.png("portrait/group_half.png", g),
              gt=C.png("gt/group_gt.png", S["t1"]), content="photo", truth_issues=["low_resolution"], faces=6,
              task="x2", notes="multi-face ×2 identity test")
        C.add(id="group_x4", category="portrait", path=C.png("portrait/group_quarter.png", down(S["t1"], 0.25)),
              gt="gt/group_gt.png", content="photo", truth_issues=["low_resolution"], faces=6, task="x4",
              notes="multi-face ×4 identity test")

    # 4. screenshots ------------------------------------------------------------------------------------
    ui = ui_screenshot()
    C.add(id="ui_clean", category="screenshot", path=C.png("screenshot/ui_clean.png", ui), gt=C.png("gt/ui_gt.png", ui),
          content="screenshot", truth_issues=[], expect={"no_op": True}, notes="clean UI with 9–22px text")
    C.add(id="ui_jpeg", category="screenshot", path=C.jpg("screenshot/ui_q45.jpg", ui, q=45), gt="gt/ui_gt.png",
          content="screenshot", truth_issues=["jpeg_artifacts"], notes="UI saved at q45")
    C.add(id="ui_x2", category="screenshot", path=C.png("screenshot/ui_half.png", down(ui, 0.5)), gt="gt/ui_gt.png",
          content="screenshot", truth_issues=["low_resolution"], task="x2", notes="UI ×0.5 → ×2 text fidelity")

    # 5. AI-generated stand-ins ---------------------------------------------------------------------------
    sq = cv2.resize(S["rocket"][:, 80:80 + 427], (1024, 1024), interpolation=cv2.INTER_CUBIC)
    wf = json.dumps({"3": {"class_type": "KSampler", "inputs": {"seed": 42, "steps": 28}},
                     "4": {"class_type": "UNETLoader", "inputs": {"unet_name": "flux1-dev.safetensors"}}})
    C.add(id="ai_scene", category="ai_generated", path=C.png("ai/scene_comfy.png", sq, {"prompt": wf, "workflow": wf}),
          gt=C.png("gt/ai_scene_gt.png", sq), content="ai_generated", truth_issues=[], expect={"no_op": True},
          notes="STAND-IN (real photo + ComfyUI prompt/workflow chunks); not a genuine FLUX/Qwen/Grok output")
    ap = cv2.resize(a, (1024, 1024), interpolation=cv2.INTER_CUBIC)
    C.add(id="ai_portrait", category="ai_generated",
          path=C.png("ai/portrait_a1111.png", ap, {"parameters": "portrait photo, astronaut\nSteps: 30, Sampler: "
                                                                 "DPM++ 2M, CFG scale: 5, Seed: 7, Model: qwen-image"}),
          gt=C.png("gt/ai_portrait_gt.png", ap), content="ai_generated", truth_issues=[], faces=1,
          expect={"no_op": True}, notes="STAND-IN with A1111-style parameters chunk")
    C.add(id="ai_portrait_jpeg", category="ai_generated",
          path=C.jpg("ai/portrait_grok_q60.jpg", ap, q=60), gt="gt/ai_portrait_gt.png", content="ai_generated",
          truth_issues=["jpeg_artifacts"], faces=1,
          notes="STAND-IN for a metadata-stripped, re-compressed AI image (as from social platforms): routing must "
                "rely on pixels; content class not recoverable from metadata")

    # 6. previously edited --------------------------------------------------------------------------------
    ch = S["chelsea"]
    C.add(id="edited_oversharp", category="edited",
          path=C.jpg("edited/chelsea_oversharp.jpg", oversharpen(ch), q=93,
                     exif=exif_bytes("Fujifilm", "X-T3", 200, software="Adobe Photoshop 25.0")),
          gt="gt/dslr_chelsea_gt.png", content="photo", truth_issues=["oversharpened"],
          expect={"forbid_kinds": ["sharpen"]}, notes="unsharp 2.2/r2 (halos)")
    C.add(id="edited_oversmooth", category="edited",
          path=C.jpg("edited/astronaut_smooth.jpg", oversmooth(a), q=93,
                     exif=exif_bytes("Apple", "iPhone 12", 50, software="Facetune 2")),
          gt="gt/portrait_astronaut_gt.png", content="photo", truth_issues=["oversmoothed"], faces=1,
          expect={"forbid_kinds": ["denoise", "sharpen"]}, notes="3× bilateral (plastic skin)")
    C.add(id="edited_hdr", category="edited", path=C.jpg("edited/coffee_hdr.jpg", hdr_look(c), q=93,
                                                         exif=exif_bytes("Google", "Pixel 7", 100, software="Snapseed")),
          gt="gt/dslr_coffee_gt.png", content="photo", truth_issues=["oversharpened"],
          expect={"forbid_kinds": ["sharpen", "tone"]}, notes="CLAHE + saturation HDR look")
    if "mcm5" in S:
        C.add(id="scan_faded", category="edited",
              path=C.jpg("edited/print_scan_faded.jpg", fade(S["mcm5"]), q=95,
                         exif=exif_bytes("EPSON", "Perfection V600 scanner", 100)),
              gt=C.png("gt/scan_faded_gt.png", S["mcm5"]), content="scan", truth_issues=["faded", "color_cast"],
              notes="faded print: lifted blacks, compressed range, yellow cast")

    # 7. cutout / recolor -----------------------------------------------------------------------------
    if (DL / "children-alpha.png").is_file():
        rgba = np.asarray(Image.open(DL / "children-alpha.png").convert("RGBA")).astype(np.float32) / 255
        fg, al = rgba[..., :3], rgba[..., 3]
        H, W = al.shape
        from skimage import data
        grass = cv2.resize(f32(np.dstack([data.grass()] * 3)), (W, H))
        bgs = {"plain": np.full_like(fg, 0.55),
               "texture": grass * np.array([0.6, 0.9, 0.5], np.float32),
               "similar": np.ones_like(fg) * fg[al > 0.9].mean(0)}
        C.png("gt/children_alpha.png", al)
        for k, bg in bgs.items():
            comp = fg * al[..., None] + bg * (1 - al[..., None])
            C.add(id=f"cutout_{k}", category="cutout", path=C.png(f"cutout/children_{k}.png", comp),
                  gt_alpha="gt/children_alpha.png", content="photo", task="cutout",
                  notes=f"children composited on {k} background; GT alpha from Real-ESRGAN test asset")
    # recolor seeds: astronaut's orange suit sleeve region and the flag
    C.add(id="recolor_suit", category="recolor", path="portrait/astronaut_clean.png", content="photo", task="recolor",
          seeds=[[150, 450]], target="#2f6fd6", notes="recolor the orange suit at seed")
    C.add(id="recolor_coffee_cup", category="recolor", path=C.png("recolor/coffee.png", c), content="photo",
          task="recolor", seeds=[[300, 50]], target="#c0392b", notes="recolor background/saucer region at seed")

    # 8. metadata stress -------------------------------------------------------------------------------
    for f, note in (("DSCN0010_gps.jpg", "real Nikon EXIF with GPS"), ("portrait_6.jpg", "EXIF Orientation=6"),
                    ("Canon_40D.jpg", "Canon MakerNote")):
        if (DL / f).is_file():
            shutil.copy(DL / f, C.path(f"metadata/{f}"))
            C.add(id=f"meta_{Path(f).stem}", category="metadata", path=f"metadata/{f}", synthetic=False,
                  content="photo", task="metadata", notes=note + " (exif-samples, real camera files)")
    full = C.jpg("metadata/full_metadata.jpg", S["chelsea"], q=93,
                 exif=exif_bytes("Nikon", "D850", 800, "2018:05:02 10:11:12", orientation=8,
                                 gps=(48.8584, 2.2945), copyright_="© Test Owner"))
    _add_iptc_xmp(C.path(full))
    C.add(id="meta_full", category="metadata", path=full, content="photo", task="metadata",
          notes="EXIF+GPS+Orientation 8+IPTC keywords/city+XMP subject/rights+copyright")
    good = (DL / "Canon_40D.jpg").read_bytes() if (DL / "Canon_40D.jpg").is_file() else jpeg_bytes(S["coffee"], 90)
    i = good.find(b"\xff\xe1")
    bad = good[:i + 4] + b"Exif\x00\x00II*\x00\xff\xff\xff\x7f" + good[i + 18:] if i > 0 else good
    C.path("metadata/malformed_exif.jpg").write_bytes(bad)
    C.add(id="meta_malformed", category="metadata", path="metadata/malformed_exif.jpg", content="photo",
          task="metadata", notes="APP1 EXIF header corrupted (bad IFD offset)")
    if "AdobeCompat-v4" in icc:
        C.add(id="meta_adobergb", category="metadata",
              path=C.jpg("metadata/adobergb.jpg", S["coffee"], q=94, icc=icc["AdobeCompat-v4"]), content="photo",
              task="metadata", icc="Adobe RGB compatible", notes="Adobe RGB (compatible) ICC JPEG")
    if "ProPhoto-v4" in icc:
        C.add(id="meta_prophoto16", category="metadata",
              path=C.tif16("metadata/prophoto_16bit.tif", S["chelsea"], icc["ProPhoto-v4"]), content="photo",
              task="metadata", icc="ProPhoto", notes="16-bit TIFF with ProPhoto ICC")
    if (DL / "tree_alpha_16bit.png").is_file():
        shutil.copy(DL / "tree_alpha_16bit.png", C.path("metadata/tree_alpha_16bit.png"))
        C.add(id="meta_alpha16", category="metadata", path="metadata/tree_alpha_16bit.png", content="photo",
              task="metadata", synthetic=False, notes="16-bit RGBA PNG (alpha preservation)")
    try:
        import pillow_heif
        hp = C.path("metadata/astronaut.heic")
        hi = Image.fromarray(u8(a))
        hi.save(hp, "HEIF", quality=90, exif=exif_bytes("Apple", "iPhone 15 Pro", 64, "2024:06:01 09:00:00"))
        C.add(id="meta_heic", category="metadata", path="metadata/astronaut.heic", content="photo", task="metadata",
              faces=1, notes=f"HEIC written by pillow-heif {pillow_heif.__version__} with EXIF")
    except Exception as e:  # pragma: no cover
        print("HEIC fixture skipped:", e)
    # duplicate filenames + exact duplicate content
    j1 = jpeg_bytes(S["coffee"], 90)
    for rel, data in (("dupes/a/IMG_0001.jpg", j1), ("dupes/b/IMG_0001.jpg", jpeg_bytes(S["chelsea"], 90)),
                      ("dupes/a/IMG_0001.png", None), ("dupes/b/copy_of_a.jpg", j1)):
        if data is None:
            C.png(rel, S["rocket"])
        else:
            C.path(rel).write_bytes(data)
    C.add(id="dupes", category="metadata", path="dupes", content="photo", task="batch",
          notes="same filename in two folders, same stem different ext, exact duplicate bytes")

    # 9. batch reliability ------------------------------------------------------------------------------
    C.path("broken/truncated.jpg").write_bytes(j1[: len(j1) // 3])
    C.path("broken/random_bytes.jpg").write_bytes(RNG.integers(0, 255, 5000, dtype=np.uint8).tobytes())
    C.path("broken/empty.png").write_bytes(b"")
    C.path("broken/text_named.tif").write_bytes(b"this is not an image\n" * 20)
    C.jpg("broken/good_1.jpg", down(S["chelsea"], 0.5), q=88)
    C.jpg("broken/good_2.jpg", down(S["coffee"], 0.5), q=60)
    C.add(id="broken", category="batch", path="broken", content="photo", task="batch",
          notes="2 good + 4 broken files (truncated, random, empty, text)")

    # 10. real-world, no ground truth ---------------------------------------------------------------------
    for f in sorted((DL / "realsr").glob("*")):
        shutil.copy(f, C.path(f"real/{f.name}"))
        cls = "screenshot" if f.stem in ("ppt3",) else "photo"
        C.add(id=f"real_{f.stem}", category="real", path=f"real/{f.name}", synthetic=False, content=cls,
              notes="RealSRSet real-world degraded image (SwinIR testsets); no ground truth: fidelity assessment "
                    "and visual review only")

    (out / "manifest.json").write_text(json.dumps({"items": C.items}, indent=2), "utf-8")
    print(f"corpus: {len(C.items)} items -> {out}")
    return out


RAW = "https://raw.githubusercontent.com"
PUBLIC_ASSETS = {
    "t1.jpg": f"{RAW}/deepinsight/insightface/master/python-package/insightface/data/images/t1.jpg",
    "Tom_Hanks_54745.png": f"{RAW}/deepinsight/insightface/master/python-package/insightface/data/images/Tom_Hanks_54745.png",
    "children-alpha.png": f"{RAW}/xinntao/Real-ESRGAN/master/inputs/children-alpha.png",
    "tree_alpha_16bit.png": f"{RAW}/xinntao/Real-ESRGAN/master/inputs/tree_alpha_16bit.png",
    "iss030e122639.NEF": f"{RAW}/letmaik/rawpy/main/test/iss030e122639.NEF",
    "AdobeCompat-v4.icc": f"{RAW}/saucecontrol/Compact-ICC-Profiles/master/profiles/AdobeCompat-v4.icc",
    "DisplayP3-v4.icc": f"{RAW}/saucecontrol/Compact-ICC-Profiles/master/profiles/DisplayP3-v4.icc",
    "ProPhoto-v4.icc": f"{RAW}/saucecontrol/Compact-ICC-Profiles/master/profiles/ProPhoto-v4.icc",
    "DSCN0010_gps.jpg": f"{RAW}/ianare/exif-samples/master/jpg/gps/DSCN0010.jpg",
    "portrait_6.jpg": f"{RAW}/ianare/exif-samples/master/jpg/orientation/portrait_6.jpg",
    "Canon_40D.jpg": f"{RAW}/ianare/exif-samples/master/jpg/Canon_40D.jpg",
    **{f"mcmaster/{i}.tif": f"{RAW}/JingyunLiang/SwinIR/main/testsets/McMaster/{i}.tif" for i in (1, 5, 9, 13, 17)},
    **{f"realsr/{n}": f"{RAW}/JingyunLiang/SwinIR/main/testsets/RealSRSet+5images/{n}" for n in (
        "oldphoto2.png", "oldphoto3.png", "oldphoto6.png", "Lincoln.png", "ppt3.png", "comic1.png",
        "dped_crop00061.png", "0030.jpg", "ADE_val_00000114.jpg")},
}


def fetch_public_assets() -> int:
    """User-directed download of the public test images used by the corpus (setup step, not runtime)."""
    import urllib.request
    n = 0
    for rel, url in PUBLIC_ASSETS.items():
        dest = DL / rel
        if dest.is_file() and dest.stat().st_size:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                dest.write_bytes(r.read())
            n += 1
            print("fetched", rel)
        except Exception as e:
            print("could not fetch", rel, e)
    return n


def _add_iptc_xmp(path: Path) -> None:
    import pyexiv2
    from ..core.metadata import EXIV2_LOCK
    with EXIV2_LOCK:
        img = pyexiv2.Image(str(path))
        try:
            img.modify_iptc({"Iptc.Application2.Keywords": ["cat", "test", "archive"],
                             "Iptc.Application2.City": "Paris", "Iptc.Application2.CountryName": "France",
                             "Iptc.Application2.Byline": "Test Photographer"})
            img.modify_xmp({"Xmp.dc.subject": ["cat", "test"], "Xmp.dc.rights": "© Test Owner",
                            "Xmp.photoshop.City": "Paris", "Xmp.crs.Exposure2012": "+0.50",
                            "Xmp.xmp.Rating": "4"})
        finally:
            img.close()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--fetch", action="store_true", help="download public test assets first")
    a = ap.parse_args()
    if a.fetch:
        fetch_public_assets()
    build(Path(a.out) if a.out else None)

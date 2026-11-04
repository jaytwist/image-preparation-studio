"""Held-out validation set — never used for threshold tuning.

Protocol (see docs/TEST_RESULTS.md):
* Built only from sources that were not inspected during calibration of ``classify.THRESHOLDS``.
* Thresholds are frozen at the commit recorded in the report before the held-out set is evaluated.
* If any rule is changed after looking at held-out results, those results are reported as
  "post-hoc" and a new held-out set is required for an unbiased estimate.

Groups:
  A. Historical / real degraded photos (no ground truth): CodeFormer & GFPGAN real low-quality inputs,
     Solvay Conference 1927 faces, RealSRSet scenes not used in tuning.
  B. Genuine AI-generated images (FLUX, FLUX.2, Chroma, Qwen-Image, HiDream, SD3.5, Hunyuan) from
     ComfyUI_examples, with their embedded workflow metadata. As-is (expected: no restoration) and
     metadata-stripped re-compressed copies (ground truth = the original).
  C. Real UI screenshots (chaiNNer docs): as-is and JPEG/downscaled copies (ground truth = original).
  D. Synthetic edits/degradations on unseen clean sources (Set5 head/woman/baby/bird, McMaster 13/17)
     with operators/parameters that differ from the tuning corpus (cool/green casts, Gaussian noise in
     lossless PNG, beauty-filter skin smoothing, local-contrast HDR, oversharpen r=1, JPEG q20/q50).

    python -m ips.eval.holdout --fetch      # download public sources (setup step) and build
"""
from __future__ import annotations

import argparse
import io
import json
import zlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ..paths import PROJECT_ROOT
from .fixtures import Corpus, cast, down, f32, jpeg_roundtrip, noise

RAW = "https://raw.githubusercontent.com"
DLH = PROJECT_ROOT / "fixtures" / "_downloads" / "holdout"
OUT = PROJECT_ROOT / "fixtures" / "holdout"
RNG = np.random.default_rng(98765)  # different seed from the tuning corpus

SOURCES = {
    # A. historical / real degraded
    **{f"hist/codeformer_{n}": f"{RAW}/sczhou/CodeFormer/master/inputs/whole_imgs/{n}"
       for n in ("00.jpg", "01.jpg", "02.png", "03.jpg", "04.jpg", "05.jpg", "06.png")},
    "hist/gfpgan_10045.png": f"{RAW}/TencentARC/GFPGAN/master/inputs/whole_imgs/10045.png",
    "hist/solvay_1927_0018.png": f"{RAW}/sczhou/CodeFormer/master/inputs/cropped_faces/Solvay_conference_1927_0018.png",
    "hist/solvay_1927_2_16.png": f"{RAW}/sczhou/CodeFormer/master/inputs/cropped_faces/Solvay_conference_1927_2_16.png",
    **{f"hist/realsr_{n}": f"{RAW}/JingyunLiang/SwinIR/main/testsets/RealSRSet+5images/{n}"
       for n in ("dog.png", "frog.png", "foreman.png", "building.png", "painting.png", "tiger.png")},
    # B. genuine AI-generated (ComfyUI examples; embedded workflows)
    **{f"ai/{n.replace('/', '__')}": f"{RAW}/comfyanonymous/ComfyUI_examples/master/{n}"
       for n in ("flux/girl_in_field.png", "chroma/fennec_girl_flowers.png", "qwen_image/qwen_image_basic_example.png",
                 "flux2/flux2_example.png", "hidream/hidream_dev_example.png", "sd3/sd3.5_simple_example.png",
                 "hunyuan_image/hunyuan_image_example.png")},
    # C. real UI screenshots
    "ui/chainner_screenshot.png": f"{RAW}/chaiNNer-org/chaiNNer/main/docs/assets/screenshot.png",
    "ui/chainner_simple.png": f"{RAW}/chaiNNer-org/chaiNNer/main/docs/assets/simple_screenshot.png",
    "ui/chainner_input_override.png": f"{RAW}/chaiNNer-org/chaiNNer/main/docs/assets/input-override.png",
    # D. clean sources for synthetic degradations (unused in tuning)
    **{f"clean/set5_{n}": f"{RAW}/JingyunLiang/SwinIR/main/testsets/Set5/HR/{n}"
       for n in ("head.png", "woman.png", "baby.png", "bird.png")},
    **{f"clean/mcm_{i}.tif": f"{RAW}/JingyunLiang/SwinIR/main/testsets/McMaster/{i}.tif" for i in (13, 17)},
}


def fetch() -> None:
    import urllib.request
    for rel, url in SOURCES.items():
        d = DLH / rel
        if d.is_file() and d.stat().st_size:
            continue
        d.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                d.write_bytes(r.read())
            print("fetched", rel)
        except Exception as e:
            print("could not fetch", rel, e)


def load(rel: str) -> np.ndarray | None:
    p = DLH / rel
    if not p.is_file():
        return None
    return f32(np.asarray(Image.open(p).convert("RGB")))


def strip_and_jpeg(img: np.ndarray, q: int) -> bytes:
    bio = io.BytesIO()
    Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)).save(bio, "JPEG", quality=q)
    return bio.getvalue()


def beauty_filter(img: np.ndarray) -> np.ndarray:
    """Typical 'beauty' skin smoothing: guided-filter smoothing restricted to skin-coloured pixels."""
    u8 = (img * 255).astype(np.uint8)
    ycrcb = cv2.cvtColor(u8, cv2.COLOR_RGB2YCrCb)
    skin = cv2.inRange(ycrcb, (0, 135, 85), (255, 180, 135)).astype(np.float32) / 255
    skin = cv2.GaussianBlur(skin, (0, 0), 3)
    from ..core.ops import guided_filter
    sm = np.dstack([guided_filter(img[..., c], img[..., c], r=6, eps=4e-3) for c in range(3)])
    return np.clip(img * (1 - skin[..., None]) + sm * skin[..., None], 0, 1).astype(np.float32)


def local_contrast_hdr(img: np.ndarray) -> np.ndarray:
    base = cv2.GaussianBlur(img, (0, 0), 25)
    return np.clip(img + 0.9 * (img - base), 0, 1).astype(np.float32)


def build() -> Path:
    C = Corpus(OUT)
    # A. historical / real (no GT)
    for rel in [r for r in SOURCES if r.startswith("hist/")]:
        im = load(rel)
        if im is None:
            continue
        name = Path(rel).name
        p = C.path(f"historical/{name}")
        p.write_bytes((DLH / rel).read_bytes())
        C.add(id=f"hist_{Path(name).stem}", category="historical", path=f"historical/{name}", synthetic=False,
              content="photo", notes="real-world degraded/historical photo; no ground truth (visual review)")
    # B. genuine AI
    for rel in [r for r in SOURCES if r.startswith("ai/")]:
        im = load(rel)
        if im is None:
            continue
        name = Path(rel).name
        C.path(f"ai/{name}").write_bytes((DLH / rel).read_bytes())
        C.add(id=f"ai_{Path(name).stem}", category="ai_generated", path=f"ai/{name}", synthetic=False,
              content="ai_generated", truth_issues=[], expect={"no_restoration": True},
              notes="genuine AI-generated image with embedded ComfyUI workflow")
        gt = C.png(f"gt/ai_{Path(name).stem}.png", im)
        q = int(RNG.choice([40, 55, 70]))
        jp = f"ai_stripped/{Path(name).stem}_q{q}.jpg"
        C.path(jp).write_bytes(strip_and_jpeg(im, q))
        C.add(id=f"aijpg_{Path(name).stem}", category="ai_generated", path=jp, gt=gt, content="ai_generated",
              truth_issues=["jpeg_artifacts"] if q <= 55 else [], synthetic=True,
              notes=f"genuine AI image, metadata stripped, re-saved JPEG q{q} (as from social platforms)")
    # C. real screenshots
    for rel in [r for r in SOURCES if r.startswith("ui/")]:
        im = load(rel)
        if im is None:
            continue
        stem = Path(rel).stem
        C.path(f"ui/{stem}.png").write_bytes((DLH / rel).read_bytes())
        C.add(id=f"ui_{stem}", category="screenshot", path=f"ui/{stem}.png", synthetic=False, content="screenshot",
              truth_issues=[], expect={"no_op_x1": True}, notes="real UI screenshot (PNG)")
        gt = C.png(f"gt/ui_{stem}.png", im)
        C.path(f"ui_jpeg/{stem}_q35.jpg").write_bytes(strip_and_jpeg(im, 35))
        C.add(id=f"uijpg_{stem}", category="screenshot", path=f"ui_jpeg/{stem}_q35.jpg", gt=gt, content="screenshot",
              truth_issues=["jpeg_artifacts"], notes="real screenshot re-saved at q35")
    # D. synthetic on unseen clean sources
    srcs = {Path(r).stem: load(r) for r in SOURCES if r.startswith("clean/")}
    srcs = {k: v for k, v in srcs.items() if v is not None}
    recipes = [
        ("cool_cast", lambda x: cast(x, (0.85, 0.97, 1.18)), ["color_cast"]),
        ("green_cast", lambda x: cast(x, (0.9, 1.12, 0.88)), ["color_cast"]),
        ("gauss_noise_png", lambda x: noise(x, 9.0, 0.6), ["noise"]),
        ("jpeg_q20", lambda x: jpeg_roundtrip(x, 20), ["jpeg_artifacts"]),
        ("beauty_filter", beauty_filter, ["oversmoothed"]),
        ("hdr_local_contrast", local_contrast_hdr, ["oversharpened"]),
        ("oversharp_r1", lambda x: np.clip(x + 1.5 * (x - cv2.GaussianBlur(x, (0, 0), 1.0)), 0, 1), ["oversharpened"]),
        ("clean", lambda x: x, []),
    ]
    for k, im in srcs.items():
        gt = C.png(f"gt/{k}.png", im)
        for name, fn, truth in recipes:
            # each source gets a subset so the set stays small and varied
            always = name in ("clean", "cool_cast") or (name == "beauty_filter" and k in ("set5_head", "set5_woman"))
            if (zlib.crc32((k + name).encode()) % 3) and not always:
                continue
            deg = fn(im)
            fmt_png = name in ("gauss_noise_png", "clean", "beauty_filter")
            rel = f"synthetic/{k}__{name}.{'png' if fmt_png else 'jpg'}"
            if fmt_png:
                C.png(rel, deg)
            else:
                C.jpg(rel, deg, q=95)
            faces = 1 if k in ("set5_head", "set5_woman", "set5_baby") else 0
            C.add(id=f"syn_{k}_{name}", category="synthetic_holdout", path=rel, gt=gt, content="photo",
                  truth_issues=truth, faces=faces, expect=({"no_restoration": True} if name == "clean" else
                                                         {"forbid_kinds": ["sharpen"]} if "sharp" in name or "hdr" in name else
                                                         {"forbid_kinds": ["denoise", "sharpen"]} if name == "beauty_filter" else {}),
                  notes=f"{name} on unseen source {k}")
        # identity regression: ×2 portrait on unseen faces
        if k in ("set5_head", "set5_woman"):
            C.add(id=f"syn_{k}_x2", category="synthetic_holdout", path=C.png(f"synthetic/{k}__half.png", down(im, 0.5)),
                  gt=gt, content="photo", truth_issues=["low_resolution"], faces=1, task="x2",
                  notes="unseen face, ×0.5 -> ×2 identity regression")
    (OUT / "manifest.json").write_text(json.dumps({"items": C.items, "protocol": "held-out; thresholds frozen"},
                                                  indent=2), "utf-8")
    print(f"holdout: {len(C.items)} items -> {OUT}")
    return OUT


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch()
    build()

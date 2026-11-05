# Release notes — 0.9.0rc1 (2026-11-04)

**Status: release candidate, Windows GPU Validation Pending.** Functionally complete and tested on CPU
(Linux container). Not yet run on the target Windows 11 / RTX 5070 Ti machine; see
WINDOWS_VALIDATION.md for the 10-step procedure that closes this gap.

## Genuinely working (exercised by automated tests in the build environment)
* **Import**: JPEG/PNG/TIFF/WebP/BMP, 16-bit PNG/TIFF, alpha, HEIC (8-bit), camera RAW (developed via
  LibRaw, e.g. Nikon NEF), CMYK→RGB; EXIF orientation applied; corrupt/empty/mislabelled files fail
  individually with a clear reason. Sources are read-only (hash verified unchanged in tests).
* **Smart Auto** (white balance is automatic only for scans; reported/offered in Manual for photos): metadata + pixel diagnostics, content classes, issue severities with evidence and
  confidence, FBCNN blind-QF verification of JPEG quality, three-estimator colour-cast consensus,
  intent preservation (grain, warm light, toned monochrome, AI style), no-op outcome, rationale text,
  review flags, deterministic recipe hash, learning from corrections (opt-in derivation).
* **Restoration**: FBCNN (native QF), SCUNet, DRUNet (native σ), SwinIR ×2, Real-ESRGAN (Balanced),
  classical WB/levels/contrast/halo-clamped sharpening/NL-means/Lanczos/nearest; tiled seamless
  inference; face-region protection; Faithful fidelity guard with identity-drift signals.
* **Manual**: every op editable (method, native params, post-blend, region mask), undo/redo, presets
  (6 samples), variant history with A/B selection, region crop preview, Reset to Auto.
* **Before/After**: exact alignment across resolutions, draggable wipe, before/after/side-by-side,
  Fit/100/200/400 % (device-pixel correct), wheel zoom around cursor, pan, face crop focus, full screen.
* **Quick Edit**: person/subject masks (U²-Net / IS-Net, IoU 0.98 vs ground-truth alpha on three
  backgrounds), brush/magic/rect selection with undo, feather/grow/shrink/invert, cutout to
  transparent PNG with edge controls and fringe reduction, background replace, LCh recolour preserving
  shading/patterns (outside-mask pixels bit-identical), object removal (LaMa or Telea), rotate/flip/
  crop/straighten with angle detection, edit history; outputs labelled Edited/Cutout (+generative).
* **Batch**: persistent queue, crash recovery, verified skip of completed jobs, pause/continue/cancel/
  resume/retry, duplicate skip, similar-image suggestions, per-image recipes and overrides, JSON/CSV
  reports with median/p95/max.
* **Outputs**: Archive (TIFF/PNG, source ICC, source bit depth), AI/Video Ready (sRGB, fit, optional
  pad), Share/Web (sRGB JPEG, location removal), Cutout; organised folders, deterministic collision-
  safe names, recipe sidecars, gallery with search/filter/approval/export; master reuse.
* **Metadata**: transfer + post-write audit (EXIF/IPTC/XMP/ICC), orientation/dimension/DPI updates,
  MakerNote/thumbnail/RAW/crs exclusions disclosed, GPS strip verified, serialised pyexiv2 (16-thread test).
* **ComfyUI**: isolated localhost client (tested against a fake server), Manual upscale through the
  user's ComfyUI models.
* **Privacy**: onnxruntime telemetry disabled; zero outbound connections in a traced workload.
* **Performance tooling**: `tools/bench.py` defines the Fast Profile per GPU; router gates routes on it.

## Known limitations
* **No GPU measurements**: the ≤ 15 s contract is enforced by mechanism but no route is certified yet.
  CPU timings in this report are not indicative of the RTX 5070 Ti.
* Windows desktop behaviour (HiDPI scaling factors, Explorer drag-and-drop, file associations) is
  untested; UI was tested offscreen on Linux.
* Auto calibration uses synthetic degradations + 9 public real images; real-world fidelity on the
  user's photos is **unverified** until reviewed. Colour-cast and fade detection are deliberately
  conservative and will miss some genuine casts (e.g. a cast in a scene dominated by the same hue).
* Balanced (GAN) upscalers visibly smooth skin on the portrait benchmark; they are never Auto-default
  in Faithful mode and always go to Review.
* No automatic deblurring; Restormer rejected for v1. Generative recovery only via Qwen-Image 2.1 on the user's
  ComfyUI (explicit mode; output quality/identity/time unmeasured until run on the RTX 5070 Ti). No SeedVR2/SUPIR/FLUX.
* HEIC is 8-bit (no HDR gain maps); RAW is developed with camera WB; MakerNotes are not carried.
* Face detection may miss very small faces (< ~16 px at 640-px detector scale) in large group photos.
* InsightFace weights are non-commercial research licensed (optional; Haar fallback).
* Held-out validation (66 unseen items) showed Auto generalises less well than on the tuning set (ranked-first
  47 % vs 87 %); JPEG cleanup generalised, automatic white balance did not and was removed for photos (post-hoc).
* RealPLKSR beautification is not caught by automatic identity signals; it is Manual-only and always reviewed.
* After JPEG/denoise, skin-texture loss is reported as a note, not a review flag (cannot be separated from cleanup).
* Faces smaller than ~140 px are not detected, so identity is not assessed on very small images.
* Work was paused before GPU validation; see docs/RESTART.md.

## Changelog
* 0.9.0rc3 — **Generative Recovery with Qwen-Image 2.1** on the local ComfyUI (Prepare + Batch + CLI `--mode generative`):
  prompt presets (the user's old-phone prompt is the default), native redraw strength, Turbo, face reference,
  generative labelling, identity check and Review. Graph validated against real ComfyUI 0.44; generation itself not yet
  run (needs the user's GPU). Prepare side panel no longer clips at 1500 px width.
* 0.9.0rc2 — Look levels (Enhanced default, Vivid, Natural) after user feedback that repair-only results were barely visible; live Before/After preview in Batch; identity check now runs for every operation on faces; real-ComfyUI v0.44 integration fixes; one-click `run.cmd` setup.
* 0.9.0rc1 — first release candidate (this document).

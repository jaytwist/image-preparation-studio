# Smart Auto: diagnostics, rules and mappings

Code: `ips/core/diagnostics.py` (measurements) → `ips/core/classify.py` (content + issues, thresholds in
`THRESHOLDS`) → `ips/core/router.py` (recipe). Observed accuracy: `reports/eval/AUTO_ACCURACY.md`
(summarised in TEST_RESULTS.md). All thresholds below were calibrated on the synthetic corpus in this
session; they must be re-checked on real user images (Review Queue + correction log).

## 1. Inputs

**Metadata** (exiv2, Pillow fallback): camera make/model, lens, ISO, exposure, date taken, software
(editor/AI generator), GPS presence, copyright/keywords, white-balance mode, ICC, EXIF-vs-actual
dimensions, IPTC DigitalSourceType, C2PA manifest presence, PNG generation chunks (`prompt`/`workflow`
ComfyUI, `parameters` A1111), JPEG quantisation tables → IJG quality estimate.

**Pixels** (analysis copy ≤1024 px for global stats; native 512² crops for resolution-dependent ones):

| Feature | Method |
|---|---|
| noise σ (luma, chroma) | Immerkær operator on low-gradient pixels, 0–255 scale |
| JPEG blocking | phase-robust 8×8 boundary discontinuity (max-phase vs median) |
| FBCNN QF prediction | FBCNN's own blind QF head on a centre crop (only when blocking/quant tables suggest JPEG) |
| blur | Crete et al. no-reference blur; gradient structure-tensor anisotropy (motion) |
| halos / oversharpening | overshoot beyond local min/max of a Gaussian base around Canny edges, on median-filtered luma |
| texture ratio | fine (Laplacian) vs mid (DoG 1–3 px) energy off-edge — low = waxy/oversmoothed |
| exposure | luma percentiles, clipping fractions, dynamic range |
| colour cast | three illuminant estimates in linear log-chromaticity: gray-world on neutrals, gray-edge (p=6, σ=2), white-patch; agreement (min pairwise cosine), smallest magnitude, consensus vector, scene-hue dominance |
| toned monochrome | a/b spread < 6 with non-zero chroma |
| screenshot/UI | exact-flat 3×3 fraction, axis-aligned strong-edge share, colour-count ratio, glyph-like component density |
| faces | SCRFD-10G (5 landmarks) or Haar; face texture (Laplacian at 128²) |

## 2. Content class (scores, highest wins; confidence = 0.45 + margin)
* **screenshot**: flat fraction > 0.30, axis-aligned edges, text density, PNG/WebP, screenshot software; −0.5 if camera EXIF.
* **ai_generated**: generator metadata (+0.75), dims multiple of 64 without camera EXIF, near-zero noise.
  Metadata-stripped AI images fall back to *photo* routing (pixels decide the operations).
* **scan**: scanner make/model. **photo**: default; camera EXIF, RAW.

## 3. Issues (severity 0–1, confidence 0–1)

| Issue | Trigger (key thresholds) | Confidence logic |
|---|---|---|
| jpeg_artifacts | blockiness 0.12→0.70 and/or quant-table q 88→45; ignored for lossless files with > 50 % exact-flat pixels (cannot survive JPEG) | metadata+pixels agree 0.9; metadata only 0.5 (severity halved); pixels only 0.6 |
| noise | σ_luma (discounted by blocking) 2.2→7.0 | 0.7; 0.9 at ISO ≥ 1600 |
| grain intent | chroma/luma noise < 0.45 and (AI source, edit software or ISO ≤ 400) | preserved in Faithful unless severity ≥ 0.75 |
| blur_soft / blur_motion | Crete 0.42→0.65; anisotropy > 0.55 = motion | 0.55 (flag only; no automatic deblur) |
| low_resolution | long side < 1024 px | 0.95 |
| color_cast | min estimator magnitude 0.08→0.30 | 0.85 if agreement ≥ 0.85 else 0.35; ≤ 0.3 if scene-hue dominance ≥ 0.35; ×0.6 warm + modern camera + undegraded (intentional light); ×0.7 EXIF manual WB |
| underexposed / overexposed | p50 < 0.22 & p99 < 0.75 / clip > 2 % & p50 > 0.65 | 0.7 (over = flag only) |
| faded | scan: DR < 0.70 & p1 > 0.08; photo: DR < 0.50 & p1 > 0.12 | 0.7 scan / 0.5 photo |
| oversharpened | halo index 0.12→0.23 | 0.55 → no sharpening |
| oversmoothed | photo, texture ratio < 0.95, σ < 2.2 | 0.45 → flag; Faithful never synthesises texture |

## 4. Routing rules (Faithful mode; fewest necessary ops)

| Order | Operation | Rule | Model / parameters | Strength mapping |
|---|---|---|---|---|
| 1 | jpeg_cleanup | jpeg severity ≥ 0.15 | FBCNN, **qf = FBCNN blind prediction** (fallback: quant-table q, +10 if unconfirmed); screenshots +10 (gentler) + legibility flag | native QF |
| 2 | denoise | noise ≥ 0.12, not grain intent | photo → SCUNet real-PSNR, post-blend 0.55+0.45·sev; scan/AI → DRUNet σ = 1.0× per-channel estimate; faces present → 40–50 % face-region protection | SCUNet: post-blend (labelled); DRUNet: native σ |
| 3 | white_balance | **scan only** (photos: reported + offered in Manual, never automatic in Faithful — held-out evidence: automatic WB raised ΔE vs ground truth in 8/8 photo cases); cast ≥ 0.2, confidence ≥ 0.6 | consensus gains, strength min(0.8, 0.45+0.4·sev)×(1−dominance), luminance preserved | native |
| 4 | exposure | underexposed ≥ 0.2 → EV (bounded so p99 stays unclipped, ≤ +1.2); faded ≥ 0.25 → black = 0.8·p1, white = p99+0.02 | levels | native |
| 5 | upscale | scale policy: explicit, AI/Video target long edge, or ×2 when long side < 1024 (setting) | photo/AI → SwinIR ×2 (+Lanczos for remainder); screenshot → Lanczos, pixel art → nearest; Balanced → Real-ESRGAN | model scale; exact target by area/Lanczos resize |
| 6 | sharpen | blur_soft ≥ 0.25 and not oversharpened | halo-clamped luminance unsharp (cannot overshoot local min/max) | native amount |
| – | no-op | nothing above triggers and scale = 1 | "No enhancement needed" with the measured values | – |

Post-run (Faithful): **fidelity guard** (DECISIONS D6). Identity review thresholds (calibrated on tuning faces,
validated on unseen faces): embedding similarity < 0.98 after an upscale (< 0.80 at 1×), landmark drift > 0.06 IOD,
mid-band skin texture < 0.92 when no JPEG/denoise ran (after JPEG/denoise it is an informational note — legitimate
cleanup and beauty smoothing overlap), skin ΔE > 1.5 without a colour op. Tight face crops are re-detected on a padded
canvas. Any GAN ("balanced") model output goes to Review regardless of signals (RealPLKSR evades all signals). Review Queue on: identity flags, guard action,
confidence < 0.55, GAN use, Quality-Override runs.

## 4b. Look (visible enhancement layer, default **Enhanced**)
User feedback: defect-only repair looked almost unchanged on good images. A separate, explicitly selected *look*
is applied after repair (Prepare/Batch/Settings/CLI `--look`). Non-generative only — no resampling of facial
geometry, no texture synthesis; identity signals run on every job with faces.

| Look | Tone/levels | Contrast | Vibrance (skin hues protected) | Clarity (CLAHE on lightness) | Sharpen (halo-clamped) | Auto ×2 below | Face protection |
|---|---|---|---|---|---|---|---|
| Natural | – | – | – | – | – | 1024 px | – |
| **Enhanced** | stretch p1/p99 | +0.20 | +0.30 | 0.70 (clip 2.5) | 0.85 | 1536 px | 35 % |
| Vivid | stretch p1/p99 | +0.32 | +0.48 | 1.00 (clip 3.2) | 1.20 | 2048 px | 30 % |

Not applied to screenshots/UI or toned monochrome; clarity halved on previously oversharpened sources; AI
images get 85 % vibrance. Measured identity (ArcFace vs source): portrait 0.992 / 0.985, AI characters
0.982–0.99 / 0.964–0.982 (Enhanced / Vivid), no identity or geometry flags.

## 5. Learning from corrections
Exports record `accepted_auto` or `manual_export` with the auto and final recipe and features.
`python -m ips.eval.harness learn` aggregates per (content class, op kind); kinds the user removes in
≥ 70 % of ≥ 5 cases are suppressed for that class (`router_adjustments.json`, human-readable, deletable).

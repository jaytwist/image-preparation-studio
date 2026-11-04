# Test results

Release 0.9.0rc1 · **Windows GPU Validation Pending**. Everything below was executed; nothing is
extrapolated. "Pending" marks what has not been measured.

## 1. Environments actually used

| Environment | What ran | GPU |
|---|---|---|
| Linux build container (4 vCPU, 15 GB) | development, full test suite, benchmarks, evaluations | none |
| **Windows (GitHub Actions `windows-latest`)**: Windows Server 2025 build 26100 (Win 11 24H2 kernel), Python 3.12.10, torch 2.15 CPU | `run.ps1` install, model fetch with checksums, full test suite, app launch smoke, privacy audit, CPU bench smoke | none |
| Target PC (Windows 11, RTX 5070 Ti 16 GB, 64 GB RAM, ComfyUI, NH Maker 2) | **not reached**: no Remote Control folder was shared | — |

## 2. Automated test suites

| Suite | Linux | Windows CI |
|---|---|---|
| Metadata/colour/bit-depth/alpha/orientation/HEIC/RAW/no-clobber/concurrency (`test_io_metadata`) | 14/14 | ✅ |
| Smart Auto routing + regression (no learned upscaler on text, all modes) (`test_auto`) | 21/21 | ✅ |
| Batch: corrupt-file isolation, duplicates, crash recovery, verified skip, pause/cancel/resume, master reuse (`test_batch`) | 6/6 | ✅ (after fix, §6) |
| ComfyUI isolation vs fake server + pipeline route (`test_comfy_client`) | 8/8 | ✅ |
| Quick Edit: cutout IoU, recolour, mask undo, LaMa/Telea removal, geometry, straighten (`test_quickedit`) | 10/10 | ✅ |
| Models: every installed model runs, seamless tiling, native params, FBCNN blind QF, identity flags (`test_models`) | 16/16 | ✅ |
| UI offscreen: compare-view alignment math, Prepare Auto→Manual→export, Quick Edit, Batch, Gallery (`test_ui`) | 4/4 | ✅ |
| Fast-Profile gate, learning from corrections (`test_perf_learning`) | 2/2 | ✅ |
| Privacy: env enforced; strace of full workload = 0 outbound connects (`test_privacy`) | 2/2 | 1 passed, 1 skipped (strace is Linux-only) |
| **Total** | see §9 final run | **82 passed, 1 skipped** (run 2) |
| App launch smoke (`run.ps1 -Smoke`) | – | ✅ `SMOKE OK` |
| Privacy audit (`run.ps1 -PrivacyAudit`, psutil, native libraries included) | – | ✅ `disallowed: []` |

## 3. Identity / beautification evidence (tuning corpus, ground truth available)

×2 / ×4 upscaling of a public-domain portrait (NASA astronaut) and a 6-face group photo. Face similarity is
ArcFace cosine vs the ground-truth face; SSIM and ΔE are full-image. Full table: `reports/eval/MODEL_COMPARISON.md`.

| ×2 portrait | PSNR | SSIM | ΔE | face sim |
|---|---|---|---|---|
| **SwinIR ×2 (Auto Faithful)** | **30.94** | **0.961** | **1.59** | **0.972** |
| Lanczos | 30.82 | 0.953 | 1.66 | 0.972 |
| Real-ESRGAN ×4plus (→×2) | 28.65 | 0.928 | 2.69 | 0.913 |
| Real-ESRGAN general v3 (→×2) | 28.40 | 0.924 | 2.55 | 0.930 |
| Real-ESRGAN ×2plus | 27.12 | 0.899 | 2.87 | 0.808 |
| RealPLKSR Nomos2 (→×2) | 28.01 | 0.882 | 2.59 | 0.960 |

| ×4 portrait | SSIM | ΔE | face sim |
|---|---|---|---|
| SwinIR ×2 + Lanczos | **0.894** | **2.47** | 0.700 |
| Lanczos | 0.851 | 2.79 | **0.750** |
| RealPLKSR | 0.785 | 4.26 | 0.666 |
| Real-ESRGAN general v3 | 0.825 | 3.96 | 0.545 |
| Real-ESRGAN ×4plus | 0.815 | 4.09 | 0.473 |

**Visual inspection** (contact sheets, 256 px face crops): Real-ESRGAN ×2plus/×4plus visibly remove skin
texture/freckles and alter eye shape (beautification class). RealPLKSR keeps more texture but raises
contrast/saturation and reddens lips; its embedding similarity stays high (0.960) — *an embedding-only check
would have missed it*, confirming that embeddings cannot certify identity. SwinIR is closest to ground truth.

**Defect found and fixed: Studio's own review signals missed all of these.** Comparing output vs *source* at
source resolution, 0/12 GAN outputs were flagged (`reports/eval/identity_signals_vs_source.json`): upscaling
artefacts sit above the source's Nyquist limit and the 0.80 similarity threshold was an uncalibrated prior.
Recalibrated on the tuning faces only (faithful upscalers ≥ 0.993 vs GAN ≤ 0.97; beauty-filter mid-band skin
texture 0.86 vs legitimate restorations ≥ 0.98; legitimate heavy JPEG cleanup down to 0.827 similarity):
upscale threshold 0.98, mid-texture 0.92, skin ΔE 1.5 when no colour op ran.
Validation on **unseen** faces: see §5.

## 4. Text / screenshot evidence

UI fixture (9–22 px text) ×0.5 → ×2, OCR (Tesseract 5.3, dev-time only) of each output vs OCR of ground truth
(`reports/eval/ocr_ui_x2.json`):

| Candidate | OCR char error | Notable reading |
|---|---|---|
| RealPLKSR | 0.268 | **"$1,049.00" → "$1,019.00"** (legible, wrong digit) |
| SwinIR ×2 | 0.449 | "$1.049.00" (digits kept) |
| Lanczos | 0.453 | blurred, not confidently wrong |
| Real-ESRGAN general v3 | 0.465 | "$1.019.0)" (wrong digit) |
| Real-ESRGAN ×4plus | 0.491 | "Approved" → "asorwred" |
| Nearest | 0.541 | blocky |
| Real-ESRGAN ×2plus | 0.670 | "Save" → "Gave" |

Pixel metrics (SSIM, glyph-mask SSIM, per-glyph NCC) **reward** the sharper GAN outputs and do not detect
these substitutions; only recognition does. Policy kept and locked by a regression test: Auto never routes
text/UI to a learned upscaler, in any mode. JPEG cleanup on text is safe and strong: FBCNN on a q45 UI capture
lifts SSIM 0.951 → 0.993 and halves ΔE.

## 5. Smart Auto accuracy

### 5a. Tuning corpus (thresholds were calibrated on this set — optimistic)
`reports/eval/AUTO_ACCURACY.md`: content class 95 %; Auto ranked best of its ablation/substitution
neighbourhood in 13/15 items (87 %), mean composite regret 0.005; all 10 expectation checks passed (no-op on
clean photo/portrait/screenshot/AI images; no re-sharpening of oversharpened, no re-smoothing of smoothed).
Misses: 16-bit P3 fixture with purely Gaussian noise (DRUNet beat SCUNet; regret 0.078) and a q60 AI portrait
where no-op tied FBCNN on the composite although PSNR/SSIM/ΔE favour FBCNN (37.4→40.1 dB).

### 5b. Held-out set — the unbiased estimate (`reports/eval/holdout_results.json`)
66 items never used for tuning (`ips/eval/holdout.py`): 16 real historical/degraded photos (CodeFormer, GFPGAN,
Solvay 1927, RealSRSet) · 7 genuine AI images (FLUX, FLUX.2, Chroma, Qwen-Image, HiDream, SD3.5, Hunyuan) + 7
metadata-stripped JPEG copies · 3 real UI screenshots + 3 JPEG copies · 30 synthetic edits on unseen sources with
different operators (cool/green casts, Gaussian noise, beauty filter, local-contrast HDR, r=1 oversharpening, q20).
Router frozen before evaluation at `7d2bc1a` (recorded as `6bf3cce` before commit re-authoring; core code identical).

| Measure (frozen router) | Tuning set | **Held-out** |
|---|---|---|
| Content-class accuracy | 95 % | **88 %** (historical 100 %, screenshots 100 %, synthetic 97 %, AI 50 %: the 7 metadata-stripped AI JPEGs read as photos, by design) |
| Expectation checks (no-op on clean / no re-sharpen / no re-smooth / AI untouched) | 10/10 | **22/24** (failed: two clean Set5 portraits received denoise/WB) |
| Auto ranked first of its neighbourhood (composite heuristic) | 87 % | **47 %** |
| Auto not worse than input — SSIM | – | **34/38** |
| Auto not worse than input — ΔE (colour) | – | **28/38** |

Findings (raw metrics; the composite heuristic proved misleading and is not used for conclusions):
* **Genuine AI images (7/7) and real screenshots (3/3) were correctly left untouched.**
* **JPEG cleanup generalises**: FBCNN improved SSIM and PSNR on 7/7 stripped AI JPEGs (+2.9 to +5.8 dB) and 3/3
  real screenshot JPEGs (SSIM +0.05 to +0.06), although the composite ranked it last on four AI items.
* **Automatic white balance does not generalise**: on unseen photos it *raised* colour error in 8/8 cases
  (e.g. clean portrait ΔE 0 → 4.8; cool-cast recall only 3/9). Gray-world-family estimates remove scene colour together
  with any cast. **Post-hoc fix** (commit after evaluation): Faithful Auto no longer white-balances photos; it reports the
  cast and offers WB in Manual (scans keep automatic correction). Recomputed from the measured "Auto − WB" variants:
  not-worse-ΔE 28 → **35/38**, SSIM unchanged. *Post-hoc: a new held-out set is needed for an unbiased confirmation.*
* Diagnosis precision/recall (actionable issues): JPEG P 0.60 / R 1.00 · noise P 0.40 / R 1.00 · colour cast
  P 0.38 / R 0.33 · oversharpening/HDR R 0.00 (6 missed — but Auto never sharpened them, so no harm) ·
  oversmoothing R 0.00 (2 missed; flag only).
* Held-out observation for review: on the colourised 1927 Solvay photograph Auto removed part of the warm
  colourisation (intent unknowable) — resolved by the photo-WB change above.

### 5c. Identity review signals on unseen faces
Two CodeFormer face sets (8 + 9 faces) never used for tuning, ×0.5 → ×2 by each method, beauty-filtered copies,
and JPEG q30 copies cleaned by FBCNN (legitimate — must not be flagged). Files: `reports/eval/identity_validation*.json`.

| Case | Set 1 (thresholds tuned on tuning corpus) | Set 2 |
|---|---|---|
| Lanczos / SwinIR ×2 | 0/6 flagged | 0/3 |
| Real-ESRGAN ×2plus | 6/6 flagged | 3/3 |
| Beauty filter | 6/6 flagged | 3/3 |
| Real-ESRGAN general v3, RealPLKSR | 0/6 (similarity 0.996 / 0.999 on large clear faces) | 0/3 |
| FBCNN on q30 (legitimate) | **5/6 flagged — false positives** | 2/3 |
| Faces detected | 6/8 | 3/9 |

Two defects followed: (a) after JPEG/denoise, legitimate cleanup (skin/non-face texture ratio 0.82–1.30) and beauty
smoothing (0.71–1.04) overlap → skin texture is now an informational note after artifact removal, not a flag;
(b) tight crops defeated the detector → padded re-detection (17/17 detected). Both face sets were used to diagnose
these, so they are no longer clean; the final check used the held-out Set5 face, never used for identity work:

| Held-out woman ×2 (final thresholds) | Flagged | Similarity vs source |
|---|---|---|
| Lanczos, SwinIR ×2 | no | 0.998 |
| Real-ESRGAN ×2plus / ×4plus / general | **yes / yes / yes** | 0.939 / 0.954 / 0.963 |
| RealPLKSR | no | 0.984 |

Visual inspection (contact sheets) agrees: all Real-ESRGAN outputs show plastic skin and darkened, redrawn eyes;
RealPLKSR raises contrast/saturation. **RealPLKSR evades every automatic signal in all sets**; it remains
Manual-only and any GAN/"balanced" output goes to the Review Queue by policy. Faces under ~140 px (held-out Set5
"head" at ×0.5) are not detected, so identity cannot be assessed on very small images.
Re-run of both face sets with the final thresholds: **pending** (running).

## 6. Defects found by testing and fixed (all re-tested)

| # | Found by | Defect | Fix |
|---|---|---|---|
| 1 | strace in privacy test | onnxruntime 1.31 sends 1DS telemetry to mobile.events.data.microsoft.com; its API switch does not stop it | `ORT_DISABLE_TELEMETRY=1` forced at import; verified 0 connects (Linux strace) and `disallowed: []` (Windows) |
| 2 | Windows CI | relative source folders stored with `\` → non-portable DB/reports | stored POSIX-style |
| 3 | identity signal check | GAN beautification not flagged (0/12) | recalibrated thresholds (§3), validated §5c |
| 4 | Auto benchmark | 8-px UI structure misread as JPEG blocking on lossless screenshots | exact-flat-region veto |
| 5 | denoise comparison | DRUNet σ under-set (0.8×) | 1.0× measured σ (better PSNR+SSIM on both fixtures) |
| 6 | screenshot review | progress/label/clipping issues in side panels | fixed |
| 7 | UI test crash | worker signal object freed while queued → segfault | workers kept alive until delivery |
| 8 | test design | colour-cast estimator confused orange cat with cast; fade vs natural low contrast | 3-estimator consensus + scene-hue dominance; fade needs scan evidence |
| 9 | bench | one-time model loads counted in warm timings | separated |
| 10 | unseen-face validation | legitimate JPEG cleanup flagged as skin smoothing (5/6) | texture → note after JPEG/denoise |
| 11 | unseen-face validation | tight face crops not detected (8/17 missed) | padded re-detection (17/17) |
| 12 | held-out evaluation | automatic WB raised colour error on 8/8 unseen photos | no automatic WB on photos (Faithful) |
| 13 | text evaluation | pixel/glyph metrics cannot detect garbled letters | OCR-based dev evaluation; policy regression test |
| 14 | held-out review | no-ground-truth sheets labelled the source "ground truth" | relabelled |

## 7. Performance (not the acceptance measurement)

CPU only; the 15 s contract is GPU-specific and **unmeasured** until `run.cmd -Bench` runs on the RTX 5070 Ti.

| Route (input) | Linux container (contended) | Windows CI runner (4 vCPU) |
|---|---|---|
| no-op, 0.5 MP | 5.5 s | 3.1 s |
| Lanczos ×2, 0.5 MP | 5.7 s | 3.2 s |
| FBCNN, 0.5 MP | 56 s | 41 s |
| Real-ESRGAN general ×4, 0.5 MP | 41 s | 22 s |
| SwinIR ×2, 0.7 MP output | 316 s | – |

## 8. Not verified (transparent list)
* Any RTX 5070 Ti behaviour: CUDA/fp16 numerics, speed, VRAM, Fast Profile, ComfyUI coexistence on the real
  install, NH Maker 2 coexistence. Procedure: `docs/WINDOWS_VALIDATION.md`.
* Interactive desktop use on a physical Windows display (HiDPI scaling factors, Explorer drag-and-drop,
  multi-monitor): UI was exercised offscreen on Linux and Windows CI only.
* User approval of any visual result. Real-world fidelity on the user's own photos.
* Generative Recovery routes (not implemented by design in v1).

## 9. Final regression run
**Pending** — final full regression run after the evaluations complete.

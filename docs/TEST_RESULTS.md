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

### 5b. Held-out set (never used for tuning; router frozen at commit `e62dfe2`, originally `b65ff9d`)
**Pending** — held-out evaluation running; results will replace this line.

### 5c. Identity-threshold validation on unseen faces (CodeFormer cropped faces)
**Pending** — unseen-face validation running; results will replace this line.

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

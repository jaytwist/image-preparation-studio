# Model & open-source matrix

Verified = file downloaded, SHA-256 recorded in `ips/backends/registry.py`, loaded by spandrel/onnxruntime
and executed in tests **on CPU in the build container**. "Win/5070 Ti" = pending until `run.ps1 -Bench`.
Measured quality numbers: see `reports/eval/MODEL_COMPARISON.md` (synthetic ground truth) and
TEST_RESULTS.md. Status: **Adopt** (Auto-eligible) · **Prototype/Baseline** (Manual only) · **Reject**.

## Restoration / upscaling models (direct backend)

| Model | Source repo (weights) | Weight licence | Character | Native control | Verified here | Status | Notes |
|---|---|---|---|---|---|---|---|
| FBCNN color | jiaxi-jiang/FBCNN (GitHub release v1.0) | Apache-2.0 | faithful | JPEG QF input (1−QF/100, from upstream test script) + blind QF prediction | ✅ load/run, QF monotonic, blind QF tracks true q | **Adopt** – JPEG | Auto sets qf from FBCNN's own prediction, cross-checked with quant tables |
| SwinIR-M CAR q10 / q30 | JingyunLiang/SwinIR v0.0 | Apache-2.0 | faithful | none (post-blend) | ✅ | Prototype | Heavy transformer; compared in harness |
| SCUNet real PSNR | cszn/KAIR v1.0 | Apache-2.0 | faithful | none (post-blend) | ✅ | **Adopt** – real noise | Blind, PSNR-trained (no GAN texture) |
| DRUNet color | cszn/KAIR v1.0 (DPIR) | Apache-2.0 | faithful | noise σ map | ✅ σ changes output | **Adopt** – scans/AI noise | Conservative σ = 0.7–0.8 × estimate |
| SwinIR-M color DN σ15 | JingyunLiang/SwinIR v0.0 | Apache-2.0 | faithful | none | ✅ | Prototype | Slow |
| SwinIR-M classical ×2 | JingyunLiang/SwinIR v0.0 | Apache-2.0 | faithful | none | ✅ | **Adopt** – faithful ×2 | Bicubic-degradation PSNR model; Auto runs it only after JPEG/noise cleanup |
| Real-ESRGAN ×2plus / ×4plus | xinntao/Real-ESRGAN | BSD-3-Clause | balanced (GAN) | none | ✅ | **Adopt – Balanced mode only** | Synthesises texture; outputs flagged for review |
| Real-ESRGAN general ×4 v3 (+wdn) | xinntao/Real-ESRGAN v0.2.5.0 | BSD-3-Clause | balanced | DNI weight (general↔wdn) | ✅ DNI changes output | Prototype | Very fast (SRVGG); candidate for a GPU fast path |
| RealPLKSR 4xNomos2 dysample | Phhofm/models | CC-BY-4.0 | balanced | none | ✅ | **Baseline (Manual)** | User reported beautification with *a* RealPLKSR checkpoint (exact file unknown). Benchmarked on equal terms; no outcome presupposed. See results |
| Restormer real-DN / defocus | swz30/Restormer | Restormer licence | faithful | none | ❌ not loadable by core spandrel | **Reject (v1)** | Needs spandrel_extra_arches (restrictive arch licences) |
| LaMa (big-lama) | advimman/lama (Sanster/models release) | Apache-2.0 | generative | mask | ✅ removal test | **Adopt – Quick Edit only** | Labelled generative fill |
| U²-Net human seg | xuebinqin/U-2-Net (rembg release) | Apache-2.0 | n/a | – | ✅ IoU 0.983–0.985 vs GT alpha | **Adopt** – person cutout | Guided-filter edge refinement + fringe decontamination |
| IS-Net general | xuebinqin/DIS (rembg release) | Apache-2.0 | n/a | – | ✅ loads/runs | Adopt – subject cutout | |
| InsightFace buffalo_l (SCRFD-10G + ArcFace R50) | deepinsight/insightface v0.7 | Code MIT; **weights non-commercial research only** | n/a | – | ✅ detects faces; drift signals | Adopt (optional) | Used only for review signals; falls back to OpenCV Haar (location only) |

## Generative / heavy candidates (not in v1 Auto)

| Candidate | Finding | Status |
|---|---|---|
| SeedVR2 3B/7B (ComfyUI node; official `comfyorg/comfyui_seedvr2`, community `numz`) | Weights on Hugging Face (Apache-2.0 per model card; node code MIT). Author reports ≥18 GB VRAM for 3B and ~1022 s for 20 images on an RTX 4090 at batch 1 — incompatible with the 15 s contract on 16 GB. HF unreachable from build container: **not executed**. | Prototype via user's ComfyUI later; never Fast Auto |
| SUPIR | Model declared **non-commercial use only**; SDXL-based, heavy, generative. Not executed. | Reject for default; possible labelled Generative Recovery |
| Qwen-Image 2.1 (image edit; Standard + Turbo) | Runs on the user's ComfyUI (≥ 0.37; graph validated against 0.44). Weights not available in the build container: **quality, identity and time unmeasured**. Generative by nature. | **Integrated** as explicit Generative Recovery mode (labelled, always reviewed); never Auto |
| FLUX 3 Klein | Not verifiable from the build container (HF blocked; no local copy). | Not integrated |
| CodeFormer / GFPGAN | Face *replacement* networks; CodeFormer weights S-Lab non-commercial. Conflicts with identity priority. | Reject |

## Applications / pipelines reviewed

| Project | What was reused / learned | Decision |
|---|---|---|
| **RAP – Restore Anything Pipeline** (eth-siplab/RAP, arXiv 2305.13093) | Idea: segmentation masks + *controllable* restoration per region (same author as FBCNN). The repo's README states code "will be released soon"; no usable code. | Design adopted (FBCNN native QF + regional masks); no code reused |
| **chaiNNer / spandrel** (MIT) | spandrel is the model loader; tiling approach informed by chaiNNer | **Adopt** (dependency) |
| **QualityScaler** (cloned, reviewed) | Tiled inference / VRAM-aware tile size ideas; app is a monolithic GUI | Reject as base; ideas reused |
| **rembg** | Model files and preprocessing constants (mean/std/size) | Models adopted; package not used (avoids runtime downloads) |
| **Real-ESRGAN** | DNI weight interpolation reproduced natively | Adopt weights |
| **LocalSR / local-upscale** | No project by that name could be identified; closest are Upscayl and Real-ESRGAN-ncnn-vulkan (Real-ESRGAN wrappers) | Nothing to reuse beyond Real-ESRGAN |
| **ComfyUI** | Localhost API only (`/system_stats`, `/object_info`, `/upload/image?type=temp`, `/prompt`, `/history`, `/view`) | Optional isolated adapter |

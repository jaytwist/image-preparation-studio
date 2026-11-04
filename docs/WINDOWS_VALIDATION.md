# Windows / RTX 5070 Ti validation procedure

The release is labelled **Windows GPU Validation Pending** until every step below has been run on the
target PC and its results are recorded in TEST_RESULTS.md. Each step is safe: it only writes inside
the project folder, the Studio app-data folder and the chosen output root.

| # | Step | Command | Pass criterion | Evidence file |
|---|---|---|---|---|
| 1 | Environment report (read-only) | `run.cmd -Inspect` | GPU = RTX 5070 Ti, `cuda_available: true`, capability (12, 0), torch arch list contains `sm_120`; ComfyUI installs/server listed | `reports\env_windows.json` |
| 2 | Models | `run.cmd -FetchModels` then `.venv\Scripts\python tools\fetch_models.py --verify` | all `ok` | console |
| 3 | Test suite | `run.cmd -Test` | all pass (strace privacy test is skipped on Windows) | console |
| 4 | Privacy | `run.cmd -PrivacyAudit` | `"disallowed": []` | console |
| 5 | Fast Profile | `run.cmd -Bench` (≈20–60 min) | table printed; routes with warm p95 ≤ 15 s are recorded as Fast | `reports\bench_NVIDIA_GeForce_RTX_5070_Ti.json`, `%LOCALAPPDATA%\ImagePreparationStudio\fast_profile.json` |
| 6 | Quality/identity on GPU | `.venv\Scripts\python -m ips.eval.harness compare` and `… auto` | numbers comparable to the CPU run (same models, fp16 on GPU) | `reports\eval\*.md`, `reports\eval\visual\*.png` |
| 7 | Real photos (yours) | Prepare tab: old phone photo, DSLR/RAW, portrait, screenshot, AI image, edited photo | visual review of faces/text in 200–400 % wipe; record verdicts | notes → TEST_RESULTS.md |
| 8 | Desktop UI | drag from Explorer, 100/125/150 % Windows scaling, multi-monitor, full screen, batch pause/cancel/resume, app restart mid-batch | aligned wipe at all scales; UI responsive during GPU jobs | screenshots |
| 9 | ComfyUI coexistence | enable in Settings while ComfyUI/NH Maker 2 is running; run a Manual ComfyUI upscale while another ComfyUI job is queued | Studio waits; foreign job unaffected; nothing in ComfyUI folders changes (compare `-Inspect` before/after) | console + env diffs |
| 10 | Known-regression check | Manual: run RealPLKSR (and your own RealPLKSR checkpoint via ComfyUI) on your portrait that showed beautification | identity signals / visual review documented either way | TEST_RESULTS.md |

If Remote Control is enabled for this folder, steps 1–6 and 9 can be executed by Claude directly.

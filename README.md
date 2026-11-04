# Image Preparation Studio

**One place to prepare every image for what comes next.** A local, offline-first Windows desktop app
that diagnoses each image from metadata *and* pixels, applies only the faithful fixes it needs (or
none), shows an aligned Before/After, lets you take over in Manual, does lightweight Quick Edits, runs
batches with per-image recipes, and exports organised, traceable outputs with audited metadata.

> **Release status: 0.9.0rc1 — Windows GPU Validation Pending.** Built and tested on CPU in a Linux
> container. The 15 s performance contract, RTX 5070 Ti VRAM use and the real Windows desktop have
> not been measured yet. See [docs/TEST_RESULTS.md](docs/TEST_RESULTS.md) for what was verified and how.

Priorities, in order: **identity/source fidelity → restoration quality → speed (≤ 15 s per supported job)**.

## Install & launch (Windows)

Requirements: Windows 10/11, Python 3.11–3.13 (python.org installer with the `py` launcher), an NVIDIA
driver that supports CUDA 12.8 (RTX 50-series needs it). Nothing outside this folder is changed.

```bat
git clone https://github.com/jaytwist/image-preparation-studio.git
cd image-preparation-studio
run.cmd -FetchModels
run.cmd
```

First run creates `.venv\` here, installs PyTorch (CUDA 12.8 build) and pinned dependencies.
`-FetchModels` downloads the model weights from their GitHub releases and verifies SHA-256 (≈1.4 GB;
user-directed, one time). After that the app runs fully offline.

| Command | What it does |
|---|---|
| `run.cmd` | launch the app |
| `run.cmd -Inspect` | read-only hardware / CUDA / ComfyUI / model inventory → `reports\env_windows.json` |
| `run.cmd -Bench` | measure the Fast Profile on this GPU → `reports\bench_*.json` (enables certified 15 s routes) |
| `run.cmd -Test` | run the automated test suite |
| `run.cmd -PrivacyAudit` | run a workload and list every network connection the process opened |
| `run.cmd -Cli prepare D:\Photos --profiles archive,share` | headless processing |

Linux/macOS: `./run.sh` (CPU or CUDA torch from PyPI).

## Using it

* **Prepare** — drop an image. Smart Auto runs immediately (toggle in Settings): diagnosis, content
  class, issues with evidence, *Why this recipe?*, the exact recipe, timing and review flags. Drag the
  wipe, switch Before/After/Side-by-side (`Space` toggles), zoom Fit/100/200/400 % (`F`,`1`,`2`,`4`),
  jump to a face crop, **Shift+drag** to preview the recipe on a region only. *Manual / Advanced* shows
  every operation with its real parameters; each slider says whether it drives a **native** model
  parameter or a labelled **post-blend**. Every run becomes a variant you can A/B against Auto.
* **Quick Edit** — Auto person/subject mask, brush ±, magic select, grow/shrink/feather/invert, mask
  undo; cutout to transparent PNG with edge feather/shift and fringe reduction; background replace;
  clothing/item **recolour** (keeps folds, shading and patterns); small-object removal (LaMa,
  labelled generative, or classical); rotate/flip/crop/straighten (with angle detection). Saves are
  always *Edited Variant* or *Transparent Cutout*.
* **Batch** — add files/folders (recursive), choose outputs; each image still gets its own adaptive
  recipe. Live table with thumbnail, state, time, diagnosis, recipe, exception; pause / continue /
  cancel / resume / retry; *Needs review* filter; double-click to override one image in Prepare.
  Interrupted batches resume after a restart; completed files are skipped only after their outputs'
  hashes verify. Broken files fail individually. Exact duplicates are skipped (never deleted);
  visually similar pairs are listed as suggestions.
* **Outputs** — search/filter by kind, approval, batch, model; sort; approve/reject; open folder;
  export selected (never overwrites).

Modes: **Faithful** (default; regression-trained models, no synthesised texture, fidelity guard on
faces) · **Balanced** (GAN upscalers that synthesise texture; always routed to Review) · Generative
Recovery (not in v1).

## Where results go

`Settings → Output root` (default `Pictures\Image Preparation Studio`):

```
<root>/Single/<date>/ or Batch/<name>/<original subfolders>/
    Restored/  <stem>__<src6>__master__<recipe8>.tif     Lossless Archive Master (source ICC, full metadata)
    AI_Ready/  …__aiready__….png                        sRGB, fit long edge, optional pad – never cropped
    Share/     …__share__….jpg                          sRGB JPEG, location metadata removed by default
    Edited/  Cutouts/                                   Quick Edit derivatives
    Recipes/   …__recipe__….json                        recipe, diagnostics, identity signals, metadata audit, timings
    Reports/   batch_<id>.json / .csv
```
Originals are opened read-only and never modified, renamed or moved; outputs are never overwritten
(`_v002`, …). Repeat exports reuse the existing master unless the recipe changed. App data (SQLite
library, logs, settings): `%LOCALAPPDATA%\ImagePreparationStudio`.

## ComfyUI (optional)
Off by default. When enabled (Settings), Studio talks only to `127.0.0.1`, uploads to ComfyUI's temp
folder, tracks only its own prompts, never interrupts or clears other work, waits while ComfyUI is busy
or VRAM is low, and lists your ComfyUI upscale models in Manual as *external (character unverified)*.
It never edits ComfyUI/NH Maker 2 files, workflows, models or settings.

## Privacy
No accounts, cloud calls, uploads or telemetry. `onnxruntime` 1.31 contains network telemetry that its
API switch does not stop; Studio forces `ORT_DISABLE_TELEMETRY=1` (verified: zero outbound connections
during a full workload). `run.cmd -PrivacyAudit` re-checks on your machine.

## Limitations (summary — details in docs/RELEASE_NOTES.md)
* No measurements on the RTX 5070 Ti yet; the Fast Profile is empty until `run.cmd -Bench` runs.
* Auto thresholds are calibrated on a synthetic corpus plus a few public real images; real-photo
  fidelity is **unverified** until you review results.
* No automatic deblur; clipped highlights are not reconstructed; generative recovery is not included.
* HEIC decodes at 8-bit; RAW is developed (LibRaw), never written back as RAW; MakerNotes are not carried.
* InsightFace face-model weights are licensed for non-commercial research use (optional component).

## Rollback / uninstall
Delete this folder (removes `.venv` and models). Optionally delete `%LOCALAPPDATA%\ImagePreparationStudio`
(library database, settings, logs) and the output root. No other files, registry keys, environment
variables, ComfyUI or NH Maker 2 files are touched.

## Documentation
[DECISIONS](docs/DECISIONS.md) · [MODEL_MATRIX](docs/MODEL_MATRIX.md) · [AUTO_RECIPES](docs/AUTO_RECIPES.md) ·
[TEST_RESULTS](docs/TEST_RESULTS.md) · [RELEASE_NOTES](docs/RELEASE_NOTES.md) · [WINDOWS_VALIDATION](docs/WINDOWS_VALIDATION.md) · **[RESTART](docs/RESTART.md)**

# Restart here

Paused on 2026-11-04 at the user's request. State is fully committed and pushed (`main`). Release status:
**0.9.0rc1 — Windows GPU Validation Pending.**

## Where things stand

| Area | State |
|---|---|
| App (Prepare / Auto / Manual / Quick Edit / Batch / Before-After / Outputs / metadata) | Implemented, tested on Linux and on real Windows (CPU) |
| Windows CI (GitHub Actions, `.github/workflows/windows.yml`) | Last code commit `c7df8f8`: **83 passed, 1 skipped**, app launch smoke OK, privacy audit `disallowed: []` |
| Linux regression | Full combined run on final code: **84 passed, 0 failed** (14 min, `pytest -n 2 --dist loadfile`) |
| Held-out validation | Done with frozen router (`reports/eval/holdout_results.json`); WB fix applied post-hoc |
| Identity review signals | Recalibrated + 2 defects fixed; validated on unseen faces (TEST_RESULTS §5c) |
| RTX 5070 Ti / ComfyUI / NH Maker 2 coexistence | **Not done** — no access (Remote Control never shared) |

## Next steps, in priority order

1. **Get GPU access**: on the RTX 5070 Ti PC, Claude desktop app → Settings → Claude Code → Remote Control → share
   the project folder. Then follow `docs/WINDOWS_VALIDATION.md` (steps 1–10). The critical ones:
   ```bat
   run.cmd -Inspect
   run.cmd -FetchModels
   run.cmd -Test
   run.cmd -Bench
   run.cmd -PrivacyAudit
   ```
   `-Bench` populates the Fast Profile; only routes with measured warm p95 ≤ 15 s become Fast Auto.
2. **Regression after any code change**: `python -m pytest -q -rs -n 2 --dist loadfile` (≈ 14 min on 4 CPU cores).
3. **Fresh held-out set** to confirm the post-hoc white-balance change without bias (the current held-out set was
   used to motivate it). Build with new sources/seed in `ips/eval/holdout.py`, then
   `python -m ips.eval.harness holdout`.
4. **GPU re-run of the quality harness** (fp16 numerics): `python -m ips.eval.harness compare`
   (CPU run was stopped before the last SwinIR-CAR candidate and the real-world stage; results so far are in
   `reports/eval/MODEL_COMPARISON.md`).
5. **User visual review** of real photos (yours) in the Prepare tab at 200–400 % — no user approval has been recorded.

## Open issues to work on next (from measured evidence)

* RealPLKSR beautification (colour/contrast shift) evades all automatic identity signals — kept Manual-only + Review
  by policy. A reference-based output-resolution check would be the next improvement.
* After JPEG/denoise, skin-texture loss cannot distinguish cleanup from smoothing → shown as a note only.
* Colour-cast detection recall is low (3/9 held-out); Auto no longer applies WB to photos.
* Oversharpening/HDR detection recall is 0/6 on held-out (harmless today: Auto never sharpens those).
* Faces below ~140 px are not detected → identity not assessed on very small images.
* DRUNet vs SCUNet choice for clean Gaussian noise (16-bit P3 fixture) — minor.

## How to resume in a new session

* Clone `https://github.com/jaytwist/image-preparation-studio`, read this file, `docs/TEST_RESULTS.md`
  and `docs/DECISIONS.md`.
* Build environment: `run.cmd -SetupOnly` (Windows) or `./run.sh` (Linux; container needs
  `apt-get install libegl1 libgl1 libxkbcommon0 libfontconfig1 libdbus-1-3`).
* Test data (not in git): `python -m ips.eval.fixtures --fetch` and `python -m ips.eval.holdout --fetch`;
  model weights: `python tools/fetch_models.py --all`.
* Commits from Claude must use `noreply@anthropic.com` (repository hook); pushes must not include personal images,
  weights or generated outputs (`.gitignore` covers fixtures, models, outputs, reports/ui, visual sheets).

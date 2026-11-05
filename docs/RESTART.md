# Restart here (local session on the RTX 5070 Ti PC)

Handed over on 2026-11-05 from the cloud build session to a **local Claude Code session on the user's Windows PC**
(`C:\Users\Jay\ImagePrepStudio`). Everything is committed and pushed (`main`, 0.9.0rc3). Release status:
**Windows GPU Validation Pending.**

## Why local
The cloud container had no GPU, no Windows desktop and no access to the user's ComfyUI (Remote Control folders were
"not available" to cloud sessions), so the user had to run test commands by hand. A local session can do all of that
itself. **Goal for the local session: the user only looks at results; Claude runs every command.**

## Ground rules (unchanged, from the handoff spec and the user)
* Never modify, update or restart ComfyUI or NH Maker 2; never write into their folders, workflows, models or
  settings. Talk to ComfyUI only via its localhost API (Studio already does: temp uploads, own prompt IDs only,
  no global interrupt/queue clear/free).
* Work only inside `C:\Users\Jay\ImagePrepStudio` (+ `%LOCALAPPDATA%\ImagePreparationStudio`, chosen output root).
* Never commit personal photos, metadata, generated outputs or weights (`.gitignore` covers them).
* Commits from Claude: author/committer `Claude <noreply@anthropic.com>`; push to `main`.
* Priorities: identity fidelity → image quality → speed (≤ 15 s per supported job, Fast Profile only).

## State
| Area | State |
|---|---|
| App (Prepare / Auto / Manual / Quick Edit / Batch with live preview / Outputs / metadata) | Implemented; Linux 92 passed; Windows CI (CPU) 92 passed, 11 skipped |
| Look levels (Enhanced default / Vivid / Natural) | Done (user asked for more visible results) |
| Generative Recovery: Qwen-Image 2.1 via the user's ComfyUI | Implemented (`ips/backends/comfy/qwen.py`); graphs accepted by a real ComfyUI 0.44 validator; **no real generation run yet** |
| RTX 5070 Ti speed, VRAM, coexistence with ComfyUI + NH Maker 2 | **Not measured** |

## Next steps, in order (do them yourself; report results to the user in plain Korean)
1. Sync: `git pull` (or clone if the folder came from a ZIP). Build/refresh env: `run.cmd -SetupOnly -Yes`.
2. Inspect: `run.cmd -Inspect` → GPU/CUDA/torch, ComfyUI URL/version, installed Qwen-Image 2.1 files.
3. **Qwen on the real GPU** (user's main request): with the user's ComfyUI running,
   `set IPS_COMFY_URL=http://127.0.0.1:8188` and `set IPS_QWEN_RUN=1`, then
   `.venv\Scripts\python -m pytest tests/test_comfy_real.py -k qwen -s` (and `IPS_QWEN_TURBO=1`).
   Then run Generative mode on 3–5 of the user's photos at strength 0.6 and 1.0; show Before/After and identity signals.
   Watch for: beautified/changed faces, invented detail, colour shifts, time per image, VRAM.
4. **Remove technical steps for the user** (agreed direction):
   * auto-enable ComfyUI when `127.0.0.1:8188` answers (no Settings checkbox needed);
   * Settings → "Test Qwen" button: one sample generation, shows time + Before/After;
   * plain-language errors ("ComfyUI is not running", "Qwen-Image 2.1 model not found").
5. `run.cmd -Bench` → Fast Profile on the RTX 5070 Ti (only routes with warm p95 ≤ 15 s become Fast Auto);
   `run.cmd -PrivacyAudit`; full `run.cmd -Test`.
6. Real Windows desktop checks (HiDPI, drag-and-drop from Explorer, Batch preview responsiveness) —
   `docs/WINDOWS_VALIDATION.md`.
7. Record measured results in `docs/TEST_RESULTS.md` (keep synthetic-ground-truth metrics separate from real-photo
   judgements), update `RELEASE_NOTES.md`; drop "GPU Validation Pending" only when steps 3–6 pass.

## Open issues (from measured evidence)
* Qwen: quality/identity/time unknown; the user's prompt asks for "significantly increase sharpness and fine
  detail", which invites invented detail — start at strength 0.5–0.7 and compare faces.
* RealPLKSR beautification evades all automatic identity signals → Manual-only + Review by policy.
* After JPEG/denoise, skin-texture loss is a note, not a flag. Faces < ~140 px are not identity-checked.
* Colour-cast detection recall is low (3/9 held-out); Auto never applies WB to photos.

## Not in git (regenerate locally)
Model weights (`run.cmd` asks, or `python tools/fetch_models.py --missing`), test fixtures
(`python -m ips.eval.fixtures --fetch`, `python -m ips.eval.holdout --fetch`), `.venv`, reports/ui screenshots.

Docs: `TEST_RESULTS.md` · `DECISIONS.md` · `AUTO_RECIPES.md` · `MODEL_MATRIX.md` · `WINDOWS_VALIDATION.md`.

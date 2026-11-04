# Architecture decisions and trade-offs

Status: v0.9.0rc1 · **Windows GPU Validation Pending** (see TEST_RESULTS.md).

## D1. Development and validation environment
The build session ran in a Linux cloud container (4 vCPU, 15 GB RAM, **no GPU, no Windows, no ComfyUI**).
Remote Control to the target PC was requested but no folder was shared during the session.
**Consequence:** everything was implemented and tested on CPU here; all RTX 5070 Ti timing, VRAM and
Windows-desktop claims are *pending* until `run.ps1 -Inspect`, `-Bench` and `-Test` run on the target.

## D2. Stack
| Choice | Why | Alternatives considered |
|---|---|---|
| Python 3.12 + **PySide6** desktop app | Spec preference; native Windows widgets, HiDPI per-monitor, no browser/server | Electron (heavier, needs a Python bridge anyway), web UI (localhost server = more attack surface) |
| **PyTorch + spandrel** for restoration models | spandrel (chaiNNer's loader, MIT) loads SwinIR/SCUNet/DRUNet/FBCNN/ESRGAN/RealPLKSR/LaMa from raw checkpoints without per-architecture code; same loader ComfyUI uses | Re-implementing architectures (error-prone); chaiNNer as a dependency (it is an app, not a library) |
| **onnxruntime (CPU EP)** for faces/matting | SCRFD/ArcFace/U²-Net/IS-Net are small (≈20–60 ms each on CPU); avoids a CUDA/cuDNN version clash with torch's bundled CUDA. GPU EP can be enabled later | onnxruntime-gpu (cuDNN coupling), insightface package (needs a compiler on Windows) |
| **pyexiv2** for EXIF/IPTC/XMP/ICC | exiv2 handles JPEG/TIFF/PNG/HEIF/RAW read + write in memory | piexif (EXIF only), exiftool (external binary) |
| **tifffile / OpenCV / Pillow / pillow-heif / rawpy** | 16-bit TIFF+ICC (tifffile), 16-bit PNG (OpenCV), HEIC (pillow-heif), RAW development (LibRaw) | libvips (extra native dependency; not needed at these sizes) |
| **SQLite (WAL)** | single-file, crash-safe, no server | JSON files (no transactions) |

## D3. Colour management
Processing happens **in the source colour space**; the source ICC is embedded in archive outputs so
wide-gamut images (Adobe RGB, P3, ProPhoto) are not clipped. Share/Web and AI-Ready exports are
converted to sRGB (LittleCMS, perceptual) and tagged. sRGB detection compares *primaries*, not profile
names (compact profiles are named "A98C"/"ROMM"). Display uses an sRGB rendering only on screen.
*Trade-off:* models were trained on sRGB data; for strongly wide-gamut sources that is a small
statistical mismatch, accepted to avoid irreversible gamut clipping in the master.

## D4. Metadata
Metadata is written into the encoded bytes **in memory** before an atomic, no-clobber commit, then
re-read and compared (audit). All pyexiv2 calls are serialised by one process-wide lock (pyexiv2 is not
thread-safe; concurrency test: 16 threads). Files > 2 GB skip exiv2 (memory) and use the Pillow
fallback reader. Dropped categories (each recorded with a reason): container/old-pixel tags,
thumbnails (IFD1), MakerNotes, DNG/RAW development tags, `crs:` Camera Raw settings (Lightroom would
re-apply them to processed pixels), old ICC/XMP packets. Orientation → 1, pixel dimensions updated,
DPI scaled with the upscale factor (print size preserved). GPS kept for archive; Share defaults to
**Remove location metadata** (GPS + IPTC/XMP location fields), user-switchable.
Provenance lives in a private XMP namespace (`ips:` source SHA-256, recipe hash/summary, output
kind, generative/edited flags); generative fills set IPTC DigitalSourceType
`compositeWithTrainedAlgorithmicMedia`.

## D5. Smart Auto = deterministic, evidence-driven router
* Content (photo/screenshot/AI/scan) and degradations are assessed **separately** from metadata + pixels.
  Metadata never acts alone: e.g. FBCNN's blind QF prediction verifies (or overrides) the JPEG
  quantisation-table quality — a re-saved q35 image in a q92 container is correctly treated as q≈35.
* Colour cast requires agreement of three illuminant estimators (gray-world on neutrals, gray-edge,
  white-patch) *and* low scene-hue dominance; otherwise it is reported as ambiguous and left alone.
* Fade correction needs scan evidence (scanner metadata) — natural low-contrast photos have identical
  global statistics (measured: clean cat photo DR 0.60 vs faded scan 0.59).
* Fewest operations, canonical order: geometry → JPEG → denoise → (deblur) → WB → exposure → tone →
  upscale → sharpen. No-op is a first-class outcome.
* Every decision has rationale text; confidence is the minimum of decision confidences.
* Manual corrections are stored; `python -m ips.eval.harness learn` turns consistent rejections
  (n ≥ 5, acceptance < 30 %) into per-class suppressions in `router_adjustments.json`. No online training.
* Heavy multi-candidate search happens **only** in the development harness, never per image.

## D6. Fidelity guard
In Faithful mode, after processing, faces detected on the source are compared (ArcFace similarity,
landmark drift, region SSIM, fine/mid texture ratios, skin ΔE). If any signal trips, the face region is
blended 60 % toward a conservative rendering (classical colour ops + Lanczos), the guard action is
shown, and the job goes to the Review Queue. Signals never *certify* identity.

## D7. Honest controls
Slider → native parameter where one exists (FBCNN QF, DRUNet σ, Real-ESRGAN-general DNI weight, all
classical parameters); otherwise the UI says "post-blend" explicitly. Face protection is labelled as a
regional post-blend.

## D8. Performance contract
`tools/bench.py` defines the Fast Profile per device: route × input-size bucket → warm p50/p95/max,
cold time, model load, peak VRAM. A route/size is Fast only if its measured warm p95 ≤ 15 s. On a
certified device, unmeasured or slower combinations require explicit **Quality Override**; nothing is
silently downscaled. On an uncertified device (this container) times are recorded and labelled
"not certified". No resolution or scale is assumed to qualify in advance.

## D9. ComfyUI isolation
Optional, off by default. Loopback URLs only; read-only discovery; uploads with `type=temp`, results
via `PreviewImage` (temp folder); tracks only its own `prompt_id`s; never global interrupt / queue
clear / `/free`; waits when other work is queued or free VRAM is below a threshold; Studio GPU work
and Studio ComfyUI jobs share one lock. User's ComfyUI upscale models appear in Manual as
"external (character unverified)" and are never Auto defaults.

## D10. Privacy
Finding during testing: **onnxruntime 1.31 opens a 1DS telemetry connection** to
`mobile.events.data.microsoft.com`; `disable_telemetry_events()` alone does not stop it. Studio forces
`ORT_DISABLE_TELEMETRY=1` (verified with strace: 0 outbound connects for a full workload) plus
`HF_HUB_OFFLINE`, `DO_NOT_TRACK`. A strace regression test and `tools/privacy_audit.py` (Windows,
psutil-based) guard this.

## D11. Scope decisions
* Generative Recovery (SeedVR2, SUPIR, Qwen/FLUX edits) is **not** in v1 Auto; weights are on Hugging
  Face (unreachable from the build container) and VRAM/latency on 16 GB is incompatible with the 15 s
  contract per published figures. They can be used through the user's own ComfyUI as labelled
  external routes later.
* Restormer rejected for v1 (needs spandrel_extra_arches with restrictive architecture licences).
* Face restoration networks (CodeFormer/GFPGAN) excluded: generative face replacement conflicts with
  priority 1.
* Object removal uses LaMa (labelled generative fill) or classical Telea; clothing recolour is
  deterministic LCh hue/chroma transfer that keeps lightness structure.

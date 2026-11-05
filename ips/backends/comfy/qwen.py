"""Generative Recovery through the user's local ComfyUI with Qwen-Image 2.1 (image edit).

The graph is the official Comfy-Org template `image_qwen_image_2_1_image_edit` / `..._turbo_image_edit`
(comfyui-workflow-templates-json 0.1.128), flattened to API format with the prompt enhancer removed so the
user's prompt reaches the model unchanged:

  UNETLoader -> QwenImage21Cache -> KSampler (standard)        | CFGGuider + SamplerCustomAdvanced (turbo)
  CLIPLoader(qwen3vl) + VAELoader -> TextEncodeQwenImage21(prompt, negative, resolution, image_1[, image_2])
  TextEncodeQwenImage21.latent (encoded source) -> sampler latent  ->  VAEDecode -> PreviewImage

Fidelity control (verified against comfy_extras/nodes_qwen.py, ComfyUI 0.44): TextEncodeQwenImage21's latent
output is an *empty* latent sized like image_1; the source reaches the model only as reference latents.
* strength = 1.0 -> exactly the official template (empty latent, full schedule): the edit is driven by the
  reference image and the prompt.
* strength < 1.0 -> img2img on top of that: the source (pre-resized by Studio to the node's own ×32 grid, so the
  node does not resample it again) is VAE-encoded and used as the start latent; KSampler.denoise = strength
  (turbo: the official sigma schedule truncated at strength). Lower keeps more of the original pixels.
With a face reference, images are tagged <image1>/<image2> as the template's prompt rules require for N >= 2.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

# Official Turbo schedule from the template (ManualSigmas)
TURBO_SIGMAS = [1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0.0]

DEFAULT_PROMPT = (
    "Enhance this old low-resolution phone photo: upscale to high resolution while preserving the original "
    "composition and framing. Significantly increase sharpness and fine detail, correct white balance to neutral "
    "and accurate colors (remove any yellow, blue, or green casts), and brighten naturally with improved clarity "
    "and contrast. Reduce noise, grain, and compression artifacts without over-smoothing. Make it photorealistic "
    "with natural skin texture, realistic lighting, and authentic detail. Strictly do not beautify, do not smooth "
    "skin, do not alter facial features, do not add makeup or glamour effects, and do not change the person's "
    "identity, age appearance, pose, clothing, or background. Keep all original imperfections and characteristics "
    "intact.")
FACE_PREFIX = "<image1> is the photo to enhance; the result must have exactly the composition and framing of <image1>. "
FACE_ANCHOR = (" <image2> is a close-up crop of the same person's face from <image1>, for reference only — do not add it "
               "to the picture. The face in the result must be identical to <image2>: same eye shape, nose, mouth, face "
               "proportions, skin tone, wrinkles, moles and expression.")
DEFAULT_NEGATIVE = ("beautified, airbrushed, plastic skin, smoothed skin, makeup, glamour, different person, changed "
                    "face, younger, older, cartoon, painting, illustration, oversharpened, halos, extra objects")

PRESETS = {
    "old_phone": DEFAULT_PROMPT,
    "photo": DEFAULT_PROMPT.replace("old low-resolution phone photo", "photo"),
    "ai_image": ("Enhance this image: increase sharpness, fine detail and clarity, reduce noise and compression "
                 "artifacts without over-smoothing, keep the original style, colors, lighting, composition and framing. "
                 "Do not change the character's face, identity, expression, pose, outfit or background."),
}


@dataclass
class QwenModels:
    unet: str
    clip: str
    vae: str
    turbo: bool


def _opts(info: dict, node: str, field: str) -> list[str]:
    from .client import _combo_options
    try:
        return _combo_options(info[node]["input"]["required"][field])
    except (KeyError, TypeError):
        return []


def detect(client, turbo: bool = False) -> tuple[QwenModels | None, str]:
    """Read-only: find Qwen-Image 2.1 nodes and model files on the user's ComfyUI."""
    need = ("TextEncodeQwenImage21", "QwenImage21Cache", "UNETLoader", "CLIPLoader", "VAELoader")
    info: dict[str, Any] = {}
    for n in need:
        try:
            info.update(client._get_json(f"/object_info/{n}"))
        except Exception:
            pass
    missing = [n for n in need if n not in info]
    if missing:
        return None, f"ComfyUI lacks nodes {missing} (Qwen-Image 2.1 needs ComfyUI >= 0.37)"
    unets = [u for u in _opts(info, "UNETLoader", "unet_name") if "qwen_image_2.1" in u.lower().replace("-", "_")]
    pick = [u for u in unets if ("turbo" in u.lower()) == turbo]
    pick.sort(key=lambda u: (0 if "int8" in u or "fp8" in u else 1, u))  # prefer 16 GB-friendly quantised files
    clips = [c for c in _opts(info, "CLIPLoader", "clip_name") if "qwen3vl" in c.lower() or "qwen3_vl" in c.lower()]
    clips.sort(key=lambda c: (0 if "int8" in c or "fp8" in c else 1, c))
    vaes = [v for v in _opts(info, "VAELoader", "vae_name") if "qwen_image_2.1" in v.lower() or "qwen_image" in v.lower()]
    vaes.sort(key=lambda v: (0 if "2.1" in v else 1, v))
    if not pick:
        return None, f"no Qwen-Image 2.1 {'Turbo ' if turbo else ''}diffusion model found in ComfyUI models/diffusion_models"
    if not clips or not vaes:
        return None, "Qwen-Image 2.1 text encoder (qwen3vl) or VAE not found in ComfyUI models"
    return QwenModels(pick[0], clips[0], vaes[0], turbo), "ok"


def resolution_for(width: int, height: int, scale: float) -> int:
    """Pixel-budget side (resolution) for an output of width*height*scale², capped at the model's 2K, ×32."""
    r = math.sqrt(width * height) * scale
    return int(max(512, min(2048, round(r / 32) * 32)))


def ref_size(width: int, height: int, resolution: int) -> tuple[int, int]:
    """The size TextEncodeQwenImage21 resizes image_1 to (and the output size), as computed by the node."""
    if resolution <= 0:
        return max(32, round(width / 32) * 32), max(32, round(height / 32) * 32)
    ratio = width / height
    w = round(math.sqrt(resolution * resolution * ratio) / 32) * 32
    h = round(math.sqrt(resolution * resolution / ratio) / 32) * 32
    return max(32, w), max(32, h)


def build(models: QwenModels, image: str, prompt: str, *, face: str | None = None,
          negative: str = DEFAULT_NEGATIVE, resolution: int = 1024, strength: float = 1.0, steps: int = 25,
          cfg: float = 1.0, seed: int | None = None) -> dict[str, Any]:
    seed = random.randint(0, 2**48) if seed is None else int(seed)
    strength = float(max(0.05, min(1.0, strength)))
    full_prompt = (FACE_PREFIX + prompt + FACE_ANCHOR) if face else prompt
    img2img = strength < 0.999
    start = ["15", 0] if img2img else ["7", 2]
    g: dict[str, Any] = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": models.unet, "weight_dtype": "default"}},
        "2": {"class_type": "QwenImage21Cache", "inputs": {"model": ["1", 0], "device": "auto", "dtype": "default"}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": models.clip, "type": "qwen_image", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": models.vae}},
        "5": {"class_type": "LoadImage", "inputs": {"image": image}},
        "7": {"class_type": "TextEncodeQwenImage21",
              "inputs": {"clip": ["3", 0], "vae": ["4", 0], "prompt": full_prompt, "negative_prompt": negative,
                         "resolution": int(resolution), "images.image_1": ["5", 0]}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["4", 0]}},
        "10": {"class_type": "PreviewImage", "inputs": {"images": ["9", 0]}},
    }
    if img2img:  # caller uploads the source already at ref_size(), so this latent matches the node's own latent
        g["15"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["5", 0], "vae": ["4", 0]}}
    if face:
        g["6"] = {"class_type": "LoadImage", "inputs": {"image": face}}
        g["7"]["inputs"]["images.image_2"] = ["6", 0]
    if models.turbo:
        sig = [strength] + [s for s in TURBO_SIGMAS if s < strength - 1e-6]
        g.update({
            "11": {"class_type": "ManualSigmas", "inputs": {"sigmas": ", ".join(f"{s:g}" for s in sig)}},
            "12": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
            "13": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
            "14": {"class_type": "CFGGuider", "inputs": {"model": ["2", 0], "positive": ["7", 0], "negative": ["7", 1],
                                                       "cfg": float(cfg)}},
            "8": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["13", 0], "guider": ["14", 0],
                                                                   "sampler": ["12", 0], "sigmas": ["11", 0],
                                                                   "latent_image": start}},
        })
    else:
        g["8"] = {"class_type": "KSampler", "inputs": {"model": ["2", 0], "positive": ["7", 0], "negative": ["7", 1],
                                                     "latent_image": start, "seed": seed, "steps": int(steps),
                                                     "cfg": float(cfg), "sampler_name": "euler", "scheduler": "simple",
                                                     "denoise": strength}}
    return g

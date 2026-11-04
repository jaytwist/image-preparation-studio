"""Model registry: what exists, where it came from, what it may be used for.

``status``: adopt (eligible for Auto), prototype (Manual only, labelled), baseline (kept for fair
comparison; Manual only unless benchmarks justify promotion), reject (not loadable from UI).
``character``: faithful (regression/PSNR-trained, does not invent texture), balanced (GAN-trained,
synthesises plausible texture), generative (diffusion/inpainting; output contains invented content).
Checksums were computed on the downloaded files; ``tools/fetch_models.py`` verifies them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..paths import models_dir


@dataclass(frozen=True)
class NativeParam:
    name: str
    label: str
    minimum: float
    maximum: float
    default: float
    step: float = 1.0
    help: str = ""


@dataclass(frozen=True)
class ModelInfo:
    id: str
    name: str
    task: str                      # jpeg | denoise | deblur | upscale | inpaint | matting | face
    file: str
    sha256: str
    url: str
    license: str
    source_repo: str
    scale: int = 1
    status: str = "adopt"
    character: str = "faithful"
    native_params: tuple[NativeParam, ...] = ()
    tile: int = 512
    notes: str = ""
    extra_files: tuple[tuple[str, str, str], ...] = ()  # (file, sha256, url)

    @property
    def path(self) -> Path:
        return models_dir() / self.file

    @property
    def installed(self) -> bool:
        return self.path.is_file()

    def native_defaults(self) -> dict[str, Any]:
        return {p.name: p.default for p in self.native_params}


_GH = "https://github.com"
MODELS: dict[str, ModelInfo] = {m.id: m for m in [
    ModelInfo("fbcnn_color", "FBCNN (color) – JPEG artifact removal", "jpeg", "fbcnn_color.pth",
              "8b0e4ef23d59cf7ac934a342cb31a17619e4fa4a0b3374a9d78c5174312387e8",
              f"{_GH}/jiaxi-jiang/FBCNN/releases/download/v1.0/fbcnn_color.pth", "Apache-2.0",
              "jiaxi-jiang/FBCNN",
              native_params=(NativeParam("qf", "JPEG quality factor (lower = stronger cleanup)", 5, 95, 50, 1,
                                         "Native FBCNN control: the quality-factor embedding fed to the network."
                                         " Auto sets it from the file's quantisation tables."),),
              notes="Flexible blind JPEG artifact removal; QF is a genuine native input."),
    ModelInfo("swinir_car_q10", "SwinIR-M color JPEG CAR q10", "jpeg", "006_colorCAR_DFWB_s126w7_SwinIR-M_jpeg10.pth",
              "0005b707e0e6f75b4d13c7447e2f184858ddbdcec95aa7eb8c07e8afa1d26bd9",
              f"{_GH}/JingyunLiang/SwinIR/releases/download/v0.0/006_colorCAR_DFWB_s126w7_SwinIR-M_jpeg10.pth",
              "Apache-2.0", "JingyunLiang/SwinIR", status="prototype", tile=256,
              notes="Fixed-strength model trained for q10; no native knob (strength = post-blend)."),
    ModelInfo("swinir_car_q30", "SwinIR-M color JPEG CAR q30", "jpeg", "006_colorCAR_DFWB_s126w7_SwinIR-M_jpeg30.pth",
              "3b906265f869a90793e91f4532f7b3aa4de8146195acd156464bbe9b473a482d",
              f"{_GH}/JingyunLiang/SwinIR/releases/download/v0.0/006_colorCAR_DFWB_s126w7_SwinIR-M_jpeg30.pth",
              "Apache-2.0", "JingyunLiang/SwinIR", status="prototype", tile=256,
              notes="Fixed-strength model trained for q30; no native knob (strength = post-blend)."),
    ModelInfo("scunet_real_psnr", "SCUNet real-noise (PSNR)", "denoise", "scunet_color_real_psnr.pth",
              "fa78899ba2caec9d235a900e91d96c689da71c42029230c2028b00f09f809c2e",
              f"{_GH}/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth", "Apache-2.0", "cszn/SCUNet",
              notes="Blind real-noise removal, PSNR-trained (faithful). No native strength: UI strength is a"
                    " labelled post-blend with the input."),
    ModelInfo("drunet_color", "DRUNet (color) – sigma-conditioned denoiser", "denoise", "drunet_color.pth",
              "479abe3c5327dfd10ff54a80ec7d4098ca80752a5c9492cdff31cee430bec4b4",
              f"{_GH}/cszn/KAIR/releases/download/v1.0/drunet_color.pth", "Apache-2.0", "cszn/DPIR",
              native_params=(NativeParam("sigma", "Noise level σ (0–255 scale)", 0, 50, 10, 0.5,
                                         "Native DRUNet input: constant noise-level map concatenated to the image."),),
              notes="Gaussian-trained; good when noise is close to AWGN. σ is a genuine native input."),
    ModelInfo("swinir_dn15", "SwinIR-M color denoise σ15", "denoise", "005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth",
              "917cd972f7ba80786871add249ad43e4477ce2db59b4ad63e2fa446f7221d013",
              f"{_GH}/JingyunLiang/SwinIR/releases/download/v0.0/005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth",
              "Apache-2.0", "JingyunLiang/SwinIR", status="prototype", tile=256,
              notes="Fixed σ15 Gaussian denoiser; heavy transformer, slow."),
    ModelInfo("restormer_real_dn", "Restormer real denoising", "denoise", "restormer_real_denoising.pth",
              "4cae18bed8a291b9a5deeeab48756bbe6f61fcf7f31a4cca0969b24608601387",
              f"{_GH}/swz30/Restormer/releases/download/v1.0/real_denoising.pth", "Restormer license (see repo)",
              "swz30/Restormer", status="reject", tile=384,
              notes="Rejected for v1: not supported by core spandrel; needs optional spandrel_extra_arches whose "
                    "bundled architectures have restrictive licenses. Weights kept for future evaluation."),
    ModelInfo("restormer_defocus", "Restormer single-image defocus deblur", "deblur",
              "restormer_defocus_deblurring.pth",
              "7dce451f33f8f5e0faf7c4e3996e5dcc1bd425ecd1ada99b0f9750e490fd4c9e",
              f"{_GH}/swz30/Restormer/releases/download/v1.0/single_image_defocus_deblurring.pth",
              "Restormer license (see repo)", "swz30/Restormer", status="reject", tile=384,
              notes="Rejected for v1 (same loader/licence reason as Restormer denoise). No faithful automatic "
                    "deblur is offered; classical halo-free sharpening remains available."),
    ModelInfo("swinir_x2", "SwinIR-M classical ×2 (PSNR)", "upscale", "001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth",
              "2032ebf8f401dd3ce2fae5f3852117cb72101ec6ed8358faa64c2a3fa09ed4ac",
              f"{_GH}/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth",
              "Apache-2.0", "JingyunLiang/SwinIR", scale=2, tile=256,
              notes="Bicubic-degradation PSNR model: conservative, no texture synthesis. Expects clean input."),
    ModelInfo("realesrgan_x2plus", "Real-ESRGAN ×2plus (GAN)", "upscale", "RealESRGAN_x2plus.pth",
              "49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb",
              f"{_GH}/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth", "BSD-3-Clause",
              "xinntao/Real-ESRGAN", scale=2, character="balanced",
              notes="GAN: removes degradations and synthesises texture. Balanced mode only."),
    ModelInfo("realesrgan_x4plus", "Real-ESRGAN ×4plus (GAN)", "upscale", "RealESRGAN_x4plus.pth",
              "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1",
              f"{_GH}/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth", "BSD-3-Clause",
              "xinntao/Real-ESRGAN", scale=4, character="balanced"),
    ModelInfo("realesr_general_x4v3", "Real-ESRGAN general ×4 v3 (compact, DNI denoise)", "upscale",
              "realesr-general-x4v3.pth", "8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292",
              f"{_GH}/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth", "BSD-3-Clause",
              "xinntao/Real-ESRGAN", scale=4, character="balanced",
              native_params=(NativeParam("denoise_strength", "Denoise strength (weight interpolation)", 0, 1, 0.5,
                                         0.05, "Native: network-interpolation (DNI) between the general and the "
                                               "weak-denoise (wdn) checkpoints, as in Real-ESRGAN's own CLI."),),
              extra_files=(("realesr-general-wdn-x4v3.pth",
                            "1641f8c4464b9f097c9fdda5589273713f67cf59f3d909e0bd688f0cee269dca",
                            f"{_GH}/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth"),),
              notes="Small/fast SRVGG model."),
    ModelInfo("realplksr_nomos2_x4", "RealPLKSR 4xNomos2 (dysample)", "upscale", "4xNomos2_realplksr_dysample.pth",
              "96c8476ed2c42107628d10562b32ae89e7be3c2f7a53d29504823e9bafe8121d",
              f"{_GH}/Phhofm/models/releases/download/4xNomos2_realplksr_dysample/4xNomos2_realplksr_dysample.pth",
              "CC-BY-4.0", "Phhofm/models", scale=4, status="baseline", character="balanced",
              notes="User previously observed beautification with a RealPLKSR checkpoint (exact file unknown). "
                    "Kept as a fairly-benchmarked candidate; Auto eligibility decided by measurements."),
    ModelInfo("lama", "LaMa (big-lama) inpainting", "inpaint", "big-lama.pt",
              "344c77bbcb158f17dd143070d1e789f38a66c04202311ae3a258ef66667a9ea9",
              f"{_GH}/Sanster/models/releases/download/add_big_lama/big-lama.pt", "Apache-2.0",
              "advimman/lama", character="generative", tile=512,
              notes="Generative fill for object removal (Quick Edit only, labelled edited derivative)."),
    ModelInfo("u2net_human_seg", "U²-Net human segmentation", "matting", "u2net_human_seg.onnx",
              "01eb6a29a5c4d8edb30b56adad9bb3a2a0535338e480724a213e0acfd2d1c73c",
              f"{_GH}/danielgatis/rembg/releases/download/v0.0.0/u2net_human_seg.onnx", "Apache-2.0",
              "xuebinqin/U-2-Net", notes="Person mask, 320 px inference; refined by guided filter."),
    ModelInfo("isnet_general", "IS-Net general-use (DIS)", "matting", "isnet-general-use.onnx",
              "60920e99c45464f2ba57bee2ad08c919a52bbf852739e96947fbb4358c0d964a",
              f"{_GH}/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx", "Apache-2.0",
              "xuebinqin/DIS", notes="Generic salient object mask, 1024 px inference."),
    ModelInfo("buffalo_l", "InsightFace buffalo_l (SCRFD + ArcFace)", "face", "buffalo_l/det_10g.onnx",
              "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
              f"{_GH}/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
              "Code MIT; pretrained weights NON-COMMERCIAL research use only", "deepinsight/insightface",
              status="adopt",
              notes="Used only for face location and identity-drift review signals. Optional: falls back to "
                    "OpenCV Haar (location only) if absent.",
              extra_files=(("buffalo_l/w600k_r50.onnx",
                            "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43", ""),)),
]}


def get(model_id: str) -> ModelInfo:
    return MODELS[model_id]


def by_task(task: str, include_status: tuple[str, ...] = ("adopt", "prototype", "baseline")) -> list[ModelInfo]:
    return [m for m in MODELS.values() if m.task == task and m.status in include_status]

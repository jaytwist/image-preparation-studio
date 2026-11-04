"""Read-only environment report: OS, CPU/RAM, GPU/driver, PyTorch/CUDA, ComfyUI install + server,
models present, custom nodes, NH Maker 2 location hints.

Nothing is written anywhere except the report file you ask for (--out). No ComfyUI/NH Maker 2 file is
modified, executed or imported.

    python tools/inspect_env.py --out reports/env.json
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

MODEL_SUBDIRS = ("upscale_models", "checkpoints", "diffusion_models", "unet", "vae", "loras", "clip",
                 "text_encoders", "SEEDVR2", "seedvr2", "facerestore_models", "facedetection", "inpaint",
                 "sams", "ultralytics", "rembg", "BiRefNet", "supir", "controlnet")


def _run(cmd: list[str], timeout: int = 15) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception:
        return None


def gpu_info() -> dict[str, Any]:
    out: dict[str, Any] = {}
    smi = shutil.which("nvidia-smi")
    if smi:
        q = _run([smi, "--query-gpu=name,driver_version,memory.total,memory.used,compute_cap,pcie.link.gen.current",
                  "--format=csv,noheader"])
        out["nvidia_smi"] = q
        out["nvidia_smi_processes"] = _run([smi, "--query-compute-apps=pid,process_name,used_memory",
                                            "--format=csv,noheader"])
    else:
        out["nvidia_smi"] = None
    try:
        import torch
        out["torch"] = torch.__version__
        out["torch_cuda_build"] = torch.version.cuda
        out["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            i = torch.cuda.current_device()
            free, total = torch.cuda.mem_get_info(i)
            out.update(device=torch.cuda.get_device_name(i), capability=torch.cuda.get_device_capability(i),
                       vram_total_mb=total // 2**20, vram_free_mb=free // 2**20,
                       arch_list=torch.cuda.get_arch_list(), cudnn=torch.backends.cudnn.version())
    except Exception as e:
        out["torch_error"] = f"{type(e).__name__}: {e}"
    try:
        import onnxruntime as ort
        out["onnxruntime"] = ort.__version__
        out["onnx_providers"] = ort.get_available_providers()
    except Exception as e:
        out["onnxruntime_error"] = str(e)
    return out


def comfy_candidates() -> list[Path]:
    c: list[Path] = []
    env = os.environ.get("COMFYUI_PATH")
    if env:
        c.append(Path(env))
    home = Path.home()
    roots = [home, home / "Desktop", home / "Documents", home / "Downloads", Path("C:/"), Path("D:/"), Path("E:/"),
             Path("F:/"), home / "AI", Path("C:/AI"), Path("D:/AI")]
    for r in roots:
        try:
            if not r.is_dir():
                continue
            for p in r.iterdir():
                n = p.name.lower()
                if p.is_dir() and "comfy" in n:
                    c.append(p)
                    for sub in ("ComfyUI", "comfyui"):
                        if (p / sub).is_dir():
                            c.append(p / sub)
        except (PermissionError, OSError):
            pass
    appdata = os.environ.get("APPDATA")
    if appdata:
        cfg = Path(appdata) / "ComfyUI" / "config.json"
        if cfg.is_file():
            try:
                base = json.loads(cfg.read_text("utf-8")).get("basePath")
                if base:
                    c.append(Path(base))
            except Exception:
                pass
    seen, out = set(), []
    for p in c:
        try:
            rp = p.resolve()
        except OSError:
            continue
        if rp not in seen and ((rp / "main.py").is_file() or (rp / "models").is_dir()):
            seen.add(rp)
            out.append(rp)
    return out


def comfy_install_report(root: Path) -> dict[str, Any]:
    rep: dict[str, Any] = {"root": str(root)}
    models = root / "models"
    inv: dict[str, list[dict[str, Any]]] = {}
    if models.is_dir():
        for sub in sorted(models.iterdir()):
            if not sub.is_dir():
                continue
            files = []
            try:
                for f in sorted(sub.rglob("*")):
                    if f.is_file() and f.suffix.lower() in (".pth", ".pt", ".safetensors", ".ckpt", ".onnx", ".bin",
                                                             ".gguf", ".sft"):
                        files.append({"name": str(f.relative_to(sub)), "mb": round(f.stat().st_size / 2**20, 1)})
                        if len(files) > 300:
                            break
            except (PermissionError, OSError):
                pass
            if files:
                inv[sub.name] = files
    rep["models"] = inv
    cn = root / "custom_nodes"
    rep["custom_nodes"] = sorted(p.name for p in cn.iterdir() if p.is_dir()) if cn.is_dir() else []
    for f in ("extra_model_paths.yaml",):
        p = root / f
        if p.is_file():
            rep[f] = p.read_text("utf-8", errors="replace")[:4000]
    for vf in ("comfyui_version.py", "pyproject.toml"):
        p = root / vf
        if p.is_file():
            rep[vf] = p.read_text("utf-8", errors="replace")[:600]
    py = root.parent / "python_embeded" / "python.exe"
    rep["embedded_python"] = str(py) if py.is_file() else None
    return rep


def comfy_server(url: str = "http://127.0.0.1:8188") -> dict[str, Any]:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url + "/system_stats", timeout=3) as r:
            stats = json.loads(r.read())
        with opener.open(url + "/queue", timeout=3) as r:
            q = json.loads(r.read())
        return {"reachable": True, "system_stats": stats, "queue_running": len(q.get("queue_running", [])),
                "queue_pending": len(q.get("queue_pending", []))}
    except Exception as e:
        return {"reachable": False, "error": str(e)}


def nh_maker_hints() -> list[str]:
    hits = []
    home = Path.home()
    for r in (home, home / "Desktop", home / "Documents", Path("C:/"), Path("D:/"), Path("E:/")):
        try:
            for p in r.iterdir():
                n = p.name.lower().replace(" ", "").replace("_", "").replace("-", "")
                if p.is_dir() and ("nhmaker" in n or n.startswith("nh") and "maker" in n):
                    hits.append(str(p))
        except (PermissionError, OSError):
            pass
    return hits


def collect() -> dict[str, Any]:
    import psutil
    vm = psutil.virtual_memory()
    rep = {
        "os": platform.platform(), "python": sys.version, "executable": sys.executable,
        "cpu": platform.processor(), "cpu_count": os.cpu_count(), "ram_gb": round(vm.total / 2**30, 1),
        "disk_free_gb": round(shutil.disk_usage(str(Path.home())).free / 2**30, 1),
        "gpu": gpu_info(), "comfyui_installs": [comfy_install_report(p) for p in comfy_candidates()],
        "comfyui_server": comfy_server(), "nh_maker_hints": nh_maker_hints(),
    }
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from ips.backends.registry import MODELS
        rep["studio_models"] = {m.id: {"installed": m.installed, "status": m.status, "license": m.license}
                                for m in MODELS.values()}
    except Exception as e:
        rep["studio_models_error"] = str(e)
    return rep


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    a = ap.parse_args()
    r = collect()
    s = json.dumps(r, indent=2, default=str)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(s, "utf-8")
    print(s)

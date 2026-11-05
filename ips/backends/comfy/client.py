"""Minimal, isolation-first client for a *local* ComfyUI server.

Isolation rules (enforced here, tested in tests/test_comfy_client.py with a fake server):
* Only loopback URLs are accepted (127.0.0.1 / localhost / ::1). Nothing leaves the machine.
* Read-only discovery: /system_stats, /object_info, /queue, /history. No config, model or workflow
  files of the existing installation are read or written by this app.
* Inputs are uploaded with ``type=temp`` and results come from PreviewImage (temp folder), which
  ComfyUI cleans itself – the user's input/output folders are not touched.
* Studio jobs carry a unique ``client_id`` and are tracked by their own ``prompt_id``. The client
  never calls a global interrupt, never clears the queue, never calls /free. Cancelling deletes
  *our* pending prompt; if our prompt is the one running, an interrupt targeted at that prompt_id
  is sent (servers that do not support targeted interrupts are left to finish).
* VRAM-aware: if other work is queued/running or free VRAM is below a threshold, Studio waits.
"""
from __future__ import annotations

import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

LOOPBACK = {"127.0.0.1", "localhost", "::1", "[::1]"}
WORKFLOW_DIR = Path(__file__).parent / "workflows"


class ComfyError(Exception):
    pass


class ComfyUnavailable(ComfyError):
    pass


class ComfyBusy(ComfyError):
    pass


@dataclass
class ComfyStatus:
    available: bool
    url: str
    version: str | None = None
    device: str | None = None
    vram_total_mb: int | None = None
    vram_free_mb: int | None = None
    queue_running: int = 0
    queue_pending: int = 0
    error: str | None = None


class ComfyClient:
    def __init__(self, url: str = "http://127.0.0.1:8188", timeout: float = 5.0):
        u = urllib.parse.urlparse(url)
        if u.hostname not in {h.strip("[]") for h in LOOPBACK}:
            raise ValueError(f"Only local ComfyUI servers are allowed (got host {u.hostname!r}).")
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.client_id = f"ips-{uuid.uuid4().hex[:12]}"
        self.our_prompts: set[str] = set()
        # bypass any system/corporate proxy for loopback
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    # ------------------------------------------------------------------ http
    def _req(self, method: str, path: str, body: bytes | None = None, ctype: str | None = None,
             timeout: float | None = None) -> bytes:
        req = urllib.request.Request(self.url + path, data=body, method=method)
        if ctype:
            req.add_header("Content-Type", ctype)
        try:
            with self._opener.open(req, timeout=timeout or self.timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            raise ComfyError(f"{method} {path}: HTTP {e.code} {e.read()[:300]!r}") from e
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
            raise ComfyUnavailable(f"ComfyUI not reachable at {self.url}: {e}") from e

    def _get_json(self, path: str) -> Any:
        return json.loads(self._req("GET", path))

    # ------------------------------------------------------------------ discovery (read-only)
    def status(self) -> ComfyStatus:
        try:
            s = self._get_json("/system_stats")
        except ComfyError as e:
            return ComfyStatus(False, self.url, error=str(e))
        dev = (s.get("devices") or [{}])[0]
        st = ComfyStatus(True, self.url, version=(s.get("system") or {}).get("comfyui_version"),
                         device=dev.get("name"),
                         vram_total_mb=int(dev.get("vram_total", 0)) // 2**20 if dev.get("vram_total") else None,
                         vram_free_mb=int(dev.get("vram_free", 0)) // 2**20 if dev.get("vram_free") else None)
        try:
            q = self._get_json("/queue")
            st.queue_running = len(q.get("queue_running", []))
            st.queue_pending = len(q.get("queue_pending", []))
        except ComfyError:
            pass
        return st

    def has_nodes(self, *class_types: str) -> dict[str, bool]:
        out = {}
        for c in class_types:
            try:
                info = self._get_json(f"/object_info/{urllib.parse.quote(c)}")
                out[c] = c in info
            except ComfyError:
                out[c] = False
        return out

    def upscale_models(self) -> list[str]:
        try:
            info = self._get_json("/object_info/UpscaleModelLoader")
            spec = info["UpscaleModelLoader"]["input"]["required"]["model_name"]
            return _combo_options(spec)
        except (ComfyError, KeyError, IndexError, TypeError):
            return []

    # ------------------------------------------------------------------ scheduling
    def wait_until_free(self, min_free_mb: int = 4000, max_wait_s: float = 120.0,
                        cancel: Callable[[], bool] | None = None) -> ComfyStatus:
        t0 = time.time()
        while True:
            st = self.status()
            if not st.available:
                raise ComfyUnavailable(st.error or "ComfyUI unavailable")
            others_busy = (st.queue_running + st.queue_pending) > 0
            vram_ok = st.vram_free_mb is None or st.vram_free_mb >= min_free_mb
            if not others_busy and vram_ok:
                return st
            if cancel and cancel():
                raise ComfyError("cancelled while waiting for ComfyUI")
            if time.time() - t0 > max_wait_s:
                raise ComfyBusy(f"ComfyUI busy (running={st.queue_running}, pending={st.queue_pending}, "
                                f"free VRAM={st.vram_free_mb} MB); Studio job deferred, nothing interrupted.")
            time.sleep(1.0)

    # ------------------------------------------------------------------ execution
    def upload_temp(self, rgb: np.ndarray, name: str) -> str:
        from PIL import Image
        bio = io.BytesIO()
        Image.fromarray((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)).save(bio, "PNG")
        boundary = uuid.uuid4().hex
        fname = f"ips_{self.client_id}_{name}.png"
        parts = []
        for k, v in (("type", "temp"), ("overwrite", "true")):
            parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{fname}\"\r\n"
                     f"Content-Type: image/png\r\n\r\n".encode() + bio.getvalue() + b"\r\n")
        parts.append(f"--{boundary}--\r\n".encode())
        r = json.loads(self._req("POST", "/upload/image", b"".join(parts),
                                 f"multipart/form-data; boundary={boundary}", timeout=60))
        sub = r.get("subfolder") or ""
        return (f"{sub}/" if sub else "") + r["name"] + " [temp]"

    def submit(self, workflow: dict[str, Any]) -> str:
        body = json.dumps({"prompt": workflow, "client_id": self.client_id}).encode()
        r = json.loads(self._req("POST", "/prompt", body, "application/json", timeout=30))
        if "prompt_id" not in r:
            raise ComfyError(f"ComfyUI rejected workflow: {r.get('error') or r.get('node_errors')}")
        pid = r["prompt_id"]
        self.our_prompts.add(pid)
        return pid

    def cancel(self, prompt_id: str) -> str:
        if prompt_id not in self.our_prompts:
            raise ComfyError("refusing to cancel a prompt this Studio session did not submit")
        try:
            q = self._get_json("/queue")
        except ComfyError:
            return "unreachable"
        running = [e[1] for e in q.get("queue_running", []) if len(e) > 1]
        pending = [e[1] for e in q.get("queue_pending", []) if len(e) > 1]
        if prompt_id in pending:
            self._req("POST", "/queue", json.dumps({"delete": [prompt_id]}).encode(), "application/json")
            return "deleted from queue"
        if prompt_id in running:
            # targeted interrupt only; older servers ignore the body and would interrupt globally, so we only
            # send it when ours is verifiably the running prompt (then global == ours).
            self._req("POST", "/interrupt", json.dumps({"prompt_id": prompt_id}).encode(), "application/json")
            return "interrupted (own running prompt)"
        return "not queued"

    def wait_result(self, prompt_id: str, timeout_s: float = 600.0,
                    cancel: Callable[[], bool] | None = None) -> dict[str, Any]:
        t0 = time.time()
        while True:
            if cancel and cancel():
                self.cancel(prompt_id)
                raise ComfyError("cancelled")
            h = self._get_json(f"/history/{prompt_id}")
            if prompt_id in h:
                entry = h[prompt_id]
                st = entry.get("status", {})
                if st.get("status_str") == "error":
                    raise ComfyError("ComfyUI execution error: " + _exec_error(st.get("messages")))
                if st.get("completed", True) and entry.get("outputs"):
                    return entry
            if time.time() - t0 > timeout_s:
                raise ComfyError(f"timeout waiting for prompt {prompt_id}")
            time.sleep(0.25)

    def fetch_images(self, entry: dict[str, Any]) -> list[np.ndarray]:
        from PIL import Image
        imgs = []
        for node_out in entry.get("outputs", {}).values():
            for im in node_out.get("images", []):
                q = urllib.parse.urlencode({"filename": im["filename"], "subfolder": im.get("subfolder", ""),
                                            "type": im.get("type", "temp")})
                data = self._req("GET", f"/view?{q}", timeout=60)
                with Image.open(io.BytesIO(data)) as pil:
                    imgs.append(np.asarray(pil.convert("RGB")).astype(np.float32) / 255.0)
        return imgs

    # ------------------------------------------------------------------ high level
    def upscale_with_model(self, rgb: np.ndarray, model_name: str, min_free_mb: int = 4000,
                           cancel: Callable[[], bool] | None = None) -> np.ndarray:
        self.wait_until_free(min_free_mb, cancel=cancel)
        name = self.upload_temp(rgb, uuid.uuid4().hex[:8])
        wf = load_workflow("upscale_with_model", image=name, model_name=model_name)
        pid = self.submit(wf)
        entry = self.wait_result(pid, cancel=cancel)
        out = self.fetch_images(entry)
        if not out:
            raise ComfyError("ComfyUI returned no image")
        return out[0]


    def qwen_edit(self, rgb: np.ndarray, prompt: str, *, face: np.ndarray | None = None, turbo: bool = False,
                  strength: float = 1.0, resolution: int = 1024, steps: int = 25, cfg: float = 1.0,
                  negative: str | None = None, seed: int | None = None, min_free_mb: int = 4000,
                  cancel: Callable[[], bool] | None = None, timeout_s: float = 1800.0) -> tuple[np.ndarray, dict]:
        """Generative Recovery with Qwen-Image 2.1 on the user's ComfyUI. Returns (image, provenance)."""
        from . import qwen
        models, why = qwen.detect(self, turbo=turbo)
        if models is None:
            raise ComfyError(why)
        self.wait_until_free(min_free_mb, cancel=cancel, max_wait_s=600)
        rw, rh = qwen.ref_size(rgb.shape[1], rgb.shape[0], resolution)
        if (rw, rh) != (rgb.shape[1], rgb.shape[0]):  # pre-size to the node's grid (one Lanczos resample, done here)
            from ...core.ops import resize_to
            rgb = resize_to(rgb, rw, rh)
        img_name = self.upload_temp(rgb, uuid.uuid4().hex[:8])
        face_name = self.upload_temp(face, uuid.uuid4().hex[:8] + "_face") if face is not None else None
        wf = qwen.build(models, img_name, prompt, face=face_name, negative=negative or qwen.DEFAULT_NEGATIVE,
                        resolution=resolution, strength=strength, steps=steps, cfg=cfg, seed=seed)
        pid = self.submit(wf)
        entry = self.wait_result(pid, timeout_s=timeout_s, cancel=cancel)
        out = self.fetch_images(entry)
        if not out:
            raise ComfyError("ComfyUI returned no image")
        prov = {"engine": "Qwen-Image 2.1" + (" Turbo" if turbo else ""), "unet": models.unet, "clip": models.clip,
                "vae": models.vae, "strength": strength, "resolution": resolution,
                "steps": None if turbo else steps, "seed": wf["13"]["inputs"]["noise_seed"] if turbo else wf["8"]["inputs"]["seed"],
                "face_reference": face is not None, "prompt_id": pid}
        return out[0], prov


def _exec_error(messages: Any) -> str:
    """Readable summary of ComfyUI's status messages (node + exception), instead of the raw event list."""
    try:
        for kind, d in messages or []:
            if kind == "execution_error":
                return (f"node {d.get('node_id')} ({d.get('node_type')}): {d.get('exception_type', '').split('.')[-1]}: "
                        f"{str(d.get('exception_message', '')).strip()}")
    except (TypeError, ValueError):
        pass
    return str(messages)[:500]


def _combo_options(spec: Any) -> list[str]:
    """Combo inputs come as ``[[opt, ...], {...}]`` (classic) or ``["COMBO", {"options": [...]}]`` (ComfyUI >= 0.3.x
    typed format, verified against v0.44)."""
    if isinstance(spec, list) and spec:
        first = spec[0]
        if isinstance(first, list):
            return [str(x) for x in first]
        if first == "COMBO" and len(spec) > 1 and isinstance(spec[1], dict):
            return [str(x) for x in spec[1].get("options", [])]
    return []


def load_workflow(name: str, **values: Any) -> dict[str, Any]:
    """Load an app-owned workflow template and substitute ``{{key}}`` placeholders."""
    text = (WORKFLOW_DIR / f"{name}.json").read_text("utf-8")
    for k, v in values.items():
        text = text.replace("{{" + k + "}}", json.dumps(v)[1:-1] if isinstance(v, str) else json.dumps(v))
    wf = json.loads(text)
    wf.pop("_meta_ips", None)
    return wf

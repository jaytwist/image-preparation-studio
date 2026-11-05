"""Integration tests against a REAL ComfyUI server (skipped unless IPS_COMFY_URL is set).

    IPS_COMFY_URL=http://127.0.0.1:8188 IPS_COMFY_ROOT=/path/to/ComfyUI pytest tests/test_comfy_real.py

The server must have RealESRGAN_x2plus.pth in models/upscale_models. Verifies against real ComfyUI code:
correctness vs Studio's direct backend, folder isolation (only temp/ changes), waiting while foreign work
runs, cancelling only Studio's own prompt, and the Studio pipeline route.
"""
import io
import json
import os
import threading
import time
import urllib.request
import uuid
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

URL = os.environ.get("IPS_COMFY_URL")
ROOT = Path(os.environ["IPS_COMFY_ROOT"]) if os.environ.get("IPS_COMFY_ROOT") else None
MODEL = "RealESRGAN_x2plus.pth"
pytestmark = pytest.mark.skipif(not URL, reason="set IPS_COMFY_URL to a real local ComfyUI server")


def _opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _snapshot(root: Path, sub: str) -> dict[str, tuple[int, float]]:
    d = root / sub
    if not d.exists():
        return {}
    return {str(p.relative_to(d)): (p.stat().st_size, p.stat().st_mtime) for p in d.rglob("*") if p.is_file()}


def _foreign_submit(img: np.ndarray) -> str:
    """Submit a job as a *different* client (simulates NH Maker 2 / the user's own ComfyUI work)."""
    from ips.backends.comfy.client import ComfyClient, load_workflow
    other = ComfyClient(URL)
    other.client_id = f"foreign-{uuid.uuid4().hex[:6]}"
    name = other.upload_temp(img, "foreign")
    return other.submit(load_workflow("upscale_with_model", image=name, model_name=MODEL)), other


def _history(pid: str) -> dict:
    with _opener().open(f"{URL}/history/{pid}", timeout=10) as r:
        return json.loads(r.read()).get(pid, {})


@pytest.fixture(scope="module")
def client():
    from ips.backends.comfy.client import ComfyClient
    c = ComfyClient(URL)
    st = c.status()
    assert st.available, st.error
    return c


def test_discovery_read_only(client):
    st = client.status()
    assert st.version
    assert MODEL in client.upscale_models()


def test_upscale_matches_direct_backend(client, corpus):
    from ips.backends.direct import DirectBackend
    from ips.core import imageio
    from skimage.metrics import structural_similarity
    img = imageio.load_image(corpus / "portrait/astronaut_quarter.png").rgb  # 128x128
    via_comfy = client.upscale_with_model(img, MODEL)
    direct = DirectBackend(device="cpu").run("realesrgan_x2plus", img, {})
    assert via_comfy.shape == direct.shape == (256, 256, 3)
    s = structural_similarity(via_comfy.mean(2), direct.mean(2), data_range=1.0)
    assert s > 0.98, f"ComfyUI and direct backend disagree (SSIM {s:.4f})"


@pytest.mark.skipif(ROOT is None, reason="set IPS_COMFY_ROOT to check folder isolation")
def test_isolation_only_temp_changes(client):
    before = {sub: _snapshot(ROOT, sub) for sub in ("input", "output", "models", "user", "custom_nodes")}
    client.upscale_with_model(np.random.default_rng(0).random((64, 64, 3)).astype(np.float32), MODEL)
    after = {sub: _snapshot(ROOT, sub) for sub in before}
    for sub in before:
        assert before[sub] == after[sub], f"Studio changed ComfyUI/{sub}"
    assert any(p.name.startswith("ips_") for p in (ROOT / "temp").rglob("*")), "upload should go to temp/"


def test_waits_for_foreign_work_and_never_touches_it(client):
    from ips.backends.comfy.client import ComfyBusy
    big = np.random.default_rng(1).random((640, 640, 3)).astype(np.float32)
    pid, other = _foreign_submit(big)
    time.sleep(0.5)
    t0 = time.time()
    with pytest.raises(ComfyBusy):
        client.wait_until_free(min_free_mb=0, max_wait_s=2.0)
    assert time.time() - t0 >= 1.5
    entry = other.wait_result(pid, timeout_s=600)
    assert entry["status"]["status_str"] == "success", "foreign job must complete untouched"
    assert client.status().queue_running == 0


def test_cancel_only_own_pending_prompt(client):
    from ips.backends.comfy.client import ComfyError, load_workflow
    big = np.random.default_rng(2).random((640, 640, 3)).astype(np.float32)
    fpid, other = _foreign_submit(big)
    time.sleep(0.5)
    with pytest.raises(ComfyError):
        client.cancel(fpid)  # refuses to touch a prompt it did not submit
    name = client.upload_temp(np.zeros((32, 32, 3), np.float32), "own")
    own = client.submit(load_workflow("upscale_with_model", image=name, model_name=MODEL))
    res = client.cancel(own)
    assert res == "deleted from queue", res
    entry = other.wait_result(fpid, timeout_s=600)
    assert entry["status"]["status_str"] == "success"
    assert not _history(own), "cancelled Studio prompt must not have executed"


def test_studio_pipeline_route(corpus, tmp_path):
    from ips.core.db import DB
    from ips.core.engine import Studio
    from ips.core.recipes import Operation, Recipe
    from ips.settings import Settings
    s = Settings.load()
    s.comfy_enabled, s.comfy_url, s.comfy_vram_min_free_mb = True, URL, 0
    s.save()
    try:
        s.output_root = str(tmp_path / "out")
        s.device = "cpu"
        st = Studio(s, DB(tmp_path / "c.sqlite"))
        r = Recipe(ops=[Operation("upscale", f"comfy:upscale:{MODEL}")], target_scale=2.0, created_by="manual")
        o = st.prepare(corpus / "portrait/astronaut_quarter.png", profiles=("archive",), recipe=r,
                       req=st.request(scale=2))
        assert o.outputs and o.outputs[0].width == 256
        assert any("ComfyUI" in w for w in o.result.warnings)
        assert o.state == "review", "external ComfyUI model output must go to review"
    finally:
        s.comfy_enabled = False
        s.save()


def test_ui_lists_comfy_models(qtbot, tmp_path):
    from ips.core.db import DB
    from ips.core.engine import Studio
    from ips.settings import Settings
    from ips.ui import recipe_editor
    from ips.ui.main_window import MainWindow
    s = Settings()
    s.comfy_enabled, s.comfy_url = True, URL
    s.output_root = str(tmp_path / "o")
    s.device = "cpu"
    w = MainWindow(Studio(s, DB(tmp_path / "u.sqlite")))
    qtbot.addWidget(w)
    w._probe_comfy()
    qtbot.waitUntil(lambda: "ComfyUI" in w.backend_lbl.text() and "connected" in w.backend_lbl.text(), timeout=30000)
    assert f"comfy:upscale:{MODEL}" in recipe_editor.EXTRA_METHODS.get("upscale", [])
    w.batch.shutdown()


# ----------------------------------------------------------------------------- Qwen-Image 2.1 (Generative Recovery)

def _qwen_or_skip(turbo=False):
    from ips.backends.comfy import qwen
    from ips.backends.comfy.client import ComfyClient
    c = ComfyClient(URL)
    m, why = qwen.detect(c, turbo=turbo)
    if m is None:
        pytest.skip(why)
    return c, m


@pytest.mark.parametrize("turbo", [False, True])
def test_qwen_graphs_pass_real_server_validation(turbo):
    """Every graph Studio can build is accepted by the real /prompt validator (node names, inputs, links, combos).
    Without IPS_QWEN_RUN=1 the prompt is withdrawn right after acceptance, so no 20 GB model load is triggered."""
    from ips.backends.comfy import qwen
    from ips.backends.comfy.client import ComfyError
    c, m = _qwen_or_skip(turbo)
    src = c.upload_temp(np.zeros((480, 640, 3), np.float32), "qsrc")
    face = c.upload_temp(np.zeros((512, 512, 3), np.float32), "qface")
    for strength, f in ((1.0, None), (0.6, face)):
        pid = c.submit(qwen.build(m, src, "test", face=f, strength=strength, resolution=1120, seed=1))
        assert pid
        try:
            c.cancel(pid)
        except ComfyError:
            pass  # already finished (e.g. failed at weight load on a test server)


@pytest.mark.skipif(not os.environ.get("IPS_QWEN_RUN"), reason="set IPS_QWEN_RUN=1 to run a real Qwen generation")
def test_qwen_real_generation():
    from ips.core.imageio import load_image
    c, _ = _qwen_or_skip(os.environ.get("IPS_QWEN_TURBO") == "1")
    src = load_image(Path(__file__).parent.parent / "fixtures" / "corpus" / "portrait_clean.png").rgb \
        if (Path(__file__).parent.parent / "fixtures" / "corpus" / "portrait_clean.png").exists() \
        else np.random.default_rng(0).random((384, 512, 3)).astype(np.float32)
    t0 = time.time()
    out, prov = c.qwen_edit(src, "Enhance this photo: sharper, cleaner, natural colours. Do not change anything else.",
                            turbo=os.environ.get("IPS_QWEN_TURBO") == "1", strength=0.6, resolution=1024, seed=1)
    print("qwen seconds", round(time.time() - t0, 1), prov)
    assert out.ndim == 3 and out.shape[2] == 3 and out.std() > 0.01

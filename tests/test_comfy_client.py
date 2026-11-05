"""ComfyUI client isolation, against an in-process fake ComfyUI server (code-level test; the real server
on the user's PC is validated separately)."""
import io
import json
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
from PIL import Image

from ips.backends.comfy.client import ComfyBusy, ComfyClient, ComfyError, ComfyUnavailable, load_workflow


class Fake:
    def __init__(self):
        self.calls = []
        self.other_running = 0
        self.vram_free = 12 * 2**30
        self.history = {}
        self.uploads = {}
        self.running = []
        self.pending = []
        self.qwen = False


def make_server(fake: Fake):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, obj, code=200):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            fake.calls.append(("GET", self.path))
            if self.path == "/system_stats":
                return self._json({"system": {"comfyui_version": "0.3.99-fake"},
                                   "devices": [{"name": "cuda:0 NVIDIA GeForce RTX 5070 Ti", "vram_total": 16 * 2**30,
                                                "vram_free": fake.vram_free}]})
            if self.path == "/queue":
                run = [[0, f"other-{i}"] for i in range(fake.other_running)] + [[0, p] for p in fake.running]
                return self._json({"queue_running": run, "queue_pending": [[1, p] for p in fake.pending]})
            if self.path.startswith("/object_info/UpscaleModelLoader"):
                # modern typed COMBO format, as returned by ComfyUI v0.44
                return self._json({"UpscaleModelLoader": {"input": {"required": {
                    "model_name": ["COMBO", {"options": ["4x_test.pth"]}]}}}})
            if fake.qwen and self.path.startswith("/object_info/"):
                node = self.path.split("/")[-1]
                combo = lambda *o: ["COMBO", {"options": list(o)}]  # noqa: E731
                req = {"UNETLoader": {"unet_name": combo("flux1-dev.safetensors",
                                                         "qwen_image_2.1_int8_convrot.safetensors",
                                                         "qwen_image_2.1_turbo_int8_convrot.safetensors")},
                       "CLIPLoader": {"clip_name": combo("t5xxl.safetensors", "qwen3vl_8b_int8_convrot.safetensors")},
                       "VAELoader": {"vae_name": combo("ae.safetensors", "qwen_image_2.1_vae_bf16.safetensors")},
                       "TextEncodeQwenImage21": {}, "QwenImage21Cache": {}}
                if node in req:
                    return self._json({node: {"input": {"required": req[node]}}})
            if self.path.startswith("/object_info/"):
                return self._json({})
            if self.path.startswith("/history/"):
                pid = self.path.split("/")[-1]
                return self._json({pid: fake.history[pid]} if pid in fake.history else {})
            if self.path.startswith("/view"):
                bio = io.BytesIO()
                Image.fromarray(np.full((8, 8, 3), 200, np.uint8)).save(bio, "PNG")
                b = bio.getvalue()
                self.send_response(200)
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)
                return
            self._json({}, 404)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(n)
            fake.calls.append(("POST", self.path, body if self.path == "/prompt" else body[:2000]))
            if self.path == "/upload/image":
                assert b'name="type"\r\n\r\ntemp' in body
                return self._json({"name": "ips_x.png", "subfolder": "", "type": "temp"})
            if self.path == "/prompt":
                d = json.loads(body)
                pid = str(uuid.uuid4())
                fake.history[pid] = {"status": {"status_str": "success", "completed": True},
                                     "outputs": {"4": {"images": [{"filename": "o.png", "subfolder": "", "type": "temp"}]}},
                                     "prompt": d}
                return self._json({"prompt_id": pid})
            if self.path in ("/queue", "/interrupt"):
                return self._json({})
            self._json({}, 404)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


@pytest.fixture()
def fake():
    f = Fake()
    srv = make_server(f)
    f.url = f"http://127.0.0.1:{srv.server_address[1]}"
    yield f
    srv.shutdown()


def test_combo_formats():
    from ips.backends.comfy.client import _combo_options
    assert _combo_options([["a.pth", "b.pth"], {}]) == ["a.pth", "b.pth"]
    assert _combo_options(["COMBO", {"options": ["a.pth"]}]) == ["a.pth"]
    assert _combo_options(["STRING", {}]) == []


def test_rejects_non_local_hosts():
    for u in ("http://192.168.1.5:8188", "http://example.com:8188", "https://comfy.cloud"):
        with pytest.raises(ValueError):
            ComfyClient(u)


def test_status_and_discovery(fake):
    c = ComfyClient(fake.url)
    st = c.status()
    assert st.available and st.vram_total_mb == 16384 and "5070" in st.device
    assert c.upscale_models() == ["4x_test.pth"]


def test_unavailable_server_reported():
    st = ComfyClient("http://127.0.0.1:9").status()
    assert not st.available and st.error


def test_upscale_round_trip_uses_temp_and_own_prompt(fake):
    c = ComfyClient(fake.url)
    out = c.upscale_with_model(np.zeros((4, 4, 3), np.float32), "4x_test.pth")
    assert out.shape == (8, 8, 3)
    posts = [x for x in fake.calls if x[0] == "POST"]
    assert {p[1] for p in posts} == {"/upload/image", "/prompt"}, "no interrupt/queue/free calls"
    pr = json.loads(next(p for p in posts if p[1] == "/prompt")[2])
    assert pr["client_id"] == c.client_id
    assert pr["prompt"]["4"]["class_type"] == "PreviewImage", "results go to temp, not output folder"
    assert "_meta_ips" not in pr["prompt"]


def test_waits_for_foreign_work_and_never_interrupts_it(fake):
    fake.other_running = 1
    c = ComfyClient(fake.url)
    with pytest.raises(ComfyBusy):
        c.wait_until_free(max_wait_s=1.5)
    assert not any(x[0] == "POST" for x in fake.calls)


def test_low_vram_defers(fake):
    fake.vram_free = 1 * 2**30
    with pytest.raises(ComfyBusy):
        ComfyClient(fake.url).wait_until_free(min_free_mb=4000, max_wait_s=1.2)


def test_cancel_only_own_prompts(fake):
    c = ComfyClient(fake.url)
    with pytest.raises(ComfyError):
        c.cancel("someone-elses-prompt")
    pid = c.submit(load_workflow("upscale_with_model", image="x.png [temp]", model_name="4x_test.pth"))
    fake.pending = [pid]
    assert c.cancel(pid) == "deleted from queue"
    fake.pending, fake.running = [], [pid]
    assert c.cancel(pid) == "interrupted (own running prompt)"
    posts = [x for x in fake.calls if x[0] == "POST" and x[1] == "/interrupt"]
    assert len(posts) == 1 and json.loads(posts[0][2])["prompt_id"] == pid


def test_pipeline_comfy_route(fake):
    from ips.core.pipeline import Pipeline
    from ips.core.recipes import Operation, Recipe
    from ips.settings import Settings
    s = Settings.load()
    s.comfy_enabled, s.comfy_url = True, fake.url
    s.save()
    try:
        r = Recipe(ops=[Operation("upscale", "comfy:upscale:4x_test.pth")], target_scale=2.0)
        res = Pipeline(backend=object()).run(np.zeros((4, 4, 3), np.float32), None, r, [], identity_check=False)
        assert res.rgb.shape == (8, 8, 3)
        assert any("ComfyUI" in w for w in res.warnings)
    finally:
        s.comfy_enabled = False
        s.save()


# ----------------------------------------------------------------------------- Qwen-Image 2.1 Generative Recovery

def test_qwen_detect_requires_nodes_and_models(fake):
    from ips.backends.comfy import qwen
    c = ComfyClient(fake.url)
    m, why = qwen.detect(c)
    assert m is None and "0.37" in why
    fake.qwen = True
    m, _ = qwen.detect(c)
    assert m.unet == "qwen_image_2.1_int8_convrot.safetensors" and not m.turbo
    assert m.clip.startswith("qwen3vl") and m.vae == "qwen_image_2.1_vae_bf16.safetensors"
    mt, _ = qwen.detect(c, turbo=True)
    assert "turbo" in mt.unet and mt.turbo


def test_qwen_graph_standard_and_turbo():
    from ips.backends.comfy import qwen
    m = qwen.QwenModels("u.safetensors", "c.safetensors", "v.safetensors", turbo=False)
    g = qwen.build(m, "in.png", "PROMPT", face="face.png", resolution=1536, strength=0.6, seed=7)
    enc = g["7"]["inputs"]
    assert enc["images.image_1"] == ["5", 0] and enc["images.image_2"] == ["6", 0]
    assert enc["prompt"].startswith("<image1>") and "PROMPT" in enc["prompt"] and "<image2>" in enc["prompt"]
    assert enc["resolution"] == 1536
    ks = g["8"]["inputs"]
    # the node's latent output is EMPTY (nodes_qwen.py): strength < 1 must start from the VAE-encoded source
    assert g["8"]["class_type"] == "KSampler" and ks["denoise"] == 0.6 and ks["latent_image"] == ["15", 0]
    assert g["15"]["class_type"] == "VAEEncode" and g["15"]["inputs"]["pixels"] == ["5", 0]
    g1 = qwen.build(m, "in.png", "PROMPT", strength=1.0)
    assert g1["8"]["inputs"]["latent_image"] == ["7", 2] and "15" not in g1, "1.0 = the official template"
    assert g1["7"]["inputs"]["prompt"] == "PROMPT", "single image: no tags, prompt unchanged"
    assert ks["model"] == ["2", 0] and g["2"]["class_type"] == "QwenImage21Cache"
    assert g["10"]["class_type"] == "PreviewImage", "results stay in ComfyUI temp"
    # every link points at an existing node
    for nid, node in g.items():
        for v in node["inputs"].values():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                assert v[0] in g, (nid, v)
    gt = qwen.build(qwen.QwenModels("u", "c", "v", turbo=True), "in.png", "P", strength=0.8, seed=1)
    sig = [float(x) for x in gt["11"]["inputs"]["sigmas"].split(",")]
    assert sig[0] == 0.8 and sig[-1] == 0 and all(a > b for a, b in zip(sig, sig[1:]))
    assert gt["8"]["class_type"] == "SamplerCustomAdvanced" and "6" not in gt
    assert qwen.build(qwen.QwenModels("u", "c", "v", turbo=True), "i", "P", strength=1.0)["11"]["inputs"][
        "sigmas"].startswith("1, 0.978453")
    assert qwen.ref_size(1280, 960, 1120) == (1280, 960) and qwen.ref_size(1000, 1000, 1024) == (1024, 1024)
    assert qwen.resolution_for(640, 480, 2.0) == 1120 and qwen.resolution_for(4000, 3000, 2.0) == 2048


def test_qwen_router_and_pipeline_round_trip(fake):
    """Generative mode end-to-end against the fake server: one op, exact geometry, labelled, provenance."""
    from ips.core.classify import Assessment
    from ips.core.diagnostics import Face
    from ips.core.pipeline import Pipeline
    from ips.core.router import RouteRequest, route
    from ips.settings import Settings
    fake.qwen = True
    a = Assessment(content_class="photo", content_scores={}, content_confidence=0.9)
    a.features["face_count"] = 1
    r = route(a, RouteRequest(mode="generative", generative={"strength": 0.7, "long_edge": 128}), 64, 48)
    assert [o.kind for o in r.ops] == ["generative"] and r.is_generative and r.uses_external
    assert r.quality_override and r.outside_fast_profile and r.target_scale == 2.0
    assert r.ops[0].params["strength"] == 0.7 and "phone photo" in r.ops[0].params["prompt"]
    s = Settings.load()
    s.comfy_enabled, s.comfy_url = True, fake.url
    s.save()
    try:
        face = Face(box=(20.0, 10.0, 40.0, 34.0), score=0.9)
        res = Pipeline(backend=object()).run(np.zeros((48, 64, 3), np.float32), None, r, [face], identity_check=False)
        assert res.rgb.shape == (96, 128, 3), "mapped back to the exact requested geometry"
        assert res.provenance and res.provenance[0]["face_reference"] and res.provenance[0]["strength"] == 0.7
        uploads = [x for x in fake.calls if x[0] == "POST" and x[1] == "/upload/image"]
        assert len(uploads) == 2, "source + face reference"
        prompt = json.loads(next(x for x in fake.calls if x[0] == "POST" and x[1] == "/prompt")[2])["prompt"]
        assert prompt["8"]["inputs"]["denoise"] == 0.7
        assert not any(x[1] in ("/interrupt", "/free") for x in fake.calls if x[0] == "POST")
    finally:
        s.comfy_enabled = False
        s.save()

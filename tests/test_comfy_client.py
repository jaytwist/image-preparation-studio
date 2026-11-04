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
                return self._json({"UpscaleModelLoader": {"input": {"required": {"model_name": [["4x_test.pth"]]}}}})
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
            fake.calls.append(("POST", self.path, body[:2000]))
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

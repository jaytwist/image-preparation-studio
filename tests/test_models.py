"""Model backend: loading, scale, tiling seams, native parameters, identity drift signals."""
import numpy as np
import pytest

from ips.backends.registry import MODELS
from ips.core import imageio
from tests.conftest import models_installed

pytestmark = pytest.mark.models

RUNNABLE = [m.id for m in MODELS.values() if m.task in ("jpeg", "denoise", "deblur", "upscale") and m.installed
            and m.status != "reject"]


@pytest.fixture(scope="module")
def be():
    from ips.backends.direct import DirectBackend
    return DirectBackend(device="cpu", max_cached=2)


@pytest.fixture(scope="module")
def small(corpus):
    im = imageio.load_image(corpus / "portrait/astronaut_clean.png")
    return np.ascontiguousarray(im.rgb[200:264, 180:260])  # 64x80 crop with facial detail


@pytest.mark.parametrize("mid", RUNNABLE)
def test_every_installed_model_runs(be, small, mid):
    out = be.run(mid, small, MODELS[mid].native_defaults())
    s = MODELS[mid].scale
    assert out.shape == (small.shape[0] * s, small.shape[1] * s, 3)
    assert np.isfinite(out).all() and 0 <= out.min() and out.max() <= 1
    import cv2
    down = cv2.resize(out, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA)
    assert np.abs(down - small).mean() < 0.08, "output must stay close to the input content"


def test_tiling_is_seamless(corpus):
    from ips.backends.direct import DirectBackend
    im = imageio.load_image(corpus / "dslr/coffee_iso6400.jpg").rgb[:160, :200]
    full = DirectBackend(device="cpu", tile=512).run("drunet_color", im, {"sigma": 10})
    tiled = DirectBackend(device="cpu", tile=96, overlap=16).run("drunet_color", im, {"sigma": 10})
    assert np.abs(full - tiled).max() < 0.03 and np.abs(full - tiled).mean() < 0.002


def test_native_params_change_output(be, corpus):
    im = imageio.load_image(corpus / "dslr/coffee_iso6400.jpg").rgb[:96, :96]
    lo = be.run("drunet_color", im, {"sigma": 2})
    hi = be.run("drunet_color", im, {"sigma": 40})
    assert np.abs(hi - im).mean() > np.abs(lo - im).mean() * 2, "σ is a real native control"
    j = imageio.load_image(corpus / "screenshot/ui_q45.jpg").rgb[:96, :128]
    q10 = be.run("fbcnn_color", j, {"qf": 10})
    q90 = be.run("fbcnn_color", j, {"qf": 90})
    assert np.abs(q10 - j).mean() > np.abs(q90 - j).mean()
    if models_installed("realesr_general_x4v3") and (MODELS["realesr_general_x4v3"].path.parent /
                                                    "realesr-general-wdn-x4v3.pth").is_file():
        a = be.run("realesr_general_x4v3", im[:32, :32], {"denoise_strength": 0.0})
        b = be.run("realesr_general_x4v3", im[:32, :32], {"denoise_strength": 1.0})
        assert np.abs(a - b).mean() > 1e-3, "DNI weight interpolation must change output"


def test_fbcnn_blind_qf_tracks_true_quality(be, corpus):
    from ips.eval.fixtures import jpeg_roundtrip
    im = imageio.load_image(corpus / "gt/dslr_coffee_gt.png").rgb
    preds = {q: be.predict_jpeg_qf(jpeg_roundtrip(im, q)) for q in (20, 50, 85)}
    assert preds[20] < preds[50] < preds[85], preds


def test_identity_flags_beautification(corpus):
    """A deliberate skin-smoothing edit on a face must raise review signals; a lossless copy must not."""
    from ips.core.identity import FaceAnalyzer, compare_faces
    import cv2
    im = imageio.load_image(corpus / "portrait/astronaut_clean.png").rgb
    fa = FaceAnalyzer.get()
    faces = fa.detect(im)
    assert faces, "face detector must find the portrait face"
    clean = compare_faces(im, im.copy(), faces, fa)
    assert not clean.flags
    u8 = (im * 255).astype(np.uint8)
    smooth = u8.copy()
    for _ in range(4):
        smooth = cv2.bilateralFilter(smooth, 9, 60, 9)
    x0, y0, x1, y1 = [int(v) for v in faces[0].box]
    beaut = u8.copy()
    beaut[y0:y1, x0:x1] = smooth[y0:y1, x0:x1]
    rep = compare_faces(im, beaut.astype(np.float32) / 255, faces, fa)
    assert any("smoothing" in f for f in rep.flags), rep.flags
    warped = cv2.resize(u8[y0:y1, x0:x1], None, fx=1.0, fy=1.12)[: y1 - y0]
    geo = u8.copy()
    geo[y0:y1, x0:x1] = cv2.resize(warped, (x1 - x0, y1 - y0)) if warped.shape[0] != y1 - y0 else warped
    rep2 = compare_faces(im, geo.astype(np.float32) / 255, faces, fa)
    assert rep2.faces[0].region_ssim < clean.faces[0].region_ssim


def test_pipeline_end_to_end_small(corpus, tmp_path):
    from ips.core.db import DB
    from ips.core.engine import Studio
    from ips.settings import Settings
    s = Settings()
    s.output_root = str(tmp_path / "o")
    s.device = "cpu"
    st = Studio(s, DB(tmp_path / "d.sqlite"))
    o = st.prepare(corpus / "old_phone/group_q28.jpg", profiles=("archive", "share"), req=st.request(scale=1))
    assert o.recipe.ops and o.recipe.ops[0].method == "model:fbcnn_color"
    assert o.result.identity and o.result.identity["faces"], "faces must be checked"
    assert o.timings_ms["total_job_ms"] > 0 and "warm_job_ms" in o.timings_ms
    assert all(x.path.is_file() for x in o.outputs)

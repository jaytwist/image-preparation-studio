"""Quick Edit: cutout quality vs ground-truth alpha, recolor properties, mask undo, removal, geometry."""
import numpy as np
import pytest
from PIL import Image

from ips.core import imageio
from ips.core import ops as cops
from ips.quickedit.edits import MaskDoc, recolor, remove_object, replace_background
from ips.quickedit.matting import Matting, adjust_edge, decontaminate
from tests.conftest import models_installed


def _iou(a, b, t=0.5):
    A, B = a > t, b > t
    return (A & B).sum() / max(1, (A | B).sum())


@pytest.mark.skipif(not models_installed("u2net_human_seg"), reason="u2net weights missing")
@pytest.mark.parametrize("bg", ["plain", "texture", "similar"])
def test_person_cutout_iou(corpus, bg):
    im = imageio.load_image(corpus / f"cutout/children_{bg}.png")
    gt = np.asarray(Image.open(corpus / "gt/children_alpha.png").convert("L")).astype(np.float32) / 255
    a = Matting.person_mask(im.rgb)
    iou = _iou(a, gt)
    sad = float(np.abs(a - gt).mean())
    print(f"cutout {bg}: IoU={iou:.3f} SAD/px={sad:.4f}")
    assert iou > (0.80 if bg != "similar" else 0.65), iou


def test_alpha_edge_controls():
    a = np.zeros((64, 64), np.float32)
    a[16:48, 16:48] = 1
    f = adjust_edge(a, feather_px=2)
    assert 0 < f[16, 30] < 1 and f[32, 32] == pytest.approx(1, abs=1e-3)
    g = adjust_edge(a, shift_px=2)
    assert g.sum() > a.sum()
    rgb = np.ones((64, 64, 3), np.float32) * 0.2
    rgb[16:48, 16:48] = [0.9, 0.1, 0.1]
    out = decontaminate(rgb, f)
    assert out.shape == rgb.shape


def test_recolor_preserves_shading_and_outside(corpus):
    im = imageio.load_image(corpus / "portrait/astronaut_clean.png")
    md = MaskDoc(im.height, im.width)
    md.magic_select(im.rgb, 150, 450, tolerance=14)  # orange suit
    assert 0.002 < md.mask.mean() < 0.4
    md.feather(1.5)
    out = recolor(im.rgb, md.mask, (0.18, 0.43, 0.84))
    import cv2
    L0 = cv2.cvtColor(im.rgb, cv2.COLOR_RGB2LAB)[..., 0]
    L1 = cv2.cvtColor(out, cv2.COLOR_RGB2LAB)[..., 0]
    inside = md.mask > 0.9
    # lightness structure (folds/shading) preserved: high correlation inside mask
    corr = np.corrcoef(L0[inside], L1[inside])[0, 1]
    assert corr > 0.97, corr
    outside = md.mask <= 1.0 / 1024
    assert np.array_equal(out[outside], im.rgb[outside]), "pixels outside the mask must be untouched"
    lab = cv2.cvtColor(out, cv2.COLOR_RGB2LAB)
    hue = np.degrees(np.arctan2(lab[..., 2][inside], lab[..., 1][inside])).mean()
    tl = cv2.cvtColor(np.array([[[0.18, 0.43, 0.84]]], np.float32), cv2.COLOR_RGB2LAB)[0, 0]
    assert abs(hue - np.degrees(np.arctan2(tl[2], tl[1]))) < 15


def test_mask_undo_redo_brush():
    m = MaskDoc(50, 50)
    m.begin_stroke()
    m.brush(25, 25, 5, add=True, x2=40, y2=25)
    s1 = m.mask.sum()
    assert s1 > 0
    m.begin_stroke()
    m.brush(25, 25, 3, add=False)
    assert m.mask.sum() < s1
    assert m.undo() and m.mask.sum() == pytest.approx(s1)
    assert m.undo() and m.mask.sum() == 0
    assert m.redo() and m.mask.sum() == pytest.approx(s1)
    m.invert()
    assert m.mask.mean() > 0.5


@pytest.mark.parametrize("method", ["telea", "lama"])
def test_object_removal(method):
    if method == "lama" and not models_installed("lama"):
        pytest.skip("LaMa missing")
    rng = np.random.default_rng(0)
    base = np.dstack([np.tile(np.linspace(0.2, 0.8, 128), (128, 1))] * 3).astype(np.float32)
    base += rng.normal(0, 0.01, base.shape).astype(np.float32)
    img = base.copy()
    img[50:70, 50:70] = [1, 0, 0]
    mask = np.zeros((128, 128), np.float32)
    mask[48:72, 48:72] = 1
    out, used = remove_object(img, mask, method)
    assert ("LaMa" in used) == (method == "lama")
    err = np.abs(out[50:70, 50:70] - base[50:70, 50:70]).mean()
    assert err < 0.08, (used, err)
    far = np.zeros_like(mask, bool)
    far[:20] = True
    assert np.abs(out[far] - img[far]).max() < 1e-5


def test_background_replace_and_geometry():
    rgb = np.random.default_rng(1).random((40, 60, 3)).astype(np.float32)
    a = np.zeros((40, 60), np.float32)
    a[10:30, 20:40] = 1
    out = replace_background(rgb, a, "color", (1, 1, 1))
    assert np.allclose(out[0, 0], 1) and np.allclose(out[20, 30], rgb[20, 30])
    r = cops.crop_rotate(rgb, rotate90=1)
    assert r.shape[:2] == (60, 40)
    c = cops.crop_rotate(rgb, crop=[5, 5, 25, 15])
    assert c.shape[:2] == (10, 20)
    s = cops.crop_rotate(np.ones((200, 300, 3), np.float32), angle=5)
    assert s.shape[0] < 200 and s.shape[1] < 300 and s.min() > 0.99, "auto-crop removes rotated borders"


def test_straighten_estimate():
    import cv2
    img = np.ones((400, 600, 3), np.float32)
    for y in range(60, 360, 50):
        cv2.line(img, (20, y), (580, y), (0, 0, 0), 2)
    M = cv2.getRotationMatrix2D((300, 200), 3.0, 1.0)
    rot = cv2.warpAffine(img, M, (600, 400), borderValue=(1, 1, 1))
    ang = cops.estimate_straighten_angle(rot)
    assert ang is not None and abs(abs(ang) - 3.0) < 0.6, ang

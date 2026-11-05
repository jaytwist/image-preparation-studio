"""Smart Auto: diagnosis + routing decisions on the corpus (expectations derived from injected truth)."""
import pytest

from ips.core.recipes import Recipe, default_op, map_strength

RESTORE_KINDS = {"jpeg_cleanup", "denoise", "deblur", "white_balance", "exposure", "tone", "sharpen"}


@pytest.fixture(scope="module")
def st(tmp_path_factory):
    from ips.core.db import DB
    from ips.core.engine import Studio
    from ips.settings import Settings
    d = tmp_path_factory.mktemp("auto")
    s = Settings()
    s.output_root = str(d / "out")
    s.device = "cpu"
    return Studio(s, DB(d / "t.sqlite"))


def _route(st, corpus, rel, **req):
    req.setdefault("look", "natural")   # defect-repair logic is tested without the visible "look" layer
    src = st.inspect(corpus / rel, record=False)
    r = st.auto_recipe(src, st.request(**req))
    return src, r


def kinds(r):
    return {o.kind for o in r.ops if o.enabled}


@pytest.mark.parametrize("rel", ["dslr/chelsea_clean.jpg", "portrait/astronaut_clean.png", "screenshot/ui_clean.png",
                                 "ai/scene_comfy.png", "ai/portrait_a1111.png"])
def test_clean_images_are_left_alone(st, corpus, rel):
    src, r = _route(st, corpus, rel, scale=1)
    assert r.no_op, f"{rel}: expected no-op, got {r.summary()}\n" + "\n".join(r.rationale)
    assert any("No enhancement needed" in x for x in r.rationale)


def test_low_res_clean_gets_only_upscale(st, corpus):
    _, r = _route(st, corpus, "dslr/chelsea_clean.jpg")
    assert not (kinds(r) & RESTORE_KINDS), r.summary()
    assert r.target_scale == 2.0


def test_old_phone_recipe(st, corpus):
    src, r = _route(st, corpus, "old_phone/astronaut_oldphone.jpg")
    k = kinds(r)
    assert {"jpeg_cleanup", "upscale"} <= k and "white_balance" not in k, r.summary()
    assert any("White balance available in Manual" in f for f in r.flags), "detected cast must be surfaced"
    j = next(o for o in r.ops if o.kind == "jpeg_cleanup")
    assert j.method == "model:fbcnn_color" and 25 <= j.params["qf"] <= 45, "qf must reflect the injected q35"
    up = next(o for o in r.ops if o.kind == "upscale")
    assert up.method == "model:swinir_x2" and up.blend == 1.0
    order = [o.kind for o in r.ops]
    assert order.index("jpeg_cleanup") < order.index("upscale")
    assert any("re-saved" in o.why for o in r.ops if o.kind == "jpeg_cleanup"), "re-save must be explained"


def test_noisy_dslr_gets_denoise_only(st, corpus):
    _, r = _route(st, corpus, "dslr/coffee_iso6400.jpg", scale=1)
    assert kinds(r) == {"denoise"}, r.summary()
    assert r.ops[0].method == "model:scunet_real_psnr"


def test_screenshot_jpeg_and_upscale(st, corpus):
    _, r = _route(st, corpus, "screenshot/ui_q45.jpg", scale=1)
    assert kinds(r) == {"jpeg_cleanup"}, r.summary()
    assert any("legibility" in f for f in r.flags)
    _, r2 = _route(st, corpus, "screenshot/ui_half.png", scale=2)
    up = [o for o in r2.ops if o.kind == "upscale"]
    assert up and all(not o.model_id for o in up), "text/UI upscale must not use a learned model in Auto"


def test_previously_edited_not_resharpened_or_resmoothed(st, corpus):
    _, r = _route(st, corpus, "edited/chelsea_oversharp.jpg", scale=1)
    assert "sharpen" not in kinds(r)
    src, r2 = _route(st, corpus, "edited/astronaut_smooth.jpg", scale=1)
    assert not ({"denoise", "sharpen"} & kinds(r2)), r2.summary()
    _, r3 = _route(st, corpus, "edited/coffee_hdr.jpg", scale=1)
    assert not ({"sharpen", "tone"} & kinds(r3)), r3.summary()


def test_faded_scan(st, corpus):
    src, r = _route(st, corpus, "edited/print_scan_faded.jpg", scale=1)
    assert src.assessment.content_class == "scan"
    assert {"exposure", "white_balance"} <= kinds(r), r.summary()


def test_ai_jpeg_routed_by_pixels(st, corpus):
    _, r = _route(st, corpus, "ai/portrait_grok_q60.jpg", scale=1)
    assert "jpeg_cleanup" in kinds(r)
    assert "white_balance" not in kinds(r)


def test_ai_metadata_detected(st, corpus):
    src, _ = _route(st, corpus, "ai/scene_comfy.png", scale=1)
    assert src.assessment.content_class == "ai_generated"
    assert src.summary["ai_generation_hints"]


def test_routing_deterministic_and_serialisable(st, corpus):
    _, a = _route(st, corpus, "old_phone/astronaut_oldphone.jpg")
    _, b = _route(st, corpus, "old_phone/astronaut_oldphone.jpg")
    assert a.hash == b.hash
    c = Recipe.from_json(a.to_json())
    assert c.hash == a.hash and c.summary() == a.summary()


def test_baseline_models_never_auto_selected(st, corpus, manifest):
    """RealPLKSR (and every non-'adopt' model) is Manual-only until measurements justify promotion."""
    from ips.backends.registry import MODELS
    for it in manifest.values():
        if it["category"] in ("batch", "metadata", "cutout", "recolor") or it["path"].endswith(".NEF"):
            continue
        for mode in ("faithful", "balanced"):
            _, r = _route(st, corpus, it["path"], mode=mode)
            for o in r.ops:
                if o.model_id:
                    assert MODELS[o.model_id].status == "adopt", (it["id"], mode, o.model_id)


def test_strength_mapping_is_honest():
    p, b, why = map_strength("model:fbcnn_color", 100)
    assert p == {"qf": 10} and b == 1.0 and why.startswith("native")
    p, b, why = map_strength("model:drunet_color", 40)
    assert p == {"sigma": 20.0} and why.startswith("native")
    p, b, why = map_strength("model:scunet_real_psnr", 30)
    assert p == {} and b == 0.3 and why.startswith("post-blend")
    p, b, why = map_strength("model:realesr_general_x4v3", 25)
    assert p == {"denoise_strength": 0.25} and why.startswith("native")
    assert default_op("upscale", "model:swinir_x2").blend == 1.0


def test_rationale_always_present(st, corpus):
    _, r = _route(st, corpus, "dslr/coffee_iso6400.jpg")
    assert r.rationale and all(o.why for o in r.ops)
    assert r.predicted_seconds is not None


@pytest.mark.parametrize("mode", ["faithful", "balanced"])
@pytest.mark.parametrize("rel", ["screenshot/ui_half.png", "screenshot/ui_clean.png"])
def test_regression_no_learned_upscaler_on_text(st, corpus, rel, mode):
    """Regression (evidence: reports/eval/ocr_ui_x2.json, docs/TEST_RESULTS.md): GAN/learned upscalers
    produced legible-but-wrong text on the UI fixture (e.g. "$1,049.00" -> "$1,019.00", "Approved" ->
    "Aprorved"). Text/UI content must never be routed to a learned upscaler by Auto, in any mode."""
    _, r = _route(st, corpus, rel, mode=mode, scale=2)
    ups = [o for o in r.ops if o.kind == "upscale"]
    assert ups and all(not o.model_id for o in ups), r.summary()


@pytest.mark.parametrize("look", ["enhanced", "vivid"])
def test_look_visibly_enhances_clean_images_without_models(st, corpus, look):
    import numpy as np
    src, r = _route(st, corpus, "portrait/astronaut_clean.png", scale=1, look=look)
    assert not r.no_op and not any(o.model_id for o in r.ops), r.summary()
    assert {"tone", "sharpen"} <= {o.kind for o in r.ops}
    res = st.process(src, r)
    diff = float(np.abs(res.rgb - src.image.rgb).mean())
    assert diff > (0.015 if look == "enhanced" else 0.025), f"{look} must be visibly different (mean |Δ| {diff:.4f})"
    sims = [f["embedding_similarity"] for f in res.identity["faces"] if f.get("embedding_similarity") is not None]
    assert sims and min(sims) >= 0.95, f"{look} look must keep identity (similarity {sims})"
    assert not [f for f in res.identity["flags"] if "identity" in f or "geometry" in f]


def test_look_stronger_levels_are_ordered(st, corpus):
    import numpy as np
    src = st.inspect(corpus / "dslr/chelsea_clean.jpg", record=False)
    d = {}
    for look in ("natural", "enhanced", "vivid"):
        res = st.process(src, st.auto_recipe(src, st.request(scale=1, look=look)))
        d[look] = float(np.abs(res.rgb - src.image.rgb).mean())
    assert d["natural"] < d["enhanced"] < d["vivid"], d


def test_look_not_applied_to_screenshots(st, corpus):
    _, r = _route(st, corpus, "screenshot/ui_clean.png", scale=1, look="vivid")
    assert r.no_op, r.summary()

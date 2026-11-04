"""Fast Profile gating (empirical 15 s contract) and evidence-driven learning from corrections."""
from ips.core.perf import FastProfile, bucket, route_key


def test_fast_profile_gate(studio, corpus, tmp_path, monkeypatch):
    from ips.backends.direct import DeviceInfo
    prof = FastProfile(tmp_path / "fp.json")
    dev = DeviceInfo("cuda", "NVIDIA GeForce RTX 5070 Ti (fake for test)", 16303, 15000, "x", "12.8", "sm_120", True)
    studio._dev = dev
    studio.profile = prof
    src = studio.inspect(corpus / "old_phone/astronaut_oldphone.jpg", record=False)
    r = studio.auto_recipe(src, studio.request())
    # uncertified device: no gating, informational rationale only
    assert not r.outside_fast_profile
    assert any("not certified" in x for x in r.rationale)
    # certify the device with a measurement of a *different* route -> this route is unmeasured
    prof.record(dev.name, {"noop@x1": {"1": {"p95_s": 1.0}}}, {}, {"t": "test"})
    r2 = studio.auto_recipe(src, studio.request())
    assert r2.outside_fast_profile and any("Quality Override" in f for f in r2.flags)
    o = studio.prepare(corpus / "old_phone/astronaut_oldphone.jpg", req=studio.request(), src=src)
    assert o.state == "needs_override" and not o.outputs, "must not run or silently degrade"
    # measured and within budget -> supported
    prof.record(dev.name, {route_key(r2): {str(bucket(src.image.megapixels)): {"p95_s": 9.5}}}, {}, {})
    r3 = studio.auto_recipe(src, studio.request())
    assert not r3.outside_fast_profile
    # measured but over budget -> requires override again; explicit override labels the run
    prof.record(dev.name, {route_key(r2): {str(bucket(src.image.megapixels)): {"p95_s": 22.0}}}, {}, {})
    r4 = studio.auto_recipe(src, studio.request())
    assert r4.outside_fast_profile and any("exceeds 15 s" in f for f in r4.flags)
    r5 = studio.auto_recipe(src, studio.request(quality_override=True))
    assert any("explicit Quality Override" in f for f in r5.flags)
    assert r5.target_scale == r2.target_scale, "resolution is never reduced to meet the deadline"


def test_learning_from_corrections(tmp_path, monkeypatch, corpus):
    import os
    monkeypatch.setenv("IPS_DATA_DIR", str(tmp_path / "data"))
    from ips.core.db import DB
    from ips.core.engine import Studio
    from ips.eval.harness import learn
    from ips.settings import Settings
    s = Settings()
    s.output_root = str(tmp_path / "o")
    st = Studio(s, DB())  # default DB path under the patched data dir
    src = st.inspect(corpus / "edited/print_scan_faded.jpg")
    auto = st.auto_recipe(src, st.request(scale=1))
    assert "white_balance" in {o.kind for o in auto.ops}
    final = auto.clone()
    final.ops = [o for o in final.ops if o.kind != "white_balance"]
    for _ in range(5):
        st.record_correction(src, auto, final, "manual_export")
    adj = learn()
    assert adj["scan"]["white_balance"]["suppress"] and not adj["scan"]["exposure"]["suppress"]
    r = st.auto_recipe(src, st.request(scale=1))
    assert "white_balance" not in {o.kind for o in r.ops}
    assert any("learned from 5 manual corrections" in x for x in r.rationale)
    os.remove(tmp_path / "data" / "router_adjustments.json")

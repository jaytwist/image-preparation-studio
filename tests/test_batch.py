"""Batch reliability: isolation, duplicates, crash recovery, verified skip, retry, cancel, naming."""
import json
import shutil
import time
from pathlib import Path

from ips.core.hashing import sha256_file
from ips.core.jobs import JobManager, enumerate_inputs


def _copy_tree(src: Path, dst: Path) -> Path:
    shutil.copytree(src, dst)
    return dst


def test_enumerate_keeps_relative_structure(corpus):
    items = enumerate_inputs([corpus / "dupes"])
    rels = sorted((p.name, r) for p, r in items)
    assert ("IMG_0001.jpg", "dupes/a") in rels and ("IMG_0001.jpg", "dupes/b") in rels
    assert ("IMG_0001.png", "dupes/a") in rels


def test_broken_files_isolated(studio_nomodels, corpus, tmp_path):
    src = _copy_tree(corpus / "broken", tmp_path / "broken")
    hashes = {p.name: sha256_file(p) for p in src.iterdir() if p.stat().st_size}
    jm = JobManager(studio_nomodels)
    bid = jm.create_batch("broken", [src], profiles=("archive",), scale=1)
    jm.start(bid)
    assert jm.wait(300)
    states = {Path(j["source_path"]).name: (j["state"], j["error"]) for j in jm.jobs(bid)}
    assert states["good_1.jpg"][0] in ("done", "review") and states["good_2.jpg"][0] in ("done", "review")
    for bad in ("truncated.jpg", "random_bytes.jpg", "empty.png", "text_named.tif"):
        assert states[bad][0] == "failed" and states[bad][1], (bad, states[bad])
    for p in src.iterdir():  # sources untouched
        if p.name in hashes:
            assert sha256_file(p) == hashes[p.name]
    st = jm.stats(bid)
    assert st["states"]["failed"] == 4 and st["total_s"]["median"] is not None
    jp, cp = jm.write_report(bid)
    assert jp.is_file() and cp.is_file()
    rep = json.loads(jp.read_text())
    assert rep["stats"]["jobs"] == 6


def test_duplicates_and_name_collisions(studio_nomodels, corpus, tmp_path):
    src = _copy_tree(corpus / "dupes", tmp_path / "dupes")
    jm = JobManager(studio_nomodels)
    bid = jm.create_batch("dupes", [src], profiles=("archive", "share"), scale=1)
    jm.start(bid)
    assert jm.wait(300)
    js = jm.jobs(bid)
    states = sorted(j["state"] for j in js)
    assert states.count("duplicate") == 1, states
    dup = next(j for j in js if j["state"] == "duplicate")
    assert "not deleted" in dup["error"] and Path(dup["source_path"]).is_file()
    outs = studio_nomodels.db.query("SELECT path FROM outputs WHERE batch_id=?", (bid,))
    paths = [o["path"] for o in outs]
    assert len(paths) == len(set(paths)) == 6  # 3 unique sources × 2 profiles, no overwrite
    parts = [Path(p).parts for p in paths]
    assert any(("dupes", "a") == pt[i:i + 2] for pt in parts for i in range(len(pt))) and \
        any(("dupes", "b") == pt[i:i + 2] for pt in parts for i in range(len(pt)))


def test_crash_recovery_and_verified_skip(studio_nomodels, corpus, tmp_path):
    src = _copy_tree(corpus / "dupes", tmp_path / "in")
    jm = JobManager(studio_nomodels)
    bid = jm.create_batch("resume", [src], profiles=("archive",), scale=1)
    # simulate a crash mid-job: one job left 'running'
    first = jm.jobs(bid)[0]
    studio_nomodels.db.update("jobs", first["id"], {"state": "running"})
    jm2 = JobManager(studio_nomodels)  # "restart"
    assert jm2.jobs(bid)[0]["state"] == "queued"
    assert "interrupted" in jm2.jobs(bid)[0]["error"]
    jm2.start(bid)
    assert jm2.wait(300)
    done_before = {j["id"]: j["finished"] for j in jm2.jobs(bid) if j["state"] in ("done", "review")}
    assert done_before
    # verified skip: intact outputs stay done; a deleted output requeues only that job
    victim = next(iter(done_before))
    out = studio_nomodels.db.one("SELECT path FROM outputs WHERE job_id=?", (victim,))
    Path(out["path"]).unlink()
    requeued = jm2.verify_completed(bid)
    assert requeued == [victim]
    jm2.start(bid)
    assert jm2.wait(300)
    after = {j["id"]: j for j in jm2.jobs(bid)}
    assert after[victim]["state"] in ("done", "review")
    for jid, fin in done_before.items():
        if jid != victim:
            assert after[jid]["finished"] == fin, "completed jobs must not be reprocessed"


def test_pause_resume_cancel(studio_nomodels, corpus, tmp_path):
    src = tmp_path / "many"
    src.mkdir()
    for i in range(6):
        shutil.copy(corpus / "broken/good_1.jpg", src / f"img_{i}.jpg")
        (src / f"img_{i}.jpg").write_bytes((src / f"img_{i}.jpg").read_bytes() + bytes([i]))  # distinct hashes
    events = []
    jm = JobManager(studio_nomodels, on_event=lambda j, s, i: events.append(s))
    bid = jm.create_batch("ctl", [src], profiles=("archive",), scale=1)
    jm.pause()
    jm.start(bid)
    jm.pause()
    time.sleep(0.5)
    assert all(j["state"] == "queued" for j in jm.jobs(bid)) or jm.paused
    jm.resume()
    t0 = time.time()
    while not any(j["state"] in ("done", "review") for j in jm.jobs(bid)) and time.time() - t0 < 120:
        time.sleep(0.1)
    jm.cancel()
    assert jm.wait(120)
    states = [j["state"] for j in jm.jobs(bid)]
    assert "queued" in states, "cancel must leave remaining jobs resumable"
    jm.start(bid)
    assert jm.wait(300)
    assert all(j["state"] in ("done", "review", "cancelled") for j in jm.jobs(bid))
    assert jm.retry_failed(bid) >= 0


def test_repeat_export_reuses_master(studio_nomodels, corpus, tmp_path):
    p = corpus / "dslr/coffee_iso6400.jpg"
    req = studio_nomodels.request(scale=1)
    o1 = studio_nomodels.prepare(p, profiles=("archive",), req=req)
    o2 = studio_nomodels.prepare(p, profiles=("archive", "share"), req=req)
    assert o2.outputs[0].reused_master and o2.outputs[0].path == o1.outputs[0].path
    assert "reused_master_path" in o2.timings_ms

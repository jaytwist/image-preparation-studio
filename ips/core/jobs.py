"""Persistent batch queue.

* Jobs live in SQLite; a crash leaves them 'running', which ``recover()`` resets to 'queued'.
* Completed jobs are skipped only if every recorded output still exists with the recorded SHA-256.
* Per-file isolation: any exception marks that job 'failed' with a reason; the batch continues.
* One GPU worker; the next file's decode/diagnosis is prefetched on a CPU thread (backpressure: 1).
* Exact duplicates (same SHA-256 within a batch) are marked 'duplicate' and skipped – never deleted.
  Visually similar images (dHash) are listed as review *suggestions* only.
"""
from __future__ import annotations

import csv
import io
import json
import logging
import statistics
import threading
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from .hashing import SIMILAR_THRESHOLD, hamming, sha256_file
from .imageio import SUPPORTED_EXTS, write_bytes_no_clobber
from .recipes import Recipe

log = logging.getLogger("ips.jobs")

STATES = ("queued", "running", "done", "review", "needs_override", "failed", "cancelled", "duplicate", "skipped")


def enumerate_inputs(paths: list[str | Path], recursive: bool = True) -> list[tuple[Path, str]]:
    """Return (file, relative_dir) pairs. Directories keep their subfolder structure; order deterministic."""
    out: list[tuple[Path, str]] = []
    seen: set[str] = set()
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            it = p.rglob("*") if recursive else p.glob("*")
            for f in sorted(x for x in it if x.is_file()):
                if f.suffix.lower() in SUPPORTED_EXTS and not f.name.startswith("."):
                    rel = f.parent.relative_to(p)
                    # stored POSIX-style so the library DB, reports and sidecars are portable across OSes
                    rel_dir = (Path(p.name) / rel).as_posix() if str(rel) != "." else p.name
                    key = str(f.resolve())
                    if key not in seen:
                        seen.add(key)
                        out.append((f, rel_dir))
        elif p.is_file():
            key = str(p.resolve())
            if key not in seen:
                seen.add(key)
                out.append((p, ""))
    return out


class JobManager:
    def __init__(self, studio, on_event: Callable[[int, str, dict[str, Any]], None] | None = None):
        self.studio = studio
        self.db = studio.db
        self.on_event = on_event or (lambda *a: None)
        self._thread: threading.Thread | None = None
        self._pause = threading.Event()
        self._stop = threading.Event()
        self._cancel_current = threading.Event()
        self._prefetch = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ips-prefetch")
        self.current_job: int | None = None
        self.recover()

    # ------------------------------------------------------------------ batches
    def create_batch(self, name: str, inputs: list[str | Path], *, recursive: bool = True,
                     profiles: tuple[str, ...] = ("archive",), mode: str | None = None, scale: Any = "auto",
                     target_long_edge: int | None = None, quality_override: bool = False,
                     look: str | None = None) -> int:
        files = enumerate_inputs(inputs, recursive)
        st = self.studio.settings
        settings = {"profiles": list(profiles), "mode": mode or self.studio.settings.default_mode, "scale": scale,
                    "target_long_edge": target_long_edge, "quality_override": quality_override,
                    "inputs": [str(x) for x in inputs], "recursive": recursive,
                    "look": look or getattr(self.studio.settings, "default_look", "enhanced"),
                    # snapshot so a resumed batch uses the same prompt/strength even if Settings change later
                    "generative": {"prompt": st.qwen_prompt, "strength": st.qwen_strength, "turbo": st.qwen_turbo,
                                   "face_reference": st.qwen_face_reference, "long_edge": st.qwen_long_edge}}
        bid = self.db.insert("batches", {"name": name, "root": ";".join(str(x) for x in inputs),
                                         "output_root": self.studio.settings.output_root, "settings_json": settings,
                                         "state": "open", "created": time.time()})
        now = time.time()
        for f, rel in files:
            self.db.insert("jobs", {"batch_id": bid, "source_path": str(f), "rel_path": rel, "state": "queued",
                                    "created": now, "updated": now})
        log.info("batch %s: %d files", bid, len(files))
        return bid

    def batch_settings(self, batch_id: int) -> dict[str, Any]:
        row = self.db.one("SELECT * FROM batches WHERE id=?", (batch_id,))
        return json.loads(row["settings_json"]) if row else {}

    def jobs(self, batch_id: int, state: str | None = None) -> list[dict[str, Any]]:
        if state:
            rows = self.db.query("SELECT * FROM jobs WHERE batch_id=? AND state=? ORDER BY id", (batch_id, state))
        else:
            rows = self.db.query("SELECT * FROM jobs WHERE batch_id=? ORDER BY id", (batch_id,))
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ recovery
    def recover(self) -> int:
        cur = self.db.execute("UPDATE jobs SET state='queued', error=COALESCE(error,'') || "
                              "'[interrupted: requeued at startup] ' WHERE state='running'")
        n = cur.rowcount or 0
        if n:
            log.warning("requeued %d interrupted job(s)", n)
        return n

    def verify_completed(self, batch_id: int) -> list[int]:
        """Requeue done/review jobs whose outputs are missing or altered."""
        bad = []
        for j in self.jobs(batch_id):
            if j["state"] not in ("done", "review"):
                continue
            outs = self.db.query("SELECT path, sha256 FROM outputs WHERE job_id=?", (j["id"],))
            ok = bool(outs) and all(Path(o["path"]).is_file() and sha256_file(o["path"]) == o["sha256"]
                                    for o in outs)
            if not ok:
                bad.append(j["id"])
                self.db.update("jobs", j["id"], {"state": "queued", "error": "outputs missing/changed: requeued",
                                                 "updated": time.time()})
        return bad

    def retry(self, job_ids: list[int]) -> None:
        for jid in job_ids:
            self.db.update("jobs", jid, {"state": "queued", "updated": time.time()})

    def retry_failed(self, batch_id: int) -> int:
        cur = self.db.execute("UPDATE jobs SET state='queued' WHERE batch_id=? AND state IN ('failed','cancelled')",
                              (batch_id,))
        return cur.rowcount or 0

    def set_override(self, job_id: int, recipe: Recipe | None) -> None:
        self.db.update("jobs", job_id, {"override_json": recipe.to_json() if recipe else None,
                                        "state": "queued", "updated": time.time()})

    # ------------------------------------------------------------------ control
    def start(self, batch_id: int) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._pause.clear()
        self.db.update("batches", batch_id, {"state": "running"})
        self._thread = threading.Thread(target=self._run, args=(batch_id,), name="ips-batch", daemon=True)
        self._thread.start()

    def pause(self) -> None:
        self._pause.set()

    def resume(self) -> None:
        self._pause.clear()

    def cancel(self) -> None:
        """Stop after cancelling the current job; remaining jobs stay queued (resumable)."""
        self._stop.set()
        self._cancel_current.set()
        self._pause.clear()

    def cancel_current(self) -> None:
        self._cancel_current.set()

    def wait(self, timeout: float | None = None) -> bool:
        if self._thread:
            self._thread.join(timeout)
            return not self._thread.is_alive()
        return True

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    @property
    def paused(self) -> bool:
        return self._pause.is_set()

    # ------------------------------------------------------------------ worker
    def _next(self, batch_id: int) -> dict[str, Any] | None:
        r = self.db.one("SELECT * FROM jobs WHERE batch_id=? AND state='queued' ORDER BY id LIMIT 1", (batch_id,))
        return dict(r) if r else None

    def _peek_after(self, batch_id: int, job_id: int) -> dict[str, Any] | None:
        r = self.db.one("SELECT * FROM jobs WHERE batch_id=? AND state='queued' AND id>? ORDER BY id LIMIT 1",
                        (batch_id, job_id))
        return dict(r) if r else None

    def _inspect(self, job: dict[str, Any]):
        return self.studio.inspect(job["source_path"], rel_dir=job["rel_path"] or "")

    def _run(self, batch_id: int) -> None:
        bs = self.batch_settings(batch_id)
        brow = self.db.one("SELECT name FROM batches WHERE id=?", (batch_id,))
        bname = f"Batch/{brow['name']}" if brow else f"Batch/{batch_id}"
        req = self.studio.request(mode=bs.get("mode"), scale=bs.get("scale", "auto"),
                                  target_long_edge=bs.get("target_long_edge"),
                                  quality_override=bs.get("quality_override", False), look=bs.get("look"),
                                  generative=bs.get("generative"))
        profiles = tuple(bs.get("profiles", ["archive"]))
        prefetched: dict[int, Future] = {}
        seen_sha: dict[str, int] = {r["source_sha256"]: r["id"] for r in self.db.query(
            "SELECT id, source_sha256 FROM jobs WHERE batch_id=? AND state IN ('done','review') "
            "AND source_sha256 IS NOT NULL", (batch_id,))}
        try:
            while not self._stop.is_set():
                while self._pause.is_set() and not self._stop.is_set():
                    time.sleep(0.1)
                if self._stop.is_set():
                    break
                job = self._next(batch_id)
                if job is None:
                    break
                jid = job["id"]
                self.current_job = jid
                self._cancel_current.clear()
                t0 = time.time()
                self.db.update("jobs", jid, {"state": "running", "started": t0, "updated": t0,
                                             "attempts": (job["attempts"] or 0) + 1})
                self.on_event(jid, "running", {"path": job["source_path"]})
                try:
                    fut = prefetched.pop(jid, None)
                    t_q = time.perf_counter()
                    src = fut.result() if fut else self._inspect(job)
                    inspect_wait_ms = (time.perf_counter() - t_q) * 1e3
                    nxt = self._peek_after(batch_id, jid)
                    if nxt and nxt["id"] not in prefetched:
                        prefetched[nxt["id"]] = self._prefetch.submit(self._inspect, nxt)
                    self.db.update("jobs", jid, {"source_sha256": src.sha256, "source_id": src.source_id})
                    if src.sha256 in seen_sha and seen_sha[src.sha256] != jid:
                        self.db.update("jobs", jid, {"state": "duplicate", "updated": time.time(),
                                                     "error": f"exact duplicate of job {seen_sha[src.sha256]} "
                                                              "(not processed, not deleted)"})
                        self.on_event(jid, "duplicate", {})
                        continue
                    override = json.loads(job["override_json"]) if job["override_json"] else None
                    recipe = Recipe.from_json(override) if override else None
                    outcome = self.studio.prepare(job["source_path"], profiles=profiles, recipe=recipe, req=req,
                                                  rel_dir=job["rel_path"] or "", batch_id=batch_id, batch_name=bname,
                                                  job_id=jid, cancel=self._cancel_current.is_set, src=src)
                    timings = dict(outcome.timings_ms)
                    timings["prefetch_wait_ms"] = inspect_wait_ms
                    timings["wall_ms"] = (time.time() - t0) * 1e3
                    self.db.update("jobs", jid, {
                        "state": outcome.state, "auto_recipe_json": outcome.auto_recipe.to_json(),
                        "recipe_json": outcome.recipe.to_json(), "recipe_hash": outcome.recipe.hash,
                        "review_json": {"flags": outcome.review_flags,
                                        "identity": outcome.result.identity if outcome.result else None,
                                        "content_class": src.assessment.content_class,
                                        "issues": [i.code + ":" + i.level for i in src.assessment.issues]},
                        "timings_json": timings, "finished": time.time(), "updated": time.time(),
                        "error": None})
                    seen_sha[src.sha256] = jid
                    self.on_event(jid, outcome.state, {"timings": timings, "flags": outcome.review_flags})
                except Exception as e:  # isolation: one failure never kills the batch
                    from .pipeline import Cancelled
                    state = "cancelled" if isinstance(e, Cancelled) else "failed"
                    msg = "cancelled by user" if state == "cancelled" else f"{type(e).__name__}: {e}"
                    log.error("job %s %s: %s", jid, state, msg)
                    if state == "failed":
                        log.debug(traceback.format_exc())
                    self.db.update("jobs", jid, {"state": state, "error": msg, "updated": time.time(),
                                                 "finished": time.time()})
                    self.on_event(jid, state, {"error": msg})
            done = self.db.one("SELECT COUNT(*) n FROM jobs WHERE batch_id=? AND state='queued'", (batch_id,))["n"]
            self.db.update("batches", batch_id, {"state": "complete" if done == 0 else "paused"})
        finally:
            for f in prefetched.values():
                f.cancel()
            self.current_job = None
            self.on_event(-1, "batch_finished", {"batch_id": batch_id})

    # ------------------------------------------------------------------ similarity + report
    def similar_suggestions(self, batch_id: int) -> list[tuple[int, int, int]]:
        rows = self.db.query("SELECT j.id, s.dhash FROM jobs j JOIN sources s ON s.id=j.source_id "
                             "WHERE j.batch_id=? AND s.dhash IS NOT NULL", (batch_id,))
        out = []
        for i in range(len(rows)):
            for k in range(i + 1, len(rows)):
                d = hamming(int(rows[i]["dhash"], 16), int(rows[k]["dhash"], 16))
                if d <= SIMILAR_THRESHOLD:
                    out.append((rows[i]["id"], rows[k]["id"], d))
        return out

    def stats(self, batch_id: int) -> dict[str, Any]:
        js = self.jobs(batch_id)
        counts: dict[str, int] = {}
        for j in js:
            counts[j["state"]] = counts.get(j["state"], 0) + 1
        tot, warm, loads = [], [], []
        for j in js:
            if j["timings_json"] and j["state"] in ("done", "review"):
                t = json.loads(j["timings_json"])
                if isinstance(t.get("total_job_ms"), (int, float)):
                    tot.append(t["total_job_ms"] / 1000)
                    warm.append(t.get("warm_job_ms", t["total_job_ms"]) / 1000)
                    loads.append(t.get("model_load_total_ms", 0) / 1000)

        def pct(v, q):
            if not v:
                return None
            s = sorted(v)
            return round(s[min(len(s) - 1, int(round(q * (len(s) - 1))))], 3)
        return {"jobs": len(js), "states": counts,
                "total_s": {"median": round(statistics.median(tot), 3) if tot else None, "p95": pct(tot, 0.95),
                            "max": round(max(tot), 3) if tot else None, "first": round(tot[0], 3) if tot else None},
                "warm_s": {"median": round(statistics.median(warm), 3) if warm else None, "p95": pct(warm, 0.95),
                           "max": round(max(warm), 3) if warm else None},
                "model_load_s_total": round(sum(loads), 3), "similar_pairs": self.similar_suggestions(batch_id),
                "device": self.studio.device.name}

    def write_report(self, batch_id: int) -> tuple[Path, Path]:
        brow = self.db.one("SELECT * FROM batches WHERE id=?", (batch_id,))
        root = Path(self.studio.settings.output_root) / f"Batch/{brow['name']}" / "Reports"
        st = self.stats(batch_id)
        js = self.jobs(batch_id)
        rep = {"batch": dict(brow), "stats": st, "jobs": js}
        jp = write_bytes_no_clobber(json.dumps(rep, indent=2, default=str).encode(), root / f"batch_{batch_id}.json")
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["job", "file", "state", "content", "issues", "recipe", "total_s", "warm_s", "flags", "error"])
        for j in js:
            rv = json.loads(j["review_json"]) if j["review_json"] else {}
            t = json.loads(j["timings_json"]) if j["timings_json"] else {}
            rc = json.loads(j["recipe_json"]) if j["recipe_json"] else {}
            w.writerow([j["id"], j["source_path"], j["state"], rv.get("content_class", ""),
                        ";".join(rv.get("issues", [])), Recipe.from_json(rc).summary() if rc else "",
                        round(t.get("total_job_ms", 0) / 1000, 3) if t else "",
                        round(t.get("warm_job_ms", 0) / 1000, 3) if t else "",
                        " | ".join(rv.get("flags", [])), j["error"] or ""])
        cp = write_bytes_no_clobber(buf.getvalue().encode("utf-8"), root / f"batch_{batch_id}.csv")
        return jp, cp

    def shutdown(self) -> None:
        self.cancel()
        self.wait(10)
        self._prefetch.shutdown(wait=False, cancel_futures=True)

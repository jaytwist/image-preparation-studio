"""SQLite persistence (WAL). One connection per thread; writes are short transactions."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from ..paths import db_path

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
CREATE TABLE IF NOT EXISTS sources(
  id INTEGER PRIMARY KEY,
  sha256 TEXT NOT NULL,
  path TEXT NOT NULL,
  rel_path TEXT,
  format TEXT, width INTEGER, height INTEGER, bit_depth INTEGER, has_alpha INTEGER,
  size_bytes INTEGER, mtime REAL, dhash TEXT,
  metadata_json TEXT, summary_json TEXT, diagnostics_json TEXT, assessment_json TEXT,
  imported_at REAL,
  UNIQUE(sha256, path)
);
CREATE INDEX IF NOT EXISTS ix_sources_sha ON sources(sha256);
CREATE TABLE IF NOT EXISTS batches(
  id INTEGER PRIMARY KEY, name TEXT, root TEXT, output_root TEXT, settings_json TEXT,
  state TEXT DEFAULT 'open', created REAL
);
CREATE TABLE IF NOT EXISTS jobs(
  id INTEGER PRIMARY KEY,
  batch_id INTEGER, source_path TEXT NOT NULL, rel_path TEXT, source_id INTEGER,
  source_sha256 TEXT,
  state TEXT NOT NULL DEFAULT 'queued',
  auto_recipe_json TEXT, recipe_json TEXT, recipe_hash TEXT, override_json TEXT,
  profile_json TEXT, error TEXT, review_json TEXT, timings_json TEXT,
  attempts INTEGER DEFAULT 0, created REAL, updated REAL, started REAL, finished REAL
);
CREATE INDEX IF NOT EXISTS ix_jobs_batch ON jobs(batch_id, state);
CREATE TABLE IF NOT EXISTS outputs(
  id INTEGER PRIMARY KEY,
  job_id INTEGER, source_id INTEGER, batch_id INTEGER,
  kind TEXT NOT NULL, profile TEXT, path TEXT NOT NULL, sha256 TEXT,
  format TEXT, width INTEGER, height INTEGER, bit_depth INTEGER,
  recipe_hash TEXT, recipe_summary TEXT, models TEXT,
  approval TEXT DEFAULT 'pending', review_flags_json TEXT, audit_json TEXT,
  elapsed_ms REAL, created REAL, parent_output_id INTEGER, label TEXT
);
CREATE INDEX IF NOT EXISTS ix_outputs_source ON outputs(source_id);
CREATE TABLE IF NOT EXISTS intermediates(
  key TEXT PRIMARY KEY, output_id INTEGER, path TEXT, created REAL
);
CREATE TABLE IF NOT EXISTS corrections(
  id INTEGER PRIMARY KEY, source_id INTEGER, source_sha256 TEXT, content_class TEXT,
  auto_recipe_json TEXT, final_recipe_json TEXT, choice TEXT, features_json TEXT, created REAL
);
CREATE TABLE IF NOT EXISTS presets(
  name TEXT PRIMARY KEY, recipe_json TEXT, created REAL
);
CREATE TABLE IF NOT EXISTS history(
  id INTEGER PRIMARY KEY, source_id INTEGER, label TEXT, recipe_json TEXT, output_id INTEGER, created REAL
);
"""


class DB:
    def __init__(self, path: Path | None = None):
        self.path = Path(path or db_path())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._write_lock = threading.RLock()
        with self.conn() as c:
            c.executescript(SCHEMA)

    def conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "c", None)
        if c is None:
            c = sqlite3.connect(str(self.path), timeout=30, check_same_thread=False)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA busy_timeout=30000")
            self._local.c = c
        return c

    def execute(self, sql: str, args: Iterable[Any] = ()) -> sqlite3.Cursor:
        with self._write_lock:
            c = self.conn()
            cur = c.execute(sql, tuple(args))
            c.commit()
            return cur

    def query(self, sql: str, args: Iterable[Any] = ()) -> list[sqlite3.Row]:
        return list(self.conn().execute(sql, tuple(args)).fetchall())

    def one(self, sql: str, args: Iterable[Any] = ()) -> sqlite3.Row | None:
        return self.conn().execute(sql, tuple(args)).fetchone()

    # ------------------------------------------------------------------ helpers
    def insert(self, table: str, row: dict[str, Any]) -> int:
        row = {k: (json.dumps(v, default=str) if isinstance(v, (dict, list)) else v) for k, v in row.items()}
        cols = ",".join(row)
        q = ",".join("?" for _ in row)
        return int(self.execute(f"INSERT INTO {table}({cols}) VALUES({q})", row.values()).lastrowid)

    def update(self, table: str, id_: int, row: dict[str, Any]) -> None:
        row = {k: (json.dumps(v, default=str) if isinstance(v, (dict, list)) else v) for k, v in row.items()}
        sets = ",".join(f"{k}=?" for k in row)
        self.execute(f"UPDATE {table} SET {sets} WHERE id=?", list(row.values()) + [id_])

    def upsert_source(self, row: dict[str, Any]) -> int:
        ex = self.one("SELECT id FROM sources WHERE sha256=? AND path=?", (row["sha256"], row["path"]))
        if ex:
            self.update("sources", ex["id"], row)
            return int(ex["id"])
        row.setdefault("imported_at", time.time())
        return self.insert("sources", row)

    def close(self) -> None:
        c = getattr(self._local, "c", None)
        if c is not None:
            c.close()
            self._local.c = None

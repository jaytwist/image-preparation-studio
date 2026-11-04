"""User-directed, checksum-verified model download (setup step only; the app never downloads at runtime).

    python tools/fetch_models.py --list
    python tools/fetch_models.py --all            # everything in the registry
    python tools/fetch_models.py fbcnn_color scunet_real_psnr swinir_x2
    python tools/fetch_models.py --verify         # re-hash installed files

Files go to the Studio models folder (IPS_MODELS_DIR or ./models). Existing files with a matching
SHA-256 are skipped; a mismatching existing file is never overwritten (download goes to *.new).
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ips.backends.registry import MODELS  # noqa: E402
from ips.paths import models_dir  # noqa: E402


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        got = 0
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
            got += len(b)
            if total:
                print(f"\r  {got / 2**20:7.1f} / {total / 2**20:7.1f} MB", end="", flush=True)
    print()
    tmp.rename(dest)


def fetch(mid: str) -> bool:
    m = MODELS[mid]
    root = models_dir()
    items = [(m.file, m.sha256, m.url)] + [x for x in m.extra_files]
    ok = True
    if mid == "buffalo_l":
        if all((root / f).is_file() and sha256(root / f) == s for f, s, _ in items):
            print(f"[ok] {mid}")
            return True
        z = root / "buffalo_l.zip"
        if not z.is_file():
            download(m.url, z)
        with zipfile.ZipFile(z) as zf:
            for name in ("det_10g.onnx", "w600k_r50.onnx"):
                target = root / "buffalo_l" / name
                if not target.is_file():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(zf.read(name))
        items_ok = all(sha256(root / f) == s for f, s, _ in items)
        print(f"[{'ok' if items_ok else 'CHECKSUM MISMATCH'}] {mid}")
        return items_ok
    for f, s, url in items:
        p = root / f
        if p.is_file():
            if sha256(p) == s:
                print(f"[ok] {f}")
                continue
            print(f"[warn] {f} exists with a different checksum; not overwritten. Downloading to {f}.new")
            p = p.with_name(p.name + ".new")
        if not url:
            print(f"[skip] {f}: no URL")
            ok = False
            continue
        download(url, p)
        got = sha256(p)
        if got != s:
            print(f"[FAIL] {f}: sha256 {got} != expected {s}")
            ok = False
        else:
            print(f"[ok] {f}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.list or not (a.models or a.all or a.verify):
        for m in MODELS.values():
            print(f"{m.id:24s} {m.status:9s} {m.character:10s} {'installed' if m.installed else 'missing':9s} "
                  f"{m.license}")
        return 0
    if a.verify:
        bad = 0
        for m in MODELS.values():
            if m.installed:
                good = sha256(m.path) == m.sha256
                bad += not good
                print(f"{'ok ' if good else 'BAD'} {m.file}")
        return 1 if bad else 0
    ids = list(MODELS) if a.all else a.models
    return 0 if all(fetch(i) for i in ids) else 1


if __name__ == "__main__":
    sys.exit(main())

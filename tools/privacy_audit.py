"""Offline/privacy audit: run a representative workload and record every network connection the
Studio process opens (native libraries included, via psutil sampling). Loopback connections to the
configured ComfyUI port are the only allowed ones.

    python tools/privacy_audit.py [--seconds 60] [--image path]
Exit code 0 = no disallowed connections observed.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psutil  # noqa: E402

import ips  # noqa: E402,F401  (enforces privacy env first)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--image", default=str(Path(__file__).resolve().parent.parent / "fixtures/corpus/old_phone/group_q28.jpg"))
    a = ap.parse_args()
    from ips.settings import Settings
    comfy_port = int(Settings.load().comfy_url.rsplit(":", 1)[-1].strip("/") or 8188)
    me = psutil.Process()
    seen: dict[tuple, str] = {}
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            try:
                for c in me.net_connections(kind="inet"):
                    if c.raddr:
                        seen.setdefault((c.raddr.ip, c.raddr.port), c.status)
            except Exception:
                pass
            time.sleep(0.02)
    t = threading.Thread(target=sample, daemon=True)
    t.start()
    from ips.core.engine import Studio
    st = Studio()
    if Path(a.image).is_file():
        st.prepare(a.image, profiles=("archive", "share"), req=st.request(scale=1))
        from ips.quickedit.matting import Matting
        Matting.person_mask(st.inspect(a.image, record=False).image.rgb)
    t_end = time.time() + a.seconds
    while time.time() < t_end:
        time.sleep(0.5)
    stop.set()
    bad = [f"{ip}:{port}" for (ip, port) in seen
           if not (ip in ("127.0.0.1", "::1") and port == comfy_port)]
    rep = {"connections": [f"{ip}:{port}" for ip, port in seen], "disallowed": bad,
           "env": {k: __import__("os").environ.get(k) for k in ("ORT_DISABLE_TELEMETRY", "HF_HUB_OFFLINE",
                                                                  "DO_NOT_TRACK")}}
    print(json.dumps(rep, indent=2))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

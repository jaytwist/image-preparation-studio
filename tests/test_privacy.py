"""No network traffic: trace every connect() syscall of a full workload (Linux/strace)."""
import os
import shutil
import subprocess
import sys
import textwrap

import pytest

from tests.conftest import ROOT


def test_privacy_env_enforced():
    import ips  # noqa: F401
    assert os.environ["ORT_DISABLE_TELEMETRY"] == "1"
    assert os.environ["HF_HUB_OFFLINE"] == "1"


@pytest.mark.skipif(not shutil.which("strace") or sys.platform != "linux", reason="needs strace (Linux)")
def test_no_outbound_connections_during_workload(tmp_path):
    script = textwrap.dedent(f"""
        import time, numpy as np
        from ips.core.engine import Studio
        from ips.settings import Settings
        s = Settings(); s.device = "cpu"; s.output_root = r"{tmp_path / 'o'}"
        st = Studio(s)
        st.prepare(r"{ROOT / 'fixtures/corpus/old_phone/group_q28.jpg'}", profiles=("archive", "share"),
                   req=st.request(scale=1))
        from ips.quickedit.matting import Matting
        Matting.person_mask(np.zeros((64, 64, 3), np.float32))
        time.sleep(25)   # onnxruntime telemetry fired within ~10 s when not disabled
    """)
    trace = tmp_path / "net.trace"
    env = dict(os.environ, IPS_DATA_DIR=str(tmp_path / "d"), PYTHONPATH=str(ROOT))
    subprocess.run(["strace", "-f", "-qq", "-e", "trace=connect", "-o", str(trace), sys.executable, "-c", script],
                   cwd=ROOT, env=env, check=True, timeout=900)
    inet = [l for l in trace.read_text().splitlines() if "AF_INET" in l]
    assert not inet, "outbound network connections detected:\n" + "\n".join(inet[:10])

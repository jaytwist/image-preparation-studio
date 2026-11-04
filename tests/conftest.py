import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
CORPUS = ROOT / "fixtures" / "corpus"


@pytest.fixture(scope="session", autouse=True)
def _isolated_env(tmp_path_factory):
    d = tmp_path_factory.mktemp("ipsdata")
    os.environ["IPS_DATA_DIR"] = str(d / "data")
    os.environ["IPS_OUTPUT_ROOT"] = str(d / "outputs")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    if not (CORPUS / "manifest.json").is_file():
        from ips.eval.fixtures import build
        build(CORPUS)
    yield


class CorpusDir:
    """``corpus / "rel/path"`` returns the Path, or skips the test if that optional asset is absent
    (public test images are fetched with ``python -m ips.eval.fixtures --fetch``)."""

    def __init__(self, root: Path):
        self.root = root

    def __truediv__(self, rel: str) -> Path:
        p = self.root / rel
        if not p.exists():
            pytest.skip(f"optional test asset missing: {rel} (run: python -m ips.eval.fixtures --fetch)")
        return p

    def __fspath__(self) -> str:
        return str(self.root)


@pytest.fixture(scope="session")
def corpus():
    return CorpusDir(CORPUS)


@pytest.fixture(scope="session")
def manifest():
    return {i["id"]: i for i in json.loads((CORPUS / "manifest.json").read_text())["items"]}


@pytest.fixture()
def studio(tmp_path):
    from ips.core.db import DB
    from ips.core.engine import Studio
    from ips.settings import Settings
    s = Settings()
    s.output_root = str(tmp_path / "out")
    s.device = "cpu"
    return Studio(s, DB(tmp_path / "t.sqlite"))


@pytest.fixture()
def studio_nomodels(tmp_path):
    from ips.core.db import DB
    from ips.core.engine import Studio
    from ips.settings import Settings
    s = Settings()
    s.output_root = str(tmp_path / "out")
    return Studio(s, DB(tmp_path / "t.sqlite"), load_models=False)


def models_installed(*ids):
    from ips.backends.registry import MODELS
    return all(MODELS[i].installed for i in ids)

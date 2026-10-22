import json
from pathlib import Path

from app.datasets.synthetic_fixture import build_synthetic_xbd


def _snapshot(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def test_fixture_is_deterministic(tmp_path: Path) -> None:
    a = _snapshot(build_synthetic_xbd(tmp_path / "a"))
    b = _snapshot(build_synthetic_xbd(tmp_path / "b"))
    assert a == b


def test_fixture_is_clearly_marked_synthetic(synthetic_root: Path) -> None:
    assert "not xBD data" in (synthetic_root / "README.md").read_text()
    for label in synthetic_root.rglob("*.json"):
        meta = json.loads(label.read_text())["metadata"]
        assert meta["synthetic"] is True
        assert meta["disaster"].startswith("synthetic-")

from pathlib import Path

import pytest

from app.core.config import REPO_ROOT
from app.core.events_config import ConfigError, load_events_config


def test_real_and_fixture_configs_load() -> None:
    real = load_events_config(REPO_ROOT / "config" / "events.yaml")
    assert real.source == "xbd"
    assert len(real.events) >= 3
    fixture = load_events_config(REPO_ROOT / "config" / "events.fixture.yaml")
    assert fixture.source == "synthetic"
    assert all(e.startswith("synthetic-") for e in fixture.events)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_events_config(tmp_path / "nope.yaml")


@pytest.mark.parametrize(
    "body, message",
    [
        ("events: [", "not valid YAML"),
        ("- a\n- b\n", "mapping"),
        ("source: xbd\ndataset_root: x\nxbd_splits: [train]\nevents: [Bad_Name]\ninspection: {output_dir: o, figures_dir: f}\n", "lowercase"),
        ("source: xbd\ndataset_root: x\nxbd_splits: [train]\nevents: [a]\nexcluded_labels: [destroyed]\ninspection: {output_dir: o, figures_dir: f}\n", "damage classes"),
        ("source: real\ndataset_root: x\nxbd_splits: [train]\nevents: [a]\ninspection: {output_dir: o, figures_dir: f}\n", "source"),
    ],
)
def test_invalid_configs_raise_readable_errors(tmp_path: Path, body: str, message: str) -> None:
    path = tmp_path / "events.yaml"
    path.write_text(body)
    with pytest.raises(ConfigError, match=message):
        load_events_config(path)

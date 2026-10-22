from pathlib import Path

import pytest

from app.core.config import REPO_ROOT
from app.core.events_config import EventsConfig, load_events_config
from app.datasets.synthetic_fixture import build_synthetic_xbd

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A fresh synthetic xBD-layout dataset (NOT xBD data), generated once per test session."""
    return build_synthetic_xbd(tmp_path_factory.mktemp("xbd_synthetic"))


@pytest.fixture
def fixture_config(synthetic_root: Path, tmp_path: Path) -> EventsConfig:
    """config/events.fixture.yaml, pointed at the generated fixture and a temp output folder."""
    config = load_events_config(REPO_ROOT / "config" / "events.fixture.yaml")
    inspection = config.inspection.model_copy(update={"output_dir": tmp_path / "inspection", "figures_dir": tmp_path / "figures"})
    return config.model_copy(update={"dataset_root": synthetic_root, "inspection": inspection})


@pytest.fixture(scope="session")
def synthetic_parquet_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Two small Parquet shards in the Hugging Face xBD-mirror schema (NOT xBD data)."""
    from app.datasets.synthetic_fixture import build_synthetic_parquet

    return build_synthetic_parquet(tmp_path_factory.mktemp("xbd_synthetic_parquet"))


@pytest.fixture
def parquet_config(synthetic_parquet_root: Path, tmp_path: Path) -> EventsConfig:
    """config/events.fixture_parquet.yaml, pointed at the generated shards and temp output/cache folders."""
    config = load_events_config(REPO_ROOT / "config" / "events.fixture_parquet.yaml")
    inspection = config.inspection.model_copy(update={"output_dir": tmp_path / "inspection", "figures_dir": tmp_path / "figures"})
    cache = config.cache.model_copy(update={"dir": tmp_path / "cache"})
    return config.model_copy(update={"dataset_root": synthetic_parquet_root, "inspection": inspection, "cache": cache})

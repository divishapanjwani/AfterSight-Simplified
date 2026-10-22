"""Feature formulas on synthetic images with known answers, plus the table build/load contract."""

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.core.config import REPO_ROOT
from app.datasets.inspection import run_inspection
from app.features.config import FeatureConfig, load_feature_config
from app.features.extract import FeatureError, building_features, compactness, glcm_contrast_homogeneity, scene_rows
from app.features.schema import FEATURE_COLUMNS, ID_COLUMNS, LABEL_COLUMN, NON_FEATURE_COLUMNS, QUALITY_COLUMNS
from app.features.table import FeatureTableError, build_feature_table, load_feature_table, table_paths

CFG = FeatureConfig(scene_cache_dir=Path("unused"), output_dir=Path("unused"))


def _square_scene(pre_value: int = 100, post_value: int = 150, size: int = 40):
    """Uniform grey background; a 10x10 building that brightens from pre_value to post_value."""
    pre = np.full((size, size, 3), 100, np.uint8)
    post = pre.copy()
    mask = np.zeros((size, size), bool)
    mask[15:25, 15:25] = True
    pre[mask] = pre_value
    post[mask] = post_value
    ring = np.zeros_like(mask)
    ring[10:30, 10:30] = True
    ring &= ~mask
    return pre, post, mask, ring


def test_uniform_building_known_answers() -> None:
    pre, post, mask, ring = _square_scene()
    f = building_features(pre, post, mask, ring, CFG)
    assert f["pre_std_r"] == 0 and f["post_std_g"] == 0 and f["gray_std_pre"] == 0
    assert f["abs_dmean_r"] == pytest.approx(50) and f["dstd_b"] == 0
    assert f["gray_absdiff_mean"] == pytest.approx(50, abs=1e-3)
    assert f["brightness_ratio"] == pytest.approx(1.5, abs=1e-4)
    assert f["edge_density_pre"] == 0  # nothing to detect in a uniform image
    assert f["glcm_contrast_pre"] == 0 and f["glcm_homogeneity_pre"] == 1
    assert f["ring_dmean"] == pytest.approx(0) and f["ring_absdiff_mean"] == pytest.approx(0)
    assert f["contrast_change"] == pytest.approx(50, abs=1e-3)  # building got 50 brighter than its ring


def test_changed_building_gains_edges_inside_footprint() -> None:
    pre, post, mask, ring = _square_scene()
    post = post.copy()
    blocks = (np.indices((10, 10)) // 3).sum(axis=0) % 2 == 1  # 3 px checker blocks = "rubble"
    post[15:25, 15:25][blocks] = 40
    f = building_features(pre, post, mask, ring, CFG)
    assert f["edge_density_post"] > 0 and f["edge_density_delta"] > 0
    assert f["glcm_contrast_delta"] > 0 and f["glcm_homogeneity_delta"] < 0


def test_glcm_checkerboard() -> None:
    board = (np.indices((8, 8)).sum(axis=0) % 2 * 255).astype(np.float32)
    contrast, homogeneity = glcm_contrast_homogeneity(board, np.ones((8, 8), bool), levels=32, distance=1)
    assert contrast == pytest.approx(31**2)  # every neighbour pair is 0 vs 31
    assert homogeneity == pytest.approx(1 / (1 + 31**2))


def test_glcm_ignores_pixels_outside_mask_and_handles_no_pairs() -> None:
    image = np.zeros((6, 6), np.float32)
    image[:, 3:] = 255
    mask = np.zeros((6, 6), bool)
    mask[:, :3] = True  # only the uniform half
    assert glcm_contrast_homogeneity(image, mask, 32, 1) == (0.0, 1.0)
    single = np.zeros((6, 6), bool)
    single[2, 2] = True
    assert all(math.isnan(v) for v in glcm_contrast_homogeneity(image, single, 32, 1))


def test_compactness() -> None:
    assert compactness("POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))") == pytest.approx(math.pi / 4)
    assert compactness("POLYGON ((0 0, 100 0, 100 1, 0 1, 0 0))") < 0.05


def test_empty_ring_gives_missing_context() -> None:
    pre, post, mask, _ring = _square_scene()
    f = building_features(pre, post, mask, np.zeros_like(mask), CFG)
    assert math.isnan(f["ring_dmean"]) and math.isnan(f["contrast_change"])


def test_scene_rows_on_cached_fixture(parquet_config) -> None:
    run_inspection(parquet_config, write_cache=True)
    folder = parquet_config.cache.dir / "synthetic-flood_00000000"
    doc = json.loads((folder / "scene.json").read_text())
    rows = scene_rows(folder, doc, CFG, 4, "flooding")
    assert [r["sample_id"] for r in rows] == [b["building_id"] for b in doc["buildings"]]
    assert [r["label"] for r in rows] == [b["damage"] for b in doc["buildings"]]
    assert all(r["disaster_type"] == "flooding" for r in rows)
    assert rows[1]["area_px"] == 100  # 10x10 blob b0002
    # the ring of b0002 never includes another building's pixels
    assert all(r["ring_px"] > 0 for r in rows)

    doc["buildings"][0]["area_px"] = 999  # cache tampered / out of sync
    with pytest.raises(FeatureError, match="scene.json says 999"):
        scene_rows(folder, doc, CFG, 4, "flooding")


@pytest.fixture
def built(parquet_config, tmp_path):
    run_inspection(parquet_config, write_cache=True)
    cfg = FeatureConfig(scene_cache_dir=parquet_config.cache.dir, output_dir=tmp_path / "features")
    return cfg, parquet_config


def test_table_keeps_ids_and_label_out_of_X(built) -> None:
    cfg, events_cfg = built
    meta = build_feature_table(cfg, events_cfg)
    assert meta["reused"] is False and meta["n_buildings"] == 6
    table = load_feature_table(table_paths(cfg)["table"])
    assert list(table.X.columns) == FEATURE_COLUMNS
    assert not set(table.X.columns) & set(NON_FEATURE_COLUMNS)
    assert "event" not in table.X.columns and "label" not in table.X.columns
    assert list(table.ids.columns) == ID_COLUMNS and table.y.name == LABEL_COLUMN
    assert list(table.quality.columns) == QUALITY_COLUMNS
    assert table.y.value_counts().to_dict() == {"no-damage": 2, "destroyed": 2, "minor-damage": 1, "major-damage": 1}


def test_table_is_cached_and_rebuilt_when_settings_change(built) -> None:
    cfg, events_cfg = built
    first = build_feature_table(cfg, events_cfg)
    assert build_feature_table(cfg, events_cfg)["reused"] is True
    assert build_feature_table(cfg, events_cfg, force=True)["reused"] is False
    changed = cfg.model_copy(update={"ring_width_px": 4})
    assert build_feature_table(changed, events_cfg)["fingerprint"] != first["fingerprint"]


def test_parallel_extraction_matches_serial(built, tmp_path) -> None:
    cfg, events_cfg = built
    build_feature_table(cfg, events_cfg)
    serial = pd.read_parquet(table_paths(cfg)["table"])
    parallel_cfg = cfg.model_copy(update={"output_dir": tmp_path / "parallel"})
    build_feature_table(parallel_cfg, events_cfg, workers=2)
    pd.testing.assert_frame_equal(serial, pd.read_parquet(table_paths(parallel_cfg)["table"]))


def test_load_refuses_other_schema_and_missing_cache(built, tmp_path) -> None:
    cfg, events_cfg = built
    with pytest.raises(FeatureTableError, match="not found"):
        load_feature_table(tmp_path / "absent.parquet")
    bare = tmp_path / "bare.parquet"
    pd.DataFrame({"a": [1]}).to_parquet(bare)
    with pytest.raises(FeatureTableError, match="no AfterSight metadata"):
        load_feature_table(bare)
    with pytest.raises(FeatureTableError, match="Scene cache not found"):
        build_feature_table(cfg.model_copy(update={"scene_cache_dir": tmp_path / "nocache"}), events_cfg)


def test_real_features_config_loads() -> None:
    cfg = load_feature_config()
    assert cfg.ring_width_px > 0


def test_every_feature_is_documented() -> None:
    doc = (REPO_ROOT / "docs" / "FEATURES.md").read_text(encoding="utf-8")
    documented = set(re.findall(r"^\| `([a-z0-9_]+)` \|", doc, flags=re.M))
    assert documented == set(FEATURE_COLUMNS)

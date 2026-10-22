"""Phase 3b feature variants and experiment runner (synthetic data, NOT xBD)."""

import json

import numpy as np
import pandas as pd
import pytest

from app.features.schema import FEATURE_COLUMNS, SHAPE
from app.features.variants import CHANGE_ONLY, SUFFIX, VARIANTS, scene_normalise, variant_frame
from app.experiments.phase3b import run_phase3b
from app.training.persistence import ModelLoadError, load_model
from app.training.splits import build_manifest, split_of_buildings
from test_models import cfg, table  # noqa: F401  (shared synthetic fixtures)

ABSOLUTE_LEVELS = [c for c in FEATURE_COLUMNS if c.startswith(("pre_", "post_")) or c.endswith(("_pre", "_post"))]


def test_change_only_drops_every_absolute_level() -> None:
    assert len(CHANGE_ONLY) == 17
    assert not set(CHANGE_ONLY) & set(ABSOLUTE_LEVELS)
    assert set(SHAPE) <= set(CHANGE_ONLY) and "contrast_change" in CHANGE_ONLY


def test_scene_normalisation_is_per_scene_and_label_free() -> None:
    X = pd.DataFrame(np.arange(6 * 37, dtype=float).reshape(6, 37), columns=FEATURE_COLUMNS)
    scenes = pd.Series(["s1", "s1", "s1", "s2", "s2", "lonely"])
    out = scene_normalise(X, scenes, FEATURE_COLUMNS)
    col = f"gray_absdiff_mean{SUFFIX}"
    assert out.loc[[0, 1, 2], col].tolist() == [-37.0, 0.0, 37.0]  # minus the s1 median
    assert out.loc[[3, 4], col].tolist() == [-18.5, 18.5]
    assert (out.loc[5, [c for c in out.columns if c.endswith(SUFFIX)]] == 0).all()  # single-building scene
    assert out["area_px"].equals(X["area_px"]) and "area_px" + SUFFIX not in out.columns
    # shuffling rows of OTHER scenes cannot change a scene's values
    shuffled = scene_normalise(X.iloc[[3, 4, 5, 0, 1, 2]], scenes.iloc[[3, 4, 5, 0, 1, 2]], FEATURE_COLUMNS)
    pd.testing.assert_frame_equal(out, shuffled.loc[out.index])


def test_variant_frames_have_expected_columns() -> None:
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(10, 37)), columns=FEATURE_COLUMNS)
    scenes = pd.Series(["a"] * 5 + ["b"] * 5)
    assert list(variant_frame("change_only", X, scenes).columns) == CHANGE_ONLY
    assert variant_frame("scene_normalised", X, scenes).shape[1] == 37
    combo = variant_frame("change_only_scene_normalised", X, scenes)
    assert combo.shape[1] == 17 and set(SHAPE) <= set(combo.columns)
    with pytest.raises(ValueError, match="unknown feature variant"):
        variant_frame("nope", X, scenes)


def test_runner_trains_separately_and_reloads(table, cfg, tmp_path) -> None:
    within = build_manifest("within_event", cfg.splits.designs["within_event"], table.ids, table.y, 42, 10, table.meta["fingerprint"])
    cross_design = cfg.splits.designs["cross_event"].model_copy(update={"holdout_event": "ev-c"})
    cross = build_manifest("cross_event", cross_design, table.ids, table.y, 42, 10, table.meta["fingerprint"])
    cfg = cfg.model_copy(update={"splits": cfg.splits.model_copy(update={"designs": {
        "within_event": cfg.splits.designs["within_event"], "cross_event": cross_design}})})

    summary = run_phase3b(cfg, table, {"within_event": within, "cross_event": cross}, ["change_only"],
                          tmp_path / "exp", tmp_path / "fig", say=lambda _m: None)
    assert len(summary["rows"]) == 4  # 2 designs x (LR, DT)
    assert len(summary["test_evaluations"]) == 5  # 4 new models + 1 confounding check, each exactly once
    assert all(k.startswith("phase3b/change_only/") for k in summary["test_evaluations"])
    assert summary["rows"][0]["n_features"] == 17 and summary["rows"][0]["event_prediction_accuracy_within_event"] is not None
    assert json.loads((tmp_path / "exp" / "summary.json").read_text())["label"].startswith("Phase 3b experiment")

    folder = tmp_path / "exp" / "change_only"
    model_path = folder / "models" / "within_event" / "four_class" / "logistic_regression.joblib"
    with pytest.raises(ModelLoadError):
        load_model(model_path)  # refuses: these are not the full v1 feature columns
    pipe, meta = load_model(model_path, feature_columns=CHANGE_ONLY)
    assert meta["feature_set"] == "phase3b:change_only"
    saved = pd.read_parquet(folder / "metrics" / "within_event" / "four_class" / "logistic_regression_test_predictions.parquet")
    test_rows = (split_of_buildings(within, table.ids) == "test").to_numpy()
    X_test = variant_frame("change_only", table.X, table.ids["scene_id"])[test_rows].reset_index(drop=True)
    assert (pipe.predict(X_test) == saved["pred"].to_numpy()).all()

"""Phase 3 on a small synthetic feature table (NOT xBD data): training, guard, persistence, explanations."""

import json

import numpy as np
import pandas as pd
import pytest

from app.core.damage_scale import DAMAGE_CLASSES
from app.evaluation.analysis import run_confounding, run_kmeans
from app.evaluation.metrics import TestSetGuard, classification_metrics
from app.explain.explain import decision_path, knn_neighbours, lr_contributions, raw_threshold
from app.features.schema import FEATURE_COLUMNS, ID_COLUMNS, QUALITY_COLUMNS
from app.features.table import FeatureTable
from app.training.config import ModelSettings, SplitDesign, TrainingConfig, load_training_config
from app.training.models import SkewLog1p, ThresholdRule, make_pipeline, tune_threshold_rule
from app.training.persistence import ModelLoadError, load_model
from app.training.run import OutputRoots, design_data, run_design
from app.training.splits import build_manifest

ORDINAL = {c: i for i, c in enumerate(DAMAGE_CLASSES)}


@pytest.fixture(scope="module")
def table() -> FeatureTable:
    """3 events x 30 scenes x 20 buildings. gray_absdiff_mean and edge_density_delta carry the label."""
    rng = np.random.default_rng(1)
    rows, feats = [], []
    for e in ("ev-a", "ev-b", "ev-c"):
        for s in range(30):
            for b in range(20):
                label = rng.choice(DAMAGE_CLASSES, p=[0.55, 0.2, 0.15, 0.1])
                x = rng.normal(0, 1, len(FEATURE_COLUMNS))
                x[FEATURE_COLUMNS.index("gray_absdiff_mean")] = 10 + 12 * ORDINAL[label] + rng.normal(0, 4)
                x[FEATURE_COLUMNS.index("edge_density_delta")] = 0.05 * ORDINAL[label] + rng.normal(0, 0.03)
                x[FEATURE_COLUMNS.index("area_px")] = rng.lognormal(6.5, 1.2)  # heavily skewed, non-negative
                x[FEATURE_COLUMNS.index("pre_mean_r")] += {"ev-a": 0, "ev-b": 3, "ev-c": -3}[e]  # event signal
                rows.append({"sample_id": f"{e}_{s:08d}_b{b:04d}", "building_uid": f"{e}_{s:08d}_b{b:04d}",
                             "scene_id": f"{e}_{s:08d}", "event": e, "disaster_type": "synthetic", "xbd_split": "train",
                             "label": label})
                feats.append(x)
    ids = pd.DataFrame(rows)
    X = pd.DataFrame(np.array(feats), columns=FEATURE_COLUMNS)
    quality = pd.DataFrame({c: 0 for c in QUALITY_COLUMNS}, index=ids.index)
    return FeatureTable(ids=ids[ID_COLUMNS], X=X, y=ids["label"], quality=quality,
                        meta={"fingerprint": {"feature_schema_version": 1}})


@pytest.fixture(scope="module")
def cfg() -> TrainingConfig:
    real = load_training_config()
    small = {"baseline": ModelSettings(quantile_step=0.1),
             "logistic_regression": ModelSettings(grid={"C": [0.1, 1.0]}),
             "knn": ModelSettings(grid={"n_neighbors": [5, 15], "weights": ["distance"]}),
             "decision_tree": ModelSettings(grid={"max_depth": [3, 5], "min_samples_leaf": [10]}),
             "random_forest": ModelSettings(grid={"max_depth": [6]}, n_estimators=20, optional=True)}
    return real.model_copy(update={"models": small, "explain": real.explain.model_copy(update={"permutation_sample": 300, "permutation_repeats": 2}),
                                   "kmeans": real.kmeans.model_copy(update={"k_values": [2, 3, 4], "silhouette_sample": 500, "pca_sample": 200})})


def _data(table: FeatureTable, cfg: TrainingConfig, method: str, scheme: str = "four_class"):
    design = (SplitDesign(label="within-event generalisation", description="d", method="stratified_group",
                          fractions={"train": 0.7, "val": 0.15, "test": 0.15}) if method == "within" else
              SplitDesign(label="cross-event generalisation", description="d", method="event_holdout",
                          holdout_event="ev-c", fractions={"train": 0.85, "val": 0.15}))
    name = "within_event" if method == "within" else "cross_event"
    manifest = build_manifest(name, design, table.ids, table.y, 42, 10, table.meta["fingerprint"])
    return design_data(table, manifest, scheme, cfg.label_schemes[scheme])


@pytest.fixture(scope="module")
def trained(table, cfg, tmp_path_factory):
    roots = OutputRoots(*(tmp_path_factory.mktemp(n) for n in ("models", "metrics", "analyses", "figures")))
    guard = TestSetGuard()
    data = _data(table, cfg, "within")
    rows = run_design(data, cfg, guard, roots, say=lambda _m: None)
    return data, roots, guard, rows


def test_all_models_train_and_beat_chance(trained) -> None:
    data, _roots, _guard, rows = trained
    names = [r["model"] for r in rows]
    # Logistic Regression is not a damage model: training (scripts/train.py -> run_design) no longer includes it
    assert names == ["baseline", "knn", "decision_tree", "random_forest"]
    assert next(r for r in rows if r["model"] == "random_forest")["optional"] is True
    for r in rows:
        assert r["test_macro_f1"] > 0.4, r  # synthetic signal is strong
    baseline = next(r for r in rows if r["model"] == "baseline")
    assert baseline["hyperparameters"]["feature"] in ("gray_absdiff_mean", "edge_density_delta")


def test_test_set_used_exactly_once_per_model(trained) -> None:
    _data_, _roots, guard, _rows = trained
    assert all(n == 1 for n in guard.used.values()) and len(guard.used) == 4
    with pytest.raises(RuntimeError, match="already evaluated"):
        guard.evaluate("within_event/four_class/knn", lambda: None)


def test_saved_model_reloads_with_identical_predictions(trained) -> None:
    data, roots, _guard, _rows = trained
    folder = roots.models / "within_event" / "four_class"
    for name in ("baseline", "knn", "decision_tree", "random_forest"):
        pipe, meta = load_model(folder / f"{name}.joblib", expected_manifest_hash=data.manifest_hash)
        saved = pd.read_parquet(roots.metrics / "within_event" / "four_class" / f"{name}_test_predictions.parquet")
        assert (pipe.predict(data.X["test"]) == saved["pred"].to_numpy()).all()
        assert meta["feature_schema_version"] == 1 and meta["test_evaluations"] == 1
        assert meta["metrics"]["test"]["macro_f1"] is not None and "hyperparameters" in meta


def test_load_refuses_incompatible_models(trained, tmp_path) -> None:
    data, roots, _g, _r = trained
    folder = roots.models / "within_event" / "four_class"
    with pytest.raises(ModelLoadError, match="different split manifest"):
        load_model(folder / "knn.joblib", expected_manifest_hash="other")
    meta = json.loads((folder / "knn.json").read_text())
    (tmp_path / "knn.joblib").write_bytes((folder / "knn.joblib").read_bytes())
    (tmp_path / "knn.json").write_text(json.dumps({**meta, "feature_schema_version": 99}))
    with pytest.raises(ModelLoadError, match="schema v99"):
        load_model(tmp_path / "knn.joblib")
    with pytest.raises(ModelLoadError, match="not trained yet"):
        load_model(tmp_path / "absent.joblib")


def test_artifacts_written(trained) -> None:
    data, roots, _g, _r = trained
    models = roots.models / "within_event" / "four_class"
    assert (models / "decision_tree_structure.json").is_file() and (models / "knn_train_index.parquet").is_file()
    comparison = json.loads((roots.metrics / "within_event" / "four_class" / "comparison.json").read_text())
    assert comparison["primary_metric"] == "test macro-F1" and len(comparison["models"]) == 4
    assert not (models / "logistic_regression.joblib").exists()
    for name in ("knn", "decision_tree"):
        explain = json.loads((roots.analyses / "within_event" / "four_class" / f"{name}_explain.json").read_text())
        assert explain["permutation_importance_val"][0]["feature"] in ("gray_absdiff_mean", "edge_density_delta")
    assert "feature_importances" in json.loads((roots.analyses / "within_event" / "four_class" / "decision_tree_explain.json").read_text())
    assert (roots.figures / "within_event" / "four_class" / "knn_confusion_test.png").stat().st_size > 0
    meta = json.loads((models / "decision_tree.json").read_text())
    assert "area_px" in meta["preprocessing"]["log1p_features"]
    assert set(meta["metrics"]["test_per_event"]) == {"ev-a", "ev-b", "ev-c"}


def test_experiment_logistic_regression_still_trains_when_named(table, cfg, tmp_path_factory) -> None:
    """The feature-set experiment (phase3b) names logistic_regression explicitly; that path still works and saves
    the per-class coefficients, so the saved experiment stays reproducible. It is not a damage model of the app."""
    roots = OutputRoots(*(tmp_path_factory.mktemp(n) for n in ("lr_models", "lr_metrics", "lr_analyses", "lr_figures")))
    data = _data(table, cfg, "within")
    rows = run_design(data, cfg, TestSetGuard(), roots, ["logistic_regression"], say=lambda _m: None)
    assert [r["model"] for r in rows] == ["logistic_regression"] and rows[0]["test_macro_f1"] > 0.4
    explain = json.loads((roots.analyses / "within_event" / "four_class" / "logistic_regression_explain.json").read_text())
    assert "coefficients_per_class" in explain
    lr, _ = load_model(roots.models / "within_event" / "four_class" / "logistic_regression.joblib")
    row = data.X["test"].iloc[[0]]
    contrib = lr_contributions(lr, row, top_k=37)  # kept for the docs build script's worked example
    assert contrib["predicted_class"] == lr.predict(row)[0]
    assert sum(f["contribution"] for f in contrib["top_features"]) + contrib["intercept"] == pytest.approx(contrib["score"], abs=1e-3)


def test_per_building_explanations(trained) -> None:
    data, roots, _g, _r = trained
    folder = roots.models / "within_event" / "four_class"
    row = data.X["test"].iloc[[0]]
    tree, _ = load_model(folder / "decision_tree.joblib")
    path = decision_path(tree, row)
    assert path["prediction"] == tree.predict(row)[0]
    for step in path["steps"]:  # raw-unit comparison agrees with the branch taken
        assert (step["value_raw"] <= step["threshold_raw"] + 1e-6) == (step["op"] == "<=")


    knn, meta = load_model(folder / "knn.joblib")
    index = pd.read_parquet(folder / "knn_train_index.parquet")
    neighbours = knn_neighbours(knn, row, index)
    assert len(neighbours["neighbours"]) == meta["hyperparameters"]["n_neighbors"]
    assert set(n["sample_id"] for n in neighbours["neighbours"]) <= set(data.ids["train"]["sample_id"])


def test_cross_event_and_three_class(table, cfg, tmp_path) -> None:
    roots = OutputRoots(tmp_path / "m", tmp_path / "x", tmp_path / "a", tmp_path / "f")
    data = _data(table, cfg, "cross", "three_class")
    assert set(data.ids["test"]["event"]) == {"ev-c"} and "ev-c" not in set(data.ids["train"]["event"])
    assert set(data.y["train"].unique()) == {"no-damage", "damaged", "destroyed"}
    rows = run_design(data, cfg, TestSetGuard(), roots, ["baseline", "decision_tree"], say=lambda _m: None)
    assert set(rows[0]["test_recall"]) == {"no-damage", "damaged", "destroyed"}
    assert len(rows[0]["hyperparameters"]["thresholds_raw"]) == 2


def test_kmeans_and_confounding(table, cfg, tmp_path) -> None:
    roots = OutputRoots(tmp_path / "m", tmp_path / "x", tmp_path / "a", tmp_path / "f")
    data = _data(table, cfg, "within")
    km = run_kmeans(data, cfg, roots)
    assert km["chosen_k_by_silhouette"] in (2, 3, 4)
    for k, v in km["per_k"].items():
        assert sum(map(sum, v["contingency_rows_cluster_cols_class"])) == len(data.X["train"])
        assert -1 <= v["adjusted_rand_index"] <= 1 and len(v["cluster_sizes"]) == int(k)
    conf = run_confounding(data, cfg, TestSetGuard(), roots)
    assert conf["target"].startswith("event") and conf["test_metrics"]["accuracy"] > conf["majority_class_accuracy"]


def test_skew_log_is_fitted_on_train_only() -> None:
    train = pd.DataFrame({"a": np.r_[np.ones(99), 1000.0], "b": np.linspace(-1, 1, 100), "c": np.r_[-np.ones(99), -500.0]})
    t = SkewLog1p(2.0).fit(train)
    assert t.log_columns_.tolist() == [0] and t.skipped_signed_.tolist() == [2]
    out = t.transform(pd.DataFrame({"a": [1000.0], "b": [0.5], "c": [3.0]}))
    assert out[0, 0] == pytest.approx(np.log1p(1000)) and out[0, 1] == 0.5 and out[0, 2] == 3.0
    assert t.inverse_value(0, np.log1p(42.0)) == pytest.approx(42.0)


def test_threshold_rule_and_tuning() -> None:
    rule = ThresholdRule(feature_index=0, direction=1, thresholds=(1.0, 2.0, 3.0), classes=tuple(DAMAGE_CLASSES)).fit(np.zeros((1, 37)))
    X = np.zeros((5, 37))
    X[:, 0] = [0.5, 1.0, 2.5, 3.0, np.nan]
    assert rule.predict(X).tolist() == ["no-damage", "minor-damage", "major-damage", "destroyed", "no-damage"]
    rng = np.random.default_rng(0)
    y = rng.choice(DAMAGE_CLASSES, 2000)
    Xtr = rng.normal(0, 1, (2000, 37))
    j = FEATURE_COLUMNS.index("ring_dmean")
    Xtr[:, j] = -np.array([ORDINAL[c] for c in y]) + rng.normal(0, 0.1, 2000)  # negatively related
    fitted, log = tune_threshold_rule(Xtr[:1000], y[:1000], Xtr[1000:], y[1000:], list(DAMAGE_CLASSES), 0.05)
    assert FEATURE_COLUMNS[fitted.feature_index] == "ring_dmean" and fitted.direction == -1
    assert log[0]["val_macro_f1"] > 0.9


def test_metrics_known_inputs_and_zero_support() -> None:
    classes = ["a", "b", "c"]
    m = classification_metrics(["a", "a", "b", "b"], ["a", "b", "b", "b"], classes)
    assert m["accuracy"] == 0.75
    assert m["per_class"]["a"] == {"precision": 1.0, "recall": 0.5, "f1": pytest.approx(0.6667, abs=1e-4), "support": 2}
    assert m["per_class"]["c"]["f1"] is None and m["notes"]
    assert m["macro_f1"] == pytest.approx((0.6667 + 0.8) / 2, abs=1e-3)  # c excluded, not counted as 0
    assert m["confusion_matrix"]["rows_true_cols_pred"] == [[1, 1, 0], [0, 2, 0], [0, 0, 0]]


def test_raw_threshold_inverts_preprocessing(table) -> None:
    pipe = make_pipeline("decision_tree", {"max_depth": 2}, 0, 2.0).fit(table.X, table.y)
    j = FEATURE_COLUMNS.index("area_px")
    z = pipe[:-1].transform(table.X.iloc[[0]])[0, j]
    assert raw_threshold(pipe, j, z) == pytest.approx(table.X.iloc[0, j], rel=1e-6)

"""End-to-end synthetic deployment: the REAL pipeline run on a synthetic Parquet mirror. NOT xBD data.

Used by the API tests and by scripts/e2e_check.py.

Steps mirror the scripts: inspect (+cache) -> features -> manifests -> train LR/KNN/DT (RF deliberately
left untrained, to test the "not trained" path) -> review thresholds. NOT xBD data.
"""

import json
from pathlib import Path

import yaml

from app.core.config import REPO_ROOT
from app.core.events_config import load_events_config
from app.core.store import Store, StorePaths
from app.datasets.inspection import run_inspection, write_outputs
from app.datasets.synthetic_fixture import build_synthetic_parquet_many
from app.evaluation.analysis import run_confounding, run_kmeans
from app.evaluation.metrics import TestSetGuard
from app.features.config import FeatureConfig
from app.features.table import build_feature_table, load_feature_table, table_paths
from app.inference.review import choose_threshold, write_thresholds
from app.training.config import ModelSettings, load_training_config
from app.training.run import OutputRoots, design_data, run_design
from app.training.splits import build_manifest, write_manifest

EVENTS = {"synthetic-flood": "flooding", "synthetic-fire": "fire", "synthetic-quake": "earthquake"}


def build_deployment(base: Path) -> Store:
    data, artifacts, outputs, config_dir = base / "data", base / "artifacts", base / "outputs", base / "config"
    raw = build_synthetic_parquet_many(base / "raw")

    events_cfg = load_events_config(REPO_ROOT / "config" / "events.fixture_parquet.yaml")
    events_cfg = events_cfg.model_copy(update={
        "dataset_root": raw, "events": list(EVENTS), "disaster_types": EVENTS,
        "cache": events_cfg.cache.model_copy(update={"dir": data / "processed" / "scenes"}),
        "inspection": events_cfg.inspection.model_copy(update={"output_dir": artifacts / "inspection" / "xbd",
                                                               "figures_dir": outputs / "figures" / "inspection"}),
    })
    write_outputs(run_inspection(events_cfg, check_alignment=True, write_cache=True))

    feature_cfg = FeatureConfig(scene_cache_dir=data / "processed" / "scenes", output_dir=data / "processed" / "features")
    build_feature_table(feature_cfg, events_cfg)
    table = load_feature_table(table_paths(feature_cfg)["table"])

    real = load_training_config()
    designs = {"within_event": real.splits.designs["within_event"],
               "cross_event": real.splits.designs["cross_event"].model_copy(update={"holdout_event": "synthetic-quake"})}
    cfg = real.model_copy(update={
        "splits": real.splits.model_copy(update={"designs": designs, "min_class_count_warning": 5}),
        "models": {"logistic_regression": ModelSettings(grid={"C": [1.0]}),
                   "knn": ModelSettings(grid={"n_neighbors": [5], "weights": ["distance"]}),
                   "decision_tree": ModelSettings(grid={"max_depth": [4], "min_samples_leaf": [5]}),
                   "baseline": ModelSettings(quantile_step=0.1)},
        "explain": real.explain.model_copy(update={"permutation_sample": 100, "permutation_repeats": 1}),
        "kmeans": real.kmeans.model_copy(update={"k_values": [2, 3, 4], "silhouette_sample": 100, "pca_sample": 100}),
    })
    manifests = {}
    for name, design in designs.items():
        manifests[name] = build_manifest(name, design, table.ids, table.y, cfg.seed, 5, table.meta["fingerprint"])
        write_manifest(manifests[name], table.ids, data / "splits")

    roots = OutputRoots(artifacts / "models", artifacts / "metrics", artifacts / "analyses", outputs / "figures" / "models")
    guard = TestSetGuard()
    serving = design_data(table, manifests["within_event"], "four_class", cfg.label_schemes["four_class"])
    run_design(serving, cfg, guard, roots, ["baseline", "logistic_regression", "knn", "decision_tree"], say=lambda _m: None)
    cross = design_data(table, manifests["cross_event"], "four_class", cfg.label_schemes["four_class"])
    run_design(cross, cfg, guard, roots, ["logistic_regression", "decision_tree"], say=lambda _m: None)
    run_kmeans(serving, cfg, roots)
    run_confounding(serving, cfg, guard, roots)

    from app.training.persistence import load_model
    thresholds = {"rule": "lowest t with accuracy(confidence >= t) >= 0.5", "models": {}}
    for name in ("logistic_regression", "knn", "decision_tree"):
        pipe, _ = load_model(roots.models / "within_event" / "four_class" / f"{name}.joblib")
        proba = pipe.predict_proba(serving.X["val"])
        thresholds["models"][name] = choose_threshold(proba.max(axis=1), pipe.classes_[proba.argmax(axis=1)] == serving.y["val"].to_numpy(), 0.5)
    write_thresholds(roots.models / "within_event" / "four_class", thresholds)

    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "events.yaml").write_text(yaml.safe_dump(json.loads(events_cfg.model_dump_json())), encoding="utf-8")
    (config_dir / "training.yaml").write_text(yaml.safe_dump(json.loads(cfg.model_dump_json())), encoding="utf-8")
    return Store(StorePaths(data, artifacts, outputs, config_dir / "events.yaml", config_dir / "training.yaml"))


def empty_store(base: Path) -> Store:
    """Nothing built yet: the dataset-missing / not-trained mode. Real configs, empty folders."""
    return Store(StorePaths(base / "data", base / "artifacts", base / "outputs",
                            REPO_ROOT / "config" / "events.yaml", REPO_ROOT / "config" / "training.yaml"))

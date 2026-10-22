"""Shared loading for the Phase 3 scripts (configs, feature table, manifests -> DesignData)."""

from pathlib import Path

from app.core.config import REPO_ROOT
from app.features.config import load_feature_config
from app.features.table import FeatureTable, load_feature_table, table_paths
from app.training.config import TrainingConfig, load_training_config
from app.training.run import DesignData, design_data
from app.training.splits import load_manifest, validate_manifest

SPLITS_DIR = REPO_ROOT / "data" / "splits"


def load_inputs() -> tuple[TrainingConfig, FeatureTable, dict[str, dict]]:
    cfg = load_training_config()
    table = load_feature_table(table_paths(load_feature_config())["table"])
    manifests = {}
    for name in cfg.splits.designs:
        manifest = load_manifest(SPLITS_DIR / f"{name}_v1.json")
        validate_manifest(manifest, table.ids)  # never train on a leaky or stale split
        if manifest["feature_table_fingerprint"] != table.meta["fingerprint"]:
            raise ValueError(f"{name} manifest was built for a different feature table; run scripts/make_splits.py")
        manifests[name] = manifest
    return cfg, table, manifests


def data_for(cfg: TrainingConfig, table: FeatureTable, manifests: dict[str, dict], design: str, scheme: str) -> DesignData:
    return design_data(table, manifests[design], scheme, cfg.label_schemes[scheme])

"""Split manifests: scene grouping, stratification, cross-event rule, leakage checks."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.core.config import REPO_ROOT
from app.core.damage_scale import DAMAGE_CLASSES
from app.features.table import load_feature_table
from app.training.config import SplitDesign, load_training_config
from app.training.splits import (
    SplitLeakageError,
    build_manifest,
    load_manifest,
    split_of_buildings,
    validate_manifest,
    write_manifest,
)

WITHIN = SplitDesign(label="within-event generalisation", description="d", method="stratified_group",
                     fractions={"train": 0.7, "val": 0.15, "test": 0.15})
CROSS = SplitDesign(label="cross-event generalisation", description="d", method="event_holdout",
                    holdout_event="ev-e", fractions={"train": 0.85, "val": 0.15})
FINGERPRINT = {"feature_schema_version": 1}


@pytest.fixture(scope="module")
def table() -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic id table: 5 events x 40 scenes x 5-40 buildings, imbalanced classes. NOT xBD data."""
    rng = np.random.default_rng(0)
    rows = []
    for e in "abcde":
        for s in range(40):
            for b in range(int(rng.integers(5, 40))):
                rows.append({"sample_id": f"ev-{e}_{s:08d}_b{b:04d}", "scene_id": f"ev-{e}_{s:08d}", "event": f"ev-{e}",
                             "label": rng.choice(DAMAGE_CLASSES, p=[0.6, 0.15, 0.15, 0.1])})
    frame = pd.DataFrame(rows)
    return frame[["sample_id", "scene_id", "event"]], frame["label"]


def test_within_event_manifest(table) -> None:
    ids, labels = table
    m = build_manifest("within_event", WITHIN, ids, labels, 42, 10, FINGERPRINT)
    shares = {k: v["building_share"] for k, v in m["counts"].items()}
    assert shares["train"] == pytest.approx(0.70, abs=0.05) and shares["test"] == pytest.approx(0.15, abs=0.04)
    overall = labels.value_counts(normalize=True)
    for split in ("train", "val", "test"):
        part = labels[split_of_buildings(m, ids) == split].value_counts(normalize=True)
        assert (part - overall).abs().max() < 0.05  # class mix preserved
    assert len(m["counts"]["test"]["events"]) == 5  # every event appears in test: within-event


def test_no_scene_overlap_in_either_design(table) -> None:
    ids, labels = table
    for name, design in (("within_event", WITHIN), ("cross_event", CROSS)):
        m = build_manifest(name, design, ids, labels, 42, 10, FINGERPRINT)
        split = split_of_buildings(m, ids)
        groups = pd.DataFrame({"scene": ids["scene_id"], "split": split}).groupby("scene")["split"].nunique()
        assert (groups == 1).all()
        sets = [set(ids.loc[split == s, "scene_id"]) for s in ("train", "val", "test")]
        assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])


def test_cross_event_holds_out_the_whole_event(table) -> None:
    ids, labels = table
    m = build_manifest("cross_event", CROSS, ids, labels, 42, 10, FINGERPRINT)
    split = split_of_buildings(m, ids)
    assert set(ids.loc[split == "test", "event"]) == {"ev-e"}
    assert (split[ids["event"] == "ev-e"] == "test").all()
    assert "ev-e" not in m["counts"]["train"]["events"] and "ev-e" not in m["counts"]["val"]["events"]


def test_deterministic_and_seed_sensitive(table) -> None:
    ids, labels = table
    a = build_manifest("within_event", WITHIN, ids, labels, 42, 10, FINGERPRINT)
    b = build_manifest("within_event", WITHIN, ids, labels, 42, 10, FINGERPRINT)
    c = build_manifest("within_event", WITHIN, ids, labels, 7, 10, FINGERPRINT)
    assert a["scenes"] == b["scenes"] and a["manifest_hash"] == b["manifest_hash"]
    assert a["scenes"] != c["scenes"]


def test_validator_catches_leakage(table) -> None:
    ids, labels = table
    m = build_manifest("cross_event", CROSS, ids, labels, 42, 10, FINGERPRINT)
    moved = dict(m["scenes"])
    michael_scene = ids.loc[ids["event"] == "ev-e", "scene_id"].iloc[0]
    moved[michael_scene] = "train"
    with pytest.raises(SplitLeakageError, match="ev-e scenes are in train"):
        validate_manifest({**m, "scenes": moved}, ids)
    missing = {k: v for k, v in m["scenes"].items() if k != michael_scene}
    with pytest.raises(SplitLeakageError, match="missing from the manifest"):
        validate_manifest({**m, "scenes": missing}, ids)


def test_rare_class_is_flagged(table) -> None:
    ids, labels = table
    m = build_manifest("within_event", WITHIN, ids, labels, 42, min_class_count_warning=10_000, feature_fingerprint=FINGERPRINT)
    assert any("destroyed" in w for w in m["warnings"])


def test_written_manifest_round_trips_and_detects_edits(table, tmp_path) -> None:
    ids, labels = table
    m = build_manifest("within_event", WITHIN, ids, labels, 42, 10, FINGERPRINT)
    paths = write_manifest(m, ids, tmp_path)
    assert load_manifest(paths["json"])["scenes"] == m["scenes"]
    assert len(pd.read_csv(paths["csv"])) == ids["scene_id"].nunique()
    doc = json.loads(paths["json"].read_text())
    first_train = next(k for k, v in doc["scenes"].items() if v == "train")
    doc["scenes"][first_train] = "test"  # hand-edit one scene's split
    paths["json"].write_text(json.dumps(doc))
    with pytest.raises(SplitLeakageError, match="hash mismatch"):
        load_manifest(paths["json"])


def test_real_training_config_designs() -> None:
    cfg = load_training_config()
    assert cfg.splits.designs["within_event"].label == "within-event generalisation"
    assert cfg.splits.designs["cross_event"].label == "cross-event generalisation"
    assert cfg.splits.designs["cross_event"].holdout_event == "hurricane-michael"


REAL_TABLE = REPO_ROOT / "data" / "processed" / "features" / "features_v1.parquet"
REAL_SPLITS = REPO_ROOT / "data" / "splits"


@pytest.mark.skipif(not (REAL_TABLE.is_file() and (REAL_SPLITS / "within_event_v1.json").is_file()),
                    reason="real feature table / manifests not built on this machine")
@pytest.mark.parametrize("design", ["within_event", "cross_event"])
def test_real_manifests_have_zero_scene_overlap(design: str) -> None:
    """Leakage test on the actual xBD manifests (runs only where they exist)."""
    table = load_feature_table(REAL_TABLE)
    manifest = load_manifest(REAL_SPLITS / f"{design}_v1.json")
    validate_manifest(manifest, table.ids)  # raises on any overlap or holdout violation
    assert sum(c["buildings"] for c in manifest["counts"].values()) == len(table.ids)
    assert manifest["feature_table_fingerprint"] == table.meta["fingerprint"]

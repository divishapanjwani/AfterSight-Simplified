"""Scene-grouped train/val/test split manifests and their leakage checks.

Every building of a scene goes to the same split: neighbouring buildings share imagery, lighting and
often damage, so splitting them apart would leak information and inflate scores. Splits are stored as
scene -> split manifests in data/splits/ and validated before they are written and again in tests.
"""

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from app.core.damage_scale import DAMAGE_CLASSES
from app.training.config import SplitDesign

MANIFEST_VERSION = 1
FOLD_GRANULARITY = 20  # fractions are realised in 5% steps of scene-grouped, class-stratified folds


class SplitLeakageError(Exception):
    """A manifest that puts one scene in two splits or breaks the design's event rule."""


def stratified_group_partition(scene_ids: pd.Series, labels: pd.Series, fractions: dict[str, float], seed: int) -> dict[str, str]:
    """Assign whole scenes to splits so each split's class mix resembles the overall mix.

    StratifiedGroupKFold makes FOLD_GRANULARITY folds that never share a group (scene) and keep the
    building-level class proportions similar; we then hand out folds to splits by their fractions.
    """
    counts = {name: round(frac * FOLD_GRANULARITY) for name, frac in fractions.items()}
    if sum(counts.values()) != FOLD_GRANULARITY:
        raise ValueError(f"fractions {fractions} are not multiples of {1 / FOLD_GRANULARITY:.0%}")
    n_scenes = scene_ids.nunique()
    if n_scenes < FOLD_GRANULARITY:
        raise ValueError(f"need at least {FOLD_GRANULARITY} scenes to split, got {n_scenes}")
    sgkf = StratifiedGroupKFold(n_splits=FOLD_GRANULARITY, shuffle=True, random_state=seed)
    fold_of_scene: dict[str, int] = {}
    for fold, (_train, held) in enumerate(sgkf.split(np.zeros(len(labels)), labels.to_numpy(), scene_ids.to_numpy())):
        for scene in scene_ids.iloc[held].unique():
            fold_of_scene[scene] = fold
    split_of_fold: dict[int, str] = {}
    fold = 0
    for name in ("test", "val", "train"):  # fixed order keeps the mapping reproducible
        for _ in range(counts.get(name, 0)):
            split_of_fold[fold] = name
            fold += 1
    return {scene: split_of_fold[f] for scene, f in sorted(fold_of_scene.items())}


def build_manifest(name: str, design: SplitDesign, ids: pd.DataFrame, labels: pd.Series, seed: int,
                   min_class_count_warning: int, feature_fingerprint: dict[str, Any]) -> dict[str, Any]:
    """Make one design's manifest from the feature table's id columns and labels."""
    if design.method == "stratified_group":
        assignment = stratified_group_partition(ids["scene_id"], labels, design.fractions, seed)
    else:
        held_out = ids["event"] == design.holdout_event
        if not held_out.any():
            raise ValueError(f"holdout_event {design.holdout_event!r} has no buildings in the feature table")
        rest = ~held_out
        assignment = stratified_group_partition(ids.loc[rest, "scene_id"], labels[rest], design.fractions, seed)
        assignment.update({scene: "test" for scene in ids.loc[held_out, "scene_id"].unique()})
        assignment = dict(sorted(assignment.items()))

    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "design": name,
        "label": design.label,
        "description": " ".join(design.description.split()),
        "method": design.method,
        "holdout_event": design.holdout_event,
        "fractions_requested": design.fractions,
        "seed": seed,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "feature_table_fingerprint": feature_fingerprint,
        "scenes": assignment,
    }
    manifest["counts"] = split_counts(assignment, ids, labels)
    manifest["warnings"] = class_warnings(manifest["counts"], min_class_count_warning)
    manifest["manifest_hash"] = manifest_hash(manifest)
    validate_manifest(manifest, ids)
    return manifest


def manifest_hash(manifest: dict[str, Any]) -> str:
    """Hash of what determines the split (design + scene assignment); stored in model metadata later."""
    payload = json.dumps({"design": manifest["design"], "scenes": manifest["scenes"]}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def split_of_buildings(manifest: dict[str, Any], ids: pd.DataFrame) -> pd.Series:
    """Split name for every building row (via its scene). Buildings of unknown scenes raise."""
    split = ids["scene_id"].map(manifest["scenes"])
    if split.isna().any():
        missing = sorted(ids.loc[split.isna(), "scene_id"].unique())[:5]
        raise SplitLeakageError(f"{split.isna().sum()} buildings belong to scenes missing from the manifest, e.g. {missing}")
    return split


def split_counts(assignment: dict[str, str], ids: pd.DataFrame, labels: pd.Series) -> dict[str, Any]:
    split = ids["scene_id"].map(assignment)
    total = len(ids)
    counts = {}
    for name in ("train", "val", "test"):
        rows = split == name
        counts[name] = {
            "scenes": int(ids.loc[rows, "scene_id"].nunique()),
            "buildings": int(rows.sum()),
            "building_share": round(float(rows.sum()) / total, 4) if total else 0.0,
            "classes": {c: int((labels[rows] == c).sum()) for c in DAMAGE_CLASSES},
            "events": {e: int(n) for e, n in ids.loc[rows, "event"].value_counts().sort_index().items()},
        }
    return counts


def class_warnings(counts: dict[str, Any], threshold: int) -> list[str]:
    warnings = []
    for split, c in counts.items():
        for cls, n in c["classes"].items():
            if n == 0:
                warnings.append(f"{split}: no {cls} buildings — per-class metrics for {cls} are undefined here")
            elif n < threshold:
                warnings.append(f"{split}: only {n} {cls} buildings (< {threshold}) — {cls} metrics will be noisy")
    return warnings


def validate_manifest(manifest: dict[str, Any], ids: pd.DataFrame) -> None:
    """Raise SplitLeakageError unless every scene is in exactly one split and the design's rules hold."""
    building_split = split_of_buildings(manifest, ids)
    per_scene = pd.DataFrame({"scene_id": ids["scene_id"], "split": building_split}).groupby("scene_id")["split"].nunique()
    if (per_scene > 1).any():
        raise SplitLeakageError(f"scenes in more than one split: {sorted(per_scene[per_scene > 1].index)[:5]}")
    by_split = {name: set(ids.loc[building_split == name, "scene_id"]) for name in ("train", "val", "test")}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = by_split[a] & by_split[b]
        if overlap:
            raise SplitLeakageError(f"{len(overlap)} scenes shared by {a} and {b}, e.g. {sorted(overlap)[:3]}")
    if manifest["method"] == "event_holdout":
        holdout = manifest["holdout_event"]
        test_events = set(ids.loc[building_split == "test", "event"])
        if test_events != {holdout}:
            raise SplitLeakageError(f"cross-event test set must contain only {holdout}, found {sorted(test_events)}")
        leaked = ids.loc[(building_split != "test") & (ids["event"] == holdout), "scene_id"]
        if len(leaked):
            raise SplitLeakageError(f"{leaked.nunique()} {holdout} scenes are in train/val")
    if not by_split["train"] or not by_split["val"] or not by_split["test"]:
        raise SplitLeakageError("every split must contain at least one scene")
    counts = Counter(manifest["scenes"].values())
    if set(counts) - {"train", "val", "test"}:
        raise SplitLeakageError(f"unknown split names {sorted(set(counts) - {'train', 'val', 'test'})}")


def write_manifest(manifest: dict[str, Any], ids: pd.DataFrame, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{manifest['design']}_v{MANIFEST_VERSION}"
    paths = {"json": out_dir / f"{stem}.json", "csv": out_dir / f"{stem}.csv"}
    paths["json"].write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    per_scene = ids.groupby("scene_id").agg(event=("event", "first"), n_buildings=("scene_id", "size")).reset_index()
    per_scene["split"] = per_scene["scene_id"].map(manifest["scenes"])
    per_scene[["scene_id", "event", "split", "n_buildings"]].to_csv(paths["csv"], index=False)
    return paths


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        raise SplitLeakageError(f"{path.name}: unsupported manifest version {manifest.get('manifest_version')}")
    if manifest_hash(manifest) != manifest.get("manifest_hash"):
        raise SplitLeakageError(f"{path.name}: manifest hash mismatch — the scene assignment was edited")
    return manifest

"""File-based artifact store behind the API (no database).

Everything the API serves is read from files that the pipeline scripts wrote. Nothing is computed from
guesses: when a file is missing, the corresponding accessor raises NotReady/ModelNotTrained carrying the
command that builds it. Request IDs are validated against these files before any path is formed, and
paths are always built from the canonical IDs in the files, never from request strings.
"""

import json
import re
import threading
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import pandas as pd

from app.core.config import REPO_ROOT, Settings
from app.core.errors import ModelNotTrained, NotFound, NotReady
from app.core.events_config import ConfigError, EventsConfig, load_events_config
from app.features.schema import FEATURE_SCHEMA_VERSION
from app.features.table import FeatureTable, FeatureTableError, load_feature_table
from app.inference.review import read_thresholds, threshold_status
from app.training.config import TrainingConfig, load_training_config
from app.training.persistence import ModelLoadError, load_model
from app.training.splits import SplitLeakageError, load_manifest

SCENE_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*_\d{8}$")
BUILDING_ID = re.compile(r"^(?P<scene>[a-z0-9]+(?:-[a-z0-9]+)*_\d{8})_b(?P<index>\d{4,6})$")
ANALYSIS_ID = re.compile(r"^[0-9a-f]{16}$")

FIX = {
    "cache": "python scripts/inspect_dataset.py --alignment",
    "features": "python scripts/build_features.py",
    "splits": "python scripts/make_splits.py",
    "models": "python scripts/train.py",
    "review": "python scripts/calibrate_review.py",
    "experiments": "python scripts/run_phase3b.py",
}


@dataclass(frozen=True)
class StorePaths:
    data: Path
    artifacts: Path
    outputs: Path
    events_config: Path
    training_config: Path

    @classmethod
    def from_settings(cls, settings: Settings) -> "StorePaths":
        return cls(settings.resolve(settings.data_dir), settings.resolve(settings.artifacts_dir),
                   settings.resolve(settings.outputs_dir), settings.resolve(settings.events_config),
                   settings.resolve(settings.training_config))

    @property
    def scene_cache(self) -> Path:
        return self.data / "processed" / "scenes"

    @property
    def feature_table(self) -> Path:
        return self.data / "processed" / "features" / f"features_v{FEATURE_SCHEMA_VERSION}.parquet"

    @property
    def splits(self) -> Path:
        return self.data / "splits"

    @property
    def inspection_scenes_csv(self) -> Path:
        return self.artifacts / "inspection" / "xbd" / "scenes.csv"

    @property
    def analysis_runs(self) -> Path:
        return self.artifacts / "analyses" / "runs"

    @property
    def reports(self) -> Path:
        return self.outputs / "reports"


class Store:
    def __init__(self, paths: StorePaths):
        self.paths = paths
        self._lock = threading.Lock()
        self._models: dict[str, tuple[Any, dict[str, Any]]] = {}
        self._scene_docs: dict[str, dict[str, Any]] = {}
        self._memo: dict[str, Any] = {}

    def memoised(self, key: str, compute):
        """In-memory cache for derived tables (cleared when models are replaced)."""
        if key not in self._memo:
            value = compute()
            with self._lock:
                self._memo.setdefault(key, value)
        return self._memo[key]

    # ------------------------------------------------------------ configs
    @cached_property
    def training_config(self) -> TrainingConfig:
        return load_training_config(self.paths.training_config)

    @cached_property
    def events_config(self) -> EventsConfig | None:
        try:
            return load_events_config(self.paths.events_config)
        except ConfigError:
            return None

    @property
    def serving(self):
        return self.training_config.serving

    # ------------------------------------------------------------ scenes
    def scene_index(self) -> dict[str, dict[str, Any]]:
        path = self.paths.scene_cache / "index.json"
        if not path.is_file():
            raise NotReady("The scene cache has not been built yet.", fix=FIX["cache"], code="dataset_not_ready")
        return {s["scene_id"]: s for s in json.loads(path.read_text(encoding="utf-8"))["scenes"]}

    def scene_index_meta(self) -> dict[str, Any]:
        doc = json.loads((self.paths.scene_cache / "index.json").read_text(encoding="utf-8"))
        return {k: v for k, v in doc.items() if k != "scenes"}

    def valid_scene_id(self, scene_id: str) -> str:
        """Return the canonical scene id, or raise NotFound. Rejects anything that is not an id."""
        if not SCENE_ID.match(scene_id or ""):
            raise NotFound("Unknown scene id.", code="scene_not_found")
        index = self.scene_index()
        if scene_id not in index:
            raise NotFound("Unknown scene id.", code="scene_not_found")
        return index[scene_id]["scene_id"]

    def scene_dir(self, scene_id: str) -> Path:
        return self.paths.scene_cache / self.valid_scene_id(scene_id)

    def scene_doc(self, scene_id: str) -> dict[str, Any]:
        sid = self.valid_scene_id(scene_id)
        if sid not in self._scene_docs:
            with self._lock:
                if len(self._scene_docs) > 256:
                    self._scene_docs.clear()
                self._scene_docs[sid] = json.loads((self.paths.scene_cache / sid / "scene.json").read_text(encoding="utf-8"))
        return self._scene_docs[sid]

    def valid_building(self, building_uid: str) -> tuple[str, dict[str, Any]]:
        match = BUILDING_ID.match(building_uid or "")
        if not match:
            raise NotFound("Unknown building id.", code="building_not_found")
        try:
            doc = self.scene_doc(match["scene"])
        except NotFound as exc:
            raise NotFound("Unknown building id.", code="building_not_found") from exc
        for b in doc["buildings"]:
            if b["building_id"] == building_uid:
                return doc["scene_id"], b
        raise NotFound("Unknown building id.", code="building_not_found")

    def disaster_type(self, event: str) -> str | None:
        return self.events_config.disaster_types.get(event) if self.events_config else None

    @cached_property
    def quality(self) -> dict[str, dict[str, Any]]:
        """Per-scene data-quality fields from the inspector (alignment check), if it was run."""
        path = self.paths.inspection_scenes_csv
        if not path.is_file():
            return {}
        frame = pd.read_csv(path, usecols=lambda c: c in {"scene_id", "alignment_shift_px", "alignment_flag", "footprint_mask_iou"})
        return {r["scene_id"]: r for r in frame.to_dict("records")}

    # ------------------------------------------------------------ features, splits
    @cached_property
    def feature_table(self) -> FeatureTable:
        try:
            return load_feature_table(self.paths.feature_table)
        except FeatureTableError as exc:
            raise NotReady(str(exc), fix=FIX["features"], code="features_not_ready") from exc

    @cached_property
    def rows_by_scene(self) -> dict[str, Any]:
        return self.feature_table.ids.groupby("scene_id").indices

    def manifest(self, design: str | None = None) -> dict[str, Any]:
        design = design or self.serving.design
        if design not in self.training_config.splits.designs:
            raise NotFound("Unknown split design.", code="design_not_found")
        path = self.paths.splits / f"{design}_v1.json"
        if not path.is_file():
            raise NotReady(f"The {design} split manifest has not been built.", fix=FIX["splits"], code="splits_not_ready")
        try:
            return load_manifest(path)
        except SplitLeakageError as exc:
            raise NotReady(str(exc), fix=FIX["splits"], code="splits_invalid") from exc

    def split_of_scene(self, scene_id: str, design: str | None = None) -> str:
        try:
            return self.manifest(design)["scenes"].get(scene_id, "not_in_manifest")
        except NotReady:
            return "not_in_manifest"

    # ------------------------------------------------------------ models
    def model_dir(self, design: str | None = None, scheme: str | None = None) -> Path:
        return self.paths.artifacts / "models" / (design or self.serving.design) / (scheme or self.serving.scheme)

    def valid_model_name(self, name: str) -> str:
        if name not in self.serving.models:
            raise NotFound(f"Unknown model. Choose one of: {', '.join(self.serving.models)}.", code="model_not_found")
        return name

    def model(self, name: str) -> tuple[Any, dict[str, Any]]:
        name = self.valid_model_name(name)
        if name not in self._models:
            with self._lock:
                if name not in self._models:
                    try:
                        manifest_hash = self.manifest()["manifest_hash"]
                    except NotReady:
                        manifest_hash = None
                    try:
                        self._models[name] = load_model(self.model_dir() / f"{name}.joblib", expected_manifest_hash=manifest_hash)
                    except ModelLoadError as exc:
                        raise ModelNotTrained(str(exc), fix=FIX["models"]) from exc
        return self._models[name]

    def model_meta(self, name: str, design: str | None = None, scheme: str | None = None) -> dict[str, Any] | None:
        path = self.model_dir(design, scheme) / f"{name}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    def knn_index(self) -> pd.DataFrame:
        path = self.model_dir() / "knn_train_index.parquet"
        if not path.is_file():
            raise ModelNotTrained("KNN training index is missing.", fix=FIX["models"])
        return pd.read_parquet(path)

    def review_thresholds(self) -> dict[str, Any] | None:
        """The validation-chosen review thresholds, or None if the file is missing or unreadable."""
        doc, _problem = read_thresholds(self.model_dir() / "review_thresholds.json")
        return doc

    def optional_models(self) -> list[str]:
        """Served models marked optional in config/training.yaml (they never make a status tile PARTIAL)."""
        settings = self.training_config.models
        return [m for m in self.serving.models if m in settings and settings[m].optional]

    def optional_untrained(self) -> list[str]:
        """Optional served models that have no saved model yet."""
        return [m for m in self.optional_models() if self.model_meta(m) is None]

    def review_threshold_status(self) -> dict[str, Any]:
        doc, problem = read_thresholds(self.model_dir() / "review_thresholds.json")
        return threshold_status(doc, problem, list(self.serving.models), self.optional_models(), self.optional_untrained())

    # ------------------------------------------------------------ saved results (Model Arena)
    def read_json(self, relative: str) -> dict[str, Any] | None:
        """Read an artifact by a FIXED relative path chosen by the server (never from a request)."""
        path = self.paths.artifacts / relative
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    def forget_models(self) -> None:
        """Drop loaded models (after a training job replaced them)."""
        with self._lock:
            self._models.clear()
            self._memo.clear()

"""Load config/training.yaml: split designs (Phase 2) and model settings (Phase 3)."""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator

from app.core.config import REPO_ROOT
from app.core.damage_scale import DAMAGE_CLASSES
from app.core.events_config import ConfigError

DEFAULT_TRAINING_CONFIG = REPO_ROOT / "config" / "training.yaml"


class SplitDesign(BaseModel):
    label: str
    description: str
    method: Literal["stratified_group", "event_holdout"]
    fractions: dict[str, float]
    holdout_event: str | None = None

    @model_validator(mode="after")
    def _check(self) -> "SplitDesign":
        expected = {"train", "val", "test"} if self.method == "stratified_group" else {"train", "val"}
        if set(self.fractions) != expected:
            raise ValueError(f"{self.method} needs fractions for exactly {sorted(expected)}")
        if abs(sum(self.fractions.values()) - 1) > 1e-9:
            raise ValueError("fractions must sum to 1")
        if self.method == "event_holdout" and not self.holdout_event:
            raise ValueError("event_holdout needs holdout_event")
        return self


class SplitsConfig(BaseModel):
    min_class_count_warning: int = Field(100, ge=1)
    designs: dict[str, SplitDesign]


class LabelScheme(BaseModel):
    label: str
    mapping: dict[str, str]

    @model_validator(mode="after")
    def _covers_classes(self) -> "LabelScheme":
        if set(self.mapping) != set(DAMAGE_CLASSES):
            raise ValueError(f"mapping must cover exactly {list(DAMAGE_CLASSES)}")
        return self

    @property
    def classes(self) -> list[str]:
        """Target classes in severity order (order of first appearance in DAMAGE_CLASSES)."""
        ordered: list[str] = []
        for cls in DAMAGE_CLASSES:
            if self.mapping[cls] not in ordered:
                ordered.append(self.mapping[cls])
        return ordered


class ModelSettings(BaseModel):
    grid: dict[str, list[Any]] = {}
    optional: bool = False
    max_iter: int = 3000
    n_estimators: int = 200
    quantile_step: float = Field(0.05, gt=0, lt=0.5)


class ExplainConfig(BaseModel):
    permutation_sample: int = Field(4000, ge=100)
    permutation_repeats: int = Field(5, ge=1)


class KMeansConfig(BaseModel):
    k_values: list[int] = list(range(2, 11))
    silhouette_sample: int = Field(5000, ge=100)
    pca_sample: int = Field(3000, ge=100)


class ConfoundingConfig(BaseModel):
    grid: dict[str, list[Any]] = {"C": [0.01, 0.1, 1.0, 10.0]}


class ServingConfig(BaseModel):
    design: str = "within_event"
    scheme: str = "four_class"
    models: list[str] = ["knn", "decision_tree"]


class ReviewConfig(BaseModel):
    target_accuracy: float = Field(0.70, gt=0, lt=1)
    disagreement_models: list[str] | None = None  # None: every served model


class TrainingConfig(BaseModel):
    seed: int = 42
    splits: SplitsConfig
    label_schemes: dict[str, LabelScheme] = {}
    preprocessing: dict[str, float] = {"skew_threshold": 2.0}
    default_model: str = "knn"
    models: dict[str, ModelSettings] = {}
    explain: ExplainConfig = ExplainConfig()
    kmeans: KMeansConfig = KMeansConfig()
    confounding: ConfoundingConfig = ConfoundingConfig()
    serving: ServingConfig = ServingConfig()
    review: ReviewConfig = ReviewConfig()

    @model_validator(mode="after")
    def _known_models(self) -> "TrainingConfig":
        unknown = set(self.models) - {"baseline", "logistic_regression", "knn", "decision_tree", "random_forest", "svm",
                                      "soft_voting"}
        if unknown:
            raise ValueError(f"unknown models in config: {sorted(unknown)}")
        return self


def load_training_config(path: Path = DEFAULT_TRAINING_CONFIG) -> TrainingConfig:
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    try:
        return TrainingConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except (yaml.YAMLError, ValidationError) as exc:
        raise ConfigError(f"{path.name} is invalid: {exc}") from exc

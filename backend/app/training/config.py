"""Load config/training.yaml (split designs now; model settings are added in Phase 3)."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator

from app.core.config import REPO_ROOT
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


class TrainingConfig(BaseModel):
    seed: int = 42
    splits: SplitsConfig


def load_training_config(path: Path = DEFAULT_TRAINING_CONFIG) -> TrainingConfig:
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    try:
        return TrainingConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except (yaml.YAMLError, ValidationError) as exc:
        raise ConfigError(f"{path.name} is invalid: {exc}") from exc

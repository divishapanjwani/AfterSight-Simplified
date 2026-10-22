"""Load and validate config/events.yaml (which xBD events to use and where the data lives)."""

import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from app.core.config import REPO_ROOT
from app.core.damage_scale import DAMAGE_CLASSES

DEFAULT_EVENTS_CONFIG = REPO_ROOT / "config" / "events.yaml"
_EVENT_NAME = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


class ConfigError(Exception):
    """Raised when a config file is missing or invalid; the message is safe to show to the user."""


class GeometryConfig(BaseModel):
    min_area_px: float = Field(16, ge=0)


class InspectionConfig(BaseModel):
    output_dir: Path
    figures_dir: Path
    n_samples: int = Field(4, ge=0, le=12)
    seed: int = 42
    alignment_flag_px: float = Field(8, gt=0)


class MaskConfig(BaseModel):
    """How rasterised annotation masks (Parquet format) map to damage classes."""

    damage_values: dict[int, str] = {1: "no-damage", 2: "minor-damage", 3: "major-damage", 4: "destroyed"}
    connectivity: Literal[4, 8] = 4  # 4 = pixels must share an edge; merges fewer touching buildings

    @field_validator("damage_values")
    @classmethod
    def _maps_onto_classes(cls, values: dict[int, str]) -> dict[int, str]:
        if 0 in values:
            raise ValueError("mask value 0 is background and cannot map to a damage class")
        if sorted(values.values()) != sorted(DAMAGE_CLASSES):
            raise ValueError(f"damage_values must map exactly once onto {list(DAMAGE_CLASSES)}")
        return values


class CacheConfig(BaseModel):
    dir: Path = Path("data/processed/scenes")
    thumbnail_px: int = Field(256, ge=64, le=1024)


class EventsConfig(BaseModel):
    source: Literal["xbd", "synthetic"]
    dataset_format: Literal["xbd_parquet", "xbd_json"]
    dataset_root: Path
    xbd_splits: list[str] = Field(["train", "tier3", "test", "hold"], min_length=1)  # xbd_json only
    image_extensions: list[str] = [".png"]  # xbd_json only
    events: list[str] = Field(min_length=1)
    # Disaster type per event. The Parquet mirror does not store it, so it is CONFIG-SOURCED and labelled so.
    disaster_types: dict[str, str] = {}
    excluded_labels: list[str] = ["un-classified"]  # xbd_json only (masks carry no such label)
    geometry: GeometryConfig = GeometryConfig()
    masks: MaskConfig = MaskConfig()
    cache: CacheConfig = CacheConfig()
    inspection: InspectionConfig

    @field_validator("events")
    @classmethod
    def _valid_event_names(cls, events: list[str]) -> list[str]:
        bad = [e for e in events if not _EVENT_NAME.match(e)]
        if bad:
            raise ValueError(f"event names must be lowercase-with-hyphens, got {bad}")
        if len(set(events)) != len(events):
            raise ValueError("events list contains duplicates")
        return events

    @model_validator(mode="after")
    def _types_cover_events(self) -> "EventsConfig":
        extra = sorted(set(self.disaster_types) - set(self.events))
        if extra:
            raise ValueError(f"disaster_types lists events that are not configured: {extra}")
        if self.dataset_format == "xbd_parquet":
            missing = [e for e in self.events if e not in self.disaster_types]
            if missing:
                raise ValueError(f"disaster_types is required for every event in the Parquet format; missing {missing}")
        return self

    @field_validator("excluded_labels")
    @classmethod
    def _excluded_not_a_class(cls, labels: list[str]) -> list[str]:
        clash = set(labels) & set(DAMAGE_CLASSES)
        if clash:
            raise ValueError(f"excluded_labels cannot contain damage classes: {sorted(clash)}")
        return labels

    @field_validator("image_extensions")
    @classmethod
    def _normalise_extensions(cls, exts: list[str]) -> list[str]:
        return [e.lower() if e.startswith(".") else f".{e.lower()}" for e in exts]

    def resolve(self, path: Path) -> Path:
        """Config paths are relative to the repository root."""
        return path if path.is_absolute() else REPO_ROOT / path


def load_events_config(path: Path = DEFAULT_EVENTS_CONFIG) -> EventsConfig:
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path.name} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path.name} must contain a YAML mapping at the top level")
    try:
        return EventsConfig.model_validate(raw)
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
        raise ConfigError(f"{path.name} is invalid — {problems}") from exc

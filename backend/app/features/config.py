"""Load config/features.yaml."""

import hashlib
import json
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, ValidationError

from app.core.config import REPO_ROOT
from app.core.events_config import ConfigError

DEFAULT_FEATURES_CONFIG = REPO_ROOT / "config" / "features.yaml"


class FeatureConfig(BaseModel):
    scene_cache_dir: Path
    output_dir: Path
    ring_width_px: int = Field(8, ge=1, le=64)
    canny_sigma: float = Field(1.5, gt=0)
    glcm_levels: int = Field(32, ge=4, le=256)
    glcm_distance_px: int = Field(1, ge=1, le=8)

    def resolve(self, path: Path) -> Path:
        return path if path.is_absolute() else REPO_ROOT / path

    def extraction_hash(self) -> str:
        """Hash of the settings that change feature values (paths excluded)."""
        settings = self.model_dump(exclude={"scene_cache_dir", "output_dir"})
        return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:16]


def load_feature_config(path: Path = DEFAULT_FEATURES_CONFIG) -> FeatureConfig:
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    try:
        return FeatureConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except (yaml.YAMLError, ValidationError) as exc:
        raise ConfigError(f"{path.name} is invalid: {exc}") from exc

"""Save/load fitted pipelines with joblib plus a metadata JSON, and refuse incompatible loads."""

import json
from pathlib import Path
from typing import Any

import joblib
import sklearn
from sklearn.pipeline import Pipeline

from app.features.schema import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION


class ModelLoadError(Exception):
    """Saved model can't be used with this code/data; the message says why and how to fix it."""


def save_model(pipe: Pipeline, metadata: dict[str, Any], folder: Path, name: str) -> dict[str, Path]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = {"model": folder / f"{name}.joblib", "meta": folder / f"{name}.json"}
    joblib.dump(pipe, paths["model"], compress=3)
    full = {
        **metadata,
        "model_name": name,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_columns": FEATURE_COLUMNS,
        "sklearn_version": sklearn.__version__,
        "artifact_bytes": paths["model"].stat().st_size,
    }
    paths["meta"].write_text(json.dumps(full, indent=1), encoding="utf-8")
    return paths


def load_model(model_path: Path, expected_manifest_hash: str | None = None) -> tuple[Pipeline, dict[str, Any]]:
    meta_path = model_path.with_suffix(".json")
    if not model_path.is_file() or not meta_path.is_file():
        raise ModelLoadError(f"{model_path.stem} is not trained yet. Run: python scripts/train.py")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("feature_schema_version") != FEATURE_SCHEMA_VERSION or meta.get("feature_columns") != FEATURE_COLUMNS:
        raise ModelLoadError(
            f"{model_path.stem} was trained on feature schema v{meta.get('feature_schema_version')}; "
            f"current is v{FEATURE_SCHEMA_VERSION}. Retrain with: python scripts/train.py"
        )
    if meta.get("sklearn_version") != sklearn.__version__:
        raise ModelLoadError(
            f"{model_path.stem} was saved with scikit-learn {meta.get('sklearn_version')}, running {sklearn.__version__}. "
            "Install the pinned version (backend/requirements.txt) or retrain."
        )
    if expected_manifest_hash is not None and meta.get("manifest_hash") != expected_manifest_hash:
        raise ModelLoadError(f"{model_path.stem} was trained on a different split manifest; retrain to match")
    return joblib.load(model_path), meta

"""Build, cache and load the feature table (one Parquet file, ids + label + quality + features).

The label and identifiers live in the same file for convenient joins, but `load_feature_table`
returns them separately from X, and the Parquet schema metadata records which columns are features.
"""

import hashlib
import json
import logging
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy
import skimage
import sklearn

from app.core.damage_scale import DAMAGE_CLASSES
from app.core.events_config import EventsConfig
from app.features.config import FeatureConfig
from app.features.extract import scene_rows
from app.features.schema import (
    FEATURE_COLUMNS,
    FEATURE_GROUPS,
    FEATURE_SCHEMA_VERSION,
    ID_COLUMNS,
    LABEL_COLUMN,
    QUALITY_COLUMNS,
)

_META_KEY = b"aftersight"


class FeatureTableError(Exception):
    """Missing/incompatible feature table or scene cache; the message says how to fix it."""


@dataclass
class FeatureTable:
    ids: pd.DataFrame  # ID_COLUMNS
    X: pd.DataFrame  # FEATURE_COLUMNS only — the model inputs
    y: pd.Series  # damage label
    quality: pd.DataFrame  # QUALITY_COLUMNS (analysis only)
    meta: dict[str, Any]


def table_paths(cfg: FeatureConfig) -> dict[str, Path]:
    out = cfg.resolve(cfg.output_dir)
    stem = f"features_v{FEATURE_SCHEMA_VERSION}"
    return {"table": out / f"{stem}.parquet", "meta": out / f"{stem}.meta.json", "log": out / f"{stem}.log"}


def _cache_index(cfg: FeatureConfig, events_cfg: EventsConfig) -> tuple[list[dict[str, Any]], str]:
    cache_root = cfg.resolve(cfg.scene_cache_dir)
    index_path = cache_root / "index.json"
    if not index_path.is_file():
        raise FeatureTableError(f"Scene cache not found at {cfg.scene_cache_dir}. Run: python scripts/inspect_dataset.py --alignment")
    raw = index_path.read_bytes()
    index = json.loads(raw)
    wanted = set(events_cfg.events)
    scenes = [s for s in index["scenes"] if s["event"] in wanted]
    missing_events = wanted - {s["event"] for s in scenes}
    if missing_events:
        raise FeatureTableError(f"Scene cache has no scenes for {sorted(missing_events)}. Re-run scripts/inspect_dataset.py.")
    return scenes, hashlib.sha256(raw).hexdigest()[:16]


def _extract_one(args: tuple[str, str, FeatureConfig, int, str | None]) -> tuple[str, list[dict[str, Any]], float]:
    scene_dir, scene_id, cfg, connectivity, disaster_type = args
    start = time.perf_counter()
    doc = json.loads((Path(scene_dir) / "scene.json").read_text(encoding="utf-8"))
    rows = scene_rows(Path(scene_dir), doc, cfg, connectivity, disaster_type)
    return scene_id, rows, time.perf_counter() - start


def build_feature_table(cfg: FeatureConfig, events_cfg: EventsConfig, workers: int = 1, force: bool = False,
                        progress: Callable[[int, int], None] | None = None) -> dict[str, Any]:
    """Extract features for every cached scene of the configured events. Returns the metadata dict.

    Skips work (returns the existing metadata with "reused": True) when the table already exists for
    the same schema version, extraction settings and scene-cache index — unless force=True.
    """
    paths = table_paths(cfg)
    scenes, index_hash = _cache_index(cfg, events_cfg)
    fingerprint = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "extraction_hash": cfg.extraction_hash(),
        "scene_cache_index_hash": index_hash,
        "events": sorted(events_cfg.events),
        "connectivity": events_cfg.masks.connectivity,
    }
    if not force and paths["table"].is_file() and paths["meta"].is_file():
        meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
        if meta.get("fingerprint") == fingerprint:
            return {**meta, "reused": True}

    paths["table"].parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("aftersight.features")
    logger.handlers.clear()
    handler = logging.FileHandler(paths["log"], mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.info("extracting %d scenes with %d worker(s); fingerprint %s", len(scenes), workers, fingerprint)

    cache_root = cfg.resolve(cfg.scene_cache_dir)
    jobs = [(str(cache_root / s["scene_id"]), s["scene_id"], cfg, events_cfg.masks.connectivity,
             events_cfg.disaster_types.get(s["event"])) for s in scenes]
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    scene_seconds: list[float] = []
    results = map(_extract_one, jobs) if workers <= 1 else ProcessPoolExecutor(max_workers=workers).map(_extract_one, jobs, chunksize=4)
    for done, (scene_id, scene_rows_, secs) in enumerate(results, start=1):
        rows.extend(scene_rows_)
        scene_seconds.append(secs)
        logger.info("%s buildings=%d secs=%.3f", scene_id, len(scene_rows_), secs)
        if progress:
            progress(done, len(jobs))
    elapsed = time.perf_counter() - started

    columns = [*ID_COLUMNS, LABEL_COLUMN, *QUALITY_COLUMNS, *FEATURE_COLUMNS]
    frame = pd.DataFrame(rows, columns=columns)
    for col in FEATURE_COLUMNS:
        frame[col] = frame[col].astype("float64")
    nan_counts = {c: int(n) for c, n in frame[FEATURE_COLUMNS].isna().sum().items() if n}
    inf_counts = {c: int(n) for c, n in np.isinf(frame[FEATURE_COLUMNS]).sum().items() if n}

    meta: dict[str, Any] = {
        "fingerprint": fingerprint,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": events_cfg.source,
        "feature_columns": FEATURE_COLUMNS,
        "feature_groups": FEATURE_GROUPS,
        "id_columns": ID_COLUMNS,
        "label_column": LABEL_COLUMN,
        "quality_columns": QUALITY_COLUMNS,
        "settings": cfg.model_dump(mode="json"),
        "n_scenes": len(scenes),
        "n_scenes_without_buildings": sum(1 for s in scenes if s["n_buildings"] == 0),
        "n_buildings": len(frame),
        "class_counts": {c: int((frame[LABEL_COLUMN] == c).sum()) for c in DAMAGE_CLASSES},
        "buildings_by_event": {e: int(n) for e, n in frame["event"].value_counts().sort_index().items()},
        "nan_counts": nan_counts,
        "inf_counts": inf_counts,
        "timing": {
            "workers": workers,
            "wall_seconds": round(elapsed, 2),
            "seconds_per_1000_buildings": round(1000 * elapsed / max(len(frame), 1), 3),
            "scene_seconds_median": round(float(np.median(scene_seconds)), 4) if scene_seconds else None,
            "scene_seconds_max": round(float(np.max(scene_seconds)), 4) if scene_seconds else None,
        },
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                     "scikit-image": skimage.__version__, "scikit-learn": sklearn.__version__, "pandas": pd.__version__},
    }
    table = pa.Table.from_pandas(frame, preserve_index=False)
    table = table.replace_schema_metadata({**(table.schema.metadata or {}), _META_KEY: json.dumps(
        {k: meta[k] for k in ("fingerprint", "feature_columns", "id_columns", "label_column", "quality_columns")}).encode()})
    pq.write_table(table, paths["table"], compression="zstd")
    meta["table_bytes"] = paths["table"].stat().st_size
    meta["table_path"] = str(cfg.output_dir / paths["table"].name)
    paths["meta"].write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("wrote %d rows, %d bytes in %.1fs", len(frame), meta["table_bytes"], elapsed)
    handler.close()
    logger.removeHandler(handler)
    return {**meta, "reused": False}


def load_feature_table(path: Path) -> FeatureTable:
    """Load a feature table and split it into ids / X / y / quality. Refuses an incompatible schema."""
    if not path.is_file():
        raise FeatureTableError(f"Feature table not found: {path.name}. Run: python scripts/build_features.py")
    schema_meta = (pq.read_schema(path).metadata or {}).get(_META_KEY)
    if schema_meta is None:
        raise FeatureTableError(f"{path.name} has no AfterSight metadata; rebuild it with scripts/build_features.py")
    info = json.loads(schema_meta)
    version = info["fingerprint"]["feature_schema_version"]
    if version != FEATURE_SCHEMA_VERSION or info["feature_columns"] != FEATURE_COLUMNS:
        raise FeatureTableError(
            f"{path.name} uses feature schema v{version}; this code expects v{FEATURE_SCHEMA_VERSION}. "
            "Rebuild with: python scripts/build_features.py --force"
        )
    frame = pd.read_parquet(path)
    return FeatureTable(
        ids=frame[ID_COLUMNS].copy(),
        X=frame[FEATURE_COLUMNS].copy(),
        y=frame[LABEL_COLUMN].copy(),
        quality=frame[QUALITY_COLUMNS].copy(),
        meta=info,
    )

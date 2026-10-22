"""Extract the per-building feature table from the scene cache (Phase 2).

Usage (repo root, venv active):
    python scripts/build_features.py                 # reuse the table if nothing changed
    python scripts/build_features.py --force         # rebuild
    python scripts/build_features.py --workers 4     # parallel over scenes

Needs the scene cache: python scripts/inspect_dataset.py --alignment
Writes data/processed/features/features_v<N>.parquet, .meta.json and .log.
"""

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401  (adds backend/ to sys.path)
from app.core.damage_scale import DAMAGE_CLASSES
from app.core.events_config import DEFAULT_EVENTS_CONFIG, ConfigError, load_events_config
from app.features.config import DEFAULT_FEATURES_CONFIG, load_feature_config
from app.features.table import FeatureTableError, build_feature_table


def _progress(done: int, total: int) -> None:
    if done == total or done % 200 == 0:
        print(f"  {done}/{total} scenes", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--events-config", type=Path, default=DEFAULT_EVENTS_CONFIG)
    parser.add_argument("--features-config", type=Path, default=DEFAULT_FEATURES_CONFIG)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true", help="rebuild even if the cached table is up to date")
    args = parser.parse_args()
    try:
        events_cfg = load_events_config(args.events_config.resolve())
        cfg = load_feature_config(args.features_config.resolve())
        meta = build_feature_table(cfg, events_cfg, workers=args.workers, force=args.force, progress=_progress)
    except (ConfigError, FeatureTableError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if meta["reused"]:
        print(f"Feature table is up to date ({meta['table_path']}); use --force to rebuild.")
    t = meta["timing"]
    print(f"\nTable: {meta['table_path']} · {meta['table_bytes'] / 1e6:.1f} MB · {meta['n_buildings']:,} buildings "
          f"x {len(meta['feature_columns'])} features from {meta['n_scenes']} scenes")
    print(f"Extraction: {t['wall_seconds']} s wall ({t['workers']} worker(s)) · {t['seconds_per_1000_buildings']} s per 1,000 buildings")
    print("Classes: " + " · ".join(f"{c} {meta['class_counts'][c]:,}" for c in DAMAGE_CLASSES))
    print(f"Buildings by event: {meta['buildings_by_event']}")
    print(f"Missing values: {meta['nan_counts'] or 'none'} · infinite: {meta['inf_counts'] or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

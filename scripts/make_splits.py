"""Build and validate the scene-grouped split manifests for every design in config/training.yaml.

Usage (repo root, venv active):
    python scripts/make_splits.py

Needs the feature table (python scripts/build_features.py). Writes data/splits/<design>_v1.{json,csv}.
Each manifest is checked for scene leakage before it is written; any violation aborts with exit code 1.
"""

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401  (adds backend/ to sys.path)
from app.core.config import REPO_ROOT
from app.core.damage_scale import DAMAGE_CLASSES
from app.core.events_config import ConfigError
from app.features.config import DEFAULT_FEATURES_CONFIG, load_feature_config
from app.features.table import FeatureTableError, load_feature_table, table_paths
from app.training.config import DEFAULT_TRAINING_CONFIG, load_training_config
from app.training.splits import SplitLeakageError, build_manifest, write_manifest

SPLITS_DIR = REPO_ROOT / "data" / "splits"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features-config", type=Path, default=DEFAULT_FEATURES_CONFIG)
    parser.add_argument("--training-config", type=Path, default=DEFAULT_TRAINING_CONFIG)
    parser.add_argument("--out", type=Path, default=SPLITS_DIR)
    args = parser.parse_args()
    try:
        cfg = load_feature_config(args.features_config.resolve())
        training = load_training_config(args.training_config.resolve())
        table = load_feature_table(table_paths(cfg)["table"])
    except (ConfigError, FeatureTableError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    for name, design in training.splits.designs.items():
        try:
            manifest = build_manifest(name, design, table.ids, table.y, training.seed,
                                      training.splits.min_class_count_warning, table.meta["fingerprint"])
        except (SplitLeakageError, ValueError) as exc:
            print(f"{name}: FAILED — {exc}", file=sys.stderr)
            return 1
        paths = write_manifest(manifest, table.ids, args.out)
        print(f"\n{name} — \"{manifest['label']}\"  (hash {manifest['manifest_hash']}, leakage check passed)")
        print(f"  {'split':<6}{'scenes':>7}{'buildings':>11}{'share':>7}" + "".join(f"{c:>14}" for c in DAMAGE_CLASSES) + "   events")
        for split, c in manifest["counts"].items():
            print(f"  {split:<6}{c['scenes']:>7}{c['buildings']:>11,}{c['building_share']:>7.1%}"
                  + "".join(f"{c['classes'][k]:>14,}" for k in DAMAGE_CLASSES) + f"   {len(c['events'])}")
        for warning in manifest["warnings"]:
            print(f"  WARNING {warning}")
        print(f"  -> {paths['json'].relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

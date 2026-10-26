"""Phase 3: train, tune (validation only), evaluate on test once, explain and save every model.

Usage (repo root, venv active):
    python scripts/train.py                           # both designs, both label schemes, all models + analyses
    python scripts/train.py --design within_event --scheme four_class --models knn
    python scripts/train.py --no-optional             # skip the optional random forest
    python scripts/train.py --skip-analyses           # no K-Means / confounding check

Needs: scripts/build_features.py and scripts/make_splits.py. Results go to artifacts/ and outputs/figures/models/.
"""

import argparse
import json
import sys
import time

import _bootstrap  # noqa: F401  (adds backend/ to sys.path)
from app.core.config import REPO_ROOT
from app.core.events_config import ConfigError
from app.evaluation.analysis import run_confounding, run_kmeans
from app.evaluation.metrics import TestSetGuard
from app.features.table import FeatureTableError
from app.training.cli import data_for, load_inputs
from app.training.models import OPTIONAL_MODELS, REQUIRED_MODELS
from app.training.run import OutputRoots, run_design
from app.training.splits import SplitLeakageError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--design", action="append", help="within_event / cross_event (default: all)")
    parser.add_argument("--scheme", action="append", help="four_class / three_class (default: all)")
    parser.add_argument("--models", nargs="+", choices=[*REQUIRED_MODELS, *OPTIONAL_MODELS])
    parser.add_argument("--no-optional", action="store_true")
    parser.add_argument("--skip-analyses", action="store_true")
    args = parser.parse_args()
    try:
        cfg, table, manifests = load_inputs()
    except (ConfigError, FeatureTableError, SplitLeakageError, ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    guard = TestSetGuard()
    roots = OutputRoots()
    started = time.perf_counter()
    designs = args.design or list(cfg.splits.designs)
    schemes = args.scheme or list(cfg.label_schemes)
    for design in designs:
        for scheme in schemes:
            data = data_for(cfg, table, manifests, design, scheme)
            print(f"\n== {design} ({data.design_label}) · {data.scheme_label} · "
                  f"train {len(data.X['train']):,} / val {len(data.X['val']):,} / test {len(data.X['test']):,}")
            run_design(data, cfg, guard, roots, args.models, include_optional=not args.no_optional)

    if not args.skip_analyses:
        for design in designs:
            print(f"\n== K-Means analysis · {design}")
            doc = run_kmeans(data_for(cfg, table, manifests, design, "four_class"), cfg, roots)
            print(f"   chosen k (silhouette) = {doc['chosen_k_by_silhouette']}, elbow heuristic k = {doc['elbow_k_heuristic']}, "
                  f"ARI at chosen k = {doc['per_k'][str(doc['chosen_k_by_silhouette'])]['adjusted_rand_index']}")
        if "within_event" in designs:
            print("\n== Event-confounding check · within_event")
            doc = run_confounding(data_for(cfg, table, manifests, "within_event", "four_class"), cfg, guard, roots)
            print(f"   event accuracy {doc['test_metrics']['accuracy']} (macro-F1 {doc['test_metrics']['macro_f1']}) "
                  f"vs majority-class {doc['majority_class_accuracy']}")

    summary = {"seconds": round(time.perf_counter() - started, 1), "test_evaluations": guard.used}
    (REPO_ROOT / "artifacts" / "metrics" / "run_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"\nDone in {summary['seconds']} s. Test set evaluated {len(guard.used)} times, once per final model. "
          "Print tables with: python scripts/evaluate.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())

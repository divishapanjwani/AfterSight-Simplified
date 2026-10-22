"""Phase 3b experiment: confounding-reduced feature sets (LR + Decision Tree, both split designs).

Usage: python scripts/run_phase3b.py [--variant change_only ...]
Writes only to artifacts/experiments/phase3b/ and outputs/figures/experiments/phase3b/.
Phase 3 models and results are not touched.
"""

import argparse
import sys

import _bootstrap  # noqa: F401
from app.experiments.phase3b import EXPERIMENT_LABEL, run_phase3b
from app.features.variants import VARIANTS
from app.training.cli import load_inputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", action="append", choices=list(VARIANTS))
    args = parser.parse_args()
    cfg, table, manifests = load_inputs()
    print(EXPERIMENT_LABEL)
    summary = run_phase3b(cfg, table, manifests, args.variant)
    print(f"\n{'variant':<30}{'design':<14}{'model':<21}{'val mF1':>8}{'test mF1':>9}{'R major':>8}{'R destr':>8}{'event acc':>10}")
    for r in summary["rows"]:
        print(f"{r['variant']:<30}{r['design']:<14}{r['model']:<21}{r['val_macro_f1']:>8.3f}{r['test_macro_f1']:>9.3f}"
              f"{r['test_recall_major']:>8.3f}{r['test_recall_destroyed']:>8.3f}{r['event_prediction_accuracy_within_event']:>10.3f}")
    print(f"\nTest evaluations in this experiment: {len(summary['test_evaluations'])} (one per new model + one confounding check per variant)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Model comparison & ensemble analysis: SVM, soft voting (SVM + RF + KNN) and GridSearchCV (Model Arena only).

Usage (repo root, venv active): python scripts/run_model_comparison.py [--adaboost-only]
  --adaboost-only  add the optional serial ensemble (AdaBoost over stumps) to an existing summary, without
                   evaluating the other models on test again
Needs the trained KNN, Decision Tree and Random Forest (scripts/train.py). Writes
artifacts/experiments/model_comparison/summary.json. Never retrains or replaces the served models.
"""

import argparse
import sys

import _bootstrap  # noqa: F401
from app.experiments.model_comparison import run_adaboost, run_model_comparison
from app.training.cli import data_for, load_inputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaboost-only", action="store_true")
    args = parser.parse_args()
    cfg, table, manifests = load_inputs()
    designs = {d: data_for(cfg, table, manifests, d, "four_class") for d in ("within_event", "cross_event")}
    doc = run_adaboost(cfg, designs) if args.adaboost_only else run_model_comparison(cfg, designs)
    print(f"-> artifacts/experiments/model_comparison/summary.json ({len(doc['test_evaluations'])} test evaluations)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

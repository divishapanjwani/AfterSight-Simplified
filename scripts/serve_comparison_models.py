"""Serve the model comparison's within-event SVM and soft-voting ensemble in the app, without retraining anything.

Usage (repo root, venv active): python scripts/serve_comparison_models.py
Needs scripts/run_model_comparison.py (the SVM) and scripts/train.py (KNN, RF). Writes svm.{joblib,json} and
soft_voting.{joblib,json} next to the served models. Then calibrate their review thresholds on validation:
  python scripts/calibrate_review.py --models svm soft_voting
"""

import sys

import _bootstrap  # noqa: F401
from app.experiments.model_comparison import serve_comparison_models
from app.training.cli import data_for, load_inputs


def main() -> int:
    cfg, table, manifests = load_inputs()
    serve_comparison_models(cfg, data_for(cfg, table, manifests, cfg.serving.design, cfg.serving.scheme))
    return 0


if __name__ == "__main__":
    sys.exit(main())

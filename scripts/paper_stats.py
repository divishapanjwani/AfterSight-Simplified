"""Scene-level bootstrap confidence intervals for the KNN vs Decision Tree comparison (paper, Table/Fig.).

Uses ONLY the saved per-building test predictions in artifacts/metrics/<design>/four_class/
(<model>_test_predictions.parquet). Nothing is trained or re-predicted.

Buildings in one scene are correlated (same image, same lighting, often the same damage), so the resampling
unit is the TEST SCENE, not the building: each bootstrap replicate draws test scenes with replacement and
recomputes the metric on all buildings of the drawn scenes. KNN and DT use the SAME draws in each replicate,
which gives a paired distribution of the macro-F1 difference (KNN - DT).

Usage (repo root): .venv/bin/python scripts/paper_stats.py [--resamples 2000] [--seed 42]
Writes paper/build/stats.json.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["no-damage", "minor-damage", "major-damage", "destroyed"]
MODELS = ["knn", "decision_tree"]
DESIGNS = ["within_event", "cross_event"]


def scene_confusions(pred: pd.DataFrame, scenes: list[str]) -> np.ndarray:
    """One 4x4 confusion matrix (rows = true, cols = predicted) per test scene, in the order of `scenes`."""
    index = {c: i for i, c in enumerate(CLASSES)}
    out = np.zeros((len(scenes), len(CLASSES), len(CLASSES)), dtype=np.int64)
    position = {s: i for i, s in enumerate(scenes)}
    for scene, t, p in zip(pred["scene_id"], pred["true"], pred["pred"]):
        out[position[scene], index[t], index[p]] += 1
    return out


def macro_f1(cm: np.ndarray) -> float:
    """Unweighted mean of per-class F1 over the classes present in the ground truth (F1 = 0 when TP = 0)."""
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros_like(tp), where=(support + predicted) > 0)
    return float(f1[support > 0].mean())


def accuracy(cm: np.ndarray) -> float:
    return float(np.trace(cm) / cm.sum())


def percentile_ci(values: np.ndarray) -> list[float]:
    return [round(float(np.percentile(values, 2.5)), 4), round(float(np.percentile(values, 97.5)), 4)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = {"method": "scene-level percentile bootstrap on saved test predictions (no retraining)",
              "resamples": args.resamples, "seed": args.seed, "confidence": 0.95, "designs": {}}
    for design in DESIGNS:
        preds = {m: pd.read_parquet(ROOT / "artifacts" / "metrics" / design / "four_class" / f"{m}_test_predictions.parquet")
                 for m in MODELS}
        # both models must be scored on exactly the same buildings
        if not preds["knn"]["sample_id"].equals(preds["decision_tree"]["sample_id"]):
            print(f"{design}: KNN and DT predictions are not aligned", file=sys.stderr)
            return 1
        scenes = sorted(preds["knn"]["scene_id"].unique())
        cms = {m: scene_confusions(preds[m], scenes) for m in MODELS}

        rng = np.random.default_rng(args.seed)
        draws = rng.integers(0, len(scenes), size=(args.resamples, len(scenes)))
        boot = {m: {"macro_f1": np.empty(args.resamples), "accuracy": np.empty(args.resamples)} for m in MODELS}
        for b, idx in enumerate(draws):
            for m in MODELS:
                cm = cms[m][idx].sum(axis=0)
                boot[m]["macro_f1"][b] = macro_f1(cm)
                boot[m]["accuracy"][b] = accuracy(cm)
        diff = boot["knn"]["macro_f1"] - boot["decision_tree"]["macro_f1"]

        entry = {"test_scenes": len(scenes), "test_buildings": int(len(preds["knn"])), "models": {}}
        for m in MODELS:
            full = cms[m].sum(axis=0)
            entry["models"][m] = {
                "macro_f1": round(macro_f1(full), 4), "macro_f1_ci95": percentile_ci(boot[m]["macro_f1"]),
                "accuracy": round(accuracy(full), 4), "accuracy_ci95": percentile_ci(boot[m]["accuracy"]),
            }
        observed = entry["models"]["knn"]["macro_f1"] - entry["models"]["decision_tree"]["macro_f1"]
        entry["paired_difference_knn_minus_dt"] = {
            "macro_f1": round(observed, 4), "ci95": percentile_ci(diff),
            "share_of_resamples_knn_higher": round(float((diff > 0).mean()), 4),
            "ci_excludes_zero": bool(np.percentile(diff, 2.5) > 0 or np.percentile(diff, 97.5) < 0),
        }
        result["designs"][design] = entry

    out = ROOT / "paper" / "build" / "stats.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

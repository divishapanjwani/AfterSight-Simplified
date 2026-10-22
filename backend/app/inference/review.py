"""Review-queue thresholds, chosen on VALIDATION predictions (never on test).

For each model with a confidence output, scan candidate thresholds (validation quantiles of the
confidence) and keep the lowest t whose "confident" set (confidence >= t) reaches the target accuracy.
Buildings under t are suggested for human review. Written to review_thresholds.json next to the models.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np


def choose_threshold(confidence: np.ndarray, correct: np.ndarray, target: float) -> dict[str, Any]:
    confidence, correct = np.asarray(confidence, float), np.asarray(correct, bool)
    candidates = np.unique(np.quantile(confidence, np.linspace(0, 0.99, 100)))
    chosen = None
    for t in candidates:  # ascending: the first that reaches the target flags the fewest buildings
        keep = confidence >= t
        if keep.any() and correct[keep].mean() >= target:
            chosen = float(t)
            break
    if chosen is None:
        return {"threshold": None, "note": f"no threshold reaches {target:.0%} accuracy on validation; review everything",
                "flagged_share_val": 1.0}
    keep = confidence >= chosen
    return {
        "threshold": round(chosen, 4),
        "flagged_share_val": round(float((~keep).mean()), 4),
        "accuracy_at_or_above_val": round(float(correct[keep].mean()), 4),
        "accuracy_below_val": round(float(correct[~keep].mean()), 4) if (~keep).any() else None,
        "n_val": int(len(confidence)),
    }


def write_thresholds(folder: Path, doc: dict[str, Any]) -> Path:
    path = folder / "review_thresholds.json"
    path.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return path


def load_thresholds(folder: Path) -> dict[str, Any] | None:
    path = folder / "review_thresholds.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

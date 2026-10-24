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
    doc, _problem = read_thresholds(folder / "review_thresholds.json")
    return doc


def read_thresholds(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """The thresholds document, or None plus the reason it can't be used (missing, bad JSON, wrong shape).

    A broken file must not crash the API: callers treat None as "no thresholds", which only means
    no building is flagged for low confidence.
    """
    if not path.is_file():
        return None, "the review threshold file has not been created"
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None, "the review threshold file is not valid JSON"
    if not isinstance(doc, dict) or not isinstance(doc.get("models"), dict):
        return None, "the review threshold file has no 'models' section"
    return doc, None


def usable_threshold(doc: dict[str, Any] | None, model: str) -> float | None:
    """The model's threshold if it can be used: a number between 0 and 1 (confidence is in that range).

    Missing entries, null (calibration found no threshold that reaches the target) and non-numbers all
    return None, which means "no low-confidence flags for this model".
    """
    entry = ((doc or {}).get("models") or {}).get(model)
    value = entry.get("threshold") if isinstance(entry, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not 0.0 <= float(value) <= 1.0:
        return None
    return float(value)


def why_unusable(doc: dict[str, Any] | None, problem: str | None, model: str) -> str | None:
    """Plain-language reason a model has no usable threshold (None when it has one)."""
    if usable_threshold(doc, model) is not None:
        return None
    if doc is None:
        return problem or "no review thresholds are available"
    entry = doc["models"].get(model)
    if entry is None:
        return f"{model} is not in the review threshold file (run calibration again)"
    if isinstance(entry, dict) and entry.get("threshold") is None:
        target = doc.get("target_accuracy")
        reach = f"{target:.0%} accuracy" if isinstance(target, (int, float)) else "the target accuracy"
        return f"calibration found no threshold for {model} that reaches {reach} on validation"
    return f"{model} has an unusable threshold value"


def threshold_status(doc: dict[str, Any] | None, problem: str | None, served: list[str],
                     optional: list[str], optional_untrained: list[str]) -> dict[str, Any]:
    """READY / PARTIAL / NOT READY for the review thresholds of the REQUIRED served models.

    Optional models (config: optional: true, e.g. random_forest) never decide the state; they are only
    mentioned in the reason (not trained, or trained without a usable threshold).
    """
    required = [m for m in served if m not in optional]
    with_threshold = [m for m in required if usable_threshold(doc, m) is not None]
    without = [m for m in required if m not in with_threshold]
    notes = []
    for m in served:
        if m in optional_untrained:
            notes.append(f"{m} (optional) not trained")
        elif m in optional and usable_threshold(doc, m) is None:
            notes.append(f"{m} (optional) has no usable threshold")
    # shown on the tile: every served model that is trained (an untrained optional model can't have one)
    listed = [m for m in served if m not in optional_untrained]

    if doc is None or not with_threshold:
        state = "not_ready"
        reason = problem if doc is None else "no required served model has a usable threshold: " + ", ".join(without)
    elif without:
        state = "partial"
        reason = f"usable for {', '.join(with_threshold)}; missing for {', '.join(without)}"
    else:
        state = "ready"
        reason = "usable for every required served model (" + ", ".join(with_threshold) + ")"
    if notes:
        reason += "; " + "; ".join(notes)
    return {"state": state, "reason": reason,
            "thresholds": {m: usable_threshold(doc, m) for m in listed},
            "rule": (doc or {}).get("rule")}

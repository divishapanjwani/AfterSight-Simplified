"""Classification metrics. Macro-F1 is the primary metric.

A class with zero support in the evaluated split gets null precision/recall/F1 and a note, and is
left out of the macro average — it is never reported as a score of 0 or 1.
"""

import time
from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def classification_metrics(y_true: Any, y_pred: Any, classes: list[str]) -> dict[str, Any]:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )
    per_class, notes = {}, []
    for i, cls in enumerate(classes):
        if support[i] == 0:
            per_class[cls] = {"precision": None, "recall": None, "f1": None, "support": 0}
            notes.append(f"{cls} has no examples here; its metrics are undefined and excluded from macro-F1")
        else:
            per_class[cls] = {"precision": round(float(precision[i]), 4), "recall": round(float(recall[i]), 4),
                              "f1": round(float(f1[i]), 4), "support": int(support[i])}
    present = support > 0
    weights = support / support.sum() if support.sum() else support
    return {
        "n": int(len(y_true)),
        "accuracy": round(float((y_true == y_pred).mean()), 4) if len(y_true) else None,
        "macro_f1": round(float(f1[present].mean()), 4) if present.any() else None,
        "weighted_f1": round(float((f1 * weights).sum()), 4) if present.any() else None,
        "per_class": per_class,
        "confusion_matrix": {"labels": classes, "rows_true_cols_pred": confusion_matrix(y_true, y_pred, labels=classes).tolist()},
        "notes": notes,
    }


def timed_predict(predict: Callable[[Any], np.ndarray], X: Any) -> tuple[np.ndarray, float]:
    """Predictions and wall-clock seconds per 1,000 rows."""
    start = time.perf_counter()
    pred = predict(X)
    elapsed = time.perf_counter() - start
    return pred, round(1000 * elapsed / max(len(X), 1), 5)


class TestSetGuard:
    """Enforces 'evaluate on test exactly once per final model'. A second request raises."""

    __test__ = False  # not a pytest test class

    def __init__(self) -> None:
        self.used: dict[str, int] = {}

    def evaluate(self, key: str, fn: Callable[[], Any]) -> Any:
        if key in self.used:
            raise RuntimeError(f"test set already evaluated for {key!r}; tune on validation instead")
        self.used[key] = 1
        return fn()

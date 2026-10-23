"""Feature Lab data, computed on the TRAINING split of the serving design only.

Separation score = eta squared: the share of a feature's variance explained by the damage class
(0 = classes look identical on this feature, 1 = the feature alone separates them perfectly).
"""

from typing import Any

import numpy as np
import pandas as pd

from app.core.damage_scale import DAMAGE_CLASSES
from app.core.errors import NotFound
from app.core.store import Store
from app.features.schema import FEATURE_COLUMNS, FEATURE_GROUPS

GROUP_OF = {f: g for g, names in FEATURE_GROUPS.items() for f in names}


def _train_frame(store: Store) -> tuple[pd.DataFrame, pd.Series]:
    """Features and labels of the TRAINING buildings only (scene assignment from the serving split manifest).

    Validation and test buildings are left out on purpose: showing their distributions here would let us
    "peek" at the data the models are scored on.
    """
    table = store.feature_table
    assignment = store.manifest()["scenes"]
    mask = (table.ids["scene_id"].map(assignment) == "train").to_numpy()
    return table.X[mask], table.y[mask]


def eta_squared(values: np.ndarray, labels: np.ndarray) -> float:
    """eta^2 = between-class sum of squares / total sum of squares (one-way ANOVA effect size)."""
    # A few features can be missing (NaN) for a building; they are skipped, like the per-class percentiles do.
    ok = ~np.isnan(values)
    values, labels = values[ok], labels[ok]
    grand = values.mean()
    total = ((values - grand) ** 2).sum()
    between = sum(((values[labels == c].mean() - grand) ** 2) * (labels == c).sum() for c in np.unique(labels))
    return float(between / total) if total > 0 else 0.0


def class_stats(values: np.ndarray) -> dict[str, Any]:
    """Count, mean and box-plot percentiles (5, 25, 50, 75, 95) of one feature for one damage class."""
    if len(values) == 0:  # a class can be absent from a split; the UI shows "no … buildings"
        return {"n": 0, "mean": None, "p5": None, "q1": None, "median": None, "q3": None, "p95": None}
    p5, q1, median, q3, p95 = np.nanpercentile(values, [5, 25, 50, 75, 95])
    return {"n": int(len(values)), "mean": round(float(np.nanmean(values)), 5),
            "p5": round(float(p5), 5), "q1": round(float(q1), 5), "median": round(float(median), 5),
            "q3": round(float(q3), 5), "p95": round(float(p95), 5)}


def summary(store: Store) -> dict[str, Any]:
    def compute() -> dict[str, Any]:
        X, y = _train_frame(store)
        labels = y.to_numpy()
        features = []
        for name in FEATURE_COLUMNS:
            values = X[name].to_numpy(dtype=float)
            per_class = {}
            for cls in DAMAGE_CLASSES:
                per_class[cls] = class_stats(values[labels == cls])
            features.append({"feature": name, "group": GROUP_OF[name], "separation_eta2": round(eta_squared(values, labels), 5),
                             "per_class": per_class})
        # most separating feature first (a stable sort, so ties keep the schema order)
        features.sort(key=lambda f: -f["separation_eta2"])
        return {"split": "train", "design": store.serving.design, "n": int(len(X)),
                "separation_score": "eta squared — share of the feature's variance explained by damage class",
                "features": features}
    return store.memoised("feature_lab_summary", compute)


def scatter(store: Store, x: str, y_name: str, n: int, seed: int = 42) -> dict[str, Any]:
    for name in (x, y_name):
        if name not in FEATURE_COLUMNS:
            raise NotFound("Unknown feature name.", code="feature_not_found")
    X, y = _train_frame(store)
    frame = pd.DataFrame({"x": X[x].to_numpy(), "y": X[y_name].to_numpy(), "label": y.to_numpy()})
    # Sample the same fraction from every class, so rare classes stay visible in proportion;
    # the fixed seed makes the same request always return the same points.
    if len(frame) > n:
        frame = frame.groupby("label").sample(frac=n / len(frame), random_state=seed)
    return {"split": "train", "x": x, "y": y_name, "n": int(len(frame)),
            "points": [{"x": round(float(a), 5), "y": round(float(b), 5), "label": c} for a, b, c in frame[["x", "y", "label"]].itertuples(index=False)]}

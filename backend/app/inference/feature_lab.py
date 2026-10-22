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
    table = store.feature_table
    assignment = store.manifest()["scenes"]
    mask = (table.ids["scene_id"].map(assignment) == "train").to_numpy()
    return table.X[mask], table.y[mask]


def eta_squared(values: np.ndarray, labels: np.ndarray) -> float:
    ok = ~np.isnan(values)
    values, labels = values[ok], labels[ok]
    grand = values.mean()
    total = ((values - grand) ** 2).sum()
    between = sum(((values[labels == c].mean() - grand) ** 2) * (labels == c).sum() for c in np.unique(labels))
    return float(between / total) if total > 0 else 0.0


def summary(store: Store) -> dict[str, Any]:
    def compute() -> dict[str, Any]:
        X, y = _train_frame(store)
        labels = y.to_numpy()
        features = []
        for name in FEATURE_COLUMNS:
            v = X[name].to_numpy(dtype=float)
            per_class = {}
            for c in DAMAGE_CLASSES:
                cv = v[labels == c]
                q = np.nanpercentile(cv, [5, 25, 50, 75, 95]) if len(cv) else [None] * 5
                per_class[c] = {"n": int(len(cv)), "mean": round(float(np.nanmean(cv)), 5) if len(cv) else None,
                                "p5": q[0], "q1": q[1], "median": q[2], "q3": q[3], "p95": q[4]}
                per_class[c] = {k: (round(float(val), 5) if isinstance(val, (float, np.floating)) else val) for k, val in per_class[c].items()}
            features.append({"feature": name, "group": GROUP_OF[name], "separation_eta2": round(eta_squared(v, labels), 5),
                             "per_class": per_class})
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
    if len(frame) > n:  # stratified sample keeps rare classes visible in proportion
        frame = frame.groupby("label").sample(frac=n / len(frame), random_state=seed)
    return {"split": "train", "x": x, "y": y_name, "n": int(len(frame)),
            "points": [{"x": round(float(a), 5), "y": round(float(b), 5), "label": c} for a, b, c in frame[["x", "y", "label"]].itertuples(index=False)]}

"""Explanations for single buildings and whole models (feed the Building Case File and Model Arena).

All functions take a fitted AfterSight pipeline (SkewLog1p -> SimpleImputer -> StandardScaler -> model)
and report thresholds/values in RAW feature units, so a decision path reads
"edge_density_delta > 0.031" rather than "scaled feature 25 > 0.82".
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from app.features.schema import FEATURE_COLUMNS


def feature_names(pipe: Pipeline) -> list[str]:
    """Input column names the pipeline was fitted on (full v1 schema for models saved before Phase 3b)."""
    log = pipe.named_steps.get("log")
    names = getattr(log, "feature_names_in_", None)
    return list(names) if names is not None else list(FEATURE_COLUMNS)


def _transform(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Apply every step except the final model."""
    return pipe[:-1].transform(X)


def raw_threshold(pipe: Pipeline, column: int, scaled_value: float) -> float:
    """Invert StandardScaler then log1p for one feature (both are monotone, so the inverse is exact)."""
    scaler = pipe.named_steps["scale"]
    unscaled = scaled_value * scaler.scale_[column] + scaler.mean_[column]
    return pipe.named_steps["log"].inverse_value(column, unscaled)


def tree_structure(pipe: Pipeline, classes: list[str], max_depth: int | None = None) -> dict[str, Any]:
    """Decision tree as nested JSON for the UI (depth-limited view if max_depth is given)."""
    tree = pipe.named_steps["model"].tree_
    model_classes = list(pipe.named_steps["model"].classes_)
    names = feature_names(pipe)

    def node(i: int, depth: int) -> dict[str, Any]:
        fractions = tree.value[i][0] / tree.value[i][0].sum()  # class-weighted fractions in the node
        out: dict[str, Any] = {
            "id": int(i),
            "depth": depth,
            "samples": int(tree.n_node_samples[i]),
            "class_fractions": {c: round(float(fractions[model_classes.index(c)]), 4) for c in classes},
            "majority": model_classes[int(np.argmax(fractions))],
        }
        is_leaf = tree.children_left[i] == tree.children_right[i]
        if is_leaf or (max_depth is not None and depth >= max_depth):
            out["leaf"] = bool(is_leaf)
            out["truncated"] = not is_leaf
            return out
        f = int(tree.feature[i])
        out.update({
            "feature": names[f],
            "threshold_raw": round(raw_threshold(pipe, f, float(tree.threshold[i])), 6),
            "threshold_scaled": round(float(tree.threshold[i]), 6),
            "left": node(int(tree.children_left[i]), depth + 1),   # feature <= threshold
            "right": node(int(tree.children_right[i]), depth + 1),  # feature > threshold
        })
        return out

    return {"n_nodes": int(tree.node_count), "max_depth": int(tree.max_depth), "n_leaves": int(tree.n_leaves),
            "classes": classes, "root": node(0, 0)}


def decision_path(pipe: Pipeline, row: pd.DataFrame) -> dict[str, Any]:
    """The tests one building passes through, from root to leaf, in raw units."""
    model = pipe.named_steps["model"]
    z = _transform(pipe, row)
    raw = row.to_numpy(dtype=float)[0]
    tree = model.tree_
    names = feature_names(pipe)
    nodes = model.decision_path(z).indices
    steps = []
    for i in nodes[:-1]:
        f = int(tree.feature[i])
        goes_left = z[0, f] <= tree.threshold[i]
        steps.append({
            "feature": names[f],
            "value_raw": round(float(raw[f]), 6),
            "op": "<=" if goes_left else ">",
            "threshold_raw": round(raw_threshold(pipe, f, float(tree.threshold[i])), 6),
        })
    leaf = nodes[-1]
    fractions = tree.value[leaf][0] / tree.value[leaf][0].sum()
    return {"steps": steps, "leaf_id": int(leaf), "leaf_samples": int(tree.n_node_samples[leaf]),
            "prediction": str(model.classes_[int(np.argmax(fractions))]),
            "leaf_class_fractions": {str(c): round(float(p), 4) for c, p in zip(model.classes_, fractions)}}


def lr_contributions(pipe: Pipeline, row: pd.DataFrame, top_k: int = 5) -> dict[str, Any]:
    """Top features pushing the predicted class: coefficient x standardised value (log-odds units)."""
    model = pipe.named_steps["model"]
    z = _transform(pipe, row)[0]
    scores = model.decision_function(z[None, :])[0]
    k = int(np.argmax(scores))
    contrib = model.coef_[k] * z
    order = np.argsort(-np.abs(contrib))[:top_k]
    names = feature_names(pipe)
    return {
        "predicted_class": str(model.classes_[k]),
        "intercept": round(float(model.intercept_[k]), 6),
        "score": round(float(scores[k]), 6),
        "top_features": [{"feature": names[j], "value_raw": round(float(row.iloc[0, j]), 6),
                          "standardised": round(float(z[j]), 4), "coefficient": round(float(model.coef_[k][j]), 4),
                          "contribution": round(float(contrib[j]), 4)} for j in order],
        "note": "contribution = coefficient x standardised value; positive pushes toward the predicted class",
    }


def lr_coefficients(pipe: Pipeline) -> dict[str, dict[str, float]]:
    model = pipe.named_steps["model"]
    names = feature_names(pipe)
    return {str(c): {names[j]: round(float(model.coef_[i][j]), 5) for j in range(len(names))}
            for i, c in enumerate(model.classes_)}


def knn_neighbours(pipe: Pipeline, row: pd.DataFrame, train_index: pd.DataFrame) -> dict[str, Any]:
    """The k training buildings closest to this one (in standardised feature space), with their labels.

    train_index holds sample_id + label in the exact row order the KNN model was fitted on.
    """
    model = pipe.named_steps["model"]
    distances, idx = model.kneighbors(_transform(pipe, row))
    neighbours = train_index.iloc[idx[0]]
    return {"k": int(model.n_neighbors), "weights": model.weights,
            "neighbours": [{"sample_id": str(s), "label": str(lab), "distance": round(float(d), 4)}
                           for s, lab, d in zip(neighbours["sample_id"], neighbours["label"], distances[0])]}


def permutation_importances(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, repeats: int, seed: int) -> list[dict[str, Any]]:
    """Drop in validation macro-F1 when one feature's values are shuffled (higher = more relied upon)."""
    result = permutation_importance(pipe, X, y, scoring="f1_macro", n_repeats=repeats, random_state=seed, n_jobs=1)
    rows = [{"feature": f, "mean": round(float(m), 5), "std": round(float(s), 5)}
            for f, m, s in zip(X.columns, result.importances_mean, result.importances_std)]
    return sorted(rows, key=lambda r: -r["mean"])

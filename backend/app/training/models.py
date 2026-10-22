"""Model pipelines: skew-aware log1p -> median imputation -> StandardScaler -> classifier.

Every step is fitted on the training split only (it lives inside the sklearn Pipeline), so validation
and test data never influence which features are logged, the imputation medians or the scaling.
"""

from itertools import combinations
from typing import Any

import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from app.features.schema import CHANGE, CONTEXT, FEATURE_COLUMNS

REQUIRED_MODELS = ["baseline", "logistic_regression", "knn", "decision_tree"]
OPTIONAL_MODELS = ["random_forest"]

# What predict_proba means for each model — shown next to any confidence value in the UI.
CONFIDENCE_KIND = {
    "baseline": "none",
    "logistic_regression": "probability (not calibrated)",
    "knn": "vote fraction of neighbours",
    "decision_tree": "class fraction in the leaf",
    "random_forest": "mean of tree leaf fractions",
}

# Candidates for the single-feature baseline: features that measure change between pre and post.
BASELINE_CANDIDATES = [*CHANGE, "gray_std_delta", "edge_density_delta", "glcm_contrast_delta",
                       "glcm_homogeneity_delta", *CONTEXT]


class SkewLog1p(TransformerMixin, BaseEstimator):
    """Apply log1p to features whose TRAINING skewness exceeds `skew_threshold`.

    Only non-negative features are eligible (log1p of a negative value is undefined or misleading);
    skewed signed features are left unchanged and listed in `skipped_signed_`.
    """

    def __init__(self, skew_threshold: float = 2.0):
        self.skew_threshold = skew_threshold

    def fit(self, X: Any, y: Any = None) -> "SkewLog1p":
        values = np.asarray(X, dtype=float)
        self.n_features_in_ = values.shape[1]
        self.skew_ = stats.skew(values, axis=0, nan_policy="omit")
        skewed = np.abs(np.asarray(self.skew_, dtype=float)) > self.skew_threshold
        non_negative = np.nanmin(values, axis=0) >= 0
        self.log_columns_ = np.flatnonzero(skewed & non_negative)
        self.skipped_signed_ = np.flatnonzero(skewed & ~non_negative)
        return self

    def transform(self, X: Any) -> np.ndarray:
        values = np.array(X, dtype=float, copy=True)
        cols = self.log_columns_
        # clip at 0: a non-negative feature can't legitimately go negative at inference time
        values[:, cols] = np.log1p(np.clip(values[:, cols], 0, None))
        return values

    def inverse_value(self, column: int, value: float) -> float:
        """Map a transformed value of one column back to raw units (used to display tree thresholds)."""
        return float(np.expm1(value)) if column in set(self.log_columns_.tolist()) else float(value)


class ThresholdRule(ClassifierMixin, BaseEstimator):
    """Non-ML baseline: ordered thresholds on ONE feature map it to ordered damage classes.

    value*direction < t1 -> classes[0]; t1 <= ... < t2 -> classes[1]; ... (thresholds ascending).
    """

    def __init__(self, feature_index: int = 0, direction: int = 1, thresholds: tuple = (), classes: tuple = ()):
        self.feature_index = feature_index
        self.direction = direction
        self.thresholds = thresholds
        self.classes = classes

    def fit(self, X: Any, y: Any = None) -> "ThresholdRule":
        if len(self.thresholds) != len(self.classes) - 1:
            raise ValueError("need exactly len(classes) - 1 thresholds")
        self.classes_ = np.array(self.classes)
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict(self, X: Any) -> np.ndarray:
        values = np.asarray(X, dtype=float)[:, self.feature_index] * self.direction
        values = np.nan_to_num(values, nan=-np.inf)  # missing -> least severe class
        return self.classes_[np.searchsorted(np.asarray(self.thresholds), values, side="right")]


def _macro_f1_from_codes(true: np.ndarray, pred: np.ndarray, n_classes: int) -> float:
    cm = np.bincount(true * n_classes + pred, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    tp = np.diag(cm).astype(float)
    precision = np.divide(tp, cm.sum(axis=0), out=np.zeros(n_classes), where=cm.sum(axis=0) > 0)
    recall = np.divide(tp, cm.sum(axis=1), out=np.zeros(n_classes), where=cm.sum(axis=1) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(n_classes), where=(precision + recall) > 0)
    present = cm.sum(axis=1) > 0
    return float(f1[present].mean())


def tune_threshold_rule(X_train: Any, y_train: Any, X_val: Any, y_val: Any, classes: list[str],
                        quantile_step: float) -> tuple[ThresholdRule, list[dict[str, Any]]]:
    """Pick the best change feature and its thresholds.

    For each candidate feature: its direction comes from the Spearman correlation with the ordinal
    label on TRAIN; candidate thresholds are TRAIN quantiles; the thresholds and the feature itself are
    chosen by macro-F1 on VALIDATION. Returns the fitted rule and a per-feature search log.
    """
    order = {c: i for i, c in enumerate(classes)}
    ytr = np.asarray([order[c] for c in y_train])
    yva = np.asarray([order[c] for c in y_val])
    Xtr, Xva = np.asarray(X_train, dtype=float), np.asarray(X_val, dtype=float)
    qs = np.arange(quantile_step, 1, quantile_step)
    n_thr = len(classes) - 1
    log, best = [], None
    for name in BASELINE_CANDIDATES:
        j = FEATURE_COLUMNS.index(name)
        rho = stats.spearmanr(Xtr[:, j], ytr, nan_policy="omit").statistic
        direction = 1 if (np.isnan(rho) or rho >= 0) else -1
        grid = np.unique(np.nanquantile(Xtr[:, j] * direction, qs))
        bucket = np.searchsorted(grid, np.nan_to_num(Xva[:, j] * direction, nan=-np.inf), side="right")
        top = (-1.0, ())
        for combo in combinations(range(len(grid)), n_thr):
            pred = sum((bucket > c).astype(int) for c in combo)
            score = _macro_f1_from_codes(yva, pred, len(classes))
            if score > top[0]:
                top = (score, combo)
        thresholds = tuple(float(grid[c]) for c in top[1])
        log.append({"feature": name, "direction": direction, "spearman_train": None if np.isnan(rho) else round(float(rho), 4),
                    "val_macro_f1": round(top[0], 4), "thresholds": thresholds})
        if best is None or top[0] > best[0]:
            best = (top[0], j, direction, thresholds)
    _score, j, direction, thresholds = best
    rule = ThresholdRule(feature_index=j, direction=direction, thresholds=thresholds, classes=tuple(classes))
    return rule.fit(Xtr, None), sorted(log, key=lambda r: -r["val_macro_f1"])


def make_pipeline(name: str, params: dict[str, Any], seed: int, skew_threshold: float, settings: Any = None) -> Pipeline:
    """Build an unfitted pipeline. The baseline has no preprocessing (it works in raw feature units)."""
    if name == "baseline":
        return Pipeline([("model", ThresholdRule(**params))])
    if name == "logistic_regression":
        model = LogisticRegression(class_weight="balanced", max_iter=settings.max_iter if settings else 3000,
                                   random_state=seed, **params)  # lbfgs = multinomial for >2 classes
    elif name == "knn":
        model = KNeighborsClassifier(n_jobs=-1, **params)
    elif name == "decision_tree":
        model = DecisionTreeClassifier(class_weight="balanced", random_state=seed, **params)
    elif name == "random_forest":
        model = RandomForestClassifier(n_estimators=settings.n_estimators if settings else 200,
                                       class_weight="balanced_subsample", n_jobs=-1, random_state=seed, **params)
    else:
        raise ValueError(f"unknown model {name!r}")
    return Pipeline([
        ("log", SkewLog1p(skew_threshold)),
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", model),
    ])


def param_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Cartesian product of a {name: [values]} grid, in a stable order."""
    combos: list[dict[str, Any]] = [{}]
    for key, values in grid.items():
        combos = [{**c, key: v} for c in combos for v in values]
    return combos

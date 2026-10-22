"""Phase 3 orchestration: tune on validation, evaluate on test once, explain, persist.

Layout of what one run writes (design = within_event | cross_event, scheme = four_class | three_class):

    artifacts/models/<design>/<scheme>/<model>.joblib + .json      fitted pipeline + metadata
    artifacts/models/<design>/<scheme>/knn_train_index.parquet      row order of KNN's training set
    artifacts/metrics/<design>/<scheme>/comparison.json|csv         one row per model
    artifacts/metrics/<design>/<scheme>/<model>_test_predictions.parquet
    artifacts/analyses/<design>/<scheme>/<model>_explain.json       permutation importance (+ coefs/tree)
    outputs/figures/models/<design>/<scheme>/<model>_confusion_<val|test>.png
"""

import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.core.config import REPO_ROOT, display_path
from app.evaluation.figures import confusion_figure
from app.evaluation.metrics import TestSetGuard, classification_metrics, timed_predict
from app.explain.explain import lr_coefficients, permutation_importances, tree_structure
from app.features.schema import FEATURE_COLUMNS
from app.features.table import FeatureTable
from app.training.config import LabelScheme, ModelSettings, TrainingConfig
from app.training.models import (
    CONFIDENCE_KIND,
    OPTIONAL_MODELS,
    REQUIRED_MODELS,
    make_pipeline,
    param_grid,
    tune_threshold_rule,
)
from app.training.persistence import save_model
from app.training.splits import split_of_buildings

SPLITS = ("train", "val", "test")


@dataclass
class DesignData:
    design: str
    design_label: str
    scheme: str
    scheme_label: str
    classes: list[str]
    manifest_hash: str
    fingerprint: dict[str, Any]
    X: dict[str, pd.DataFrame]
    y: dict[str, pd.Series]
    ids: dict[str, pd.DataFrame]


@dataclass
class OutputRoots:
    models: Path = REPO_ROOT / "artifacts" / "models"
    metrics: Path = REPO_ROOT / "artifacts" / "metrics"
    analyses: Path = REPO_ROOT / "artifacts" / "analyses"
    figures: Path = REPO_ROOT / "outputs" / "figures" / "models"

    def for_run(self, data: DesignData) -> dict[str, Path]:
        sub = Path(data.design) / data.scheme
        return {"models": self.models / sub, "metrics": self.metrics / sub,
                "analyses": self.analyses / sub, "figures": self.figures / sub}


def design_data(table: FeatureTable, manifest: dict[str, Any], scheme_name: str, scheme: LabelScheme) -> DesignData:
    split = split_of_buildings(manifest, table.ids)
    y_all = table.y.map(scheme.mapping)
    return DesignData(
        design=manifest["design"], design_label=manifest["label"], scheme=scheme_name, scheme_label=scheme.label,
        classes=scheme.classes, manifest_hash=manifest["manifest_hash"], fingerprint=manifest["feature_table_fingerprint"],
        X={s: table.X[split == s].reset_index(drop=True) for s in SPLITS},
        y={s: y_all[split == s].reset_index(drop=True) for s in SPLITS},
        ids={s: table.ids[split == s].reset_index(drop=True) for s in SPLITS},
    )


def per_event_metrics(events: pd.Series, y_true: pd.Series, y_pred: np.ndarray, classes: list[str]) -> dict[str, Any]:
    out = {}
    for event in sorted(events.unique()):
        rows = (events == event).to_numpy()
        out[event] = classification_metrics(y_true[rows], y_pred[rows], classes)
    return out


def _stratified_sample(X: pd.DataFrame, y: pd.Series, n: int, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    if len(X) <= n:
        return X, y
    Xs, _, ys, _ = train_test_split(X, y, train_size=n, stratify=y, random_state=seed)
    return Xs, ys


def _preprocessing_info(pipe: Pipeline) -> dict[str, Any]:
    if "log" not in pipe.named_steps:
        return {"steps": ["none — the threshold rule works in raw feature units"]}
    log = pipe.named_steps["log"]
    return {
        "steps": ["log1p on skewed features (fitted on train)", "median imputation (fitted on train)", "StandardScaler (fitted on train)"],
        "skew_threshold": log.skew_threshold,
        "log1p_features": [FEATURE_COLUMNS[j] for j in log.log_columns_],
        "skewed_signed_features_left_unchanged": [FEATURE_COLUMNS[j] for j in log.skipped_signed_],
        "train_skewness": {FEATURE_COLUMNS[j]: round(float(s), 3) for j, s in enumerate(log.skew_)},
    }


def train_one(name: str, data: DesignData, cfg: TrainingConfig, guard: TestSetGuard, out: dict[str, Path],
              say: Callable[[str], None] = print) -> dict[str, Any]:
    settings = cfg.models.get(name, ModelSettings())
    skew = float(cfg.preprocessing.get("skew_threshold", 2.0))
    Xtr, ytr, Xva, yva, Xte, yte = (data.X["train"], data.y["train"], data.X["val"], data.y["val"],
                                    data.X["test"], data.y["test"])
    search: list[dict[str, Any]] = []

    if name == "baseline":
        start = time.perf_counter()
        rule, search = tune_threshold_rule(Xtr, ytr, Xva, yva, data.classes, settings.quantile_step)
        train_seconds = time.perf_counter() - start  # includes the validation search
        pipe = Pipeline([("model", rule)])
        params = {"feature": FEATURE_COLUMNS[rule.feature_index], "direction": rule.direction,
                  "thresholds_raw": list(rule.thresholds)}
        val_pred = pipe.predict(Xva)
    else:
        best = None
        for grid_params in param_grid(settings.grid):
            pipe = make_pipeline(name, grid_params, cfg.seed, skew, settings)
            start = time.perf_counter()
            pipe.fit(Xtr, ytr)
            fit_seconds = time.perf_counter() - start
            pred = pipe.predict(Xva)
            score = classification_metrics(yva, pred, data.classes)["macro_f1"]
            search.append({"params": grid_params, "val_macro_f1": score, "fit_seconds": round(fit_seconds, 3)})
            say(f"    {name} {grid_params}: val macro-F1 {score:.4f} ({fit_seconds:.1f}s)")
            if best is None or score > best[0]:
                best = (score, pipe, grid_params, fit_seconds, pred)
        _score, pipe, params, train_seconds, val_pred = best

    val_metrics = classification_metrics(yva, val_pred, data.classes)
    test_pred, infer_per_1000 = guard.evaluate(f"{data.design}/{data.scheme}/{name}", lambda: timed_predict(pipe.predict, Xte))
    test_metrics = classification_metrics(yte, test_pred, data.classes)
    events = per_event_metrics(data.ids["test"]["event"], yte, test_pred, data.classes)

    # Explanations (validation data only).
    Xs, ys = _stratified_sample(Xva, yva, cfg.explain.permutation_sample, cfg.seed)
    explain: dict[str, Any] = {"permutation_importance_val": permutation_importances(pipe, Xs, ys, cfg.explain.permutation_repeats, cfg.seed),
                               "permutation_sample": len(Xs)}
    if name == "logistic_regression":
        explain["coefficients_per_class"] = lr_coefficients(pipe)
        explain["coefficient_note"] = "per one standard deviation of the (log-)transformed feature; log-odds units"
    if name == "decision_tree":
        model = pipe.named_steps["model"]
        explain["feature_importances"] = sorted(
            ({"feature": f, "importance": round(float(v), 5)} for f, v in zip(FEATURE_COLUMNS, model.feature_importances_)),
            key=lambda r: -r["importance"])
        (out["models"]).mkdir(parents=True, exist_ok=True)
        (out["models"] / "decision_tree_structure.json").write_text(json.dumps(tree_structure(pipe, data.classes)), encoding="utf-8")
    if name == "knn":
        out["models"].mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"sample_id": data.ids["train"]["sample_id"], "label": ytr}).to_parquet(out["models"] / "knn_train_index.parquet")
    if name == "baseline":
        explain["feature_search_val"] = search

    for d in out.values():
        d.mkdir(parents=True, exist_ok=True)
    (out["analyses"] / f"{name}_explain.json").write_text(json.dumps(explain, indent=1), encoding="utf-8")

    figures = {}
    for split_name, m in (("val", val_metrics), ("test", test_metrics)):
        tag = "VALIDATION" if split_name == "val" else "TEST SET"
        figures[split_name] = display_path(confusion_figure(
            m, f"{name} · {data.design_label} · {data.scheme_label}",
            f"{tag} · n={m['n']:,} · macro-F1 {m['macro_f1']:.3f} · accuracy {m['accuracy']:.3f}",
            out["figures"] / f"{name}_confusion_{split_name}.png"))

    predictions = pd.DataFrame({"sample_id": data.ids["test"]["sample_id"], "scene_id": data.ids["test"]["scene_id"],
                                "event": data.ids["test"]["event"], "true": yte, "pred": test_pred})
    if hasattr(pipe, "predict_proba") and name != "baseline":
        proba = pipe.predict_proba(Xte)
        for j, cls in enumerate(pipe.classes_):
            predictions[f"p_{cls}"] = proba[:, j]
    predictions.to_parquet(out["metrics"] / f"{name}_test_predictions.parquet", index=False)

    metadata = {
        "design": data.design, "design_label": data.design_label, "scheme": data.scheme, "scheme_label": data.scheme_label,
        "classes": data.classes, "optional": name in OPTIONAL_MODELS,
        "hyperparameters": params, "search_on_validation": search if name != "baseline" else "see explain file",
        "preprocessing": _preprocessing_info(pipe), "seed": cfg.seed,
        "manifest_hash": data.manifest_hash, "feature_table_fingerprint": data.fingerprint,
        "n": {s: len(data.X[s]) for s in SPLITS},
        "train_seconds": round(train_seconds, 3),
        "inference_seconds_per_1000_test": infer_per_1000,
        "confidence": CONFIDENCE_KIND[name],
        "metrics": {"val": val_metrics, "test": test_metrics, "test_per_event": events},
        "test_evaluations": guard.used[f"{data.design}/{data.scheme}/{name}"],
        "figures": figures,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    save_model(pipe, metadata, out["models"], name)
    return {"model": name, "optional": name in OPTIONAL_MODELS, "hyperparameters": params,
            "val_macro_f1": val_metrics["macro_f1"], "test_macro_f1": test_metrics["macro_f1"],
            "test_accuracy": test_metrics["accuracy"], "test_weighted_f1": test_metrics["weighted_f1"],
            "test_recall": {c: v["recall"] for c, v in test_metrics["per_class"].items()},
            "train_seconds": round(train_seconds, 3), "inference_seconds_per_1000": infer_per_1000,
            "top_permutation_features": [r["feature"] for r in explain["permutation_importance_val"][:5]]}


def run_design(data: DesignData, cfg: TrainingConfig, guard: TestSetGuard, roots: OutputRoots,
               models: list[str] | None = None, include_optional: bool = True,
               say: Callable[[str], None] = print) -> list[dict[str, Any]]:
    """Train every required model, then (optionally) the optional ones, for one design x scheme."""
    out = roots.for_run(data)
    names = models or [*REQUIRED_MODELS, *(OPTIONAL_MODELS if include_optional else [])]
    rows = []
    for name in names:
        say(f"  [{data.design} / {data.scheme}] {name}")
        rows.append(train_one(name, data, cfg, guard, out, say))
    write_comparison(rows, data, out["metrics"])
    return rows


def write_comparison(rows: list[dict[str, Any]], data: DesignData, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    doc = {"design": data.design, "design_label": data.design_label, "scheme": data.scheme,
           "scheme_label": data.scheme_label, "classes": data.classes, "primary_metric": "test macro-F1",
           "manifest_hash": data.manifest_hash, "models": rows}
    (folder / "comparison.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
    with (folder / "comparison.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["model", "optional", "val_macro_f1", "test_macro_f1", "test_accuracy", "test_weighted_f1",
                         *[f"test_recall_{c}" for c in data.classes], "train_seconds", "inference_seconds_per_1000"])
        for r in rows:
            writer.writerow([r["model"], r["optional"], r["val_macro_f1"], r["test_macro_f1"], r["test_accuracy"],
                             r["test_weighted_f1"], *[r["test_recall"][c] for c in data.classes],
                             r["train_seconds"], r["inference_seconds_per_1000"]])

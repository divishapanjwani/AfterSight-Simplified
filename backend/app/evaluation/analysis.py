"""Analyses that are NOT damage predictions: K-Means structure and the event-confounding check."""

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.pipeline import Pipeline

from app.core.config import REPO_ROOT, display_path
from app.evaluation.figures import confusion_figure, heatmap, kmeans_curves
from app.evaluation.metrics import TestSetGuard, classification_metrics
from app.training.config import TrainingConfig
from app.training.models import make_pipeline, param_grid
from app.training.run import DesignData, OutputRoots


def _preprocess(data: DesignData, cfg: TrainingConfig) -> Pipeline:
    """Same preprocessing as the classifiers (log1p skewed -> impute -> scale), fitted on train only."""
    pipe = make_pipeline("logistic_regression", {}, cfg.seed, float(cfg.preprocessing.get("skew_threshold", 2.0)))
    return pipe[:-1].fit(data.X["train"])


def run_kmeans(data: DesignData, cfg: TrainingConfig, roots: OutputRoots) -> dict[str, Any]:
    """Cluster the TRAINING buildings for several k; compare clusters with the (4-class) labels.

    Clusters are an analysis of structure in the features. They are never used or shown as predictions.
    """
    prep = _preprocess(data, cfg)
    Z = prep.transform(data.X["train"])
    labels = data.y["train"].to_numpy()
    rng = np.random.default_rng(cfg.seed)
    sil_idx = rng.choice(len(Z), size=min(cfg.kmeans.silhouette_sample, len(Z)), replace=False)
    pca_idx = rng.choice(len(Z), size=min(cfg.kmeans.pca_sample, len(Z)), replace=False)
    pca = PCA(n_components=2, random_state=cfg.seed).fit(Z)
    coords = pca.transform(Z[pca_idx])

    per_k: dict[str, Any] = {}
    inertia, silhouette = [], []
    for k in cfg.kmeans.k_values:
        start = time.perf_counter()
        km = KMeans(n_clusters=k, n_init=10, random_state=cfg.seed).fit(Z)
        seconds = time.perf_counter() - start
        clusters = km.labels_
        sil = float(silhouette_score(Z[sil_idx], clusters[sil_idx]))
        contingency = [[int(((clusters == c) & (labels == cls)).sum()) for cls in data.classes] for c in range(k)]
        inertia.append(float(km.inertia_))
        silhouette.append(sil)
        per_k[str(k)] = {
            "inertia": round(float(km.inertia_), 2),
            "silhouette_sampled": round(sil, 4),
            "adjusted_rand_index": round(float(adjusted_rand_score(labels, clusters)), 4),
            "cluster_sizes": np.bincount(clusters, minlength=k).tolist(),
            "contingency_rows_cluster_cols_class": contingency,
            "fit_seconds": round(seconds, 2),
            "pca_points": [{"x": round(float(x), 3), "y": round(float(y), 3), "cluster": int(clusters[i]),
                            "label": str(labels[i])} for (x, y), i in zip(coords, pca_idx)],
        }
    ks = cfg.kmeans.k_values
    chosen = ks[int(np.argmax(silhouette))]
    second_diff = np.diff(inertia, 2)  # elbow heuristic: largest bend in the inertia curve
    elbow = ks[int(np.argmax(second_diff)) + 1] if len(second_diff) else ks[0]

    figures_dir = roots.figures / "kmeans" / data.design
    figs = [kmeans_curves(ks, inertia, silhouette, chosen, f"K-Means on training buildings · {data.design_label}",
                          figures_dir / "kmeans_elbow_silhouette.png")]
    for k in sorted({chosen, len(data.classes)}):
        figs.append(heatmap(np.array(per_k[str(k)]["contingency_rows_cluster_cols_class"]),
                            [f"cluster {c}" for c in range(k)], data.classes,
                            f"K-Means clusters vs damage labels (k={k}) · analysis only, not predictions",
                            f"{data.design_label} · train · ARI {per_k[str(k)]['adjusted_rand_index']:.3f} · row % = share of cluster",
                            "ground-truth damage class", "cluster", figures_dir / f"kmeans_contingency_k{k}.png"))
    doc = {
        "design": data.design, "design_label": data.design_label, "fitted_on": "training split, preprocessed as for the classifiers",
        "classes": data.classes, "k_values": ks, "chosen_k_by_silhouette": chosen, "elbow_k_heuristic": elbow,
        "pca_explained_variance": [round(float(v), 4) for v in pca.explained_variance_ratio_],
        "pca_note": "2-D PCA projection of a random training sample — for visualisation only",
        "per_k": per_k,
        "interpretation_note": "Clusters group buildings with similar features; ARI near 0 means they do not line up with "
                               "damage classes. Clusters are never used as damage predictions.",
        "figures": [display_path(p) for p in figs],
    }
    out = roots.analyses / "kmeans"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{data.design}.json").write_text(json.dumps(doc), encoding="utf-8")
    return doc


def run_confounding(data: DesignData, cfg: TrainingConfig, guard: TestSetGuard, roots: OutputRoots) -> dict[str, Any]:
    """Can the 37 features tell WHICH DISASTER a building comes from? (within-event split only)

    Event is the target here — it is never an input to any damage model.
    """
    y = {s: data.ids[s]["event"] for s in ("train", "val", "test")}
    events = sorted(y["train"].unique())
    best = None
    search = []
    for params in param_grid(cfg.confounding.grid):
        pipe = make_pipeline("logistic_regression", params, cfg.seed, float(cfg.preprocessing.get("skew_threshold", 2.0)))
        pipe.fit(data.X["train"], y["train"])
        score = classification_metrics(y["val"], pipe.predict(data.X["val"]), events)["macro_f1"]
        search.append({"params": params, "val_macro_f1": score})
        if best is None or score > best[0]:
            best = (score, pipe, params)
    _s, pipe, params = best
    test_pred = guard.evaluate(f"{data.design}/confounding/event", lambda: pipe.predict(data.X["test"]))
    metrics = classification_metrics(y["test"], test_pred, events)
    majority = y["train"].value_counts().idxmax()
    majority_acc = float((y["test"] == majority).mean())
    fig = confusion_figure(metrics, f"Event-confounding check · predict EVENT from the 37 features · {data.design_label}",
                           f"TEST SET · logistic regression · accuracy {metrics['accuracy']:.3f} vs majority-class {majority_acc:.3f}",
                           roots.figures / "confounding" / f"{data.design}_event_confusion_test.png")
    doc = {
        "design": data.design, "design_label": data.design_label, "target": "event (analysis only)",
        "model": "logistic_regression", "hyperparameters": params, "search_on_validation": search,
        "test_metrics": metrics, "majority_class": majority, "majority_class_accuracy": round(majority_acc, 4),
        "chance_accuracy_uniform": round(1 / len(events), 4),
        "figure": display_path(fig),
    }
    out = roots.analyses / "confounding"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{data.design}.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return doc

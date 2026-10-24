"""Per-scene analysis with one serving model: predictions, confidence, review flags, explanations.

Results are cached on disk per (scene, model version) in artifacts/analyses/runs/<analysis_id>.json.
Analysis only ever LOADS saved models; it never trains anything.
"""

import hashlib
import json
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np

from app.core.errors import Conflict, NotFound
from app.core.store import ANALYSIS_ID, Store
from app.evaluation.metrics import classification_metrics
from app.explain.explain import decision_path, knn_neighbours, lr_contributions
from app.inference.review import usable_threshold
from app.inference.scenes import polygon_parts, provenance
from app.training.models import CONFIDENCE_KIND

ANALYSIS_VERSION = 1
MODEL_LABELS = {"logistic_regression": "Logistic Regression", "knn": "K-Nearest Neighbours",
                "decision_tree": "Decision Tree", "random_forest": "Random Forest (optional)", "baseline": "Threshold baseline"}


def _analysis_id(scene_id: str, model: str, meta: dict[str, Any]) -> str:
    key = f"{ANALYSIS_VERSION}|{scene_id}|{model}|{meta['created_at']}|{meta['manifest_hash']}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _model_summary(name: str, meta: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "label": MODEL_LABELS[name], "design": meta["design"], "design_label": meta["design_label"],
            "scheme": meta["scheme"], "trained_at": meta["created_at"], "hyperparameters": meta["hyperparameters"],
            "test_macro_f1": meta["metrics"]["test"]["macro_f1"], "optional": meta.get("optional", False)}


def predict_rows(pipe, X) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    proba = pipe.predict_proba(X)
    pred = pipe.classes_[proba.argmax(axis=1)]
    return pred, proba.max(axis=1), proba


def run_analysis(store: Store, scene_id: str, model_name: str) -> dict[str, Any]:
    sid = store.valid_scene_id(scene_id)
    name = store.valid_model_name(model_name)
    pipe, meta = store.model(name)
    analysis_id = _analysis_id(sid, name, meta)
    path = store.paths.analysis_runs / f"{analysis_id}.json"
    if path.is_file():
        return {**json.loads(path.read_text(encoding="utf-8")), "cached": True}

    start = time.perf_counter()
    doc = store.scene_doc(sid)
    table = store.feature_table
    rows = store.rows_by_scene.get(sid)
    thresholds = store.review_thresholds() or {}
    threshold = usable_threshold(thresholds, name)  # None -> no low-confidence flags for this model
    by_id = {b["building_id"]: b for b in doc["buildings"]}
    buildings, counts_true, counts_pred = [], {}, {}
    classes = list(pipe.classes_)
    if rows is not None and len(rows):
        X, ids, y = table.X.iloc[rows], table.ids.iloc[rows], table.y.iloc[rows].to_numpy()
        pred, conf, proba = predict_rows(pipe, X)
        for i, uid in enumerate(ids["sample_id"]):
            b = by_id[uid]
            reasons = [f"confidence {conf[i]:.2f} is below the review threshold {threshold:.2f} (chosen on validation)"] \
                if threshold is not None and conf[i] < threshold else []
            buildings.append({
                "building_uid": uid, "true_label": y[i], "predicted": str(pred[i]), "correct": bool(pred[i] == y[i]),
                "confidence": round(float(conf[i]), 4), "scores": {c: round(float(p), 4) for c, p in zip(classes, proba[i])},
                "needs_review": bool(reasons), "review_reasons": reasons, "area_px": b["area_px"],
                "mixed_label": b["mixed_label"], "touches_border": b["touches_border"], "polygon": polygon_parts(b["wkt"]),
            })
        scene_metrics = classification_metrics(y, pred, meta["classes"])
    else:
        scene_metrics = None
    for b in buildings:
        counts_true[b["true_label"]] = counts_true.get(b["true_label"], 0) + 1
        counts_pred[b["predicted"]] = counts_pred.get(b["predicted"], 0) + 1

    result = {
        "analysis_id": analysis_id, "cached": False,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scene_id": sid, "event": doc["event"], "disaster_type": store.disaster_type(doc["event"]),
        "model": _model_summary(name, meta), "confidence_kind": CONFIDENCE_KIND[name],
        "review_threshold": threshold, "review_rule": thresholds.get("rule"),
        "provenance": provenance(store, sid),
        "counts": {"ground_truth": {c: counts_true.get(c, 0) for c in meta["classes"]},
                   "predicted": {c: counts_pred.get(c, 0) for c in meta["classes"]}},
        "scene_metrics": scene_metrics,
        "seconds": round(time.perf_counter() - start, 4),
        "buildings": buildings,
    }
    store.paths.analysis_runs.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result), encoding="utf-8")
    tmp.replace(path)
    return result


def load_analysis(store: Store, analysis_id: str) -> dict[str, Any]:
    if not ANALYSIS_ID.match(analysis_id or ""):
        raise NotFound("Unknown analysis id.", code="analysis_not_found")
    path = store.paths.analysis_runs / f"{analysis_id}.json"
    if not path.is_file():
        raise NotFound("Unknown analysis id.", code="analysis_not_found")
    return json.loads(path.read_text(encoding="utf-8"))


def crop_urls(building_uid: str) -> dict[str, str]:
    return {phase: f"/api/buildings/{building_uid}/crop/{phase}" for phase in ("pre", "post")}


def feature_row(store: Store, building_uid: str):
    """The building's one-row feature frame (a DataFrame, because the pipeline needs column names)."""
    table = store.feature_table

    def build_row_lookup() -> dict[str, int]:
        return {sample_id: i for i, sample_id in enumerate(table.ids["sample_id"])}

    # the lookup over all ~71k rows is built once and kept in memory
    row_of_sample = store.memoised("row_of_sample", build_row_lookup)
    return table.X.iloc[[row_of_sample[building_uid]]]


def explain_building(store: Store, analysis_id: str, building_uid: str) -> dict[str, Any]:
    analysis = load_analysis(store, analysis_id)
    building = next((b for b in analysis["buildings"] if b["building_uid"] == building_uid), None)
    if building is None:
        raise NotFound("That building is not part of this analysis.", code="building_not_found")
    name = analysis["model"]["name"]
    pipe, meta = store.model(name)
    # explaining an old analysis with a newer model would describe a prediction it never made
    if meta["created_at"] != analysis["model"]["trained_at"]:
        raise Conflict("The model was retrained after this analysis; run the analysis again.", code="analysis_stale")
    row = feature_row(store, building_uid)
    if name == "logistic_regression":
        kind, detail = "lr_contributions", lr_contributions(pipe, row, top_k=8)
    elif name == "decision_tree":
        kind, detail = "decision_path", decision_path(pipe, row)
    elif name == "knn":
        kind, detail = "nearest_neighbours", knn_neighbours(pipe, row, store.knn_index())
        for neighbour in detail["neighbours"]:
            neighbour["crop_urls"] = crop_urls(neighbour["sample_id"])
            neighbour["split"] = "train"  # KNN only ever searches the training buildings
    else:
        kind, detail = "none", {"note": "Random forests have no compact per-building explanation here; "
                                        "see the global feature importances below."}
    explain = store.read_json(f"analyses/{meta['design']}/{meta['scheme']}/{name}_explain.json") or {}
    return {
        "analysis_id": analysis_id, "building_uid": building_uid, "model": name, "kind": kind,
        "predicted": building["predicted"], "true_label": building["true_label"],
        "confidence": building["confidence"], "confidence_kind": analysis["confidence_kind"], "detail": detail,
        "global_top_features": explain.get("permutation_importance_val", [])[:5],
        "crop_urls": crop_urls(building_uid),
    }


def list_analyses(store: Store, limit: int) -> dict[str, Any]:
    """Most recent cached analyses (newest first) as short summaries for the Reports page."""
    folder = store.paths.analysis_runs
    files = sorted(folder.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True) if folder.is_dir() else []
    items = []
    for path in files[:limit]:
        a = json.loads(path.read_text(encoding="utf-8"))
        m = a["scene_metrics"] or {}
        items.append({
            "analysis_id": a["analysis_id"], "scene_id": a["scene_id"], "event": a["event"], "created_at": a["created_at"],
            "model": a["model"]["name"], "model_label": a["model"]["label"], "split": a["provenance"]["split"],
            "warning": a["provenance"]["warning"], "n_buildings": len(a["buildings"]),
            "n_review": sum(b["needs_review"] for b in a["buildings"]),
            "accuracy": m.get("accuracy"), "macro_f1": m.get("macro_f1"),
            "csv_url": f"/api/analysis/{a['analysis_id']}/export.csv", "pdf_url": f"/api/analysis/{a['analysis_id']}/report.pdf",
        })
    return {"total": len(files), "items": items}

"""Review queue: buildings suggested for human review — low confidence or the serving models disagree.

Framed as suggestions, never as "errors". The confidence threshold per model was chosen on validation
(scripts/calibrate_review.py). Predictions for a split are computed once per model and kept in memory.
"""

from typing import Any

import numpy as np

from app.core.errors import AppError
from app.core.store import Store
from app.inference.analysis import predict_rows


def _split_predictions(store: Store, name: str, split: str) -> dict[str, Any]:
    def compute() -> dict[str, Any]:
        table = store.feature_table
        assignment = store.manifest()["scenes"]
        mask = table.ids["scene_id"].map(assignment).fillna("not_in_manifest").to_numpy() == split
        pipe, _meta = store.model(name)
        pred, conf, _proba = predict_rows(pipe, table.X[mask])
        return {"rows": np.flatnonzero(mask), "pred": pred, "conf": conf}
    return store.memoised(f"split_predictions/{name}/{split}", compute)


def review_queue(store: Store, model: str, split: str, offset: int, limit: int) -> dict[str, Any]:
    name = store.valid_model_name(model)
    store.model(name)  # raises ModelNotTrained with the fix command
    main = _split_predictions(store, name, split)
    others: dict[str, dict[str, Any]] = {}
    for other in store.serving.models:
        if other == name:
            continue
        try:
            others[other] = _split_predictions(store, other, split)
        except AppError:
            continue  # untrained optional models simply don't take part in the disagreement check
    thresholds = store.review_thresholds() or {}
    threshold = (thresholds.get("models", {}).get(name) or {}).get("threshold")

    table = store.feature_table
    ids, y = table.ids.iloc[main["rows"]], table.y.iloc[main["rows"]].to_numpy()
    low = main["conf"] < threshold if threshold is not None else np.zeros(len(y), bool)
    disagree = np.zeros(len(y), bool)
    for o in others.values():
        disagree |= o["pred"] != main["pred"]
    flagged = np.flatnonzero(low | disagree)
    flagged = flagged[np.argsort(main["conf"][flagged], kind="stable")]  # least confident first
    items = []
    for i in flagged[offset:offset + limit]:
        reasons = []
        if low[i]:
            reasons.append(f"low confidence ({main['conf'][i]:.2f} < {threshold:.2f})")
        disagreeing = {k: str(o["pred"][i]) for k, o in others.items() if o["pred"][i] != main["pred"][i]}
        if disagreeing:
            reasons.append("other models disagree: " + ", ".join(f"{k} says {v}" for k, v in disagreeing.items()))
        items.append({"building_uid": ids["sample_id"].iloc[i], "scene_id": ids["scene_id"].iloc[i], "event": ids["event"].iloc[i],
                      "predicted": str(main["pred"][i]), "true_label": str(y[i]), "confidence": round(float(main["conf"][i]), 4),
                      "reasons": reasons, "other_models": {k: str(o["pred"][i]) for k, o in others.items()}})
    return {"model": name, "split": split, "threshold": threshold, "threshold_rule": thresholds.get("rule"),
            "total_buildings": int(len(y)), "total_flagged": int(len(flagged)),
            "flagged_low_confidence": int(low.sum()), "flagged_disagreement_only": int((disagree & ~low).sum()), "items": items,
            "offset": offset, "limit": limit}

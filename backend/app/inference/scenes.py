"""Scene listing and detail, including the split each scene belongs to under the serving design."""

import math
from collections import Counter
from typing import Any

import pandas as pd
from shapely import get_parts, wkt as shapely_wkt

from app.core.damage_scale import DAMAGE_CLASSES
from app.core.errors import AppError
from app.core.store import Store

PROVENANCE_TEXT = {
    "train": ("This scene was used for training. Accuracy here is optimistic.",
              "Training scene: the models have seen these buildings and their labels."),
    "val": ("This scene was used to choose model settings (validation). Accuracy here is somewhat optimistic.",
            "Validation scene: used for tuning, not for fitting."),
    "test": (None, "Held-out test scene: never used for training or tuning."),
    "not_in_manifest": (None, "No labelled buildings, so this scene is not part of any split."),
}
SORTS = {"scene_id": ("scene_id", True), "-n_buildings": ("n_buildings", False), "-destroyed": ("destroyed", False)}


def provenance(store: Store, scene_id: str) -> dict[str, Any]:
    design = store.serving.design
    split = store.split_of_scene(scene_id, design)
    warning, note = PROVENANCE_TEXT[split]
    return {"split": split, "design": design, "design_label": store.training_config.splits.designs[design].label,
            "warning": warning, "note": note}


def polygon_parts(wkt: str) -> list[list[list[tuple[float, float]]]]:
    geom = shapely_wkt.loads(wkt)
    parts = []
    for poly in get_parts(geom):
        rings = [poly.exterior, *poly.interiors]
        parts.append([[(round(x, 1), round(y, 1)) for x, y in ring.coords] for ring in rings])
    return parts


def _summaries(store: Store) -> pd.DataFrame:
    """One row per cached scene: GT counts and quality flags from the feature table, split from the manifest."""
    index = store.scene_index()
    frame = pd.DataFrame(index.values())[["scene_id", "event", "xbd_split", "n_buildings"]]
    table = store.feature_table
    labelled = pd.concat([table.ids[["scene_id"]], table.y.rename("label"), table.quality[["mixed_label", "touches_border"]]], axis=1)
    counts = labelled.pivot_table(index="scene_id", columns="label", aggfunc="size", fill_value=0)
    counts = counts.reindex(columns=list(DAMAGE_CLASSES), fill_value=0)
    flags = labelled.groupby("scene_id")[["mixed_label", "touches_border"]].sum()
    frame = frame.join(counts, on="scene_id").join(flags, on="scene_id").fillna(0)
    for col in [*DAMAGE_CLASSES, "mixed_label", "touches_border"]:
        frame[col] = frame[col].astype(int)
    try:
        assignment = store.manifest()["scenes"]
    except AppError:
        assignment = {}
    frame["split"] = frame["scene_id"].map(assignment).fillna("not_in_manifest")
    frame["disaster_type"] = frame["event"].map(store.disaster_type)
    return frame.sort_values("scene_id").reset_index(drop=True)


def summaries(store: Store) -> pd.DataFrame:
    return store.memoised("scene_summaries", lambda: _summaries(store))


def _summary_row(store: Store, row: dict[str, Any]) -> dict[str, Any]:
    q = store.quality.get(row["scene_id"], {})
    shift = q.get("alignment_shift_px")
    shift = None if shift is None or (isinstance(shift, float) and math.isnan(shift)) else float(shift)
    flag = q.get("alignment_flag")
    flag = None if flag is None or (isinstance(flag, float) and math.isnan(flag)) else bool(flag)
    warnings = []
    if flag:
        warnings.append(f"Pre/post images may be misaligned (estimated shift {shift} px).")
    if row["mixed_label"]:
        warnings.append(f"{row['mixed_label']} footprints mix damage values (possibly merged buildings).")
    if row["n_buildings"] == 0:
        warnings.append("No labelled buildings in this scene.")
    return {
        "scene_id": row["scene_id"], "event": row["event"], "disaster_type": row["disaster_type"],
        "xbd_split": row["xbd_split"], "split": row["split"], "n_buildings": int(row["n_buildings"]),
        "ground_truth_counts": {c: int(row[c]) for c in DAMAGE_CLASSES},
        "quality": {"alignment_shift_px": shift, "alignment_flag": flag,
                    "n_mixed_label": int(row["mixed_label"]), "n_touching_border": int(row["touches_border"])},
        "warnings": warnings,
    }


def list_scenes(store: Store, event: str | None, disaster_type: str | None, split: str | None,
                has_buildings: bool | None, sort: str, page: int, page_size: int) -> dict[str, Any]:
    frame = summaries(store)
    facets = {"event": dict(Counter(frame["event"])), "split": dict(Counter(frame["split"])),
              "disaster_type": dict(Counter(frame["disaster_type"].fillna("unknown")))}
    if event:
        frame = frame[frame["event"] == event]
    if disaster_type:
        frame = frame[frame["disaster_type"] == disaster_type]
    if split:
        frame = frame[frame["split"] == split]
    if has_buildings is not None:
        frame = frame[(frame["n_buildings"] > 0) == has_buildings]
    column, ascending = SORTS[sort]
    frame = frame.sort_values([column, "scene_id"], ascending=[ascending, True])
    start = (page - 1) * page_size
    design = store.serving.design
    return {
        "items": [_summary_row(store, r) for r in frame.iloc[start:start + page_size].to_dict("records")],
        "total": int(len(frame)), "page": page, "page_size": page_size,
        "split_design": design, "split_design_label": store.training_config.splits.designs[design].label,
        "facets": facets,
    }


def scene_detail(store: Store, scene_id: str) -> dict[str, Any]:
    sid = store.valid_scene_id(scene_id)
    frame = summaries(store)
    row = frame[frame["scene_id"] == sid].iloc[0].to_dict()
    doc = store.scene_doc(sid)
    buildings = [{
        "building_uid": b["building_id"], "true_label": b["damage"], "area_px": b["area_px"],
        "label_purity": b["label_purity"], "mixed_label": b["mixed_label"], "touches_border": b["touches_border"],
        "polygon": polygon_parts(b["wkt"]),
    } for b in doc["buildings"]]
    return {
        **_summary_row(store, row),
        "width": doc["width"], "height": doc["height"],
        "provenance": provenance(store, sid),
        "n_dropped": len(doc["dropped"]),
        "dropped_reasons": dict(Counter(d["reason"] for d in doc["dropped"])),
        "footprint_source": doc["footprint_source"],
        "image_urls": {f"{phase}_{size}": f"/api/scenes/{sid}/image/{phase}?size={size}"
                       for phase in ("pre", "post") for size in ("thumb", "view")},
        "buildings": buildings,
    }

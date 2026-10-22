"""Phase 3b feature-set variants for the event-confounding experiment.

The Phase 3 event check showed the 37 features identify the disaster with 83% accuracy, largely from
scene appearance (absolute roof/ground colour, sensor, vegetation). These variants try to remove that:

  change_only        17 features: the change group, the four texture deltas, the context ring
                     (ring change and building-vs-ring contrast change) and shape. Every absolute
                     pre- or post-disaster colour/texture level is removed.
  scene_normalised   all 37 features, each image feature re-expressed as
                     (building value - median of that feature over all buildings in the same scene).
                     Shape features (area_px, compactness) stay raw: they describe the building, not
                     the image, and "bigger than its neighbours" is not what they are for.
  change_only_scene_normalised   both: the 17 change-only features, scene-normalised the same way.

Difference, not ratio: many features are signed or near zero (deltas, ring_dmean), where a ratio is
undefined or explodes; a difference keeps the feature's units and cancels any shift shared by the whole
scene (lighting, sensor, season, regional roof colour).

Leak-free: normalisation uses only feature values of buildings in the same scene — never labels — and
every scene lies entirely inside one split. At analysis time the app processes whole scenes, so the
same scene median is available there. Caveat: if almost every building in a scene is destroyed, the
scene median itself is "destroyed-looking" and the normalised values shrink toward zero.
"""

import pandas as pd

from app.features.schema import CHANGE, CONTEXT, FEATURE_COLUMNS, SHAPE

TEXTURE_DELTAS = ["gray_std_delta", "edge_density_delta", "glcm_contrast_delta", "glcm_homogeneity_delta"]
CHANGE_ONLY = [*CHANGE, *TEXTURE_DELTAS, *CONTEXT, *SHAPE]
SUFFIX = "_vs_scene"

VARIANTS: dict[str, str] = {
    "change_only": "17 change / texture-delta / ring / shape features (no absolute pre or post levels)",
    "scene_normalised": "all 37 features, image features minus their scene median (shape raw)",
    "change_only_scene_normalised": "the 17 change-only features, scene-normalised (shape raw)",
}


def scene_normalise(X: pd.DataFrame, scene_ids: pd.Series, columns: list[str]) -> pd.DataFrame:
    """Subtract each scene's median from every non-shape column; shape columns pass through unchanged."""
    out = pd.DataFrame(index=X.index)
    medians = X[columns].groupby(scene_ids.to_numpy()).transform("median")
    for col in columns:
        if col in SHAPE:
            out[col] = X[col]
        else:
            out[f"{col}{SUFFIX}"] = X[col] - medians[col]
    return out


def variant_frame(name: str, X: pd.DataFrame, scene_ids: pd.Series) -> pd.DataFrame:
    """Model inputs for one variant. Takes features + scene ids only (no labels can leak in)."""
    if list(X.columns) != FEATURE_COLUMNS:
        raise ValueError("variant_frame expects the full v1 feature columns in schema order")
    if name == "change_only":
        return X[CHANGE_ONLY].copy()
    if name == "scene_normalised":
        return scene_normalise(X, scene_ids, FEATURE_COLUMNS)
    if name == "change_only_scene_normalised":
        return scene_normalise(X, scene_ids, CHANGE_ONLY)
    raise ValueError(f"unknown feature variant {name!r}; choose from {sorted(VARIANTS)}")

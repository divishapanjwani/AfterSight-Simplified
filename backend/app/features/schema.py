"""The per-building feature vector: names, groups and the columns that must never be model inputs.

Bump FEATURE_SCHEMA_VERSION whenever a feature is added, removed or its formula changes; saved models
record the version and refuse to load against a different one.
"""

FEATURE_SCHEMA_VERSION = 1

_CHANNELS = ("r", "g", "b")

COLOUR = [f"{phase}_{stat}_{c}" for phase in ("pre", "post") for stat in ("mean", "std") for c in _CHANNELS]
CHANGE = (
    [f"abs_dmean_{c}" for c in _CHANNELS]
    + [f"dstd_{c}" for c in _CHANNELS]
    + ["gray_absdiff_mean", "brightness_ratio"]
)
TEXTURE = [
    f"{name}_{phase}"
    for name in ("gray_std", "edge_density", "glcm_contrast", "glcm_homogeneity")
    for phase in ("pre", "post", "delta")
]
CONTEXT = ["ring_dmean", "ring_absdiff_mean", "contrast_change"]
SHAPE = ["area_px", "compactness"]

FEATURE_GROUPS: dict[str, list[str]] = {
    "colour": COLOUR,
    "change": CHANGE,
    "texture": TEXTURE,
    "context": CONTEXT,
    "shape": SHAPE,
}
FEATURE_COLUMNS: list[str] = [name for group in FEATURE_GROUPS.values() for name in group]

# Identifiers and provenance — kept in the table for joins/UI, never fed to a model.
ID_COLUMNS = ["sample_id", "building_uid", "scene_id", "event", "disaster_type", "xbd_split"]
LABEL_COLUMN = "label"  # one of DAMAGE_CLASSES
# Per-building data-quality flags (from the mask derivation). Not model inputs: they describe the label,
# not the image, and would let a model shortcut.
QUALITY_COLUMNS = ["label_purity", "mixed_label", "touches_border", "ring_px"]
NON_FEATURE_COLUMNS = [*ID_COLUMNS, LABEL_COLUMN, *QUALITY_COLUMNS]

if len(set(FEATURE_COLUMNS)) != len(FEATURE_COLUMNS):
    raise RuntimeError("duplicate feature name in FEATURE_GROUPS")
if set(FEATURE_COLUMNS) & set(NON_FEATURE_COLUMNS):
    raise RuntimeError("a feature name collides with an id/label/quality column")

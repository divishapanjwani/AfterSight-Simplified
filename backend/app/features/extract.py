"""Per-building feature extraction from the scene cache. Formulas are documented in docs/FEATURES.md.

Each building's pixels are its exact mask blob: the cached pre_mask.png is re-labelled with the same
connected-component settings used to derive the footprints, so blob n is building "<scene_id>_b<n>".
Most features are region statistics (means, stds, proportions), which tolerate the small pre/post
misalignment xBD pairs can have far better than raw pixel-by-pixel subtraction.
"""

import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage
from shapely import wkt as shapely_wkt
from skimage.feature import canny

from app.datasets.mask_buildings import CONNECTIVITY_STRUCTURES
from app.features.config import FeatureConfig
from app.features.schema import FEATURE_COLUMNS

_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)  # ITU-R BT.601 grey from RGB
_RING_STRUCTURE = ndimage.generate_binary_structure(2, 2)  # 8-neighbour growth -> square ring


class FeatureError(Exception):
    """Cached scene that is inconsistent with its scene.json (message names the scene and the problem)."""


def to_gray(rgb: np.ndarray) -> np.ndarray:
    return rgb.astype(np.float32) @ _LUMA


def glcm_contrast_homogeneity(gray: np.ndarray, mask: np.ndarray, levels: int, distance: int) -> tuple[float, float]:
    """Grey-level co-occurrence contrast and homogeneity using only pixel pairs that lie inside the mask.

    Pairs are taken horizontally and vertically at `distance`. For a symmetric, normalised GLCM P:
    contrast = sum P(i,j)(i-j)^2 and homogeneity = sum P(i,j)/(1+(i-j)^2), which equal the mean of
    (i-j)^2 and of 1/(1+(i-j)^2) over the pairs — so the matrix itself never has to be built.
    """
    q = np.clip((gray * levels / 256.0).astype(np.int32), 0, levels - 1)
    diffs = []
    for a, b, ma, mb in (
        (q[:, :-distance], q[:, distance:], mask[:, :-distance], mask[:, distance:]),
        (q[:-distance, :], q[distance:, :], mask[:-distance, :], mask[distance:, :]),
    ):
        valid = ma & mb
        diffs.append((a[valid] - b[valid]).astype(np.float64))
    d = np.concatenate(diffs)
    if d.size == 0:
        return math.nan, math.nan
    sq = d * d
    return float(sq.mean()), float((1.0 / (1.0 + sq)).mean())


def compactness(polygon_wkt: str) -> float:
    """Isoperimetric quotient 4*pi*A/P^2: 1 for a circle, pi/4 ~ 0.785 for a square, -> 0 for thin shapes."""
    geom = shapely_wkt.loads(polygon_wkt)
    return float(4 * math.pi * geom.area / geom.length**2) if geom.length > 0 else math.nan


def building_features(pre: np.ndarray, post: np.ndarray, mask: np.ndarray, ring: np.ndarray,
                      cfg: FeatureConfig) -> dict[str, float]:
    """All image features for one building, given RGB crops and boolean masks of the same shape."""
    f: dict[str, float] = {}
    pre_px, post_px = pre[mask].astype(np.float64), post[mask].astype(np.float64)
    for phase, px in (("pre", pre_px), ("post", post_px)):
        for k, c in enumerate("rgb"):
            f[f"{phase}_mean_{c}"] = float(px[:, k].mean())
            f[f"{phase}_std_{c}"] = float(px[:, k].std())
    for c in "rgb":
        f[f"abs_dmean_{c}"] = abs(f[f"post_mean_{c}"] - f[f"pre_mean_{c}"])
        f[f"dstd_{c}"] = f[f"post_std_{c}"] - f[f"pre_std_{c}"]

    g_pre, g_post = to_gray(pre), to_gray(post)
    b_pre, b_post = float(g_pre[mask].mean()), float(g_post[mask].mean())
    f["gray_absdiff_mean"] = float(np.abs(g_post[mask] - g_pre[mask]).mean())
    f["brightness_ratio"] = b_post / max(b_pre, 1.0)  # floor avoids dividing by a black roof

    for phase, g in (("pre", g_pre), ("post", g_post)):
        f[f"gray_std_{phase}"] = float(g[mask].std())
        # Canny runs on the whole crop (footprint + margin) so the footprint edge itself is detectable.
        edges = canny(g / 255.0, sigma=cfg.canny_sigma)
        f[f"edge_density_{phase}"] = float(edges[mask].mean())
        contrast, homogeneity = glcm_contrast_homogeneity(g, mask, cfg.glcm_levels, cfg.glcm_distance_px)
        f[f"glcm_contrast_{phase}"] = contrast
        f[f"glcm_homogeneity_{phase}"] = homogeneity
    for name in ("gray_std", "edge_density", "glcm_contrast", "glcm_homogeneity"):
        f[f"{name}_delta"] = f[f"{name}_post"] - f[f"{name}_pre"]

    if ring.any():
        r_pre, r_post = float(g_pre[ring].mean()), float(g_post[ring].mean())
        f["ring_dmean"] = r_post - r_pre
        f["ring_absdiff_mean"] = float(np.abs(g_post[ring] - g_pre[ring]).mean())
        f["contrast_change"] = (b_post - r_post) - (b_pre - r_pre)
    else:  # no ground visible around the building (dense blocks): left missing, imputed on train only later
        f["ring_dmean"] = f["ring_absdiff_mean"] = f["contrast_change"] = math.nan
    return f


def scene_rows(scene_dir: Path, doc: dict[str, Any], cfg: FeatureConfig, connectivity: int,
               disaster_type: str | None) -> list[dict[str, Any]]:
    """Feature rows (ids + label + quality + features) for every kept building of one cached scene."""
    with Image.open(scene_dir / "pre.png") as im:
        pre = np.asarray(im.convert("RGB"))
    with Image.open(scene_dir / "post.png") as im:
        post = np.asarray(im.convert("RGB"))
    with Image.open(scene_dir / "pre_mask.png") as im:
        footprint = np.asarray(im) > 0
    if pre.shape != post.shape or pre.shape[:2] != footprint.shape:
        raise FeatureError(f"{doc['scene_id']}: image/mask shapes differ {pre.shape}, {post.shape}, {footprint.shape}")

    labels, _n = ndimage.label(footprint, structure=CONNECTIVITY_STRUCTURES[connectivity])
    windows = ndimage.find_objects(labels)
    height, width = footprint.shape
    pad = cfg.ring_width_px + 2
    rows = []
    for b in doc["buildings"]:
        index = int(b["building_id"].rsplit("_b", 1)[1])
        if index > len(windows) or windows[index - 1] is None:
            raise FeatureError(f"{b['building_id']}: no matching blob in pre_mask.png")
        ys, xs = windows[index - 1]
        y0, y1 = max(ys.start - pad, 0), min(ys.stop + pad, height)
        x0, x1 = max(xs.start - pad, 0), min(xs.stop + pad, width)
        mask = labels[y0:y1, x0:x1] == index
        area = int(mask.sum())
        if area != int(b["area_px"]):
            raise FeatureError(f"{b['building_id']}: blob has {area} px but scene.json says {b['area_px']}")
        grown = ndimage.binary_dilation(mask, structure=_RING_STRUCTURE, iterations=cfg.ring_width_px)
        ring = grown & ~footprint[y0:y1, x0:x1]  # exclude this AND every other building

        features = building_features(pre[y0:y1, x0:x1], post[y0:y1, x0:x1], mask, ring, cfg)
        features["area_px"] = float(area)
        features["compactness"] = compactness(b["wkt"])
        rows.append({
            "sample_id": b["building_id"],
            "building_uid": b["building_id"],
            "scene_id": doc["scene_id"],
            "event": doc["event"],
            "disaster_type": disaster_type,  # config-sourced (see config/events.yaml)
            "xbd_split": doc["xbd_split"],
            "label": b["damage"],
            "label_purity": b["label_purity"],
            "mixed_label": b["mixed_label"],
            "touches_border": b["touches_border"],
            "ring_px": int(ring.sum()),
            **{name: features[name] for name in FEATURE_COLUMNS},
        })
    return rows

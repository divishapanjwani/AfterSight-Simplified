"""Derive building instances from rasterised xBD annotation masks.

The Parquet mirror of xBD has no polygons, only two masks per scene:
  - footprint mask (t1_mask, pre-disaster): 0 = background, >0 = building
  - damage mask    (t2_mask, post-disaster): 0 = background, 1..4 = damage class

Each connected blob of the footprint mask becomes one "building". Its damage label is the majority
vote of the damage-mask values inside it. Limitations (documented in data/README.md):
  - touching buildings rasterise into one blob and cannot be separated;
  - original xBD building UIDs and lng/lat geometry are not available;
  - a building whose post label was not rasterised (e.g. "un-classified") shows up as a footprint
    with no damage value; it is excluded and counted, never given a class.
"""

from collections import Counter
from dataclasses import dataclass, field

import cv2
import numpy as np
import shapely
from scipy import ndimage
from shapely.geometry import Polygon

from app.datasets.annotations import Building, DropReason, DroppedBuilding, polygonal_part

CONNECTIVITY_STRUCTURES = {4: ndimage.generate_binary_structure(2, 1), 8: ndimage.generate_binary_structure(2, 2)}


@dataclass(frozen=True)
class MixedBlob:
    """A kept or dropped blob whose pixels carry more than one damage value."""

    building_id: str
    value_counts: dict[int, int]  # damage-mask value -> pixel count inside the blob
    chosen_value: int | None  # None when the vote was a tie
    purity: float


@dataclass
class MaskParseResult:
    scene_id: str
    width: int
    height: int
    buildings: list[Building] = field(default_factory=list)
    dropped: list[DroppedBuilding] = field(default_factory=list)
    mixed: list[MixedBlob] = field(default_factory=list)
    blob_areas: list[int] = field(default_factory=list)  # every blob, before any filtering
    footprint_values: Counter[int] = field(default_factory=Counter)  # pixel counts per footprint-mask value
    damage_values: Counter[int] = field(default_factory=Counter)  # pixel counts per damage-mask value
    footprint_px_without_damage_value: int = 0
    damage_px_outside_footprint: int = 0

    @property
    def n_blobs(self) -> int:
        return len(self.blob_areas)

    @property
    def footprint_iou(self) -> float | None:
        """Overlap of footprint pixels and damage-labelled pixels (None for an empty scene)."""
        both = self.footprint_px - self.footprint_px_without_damage_value
        union = self.footprint_px + self.damage_px_outside_footprint
        return both / union if union else None

    @property
    def footprint_px(self) -> int:
        return sum(n for v, n in self.footprint_values.items() if v != 0)


def building_id(scene_id: str, index: int) -> str:
    """Deterministic: blobs are numbered in raster order (top-left first) by scipy.ndimage.label."""
    return f"{scene_id}_b{index:04d}"


def _blob_polygon(blob: np.ndarray, row0: int, col0: int) -> tuple[shapely.Geometry, bool]:
    """Trace the blob outline (with holes) into a polygon in image pixel coordinates.

    cv2 returns boundary pixel indices; +0.5 puts vertices at pixel centres, so the polygon sits
    about half a pixel inside the true footprint edge. Exact areas use the pixel count instead.
    """
    padded = np.pad(blob.astype(np.uint8), 1)
    contours, hierarchy = cv2.findContours(padded, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    offset = np.array([col0 - 1 + 0.5, row0 - 1 + 0.5])
    rings = [c.reshape(-1, 2) + offset for c in contours]
    shells: list[Polygon] = []
    for i, ring in enumerate(rings):
        parent = hierarchy[0][i][3]
        if parent != -1 or len(ring) < 3:
            continue
        holes = [r for j, r in enumerate(rings) if hierarchy[0][j][3] == i and len(r) >= 3]
        shells.append(Polygon(ring, holes))
    geom = shapely.union_all(shells) if shells else Polygon()
    repaired = False
    if not geom.is_empty and not geom.is_valid:
        geom = polygonal_part(shapely.make_valid(geom))
        repaired = True
    return geom, repaired


def buildings_from_masks(
    scene_id: str,
    footprint_mask: np.ndarray,
    damage_mask: np.ndarray,
    damage_values: dict[int, str],
    min_area_px: float,
    connectivity: int = 4,
) -> MaskParseResult:
    if footprint_mask.shape != damage_mask.shape or footprint_mask.ndim != 2:
        raise ValueError(f"masks must be 2-D and the same shape, got {footprint_mask.shape} and {damage_mask.shape}")
    height, width = footprint_mask.shape
    result = MaskParseResult(scene_id, width, height)

    values, counts = np.unique(footprint_mask, return_counts=True)
    result.footprint_values.update(dict(zip(values.tolist(), counts.tolist())))
    values, counts = np.unique(damage_mask, return_counts=True)
    result.damage_values.update(dict(zip(values.tolist(), counts.tolist())))

    footprint = footprint_mask > 0
    labelled = damage_mask > 0
    result.footprint_px_without_damage_value = int(np.count_nonzero(footprint & ~labelled))
    result.damage_px_outside_footprint = int(np.count_nonzero(labelled & ~footprint))

    blobs, _n_blobs = ndimage.label(footprint, structure=CONNECTIVITY_STRUCTURES[connectivity])
    for index, window in enumerate(ndimage.find_objects(blobs), start=1):
        uid = building_id(scene_id, index)
        blob = blobs[window] == index
        area = int(np.count_nonzero(blob))
        result.blob_areas.append(area)

        inside = damage_mask[window][blob]
        value_counts = Counter(inside[inside > 0].tolist())
        ranked = value_counts.most_common()
        top_value = ranked[0][0] if ranked else None
        raw_label = "" if top_value is None else str(top_value)

        def drop(reason: DropReason, detail: str = "") -> None:
            result.dropped.append(DroppedBuilding(scene_id, uid, raw_label, reason, detail))

        is_tie = len(ranked) > 1 and ranked[0][1] == ranked[1][1]
        purity = ranked[0][1] / sum(value_counts.values()) if ranked else 0.0
        if len(ranked) > 1:
            result.mixed.append(MixedBlob(uid, dict(sorted(value_counts.items())), None if is_tie else top_value, round(purity, 4)))

        if area < min_area_px:
            drop(DropReason.TOO_SMALL, f"area {area} px < {min_area_px}")
            continue
        if top_value is None:
            drop(DropReason.LABEL_MISSING, "no damage value inside the footprint")
            continue
        if is_tie:
            drop(DropReason.LABEL_TIE, f"values {dict(ranked[:2])} have equal pixel counts")
            continue
        if top_value not in damage_values:
            drop(DropReason.LABEL_UNKNOWN, f"majority mask value {top_value} is not a damage class")
            continue

        row0, col0 = window[0].start, window[1].start
        geom, repaired = _blob_polygon(blob, row0, col0)
        if geom.is_empty or geom.area == 0:
            drop(DropReason.DEGENERATE_SHAPE, f"{area} px blob has no polygon area (too thin)")
            continue

        touches_border = row0 == 0 or col0 == 0 or window[0].stop == height or window[1].stop == width
        result.buildings.append(
            Building(
                uid=uid,
                damage=damage_values[top_value],
                polygon=geom,
                area_px=float(area),
                repaired=repaired,
                clipped=False,
                label_purity=round(purity, 4),
                mixed_label=len(ranked) > 1,
                touches_border=touches_border,
            )
        )
    return result

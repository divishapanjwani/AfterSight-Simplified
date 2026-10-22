"""Parse xBD label JSON into validated building footprints in image pixel coordinates.

xBD label files (format verified by the inspector on real data) look like:

    {"features": {"xy":      [{"properties": {"feature_type": "building", "subtype": "minor-damage",
                                              "uid": "..."}, "wkt": "POLYGON ((x y, ...))"}, ...],
                  "lng_lat": [... same buildings in longitude/latitude ...]},
     "metadata": {"disaster": "...", "disaster_type": "...", "width": 1024, "height": 1024, ...}}

We use the "xy" polygons (pixels; x to the right, y downwards, origin top-left) because features are
computed on the image grid. Damage labels ("subtype") exist only on post-disaster files.
"""

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import shapely
from shapely import wkt as shapely_wkt
from shapely.errors import ShapelyError
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.geometry.base import BaseGeometry

from app.core.damage_scale import DAMAGE_CLASSES


class AnnotationError(Exception):
    """A label file that cannot be used at all (unreadable JSON, missing required structure)."""


class DropReason(StrEnum):
    NOT_BUILDING = "not_building"
    MISSING_UID = "missing_uid"
    DUPLICATE_UID = "duplicate_uid"
    LABEL_MISSING = "label_missing"  # post feature without a "subtype"
    LABEL_EXCLUDED = "label_excluded"  # configured exclusion, e.g. "un-classified"
    LABEL_UNKNOWN = "label_unknown"  # anything else outside the four classes
    WKT_INVALID = "wkt_invalid"
    NOT_POLYGON = "not_polygon"
    EMPTY_AFTER_REPAIR = "empty_after_repair"
    OUTSIDE_IMAGE = "outside_image"
    TOO_SMALL = "too_small"
    LABEL_TIE = "label_tie"  # mask blob whose two most common damage values have equal pixel counts
    DEGENERATE_SHAPE = "degenerate_shape"  # mask blob whose outline is not an area (e.g. 1 px wide)


@dataclass(frozen=True)
class Building:
    uid: str
    damage: str  # one of DAMAGE_CLASSES
    polygon: BaseGeometry  # Polygon or MultiPolygon in pixel coordinates, inside the image
    area_px: float
    repaired: bool  # geometry was invalid and fixed with make_valid
    clipped: bool  # part of the footprint lay outside the image and was cut off
    # Mask-derived footprints only (Parquet source):
    label_purity: float = 1.0  # share of the blob's labelled pixels that carry the chosen damage value
    mixed_label: bool = False  # blob contains more than one damage value (often two merged buildings)
    touches_border: bool = False  # blob reaches the image edge, so the building may be cut off


@dataclass(frozen=True)
class DroppedBuilding:
    scene_id: str
    uid: str
    raw_label: str
    reason: DropReason
    detail: str = ""


@dataclass(frozen=True)
class SceneMetadata:
    disaster: str | None
    disaster_type: str | None
    width: int | None
    height: int | None
    img_name: str | None
    sensor: str | None
    capture_date: str | None
    gsd: float | None


@dataclass
class ParsedLabels:
    scene_id: str
    metadata: SceneMetadata
    width: int
    height: int
    buildings: list[Building] = field(default_factory=list)
    dropped: list[DroppedBuilding] = field(default_factory=list)
    n_features: int = 0
    n_lng_lat_features: int = 0


def load_label_json(path: Path) -> dict[str, Any]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AnnotationError(f"cannot read label JSON: {exc}") from exc
    if not isinstance(doc, dict):
        raise AnnotationError("label JSON is not an object")
    features = doc.get("features")
    if not isinstance(features, dict) or not isinstance(features.get("xy"), list):
        raise AnnotationError("label JSON has no features.xy list")
    if not isinstance(doc.get("metadata", {}), dict):
        raise AnnotationError("label JSON metadata is not an object")
    return doc


def read_metadata(doc: dict[str, Any]) -> SceneMetadata:
    meta = doc.get("metadata") or {}

    def as_int(value: Any) -> int | None:
        return int(value) if isinstance(value, (int, float)) else None

    gsd = meta.get("gsd")
    return SceneMetadata(
        disaster=meta.get("disaster"),
        disaster_type=meta.get("disaster_type"),
        width=as_int(meta.get("width")),
        height=as_int(meta.get("height")),
        img_name=meta.get("img_name"),
        sensor=meta.get("sensor"),
        capture_date=meta.get("capture_date"),
        gsd=float(gsd) if isinstance(gsd, (int, float)) else None,
    )


def building_uids(doc: dict[str, Any]) -> set[str]:
    """UIDs of building features in a label file (used to check pre/post consistency)."""
    uids = set()
    for feature in doc["features"]["xy"]:
        props = feature.get("properties") or {}
        if props.get("feature_type") == "building" and props.get("uid"):
            uids.add(str(props["uid"]))
    return uids


def polygonal_part(geom: BaseGeometry) -> BaseGeometry:
    """Keep only the area-bearing parts of a geometry (make_valid can also return lines/points)."""
    parts: list[BaseGeometry] = []
    for part in shapely.get_parts(geom):
        if isinstance(part, (Polygon, MultiPolygon)) and not part.is_empty:
            parts.append(part)
        elif part.geom_type == "GeometryCollection":
            sub = polygonal_part(part)
            if not sub.is_empty:
                parts.append(sub)
    return shapely.union_all(parts) if parts else Polygon()


def clean_footprint(raw_wkt: Any, width: int, height: int) -> tuple[BaseGeometry, bool, bool]:
    """Parse, repair and clip one footprint.

    Returns (geometry, repaired, clipped). Raises ValueError(DropReason, detail) if unusable.
    """
    if not isinstance(raw_wkt, str):
        raise ValueError(DropReason.WKT_INVALID, "wkt is missing or not a string")
    try:
        geom = shapely_wkt.loads(raw_wkt)
    except ShapelyError as exc:
        raise ValueError(DropReason.WKT_INVALID, str(exc).splitlines()[0]) from exc
    if not isinstance(geom, (Polygon, MultiPolygon)):
        raise ValueError(DropReason.NOT_POLYGON, geom.geom_type)

    repaired = False
    if not geom.is_valid:
        geom = polygonal_part(shapely.make_valid(geom))
        repaired = True
    if geom.is_empty or geom.area == 0:
        raise ValueError(DropReason.EMPTY_AFTER_REPAIR, "no area left after make_valid")

    frame = box(0, 0, width, height)
    clipped = not frame.contains(geom)
    if clipped:
        geom = polygonal_part(geom.intersection(frame))
        if geom.is_empty or geom.area == 0:
            raise ValueError(DropReason.OUTSIDE_IMAGE, "footprint lies entirely outside the image")
    return geom, repaired, clipped


def parse_post_labels(
    doc: dict[str, Any],
    scene_id: str,
    excluded_labels: list[str],
    min_area_px: float,
    image_size: tuple[int, int] | None = None,
) -> ParsedLabels:
    """Turn a post-disaster label document into kept buildings plus a log of every dropped one.

    image_size (width, height) comes from the actual image; it wins over metadata if both exist.
    """
    metadata = read_metadata(doc)
    if image_size is not None:
        width, height = image_size
    elif metadata.width and metadata.height:
        width, height = metadata.width, metadata.height
    else:
        raise AnnotationError("image size unknown: no image given and metadata has no width/height")

    lng_lat = doc["features"].get("lng_lat")
    result = ParsedLabels(
        scene_id=scene_id,
        metadata=metadata,
        width=width,
        height=height,
        n_lng_lat_features=len(lng_lat) if isinstance(lng_lat, list) else 0,
    )
    excluded = set(excluded_labels)
    seen_uids: set[str] = set()

    for feature in doc["features"]["xy"]:
        result.n_features += 1
        props = feature.get("properties") or {}
        uid = str(props.get("uid") or "")
        label = props.get("subtype")
        raw_label = "" if label is None else str(label)

        def drop(reason: DropReason, detail: str = "") -> None:
            result.dropped.append(DroppedBuilding(scene_id, uid, raw_label, reason, detail))

        if props.get("feature_type") != "building":
            drop(DropReason.NOT_BUILDING, f"feature_type={props.get('feature_type')!r}")
            continue
        if not uid:
            drop(DropReason.MISSING_UID)
            continue
        if uid in seen_uids:
            drop(DropReason.DUPLICATE_UID)
            continue
        seen_uids.add(uid)
        if label is None:
            drop(DropReason.LABEL_MISSING)
            continue
        if raw_label in excluded:
            drop(DropReason.LABEL_EXCLUDED)
            continue
        if raw_label not in DAMAGE_CLASSES:
            drop(DropReason.LABEL_UNKNOWN)
            continue

        try:
            geom, repaired, clipped = clean_footprint(feature.get("wkt"), width, height)
        except ValueError as exc:
            reason, detail = exc.args
            drop(reason, detail)
            continue
        if geom.area < min_area_px:
            drop(DropReason.TOO_SMALL, f"area {geom.area:.1f} px < {min_area_px}")
            continue

        result.buildings.append(Building(uid, raw_label, geom, float(geom.area), repaired, clipped))

    return result

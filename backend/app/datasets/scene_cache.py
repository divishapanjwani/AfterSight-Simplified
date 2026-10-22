"""Per-scene disk cache for the selected events (Parquet source only).

The Parquet mirror packs ~165 scenes per 500 MB shard. Later phases need random access to single
scenes, so the inspector extracts just the configured events into

    <cache.dir>/<scene_id>/pre.png, post.png         original PNG bytes, not re-encoded
                          /pre_mask.png, post_mask.png
                          /pre_thumb.jpg, post_thumb.jpg   small previews for the UI
                          /scene.json                 derived footprints + provenance

Scenes of events that are not configured are never written.
"""

import json
from pathlib import Path
from typing import Any

from PIL import Image
from shapely import wkt as shapely_wkt

from app.datasets.annotations import Building, DroppedBuilding

SCENE_SCHEMA_VERSION = 1


def scene_dir(cache_root: Path, scene_id: str) -> Path:
    return cache_root / scene_id


def is_cached(cache_root: Path, scene_id: str) -> bool:
    return (scene_dir(cache_root, scene_id) / "scene.json").is_file()


def _thumbnail(image: Image.Image, size: int, path: Path) -> int:
    thumb = image.convert("RGB")
    thumb.thumbnail((size, size))
    thumb.save(path, "JPEG", quality=85)
    return path.stat().st_size


def building_record(b: Building) -> dict[str, Any]:
    return {
        "building_id": b.uid,
        "damage": b.damage,
        "area_px": b.area_px,
        "label_purity": b.label_purity,
        "mixed_label": b.mixed_label,
        "touches_border": b.touches_border,
        "repaired": b.repaired,
        "wkt": b.polygon.wkt,
    }


def write_scene(
    cache_root: Path,
    scene_id: str,
    *,
    pngs: dict[str, bytes],
    pre_image: Image.Image,
    post_image: Image.Image,
    thumbnail_px: int,
    meta: dict[str, Any],
    buildings: list[Building],
    dropped: list[DroppedBuilding],
) -> int:
    """Write one scene and return the bytes written. scene.json goes last, so it marks a complete entry."""
    folder = scene_dir(cache_root, scene_id)
    folder.mkdir(parents=True, exist_ok=True)
    written = 0
    for name in ("pre", "post", "pre_mask", "post_mask"):
        path = folder / f"{name}.png"
        path.write_bytes(pngs[name])
        written += len(pngs[name])
    written += _thumbnail(pre_image, thumbnail_px, folder / "pre_thumb.jpg")
    written += _thumbnail(post_image, thumbnail_px, folder / "post_thumb.jpg")
    doc = {
        "schema_version": SCENE_SCHEMA_VERSION,
        "scene_id": scene_id,
        **meta,
        "footprint_source": "derived from rasterised xBD annotation masks (connected components)",
        "buildings": [building_record(b) for b in buildings],
        "dropped": [{"building_id": d.uid, "reason": d.reason.value, "raw_label": d.raw_label, "detail": d.detail} for d in dropped],
    }
    scene_json = folder / "scene.json"
    scene_json.write_text(json.dumps(doc), encoding="utf-8")
    return written + scene_json.stat().st_size


def stale_scenes(cache_root: Path, keep: set[str]) -> list[str]:
    """Cached scene folders that are not in the current selection (reported, not deleted)."""
    if not cache_root.is_dir():
        return []
    return sorted(p.name for p in cache_root.iterdir() if p.is_dir() and p.name not in keep)


def load_scene(cache_root: Path, scene_id: str) -> tuple[Image.Image, Image.Image, list[Building], dict[str, Any]]:
    """Read a cached scene back: (pre image, post image, buildings, scene.json document)."""
    folder = scene_dir(cache_root, scene_id)
    doc = json.loads((folder / "scene.json").read_text(encoding="utf-8"))
    buildings = [
        Building(
            uid=b["building_id"], damage=b["damage"], polygon=shapely_wkt.loads(b["wkt"]), area_px=b["area_px"],
            repaired=b["repaired"], clipped=False, label_purity=b["label_purity"],
            mixed_label=b["mixed_label"], touches_border=b["touches_border"],
        )
        for b in doc["buildings"]
    ]
    with Image.open(folder / "pre.png") as pre, Image.open(folder / "post.png") as post:
        return pre.convert("RGB"), post.convert("RGB"), buildings, doc

"""Scene images and building crops as JPEG files (cached on disk next to the scene; never PNG-in-JSON)."""

import os
import tempfile
from pathlib import Path

from PIL import Image
from shapely import wkt as shapely_wkt

from app.core.store import Store

VIEW_MAX_PX = 1024
CROP_SIZES = (96, 192, 384)


def _atomic_jpeg(image: Image.Image, path: Path, quality: int = 85) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".jpg")
    os.close(fd)
    image.convert("RGB").save(tmp, "JPEG", quality=quality)
    os.replace(tmp, path)  # concurrent requests never see a half-written file
    return path


def scene_image(store: Store, scene_id: str, phase: str, size: str) -> Path:
    folder = store.scene_dir(scene_id)
    if size == "thumb":
        return folder / f"{phase}_thumb.jpg"
    view = folder / f"{phase}_view.jpg"
    if not view.is_file():
        with Image.open(folder / f"{phase}.png") as img:
            img.thumbnail((VIEW_MAX_PX, VIEW_MAX_PX))
            _atomic_jpeg(img, view, quality=88)
    return view


def building_crop(store: Store, building_uid: str, phase: str, size: int) -> Path:
    scene_id, building = store.valid_building(building_uid)
    folder = store.scene_dir(scene_id)
    out = folder / "crops" / f"{building['building_id']}_{phase}_{size}.jpg"
    if out.is_file():
        return out
    minx, miny, maxx, maxy = shapely_wkt.loads(building["wkt"]).bounds
    side = max(maxx - minx, maxy - miny)
    pad = max(16.0, 0.35 * side)
    cx, cy, half = (minx + maxx) / 2, (miny + maxy) / 2, side / 2 + pad
    with Image.open(folder / f"{phase}.png") as img:
        box = (max(0, int(cx - half)), max(0, int(cy - half)), min(img.width, int(cx + half) + 1), min(img.height, int(cy + half) + 1))
        crop = img.crop(box)
    crop.thumbnail((size, size))  # downscale only; small buildings are upscaled by the browser
    return _atomic_jpeg(crop, out)

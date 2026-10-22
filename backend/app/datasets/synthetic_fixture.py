"""Deterministic synthetic dataset in the xBD folder + JSON layout.

FOR TESTS AND PIPELINE DEVELOPMENT ONLY — this is not xBD data and must never be shown as such.
Every event is named "synthetic-*" and every label file carries metadata.synthetic = true.

Images are small (256x256) drawings: rectangles for buildings on a noisy background. The post image
changes each building according to its damage label, so later phases can sanity-check features.

Built-in edge cases (the tests rely on them):
  train/synthetic-flood_00000000   bowtie polygon (repaired), footprint crossing the right edge (clipped),
                                   tiny footprint (too_small), malformed WKT, "un-classified" label
  train/synthetic-flood_00000001   post image shifted 12 px (alignment flag)
  train/synthetic-flood_00000004   pre image + pre label only (incomplete pair)
  train/synthetic-fire_00000001    footprint fully outside the image, unknown label "partial-damage",
                                   non-building feature, post feature with no damage label
  train/synthetic-fire_00000002    corrupt post image
  train/synthetic-quake_00000000   an event that the fixture config does not select
"""

import json
import uuid
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from app.core.damage_scale import DAMAGE_CLASSES

IMAGE_SIZE = 256
ALIGNMENT_SHIFT_PX = 12
_UID_NAMESPACE = uuid.UUID("6f1e2c1a-4b7d-4c38-9a51-0c2b5a7e9d10")
_DISASTER_TYPES = {"synthetic-flood": "flooding", "synthetic-fire": "fire", "synthetic-quake": "earthquake"}

# (xbd_split, event, number)
_SCENES = [
    ("train", "synthetic-flood", 0),
    ("train", "synthetic-flood", 1),
    ("train", "synthetic-flood", 2),
    ("train", "synthetic-flood", 4),
    ("train", "synthetic-fire", 0),
    ("train", "synthetic-fire", 1),
    ("train", "synthetic-fire", 2),
    ("train", "synthetic-quake", 0),
    ("test", "synthetic-flood", 3),
    ("test", "synthetic-fire", 3),
]


def _rect(x0: float, y0: float, x1: float, y1: float) -> list[tuple[float, float]]:
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _wkt(points: list[tuple[float, float]]) -> str:
    ring = points + [points[0]]
    return "POLYGON ((" + ", ".join(f"{x:g} {y:g}" for x, y in ring) + "))"


def _regular_buildings(number: int) -> list[dict]:
    """Eight 36x30 px buildings in two rows; labels rotate so each scene has two of every class."""
    buildings = []
    for i in range(8):
        col, row = i % 4, i // 4
        x0, y0 = 20 + col * 58, 20 + row * 60
        buildings.append({"points": _rect(x0, y0, x0 + 36, y0 + 30), "subtype": DAMAGE_CLASSES[(i + number) % 4]})
    return buildings


def _edge_cases(event: str, number: int) -> list[dict]:
    if (event, number) == ("synthetic-flood", 0):
        return [
            {"points": [(100, 150), (140, 190), (140, 150), (100, 190)], "subtype": "destroyed"},  # bowtie
            {"points": _rect(230, 150, 280, 180), "subtype": "major-damage"},  # crosses right edge
            {"points": _rect(10, 200, 12, 202), "subtype": "no-damage"},  # 4 px² — too small
            {"raw_wkt": "POLYGON ((1 2, 3", "subtype": "minor-damage"},  # malformed
            {"points": _rect(30, 150, 60, 175), "subtype": "un-classified"},
        ]
    if (event, number) == ("synthetic-fire", 1):
        return [
            {"points": _rect(300, 20, 330, 50), "subtype": "destroyed"},  # fully outside
            {"points": _rect(30, 150, 60, 175), "subtype": "partial-damage"},  # unknown label
            {"points": _rect(80, 150, 110, 175), "subtype": "no-damage", "feature_type": "road"},
            {"points": _rect(130, 150, 160, 175), "subtype": None},  # no damage label
        ]
    return []


def _render_pair(buildings: list[dict], event: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    size = IMAGE_SIZE
    ground = np.array([104, 112, 90], dtype=np.float32)
    pre = ground + rng.normal(0, 4, (size, size, 3)).astype(np.float32)
    post_ground_shift = np.array([-18, -8, 10] if event == "synthetic-flood" else [-6, -6, -6], dtype=np.float32)

    masks = []
    for b in buildings:
        mask_img = Image.new("L", (size, size), 0)
        if "points" in b:
            ImageDraw.Draw(mask_img).polygon(b["points"], fill=255)
        mask = np.asarray(mask_img) > 0
        masks.append(mask)
        roof = rng.choice([[176, 168, 160], [150, 70, 60], [200, 200, 205]]).astype(np.float32)
        pre[mask] = roof + rng.normal(0, 3, (int(mask.sum()), 3))

    any_building = np.any(masks, axis=0) if masks else np.zeros((size, size), bool)
    post = pre.copy()
    post[~any_building] += post_ground_shift
    for b, mask in zip(buildings, masks):
        n = int(mask.sum())
        if n == 0:
            continue
        rubble = np.array([92, 78, 64], dtype=np.float32) + rng.normal(0, 28, (n, 3))
        if b["subtype"] == "minor-damage":
            post[mask] = 0.75 * post[mask] + 0.25 * np.array([110, 90, 70])
        elif b["subtype"] == "major-damage":
            ys, xs = np.nonzero(mask)
            half = mask.copy()
            half[ys[xs > xs.mean()], xs[xs > xs.mean()]] = False
            post[half] = np.array([92, 78, 64]) + rng.normal(0, 28, (int(half.sum()), 3))
        elif b["subtype"] == "destroyed":
            post[mask] = rubble
    return np.clip(pre, 0, 255).astype(np.uint8), np.clip(post, 0, 255).astype(np.uint8)


def _label_doc(scene_id: str, event: str, phase: str, buildings: list[dict], file_stem: str) -> dict:
    features = []
    for i, b in enumerate(buildings):
        props: dict = {"feature_type": b.get("feature_type", "building"), "uid": str(uuid.uuid5(_UID_NAMESPACE, f"{scene_id}/{i}"))}
        if phase == "post" and b["subtype"] is not None:
            props["subtype"] = b["subtype"]
        features.append({"properties": props, "wkt": b.get("raw_wkt") or _wkt(b["points"])})
    return {
        "features": {"lng_lat": [], "xy": features},
        "metadata": {
            "synthetic": True,
            "sensor": "synthetic",
            "disaster": event,
            "disaster_type": _DISASTER_TYPES[event],
            "width": IMAGE_SIZE,
            "height": IMAGE_SIZE,
            "img_name": f"{file_stem}.png",
        },
    }


def build_synthetic_xbd(root: Path, seed: int = 7) -> Path:
    """Write the fixture under `root` (overwriting files it owns) and return `root`."""
    for split, event, number in _SCENES:
        scene_id = f"{event}_{number:08d}"
        images_dir, labels_dir = root / split / "images", root / split / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng([seed, number, sum(map(ord, event))])
        buildings = _regular_buildings(number) + _edge_cases(event, number)
        pre, post = _render_pair(buildings, event, rng)
        if (event, number) == ("synthetic-flood", 1):
            post = np.roll(post, ALIGNMENT_SHIFT_PX, axis=1)

        for phase, pixels in (("pre", pre), ("post", post)):
            stem = f"{scene_id}_{phase}_disaster"
            if phase == "post" and (event, number) == ("synthetic-flood", 4):
                continue  # incomplete pair: no post image / label
            image_path = images_dir / f"{stem}.png"
            if phase == "post" and (event, number) == ("synthetic-fire", 2):
                image_path.write_bytes(b"this is not a png")  # corrupt image
            else:
                Image.fromarray(pixels, "RGB").save(image_path, optimize=True)
            doc = _label_doc(scene_id, event, phase, buildings, stem)
            (labels_dir / f"{stem}.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")

    (root / "README.md").write_text(
        "# SYNTHETIC FIXTURE — not xBD data\n\n"
        "Generated by `python scripts/make_fixture.py` (app/datasets/synthetic_fixture.py).\n"
        "Used only by tests and for pipeline development. Never present it as real results.\n",
        encoding="utf-8",
    )
    return root


# ---------------------------------------------------------------- Parquet-mirror fixture

PARQUET_SIZE = 128


def _parquet_masks(kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Footprint (t1) and damage (t2) masks with hand-placed blobs. Blob numbers follow raster order."""
    t1 = np.zeros((PARQUET_SIZE, PARQUET_SIZE), np.uint8)
    t2 = np.zeros_like(t1)

    def blob(rows: slice, cols: slice, value: int) -> None:
        t1[rows, cols] = 1
        t2[rows, cols] = value

    if kind == "edge_cases":
        blob(slice(0, 10), slice(118, 128), 2)  # b0001 touches the top/right border, minor-damage
        blob(slice(10, 20), slice(10, 20), 1)  # b0002 no-damage
        blob(slice(10, 20), slice(30, 40), 4)  # b0003 destroyed
        blob(slice(30, 40), slice(10, 30), 3)  # b0004 mixed: 140 px major + 60 px minor -> major
        t2[30:40, 24:30] = 2
        blob(slice(30, 40), slice(40, 50), 1)  # b0005 tie: 50 px no-damage + 50 px minor -> dropped
        t2[30:40, 45:50] = 2
        blob(slice(50, 60), slice(10, 20), 0)  # b0006 footprint with no damage value -> dropped
        blob(slice(60, 62), slice(60, 62), 1)  # b0007 4 px -> too small
        blob(slice(70, 80), slice(10, 20), 5)  # b0008 unexpected mask value 5 -> excluded, counted
        blob(slice(90, 91), slice(10, 40), 1)  # b0009 1 px wide line -> no polygon area
        t2[100:105, 100:105] = 3  # damage value with no footprint underneath
    elif kind == "two_buildings":
        blob(slice(20, 40), slice(20, 40), 1)
        blob(slice(60, 80), slice(60, 90), 4)
    elif kind == "one_building":
        blob(slice(20, 40), slice(20, 40), 1)
    return t1, t2


def _png(array: np.ndarray, mode: str) -> bytes:
    import io

    buffer = io.BytesIO()
    Image.fromarray(array, mode).save(buffer, "PNG")
    return buffer.getvalue()


def build_synthetic_parquet(root: Path, seed: int = 7) -> Path:
    """Write two Parquet shards in the same schema as the Hugging Face xBD mirror. NOT xBD data.

    shard 0 (one row per row group): synthetic-flood_00000000 (all blob edge cases),
                                     synthetic-flood_00000001 (no buildings)
    shard 1: synthetic-fire_00000000 (two buildings), synthetic-fire_00000001 (corrupt post image),
             synthetic-quake_00000000 (event not configured), and a row with a malformed image_name
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    cell = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    rows_by_shard = [
        [("train/images/synthetic-flood_00000000", "edge_cases"), ("train/images/synthetic-flood_00000001", "empty")],
        [("train/images/synthetic-fire_00000000", "two_buildings"), ("train/images/synthetic-fire_00000001", "corrupt"),
         ("train/images/synthetic-quake_00000000", "one_building"), ("train/images/Not A Scene", "one_building")],
    ]
    root.mkdir(parents=True, exist_ok=True)
    for shard_index, rows in enumerate(rows_by_shard):
        columns: dict[str, list] = {"t1_image": [], "t2_image": [], "t1_mask": [], "t2_mask": [], "image_name": []}
        for name, kind in rows:
            rng = np.random.default_rng([seed, sum(map(ord, name))])
            pre = rng.integers(60, 200, (PARQUET_SIZE, PARQUET_SIZE, 3), dtype=np.uint8)
            post = np.clip(pre.astype(int) - 20, 0, 255).astype(np.uint8)
            t1, t2 = _parquet_masks("one_building" if kind == "corrupt" else kind)
            columns["t1_image"].append({"bytes": _png(pre, "RGB"), "path": None})
            columns["t2_image"].append({"bytes": b"this is not a png" if kind == "corrupt" else _png(post, "RGB"), "path": None})
            columns["t1_mask"].append({"bytes": _png(t1, "L"), "path": None})
            columns["t2_mask"].append({"bytes": _png(t2, "L"), "path": None})
            columns["image_name"].append(name)
        table = pa.table({k: pa.array(v, type=pa.string() if k == "image_name" else cell) for k, v in columns.items()})
        pq.write_table(table, root / f"train-{shard_index:05d}-of-00002.parquet", row_group_size=1 if shard_index == 0 else 100)
    (root / "README.md").write_text(
        "# SYNTHETIC PARQUET FIXTURE — not xBD data\n\nGenerated by `python scripts/make_fixture.py`. Tests only.\n",
        encoding="utf-8",
    )
    return root

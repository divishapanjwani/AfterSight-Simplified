"""Stream the Hugging Face-style Parquet mirror of xBD one row at a time.

Verified layout (data/raw/xbd_parquet/, inspected 2026-10-22): shards named train-000NN-of-00017.parquet,
two row groups each, columns

    t1_image, t2_image, t1_mask, t2_mask : struct<bytes: binary, path: string>  (PNG bytes; path is null)
    image_name                           : string, e.g. "train/images/socal-fire_00000106"

t1 = pre-disaster, t2 = post-disaster. Images are 1024x1024 RGB; masks are single-channel.
A row is ~3 MB, so rows are streamed in small batches and image bytes are only materialised for the
rows the caller asks for — the full 8 GB mirror is never held in RAM.
"""

import io
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

IMAGE_COLUMNS = ("t1_image", "t2_image")
MASK_COLUMNS = ("t1_mask", "t2_mask")
NAME_COLUMN = "image_name"
REQUIRED_COLUMNS = (*IMAGE_COLUMNS, *MASK_COLUMNS, NAME_COLUMN)
IMAGE_NAME = re.compile(r"^(?P<split>[a-z0-9]+)/images/(?P<event>[a-z0-9]+(?:-[a-z0-9]+)*)_(?P<number>\d{8})$")
BATCH_SIZE = 4
# Without pre-buffering pyarrow reads pages as needed; peak RAM is then bounded by one row group of the
# mirror (~100 scenes, ~310 MB of PNG bytes) instead of growing with the number of shards.
_READ_OPTIONS = {"pre_buffer": False, "buffer_size": 1 << 20}


class ParquetFormatError(Exception):
    """A shard or cell does not have the expected structure; the message is safe to show."""


@dataclass(frozen=True)
class SceneRef:
    scene_id: str  # "<event>_<8 digits>"
    event: str
    xbd_split: str
    shard: str  # shard file name
    row: int  # row index within the shard


@dataclass
class ParquetRow:
    """One scene's row. Mask bytes are always loaded; image bytes only on request."""

    ref: SceneRef | None  # None when image_name does not match the expected pattern
    image_name: str
    t1_mask: bytes
    t2_mask: bytes
    load_images: Callable[[], tuple[bytes, bytes]]  # (pre PNG bytes, post PNG bytes)


def list_shards(root: Path) -> list[Path]:
    return sorted(p for p in root.glob("*.parquet") if p.is_file())


def missing_columns(shard: Path) -> list[str]:
    names = set(pq.ParquetFile(shard).schema_arrow.names)
    return [c for c in REQUIRED_COLUMNS if c not in names]


def parse_image_name(name: str) -> tuple[str, str, str] | None:
    """'train/images/socal-fire_00000106' -> ('train', 'socal-fire', '00000106')."""
    match = IMAGE_NAME.match(name or "")
    return (match["split"], match["event"], match["number"]) if match else None


def cell_bytes(cell: Any, column: str) -> bytes:
    if not isinstance(cell, dict) or not isinstance(cell.get("bytes"), (bytes, bytearray)):
        raise ParquetFormatError(f"{column} cell has no PNG bytes")
    return bytes(cell["bytes"])


def iter_rows(shard: Path) -> Iterator[ParquetRow]:
    """Yield every row of a shard in order, holding at most one small batch in memory."""
    parquet = pq.ParquetFile(shard, **_READ_OPTIONS)
    row_index = 0
    for batch in parquet.iter_batches(batch_size=BATCH_SIZE, columns=list(REQUIRED_COLUMNS), use_threads=False):
        names = batch.column(NAME_COLUMN).to_pylist()
        t1_masks = batch.column("t1_mask").to_pylist()
        t2_masks = batch.column("t2_mask").to_pylist()
        pre_col, post_col = batch.column("t1_image"), batch.column("t2_image")
        for i, name in enumerate(names):
            parsed = parse_image_name(name)
            ref = None
            if parsed:
                split, event, number = parsed
                ref = SceneRef(f"{event}_{number}", event, split, shard.name, row_index)

            def load_images(i: int = i) -> tuple[bytes, bytes]:
                return cell_bytes(pre_col[i].as_py(), "t1_image"), cell_bytes(post_col[i].as_py(), "t2_image")

            yield ParquetRow(ref, name, cell_bytes(t1_masks[i], "t1_mask"), cell_bytes(t2_masks[i], "t2_mask"), load_images)
            row_index += 1


def row_count(shard: Path) -> int:
    return pq.ParquetFile(shard).metadata.num_rows


def read_cells(shard: Path, row: int) -> dict[str, bytes]:
    """Fetch one row's image and mask PNG bytes by position (reads only that row's row group)."""
    parquet = pq.ParquetFile(shard, **_READ_OPTIONS)
    start = 0
    for group in range(parquet.metadata.num_row_groups):
        n = parquet.metadata.row_group(group).num_rows
        if row < start + n:
            columns = [*IMAGE_COLUMNS, *MASK_COLUMNS]
            offset = row - start
            # Stream the row group in small batches and stop at the target row (avoids loading it whole).
            for batch in parquet.iter_batches(batch_size=BATCH_SIZE, row_groups=[group], columns=columns, use_threads=False):
                if offset < batch.num_rows:
                    return {c: cell_bytes(batch.column(c)[offset].as_py(), c) for c in columns}
                offset -= batch.num_rows
        start += n
    raise ParquetFormatError(f"row {row} is beyond the end of {shard.name}")


def decode_mask(png: bytes) -> np.ndarray:
    """Decode a single-channel mask PNG to a 2-D integer array. Raises OSError/ValueError if unusable."""
    with Image.open(io.BytesIO(png)) as img:
        img.load()
        if img.mode not in ("L", "P", "I;16", "I"):
            raise ValueError(f"mask has {img.mode} mode, expected single-channel")
        return np.asarray(img)

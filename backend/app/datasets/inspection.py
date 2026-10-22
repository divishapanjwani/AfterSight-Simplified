"""Dataset inspection: verify the xBD files on disk and summarise what is actually there.

Two input formats (config `dataset_format`):
  - xbd_parquet: the Hugging Face-style Parquet mirror (masks, no polygons) — the real data in use;
  - xbd_json:    the original folder layout with JSON polygons — kept for the synthetic fixture.

Streams one scene at a time, so it scales to the full dataset on a laptop. Every number in the
report is counted from the data; the two estimates are labelled as estimates.
"""

import csv
import io
import json
import resource
import sys
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError
from skimage.registration import phase_cross_correlation

from app.core.damage_scale import DAMAGE_CLASSES
from app.core.events_config import EventsConfig
from app.datasets import scene_cache, xbd_parquet
from app.datasets.annotations import (
    AnnotationError,
    Building,
    DropReason,
    DroppedBuilding,
    building_uids,
    load_label_json,
    parse_post_labels,
)
from app.datasets.mask_buildings import MaskParseResult, MixedBlob, buildings_from_masks
from app.datasets.xbd_layout import DiscoveryResult, SceneFiles, discover_scenes

REPORT_SCHEMA_VERSION = 2
FEATURES_PER_BUILDING_ESTIMATE = 45  # upper end of the planned feature vector (Phase 2)
ALIGNMENT_DOWNSAMPLE = 4  # phase correlation on 1/4-size grayscale keeps it fast
SIZE_BINS = [0, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]  # px; last bin is open-ended
IMAGE_ERRORS = (OSError, UnidentifiedImageError, ValueError, SyntaxError)

# A loader returns (pre image, post image, buildings) for drawing a sample figure.
SampleLoader = Callable[[], tuple[Image.Image, Image.Image, list[Building]]]


@dataclass
class SceneRecord:
    scene_id: str
    event: str
    xbd_split: str
    disaster_type: str | None = None
    pair_complete: bool = False
    unreadable_images: int = 0
    usable: bool = False
    issues: list[str] = field(default_factory=list)
    width: int | None = None
    height: int | None = None
    image_mode: str | None = None
    n_raw_footprints: int = 0  # polygons in the label file, or blobs in the footprint mask
    n_buildings: int = 0
    class_counts: dict[str, int] = field(default_factory=dict)
    n_dropped: int = 0
    n_repaired: int = 0
    n_clipped: int = 0
    n_mixed_label: int = 0
    n_touching_border: int = 0
    footprint_mask_iou: float | None = None  # Parquet only: footprint mask vs damage-labelled pixels
    n_lng_lat_features: int = 0
    pre_post_uid_mismatch: int | None = None
    alignment_shift_px: float | None = None
    alignment_flag: bool = False
    raw_bytes: int = 0


@dataclass
class EventSummary:
    """Mask-derived counts for one event over ALL scenes in the mirror (Parquet only)."""

    scenes: int = 0
    scenes_without_buildings: int = 0
    mask_errors: int = 0
    blobs: int = 0
    mixed_label_blobs: int = 0
    class_counts: Counter[str] = field(default_factory=Counter)
    drop_reasons: Counter[str] = field(default_factory=Counter)


@dataclass
class MaskTotals:
    """Pixel-level mask checks over every scene of every event (Parquet only)."""

    footprint_values: Counter[int] = field(default_factory=Counter)
    damage_values: Counter[int] = field(default_factory=Counter)
    footprint_px_without_damage_value: int = 0
    damage_px_outside_footprint: int = 0

    def add(self, parsed: MaskParseResult) -> None:
        self.footprint_values.update(parsed.footprint_values)
        self.damage_values.update(parsed.damage_values)
        self.footprint_px_without_damage_value += parsed.footprint_px_without_damage_value
        self.damage_px_outside_footprint += parsed.damage_px_outside_footprint


@dataclass
class InspectionResult:
    config: EventsConfig
    discovery: DiscoveryResult
    scenes: list[SceneRecord] = field(default_factory=list)
    dropped: list[DroppedBuilding] = field(default_factory=list)
    mixed: list[MixedBlob] = field(default_factory=list)
    kept_areas: list[float] = field(default_factory=list)
    all_blob_areas: list[int] | None = None  # every footprint blob before filtering (Parquet, selected events)
    alignment_checked: bool = False
    peak_rss_bytes: int = 0
    generated_at: str = ""
    sample_keys: list[str] = field(default_factory=list)
    sample_loaders: dict[str, SampleLoader] = field(default_factory=dict)
    all_events: dict[str, EventSummary] | None = None
    mask_totals: MaskTotals | None = None
    shards: list[str] | None = None
    schema_problems: dict[str, list[str]] = field(default_factory=dict)
    unexpected_image_names: list[str] = field(default_factory=list)
    cache: dict[str, Any] | None = None

    @property
    def usable_scenes(self) -> list[SceneRecord]:
        return [s for s in self.scenes if s.usable]

    def add_scene(self, record: SceneRecord, buildings: list[Building], dropped: list[DroppedBuilding],
                  mixed: list[MixedBlob], loader: SampleLoader | None) -> None:
        self.scenes.append(record)
        if not record.usable:
            return
        self.dropped.extend(dropped)
        self.mixed.extend(mixed)
        self.kept_areas.extend(b.area_px for b in buildings)
        if loader is not None:
            self.sample_loaders[f"{record.xbd_split}/{record.scene_id}"] = loader

    def to_report(self) -> dict[str, Any]:
        usable = self.usable_scenes
        overall: Counter[str] = Counter()
        by_event: dict[str, Counter[str]] = defaultdict(Counter)
        by_split: dict[str, Counter[str]] = defaultdict(Counter)
        for s in usable:
            for cls, n in s.class_counts.items():
                overall[cls] += n
                by_event[s.event][cls] += n
                by_split[s.xbd_split][cls] += n

        drop_reasons = Counter(d.reason.value for d in self.dropped)
        split_of_scene = {s.scene_id: s.xbd_split for s in usable}
        unlabelled_by_split = Counter(split_of_scene[d.scene_id] for d in self.dropped if d.reason is DropReason.LABEL_MISSING)
        disaster_types: dict[str, set[str]] = defaultdict(set)
        for s in self.scenes:
            if s.disaster_type:
                disaster_types[s.event].add(s.disaster_type)
        raw_by_event: Counter[str] = Counter()
        for s in self.scenes:
            raw_by_event[s.event] += s.raw_bytes
        empty_by_event = Counter(s.event for s in usable if s.n_raw_footprints == 0)

        n_buildings = sum(overall.values())
        max_pixels = max((s.width * s.height for s in usable if s.width and s.height), default=0)
        found_events = sorted({s.event for s in self.scenes})
        is_parquet = self.config.dataset_format == "xbd_parquet"

        return {
            "schema_version": REPORT_SCHEMA_VERSION,
            "source": self.config.source,
            "dataset_format": self.config.dataset_format,
            "generated_at": self.generated_at,
            "dataset_root": str(self.config.dataset_root),
            "dataset_root_exists": self.discovery.root_exists,
            "root_subdirs": self.discovery.root_subdirs,
            "footprint_source": (
                "derived from rasterised xBD annotation masks (connected components of t1_mask, "
                f"{self.config.masks.connectivity}-connected; label = majority t2_mask value)"
                if is_parquet else "xBD label JSON polygons (features.xy)"
            ),
            "parquet_shards": self.shards,
            "parquet_schema_problems": self.schema_problems,
            "xbd_splits_found": self.discovery.splits_found,
            "events_requested": self.config.events,
            "events_found": found_events,
            "events_missing": [e for e in self.config.events if e not in found_events],
            "events_on_disk_not_configured": self.discovery.other_events,
            "disaster_type_by_event": {e: sorted(t) for e, t in sorted(disaster_types.items())},
            "disaster_type_source": "config (not stored in the Parquet mirror)" if is_parquet else "label JSON metadata",
            "totals": {
                "scenes_discovered": len(self.scenes),
                "pairs_complete": sum(s.pair_complete for s in self.scenes),
                "scenes_usable": len(usable),
                "scenes_without_buildings": sum(empty_by_event.values()),
                "images_unreadable": sum(s.unreadable_images for s in self.scenes),
                "raw_footprints": sum(s.n_raw_footprints for s in usable),
                "buildings_kept": n_buildings,
                "buildings_dropped": len(self.dropped),
                "buildings_excluded_label": drop_reasons.get(DropReason.LABEL_EXCLUDED.value, 0),
                "buildings_repaired": sum(s.n_repaired for s in usable),
                "buildings_clipped": sum(s.n_clipped for s in usable),
                "mixed_label_blobs": len(self.mixed),
                "mixed_label_blobs_kept": sum(s.n_mixed_label for s in usable),
                "buildings_touching_border": sum(s.n_touching_border for s in usable),
                "unexpected_files": self.discovery.unexpected_file_count + len(self.unexpected_image_names),
            },
            "buildings_by_event": {e: sum(c.values()) for e, c in sorted(by_event.items())},
            "scenes_without_buildings_by_event": dict(sorted(empty_by_event.items())),
            "class_counts": {
                "overall": _ordered(overall),
                "by_event": {e: _ordered(c) for e, c in sorted(by_event.items())},
                "by_xbd_split": {sp: _ordered(c) for sp, c in sorted(by_split.items())},
            },
            "labelled_xbd_splits": {sp: sum(by_split[sp].values()) > 0 for sp in self.discovery.splits_found},
            "buildings_without_label_by_xbd_split": dict(unlabelled_by_split),
            "drop_reasons": dict(sorted(drop_reasons.items())),
            "excluded_label_values": dict(Counter(d.raw_label for d in self.dropped if d.reason is DropReason.LABEL_EXCLUDED)),
            "unknown_label_values": dict(Counter(d.raw_label for d in self.dropped if d.reason is DropReason.LABEL_UNKNOWN)),
            "footprint_size_px": {
                "min_area_px": self.config.geometry.min_area_px,
                "kept": _size_summary(self.kept_areas),
                "all_blobs_before_filter": _size_summary(self.all_blob_areas) if self.all_blob_areas is not None else None,
                "bin_edges": SIZE_BINS,
            },
            "mask_checks": _mask_checks(self.mask_totals, self.config) if self.mask_totals else None,
            "all_events": _all_events_table(self.all_events) if self.all_events is not None else None,
            "image_sizes": dict(Counter(f"{s.width}x{s.height}" for s in usable)),
            "image_modes": dict(Counter(s.image_mode for s in usable)),
            "duplicate_scene_ids": self.discovery.duplicate_scene_ids,
            "unexpected_files_sample": self.discovery.unexpected_files + self.unexpected_image_names[:50],
            "scenes_with_issues": [
                {"scene_id": s.scene_id, "xbd_split": s.xbd_split, "usable": s.usable, "issues": s.issues}
                for s in self.scenes if s.issues
            ],
            "alignment": {
                "checked": self.alignment_checked,
                "flag_threshold_px": self.config.inspection.alignment_flag_px,
                "scenes_flagged": [s.scene_id for s in usable if s.alignment_flag],
                "note": "Phase correlation on downsampled grayscale; real damage also moves the estimate, "
                "so a flag means 'check this pair', not 'misregistered'.",
            },
            "cache": self.cache,
            "resources": {
                "raw_bytes_selected_events": sum(raw_by_event.values()),
                "raw_bytes_by_event": dict(sorted(raw_by_event.items())),
                "peak_rss_bytes_measured": self.peak_rss_bytes,
                "estimates": {
                    "feature_table_bytes_upper": n_buildings * FEATURES_PER_BUILDING_ESTIMATE * 8,
                    "working_ram_per_scene_bytes": max_pixels * (2 * 3 + 2 * 3 * 4),
                    "assumptions": (
                        f"Feature table: buildings_kept x {FEATURES_PER_BUILDING_ESTIMATE} float64 values, "
                        "before Parquet compression. Working RAM: pre+post as uint8 RGB plus float32 copies, "
                        "for the largest image; scenes are processed one at a time."
                    ),
                },
            },
        }


def _ordered(counter: Counter[str]) -> dict[str, int]:
    return {cls: counter.get(cls, 0) for cls in DAMAGE_CLASSES}


def _size_summary(areas: list[float] | list[int]) -> dict[str, Any]:
    if not areas:
        return {"count": 0}
    values = np.asarray(areas, dtype=float)
    hist, _ = np.histogram(values, bins=[*SIZE_BINS, np.inf])
    pct = np.percentile(values, [5, 25, 50, 75, 95])
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "p5": float(pct[0]), "p25": float(pct[1]), "median": float(pct[2]), "p75": float(pct[3]), "p95": float(pct[4]),
        "max": float(values.max()),
        "histogram": hist.astype(int).tolist(),
    }


def _mask_checks(totals: MaskTotals, config: EventsConfig) -> dict[str, Any]:
    known = set(config.masks.damage_values)
    footprint_px = sum(n for v, n in totals.footprint_values.items() if v != 0)
    return {
        "scope": "every scene of every event in the mirror",
        "footprint_mask_values_px": {str(k): v for k, v in sorted(totals.footprint_values.items())},
        "damage_mask_values_px": {str(k): v for k, v in sorted(totals.damage_values.items())},
        "unexpected_damage_values_px": {str(k): v for k, v in sorted(totals.damage_values.items()) if k != 0 and k not in known},
        "damage_value_mapping": {str(k): v for k, v in config.masks.damage_values.items()},
        "footprint_px": footprint_px,
        "footprint_px_without_damage_value": totals.footprint_px_without_damage_value,
        "damage_px_outside_footprint": totals.damage_px_outside_footprint,
    }


def _all_events_table(events: dict[str, EventSummary]) -> dict[str, Any]:
    table = {}
    for name, e in sorted(events.items()):
        table[name] = {
            "scenes": e.scenes,
            "scenes_without_buildings": e.scenes_without_buildings,
            "mask_errors": e.mask_errors,
            "blobs": e.blobs,
            "buildings_kept": sum(e.class_counts.values()),
            "class_counts": _ordered(e.class_counts),
            "mixed_label_blobs": e.mixed_label_blobs,
            "drop_reasons": dict(sorted(e.drop_reasons.items())),
        }
    return table


# ---------------------------------------------------------------- shared helpers


def _open_image(source: Path | bytes) -> Image.Image:
    """Fully decode an image (so truncated files are caught). Raises one of IMAGE_ERRORS if unreadable."""
    with Image.open(io.BytesIO(source) if isinstance(source, bytes) else source) as img:
        img.load()
        return img.copy()


def _gray_small(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L").reduce(ALIGNMENT_DOWNSAMPLE), dtype=np.float32)


def _check_alignment(record: SceneRecord, pre: Image.Image, post: Image.Image, config: EventsConfig) -> None:
    shift, _error, _phase = phase_cross_correlation(_gray_small(pre), _gray_small(post))
    record.alignment_shift_px = round(float(np.hypot(*shift) * ALIGNMENT_DOWNSAMPLE), 2)
    record.alignment_flag = record.alignment_shift_px > config.inspection.alignment_flag_px
    if record.alignment_flag:
        record.issues.append(f"alignment check: estimated shift {record.alignment_shift_px} px")


def _fill_from_buildings(record: SceneRecord, buildings: list[Building], n_dropped: int) -> None:
    record.n_buildings = len(buildings)
    record.class_counts = dict(Counter(b.damage for b in buildings))
    record.n_dropped = n_dropped
    record.n_repaired = sum(b.repaired for b in buildings)
    record.n_clipped = sum(b.clipped for b in buildings)
    record.n_mixed_label = sum(b.mixed_label for b in buildings)
    record.n_touching_border = sum(b.touches_border for b in buildings)


def _peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)  # macOS reports bytes, Linux KiB


# ---------------------------------------------------------------- xbd_json format


def inspect_json_scene(scene: SceneFiles, config: EventsConfig, check_alignment: bool):
    """Inspect one JSON-layout scene. Returns (record, buildings, dropped)."""
    record = SceneRecord(scene.scene_id, scene.event, scene.xbd_split)
    record.raw_bytes = sum(p.stat().st_size for p in scene.files())
    if not scene.complete:
        record.issues.append(f"incomplete pair: missing {', '.join(scene.missing_parts)}")
        return record, [], []
    record.pair_complete = True

    images: dict[str, Image.Image] = {}
    for phase, path in (("pre", scene.pre_image), ("post", scene.post_image)):
        try:
            images[phase] = _open_image(path)
        except IMAGE_ERRORS as exc:
            record.unreadable_images += 1
            record.issues.append(f"unreadable {phase} image: {type(exc).__name__}")
    try:
        post_doc = load_label_json(scene.post_label)
        pre_doc = load_label_json(scene.pre_label)
    except AnnotationError as exc:
        record.issues.append(f"bad label file: {exc}")
        return record, [], []
    if len(images) < 2:
        return record, [], []

    pre, post = images["pre"], images["post"]
    record.image_mode = pre.mode if pre.mode == post.mode else f"{pre.mode}/{post.mode}"
    record.width, record.height = pre.size
    if pre.size != post.size:
        record.issues.append(f"pre/post size differ: {pre.size} vs {post.size}")
        return record, [], []
    if pre.mode != "RGB" or post.mode != "RGB":
        record.issues.append(f"non-RGB image mode: {record.image_mode}")

    parsed = parse_post_labels(post_doc, scene.scene_id, config.excluded_labels, config.geometry.min_area_px, pre.size)
    meta = parsed.metadata
    record.disaster_type = meta.disaster_type
    if meta.width and meta.height and (meta.width, meta.height) != pre.size:
        record.issues.append(f"label metadata size {meta.width}x{meta.height} differs from image {pre.size[0]}x{pre.size[1]}")
    if meta.disaster and meta.disaster != scene.event:
        record.issues.append(f"metadata disaster {meta.disaster!r} differs from filename event")
    record.pre_post_uid_mismatch = len(building_uids(post_doc) ^ building_uids(pre_doc))
    record.n_lng_lat_features = parsed.n_lng_lat_features
    record.n_raw_footprints = sum(
        1 for f in post_doc["features"]["xy"] if (f.get("properties") or {}).get("feature_type") == "building"
    )
    _fill_from_buildings(record, parsed.buildings, len(parsed.dropped))
    if check_alignment:
        _check_alignment(record, pre, post, config)
    record.usable = True
    return record, parsed.buildings, parsed.dropped


def _inspect_json(config: EventsConfig, check_alignment: bool, limit_per_event: int | None,
                  progress: Callable[[int, int], None] | None) -> InspectionResult:
    discovery = discover_scenes(config.resolve(config.dataset_root), config.xbd_splits, config.events, config.image_extensions)
    result = InspectionResult(config=config, discovery=discovery, alignment_checked=check_alignment)
    scenes = _limit(discovery.scenes, limit_per_event)
    for i, scene in enumerate(scenes, start=1):
        record, buildings, dropped = inspect_json_scene(scene, config, check_alignment)

        def loader(scene: SceneFiles = scene) -> tuple[Image.Image, Image.Image, list[Building]]:
            _record, kept, _dropped = inspect_json_scene(scene, config, False)
            return _open_image(scene.pre_image), _open_image(scene.post_image), kept

        result.add_scene(record, buildings, dropped, [], loader)
        if progress:
            progress(i, len(scenes))
    return result


def _limit(scenes: list[SceneFiles], limit_per_event: int | None) -> list[SceneFiles]:
    if limit_per_event is None:
        return scenes
    taken: Counter[str] = Counter()
    kept = []
    for scene in scenes:
        if taken[scene.event] < limit_per_event:
            taken[scene.event] += 1
            kept.append(scene)
    return kept


# ---------------------------------------------------------------- xbd_parquet format


def _masks_to_buildings(scene_id: str, t1_png: bytes, t2_png: bytes, config: EventsConfig) -> MaskParseResult:
    return buildings_from_masks(
        scene_id,
        xbd_parquet.decode_mask(t1_png),
        xbd_parquet.decode_mask(t2_png),
        config.masks.damage_values,
        config.geometry.min_area_px,
        config.masks.connectivity,
    )


def _inspect_parquet(config: EventsConfig, check_alignment: bool, limit_per_event: int | None,
                     progress: Callable[[int, int], None] | None, write_cache: bool) -> InspectionResult:
    root = config.resolve(config.dataset_root)
    root_exists = root.is_dir()
    discovery = DiscoveryResult(root_exists, sorted(p.name for p in root.iterdir() if p.is_dir()) if root_exists else [], [], [], {})
    result = InspectionResult(config=config, discovery=discovery, alignment_checked=check_alignment,
                              all_blob_areas=[], all_events=defaultdict(EventSummary), mask_totals=MaskTotals())
    if not root_exists:
        return result

    shards = xbd_parquet.list_shards(root)
    result.shards = [s.name for s in shards]
    for shard in shards:
        missing = xbd_parquet.missing_columns(shard)
        if missing:
            result.schema_problems[shard.name] = missing
    good = [s for s in shards if s.name not in result.schema_problems]
    total_rows = sum(xbd_parquet.row_count(s) for s in good)

    selected = set(config.events)
    cache_root = config.resolve(config.cache.dir)
    cache = {"enabled": write_cache, "dir": str(config.cache.dir), "scenes_written": 0, "bytes_written": 0}
    taken: Counter[str] = Counter()
    seen: Counter[str] = Counter()
    other: Counter[str] = Counter()
    splits: list[str] = []
    done = 0

    for shard in good:
        for row in xbd_parquet.iter_rows(shard):
            done += 1
            if progress:
                progress(done, total_rows)
            ref = row.ref
            if ref is None:
                result.unexpected_image_names.append(f"{shard.name}: {row.image_name!r}")
                continue
            if ref.xbd_split not in splits:
                splits.append(ref.xbd_split)
            seen[ref.scene_id] += 1
            if seen[ref.scene_id] > 1:
                continue  # reported via duplicate_scene_ids; the first occurrence is used
            is_selected = ref.event in selected and (limit_per_event is None or taken[ref.event] < limit_per_event)

            summary = result.all_events[ref.event]
            summary.scenes += 1
            try:
                parsed = _masks_to_buildings(ref.scene_id, row.t1_mask, row.t2_mask, config)
            except IMAGE_ERRORS as exc:
                summary.mask_errors += 1
                if is_selected:
                    taken[ref.event] += 1
                    record = SceneRecord(ref.scene_id, ref.event, ref.xbd_split, pair_complete=True)
                    record.issues.append(f"unusable mask: {type(exc).__name__}: {exc}")
                    result.add_scene(record, [], [], [], None)
                continue
            summary.blobs += parsed.n_blobs
            summary.scenes_without_buildings += parsed.n_blobs == 0
            summary.mixed_label_blobs += len(parsed.mixed)
            summary.class_counts.update(b.damage for b in parsed.buildings)
            summary.drop_reasons.update(d.reason.value for d in parsed.dropped)
            result.mask_totals.add(parsed)

            if ref.event not in selected:
                other[ref.event] += 1
                continue
            if not is_selected:
                continue
            taken[ref.event] += 1
            record = _parquet_scene_record(ref, row, parsed, config, check_alignment, write_cache, cache_root, cache)
            if record.usable:
                result.all_blob_areas.extend(parsed.blob_areas)

            def loader(shard: Path = shard, row_index: int = ref.row, scene_id: str = ref.scene_id):
                if scene_cache.is_cached(cache_root, scene_id) and write_cache:
                    pre, post, kept, _doc = scene_cache.load_scene(cache_root, scene_id)  # draws what was cached
                    return pre, post, kept
                cells = xbd_parquet.read_cells(shard, row_index)
                kept = _masks_to_buildings(scene_id, cells["t1_mask"], cells["t2_mask"], config).buildings
                return _open_image(cells["t1_image"]), _open_image(cells["t2_image"]), kept

            result.add_scene(record, parsed.buildings, parsed.dropped, parsed.mixed, loader)

    discovery.splits_found = splits
    discovery.other_events = dict(sorted(other.items()))
    discovery.duplicate_scene_ids = sorted(sid for sid, n in seen.items() if n > 1)
    if write_cache:
        usable_ids = {s.scene_id for s in result.usable_scenes}
        cache["stale_scenes_not_in_selection"] = len(scene_cache.stale_scenes(cache_root, usable_ids))
        if usable_ids:
            _write_cache_index(cache_root, result)
    result.cache = cache
    return result


def _parquet_scene_record(ref: xbd_parquet.SceneRef, row: xbd_parquet.ParquetRow, parsed: MaskParseResult,
                          config: EventsConfig, check_alignment: bool, write_cache: bool,
                          cache_root: Path, cache: dict[str, Any]) -> SceneRecord:
    record = SceneRecord(ref.scene_id, ref.event, ref.xbd_split, pair_complete=True)
    record.disaster_type = config.disaster_types.get(ref.event)
    record.n_raw_footprints = parsed.n_blobs
    record.footprint_mask_iou = None if parsed.footprint_iou is None else round(parsed.footprint_iou, 4)
    try:
        pre_png, post_png = row.load_images()
    except xbd_parquet.ParquetFormatError as exc:
        record.issues.append(str(exc))
        return record
    record.raw_bytes = len(pre_png) + len(post_png) + len(row.t1_mask) + len(row.t2_mask)

    images: dict[str, Image.Image] = {}
    for phase, png in (("pre", pre_png), ("post", post_png)):
        try:
            images[phase] = _open_image(png)
        except IMAGE_ERRORS as exc:
            record.unreadable_images += 1
            record.issues.append(f"unreadable {phase} image: {type(exc).__name__}")
    if len(images) < 2:
        return record

    pre, post = images["pre"], images["post"]
    record.image_mode = pre.mode if pre.mode == post.mode else f"{pre.mode}/{post.mode}"
    record.width, record.height = pre.size
    if pre.size != post.size or pre.size != (parsed.width, parsed.height):
        record.issues.append(f"size mismatch: pre {pre.size}, post {post.size}, masks {(parsed.width, parsed.height)}")
        return record
    if pre.mode != "RGB" or post.mode != "RGB":
        record.issues.append(f"non-RGB image mode: {record.image_mode}")
    _fill_from_buildings(record, parsed.buildings, len(parsed.dropped))
    if check_alignment:
        _check_alignment(record, pre, post, config)
    record.usable = True

    if write_cache:
        cache["bytes_written"] += scene_cache.write_scene(
            cache_root, ref.scene_id,
            pngs={"pre": pre_png, "post": post_png, "pre_mask": row.t1_mask, "post_mask": row.t2_mask},
            pre_image=pre, post_image=post, thumbnail_px=config.cache.thumbnail_px,
            meta={"event": ref.event, "xbd_split": ref.xbd_split, "width": pre.size[0], "height": pre.size[1],
                  "source": {"shard": ref.shard, "row": ref.row}},
            buildings=parsed.buildings, dropped=parsed.dropped,
        )
        cache["scenes_written"] += 1
    return record


def _write_cache_index(cache_root: Path, result: InspectionResult) -> None:
    config = result.config
    index = {
        "schema_version": scene_cache.SCENE_SCHEMA_VERSION,
        "source": config.source,
        "events": config.events,
        "min_area_px": config.geometry.min_area_px,
        "connectivity": config.masks.connectivity,
        "scenes": [
            {"scene_id": s.scene_id, "event": s.event, "xbd_split": s.xbd_split, "n_buildings": s.n_buildings}
            for s in result.usable_scenes
        ],
    }
    (cache_root / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- entry points


def run_inspection(
    config: EventsConfig,
    check_alignment: bool = False,
    limit_per_event: int | None = None,
    progress: Callable[[int, int], None] | None = None,
    write_cache: bool = False,
) -> InspectionResult:
    if config.dataset_format == "xbd_parquet":
        result = _inspect_parquet(config, check_alignment, limit_per_event, progress, write_cache)
    else:
        result = _inspect_json(config, check_alignment, limit_per_event, progress)
    result.sample_keys = _choose_samples(result)
    result.peak_rss_bytes = _peak_rss_bytes()
    result.generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return result


def _choose_samples(result: InspectionResult) -> list[str]:
    """Pick scenes that show the most damage classes, one per event first, deterministically."""
    n = result.config.inspection.n_samples
    rng = np.random.default_rng(result.config.inspection.seed)
    candidates = [s for s in result.usable_scenes if s.n_buildings > 0]
    tiebreak = {s.scene_id: rng.random() for s in candidates}
    candidates.sort(key=lambda s: (-len(s.class_counts), -s.n_buildings, tiebreak[s.scene_id]))
    chosen: list[SceneRecord] = []
    for s in candidates:
        if len(chosen) < n and s.event not in {c.event for c in chosen}:
            chosen.append(s)
    for s in candidates:
        if len(chosen) < n and s not in chosen:
            chosen.append(s)
    return [f"{s.xbd_split}/{s.scene_id}" for s in chosen]


def write_outputs(result: InspectionResult,
                  figure_writer: Callable[[InspectionResult, Path], list[Path]] | None = None) -> dict[str, Path]:
    """Persist the report (JSON), CSVs (scenes, dropped, mixed-label blobs, class counts) and figures."""
    out_dir = result.config.resolve(result.config.inspection.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = result.to_report()
    paths = {
        "report": out_dir / "inspection_report.json",
        "scenes": out_dir / "scenes.csv",
        "dropped": out_dir / "dropped_buildings.csv",
        "mixed": out_dir / "mixed_label_blobs.csv",
        "class_counts": out_dir / "class_counts.csv",
    }

    scene_fields = [f for f in SceneRecord.__dataclass_fields__ if f not in ("class_counts", "issues")]
    with paths["scenes"].open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=scene_fields + list(DAMAGE_CLASSES) + ["issues"])
        writer.writeheader()
        for s in result.scenes:
            row = {k: v for k, v in asdict(s).items() if k in scene_fields}
            row.update({cls: s.class_counts.get(cls, 0) for cls in DAMAGE_CLASSES})
            row["issues"] = " | ".join(s.issues)
            writer.writerow(row)

    with paths["dropped"].open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["scene_id", "building_id", "raw_label", "reason", "detail"])
        for d in result.dropped:
            writer.writerow([d.scene_id, d.uid, d.raw_label, d.reason.value, d.detail])

    with paths["mixed"].open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["building_id", "value_counts", "chosen_value", "purity"])
        for m in result.mixed:
            writer.writerow([m.building_id, json.dumps(m.value_counts), "" if m.chosen_value is None else m.chosen_value, m.purity])

    with paths["class_counts"].open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["event", "xbd_split", "damage_class", "count"])
        grouped: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        for s in result.usable_scenes:
            grouped[(s.event, s.xbd_split)].update(s.class_counts)
        for (event, split), counts in sorted(grouped.items()):
            for cls in DAMAGE_CLASSES:
                writer.writerow([event, split, cls, counts.get(cls, 0)])

    if figure_writer is not None:
        figures_dir = result.config.resolve(result.config.inspection.figures_dir)
        figures = figure_writer(result, figures_dir)
        report["figures_dir"] = str(result.config.inspection.figures_dir)
        report["figures"] = [str(p.relative_to(figures_dir)) for p in figures]

    paths["report"].write_text(json.dumps(report, indent=2), encoding="utf-8")
    return paths

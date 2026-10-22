"""Parquet-mirror loader and inspector, on two tiny synthetic shards (NOT xBD data)."""

import csv
import json

import pyarrow as pa
import pyarrow.parquet as pq

from app.datasets import xbd_parquet
from app.datasets.inspection import run_inspection, write_outputs
from app.datasets.inspection_figures import write_inspection_figures


def test_parse_image_name() -> None:
    assert xbd_parquet.parse_image_name("train/images/socal-fire_00000106") == ("train", "socal-fire", "00000106")
    assert xbd_parquet.parse_image_name("train/images/Not A Scene") is None
    assert xbd_parquet.parse_image_name("socal-fire_00000106") is None


def test_iter_rows_streams_every_row_in_order(synthetic_parquet_root) -> None:
    shards = xbd_parquet.list_shards(synthetic_parquet_root)
    assert [s.name for s in shards] == ["train-00000-of-00002.parquet", "train-00001-of-00002.parquet"]
    rows = [r for shard in shards for r in xbd_parquet.iter_rows(shard)]
    assert [r.image_name.rsplit("/", 1)[1] for r in rows] == [
        "synthetic-flood_00000000", "synthetic-flood_00000001", "synthetic-fire_00000000",
        "synthetic-fire_00000001", "synthetic-quake_00000000", "Not A Scene",
    ]
    assert rows[0].ref.scene_id == "synthetic-flood_00000000" and rows[0].ref.row == 0
    assert rows[1].ref.row == 1  # second row group of shard 0
    assert rows[5].ref is None
    pre, post = rows[0].load_images()
    assert pre.startswith(b"\x89PNG") and post.startswith(b"\x89PNG")


def test_read_cells_by_position_matches_streaming(synthetic_parquet_root) -> None:
    shard = xbd_parquet.list_shards(synthetic_parquet_root)[0]
    streamed = list(xbd_parquet.iter_rows(shard))[1]
    cells = xbd_parquet.read_cells(shard, 1)  # lives in row group 1
    assert cells["t1_mask"] == streamed.t1_mask
    assert cells["t1_image"] == streamed.load_images()[0]


def test_missing_columns_are_detected(tmp_path) -> None:
    path = tmp_path / "bad.parquet"
    pq.write_table(pa.table({"image_name": ["train/images/x_00000000"]}), path)
    assert xbd_parquet.missing_columns(path) == ["t1_image", "t2_image", "t1_mask", "t2_mask"]


def test_inspector_counts_match_the_fixture_design(parquet_config) -> None:
    report = run_inspection(parquet_config).to_report()
    t = report["totals"]
    assert report["dataset_format"] == "xbd_parquet"
    assert report["xbd_splits_found"] == ["train"]
    assert t["scenes_discovered"] == 4  # flood x2 + fire x2; quake not configured
    assert t["scenes_usable"] == 3 and t["images_unreadable"] == 1  # fire_1 has a corrupt post image
    assert t["scenes_without_buildings"] == 1
    assert t["buildings_kept"] == 6
    assert report["class_counts"]["overall"] == {"no-damage": 2, "minor-damage": 1, "major-damage": 1, "destroyed": 2}
    assert report["buildings_by_event"] == {"synthetic-fire": 2, "synthetic-flood": 4}
    assert report["drop_reasons"] == {"degenerate_shape": 1, "label_missing": 1, "label_tie": 1, "label_unknown": 1, "too_small": 1}
    assert report["unknown_label_values"] == {"5": 1}
    assert t["mixed_label_blobs"] == 2 and t["mixed_label_blobs_kept"] == 1
    assert t["buildings_touching_border"] == 1
    assert report["events_on_disk_not_configured"] == {"synthetic-quake": 1}
    assert t["unexpected_files"] == 1  # the malformed image_name

    checks = report["mask_checks"]
    assert checks["unexpected_damage_values_px"] == {"5": 100}
    assert checks["damage_px_outside_footprint"] == 25
    assert set(report["all_events"]) == {"synthetic-fire", "synthetic-flood", "synthetic-quake"}
    assert report["all_events"]["synthetic-quake"]["class_counts"]["no-damage"] == 1
    assert report["footprint_size_px"]["all_blobs_before_filter"]["count"] == 9 + 2  # flood_0 + fire_0


def test_cache_holds_only_selected_usable_scenes(parquet_config) -> None:
    result = run_inspection(parquet_config, write_cache=True)
    cache_root = parquet_config.cache.dir
    cached = sorted(p.name for p in cache_root.iterdir() if p.is_dir())
    assert cached == ["synthetic-fire_00000000", "synthetic-flood_00000000", "synthetic-flood_00000001"]
    folder = cache_root / "synthetic-flood_00000000"
    assert {p.name for p in folder.iterdir()} == {
        "pre.png", "post.png", "pre_mask.png", "post_mask.png", "pre_thumb.jpg", "post_thumb.jpg", "scene.json"}
    scene = json.loads((folder / "scene.json").read_text())
    assert [b["building_id"] for b in scene["buildings"]] == [f"synthetic-flood_00000000_b000{i}" for i in (1, 2, 3, 4)]
    assert "derived from rasterised xBD annotation masks" in scene["footprint_source"]
    index = json.loads((cache_root / "index.json").read_text())
    assert len(index["scenes"]) == 3
    assert result.cache["scenes_written"] == 3


def test_writes_outputs_and_figures(parquet_config) -> None:
    result = run_inspection(parquet_config)
    paths = write_outputs(result, write_inspection_figures)
    with paths["mixed"].open() as fh:
        assert len(list(csv.DictReader(fh))) == 2
    figures = json.loads(paths["report"].read_text())["figures"]
    assert {"class_distribution.png", "class_distribution_all_events.png", "footprint_sizes.png"} <= set(figures)
    assert sum(f.startswith("sample_") for f in figures) == 2  # only two usable scenes have buildings


def test_missing_root(parquet_config, tmp_path) -> None:
    config = parquet_config.model_copy(update={"dataset_root": tmp_path / "absent"})
    report = run_inspection(config).to_report()
    assert report["dataset_root_exists"] is False and report["totals"]["scenes_usable"] == 0


def test_cached_scene_round_trips(parquet_config) -> None:
    from app.datasets.scene_cache import load_scene

    result = run_inspection(parquet_config, write_cache=True)
    pre, post, buildings, doc = load_scene(parquet_config.cache.dir, "synthetic-flood_00000000")
    assert pre.size == (128, 128) and post.mode == "RGB"
    assert [b.damage for b in buildings] == ["minor-damage", "no-damage", "destroyed", "major-damage"]
    assert doc["source"] == {"shard": "train-00000-of-00002.parquet", "row": 0}
    # the inspector's sample loader reads the cache when it was written this run
    key = result.sample_keys[0]
    assert len(result.sample_loaders[key]()[2]) > 0
    assert run_inspection(parquet_config).to_report()["disaster_type_source"].startswith("config")

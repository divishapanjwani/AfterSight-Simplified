from pathlib import Path

from app.datasets.xbd_layout import discover_scenes

EVENTS = ["synthetic-flood", "synthetic-fire"]


def test_discovers_and_pairs_scenes(synthetic_root: Path) -> None:
    result = discover_scenes(synthetic_root, ["train", "tier3", "test", "hold"], EVENTS, [".png"])
    assert result.root_exists
    assert result.splits_found == ["train", "test"]  # tier3/hold absent → skipped, in config order
    assert len(result.scenes) == 9
    assert {s.event for s in result.scenes} == set(EVENTS)
    assert result.other_events == {"synthetic-quake": 1}  # present on disk, not configured

    complete = [s for s in result.scenes if s.complete]
    assert len(complete) == 8
    orphan = next(s for s in result.scenes if not s.complete)
    assert orphan.scene_id == "synthetic-flood_00000004"
    assert orphan.missing_parts == ["post_image", "post_label"]


def test_pairs_point_at_matching_files(synthetic_root: Path) -> None:
    result = discover_scenes(synthetic_root, ["train"], EVENTS, [".png"])
    for scene in result.scenes:
        for path in scene.files():
            assert path.name.startswith(scene.scene_id + "_")


def test_unexpected_and_duplicate_files(tmp_path: Path) -> None:
    images = tmp_path / "train" / "images"
    labels = tmp_path / "train" / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    (images / "notes.txt").write_text("x")
    (images / "ev_00000001_pre_disaster.png").write_bytes(b"")
    (images / "ev_00000001_pre_disaster.PNG.bak").write_bytes(b"")
    (tmp_path / "test" / "images").mkdir(parents=True)
    (tmp_path / "test" / "images" / "ev_00000001_pre_disaster.png").write_bytes(b"")

    result = discover_scenes(tmp_path, ["train", "test"], ["ev"], [".png"])
    assert result.unexpected_file_count == 2
    assert result.duplicate_scene_ids == ["ev_00000001"]  # same scene in two xBD splits


def test_missing_root(tmp_path: Path) -> None:
    result = discover_scenes(tmp_path / "absent", ["train"], ["ev"], [".png"])
    assert not result.root_exists and result.scenes == []

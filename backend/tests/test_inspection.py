import csv
import json

from app.datasets.inspection import run_inspection, write_outputs
from app.datasets.inspection_figures import write_inspection_figures


def test_report_counts_match_the_fixture_design(fixture_config) -> None:
    """Expected numbers are derived by hand from synthetic_fixture.py's documented layout."""
    result = run_inspection(fixture_config, check_alignment=True)
    report = result.to_report()
    t = report["totals"]

    assert report["source"] == "synthetic"
    assert t["scenes_discovered"] == 9
    assert t["pairs_complete"] == 8  # flood_4 has no post files
    assert t["scenes_usable"] == 7  # fire_2 has a corrupt post image
    assert t["images_unreadable"] == 1
    # 7 usable scenes x 8 regular buildings (2 per class) + bowtie (destroyed) + clipped (major)
    assert t["buildings_kept"] == 58
    assert report["class_counts"]["overall"] == {"no-damage": 14, "minor-damage": 14, "major-damage": 15, "destroyed": 15}
    assert t["buildings_repaired"] == 1 and t["buildings_clipped"] == 1
    assert report["drop_reasons"] == {
        "label_excluded": 1, "label_missing": 1, "label_unknown": 1, "not_building": 1,
        "outside_image": 1, "too_small": 1, "wkt_invalid": 1,
    }
    assert report["excluded_label_values"] == {"un-classified": 1}
    assert report["labelled_xbd_splits"] == {"train": True, "test": True}
    assert report["events_missing"] == []
    assert report["alignment"]["scenes_flagged"] == ["synthetic-flood_00000001"]
    assert report["disaster_type_by_event"] == {"synthetic-fire": ["fire"], "synthetic-flood": ["flooding"]}


def test_corrupt_image_is_reported_not_fatal(fixture_config) -> None:
    result = run_inspection(fixture_config)
    corrupt = next(s for s in result.scenes if s.scene_id == "synthetic-fire_00000002")
    assert not corrupt.usable
    assert corrupt.issues == ["unreadable post image: UnidentifiedImageError"]


def test_alignment_is_skipped_unless_requested(fixture_config) -> None:
    result = run_inspection(fixture_config)
    assert all(s.alignment_shift_px is None for s in result.scenes)


def test_limit_per_event(fixture_config) -> None:
    result = run_inspection(fixture_config, limit_per_event=2)
    assert len(result.scenes) == 4


def test_writes_json_csv_and_figures(fixture_config) -> None:
    result = run_inspection(fixture_config)
    paths = write_outputs(result, write_inspection_figures)

    report = json.loads(paths["report"].read_text())
    assert report["totals"]["buildings_kept"] == 58
    with paths["scenes"].open() as fh:
        assert len(list(csv.DictReader(fh))) == 9
    with paths["dropped"].open() as fh:
        assert len(list(csv.DictReader(fh))) == 7
    with paths["class_counts"].open() as fh:
        rows = list(csv.DictReader(fh))
        assert sum(int(r["count"]) for r in rows) == 58

    figures_dir = fixture_config.inspection.figures_dir
    assert (figures_dir / "class_distribution.png").stat().st_size > 0
    assert len(list(figures_dir.glob("sample_*.png"))) == fixture_config.inspection.n_samples


def test_missing_dataset_root_gives_empty_report(fixture_config, tmp_path) -> None:
    config = fixture_config.model_copy(update={"dataset_root": tmp_path / "absent"})
    report = run_inspection(config).to_report()
    assert report["dataset_root_exists"] is False
    assert report["totals"]["scenes_usable"] == 0
    assert report["class_counts"]["overall"] == {"no-damage": 0, "minor-damage": 0, "major-damage": 0, "destroyed": 0}

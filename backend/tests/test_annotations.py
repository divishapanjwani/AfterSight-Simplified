import json
from pathlib import Path

import pytest

from app.datasets.annotations import AnnotationError, DropReason, load_label_json, parse_post_labels

HANDWRITTEN = Path(__file__).parent / "fixtures" / "labels" / "handwritten_post.json"


@pytest.fixture
def parsed():
    doc = load_label_json(HANDWRITTEN)
    return parse_post_labels(doc, "handwritten-test_00000000", ["un-classified"], min_area_px=16)


def test_keeps_valid_repaired_and_clipped_buildings(parsed) -> None:
    kept = {b.uid: b for b in parsed.buildings}
    assert set(kept) == {"ok-square", "bowtie", "crosses-edge"}

    square = kept["ok-square"]
    assert square.damage == "no-damage" and square.area_px == 400
    assert not square.repaired and not square.clipped

    bowtie = kept["bowtie"]  # self-intersecting → make_valid → two triangles of 100 px² each
    assert bowtie.repaired and bowtie.polygon.is_valid
    assert bowtie.area_px == pytest.approx(200)

    edge = kept["crosses-edge"]  # 20x10 box, half of it beyond x=100
    assert edge.clipped and edge.area_px == pytest.approx(100)
    assert edge.polygon.bounds[2] <= 100


def test_every_drop_is_logged_with_a_reason(parsed) -> None:
    reasons = {d.uid: d.reason for d in parsed.dropped}
    assert reasons == {
        "outside": DropReason.OUTSIDE_IMAGE,
        "tiny": DropReason.TOO_SMALL,
        "unclassified": DropReason.LABEL_EXCLUDED,
        "unknown-label": DropReason.LABEL_UNKNOWN,
        "no-label": DropReason.LABEL_MISSING,
        "bad-wkt": DropReason.WKT_INVALID,
        "a-line": DropReason.NOT_POLYGON,
        "zero-area": DropReason.EMPTY_AFTER_REPAIR,
        "road": DropReason.NOT_BUILDING,
        "": DropReason.MISSING_UID,
        "ok-square": DropReason.DUPLICATE_UID,  # second feature with the same uid
    }
    assert parsed.n_features == len(parsed.buildings) + len(parsed.dropped)


def test_unknown_labels_are_never_coerced(parsed) -> None:
    assert all(b.damage in {"no-damage", "minor-damage", "major-damage", "destroyed"} for b in parsed.buildings)
    unknown = [d for d in parsed.dropped if d.reason is DropReason.LABEL_UNKNOWN]
    assert unknown[0].raw_label == "partial-damage"


def test_image_size_argument_overrides_metadata() -> None:
    doc = load_label_json(HANDWRITTEN)
    parsed = parse_post_labels(doc, "x", ["un-classified"], 16, image_size=(200, 200))
    assert {b.uid for b in parsed.buildings} >= {"crosses-edge", "outside"}
    assert not next(b for b in parsed.buildings if b.uid == "crosses-edge").clipped


def test_missing_size_is_an_error() -> None:
    doc = load_label_json(HANDWRITTEN)
    doc["metadata"] = {}
    with pytest.raises(AnnotationError, match="image size unknown"):
        parse_post_labels(doc, "x", [], 16)


@pytest.mark.parametrize("content", ["{not json", "[]", '{"features": {}}', '{"features": {"xy": []}, "metadata": 3}'])
def test_unusable_label_files_raise(tmp_path: Path, content: str) -> None:
    path = tmp_path / "bad.json"
    path.write_text(content)
    with pytest.raises(AnnotationError):
        load_label_json(path)


def test_handwritten_fixture_is_marked_not_xbd() -> None:
    assert "NOT xBD" in json.loads(HANDWRITTEN.read_text())["_comment"]

"""Mask -> building instances, on hand-placed blobs (see synthetic_fixture._parquet_masks)."""

import numpy as np
import pytest

from app.datasets.annotations import DropReason
from app.datasets.mask_buildings import building_id, buildings_from_masks
from app.datasets.synthetic_fixture import _parquet_masks

VALUES = {1: "no-damage", 2: "minor-damage", 3: "major-damage", 4: "destroyed"}


@pytest.fixture
def parsed():
    t1, t2 = _parquet_masks("edge_cases")
    return buildings_from_masks("s_00000000", t1, t2, VALUES, min_area_px=16)


def test_ids_are_deterministic_raster_order(parsed) -> None:
    assert building_id("s_00000000", 7) == "s_00000000_b0007"
    assert parsed.n_blobs == 9
    assert [b.uid for b in parsed.buildings] == [f"s_00000000_b000{i}" for i in (1, 2, 3, 4)]


def test_majority_vote_and_flags(parsed) -> None:
    kept = {b.uid.rsplit("_", 1)[1]: b for b in parsed.buildings}
    assert kept["b0001"].damage == "minor-damage" and kept["b0001"].touches_border
    assert kept["b0002"].damage == "no-damage" and not kept["b0002"].mixed_label
    assert kept["b0003"].damage == "destroyed"
    mixed = kept["b0004"]
    assert mixed.damage == "major-damage" and mixed.mixed_label
    assert mixed.label_purity == pytest.approx(0.7)
    assert mixed.area_px == 200  # exact pixel count, not polygon area


def test_every_drop_has_a_reason_and_value_5_is_never_coerced(parsed) -> None:
    reasons = {d.uid.rsplit("_", 1)[1]: d.reason for d in parsed.dropped}
    assert reasons == {
        "b0005": DropReason.LABEL_TIE,
        "b0006": DropReason.LABEL_MISSING,
        "b0007": DropReason.TOO_SMALL,
        "b0008": DropReason.LABEL_UNKNOWN,
        "b0009": DropReason.DEGENERATE_SHAPE,
    }
    unknown = next(d for d in parsed.dropped if d.reason is DropReason.LABEL_UNKNOWN)
    assert unknown.raw_label == "5"
    assert parsed.damage_values[5] == 100
    assert all(b.damage in VALUES.values() for b in parsed.buildings)


def test_mixed_blobs_are_logged(parsed) -> None:
    logged = {m.building_id.rsplit("_", 1)[1]: m for m in parsed.mixed}
    assert set(logged) == {"b0004", "b0005"}
    assert logged["b0004"].value_counts == {2: 60, 3: 140} and logged["b0004"].chosen_value == 3
    assert logged["b0005"].chosen_value is None  # tie


def test_polygons_trace_the_blob_in_pixel_coordinates(parsed) -> None:
    square = next(b for b in parsed.buildings if b.uid.endswith("b0002"))  # rows 10-19, cols 10-19
    assert square.polygon.is_valid
    minx, miny, maxx, maxy = square.polygon.bounds
    assert (minx, miny, maxx, maxy) == (10.5, 10.5, 19.5, 19.5)  # pixel centres of the boundary


def test_polygon_keeps_holes() -> None:
    t1 = np.zeros((40, 40), np.uint8)
    t1[5:30, 5:30] = 1
    t1[12:20, 12:20] = 0  # courtyard
    building = buildings_from_masks("h_00000000", t1, t1.copy(), VALUES, 16).buildings[0]
    assert len(building.polygon.interiors) == 1
    assert building.area_px == 25 * 25 - 64


def test_mask_agreement_counters(parsed) -> None:
    assert parsed.footprint_px_without_damage_value == 100  # blob b0006
    assert parsed.damage_px_outside_footprint == 25
    assert 0 < parsed.footprint_iou < 1


def test_empty_scene() -> None:
    zeros = np.zeros((32, 32), np.uint8)
    result = buildings_from_masks("e_00000000", zeros, zeros, VALUES, 16)
    assert result.n_blobs == 0 and result.buildings == [] and result.footprint_iou is None


def test_connectivity_choice_matters() -> None:
    t1 = np.zeros((20, 20), np.uint8)
    t1[2:8, 2:8] = 1
    t1[8:14, 8:14] = 1  # touches the first square only at a corner
    assert buildings_from_masks("c_00000000", t1, t1, VALUES, 16, connectivity=4).n_blobs == 2
    assert buildings_from_masks("c_00000000", t1, t1, VALUES, 16, connectivity=8).n_blobs == 1


def test_shape_mismatch_is_an_error() -> None:
    with pytest.raises(ValueError, match="same shape"):
        buildings_from_masks("x", np.zeros((4, 4)), np.zeros((4, 5)), VALUES, 16)

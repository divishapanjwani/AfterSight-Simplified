"""PDF/CSV report wording: thresholds only when usable, scene-level labels, 'not available' for missing values."""

import csv

from app.reports.reports import REPORT_VERSION, na, pdf_name, review_lines, write_csv, write_pdf

POLY = [[[(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 0.0)]]]


def _analysis(threshold=0.48, test_f1=0.5268, needs_review=(True, False)):
    buildings = [{"building_uid": f"s_b{i:04d}", "true_label": "destroyed", "predicted": "destroyed", "correct": True,
                  "confidence": 0.4 if flag else 0.9, "scores": {"no-damage": 0.1, "destroyed": 0.9}, "needs_review": flag,
                  "review_reasons": [], "area_px": 100.0, "mixed_label": False, "touches_border": False, "polygon": POLY}
                 for i, flag in enumerate(needs_review)]
    return {"analysis_id": "0123456789abcdef", "scene_id": "ev-a_00000001", "event": "ev-a", "disaster_type": None,
            "created_at": "t", "model": {"name": "logistic_regression", "label": "Logistic Regression", "trained_at": "t",
                                         "design_label": "within-event generalisation", "test_macro_f1": test_f1},
            "confidence_kind": "probability (not calibrated)", "review_threshold": threshold,
            "provenance": {"split": "test", "design_label": "within-event generalisation", "warning": None, "note": "Held-out test scene."},
            "counts": {"ground_truth": {"no-damage": 0, "destroyed": 2}, "predicted": {"no-damage": 0, "destroyed": 2}},
            "scene_metrics": None, "buildings": buildings}


def test_review_lines_mention_validation_only_with_a_threshold() -> None:
    with_threshold = review_lines(_analysis())
    assert with_threshold[0] == "Suggested for human review: 1 buildings with confidence below 0.48 (threshold chosen on the validation split)."
    without = review_lines(_analysis(threshold=None, needs_review=(False, False)))
    assert without[0] == "This model has no usable review threshold, so no building is flagged for low confidence."
    assert not any("validation" in line or "None" in line for line in without)
    for lines in (with_threshold, without):
        assert "Model-disagreement flags" in lines[1] and "Review Queue" in lines[1]  # explained separately
        assert lines[2].endswith("a model score, not a calibrated probability of being correct.")


def test_missing_values_read_not_available() -> None:
    assert na(None) == "not available" and na(0) == "0" and na(0.5268) == "0.5268"


def test_pdf_cache_name_is_versioned() -> None:
    assert pdf_name("0123456789abcdef") == f"0123456789abcdef.r{REPORT_VERSION}.pdf" and REPORT_VERSION >= 2


def test_pdf_builds_without_threshold_or_test_score(tmp_path) -> None:
    path = write_pdf(_analysis(threshold=None, test_f1=None, needs_review=(False, False)), tmp_path / pdf_name("x"))
    assert path.read_bytes().startswith(b"%PDF")


def test_csv_matches_the_analysis(tmp_path) -> None:
    analysis = _analysis()
    with write_csv(analysis, tmp_path / "a.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["building_uid"] for r in rows] == [b["building_uid"] for b in analysis["buildings"]]
    assert [r["needs_review"] for r in rows] == ["True", "False"] and rows[0]["confidence"] == "0.4"
    assert rows[0]["score_destroyed"] == "0.9" and rows[0]["split"] == "test"

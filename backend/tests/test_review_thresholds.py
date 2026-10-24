"""Review-threshold reading and status: missing / malformed files, missing entries, null thresholds, partial."""

import json

from app.inference.review import read_thresholds, threshold_status, usable_threshold, why_unusable

SERVED = ["logistic_regression", "knn", "decision_tree", "random_forest"]


def _doc(**thresholds):
    return {"rule": "lowest t with accuracy(confidence >= t) >= 0.7", "target_accuracy": 0.7,
            "models": {m: {"threshold": t} for m, t in thresholds.items()}}


ALL = dict(logistic_regression=0.4836, knn=0.5333, decision_tree=0.5257, random_forest=0.3865)


def test_read_missing_malformed_and_wrong_shape(tmp_path) -> None:
    path = tmp_path / "review_thresholds.json"
    assert read_thresholds(path) == (None, "the review threshold file has not been created")
    path.write_text("{oops")
    assert read_thresholds(path) == (None, "the review threshold file is not valid JSON")
    path.write_text(json.dumps({"rule": "x"}))
    assert read_thresholds(path) == (None, "the review threshold file has no 'models' section")
    path.write_text(json.dumps(_doc(**ALL)))
    doc, problem = read_thresholds(path)
    assert problem is None and doc["models"]["knn"]["threshold"] == 0.5333


def test_usable_threshold_only_accepts_numbers_in_0_1() -> None:
    doc = _doc(logistic_regression=0.48, knn=None, decision_tree="0.5", random_forest=True)
    doc["models"]["extra"] = {"threshold": 1.7}
    assert usable_threshold(doc, "logistic_regression") == 0.48
    for model in ("knn", "decision_tree", "random_forest", "extra", "not_listed"):
        assert usable_threshold(doc, model) is None
    assert usable_threshold(None, "knn") is None


def test_reasons_name_the_problem() -> None:
    doc = _doc(logistic_regression=None, knn=0.5)
    assert "no threshold for logistic_regression that reaches 70% accuracy on validation" in why_unusable(doc, None, "logistic_regression")
    assert "decision_tree is not in the review threshold file" in why_unusable(doc, None, "decision_tree")
    assert why_unusable(doc, None, "knn") is None
    assert why_unusable(None, "the review threshold file is not valid JSON", "knn") == "the review threshold file is not valid JSON"


def test_status_all_present_partial_and_none() -> None:
    ready = threshold_status(_doc(**ALL), None, SERVED, [])
    assert ready["state"] == "ready" and ready["thresholds"]["random_forest"] == 0.3865

    partial = threshold_status(_doc(**{**ALL, "knn": None}), None, SERVED, [])
    assert partial["state"] == "partial" and partial["reason"].endswith("missing for knn")

    missing_entry = _doc(**ALL)
    del missing_entry["models"]["decision_tree"]
    assert threshold_status(missing_entry, None, SERVED, [])["state"] == "partial"

    none_usable = threshold_status(_doc(**{m: None for m in ALL}), None, SERVED, [])
    assert none_usable["state"] == "not_ready" and "no served model has a usable threshold" in none_usable["reason"]

    no_file = threshold_status(None, "the review threshold file has not been created", SERVED, [])
    assert no_file["state"] == "not_ready" and no_file["reason"] == "the review threshold file has not been created"


def test_untrained_optional_model_does_not_make_it_partial() -> None:
    doc = _doc(logistic_regression=0.48, knn=0.53, decision_tree=0.53)  # RF skipped by calibration
    status = threshold_status(doc, None, SERVED, ["random_forest"])
    assert status["state"] == "ready" and "random_forest (optional) not trained" in status["reason"]
    assert "random_forest" not in status["thresholds"]

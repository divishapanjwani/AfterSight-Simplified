"""API tests against a synthetic end-to-end deployment (NOT xBD data) and an empty, not-built store."""

import contextlib
import io
import json
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api_fixture import build_deployment, empty_store
from app.api.deps import get_jobs, get_store
from app.core.jobs import JobManager
from app.main import app
from app.reports.reports import REPORT_VERSION

TRAIN_WARNING = "This scene was used for training. Accuracy here is optimistic."


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    return build_deployment(tmp_path_factory.mktemp("deploy"))


@pytest.fixture(scope="module")
def client(store, tmp_path_factory):
    jobs = JobManager(tmp_path_factory.mktemp("jobs"))
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_jobs] = lambda: jobs
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def empty_client(tmp_path):
    previous = app.dependency_overrides.get(get_store)
    app.dependency_overrides[get_store] = lambda: empty_store(tmp_path)
    yield TestClient(app)
    if previous is None:
        app.dependency_overrides.pop(get_store, None)
    else:
        app.dependency_overrides[get_store] = previous  # restore the deployment for later tests


def _scene(client, split: str) -> str:
    items = client.get("/api/scenes", params={"split": split, "has_buildings": True}).json()["items"]
    assert items, f"fixture has no {split} scene with buildings"
    return items[0]["scene_id"]


def _assert_clean_error(response, status: int, code: str | None = None) -> dict:
    assert response.status_code == status, response.text
    body = response.json()
    assert set(body) == {"error"} and "message" in body["error"]
    if code:
        assert body["error"]["code"] == code
    text = response.text
    assert "Traceback" not in text and "/Users/" not in text and "/private/" not in text and "site-packages" not in text
    return body["error"]


# ---------------------------------------------------------------- readiness
def test_health(client) -> None:
    body = client.get("/api/health").json()
    assert body["scope"] == "Damage classification on building footprints derived from xBD annotations."


def test_status_ready_on_built_deployment(client) -> None:
    tiles = client.get("/api/dataset/status").json()["tiles"]
    assert all(tiles[t]["ready"] for t in ("scene_cache", "features", "splits", "models", "review_thresholds"))
    # readiness follows the served models in config (KNN and the Decision Tree). The fixture also has a saved Logistic
    # Regression (like the old files left on disk) and an untrained Random Forest; neither is served or affects the tiles.
    assert tiles["models"]["detail"]["trained"] == {"knn": True, "decision_tree": True}
    assert tiles["models"]["fix"] is None
    assert tiles["models"]["state"] == "ready" and tiles["models"]["reason"] == "all required served models are trained"
    thresholds = tiles["review_thresholds"]
    assert thresholds["state"] == "ready" and set(thresholds["detail"]["thresholds"]) == {"knn", "decision_tree"}
    assert thresholds["reason"] == "usable for every required served model (knn, decision_tree)"


def test_readiness_depends_only_on_knn_and_decision_tree(client, store) -> None:
    """Removing the saved Logistic Regression changes nothing; removing the Decision Tree makes the models tile PARTIAL."""
    from app.core.store import Store
    before = client.get("/api/dataset/status").json()["tiles"]
    for name, expected in (("logistic_regression", "ready"), ("decision_tree", "partial")):
        model_file = store.model_dir() / f"{name}.json"
        hidden = model_file.with_suffix(".json.hidden")
        model_file.rename(hidden)
        app.dependency_overrides[get_store] = lambda: Store(store.paths)
        try:
            tiles = client.get("/api/dataset/status").json()["tiles"]
            assert tiles["models"]["state"] == expected, name
            if name == "logistic_regression":
                assert tiles["models"] == before["models"] and tiles["review_thresholds"] == before["review_thresholds"]
            else:
                assert tiles["models"]["reason"] == "not trained: decision_tree"
        finally:
            hidden.rename(model_file)
            app.dependency_overrides[get_store] = lambda: store


def test_dataset_missing_mode(empty_client) -> None:
    status = empty_client.get("/api/dataset/status").json()
    assert status["demo_mode"] is False
    for name, fix in (("scene_cache", "inspect_dataset.py"), ("features", "build_features.py"), ("splits", "make_splits.py")):
        assert status["tiles"][name]["ready"] is False and fix in status["tiles"][name]["fix"]
    error = _assert_clean_error(empty_client.get("/api/scenes"), 503, "dataset_not_ready")
    assert "inspect_dataset.py" in error["fix"]
    _assert_clean_error(empty_client.post("/api/analysis", json={"scene_id": "x_00000000", "model_name": "knn"}), 503)
    models = empty_client.get("/api/models").json()["models"]
    assert not any(m["trained"] for m in models) and all(m["fix"] for m in models)
    _assert_clean_error(empty_client.get("/api/results/comparison"), 404, "results_not_found")


# ---------------------------------------------------------------- scenes & split honesty
def test_scene_list_filters_and_pagination(client) -> None:
    body = client.get("/api/scenes", params={"split": "test"}).json()
    assert body["default_split"] == "test" and body["split_design_label"] == "within-event generalisation"
    assert body["total"] > 0 and all(s["split"] == "test" for s in body["items"])
    assert body["facets"]["split"]["test"] == body["total"]
    flood = client.get("/api/scenes", params={"event": "synthetic-flood", "page_size": 5, "page": 2}).json()
    assert flood["page"] == 2 and len(flood["items"]) == 5 and all(s["event"] == "synthetic-flood" for s in flood["items"])
    assert all(s["disaster_type"] == "flooding" and s["disaster_type_source"] == "config" for s in flood["items"])
    empty = client.get("/api/scenes", params={"has_buildings": False}).json()
    assert empty["total"] == 3 and all(s["split"] == "not_in_manifest" for s in empty["items"])
    for bad in ({"split": "holdout"}, {"page_size": 500}, {"page": 0}, {"event": "../x"}, {"sort": "password"}):
        _assert_clean_error(client.get("/api/scenes", params=bad), 422, "invalid_request")


def test_every_scene_states_its_split(client) -> None:
    train = client.get(f"/api/scenes/{_scene(client, 'train')}").json()
    assert train["split"] == "train" and train["provenance"]["warning"] == TRAIN_WARNING
    val = client.get(f"/api/scenes/{_scene(client, 'val')}").json()
    assert val["provenance"]["warning"].startswith("This scene was used to choose model settings")
    test = client.get(f"/api/scenes/{_scene(client, 'test')}").json()
    assert test["provenance"]["warning"] is None and "Held-out" in test["provenance"]["note"]
    assert test["buildings"] and test["buildings"][0]["polygon"][0][0]  # pixel-coordinate rings
    assert sum(test["ground_truth_counts"].values()) == len(test["buildings"])


@pytest.mark.parametrize("bad", [
    "../../etc/passwd", "..%2F..%2Fetc%2Fpasswd", "%2e%2e%2f%2e%2e%2fetc%2fpasswd", "synthetic-flood_00000001/..",
    "synthetic-flood_00000001%00", "synthetic-flood_99999999", "SYNTHETIC-FLOOD_00000001", "a" * 300, "~root", ".",
])
def test_scene_id_traversal_and_unknown_ids(client, bad) -> None:
    for url in (f"/api/scenes/{bad}", f"/api/scenes/{bad}/image/pre", f"/api/buildings/{bad}_b0001/crop/pre"):
        response = client.get(url)
        assert b"root:" not in response.content
        if response.status_code == 200:
            # "x/.." and "." are dot-segments: the URL is normalised to /api/scenes/ — the scene LIST, never a file
            assert response.headers["content-type"] == "application/json" and "items" in response.json(), url
            continue
        assert response.status_code in (404, 422), (url, response.status_code)
        _assert_clean_error(response, response.status_code)


def test_images_are_jpegs(client) -> None:
    sid = _scene(client, "test")
    for size, limit in (("thumb", 256), ("view", 1024)):
        response = client.get(f"/api/scenes/{sid}/image/post", params={"size": size})
        assert response.status_code == 200 and response.headers["content-type"] == "image/jpeg"
        image = Image.open(io.BytesIO(response.content))
        assert image.format == "JPEG" and max(image.size) <= limit
    _assert_clean_error(client.get(f"/api/scenes/{sid}/image/during"), 422)
    _assert_clean_error(client.get(f"/api/scenes/{sid}/image/pre", params={"size": "full.png"}), 422)
    uid = client.get(f"/api/scenes/{sid}").json()["buildings"][0]["building_uid"]
    crop = client.get(f"/api/buildings/{uid}/crop/pre", params={"size": 96})
    assert crop.headers["content-type"] == "image/jpeg" and max(Image.open(io.BytesIO(crop.content)).size) <= 96
    _assert_clean_error(client.get(f"/api/buildings/{uid}/crop/pre", params={"size": 4000}), 422, "invalid_request")
    _assert_clean_error(client.get(f"/api/buildings/{sid}_b9999/crop/pre"), 404, "building_not_found")


# ---------------------------------------------------------------- analysis
@pytest.mark.parametrize("model, kind", [("knn", "vote fraction of neighbours"),
                                         ("decision_tree", "class fraction in the leaf")])
def test_analysis_per_model_and_cache(client, store, model, kind) -> None:
    sid = _scene(client, "test")
    first = client.post("/api/analysis", json={"scene_id": sid, "model_name": model})
    assert first.status_code == 200, first.text
    body = first.json()
    assert body["confidence_kind"] == kind and body["provenance"]["split"] == "test"
    assert body["buildings"] and {"true_label", "predicted", "confidence", "scores", "needs_review"} <= set(body["buildings"][0])
    assert sum(body["counts"]["predicted"].values()) == len(body["buildings"])
    second = client.post("/api/analysis", json={"scene_id": sid, "model_name": model}).json()
    assert second["cached"] is True and second["analysis_id"] == body["analysis_id"]
    assert (store.paths.analysis_runs / f"{body['analysis_id']}.json").is_file()
    assert client.get(f"/api/analysis/{body['analysis_id']}").json()["scene_id"] == sid


def test_analysis_on_train_scene_carries_warning(client) -> None:
    body = client.post("/api/analysis", json={"scene_id": _scene(client, "train"), "model_name": "knn"}).json()
    assert body["provenance"]["warning"] == TRAIN_WARNING


def test_analysis_input_validation(client) -> None:
    _assert_clean_error(client.post("/api/analysis", json={"scene_id": "../../etc/passwd", "model_name": "knn"}), 422)
    _assert_clean_error(client.post("/api/analysis", json={"scene_id": "synthetic-flood_00000001", "model_name": "knn", "path": "/etc"}), 422)
    _assert_clean_error(client.post("/api/analysis", json={"scene_id": "synthetic-flood_99999999", "model_name": "knn"}), 404, "scene_not_found")
    _assert_clean_error(client.post("/api/analysis", json={"scene_id": "synthetic-flood_00000001", "model_name": "svm"}), 404, "model_not_found")
    _assert_clean_error(client.get("/api/analysis/0123456789abcdef"), 404, "analysis_not_found")
    _assert_clean_error(client.get("/api/analysis/..%2F..%2Fsecret"), 404)


@pytest.mark.parametrize("model", ["logistic_regression", "random_forest", "svm"])
def test_models_not_served_are_rejected_cleanly(client, model) -> None:
    """LR is saved in the fixture (like the old files on disk) and RF is not trained; both are outside the served list,
    so both get exactly the error of an unknown model."""
    sid = _scene(client, "test")
    error = _assert_clean_error(client.post("/api/analysis", json={"scene_id": sid, "model_name": model}), 404, "model_not_found")
    assert error["message"] == "Unknown model. Choose one of: knn, decision_tree."
    _assert_clean_error(client.get("/api/review-queue", params={"model": model}), 404, "model_not_found")
    assert model not in [m["name"] for m in client.get("/api/models").json()["models"]]


def test_logistic_regression_is_not_in_the_research_endpoints(client) -> None:
    """The Arena's endpoints refuse LR like any unknown model, and its saved comparison rows are left out."""
    for model in ("logistic_regression", "svm"):
        _assert_clean_error(client.get(f"/api/models/{model}/metrics"), 422, "invalid_request")
    for design in ("within_event", "cross_event"):
        rows = [m["model"] for m in client.get("/api/results/comparison", params={"design": design}).json()["models"]]
        assert "logistic_regression" not in rows and {"knn", "decision_tree"} <= set(rows)


def test_cached_analysis_of_a_model_no_longer_served(client, store) -> None:
    """An old analysis record of an unserved model stays on disk but is not listed, opened, explained or exported."""
    served = client.post("/api/analysis", json={"scene_id": _scene(client, "test"), "model_name": "knn"}).json()
    old = json.loads((store.paths.analysis_runs / f"{served['analysis_id']}.json").read_text(encoding="utf-8"))
    old["analysis_id"] = "dddddddddddddddd"
    old["model"]["name"] = "logistic_regression"
    path = store.paths.analysis_runs / "dddddddddddddddd.json"
    path.write_text(json.dumps(old), encoding="utf-8")
    try:
        uid = old["buildings"][0]["building_uid"]
        for url in ("/api/analysis/dddddddddddddddd", f"/api/analysis/dddddddddddddddd/buildings/{uid}/explain",
                    "/api/analysis/dddddddddddddddd/export.csv", "/api/analysis/dddddddddddddddd/report.pdf"):
            error = _assert_clean_error(client.get(url), 404, "model_not_served")
            assert "logistic_regression" in error["message"]
        listed = client.get("/api/analyses", params={"limit": 100}).json()
        assert "dddddddddddddddd" not in [i["analysis_id"] for i in listed["items"]]
        assert all(i["model"] in ("knn", "decision_tree") for i in listed["items"])
        on_disk = [json.loads(f.read_text(encoding="utf-8"))["model"]["name"] for f in store.paths.analysis_runs.glob("*.json")]
        assert listed["total"] == sum(m in ("knn", "decision_tree") for m in on_disk) < len(on_disk)
        last = client.get("/api/dataset/status").json()["tiles"]["last_analysis"]
        assert last["detail"]["analysis_id"] != "dddddddddddddddd"
        assert path.is_file()  # the record itself is never deleted
    finally:
        path.unlink()


def test_missing_model_never_triggers_training(client, store) -> None:
    """A served model whose saved file is missing is reported as not trained; analysis never starts training."""
    from app.core.store import Store
    model_file = store.model_dir() / "knn.joblib"
    hidden = model_file.with_suffix(".joblib.hidden")
    model_file.rename(hidden)
    app.dependency_overrides[get_store] = lambda: Store(store.paths)  # fresh store: nothing loaded in memory yet
    try:
        jobs_before = len(app.dependency_overrides[get_jobs]()._jobs)
        error = _assert_clean_error(client.post("/api/analysis", json={"scene_id": _scene(client, "test"), "model_name": "knn"}),
                                    409, "model_not_trained")
        assert error["fix"] == "python scripts/train.py"
        assert len(app.dependency_overrides[get_jobs]()._jobs) == jobs_before
    finally:
        hidden.rename(model_file)
        app.dependency_overrides[get_store] = lambda: store


@pytest.mark.parametrize("model, kind", [("knn", "nearest_neighbours"), ("decision_tree", "decision_path")])
def test_explanations(client, model, kind) -> None:
    analysis = client.post("/api/analysis", json={"scene_id": _scene(client, "test"), "model_name": model}).json()
    building = analysis["buildings"][0]
    body = client.get(f"/api/analysis/{analysis['analysis_id']}/buildings/{building['building_uid']}/explain").json()
    assert body["kind"] == kind and body["predicted"] == building["predicted"]
    if kind == "decision_path":
        # the root-to-leaf questions in raw units; the leaf's majority is the prediction
        steps, fractions = body["detail"]["steps"], body["detail"]["leaf_class_fractions"]
        assert steps and all(s["op"] in ("<=", ">") and (s["value_raw"] <= s["threshold_raw"] + 1e-6) == (s["op"] == "<=") for s in steps)
        assert body["detail"]["prediction"] == building["predicted"] == max(fractions, key=fractions.get)
        assert body["detail"]["leaf_samples"] > 0 and sum(fractions.values()) == pytest.approx(1, abs=1e-3)
    if kind == "nearest_neighbours":
        neighbour = body["detail"]["neighbours"][0]
        assert neighbour["split"] == "train"
        assert client.get(neighbour["crop_urls"]["post"]).headers["content-type"] == "image/jpeg"
    other = client.get(f"/api/scenes/{_scene(client, 'train')}").json()["buildings"][0]["building_uid"]
    _assert_clean_error(client.get(f"/api/analysis/{analysis['analysis_id']}/buildings/{other}/explain"), 404, "building_not_found")


@pytest.mark.parametrize("model", ["knn", "decision_tree"])
def test_exports(client, store, model) -> None:
    analysis = client.post("/api/analysis", json={"scene_id": _scene(client, "test"), "model_name": model}).json()
    csv = client.get(f"/api/analysis/{analysis['analysis_id']}/export.csv")
    assert csv.status_code == 200 and csv.headers["content-type"].startswith("text/csv")
    lines = csv.text.strip().splitlines()
    assert lines[0].startswith("building_uid,scene_id,split,model,predicted,true_label") and len(lines) == 1 + len(analysis["buildings"])
    pdf = client.get(f"/api/analysis/{analysis['analysis_id']}/report.pdf")
    assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF") and len(pdf.content) > 5000
    assert (store.paths.reports / f"{analysis['analysis_id']}.r{REPORT_VERSION}.pdf").is_file()  # versioned cache name


def test_review_queue(client) -> None:
    body = client.get("/api/review-queue", params={"limit": 500 // 5}).json()
    assert body["model"] == "knn"  # the default
    assert body["framing"].startswith("Suggested for human review") and body["split"] == "test"
    confidences = [i["confidence"] for i in body["items"]]
    assert confidences == sorted(confidences) and all(i["reasons"] for i in body["items"])
    assert body["total_flagged"] <= body["total_buildings"]
    assert body["flagged_low_confidence"] + body["flagged_disagreement_only"] == body["total_flagged"]
    _assert_clean_error(client.get("/api/review-queue", params={"split": "everything"}), 422)
    # disagreement is between the served models only: KNN vs the Decision Tree, never the saved Logistic Regression
    for model, other in (("knn", "decision_tree"), ("decision_tree", "knn")):
        items = client.get("/api/review-queue", params={"model": model, "limit": 100}).json()["items"]
        assert all(set(i["other_models"]) == {other} for i in items)
        disagreeing = [r for i in items for r in i["reasons"] if r.startswith("other models disagree")]
        assert items and disagreeing and all(r.startswith(f"other models disagree: {other} says") for r in disagreeing)


@contextlib.contextmanager
def thresholds_file(store, content):
    """Temporarily replace the deployment's review_thresholds.json (None = delete, str = raw text, dict = JSON)."""
    path = store.model_dir() / "review_thresholds.json"
    original = path.read_bytes()
    try:
        if content is None:
            path.unlink()
        else:
            path.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")
        yield
    finally:
        path.write_bytes(original)


def _thresholds_with(store, **changes):
    doc = json.loads((store.model_dir() / "review_thresholds.json").read_text(encoding="utf-8"))
    for model, value in changes.items():
        if value == "drop":
            doc["models"].pop(model)
        else:
            doc["models"][model]["threshold"] = value
    return doc


@pytest.mark.parametrize("content, state, words", [
    (None, "not_ready", "has not been created"),
    ("{not json", "not_ready", "not valid JSON"),
    ({"rule": "r"}, "not_ready", "no 'models' section"),
])
def test_threshold_tile_when_file_unusable(client, store, content, state, words) -> None:
    with thresholds_file(store, content):
        tile = client.get("/api/dataset/status").json()["tiles"]["review_thresholds"]
        assert (tile["ready"], tile["state"]) == (False, state) and words in tile["reason"]
        assert tile["fix"] == "python scripts/calibrate_review.py"
        # a broken file never breaks the other endpoints (it used to return HTTP 500)
        assert client.get("/api/models").status_code == 200
        assert all(m["review_threshold"] is None for m in client.get("/api/models").json()["models"])


def test_threshold_tile_partial_ready_and_not_ready(client, store) -> None:
    tile = lambda: client.get("/api/dataset/status").json()["tiles"]["review_thresholds"]  # noqa: E731
    with thresholds_file(store, _thresholds_with(store, knn="drop")):
        t = tile()
        assert t["state"] == "partial" and t["ready"] is False and "missing for knn" in t["reason"]
    # a model that is not served (the fixture's file also has a Logistic Regression entry) never affects readiness
    with thresholds_file(store, _thresholds_with(store, logistic_regression=None)):
        t = tile()
        assert t["state"] == "ready" and "logistic_regression" not in t["detail"]["thresholds"]
    with thresholds_file(store, _thresholds_with(store, decision_tree=None, knn=None)):
        t = tile()
        assert t["state"] == "not_ready" and "no required served model has a usable threshold" in t["reason"]
    t = tile()  # the real (all present) file again
    assert t["state"] == "ready" and t["ready"] is True and t["fix"] is None


def test_review_queue_without_a_usable_threshold(client, store) -> None:
    params = {"model": "decision_tree", "limit": 200}
    with_threshold = client.get("/api/review-queue", params=params).json()
    assert with_threshold["threshold"] is not None and with_threshold["threshold_note"] is None
    for doc in (_thresholds_with(store, decision_tree=None), _thresholds_with(store, decision_tree="drop"), None):
        with thresholds_file(store, doc):
            body = client.get("/api/review-queue", params=params).json()
            # no threshold -> no low-confidence flags, no rule quoted, and a note saying why
            assert body["threshold"] is None and body["threshold_rule"] is None and body["target_accuracy"] is None
            assert body["flagged_low_confidence"] == 0 and body["threshold_note"]
            assert not any("low confidence" in r for item in body["items"] for r in item["reasons"])
            # model disagreement still works, and the counts agree with the items
            assert body["total_flagged"] == body["flagged_disagreement_only"] and len(body["items"]) == min(200, body["total_flagged"])
            assert all(r.startswith("other models disagree") for item in body["items"] for r in item["reasons"])
            other = client.get("/api/models").json()["models"]
            assert next(m for m in other if m["name"] == "decision_tree")["review_threshold"] is None


def _summary_row(**changes):
    row = {"variant": "change_only", "n_features": 17, "design": "within_event", "model": "logistic_regression",
           "val_macro_f1": 0.4636, "test_macro_f1": 0.4501, "test_recall_major": 0.5006, "test_recall_destroyed": 0.764,
           "event_prediction_accuracy_within_event": 0.6324}
    row.update(changes)
    return {k: v for k, v in row.items() if v != "drop"}


VALID_SUMMARY = {"label": "Phase 3b experiment", "created_at": "t", "variants": {"change_only": "17 features"}, "models": [],
                 "test_evaluations": [], "rows": [_summary_row(), _summary_row(design="cross_event", event_prediction_accuracy_within_event=None)]}


@contextlib.contextmanager
def summary_file(store, content):
    """Temporarily create the Phase 3b summary in the deployment (it has none by default)."""
    path = store.paths.artifacts / "experiments" / "phase3b" / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")
    try:
        yield
    finally:
        path.unlink()


def test_phase3b_summary_valid_is_passed_through_exactly(client, store) -> None:
    with summary_file(store, VALID_SUMMARY):
        body = client.get("/api/results/experiments").json()
        assert body["rows"] == VALID_SUMMARY["rows"] and body["variants"] == VALID_SUMMARY["variants"]
        assert client.get("/api/results/overview").json()["experiments"] is True
    assert client.get("/api/results/overview").json()["experiments"] is False  # missing again


@pytest.mark.parametrize("content, words", [
    ("{not json", "not valid JSON"),
    ({k: v for k, v in VALID_SUMMARY.items() if k != "rows"}, "no valid 'rows'"),
    ({k: v for k, v in VALID_SUMMARY.items() if k != "variants"}, "no valid 'variants'"),
    ({**VALID_SUMMARY, "rows": []}, "no result rows"),
    ({**VALID_SUMMARY, "rows": [_summary_row(test_macro_f1="drop")]}, "row 1 (change_only, logistic_regression) is missing 'test_macro_f1'"),
    ({**VALID_SUMMARY, "rows": [_summary_row(event_prediction_accuracy_within_event="drop")]}, "missing 'event_prediction_accuracy_within_event'"),
    ({**VALID_SUMMARY, "rows": [_summary_row(variant="other")]}, "not described in 'variants'"),
])
def test_phase3b_summary_malformed_or_incomplete(client, store, content, words) -> None:
    with summary_file(store, content):
        error = _assert_clean_error(client.get("/api/results/experiments"), 503, "results_invalid")
        assert words in error["message"] and "run_phase3b.py" in error["fix"]
        assert client.get("/api/results/overview").json()["experiments"] is False  # never shown as a successful run


# ---------------------------------------------------------------- models, results, feature lab, clusters
def test_models_and_results(client) -> None:
    body = client.get("/api/models").json()
    assert body["default_model"] == "knn"
    assert [m["name"] for m in body["models"]] == ["knn", "decision_tree"]  # the served list from config
    assert [m["name"] for m in body["models"] if m["default"]] == ["knn"]
    assert not any(m["optional"] for m in body["models"])
    metrics = client.get("/api/models/knn/metrics").json()
    assert metrics["test_evaluations"] == 1 and "test_per_event" in metrics["metrics"]
    cross = client.get("/api/models/decision_tree/metrics", params={"design": "cross_event"}).json()
    assert cross["design_label"] == "cross-event generalisation"
    tree = client.get("/api/models/decision_tree/tree", params={"max_depth": 2}).json()

    def depth(node):
        return 0 if "left" not in node else 1 + max(depth(node["left"]), depth(node["right"]))
    assert depth(tree["root"]) <= 2
    _assert_clean_error(client.get("/api/models/knn/tree"), 422)
    comparison = client.get("/api/results/comparison").json()
    assert comparison["provenance"].startswith("TEST SET") and comparison["models"]
    assert client.get("/api/results/confounding").json()["target"].startswith("event")
    error = _assert_clean_error(client.get("/api/results/experiments"), 404, "results_not_found")
    assert "run_phase3b.py" in error["fix"]
    assert client.get("/api/results/overview").json()["comparisons"]["within_event/four_class"] is True


def test_feature_lab_and_clusters(client) -> None:
    summary = client.get("/api/features/summary").json()
    assert summary["split"] == "train" and len(summary["features"]) == 37
    assert summary["features"][0]["separation_eta2"] >= summary["features"][-1]["separation_eta2"]
    scatter = client.get("/api/features/scatter", params={"x": "gray_absdiff_mean", "y": "area_px", "n": 100}).json()
    assert scatter["split"] == "train" and scatter["points"]
    _assert_clean_error(client.get("/api/features/scatter", params={"x": "label", "y": "area_px"}), 404, "feature_not_found")
    clusters = client.get("/api/clusters", params={"k": 3}).json()
    assert clusters["never_predictions"] is True and len(clusters["cluster_sizes"]) == 3
    _assert_clean_error(client.get("/api/clusters", params={"k": 9}), 404, "k_not_available")


# ---------------------------------------------------------------- training jobs
def test_training_job_lifecycle(client, monkeypatch) -> None:
    jobs = app.dependency_overrides[get_jobs]()
    import sys
    monkeypatch.setattr(jobs, "training_command", lambda *a: [sys.executable, "-c", "import time; print('training'); time.sleep(0.5)"])
    error = _assert_clean_error(client.post("/api/training/run", json={"confirm": False}), 400, "confirmation_required")
    assert "replaces the saved models" in error["message"]
    _assert_clean_error(client.post("/api/training/run", json={"confirm": True, "designs": ["../../x"]}), 422)
    started = client.post("/api/training/run", json={"confirm": True})
    assert started.status_code == 202
    job_id = started.json()["job_id"]
    _assert_clean_error(client.post("/api/training/run", json={"confirm": True}), 409, "job_running")
    for _ in range(50):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.1)
    assert job["status"] == "succeeded" and job["return_code"] == 0 and "training" in job["log_tail"]
    _assert_clean_error(client.get("/api/jobs/does-not-exist"), 404, "job_not_found")


def test_cors_only_for_vite(client) -> None:
    ok = client.get("/api/health", headers={"Origin": "http://localhost:5200"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:5200"
    assert "access-control-allow-origin" not in client.get("/api/health", headers={"Origin": "http://evil.example"}).headers


def test_unknown_route_is_structured(client) -> None:
    _assert_clean_error(client.get("/api/nope"), 404, "not_found")


def test_recent_analyses_list(client) -> None:
    analysis = client.post("/api/analysis", json={"scene_id": _scene(client, "train"), "model_name": "knn"}).json()
    body = client.get("/api/analyses", params={"limit": 5}).json()
    assert body["total"] >= 1 and len(body["items"]) <= 5
    item = next(i for i in body["items"] if i["analysis_id"] == analysis["analysis_id"])
    assert item["split"] == "train" and item["warning"] == TRAIN_WARNING
    assert item["pdf_url"].endswith("/report.pdf") and item["n_buildings"] == len(analysis["buildings"])
    _assert_clean_error(client.get("/api/analyses", params={"limit": 0}), 422)

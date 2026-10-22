"""API tests against a synthetic end-to-end deployment (NOT xBD data) and an empty, not-built store."""

import io
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api_fixture import build_deployment, empty_store
from app.api.deps import get_jobs, get_store
from app.core.jobs import JobManager
from app.main import app

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
    assert tiles["models"]["detail"]["trained"] == {"logistic_regression": True, "knn": True, "decision_tree": True, "random_forest": False}
    assert tiles["models"]["fix"] == "python scripts/train.py"  # RF missing


def test_dataset_missing_mode(empty_client) -> None:
    status = empty_client.get("/api/dataset/status").json()
    assert status["demo_mode"] is False
    for name, fix in (("scene_cache", "inspect_dataset.py"), ("features", "build_features.py"), ("splits", "make_splits.py")):
        assert status["tiles"][name]["ready"] is False and fix in status["tiles"][name]["fix"]
    error = _assert_clean_error(empty_client.get("/api/scenes"), 503, "dataset_not_ready")
    assert "inspect_dataset.py" in error["fix"]
    _assert_clean_error(empty_client.post("/api/analysis", json={"scene_id": "x_00000000", "model_name": "logistic_regression"}), 503)
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
@pytest.mark.parametrize("model, kind", [("logistic_regression", "probability (not calibrated)"),
                                         ("knn", "vote fraction of neighbours"), ("decision_tree", "class fraction in the leaf")])
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
    body = client.post("/api/analysis", json={"scene_id": _scene(client, "train"), "model_name": "logistic_regression"}).json()
    assert body["provenance"]["warning"] == TRAIN_WARNING


def test_analysis_input_validation(client) -> None:
    _assert_clean_error(client.post("/api/analysis", json={"scene_id": "../../etc/passwd", "model_name": "knn"}), 422)
    _assert_clean_error(client.post("/api/analysis", json={"scene_id": "synthetic-flood_00000001", "model_name": "knn", "path": "/etc"}), 422)
    _assert_clean_error(client.post("/api/analysis", json={"scene_id": "synthetic-flood_99999999", "model_name": "knn"}), 404, "scene_not_found")
    _assert_clean_error(client.post("/api/analysis", json={"scene_id": "synthetic-flood_00000001", "model_name": "svm"}), 404, "model_not_found")
    _assert_clean_error(client.get("/api/analysis/0123456789abcdef"), 404, "analysis_not_found")
    _assert_clean_error(client.get("/api/analysis/..%2F..%2Fsecret"), 404)


def test_missing_model_never_triggers_training(client) -> None:
    jobs_before = len(app.dependency_overrides[get_jobs]()._jobs)
    error = _assert_clean_error(client.post("/api/analysis", json={"scene_id": _scene(client, "test"), "model_name": "random_forest"}),
                                409, "model_not_trained")
    assert error["fix"] == "python scripts/train.py"
    assert len(app.dependency_overrides[get_jobs]()._jobs) == jobs_before


@pytest.mark.parametrize("model, kind", [("logistic_regression", "lr_contributions"), ("decision_tree", "decision_path"),
                                         ("knn", "nearest_neighbours")])
def test_explanations(client, model, kind) -> None:
    analysis = client.post("/api/analysis", json={"scene_id": _scene(client, "test"), "model_name": model}).json()
    building = analysis["buildings"][0]
    body = client.get(f"/api/analysis/{analysis['analysis_id']}/buildings/{building['building_uid']}/explain").json()
    assert body["kind"] == kind and body["predicted"] == building["predicted"]
    if kind == "decision_path":
        assert body["detail"]["prediction"] == building["predicted"] and body["detail"]["steps"]
    if kind == "lr_contributions":
        assert len(body["detail"]["top_features"]) == 8
    if kind == "nearest_neighbours":
        neighbour = body["detail"]["neighbours"][0]
        assert neighbour["split"] == "train"
        assert client.get(neighbour["crop_urls"]["post"]).headers["content-type"] == "image/jpeg"
    other = client.get(f"/api/scenes/{_scene(client, 'train')}").json()["buildings"][0]["building_uid"]
    _assert_clean_error(client.get(f"/api/analysis/{analysis['analysis_id']}/buildings/{other}/explain"), 404, "building_not_found")


def test_exports(client) -> None:
    analysis = client.post("/api/analysis", json={"scene_id": _scene(client, "test"), "model_name": "logistic_regression"}).json()
    csv = client.get(f"/api/analysis/{analysis['analysis_id']}/export.csv")
    assert csv.status_code == 200 and csv.headers["content-type"].startswith("text/csv")
    lines = csv.text.strip().splitlines()
    assert lines[0].startswith("building_uid,scene_id,split,model,predicted,true_label") and len(lines) == 1 + len(analysis["buildings"])
    pdf = client.get(f"/api/analysis/{analysis['analysis_id']}/report.pdf")
    assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF") and len(pdf.content) > 5000


def test_review_queue(client) -> None:
    body = client.get("/api/review-queue", params={"model": "logistic_regression", "limit": 500 // 5}).json()
    assert body["framing"].startswith("Suggested for human review") and body["split"] == "test"
    confidences = [i["confidence"] for i in body["items"]]
    assert confidences == sorted(confidences) and all(i["reasons"] for i in body["items"])
    assert body["total_flagged"] <= body["total_buildings"]
    assert body["flagged_low_confidence"] + body["flagged_disagreement_only"] == body["total_flagged"]
    _assert_clean_error(client.get("/api/review-queue", params={"model": "random_forest"}), 409, "model_not_trained")
    _assert_clean_error(client.get("/api/review-queue", params={"split": "everything"}), 422)


# ---------------------------------------------------------------- models, results, feature lab, clusters
def test_models_and_results(client) -> None:
    body = client.get("/api/models").json()
    assert body["default_model"] == "logistic_regression"
    rf = next(m for m in body["models"] if m["name"] == "random_forest")
    assert rf["optional"] is True and rf["trained"] is False and rf["fix"]
    metrics = client.get("/api/models/logistic_regression/metrics").json()
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
    analysis = client.post("/api/analysis", json={"scene_id": _scene(client, "train"), "model_name": "decision_tree"}).json()
    body = client.get("/api/analyses", params={"limit": 5}).json()
    assert body["total"] >= 1 and len(body["items"]) <= 5
    item = next(i for i in body["items"] if i["analysis_id"] == analysis["analysis_id"])
    assert item["split"] == "train" and item["warning"] == TRAIN_WARNING
    assert item["pdf_url"].endswith("/report.pdf") and item["n_buildings"] == len(analysis["buildings"])
    _assert_clean_error(client.get("/api/analyses", params={"limit": 0}), 422)

"""All API routes. Handlers only validate input and call services; logic lives in app/inference etc."""

from pathlib import Path
from typing import Annotated, Any, Literal, get_args

from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse

from app.api.deps import get_jobs, get_store
from app.core.config import SCOPE_LABEL
from app.core.errors import AppError, InvalidRequest, NotFound
from app.core.jobs import JobManager
from app.core.store import FIX, Store, model_fix
from app.features.schema import FEATURE_COLUMNS
from app.inference import feature_lab, images, review_queue, scenes
from app.inference.analysis import (MODEL_LABELS, explain_building, list_analyses, load_analysis, run_analysis,
                                    served_analysis_records)
from app.experiments import model_comparison
from app.experiments import summary as phase3b_summary
from app.inference.review import usable_threshold
from app.reports.reports import pdf_name, report_context, write_csv, write_pdf
from app.schemas.api import (
    AnalysisRequest, AnalysisResult, DatasetStatus, ErrorResponse, Explanation, Job, ModelList, ReviewQueue,
    SceneDetail, SceneList, StatusTile, TrainingRequest,
)
from app.training.models import CONFIDENCE_KIND

StoreDep = Annotated[Store, Depends(get_store)]
JobsDep = Annotated[JobManager, Depends(get_jobs)]
ERRORS = {r: {"model": ErrorResponse} for r in (404, 409, 422, 503)}
JPEG = {200: {"content": {"image/jpeg": {}}, "description": "JPEG image"}}
Phase = Literal["pre", "post"]
Design = Literal["within_event", "cross_event"]
Scheme = Literal["four_class", "three_class"]

router = APIRouter(responses=ERRORS)


def _jpeg(path: Path) -> FileResponse:
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})


# ---------------------------------------------------------------- readiness
@router.get("/dataset/status", response_model=DatasetStatus, tags=["dataset"])
def dataset_status(store: StoreDep) -> dict[str, Any]:
    tiles: dict[str, StatusTile] = {}

    def tile(name: str, fn, fix: str) -> None:
        try:
            tiles[name] = StatusTile(ready=True, detail=fn())
        except (AppError, FileNotFoundError, KeyError) as exc:
            tiles[name] = StatusTile(ready=False, detail={"reason": getattr(exc, "message", "missing")}, fix=fix)

    tile("scene_cache", lambda: {"scenes": len(store.scene_index()), "events": sorted({s["event"] for s in store.scene_index().values()}),
                                 "buildings": sum(s["n_buildings"] for s in store.scene_index().values())}, FIX["cache"])
    tile("features", lambda: {"buildings": len(store.feature_table.ids), "features": len(FEATURE_COLUMNS),
                              "schema_version": store.feature_table.meta["fingerprint"]["feature_schema_version"]}, FIX["features"])
    tile("splits", lambda: {d: store.manifest(d)["manifest_hash"] for d in store.training_config.splits.designs}, FIX["splits"])
    tiles["models"] = models_tile(store)
    tiles["review_thresholds"] = review_thresholds_tile(store)
    runs = served_analysis_records(store)  # newest first; analyses of models no longer served are not counted
    tiles["last_analysis"] = StatusTile(ready=bool(runs), detail={"analysis_id": runs[0][0].stem, "count": len(runs)} if runs else {},
                                        fix=None if runs else "POST /api/analysis")
    events = store.events_config
    return {"scope": SCOPE_LABEL, "source": events.source if events else None, "demo_mode": False, "tiles": tiles}


def models_tile(store) -> StatusTile:
    """READY when every REQUIRED served model is trained. Optional ones (config: optional: true) never cause PARTIAL."""
    trained = {name: store.model_meta(name) is not None for name in store.serving.models}
    optional_untrained = store.optional_untrained()
    required = [m for m in trained if m not in optional_untrained]
    missing = [m for m in required if not trained[m]]
    trained_required = [m for m in required if trained[m]]
    if not trained_required:
        state, reason = "not_ready", "no served model is trained"
    elif missing:
        state, reason = "partial", "not trained: " + ", ".join(missing)
    else:
        state, reason = "ready", "all required served models are trained"
    if optional_untrained:
        reason += "; " + "; ".join(f"{m} (optional) not trained" for m in optional_untrained)
    return StatusTile(ready=any(trained.values()), state=state, reason=reason,  # `ready` keeps its v1 meaning
                      detail={"serving_design": store.serving.design, "trained": trained,
                              "default_model": store.training_config.default_model},
                      fix=None if all(trained.values()) else model_fix(next(m for m, t in trained.items() if not t)))


def review_thresholds_tile(store) -> StatusTile:
    """READY only when every served model has a usable validation-chosen threshold (see inference/review.py)."""
    status = store.review_threshold_status()
    return StatusTile(ready=status["state"] == "ready", state=status["state"], reason=status["reason"],
                      detail={"rule": status["rule"], "thresholds": status["thresholds"]},
                      fix=None if status["state"] == "ready" else FIX["review"])


# ---------------------------------------------------------------- scenes & images
@router.get("/scenes", response_model=SceneList, tags=["scenes"])
def list_scenes(store: StoreDep,
                event: Annotated[str | None, Query(max_length=60, pattern=r"^[a-z0-9-]+$")] = None,
                disaster_type: Annotated[str | None, Query(max_length=30, pattern=r"^[a-z-]+$")] = None,
                split: Literal["train", "val", "test", "not_in_manifest"] | None = None,
                has_buildings: bool | None = None,
                sort: Literal["scene_id", "-n_buildings", "-destroyed"] = "scene_id",
                page: Annotated[int, Query(ge=1)] = 1,
                page_size: Annotated[int, Query(ge=1, le=100)] = 25) -> dict[str, Any]:
    return scenes.list_scenes(store, event, disaster_type, split, has_buildings, sort, page, page_size)


@router.get("/scenes/{scene_id}", response_model=SceneDetail, tags=["scenes"])
def get_scene(scene_id: str, store: StoreDep) -> dict[str, Any]:
    return scenes.scene_detail(store, scene_id)


@router.get("/scenes/{scene_id}/image/{phase}", response_class=FileResponse, responses=JPEG, tags=["scenes"])
def scene_image(scene_id: str, phase: Phase, store: StoreDep, size: Literal["thumb", "view"] = "view") -> FileResponse:
    return _jpeg(images.scene_image(store, scene_id, phase, size))


@router.get("/buildings/{building_uid}/crop/{phase}", response_class=FileResponse, responses=JPEG, tags=["scenes"])
def building_crop(building_uid: str, phase: Phase, store: StoreDep, size: Annotated[int, Query(description="96, 192 or 384")] = 192) -> FileResponse:
    if size not in images.CROP_SIZES:
        raise InvalidRequest(f"size must be one of {list(images.CROP_SIZES)}.")
    return _jpeg(images.building_crop(store, building_uid, phase, size))


# ---------------------------------------------------------------- feature lab
@router.get("/features/summary", tags=["features"])
def features_summary(store: StoreDep) -> dict[str, Any]:
    return feature_lab.summary(store)


@router.get("/features/scatter", tags=["features"])
def features_scatter(store: StoreDep, x: Annotated[str, Query(max_length=40)], y: Annotated[str, Query(max_length=40)],
                     n: Annotated[int, Query(ge=100, le=5000)] = 1500) -> dict[str, Any]:
    return feature_lab.scatter(store, x, y, n)


# ---------------------------------------------------------------- models & saved results
@router.get("/models", response_model=ModelList, tags=["models"])
def list_models(store: StoreDep) -> dict[str, Any]:
    thresholds = store.review_thresholds()
    models = []
    for name in store.serving.models:
        meta = store.model_meta(name)
        models.append({
            "name": name, "label": MODEL_LABELS[name], "selectable": True,
            "optional": name in store.optional_models(),  # from config (optional: true); no served model is optional now
            "default": name == store.training_config.default_model, "trained": meta is not None,
            "design": store.serving.design, "scheme": store.serving.scheme, "confidence": CONFIDENCE_KIND[name],
            "hyperparameters": meta["hyperparameters"] if meta else None,
            "val_macro_f1": meta["metrics"]["val"]["macro_f1"] if meta else None,
            "test_macro_f1": meta["metrics"]["test"]["macro_f1"] if meta else None,
            "trained_at": meta["created_at"] if meta else None,
            "review_threshold": usable_threshold(thresholds, name),
            "fix": None if meta else model_fix(name),
        })
    design = store.serving.design
    return {"serving_design": design, "serving_design_label": store.training_config.splits.designs[design].label,
            "serving_scheme": store.serving.scheme, "default_model": store.training_config.default_model, "models": models}


# Saved damage-model results the research endpoints read. Logistic Regression is not a damage model of the app, and the
# threshold baseline is no longer in active use: a request for either gets the same validation error as any unknown
# model, and their saved rows are left out of the comparison (the files stay on disk).
ARENA_MODELS = Literal["knn", "decision_tree", "random_forest"]


@router.get("/models/{name}/metrics", tags=["models"])
def model_metrics(name: ARENA_MODELS, store: StoreDep, design: Design = "within_event", scheme: Scheme = "four_class") -> dict[str, Any]:
    meta = store.model_meta(name, design, scheme)
    if meta is None:
        raise NotFound(f"No saved {name} model for {design}/{scheme}.", code="model_not_trained", fix=FIX["models"])
    keep = ("design", "design_label", "scheme", "scheme_label", "classes", "optional", "hyperparameters", "search_on_validation",
            "preprocessing", "n", "train_seconds", "inference_seconds_per_1000_test", "confidence", "metrics", "created_at",
            "test_evaluations", "manifest_hash", "feature_schema_version", "sklearn_version")
    return {"model": name, "label": MODEL_LABELS[name], **{k: meta.get(k) for k in keep}}


@router.get("/models/{name}/tree", tags=["models"])
def model_tree(name: Literal["decision_tree"], store: StoreDep, design: Design = "within_event", scheme: Scheme = "four_class",
               max_depth: Annotated[int, Query(ge=1, le=12)] = 3) -> dict[str, Any]:
    tree = store.read_json(f"models/{design}/{scheme}/decision_tree_structure.json")
    if tree is None:
        raise NotFound("No saved decision tree for this design/scheme.", code="model_not_trained", fix=FIX["models"])

    def prune(node: dict[str, Any]) -> dict[str, Any]:
        if "left" not in node:
            return node
        if node["depth"] >= max_depth:
            return {k: v for k, v in node.items() if k not in ("left", "right")} | {"truncated": True}
        return {**node, "left": prune(node["left"]), "right": prune(node["right"])}

    return {**tree, "view_max_depth": max_depth, "root": prune(tree["root"]), "design": design, "scheme": scheme}


@router.get("/results/overview", tags=["results"])
def results_overview(store: StoreDep) -> dict[str, Any]:
    available = {f"{d}/{s}": store.read_json(f"metrics/{d}/{s}/comparison.json") is not None
                 for d in ("within_event", "cross_event") for s in ("four_class", "three_class")}
    return {"comparisons": available, "primary": "within_event/four_class and cross_event/four_class",
            "confounding": store.read_json("analyses/confounding/within_event.json") is not None,
            "kmeans": {d: store.read_json(f"analyses/kmeans/{d}.json") is not None for d in ("within_event", "cross_event")},
            "experiments": phase3b_summary.summary_state(store)[0] == "available"}


@router.get("/results/comparison", tags=["results"])
def results_comparison(store: StoreDep, design: Design = "within_event", scheme: Scheme = "four_class") -> dict[str, Any]:
    doc = store.read_json(f"metrics/{design}/{scheme}/comparison.json")
    if doc is None:
        raise NotFound("No saved comparison for this design/scheme.", code="results_not_found", fix=FIX["models"])
    design_cfg = store.training_config.splits.designs[design]
    manifest = store.manifest(design)
    return {**doc, "models": [m for m in doc["models"] if m["model"] in get_args(ARENA_MODELS)],
            "split_note": " ".join(design_cfg.description.split()),
            "holdout_event": design_cfg.holdout_event,
            "test_events": sorted(manifest["counts"]["test"]["events"]),
            "experiment": scheme != "four_class", "provenance": "TEST SET (each final model evaluated once)"}


@router.get("/results/confounding", tags=["results"])
def results_confounding(store: StoreDep) -> dict[str, Any]:
    doc = store.read_json("analyses/confounding/within_event.json")
    if doc is None:
        raise NotFound("The confounding check has not been run.", code="results_not_found", fix=FIX["models"])
    return doc


@router.get("/results/model-comparison", tags=["results"])
def results_model_comparison(store: StoreDep) -> dict[str, Any]:
    """SVM, soft-voting ensemble and GridSearchCV results (Model Arena only; none of these models is served)."""
    return model_comparison.load_summary(store)


@router.get("/results/experiments", tags=["results"])
def results_experiments(store: StoreDep) -> dict[str, Any]:
    doc = phase3b_summary.load_summary(store)  # 404 if never run, 503 results_invalid if the file is broken
    return {**doc, "section": "Experiments", "selectable": False,
            "note": "Separate models for analysis only; they are not served in the Triage Desk."}


@router.get("/clusters", tags=["results"])
def clusters(store: StoreDep, k: Annotated[int, Query(ge=2, le=20)] = 4, design: Design = "within_event") -> dict[str, Any]:
    doc = store.read_json(f"analyses/kmeans/{design}.json")
    if doc is None:
        raise NotFound("K-Means has not been run.", code="results_not_found", fix="python scripts/run_kmeans.py")
    if str(k) not in doc["per_k"]:
        raise NotFound(f"k={k} was not computed; available: {doc['k_values']}.", code="k_not_available")
    return {key: doc[key] for key in ("design", "design_label", "classes", "k_values", "chosen_k_by_silhouette", "elbow_k_heuristic",
                                      "pca_explained_variance", "pca_note", "interpretation_note", "fitted_on")} | {
        "k": k, **doc["per_k"][str(k)], "curves": {kk: {"inertia": v["inertia"], "silhouette": v["silhouette_sampled"]} for kk, v in doc["per_k"].items()},
        "never_predictions": True}


# ---------------------------------------------------------------- analysis
@router.post("/analysis", response_model=AnalysisResult, tags=["analysis"])
def create_analysis(request: AnalysisRequest, store: StoreDep) -> dict[str, Any]:
    return run_analysis(store, request.scene_id, request.model_name)


@router.get("/analyses", tags=["reports"])
def recent_analyses(store: StoreDep, limit: Annotated[int, Query(ge=1, le=200)] = 50) -> dict[str, Any]:
    return list_analyses(store, limit)


@router.get("/analysis/{analysis_id}", response_model=AnalysisResult, tags=["analysis"])
def get_analysis(analysis_id: str, store: StoreDep) -> dict[str, Any]:
    return {**load_analysis(store, analysis_id), "cached": True}


@router.get("/analysis/{analysis_id}/buildings/{building_uid}/explain", response_model=Explanation, tags=["analysis"])
def explain(analysis_id: str, building_uid: str, store: StoreDep) -> dict[str, Any]:
    return explain_building(store, analysis_id, building_uid)


@router.get("/analysis/{analysis_id}/export.csv", response_class=FileResponse, tags=["reports"],
            responses={200: {"content": {"text/csv": {}}}})
def export_csv(analysis_id: str, store: StoreDep) -> FileResponse:
    analysis = load_analysis(store, analysis_id)
    path = store.paths.reports / f"{analysis['analysis_id']}.csv"
    if not path.is_file():
        write_csv(analysis, path)
    return FileResponse(path, media_type="text/csv", filename=f"aftersight_{analysis['scene_id']}_{analysis['model']['name']}.csv")


@router.get("/analysis/{analysis_id}/report.pdf", response_class=FileResponse, tags=["reports"],
            responses={200: {"content": {"application/pdf": {}}}})
def export_pdf(analysis_id: str, store: StoreDep) -> FileResponse:
    analysis = load_analysis(store, analysis_id)
    path = store.paths.reports / pdf_name(analysis["analysis_id"])  # versioned: wording changes reach new downloads
    if not path.is_file():
        write_pdf(analysis, path, report_context(store, analysis["model"]["name"]))
    return FileResponse(path, media_type="application/pdf", filename=f"aftersight_{analysis['scene_id']}_{analysis['model']['name']}.pdf")


@router.get("/review-queue", response_model=ReviewQueue, tags=["analysis"])
def get_review_queue(store: StoreDep, model: Annotated[str, Query(max_length=40)] = "knn",
                     split: Literal["train", "val", "test"] = "test",
                     offset: Annotated[int, Query(ge=0)] = 0, limit: Annotated[int, Query(ge=1, le=200)] = 50) -> dict[str, Any]:
    return review_queue.review_queue(store, model, split, offset, limit)


# ---------------------------------------------------------------- training jobs (explicit only)
@router.post("/training/run", response_model=Job, status_code=202, tags=["training"])
def start_training(request: TrainingRequest, jobs: JobsDep) -> dict[str, Any]:
    if not request.confirm:
        raise AppError("Set confirm=true: training replaces the saved models and re-evaluates the test sets.", code="confirmation_required")
    return jobs.start("training", jobs.training_command(request.designs, request.schemes, request.include_optional))


@router.get("/jobs/{job_id}", response_model=Job, tags=["training"])
def get_job(job_id: str, jobs: JobsDep) -> dict[str, Any]:
    return jobs.get(job_id)

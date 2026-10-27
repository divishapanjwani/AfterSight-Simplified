"""Pydantic response/request schemas for the AfterSight API.

Field names carry provenance where it matters: `true_label` is GROUND TRUTH, `predicted` is PREDICTED,
and every scene/analysis states which split it belongs to under the serving model's design.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Split = Literal["train", "val", "test", "not_in_manifest"]
Ring = list[tuple[float, float]]
PolygonParts = list[list[Ring]]  # parts -> rings (exterior first, then holes) -> (x, y) in image pixels


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ErrorBody(BaseModel):
    code: str
    message: str
    fix: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody


class Provenance(BaseModel):
    split: Split
    design: str
    design_label: str
    warning: str | None = Field(None, description="Shown whenever results on this scene are optimistic")
    note: str


# ---------------------------------------------------------------- health / status
class Health(BaseModel):
    status: str
    version: str
    scope: str


class StatusTile(BaseModel):
    ready: bool
    state: Literal["ready", "partial", "not_ready"] | None = Field(
        None, description="Three-way state where a tile can be partly ready (models, review thresholds)")
    reason: str | None = Field(None, description="One line naming the affected models")
    detail: dict[str, Any] = {}
    fix: str | None = Field(None, description="Command that makes this tile ready")


class DatasetStatus(BaseModel):
    scope: str
    source: str | None
    demo_mode: bool = False
    tiles: dict[str, StatusTile]


# ---------------------------------------------------------------- scenes
class SceneQuality(BaseModel):
    alignment_shift_px: float | None
    alignment_flag: bool | None
    n_mixed_label: int
    n_touching_border: int


class SceneSummary(BaseModel):
    scene_id: str
    event: str
    disaster_type: str | None
    disaster_type_source: Literal["config"] = "config"
    xbd_split: str
    split: Split
    n_buildings: int
    ground_truth_counts: dict[str, int]
    quality: SceneQuality
    warnings: list[str]


class SceneList(BaseModel):
    items: list[SceneSummary]
    total: int
    page: int
    page_size: int
    split_design: str
    split_design_label: str
    default_split: Literal["test"] = "test"
    facets: dict[str, dict[str, int]]


class GroundTruthBuilding(BaseModel):
    building_uid: str
    true_label: str = Field(description="GROUND TRUTH (majority value of the xBD damage mask)")
    area_px: float
    label_purity: float
    mixed_label: bool
    touches_border: bool
    polygon: PolygonParts


class SceneDetail(SceneSummary):
    width: int
    height: int
    provenance: Provenance
    n_dropped: int
    dropped_reasons: dict[str, int]
    footprint_source: str
    image_urls: dict[str, str]
    buildings: list[GroundTruthBuilding]


# ---------------------------------------------------------------- models / results
class ModelInfo(BaseModel):
    name: str
    label: str
    selectable: bool
    optional: bool
    default: bool
    trained: bool
    design: str
    scheme: str
    confidence: str
    hyperparameters: dict[str, Any] | None = None
    val_macro_f1: float | None = None
    test_macro_f1: float | None = None
    trained_at: str | None = None
    review_threshold: float | None = None
    fix: str | None = None


class ModelList(BaseModel):
    serving_design: str
    serving_design_label: str
    serving_scheme: str
    default_model: str
    models: list[ModelInfo]


# ---------------------------------------------------------------- analysis
class AnalysisRequest(Strict):
    scene_id: str = Field(min_length=1, max_length=80, pattern=r"^[a-z0-9-]+_\d{8}$")
    model_name: str = Field(min_length=1, max_length=40, pattern=r"^[a-z_]+$")


class BuildingPrediction(BaseModel):
    building_uid: str
    true_label: str = Field(description="GROUND TRUTH")
    predicted: str = Field(description="PREDICTED")
    correct: bool
    confidence: float = Field(description="max class score; meaning given by `confidence_kind`")
    scores: dict[str, float]
    needs_review: bool
    review_reasons: list[str]
    area_px: float
    mixed_label: bool
    touches_border: bool
    polygon: PolygonParts


class AnalysisResult(BaseModel):
    analysis_id: str
    cached: bool
    created_at: str
    scene_id: str
    event: str
    disaster_type: str | None
    model: dict[str, Any]
    confidence_kind: str
    review_threshold: float | None
    review_rule: str | None
    provenance: Provenance
    counts: dict[str, dict[str, int]]
    scene_metrics: dict[str, Any] | None
    seconds: float
    buildings: list[BuildingPrediction]


class Explanation(BaseModel):
    analysis_id: str
    building_uid: str
    model: str
    kind: Literal["decision_path", "nearest_neighbours", "none"]
    predicted: str
    true_label: str
    confidence: float
    confidence_kind: str
    detail: dict[str, Any]
    global_top_features: list[dict[str, Any]]
    crop_urls: dict[str, str]


# ---------------------------------------------------------------- review queue / jobs
class ReviewItem(BaseModel):
    building_uid: str
    scene_id: str
    event: str
    predicted: str
    true_label: str
    confidence: float
    reasons: list[str]
    other_models: dict[str, str]


class ReviewQueue(BaseModel):
    model: str
    split: Split
    framing: str = "Suggested for human review — low confidence or model disagreement, not confirmed errors."
    threshold: float | None
    threshold_rule: str | None
    target_accuracy: float | None = None
    threshold_note: str | None = Field(None, description="Why this model has no usable review threshold (null when it has one)")
    total_buildings: int
    total_flagged: int
    flagged_low_confidence: int
    flagged_disagreement_only: int
    items: list[ReviewItem]
    offset: int
    limit: int


class TrainingRequest(Strict):
    confirm: bool = Field(description="Must be true: training replaces the saved models and re-evaluates test sets")
    designs: list[Literal["within_event", "cross_event"]] | None = None
    schemes: list[Literal["four_class", "three_class"]] | None = None
    include_optional: bool = True


class Job(BaseModel):
    job_id: str
    kind: str
    status: Literal["queued", "running", "succeeded", "failed"]
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    return_code: int | None = None
    log_tail: list[str] = []

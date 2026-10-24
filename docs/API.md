# AfterSight API (Phase 4)

FastAPI, base URL `http://localhost:8020/api`. Interactive docs are at `http://localhost:8020/docs`.
Run with `cd backend && uvicorn app.main:app --port 8020`. The API is file-backed (no database), and
every value it returns is read from an artifact written by the pipeline scripts.

## Rules

- **IDs, never paths.** `scene_id`, `building_uid`, `analysis_id` and `job_id` are pattern-checked and
  must exist in the scene cache, analysis store or job list. File paths are built only from those
  canonical IDs. Traversal attempts return a structured 404 or 422.
- **Structured errors.** Errors look like `{"error": {"code", "message", "fix"?, "details"?}}`. They never
  include a traceback or a server path. When something hasn't been built yet, `fix` gives the command
  that builds it.
- **Split honesty.** Every scene and analysis carries `split` (`train` / `val` / `test` /
  `not_in_manifest`) under the serving design (within-event). Train and val scenes also carry a
  `provenance.warning`. `GET /scenes?split=test` lists held-out scenes; the Triage Desk default is `test`.
- **Analysis never trains.** If a model is missing you get `409 model_not_trained` with
  `fix: python scripts/train.py`. Training runs only via an explicit `POST /training/run {"confirm": true}`.
- **Caching.** Analyses are cached per (scene, model version) in `artifacts/analyses/runs/`. Scene JPEGs,
  crops and reports are written once and then served from disk.
- **CORS** allows only the Vite dev server origin (`http://localhost:5200`).

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness + scope label |
| GET | `/dataset/status` | readiness tiles (cache, features, splits, models, review thresholds, last analysis) with fix commands |
| GET | `/scenes` | `event`, `disaster_type`, `split`, `has_buildings`, `sort`, `page`, `page_size ≤ 100`; facets; split per scene |
| GET | `/scenes/{scene_id}` | ground-truth footprints (pixel polygons), split + provenance, data-quality warnings, image URLs |
| GET | `/scenes/{scene_id}/image/{pre\|post}` | `size=thumb` (256 px) or `view` (≤1024 px). JPEG only |
| GET | `/buildings/{building_uid}/crop/{pre\|post}` | building crop JPEG, `size` 96 / 192 / 384 |
| GET | `/features/summary` | Feature Lab: per-feature class distributions + separation score (η²), **training split only** |
| GET | `/features/scatter` | `x`, `y`, `n`: stratified sample of training buildings for a 2-feature scatter |
| GET | `/models` | Triage Desk models (within-event, 4-class): trained state, default (LR), RF marked optional, review thresholds |
| GET | `/models/{name}/metrics` | saved val/test/per-event metrics, search log, preprocessing, timings (`design`, `scheme`; baseline included) |
| GET | `/models/decision_tree/tree` | tree structure in raw-unit thresholds, pruned to `max_depth` |
| GET | `/results/overview` | which saved result groups exist |
| GET | `/results/comparison` | model comparison per `design` × `scheme` (3-class flagged as an experiment), with `holdout_event` and `test_events` |
| GET | `/results/confounding` | event-prediction check (analysis only) |
| GET | `/results/experiments` | Phase 3b, labelled "Experiments", not selectable |
| GET | `/clusters` | precomputed K-Means for `k` (and `design`): sizes, contingency, ARI, silhouette, PCA points, inertia curves; `never_predictions: true`. 404 `results_not_found` if not run, 404 `k_not_available` for a k that was not computed |
| POST | `/analysis` | `{scene_id, model_name}` → predictions, scores, review flags, counts, scene metrics, provenance |
| GET | `/analyses` | recent cached analyses (newest first) for the Reports page |
| GET | `/analysis/{id}` | a cached analysis |
| GET | `/analysis/{id}/buildings/{uid}/explain` | LR contributions / DT decision path / KNN neighbours (+ crop URLs); RF: global importances only |
| GET | `/analysis/{id}/export.csv` | per-building CSV |
| GET | `/analysis/{id}/report.pdf` | PDF summary with provenance, counts, scene-level metrics, confusion matrix, review flags, limitations. Cached as `outputs/reports/<id>.r<REPORT_VERSION>.pdf` (see `reports.py`) |
| GET | `/review-queue` | `model`, `split` (default test): low confidence (threshold chosen on validation) or model disagreement; counts per reason. Without a usable threshold: `threshold`, `threshold_rule` and `target_accuracy` are null, no low-confidence flags, and `threshold_note` says why |
| POST | `/training/run` | background training job; requires `confirm: true`; one at a time |
| GET | `/jobs/{job_id}` | job status + last 40 log lines (server paths redacted) |

> Since v1.1 the frontend no longer displays `val_macro_f1` from `/models` (it was the Readiness table's
> validation column). The field is still returned, so the API contract is unchanged. Validation macro-F1 is still
> shown in the Model Arena from `/results/comparison`.

## Changes in v1.2 (additive; existing fields keep their meaning)

- `/dataset/status`: tiles may carry `state` (`ready` / `partial` / `not_ready`) and `reason` (one line naming the
  models). The **models** tile counts required served models only (an untrained optional model, e.g.
  `random_forest`, is named in the reason); its `ready` still means "at least one trained". The
  **review_thresholds** tile is `ready` only when every counted served model has a usable threshold, and its
  `detail.thresholds` lists each model's threshold or null. A missing, invalid-JSON or wrongly-shaped
  thresholds file is `not_ready` with the reason (it used to return HTTP 500 on four endpoints).
- A **usable threshold** is a number between 0 and 1. Missing entries and `null` (calibration found none that
  reaches the target) mean no low-confidence flags for that model.
- `/review-queue`: new `threshold_note`.
- `/results/experiments`: `404 results_not_found` when Phase 3b was never run (unchanged); **`503 results_invalid`**
  with the reason and `fix` when `summary.json` is not valid JSON or lacks what the Arena shows (`label`,
  `variants`, rows with `variant`, `design`, `model`, `n_features`, test macro-F1, recall major/destroyed and
  `event_prediction_accuracy_within_event`, which may be null). Valid summaries are returned unchanged.
  `/results/overview.experiments` is true only for a usable summary.

## Changes in v1.3

- `/clusters` is unchanged, but the Cluster Atlas no longer **displays** `curves` (inertia and silhouette per k),
  `inertia`, `elbow_k_heuristic`, `chosen_k_by_silhouette`, `pca_points`, `pca_explained_variance` or `pca_note`.
  They are still returned (and still in `artifacts/analyses/kmeans/<design>.json`).
- `/dataset/status` → `review_thresholds`: only **required** served models decide the state; an optional one
  (config `optional: true`) is named in the reason when it is untrained or has no usable threshold.

## Confidence

`confidence` is the model's highest class score. Its meaning comes from the model metadata (`confidence_kind`):

- **LR:** "probability (not calibrated)";
- **KNN:** "vote fraction of neighbours";
- **DT:** "class fraction in the leaf";
- **RF:** "mean of tree leaf fractions".

The review-queue threshold per model is the lowest `t` on **validation** at which buildings with
confidence ≥ `t` are at least 70% correct (`scripts/calibrate_review.py` →
`artifacts/models/within_event/four_class/review_thresholds.json`):

| Model | t | share flagged on val |
|---|---:|---:|
| LR | 0.484 | 40% |
| KNN | 0.533 | 22% |
| DT | 0.526 | 55% |
| RF | 0.387 | 14% |

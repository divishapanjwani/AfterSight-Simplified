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
| GET | `/models` | the served models (`serving.models` in `config/training.yaml`: KNN and Decision Tree; within-event, 4-class): trained state, default (KNN), review thresholds |
| GET | `/models/{name}/metrics` | saved val/test/per-event metrics, search log, preprocessing, timings (`design`, `scheme`). `name`: `baseline`, `knn`, `decision_tree` or `random_forest` (saved, not served or shown); anything else, including `logistic_regression`, is `422 invalid_request` |
| GET | `/models/decision_tree/tree` | the saved tree's structure in raw-unit thresholds, pruned to `max_depth` (the Arena's tree section) |
| GET | `/results/overview` | which saved result groups exist |
| GET | `/results/comparison` | model comparison per `design` × `scheme` (3-class flagged as an experiment), with `holdout_event` and `test_events`. Saved `logistic_regression` rows are left out |
| GET | `/results/confounding` | event-prediction check (analysis only): an analysis classifier (multinomial logistic regression) predicting the disaster; not a damage model |
| GET | `/results/experiments` | the feature-set experiments (called "Phase 3b" during development, hence `artifacts/experiments/phase3b/` and `scripts/run_phase3b.py`), labelled "Experiments", not selectable |
| GET | `/clusters` | precomputed K-Means for `k` (and `design`): sizes, contingency, ARI, silhouette, PCA points, inertia curves; `never_predictions: true`. 404 `results_not_found` if not run, 404 `k_not_available` for a k that was not computed |
| POST | `/analysis` | `{scene_id, model_name}` → predictions, scores, review flags, counts, scene metrics, provenance. `model_name` must be a served model; anything else (including `logistic_regression` and `random_forest`) is `404 model_not_found` |
| GET | `/analyses` | recent cached analyses of **served** models (newest first) for the Reports page |
| GET | `/analysis/{id}` | a cached analysis. An analysis of a model that is no longer served is `404 model_not_served` (the file stays on disk); the same applies to `/explain`, `export.csv` and `report.pdf` |
| GET | `/analysis/{id}/buildings/{uid}/explain` | `nearest_neighbours` (KNN, + crop URLs) / `decision_path` (Decision Tree: root-to-leaf steps in raw units, leaf size and class shares) |
| GET | `/analysis/{id}/export.csv` | per-building CSV |
| GET | `/analysis/{id}/report.pdf` | PDF summary with provenance, counts, scene-level metrics, confusion matrix, review flags, limitations. Cached as `outputs/reports/<id>.r<REPORT_VERSION>.pdf` (see `reports.py`) |
| GET | `/review-queue` | `model` (a served model), `split` (default test): low confidence (threshold chosen on validation) or disagreement with the other served model (KNN vs Decision Tree); default model `knn`; counts per reason. Without a usable threshold: `threshold`, `threshold_rule` and `target_accuracy` are null, no low-confidence flags, and `threshold_note` says why |
| POST | `/training/run` | background training job; requires `confirm: true`; one at a time |
| GET | `/jobs/{job_id}` | job status + last 40 log lines (server paths redacted) |

> Since v1.1 the frontend no longer displays `val_macro_f1` from `/models` (it was the Readiness table's
> validation column). The field is still returned, so the API contract is unchanged. Validation macro-F1 is still
> shown in the Model Arena from `/results/comparison`.

## Served models: KNN (default) and Decision Tree

- **Served set:** `config/training.yaml` → `serving.models: [knn, decision_tree]`, `default_model: knn`. `/models`,
  `/analysis`, `/explain`, `/review-queue` (and its disagreement reasons and `other_models`), the readiness tiles
  and reports all follow it. The Decision Tree's explanation (`kind: decision_path`) is served again.
- **Logistic Regression is not a damage model of the app:** `logistic_regression` gets `404 model_not_found`
  ("Unknown model. Choose one of: knn, decision_tree.") from `/analysis` and `/review-queue`, and `422` from
  `/models/{name}/metrics`, like any unknown model; its saved rows are left out of `/results/comparison`. Old
  cached LR analyses get `404 model_not_served` and are not listed by `/analyses` or counted by the last-analysis
  tile. The `lr_contributions` explanation kind no longer exists.
- **Random Forest** gets the same `404 model_not_found`; `/models/random_forest/metrics` and its comparison rows
  still return its saved results, which the UI does not show.
- **PDF reports** (`REPORT_VERSION` 3): the limitations quote the report's own model's saved within- and
  cross-event macro-F1 (they used to quote the default model's).
- **Returned but no longer shown by the UI:**
  - `/results/experiments` (the whole feature-set experiment summary) is unchanged and still served, but no page
    requests it: the Arena's Experiments section was removed, together with the feature-set bars in the
    confounding chart;
  - `/results/overview` → `experiments` is still returned and not displayed;
  - `/results/comparison` rows for `random_forest`;
  - `review_thresholds.json` entries for `logistic_regression` and `random_forest` (the file was not regenerated;
    only the served models' entries are read);
  - the `optional` flag on `/models` and comparison rows (no served model is optional now).

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
- `/results/experiments`: `404 results_not_found` when the feature-set experiments were never run (unchanged); **`503 results_invalid`**
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

- **KNN:** "vote fraction of neighbours";
- **Decision Tree:** "class fraction in the leaf".

The review-queue threshold per model is the lowest `t` on **validation** at which buildings with
confidence ≥ `t` are at least 70% correct (`scripts/calibrate_review.py` →
`artifacts/models/within_event/four_class/review_thresholds.json`):

| Model | t | share flagged on val |
|---|---:|---:|
| KNN | 0.533 | 22% |
| Decision Tree | 0.526 | 55% |

The saved file also has entries for Logistic Regression (0.484) and the Random Forest (0.387) from when they were
served; they are not read. `python scripts/calibrate_review.py --out <folder>` recomputes the thresholds into
another folder for checking, without touching the saved file.

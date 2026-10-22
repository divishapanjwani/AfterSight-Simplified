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
| GET | `/clusters` | K-Means for `k` (and `design`): sizes, contingency, ARI, PCA points; `never_predictions: true` |
| POST | `/analysis` | `{scene_id, model_name}` → predictions, scores, review flags, counts, scene metrics, provenance |
| GET | `/analyses` | recent cached analyses (newest first) for the Reports page |
| GET | `/analysis/{id}` | a cached analysis |
| GET | `/analysis/{id}/buildings/{uid}/explain` | LR contributions / DT decision path / KNN neighbours (+ crop URLs); RF: global importances only |
| GET | `/analysis/{id}/export.csv` | per-building CSV |
| GET | `/analysis/{id}/report.pdf` | PDF summary with provenance, counts, metrics, confusion matrix, limitations |
| GET | `/review-queue` | `model`, `split` (default test): low confidence (threshold chosen on validation) or model disagreement; counts per reason and `target_accuracy` |
| POST | `/training/run` | background training job; requires `confirm: true`; one at a time |
| GET | `/jobs/{job_id}` | job status + last 40 log lines (server paths redacted) |

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

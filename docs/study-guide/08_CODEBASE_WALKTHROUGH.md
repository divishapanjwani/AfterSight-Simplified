# 08 · Codebase walkthrough

Every path and function below exists in this clone. There is no database layer: storage is files, read
through `backend/app/core/store.py`.

## Top level

| Path | Purpose |
|---|---|
| `start.sh` | starts the API (8020) and the web app (5200), freeing those ports first |
| `scripts/run_pipeline.sh` | runs the 10 build steps in order (inspect → features → splits → train → model comparison → AdaBoost → serve SVM + soft voting → thresholds → feature-set experiments → demo cache) |
| `config/events.yaml` | events, damage-value mapping, connectivity, minimum blob area, cache settings |
| `config/features.yaml` | ring width, Canny σ, GLCM levels and distance |
| `config/training.yaml` | split designs, seed, skew threshold, model grids, serving set, review target accuracy |
| `backend/` | Python package `app` and tests |
| `frontend/` | React app |
| `data/`, `artifacts/`, `outputs/` | data, saved models and results, reports (git-ignored) |
| `docs/` | documentation, the final paper (`docs/research/`), this guide |

## Pipeline scripts (`scripts/`)

| Script | What it calls | Output |
|---|---|---|
| `inspect_dataset.py` | `app.datasets.inspection` (uses `xbd_parquet.iter_rows`, `mask_buildings.buildings_from_masks`, `scene_cache.write_scene`) | inspection report + scene cache |
| `build_features.py` | `app.features.table.build_feature_table` → `extract.scene_rows` → `building_features` | `features_v1.parquet` |
| `make_splits.py` | `app.training.splits.build_manifest`, `validate_manifest`, `write_manifest` | split manifests |
| `train.py` | `app.training.run.run_design` → `train_one`; `app.evaluation.analysis.run_kmeans`, `run_confounding` | models, metrics, K-Means, confounding check |
| `calibrate_review.py` | `app.inference.review.choose_threshold`, `write_thresholds` (validation only) | `review_thresholds.json` |
| `run_phase3b.py` | `app.experiments.phase3b.run_phase3b` (feature-set experiments; development name "Phase 3b") | `artifacts/experiments/phase3b/` |
| `prepare_demo.py` | `app.inference.analysis.run_analysis`, `app.reports.reports.write_pdf` | 3 pre-cached demo scenes and reports |
| `snapshot_behaviour.py` | the real API routes via FastAPI `TestClient` | proves a change altered no prediction or explanation |
| `paper_stats.py` | reads saved test predictions | scene-level bootstrap → `paper/build/stats.json` |
| `e2e_check.py` | builds a synthetic deployment and runs backend + built frontend | 22 end-to-end checks |

## Backend package (`backend/app/`)

| Module | Key functions and classes | Role |
|---|---|---|
| `datasets/xbd_parquet.py` | `list_shards`, `iter_rows`, `read_cells`, `decode_mask`, `parse_image_name` | stream Parquet rows, decode PNGs (read-only) |
| `datasets/mask_buildings.py` | `buildings_from_masks`, `_blob_polygon`, `building_id` | masks → labelled buildings and polygons |
| `datasets/inspection.py` | inspector, alignment check | dataset report |
| `datasets/scene_cache.py` | `write_scene`, `load_scene`, `is_cached` | per-scene cache on disk |
| `features/schema.py` | `FEATURE_COLUMNS`, `FEATURE_GROUPS` | the 37 names, in order |
| `features/extract.py` | `building_features`, `glcm_contrast_homogeneity`, `compactness`, `scene_rows`, `to_gray` | feature maths |
| `features/table.py` | `build_feature_table`, `load_feature_table`, `FeatureTable` | the feature table and its fingerprint |
| `features/variants.py` | `variant_frame`, `scene_normalise` | feature-set experiment variants |
| `training/splits.py` | `stratified_group_partition`, `build_manifest`, `validate_manifest`, `manifest_hash` | scene-grouped, leakage-checked splits |
| `training/models.py` | `SkewLog1p`, `make_pipeline`, `param_grid`, `ThresholdRule`, `tune_threshold_rule` | pipelines (KNN, DT, RF, baseline; LR only for the confounding check and the feature-set experiment) |
| `training/run.py` | `train_one`, `run_design`, `write_comparison`, `per_event_metrics` | grid search on val, one test evaluation, saving |
| `training/persistence.py` | `save_model`, `load_model`, `ModelLoadError` | joblib + metadata; refuses incompatible models |
| `evaluation/metrics.py` | `classification_metrics`, `TestSetGuard`, `timed_predict` | metrics; one test evaluation per model |
| `evaluation/analysis.py` | `run_kmeans`, `run_confounding` | K-Means and the event-prediction check |
| `explain/explain.py` | `knn_neighbours`, `decision_path`, `tree_structure`, `lr_contributions`, `raw_threshold`, `permutation_importances` | explanations in raw feature units. The app uses `knn_neighbours` and `decision_path`; `tree_structure` saves the tree for the Arena. `lr_contributions` is not used by the app (only by `docs/build/understanding_data.py`) |
| `inference/analysis.py` | `run_analysis`, `predict_rows`, `explain_building`, `load_analysis`, `list_analyses` | per-scene predictions (cached) and explanations |
| `inference/review.py` | `choose_threshold`, `read_thresholds`, `usable_threshold`, `threshold_status` | review thresholds and READY / PARTIAL / NOT READY |
| `inference/review_queue.py` | `review_queue` | low-confidence or disagreement suggestions (disagreement between the served models: KNN vs DT) |
| `inference/feature_lab.py` | `summary`, `scatter`, `eta_squared`, `class_stats` | Feature Lab statistics (training split only) |
| `inference/scenes.py` | `list_scenes`, `scene_detail`, `provenance` | Library and scene details, split provenance |
| `experiments/summary.py` | `load_summary`, `summary_state`, `summary_problem` | validates the saved feature-set experiment summary |
| `reports/reports.py` | `write_csv`, `write_pdf`, `review_lines`, `pdf_name`, `REPORT_VERSION` | CSV/PDF exports (PDF cache file named by version) |
| `core/store.py` | `Store`, `StorePaths`, `FIX` | all file access; validates IDs; loads models |
| `core/errors.py` | `AppError`, `NotFound`, `NotReady`, `ResultsInvalid`, `ModelNotTrained` | structured errors |
| `api/routes.py`, `api/health.py` | all `/api` routes | thin HTTP layer |
| `schemas/api.py` | Pydantic models (`AnalysisResult`, `Explanation`, `StatusTile`, `ReviewQueue`, …) | API contracts |

## The inference path in code

```
POST /api/analysis  (api/routes.py: create_analysis)
  └─ inference/analysis.py: run_analysis(store, scene_id, model_name)
       ├─ store.valid_scene_id / valid_model_name           IDs, never paths
       ├─ store.model(name) → (pipeline, metadata)            training/persistence.load_model
       ├─ cached? artifacts/analyses/runs/<id>.json → return
       └─ predict_rows(pipe, X) → pipe.predict_proba(X)
            predicted = argmax; confidence = max score
            needs_review = confidence < usable_threshold(...)  (validation-chosen)
GET /api/analysis/{id}/buildings/{uid}/explain
  └─ explain_building → knn_neighbours | decision_path   (served models only)
```

## Frontend (`frontend/src/`)

| Path | Role |
|---|---|
| `App.tsx` | routes: `/`, `/library`, `/triage[/:sceneId]`, `/case[/:analysisId/:buildingUid]`, `/arena`, `/features`, `/clusters`, `/review`, `/reports`, fallback |
| `services/api.ts` | `api.*` fetch functions, `ApiError` (`notReady` for 503 / not-trained / not-found results) |
| `types/api.ts` | TypeScript types matching the Pydantic schemas |
| `components/AppShell.tsx` | header, stepper, tools, API status dot, theme toggle |
| `components/CaseFile.tsx` | the Case File (`cropWindow`, `polygonPath`, `ExplanationView`, `KnnView`, `LrView`) |
| `viewer/SceneViewer.tsx` | swipe, zoom and overlay viewer (SVG polygons over the images) |
| `components/States.tsx` | `LoadingState`, `ErrorState`, `NotReadyState`, `EmptyState`, `CommandBlock` |
| `components/Workflow.tsx` | remembers the current scene, model, analysis and building for the stepper links |
| `theme/` | damage colours and patterns, feature labels, light/dark theme |
| `charts/` | `BarList`, `BoxPlot`, `Dumbbell`, `Heatmap`, `ScatterPlot` (hand-built SVG) |

## Tests

| Suite | Count | What it covers |
|---|---:|---|
| `backend/tests/` (pytest) | 149 | parser, features with known answers, split leakage, models, save/reload, API (including path traversal), review thresholds, the feature-set experiment summary, reports |
| `frontend/src/**/*.test.tsx` (Vitest) | 37 | pages and their states, the Triage → Case File flow, Arena, Readiness, Review Queue, Cluster Atlas |
| `scripts/e2e_check.py` | 20 checks | synthetic end-to-end run of the backend and the built frontend |

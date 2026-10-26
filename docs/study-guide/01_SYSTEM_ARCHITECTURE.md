# 01 · System architecture

## The big picture

```
xBD Parquet shards (raw, read-only)
   │  scripts/inspect_dataset.py        masks → buildings, scene cache
   ▼
data/processed/scenes/<scene_id>/       images, masks, scene.json (footprints, labels)
   │  scripts/build_features.py         37 features per building
   ▼
data/processed/features/features_v1.parquet
   │  scripts/make_splits.py            scene-grouped split manifests
   ▼
data/splits/{within_event,cross_event}_v1.json
   │  scripts/train.py                  tune on val, test once, save
   ▼
artifacts/models/…  artifacts/metrics/…  artifacts/analyses/…
   │
   ▼  FastAPI backend (port 8020) reads these files
React frontend (port 5200) ◄── JSON / JPEG / CSV / PDF ──► browser
```

There are two halves:

1. **Offline pipeline (scripts).** It builds everything once. Training happens only here.
2. **Online app (backend + frontend).** It only *loads* saved models and results. Analysis never trains.

## There is no database

All data lives in **files**:

- Parquet tables;
- JSON documents;
- joblib model files;
- JPEG and PNG images.

The backend's `Store` class (`backend/app/core/store.py`) is the only way the API reads them. It builds file
paths only from **validated IDs** (a scene ID must exist in the scene cache index; a model name must be in the
serving list), never from user-supplied paths. That blocks path-traversal attacks.

**Why files and not a database?**

- The data is written once by the pipeline and then only read.
- Files are easy to version, hash and inspect.
- It keeps the project laptop-sized and offline.

A database would add setup cost without adding anything for this read-mostly workload.

## Frontend

| Item | Detail |
|---|---|
| Stack | React 19.3 + TypeScript 6, Vite 8, Tailwind 4, TanStack Query 5 (data fetching and caching), React Router 7 (versions pinned in `frontend/package.json`) |
| Pages | `frontend/src/pages/`: `ReadinessPage`, `SceneLibraryPage`, `TriageDeskPage`, `CasePage`, `ModelArenaPage`, `FeatureLabPage`, `ClusterAtlasPage`, `ReviewQueuePage`, `ReportsPage`, `NotFoundPage` |
| Key components | `AppShell` (header, workflow stepper, tool links), `CaseFile` (one building), `viewer/SceneViewer` (swipe/zoom/overlays), `States` (loading, error and "not available yet" states), `Chip` (provenance labels) |
| Charts | hand-built SVG components in `frontend/src/charts/`: `BarList`, `BoxPlot`, `Dumbbell`, `Heatmap`, `ScatterPlot` |
| API client | `frontend/src/services/api.ts`; types in `frontend/src/types/api.ts` |

The frontend never computes a score itself. Every number on screen comes from an API response.

## Backend

| Item | Detail |
|---|---|
| Stack | Python 3.11, FastAPI 0.153 + Uvicorn, Pydantic 2 schemas (`backend/app/schemas/api.py`), scikit-learn 1.9.1, scikit-image (Canny), ReportLab (PDF) |
| Entry point | `backend/app/main.py` (app + error handlers) |
| Routes | `backend/app/api/routes.py` (thin; the logic lives in modules) |
| Errors | `backend/app/core/errors.py`: structured `{"error": {"code", "message", "fix"?}}`, never a traceback or server path |

### The API routes (all under `/api`)

| Method | Route | Used by |
|---|---|---|
| GET | `/health` (in `backend/app/api/health.py`) | status dot |
| GET | `/dataset/status` | Readiness tiles |
| GET | `/scenes`, `/scenes/{scene_id}` | Library, Triage |
| GET | `/scenes/{scene_id}/image/{pre\|post}`, `/buildings/{uid}/crop/{pre\|post}` | images and crops |
| GET | `/features/summary`, `/features/scatter` | Feature Lab |
| GET | `/models`, `/models/{name}/metrics`, `/models/{name}/tree` | Readiness, Triage, Arena |
| GET | `/results/overview`, `/results/comparison`, `/results/confounding`, `/results/experiments` | Arena |
| GET | `/clusters` | Cluster Atlas |
| POST | `/analysis` | Triage "Run analysis" |
| GET | `/analyses`, `/analysis/{id}` | Reports, Case File page |
| GET | `/analysis/{id}/buildings/{uid}/explain` | Case File explanation |
| GET | `/analysis/{id}/export.csv`, `/analysis/{id}/report.pdf` | Reports |
| GET | `/review-queue` | Review Queue |
| POST/GET | `/training/run`, `/jobs/{job_id}` | background training API (no UI uses it; needs `confirm: true`) |

Swagger documentation is at `http://localhost:8020/docs` while the backend runs.

## Machine-learning components

| Component | File | Role |
|---|---|---|
| Building extraction | `backend/app/datasets/mask_buildings.py` → `buildings_from_masks` | masks → buildings, labels, polygons |
| Feature extraction | `backend/app/features/extract.py` → `building_features`, `scene_rows` | 37 numbers per building |
| Splits | `backend/app/training/splits.py` → `build_manifest`, `validate_manifest` | scene-grouped, leakage-checked |
| Pipelines | `backend/app/training/models.py` → `make_pipeline` | `SkewLog1p` → `SimpleImputer` → `StandardScaler` → model |
| Training | `backend/app/training/run.py` → `train_one`, `run_design` | grid search on validation, single test evaluation |
| Save/load | `backend/app/training/persistence.py` → `save_model`, `load_model` | joblib + metadata JSON, compatibility checks |
| Metrics | `backend/app/evaluation/metrics.py` → `classification_metrics`, `TestSetGuard` | metrics; one test evaluation per model |
| Explanations | `backend/app/explain/explain.py` → `knn_neighbours`, `decision_path` (the Case File); `tree_structure` (the Arena's tree view) | per-building "why" |
| Inference | `backend/app/inference/analysis.py` → `run_analysis`, `explain_building` | predict a scene, cache, explain |

## Data flow when you click "Run analysis"

1. The browser sends `POST /api/analysis {scene_id, model_name}`.
2. `run_analysis` checks both IDs, loads the saved pipeline (`Store.model`), and looks for a cached result
   in `artifacts/analyses/runs/<analysis_id>.json`.
3. If there is none, it takes the scene's rows from the feature table and calls `predict_proba`.
   - The **predicted class** is the highest score.
   - The **confidence** is that highest score.
   - `needs_review` is set when the confidence is below the model's validation-chosen threshold.
4. It saves the result as JSON (the cache) and returns the buildings, polygons, predictions and split
   provenance.
5. The browser draws the footprints. Clicking one calls `/explain`, which returns the KNN neighbours or the
   DT path.

Typical time: a few milliseconds for a cached scene. Training is never triggered.

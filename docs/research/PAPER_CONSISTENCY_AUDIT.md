# Paper ↔ project consistency audit

- **Paper:** `docs/research/AfterSight_Final_Research_Paper.docx` / `.pdf` (5 pages incl. references).
- **Project:** the simplified clone, branch `simplified`.

**Status key:**

- **Verified (artifact):** read from a saved file by the builder at build time.
- **Verified (re-run):** recomputed during this audit by inference or resampling only; no retraining.
- **Verified (code/config):** matches the implementation.
- **Documented:** taken from project docs; not re-measured.
- **Unverified:** none remain in the paper.

## Data

| Claim in the paper | Evidence | Status |
|---|---|---|
| A third-party Parquet copy of xBD's **training** split; 10 events, 2,799 pairs; 1024 × 1024 RGB; no official test or hold-out split | `artifacts/inspection/xbd/inspection_report.json` (`all_events`, `labelled_xbd_splits`, `image_sizes`, `image_modes`); `data/README.md` | Verified (artifact); provenance Documented |
| Five events; selection was a design decision (Santa Rosa has 55 major-damage buildings) | `config/events.yaml`; `class_counts.by_event` | Verified (config/artifact) |
| Table I per-event and per-class counts; total 1,239 pairs, 71,340 buildings | `inspection_report.json` → `class_counts`, `all_events[e].scenes`, `totals` | Verified (artifact) |
| 1,239 = 52 pairs with no building component + 8 with only excluded components (28 blobs, all without a damage value) + **1,179** contributing pairs; the 1,179 are exactly the scenes in both manifests | `artifacts/inspection/xbd/scenes.csv` (`n_raw_footprints`, `n_buildings`); `data/processed/scenes/<id>/scene.json` drop reasons (`label_missing` ×28); `data/splits/*_v1.json` → `scenes`. The builder **asserts** the set equality | Verified (re-run) |
| Buildings = 4-connected components of the footprint mask; label = majority damage value; dropped: < 16 px (54), no damage value (1,396), tie (1); 782 mixed components | `backend/app/datasets/mask_buildings.py`; `config/events.yaml`; `inspection_report.json` → `drop_reasons`, `totals.mixed_label_blobs` | Verified (code + artifact) |
| 58% of buildings undamaged | 41,449 / 71,340 | Verified (artifact) |
| 168 pairs flagged by the alignment check, kept | `inspection_report.json` → `alignment.scenes_flagged` | Verified (artifact) |

## Features and preprocessing

| Claim | Evidence | Status |
|---|---|---|
| 37 features in five groups (12/8/12/3/2), the same order for both models, no feature selection | `backend/app/features/schema.py` → `FEATURE_GROUPS`, `FEATURE_COLUMNS`; model metadata `feature_columns` | Verified (code + artifact) |
| Grey = 0.299R + 0.587G + 0.114B; Eqs. (1)–(4); Canny σ = 1.5; GLCM 32 levels, horizontal and vertical pairs inside the footprint; 8 px ring excluding buildings; compactness 4πA/P² | `backend/app/features/extract.py`; `config/features.yaml` | Verified (code/config) |
| log1p on 7 features with training skewness > 2; median imputation; standardisation; all fitted on train | `models.py` → `make_pipeline`, `SkewLog1p`; metadata `preprocessing.log1p_features` | Verified (code + artifact) |
| No missing values, so imputation has no effect | 0 NaN cells in `features_v1.parquet` | Verified (re-run) |

## Classifiers and protocol

| Claim | Evidence | Status |
|---|---|---|
| KNN: Euclidean (Minkowski p = 2), uniform votes, k = 15 in both designs | saved estimators `artifacts/models/*/four_class/knn.joblib`; metadata | Verified (artifact) |
| DT: Gini, balanced class weights, max depth 8, min leaf 100 in both designs | `decision_tree.joblib` attributes; metadata | Verified (artifact) |
| Grids: KNN 6 settings, DT 8 settings; selection by validation macro-F1; selected model fitted on train only; test evaluated once (guard) | `config/training.yaml`; `run.py` (`train_one`); metadata `search_on_validation`, `test_evaluations = 1` | Verified (code + artifact) |
| Protocol differences: DT class-weighted, KNN unweighted (no option in scikit-learn); grid sizes; scaling irrelevant to DT splits | `make_pipeline`; algorithm property | Verified (code) |
| Within-event split 821/181/177 scenes (49,929/10,705/10,706 buildings); cross-event 716/130/333 (42,425/7,486/21,429); Michael test = 333 of 343 pairs | `data/splits/*_v1.json` → `counts`; `scenes.csv` | Verified (artifact) |
| Michael has 23.3% minor damage vs 13.6% in the other events | `class_counts.by_event` | Verified (artifact) |

## Results

| Claim | Evidence | Status |
|---|---|---|
| Test macro-F1 KNN 0.520 / DT 0.470 (within), 0.271 / 0.294 (cross); accuracy 0.636 / 0.484 and 0.554 / 0.502 | `artifacts/models/*/four_class/{knn,decision_tree}.json` → `metrics.test` | **Verified (re-run):** recomputed from `*_test_predictions.parquet`, and re-inference with the saved models reproduces every prediction |
| 95% scene-bootstrap CIs and paired differences (+0.050 [+0.013, +0.084]; −0.023 [−0.034, −0.011]); 2,000 replicates, seed 42 | `scripts/paper_stats.py` → `paper/build/stats.json` | Verified (re-run, byte-identical) |
| Validation macro-F1 0.581 / 0.576 (cross) | metadata `metrics.val` | Verified (artifact) |
| Table V per-class P/R/F1; Fig. 3 confusion matrices | metadata `per_class`, `confusion_matrix` | Verified (artifact) |
| Context row: LR 0.527/0.290, RF 0.546/0.291, threshold rule 0.383/0.193 | `artifacts/metrics/*/four_class/comparison.json` | Verified (artifact) |
| Event signature: event-prediction accuracy 0.829 vs 0.252 | `artifacts/analyses/confounding/within_event.json` | Verified (artifact) |
| Michael within-event per-event macro-F1 0.367 (KNN) / 0.329 (DT) | metadata `test_per_event` | Verified (artifact) |
| Drops of 48% (KNN) and 37% (DT) | computed from the four test macro-F1 values | Verified (re-run) |
| Inference 19.7 ms vs 0.27 ms per 1,000 buildings; 49,929 training buildings | metadata `inference_seconds_per_1000_test`; manifest | Verified (artifact); single, hardware-specific measurement (stated) |
| Python 3.11, scikit-learn 1.9.1, Apple M4 | metadata `sklearn_version`; venv; machine | Verified |

## Application statements in the paper

| Claim | Evidence | Status |
|---|---|---|
| A KNN prediction is explained by its nearest labelled training buildings, a DT prediction by its root-to-leaf path in raw feature units | `backend/app/explain/explain.py` (`knn_neighbours`, `decision_path`); Case File (`frontend/src/components/CaseFile.tsx`); screenshots `docs/screenshots/04_casefile_knn.png`, `05_casefile_dt.png` | Verified (code + clone UI) |
| No claim about removed UI elements (re-run button, demo chip, internal labels) | text search of the final PDF: 0 matches for "Phase 3b", "re-run", "demo mode", "readiness" | Verified |

## References

All 17 references were verified during drafting: 15 through Crossref DOI records, JMLR through its publisher
page, and the xBD workshop paper through CVF/mlanthology. Valentijn *et al.* was read in full. Details are in
`paper/references.md`. Every reference is cited, numbered in order of first citation.

## Not claimed

- Operational readiness.
- Compliance with any venue's template: the layout is provisional and labelled on every page.
- Statistical significance: the paper says only that a bootstrap interval excludes zero, and describes the
  intervals' scope.
- Generalisation to unseen disasters in general: there is one held-out event.

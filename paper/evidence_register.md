# Evidence register: AfterSight KNN vs Decision Tree paper

Repository state: branch `paper` from `main` at tag `v1.3` (commit `e32e8b0`). Audited on 2026-10-25 on the
machine that holds the real data and artifacts (Apple M4, 10 cores, 16 GB RAM, macOS, Python 3.11.17 venv,
scikit-learn 1.9.1).

**Status key**

- **Verified (artifact):** read directly from a saved artifact or the code.
- **Verified (re-run):** recomputed during this audit; inference and metrics only, nothing retrained.
- **Documented:** stated in the project docs only.
- **Inferred:** a reasonable interpretation that the evidence does not establish.
- **Unavailable:** the evidence is missing.

Numbers in this register come from the artifacts, never from the docs. Two doc errors found are listed at
the end.

## 1. Data source and subset

| # | Claim | Evidence | Status |
|---|---|---|---|
| 1.1 | Source is a third-party Parquet repackaging of the **xBD train split** (17 shards `train-000NN-of-00017.parquet`), obtained from a shared Google Drive, not from xview2.org | `data/README.md` L7, L16; `inspection_report.json` → `parquet_shards` (17) | Verified (artifact) for the shard set; Documented for the provenance |
| 1.2 | Only xBD's **train** split is present; no official test/holdout partition is used | `inspection_report.json` → `labelled_xbd_splits = {train: true}`; `data/README.md` L74 | Verified (artifact) |
| 1.3 | The mirror contains 10 events and 2,799 scene pairs | `inspection_report.json` → `all_events` (10 events; scenes sum 2,799) | Verified (artifact) |
| 1.4 | Five events are used: hurricane-harvey, hurricane-matthew, hurricane-michael, palu-tsunami, santa-rosa-wildfire | `config/events.yaml`; manifests' `feature_table_fingerprint.events` | Verified (artifact) |
| 1.5 | Why these five: "enough major-damage and destroyed examples (mexico-earthquake and midwest-flooding had almost none)" | comment in `config/events.yaml` L8–9 | Documented. A design decision, not a formal rule: santa-rosa-wildfire was included with only 55 major-damage buildings, while midwest-flooding (108 major, 72 destroyed) and socal-fire (44 major, 1,169 destroyed) were excluded. Reported in the paper as a threat to validity |
| 1.6 | Disaster types (wind, flooding, fire, tsunami) come from config, not from the dataset | `config/events.yaml` → `disaster_types`; `data/README.md` L72 | Verified (artifact) |
| 1.7 | 1,239 scene pairs in the five events, all 1024×1024 RGB, 0 unreadable, 0 duplicate scene IDs | `inspection_report.json` → `totals`, `image_sizes`, `image_modes`, `duplicate_scene_ids` | Verified (artifact) |
| 1.8 | 1,179 scenes contain at least one labelled building (52 have no building blob at all; the remaining 8 have only dropped blobs) | `features_v1.parquet` (1,179 unique `scene_id`); `totals.scenes_without_buildings` = 52 | Verified (re-run) |
| 1.9 | 71,340 labelled buildings: no-damage 41,449 · minor 11,766 · major 10,270 · destroyed 7,855 | `inspection_report.json` → `class_counts.overall`; feature table 71,340 rows | Verified (artifact + re-run) |
| 1.10 | Per-event class counts (e.g. palu-tsunami has 1 minor; santa-rosa 76 minor, 55 major) | `inspection_report.json` → `class_counts.by_event` | Verified (artifact) |
| 1.11 | 168 scenes are flagged by a pre/post alignment check (phase-correlation shift > 8 px on downsampled grey images) and kept. The report notes real damage also moves the estimate, so a flag means "check", not "misregistered" | `inspection_report.json` → `alignment` (`scenes_flagged`: 168, `note`) | Verified (artifact); effect on results not measured |

## 2. Labels and building regions (annotation-derived, not detected)

| # | Claim | Evidence | Status |
|---|---|---|---|
| 2.1 | Damage mask values map 1 → no-damage, 2 → minor, 3 → major, 4 → destroyed; 0 = background; a full scan found no other values | `config/events.yaml` → `masks.damage_values`; `inspection_report.json` → `mask_checks` (`unexpected_damage_values_px = {}`) | Verified (artifact) |
| 2.2 | Buildings are **connected components (4-connectivity) of the pre-event footprint mask `t1_mask`**; label = **majority `t2_mask` value** inside the blob | `backend/app/datasets/mask_buildings.py`; `footprint_source` in the report; `connectivity: 4` in config | Verified (code + artifact) |
| 2.3 | Blobs are dropped (and logged) when: area < 16 px (54), no damage value (1,396), or a tied vote (1); total dropped 1,451 of 72,791 | `inspection_report.json` → `drop_reasons`, `totals` | Verified (artifact) |
| 2.4 | 782 blobs contain more than one damage value (possible merged buildings); 781 kept with the majority label | `inspection_report.json` → `totals.mixed_label_blobs`, `mixed_label_blobs.csv` (782 rows) | Verified (artifact) |
| 2.5 | No original xBD polygons, building UIDs or geographic coordinates; polygons are in image pixels | `data/README.md` L28, L62; mirror schema | Verified (schema) / Documented |
| 2.6 | 2,674 polygons needed geometric repair when traced (validity fix); labels are unaffected | `totals.buildings_repaired`; `mask_buildings.py` L92–96 | Verified (artifact) |
| 2.7 | The system does **not** detect footprints from raw imagery | code path above | Verified (code) |

## 3. Features (37, identical for both classifiers)

| # | Claim | Evidence | Status |
|---|---|---|---|
| 3.1 | 37 features in five groups: colour 12, change 8, texture 12, context 3, shape 2 (names below) | `backend/app/features/schema.py` → `FEATURE_GROUPS` | Verified (code) |
| 3.2 | Texture uses Canny edge density and GLCM contrast and homogeneity on grey-level crops; context uses a surrounding ring | `backend/app/features/extract.py`, `config/features.yaml`, `docs/FEATURES.md` | Verified (code); exact formulas to be transcribed from `extract.py` for the paper |
| 3.3 | **No missing values** in the feature table (0 NaN cells), so median imputation is a no-op on this data | `features_v1.parquet` scan | Verified (re-run) |
| 3.4 | No feature selection; both models receive all 37 columns in the same order | `run.py` / `FEATURE_COLUMNS`; model metadata `feature_columns` | Verified (code + artifact) |

Feature names:

- **colour:** `{pre,post}_{mean,std}_{r,g,b}`
- **change:** `abs_dmean_{r,g,b}`, `dstd_{r,g,b}`, `gray_absdiff_mean`, `brightness_ratio`
- **texture:** `gray_std_{pre,post,delta}`, `edge_density_{pre,post,delta}`, `glcm_contrast_{pre,post,delta}`, `glcm_homogeneity_{pre,post,delta}`
- **context:** `ring_dmean`, `ring_absdiff_mean`, `contrast_change`
- **shape:** `area_px`, `compactness`

## 4. Splits and leakage control

| # | Claim | Evidence | Status |
|---|---|---|---|
| 4.1 | **Within-event:** scene-grouped, class-stratified split via `StratifiedGroupKFold` (seed 42), requested 70/15/15. Realised: train 821 scenes / 49,929 buildings, val 181 / 10,705, test 177 / 10,706; all five events in every split | `data/splits/within_event_v1.json` (hash `7bba8a9fbbd2740a`); `backend/app/training/splits.py` | Verified (artifact) |
| 4.2 | **Cross-event:** all hurricane-michael scenes with labelled buildings are the test set (333 scenes / 21,429 buildings); train 716 / 42,425 and val 130 / 7,486 from the other four events (requested 85/15) | `data/splits/cross_event_v1.json` (hash `dd7d8566b7dbaf7a`) | Verified (artifact) |
| 4.3 | No scene appears in two splits; the manifests are validated and hashed | `splits.py` → `validate_manifest`; tests `test_no_scene_overlap_in_either_design`, `test_real_manifests_have_zero_scene_overlap`, `test_validator_catches_leakage` (148 backend tests pass) | Verified (code + tests) |
| 4.4 | The cross-event test class mix differs from training: minor 23.3% of Michael vs 13.6% of the other events | split counts | Verified (artifact) |
| 4.5 | Within-event: scenes of the same disaster appear in train and test, so scores are optimistic for new disasters | manifest description; design | Verified (design) |

## 5. Classifiers and protocol (the fairness points)

| # | Claim | Evidence | Status |
|---|---|---|---|
| 5.1 | Both use the **same pipeline:** `SkewLog1p` (log1p on features with training skew > 2; 7 features) → median `SimpleImputer` → `StandardScaler` → model, all fitted on the training split only | `models.py` → `make_pipeline`; saved pipeline steps `['log','impute','scale','model']`; metadata `preprocessing` | Verified (code + artifact) |
| 5.2 | **KNN:** `KNeighborsClassifier(n_jobs=-1)`, Euclidean (Minkowski p = 2), algorithm auto; selected k = 15, uniform weights (both designs) | saved estimator attributes; metadata `hyperparameters` | Verified (artifact) |
| 5.3 | **Decision Tree:** `DecisionTreeClassifier(class_weight="balanced", random_state=42)`, Gini, splitter best, `min_samples_split` = 2, no cost-complexity pruning (`ccp_alpha` = 0); selected max_depth 8, min_samples_leaf 100 (both designs); 115 leaves (within) / 111 (cross) | saved estimator attributes | Verified (artifact) |
| 5.4 | **Protocol difference 1 (class imbalance):** DT uses balanced class weights; KNN has no `class_weight` in scikit-learn and is unweighted. The models treat imbalance differently, which is likely relevant to their different recall patterns | `make_pipeline` | Verified (code); the effect on the results is **Inferred** |
| 5.5 | **Protocol difference 2 (search size):** KNN grid 6 configs (k ∈ {5, 15, 31} × weights ∈ {uniform, distance}); DT grid 8 configs (max_depth ∈ {4, 6, 8, 12} × min_samples_leaf ∈ {20, 100}) | `config/training.yaml`; `search_on_validation` logs | Verified (artifact) |
| 5.6 | Same selection rule: highest **validation macro-F1** on the same validation split; each final model evaluated on test **exactly once** (`TestSetGuard`, `test_evaluations` = 1) | `run.py` L135–150; metadata | Verified (code + artifact) |
| 5.7 | Scaling is irrelevant to the DT's split choice (per-feature monotone transforms), but it is in its pipeline; it matters for KNN distances | algorithm property; pipeline | Verified (property + code) |
| 5.8 | Same features, labels, splits and metric code for both models | metadata `feature_columns`, `manifest_hash`; `evaluation/metrics.py` | Verified (artifact) |
| 5.9 | Single training run per model per design (deterministic seed 42); **no repeated runs** | `run.py`; one artifact per model | Verified (artifact) |

## 6. Results (test, 4-class)

| # | Claim | Evidence | Status |
|---|---|---|---|
| 6.1 | Within-event test macro-F1: **KNN 0.5202, DT 0.4700**; accuracy KNN 0.6363, DT 0.4839 | `artifacts/models/within_event/four_class/{knn,decision_tree}.json`; `comparison.json` | **Verified (re-run):** recomputed from the saved predictions, and re-inference reproduces every prediction |
| 6.2 | Cross-event test macro-F1: **KNN 0.2707, DT 0.2938**; accuracy KNN 0.5544, DT 0.5019 | `artifacts/models/cross_event/four_class/*.json` | **Verified (re-run)**, as above |
| 6.3 | Validation macro-F1 (selection): within KNN 0.5184 / DT 0.5110; cross KNN 0.5811 / DT 0.5757 | metadata `metrics.val` | Verified (artifact) |
| 6.4 | Per-class test results and confusion matrices for both models and designs (e.g. within: KNN recall no-damage 0.831 / minor 0.270 / major 0.254 / destroyed 0.658; DT 0.434 / 0.449 / 0.499 / 0.780) | metadata `metrics.test.per_class`, `confusion_matrix` | Verified (artifact) |
| 6.5 | Accuracy and macro-F1 disagree: KNN has the higher accuracy in both designs, while DT has the higher cross-event macro-F1 | 6.1–6.2 | Verified (re-run) |
| 6.6 | Major damage is the hardest class cross-event (F1: KNN 0.037, DT 0.026) | 6.4 | Verified (artifact) |
| 6.7 | Within-event per-event test macro-F1 (KNN / DT): harvey 0.338 / 0.259, matthew 0.338 / 0.324, michael 0.367 / 0.329, palu 0.470 / 0.420 (3 classes; no minor in test), santa-rosa 0.494 / 0.410 | metadata `test_per_event` | Verified (artifact) |
| 6.8 | Timing on the M4 laptop: training KNN 0.19 s / DT 1.18 s (within); test inference KNN 19.7 ms vs DT 0.27 ms per 1,000 buildings (KNN parallel, `n_jobs=-1`) | metadata | Verified (artifact); single measurement, hardware-specific |
| 6.9 | Confidence intervals / significance | not yet computed. Saved per-building test predictions with `scene_id` exist for both models and designs, so a **scene-level bootstrap is feasible without retraining** (`scripts/paper_stats.py`, after approval) | Unavailable now; feasible |
| 6.10 | Fresh-clone reproduction of every test score | `docs/VIVA_NOTES.md` / earlier phase report | Documented (not re-run in this audit; this audit reproduced by inference only) |

## 7. Secondary analyses (context only)

| # | Claim | Evidence | Status |
|---|---|---|---|
| 7.1 | A logistic-regression classifier predicts the **event** from the same 37 features with test accuracy 0.829 (majority baseline 0.252), so the features carry a strong event signature | `artifacts/analyses/confounding/within_event.json` | Verified (artifact) |
| 7.2 | Phase 3b confounding-reduced feature sets (LR and DT only, no KNN) did not improve cross-event macro-F1 | `artifacts/experiments/phase3b/summary.json` | Verified (artifact); DT rows only usable for the paper |
| 7.3 | Other models (context row only): LR 0.527 / 0.290, RF 0.546 / 0.291, threshold baseline 0.383 / 0.193 (within / cross) | `comparison.json` | Verified (artifact) |

## 8. System (brief, for one paragraph and one figure at most)

| # | Claim | Evidence | Status |
|---|---|---|---|
| 8.1 | FastAPI backend serving saved scikit-learn pipelines from files; React + TypeScript (Vite) frontend; per-building explanations (KNN neighbours, DT decision path) | `backend/app/api/routes.py`, `backend/app/explain/explain.py`, `frontend/` | Verified (code) |
| 8.2 | The app never trains during analysis; test-set guard; tests: 148 backend, 37 frontend, 22-check end-to-end (all passing at v1.3) | code; v1.3 verification | Verified (re-run at v1.3) |

## 9. Missing or unresolved

| Item | Status |
|---|---|
| Authors, affiliations, emails, order | **Unavailable:** the prompt's blank was not filled in |
| Target venue and page limit | **Unavailable:** not filled in. A provisional IEEE conference format will be used unless told otherwise |
| Official xBD test/holdout splits | Unavailable (not in the mirror) |
| Repeated runs / variance across seeds | Unavailable (single deterministic run; retraining is out of scope) |
| Formal rationale for the event selection | Only the config comment (1.5) |
| Pixel resolution of the imagery | Not in the mirror; the xBD paper states targeting below 0.8 m GSD (to be cited, not measured) |

## 10. Doc errors found during the audit (not used in the paper)

- `data/README.md` L61 says "The largest blob in the configured events is 185,291 px". The feature table
  shows the largest is **648,034 px** (`hurricane-harvey_00000448_b0003`); 185,291 px is palu-tsunami's
  maximum.
- The same line says "474 blobs … contain more than one damage value". The inspection report and
  `mixed_label_blobs.csv` give **782** (781 kept). Both numbers were probably written before
  hurricane-michael was added.

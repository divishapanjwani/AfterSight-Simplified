# Reproducing the paper's experiments

Every number in `AfterSight_Final_Research_Paper.docx` is read from saved files by
`paper/build/build_paper.py`. You can do two levels of reproduction:

- **A. Rebuild the paper from the saved results.** Minutes. No training. This is what was done for the final
  paper.
- **B. Regenerate the results from the raw data.** This retrains every model, about 10 minutes on the
  original laptop. It was **not** re-run for the final paper, because retraining would replace the saved
  results.

## Requirements

| Item | Detail |
|---|---|
| OS | macOS (zsh) was used. Windows works for Python and Node steps (see `docs/study-guide/09_SETUP_AND_TROUBLESHOOTING.md`); `run_pipeline.sh` and `start.sh` are bash scripts |
| Python | 3.11, in a venv at the repo root: `python3.11 -m venv .venv`, then `.venv/bin/python -m pip install -r backend/requirements.txt` (scikit-learn 1.9.1 pinned) |
| Paper build | python-docx in its own venv: `python3.11 -m venv paper/build/.venv`, then `paper/build/.venv/bin/python -m pip install -r paper/build/requirements.txt` (python-docx 1.1.2) |
| PDF export | LibreOffice (`/Applications/LibreOffice.app/Contents/MacOS/soffice`) |
| Data for B | The 17 shards `train-000NN-of-00017.parquet` (a third-party Parquet copy of xBD's **training** split, 7.9 GB) in `data/raw/xbd_parquet/`. In this clone that folder is a read-only symlink to the original's copy |
| Hardware used | Apple M4 (10 cores), 16 GB RAM. Peak memory during inspection ≈ 1.2 GB (`artifacts/inspection/xbd/inspection_report.json` → `resources.peak_rss_bytes_measured`) |

## Configuration that defines the experiments

| File | What it fixes |
|---|---|
| `config/events.yaml` | the 5 events; damage mask values 1–4 → classes; `connectivity: 4`; `min_area_px: 16` |
| `config/features.yaml` | `ring_width_px: 8`, `canny_sigma: 1.5`, `glcm_levels: 32`, `glcm_distance_px: 1` |
| `config/training.yaml` | split designs (within-event 70/15/15 stratified-group, cross-event holdout `hurricane-michael` 85/15), `seed: 42`, `skew_threshold: 2.0`, grids (KNN `n_neighbors` {5, 15, 31} × `weights` {uniform, distance}; DT `max_depth` {4, 6, 8, 12} × `min_samples_leaf` {20, 100}), serving set |
| `backend/app/training/models.py` → `make_pipeline` | `KNeighborsClassifier(n_jobs=-1)` (unweighted); `DecisionTreeClassifier(class_weight="balanced", random_state=seed)`; shared `SkewLog1p → SimpleImputer(median) → StandardScaler` |

## A. Rebuild the paper from the saved results (no training)

```zsh
cd ~/AfterSight-Simplified
.venv/bin/python scripts/snapshot_behaviour.py save verification/snapshot_before.json    # once: record a baseline
.venv/bin/python scripts/snapshot_behaviour.py check verification/snapshot_before.json   # later: saved models behave as before
.venv/bin/python scripts/paper_stats.py --resamples 2000 --seed 42    # scene-level bootstrap -> paper/build/stats.json (<1 s)
.venv/bin/python paper/build/make_figures.py                          # Figs. 1-3 -> paper/figures/
paper/build/.venv/bin/python paper/build/build_paper.py               # -> docs/research/AfterSight_Final_Research_Paper.docx
/Applications/LibreOffice.app/Contents/MacOS/soffice --headless --convert-to pdf \
  --outdir docs/research docs/research/AfterSight_Final_Research_Paper.docx
```

**Inputs read:**

- `artifacts/models/{within_event,cross_event}/four_class/{knn,decision_tree}.json` (hyperparameters, search
  logs, val/test metrics, confusion matrices, timings);
- `artifacts/metrics/*/four_class/comparison.json` (context row);
- `artifacts/metrics/*/four_class/{knn,decision_tree}_test_predictions.parquet` (bootstrap);
- `data/splits/*_v1.json`;
- `artifacts/inspection/xbd/inspection_report.json` and `scenes.csv`;
- `artifacts/analyses/confounding/within_event.json`.

**What was checked for the final paper:**

- `paper_stats.py` reproduced `stats.json` byte-for-byte.
- The figures regenerated identically.
- The four KNN/DT test macro-F1 values equal a recomputation from the saved predictions, and re-inference
  with the saved models reproduces every saved prediction.

## B. Regenerate everything from the raw data (retrains; not done for the paper)

```zsh
scripts/run_pipeline.sh                  # 10 steps; add --no-experiments to skip the feature-set experiments
```

| Step | Command | Output |
|---|---|---|
| 1 | `python scripts/inspect_dataset.py --alignment` | `artifacts/inspection/xbd/`, scene cache `data/processed/scenes/` |
| 2 | `python scripts/build_features.py --workers 4` | `data/processed/features/features_v1.parquet` (71,340 × 37) |
| 3 | `python scripts/make_splits.py` | `data/splits/within_event_v1.json`, `cross_event_v1.json` |
| 4 | `python scripts/train.py` | `artifacts/models/…`, `artifacts/metrics/…` (each test set evaluated once), K-Means, confounding check |
| 5 | `python scripts/run_model_comparison.py` | SVM, soft voting, GridSearchCV → `artifacts/experiments/model_comparison/` (needs step 4) |
| 6 | `python scripts/run_model_comparison.py --adaboost-only` | adds the AdaBoost row to the same summary |
| 7 | `python scripts/serve_comparison_models.py` | serves the saved SVM and the soft-voting ensemble (no refit) → `artifacts/models/within_event/four_class/` |
| 8 | `python scripts/calibrate_review.py` | `artifacts/models/within_event/four_class/review_thresholds.json` for all five served models (validation only) |
| 9 | `python scripts/run_phase3b.py` | feature-set experiments (called "Phase 3b" during development) → `artifacts/experiments/phase3b/` |
| 10 | `python scripts/prepare_demo.py` | 3 pre-cached demo scenes and reports |

**Run time:** a fresh end-to-end run of the original 7 steps (before steps 5–7 were added) took **571 s** on the M4 laptop. That figure is documented from the
project's fresh-setup check (`docs/UNDERSTANDING_AFTERSIGHT.md`, Section 5) and was not re-measured here.
The saved per-fit training times for the paper's models are:

- KNN: 0.19 s within-event, 0.16 s cross-event;
- DT: 1.18 s within-event, 1.00 s cross-event;
- grid fits: 0.15–0.19 s for KNN and 0.65–1.53 s for DT.

**Determinism:** the seed (42) fixes the splits and the DT. KNN has no randomness. A rerun with the same
data, configs and library versions is expected to reproduce the saved models. The project's earlier
fresh-clone check reported identical test scores. That is documented, not re-run here.

Before a rerun, record a baseline with `scripts/snapshot_behaviour.py save verification/snapshot_before.json`; after it, compare with `… check verification/snapshot_before.json`. Any
difference means the results changed, and the paper must be rebuilt (step A) and re-audited.

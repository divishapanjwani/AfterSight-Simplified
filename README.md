# AfterSight

**Explainable Building Damage Triage from Pre/Post-Disaster Satellite Imagery using Classical Machine Learning**

Given a disaster scene and its building footprints, AfterSight classifies each building's damage
(no damage / minor / major / destroyed) with classical machine learning, and explains *why* the model
decided that. It is built as a web app: a triage desk, building case files, a model arena and feature
and cluster explorers.

> **Scope:** *Damage classification on building footprints derived from xBD annotations.*
> AfterSight does not detect buildings in imagery, map floods or burn scars, prove what caused a
> change, or give rescue or routing advice.

## Contents

[Problem](#problem) · [What it does](#what-it-does) · [Results](#results-honestly) · [Quick start](#quick-start-macos) ([Windows](#quick-start-windows-powershell)) ·
[Usage](#usage) · [Architecture](#architecture) · [Dataset and licence](#dataset-and-licence) · [API](#api) · [Tests](#tests) ·
[Limitations](#limitations) · [Troubleshooting](#troubleshooting) · [Future scope](#future-scope) · [Changelog](#changelog) ·
[Related work](#related-work)

## Problem

After a hurricane, flood, fire or tsunami, responders need to know quickly which buildings are damaged and how
badly. Ground surveys are slow and dangerous. Pre- and post-disaster satellite images cover a whole city, but
nobody can inspect tens of thousands of roofs by hand. AfterSight asks how far **classical, explainable machine
learning** on hand-crafted image features can go on this task, and where it stops working.

## What it does

1. **Data:** reads a Parquet mirror of the xBD train split. It reconstructs building footprints from the
   annotation masks and caches the five selected events (1,239 scenes, 71,340 labelled buildings).
2. **Features:** computes 37 explainable numbers per building (colour, change, texture, surroundings, shape).
3. **Honest splits:** grouped by scene, in two designs:
   - **within-event:** new scenes of the same disasters;
   - **cross-event:** hurricane-michael held out completely, a disaster the models never saw.
4. **Models:** the app serves **KNN** (default), the **Decision Tree**, the **Random Forest**, the **SVM** and the
   **soft-voting ensemble** (SVM + RF + KNN) (`serving.models` in `config/training.yaml`). The SVM and the ensemble
   come from `scripts/run_model_comparison.py` and are served by `scripts/serve_comparison_models.py` without
   retraining. The Model Arena also compares **GridSearchCV-optimised** KNN / DT / SVM and an AdaBoost (boosting)
   ensemble (Arena only). K-Means is analysis only. Every model is tuned on validation
   or grouped cross-validation, and each final model is evaluated on its test set **once**. Logistic Regression is
   not a damage model of the app; a multinomial logistic regression is used only as an *analysis classifier*
   in the confounding check (see below).
5. **Explanations:** KNN nearest neighbours, the Decision Tree's root-to-leaf decision path, and permutation
   importance. The Random Forest, the SVM and the ensemble have no per-building explanation (the Case File says so and
   shows their scores).
6. **Web app:** FastAPI + React.
   - Triage Desk (before/after swipe, overlays) and Building Case File.
   - Model Arena, Feature Lab, Cluster Atlas, Review Queue and Reports (CSV/PDF).
   - Every number on screen comes from a saved artifact, and every scene says whether it was used for
     training.

## Results, honestly

Test results (4-class), each model evaluated once. Macro-F1 is the primary metric. KNN, DT and RF come from their
saved models; the rest from `artifacts/experiments/model_comparison/summary.json`. Full tables:
[docs/RESULTS.md](docs/RESULTS.md).

| Model | Type | within-event acc / macro-P / macro-R / **macro-F1** | cross-event acc / macro-P / macro-R / **macro-F1** |
|---|---|---|---|
| **KNN** (app default) | default | 0.636 / 0.560 / 0.503 / **0.520** | 0.554 / 0.275 / 0.317 / **0.271** |
| **Decision Tree** | default | 0.484 / 0.467 / 0.541 / **0.470** | 0.502 / 0.284 / 0.339 / **0.294** |
| **Random Forest** | default (bagging) | 0.590 / 0.528 / 0.586 / **0.546** | 0.606 / 0.299 / 0.349 / **0.291** |
| SVM (RBF, C=1; 10k-building sample) | default, served | 0.536 / 0.505 / 0.581 / **0.514** | 0.380 / 0.287 / 0.340 / **0.275** |
| Soft voting (SVM + RF + KNN) | parallel ensemble, served | 0.635 / 0.563 / 0.541 / **0.542** | 0.597 / 0.283 / 0.341 / **0.282** |
| AdaBoost (200 stumps) | serial ensemble, Arena only | 0.624 / 0.523 / 0.404 / **0.396** | 0.619 / 0.285 / 0.314 / **0.273** |
| KNN / DT / SVM optimised (GridSearchCV) | optimised, Arena only | macro-F1 **0.520 / 0.474 / 0.514** | not evaluated |

The soft-voting ensemble beats KNN and the SVM but **not** the Random Forest, its best member, in either design.
GridSearchCV (grouped by scene, training split only) chose the same KNN and SVM settings as the defaults and a
deeper Decision Tree (depth 12: test macro-F1 0.470 → 0.474).

- **Two numbers to remember:** about **0.52** (KNN) on new scenes of known disasters, which is an optimistic
  upper bound, and about **0.27–0.29** on an unseen disaster, which is the realistic figure for a new event.
  The Decision Tree scores lower within-event but holds up slightly better on the unseen disaster.
- **Why the gap:** the same 37 features identify *which disaster* a building is from with 82.9% accuracy
  (measured with an analysis classifier, a multinomial logistic regression, used only for this check; it is
  not a damage model in the app).
  Removing that signature (the feature-set experiments; saved results, not shown in the app) did not improve transfer, so the gap is a genuine
  domain shift between disasters.
- **Easiest class:** *destroyed* (recall around 0.8 within-event).
- **Hardest:** separating minor from major damage.

## Quick start (macOS)

> **This is the simplified clone** (`~/AfterSight-Simplified`, branch `simplified`). It uses the same ports as
> the original (`8020` API, `5200` web) and `start.sh` stops whatever is on them, so **run only one copy at a
> time**. (The clone's verification notes are kept locally and are not part of this repository.)

Requires Python 3.11 (`brew install python@3.11`) and Node 20+. The virtualenv lives at the **repo root**.

**The trained models are not in git** (`data/`, `artifacts/` and `outputs/` are git-ignored), so a fresh clone
has to build them. The order matters: the model comparison needs the trained KNN, Decision Tree and Random
Forest; `serve_comparison_models.py` needs the comparison's saved SVM; the review thresholds and the demo cache
need all five served models (KNN, DT, RF, SVM, soft voting). `scripts/run_pipeline.sh` runs these steps in order.

```zsh
# 1. once, from the repo root: Python + JS dependencies
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r backend/requirements.txt
(cd frontend && npm ci)
#   Always run the tests inside this venv (python -m pytest). A global Python can see other packages (e.g. torch
#   in user site-packages), which makes test_no_deep_learning_frameworks_installed fail; the venv does not.

# 2. put the xBD Parquet shards in data/raw/xbd_parquet/  (see data/README.md)

# 3. build everything (~11 min): scripts/run_pipeline.sh runs, in this order,
#      inspect → features → splits → train KNN/DT/RF → model comparison (SVM, soft voting, GridSearchCV)
#      → AdaBoost row → serve the SVM + soft voting → review thresholds → Phase 3b experiments → demo cache
scripts/run_pipeline.sh

# 4. run the app (frees ports 8020/5200 if busy, prints the URL; Ctrl-C stops both)
./start.sh                      # → http://localhost:5200   (API docs: http://localhost:8020/docs)
```

The same steps one by one (what `run_pipeline.sh` runs, venv active, from the repo root):

```zsh
python scripts/inspect_dataset.py --alignment          # 1. inspect the shards, fill the scene cache
python scripts/build_features.py --workers 4           # 2. per-building features
python scripts/make_splits.py                          # 3. scene-grouped split manifests
python scripts/train.py                                # 4. train + evaluate KNN, DT, RF (test sets once)
python scripts/run_model_comparison.py                 # 5. SVM, soft voting, GridSearchCV (needs step 4)
python scripts/run_model_comparison.py --adaboost-only # 6. AdaBoost row for the Arena
python scripts/serve_comparison_models.py              # 7. serve the SVM + soft voting (no refit; needs step 5)
python scripts/calibrate_review.py                     # 8. review thresholds for all five models (validation)
python scripts/run_phase3b.py                          # 9. feature-set experiments (optional)
python scripts/prepare_demo.py                         # 10. pre-cache the demo scenes
```

Without steps 2–3 the app still starts, and every page shows what is missing and the command that builds
it. A clean clone following the original steps was tested in Phase 6: about 11 minutes end to end, with
results identical to the original run. Steps 5–7 were added later, and the full sequence has not been re-run
on a fresh clone since. After installation the app works fully offline.

### Quick start (Windows, PowerShell)

`run_pipeline.sh` and `start.sh` are bash scripts, so on Windows run the steps directly, in the same order.
These commands mirror the macOS ones but have not been tested on Windows.

```powershell
# 1. once, from the repo root: Python + JS dependencies
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install -r backend\requirements.txt
cd frontend; npm ci; cd ..

# 2. put the xBD Parquet shards in data\raw\xbd_parquet\  (see data\README.md)

# 3. build everything, in this order (venv active, repo root)
python scripts\inspect_dataset.py --alignment
python scripts\build_features.py --workers 4
python scripts\make_splits.py
python scripts\train.py                                # KNN, DT, RF
python scripts\run_model_comparison.py                 # SVM, soft voting, GridSearchCV (needs train.py)
python scripts\run_model_comparison.py --adaboost-only # AdaBoost row for the Arena
python scripts\serve_comparison_models.py              # serve the SVM + soft voting (needs the comparison)
python scripts\calibrate_review.py                     # review thresholds for all five models
python scripts\run_phase3b.py                          # feature-set experiments (optional)
python scripts\prepare_demo.py                         # pre-cache the demo scenes

# 4. run the app in two PowerShell windows
.\.venv\Scripts\Activate.ps1; cd backend; python -m uvicorn app.main:app --port 8020   # window 1: API
cd frontend; npx vite --port 5200 --strictPort                                       # window 2 → http://localhost:5200
```

### Running the servers by hand

```zsh
source .venv/bin/activate && cd backend && uvicorn app.main:app --port 8020   # terminal 1
cd frontend && npm run dev                                                    # terminal 2
```

## Usage

| Page | What to do there |
|---|---|
| **Readiness** (`/`) | what is built and trained, headline results, served models (test macro-F1, what confidence means, review threshold) |
| **Library** | browse scenes (held-out **test** scenes by default); filter by event, disaster type and split |
| **Triage Desk** | pick a model (KNN, Decision Tree, Random Forest, SVM, Soft Voting) → **Run analysis** → swipe pre/post, zoom/pan, switch overlay (Ground truth / Prediction / Correct vs wrong), click a footprint |
| **Case file** | crops, predicted vs ground truth, confidence and what it means, the model's explanation |
| **Arena** | within- vs cross-event chart, comparison (KNN, Decision Tree, Random Forest), confusion matrices, per-event scores, tree, and **Model Comparison & Ensemble Analysis** (SVM, soft voting, AdaBoost, GridSearchCV, per-model confusion matrix and per-class metrics) |
| **Feature Lab** | which features separate the classes (training split), box plots and a scatter |
| **Cluster Atlas** | precomputed K-Means for k = 2…10: cluster sizes, silhouette, adjusted Rand index, clusters vs reference labels. Analysis only, never predictions |
| **Review Queue** | buildings *suggested for human review* (low confidence, or KNN, the Decision Tree or the Random Forest disagrees) |
| **Reports** | CSV and PDF per analysis |

Three demo scenes are pre-cached (see [docs/VIVA_NOTES.md](docs/VIVA_NOTES.md)).

## Architecture

```
xBD Parquet shards ─► inspector (masks → building polygons, scene cache) ─► 37 features (Parquet)
      ─► scene-grouped split manifests ─► scikit-learn pipelines (joblib + metadata JSON)
      ─► FastAPI (file-based artifact store, IDs validated, cached analyses) ─► React UI (Vite)
```

| Folder | Contents |
|---|---|
| `backend/app/datasets` | Parquet loader, mask → building polygons, inspector, scene cache, synthetic fixtures |
| `backend/app/features` | feature extraction, schema, table, feature-set experiment variants |
| `backend/app/training` | splits, pipelines (log1p → impute → scale → model), training runs, persistence |
| `backend/app/evaluation`, `explain` | metrics, test-set guard, K-Means, confounding; per-building explanations |
| `backend/app/api`, `inference`, `reports` | routes, store, analysis/review/feature-lab services, CSV/PDF |
| `frontend/src` | pages, components, SVG charts, scene viewer, design tokens ([DESIGN.md](frontend/DESIGN.md)) |
| `scripts/` | `run_pipeline.sh` steps, `e2e_check.py`, `prepare_demo.py`, `snapshot_behaviour.py`, fixtures |
| `config/` | `events.yaml` (events, disaster types), `features.yaml`, `training.yaml` (splits, grids, serving) |
| `docs/` | [FEATURES](docs/FEATURES.md), [ML_EXPLAINED](docs/ML_EXPLAINED.md), [RESULTS](docs/RESULTS.md), [API](docs/API.md), [VIVA_NOTES](docs/VIVA_NOTES.md) |

Data, models, caches and outputs are git-ignored. They are rebuilt by `scripts/run_pipeline.sh`.

## Dataset and licence

- **Data:** the xBD train split (Gupta et al., 2019), used here through a third-party Parquet mirror.
  Details, schema and the footprint derivation are in [data/README.md](data/README.md).
- **xBD licence:** xBD is published by the xView2 organisers under **CC BY-NC-SA 4.0** (non-commercial,
  share-alike, attribution required). It is **not** redistributed in this repository.
- **Imagery in figures:** satellite imagery in screenshots or figures (e.g. in `report/`) is derived from xBD
  and remains under CC BY-NC-SA 4.0.
- **Code licence:** the AfterSight code is MIT-licensed ([LICENSE](LICENSE)); see [NOTICE](NOTICE) for the data licence.

> R. Gupta, R. Hosfelt, S. Sajeev, N. Patel, B. Goodman, J. Doshi, E. Heim, H. Choset and M. Gaston,
> "xBD: A Dataset for Assessing Building Damage from Satellite Imagery," arXiv:1911.09296, 2019.

## API

FastAPI on port 8020 with 25 endpoints. Swagger is at `/docs`, and the full list with the rules (IDs not
paths, structured errors, split provenance, caching) is in [docs/API.md](docs/API.md).

## Tests

```zsh
source .venv/bin/activate
python -m pytest                      # backend: 160 tests (parser, features, leakage, models, ensembles, API incl. traversal)
(cd frontend && npm test)             # frontend: 44 tests (pages, states, triage → case-file flow, Arena comparison)
python scripts/e2e_check.py           # end-to-end: real pipeline on synthetic data, backend + built frontend (25 checks)
```

After a refactor, `python scripts/snapshot_behaviour.py save before.json` (before) and `… check before.json` (after)
prove that predictions, confidences, explanations, Feature Lab numbers and `/models` are unchanged. It uses the saved
models and writes analyses to a temporary folder, never to `artifacts/`.

Synthetic fixtures are generated by the tests and are never shown as xBD data.

## Limitations

- **Footprints are derived from rasterised xBD annotation masks**, not the original JSON polygons. Touching
  buildings merge into one blob, and the label is the majority value inside it.
- **No original identifiers or geography:** original building IDs and lng/lat are unavailable, so there is
  no map and no real-world area.
- **Excluded buildings:** buildings without a damage value (expected to include xBD's `un-classified`) are
  excluded and counted.
- **Optimistic within-event scores:** features carry a strong per-disaster signature. The cross-event
  score (~0.29) is the honest estimate for a new disaster.
- **Uncalibrated confidence:** confidence values are model scores (KNN vote fraction, Decision Tree
  leaf fraction), not calibrated probabilities.
- **Train split only:** only xBD's train split is in the mirror, so all splits are carved from it.
- **Research tool:** not an operational tool. No rescue or routing decisions should be made from it.

## Troubleshooting

| Symptom | Fix |
|---|---|
| A page says **"Not available yet"** | run the command it shows (usually `scripts/run_pipeline.sh`) |
| `start.sh`: port in use | the script frees 8020/5200 itself; if that fails, `lsof -ti tcp:8020 \| xargs kill` |
| `.venv/bin/activate` not found | the venv must be at the repo root: `python3.11 -m venv .venv` |
| `model_not_trained` (409) | `source .venv/bin/activate && python scripts/train.py` |
| Model refuses to load (schema/sklearn version) | install the pinned `backend/requirements.txt`, or retrain |
| Blank page in the browser | check `outputs/logs/web.log`; run `(cd frontend && npm ci)` |
| Readiness: **Review thresholds PARTIAL / NOT READY** | the tile names the models without a usable threshold; `python scripts/calibrate_review.py` (validation only). A model whose calibration found no threshold stays without low-confidence flags |
| `model_not_found` (404) for `logistic_regression` or `baseline` | expected: the app serves KNN, the Decision Tree, the Random Forest, the SVM and soft voting. Logistic Regression is not a damage model and the threshold baseline is no longer used |
| SVM / Soft Voting "not trained" | `python scripts/run_model_comparison.py && python scripts/serve_comparison_models.py`, then `python scripts/calibrate_review.py --models svm soft_voting` |
| Arena: "Not available yet: the model comparison" | `python scripts/run_model_comparison.py` (about a minute; add `--adaboost-only` afterwards for the boosting row) |
| `model_not_served` (404) when opening an old analysis | the analysis used a model the app no longer serves; the file stays in `artifacts/analyses/runs/` |

## Future scope

- Learned features (CNNs or Siamese networks on pre/post crops) compared against this classical baseline
  on the same honest splits.
- Original xBD JSON polygons to avoid merged buildings; more events and xBD's test/holdout splits.
- Domain adaptation or per-event normalisation that keeps scene-wide damage, since the feature-set experiments' scene
median removed it.
- Calibrated probabilities (e.g. isotonic) so confidence values can be read as probabilities.
- Georeferenced output for real map overlays.

## Changelog

- **Enhancements for the faculty review.** Random Forest served again (saved model, saved review threshold); SVM,
  soft-voting ensemble, GridSearchCV-optimised KNN/DT/SVM and AdaBoost in a new Arena section; the threshold
  baseline is no longer in active use. KNN and DT predictions are unchanged. Details: [CHANGELOG.md](CHANGELOG.md).
- **KNN and Decision Tree.** The app serves KNN (default) and the Decision Tree; Logistic Regression is removed as
  a damage model (serving, UI, training, reports, tests and docs). The Case File's decision-path view is back.
  The confounding check and the feature-set experiment keep their saved logistic-regression results as analysis
  classifiers. KNN and DT predictions, confidences and explanations are unchanged from before. Details:
  [CHANGELOG.md](CHANGELOG.md).
- **Two models (superseded).** The app served only Logistic Regression and KNN; the feature-set Experiments section was
  removed from the Model Arena. Decision Tree and Random Forest are still trained and their results kept;
  the Arena shows the Decision Tree as a research comparison. LR and KNN predictions, confidences and
  explanations are unchanged. Details: [CHANGELOG.md](CHANGELOG.md).
- **v1.3: audit and Cluster Atlas.** Audit of the v1.2 fixes: optional served models (Random Forest) no longer
  make the review-threshold tile PARTIAL even when trained without a usable threshold; everything else was
  already correct. Cluster Atlas simplified to the k selector, cluster sizes, silhouette and adjusted Rand index
  (each explained) and the cluster-vs-damage heatmap; the PCA scatter and the inertia/silhouette curves were
  removed from the page (the API still returns them). Windows venv note added. Details:
  [docs/CHANGES_V1_3.md](docs/CHANGES_V1_3.md).
- **v1.2: consistency and reliability fixes.** Readiness tiles for models and review thresholds now say
  READY, PARTIAL or NOT READY with the affected models (an untrained *optional* model never makes them PARTIAL).
  A missing, broken or incomplete review-threshold file no longer crashes the API; a model without a usable
  threshold gets no low-confidence flags and the Review Queue says why. The Model Arena tells a missing feature-set experiment
  summary from a malformed or incomplete one (`results_invalid`). Report wording: thresholds are quoted only
  when they exist, scene-level metrics are labelled as such, missing values read "not available", and cached
  PDFs are versioned (`<id>.r2.pdf`). Predictions, confidences, explanations and Feature Lab numbers are
  unchanged (`scripts/snapshot_behaviour.py`). Details: [docs/FIXES_V1_2.md](docs/FIXES_V1_2.md).
- **v1.1: simplification.** The Case File, the Feature Lab and their backend code were rewritten to be easier
  to read (small named functions, plain loops, comments that explain *why*). Behaviour is unchanged: every
  prediction, confidence, explanation and Feature Lab number was checked to be identical with
  `scripts/snapshot_behaviour.py`. The Readiness page's served-models table no longer shows the validation
  macro-F1 column (the API still returns it). Details: [docs/SIMPLIFICATION_NOTES.md](docs/SIMPLIFICATION_NOTES.md).
- **v1.0:** the complete project (phases 0–7), the report and the beginner guide.

## Related work

This project is separate from my earlier **TerraWatch AI**, which does pixel/scene-level satellite change
detection on GeoTIFFs with a map dashboard. AfterSight works per **building**, does 4-class **damage
classification** with **explainable classical ML**, and has its own triage-desk interface. No code, assets
or design are shared between the two.

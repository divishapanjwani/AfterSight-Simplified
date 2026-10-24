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

[Problem](#problem) · [What it does](#what-it-does) · [Results](#results-honestly) · [Quick start](#quick-start-macos) ·
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
4. **Models:** a non-ML threshold baseline, Logistic Regression, KNN, a Decision Tree and an optional
   Random Forest, plus K-Means as analysis. Every model is tuned on validation, and each final model is
   evaluated on its test set **once**.
5. **Explanations:** LR feature contributions, Decision Tree paths, KNN nearest neighbours, and permutation
   importance.
6. **Web app:** FastAPI + React.
   - Triage Desk (before/after swipe, overlays) and Building Case File.
   - Model Arena, Feature Lab, Cluster Atlas, Review Queue and Reports (CSV/PDF).
   - Every number on screen comes from a saved artifact, and every scene says whether it was used for
     training.

## Results, honestly

Test macro-F1 (4-class, primary metric), each model evaluated once. Full tables are in
[docs/RESULTS.md](docs/RESULTS.md).

| Model | within-event | cross-event (unseen disaster) |
|---|---:|---:|
| Threshold baseline (non-ML) | 0.383 | 0.193 |
| **Logistic Regression** (app default) | **0.527** | **0.290** |
| KNN | 0.520 | 0.271 |
| Decision Tree | 0.470 | 0.294 |
| Random Forest (optional reference) | 0.546 | 0.291 |

- **Two numbers to remember:** about **0.53** on new scenes of known disasters, which is an optimistic upper
  bound, and about **0.29** on an unseen disaster, which is the realistic figure for a new event.
- **Why the gap:** the same 37 features identify *which disaster* a building is from with 82.9% accuracy.
  Removing that signature (the Phase 3b experiments) did not improve transfer, so the gap is a genuine
  domain shift between disasters.
- **Easiest class:** *destroyed* (recall around 0.8 within-event).
- **Hardest:** separating minor from major damage.

## Quick start (macOS)

Requires Python 3.11 (`brew install python@3.11`) and Node 20+. The virtualenv lives at the **repo root**.

```zsh
# 1. once, from the repo root: Python + JS dependencies
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r backend/requirements.txt
(cd frontend && npm ci)

# 2. put the xBD Parquet shards in data/raw/xbd_parquet/  (see data/README.md)

# 3. build everything: inspect → features → splits → train → thresholds → experiments → demo cache (~10 min)
scripts/run_pipeline.sh

# 4. run the app (frees ports 8020/5200 if busy, prints the URL; Ctrl-C stops both)
./start.sh                      # → http://localhost:5200   (API docs: http://localhost:8020/docs)
```

Without steps 2–3 the app still starts, and every page shows what is missing and the command that builds
it. A clean clone following exactly these steps was tested in Phase 6: about 11 minutes end to end, with
results identical to the original run. After installation the app works fully offline.

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
| **Triage Desk** | pick a model → **Run analysis** → swipe pre/post, zoom/pan, switch overlay (Ground truth / Prediction / Correct vs wrong), click a footprint |
| **Case file** | crops, predicted vs ground truth, confidence and what it means, the model's explanation |
| **Arena** | within- vs cross-event chart, comparison, confusion matrices, per-event scores, tree, confounding, 3-class, Experiments |
| **Feature Lab** | which features separate the classes (training split), box plots and a scatter |
| **Cluster Atlas** | K-Means elbow/silhouette, PCA (visualisation only), clusters vs labels. Never predictions |
| **Review Queue** | buildings *suggested for human review* (low confidence or models disagree) |
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
| `backend/app/features` | feature extraction, schema, table, Phase 3b variants |
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
python -m pytest                      # backend: 146 tests (parser, features, leakage, models, API incl. traversal)
(cd frontend && npm test)             # frontend: 33 tests (pages, states, triage → case-file flow)
python scripts/e2e_check.py           # end-to-end: real pipeline on synthetic data, backend + built frontend (22 checks)
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
- **Uncalibrated confidence:** confidence values are model scores (LR probability, KNN vote fraction,
  tree leaf fraction), not calibrated probabilities.
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
| Arena: **Saved Phase 3b results are malformed or incomplete** | the summary file is broken; `python scripts/run_phase3b.py` rebuilds it |

## Future scope

- Learned features (CNNs or Siamese networks on pre/post crops) compared against this classical baseline
  on the same honest splits.
- Original xBD JSON polygons to avoid merged buildings; more events and xBD's test/holdout splits.
- Domain adaptation or per-event normalisation that keeps scene-wide damage, since the Phase 3b scene
  median removed it.
- Calibrated probabilities (e.g. isotonic) so confidence values can be read as probabilities.
- Georeferenced output for real map overlays.

## Changelog

- **v1.2: consistency and reliability fixes.** Readiness tiles for models and review thresholds now say
  READY, PARTIAL or NOT READY with the affected models (an untrained *optional* model never makes them PARTIAL).
  A missing, broken or incomplete review-threshold file no longer crashes the API; a model without a usable
  threshold gets no low-confidence flags and the Review Queue says why. The Model Arena tells a missing Phase 3b
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

# AfterSight

**Explainable Building Damage Triage from Pre/Post-Disaster Satellite Imagery using Classical Machine Learning**

Given a disaster scene and its building footprints, AfterSight classifies each building's damage level
with classical ML (scikit-learn) and explains *why* the model decided that.

> Scope: *Damage classification on building footprints derived from xBD annotations.* AfterSight does not
> detect buildings in imagery, map hazards, or give rescue advice. See `CLAUDE.md` for the full rules.

Status: **Phase 4 — API.** The full README (architecture, dataset, usage, limitations) is written in Phase 7.

## Quick start (macOS)

Requires Python 3.11 (`brew install python@3.11`) and Node 20+.

```zsh
# Backend — API on http://localhost:8020 (Swagger: /docs)
cd backend
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
uvicorn app.main:app --reload --port 8020
```

Tests run from the repo root or from `backend/` (same venv):

```zsh
source backend/.venv/bin/activate
python -m pytest
```

```zsh
# Frontend — UI on http://localhost:5200 (second terminal)
cd frontend
npm ci
npm test
npm run dev
```

## Data (Phase 1)

AfterSight reads a Parquet mirror of the xBD train split (17 shards in `data/raw/xbd_parquet/`).
See [data/README.md](data/README.md) for the schema, how footprints are derived, and the cache. From the repo root:

```zsh
source backend/.venv/bin/activate
python scripts/inspect_dataset.py --alignment          # verify the shards, write report + figures, fill the scene cache
python scripts/make_fixture.py                         # synthetic test fixtures (NOT xBD)
python scripts/inspect_dataset.py --config config/events.fixture_parquet.yaml
```

## Features and splits (Phase 2)

```zsh
source backend/.venv/bin/activate
python scripts/build_features.py --workers 4   # 37 features per building -> data/processed/features/
python scripts/make_splits.py                  # scene-grouped manifests -> data/splits/
```

Two split designs, both grouped by scene and checked for leakage:
- **within-event generalisation:** a stratified ~70/15/15 split across all 5 events;
- **cross-event generalisation:** hurricane-michael held out entirely as the test set.

The features are documented in [docs/FEATURES.md](docs/FEATURES.md).

## Models (Phase 3)

```zsh
source backend/.venv/bin/activate
python scripts/train.py        # baseline, LR, KNN, DT (+ optional RF), K-Means, confounding check (~5 min)
python scripts/evaluate.py     # print saved comparison tables (never re-touches the test set)
```

Results and an honest reading of them: [docs/RESULTS.md](docs/RESULTS.md).

## API (Phase 4)

```zsh
source backend/.venv/bin/activate
python scripts/calibrate_review.py              # review-queue thresholds, chosen on validation
cd backend && uvicorn app.main:app --port 8020  # Swagger UI: http://localhost:8020/docs
```

The endpoints, rules and confidence meanings are in [docs/API.md](docs/API.md).

## Limitations

- **Footprints are derived from rasterised xBD annotation masks**, not from the original JSON polygons.
  Each connected blob in the pre-disaster footprint mask is treated as one building, and its label is
  the majority damage value inside it. Touching buildings therefore merge into one blob.
- Original xBD building IDs and geographic (lng/lat) coordinates are not available. Polygons are in image
  pixel coordinates only, and no real-world areas are reported.
- Buildings without a damage value (expected to include xBD's `un-classified`) and blobs with unexpected
  mask values are excluded and counted, never assigned a class.
- Only xBD's train split is present in the mirror.
- Full details: [data/README.md § Limitations](data/README.md#3-limitations).

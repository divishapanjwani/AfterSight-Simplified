# AfterSight — rules for working in this repo

Explainable building damage triage on xBD pre/post imagery using **classical ML only**.
UI scope label (use verbatim): *"Damage classification on building footprints derived from xBD annotations."*
Data = Parquet mirror of the xBD train split (masks only, no polygons); footprints come from mask blobs.

## Scope
- Classify damage (no-damage / minor-damage / major-damage / destroyed) on footprints **derived from xBD annotation masks**
  (connected components of t1_mask, majority t2_mask label). Always disclose: merged blobs possible, no original UIDs/lng-lat.
- Never claim: building detection/segmentation, pixel-level change detection, flood/burn/landslide mapping,
  disaster attribution, or rescue/routing advice.
- Polygons render in image pixel coordinates (non-geographic canvas). Never report area in km².
- `un-classified` / unknown labels / unexpected mask values / blobs with no damage value: exclude and count, never coerce.

## ML
- scikit-learn only. Served damage models: **KNN (default)**, **Decision Tree**, **Random Forest**, **SVM** and the
  **soft-voting ensemble** (SVM + RF + KNN); RF, SVM and the ensemble have no per-building explanation. Arena-only:
  GridSearchCV-optimised KNN/DT/SVM, AdaBoost. K-Means is analysis only. The threshold baseline is no longer in active use.
- Logistic Regression is **not a damage model**: it is used only as an analysis classifier for the confounding
  check and the feature-set experiment.
  Optional (after required work): GaussianNB, LinearSVC.
- **Forbidden:** PyTorch, TensorFlow, Keras, CNNs, pretrained vision models, external AI APIs. Do not install them.
- Splits are grouped by scene (event-level hold-out preferred). Fit preprocessing on train only;
  tune on validation; touch test once. Leakage test must pass.
- K-Means clusters are never presented as predictions.

## Integrity
- No invented scores, counts, polygons or dataset statistics. Every on-screen number traces to a persisted artifact.
- Missing data/models → explicit "not configured / not trained yet" state with the fix command.
- Demo mode only as a separate, watermarked mode ("DEMO DATA — not real results"), never mixed with real results.
- Synthetic test fixtures live in `backend/tests/fixtures/` and are never shown as xBD data.
- Never say a test passed without running it.

## Independence
- This is not TerraWatch. Do not open, copy or reuse TerraWatch code, assets, branding, colours or layouts.
  "TerraWatch" appears only in the README's "Related work" paragraph.

## Environment
- User runs **macOS (zsh)**; give macOS/zsh commands.
- Python 3.11 venv at the repo root: `.venv` (activate from the root, then `cd backend` for uvicorn). Ports: backend **8020**, frontend **5200**.
- Data, artifacts and outputs are git-ignored (READMEs excepted).

## Workflow
- Work in phases 0–7; stop and report after each: built · files · commands · real results · blockers · next steps.
- Small verified steps. No empty placeholder modules, no TODO stubs in core features, no swallowed exceptions.
- API: Pydantic schemas, IDs validated against manifest (never file paths), no stack traces in errors,
  business logic outside route handlers, analysis never triggers training.

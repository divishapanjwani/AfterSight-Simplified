# Changelog: AfterSight-Simplified

The simplified clone, branch `simplified`, starts from the original's `paper` branch (`22af80a`, built on
tag `v1.3`). Every change was made **in the clone only**.

## Code and interface (approved rows 1–9 of the change table)

| Change | Files | Why |
|---|---|---|
| Removed the never-used `ready` flag on navigation items and its disabled branch with the tooltip "Arrives in Phase 5 Part B" | `frontend/src/components/AppShell.tsx` | Dead code holding a development-phase note |
| The API status dot's tooltip says "API online" instead of the internal API version ("API v0.1.0") | `AppShell.tsx` | The internal version contradicted the release tags |
| Removed the unused `demo` chip kind | `frontend/src/components/Chip.tsx` | Demo mode was never built. The API's `demo_mode` field (always `false`) is kept, so the API contract is unchanged |
| The Triage Desk shows **Run analysis** only until the result appears; the "Re-run analysis" state is gone | `frontend/src/pages/TriageDeskPage.tsx` | Analyses are deterministic and cached per (scene, model), so re-running only re-fetched the same result. Choosing another model already runs that model's analysis |
| "Phase 3b" → **"feature-set experiment"**: the Arena section tab ("EXPERIMENTS · FEATURE SETS"), the confounding bar labels, the confounding text, the invalid-summary state and the backend error message. The saved `label` string is no longer displayed | `frontend/src/pages/ModelArenaPage.tsx`, `backend/app/experiments/summary.py` | Internal development name. The saved artifact, the artifact folder (`artifacts/experiments/phase3b/`), the script name and the API routes are unchanged |
| Stale note "(review queue, later)" | `frontend/DESIGN.md` | Outdated |

**Kept, by decision:**

- confidence scores and the low-confidence review mechanism (option A: already labelled "not calibrated"
  or "model score");
- every working page, including the training-job API.

**Tests:** wording updated in `ResultsPages.test.tsx`. New assertions check that the old name is not shown
(even though the API still sends it) and that no run button remains after an analysis
(`TriageDeskPage.test.tsx`). No test was removed or weakened.

## Documentation

- `README.md` and `docs/*.md` use "feature-set experiment". Each mentions the old name "Phase 3b" once, next
  to the artifact paths. The README notes that the clone and the original share ports, so only one should
  run at a time.
- **New:**
  - local-only verification files (clone notes, the original's checksum list, the behaviour baseline), kept on the
    machine that made the clone and **not published** in this repository;
  - `docs/screenshots/` (workflow screenshots from the clone);
  - `docs/research/` (final paper and its audit, changelog and reproduction notes);
  - `docs/study-guide/` (14 files);
  - `FINAL_VERIFICATION_REPORT.md`.
- **Left as historical records, unchanged:**
  - `report/`: the submitted mini-project report and the beginner-guide PDF;
  - `paper/`: the earlier paper draft, superseded by `docs/research/`.

## Not changed

Dataset, features, splits, training, models, saved results, configs, API routes and response shapes. The
behaviour snapshot (12 demo analyses, 752 explanations, Feature Lab, `/models`) is **IDENTICAL** to the
baseline taken before any edit.

## Enhancements for the faculty review (branch `enhancements`)

| Change | Files | Why |
|---|---|---|
| **Random Forest served again** (Triage Desk, review queue, readiness, reports), reusing the saved model (not retrained) and its saved validation threshold (0.3865). Its Case File shows the class scores and "No per-building explanation for this model." | `config/training.yaml`, `backend/app/inference/analysis.py`, `schemas/api.py`, `CaseFile.tsx` | Faculty request |
| **SVM** (shared preprocessing + `SVC(class_weight="balanced", probability=True)`, RBF, C=1) on a scene-grouped random sample of training buildings (10,128 within-event, 10,061 cross-event; seed 42); class order checked against the other models | `backend/app/experiments/model_comparison.py`, `training/models.py` (`svm` pipeline) | SVC does not scale to ~50k buildings in our time budget |
| **GridSearchCV** (GroupKFold(3) by scene, `f1_macro`, preprocessing inside the pipeline, within-event training split only) for KNN, DT and SVM; best configurations refitted as "optimised" versions and evaluated once on the within-event test set | same | Faculty request; never uses the test sets for selection |
| **Soft-voting ensemble** (equal-weight average of `predict_proba` of SVM + RF + KNN) via `VotingClassifier(voting="soft")` over `FrozenEstimator` members, so nothing is refitted; checked to equal the plain average | same | Parallel ensemble |
| **AdaBoost** over 200 decision stumps (optional, serial ensemble), appended with `--adaboost-only` | same | Optional step; time allowed it |
| New endpoint `GET /api/results/model-comparison`; new Arena card **"ARENA · 08 Model Comparison & Ensemble Analysis"** (comparison table, macro-F1 chart with a within/cross toggle, GridSearchCV table, confusion matrix and per-class metrics for any model, ensemble explanation) | `routes.py`, `frontend/src/components/ModelComparisonSection.tsx`, `ModelArenaPage.tsx` | Show everything in the app, from saved results only |
| **Threshold baseline removed from active use**: Arena rows and charts, `/models/{name}/metrics`, comparison rows and `evaluate.py`. Its saved files stay on disk | `routes.py`, `ModelArenaPage.tsx`, `scripts/evaluate.py` | Faculty request |
| **Two Arena cards removed from the UI**: "Event-confounding check" and "3-class experiment" (with their frontend-only code: `ConfoundingSection`, `api.confounding`, the `Confounding` type and the 3-class queries). Their saved results, `GET /api/results/confounding`, `GET /api/results/comparison?scheme=three_class` and their docs are kept. The model-comparison card is now ARENA · 06 | `ModelArenaPage.tsx`, `services/api.ts`, `types/api.ts` | Faculty choice |

**Real results** are in the README and `docs/VIVA_NOTES.md`. Soft voting (0.542 within, 0.282 cross) beats KNN and
the SVM but not the Random Forest (0.546, 0.291). GridSearchCV kept KNN (k=15, uniform) and SVM (RBF, C=1) at their
defaults and chose a depth-12 Decision Tree (test macro-F1 0.470 → 0.474). AdaBoost: 0.396 within, 0.273 cross.

**Unchanged:** KNN and DT predictions, confidences and explanations are identical to the snapshot taken before this
branch; the served RF's analyses match the snapshot from before the two-model change (only its label lost
"(optional)"). No served model was retrained.

**Review Queue** (test split, 10,706 buildings) now checks disagreement across three served models: KNN 6,107 flagged
(was 5,889; low-confidence 2,388 unchanged), DT 7,457 (was 7,372; low-confidence 5,652 unchanged), RF 5,436
(1,624 low-confidence).

**Honest note:** `run_model_comparison.py` was accidentally run a second time during development (a deterministic
rerun with the same seed). Every test result was identical and nothing was selected using it; the summary records this.

## KNN and Decision Tree; Logistic Regression removed (branch `knn-dt`)

The app's damage models are now **KNN (default)** and the **Decision Tree**, the two models the paper compares.
**Logistic Regression is removed as a damage model** everywhere: serving, UI, Arena, training, reports, tests
and docs. Nothing was retrained or regenerated, and nothing in `data/`, `artifacts/` or `outputs/` was deleted
(the saved LR files stay on disk, unread).

| Change | Files | Why |
|---|---|---|
| Serving list `[knn, decision_tree]`, `default_model: knn` | `config/training.yaml`, `backend/app/training/config.py` (defaults) | The paper's two models are the app's two models |
| The Decision Tree is served again: model pickers, analysis, review queue, readiness, reports | config only (everything reads the list) | |
| The Case File's **decision-path view** (yes/no questions from root to leaf, leaf shares) is restored verbatim from git history (`bc15abf^` frontend, `ddc7b85^` backend) | `frontend/src/components/CaseFile.tsx`, `types/api.ts`, `backend/app/inference/analysis.py`, `schemas/api.py` | Recovered, not rewritten |
| `logistic_regression` gets `404 model_not_found` like any unknown model; cached LR analyses/PDFs stay on disk but are hidden and refused (`model_not_served`); the Arena endpoints refuse it (`422`) and leave its comparison rows out | `backend/app/api/routes.py` | Same handling DT/RF had before |
| The LR contribution view, the `lr_contributions` kind and its types are removed from the app | `CaseFile.tsx`, `types/api.ts`, `schemas/api.py`, `analysis.py` | No served model uses them |
| Model Arena: baseline + KNN + DT; the "research comparison: not served in the app" labels are gone; KNN preselected | `frontend/src/pages/ModelArenaPage.tsx` | DT is a normal served model |
| Readiness headline: each served model's saved test macro-F1 (KNN 0.520 → 0.271, DT 0.470 → 0.294), read from `/api/results/comparison` | `frontend/src/pages/ReadinessPage.tsx` | It quoted LR's 0.527 → 0.290 |
| Review Queue: disagreement is KNN vs DT | config only | |
| PDF limitations quote the report's own model, not the default model; `REPORT_VERSION` 3 | `backend/app/reports/reports.py` | Cached KNN PDFs quoted LR's scores as "the default model" |
| `train.py` no longer trains LR (removed from `REQUIRED_MODELS`) | `backend/app/training/models.py` | Not a damage model |

**Kept, as analysis classifiers (not damage models in the app):**

- the **confounding check** (a multinomial logistic regression predicting the disaster, 82.9% vs 25.2%), its own
  `confounding.grid`, code and saved result; the Arena now names it as such;
- the **feature-set experiment** (`phase3b.py` names `logistic_regression` explicitly): the
  `models.logistic_regression` grid stays in `training.yaml` with a comment, and `make_pipeline`'s LR branch,
  `lr_coefficients` and the LR entry of `CONFIDENCE_KIND` stay so it remains reproducible;
- `lr_contributions` in `backend/app/explain/explain.py`: not used by the app, kept because
  `docs/build/understanding_data.py` computes the LR worked example in `docs/UNDERSTANDING_AFTERSIGHT.md` with it.

**Tests:** updated for KNN + DT. New: LR is rejected (analysis, review queue, Arena metrics, comparison rows);
DT is served and explained (`decision_path`); readiness depends only on KNN + DT (hiding the saved LR changes
nothing, hiding DT makes it PARTIAL); disagreement is KNN vs DT; the decision-path view renders; the headline
and Arena never show an LR row; PDF limitations quote their own model; LR still trains when the experiment
names it.

**Verification:** KNN analyses and explanations are identical to the baseline taken before this change; DT's are
identical to the snapshot from before the two-model change. Review Queue (test split, 10,706 buildings): KNN
flags 5,889 (was 5,586 against LR; low-confidence 2,388 unchanged, disagreement-only 3,198 → 3,501); DT flags
7,372 (5,652 low-confidence, 1,720 disagreement-only). Calibration rerun into a temporary folder: KNN 0.5333,
DT 0.5257, identical to the saved file.

## Two models: LR and KNN only (branch `two-models`, superseded by `knn-dt`)

The app now serves only **Logistic Regression** (default) and **KNN**, and the Model Arena no longer has the
feature-set Experiments section. Nothing was retrained or regenerated, and nothing in `data/`, `artifacts/` or
`outputs/` was deleted.

| Change | Files | Why |
|---|---|---|
| Serving list `[logistic_regression, knn]`; model listing, analysis, explanations, review queue, readiness and reports follow it | `config/training.yaml` | Two models in the app. Everything reads the list from config, so nothing is hardcoded |
| DT and RF requests get `404 model_not_found`, the same as an unknown model (both used to be served) | `backend/app/core/store.py` (unchanged; it already validated against config) | A clean validation error, never a 500 |
| Cached analyses of unserved models stay on disk but are not listed or counted, and opening, explaining or exporting one gives `404 model_not_served` | `backend/app/inference/analysis.py`, `backend/app/api/routes.py` | The app shows only active models; a listed record that could only fail would confuse. No file is deleted |
| The Case File's decision-path view and the Random Forest note are removed; `explain_building` handles LR and KNN | `analysis.py`, `frontend/src/components/CaseFile.tsx`, `types/api.ts` | Unused once DT and RF are not served |
| The Review Queue's disagreement is LR vs KNN | (follows the serving list in `review_queue.py`) | Only served models are compared |
| Readiness counts LR and KNN only; the headline's "best" line considers served models only | `routes.py`, `frontend/src/pages/ReadinessPage.tsx` | It used to name Random Forest (0.546) and the Decision Tree (0.294) as "best" |
| Model Arena: the **Experiments · Feature sets** section, its unavailable/invalid states and command hint, and the feature-set bars and sentences in the confounding check are removed. The 0.829 vs 0.252 result stays | `frontend/src/pages/ModelArenaPage.tsx`, `services/api.ts`, `types/api.ts` | Requested removal. The backend (`/results/experiments`, `/results/overview`, `experiments/summary.py`) and the saved summary are unchanged |
| Model Arena: Random Forest is not shown. The Decision Tree's saved results stay in the dumbbell, the comparison table, the confusion matrices, the per-event table and the tree section, labelled **"research comparison: not served in the app"** (short form "Decision Tree (research)" in selectors and the chart) | `ModelArenaPage.tsx` | The paper compares KNN with the DT. Served vs research is decided from `/api/models`, not hardcoded |
| Triage and Review Queue no longer special-case an "(optional)" label | `TriageDeskPage.tsx`, `ReviewQueuePage.tsx` | No served model is optional |
| `calibrate_review.py --out <folder>` | `scripts/calibrate_review.py` | To check the calibration without overwriting the saved thresholds |
| `snapshot_behaviour.py` and `e2e_check.py` read the served models from `/api/models`; `check` compares every entry the two snapshots share and lists entries no longer produced | `scripts/` | They had the four-model list built in |

**Kept on purpose:**

- Training still produces DT and RF (`scripts/train.py`, `REQUIRED_MODELS` / `OPTIONAL_MODELS` in
  `backend/app/training/models.py`).
- `explain.decision_path` (used by `docs/build/understanding_data.py`) and `explain.tree_structure` (writes the
  saved tree during training; the Arena's research tree endpoint reads it).
- The feature-set experiments' code (`experiments/phase3b.py`, which trains LR and DT), `scripts/run_phase3b.py`
  and `paper_stats.py` (KNN vs DT): they reproduce the paper's results.
- The research endpoints `/models/{name}/metrics`, `/models/decision_tree/tree` and `/results/comparison`.
- The paper (`docs/research/`) is unchanged.

**Tests:** the four-model tests now use LR and KNN. New tests: DT and RF rejected with `model_not_found`
(analysis and review queue); an old DT analysis is not listed and gives `model_not_served` on open, explain,
CSV and PDF, while its file stays; readiness lists only LR and KNN; the review queue compares LR only with
KNN and KNN only with LR; the Arena has no Experiments section and never requests its results; the Arena
labels DT as research and shows no Random Forest; the Readiness "best" line ignores unserved models.

**Behaviour:** `snapshot_behaviour.py check` against the baseline taken before this change: every LR and KNN
entry is **identical** (6 demo analyses, 376 explanations, 37 Feature Lab features, 4 scatter samples, `/models`
entries for LR and KNN). DT and RF entries are no longer produced (382 entries), as expected.

**Review queue (test split, 10,706 buildings):**

| Model | flagged before | flagged after | low confidence | disagreement only, before → after |
|---|---:|---:|---:|---:|
| Logistic Regression | 7,466 | 6,613 | 4,850 (unchanged) | 2,616 → 1,763 |
| KNN | 6,989 | 5,586 | 2,388 (unchanged) | 4,601 → 3,198 |

Low-confidence flags are unchanged; fewer buildings are flagged for disagreement because each model is now
compared with one other model instead of three.

**Calibration check:** `calibrate_review.py --out <scratch folder>` gives LR 0.4836 and KNN 0.5333, identical
to the saved entries; the saved file is byte-identical before and after.

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

## Two models: LR and KNN only (branch `two-models`)

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

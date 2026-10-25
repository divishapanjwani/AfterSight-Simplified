# v1.2: consistency and reliability fixes

*For our group, to prepare for the practical exam.*

v1.2 fixes how the app reports **status** (what is built, what is missing), how it **words** results, and
what happens when a saved file is **missing or broken**. It does not change any model, feature, split or
score.

- The same five models: threshold baseline, Logistic Regression, KNN, Decision Tree, Random Forest.
- The same served set: LR (default), KNN, Decision Tree, and Random Forest marked optional.

## How we checked that the ML is untouched

- **Snapshot before and after.** We recorded `scripts/snapshot_behaviour.py` on v1.1 before any edit, and
  compared after every step. It stayed `IDENTICAL` throughout:
  - 12 demo analyses: every prediction, confidence and score;
  - 752 explanations;
  - all 37 Feature Lab features;
  - `/models`.
- **Real files untouched.** `data/` and `artifacts/` were never edited, nothing was retrained, and the real
  feature-set experiment results were **used, not regenerated**.
- **Edge cases on temporary copies.** We tested them on a temporary copy of `artifacts/` (symlinks to the
  real files, with only the file under test replaced), so the real files were never changed.

## How we tested each fix

| Check | Result |
|---|---|
| backend `pytest` | 146 passed (123 before) |
| frontend Vitest | 33 passed (27 before); TypeScript check, lint and production build pass |
| end-to-end check | 22/22 passed |

---

## 1. Review-threshold readiness: READY / PARTIAL / NOT READY

**What was wrong.**

- The Readiness tile said **READY** whenever `review_thresholds.json` existed, even if a model was missing
  from it or every threshold was `null`.
- A file with broken JSON made four endpoints fail with **HTTP 500**: the readiness board, `/models`, the
  review queue and analysis.

**What changed.** `backend/app/inference/review.py` now has small functions:

- `read_thresholds()`: the file, or *why* it can't be used (missing, not valid JSON, no `models` section).
- `usable_threshold()`: a threshold counts only if it is a number between 0 and 1.
- `threshold_status()`:
  - **READY** if every served model has a usable threshold;
  - **PARTIAL** if only some do;
  - **NOT READY** if the file is missing or broken, or none do.

  Each state comes with a one-line reason naming the models.

An *optional* model (Random Forest, marked `optional: true` in `config/training.yaml`) that isn't trained is
left out of the count and named in the reason. Otherwise it would hold the tile at PARTIAL forever.

The **calibration method is unchanged**: the lowest validation threshold that reaches 70% accuracy, never
using test data.

**How it's tested.**

- Unit tests: missing file, bad JSON, wrong shape, missing entry, `null`, non-numbers, partial, all present,
  and an untrained optional model.
- API tests that swap the file temporarily.
- A frontend test for the PARTIAL badge.
- With the real file, the tile is READY: LR 0.48, KNN 0.53, DT 0.53, RF 0.39.

**Exam answer.** "The tile now checks each served model's threshold instead of just checking that the file
exists, and a broken file is reported instead of crashing the API."

## 2. Models tile: required vs optional

**What was wrong.** The models tile said READY if *any* served model was trained, so one trained model made
the whole tile look ready.

**What changed.**

- READY means every **required** served model is trained.
- PARTIAL means only some of them are.
- NOT READY means none are.
- An untrained optional model keeps the tile READY and is mentioned in the reason, for example
  "random_forest (optional) not trained".

**Exam answer.** "Required models decide the state. Optional ones are only mentioned."

## 3. Review Queue without a usable threshold

**What was wrong.** The backend already did the right thing: no low-confidence flags, while disagreement
flags still worked. But the page still said "confidence below **—** (chosen on validation … 70% correct)",
which suggests a threshold was in use. It also called the remaining buildings "confident".

**What changed.**

- The API adds `threshold_note`, for example "calibration found no threshold for logistic_regression that
  reaches 70% accuracy on validation". It stops quoting the rule and target when there is no threshold.
- The page says "No low-confidence flags for this model", gives the reason, and says that only
  model-disagreement flags are listed.
- A page number past the end of the queue no longer says "nothing is suggested".

**How it's tested.**

- API: the queue with a `null` threshold, a missing entry and a missing file. Each time, zero low-confidence
  flags, disagreement reasons only, and counts that add up.
- Frontend tests for the note and for paging.

**Exam answer.** "Without a threshold chosen on validation we can't call anything low-confidence, so the
queue shows only buildings where the models disagree, and it says why."

## 4. Feature-set experiment results in the Model Arena

**What was wrong.** A missing summary already showed "not available yet" plus the command. But:

- a **malformed** summary (bad JSON) caused HTTP 500;
- an **incomplete** one was served as if it were fine, and two such cases crashed the Arena page;
- the confounding section always claimed what the feature-set experiments found, even with no results.

**What changed.**

- `backend/app/experiments/summary.py` checks the file has what the Arena shows, and returns the saved
  values **unchanged**.
- A broken file gives a structured **503 `results_invalid`** naming the problem (for example "row 3 … is
  missing 'test_macro_f1'") plus `python scripts/run_phase3b.py`.
- The Arena shows a separate "malformed or incomplete" state.
- The feature-set experiment sentence appears only when the results are available.
- The writer and reader already agreed on path and keys. A test now checks that what `run_phase3b` writes
  is exactly what the reader accepts.
- The workflow doesn't record run failures, so there is no "run failed" state; we didn't invent one.

**How it's tested.**

- Backend: valid passthrough, bad JSON, missing `rows` or `variants`, empty rows, a row missing a metric, an
  unknown variant.
- Frontend: the invalid and missing states.
- With the real file, the Arena shows the real 12 rows (identical to `summary.json`).

**Exam answer.** "The Arena now distinguishes 'never run' from 'saved file is broken', and it never shows
numbers from a broken file."

## 5. Report and confidence wording

**What was wrong.**

- With no threshold, the PDF would print "confidence below **None**, threshold chosen on validation".
- The heading "Metrics on this scene" and the overall test score sat close together and could be confused.
- Cached PDFs are never rebuilt, so wording fixes would not reach existing analyses.

**What changed.**

- The PDF mentions the validation threshold only when one exists.
- Low-confidence and model-disagreement flags are explained separately.
- Confidence is described as "a model score, not a calibrated probability of being correct".
- The heading is now "Scene-level metrics (this scene only)", and the test macro-F1 is labelled "overall
  held-out test … not the score of this scene".
- Ground truth is described as "the reference labels … used for research evaluation".
- Missing values read "not available".
- `REPORT_VERSION = 2` in `reports.py` names cached PDFs `<id>.r2.pdf`. Bump it whenever the wording
  changes. Old PDFs are left untouched.
- The Triage Desk labels its scene metrics "scene-level". Readiness, Reports and the Arena show "not
  available" instead of a dash.
- The CSV is unchanged; empty cells mean missing.

**How it's tested.** `tests/test_reports.py` checks:

- the review text with and without a threshold;
- "not available";
- the versioned file name;
- a PDF with missing values;
- the CSV against the analysis.

We also downloaded a real PDF and checked it matches the analysis (36 buildings, 8 flagged at 0.4836).

**Exam answer.** "Every number in the report says what it is: scene-level or overall test, validation-chosen
or not available. Confidence is a model score, not a guaranteed probability."

---

## Concepts to study

- **Robustness:**
  - validating files before using them;
  - structured errors (404 "not built" vs 503 "saved file is broken");
  - failing safe (a missing threshold means *no* low-confidence flags, never a guessed one).
- **Status design:** tri-state status (ready / partial / not ready); required vs optional components.
- **Evaluation honesty:**
  - validation-chosen thresholds;
  - scene-level vs overall test metrics;
  - why confidence is not a calibrated probability.
- **Caching:** cache invalidation by versioning the file name.
- **Testing:**
  - regression tests for edge cases;
  - testing against temporary copies;
  - snapshot (characterisation) tests proving behaviour did not change.

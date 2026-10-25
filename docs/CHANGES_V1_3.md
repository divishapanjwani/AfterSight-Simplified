# v1.3: audit, environment diagnosis and a simpler Cluster Atlas

*For our group, to prepare for the practical exam.*

v1.3 does four things:

- checks the v1.2 reliability fixes independently;
- explains two problems seen on a Windows clone;
- fixes the one audit gap;
- simplifies the Cluster Atlas.

No model, feature, split, score or saved artifact changed. The behaviour snapshot
(`scripts/snapshot_behaviour.py`) was captured before any edit and is still `IDENTICAL`:

- 12 demo analyses;
- 752 explanations;
- 37 Feature Lab features;
- `/models`.

## 1. The inspection bug (`UnboundLocalError` in `inspection.py`)

**What was reported.** On a Windows clone, a local, uncommitted edit to `backend/app/datasets/inspection.py`
made five `test_inspection.py` tests fail with `UnboundLocalError` at
`record.image_mode = pre.mode if …`.

**What we found.**

- The clone used for v1.3 (macOS) has **no** such edit. Its `inspection.py` is exactly v1.2, and all 6
  inspection tests pass.
- In v1.2, `pre, post = images["pre"], images["post"]` sits at function level, straight after the guard
  `if len(images) < 2: return …`, so it always runs before `pre` is used.
- Python raises `UnboundLocalError` when a function uses a local variable on a path where it was never
  assigned. The local edit must have moved that assignment into a branch that doesn't always run: inside an
  `if`, an `except`, or after a `return`/`continue`.

**Fix (to apply on the clone that has the edit).**

1. Commit the edit as it is on a branch.
2. Make sure `pre, post = images["pre"], images["post"]` runs on every path that reaches
   `record.image_mode = …`. Usually that means de-indenting it back to the function level after the early
   returns, or returning early in the branch that skips it.
3. Rerun `python -m pytest backend/tests/test_inspection.py` inside the venv.

**Exam answer.** "A variable was only assigned on some paths, but used on all of them; the fix is to assign
it before every use, not to delete the check."

## 2. The PyTorch test failure (an environment problem, not a code bug)

**What was reported.** `test_no_deep_learning_frameworks_installed` failed on Windows.

**What we found.**

- No project file imports `torch`, `tensorflow`, `keras` or `jax`, and none is in the requirements. The
  only mentions are inside the guard test.
- The tests were run with the **global** Python 3.12, whose user site-packages contain `torch`. The test
  correctly reported that.
- The project's venv (Python 3.11, `include-system-site-packages = false`) has
  `site.ENABLE_USER_SITE = False`, so user site-packages are invisible inside it. The test passes there.
- `sys.flags.no_user_site` reads 0 even in a correct venv, because it only reflects the `-s` flag.
- `python -s -m pytest` would also hide user site-packages, but the venv is the proper fix.

**Fix.**

- Run the tests inside the venv. The README now has the Windows commands:
  `py -3.11 -m venv .venv`, `.\.venv\Scripts\Activate.ps1`, `python -m pip install -r backend\requirements.txt`.
- If activation is blocked, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` first.
- The test was not weakened, and torch was not uninstalled.

**Exam answer.** "The test caught a deep-learning library in the global Python, not in our project; inside
the project's own environment it passes, which is exactly what the test protects."

## 3. Audit of the v1.2 fixes

| Item | Outcome | Evidence |
|---|---|---|
| D1 threshold readiness | **fixed now** | see below; tests in `test_review_thresholds.py`, `test_api.py::test_threshold_tile_*`, `ReadinessPage.test.tsx` |
| D2 Review Queue | already correct | `test_api.py::test_review_queue_without_a_usable_threshold`, `ResultsPages.test.tsx` (no-threshold note, paging) |
| D3 feature-set experiments | already correct | `test_api.py::test_phase3b_summary_*`, `test_phase3b.py` (writer output accepted by reader); real rows identical to `summary.json` |
| D4 models | already correct | per-model `trained` flags, models tile counts required models only; Case File explanation for LR, KNN, DT and RF in the snapshot (752 explanations) |
| D5 reports and wording | already correct | `test_reports.py`; every UI and PDF mention of "probability" says "not calibrated" or "model score" |

**The D1 gap.** v1.2 left an optional model out of the count only when it was *untrained*. A **trained**
Random Forest with a missing or `null` threshold still made the tile PARTIAL. Now only the **required**
served models decide READY / PARTIAL / NOT READY. Optional models are named in the reason, for example
"random_forest (optional) has no usable threshold", and are still listed on the tile. The page shows the
reason in every state. Calibration is unchanged (validation only).

**Exam answer.** "Optional models can't hold the system back: only the required ones decide whether review
thresholds are ready."

## 4. Cluster Atlas, simplified

**How it worked before.**

- K-Means results are **precomputed** by `scripts/run_kmeans.py` for k = 2…10 and saved in
  `artifacts/analyses/kmeans/<design>.json`.
- The page reads them through `GET /clusters?k=…`; nothing is refitted.
- It showed:
  - an inertia (elbow) curve and a silhouette-vs-k curve;
  - cluster sizes, silhouette and ARI;
  - a "can / can't" note;
  - a PCA scatter (with a by-cluster / by-ground-truth toggle);
  - the cluster-vs-damage heatmap.

**What changed.**

- **Removed from the page:** the PCA scatter and the two curves. `LineChart.tsx` was deleted, because only
  this page used it. `ScatterPlot` stays, because the Feature Lab uses it.
- **Kept:**
  - the k selector;
  - the cluster sizes;
  - the silhouette with one line: "how well separated the clusters are, from −1 to 1";
  - the ARI with one line: "agreement between clusters and the reference damage labels: 0 = random,
    1 = identical. It is not prediction accuracy";
  - the heatmap;
  - the can / can't note;
  - the "analysis only, never predictions" label;
  - the missing-artifact state and the unavailable-k state.
- **New wording:** clusters group buildings by their features; a cluster ID is not a damage class; results
  exist only for the precomputed k values.
- **Backend unchanged:** the endpoint, the response (inertia curves and PCA points are still returned, just
  not shown) and the artifacts.

**Exam answers.**

- Silhouette: "How far each building is from its own cluster compared with the next one; near 0 means
  overlapping groups."
- ARI: "Whether the clusters match the damage labels, corrected for chance; 0.095 at k=4 means they barely do,
  and it is not an accuracy score."
- Why no refitting: "K-Means was run once on the training split; the page only shows those saved results."

## Concepts to study

- **Python scope:** local variables and `UnboundLocalError`.
- **Virtual environments:** user site-packages and why tests must run in the project venv.
- **Required vs optional components** in status reporting.
- **K-Means:** silhouette, adjusted Rand index (chance-corrected agreement), precomputed analysis vs live
  fitting.
- **Refactoring safety:** snapshot tests, and keeping an API stable while the UI shows less.

# Simplification notes (v1.1)

*For our group, to prepare for the practical exam.*

v1.1 makes the Case File and the Feature Lab easier to read and explain. It also removes one column from the
Readiness page. It does **not** change what the app computes:

- the same five models;
- the same 37 features and preprocessing;
- the same splits, metrics and API responses.

The original code is tagged `v1.0` (`git checkout v1.0` to see it).

## How we proved nothing changed

`scripts/snapshot_behaviour.py` calls the real API routes with the saved models (nothing is retrained). It
records:

- a fresh analysis of each of the 3 demo scenes with each of the 4 served models (12 analyses): every
  building's prediction, confidence, class scores and review flag;
- the `/explain` response for every building in those analyses (752 explanations);
- `/features/summary` (all 37 features) and 4 fixed `/features/scatter` requests;
- `/models` (the Readiness table's data).

Only timestamps, timings and the "cached" flag are ignored. The analyses go to a temporary folder, so
`artifacts/` is never written.

```zsh
.venv/bin/python scripts/snapshot_behaviour.py save before.json   # on the old code
.venv/bin/python scripts/snapshot_behaviour.py check before.json  # on the new code
# IDENTICAL: 12 analyses, 752 explanations, 37 summary features, 4 scatter samples
```

We ran `check` after every step below, and it printed `IDENTICAL` each time. We also ran these on the final
code:

- 123 backend tests;
- 27 frontend tests (26 old + 1 new);
- the TypeScript check and the production build;
- the end-to-end check (22/22).

The new Case File (LR) and Feature Lab screenshots are pixel-identical to v1.0. In the KNN screenshot, 83 of
360,096 pixels differ, all inside the photo thumbnails, where Chrome resamples the same JPEG crops slightly
differently between runs.

---

## 1. Case File: the crop window

**File:** `frontend/src/components/CaseFile.tsx`, the new `cropWindow()` and `polygonPath()`.

**What the original did.** It cut a square out of the 1024 × 1024 image around the building, with a margin,
and drew the outline. All of that was packed into one-liners:

```ts
const inside = (c: number) => Math.min(Math.max(c, Math.min(half, 512)), 1024 - Math.min(half, 512))
const d = building.polygon.map((rings) => rings.map((r) => `M${r.map(([x, y]) => `${x},${y}`).join('L')}Z`).join('')).join('')
```

**Why it was hard to follow.** Three nested `min`/`max` calls, a magic number (512) and three nested `map`s
on one line. It took a while to see what each part was for.

**What changed.**

- `cropWindow(polygon)` computes `minX`, `maxX`, `minY`, `maxY`, the side, the half-width with margin, and a
  clamped centre, each as a named variable.
- `IMAGE_SIZE = 1024` replaces the magic numbers, and `limit = Math.min(half, IMAGE_SIZE / 2)` replaces
  `Math.min(half, 512)`.
- `polygonPath()` builds the SVG path with two plain `for` loops.

**Why it is easier.** Each line does one thing and has a name, and the comment explains why the clamp
exists: buildings at the edge would otherwise show an empty band.

**How behaviour is preserved.** The arithmetic is the same expression, just split up. The LR Case File
screenshot is pixel-identical to v1.0, and the existing Triage → Case File tests pass.

**Concepts.**

- Bounding box.
- Clamping a value to a range.
- SVG `viewBox` (which part of the image is shown) and path syntax (`M` move, `L` line, `Z` close).
- `fill-rule="evenodd"`, which cuts holes in polygons.

**Exam answer.** "The crop is the building's bounding square plus a margin. We clamp its centre so the window
never goes past the image edge. The outline is an SVG path in the same pixel coordinates as the image, so it
lines up exactly."

## 2. Case File: choosing the explanation

**What the original did.** Four separate conditional lines, one per model:

```tsx
{explain.data?.kind === 'lr_contributions' && <LrView e={explain.data} />}
{explain.data?.kind === 'decision_path' && <TreeView e={explain.data} />}
...
```

**Why it was hard to follow.** You had to read all four lines to see that exactly one of them shows. The
Random Forest case was a bare `<p>` with no hint of why it was different.

**What changed.** A new `ExplanationView` component uses one `switch (e.kind)`, with a comment on the
`'none'` case: the Random Forest has no compact per-building explanation, so the API sends a note.

**Why it is easier.** A `switch` says "exactly one of these" in a single place.

**How behaviour is preserved.** The same four views receive the same data. The screenshots for LR, KNN,
Decision Tree and Random Forest show the right explanation each time, and the `/explain` responses are
identical.

**Concepts.**

- `switch` statements.
- Discriminated unions: the `kind` field tells you which shape `detail` has.
- Conditional rendering in React.

**Exam answer.** "The API tells the page what kind of explanation it is sending, and the page picks the
matching view with a `switch`."

## 3. Case File: counting the KNN votes, and short class names

**What the original did.**

```ts
const votes = n.reduce<Record<string, number>>((acc, x) => ({ ...acc, [x.label]: (acc[x.label] ?? 0) + 1 }), {})
```

It also repeated `DAMAGE[c as DamageClass]?.short ?? c` four times.

**Why it was hard to follow.** `reduce` with an object spread is a compact functional idiom that many of us
don't read fluently. The repeated lookup made the lines long.

**What changed.**

- `countVotes()` is a plain `for` loop that adds 1 per neighbour.
- `shortName(cls)` replaces the four repeated lookups. It falls back to the raw key if the API ever sends an
  unknown class.

**How behaviour is preserved.** Both versions insert labels in the order they first appear, so the vote text
reads the same ("destroyed 15"). The KNN screenshot text is identical.

**Concepts.**

- Counting with a dictionary (a frequency table).
- `reduce` vs a loop.
- The `??` (nullish coalescing) operator.

**Exam answer.** "KNN's prediction is a majority vote of the 15 nearest training buildings. We count the
votes per label with a simple loop and show them, so you can see why it predicted that class."

## 4. Backend: `explain_building` helpers

**File:** `backend/app/inference/analysis.py`, the new `crop_urls()` and `feature_row()`.

**What the original did.**

- It built the crop URLs with the same dict comprehension in two places.
- It found the building's feature row through a lambda passed to the cache, all on one line.

**What changed.**

- `crop_urls(uid)` is written once and used twice.
- `feature_row(store, uid)` has a named `build_row_lookup()`, and a comment says the lookup over the ~71k
  rows is built once and kept in memory.
- Two new comments explain the "model was retrained" check and why KNN neighbours are always `"train"`.

**Why it is easier.** Each helper's name says what it returns. `explain_building` now reads as four steps:
load the analysis, check the model, get the row, explain.

**How behaviour is preserved.** All 752 `/explain` responses are identical, and the API tests pass.

**Concepts.**

- Memoisation (caching a computed value).
- The DRY principle (don't repeat yourself).
- Why the pipeline needs a one-row **DataFrame**, not an array: it was fitted with column names.

**Exam answer.** "To explain one building we reload the exact saved model and that building's 37 features,
then ask the model-specific explainer why. If the model was retrained after the analysis, we refuse, because
we would be explaining a prediction it never made."

## 5. Backend: `lr_contributions` as a plain loop

**File:** `backend/app/explain/explain.py`.

**What the original did.** It built the top-8 list in one long dict comprehension, with five rounded values
per feature.

**What changed.** It is now a `for` loop that appends one dictionary per feature, with the same rounding.
New comments say what `z` is (the scaled values), what `k` is (the predicted class) and why we sort by
*absolute* contribution.

**How behaviour is preserved.** The same numbers, the same rounding and the same order. The identical LR
explanations prove it.

**Concepts.**

- Logistic Regression score = intercept + Σ (coefficient × standardised value).
- Log-odds.
- Why the contribution uses the *standardised* value: coefficients are learned on scaled features.

**Exam answer.** "For LR, each feature's push is its coefficient times its standardised value. We show the 8
largest pushes in either direction. They add up, with the intercept, to the class score."

## 6. Feature Lab backend: `class_stats()`

**File:** `backend/app/inference/feature_lab.py`.

**What the original did.** Inside a double loop, it built each class's statistics in one dictionary. Then it
rebuilt that dictionary with a second comprehension that rounded only the float values.

**Why it was hard to follow.** Two comprehensions in a row, and a type check (`isinstance(val, float)`) whose
only job was to skip `n` and `None`.

**What changed.**

- `class_stats(values)` returns the count, mean and 5 percentiles, rounded to 5 decimals.
- It has an explicit early return for a class with no buildings (all `None`).
- New docstrings and comments explain:
  - the **training-split-only** rule (looking at validation or test data here would be peeking);
  - what η² is (between-class sum of squares ÷ total sum of squares);
  - why missing values are skipped;
  - why the scatter sample is stratified and seeded.

**What did not change.** The `eta_squared` formula, the feature order, the sort by η², and the stratified
scatter sample are all kept exactly as they were.

**How behaviour is preserved.** The `/features/summary` JSON (all 37 features × 4 classes) and the 4 scatter
samples are identical. The Feature Lab screenshot is pixel-identical.

**Concepts.**

- η² (eta squared), the effect size from one-way ANOVA: 0 = the classes look the same on this feature, 1 =
  the feature alone separates them.
- Percentiles and box plots.
- Stratified sampling.
- Random seeds and reproducibility.
- Data leakage.

**Exam answer.** "For each feature we measure how much of its variation is explained by the damage class
(η²), using training buildings only so we never look at the data we test on. The box plots show the middle
half of each class. Overlapping boxes mean that feature alone can't separate the classes."

## 7. Feature Lab page and charts

**Files:** `frontend/src/pages/FeatureLabPage.tsx`, `frontend/src/charts/BoxPlot.tsx`,
`frontend/src/charts/ScatterPlot.tsx`.

**What the original did.**

- The x and y `<select>` boxes were copy-pasted.
- The URL helper was called `set` and a CSS string was called `select`.
- A hard-coded `SKEWED` list chose log axes.
- In `BoxPlot`, a five-part null check decided whether a box could be drawn.
- In `ScatterPlot`, the expression `pad.l + t * (width - pad.l - pad.r)` appeared six times.

**What changed.**

- One `FeatureSelect` component is used twice, and `setParam()` replaces `set`.
- `SKEWED` is renamed `LOG_AXIS_FEATURES`, with a comment that it lists the same 7 features the training
  pipeline log-transforms, and that it only affects drawing.
- `hasStats()` is a TypeScript type guard.
- `plotWidth`, `plotHeight`, `xAt()`, `yAt()` and `axisValueAt()` give the chart geometry names.

**Risk we avoided.** We could have read the log-axis list from the API instead of keeping it in the
frontend. That would add a request and change how the page loads, so we kept the list and documented it.

**How behaviour is preserved.** The same pixel positions (the Feature Lab screenshot is identical), and the
Feature Lab test passes.

**Concepts.**

- Reusable components and props.
- URL query parameters as page state.
- TypeScript type guards (`s is CompleteStats`).
- Linear scales: mapping data values to pixels.
- Log axes for skewed data.

**Exam answer.** "The scatter maps each value to a pixel with a linear scale. For the 7 very skewed features
we scale by log10, so a few huge buildings don't squash everything else into a corner."

## 8. Readiness: the validation column removed

**File:** `frontend/src/pages/ReadinessPage.tsx`, plus a new test in `ReadinessPage.test.tsx`.

**What the original did.** The "Models served to the Triage Desk" table had two macro-F1 columns:
VALIDATION and TEST SET.

**What changed.** The VALIDATION column (header and cell) is gone. The test macro-F1, "Confidence means" and
"Review below" columns stay. Nothing else on the page changed.

**What we kept.**

- `GET /api/models` still returns `val_macro_f1`; it is just no longer displayed on that page.
- The validation split, validation tuning, the validation metrics in the saved artifacts, and the Model
  Arena's validation column all remain.

**Why it is easier.** A headline table that shows only the honest test score has one less number to explain.
Validation scores are optimistic, because they were used to choose the settings.

**How behaviour is preserved.** A new test checks that the table shows the test macro-F1 (0.622 in the
fixture), the confidence meaning and the review threshold, and does **not** show the validation value (0.611)
or the VALIDATION chip. We confirmed the test fails on the old page and passes on the new one. The new
screenshot differs from v1.0 only in the table and in the "Last analysis" tile, whose cached-run count changes
whenever analyses are run.

**Concepts.**

- Validation vs test scores, and why validation is optimistic.
- API contracts: removing a field from the UI doesn't require removing it from the API.

**Exam answer.** "Validation scores were used to pick hyperparameters, so they're slightly optimistic. The
Readiness page now shows only the test score, which was measured once on data never used for any choice."

## 9. New tool: `scripts/snapshot_behaviour.py`

**Why we added it.** "We only refactored" is a claim. This script turns it into a check anyone can run. It
is a simple form of **regression testing** (also called characterisation testing): record what the code
does, change the code, and confirm the output is the same.

**Concepts.**

- Regression testing.
- FastAPI's `TestClient`.
- Dependency overrides, which here point the analysis cache at a temporary folder.
- JSON comparison.

---

## What we deliberately did not change

- **Models:** the threshold baseline, Logistic Regression, KNN, Decision Tree and Random Forest. Their
  definitions, grids, training, evaluation, saved files and predictions are all unchanged.
- **Data:** dataset loading, building extraction, the 37 features and their order, log1p + imputation +
  scaling, the splits and manifests, the label mapping and the damage classes.
- **Interfaces:** every API endpoint and response shape, the config files, and the design system.
- No new dependencies. No retraining. Nothing in `data/`, `artifacts/` or `outputs/` was regenerated.

## Concepts to study

1. **ML:**
   - training / validation / test splits, and why validation scores are optimistic;
   - data leakage;
   - Logistic Regression contributions (coefficient × standardised value, log-odds);
   - KNN majority voting;
   - Decision Tree paths;
   - why Random Forests lack a single readable path;
   - standardisation (z-scores) and log1p for skewed features;
   - η² (eta squared) and one-way ANOVA;
   - percentiles and box plots;
   - stratified sampling and random seeds.
2. **Programming:**
   - helper functions and the DRY principle;
   - `for` loops vs `map`/`reduce`;
   - `switch` statements and discriminated unions;
   - TypeScript type guards;
   - memoisation;
   - React components, props and conditional rendering;
   - URL query parameters as state;
   - SVG `viewBox`, paths and linear scales.
3. **Software engineering:**
   - refactoring vs changing behaviour;
   - regression (snapshot) testing;
   - positive controls (making sure a test can fail);
   - API contracts;
   - git tags and branches (`v1.0`, `simplify`).

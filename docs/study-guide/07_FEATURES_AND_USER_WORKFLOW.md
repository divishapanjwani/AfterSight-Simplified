# 07 · Features and user workflow

The header has two kinds of navigation:

- **The workflow stepper:** `01 Library → 02 Triage → 03 Case file → 04 Arena → 05 Report`.
- **Tools:** Readiness, Feature Lab, Cluster Atlas, Review Queue.

Screenshots of the simplified clone are in `docs/screenshots/`.

## The main workflow: from a scene to an explained prediction

1. **Library** (`/library`, `SceneLibraryPage.tsx`). Choose a scene.
   - It opens on the **held-out test scenes** (177), so you don't look at training data by mistake.
   - Filters: event, disaster type, split. Sorting is available.
   - Each row shows pre/post thumbnails, the split chip, the building count, a ground-truth damage bar and
     data-quality warnings (e.g. a possible misalignment).
   - Training and validation scenes carry a warning that results on them are optimistic.
2. **Triage Desk** (`/triage/<scene>`, `TriageDeskPage.tsx`, viewer `viewer/SceneViewer.tsx`).
   - Pick a model: Logistic Regression (default), K-Nearest Neighbours, Decision Tree, or Random Forest
     (optional). Then press **Run analysis**.
   - Once the result appears, the button disappears. Analyses are deterministic and cached, so there is
     nothing to re-run. Choosing another model runs that model's analysis automatically.
   - The swipe slider moves between the pre and post image; you can zoom and pan.
   - **Overlays:**
     - *Ground truth:* the reference labels;
     - *Prediction:* the model's labels;
     - *Correct vs wrong:* green solid outline = agrees, red dashed = differs;
     - *Off.*
   - Damage classes have patterns as well as colours (outline, dots, hatch, solid), so they can be read
     without colour.
   - **The analysis panel** shows predicted vs reference counts per class, and the **scene-level** accuracy
     and macro-F1 (one scene, a small sample, not the overall test score). It also shows how many buildings
     are suggested for review (confidence below the threshold chosen on validation), what "confidence" means
     for the selected model, and CSV/PDF links.
   - **The building list** can be sorted. "◌" marks a building suggested for review.
3. **Case File** (click a footprint, or open as a full page at `/case/<analysis>/<building>`;
   `components/CaseFile.tsx`).
   - Pre and post crops with the footprint outlined.
   - Predicted vs reference label, with "agrees / differs".
   - The **confidence** value, and the class scores labelled with their meaning:
     - LR: "probability (not calibrated)";
     - KNN: "vote fraction of neighbours";
     - DT: "class fraction in the leaf";
     - RF: "mean of tree leaf fractions".
   - The split warning on training or validation scenes, and any review reasons.
   - **"Why the model decided this":**
     - LR: top feature contributions (coefficient × standardised value);
     - **KNN:** the nearest training buildings, with thumbnails, labels and distances;
     - **DT:** the root-to-leaf questions in raw feature units, and the leaf's class shares;
     - RF: a note that there is no compact per-building explanation, plus global importances.
   - Quality flags: footprint area, mixed labels (possibly merged), at image edge (may be cut off).
4. **Model Arena** (`/arena`, `ModelArenaPage.tsx`). How good is each model, honestly?
   - Within vs cross-event comparison for every model (dumbbell chart).
   - Full comparison table (val and test macro-F1, accuracy, per-class recall, timing).
   - Confusion matrices (test or validation).
   - Per-event scores.
   - The decision tree, as an expandable view.
   - The **event-confounding check**.
   - The 3-class experiment.
   - **"EXPERIMENTS · FEATURE SETS":** the feature-set experiments with reduced feature sets, analysis only
     and not selectable in the Triage Desk.
5. **Reports** (`/reports`, `ReportsPage.tsx`). Every cached analysis with its split, model, scene macro-F1
   and review count, and **CSV** and **PDF** downloads. The PDF:
   - quotes the overall test score separately from scene-level metrics;
   - mentions the validation threshold only when one exists;
   - describes the reference labels as research ground truth;
   - prints "not available" for missing values.

## Tools

- **Readiness** (`/`, `ReadinessPage.tsx`).
  - Status tiles: scene cache, feature table, split manifests, models, review thresholds, last analysis.
  - Each tile is **READY**, **PARTIAL** or **NOT READY**, with a reason. A missing piece shows the command
    that builds it.
  - The headline result card: 0.527 within vs 0.290 cross for the default LR.
  - The table of served models: test macro-F1, what confidence means, review threshold.
- **Feature Lab** (`/features`, `FeatureLabPage.tsx`). Using the **training split only**:
  - every feature ranked by η² (separation score);
  - a per-class box plot of the selected feature;
  - a two-feature scatter from a stratified sample.
- **Cluster Atlas** (`/clusters`, `ClusterAtlasPage.tsx`). Precomputed K-Means for k = 2…10:
  - cluster sizes;
  - silhouette (how well separated, −1 to 1);
  - adjusted Rand index against the reference labels (0 = random, 1 = identical; not accuracy);
  - which damage classes fall in each cluster.

  It is clearly labelled "analysis only, never predictions".
- **Review Queue** (`/review`, `ReviewQueuePage.tsx`). Held-out test buildings **suggested for human
  review**, because their confidence is below the validation-chosen threshold or another served model
  disagrees.
  - Framed as suggestions, not errors.
  - If a model has no usable threshold, the page says why and lists only disagreement flags.

## Removed in the simplified clone (for scope only)

These were not working features: a dead navigation state labelled "Arrives in Phase 5 Part B", an unused
"DEMO" chip, a "Re-run analysis" button that only re-fetched the same cached result, and internal "Phase 3b"
labels. See `CHANGELOG.md`.

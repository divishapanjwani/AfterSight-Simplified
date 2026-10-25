# 12 · Limitations and future work

This file keeps three things apart: **what is implemented**, **what the experiments show**, and **what is
only proposed**.

## What is implemented (and works in this clone)

- **Data:** building extraction from xBD masks, plus a dataset inspector with drop logs and an alignment check.
- **Features:** 37 engineered features, shared preprocessing fitted on train only.
- **Splits:** scene-grouped within-event and cross-event splits with leakage validation and hashes.
- **Models:** KNN, Decision Tree, Logistic Regression, Random Forest (optional) and a threshold baseline, tuned
  on validation, each evaluated on test once.
- **Analyses:** K-Means, the event-confounding check, the feature-set experiments, and the scene-level
  bootstrap (from saved predictions).
- **App:** a web application with explanations (KNN neighbours, DT path, LR contributions), a review queue,
  and CSV/PDF reports.

## What the experiments show (and only that)

- **Within events,** KNN beats the DT on macro-F1 (0.520 vs 0.470). **On held-out Hurricane Michael,** the DT
  is slightly better (0.294 vs 0.271). Both paired bootstrap intervals exclude zero, for fixed trained models.
- **KNN is more accurate in both designs,** because of the majority class.
- **Both models lose a large share of their score** on the unseen event: −48% KNN, −37% DT.
- **The features strongly identify the source event** (0.829 event-prediction accuracy). Removing absolute
  levels and normalising by scene (the feature-set experiments) reduced that, but did not improve cross-event
  macro-F1.

## Genuine limitations

| Area | Limitation | Why it matters |
|---|---|---|
| **Dataset** | Third-party Parquet copy of xBD's **training split only**; official test and hold-out sets unused | Results aren't comparable to published xBD leaderboards |
| | 5 of 10 events, chosen by design decision | A different selection could change the results |
| | xBD targeted imagery below 0.8 m per pixel; minor vs major damage is subtle at that scale | Limits what any image feature can see |
| **Labels / buildings** | Buildings come from masks: touching buildings merge (782 mixed blobs), majority labels | Some labels describe merged groups, not single houses |
| | 1,396 blobs without a damage value are excluded | Possible bias if they differ systematically |
| | No geography (pixel coordinates only) | No map overlay; areas in pixels |
| | 168 pairs flagged for possible pre/post misalignment, kept | Misalignment creates false "change" |
| **Features** | Hand-crafted colour, texture and change; sensitive to sensor, season, lighting and vegetation | Domain shift between disasters |
| **Models** | One training run each (seed 42) | No measure of training variability |
| | The DT is class-weighted, KNN unweighted; grids of 8 vs 6 settings | The comparison isn't fully controlled |
| | Confidence scores are **not calibrated** | A confidence of 0.9 doesn't mean 90% chance of being right |
| **Evaluation** | **One** held-out event | Cross-event results describe Michael only |
| | Bootstrap intervals cover test-scene sampling only | Not training or event variability |
| | Macro-F1 is sensitive to rare classes | Small changes on rare classes move the score |
| **Deployment** | Research and teaching tool; no operational validation; no rescue or routing advice | Must not be used for real emergency decisions |

## Proposed future work (not done)

1. **Equalise class handling:** class-balanced KNN, e.g. resampling the training set, so the KNN–DT comparison
   is controlled.
2. **Repeat training with several seeds,** and report variability across runs.
3. **Hold out each event in turn** (leave-one-event-out over all five), not just Michael.
4. **More events,** and xBD's official test split if a full copy is available.
5. **Calibrate** confidence scores (e.g. isotonic regression) so they can be read as probabilities.
6. **Compare with learned features** (CNNs or Siamese networks) under the same protocol. This is outside this
   project's classical-ML scope.
7. **Use the original polygons** to avoid merged buildings, and add georeferencing.
8. **Domain adaptation** that keeps scene-wide damage. Scene-median normalisation removed real signal too.

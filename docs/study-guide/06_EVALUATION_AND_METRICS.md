# 06 · Evaluation and metrics

Metric code: `backend/app/evaluation/metrics.py` → `classification_metrics`. Saved results:
`artifacts/models/<design>/four_class/<model>.json` → `metrics.test`.

## The confusion matrix

Rows are the **reference (true) label**; columns are the **prediction**; the diagonal is correct.
**KNN, within-event test** (10,706 buildings):

| true ↓ / predicted → | none | minor | major | destroyed | total |
|---|---:|---:|---:|---:|---:|
| **none** | **5,169** | 233 | 729 | 89 | 6,220 |
| **minor** | 1,071 | **477** | 124 | 96 | 1,768 |
| **major** | 911 | 142 | **391** | 96 | 1,540 |
| **destroyed** | 279 | 90 | 34 | **775** | 1,178 |
| column total | 7,430 | 942 | 1,278 | 1,056 | 10,706 |

The first column is the story: 1,071 minor and 911 major buildings are called "no damage".

## The metrics, with worked examples from that matrix

| Metric | Definition | Worked example (KNN, within) |
|---|---|---|
| **Accuracy** | correct / all | (5,169 + 477 + 391 + 775) / 10,706 = 6,812 / 10,706 = **0.636** |
| **Precision** of class c | of all predicted c, the share truly c | destroyed: 775 / 1,056 = **0.734** |
| **Recall** of class c | of all truly c, the share found | destroyed: 775 / 1,178 = **0.658** |
| **F1** of class c | 2PR / (P + R), the harmonic mean | destroyed: 2 × 0.734 × 0.658 / (0.734 + 0.658) = **0.694** |
| **Macro-F1** | plain average of the 4 class F1s | (0.757 + 0.352 + 0.278 + 0.694) / 4 = **0.520** |

**A second example, from the DT cross-event matrix:**

- minor-damage predicted 4,388 times (column sum: 2,426 + 1,206 + 514 + 242), of which 1,206 are correct;
- precision = 1,206 / 4,388 = **0.275**;
- recall = 1,206 / 4,998 = **0.241**;
- F1 = **0.257**.

## Why macro-F1 is the main metric (class imbalance)

58% of test buildings within events are undamaged. A useless model that always answers "no damage" would get:

- **accuracy 0.581**: higher than the DT's 0.484!
- **macro-F1 0.184**: the F1 for "none" is 0.735, and 0 for the three damage classes.

Macro-F1 gives every class equal weight, so it can't be gamed by the majority class. This is also why:

- **KNN's higher accuracy** in both designs (0.636 and 0.554) mostly reflects its strength on "no damage";
- **the DT's higher cross-event macro-F1** reflects its better minor-damage results.

The two metrics give **different winners cross-event**. The paper reports both.

## The two evaluation designs

| | Within-event | Cross-event |
|---|---|---|
| Question | new scenes of disasters seen in training? | a disaster never seen in training? |
| Test set | 177 scenes, 10,706 buildings, all 5 events | all 333 Hurricane Michael scenes with buildings, 21,429 buildings |
| Train / val | 821 / 181 scenes, all 5 events | 716 / 130 scenes from the other 4 events |
| KNN / DT macro-F1 | **0.520** / 0.470 | 0.271 / **0.294** |

**Why the within-event score is optimistic:**

- Within events, test scenes share the satellite, season, region and building style with training scenes.
- The features carry a strong **event signature**: a logistic regression (an analysis classifier, not a damage model in the app) predicts *which disaster* a building
  is from with **0.829** accuracy, against 0.252 for always guessing the largest event
  (`artifacts/analyses/confounding/within_event.json`).
- When Michael scenes were in training, Michael's test scenes scored 0.367 (KNN) and 0.329 (DT); unseen, 0.271
  and 0.294.
- Michael's class mix also differs: 23.3% minor damage, against 13.6% in the other events.

**Validation is optimistic too.** Cross-event validation scenes come from the four training events, so the
selected models scored 0.581 (KNN) and 0.576 (DT) on validation, about twice their test scores on Michael.

## Single test evaluation and the bootstrap

- **One run per model, one test evaluation.** `TestSetGuard` raises an error if a model's test set is
  evaluated twice, which prevents "peeking" until the number looks good.
- **Uncertainty.** `scripts/paper_stats.py` resamples **test scenes** (not buildings, which are correlated
  within a scene) 2,000 times with seed 42, and recomputes macro-F1. The 2.5th–97.5th percentiles give 95%
  intervals:

| | KNN | DT | KNN − DT (paired, same resamples) |
|---|---|---|---|
| Within-event | [0.459, 0.560] | [0.401, 0.526] | +0.050 [+0.013, +0.084] |
| Cross-event | [0.259, 0.284] | [0.284, 0.304] | −0.023 [−0.034, −0.011] |

Both paired intervals **exclude zero**. **What this does and does not mean:**

- It covers **variation over test scenes for fixed, trained models**.
- It does **not** cover retraining variability (each model was trained once) or other disasters (only one
  is held out).
- So the paper says "the interval excludes zero", not "significant".

## Per-class picture

| F1 | KNN within | DT within | KNN cross | DT cross |
|---|---:|---:|---:|---:|
| no damage | 0.76 | 0.56 | 0.73 | 0.68 |
| minor | 0.35 | 0.40 | 0.11 | 0.26 |
| major | 0.28 | 0.33 | 0.04 | 0.03 |
| destroyed | 0.69 | 0.59 | 0.20 | 0.22 |

- **Destroyed** is the easiest damage class: rubble changes texture a lot.
- **Major damage** almost disappears cross-event.
- **Minor vs major** is hard even for people at sub-metre resolution.

## Scene-level vs overall scores in the app

The Triage Desk and the PDF report show **scene-level** metrics: one scene, a small sample. These are not the
overall test score. For example, on the demo scene `santa-rosa-wildfire_00000006` KNN gets a scene macro-F1
of 0.554, against 0.520 overall. The app labels them "Scene-level (this scene only)".

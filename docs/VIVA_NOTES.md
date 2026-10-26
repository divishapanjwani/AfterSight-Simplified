# AfterSight: viva notes

Short answers to likely questions, a 5-minute demo script, and the demo scenes. Numbers are from the saved
results ([RESULTS.md](RESULTS.md)).

## Elevator pitch (30 s)

AfterSight classifies the damage of each building after a disaster (no / minor / major / destroyed) from pre-
and post-disaster satellite images, using **classical, explainable ML** on 37 hand-crafted features. It shows
*why* each prediction was made, with two models: **KNN** (default) and a **Decision Tree**. It is honest about how
well it works: about **0.52 macro-F1** (KNN) on new scenes of disasters it has seen, but only **0.27–0.29** on a
disaster it has never seen. We show that this gap is a genuine
domain shift.

## Likely questions

**Data and labels**

- **What is xBD?** A public dataset (Gupta et al., 2019) of pre/post-disaster satellite image pairs with
  building annotations and 4 damage levels. It is released by the xView2 organisers under CC BY-NC-SA 4.0.
- **Where do your building polygons come from?** Our copy is a Parquet mirror that has masks, not polygons.
  Each connected blob of the footprint mask is a building, and its label is the majority damage value inside
  it. Limitation: touching buildings merge.
- **What about `un-classified` buildings?** They appear as footprints with no damage value. We exclude and
  count them (1,396 blobs in the five selected events); we never guess a class.
- **Why these 5 events?** They span 4 disaster types (wind, flooding, fire, tsunami) and have enough
  major-damage and destroyed examples. Mexico-earthquake and midwest-flooding had almost none.

**Features and preprocessing**

- **Why hand-crafted features?** The syllabus is classical ML, and features are explainable ("texture got
  rougher"). 37 numbers: colour, change, texture (Canny edges, GLCM), surroundings ring, shape.
- **Why log1p?** Seven features are very skewed (|skew| > 2 on train), e.g. area up to 648,034 px. log1p
  compresses the tails. It is fitted inside the pipeline on the training split only.
- **What if images are misaligned?** Most features are region averages, which tolerate small shifts. The
  inspector flags pairs with an estimated shift over 8 px.

**Splits and evaluation**

- **Why group by scene?** Neighbouring buildings share imagery and often damage. A random split would leak
  near-duplicates into test. We assert zero scene overlap in tests.
- **Within- vs cross-event?** Within: new scenes of the same 5 disasters. Cross: hurricane-michael held out
  entirely, an unseen disaster.
- **Why macro-F1?** 58% of buildings are no-damage, so accuracy rewards ignoring the rare classes. Macro-F1
  weights each class equally.
- **How did you avoid overfitting to test?** All tuning is on validation, and test is evaluated once per
  final model, enforced in code by `TestSetGuard`.

**Models**

- **Which model is best?** It depends on the test. KNN (the default) scores 0.520 within-event and 0.271
  cross-event; the Decision Tree 0.470 and 0.294. KNN is better on new scenes of known disasters, the Decision
  Tree holds up slightly better on the unseen disaster. The saved Random Forest reference (0.546 within-event)
  is not served or shown in the app.
- **Why these two models?** They are the two models the paper compares, and they explain themselves in two
  different, easy-to-show ways: KNN by its most similar labelled training buildings, the Decision Tree by the
  yes/no questions from the root to its leaf (both in the Case File).
- **Did you use Logistic Regression?** Not as a damage model in the app. A multinomial logistic regression is
  used only as an *analysis classifier* in the confounding check (and the feature-set experiment), to measure how
  strongly the features identify the disaster.
- **Why is KNN slow?** It stores all 49,929 training buildings and compares each new building with them at
  prediction time (about 20 ms per 1,000 buildings vs 0.3 ms for the Decision Tree).
- **What does the baseline show?** One feature (texture-smoothness change) with three thresholds already gets
  0.383 within-event. Learned models add about 0.14.
- **What does K-Means tell you?** That the strongest natural grouping is *not* damage (ARI 0.04–0.10).
  Clusters are never used as predictions. In the Cluster Atlas, silhouette (−1 to 1) says how well separated the
  clusters are, and ARI (0 = random, 1 = identical) says how well they match the reference labels; ARI is not
  prediction accuracy. Results exist only for the precomputed k = 2…10.

**Honesty and limits**

- **Why the drop from 0.52 to 0.27 (KNN)?** The features identify the disaster with 82.9% accuracy (measured
  with the logistic-regression analysis classifier, not a damage model), so part of the
  within-event score comes from recognising the event. Removing that signature (the feature-set experiments; saved
  results, no longer shown in the app) lowered event prediction to 37% but did not improve cross-event scores. It is
  domain shift.
- **Can I trust the confidence?** It is a model score, not a calibrated probability. For review we use a
  threshold chosen on validation. On the test split KNN flags 2,388 of 10,706 buildings as low-confidence
  (the Decision Tree 5,652); the queue also adds buildings where KNN and the Decision Tree disagree.
- **Is it ready for real disasters?** No. It is a research and teaching tool, with no rescue or routing use.
- **What does PARTIAL on the Readiness board mean?** One of the served models (KNN, Decision Tree) has a usable review
  threshold and the other doesn't (the tile names it). A model without one gets no low-confidence flags; only model disagreement is
  flagged. The thresholds are still chosen on validation only. See [FIXES_V1_2.md](FIXES_V1_2.md).
- **What changed in v1.1, and how do you know nothing broke?** We simplified the Case File and Feature Lab code
  (named helper functions, plain loops, fewer repeated expressions) and removed the validation column from the
  Readiness models table. `scripts/snapshot_behaviour.py` recorded every demo-scene prediction and confidence,
  all 752 explanations, the Feature Lab numbers and `/models` before the change, and they were identical after
  it. See [SIMPLIFICATION_NOTES.md](SIMPLIFICATION_NOTES.md).

## 5-minute demo script

1. **Readiness (30 s):** everything is built. The headline card shows KNN 0.520 within-event vs 0.271
   cross-event (Decision Tree 0.470 vs 0.294), the honest story in one glance.
2. **Library (30 s):** opens on held-out **test** scenes. Switch to Training to show the caution banner, then
   back.
3. **Triage Desk, fire scene `santa-rosa-wildfire_00000006` (60 s):** Run analysis (KNN, the default). Swipe
   pre/post, then the Prediction overlay, then Correct vs wrong. Burned roofs → *destroyed* is the easiest class
   (scene macro-F1 0.55 with KNN, 0.58 with the Decision Tree).
4. **Case file (60 s):** click a destroyed building and show the outlined crops, predicted vs ground truth, and
   the 15 most similar training buildings (KNN). Switch the model to the Decision Tree for its root-to-leaf
   questions.
5. **Wind scene `hurricane-matthew_00000185` (30 s):** all four classes mixed. Minor vs major get confused, which is
   the hardest distinction.
6. **Arena (60 s):** the dumbbell (every model drops on the unseen disaster), then the confounding check (82.9%
   event-prediction accuracy vs 25.2% for always guessing the most common event; an analysis classifier, not a
   damage model). The KNN and Decision Tree rows are the paper's comparison.
7. **Report (30 s):** download the PDF. It carries the split, the metrics and the limitations.

Backup: the flooding scene `hurricane-harvey_00000356` is mostly minor damage under water, and the models struggle
(KNN 0.17, Decision Tree 0.26).

## Demo scenes (pre-cached, work offline)

`python scripts/prepare_demo.py` chose these from the **held-out test split**. Each has 25–160 buildings and
at least 3 ground-truth damage classes, ranked by how evenly the classes are mixed, with the best scene per
disaster type. For KNN and the Decision Tree, the analysis and view images are already on disk and open in the
app (PDFs are written on first download). The LR and RF analyses from earlier versions also stay on disk, but the
app does not list or open them. Generated from
`artifacts/demo/demo_scenes.json`; re-run the script after retraining.

| # | Scene | Disaster type (config) | Buildings | Ground truth: none · minor · major · destroyed | Scene macro-F1: KNN · DT |
|---|---|---|---:|---|---|
| 1 | `hurricane-matthew_00000185` | wind | 37 | 12 · 10 · 8 · 7 | 0.35 · 0.16 |
| 2 | `hurricane-harvey_00000356` | flooding | 115 | 25 · 61 · 27 · 2 | 0.17 · 0.26 |
| 3 | `santa-rosa-wildfire_00000006` | fire | 36 | 14 · 1 · 2 · 19 | 0.55 · 0.58 |

Open one directly: `http://localhost:5200/triage/<scene id>?model=knn&run=1` (or `model=decision_tree`).
Single-scene scores are small samples. Quote the Arena's test-set numbers for overall performance.

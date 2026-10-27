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

- **Which model is best?** It depends on the test. Within-event (macro-F1): Random Forest 0.546, soft voting
  0.542, KNN 0.520, SVM 0.514, Decision Tree 0.470, AdaBoost 0.396. Cross-event: Decision Tree 0.294, Random Forest
  0.291, soft voting 0.282, SVM 0.275, AdaBoost 0.273, KNN 0.271. Every model drops to about 0.27–0.29 on the unseen
  disaster, so no choice of classifier fixes the domain shift.
- **Which models does the app serve?** KNN (default), the Decision Tree and the Random Forest (Triage Desk, Case
  File, Review Queue, reports). The SVM, the ensembles and the optimised versions are in the Model Arena only.
- **Why these two models?** They are the two models the paper compares, and they explain themselves in two
  different, easy-to-show ways: KNN by its most similar labelled training buildings, the Decision Tree by the
  yes/no questions from the root to its leaf (both in the Case File).
- **Did you use Logistic Regression?** Not as a damage model in the app. A multinomial logistic regression is
  used only as an *analysis classifier* in the confounding check (and the feature-set experiment), to measure how
  strongly the features identify the disaster.
- **Why is KNN slow?** It stores all 49,929 training buildings and compares each new building with them at
  prediction time (about 20 ms per 1,000 buildings vs 0.3 ms for the Decision Tree).
- **What happened to the threshold baseline?** It is no longer in active use (faculty request); its saved files
  stay on disk. It was one feature with three thresholds (0.383 within-event).
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

**Model comparison and ensembles (Arena · 08)**

- **SVM?** A support vector machine finds the boundary between classes with the widest margin; the RBF kernel lets
  that boundary curve. We use the same preprocessing as the other models, balanced class weights, and
  `probability=True` (Platt scaling) so it can join soft voting. Test macro-F1 0.514 within-event, 0.275 cross-event.
- **Why sample 10,000 buildings for the SVM?** SVC training time grows roughly with the square of the number of
  samples, so ~50k buildings would take far too long. We took whole training *scenes* in a seeded random order until
  we had 10,128 buildings (168 scenes), so buildings from one scene are never split, as in every other split here.
- **Random Forest?** 200 decision trees, each grown on a bootstrap sample of buildings with a random subset of
  features at each split; the forest averages their votes. That is a **parallel ensemble (bagging)**. Best
  within-event macro-F1 (0.546), but no single readable path, so the Case File shows only its scores.
- **Soft voting?** Each of SVM, Random Forest and KNN gives a probability-like score per class; we average the
  three with equal weights and predict the highest average (sklearn `VotingClassifier(voting="soft")` over the
  already-fitted models, so nothing is retrained). Result: 0.542 within, 0.282 cross. It beats KNN and the SVM,
  but **not** the Random Forest: averaging in KNN, which leans towards "no damage", lowers the rare-class recall
  (within-event minor-damage recall 0.299 for the ensemble vs 0.486 for the Random Forest; major 0.339 vs 0.430).
- **Parallel vs serial ensembles?** Parallel: members are trained independently and combined (soft voting,
  bagging/Random Forest). Serial: each new member is trained to fix the previous ones' errors (boosting). Our
  AdaBoost over 200 decision stumps scored 0.396 within-event and 0.273 cross-event: unweighted boosting of
  one-question stumps under-predicts the rare damage classes (macro recall 0.404).
- **GridSearchCV with grouped CV?** For each setting in the grid, train on two thirds of the training *scenes*
  and score macro-F1 on the remaining third (GroupKFold(3) grouped by scene, so no scene is in both), three times;
  pick the best average, refit it on all training data, and test it once. Preprocessing is inside the pipeline, so
  it is refitted in every fold. Test data never takes part. Results: KNN k=15 uniform (CV 0.499) and SVM RBF C=1
  (CV 0.522) were already the defaults, so their test scores did not change; the Decision Tree chose depth 12
  (CV 0.483), test macro-F1 0.470 → 0.474 (+0.004, a negligible gain).
- **Why not tune on the test set?** Then the test score would no longer be an unbiased estimate; the grouped CV
  keeps selection on training data only.

## 5-minute demo script

1. **Readiness (30 s):** everything is built. The headline card shows KNN 0.520 within-event vs 0.271
   cross-event (Decision Tree 0.470 vs 0.294), the honest story in one glance.
2. **Library (30 s):** opens on held-out **test** scenes. Switch to Training to show the caution banner, then
   back.
3. **Triage Desk, fire scene `santa-rosa-wildfire_00000006` (60 s):** Run analysis (KNN, the default). Swipe
   pre/post, then the Prediction overlay, then Correct vs wrong. Burned roofs → *destroyed* is the easiest class
   (scene macro-F1 0.55 with KNN, 0.58 with the Decision Tree). Then pick **Random Forest** to show the third
   served model.
4. **Case file (60 s):** click a destroyed building and show the outlined crops, predicted vs ground truth, and
   the 15 most similar training buildings (KNN). Switch the model to the Decision Tree for its root-to-leaf
   questions.
5. **Wind scene `hurricane-matthew_00000185` (30 s):** all four classes mixed. Minor vs major get confused, which is
   the hardest distinction.
6. **Arena (60 s):** the dumbbell (every model drops on the unseen disaster). If asked why, quote the confounding
   check (82.9% event-prediction accuracy vs 25.2% for always guessing the most common event; an analysis classifier,
   not a damage model; a saved result, no longer an Arena card). The KNN and Decision Tree rows are the paper's
   comparison. Then **Arena · 06**: the comparison
   table, the GridSearchCV table and the soft-voting explanation (0.542 vs the Random Forest's 0.546).
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

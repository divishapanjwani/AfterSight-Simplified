# AfterSight: viva notes

Short answers to likely questions, a 5-minute demo script, and the demo scenes. Numbers are from the saved
results ([RESULTS.md](RESULTS.md)).

## Elevator pitch (30 s)

AfterSight classifies the damage of each building after a disaster (no / minor / major / destroyed) from pre-
and post-disaster satellite images, using **classical, explainable ML** on 37 hand-crafted features. It shows
*why* each prediction was made. It is honest about how well it works: about **0.53 macro-F1** on new scenes of
disasters it has seen, but about **0.29** on a disaster it has never seen. We show that this gap is a genuine
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

- **Which model is best?** Within-event: Random Forest 0.546 and LR 0.527. Cross-event: all about 0.29 (DT
  0.294). LR is the app default: nearly as good, fast, and the most explainable.
- **Why is KNN slow?** It stores all 49,929 training buildings and compares each new building with them at
  prediction time (about 20 ms per 1,000 buildings vs 0.2 ms for LR).
- **What does the baseline show?** One feature (texture-smoothness change) with three thresholds already gets
  0.383 within-event. Learned models add about 0.14.
- **What does K-Means tell you?** That the strongest natural grouping is *not* damage (ARI 0.04–0.10).
  Clusters are never used as predictions. In the Cluster Atlas, silhouette (−1 to 1) says how well separated the
  clusters are, and ARI (0 = random, 1 = identical) says how well they match the reference labels; ARI is not
  prediction accuracy. Results exist only for the precomputed k = 2…10.

**Honesty and limits**

- **Why the drop from 0.53 to 0.29?** The features identify the disaster with 82.9% accuracy, so part of the
  within-event score comes from recognising the event. Removing that signature (the feature-set experiments) lowered event
  prediction to 37% but did not improve cross-event scores. It is domain shift.
- **Can I trust the confidence?** It is a model score, not a calibrated probability. For review we use a
  threshold chosen on validation. LR flags 40% of buildings as low-confidence; the queue also adds model
  disagreements.
- **Is it ready for real disasters?** No. It is a research and teaching tool, with no rescue or routing use.
- **What does PARTIAL on the Readiness board mean?** Some served models have a usable review threshold and some
  don't (the tile names them). A model without one gets no low-confidence flags; only model disagreement is
  flagged. The thresholds are still chosen on validation only. See [FIXES_V1_2.md](FIXES_V1_2.md).
- **What changed in v1.1, and how do you know nothing broke?** We simplified the Case File and Feature Lab code
  (named helper functions, plain loops, fewer repeated expressions) and removed the validation column from the
  Readiness models table. `scripts/snapshot_behaviour.py` recorded every demo-scene prediction and confidence,
  all 752 explanations, the Feature Lab numbers and `/models` before the change, and they were identical after
  it. See [SIMPLIFICATION_NOTES.md](SIMPLIFICATION_NOTES.md).

## 5-minute demo script

1. **Readiness (30 s):** everything is built. The headline card shows 0.527 within-event vs 0.290
   cross-event, the honest story in one glance.
2. **Library (30 s):** opens on held-out **test** scenes. Switch to Training to show the caution banner, then
   back.
3. **Triage Desk, fire scene `santa-rosa-wildfire_00000006` (60 s):** Run analysis (LR). Swipe pre/post, then the
   Prediction overlay, then Correct vs wrong. Burned roofs → *destroyed* is the easiest class
   (scene macro-F1 0.73).
4. **Case file (60 s):** click a destroyed building and show the outlined crops, predicted vs ground truth, and
   the LR contributions. Switch the model to Decision Tree for the path of questions, and to KNN for the 15
   similar training buildings.
5. **Wind scene `hurricane-matthew_00000185` (30 s):** all four classes mixed. Minor vs major get confused, which is
   the hardest distinction.
6. **Arena (60 s):** the dumbbell (every model drops on the unseen disaster), the confounding check (82.9%),
   and the Experiments section (removing the signature didn't help).
7. **Report (30 s):** download the PDF. It carries the split, the metrics and the limitations.

Backup: the flooding scene `hurricane-harvey_00000356` is mostly minor damage under water, and the models struggle
(LR 0.21).

## Demo scenes (pre-cached, work offline)

`python scripts/prepare_demo.py` chose these from the **held-out test split**. Each has 25–160 buildings and
at least 3 ground-truth damage classes, ranked by how evenly the classes are mixed, with the best scene per
disaster type. For every model, the analysis, view images, CSV and PDF are already on disk. Generated from
`artifacts/demo/demo_scenes.json`; re-run the script after retraining.

| # | Scene | Disaster type (config) | Buildings | Ground truth: none · minor · major · destroyed | Scene macro-F1: LR · KNN · DT · RF |
|---|---|---|---:|---|---|
| 1 | `hurricane-matthew_00000185` | wind | 37 | 12 · 10 · 8 · 7 | 0.22 · 0.35 · 0.16 · 0.20 |
| 2 | `hurricane-harvey_00000356` | flooding | 115 | 25 · 61 · 27 · 2 | 0.21 · 0.17 · 0.26 · 0.23 |
| 3 | `santa-rosa-wildfire_00000006` | fire | 36 | 14 · 1 · 2 · 19 | 0.73 · 0.55 · 0.58 · 0.78 |

Open one directly: `http://localhost:5200/triage/<scene id>?model=logistic_regression&run=1`.
Single-scene scores are small samples. Quote the Arena's test-set numbers for overall performance.

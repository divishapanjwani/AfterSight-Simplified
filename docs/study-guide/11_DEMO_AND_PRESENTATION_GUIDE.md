# 11 · Demo and presentation guide

## Before you start

1. Make sure the original `~/aftersight` isn't running: both copies use ports 8020 and 5200.
2. `cd ~/AfterSight-Simplified && ./start.sh`, then open http://localhost:5200.
3. The three demo scenes are **pre-cached** (analyses and PDFs), so the demo works offline and instantly.
   They were chosen by `scripts/prepare_demo.py` from the **held-out test split** (deterministic rule: 25–160
   buildings, at least 3 damage classes, ranked by how evenly the classes are mixed, best scene per disaster
   type, top 3 types):

| Scene | Hazard | Buildings | Reference labels (none / minor / major / destroyed) | Scene macro-F1: KNN · DT |
|---|---|---:|---|---|
| `santa-rosa-wildfire_00000006` | fire | 36 | 14 / 1 / 2 / 19 | 0.554 · 0.581 |
| `hurricane-matthew_00000185` | wind | 37 | 12 / 10 / 8 / 7 | 0.349 · 0.159 |
| `hurricane-harvey_00000356` | flooding | 115 | 25 / 61 / 27 / 2 | 0.173 · 0.261 |

Scene-level scores are small samples, not the overall result. Say so.

## A 5–7 minute demo script

1. **Readiness (30 s).** "Everything is built: 1,239 scene pairs, 71,340 buildings, 37 features, both split
   manifests, two served models." Point at the headline: **0.520 within-event vs 0.271 cross-event** for the
   default KNN (Decision Tree 0.470 vs 0.294). "The gap is the honest story."
2. **Library (30 s).** It opens on **held-out test scenes**. Open `santa-rosa-wildfire_00000006`.
3. **Triage Desk (90 s).**
   - Drag the swipe slider: the burned houses turn ash-grey.
   - Choose **K-Nearest Neighbours** and **Run analysis**.
   - Switch the overlay to *Correct vs wrong*.
   - Point at the scene-level numbers and say they cover one scene only.
   - Then pick **Decision Tree**; it runs automatically.
4. **Case File, KNN (60 s).** Click building `b0002`.
   - Predicted **destroyed** = reference **destroyed**.
   - Confidence **1.00** means "15 of 15 neighbours voted destroyed" (a vote fraction, not a probability).
   - Show the 15 nearest training buildings, all destroyed, nearest distance 1.447.
5. **Case File, Decision Tree (45 s).** Same building with the Decision Tree. Show the 8 yes/no questions from
   the root to a leaf of 971 training buildings, almost all destroyed (see `05_DECISION_TREE_EXPLAINED.md`).
6. **Model Arena (90 s).**
   - The dumbbell: every model drops on the unseen disaster.
   - The table: **KNN 0.520 vs DT 0.470 within; 0.271 vs 0.294 cross**.
   - The confusion matrices: KNN calls most minor and major damage "no damage".
   - The confounding check: **0.829** event-prediction accuracy, so the features recognise *which disaster*
     (an analysis classifier, not a damage model).
   - If asked: the feature-set experiments (saved results, no longer shown in the app) reduced that signature
     but did not fix transfer.
7. **Reports (30 s).** Download the PDF. It separates the overall test score from scene-level metrics.

## Presenting the findings (keep to the evidence)

- **Claim:** "Within known disasters KNN is better on macro-F1 (0.520 vs 0.470); on the held-out Hurricane
  Michael the tree is slightly better (0.294 vs 0.271). The 95% scene-bootstrap intervals of both differences
  exclude zero."
- **Qualify:**
  - "These intervals cover test-scene variation for one trained model each, not retraining."
  - "One held-out disaster is not all disasters."
- **Explain** (as hypotheses): the class-weighted tree pushes towards minority classes, and the event
  signature makes known-disaster scores optimistic.
- **Never say:**
  - "the model is X% accurate at detecting damage" without the design;
  - "probability" for KNN or DT confidence;
  - "ready for disaster response".

## Likely live questions during the demo

- *"Why does the tree get more minor damage right?"* Balanced class weights; KNN has none (file 05).
- *"Why is the scene score different from 0.520?"* One scene is a small sample; 0.520 is over 177 test scenes.
- *"Where do the building outlines come from?"* xBD annotation masks, 4-connected blobs. Not detected.

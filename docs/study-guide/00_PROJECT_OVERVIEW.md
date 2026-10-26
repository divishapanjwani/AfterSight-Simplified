# 00 · Project overview

*This guide is part of the AfterSight study guide (`docs/study-guide/`). Every number comes from the saved
results in this repository, and they are the same numbers as in the final paper
(`docs/research/AfterSight_Final_Research_Paper.pdf`).*

## The problem

After a hurricane, flood, wildfire or tsunami, relief teams need to know quickly **which buildings are
damaged and how badly**. Satellites photograph a whole city before and after the event, but nobody can
inspect tens of thousands of roofs by eye.

AfterSight predicts one of **four damage levels** for every building in a pre/post satellite image pair:

- **no damage**;
- **minor damage**;
- **major damage**;
- **destroyed**.

It also shows **why** the model decided.

## The research question (from the paper)

> How do a **K-Nearest Neighbours (KNN)** classifier and a **Decision Tree (DT)**, trained on 37 engineered
> pre/post-event image features of annotation-derived building regions, compare for four-class damage
> classification, and how does that comparison change between a scene-grouped **within-event** evaluation
> and a **cross-event** evaluation that holds out one disaster (Hurricane Michael)?

It is split into three questions:

- **RQ1:** How do KNN and the DT compare on new scenes from disasters seen in training?
- **RQ2:** How does the comparison change when a whole disaster is held out?
- **RQ3:** Which damage classes cause the errors?

## Objectives

1. Build a leakage-controlled pipeline:
   - whole scenes go to one split;
   - preprocessing is fitted on training data only;
   - each model is tuned on validation and evaluated on test exactly once.
2. Compare KNN and the DT with macro-F1, per-class precision/recall/F1, confusion matrices and accuracy,
   under both evaluation designs.
3. Give uncertainty: 95% confidence intervals from a scene-level bootstrap over the saved test predictions.
4. Report honestly where the comparison is not perfectly controlled: the DT is class-weighted and KNN is not.
5. Deliver a working, explainable web application around the models.

## Scope: what AfterSight does and does not do

| Does | Does not |
|---|---|
| Classify damage on buildings **derived from xBD annotation masks** | Detect buildings in raw imagery |
| Use classical, explainable ML (scikit-learn) | Use deep learning (it is forbidden in this project) |
| Explain each prediction (KNN neighbours, DT path) | Give calibrated probabilities |
| Evaluate on held-out scenes and a held-out disaster | Claim readiness for real emergency response |
| Work offline on a laptop | Use geographic coordinates (everything is in image pixels) |

## Main features of the application

- **Readiness:** what is built and trained, the headline results, the served models.
- **Scene Library:** choose a scene; held-out test scenes are shown by default.
- **Triage Desk:** before/after swipe viewer, damage overlays, Run analysis, building list. The app serves two
  models, the two the paper compares: KNN (default) and the Decision Tree.
- **Case File:** one building: crops, predicted vs reference label, confidence, model explanation.
- **Model Arena:** KNN, the Decision Tree and the threshold baseline under both designs, confusion matrices,
  per-event scores, the tree, and the event-confounding check (made with a logistic-regression analysis
  classifier, which is not a damage model in the app).
- **Feature Lab:** which features separate the classes (training data only).
- **Cluster Atlas:** K-Means groups compared with the labels. Analysis only, never predictions.
- **Review Queue:** buildings suggested for human review (low confidence, or KNN and the Decision Tree disagree).
- **Reports:** CSV and PDF per analysis.

## The key results (test sets, each model evaluated once)

| | KNN | Decision Tree |
|---|---:|---:|
| Within-event macro-F1 [95% CI] | **0.520** [0.459, 0.560] | 0.470 [0.401, 0.526] |
| Cross-event (Michael held out) macro-F1 [95% CI] | 0.271 [0.259, 0.284] | **0.294** [0.284, 0.304] |
| Accuracy (within / cross) | 0.636 / 0.554 | 0.484 / 0.502 |

**In one sentence:** KNN wins on known disasters, the tree wins slightly on the unseen one, and accuracy
favours KNN in both. So the answer depends on the evaluation design and the metric, and the real weakness is
moving to a new disaster.

## Contributions, honestly framed

1. **A leakage-controlled protocol:**
   - scene-grouped and event-held-out splits;
   - preprocessing fitted on train only;
   - a code guard that allows one test evaluation per model.
2. **A KNN–DT comparison** with per-class results and scene-level bootstrap intervals.
3. **A transparent account** of the protocol differences, and of how the conclusion depends on the metric.
4. **A working, explainable application** around the models.

It is an **applied comparative study**. There is no new algorithm, no new dataset and no state-of-the-art
claim.

## Where to go next

| File | What you will learn |
|---|---|
| `01_SYSTEM_ARCHITECTURE.md` | how the parts fit together |
| `02_DATASET_AND_PREPROCESSING.md` | where the buildings and labels come from |
| `03_FEATURE_ENGINEERING.md` | the 37 numbers per building |
| `04_KNN_EXPLAINED.md`, `05_DECISION_TREE_EXPLAINED.md` | the two models |
| `06_EVALUATION_AND_METRICS.md` | how the scores are computed, with our real confusion matrices |
| `07`–`09` | the app, the code and the setup |
| `10`–`13` | viva questions, demo, limitations and revision |

# Viva preparation: the KNN vs Decision Tree paper (Deliverable 6)

All numbers below match the manuscript, and every one comes from the saved artifacts.

## The paper in one minute

We compare two classical, explainable classifiers, KNN and a decision tree, for classifying each building's
damage into four classes (no damage, minor, major, destroyed). Each building is described by 37 numbers
measured on its pre- and post-disaster image patches. We test the two models in two ways:

- **within-event:** new scenes from the same five disasters the models were trained on;
- **cross-event:** a whole disaster, Hurricane Michael, never seen in training.

Results:

- **Within-event:** KNN is better (macro-F1 0.520 vs 0.470).
- **Cross-event:** both drop sharply and the tree is slightly better (0.271 vs 0.294).
- **Accuracy** favours KNN in both designs.

So "which model is better" depends on the evaluation design and the metric, and the real weakness is
moving to a new disaster.

## Key concepts in plain language

- **Research question:** how do KNN and DT compare, and how does that change when a disaster is unseen?
- **Dataset:**
  - a third-party copy of the xBD *training* split;
  - five events, 1,239 image pairs, 71,340 labelled buildings (58% undamaged).
  - Buildings are not detected. They come from the annotation masks: each connected blob of the footprint
    mask is a building, labelled by the majority damage value inside it.
- **Features (37):**
  - colour (12);
  - change between pre and post (8);
  - texture (12): grey spread, Canny edge density, GLCM contrast and homogeneity;
  - surroundings (3): an 8-pixel ring of ground around the building;
  - shape (2): area and compactness.
- **Shared preprocessing:** log(1 + x) on 7 very skewed features, then standardise. It is fitted on training
  data only, so no information leaks from validation or test.
- **KNN:** find the 15 most similar training buildings (Euclidean distance on standardised features) and take
  the majority label. It explains itself with similar examples, and prediction compares against all ~50k
  training buildings.
- **Decision tree:** a sequence of yes/no threshold questions, depth 8, at least 100 buildings per leaf,
  Gini impurity, balanced class weights. It explains itself with a readable rule path, and prediction is very
  fast.
- **Splits:** whole scenes go to one split, because buildings in one image share lighting and damage.
  - Within-event: 821 / 181 / 177 scenes.
  - Cross-event: all 333 Michael scenes (21,429 buildings) are the test set.
- **Tuning:** each grid setting is fitted on train, the best on validation macro-F1 is kept, and test is
  used exactly once.
- **Macro-F1:** the average of the four per-class F1 scores, so rare classes count as much as no-damage.
  Accuracy alone would reward a model that mostly answers "no damage".
- **Bootstrap intervals:** we resample test *scenes* 2,000 times and recompute the score. The 95% range shows
  how much the result depends on which test scenes happened to be chosen.

## Likely questions and answers

1. **Why only KNN and a decision tree?**
   They are the two classical models in the project that give per-building explanations of different kinds:
   similar examples versus rules. That makes them a meaningful pair for an explainable triage tool. Other
   models were trained in the same pipeline and appear only as a context row.

2. **Is this novel?**
   It is an applied comparative study. The contribution is the careful protocol (scene-grouped and
   held-out-event designs, one test evaluation, scene-level intervals) and an honest comparison, not a new
   algorithm or dataset.

3. **Why does the ranking flip between the two designs?**
   Measured facts:
   - KNN is very good at the large no-damage class but weak on minor and major damage;
   - the DT spreads recall across classes;
   - on Michael, the DT's advantage comes mainly from minor damage (F1 0.26 vs 0.11).

   A likely contributor is that the DT used balanced class weights and KNN could not. We did not vary the
   weighting, so that is a hypothesis.

4. **Is the difference statistically significant?**
   The 95% bootstrap interval of the paired difference excludes zero in both designs: +0.013 to +0.084
   within-event, −0.034 to −0.012 cross-event. But this covers only test-scene sampling for fixed models. It
   doesn't include retraining variability or other disasters, so we avoid calling it significant.

5. **Why do both models do so badly on Hurricane Michael?**
   Two measured facts:
   - Michael has a different class mix (23.3% minor vs 13.6% in the other events).
   - The features strongly identify which disaster a building comes from: a classifier predicts the event
     with 82.9% accuracy. When Michael scenes were in training, Michael's test scenes scored 0.367 (KNN) and
     0.329 (DT); unseen, 0.271 and 0.294.

   A plausible reading is that event-specific appearance carries much of the within-event performance.

6. **Why is accuracy higher for KNN but macro-F1 lower cross-event?**
   58% of buildings are undamaged, and KNN labels most buildings "no damage", so it gets many easy ones right
   (high accuracy) while missing damaged ones (low recall on minor and major).

7. **Did you use the official xBD test set?**
   No. Our copy contains only the training split, so all test sets are carved out of it by scene. We say this
   in the paper.

8. **Could the comparison be unfair?**
   Yes, in three ways we report:
   - class weighting (DT balanced, KNN unweighted);
   - grid sizes (6 vs 8 settings);
   - one training run each.

   Both models do share data, splits, features, preprocessing, selection rule and metric code.

9. **What are the biggest limitations?**
   - One held-out event.
   - Mask-derived buildings: touching buildings can merge (782 mixed blobs), and 1,396 unlabelled blobs were
     excluded.
   - Five of ten events chosen by design decision.
   - A third-party data copy.
   - Uncalibrated confidences.
   - No operational validation.

10. **Why resample scenes and not buildings?**
    Buildings in one scene are correlated. Resampling single buildings would treat them as independent and
    make the intervals too narrow.

11. **How fast are the models?**
    Per 1,000 buildings on an Apple M4 laptop: KNN 19.7 ms, DT 0.27 ms. KNN compares each building with
    every training building.

12. **What would you do next?**
    - A class-balanced KNN, e.g. by resampling.
    - Several seeds.
    - Hold out each event in turn.
    - More events.
    - A comparison with learned features under the same protocol.

## Numbers to remember

| | KNN | DT |
|---|---:|---:|
| Within-event macro-F1 [95% CI] | 0.520 [0.459, 0.560] | 0.470 [0.401, 0.526] |
| Cross-event macro-F1 [95% CI] | 0.271 [0.259, 0.284] | 0.294 [0.284, 0.304] |
| Accuracy (within / cross) | 0.636 / 0.554 | 0.484 / 0.502 |
| KNN − DT macro-F1 (within / cross) | +0.050 [+0.013, +0.084] / −0.023 [−0.034, −0.012] | |

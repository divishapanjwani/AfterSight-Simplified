# How the machine learning in AfterSight works

A plain-English walk through every ML decision, in pipeline order. Numbers come from the saved artifacts
(see [RESULTS.md](RESULTS.md)).

## 1. From xBD to labelled buildings (label creation)

xBD gives, for each scene, a **pre-disaster** and a **post-disaster** satellite image (1024×1024 px, RGB).
The Parquet mirror we use stores two annotation masks per scene instead of polygons:

- **footprint mask** (`t1_mask`): 1 where a building is, 0 elsewhere;
- **damage mask** (`t2_mask`): 0 background, 1 no-damage, 2 minor, 3 major, 4 destroyed.

Each connected blob of the footprint mask (4-connectivity) becomes **one building**. Its **label** is the
majority damage value inside the blob. Blobs are dropped, and the drop is logged, when:

- they are under 16 px;
- they have no damage value (probably xBD's `un-classified`);
- the vote ties;
- their value is not 1–4. A full scan of all 2,799 scenes found only values 0–4.

Result for the 5 chosen events: **71,340 buildings** in 1,239 scenes. Class counts: no-damage 41,449 ·
minor 11,766 · major 10,270 · destroyed 7,855.

## 2. The feature vector (37 numbers per building)

Computed on the building's exact pixels in both images ([FEATURES.md](FEATURES.md) has every formula):

| Group | Count | Examples | Idea |
|---|---:|---|---|
| Colour | 12 | mean and spread of R, G, B, pre and post | what the roof looks like |
| Change | 8 | \|Δ mean colour\|, mean pixel change, brightness ratio | how much it changed |
| Texture | 12 | grey spread, Canny edge density, GLCM contrast and homogeneity (pre, post, Δ) | rubble is rough and patchy |
| Context ring | 3 | change of the ground around the building, building-vs-ring contrast change | did the building change *relative to* its surroundings? |
| Shape | 2 | area (px), compactness | size and shape for normalisation; not damage itself |

Most features are **region statistics** (means, spreads, proportions). That makes them tolerant of the few
pixels of misregistration that xBD pre/post pairs can have.

**Never model inputs:** the label, the event, the scene, the disaster type and the quality flags.
`load_feature_table()` returns these separately from X, and a test checks that.

## 3. Honest splits (grouped, two designs)

Buildings in the same scene share imagery, lighting and often damage. If they were split across train and
test, the model would effectively see the test answers. So **splits are made per scene**:

- **Within-event** (*"within-event generalisation"*): scene-grouped, class-stratified 70/15/15 across all 5
  events (`StratifiedGroupKFold`, 20 folds of 5%). The test set is new scenes of disasters the model knows.
- **Cross-event** (*"cross-event generalisation"*): **all** hurricane-michael scenes are the test set (21,429
  buildings). Train and validation come from the other 4 events. The test set is a disaster never seen.

A leakage check runs before every manifest is written, and a pytest test runs it again on the real
manifests: no scene appears in two splits, and the held-out event appears only in test.

## 4. Preprocessing (inside every pipeline, fitted on train only)

1. **log1p** on features whose training-set skewness is above 2. Seven qualified: area, the two GLCM
   contrasts, the three |Δ mean colour| features and mean pixel change. This tames very long tails, such as
   a 648,034 px mall complex.
2. **Median imputation:** there were no missing values in the real data, but it is kept as a safeguard.
3. **StandardScaler:** puts every feature on the same scale, which KNN needs (its distances depend on it).

Because these steps live inside the scikit-learn `Pipeline`, validation and test data never influence
them.

## 5. The algorithms and why each is here

| Algorithm | Why include it | Key settings (tuned on validation) |
|---|---|---|
| **Threshold baseline** (non-ML) | the floor: how far one feature and three thresholds get | best "change" feature + thresholds chosen on validation: `glcm_homogeneity_delta` |
| **K-Nearest Neighbours** | "buildings that look like this were…"; explanation by example; **app default** | k ∈ {5, 15, 31}, uniform/distance → k=15 uniform; slow at inference (compares with all 49,929 training buildings) |
| **Decision Tree** | human-readable if/then rules; served by the app | depth ∈ {4, 6, 8, 12}, min leaf ∈ {20, 100}, balanced → depth 8, leaf 100 |
| **Random Forest** (optional) | a stronger classical reference: many trees averaged | 200 trees, depth ∈ {12, None}, leaf ∈ {2, 10} |
| **K-Means** (unsupervised) | do buildings form natural groups at all? Analysis only | k = 2…10, chosen by silhouette |
| *Logistic regression* (analysis classifier only) | predicts the **disaster**, not damage, to measure the event signature (confounding check, feature-set experiment); not a damage model in the app | multinomial, C ∈ {0.01, 0.1, 1, 10} → 10 |

Tuning uses **validation macro-F1**. The test set is used **exactly once** per final model, and
`TestSetGuard` raises an error if code tries to evaluate test again.

## 6. Class imbalance and why macro-F1

No-damage is 58% of buildings. A model that always says "no damage" would score 58% accuracy and be
useless. So:

- **Balanced class weights:** errors on rare classes cost more during training.
- **Macro-F1 is the primary metric:** F1 is computed per class, then averaged with **equal weight per class**,
  so doing well on *destroyed* and *major* matters as much as on *no-damage*. Accuracy and weighted-F1 are
  reported too.
- **Side effect:** balanced weights push some no-damage buildings into damaged classes: the Decision Tree's
  no-damage recall is 0.434. KNN has no class weights, so it leans the other way (no-damage recall 0.831,
  minor 0.270, major 0.254).

## 7. What the results mean

- **Within-event:** KNN reaches 0.520 test macro-F1 and the Decision Tree 0.470 (the saved RF reference 0.546).
  **Cross-event:** 0.271 (KNN) and 0.294 (Decision Tree).
- **The features know which disaster a building is from:** an analysis classifier (multinomial logistic regression)
  predicting the *event* from the same 37 features scores **82.9%** (majority class 25.2%). Each event alone
  scores below the pooled within-event figure, so part of 0.52 comes from recognising the disaster.
- **Feature-set experiments** (called "Phase 3b" during development; files in `artifacts/experiments/phase3b/`): removing absolute colour levels and normalising by the scene median cut that signature
  (82.9% → 37.4%), but did **not** improve cross-event scores. The gap is real **domain shift**: different
  roofs, imagery and damage appearance across disasters.

## 8. What K-Means can and cannot tell us

- K-Means groups training buildings by feature similarity, with no labels.
- Silhouette is highest at k=2 (0.25), and agreement with the damage labels is weak: adjusted Rand index
  0.04 at k=2 and 0.10 at k=4.
- **Can:** show that the strongest structure in the features is *not* damage. It is more likely roof type,
  scene appearance or disaster.
- **Cannot:** label damage. Clusters have no names and are **never** used as predictions. The Cluster Atlas shows
  the precomputed results per k (sizes, silhouette, ARI and the cluster-vs-label table); a cluster ID is not a
  damage class.

## 9. Explanations

| Model | Per building | Global |
|---|---|---|
| Decision Tree | the exact path of questions, with thresholds converted back to raw units | feature importances |
| KNN | the 15 most similar training buildings, with their labels, distances and crops | — |
| all models | — | permutation importance on validation (drop in macro-F1 when a feature is shuffled) |

## 10. From prediction to coloured footprints

1. The Triage Desk sends `POST /api/analysis {scene_id, model}`.
2. The API looks up that scene's buildings in the feature table and runs the saved pipeline's
   `predict_proba`.
3. The predicted class is the highest score. The confidence is that score, and its meaning depends on the
   model: a KNN vote fraction or a tree leaf fraction (neither is a calibrated probability).
4. Buildings whose confidence is below the threshold, chosen on **validation** so that confident buildings
   are at least 70% correct, are *suggested for human review*.
5. The browser draws each building's polygon in **image pixel coordinates** over the pre/post images. The
   fill pattern and colour show the predicted class: outline = none, dots = minor, hatch = major,
   solid = destroyed. "Correct vs wrong" mode compares with ground truth using solid and dashed outlines.

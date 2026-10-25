# 05 · Decision Tree (DT), explained

## The idea in one line

A flowchart of yes/no questions on single features ("is texture change ≤ −0.146?"), learned from data, that
ends in a leaf predicting a class. AfterSight uses scikit-learn's CART-style `DecisionTreeClassifier`
(Breiman *et al.*, 1984).

**Analogy:** a triage checklist a nurse follows.

## How the tree is built (training)

At every node, starting with all training buildings at the root:

1. Try candidate splits: for each feature *j* and threshold *t*, divide the buildings into "x_j ≤ t" (left)
   and "x_j > t" (right).
2. Score each split by how **pure** the two children are, using the **Gini impurity**.
3. Keep the best split and repeat in each child.
4. Stop when a limit is reached (max depth, minimum leaf size) or a node is pure.

### Gini impurity

> G = 1 − Σ_c p_c²   (p_c = share of class c in the node, here class-weighted)

| Node contents | G |
|---|---|
| 100% one class | 1 − 1² = **0**, pure |
| 50% / 50% of two classes | 1 − (0.25 + 0.25) = **0.5** |
| 25% each of four classes | 1 − 4 × 0.0625 = **0.75**, the worst for 4 classes |

A split is good when the weighted average Gini of the children is much lower than the parent's.

### Balanced class weights

AfterSight uses `DecisionTreeClassifier(class_weight="balanced", random_state=42)` in
`backend/app/training/models.py`. Each class gets the weight

> w_c = N / (4 · N_c)   (N = training buildings, N_c = buildings of class c)

| Within-event training class | N_c | weight |
|---|---:|---:|
| no damage | 29,008 | 0.43 |
| minor | 8,232 | 1.52 |
| major | 7,190 | 1.74 |
| destroyed | 5,499 | 2.27 |

So one destroyed building counts about 5× as much as one undamaged building when the tree chooses its splits.
The result is a tree that predicts minority classes much more often. **KNN has no such option**, so this is
the main protocol difference between the two models.

## Hyperparameters, overfitting and pruning

A tree grown without limits memorises the training data, one leaf per handful of buildings, and overfits. We
control that with two limits, chosen on validation macro-F1:

| max_depth | min_samples_leaf | val macro-F1 (within) | val macro-F1 (cross) |
|---:|---:|---:|---:|
| 4 | 20 / 100 | 0.421 / 0.421 | 0.532 / 0.531 |
| 6 | 20 / 100 | 0.474 / 0.472 | 0.546 / 0.546 |
| **8** | 20 / **100** | 0.510 / **0.511** | 0.570 / **0.576** |
| 12 | 20 / 100 | 0.499 / 0.504 | 0.548 / 0.557 |

**Selected: max depth 8, at least 100 training buildings per leaf, in both designs.** The fitted trees have
115 leaves (within-event) and 111 (cross-event).

- **Deeper trees** (12) scored worse on validation: overfitting.
- **No cost-complexity pruning** was used (`ccp_alpha = 0`); the depth and leaf-size limits act as
  pre-pruning.

**Scaling** doesn't change the tree's choice of splits, because splits only compare a feature with a
threshold and any monotone rescaling keeps the order. The tree still sits in the shared pipeline so both
models get identical input. Explanations convert thresholds back to raw units (`raw_threshold` in
`backend/app/explain/explain.py`).

## The real tree (within-event)

- **Root:** `glcm_homogeneity_delta ≤ −0.146` (texture became much less uniform), over all 49,929 training
  buildings.
- **Left child:** `glcm_homogeneity_post ≤ 0.576`.
- **Right child:** `glcm_homogeneity_post ≤ 0.758`.

So texture change is the single most useful question, which matches the Feature Lab ranking.

## How AfterSight explains a DT prediction

`decision_path` (`backend/app/explain/explain.py`) returns the questions this building answered from the root
to its leaf.

**Example:** `santa-rosa-wildfire_00000006_b0002` (destroyed).

| # | Question | Value | Answer |
|---|---|---:|---|
| 1 | `glcm_homogeneity_delta` ≤ −0.146 | −0.240 | yes |
| 2 | `glcm_homogeneity_post` ≤ 0.576 | 0.515 | yes |
| 3 | `area_px` > 521 | 1,952 | yes |
| 4 | `glcm_homogeneity_delta` ≤ −0.194 | −0.240 | yes |
| 5 | `brightness_ratio` > 0.978 | 1.64 | yes |
| 6 | `post_mean_r` > 127.8 | 155.4 | yes |
| 7 | `ring_dmean` > 5.68 | 32.1 | yes |
| 8 | `contrast_change` > −6.15 | 27.5 | yes |
| → | leaf of 971 training buildings | | destroyed 99.8% |

**Confidence** = the (class-weighted) share of the predicted class in the leaf, here 0.998. It is not a
calibrated probability.

## Our results (test sets, evaluated once)

| | Within-event | Cross-event |
|---|---:|---:|
| macro-F1 [95% scene-bootstrap CI] | 0.470 [0.401, 0.526] | **0.294** [0.284, 0.304] |
| accuracy | 0.484 | 0.502 |
| recall: none / minor / major / destroyed | 0.43 / 0.45 / 0.50 / 0.78 | 0.66 / 0.24 / 0.02 / 0.43 |
| F1: none / minor / major / destroyed | 0.56 / 0.40 / 0.33 / 0.59 | 0.68 / 0.26 / 0.03 / 0.22 |

**How to read it:**

- Because of the class weights, the DT **spreads its predictions**: only 32% "no damage" within events,
  against a true share of 58%. Minority recall is higher, but many undamaged buildings are called damaged:
  major-damage precision is only 0.24.
- It is **worse than KNN within events**, but **slightly better on the unseen disaster**, mainly through
  minor damage (F1 0.26 vs KNN's 0.11).
- It loses 37% of its macro-F1 on the new event, against 48% for KNN.

## Cost

| | Detail |
|---|---|
| Training | about 1.2 s |
| Prediction | 0.27 ms per 1,000 buildings, at most 8 comparisons per building |

## Strengths and weaknesses

| Strengths | Weaknesses |
|---|---|
| Fully readable rules | Unstable: small data changes can change the tree |
| Very fast prediction | Overfits without depth and leaf limits |
| Handles non-linear patterns and mixed feature scales | Axis-aligned splits only |
| Supports class weights | One tree is usually less accurate than an ensemble (our optional random forest scores 0.546 within events) |

# 04 · K-Nearest Neighbours (KNN), explained

## The idea in one line

"To label a new building, find the *k* most similar buildings whose labels we already know, and take a
majority vote." Cover and Hart introduced it in 1967. There is no real training: the model **stores** the
training data.

**Analogy:** estimating a house's price from the prices of the most similar houses you know.

## Step by step

1. **Store** all training buildings: 49,929 within-event, 42,425 cross-event. Each is a 37-number vector
   after preprocessing.
2. For a query building **x**, compute its **distance** to every stored building.
3. Take the **k** nearest.
4. **Vote:** the most common label wins.
5. **Confidence** = votes for the winner ÷ k ("vote fraction of neighbours").

## Distance

AfterSight uses the **Euclidean** distance (scikit-learn's default: Minkowski with p = 2) on the standardised
features:

> d(**x**, **x**ᵢ) = √( Σ_{j=1}^{37} (x_j − x_{ij})² )

**Small example with 2 features** (already standardised):

- query = (1.0, −2.0); A = (0.8, −1.5); B = (−0.5, 0.3).
- d(query, A) = √(0.2² + 0.5²) = √0.29 ≈ 0.54
- d(query, B) = √(1.5² + 2.3²) = √7.54 ≈ 2.75

So A is the nearer neighbour.

## Why scaling matters for KNN

Distances add up the squared differences of *all* features. Without scaling:

- `area_px` ranges from tens to 648,034 px;
- `glcm_homogeneity_post` lies between 0 and 1.

Area would decide every distance on its own. The shared pipeline therefore applies:

1. log(1 + x) to 7 long-tailed features;
2. standardisation, z = (x − mean_train) / std_train, using statistics from the **training split only**.

## Choosing k and the weighting

The grid comes from `config/training.yaml`. Each setting was fitted on train and scored on validation
macro-F1:

| k | weights | val macro-F1 (within) | val macro-F1 (cross) |
|---:|---|---:|---:|
| 5 | uniform | 0.514 | 0.565 |
| 5 | distance | 0.509 | 0.559 |
| **15** | **uniform** | **0.518** | **0.581** |
| 15 | distance | 0.510 | 0.578 |
| 31 | uniform | 0.499 | 0.577 |
| 31 | distance | 0.496 | 0.574 |

**Selected: k = 15, uniform votes, in both designs.**

- *Small k* (e.g. 1) copies noise from a single neighbour, which is high variance.
- *Large k* blurs class boundaries towards the majority class, which is high bias.
- *Distance weighting* lets closer neighbours count more. It didn't help here.

## Class imbalance and KNN

scikit-learn's `KNeighborsClassifier` has **no `class_weight` option**, and AfterSight trains it unweighted:
`KNeighborsClassifier(n_jobs=-1, **params)` in `backend/app/training/models.py`.

58% of buildings are undamaged, so many neighbourhoods are dominated by "no damage". In the within-event test
set **69.4% of KNN's predictions are "no damage"** (the true share is 58.1%).

**This is a protocol difference from the Decision Tree,** which uses balanced class weights. The paper
reports it as a threat to validity.

## Our results (test sets, evaluated once)

| | Within-event | Cross-event |
|---|---:|---:|
| macro-F1 [95% scene-bootstrap CI] | **0.520** [0.459, 0.560] | 0.271 [0.259, 0.284] |
| accuracy | 0.636 | 0.554 |
| recall: none / minor / major / destroyed | 0.83 / 0.27 / 0.25 / 0.66 | 0.81 / 0.07 / 0.03 / 0.35 |
| F1: none / minor / major / destroyed | 0.76 / 0.35 / 0.28 / 0.69 | 0.73 / 0.11 / 0.04 / 0.20 |

**How to read it:**

- KNN is excellent at "no damage" but misses most minor and major damage. In the within-event set, 61% of
  minor-damage buildings are predicted "no damage".
- It is the **best of the two within events**.
- It **drops the most on the unseen disaster** (−48% macro-F1). A plausible, *untested* reason: KNN relies
  on similarity to stored examples, and an unseen disaster looks different from all of them.

## How AfterSight explains a KNN prediction

`backend/app/explain/explain.py` → `knn_neighbours` returns the 15 training buildings that were nearest, with
their labels and distances. The Case File shows their post-disaster crops.

**Example:** `santa-rosa-wildfire_00000006_b0002`, a destroyed house.

- All 15 neighbours are destroyed buildings from other Santa Rosa scenes. The nearest is
  `santa-rosa-wildfire_00000007_b0047` at distance 1.447.
- Vote 15 / 15, so confidence = 1.00.

## Cost

| | Detail |
|---|---|
| Training | 0.19 s (it only stores and indexes the data) |
| Prediction | 19.7 ms per 1,000 test buildings on the M4 laptop, using all cores. That is about 70× slower than the DT, because every query is compared with ~50,000 stored buildings. |

## Strengths and weaknesses

| Strengths | Weaknesses |
|---|---|
| Simple; no assumptions about class shapes | Slow prediction on large training sets |
| Explanation by real examples is intuitive | Needs careful scaling |
| Handles non-linear boundaries naturally | Unweighted votes favour the majority class |
| Training is essentially free | Suffers when test data look unlike any stored data (domain shift) |

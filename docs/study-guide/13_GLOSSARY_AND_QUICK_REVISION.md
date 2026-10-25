# 13 · Glossary and quick revision

## Glossary

| Term | Meaning |
|---|---|
| **xBD** | public dataset of pre/post-disaster satellite pairs with building damage labels (Gupta *et al.*, 2019) |
| **Parquet** | compressed, column-based table file; our copy of xBD is 17 Parquet files |
| **Mask** | image whose pixels hold labels: footprint (0/1) or damage (0–4) |
| **Connected component / blob** | group of touching pixels; with 4-connectivity, touching through edges only |
| **Majority vote (label)** | the most frequent damage value inside a blob becomes its label |
| **Purity** | share of a blob's pixels that agree with its label |
| **Scene / image pair** | one 1024 × 1024 pre + post image of the same place |
| **Feature** | a number describing a building (37 per building) |
| **GLCM** | grey-level co-occurrence matrix: statistics of neighbouring grey levels (contrast, homogeneity) |
| **Canny edge** | pixel on a sharp brightness edge (Canny 1986), after Gaussian smoothing (σ = 1.5) |
| **Context ring** | ground pixels within 8 px of a building, excluding all buildings |
| **log1p** | log(1 + x); shrinks long tails |
| **Standardisation (z-score)** | (x − mean) / std, with training-split statistics |
| **Pipeline** | fixed chain: log1p → median imputation → scaling → model, fitted together on train |
| **Within-event** | new scenes of disasters seen in training |
| **Cross-event** | a disaster held out entirely (Hurricane Michael) |
| **Leakage** | information from validation or test reaching training; makes scores too optimistic |
| **Scene-grouped split** | every building of a scene goes to the same split |
| **Hyperparameter** | setting chosen before training (k, depth, leaf size), tuned on validation |
| **Class imbalance** | unequal class sizes (58% no damage) |
| **Balanced class weights** | w_c = N / (C · N_c), so rare classes count more when the DT chooses splits |
| **Gini impurity** | G = 1 − Σ p_c²; 0 = pure node |
| **Precision / recall / F1** | correct among predicted / found among actual / their harmonic mean |
| **Macro-F1** | unweighted mean of per-class F1; the main metric |
| **Confusion matrix** | rows = reference label, columns = prediction |
| **Bootstrap** | resampling with replacement to estimate uncertainty; we resample test scenes |
| **Confidence interval (95%)** | range from the 2.5th to the 97.5th percentile of bootstrap scores |
| **Domain shift** | test data systematically different from training data |
| **Confounding (event signature)** | features that reveal *which disaster*, which can inflate within-event scores |
| **Feature-set experiments** | reduced or normalised feature sets tested for the event signature (development name "Phase 3b"; files in `artifacts/experiments/phase3b/`). Saved results; no longer shown in the app |
| **K-Means / silhouette / ARI** | unsupervised clustering / cluster separation (−1…1) / agreement with labels (0 random, 1 identical) |
| **Calibration** | making scores match real probabilities; **not** done here |

## Formulas

- **Euclidean distance:** d(**x**, **x**ᵢ) = √Σⱼ (xⱼ − xᵢⱼ)²
- **Gini impurity:** G = 1 − Σ_c p_c²
- **Balanced weights:** w_c = N / (4 N_c)
- **F1:** 2PR / (P + R)
- **Macro-F1:** (1/4) Σ_c F1_c
- **Mean absolute grey change:** D = (1/|Ω|) Σ |g_post − g_pre|
- **Brightness ratio:** B = ḡ_post / max(ḡ_pre, 1)
- **GLCM contrast:** C = Σ P(i,j)(i − j)²
- **GLCM homogeneity:** H = Σ P(i,j)/(1 + (i − j)²)
- **Relative brightening:** K = (ḡ_post − r̄_post) − (ḡ_pre − r̄_pre)
- **Compactness:** 4πA / P²
- **Grey value:** g = 0.299R + 0.587G + 0.114B

## KNN vs Decision Tree at a glance

| | KNN | Decision Tree |
|---|---|---|
| How it predicts | majority vote of the 15 nearest training buildings (Euclidean, standardised) | yes/no questions to a leaf (depth ≤ 8, leaf ≥ 100) |
| Training | stores data (0.19 s) | grows the tree with Gini (≈ 1.2 s) |
| Prediction cost | 19.7 ms / 1,000 buildings | 0.27 ms / 1,000 |
| Needs scaling? | yes | no (but shares the pipeline) |
| Class imbalance | unweighted (no option in scikit-learn) | balanced class weights |
| Explanation | similar labelled buildings | rule path in raw feature units |
| Confidence | vote fraction | class share in the leaf |
| Within-event macro-F1 | **0.520** [0.459, 0.560] | 0.470 [0.401, 0.526] |
| Cross-event macro-F1 | 0.271 [0.259, 0.284] | **0.294** [0.284, 0.304] |
| Accuracy (within / cross) | **0.636 / 0.554** | 0.484 / 0.502 |
| Strength | "no damage", destroyed | minority classes (minor, major) |
| Drop on the unseen event | −48% | −37% |

## Last-minute numbers

- **Scenes:** 10 events / 2,799 pairs in the copy; we use 5 events, 1,239 pairs → **1,179 with buildings**
  (52 no blobs, 8 only label-less blobs).
- **Buildings:** **71,340**: none 41,449 · minor 11,766 · major 10,270 · destroyed 7,855 (58% none).
- **Splits:** within-event 821 / 181 / 177 scenes; cross-event 716 / 130 / **333** (Michael, 21,429 buildings).
- **Features:** **37** = colour 12 + change 8 + texture 12 + context 3 + shape 2.
- **Event-prediction accuracy:** 0.829 (vs 0.252 majority).
- **Bootstrap:** 2,000 scene resamples, seed 42. KNN − DT: **+0.050** [+0.013, +0.084] within;
  **−0.023** [−0.034, −0.011] cross.

## Five sentences to remember

1. Buildings come from annotation masks; nothing is detected.
2. Whole scenes go to one split; preprocessing is fitted on train; test is used once.
3. KNN wins within events, the DT wins slightly on the unseen disaster, and accuracy prefers KNN, so the
   metric and the design matter.
4. The DT is class-weighted and KNN is not; that's a protocol difference, reported as a threat to validity.
5. One held-out event, one training run, uncalibrated confidences, and no operational validation.

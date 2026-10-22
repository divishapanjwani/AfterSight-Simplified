# Phase 3 results (classical ML, test sets evaluated once)

Copied from `artifacts/metrics/*/*/comparison.json`, written by `python scripts/train.py` (seed 42,
scikit-learn 1.9.1). Reprint anytime with `python scripts/evaluate.py`. Macro-F1 is the primary metric.
Recall is per class on the test set. Inference time is in milliseconds per 1,000 buildings (KNN and the
random forest use all CPU cores).

## Within-event generalisation · 4-class (primary)

Test: 10,706 buildings from unseen scenes of the same 5 disasters.

| Model | val macro-F1 | **test macro-F1** | accuracy | R no-dmg | R minor | R major | R destroyed | train s | ms / 1k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline (`glcm_homogeneity_delta` thresholds) | 0.332 | 0.383 | 0.584 | 0.876 | 0.109 | 0.057 | 0.443 | 0.8 | 0.0 |
| Logistic Regression (C=10) | 0.538 | **0.527** | 0.552 | 0.525 | 0.461 | 0.566 | 0.817 | 1.1 | 0.2 |
| KNN (k=15, uniform) | 0.518 | 0.520 | 0.636 | 0.831 | 0.270 | 0.254 | 0.658 | 0.2 | 19.7 |
| Decision Tree (depth 8, leaf 100) | 0.511 | 0.470 | 0.484 | 0.434 | 0.449 | 0.499 | 0.780 | 1.2 | 0.3 |
| Random Forest — *optional reference* | 0.589 | 0.546 | 0.590 | 0.617 | 0.486 | 0.430 | 0.811 | 4.7 | 3.9 |

## Cross-event generalisation · 4-class (primary)

Test: all 21,429 hurricane-michael buildings. The model never saw this disaster.

| Model | val macro-F1 | **test macro-F1** | accuracy | R no-dmg | R minor | R major | R destroyed | train s | ms / 1k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline (`abs_dmean_r` thresholds) | 0.362 | 0.193 | 0.245 | 0.264 | 0.153 | 0.260 | 0.461 | 1.0 | 0.0 |
| Logistic Regression (C=10) | 0.665 | 0.290 | 0.433 | 0.561 | 0.179 | 0.153 | 0.447 | 0.9 | 0.2 |
| KNN (k=15, uniform) | 0.581 | 0.271 | 0.554 | 0.808 | 0.071 | 0.033 | 0.354 | 0.2 | 16.8 |
| Decision Tree (depth 8, leaf 100) | 0.576 | **0.294** | 0.502 | 0.663 | 0.241 | 0.024 | 0.428 | 1.0 | 0.2 |
| Random Forest — *optional reference* | 0.639 | 0.291 | 0.606 | 0.888 | 0.069 | 0.008 | 0.431 | 4.7 | 3.3 |

## 3-class experiment (no-damage / damaged / destroyed)

A separate experiment; the 4-class task above stays primary.

| Model | within-event test macro-F1 | cross-event test macro-F1 |
|---|---:|---:|
| Baseline | 0.373 | 0.334 |
| Logistic Regression | 0.612 | 0.413 |
| KNN | 0.634 | 0.379 |
| Decision Tree | 0.558 | 0.371 |
| Random Forest (optional) | 0.640 | 0.414 |

## Event-confounding check (analysis only)

Logistic regression predicting the **event** from the same 37 features, on the within-event split.

- Test accuracy: **0.829** (macro-F1 0.819).
- Majority class: 0.252. Uniform chance over 5 events: 0.200.
- Event is never an input to any damage model.

**Per-event damage scores.** Within-event test, LR: macro-F1 0.33 (harvey), 0.33 (matthew),
0.36 (michael), 0.53 (palu), 0.46 (santa-rosa). Every event scores **below** the pooled 0.527.
Part of the pooled score comes from telling events apart, not from reading damage within an event.

## K-Means (analysis only, never predictions)

- **Training buildings, within-event split:** silhouette is highest at k=2 (0.253) and falls steadily
  as k grows. The elbow heuristic suggests k=3.
- **Agreement with labels:** adjusted Rand index 0.041 at k=2 and 0.095 at k=4. The clusters barely
  line up with damage classes. They mostly reflect other structure in the features: roof type, scene
  appearance and event.

## Figures

- Confusion matrices: `outputs/figures/models/<design>/<scheme>/<model>_confusion_{val,test}.png`
- Event-confounding: `outputs/figures/models/confounding/within_event_event_confusion_test.png`
- K-Means: `outputs/figures/models/kmeans/<design>/kmeans_elbow_silhouette.png`, `kmeans_contingency_k{2,4}.png`

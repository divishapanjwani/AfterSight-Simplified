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

---

# Phase 3b experiment: confounding-reduced feature sets

*A separate experiment. These are new Logistic Regression and Decision Tree models, each tuned on
validation and evaluated on test once, stored under `artifacts/experiments/phase3b/`. The Phase 3
models were not replaced or re-tested. The Phase 3 rows below are copied from the tables above
for reference.*

Variants (`backend/app/features/variants.py`):

- **change_only:** 17 features: the change group, the 4 texture deltas, the context ring and shape.
  Every absolute pre/post colour and texture level is removed.
- **scene_normalised:** all 37 features, each image feature minus its **scene median**. A difference
  rather than a ratio, because many features are signed or near zero. Shape stays raw. Only feature
  values are used, never labels, and scenes never cross splits, so this is leak-free.
- **change_only_scene_normalised:** both of the above.

Event-prediction accuracy is the confounding classifier retrained on each variant's features
(within-event test; chance 0.20, majority class 0.25). Lower means less confounding.

| Variant (features) | Design | Model | test macro-F1 | R major | R destroyed | event-pred. acc. |
|---|---|---|---:|---:|---:|---:|
| *Phase 3 full (37)* | within | LR | *0.527* | *0.566* | *0.817* | *0.829* |
| *Phase 3 full (37)* | within | DT | *0.470* | *0.499* | *0.780* | *0.829* |
| *Phase 3 full (37)* | cross | LR | *0.290* | *0.153* | *0.447* | |
| *Phase 3 full (37)* | cross | DT | *0.294* | *0.024* | *0.428* | |
| change_only (17) | within | LR | 0.450 | 0.501 | 0.764 | 0.632 |
| change_only (17) | within | DT | 0.436 | 0.442 | 0.717 | 0.632 |
| change_only (17) | cross | LR | 0.243 | 0.087 | 0.579 | |
| change_only (17) | cross | DT | 0.278 | 0.022 | 0.314 | |
| scene_normalised (37) | within | LR | 0.330 | 0.511 | 0.408 | 0.518 |
| scene_normalised (37) | within | DT | 0.342 | 0.438 | 0.616 | 0.518 |
| scene_normalised (37) | cross | LR | 0.236 | 0.200 | 0.804 | |
| scene_normalised (37) | cross | DT | 0.241 | 0.168 | 0.506 | |
| change_only + scene_normalised (17) | within | LR | 0.309 | 0.388 | 0.431 | 0.374 |
| change_only + scene_normalised (17) | within | DT | 0.374 | 0.129 | 0.523 | 0.374 |
| change_only + scene_normalised (17) | cross | LR | 0.248 | 0.330 | 0.711 | |
| change_only + scene_normalised (17) | cross | DT | 0.262 | 0.188 | 0.464 | |

**Mean per-event within-event test macro-F1 (LR):** full 0.403 · change_only 0.345 · scene_normalised 0.260 ·
combined 0.252.

**Interpretation.** Removing absolute levels and normalising by scene does cut the event signature
sharply: event prediction falls from 0.83 to 0.37 with both changes. However, **no variant improves
cross-event macro-F1** (0.24–0.28 vs 0.29), and within-event scores fall by 0.08–0.22. Per-event scores
fall too, so the variants removed real damage information, not only the "which disaster" shortcut.
Scene normalisation is hit hardest: where most of a neighbourhood is damaged (Santa Rosa: 0.46 → 0.22),
the scene median itself looks damaged, which hides exactly the signal we need. The poor transfer to
hurricane-michael is therefore not mainly the model exploiting event identity. It is a genuine domain
shift between disasters (different roofs, imagery and damage appearance, and a minor-heavy class mix)
that these hand-crafted features cannot bridge.

**Decision.** Keep the full 37-feature set (Phase 3 models) as the app default. No variant meets the
"better cross-event without hurting within-event" bar.

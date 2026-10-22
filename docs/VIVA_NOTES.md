# AfterSight viva notes

_The examiner Q&A and the 5-minute demo script are written in Phase 7. This file currently holds the demo scenes chosen in Phase 6._

## Demo scenes (pre-cached, work offline)

`python scripts/prepare_demo.py` chose these from the **held-out test split**. Each has 25–160 buildings and at least 3 ground-truth
damage classes, ranked by how evenly the classes are mixed, with the best scene per disaster type.
For every model, the analysis, the view images, the CSV and the PDF are already on disk, so the demo needs no computation.
The table below is generated from `artifacts/demo/demo_scenes.json`; re-run the script after retraining.

| # | Scene | Disaster type (config) | Buildings | Ground truth: none · minor · major · destroyed | Scene macro-F1: LR · KNN · DT · RF |
|---|---|---|---:|---|---|
| 1 | `hurricane-matthew_00000185` | wind | 37 | 12 · 10 · 8 · 7 | 0.22 · 0.35 · 0.16 · 0.20 |
| 2 | `hurricane-harvey_00000356` | flooding | 115 | 25 · 61 · 27 · 2 | 0.21 · 0.17 · 0.26 · 0.23 |
| 3 | `santa-rosa-wildfire_00000006` | fire | 36 | 14 · 1 · 2 · 19 | 0.73 · 0.55 · 0.58 · 0.78 |

Open one directly: `http://localhost:5200/triage/<scene id>?model=logistic_regression&run=1`.

Talking points:

- **santa-rosa-wildfire_00000006** (fire) is the success case: burned roofs change texture strongly, and *destroyed* is the class the models find easiest.
- **hurricane-matthew_00000185** (wind) has an even mix of all four classes. Minor and major damage get confused, which matches the confusion matrices.
- **hurricane-harvey_00000356** (flooding) is mostly *minor damage* under flood water. The roof often looks unchanged from above, and the models struggle.
- All three are **test** scenes, so no provenance warning appears. To show the warning, open any training scene from the Library (split = Training).
- Single-scene scores are small samples. Quote the Arena's test-set numbers for overall performance.

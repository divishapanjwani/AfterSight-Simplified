# Research plan: KNN vs Decision Tree for building-damage classification (Deliverable 2)

## Research question

> How do a K-Nearest Neighbours (KNN) classifier and a Decision Tree (DT) classifier, trained on 37 engineered
> pre/post-event image features of annotation-derived building regions, compare for four-class building
> damage classification, and how does that comparison change between a scene-grouped **within-event**
> evaluation and a **cross-event** evaluation that holds out one disaster (Hurricane Michael)?

The question is the brief's starting point with two refinements, both justified by the code:

- "annotation-derived building regions": footprints come from masks; nothing is detected.
- "holds out one disaster": the cross-event result rests on one event only.

## Objectives

1. Describe a leakage-controlled protocol: scene-grouped splits, preprocessing fitted on training data only,
   tuning on validation macro-F1, one test evaluation per model.
2. Compare KNN and DT on macro-F1, per-class precision/recall/F1, confusion matrices and accuracy, under
   both designs.
3. Quantify uncertainty with a **scene-level bootstrap**: 95% confidence intervals for each macro-F1 and for
   the paired KNN − DT difference, computed from the saved test predictions without retraining.
4. Report the protocol differences that limit the comparison: DT is class-weighted, KNN is not; the grid
   sizes differ (6 vs 8).
5. Interpret the gap between the two designs using only measured evidence (class-mix shift, the event
   signature in the features), with explanations labelled as hypotheses.

**Hypotheses.** The results already exist, so the paper will **not** present hypotheses as if they were
stated before the experiments. It poses research questions instead, and labels every causal explanation in
the discussion as a hypothesis.

## Why this comparison is useful

- **KNN** is non-parametric and instance-based. It is sensitive to feature scaling and to the class balance
  among neighbours. Its explanation is a set of similar labelled buildings. Prediction cost grows with the
  training set.
- **DT** learns axis-aligned splits. It is invariant to per-feature monotone scaling and supports class
  weighting. Its explanation is an explicit rule path. Prediction is cheap.

Both kinds of explanation are already implemented in the application. So for an explainable triage tool,
the trade-off between accuracy under distribution shift and the type of explanation is a practical
question. These are algorithm properties, cited. The measured findings are kept separate.

## Literature themes (every source to be checked via DOI or publisher page before use)

1. **xBD and the Joint Damage Scale:** Gupta et al. (CVPR Workshops 2019 / arXiv:1911.09296).
2. **Building damage assessment from satellite imagery:** xView2 challenge-era deep models (e.g. Siamese
   and change-detection networks), as the context we deliberately do not use.
3. **Classical and object-based change detection in remote sensing:** pre/post differencing, texture.
4. **Texture and edge features:** Haralick GLCM (1973); Canny (1986).
5. **Algorithms:** KNN (Cover & Hart 1967); CART (Breiman et al. 1984); scikit-learn (Pedregosa et al. 2011).
6. **Class imbalance and macro-averaged metrics.**
7. **Domain shift / cross-event generalisation in damage assessment:** works reporting a drop on unseen
   disasters.
8. **Evaluation uncertainty:** bootstrap confidence intervals (Efron & Tibshirani), resampling clusters
   (scenes) rather than correlated items.

## Defensible contribution (proportional to the evidence)

| Kind | What we can claim |
|---|---|
| Application | An explainable, leakage-guarded building-damage triage tool on a public dataset, with per-building KNN-neighbour and DT-path explanations. |
| Experimental design | One feature representation and pipeline evaluated under two designs on the same data. Reporting includes per-class results, scene-level bootstrap CIs and the protocol differences. |
| Findings (measured) | The verified macro-F1 and accuracy values, the per-class patterns, and the protocol-dependence of the ranking. Significance only if the bootstrap supports it. |
| Adopted methods | xBD labels, GLCM and Canny features, KNN, CART, stratified group splitting, macro-F1 and the bootstrap: all from prior work and cited. |

It is **not**:

- a novel algorithm;
- a new dataset;
- state of the art;
- evidence about all unseen disasters (one held-out event).

## Section outline (provisional IEEE conference format)

Targets: about 6 pages (venue not yet chosen), abstract about 200 words.

- **Abstract; Index Terms.**
- **I. Introduction:** the motivation (rapid triage) and why classical, explainable models; the research
  questions; the contributions, framed as an applied comparative study.
- **II. Related Work:** xBD and deep approaches; classical change features; KNN/DT in remote sensing;
  domain shift; evaluation practice.
- **III. Dataset and Problem Formulation:**
  - the xBD train-split mirror and the five events;
  - mask-derived buildings: 4-connectivity, majority label, drops, mixed blobs;
  - class distribution;
  - the 4-class task definition.
- **IV. Features and Classifiers:**
  - the 37 features in five groups, with equations for the key ones (taken from `extract.py`);
  - the shared pipeline;
  - KNN (distance equation, k, weights);
  - DT (Gini, depth, leaf size, class weights).
- **V. Experimental Setup:**
  - the two split designs with counts;
  - tuning grids and selection;
  - the single test evaluation;
  - metrics (macro-F1 defined);
  - the bootstrap method;
  - the protocol differences;
  - hardware and software.
- **VI. Results and Discussion:**
  - macro-F1 and accuracy with CIs;
  - per-class results and confusion matrices;
  - the ranking change across designs;
  - accuracy vs macro-F1;
  - the hardest classes;
  - the class-mix shift and the event signature as candidate (hypothesised) explanations;
  - inference cost;
  - the explanation types.
- **VII. Limitations and Threats to Validity:**
  - internal: unweighted KNN vs class-weighted DT, unequal grids, a single run;
  - construct: mask-derived merged buildings and majority labels;
  - external: one held-out event, 5 of 10 events selected by a design decision, the third-party mirror,
    train split only;
  - measurement: macro-F1 instability on rare classes; uncalibrated confidences;
  - no operational validation.
- **VIII. Conclusion and Future Work.**
- **Acknowledgements:** includes an AI-assisted writing disclosure placeholder.
- **References.**

## Proposed tables and figures (all generated from artifacts by scripts in `paper/build/`)

| ID | Content | Source |
|---|---|---|
| Table I | Dataset: buildings per event × class, scenes per event | `inspection_report.json` |
| Table II | Split sizes (scenes, buildings, class mix) for both designs | split manifests |
| Table III | The 37 features by group | `schema.py` |
| Table IV | Configuration: pipeline, grids, selected hyperparameters, class weighting, search sizes | `make_pipeline`, model metadata |
| Table V | Test macro-F1 (95% bootstrap CI), accuracy, weighted-F1; KNN vs DT, both designs; plus one greyed context row for LR, RF and the baseline | model metadata + `paper_stats.py` |
| Table VI | Per-class precision / recall / F1 for both models and designs | model metadata |
| Fig. 1 | Pipeline overview (implemented components only) | drawn from code |
| Fig. 2 | Macro-F1 with CIs, within vs cross, KNN vs DT | `paper_stats.py` output |
| Fig. 3 | Row-normalised confusion matrices (2 models × 2 designs) | model metadata |
| Fig. 4 *(optional, if space)* | Top of the actual saved tree (depth ≤ 2) | `decision_tree_structure.json` |

## Extra analysis (inference and resampling only)

`scripts/paper_stats.py`:

- **Inputs:** the saved per-building test predictions, which are already verified to equal fresh inference
  from the saved models.
- **Per-model CIs:** for each model and design, resample **test scenes** with replacement (1,000+ resamples,
  fixed seed) and recompute macro-F1 → 95% percentile CI.
- **Paired difference:** the same resampled scenes for KNN and DT → a CI for the macro-F1 difference
  (KNN − DT). We report whether the CI excludes 0; we do not call it a significance test unless framed
  properly.
- **Safety check:** `snapshot_behaviour.py` must stay IDENTICAL afterwards.

## Template and outputs

- I'll try the official IEEE conference Word template first.
- If the download is blocked, I'll build a clearly labelled *provisional* two-column `.docx` with
  python-docx (Times New Roman 10 pt, US Letter), and you should move the text into the official template
  before submitting.
- Outputs go in `paper/` on branch `paper`; nothing is merged or pushed without approval.

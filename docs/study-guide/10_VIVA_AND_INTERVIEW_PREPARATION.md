# 10 · Viva and interview preparation (64 questions)

The questions run from basic to advanced. Each has a **short answer** (say this first), a **detailed
explanation** (if they ask "why?") and **likely follow-ups**. All numbers match the paper and the saved
results.

---

## A. Project basics (Q1–Q12)

**Q1. What does AfterSight do?**
- **Short:** It classifies the damage of each building in a pre/post satellite image pair into four levels
  and explains each prediction.
- **Detailed:** The levels are no damage, minor, major and destroyed. Buildings come from xBD annotation
  masks. Two classical ML models, KNN and a Decision Tree, predict from 37 engineered
  features, and the web app shows predictions with explanations.
- **Follow-ups:** Who would use it? (A damage analyst, as a first pass. It is a research and teaching tool,
  not operational.)

**Q2. What is your research question?**
- **Short:** How do KNN and a Decision Tree compare, and how does that change when the test disaster was
  never seen in training?
- **Detailed:** RQ1 is within-event (new scenes of known disasters), RQ2 is cross-event (Hurricane Michael
  held out), and RQ3 asks which classes cause the errors.
- **Follow-ups:** Why these two models? (Q13)

**Q3. What dataset did you use?**
- **Short:** A third-party Parquet copy of xBD's training split; five events, 71,340 buildings.
- **Detailed:** The copy has 10 events and 2,799 pairs. We used Harvey, Matthew, Michael, Palu and Santa Rosa:
  1,239 pairs, of which 1,179 contain labelled buildings.
- **Follow-ups:** Did you use xBD's test set? (No. It isn't in the copy; all splits come from the training
  split.)

**Q4. What are the four classes and how imbalanced are they?**
- **Short:** No damage 41,449, minor 11,766, major 10,270, destroyed 7,855: 58% undamaged.
- **Detailed:** The labels come from the damage mask values 1–4. The imbalance is why we use macro-F1 and why
  class weighting matters.
- **Follow-ups:** What would a model that always says "no damage" score? (Accuracy 0.581 within, but
  macro-F1 0.184.)

**Q5. Do you detect buildings?**
- **Short:** No. Buildings are reconstructed from xBD's annotation masks.
- **Detailed:** Each 4-connected blob of the footprint mask becomes a building, labelled by majority vote of
  the damage mask. We never claim building detection.
- **Follow-ups:** What goes wrong? (Touching buildings merge: 782 mixed blobs.)

**Q6. What is a "scene"?**
- **Short:** One 1024 × 1024 pre + post image pair of the same place.
- **Detailed:** Splits are made per scene. The building ID is `<scene>_bNNNN`, e.g.
  `santa-rosa-wildfire_00000006_b0002`.
- **Follow-ups:** Why do 1,239 pairs become 1,179 split scenes? (Q22)

**Q7. Why classical ML and not deep learning?**
- **Short:** Explainability, the course scope, and laptop-scale compute. Deep learning is excluded by design.
- **Detailed:** Every prediction can be traced to features, neighbours or rules. The price is lower accuracy
  than published CNNs on xBD.
- **Follow-ups:** Would a CNN do better? (Likely within events. Transfer across disasters is also hard for
  CNNs: Valentijn *et al.* found performance varied strongly between disasters.)

**Q8. What does the app consist of?**
- **Short:** A FastAPI backend serving saved scikit-learn pipelines from files, and a React frontend.
- **Detailed:**
  - Pages: Readiness, Library, Triage Desk, Case File, Model Arena, Feature Lab, Cluster Atlas, Review
    Queue, Reports.
  - Ports 8020 and 5200.
  - There is no database.
- **Follow-ups:** Why no database? (Read-mostly files written by the pipeline; simple and versionable.)

**Q9. Does the app train models?**
- **Short:** No. Analysis only loads saved models.
- **Detailed:** Training happens in `scripts/train.py`. There is a guarded training API
  (`POST /training/run` with `confirm: true`), but the UI doesn't use it.
- **Follow-ups:** How fast is an analysis? (Milliseconds; results are cached per scene and model.)

**Q10. What is a feature here? Give an example.**
- **Short:** A number describing a building, e.g. `glcm_homogeneity_delta`, the change in texture uniformity.
- **Detailed:** For the destroyed demo house it is −0.240: the roof became much less uniform.
- **Follow-ups:** How many features? (37: colour 12, change 8, texture 12, context 3, shape 2.)

**Q11. What are your headline results?**
- **Short:** KNN 0.520 vs DT 0.470 macro-F1 within events; 0.271 vs 0.294 on held-out Michael.
- **Detailed:** Accuracy is 0.636 vs 0.484 within, and 0.554 vs 0.502 cross. Each model was evaluated on its
  test set once.
- **Follow-ups:** Which is better then? (Q42)

**Q12. What is your contribution?**
- **Short:** An applied comparative study with a leakage-controlled two-design protocol, honest uncertainty
  and reported protocol differences, plus a working explainable app.
- **Detailed:** No new algorithm or dataset, and no state-of-the-art claim.
- **Follow-ups:** What is novel then? (The careful comparison and reporting, not the methods.)

## B. Data, features and leakage (Q13–Q26)

**Q13. Why compare KNN and a Decision Tree specifically?**
- **Short:** Both are classical and explainable, but explain differently: by examples vs by rules.
- **Detailed:** For a triage tool, the type of explanation matters. They also differ in scaling needs, cost
  and class handling.
- **Follow-ups:** Where do LR and RF fit? (Neither is a damage model of the app. RF is a saved reference, 0.546/0.291, not shown in the app; a logistic regression is used only as the confounding check's analysis classifier.)

**Q14. How is a building's label decided?**
- **Short:** By majority vote of the damage-mask values inside the blob.
- **Detailed:** Ties (1), blobs without a damage value (1,396) and blobs under 16 px (54) are dropped and
  logged, never guessed.
- **Follow-ups:** Why not drop mixed blobs? (781 of 782 are kept with the majority label. That's disclosed as a
  limitation.)

**Q15. Why 4-connectivity?**
- **Short:** It merges fewer touching buildings than 8-connectivity.
- **Detailed:** With 8-connectivity, two roofs touching only at a corner would merge into one blob.
- **Follow-ups:** Does it prevent all merges? (No: buildings sharing an edge still merge.)

**Q16. How do you compare pre and post images?**
- **Short:** Change features on the same footprint pixels, before vs after.
- **Detailed:** The absolute colour change, the mean absolute grey change *D*, the brightness ratio *B*, and
  the texture deltas (post − pre).
- **Follow-ups:** What about misalignment? (168 pairs are flagged by the alignment check and kept; this is a
  limitation.)

**Q17. What is GLCM homogeneity?**
- **Short:** How similar neighbouring pixels are: Σ P(i,j)/(1 + (i − j)²), 1 = identical.
- **Detailed:** Grey is quantised to 32 levels; horizontal and vertical neighbour pairs inside the footprint
  are used. It is the most separating feature (η² = 0.185 post, 0.136 delta).
- **Follow-ups:** And contrast? (Σ P(i,j)(i − j)²; large when neighbours differ.)

**Q18. What is the context ring for?**
- **Short:** It separates "the building changed" from "everything changed".
- **Detailed:** The ring is ground within 8 px, excluding all buildings. *K* = (ḡ_post − r̄_post) −
  (ḡ_pre − r̄_pre) measures relative change.
- **Follow-ups:** What if the ring is empty? (Features would be NaN, imputed by the training median; none are
  missing in our data.)

**Q19. Why log-transform some features?**
- **Short:** 7 features have a very long tail (training skewness > 2), e.g. area up to 648,034 px.
- **Detailed:** log(1 + x) stops extreme values dominating distances. The decision is fitted on the training
  split.
- **Follow-ups:** Which features? (abs_dmean_r/g/b, gray_absdiff_mean, glcm_contrast_pre/post, area_px.)

**Q20. Why scale features?**
- **Short:** KNN's distance would otherwise be dominated by large-range features.
- **Detailed:** z = (x − mean_train)/std_train. The DT doesn't need it, but uses the same pipeline.
- **Follow-ups:** Why fit on train only? (Fitting on all data leaks test statistics into training.)

**Q21. How do you prevent data leakage?**
- **Short:** Scene-grouped splits, preprocessing fitted on train only, tuning on validation, test used once.
- **Detailed:**
  - `validate_manifest` rejects any scene in two splits;
  - manifests are hashed;
  - `TestSetGuard` forbids a second test evaluation.
- **Follow-ups:** Why group by scene? (Buildings in one image share lighting and damage; splitting them
  inflates scores.)

**Q22. Why 1,239 pairs but 1,179 scenes in the splits?**
- **Short:** 52 pairs have no building blob, and 8 have only blobs without a damage value.
- **Detailed:** Verified from `scenes.csv` and the drop reasons (all 28 blobs `label_missing`). The 1,179 are
  exactly the scenes in both manifests.
- **Follow-ups:** And Michael's 343 vs 333? (10 Michael pairs have no labelled building.)

**Q23. Describe the two split designs.**
- **Short:** Within-event: 821/181/177 scenes across all 5 events. Cross-event: all 333 Michael scenes are
  the test set; train and validation (716/130) come from the other 4 events.
- **Detailed:** Within-event uses `StratifiedGroupKFold`, 20 folds, seed 42, combined to about 70/15/15.
- **Follow-ups:** Why Michael? (A large event, 21,429 buildings, with all four classes.)

**Q24. Why is within-event evaluation optimistic?**
- **Short:** Test scenes share the satellite, season, region and building style with training scenes.
- **Detailed:** The features identify the disaster with 0.829 accuracy; a model can partly rely on that.
- **Follow-ups:** How do you know? (The confounding check, and Michael's within-event score of 0.367 vs 0.271
  cross for KNN.)

**Q25. How were the five events chosen?**
- **Short:** By design decision, for four hazard types with many severely damaged buildings.
- **Detailed:** It isn't a formal rule (Santa Rosa has only 55 major-damage buildings). This is reported as a
  threat to validity.
- **Follow-ups:** Would other events change the results? (Possibly; that's future work.)

**Q26. What happened to the "un-classified" buildings?**
- **Short:** Blobs with no damage value (1,396) are excluded and counted, never given a class.
- **Detailed:** Likely xBD's "un-classified" label. Guessing would invent labels.
- **Follow-ups:** Bias? (Possible if they differ systematically; disclosed.)

## C. KNN (Q27–Q34)

**Q27. How does KNN predict?**
- **Short:** The majority label of the 15 nearest training buildings in standardised feature space.
- **Detailed:** Euclidean distance over 37 features; uniform votes.
- **Follow-ups:** Is there training? (Only storing and indexing: 0.19 s.)

**Q28. Write the distance formula.**
- **Short:** d(**x**, **x**ᵢ) = √Σⱼ(xⱼ − xᵢⱼ)².
- **Detailed:** This is Minkowski with p = 2, scikit-learn's default.
- **Follow-ups:** Alternatives? (Manhattan, p = 1; or cosine.)

**Q29. How was k chosen?**
- **Short:** Grid {5, 15, 31} × {uniform, distance} on validation macro-F1; k = 15 uniform won in both designs.
- **Detailed:** Validation scores: 0.518 within, 0.581 cross.
- **Follow-ups:** Small vs large k? (Small: noisy, high variance. Large: favours the majority, high bias.)

**Q30. What does KNN's confidence mean?**
- **Short:** The vote fraction, e.g. 15/15 = 1.00.
- **Detailed:** It is not a calibrated probability.
- **Follow-ups:** Is it used anywhere? (Review flags: confidence below the validation threshold of 0.53.)

**Q31. Why does KNN favour "no damage"?**
- **Short:** Unweighted votes, and 58% of neighbours tend to be undamaged.
- **Detailed:** 69.4% of its within-event predictions are "no damage" (true share 58.1%). It calls 61% of
  minor-damage buildings "no damage".
- **Follow-ups:** Can you weight KNN by class? (Not in scikit-learn's classifier. You could resample or
  undersample the training set; that's future work.)

**Q32. How is a KNN prediction explained?**
- **Short:** Its nearest labelled training buildings, with crops and distances.
- **Detailed:** For `…_00000006_b0002`, all 15 neighbours are destroyed Santa Rosa buildings; the nearest is
  at distance 1.447.
- **Follow-ups:** Why only training buildings? (Labels from validation or test would leak.)

**Q33. KNN's weaknesses here?**
- **Short:** Slow prediction, scaling-dependent, majority bias, and suffers under domain shift.
- **Detailed:** 19.7 ms per 1,000 buildings vs 0.27 for the DT; −48% macro-F1 on the unseen event.
- **Follow-ups:** How would you speed it up? (KD-trees or ball trees, approximate neighbours, a smaller
  reference set.)

**Q34. Why might KNN drop more on a new disaster?**
- **Short:** Hypothesis: it relies on similarity to stored examples, and a new disaster looks unlike them.
- **Detailed:** This is not tested here; it is consistent with the event signature.
- **Follow-ups:** How would you test it? (Leave-one-event-out, and domain-adaptation experiments.)

## D. Decision Tree (Q35–Q41)

**Q35. How does a decision tree split?**
- **Short:** It picks the feature and threshold that make the children purest by Gini impurity.
- **Detailed:** G = 1 − Σ p_c²; repeat recursively until the depth or leaf limit is reached.
- **Follow-ups:** Entropy instead? (Similar results in practice; Gini is scikit-learn's default.)

**Q36. Compute Gini for a 50/50 node.**
- **Short:** 1 − (0.5² + 0.5²) = 0.5.
- **Detailed:** A pure node is 0; four equal classes give 0.75.
- **Follow-ups:** What makes a split good? (A large drop in weighted child impurity.)

**Q37. What are balanced class weights?**
- **Short:** w_c = N/(4N_c): rare classes count more.
- **Detailed:** Within-event: no damage 0.43, minor 1.52, major 1.74, destroyed 2.27.
- **Follow-ups:** Effect? (The DT predicts only 32% "no damage"; minority recall rises, no-damage recall falls.)

**Q38. How did you stop overfitting?**
- **Short:** Max depth and minimum leaf size, tuned on validation: depth 8, leaf ≥ 100.
- **Detailed:** Depth 12 scored lower on validation, which is a sign of overfitting. No cost-complexity
  pruning was used.
- **Follow-ups:** How many leaves? (115 within-event, 111 cross-event.)

**Q39. What is the first question the tree asks?**
- **Short:** "`glcm_homogeneity_delta` ≤ −0.146?", i.e. did texture uniformity drop a lot?
- **Detailed:** Its children both split on `glcm_homogeneity_post`. Texture dominates, matching the Feature
  Lab.
- **Follow-ups:** Is that the most important feature overall? (It is the single most useful *split*;
  permutation importance is a different measure.)

**Q40. How is a DT prediction explained?**
- **Short:** The root-to-leaf path in raw feature units, and the leaf's class shares.
- **Detailed:** For the demo house: 8 questions, leaf of 971 buildings, 99.8% destroyed.
- **Follow-ups:** Why raw units? (Thresholds are converted back through the scaler and log1p, so they're
  readable.)

**Q41. Does the tree need scaling?**
- **Short:** No. Splits are invariant to monotone rescaling of a feature.
- **Detailed:** It shares the pipeline so both models get identical input.
- **Follow-ups:** Then why is it in the pipeline? (Simplicity, and the same imputation.)

## E. Evaluation and results (Q42–Q55)

**Q42. Which model is better?**
- **Short:** It depends. KNN on known disasters (0.520 vs 0.470), the DT slightly on the unseen one (0.294
  vs 0.271), and KNN on accuracy in both designs.
- **Detailed:** That is the paper's main message: the conclusion depends on the design and the metric.
- **Follow-ups:** Which would you deploy? (Neither for real decisions. For a known region, KNN; for a new
  event, the evidence slightly favours the DT, but only one event was tested.)

**Q43. Define precision, recall and F1, with an example.**
- **Short:** For KNN within, destroyed: precision 775/1,056 = 0.734, recall 775/1,178 = 0.658, F1 = 0.694.
- **Detailed:** F1 is the harmonic mean, 2PR/(P + R).
- **Follow-ups:** Why the harmonic mean? (It punishes a low value of either.)

**Q44. Why macro-F1?**
- **Short:** Every class counts equally, so imbalance can't hide failures.
- **Detailed:** "Always no damage" gets 0.581 accuracy but 0.184 macro-F1.
- **Follow-ups:** Weighted-F1? (Weights classes by support; it is closer to accuracy. We report it in the
  artifacts, not as the main metric.)

**Q45. Why do accuracy and macro-F1 disagree cross-event?**
- **Short:** KNN predicts "no damage" a lot (78% cross-event), which is right for the majority but misses
  minor damage.
- **Detailed:** The DT's minor-damage F1 is 0.26 vs KNN's 0.11, which raises its macro-F1.
- **Follow-ups:** Which metric matters for triage? (Finding damaged buildings, so macro-F1 and recall.)

**Q46. Read your confusion matrix.**
- **Short:** Rows are the reference label, columns the prediction, the diagonal is correct.
- **Detailed:** KNN within: 1,071 minor and 911 major buildings were predicted "no damage".
- **Follow-ups:** Which class is easiest? (Destroyed, among damage classes.)

**Q47. Which classes are hardest?**
- **Short:** Minor and major, especially major cross-event (F1 0.04 / 0.03).
- **Detailed:** The difference is subtle at sub-metre resolution, and Michael's damage mix differs from
  training.
- **Follow-ups:** Would merging minor and major help? (The project's 3-class experiment explores this; it is
  context only.)

**Q48. How large is the cross-event drop?**
- **Short:** −48% for KNN, −37% for the DT.
- **Detailed:** From 0.520 → 0.271 and 0.470 → 0.294.
- **Follow-ups:** Is that expected? (Yes, under domain shift; the confounding check supports it.)

**Q49. What is the event-confounding check?**
- **Short:** An analysis classifier (multinomial logistic regression, not a damage model) predicting the *disaster* from the same 37 features: 0.829 accuracy vs
  0.252 for the majority guess.
- **Detailed:** It shows a strong event signature that can inflate within-event scores.
- **Follow-ups:** Is the event used as a model input? (Never.)

**Q50. What were the feature-set experiments?**
- **Short:** Reduced or normalised feature sets (called "Phase 3b" during development) to remove the event
  signature.
- **Detailed:** change-only (17 features), scene-normalised, and both, with a logistic regression and the DT as analysis classifiers. Event prediction fell
  to as low as 0.374, but cross-event macro-F1 did not improve.
- **Follow-ups:** Conclusion? (The gap is genuine domain shift, not just the model "cheating".)

**Q51. How did you compute confidence intervals?**
- **Short:** A scene-level bootstrap: resample test scenes 2,000 times (seed 42), recompute macro-F1, take
  the 2.5th–97.5th percentiles.
- **Detailed:** It uses saved predictions only. Within-event: KNN [0.459, 0.560], DT [0.401, 0.526].
- **Follow-ups:** Why scenes, not buildings? (Buildings in a scene are correlated; resampling buildings would
  make intervals too narrow.)

**Q52. Is the difference significant?**
- **Short:** The paired intervals exclude zero: +0.050 [+0.013, +0.084] within, −0.023 [−0.034, −0.011] cross.
- **Detailed:** They only cover test-scene variation for fixed models, not retraining or other disasters, so
  we avoid the word "significant".
- **Follow-ups:** What would strengthen it? (Several seeds, more held-out events.)

**Q53. Why is validation so much higher than test cross-event?**
- **Short:** Validation scenes come from the four training events, so they share the event signature.
- **Detailed:** Validation 0.581 / 0.576 vs test 0.271 / 0.294.
- **Follow-ups:** Better validation for transfer? (Hold out a whole event for validation too.)

**Q54. What does the Review Queue do?**
- **Short:** It lists test buildings suggested for human review: confidence below a validation-chosen
  threshold, or a model disagreement.
- **Detailed:** The threshold is the lowest *t* at which validation buildings with confidence ≥ *t* are at
  least 70% correct: KNN 0.53 and DT 0.53 (the served models).
- **Follow-ups:** Is a flag an error? (No, it's a suggestion.)

**Q55. What do K-Means results show?**
- **Short:** Natural clusters don't match damage: ARI 0.04–0.10.
- **Detailed:** The best silhouette is at k = 2 (0.253). Clusters reflect scene appearance and disaster more
  than damage. It is analysis only.
- **Follow-ups:** Why include it? (It shows supervised labels are needed.)

## F. Engineering and integrity (Q56–Q64)

**Q56. How do you know results weren't tuned on test?**
- **Short:** `TestSetGuard` permits one test evaluation per model, and the metadata records
  `test_evaluations = 1`.
- **Detailed:** Hyperparameters are selected on validation only.
- **Follow-ups:** Could you have rerun training? (That would replace the saved results; we didn't.)

**Q57. How do you know your simplifications changed nothing?**
- **Short:** `scripts/snapshot_behaviour.py`: 12 demo analyses, 752 explanations, Feature Lab and `/models`
  are identical before and after.
- **Detailed:** It calls the real API routes with the saved models and compares JSON.
- **Follow-ups:** What changed then? (UI wording and dead code: see `CHANGELOG.md`.)

**Q58. Where are files stored, and how is the API secured against path traversal?**
- **Short:** In files read through `Store`, which accepts only validated IDs, never paths.
- **Detailed:** API tests try traversal attempts and expect structured 404/422 errors.
- **Follow-ups:** Error format? (`{"error": {"code", "message", "fix"}}`, no tracebacks.)

**Q59. What happens if a saved file is broken?**
- **Short:** An explicit state, not a crash. Readiness shows NOT READY or PARTIAL with the fix command; a
  broken experiment summary returns `503 results_invalid`.
- **Detailed:** Tests cover a missing file, bad JSON and missing keys.
- **Follow-ups:** Which models count? (Only the served ones in config, KNN and the Decision Tree; Logistic
  Regression and Random Forest are not served, so they never affect a tile.)

**Q60. Are confidence scores probabilities?**
- **Short:** No. They are model scores and are not calibrated.
- **Detailed:** KNN's is a vote fraction; the DT's a leaf share (class-weighted).
- **Follow-ups:** How to calibrate? (Isotonic or Platt scaling on validation; future work.)

**Q61. How many tests are there?**
- **Short:** 155 backend (pytest), 40 frontend (Vitest), and a 20-check end-to-end script, all passing in the
  final clone.
- **Detailed:** They include leakage, model reload, API and UI states.
- **Follow-ups:** What do the tests use for data? (Synthetic fixtures, never shown as xBD.)

**Q62. Why did you get a PyTorch error on Windows?**
- **Short:** Pytest ran with a global Python that had torch in user site-packages. Inside the project venv it
  passes.
- **Detailed:** The project imports no deep-learning library; the test exists to keep it that way.
- **Follow-ups:** Did you delete the test? (No.)

**Q63. What are your main limitations?**
- **Short:** One held-out event, one training run, unequal class handling, mask-derived buildings,
  uncalibrated scores, and no operational validation.
- **Detailed:** See `12_LIMITATIONS_AND_FUTURE_WORK.md`.
- **Follow-ups:** The most important one? (One held-out event: the cross-event result describes Michael only.)

**Q64. What would you do next?**
- **Short:** Class-balanced KNN, several seeds, leave-one-event-out, more events, calibration, and a comparison
  with learned features.
- **Detailed:** These address internal validity (fair class handling), statistical validity (seeds) and
  external validity (events).
- **Follow-ups:** Which first? (Leave-one-event-out: it tests whether the ranking change is general.)

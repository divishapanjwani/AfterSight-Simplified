# Verification checklist (Deliverable 5)

Manuscript: `paper/AfterSight_KNN_vs_DT_IEEE.docx` and PDF export `paper/AfterSight_KNN_vs_DT_IEEE.pdf`
(5 pages, references included; target ≤ 6). Branch `paper`, built from `main` at v1.3.

**This is not publication-ready.** The open items are listed at the end.

## How to rebuild

```zsh
.venv/bin/python scripts/paper_stats.py                 # scene-level bootstrap → paper/build/stats.json
.venv/bin/python paper/build/make_figures.py            # Figs. 1–3 → paper/figures/
python3.11 -m venv paper/build/.venv && paper/build/.venv/bin/python -m pip install -r paper/build/requirements.txt
paper/build/.venv/bin/python paper/build/build_paper.py # → paper/AfterSight_KNN_vs_DT_IEEE.docx
/Applications/LibreOffice.app/Contents/MacOS/soffice --headless --convert-to pdf --outdir paper paper/AfterSight_KNN_vs_DT_IEEE.docx
```

## Technical accuracy

- [x] **Numbers come from artifacts.** Every number in the text and tables is read from saved artifacts at
  build time (`build_paper.py`); none is typed in. The exceptions are configuration constants, each checked
  against its config file:
  - σ = 1.5, 32 GLCM levels, 8 px ring (`config/features.yaml`);
  - 16 px minimum area and 4-connectivity (`config/events.yaml`);
  - the grids, seed 42 and split ratios (`config/training.yaml`).
- [x] **Headline metrics reproduced.** The four headline macro-F1 values were recomputed from the saved
  predictions, and re-inference with the saved models reproduces every prediction (evidence register 6.1–6.2).
- [x] **Bootstrap.** The scene-level bootstrap's point estimates equal the saved metrics.
  `scripts/snapshot_behaviour.py` was IDENTICAL afterwards; no model was retrained.
- [x] **Features.** The 37 features, the grey-level weights, the GLCM definition (pairs inside the
  footprint, horizontal and vertical, 32 levels), the context ring and compactness match `extract.py`.
- [x] **Classifiers and preprocessing.** KNN (Euclidean, k = 15, uniform) and DT (Gini, depth 8, leaf ≥ 100,
  balanced weights, no pruning) match the saved estimators. The shared preprocessing is fitted on train only.
- [x] **Model selection.** The selected models are the ones fitted on the training split (no refit on
  train + val).
- [x] **Building regions** are described as annotation-derived, not detected.
- [x] **Protocol differences** (class weighting, grid sizes 6 vs 8, scaling) are stated in Sec. V-B and
  discussed in Sec. VII.
- [x] **Claims re-checked.** Each overstated sentence found in review was corrected:
  - "about half" → 48% / 37%;
  - "destroyed best" → "best among the damage classes";
  - the DT's cross-event edge → attributed to minor damage;
  - the event-selection rationale.
- [x] **Abstract, results and conclusion agree** (all three use the same computed values).
- [x] **Equations (1)–(7)** have every symbol defined in the surrounding text.

## Results and statistics

- [x] **Scope of the intervals.** The bootstrap intervals cover test-scene sampling only, as the text says.
  Training variability and between-event variability are not included (Sec. V-C, VII).
- [x] **No significance wording.** The text says a paired interval "excludes zero"; there is no p-value and
  no "significant".
- [x] **One held-out event.** No claim about unseen disasters in general.
- [x] **Causal explanations are labelled as hypotheses** (class weighting; event signature).

## Reproducibility

- [x] **Saved state.** Split manifests are hashed (`7bba8a9fbbd2740a`, `dd7d8566b7dbaf7a`); seed 42; the
  sklearn and Python versions and the hardware are stated.
- [ ] **Code and data availability.** No statement yet (repository URL, licence, how to obtain the
  third-party data copy). Decide what the venue allows (anonymous review?).
- [ ] **Single training run.** Repeated seeds were out of scope; noted as future work.

## Citations

- [x] **All 17 references verified** (`paper/references.md`): 15 via Crossref DOI records, JMLR via its
  publisher page, and the xBD workshop paper via mlanthology/CVF.
- [x] **Valentijn *et al.*** was read in full; it is cited as a contrast (high AUC on unseen disasters in
  their CNN setting), not as support.
- [x] **Numbering.** References are numbered in order of first citation, and every reference is cited.
- [ ] **Style.** Check the reference style against the chosen venue (IEEE abbreviations used; venue may
  differ).

## Formatting

- [ ] **Official template not used.** It could not be downloaded (ieee.org returned an empty response to
  scripted requests), so the paper uses a **provisional** IEEE-style layout built with python-docx: US Letter,
  Times New Roman 10 pt, two columns 3.5 in + 0.25 in gap, small-caps section headings, numbered tables,
  figures and equations. Every page header says so. **Move the text into the official IEEE conference
  template before submission.**
- [ ] **Venue-specific compliance is unconfirmed.** No venue chosen; check its page limit, margins,
  abstract length, keywords, copyright notice and PDF eXpress rules.
- [ ] **Equations** are typed as text with Unicode and sub/superscripts, not as Word equation objects;
  re-enter them in the template's equation format if required.
- [x] **Figures** are 300 dpi PNG, generated from artifacts, and each is numbered, captioned and cited in
  the text.
- [x] **Tables I–V** are numbered, captioned above, and cited in the text.

## Author details and ethics

- [ ] **Author block.** It holds placeholders (4 × "Author Name" plus "Guide Name"); fill it in the Word file.
- [ ] **Acknowledgement and AI-assisted writing disclosure** are placeholders; complete them to institution
  and venue policy.
- [x] **Novelty wording.** No novelty claim beyond an applied comparative study; no state-of-the-art claim.
- [x] **No checker reports.** No similarity score or AI-detection result was produced or claimed. Run the
  institution's or venue's checker yourselves.

## Originality review: passages to check against your institution's checker

- **Sec. II, related-work summaries** of [3], [7], [9], [12]: paraphrased from the abstracts. Compare them
  with the sources and adjust any phrase that remains close.
- **Standard definitions** of precision, recall, F1, Gini and Euclidean distance: these are textbook
  formulations, and overlap there is expected. Keep the citations.
- **Text shared with the project's own docs and report:** the paper was written fresh, but some phrases
  (e.g. "scene-grouped", "mask-derived") recur deliberately as terminology. If the group's mini-project
  report was already submitted to a checker, expect matches against it.

## Still unresolved before submission

1. Venue and template (see above).
2. Authors, affiliations, acknowledgements and the AI-use disclosure.
3. A code and data availability statement.
4. Optional, if the venue wants stronger evidence: repeated-seed training, a class-balanced KNN variant,
   and leave-one-event-out over all five events. These need new experiments, which were out of scope here.

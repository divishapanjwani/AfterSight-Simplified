# Paper changelog: draft → final

- **Draft:** `paper/AfterSight_KNN_vs_DT_IEEE.docx`, commit `22af80a` on the original's `paper` branch.
- **Final:** `docs/research/AfterSight_Final_Research_Paper.docx` / `.pdf`, built in the simplified clone by
  the same script, `paper/build/build_paper.py`.

The title, research questions, framing, structure (Sections I–VIII), tables, figures and all 17 verified
references are kept. Only what the evidence or the clone required was changed.

| # | Change | Where | Why |
|---|---|---|---|
| 1 | The abstract now says the 71,340 buildings come from **1,179** image pairs (was "in 1,239 image pairs") | Abstract | 60 of the 1,239 cached pairs contain no labelled building, so "in 1,239 pairs" overstated the data behind the experiments |
| 2 | New sentences give the scene accounting: of 1,239 pairs, **52 contain no building component** and **8 contain only excluded components (all 28 lacked a damage value)**, so 1,179 pairs contribute buildings, exactly the scenes in both split designs | Sec. III-B | The open 1,239 vs 821 + 181 + 177 question. Verified from `artifacts/inspection/xbd/scenes.csv`, the drop reasons in `data/processed/scenes/<id>/scene.json` (all `label_missing`) and both manifests; the builder asserts the identity |
| 3 | Footnote added to Table I: "Scenes = cached image pairs; 1,179 of them contain at least one labelled building" | Table I | The table's scene counts (e.g. Michael 343) are cached pairs, not split scenes |
| 4 | The cross-event test set is "every Hurricane Michael scene that contains labelled buildings (333 of 343)" (was "every Hurricane Michael scene") | Sec. V-A | Precision: 10 Michael pairs have no labelled building |
| 5 | Output path and file name | — | Deliverable location in the clone |
| 6 | The explanation sentence now says the application explains KNN predictions with nearest labelled training buildings, and that DT root-to-leaf paths are reported from the saved research models and shown in the Model Arena's tree view, but the DT is not served for live predictions (was "In the AfterSight application … a DT prediction by its root-to-leaf path") | Sec. VI-C, last paragraph | The app now serves only LR and KNN (`CHANGELOG.md`, "Two models"). No other text, number, table or figure changed: the rebuilt paper's text differs from the previous build in that sentence only, the embedded figures are byte-identical, and it is still 5 pages |

## Checked and unchanged

- All results, intervals and figures: `paper_stats.py` reproduced `stats.json` byte-for-byte, and the
  figures regenerated identically.
- The 37 features and their names; data provenance; the KNN and DT configurations and grids; the protocol
  differences; the hardware and software versions.
- The limitations section is kept in full, including the uncalibrated-confidence and no-operational-validation
  statements.
- The author block, acknowledgements and AI-use disclosure are still placeholders.
- The page header still says the layout is provisional and that no venue's compliance is claimed.

## UI simplifications and the paper

The clone's UI changes are listed in `CHANGELOG.md`:

- the dead navigation branch, the internal labels, the unused demo chip, the misleading "Re-run analysis"
  button, and "Phase 3b" → "feature-set experiment".

None of these was described in the draft paper, so no paper text had to be removed. Later the app was
reduced to LR and KNN; change 6 above updates the one sentence that described the DT path as an app feature. The paper
never used the name "Phase 3b"; it reports only the confounding check.

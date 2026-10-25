# Final verification report

| | |
|---|---|
| Clone | `~/AfterSight-Simplified`, branch `simplified`, local only (no git remote; nothing pushed) |
| Original | `~/aftersight`, branch `paper` at `22af80a`, tag `v1.3` on `main` |
| Date | 2026-10-25 |

Every check below was run on the final state of the clone. Results are copied from the command output.

## 1. The original is untouched

- `verification/original_checksums.sha256` holds SHA-256 checksums of all **10,358** files of
  `~/aftersight`, recorded *before* cloning (it excludes `.venv`, `frontend/node_modules`,
  `paper/build/.venv` and `report/build/node_modules`).
- **Final check:** `shasum -a 256 -c` in `~/aftersight` → **all 10,358 files match**.
- The original's working tree is clean, still on `paper` at `22af80a`.
- The clone's git remote was removed, so it cannot push into the original.

## 2. The clone starts from its README, and the main workflow works

- `./start.sh` (README Quick start) started the API on 8020 and the web app on 5200 **from the clone**; the
  API process's working directory is `~/AfterSight-Simplified/backend`.
- Every page route returns HTTP 200: `/`, `/library`, `/triage`, `/triage/<scene>`, `/case`, `/arena`,
  `/features`, `/clusters`, `/review`, `/reports`.
- The main API endpoints return 200: health, dataset status, scenes, models, comparison, confounding,
  experiments, clusters, feature summary, review queue, analyses, and `POST /analysis`.
- **Walkthrough:** Library → Triage (Run analysis, KNN and DT) → Case File (KNN neighbours, DT path) → Arena
  (feature-set experiments) → Reports (PDF download). Screenshots are in `docs/screenshots/`.
- **Run one copy at a time:** both use ports 8020 and 5200, and `start.sh` stops whatever is on them.

## 3. No broken references

- The TypeScript check and the production build pass, so there are no dangling imports.
- No remaining references to the removed elements: a search of `frontend/src` for "Arrives in",
  `` API v${ ``, `kind="demo"` and "Re-run analysis" finds nothing.
- "Phase 3b" is no longer shown in the UI. A test checks this even though the API still sends the old label.
  The docs name it once, next to the artifact paths.
- All 139 file paths cited in the study guide, the research docs, `CHANGELOG.md` and `CLONE_VERIFICATION.md`
  exist in the clone. Bare file names were resolved by searching.

## 4. Tests, build and lint (real counts)

| Check | Result |
|---|---|
| `python -m pytest` (backend) | **148 passed** |
| Vitest (frontend) | **37 passed** (4 files) |
| TypeScript (`tsc -b`) | passes |
| Lint (`oxlint src`) | 0 errors, 6 warnings (the same 6 as v1.3; none new) |
| Production build (`npm run build`) | passes |
| `scripts/e2e_check.py` | **22/22** checks passed |

No test was removed or weakened. Three assertions were added (the old name is not shown, and there is no run
button once an analysis is displayed).

## 5. Behaviour snapshot

`scripts/snapshot_behaviour.py check verification/snapshot_before.json` → **IDENTICAL**: 12 demo analyses
(predictions, confidences, scores, review flags), 752 explanations, 37 Feature Lab features, 4 scatter
samples and `/models`.

- The baseline was captured in the clone before any edit, and it is identical to the original v1.3
  baseline.
- The UI changes did not alter the snapshot's shape or its values.

## 6. Paper ↔ artifacts

`docs/research/PAPER_CONSISTENCY_AUDIT.md` maps every claim to its evidence. In summary:

- Every number is read from saved artifacts when the paper is built.
- **Reproduced in the clone without retraining:**
  - the bootstrap statistics (`scripts/paper_stats.py`, byte-identical to the draft's `stats.json`);
  - the figures (identical);
  - the four headline macro-F1 values (recomputed from the saved predictions, which equal re-inference with
    the saved models).
- **The scene-count question is resolved and stated in the paper:** 1,239 cached pairs = 52 with no building
  component + 8 with only label-less components (28 blobs, all `label_missing`) + **1,179** with buildings,
  exactly the scenes in both manifests. The builder asserts this.
- The paper makes no claim about removed UI elements and keeps every limitation.
- The author block, acknowledgements and AI disclosure are placeholders.
- The layout is a labelled provisional IEEE style; no venue compliance is claimed.

## 7. Study guide ↔ code and paper

- The 14 files in `docs/study-guide/` were written from the clone's code and the final paper.
- **Function and route names** were taken from the code (`grep` of every module's definitions and of
  `routes.py`).
- **Numbers** come from the artifacts and match the paper:
  - 0.520 / 0.470 and 0.271 / 0.294 macro-F1;
  - the CIs and paired differences;
  - the per-class values;
  - the class weights;
  - the grids;
  - the scene accounting.
- **Library versions** were checked against `frontend/package.json` and `backend/requirements.txt` (one
  error, React 18 → 19, was corrected).
- **Q&A count:** 64.

## 8. Deliverables

| Deliverable | Path |
|---|---|
| Simplified clone | `~/AfterSight-Simplified` (branch `simplified`) |
| Final paper | `docs/research/AfterSight_Final_Research_Paper.docx`, `.pdf` (5 pages incl. references) |
| Paper changelog / audit / reproduction | `docs/research/PAPER_CHANGELOG.md`, `PAPER_CONSISTENCY_AUDIT.md`, `EXPERIMENT_REPRODUCTION.md` |
| Study guide (14 files) | `docs/study-guide/00_PROJECT_OVERVIEW.md` … `13_GLOSSARY_AND_QUICK_REVISION.md` |
| Clone verification | `CLONE_VERIFICATION.md`, `verification/original_checksums.sha256`, `verification/snapshot_before.json` |
| Code and UI changelog | `CHANGELOG.md` |
| Screenshots | `docs/screenshots/` |
| This report | `FINAL_VERIFICATION_REPORT.md` |

## 9. Not verified, and why

- **A full pipeline rerun from raw data** (`scripts/run_pipeline.sh`) was not done. It retrains every model and
  overwrites the saved results, which was ruled out. The documented 571 s run time and the fresh-clone
  reproducibility come from earlier project checks.
- **The Windows setup steps** in `09_SETUP_AND_TROUBLESHOOTING.md` were not executed here, because this machine
  runs macOS. They follow the project's earlier Windows diagnosis.
- **The official IEEE template** was not applied (ieee.org blocks scripted downloads). The paper's layout is
  provisional.
- **`report/` and `paper/`** in the clone are the earlier mini-project report and paper draft. They are kept
  unchanged as historical records and still use the name "Phase 3b". `docs/research/` supersedes the draft.

## 10. Decisions made without asking

- Removed the clone's git remote, so it can't push into the original.
- Excluded the two build-tool dependency folders (`paper/build/.venv`, `report/build/node_modules`) from the
  checksum list, alongside `.venv` and `frontend/node_modules`.
- Hid the Triage "Run analysis" button once an analysis is shown, rather than relabelling it.
- Stopped displaying the saved experiment `label` (which contains "Phase 3b") instead of rewriting the saved
  artifact.
- Also renamed the `run_pipeline.sh` console label to "feature-set experiments (Phase 3b)".
- Left the historical `report/` and `paper/` folders unchanged; the final paper lives in `docs/research/`.
- Committed the 2.9 MB behaviour baseline (`verification/snapshot_before.json`) as evidence.

## How to run the clone

```zsh
# make sure the original ~/aftersight is not running (same ports)
cd ~/AfterSight-Simplified
./start.sh                    # API http://localhost:8020 (Swagger at /docs), app http://localhost:5200
# Ctrl-C stops both
```

Checks: `.venv/bin/python -m pytest`; `(cd frontend && npx vitest run)`; `.venv/bin/python scripts/e2e_check.py`;
`.venv/bin/python scripts/snapshot_behaviour.py check verification/snapshot_before.json`.

# Changelog: AfterSight-Simplified

The simplified clone, branch `simplified`, starts from the original's `paper` branch (`22af80a`, built on
tag `v1.3`). Every change was made **in the clone only**.

## Code and interface (approved rows 1–9 of the change table)

| Change | Files | Why |
|---|---|---|
| Removed the never-used `ready` flag on navigation items and its disabled branch with the tooltip "Arrives in Phase 5 Part B" | `frontend/src/components/AppShell.tsx` | Dead code holding a development-phase note |
| The API status dot's tooltip says "API online" instead of the internal API version ("API v0.1.0") | `AppShell.tsx` | The internal version contradicted the release tags |
| Removed the unused `demo` chip kind | `frontend/src/components/Chip.tsx` | Demo mode was never built. The API's `demo_mode` field (always `false`) is kept, so the API contract is unchanged |
| The Triage Desk shows **Run analysis** only until the result appears; the "Re-run analysis" state is gone | `frontend/src/pages/TriageDeskPage.tsx` | Analyses are deterministic and cached per (scene, model), so re-running only re-fetched the same result. Choosing another model already runs that model's analysis |
| "Phase 3b" → **"feature-set experiment"**: the Arena section tab ("EXPERIMENTS · FEATURE SETS"), the confounding bar labels, the confounding text, the invalid-summary state and the backend error message. The saved `label` string is no longer displayed | `frontend/src/pages/ModelArenaPage.tsx`, `backend/app/experiments/summary.py` | Internal development name. The saved artifact, the artifact folder (`artifacts/experiments/phase3b/`), the script name and the API routes are unchanged |
| Stale note "(review queue, later)" | `frontend/DESIGN.md` | Outdated |

**Kept, by decision:**

- confidence scores and the low-confidence review mechanism (option A: already labelled "not calibrated"
  or "model score");
- every working page, including the training-job API.

**Tests:** wording updated in `ResultsPages.test.tsx`. New assertions check that the old name is not shown
(even though the API still sends it) and that no run button remains after an analysis
(`TriageDeskPage.test.tsx`). No test was removed or weakened.

## Documentation

- `README.md` and `docs/*.md` use "feature-set experiment". Each mentions the old name "Phase 3b" once, next
  to the artifact paths. The README notes that the clone and the original share ports, so only one should
  run at a time.
- **New:**
  - `CLONE_VERIFICATION.md` and `verification/original_checksums.sha256`;
  - `verification/snapshot_before.json` (behaviour baseline);
  - `docs/screenshots/` (workflow screenshots from the clone);
  - `docs/research/` (final paper and its audit, changelog and reproduction notes);
  - `docs/study-guide/` (14 files);
  - `FINAL_VERIFICATION_REPORT.md`.
- **Left as historical records, unchanged:**
  - `report/`: the submitted mini-project report and the beginner-guide PDF;
  - `paper/`: the earlier paper draft, superseded by `docs/research/`.

## Not changed

Dataset, features, splits, training, models, saved results, configs, API routes and response shapes. The
behaviour snapshot (12 demo analyses, 752 explanations, Feature Lab, `/models`) is **IDENTICAL** to the
baseline taken before any edit.

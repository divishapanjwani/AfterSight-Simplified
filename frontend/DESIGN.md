# AfterSight design system: "field case file"

The UI should feel like an incident-response case file: calm, precise and editorial. Paper and ink,
thin rules, labelled folders, and numbers you can trust because each one says where it came from.
It is deliberately **not** a dark map dashboard.

Tokens live in `src/index.css`: CSS variables on `:root`, overridden under `:root[data-theme='dark']`
and exposed to Tailwind through `@theme inline`. The damage scale is in `src/theme/damage.tsx` and the
chips in `src/components/Chip.tsx`.

## Themes

- **Light (default):** paper `#f6f4ef`, sheet `#fcfcfb`, ink `#1b1b1a`.
- **Dark:** paper `#141413`, sheet `#1c1b19`, ink `#ecebe6`. Toggle it from the top bar. The choice is
  remembered per browser; `?theme=dark` forces it for reviews and screenshots.
- Dark mode has its own validated values. It is never an automatic inversion.

## Colour roles

Each colour has one job:

| Token | Light | Use | Never |
|---|---|---|---|
| `--signal` (safety orange) | `#c2410c` | **primary actions only** (one per view, e.g. "Open held-out test scenes") | navigation state, labels, data |
| `--caution` (ochre) | `#8a5a00` | provenance warnings: training/validation scene | errors |
| `--danger` (brick) | `#a12a1c` | API/system errors | damage classes |
| `--ok` (green) | `#2f6b3a` | "READY" badge, always with ✓ and the word | damage classes |
| `--dmg-0…3` (plum ramp) | see below | damage classes only | anything else |
| magnitude blue (figures) | `#f3f7fd → #0d366b` | heatmap magnitude in report figures | damage classes |

The orange is a slightly deeper step than the original `#e8590c`, so white text on primary buttons
reaches about 5:1 contrast. Active navigation is marked with an **ink** underline, not orange.

## Damage scale: one ordinal ramp + a pattern per class

| Class | Light | Dark | Pattern | Why the pattern |
|---|---|---|---|---|
| no-damage | `#c9a8d8` | `#e3cdee` | outline + faint fill | reads as "intact" |
| minor-damage | `#9a64b5` | `#c39ad8` | **dots** | minor and major are close in lightness on imagery, |
| major-damage | `#6b3590` | `#a066bf` | **diagonal hatch** | so their patterns differ the most |
| destroyed | `#3d1559` | `#7d3fa3` | solid | heaviest mark = most severe |

- One hue, light → dark means more severe, in both themes.
- Both ramps pass the ordinal checks of the palette validator: monotone lightness, adjacent gaps
  ≥ 0.06, a single hue, and the step nearest the surface at ≥ 2:1 (2.03:1 light, 2.58:1 dark).
- **Colour is never alone.** Every swatch sits next to the class name. Overlays use the pattern fills.
  Counts are written out ("4 · 1 · 0 · 2"), and the bar carries an `aria-label`.
- The same ramp is used by the backend report figures (`backend/app/core/damage_scale.py`).

## Type

- **UI:** IBM Plex Sans 400/500/600, self-hosted through `@fontsource`, so the app works offline.
- **IDs and numbers:** IBM Plex Mono with `font-variant-numeric: tabular-nums` (the `.num` class), so
  columns of scores line up.
- **Eyebrows and tabs:** mono, 10–11 px, uppercase, letter-spaced (e.g. `SCENE LIBRARY`, `STATUS · 01`).

## Layout

- **Top bar:** the wordmark, then the exact scope label *"Damage classification on building footprints
  derived from xBD annotations."*, the API status dot and the theme toggle.
- **Workflow stepper:** `01 Library → 02 Triage → 03 Case file → 04 Arena → 05 Report` on the left;
  tools (Readiness, Feature Lab, Cluster Atlas, Review Queue) on the right. There is no sidebar.
  Pages that are not built yet are shown disabled, never as empty placeholders.
- **Width:** content is at most 1240 px with a 20 px gutter, designed for 1280 px laptops. Tables keep
  IDs on one line.
- **Case folders:** every card is a `CaseFolder`, a sheet with a thin tab label on its top edge
  (`STATUS · 03`, `SCENES · 177 MATCH`, later `CASE · B-0142`). Radius 4 px, 1 px rule, faint shadow.

## Provenance chips

Every number or label that could be mistaken for something else carries a chip with a glyph and a word:

| Chip | Meaning |
|---|---|
| `◆ PREDICTED` | model output |
| `● GROUND TRUTH` | xBD annotation (mask-derived), double border |
| `▲ TEST SET` | held-out; never used to fit or tune |
| `△ VALIDATION` | used for tuning; caution tone |
| `! TRAINING SCENE` | used for fitting; caution tone; results optimistic |
| `– NOT AVAILABLE` | not computed / not built, dashed |
| `⚙ FROM CONFIG` | e.g. disaster type, which the dataset does not store |
| `◇ DEMO` | reserved for a watermarked demo mode (not implemented) |

Train and val scenes also get a full-width caution banner. Chips alone are not enough there.

## States (`src/components/States.tsx`)

| State | Look | Content |
|---|---|---|
| Loading | mono caption + pulsing skeleton bars (disabled under reduced motion) | what is being loaded |
| Empty | dashed folder | why nothing is shown + a reset action |
| Not built / not trained | dashed, neutral (a setup step, not a failure) | API message + copyable fix command |
| Error | brick border, `role="alert"` | message, `code · HTTP status`, fix command if the API sent one |

"Suggested for human review" (review queue, later) uses neutral styling. It is never shown as an error.

## Overlays (Triage Desk)

- **Ground truth / Prediction:** the class's pattern fill plus its ramp colour as the outline.
- **Correct vs wrong:** a thin **solid** green outline means the prediction agrees with ground truth; a
  thicker **dashed** red outline means it differs. Line style carries the meaning without colour, which
  matters for red/green colour blindness.
- **Selection:** a thick focus-blue (`--focus`) outline; other footprints dim. Orange is never used for
  selection.
- **Line width:** stays constant on screen at any zoom, because the stroke is divided by the zoom factor.
- **Crops:** the Case File crops come from the cached view JPEG with the footprint outlined in the same
  pixel coordinates. KNN neighbour thumbnails (other scenes) use the server crop endpoint.

## Charts

All charts are hand-built SVG components (`src/charts/`) rather than Recharts, which the original stack
listed. This deviation gives exact control over pattern fills, direct labels and theme tokens, and makes
the charts testable in jsdom.

| Need | Encoding |
|---|---|
| model identity (Arena dumbbell) | categorical palette `--cat-1` blue (●) / `--cat-2` aqua (■). These are the first slots of the validated reference palette, which pass all-pairs; its orange slot is skipped because orange means "action" here |
| clusters (Cluster Atlas) | `--cat-1…7` + marker shape; more than 7 clusters cycle the shape |
| magnitude (confusion, contingency) | single-hue blue `--mag-0…4`; every cell also prints its count and row % |
| damage class (box plots, scatter) | damage ramp + pattern (boxes) or ramp + marker shape (scatter) |

Every chart has a legend, direct value labels or a table, and native `<title>` hover text. There are no
dual axes. Log axes are used only for the skewed features, and they are labelled "(log)".

## Rules for pages

1. Every number comes from an API response. No hard-coded scores or counts. Tests assert that the
   values rendered are the mocked API values.
2. Train and val scenes show their split warning prominently.
3. Polygons are drawn in image pixel coordinates (SVG). There is no world map and no areas in m² or km².

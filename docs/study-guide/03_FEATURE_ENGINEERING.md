# 03 · Feature engineering: the 37 features

Code: `backend/app/features/extract.py` (`building_features`, `glcm_contrast_homogeneity`, `compactness`,
`scene_rows`). Names and order: `backend/app/features/schema.py` (`FEATURE_COLUMNS`, `FEATURE_GROUPS`).
Settings: `config/features.yaml`. Output: `data/processed/features/features_v1.parquet` (71,340 rows).

## Notation

- **Ω:** the building's footprint pixels (from the footprint mask).
- **Pre and post RGB:** the pre-disaster and post-disaster RGB pixels inside Ω.
- **Grey value** (ITU-R BT.601 luma): *g = 0.299R + 0.587G + 0.114B*.
- **ḡ_pre, ḡ_post:** mean grey over Ω before and after.
- **Ring:** ground pixels within **8 px** of the building, **excluding every building pixel** (grown with
  an 8-neighbour dilation).

## 1. Colour, 12 features

| Feature | Formula | Meaning |
|---|---|---|
| `pre_mean_r/g/b` | mean of R, G or B over Ω (pre) | roof colour before |
| `pre_std_r/g/b` | standard deviation of R/G/B over Ω (pre) | colour variation before |
| `post_mean_r/g/b`, `post_std_r/g/b` | the same on the post image | roof colour and variation after |

**Why:** collapsed or burned roofs change colour; ash is grey-white, and wet debris is darker.
**Limits:** absolute colour also depends on the sensor, season, sun angle and roof material, so these
features carry a strong *which-disaster* signature (see the confounding check in file 06).

## 2. Change, 8 features

| Feature | Formula | Meaning |
|---|---|---|
| `abs_dmean_r/g/b` | \|post_mean − pre_mean\| per channel | how much the average colour changed |
| `dstd_r/g/b` | post_std − pre_std per channel | did the roof get more or less uniform? |
| `gray_absdiff_mean` | *D* = (1/\|Ω\|) Σ_{p∈Ω} \|g_post(p) − g_pre(p)\| | average per-pixel brightness change |
| `brightness_ratio` | *B* = ḡ_post / max(ḡ_pre, 1) | relative brightening (the max avoids dividing by a black roof) |

**Why:** damage *is* change; comparing the same pixels before and after removes much of the roof-type effect.
**Limits:** a small pre/post misalignment also creates change (168 pairs are alignment-flagged).

## 3. Texture, 12 features

For each of pre and post, plus the difference (post − pre):

| Feature | How it is computed | Meaning |
|---|---|---|
| `gray_std_pre/post/delta` | std of *g* over Ω | roughness of the roof |
| `edge_density_pre/post/delta` | fraction of Ω pixels on a **Canny** edge (σ = 1.5, run on the whole crop so the outline is detectable) | rubble has many edges; intact roofs few |
| `glcm_contrast_pre/post/delta` | *C* = Σ P(i,j)(i − j)² | how different neighbouring grey levels are |
| `glcm_homogeneity_pre/post/delta` | *H* = Σ P(i,j) / (1 + (i − j)²) | how similar neighbours are (1 = identical) |

**GLCM details** (grey-level co-occurrence matrix, Haralick 1973):

- grey is quantised to **32 levels**;
- pixel pairs are taken **horizontally and vertically at distance 1**, using only pairs where *both* pixels
  are inside the footprint.

The code computes the mean of (i − j)² and of 1/(1 + (i − j)²) over those pairs. That equals the sum over a
normalised symmetric GLCM, so the matrix never has to be built.

**Why:** texture is the strongest signal. In the Feature Lab, the features that best separate the classes on
the training split are:

- `glcm_homogeneity_post` (η² = 0.185);
- `glcm_homogeneity_delta` (0.136);
- `glcm_contrast_post` (0.096).

η² is the share of a feature's variance explained by the damage class.

**Limits:** texture depends on resolution and blur, and trees or shadows over a roof add texture without
damage.

## 4. Context, 3 features

| Feature | Formula | Meaning |
|---|---|---|
| `ring_dmean` | mean(g_post over ring) − mean(g_pre over ring) | did the surroundings change (flood water, ash)? |
| `ring_absdiff_mean` | mean \|g_post − g_pre\| over the ring | how much the surroundings changed |
| `contrast_change` | *K* = (ḡ_post − r̄_post) − (ḡ_pre − r̄_pre) | did the building change **relative to** its surroundings? |

**Why:** a whole flooded street brightens or darkens together; *K* separates "the building changed" from
"everything changed".
**Limits:** in dense blocks the ring can be empty; these features are then missing (NaN). This never happens
in our data, where the table has 0 missing values.

## 5. Shape, 2 features

| Feature | Formula | Meaning |
|---|---|---|
| `area_px` | number of footprint pixels | building size (log-transformed before scaling) |
| `compactness` | 4πA / P² (polygon area A, perimeter P) | 1 for a circle, π/4 ≈ 0.785 for a square, near 0 for thin shapes |

**Why:** size and shape interact with how damage looks, for example large warehouses vs small houses.
**Limits:** merged blobs (touching buildings) distort both features. Area alone separates the classes very
weakly (η² = 0.002).

## A worked example (real numbers)

Building `santa-rosa-wildfire_00000006_b0002` is a destroyed house in a held-out test scene. These values
were traced through the real code (`docs/build/understanding_data.json`):

| Feature | Value | Reading |
|---|---:|---|
| `post_mean_r` (pre 89.2) | 155.4 | the roof got much lighter (ash) |
| `gray_absdiff_mean` | 60.1 | each pixel changed by about 60 of 255 |
| `brightness_ratio` | 1.64 | 64% brighter after |
| `gray_std_delta` | +12.0 | roughness doubled (11.4 → 23.4) |
| `edge_density` pre → post | 0.023 → 0.181 | 2% → 18% of pixels on edges |
| `glcm_homogeneity` pre → post (delta) | 0.755 → 0.515 (−0.240) | neighbours became much less alike |
| `ring_dmean`, `contrast_change` | +32.1, +27.5 | the ground brightened, the house even more |

## Design choices and alternatives

| Choice | Why | Alternative |
|---|---|---|
| Hand-crafted features | explainable; fits the classical-ML scope | CNN features (better accuracy, less explainable, forbidden here) |
| Pre/post change features | damage = change; reduces roof-type effects | post-only features |
| Texture via GLCM + Canny | well-established, cheap | Gabor filters, local binary patterns |
| One fixed set of 37 features | same input for both models, a fair comparison | per-model feature selection |

**Feature-set experiments** (called "Phase 3b" during development, so their files are in
`artifacts/experiments/phase3b/`) tested reduced feature sets:

- `change_only`: 17 change, texture-delta, ring and shape features;
- `scene_normalised`: each feature minus its scene median;
- `change_only_scene_normalised`: both.

They lowered the event signature (event-prediction accuracy 0.829 → as low as 0.374), but did **not**
improve cross-event macro-F1. They are analysis only; the served models use all 37 features. Their saved
results stay in `artifacts/experiments/phase3b/` and the API (`/results/experiments`), but the app no longer
shows them.

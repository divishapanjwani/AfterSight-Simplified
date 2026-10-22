# AfterSight feature dictionary (schema v1)

Each building is described by **37 numbers** computed from its pixels in the pre- and post-disaster
images. Code: `backend/app/features/extract.py`. Settings: `config/features.yaml`.

**Which pixels.** A building's pixels are its exact footprint blob from the pre-disaster mask (the same
blob that defines the building; see `data/README.md`). The **context ring** is every pixel within
`ring_width_px = 8` px of the footprint (8-neighbour growth) that is not part of *any* building.

**Grey value.** `gray = 0.299 R + 0.587 G + 0.114 B` on the 0–255 scale.

**Why mostly region statistics.** xBD pre/post pairs can be shifted by a few pixels. Means, standard
deviations and proportions over a region barely change under a small shift. Pixel-by-pixel differences
do, so only `gray_absdiff_mean` and `ring_absdiff_mean` are pixel-wise, and they are labelled
"misalignment-sensitive" below.

**Not features.** These columns are in the table for joins and analysis, but never go into a model:
`sample_id`, `building_uid`, `scene_id`, `event`, `disaster_type`, `xbd_split`, the label, and the
quality flags `label_purity`, `mixed_label`, `touches_border` and `ring_px`.

## Colour (12)

| Feature | Formula (plain English) | Why it might indicate damage |
|---|---|---|
| `pre_mean_r` | Average red value of the footprint pixels, pre-disaster | Baseline roof colour; lets the model judge post values relative to it |
| `pre_mean_g` | Average green, pre | as above |
| `pre_mean_b` | Average blue, pre | as above |
| `pre_std_r` | Standard deviation of red inside the footprint, pre | Baseline roof uniformity |
| `pre_std_g` | Std of green, pre | as above |
| `pre_std_b` | Std of blue, pre | as above |
| `post_mean_r` | Average red, post | Debris, ash, mud or exposed timber change roof colour |
| `post_mean_g` | Average green, post | Vegetation or algae on wrecked roofs, mud |
| `post_mean_b` | Average blue, post | Tarps (blue) on damaged roofs, water |
| `post_std_r` | Std of red, post | Rubble is patchy, while intact roofs are uniform |
| `post_std_g` | Std of green, post | as above |
| `post_std_b` | Std of blue, post | as above |

## Change (8)

| Feature | Formula (plain English) | Why it might indicate damage |
|---|---|---|
| `abs_dmean_r` | \|post mean red − pre mean red\| | Size of the colour change on the roof |
| `abs_dmean_g` | \|post − pre\| mean green | as above |
| `abs_dmean_b` | \|post − pre\| mean blue | as above |
| `dstd_r` | post std red − pre std red (signed) | Positive = the roof became patchier (debris, holes) |
| `dstd_g` | post − pre std green | as above |
| `dstd_b` | post − pre std blue | as above |
| `gray_absdiff_mean` | Average of \|post grey − pre grey\| over footprint pixels (misalignment-sensitive) | Large, local change anywhere on the roof |
| `brightness_ratio` | post grey mean ÷ max(pre grey mean, 1) | Burnt or flooded roofs darken (< 1); exposed concrete or debris can brighten (> 1) |

## Texture (12)

`*_delta` = post value − pre value.

| Feature | Formula (plain English) | Why it might indicate damage |
|---|---|---|
| `gray_std_pre` | Std of grey inside the footprint, pre | Baseline texture |
| `gray_std_post` | Std of grey, post | Rubble is rough |
| `gray_std_delta` | post − pre grey std | Increase = new roughness |
| `edge_density_pre` | Share of footprint pixels marked as edges by Canny (σ = 1.5, run on the footprint + margin) | Baseline structure (roof ridges, panels) |
| `edge_density_post` | Same, post | Broken roofs and debris create many short edges |
| `edge_density_delta` | post − pre edge density | Increase = fragmentation; decrease = a roof replaced by flat ash or water |
| `glcm_contrast_pre` | Mean of (i − j)² over neighbouring pixel pairs inside the footprint (grey quantised to 32 levels; right and down neighbours) | Baseline local contrast |
| `glcm_contrast_post` | Same, post | Rubble has high neighbour-to-neighbour contrast |
| `glcm_contrast_delta` | post − pre | Increase = rougher surface |
| `glcm_homogeneity_pre` | Mean of 1 / (1 + (i − j)²) over the same pairs (1 = perfectly smooth) | Baseline smoothness |
| `glcm_homogeneity_post` | Same, post | Intact roofs stay smooth |
| `glcm_homogeneity_delta` | post − pre | Decrease = surface broke up |

Note: the σ = 1.5 smoothing before Canny hides texture finer than about 2 px (1 px stripes give no
edges). The GLCM features work at the single-pixel scale and pick that texture up.

## Context ring (3)

| Feature | Formula (plain English) | Why it might indicate damage |
|---|---|---|
| `ring_dmean` | post ring grey mean − pre ring grey mean | Surroundings changed: flood water, burn scar, debris field. Separates event-wide change from building change |
| `ring_absdiff_mean` | Average \|post − pre\| grey over ring pixels (misalignment-sensitive) | Strength of change around the building |
| `contrast_change` | (post building − post ring) − (pre building − pre ring), grey means | Did the building change *relative to* its surroundings? Large values point to damage to the building itself rather than a lighting change |

## Shape (2)

These do not depend on damage. They help the model normalise: a small hut and a warehouse with the
same colour change are not equally likely to be destroyed.

| Feature | Formula (plain English) | Why it might indicate damage |
|---|---|---|
| `area_px` | Number of pixels in the footprint blob (exact) | Context: building size. Heavy right tail (largest = 648,034 px, a mall complex); scaling matters |
| `compactness` | 4π × area ÷ perimeter² of the footprint polygon (1 = circle, 0.785 = square, → 0 = thin or irregular) | Context: very irregular blobs are often several merged buildings, whose label is less reliable |

## Missing values

If a building has no ground pixels in its ring (fully enclosed by other buildings), the three ring
features are left missing (NaN), never filled with a made-up value. They are imputed in Phase 3,
with the imputer fitted on the training split only. In the current xBD table there are **0** such
buildings (minimum ring = 70 px), and no feature has NaN or infinite values.

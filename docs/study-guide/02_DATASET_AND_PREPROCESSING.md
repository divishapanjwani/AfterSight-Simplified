# 02 · Dataset and preprocessing

## Source

- **xBD** (Gupta *et al.*, CVPR Workshops 2019) is a public dataset of pre- and post-disaster satellite image
  pairs with building annotations and a four-level damage scale. The imagery targets below 0.8 m ground
  sample distance. Licence: CC BY-NC-SA 4.0.
- We use a **third-party Parquet copy of xBD's training split**: 17 files `train-000NN-of-00017.parquet`,
  7.9 GB. It was obtained from a shared Google Drive (see `data/README.md`). Because it holds only the
  training split, **the official xBD test and hold-out sets are not used**. All our train, validation and
  test sets are carved out of the training split.
- The copy contains **10 events and 2,799 image pairs**, all 1024 × 1024 RGB.

## Schema (one row = one image pair)

| Column | Meaning |
|---|---|
| `t1_image` | pre-disaster image (PNG bytes) |
| `t2_image` | post-disaster image (PNG bytes) |
| `t1_mask` | footprint mask: 1 = building pixel, 0 = background |
| `t2_mask` | damage mask: 0 background, **1 no damage, 2 minor, 3 major, 4 destroyed** |
| `image_name` | e.g. `train/images/santa-rosa-wildfire_00000006`, which gives the event and scene number |

A full scan of all 2,799 pairs found only the damage values 0–4 (`inspection_report.json` →
`mask_checks.unexpected_damage_values_px = {}`). There are **no polygons, building IDs or coordinates** in
this copy.

## The five events used

They were chosen in an earlier project phase to cover four hazard types with many severely damaged
buildings. It was a design decision, not a formal rule; for example, Santa Rosa has only 55 major-damage
buildings. Source: `config/events.yaml`.

| Event | Hazard | Cached pairs | No damage | Minor | Major | Destroyed |
|---|---|---:|---:|---:|---:|---:|
| Hurricane Harvey | flooding | 319 | 9,463 | 2,251 | 7,034 | 373 |
| Hurricane Matthew | wind | 238 | 2,103 | 4,440 | 1,140 | 1,887 |
| Hurricane Michael | wind | 343 | 13,870 | 4,998 | 1,841 | 720 |
| Palu tsunami | tsunami | 113 | 7,360 | 1 | 200 | 1,596 |
| Santa Rosa wildfire | fire | 226 | 8,653 | 76 | 55 | 3,279 |
| **Total** | | **1,239** | **41,449** | **11,766** | **10,270** | **7,855** |

There are 71,340 buildings in total, and **58%** are undamaged. That is class imbalance.

## From masks to buildings (`backend/app/datasets/mask_buildings.py` → `buildings_from_masks`)

1. **Find blobs.** Connected components of the footprint mask with **4-connectivity**: pixels touch only
   through edges, not corners (`scipy.ndimage.label`). Each blob is one building. 4-connectivity merges
   fewer touching buildings than 8-connectivity would.
2. **Label by majority vote.** Read the damage-mask values under the blob; the most frequent value
   (1–4) is the label. *Purity* is the share of pixels that agree.
3. **Drop and log, never guess.** A blob is dropped if:
   - it is under 16 px (`too_small`, 54 blobs);
   - it has no damage value at all (`label_missing`, 1,396 blobs, probably xBD's "un-classified");
   - its vote is tied (`label_tie`, 1 blob).

   1,451 of 72,791 blobs were dropped in total.
4. **Make a polygon** with `cv2.findContours` → Shapely, in image pixel coordinates, for drawing.
   2,674 polygons needed a geometric validity repair.
5. **Give it an ID:** `{scene_id}_b{index:04d}`, e.g. `santa-rosa-wildfire_00000006_b0002`.

**Known side effects:**

- Touching buildings can **merge into one blob**: 782 blobs contain more than one damage value, and the
  majority vote gives each a single label.
- There is no geography, so areas are in pixels, never m².

## Why 1,239 pairs but only 1,179 scenes in the splits

| Group | Pairs | Why |
|---|---:|---|
| No building blob at all | 52 | the footprint mask is empty |
| Blobs, but all dropped | 8 | all 28 blobs had no damage value (`label_missing`) |
| At least one labelled building | **1,179** | these are the scenes in both split manifests |
| **Total** | **1,239** | |

Verified from `artifacts/inspection/xbd/scenes.csv`, the drop reasons in `data/processed/scenes/<id>/scene.json`
and both manifests. Hurricane Michael has 343 cached pairs, of which **333** contain buildings. That is why
the cross-event test set has 333 scenes.

## Data validation done by the inspector (`scripts/inspect_dataset.py`)

- Every pair is complete and readable; image sizes and modes are counted.
- Every damage-mask value is accounted for.
- Footprint sizes, mixed blobs and drop reasons are logged.
- **Alignment check:** phase correlation estimates the pre/post shift. 168 pairs exceed 8 px and are
  flagged, not removed. Real damage also moves the estimate, so a flag means "check", not "wrong".
- Results go to `artifacts/inspection/xbd/inspection_report.json`, `scenes.csv` and
  `mixed_label_blobs.csv`.

## Splits and leakage prevention (`backend/app/training/splits.py`)

**Rule: whole scenes go to one split.** Buildings in one image share lighting, sensor and often damage. If
neighbouring houses landed in both training and test, the test would be partly "seen".

| Design | Method | Train | Validation | Test |
|---|---|---|---|---|
| **Within-event** | `StratifiedGroupKFold` (20 folds, seed 42) combined to about 70/15/15, stratified by class, grouped by scene | 821 scenes / 49,929 buildings | 181 / 10,705 | 177 / 10,706 |
| **Cross-event** | all Hurricane Michael scenes with buildings = test; the other 4 events split about 85/15 | 716 / 42,425 | 130 / 7,486 | 333 / 21,429 |

**Safeguards:**

- `validate_manifest` refuses to save a manifest in which any scene appears in two splits. Tests include
  `test_no_scene_overlap_in_either_design` and `test_real_manifests_have_zero_scene_overlap`.
- Each manifest stores a **hash** (`7bba8a9fbbd2740a` within-event, `dd7d8566b7dbaf7a` cross-event). Models
  record the hash they were trained with, and refuse to load if it changes.
- **Preprocessing is fitted on the training split only**, inside the scikit-learn pipeline. Validation and
  test never influence the scaler.
- **The test set is used once:** `TestSetGuard` raises an error on a second test evaluation of the same model.

## Preprocessing of the features (shared by KNN and the DT)

```
SkewLog1p (log(1+x) on 7 features with training skewness > 2)
   → SimpleImputer(median)
   → StandardScaler (z = (x − mean_train) / std_train)
   → model
```

- **The 7 log-transformed features:** `abs_dmean_r`, `abs_dmean_g`, `abs_dmean_b`, `gray_absdiff_mean`,
  `glcm_contrast_pre`, `glcm_contrast_post`, `area_px`. Area has a very long tail, up to 648,034 px.
- **Imputation:** the feature table has **no missing values**, so imputation changes nothing on this data. It
  is there in case a building has no visible ground ring.
- **Why scale?** KNN measures distances, so a feature in the hundreds (area) would otherwise drown one
  between 0 and 1 (homogeneity). The DT does not need scaling, because its splits are unaffected by
  monotone rescaling, but sharing one pipeline keeps the comparison simple.

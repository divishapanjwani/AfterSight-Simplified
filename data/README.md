# data/

Git-ignored (only this README is tracked). Holds xBD and everything derived from it.

| Folder | Contents |
|---|---|
| `raw/xbd_parquet/` | the xBD train split as a Hugging Face-style Parquet mirror — never modified |
| `processed/scenes/` | per-scene cache for the **configured events only** (see below) |
| `processed/` (Phase 2) | feature tables (Parquet) |
| `splits/` (Phase 2) | AfterSight's train/val/test manifests, grouped by scene/event |

## 1. The data in use

`data/raw/xbd_parquet/` holds 17 shards, `train-00000-of-00017.parquet` … `train-00016-of-00017.parquet`
(8.0 GB). Together they contain 2,799 rows, one per pre/post scene pair of the xBD **train** split.
The mirror was obtained from a shared Google Drive, not downloaded from xview2.org by AfterSight.

Schema, verified by reading the files on 2026-10-22:

| Column | Type | Content |
|---|---|---|
| `t1_image` | struct{bytes, path} | pre-disaster image, PNG, 1024×1024 RGB (`path` is null) |
| `t2_image` | struct{bytes, path} | post-disaster image, PNG, 1024×1024 RGB |
| `t1_mask` | struct{bytes, path} | building footprint mask, PNG, single channel: 0 background, 1 building |
| `t2_mask` | struct{bytes, path} | damage mask, PNG, single channel: 0 background, 1 no-damage, 2 minor, 3 major, 4 destroyed |
| `image_name` | string | e.g. `train/images/socal-fire_00000106` → event `socal-fire`, scene `socal-fire_00000106` |

There are **no JSON label files and no polygons** in this mirror. The original xBD tiles are also
available as folders with JSON polygons. AfterSight can still read that layout
(`dataset_format: xbd_json`), but it is only exercised by the synthetic test fixture at the moment.

## 2. How building footprints are derived

Because there are no polygons, buildings are reconstructed from the masks
(`backend/app/datasets/mask_buildings.py`):

1. **Instances:** connected components of `t1_mask > 0`, using 4-connectivity (pixels join only through
   shared edges). `t1_mask` was chosen over `t2_mask > 0` after a full scan. The two agree closely
   (median per-scene overlap 0.9994), but 1,738,125 footprint pixels have no damage value. Using
   `t2_mask` would silently hide those buildings. Using `t1_mask` lets us count and exclude them.
2. **Label:** the majority `t2_mask` value inside the blob.
   - Blobs with more than one value are logged in `mixed_label_blobs.csv`.
   - A tie is dropped (`label_tie`).
   - A blob with no damage value is dropped (`label_missing`).
   - Any value other than 1–4 is dropped (`label_unknown`) and counted, never mapped to a class.
     The full scan of all 2,799 scenes found only the values 0–4.
3. **Size:** blobs under `min_area_px = 16` are dropped (`too_small`). The size histogram shows a
   natural gap there: 87 blobs are under 8 px, none fall between 8 and 16 px, and counts rise smoothly
   after that.
4. **Polygon:** the blob outline is traced with OpenCV (holes kept), and vertices are placed at pixel
   centres, in image pixel coordinates. Self-touching outlines are fixed with Shapely `make_valid`.
   A blob with no polygon area (e.g. a 1 px wide line) is dropped (`degenerate_shape`). The building's
   `area_px` is its exact pixel count, not the polygon area.
5. **ID:** `{scene_id}_b{index:04d}`. The index is the blob number in raster order (top-left first),
   so it is deterministic.

## 3. Limitations

- **Footprints are derived from rasterised xBD annotation masks, not the original JSON polygons.**
- **Merged buildings:** buildings that touch in the mask merge into one blob and cannot be separated.
  The largest blob in the configured events is 185,291 px. 474 blobs there contain more than one damage
  value, and the majority vote gives each of them a single label.
- **No original identifiers or geography:** xBD building UIDs and lng/lat coordinates are not available,
  so nothing is drawn on a world map and no area is reported in m² or km².
- **`un-classified` buildings** appear only as footprints with no damage value. They are excluded and
  counted as `label_missing`, not identified by name. If the mask creator mapped `un-classified` to a
  damage value, that cannot be detected from the masks.
- **Border buildings:** buildings cut by the tile edge are kept but flagged (`touches_border`).
- **Image alignment:** the pre/post pair may be slightly misaligned. The optional phase-correlation check
  (`--alignment`) flags pairs over 8 px. Real damage also moves this estimate, so a flag means "look at
  this pair", not "misregistered".
- **Disaster type** is not stored in the mirror. Any type shown per event comes from config and is
  labelled as such.
- **Train split only:** the mirror has only xBD's train split, so the test/holdout scenes are not available.

## 4. Inspect / validate

```zsh
source backend/.venv/bin/activate
python scripts/inspect_dataset.py --alignment --no-cache   # full scan, report + figures, no extraction
python scripts/inspect_dataset.py --alignment              # same, and fill the scene cache
python scripts/inspect_dataset.py --limit-per-event 20     # quick look
```

A full scan of the 17 shards took about 96 s on this laptop, with a measured peak RAM of about 1.3 GB.
RAM is bounded by one Parquet row group (100 scenes, about 310 MB of PNG bytes), not by the dataset size.

| Output | What it holds |
|---|---|
| `artifacts/inspection/xbd/inspection_report.json` | totals, class counts, all-events table, mask checks, footprint sizes, drop reasons, disk & RAM |
| `artifacts/inspection/xbd/scenes.csv` | one row per configured scene: readability, counts, mask overlap, alignment shift, issues |
| `artifacts/inspection/xbd/dropped_buildings.csv` | every blob not used, with its reason |
| `artifacts/inspection/xbd/mixed_label_blobs.csv` | every blob with more than one damage value: value counts, chosen value, purity |
| `artifacts/inspection/xbd/class_counts.csv` | counts per event × xBD split × class |
| `outputs/figures/inspection/xbd/` | class distribution (configured events, all events), footprint sizes, sample pairs |

## 5. Scene cache (`data/processed/scenes/`)

The configured events are extracted one folder per scene, so later phases can open single scenes
without scanning 500 MB shards. Each folder holds:

- `pre.png` and `post.png`: the original bytes, not re-encoded;
- the two masks;
- `pre_thumb.jpg` and `post_thumb.jpg`: 256 px previews;
- `scene.json`: the derived footprints with IDs, labels, purity and WKT, plus the source shard and row.

Scenes of other events are never written. After changing `events`, the inspector reports any stale
folders but does not delete them. This takes about 3.5 MB per scene, measured on real scenes.

## 6. Synthetic fixtures (tests only, not xBD)

`python scripts/make_fixture.py` writes two small, deterministic datasets:

- `backend/tests/fixtures/xbd_synthetic/`: JSON-polygon layout;
- `backend/tests/fixtures/xbd_synthetic_parquet/`: two shards in the mirror's schema, with
  hand-placed edge cases.

Their events are named `synthetic-*`, their figures are watermarked, and their outputs go to
`*/synthetic*` folders. They are never presented as real data.

## Licence and attribution

xBD is distributed by the xView2 organisers under CC BY-NC-SA 4.0 (check the current terms on
<https://xview2.org/dataset>). This Parquet mirror is a third-party repackaging. AfterSight does not
redistribute it, and the original licence still applies. Non-commercial academic use only. Please cite:

> Gupta, R., Hosfelt, R., Sajeev, S., Patel, N., Goodman, B., Doshi, J., Heim, E., Choset, H., & Gaston, M.
> (2019). *xBD: A Dataset for Assessing Building Damage from Satellite Imagery.* CVPR Workshops.
> arXiv:1911.09296.

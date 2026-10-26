"""Collect REAL values and figures for docs/UNDERSTANDING_AFTERSIGHT.md.

Traces one building through every pipeline step using the project's own code, and renders the teaching
figures (class crops, algorithm illustrations on real data). Run from the repo root:
    .venv/bin/python docs/build/understanding_data.py
Writes docs/build/understanding_data.json and docs/figures/understanding/*.png
"""

import io
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from PIL import Image
from scipy import ndimage
from shapely import wkt as shapely_wkt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.core.config import get_settings  # noqa: E402
from app.core.store import Store, StorePaths  # noqa: E402
from app.datasets import xbd_parquet  # noqa: E402
from app.explain.explain import decision_path, knn_neighbours, lr_contributions  # noqa: E402
from app.features.schema import FEATURE_COLUMNS  # noqa: E402
from app.training.persistence import load_model  # noqa: E402
from app.training.splits import load_manifest  # noqa: E402

OUT = ROOT / "docs" / "figures" / "understanding"
OUT.mkdir(parents=True, exist_ok=True)
SCENE = "santa-rosa-wildfire_00000006"
UID = f"{SCENE}_b0002"
DMG = {"no-damage": "#c9a8d8", "minor-damage": "#9a64b5", "major-damage": "#6b3590", "destroyed": "#3d1559"}
CLASSES = list(DMG)
INK = "#1b1b1a"

store = Store(StorePaths.from_settings(get_settings()))
doc = store.scene_doc(SCENE)
b = next(x for x in doc["buildings"] if x["building_id"] == UID)
shard, row = doc["source"]["shard"], doc["source"]["row"]

# ---------------------------------------------------------------- 1. the Parquet row
cells = xbd_parquet.read_cells(ROOT / "data/raw/xbd_parquet" / shard, row)
pre_img = Image.open(io.BytesIO(cells["t1_image"]))
t1 = xbd_parquet.decode_mask(cells["t1_mask"])
t2 = xbd_parquet.decode_mask(cells["t2_mask"])
pre = np.asarray(pre_img.convert("RGB"))
post = np.asarray(Image.open(io.BytesIO(cells["t2_image"])).convert("RGB"))
v1, c1 = np.unique(t1, return_counts=True)
v2, c2 = np.unique(t2, return_counts=True)

# ---------------------------------------------------------------- 2. the blob and its label
labels, n_blobs = ndimage.label(t1 > 0, structure=ndimage.generate_binary_structure(2, 1))
index = int(UID.rsplit("_b", 1)[1])
window = ndimage.find_objects(labels)[index - 1]
blob = labels == index
inside = t2[blob]
votes = Counter(int(v) for v in inside[inside > 0])
geom = shapely_wkt.loads(b["wkt"])
n_vertices = sum(len(p.exterior.coords) for p in getattr(geom, "geoms", [geom]))

# ---------------------------------------------------------------- 3. features, split, scaling, predictions
table = store.feature_table
i = table.ids.index[table.ids["sample_id"] == UID][0]
x = table.X.loc[[i]]
feat = {k: float(v) for k, v in x.iloc[0].items()}
within = load_manifest(ROOT / "data/splits/within_event_v1.json")
cross = load_manifest(ROOT / "data/splits/cross_event_v1.json")


def saved_model(name):
    """Load a saved model straight from its artifact (not through the app's served-model list): this guide also
    uses models the app does not serve (the earlier logistic regression, the research random forest)."""
    return load_model(store.model_dir() / f"{name}.joblib", expected_manifest_hash=store.manifest()["manifest_hash"])


lr, lr_meta = saved_model("logistic_regression")
knn, _ = saved_model("knn")
dt, _ = saved_model("decision_tree")
rf, _ = saved_model("random_forest")
z = lr[:-1].transform(x)[0]
z_of = {f: float(z[j]) for j, f in enumerate(FEATURE_COLUMNS)}
logged = [FEATURE_COLUMNS[j] for j in lr.named_steps["log"].log_columns_]
scaler = lr.named_steps["scale"]
proba = {m: {c: round(float(p), 4) for c, p in zip(pipe.classes_, pipe.predict_proba(x)[0])} for m, pipe in
         (("logistic_regression", lr), ("knn", knn), ("decision_tree", dt), ("random_forest", rf))}
thresholds = store.review_thresholds()["models"]
contrib = lr_contributions(lr, x, top_k=6)
neigh = knn_neighbours(knn, x, store.knn_index())
path = decision_path(dt, x)
analysis_files = sorted(store.paths.analysis_runs.glob("*.json"))
lr_analysis = next(json.loads(f.read_text()) for f in analysis_files
                   if json.loads(f.read_text())["scene_id"] == SCENE and json.loads(f.read_text())["model"]["name"] == "logistic_regression")
lr_b = next(x_ for x_ in lr_analysis["buildings"] if x_["building_uid"] == UID)

KEY = ["pre_mean_r", "post_mean_r", "abs_dmean_r", "gray_absdiff_mean", "brightness_ratio", "gray_std_pre", "gray_std_post",
       "gray_std_delta", "edge_density_pre", "edge_density_post", "glcm_homogeneity_pre", "glcm_homogeneity_post",
       "glcm_homogeneity_delta", "ring_dmean", "contrast_change", "area_px", "compactness"]
scene_ids = table.ids["scene_id"] == SCENE

data = {
    "scene": SCENE, "uid": UID, "event": doc["event"], "disaster_type": store.disaster_type(doc["event"]),
    "parquet": {"shard": shard, "row": row, "image_name": f"train/images/{SCENE}",
                "bytes": {k: len(v) for k, v in cells.items()}, "image_size": list(pre_img.size), "image_mode": pre_img.mode,
                "t1_values": {int(a): int(b_) for a, b_ in zip(v1, c1)}, "t2_values": {int(a): int(b_) for a, b_ in zip(v2, c2)}},
    "blobs": {"n_blobs": int(n_blobs), "n_kept": len(doc["buildings"]), "n_dropped": len(doc["dropped"]),
              "dropped_reasons": dict(Counter(d["reason"] for d in doc["dropped"]))},
    "building": {"index": index, "bbox_rows": [window[0].start, window[0].stop], "bbox_cols": [window[1].start, window[1].stop],
                 "pixel_count": int(blob.sum()), "area_px_scene_json": b["area_px"], "votes": dict(votes),
                 "label": b["damage"], "purity": b["label_purity"], "mixed": b["mixed_label"], "touches_border": b["touches_border"],
                 "polygon_vertices": n_vertices, "polygon_area": round(geom.area, 1), "wkt_start": b["wkt"][:80],
                 "pre_pixel_example": pre[(window[0].start + window[0].stop) // 2, (window[1].start + window[1].stop) // 2].tolist(),
                 "post_pixel_example": post[(window[0].start + window[0].stop) // 2, (window[1].start + window[1].stop) // 2].tolist()},
    "features": {k: round(feat[k], 4) for k in KEY},
    "n_features": len(feat),
    "scene_building_count": int(scene_ids.sum()),
    "split": {"within_event": within["scenes"][SCENE], "cross_event": cross["scenes"][SCENE]},
    "scaling": {"logged_features": logged,
                "examples": {f: {"raw": round(feat[f], 4), "after_log1p": round(float(np.log1p(feat[f])) if f in logged else feat[f], 4),
                                 "train_mean": round(float(scaler.mean_[FEATURE_COLUMNS.index(f)]), 4),
                                 "train_std": round(float(scaler.scale_[FEATURE_COLUMNS.index(f)]), 4), "z": round(z_of[f], 3)}
                             for f in ["gray_std_delta", "glcm_homogeneity_delta", "area_px", "gray_absdiff_mean"]}},
    "proba": proba, "thresholds": {m: v["threshold"] for m, v in thresholds.items()},
    "lr": {"C": lr_meta["hyperparameters"]["C"], **contrib},
    "knn": neigh, "dt": path,
    "analysis": {"analysis_id": lr_analysis["analysis_id"], "predicted": lr_b["predicted"], "confidence": lr_b["confidence"],
                 "needs_review": lr_b["needs_review"], "scene_metrics": lr_analysis["scene_metrics"]["macro_f1"],
                 "scene_accuracy": lr_analysis["scene_metrics"]["accuracy"], "counts": lr_analysis["counts"]},
}

# ---------------------------------------------------------------- confusion matrix for worked metric examples
m = json.loads((ROOT / "artifacts/models/within_event/four_class/logistic_regression.json").read_text())
data["cm_lr_within"] = m["metrics"]["test"]["confusion_matrix"]
data["metrics_lr_within"] = {k: m["metrics"]["test"][k] for k in ("accuracy", "macro_f1", "weighted_f1", "per_class", "n")}

# ---------------------------------------------------------------- figures
def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# (a) the traced building: pre / post crop with outline, plus masks
r0, r1 = window[0].start, window[0].stop
c0, c1_ = window[1].start, window[1].stop
pad = 40
rs, re_ = max(0, r0 - pad), min(1024, r1 + pad)
cs, ce = max(0, c0 - pad), min(1024, c1_ + pad)
fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
for ax, (img, title) in zip(axes, [(pre, "pre-disaster image"), (post, "post-disaster image"), (t1, "footprint mask (t1)"), (t2, "damage mask (t2)")]):
    crop = img[rs:re_, cs:ce]
    if crop.ndim == 2:
        ax.imshow(crop, cmap="gray" if "t1" in title else "magma", vmin=0, vmax=1 if "t1" in title else 4, interpolation="nearest")
    else:
        ax.imshow(crop)
    for p in getattr(geom, "geoms", [geom]):
        xs, ys = p.exterior.xy
        ax.plot(np.array(xs) - cs, np.array(ys) - rs, color="#00e5ff", lw=1.4)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
fig.suptitle(f"{UID}: the same 1024×1024 scene, cropped around the building (cyan = derived polygon)", fontsize=10.5)
save(fig, "trace_building.png")

# (b) what the four classes look like: real crops from held-out test scenes
# one class per event where that class is common, so the examples are typical rather than rare cases
PREFER = {"no-damage": "hurricane-michael", "minor-damage": "hurricane-matthew", "major-damage": "hurricane-harvey", "destroyed": "santa-rosa-wildfire"}
picked = {}
test_scenes = sorted(s for s, sp in within["scenes"].items() if sp == "test")
for cls, event in PREFER.items():
    for sid in (s for s in test_scenes if s.startswith(event)):
        hit = next((bb for bb in store.scene_doc(sid)["buildings"] if bb["damage"] == cls and bb["label_purity"] == 1.0
                    and 1200 <= bb["area_px"] <= 6000 and not bb["touches_border"]), None)
        if hit:
            picked[cls] = (sid, hit)
            break
fig, axes = plt.subplots(2, 4, figsize=(12, 6.3))
crops_meta = {}
for j, cls in enumerate(CLASSES):
    sid, bb = picked[cls]
    g = shapely_wkt.loads(bb["wkt"])
    minx, miny, maxx, maxy = g.bounds
    half = max(maxx - minx, maxy - miny) / 2 + 30
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    box = (int(max(0, cx - half)), int(max(0, cy - half)), int(min(1024, cx + half)), int(min(1024, cy + half)))
    folder = ROOT / "data/processed/scenes" / sid
    for i_, phase in enumerate(("pre", "post")):
        im = Image.open(folder / f"{phase}.png").convert("RGB").crop(box)
        ax = axes[i_, j]
        ax.imshow(im)
        for p in getattr(g, "geoms", [g]):
            xs, ys = p.exterior.xy
            ax.plot(np.array(xs) - box[0], np.array(ys) - box[1], color="#00e5ff", lw=1.3)
        ax.set_xticks([]); ax.set_yticks([])
        if i_ == 0:
            ax.set_title(cls, fontsize=12, fontweight="bold")
        if j == 0:
            ax.set_ylabel(f"{phase}-disaster", fontsize=11)
    crops_meta[cls] = {"scene": sid, "building": bb["building_id"], "area_px": bb["area_px"]}
fig.suptitle("The four damage classes from above (real buildings from held-out test scenes; cyan = footprint)", fontsize=11)
save(fig, "damage_classes.png")
data["class_examples"] = crops_meta

# (c) RGB channels of a small patch
patch = pre[r0:r0 + 6, c0:c0 + 6]
fig, axes = plt.subplots(1, 4, figsize=(11, 3))
axes[0].imshow(patch, interpolation="nearest"); axes[0].set_title("6×6 pixels (pre image)")
for k, (ax, name, cmap) in enumerate(zip(axes[1:], ["red", "green", "blue"], ["Reds", "Greens", "Blues"])):
    ax.imshow(patch[:, :, k], cmap=cmap, vmin=0, vmax=255, interpolation="nearest")
    for (yy, xx), val in np.ndenumerate(patch[:, :, k]):
        ax.text(xx, yy, str(val), ha="center", va="center", fontsize=7, color="black" if val < 150 else "white")
    ax.set_title(f"{name} channel (0–255)")
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
save(fig, "rgb_channels.png")
data["rgb_patch_first_pixel"] = patch[0, 0].tolist()

# (d) baseline: glcm_homogeneity_delta by class with the real thresholds
split = table.ids["scene_id"].map(within["scenes"])
train = (split == "train").to_numpy()
base = json.loads((ROOT / "artifacts/models/within_event/four_class/baseline.json").read_text())["hyperparameters"]
fig, ax = plt.subplots(figsize=(9, 3.4))
vals = table.X.loc[train, base["feature"]].to_numpy()
ys = table.y[train].to_numpy()
bins = np.linspace(np.percentile(vals, 1), np.percentile(vals, 99), 60)
for cls in CLASSES:
    ax.hist(vals[ys == cls], bins=bins, histtype="step", lw=2, color=DMG[cls], label=cls, density=True)
for t in base["thresholds_raw"]:
    ax.axvline(-t, color="#e8590c", ls="--", lw=1.4)
ax.set_xlabel(f"{base['feature']} (training buildings)")
ax.set_ylabel("share of buildings")
ax.legend(fontsize=8, frameon=False)
ax.set_title("Baseline: three thresholds (orange) on one feature → four classes", fontsize=10.5)
save(fig, "baseline_thresholds.png")
data["baseline"] = base

# (e) KNN illustration on two real features (the model itself uses all 37)
rng = np.random.default_rng(0)
sample = rng.choice(np.flatnonzero(train), size=1500, replace=False)
fx, fy = "glcm_homogeneity_post", "glcm_homogeneity_delta"
fig, ax = plt.subplots(figsize=(7.2, 5))
for cls in CLASSES:
    sel = sample[table.y.iloc[sample].to_numpy() == cls]
    ax.scatter(table.X[fx].iloc[sel], table.X[fy].iloc[sel], s=8, color=DMG[cls], label=cls, alpha=0.7)
nb = table.ids["sample_id"].isin([n["sample_id"] for n in neigh["neighbours"]]).to_numpy()
ax.scatter(table.X.loc[nb, fx], table.X.loc[nb, fy], s=60, facecolors="none", edgecolors="#e8590c", lw=1.6, label="its 15 neighbours (37-D)")
ax.scatter([feat[fx]], [feat[fy]], marker="*", s=320, color="#00b8d4", edgecolors=INK, label="our building", zorder=5)
ax.set_xlabel(fx); ax.set_ylabel(fy)
ax.legend(fontsize=8, frameon=False, loc="lower right")
ax.set_title("KNN: a new building takes the majority label of its nearest training buildings", fontsize=10)
save(fig, "knn_idea.png")

# (f) decision tree: real top of the tree
tree = json.loads((ROOT / "artifacts/models/within_event/four_class/decision_tree_structure.json").read_text())
from app.inference.analysis import MODEL_LABELS  # noqa: E402,F401

def draw(node, x, y, dx, ax, depth=0):
    if node.get("feature") and depth < 2:
        label = f"{node['feature']}\n≤ {node['threshold_raw']:.3f}?\nn={node['samples']:,}"
    else:
        label = f"majority: {node['majority']}\nn={node['samples']:,}"
    ax.add_patch(FancyBboxPatch((x - 1.15, y - 0.42), 2.3, 0.84, boxstyle="round,pad=0.03", fc="#efece5" if node.get("feature") and depth < 2 else DMG.get(node["majority"], "#ddd"), ec=INK))
    ax.text(x, y, label, ha="center", va="center", fontsize=7.5, color="white" if node["majority"] in ("major-damage", "destroyed") and not (node.get("feature") and depth < 2) else INK)
    if node.get("feature") and depth < 2:
        for side, sx, txt in (("left", -1, "yes"), ("right", 1, "no")):
            nx, ny = x + sx * dx, y - 1.5
            ax.plot([x, nx], [y - 0.42, ny + 0.42], color=INK, lw=1)
            ax.text((x + nx) / 2 + 0.1 * sx, (y + ny) / 2, txt, fontsize=8, color="#6b6a66")
            draw(node[side], nx, ny, dx / 2, ax, depth + 1)

fig, ax = plt.subplots(figsize=(11, 4.6))
ax.set_xlim(-6, 6); ax.set_ylim(-3.6, 0.9); ax.axis("off")
draw(tree["root"], 0, 0.2, 3.1, ax)
ax.set_title(f"Top of the real Decision Tree (depth {tree['max_depth']}, {tree['n_leaves']} leaves in total; first 2 levels shown)", fontsize=10)
save(fig, "tree_top.png")

# (g) logistic regression: softmax probabilities for our building + the sigmoid idea
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.3))
s = np.linspace(-6, 6, 200)
a1.plot(s, 1 / (1 + np.exp(-s)), color="#2a78d6", lw=2)
a1.set_xlabel("score z = w·x + b"); a1.set_ylabel("probability"); a1.set_title("sigmoid: turns a score into 0…1", fontsize=10)
a1.grid(alpha=0.3)
p = proba["logistic_regression"]
a2.barh(CLASSES, [p[c] for c in CLASSES], color=[DMG[c] for c in CLASSES])
for k, c in enumerate(CLASSES):
    a2.text(p[c] + 0.01, k, f"{p[c]:.2f}", va="center", fontsize=9)
a2.set_xlim(0, 1.1); a2.set_title(f"softmax output for {UID.split('_')[-1]}", fontsize=10)
save(fig, "lr_idea.png")

# (h) K-Means on the real PCA sample
km = json.loads((ROOT / "artifacts/analyses/kmeans/within_event.json").read_text())
pts = km["per_k"]["4"]["pca_points"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
cols = ["#2a78d6", "#1baf7a", "#eda100", "#e87ba4"]
for c in range(4):
    sel = [q for q in pts if q["cluster"] == c]
    a1.scatter([q["x"] for q in sel], [q["y"] for q in sel], s=6, color=cols[c], label=f"cluster {c}")
for cls in CLASSES:
    sel = [q for q in pts if q["label"] == cls]
    a2.scatter([q["x"] for q in sel], [q["y"] for q in sel], s=6, color=DMG[cls], label=cls)
a1.set_title("coloured by K-Means cluster (k = 4)", fontsize=10); a2.set_title("same points coloured by damage label", fontsize=10)
for a in (a1, a2):
    a.legend(fontsize=7, frameon=False, markerscale=2); a.set_xticks([]); a.set_yticks([])
fig.suptitle("K-Means groups buildings by look, not by damage (2-D PCA of training buildings, visualisation only)", fontsize=10.5)
save(fig, "kmeans_idea.png")

# (i) split idea: real scene counts per split
fig, ax = plt.subplots(figsize=(9, 2.2))
counts = within["counts"]
left = 0
for name, col in (("train", "#2a78d6"), ("val", "#eda100"), ("test", "#1baf7a")):
    w = counts[name]["scenes"]
    ax.barh([0], [w], left=left, color=col, edgecolor="white")
    ax.text(left + w / 2, 0, f"{name}\n{w} scenes\n{counts[name]['buildings']:,} buildings", ha="center", va="center", fontsize=9, color="white")
    left += w
ax.axis("off")
ax.set_title("Within-event split: whole scenes go to one split (≈70/15/15)", fontsize=10)
save(fig, "split_idea.png")

data["split_counts"] = {d: {s: {"scenes": v["scenes"], "buildings": v["buildings"]} for s, v in mm["counts"].items()} for d, mm in (("within_event", within), ("cross_event", cross))}
(ROOT / "docs/build/understanding_data.json").write_text(json.dumps(data, indent=1, default=float), encoding="utf-8")
print(json.dumps({k: data[k] for k in ("parquet", "blobs", "building", "split", "proba", "thresholds")}, indent=1, default=float))

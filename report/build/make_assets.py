"""Collect everything the mini-project report needs from SAVED artifacts (no numbers are typed by hand).

Writes report/assets/*.png (pipeline diagram, git graph, copies of existing figures) and
report/build/data.json (all table values). Run from the repo root:  .venv/bin/python report/build/make_assets.py
"""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.core.config import get_settings  # noqa: E402
from app.core.store import Store, StorePaths  # noqa: E402
from app.features.schema import FEATURE_GROUPS  # noqa: E402
from app.inference import feature_lab  # noqa: E402

ASSETS = ROOT / "report" / "assets"
ART = ROOT / "artifacts"
FIG = ROOT / "outputs" / "figures"
ASSETS.mkdir(parents=True, exist_ok=True)


def read(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- data
fmeta = read(ROOT / "data/processed/features/features_v1.meta.json")
index = read(ROOT / "data/processed/scenes/index.json")
inspection = read(ART / "inspection/xbd/inspection_report.json")
events_cfg = __import__("yaml").safe_load((ROOT / "config/events.yaml").read_text())

per_event = []
for ev in sorted(fmeta["buildings_by_event"]):
    scenes = [s for s in index["scenes"] if s["event"] == ev]
    cc = inspection["class_counts"]["by_event"][ev]
    per_event.append({"event": ev, "type": events_cfg["disaster_types"][ev], "scenes": len(scenes),
                      "empty": sum(1 for s in scenes if s["n_buildings"] == 0), "buildings": fmeta["buildings_by_event"][ev], **cc})

# feature dictionary rows parsed from docs/FEATURES.md (single source of truth for descriptions)
features_md = (ROOT / "docs/FEATURES.md").read_text(encoding="utf-8")
desc = {}
for m in re.finditer(r"^\| `([a-z0-9_]+)` \| (.+?) \| (.+?) \|$", features_md, flags=re.M):
    desc[m.group(1)] = {"formula": m.group(2).replace("\\|", "|"), "why": m.group(3).replace("\\|", "|")}
group_of = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}
feature_rows = [{"name": f, "group": group_of[f], "formula": desc[f]["formula"], "why": desc[f]["why"]}
                for g in FEATURE_GROUPS for f in FEATURE_GROUPS[g]]

comparisons = {f"{d}/{s}": read(ART / f"metrics/{d}/{s}/comparison.json")
               for d in ("within_event", "cross_event") for s in ("four_class", "three_class")}
model_meta = {f"{d}/{m}": read(ART / f"models/{d}/four_class/{m}.json")
              for d in ("within_event", "cross_event") for m in ("baseline", "logistic_regression", "knn", "decision_tree", "random_forest")}
models_view = {}
for k, m in model_meta.items():
    models_view[k] = {"hyperparameters": m["hyperparameters"], "train_seconds": m["train_seconds"],
                      "infer_ms_per_1000": round(1000 * m["inference_seconds_per_1000_test"], 2),
                      "val": {"accuracy": m["metrics"]["val"]["accuracy"], "macro_f1": m["metrics"]["val"]["macro_f1"]},
                      "test": {"accuracy": m["metrics"]["test"]["accuracy"], "macro_f1": m["metrics"]["test"]["macro_f1"],
                               "weighted_f1": m["metrics"]["test"]["weighted_f1"],
                               "per_class": m["metrics"]["test"]["per_class"]},
                      "per_event": {e: v["macro_f1"] for e, v in m["metrics"]["test_per_event"].items()},
                      "log1p": m["preprocessing"].get("log1p_features")}
confounding = read(ART / "analyses/confounding/within_event.json")
kmeans = read(ART / "analyses/kmeans/within_event.json")
phase3b = read(ART / "experiments/phase3b/summary.json")
review = read(ART / "models/within_event/four_class/review_thresholds.json")
demo = read(ART / "demo/demo_scenes.json")
manifests = {d: read(ROOT / f"data/splits/{d}_v1.json") for d in ("within_event", "cross_event")}
training_cfg = __import__("yaml").safe_load((ROOT / "config/training.yaml").read_text())
store = Store(StorePaths.from_settings(get_settings()))
lab = feature_lab.summary(store)

reqs = [l.strip() for l in (ROOT / "backend/requirements.txt").read_text().splitlines() if "==" in l and not l.startswith("#")]
pkg = read(ROOT / "frontend/package.json")
commits = subprocess.run(["git", "log", "--reverse", "--format=%h|%ad|%s", "--date=short"], cwd=ROOT, capture_output=True, text=True).stdout.strip().splitlines()

data = {
    "inspection": {"shards": len(inspection["parquet_shards"] or []), "rows_all_events": sum(e["scenes"] for e in inspection["all_events"].values()),
                   "events_all": inspection["all_events"], "mask_checks": inspection["mask_checks"], "totals": inspection["totals"],
                   "drop_reasons": inspection["drop_reasons"], "footprint_size": inspection["footprint_size_px"],
                   "peak_rss_mb": round(inspection["resources"]["peak_rss_bytes_measured"] / 1e6), "raw_mb": round(inspection["resources"]["raw_bytes_selected_events"] / 1e6),
                   "alignment_flagged": len(inspection["alignment"]["scenes_flagged"])},
    "features_meta": {k: fmeta[k] for k in ("n_buildings", "n_scenes", "class_counts", "buildings_by_event", "nan_counts", "inf_counts", "timing", "table_bytes")},
    "per_event": per_event, "feature_rows": feature_rows,
    "comparisons": comparisons, "models": models_view, "confounding": {k: confounding[k] for k in ("hyperparameters", "majority_class", "majority_class_accuracy", "chance_accuracy_uniform")} | {"accuracy": confounding["test_metrics"]["accuracy"], "macro_f1": confounding["test_metrics"]["macro_f1"]},
    "kmeans": {"chosen": kmeans["chosen_k_by_silhouette"], "elbow": kmeans["elbow_k_heuristic"], "per_k": {k: {"silhouette": v["silhouette_sampled"], "ari": v["adjusted_rand_index"], "sizes": v["cluster_sizes"]} for k, v in kmeans["per_k"].items()}},
    "phase3b": phase3b["rows"], "phase3b_variants": phase3b["variants"],
    "review": review, "demo": demo["scenes"],
    "splits": {d: {"hash": m["manifest_hash"], "label": m["label"], "counts": m["counts"]} for d, m in manifests.items()},
    "grids": {k: v.get("grid") for k, v in training_cfg["models"].items()} | {"rf_trees": training_cfg["models"]["random_forest"]["n_estimators"]},
    "feature_lab_top": [{"feature": f["feature"], "eta2": f["separation_eta2"], "group": f["group"],
                         "median": {c: f["per_class"][c]["median"] for c in f["per_class"]}} for f in lab["features"][:8]],
    "feature_lab_bottom": [{"feature": f["feature"], "eta2": f["separation_eta2"]} for f in lab["features"][-3:]],
    "requirements": reqs, "npm": {**pkg["dependencies"], **pkg["devDependencies"]},
    "commits": [dict(zip(("hash", "date", "subject"), c.split("|", 2))) for c in commits],
}
(ROOT / "report/build/data.json").write_text(json.dumps(data, indent=1), encoding="utf-8")

# ---------------------------------------------------------------- figures copied from the pipeline outputs
copies = {
    "fig_class_distribution.png": FIG / "inspection/xbd/class_distribution.png",
    "fig_footprint_sizes.png": FIG / "inspection/xbd/footprint_sizes.png",
    "fig_sample_fire.png": FIG / "inspection/xbd/sample_santa-rosa-wildfire_00000138.png",
    "fig_cm_lr_within.png": FIG / "models/within_event/four_class/logistic_regression_confusion_test.png",
    "fig_cm_lr_cross.png": FIG / "models/cross_event/four_class/logistic_regression_confusion_test.png",
    "fig_cm_rf_within.png": FIG / "models/within_event/four_class/random_forest_confusion_test.png",
    "fig_confounding.png": FIG / "models/confounding/within_event_event_confusion_test.png",
    "fig_kmeans_curves.png": FIG / "models/kmeans/within_event/kmeans_elbow_silhouette.png",
    "fig_kmeans_contingency.png": FIG / "models/kmeans/within_event/kmeans_contingency_k4.png",
}
for name, src in copies.items():
    shutil.copy(src, ASSETS / name)

# ---------------------------------------------------------------- pipeline diagram
INK, MUTED, RULE, ACCENT = "#1b1b1a", "#4a4945", "#b9b3a6", "#6b3590"
fm = data["features_meta"]
stages = [
    ("1  xBD Parquet mirror", f"{data['inspection']['shards']} shards · {data['inspection']['rows_all_events']:,} scene pairs\npre/post images + 2 masks"),
    ("2  Masks → buildings", "connected components of t1_mask\nlabel = majority t2_mask value"),
    ("3  Scene cache", f"{fm['n_scenes']:,} scenes of 5 events\nPNGs, thumbnails, polygons"),
    ("4  Feature extraction", f"37 features × {fm['n_buildings']:,} buildings\ncolour · change · texture · ring · shape"),
    ("5  Scene-grouped splits", "within-event 70/15/15\ncross-event: michael held out"),
    ("6  Model training", "baseline · LR · KNN · DT · RF\ntuned on val · test once"),
    ("7  Artifacts", "joblib pipelines + metadata JSON\nmetrics · explanations · figures"),
    ("8  FastAPI backend", "file-based store · cached analyses\nexplain · CSV/PDF · review queue"),
    ("9  React web app", "Triage Desk · Case File · Arena\nFeature Lab · Cluster Atlas · Reports"),
]
fig, ax = plt.subplots(figsize=(11, 6.2))
ax.set_xlim(0, 11.0)
ax.set_ylim(0, 6.2)
ax.axis("off")
w, h = 3.3, 1.25
pos = [(0.2, 4.55), (3.85, 4.55), (7.5, 4.55), (7.5, 2.55), (3.85, 2.55), (0.2, 2.55), (0.2, 0.55), (3.85, 0.55), (7.5, 0.55)]
for (title, body), (x, y) in zip(stages, pos):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc="#fcfcfb", ec=INK, lw=1.3))
    ax.add_patch(FancyBboxPatch((x, y + h - 0.36), w, 0.36, boxstyle="round,pad=0.02,rounding_size=0.08", fc="#efece5", ec=INK, lw=1.3))
    ax.text(x + 0.12, y + h - 0.18, title, fontsize=11, fontweight="bold", va="center", color=INK, family="serif")
    ax.text(x + 0.12, y + (h - 0.36) / 2, body, fontsize=8.8, va="center", color=MUTED, family="serif", linespacing=1.35)
order = list(range(len(pos)))
for a, b in zip(order, order[1:]):
    (x1, y1), (x2, y2) = pos[a], pos[b]
    if y1 == y2:
        start, end = ((x1 + w, y1 + h / 2), (x2, y2 + h / 2)) if x2 > x1 else ((x1, y1 + h / 2), (x2 + w, y2 + h / 2))
    else:
        start, end = (x1 + w / 2, y1), (x2 + w / 2, y2 + h)
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16, lw=1.6, color=ACCENT))
ax.text(5.5, 6.0, "AfterSight pipeline: from xBD Parquet shards to the web app", ha="center", fontsize=13, fontweight="bold", color=INK, family="serif")
fig.savefig(ASSETS / "fig_pipeline.png", dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---------------------------------------------------------------- git graph rendered as an image
graph = subprocess.run(["git", "log", "--graph", "--oneline", "--decorate=short", "--date-order"], cwd=ROOT, capture_output=True, text=True).stdout.rstrip().splitlines()
lines = [l if len(l) <= 110 else l[:107] + "..." for l in graph]
fig = plt.figure(figsize=(11, 0.34 * len(lines) + 0.9))
fig.patch.set_facecolor("#1c1b19")
fig.text(0.02, 1 - 0.35 / (0.34 * len(lines) + 0.9), "$ git log --graph --oneline --decorate", family="monospace", fontsize=10.5, color="#9a988f", va="top")
for i, line in enumerate(lines):
    fig.text(0.02, 1 - (0.75 + 0.34 * i) / (0.34 * len(lines) + 0.9), line, family="monospace", fontsize=10.5, color="#ecebe6", va="top")
fig.savefig(ASSETS / "fig_git_graph.png", dpi=180, facecolor=fig.get_facecolor())
plt.close(fig)

def side_by_side(left: str, right: str, out: str) -> None:
    a, b = Image.open(ASSETS / left).convert("RGB"), Image.open(ASSETS / right).convert("RGB")
    h = max(a.height, b.height)
    canvas = Image.new("RGB", (a.width + b.width + 40, h), "white")
    canvas.paste(a, (0, (h - a.height) // 2))
    canvas.paste(b, (a.width + 40, (h - b.height) // 2))
    canvas.save(ASSETS / out)


side_by_side("fig_cm_lr_within.png", "fig_cm_lr_cross.png", "fig_cm_lr_pair.png")

sizes = {p.name: Image.open(p).size for p in sorted(ASSETS.glob("*.png"))}
(ROOT / "report/build/image_sizes.json").write_text(json.dumps(sizes, indent=1), encoding="utf-8")
print(f"assets: {len(sizes)} images · data.json with {len(feature_rows)} feature rows, {len(per_event)} events, {len(data['commits'])} commits")

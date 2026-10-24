"""Pick 3 held-out demo scenes and pre-cache everything the demo opens, so it runs instantly offline.

Choice (deterministic, test split only): 25-160 buildings, at least 3 ground-truth damage classes,
ranked by how evenly the classes are mixed (normalised entropy); best scene per disaster type, top 3 types.
For each scene and every trained serving model: analysis, view JPEGs, CSV and PDF are written to disk.

Usage (repo root): source .venv/bin/activate && python scripts/prepare_demo.py
Writes artifacts/demo/demo_scenes.json.
"""

import json
import math
import sys
from datetime import datetime, timezone

import _bootstrap  # noqa: F401
from app.core.config import get_settings
from app.core.errors import AppError
from app.core.store import Store, StorePaths
from app.inference import images, scenes
from app.inference.analysis import run_analysis
from app.reports.reports import pdf_name, report_context, write_csv, write_pdf


def evenness(counts: dict[str, int]) -> float:
    total = sum(counts.values())
    probs = [n / total for n in counts.values() if n]
    return -sum(p * math.log(p) for p in probs) / math.log(len(counts)) if total else 0.0


def main() -> int:
    store = Store(StorePaths.from_settings(get_settings()))
    try:
        table = scenes.summaries(store)
    except AppError as exc:
        print(f"Not ready: {exc.message} Fix: {exc.fix}", file=sys.stderr)
        return 1
    pool = table[(table["split"] == "test") & table["n_buildings"].between(25, 160)].copy()
    classes = ["no-damage", "minor-damage", "major-damage", "destroyed"]
    pool["n_classes"] = (pool[classes] > 0).sum(axis=1)
    pool = pool[pool["n_classes"] >= 3]
    pool["evenness"] = [evenness({c: int(r[c]) for c in classes}) for _, r in pool.iterrows()]
    best = pool.sort_values(["evenness", "scene_id"], ascending=[False, True]).groupby("disaster_type").head(1)
    chosen = best.sort_values("evenness", ascending=False).head(3)
    if len(chosen) < 3:
        print(f"Only {len(chosen)} scenes meet the demo criteria.", file=sys.stderr)

    models = [m for m in store.serving.models if store.model_meta(m) is not None]
    demo = []
    for _, row in chosen.iterrows():
        sid = row["scene_id"]
        for phase in ("pre", "post"):
            for size in ("thumb", "view"):
                images.scene_image(store, sid, phase, size)
        entry = {"scene_id": sid, "event": row["event"], "disaster_type": row["disaster_type"], "n_buildings": int(row["n_buildings"]),
                 "ground_truth_counts": {c: int(row[c]) for c in classes}, "evenness": round(float(row["evenness"]), 3), "analyses": {}}
        for model in models:
            analysis = run_analysis(store, sid, model)
            csv_path = store.paths.reports / f"{analysis['analysis_id']}.csv"
            pdf_path = store.paths.reports / pdf_name(analysis["analysis_id"])  # same versioned name the API serves
            if not csv_path.is_file():
                write_csv(analysis, csv_path)
            if not pdf_path.is_file():
                write_pdf(analysis, pdf_path, report_context(store))
            m = analysis["scene_metrics"] or {}
            entry["analyses"][model] = {"analysis_id": analysis["analysis_id"], "scene_macro_f1": m.get("macro_f1"),
                                        "scene_accuracy": m.get("accuracy"),
                                        "suggested_for_review": sum(b["needs_review"] for b in analysis["buildings"])}
        demo.append(entry)
        lr = entry["analyses"].get("logistic_regression", {})
        print(f"{sid:<32} {row['disaster_type']:<9} {int(row['n_buildings']):>4} buildings  GT {entry['ground_truth_counts']}  "
              f"LR scene macro-F1 {lr.get('scene_macro_f1')}")
    out = store.paths.artifacts / "demo"
    out.mkdir(parents=True, exist_ok=True)
    (out / "demo_scenes.json").write_text(json.dumps({"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                                      "split": "test", "models": models, "scenes": demo}, indent=1), encoding="utf-8")
    print(f"-> artifacts/demo/demo_scenes.json ({len(demo)} scenes x {len(models)} models pre-cached)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

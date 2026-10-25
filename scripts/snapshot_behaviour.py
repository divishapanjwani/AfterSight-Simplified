"""Snapshot what the API returns, so a refactor can be proven to change nothing.

Records, through the real FastAPI routes and the saved models (nothing is trained):
  * a fresh analysis of every demo scene with every served model (predictions, confidences, scores, flags);
  * the /explain response for every building in those analyses;
  * /features/summary and a fixed set of /features/scatter requests;
  * /models (the table on the Readiness page).

Analyses are written to a temporary folder, never to artifacts/analyses/runs, so the run cache is untouched.

Usage (repo root):
  .venv/bin/python scripts/snapshot_behaviour.py save  <out.json>
  .venv/bin/python scripts/snapshot_behaviour.py check <out.json>     # exits 1 on any difference
"""

import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import _bootstrap  # noqa: F401
from fastapi.testclient import TestClient

from app.api.deps import get_store
from app.core.config import get_settings
from app.core.store import Store, StorePaths
from app.main import app

# fields that legitimately differ between two runs (timestamps, timings, cache state)
VOLATILE = {"created_at", "seconds", "cached"}
SCATTER_PAIRS = [("glcm_homogeneity_delta", "glcm_homogeneity_post"), ("area_px", "gray_std_delta"),
                 ("abs_dmean_r", "glcm_contrast_post"), ("brightness_ratio", "compactness")]


@dataclass(frozen=True)
class ScratchRunsPaths(StorePaths):
    """The real data, models and artifacts, but analyses are cached in a temporary folder."""
    runs: Path = Path()

    @property
    def analysis_runs(self) -> Path:
        return self.runs


def strip_volatile(value):
    if isinstance(value, dict):
        return {k: strip_volatile(v) for k, v in value.items() if k not in VOLATILE}
    if isinstance(value, list):
        return [strip_volatile(v) for v in value]
    return value


def take_snapshot() -> dict:
    real = StorePaths.from_settings(get_settings())
    demo = json.loads((real.artifacts / "demo" / "demo_scenes.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as tmp:
        paths = ScratchRunsPaths(real.data, real.artifacts, real.outputs, real.events_config, real.training_config, Path(tmp))
        store = Store(paths)
        app.dependency_overrides[get_store] = lambda: store
        client = TestClient(app)
        snap: dict = {"analyses": {}, "explanations": {}, "scatter": {}}
        # the models the app serves now (from config via /api/models), not the list frozen in demo_scenes.json
        served = [m["name"] for m in client.get("/api/models").json()["models"] if m["trained"]]
        for scene in demo["scenes"]:
            for model in served:
                response = client.post("/api/analysis", json={"scene_id": scene["scene_id"], "model_name": model})
                response.raise_for_status()
                analysis = response.json()
                key = f"{scene['scene_id']}|{model}"
                snap["analyses"][key] = strip_volatile(analysis)
                for building in analysis["buildings"]:
                    uid = building["building_uid"]
                    r = client.get(f"/api/analysis/{analysis['analysis_id']}/buildings/{uid}/explain")
                    r.raise_for_status()
                    snap["explanations"][f"{key}|{uid}"] = r.json()
        snap["models"] = client.get("/api/models").json()
        snap["feature_summary"] = client.get("/api/features/summary").json()
        for x, y in SCATTER_PAIRS:
            snap["scatter"][f"{x}|{y}"] = client.get("/api/features/scatter", params={"x": x, "y": y}).json()
        app.dependency_overrides.clear()
    return snap


def main() -> int:
    mode, out = sys.argv[1], Path(sys.argv[2])
    snap = take_snapshot()
    counts = f"{len(snap['analyses'])} analyses, {len(snap['explanations'])} explanations, " \
             f"{len(snap['feature_summary']['features'])} summary features, {len(snap['scatter'])} scatter samples"
    if mode == "save":
        out.write_text(json.dumps(snap, sort_keys=True), encoding="utf-8")
        print(f"saved {counts} -> {out}")
        return 0
    before = json.loads(out.read_text(encoding="utf-8"))
    after = json.loads(json.dumps(snap, sort_keys=True))
    return compare(before, after, counts)


def compare(before: dict, after: dict, counts: str) -> int:
    """Every entry present in BOTH snapshots must be identical. Entries only in the baseline (e.g. analyses of a
    model that is no longer served) are reported, not failed; entries only in the new snapshot are a failure."""
    problems, dropped = [], []
    for section in ("analyses", "explanations", "scatter"):
        b, a = before[section], after[section]
        problems += [f"{section}:{k} changed" for k in b.keys() & a.keys() if b[k] != a[k]]
        problems += [f"{section}:{k} is new" for k in a.keys() - b.keys()]
        dropped += [f"{section}:{k}" for k in b.keys() - a.keys()]
    if before["feature_summary"] != after["feature_summary"]:
        problems.append("feature_summary changed")
    bm = {m["name"]: m for m in before["models"]["models"]}
    am = {m["name"]: m for m in after["models"]["models"]}
    problems += [f"models:{n} changed" for n in bm.keys() & am.keys() if bm[n] != am[n]]
    problems += [f"models:{n} is new" for n in am.keys() - bm.keys()]
    dropped_models = sorted(bm.keys() - am.keys())
    rest = lambda d: {k: v for k, v in d["models"].items() if k != "models"}  # noqa: E731
    if rest(before) != rest(after):
        problems.append("models: top-level fields changed")
    if problems:
        print(f"DIFFERENT ({len(problems)}), e.g. {problems[:5]}")
        return 1
    if dropped or dropped_models:
        gone = sorted({k.split("|")[1] for k in dropped if k.startswith(("analyses:", "explanations:"))})
        print(f"IDENTICAL for every shared entry ({counts}); no longer produced: {len(dropped)} entries "
              f"(models {', '.join(gone) or '-'}), /models entries removed: {', '.join(dropped_models) or '-'}")
        return 0
    print(f"IDENTICAL: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

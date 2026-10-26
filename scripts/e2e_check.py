"""End-to-end check on a SYNTHETIC deployment (never touches real data or artifacts).

1. Builds the synthetic Parquet mirror and runs the real pipeline on it (inspect -> features -> splits
   -> train KNN/DT -> review thresholds) in a temporary folder.
2. Starts the backend on that folder and the built frontend (vite preview) proxied to it, on free ports.
3. For every served model (KNN, Decision Tree): analysis, explanation, CSV and PDF. Unserved models
   (Logistic Regression, Random Forest) must give a clean 404 model_not_found.
4. Checks the frontend serves the app and reaches the API through its proxy.

Usage (repo root):  source .venv/bin/activate && python scripts/e2e_check.py [--skip-build]
Exit code 0 only if every check passed.
"""

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import _bootstrap  # noqa: F401
from app.core.config import REPO_ROOT
from app.datasets.synthetic_deployment import build_deployment

EXPLAIN_KIND = {"knn": "nearest_neighbours", "decision_tree": "decision_path"}
NOT_SERVED = ["logistic_regression", "random_forest"]  # LR is not a damage model of the app; RF is not served


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def request(url: str, body: dict | None = None) -> tuple[int, bytes, str]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read(), r.headers.get("content-type", "")
    except urllib.error.HTTPError as e:
        return e.code, e.read(), e.headers.get("content-type", "")


def wait_for(url: str, seconds: float = 40) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            if request(url)[0] == 200:
                return True
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(0.25)
    return False


class Checks:
    def __init__(self) -> None:
        self.rows: list[tuple[bool, str, str]] = []

    def check(self, ok: bool, name: str, detail: str = "") -> bool:
        self.rows.append((bool(ok), name, detail))
        print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  — ' + detail if detail else ''}", flush=True)
        return bool(ok)

    @property
    def failed(self) -> int:
        return sum(not ok for ok, _, _ in self.rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-build", action="store_true", help="reuse frontend/dist instead of running npm run build")
    args = parser.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix="aftersight-e2e-"))
    procs: list[subprocess.Popen] = []
    c = Checks()
    try:
        print(f"Building synthetic deployment in {workdir} (NOT xBD data)…", flush=True)
        t = time.perf_counter()
        build_deployment(workdir)
        c.check(True, "synthetic pipeline: inspect → features → splits → train → thresholds", f"{time.perf_counter() - t:.1f} s")

        api_port, web_port = free_port(), free_port()
        env = {**os.environ,
               "AFTERSIGHT_DATA_DIR": str(workdir / "data"), "AFTERSIGHT_ARTIFACTS_DIR": str(workdir / "artifacts"),
               "AFTERSIGHT_OUTPUTS_DIR": str(workdir / "outputs"), "AFTERSIGHT_EVENTS_CONFIG": str(workdir / "config" / "events.yaml"),
               "AFTERSIGHT_TRAINING_CONFIG": str(workdir / "config" / "training.yaml"),
               "AFTERSIGHT_FRONTEND_ORIGIN": f"http://localhost:{web_port}", "AFTERSIGHT_API_URL": f"http://localhost:{api_port}"}
        logs = workdir / "logs"
        logs.mkdir()
        procs.append(subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(api_port)], cwd=REPO_ROOT / "backend",
                                      env=env, stdout=(logs / "api.log").open("w"), stderr=subprocess.STDOUT))
        frontend = REPO_ROOT / "frontend"
        if not args.skip_build or not (frontend / "dist" / "index.html").is_file():
            subprocess.run(["npm", "run", "build"], cwd=frontend, check=True, capture_output=True)
        procs.append(subprocess.Popen(["npx", "vite", "preview", "--port", str(web_port), "--strictPort"], cwd=frontend,
                                      env=env, stdout=(logs / "web.log").open("w"), stderr=subprocess.STDOUT))

        api = f"http://localhost:{api_port}/api"
        if not c.check(wait_for(f"{api}/health"), "backend starts", api):
            return 1
        status = json.loads(request(f"{api}/dataset/status")[1])
        tiles = status["tiles"]
        c.check(all(tiles[t]["ready"] for t in ("scene_cache", "features", "splits", "models", "review_thresholds")), "readiness tiles ready")
        scenes = json.loads(request(f"{api}/scenes?split=test&has_buildings=true")[1])
        c.check(scenes["total"] > 0 and all(s["split"] == "test" for s in scenes["items"]), "test scenes listed", f"{scenes['total']} test scenes")
        scene = scenes["items"][0]["scene_id"]

        served = [m["name"] for m in json.loads(request(f"{api}/models")[1])["models"]]  # from config/training.yaml
        c.check(served == list(EXPLAIN_KIND), "served models are KNN and Decision Tree", ", ".join(served))
        for model in served:
            t = time.perf_counter()
            code, body, _ = request(f"{api}/analysis", {"scene_id": scene, "model_name": model})
            analysis = json.loads(body) if code == 200 else {}
            c.check(code == 200 and analysis.get("buildings"), f"{model}: analysis", f"{len(analysis.get('buildings', []))} buildings, {1000 * (time.perf_counter() - t):.0f} ms")
            if code != 200:
                continue
            aid, uid = analysis["analysis_id"], analysis["buildings"][0]["building_uid"]
            code, body, _ = request(f"{api}/analysis/{aid}/buildings/{uid}/explain")
            c.check(code == 200 and json.loads(body)["kind"] == EXPLAIN_KIND[model], f"{model}: explanation", EXPLAIN_KIND[model])
            code, body, ctype = request(f"{api}/analysis/{aid}/export.csv")
            rows = body.decode().strip().splitlines()
            c.check(code == 200 and ctype.startswith("text/csv") and len(rows) == 1 + len(analysis["buildings"]), f"{model}: CSV export", f"{len(rows) - 1} rows")
            code, body, ctype = request(f"{api}/analysis/{aid}/report.pdf")
            c.check(code == 200 and body.startswith(b"%PDF"), f"{model}: PDF report", f"{len(body):,} bytes")

        for model in NOT_SERVED:
            code, body, _ = request(f"{api}/analysis", {"scene_id": scene, "model_name": model})
            err = json.loads(body).get("error", {})
            c.check(code == 404 and err.get("code") == "model_not_found" and b"Traceback" not in body, f"{model}: not served, clean 404")
        code, body, _ = request(f"{api}/scenes/..%2F..%2Fetc%2Fpasswd")
        c.check(code in (404, 422) and b"root:" not in body and b"Traceback" not in body, "path traversal rejected cleanly")

        web = f"http://localhost:{web_port}"
        if c.check(wait_for(web), "frontend serves the app", web):
            code, body, _ = request(f"{web}/triage/{scene}")
            c.check(code == 200 and b'id="root"' in body, "SPA route served (deep link)")
            code, body, _ = request(f"{web}/api/health")
            c.check(code == 200 and json.loads(body)["status"] == "ok", "frontend reaches the API through its proxy")
            code, body, ctype = request(f"{web}/api/scenes/{scene}/image/post?size=thumb")
            c.check(code == 200 and ctype == "image/jpeg", "images through the proxy are JPEG")
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
        shutil.rmtree(workdir, ignore_errors=True)  # the temporary folder this script created

    print(f"\n{len(c.rows) - c.failed}/{len(c.rows)} checks passed")
    return 0 if c.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

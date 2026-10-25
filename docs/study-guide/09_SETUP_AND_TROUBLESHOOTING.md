# 09 · Setup and troubleshooting

## Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Python | **3.11** | the project venv must use 3.11 (`backend/requirements.txt` pins scikit-learn 1.9.1, FastAPI 0.153, …) |
| Node.js | 20+ | for the React frontend (`frontend/package.json` pins exact versions) |
| Data | xBD Parquet shards | only needed to *rebuild* everything; the clone already has the processed data, models and results |
| Optional | LibreOffice | only to export the paper or report to PDF |

## macOS (zsh)

```zsh
cd ~/AfterSight-Simplified
python3.11 -m venv .venv                         # once (already done in this clone)
source .venv/bin/activate
python -m pip install -r backend/requirements.txt
(cd frontend && npm ci)

./start.sh                                       # API on http://localhost:8020, app on http://localhost:5200
```

Open http://localhost:5200. Press Ctrl-C in the terminal to stop both servers.

Without `start.sh`, use two terminals:

```zsh
source .venv/bin/activate && cd backend && uvicorn app.main:app --port 8020      # terminal 1
cd frontend && npx vite --port 5200 --strictPort                                  # terminal 2
```

## Windows (PowerShell)

`start.sh` and `run_pipeline.sh` are **bash** scripts, so on Windows start the two servers in two PowerShell
windows.

```powershell
cd C:\path\to\AfterSight-Simplified
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install -r backend\requirements.txt
cd frontend; npm ci; cd ..

# window 1: API
.\.venv\Scripts\Activate.ps1; cd backend; python -m uvicorn app.main:app --port 8020
# window 2: web app
cd frontend; npx vite --port 5200 --strictPort
```

Run the backend tests on Windows **inside the venv**: `python -m pytest` from the repo root.

## Rebuilding from raw data (only if you must; it retrains every model)

```zsh
# put the 17 shards train-000NN-of-00017.parquet into data/raw/xbd_parquet/   (see data/README.md)
scripts/run_pipeline.sh            # about 10 minutes on an M-series laptop; replaces the saved results
```

In this clone `data/raw/xbd_parquet` is a **read-only symlink** to the original project's copy.

## Checks

```zsh
python -m pytest                          # 149 backend tests
(cd frontend && npx vitest run)           # 37 frontend tests
(cd frontend && npx tsc -b && npm run build)
python scripts/e2e_check.py               # 20 end-to-end checks
python scripts/snapshot_behaviour.py save verification/snapshot_before.json    # once, before any change
python scripts/snapshot_behaviour.py check verification/snapshot_before.json   # predictions unchanged?
```

## Common problems

| Symptom | Cause | Fix |
|---|---|---|
| `test_no_deep_learning_frameworks_installed` fails ("torch must not be installed") | You ran pytest with a **global** Python (e.g. 3.12) whose user site-packages contain PyTorch. The project imports no deep-learning library, and the test correctly spots torch in that environment. | Run inside the project venv (`.venv`), where `include-system-site-packages = false` and user site-packages are disabled. `python -s -m pytest` also hides user site-packages. Don't uninstall torch globally and don't skip the test. |
| `start.sh`: "Port 8020 is in use … stopping it" | Another copy (the original `~/aftersight`, or an earlier run) is using 8020 or 5200 | Expected: the script frees the ports. Run **one copy at a time**. To free a port by hand, use `lsof -ti tcp:8020 \| xargs kill` (macOS) or `Get-NetTCPConnection -LocalPort 8020 \| ForEach-Object { Stop-Process -Id $_.OwningProcess }` (PowerShell) |
| "Operation not permitted" when the project is in `~/Downloads`, `~/Desktop` or `~/Documents` on macOS | macOS privacy protection blocks the terminal or editor from those folders | Move the project to your home folder (e.g. `~/AfterSight-Simplified`), or allow the terminal or VS Code under *System Settings → Privacy & Security → Files and Folders* (or *Full Disk Access*) |
| `.venv/bin/activate` not found | The venv is not at the repo root | `python3.11 -m venv .venv` in the repo root |
| A page says **"Not available yet"** with a command | That artifact hasn't been built | Run the command shown (usually a script from `scripts/`) |
| `model_not_trained` (409) | No saved model | `python scripts/train.py` (retrains) |
| Readiness: **Review thresholds PARTIAL** | Some *required* served model has no usable validation threshold | `python scripts/calibrate_review.py` (validation only). Only the served models (LR and KNN) count. To check without overwriting the saved file: `python scripts/calibrate_review.py --out /tmp/calib_check` |
| `model_not_found` (404) for `decision_tree` or `random_forest` | The app serves only LR and KNN (`config/training.yaml` → `serving.models`) | Expected. DT results are in the Model Arena as a research comparison |
| `model_not_served` (404) opening an old analysis or its PDF | The analysis used a model the app no longer serves | Expected. The file stays in `artifacts/analyses/runs/`; run the scene again with LR or KNN |
| Model refuses to load (schema or scikit-learn version) | Different library version or feature schema | Install the pinned `backend/requirements.txt`, or retrain |
| Blank page | The frontend failed to start | Check `outputs/logs/web.log`; run `(cd frontend && npm ci)` |

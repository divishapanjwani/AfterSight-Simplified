# artifacts/

Git-ignored. Persisted outputs that the app reads; every number shown in the UI traces back to a file here.

| Folder | Contents |
|---|---|
| `models/` | fitted scikit-learn pipelines (`.joblib`) + metadata JSON |
| `metrics/` | validation/test metrics per model |
| `analyses/` | per-scene analysis runs (predictions + explanations) |

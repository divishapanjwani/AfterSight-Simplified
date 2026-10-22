"""Phase 3b experiment: do confounding-reduced feature sets transfer better to an unseen disaster?

Trains Logistic Regression and the Decision Tree on each feature variant, for both split designs,
tuned on validation, each new model evaluated on test exactly once. Everything is written under
artifacts/experiments/phase3b/ and outputs/figures/experiments/phase3b/ — the Phase 3 models and their
test results are never touched, replaced or re-evaluated.
"""

import json
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Callable

from app.core.config import REPO_ROOT
from app.evaluation.analysis import run_confounding
from app.evaluation.metrics import TestSetGuard
from app.features.table import FeatureTable
from app.features.variants import VARIANTS, variant_frame
from app.training.config import TrainingConfig
from app.training.run import OutputRoots, design_data, run_design

EXPERIMENT_LABEL = "Phase 3b experiment — confounding-reduced feature sets (separate from Phase 3; not app models)"
MODELS = ["logistic_regression", "decision_tree"]
EXPERIMENT_ROOT = REPO_ROOT / "artifacts" / "experiments" / "phase3b"
FIGURE_ROOT = REPO_ROOT / "outputs" / "figures" / "experiments" / "phase3b"


def roots_for(variant: str, artifacts_root=EXPERIMENT_ROOT, figures_root=FIGURE_ROOT) -> OutputRoots:
    return OutputRoots(models=artifacts_root / variant / "models", metrics=artifacts_root / variant / "metrics",
                       analyses=artifacts_root / variant / "analyses", figures=figures_root / variant)


def run_phase3b(cfg: TrainingConfig, table: FeatureTable, manifests: dict[str, dict], variants: list[str] | None = None,
                artifacts_root=EXPERIMENT_ROOT, figures_root=FIGURE_ROOT,
                say: Callable[[str], None] = print) -> dict[str, Any]:
    guard = TestSetGuard()  # this experiment's own test evaluations, one per new model
    rows: list[dict[str, Any]] = []
    for variant in variants or list(VARIANTS):
        X = variant_frame(variant, table.X, table.ids["scene_id"])
        variant_table = replace(table, X=X)
        roots = roots_for(variant, artifacts_root, figures_root)
        say(f"\n== {variant}: {VARIANTS[variant]} ({X.shape[1]} features)")
        event_accuracy = None
        for design in cfg.splits.designs:
            data = design_data(variant_table, manifests[design], "four_class", cfg.label_schemes["four_class"],
                               feature_set=f"phase3b:{variant}")
            results = run_design(data, cfg, _Namespaced(guard, variant), roots, MODELS, include_optional=False,
                                 say=lambda m: None if m.startswith("    ") else say(m))  # hide per-grid-point lines
            if design == "within_event":
                conf = run_confounding(data, cfg, _Namespaced(guard, variant), roots)
                event_accuracy = conf["test_metrics"]["accuracy"]
                say(f"   event-prediction accuracy (within-event test): {event_accuracy}")
            for r in results:
                meta = json.loads((roots.models / design / "four_class" / f"{r['model']}.json").read_text())
                rows.append({
                    "variant": variant, "n_features": X.shape[1], "design": design, "model": r["model"],
                    "val_macro_f1": r["val_macro_f1"], "test_macro_f1": r["test_macro_f1"],
                    "test_recall_major": r["test_recall"]["major-damage"], "test_recall_destroyed": r["test_recall"]["destroyed"],
                    "hyperparameters": r["hyperparameters"], "top_permutation_features": r["top_permutation_features"],
                    "per_event_test_macro_f1": {e: m["macro_f1"] for e, m in meta["metrics"]["test_per_event"].items()},
                })
        for row in rows:
            if row["variant"] == variant:
                row["event_prediction_accuracy_within_event"] = event_accuracy
    summary = {"label": EXPERIMENT_LABEL, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "variants": {v: VARIANTS[v] for v in (variants or VARIANTS)}, "models": MODELS,
               "test_evaluations": guard.used, "rows": rows}
    artifacts_root.mkdir(parents=True, exist_ok=True)
    (artifacts_root / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


class _Namespaced(TestSetGuard):
    """Prefix guard keys with 'phase3b/<variant>/' while sharing the parent's record of evaluations."""

    def __init__(self, parent: TestSetGuard, variant: str) -> None:
        self.parent, self.prefix = parent, f"phase3b/{variant}/"

    @property
    def used(self) -> dict[str, int]:  # type: ignore[override]
        return {k[len(self.prefix):]: v for k, v in self.parent.used.items() if k.startswith(self.prefix)}

    def evaluate(self, key: str, fn):  # type: ignore[override]
        return self.parent.evaluate(self.prefix + key, fn)

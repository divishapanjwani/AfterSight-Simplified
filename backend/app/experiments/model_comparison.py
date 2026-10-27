"""Model comparison & ensemble analysis (Model Arena only; nothing here is served in the Triage Desk).

For each split design (within-event, cross-event), 4-class:
  * SVM: the shared preprocessing + SVC(class_weight="balanced", probability=True), trained on a scene-grouped
    random sample of ~10,000 training buildings (SVC does not scale to ~50k buildings in our time budget);
  * soft-voting ensemble ("parallel ensemble"): the equal-weight average of predict_proba of the SVM, the saved
    Random Forest and the saved KNN (VotingClassifier over frozen, already-fitted models: nothing is refitted);
  * the saved KNN, Decision Tree and Random Forest are NOT re-evaluated: their test metrics are read from their
    saved metadata (each was evaluated on test once when it was trained).
Within-event only: GridSearchCV (GroupKFold by scene, macro-F1, preprocessing inside the pipeline, training split
only) for KNN, DT and SVM; each best configuration is refitted and evaluated once on the within-event test set as an
"optimised" version. Every new model is evaluated on each test set at most once (TestSetGuard).

Writes artifacts/experiments/model_comparison/summary.json (+ the SVM pipelines).
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import GridSearchCV, GroupKFold

from app.core.config import REPO_ROOT
from app.core.errors import NotFound, ResultsInvalid
from app.evaluation.metrics import TestSetGuard, classification_metrics
from app.training.config import TrainingConfig
from app.training.models import make_pipeline
from app.training.persistence import load_model
from app.training.run import DesignData

ROOT = REPO_ROOT / "artifacts" / "experiments" / "model_comparison"
SVM_SAMPLE = 10_000
SVM_DEFAULT = {"C": 1.0, "kernel": "rbf", "gamma": "scale"}
GRIDS = {
    "knn": {"model__n_neighbors": [5, 15, 31], "model__weights": ["uniform", "distance"]},
    "decision_tree": {"model__max_depth": [4, 6, 8, 12], "model__min_samples_leaf": [20, 100],
                      "model__class_weight": ["balanced"]},
    "svm": {"model__C": [0.1, 1, 10], "model__kernel": ["linear", "rbf"]},
}
ENSEMBLE_MEMBERS = ["svm", "random_forest", "knn"]
LABELS = {"knn": "KNN", "decision_tree": "Decision Tree", "svm": "SVM", "random_forest": "Random Forest",
          "soft_voting": "Soft Voting (SVM + RF + KNN)"}


def scene_sample(scene_ids: Any, target: int, seed: int) -> np.ndarray:
    """Row positions of whole training scenes, taken in a seeded random order until at least `target` buildings."""
    scene_ids = np.asarray(scene_ids)
    scenes = np.unique(scene_ids)
    np.random.default_rng(seed).shuffle(scenes)
    chosen, total = [], 0
    counts = {s: c for s, c in zip(*np.unique(scene_ids, return_counts=True))}
    for s in scenes:
        if total >= target:
            break
        chosen.append(s)
        total += counts[s]
    return np.flatnonzero(np.isin(scene_ids, chosen))


def summarise(y_true: Any, y_pred: Any, classes: list[str]) -> dict[str, Any]:
    """Accuracy, macro precision / recall / F1, per-class metrics and the confusion matrix (fixed class order)."""
    m = classification_metrics(y_true, y_pred, classes)
    present = [v for v in m["per_class"].values() if v["support"]]
    macro = lambda k: round(float(np.mean([v[k] for v in present])), 4)  # noqa: E731
    return {"n": m["n"], "accuracy": m["accuracy"], "macro_precision": macro("precision"),
            "macro_recall": macro("recall"), "macro_f1": m["macro_f1"], "per_class": m["per_class"],
            "confusion_matrix": m["confusion_matrix"]}


MODELS_ROOT = REPO_ROOT / "artifacts" / "models"


def _saved(models_root: Path, design: str, name: str, data: DesignData) -> tuple[Any, dict[str, Any]]:
    return load_model(models_root / design / "four_class" / f"{name}.joblib",
                      expected_manifest_hash=data.manifest_hash)


def _saved_entry(name: str, meta: dict[str, Any], classes: list[str]) -> dict[str, Any]:
    """A saved model's existing test metrics (from its metadata), in the same shape as a new evaluation."""
    t = meta["metrics"]["test"]
    present = [v for v in t["per_class"].values() if v["support"]]
    return {"model": name, "label": LABELS[name], "kind": "default", "source": "saved model (metadata; not re-evaluated)",
            "hyperparameters": meta["hyperparameters"],
            "metrics": {"n": t["n"], "accuracy": t["accuracy"],
                        "macro_precision": round(float(np.mean([v["precision"] for v in present])), 4),
                        "macro_recall": round(float(np.mean([v["recall"] for v in present])), 4),
                        "macro_f1": t["macro_f1"], "per_class": t["per_class"],
                        "confusion_matrix": t["confusion_matrix"]}}


def run_model_comparison(cfg: TrainingConfig, designs: dict[str, DesignData], root: Path = ROOT,
                         say: Callable[[str], None] = print, svm_sample: int = SVM_SAMPLE,
                         grids: dict[str, dict] | None = None, models_root: Path = MODELS_ROOT) -> dict[str, Any]:
    grids = grids or GRIDS
    guard = TestSetGuard()
    skew = float(cfg.preprocessing.get("skew_threshold", 2.0))
    out: dict[str, Any] = {"label": "Model comparison & ensemble analysis (Model Arena only; not served)",
                           "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "seed": cfg.seed,
                           "designs": {}, "grid_search": {}, "notes": []}
    (root / "models").mkdir(parents=True, exist_ok=True)
    for design, data in designs.items():
        classes = data.classes
        Xtr, ytr = data.X["train"], data.y["train"]
        Xte, yte = data.X["test"], data.y["test"].to_numpy()
        rows = scene_sample(data.ids["train"]["scene_id"], svm_sample, cfg.seed)
        Xs, ys = Xtr.iloc[rows], ytr.iloc[rows]
        say(f"== {design}: SVM on {len(rows):,} of {len(Xtr):,} training buildings "
            f"({data.ids['train']['scene_id'].iloc[rows].nunique()} scenes)")
        t0 = time.perf_counter()
        svm = make_pipeline("svm", {**SVM_DEFAULT, "probability": True}, cfg.seed, skew).fit(Xs, ys)
        say(f"   fitted in {time.perf_counter() - t0:.0f} s")
        knn, knn_meta = _saved(models_root, design, "knn", data)
        dt, dt_meta = _saved(models_root, design, "decision_tree", data)
        rf, rf_meta = _saved(models_root, design, "random_forest", data)
        orders = {n: [str(c) for c in p.classes_] for n, p in (("svm", svm), ("knn", knn), ("decision_tree", dt), ("random_forest", rf))}
        if len({tuple(o) for o in orders.values()}) != 1 or sorted(orders["svm"]) != sorted(classes):
            raise ValueError(f"class orders differ between models: {orders}")
        joblib.dump(svm, root / "models" / f"svm_{design}.joblib")

        svm_pred = guard.evaluate(f"{design}/svm", lambda: svm.predict(Xte))
        # soft voting over the already-fitted models; FrozenEstimator makes VotingClassifier.fit a no-op for them
        members = {"svm": svm, "random_forest": rf, "knn": knn}
        voting = VotingClassifier([(n, FrozenEstimator(members[n])) for n in ENSEMBLE_MEMBERS], voting="soft").fit(Xs, ys)
        if [str(c) for c in voting.classes_] != orders["svm"]:
            raise ValueError("the voting ensemble's class order differs from its members'")
        vote_proba = guard.evaluate(f"{design}/soft_voting", lambda: voting.predict_proba(Xte))
        mean_proba = np.mean([members[n].predict_proba(Xte) for n in ENSEMBLE_MEMBERS], axis=0)
        if not np.allclose(vote_proba, mean_proba):  # must equal the plain equal-weight average
            raise ValueError("soft voting does not equal the average of the members' probabilities")
        vote_pred = voting.classes_[vote_proba.argmax(axis=1)]

        entries = [_saved_entry("knn", knn_meta, classes), _saved_entry("decision_tree", dt_meta, classes),
                   _saved_entry("random_forest", rf_meta, classes),
                   {"model": "svm", "label": LABELS["svm"], "kind": "default", "source": "new (this experiment)",
                    "hyperparameters": SVM_DEFAULT, "trained_on": f"scene-grouped sample of {len(rows):,} training buildings",
                    "metrics": summarise(yte, svm_pred, classes)},
                   {"model": "soft_voting", "label": LABELS["soft_voting"], "kind": "ensemble", "ensemble_type": "parallel",
                    "source": "new (this experiment)", "members": ENSEMBLE_MEMBERS, "weights": [1, 1, 1],
                    "implementation": "sklearn VotingClassifier(voting='soft') over FrozenEstimator members (no refit)",
                    "metrics": summarise(yte, vote_pred, classes)}]
        out["designs"][design] = {"design_label": data.design_label, "classes": classes, "n_test": int(len(yte)),
                                  "svm_sample": {"buildings": int(len(rows)), "scenes": int(data.ids["train"]["scene_id"].iloc[rows].nunique()),
                                                 "of_training_buildings": int(len(Xtr)), "seed": cfg.seed},
                                  "models": entries}
        say(f"   test macro-F1: SVM {entries[3]['metrics']['macro_f1']}, soft voting {entries[4]['metrics']['macro_f1']}")

        if design != "within_event":
            continue
        groups = data.ids["train"]["scene_id"].to_numpy()
        for name in ("knn", "decision_tree", "svm"):
            if name not in grids:
                continue
            X_, y_, g_ = (Xs, ys, groups[rows]) if name == "svm" else (Xtr, ytr, groups)
            base = make_pipeline(name, {"gamma": "scale", "probability": False} if name == "svm" else {}, cfg.seed, skew)
            t0 = time.perf_counter()
            search = GridSearchCV(base, grids[name], scoring="f1_macro", cv=GroupKFold(n_splits=3), n_jobs=-1,
                                  refit=True).fit(X_, y_, groups=g_)
            seconds = time.perf_counter() - t0
            pred = guard.evaluate(f"{design}/{name}_optimised", lambda: search.best_estimator_.predict(Xte))
            metrics = summarise(yte, pred, classes)
            default = next(e for e in entries if e["model"] == name)["metrics"]["macro_f1"]
            best = {k.removeprefix("model__"): v for k, v in search.best_params_.items()}
            out["grid_search"][name] = {
                "label": LABELS[name], "grid": {k.removeprefix("model__"): v for k, v in grids[name].items()},
                "fixed": {"gamma": "scale", "probability": False} if name == "svm" else {},
                "cv": "GroupKFold(n_splits=3) grouped by scene, scoring f1_macro, within-event training split only",
                "trained_on": f"scene-grouped sample of {len(rows):,} training buildings" if name == "svm" else
                              f"full within-event training split ({len(Xtr):,} buildings)",
                "best_params": best, "best_cv_macro_f1": round(float(search.best_score_), 4),
                "cv_results": [{"params": {k.removeprefix("model__"): v for k, v in p.items()},
                                "mean_cv_macro_f1": round(float(m), 4), "std": round(float(sd), 4)}
                               for p, m, sd in zip(search.cv_results_["params"], search.cv_results_["mean_test_score"],
                                                   search.cv_results_["std_test_score"])],
                "test_macro_f1_default": default, "test_macro_f1_optimised": metrics["macro_f1"],
                "difference": round(metrics["macro_f1"] - default, 4), "seconds": round(seconds, 1)}
            entries.append({"model": f"{name}_optimised", "label": f"{LABELS[name]} (optimised)", "kind": "optimised",
                            "source": "new (GridSearchCV best, refitted)", "hyperparameters": best, "metrics": metrics})
            say(f"   GridSearchCV {name}: best {best}, CV macro-F1 {search.best_score_:.4f}, "
                f"test {default} -> {metrics['macro_f1']} ({seconds:.0f} s)")

    out["test_evaluations"] = guard.used
    out["notes"] = [
        "KNN, Decision Tree and Random Forest test metrics are read from their saved models' metadata (each was "
        "evaluated on its test set once at training time); they are not re-evaluated here.",
        f"The SVM is trained on a scene-grouped random sample of about {svm_sample:,} training buildings (seed {cfg.seed}), "
        "because SVC training time grows roughly quadratically with the number of samples.",
        "Optimised versions are tuned by grouped cross-validation on the within-event training split and evaluated once "
        "on the within-event test set; they are not evaluated on the cross-event design.",
        "Random Forest is itself a parallel ensemble (bagging of decision trees).",
    ]
    (root / "summary.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    return out


FIX = "python scripts/run_model_comparison.py"


def load_summary(store: Any) -> dict[str, Any]:
    """The saved summary for the Arena, or a structured error with the command that builds it."""
    path = store.paths.artifacts / "experiments" / "model_comparison" / "summary.json"
    if not path.is_file():
        raise NotFound("The model comparison has not been run.", code="results_not_found", fix=FIX)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raise ResultsInvalid("The saved model-comparison summary is not valid JSON.", fix=FIX) from None
    designs = doc.get("designs") if isinstance(doc, dict) else None
    if not isinstance(designs, dict) or not all(isinstance(d, dict) and isinstance(d.get("models"), list) for d in designs.values()):
        raise ResultsInvalid("The saved model-comparison summary has no valid 'designs' section.", fix=FIX)
    return doc


ADABOOST = {"n_estimators": 200, "learning_rate": 0.5}


def run_adaboost(cfg: TrainingConfig, designs: dict[str, DesignData], root: Path = ROOT,
                 say: Callable[[str], None] = print) -> dict[str, Any]:
    """Optional serial ensemble: AdaBoost over decision stumps, trained on each design's full training split and
    evaluated once on its test set. Appended to an existing summary, so the other models are not evaluated again."""
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.tree import DecisionTreeClassifier

    path = root / "summary.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    guard = TestSetGuard()
    for design, data in designs.items():
        entries = doc["designs"][design]["models"]
        if any(e["model"] == "adaboost" for e in entries):
            raise ValueError(f"AdaBoost was already evaluated on the {design} test set")
        base = make_pipeline("decision_tree", {}, cfg.seed, float(cfg.preprocessing.get("skew_threshold", 2.0)))
        stump = DecisionTreeClassifier(max_depth=1, random_state=cfg.seed)
        pipe = Pipeline([*base.steps[:-1], ("model", AdaBoostClassifier(stump, random_state=cfg.seed, **ADABOOST))])
        t0 = time.perf_counter()
        pipe.fit(data.X["train"], data.y["train"])
        pred = guard.evaluate(f"{design}/adaboost", lambda: pipe.predict(data.X["test"]))
        entries.append({"model": "adaboost", "label": "AdaBoost (decision stumps)", "kind": "ensemble", "ensemble_type": "serial",
                        "source": "new (this experiment, optional)",
                        "hyperparameters": {**ADABOOST, "base": "DecisionTreeClassifier(max_depth=1)"},
                        "trained_on": f"full {design} training split ({len(data.X['train']):,} buildings)",
                        "metrics": summarise(data.y["test"].to_numpy(), pred, data.classes)})
        say(f"   {design}: AdaBoost test macro-F1 {entries[-1]['metrics']['macro_f1']} ({time.perf_counter() - t0:.0f} s)")
        doc["test_evaluations"][f"{design}/adaboost"] = 1
    doc["notes"].append("AdaBoost (serial ensemble, boosting) over decision stumps is unweighted: no class weights.")
    path.write_text(json.dumps(doc, indent=1, default=float), encoding="utf-8")
    return doc

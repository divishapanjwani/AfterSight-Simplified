"""Choose each serving model's review-queue confidence threshold on the VALIDATION split.

Usage: python scripts/calibrate_review.py
Writes artifacts/models/<serving design>/<scheme>/review_thresholds.json. Never uses test data.
"""

import sys
from datetime import datetime, timezone

import _bootstrap  # noqa: F401
from app.core.config import REPO_ROOT
from app.inference.review import choose_threshold, write_thresholds
from app.training.cli import data_for, load_inputs
from app.training.models import CONFIDENCE_KIND
from app.training.persistence import ModelLoadError, load_model


def main() -> int:
    cfg, table, manifests = load_inputs()
    s = cfg.serving
    data = data_for(cfg, table, manifests, s.design, s.scheme)
    folder = REPO_ROOT / "artifacts" / "models" / s.design / s.scheme
    doc = {"chosen_on": "validation split", "rule": f"lowest t with accuracy(confidence >= t) >= {cfg.review.target_accuracy}",
           "target_accuracy": cfg.review.target_accuracy, "design": s.design, "scheme": s.scheme,
           "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "models": {}}
    for name in s.models:
        try:
            pipe, _meta = load_model(folder / f"{name}.joblib", expected_manifest_hash=data.manifest_hash)
        except ModelLoadError as exc:
            print(f"{name}: skipped — {exc}", file=sys.stderr)
            continue
        proba = pipe.predict_proba(data.X["val"])
        pred = pipe.classes_[proba.argmax(axis=1)]
        result = choose_threshold(proba.max(axis=1), pred == data.y["val"].to_numpy(), cfg.review.target_accuracy)
        doc["models"][name] = {**result, "confidence": CONFIDENCE_KIND[name]}
        print(f"{name:<20} threshold {result['threshold']}  flags {result['flagged_share_val']:.1%} of val  "
              f"acc above {result.get('accuracy_at_or_above_val')}  below {result.get('accuracy_below_val')}")
    print(f"-> {write_thresholds(folder, doc).relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

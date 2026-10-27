"""Choose each serving model's review-queue confidence threshold on the VALIDATION split.

Usage: python scripts/calibrate_review.py [--out <folder>] [--models NAME ...]
Writes <folder>/review_thresholds.json (default folder: artifacts/models/<serving design>/<scheme>/).
Never uses test data. --out lets you check calibration without replacing the saved thresholds.
--models calibrates only the named served models and merges them into the existing file: every other entry
(and the file's created_at) is kept exactly as it is.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timezone

import _bootstrap  # noqa: F401
from app.core.config import REPO_ROOT
from app.inference.review import choose_threshold, read_thresholds, write_thresholds
from app.training.cli import data_for, load_inputs
from app.training.models import CONFIDENCE_KIND
from app.training.persistence import ModelLoadError, load_model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None, help="folder for review_thresholds.json (default: next to the models)")
    parser.add_argument("--models", nargs="+", default=None, help="only these served models, merged into the existing file")
    args = parser.parse_args()
    cfg, table, manifests = load_inputs()
    s = cfg.serving
    data = data_for(cfg, table, manifests, s.design, s.scheme)
    folder = REPO_ROOT / "artifacts" / "models" / s.design / s.scheme
    doc = {"chosen_on": "validation split", "rule": f"lowest t with accuracy(confidence >= t) >= {cfg.review.target_accuracy}",
           "target_accuracy": cfg.review.target_accuracy, "design": s.design, "scheme": s.scheme,
           "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "models": {}}
    names = args.models or s.models
    unknown = [n for n in names if n not in s.models]
    if unknown:
        print(f"not served: {', '.join(unknown)}", file=sys.stderr)
        return 1
    for name in names:
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
    if args.models:  # merge: keep every other model's entry unchanged
        existing, problem = read_thresholds(folder / "review_thresholds.json")
        if existing is None:
            print(f"cannot merge: {problem}", file=sys.stderr)
            return 1
        stamp = doc["created_at"]
        doc = {**existing, "models": {**existing["models"],
                                      **{n: {**v, "calibrated_at": stamp} for n, v in doc["models"].items()}}}
    out_folder = args.out or folder
    out_folder.mkdir(parents=True, exist_ok=True)
    print(f"-> {write_thresholds(out_folder, doc)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

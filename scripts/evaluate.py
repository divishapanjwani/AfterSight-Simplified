"""Print the saved Phase 3 results (read-only: never re-evaluates the test set).

Usage: python scripts/evaluate.py
"""

import json
import sys

import _bootstrap  # noqa: F401
from app.core.config import REPO_ROOT

METRICS = REPO_ROOT / "artifacts" / "metrics"
ANALYSES = REPO_ROOT / "artifacts" / "analyses"
# Saved rows not shown (the files are left as they are): logistic regression is not a damage model of this project,
# and the threshold baseline is no longer in active use.
NOT_DAMAGE_MODELS = {"logistic_regression", "baseline"}


def main() -> int:
    files = sorted(METRICS.glob("*/*/comparison.json"))
    if not files:
        print("No trained models yet. Run: python scripts/train.py", file=sys.stderr)
        return 1
    for path in files:
        doc = json.loads(path.read_text())
        classes = doc["classes"]
        print(f"\n{doc['design_label']} · {doc['scheme_label']}  (TEST SET unless noted)")
        head = f"  {'model':<22}{'val mF1':>8}{'test mF1':>9}{'acc':>7}" + "".join(f"{'R ' + c[:9]:>13}" for c in classes) + f"{'train s':>9}{'ms/1k':>9}"
        print(head)
        for r in (r for r in doc["models"] if r["model"] not in NOT_DAMAGE_MODELS):
            name = r["model"] + (" (optional)" if r["optional"] else "")
            print(f"  {name:<22}{r['val_macro_f1']:>8.3f}{r['test_macro_f1']:>9.3f}{r['test_accuracy']:>7.3f}"
                  + "".join(f"{(r['test_recall'][c] or 0):>13.3f}" for c in classes)
                  + f"{r['train_seconds']:>9.1f}{1000 * r['inference_seconds_per_1000']:>9.1f}")
    print("\n(Saved logistic_regression and baseline rows are not shown. SVM and ensembles: "
          "artifacts/experiments/model_comparison/summary.json.)")
    conf = ANALYSES / "confounding" / "within_event.json"
    if conf.is_file():
        c = json.loads(conf.read_text())
        print(f"\nEvent-confounding (within-event, TEST; analysis classifier: multinomial logistic regression): accuracy {c['test_metrics']['accuracy']}, "
              f"macro-F1 {c['test_metrics']['macro_f1']}, majority-class {c['majority_class_accuracy']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

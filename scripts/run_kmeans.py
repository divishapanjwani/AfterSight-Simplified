"""K-Means analysis only (no classifiers): elbow + silhouette, cluster-vs-label contingency, ARI.

Usage: python scripts/run_kmeans.py [--design within_event]
Clusters are an analysis of feature structure; they are never used as damage predictions.
"""

import argparse
import sys

import _bootstrap  # noqa: F401
from app.evaluation.analysis import run_kmeans
from app.training.cli import data_for, load_inputs
from app.training.run import OutputRoots


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", action="append")
    args = parser.parse_args()
    cfg, table, manifests = load_inputs()
    for design in args.design or list(cfg.splits.designs):
        doc = run_kmeans(data_for(cfg, table, manifests, design, "four_class"), cfg, OutputRoots())
        k = doc["chosen_k_by_silhouette"]
        print(f"{design}: chosen k={k} (elbow heuristic {doc['elbow_k_heuristic']}), ARI {doc['per_k'][str(k)]['adjusted_rand_index']}, "
              f"ARI at k=4 {doc['per_k']['4']['adjusted_rand_index']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

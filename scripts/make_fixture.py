"""Generate the deterministic SYNTHETIC fixture (xBD layout) used by tests — never real data.

Usage (repo root, venv active):
    python scripts/make_fixture.py                # writes backend/tests/fixtures/xbd_synthetic{,_parquet}/
    python scripts/make_fixture.py --out /tmp/fx  # somewhere else (adds a _parquet sibling folder)
"""

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401  (adds backend/ to sys.path)
from app.core.config import REPO_ROOT
from app.datasets.synthetic_fixture import build_synthetic_parquet, build_synthetic_xbd

DEFAULT_OUT = REPO_ROOT / "backend" / "tests" / "fixtures" / "xbd_synthetic"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output folder")
    args = parser.parse_args()
    for root in (build_synthetic_xbd(args.out), build_synthetic_parquet(args.out.with_name(args.out.name + "_parquet"))):
        n_files = sum(1 for p in root.rglob("*") if p.is_file())
        print(f"Synthetic fixture written to {root} ({n_files} files). NOT xBD data.")


if __name__ == "__main__":
    main()

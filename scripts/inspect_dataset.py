"""Inspect the xBD files on disk and write a verified report before any feature/model work.

Usage (repo root, venv active):
    python scripts/inspect_dataset.py                                   # real data, config/events.yaml
    python scripts/inspect_dataset.py --alignment                       # + pre/post shift check (slower)
    python scripts/inspect_dataset.py --limit-per-event 20              # quick look
    python scripts/inspect_dataset.py --no-cache                        # don't extract scenes to the cache
    python scripts/inspect_dataset.py --config config/events.fixture.yaml   # synthetic fixture

Writes <inspection.output_dir>/{inspection_report.json, scenes.csv, dropped_buildings.csv,
mixed_label_blobs.csv, class_counts.csv} and figures to <inspection.figures_dir>/. For the Parquet
format it also extracts the configured events' scenes to <cache.dir>/ (never other events).
Exit code 1 if no usable scenes were found.
"""

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401  (adds backend/ to sys.path)
from app.core.damage_scale import DAMAGE_CLASSES
from app.core.events_config import DEFAULT_EVENTS_CONFIG, ConfigError, load_events_config
from app.datasets.inspection import run_inspection, write_outputs
from app.datasets.inspection_figures import write_inspection_figures


def _mb(n: int) -> str:
    return f"{n / 1e6:,.1f} MB"


def _progress(done: int, total: int) -> None:
    if done == total or done % 250 == 0:
        print(f"  inspected {done}/{total} scenes", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_EVENTS_CONFIG)
    parser.add_argument("--alignment", action="store_true", help="estimate pre/post shift per pair (phase correlation)")
    parser.add_argument("--limit-per-event", type=int, default=None, help="inspect at most N scenes per event")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--no-cache", action="store_true", help="Parquet only: skip extracting scenes to the cache")
    args = parser.parse_args()

    try:
        config = load_events_config(args.config.resolve())
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2

    root = config.resolve(config.dataset_root)
    label = "SYNTHETIC FIXTURE (not xBD)" if config.source == "synthetic" else "xBD"
    print(f"Inspecting {label} ({config.dataset_format}) at {root}")
    result = run_inspection(config, check_alignment=args.alignment, limit_per_event=args.limit_per_event,
                            progress=_progress, write_cache=not args.no_cache)
    report = result.to_report()

    # Write nothing when there is no data, so an empty report is never mistaken for real statistics.
    if not report["dataset_root_exists"]:
        print(f"\nDataset folder not found: {root}\nSee data/README.md for where to extract xBD.", file=sys.stderr)
        return 1
    if report["parquet_schema_problems"]:
        print(f"\nShards with missing columns (skipped): {report['parquet_schema_problems']}", file=sys.stderr)
    if not report["xbd_splits_found"]:
        if config.dataset_format == "xbd_parquet":
            print(f"\nNo readable *.parquet shards under {root}.", file=sys.stderr)
        else:
            print(f"\nNo split folders ({', '.join(config.xbd_splits)}) under {root}.", file=sys.stderr)
        print(f"Folders present: {report['root_subdirs'] or 'none'} — did the data land one level deeper?", file=sys.stderr)
        return 1

    paths = write_outputs(result, None if args.no_figures else write_inspection_figures)
    t = report["totals"]
    print(f"\nxBD split folders found: {', '.join(report['xbd_splits_found'])}")
    print(f"Events found:   {', '.join(report['events_found']) or 'none'}")
    if report["events_missing"]:
        print(f"Events MISSING: {', '.join(report['events_missing'])}")
    if report["events_on_disk_not_configured"]:
        print(f"Also on disk (not configured): {report['events_on_disk_not_configured']}")
    print(f"Scenes: {t['scenes_discovered']} discovered · {t['pairs_complete']} complete pairs · "
          f"{t['scenes_usable']} usable · {t['images_unreadable']} unreadable images")
    print(f"Scenes without buildings: {t['scenes_without_buildings']}")
    print(f"Footprints: {t['raw_footprints']:,} raw · {t['buildings_kept']:,} kept · {t['buildings_dropped']:,} dropped "
          f"({t['buildings_excluded_label']:,} excluded label) · {t['buildings_repaired']} repaired · {t['buildings_clipped']} clipped")
    print(f"Mixed-label blobs: {t['mixed_label_blobs']:,} ({t['mixed_label_blobs_kept']:,} kept by majority vote) · "
          f"touching image border: {t['buildings_touching_border']:,}")
    overall = report["class_counts"]["overall"]
    print("Class counts:   " + " · ".join(f"{c} {overall[c]:,}" for c in DAMAGE_CLASSES))
    print(f"Labelled xBD splits: {report['labelled_xbd_splits']}")
    if report["drop_reasons"]:
        print(f"Drop reasons:   {report['drop_reasons']}")
    sizes = report["footprint_size_px"]["all_blobs_before_filter"] or report["footprint_size_px"]["kept"]
    if sizes.get("count"):
        print(f"Footprint size px: min {sizes['min']:g} · p5 {sizes['p5']:g} · median {sizes['median']:g} · "
              f"p95 {sizes['p95']:g} · max {sizes['max']:g}  (min_area_px = {config.geometry.min_area_px:g})")
    if report["mask_checks"]:
        m = report["mask_checks"]
        print(f"Damage-mask values (px, all events): {m['damage_mask_values_px']}")
        print(f"Unexpected damage values: {m['unexpected_damage_values_px'] or 'none'}")
        print(f"Footprint px without damage value: {m['footprint_px_without_damage_value']:,} of {m['footprint_px']:,} · "
              f"damage px outside footprints: {m['damage_px_outside_footprint']:,}")
    if report["all_events"]:
        print("\nAll events in the mirror (mask-derived, same rules):")
        print(f"  {'event':<22}{'scenes':>7}{'empty':>7}{'kept':>8}" + "".join(f"{c:>14}" for c in DAMAGE_CLASSES) + f"{'mixed':>7}{'no label':>9}")
        for name, e in report["all_events"].items():
            cc = e["class_counts"]
            print(f"  {name:<22}{e['scenes']:>7}{e['scenes_without_buildings']:>7}{e['buildings_kept']:>8,}"
                  + "".join(f"{cc[c]:>14,}" for c in DAMAGE_CLASSES)
                  + f"{e['mixed_label_blobs']:>7}{e['drop_reasons'].get('label_missing', 0):>9}")
    if report["cache"] and report["cache"]["enabled"]:
        c = report["cache"]
        print(f"\nCache: {c['scenes_written']} scenes written ({_mb(c['bytes_written'])}) to {c['dir']} · "
              f"stale folders from other selections: {c.get('stale_scenes_not_in_selection', 0)}")
    if report["alignment"]["checked"]:
        print(f"Alignment flags: {len(report['alignment']['scenes_flagged'])} scenes over {report['alignment']['flag_threshold_px']} px")
    r = report["resources"]
    print(f"Disk (selected events): {_mb(r['raw_bytes_selected_events'])} · peak RAM (measured): {_mb(r['peak_rss_bytes_measured'])}")
    print(f"\nReport: {paths['report']}")
    return 0 if t["scenes_usable"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())

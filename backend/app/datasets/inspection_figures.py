"""Figures for the dataset inspector: class distribution chart and sample pre/post pairs with GT footprints."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only; no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.patches import Patch
from matplotlib.patches import Polygon as PolygonPatch
from PIL import Image
from shapely import get_parts

from app.core.damage_scale import DAMAGE_CLASSES, DAMAGE_COLORS
from app.datasets.inspection import SIZE_BINS, InspectionResult

SURFACE = "#fcfcfb"
INK = "#1b1b1a"
INK_MUTED = "#6b6a66"
GRID = "#e4e2dc"
SYNTHETIC_MARK = "SYNTHETIC FIXTURE — not xBD data"


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _watermark(fig: plt.Figure, result: InspectionResult) -> None:
    if result.config.source == "synthetic":
        fig.text(0.5, 0.5, SYNTHETIC_MARK, ha="center", va="center", rotation=20,
                 fontsize=22, color="#e8590c", alpha=0.35, weight="bold")


def class_distribution(overall: dict[str, int], by_event: dict[str, dict[str, int]], title: str,
                       result: InspectionResult, path: Path) -> Path | None:
    if sum(overall.values()) == 0:
        return None

    fig, (ax_total, ax_share) = plt.subplots(
        1, 2, figsize=(12, 1.4 + 0.5 * max(4, len(by_event))), gridspec_kw={"width_ratios": [1, 1.4]}
    )
    fig.patch.set_facecolor(SURFACE)

    # Panel 1: absolute building counts per class.
    classes = list(DAMAGE_CLASSES)[::-1]  # barh draws bottom-up, so no-damage ends up on top
    values = [overall[c] for c in classes]
    bars = ax_total.barh(classes, values, color=[DAMAGE_COLORS[c] for c in classes],
                         edgecolor=SURFACE, linewidth=2, height=0.7)
    ax_total.bar_label(bars, labels=[f"{v:,}" for v in values], padding=4, color=INK, fontsize=9)
    ax_total.set_title("Buildings per damage class", loc="left", color=INK, fontsize=11)
    ax_total.set_xlabel("buildings (ground truth)", color=INK_MUTED, fontsize=9)
    ax_total.margins(x=0.2)
    _style(ax_total)

    # Panel 2: class share within each event (stacked to 100%), n shown in the event label.
    events = sorted(by_event)
    left = [0.0] * len(events)
    totals = [sum(by_event[e].values()) for e in events]
    for cls in DAMAGE_CLASSES:
        shares = [by_event[e][cls] / t if t else 0 for e, t in zip(events, totals)]
        ax_share.barh([f"{e}  (n={t:,})" for e, t in zip(events, totals)], shares, left=left,
                      color=DAMAGE_COLORS[cls], edgecolor=SURFACE, linewidth=2, height=0.7, label=cls)
        left = [a + b for a, b in zip(left, shares)]
    ax_share.set_xlim(0, 1)
    ax_share.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax_share.set_title("Class share per event", loc="left", color=INK, fontsize=11)
    ax_share.invert_yaxis()
    ax_total.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    _style(ax_share)
    handles, labels = ax_share.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="lower right", frameon=False, fontsize=9, labelcolor=INK)

    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=12)
    _watermark(fig, result)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def footprint_sizes(result: InspectionResult, path: Path) -> Path | None:
    """Histogram of footprint areas on log-spaced bins, with the min-area threshold marked."""
    areas = result.all_blob_areas if result.all_blob_areas else result.kept_areas
    if not areas:
        return None
    population = "all footprint blobs before filtering" if result.all_blob_areas else "kept footprints"
    edges = [1, *SIZE_BINS[1:], max(SIZE_BINS[-1] * 2, max(areas) + 1)]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    fig.patch.set_facecolor(SURFACE)
    ax.hist(areas, bins=edges, color="#6b6a66", edgecolor=SURFACE, linewidth=2)
    ax.set_xscale("log", base=2)
    threshold = result.config.geometry.min_area_px
    ax.axvline(threshold, color="#e8590c", linewidth=2)
    ax.annotate(f"min_area_px = {threshold:g}", xy=(threshold, ax.get_ylim()[1] * 0.92), xytext=(6, 0),
                textcoords="offset points", color=INK, fontsize=9)
    ax.set_xlabel("footprint area (pixels, log2 scale)", color=INK_MUTED, fontsize=9)
    ax.set_ylabel("footprints", color=INK_MUTED, fontsize=9)
    ax.set_title(f"Footprint size distribution — {population} (n={len(areas):,})", loc="left", color=INK, fontsize=11)
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    _watermark(fig, result)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def sample_pairs(result: InspectionResult, out_dir: Path) -> list[Path]:
    derived = result.config.dataset_format == "xbd_parquet"
    note = "footprints derived from xBD annotation masks" if derived else "ground-truth footprints from post-disaster labels"
    if derived and result.cache and result.cache.get("enabled"):
        note += " · read back from scene cache"
    written = []
    for key in result.sample_keys:
        split, scene_id = key.split("/", 1)
        pre_img, post_img, buildings = result.sample_loaders[key]()
        fig, axes = plt.subplots(1, 2, figsize=(11, 6.1))
        fig.patch.set_facecolor(SURFACE)
        for ax, phase, img in ((axes[0], "pre-disaster", pre_img), (axes[1], "post-disaster", post_img)):
            ax.imshow(img.convert("RGB"))
            for b in buildings:
                for part in get_parts(b.polygon):
                    xy = list(part.exterior.coords)
                    # dark halo under the class colour keeps light outlines visible on bright imagery
                    ax.add_patch(PolygonPatch(xy, closed=True, fill=False, edgecolor="#000000", linewidth=2.4, alpha=0.7))
                    ax.add_patch(PolygonPatch(xy, closed=True, fill=False, edgecolor=DAMAGE_COLORS[b.damage], linewidth=1.3))
            ax.set_title(phase, loc="left", color=INK, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        counts = {c: sum(b.damage == c for b in buildings) for c in DAMAGE_CLASSES}
        fig.legend(handles=[Patch(facecolor=DAMAGE_COLORS[c], edgecolor="#000000", label=f"{c} ({counts[c]})") for c in DAMAGE_CLASSES],
                   ncol=4, loc="lower left", frameon=False, fontsize=9, labelcolor=INK)
        fig.suptitle(f"{scene_id}  ·  xBD split: {split}  ·  {note}", x=0.01, ha="left", color=INK, fontsize=11)
        _watermark(fig, result)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        out = out_dir / f"sample_{scene_id}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=110, facecolor=SURFACE)
        plt.close(fig)
        written.append(out)
    return written


def write_inspection_figures(result: InspectionResult, figures_dir: Path) -> list[Path]:
    report = result.to_report()
    source = "synthetic fixture" if result.config.source == "synthetic" else "xBD ground truth"
    written = []
    chart = class_distribution(report["class_counts"]["overall"], report["class_counts"]["by_event"],
                               f"Damage class distribution — {source}, configured events, usable scenes",
                               result, figures_dir / "class_distribution.png")
    if chart is not None:
        written.append(chart)
    if report["all_events"]:
        table = report["all_events"]
        overall = {c: sum(e["class_counts"][c] for e in table.values()) for c in DAMAGE_CLASSES}
        chart = class_distribution(overall, {name: e["class_counts"] for name, e in table.items()},
                                   f"Damage class distribution — {source}, ALL events in the mirror (event selection aid)",
                                   result, figures_dir / "class_distribution_all_events.png")
        if chart is not None:
            written.append(chart)
    sizes = footprint_sizes(result, figures_dir / "footprint_sizes.png")
    if sizes is not None:
        written.append(sizes)
    written.extend(sample_pairs(result, figures_dir))
    return written

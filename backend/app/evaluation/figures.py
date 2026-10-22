"""Evaluation figures: confusion matrices, K-Means elbow/silhouette, contingency heatmaps."""

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK = "#1b1b1a"
INK_MUTED = "#6b6a66"
GRID = "#e4e2dc"
# Single-hue sequential ramp for magnitudes (share of row). Distinct from the plum damage ramp, which
# encodes class identity, so a heatmap cell's colour is never read as a damage class.
MAGNITUDE = LinearSegmentedColormap.from_list("magnitude", ["#f3f7fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    return path


def heatmap(matrix: np.ndarray, row_labels: list[str], col_labels: list[str], title: str, subtitle: str,
            xlabel: str, ylabel: str, path: Path, normalise_rows: bool = True) -> Path:
    counts = np.asarray(matrix, dtype=float)
    shares = counts / np.where(counts.sum(axis=1, keepdims=True) == 0, 1, counts.sum(axis=1, keepdims=True)) if normalise_rows else counts / max(counts.max(), 1)
    fig, ax = plt.subplots(figsize=(1.6 + 1.25 * len(col_labels), 1.6 + 0.85 * len(row_labels)))
    fig.patch.set_facecolor(SURFACE)
    ax.imshow(shares, cmap=MAGNITUDE, vmin=0, vmax=1)
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            colour = "white" if shares[i, j] > 0.55 else INK
            text = f"{int(counts[i, j]):,}\n{shares[i, j]:.0%}" if normalise_rows else f"{int(counts[i, j]):,}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8.5, color=colour)
    ax.set_xticks(range(len(col_labels)), col_labels, rotation=30, ha="right", fontsize=9, color=INK)
    ax.set_yticks(range(len(row_labels)), row_labels, fontsize=9, color=INK)
    ax.set_xlabel(xlabel, color=INK_MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_MUTED, fontsize=9)
    ax.set_xticks(np.arange(-0.5, len(col_labels)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.suptitle(title, x=0.02, ha="left", fontsize=11, color=INK)
    ax.set_title(subtitle, loc="left", fontsize=8.5, color=INK_MUTED)
    fig.tight_layout()
    return _save(fig, path)


def confusion_figure(metrics: dict[str, Any], title: str, subtitle: str, path: Path) -> Path:
    cm = metrics["confusion_matrix"]
    return heatmap(np.array(cm["rows_true_cols_pred"]), cm["labels"], cm["labels"], title, subtitle,
                   "predicted", "true (row % = recall)", path)


def kmeans_curves(k_values: list[int], inertia: list[float], silhouette: list[float], chosen: int,
                  title: str, path: Path) -> Path:
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.4))
    fig.patch.set_facecolor(SURFACE)
    for ax, values, label in ((a, inertia, "inertia (within-cluster sum of squares)"),
                              (b, silhouette, "silhouette (sampled; higher = better separated)")):
        ax.plot(k_values, values, color="#3987e5", linewidth=2, marker="o", markersize=6)
        ax.axvline(chosen, color="#e8590c", linewidth=1.5)
        ax.set_xlabel("k", color=INK_MUTED, fontsize=9)
        ax.set_title(label, loc="left", fontsize=10, color=INK)
        ax.set_facecolor(SURFACE)
        ax.grid(color=GRID, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=INK_MUTED, labelsize=9)
    b.annotate(f"chosen k = {chosen}", xy=(chosen, max(silhouette)), xytext=(6, -2), textcoords="offset points", fontsize=9, color=INK)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    return _save(fig, path)

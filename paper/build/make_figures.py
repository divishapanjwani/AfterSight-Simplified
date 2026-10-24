"""Figures for the KNN vs Decision Tree paper, generated only from saved artifacts.

Inputs: artifacts/models/<design>/four_class/{knn,decision_tree}.json (test confusion matrices) and
paper/build/stats.json (scene-level bootstrap from scripts/paper_stats.py).
Usage (repo root): .venv/bin/python paper/build/make_figures.py   -> paper/figures/*.png (300 dpi)
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "figures"
CLASSES = ["no-damage", "minor-damage", "major-damage", "destroyed"]
SHORT = ["None", "Minor", "Major", "Destr."]
MODELS = {"knn": "KNN", "decision_tree": "Decision Tree"}
DESIGNS = {"within_event": "Within-event", "cross_event": "Cross-event (Michael held out)"}
# two series: colour AND marker differ, so the figure also reads in greyscale print
STYLE = {"knn": {"color": "#1f5fa8", "marker": "o"}, "decision_tree": {"color": "#c0631a", "marker": "s"}}

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "font.size": 8, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})


def meta(design: str, model: str) -> dict:
    return json.loads((ROOT / "artifacts" / "models" / design / "four_class" / f"{model}.json").read_text())


def fig_pipeline() -> None:
    """Two rows read as a snake: top row left to right, bottom row right to left."""
    top = ["xBD train-split\nmirror, 5 events", "Mask-derived\nbuildings", "37 engineered\nfeatures", "Scene-grouped\nsplits"]
    bottom = ["Shared pipeline:\nlog1p, z-score\n(fit on train)", "KNN / DT tuned\non validation\nmacro-F1", "Single test\nevaluation +\nscene bootstrap"]
    fig, ax = plt.subplots(figsize=(3.5, 1.45))
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 2)
    ax.axis("off")
    arrow = {"arrowstyle": "-|>", "lw": 0.6, "color": "#333333", "mutation_scale": 6}

    def box(x: float, y: float, label: str) -> None:
        ax.add_patch(FancyBboxPatch((x + 0.07, y + 0.12), 0.86, 0.76, boxstyle="round,pad=0.01,rounding_size=0.06",
                                    linewidth=0.6, edgecolor="#333333", facecolor="#f2f2f2"))
        ax.text(x + 0.5, y + 0.5, label, ha="center", va="center", fontsize=5.6, linespacing=1.15)

    for i, label in enumerate(top):
        box(i, 1, label)
        if i < len(top) - 1:
            ax.annotate("", xy=(i + 1.07, 1.5), xytext=(i + 0.93, 1.5), arrowprops=arrow)
    ax.annotate("", xy=(3.5, 0.88), xytext=(3.5, 1.12), arrowprops=arrow)  # down from the last top box
    for k, label in enumerate(bottom):
        x = 3 - k
        box(x, 0, label)
        if k < len(bottom) - 1:
            ax.annotate("", xy=(x - 0.07, 0.5), xytext=(x + 0.07, 0.5), arrowprops=arrow)
    fig.savefig(OUT / "fig1_pipeline.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig_macro_f1(stats: dict) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 1.9))
    for d_index, design in enumerate(DESIGNS):
        for m_index, model in enumerate(MODELS):
            entry = stats["designs"][design]["models"][model]
            x = d_index + (m_index - 0.5) * 0.28
            low, high = entry["macro_f1_ci95"]
            ax.errorbar(x, entry["macro_f1"], yerr=[[entry["macro_f1"] - low], [high - entry["macro_f1"]]],
                        fmt=STYLE[model]["marker"], color=STYLE[model]["color"], markersize=4.5, capsize=2.5,
                        elinewidth=0.8, capthick=0.8, label=MODELS[model] if d_index == 0 else None)
            ax.text(x + 0.05, entry["macro_f1"], f"{entry['macro_f1']:.3f}", va="center", fontsize=6.5)
    ax.set_xticks([0, 1], list(DESIGNS.values()))
    ax.set_xlim(-0.55, 1.6)
    ax.set_ylim(0.2, 0.6)
    ax.set_ylabel("Test macro-F1")
    ax.grid(axis="y", linewidth=0.3, color="#cccccc")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    fig.savefig(OUT / "fig2_macro_f1.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig_confusions() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(3.5, 3.35))
    for r, design in enumerate(DESIGNS):
        for c, model in enumerate(MODELS):
            ax = axes[r, c]
            cm = np.array(meta(design, model)["metrics"]["test"]["confusion_matrix"]["rows_true_cols_pred"], dtype=float)
            assert meta(design, model)["metrics"]["test"]["confusion_matrix"]["labels"] == CLASSES
            share = cm / cm.sum(axis=1, keepdims=True)  # row-normalised: the diagonal is per-class recall
            ax.imshow(share, cmap="Greys", vmin=0, vmax=1)
            for i in range(4):
                for j in range(4):
                    ax.text(j, i, f"{share[i, j] * 100:.0f}", ha="center", va="center", fontsize=6,
                            color="white" if share[i, j] > 0.55 else "black")
            ax.set_xticks(range(4), SHORT, fontsize=6)
            ax.set_yticks(range(4), SHORT if c == 0 else [], fontsize=6)
            ax.tick_params(length=0)
            ax.set_title(f"{MODELS[model]}, {'within' if design == 'within_event' else 'cross'}-event", fontsize=7, pad=3)
            if r == 1:
                ax.set_xlabel("Predicted", fontsize=6.5)
            if c == 0:
                ax.set_ylabel("Reference label", fontsize=6.5)
    fig.tight_layout(pad=0.3, h_pad=0.6, w_pad=0.4)
    fig.savefig(OUT / "fig3_confusion.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stats = json.loads((ROOT / "paper" / "build" / "stats.json").read_text())
    fig_pipeline()
    fig_macro_f1(stats)
    fig_confusions()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()

"""Small, consistent chart helpers (matplotlib) used by the scripts."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.titleweight": "bold",
    "axes.titlesize": 12, "font.size": 10, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": False, "grid.color": "#e6e5e1",
    "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 6,
})


def confusion_matrix_plot(cm, row_labels, col_labels, title, path, note=None):
    """Counts in each cell + row percentage. Rows = true class, cols = outcome."""
    cm = np.asarray(cm)
    row_pct = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(1.35 * len(col_labels) + 2.6, 0.9 * len(row_labels) + 2))
    ax.imshow(row_pct, cmap=BLUES, vmin=0, vmax=1, aspect="auto")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            dark = row_pct[i, j] > 0.55
            ax.text(j, i, f"{cm[i, j]}\n{row_pct[i, j]:.0%}", ha="center", va="center",
                    fontsize=9, color="#ffffff" if dark else INK)
    ax.set_xticks(range(len(col_labels)), col_labels, rotation=20, ha="right")
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.set_xlabel("Model outcome"); ax.set_ylabel("True class")
    ax.set_title(title, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(col_labels)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="both", length=0)
    if note:
        fig.text(0.01, 0.01, note, fontsize=8, color=INK_2)
    fig.tight_layout(rect=(0, 0.04 if note else 0, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def line_plot(x, series, title, xlabel, ylabel, path, percent=True, vline=None, vlabel=None):
    """series: {name: list of y values}. Legend + direct end labels."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.grid(axis="y", alpha=1)
    for color, (name, ys) in zip(SERIES, series.items()):
        ax.plot(x, ys, color=color, marker="o", label=name)
        ax.annotate(name, (x[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK_2)
    if vline is not None:
        ax.axvline(vline, color=MUTED, linestyle="--", linewidth=1)
        if vlabel:
            ax.text(vline, ax.get_ylim()[1], " " + vlabel, va="top", fontsize=8.5, color=INK_2)
    if percent:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title(title, loc="left"); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.legend(loc="lower left", fontsize=8.5)
    ax.margins(x=0.12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def grouped_bar_plot(groups, series, title, ylabel, path):
    """groups: x labels; series: {name: values per group} (values 0-1)."""
    n = len(series)
    width = 0.8 / n
    x = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(max(7, 1.3 * len(groups) + 2), 4.3))
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    for k, (color, (name, vals)) in enumerate(zip(SERIES, series.items())):
        pos = x - 0.4 + width * (k + 0.5)
        ax.bar(pos, vals, width=width - 0.02, color=color, label=name,
               edgecolor=SURFACE, linewidth=1)
        for px, v in zip(pos, vals):
            ax.text(px, v + 0.01, f"{v:.0%}", ha="center", va="bottom", fontsize=7.5, color=INK_2)
    ax.set_xticks(x, groups, rotation=15, ha="right")
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title(title, loc="left"); ax.set_ylabel(ylabel)
    ax.legend(loc="upper right", fontsize=8.5, ncol=n)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)

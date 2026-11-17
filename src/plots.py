"""README figures. Palette: validated categorical slots 1-3 (blue, orange, aqua)
and a single-hue blue ramp for confusion matrices."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = {"Isolation Forest": "#2a78d6", "XGBoost": "#eb6834", "LSTM": "#1baf7a"}
BLUES = LinearSegmentedColormap.from_list(
    "seq_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _fig(w, h):
    fig, ax = plt.subplots(figsize=(w, h), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    return fig, ax


def pr_curves(curves, path, title):
    """curves: list of (name, is_attack, scores, (recall, precision) operating point)."""
    fig, ax = _fig(6.4, 4.6)
    for name, y, s, op in curves:
        p, r, _ = precision_recall_curve(y, s)
        ax.plot(r, p, color=SERIES[name], linewidth=2, label=name)
        ax.plot(*op, marker="o", markersize=8, color=SERIES[name],
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0.4, 1.01)
    ax.set_xlabel("Recall (detection rate)", color=INK)
    ax.set_ylabel("Precision", color=INK)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower left",
              title="dot = alert threshold", title_fontsize=8)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def detection_by_category(rates, categories, path, title):
    """rates: {model: {category: rate}} -> grouped bars with value labels."""
    fig, ax = _fig(7.2, 4.2)
    models = list(rates)
    n = len(models)
    width = 0.8 / n
    x = np.arange(len(categories))
    for i, m in enumerate(models):
        vals = [rates[m][c] for c in categories]
        # 2px surface gap between adjacent bars via the edge.
        bars = ax.bar(x + (i - (n - 1) / 2) * width, vals, width,
                      color=SERIES[m], label=m, edgecolor=SURFACE, linewidth=1.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.0%}",
                    ha="center", va="bottom", fontsize=7.5, color=INK_2)
    ax.set_xticks(x, categories)
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("Share of connections flagged", color=INK)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, ncol=n, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def confusion(cm, labels, path, title):
    cm = np.asarray(cm, dtype=float)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.imshow(norm, cmap=BLUES, vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ink = SURFACE if norm[i, j] > 0.55 else INK
            ax.text(j, i, f"{norm[i, j]:.0%}\n{int(cm[i, j]):,}", ha="center",
                    va="center", fontsize=7.5, color=ink)
    ax.set_xticks(range(len(labels)), labels, fontsize=9, color=INK_2)
    ax.set_yticks(range(len(labels)), labels, fontsize=9, color=INK_2)
    ax.set_xlabel("Predicted", color=INK)
    ax.set_ylabel("Actual (row-normalised)", color=INK)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)

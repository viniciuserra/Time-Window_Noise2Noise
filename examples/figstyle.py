"""Plot style shared by the example notebooks.

Each series keeps the same color and line style in every figure. The colors
come from the Okabe-Ito palette, which stays distinguishable under the common
forms of color blindness; the line styles repeat the distinction for print.
"""
import matplotlib.pyplot as plt

COLORS = {
    "clean": "#000000",
    "noisy": "#999999",
    "twn2n": "#0072B2",
    "lowpass": "#D55E00",
    "visushrink": "#009E73",
    "savgol": "#CC79A7",
    "paper": "#E69F00",
}
STYLES = {"clean": "--", "noisy": "-", "twn2n": "-", "lowpass": "-.", "visushrink": ":", "savgol": (0, (5, 1, 1, 1)),
          "paper": "-"}
LABELS = {"clean": "Clean", "noisy": "Noisy", "twn2n": "TWN2N", "lowpass": "Low-pass", "visushrink": "VisuShrink",
          "savgol": "Savitzky-Golay", "paper": "Paper"}


def apply():
    """Set the matplotlib defaults used in the notebooks."""
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "figure.figsize": (7.0, 3.2),
        "font.family": "serif", "axes.labelsize": 10, "axes.titlesize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5,
        "lines.linewidth": 1.1, "axes.grid": True, "grid.color": "#dddddd",
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    })


def line(ax, x, y, series, **kwargs):
    """Plot one series with its fixed color, line style, and legend label."""
    options = dict(color=COLORS[series], linestyle=STYLES[series], label=LABELS[series])
    options.update(kwargs)
    return ax.plot(x, y, **options)

"""Shared plot style for the manuscript figures.

One module so every figure uses the same font, sizes and role colours; the
style follows the conventions of the systems-paper corpus this submission
targets (IEEE two-column, single-column figures):

* Times-metric serif matching the IEEEtran body text. STIXGeneral ships with
  matplotlib, so the rendering is identical on a developer box and in CI.
* Role colours, not per-figure palettes: baseline grey, studied method dark
  red, second comparator blue. Every series also differs in marker or line
  style, so the figures survive greyscale printing.
* Legend above the axes, horizontal, frameless, method order fixed.
* Absolute values printed on normalised bars, so no absolute number hides
  behind a ratio.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Role colours. Keep the mapping stable across figures.
BASELINE = "#9a9a9a"   # heuristic baseline
OURS = "#8c1d1d"       # the scorer this paper audits
ALT = "#3f76a8"        # second comparator
REF = "#8a8a8a"        # reference / ideal lines
EDGE = "black"
BAR_EDGE_WIDTH = 0.4
# Error bars: thin black caps, as in the target corpus.
ERRORBAR = {"ecolor": "black", "elinewidth": 0.6, "capsize": 1.4, "capthick": 0.6}

COLUMN_WIDTH = 3.45    # IEEEtran \columnwidth in inches


MIN_FONT_PT = 7.0


def use_paper_style(base: float = 8.0) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": base,
        "axes.labelsize": base,
        "axes.titlesize": base,
        "xtick.labelsize": base - 0.5,
        "ytick.labelsize": base - 0.5,
        "legend.fontsize": base - 0.5,
        # Full frame and outward ticks, as in the target corpus.
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.linewidth": 0.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.4,
        "ytick.major.size": 2.4,
        "axes.grid": False,
        "legend.frameon": False,
        "legend.handlelength": 1.2,
        "legend.handletextpad": 0.4,
        "legend.columnspacing": 1.0,
        "legend.borderpad": 0.0,
        "lines.linewidth": 1.0,
        "lines.markersize": 2.6,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        # Preserve font descenders at the tight crop (the forest x-label's
        # extracted glyph bounds otherwise extend beyond the PDF MediaBox).
        "savefig.pad_inches": 0.045,
        # Embed TrueType (Type 42) outlines instead of Type 3 bitmaps-of-glyphs;
        # IEEE PDF eXpress flags Type 3 fonts.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def top_legend(fig, handles, labels, ncol: int | None = None, y: float = 1.0) -> None:
    """Shared horizontal legend above the panels, in fixed method order."""
    fig.legend(handles, labels, ncol=ncol or len(labels), loc="upper center",
               bbox_to_anchor=(0.5, y), frameon=False)


def panel_labels(fig, axes, texts, y: float = 0.01) -> None:
    """`(a) Title` under each panel, all on one baseline.

    Call after the layout is fixed: positions come from the final axes boxes,
    so panels with and without an x-axis label still align.
    """
    for ax, text in zip(axes, texts):
        box = ax.get_position()
        fig.text((box.x0 + box.x1) / 2, y, text, ha="center", va="bottom")


def small(delta: float = 1.0) -> float:
    """A reduced font size that never drops below MIN_FONT_PT."""
    return max(MIN_FONT_PT, plt.rcParams["font.size"] - delta)


def bar_values(ax, xs, tops, values, fmt="{:.2f}", pad_frac: float = 0.03):
    """Print the absolute value above each normalised bar, read bottom-to-top.

    `tops` is where each bar (including its error bar) ends, `values` the
    absolute numbers. Normalising without this would hide the baseline's own
    magnitude behind a ratio.
    """
    pad = ax.get_ylim()[1] * pad_frac
    for x, top, v in zip(xs, tops, values):
        ax.text(x, top + pad, fmt.format(v), ha="center", va="bottom", rotation=90,
                fontsize=small(1.0))

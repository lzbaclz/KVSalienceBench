"""Exp#1 figure: the cross-layer offset sweep and reliability on the VERSION-2 corpus.

(The file name is historical: panel (a) used to plot per-view relevance, whose
numbers now live in the text. Renaming would ripple through the exporter, the
figure checker and the artifact index for no gain.)

(a) Scale the fitted cross-layer offset w_cross/w_within from 0 (the raw within-layer
    ordering) to 2x its fitted value, everything else fixed, on every held-out row:
    pooled AUC rises while per-decision top-decile recall falls. Both are plotted as
    the change from the raw within-layer EMA. Data: icdm_v2_decomposition.json.
    Its two series are labelled in place and drawn in neutral ink with their own
    markers, so that no colour or marker of panel (b)'s legend recurs here with a
    different meaning.
(b) Calibration is a property of the fit configuration, not of the model
    family: the two-view logistic fit, the class-balanced LightGBM the archived
    table used, an unweighted LightGBM and an isotonic-recalibrated LightGBM, all
    scored on the same held-out rows. Legend labels say "LightGBM" to match the
    manuscript; the result-JSON keys keep the historical "GBDT" spelling.
    (PC audit item P1-7: the figure previously plotted the four-view fit under a
    "linear fit" label against the balanced GBDT only.) The legend sits in panel
    (b)'s own column, above its axes: it used to span the figure, which put it over
    panel (a) and ran it into that panel's y-label.

Data: experiments/results/icdm_v2_decomposition.json (sweep) and
experiments/results/icdm_v2.json (pooled_request_split.calibration).
"""
import json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import BASELINE, OURS, ALT, REF, EDGE, BAR_EDGE_WIDTH, COLUMN_WIDTH, \
    use_paper_style, panel_labels, small
import matplotlib.pyplot as plt

# Panel (a) inks: neither is a role colour of panel (b)'s legend.
AUC_INK, REC_INK = "#1a1a1a", "#6b6b6b"

OUT = "paper_icdm/figures"
d = json.load(open("experiments/results/icdm_v2.json"))

use_paper_style()
fig = plt.figure(figsize=(COLUMN_WIDTH, 1.74))
# (a) takes the full height on the left; on the right the legend of (b) has its own
# cell above the (b) axes, so it cannot be read as belonging to (a).
grid = fig.add_gridspec(2, 2, width_ratios=(1.0, 1.42), height_ratios=(0.66, 1.0))
axA = fig.add_subplot(grid[:, 0])
axL = fig.add_subplot(grid[0, 1])
axB = fig.add_subplot(grid[1, 1])
axL.axis("off")

# (a) cross-layer offset sweep, every held-out row (icdm_v2_decomposition.json).
dec = json.load(open("experiments/results/icdm_v2_decomposition.json"))
sweep = dec["sweep"]
lam = np.array([r["lam"] for r in sweep])
d_auc = np.array([r["auc_pooled"] for r in sweep]) - sweep[0]["auc_pooled"]
d_rec = np.array([r["recall_0.10"] for r in sweep]) - sweep[0]["recall_0.10"]
axA.axhline(0, color=REF, lw=0.5, ls=":")
axA.axvline(1.0, color=REF, lw=0.6, ls=":")           # the fitted offset
axA.plot(lam, d_auc, "-D", color=AUC_INK, ms=2.2, lw=1.0)
axA.plot(lam, d_rec, "--x", color=REC_INK, ms=3.0, mew=0.9, lw=1.0)
axA.set_xlabel(r"offset scale $\lambda$")
axA.set_ylabel("change vs. raw EMA")
axA.set_xticks([0, 1, 2])
axA.set_yticks([-0.04, -0.02, 0, 0.02])
axA.set_xlim(-0.08, 2.08)
axA.set_ylim(-0.062, 0.036)
# white boxes so the labels mask the dotted line at the fitted offset instead of crossing it
mask = dict(boxstyle="square,pad=0.08", fc="white", ec="none")
axA.text(1.98, d_auc[-1] + 0.004, "pooled AUC", ha="right", va="bottom", fontsize=small(1.0), color=AUC_INK,
         bbox=mask, zorder=5)
axA.text(1.98, d_rec[-1] - 0.004, "top-10% recall", ha="right", va="top", fontsize=small(1.0), color=REC_INK,
         bbox=mask, zorder=5)
vals = [d_auc[lam == 1.0][0], d_rec[lam == 1.0][0]]

# (b) reliability of four fits on identical held-out rows. The result-JSON keys
# are the historical "GBDT ..." names; the plotted legend uses the backend the
# paper names throughout (LightGBM), so figure and text agree.
cal = d["pooled_request_split"]["calibration"]
axB.plot([0, 1], [0, 1], ":", color=REF, lw=0.7)
series = [("within+cross(2)", "two-view", "o", "-", OURS),
          ("GBDT balanced", "LightGBM balanced", "s", "--", ALT),
          ("GBDT unweighted", "LightGBM unweighted", "^", "-", "#2a7f62"),
          ("GBDT balanced+isotonic", "LightGBM isotonic", "v", "-.", "#8a6d1f")]
for key, short, marker, style, col in series:
    rc = cal[key]["reliability"]
    conf = np.array([c if c is not None else np.nan for c in rc["confidence"]], float)
    acc = np.array([c if c is not None else np.nan for c in rc["accuracy"]], float)
    axB.plot(conf, acc, marker=marker, linestyle=style, color=col, ms=2.4,
             label=f"{short} ({cal[key]['ece']:.3f})")
axB.set_xlabel("Predicted prob.")
axB.set_ylabel("Frequency")
axB.set_xlim(0, 1); axB.set_ylim(0, 1)
axB.set_xticks([0, 0.5, 1]); axB.set_yticks([0, 0.5, 1])
fig.tight_layout(pad=0.2, w_pad=1.0, h_pad=0.5, rect=(0, 0.09, 1, 1))
panel_labels(fig, (axA, axB), ("(a) Cross-layer offset, v2", "(b) Reliability (ECE), v2"))
# Four series will not fit inside panel (b) without covering the curves, so its
# legend is one column in the cell above it. The cell is as wide as panel (b)
# with its tick labels: a legend wider than that would widen the tight bbox, scale
# the whole figure down at one IEEEtran column and drop label sizes below the
# ~9 pt floor that scripts/check_figure_assets.py enforces.
handles, labels = axB.get_legend_handles_labels()
boxL, boxB = axL.get_position(), axB.get_position()
fig.legend(handles, labels, ncol=1, loc="upper right", bbox_to_anchor=(boxB.x1, 1.0),
           frameon=False, fontsize=small(1.0), labelspacing=0.08, handlelength=1.0,
           handletextpad=0.2, borderaxespad=0.0)
os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, "fig_redund_calib.pdf"))
print(f"WROTE {OUT}/fig_redund_calib.pdf | change at the fitted offset: "
      f"pooled AUC {vals[0]:+.4f}, top-10% recall {vals[1]:+.4f} | "
      f"ECE={ {k: round(cal[k]['ece'], 3) for k in cal} }")

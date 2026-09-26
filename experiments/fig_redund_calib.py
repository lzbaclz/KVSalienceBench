"""Exp#1 figure: per-view relevance and reliability on the VERSION-2 corpus.

(a) The two attention-magnitude views carry the recorded relevance; the
    (phase-aligned) query and age proxies do not.
(b) Calibration is a property of the fit configuration, not of the model
    family: the two-view logistic fit, the class-balanced GBDT the archived
    table used, an unweighted GBDT and an isotonic-recalibrated GBDT, all scored
    on the same held-out rows. (PC audit item P1-7: the figure previously
    plotted the four-view fit under a "linear fit" label against the balanced
    GBDT only.)

Data: experiments/results/icdm_v2.json (pooled_per_view.redundancy,
pooled_request_split.calibration).
"""
import json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import BASELINE, OURS, ALT, REF, EDGE, BAR_EDGE_WIDTH, COLUMN_WIDTH, \
    use_paper_style, panel_labels, small
import matplotlib.pyplot as plt

OUT = "paper_icdm/figures"
d = json.load(open("experiments/results/icdm_v2.json"))

use_paper_style()
fig, (axA, axB) = plt.subplots(1, 2, figsize=(COLUMN_WIDTH, 1.62))

# (a) per-view relevance on the MI sample of the v2 corpus.
mi = d["pooled_per_view"]["redundancy"]["per_feature_mi"]
order = ["s_within", "s_cross", "s_query", "s_pos"]
labels = ["within", "cross", "query", "age"]
vals = [mi[k] for k in order]
axA.bar(range(4), vals, color=[OURS, OURS, BASELINE, BASELINE], width=0.66,
        edgecolor=EDGE, linewidth=BAR_EDGE_WIDTH)
axA.set_xticks(range(4)); axA.set_xticklabels(labels)
axA.set_ylabel("$I(X_i;Z)$")
axA.set_ylim(0, max(vals) * 1.26)
for i, v in enumerate(vals):
    # 0.000 would read as exactly zero; the age proxy is small, not absent.
    text = f"{v:.3f}" if v >= 5e-4 else "<0.001"
    axA.text(i, v + max(vals) * 0.03, text, ha="center", va="bottom", fontsize=small(1.0))

# (b) reliability of four fits on identical held-out rows.
cal = d["pooled_request_split"]["calibration"]
axB.plot([0, 1], [0, 1], ":", color=REF, lw=0.7)
series = [("within+cross(2)", "two-view logistic", "o", "-", OURS),
          ("GBDT balanced", "GBDT, balanced", "s", "--", ALT),
          ("GBDT unweighted", "GBDT, unweighted", "^", "-", "#2a7f62"),
          ("GBDT balanced+isotonic", "GBDT, isotonic", "v", "-.", "#8a6d1f")]
for key, short, marker, style, col in series:
    rc = cal[key]["reliability"]
    conf = np.array([c if c is not None else np.nan for c in rc["confidence"]], float)
    acc = np.array([c if c is not None else np.nan for c in rc["accuracy"]], float)
    axB.plot(conf, acc, marker=marker, linestyle=style, color=col, ms=2.4,
             label=f"{short} ({cal[key]['ece']:.3f})")
axB.set_xlabel("Predicted prob.")
axB.set_ylabel("Empirical freq.")
axB.set_xlim(0, 1); axB.set_ylim(0, 1)
axB.set_xticks([0, 0.5, 1]); axB.set_yticks([0, 0.5, 1])
fig.tight_layout(pad=0.2, w_pad=1.6, rect=(0, 0.09, 1, 0.845))
panel_labels(fig, (axA, axB), ("(a) View relevance, v2", "(b) Reliability (ECE), v2"))
# Four series will not fit inside the panel without covering the curves, so the
# legend for (b) sits above the figure in two rows.
handles, labels = axB.get_legend_handles_labels()
fig.legend(handles, labels, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.0),
           frameon=False, fontsize=small(1.0), labelspacing=0.2, handlelength=1.6,
           columnspacing=1.2)
os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, "fig_redund_calib.pdf"))
print(f"WROTE {OUT}/fig_redund_calib.pdf | relevance={[round(v, 3) for v in vals]} | "
      f"ECE={ {k: round(cal[k]['ece'], 3) for k in cal} }")

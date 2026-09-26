"""Exp#7 figure: served-oracle miss versus masked-loop answer F1.

Four archived Llama LongBench tasks at 20% retention. The point of the figure
is the disagreement between the two panels: the accumulator wins the proxy on
every task, while the answer-quality ordering flips across tasks. This is an
illustration of the proxy/quality gap, not the confirmatory equivalence test,
which uses the larger two-architecture set.

Request is the inferential unit (see the paper's statistical protocol), so
per-step miss is averaged within a request before averaging over requests.
This matches experiments/results/served_oracle_ci/c7_oracle_miss_pertask.json,
whose paired interval the text reports. Bars are normalised per task to the
H2O-style baseline; the absolute means are printed above them so no absolute
value hides behind a ratio.
"""
import json, glob, os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import BASELINE, OURS, EDGE, BAR_EDGE_WIDTH, COLUMN_WIDTH, ERRORBAR, \
    use_paper_style, top_legend, panel_labels, bar_values, small
import matplotlib.pyplot as plt

R = "experiments/results/e2e_confirm"
OUT = "paper_icdm/figures"
SHORT = {"hotpotqa": "Hotpot", "multifieldqa_en": "MFQA",
         "narrativeqa": "NarrQA", "qasper": "Qasper"}

datasets = sorted({os.path.basename(f).rsplit("_", 1)[0]
                   for f in glob.glob(f"{R}/*_h2o.json")})


def per_request(pol):
    """Per-dataset lists of per-request served-oracle miss and answer F1."""
    eps, f1 = [], []
    for ds in datasets:
        f = f"{R}/{ds}_{pol}.json"
        if not os.path.exists(f):
            raise FileNotFoundError(f"missing paired input: {f}")
        rows = json.load(open(f))["results"]
        eps.append(np.array([float(np.mean(r["per_step_eps_measured"])) for r in rows
                             if r.get("per_step_eps_measured")]))
        f1.append(np.array([float(r["f1"]) for r in rows]))
    return eps, f1


def mean_and_ci95(samples):
    """Mean and half-width of a 95% CI of the mean, over requests."""
    m = np.array([s.mean() for s in samples])
    half = np.array([1.96 * s.std(ddof=1) / np.sqrt(len(s)) for s in samples])
    return m, half


h2o_eps, h2o_f1 = per_request("h2o")
xqp_eps, xqp_f1 = per_request("xqp")
stats = {"miss": (mean_and_ci95(h2o_eps), mean_and_ci95(xqp_eps)),
         "f1": (mean_and_ci95(h2o_f1), mean_and_ci95(xqp_f1))}
for key, ((bm, _), (om, _)) in stats.items():
    print(f"{key:5s} H2O={bm.round(3).tolist()} mean={bm.mean():.4f} | "
          f"learned={om.round(3).tolist()} mean={om.mean():.4f}")

use_paper_style()
fig, (axA, axB) = plt.subplots(1, 2, figsize=(COLUMN_WIDTH, 1.66))
x = np.arange(len(datasets))
w = 0.38

for ax, key, ylabel in [(axA, "miss", "Norm. miss"), (axB, "f1", "Norm. F1")]:
    (base_m, base_e), (ours_m, ours_e) = stats[key]
    norm = base_m  # per task, relative to the H2O-style baseline
    for i, (label, colour, m, e) in enumerate([("H2O-style", BASELINE, base_m, base_e),
                                               ("learned 2-view", OURS, ours_m, ours_e)]):
        pos = x + (i - 0.5) * w
        ax.bar(pos, m / norm, width=w, yerr=e / norm, color=colour, label=label,
               edgecolor=EDGE, linewidth=BAR_EDGE_WIDTH, error_kw=ERRORBAR)
        bar_values(ax, pos, m / norm + e / norm, m, fmt="{:.2f}")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.75)
    ax.set_yticks([0, 0.5, 1.0, 1.5])
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[d] for d in datasets])
    # Four groups in a half-column panel: shrink the labels so they do not touch.
    ax.tick_params(axis="x", labelsize=small(1.0), pad=1.5)

fig.tight_layout(pad=0.2, w_pad=1.6, rect=(0, 0.10, 1, 0.90))
panel_labels(fig, (axA, axB), ("(a) Served-oracle miss", "(b) Answer F1"))
handles, labels = axA.get_legend_handles_labels()
top_legend(fig, handles, labels, y=1.0)
os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, "fig_proxy_vs_quality.pdf"))
print("WROTE", os.path.join(OUT, "fig_proxy_vs_quality.pdf"))

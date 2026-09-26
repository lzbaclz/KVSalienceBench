"""Forest plot of per-cell paired F1 differences (PC item P1-3).

Every architecture x dataset cell of a matched-retention sweep, learned scorer
minus the H2O-style accumulator, with a request-level 95% t interval, plus the
pooled cluster estimate. The global TOST average hides gains on some tasks and
losses on others; the plot shows that heterogeneity directly.

Input: the JSON written by experiments/analyze_expand_sensitivity.py.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import BASELINE, OURS, ALT, REF, COLUMN_WIDTH, use_paper_style, small
import matplotlib.pyplot as plt

SHORT = {"narrativeqa": "NarrQA", "qasper": "Qasper", "multifieldqa_en": "MFQA",
         "hotpotqa": "Hotpot", "2wikimqa": "2Wiki", "musique": "MuSiQue", "triviaqa": "Trivia"}
ARCH = {"llama31_8b": "Llama-3.1-8B", "qwen25_7b": "Qwen2.5-7B"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--analysis", default="experiments/results/tost/expand_v2_sensitivity.json")
    ap.add_argument("--contrast", default="xqp_vs_h2o::pooled::f1_longbench_all_refs")
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--out", default="paper_icdm/figures/fig_cell_forest.pdf")
    args = ap.parse_args(argv)
    d = json.load(open(args.analysis))["contrasts"][args.contrast]
    cells = d["cells"]
    use_paper_style()
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.35))
    y = np.arange(len(cells))[::-1] + 1.5
    for yi, c in zip(y, cells):
        col = OURS if c["arch"] == "llama31_8b" else ALT
        lo, hi = c["ci95_t"]
        ax.plot([lo, hi], [yi, yi], color=col, lw=0.9)
        ax.plot(c["mean"], yi, marker="o" if c["arch"] == "llama31_8b" else "s", color=col, ms=3)
    pooled = d["architecture_dataset_cells"][str(args.margin)]
    ax.plot(pooled["ci90_t"], [0.5, 0.5], color="black", lw=1.4)
    ax.plot(pooled["mean"], 0.5, marker="D", color="black", ms=3.2)
    ax.axvline(0, color=REF, lw=0.6, ls=":")
    ax.axvspan(-args.margin, args.margin, color=BASELINE, alpha=0.18, lw=0)
    ax.set_yticks(list(y) + [0.5])
    prefix = {"llama31_8b": "Llama", "qwen25_7b": "Qwen"}
    ax.set_yticklabels([f"{prefix[c['arch']]} {SHORT[c['dataset']]}" for c in cells]
                       + [f"pooled ({d['k_cells']} cells)"], fontsize=small(1.0))
    ax.set_xlabel("F1 difference, learned $-$ H2O-style")
    ax.set_ylim(0, len(cells) + 1)
    lim = max(abs(v) for c in cells for v in c["ci95_t"]) * 1.08
    ax.set_xlim(-lim, lim)
    handles = [plt.Line2D([], [], color=OURS, marker="o", ms=3, lw=0.9, label="Llama-3.1-8B"),
               plt.Line2D([], [], color=ALT, marker="s", ms=3, lw=0.9, label="Qwen2.5-7B"),
               plt.Line2D([], [], color="black", marker="D", ms=3.2, lw=1.4, label="pooled")]
    # Inside the axes the legend covered the Qwen intervals, so it goes above.
    fig.tight_layout(pad=0.2, rect=(0, 0, 1, 0.94))
    fig.legend(handles=handles, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               frameon=False, fontsize=small(1.0), columnspacing=1.2, handlelength=1.6)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out)
    print("WROTE", args.out, "pooled", pooled["mean"], pooled["ci90_t"])


if __name__ == "__main__":
    main()

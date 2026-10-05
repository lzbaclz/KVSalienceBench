"""Forest plot of per-cell paired F1 differences (PC item P1-3).

Every architecture x dataset cell of a matched-retention sweep, the reconstructed
scorer minus the H2O-style accumulator, with a request-level 95% t interval. The two
architectures answer the same questions, so the primary summary averages them within
each dataset (seven clusters, df 6, 90% t interval); the 14-cell summary, which treats
the cells as independent, is the sensitivity analysis and is drawn hollow. Cells are
ordered dataset by dataset with the two models adjacent so the pairing is visible. The
global TOST average hides gains on some tasks and losses on others; the plot shows that
heterogeneity directly.

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
    order = list(SHORT)                     # dataset order of the manuscript
    cells = sorted(d["cells"], key=lambda c: (order.index(c["dataset"]), c["arch"] != "llama31_8b"))
    use_paper_style()
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.45))
    y = np.arange(len(cells))[::-1] + 2.5
    for yi, c in zip(y, cells):
        col = OURS if c["arch"] == "llama31_8b" else ALT
        lo, hi = c["ci95_t"]
        ax.plot([lo, hi], [yi, yi], color=col, lw=0.9)
        ax.plot(c["mean"], yi, marker="o" if c["arch"] == "llama31_8b" else "s", color=col, ms=3)
    # primary: models averaged within each dataset (7 clusters, df 6); sensitivity: 14 cells
    primary = d["dataset_clusters"][str(args.margin)]
    sens = d["architecture_dataset_cells"][str(args.margin)]
    ax.plot(primary["ci90_t"], [1.5, 1.5], color="black", lw=1.4)
    ax.plot(primary["mean"], 1.5, marker="D", color="black", ms=3.2)
    ax.plot(sens["ci90_t"], [0.5, 0.5], color="#555555", lw=1.0)
    ax.plot(sens["mean"], 0.5, marker="D", color="#555555", mfc="white", ms=3.2)
    ax.axhline(2.0, color=REF, lw=0.4)
    ax.axvline(0, color=REF, lw=0.6, ls=":")
    ax.axvspan(-args.margin, args.margin, color=BASELINE, alpha=0.18, lw=0)
    ax.set_yticks(list(y) + [1.5, 0.5])
    prefix = {"llama31_8b": "Llama", "qwen25_7b": "Qwen"}
    ax.set_yticklabels([f"{prefix[c['arch']]} {SHORT[c['dataset']]}" for c in cells]
                       + [f"pooled, {primary['n_clusters']} datasets", f"pooled, {sens['n_clusters']} cells"],
                       fontsize=small(1.0))
    ax.set_xlabel("F1 difference, reconstructed $-$ H2O-style")
    ax.set_ylim(0, len(cells) + 2)
    lim = max(abs(v) for c in cells for v in c["ci95_t"]) * 1.08
    ax.set_xlim(-lim, lim)
    # No legend: every row label names its model, and colour/marker repeat it.
    fig.tight_layout(pad=0.2)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out)
    print("WROTE", args.out, "primary (7 datasets)", primary["mean"], primary["ci90_t"],
          "| sensitivity (14 cells)", sens["mean"], sens["ci90_t"])


if __name__ == "__main__":
    main()

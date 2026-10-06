#!/usr/bin/env python3
"""Print the LaTeX rows of the manuscript's version-2 offline table, so that no number
is retyped by hand. ``--block v2`` (default) prints Table II's ranking-and-calibration
rows from experiments/results/icdm_v2.json; ``--block dec`` prints the second block
(pooled AUC split by pair type, and per-decision recall) from
experiments/results/icdm_v2_decomposition.json.

Block v2 columns: version-2 AUC under the request-level split (the source-prompt split
agrees to three decimals and is asserted in the tests), tie-aware AUPRC, pooled top-decile
precision, per-decision (macro) top-decile precision and equal-width ECE; all from the
request-level split. ECE is printed only for scorers that emit probabilities: the four
single-signal rows are raw scores, so an ECE for them would compare a quantity that was
never meant to be a probability with fitted models, and prints ``---``.

Tie rule of the per-decision column. The stored value uses the stable row-order rule. A
cell whose value that rule decides (it moves at the printed precision under exact random
tie-breaking, experiments/results/icdm_v2_tie_sensitivity.json) is printed as the
random-tie EXPECTATION and marked with a dagger. Only the age proxy qualifies: its prompt
blocks share one creation step, so 99.9% of its rows tie and row order picked the blocks.

Block dec columns: pooled AUC, AUC over same-decision pairs, AUC over cross-decision
pairs, and per-decision recall at 10% and 20% retention, each on EVERY held-out row.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROWS = [  # (label in the paper, method name in the JSON)
    ("age proxy", "recency"),
    ("query proxy (cosine)", "Quest"),
    ("prev-layer indicator", "InfiniGen"),
    ("within-layer EMA", "H2O/attn-EMA"),
    ("LightGBM, balanced", "GBDT(LightGBM)"),
    ("two-view logistic, refit", "within+cross(2)"),
    ("two-view, archived weights", "archived two-view checkpoint (frozen, v2 features)"),
]
# Rows left out of the paper for space (they stay in the JSON and the artifact): fitted MLP,
# two-view + dot-max, four-view logistic. Their headline numbers are quoted in the text.
RAW_SCORES = {"recency", "Quest", "InfiniGen", "H2O/attn-EMA"}   # not probabilities: no ECE
DEC_ROWS = [("within-layer EMA", "within-layer EMA"),
            ("two-view logistic, refit", "two-view logistic (refit)"),
            ("LightGBM, balanced", "LightGBM, balanced")]


def fmt(x):
    return f"{x:.3f}"[1:] if 0 <= x < 1 else f"{x:.3f}"


def decision_cell(label, stored):
    """The per-decision top-decile cell: the stored (stable-rule) value, or the random-tie
    expectation with a dagger when the tie rule decides the printed digits."""
    t = json.loads((ROOT / "experiments/results/icdm_v2_tie_sensitivity.json").read_text())
    row = t["splits"]["request"]["rows"][label]
    if abs(row["recall_0.10"] - stored) > 5e-4:
        raise ValueError(f"{label}: tie-sensitivity record disagrees with icdm_v2.json")
    if abs(row["delta_0.10"]) >= t["config"]["print_tolerance"]:
        return fmt(row["recall_0.10_random_ties"]) + "$^{\\dagger}$"
    return fmt(stored)


def v2_rows():
    d = json.loads((ROOT / "experiments/results/icdm_v2.json").read_text())
    req = {r["method"]: r for r in d["pooled_request_split"]["table"]}
    out = []
    for label, key in ROWS:
        r = req[key]
        ece = "---" if key in RAW_SCORES else fmt(r["ece"])
        out.append(f"{label} & {fmt(r['auc'])} & {fmt(r['auprc'])} & "
                   f"{fmt(r['p_at_10_pooled'])} & {decision_cell(label, r['p_at_10_grouped_macro'])} & {ece} \\\\")
    return out


def dec_rows():
    d = json.loads((ROOT / "experiments/results/icdm_v2_decomposition.json").read_text())
    out = []
    for label, key in DEC_ROWS:
        r = d["scorers"][key]
        out.append(f"{label} & {fmt(r['auc_pooled'])} & {fmt(r['auc_same'])} & {fmt(r['auc_cross'])} & "
                   f"{fmt(r['recall_0.10'])} & {fmt(r['recall_0.20'])} \\\\")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--block", choices=("v2", "dec"), default="v2")
    a = ap.parse_args(argv)
    print("\n".join(v2_rows() if a.block == "v2" else dec_rows()))
    return 0


if __name__ == "__main__":
    sys.exit(main())

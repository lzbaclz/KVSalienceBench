#!/usr/bin/env python3
"""Print the LaTeX rows of the manuscript's version-2 offline table from
experiments/results/icdm_v2.json, so that no number is retyped by hand.

Columns: v2 AUC under the request-level split (the source-prompt
split agrees to three decimals and is asserted in the tests), tie-aware AUPRC, pooled top-decile precision, per-decision (macro) top-decile
precision, and equal-width ECE; the last four are from the request-level split.
"""
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
    ("fitted MLP", "sklearnMLP(16,)"),
    ("two-view logistic, refit", "within+cross(2)"),
    ("two-view, archived weights", "archived two-view checkpoint (frozen, v2 features)"),
    ("two-view + dot-max", "within+cross+dotmax logistic"),
    ("four-view logistic", "XQP-closed"),
]


def fmt(x):
    return f"{x:.3f}"[1:] if 0 <= x < 1 else f"{x:.3f}"


def main():
    d = json.loads((ROOT / "experiments/results/icdm_v2.json").read_text())
    req = {r["method"]: r for r in d["pooled_request_split"]["table"]}
    src = {r["method"]: r for r in d["pooled_source_split"]["table"]}
    for label, key in ROWS:
        r, s = req[key], src[key]
        print(f"{label} & {fmt(r['auc'])} & {fmt(r['auprc'])} & "
              f"{fmt(r['p_at_10_pooled'])} & {fmt(r['p_at_10_grouped_macro'])} & {fmt(r['ece'])} \\\\")
    return 0


if __name__ == "__main__":
    sys.exit(main())

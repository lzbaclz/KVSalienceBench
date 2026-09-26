#!/usr/bin/env python3
"""Print Exp#8's pinned-rerun table from the sensitivity JSON.

Columns: official LongBench F1 with all references, delta and 14-cell TOST p.
Historical v1 columns remain in the artifact and are omitted to prioritize v2.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROWS = [("full cache", "full"), ("H2O-style accumulation", "h2o"), ("reconstructed two-view", "xqp"),
        ("Ada-KV-style allocation", "adakv"), ("PyramidKV-style allocation", "pyramidkv")]


def cols(analysis, metric):
    means = analysis["policy_means"]
    base = means["h2o"][metric]["mean"]
    out = {}
    for _, pol in ROWS:
        f1 = means[pol][metric]["mean"]
        if pol == "h2o":
            out[pol] = (f1, None, None)
            continue
        key = f"{pol}_vs_h2o::pooled::{metric}"
        p = analysis["contrasts"][key]["architecture_dataset_cells"]["0.02"]["p"] if key in analysis["contrasts"] else None
        out[pol] = (f1, f1 - base, p)
    return out


def fmt3(x):
    return f"{x:.3f}"[1:] if x >= 0 else "-" + f"{-x:.3f}"[1:]


def fmt_delta(d):
    return "---" if d is None else (("+" if d >= 0 else "-") + f"{abs(d):.3f}"[1:])


def fmt_p(p):
    """p >= 0.01 to three decimals, smaller ones to four, as the archived table did."""
    if p is None:
        return "---"
    if p >= 0.01:
        return f"{p:.3f}"[1:]
    return f"{p:.4f}"[1:] if p >= 0.0001 else f"{p:.1e}"


def main(v2="experiments/results/tost/expand_v2_sensitivity.json"):
    b = cols(json.loads((ROOT / v2).read_text()), "f1_longbench_all_refs")
    for label, pol in ROWS:
        f1b, db, pb = b[pol]
        print(f"{label} & {fmt3(f1b)} & {fmt_delta(db)} & {fmt_p(pb)} \\\\")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))

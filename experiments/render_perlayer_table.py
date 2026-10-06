#!/usr/bin/env python3
"""Print Exp#8's pinned-rerun table from the sensitivity JSON.

Columns: official LongBench F1 with all references; the difference from the H2O-style
accumulator with its 90% t interval over the seven datasets; and the TOST p value at the
two-point margin, p_TOST,+-.02. The seven datasets are the primary unit of inference: the
two architectures answer the same questions, so the two models are averaged within each
dataset (df 6). The 14 architecture-by-dataset cells treat the cells as independent and
are reported in the text as the sensitivity analysis. Historical v1 columns remain in the
artifact and are omitted to prioritize v2.

The full-cache row carries the same seven-dataset interval, read from
experiments/results/tost/expand_v2_full_cache_gap.json (analyze_full_cache_gap.py); it has
no TOST p because the question there is the size of the loss, not equivalence. A bound
that is not zero but would print as a signed zero (``-.000``) gets one more decimal.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROWS = [("full cache", "full"), ("H2O-style", "h2o"), ("reconstructed", "xqp"),
        ("Ada-KV-style", "adakv"), ("PyramidKV-style", "pyramidkv")]
MARGIN = "0.02"


def cols(analysis, metric, gap):
    means = analysis["policy_means"]
    base = means["h2o"][metric]["mean"]
    out = {}
    for _, pol in ROWS:
        f1 = means[pol][metric]["mean"]
        if pol == "h2o":
            out[pol] = (f1, None, None, None)
            continue
        key = f"{pol}_vs_h2o::pooled::{metric}"
        if key in analysis["contrasts"]:
            c = analysis["contrasts"][key]["dataset_clusters"][MARGIN]
            out[pol] = (f1, f1 - base, c["p"], c["ci90_t"])
        else:
            g = gap["contrasts"][key]
            if abs(g["grand_mean"] - (f1 - base)) > 1e-12:
                raise ValueError(f"{key}: gap record disagrees with the policy means")
            out[pol] = (f1, f1 - base, None, g["dataset_clusters"]["ci90_t"])
    return out


def fmt3(x):
    return f"{x:.3f}"[1:] if x >= 0 else "-" + f"{-x:.3f}"[1:]


def fmt_signed(d):
    digits = 3 if d == 0 or round(abs(d), 3) > 0 else 4      # never print a nonzero bound as -.000
    return ("+" if d >= 0 else "-") + f"{abs(d):.{digits}f}"[1:]


def fmt_delta(d, ci):
    if d is None:
        return "---"
    if ci is None:
        return fmt_signed(d)
    return f"{fmt_signed(d)} [{fmt_signed(ci[0])}, {fmt_signed(ci[1])}]"


def fmt_p(p):
    """p >= 0.01 to three decimals, smaller ones to four, as the archived table did."""
    if p is None:
        return "---"
    if p >= 0.01:
        return f"{p:.3f}"[1:]
    return f"{p:.4f}"[1:] if p >= 0.0001 else f"{p:.1e}"


def main(v2="experiments/results/tost/expand_v2_sensitivity.json",
         gap="experiments/results/tost/expand_v2_full_cache_gap.json"):
    b = cols(json.loads((ROOT / v2).read_text()), "f1_longbench_all_refs", json.loads((ROOT / gap).read_text()))
    for label, pol in ROWS:
        f1b, db, pb, cb = b[pol]
        print(f"{label} & {fmt3(f1b)} & {fmt_delta(db, cb)} & {fmt_p(pb)} \\\\")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))

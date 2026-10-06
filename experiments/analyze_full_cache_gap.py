#!/usr/bin/env python3
"""Exp#8: the compression loss against full cache, with the interval Table IV gives its other rows.

``analyze_expand_sensitivity.py`` contrasts every compressed policy with the H2O-style
baseline and skips the uncompressed run, so Table IV printed the full-cache difference
as a bare point estimate under a caption that promised an interval for the column.
This driver computes that contrast with the SAME code path and unit of inference
(``analyze_contrast``: official LongBench scorer, all references; the two architectures
averaged within each dataset; 90% t interval over the seven datasets, df 6), and the
loss of each compressed policy against full cache the same way.

No equivalence test is attached to these contrasts: the question for the full-cache
row is how much quality compression costs, not whether it is within a margin.

It reads the per-request records of the pinned rerun and the local LongBench rows (for
all references) and writes a NEW record; ``expand_v2_sensitivity.json`` is not touched.

Usage:
  python experiments/analyze_full_cache_gap.py --root experiments/results/expand_v2 \
      --out experiments/results/tost/expand_v2_full_cache_gap.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from analyze_expand_sensitivity import ARCHS, analyze_contrast, load_cells  # noqa: E402

METRIC = "f1_longbench_all_refs"
COMPRESSED = ("h2o", "xqp", "adakv", "pyramidkv")


def gap(cells, pol_a, pol_b, archs):
    """``pol_a - pol_b`` per cell and its seven-dataset 90% t interval (no margin test)."""
    c = analyze_contrast(cells, pol_a, pol_b, archs, METRIC)
    out = dict(policy_a=pol_a, policy_b=pol_b, archs=list(archs), metric=METRIC, k_cells=c["k_cells"],
               n_items=c["n_items"], grand_mean=c["grand_mean"], cells=c["cells"])
    for unit in ("architecture_dataset_cells", "dataset_clusters"):
        if unit in c:
            t = c[unit]["0.02"]          # mean, se, df and the 90% t interval do not depend on the margin
            out[unit] = {k: t[k] for k in ("n_clusters", "mean", "se", "df", "ci90_t")}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT / "experiments/results/expand_v2"))
    ap.add_argument("--longbench-dir", default="/public/data_zoo/longbench/data")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite {a.out}")
    lb = Path(a.longbench_dir)
    if not lb.is_dir():
        raise SystemExit(f"LongBench directory missing: {lb} (all references are needed)")
    cells, provenance = load_cells(Path(a.root), lb, ("full",) + COMPRESSED)
    if provenance["reference_row_mismatches"] or provenance["rows"] != 4480:
        raise SystemExit(f"unexpected provenance: {provenance}")
    report = dict(schema="expand-full-cache-gap-v1", root=str(Path(a.root)), provenance=provenance,
                  note=("full minus <policy>, official LongBench scorer, maximum over all references; "
                        "dataset_clusters averages the two architectures within each dataset (df 6)"),
                  contrasts={})
    for pol in COMPRESSED:
        for label, archs in (("pooled", ARCHS), ("llama", ARCHS[:1]), ("qwen", ARCHS[1:])):
            report["contrasts"][f"full_vs_{pol}::{label}::{METRIC}"] = gap(cells, "full", pol, archs)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2) + "\n")
    for pol in COMPRESSED:
        c = report["contrasts"][f"full_vs_{pol}::pooled::{METRIC}"]
        lo, hi = c["dataset_clusters"]["ci90_t"]
        print(f"full - {pol:<10} {c['grand_mean']:+.4f}  seven-dataset 90% t interval [{lo:+.4f}, {hi:+.4f}]")
    print(f"WROTE {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

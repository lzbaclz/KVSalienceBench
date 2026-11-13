#!/usr/bin/env python3
"""Sensitivity re-analysis of a matched-retention expansion sweep (PC items P0-2, P1-2, P1-3).

Given a results root laid out as ``<root>/<arch>/<dataset>_<policy>.json`` (the
SEER runner's per-request records with ``id``, ``pred``, ``ref``, ``f1``), this
script re-derives, without modifying any input:

* **Scorer provenance (P0-2).** Every stored F1 is re-derived with the runner's
  own normalization (articles removed before punctuation, single stored
  reference), and alongside it the official LongBench ``qa_f1_score`` (SQuAD
  normalization order, maximum over *all* references, first-line truncation for
  ``triviaqa``). References are recovered from the local LongBench rows because
  the runner's default sampling is the canonical first-N rows, so request id i
  is source row i. Row order is verified against the stored single reference.
* **Inferential unit (P1-2).** Paired TOST on 14 architecture x dataset cell
  means (df=13) and on 7 dataset clusters after averaging the two
  architectures (df=6), at margins 0.005..0.03, reporting the t-based 90%
  interval and the percentile cluster bootstrap 90% interval as two different
  procedures, plus a Llama-only analysis.
* **Per-cell heterogeneity (P1-3).** Every cell's mean paired difference with
  a request-level 95% t interval, for the forest plot.

Usage:
  python experiments/analyze_expand_sensitivity.py --root experiments/results/expand \
      --out experiments/results/tost/expand_tost_sensitivity.json
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import string
import sys

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
ARCHS = ("llama31_8b", "qwen25_7b")
DATASETS = ("narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "triviaqa")
MARGINS = (0.005, 0.01, 0.015, 0.02, 0.03)


# ------------------------------------------------------------------ scorers
def _f1(p, r):
    if not p and not r:
        return 1.0
    if not p or not r:
        return 0.0
    c = sum((Counter(p) & Counter(r)).values())
    if c == 0:
        return 0.0
    prec, rec = c / len(p), c / len(r)
    return 2 * prec * rec / (prec + rec)


def runner_normalize(s: str) -> list[str]:
    """SEER ``seer.eval.metrics.normalize``: articles BEFORE punctuation."""
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split()).split()


def longbench_normalize(s: str) -> list[str]:
    """LongBench/SQuAD ``normalize_answer``: punctuation BEFORE articles."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split()).split()


def runner_f1(pred: str, ref: str) -> float:
    return _f1(runner_normalize(pred), runner_normalize(ref))


def longbench_f1(pred: str, refs, dataset: str) -> float:
    if dataset in ("trec", "triviaqa", "samsum", "lsht"):
        pred = pred.lstrip("\n").split("\n")[0]
    return max(_f1(longbench_normalize(pred), longbench_normalize(r)) for r in refs)


# ------------------------------------------------------------------ loading
def load_cells(root: Path, longbench_dir: Path | None, policies):
    """Return {(arch, dataset, policy): {id: record}} with re-derived scores."""
    cells, provenance = {}, {"stored_f1_reproduced_by_runner_scorer": 0, "rows": 0,
                             "reference_row_mismatches": 0}
    lb_rows = {}
    for arch in ARCHS:
        for ds in DATASETS:
            if longbench_dir is not None and ds not in lb_rows:
                path = longbench_dir / f"{ds}.jsonl"
                lb_rows[ds] = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
            for pol in policies:
                f = root / arch / f"{ds}_{pol}.json"
                if not f.exists():
                    continue
                obj = json.loads(f.read_text())
                rows = obj["results"]
                if len({r["id"] for r in rows}) != len(rows):
                    raise ValueError(f"duplicate ids in {f}")
                recs = {}
                for r in rows:
                    rec = dict(id=r["id"], f1_stored=float(r["f1"]), pred=r["pred"], ref=r["ref"],
                               f1_runner_scorer=runner_f1(r["pred"], r["ref"]))
                    provenance["rows"] += 1
                    provenance["stored_f1_reproduced_by_runner_scorer"] += int(
                        abs(rec["f1_runner_scorer"] - rec["f1_stored"]) < 1e-9)
                    if longbench_dir is not None:
                        src = lb_rows[ds][r["id"]]
                        refs = src.get("answers") or []
                        if not refs or refs[0] != r["ref"]:
                            provenance["reference_row_mismatches"] += 1
                        rec["n_references"] = len(refs)
                        rec["f1_longbench_all_refs"] = longbench_f1(r["pred"], refs, ds)
                        rec["f1_longbench_single_ref"] = longbench_f1(r["pred"], [r["ref"]], ds)
                    recs[r["id"]] = rec
                cells[(arch, ds, pol)] = recs
    return cells, provenance


# ------------------------------------------------------------------ inference
def tost_t(values, margin):
    v = np.asarray(values, float)
    if v.size < 2:
        raise ValueError("TOST needs at least two clusters")
    mean, se = float(v.mean()), float(stats.sem(v))
    if se == 0:
        raise ValueError("zero-variance clusters")
    df = v.size - 1
    p = max(stats.t.sf((mean + margin) / se, df), stats.t.cdf((mean - margin) / se, df))
    lo, hi = stats.t.interval(0.90, df, loc=mean, scale=se)
    return dict(n_clusters=int(v.size), mean=mean, se=se, df=int(df), margin=margin, p=float(p),
                ci90_t=[float(lo), float(hi)], equivalent_at_005=bool(p < 0.05))


def cluster_bootstrap_ci90(cell_diffs, boot=10000, seed=1234):
    """Percentile 90% interval of the grand mean, resampling whole cells (as the
    archived driver did). This is a different procedure from the t interval."""
    rng = np.random.default_rng(seed)
    k = len(cell_diffs)
    draws = np.empty(boot)
    for j in range(boot):
        idx = rng.integers(0, k, k)
        draws[j] = np.concatenate([cell_diffs[i] for i in idx]).mean()
    return [float(np.percentile(draws, 5)), float(np.percentile(draws, 95))]


def analyze_contrast(cells, pol_a, pol_b, archs, metric):
    per_cell = []
    for arch in archs:
        for ds in DATASETS:
            a, b = cells.get((arch, ds, pol_a)), cells.get((arch, ds, pol_b))
            if not a or not b:
                continue
            ids = sorted(set(a) & set(b))
            if len(ids) != len(a) or len(ids) != len(b):
                raise ValueError(f"unpaired ids in {arch}/{ds}: {pol_a} vs {pol_b}")
            d = np.array([a[i][metric] - b[i][metric] for i in ids])
            ci = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d)) if d.std(ddof=1) > 0 else (d.mean(), d.mean())
            per_cell.append(dict(arch=arch, dataset=ds, n=int(len(d)), mean=float(d.mean()),
                                 ci95_t=[float(ci[0]), float(ci[1])],
                                 mean_a=float(np.mean([a[i][metric] for i in ids])),
                                 mean_b=float(np.mean([b[i][metric] for i in ids])), _diffs=d))
    if not per_cell:
        return None
    cell_means = np.array([c["mean"] for c in per_cell])
    out = dict(policy_a=pol_a, policy_b=pol_b, archs=list(archs), metric=metric,
               k_cells=len(per_cell), n_items=int(sum(c["n"] for c in per_cell)),
               grand_mean=float(cell_means.mean()),
               cells=[{k: v for k, v in c.items() if k != "_diffs"} for c in per_cell],
               ci90_cluster_bootstrap=cluster_bootstrap_ci90([c["_diffs"] for c in per_cell]),
               architecture_dataset_cells={str(m): tost_t(cell_means, m) for m in MARGINS})
    if len(archs) > 1:
        by_ds = {}
        for c in per_cell:
            by_ds.setdefault(c["dataset"], []).append(c["mean"])
        ds_means = np.array([np.mean(v) for _, v in sorted(by_ds.items())])
        out["dataset_clusters"] = {str(m): tost_t(ds_means, m) for m in MARGINS}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(ROOT / "experiments/results/expand"))
    ap.add_argument("--longbench-dir", default="/public/data_zoo/longbench/data",
                    help="local LongBench rows for all references; pass '' to skip rescoring")
    ap.add_argument("--policies", default="full,h2o,xqp,adakv,pyramidkv")
    ap.add_argument("--baseline", default="h2o")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    root = Path(args.root)
    lb = Path(args.longbench_dir) if args.longbench_dir else None
    if lb is not None and not lb.is_dir():
        raise SystemExit(f"LongBench directory missing: {lb}; pass --longbench-dir '' to skip rescoring")
    policies = args.policies.split(",")
    cells, provenance = load_cells(root, lb, policies)
    metrics = ["f1_stored"] + (["f1_longbench_all_refs", "f1_longbench_single_ref"] if lb else [])
    report = dict(schema="expand-sensitivity-v1", root=str(root), provenance=provenance,
                  scorer_note=("f1_stored is what the runner wrote; f1_runner_scorer re-derives it with the "
                               "runner's articles-before-punctuation normalization; f1_longbench_* use the "
                               "official LongBench order and (all_refs) the maximum over every reference"),
                  policy_means={}, contrasts={})
    for pol in policies:
        for metric in metrics:
            vals = [r[metric] for (a, d, p), recs in cells.items() if p == pol for r in recs.values()]
            if vals:
                report["policy_means"].setdefault(pol, {})[metric] = dict(mean=float(np.mean(vals)), n=len(vals))
    for pol in policies:
        if pol in (args.baseline, "full"):
            continue
        for metric in metrics:
            for label, archs in (("pooled", ARCHS), ("llama", ARCHS[:1]), ("qwen", ARCHS[1:])):
                res = analyze_contrast(cells, pol, args.baseline, archs, metric)
                if res is not None:
                    report["contrasts"][f"{pol}_vs_{args.baseline}::{label}::{metric}"] = res
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    main_key = f"xqp_vs_{args.baseline}::pooled::f1_stored"
    if main_key in report["contrasts"]:
        c = report["contrasts"][main_key]
        print(f"pooled stored: mean {c['grand_mean']:+.5f}, 14-cell p(.02)={c['architecture_dataset_cells']['0.02']['p']:.5f}, "
              f"7-dataset p(.02)={c['dataset_clusters']['0.02']['p']:.5f}, "
              f"t-CI90 {c['architecture_dataset_cells']['0.02']['ci90_t']}, boot-CI90 {c['ci90_cluster_bootstrap']}")
    print(f"provenance: {provenance}")
    print(f"WROTE {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

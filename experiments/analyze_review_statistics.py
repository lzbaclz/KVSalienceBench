#!/usr/bin/env python3
"""CPU-only source-prompt bootstrap sensitivity for the final mentor review.

Refit the three headline scorers on the original v2 training sample, verify
their point estimates against the archived table, and resample source prompts
jointly across models. Both the original request split and the source-disjoint
split are evaluated. No test row is used for fitting. The original JSON stays
unchanged; sufficient statistics permit inexpensive independent CI replay.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from run_icdm_full import HEADLINE_H, SEED, subsample, roc_auc
from run_icdm_v2 import TRAIN_N, TEST_N, load_v2, pool, request_split_v2, decision_groups
from xqp.baselines import fit_gbdt
from xqp.dm_metrics import precision_recall_at_k_grouped
from xqp.predictor import ClosedFormXQP

LABELS = ("within-layer EMA", "two-view logistic (refit)", "LightGBM, balanced")
KEYS = ("H2O/attn-EMA", "within+cross(2)", "GBDT(LightGBM)")


def auc_sufficient_statistics(y, score, groups):
    """A[g,h] counts positive(g)>negative(h), with half credit for ties.

    With cluster multiplicities w, pooled AUC is (w'A w)/(w'p * w'n).
    This is exactly the row-duplication bootstrap, including between-cluster
    comparisons; it is NOT an average of within-request AUCs.
    """
    y, score, groups = map(np.asarray, (y, score, groups))
    if not (y.ndim == score.ndim == groups.ndim == 1 and len(y) == len(score) == len(groups)):
        raise ValueError("aligned one-dimensional arrays required")
    if not len(y) or not np.isfinite(score).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("finite scores and nonempty binary labels required")
    names, group = np.unique(groups, return_inverse=True)
    order = np.argsort(score, kind="stable")
    s, yy, gg = score[order], y[order], group[order]
    first = np.r_[0, np.flatnonzero(s[1:] != s[:-1]) + 1]
    lengths = np.diff(np.r_[first, len(s)])
    pos = yy == 1
    credit = np.zeros((len(names), len(names)))
    for h in range(len(names)):
        negatives = np.add.reduceat(((yy == 0) & (gg == h)).astype(float), first)
        below_plus_half_tied = np.cumsum(negatives) - 0.5 * negatives
        row_credit = np.repeat(below_plus_half_tied, lengths)
        credit[:, h] = np.bincount(gg[pos], weights=row_credit[pos], minlength=len(names))
    return {"groups": names.tolist(), "positive": np.bincount(group[y == 1], minlength=len(names)).tolist(),
            "negative": np.bincount(group[y == 0], minlength=len(names)).tolist(),
            "pair_credit": credit.tolist()}


def bootstrap_auc(stats, counts):
    p, n, a = (np.asarray(stats[k]) for k in ("positive", "negative", "pair_credit"))
    denom = (counts @ p) * (counts @ n)
    if np.any(denom == 0):
        raise ValueError("bootstrap draw has only one label class")
    return np.einsum("bi,ij,bj->b", counts, a, counts, optimize=True) / denom


def interval(values):
    return np.quantile(values, [0.025, 0.975]).tolist()


def cluster_counts(n, n_boot, seed):
    # Match the existing request bootstrap's rng.choice sampling scheme.
    draws = np.random.default_rng(seed).integers(n, size=(n_boot, n))
    return np.array([np.bincount(row, minlength=n) for row in draws])


def analyze_split(d, kind, archived, n_boot, seed):
    train, test, test_requests = request_split_v2(d, kind, seed=seed)
    tr, te = subsample(train, TRAIN_N, seed), subsample(test, TEST_N, seed)
    ftr, ytr = d["F"][tr], d["y"][HEADLINE_H][tr].astype(np.float32)
    mask = np.array([1, 1, 0, 0], np.float32)
    two = ClosedFormXQP.from_fit(ftr * mask, ytr)
    gb = fit_gbdt(ftr, ytr, seed=seed)
    if gb.name != KEYS[2]:
        raise ValueError("LightGBM is required; fallback would change the experiment")
    scorers = (lambda f: f[:, 0], lambda f: two.score(f * mask), gb.score)
    reference = {r["method"]: r for r in archived[f"pooled_{kind}_split"]["table"]}
    y = d["y"][HEADLINE_H][te]
    scores = {label: np.asarray(fn(d["F"][te])) for label, fn in zip(LABELS, scorers)}
    observed = {label: roc_auc(y, scores[label]) for label in LABELS}
    for label, key in zip(LABELS, KEYS):
        if not np.isclose(observed[label], reference[key]["auc"], atol=1e-8, rtol=0):
            raise ValueError(f"{kind}/{label}: refit AUC differs from archive: {observed[label]} vs {reference[key]['auc']}")

    sources = sorted(set(tuple(d["source"][r]) for r in test_requests))
    source_index = {s: i for i, s in enumerate(sources)}
    source_of_request = {r: source_index[tuple(d["source"][r])] for r in test_requests}
    out = {"held_out_requests": test_requests, "source_prompts": sources,
           "source_of_request": source_of_request, "auc": {}, "operating_points": {}}
    for unit, groups in (("model_request", d["rid"][te]),
                         ("source_prompt", np.array([source_of_request[r] for r in d["rid"][te]]))):
        counts = cluster_counts(len(np.unique(groups)), n_boot, seed)
        stats = {label: auc_sufficient_statistics(y, scores[label], groups) for label in LABELS}
        boot = {label: bootstrap_auc(stats[label], counts) for label in LABELS}
        out["auc"][unit] = {"n_clusters": counts.shape[1], "sufficient_statistics": stats,
                            "per_scorer": {label: {"auc": observed[label], "ci95": interval(boot[label])}
                                           for label in LABELS},
                            "two_view_minus_gbdt": {"mean": observed[LABELS[1]] - observed[LABELS[2]],
                                                    "ci95": interval(boot[LABELS[1]] - boot[LABELS[2]])}}

    # Complete cache decisions, then one macro mean per model-request.
    rows = []
    for r in test_requests:
        idx = test[d["rid"][test] == r]
        f, yr, groups = d["F"][idx], d["y"][HEADLINE_H][idx], decision_groups(d, idx)
        sr = {label: fn(f) for label, fn in zip(LABELS, scorers)}
        record = {"request": int(r), "source": source_of_request[r], "values": {}}
        for k in (0.1, 0.2):
            record["values"][f"{k:.2f}"] = {label: precision_recall_at_k_grouped(yr, sr[label], groups, k)["macro_recall"]
                                            for label in LABELS}
        rows.append(record)
    out["per_request"] = rows
    for k in ("0.10", "0.20"):
        a = np.array([[r["values"][k][label] for label in LABELS] for r in rows])
        out["operating_points"][k] = {"per_scorer": dict(zip(LABELS, a.mean(0).tolist()))}
        for unit, membership in (("model_request", np.arange(len(rows))),
                                 ("source_prompt", np.array([r["source"] for r in rows]))):
            n = len(np.unique(membership))
            counts = cluster_counts(n, n_boot, seed)
            size = np.bincount(membership, minlength=n)
            sums = np.array([np.bincount(membership, weights=a[:, j], minlength=n) for j in range(3)]).T
            boot = (counts @ sums) / (counts @ size)[:, None]
            out["operating_points"][k][unit] = {
                label: {"mean": float((a[:, 0] - a[:, j]).mean()), "ci95": interval(boot[:, 0] - boot[:, j]),
                        "n_positive": int((a[:, 0] > a[:, j]).sum()), "n_requests": len(rows), "n_clusters": n}
                for j, label in enumerate(LABELS) if j}
    return out


def replay_intervals(result):
    """Verify every saved CI from portable statistics, without raw traces."""
    checked = 0
    for kind in ("request", "source"):
        block = result[kind]
        for unit, data in block["auc"].items():
            counts = cluster_counts(data["n_clusters"], result["n_boot"], result["seed"])
            boot = {}
            for label in LABELS:
                stats = data["sufficient_statistics"][label]
                boot[label] = bootstrap_auc(stats, counts)
                observed = bootstrap_auc(stats, np.ones((1, data["n_clusters"])))[0]
                np.testing.assert_allclose(observed, data["per_scorer"][label]["auc"], atol=1e-12, rtol=0)
                np.testing.assert_allclose(interval(boot[label]), data["per_scorer"][label]["ci95"], atol=1e-12, rtol=0)
                checked += 1
            np.testing.assert_allclose(interval(boot[LABELS[1]] - boot[LABELS[2]]),
                                       data["two_view_minus_gbdt"]["ci95"], atol=1e-12, rtol=0)
            checked += 1
        rows = block["per_request"]
        for k, data in block["operating_points"].items():
            a = np.array([[r["values"][k][label] for label in LABELS] for r in rows])
            for unit, membership in (("model_request", np.arange(len(rows))),
                                     ("source_prompt", np.array([r["source"] for r in rows]))):
                n = len(np.unique(membership))
                counts = cluster_counts(n, result["n_boot"], result["seed"])
                size = np.bincount(membership, minlength=n)
                sums = np.array([np.bincount(membership, weights=a[:, j], minlength=n) for j in range(3)]).T
                boot = (counts @ sums) / (counts @ size)[:, None]
                for j, label in enumerate(LABELS[1:], start=1):
                    np.testing.assert_allclose(interval(boot[:, 0] - boot[:, j]), data[unit][label]["ci95"], atol=1e-12, rtol=0)
                    checked += 1
    return checked


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--out", type=Path)
    p.add_argument("--replay", type=Path, help="verify saved intervals from sufficient statistics; no traces needed")
    args = p.parse_args(argv)
    if args.replay:
        if args.out:
            p.error("--out and --replay are mutually exclusive")
        print(f"PASS: {replay_intervals(json.loads(args.replay.read_text()))} intervals replayed from stored statistics")
        return
    if args.out is None:
        p.error("provide --out or --replay")
    if args.out.exists() or args.n_boot < 1000:
        p.error("use a fresh output path and at least 1000 resamples")
    archived = json.loads((ROOT / "experiments/results/icdm_v2.json").read_text())
    models = {}
    for name, prov in archived["provenance"].items():
        print(f"Loading and verifying {name}", flush=True)
        models[name] = load_v2(ROOT / prov["trace"], True)
    d = pool(models)
    del models
    out = {"schema": "mentor-statistics-v1", "n_boot": args.n_boot, "seed": SEED, "confidence": 0.95,
           "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "runtime": {"python": platform.python_version(), **{package: importlib.metadata.version(package)
                       for package in ("numpy", "scipy", "scikit-learn", "lightgbm")}},
           "provenance": archived["provenance"],
           "estimand": "pooled AUC on the original 150K-row sample; request-macro recall on all held-out decisions",
           "source_resampling": "jointly draw every held-out model-request sharing a (dataset, source-id); weight by number of drawn model-requests"}
    for kind in ("request", "source"):
        print(f"Analyzing {kind} split", flush=True)
        out[kind] = analyze_split(d, kind, archived, args.n_boot, SEED)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(f"WROTE {args.out}", flush=True)
    for kind in ("request", "source"):
        print(kind, out[kind]["auc"]["source_prompt"]["two_view_minus_gbdt"])
        for k, value in out[kind]["operating_points"].items():
            print(k, value["source_prompt"])


if __name__ == "__main__":
    main()

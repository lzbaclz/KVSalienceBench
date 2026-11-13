#!/usr/bin/env python3
"""Exp#1 sensitivity: does the compact scorer's small AUC deficit depend on TRAIN_N?

`run_icdm_v2.py` fits every learned comparator on TRAIN_N = 120,000 rows sampled
from the 22.8M held-in rows, so all methods see the same budget. A three-parameter
logistic fit is saturated there, but a 150-tree LightGBM need not be, and an
under-trained tree comparator would flatter the compact model. This driver keeps
the corpus, the request split, the seed and the 150K held-out evaluation sample
exactly as the headline run has them and varies ONLY the training-row budget.

It reproduces the published 120K row as its own correctness gate, then refits at
larger budgets up to every held-in row. Nothing here rewrites an existing result
file; the output is a separate JSON.

Like every other trace-dependent driver, this one needs the raw version-2 JSONL
traces, which are not in the public artifact (their `.meta.json`/`.cohort.json`
manifests are). Request the traces to rerun it; the frozen output it produced,
`experiments/results/icdm_v2_train_size.json`, ships with the artifact and is
checked against the manuscript by tests/test_paper_numbers.py.

Usage (pinned env, CPU only):
  python experiments/analyze_train_size_sensitivity.py \
    --traces Llama-3.1-8B-Instruct=experiments/results/physical_kv/query-v2/query-v2.llama.bs32.jsonl,\
Qwen2.5-7B-Instruct=experiments/results/physical_kv/query-v2/query-v2.qwen.bs32.jsonl \
    --out experiments/results/icdm_v2_train_size.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from run_icdm_full import SEED, N_BOOT, roc_auc, subsample, clustered_paired_test  # noqa: E402
from run_icdm_v2 import HEADLINE_H, request_split_v2, decision_groups  # noqa: E402
from xqp.dm_metrics import precision_recall_at_k_grouped  # noqa: E402
from xqp.predictor import ClosedFormXQP  # noqa: E402

# The headline budget plus four progressively larger ones; the last entry is
# replaced by "every held-in row" at run time.
TRAIN_SIZES = (120_000, 480_000, 1_920_000, 7_680_000, None)
TEST_N = 150_000
MASK2 = np.array([1, 1, 0, 0], np.float32)  # within + cross, the compact views


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def load_v2_slim(path: Path, verify_hash: bool) -> dict:
    """Stream one version-2 trace into the columns this analysis needs.

    Row order is preserved exactly, because the held-out sample is drawn by row
    position and must match `run_icdm_v2.py` draw for draw.
    """
    meta = json.loads(Path(str(path) + ".meta.json").read_text())
    if meta.get("trace_version") != 2:
        raise ValueError(f"{path}: not a version-2 trace")
    if verify_hash and sha256_file(path) != meta["trace_sha256"]:
        raise ValueError(f"{path}: sha256 mismatch with its manifest")
    n = int(meta["rows"])
    rid = np.empty(n, np.int32)
    layer = np.empty(n, np.int16)
    step = np.empty(n, np.int16)
    F = np.empty((n, 4), np.float32)
    y = np.empty(n, np.int8)
    i = 0
    with path.open() as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("trace_version") != 2:
                raise ValueError("mixed trace versions")
            rid[i] = int(r["request_id"][1:])
            layer[i] = r["layer"]
            step[i] = r["step"]
            F[i, 0] = r["f_within"]; F[i, 1] = r["f_cross"]
            F[i, 2] = r["f_query"]; F[i, 3] = r["f_pos"]
            y[i] = r[f"y_{HEADLINE_H}"]
            i += 1
    if i != n:
        raise ValueError(f"{path}: read {i} rows, manifest says {n}")
    if not np.isfinite(F).all():
        raise ValueError("non-finite features")
    cohort = json.loads(Path(str(path) + ".cohort.json").read_text())
    source_of = {r["trace_id"]: (r["dataset"], r["id"]) for r in cohort["ids"]}
    n_req = int(rid.max()) + 1
    return dict(rid=rid, layer=layer, step=step, F=F, y=y, n_requests=n_req,
                source=[source_of[f"p{k}"] for k in range(n_req)],
                provenance=dict(trace=str(path), trace_sha256=meta["trace_sha256"],
                                rows=n, collector_sha256=meta["collector_sha256"]))


def pool(models: dict) -> dict:
    rid, layer, step, F, y, source, offset = [], [], [], [], [], [], 0
    for d in models.values():
        rid.append(d["rid"].astype(np.int64) + offset)
        layer.append(d["layer"]); step.append(d["step"])
        F.append(d["F"]); y.append(d["y"])
        source.extend(d["source"])
        offset += d["n_requests"]
    return dict(rid=np.concatenate(rid), layer=np.concatenate(layer),
                step=np.concatenate(step), F=np.concatenate(F),
                y=np.concatenate(y), source=source, n_requests=offset)


def fit_gbdt_balanced(Ftr, ytr, seed=SEED):
    """The exact headline LightGBM configuration (xqp.baselines.fit_gbdt)."""
    import lightgbm as lgb
    t0 = time.perf_counter()
    clf = lgb.LGBMClassifier(max_depth=3, n_estimators=150, class_weight="balanced",
                             verbosity=-1, random_state=seed).fit(Ftr, ytr.astype(int))
    return clf, time.perf_counter() - t0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traces", required=True, help="comma-separated NAME=path.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-verify-hash", action="store_true")
    ap.add_argument("--no-ci", action="store_true")
    a = ap.parse_args(argv)
    t_start = time.time()

    models = {}
    for spec in a.traces.split(","):
        name, _, path = spec.partition("=")
        print(f"[load] {name} <- {path}", flush=True)
        models[name] = load_v2_slim(Path(path), not a.no_verify_hash)
    d = pool(models)
    print(f"[pool] {d['F'].shape[0]:,} rows, {d['n_requests']} requests", flush=True)

    tr_idx, te_idx, _ = request_split_v2(d, "request", seed=SEED)
    te = subsample(te_idx, TEST_N, SEED)
    Fte, yte = d["F"][te], d["y"][te].astype(np.float32)
    te_req = d["rid"][te]
    # Per-decision metrics need complete (request, layer, step) groups, so they
    # use every held-out row, exactly as the headline driver does.
    Fte_all, yte_all = d["F"][te_idx], d["y"][te_idx].astype(np.float32)
    groups_all = decision_groups(d, te_idx)
    within_all = Fte_all[:, 0]
    g_within = precision_recall_at_k_grouped(yte_all, within_all, groups_all, 0.10)
    print(f"[ref] raw within-layer EMA per-decision recall {g_within['macro_recall']:.4f}", flush=True)

    sizes = [s if s is not None else int(len(tr_idx)) for s in TRAIN_SIZES]
    rows = []
    for n_train in sizes:
        t0 = time.time()
        tr = subsample(tr_idx, n_train, SEED)
        Ftr, ytr = d["F"][tr], d["y"][tr].astype(np.float32)
        two = ClosedFormXQP.from_fit(Ftr * MASK2, ytr)
        gbdt, gb_seconds = fit_gbdt_balanced(Ftr, ytr)
        s_two = np.asarray(two.score(Fte * MASK2), np.float64)
        s_gb = np.asarray(gbdt.predict_proba(Fte)[:, 1], np.float64)
        auc_two, auc_gb = roc_auc(yte, s_two), roc_auc(yte, s_gb)
        g_two = precision_recall_at_k_grouped(
            yte_all, np.asarray(two.score(Fte_all * MASK2), np.float64), groups_all, 0.10)
        g_gb = precision_recall_at_k_grouped(
            yte_all, np.asarray(gbdt.predict_proba(Fte_all)[:, 1], np.float64), groups_all, 0.10)
        row = dict(n_train=int(len(tr)), auc_two_view=auc_two, auc_gbdt=auc_gb,
                   auc_delta=auc_two - auc_gb,
                   dec_recall_two_view=g_two["macro_recall"], dec_recall_gbdt=g_gb["macro_recall"],
                   dec_precision_two_view=g_two["macro_precision"],
                   dec_precision_gbdt=g_gb["macro_precision"],
                   gbdt_fit_seconds=gb_seconds)
        if not a.no_ci:
            t = clustered_paired_test(roc_auc, yte, s_two, s_gb, te_req, n_boot=N_BOOT, seed=SEED)
            row.update(auc_delta_lo=t["lo"], auc_delta_hi=t["hi"], auc_delta_p=t["p_value"])
        rows.append(row)
        print(f"[fit] n_train={len(tr):>9,}  two-view {auc_two:.4f}  LightGBM {auc_gb:.4f}  "
              f"delta {auc_two - auc_gb:+.4f}  dec-recall {g_two['macro_recall']:.4f}/"
              f"{g_gb['macro_recall']:.4f}  ({time.time() - t0:.0f}s)", flush=True)

    out = dict(
        schema="kvsaliencebench/exp1-train-size-sensitivity/1",
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        config=dict(seed=SEED, headline_horizon=HEADLINE_H, TEST_N=TEST_N, N_BOOT=N_BOOT,
                    split="request", train_sizes=sizes,
                    gbdt="LGBMClassifier(max_depth=3, n_estimators=150, class_weight=balanced)"),
        provenance=[m["provenance"] for m in models.values()],
        n_rows=int(d["F"].shape[0]), n_requests=int(d["n_requests"]),
        n_train_rows_available=int(len(tr_idx)), n_test_rows_all=int(len(te_idx)),
        n_test_rows_sampled=int(len(te)), n_test_requests=int(len(np.unique(te_req))),
        raw_within_layer_decision_recall=g_within["macro_recall"],
        n_decision_groups=g_within["n_groups"],
        by_train_size=rows, seconds=time.time() - t_start)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"WROTE {a.out} in {out['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()

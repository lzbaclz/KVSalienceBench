#!/usr/bin/env python3
"""Exp#1 sensitivity: is the compact scorer's small AUC deficit a tree-capacity artifact?

The headline LightGBM comparator is ``max_depth=3, n_estimators=150``: few leaves and
a modest tree count. ``analyze_train_size_sensitivity.py`` shows that more TRAINING ROWS
do not change the comparison; this driver varies the other axis, MODEL CAPACITY
(depth/leaves, tree count, learning rate, class weighting), at the headline 120K rows
and at 1.92M rows, on the same split, seed and held-out rows.

For each configuration it records pooled AUC on the 150K evaluation sample (as in
Table II), pooled AUC and per-decision top-decile and 20% recall on every held-out row,
and the gap to the two-view logistic fit. It also records whether any configuration
reaches the raw within-layer EMA's per-decision recall, which is the quantity the
paper's reversal is about. Nothing here rewrites an existing result file.

Needs the raw version-2 traces (or the local cache written by
``analyze_decision_decomposition.py``); the frozen output
``experiments/results/icdm_v2_gbdt_capacity.json`` ships with the artifact.

Usage (pinned env, CPU only):
  python experiments/analyze_gbdt_capacity.py --traces NAME=path,... \
      --out experiments/results/icdm_v2_gbdt_capacity.json [--cache npz]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from run_icdm_full import SEED, subsample, roc_auc  # noqa: E402
from run_icdm_v2 import TRAIN_N, TEST_N, request_split_v2, decision_groups  # noqa: E402
from analyze_train_size_sensitivity import MASK2  # noqa: E402
from analyze_decision_decomposition import load_pooled  # noqa: E402
from xqp.decision_eval import DecisionData, Ranked, macro_recall  # noqa: E402
from xqp.predictor import ClosedFormXQP  # noqa: E402

TRAIN_SIZES = (TRAIN_N, 1_920_000)
N_JOBS = 32
CONFIGS = {   # name -> LGBMClassifier kwargs (class_weight added separately)
    "depth3_150_balanced (headline)": dict(max_depth=3, n_estimators=150, class_weight="balanced"),
    "depth3_150_unweighted": dict(max_depth=3, n_estimators=150),
    "depth3_600_lr0.1_balanced": dict(max_depth=3, n_estimators=600, learning_rate=0.1, class_weight="balanced"),
    "depth6_150_balanced": dict(max_depth=6, num_leaves=64, n_estimators=150, class_weight="balanced"),
    "depth8_300_lr0.05_balanced": dict(max_depth=8, num_leaves=255, n_estimators=300, learning_rate=0.05,
                                       class_weight="balanced"),
    "leaves63_500_lr0.05_balanced": dict(max_depth=-1, num_leaves=63, n_estimators=500, learning_rate=0.05,
                                         class_weight="balanced"),
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--no-verify-hash", action="store_true")
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite {a.out}")
    import lightgbm as lgb
    t_start = time.time()
    d = load_pooled(a.traces, a.cache, not a.no_verify_hash)
    tr_idx, te_idx, _ = request_split_v2(d, "request", seed=SEED)
    te = subsample(te_idx, TEST_N, SEED)
    Fte, yte = d["F"][te], d["y"][te].astype(np.float32)
    F_all, y_all = d["F"][te_idx], d["y"][te_idx]
    data = DecisionData.build(y_all, decision_groups(d, te_idx), d["rid"][te_idx])

    def evaluate(score_sample, score_all):
        r10 = macro_recall(data, Ranked(data, score_all).topk(0.10)["tp"])
        r20 = macro_recall(data, Ranked(data, score_all).topk(0.20)["tp"])
        from xqp.decision_eval import pooled_decomposition
        dec = pooled_decomposition(Ranked(data, score_all))
        return {"auc_150k_sample": roc_auc(yte, score_sample), "auc_all_rows": dec["auc_pooled"],
                "auc_same_decision": dec["auc_same"], "recall_0.10": r10, "recall_0.20": r20}

    raw = F_all[:, 0].astype(np.float64)
    ref_raw = evaluate(Fte[:, 0].astype(np.float64), raw)
    print(f"[raw within-layer EMA] {ref_raw}", flush=True)
    rows = []
    for n_train in TRAIN_SIZES:
        tr = subsample(tr_idx, n_train, SEED)
        Ftr, ytr = d["F"][tr], d["y"][tr].astype(np.float32)
        two = ClosedFormXQP.from_fit(Ftr * MASK2, ytr)
        two_eval = evaluate(np.asarray(two.score(Fte * MASK2), np.float64),
                            np.asarray(two.score(F_all * MASK2), np.float64))
        print(f"[two-view @ {len(tr):,}] {two_eval}", flush=True)
        rows.append(dict(n_train=int(len(tr)), config="two-view logistic", **two_eval))
        for name, kw in CONFIGS.items():
            t0 = time.time()
            clf = lgb.LGBMClassifier(verbosity=-1, random_state=SEED, n_jobs=N_JOBS, **kw).fit(Ftr, ytr.astype(int))
            fit_s = time.time() - t0
            ev = evaluate(clf.predict_proba(Fte)[:, 1], clf.predict_proba(F_all)[:, 1])
            row = dict(n_train=int(len(tr)), config=name, fit_seconds=fit_s, auc_gap_to_two_view=ev["auc_150k_sample"] - two_eval["auc_150k_sample"], **ev)
            rows.append(row)
            print(f"[gbdt {len(tr):>9,}] {name:<32} AUC {ev['auc_150k_sample']:.4f} (two-view {two_eval['auc_150k_sample']:.4f})  "
                  f"same {ev['auc_same_decision']:.4f}  rec10 {ev['recall_0.10']:.4f}  rec20 {ev['recall_0.20']:.4f}  "
                  f"fit {fit_s:.0f}s", flush=True)
    out = dict(schema="kvsaliencebench/exp1-gbdt-capacity/1",
               generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               config=dict(seed=SEED, split="request", train_sizes=list(TRAIN_SIZES), evaluation="150K sample for AUC; every held-out row for recall",
                           lightgbm_configs={k: {kk: vv for kk, vv in v.items()} for k, v in CONFIGS.items()}),
               provenance=d["provenance"], raw_within_layer_ema=ref_raw, rows=rows, seconds=time.time() - t_start)
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print(f"WROTE {a.out} in {out['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()

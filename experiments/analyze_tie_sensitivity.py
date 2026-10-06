#!/usr/bin/env python3
"""Exp#1: how much of each per-decision top-k value is decided by the tie rule.

Per-decision top-k uses a stable descending sort (``xqp.dm_metrics``), so candidates
with equal scores are resolved by ascending row index, and inside one decision the
collector writes rows in ascending block index. For a continuous score the rule is
inert. For a score with many equal values it is a positional prior: the block order,
not the signal, picks the blocks.

``analyze_decision_decomposition.py`` already recomputes the per-decision recall of the
three main scorers (within-layer EMA, two-view logistic, LightGBM) under exact random
tie-breaking. This driver does the same for EVERY row of Table II's upper block, at
10% and 20% retention, on every held-out row of both splits:

* the recall under the stable rule (the value the table and ``icdm_v2.json`` hold);
* its exact EXPECTATION when the tie group that straddles the k-th place is resolved
  uniformly at random (``xqp.decision_eval.Ranked.topk(ties='random')``);
* the share of decisions with such a straddling tie, the share of rows inside any
  within-decision tie, and the largest change of one decision's expected recall.

Scope of the check. It is an expectation of the decision-macro statistic, not a bound
on a single random draw and not a bound on each decision. It randomizes SCORE ties
only: the future-attention labels keep the collector's ``argpartition`` boundary with
exactly ``ceil(r n)`` positives, which the traces do not let us re-derive.

The fits, split, seed and held-out rows are those of ``run_icdm_v2.py``; the stable
values must reproduce the published table (the gate) and no existing record is
modified. Like every trace-dependent driver this needs the raw version-2 traces, or
the local parse cache that ``analyze_decision_decomposition.py --cache`` writes.

Usage (pinned env, CPU only, about one minute with a parse cache):
  python experiments/analyze_tie_sensitivity.py \
    --traces Llama-3.1-8B-Instruct=experiments/results/physical_kv/query-v2/query-v2.llama.bs32.jsonl,\
Qwen2.5-7B-Instruct=experiments/results/physical_kv/query-v2/query-v2.qwen.bs32.jsonl \
    --out experiments/results/icdm_v2_tie_sensitivity.json
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

from run_icdm_full import SEED, subsample  # noqa: E402
from run_icdm_v2 import TRAIN_N, request_split_v2, decision_groups  # noqa: E402
from analyze_train_size_sensitivity import MASK2, fit_gbdt_balanced  # noqa: E402
from analyze_decision_decomposition import load_pooled  # noqa: E402
from xqp.decision_eval import DecisionData, Ranked, macro_recall  # noqa: E402
from xqp.predictor import ClosedFormXQP  # noqa: E402

K_FRACS = (0.10, 0.20)
FROZEN = ROOT / "experiments/predictors/xqp_closed_2view_h4.json"
# Table II label -> method name in icdm_v2.json, in the table's row order.
ROWS = (("age proxy", "recency"), ("query proxy (cosine)", "Quest"),
        ("prev-layer indicator", "InfiniGen"), ("within-layer EMA", "H2O/attn-EMA"),
        ("LightGBM, balanced", "GBDT(LightGBM)"), ("two-view logistic, refit", "within+cross(2)"),
        ("two-view, archived weights", "archived two-view checkpoint (frozen, v2 features)"))
MAIN = ("within-layer EMA", "LightGBM, balanced", "two-view logistic, refit")
PRINT_TOL = 5e-4   # a change this large can alter a value printed to three decimals


def tie_row(data: DecisionData, scores) -> dict:
    rk = Ranked(data, scores)
    ok = data.npos_g > 0
    row = dict(tied_row_share=rk.tied_row_share())
    for kf in K_FRACS:
        tk = rk.topk(kf, ties="random")
        stable, expected = macro_recall(data, tk["tp"]), macro_recall(data, tk["tp_random_ties"])
        per_decision = (tk["tp_random_ties"] - tk["tp"])[ok] / data.npos_g[ok]
        row[f"recall_{kf:.2f}"] = stable
        row[f"recall_{kf:.2f}_random_ties"] = expected
        row[f"delta_{kf:.2f}"] = expected - stable
        row[f"boundary_tie_share_{kf:.2f}"] = float(tk["boundary_tie"].mean())
        row[f"max_abs_decision_delta_{kf:.2f}"] = float(np.abs(per_decision).max())
    return row


def analyze_split(d: dict, kind: str, frozen: ClosedFormXQP, published: dict) -> dict:
    tr_idx, te_idx, _ = request_split_v2(d, kind, seed=SEED)
    tr = subsample(tr_idx, TRAIN_N, SEED)
    Ftr, ytr = d["F"][tr], d["y"][tr].astype(np.float32)
    two = ClosedFormXQP.from_fit(Ftr * MASK2, ytr)
    gbdt, _ = fit_gbdt_balanced(Ftr, ytr)
    F, y = d["F"][te_idx], d["y"][te_idx]
    data = DecisionData.build(y, decision_groups(d, te_idx), d["rid"][te_idx])
    scores = {"age proxy": F[:, 3], "query proxy (cosine)": F[:, 2], "prev-layer indicator": F[:, 1],
              "within-layer EMA": F[:, 0], "LightGBM, balanced": gbdt.predict_proba(F)[:, 1],
              "two-view logistic, refit": two.score(F * MASK2),
              "two-view, archived weights": frozen.score(F * MASK2)}
    rows, gate = {}, {}
    for label, key in ROWS:
        row = tie_row(data, np.asarray(scores[label], np.float64))
        pub = published[key]["r_at_10_grouped_macro"]
        # the stable value must be the published one, or the comparison means nothing
        assert abs(row["recall_0.10"] - pub) < 5e-4, (kind, label, row["recall_0.10"], pub)
        gate[label] = dict(published_method=key, published_dec_recall=pub, recomputed_stable=row["recall_0.10"])
        rows[label] = row
        print(f"[{kind}] {label:<28} 10%: {row['recall_0.10']:.4f} -> {row['recall_0.10_random_ties']:.4f} "
              f"({row['delta_0.10']:+.2e}, boundary ties {row['boundary_tie_share_0.10']:.4f})  "
              f"20%: {row['recall_0.20']:.4f} -> {row['recall_0.20_random_ties']:.4f} ({row['delta_0.20']:+.2e})  "
              f"tied rows {row['tied_row_share']:.4f}", flush=True)
    main_worst = max(abs(rows[m][f"delta_{kf:.2f}"]) for m in MAIN for kf in K_FRACS)
    changed = [dict(row=label, k_frac=kf, stable=rows[label][f"recall_{kf:.2f}"],
                    random_ties=rows[label][f"recall_{kf:.2f}_random_ties"])
               for label, _ in ROWS for kf in K_FRACS if abs(rows[label][f"delta_{kf:.2f}"]) >= PRINT_TOL]
    return dict(split=kind, n_rows_heldout=int(len(te_idx)), n_decisions=int(data.n_groups),
                n_requests_heldout=int(data.n_req), rows=rows, gate=gate,
                main_scorers_max_abs_delta=main_worst, cells_changed_at_three_decimals=changed)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", required=True, help="comma-separated NAME=path.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=None, help="optional local npz cache of the parsed traces")
    ap.add_argument("--no-verify-hash", action="store_true")
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite {a.out}")
    t0 = time.time()
    d = load_pooled(a.traces, a.cache, not a.no_verify_hash)
    print(f"[pool] {d['F'].shape[0]:,} rows, {d['n_requests']} requests", flush=True)
    frozen = ClosedFormXQP.load(str(FROZEN))
    v2 = json.loads((ROOT / "experiments/results/icdm_v2.json").read_text())
    out = dict(
        schema="kvsaliencebench/exp1-tie-sensitivity/1",
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        config=dict(seed=SEED, headline_horizon="h4", train_rows=TRAIN_N, k_fracs=list(K_FRACS),
                    evaluation_rows="every held-out row", print_tolerance=PRINT_TOL,
                    stable_rule="descending score, ties by ascending row index (ascending block index inside a decision)",
                    random_rule="exact expectation when the tie group straddling the k-th place is resolved uniformly at random",
                    not_randomized="future-attention label boundaries (collector argpartition, exactly ceil(r n) positives)",
                    gbdt="LGBMClassifier(max_depth=3, n_estimators=150, class_weight=balanced)"),
        provenance=d["provenance"], splits={})
    for kind in ("request", "source"):
        out["splits"][kind] = analyze_split(
            d, kind, frozen, {r["method"]: r for r in v2[f"pooled_{kind}_split"]["table"]})
    out["seconds"] = time.time() - t0
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print(f"WROTE {a.out} in {out['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()

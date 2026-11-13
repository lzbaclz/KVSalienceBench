#!/usr/bin/env python3
"""Exp#1: what a per-decision budget sees that pooled AUC does not.

The raw within-layer EMA has lower pooled AUC than the two-view logistic fit and
LightGBM, yet it selects more of each decision's future top-decile blocks. Three
things change at once between those two comparisons (metric, aggregation, sample),
so the contrast alone does not say where the pooled gain comes from. This driver
measures it, on EVERY held-out row of the headline request split (no row sample):

1. Pooled AUC is split exactly into pairs whose positive and negative lie in the
   same cache decision and pairs from different decisions (pair-count weights).
2. The fitted cross-layer offset ``w_cross / w_within`` is scaled from 0 (the raw
   within-layer ordering) to 1 (the two-view ordering) with every other quantity
   fixed, tracing pooled AUC, same-/cross-decision AUC and per-decision recall.
3. The top-decile boundary is inspected where the two orderings disagree: which
   blocks the cross-layer term promotes, and how many of them are positive.
4. Boundary ties are quantified and the per-decision recall is recomputed under
   exact random tie-breaking.

Uncertainty is a bootstrap over source prompts (every held-out request of a drawn
prompt is repeated), computed from request-by-request concordant-pair counts so
that it is exact for the pooled statistic. The fits, split, seed and held-out rows
are those of ``run_icdm_v2.py``; the script reproduces Table II's pooled AUC and
per-decision recall as its own gate and does not modify any existing result.

``--split source`` repeats the whole analysis on the source-disjoint split (the same
32 source prompts held out for both models, the benchmark's default split), gated
against ``pooled_source_split`` of ``icdm_v2.json``; its record is
``experiments/results/icdm_v2_decomposition_source.json``. The request split of the
headline table holds out model-requests, so 50 of its 57 held-out source prompts
also occur in training through the other model.

Like every trace-dependent driver this needs the raw version-2 traces (not in the
public artifact; their manifests are). The frozen output
``experiments/results/icdm_v2_decomposition.json`` ships with the artifact and is
checked against the manuscript by tests/test_paper_numbers.py.

Usage (pinned env, CPU only, about ten minutes):
  python experiments/analyze_decision_decomposition.py \
    --traces Llama-3.1-8B-Instruct=experiments/results/physical_kv/query-v2/query-v2.llama.bs32.jsonl,\
Qwen2.5-7B-Instruct=experiments/results/physical_kv/query-v2/query-v2.qwen.bs32.jsonl \
    --out experiments/results/icdm_v2_decomposition.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from run_icdm_full import SEED, subsample, roc_auc  # noqa: E402
from run_icdm_v2 import TRAIN_N, TEST_N, MI_N, request_split_v2, decision_groups  # noqa: E402
from analyze_train_size_sensitivity import (MASK2, load_v2_slim, pool,  # noqa: E402
                                            fit_gbdt_balanced)
from xqp.decision_eval import (DecisionData, Ranked, RequestStats, bootstrap_statistics,  # noqa: E402
                               cluster_multiplicities, macro_recall, pair_matrix,
                               percentile_ci, pooled_decomposition)
from xqp.dm_metrics import precision_recall_at_k_grouped  # noqa: E402
from xqp.features import FEATURE_NAMES  # noqa: E402
from xqp.info_theory import redundancy_report  # noqa: E402
from xqp.predictor import ClosedFormXQP  # noqa: E402

K_FRACS = (0.10, 0.20)
LAMBDAS = (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.5, 2.0)
WITHIN, TWO, GBDT = "within-layer EMA", "two-view logistic (refit)", "LightGBM, balanced"
PUBLISHED_KEY = {WITHIN: "H2O/attn-EMA", TWO: "within+cross(2)", GBDT: "GBDT(LightGBM)"}


def load_pooled(specs: str, cache: Path | None, verify_hash: bool):
    """Parse the traces (slow), or reload a previous parse. The cache is a local
    scratch file, never part of the artifact."""
    names = [s.partition("=")[0] for s in specs.split(",")]
    if cache is not None and cache.exists():
        z = np.load(cache, allow_pickle=True)
        d = {k: z[k] for k in ("rid", "layer", "step", "F", "y")}
        d["source"] = [tuple(x) for x in z["source"].tolist()]
        d["n_requests"] = int(z["n_requests"])
        d["n_first"] = int(z["n_first"])
        d["provenance"] = json.loads(str(z["provenance"]))
        print(f"[cache] loaded {cache}", flush=True)
        return d
    models = {}
    for spec in specs.split(","):
        name, _, path = spec.partition("=")
        print(f"[load] {name} <- {path}", flush=True)
        models[name] = load_v2_slim(Path(path), verify_hash)
    d = pool(models)
    d["n_first"] = int(models[names[0]]["n_requests"])
    d["provenance"] = [m["provenance"] for m in models.values()]
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, rid=d["rid"], layer=d["layer"], step=d["step"], F=d["F"], y=d["y"],
                 source=np.array(d["source"], dtype=object), n_requests=d["n_requests"],
                 n_first=d["n_first"], provenance=json.dumps(d["provenance"]))
        print(f"[cache] wrote {cache}", flush=True)
    return d


def parts_row(data: DecisionData, scores) -> dict:
    """Point estimates for one scorer on every row of ``data``."""
    rk = Ranked(data, scores)
    dec = pooled_decomposition(rk)
    row = dict(auc_pooled=dec["auc_pooled"], auc_same=dec["auc_same"], auc_cross=dec["auc_cross"],
               auc_same_macro=dec["auc_same_macro"])
    for kf in K_FRACS:
        tk = rk.topk(kf, ties="random")
        row[f"recall_{kf:.2f}"] = macro_recall(data, tk["tp"])
        row[f"recall_{kf:.2f}_random_ties"] = macro_recall(data, tk["tp_random_ties"])
        row[f"boundary_tie_share_{kf:.2f}"] = float(tk["boundary_tie"].mean())
    row["tied_row_share"] = rk.tied_row_share()
    return row, rk, dec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", required=True, help="comma-separated NAME=path.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--split", choices=("request", "source"), default="request",
                    help="request: the headline split of Table II; source: the same source prompts held out for every model")
    ap.add_argument("--cache", type=Path, default=None, help="optional local npz cache of the parsed traces")
    ap.add_argument("--no-verify-hash", action="store_true")
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite {a.out}")
    t_start = time.time()

    d = load_pooled(a.traces, a.cache, not a.no_verify_hash)
    print(f"[pool] {d['F'].shape[0]:,} rows, {d['n_requests']} requests", flush=True)
    tr_idx, te_idx, test_reqs = request_split_v2(d, a.split, seed=SEED)
    tr = subsample(tr_idx, TRAIN_N, SEED)
    Ftr, ytr = d["F"][tr], d["y"][tr].astype(np.float32)
    two = ClosedFormXQP.from_fit(Ftr * MASK2, ytr)
    gbdt, _ = fit_gbdt_balanced(Ftr, ytr)
    w1, w2, b0 = float(two.weights[0]), float(two.weights[1]), float(two.bias)
    ratio = w2 / w1
    print(f"[two-view] w_within={w1:.4f} w_cross={w2:.4f} bias={b0:.4f}  offset w2/w1={ratio:.4f}", flush=True)

    F_all, y_all = d["F"][te_idx], d["y"][te_idx]
    rid_all = d["rid"][te_idx]
    groups_all = decision_groups(d, te_idx)
    data = DecisionData.build(y_all, groups_all, rid_all)
    scores = {WITHIN: F_all[:, 0].astype(np.float64),
              TWO: np.asarray(two.score(F_all * MASK2), np.float64),
              GBDT: np.asarray(gbdt.predict_proba(F_all)[:, 1], np.float64)}
    uniq_req = np.unique(rid_all)
    prompt_id = {}
    cluster_of_req = np.array([prompt_id.setdefault(d["source"][r], len(prompt_id)) for r in uniq_req])
    model_of_req = (uniq_req >= d["n_first"]).astype(int)
    train_sources = {d["source"][r] for r in np.unique(d["rid"][tr_idx])}
    shared_with_train = sum(s in train_sources for s in prompt_id)
    print(f"[data] held-out rows {len(te_idx):,}  decisions {data.n_groups:,}  requests {data.n_req} "
          f"(Llama {int((model_of_req == 0).sum())}, Qwen {int((model_of_req == 1).sum())})  "
          f"source prompts {len(prompt_id)}", flush=True)

    # ---- gate: Table II's published numbers must be reproduced -----------------
    published = {r["method"]: r for r in json.loads(
        (ROOT / "experiments/results/icdm_v2.json").read_text())[f"pooled_{a.split}_split"]["table"]}
    te = subsample(te_idx, TEST_N, SEED)
    yte = d["y"][te].astype(np.float32)
    gate = {}
    for name, key in PUBLISHED_KEY.items():
        s_sample = {WITHIN: d["F"][te][:, 0].astype(np.float64),
                    TWO: np.asarray(two.score(d["F"][te] * MASK2), np.float64),
                    GBDT: np.asarray(gbdt.predict_proba(d["F"][te])[:, 1], np.float64)}[name]
        auc_sample = roc_auc(yte, s_sample)
        ref = precision_recall_at_k_grouped(y_all.astype(np.float32), scores[name], groups_all, 0.10)
        gate[name] = dict(auc_150k_sample=auc_sample, published_auc=published[key]["auc"],
                          dec_recall_reference_impl=ref["macro_recall"],
                          published_dec_recall=published[key]["r_at_10_grouped_macro"])
        assert abs(auc_sample - published[key]["auc"]) < 5e-4, (name, auc_sample, published[key]["auc"])
        assert abs(ref["macro_recall"] - published[key]["r_at_10_grouped_macro"]) < 5e-4, name
        print(f"[gate] {name:<26} AUC(150K sample) {auc_sample:.4f} (published {published[key]['auc']:.4f})  "
              f"dec-recall {ref['macro_recall']:.4f} (published {published[key]['r_at_10_grouped_macro']:.4f})",
              flush=True)

    # ---- 1. decomposition, ties ---------------------------------------------------
    rows, ranked = {}, {}
    for name, s in scores.items():
        t0 = time.time()
        row, ranked[name], dec = parts_row(data, s)
        row["auc_150k_sample"] = gate[name]["auc_150k_sample"]
        # the vectorized top-k must equal the reference implementation exactly
        ref = precision_recall_at_k_grouped(y_all.astype(np.float32), s, groups_all, 0.10)
        assert abs(row["recall_0.10"] - ref["macro_recall"]) < 1e-12, name
        rows[name] = row
        pairs = dict(pairs_total=dec["pairs_total"], pairs_same=dec["pairs_same"], share_same=dec["share_same"])
        print(f"[decomp] {name:<26} pooled {row['auc_pooled']:.4f}  same {row['auc_same']:.4f}  "
              f"cross {row['auc_cross']:.4f}  rec10 {row['recall_0.10']:.4f}/{row['recall_0.10_random_ties']:.4f}(rand)  "
              f"rec20 {row['recall_0.20']:.4f}  ({time.time() - t0:.0f}s)", flush=True)
    # per-model view of the same quantities
    per_model = {}
    for mi, mname in enumerate(("Llama-3.1-8B-Instruct", "Qwen2.5-7B-Instruct")):
        sel = np.isin(rid_all, uniq_req[model_of_req == mi])
        dm = DecisionData.build(y_all[sel], groups_all[sel], rid_all[sel])
        per_model[mname] = {n: {k: v for k, v in parts_row(dm, s[sel])[0].items()
                                if k in ("auc_pooled", "auc_same", "auc_cross", "recall_0.10", "recall_0.20")}
                            for n, s in scores.items()}
        print(f"[per-model] {mname} done", flush=True)

    # ---- 2. bootstrap over source prompts ------------------------------------------
    M = cluster_multiplicities(cluster_of_req, a.n_boot, SEED)
    boot = {}
    for name, s in scores.items():
        t0 = time.time()
        C = pair_matrix(data, s)
        st = RequestStats.build(ranked[name], C, K_FRACS)
        assert abs(C.sum() - rows[name]["auc_pooled"] * data.npos_g.sum() * data.nneg_g.sum()) < 1e-6 * C.sum()
        boot[name] = bootstrap_statistics(M, st)
        print(f"[boot] {name:<26} pair matrix + {a.n_boot} draws ({time.time() - t0:.0f}s)", flush=True)
    metric_keys = ("auc_pooled", "auc_same", "auc_cross", "recall_0.1", "recall_0.2")
    paired = {}
    for lhs, rhs in ((TWO, WITHIN), (GBDT, WITHIN), (TWO, GBDT)):
        entry = {}
        for k in metric_keys:
            diff = boot[lhs][k] - boot[rhs][k]
            pt = {"auc_pooled": "auc_pooled", "auc_same": "auc_same", "auc_cross": "auc_cross",
                  "recall_0.1": "recall_0.10", "recall_0.2": "recall_0.20"}[k]
            entry[pt] = dict(delta=rows[lhs][pt] - rows[rhs][pt], ci95=percentile_ci(diff),
                             share_positive=float((diff > 0).mean()))
        paired[f"{lhs} minus {rhs}"] = entry
    marginal = {n: {({"recall_0.1": "recall_0.10", "recall_0.2": "recall_0.20"}.get(k, k)):
                    percentile_ci(v) for k, v in b.items()} for n, b in boot.items()}

    # ---- 3. cross-layer offset sweep ------------------------------------------------
    sweep = []
    for lam in LAMBDAS:
        t0 = time.time()
        s = F_all[:, 0].astype(np.float64) + lam * ratio * F_all[:, 1].astype(np.float64)
        row, _, _ = parts_row(data, s)
        row = dict(lam=lam, offset=lam * ratio, **{k: row[k] for k in ("auc_pooled", "auc_same", "auc_cross",
                   "recall_0.10", "recall_0.20")})
        sweep.append(row)
        print(f"[sweep] lambda={lam:<5} offset={lam * ratio:.4f}  pooled {row['auc_pooled']:.4f}  "
              f"same {row['auc_same']:.4f}  cross {row['auc_cross']:.4f}  rec10 {row['recall_0.10']:.4f}  "
              f"rec20 {row['recall_0.20']:.4f}  ({time.time() - t0:.0f}s)", flush=True)

    # ---- 4. what the cross-layer term swaps at the top-decile boundary ---------------
    sel0 = ranked[WITHIN].selected_mask(0.10)
    sel1 = ranked[TWO].selected_mask(0.10)
    yb = y_all == 1
    cl = F_all[:, 1] > 0.5
    swapped_in, swapped_out = sel1 & ~sel0, sel0 & ~sel1
    k_total = int(ranked[WITHIN].topk(0.10)["k"].sum())
    swaps = dict(
        top_fraction=0.10, k_total=k_total, n_decisions=int(data.n_groups),
        swapped_in=int(swapped_in.sum()), swapped_out=int(swapped_out.sum()),
        swapped_share_of_selected=float(swapped_in.sum() / k_total),
        positive_rate_swapped_in=float(yb[swapped_in].mean()),
        positive_rate_swapped_out=float(yb[swapped_out].mean()),
        cross_indicator_rate_swapped_in=float(cl[swapped_in].mean()),
        cross_indicator_rate_swapped_out=float(cl[swapped_out].mean()),
        net_positives=int(yb[swapped_in].sum() - yb[swapped_out].sum()),
        net_positives_per_decision=float((yb[swapped_in].sum() - yb[swapped_out].sum()) / data.n_groups),
        cross_indicator_rate_all_rows=float(cl.mean()),
        positive_rate_given_cross_1=float(yb[cl].mean()), positive_rate_given_cross_0=float(yb[~cl].mean()),
        positive_rate_all_rows=float(yb.mean()),
        raw_top10_with_cross_1=float(cl[sel0].mean()))
    print("[swaps]", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in swaps.items()}),
          flush=True)

    # ---- 5. relevance estimates from TRAINING rows only ------------------------------
    # run_icdm_v2.per_view samples its 400K MI rows from ALL pooled rows (held-out
    # requests included). Those estimates select nothing in version 2 (the two-view
    # subset was fixed from version-1 diagnostics), but the roles should not rest on
    # that argument: recompute them from the held-in rows and compare.
    idx_mi = subsample(tr_idx, MI_N, SEED)
    mi_train = {k: float(v) for k, v in redundancy_report(
        d["F"][idx_mi], d["y"][idx_mi], feature_names=list(FEATURE_NAMES))["per_feature_mi"].items()}
    mi_published = json.loads((ROOT / "experiments/results/icdm_v2.json").read_text())[
        "pooled_per_view"]["redundancy"]["per_feature_mi"]
    print("[mi] train-rows-only vs published (all pooled rows):",
          {k: (round(mi_train[k], 4), round(mi_published[k], 4)) for k in mi_train}, flush=True)

    sizes = data.group_size
    out = dict(
        schema="kvsaliencebench/exp1-decision-decomposition/1",
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        config=dict(seed=SEED, headline_horizon="h4", train_rows=TRAIN_N, split=a.split,
                    evaluation_rows="every held-out row", bootstrap="source prompts",
                    n_boot=a.n_boot, k_fracs=list(K_FRACS), lambdas=list(LAMBDAS),
                    gbdt="LGBMClassifier(max_depth=3, n_estimators=150, class_weight=balanced)"),
        provenance=d["provenance"],
        data=dict(n_rows_heldout=int(len(te_idx)), n_decisions=int(data.n_groups),
                  n_requests_heldout=int(data.n_req), n_source_prompts_heldout=int(len(prompt_id)),
                  n_heldout_source_prompts_also_in_training=int(shared_with_train),
                  n_positives=int(data.npos_g.sum()), positive_rate=float(data.npos_g.sum() / len(te_idx)),
                  decision_size=dict(min=int(sizes.min()), median=float(np.median(sizes)), max=int(sizes.max())),
                  heldout_requests_by_model=dict(llama=int((model_of_req == 0).sum()), qwen=int((model_of_req == 1).sum()))),
        pairs=dict(pairs_total=float(data.npos_g.sum() * data.nneg_g.sum()),
                   pairs_same_decision=float((data.npos_g * data.nneg_g).sum()),
                   share_same_decision=float((data.npos_g * data.nneg_g).sum() / (data.npos_g.sum() * data.nneg_g.sum()))),
        two_view=dict(w_within=w1, w_cross=w2, bias=b0, offset_ratio=ratio),
        gate=gate, scorers=rows, per_model=per_model, paired=paired, marginal_ci95=marginal,
        sweep=sweep, swaps_at_top_decile=swaps,
        relevance_mi=dict(sample_rows=int(MI_N), train_rows_only=mi_train, published_all_rows_sample=mi_published),
        seconds=time.time() - t_start)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    # convenience CSVs for readers who do not want to open the JSON
    stem = a.out.with_suffix("")
    with open(f"{stem}.decomposition.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        cols = ["auc_pooled", "auc_same", "auc_cross", "auc_same_macro", "recall_0.10", "recall_0.20",
                "recall_0.10_random_ties", "recall_0.20_random_ties", "boundary_tie_share_0.10",
                "boundary_tie_share_0.20", "tied_row_share"]
        w.writerow(["scorer"] + cols)
        for n, r in rows.items():
            w.writerow([n] + [f"{r[c]:.6f}" for c in cols])
    with open(f"{stem}.offset_sweep.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        cols = ["lam", "offset", "auc_pooled", "auc_same", "auc_cross", "recall_0.10", "recall_0.20"]
        w.writerow(cols)
        for r in sweep:
            w.writerow([f"{r[c]:.6f}" for c in cols])
    print(f"WROTE {a.out} in {out['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()

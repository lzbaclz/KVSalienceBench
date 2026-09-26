#!/usr/bin/env python3
"""Offline prediction analysis on VERSION-2 traces (PC items P0-3, P0-4, P1-4, P1-5, P1-7, P1-8).

The archived Table I was computed from unpinned version-1 traces whose raw rows
no longer exist on disk. This driver re-establishes the offline evidence on
checksum-verified version-2 traces (phase-aligned queries, prefill-predicted
first token, genuine future horizons, pinned collector), using the same
protocol as ``run_icdm_full.py`` where the protocol was sound and correcting
it where the audit found problems:

* tie-aware AUPRC (``xqp.dm_metrics.average_precision``);
* pooled top-decile P/R reported *as pooled*, next to per-decision-group
  (request, layer, step) macro/micro P/R at the label's own ``ceil(0.1 n)``;
* two splits: (a) request-level (25% of each model's requests held out) and
  (b) source-prompt-level, where the SAME LongBench rows are held out for every
  model, so text seen in one model's training never appears in another's test;
* calibration reliability for the two-view fit, the class-balanced GBDT, an
  unweighted GBDT and an isotonic-recalibrated GBDT, all on the same rows;
* the frozen archived two-view checkpoint scored on the v2 features, so the
  effect of re-fitting is visible.

Every trace's ``.meta.json`` (trace version 2, sha256, collector hash) and
``.cohort.json`` (source dataset/id per request) is required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from run_icdm_full import (HORIZONS, HEADLINE_H, N_BOOT, SEED, roc_auc, subsample,  # noqa: E402
                           clustered_bootstrap_ci, clustered_paired_test, fit_all_methods)
from xqp.dm_metrics import (average_precision, precision_at_k, recall_at_k,  # noqa: E402
                            precision_recall_at_k_grouped, expected_calibration_error,
                            brier_score, reliability_curve)
from xqp.features import FEATURE_NAMES  # noqa: E402
from xqp.info_theory import redundancy_report, mutual_information  # noqa: E402
from xqp.predictor import ClosedFormXQP  # noqa: E402

TRAIN_N = 120_000
TEST_N = 150_000
MI_N = 400_000
EXTRA = ("f_query_dotmax",)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_v2(path: Path, verify_hash: bool) -> dict:
    meta = json.loads(Path(str(path) + ".meta.json").read_text())
    cohort = json.loads(Path(str(path) + ".cohort.json").read_text())
    if meta.get("trace_version") != 2:
        raise ValueError(f"{path}: not a version-2 trace")
    if verify_hash and sha256_file(path) != meta["trace_sha256"]:
        raise ValueError(f"{path}: sha256 mismatch with its manifest")
    source_of = {r["trace_id"]: (r["dataset"], r["id"]) for r in cohort["ids"]}
    rid, layer, step, blk = [], [], [], []
    F, extra = [], []
    ys = {h: [] for h in HORIZONS}
    with path.open() as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("trace_version") != 2:
                raise ValueError("mixed trace versions")
            rid.append(int(r["request_id"][1:])); layer.append(r["layer"]); step.append(r["step"]); blk.append(r["block_idx"])
            F.append((r["f_within"], r["f_cross"], r["f_query"], r["f_pos"]))
            extra.append(tuple(r[k] for k in EXTRA))
            for h in HORIZONS:
                ys[h].append(r[f"y_{h}"])
    rid = np.asarray(rid, np.int32)
    F = np.asarray(F, np.float32)
    if not np.isfinite(F).all():
        raise ValueError("non-finite features; the v2 collector should have rejected them")
    n_req = int(rid.max()) + 1
    if n_req != len(meta["requests"]) or n_req != len(source_of):
        raise ValueError("request count disagrees with manifest/cohort")
    return dict(rid=rid, layer=np.asarray(layer, np.int16), step=np.asarray(step, np.int16),
                blk=np.asarray(blk, np.int32), F=F, extra=np.asarray(extra, np.float32),
                y={h: np.asarray(ys[h], np.int8) for h in HORIZONS}, n_requests=n_req,
                source=[source_of[f"p{i}"] for i in range(n_req)],
                provenance=dict(trace=str(path), trace_sha256=meta["trace_sha256"], rows=meta["rows"],
                                collector_sha256=meta["collector_sha256"], transformers=meta["transformers"],
                                torch=meta["torch"], feature_steps=meta["feature_steps"],
                                lookahead_steps=meta["lookahead_steps"], block_size=meta["block_size"],
                                r_label=meta["r_label"], ema_decay=meta["ema_decay"],
                                model_type=meta["model_config"].get("model_type"),
                                cohort_data_sha256=cohort["data_sha256"],
                                model_weight_sha256=cohort["model_weight_sha256"]))


def pool(models: dict) -> dict:
    out = {k: [] for k in ("rid", "layer", "step", "blk", "F", "extra", "mid", "source")}
    ys = {h: [] for h in HORIZONS}
    offset = 0
    for mi, (name, d) in enumerate(models.items()):
        out["rid"].append(d["rid"].astype(np.int64) + offset)
        out["mid"].append(np.full(d["rid"].shape[0], mi, np.int8))
        for k in ("layer", "step", "blk", "F", "extra"):
            out[k].append(d[k])
        for h in HORIZONS:
            ys[h].append(d["y"][h])
        out["source"].extend(d["source"])
        offset += d["n_requests"]
    res = {k: np.concatenate(v) for k, v in out.items() if k != "source"}
    res["y"] = {h: np.concatenate(ys[h]) for h in HORIZONS}
    res["n_requests"] = offset
    res["source"] = out["source"]
    res["model_names"] = list(models)
    return res


def request_split_v2(d: dict, kind: str, frac: float = 0.25, seed: int = SEED):
    """kind='request': 25% of each model's requests. kind='source': the same
    source (dataset,id) rows held out in every model."""
    rng = np.random.default_rng(seed)
    sources = sorted(set(d["source"]))
    if kind == "request":
        uniq = np.unique(d["rid"])
        n_test = max(1, int(frac * len(uniq)))
        test_reqs = set(rng.permutation(uniq)[:n_test].tolist())
    elif kind == "source":
        n_test = max(1, int(frac * len(sources)))
        held = set(tuple(x) for x in np.array(sources, dtype=object)[rng.permutation(len(sources))[:n_test]])
        test_reqs = {r for r, s in enumerate(d["source"]) if tuple(s) in held}
    else:
        raise ValueError(kind)
    is_test = np.isin(d["rid"], list(test_reqs))
    return np.where(~is_test)[0], np.where(is_test)[0], sorted(test_reqs)


def decision_groups(d: dict, idx: np.ndarray) -> np.ndarray:
    return (d["rid"][idx].astype(np.int64) * 1000 + d["layer"][idx].astype(np.int64)) * 1000 + d["step"][idx].astype(np.int64)


def fit_calibration_variants(Ftr, ytr, rid_tr, seed=SEED):
    """Balanced GBDT (archived config), unweighted GBDT, and isotonic on held-out calibration requests."""
    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression
    rng = np.random.default_rng(seed)
    uniq = np.unique(rid_tr)
    cal_reqs = set(rng.permutation(uniq)[: max(1, len(uniq) // 4)].tolist())
    cal = np.isin(rid_tr, list(cal_reqs))
    fit_rows = ~cal
    bal = lgb.LGBMClassifier(max_depth=3, n_estimators=150, class_weight="balanced", verbosity=-1,
                             random_state=seed).fit(Ftr[fit_rows], ytr[fit_rows].astype(int))
    unw = lgb.LGBMClassifier(max_depth=3, n_estimators=150, verbosity=-1,
                             random_state=seed).fit(Ftr[fit_rows], ytr[fit_rows].astype(int))
    iso = IsotonicRegression(out_of_bounds="clip").fit(bal.predict_proba(Ftr[cal])[:, 1], ytr[cal])
    return {"GBDT balanced": lambda F: bal.predict_proba(F)[:, 1],
            "GBDT unweighted": lambda F: unw.predict_proba(F)[:, 1],
            "GBDT balanced+isotonic": lambda F: iso.predict(bal.predict_proba(F)[:, 1])}, len(cal_reqs)


def evaluate_split(d: dict, kind: str, frozen: ClosedFormXQP, seed=SEED, with_ci=True) -> dict:
    tr_idx, te_idx, test_reqs = request_split_v2(d, kind, seed=seed)
    tr = subsample(tr_idx, TRAIN_N, seed)
    te = subsample(te_idx, TEST_N, seed)
    Ftr, ytr = d["F"][tr], d["y"][HEADLINE_H][tr].astype(np.float32)
    Fte, yte = d["F"][te], d["y"][HEADLINE_H][te].astype(np.float32)
    te_req = d["rid"][te]
    # Per-decision metrics need COMPLETE (request, layer, step) groups: a row
    # subsample leaves ~3 rows per group and makes top-ceil(0.1 n) meaningless.
    # They are therefore computed on every held-out row, not on the 150K sample.
    Fte_all, yte_all = d["F"][te_idx], d["y"][HEADLINE_H][te_idx].astype(np.float32)
    groups_all = decision_groups(d, te_idx)
    extra_all = d["extra"][te_idx]
    methods, cf, pw, mlp = fit_all_methods(Ftr, ytr, seed=seed)
    mask2 = np.array([1, 1, 0, 0], np.float32)
    methods.append(("archived two-view checkpoint (frozen, v2 features)",
                    lambda F: frozen.score(F * mask2), 3))
    # two views + the phase-aligned dot-max probe, standardized on train rows only
    Xtr = np.concatenate([Ftr[:, :2], d["extra"][tr]], axis=1)
    Xte = np.concatenate([Fte[:, :2], d["extra"][te]], axis=1)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(max_iter=1000, random_state=seed).fit((Xtr - mu) / sd, ytr)
    def dotmax_scorer(F, X=None):
        # F is either the 150K sample or every held-out row; pick the matching probe columns
        X = Xte if F.shape[0] == Fte.shape[0] and F is Fte else np.concatenate([F[:, :2], extra_all], axis=1)
        return lr.predict_proba((X - mu) / sd)[:, 1]
    methods.append(("within+cross+dotmax logistic", dotmax_scorer, 4))
    ref = cf.score(Fte)
    rows = []
    for name, scorer, params in methods:
        s = np.asarray(scorer(Fte), np.float64)
        s_all = np.asarray(scorer(Fte_all), np.float64)
        g = precision_recall_at_k_grouped(yte_all, s_all, groups_all, 0.10)
        row = dict(method=name, params=int(params), auc=roc_auc(yte, s), auprc=average_precision(yte, s),
                   p_at_10_pooled=precision_at_k(yte, s, 0.10), r_at_10_pooled=recall_at_k(yte, s, 0.10),
                   p_at_10_grouped_macro=g["macro_precision"], r_at_10_grouped_macro=g["macro_recall"],
                   p_at_10_grouped_micro=g["micro_precision"], r_at_10_grouped_micro=g["micro_recall"],
                   n_decision_groups=g["n_groups"], n_rows_grouped=int(yte_all.shape[0]),
                   ece=expected_calibration_error(yte, s), brier=brier_score(yte, s))
        if with_ci:
            aci = clustered_bootstrap_ci(roc_auc, yte, s, te_req, n_boot=N_BOOT, seed=seed)
            api = clustered_bootstrap_ci(average_precision, yte, s, te_req, n_boot=N_BOOT, seed=seed)
            row.update(auc_lo=aci["lo"], auc_hi=aci["hi"], auprc_lo=api["lo"], auprc_hi=api["hi"],
                       n_test_requests=aci["n_groups"])
            if name != "XQP-closed":
                t = clustered_paired_test(roc_auc, yte, ref, s, te_req, n_boot=N_BOOT, seed=seed)
                row.update(auc_delta_vs_closed=t["delta"], p_vs_closed=t["p_value"])
        rows.append(row)
    two = next(r for r in rows if r["method"] == "within+cross(2)")
    gb = next(r for r in rows if r["method"].startswith("GBDT"))
    two_s = np.asarray(methods[1][1](Fte), np.float64)
    gb_s = np.asarray(next(m for m in methods if m[0].startswith("GBDT"))[1](Fte), np.float64)
    paired = clustered_paired_test(roc_auc, yte, two_s, gb_s, te_req, n_boot=N_BOOT, seed=seed) if with_ci else None
    # calibration variants on identical rows
    variants, n_cal = fit_calibration_variants(Ftr, ytr, d["rid"][tr], seed)
    calibration = {"within+cross(2)": dict(ece=two["ece"], brier=two["brier"],
                                            reliability=reliability_curve(yte, two_s, 10))}
    for name, fn in variants.items():
        s = np.asarray(fn(Fte), np.float64)
        calibration[name] = dict(ece=expected_calibration_error(yte, s), brier=brier_score(yte, s),
                                 auc=roc_auc(yte, s), reliability=reliability_curve(yte, s, 10))
    # leave-one-view-out for the four-view fit (train rows only refit)
    full = roc_auc(yte, cf.score(Fte))
    ablation = {"full_auc": full, "drop": {}}
    for i in range(4):
        m = np.ones(4, np.float32); m[i] = 0
        auc = roc_auc(yte, ClosedFormXQP.from_fit(Ftr * m, ytr).score(Fte * m))
        ablation["drop"][FEATURE_NAMES[i]] = dict(auc_without=auc, auc_drop=full - auc)
    return dict(split=kind, n_train=int(len(tr)), n_test=int(len(te)), n_test_rows_all=int(len(te_idx)),
                n_test_requests=int(len(np.unique(te_req))),
                held_out_requests=[int(x) for x in test_reqs], pos_rate_test=float(yte.mean()),
                table=rows, two_view_minus_gbdt_auc=paired, calibration=calibration,
                calibration_requests_held_for_isotonic=int(n_cal), view_ablation=ablation)


def per_view(d: dict, seed=SEED) -> dict:
    idx = subsample(np.arange(d["F"].shape[0]), MI_N, seed)
    F, out = d["F"][idx], {}
    for h in HORIZONS:
        y = d["y"][h][idx].astype(np.float32)
        out[h] = {FEATURE_NAMES[i]: dict(auc=roc_auc(y, F[:, i]), auprc=average_precision(y, F[:, i]),
                                          relevance_mi=mutual_information(F[:, i], y)) for i in range(4)}
        out[h]["f_query_dotmax"] = dict(auc=roc_auc(y, d["extra"][idx][:, 0]),
                                        auprc=average_precision(y, d["extra"][idx][:, 0]),
                                        relevance_mi=mutual_information(d["extra"][idx][:, 0], y))
    return dict(sample_rows=int(len(idx)), note="MI sample of rows, NOT the held-out test set", by_horizon=out,
                redundancy=redundancy_report(F, d["y"][HEADLINE_H][idx], feature_names=list(FEATURE_NAMES)))


def summary(d: dict) -> dict:
    return dict(n_rows=int(d["F"].shape[0]), n_requests=int(d["n_requests"]),
                n_layers=int(d["layer"].max() + 1), max_step=int(d["step"].max()),
                pos_rate={h: float(d["y"][h].mean()) for h in HORIZONS},
                feature_mean={FEATURE_NAMES[i]: float(d["F"][:, i].mean()) for i in range(4)})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traces", required=True, help="comma-separated NAME=path.jsonl")
    ap.add_argument("--frozen", default=str(ROOT / "experiments/predictors/xqp_closed_2view_h4.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-verify-hash", action="store_true")
    ap.add_argument("--no-ci", action="store_true")
    a = ap.parse_args(argv)
    if Path(a.out).exists():
        raise SystemExit(f"refusing to overwrite {a.out}")
    t0 = time.time()
    models = {}
    for spec in a.traces.split(","):
        name, _, path = spec.partition("=")
        print(f"[load] {name} <- {path}", flush=True)
        models[name] = load_v2(Path(path), not a.no_verify_hash)
        print(f"       {models[name]['F'].shape[0]:,} rows, {models[name]['n_requests']} requests", flush=True)
    frozen = ClosedFormXQP.load(a.frozen)
    pooled = pool(models)
    out = dict(schema="icdm-v2-analysis-v1", generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               config=dict(TRAIN_N=TRAIN_N, TEST_N=TEST_N, MI_N=MI_N, N_BOOT=N_BOOT, seed=SEED,
                           headline_horizon=HEADLINE_H, frozen_checkpoint_sha256=sha256_file(Path(a.frozen))),
               provenance={n: d["provenance"] for n, d in models.items()},
               per_model_summary={n: summary(d) for n, d in models.items()},
               pooled_summary=summary(pooled),
               shared_source_prompts=len(set(pooled["source"])), n_model_requests=int(pooled["n_requests"]))
    for kind in ("request", "source"):
        print(f"[pooled] split={kind} ...", flush=True)
        out[f"pooled_{kind}_split"] = evaluate_split(pooled, kind, frozen, with_ci=not a.no_ci)
    out["pooled_per_view"] = per_view(pooled)
    out["per_model"] = {}
    for n, d in models.items():
        print(f"[per-model] {n} ...", flush=True)
        out["per_model"][n] = dict(request_split=evaluate_split(d, "request", frozen, with_ci=False),
                                   per_view=per_view(d))
    out["seconds"] = round(time.time() - t0, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(f"WROTE {a.out} ({out['seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

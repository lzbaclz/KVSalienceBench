#!/usr/bin/env python3
"""Per-request paired intervals for the per-decision operating point (PC H1, H2).

Table II reports that the raw within-layer EMA selects more of each decision's
future top-decile blocks than the two-view fit or LightGBM, while losing on
pooled AUC and pooled top-k. That table gives point estimates macro-averaged
over every cache decision, which is not an inferential unit: decisions inside a
request are dependent. This script recomputes the same quantity per held-out
*request* and pairs the scorers within a request, so the reversal comes with an
interval, and repeats it at the closed loop's own 20% budget instead of only at
the label's 10% operating point.

It reuses the loader, split and fits of ``run_icdm_v2.py`` with the same seed,
so the held-out requests are the ones behind Table II. Nothing is refit on test
rows and ``icdm_v2.json`` is not modified.
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

from run_icdm_full import HEADLINE_H, SEED, subsample, fit_all_methods  # noqa: E402
from run_icdm_v2 import TRAIN_N, load_v2, pool, request_split_v2, decision_groups  # noqa: E402
from xqp.dm_metrics import precision_recall_at_k_grouped  # noqa: E402

# The three rows whose ordering flips between pooled and per-decision metrics.
WITHIN = "within-layer EMA"
TWO_VIEW = "two-view logistic (refit)"
GBDT = "LightGBM, balanced"


def paired_bootstrap(diff: np.ndarray, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    return {"mean": float(diff.mean()),
            "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
            "n_requests": int(len(diff)),
            "n_negative": int((diff < 0).sum()),
            "n_positive": int((diff > 0).sum())}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--traces", nargs="+", required=True,
                   help="NAME=PATH for each version-2 trace, as run_icdm_v2.py takes them")
    p.add_argument("--k-fracs", type=float, nargs="+", default=[0.10, 0.20])
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--verify-hash", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; choose a fresh path")

    started = time.time()
    models = {}
    for spec in args.traces:
        name, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"expected NAME=PATH, got {spec!r}")
        models[name] = load_v2(Path(path), args.verify_hash)
    d = pool(models)

    tr_idx, te_idx, test_reqs = request_split_v2(d, "request", seed=args.seed)
    tr = subsample(tr_idx, TRAIN_N, args.seed)
    Ftr, ytr = d["F"][tr], d["y"][HEADLINE_H][tr].astype(np.float32)
    methods, _cf, _pw, _mlp = fit_all_methods(Ftr, ytr, seed=args.seed)
    by_name = {name: scorer for name, scorer, _ in methods}
    # Exact names, so a silent fallback (e.g. sklearn GBDT when LightGBM is
    # missing) fails here instead of quietly changing what is compared.
    wanted = {}
    for label, key in [(WITHIN, "H2O/attn-EMA"), (TWO_VIEW, "within+cross(2)"),
                       (GBDT, "GBDT(LightGBM)")]:
        if key not in by_name:
            raise SystemExit(f"scorer {key!r} not among {sorted(by_name)}")
        wanted[label] = by_name[key]

    F_all = d["F"][te_idx]
    y_all = d["y"][HEADLINE_H][te_idx].astype(np.float32)
    groups_all = decision_groups(d, te_idx)
    req_all = d["rid"][te_idx]
    scores = {label: np.asarray(fn(F_all), np.float64) for label, fn in wanted.items()}

    out = {"schema": "decision-operating-point-v1",
           "estimand": "per-request macro top-k precision/recall over cache decisions, "
                       "paired across scorers within a request",
           "seed": args.seed, "n_boot": args.n_boot,
           "held_out_requests": len(test_reqs), "held_out_rows": int(len(te_idx)),
           "scorers": sorted(wanted), "k_fracs": args.k_fracs,
           "provenance": {n: m["provenance"] for n, m in models.items()},
           "operating_points": {}}

    for k_frac in args.k_fracs:
        per_request = {label: [] for label in wanted}
        for req in sorted(set(req_all.tolist())):
            sel = req_all == req
            for label, s in scores.items():
                m = precision_recall_at_k_grouped(y_all[sel], s[sel], groups_all[sel], k_frac)
                per_request[label].append((m["macro_precision"], m["macro_recall"], m["n_groups"]))
        block = {"k_frac": k_frac, "per_scorer": {}, "paired": {}}
        arr = {label: np.array([v[:2] for v in rows], float) for label, rows in per_request.items()}
        groups_per_request = [v[2] for v in per_request[WITHIN]]
        block["decision_groups_total"] = int(sum(groups_per_request))
        for label, a in arr.items():
            block["per_scorer"][label] = {
                "macro_precision_mean": float(a[:, 0].mean()),
                "macro_recall_mean": float(a[:, 1].mean())}
        for other in (TWO_VIEW, GBDT):
            for j, metric in enumerate(("precision", "recall")):
                block["paired"][f"{WITHIN} minus {other} ({metric})"] = paired_bootstrap(
                    arr[WITHIN][:, j] - arr[other][:, j], args.n_boot, args.seed)
        out["operating_points"][f"{k_frac:.2f}"] = block

    out["wall_s"] = time.time() - started
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print("WROTE", args.out)
    for k, block in out["operating_points"].items():
        print(f"  k={k}: " + ", ".join(
            f"{n}={v['macro_recall_mean']:.3f}" for n, v in block["per_scorer"].items()))
        for name, v in block["paired"].items():
            if "recall" in name:
                print(f"    {name}: {v['mean']:+.4f} {v['ci95']} "
                      f"({v['n_positive']}/{v['n_requests']} positive)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

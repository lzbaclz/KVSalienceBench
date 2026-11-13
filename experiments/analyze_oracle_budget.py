#!/usr/bin/env python3
"""How much served-oracle miss is forced by the budget rather than by the selector.

The served oracle uses a minimum-32/top-decile rule clipped to available blocks;
its exact block-boundary convention is documented by audit_oracle_count_mapping.
The diagnostic uses recorded oracle cardinality, not a reconstruction from n.
When the oracle set is larger than the budget, no selector can reach recall 1,
so part of the reported miss is arithmetic, not a ranking failure:

    recall <= kept / |oracle|   =>   miss >= 1 - kept / |oracle|

The stored counts are rounded layer means. This script computes a logged-count
diagnostic, an exact mean layerwise capacity floor only if B and K cardinalities
are common across layers. The unpinned archive lacks layer arrays and cannot
independently establish that premise. See audit_oracle_count_mapping.py for the
field mapping and a CPU invariant check of the pinned path. Historical JSON
keys such as forced_floor_mean are retained for compatibility, not certification
of the missing layer-level evidence. No model execution is performed.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np


def per_request(path: Path) -> list[dict]:
    obj = json.loads(path.read_text())
    out = []
    for r in obj["results"]:
        kept = np.asarray(r["per_step_block_count"], float)
        oracle = np.asarray(r["per_step_B_t_oracle"], float)
        miss = np.asarray(r["per_step_eps_measured"], float)
        if not (len(kept) == len(oracle) == len(miss)) or len(kept) == 0:
            raise ValueError(f"{path}: ragged or empty per-step records")
        if not all(np.isfinite(v).all() for v in (kept, oracle, miss)) \
                or np.any(oracle <= 0) or np.any(kept < 0) \
                or np.any((miss < 0) | (miss > 1)):
            raise ValueError(f"{path}: invalid counts or miss rates")
        floor = np.clip(1.0 - kept / oracle, 0.0, 1.0)
        if np.any(miss < floor - 1e-9):
            raise ValueError(f"{path}: miss below logged-count diagnostic; inspect layer aggregation")
        out.append({
            "id": r["id"],
            "steps": int(len(kept)),
            "kept_mean": float(kept.mean()),
            "oracle_mean": float(oracle.mean()),
            "budget_short_frac": float((kept < oracle).mean()),
            "miss_mean": float(miss.mean()),
            "forced_floor_mean": float(floor.mean()),
            "residual_mean": float((miss - floor).mean()),
            "kept_min": int(kept.min()), "kept_max": int(kept.max()),
            "oracle_min": int(oracle.min()), "oracle_max": int(oracle.max()),
            "positive_floor_steps": int((floor > 0).sum()),
            "positive_floor_outside_k32": int(((floor > 0) & (oracle != 32)).sum()),
        })
    return out


def summarize(rows: list[dict]) -> dict:
    def stat(key):
        v = np.array([r[key] for r in rows], float)
        return {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max())}
    return {
        "n_requests": len(rows),
        "kept_blocks": stat("kept_mean"),
        "oracle_set_size": stat("oracle_mean"),
        "steps_with_budget_below_oracle": float(np.mean([r["budget_short_frac"] for r in rows])),
        "measured_miss": stat("miss_mean"),
        "forced_floor": stat("forced_floor_mean"),
        "selector_residual": stat("residual_mean"),
    }


def paired_summary(a: list[dict], b: list[dict], n_boot=10000, seed=1234) -> dict:
    """Align by (dataset, request), never infer paired equality from grand means."""
    def index(rows):
        result = {(r["dataset"], r["id"]): r for r in rows}
        if len(result) != len(rows):
            raise ValueError("duplicate dataset/request ids")
        return result
    aa, bb = index(a), index(b)
    if not aa or aa.keys() != bb.keys():
        raise ValueError("policies must cover exactly the same dataset/request pairs")
    keys = sorted(aa)
    floor_delta = np.array([bb[k]["forced_floor_mean"] - aa[k]["forced_floor_mean"] for k in keys])
    miss_delta = np.array([bb[k]["miss_mean"] - aa[k]["miss_mean"] for k in keys])
    residual_delta = np.array([bb[k]["residual_mean"] - aa[k]["residual_mean"] for k in keys])
    draws = np.random.default_rng(seed).integers(len(keys), size=(n_boot, len(keys)))
    def stat(values):
        return {"mean": float(values.mean()), "ci95": np.quantile(values[draws].mean(1), [.025, .975]).tolist()}
    return {"n_pairs": len(keys), "n_boot": n_boot, "seed": seed,
            "pairing": "dataset and request id; steps averaged within each policy/request first",
            "floors_equal_per_request": bool(np.all(np.abs(floor_delta) < 1e-12)),
            "max_abs_request_floor_delta": float(np.abs(floor_delta).max()),
            "floor_delta": stat(floor_delta), "miss_delta": stat(miss_delta),
            "residual_delta": stat(residual_delta),
            "pairs": [{"dataset": k[0], "id": k[1], "floor_delta": float(floor_delta[i]),
                       "miss_delta": float(miss_delta[i]), "residual_delta": float(residual_delta[i])}
                      for i, k in enumerate(keys)]}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="experiments/results/e2e_confirm")
    p.add_argument("--policies", nargs="+", default=["h2o", "xqp"])
    p.add_argument("--block-size", type=int, default=32)
    p.add_argument("--out", type=Path)
    p.add_argument("--quiet", action="store_true", help="write JSON without printing the full paired records")
    args = p.parse_args(argv)

    root = Path(args.root)
    datasets = sorted({os.path.basename(f).rsplit("_", 1)[0]
                       for f in glob.glob(str(root / "*_h2o.json"))})
    if not datasets:
        raise SystemExit(f"no closed-loop cells under {root}")

    result = {"schema": "oracle-budget-v3", "root": str(root), "datasets": datasets,
              "label_rule": "use logged K, the rounded layer mean of oracle cardinalities; do not reconstruct it from n or context cap",
              "aggregation": "F_log=max(0,1-B_log/K_log); mean over recorded steps per request, then equal-weight mean over requests",
              "field_mapping": {"per_step_block_count": "B_log=round(mean_layer(B_actual))",
                                "per_step_B_t_oracle": "K_log=round(mean_populated_layer(K_actual))"},
              "count_provenance": "logged-count diagnostic; equal layer B and K imply equality with the mean layerwise floor; historical layer equality is not independently verifiable from scalar logs",
              "policies": {}}
    policy_rows = {}
    context = budget_frac = None
    for policy in args.policies:
        rows = []
        for ds in datasets:
            path = root / f"{ds}_{policy}.json"
            obj = json.loads(path.read_text())
            context = context or obj["context_length"]
            if obj["context_length"] != context:
                raise ValueError("cells disagree on context length")
            fracs = {r["budget_frac"] for r in obj["results"]}
            if len(fracs) != 1:
                raise ValueError(f"{path}: mixed budget fractions {fracs}")
            frac = fracs.pop()
            if budget_frac is None:
                budget_frac = frac
            elif budget_frac != frac:
                raise ValueError("cells disagree on the budget fraction")
            rows += [dict(row, dataset=ds) for row in per_request(path)]
        result["policies"][policy] = summarize(rows)
        result["policies"][policy]["recorded_count_ranges"] = {
            "kept": [min(r["kept_min"] for r in rows), max(r["kept_max"] for r in rows)],
            "oracle": [min(r["oracle_min"] for r in rows), max(r["oracle_max"] for r in rows)],
            "requests_with_positive_floor": sum(r["positive_floor_steps"] > 0 for r in rows),
            "positive_floor_steps": sum(r["positive_floor_steps"] for r in rows),
            "positive_floor_steps_outside_k32": sum(r["positive_floor_outside_k32"] for r in rows)}
        policy_rows[policy] = rows
    if len(args.policies) == 2:
        result["paired_second_minus_first"] = paired_summary(*(policy_rows[p] for p in args.policies))

    blocks = int(np.ceil(context / args.block_size))
    result["context_tokens"] = context
    result["prompt_blocks"] = blocks
    result["budget_fraction"] = budget_frac
    result["offline_label_k"] = int(np.ceil(0.1 * blocks))
    result["served_oracle_k"] = min(blocks, max(32, int(np.floor(0.1 * blocks))))
    result["policy_budget_blocks"] = int(round(budget_frac * blocks))
    result["oracle_floor_branch_active"] = 32 > int(np.floor(0.1 * blocks))
    result["recall_ceiling_at_full_prompt"] = min(
        1.0, result["policy_budget_blocks"] / result["served_oracle_k"])

    text = json.dumps(result, indent=2) + "\n"
    if args.out:
        if args.out.exists():
            raise SystemExit(f"refusing to overwrite {args.out}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print("WROTE", args.out)
    if not args.quiet:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

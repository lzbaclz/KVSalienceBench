#!/usr/bin/env python3
"""Audit Exp#7's logged count mapping and the pinned uniform-layer invariant.

No model forward pass or GPU is used. The archive stores rounded layer means,
not layer arrays: the source check cannot retroactively certify an unpinned
historical execution. It demonstrates exactly when the scalar floor is valid.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_seer import DEFAULT_EXPORT, REVISION, expected_sources

FIELD_MAPPING = {
    "per_step_block_count": "B_log = round(mean_layer(actual retained-set cardinality B_l))",
    "per_step_B_t_oracle": "K_log = round(mean_populated_layer(actual oracle-set cardinality K_l))",
    "per_step_eps_measured": "mean_populated_layer(|oracle_l minus retained_l| / K_l)",
    "candidate_count_n": "not present in archived per-request records; not needed when K_log is supplied",
}
ROOTS = ("e2e_confirm", "budget_sweep/b0.10", "budget_sweep/b0.30",
         "budget_sweep/b0.50", "longctx/c16384_n64")


def floor_comparison(kept, oracle):
    kept, oracle = np.asarray(kept, float), np.asarray(oracle, float)
    if kept.shape != oracle.shape or kept.ndim != 1 or not len(kept) \
            or not np.isfinite(kept).all() or not np.isfinite(oracle).all() \
            or np.any(kept < 0) or np.any(oracle <= 0) \
            or np.any(kept != np.floor(kept)) or np.any(oracle != np.floor(oracle)):
        raise ValueError("aligned finite integer layer counts, B >= 0 and K > 0 required")
    return {"rounded_count_floor": float(max(0, 1 - round(float(kept.mean())) / round(float(oracle.mean())))),
            "mean_layer_floor": float(np.maximum(0, 1 - kept / oracle).mean()),
            "equal_cardinalities": bool(np.all(kept == kept[0]) and np.all(oracle == oracle[0]))}


def inspect_archive():
    reports = {}
    for root in ROOTS:
        files = sorted(p for p in (ROOT / "experiments/results" / root).glob("*.json")
                       if p.stem.endswith(("_h2o", "_xqp")))
        if not files:
            raise ValueError(f"no archived cells: {root}")
        n_requests = n_steps = 0
        keys = set()
        source_hashes = {}
        for path in files:
            blob = path.read_bytes()
            source_hashes[str(path.relative_to(ROOT))] = hashlib.sha256(blob).hexdigest()
            for row in json.loads(blob)["results"]:
                n_requests += 1
                keys.update(row)
                arrays = [np.asarray(row[k]) for k in list(FIELD_MAPPING)[:3]]
                if not all(a.ndim == 1 and len(a) == len(arrays[0]) and np.isfinite(a).all() for a in arrays):
                    raise ValueError(f"invalid scalar step series: {path}")
                if not len(arrays[0]) or np.any(arrays[0] < 0) or np.any(arrays[1] <= 0):
                    raise ValueError(f"invalid block/oracle counts: {path}")
                n_steps += len(arrays[0])
        reports[root] = {"cells": len(files), "requests": n_requests, "recorded_steps": n_steps,
                         "request_fields": sorted(keys), "source_sha256": source_hashes,
                         "historical_layer_equality": "not identifiable from scalar records; no per-layer B/K arrays or producer revision"}
    return reports


def check_pinned_uniform_counts(seer_root):
    actual = {str(p.relative_to(seer_root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(seer_root.rglob("*.py"))}
    if actual != expected_sources() or (seer_root / "PINNED_GIT_SHA.txt").read_text().strip() != REVISION:
        raise ValueError("the source export does not match the pinned experiment")
    sys.path.insert(0, str(seer_root))
    import torch
    from seer.eval.sim import MaskingSimulator, _wanted_topk_set
    from seer.eval.block_stats import BlockStatsBuffer
    from seer.policy.baselines import H2OPolicy
    from seer.policy.xqp import XQPPolicy

    torch.set_num_threads(1)
    records = []
    for name in ("h2o", "xqp"):
        for n in (7, 31, 32, 128, 160, 329, 512):
            for fraction in (.1, .2, .3, .5):
                # Invoke the real mask-recomputation path with layer-dependent
                # scores and one unseen block requiring the real stub-fill path.
                s = MaskingSimulator.__new__(MaskingSimulator)
                s._n_layers, s.block_size, s.budget_frac = 32, 32, fraction
                s.policy = H2OPolicy() if name == "h2o" else XQPPolicy.from_ckpt(
                    ROOT / "experiments/predictors/xqp_closed_2view_h4.json")
                s._stats_buffers = [BlockStatsBuffer() for _ in range(s._n_layers)]
                oracle_buffers = [BlockStatsBuffer() for _ in range(s._n_layers)]
                s._kept_set_per_layer = [set() for _ in range(s._n_layers)]
                s._prefetch_set_per_layer = [set() for _ in range(s._n_layers)]
                s.qk_recall, s.disable_prefetch = False, True
                for layer, (buf, obuf) in enumerate(zip(s._stats_buffers, oracle_buffers)):
                    values = {bid: float((bid + 3 * layer) % n) / n for bid in range(n)}
                    buf.update_step(1, {bid: v for bid, v in values.items() if bid < n - 1}, {})
                    obuf.update_step(1, values, {})
                    for bid in values:
                        buf.set_position(bid, 32 * bid)
                tokens = 32 * n
                s._recompute_kept_masks(tokens, "cpu", torch.float32)
                kept = [len(x) for x in s._kept_set_per_layer]
                # The logger uses floor(T/32)+1, even at exact block boundaries;
                # record that convention rather than silently replacing with ceil.
                oracle_argument = tokens // 32 + 1
                oracle = [len(_wanted_topk_set(buf, oracle_argument)) for buf in oracle_buffers]
                expected_b = max(1, round(fraction * n))
                expected_k = min(n, max(32, int(.1 * oracle_argument)))
                if set(kept) != {expected_b} or set(oracle) != {expected_k}:
                    raise AssertionError(f"nonuniform cardinalities: {name}, n={n}, f={fraction}")
                comparison = floor_comparison(kept, oracle)
                np.testing.assert_allclose(comparison["rounded_count_floor"], comparison["mean_layer_floor"], atol=1e-14)
                records.append({"policy": name, "candidate_blocks": n, "budget_fraction": fraction,
                                "layers": s._n_layers, "B_each_layer": expected_b, "K_each_layer": expected_k,
                                "oracle_count_argument": oracle_argument, **comparison})
    return {"kind": "CPU algorithm invariant check; not a historical model-execution replay",
            "source_revision": REVISION, "cases": len(records), "records": records}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seer-root", type=Path, default=DEFAULT_EXPORT)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    if args.out.exists():
        p.error("choose a fresh output path")
    report = {"schema": "oracle-count-mapping-v1", "field_mapping": FIELD_MAPPING,
              "interpretation": "archived F_log is a recorded-count diagnostic, an exact layerwise floor only conditional on common B and K",
              "archived_records": inspect_archive(),
              "pinned_invariant": check_pinned_uniform_counts(args.seer_root),
              "nonuniform_counterexample": {"B": [1, 3], "K": [2, 2], **floor_comparison([1, 3], [2, 2])}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"PASS: {report['pinned_invariant']['cases']} pinned uniform-cardinality cases; archive field mapping recorded")
    print("Historical per-layer equality remains unverified; it is not inferred from rounded scalar logs.")
    print(f"WROTE {args.out}")


if __name__ == "__main__":
    main()

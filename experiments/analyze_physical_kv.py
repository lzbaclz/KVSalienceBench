#!/usr/bin/env python3
"""Strict paired analysis of TWO validation cells, never silently intersect IDs."""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy import stats


def _index(obj):
    if obj.get("status") != "passed" or not obj.get("gates", {}).get("passed"):
        raise ValueError("only completed cells with passing gates may be analyzed")
    rows = obj.get("results", [])
    if not rows:
        raise ValueError("gate-only/empty result is not a benchmark")
    result = {}
    for row in rows:
        key = (row["dataset"], row["id"])
        if key in result:
            raise ValueError(f"duplicate request {key}")
        result[key] = row
    return result


def analyze(baseline, candidate, margin=0.02, allow_budget_difference=False):
    if not 0 < margin <= 1:
        raise ValueError("margin must be in (0,1]")
    a, b = _index(baseline), _index(candidate)
    if not baseline.get("weights_hashed") or not candidate.get("weights_hashed"):
        raise ValueError("comparison requires --hash-model-weights in both cells")
    if set(a) != set(b):
        raise ValueError("request sets differ; refusing intersection-only comparison")
    for key in ("data_sha256", "checkpoint_files_sha256", "source_sha256", "quality_metric"):
        if baseline[key] != candidate[key]:
            raise ValueError(f"incompatible {key}")
    for key in ("dtype", "device", "max_new_tokens", "chat", "fixed_output", "seed", "warmup"):
        if baseline["args"][key] != candidate["args"][key]:
            raise ValueError(f"incompatible argument {key}")
    for key in ("torch", "transformers", "cuda", "gpu_name", "cpu_threads"):
        if baseline["runtime"][key] != candidate["runtime"][key]:
            raise ValueError(f"incompatible runtime {key}")
    for key in ("block_size", "sink_blocks", "recent_blocks", "ema_decay", "cross_fraction",
                "prefill_chunk", "observation_window"):
        if baseline["config"][key] != candidate["config"][key]:
            raise ValueError(f"incompatible selector configuration {key}")
    if not allow_budget_difference and baseline["config"]["budget_fraction"] != candidate["config"]["budget_fraction"]:
        raise ValueError("budgets differ; explicit --allow-budget-difference required for a full-cache reference")
    if baseline["args"]["policy"] == candidate["args"]["policy"] and baseline["predictor_sha256"] != candidate["predictor_sha256"]:
        raise ValueError("same-policy comparison changed predictor checkpoint")
    by_dataset = defaultdict(list)
    pairs = []
    for key in sorted(a):
        x, y = a[key], b[key]
        if x["input_token_sha256"] != y["input_token_sha256"]:
            raise ValueError(f"tokenized input differs for {key}")
        if (x["f1"] is None) != (y["f1"] is None):
            raise ValueError("incompatible per-request quality availability")
        if baseline["quality_metric"] is not None and x["f1"] is None:
            raise ValueError("quality run contains missing F1")
        if x["f1"] is not None and y["f1"] is not None:
            if not all(np.isfinite(z) and 0 <= z <= 1 for z in (x["f1"], y["f1"])):
                raise ValueError("invalid F1")
            by_dataset[key[0]].append(y["f1"] - x["f1"])
        pairs.append({"dataset": key[0], "id": key[1],
                      "baseline_output_tokens": x["output_tokens"],
                      "candidate_output_tokens": y["output_tokens"],
                      "f1_delta": None if x["f1"] is None else y["f1"]-x["f1"],
                      "final_kv_storage_delta_bytes": y["final_cache"]["kv_storage_bytes"] - x["final_cache"]["kv_storage_bytes"]})
    means = np.array([np.mean(v) for _, v in sorted(by_dataset.items())])
    quality = {"dataset_mean_deltas": {k: float(np.mean(v)) for k, v in sorted(by_dataset.items())},
               "number_of_dataset_clusters": len(means),
               "equal_dataset_mean_delta": float(means.mean()) if len(means) else None,
               "margin": margin, "tost_p": None, "equivalent": None,
               "note": "With fewer than five dataset clusters, report descriptive validation only; no equivalence conclusion."}
    if len(means) >= 5:
        se = stats.sem(means)
        if se > 0:
            p = max(stats.t.sf((means.mean()+margin)/se, len(means)-1),
                    stats.t.cdf((means.mean()-margin)/se, len(means)-1))
            ci = stats.t.interval(.90, len(means)-1, loc=means.mean(), scale=se)
            quality.update(tost_p=float(p), equivalent=bool(p < .05),
                           mean_delta_ci90=[float(v) for v in ci],
                           note="Exploratory paired dataset-cluster t-TOST; clusters treated as independent. Margin must be fixed before seeing results. Not a substitute for the paper's historical confirmatory experiment.")
        else:
            quality["note"] = "Zero estimated between-dataset variance; no inferential TOST reported."
    def summarize(rows):
        out = {"requests": len(rows)}
        for key in ("local_ttft_ms", "local_tpot_ms", "local_generation_ms"):
            values = np.array([r[key] for r in rows.values() if r[key] is not None])
            if not np.isfinite(values).all():
                raise ValueError("non-finite latency")
            out[key] = {"n": len(values), "median": float(np.median(values)) if len(values) else None}
            for q, min_n in ((95, 20), (99, 100), (99.9, 1000)):
                out[key][f"p{q}"] = float(np.percentile(values, q)) if len(values) >= min_n else None
        return out
    return {"schema_version": 1, "paired_requests": len(a), "quality": quality,
            "baseline": summarize(a), "candidate": summarize(b), "pairs": pairs,
            "weights_fully_hashed": bool(baseline["weights_hashed"] and candidate["weights_hashed"]),
            "claim_boundary": "No production throughput, HBM-pool capacity, paging or DMA conclusion. Nonsignificance does not prove equivalence. Different output lengths confound latency comparisons."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", required=True); p.add_argument("--candidate", required=True)
    p.add_argument("--out", required=True); p.add_argument("--margin", type=float, default=.02)
    p.add_argument("--allow-budget-difference", action="store_true")
    args = p.parse_args()
    result = analyze(json.loads(Path(args.baseline).read_text()), json.loads(Path(args.candidate).read_text()),
                     args.margin, args.allow_budget_difference)
    with Path(args.out).open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False); f.write("\n")


if __name__ == "__main__":
    main()

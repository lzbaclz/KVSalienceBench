#!/usr/bin/env python3
"""Regenerate (or verify) a physical-KV run summary from its raw per-request cells.

The manuscript's Llama/Qwen tables are read off `summary.json`. Without a
committed producer that file is an unverifiable hand-assembled aggregate, so
this script derives every field from the tracked inputs:

  <dir>/{full,<policy>.<backend>.b<budget>}.r<rep>.json  per-request measurements
  <dir>/paired.b<budget>.r<rep>.json                     strict paired analysis
  <qa>.jsonl.meta.json                                   frozen prompt cohort
  query-v2/<tag>.analysis.json + <tag>.jsonl.meta.json   corrected query probe

It computes nothing new: means, medians and match counts only. `--check`
compares against the committed file and exits nonzero on any numeric drift.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median

MIB = 1024 ** 2
GIB = 1024 ** 3


def _cell_name(path: Path) -> str:
    return path.name


def _load(path: Path) -> dict:
    obj = json.loads(path.read_text())
    if obj.get("status") != "passed" or not obj.get("gates", {}).get("passed"):
        raise ValueError(f"{path}: only completed cells with passing gates may be summarized")
    if not obj.get("results"):
        raise ValueError(f"{path}: gate-only output is not a measurement cell")
    return obj


def cell_summary(obj: dict) -> dict:
    rows = obj["results"]
    return {
        "policy": obj["args"]["policy"],
        "backend": obj["args"]["backend"],
        "budget": obj["config"]["budget_fraction"],
        "n": len(rows),
        "gates_passed": bool(obj["gates"]["passed"]),
        "f1_mean": mean(r["f1"] for r in rows),
        "f1_by_dataset": {
            ds: mean(r["f1"] for r in rows if r["dataset"] == ds)
            for ds in sorted({r["dataset"] for r in rows})
        },
        "out_tokens_mean": mean(r["output_tokens"] for r in rows),
        "ttft_ms_median": median(r["local_ttft_ms"] for r in rows),
        "tpot_ms_median": median(r["local_tpot_ms"] for r in rows),
        "kv_storage_mib_mean": mean(r["final_cache"]["kv_storage_bytes"] for r in rows) / MIB,
        "kv_live_mib_mean": mean(r["final_cache"]["logical_live_kv_bytes"] for r in rows) / MIB,
        "peak_prefill_gib_mean": mean(
            r["allocator_prefill_and_initial_compaction"]["peak_allocated_bytes"] for r in rows) / GIB,
        "peak_decode_gib_mean": mean(r["allocator_decode"]["peak_allocated_bytes"] for r in rows) / GIB,
        "eos_rate": mean(1.0 if r["stopped_on_eos"] else 0.0 for r in rows),
    }


def parity(masked: dict, physical: dict) -> dict:
    a = {(r["dataset"], r["id"]): r for r in masked["results"]}
    b = {(r["dataset"], r["id"]): r for r in physical["results"]}
    if set(a) != set(b):
        raise ValueError("mask/physical request sets differ; refusing an intersection-only report")
    keys = sorted(a)
    for k in keys:
        if a[k]["input_token_sha256"] != b[k]["input_token_sha256"]:
            raise ValueError(f"tokenized input differs for {k}")
    return {
        "n": len(keys),
        "exact_f1_match": sum(1 for k in keys if a[k]["f1"] == b[k]["f1"]),
        "exact_token_match": sum(1 for k in keys
                                 if a[k]["generated_token_ids"] == b[k]["generated_token_ids"]),
        "mean_abs_f1": mean(abs(a[k]["f1"] - b[k]["f1"]) for k in keys),
        "mean_f1_masked": mean(a[k]["f1"] for k in keys),
        "mean_f1_physical": mean(b[k]["f1"] for k in keys),
        "mean_store_masked_mib": mean(a[k]["final_cache"]["kv_storage_bytes"] for k in keys) / MIB,
        "mean_store_physical_mib": mean(b[k]["final_cache"]["kv_storage_bytes"] for k in keys) / MIB,
        "mean_live_masked_mib": mean(a[k]["final_cache"]["logical_live_kv_bytes"] for k in keys) / MIB,
        "mean_live_physical_mib": mean(b[k]["final_cache"]["logical_live_kv_bytes"] for k in keys) / MIB,
    }


def query_v2_summary(analysis_path: Path, meta_path: Path) -> dict:
    an = json.loads(analysis_path.read_text())
    meta = json.loads(meta_path.read_text())
    if an.get("protocol") != "phase-aligned-query-v2" or meta.get("trace_version") != 2:
        raise ValueError("expected a version-2 phase-aligned query trace and analysis")
    if an["trace_sha256"] != meta["trace_sha256"]:
        raise ValueError("analysis and trace manifest disagree on the trace hash")
    deltas = [r["delta"] for r in an["requests"]]
    return {
        "mean_request_auc_delta": an["mean_request_auc_delta"],
        "ci95": an["exploratory_request_bootstrap_ci95"],
        "n_train": len(an["train_ids"]),
        "n_test": len(an["test_ids"]),
        "n_test_negative_delta": sum(1 for d in deltas if d < 0),
        "n_test_positive_delta": sum(1 for d in deltas if d > 0),
        "mean_two_view_auc": mean(r["two_view_auc"] for r in an["requests"]),
        "mean_plus_dotmax_auc": mean(r["plus_dotmax_auc"] for r in an["requests"]),
        "trace_rows": meta["rows"],
        "trace_version": meta["trace_version"],
        "query_space": meta["query_space"],
        "query_probe": meta["query_probe"],
        "trace_sha256": an["trace_sha256"],
    }


def build(run_dir: Path, model: str, qa_meta: Path, query_analysis: Path | None,
          query_meta: Path | None, note: str | None) -> dict:
    cells = {}
    for path in sorted(run_dir.glob("*.r*.json")):
        if path.name.startswith(("paired.", "probe")) or path.name.endswith(".failed.json"):
            continue
        cells[_cell_name(path)] = _load(path)
    if not cells:
        raise ValueError(f"no measurement cells under {run_dir}")
    reference = next(iter(cells.values()))
    for obj in cells.values():
        for key in ("data_sha256", "checkpoint_files_sha256", "source_sha256"):
            if obj[key] != reference[key]:
                raise ValueError("cells disagree on input/model/source hashes")
        for key in ("dtype", "max_new_tokens", "max_input_tokens", "chat", "seed"):
            if obj["args"][key] != reference["args"][key]:
                raise ValueError(f"cells disagree on argument {key}")
    cohort = json.loads(qa_meta.read_text())
    if cohort["data_sha256"] != reference["data_sha256"]:
        raise ValueError("prompt cohort manifest does not match the measured data file")
    requests = cohort["requests"]
    datasets = {}
    for row in requests:
        datasets[row["dataset"]] = datasets.get(row["dataset"], 0) + 1

    out = {
        "model": Path(reference["args"]["model"]).name,
        "gpu": reference["runtime"]["gpu_name"],
        "dtype": reference["args"]["dtype"],
        "transformers": reference["runtime"]["transformers"],
        "n_requests": len(requests),
        "datasets": dict(sorted(datasets.items())),
        "truncated_prompts": sum(1 for r in requests if r["truncated"]),
        "max_input_tokens": reference["args"]["max_input_tokens"],
        "max_new_tokens": reference["args"]["max_new_tokens"],
    }
    if note:
        out["note"] = note
    out["cells"] = {name: cell_summary(obj) for name, obj in cells.items()}

    comparisons = {}
    for name, obj in cells.items():
        if obj["args"]["backend"] != "masked":
            continue
        policy, budget = obj["args"]["policy"], obj["config"]["budget_fraction"]
        twin = name.replace(".masked.", ".physical.")
        if twin not in cells:
            raise ValueError(f"masked cell {name} has no physical twin")
        comparisons[f"{policy}.b{budget}"] = parity(obj, cells[twin])
    out["masked_vs_physical"] = comparisons

    paired = {}
    for path in sorted(run_dir.glob("paired.*.json")):
        paired[path.name] = json.loads(path.read_text())["quality"]
    if paired:
        out["xqp_vs_h2o_physical"] = paired
    if query_analysis is not None:
        out["query_v2"] = query_v2_summary(query_analysis, query_meta)
    if model and out["model"] != model:
        raise ValueError(f"expected model {model}, cells name {out['model']}")
    return out


def _drift(new, old, path="summary"):
    """Yield human-readable differences, tolerating float round-trip noise."""
    if isinstance(new, dict) and isinstance(old, dict):
        for key in sorted(set(new) | set(old)):
            if key not in new:
                yield f"{path}.{key}: missing from regenerated summary"
            elif key not in old:
                yield f"{path}.{key}: added by regenerated summary"
            else:
                yield from _drift(new[key], old[key], f"{path}.{key}")
    elif isinstance(new, (int, float)) and isinstance(old, (int, float)) \
            and not isinstance(new, bool) and not isinstance(old, bool):
        if abs(float(new) - float(old)) > 1e-9 * max(1.0, abs(float(old))):
            yield f"{path}: {new!r} != {old!r}"
    elif new != old:
        yield f"{path}: {new!r} != {old!r}"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--qa-meta", required=True, type=Path)
    p.add_argument("--query-analysis", type=Path)
    p.add_argument("--query-meta", type=Path)
    p.add_argument("--model", default="")
    p.add_argument("--note", default="")
    p.add_argument("--out", type=Path)
    p.add_argument("--check", action="store_true",
                   help="compare with --out instead of writing it; nonzero exit on drift")
    args = p.parse_args(argv)
    if (args.query_analysis is None) != (args.query_meta is None):
        p.error("--query-analysis and --query-meta must be given together")
    summary = build(args.run_dir, args.model, args.qa_meta,
                    args.query_analysis, args.query_meta, args.note or None)
    target = args.out or (args.run_dir / "summary.json")
    if args.check:
        differences = list(_drift(summary, json.loads(target.read_text())))
        if differences:
            print(f"DRIFT in {target}:")
            for line in differences:
                print("  " + line)
            return 1
        print(f"OK: {target} matches the raw cells")
        return 0
    target.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"WROTE {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

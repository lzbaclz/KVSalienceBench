#!/usr/bin/env python3
"""Run a fail-closed physical-KV validation cell; see docs/PHYSICAL_KV_VALIDATION.md."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="local HF checkpoint directory, no automatic downloads")
    p.add_argument("--data", required=True, help="English QA JSONL: id,dataset,prompt,answers")
    p.add_argument("--out", required=True, help="new JSON output; existing files are never overwritten")
    p.add_argument("--policy", choices=["full", "h2o_block", "xqp_reconstructed"], required=True)
    p.add_argument("--backend", choices=["physical", "masked"], default="physical")
    p.add_argument("--budget", type=float, required=True)
    p.add_argument("--block-size", type=int, default=32)
    p.add_argument("--sink-blocks", type=int, default=1)
    p.add_argument("--recent-blocks", type=int, default=1)
    p.add_argument("--prefill-chunk", type=int, default=256)
    p.add_argument("--observation-window", type=int, default=64)
    p.add_argument("--weights", default=str(ROOT / "experiments/predictors/xqp_closed_2view_h4.json"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--max-input-tokens", type=int, default=4096,
                   help="reject longer rendered inputs; NEVER silently truncate")
    p.add_argument("--chat", action="store_true", help="apply tokenizer chat template once")
    p.add_argument("--fixed-output", action="store_true",
                   help="ignore EOS for controlled-length timing; quality is NOT scored")
    p.add_argument("--gate-only", action="store_true")
    p.add_argument("--gate-tokens", type=int, default=1024)
    p.add_argument("--gate-steps", type=int, default=4)
    p.add_argument("--gate-report-only", action="store_true",
                   help="record every gated step instead of raising at the first tolerance violation; "
                        "the output status is 'gates_reported', never 'passed' (diagnostic, not validation)")
    p.add_argument("--gate-requests", default="0",
                   help="comma-separated request indices to gate (default: the first request only). "
                        "Every listed request must pass; per-request results are stored under "
                        "'extended_gates' and the first one under 'gates' (PC audit item A4).")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hash-model-weights", action="store_true",
                   help="hash all checkpoint weight files (slow but recommended for final runs)")
    return p


def git_info():
    def cmd(*parts):
        try:
            return subprocess.check_output(["git", *parts], cwd=ROOT, text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return {"revision": cmd("rev-parse", "HEAD"),
            "dirty": bool(cmd("status", "--porcelain"))}


def atomic_json(path, obj):
    # Exclusive final creation prevents two jobs from clobbering one result.
    text = json.dumps(obj, indent=2, allow_nan=False) + "\n"
    import tempfile
    destination = Path(path)
    fd, name = tempfile.mkstemp(prefix=destination.name + ".tmp.", dir=destination.parent)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text); f.flush(); os.fsync(f.fileno())
        os.link(name, destination)  # atomic publication, no overwrite
    finally:
        Path(name).unlink(missing_ok=True)


def main(argv=None):
    args = parser().parse_args(argv)
    out = Path(args.out).resolve()
    if out.exists() or Path(str(out) + ".failed.json").exists():
        raise SystemExit(f"output already exists: {out}")
    if min(args.max_new_tokens, args.max_input_tokens, args.gate_tokens, args.gate_steps) < 1 or args.warmup < 0:
        raise SystemExit("positive token/gate counts and nonnegative warmup required")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "status": "running", "args": vars(args), "git": git_info()}
    try:
        import random
        import numpy as np
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from xqp.physical_kv import EvictionConfig, load_two_view_weights
        from xqp.physical_validation import (read_requests, sha256_file, correctness_gates,
                                            measure_request, qa_f1)
        if not Path(args.model).is_dir():
            raise ValueError("--model must be a local HF checkpoint directory")
        device = torch.device(args.device)
        if device.type not in {"cpu", "cuda"}:
            raise ValueError("only CPU or CUDA supported")
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable; no CPU fallback")
            torch.cuda.set_device(device)
            if args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
                raise ValueError("device lacks bfloat16 support")
        if device.type == "cpu" and args.dtype != "float32":
            raise ValueError("CPU validation requires float32")
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cuda.matmul.allow_tf32 = False
        cfg = EvictionConfig(policy=args.policy, backend=args.backend, budget_fraction=args.budget,
                             block_size=args.block_size, sink_blocks=args.sink_blocks,
                             recent_blocks=args.recent_blocks, prefill_chunk=args.prefill_chunk,
                             observation_window=args.observation_window)
        weights = load_two_view_weights(args.weights) if args.policy == "xqp_reconstructed" else None
        requests = read_requests(args.data)
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, local_files_only=True, trust_remote_code=False,
            torch_dtype=getattr(torch, args.dtype), attn_implementation="eager"
        ).to(device).eval()
        encoded = []
        for row in requests:
            if args.chat:
                ids = tokenizer.apply_chat_template([{"role": "user", "content": row["prompt"]}],
                        tokenize=True, add_generation_prompt=True, return_tensors="pt")
            else:
                ids = tokenizer(row["prompt"], return_tensors="pt", add_special_tokens=True).input_ids
            if ids.shape[1] > args.max_input_tokens:
                raise ValueError(f"{row['dataset']}/{row['id']}: {ids.shape[1]} input tokens exceed cap; prepare explicit frozen truncation before running")
            cfg.budget(ids.shape[1])
            if ids.shape[1] + args.max_new_tokens > model.config.max_position_embeddings:
                raise ValueError("input+output exceeds checkpoint context limit")
            encoded.append(ids)  # CPU copies only; one request owns GPU memory.
        files = sorted(Path(args.model).glob("*.json"))
        if args.hash_model_weights:
            weight_files = sorted(Path(args.model).glob("*.safetensors")) + sorted(Path(args.model).glob("*.bin"))
            if not weight_files:
                raise ValueError("--hash-model-weights found no checkpoint weight files")
            files += weight_files
        source_files = [ROOT / "xqp/physical_kv.py", ROOT / "xqp/physical_validation.py", Path(__file__)]
        payload.update({"config": asdict(cfg), "data_sha256": sha256_file(args.data),
            "checkpoint_files_sha256": {p.name: sha256_file(p) for p in files},
            "weights_hashed": args.hash_model_weights,
            "predictor_sha256": sha256_file(args.weights) if weights else None,
            "source_sha256": {p.name: sha256_file(p) for p in source_files},
            "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                "transformers": importlib.metadata.version("transformers"), "cuda": torch.version.cuda,
                "device": str(device), "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "cpu_threads": torch.get_num_threads(), "pid": os.getpid()},
            "measurement_scope": "reference eager, single request, irreversible per-layer block selection; NOT production throughput, offload, DMA or allocator-pool capacity",
            "policy_scope": "h2o_block is a shared-per-KV-head block accumulator; xqp_reconstructed is a new explicitly defined online realization, not the exact historical SEER scorer",
            "quality_metric": None if args.fixed_output else "English QA token F1, max over references"})
        gate_indices = [int(x) for x in args.gate_requests.split(",")]
        if not gate_indices or any(i < 0 or i >= len(encoded) for i in gate_indices):
            raise ValueError("--gate-requests must name existing request indices")
        extended = []
        for gi in gate_indices:
            gate_ids = encoded[gi][:, :args.gate_tokens].to(device)
            cfg.budget(gate_ids.shape[1])
            if gate_ids.shape[1] + args.gate_steps > model.config.max_position_embeddings:
                raise ValueError("gate exceeds model context")
            g = correctness_gates(model, gate_ids, cfg, weights, args.gate_steps,
                                  strict=not args.gate_report_only)
            # Three evidence levels, reported separately: numeric tolerance (the pass
            # condition), top-1 agreement at every gated step, and (for measured
            # cells) free-running greedy parity in the paired summary.
            g["top1_agreement"] = {
                "native_vs_full_session": sum(c["argmax_equal"] for c in g["native_full_checks"]),
                "masked_vs_physical_replay": sum(c["argmax_equal"] for c in g["masked_replay_checks"]),
                "steps_checked": len(g["native_full_checks"])}
            g["request_index"] = gi
            g["dataset"], g["id"] = requests[gi]["dataset"], requests[gi]["id"]
            extended.append(g)
            del gate_ids
        payload["gates"] = extended[0]
        payload["extended_gates"] = extended
        if args.gate_report_only:
            if not args.gate_only:
                raise ValueError("--gate-report-only is a diagnostic; combine it with --gate-only")
            payload["status"] = "gates_reported"
            payload["gate_report"] = {
                "all_passed_numeric": all(g["passed"] for g in extended),
                "top1_agreement_native": sum(g["top1_agreement"]["native_vs_full_session"] for g in extended),
                "top1_agreement_replay": sum(g["top1_agreement"]["masked_vs_physical_replay"] for g in extended),
                "steps_checked": sum(g["top1_agreement"]["steps_checked"] for g in extended),
                "max_tolerance_ratio_native": max(c["max_tolerance_ratio"] for g in extended for c in g["native_full_checks"]),
                "max_tolerance_ratio_replay": max(c["max_tolerance_ratio"] for g in extended for c in g["masked_replay_checks"]),
                "max_abs_error_native": max(c["max_abs_error"] for g in extended for c in g["native_full_checks"]),
                "max_abs_error_replay": max(c["max_abs_error"] for g in extended for c in g["masked_replay_checks"]),
                "note": "diagnostic record; a numeric-tolerance miss with top-1 agreement is reported as such, not as a pass"}
            atomic_json(out, payload)
            print(out)
            return 0
        if not args.gate_only:
            eos = model.generation_config.eos_token_id
            if eos is None:
                eos = tokenizer.eos_token_id
            eos_ids = set(eos if isinstance(eos, list) else [eos]) - {None}
            for _ in range(args.warmup):
                measure_request(model, encoded[0].to(device), cfg, weights,
                                min(8, args.max_new_tokens), eos_ids, True)
            # Outside measurement. Kernel warm-up persists; cache allocator starts
            # empty apart from model state. Later requests may reuse allocations.
            import gc
            gc.collect()
            if device.type == "cuda":
                torch.cuda.synchronize(device); torch.cuda.empty_cache()
            results = []
            for index, (row, ids) in enumerate(zip(requests, encoded)):
                result = measure_request(model, ids.to(device), cfg, weights,
                                         args.max_new_tokens, eos_ids, args.fixed_output)
                result.update({"id": row["id"], "dataset": row["dataset"]})
                pred = tokenizer.decode(result["generated_token_ids"], skip_special_tokens=True)
                result["prediction"] = pred
                result["f1"] = None if args.fixed_output else qa_f1(pred, row["answers"])
                results.append(result)
                print(f"[{index+1}/{len(requests)}] {row['dataset']}/{row['id']} "
                      f"F1={result['f1']} tokens={result['output_tokens']}", flush=True)
            payload["results"] = results
        payload["status"] = "passed"
        atomic_json(out, payload)
        print(out)
        return 0
    except Exception as exc:
        payload.update({"status": "failed", "error": str(exc), "traceback": traceback.format_exc()})
        atomic_json(str(out) + ".failed.json", payload)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

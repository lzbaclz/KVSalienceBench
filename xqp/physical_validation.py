"""Correctness gates and single-request measurements for physical_kv.

No networking, model downloads, queueing model or synthetic DMA penalties.
A failed gate raises: there is deliberately no best-effort success fallback.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import gc
import hashlib
import json
from pathlib import Path
import re
import string
import time
from collections import Counter

import torch

from .physical_kv import EvictionConfig, PhysicalKVSession


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def token_digest(ids: torch.Tensor) -> str:
    return hashlib.sha256(ids.detach().cpu().numpy().astype("<i8").tobytes()).hexdigest()


def read_requests(path: str | Path) -> list[dict]:
    """English QA JSONL only; explicit IDs, dataset and references are mandatory."""
    rows, seen = [], set()
    for line_no, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"line {line_no}: expected object")
        for field in ("id", "dataset", "prompt"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"line {line_no}: nonempty string {field} required")
        if not isinstance(row.get("answers"), list) or not row["answers"] or not all(
                isinstance(x, str) and x.strip() for x in row["answers"]):
            raise ValueError(f"line {line_no}: nonempty answers list required")
        key = (row["dataset"], row["id"])
        if key in seen:
            raise ValueError(f"duplicate request {key}")
        seen.add(key)
        rows.append(row)
    if not rows:
        raise ValueError("no requests")
    return rows


def qa_f1(prediction: str, answers: list[str]) -> float:
    """Standard English token-F1; NOT a metric for code/summaries/Chinese QA."""
    def normalize(s):
        s = s.lower().translate(str.maketrans("", "", string.punctuation))
        return re.sub(r"\b(a|an|the)\b", " ", s).split()
    p = normalize(prediction)
    scores = []
    for answer in answers:
        a = normalize(answer)
        if not p or not a:
            scores.append(float(p == a))
            continue
        common = sum((Counter(p) & Counter(a)).values())
        scores.append(2.0 * common / (len(p) + len(a)))
    return max(scores)


def synchronize(device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def allocator_snapshot(device) -> dict:
    if torch.device(device).type != "cuda":
        return {"allocated_bytes": None, "reserved_bytes": None,
                "peak_allocated_bytes": None, "peak_reserved_bytes": None}
    return {"allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}


def reset_peak(device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _tolerances(dtype):
    # Report all errors/tolerances. Passing is numerical validation, not proof
    # of identical greedy choices on near-tied logits in arbitrary checkpoints.
    if dtype == torch.float32:
        return 2e-5, 2e-4
    if dtype == torch.float16:
        return 0.02, 0.01
    if dtype == torch.bfloat16:
        return 0.15, 0.02
    raise ValueError("unsupported validation dtype")


def _compare_logits(actual, expected, atol, rtol, strict: bool = True) -> dict:
    diff = (actual.float() - expected.float()).abs()
    bound = atol + rtol * expected.float().abs()
    finite = bool(torch.isfinite(diff).all())
    passed = finite and bool((diff <= bound).all())
    result = {"passed": passed, "max_abs_error": float(diff.max()),
              "max_tolerance_ratio": float((diff / bound).max()),
              "argmax_equal": bool(torch.equal(actual.argmax(-1), expected.argmax(-1))),
              "atol": atol, "rtol": rtol}
    if not passed and strict:
        raise AssertionError(f"logit parity failed: {result}")
    return result


@torch.inference_mode()
def correctness_gates(model, ids: torch.Tensor, cfg: EvictionConfig,
                      weights=None, steps: int = 4, strict: bool = True) -> dict:
    """1) Native full-cache parity. 2) Irreversible mask/compact schedule replay.

    The teacher tokens and schedules are frozen by the masked reference, so
    the second test isolates storage/position semantics, not policy tie noise.
    It does not certify parity with the historical external SEER simulator.

    ``strict=False`` is a REPORT mode: no check raises, every gated step's error
    and top-1 agreement is recorded, and the returned ``passed`` is the
    conjunction of all checks. It exists so that a tolerance violation at a
    longer prompt or after more evictions is measured rather than hidden behind
    the first failing step; it is never a validation pass.
    """
    if steps < 1:
        raise ValueError("at least one gate decode step required")
    atol, rtol = _tolerances(next(model.parameters()).dtype)
    n = ids.shape[1]
    full_cfg = replace(cfg, policy="full", budget_fraction=1, backend="physical")
    native = []
    tokens = []
    # Native unmodified HF model, same chunk boundaries but ordinary causal mask.
    from transformers.cache_utils import DynamicCache
    cache = DynamicCache()
    tail = min(cfg.observation_window, n)
    bounds = [(a, min(a + cfg.prefill_chunk, n - tail)) for a in range(0, n - tail, cfg.prefill_chunk)]
    bounds.append((n - tail, n))
    for a, b in bounds:
        out = model(ids[:, a:b], past_key_values=cache, use_cache=True,
                    output_attentions=False, logits_to_keep=1, return_dict=True)
        cache = out.past_key_values
    logits = out.logits[:, -1, :]
    native.append(logits.detach().cpu())
    for _ in range(steps):
        token = logits.argmax(-1)[:, None]
        tokens.append(token)
        out = model(token, past_key_values=cache, use_cache=True,
                    output_attentions=False, logits_to_keep=1, return_dict=True)
        cache = out.past_key_values
        logits = out.logits[:, -1, :]
        native.append(logits.detach().cpu())
    del cache, out, logits
    full_checks = []
    with PhysicalKVSession(model, full_cfg, n) as session:
        logits = session.prefill(ids)
        session.evict()
        full_checks.append(_compare_logits(logits.cpu(), native[0], atol, rtol, strict))
        for i, token in enumerate(tokens, 1):
            logits = session.decode(token)
            session.evict()
            full_checks.append(_compare_logits(logits.cpu(), native[i], atol, rtol, strict))
        session.assert_valid()
    del logits
    schedules, teacher, masked_logits = [], [], []
    with PhysicalKVSession(model, replace(cfg, backend="masked"), n, weights) as session:
        logits = session.prefill(ids)
        masked_logits.append(logits.cpu())
        session.evict()
        schedules.append(session.schedule())
        for _ in range(steps):
            token = logits.argmax(-1)[:, None]
            teacher.append(token)
            logits = session.decode(token)
            masked_logits.append(logits.cpu())
            session.evict()
            session.assert_valid()
            schedules.append(session.schedule())
        masked_memory = session.memory()
    del logits
    physical_checks = []
    with PhysicalKVSession(model, replace(cfg, backend="physical"), n, weights) as session:
        logits = session.prefill(ids)
        physical_checks.append(_compare_logits(logits.cpu(), masked_logits[0], atol, rtol, strict))
        session.evict(schedules[0])
        for i, token in enumerate(teacher, 1):
            logits = session.decode(token)
            physical_checks.append(_compare_logits(logits.cpu(), masked_logits[i], atol, rtol, strict))
            session.evict(schedules[i])
            session.assert_valid()
        physical_memory = session.memory()
    if physical_memory["kv_storage_bytes"] != physical_memory["logical_live_kv_bytes"]:
        raise AssertionError("physical storage is not compact")
    if physical_memory["logical_live_kv_bytes"] != masked_memory["logical_live_kv_bytes"]:
        raise AssertionError("replay live budget mismatch")
    if cfg.policy != "full" and cfg.budget_fraction < 1 and not (
            physical_memory["kv_storage_bytes"] < masked_memory["kv_storage_bytes"]):
        raise AssertionError("physical backend did not reduce retained tensor storage")
    all_passed = all(c["passed"] for c in full_checks + physical_checks)
    return {"passed": all_passed, "strict": strict, "prompt_tokens": n, "decode_steps": steps,
            "input_token_sha256": token_digest(ids), "config": asdict(cfg),
            "native_full_checks": full_checks, "masked_replay_checks": physical_checks,
            "masked_memory": masked_memory, "physical_memory": physical_memory,
            "scope": "this pinned adapter; fixed-schedule numerical parity, not legacy SEER or serving performance"}


@torch.inference_mode()
def measure_request(model, ids: torch.Tensor, cfg: EvictionConfig, weights=None,
                    max_new_tokens: int = 128, eos_ids: set[int] | None = None,
                    fixed_output: bool = False) -> dict:
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if ids.shape[1] + max_new_tokens > model.config.max_position_embeddings:
        raise ValueError("prompt plus requested output exceeds declared model context")
    eos_ids = eos_ids or set()
    device = ids.device
    generated, delivery = [], []
    gc.collect()
    synchronize(device)
    before = allocator_snapshot(device)
    reset_peak(device)
    with PhysicalKVSession(model, cfg, ids.shape[1], weights) as session:
        start = time.perf_counter()
        logits = session.prefill(ids)
        session.evict()
        # Scalar delivery waits for the default stream, including eviction.
        # Single-request local API, not network TTFT or production service TPOT.
        token = int(logits.argmax(-1).item())
        delivery.append(time.perf_counter())
        generated.append(token)
        post_prefill = allocator_snapshot(device)
        reset_peak(device)
        while len(generated) < max_new_tokens and (fixed_output or generated[-1] not in eos_ids):
            next_id = torch.tensor([[generated[-1]]], device=device, dtype=torch.long)
            logits = session.decode(next_id)
            session.evict()
            token = int(logits.argmax(-1).item())
            delivery.append(time.perf_counter())
            generated.append(token)
        after = allocator_snapshot(device)
        # Synchronous audits and CPU serialization only AFTER the last delivery.
        session.assert_valid()
        memory = session.memory()
        digest = session.selection_digest()
        budget = session.budget_blocks
    total = delivery[-1] - start
    tpot = (delivery[-1] - delivery[0]) / (len(delivery) - 1) if len(delivery) > 1 else None
    return {"generated_token_ids": generated, "output_tokens": len(generated),
            "input_tokens": ids.shape[1], "input_token_sha256": token_digest(ids),
            "budget_blocks": budget, "final_selection_sha256": digest,
            "local_ttft_ms": (delivery[0] - start) * 1000,
            "local_tpot_ms": None if tpot is None else tpot * 1000,
            "local_itl_ms": [(b-a)*1000 for a, b in zip(delivery[:-1], delivery[1:])],
            "local_generation_ms": total * 1000,
            "stopped_on_eos": not fixed_output and generated[-1] in eos_ids,
            "allocator_before": before,
            "allocator_prefill_and_initial_compaction": post_prefill,
            "allocator_decode": after, "final_cache": memory,
            "timing_scope": "single-request tensor-ready to token-ID delivery; selection/compaction included; tokenization, text decoding and diagnostics excluded"}

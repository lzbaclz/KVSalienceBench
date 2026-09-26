#!/usr/bin/env python3
"""Root-cause diagnosis for the corrupted Qwen2.5 full-cache outputs (PC item P0-1).

The archived expansion sweep (experiments/results/expand) was produced by the
external SEER runner, whose ``--dtype`` default is float16 and whose ``full``
policy is a plain HuggingFace greedy ``generate``. In the Qwen2.5 tokenizer,
token id 0 decodes to "!", and ``torch.argmax`` over an all-NaN logit row
returns index 0. This script reproduces the exact prompt rendering of the
pinned SEER revision (local LongBench rows, first-N sampling, 4096-token
truncation, chat wrapping) and runs a *native, unmodified* HF greedy loop in
float16, bfloat16 and float32, recording per step whether the logits were
finite, whether the whole row was NaN, and which token was emitted.

Outputs one JSON per dtype under --out-dir with, per request: the emitted token
ids, the decoded text, the count of all-NaN steps, the count of token-0
emissions, the archived prediction for the same (dataset, id) and whether the
new float16 text equals it. This is a diagnosis of the generating path, not a
new task-quality experiment; it changes no archived file.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import string
import re
import sys
import time
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATASETS = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "triviaqa"]
SUFFIX = "\n\nAnswer as concisely as possible, with no explanation."


def render_seer_prompt(row: dict, tokenizer, context_tokens: int, chat: bool) -> str:
    """Bit-for-bit copy of the pinned SEER old_feat path (datasets._read_longbench_jsonl
    + datasets._truncate + runner chat wrapping)."""
    prompt = f"{row.get('context', '')}\n\nQuestion: {row.get('input', '')}\nAnswer:"
    ids = tokenizer.encode(prompt)[:context_tokens]
    prompt = tokenizer.decode(ids, skip_special_tokens=True)
    if not chat:
        return prompt
    body = prompt.rstrip()
    if body.endswith("Answer:"):
        body = body[: -len("Answer:")].rstrip()
    msg = [{"role": "user", "content": body + SUFFIX}]
    text = tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
    bos = tokenizer.bos_token or ""
    if bos and text.startswith(bos):
        text = text[len(bos):]
    return text


def seer_f1(pred: str, ref: str) -> float:
    """SEER's scorer: articles stripped BEFORE punctuation (not the SQuAD order)."""
    def norm(s):
        s = s.lower().strip()
        s = re.sub(r"\b(a|an|the)\b", " ", s)
        s = "".join(ch for ch in s if ch not in string.punctuation)
        return " ".join(s.split())
    return _f1(norm(pred).split(), norm(ref).split())


def longbench_f1(pred: str, refs: list[str], dataset: str) -> float:
    """Official LongBench qa_f1_score: SQuAD normalization order, max over references,
    first-line truncation for triviaqa."""
    if dataset in ("trec", "triviaqa", "samsum", "lsht"):
        pred = pred.lstrip("\n").split("\n")[0]
    def norm(s):
        s = s.lower()
        s = "".join(ch for ch in s if ch not in string.punctuation)
        s = re.sub(r"\b(a|an|the)\b", " ", s)
        return " ".join(s.split())
    return max(_f1(norm(pred).split(), norm(r).split()) for r in refs)


def _f1(p, r):
    if not p and not r:
        return 1.0
    if not p or not r:
        return 0.0
    c = sum((Counter(p) & Counter(r)).values())
    if c == 0:
        return 0.0
    prec, rec = c / len(p), c / len(r)
    return 2 * prec * rec / (prec + rec)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="/public/model_zoo/Qwen2.5-7B-Instruct")
    ap.add_argument("--longbench-dir", default="/public/data_zoo/longbench/data")
    ap.add_argument("--archived", default=str(ROOT / "experiments/results/expand/qwen25_7b"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dtypes", default="float16,bfloat16,float32")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--per-dataset", type=int, default=8, help="first-N rows, matching SEER ids 0..N-1")
    ap.add_argument("--context-tokens", type=int, default=4096)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos_ids = set()
    gen_eos = None
    rows_by_dataset = {}
    for ds in args.datasets.split(","):
        path = Path(args.longbench_dir) / f"{ds}.jsonl"
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        rows_by_dataset[ds] = rows[: args.per_dataset]
    archived = {}
    for ds in rows_by_dataset:
        p = Path(args.archived) / f"{ds}_full.json"
        if p.exists():
            archived[ds] = {r["id"]: r for r in json.loads(p.read_text())["results"]}
    prompts = {ds: [render_seer_prompt(r, tokenizer, args.context_tokens, True) for r in rows]
               for ds, rows in rows_by_dataset.items()}
    env = {"python": platform.python_version(), "torch": torch.__version__,
           "transformers": importlib.metadata.version("transformers"), "cuda": torch.version.cuda,
           "gpu": torch.cuda.get_device_name(args.device) if torch.cuda.is_available() else None,
           "co_tenancy_note": "GPU may be shared with other processes; timings are not reported"}
    for dtype_name in args.dtypes.split(","):
        dtype = getattr(torch, dtype_name)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dtype, attn_implementation="eager", local_files_only=True,
            trust_remote_code=False).to(args.device).eval()
        if gen_eos is None:
            eos = model.generation_config.eos_token_id
            eos_ids = set(eos if isinstance(eos, list) else [eos]) - {None}
            gen_eos = sorted(eos_ids)
        results = []
        t0 = time.time()
        for ds, plist in prompts.items():
            for i, prompt in enumerate(plist):
                ids = tokenizer(prompt, return_tensors="pt").input_ids.to(args.device)
                generated, nan_steps, token0_steps, partial_nan_steps = [], 0, 0, 0
                with torch.inference_mode():
                    out = model(ids, use_cache=True)
                    past = out.past_key_values
                    logits = out.logits[:, -1, :].float()
                    for step in range(args.max_new_tokens):
                        finite = torch.isfinite(logits)
                        if not bool(finite.all()):
                            if not bool(finite.any()):
                                nan_steps += 1
                            else:
                                partial_nan_steps += 1
                        nxt = int(logits.argmax(-1).item())
                        token0_steps += int(nxt == 0)
                        generated.append(nxt)
                        if nxt in eos_ids:
                            break
                        out = model(torch.tensor([[nxt]], device=args.device), past_key_values=past, use_cache=True)
                        past = out.past_key_values
                        logits = out.logits[:, -1, :].float()
                pred = tokenizer.decode(generated, skip_special_tokens=True).strip()
                refs = rows_by_dataset[ds][i].get("answers") or []
                arch = archived.get(ds, {}).get(i)
                rec = dict(dataset=ds, id=i, input_tokens=int(ids.shape[1]),
                           prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                           generated_token_ids=generated, n_gen_tokens=len(generated),
                           all_nan_logit_steps=nan_steps, partial_nan_logit_steps=partial_nan_steps,
                           token0_emissions=token0_steps, pred=pred,
                           exclamation_marks=pred.count("!"),
                           seer_f1_single_ref=seer_f1(pred, refs[0]) if refs else None,
                           longbench_f1_all_refs=longbench_f1(pred, refs, ds) if refs else None,
                           archived_pred=None if arch is None else arch["pred"],
                           archived_f1=None if arch is None else arch["f1"],
                           equals_archived=None if arch is None else (pred == arch["pred"]))
                results.append(rec)
                print(f"[{dtype_name}] {ds}/{i} allNaN={nan_steps} tok0={token0_steps} "
                      f"eq_archived={rec['equals_archived']} F1={rec['longbench_f1_all_refs']}", flush=True)
        summary = dict(dtype=dtype_name, n=len(results),
                       requests_with_all_nan_steps=sum(r["all_nan_logit_steps"] > 0 for r in results),
                       requests_with_token0=sum(r["token0_emissions"] > 0 for r in results),
                       total_all_nan_steps=sum(r["all_nan_logit_steps"] for r in results),
                       total_partial_nan_steps=sum(r["partial_nan_logit_steps"] for r in results),
                       total_token0_emissions=sum(r["token0_emissions"] for r in results),
                       requests_ge10_exclamations=sum(r["exclamation_marks"] >= 10 for r in results),
                       requests_equal_to_archived=sum(1 for r in results if r["equals_archived"]),
                       requests_with_archived=sum(1 for r in results if r["archived_pred"] is not None),
                       mean_longbench_f1=sum(r["longbench_f1_all_refs"] for r in results) / len(results),
                       wall_s=time.time() - t0)
        payload = dict(schema="qwen-fp16-diagnosis-v1", model=args.model, args=vars(args), env=env,
                       prompt_rendering="pinned SEER old_feat: local first-N LongBench rows, "
                                        "encode[:4096]->decode, chat template with concise-answer suffix",
                       eos_token_ids=gen_eos, summary=summary, results=results)
        (out_dir / f"diagnosis.{dtype_name}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps(summary), flush=True)
        del model, past, out, logits
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

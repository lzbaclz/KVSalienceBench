#!/usr/bin/env python3
"""Freeze local English-QA inputs for the NEW physical protocol (not historical SEER prompts)."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def render_prompt(row):
    if isinstance(row.get("prompt"), str):
        return row["prompt"]
    if not isinstance(row.get("context"), str) or not isinstance(row.get("input"), str):
        raise ValueError("source needs prompt, or context and input strings")
    return ("Read the passage and answer the question. Return only the answer.\n\n"
            f"Passage:\n{row['context']}\n\nQuestion: {row['input']}\nAnswer:")


def token_count(tokenizer, text, chat):
    if chat:
        return len(tokenizer.apply_chat_template([{"role": "user", "content": text}],
                    tokenize=True, add_generation_prompt=True))
    return len(tokenizer(text, add_special_tokens=True).input_ids)


def truncate_middle(tokenizer, text, cap, chat):
    """Explicit head/tail truncation; verify the re-encoded final prompt fits."""
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if token_count(tokenizer, text, chat) <= cap:
        return text
    size = min(len(ids)-1, cap)
    while size > 0:
        head = (size+1)//2; tail = size//2
        clipped = ids[:head] + (ids[-tail:] if tail else [])
        text = tokenizer.decode(clipped, skip_special_tokens=True)
        n = token_count(tokenizer, text, chat)
        if n <= cap:
            return text
        size -= max(1, n-cap)
    raise ValueError("cap cannot fit prompt/chat overhead")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="local tokenizer directory")
    p.add_argument("--input", action="append", required=True, metavar="DATASET=FILE.jsonl")
    p.add_argument("--out", required=True)
    p.add_argument("--per-dataset", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-input-tokens", type=int, default=4096)
    p.add_argument("--chat", action="store_true")
    p.add_argument("--truncate-middle", action="store_true", help="explicitly allow frozen head/tail truncation")
    args = p.parse_args(argv)
    if args.per_dataset < 1 or args.max_input_tokens < 1:
        raise ValueError("positive sample and token counts required")
    dest = Path(args.out)
    if dest.exists() or Path(str(dest)+".meta.json").exists():
        raise FileExistsError(dest)
    from transformers import AutoTokenizer
    from xqp.physical_validation import sha256_file, read_requests
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    output, provenance, datasets = [], [], set()
    for spec in args.input:
        dataset, sep, source = spec.partition("=")
        if not sep or not dataset or dataset in datasets:
            raise ValueError("each --input must have a unique DATASET=PATH")
        datasets.add(dataset)
        rows = [json.loads(x) for x in Path(source).read_text().splitlines() if x.strip()]
        if len(rows) < args.per_dataset:
            raise ValueError(f"{dataset}: requested {args.per_dataset}, only {len(rows)} rows")
        # Deterministic sampling of actual rows, not a repeated greedy-generation seed.
        selected = sorted(random.Random(args.seed).sample(range(len(rows)), args.per_dataset))
        source_hash = sha256_file(source)
        seen = set()
        for i in selected:
            row = rows[i]
            rid = str(row.get("_id", row.get("id", f"row-{i}")))
            if rid in seen:
                raise ValueError(f"duplicate source ID {dataset}/{rid}")
            seen.add(rid)
            text = render_prompt(row)
            original_hash = hashlib.sha256(text.encode()).hexdigest()
            original_tokens = token_count(tok, text, args.chat)
            if original_tokens > args.max_input_tokens:
                if not args.truncate_middle:
                    raise ValueError(f"{dataset}/{rid}: {original_tokens} tokens; enable explicit --truncate-middle or change cap")
                text = truncate_middle(tok, text, args.max_input_tokens, args.chat)
            answers = row.get("answers")
            if not isinstance(answers, list) or not answers or not all(isinstance(a, str) and a.strip() for a in answers):
                raise ValueError(f"{dataset}/{rid}: missing nonempty answers list")
            output.append(dict(id=rid, dataset=dataset, prompt=text, answers=answers))
            provenance.append(dict(dataset=dataset, id=rid, source_row=i,
                source_sha256=source_hash, original_prompt_sha256=original_hash,
                original_tokens=original_tokens, final_tokens=token_count(tok, text, args.chat),
                truncated=original_tokens > args.max_input_tokens))
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Outputs are new immutable inputs; neither existing cohorts nor raw data change.
    with dest.open("x") as f:
        for row in output:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
    read_requests(dest)
    with Path(str(dest)+".meta.json").open("x") as f:
        json.dump(dict(protocol="physical-kv-v1-new-prompts", args=vars(args),
                       data_sha256=sha256_file(dest), requests=provenance), f, indent=2)
        f.write("\n")
    print(f"{len(output)} requests -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Complete offline CPU example: tiny random Llama through the pinned simulator.

This checks execution and memory-patch parity, not benchmark task quality.
No downloaded model, dataset, GPU, or upstream repository access is needed.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    subprocess.run([sys.executable, str(ROOT / "scripts/prepare_seer.py")], check=True)
    sys.path.insert(0, str(ROOT / "build/seer-5a7fbee19045"))
    sys.path.insert(0, str(ROOT))
    import importlib
    import torch
    import transformers
    if transformers.__version__ != "4.51.3":
        raise SystemExit("this example requires transformers==4.51.3")
    from transformers import LlamaConfig, LlamaForCausalLM, BatchEncoding
    from seer.policy import build_policy
    from experiments.seer_pinned.run_cell import install_attention_memory_patch

    class Tokenizer:
        eos_token_id = 1
        bos_token = None
        pad_token = "<pad>"

        def __call__(self, text, **kwargs):
            ids = torch.tensor([[2 + ord(c) % 90 for c in text[:300]]])
            return BatchEncoding({"input_ids": ids, "attention_mask": torch.ones_like(ids)})

        def decode(self, ids, **kwargs):
            return " ".join(str(int(i)) for i in ids)

    torch.set_num_threads(1)
    runs = []
    for patch in (False, True):
        sim = importlib.reload(importlib.import_module("seer.eval.sim"))
        if patch:
            install_attention_memory_patch(sim)
        torch.manual_seed(0)
        config = LlamaConfig(vocab_size=97, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                             max_position_embeddings=512, attn_implementation="eager")
        model = LlamaForCausalLM(config).eval()
        runner = sim.MaskingSimulator(model=model, tokenizer=Tokenizer(), policy=build_policy("h2o"),
                                      budget_frac=.3, decision_period=2, block_size=8, io_mode="analytical")
        runner.attach()
        try:
            result = runner.run("the quick brown fox jumps over the lazy dog " * 12, max_new_tokens=6)
        finally:
            runner.detach()
        runs.append({key: result[key] for key in ("pred", "n_gen_tokens", "per_step_block_count", "per_step_eps_measured")})
    if runs[0] != runs[1] or runs[0]["n_gen_tokens"] < 2:
        raise SystemExit("FAIL: memory-patch parity or decode check")
    print(json.dumps({"status": "passed", "model": "random tiny Llama; execution check only",
                      "memory_patch_equal": True, "result": runs[0]}, indent=2))


if __name__ == "__main__":
    main()

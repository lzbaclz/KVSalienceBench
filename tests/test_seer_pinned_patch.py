"""The prefill-attention memory patch applied by experiments/seer_pinned/run_cell.py
must not change what the pinned SEER simulator computes.

Runs when KVSALIENCE_SEER_ROOT points at the verified export prepared from
the bundled source by scripts/prepare_seer.py; a tiny random Llama on CPU is enough because the
patch touches the hook return value, not the arithmetic.
"""
import os
from pathlib import Path
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

SEER_ROOT = os.environ.get("KVSALIENCE_SEER_ROOT", "")
pytestmark = pytest.mark.skipif(
    not SEER_ROOT or not (Path(SEER_ROOT) / "PINNED_GIT_SHA.txt").exists(),
    reason="set KVSALIENCE_SEER_ROOT to a pinned SEER export to run this test")


def _tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                      attn_implementation="eager")
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg).eval()


class _Tok:
    eos_token_id = 1
    bos_token = None
    pad_token = "<pad>"

    def __call__(self, text, return_tensors="pt", truncation=True, max_length=512):
        from transformers import BatchEncoding
        ids = torch.tensor([[2 + (ord(c) % 90) for c in text[:300]]])
        return BatchEncoding({"input_ids": ids, "attention_mask": torch.ones_like(ids)})

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(i)) for i in ids)


def _run(patched: bool):
    sys.path.insert(0, SEER_ROOT)
    import importlib
    sim = importlib.import_module("seer.eval.sim")
    importlib.reload(sim)
    from seer.policy import build_policy
    if patched:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/seer_pinned"))
        from run_cell import install_attention_memory_patch
        install_attention_memory_patch(sim)
    model = _tiny_llama()
    s = sim.MaskingSimulator(model=model, tokenizer=_Tok(), policy=build_policy("h2o"),
                             budget_frac=0.3, decision_period=2, block_size=8, io_mode="analytical")
    s.attach()
    try:
        out = s.run("the quick brown fox jumps over the lazy dog " * 12, max_new_tokens=6)
    finally:
        s.detach()
    return out


def test_memory_patch_is_numerically_neutral():
    a, b = _run(False), _run(True)
    assert a["pred"] == b["pred"]
    assert a["per_step_block_count"] == b["per_step_block_count"]
    assert np.allclose(a["per_step_eps_measured"], b["per_step_eps_measured"])
    assert a["n_gen_tokens"] == b["n_gen_tokens"] and a["n_gen_tokens"] >= 2

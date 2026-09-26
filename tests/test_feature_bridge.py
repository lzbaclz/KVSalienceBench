"""End-to-end CPU check of the offline/runtime feature bridge (PC item P1-1).

A random tiny Llama on CPU exercises the whole path: two trajectories per
request, snapshot recording inside the reference backend's hooks, offline-style
and runtime feature reconstruction, horizon labels, per-request and pooled AUC,
JSON output. It is a correctness fixture, not evidence about the 8B models.
"""
import json
from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytestmark = pytest.mark.skipif(transformers.__version__ != "4.51.3", reason="pinned runtime required")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import run_feature_bridge as fb  # noqa: E402


def _tiny_checkpoint(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    torch.set_num_threads(1); torch.manual_seed(13)
    cfg = transformers.LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                                   num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                   max_position_embeddings=256)
    cfg._attn_implementation = "eager"
    checkpoint = tmp_path / "model"
    transformers.LlamaForCausalLM(cfg).save_pretrained(checkpoint)
    vocab = {"[UNK]": 0, "[EOS]": 1, **{f"t{i}": i + 2 for i in range(62)}}
    raw = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]")); raw.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]", eos_token="[EOS]").save_pretrained(checkpoint)
    return checkpoint


def test_bridge_end_to_end_on_cpu(tmp_path):
    checkpoint = _tiny_checkpoint(tmp_path)
    data = tmp_path / "qa.jsonl"
    rows = [dict(id=str(i), dataset="d", prompt=" ".join(f"t{(j * 7 + i) % 62}" for j in range(40 + 6 * i)),
                 answers=["t1"]) for i in range(2)]
    data.write_text("".join(json.dumps(r) + "\n" for r in rows))
    weights = tmp_path / "w.json"
    weights.write_text(json.dumps(dict(weights=[34.7585, 3.3461, 0.0, 0.0], bias=-3.6048, per_layer=False)))
    out = tmp_path / "bridge.json"
    argv = ["--model", str(checkpoint), "--data", str(data), "--out", str(out), "--weights", str(weights),
            "--dtype", "float32", "--device", "cpu", "--budget", "0.5", "--steps", "12", "--horizon", "2",
            "--max-input-tokens", "256", "--block-size", "2", "--prefill-chunk", "8", "--observation-window", "4"]
    assert fb.main(argv) == 0
    res = json.loads(out.read_text())
    assert res["n_requests"] == 2 and len(res["per_request"]) == 2
    for r in res["per_request"]:
        assert r["rows_full"] > 0 and r["rows_policy"] > 0
        assert 0 < r["pos_rate_full"] < 1
        for k in ("A_offline_full", "B_runtime_full", "C_runtime_policy"):
            assert 0.0 <= r[k] <= 1.0
    s = res["summary"]
    for k in ("A_offline_full", "B_runtime_full", "C_runtime_policy"):
        assert 0.0 <= s[k]["mean_request_auc"] <= 1.0 and len(s[k]["ci95"]) == 2
        assert 0.0 <= s[k]["pooled_auc"] <= 1.0
    assert "B_minus_A_feature_realization" in s and "C_minus_B_trajectory" in s
    with pytest.raises(SystemExit, match="refusing"):
        fb.main(argv)

"""Offline CPU correctness tests; random tiny models are NOT quality evidence."""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytestmark = pytest.mark.skipif(transformers.__version__ != "4.51.3", reason="pinned physical validation runtime required")

from xqp.physical_kv import (EvictionConfig, PhysicalKVSession, compact_kv,
                              aligned_values, load_two_view_weights)
from xqp.physical_validation import correctness_gates, measure_request, read_requests, qa_f1


@pytest.fixture(params=["llama", "qwen2"])
def model(request):
    torch.set_num_threads(1)
    torch.manual_seed(19)
    cls = transformers.LlamaConfig if request.param == "llama" else transformers.Qwen2Config
    model_cls = transformers.LlamaForCausalLM if request.param == "llama" else transformers.Qwen2ForCausalLM
    cfg = cls(vocab_size=64, hidden_size=32, intermediate_size=64,
              num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
              max_position_embeddings=192, attention_dropout=0.)
    cfg._attn_implementation = "eager"
    return model_cls(cfg).eval()


@pytest.fixture
def config():
    return EvictionConfig(budget_fraction=.4, block_size=2, prefill_chunk=8,
                          observation_window=4)


@pytest.mark.parametrize("policy", ["h2o_block", "xqp_reconstructed", "full"])
def test_native_and_masked_replay_gates(model, config, policy):
    ids = torch.arange(21)[None, :]
    cfg = replace(config, policy=policy, budget_fraction=1 if policy == "full" else .4)
    gate = correctness_gates(model, ids, cfg, weights=(34.7585, 3.3461, -3.6048), steps=6)
    assert gate["passed"]
    assert max(c["max_abs_error"] for c in gate["masked_replay_checks"]) < 2e-5
    if policy != "full":
        assert gate["physical_memory"]["kv_storage_bytes"] < gate["masked_memory"]["kv_storage_bytes"]
    assert not hasattr(model, "_physical_kv_owner")
    assert all(not layer.self_attn._forward_hooks for layer in model.model.layers)


@pytest.mark.parametrize("backend", ["physical", "masked"])
def test_long_growth_no_resurrection_and_strict_budget(model, config, backend):
    cfg = replace(config, backend=backend)
    ids = torch.arange(21)[None, :]
    with PhysicalKVSession(model, cfg, 21) as session:
        logits = session.prefill(ids); session.evict()
        prev = [set(s.positions[s.eligible].tolist()) for s in session.states]
        for _ in range(35):
            absolute_new_position = session.logical_length
            logits = session.decode(logits.argmax(-1)[:, None]); session.evict(); session.assert_valid()
            for i, state in enumerate(session.states):
                active = set(state.positions[state.eligible].tolist())
                assert active <= prev[i] | {absolute_new_position}
                assert 0 in active and absolute_new_position in active
                assert len(active) <= session.budget_blocks * cfg.block_size
                prev[i] = active
        if backend == "physical":
            mem = session.memory()
            assert mem["kv_storage_bytes"] == mem["logical_live_kv_bytes"]
        else:
            assert session.cache.key_cache[0].shape[-2] == 56


def test_compaction_has_independent_storage():
    old_k = torch.randn(1, 2, 17, 8); old_v = torch.randn_like(old_k)
    indices = torch.tensor([0, 3, 6, 16])
    key, value = compact_kv(old_k, old_v, indices)
    expected_k, expected_v = key.clone(), value.clone()
    assert key.untyped_storage().data_ptr() != old_k.untyped_storage().data_ptr()
    assert key.untyped_storage().nbytes() == key.numel() * key.element_size()
    old_k.fill_(1e20); old_v.fill_(-1e20)
    torch.testing.assert_close(key, expected_k); torch.testing.assert_close(value, expected_v)


def test_alignment_uses_logical_ids():
    ids = torch.tensor([0, 4, 7]); values = torch.tensor([.1, .5, .9])
    result = aligned_values(ids, values, torch.tensor([0, 1, 7, 9]))
    torch.testing.assert_close(result, torch.tensor([.1, 0, .9, 0]))


@pytest.mark.parametrize("kwargs", [dict(budget_fraction=0), dict(budget_fraction=1.1),
    dict(block_size=0), dict(recent_blocks=0), dict(ema_decay=1), dict(policy="adakv"),
    dict(prefill_chunk=8, observation_window=9), dict(policy="full", budget_fraction=.5)])
def test_bad_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        EvictionConfig(**kwargs)


def test_impossible_budget_does_not_silently_expand(config):
    with pytest.raises(ValueError, match="cannot accommodate"):
        replace(config, budget_fraction=.1).budget(20)


def test_forced_resurrection_rejected_and_hooks_cleaned(model, config):
    with pytest.raises(ValueError, match="resurrection"):
        with PhysicalKVSession(model, config, 21) as session:
            session.prefill(torch.arange(21)[None, :]); session.evict()
            choices = session.schedule(); choices[0] = [0, 999]
            session.evict(choices)
    assert not hasattr(model, "_physical_kv_owner")
    for layer in model.model.layers:
        assert not layer.self_attn._forward_hooks
        assert not layer.self_attn._forward_pre_hooks


def test_nested_session_rejected(model, config):
    with PhysicalKVSession(model, config, 21):
        with pytest.raises(RuntimeError, match="concurrently"):
            with PhysicalKVSession(model, config, 21):
                pass


def test_missing_weights_and_unsupported_models_fail(model, config):
    with pytest.raises(ValueError, match="weights"):
        PhysicalKVSession(model, replace(config, policy="xqp_reconstructed"), 21)
    model.config._attn_implementation = "sdpa"
    with pytest.raises(ValueError, match="eager"):
        PhysicalKVSession(model, config, 21)


def test_nonfinite_attention_fails(model, config):
    with torch.no_grad():
        model.model.layers[0].self_attn.q_proj.weight.fill_(float("nan"))
    with PhysicalKVSession(model, config, 21) as session:
        session.prefill(torch.arange(21)[None, :]); session.evict()
        with pytest.raises(FloatingPointError):
            session.assert_valid()


def test_generation_first_token_eos_and_one_token_tpot(model, config):
    ids = torch.arange(21)[None, :]
    with torch.inference_mode():
        expected = int(model(ids).logits[:, -1, :].argmax(-1))
    result = measure_request(model, ids, config, max_new_tokens=8, eos_ids={expected})
    assert result["generated_token_ids"] == [expected]
    assert result["local_tpot_ms"] is None and result["local_itl_ms"] == []
    assert result["stopped_on_eos"]
    result = measure_request(model, ids, config, max_new_tokens=5, eos_ids={expected}, fixed_output=True)
    assert result["output_tokens"] == 5
    assert len(result["local_itl_ms"]) == 4
    assert result["allocator_decode"]["peak_allocated_bytes"] is None
    assert result["local_tpot_ms"] == pytest.approx(np.mean(result["local_itl_ms"]))


def test_weights_and_qa_metric(tmp_path):
    p = tmp_path / "weights.json"
    p.write_text(json.dumps(dict(weights=[1, 2, 0, 0], bias=-1)))
    assert load_two_view_weights(p) == (1, 2, -1)
    p.write_text(json.dumps(dict(weights=[1, 2, 1, 0], bias=-1)))
    with pytest.raises(ValueError): load_two_view_weights(p)
    assert qa_f1("The red fox.", ["red fox"]) == 1
    assert qa_f1("", ["red fox"]) == 0
    assert qa_f1("fox", ["red fox"]) == pytest.approx(2/3)


def test_requests_reject_duplicate_ids_and_missing_answers(tmp_path):
    p = tmp_path / "data.jsonl"
    row = dict(id="1", dataset="qa", prompt="What?", answers=["That"])
    p.write_text(json.dumps(row)+"\n"+json.dumps(row)+"\n")
    with pytest.raises(ValueError, match="duplicate"): read_requests(p)
    row.pop("answers"); p.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="answers"): read_requests(p)


def test_trace_v2_rope_alignment_first_token_and_future_horizons(tmp_path):
    from xqp.attn_trace_extract import extract_attention_traces, labels_for_horizons
    torch.set_num_threads(1); torch.manual_seed(3)
    cfg = transformers.LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
          num_hidden_layers=1, num_attention_heads=1, num_key_value_heads=1,
          max_position_embeddings=128)
    cfg._attn_implementation = "eager"
    model = transformers.LlamaForCausalLM(cfg).eval()
    ids = torch.arange(16)[None, :]
    with torch.inference_mode(): expected = int(model(ids).logits[:, -1, :].argmax(-1))
    observed = []
    def capture(module, args, output):
        if output[1] is not None and output[1].shape[-2] == 1:
            observed.append(output[1][0, 0, 0].detach().clone())
    hook = model.model.layers[0].self_attn.register_forward_hook(capture)
    p = tmp_path / "trace.jsonl"
    extract_attention_traces(model=model, input_ids=[ids], out_path=p, device="cpu",
                             block_size=1, max_new_tokens=2, query_variants=True)
    hook.remove()
    rows = [json.loads(x) for x in p.read_text().splitlines()]
    metadata = json.loads(Path(str(p)+".meta.json").read_text())
    assert metadata["trace_version"] == 2
    assert metadata["requests"][0]["forwarded_generated_token_ids"][0] == expected
    assert len(metadata["requests"][0]["forwarded_generated_token_ids"]) == 66
    logits = torch.tensor([r["f_query_dotmax"] for r in rows if r["step"] == 0])
    # For one head and block-size 1 the corrected scaled q.k must reproduce
    # the model's actual attention softmax. Pre-RoPE queries fail this test.
    torch.testing.assert_close(logits.softmax(0), observed[0], atol=1e-6, rtol=1e-5)
    with pytest.raises(ValueError, match="unavailable"):
        labels_for_horizons([np.array([.1, .9])], 0, 2, .1)
    with pytest.raises(FileExistsError):
        extract_attention_traces(model=model, input_ids=[ids], out_path=p, device="cpu")


def test_full_cli_offline_with_saved_tiny_model(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    from experiments.run_physical_kv import main
    torch.set_num_threads(1); torch.manual_seed(13)
    cfg = transformers.LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
          num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)
    cfg._attn_implementation = "eager"
    checkpoint = tmp_path / "model"
    transformers.LlamaForCausalLM(cfg).save_pretrained(checkpoint)
    vocab = {"[UNK]": 0, "[EOS]": 1, **{f"t{i}": i+2 for i in range(62)}}
    raw = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]")); raw.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]", eos_token="[EOS]").save_pretrained(checkpoint)
    data = tmp_path / "qa.jsonl"
    data.write_text(json.dumps(dict(id="0", dataset="synthetic-smoke", prompt=" ".join(f"t{i}" for i in range(24)), answers=["t1"])))
    out = tmp_path / "result.json"
    argv = ["--model", str(checkpoint), "--data", str(data), "--out", str(out),
            "--policy", "h2o_block", "--budget", ".4", "--block-size", "2", "--device", "cpu",
            "--dtype", "float32", "--max-new-tokens", "3", "--prefill-chunk", "8",
            "--observation-window", "4", "--warmup", "0", "--gate-steps", "2", "--hash-model-weights"]
    assert main(argv) == 0
    obj = json.loads(out.read_text())
    assert obj["status"] == "passed" and obj["gates"]["passed"]
    assert len(obj["results"]) == 1
    with pytest.raises(SystemExit, match="exists"): main(argv)
    bad = argv.copy(); bad[bad.index("--out")+1] = str(tmp_path / "bad.json")
    bad[bad.index("--device")+1] = "mps"
    assert main(bad) == 1
    assert not (tmp_path / "bad.json").exists()
    assert json.loads((tmp_path / "bad.json.failed.json").read_text())["status"] == "failed"


def test_gate_report_mode_records_every_step_instead_of_raising(model, config, monkeypatch):
    """PC audit item A4: a tolerance miss must be measurable, not hidden behind
    the first failing step, and must never be reported as a pass."""
    import xqp.physical_validation as pv
    ids = torch.arange(21)[None, :]
    cfg = replace(config, policy="h2o_block", budget_fraction=.4)
    monkeypatch.setattr(pv, "_tolerances", lambda dtype: (0.0, 0.0))     # impossible tolerance
    with pytest.raises(AssertionError, match="logit parity failed"):
        correctness_gates(model, ids, cfg, steps=3)
    report = correctness_gates(model, ids, cfg, steps=3, strict=False)
    assert report["passed"] is False and report["strict"] is False
    assert len(report["native_full_checks"]) == 4 and len(report["masked_replay_checks"]) == 4
    assert all("argmax_equal" in c for c in report["native_full_checks"] + report["masked_replay_checks"])
    assert not hasattr(model, "_physical_kv_owner")

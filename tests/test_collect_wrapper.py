"""Behaviour of the trace-collection wrapper after the PC audit (items A2, A3).

* A2: an unavailable workload loader must fail, never fall back to synthetic
  prompts under the requested workload's name; synthetic prompts are an
  explicit workload.
* A3: per-request version-2 parts keep their sidecars, an incomplete part is
  never resumed, a complete part is reused, and the aggregate trace is
  immutable with a manifest tying it to every part.
"""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import collect_traces_attn as cw  # noqa: E402


def test_unavailable_workload_fails_instead_of_substituting_synthetic(monkeypatch):
    monkeypatch.setitem(sys.modules, "seer", None)          # make `import seer...` fail
    with pytest.raises(SystemExit, match="refusing to substitute"):
        cw._load_prompts("mooncake", 2, 256)


def test_synthetic_workload_is_explicit():
    prompts = cw._load_prompts(cw.SYNTHETIC_WORKLOAD, 3, 256)
    assert len(prompts) == 3 and "secret password" in prompts[0]


def _fake_extract(model_path, prompts, out_path, *, request_id_start, **kw):
    """Stand-in for xqp.attn_trace_extract.extract_attention_traces: writes a
    part and its v2 sidecar exactly like the real collector does."""
    import hashlib
    rows = 3
    with open(out_path, "x") as f:
        for b in range(rows):
            f.write(json.dumps(dict(trace_version=2, request_id=f"p{request_id_start}", layer=0, step=0,
                                    block_idx=b)) + "\n")
    digest = hashlib.sha256(Path(out_path).read_bytes()).hexdigest()
    Path(str(out_path) + ".meta.json").write_text(json.dumps(dict(trace_version=2, rows=rows, trace_sha256=digest)))
    return rows


class _Tok:
    def __call__(self, p, return_tensors="pt"):
        class E: pass
        e = E()
        import torch
        e.input_ids = torch.zeros((1, 4), dtype=torch.long)
        return e


def _run(tmp_path, monkeypatch, calls):
    pytest.importorskip("torch")
    import types
    fake_tf = types.SimpleNamespace(
        AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=lambda *a, **k: types.SimpleNamespace(
            to=lambda d: types.SimpleNamespace(eval=lambda: object()))),
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: _Tok()))
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)
    fake_xqp = types.ModuleType("xqp.attn_trace_extract")

    def counting(*a, **k):
        calls.append(k["request_id_start"])
        return _fake_extract(*a, **k)
    fake_xqp.extract_attention_traces = counting
    monkeypatch.setitem(sys.modules, "xqp.attn_trace_extract", fake_xqp)
    monkeypatch.setattr(cw, "_free_gb", lambda p: 10_000.0)
    monkeypatch.setattr(cw, "ROOT", tmp_path)
    status = {"models": {}}
    return cw._collect_model("m", "/model", ["a", "b"], prompt_offset=0, out_path=tmp_path / "m.jsonl",
                             device="cpu", block_size=32, max_new_tokens=4, worker_id=None, status=status)


def test_parts_are_kept_resumed_and_manifested(tmp_path, monkeypatch):
    calls = []
    out = _run(tmp_path, monkeypatch, calls)
    assert calls == [0, 1] and out["rows"] == 6
    parts = tmp_path / "m.jsonl.parts"
    assert (parts / "p0.jsonl").exists() and (parts / "p0.jsonl.meta.json").exists()
    manifest = json.loads((tmp_path / "m.jsonl.manifest.json").read_text())
    assert manifest["rows"] == 6 and len(manifest["parts"]) == 2
    assert (tmp_path / "m.jsonl").read_text().count("\n") == 6
    # aggregate is immutable
    with pytest.raises(SystemExit, match="immutable"):
        _run(tmp_path, monkeypatch, [])
    # a second output with one complete and one corrupted part: only the corrupt one is redone
    (tmp_path / "m.jsonl").unlink(); (tmp_path / "m.jsonl.manifest.json").unlink()
    (parts / "p1.jsonl").write_text("corrupted\n")          # hash no longer matches its sidecar
    calls2 = []
    out2 = _run(tmp_path, monkeypatch, calls2)
    assert calls2 == [1] and out2["rows"] == 6


def test_incomplete_part_without_sidecar_is_not_resumed(tmp_path):
    part = tmp_path / "p0.jsonl"; part.write_text("{}\n")
    assert cw._complete_part_rows(part, Path(str(part) + ".meta.json")) is None

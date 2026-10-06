"""Protocol 2.0 of the benchmark: schema, evaluator, loaders, splits and runner.

The evaluator itself is checked against brute force in ``test_decision_eval.py``; here
the benchmark layer is checked: that the shipped example reproduces its recorded
results, that incomplete or malformed tables are refused, that the label-count check is
not mistaken for a completeness proof (a candidate manifest is), that illegal labels are
refused instead of coerced, that the trace loader and both splits behave as documented
(and the request split IS the paper's), and that the version-1 interface is kept but
marked legacy.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from benchmark import protocol as P  # noqa: E402
from benchmark import run_leaderboard  # noqa: E402
from benchmark.fit_reference import fit_reference, sample_rows, score_fn  # noqa: E402
from xqp.dm_metrics import (average_precision, precision_at_k,  # noqa: E402
                            precision_recall_at_k_grouped)

EXAMPLE = ROOT / "benchmark/example"


@pytest.fixture(scope="module")
def example():
    return P.read_table(EXAMPLE / "prediction_table.csv")


# ----------------------------------------------------------------------- the example
def test_example_table_reproduces_its_recorded_results(example):
    want = json.loads((EXAMPLE / "expected_results.json").read_text())
    got = P.evaluate_table(example, n_boot=200, seed=0, probabilistic=True)
    assert got["protocol"] == P.PROTOCOL_VERSION == want["protocol"]
    for k in ("n_rows", "n_decisions", "n_requests", "n_source_prompts"):
        assert got[k] == want[k]
    for key in ("auc", "ap", "p_at_k"):
        assert got["pooled"][key] == pytest.approx(want["pooled"][key], abs=1e-9)
    for key in want["pairs"]:
        assert got["pairs"][key] == pytest.approx(want["pairs"][key], abs=1e-9)
    for r in ("0.10", "0.20"):
        for key in ("macro_recall", "macro_recall_random_ties", "boundary_tie_share"):
            assert got["per_decision"][r][key] == pytest.approx(want["per_decision"][r][key], abs=1e-9)
    for key in ("ece", "brier"):
        assert got["calibration"][key] == pytest.approx(want["calibration"][key], abs=1e-9)
    # bootstrap draws come from numpy's generator, whose stream is not frozen across versions
    assert got["ci95"]["unit"] == want["ci95"]["unit"] == "source_prompt"
    for key, interval in want["ci95"].items():
        if isinstance(interval, list):
            assert got["ci95"][key][0] == pytest.approx(interval[0], abs=0.03)
            assert got["ci95"][key][1] == pytest.approx(interval[1], abs=0.03)


def test_example_is_synthetic_and_complete(example):
    assert set(example["source_id"].tolist()) == {f"synthetic/q{i:02d}" for i in range(8)}
    res = P.evaluate_table(example, n_boot=0)
    assert res["label_count_consistent"] is True and res["n_decisions"] == 192
    # without a manifest the evaluator claims nothing about candidate completeness
    assert res["candidates"]["verified"] is None and "complete_decisions" not in res
    shipped = P.read_manifest(EXAMPLE / "candidate_manifest.json")
    assert shipped == P.candidate_manifest(example) and shipped["n_decisions"] == 192 and shipped["n_rows"] == 5748
    assert P.evaluate_table(example, n_boot=0, manifest=shipped)["candidates"]["verified"] is True
    assert (ROOT / "benchmark/example/make_example_table.py").read_text().count("SYNTHETIC") >= 1


def test_evaluator_agrees_with_the_reference_metric_functions(example):
    res = P.evaluate_table(example, n_boot=0)
    y, s = example["label"].astype(np.float32), example["score"]
    dec = (np.unique(example["model_id"], return_inverse=True)[1].astype(np.int64) * 1_000_000
           + example["request_id"] * 10_000 + example["layer"] * 100 + example["step"])
    assert res["pooled"]["ap"] == pytest.approx(average_precision(y, s))
    assert res["pooled"]["p_at_k"] == pytest.approx(precision_at_k(y, s, P.TOP_R))
    for r in P.RETENTIONS:
        ref = precision_recall_at_k_grouped(y, s, dec, r)
        assert res["per_decision"][f"{r:.2f}"]["macro_recall"] == pytest.approx(ref["macro_recall"], abs=1e-12)


# ---------------------------------------------------------------------- validation
def test_incomplete_decisions_are_rejected_unless_asked(example):
    keep = np.random.default_rng(0).random(len(example["label"])) < 0.5
    sample = {k: v[keep] for k, v in example.items()}
    with pytest.raises(ValueError, match="complete decisions"):
        P.evaluate_table(sample, n_boot=0)
    res = P.evaluate_table(sample, n_boot=0, strict=False)
    assert res["label_count_consistent"] is False
    assert P.evaluate_table(sample, n_boot=0, label_rate=None)["label_count_consistent"] is None


def one_decision(n: int = 20, positives: int = 2) -> dict:
    """A single decision of ``n`` candidates whose first ``positives`` blocks are positive."""
    label = np.zeros(n, np.int8); label[:positives] = 1
    return dict(model_id=np.array(["m"] * n), source_id=np.array(["s"] * n), request_id=np.zeros(n, np.int64),
                layer=np.zeros(n, np.int64), step=np.zeros(n, np.int64), block_idx=np.arange(n, dtype=np.int64),
                label=label, score=np.linspace(1.0, 0.0, n))


def test_label_count_check_is_not_a_completeness_proof():
    """Drop one NEGATIVE from a decision of 20 candidates with 2 positives: 19 rows still
    hold ceil(0.1 * 19) = 2 positives, so the label-count check passes although a
    candidate is missing. Only a candidate manifest catches it."""
    full = one_decision(20, 2)
    assert int(np.ceil(0.1 * 20)) == int(np.ceil(0.1 * 19)) == 2
    manifest = P.candidate_manifest(full)
    assert manifest["decisions"]["m|0|0|0"][0] == 20
    assert P.evaluate_table(full, n_boot=0, manifest=manifest)["candidates"]["verified"] is True

    keep = np.ones(20, bool); keep[7] = False                 # block 7 is a negative
    cut = {k: v[keep] for k, v in full.items()}
    res = P.evaluate_table(cut, n_boot=0)                      # strict, and it still passes
    assert res["label_count_consistent"] is True and res["candidates"]["verified"] is None
    with pytest.raises(ValueError, match="candidate check failed"):
        P.evaluate_table(cut, n_boot=0, manifest=manifest)
    report = P.evaluate_table(cut, n_boot=0, manifest=manifest, strict=False)["candidates"]
    assert report["verified"] is False and report["decisions_with_wrong_candidate_count"] == 1

    # same count, another block: caught by the digest
    swapped = dict(full, block_idx=full["block_idx"].copy()); swapped["block_idx"][7] = 99
    report = P.evaluate_table(swapped, n_boot=0, manifest=manifest, strict=False)["candidates"]
    assert report["verified"] is False and report["decisions_with_wrong_candidate_ids"] == 1
    # a whole decision missing, or one the manifest does not know
    other = dict(full, step=np.ones(20, np.int64))
    report = P.evaluate_table(other, n_boot=0, manifest=manifest, strict=False)["candidates"]
    assert report["missing_decisions"] == 1 and report["unexpected_decisions"] == 1
    with pytest.raises(ValueError, match="not a candidate manifest"):
        P.evaluate_table(full, n_boot=0, manifest={"decisions": {}})


@pytest.mark.parametrize("bad", [2, -1, 0.5, float("nan"), float("inf")])
def test_illegal_labels_are_refused_not_coerced(bad):
    """``label > 0`` used to score a 2 as a positive and a -1 or NaN as a negative."""
    table = one_decision(20, 2)
    table["label"] = table["label"].astype(np.float64)
    table["label"][5] = bad
    with pytest.raises(ValueError, match="labels must be exactly 0 or 1|non-finite labels"):
        P.evaluate_table(table, n_boot=0, strict=False)


def test_legal_label_encodings_are_accepted():
    table = one_decision(20, 2)
    want = P.evaluate_table(table, n_boot=0)["pooled"]["auc"]
    for cast in (bool, np.float32, np.int64):
        assert P.evaluate_table(dict(table, label=table["label"].astype(cast)), n_boot=0)["pooled"]["auc"] == want
    with pytest.raises(ValueError, match="numeric"):
        P.evaluate_table(dict(table, label=np.array(["yes"] * 20)), n_boot=0)
    with pytest.raises(ValueError, match="equally long"):
        P.evaluate_table(dict(table, label=table["label"][:-1]), n_boot=0)


def test_manifest_roundtrip_and_runner(tmp_path, capsys, example):
    for name in ("m.json", "m.json.gz"):
        P.write_manifest(tmp_path / name, P.candidate_manifest(example))
        assert P.read_manifest(tmp_path / name) == P.candidate_manifest(example)
    (tmp_path / "x.json").write_text("{}")
    with pytest.raises(ValueError, match="not a candidate manifest"):
        P.read_manifest(tmp_path / "x.json")
    table = str(EXAMPLE / "prediction_table.csv")
    run_leaderboard.main(["--table", table, "--write-manifest", str(tmp_path / "w.json")])
    assert P.read_manifest(tmp_path / "w.json") == P.read_manifest(EXAMPLE / "candidate_manifest.json")
    capsys.readouterr()
    run_leaderboard.main(["--table", table, "--n-boot", "0", "--manifest", str(tmp_path / "w.json")])
    assert "candidate sets verified against the manifest" in capsys.readouterr().out


def test_malformed_tables_are_refused(example):
    short = {k: v for k, v in example.items() if k != "score"}
    with pytest.raises(ValueError, match="lacks column"):
        P.evaluate_table(short, n_boot=0)
    bad = dict(example, score=example["score"].copy()); bad["score"][0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        P.evaluate_table(bad, n_boot=0)
    out_of_range = dict(example, score=example["score"] * 3.0)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        P.evaluate_table(out_of_range, n_boot=0, probabilistic=True)
    dup = {k: np.concatenate([v, v[:1]]) for k, v in example.items()}
    with pytest.raises(ValueError):
        P.evaluate_table(dup, n_boot=0, strict=False)
    two_sources = dict(example, source_id=example["source_id"].copy())
    two_sources["source_id"][0] = "synthetic/other"
    with pytest.raises(ValueError, match="more than one source_id"):
        P.evaluate_table(two_sources, n_boot=0)


def test_calibration_only_when_declared_probabilistic(example):
    assert "calibration" not in P.evaluate_table(example, n_boot=0)
    assert set(P.evaluate_table(example, n_boot=0, probabilistic=True)["calibration"]) == {"ece", "brier"}


def test_table_roundtrip(tmp_path, example):
    for name in ("t.csv", "t.csv.gz"):
        P.write_table(tmp_path / name, example)
        back = P.read_table(tmp_path / name)
        for k in P.TABLE_COLUMNS:
            if k == "score":
                assert np.allclose(back[k], example[k], rtol=1e-7)
            else:
                assert (back[k] == example[k]).all(), k
    with open(tmp_path / "bad.csv", "w") as fh:
        fh.write("model_id,score\na,0.1\n")
    with pytest.raises(ValueError, match="missing columns"):
        P.read_table(tmp_path / "bad.csv")


# ---------------------------------------------------- loaders, splits, the runner
def write_trace(path: Path, n_req: int, seed: int, layers=2, steps=3):
    """A tiny version-2 trace with complete decisions, plus its manifests."""
    rng = np.random.default_rng(seed)
    lines, rows = [], 0
    for q in range(n_req):
        for layer in range(layers):
            for step in range(steps):
                n = int(rng.integers(12, 25))
                latent = rng.lognormal(size=n)
                k = int(np.ceil(0.1 * n))
                lab = np.zeros(n, int); lab[np.argsort(-latent)[:k]] = 1
                for b in range(n):
                    rec = dict(trace_version=2, request_id=f"p{q}", layer=layer, step=step, block_idx=b,
                               f_within=float(latent[b] / latent.max()), f_cross=float(rng.random() < 0.1),
                               f_query=float(rng.random()), f_pos=float(rng.random()),
                               **{f"y_{h}": int(lab[b]) for h in P.HORIZONS})
                    lines.append(json.dumps(rec)); rows += 1
    path.write_text("\n".join(lines) + "\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    Path(str(path) + ".meta.json").write_text(json.dumps(dict(
        trace_version=2, rows=rows, trace_sha256=digest, collector_sha256="0" * 64,
        requests=[f"p{q}" for q in range(n_req)])))
    Path(str(path) + ".cohort.json").write_text(json.dumps(dict(
        ids=[dict(trace_id=f"p{q}", dataset="ds" if q % 2 == 0 else "ds2", id=f"id{q:02d}") for q in range(n_req)],
        data_sha256="1" * 64, model_weight_sha256="2" * 64)))
    return rows


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    d = tmp_path_factory.mktemp("traces")
    write_trace(d / "a.jsonl", 12, seed=1)
    write_trace(d / "b.jsonl", 12, seed=2)
    return P.load_corpus([f"model-a={d / 'a.jsonl'}", f"model-b={d / 'b.jsonl'}"], verify_hash=True)


def test_trace_loader_roundtrip(corpus, tmp_path):
    assert corpus["n_requests"] == 24 and len(corpus["source_names"]) == 12
    assert corpus["F"].shape[1] == 4 and set(np.unique(corpus["y"])) <= {0, 1}
    assert corpus["request"].min() == 0 and corpus["request"].max() == 23     # running offset, as run_icdm_v2.pool
    # a corrupted trace must not load when the hash is verified
    bad = tmp_path / "bad.jsonl"
    write_trace(bad, 3, seed=3)
    bad.write_text(bad.read_text() + "\n")
    with pytest.raises(ValueError, match="sha256"):
        P.load_trace(bad, verify_hash=True)


def test_source_split_holds_out_the_same_prompts_for_every_model(corpus):
    tr, te = P.split(corpus, "source")
    held = set(corpus["request_source"][corpus["request"][te]].tolist())
    seen = set(corpus["request_source"][corpus["request"][tr]].tolist())
    assert held and not (held & seen), "a source prompt appears on both sides of the split"
    models = set(corpus["model"][te].tolist())
    assert models == {0, 1}, "both models contribute test rows"
    assert len(tr) + len(te) == len(corpus["y"])


@pytest.mark.parametrize("kind", ["request", "source"])
def test_splits_reproduce_run_icdm_v2(corpus, kind):
    """The benchmark's splits are the paper's: same RNG calls, same held-out requests."""
    from run_icdm_v2 import request_split_v2
    names = [tuple(s.split("/", 1)) for s in corpus["source_names"]]
    d = dict(rid=corpus["request"], source=[names[c] for c in corpus["request_source"]])
    _, _, held = request_split_v2(d, kind, seed=P.SEED)
    _, te = P.split(corpus, kind)
    assert sorted(held) == sorted(np.unique(corpus["request"][te]).tolist())


def test_evaluate_scores_a_function_on_the_held_out_part(corpus):
    res = P.evaluate(lambda F: F[:, 0], corpus, "source", n_boot=100)
    assert res["protocol"] == P.PROTOCOL_VERSION and res["split"] == "source"
    assert res["label_count_consistent"] is True
    # the result says what was scored: split, horizon, held-out prompts and traces
    cfg = res["config"]
    assert cfg["split"] == "source" and cfg["horizon"] == P.HEADLINE_H and cfg["label_rate"] == P.TOP_R
    assert cfg["retentions"] == list(P.RETENTIONS) and cfg["models"] == ["model-a", "model-b"]
    _, te = P.split(corpus, "source")
    held = sorted({corpus["source_names"][c] for c in corpus["request_source"][np.unique(corpus["request"][te])]})
    assert cfg["held_out_source_prompts"] == held and cfg["held_out_requests"] == 2 * len(held)
    assert len(cfg["traces"]) == 2 and any("fit without" in item for item in res["not_checked"])
    assert 0.5 < res["pooled"]["auc"] <= 1.0 and 0.0 <= res["per_decision"]["0.10"]["macro_recall"] <= 1.0
    with pytest.raises(ValueError, match="one score per row|scores for"):
        P.evaluate(lambda F: F[:5, 0], corpus)


def test_reference_baseline_is_trained_on_training_rows_only(corpus):
    tr, te = P.split(corpus, "source")
    ref = fit_reference(corpus["F"], corpus["y"], tr, rows=500)
    assert float(ref.weights[2]) == 0.0 and float(ref.weights[3]) == 0.0, "only within+cross are fit"
    res = P.evaluate(score_fn(ref), corpus, "source", n_boot=0, probabilistic=True)
    assert "calibration" in res
    assert sample_rows(np.arange(100), 10).tolist() == np.arange(100)[
        np.random.default_rng(P.SEED).choice(100, size=10, replace=False)].tolist()


def test_frozen_reference_model_scores_like_its_formula():
    name, fn = P.reference_scorer()
    d = json.loads((ROOT / "benchmark/reference_model_v2.json").read_text())
    F = np.array([[0.3, 1.0, 0.7, 0.2], [0.0, 0.0, 0.1, 0.9]], np.float32)
    z = F[:, 0] * d["weights"]["s_within"] + F[:, 1] * d["weights"]["s_cross"] + d["bias"]
    assert fn(F) == pytest.approx(1.0 / (1.0 + np.exp(-z)), rel=1e-5)
    assert d["protocol"] == P.PROTOCOL_VERSION and d["used_features"] == ["s_within", "s_cross"]
    assert d["weights"]["s_query"] == 0.0 and d["weights"]["s_pos"] == 0.0
    # the frozen coefficients are the ones the paper's decomposition record fitted
    rec = json.loads((ROOT / "experiments/results/icdm_v2_decomposition.json").read_text())["two_view"]
    assert d["weights"]["s_within"] == pytest.approx(rec["w_within"]) and d["bias"] == pytest.approx(rec["bias"])


def test_runner_scores_a_table(capsys):
    run_leaderboard.main(["--table", str(EXAMPLE / "prediction_table.csv"), "--n-boot", "50", "--probabilistic"])
    out = capsys.readouterr().out
    assert "prediction_table" in out and "AUC same-dec" in out and "ECE" in out
    # the header says what could not be verified from a table alone
    assert "candidate sets not verified (no manifest)" in out and "not recorded in a table" in out
    assert "not a closed leaderboard" in " ".join(run_leaderboard.__doc__.split())
    assert "not a closed leaderboard" in " ".join(P.__doc__.split())


def test_split_manifest_is_the_recorded_split():
    """benchmark/splits/paper_v2_splits.json is rebuilt from the tracked cohort files and
    equals the held-out sets recorded by the paper's analysis."""
    from benchmark import make_split_manifest as M
    shipped = json.loads((ROOT / "benchmark/splits/paper_v2_splits.json").read_text())
    assert shipped == M.build() and shipped["schema"] == M.SCHEMA
    recorded = json.loads((ROOT / "experiments/results/icdm_v2.json").read_text())
    for kind in ("source", "request"):
        got = [r["request"] for r in shipped["splits"][kind]["held_out_requests"]]
        assert got == sorted(recorded[f"pooled_{kind}_split"]["held_out_requests"])
    src, req = shipped["splits"]["source"], shipped["splits"]["request"]
    assert (src["n_held_out_requests"], src["n_held_out_source_prompts"],
            src["held_out_source_prompts_also_in_training"]) == (64, 32, 0)
    # the request split is transfer to new model-requests, not to unseen text
    assert (req["n_held_out_requests"], req["n_held_out_source_prompts"],
            req["source_prompts_with_every_model_held_out"],
            req["held_out_source_prompts_also_in_training"]) == (64, 57, 7, 50)
    assert req["held_out_requests_by_model"] == {"Llama-3.1-8B-Instruct": 30, "Qwen2.5-7B-Instruct": 34}


# ------------------------------------------------------------------------- legacy
def test_version_one_is_kept_but_marked_legacy():
    from benchmark.legacy_v1 import protocol as legacy
    assert legacy.PROTOCOL_VERSION == "kvsaliencebench-1.0" and P.PROTOCOL_VERSION == "kvsaliencebench-2.0"
    assert "LEGACY" in legacy.__doc__ and "superseded" in legacy.__doc__
    assert "pooled metrics only" in legacy.__doc__.lower()
    # its reference model is the archived one, not the version-2 refit
    old = json.loads((ROOT / "benchmark/legacy_v1/reference_model.json").read_text())
    assert old["protocol"] == "kvsaliencebench-1.0"

"""docs/FROZEN_CONFIG.md names every constant behind the version-2 evidence in one place.

The table is prose, so it can drift from the code. These tests check it against the code
defaults (function signatures and behaviour) and against the result records, including the
two rules a reader most often has to guess: what the cross-layer view is at layer 0, and
what the masked loop keeps when its budget is below its mandatory set.
"""
from __future__ import annotations

import inspect
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DOC = " ".join((ROOT / "docs/FROZEN_CONFIG.md").read_text().split())
RESULTS = ROOT / "experiments/results"


def load(rel):
    return json.loads((RESULTS / rel).read_text())


def test_logistic_fit_constants():
    from xqp.predictor import ClosedFormXQP
    fit = inspect.signature(ClosedFormXQP._fit_single).parameters
    assert fit["l2"].default == 1e-3 and fit["max_iter"].default == 40 and fit["tol"].default == 1e-6
    assert fit["class_weight"].default is None
    assert inspect.signature(ClosedFormXQP.from_fit).parameters["l2"].default == 1e-3
    for fragment in ("`l2 = 1e-3` on the two coefficients", "the intercept is not penalized",
                     "after 40 iterations", "by less than `1e-6`", "No class weights",
                     "Features are not standardized"):
        assert fragment in DOC, fragment
    # behaviour: the ridge shrinks the coefficients and leaves the intercept alone
    rng = np.random.default_rng(0)
    F = np.zeros((4000, 4), np.float32)
    F[:, 0] = rng.random(4000)
    y = (rng.random(4000) < 0.1 + 0.3 * F[:, 0]).astype(np.float32)
    w, b = ClosedFormXQP._fit_single(F, y, l2=1e9)
    assert np.abs(w).max() < 1e-3
    assert b == pytest.approx(math.log(y.mean() / (1 - y.mean())), abs=1e-3)


def test_feature_constants_and_the_layer0_rule():
    from xqp.attn_trace_extract import extract_attention_traces, update_ema
    from xqp.features import extract_features
    p = inspect.signature(extract_features).parameters
    assert p["r_cross"].default == 0.10 and p["w_recency"].default == 64.0
    assert p["cross_signal"].default == "indicator"
    collector = inspect.signature(extract_attention_traces).parameters
    assert collector["block_size"].default == 32 and collector["r_label"].default == 0.10
    assert collector["ema_decay"].default == 0.9 and collector["horizons"].default == (1, 4, 16, 64)
    assert inspect.signature(update_ema).parameters["decay"].default == 0.9
    first = np.array([0.2, 0.7, 0.1], np.float32)
    assert np.array_equal(update_ema(None, first), first), "the EMA starts at the first decode value"
    assert update_ema(first, np.array([1.0, 1.0, 1.0, 0.4], np.float32))[3] == pytest.approx(0.4), \
        "a block created later starts at its first value"

    rng = np.random.default_rng(1)
    n = 37
    ema = rng.random(n).astype(np.float32)
    kw = dict(K_layer=rng.normal(size=(n, 8)).astype(np.float32), q_prev=rng.normal(size=8).astype(np.float32),
              step=10, last_used=np.r_[np.zeros(n - 1), 6.0].astype(np.float32))
    layer0 = extract_features(ema_within=ema, ema_prev_layer=None, **kw)
    k = math.ceil(0.10 * n)
    # layer 0: the cross view is the BINARY top-ceil(0.1 n) indicator of the layer's own EMA
    assert set(np.unique(layer0[:, 1]).tolist()) == {0.0, 1.0} and layer0[:, 1].sum() == k
    assert set(np.flatnonzero(layer0[:, 1]).tolist()) == set(np.argsort(-ema)[:k].tolist())
    prev = rng.random(n).astype(np.float32)
    deeper = extract_features(ema_within=ema, ema_prev_layer=prev, **kw)
    assert set(np.flatnonzero(deeper[:, 1]).tolist()) == set(np.argsort(-prev)[:k].tolist())
    assert deeper[:, 0].max() == pytest.approx(1.0, abs=1e-6)                       # within view: EMA / max
    assert deeper[0, 3] == pytest.approx(math.exp(-10 / 64)) and deeper[-1, 3] == pytest.approx(math.exp(-4 / 64))
    assert np.all((deeper[:, 2] >= 0) & (deeper[:, 2] <= 1))                        # (1 + cos) / 2
    for fragment in ("decay 0.9", "`r_cross = 0.10`, `cross_signal = \"indicator\"`", "`w = 64` steps",
                     "Layer 0 has no predecessor and takes the same indicator from its own EMA",
                     "Always binary, never the continuous magnitude", "0 for every prompt block"):
        assert fragment in DOC, fragment


def test_record_constants():
    v2 = load("icdm_v2.json")
    cfg = v2["config"]
    assert (cfg["TRAIN_N"], cfg["TEST_N"], cfg["MI_N"], cfg["seed"], cfg["headline_horizon"]) == \
        (120000, 150000, 400000, 0, "h4")
    assert cfg["frozen_checkpoint_sha256"].startswith("4696ed039d78") and "`4696ed039d78...`" in DOC
    rows = {name: p["rows"] for name, p in v2["provenance"].items()}
    assert f"{v2['pooled_summary']['n_rows']:,} ({rows['Llama-3.1-8B-Instruct']:,} Llama, " \
           f"{rows['Qwen2.5-7B-Instruct']:,} Qwen), 256 model-requests" in DOC
    for prov in v2["provenance"].values():
        assert (prov["feature_steps"], prov["lookahead_steps"], prov["block_size"], prov["r_label"],
                prov["ema_decay"]) == (32, 64, 32, 0.1, 0.9)
    assert "32 feature steps, then 64 further steps that only supply labels" in DOC
    req, src = load("icdm_v2_decomposition.json"), load("icdm_v2_decomposition_source.json")
    for rec, text in ((req, "97.67 (within), 2.465 (cross), bias -3.227 on the request split"),
                      (src, "88.08, 2.492, -3.213 on the source-disjoint split")):
        tv = rec["two_view"]
        for value in (tv["w_within"], tv["w_cross"], tv["bias"]):
            assert f"{value:.4g}" in text.replace(",", " ").split() or f"{value:.3f}" in text, (value, text)
        assert text in DOC
    assert req["config"]["gbdt"] == "LGBMClassifier(max_depth=3, n_estimators=150, class_weight=balanced)"
    assert "`LGBMClassifier(max_depth=3, n_estimators=150, class_weight=\"balanced\", random_state=0)`" in DOC
    assert f"{req['data']['n_rows_heldout']:,} rows and {req['data']['n_decisions']:,} decisions (request split); " \
           f"{src['data']['n_rows_heldout']:,} and {src['data']['n_decisions']:,} (source-disjoint)" in DOC
    size = req["data"]["decision_size"]
    assert (size["min"], size["median"], size["max"]) == (64, 129.0, 129)
    assert "64 to 129 candidates per held-out decision, median 129" in DOC
    train = load("icdm_v2_train_size.json")
    assert f"{train['n_train_rows_available']:,} on the request split" in DOC and train["config"]["N_BOOT"] == 250
    archived = json.loads((ROOT / "experiments/predictors/xqp_closed_2view_h4.json").read_text())
    assert [f"{archived['weights'][0]:.2f}", f"{archived['weights'][1]:.2f}", f"{archived['bias']:.2f}"] == \
        ["34.76", "3.35", "-3.60"] and "Weights 34.76 (within), 3.35 (cross), bias -3.60" in DOC
    capacity = load("icdm_v2_gbdt_capacity.json")["config"]["lightgbm_configs"]
    assert len(capacity) == 6 and "63 leaves with unbounded depth / 500 trees" in DOC
    physical = json.loads((RESULTS / "physical_kv/llama3.1-v1/summary.json").read_text())
    assert physical["truncated_prompts"] == 104 and "104 of the 128 prompts are truncated" in DOC


def test_evaluator_constants_and_split_facts():
    from benchmark import protocol as P
    import run_icdm_full
    assert (P.SEED, P.TEST_FRAC, P.TOP_R, P.RETENTIONS, P.BLOCK_SIZE, P.HEADLINE_H) == \
        (0, 0.25, 0.10, (0.10, 0.20), 32, "h4")
    assert run_icdm_full.SEED == 0 and run_icdm_full.N_BOOT == 250
    splits = json.loads((ROOT / "benchmark/splits/paper_v2_splits.json").read_text())["splits"]
    req, src = splits["request"], splits["source"]
    assert "64 held out (30 Llama, 34 Qwen) from 57 source prompts, 50 of which also occur in training" in DOC
    assert (req["n_held_out_requests"], req["n_held_out_source_prompts"],
            req["held_out_source_prompts_also_in_training"]) == (64, 57, 50)
    assert list(req["held_out_requests_by_model"].values()) == [30, 34]
    assert (src["n_held_out_source_prompts"], src["n_held_out_requests"]) == (32, 64)
    assert "25% of the 128 source prompts (32); both models' requests held out (64)" in DOC


SEER_ROOT = os.environ.get("KVSALIENCE_SEER_ROOT", "")


@pytest.mark.skipif(not SEER_ROOT or not (Path(SEER_ROOT) / "PINNED_GIT_SHA.txt").exists(),
                    reason="set KVSALIENCE_SEER_ROOT to a pinned SEER export to run this test")
def test_runtime_constants_of_the_pinned_simulator():
    """Table I and section 6 of the doc: the masked loop's constants and its rule for a budget
    below the mandatory set, read from the simulator revision the rerun used."""
    sys.path.insert(0, SEER_ROOT)
    from seer.eval.sim import MaskingSimulator
    from seer.lap.features import HISTORY_N
    from seer.policy.baselines import H2OPolicy
    from seer.policy.xqp import XQPPolicy
    from seer.trace.schema import BLOCK_SIZE, MIN_TOP_K, TOP_K_FRACTION, compute_top_k
    p = inspect.signature(XQPPolicy.__init__).parameters
    assert (p["sink"].default, p["window"].default, p["r_cross"].default, p["w_recency"].default,
            p["ema_decay"].default, p["history_k"].default) == (4, 4, 0.10, 64.0, 0.9, 8)
    assert inspect.signature(H2OPolicy.__init__).parameters["hh_frac"].default == 0.5
    assert inspect.signature(MaskingSimulator.__init__).parameters["decision_period"].default == 8
    assert (BLOCK_SIZE, MIN_TOP_K, TOP_K_FRACTION, HISTORY_N) == (32, 32, 0.10, 32)
    # served oracle: K = min(n, max(32, floor(0.1 n)))
    assert [compute_top_k(n) for n in (20, 129, 320, 512)] == [32, 32, 32, 51]
    assert "`K = min(n, max(32, floor(0.1 n)))`" in DOC

    def stats(n):
        return {b: dict(attn_history=[0.01 * (b + 1)], attn_score_now=0.01 * (b + 1), position=32 * b,
                        steps_since_top_k=3, layer_scalar=0.5, io_cost=0.0) for b in range(n)}
    policy = XQPPolicy(weights=[1.0, 1.0, 0.0, 0.0])
    # budget 7 < 4 sink + 4 recent: the recent blocks are kept (newest first), then sinks
    assert policy.select_to_keep(stats(35), budget=7, step=0) == {31, 32, 33, 34, 0, 1, 2}
    assert policy.select_to_keep(stats(35), budget=3, step=0) == {32, 33, 34}
    # an ordinary budget contains the mandatory blocks; it is not enlarged by them
    kept = policy.select_to_keep(stats(129), budget=26, step=0)
    assert len(kept) == 26 and {0, 1, 2, 3, 125, 126, 127, 128} <= kept
    assert len(H2OPolicy().select_to_keep(stats(129), budget=26, step=0)) == 26
    for fragment in ("Budget `max(1, round(0.2 n))` blocks with mandatory blocks inside it, recomputed every 8 decode steps",
                     "history of the last 32 steps", "4 sink and 4 recent blocks mandatory",
                     "EMA over the last 8 history values with decay 0.9", "query view fixed at 0.5",
                     "it keeps the recent blocks, newest first, then the sinks",
                     "half the budget by summed history, the rest by recency", "at most 48 new tokens"):
        assert fragment in DOC, fragment

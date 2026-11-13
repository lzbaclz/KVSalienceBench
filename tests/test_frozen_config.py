"""Regression checks for feature, fitting and runtime constants."""
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




def test_logistic_fit_constants():
    from xqp.predictor import ClosedFormXQP
    fit = inspect.signature(ClosedFormXQP._fit_single).parameters
    assert fit["l2"].default == 1e-3 and fit["max_iter"].default == 40 and fit["tol"].default == 1e-6
    assert fit["class_weight"].default is None
    assert inspect.signature(ClosedFormXQP.from_fit).parameters["l2"].default == 1e-3
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

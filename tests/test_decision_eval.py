"""xqp.decision_eval against brute-force enumeration.

Every statistic is recomputed here the slow, obviously-correct way (all
(positive, negative) pairs, all tie permutations, explicitly duplicated requests)
on small random data that is full of ties, so a vectorization slip cannot hide.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from xqp.decision_eval import (DecisionData, Ranked, RequestStats, bootstrap_statistics,
                               cluster_multiplicities, macro_recall, pair_matrix,
                               pooled_concordant, pooled_decomposition)
from xqp.dm_metrics import precision_recall_at_k_grouped


def make_data(seed=0, n_req=6, decisions=(2, 4), size=(5, 13), pos_rate=0.3, decimals=1):
    rng = np.random.default_rng(seed)
    y, s, g, r = [], [], [], []
    gid = 0
    for req in range(n_req):
        for _ in range(int(rng.integers(decisions[0], decisions[1] + 1))):
            n = int(rng.integers(size[0], size[1] + 1))
            lab = (rng.random(n) < pos_rate).astype(np.int8)
            lab[0] = 1                       # every decision has a positive...
            lab[-1] = 0                      # ...and a negative
            y.append(lab)
            s.append(np.round(rng.normal(size=n) + 1.2 * lab, decimals))   # rounding creates ties
            g.append(np.full(n, gid)); r.append(np.full(n, req)); gid += 1
    return (np.concatenate(y), np.concatenate(s), np.concatenate(g), np.concatenate(r))


def brute_pairs(y, s, g):
    conc_total = conc_same = pairs_total = pairs_same = 0.0
    for i in np.flatnonzero(y == 1):
        for j in np.flatnonzero(y == 0):
            v = 1.0 if s[i] > s[j] else 0.5 if s[i] == s[j] else 0.0
            conc_total += v; pairs_total += 1
            if g[i] == g[j]:
                conc_same += v; pairs_same += 1
    return conc_total, conc_same, pairs_total, pairs_same


@pytest.mark.parametrize("seed", range(5))
def test_pooled_decomposition_matches_pair_enumeration(seed):
    y, s, g, r = make_data(seed)
    d = DecisionData.build(y, g, r)
    got = pooled_decomposition(Ranked(d, s))
    ct, cs, pt, ps = brute_pairs(y, s, g)
    assert got["conc_total"] == pytest.approx(ct) and got["conc_same"] == pytest.approx(cs)
    assert got["pairs_total"] == pt and got["pairs_same"] == ps
    assert got["auc_pooled"] == pytest.approx(ct / pt)
    assert got["auc_same"] == pytest.approx(cs / ps)
    assert got["auc_cross"] == pytest.approx((ct - cs) / (pt - ps))
    # the weights make the split exact
    w = got["share_same"]
    assert w * got["auc_same"] + (1 - w) * got["auc_cross"] == pytest.approx(got["auc_pooled"])
    assert pooled_concordant(y, s) == pytest.approx(ct)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("k_frac", [0.1, 0.2, 0.5])
def test_topk_matches_the_reference_implementation(seed, k_frac):
    y, s, g, r = make_data(seed)
    d = DecisionData.build(y, g, r)
    tk = Ranked(d, s).topk(k_frac)
    ref = precision_recall_at_k_grouped(y.astype(np.float32), s, g, k_frac)
    assert d.n_groups == ref["n_groups"]
    assert macro_recall(d, tk["tp"]) == pytest.approx(ref["macro_recall"], abs=1e-12)
    assert float(np.mean(tk["tp"] / tk["k"])) == pytest.approx(ref["macro_precision"], abs=1e-12)
    assert tk["tp"].sum() / tk["k"].sum() == pytest.approx(ref["micro_precision"], abs=1e-12)
    assert tk["tp"].sum() / d.npos_g.sum() == pytest.approx(ref["micro_recall"], abs=1e-12)


def test_random_tie_expectation_matches_enumeration():
    # decision 0: the k-th place falls inside a four-way tie; 1: no ties at all;
    # 2: a tie that ends exactly at the boundary; 3: a two-way tie straddling it
    s = np.array([5, 4, 3, 3, 3, 3, 2, 1, 0, 0,   9, 8, 7, 6, 5, 4,   2, 2, 1, 0, 0,
                  2, 2, 1, 1, 0, 0], float)
    y = np.array([0, 1, 1, 0, 0, 1, 0, 0, 1, 0,   1, 0, 0, 1, 0, 0,   0, 1, 1, 0, 1,
                  0, 1, 1, 0, 0, 1], np.int8)
    g = np.array([0] * 10 + [1] * 6 + [2] * 5 + [3] * 6)
    d = DecisionData.build(y, g, np.zeros_like(g))
    tk = Ranked(d, s).topk(0.4, ties="random")
    for gi in range(4):
        rows = np.flatnonzero(g == gi)
        k = int(np.ceil(0.4 * rows.size))
        # enumerate every order consistent with the scores (tied rows in any order)
        runs = [rows[s[rows] == v] for v in sorted(set(s[rows]), reverse=True)]
        totals = []
        for perms in itertools.product(*[list(itertools.permutations(run)) for run in runs]):
            ordering = [i for run in perms for i in run]
            totals.append(y[ordering[:k]].sum())
        assert tk["tp_random_ties"][gi] == pytest.approx(np.mean(totals)), gi
    assert [bool(b) for b in tk["boundary_tie"]] == [True, False, False, True]


def explicit_resample(y, s, g, r, mult):
    ys, ss, gs, rs, new = [], [], [], [], 0
    for req, m in enumerate(mult):
        rows = np.flatnonzero(r == req)
        for _ in range(int(m)):
            ys.append(y[rows]); ss.append(s[rows])
            gs.append(g[rows] + 10_000 * new); rs.append(np.full(rows.size, new)); new += 1
    return np.concatenate(ys), np.concatenate(ss), np.concatenate(gs), np.concatenate(rs)


@pytest.mark.parametrize("seed", range(4))
def test_bootstrap_quadratic_form_equals_explicit_resampling(seed):
    y, s, g, r = make_data(seed, n_req=7)
    d = DecisionData.build(y, g, r)
    rk = Ranked(d, s)
    C = pair_matrix(d, s)
    assert C.sum() == pytest.approx(pooled_concordant(y, s))
    k_fracs = (0.1, 0.2)
    st = RequestStats.build(rk, C, k_fracs)
    rng = np.random.default_rng(seed)
    mult = rng.integers(0, 4, size=d.n_req).astype(float)
    mult[0] = max(mult[0], 1)
    got = bootstrap_statistics(mult[None, :], st)
    y2, s2, g2, r2 = explicit_resample(y, s, g, r, mult)
    d2 = DecisionData.build(y2, g2, r2)
    rk2 = Ranked(d2, s2)
    want = pooled_decomposition(rk2)
    assert got["auc_pooled"][0] == pytest.approx(want["auc_pooled"])
    assert got["auc_same"][0] == pytest.approx(want["auc_same"])
    assert got["auc_cross"][0] == pytest.approx(want["auc_cross"])
    for kf in k_fracs:
        assert got[f"recall_{kf:g}"][0] == pytest.approx(macro_recall(d2, rk2.topk(kf)["tp"]))
    # the identity multiplicity reproduces the full-data statistics
    ident = bootstrap_statistics(np.ones((1, d.n_req)), st)
    full = pooled_decomposition(rk)
    assert ident["auc_pooled"][0] == pytest.approx(full["auc_pooled"])
    assert ident["auc_same"][0] == pytest.approx(full["auc_same"])


def test_cluster_multiplicities_resample_whole_clusters():
    clusters = np.array([0, 0, 1, 2, 2, 2])         # requests 0,1 share a cluster; 3,4,5 share one
    M = cluster_multiplicities(clusters, n_boot=500, seed=1)
    assert M.shape == (500, 6)
    assert (M[:, 0] == M[:, 1]).all() and (M[:, 3] == M[:, 4]).all() and (M[:, 4] == M[:, 5]).all()
    assert (M[:, [0, 2, 3]].sum(axis=1) == 3).all()    # one draw of K=3 clusters per replicate
    # each cluster is drawn K * (1/K) = 1 time on average
    assert M[:, 0].mean() == pytest.approx(1.0, abs=0.15)


def test_guards():
    y, s, g, r = make_data(0)
    d = DecisionData.build(y, g, r)
    bad = s.copy(); bad[3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        Ranked(d, bad)
    with pytest.raises(ValueError, match="more than one request"):
        DecisionData.build(y, g, np.where(g == g[0], np.arange(g.size) % 2, r))

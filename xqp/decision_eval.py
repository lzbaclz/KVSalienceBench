"""Exact decision-level evaluation of retention scorers.

A *decision* is one cache-retention choice: the (request, layer, step) group of
candidate blocks of the KVSalienceBench protocol. A selector with a per-step budget
uses only the order *inside* a decision, whereas pooled AUC ranks rows across
decisions. This module separates the two on every row, without sampling:

* ``pooled_decomposition`` splits the Mann-Whitney statistic into pairs whose
  positive and negative come from the SAME decision and pairs from different
  decisions. The weights are pair counts, so
  ``AUC_pooled = w * AUC_same + (1 - w) * AUC_cross`` holds exactly.
* ``Ranked.topk`` is a vectorized per-decision top-k with the deterministic tie rule
  of ``xqp.dm_metrics.precision_recall_at_k_grouped`` (descending score, ties by
  ascending row index), plus the exact expectation under random tie-breaking.
* ``pair_matrix`` stores concordant-pair counts request by request, which turns a
  cluster bootstrap of the pooled and per-decision statistics into a quadratic form:
  thousands of draws over millions of rows cost milliseconds.

Everything here is checked against brute-force enumeration in
``tests/test_decision_eval.py``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata


def _dense(ids):
    uniq, inv = np.unique(np.asarray(ids).reshape(-1), return_inverse=True)
    return inv.astype(np.int64), uniq


@dataclass(frozen=True)
class DecisionData:
    """Labels, dense decision codes and the request each decision belongs to."""

    y: np.ndarray            # (n,) int8 labels in {0, 1}
    gcode: np.ndarray        # (n,) int64 dense decision code, 0..G-1
    req: np.ndarray          # (n,) int64 dense request index, 0..R-1
    group_req: np.ndarray    # (G,) request of each decision
    group_size: np.ndarray   # (G,) candidates per decision
    npos_g: np.ndarray       # (G,) positives per decision (float64)
    nneg_g: np.ndarray       # (G,) negatives per decision (float64)
    n_req: int

    @property
    def n_groups(self) -> int:
        return int(self.group_size.size)

    @classmethod
    def build(cls, y, decisions, requests) -> "DecisionData":
        y = (np.asarray(y).reshape(-1) > 0.5).astype(np.int8)
        gcode, _ = _dense(decisions)
        req, _ = _dense(requests)
        if not (y.shape == gcode.shape == req.shape):
            raise ValueError("labels, decisions and requests must align row for row")
        G, R = int(gcode.max()) + 1, int(req.max()) + 1
        group_req = np.full(G, -1, np.int64)
        group_req[gcode] = req
        if not np.array_equal(group_req[gcode], req):
            raise ValueError("a decision spans more than one request")
        size = np.bincount(gcode, minlength=G)
        npos = np.bincount(gcode, weights=y, minlength=G)
        return cls(y=y, gcode=gcode, req=req, group_req=group_req, group_size=size,
                   npos_g=npos, nneg_g=size - npos, n_req=R)

    @property
    def pairs_same_r(self) -> np.ndarray:
        """Same-decision (positive, negative) pairs per request."""
        return np.bincount(self.group_req, weights=self.npos_g * self.nneg_g, minlength=self.n_req)

    @property
    def pos_r(self) -> np.ndarray:
        return np.bincount(self.req, weights=self.y, minlength=self.n_req)

    @property
    def neg_r(self) -> np.ndarray:
        return np.bincount(self.req, minlength=self.n_req) - self.pos_r


class Ranked:
    """One descending within-decision sort of a score vector.

    Rows are ordered by decision, then by descending score, then by ascending row
    index (``np.lexsort`` is stable), which is exactly the tie rule of the reference
    top-k. All decision-level statistics derive from this single sort.
    """

    def __init__(self, data: DecisionData, scores):
        s = np.asarray(scores, np.float64).reshape(-1)
        if s.shape != data.y.shape:
            raise ValueError("scores must align with the labels")
        if not np.isfinite(s).all():
            raise ValueError("non-finite scores")
        self.data, self.s = data, s
        n = s.size
        order = np.lexsort((-s, data.gcode))
        gs, ss = data.gcode[order], s[order]
        offsets = np.r_[0, np.cumsum(data.group_size)]
        pos = np.arange(n) - offsets[gs]
        new_run = np.empty(n, bool)
        new_run[0] = True
        np.not_equal(gs[1:], gs[:-1], out=new_run[1:])
        new_run[1:] |= ss[1:] != ss[:-1]
        run_start = np.flatnonzero(new_run)
        self.order, self.gs, self.pos, self.offsets = order, gs, pos, offsets
        self.run_start = run_start
        self.run_id = np.cumsum(new_run) - 1
        self.run_len = np.diff(np.r_[run_start, n])
        self.y_sorted = data.y[order]

    # -- within-decision AUC ---------------------------------------------------
    def same_decision_concordant(self) -> np.ndarray:
        """Per decision: (positive, negative) pairs with the positive scored higher,
        ties counting one half (the Mann-Whitney U of that decision)."""
        d = self.data
        avg_desc = (self.pos[self.run_start] + 1.0 + (self.run_len - 1) / 2.0)[self.run_id]
        r_asc = d.group_size[self.gs] + 1.0 - avg_desc
        rank_sum = np.bincount(self.gs, weights=r_asc * self.y_sorted, minlength=d.n_groups)
        return rank_sum - d.npos_g * (d.npos_g + 1.0) / 2.0

    # -- per-decision top-k ----------------------------------------------------
    def topk(self, k_frac: float, ties: str = "stable") -> dict:
        """Per-decision selected positives for k = ceil(k_frac * n) candidates.

        ``ties='stable'`` reproduces ``precision_recall_at_k_grouped``. The returned
        ``tp_random_ties`` is the exact expectation when the tie group straddling the
        k-th place is resolved uniformly at random; ``boundary_tie`` marks the
        decisions where such a straddle exists.
        """
        d = self.data
        k = np.maximum(1, np.ceil(k_frac * d.group_size.astype(np.float64)).astype(np.int64))
        selected = self.pos < k[self.gs]
        tp = np.bincount(self.gs, weights=(selected & (self.y_sorted == 1)), minlength=d.n_groups)
        out = dict(k=k, tp=tp)
        if ties == "random":
            last = self.offsets[:-1] + k - 1
            rid = self.run_id[last]
            a0 = self.pos[self.run_start[rid]]
            m = self.run_len[rid]
            straddle = (a0 + m) > k
            q = k - a0
            cs = np.concatenate([[0], np.cumsum(self.y_sorted, dtype=np.int64)])
            r0 = self.run_start[rid]
            positives_in_run = cs[r0 + m] - cs[r0]
            stable_in_run = cs[r0 + q] - cs[r0]
            out["tp_random_ties"] = tp + np.where(straddle, q * positives_in_run / m - stable_in_run, 0.0)
            out["boundary_tie"] = straddle
        elif ties != "stable":
            raise ValueError(ties)
        return out

    def selected_mask(self, k_frac: float) -> np.ndarray:
        """Boolean mask, in the ORIGINAL row order, of the stable top-k rows."""
        d = self.data
        k = np.maximum(1, np.ceil(k_frac * d.group_size.astype(np.float64)).astype(np.int64))
        mask = np.zeros(self.s.size, bool)
        mask[self.order] = self.pos < k[self.gs]
        return mask

    def tied_row_share(self) -> float:
        """Share of rows that sit in a within-decision tie run longer than one."""
        return float(self.run_len[self.run_len > 1].sum() / self.s.size)


def macro_recall(data: DecisionData, tp: np.ndarray) -> float:
    """Mean over decisions of selected positives / positives."""
    ok = data.npos_g > 0
    return float(np.mean(tp[ok] / data.npos_g[ok]))


def recall_by_request(data: DecisionData, tp: np.ndarray):
    """Per request: sum of per-decision recalls and number of decisions (macro numerator
    and denominator), so a cluster bootstrap can resum them."""
    ok = data.npos_g > 0
    num = np.bincount(data.group_req[ok], weights=tp[ok] / data.npos_g[ok], minlength=data.n_req)
    den = np.bincount(data.group_req[ok], minlength=data.n_req).astype(np.float64)
    return num, den


# ----------------------------- pooled decomposition ---------------------------
def pooled_concordant(y, scores) -> float:
    """Mann-Whitney U over ALL rows: (positive, negative) pairs ranked correctly, ties 1/2."""
    y = (np.asarray(y).reshape(-1) > 0.5)
    r = rankdata(np.asarray(scores, np.float64).reshape(-1), method="average")
    n_pos = float(y.sum())
    return float(r[y].sum() - n_pos * (n_pos + 1.0) / 2.0)


def pooled_decomposition(ranked: Ranked) -> dict:
    """Exact split of pooled AUC into same-decision and cross-decision pairs."""
    d = ranked.data
    conc_total = pooled_concordant(d.y, ranked.s)
    u_g = ranked.same_decision_concordant()
    conc_same = float(u_g.sum())
    n_pos, n_neg = float(d.npos_g.sum()), float(d.nneg_g.sum())
    pairs_total = n_pos * n_neg
    pairs_same = float((d.npos_g * d.nneg_g).sum())
    has_both = (d.npos_g > 0) & (d.nneg_g > 0)
    pairs_cross = pairs_total - pairs_same
    return dict(
        auc_pooled=conc_total / pairs_total,
        auc_same=conc_same / pairs_same if pairs_same else float("nan"),
        auc_cross=(conc_total - conc_same) / pairs_cross if pairs_cross else float("nan"),
        auc_same_macro=float(np.mean(u_g[has_both] / (d.npos_g[has_both] * d.nneg_g[has_both]))),
        share_same=pairs_same / pairs_total,
        pairs_total=pairs_total, pairs_same=pairs_same,
        conc_total=conc_total, conc_same=conc_same)


def pair_matrix(data: DecisionData, scores) -> np.ndarray:
    """C[r, r'] = concordant (positive in request r, negative in request r') pairs,
    ties counting one half; ``C.sum()`` is the pooled Mann-Whitney U."""
    s = np.asarray(scores, np.float64).reshape(-1)
    R = data.n_req
    is_pos = data.y == 1
    sp, rp = s[is_pos], data.req[is_pos]
    p_order = np.argsort(sp, kind="stable")           # sorted queries make searchsorted cache-friendly
    sp, rp = sp[p_order], rp[p_order]
    neg_idx = np.flatnonzero(~is_pos)
    sn, rn = s[neg_idx], data.req[neg_idx]
    o = np.lexsort((sn, rn))
    sn, rn = sn[o], rn[o]
    bounds = np.searchsorted(rn, np.arange(R + 1))
    C = np.zeros((R, R))
    for r2 in range(R):
        block = sn[bounds[r2]:bounds[r2 + 1]]
        left = np.searchsorted(block, sp, side="left")
        right = np.searchsorted(block, sp, side="right")
        C[:, r2] = np.bincount(rp, weights=0.5 * (left + right), minlength=R)
    return C


# --------------------------------- cluster bootstrap --------------------------
def cluster_multiplicities(cluster_of_request, n_boot: int, seed: int = 0) -> np.ndarray:
    """(n_boot, R) request multiplicities of a bootstrap that resamples whole clusters
    (e.g. source prompts) with replacement; every request of a drawn cluster is repeated
    with the cluster's multiplicity."""
    cl = np.asarray(cluster_of_request).reshape(-1)
    clusters = np.unique(cl)
    K = clusters.size
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(K, np.full(K, 1.0 / K), size=n_boot)
    return draws[:, np.searchsorted(clusters, cl)].astype(np.float64)


@dataclass(frozen=True)
class RequestStats:
    """Sufficient statistics that make every pooled and per-decision statistic additive
    over requests (except the pooled pair count, which needs the matrix ``C``)."""

    pos_r: np.ndarray
    neg_r: np.ndarray
    pairs_same_r: np.ndarray
    conc_same_r: np.ndarray
    C: np.ndarray
    recall_num_r: dict   # k_frac -> (R,)
    recall_den_r: np.ndarray

    @classmethod
    def build(cls, ranked: Ranked, pair_mat: np.ndarray, k_fracs) -> "RequestStats":
        d = ranked.data
        u_g = ranked.same_decision_concordant()
        conc_same_r = np.bincount(d.group_req, weights=u_g, minlength=d.n_req)
        num, den = {}, None
        for kf in k_fracs:
            tp = ranked.topk(kf)["tp"]
            num[kf], den = recall_by_request(d, tp)
        return cls(pos_r=d.pos_r, neg_r=d.neg_r, pairs_same_r=d.pairs_same_r,
                   conc_same_r=conc_same_r, C=pair_mat, recall_num_r=num, recall_den_r=den)


def bootstrap_statistics(M: np.ndarray, st: RequestStats) -> dict:
    """Vectorized statistics for every row of ``M`` (request multiplicities)."""
    P, N = M @ st.pos_r, M @ st.neg_r
    total = np.einsum("bi,ij,bj->b", M, st.C, M)
    pairs_same = M @ st.pairs_same_r
    conc_same = M @ st.conc_same_r
    pairs_total = P * N
    out = dict(auc_pooled=total / pairs_total,
               auc_same=conc_same / pairs_same,
               auc_cross=(total - conc_same) / (pairs_total - pairs_same))
    den = M @ st.recall_den_r
    for kf, num in st.recall_num_r.items():
        out[f"recall_{kf:g}"] = (M @ num) / den
    return out


def percentile_ci(x: np.ndarray, level: float = 0.95):
    a = (1.0 - level) / 2.0
    return [float(np.percentile(x, 100 * a)), float(np.percentile(x, 100 * (1 - a)))]

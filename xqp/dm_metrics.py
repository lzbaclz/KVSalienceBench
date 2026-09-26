"""Data-mining evaluation metrics for the ICDM framing.

The systems framing cared about TPOT/WCET; the DM framing cares about ranking
quality on an *imbalanced* stream and whether the predicted probabilities are
*trustworthy*. This module adds the metrics ICDM reviewers expect and the
existing `eval.py` lacks: average precision (AUPRC), expected calibration error
+ reliability curve, precision@k, and the Brier score.

All functions are NumPy-only and CPU-runnable; pair them with `eval.roc_auc`.
"""
from __future__ import annotations

import numpy as np


def _check_pair(y_true, y_score):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)
    if y_true.shape != y_score.shape:
        raise ValueError(f"shape mismatch {y_true.shape} vs {y_score.shape}")
    if y_score.size and not np.isfinite(y_score).all():
        raise ValueError("non-finite scores; refusing to rank NaN/inf silently")
    return y_true, y_score


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Threshold-based average precision (AUPRC), tie-aware.

    AP = sum_n (R_n - R_{n-1}) * P_n over the *distinct score thresholds*, in
    decreasing order, where P_n/R_n are precision/recall of the set
    {score >= threshold_n}. Tied scores therefore enter as one group and the
    result cannot depend on the input row order. This matches
    ``sklearn.metrics.average_precision_score``.

    History: until 2026-09 this function integrated one row at a time after a
    stable sort, so equal scores were ordered by row position and a constant
    predictor could score anywhere between the base rate and 1.0 depending on
    where the positives happened to sit. Archived AUPRC values for scorers with
    many exact ties (binary indicators, integer-valued age proxies) were
    produced by that rule and are flagged in the manuscript.
    Returns NaN when there is no positive example.
    """
    y_true, y_score = _check_pair(y_true, y_score)
    n_pos = float(y_true.sum())
    if n_pos == 0 or y_true.shape[0] == 0:
        return float("nan")
    order = np.argsort(-y_score, kind="stable")
    s = y_score[order]
    y = y_true[order]
    # Last row of every tie group (scores are sorted descending).
    last = np.r_[np.flatnonzero(np.diff(s) != 0.0), s.size - 1]
    tp = np.cumsum(y)[last]
    fp = np.cumsum(1.0 - y)[last]
    precision = tp / (tp + fp)
    recall = tp / n_pos
    rec_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - rec_prev) * precision))


def _top_k_indices(y_score: np.ndarray, k: int) -> np.ndarray:
    """Deterministic top-k: descending score, ties by ascending row index."""
    return np.argsort(-y_score, kind="stable")[:k]


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k_frac: float = 0.10) -> float:
    """POOLED top-decile precision: k = ceil(k_frac * N) over ALL rows passed in.

    This is one global ranking over the pooled rows (e.g. 150K test rows from
    many requests, layers and steps), not a per-cache-decision budget: the
    global top-k may take more than 10% of one request/layer/step and less of
    another. For the per-decision-group operating point use
    :func:`precision_recall_at_k_grouped`. Ties are broken deterministically
    by row order (stable sort); the archived driver used ``argpartition``,
    whose tie order is implementation-defined.
    """
    y_true, y_score = _check_pair(y_true, y_score)
    n = y_true.shape[0]
    if n == 0:
        return float("nan")
    k = max(1, int(np.ceil(k_frac * n)))
    idx = _top_k_indices(y_score, k)
    return float(y_true[idx].sum()) / k


def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k_frac: float = 0.10) -> float:
    """POOLED top-decile recall (same global ranking as :func:`precision_at_k`)."""
    y_true, y_score = _check_pair(y_true, y_score)
    n = y_true.shape[0]
    n_pos = float((y_true > 0.5).sum())
    if n == 0 or n_pos == 0:
        return float("nan")
    k = max(1, int(np.ceil(k_frac * n)))
    idx = _top_k_indices(y_score, k)
    return float(y_true[idx].sum()) / n_pos


def precision_recall_at_k_grouped(y_true: np.ndarray, y_score: np.ndarray, groups: np.ndarray,
                                  k_frac: float = 0.10) -> dict:
    """Per-decision-group top-k precision/recall.

    Rows are partitioned by ``groups`` (one group = one (request, layer, step)
    cache decision). Inside each group the predicted set is the top
    ``ceil(k_frac * n_group)`` rows by score, ties broken by ascending row
    index, and the label set is whatever ``y_true`` marks (the label protocol
    already uses ``ceil(r * n_group)`` positives per group). Returns macro
    averages over groups and micro (pooled true-positive) rates, plus the
    number of groups, so the operating point is the one a per-step budget
    would actually impose.
    """
    y_true, y_score = _check_pair(y_true, y_score)
    groups = np.asarray(groups).reshape(-1)
    if groups.shape != y_true.shape:
        raise ValueError("groups must align with rows")
    if y_true.size == 0:
        return dict(n_groups=0, macro_precision=float("nan"), macro_recall=float("nan"),
                    micro_precision=float("nan"), micro_recall=float("nan"))
    order = np.argsort(groups, kind="stable")
    g = groups[order]
    starts = np.r_[0, np.flatnonzero(g[1:] != g[:-1]) + 1, g.size]
    tp_sum = k_sum = pos_sum = 0.0
    prec, rec = [], []
    for a, b in zip(starts[:-1], starts[1:]):
        rows = order[a:b]
        n = b - a
        k = max(1, int(np.ceil(k_frac * n)))
        idx = rows[_top_k_indices(y_score[rows], k)]
        tp = float(y_true[idx].sum())
        n_pos = float((y_true[rows] > 0.5).sum())
        tp_sum += tp; k_sum += k; pos_sum += n_pos
        prec.append(tp / k)
        if n_pos > 0:
            rec.append(tp / n_pos)
    return dict(n_groups=int(len(starts) - 1),
                macro_precision=float(np.mean(prec)),
                macro_recall=float(np.mean(rec)) if rec else float("nan"),
                micro_precision=float(tp_sum / k_sum),
                micro_recall=float(tp_sum / pos_sum) if pos_sum else float("nan"))


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean squared error of probabilistic predictions (lower = better)."""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    if y_true.shape[0] == 0:
        return float("nan")
    return float(np.mean((y_prob - y_true) ** 2))


def reliability_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict:
    """Per-bin (confidence, accuracy, count) for a reliability diagram."""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    conf, acc, cnt = [], [], []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (y_prob > lo) & (y_prob <= hi) if b > 0 else (y_prob >= lo) & (y_prob <= hi)
        c = int(mask.sum())
        cnt.append(c)
        conf.append(float(y_prob[mask].mean()) if c else float("nan"))
        acc.append(float(y_true[mask].mean()) if c else float("nan"))
    return dict(bin_edges=edges.tolist(), confidence=conf, accuracy=acc, count=cnt)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """ECE = sum_b (n_b/N) |conf_b - acc_b|. 0 = perfectly calibrated.

    A proper-scoring-rule fit (the closed-form logistic regression) should be
    well-calibrated out of the box — a genuine advantage over threshold
    heuristics that the DM framing rewards and the systems framing ignored.
    """
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    n = y_true.shape[0]
    if n == 0:
        return float("nan")
    rc = reliability_curve(y_true, y_prob, n_bins=n_bins)
    ece = 0.0
    for conf, acc, c in zip(rc["confidence"], rc["accuracy"], rc["count"]):
        if c == 0:
            continue
        ece += (c / n) * abs(conf - acc)
    return float(ece)

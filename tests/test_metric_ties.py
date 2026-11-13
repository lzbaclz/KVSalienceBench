"""Tie handling of the ranking metrics (PC audit item P0-3).

The archived ``average_precision`` integrated row by row after a stable sort,
so equal scores were ordered by row position and a constant predictor could
score anywhere between the base rate and 1.0. These tests pin the corrected,
threshold-based definition and its agreement with scikit-learn.
"""
import numpy as np
import pytest

from xqp.dm_metrics import (average_precision, precision_at_k, recall_at_k,
                            precision_recall_at_k_grouped)

sklearn_metrics = pytest.importorskip("sklearn.metrics")


def test_constant_predictor_scores_the_base_rate_whatever_the_row_order():
    tied = np.full(4, 0.5)
    for y in ([1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0]):
        assert average_precision(np.array(y, float), tied) == pytest.approx(0.5)


def test_tie_groups_are_permutation_invariant():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.1).astype(float)
    s = rng.integers(0, 5, 2000).astype(float)          # five distinct scores -> huge tie groups
    ref = average_precision(y, s)
    for _ in range(5):
        perm = rng.permutation(2000)
        assert average_precision(y[perm], s[perm]) == pytest.approx(ref, abs=1e-12)


@pytest.mark.parametrize("kind", ["continuous", "binary", "coarse"])
def test_matches_sklearn_average_precision(kind):
    rng = np.random.default_rng(1)
    y = (rng.random(5000) < 0.12).astype(float)
    if kind == "continuous":
        s = rng.random(5000)
    elif kind == "binary":
        s = (rng.random(5000) < 0.1).astype(float)
    else:
        s = np.round(rng.random(5000), 1)
    assert average_precision(y, s) == pytest.approx(
        sklearn_metrics.average_precision_score(y, s), abs=1e-12)


def test_separable_and_no_positive_cases_unchanged():
    y = np.array([0, 0, 1, 1], float)
    assert average_precision(y, y) == pytest.approx(1.0)
    assert np.isnan(average_precision(np.zeros(8), np.linspace(0, 1, 8)))


def test_non_finite_scores_are_rejected_not_ranked():
    with pytest.raises(ValueError):
        average_precision(np.array([0, 1.0]), np.array([np.nan, 0.5]))
    with pytest.raises(ValueError):
        precision_at_k(np.array([0, 1.0]), np.array([np.inf, 0.5]))


def test_pooled_top_k_is_deterministic_under_ties():
    y = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0], float)
    s = np.zeros(10)                                     # all tied: first ceil(0.1*10)=1 row wins
    assert precision_at_k(y, s, 0.10) == pytest.approx(1.0)
    assert recall_at_k(y, s, 0.10) == pytest.approx(0.5)


def test_grouped_top_k_differs_from_pooled_when_scores_are_not_comparable_across_groups():
    # Group A scores live in [10, 11], group B in [0, 1]; a pooled top-decile
    # spends the whole budget on group A, a per-group budget does not.
    rng = np.random.default_rng(2)
    n = 100
    y = np.zeros(2 * n); s = np.zeros(2 * n); g = np.repeat([0, 1], n)
    for grp, base in ((0, 10.0), (1, 0.0)):
        rows = np.flatnonzero(g == grp)
        pos = rng.choice(rows, 10, replace=False)
        y[pos] = 1.0
        s[rows] = base + 0.8 * rng.random(n)              # negatives in [base, base+0.8)
        s[pos] = base + 0.9 + 0.1 * rng.random(10)        # positives rank at the top inside their group
    grouped = precision_recall_at_k_grouped(y, s, g, 0.10)
    assert grouped["n_groups"] == 2
    assert grouped["macro_precision"] == pytest.approx(1.0)
    assert grouped["micro_recall"] == pytest.approx(1.0)
    pooled = recall_at_k(y, s, 0.10)
    assert pooled == pytest.approx(0.5)                   # group B's positives never enter the pooled top-10%


def test_grouped_rejects_misaligned_groups():
    with pytest.raises(ValueError):
        precision_recall_at_k_grouped(np.zeros(4), np.zeros(4), np.zeros(3))

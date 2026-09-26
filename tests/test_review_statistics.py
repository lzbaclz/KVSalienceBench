"""Check the accelerated cluster bootstrap against literal row duplication."""
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from analyze_review_statistics import auc_sufficient_statistics, bootstrap_auc


@pytest.mark.parametrize("tied", [False, True])
def test_cluster_auc_matches_literal_bootstrap(tied):
    rng = np.random.default_rng(12)
    groups = np.repeat(np.arange(7), [7, 8, 9, 5, 12, 8, 6])
    y = rng.integers(2, size=len(groups))
    score = rng.integers(3, size=len(y)) if tied else rng.random(len(y))
    stats = auc_sufficient_statistics(y, score, groups)
    counts = rng.multinomial(7, np.full(7, 1 / 7), size=50)
    expected = []
    for w in counts:
        idx = np.repeat(np.arange(len(y)), w[groups])
        expected.append(roc_auc_score(y[idx], score[idx]))
    np.testing.assert_allclose(bootstrap_auc(stats, counts), expected, atol=1e-14)


def test_constant_predictor_auc_is_half():
    stats = auc_sufficient_statistics([0, 1, 1, 0], [1, 1, 1, 1], [0, 0, 1, 1])
    np.testing.assert_array_equal(bootstrap_auc(stats, np.array([[1, 1], [0, 2]])), [0.5, 0.5])


def test_committed_intervals_replay_without_traces():
    import json
    from pathlib import Path
    from analyze_review_statistics import replay_intervals
    result = json.loads((Path(__file__).resolve().parents[1] / "experiments/results/mentor_statistics.json").read_text())
    assert replay_intervals(result) == 32

"""KVSalienceBench submission template (protocol 2.0).

Copy this file, implement ``score``, and evaluate with:

    python -m benchmark.run_leaderboard --submission your_method.py \
        --traces Llama=path.jsonl,Qwen=path.jsonl

CONTRACT
--------
``score(F)`` receives an (N, 4) float32 feature matrix whose columns are, in order,
``benchmark.protocol.FEATURE_COLUMNS`` = ("s_within", "s_cross", "s_query", "s_pos"):
  s_within : current-layer attention-magnitude EMA, normalized by the decision's maximum
  s_cross  : previous-layer top-r membership indicator in {0, 1}
  s_query  : cosine query-key affinity proxy
  s_pos    : block-age proxy, exp(-(t - creation step)/w)
and returns one score per row; higher means "more likely to be in the top-10% attended
set of the same decision 4 decode steps later". Rows of one decision arrive together,
but the contract is row-wise: a scorer that needs decision context must recover it from
the features itself, because a deployed selector sees one decision at a time.

SCORING
-------
The protocol scores both the pooled ranking and the per-decision budget a selector
actually imposes, so a method can be better at one and worse at the other. Train on the
training rows of the split you are evaluated on; the evaluator holds out whole source
prompts by default. Calibration (ECE, Brier) is reported only if you set

    PROBABILISTIC = True

because a fixed top-k budget needs only the ordering, while a thresholded selector
needs probabilities. The reference baseline to compare against is the three-parameter
within+cross logistic (``benchmark/reference_model_v2.json``).
"""
from __future__ import annotations

import numpy as np

PROBABILISTIC = False


def score(F: np.ndarray) -> np.ndarray:
    """REPLACE THIS. The trivial example returns the within-layer view, a strong
    single-signal baseline whose per-decision recall the paper shows is hard to beat."""
    return np.asarray(F, np.float32)[:, 0]

"""Map an empirical miss target to layer thresholds and resulting retention.

The corrected lower split-conformal quantile has a marginal interpretation
only for exchangeable calibration/test units. Correlated block and step rows,
deployment feedback and transferred thresholds do not establish those
assumptions or a per-request guarantee. Archived measurements used the older
zero-based rank and remain unchanged; see the manuscript's GuardKV section.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .gated_predictor import _gather4


def _conformal_tau(scores_salient: np.ndarray, alpha: float) -> float:
    """Lower one-based split-conformal quantile, with score-zero sentinel.

    Retain {p >= tau}; the rank is floor(alpha*(n+1)), capped at n.
    This is a marginal rank construction, not an empirical per-request bound.
    """
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must lie in [0, 1]")
    s = np.sort(np.asarray(scores_salient, np.float64).reshape(-1))
    if not np.isfinite(s).all() or np.any((s < 0) | (s > 1)):
        raise ValueError("salient scores must be finite probabilities in [0, 1]")
    n = s.shape[0]
    if n == 0:
        return 0.0
    # k is ONE-based. k=0 means include all probability scores. The archived
    # helper used s[k] (clipped), one order statistic higher; archived result
    # JSON/checkpoints are retained and are not results of this corrected rule.
    k = min(int(np.floor(alpha * (n + 1))), n)
    return 0.0 if k == 0 else float(s[k - 1])


@dataclass
class CoverageDrivenBudgeter:
    """Turn a target miss-rate alpha into per-layer thresholds (hence budgets)."""
    scorer: object                  # exposes .score(F4) -> prob in [0,1]
    cols: tuple = (0, 1)            # within+cross (the calibrated minimal model)
    alpha: float = 0.10
    tau_by_layer: dict = field(default_factory=dict)
    tau_global: float = 0.5

    @classmethod
    def calibrate(cls, scorer, F, y, layer_ids, *, cols=(0, 1), alpha=0.10,
                  min_per_layer=64) -> "CoverageDrivenBudgeter":
        F = np.asarray(F, np.float32); y = np.asarray(y).reshape(-1)
        layer_ids = np.asarray(layer_ids).reshape(-1)
        p = np.asarray(scorer.score(_gather4(F, cols)))
        sal = y > 0.5
        tau = {}
        for l in np.unique(layer_ids):
            m = sal & (layer_ids == l)
            if m.sum() >= min_per_layer:
                tau[int(l)] = _conformal_tau(p[m], alpha)
        tau_g = _conformal_tau(p[sal], alpha)
        return cls(scorer=scorer, cols=tuple(cols), alpha=float(alpha),
                   tau_by_layer=tau, tau_global=float(tau_g))

    def _thresholds(self, layer_ids):
        return np.array([self.tau_by_layer.get(int(l), self.tau_global)
                         for l in layer_ids], np.float32)

    def norm_curve(self):
        """Thresholds keyed by normalized layer position in [0,1].

        This defines transfer interpolation, not a coverage guarantee on a new
        model. Returns sorted [(pos, tau), ...].
        """
        if not self.tau_by_layer:
            return [(0.0, self.tau_global), (1.0, self.tau_global)]
        lid = sorted(self.tau_by_layer)
        denom = max(1, lid[-1])
        return [(l / denom, float(self.tau_by_layer[l])) for l in lid]

    def save(self, path):
        """Persist the risk target + per-layer thresholds (raw + normalized) so the
        SEER GuardKV policy can reload the thresholds. JSON, numpy-free."""
        import json
        from pathlib import Path
        obj = dict(alpha=float(self.alpha), cols=list(self.cols),
                   tau_global=float(self.tau_global),
                   tau_by_layer={str(k): float(v) for k, v in self.tau_by_layer.items()},
                   norm_curve=[[float(p), float(t)] for p, t in self.norm_curve()])
        Path(path).write_text(json.dumps(obj, indent=2))
        return path

    def keep_mask(self, F, layer_ids, per_layer=True):
        p = np.asarray(self.scorer.score(_gather4(np.asarray(F, np.float32), self.cols)))
        thr = self._thresholds(layer_ids) if per_layer else self.tau_global
        return p >= thr, p

    def evaluate(self, F, y, layer_ids, per_layer=True) -> dict:
        """Realized miss-rate + emergent budget, overall and per layer."""
        y = np.asarray(y).reshape(-1); layer_ids = np.asarray(layer_ids).reshape(-1)
        keep, _ = self.keep_mask(F, layer_ids, per_layer=per_layer)
        sal = y > 0.5
        miss = float((sal & ~keep).sum() / max(1, sal.sum()))
        budget = float(keep.mean())
        per = {}
        for l in np.unique(layer_ids):
            m = layer_ids == l
            ms = sal & m
            per[int(l)] = dict(
                miss=float((ms & ~keep).sum() / max(1, ms.sum())),
                budget=float(keep[m].mean()))
        miss_spread = float(np.std([v["miss"] for v in per.values()]))
        return dict(target_alpha=self.alpha, realized_miss=miss, emergent_budget=budget,
                    per_layer_miss_std=miss_spread, per_layer=per)


def fixed_ratio_select(scores: np.ndarray, budget: float) -> np.ndarray:
    """Baseline: keep the top-`budget` fraction by score (global), the way Ada-KV /
    SnapKV consume a ratio. Used to contrast against coverage-driven allocation."""
    n = scores.shape[0]
    k = max(1, int(round(budget * n)))
    m = np.zeros(n, bool)
    m[np.argpartition(-scores, kth=min(k - 1, n - 1))[:k]] = True
    return m

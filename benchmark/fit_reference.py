"""Fit the protocol-2.0 reference baseline: the three-parameter within+cross logistic.

The reference is trained ONLY on the training rows of the split being evaluated, so it
never sees a held-out prompt. With ``--split request`` and the paper's two traces this
reproduces ``benchmark/reference_model_v2.json`` (the 'two-view logistic, refit' row of
the paper's Table II): 120,000 rows are drawn with seed 0 from the held-in rows.

    python benchmark/fit_reference.py --traces Llama=path.jsonl,Qwen=path.jsonl \
        --split request [--out benchmark/reference_model_v2.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark import protocol as P  # noqa: E402
from xqp.predictor import ClosedFormXQP  # noqa: E402

TRAIN_ROWS = 120_000
MASK2 = np.array([1, 1, 0, 0], np.float32)     # within + cross only: two weights and a bias


def sample_rows(idx: np.ndarray, n: int, seed: int = P.SEED) -> np.ndarray:
    """The draw ``experiments/run_icdm_full.subsample`` makes (same RNG calls)."""
    if len(idx) <= n:
        return idx
    return idx[np.random.default_rng(seed).choice(len(idx), size=n, replace=False)]


def fit_reference(F: np.ndarray, y: np.ndarray, train_idx: np.ndarray, rows: int = TRAIN_ROWS,
                  seed: int = P.SEED) -> ClosedFormXQP:
    tr = sample_rows(np.asarray(train_idx), rows, seed)
    return ClosedFormXQP.from_fit(np.asarray(F[tr], np.float32) * MASK2, np.asarray(y[tr], np.float32))


def score_fn(model: ClosedFormXQP):
    return lambda F: np.asarray(model.score(np.asarray(F, np.float32) * MASK2), np.float64)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", required=True, help="comma-separated NAME=path.jsonl (version-2 traces)")
    ap.add_argument("--split", choices=("source", "request"), default="source")
    ap.add_argument("--rows", type=int, default=TRAIN_ROWS)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    corpus = P.load_corpus(a.traces)
    tr, _ = P.split(corpus, a.split)
    model = fit_reference(corpus["F"], corpus["y"], tr, a.rows)
    w = model.weights.tolist()
    print(f"split={a.split}  weights within={w[0]:.4f} cross={w[1]:.4f}  bias={float(model.bias):.4f}  "
          f"offset w_cross/w_within={w[1] / w[0]:.4f}")
    if a.out:
        ref = dict(name=f"within+cross logistic, refit on the version-2 {a.split} split",
                   protocol=P.PROTOCOL_VERSION, feature_columns=P.FEATURE_COLUMNS,
                   used_features=["s_within", "s_cross"], params=3,
                   score="sigmoid(w_within*s_within + w_cross*s_cross + bias)",
                   weights={P.FEATURE_COLUMNS[i]: float(w[i]) for i in range(4)}, bias=float(model.bias),
                   fit=dict(method="Newton-IRLS + L2 ridge (l2=1e-3)", rows=int(min(a.rows, len(tr))),
                            split=a.split, traces=a.traces))
        a.out.write_text(json.dumps(ref, indent=2) + "\n")
        print("WROTE", a.out)


if __name__ == "__main__":
    main()

"""LEGACY (protocol 1.0): fit + serialize the version-1 reference baseline, the
3-parameter within+cross logistic model. Writes
benchmark/legacy_v1/reference_model.json. The version-2 reference lives in
benchmark/reference_model_v2.json (see benchmark/DATASHEET.md).

    python benchmark/legacy_v1/fit_reference.py --traces '/public/xqp_traces/*.jsonl'
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from benchmark.legacy_v1 import protocol as P
from xqp.predictor import ClosedFormXQP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="/public/xqp_traces/*.jsonl")
    ap.add_argument("--cap", type=int, default=400_000, help="rows/file cap for fitting")
    ap.add_argument("--out", default="benchmark/legacy_v1/reference_model.json")
    a = ap.parse_args()

    corpus = P.load_corpus(a.traces, cap=a.cap)
    tr, _ = P.request_split(corpus["rid"])
    F = corpus["F"]; y = corpus["y"].astype(np.float32)
    # within+cross only: zero the query/pos columns so only 2 features + bias are fit.
    Fz = F.copy(); Fz[:, 2] = 0.0; Fz[:, 3] = 0.0
    model = ClosedFormXQP.from_fit(Fz[tr], y[tr])

    res = P.evaluate(lambda Feat: model.score(_z(Feat)), corpus)

    w = model.weights.tolist()
    ref = dict(
        name="within+cross calibrated logistic (KVSalienceBench reference baseline)",
        protocol=P.PROTOCOL_VERSION,
        feature_columns=P.FEATURE_COLUMNS,                  # 4 cols; only within+cross used
        used_features=["s_within", "s_cross"],
        params=3,
        score="sigmoid(w_within*s_within + w_cross*s_cross + bias)",
        weights={P.FEATURE_COLUMNS[i]: float(w[i]) for i in range(4)},
        bias=float(model.bias),
        fit=dict(method="Newton-IRLS + L2 ridge (l2=1e-3)", trained_on=a.traces,
                 cap_rows_per_file=a.cap, n_train_rows=int(tr.size),
                 n_workloads=len(corpus["files"])),
        reference_metrics=res,
    )
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(ref, open(a.out, "w"), indent=2)
    print("REFERENCE MODEL:")
    print(f"  weights: within={ref['weights']['s_within']:.4f}  cross={ref['weights']['s_cross']:.4f}  "
          f"bias={ref['bias']:.4f}")
    print(f"  protocol metrics: AUC={res['auc']:.4f} CI{res.get('auc_ci95')}  "
          f"AUPRC={res['auprc']:.4f}  P@10={res['p_at_k']:.3f}  ECE={res['ece']:.4f}")
    print("WROTE", a.out)


def _z(F):
    F = np.asarray(F, np.float32).copy(); F[:, 2] = 0.0; F[:, 3] = 0.0
    return F


if __name__ == "__main__":
    main()

"""Write the synthetic prediction table that ships as the protocol-2.0 example.

The table has the schema ``benchmark.protocol.TABLE_COLUMNS`` and needs no traces or
model weights. It is SYNTHETIC: a lognormal latent "future attention" per candidate
block, the label marks the top ``ceil(0.1 n)`` of each decision, and the score is a
noisy copy of the latent, so every decision is complete exactly as the protocol
requires. It exists to show the schema and to let a third party check the evaluator
(``expected_results.json``); it carries no empirical claim.

    python benchmark/example/make_example_table.py   # rewrites the two files
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmark import protocol as P  # noqa: E402

HERE = Path(__file__).resolve().parent


def make_table(seed: int = 0, models=("model-a", "model-b"), n_prompts: int = 8, layers: int = 3,
               steps: int = 4) -> dict:
    rng = np.random.default_rng(seed)
    cols = {k: [] for k in P.TABLE_COLUMNS}
    for model in models:
        for p in range(n_prompts):
            for layer in range(layers):
                for step in range(steps):
                    n = int(rng.integers(20, 41))
                    latent = rng.lognormal(0.0, 1.0, size=n)            # future attention
                    k = int(np.ceil(P.TOP_R * n))
                    label = np.zeros(n, np.int8)
                    label[np.argsort(-latent)[:k]] = 1                  # exactly ceil(r n) positives
                    score = 1.0 / (1.0 + np.exp(-(np.log(latent) + rng.normal(0.0, 0.8, size=n) - 1.5)))
                    cols["model_id"] += [model] * n
                    cols["source_id"] += [f"synthetic/q{p:02d}"] * n
                    cols["request_id"] += [p] * n
                    cols["layer"] += [layer] * n
                    cols["step"] += [step] * n
                    cols["block_idx"] += list(range(n))
                    cols["label"] += label.tolist()
                    cols["score"] += score.tolist()
    return {k: np.asarray(v) for k, v in cols.items()}


def main():
    table = make_table()
    P.write_table(HERE / "prediction_table.csv", table)
    # evaluate what a reader will actually load back from the CSV
    res = P.evaluate_table(P.read_table(HERE / "prediction_table.csv"), n_boot=200, seed=0, probabilistic=True)
    (HERE / "expected_results.json").write_text(json.dumps(res, indent=2) + "\n")
    print(f"wrote {HERE / 'prediction_table.csv'} ({len(table['label']):,} rows) and expected_results.json")
    print(json.dumps({k: res[k] for k in ("n_rows", "n_decisions", "n_requests", "n_source_prompts")}))
    print("pooled", {k: round(v, 4) for k, v in res["pooled"].items()}, "| same-decision AUC",
          round(res["pairs"]["auc_same_decision"], 4), "| recall@10%", round(res["per_decision"]["0.10"]["macro_recall"], 4))


if __name__ == "__main__":
    main()

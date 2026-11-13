# KVSalienceBench

KVSalienceBench evaluates methods that decide **which parts of an LLM's KV cache
to keep**. Compare their scores and test how cache selection affects answer
quality and memory use.

Code for *Mining Attention Dynamics: Auditing KV-Saliency Prediction in LLMs*.

## Quick start

Use **Python 3.11**. This example runs on a CPU with bundled synthetic data;
no GPU, model weights or dataset downloads are needed.

```bash
git clone https://github.com/lzbaclz/KVSalienceBench.git
cd KVSalienceBench
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e .

python -m benchmark.run_leaderboard \
  --table benchmark/example/prediction_table.csv \
  --manifest benchmark/example/candidate_manifest.json
```

You will see ranking scores and recall at different cache budgets. For this
example, **pooled AUC is about 0.888** and **recall@10% is about 0.529**.
The latter means the scorer recovers about 53% of the blocks labeled important
when it can retain only 10% of the candidates. Higher recall is better.

## Evaluate your own scores

Prepare a CSV with these columns, one row per candidate cache block:

```text
model_id,source_id,request_id,layer,step,block_idx,label,score
```

`label` is 1 for blocks in the top 10% by future attention and 0 otherwise.
A higher `score` means your method prefers to keep the block. Include every
candidate for each decision. See the [input format and label rules](benchmark/DATASHEET.md).

```bash
mkdir -p results
python -m benchmark.run_leaderboard \
  --table my_predictions.csv --json results/metrics.json
```

For your own data, a matching `--manifest` can also verify that no candidates
are missing. Use `--probabilistic` if your scores are probabilities and you want
calibration metrics.

## Go further

| I want to… | Start here |
| --- | --- |
| Understand the metrics and data splits | [Benchmark guide](benchmark/DATASHEET.md) |
| Evaluate a custom scoring function on attention traces | [Scorer template](benchmark/submit_template.py) |
| Run the simulator or its CPU tests | [Masked-loop guide](docs/REPRODUCING_MASKED_LOOP.md) |
| Measure answer quality and actual KV memory use | [Physical KV guide](docs/PHYSICAL_KV_VALIDATION.md) |

Experiments with real models require your own local checkpoints and datasets.
The guides cover their extra dependencies and supported runtimes.

[Apache-2.0 license](LICENSE) · [Third-party licenses](third_party/README.md) ·
[Citation](CITATION.cff) · [Contributing](CONTRIBUTING.md)

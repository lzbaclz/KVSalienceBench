# KVSalienceBench

Code for **Mining Attention Dynamics: Auditing KV-Saliency Prediction in LLMs**.
The benchmark evaluates KV-cache saliency scores using pooled and per-decision
metrics, and provides attention-trace collection and physical-KV validation tools.

## Install

Use Python 3.11 for the pinned validation runtime.

```bash
git clone https://github.com/lzbaclz/KVSalienceBench.git
cd KVSalienceBench
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
```

For the masked-loop and physical-cache CPU tests, install PyTorch first:

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[test,validation]'
```

GPU experiments require a suitable CUDA PyTorch build and locally supplied model
weights and datasets. The validation adapter uses Transformers **4.51.3** and
supports Llama/Qwen2, eager attention, batch size one and a single device.

## Run the benchmark example

The included prediction table is synthetic and needs no model or external data.

```bash
python -m benchmark.run_leaderboard \
  --table benchmark/example/prediction_table.csv \
  --manifest benchmark/example/candidate_manifest.json --probabilistic
```

The evaluator reports pooled AUC/AP, same-decision and cross-decision pair AUC,
per-decision recall at 10% and 20% retention, tie sensitivity, and optional
calibration metrics. A candidate manifest verifies the candidate inventory;
a prediction table alone cannot verify how a scorer was trained.

See [the benchmark protocol](benchmark/DATASHEET.md) and
[the submission template](benchmark/submit_template.py). With local version-2
traces, `--traces` and `--submission` evaluate a custom scorer. The default split
holds out source prompts shared across models.

## Test

```bash
python scripts/prepare_seer.py
export KVSALIENCE_SEER_ROOT="$PWD/build/seer-5a7fbee19045"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 pytest -q
python experiments/seer_pinned/smoke_cpu.py
python scripts/check_public_tree.py
```

The simulator smoke test needs the `validation` extra. Tests use synthetic inputs
and tiny random models; they do not require model downloads, a GPU, or archived
experiment results.

## Collect and validate

- [Physical KV runbook](docs/PHYSICAL_KV_VALIDATION.md): prepare local QA inputs,
  compare logical masking with irreversible storage compaction, and inspect
  numerical gates, answer F1, tensor storage and local latency.
- [Pinned masked-loop runbook](docs/REPRODUCING_MASKED_LOOP.md): verify the bundled
  simulator, check external inputs, and run individual cells or a sweep.
- `scripts/collect_v2_traces.sh`: collect version-2 attention traces from local
  Llama/Qwen2 checkpoints.
- `experiments/`: analysis and experiment drivers. Drivers that consume prior
  results require you to supply those inputs; outputs belong in ignored local
  directories such as `experiments/results/` or `build/`.

Logical masking, physical compaction and production serving measure different
things. This repository provides benchmark and reference-validation code; local
generation measurements do not establish production throughput gains.

## Repository contents

| Path | Purpose |
| --- | --- |
| `xqp/` | Predictors, metrics, feature extraction and cache implementations |
| `benchmark/` | Evaluation protocol, reference coefficients, examples and split manifests |
| `experiments/` | Experiment and analysis code, with small predictor configurations |
| `scripts/` | Collection, validation and dependency preparation commands |
| `tests/` | Algorithm, protocol and runtime correctness tests |
| `configs/` | Runtime settings and external-input checksums |
| `third_party/` | Pinned simulator source, source hashes and required license notices |
| `docs/` | Instructions for running the code |

Only code-related files are published here. Manuscripts, slides, review records,
generated experiment outputs, model weights and raw datasets stay outside Git.
The [contribution policy](CONTRIBUTING.md), ignore rules and CI content check
enforce this boundary.

## License and citation

Project code is licensed under [Apache-2.0](LICENSE). The bundled SEER source keeps
its MIT license, and LongBench-derived scoring code retains its
[notice](third_party/LongBench-NOTICE.md). See [CITATION.cff](CITATION.cff) for
software citation metadata.

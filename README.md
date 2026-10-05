# KVSalienceBench

Code for **Mining Attention Dynamics: Auditing KV-Saliency Prediction in LLMs**.

## TL;DR

KVSalienceBench audits whether KV-cache saliency predictors and their offline
metrics explain downstream answer quality. It is a measurement artifact, not a
claim of a faster production serving system.

**Start here:** [paper](paper_icdm/main.pdf) · [English slides](paper_icdm/output/final_presentation_en.pdf) ·
[physical-KV runbook](docs/PHYSICAL_KV_VALIDATION.md) · [validation status](docs/VALIDATION_STATUS.md) ·
[artifact pointers](docs/ARTIFACT_POINTERS.md).

## Overview

The main offline evidence uses **version-2 traces: 30,445,332 labeled rows** from
Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct. A three-parameter two-view logistic
scorer achieves pooled AUC **0.890**, versus **0.896** for LightGBM. At each
decision's top-decile budget, the within-layer signal recovers **0.659** of future
top-decile blocks, versus **0.613** for the logistic scorer and **0.615** for
LightGBM. The ranking depends on the evaluation budget and aggregation: pooled AUC
is 99.998% cross-decision pairs, and the logistic scorer's binary cross-layer term
raises it by **+0.017** while lowering within-decision AUC (**−0.0025**) and
top-decile recall (**−0.046**); the exact split is in
[`icdm_v2_decomposition.json`](experiments/results/icdm_v2_decomposition.json).

A separate **pinned bfloat16 masked-loop rerun (v2)** evaluates the
**runtime-reconstructed** scorer over 448 source prompts on two models (896
model–prompt evaluations, seven datasets). Official all-reference LongBench F1 is
**0.374** versus **0.372** for the H2O-style accumulator (unrounded difference
**+0.0017**). The primary analysis averages the two models within each dataset and
runs a paired TOST over the seven datasets: **p=0.0023** at a ±0.02 margin, 90% t
interval **[−0.0064, +0.0098]**; the 14-cell sensitivity analysis gives **p=0.0007**,
**[−0.0063, +0.0097]**. The margin is a practical tolerance, not pre-registered.
Both policies remain below full-cache F1 **0.419**. This is mean equivalence
between two complete runtime policy realizations on these tasks; it is not a
scorer-only intervention, per-task equivalence or a physical-eviction performance
result.

These numbers are read from [`icdm_v2.json`](experiments/results/icdm_v2.json)
and [`expand_v2_sensitivity.json`](experiments/results/tost/expand_v2_sensitivity.json).
The offline predictor and runtime reconstruction are different realizations.
LightGBM's 150 is a tree count, not a scalar-parameter count.

These findings motivate direct task-quality checks of proxy rankings. They do
not establish that no better learned or query-aware selector can exist.

### Historical version 1

The unpinned archive contains **43,931,784 rows** from four 7–8B instruction-tuned
families, with two-view AUC **0.948** versus **0.949** for LightGBM. Its masked-loop
F1 difference **+0.00511** and TOST **p=0.001066** use the earlier single-reference
scorer and a float16 Qwen path subsequently found to emit invalid logits. These
values are retained for provenance; the manuscript's main evidence is v2 above.
Multi-workload, 14B and 16K extensions are separately scoped historical checks.

### Claim boundaries and implementation audit

The simulator measures selector behavior and generated-answer F1, not HBM
reclamation, paging, DMA contention, concurrency, production TPOT or throughput.
Its reconstructed cross-layer feature is not identical to the offline predictor's
feature realization. Feature changes confound attribution of the offline-to-loop gap.
The main evidence and the smaller 14B/16K checks must not be read as validation of
MoE, base checkpoints, ≥32K contexts or temperature sampling.

The available legacy collector mixes **pre-RoPE queries with post-RoPE keys**,
refeeds the last prompt token, and clamps unavailable future horizons. The old
producer revision is not pinned, so the affected historical trace cohort cannot
be certified from aggregate results alone. Old values remain archived diagnostics;
they are **not** relabeled as corrected results or a faithful Quest control.
The version-2 collector fixes these paths and records provenance. Llama-3.1 and
Qwen2.5 phase-aligned recollections are reported in the paper; they are not a silent
replacement of archived pooled-AUC numbers. The old exception-swallowing HBM probe is retired,
with its source preserved in `experiments/legacy/`.

See the [availability statement](paper_icdm/DATA_AVAILABILITY.md),
[validation status](docs/VALIDATION_STATUS.md) and [manuscript](paper_icdm/).

## Installation

```bash
git clone https://github.com/lzbaclz/KVSalienceBench.git
cd KVSalienceBench
python -m pip install -e '.[test]'
```

For the complete artifact checks, including the bundled simulator's CPU example,
follow the [CPU installation recipe](docs/REPRODUCING_MASKED_LOOP.md): install
the CPU PyTorch wheel first, then `python -m pip install -e '.[test,validation]'`.
The validation extra includes the simulator's pandas dependency.

For physical-KV validation, use a **separate environment**, install the appropriate
CUDA-enabled PyTorch wheel, then `python -m pip install -e '.[test,validation]'`.
The adapter pins Transformers **4.51.3** and supports Llama/Qwen2, batch size one,
a single device, eager attention, and non-quantized K/V. Other cache APIs and
model families are rejected rather than silently approximated. See the
[runbook](docs/PHYSICAL_KV_VALIDATION.md) for the complete CUDA setup.

## Data

Large attention traces and model checkpoints are not committed. Historical
analysis JSON is tracked under `experiments/results/` and is not overwritten by
new validation. The exact version-2 SEER simulator is bundled in `third_party/`
with its MIT notice. `scripts/prepare_seer.py` checks every source file against
all 70 cell provenance records under `experiments/results/expand_v2/`, and
`experiments/seer_pinned/run_cell.py` rejects mismatched source. The original
version-1 producer remains unpinned and cannot be certified as the same code.

New runs use local model directories and a frozen English-QA JSONL cohort with
`id`, `dataset`, `prompt`, and nonempty `answers`. The preparation utility supports
local LongBench-style rows, deterministic sampling and explicit opt-in truncation.
Model and dataset downloads are not automatic. See the runbook before preparing
or collecting data; corrected traces must use new output paths.

## Reproduce and validate

### Physical KV implementation check

The reference backend physically compacts retained K/V storage, preserves absolute
positions, forbids resurrection and enforces a fixed prompt-derived block budget.
Its `h2o_block` policy is an **H2O-style shared-head block accumulator**, not the
original H2O or Ada-KV implementation. `xqp_reconstructed` uses frozen repository
weights with explicitly documented new online features. Full prefill is still
uncompressed; transient compaction peaks are measured separately from decode.

Every cell first passes native-full-cache and fixed-schedule mask/physical parity
gates. Selection and compaction are included in local generation timing. Failed
runs exit nonzero and write a `.failed.json`, not a successful benchmark record.
Llama-3.1-8B physical-KV and query-v2 GPU cells are in
`experiments/results/physical_kv/`; Qwen2.5-7B uses the same layout in float32
(bfloat16 physical gates failed).

```bash
export MODEL=/absolute/path/to/Llama-3.1-8B-Instruct
export QA_DATA=/absolute/path/to/frozen-qa.jsonl
export OUT=/absolute/path/to/new-physical-results
export CHAT=1 DTYPE=bfloat16 MAX_INPUT=4096 MAX_NEW=128 REPEATS=1
bash scripts/run_physical_kv_matrix.sh
```

The nine-cell matrix uses fresh processes, immutable outputs, hashed checkpoints
and strict paired IDs. It reports live payload, physical tensor storage, allocator
statistics, local latency and answer F1. It does not implement a paged serving
engine, offload, network requests or production goodput. The
[runbook](docs/PHYSICAL_KV_VALIDATION.md) also documents the corrected query-v2
probe and how to interpret small-sample results without overstating equivalence.

### Versioned evidence and reproducibility

The archived Qwen2.5 masked-loop cells were generated in float16,
where the model emits all-NaN logit rows (decoded as "!"); `experiments/
diagnose_qwen_fp16.py` reproduces the mechanism with an unmodified HF loop and
`experiments/seer_pinned/` reruns the complete 14-cell experiment with the external
simulator pinned to one revision in bfloat16. The archived per-item F1 used the
runner's own normalization order and a single reference; `experiments/
analyze_expand_sensitivity.py` re-derives every stored value, re-scores with the
official LongBench scorer and all references, and reports the cluster/margin
sensitivity. `xqp.dm_metrics.average_precision` is now tie-aware (threshold-based),
and top-decile metrics are reported pooled *and* per cache decision. The offline
analysis is repeated on checksum-verified version-2 traces by
`experiments/run_icdm_v2.py`, with a source-prompt-level split shared across
models, and `experiments/analyze_train_size_sensitivity.py` refits both the compact
scorer and the LightGBM comparator at training budgets from the headline 120K rows
up to all 22.8M held-in rows, so the small AUC deficit between them cannot be read
as an under-trained tree. `docs/ARTIFACT_POINTERS.md` names the file behind every
"in the artifact" sentence in the manuscript.
Legacy launchers live under `scripts/legacy/`; `scripts/collect_v2_traces.sh`
is the fail-closed version-2 entry point.

### CPU tests and paper build

```bash
python scripts/prepare_seer.py
export KVSALIENCE_SEER_ROOT="$PWD/build/seer-5a7fbee19045"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 pytest -q
python experiments/fig_redund_calib.py
python experiments/fig_cell_forest.py
# Only after editing the diagram (requires draw.io Desktop and Xvfb headlessly):
bash scripts/export_diagrams.sh
(cd paper_icdm && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex)
python scripts/check_camera_ready.py
```

Random tiny-model tests check implementation correctness, not empirical task
quality. Figure-label rebuilds reuse frozen values. CI independently tests the
pinned runtime and compiles the standard IEEE-format manuscript.

The architecture overview has an editable [draw.io source](paper_icdm/figures/evaluation_paths.drawio)
and a PDF exported by draw.io. Other manuscript figures are generated by Python.
See the [figure source map](paper_icdm/figures/README.md) and
[rendered layout audit](docs/FIGURE_QA.md).

To rebuild the English presentation, install `python -m pip install -e '.[paper]'`
and run `python paper_icdm/build_deck.py`. With LibreOffice installed:

```bash
libreoffice --headless --convert-to pdf --outdir paper_icdm/output paper_icdm/output/final_presentation_en.pptx
```

## Score a prediction table

The default benchmark entry scores pooled and per-decision metrics together from a
prediction table (columns `model_id, source_id, request_id, layer, step, block_idx,
label, score`), with no traces or model weights. Every decision must be complete
(exactly ⌈0.1·n⌉ positives); the evaluator refuses a row sample. The shipped example is
synthetic.

```bash
python -m benchmark.run_leaderboard --table benchmark/example/prediction_table.csv --probabilistic
python -m benchmark.run_leaderboard --traces Llama=…jsonl,Qwen=…jsonl --submission my_method.py
```

See [`benchmark/DATASHEET.md`](benchmark/DATASHEET.md) and `benchmark/submit_template.py`.

## Content

| Path | Purpose |
|---|---|
| `benchmark/` | Decision-level protocol 2.0 (default), prediction-table example, reference baseline and data card; protocol 1.0 kept in `legacy_v1/` |
| `xqp/` | Predictors, statistics, corrected collector and physical-KV backend |
| `experiments/` | Archived analyses, prospective validation and paired analysis |
| `scripts/run_physical_kv_matrix.sh` | Serial matched-input GPU matrix |
| `tests/` | CPU correctness, provenance and failure-path tests |
| `paper_icdm/` | LaTeX, figures, compiled paper and reviewer response |
| `docs/` | Evidence audit, operator runbook, artifact pointers and validation status |

## License

Apache 2.0 for this repository; the pinned SEER source in `third_party/` retains
its upstream MIT license and copyright notice. Model and dataset licenses remain separate.
The [release inventory](docs/RELEASE_AUDIT.md) identifies excluded raw data and
internal materials. Zenodo archiving is deferred to the authors; no DOI is claimed.

The [masked-loop runbook](docs/REPRODUCING_MASKED_LOOP.md)
includes a complete offline CPU example, a hash-verified simulator bundle and
GPU reproduction instructions. The [D2AI change summary](paper_icdm/RESPONSE_D2AI_2026.md)
describes how the four reviewer concerns are reflected in the manuscript.

## Related Projects

[H2O](https://github.com/FMInference/H2O),
[Quest](https://github.com/mit-han-lab/quest),
[SnapKV](https://github.com/FasterDecoding/SnapKV),
[StreamingLLM](https://github.com/mit-han-lab/streaming-llm),
[PyramidKV](https://github.com/Zefan-Cai/KVCache-Factory),
[LongBench](https://github.com/THUDM/LongBench).

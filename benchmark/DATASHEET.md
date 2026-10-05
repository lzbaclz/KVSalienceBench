# KVSalienceBench data card

## Current evidence: version 2

The main offline corpus has **30,445,332 labeled block rows**, collected from
Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct on the same 128 English QA prompts
(64 Qasper and 64 NarrativeQA): 256 model-requests. The collector records 32
feature steps and genuine lookahead for horizons 1, 4, 16 and 64, using 32-token
blocks. Queries and cached keys are aligned after RoPE and decoding starts from
the prefill-predicted token. Data and model hashes accompany the analysis in
`experiments/results/icdm_v2.json`.

The label is future top-decile block attention among candidates present at the
feature step. The recorded views are within-layer attention EMA, a previous-layer
hot-block indicator, a cosine query proxy and block age. The dot-max control is a
separate phase-aligned probe, not Quest's page-bound algorithm.

## Protocol 2.0 (default benchmark entry)

`benchmark/protocol.py` scores a retention scorer at the object a selector acts on, the
cache **decision** (one request, layer and decode step). It reports together: pooled AUC,
tie-aware AP and pooled top-k; the same pooled AUC split exactly into same-decision and
cross-decision pairs (same-decision pairs are 0.0016% of all pairs on the version-2
corpus); per-decision top-k recall at 10% and 20% retention, with the share of decisions
whose k-th place is a tie and the recall under exact random tie-breaking; and, only for
scorers declared probabilistic, ECE and Brier. Uncertainty is a bootstrap over source
prompts. The default split holds out whole source prompts, the same ones for every model;
`split='request'` reproduces the paper's pooled request split (a global permutation, not
stratified by model: 30 Llama and 34 Qwen requests are held out).

A **prediction table** (CSV, columns `model_id, source_id, request_id, layer, step,
block_idx, label, score`) needs no traces or model weights. Every decision must hold
exactly ceil(0.1 n) positives; the evaluator refuses a row sample because per-decision
metrics are meaningless on a few rows per decision.

```bash
python -m benchmark.run_leaderboard --table benchmark/example/prediction_table.csv --probabilistic
```

`benchmark/example/` ships a **synthetic** table (5,748 rows, 192 decisions) and the
results `evaluate_table` must reproduce (`tests/test_benchmark_v2.py`); it carries no
empirical claim. `benchmark/reference_model_v2.json` holds the paper's three-parameter
within+cross logistic refit as the baseline to compare against; `run_leaderboard --traces`
refits it on the training rows of the evaluated split so it never sees a test prompt.

## Splits and metrics in the paper's analysis

`experiments/run_icdm_v2.py` is the analysis entry point behind Table II. It samples
120,000 training and 150,000 held-out rows for pooled metrics and evaluates all
held-out rows at per-decision budgets. Request-disjoint and source-prompt-disjoint
splits are both reported. Source prompts shared across models are resampled jointly
for uncertainty estimates in `experiments/results/mentor_statistics.json`. The exact
pair-count decomposition behind Table II's lower block is
`experiments/analyze_decision_decomposition.py` (`xqp/decision_eval.py`).

Report pooled AUC, threshold-based AP, pooled and per-decision top-decile
precision/recall, and calibration (ECE/Brier). The three-parameter logistic scorer
has pooled AUC 0.890 versus LightGBM's 0.896; within-decision top-decile recall is
0.613 versus 0.615 and the raw within-layer signal's 0.659. The ordering changes
with the metric and budget. Low ECE is not specific to a model family: unweighted
and recalibrated tree controls are included.

## Separate downstream experiment

The pinned bfloat16 masked-loop study covers two architectures, seven LongBench QA
datasets and 896 model–prompt evaluations (448 source prompts × 2 models) at 20% logical retention. Its runtime scorer
reconstructs features rather than executing the offline predictor unchanged.
Official all-reference F1 is 0.374 versus the H2O-style accumulator's 0.372
(difference +0.0017; seven-dataset TOST p=0.0023 at ±0.02, the primary analysis; 14-cell
sensitivity p=0.0007). This is separate evidence,
not a jointly measured causal proxy-to-quality relationship. Logical masking,
irreversible reference compaction and production serving have different semantics.

## Historical compatibility

The 43,931,784-row v1 trace cohort spans four instruction-tuned model families.
Workload, 14B and 16K extensions are separate historical collections. Its producer
is unpinned; query-coordinate and terminal-label issues are documented in the
paper's limitations. Archived JSON remains unchanged. The version-1 protocol, reference
weights and leaderboard helpers are kept, marked legacy, in `benchmark/legacy_v1/`; they
report pooled metrics only and must not relabel old numbers as version 2.

## Access and scope

The code release contains result records and sufficient statistics, not source
input corpora, model weights or the optional roughly 9 GiB raw v2 trace package.
Obtain source datasets and models from their upstream providers using the checksums
and preparation instructions in `docs/REPRODUCING_MASKED_LOOP.md`. Request the
optional trace package from the corresponding author (`chenjx@hust.edu.cn`).
Experimental result records retain model predictions and scoring references for
reanalysis. Third-party source terms continue to apply to those references.

No conclusion extends to MoE, base models, at least 32K contexts or temperature
sampling. Small query-probe gains do not establish intrinsic query redundancy.

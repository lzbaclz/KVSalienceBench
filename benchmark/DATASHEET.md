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

## Splits and metrics

`experiments/run_icdm_v2.py` is the current analysis entry point. It samples
120,000 training and 150,000 held-out rows for pooled metrics and evaluates all
held-out rows at per-decision budgets. Request-disjoint and source-prompt-disjoint
splits are both reported. Source prompts shared across models are resampled jointly
for uncertainty estimates in `experiments/results/mentor_statistics.json`.

Report pooled AUC, threshold-based AP, pooled and per-decision top-decile
precision/recall, and calibration (ECE/Brier). The three-parameter logistic scorer
has pooled AUC 0.890 versus LightGBM's 0.896; within-decision top-decile recall is
0.613 versus 0.615 and the raw within-layer signal's 0.659. The ordering changes
with the metric and budget. Low ECE is not specific to a model family: unweighted
and recalibrated tree controls are included.

## Separate downstream experiment

The pinned bfloat16 masked-loop study covers two architectures, seven LongBench QA
datasets and 896 paired model-requests at 20% logical retention. Its runtime scorer
reconstructs features rather than executing the offline predictor unchanged.
Official all-reference F1 is 0.374 versus the H2O-style accumulator's 0.372
(difference +0.0017; 14-cell TOST p=0.0007 at ±0.02). This is separate evidence,
not a jointly measured causal proxy-to-quality relationship. Logical masking,
irreversible reference compaction and production serving have different semantics.

## Historical compatibility

The 43,931,784-row v1 trace cohort spans four instruction-tuned model families.
Workload, 14B and 16K extensions are separate historical collections. Its producer
is unpinned; query-coordinate and terminal-label issues are documented in the
paper's limitations. Archived JSON remains unchanged. The `benchmark/protocol.py`,
reference weights and leaderboard helpers preserve that historical interface;
they are not the v2 analysis driver and should not relabel old numbers as v2.

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

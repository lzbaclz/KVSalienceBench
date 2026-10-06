# Frozen configuration of the version-2 evidence

One place for every constant behind the version-2 results (Table II, Figure 2, the
pooled-AUC split) and the constants of the two runtime paths, with the code that holds
each. Nothing here is a tuning knob that was searched: the values are the defaults the
collector, the fits and the analyses ran with. `tests/test_frozen_config.py` checks the
table against the code defaults and the result records, so it cannot drift silently.

## 1. Corpus and labels

Collector: `xqp/attn_trace_extract.py` (`extract_attention_traces`), driven by
`scripts/collect_v2_traces.sh`. Per-trace manifests:
`experiments/results/physical_kv/query-v2/*.meta.json` and `*.cohort.json`.

| Quantity | Value |
|---|---|
| Models | Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct (weight hashes in the cohort files) |
| Prompts | The same 128 LongBench QA prompts for both models: 64 Qasper, 64 NarrativeQA, a seeded sample (seed 0), chat template |
| Input cap | 4096 tokens, head/tail truncation (104 of the 128 prompts are truncated). A candidate set can exceed 4096/32 = 128 blocks once decoding adds a block: 64 to 129 candidates per held-out decision, median 129 |
| Block | 32 tokens; a ragged last block is averaged over the tokens it has |
| Decode | Greedy; 32 feature steps, then 64 further steps that only supply labels; EOS is not a stopping rule; decoding starts from the token the prefill predicts |
| Attention statistic | Attention of the token being decoded, averaged over heads and over the tokens of a block |
| Label | 1 if the block is among the top `ceil(0.1 n)` of block attention `h` steps later, ranked among the `n` blocks present at the feature step; `h` in {1, 4, 16, 64}, headline `h = 4` |
| Label ties | `numpy.argpartition` (`xqp.features.topk_indicator`); exactly `ceil(0.1 n)` positives per decision; not randomized |
| Rows | 30,445,332 (16,215,424 Llama, 14,229,908 Qwen), 256 model-requests |

## 2. Features (`xqp/features.py::extract_features`)

| Feature | Definition | Constant |
|---|---|---|
| EMA | `ema <- 0.9 * ema + 0.1 * attention`, initialized at the first decode step's value; a block created later starts at its first value | decay 0.9 |
| within-layer view | `ema / (max over the decision's blocks + 1e-9)` | |
| cross-layer view | 1 if the block is among the top `ceil(0.1 n)` of the PREVIOUS layer's EMA, else 0. Layer 0 has no predecessor and takes the same indicator from its own EMA. Always binary, never the continuous magnitude | `r_cross = 0.10`, `cross_signal = "indicator"` |
| query proxy | `(1 + cos(q, K_b)) / 2`, with `q` the head-mean query of the decoded token after RoPE and `K_b` the head-mean, block-mean key | |
| age proxy | `exp(-(t - p_b) / w)`, `p_b` the first step at which the block exists (0 for every prompt block) | `w = 64` steps |

Because every prompt block has `p_b = 0`, the age proxy takes one value on all of them;
see `docs/HISTORICAL_DIAGNOSTICS.md` section 13 for what that does to a top-k rule.

## 3. Fits

| Model | Configuration | Code |
|---|---|---|
| Two-view logistic (three parameters) | Features: within and cross (the other two columns zeroed). Objective: mean log-loss plus `(l2 / 2N) * ||w||^2` with `l2 = 1e-3` on the two coefficients; the intercept is not penalized. Newton-IRLS from zero, `1e-6` added to the Hessian diagonal, stop when the objective changes by less than `1e-6` or after 40 iterations. No class weights. Features are not standardized (both lie in [0, 1]) | `xqp/predictor.py::ClosedFormXQP._fit_single` |
| LightGBM (headline) | `LGBMClassifier(max_depth=3, n_estimators=150, class_weight="balanced", random_state=0)` on all four views, other parameters at the library defaults (lightgbm 4.7.0 in the pinned environment) | `xqp/baselines.py::fit_gbdt` |
| Calibration controls | The same trees without class weights; isotonic regression fitted on a quarter of the training requests (48 of 192), trees on the other 144 | `experiments/run_icdm_v2.py::fit_calibration_variants` |
| Capacity controls | Six configurations at 120,000 and 1,920,000 rows: depth 3 / 150 trees balanced (headline) and unweighted; depth 3 / 600 trees; depth 6 / 150 trees; depth 8 / 300 trees; 63 leaves with unbounded depth / 500 trees | `experiments/analyze_gbdt_capacity.py` |
| Archived two-view checkpoint | Weights 34.76 (within), 3.35 (cross), bias -3.60; sha256 `4696ed039d78...` | `experiments/predictors/xqp_closed_2view_h4.json` |

Fitted two-view coefficients: 97.67 (within), 2.465 (cross), bias -3.227 on the request
split (`benchmark/reference_model_v2.json`); 88.08, 2.492, -3.213 on the source-disjoint
split.

## 4. Splits, samples and seeds

| Quantity | Value | Code / record |
|---|---|---|
| Seed | 0 everywhere | `run_icdm_full.SEED`, `benchmark.protocol.SEED` |
| Request split (Table II) | 25% of the 256 model-requests by one permutation: 64 held out (30 Llama, 34 Qwen) from 57 source prompts, 50 of which also occur in training through the other model | `run_icdm_v2.request_split_v2`, `benchmark.protocol.split(kind="request")` |
| Source-disjoint split (evaluator default) | 25% of the 128 source prompts (32); both models' requests held out (64) | `benchmark.protocol.split(kind="source")` |
| Held-out sets | Listed request by request | `benchmark/splits/paper_v2_splits.json` |
| Training rows | 120,000 sampled from the held-in rows (22,811,344 on the request split) | `run_icdm_v2.TRAIN_N` |
| Pooled-metric rows | 150,000 sampled from the held-out rows | `run_icdm_v2.TEST_N` |
| Per-decision and pair-split rows | Every held-out row: 7,633,988 rows and 61,184 decisions (request split); 7,719,480 and 61,440 (source-disjoint) | `icdm_v2_decomposition*.json` |
| Relevance sample | 400,000 rows, 16 quantile bins | `run_icdm_v2.MI_N`, `xqp/info_theory.py` |

## 5. Metrics and intervals

| Quantity | Value | Code |
|---|---|---|
| Per-decision budget | `k = ceil(r n)`, `r` in {0.10, 0.20}; decision-macro (each decision weighted equally) | `xqp/decision_eval.py::Ranked.topk` |
| Score ties | Stable descending sort: ties by ascending row index, the block index inside a decision | `xqp/dm_metrics.py::_top_k_indices` |
| Pair split | `AUC_pool = w * AUC_in + (1 - w) * AUC_x`, `w = sum_d n+_d n-_d / (sum_d n+_d * sum_d n-_d)`; `AUC_in` is pair-weighted | `xqp/decision_eval.py::pooled_decomposition` |
| AUPRC | Tie-aware (threshold-based) | `xqp/dm_metrics.py::average_precision` |
| ECE | 10 equal-width bins; only for probability outputs | `xqp/dm_metrics.py::expected_calibration_error` |
| Intervals on the pair split and recall gaps | Bootstrap over source prompts (every held-out request of a drawn prompt moves together), 10,000 draws, of the pooled or decision-macro statistic itself | `xqp/decision_eval.py::bootstrap_statistics` |
| Intervals on pooled AUC | Same clustering, 10,000 draws, exact for the pooled statistic | `experiments/analyze_review_statistics.py` |
| Training-size record | Request-cluster bootstrap, 250 draws | `run_icdm_full.N_BOOT` |

## 6. Runtime paths (Table I of the paper)

| Path | Constants | Code |
|---|---|---|
| Masked loop (Exp#8) | Budget `max(1, round(0.2 n))` blocks with mandatory blocks inside it, recomputed every 8 decode steps; 4096-token input cap; at most 48 new tokens; bfloat16; history of the last 32 steps. Reconstructed scorer: 4 sink and 4 recent blocks mandatory, EMA over the last 8 history values with decay 0.9, cross view rebuilt from the previous layer's EMA top 10%, query view fixed at 0.5, age constant 64 steps. When the budget is below the 8 mandatory blocks it keeps the recent blocks, newest first, then the sinks. H2O-style accumulator: half the budget by summed history, the rest by recency | bundled `seer/policy/xqp.py`, `seer/policy/baselines.py`, `seer/eval/sim.py`; `docs/REPRODUCING_MASKED_LOOP.md` |
| Served oracle (Exp#7) | `K = min(n, max(32, floor(0.1 n)))` blocks of the step's unmasked attention, per layer | bundled `seer/trace/schema.py`; `docs/HISTORICAL_DIAGNOSTICS.md` section 6 |
| Physical reference (Exp#6) | Budget `floor(0.2 * ceil(L / 32))` blocks fixed from the prompt; 1 sink and 1 recent block; EMA decay 0.9 seeded from the last 64 prefill queries; irreversible; at most 128 new tokens; bfloat16 (Llama) or float32 (Qwen) | `xqp/physical_kv.py`, `experiments/run_physical_kv.py`; `docs/PHYSICAL_KV_VALIDATION.md` |

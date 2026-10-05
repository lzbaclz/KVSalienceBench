# Historical diagnostics (version-1 pipeline) and other details moved out of the paper

The 8-page camera-ready keeps the conclusions of these measurements and the numbers
the argument needs. This file holds the rest, verbatim from the previous revision, so
nothing was retired silently. `tests/test_paper_numbers.py` checks every number below
against the tracked result JSON (and checks the ones the manuscript still prints
against the LaTeX).

**None of the numbers in sections 1-7 is version-2 evidence.** They come from the
unpinned version-1 pipeline, whose defects are float16 Qwen2.5 NaN logits, a query
extractor that never formed the attention-space product, and an AUPRC that credited
tied scores for row order (paper section "Limitations and artifact audit"). The raw
rows are gone, so none of them can be recomputed under the corrected protocol.
Identifiers match the artifact's and are therefore not in order.

## 1. Exp#1, version-1 corpus

43,931,784 rows from four instruction-tuned families, 32 chat prompts each, positive
rate 0.105, 32 requests held out. The two-view AUC is 0.948 [0.942, 0.954] against
LightGBM's 0.949 (paired deficit -0.0010 [-0.0016, -0.0003]): the ordering the
corrected corpus shows, at a narrower margin.
Source: `experiments/results/icdm_full.json`, `revision_analysis.json`.

## 2. Exp#2, workload extensions and transfer

Extending to chat, multi-/single-document QA, code completion and summarization in
English and Chinese (128 prompts per workload per family) keeps both magnitude views
useful (single-view AUC about 0.88-0.90 and 0.81-0.86), with pooled AUC 0.929 against
0.934 for LightGBM and a gap of about 0.016 AUC on short multi-turn chat. Its AUPRC
used the row-order tie rule and the rows are gone, so the paper does not restate it
(the biased pair 0.777 / 0.803 is pinned in the tests as withdrawn). A shared
standardized four-weight predictor has mean cross-model AUC 0.920 against 0.921
within model (drop 0.0012): offline transfer only.
Source: `icdm_multiworkload.json`, `icdm_full_sharegpt.json`, `icdm_full.json`.

## 3. Exp#3, temporal variation and threshold behavior

In the within-generation split (fit on steps 0-77, test on 102-127) a frozen scorer
keeps AUC 0.948, about the same as an oracle refit, whereas the online IRLS update
reaches 0.919 (evidence against that update, not against online learning); across
forced topic switches, salient-set Jaccard falls to about 0.35 from 0.83 within a
turn. At target miss 0.1 an offline-calibrated split-conformal threshold reaches miss
0.087 at retention 0.264 and the adaptive update 0.100 at the same retention, against
0.274 at retention 0.095 for a fixed threshold of 0.5: the calibrated static threshold
carries the result.
Source: `icdm_full.json` (drift), `drift_multiturn.json`, `icdm_extra.json` (conformal).

## 4. Exp#4, risk-target calibration ("GuardKV")

GuardKV is our name for the risk-target thresholding rule, not a published system. It
turns a target miss rate into a per-layer score threshold from held-out calibration
quantiles, and adds four sink and four most-recent blocks. Measured miss tracks the
target within 0.007 in distribution, with targets of 0.05 and 0.20 giving retention
near 34% and 14%. Against the served-oracle target, however, the reconstructed scorer
needs about full retention for 90% coverage, which is why the version-2 line does not
build on this construction; the order-statistic rule, its off-by-one repair and the
exchangeability caveat are in the artifact.
Source: `coverage_budget.json`, `serving_calib/SUMMARY.json`.

## 5. Exp#5, scoring-only cost

The microbenchmark times 28.7 us at P99.9 for one 4096-vector batch of the four-weight
kernel (P50 20.5 us, P99 24.6 us): the four-view comparator, not the three-parameter
model the findings are about, and it excludes feature extraction, selection and
compaction. Source: `wcet_gpu.json`.

## 6. Exp#7, offline and in-loop targets differ

The archived offline two-view AUC is 0.948, while on the served-oracle calibration set
the reconstructed scorer reaches 0.643 ([0.632, 0.655], per-layer 0.54-0.76). The
paper reports the drop but does not rely on it: the bridge reproduces neither its size
nor its direction on version-2 traces, and target, trajectory and feature realization
change together, so its cause is unidentified. On four LongBench tasks at 20%
retention (48 requests each) the H2O-style accumulator has request-mean served-oracle
miss 0.579 against 0.623 for the reconstructed scorer (paired difference
[0.033, 0.055]) while the per-task F1 intervals all overlap; the confirmatory
comparison is Exp#8, which measures something else.
Source: `served_oracle_ci/c3_served_auc.json`, `c7_oracle_miss_pertask.json`.

**Capacity-forced miss.** Part of any served-oracle miss is arithmetic: a budget below
the oracle set cannot be met. For request j, step t and layer l the capacity floor is

    F_jtl = max(0, 1 - B_actual_jtl / K_jtl),

with B_actual counting mandatory blocks and K the oracle-set size under the
simulator's minimum-32/top-decile rule. Recomputed from logged per-step counts, the
4K-cap mean floor is 0.219, leaving residuals 0.360 and 0.404 and a paired residual
interval still [0.033, 0.055]: absolute miss is not a policy measurement, but the
policy difference survives the correction. Layerwise floors are verified only on the
pinned path; the other caps and budget fractions are in the artifact
(`served_oracle_ci/oracle_budget.*.json`). A scorer trained on the served-oracle
target improves off-policy recall at 20% retention by 0.06 [0.03, 0.10] but worsens
on-policy miss by 0.13 [0.10, 0.16], consistent with feedback-induced shift.
Source: `served_oracle_ci/c6_onpolicy_reversal.json`.

## 7. Exp#9 details removed from the paper

Five alternative Llama prompt subsets give archived means between -0.008 and +0.011 and
are not replications.

A synthetic needle probe is a negative control, not evidence about query-aware
selection: over 64 RULER requests at 16K it returns exactly 0.737 F1 for the
query-anchored, H2O-style and reconstructed policies alike and exactly 0 for recency,
streaming and a fixed SnapKV-style rule, at every retention from 5% to 20% - two
values decided by repeated filler and lexical cues, not by ranking.

## 8. Query-coordinate audit: a sentence removed from the paper

Some legacy long-context drivers also disabled prefill attention, depriving
query-aware selectors of observations while leaving accumulators usable, so neither
those comparisons nor the needle probe rank faithful query-aware implementations.

## 9. Retained-block counts of the pinned rerun (Exp#8)

Logged `per_step_block_count` over the 896 model-prompt evaluations of
`experiments/results/expand_v2/`: mean 25.02 (H2O-style, reconstructed, Ada-KV-style) and
25.01 (PyramidKV-style) retained blocks per decode step, median 26, range 7-26. The
step vectors are identical to the H2O-style policy's on 605/896 (reconstructed), 695/896
(Ada-KV-style) and 508/896 (PyramidKV-style) requests. The remainder differ in length
because the policies generate different numbers of tokens (all of them for the
reconstructed and Ada-KV-style policies; all but 6 for the PyramidKV-style one, whose
6 same-length exceptions differ in counts). Only one prompt per
model (2WikiMQA id 18) has a decode step below the eight mandatory blocks, and every
policy keeps 7 blocks there, so the budget wins over the mandatory set.

## 10. Exp#1 sentences moved out of the paper (version-2 corpus)

The four-view fit is worse (AUC 0.873): dropping its cosine query proxy *raises* AUC by
0.012, and the phase-aligned dot-max probe adds 0.001 (0.891), a pooled near-zero
averaging over opposite signs in the tested configurations (the per-model request-mean
effects, -0.0070 on Llama and +0.0020 on Qwen, are in the paper's limitations). The
archived two-view weights, applied frozen to the version-2 features, give the same
rounded AUC (0.890) with different top-k values and are slightly less calibrated (ECE
0.019 versus 0.006). The fitted MLP (AUC .874, per-decision precision .613, ECE .005),
the two-view + dot-max logistic and the four-view logistic rows of the full table are in
`experiments/results/icdm_v2.json` (`pooled_request_split.table`).
Source: `experiments/results/icdm_v2.json`.

## 11. More numbers moved out of the paper

The float16 Qwen2.5 replay (56 prompts) found all-NaN logit steps in 54 requests: 687 NaN
steps and 739 token-0 emissions in total, against none in bfloat16 or float32
(`experiments/results/qwen_fp16_diagnosis/`). In the pinned rerun the runner's own scorer
gives a reconstructed-minus-H2O-style F1 difference of +0.0024 (14-cell TOST p=0.0016 at
+-0.02; `experiments/results/tost/expand_v2_sensitivity.json`, `f1_stored`).

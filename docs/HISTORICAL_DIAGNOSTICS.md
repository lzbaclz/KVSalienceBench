# Historical diagnostics (version-1 pipeline) and other details moved out of the paper

The 8-page camera-ready keeps the conclusions of these measurements and the numbers
the argument needs. This file holds the rest, verbatim from the previous revision, so
nothing was retired silently. `tests/test_paper_numbers.py` checks every number below
against the tracked result JSON (and checks the ones the manuscript still prints
against the LaTeX).

**None of the numbers in sections 1-7 and 12 is version-2 evidence.** They come from the
unpinned version-1 pipeline, whose defects are float16 Qwen2.5 NaN logits, a query
extractor that never formed the attention-space product, and an AUPRC that credited
tied scores for row order (paper section "Limitations and artifact audit"). The raw
rows are gone, so none of them can be recomputed under the corrected protocol.
Identifiers match the artifact's and are therefore not in order.

Sections 9-11 and 13-17 hold version-2 details that the 8-page text states in one
sentence or one table cell. The 2026-10-06 revision moved the version-1 confidence
intervals of section 1, the online-update, threshold and scoring-cost results of sections
3-5, the feedback-shift check of section 6 and the conditional-information grids of
section 12 out of the manuscript, to make room for the pooled-AUC split, the
served-oracle definition and the scope of the split.

## 1. Exp#1, version-1 corpus

43,931,784 rows from four instruction-tuned families, 32 chat prompts each, positive
rate 0.105, 32 requests held out. The two-view AUC is 0.948 [0.942, 0.954] against
LightGBM's 0.949 (paired deficit -0.0010 [-0.0016, -0.0003]): the ordering the
corrected corpus shows, at a wider margin there (-0.0057).
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

**Served oracle and capacity-forced miss.** The definitions are the evaluator's
(`seer/eval/sim.py`, `seer/trace/schema.py` in the bundled simulator; checked on CPU by
`experiments/audit_oracle_count_mapping.py`):

- *Oracle set.* At a decode step and for each layer, the K blocks with the largest
  attention of that step's UNMASKED forward pass, with

      K = min(n, max(32, floor(0.1 * n')))        (MIN_TOP_K = 32, TOP_K_FRACTION = 0.10)

  where n is the number of blocks present. The logger passes n' = floor(T/32) + 1 for T
  KV tokens, one more than ceil(T/32) at an exact block boundary. At the 4K cap
  floor(0.1 * n') is at most 13, so K = min(n, 32) and the convention is immaterial:
  the logged K is 32 at every recorded step of the four tasks.
- *Kept set.* B = max(1, round(0.2 * n)) blocks with n = ceil(T/32), mandatory blocks
  included (they are inside the budget, not added to it).
- *Miss.* Per step, the mean over populated layers of |oracle minus kept| / K.
- *Floor.* A budget below the oracle set forces a miss of at least

      F = max(0, 1 - B / K)

  at that step. The archive stores rounded layer means (B_log, K_log), not layer arrays,
  so the reported floor is a logged-count diagnostic: exact when B and K are common
  across layers, which the CPU check verifies on the pinned source but which cannot be
  re-verified for the unpinned historical run.
- *Aggregation.* Mean over recorded steps within a request, then an equal-weight mean
  over the 192 requests: the same weights as the miss.

The 4K-cap mean floor is 0.219 (B is at most 26), leaving residuals 0.360 and 0.404.
The floor is identical for the two policies on every request
(`mentor_e2e_confirm.json`: `floors_equal_per_request` true, largest difference 0), so
`(M_A - F) - (M_B - F) = M_A - M_B` and the paired interval [0.033, 0.055] is unchanged
**by algebra**. Subtracting the floor shows how much of the absolute miss no selector
could avoid; it is not an independent robustness check of the policy difference. At the
16K cap the floors are *not* identical per request (largest difference 0.024), so there
the residual difference is reported separately (`mentor_longctx.json`). Layerwise
floors are verified only on the pinned path; the other caps and budget fractions are in
the artifact (`served_oracle_ci/oracle_budget.*.json`). A scorer trained on the served-oracle
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

Supplementary tests of the pinned rerun at a tighter margin of +-0.01 give p=0.047 over
the seven datasets and p=0.045 over the 14 cells; that margin was not set in advance, so
they are not counted (`expand_v2_sensitivity.json`). The archived short-context
Qwen2.5-14B check has F1 0.301 / 0.307 / 0.301 for the accumulator, the reconstructed
scorer and adaptive allocation (`experiments/results/largemodel/`). The old
physical-memory probe, which swallowed decode exceptions and put full prefill in its
peak statistic, is retired; its source and results are kept under `experiments/legacy/`.
No archived number was silently changed by any revision.

The float16 Qwen2.5 replay (56 prompts) found all-NaN logit steps in 54 requests: 687 NaN
steps and 739 token-0 emissions in total, against none in bfloat16 or float32
(`experiments/results/qwen_fp16_diagnosis/`). In the pinned rerun the runner's own scorer
gives a reconstructed-minus-H2O-style F1 difference of +0.0024 (14-cell TOST p=0.0016 at
+-0.02; `experiments/results/tost/expand_v2_sensitivity.json`, `f1_stored`).

## 12. Archived conditional-information diagnostics (version-1 corpus)

Conditional information U_i = I(X_i; Z | X_-i) under nine binning grids (8, 12 or 16
bins for the feature, 3, 5 or 8 for the conditioning set): the within-layer view is
0.133-0.155 nats on every grid; the binary cross-layer estimate is 0.048-0.084 on the six
finer grids and collapses to zero under the three coarsest, which is its grid
sensitivity; the query and age proxies stay at or below 0.006. These estimates motivated
the two-view subset and are not a sufficiency certificate.
Source: `experiments/results/revision_analysis.json` (`A9_ucmi.ucmi_sweep`).

## 13. Tie rule of the per-decision column (version-2 corpus)

Per-decision top-k uses a stable descending sort, so equal scores are resolved by row
order, the block index inside a decision. `experiments/analyze_tie_sensitivity.py`
recomputes every row of Table II's upper block under the exact EXPECTATION of uniform
random tie-breaking, on every held-out row
(`experiments/results/icdm_v2_tie_sensitivity.json`; request split, source-disjoint
split in parentheses):

| Table II row | recall@10%, stable | expected | recall@20%, stable | expected | rows in a tie |
|---|---|---|---|---|---|
| age proxy | 0.208 (0.207) | 0.109 (0.109) | 0.260 (0.296) | 0.208 (0.209) | 99.9% |
| query proxy (cosine) | 0.415 (0.408) | 0.415 (0.408) | 0.570 (0.561) | 0.570 (0.561) | 0.01% |
| prev-layer indicator | 0.610 (0.607) | 0.610 (0.607) | 0.677 (0.680) | 0.653 (0.651) | 100% |
| within-layer EMA | 0.659 (0.656) | 0.659 (0.656) | 0.840 (0.837) | 0.840 (0.837) | 0.0% |
| LightGBM, balanced | 0.615 (0.612) | 0.615 (0.612) | 0.830 (0.828) | 0.830 (0.828) | 46% (48%) |
| two-view logistic, refit | 0.613 (0.610) | 0.613 (0.610) | 0.838 (0.835) | 0.838 (0.835) | 1.6% |
| two-view, archived weights | 0.611 (0.609) | 0.611 (0.609) | 0.838 (0.835) | 0.838 (0.835) | 0.7% |

- **Age proxy.** Every prompt block shares one creation step, so 99.9% of its rows tie
  and every decision has a tie at the k-th place. The stable rule then keeps the lowest
  block indices, which include the first (sink) block: 0.208 is the row order's value,
  not the age signal's. Table II prints the expectation, 0.109, with a dagger.
- **Previous-layer indicator.** Exactly ceil(0.1 n) blocks carry it, so the top decile is
  tie-free; at 20% retention the second decile is filled by row order (0.677 against an
  expected 0.653). The paper prints only its 10% value.
- **The three main scorers** change by at most 1.6e-5 at either budget on the request
  split and by at most 1.1e-4 on the source-disjoint split (LightGBM at 20%).
- **Scope.** This is the expectation of the decision-macro statistic. It is not a bound
  on one random draw or on one decision: a single decision's expected recall moves by up
  to 0.05 (10%) and 0.16 (20%) for LightGBM. Ties in the future-attention labels are not
  randomized: the traces store labels, not the attention values behind them.

The previous revision's sentence "recomputing every per-decision recall under exact
random tie-breaking moves none by more than 2e-5" was true of the three main scorers
only and is corrected in the text.

## 14. The pooled-AUC split on the source-disjoint split (version-2 corpus)

Table II's request split holds out 64 model-requests from 57 source prompts: 7 prompts
with both models' requests held out and 50 whose other model's request is in training.
It measures transfer to new model-requests, not to unseen text. The same analysis on
the source-disjoint split (the same 32 prompts held out for both models; 64 requests,
61,440 decisions, 7,719,480 rows; `experiments/results/icdm_v2_decomposition_source.json`):

| scorer | pooled AUC | within-decision | cross-decision | recall@10% | recall@20% |
|---|---|---|---|---|---|
| within-layer EMA | 0.874 | 0.929 | 0.874 | 0.656 | 0.837 |
| two-view logistic, refit | 0.890 | 0.927 | 0.890 | 0.610 | 0.835 |
| LightGBM, balanced | 0.896 | 0.924 | 0.896 | 0.612 | 0.828 |

Every entry is within 0.004 of the request-split value (the largest change is 0.0032, LightGBM's recall at 10%). Same-decision pairs are again
0.0016% of all pairs; the cross-layer term raises pooled AUC by +0.016 [0.014, 0.018],
lowers within-decision AUC by 0.0025 [0.0023, 0.0028] and top-decile recall by 0.046
[0.043, 0.049] (LightGBM: 0.045 [0.042, 0.048]); at the top-decile boundary it swaps in
22% of the selected blocks, 20% of them positive against 41% of those displaced. The
fitted offset is 0.028 (w_within 88.1, w_cross 2.49).

**Estimand of the recall gaps.** Table II's per-decision values are decision-macro (every
decision weighted equally), and the intervals the paper prints for the gaps are a
source-cluster bootstrap of that same statistic (`paired` in the two decomposition
records): +0.046 [0.042, 0.049] and +0.044 [0.041, 0.047] on the request split.
`mentor_statistics.json` holds the request-macro variant (one mean per request, then
the mean over requests): +0.046 [0.043, 0.049] and +0.044 [0.041, 0.047] on the request
split, +0.047 [0.044, 0.050] and +0.045 [0.043, 0.048] on the source-disjoint split,
positive in all 64 requests on both. The two estimands differ when requests hold
different numbers of decisions; here the point estimates agree to the third decimal.

## 15. Which F1 Table III reports (physical-KV check)

`experiments/analyze_physical_scorer.py` rescored every stored generation of the 18
physical cells with the function Exp#8's analysis uses (official LongBench
normalization, maximum over all references). The stored per-request F1 equals it on
every request of every cell (largest difference 2e-16), so Table III and Table IV use
the same scorer; 46 of the 128 prompts have more than one reference, and scoring against
the first reference alone would lower the full-cache means from 0.223 to 0.208 (Llama)
and from 0.246 to 0.233 (Qwen). Record: `experiments/results/physical_kv/scorer_check.json`.

The tables differ in cohort, not in scorer. Table III uses a seeded sample of 64 rows
from each of Qasper and NarrativeQA, the prompt "Read the passage and answer the
question. Return only the answer." and at most 128 new tokens; Exp#8 uses the first 64
rows of seven datasets, a context-question prompt ending "Answer as concisely as
possible, with no explanation." and at most 48 new tokens. Both cap the input at 4096
tokens by head/tail truncation. In Exp#8 the full-cache F1 on Qasper and NarrativeQA is
0.41 and 0.19 (0.30 on average) against 0.419 over all seven, so most of the gap between
the tables is the choice of datasets. An earlier revision called them "the two hardest"
sets and attributed part of the gap to all-reference scoring; neither was right
(MuSiQue is lower than Qasper in Exp#8, and both tables score against all references).

## 16. Full-cache gap of the pinned rerun (Exp#8)

`experiments/analyze_full_cache_gap.py` gives the full-cache row of Table IV the same
seven-dataset 90% t interval as the other rows
(`experiments/results/tost/expand_v2_full_cache_gap.json`): full minus H2O-style
+0.047 [+0.015, +0.078], full minus reconstructed +0.045 [+0.015, +0.075], full minus
Ada-KV-style +0.055 [+0.019, +0.091], full minus PyramidKV-style +0.069 [+0.033, +0.106].
No equivalence test is attached: the question for that row is the size of the loss.

## 17. What the logged retained-block count is (Exp#8)

`per_step_block_count` is, per decode step, the rounded mean over layers of the size of
the set the policy chose at its last decision (decisions every eight steps). It is a
selected-set size. Tokens decoded since the last decision are not masked until the next
one, and the last block of a prompt can hold fewer than 32 tokens, so the count does not
establish equal numbers of active tokens at every step; that quantity was not logged.

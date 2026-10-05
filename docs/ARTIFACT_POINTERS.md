# Where each "in the artifact" claim lives

The camera-ready moves implementation detail out of the manuscript and into this
repository. Every place the paper says a number or rule "is in the artifact",
this table says which file holds it, so the claim is checkable rather than a
promise.

| Manuscript sentence | Section | Artifact location |
|---|---|---|
| Version-1 ranking/calibration tables | §VIII | `experiments/results/icdm_full.json`, `experiments/results/revision_analysis.json` |
| Table II rows not printed (fitted MLPs, online logistic baseline, two-view + dot-max, four-view and interaction logistic fits) | §V ("left to the artifact"), Table II caption ("further rows") | `experiments/results/icdm_v2.json` → `pooled_request_split.table`; headline numbers in `docs/HISTORICAL_DIAGNOSTICS.md` §10 |
| Horizons 1, 16 and 64 ("recorded only"; the paper reports h=4) | §II-A | `experiments/results/icdm_v2.json` → `pooled_per_view.by_horizon`; labels for all four horizons are in the version-2 traces |
| Allocation rules of the Ada-KV-style and PyramidKV-style comparators (a fixed total budget moved across layers; within-layer selection as H2O-style) | §VII (Exp#8), Table IV | `seer/policy/perlayer_baselines.py` in `third_party/seer-5a7fbee19045.tar.gz` (`AdaKVPolicy`, `PyramidKVPolicy`, `_select_within_layer`) |
| Parity tolerance "0.15 plus 2% of the logit" (bfloat16) | §VI (Exp#6), §IX | `xqp/physical_validation.py` (`_tolerances`, `_compare_logits`) |
| Local inter-token latency of the physical cells (a Table III column until 2026-10-05) | §VI (Exp#6) | `experiments/results/physical_kv/*/summary.json` → `cells[*].tpot_ms_median` |
| Conditional-information grid sensitivity, all binning grids | §III-B | `experiments/results/revision_analysis.json` → `A9_ucmi.ucmi_sweep` |
| Tie rules: pooled row order and within-decision block index | §II-C | `xqp/dm_metrics.py` (`_top_k_indices`, `precision_recall_at_k_grouped`) |
| Training-budget sensitivity of the LightGBM deficit | §V (Exp#1) | `experiments/analyze_train_size_sensitivity.py`, `experiments/results/icdm_v2_train_size.json` |
| Exact split of pooled AUC into same-/cross-decision pairs, offset sweep, boundary swaps, tie sensitivity, train-rows-only relevance | §V (Exp#1), Table II lower block, Fig. 2(a) | `experiments/analyze_decision_decomposition.py`, `xqp/decision_eval.py`, `experiments/results/icdm_v2_decomposition.json` (+ `.decomposition.csv`, `.offset_sweep.csv`) |
| Tree-capacity sensitivity (six LightGBM configurations at 120K and 1.92M rows) | §V (Exp#1) | `experiments/analyze_gbdt_capacity.py`, `experiments/results/icdm_v2_gbdt_capacity.json` |
| Logged retained blocks per decode step (matched retention in counts) | §VII (Exp#8) | `experiments/results/expand_v2/*/*_{h2o,xqp,adakv,pyramidkv}.json` (`per_step_block_count`); summary in `docs/HISTORICAL_DIAGNOSTICS.md` §9 |
| Default benchmark entry: protocol 2.0, prediction-table schema, synthetic example, reference baseline | §I, §IX (Availability) | `benchmark/protocol.py`, `benchmark/run_leaderboard.py`, `benchmark/example/`, `benchmark/reference_model_v2.json`; protocol 1.0 kept as `benchmark/legacy_v1/` |
| Details of every version-1 measurement (Exp#1-5, #7) and other numbers moved out of the text | §VIII | `docs/HISTORICAL_DIAGNOSTICS.md` |
| 30% block-budget physical cells | §VI (Exp#6) | `experiments/results/physical_kv/*/summary.json` |
| Extended parity-gate records | §IX | `docs/PHYSICAL_KV_VALIDATION.md` |
| GuardKV order-statistic rule and its one-based repair | §VIII (Exp#4) | `xqp/budgeter.py` (`_conformal_tau`, `CoverageDrivenBudgeter`), `xqp/guardkv.py` |
| Split-conformal exchangeability caveat | §VIII (Exp#4) | `xqp/conformal.py` module docstring (SCOPE paragraph) |
| Capacity floors at the 16K cap and at f ∈ {0.10, 0.30, 0.50} | §VIII | `experiments/results/served_oracle_ci/oracle_budget.*.json` |
| Capacity-floor block-boundary rules and layerwise invariant check | §VIII | `experiments/audit_oracle_count_mapping.py` |
| Archived version-1 AUPRC under the row-order tie rule | §VIII (Exp#2) | `experiments/results/icdm_multiworkload.json` (withdrawn from the text; see below) |
| Archived masked-loop scores, scorer checks and repair record | §VII | `experiments/results/expand/`, `experiments/results/expand_v2/`, `experiments/results/qwen_fp16_diagnosis/` |

Drivers marked with a script path need the raw version-2 JSONL traces, which are
available by request rather than in the artifact; the frozen JSON they produced is
included, and `tests/test_paper_numbers.py` checks the manuscript against it.

## Numbers moved out of the manuscript (not withdrawn)

The 2026-10-04 revision moved these from the 8-page text into
`docs/HISTORICAL_DIAGNOSTICS.md`, where `tests/test_paper_numbers.py` still checks each
against the result JSON: the version-1 positive rate, workload-extension and transfer
AUCs, the temporal-variation Jaccard values and the split-conformal threshold results,
the float16 NaN-step counts, the runner-scorer TOST, the three Table II rows above, and
the needle-probe values.

The 2026-10-05 revision removed Table III's local inter-token-latency column (median
76.1 / 61.0 / 56.9 / 60.0 / 57.9 ms for Llama and 71.4 / 58.2 / 54.4 / 58.1 / 54.4 ms for
Qwen, in the table's row order). The masked rows are about 20% faster than full cache,
the text never interpreted that, and the paper makes no latency claim; the values remain
in the summary JSON above and `tests/test_paper_numbers.py` still pins them.

## Numbers deliberately withdrawn from the manuscript

- **Exp#2 AUPRC 0.777 / 0.803.** Computed under the version-1 row-order tie rule,
  which credits tied scores for their position in the file. The version-1 rows no
  longer exist, so the pair cannot be recomputed tie-aware. The values remain in
  `icdm_multiworkload.json` and `tests/test_paper_numbers.py` asserts both that
  the JSON still holds them and that the LaTeX no longer prints them.
- **Per-budget capacity floors.** The formula and the 4K-cap result stay in §VIII;
  the f-sweep and the 16K cap are recorded-count diagnostics and live in the JSON
  above.
- **`sections/gated.tex`.** An unpublished draft of a regime-gated selective
  cascade, never `\input` into `main.tex`, which nevertheless shipped in the first
  public export. It is removed, and `scripts/export_public_artifact.py` now
  derives the exported section list from the `\input` lines of `main.tex` so an
  un-input draft cannot leak again. The negative result it described (a faithful
  per-token/per-head-max query signal is individually informative but redundant
  with attention magnitude) is the refutation the manuscript already makes in
  §III-B and §IX; the reusable cascade code stays in `xqp/gated_predictor.py`
  with `experiments/GATED_DESIGN.md` and `experiments/run_gated_eval.py`.

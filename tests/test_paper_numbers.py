"""Tie every headline number printed in the manuscript back to tracked result JSON.

Each case asserts two things at once:

1. the value derived from `experiments/results/**` still rounds to what the paper
   prints (catches silent data regeneration), and
2. that rendering literally occurs in the LaTeX source (catches a prose edit that
   drifts away from the evidence).

Adding a number to the paper without adding it here is how the camera-ready audit
started; keep the two in sync. Values are quoted exactly as typeset, so `.948`
and `0.948` are different strings on purpose.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import mean, median

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments/results"
TEX = ROOT / "paper_icdm"
MIB = 1024 ** 2
GIB = 1024 ** 3


def load(rel: str) -> dict:
    return json.loads((RESULTS / rel).read_text())


def norm(text: str) -> str:
    """Collapse whitespace so a phrase wrapped across source lines still matches."""
    return re.sub(r"\s+", " ", text)


def tex(name: str) -> str:
    return norm((TEX / "sections" / f"{name}.tex").read_text() if name != "main"
                else (TEX / "main.tex").read_text())


@pytest.fixture(scope="module")
def sources() -> str:
    return norm("\n".join([tex("main")] + [tex(p.stem) for p in sorted((TEX / "sections").glob("*.tex"))]))


@pytest.fixture(scope="module")
def hist() -> str:
    """Numbers that the 8-page manuscript no longer prints live in the artifact's
    docs/HISTORICAL_DIAGNOSTICS.md; they are still checked against the result JSON."""
    return norm((ROOT / "docs/HISTORICAL_DIAGNOSTICS.md").read_text())


def _canonical(text: str) -> str:
    """Compare `-0.001`, `-.001` and `.001` on value, not on typography."""
    sign = "-" if text.startswith("-") else ""
    body = text.lstrip("+-")
    return sign + ("0" + body if body.startswith(".") else body)


def check(sources, printed: str, value: float, digits: int, section_hint: str = ""):
    """`value` must round to `printed`, and `printed` must appear in the source."""
    rendered = f"{value:.{digits}f}"
    assert _canonical(rendered) == _canonical(printed), \
        f"{section_hint}: paper prints {printed}, data gives {rendered}"
    assert printed in sources, f"{section_hint}: {printed} no longer appears in the LaTeX source"


# --------------------------- archived offline table --------------------------
HEADLINE = {"age proxy": "recency", "query proxy": "Quest",
            "previous-layer indicator": "InfiniGen", "within-layer attention": "H2O/attn-EMA",
            "online logistic": "OnlineSGD(logloss)", "LightGBM": "GBDT(LightGBM)",
            "fitted MLP": "sklearnMLP(16,)", "within+cross logistic": "within+cross(2)",
            "four-view closed form": "XQP-closed", "pairwise model": "XQP-pairwise",
            "tiny MLP": "XQP-MLP"}
TAB_MAIN = {  # row label -> (AUC, AUPRC, P@k, ECE) exactly as typeset in tab:main
    "age proxy": (".563", ".126", ".144", ".334"),
    "query proxy": (".575", ".142", ".187", ".340"),
    "previous-layer indicator": (".857", ".609", ".743", ".054"),
    "within-layer attention": (".930", ".741", ".661", ".081"),
    "LightGBM": (".949", ".812", ".769", ".132"),
    "fitted MLP": (".947", ".811", ".769", ".003"),
    "within+cross logistic": (".948", ".810", ".769", ".006"),
    "four-view closed form": (".928", ".800", ".766", ".005"),
}


@pytest.mark.parametrize("row", sorted(TAB_MAIN))
def test_archived_table_is_preserved(row):
    """The historical table remains unchanged in the artifact; Table II now
    prioritizes v2 and omits the v1 column to allow an 8 pt table font."""
    table = {r["method"]: r for r in load("icdm_full.json")["pooled"]["headline"]["table"]}
    r = table[HEADLINE[row]]
    printed_auc = TAB_MAIN[row][0]
    assert f"{r['auc']:.3f}"[1:] == printed_auc


def test_corpus_scale_and_split(sources, hist):
    s = load("icdm_full.json")["pooled"]
    assert f"{s['summary']['n_rows']:,}" == "43,931,784" and "43,931,784" in sources
    assert s["headline"]["n_train"] == 120000 and s["headline"]["n_test"] == 150000
    assert s["headline"]["n_test_requests"] == 32
    check(hist, "0.105", s["summary"]["pos_rate"]["h4"], 3, "positive rate (moved to the artifact doc)")
    assert len(load("icdm_full.json")["models"]) == 4


def test_two_view_interval_and_operating_point(sources):
    table = {r["method"]: r for r in load("icdm_full.json")["pooled"]["headline"]["table"]}
    two, gbdt = table["within+cross(2)"], table["GBDT(LightGBM)"]
    check(sources, "0.942", two["auc_lo"], 3, "two-view CI lo")
    check(sources, "0.954", two["auc_hi"], 3, "two-view CI hi")
    a11 = load("revision_analysis.json")["A11_A12_A27"]["A11"]["auc"]
    check(sources, "-0.0010", a11["delta_2view_minus_gbdt"], 4, "paired AUC delta")
    check(sources, "-0.0016", a11["ci95_lo"], 4, "paired AUC delta lo")
    check(sources, "-0.0003", a11["ci95_hi"], 4, "paired AUC delta hi")
    assert two["auc"] < gbdt["auc"], "the paper describes a small deficit, not a tie"


def test_single_view_and_baseline_probe_aucs_are_distinct_objects():
    """Preserve the unresolved historical discrepancy; fitting alone cannot
    explain AUC changes. This comparison was withdrawn from the paper."""
    pooled = load("icdm_full.json")["pooled"]
    table = {r["method"]: r for r in pooled["headline"]["table"]}
    views = pooled["per_view"]["h4"]
    assert table["Quest"]["auc"] != views["s_query"]["auc"]
    assert table["InfiniGen"]["auc"] != views["s_cross"]["auc"]
    assert abs(table["H2O/attn-EMA"]["auc"] - views["s_within"]["auc"]) < 1e-3


def test_relevance_table_and_conditional_information(sources):
    sweep = load("revision_analysis.json")["A9_ucmi"]["ucmi_sweep"].values()
    within = [c["s_within"] for c in sweep]
    cross = [c["s_cross"] for c in sweep]
    other = [c[k] for c in sweep for k in ("s_query", "s_pos")]
    check(sources, "0.133", min(within), 3, "within-layer U_i lower")
    check(sources, "0.155", max(within), 3, "within-layer U_i upper")
    check(sources, "0.048", min(c for c in cross if c > 0), 3, "cross-layer U_i lower (finer grids)")
    check(sources, "0.084", max(cross), 3, "cross-layer U_i upper")
    assert min(cross) == 0.0, "the paper states the coarsest grid collapses to zero"
    assert max(other) <= 0.006, "remaining proxies are claimed to stay at or below 0.006"


def test_calibration_split_is_disclosed(sources):
    v2 = load("icdm_v2.json")
    for kind in ("request", "source"):
        assert v2[f"pooled_{kind}_split"]["calibration_requests_held_for_isotonic"] == 48
    assert "48 of the" in sources and "192 training model-requests" in sources
    assert "other 144" in sources and "excluded from both fitting stages" in sources


def test_workload_extension_and_transfer(sources, hist):
    ext = load("icdm_multiworkload.json")["headline_by_model_pooled"]
    check(hist, "0.929", ext["within+cross(2)"]["auc"], 3, "extension two-view AUC")
    check(hist, "0.934", ext["GBDT"]["auc"], 3, "extension GBDT AUC")
    # The archived AUPRC pair (0.777 / 0.803) came from the row-order tie rule and
    # the v1 rows are gone, so it cannot be recomputed tie-aware. The camera-ready
    # withdraws both numbers instead of reprinting them; they stay checked here.
    assert f'{ext["within+cross(2)"]["auprc"]:.3f}' == "0.777"
    assert f'{ext["GBDT"]["auprc"]:.3f}' == "0.803"
    assert "0.777" not in sources and "0.803" not in sources, \
        "the biased-tie AUPRC pair must stay out of the manuscript"
    assert "row-order tie rule and the rows are gone" in sources, \
        "the withdrawal must be stated, not silent"
    chat = {r["method"]: r["auc"] for r in load("icdm_full_sharegpt.json")["pooled"]["headline"]["table"]}
    check(hist, "0.016", chat["GBDT(LightGBM)"] - chat["within+cross(2)"], 3, "chat AUC gap")
    # "Each of the four model families contributes 128 prompts per workload."
    for name in ("icdm_full_longbench.json", "icdm_full_mooncake.json", "icdm_full_sharegpt.json"):
        assert load(name)["pooled"]["summary"]["n_requests"] == 4 * 128
    transfer = load("icdm_full.json")["transfer"]["standardized"]
    check(hist, "0.920", transfer["mean_cross"], 3, "cross-model AUC")
    check(hist, "0.921", transfer["mean_within"], 3, "within-model AUC")
    check(hist, "0.0012", transfer["mean_transfer_drop"], 4, "transfer drop")


def test_drift_and_threshold_behaviour(sources, hist):
    drift = load("icdm_full.json")["pooled"]["drift"]
    assert drift["train_steps"] == [0, 77] and drift["test_steps"] == [102, 127]
    check(sources, "0.948", drift["auc_static"], 3, "frozen scorer AUC")
    check(sources, "0.919", drift["auc_online"], 3, "online IRLS AUC")
    multiturn = load("drift_multiturn.json")
    check(hist, "0.35", multiturn["drift_jaccard_boundary_mean"], 2, "boundary Jaccard")
    check(hist, "0.83", multiturn["drift_jaccard_within_turn_mean"], 2, "within-turn Jaccard")
    # the budget-mismatched adaptive/static tail comparison stays in the artifact only
    assert multiturn["policies"]["adaptive_conformal"]["frac_blowout"] > multiturn["policies"]["fixed_global"]["frac_blowout"]
    conformal = load("icdm_extra.json")["conformal"]
    assert conformal["alpha"] == 0.1
    check(hist, "0.274", conformal["fixed_tau05"]["mean_miss_2nd_half"], 3, "naive tau=0.5 miss")
    check(hist, "0.095", conformal["fixed_tau05"]["mean_set_size_2nd_half"], 3, "naive tau=0.5 retention")
    check(hist, "0.087", conformal["fixed_split_conformal"]["mean_miss_2nd_half"], 3, "split-conformal miss")
    check(hist, "0.264", conformal["fixed_split_conformal"]["mean_set_size_2nd_half"], 3, "split-conformal retention")
    check(hist, "0.100", conformal["adaptive_g10"]["mean_miss_2nd_half"], 3, "adaptive miss")
    check(hist, "0.264", conformal["adaptive_g10"]["mean_set_size_2nd_half"], 3, "adaptive retention")


def test_guardkv_coverage(sources):
    rows = {r["target_alpha"]: r for r in load("coverage_budget.json")["by_alpha"]}
    assert max(abs(a - r["realized_miss"]) for a, r in rows.items()) <= 0.007
    assert "0.007" in sources
    check(sources, "34", 100 * rows[0.05]["emergent_budget"], 0, "alpha=0.05 retention")
    check(sources, "14", 100 * rows[0.20]["emergent_budget"], 0, "alpha=0.20 retention")
    served = {r["target_alpha"]: r for r in load("serving_calib/SUMMARY.json")["by_alpha"]}
    assert served[0.1]["emergent_budget"] == 1.0, "paper claims ~full retention for 90% coverage"


def test_scoring_only_microbenchmark(sources):
    fp16 = load("wcet_gpu.json")["results"]["fp16"]
    assert load("wcet_gpu.json")["batch"] == 4096 and "4096" in sources
    # The manuscript now prints only the four-weight kernel (page budget); the
    # pairwise/tiny-MLP percentiles remain in the tracked JSON.
    g = fp16["closed"]["cuda_graph_us"]
    check(sources, "28.7", g["p999"], 1, "scoring-only P99.9 batch time")
    assert f"{g['p50']:.1f}" == "20.5" and f"{g['p99']:.1f}" == "24.6"   # in the artifact, not the manuscript


def test_masked_loop_task_quality(sources):
    plain, clustered = load("tost/expand_tost.json"), load("tost/expand_tost_clustered.json")
    main = clustered["xqp_vs_h2o::pooled(2arch)"]
    assert main["k_clusters"] == 14 and main["n_items"] == 896
    check(sources, "0.00511", main["diff"], 5, "reconstructed-minus-H2O delta")
    # Detailed v1 intervals and columns moved to PC_AUDIT_RESPONSE.md; the
    # manuscript now reports v2. Preserve checks on the unchanged archive.
    assert main["tost"]["0.02"]["tost_p_clustered"] == pytest.approx(.0010662412)
    assert clustered["pyramidkv_vs_h2o::pooled"]["tost"]["0.03"]["tost_p_clustered"] > .05
    for printed, value, label in [
            (".292", plain["full_vs_xqp::pooled"]["mean_A"], "full cache"),
            (".253", plain["xqp_vs_h2o::pooled(2arch)"]["mean_B"], "H2O"),
            (".258", plain["xqp_vs_h2o::pooled(2arch)"]["mean_A"], "reconstructed"),
            (".249", plain["adakv_vs_h2o::pooled"]["mean_A"], "Ada-KV-style"),
            (".234", plain["pyramidkv_vs_h2o::pooled"]["mean_A"], "PyramidKV-style")]:
        assert f"{value:.3f}"[1:] == printed, f"archived {label} F1 changed"


def test_served_oracle_diagnostics(sources):
    c3 = load("served_oracle_ci/c3_served_auc.json")["overall"]
    check(sources, "0.643", c3["auc_point"], 3, "served-oracle AUC")
    check(sources, "0.632", c3["cluster_bootstrap_prompt"]["ci95_lo"], 3, "served-oracle CI lo")
    check(sources, "0.655", c3["cluster_bootstrap_prompt"]["ci95_hi"], 3, "served-oracle CI hi")
    per_layer = [r["auc_point"] for r in load("served_oracle_ci/c3_served_auc.json")["per_layer"]]
    check(sources, "0.54", min(per_layer), 2, "per-layer AUC lo")
    check(sources, "0.76", max(per_layer), 2, "per-layer AUC hi")
    c7 = load("served_oracle_ci/c7_oracle_miss_pertask.json")["pooled"]
    check(sources, "0.579", c7["h2o_eps"], 3, "H2O served-oracle miss")
    check(sources, "0.623", c7["xqp_eps"], 3, "reconstructed served-oracle miss")
    check(sources, "0.033", c7["diff_ci95"][0], 3, "miss difference lo")
    check(sources, "0.055", c7["diff_ci95"][1], 3, "miss difference hi")
    c6 = load("served_oracle_ci/c6_onpolicy_reversal.json")
    llama = c6["off_policy_recall20"]["per_model"]["llama"]
    check(sources, "0.06", llama["gain_point"], 2, "off-policy recall gain")
    check(sources, "0.03", llama["gain_ci95"][0], 2, "recall gain lo")
    check(sources, "0.10", llama["gain_ci95"][1], 2, "recall gain hi")
    on_policy = c6["on_policy_eps"]["per_config"]["llama_b0.20"]
    check(sources, "0.13", on_policy["delta_native_minus_h2o"], 2, "on-policy miss penalty")
    check(sources, "0.10", on_policy["delta_ci95"][0], 2, "on-policy penalty lo")
    check(sources, "0.16", on_policy["delta_ci95"][1], 2, "on-policy penalty hi")


def test_budget_sweep_and_long_context(sources):
    curves = load("tost/budget_at_quality.json")["curves"]
    assert sorted(float(b) for b in curves["h2o"]) == [0.1, 0.2, 0.3, 0.5]
    longctx = ROOT / "experiments/results/longctx"

    def pooled(directory, datasets):
        deltas, n = [], 0
        for ds in datasets:
            h = json.loads((directory / f"{ds}_h2o.json").read_text())["results"]
            x = json.loads((directory / f"{ds}_xqp.json").read_text())["results"]
            assert len(h) == len(x)
            deltas += [b["f1"] - a["f1"] for a, b in zip(h, x)]
            n += len(h)
        return deltas, n

    seven = ["2wikimqa", "hotpotqa", "multifieldqa_en", "musique", "narrativeqa", "qasper", "triviaqa"]
    deltas, n = pooled(longctx / "c16384_n64", seven)
    assert n == 448
    check(sources, "-0.0006", mean(deltas), 4, "16K mean F1 difference")
    pilot, n_pilot = pooled(longctx / "c16384",
                            ["2wikimqa", "hotpotqa", "multifieldqa_en", "narrativeqa", "qasper"])
    assert n_pilot == 160 and "160-prompt" in sources
    check(sources, "0.010", mean(pilot), 3, "16K pilot advantage")


def test_large_model_side_check(sources):
    base = ROOT / "experiments/results/largemodel/qwen2_5-14b-instruct/b0.20"
    values = {}
    for policy, printed in [("h2o", "0.301"), ("xqp", "0.307"), ("adakv", "0.301")]:
        rows = [r["f1"] for p in sorted(base.glob(f"*_{policy}.json"))
                for r in json.loads(p.read_text())["results"]]
        values[policy] = mean(rows)
        check(sources, printed, values[policy], 3, f"Qwen2.5-14B {policy}")
    assert json.loads(next(base.glob("*_h2o.json")).read_text())["context_length"] == 2048


# ------------------------- new physical-KV GPU results ------------------------
PHYSICAL = {
    "llama3.1-v1": {"dtype": "bfloat16", "store_full": "496", "store_kept": "94", "store_30": "144",
                    "peak_mask": "15.51", "peak_phys": "15.10", "peak_prefill": "15.77",
                    "rows": {"full.r0.json": (".223", "76.1"),
                             "h2o_block.masked.b0.2.r0.json": (".183", "61.0"),
                             "h2o_block.physical.b0.2.r0.json": (".181", "56.9"),
                             "xqp_reconstructed.masked.b0.2.r0.json": (".177", "60.0"),
                             "xqp_reconstructed.physical.b0.2.r0.json": (".178", "57.9")},
                    "budget30": {"h2o_block.physical.b0.3.r0.json": "0.203",
                                 "xqp_reconstructed.physical.b0.3.r0.json": "0.204"}},
    "qwen25-v1": {"dtype": "float32", "store_full": "434", "store_kept": "82", "store_30": "126",
                  "peak_mask": "29.05", "peak_phys": "28.63", "peak_prefill": "29.23",
                  "rows": {"full.r0.json": (".246", "71.4"),
                           "h2o_block.masked.b0.2.r0.json": (".233", "58.2"),
                           "h2o_block.physical.b0.2.r0.json": (".233", "54.4"),
                           "xqp_reconstructed.masked.b0.2.r0.json": (".232", "58.1"),
                           "xqp_reconstructed.physical.b0.2.r0.json": (".232", "54.4")},
                  "budget30": {"h2o_block.physical.b0.3.r0.json": "0.240",
                               "xqp_reconstructed.physical.b0.3.r0.json": "0.240"}},
}


@pytest.mark.parametrize("run", sorted(PHYSICAL))
def test_physical_kv_tables(sources, run):
    spec = PHYSICAL[run]
    summary = json.loads((RESULTS / "physical_kv" / run / "summary.json").read_text())
    cells = summary["cells"]
    assert summary["dtype"] == spec["dtype"] and summary["n_requests"] == 128
    assert summary["truncated_prompts"] == 104 and "104/128" in sources
    for name, (f1, tpot) in spec["rows"].items():
        check(sources, f1, cells[name]["f1_mean"], 3, f"{run} {name} F1")
        # Local inter-token latency left Table III on 2026-10-05: the masked rows are about
        # 20% faster than full cache, the text never interpreted that, and the paper makes no
        # latency claim. The record keeps the values, so they stay pinned here.
        assert f"{cells[name]['tpot_ms_median']:.1f}" == tpot, f"{run} {name} local ITL"
    assert "ITL" not in sources and "inter-token" not in sources, "Table III no longer has a latency column"
    for name, f1 in spec["budget30"].items():   # 30% cells stay in the artifact, not in the manuscript
        assert f"{cells[name]['f1_mean']:.3f}" == f1 and round(cells[name]["kv_storage_mib_mean"]) == int(spec["store_30"])
    check(sources, spec["store_full"], cells["full.r0.json"]["kv_storage_mib_mean"], 0, f"{run} full storage")
    for name in spec["rows"]:
        if ".physical.b0.2" in name:
            check(sources, spec["store_kept"], cells[name]["kv_storage_mib_mean"], 0, f"{run} compact storage")
            check(sources, spec["peak_phys"], cells[name]["peak_decode_gib_mean"], 2, f"{run} physical decode peak")
        if ".masked.b0.2" in name:
            check(sources, spec["store_full"], cells[name]["kv_storage_mib_mean"], 0, f"{run} masked storage")
            check(sources, spec["store_kept"], cells[name]["kv_live_mib_mean"], 0, f"{run} masked live payload")
            check(sources, spec["peak_mask"], cells[name]["peak_decode_gib_mean"], 2, f"{run} masked decode peak")
        check(sources, spec["peak_prefill"], cells[name]["peak_prefill_gib_mean"], 2, f"{run} prefill peak")


def test_mask_physical_parity_counts(sources):
    llama = json.loads((RESULTS / "physical_kv/llama3.1-v1/summary.json").read_text())["masked_vs_physical"]
    assert llama["h2o_block.b0.2"]["exact_token_match"] == 98 and "98/128" in sources
    assert llama["xqp_reconstructed.b0.2"]["exact_token_match"] == 100 and "100/128" in sources
    check(sources, "0.014", llama["h2o_block.b0.2"]["mean_abs_f1"], 3, "Llama H2O |dF1|")
    check(sources, "0.007", llama["xqp_reconstructed.b0.2"]["mean_abs_f1"], 3, "Llama reconstructed |dF1|")
    qwen = json.loads((RESULTS / "physical_kv/qwen25-v1/summary.json").read_text())["masked_vs_physical"]
    assert all(v["exact_token_match"] == 128 for v in qwen.values()) and "128/128" in sources
    assert all(v["mean_abs_f1"] == 0.0 for v in qwen.values())
    gate = json.loads((RESULTS / "physical_kv/qwen25-v1/physical-gates-v1.json.failed.json").read_text())
    assert gate["status"] == "failed" and "'max_abs_error': 0.25" in gate["error"]
    assert "'atol': 0.15" in gate["error"] and "'argmax_equal': True" in gate["error"]
    assert "0.25 versus 0.15" in (TEX / "sections" / "deployment.tex").read_text()


def test_abstract_compaction_ratio(sources):
    ratios = []
    for run in ("llama3.1-v1", "qwen25-v1"):
        cells = json.loads((RESULTS / "physical_kv" / run / "summary.json").read_text())["cells"]
        for name, cell in cells.items():
            if ".masked.b0.2" in name:
                twin = cells[name.replace(".masked.", ".physical.")]
                ratios.append(cell["kv_storage_mib_mean"] / twin["kv_storage_mib_mean"])
    assert all(abs(r - 5.3) < 0.05 for r in ratios), f"abstract says about 5.3x, data gives {ratios}"
    assert "5.3" in sources


def test_query_control_v2(sources):
    for run, printed_delta, printed_two_view, counts in [
            ("llama3.1-v1", "-0.0070", "0.899", ("32 test", "all negative")),
            ("qwen25-v1", "+0.0020", "0.884", ("31/32",))]:
        q = json.loads((RESULTS / "physical_kv" / run / "summary.json").read_text())["query_v2"]
        signed = f"{q['mean_request_auc_delta']:+.4f}"
        assert signed == printed_delta and printed_delta in sources, f"{run}: {signed}"
        check(sources, printed_two_view, q["mean_two_view_auc"], 3, f"{run} two-view request AUC")
        assert q["n_test"] == 32 and q["n_train"] == 96 and q["trace_version"] == 2
        for fragment in counts:
            assert fragment in sources
    llama = json.loads((RESULTS / "physical_kv/llama3.1-v1/summary.json").read_text())["query_v2"]
    assert llama["n_test_negative_delta"] == 32
    check(sources, "-0.0082", llama["ci95"][0], 4, "Llama query CI lo")
    check(sources, "-0.0058", llama["ci95"][1], 4, "Llama query CI hi")
    qwen = json.loads((RESULTS / "physical_kv/qwen25-v1/summary.json").read_text())["query_v2"]
    assert qwen["n_test_positive_delta"] == 31
    check(sources, "0.0016", qwen["ci95"][0], 4, "Qwen query CI lo")
    check(sources, "0.0024", qwen["ci95"][1], 4, "Qwen query CI hi")


# ------------------------- PC audit (2026-09): sensitivity, scorer, generation path ----
def test_archived_sensitivity_preserved_and_rescoring_disclosed(sources):
    sens = load("tost/expand_tost_sensitivity.json")
    assert sens["provenance"]["stored_f1_reproduced_by_runner_scorer"] == 4480
    assert sens["provenance"]["reference_row_mismatches"] == 0
    stored = sens["contrasts"]["xqp_vs_h2o::pooled::f1_stored"]
    # Historical details are now in the artifact, not mandatory paper text.
    assert stored["dataset_clusters"]["0.02"]["p"] < .05
    assert stored["architecture_dataset_cells"]["0.01"]["p"] > .05
    assert stored["dataset_clusters"]["0.01"]["p"] > .05
    official = sens["contrasts"]["xqp_vs_h2o::pooled::f1_longbench_all_refs"]
    check(sources, "+0.0030", official["grand_mean"], 4, "official-scorer delta")
    assert official["architecture_dataset_cells"]["0.02"]["p"] < .05


def test_float16_generation_path_audit(sources, hist):
    d = load("qwen_fp16_diagnosis/diagnosis.float16.json")["summary"]
    assert d["n"] == 56 and "56-prompt" in sources
    assert d["requests_with_all_nan_steps"] == 54 and "54 requests" in sources
    assert d["total_all_nan_steps"] == 687 and "687 NaN steps" in hist
    assert d["total_token0_emissions"] == 739 and "739 token-0 emissions" in hist
    assert d["total_partial_nan_steps"] == 0
    for dtype in ("bfloat16", "float32"):
        clean = load(f"qwen_fp16_diagnosis/diagnosis.{dtype}.json")["summary"]
        assert clean["total_all_nan_steps"] == 0 and clean["total_token0_emissions"] == 0
    archived = 0
    for task in ("narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "triviaqa"):
        rows = load(f"expand/qwen25_7b/{task}_full.json")["results"]
        archived += sum(r["pred"].count("!") >= 10 for r in rows)
    assert archived == 181 and "181/448" in sources


# ------------------------------- version-2 offline table -----------------------
V2_ROWS = {"age proxy": "recency", "query proxy (cosine)": "Quest",
           "prev-layer indicator": "InfiniGen", "within-layer EMA": "H2O/attn-EMA",
           "LightGBM, balanced": "GBDT(LightGBM)",
           "two-view logistic, refit": "within+cross(2)",
           "two-view, archived weights": "archived two-view checkpoint (frozen, v2 features)"}
# Rows kept out of the 8-page paper (still in icdm_v2.json and quoted in the artifact doc).
V2_ROWS_IN_ARTIFACT_ONLY = {"fitted MLP": "sklearnMLP(16,)", "two-view + dot-max": "within+cross+dotmax logistic",
                            "four-view logistic": "XQP-closed"}
# Single-signal rows are raw scores, not probabilities: the table prints no ECE for them.
RAW_SCORE_ROWS = {"recency", "Quest", "InfiniGen", "H2O/attn-EMA"}


def _v2():
    return load("icdm_v2.json")


def test_v2_table_rows_are_printed_from_the_json(sources):
    d = _v2()
    req = {r["method"]: r for r in d["pooled_request_split"]["table"]}
    src = {r["method"]: r for r in d["pooled_source_split"]["table"]}
    for label, key in V2_ROWS.items():
        r, s = req[key], src[key]
        assert abs(r["auc"] - s["auc"]) < 6.5e-3, "request- and source-split AUCs are described as agreeing within 0.006"
        if key in ("within+cross(2)", "GBDT(LightGBM)"):
            assert abs(r["auc"] - s["auc"]) < 5e-4, "described as agreeing to three decimals"
        cells = [f"{v:.3f}"[1:] for v in (r["auc"], r["auprc"], r["p_at_10_pooled"], r["p_at_10_grouped_macro"])]
        cells.append("---" if key in RAW_SCORE_ROWS else f"{r['ece']:.3f}"[1:])
        row = f"{label} & " + " & ".join(cells) + " \\\\"
        assert row in sources, f"tab:v2 row drifted: {row}"
    for label, key in V2_ROWS_IN_ARTIFACT_ONLY.items():
        assert f"{label} & " not in sources, f"{label} was moved out of the paper"
        assert key in req and key in src, f"{label} must stay in the result JSON"


def test_v2_table_blocks_come_from_the_renderers(sources):
    """Both blocks of Table II are the renderers' literal output."""
    import io
    import sys
    from contextlib import redirect_stdout
    sys.path.insert(0, str(ROOT / "experiments"))
    import render_v2_table as rv
    for argv, expected_rows in ((["--block", "v2"], len(V2_ROWS)), (["--block", "dec"], 3)):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rv.main(argv)
        rows = [line for line in buf.getvalue().splitlines() if line.strip()]
        assert len(rows) == expected_rows
        for row in rows:
            assert row in sources, f"table row drifted: {row}"


def test_v2_prose(sources, hist):
    d = _v2()
    req = {r["method"]: r for r in d["pooled_request_split"]["table"]}
    src = {r["method"]: r for r in d["pooled_source_split"]["table"]}
    two, gb, four = req["within+cross(2)"], req["GBDT(LightGBM)"], req["XQP-closed"]
    check(sources, "0.890", two["auc"], 3, "v2 two-view AUC")
    current = load("mentor_statistics.json")["request"]["auc"]["source_prompt"]
    two_ci = current["per_scorer"]["two-view logistic (refit)"]["ci95"]
    check(sources, "0.882", two_ci[0], 3, "v2 two-view source-bootstrap CI lo")
    check(sources, "0.898", two_ci[1], 3, "v2 two-view source-bootstrap CI hi")
    check(sources, "0.896", gb["auc"], 3, "v2 GBDT AUC")
    assert abs(two["auc"] - src["within+cross(2)"]["auc"]) < 5e-4, "request and source splits are said to agree to three decimals"
    paired = current["two_view_minus_gbdt"]
    check(sources, "-0.0057", paired["mean"], 4, "v2 paired deficit")
    check(sources, "-0.0068", paired["ci95"][0], 4, "v2 paired lo")
    check(sources, "-0.0046", paired["ci95"][1], 4, "v2 paired hi")
    check(sources, "0.873", four["auc"], 3, "v2 four-view AUC")
    frozen = req["archived two-view checkpoint (frozen, v2 features)"]
    check(sources, "0.890", frozen["auc"], 3, "frozen archived weights on v2")
    check(hist, "0.019", frozen["ece"], 3, "frozen archived weights ECE (artifact doc; the Table II row prints .019)")
    assert "two-view, archived weights & .890 & .667 & .622 & .611 & .019" in sources
    check(sources, "0.006", two["ece"], 3, "v2 two-view ECE")
    abl = d["pooled_request_split"]["view_ablation"]["drop"]["s_query"]["auc_drop"]
    check(hist, "0.012", -abl, 3, "AUC gain from dropping the cosine query proxy")
    check(hist, "0.891", req["within+cross+dotmax logistic"]["auc"], 3, "two views + dot-max")
    cal = d["pooled_request_split"]["calibration"]
    check(sources, "0.208", cal["GBDT balanced"]["ece"], 3, "balanced GBDT ECE (v2)")
    check(sources, "0.005", cal["GBDT unweighted"]["ece"], 3, "unweighted GBDT ECE (v2)")
    check(sources, "0.003", cal["GBDT balanced+isotonic"]["ece"], 3, "isotonic GBDT ECE (v2)")
    mi = d["pooled_per_view"]["redundancy"]["per_feature_mi"]
    check(sources, "0.104", mi["s_within"], 3, "v2 within MI")
    check(sources, "0.099", mi["s_cross"], 3, "v2 cross MI")
    check(sources, "0.007", mi["s_query"], 3, "v2 query MI")
    check(sources, "0.000", mi["s_pos"], 3, "v2 age MI")
    check(sources, "0.025", d["pooled_per_view"]["by_horizon"]["h4"]["f_query_dotmax"]["relevance_mi"], 3, "dot-max MI")
    assert d["pooled_summary"]["n_rows"] == 30445332 and "30.4M rows" in sources
    assert d["n_model_requests"] == 256 and "256 model-requests" in sources
    assert d["shared_source_prompts"] == 128
    assert d["pooled_request_split"]["n_test_requests"] == 64


def test_extended_gate_levels(sources):
    import glob
    root = RESULTS / "physical_kv/extended_gates"
    files = sorted(glob.glob(str(root / "*.json")))
    assert len(files) == 12 and "12 runs" in sources
    stats = {"llama": [], "qwen": []}
    for f in files:
        d = json.loads(Path(f).read_text())
        assert d["status"] == "gates_reported" and d["gates"]["strict"] is False
        r = d["gate_report"]
        assert r["max_abs_error_native"] == 0.0 and r["top1_agreement_native"] == r["steps_checked"]
        assert r["all_passed_numeric"] is False, "the extended gates are reported, not passed"
        stats[Path(f).name.split(".")[0]].append(r)
    llama, qwen = stats["llama"], stats["qwen"]
    assert len(llama) == 6 and len(qwen) == 6
    check(sources, "0.66", min(r["max_abs_error_replay"] for r in llama), 2, "bf16 replay error lo")
    check(sources, "1.44", max(r["max_abs_error_replay"] for r in llama), 2, "bf16 replay error hi")
    check(sources, "3.5", min(r["max_tolerance_ratio_replay"] for r in llama), 1, "bf16 ratio lo")
    check(sources, "5.9", max(r["max_tolerance_ratio_replay"] for r in llama), 1, "bf16 ratio hi")
    check(sources, "1.3", min(r["max_tolerance_ratio_replay"] for r in qwen), 1, "fp32 ratio lo")
    check(sources, "6.1", max(r["max_tolerance_ratio_replay"] for r in qwen), 1, "fp32 ratio hi")
    assert f"{min(r['max_abs_error_replay'] for r in qwen):.1e}" == "3.6e-05" and "3.6\\times10^{-5}" in sources
    assert f"{max(r['max_abs_error_replay'] for r in qwen):.1e}" == "1.9e-04" and "1.9\\times10^{-4}" in sources
    assert sum(r["top1_agreement_replay"] for r in llama) == 972 and "972/984" in sources
    assert sum(r["top1_agreement_replay"] for r in qwen) == 984 and "984/984" in sources
    assert sum(r["steps_checked"] for r in llama) == 984


def test_v2_per_decision_operating_point(sources):
    req = {r["method"]: r for r in _v2()["pooled_request_split"]["table"]}
    check(sources, "0.659", req["H2O/attn-EMA"]["p_at_10_grouped_macro"], 3, "within-layer per-decision P@k")
    check(sources, "0.613", req["within+cross(2)"]["p_at_10_grouped_macro"], 3, "two-view per-decision P@k")
    check(sources, "0.615", req["GBDT(LightGBM)"]["p_at_10_grouped_macro"], 3, "GBDT per-decision P@k")
    check(sources, "0.542", req["H2O/attn-EMA"]["p_at_10_pooled"], 3, "within-layer pooled P@k")
    check(sources, "0.873", req["H2O/attn-EMA"]["auc"], 3, "within-layer AUC")
    for r in req.values():   # per decision, k equals the label count, so precision == recall
        assert abs(r["p_at_10_grouped_macro"] - r["r_at_10_grouped_macro"]) < 1e-9
    assert req["H2O/attn-EMA"]["n_decision_groups"] == 61184


def test_exp8_table_rows_are_printed_from_both_sensitivity_files(sources):
    """Every row of Exp#8's pinned bfloat16 rerun table must equal what
    experiments/render_perlayer_table.py prints; v1 stays in the artifact."""
    import io
    import sys
    from contextlib import redirect_stdout
    sys.path.insert(0, str(ROOT / "experiments"))
    import render_perlayer_table as rp
    buf = io.StringIO()
    with redirect_stdout(buf):
        rp.main()
    rows = [line for line in buf.getvalue().splitlines() if line.strip()]
    assert len(rows) == 5
    for row in rows:
        assert row in sources, f"tab:perlayer row drifted: {row}"
    v2 = load("tost/expand_v2_sensitivity.json")
    assert v2["provenance"]["rows"] == 4480 and v2["provenance"]["reference_row_mismatches"] == 0
    assert v2["provenance"]["stored_f1_reproduced_by_runner_scorer"] == 4480


def test_exp8_rerun_prose(sources, hist):
    v2 = load("tost/expand_v2_sensitivity.json")
    main = v2["contrasts"]["xqp_vs_h2o::pooled::f1_longbench_all_refs"]
    assert main["k_cells"] == 14 and main["n_items"] == 896
    check(sources, "+0.0017", main["grand_mean"], 4, "rerun delta (official scorer)")
    check(sources, "0.0007", main["architecture_dataset_cells"]["0.02"]["p"], 4, "rerun TOST at .02")
    check(sources, "0.045", main["architecture_dataset_cells"]["0.01"]["p"], 3, "rerun TOST at .01")
    check(tex("transfer_gap"), "0.047", main["dataset_clusters"]["0.01"]["p"], 3, "rerun 7-dataset TOST at .01")
    assert main["dataset_clusters"]["0.01"]["equivalent_at_005"] is True
    assert "14-cell $t$-based and cell-bootstrap 90\\% intervals" in tex("transfer_gap")
    # the primary analysis is the seven-dataset TOST; the 14 cells are the sensitivity analysis
    check(sources, "0.0023", main["dataset_clusters"]["0.02"]["p"], 4, "rerun 7-cluster TOST (primary)")
    check(sources, "-0.0064", main["dataset_clusters"]["0.02"]["ci90_t"][0], 4, "7-dataset t-CI lo")
    check(sources, "0.0098", main["dataset_clusters"]["0.02"]["ci90_t"][1], 4, "7-dataset t-CI hi")
    assert main["dataset_clusters"]["0.02"]["n_clusters"] == 7 and main["dataset_clusters"]["0.02"]["df"] == 6
    assert "448 source prompts" in sources and "896 model--prompt evaluations" in sources
    assert "not pre-registered" in sources and "pre-specified" not in sources, \
        "the margin is an analysis-defined practical tolerance, not a pre-registered one"
    h2o_f1 = v2["policy_means"]["h2o"]["f1_longbench_all_refs"]["mean"]
    check(sources, "5.4", 100 * 0.02 / h2o_f1, 1, "margin as a share of the rerun baseline")
    check(sources, "0.372", h2o_f1, 3, "rerun H2O-style baseline")
    check(sources, "-0.0063", main["architecture_dataset_cells"]["0.02"]["ci90_t"][0], 4, "rerun t-CI lo")
    check(sources, "0.0097", main["architecture_dataset_cells"]["0.02"]["ci90_t"][1], 4, "rerun t-CI hi")
    check(sources, "-0.0056", main["ci90_cluster_bootstrap"][0], 4, "rerun boot lo")
    check(sources, "0.0087", main["ci90_cluster_bootstrap"][1], 4, "rerun boot hi")
    stored = v2["contrasts"]["xqp_vs_h2o::pooled::f1_stored"]
    check(hist, "+0.0024", stored["grand_mean"], 4, "rerun delta (runner scorer; artifact doc)")
    check(hist, "0.0016", stored["architecture_dataset_cells"]["0.02"]["p"], 4, "rerun TOST runner scorer (artifact doc)")
    llama = v2["contrasts"]["xqp_vs_h2o::llama::f1_longbench_all_refs"]
    qwen = v2["contrasts"]["xqp_vs_h2o::qwen::f1_longbench_all_refs"]
    check(sources, "+0.0052", llama["grand_mean"], 4, "rerun Llama delta")
    check(sources, "0.016", llama["architecture_dataset_cells"]["0.02"]["p"], 3, "rerun Llama TOST")
    check(sources, "-0.0018", qwen["grand_mean"], 4, "rerun Qwen delta")
    check(sources, "0.026", qwen["architecture_dataset_cells"]["0.02"]["p"], 3, "rerun Qwen TOST")
    ada = v2["contrasts"]["adakv_vs_h2o::pooled::f1_longbench_all_refs"]
    check(sources, "-0.0084", ada["grand_mean"], 4, "rerun Ada delta")
    check(sources, "0.015", ada["dataset_clusters"]["0.02"]["p"], 3, "rerun Ada TOST (seven datasets)")
    pyr = v2["contrasts"]["pyramidkv_vs_h2o::pooled::f1_longbench_all_refs"]
    assert pyr["k_cells"] == 14
    check(sources, "-0.023", pyr["grand_mean"], 3, "rerun Pyramid delta")
    check(sources, "0.58", pyr["dataset_clusters"]["0.02"]["p"], 2, "rerun Pyramid TOST (seven datasets)")
    assert pyr["dataset_clusters"]["0.02"]["p"] > 0.05
    # the corrupted-output signature is gone in the rerun
    import glob
    bang = long = n = 0
    for f in sorted(glob.glob(str(RESULTS / "expand_v2/qwen25_7b/*_full.json"))):
        for r in json.loads(Path(f).read_text())["results"]:
            n += 1; bang += r["pred"].count("!") >= 10; long += r["n_gen_tokens"] >= 48
    assert n == 448 and bang == 0 and "0/448" in sources
    assert long == 34 and "34/448" in sources


def test_feature_bridge_llama(sources):
    d = load("feature_bridge/bridge.llama.bf16.json")
    s = d["summary"]
    assert d["n_requests"] == 128 and d["args"]["dtype"] == "bfloat16" and d["args"]["budget"] == 0.2
    check(sources, "0.902", s["A_offline_full"]["mean_request_auc"], 3, "bridge A")
    check(sources, "0.899", s["B_runtime_full"]["mean_request_auc"], 3, "bridge B")
    check(sources, "0.989", s["C_runtime_policy"]["mean_request_auc"], 3, "bridge C")
    ba = s["B_minus_A_feature_realization"]
    check(sources, "-0.003", ba["mean"], 3, "bridge B-A")
    check(sources, "-0.004", ba["ci95"][0], 3, "bridge B-A lo")
    check(sources, "-0.001", ba["ci95"][1], 3, "bridge B-A hi")
    assert ba["n_negative"] == 84 and "84/128" in sources
    cb = s["C_minus_B_trajectory"]
    check(sources, "+0.090", cb["mean"], 3, "bridge C-B")
    assert cb["n_positive"] == 128


def test_feature_bridge_qwen(sources):
    d = load("feature_bridge/bridge.qwen.fp32.json")
    s = d["summary"]
    assert d["n_requests"] == 128 and d["args"]["dtype"] == "float32" and d["args"]["budget"] == 0.2
    check(sources, "0.894", s["A_offline_full"]["mean_request_auc"], 3, "Qwen bridge A")
    check(sources, "0.892", s["B_runtime_full"]["mean_request_auc"], 3, "Qwen bridge B")
    check(sources, "0.989", s["C_runtime_policy"]["mean_request_auc"], 3, "Qwen bridge C")
    ba = s["B_minus_A_feature_realization"]
    check(sources, "-0.002", ba["mean"], 3, "Qwen bridge B-A")
    check(sources, "-0.003", ba["ci95"][0], 3, "Qwen bridge B-A lo")
    check(sources, "-0.000", ba["ci95"][1], 3, "Qwen bridge B-A hi")
    assert ba["n_negative"] == 75 and "75/128" in sources
    check(sources, "+0.098", s["C_minus_B_trajectory"]["mean"], 3, "Qwen bridge C-B")
    assert s["C_minus_B_trajectory"]["n_positive"] == 128


def test_served_oracle_budget_decomposition(sources):
    """The share of served-oracle miss that no selector can remove (PC M2)."""
    head = load("served_oracle_ci/oracle_budget.e2e_confirm.json")
    assert head["prompt_blocks"] == 128 and head["offline_label_k"] == 13
    assert head["served_oracle_k"] == 32 and head["policy_budget_blocks"] == 26
    assert head["oracle_floor_branch_active"], "at 4K the max(32, .) branch must bind"
    for policy, miss, residual in [("h2o", "0.579", "0.360"), ("xqp", "0.623", "0.404")]:
        p = head["policies"][policy]
        assert p["steps_with_budget_below_oracle"] == 1.0
        check(sources, miss, p["measured_miss"]["mean"], 3, f"{policy} served-oracle miss")
        check(sources, residual, p["selector_residual"]["mean"], 3, f"{policy} selector residual")
    check(sources, "0.219", head["policies"]["h2o"]["forced_floor"]["mean"], 3, "forced floor")
    # 16K is a cap; shorter effective inputs still have a nonzero floor. These
    # per-budget floors moved to the artifact in the camera-ready, so they are
    # verified against the JSON but no longer required to appear in the LaTeX.
    sweep = {"budget_sweep_b0.10": ("0.610", "0.727", True),
             "budget_sweep_b0.30": ("0.022", None, True),
             "budget_sweep_b0.50": ("0.004", None, True),
             "longctx_c16384_n64": ("0.063", None, False)}
    for tag, (floor, miss, binding) in sweep.items():
        d = load(f"served_oracle_ci/oracle_budget.{tag}.json")
        assert d["oracle_floor_branch_active"] is binding, tag
        assert f'{d["policies"]["h2o"]["forced_floor"]["mean"]:.3f}' == floor, f"{tag} floor"
        if miss:
            assert f'{d["policies"]["h2o"]["measured_miss"]["mean"]:.3f}' == miss, f"{tag} miss"
    assert "the other caps and budget fractions are in the" in sources, \
        "the manuscript must say where the per-budget floors went"


def test_training_size_sensitivity(sources):
    """The compact scorer's AUC deficit must not be an artifact of TRAIN_N (audit item).

    `experiments/analyze_train_size_sensitivity.py` reruns the headline fit on the
    same split, seed and held-out rows at training budgets up to every held-in row.
    Its 120K point reproduces the published Table II numbers, which is the gate on
    the rerun itself; the paper then quotes the largest budget.
    """
    d = load("icdm_v2_train_size.json")
    rows = {r["n_train"]: r for r in d["by_train_size"]}
    base, full = rows[120_000], rows[d["n_train_rows_available"]]
    v2 = {r["method"]: r for r in _v2()["pooled_request_split"]["table"]}
    # the 120K rerun must land on the published table, or the sweep means nothing
    assert abs(base["auc_two_view"] - v2["within+cross(2)"]["auc"]) < 5e-4
    assert abs(base["auc_gbdt"] - v2["GBDT(LightGBM)"]["auc"]) < 5e-4
    assert abs(base["dec_recall_gbdt"] - v2["GBDT(LightGBM)"]["r_at_10_grouped_macro"]) < 5e-4
    assert abs(d["raw_within_layer_decision_recall"] - v2["H2O/attn-EMA"]["r_at_10_grouped_macro"]) < 5e-4
    check(sources, "0.8961", base["auc_gbdt"], 4, "LightGBM AUC at the 120K cap")
    check(sources, "0.8966", full["auc_gbdt"], 4, "LightGBM AUC on every held-in row")
    check(sources, "-0.0057", base["auc_delta"], 4, "paired deficit at the cap")
    check(sources, "-0.0062", full["auc_delta"], 4, "paired deficit at full data")
    check(sources, "-0.0073", full["auc_delta_lo"], 4, "full-data deficit CI lo")
    check(sources, "-0.0051", full["auc_delta_hi"], 4, "full-data deficit CI hi")
    assert "22.8M" in sources and "190-fold" in sources
    assert round(d["n_train_rows_available"] / 1e6, 1) == 22.8
    assert round(d["n_train_rows_available"] / 120_000) == 190
    # the tree comparator's per-decision recall is claimed not to move
    assert all(abs(r["dec_recall_gbdt"] - 0.615) < 5e-4 for r in d["by_train_size"]), \
        "the paper says LightGBM per-decision recall holds at 0.615 across budgets"
    assert "recall holds at 0.615" in sources


def test_per_decision_operating_point(sources):
    """Paired intervals for the pooled-versus-per-decision reversal (PC H1, H2)."""
    d = load("decision_operating_point.json")
    assert d["held_out_requests"] == 64
    label = {"within": "within-layer EMA", "two": "two-view logistic (refit)",
             "gbdt": "LightGBM, balanced"}
    at10 = d["operating_points"]["0.10"]
    # the point estimates must still be the ones Table II prints
    for key, printed in [("within", "0.659"), ("two", "0.613"), ("gbdt", "0.615")]:
        check(sources, printed, at10["per_scorer"][label[key]]["macro_recall_mean"], 3,
              f"per-decision recall at 10% ({key})")
    sensitivity = load("mentor_statistics.json")
    assert sensitivity["n_boot"] == 10000
    for other, printed, lo, hi in [("two", "0.046", "0.043", "0.049"),
                                   ("gbdt", "0.044", "0.041", "0.047")]:
        v = sensitivity["request"]["operating_points"]["0.10"]["source_prompt"][label[other]]
        assert v["n_positive"] == v["n_requests"], "claimed positive on every request"
        check(sources, printed, v["mean"], 3, f"10% gap vs {other}")
        check(sources, lo, v["ci95"][0], 3, f"10% gap lo vs {other}")
        check(sources, hi, v["ci95"][1], 3, f"10% gap hi vs {other}")
    for other, printed, lo, hi in [("two", "0.047", "0.044", "0.050"),
                                   ("gbdt", "0.045", "0.043", "0.048")]:
        v = sensitivity["source"]["operating_points"]["0.10"]["source_prompt"][label[other]]
        assert v["n_clusters"] == 32 and v["n_requests"] == 64
        for printed_value, value in zip((printed, lo, hi), (v["mean"], *v["ci95"])):
            check(sources, printed_value, value, 3, "source-disjoint sensitivity")
    at20 = d["operating_points"]["0.20"]
    for key, printed in [("within", "0.840"), ("two", "0.838"), ("gbdt", "0.830")]:
        check(sources, printed, at20["per_scorer"][label[key]]["macro_recall_mean"], 3,
              f"per-decision recall at 20% ({key})")
    shrunk = at20["paired"][f"{label['within']} minus {label['two']} (recall)"]
    assert shrunk["n_positive"] == 52 and "52/64" in sources
    check(sources, "0.002", shrunk["mean"], 3, "20% gap vs two-view")
    check(sources, "0.010", at20["paired"][f"{label['within']} minus {label['gbdt']} (recall)"]["mean"],
          3, "20% gap vs LightGBM")
    assert at10["per_scorer"][label["within"]]["macro_recall_mean"] - \
           at10["per_scorer"][label["two"]]["macro_recall_mean"] > \
           at20["per_scorer"][label["within"]]["macro_recall_mean"] - \
           at20["per_scorer"][label["two"]]["macro_recall_mean"], \
        "the paper says the reversal shrinks at the larger budget"


# ----------------------------- 2026-10-04 review: decomposition, capacity, counts -------------
def test_decision_decomposition_record(sources):
    """`experiments/analyze_decision_decomposition.py`: pooled AUC split into same- and
    cross-decision pairs, the cross-layer offset sweep, the boundary swaps, tie
    sensitivity and the train-rows-only relevance check."""
    d, v2 = load("icdm_v2_decomposition.json"), _v2()
    pub = {r["method"]: r for r in v2["pooled_request_split"]["table"]}
    # gate: the record reproduces Table II's pooled AUC and per-decision recall
    for name, key in [("within-layer EMA", "H2O/attn-EMA"), ("two-view logistic (refit)", "within+cross(2)"),
                      ("LightGBM, balanced", "GBDT(LightGBM)")]:
        assert abs(d["gate"][name]["auc_150k_sample"] - pub[key]["auc"]) < 5e-4, name
        assert abs(d["gate"][name]["dec_recall_reference_impl"] - pub[key]["r_at_10_grouped_macro"]) < 5e-4, name
        assert abs(d["scorers"][name]["recall_0.10"] - pub[key]["r_at_10_grouped_macro"]) < 5e-4, name
    assert d["data"]["n_decisions"] == 61184 and "61,184" in sources
    assert d["data"]["n_rows_heldout"] == 7633988 and d["data"]["n_requests_heldout"] == 64
    assert d["data"]["n_source_prompts_heldout"] == 57 and d["data"]["heldout_requests_by_model"] == {"llama": 30, "qwen": 34}
    # pooled AUC is almost entirely cross-decision pairs
    check(sources, "0.0016", 100 * d["pairs"]["share_same_decision"], 4, "same-decision pairs, percent")
    for name in d["scorers"]:
        s = d["scorers"][name]
        assert abs(s["auc_pooled"] - s["auc_cross"]) < 1e-5, "pooled AUC equals cross-decision AUC to 1e-5"
    # the offset the two-view fit adds to the raw within-layer ordering
    tv = d["two_view"]
    check(sources, "0.025", tv["offset_ratio"], 3, "w_cross / w_within")
    check(sources, "97.7", tv["w_within"], 1, "fitted within-layer weight")
    check(sources, "2.46", tv["w_cross"], 2, "fitted cross-layer weight")
    check(sources, "0.015", v2["pooled_summary"]["feature_mean"]["s_within"], 3, "corpus mean x_wl")
    p = d["paired"]["two-view logistic (refit) minus within-layer EMA"]
    check(sources, "+0.017", p["auc_pooled"]["delta"], 3, "pooled AUC gain of the cross-layer term")
    check(sources, "0.015", p["auc_pooled"]["ci95"][0], 3, "pooled gain CI lo")
    check(sources, "0.019", p["auc_pooled"]["ci95"][1], 3, "pooled gain CI hi")
    assert p["auc_pooled"]["share_positive"] == 1.0 and p["auc_same"]["share_positive"] == 0.0
    check(sources, "0.0025", -p["auc_same"]["delta"], 4, "within-decision AUC loss")
    check(sources, "0.0023", -p["auc_same"]["ci95"][1], 4, "within-decision loss CI lo")
    check(sources, "0.0028", -p["auc_same"]["ci95"][0], 4, "within-decision loss CI hi")
    check(sources, "0.046", -p["recall_0.10"]["delta"], 3, "top-decile recall loss")
    g = d["paired"]["LightGBM, balanced minus within-layer EMA"]
    check(sources, "+0.022", g["auc_pooled"]["delta"], 3, "LightGBM pooled AUC gain")
    check(sources, "-0.005", g["auc_same"]["delta"], 3, "LightGBM within-decision AUC")
    check(sources, "-0.044", g["recall_0.10"]["delta"], 3, "LightGBM top-decile recall")
    # the offset sweep behind Fig. 2(a): endpoints are the two scorers, and one eighth of the
    # offset already carries most of the pooled gain
    sw = {round(r["lam"], 3): r for r in d["sweep"]}
    raw, two = d["scorers"]["within-layer EMA"], d["scorers"]["two-view logistic (refit)"]
    assert abs(sw[0.0]["auc_pooled"] - raw["auc_pooled"]) < 1e-9 and abs(sw[0.0]["recall_0.10"] - raw["recall_0.10"]) < 1e-9
    assert abs(sw[1.0]["auc_pooled"] - two["auc_pooled"]) < 5e-5 and abs(sw[1.0]["recall_0.10"] - two["recall_0.10"]) < 5e-5
    gain = (sw[0.125]["auc_pooled"] - sw[0.0]["auc_pooled"]) / (sw[1.0]["auc_pooled"] - sw[0.0]["auc_pooled"])
    loss = (sw[0.125]["recall_0.10"] - sw[0.0]["recall_0.10"]) / (sw[1.0]["recall_0.10"] - sw[0.0]["recall_0.10"])
    assert round(100 * gain) == 83 and round(100 * loss) == 65 and "83\\% of the first and 65\\% of the second" in sources
    # what the term swaps at the top-decile boundary
    s = d["swaps_at_top_decile"]
    assert s["cross_indicator_rate_swapped_in"] == 1.0 and s["cross_indicator_rate_swapped_out"] == 0.0
    assert [round(100 * s[k]) for k in ("swapped_share_of_selected", "positive_rate_swapped_in", "positive_rate_swapped_out",
                                          "positive_rate_given_cross_1", "positive_rate_given_cross_0")] == [22, 20, 41, 61, 4]
    for fragment in ("swaps in 22\\%", "20\\% are positive, against 41\\%", "61\\% of previous-layer-hot", "4\\% of the rest"):
        assert fragment in sources, fragment
    assert s["net_positives"] < 0
    # every decision holds exactly ceil(0.1 n) positives: no decision-level prior to flag
    # (benchmark.protocol enforces the same rule; the corpus passed it)
    # ties: stable versus exact random tie-breaking changes no per-decision recall by more than 2e-5
    worst = max(abs(r[f"recall_{k}_random_ties"] - r[f"recall_{k}"]) for r in d["scorers"].values() for k in ("0.10", "0.20"))
    assert worst < 2e-5 and "$2\\times10^{-5}$" in sources
    # relevance estimates: recomputed from held-in rows only they barely move
    mi = d["relevance_mi"]
    drift = max(abs(mi["train_rows_only"][k] - mi["published_all_rows_sample"][k]) for k in mi["train_rows_only"])
    check(sources, "0.0011", drift, 4, "largest change of a relevance estimate on held-in rows only")
    for k, v in mi["published_all_rows_sample"].items():
        assert abs(v - v2["pooled_per_view"]["redundancy"]["per_feature_mi"][k]) < 1e-12
    # the Table II lower block quotes this record (rows are asserted via the renderer test)
    assert "61,184 decisions" in sources


def test_gbdt_capacity_record(sources):
    """`experiments/analyze_gbdt_capacity.py`: the compact scorer's AUC deficit across tree capacities."""
    c = load("icdm_v2_gbdt_capacity.json")
    two = {r["n_train"]: r for r in c["rows"] if r["config"] == "two-view logistic"}
    gb = [r for r in c["rows"] if r["config"] != "two-view logistic"]
    assert c["config"]["train_sizes"] == [120000, 1920000] and len(gb) == 12 and len({r["config"] for r in gb}) == 6
    lead = [r["auc_150k_sample"] - two[r["n_train"]]["auc_150k_sample"] for r in gb]
    check(sources, "0.0005", min(lead), 4, "smallest LightGBM AUC lead over the compact fit")
    check(sources, "0.0076", max(lead), 4, "largest LightGBM AUC lead")
    assert max(lead) < 0.008 and "0.008 at most" in sources and "at most 0.008" in sources
    check(sources, "0.615", min(r["recall_0.10"] for r in gb), 3, "lowest LightGBM per-decision recall")
    check(sources, "0.622", max(r["recall_0.10"] for r in gb), 3, "highest LightGBM per-decision recall")
    raw = c["raw_within_layer_ema"]["recall_0.10"]
    check(sources, "0.659", raw, 3, "raw EMA per-decision recall")
    assert max(r["recall_0.10"] for r in gb) < raw, "no tree capacity reaches the raw EMA's per-decision recall"
    assert max(r["auc_same_decision"] for r in gb) < c["raw_within_layer_ema"]["auc_same_decision"]
    head = next(r for r in gb if r["n_train"] == 120000 and r["config"].startswith("depth3_150_balanced"))
    check(sources, "0.8961", head["auc_150k_sample"], 4, "headline LightGBM AUC (Table II)")
    assert "depth 3--8, 150--600 trees" in sources


def test_retained_blocks_are_matched_in_counts(sources, hist):
    """Logged `per_step_block_count` of the pinned rerun: retention is matched in counts."""
    import statistics
    counts, below_mandatory = {}, {}
    for pol in ("h2o", "xqp", "adakv", "pyramidkv"):
        steps, low = [], set()
        for arch in ("llama31_8b", "qwen25_7b"):
            for ds in ("narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "triviaqa"):
                for r in json.loads((RESULTS / f"expand_v2/{arch}/{ds}_{pol}.json").read_text())["results"]:
                    steps.append(r["per_step_block_count"])
                    if min(r["per_step_block_count"]) < 8:
                        low.add((arch, ds, r["id"]))
        assert len(steps) == 896
        counts[pol] = [statistics.mean(s) for s in steps]
        below_mandatory[pol] = low
        flat = [c for s in steps for c in s]
        assert statistics.median(flat) == 26 and min(flat) == 7 and max(flat) == 26
    means = {p: sum(v) / len(v) for p, v in counts.items()}
    for pol, printed in (("h2o", "25.02"), ("xqp", "25.02"), ("adakv", "25.02"), ("pyramidkv", "25.01")):
        assert f"{means[pol]:.2f}" == printed, (pol, means[pol])
    assert "average 25.02 for the H2O-style, reconstructed and Ada-KV-style policies and 25.01 for the PyramidKV-style" in sources
    assert "median 26, range 7--26" in sources
    # only one prompt per architecture dips below the eight mandatory blocks; every policy keeps 7 there
    assert all(low == {("llama31_8b", "2wikimqa", 18), ("qwen25_7b", "2wikimqa", 18)} for low in below_mandatory.values())
    assert "7 blocks on one prompt per model" in sources and "2WikiMQA id 18" in hist

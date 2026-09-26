"""The archived expansion sweep's scorer provenance and cluster sensitivity are
re-derivable from the tracked cells (PC items P0-2, P1-2)."""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import analyze_expand_sensitivity as aes  # noqa: E402

ARCHIVED = ROOT / "experiments/results/expand"
COMMITTED = ROOT / "experiments/results/tost/expand_tost_sensitivity.json"


@pytest.fixture(scope="module")
def cells():
    return aes.load_cells(ARCHIVED, None, ["full", "h2o", "xqp", "adakv", "pyramidkv"])


def test_every_stored_f1_is_reproduced_by_the_runner_scorer(cells):
    _, prov = cells
    assert prov["rows"] == 4480
    assert prov["stored_f1_reproduced_by_runner_scorer"] == 4480


def test_normalization_order_is_the_documented_difference():
    # articles-before-punctuation (runner) vs punctuation-before-articles (LongBench)
    assert aes.runner_f1("![](The!Rebirth)", "The Rebirth") == 1.0
    assert aes.longbench_f1("![](The!Rebirth)", ["The Rebirth"], "hotpotqa") == 0.0
    assert aes.longbench_f1("first line\nsecond", ["first line"], "triviaqa") == 1.0


def test_committed_sensitivity_matches_a_fresh_computation(cells):
    committed = json.loads(COMMITTED.read_text())["contrasts"]["xqp_vs_h2o::pooled::f1_stored"]
    fresh = aes.analyze_contrast(cells[0], "xqp", "h2o", aes.ARCHS, "f1_stored")
    assert fresh["k_cells"] == 14 and fresh["n_items"] == 896
    assert abs(fresh["grand_mean"] - committed["grand_mean"]) < 1e-12
    for margin in ("0.01", "0.02"):
        assert abs(fresh["architecture_dataset_cells"][margin]["p"]
                   - committed["architecture_dataset_cells"][margin]["p"]) < 1e-9
        assert abs(fresh["dataset_clusters"][margin]["p"] - committed["dataset_clusters"][margin]["p"]) < 1e-9
    # the two interval procedures are different objects and both are kept
    assert committed["ci90_cluster_bootstrap"] != committed["architecture_dataset_cells"]["0.02"]["ci90_t"]
    # the archived headline numbers are unchanged by this re-analysis
    assert round(committed["grand_mean"], 5) == 0.00511
    assert round(committed["architecture_dataset_cells"]["0.02"]["p"], 6) == 0.001066
    assert committed["architecture_dataset_cells"]["0.01"]["p"] > 0.05      # not equivalent at +-0.01
    assert committed["dataset_clusters"]["0.02"]["p"] < 0.05               # still equivalent with 7 clusters

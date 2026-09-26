import numpy as np
import pytest
from analyze_oracle_budget import paired_summary
from audit_oracle_count_mapping import floor_comparison


def row(i, floor, miss):
    return dict(dataset="qa", id=i, forced_floor_mean=floor,
                miss_mean=miss, residual_mean=miss-floor)


def test_equal_grand_mean_does_not_imply_equal_paired_floors():
    a = [row(0, .1, .5), row(1, .3, .5)]
    b = [row(1, .1, .6), row(0, .3, .6)]
    s = paired_summary(a, b)
    assert s["floor_delta"]["mean"] == 0
    assert not s["floors_equal_per_request"]
    assert s["miss_delta"]["ci95"] != s["residual_delta"]["ci95"]


def test_paired_equal_floors_cancel():
    a = [row(0, .1, .5), row(1, .3, .6)]
    b = [row(1, .3, .7), row(0, .1, .55)]
    s = paired_summary(a, b)
    assert s["floors_equal_per_request"]
    np.testing.assert_allclose(s["miss_delta"]["ci95"], s["residual_delta"]["ci95"], atol=1e-14)


def test_incomplete_pairs_rejected():
    with pytest.raises(ValueError, match="exactly the same"):
        paired_summary([row(0, .1, .5)], [row(1, .1, .5)])


def test_rounded_counts_cannot_certify_mean_layer_floor():
    nonuniform = floor_comparison([1, 3], [2, 2])
    uniform = floor_comparison([2, 2], [2, 2])
    assert nonuniform["rounded_count_floor"] == uniform["rounded_count_floor"] == 0
    assert nonuniform["mean_layer_floor"] == .25
    assert uniform["mean_layer_floor"] == 0
    assert not nonuniform["equal_cardinalities"] and uniform["equal_cardinalities"]
    # With differing denominators it need not even be a lower bound.
    unequal_k = floor_comparison([1, 1], [1, 3])
    assert unequal_k["rounded_count_floor"] == .5
    assert unequal_k["mean_layer_floor"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("directory,report", [("e2e_confirm", "mentor_e2e_confirm"),
                                              ("longctx/c16384_n64", "mentor_longctx")])
def test_committed_pairing_is_rederived_from_raw_records(directory, report):
    import json
    from pathlib import Path
    from analyze_oracle_budget import per_request
    root = Path(__file__).resolve().parents[1] / "experiments/results"
    saved = json.loads((root / f"served_oracle_ci/{report}.json").read_text())
    rows = {p: [dict(r, dataset=ds) for ds in saved["datasets"]
                for r in per_request(root / directory / f"{ds}_{p}.json")]
            for p in ("h2o", "xqp")}
    assert paired_summary(rows["h2o"], rows["xqp"]) == saved["paired_second_minus_first"]

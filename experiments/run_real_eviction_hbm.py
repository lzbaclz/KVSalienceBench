"""Retired v1 HBM probe; historical source is in legacy/run_real_eviction_hbm_v1.py.

V1 swallowed decode errors, inferred positions from compact cache length, and
mixed full-prefill peaks with decode. Its archived output cannot substantiate
quality parity, peak-memory saving or a serving claim. Do not silently rerun it
and mix results with the corrected validation protocol.
"""
if __name__ == "__main__":
    raise SystemExit(
        "V1 HBM probe is retired. Use experiments/run_physical_kv.py with explicit "
        "model/data/policy/budget arguments; see docs/PHYSICAL_KV_VALIDATION.md. "
        "Historical results are preserved but are not corrected GPU measurements."
    )

#!/usr/bin/env python3
"""Exp#6: which F1 the physical-KV cells store, checked by rescoring every generation.

Table III (physical-KV check) and Table IV (masked-loop rerun) report F1 on different
cohorts, and the manuscript once explained part of the difference by the scorer. This
driver settles what the physical cells actually store. For every request of every
cell it recomputes, from the stored generation and the frozen cohort's references:

* the official LongBench ``qa_f1_score`` (punctuation removed before articles, maximum
  over ALL references), using the same function as Exp#8's analysis
  (``analyze_expand_sensitivity.longbench_f1``);
* the same score against the first reference only.

and compares both with the stored per-request ``f1`` (``xqp.physical_validation.qa_f1``).
The stored values equal the official all-reference score on every request, so Table III
uses Table IV's scorer; the single-reference means are recorded to show the comparison
is not vacuous (46 of the 128 prompts have more than one reference).

Needs the frozen cohort files (``physical-qa-*.jsonl``: prompts and references, not in
the public artifact). The record it writes ships with the artifact and no existing
result is modified.

Usage:
  python experiments/analyze_physical_scorer.py --out experiments/results/physical_kv/scorer_check.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from analyze_expand_sensitivity import longbench_f1  # noqa: E402

BASE = ROOT / "experiments/results/physical_kv"
RUNS = (("llama3.1-v1", "physical-qa-llama-v1.jsonl"), ("qwen25-v1", "physical-qa-qwen-v1.jsonl"))
TOL = 1e-12


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite {a.out}")
    report = dict(schema="physical-kv-scorer-check-v1",
                  scorer="official LongBench qa_f1_score: lowercase, punctuation removed, then articles, "
                         "token F1, maximum over all references",
                  rescoring_function="experiments/analyze_expand_sensitivity.py::longbench_f1 (Exp#8's)",
                  stored_function="xqp/physical_validation.py::qa_f1", tolerance=TOL, runs={})
    for run, cohort in RUNS:
        blob = (BASE / cohort).read_bytes()
        refs = {}
        for line in blob.decode("utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                refs[r["id"]] = (r["dataset"], r["answers"])
        cells = {}
        for path in sorted((BASE / run).glob("*.r0.json")):
            if path.name.startswith("paired"):
                continue
            obj = json.loads(path.read_text())
            if obj["data_sha256"] != hashlib.sha256(blob).hexdigest():
                raise SystemExit(f"{path}: cell was not run on {cohort}")
            rows = obj["results"]
            stored = [r["f1"] for r in rows]
            all_refs = [longbench_f1(r["prediction"], refs[r["id"]][1], refs[r["id"]][0]) for r in rows]
            single = [longbench_f1(r["prediction"], refs[r["id"]][1][:1], refs[r["id"]][0]) for r in rows]
            diffs = [abs(s - o) for s, o in zip(stored, all_refs)]
            cells[path.name] = dict(n=len(rows), f1_stored_mean=sum(stored) / len(stored),
                                    f1_official_all_refs_mean=sum(all_refs) / len(all_refs),
                                    f1_official_single_ref_mean=sum(single) / len(single),
                                    max_abs_difference=max(diffs),
                                    requests_differing=sum(d > TOL for d in diffs))
            print(f"{run:<12} {path.name:<42} stored {cells[path.name]['f1_stored_mean']:.4f}  "
                  f"official all-refs {cells[path.name]['f1_official_all_refs_mean']:.4f}  "
                  f"single-ref {cells[path.name]['f1_official_single_ref_mean']:.4f}  "
                  f"differing {cells[path.name]['requests_differing']}")
        n_refs = [len(v[1]) for v in refs.values()]
        report["runs"][run] = dict(cohort=cohort, cohort_sha256=hashlib.sha256(blob).hexdigest(),
                                   n_requests=len(refs), requests_with_several_references=sum(n > 1 for n in n_refs),
                                   max_references=max(n_refs), cells=cells)
    report["all_stored_equal_official_all_refs"] = all(
        c["requests_differing"] == 0 for r in report["runs"].values() for c in r["cells"].values())
    a.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"all stored F1 equal the official all-reference score: {report['all_stored_equal_official_all_refs']}")
    print(f"WROTE {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

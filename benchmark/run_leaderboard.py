"""KVSalienceBench evaluator, protocol 2.0 (decision-level).

The module keeps its historical name. It is an evaluator for a stated protocol, not a
closed leaderboard: it cannot tell how a scorer was trained, and without a candidate
manifest it cannot tell whether a table lists every candidate block of a decision.
Each run prints what it scored (split, horizon, label rate, retentions, held-out source
prompts, scorer) and the JSON output records what was and was not checked.

Score a prediction table. No traces or model weights are needed; the shipped example
is synthetic and exists to show the schema and to check the evaluator:

    python -m benchmark.run_leaderboard --table benchmark/example/prediction_table.csv --probabilistic

Add ``--manifest example_manifest.json`` (written by ``--write-manifest`` from a table
known to be complete) to have every decision's candidate set verified, not only its
positive count.

Score a submission on version-2 traces (source-disjoint split by default; the
reference baseline is refit on the training rows of the same split):

    python -m benchmark.run_leaderboard --traces Llama=path.jsonl,Qwen=path.jsonl \
        --submission my_method.py [--split request]

The paper's Table II uses ``--split request`` (new model-requests; most of their source
prompts also occur in training through the other model); the default ``source`` split
holds out the same source prompts for every model. Both held-out sets of the paper's
corpus are listed in ``benchmark/splits/paper_v2_splits.json``.

A submission is a ``.py`` file exposing ``score(F)``: an (N, 4) feature matrix in the
column order ``benchmark.protocol.FEATURE_COLUMNS`` to N scores. Set a module attribute
``PROBABILISTIC = True`` if the scores are probabilities; ECE and Brier are then
reported, and only then. See ``benchmark/submit_template.py``. The version-1 runner is
``benchmark/legacy_v1/run_leaderboard.py``.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark import protocol as P  # noqa: E402
from benchmark.fit_reference import fit_reference, score_fn  # noqa: E402


def load_submission(pyfile):
    spec = importlib.util.spec_from_file_location("submission", pyfile)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "score"):
        raise AttributeError(f"{pyfile} must define score(F) -> one score per row")
    return Path(pyfile).name, mod.score, bool(getattr(mod, "PROBABILISTIC", False))


def _ci(res, key):
    lo, hi = res["ci95"][key]
    return f"[{lo:.3f},{hi:.3f}]"


def describe(name, r):
    """One line saying what was scored and what was verified."""
    c, cand = r["config"], r["candidates"]
    where = (f"split={c['split']} horizon={c['horizon']} held-out source prompts={len(c['held_out_source_prompts'])}"
             if "split" in c else "split/horizon: not recorded in a table (the caller's)")
    checked = {True: "verified against the manifest", False: "FAILED the manifest check",
               None: "not verified (no manifest)"}[cand["verified"]]
    print(f"[{name}] {where}; label rate={c['label_rate']} retentions={c['retentions']}; "
          f"label counts consistent={r['label_count_consistent']}; candidate sets {checked}; "
          f"rows={r['n_rows']:,} decisions={r['n_decisions']:,} source prompts={r['n_source_prompts']}")


def print_rows(rows):
    w = 24
    head = (f"{'scorer':<34}{'AUC pooled':>{w}}{'AUC same-dec':>{w}}{'AUC cross-dec':>{w}}"
            f"{'recall@10%':>{w}}{'recall@20%':>{w}}")
    print(head)
    print("-" * len(head))
    for name, r in rows:
        pd = r["per_decision"]
        cell = lambda key, val: f"{val:.3f} {_ci(r, key)}" if "ci95" in r else f"{val:.3f}"
        print(f"{name[:33]:<34}{cell('auc_pooled', r['pooled']['auc']):>{w}}"
              f"{cell('auc_same_decision', r['pairs']['auc_same_decision']):>{w}}"
              f"{cell('auc_cross_decision', r['pairs']['auc_cross_decision']):>{w}}"
              f"{cell('macro_recall_0.10', pd['0.10']['macro_recall']):>{w}}"
              f"{cell('macro_recall_0.20', pd['0.20']['macro_recall']):>{w}}")
        if "calibration" in r:
            print(f"{'':<34}ECE {r['calibration']['ece']:.4f}  Brier {r['calibration']['brier']:.4f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", nargs="+", help="prediction table CSV(s) with columns "
                    + ",".join(P.TABLE_COLUMNS))
    ap.add_argument("--traces", help="comma-separated NAME=path.jsonl version-2 traces")
    ap.add_argument("--submission", help="a .py exposing score(F)")
    ap.add_argument("--split", choices=("source", "request"), default="source")
    ap.add_argument("--n-boot", type=int, default=P.N_BOOT)
    ap.add_argument("--probabilistic", action="store_true", help="--table scores are probabilities")
    ap.add_argument("--no-strict", action="store_true",
                    help="score tables that fail the label-count or the manifest check anyway")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="candidate manifest to verify every --table against (see --write-manifest)")
    ap.add_argument("--write-manifest", type=Path, default=None,
                    help="write the candidate manifest of the single --table (which must be complete) and exit")
    ap.add_argument("--json", type=Path, default=None, help="write all results here")
    a = ap.parse_args(argv)
    if not (a.table or a.traces):
        ap.error("give --table or --traces")
    if a.write_manifest:
        if not a.table or len(a.table) != 1:
            ap.error("--write-manifest takes exactly one --table")
        manifest = P.candidate_manifest(P.read_table(a.table[0]))
        P.write_manifest(a.write_manifest, manifest)
        print(f"WROTE {a.write_manifest}: {manifest['n_decisions']:,} decisions, {manifest['n_rows']:,} candidates")
        return
    manifest = P.read_manifest(a.manifest) if a.manifest else None
    rows = []
    for t in a.table or []:
        rows.append((Path(t).stem, P.evaluate_table(P.read_table(t), n_boot=a.n_boot, probabilistic=a.probabilistic,
                                                    strict=not a.no_strict, manifest=manifest)))
    if a.traces:
        corpus = P.load_corpus(a.traces)
        tr, te = P.split(corpus, a.split)
        print(f"corpus: {corpus['n_requests']} requests, {len(corpus['source_names'])} source prompts; "
              f"split={a.split}: {len(tr):,} train / {len(te):,} test rows\n")
        ref = fit_reference(corpus["F"], corpus["y"], tr)
        w = ref.weights.tolist()
        print(f"reference scorer: within+cross logistic refit on the {a.split}-split training rows "
              f"(w_within={w[0]:.4f}, w_cross={w[1]:.4f}, bias={float(ref.bias):.4f})"
              + (f"; submission: {a.submission}" if a.submission else ""))
        rows.append(("REFERENCE within+cross logistic",
                     P.evaluate(score_fn(ref), corpus, a.split, n_boot=a.n_boot, probabilistic=True,
                                strict=not a.no_strict)))
        if a.submission:
            name, fn, prob = load_submission(a.submission)
            rows.append(("SUBMISSION " + name, P.evaluate(fn, corpus, a.split, n_boot=a.n_boot,
                                                         probabilistic=prob, strict=not a.no_strict)))
    for name, r in rows:
        describe(name, r)
    print()
    print_rows(rows)
    if a.json:
        a.json.write_text(json.dumps({n: r for n, r in rows}, indent=2) + "\n")
        print("WROTE", a.json)


if __name__ == "__main__":
    main()

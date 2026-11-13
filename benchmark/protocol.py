"""KVSalienceBench protocol 2.0: decision-level evaluation of retention scorers.

Task. For every cache-retention decision (one request, layer and decode step), score
each candidate KV block by the probability that it lands in the top-``r`` most
attended blocks of that decision ``h`` steps ahead (``r = 0.10``, ``h = 4`` headline,
32-token blocks). The label is future attention among the candidates present at the
decision, with exactly ``ceil(r * n)`` positives per decision.

What changed from protocol 1.0 (``benchmark.legacy_v1``). Version 1 kept only pooled
metrics and discarded the decision identifiers, so it could not score the object a
selector actually acts on. Version 2 keeps them and therefore reports, together:

* pooled ranking (AUC, tie-aware AP, pooled top-k), which compares rows ACROSS
  decisions;
* the same pooled AUC split exactly into same-decision and cross-decision pairs;
* per-decision top-k recall at each retention in ``RETENTIONS`` (macro over decisions),
  with the number of decisions whose k-th place is a tie, and the recall under exact
  random tie-breaking;
* a bootstrap over SOURCE PROMPTS (all held-out requests of a drawn prompt move
  together), exact for the pooled statistic.

Splits. ``split='source'`` (default) holds out whole source prompts, the same ones for
every model, so text seen by one model in training never appears in another's test.
``split='request'`` reproduces the pooled request split of the paper's Table II
(``experiments/run_icdm_v2.py``): a global permutation of all requests, NOT stratified
by model.

What the evaluator checks, and what it cannot. A per-decision metric is only meaningful
on COMPLETE decisions: every candidate block of the decision must be a row.

* Label-count check (always). ``evaluate_table`` refuses a table whose decisions do
  not each hold exactly ``ceil(label_rate * n)`` positives (unless ``strict=False``)
  and reports the outcome as ``label_count_consistent``. This catches a row sample.
  It does NOT prove completeness: drop one negative from a decision of 20 candidates
  with 2 positives and the remaining 19 rows still hold ``ceil(0.1 * 19) = 2``.
* Candidate check (only with a manifest). ``candidate_manifest`` records, for every
  decision of a complete table, the number of candidates and a digest of their block
  indices. Pass it as ``manifest=`` and the evaluator verifies the decision set, every
  candidate count and every digest, and reports ``candidates.verified``. Without a
  manifest ``candidates.verified`` is ``None``: completeness is the caller's claim.
* Labels must be exactly 0 or 1 (finite); anything else is refused, not coerced.
* Never checked: how the scorer was trained. A prediction table cannot show that its
  scorer never saw the scored rows; report the split you trained under
  (``benchmark/splits/paper_v2_splits.json`` fixes both splits of the paper's corpus).
  This is an evaluator for a stated protocol, not a closed leaderboard.

Calibration (ECE, Brier) is reported only when the scorer declares
``probabilistic=True``: a fixed top-k budget needs the ordering, a thresholded
(variable-size) selector needs probabilities.

Two entry points: ``evaluate_table`` scores a prediction table (a CSV with the
columns in ``TABLE_COLUMNS``; no model weights or traces needed), and ``evaluate``
scores a ``score(F)`` function on a version-2 trace corpus.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from xqp.features import FEATURE_NAMES  # noqa: E402  ("s_within", "s_cross", "s_query", "s_pos")
from xqp.decision_eval import (DecisionData, Ranked, RequestStats, bootstrap_statistics,  # noqa: E402
                               cluster_multiplicities, macro_recall, pair_matrix,
                               percentile_ci, pooled_decomposition)
from xqp.dm_metrics import (average_precision, brier_score, expected_calibration_error,  # noqa: E402
                            precision_at_k)

PROTOCOL_VERSION = "kvsaliencebench-2.0"
HORIZONS = ("h1", "h4", "h16", "h64")
HEADLINE_H = "h4"
TOP_R = 0.10                 # label: block in the top 10% attended at t+h, per decision
BLOCK_SIZE = 32
RETENTIONS = (0.10, 0.20)    # per-decision budgets reported
TEST_FRAC = 0.25
SEED = 0
N_BOOT = 2000
FEATURE_COLUMNS = list(FEATURE_NAMES)
TABLE_COLUMNS = ("model_id", "source_id", "request_id", "layer", "step", "block_idx", "label", "score")
MANIFEST_SCHEMA = "kvsaliencebench/candidate-manifest/1"


# --------------------------------------------------------------------------- loading
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def load_trace(path, horizon: str = HEADLINE_H, verify_hash: bool = False) -> dict:
    """Stream one version-2 trace (``.meta.json`` and ``.cohort.json`` beside it are
    required) into arrays. Rows keep their file order."""
    path = Path(path)
    meta = json.loads(Path(str(path) + ".meta.json").read_text())
    cohort = json.loads(Path(str(path) + ".cohort.json").read_text())
    if meta.get("trace_version") != 2:
        raise ValueError(f"{path}: not a version-2 trace")
    if verify_hash and sha256_file(path) != meta["trace_sha256"]:
        raise ValueError(f"{path}: sha256 mismatch with its manifest")
    if horizon not in HORIZONS:
        raise ValueError(f"unknown horizon {horizon!r}")
    n = int(meta["rows"])
    rid = np.empty(n, np.int32)
    layer = np.empty(n, np.int16)
    step = np.empty(n, np.int16)
    blk = np.empty(n, np.int32)
    F = np.empty((n, 4), np.float32)
    y = np.empty(n, np.int8)
    i = 0
    with path.open() as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("trace_version") != 2:
                raise ValueError("mixed trace versions")
            rid[i] = int(r["request_id"][1:])
            layer[i], step[i], blk[i] = r["layer"], r["step"], r["block_idx"]
            F[i] = (r["f_within"], r["f_cross"], r["f_query"], r["f_pos"])
            y[i] = r[f"y_{horizon}"]
            i += 1
    if i != n:
        raise ValueError(f"{path}: read {i} rows, manifest says {n}")
    if not np.isfinite(F).all():
        raise ValueError("non-finite features; the version-2 collector should have rejected them")
    n_req = int(rid.max()) + 1
    source_of = {r["trace_id"]: (r["dataset"], r["id"]) for r in cohort["ids"]}
    if n_req != len(meta["requests"]) or n_req != len(source_of):
        raise ValueError("request count disagrees with manifest/cohort")
    return dict(request_id=rid, layer=layer, step=step, block_idx=blk, F=F, y=y, n_requests=n_req,
                source=[source_of[f"p{k}"] for k in range(n_req)],
                provenance=dict(trace=str(path), trace_sha256=meta["trace_sha256"], rows=n,
                                collector_sha256=meta["collector_sha256"]))


def load_corpus(specs, horizon: str = HEADLINE_H, verify_hash: bool = False) -> dict:
    """Pool several traces. ``specs`` is a list (or comma-separated string) of
    ``NAME=path.jsonl``. Request ids become global exactly as in ``run_icdm_v2.pool``
    (a running offset in the order given), so the paper's request split is reproducible.
    """
    if isinstance(specs, str):
        specs = specs.split(",")
    parts, names = [], []
    for spec in specs:
        name, _, path = spec.partition("=")
        if not path:
            raise ValueError(f"expected NAME=path, got {spec!r}")
        names.append(name)
        parts.append(load_trace(path, horizon, verify_hash))
    sources = sorted({s for p in parts for s in p["source"]})          # tuple order, as run_icdm_v2
    code = {s: i for i, s in enumerate(sources)}
    offset, req_source, cols = 0, [], {k: [] for k in
                                       ("model", "request", "layer", "step", "block_idx", "F", "y")}
    for mi, p in enumerate(parts):
        cols["model"].append(np.full(p["y"].shape[0], mi, np.int8))
        cols["request"].append(p["request_id"].astype(np.int64) + offset)
        for k in ("layer", "step", "block_idx", "F", "y"):
            cols[k].append(p[k])
        req_source.extend(code[s] for s in p["source"])
        offset += p["n_requests"]
    out = {k: np.concatenate(v) for k, v in cols.items()}
    out.update(model_names=names, source_names=[f"{a}/{b}" for a, b in sources],
               request_source=np.asarray(req_source, np.int32), n_requests=offset,
               horizon=horizon, provenance=[p["provenance"] for p in parts])
    return out


def split(corpus: dict, kind: str = "source", frac: float = TEST_FRAC, seed: int = SEED):
    """Train/test row indices. See the module docstring for the two kinds."""
    rng = np.random.default_rng(seed)
    if kind == "source":
        n = len(corpus["source_names"])
        held = rng.permutation(n)[: max(1, int(frac * n))]
        test_reqs = np.flatnonzero(np.isin(corpus["request_source"], held))
    elif kind == "request":
        uniq = np.unique(corpus["request"])
        test_reqs = rng.permutation(uniq)[: max(1, int(frac * len(uniq)))]
    else:
        raise ValueError(kind)
    is_test = np.isin(corpus["request"], test_reqs)
    return np.flatnonzero(~is_test), np.flatnonzero(is_test)


# ---------------------------------------------------------------------------- tables
def read_table(path) -> dict:
    """Read a prediction table (CSV, optionally gzipped) into column arrays."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in TABLE_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing columns {missing}; need {list(TABLE_COLUMNS)}")
        rows = list(reader)
    return dict(model_id=np.array([r["model_id"] for r in rows]),
                source_id=np.array([r["source_id"] for r in rows]),
                request_id=np.array([int(r["request_id"]) for r in rows], np.int64),
                layer=np.array([int(r["layer"]) for r in rows], np.int64),
                step=np.array([int(r["step"]) for r in rows], np.int64),
                block_idx=np.array([int(r["block_idx"]) for r in rows], np.int64),
                label=np.array([int(r["label"]) for r in rows], np.int8),
                score=np.array([float(r["score"]) for r in rows], np.float64))


def write_table(path, table: dict, float_digits: int = 8):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    n = len(table["label"])
    with opener(path, "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(TABLE_COLUMNS)
        for i in range(n):
            w.writerow([table["model_id"][i], table["source_id"][i], int(table["request_id"][i]),
                        int(table["layer"][i]), int(table["step"][i]), int(table["block_idx"][i]),
                        int(table["label"][i]), f"{float(table['score'][i]):.{float_digits}g}"])


def corpus_table(corpus: dict, idx, scores, readable: bool = True) -> dict:
    """Prediction table for the rows ``idx`` of a corpus under ``scores``. With
    ``readable=False`` the model and source columns stay integer codes (faster; fine
    for ``evaluate_table``, not for a CSV a person will read)."""
    idx = np.asarray(idx)
    req = corpus["request"][idx]
    model, source = corpus["model"][idx], corpus["request_source"][req]
    if readable:
        model = np.array(corpus["model_names"])[model]
        source = np.array(corpus["source_names"])[source]
    return dict(model_id=model, source_id=source,
                request_id=req, layer=corpus["layer"][idx].astype(np.int64),
                step=corpus["step"][idx].astype(np.int64),
                block_idx=corpus["block_idx"][idx].astype(np.int64),
                label=corpus["y"][idx], score=np.asarray(scores, np.float64))


# ------------------------------------------------------------------------- evaluation
def _codes(*cols):
    """Dense integer codes of the distinct rows of several equal-length columns, via a
    mixed-radix key (a 2-D ``np.unique`` is far too slow on millions of rows)."""
    key, radix_product = None, 1
    for c in cols:
        inv = np.unique(np.asarray(c), return_inverse=True)[1].reshape(-1).astype(np.int64)
        base = int(inv.max()) + 1
        radix_product *= base
        if radix_product >= 2 ** 62:
            raise ValueError("too many distinct values to encode one decision key")
        key = inv if key is None else key * base + inv
    return np.unique(key, return_inverse=True)[1].reshape(-1).astype(np.int64)


def _binary_labels(label) -> np.ndarray:
    """Labels as int8, refusing anything that is not exactly 0 or 1.

    ``label > 0`` would score a 2 as a positive and a -1 or a NaN as a negative, that
    is, evaluate an object the caller did not describe."""
    lab = np.asarray(label)
    if lab.ndim != 1:
        raise ValueError("label must be one-dimensional")
    if lab.dtype == bool:
        return lab.astype(np.int8)
    try:
        f = lab.astype(np.float64)
    except (TypeError, ValueError):
        raise ValueError("labels must be numeric 0/1") from None
    if not np.isfinite(f).all():
        raise ValueError("non-finite labels")
    if not np.isin(f, (0.0, 1.0)).all():
        bad = np.unique(f[~np.isin(f, (0.0, 1.0))])[:5].tolist()
        raise ValueError(f"labels must be exactly 0 or 1; found {bad}")
    return f.astype(np.int8)


def _decision_inventory(table: dict) -> dict:
    """{decision key: [candidate count, digest of the sorted block indices]}."""
    model = np.asarray(table["model_id"]).astype(str)
    req, layer, step = (np.asarray(table[c], np.int64) for c in ("request_id", "layer", "step"))
    blk = np.asarray(table["block_idx"], np.int64)
    dec = _codes(model, req, layer, step)
    order = np.lexsort((blk, dec))
    d_sorted, b_sorted = dec[order], blk[order].astype("<i8")
    starts = np.r_[0, np.flatnonzero(d_sorted[1:] != d_sorted[:-1]) + 1, d_sorted.size]
    inventory = {}
    for a, b in zip(starts[:-1], starts[1:]):
        i = order[a]
        key = f"{model[i]}|{int(req[i])}|{int(layer[i])}|{int(step[i])}"
        inventory[key] = [int(b - a), hashlib.sha256(b_sorted[a:b].tobytes()).hexdigest()[:16]]
    return inventory


def candidate_manifest(table: dict) -> dict:
    """Candidate inventory of a table that is KNOWN to be complete (e.g. built from a
    checksum-verified trace): per decision, the candidate count and a digest of the
    block indices. ``evaluate_table(..., manifest=...)`` checks a later table against
    it. The manifest holds no labels and no scores."""
    for c in ("model_id", "request_id", "layer", "step", "block_idx"):
        if c not in table:
            raise ValueError(f"table lacks column {c!r}")
    inventory = _decision_inventory(table)
    return dict(schema=MANIFEST_SCHEMA, key="model_id|request_id|layer|step",
                value="[candidate count, first 16 hex digits of sha256 over the sorted block indices as little-endian int64]",
                n_decisions=len(inventory), n_rows=int(sum(v[0] for v in inventory.values())),
                decisions=inventory)


def write_manifest(path, manifest: dict):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt") as fh:
        json.dump(manifest, fh)


def read_manifest(path) -> dict:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        manifest = json.load(fh)
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"{path}: not a candidate manifest ({MANIFEST_SCHEMA})")
    return manifest


def verify_candidates(table: dict, manifest: dict) -> dict:
    """Compare a table's decisions with a candidate manifest: the decision set, every
    candidate count and every block-index digest."""
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"not a candidate manifest ({MANIFEST_SCHEMA})")
    want, got = manifest["decisions"], _decision_inventory(table)
    shared = want.keys() & got.keys()
    size = sum(got[k][0] != want[k][0] for k in shared)
    ids = sum(got[k][0] == want[k][0] and got[k][1] != want[k][1] for k in shared)
    report = dict(manifest_decisions=len(want), missing_decisions=len(want.keys() - got.keys()),
                  unexpected_decisions=len(got.keys() - want.keys()),
                  decisions_with_wrong_candidate_count=int(size),
                  decisions_with_wrong_candidate_ids=int(ids))
    report["verified"] = not any(report[k] for k in (
        "missing_decisions", "unexpected_decisions", "decisions_with_wrong_candidate_count",
        "decisions_with_wrong_candidate_ids"))
    return report


def evaluate_table(table: dict, retentions=RETENTIONS, n_boot: int = N_BOOT, seed: int = SEED,
                   label_rate: float | None = TOP_R, probabilistic: bool = False,
                   strict: bool = True, manifest: dict | None = None) -> dict:
    """Score a prediction table under protocol 2.0 (see the module docstring).

    ``label_count_consistent`` reports the label-count check; it is not a completeness
    proof. ``candidates.verified`` is ``True``/``False`` when a ``manifest`` is given
    and ``None`` otherwise."""
    for c in TABLE_COLUMNS:
        if c not in table:
            raise ValueError(f"table lacks column {c!r}")
    n_rows = {c: np.asarray(table[c]).shape for c in TABLE_COLUMNS}
    if len(set(n_rows.values())) != 1 or len(next(iter(n_rows.values()))) != 1:
        raise ValueError(f"columns must be one-dimensional and equally long: {n_rows}")
    y = _binary_labels(table["label"])
    score = np.asarray(table["score"], np.float64)
    if not np.isfinite(score).all():
        raise ValueError("non-finite scores")
    if probabilistic and (score.min() < 0 or score.max() > 1):
        raise ValueError("probabilistic=True needs scores in [0, 1]")
    req = _codes(table["model_id"], table["request_id"])
    dec = _codes(table["model_id"], table["request_id"], table["layer"], table["step"])
    data = DecisionData.build(y, dec, req)
    # one source prompt per request
    src_codes = _codes(table["source_id"])
    cluster_of_req = np.full(data.n_req, -1, np.int64)
    cluster_of_req[data.req] = src_codes
    if not np.array_equal(cluster_of_req[data.req], src_codes):
        raise ValueError("a request lists more than one source_id")
    if np.unique(_codes(dec, table["block_idx"])).size != y.size:
        raise ValueError("duplicate (decision, block) rows")
    consistent = None
    if label_rate is not None:
        want = np.ceil(label_rate * data.group_size.astype(np.float64))
        bad = int((data.npos_g != want).sum())
        consistent = bad == 0
        if bad and strict:
            raise ValueError(
                f"label-count check failed: {bad} of {data.n_groups} decisions do not hold exactly "
                f"ceil({label_rate} * n) positives. Per-decision metrics need complete decisions, not a "
                "row sample (pass strict=False to score anyway, or label_rate=None for another label rate)")
    candidates = dict(verified=None, note="no manifest given: that every decision lists all of its "
                                          "candidate blocks is the caller's claim, not a checked fact")
    if manifest is not None:
        candidates = verify_candidates(table, manifest)
        if not candidates["verified"] and strict:
            raise ValueError(
                "candidate check failed against the manifest: "
                + ", ".join(f"{k}={v}" for k, v in candidates.items() if k not in ("verified", "manifest_decisions") and v)
                + " (pass strict=False to score anyway)")
    rk = Ranked(data, score)
    dec_parts = pooled_decomposition(rk)
    out = dict(protocol=PROTOCOL_VERSION, n_rows=int(y.size), n_decisions=int(data.n_groups),
               n_requests=int(data.n_req), n_source_prompts=int(np.unique(src_codes).size),
               positive_rate=float(y.mean()), label_count_consistent=consistent, candidates=candidates,
               config=dict(label_rate=label_rate, retentions=[float(r) for r in retentions],
                           n_boot=int(n_boot), seed=int(seed), probabilistic=bool(probabilistic),
                           strict=bool(strict), block_size=BLOCK_SIZE),
               not_checked=["that the scorer was fit without the scored rows",
                            "which split and horizon a prediction table was built from"])
    out["pooled"] = dict(auc=dec_parts["auc_pooled"], ap=average_precision(y.astype(np.float32), score),
                         p_at_k=precision_at_k(y.astype(np.float32), score, TOP_R))
    out["pairs"] = dict(auc_same_decision=dec_parts["auc_same"], auc_cross_decision=dec_parts["auc_cross"],
                        auc_same_decision_macro=dec_parts["auc_same_macro"],
                        share_same_decision=dec_parts["share_same"])
    out["per_decision"] = {}
    for r in retentions:
        tk = rk.topk(r, ties="random")
        out["per_decision"][f"{r:.2f}"] = dict(
            macro_recall=macro_recall(data, tk["tp"]),
            macro_recall_random_ties=macro_recall(data, tk["tp_random_ties"]),
            boundary_tie_share=float(tk["boundary_tie"].mean()))
    out["ties"] = dict(tied_row_share=rk.tied_row_share())
    if probabilistic:
        out["calibration"] = dict(ece=expected_calibration_error(y.astype(np.float32), score),
                                  brier=brier_score(y.astype(np.float32), score))
    if n_boot:
        M = cluster_multiplicities(cluster_of_req, n_boot, seed)
        st = RequestStats.build(rk, pair_matrix(data, score), tuple(retentions))
        b = bootstrap_statistics(M, st)
        out["ci95"] = dict(unit="source_prompt", n_boot=int(n_boot),
                           statistic="the pooled and decision-macro statistics recomputed on each draw of source prompts",
                           auc_pooled=percentile_ci(b["auc_pooled"]),
                           auc_same_decision=percentile_ci(b["auc_same"]),
                           auc_cross_decision=percentile_ci(b["auc_cross"]),
                           **{f"macro_recall_{r:.2f}": percentile_ci(b[f"recall_{r:g}"]) for r in retentions})
    return out


def evaluate(score_fn, corpus: dict, split_kind: str = "source", retentions=RETENTIONS,
             n_boot: int = N_BOOT, seed: int = SEED, probabilistic: bool = False, strict: bool = True) -> dict:
    """Score ``score_fn`` (an (N,4) feature matrix in ``FEATURE_COLUMNS`` order -> N
    scores) on the held-out part of a version-2 corpus."""
    _, te = split(corpus, split_kind, seed=seed)
    s = np.asarray(score_fn(corpus["F"][te]), np.float64).reshape(-1)
    if s.shape[0] != te.shape[0]:
        raise ValueError(f"score_fn returned {s.shape[0]} scores for {te.shape[0]} rows")
    res = evaluate_table(corpus_table(corpus, te, s, readable=False), retentions, n_boot, seed, TOP_R,
                         probabilistic, strict)
    held = np.unique(corpus["request"][te])
    res.update(split=split_kind, horizon=corpus["horizon"])
    res["candidates"] = dict(verified=None, note="every held-out row of the trace corpus is scored; "
                             "load_corpus(verify_hash=True) checks each trace against its manifest")
    res["config"].update(split=split_kind, horizon=corpus["horizon"], test_fraction=TEST_FRAC,
                         models=list(corpus["model_names"]), held_out_requests=int(held.size),
                         held_out_source_prompts=sorted({corpus["source_names"][c]
                                                         for c in corpus["request_source"][held]}),
                         traces=[p["trace_sha256"] for p in corpus["provenance"]])
    res["not_checked"] = ["that the scorer was fit without the held-out rows"]
    return res


def reference_scorer(path: str | os.PathLike = ROOT / "benchmark/reference_model_v2.json"):
    """The version-2 reference baseline: the two-view logistic of Table II, refit on
    the request split. Returns ``(name, score_fn)``."""
    d = json.loads(Path(path).read_text())
    w = np.array([d["weights"][c] for c in FEATURE_COLUMNS], np.float32)
    b = np.float32(d["bias"])

    def score(F):
        z = np.asarray(F, np.float32) @ w + b
        return 1.0 / (1.0 + np.exp(-z.astype(np.float64)))
    return d["name"], score

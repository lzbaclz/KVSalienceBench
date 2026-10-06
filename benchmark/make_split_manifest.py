"""Write the fixed train/test split manifest of the paper's version-2 corpus.

A prediction table cannot show how its scorer was trained, so the held-out sets have
to be public and fixed. This writes them for the two-model corpus of the paper, for
both splits of ``benchmark.protocol.split``:

* ``source`` (the evaluator's default): the same 32 source prompts are held out for
  both models, so no held-out text occurs in training;
* ``request`` (the paper's Table II): 64 of the 256 model-requests are held out by a
  global permutation. They cover 57 source prompts: 7 with both models' requests held
  out and 50 whose other model's request is in training. That split measures transfer
  to new model-requests, not to unseen text.

No traces are needed: the two tracked cohort files give the source prompt of every
request and both splits are a seeded permutation. The result is checked against the
held-out request lists recorded in ``experiments/results/icdm_v2.json``.

    python benchmark/make_split_manifest.py   # rewrites benchmark/splits/paper_v2_splits.json
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark import protocol as P  # noqa: E402

COHORTS = (("Llama-3.1-8B-Instruct", "experiments/results/physical_kv/query-v2/query-v2.llama.bs32.jsonl.cohort.json"),
           ("Qwen2.5-7B-Instruct", "experiments/results/physical_kv/query-v2/query-v2.qwen.bs32.jsonl.cohort.json"))
OUT = ROOT / "benchmark/splits/paper_v2_splits.json"
SCHEMA = "kvsaliencebench/split-manifest/1"


def request_corpus(cohorts=COHORTS) -> dict:
    """One row per request, with the fields ``benchmark.protocol.split`` reads; request
    ids and source codes are built exactly as ``load_corpus`` builds them."""
    per_model, digests = [], []
    for _, rel in cohorts:
        cohort = json.loads((ROOT / rel).read_text())
        source_of = {r["trace_id"]: (r["dataset"], r["id"]) for r in cohort["ids"]}
        per_model.append([source_of[f"p{k}"] for k in range(len(source_of))])
        digests.append(cohort["data_sha256"])
    sources = sorted({s for m in per_model for s in m})
    code = {s: i for i, s in enumerate(sources)}
    request_source = np.asarray([code[s] for m in per_model for s in m], np.int32)
    model = np.concatenate([np.full(len(m), i, np.int8) for i, m in enumerate(per_model)])
    return dict(request=np.arange(request_source.size, dtype=np.int64), request_source=request_source, model=model,
                source_names=[f"{a}/{b}" for a, b in sources], model_names=[n for n, _ in cohorts],
                cohort_data_sha256=digests)


def build() -> dict:
    corpus = request_corpus()
    names, models = corpus["source_names"], corpus["model_names"]
    recorded = json.loads((ROOT / "experiments/results/icdm_v2.json").read_text())
    splits = {}
    for kind in ("source", "request"):
        train, test = P.split(corpus, kind)
        if sorted(test.tolist()) != sorted(recorded[f"pooled_{kind}_split"]["held_out_requests"]):
            raise SystemExit(f"{kind} split differs from the one recorded in icdm_v2.json")
        held_sources = sorted({names[c] for c in corpus["request_source"][test]})
        train_sources = {names[c] for c in corpus["request_source"][train]}
        per_source = np.bincount(corpus["request_source"][test], minlength=len(names))
        splits[kind] = dict(
            n_held_out_requests=int(test.size), n_training_requests=int(train.size),
            held_out_requests_by_model={m: int((corpus["model"][test] == i).sum()) for i, m in enumerate(models)},
            n_held_out_source_prompts=len(held_sources),
            source_prompts_with_every_model_held_out=int((per_source == len(models)).sum()),
            held_out_source_prompts_also_in_training=int(sum(s in train_sources for s in held_sources)),
            held_out_requests=[dict(request=int(r), model=models[corpus["model"][r]],
                                    source=names[corpus["request_source"][r]]) for r in sorted(test.tolist())])
    return dict(schema=SCHEMA, protocol=P.PROTOCOL_VERSION, seed=P.SEED, test_fraction=P.TEST_FRAC,
                corpus=dict(models=models, cohort_files=[rel for _, rel in COHORTS],
                            cohort_data_sha256=corpus["cohort_data_sha256"], n_requests=int(corpus["request"].size),
                            n_source_prompts=len(names),
                            request_index="position in the model order above: model i owns requests "
                                          "128*i .. 128*i+127, in its trace's prompt order (p0..p127)",
                            source="<LongBench dataset>/<LongBench row id>"),
                note=("source: the evaluator's default, no held-out text in training. request: the paper's "
                      "Table II, new model-requests whose source prompt usually occurs in training through "
                      "the other model."),
                splits=splits)


def main():
    manifest = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(manifest, indent=1) + "\n")
    for kind, s in manifest["splits"].items():
        print(f"{kind:<8} held-out requests {s['n_held_out_requests']} {s['held_out_requests_by_model']}  "
              f"source prompts {s['n_held_out_source_prompts']}  both models held out "
              f"{s['source_prompts_with_every_model_held_out']}  also in training "
              f"{s['held_out_source_prompts_also_in_training']}")
    print(f"WROTE {OUT}")


if __name__ == "__main__":
    main()

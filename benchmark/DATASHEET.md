# KVSalienceBench protocol

`benchmark.protocol` evaluates KV retention scores at the cache-decision level:
one model request, layer and decode step. Protocol 2.0 reports pooled AUC,
tie-aware AP, same-decision/cross-decision AUC, and per-decision recall at 10%
and 20% retention. Probabilistic scores also receive ECE and Brier metrics.
Bootstrap uncertainty resamples source prompts.

## Prediction tables

CSV columns: `model_id,source_id,request_id,layer,step,block_idx,label,score`.

```bash
python -m benchmark.run_leaderboard \
  --table benchmark/example/prediction_table.csv \
  --manifest benchmark/example/candidate_manifest.json --probabilistic
```

Labels must be binary and scores finite. Each request belongs to one source
prompt, each decision/block pair must be unique, and each decision must contain
exactly `ceil(0.1 * n)` positive labels in strict mode. The positive label denotes
future top-decile attention among candidates present at the feature step.

The label-count check is not a completeness proof. A candidate manifest records
the decision set, candidate counts and block-index digests. Supply `--manifest`
to verify the inventory. Without one, `candidates.verified` is null. A table
cannot establish that training excluded the scored rows.

The included example is synthetic: 5,748 rows and 192 decisions. Its expected
metrics are a regression fixture, not empirical results.

## Trace evaluation and splits

Use `--traces Llama=/data/llama.v2.jsonl,Qwen=/data/qwen.v2.jsonl` with version-2
traces and their checksum/cohort sidecars. `submit_template.py` documents the
custom scorer interface. The collector uses 32-token blocks, attention EMA
decay 0.9, and horizons 1, 4, 16 and 64. Queries and cached keys share post-RoPE
coordinates. See `xqp/features.py` and `xqp/attn_trace_extract.py`.

The default `source` split holds out source prompts across models, so their text
cannot occur in training through another model. The `request` split holds out
model-requests and can share source text with training. Seed 0 and a test
fraction of 0.25 are defined in `benchmark/protocol.py`.

The frozen two-model split is in `splits/paper_v2_splits.json`. Accompanying
cohort manifests contain IDs and digests only. `make_split_manifest.py` rebuilds
both held-out sets; tests compare the full resulting manifest. The source split
holds out 32 sources/64 model-requests; the request split holds out 64 requests
covering 57 sources, of which 50 also occur in training.

`reference_model_v2.json` contains frozen within+cross logistic coefficients.
The trace runner refits the reference on the training rows of the chosen split.

## Interpretation

Pooled AUC, per-decision recall and answer quality measure different properties.
Report the split, retention budget and tie rule. Logical masking does not
reclaim physical K/V memory. The physical validator checks tensor storage;
neither implementation is a production serving engine.

`legacy_v1/` preserves the older pooled-metric API. Its outputs must retain the
version-1 protocol label. Raw traces, model weights, datasets and generated
experimental results are supplied locally and excluded from Git.

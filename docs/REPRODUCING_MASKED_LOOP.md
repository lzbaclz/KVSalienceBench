# Reproducing the pinned masked loop

The exact MIT-licensed SEER source is included in `third_party/`; upstream
repository access is no longer required. `scripts/prepare_seer.py` verifies
65 Python files against **all 70** recorded v2 cells before extracting them.
The upstream files are unchanged; the prefill-attention memory patch is applied
explicitly by `experiments/seer_pinned/run_cell.py`.

## Environment and complete CPU example

Use Python 3.11, PyTorch 2.6.0 and Transformers 4.51.3. The GPU experiment used
PyTorch 2.6.0+cu124. For an independent CPU installation:

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[test,validation]'
python experiments/seer_pinned/smoke_cpu.py
KVSALIENCE_SEER_ROOT="$PWD/build/seer-5a7fbee19045" python -m pytest -q tests/test_seer_pinned_patch.py
```

The example runs a random two-layer Llama through actual masked decoding,
with and without the memory patch, and compares tokens, retained counts and
served-oracle misses. It needs no external data and makes no empirical quality
claim. For GPU reproduction, install the PyTorch 2.6.0 CUDA 12.4 wheel instead;
keep Transformers pinned. The paper does not report rerun latency because the
original GPUs were shared.

The `validation` extra includes pandas: importing the pinned simulator's policy
registry loads `seer.lap.features`, which imports pandas even for the CPU example.
Use the complete extra above in a fresh environment; an existing analysis
environment may already contain this dependency and conceal an incomplete install.

## Model and data provenance

| Model ID | Verified retrieval revision | Experiment dtype | Immutable weight identity |
|---|---|---|---|
| `meta-llama/Llama-3.1-8B-Instruct` | [`0e9e39f249a16976918f6564b8830bc894c89659`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/tree/0e9e39f249a16976918f6564b8830bc894c89659) | bfloat16 | four shard SHA256 values under `icdm_v2.json:provenance:Llama-3.1-8B-Instruct:model_weight_sha256` |
| `Qwen/Qwen2.5-7B-Instruct` | [`a09a35458c702b33eeacc393d103063234e8bc28`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/tree/a09a35458c702b33eeacc393d103063234e8bc28) | bfloat16 for this loop; float32 for the separate physical check | four shard SHA256 values under `icdm_v2.json:provenance:Qwen2.5-7B-Instruct:model_weight_sha256` |

The trace manifest did not record upstream Hub commits. On 2026-09-22, the
official Hub API's LFS SHA256 values for all eight shards at the revisions above
were matched to the recorded experiment hashes and to the local weights.
Qwen's local download metadata also identifies the listed revision. Llama's
listed revision is a verified retrieval location for identical weights, not
a retrospectively inferred original producer revision. Keep each snapshot's
matching tokenizer and configuration; weight identity alone does not establish
that every historical auxiliary file was identical. Meta's access terms require the reader's
own model access; no weights or access credentials are distributed here.

LongBench is available from its [official dataset repository](https://huggingface.co/datasets/zai-org/LongBench)
(the earlier `THUDM/LongBench` URL redirects there), pinned for retrieval to
`5e628be450b7e67fb7ae6e201bd6d8f7056f7672`. Its `data.zip` SHA256 is
`cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64`,
matching the local archive. Download `data.zip`, unpack
its `data/` directory, and verify the seven JSONL files. The exact file hashes,
runner arguments, package versions and row-selection rule are stored in every
`experiments/results/expand_v2/<model>/<task>_<policy>.json.provenance.json`.
The masked loop selects the canonical first 64 rows, without shuffling, for
`qasper`, `narrativeqa`, `hotpotqa`, `multifieldqa_en`, `2wikimqa`, `musique`
and `triviaqa`. This differs from the frozen physical/trace cohort; do not
substitute `prepare_physical_qa.py` inputs into the masked-loop experiment.

```bash
hf download meta-llama/Llama-3.1-8B-Instruct --revision 0e9e39f249a16976918f6564b8830bc894c89659 --local-dir assets/llama --exclude '*.bin' 'original/*'
hf download Qwen/Qwen2.5-7B-Instruct --revision a09a35458c702b33eeacc393d103063234e8bc28 --local-dir assets/qwen --exclude '*.bin'
hf download zai-org/LongBench data.zip --revision 5e628be450b7e67fb7ae6e201bd6d8f7056f7672 --repo-type dataset --local-dir assets/longbench
unzip -n assets/longbench/data.zip -d assets/longbench
python scripts/check_reproduction_assets.py --model llama --model-dir assets/llama --longbench-dir assets/longbench/data
python scripts/check_reproduction_assets.py --model qwen --model-dir assets/qwen
```

These commands retrieve external licensed assets, not files to commit. The hash
checker fails on mismatches; a changed upstream file is a new experiment, not
an exact rerun. The historical upstream snapshot ID is unrecorded, and hashes
alone cannot guarantee continued availability of matching files.

## One complete GPU cell and the full sweep

From the repository root, after the asset checks:

```bash
python scripts/prepare_seer.py
python experiments/seer_pinned/run_cell.py \
  --seer-root build/seer-5a7fbee19045 \
  --model "$PWD/assets/llama" --longbench-dir "$PWD/assets/longbench/data" \
  --dataset qasper --policy xqp --dtype bfloat16 \
  --context-length 4096 --num-requests 64 --max-new-tokens 48 \
  --hbm-budget 0.20 --decision-period 8 --seed 0 \
  --out build/reproduction/qasper_xqp.json
```

Chat formatting is enabled by default, and `SEER_STRICT_WORKLOAD=1` prevents
synthetic fallback. A fresh output path is required. This is the full 64-request
paper configuration for one cell, not a shortened replacement experiment.
Repeat for both models, all seven tasks and `{full,h2o,xqp,adakv,pyramidkv}`;
`experiments/seer_pinned/run_sweep.sh` automates the same matrix with configurable
paths. Re-score with `experiments/analyze_expand_sensitivity.py` (see `--help`)
using the same LongBench files and the new result root. Do not overwrite the
archived cells. This reproduces the tested shared-selection comparators, not
the original H2O/Ada-KV implementations.

## CPU reanalysis for this review

```bash
python experiments/analyze_oracle_budget.py --root experiments/results/e2e_confirm --quiet --out build/oracle-4k.json
python experiments/analyze_oracle_budget.py --root experiments/results/longctx/c16384_n64 --quiet --out build/oracle-16k.json
python experiments/analyze_review_statistics.py --out build/source-bootstrap.json
python experiments/analyze_review_statistics.py --replay experiments/results/mentor_statistics.json
python experiments/audit_oracle_count_mapping.py --out build/oracle-count-mapping.json
```

The `--out` statistics command requires both checksum-matched raw v2 traces and their sidecars
at the paths recorded in `icdm_v2.json`, about 9 GiB of JSONL. They are not
committed. Obtain that optional trace package from the corresponding author, or
collect new traces using the versioned collector (a new numerical execution).
`mentor_statistics.json` includes cluster AUC sufficient statistics and all
per-request recall values, so the 10,000-draw interval calculation can also be
checked with `--replay` without distributing those large raw traces. All preceding CPU examples
and oracle reanalysis work directly from the delivered repository.

The count-mapping audit needs the extracted pinned source and PyTorch, but no
model forward pass or GPU. It distinguishes verified source invariants from
missing historical per-layer records; see the manuscript's Exp#7. Earlier
`forced_floor` JSON field names are preserved for compatibility, while the
current analysis explicitly labels them as conditional logged-count diagnostics.

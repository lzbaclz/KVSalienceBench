# Physical KV validation

The reference backend compares irreversible masking with physical compaction of
retained K/V tensors. It preserves absolute positions, prevents resurrection and
keeps a fixed prompt-derived block budget. It measures F1, local latency, live
payload, physical storage and allocator peaks.

Supported: Transformers **4.51.3**, Llama/Qwen2, eager attention, a single device,
batch size one and non-quantized K/V. Unsupported cache APIs, families, sliding
windows and sharding are rejected. Full prefill remains uncompressed.

## Environment and inputs

```bash
python3.11 -m venv .venv-physical
source .venv-physical/bin/activate
# Example for CUDA 12.4; select the PyTorch wheel for your machine.
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e '.[test,validation]'
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
```

Supply a local checkpoint and English QA JSONL with unique `dataset`/`id`,
`prompt` and nonempty `answers`:

```json
{"id":"1","dataset":"example","prompt":"Passage ... Question ...","answers":["answer"]}
```

The preparation utility also accepts LongBench-style `context`/`input` rows:

```bash
export MODEL=/data/models/Llama-3.1-8B-Instruct
python experiments/prepare_physical_qa.py \
  --model "$MODEL" \
  --input qasper=/data/longbench/qasper.jsonl \
  --input narrativeqa=/data/longbench/narrativeqa.jsonl \
  --per-dataset 64 --seed 0 --max-input-tokens 4096 --chat \
  --truncate-middle --out /data/physical-qa.jsonl
```

Preparation records hashes, selected IDs, lengths and truncation flags. Omit
`--truncate-middle` to reject long prompts. Use the same input file, chat format,
tokenizer and generation settings for all compared policies.

## Correctness gates and measurements

```bash
python experiments/run_physical_kv.py \
  --model "$MODEL" --data /data/physical-qa.jsonl \
  --policy xqp_reconstructed --backend physical --budget 0.2 \
  --device cuda:0 --dtype bfloat16 --chat --gate-only \
  --hash-model-weights --out /data/physical-gates.json
```

Gates compare native full-cache behavior, then fixed-token/fixed-selection
masked and physical logits, positions and storage. Failures exit nonzero and
write `.failed.json`. Numerical tolerance, top-1 agreement and free-running
token agreement are separate measurements. `--gate-requests`, `--gate-tokens`
and `--gate-steps` expand the short default preflight.

```bash
export QA_DATA=/data/physical-qa.jsonl
export OUT=/data/physical-results
export CHAT=1 DTYPE=bfloat16 MAX_INPUT=4096 MAX_NEW=128 REPEATS=1
bash scripts/run_physical_kv_matrix.sh
```

The matrix launches full cache and two policies at two budgets on masked and
physical backends in fresh processes. Selection and compaction are timed.
Choose fresh output paths for each run. Tokenization, loading and warm-up are
outside generation timing. Use an idle GPU for comparable latency measurements.

The per-layer budget is `floor(fraction * ceil(prompt_tokens / block_size))`.
Defaults are 32-token blocks with one sink and one recent block. `h2o_block`
is a shared-head block accumulator. `xqp_reconstructed` uses included coefficients
with explicit online features, which differ from offline feature extraction.

## Analyze

```bash
python experiments/analyze_physical_kv.py --help
python experiments/summarize_physical_kv.py \
  --run-dir "$OUT" --qa-meta /data/physical-qa.jsonl.meta.json
```

Pairing checks IDs, input tokens, hashes, runtime and generation settings.
Gate-only outputs are not measurement cells. Summaries report masked/physical
token and F1 agreement, retained storage and allocator peaks. `--check` verifies
a saved summary against its inputs. The query-feature probe is documented by
`experiments/run_query_control_v2.py --help`.

These are local reference measurements, not production serving throughput.
Weights, datasets, traces and generated results remain local and outside Git.

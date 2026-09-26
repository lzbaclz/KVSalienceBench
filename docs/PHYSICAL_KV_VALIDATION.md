# Physical KV validation: operator runbook

## 0. What this experiment is—and is not

This is a new, single-request, eager-attention reference experiment. It checks
whether irreversible KV selection actually reduces retained K/V tensor storage
and preserves numerical behavior relative to an **irreversible** masking
reference. It measures local generation latency, task quality and allocator
statistics. It does **not** implement a paged serving engine, CPU offload,
PCIe/NVLink transfer, continuous batching, network queues or production goodput.
The manuscript reports the completed Llama (bfloat16) and Qwen (float32) matrices as an implementation check, not as a serving result.

The implementation is restricted to **Transformers 4.51.3, Llama and Qwen2**, a
single device, batch size one, eager attention, ordinary non-quantized K/V and no
sliding-window attention. This includes the intended Llama-3.1 and Qwen2.5 model
families when those local checkpoints load through these adapters. Qwen3,
Mistral, FlashAttention/SDPA, model sharding and vLLM are deliberately rejected
rather than silently approximated. Use a separate environment; do not downgrade
your production serving environment in place.

## 1. Environment and hardware

Example for the project's existing CUDA 12.4 generation of PyTorch:

```bash
python3.11 -m venv .venv-physical
source .venv-physical/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e '.[test,validation]'
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -c 'import torch,transformers; print(torch.__version__,transformers.__version__,torch.cuda.is_available()); assert transformers.__version__=="4.51.3"; assert torch.cuda.is_available()'
pytest -q
```

Choose the PyTorch CUDA wheel matching your driver and compute environment;
the above is an example, not an assertion that any NVIDIA driver supports it.
An A100 40/80GB is a reasonable starting machine for these 7–8B full-precision
checkpoints, but measured peak depends on model, context, eager-attention
workspace and prompt chunk size. Full prefill remains full-cache; the validator
does not let an otherwise-too-large prompt fit by evicting during prefill.

No model or dataset is downloaded automatically. Supply local, legitimately
available HF model directories and local English QA JSONL. Use an otherwise idle
GPU. Run one cell at a time. Capture `nvidia-smi`, driver version, GPU utilization
and any co-tenancy outside the timed experiment. Never compare runs made on
unrecorded different GPUs as a policy improvement.

Official runtime references:
- https://pytorch.org/get-started/previous-versions/
- https://huggingface.co/docs/transformers/v4.51.3/en/cache_explanation

## 2. Freeze a small, explicit QA cohort

The runner accepts one JSON object per line:

```json
{"id":"example-1","dataset":"qasper","prompt":"Read this passage ... Question ...","answers":["reference answer","alternative answer"]}
```

`id`, `dataset`, `prompt` and at least one nonempty answer are mandatory. IDs
must be unique **within each dataset**. English token F1 is not a substitute for
ROUGE, code evaluation, Chinese tokenization or an official multi-task scorer.

Existing LongBench-style local rows with `context`, `input`, `answers` and
optionally `_id` are supported by the preparation utility:

```bash
export MODEL=/public/model_zoo/Llama-3.1-8B-Instruct
python experiments/prepare_physical_qa.py \
  --model "$MODEL" \
  --input qasper=/data/longbench/qasper.jsonl \
  --input narrativeqa=/data/longbench/narrativeqa.jsonl \
  --per-dataset 64 --seed 0 --max-input-tokens 4096 --chat \
  --truncate-middle --out /data/physical-qa-v1.jsonl
```

This produces a deterministic 128-request cohort and a sidecar with source
hashes, selected IDs, original/final lengths and explicit truncation flags.
Middle truncation can remove evidence; it is an experimental condition, not a
free repair. Omit `--truncate-middle` to reject overlong prompts. The exact
rendered input is checked again by the runner. Preparation and execution must
both use or both omit `--chat`. Head/tail truncation and the neutral QA prompt
are **new protocol choices**, not a retroactive reproduction of the historical
external SEER/LongBench driver. Give every method the exact same prepared file.

Do not select prompts after inspecting policy outcomes. Two datasets are enough
for a descriptive implementation check, not for a strong general equivalence
claim. More samples or datasets should be planned before seeing differences.

## 3. Run correctness gates before collecting measurements

```bash
python experiments/run_physical_kv.py \
  --model "$MODEL" --data /data/physical-qa-v1.jsonl \
  --policy xqp_reconstructed --backend physical --budget 0.2 \
  --device cuda:0 --dtype bfloat16 --chat --gate-only \
  --hash-model-weights --out /data/physical-gates-v1.json
```

A nonzero exit or `*.failed.json` is a failure, not a benchmark result. Do not
increase tolerances just to pass. Inspect the failed layer/cache/position logic
and repeat with FP32 or a smaller diagnostic checkpoint to isolate the cause.
Numerical tolerances and maximum errors are stored; a pass is not a proof of
bitwise-identical greedy output for arbitrary near-tied logits.

Gates check native full-cache parity, then fixed-teacher-token/fixed-kept-set
parity between the new dense irreversible mask and compact K/V storage. The
same absolute positions and causal mask are used. Compact physical storage
must equal its live payload and be smaller than the dense reference at a
nontrivial budget. Tests additionally cover long decode growth, independent
storage, old-storage poisoning, no resurrection, malformed inputs, hook cleanup,
first-token handling, EOS and unsupported model rejection.

The gate has three evidence levels, and a pass at one level is not a claim at the
next: (1) the **numeric tolerance** on every gated logit row (the pass condition;
bf16 uses atol 0.15 / rtol 0.02, fp32 2e-5 / 2e-4); (2) **top-1 agreement** at every
gated step, stored separately as `top1_agreement` because two rows can be within
tolerance and still disagree on a near-tied argmax; (3) **free-running greedy
parity** of the measured masked and physical cells (`exact_token_match` in
`summary.json`), which is the only level that speaks to whole trajectories. Task F1
closeness is yet another statement. Report all of them, never the first as the last.

The gate uses at most 1,024 prompt tokens and the first request by default. This
is deliberately only a preflight; `--gate-requests 0,1,64,65 --gate-tokens 4096
--gate-steps 40` gates several prompts across both datasets at the full input cap
with repeated eviction across block boundaries (`experiments/results/physical_kv/
extended_gates/` holds such runs for both models). Each real request also runs budget/finite/storage checks, and the
full masked-vs-physical cohort remains part of the validation. For short prompts,
ensure the configured sink/recent windows fit the floor-rounded budget. The
runner rejects an impossible budget rather than increasing it silently.

## 4. Execute the small matrix

```bash
export MODEL=/public/model_zoo/Llama-3.1-8B-Instruct
export QA_DATA=/data/physical-qa-v1.jsonl
export OUT=/data/physical-kv/llama3.1-v1
export CHAT=1 DTYPE=bfloat16 MAX_INPUT=4096 MAX_NEW=128 REPEATS=1
bash scripts/run_physical_kv_matrix.sh
```

One matrix comprises nine fresh-process cells: full physical cache, then two
policies × two budgets (0.2/0.3) × two backends. `REPEATS=3` is useful for timing
variation; the same prompts decoded greedily three times are **not three new
independent quality cohorts**. A second checkpoint requires a separately frozen
cohort if its tokenizer changes the input cap. Never pool results by row number.

Each policy includes feature extraction, ranking and compaction in the measured
request path. No precomputed oracle schedules are used for the measured runs.
Fixed schedules occur only in the correctness gate. The initial full prefill
and first compaction have a separate peak-memory window from decode. Tokenization,
model loading and warm-up are outside local latency. Scalar token delivery waits
for computation; no heavy profiler runs inside the headline measurements.

For a controlled-output-length timing study, run a **separate** matrix with
`FIXED_OUTPUT=1`, a new output directory and the same `MAX_NEW`. It ignores EOS
and deliberately reports `f1=null`. Do not use that study as an answer-quality
experiment. Natural-EOS latency comparisons must disclose different output
lengths.

A focused single cell is also supported:

```bash
python experiments/run_physical_kv.py \
  --model "$MODEL" --data "$QA_DATA" --policy h2o_block \
  --backend physical --budget 0.2 --chat --dtype bfloat16 \
  --max-input-tokens 4096 --max-new-tokens 128 \
  --hash-model-weights --out /data/h2o-physical-b02.json
```

Output names are immutable. The shell script fails on the first unsuccessful
cell. It does not skip corrupt or partial files; use a new run directory after
investigating a failure. Weight hashing adds startup time outside measurements.

## 5. Exact selector semantics

The budget is `floor(fraction * ceil(prompt_tokens / block_size))` blocks per
layer and remains fixed during generation. Default blocks are 32 tokens, one
initial sink block and one most-recent block. A partially filled block counts
as one block. Budgets are shared across KV heads within a layer. New generated
tokens remain eligible; previously deleted tokens cannot return, even when a
later token happens to share their original logical block number.

`h2o_block` is an explicitly labeled **H2O-style shared-head block accumulator**,
not the original token/head-wise H2O implementation. Observations average query
heads, observation queries and tokens in each original block. Initial statistics
come from the final 64 prompt queries, followed by actual decode attention.

`xqp_reconstructed` loads the checked-in two-view coefficient checkpoint and
uses normalized current-layer EMA plus a hot-block indicator reconstructed from
the previous layer's EMA. The first-layer cross view is zero, and missing previous-
layer block IDs map to zero. Query and age coefficients must be exactly zero.
The logistic score is monotone, so ranking uses its logit. This is a precisely
specified **new online realization**, not a claim that the old unpinned SEER
scorer had identical feature or fallback semantics. Do not attribute all
historical offline/online differences to proxy failure based on this implementation.

Ada-KV is not included. Implementing a shared-head approximation and calling it
Ada-KV would not validate head-wise adaptive allocation. Likewise, the validator
contains no hidden CPU copy or free reload of an evicted KV block.

## 6. Reading measurements and paired results

Important JSON fields:

| Field | Interpretation |
|---|---|
| `gates` | Native/model and masked/physical fixed-schedule correctness checks |
| `checkpoint_files_sha256`, `source_sha256`, `data_sha256` | Exact model, implementation and input provenance |
| `final_cache.logical_live_kv_bytes` | Bytes needed for actually eligible K/V tokens |
| `final_cache.kv_storage_bytes` | Bytes held by the K/V tensors' unique storages |
| `final_cache.metadata_storage_bytes` | Selector and index tensor payload, not Python allocator overhead |
| `allocator_prefill_and_initial_compaction` | Full prefill plus initial compaction peak |
| `allocator_decode` | Decode window peak; includes remaining model/allocator state |
| `local_ttft_ms`, `local_tpot_ms`, `local_itl_ms` | Local single-request timing, not a network service |
| `output_tokens`, `stopped_on_eos`, `f1` | Length, stopping behavior and QA score |

Reserved allocator memory is not the same as live tensor memory. A caching
allocator can retain freed storage. Even a real tensor-byte decrease does not
establish production pool reuse or capacity for more concurrent requests. The
compaction implementation can temporarily hold old and new storage together;
inspect peaks, not just the final value. Eager attention and Python selection
make this a reference implementation, not an optimized latency target.

Strict paired comparison:

```bash
python experiments/analyze_physical_kv.py \
  --baseline "$OUT/h2o_block.physical.b0.2.r0.json" \
  --candidate "$OUT/xqp_reconstructed.physical.b0.2.r0.json" \
  --out "$OUT/paired.manual.json"
```

The analyzer rejects mismatched IDs, checkpoint hashes, input tokens, precision,
runtime, selector configuration and missing quality. It never silently uses only
the intersection of successful requests. Both cells must hash checkpoint weights.
Fewer than five dataset clusters yields descriptive differences only. An
exploratory dataset-cluster t-TOST is available for larger cohorts, but five
clusters is not a universal sample-size guarantee and is not the paper's old
14 architecture–dataset experiment. Fix the margin before looking at results.
Tail percentiles require corresponding minimum request counts; 64 requests do
not justify a confident P99.9 claim. Use separate profiling runs for root causes.

Aggregate a finished run into the `summary.json` the manuscript tables read from,
or re-check a committed one:

```bash
python experiments/summarize_physical_kv.py \
  --run-dir "$OUT" \
  --qa-meta "$QA_DATA.meta.json" \
  --query-analysis "$OUT/../query-v2/<tag>.analysis.json" \
  --query-meta "$OUT/../query-v2/<tag>.jsonl.meta.json" \
  --model <checkpoint-directory-name>            # add --check to verify, not write
```

It derives every field from the tracked per-request cells, the paired analyses and
the frozen cohort manifest; it computes no new statistic and refuses gate-only
cells, mismatched cohorts or inconsistent hashes. `--check` exits nonzero on any
numeric drift, which is how `tests/test_validation_protocol.py` guards the two
committed summaries. Report a number in the paper only if this regenerates it.

## 7. Corrected query probe: separate rerun, not a silent archive replacement

The legacy query extractor mixed pre-RoPE queries and post-RoPE cached keys. Its
first-token and final-horizon paths also require correction. The old producing
revision is unknown, so numerical archival results are not assigned a fabricated
corrected value. `run_quest_headline.py` is retired; its source remains under
`experiments/legacy/`. Run a new phase-aligned cohort:

```bash
python experiments/run_query_control_v2.py \
  --model "$MODEL" --data "$QA_DATA" --chat --device cuda:0 \
  --block-size 32 --feature-steps 32 --max-input-tokens 4096 \
  --trace /data/query-v2.llama.bs32.jsonl \
  --out /data/query-v2.llama.bs32.analysis.json
```

Choose a fresh trace path for block sizes 16 and 1. Token-granularity JSONL can
be large; pilot storage/runtime first. At least 16 independent requests are
required. Features are emitted for 32 decode steps plus 64 additional lookahead
steps supplying genuine future labels. The collector ignores EOS for that trace
protocol and records the forwarded generated tokens. Post-RoPE dot-max is still
**not** Quest's page-bound algorithm. The analysis uses train-only scaling and
request-disjoint fitting, then reports paired held-out **request-mean** AUC gains
and an exploratory request-bootstrap interval. It does not restore the old
pooled-AUC, CMI or universal query-redundancy claims automatically.

CPU re-analysis requires both the trace and its checksum-verified metadata:

```bash
python experiments/run_query_control_v2.py --analyze-only \
  --trace /data/query-v2.llama.bs32.jsonl \
  --out /data/query-v2.llama.bs32.reanalysis.json
```

## 8. What may enter the camera-ready

The current paper contains claim corrections, exact archival provenance and
explicit limitations—not invented physical-memory/latency numbers. Once the
GPU matrix completes and is independently checked, a small implementation table
may report the actual model, two policies, budgets, quality, live/peak memory and
local timing. Label it a reference implementation check. Do not assert complete
serving superiority, intrinsic query redundancy, or equivalence from a
nonsignificant difference. The full serving, offload, DMA and concurrency study
remains separate work.

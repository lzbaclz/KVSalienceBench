# Running the pinned masked loop

The simulator source is bundled in `third_party/`. Preparation checks its archive
digest, all 65 Python file hashes and its MIT license against the source manifest.
The runner repeats verification and writes provenance beside each new result.

## CPU setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[test,validation]'
python scripts/prepare_seer.py
python experiments/seer_pinned/smoke_cpu.py
KVSALIENCE_SEER_ROOT="$PWD/build/seer-5a7fbee19045" pytest -q tests/test_seer_pinned_patch.py
```

The smoke test compares patched and unpatched masked decoding on a random
small Llama. It needs no dataset or model download. The validation extra pins
Transformers 4.51.3 and includes pandas. Use a suitable CUDA PyTorch wheel for
GPU runs.

## External inputs

Supply local Llama-3.1-8B-Instruct or Qwen2.5-7B-Instruct checkpoints and the
LongBench `data/` directory. `configs/reproduction_assets.json` contains model
shard and dataset hashes; no inputs are downloaded automatically.

```bash
python scripts/check_reproduction_assets.py --model llama \
  --model-dir /data/models/Llama-3.1-8B-Instruct --longbench-dir /data/longbench/data
python scripts/check_reproduction_assets.py --model qwen \
  --model-dir /data/models/Qwen2.5-7B-Instruct
```

Matching hashes identify reference inputs. Different inputs form a new experiment.
Models, datasets, traces and generated results are excluded from public Git.

## Run a cell or sweep

```bash
python experiments/seer_pinned/run_cell.py \
  --seer-root build/seer-5a7fbee19045 \
  --model /data/models/Llama-3.1-8B-Instruct \
  --longbench-dir /data/longbench/data \
  --dataset qasper --policy xqp --dtype bfloat16 \
  --context-length 4096 --num-requests 64 --max-new-tokens 48 \
  --hbm-budget 0.20 --decision-period 8 --seed 0 \
  --out build/reproduction/qasper_xqp.json
```

The runner selects the canonical first N dataset rows, enables chat formatting
by default and requires a fresh output path. Provenance records source hashes,
runtime versions, input hashes, arguments and completion status. Use explicit
dtype: the float16 Qwen path can emit non-finite logits. The bfloat16 reference
configuration avoids that path.

The default scorer is `experiments/predictors/xqp_closed_2view_h4.json`.
The memory patch releases large prefill attention tensors after their statistics
are consumed. Tests check tokens, retained counts and oracle-miss parity.

```bash
ARCH=llama31_8b MODEL=/data/models/Llama-3.1-8B-Instruct \
DTYPE=bfloat16 LONGBENCH_DIR=/data/longbench/data \
OUT_ROOT="$PWD/build/reproduction" \
bash experiments/seer_pinned/run_sweep.sh
```

The sweep accepts `GPU`, `PY`, `SEER_ROOT`, `DATASETS`, `POLICIES` and `N` overrides.
Its default matrix covers seven datasets and five policies. Use
`experiments/analyze_expand_sensitivity.py --help` to analyze supplied result
files and original input rows. Generated reports belong in ignored directories.

Logical masking retains dense storage. See the separate
[physical KV runbook](PHYSICAL_KV_VALIDATION.md) for tensor storage measurements.

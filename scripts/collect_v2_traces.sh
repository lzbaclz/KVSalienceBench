#!/usr/bin/env bash
# Version-2 trace collection for the model families the pinned collector
# actually supports (eager Llama / Qwen2, transformers 4.51.3). Fails closed
# before loading anything if a checkpoint's model_type is unsupported, so an
# unsupported family never crashes mid-run or silently changes cache semantics.
#
#   MODELS="/public/model_zoo/Llama-3.1-8B-Instruct /public/model_zoo/Qwen2.5-7B-Instruct" \
#   TRACEDIR=/hole0/lzq/xqp_traces_v2 N_TRACES=32 WORKLOAD=mooncake bash scripts/collect_v2_traces.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
: "${MODELS:?MODELS: space-separated local checkpoint directories}"
: "${TRACEDIR:?TRACEDIR: output directory on a large volume}"
PY="${PY:-/home/lzq/miniconda3/envs/kvsalience-cr/bin/python}"
N_TRACES="${N_TRACES:-32}"
MAX_CONTEXT="${MAX_CONTEXT:-4096}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
WORKLOAD="${WORKLOAD:-mooncake}"
GPU="${GPU:-0}"
export PYTHONNOUSERSITE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$TRACEDIR"
"$PY" - "$@" $MODELS <<'PYEOF'
import json, sys
from pathlib import Path
bad = []
for m in sys.argv[1:]:
    cfg = Path(m) / "config.json"
    if not cfg.exists():
        bad.append(f"{m}: no config.json"); continue
    t = json.loads(cfg.read_text()).get("model_type")
    if t not in {"llama", "qwen2"}:
        bad.append(f"{m}: model_type={t!r} is not supported by the version-2 collector")
if bad:
    sys.exit("preflight failed:\n  " + "\n  ".join(bad))
import transformers
assert transformers.__version__ == "4.51.3", transformers.__version__
print("preflight OK:", len(sys.argv) - 1, "supported checkpoints")
PYEOF
for m in $MODELS; do
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/collect_traces_attn.py \
    --n-traces "$N_TRACES" --prompt-start 0 --prompt-end "$N_TRACES" \
    --max-context "$MAX_CONTEXT" --max-new-tokens "$MAX_NEW_TOKENS" \
    --workload "$WORKLOAD" --device cuda:0 --out-dir "$TRACEDIR" --out-suffix "$WORKLOAD.v2" --models "$m"
done

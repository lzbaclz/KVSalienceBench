#!/usr/bin/env bash
# Serial fresh-process cells. Never run this matrix concurrently on one GPU.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
: "${MODEL:?Set MODEL to a local Llama or Qwen2 checkpoint directory}"
: "${QA_DATA:?Set QA_DATA to a frozen English QA JSONL}"
OUT="${OUT:-experiments/results/physical_kv/$(date -u +%Y%m%dT%H%M%SZ)}"
REPEATS="${REPEATS:-1}"
DTYPE="${DTYPE:-bfloat16}"
MAX_INPUT="${MAX_INPUT:-4096}"
MAX_NEW="${MAX_NEW:-128}"
# CHAT=1 must match preparation. FIXED_OUTPUT=1 is timing-only (no F1).
extra=()
[[ "${CHAT:-1}" == 1 ]] && extra+=(--chat)
[[ "${FIXED_OUTPUT:-0}" == 1 ]] && extra+=(--fixed-output)
mkdir -p "$OUT"
for ((repeat=0; repeat<REPEATS; repeat++)); do
  # Rotate policy order between repeats; do not count repeats as new QA samples.
  policies=(h2o_block xqp_reconstructed)
  (( repeat % 2 == 1 )) && policies=(xqp_reconstructed h2o_block)
  python experiments/run_physical_kv.py --model "$MODEL" --data "$QA_DATA" \
    --policy full --budget 1 --backend physical --dtype "$DTYPE" \
    --max-input-tokens "$MAX_INPUT" --max-new-tokens "$MAX_NEW" \
    --hash-model-weights "${extra[@]}" --out "$OUT/full.r${repeat}.json"
  for budget in 0.2 0.3; do
    for policy in "${policies[@]}"; do
      for backend in masked physical; do
        python experiments/run_physical_kv.py --model "$MODEL" --data "$QA_DATA" \
          --policy "$policy" --budget "$budget" --backend "$backend" --dtype "$DTYPE" \
          --max-input-tokens "$MAX_INPUT" --max-new-tokens "$MAX_NEW" \
          --hash-model-weights "${extra[@]}" \
          --out "$OUT/${policy}.${backend}.b${budget}.r${repeat}.json"
      done
    done
    python experiments/analyze_physical_kv.py \
      --baseline "$OUT/h2o_block.physical.b${budget}.r${repeat}.json" \
      --candidate "$OUT/xqp_reconstructed.physical.b${budget}.r${repeat}.json" \
      --out "$OUT/paired.b${budget}.r${repeat}.json"
  done
done
printf '\nCompleted cells: %s\n' "$OUT"

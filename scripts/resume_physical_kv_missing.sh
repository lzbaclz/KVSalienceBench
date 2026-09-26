#!/usr/bin/env bash
# Continue a physical-KV matrix without re-running completed cells.
# Same serial contract as run_physical_kv_matrix.sh: one GPU, fresh processes.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
: "${MODEL:?Set MODEL to a local Llama or Qwen2 checkpoint directory}"
: "${QA_DATA:?Set QA_DATA to a frozen English QA JSONL}"
: "${OUT:?Set OUT to the existing run directory}"
REPEATS="${REPEATS:-1}"
DTYPE="${DTYPE:-bfloat16}"
MAX_INPUT="${MAX_INPUT:-4096}"
MAX_NEW="${MAX_NEW:-128}"
extra=()
[[ "${CHAT:-1}" == 1 ]] && extra+=(--chat)
[[ "${FIXED_OUTPUT:-0}" == 1 ]] && extra+=(--fixed-output)

run_if_missing() {
  local dest="$1"; shift
  if [[ -e "$dest" ]]; then
    python - "$dest" <<'PY'
import json, sys
p = sys.argv[1]
obj = json.load(open(p))
if obj.get("status") != "passed":
    raise SystemExit(f"refusing to skip incomplete {p}")
print(f"skip {p}")
PY
    return
  fi
  if [[ -e "${dest}.failed.json" ]]; then
    echo "previous failure remains: ${dest}.failed.json" >&2
    exit 1
  fi
  python experiments/run_physical_kv.py "$@" --out "$dest"
}

mkdir -p "$OUT"
for ((repeat=0; repeat<REPEATS; repeat++)); do
  policies=(h2o_block xqp_reconstructed)
  (( repeat % 2 == 1 )) && policies=(xqp_reconstructed h2o_block)
  run_if_missing "$OUT/full.r${repeat}.json" --model "$MODEL" --data "$QA_DATA" \
    --policy full --budget 1 --backend physical --dtype "$DTYPE" \
    --max-input-tokens "$MAX_INPUT" --max-new-tokens "$MAX_NEW" \
    --hash-model-weights "${extra[@]}"
  for budget in 0.2 0.3; do
    for policy in "${policies[@]}"; do
      for backend in masked physical; do
        run_if_missing "$OUT/${policy}.${backend}.b${budget}.r${repeat}.json" \
          --model "$MODEL" --data "$QA_DATA" \
          --policy "$policy" --budget "$budget" --backend "$backend" --dtype "$DTYPE" \
          --max-input-tokens "$MAX_INPUT" --max-new-tokens "$MAX_NEW" \
          --hash-model-weights "${extra[@]}"
      done
    done
    paired="$OUT/paired.b${budget}.r${repeat}.json"
    if [[ -e "$paired" ]]; then
      echo "skip $paired"
    else
      python experiments/analyze_physical_kv.py \
        --baseline "$OUT/h2o_block.physical.b${budget}.r${repeat}.json" \
        --candidate "$OUT/xqp_reconstructed.physical.b${budget}.r${repeat}.json" \
        --out "$paired"
    fi
  done
done
printf '\nCompleted cells: %s\n' "$OUT"

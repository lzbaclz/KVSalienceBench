#!/usr/bin/env bash
# Re-run the matched-retention expansion sweep with the PINNED SEER revision and an
# explicit dtype (the archived sweep ran the runner's float16 default, which corrupts
# Qwen2.5 outputs with all-NaN logit steps; see experiments/diagnose_qwen_fp16.py).
#
# One architecture per invocation, serial cells on one GPU:
#   ARCH=qwen25_7b MODEL=/public/model_zoo/Qwen2.5-7B-Instruct DTYPE=bfloat16 GPU=0 \
#   OUT_ROOT=/hole0/lzq/kvsalience_pc_response/expand_v2 bash experiments/seer_pinned/run_sweep.sh
#
# Completion is strict: a cell counts as done only if its provenance sidecar says
# status=passed with the requested number of unique request ids. Anything else is
# moved aside with a timestamp and rerun; nothing is silently reused or overwritten.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
: "${ARCH:?ARCH (output subdirectory, e.g. qwen25_7b)}"
: "${MODEL:?MODEL (local HF checkpoint directory)}"
: "${DTYPE:?DTYPE (float16|bfloat16|float32) - nothing defaults to float16}"
: "${OUT_ROOT:?OUT_ROOT}"
GPU="${GPU:-0}"
SEER_ROOT="${SEER_ROOT:-/hole0/lzq/kvsalience_pc_response/SEER-old_feat-5a7fbee19045}"
PY="${PY:-/home/lzq/miniconda3/envs/kvsalience-cr/bin/python}"
DATASETS="${DATASETS:-narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique triviaqa}"
POLICIES="${POLICIES:-full h2o xqp adakv pyramidkv}"
N="${N:-64}"
export PYTHONNOUSERSITE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT_ROOT/$ARCH"
MANIFEST="$OUT_ROOT/$ARCH/SWEEP_MANIFEST.txt"
echo "[$(date -u +%FT%TZ)] start arch=$ARCH model=$MODEL dtype=$DTYPE gpu=$GPU seer=$SEER_ROOT" >> "$MANIFEST"

is_done () {  # $1 = provenance sidecar
  [ -f "$1" ] || return 1
  "$PY" - "$1" "$N" <<'PYEOF' >/dev/null 2>&1
import json, sys
p = json.load(open(sys.argv[1])); n = int(sys.argv[2])
sys.exit(0 if p.get("status") == "passed" and p.get("n_results") == n and p.get("n_unique_ids") == n else 1)
PYEOF
}

failed=0
for ds in $DATASETS; do
  for pol in $POLICIES; do
    out="$OUT_ROOT/$ARCH/${ds}_${pol}.json"
    prov="$out.provenance.json"
    if is_done "$prov"; then
      echo "[$(date -u +%FT%TZ)] SKIP  $ARCH/$ds/$pol (complete)" | tee -a "$MANIFEST"
      continue
    fi
    for stale in "$out" "$prov"; do
      if [ -e "$stale" ]; then
        mv "$stale" "$stale.incomplete.$(date -u +%Y%m%dT%H%M%SZ)"
        echo "[$(date -u +%FT%TZ)] MOVED stale $stale" | tee -a "$MANIFEST"
      fi
    done
    echo "[$(date -u +%FT%TZ)] RUN   $ARCH/$ds/$pol" | tee -a "$MANIFEST"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" experiments/seer_pinned/run_cell.py \
      --seer-root "$SEER_ROOT" --model "$MODEL" --policy "$pol" --dataset "$ds" \
      --dtype "$DTYPE" --num-requests "$N" --out "$out" > "$out.log" 2>&1
    rc=$?
    echo "[$(date -u +%FT%TZ)] END   $ARCH/$ds/$pol rc=$rc" | tee -a "$MANIFEST"
    [ "$rc" -eq 0 ] || failed=1
  done
done
echo "[$(date -u +%FT%TZ)] finished arch=$ARCH failed=$failed" | tee -a "$MANIFEST"
exit $failed

# Attention traces (regenerable; not committed)

The `*.jsonl` trace files are **large** and **regenerable**, so they are
git-ignored. The small analysis JSON the paper figures/tables are built from
lives in `../results/` and *is* tracked.

Two corpora exist:
- **Headline (committed-results) corpus:** 4 architectures × 32 mooncake prompts
  here under `experiments/traces/` — ~2–2.7 GB each, **43.9M rows total** (Llama
  10.53M / Qwen2.5 9.31M / Qwen3 11.97M / Mistral 12.45M). This is what every
  number in `paper_icdm/` is built on.
- **Expanded corpus (optional):** larger multi-workload collection written to a
  big volume via `TRACEDIR=/public/xqp_traces` (mooncake + sharegpt + LongBench
  tasks), filenames `<model>.<workload>.jsonl`. Used for the external-validity /
  tighter-CI extensions. **Never** write the expanded corpus under `/` (the home
  volume is only ~47 GB; see `scripts/run_collect_expanded.sh`).

## What they are
One row per `(request, layer, decode-step, KV-block)`, schema:
`request_id, layer, step, block_idx, f_within, f_cross, f_query, f_pos,
y_h1, y_h4, y_h16, y_h64` (4 views + top-r saliency labels at horizons
{1,4,16,64}, r=0.10, 32-token blocks).

## How to regenerate (A100, env `csp-llm`)
```bash
# 4 architectures × 32 mooncake long-context prompts (≤4096 ctx), 128 decode steps
bash scripts/legacy/run_collect_4models_v1.sh   # LEGACY record of the v1 launch (exits; see scripts/collect_v2_traces.sh)
```
Produces **43.9M rows total** (Llama 10.53M / Qwen2.5 9.31M / Qwen3 11.97M /
Mistral 12.45M). Models load from `/public/model_zoo/`; fp16. Qwen2.5's last
layer emits NaN attention under fp16 and is dropped at load (332,536 rows,
~3.6% of its rows; aggregated into the pooled provenance).

## Then reproduce all results
```bash
python experiments/run_icdm_full.py   --traces experiments/traces --out experiments/results/icdm_full.json
python experiments/run_icdm_extra.py  --traces experiments/traces --out experiments/results/icdm_extra.json
python experiments/run_heavy_hitter.py --traces experiments/traces --out experiments/results/heavy_hitter.json
python experiments/bench_wcet_gpu.py  --out experiments/results/wcet_gpu.json
python experiments/generate_figures.py --results experiments/results --out paper_icdm/figures
```

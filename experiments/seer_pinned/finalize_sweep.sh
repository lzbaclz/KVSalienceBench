#!/usr/bin/env bash
# Mechanical post-sweep steps for the pinned bfloat16 rerun (PC audit P0-1):
#   1. verify + copy all 70 cells into experiments/results/expand_v2 (refuses partial sweeps)
#   2. re-derive the paired statistics with both scorers
#   3. render the Exp#8 table rows into the manuscript between the markers
#   4. redraw the forest plot from the rerun (official LongBench scorer)
# Prose that interprets the numbers is written by hand afterwards and guarded by
# tests/test_paper_numbers.py.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="${PY:-/home/lzq/miniconda3/envs/kvsalience-cr/bin/python}"
SRC="${SRC:-/hole0/lzq/kvsalience_pc_response/expand_v2}"
export PYTHONNOUSERSITE=1
"$PY" experiments/seer_pinned/collect_results.py --src "$SRC" --dest experiments/results/expand_v2
"$PY" experiments/analyze_expand_sensitivity.py --root experiments/results/expand_v2 \
  --out experiments/results/tost/expand_v2_sensitivity.json
"$PY" experiments/render_perlayer_table.py > /tmp/perlayer_rows.txt
"$PY" - <<'PYEOF'
import pathlib, re
rows = pathlib.Path('/tmp/perlayer_rows.txt').read_text().strip()
p = pathlib.Path('paper_icdm/sections/transfer_gap.tex'); s = p.read_text()
s2 = re.sub(r"%%PERLAYER-ROWS-BEGIN%%.*?%%PERLAYER-ROWS-END%%",
            lambda m: "%%PERLAYER-ROWS-BEGIN%%\n" + rows + "\n%%PERLAYER-ROWS-END%%", s, flags=re.S)
assert s2 != s or rows in s
p.write_text(s2)
print(rows)
PYEOF
"$PY" experiments/fig_cell_forest.py --analysis experiments/results/tost/expand_v2_sensitivity.json \
  --contrast "xqp_vs_h2o::pooled::f1_longbench_all_refs" --out paper_icdm/figures/fig_cell_forest.pdf
echo "finalize_sweep: done (now write the Exp#8 prose and run the tests)"

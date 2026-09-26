#!/usr/bin/env bash
# Export the editable architecture figure with the draw.io desktop renderer.
set -euo pipefail
cd "$(dirname "$0")/.."
drawio_bin="${DRAWIO_BIN:-drawio}"
drawio_args=(--no-sandbox --disable-gpu --export --format pdf --crop --border 2 --embed-diagram
  --output paper_icdm/figures/evaluation_paths.pdf paper_icdm/figures/evaluation_paths.drawio)
if [[ -z "${DISPLAY:-}" ]]; then
  xvfb-run -a "$drawio_bin" "${drawio_args[@]}"
else
  "$drawio_bin" "${drawio_args[@]}"
fi
pdffonts paper_icdm/figures/evaluation_paths.pdf

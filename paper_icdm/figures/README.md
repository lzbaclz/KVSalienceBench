# Camera-ready figures

The manuscript includes exactly these three vector PDFs:

| Figure | Editable source | Renderer |
|---|---|---|
| Evaluation paths | `evaluation_paths.drawio` | draw.io Desktop 31.4.5 PDF export |
| Relevance and reliability, v2 | `../../experiments/fig_redund_calib.py` | Python / Matplotlib |
| Per-cell F1 differences, v2 | `../../experiments/fig_cell_forest.py` | Python / Matplotlib |

From the repository root:

```bash
bash scripts/export_diagrams.sh
python experiments/fig_redund_calib.py
python experiments/fig_cell_forest.py
```

The draw.io source contains editable text, rectangles and connectors. Its PDF is
exported by the installed draw.io application with crop and embedded-source options;
no Matplotlib or LaTeX diagram renderer is used. Open the `.drawio` source in draw.io
to edit it. The PDF uses embedded Liberation Serif; the Python PDFs use embedded
STIXGeneral/TrueType outlines. Paper body and captions retain IEEEtran defaults.

The English presentation reuses these PDFs (rasterized at 288 dpi for PPTX), with
the PDF-source hashes recorded in `../output/asset_manifest.md`. Old schematic
sources and the retired Chinese presentation are internal history and are excluded
from the clean public export. Other unused historical plot PDFs are also excluded.

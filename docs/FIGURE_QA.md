# Figure and layout audit — 2026-09-26

The final manuscript uses the unmodified IEEEtran conference page, body font and
spacing settings. It is 8 US-letter pages including references, with a 232-word
single-paragraph abstract. All eight rendered pages were inspected at 108 dpi;
figure details were also viewed enlarged. Tables, equations, captions, author
blocks and references are legible, without observed text collisions or clipping.

## Figure source and export checks

| Figure | Source / renderer | Font and artwork checks |
|---|---|---|
| Evaluation paths, Fig. 1 | Editable `paper_icdm/figures/evaluation_paths.drawio`; draw.io Desktop 31.4.5 CLI export | Embedded Liberation Serif; ordinary labels 9.20 pt in the source PDF; vector paths and text, no raster image |
| Relevance / reliability, Fig. 2 | `experiments/fig_redund_calib.py`, frozen `icdm_v2.json` | Embedded STIXGeneral, ordinary labels at least 7 pt; the mathematical subscript is smaller; vector PDF |
| Per-cell forest, Fig. 3 | `experiments/fig_cell_forest.py`, frozen `expand_v2_sensitivity.json` | Embedded STIXGeneral, ordinary labels at least 7 pt; vector PDF |

`python scripts/check_figure_assets.py` proves that the draw.io PDF's embedded
editable XML equals the saved `.drawio` source, the exported labels are present,
all three manuscript figure paths are accounted for, fonts are embedded, and text
bounds lie inside each PDF. It also rejects Type 3 fonts and raster artwork.
The shared Python crop padding was increased to 0.045 inches after bounding-box
checks exposed a forest-axis descender and a rotated math-label boundary too close
to the old crop. Both figures were rebuilt and the corrected bounds pass.

The 8-page paper passes `scripts/check_camera_ready.py`: no overfull boxes,
unresolved references/citations, font substitutions, encryption or Type 3 fonts.
All paper fonts are embedded. A text-span collision scan found only intentional
mathematical accents/subscripts and an accented author name in the bibliography;
these were checked in the render. The scan is a candidate finder, not a proof of
readability. No body font, margin or line-spacing reduction was used to recover
the page limit; repeated introduction text was shortened instead.

## English presentation and response summary

`paper_icdm/output/final_presentation_en.pptx` contains 13 English 16:9 slides,
with speaker notes on every slide. It was exported by LibreOffice to
`final_presentation_en.pdf`, and all 13 rendered slides were visually inspected.
The calibration, forest and architecture slides were checked enlarged. Editable
slide text uses Liberation Sans, with embedded regular/bold fonts in the PDF.
There are no out-of-slide shapes or substantial text-span collisions in the
rendered PDF. The three paper PDFs are reused at 288 dpi in the deck; their hashes
are in `paper_icdm/output/asset_manifest.md`.

The D2AI change summary was rendered as `paper_icdm/response.pdf`: one page,
embedded Times-compatible fonts and no overfull boxes. Its four points match the
clean Markdown summary. The old Chinese deck and Python-drawn schematics are
retained only as excluded internal history.

These are local layout and typography checks, not an IEEE PDF eXpress certificate.
The final manuscript SHA256 is recorded in `VALIDATION_STATUS.md`.

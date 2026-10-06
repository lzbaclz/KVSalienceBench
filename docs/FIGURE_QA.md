# Figure and layout audit — 2026-09-26

The final manuscript uses the unmodified IEEEtran conference page, body font and
spacing settings. It is 8 US-letter pages including references, with a 230-word
single-paragraph abstract (2026-10-06 candidate). All eight rendered pages were inspected at 108 dpi;
figure details were also viewed enlarged. Tables, equations, captions, author
blocks and references are legible, without observed text collisions or clipping.

The [IEEE conference graphics guide](https://conferences.ieeeauthorcenter.ieee.org/write-your-paper/improve-your-graphics/)
recommends approximately 9–10 pt lettering at final size. All ordinary labels in
the three paper figures are now 9.00–9.46 pt at the 3.5-inch column width. The
relevance panel uses horizontal bars to keep full category names and numerical
annotations apart at that size.

## Figure source and export checks

| Figure | Source / renderer | Font and artwork checks |
|---|---|---|
| Evaluation paths, Fig. 1 | Editable `paper_icdm/figures/evaluation_paths.drawio`; draw.io Desktop 31.4.5 CLI export | Embedded Liberation Serif; ordinary labels 9.46 pt at final column width; vector paths and text, no raster image |
| Offset sweep / reliability, Fig. 2 | `experiments/fig_redund_calib.py`, frozen `icdm_v2_decomposition.json` and `icdm_v2.json` | Embedded STIXGeneral, ordinary labels approximately 9 pt at final column width; the mathematical subscript is smaller; vector PDF. Since 2026-10-05 the legend of panel (b) sits in that panel's own column and panel (a) shares no colour or marker with it |
| Per-cell forest, Fig. 3 | `experiments/fig_cell_forest.py`, frozen `expand_v2_sensitivity.json` | Embedded STIXGeneral, ordinary labels approximately 9 pt at final column width; vector PDF |

`python scripts/check_figure_assets.py` proves that the draw.io PDF's embedded
editable XML equals the saved `.drawio` source, the exported labels are present,
all three manuscript figure paths are accounted for, fonts are embedded, and text
bounds lie inside each PDF. It also rejects Type 3 fonts and raster artwork.
The shared Python crop padding was increased to 0.05 inches after bounding-box
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

**2026-10-06.** No figure changed. The text gained a numbered equation (§II-C), Table II
moved its long caption into a note under the table (with a dagger on one cell), Table I's
caption and one row changed, and Table IV's first row gained an interval. All eight pages
of the final build were rendered at 110 dpi and inspected: the equation keeps its number
on its line, the table notes and captions fit their columns, there are no overfull boxes,
and the two columns that an intermediate build left with stretched paragraph gaps (page 1
right, page 4 left) are filled. The last column of page 8 ends 58 pt above the margin.

These are local layout and typography checks, not an IEEE PDF eXpress certificate.
The final manuscript SHA256 is recorded in `VALIDATION_STATUS.md`.

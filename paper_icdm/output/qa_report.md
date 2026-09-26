# English presentation QA — 2026-09-26

- Deliverables: `final_presentation_en.pptx` and its LibreOffice-exported PDF.
- 13 English 16:9 slides; 13 sets of speaker notes; no out-of-bounds shapes.
- All 13 PDF pages visually inspected; enlarged inspection of diagram and charts.
- Editable text uses Liberation Sans; PDF fonts are embedded regular/bold TrueType.
- Main numbers are loaded from the frozen v2 JSON by `../build_deck.py`.
- Mean F1 difference +0.0017; 14-cell TOST p=0.0007; 14-cell 90% t interval
  [−0.0063, +0.0097]; seven-dataset p=0.0023, with the inferential units labeled.
- V2 two-view / LightGBM AUC 0.890 / 0.896. Calibration 0.003 is labeled as
  isotonic GBDT, not the retired deck's MLP claim.
- The runtime reconstruction, query-probe scope, physical-parity failure and
  model/context limitations are explicit. No production serving speedup claim.
- Architecture figure comes from the saved draw.io source via the draw.io PDF.
  Other figures come from Python; exact source PDF hashes are in `asset_manifest.md`.
- The retired Chinese deck is preserved under `docs/internal/presentation_cn/`
  in the private working repository and is excluded from public releases.

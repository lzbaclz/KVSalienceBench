# English presentation QA — 2026-10-05

- Deliverables: `final_presentation_en.pptx` and its LibreOffice-exported PDF
  (TrueType fonts embedded, 13 pages of 960 x 540 pt).
- 13 English 16:9 slides; 13 sets of speaker notes; no out-of-bounds shapes
  (asserted by `../build_deck.py`).
- Not rebuilt on 2026-10-06: the round-2 review revision changes no figure, and the slide
  numbers are loaded from result records that revision did not change. The slide text and
  notes in `../build_deck.py` were read against the revised manuscript; none is contradicted.
- Rebuilt on 2026-10-05 after Figure 2 was regenerated (its legend now sits in panel
  (b)'s own column; panel (a) shares no colour or marker with it). Only slide 6 changed;
  it was visually inspected after export, and `asset_manifest.md` carries the new
  source-PDF hash. No slide text or speaker note changed.
- Rebuilt on 2026-10-04 against the revised manuscript. Slides 5, 6, 8, 9, 10 and 13
  changed and were visually inspected after export, and slides 2, 7, 11 and 12 were
  re-inspected:
  - 5: pooled AUC, within-decision AUC and top-decile recall from the exact decomposition
    record (`icdm_v2_decomposition.json`); the callout no longer implies a causal story.
  - 6: the figure's panel (a) is now the cross-layer offset sweep; notes state the 83% / 65%
    result and keep the calibration point.
  - 7: notes name the on-policy arm survivor-conditioned (candidates retained now and at
    t+h) and say its 0.989 is not evidence that retention makes prediction easier.
  - 8, 9, 10: the primary analysis is the seven-dataset TOST (p=0.0023, 90% t interval
    [-0.0064, +0.0098]); the 14 cells (p=0.0007) are the sensitivity analysis; the margin is
    a practical tolerance, not pre-registered; 448 source prompts x 2 models = 896
    model-prompt evaluations. The forest plot shows both pooled estimates.
  - 13: the default protocol scores pooled and per-decision metrics together.
- Editable text uses Liberation Sans; PDF fonts are embedded regular/bold TrueType.
- Main numbers are loaded from the frozen v2 JSON by `../build_deck.py`.
- V2 two-view / LightGBM AUC 0.890 / 0.896; per-decision top-decile recall of the
  within-layer signal 0.659 against 0.613 / 0.615. Calibration 0.003 is labeled as
  isotonic LightGBM.
- The runtime reconstruction, query-probe scope, physical-parity failure and
  model/context limitations are explicit. No production serving speedup claim.
- Architecture figure comes from the saved draw.io source via the draw.io PDF.
  Other figures come from Python; exact source PDF hashes are in `asset_manifest.md`.
- The retired Chinese deck is preserved under `docs/internal/presentation_cn/`
  in the private working repository and is excluded from public releases.

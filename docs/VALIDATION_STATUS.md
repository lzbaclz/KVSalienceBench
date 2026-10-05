# Validation status — 2026-10-05

## Final camera-ready candidate

`paper_icdm/main.pdf` is **8 IEEE two-column US-letter pages including references**,
with a **232-word abstract**, complete named authors, and Jiawei Guo's confirmed
email `jwguo@buaa.edu.cn`. No funding acknowledgment was requested. The standard
IEEEtran body font, margins and line spacing were retained. The final PDF SHA256 is:

```
f6e4033c2a45ca79c5825cdf4acbf3604fbac425a5709bb0f48c6ae3130d5634
```

This identifies the locally validated upload candidate, not a CPS upload receipt. It
**supersedes the 2026-10-04 candidate** (`3d96b3504b95...`) and the 2026-09-29 one
(`b7a80e8b62d3...`), both described below: upload this PDF, not those, and run PDF
eXpress on this exact file. If the authors subsequently add a copyright line or DOI,
rerun the sequence below and replace this hash with the exact PDF submitted to the
workshop. The last column of page 8 ends about six text lines above the bottom margin,
which is the room a first-page copyright notice would need.

## 2026-10-05 panel revision

A five-reviewer simulated panel read the 2026-10-04 candidate. It found no wrong number
and no unsupported conclusion; every item was a wording, labelling or presentation
defect, and each was checked against the code and the result JSON before it was applied.
No experiment was run and no result JSON was rewritten.

- **Wording that contradicted the body.** The abstract ended "not a serving or
  policy-equivalence claim" after claiming mean-F1 equivalence of two policies (now
  "per-task equivalence"). §V-A said the reversal "belongs to the operating point, not the
  scorers" one paragraph before tracing it to a scorer term (now "its size depends on the
  operating point"). §VIII said the corrected corpus shows the LightGBM ordering "at a
  narrower margin" when its deficit is larger (-0.0057 against -0.0010; now "wider"). The
  conclusion's "narrows at 20% retention" read as a narrower learned-scorer advantage
  (now "less sharply"), and its "-0.003 AUC" was the Llama value alone for a
  reference-backend measurement (now "reference-backend feature reconstruction within
  0.003 AUC"; Qwen is -0.002).
- **The parity tolerance is stated in full.** §VI printed "0.25 versus 0.15", against
  which §IX's "3.5--5.9x the preset tolerance" could not be recomputed. The gate is
  `atol + rtol*|logit|` with (0.15, 0.02) in bfloat16 (`xqp/physical_validation.py`,
  `_tolerances` and `_compare_logits`), so §VI now reads "0.15 plus 2% of the logit" and
  the multiples are consistent as printed.
- **Comparators are defined where they are used.** Table I said "H2O:" twice (now
  "H2O-style:"), §VI named the code identifier `h2o_block`, and the two allocation
  comparators of Table IV had no definition. From the pinned simulator source
  (`seer/policy/perlayer_baselines.py` in the bundled revision): both only move a fixed
  total budget across layers --- Ada-KV-style by attention concentration, PyramidKV-style
  down a linear pyramid --- and select within a layer exactly as the H2O-style policy
  does. §VII says so, and its disclaimer now also names PyramidKV.
- **Text with no result behind it was removed.** The interaction-information display of
  §III-B and the sentence about its signs (no interaction estimate is printed), a sentence
  about an omitted historical comparison, the list of comparators in §IV-A that Table II
  does not report, and the horizon set of §II-A (the paper reports h=4; horizons 1, 16 and
  64 are recorded). Exp#9 is headed "Archived", and "we retract the context-growing win"
  (no such claim exists in this version) became "we claim no long-context gain".
- **Table III lost its local inter-token-latency column.** The masked rows were about 20%
  faster than full cache, the text never interpreted that, and the paper makes no latency
  claim. The values stay in `experiments/results/physical_kv/*/summary.json`
  (`tpot_ms_median`) and stay pinned by `tests/test_paper_numbers.py`, which now also
  asserts that the manuscript prints no latency column.
- **Three limitations were added to §IX,** each raised independently by two reviewers:
  both learned scorers minimize a pooled loss, so a per-decision ranking objective is
  untested; where one budget is shared across layers or requests, cross-decision ordering
  is itself the operating point; and the offline candidate set includes the sink and
  recent blocks that the loop keeps unconditionally.
- **Readers outside the area.** "KV" is glossed as cached attention keys and values in the
  abstract and the first sentence; TOST and EMA are expanded at first use; "TPOT" and
  "HBM" became "per-token latency" and "GPU-memory release". The abstract states the
  4K-token cap and calls the closed-loop scorer "runtime-reconstructed". §IV-B names the
  bridge's checkpoint as the archived one and reports "per-request" (not "held-out")
  AUC; §V says LightGBM is fit on all four views; the Fig. 2 caption gives the ECE
  binning (10 equal-width bins, `xqp/dm_metrics.py`).
- **Positioning.** §X relates the pooled-versus-within-decision split to user-weighted
  (grouped) AUC in click-through prediction and cites Zhou et al., KDD 2018
  (DOI 10.1145/3219819.3219823). The record was checked against Crossref and the metric's
  definition against §6.3 of the paper. The bibliography now has 27 entries.
- **Figure 2 was regenerated.** Its legend spanned the figure, so it sat over panel (a)
  and ran into that panel's y-label, although it belongs to panel (b); panel (a) also
  reused the legend's red circles and blue squares for other quantities. The legend is now
  in panel (b)'s own column and panel (a) is drawn in neutral ink with its own markers.
  Slide 6 of the deck was rebuilt from the new PDF.
- **Reference [13]** (Peng, Long and Ding, TPAMI 2005) is printed with the colon again
  ("...mutual information: Criteria of..."), as on the article and in PubMed
  (PMID 16119262); Crossref's deposited metadata has no colon.
- **Page budget.** The removals above paid for the additions; about ten sentences were
  tightened (none drops a number or a caveat) so that section X's heading stays at the
  foot of its column instead of leaving a stretched one.

## 2026-10-04 external-review revision

An independent review of the camera-ready (a reviewer-style report plus a 25-reference
audit) was checked claim by claim against the code, the result JSON and the primary
sources before anything was changed; `docs/EXTERNAL_REVIEW_2026-10-04_RESPONSE.md` in
the private working copy maps every item to its action. What changed, and what it cost:

- **The mechanism sentence was replaced by a measurement.** The earlier text said the
  binary cross-layer feature was informative "about which decisions hold salient blocks".
  `experiments/analyze_decision_decomposition.py` (CPU, every held-out row, exact pair
  counts, source-prompt bootstrap) shows why that cannot be the mechanism: every decision
  holds exactly ceil(0.1 n) positives, and same-decision pairs are only 0.0016% of all
  (positive, negative) pairs, so pooled AUC is a cross-decision statistic. The
  logistic scorer's cross-layer term raises it by +0.017 [0.015, 0.019] but lowers
  pair-weighted within-decision AUC by 0.0025 [0.0023, 0.0028] and top-decile recall by
  0.046; one eighth of the fitted offset already gives 83% of the first and 65% of the
  second. At the top-decile boundary it swaps in 22% of the selected blocks, 20% of which
  are positive against 41% of the blocks they displace. The record reproduces Table II's
  pooled AUC and per-decision recall to four decimals as its own gate (`gate` in the
  JSON). Exact random tie-breaking moves no per-decision recall by more than 1.7e-5.
- **A factual error in the previous camera-ready was found and fixed.** The text quoted
  an offset w_cross/w_within of 0.096 for the refit two-view scorer; that is the
  *archived frozen checkpoint's* ratio (34.76 and 3.35). The refit scorer in Table II has
  w_within=97.7, w_cross=2.46, ratio 0.025. The sentence now cites the refit values and
  the corpus mean x_wl of 0.015.
- **LightGBM capacity.** `experiments/analyze_gbdt_capacity.py` fits six configurations
  (depth 3-8, 150-600 trees, balanced or not) at 120K and 1.92M rows: LightGBM's pooled
  AUC lead over the compact fit stays between 0.0005 and 0.0076, its per-decision recall
  between 0.615 and 0.622, and none reaches the raw EMA's 0.659 or its within-decision
  AUC. "Nothing below is an artifact of the cap" was removed.
- **Bridge, margin, physical parity.** The bridge's on-policy arm is now named for what
  the code computes (`experiments/run_feature_bridge.py` keeps blocks present now *and*
  at t+h: survivor-conditioned). The seven-dataset TOST (p=0.0023) is the primary analysis
  and the 14 cells (p=0.0007) a sensitivity analysis, because the two architectures answer
  the same questions; "896 paired prompts" became 448 source prompts on two models. The
  +-0.02 margin appears in the archived analysis before the pinned rerun but as one of a
  swept set, so it is described as a practical tolerance, not pre-registered. The abstract
  no longer headlines exact parity without the extended-replay failure. Logged retained
  blocks per decode step (25.02 / 25.02 / 25.02 / 25.01, median 26, range 7-26) support
  "matched retention" in counts.
- **The default benchmark entry now implements the paper's protocol.**
  `benchmark/protocol.py` is protocol 2.0 (decision identifiers kept, source-disjoint
  default split, pooled and per-decision metrics, source-prompt bootstrap, completeness
  check, ECE only for declared probabilistic scorers, a prediction-table schema with a
  synthetic example and a refit reference baseline); protocol 1.0 moved, marked legacy,
  to `benchmark/legacy_v1/`. Validated against the 7.6M-row analysis to 1e-9.
- **References.** [PyramidKV] authors follow the COLM 2025 list (Yucheng Li added, Baobao
  Chang removed) and [RULER] gains Yang Zhang, both checked against arXiv and the venue
  list; volume numbers were added for the NeurIPS and MLSys entries, Gama et al. gained
  its article number, and AttentionPredictor was added and cited (26 references).
- **Page budget.** The new analysis had to be paid for, so the historical section was cut
  to conclusions with every moved number kept in `docs/HISTORICAL_DIAGNOSTICS.md` (still
  checked against the JSON), three secondary rows left Table II, Table III was transposed
  and Exp#9 became a paragraph. Section numbers cited elsewhere are unchanged.
- **Not done, deliberately.** A scorer-only masked-loop ablation (needs GPU runs; the
  review marks it optional), a window-target horizon sweep, a third model, 32K contexts
  and any serving measurement. The paper scopes every claim to what was measured.

No existing result JSON was rewritten and no GPU experiment was run. Two new CPU-only
records were added: `experiments/results/icdm_v2_decomposition.json` (with two CSV views)
and `experiments/results/icdm_v2_gbdt_capacity.json`.

## 2026-09-29 audit revision

An external camera-ready read produced three tiers of findings; all were applied.
The substantive ones, and what they cost:

- **An unpublished draft section was withdrawn.** `paper_icdm/sections/gated.tex`
  described an unverified regime-gated selective cascade, carried internal decision
  comments, and was never `\input` into `main.tex` --- yet it shipped in the public
  export and the release tarball, because the exporter took every `.tex` under
  `paper_icdm/sections`. The file is deleted and
  `scripts/export_public_artifact.py` now derives its section list from the
  `\input` lines of `main.tex`, so an un-input draft cannot leak again. The
  release must be re-cut; the previous tarball and the public repository contain
  the draft.
- **A new CPU sensitivity analysis was run** (no GPU, no existing result JSON
  rewritten). `experiments/analyze_train_size_sensitivity.py` refits the two-view
  logistic scorer and the headline LightGBM on the same request split, seed and
  held-out rows at 120K, 0.48M, 1.92M, 7.68M and all 22.8M held-in rows. Its 120K
  point reproduces the published Table II values exactly (two-view 0.8904,
  LightGBM 0.8961, paired deficit -0.0057 [-0.0068,-0.0046], per-decision recalls
  0.6130/0.6148, raw within-layer 0.6586), which is the gate on the rerun itself.
  At every held-in row LightGBM reaches 0.8966 and the deficit widens only to
  -0.0062, so the compact-model comparison is not an artifact of the 120K training
  cap. Output: `experiments/results/icdm_v2_train_size.json`; pinned by
  `tests/test_paper_numbers.py::test_training_size_sensitivity`.
- **Both Python figures and the draw.io diagram were regenerated.** Figure 2's
  legend now says LightGBM, matching the text and Table II, and its relevance axis
  carries the unit (nats; `xqp/info_theory.py` uses natural logarithms). Figure 1
  gained coloured per-row labels (Offline / Closed-loop / Physical) and its caption
  explains the colours. Figure 3's x-label follows the manuscript's single name for
  the policy ("reconstructed"). Legend spacing in Figure 2 was tightened so the
  longer labels keep the tight bounding box at one column and the ~9 pt label floor
  still passes.
- **Structure.** Version-1 results moved out of the v2 line into a single
  "Historical diagnostics" section (§VIII), which states their provenance once
  instead of per paragraph; the GuardKV construction folded into it as Exp#4. The
  scattered "not a serving claim" disclaimers were merged into one Scope paragraph
  at the head of §VI. Implementation detail that left the text is indexed by
  `docs/ARTIFACT_POINTERS.md`, which names the file behind every "in the artifact"
  sentence.
- **Two numbers were deliberately withdrawn** rather than reprinted: the Exp#2
  AUPRC pair 0.777/0.803 (computed under the version-1 row-order tie rule, not
  recomputable tie-aware) and the per-budget capacity floors. Both remain in the
  result JSON, and `tests/test_paper_numbers.py` now asserts the JSON still holds
  them *and* that the LaTeX no longer prints them.
- **`xqp/conformal.py` gained a SCOPE paragraph.** Its docstring claimed a
  distribution-free guarantee the manuscript explicitly retracts; the released code
  now carries the same exchangeability and per-request caveats as the paper.

No result JSON was rewritten, no GPU experiment or latency measurement was run, and
every figure change is a regeneration from committed source.

## Executed checks

- **278 tests passed, no skips** in the Python 3.11 pinned runtime with CUDA hidden
  (231 after the 2026-09-29 revision, plus 27 decision-evaluation tests checked against
  brute-force pair enumeration, 16 benchmark-protocol tests and 4 new paper-number pins).
- The clean public export (re-cut 2026-10-05) independently passed **278 tests, no
  skips**; its figure assets pass; and its LaTeX sources rebuilt to an 8-page PDF whose
  text is identical to the shipped PDF's. It contains no raw JSONL input files, pretrained model
  payloads, internal review directories or the 2026-09-29 draft sections.
- `check_camera_ready.py` was re-run after the final clean rebuild of 2026-10-05 and
  passes: 8 pages, 232-word abstract, **no overfull boxes**, no undefined references,
  27 BibTeX entries and no BibTeX warnings. (An overfull box in Table II's new lower block was caught by this check
  and fixed; inspect logs with a tool that does not honour `.gitignore`.)
- All **65 bundled simulator Python source hashes** match all **70** v2 cell
  provenance records. The original MIT notice is retained.
- `check_camera_ready.py` passes: authors, abstract length, page limit, citations,
  references, font embedding and encryption. No overfull boxes or Type 3 fonts.
- `check_figure_assets.py` passes: exact draw.io source embedded in its exported
  PDF, complete figure inventory, vector artwork, embedded fonts and unclipped text.
- All eight rendered paper pages were visually inspected after the 2026-10-05
  revision. The inspected render is of the final rebuild, the file whose hash is given
  above, including approximately 9 pt figure labels.
- The English presentation is **13 slides**, with notes on all slides, exported
  through LibreOffice to PDF. All slides were visually inspected; no out-of-bounds
  shapes or substantial text-span collisions were found. It was last rebuilt on
  2026-10-05 against the regenerated Figure 2 --- `asset_manifest.md` records the
  source-PDF hashes --- and slide 6 was re-inspected. The clean reviewer-change
  summary is a separately rendered **one-page PDF**.
- `git diff --check` passes. **No existing experimental result JSON changed and no
  GPU experiment or latency measurement was run.** The 2026-09-29 revision adds one
  new CPU-only record, `experiments/results/icdm_v2_train_size.json`, and rewrites
  none. See `FIGURE_QA.md`, `ARTIFACT_POINTERS.md` and `RELEASE_AUDIT.md` for scope
  and evidence.

The source-linked number checks were strengthened after direct JSON inspection
found a stale sentence about the v2 seven-dataset supplementary ±0.01 test. It
actually has p=0.047, while the 14-cell test has p=0.045; both remain supplementary
because that tighter margin was not pre-specified. The printed 90% t interval is
now explicitly attributed to the 14-cell analysis. Result JSON was not rewritten.

## Final validation sequence

The first public CI run exposed an export omission beyond pytest's coverage:
the small `experiments/predictors/` JSON coefficient files were absent, so the
standalone count-mapping audit could not load its frozen scorer. The exporter now
includes those files and requires the two-view checkpoint explicitly. This is a
packaging correction; source coefficients and result JSON are unchanged. Release
validation also exercises the CPU example, count audit and raw-record summary
commands, not only the pytest suite.

Use the pinned Python environment documented in `REPRODUCING_MASKED_LOOP.md`.
These checks are CPU-only; the tiny random-model tests do not reproduce 8B-model
experiments or certify production serving performance.

```bash
python scripts/prepare_seer.py
export KVSALIENCE_SEER_ROOT="$PWD/build/seer-5a7fbee19045"
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q
(cd paper_icdm && latexmk -g -pdf -interaction=nonstopmode -halt-on-error main.tex)
#   check latexmk's exit status: after deleting main.aux/main.bbl by hand it stops at
#   bibtex (exit 12) and leaves the PREVIOUS main.pdf in place; `latexmk -C` first, or
#   also delete main.fdb_latexmk, for a clean rebuild
python scripts/check_camera_ready.py
python scripts/check_figure_assets.py   # imports pymupdf: pip install -e '.[paper]'
sha256sum paper_icdm/main.pdf
git diff --check
```

For a figure edit, first rebuild the relevant asset using
`paper_icdm/figures/README.md`, then repeat the sequence. Public GitHub Actions
repeats the source/CPU checks, chart generation and paper build; its run result
is independently visible in the repository's Actions tab. Local success is not
itself proof of a remote CI pass.

## Author-account boundary

The author supplied the confirmed ICDMW CPS route and named Ziqing Li as the
in-person presenter. The specific camera-ready deadline is **October 4, 2026,
23:59 PDT (October 5, 14:59 in China)**. Venue PDF eXpress certification, copyright
agreement, registration payment and final CPS upload are author-account actions,
not certified by this repository. Zenodo account work and DOI registration were
explicitly deferred to the authors; no DOI is invented. The private working copy
contains the complete publication handoff with account links and contacts.

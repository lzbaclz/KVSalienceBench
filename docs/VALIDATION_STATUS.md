# Validation status — 2026-09-26

## Final camera-ready candidate

`paper_icdm/main.pdf` is **8 IEEE two-column US-letter pages including references**,
with a **232-word abstract**, complete named authors, and Jiawei Guo's confirmed
email `jwguo@buaa.edu.cn`. No funding acknowledgment was requested. The standard
IEEEtran body font, margins and line spacing were retained. The final PDF SHA256 is:

```
45e646ca9b1be46b802e531818f55c3d66bb1466c865d3fb176f6252041e025b
```

This identifies the locally validated upload candidate, not a CPS upload receipt.
If the authors subsequently add a copyright line or DOI, rerun the sequence below
and replace this hash with the exact PDF submitted to the workshop.

## Executed checks

- **230 tests passed, no skips** in the Python 3.11 pinned runtime with CUDA hidden.
- The clean public export independently passed **230 tests, no skips**, then
  regenerated both Python figures and rebuilt an 8-page paper. It contains no raw
  JSONL input files, pretrained model payloads or internal review directories.
- All **65 bundled simulator Python source hashes** match all **70** v2 cell
  provenance records. The original MIT notice is retained.
- `check_camera_ready.py` passes: authors, abstract length, page limit, citations,
  references, font embedding and encryption. No overfull boxes or Type 3 fonts.
- `check_figure_assets.py` passes: exact draw.io source embedded in its exported
  PDF, complete figure inventory, vector artwork, embedded fonts and unclipped text.
- All eight rendered paper pages were visually inspected. Rebuilding after the
  final test run produced identical page pixels to the inspected render.
- The English presentation is **13 slides**, with notes on all slides, exported
  through LibreOffice to PDF. All slides were visually inspected; no out-of-bounds
  shapes or substantial text-span collisions were found. The clean reviewer-change
  summary is a separately rendered **one-page PDF**.
- `git diff --check` passes. **No experimental result JSON changed and no new GPU
  experiment or latency measurement was run.** The NVIDIA log change strips only
  trailing whitespace. See `FIGURE_QA.md` and `RELEASE_AUDIT.md` for scope and evidence.

The source-linked number checks were strengthened after direct JSON inspection
found a stale sentence about the v2 seven-dataset supplementary ±0.01 test. It
actually has p=0.047, while the 14-cell test has p=0.045; both remain supplementary
because that tighter margin was not pre-specified. The printed 90% t interval is
now explicitly attributed to the 14-cell analysis. Result JSON was not rewritten.

## Final validation sequence

Use the pinned Python environment documented in `REPRODUCING_MASKED_LOOP.md`.
These checks are CPU-only; the tiny random-model tests do not reproduce 8B-model
experiments or certify production serving performance.

```bash
python scripts/prepare_seer.py
export KVSALIENCE_SEER_ROOT="$PWD/build/seer-5a7fbee19045"
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q
(cd paper_icdm && latexmk -g -pdf -interaction=nonstopmode -halt-on-error main.tex)
python scripts/check_camera_ready.py
python scripts/check_figure_assets.py
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

# Clean artifact release audit — 2026-10-06

## Withdrawal notice for the 2026-09-26 release

The 2026-09-26 export and its tarball contain `paper_icdm/sections/gated.tex`, an
unpublished draft section on a regime-gated selective cascade that `main.tex`
never `\input`, with internal decision comments still in it. It was shipped
because the exporter selected every `.tex` under `paper_icdm/sections`. That
release is superseded: the file is deleted, the exporter now derives its section
list from the `\input` lines of `main.tex`, and the tarball and public repository
must be re-cut. The negative result the draft described is already the manuscript's
published position (§III-B, §IX), and the reusable cascade code stays released in
`xqp/gated_predictor.py`; only the draft prose and its internal annotations are
withdrawn. See `docs/ARTIFACT_POINTERS.md`.

**Re-cut of 2026-10-04.** The tarball `KVSalienceBench-d2ai-camera-ready-r3.tar.gz` and
its export directory supersede both the withdrawn 2026-09-26 release and the
2026-09-29 `-r2` tarball: the paper, the English slides and the docs cite the new PDF
(`VALIDATION_STATUS.md`), and the release adds the decision-level benchmark protocol 2.0
(`benchmark/protocol.py`, a synthetic example, a reference baseline; protocol 1.0 under
`benchmark/legacy_v1/`), the exact decision-level evaluator (`xqp/decision_eval.py`), two
new CPU-only records (`experiments/results/icdm_v2_decomposition.json` with two CSV views
and `icdm_v2_gbdt_capacity.json`) with their drivers, and `docs/HISTORICAL_DIAGNOSTICS.md`.
The exporter still takes only the sections `main.tex` inputs and keeps the
internal review-response notes out of the public tree.

**Re-cut of 2026-10-05.** The tarball `KVSalienceBench-d2ai-camera-ready-r5.tar.gz` and
its export directory supersede `-r3`, and also `-r4`, a first cut of the same day whose
paper lacks only the corresponding-author mark on the title block. The manuscript changed in wording, one table column
and one figure (`VALIDATION_STATUS.md`, "2026-10-05 panel revision"), so the paper PDF,
its LaTeX sources, Figure 2 and its script, slide 6 of the deck, one test and these
documents differ; no result record and no analysis code changed. This is the release the
manuscript describes: it contains `docs/HISTORICAL_DIAGNOSTICS.md` and the protocol-2.0
benchmark entry that §I, §VIII, §IX and §XI name, which the 2026-09-26 public
repository did not.

**Re-cut of 2026-10-06.** The tarball `KVSalienceBench-d2ai-camera-ready-r6.tar.gz` and
its export directory supersede `-r5`. The manuscript changed in substance, not only in
wording (`VALIDATION_STATUS.md`, "2026-10-06 round-2 review revision"): one Table II cell
and its note, the paragraph comparing Table III with Table IV, an equation, the
served-oracle definition and the reference list (28 entries). The release adds four
CPU-only records with their drivers (`icdm_v2_tie_sensitivity.json`,
`icdm_v2_decomposition_source.json` with two CSV views,
`tost/expand_v2_full_cache_gap.json`, `physical_kv/scorer_check.json`),
`docs/FROZEN_CONFIG.md`, a candidate manifest for the benchmark example and
`benchmark/splits/paper_v2_splits.json`; no existing record changed. The evaluator's
output key `complete_decisions` is renamed `label_count_consistent`, labels other than
0/1 are refused, and a candidate manifest can be verified: a caller that read the old key
has to follow the rename. Validation of this export: 298 CPU tests pass in the extracted
tree with no skips; its figure assets pass; its LaTeX sources rebuild to an 8-page PDF
whose text is identical to the shipped PDF's; and the steps of the public CI workflow
were replayed in it locally (pinned-simulator smoke run, byte-identical count-mapping
audit, oracle-budget and physical-summary re-derivations, figure regeneration, paper
build, both checkers).

**Re-cut of 2026-10-06 (second, `-r7`).** The tarball
`KVSalienceBench-d2ai-camera-ready-r7.tar.gz` supersedes `-r6`, cut the same morning.
The only manuscript change is the title block: the authors are listed row by row with
the IEEE template's ordinals, so the PDF's content stream carries the registered author
order (`VALIDATION_STATUS.md`, "2026-10-06 template check and title block"). Pages 2-8 of
the paper are pixel-identical to `-r6`. The release adds `tests/test_title_block.py` and
an author-order and page-size check in `scripts/check_camera_ready.py`; no result record
and no analysis code changed. Validation of this export: 303 CPU tests pass in the
extracted tree with no skips; its figure assets pass; and its LaTeX sources rebuild to an
8-page PDF whose text is identical to the shipped PDF's.

Target repository: <https://github.com/lzbaclz/KVSalienceBench>.
The authors authorized a new public artifact repository. The old working repository
and its history stay private. This export is a new Git history, not a visibility
change to the working repository.

## Included material

Code, numerical experiment JSON (and two small CSV views of one record),
model-generated outputs and scoring references, small predictor coefficients,
provenance hashes, CPU tests, a synthetic prediction-table example, reproduction runbooks,
the pinned simulator source bundle, paper source/PDF, a one-page change summary,
Python chart sources, editable draw.io source/PDF and the English PPTX/PDF.
Included result JSON is copied byte for byte: no result record is regenerated,
redacted or silently rewritten. `MANIFEST.sha256` enumerates the delivered files;
`EXPORT_PROVENANCE.json` identifies the export's source revision and state.

## Excluded material

- Private Git history, `ICDM_PIVOT.md`, `ITERATIONS.md`, internal planning reports.
- `reviews/`, simulated reviewer/PC/mentor response files, `docs/internal/`,
  the legacy MLSys paper and unrelated exploratory audit/RCRG projects.
- Raw JSONL input datasets, the original QA prompt files, raw attention traces,
  checkpoints/tensor files and model-weight archives.
- The retired Chinese presentation and obsolete/unused figure PDFs.
- Local build environments, logs, caches, temporary files and credentials.

The exporter filters Git-listed files rather than copying the workspace recursively,
refuses a nonempty destination, rejects symlinks and hashes every copied file.
The private export script is `scripts/export_public_artifact.py`; exclusions are
recorded outside the public tree. Retaining experiment predictions/reference
answers for rescoring does not make the source input corpus part of this release.

## License and provenance

The repository code is Apache-2.0. The SEER bundle retains its upstream MIT notice
and contains only source/package resources and its license. All 65 Python files
match all 70 pinned v2 provenance records. The bundle SHA256 is
`213c83255daf0fa4698156b34cbd41d847217b18e71a7d3fa1c190d841906dff`.
LongBench attribution and the upstream MIT notice are in `third_party/`; source
model/dataset terms remain separate. No pretrained LLM weights or original input
corpora are bundled. Absolute paths in immutable provenance are historical records,
not dependencies on access to the authors' machines.

## Validation and later DOI

See `VALIDATION_STATUS.md` for the full test suite, PDF hash and export validation,
and `FIGURE_QA.md` for rendered layout checks. Public CI repeats CPU/source checks,
figure generation and the paper build. It does not execute new GPU experiments.

Zenodo registration was explicitly deferred to the author. `.zenodo.json` provides
reviewable deposit metadata; `CITATION.cff` points to the repository. Once an actual
DOI is published, update the citation and manuscript and revalidate the PDF. There
is no invented DOI or claim of completed Zenodo deposit.

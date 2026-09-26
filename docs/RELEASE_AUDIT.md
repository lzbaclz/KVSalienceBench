# Clean artifact release audit — 2026-09-26

Target repository: <https://github.com/lzbaclz/KVSalienceBench>.
The authors authorized a new public artifact repository. The old working repository
and its history stay private. This export is a new Git history, not a visibility
change to the working repository.

## Included material

Code, numerical experiment JSON, model-generated outputs and scoring references,
small predictor coefficients, provenance hashes, CPU tests, reproduction runbooks,
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

# Data and code availability — D2AI 2026 camera-ready

## Release contents

The clean artifact repository is <https://github.com/lzbaclz/KVSalienceBench>.
It contains the 8-page paper and LaTeX, editable draw.io source and exported PDF,
Python chart sources, the English presentation, immutable experimental result
JSON, predictor coefficients, tests, analysis code, version-2 collection code and
the physical-KV reference validator. The pinned v2 SEER runtime is bundled with
its MIT license. Its 65 Python source files are verified against 70 cell records.
The original working repository's history and simulated review materials are
excluded. See `../docs/RELEASE_AUDIT.md` for the inventory and validation.

The main offline evidence is the 30,445,332-row v2 corpus from two models; the
43,931,784-row, four-model v1 archive and workload extensions are historical.
Source datasets, original QA prompt files, model weights and raw attention traces
are not included. Experiment logs retain predictions and scoring references for
reanalysis; these are not a redistribution of the input corpora. Obtain models
and datasets separately under their upstream terms. The optional raw v2 trace
package can be requested from Jianxi Chen at `chenjx@hust.edu.cn`.

Zenodo account work and DOI registration are deferred to the authors. No DOI or
completed Zenodo archive is claimed. `CITATION.cff` supplies the repository URL.

## Distinct reproducibility paths

1. **Inspect archived evidence:** use the unchanged JSON under `experiments/results/`
   and its analysis scripts. This does not regenerate the GPU executions.
2. **Collect corrected traces:** version-2 `xqp/attn_trace_extract.py`, pinned HF
   runtime, local model weights and explicit new output paths. V2 is not numerically
   interchangeable with v1.
3. **Run prospective physical validation:** follow `../docs/PHYSICAL_KV_VALIDATION.md`.
   It is self-contained for the declared Llama/Qwen2 reference policies and does not
   need the historical SEER checkout. No production engine or original Ada-KV is implied.
4. **Reproduce the masked-loop results:** the archived cells came from an unpinned
   SEER revision in float16 and cannot be replayed exactly. The version-2 rerun
   (`experiments/results/expand_v2/`) records, per cell, the SEER git SHA
   (`old_feat` branch, `5a7fbee19045eb0bea92d84c999eafcc2d6de686`), the sha256 of
   every SEER source file, the exact runner arguments, dtype, runtime versions and
   the LongBench file hash; `experiments/seer_pinned/run_cell.py` reproduces a cell
   from the bundled export of that SHA. `python scripts/prepare_seer.py` verifies
   all 65 Python files against all 70 provenance records, retains the MIT notice,
   and creates the local export without upstream repository access. See
   [the complete runbook](../docs/REPRODUCING_MASKED_LOOP.md) for an offline CPU
   example, dependencies, model IDs/checksums, data preparation and a complete
   64-request GPU cell. The runbook now pins retrievable model snapshots whose
   eight shard hashes match the recorded weights; it distinguishes those
   verified retrieval locations from unrecorded historical producer revisions.

The final-review CPU sensitivity analysis is in
`experiments/results/mentor_statistics.json`, with per-request recalls and
cluster AUC sufficient statistics for independent interval replay. The optional
9 GiB raw v2 trace package is not in git; it can be requested through the same
author contact. The simulator source, CPU example and stored-record reanalysis
do not depend on that optional package.

## Later Zenodo archive

Archive the reviewed release bundle using `.zenodo.json`, then add the actual
published DOI to `CITATION.cff` and the paper's Availability paragraph. Rebuild and
validate the PDF after that change, and record its new SHA256 before CPS upload.

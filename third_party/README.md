# Pinned runtime dependency

`seer-5a7fbee19045.tar.gz` contains the `seer/` Python package and its MIT license
from revision `5a7fbee19045eb0bea92d84c999eafcc2d6de686`.
Copyright (c) 2026 SEER Authors. This dependency keeps its MIT license separately
from the project's Apache-2.0 license; the full notice is inside the archive.
The bundle contains source code without repository history, models or datasets.

`seer-source-manifest.json` pins the archive digest and all 65 Python file hashes.
These hashes were identical in all 70 provenance records of the pinned execution.
The compact manifest allows verification without distributing generated results.

```bash
python scripts/prepare_seer.py --check-only
python scripts/prepare_seer.py
KVSALIENCE_SEER_ROOT="$PWD/build/seer-5a7fbee19045" pytest -q tests/test_seer_pinned_patch.py
```

Extraction checks archive paths, the complete source inventory and the license
before writing. The explicit runtime memory patch is in
`experiments/seer_pinned/run_cell.py`; upstream source bytes remain unchanged.
See [the runbook](../docs/REPRODUCING_MASKED_LOOP.md).

The LongBench-derived scoring functions retain the separate `LongBench-LICENSE`
and `LongBench-NOTICE.md` files.

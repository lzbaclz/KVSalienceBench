# Pinned simulator source

`seer-5a7fbee19045.tar.gz` contains only the `seer/` package and `LICENSE`
from SEER commit `5a7fbee19045eb0bea92d84c999eafcc2d6de686`.
Upstream: `https://github.com/lzbaclz/SEER-private` (access restricted).
Copyright (c) 2026 SEER Authors; MIT license, with the full upstream notice in
the archive. This dependency retains its MIT license, separately from this
repository's Apache-2.0 license. No private repository history, research notes,
model weights, credentials, or datasets are included.

The bundle makes the exact source available to anyone who receives this
artifact; no access to the upstream repository is needed. It does not imply
that the enclosing artifact has already been published.

From the repository root:

```bash
python scripts/prepare_seer.py
KVSALIENCE_SEER_ROOT="$PWD/build/seer-5a7fbee19045" pytest -q tests/test_seer_pinned_patch.py
```

Extraction checks every Python file against all 70 tracked experiment
provenance records before writing anything. `--check-only` verifies without
extracting. The memory patch remains explicit in
`experiments/seer_pinned/run_cell.py`; the upstream source is unchanged.
See `docs/REPRODUCING_MASKED_LOOP.md` for dependencies and a full example.

Maintainers can reproduce the archive from a local checkout without modifying it:

```bash
python scripts/prepare_seer.py --source-repo /path/to/SEER --check-only
```

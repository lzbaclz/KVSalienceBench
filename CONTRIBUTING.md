# Contributing

This public repository contains executable code, tests, runtime configuration,
small benchmark examples and manifests, dependency licenses, and instructions
needed to use or develop the code.

Keep manuscripts, presentations, editorial/review notes, previews, generated
experiment results, raw datasets and model checkpoints outside the repository.
Local experiment outputs may use the ignored `experiments/results/`,
`experiments/traces/`, `build/` and `assets/` directories.

Before submitting changes:

```bash
git add <code-files>
python scripts/check_public_tree.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 pytest -q
```

The content check examines Git's tracked/staged inventory, including files added
with `git add -f`. CI runs it for every push and pull request. A new file category
requires an explicit update to its allowlist and must serve the code.

`third_party/seer-source-manifest.json` pins the simulator revision, archive hash
and every Python source hash. Keep its upstream license intact when updating the
dependency. Benchmark coefficients and split manifests are inputs to the code;
changes to them require corresponding protocol checks.

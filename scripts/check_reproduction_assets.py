#!/usr/bin/env python3
"""Check externally downloaded model shards and LongBench files against v2 hashes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"llama": "Llama-3.1-8B-Instruct", "qwen": "Qwen2.5-7B-Instruct"}


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=MODELS)
    p.add_argument("--model-dir", type=Path)
    p.add_argument("--longbench-dir", type=Path)
    a = p.parse_args(argv)
    if bool(a.model) != bool(a.model_dir) or not (a.model or a.longbench_dir):
        p.error("provide --model and --model-dir together and/or --longbench-dir")
    checks = {}
    if a.model:
        prov = json.loads((ROOT / "experiments/results/icdm_v2.json").read_text())["provenance"]
        checks.update({a.model_dir / name: sha for name, sha in prov[MODELS[a.model]]["model_weight_sha256"].items()})
    if a.longbench_dir:
        for path in (ROOT / "experiments/results/expand_v2").glob("*/*.provenance.json"):
            record = json.loads(path.read_text())
            source = a.longbench_dir / Path(record["longbench_path"]).name
            if source in checks and checks[source] != record["longbench_sha256"]:
                raise ValueError(f"inconsistent recorded source hash: {source.name}")
            checks[source] = record["longbench_sha256"]
    for path, expected in sorted(checks.items()):
        if not path.is_file() or digest(path) != expected:
            raise SystemExit(f"missing or mismatched file: {path}; do not label this an exact rerun")
        print(f"PASS {path.name}", flush=True)
    print(f"PASS: {len(checks)} externally supplied files match archived checksums")


if __name__ == "__main__":
    main()

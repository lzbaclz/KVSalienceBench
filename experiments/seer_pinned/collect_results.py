#!/usr/bin/env python3
"""Verify a completed pinned-SEER sweep and copy it into the repository.

Every expected cell must have a provenance sidecar with status=passed, the
requested number of unique request ids, an output hash that still matches the
file, one SEER git SHA and one dtype across the whole sweep. Only then are the
cell JSONs, sidecars and manifests copied (never overwriting) into --dest and a
sweep-level manifest written. Partial sweeps are refused: nothing is copied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ARCHS = ("llama31_8b", "qwen25_7b")
DATASETS = ("narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "triviaqa")
POLICIES = ("full", "h2o", "xqp", "adakv", "pyramidkv")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dest", required=True, type=Path)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--archs", default=",".join(ARCHS))
    args = ap.parse_args(argv)
    archs = args.archs.split(",")
    problems, cells = [], {}
    for arch in archs:
        for ds in DATASETS:
            for pol in POLICIES:
                out = args.src / arch / f"{ds}_{pol}.json"
                prov = Path(str(out) + ".provenance.json")
                if not out.exists() or not prov.exists():
                    problems.append(f"missing {arch}/{ds}/{pol}")
                    continue
                p = json.loads(prov.read_text())
                if p.get("status") != "passed" or p.get("n_results") != args.n or p.get("n_unique_ids") != args.n:
                    problems.append(f"incomplete {arch}/{ds}/{pol}: {p.get('status')} n={p.get('n_results')}")
                    continue
                if sha256_file(out) != p.get("output_sha256"):
                    problems.append(f"hash mismatch {arch}/{ds}/{pol}")
                    continue
                rows = json.loads(out.read_text())["results"]
                if len(rows) != args.n or len({r["id"] for r in rows}) != args.n:
                    problems.append(f"row count {arch}/{ds}/{pol}")
                    continue
                cells[(arch, ds, pol)] = (out, prov, p)
    shas = {p["seer_git_sha"] for _, _, p in cells.values()}
    dtypes = {p["wrapper_args"]["dtype"] for _, _, p in cells.values()}
    patches = {p["memory_patch"] for _, _, p in cells.values()}
    if len(shas) != 1 or len(dtypes) != 1 or len(patches) != 1:
        problems.append(f"sweep is not homogeneous: shas={shas} dtypes={dtypes} patches={patches}")
    if problems:
        print("REFUSING to collect:\n  " + "\n  ".join(problems))
        return 1
    args.dest.mkdir(parents=True, exist_ok=True)
    manifest = {"seer_git_sha": shas.pop(), "dtype": dtypes.pop(), "memory_patch": patches.pop(),
                "n_cells": len(cells), "n_requests_per_cell": args.n, "cells": {}}
    for (arch, ds, pol), (out, prov, p) in sorted(cells.items()):
        d = args.dest / arch
        d.mkdir(exist_ok=True)
        for src in (out, prov):
            target = d / src.name
            if target.exists():
                if sha256_file(target) != sha256_file(src):
                    print(f"REFUSING: {target} exists with different content")
                    return 1
                continue
            shutil.copy2(src, target)
        manifest["cells"][f"{arch}/{ds}_{pol}"] = {"output_sha256": p["output_sha256"], "wall_s": p["wall_s"],
                                                   "transformers": p["runtime"]["transformers"],
                                                   "torch": p["runtime"]["torch"]}
        for extra in ("SWEEP_MANIFEST.txt",):
            src_extra = args.src / arch / extra
            if src_extra.exists():
                shutil.copy2(src_extra, d / extra)
    (args.dest / "SWEEP.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"collected {len(cells)} cells into {args.dest} (SEER {manifest['seer_git_sha'][:12]}, {manifest['dtype']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

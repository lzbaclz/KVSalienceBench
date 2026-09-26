#!/usr/bin/env python3
"""Verify and unpack the MIT-licensed simulator used by the pinned v2 rerun.

The bundled source is checked against every committed v2 provenance record,
not merely against a self-reported git revision. No model or dataset is bundled.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
REVISION = "5a7fbee19045eb0bea92d84c999eafcc2d6de686"
BUNDLE = ROOT / "third_party" / f"seer-{REVISION[:12]}.tar.gz"
DEFAULT_EXPORT = ROOT / "build" / f"seer-{REVISION[:12]}"


def expected_sources():
    records = sorted((ROOT / "experiments/results/expand_v2").glob("*/*.provenance.json"))
    if len(records) != 70:
        raise ValueError(f"expected all 70 v2 provenance records, got {len(records)}")
    expected = None
    for path in records:
        record = json.loads(path.read_text())
        if record["seer_git_sha"] != REVISION:
            raise ValueError(f"unexpected simulator revision in {path}")
        if expected is None:
            expected = record["seer_source_sha256"]
        elif expected != record["seer_source_sha256"]:
            raise ValueError(f"inconsistent simulator hashes in {path}")
    return expected


def checked_contents(blob):
    contents = {}
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as archive:
        for member in archive.getmembers():
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or not path.parts:
                raise ValueError(f"unsafe archive path: {member.name}")
            if member.isdir():
                continue
            if not member.isfile() or (path.parts[0] != "seer" and str(path) != "LICENSE"):
                raise ValueError(f"unexpected archive member: {member.name}")
            if str(path) in contents:
                raise ValueError(f"duplicate member: {path}")
            contents[str(path)] = archive.extractfile(member).read()
    expected = expected_sources()
    actual = {p: hashlib.sha256(data).hexdigest() for p, data in contents.items() if p.endswith(".py")}
    if actual != expected:
        raise ValueError("bundled Python sources differ from the recorded experiment")
    license_text = contents.get("LICENSE", b"")
    if b"MIT License" not in license_text or b"Copyright (c) 2026 SEER Authors" not in license_text:
        raise ValueError("missing upstream MIT copyright and permission notice")
    return contents


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-repo", type=Path, help="maintainer only: create the bundle from this local git object database")
    p.add_argument("--dest", type=Path, default=DEFAULT_EXPORT)
    p.add_argument("--check-only", action="store_true")
    args = p.parse_args(argv)
    if args.source_repo:
        blob = subprocess.check_output(["git", "-C", str(args.source_repo), "archive", "--format=tar.gz",
                                        REVISION, "seer", "LICENSE"])
    else:
        blob = BUNDLE.read_bytes()
    contents = checked_contents(blob)
    if args.source_repo:
        if BUNDLE.exists() and BUNDLE.read_bytes() != blob:
            raise SystemExit(f"refusing to replace a different bundle: {BUNDLE}")
        BUNDLE.parent.mkdir(parents=True, exist_ok=True)
        BUNDLE.write_bytes(blob)
    if not args.check_only:
        dest = args.dest.resolve()
        if dest.exists():
            files = {str(f.relative_to(dest)): f.read_bytes() for f in dest.rglob("*")
                     if f.is_file() and "__pycache__" not in f.parts and f.name != "PINNED_GIT_SHA.txt"}
            if files != contents or (dest / "PINNED_GIT_SHA.txt").read_text().strip() != REVISION:
                raise SystemExit(f"existing export does not match; choose a fresh --dest: {dest}")
        else:
            dest.mkdir(parents=True)
            for name, data in contents.items():
                target = dest / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            (dest / "PINNED_GIT_SHA.txt").write_text(REVISION + "\n")
        print(f"Export: {dest}")
    print(f"PASS: {len(expected_sources())} Python source hashes match all 70 rerun cells; MIT notice included")
    print(f"Bundle SHA256: {hashlib.sha256(blob).hexdigest()}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Reject non-code artifacts in the staged/tracked public Git tree."""
from pathlib import Path, PurePosixPath
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TOP = {'.gitignore', 'README.md', 'CONTRIBUTING.md', 'LICENSE', 'CITATION.cff',
       'pyproject.toml', 'environment.yml'}
DOCS = {'PHYSICAL_KV_VALIDATION.md', 'REPRODUCING_MASKED_LOOP.md'}
THIRD_PARTY = {'README.md', 'LongBench-LICENSE', 'LongBench-NOTICE.md',
               'seer-5a7fbee19045.tar.gz', 'seer-source-manifest.json'}


def allowed(name):
    p = PurePosixPath(name)
    if name in TOP:
        return True
    if len(p.parts) < 2:
        return False
    top = p.parts[0]
    if top == '.github':
        return p.parts == ('.github', 'workflows', 'validate.yml')
    if top == 'docs':
        return len(p.parts) == 2 and p.name in DOCS
    if top == 'third_party':
        return len(p.parts) == 2 and p.name in THIRD_PARTY
    if top in {'xqp', 'tests'}:
        return p.suffix == '.py'
    if top == 'scripts':
        return len(p.parts) == 2 and p.suffix in {'.py', '.sh'}
    if top == 'configs':
        return len(p.parts) == 2 and p.suffix in {'.yaml', '.yml', '.json'}
    if top == 'benchmark':
        return p.suffix in {'.py', '.md', '.json', '.csv'} and p.parts[1] not in {'results', 'outputs'}
    if top == 'experiments':
        if len(p.parts) == 2:
            return p.suffix in {'.py', '.sh'}
        if p.parts[1] == 'predictors':
            return len(p.parts) == 3 and p.suffix == '.json'
        return p.parts[1] in {'seer_pinned', 'legacy'} and p.suffix in {'.py', '.sh'}
    return False


def main():
    entries = subprocess.check_output(['git', 'ls-files', '--stage', '-z'], cwd=ROOT).split(b'\0')
    errors = []
    count = 0
    for entry in filter(None, entries):
        metadata, raw_name = entry.split(b'\t', 1)
        mode, _, stage = metadata.decode().split()
        name = raw_name.decode('utf-8')
        count += 1
        if mode not in {'100644', '100755'} or stage != '0':
            errors.append(f'{name}: unsupported Git mode or unresolved merge')
        elif not allowed(name):
            errors.append(f'{name}: outside the public code allowlist')
    if not count:
        errors.append('no tracked/staged files found')
    if errors:
        print('\n'.join(errors), file=sys.stderr)
        return 1
    print(f'PASS: {count} tracked/staged files satisfy the public code policy')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

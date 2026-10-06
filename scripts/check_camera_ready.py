#!/usr/bin/env python3
"""Fail on missing references, overflowing boxes, anonymous authors or >8 pages.

Run after latexmk. Requires pdftotext/pdfinfo (poppler-utils); this is a
structural guard, NOT a substitute for rendering and visually inspecting PDF.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import re
import subprocess


def inspect_log(text):
    failures = []
    for pattern in [r'Overfull \\[hv]box', r'LaTeX Warning: (?:Reference|Citation).*undefined',
                    r'There were undefined references', r'^! ', r'Fatal error occurred',
                    r'LaTeX Font Warning:.*not available']:
        if re.search(pattern, text, flags=re.M):
            failures.append(pattern)
    return failures


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--paper-dir', default='paper_icdm')
    p.add_argument('--max-pages', type=int, default=8)
    args=p.parse_args(argv); root=Path(args.paper_dir)
    failures=inspect_log((root/'main.log').read_text(errors='replace'))
    source=(root/'main.tex').read_text()
    abstract=re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', source, re.S)
    if abstract is None:
        raise SystemExit('missing abstract')
    abstract_text=re.sub(r'\\[a-zA-Z]+(?:\{([^{}]*)\})?', r'\1', abstract[1])
    abstract_words=len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", abstract_text))
    if abstract_words > 250:
        failures.append(f'abstract over 250 words: {abstract_words}')
    info=subprocess.check_output(['pdfinfo', str(root/'main.pdf')], text=True)
    match=re.search(r'^Pages:\s+(\d+)', info, flags=re.M)
    if not match or not 1 <= int(match[1]) <= args.max_pages:
        failures.append('page limit')
    if not re.search(r'^Encrypted:\s+no\b', info, flags=re.M):
        failures.append('PDF encryption')
    text=subprocess.check_output(['pdftotext', str(root/'main.pdf'), '-'], text=True)
    for token in ('??', 'withheld for', 'Anonymous Author'):
        if token in text: failures.append(f'unresolved/anonymous text: {token}')
    authors=('Ziqing Li', 'Jiawei Guo', 'Kaicheng Tang', 'Jianxi Chen')   # registered order
    for name in authors:
        if name not in text: failures.append(f'missing author: {name}')
    # IEEE conference template: names run left to right, then down to the next row, "not
    # in columns"; that is the sequence indexing services read. A 2x2 block stacked by
    # columns looks right and reads wrong, so check the CONTENT-STREAM order of page 1.
    raw=subprocess.check_output(['pdftotext', '-raw', '-f', '1', '-l', '1', str(root/'main.pdf'), '-'], text=True)
    found=[raw.find(name) for name in authors]
    if -1 in found or found != sorted(found):
        failures.append('author order in the PDF content stream differs from the registered order '+', '.join(authors))
    if not re.search(r'^Page size:\s+612 x 792 pts \(letter\)', info, flags=re.M):
        failures.append('page size is not US Letter')
    # IEEE PDF eXpress rejects Type 3 fonts (bitmap glyphs from matplotlib defaults).
    fonts=subprocess.check_output(['pdffonts', str(root/'main.pdf')], text=True)
    type3=[ln.split()[0] for ln in fonts.splitlines()[2:] if ln.split()[1:3]==['Type','3']]
    if type3: failures.append(f'Type 3 fonts embedded: {type3}')
    unembedded=[ln.split()[0] for ln in fonts.splitlines()[2:] if ln.rsplit(None, 5)[1] != 'yes']
    if unembedded: failures.append(f'fonts not embedded: {unembedded}')
    if failures: raise SystemExit('PDF checks failed: '+', '.join(failures))
    print(f'PASS: {match[1]} pages <= {args.max_pages}; abstract {abstract_words} words; authors and their order, US Letter, log, references, encryption and embedded fonts checked. Visual inspection still required.')


if __name__=='__main__': main()

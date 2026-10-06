"""The title block follows the IEEE conference template, and the author order is unambiguous.

D2AI and ICDM 2026 ask for the IEEE two-column conference format. Its template says of the
author block: "Author names should be listed starting from left to right and then moving
down to the next line. This is the author sequence that will be used in future citations and
by indexing services. Names should not be listed in columns nor group by affiliation."

From 2026-09-26 to 2026-10-06 the manuscript drew its 2x2 title block by stacking two authors
in each IEEEtran column, without ordinals: it looked right and its content stream read
Li, Tang, Guo, Chen. These tests keep the source row-major, the ordinals in place, the class
file unmodified and every record of the author list in one order.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper_icdm"
MAIN = (PAPER / "main.tex").read_text()
AUTHORS = ["Ziqing Li", "Jiawei Guo", "Kaicheng Tang", "Jianxi Chen"]      # registered order
ORDINALS = ["1\\textsuperscript{st}", "2\\textsuperscript{nd}", "3\\textsuperscript{rd}", "4\\textsuperscript{th}"]
# IEEEtran.cls V1.8b (2015/08/26) as distributed on CTAN and in TeX Live, line endings normalized
IEEETRAN_V18B_LF_SHA256 = "da751920a317ed318b7b5cd7fa585a6cc7d28502d457856382e9be24b10a3bd7"


def author_block() -> str:
    start = MAIN.index("\\author{")
    return MAIN[start:MAIN.index("\\pdfinfo{", start)]


def test_document_class_is_the_unmodified_ieee_conference_class():
    assert "\\documentclass[conference]{IEEEtran}" in MAIN
    cls = (PAPER / "IEEEtran.cls").read_bytes().replace(b"\r\n", b"\n")
    assert b"\\ProvidesClass{IEEEtran}[2015/08/26 V1.8b by Michael Shell]" in cls
    assert hashlib.sha256(cls).hexdigest() == IEEETRAN_V18B_LF_SHA256, \
        "paper_icdm/IEEEtran.cls is not the published V1.8b class"
    # nothing in the preamble resizes the page, the margins or the fonts
    preamble = MAIN[:MAIN.index("\\begin{document}")]
    for forbidden in ("geometry", "fullpage", "savetrees", "\\setlength", "\\addtolength",
                      "\\baselinestretch", "\\linespread", "\\fontsize", "\\enlargethispage"):
        assert forbidden not in preamble, forbidden
    assert not re.search(r"\\(textwidth|textheight|columnsep|topmargin|oddsidemargin|voffset|hoffset)\s*=", preamble)
    assert "\\bibliographystyle{IEEEtran}" in MAIN and "\\begin{IEEEkeywords}" in MAIN


def test_authors_are_listed_row_by_row_in_the_registered_order():
    block = author_block()
    names = re.findall(r"\\IEEEauthorblockN\{\\authorbox\{(.*?)\}\}\s*\n\\IEEEauthorblockA", block, flags=re.S)
    assert len(names) == 4, names
    for name, ordinal, entry in zip(AUTHORS, ORDINALS, names):
        assert entry.startswith(f"{ordinal} {name}"), entry        # template ordinals state the order
    # one \and-separated block per author: no two author blocks stacked in one column
    pieces = re.split(r"\\and(?:\[[^\]]*\])?", block)
    assert len(pieces) == 4 and all(p.count("\\IEEEauthorblockN") == 1 for p in pieces)
    assert all(p.count("\\IEEEauthorblockA") == 1 for p in pieces)
    # the row break comes after the second author, through \and's own optional argument
    assert block.count("\\and[\\hfill\\mbox{}\\par\\mbox{}\\hfill]") == 1
    assert block.index("Jiawei Guo") < block.index("\\and[") < block.index("Kaicheng Tang")
    # the corresponding author is marked on the name and in the first-page footnote
    assert "Jianxi Chen$^{\\ast}$\\thanks{$^{\\ast}$Corresponding author: Jianxi Chen (chenjx@hust.edu.cn).}" in block
    assert "\\IEEEoverridecommandlockouts" in MAIN


def test_affiliations_follow_the_template_lines():
    block = author_block()
    want = [("Huazhong University of Science and Technology", "Wuhan, China", "d202381502@hust.edu.cn"),
            ("Beihang University", "Beijing, China", "jwguo@buaa.edu.cn"),
            ("The Chinese University of Hong Kong", "Hong Kong, China", "kctang0410@link.cuhk.edu.hk"),
            ("Huazhong University of Science and Technology", "Wuhan, China", "chenjx@hust.edu.cn")]
    got = re.findall(r"\\IEEEauthorblockA\{\\textit\{(.*?)\}\\\\\n(.*?)\\\\\n(.*?)\}", block)
    assert got == want                         # organization (italic), "City, Country", e-mail


def test_every_record_of_the_author_list_has_the_same_order():
    info = re.search(r"/Author \((.*?)\)", MAIN)[1]
    assert [a.strip() for a in info.split(",")] == AUTHORS
    cff = (ROOT / "CITATION.cff").read_text()
    pairs = re.findall(r"- family-names: (\S+)\n\s+given-names: (\S+)", cff)
    assert [f"{given} {family}" for family, given in pairs] == AUTHORS
    creators = json.loads((ROOT / ".zenodo.json").read_text())["creators"]
    assert [" ".join(reversed(c["name"].split(", "))) for c in creators] == AUTHORS
    assert re.findall(r'name = "([^"]+)"', (ROOT / "pyproject.toml").read_text().split("authors = ")[1].split("\n")[0]) == AUTHORS


@pytest.mark.skipif(shutil.which("pdftotext") is None or not (PAPER / "main.pdf").exists(),
                    reason="needs poppler-utils and the built PDF")
def test_pdf_content_stream_has_the_registered_author_order():
    """What an indexing service reads: the order of the names in the PDF's content stream."""
    raw = subprocess.check_output(["pdftotext", "-raw", "-f", "1", "-l", "1", str(PAPER / "main.pdf"), "-"], text=True)
    found = [raw.find(name) for name in AUTHORS]
    assert -1 not in found and found == sorted(found), dict(zip(AUTHORS, found))
    info = subprocess.check_output(["pdfinfo", str(PAPER / "main.pdf")], text=True)
    assert re.search(r"^Page size:\s+612 x 792 pts \(letter\)", info, flags=re.M)
    assert re.search(r"^Author:\s+Ziqing Li, Jiawei Guo, Kaicheng Tang, Jianxi Chen$", info, flags=re.M)

#!/usr/bin/env python3
"""Check diagram export provenance, vector artwork and embedded figure fonts.

This is structural evidence; rendered-page visual inspection remains required.
Install the paper extra for PyMuPDF. Run from the repository root.
"""
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote
import xml.etree.ElementTree as ET

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper_icdm/figures"
ACTIVE = {"evaluation_paths.pdf", "fig_redund_calib.pdf", "fig_cell_forest.pdf"}
used = set()
for p in (ROOT / "paper_icdm/sections").glob("*.tex"):
    used.update(re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", p.read_text()))
assert used == ACTIVE, f"Unmapped manuscript figures: {used ^ ACTIVE}"
source = (FIG / "evaluation_paths.drawio").read_text().strip()
source_xml = ET.fromstring(source)
diagram = pymupdf.open(FIG / "evaluation_paths.pdf")
assert diagram.metadata["creator"] == "diagrams.net"
embedded = unquote(diagram.metadata["subject"]).strip()
assert ET.tostring(ET.fromstring(embedded)) == ET.tostring(source_xml), "PDF has stale embedded draw.io source"
labels = [c.attrib["value"] for c in source_xml.iter("mxCell") if "value" in c.attrib]
pdftext = diagram[0].get_text()
assert all(line in pdftext for label in labels for line in label.splitlines()), "Missing draw.io text in PDF"
for name in sorted(ACTIVE):
    doc = pymupdf.open(FIG / name)
    assert len(doc) == 1 and not doc.is_encrypted
    p = doc[0]
    assert not p.get_images(), f"Raster content in {name}"
    assert p.get_drawings(), f"No vector artwork in {name}"
    fonts = subprocess.check_output(["pdffonts", str(FIG/name)], text=True)
    for line in fonts.splitlines()[2:]:
        assert "Type 3" not in line, f"Type 3 font: {line}"
        assert line.rsplit(None, 5)[1] == "yes", f"Unembedded font: {line}"
    spans = [s for b in p.get_text("dict")["blocks"] if "lines" in b for l in b["lines"] for s in l["spans"] if s["text"].strip()]
    # The math subscript i in I(X_i; Z) is intentionally smaller than its label.
    ordinary = [s for s in spans if not (name == "fig_redund_calib.pdf" and s["text"].strip() == "i")]
    minimum = min(s["size"] for s in ordinary)
    printed_minimum = minimum * 252 / p.rect.width  # IEEEtran column = 3.5 in
    assert printed_minimum >= 8.9, f"Ordinary labels below approximately 9 pt in print: {name}, {printed_minimum}"
    for s in spans:
        assert p.rect.contains(pymupdf.Rect(s["bbox"])), f"Clipped text: {name}: {s['text']}"
    print(f"PASS {name}: vector PDF, embedded fonts, ordinary labels >= {printed_minimum:.2f} pt at column width")
print("PASS draw.io PDF embeds the exact editable source; all manuscript figures mapped")

"""Replace the raster figures in the exported report PDF with their vector versions.

Pages can only take bitmaps from a .docx, so the report is exported with 600-dpi PNGs; this step
swaps every embedded figure image for the matching vector PDF (same file name, .pdf), drawn in
exactly the same box. Figures are matched in reading order with the image links in content.md.

    uv run --with pikepdf python experiments/vectorize_pdf.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pikepdf

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "report"


def main(pdf_path: Path = REP / "Apart_report.pdf"):
    order = re.findall(r"^!\[.*\]\((figures/[^)]+)\.png\)", (REP / "content.md").read_text(),
                       flags=re.MULTILINE)
    vectors = [REP / f"{p}.pdf" for p in order]
    missing = [v for v in vectors if not v.exists()]
    if missing:
        sys.exit(f"missing vector figures: {missing}")
    pdf = pikepdf.open(pdf_path, allow_overwriting_input=True)
    sources = [pikepdf.open(v) for v in vectors]
    k = 0
    for page in pdf.pages:
        xobjs = page.Resources.get("/XObject")
        if xobjs is None:
            continue
        for operands, op in pikepdf.parse_content_stream(page):
            if str(op) != "Do":
                continue
            name = operands[0]
            xo = xobjs.get(name)
            if xo is None or xo.get("/Subtype") != "/Image":
                continue
            if k >= len(sources):
                sys.exit("more images in the PDF than figures in content.md")
            form = pdf.copy_foreign(sources[k].pages[0].as_form_xobject())
            x0, y0, x1, y1 = (float(v) for v in form.BBox)
            w, h = x1 - x0, y1 - y0
            ratio_img = float(xo.Width) / float(xo.Height)
            if abs(ratio_img - w / h) > 0.02 * ratio_img:
                sys.exit(f"aspect mismatch for {vectors[k].name}: {ratio_img:.3f} vs {w / h:.3f}")
            form.Matrix = pikepdf.Array([1 / w, 0, 0, 1 / h, -x0 / w, -y0 / h])
            xobjs[name] = form
            k += 1
    if k != len(sources):
        sys.exit(f"replaced {k} images but content.md has {len(sources)} figures")
    pdf.save(pdf_path)
    print(f"vectorized {k} figures in {pdf_path.name}")


if __name__ == "__main__":
    main()

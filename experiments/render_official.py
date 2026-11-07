"""Render report/content.md into the OFFICIAL Apart template (docx) and export a PDF with Pages.

Numbers come from report/numbers.json, written by the analysis scripts. Pre-existing code (Oct 2026), adapted.
Markup in content.md (deliberately small):
  % TITLE: ...            % AUTHOR: Name | Affiliation        % ABSTRACT ... % END
  ## Heading / ### Subheading
  paragraphs with **bold**, *italic*, `code`, ^sup^ and {{placeholders}}
  "1. " numbered and "- " bulleted lines
  ![Figure caption with **bold** lead](figures/file.png){6.3}   (width in inches)
  | table | rows |  (first row = header; separator row of dashes is skipped)
  :::box Title  ...lines...  :::      (bulleted box)
  :::algo Title ...lines...  :::      (box, lines as written: numbered steps)
  $$ latex                            (display equation, typeset as Word math)
  \\( latex \\) inline                 (inline equation, typeset as Word math)
  <<pagebreak>>

    uv run python experiments/render_official.py
Writes report/official/Apart_report.docx and report/Apart_report.pdf; prints the page count, the
abstract word count and any unfilled placeholder.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import json
from omml import to_omml
import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "report"
TEMPLATE = REP / "official" / "apart_template_nov2026.docx"
OUT_DOCX = REP / "official" / "Apart_report.docx"
OUT_PDF = REP / "Apart_report.pdf"
BODY_PT = 10
FONT = "Old Standard TT"
GREEK = re.compile(r"([\u0370-\u03ff]+)")
INLINE = re.compile(r"(\\\(.+?\\\)|\*\*.+?\*\*|\*.+?\*|`.+?`|\^.+?\^|~[^~\s]+?~)")



_SIMPLE_MACROS = [("\\dots", "…"), ("\\ldots", "…"), ("\\ge", "≥"), ("\\le", "≤"), ("\\ne", "≠"),
    ("\\alpha", "α"), ("\\gamma", "γ"), ("\\Gamma", "Γ"), ("\\tau", "τ"), ("\\pi", "π"),
    ("\\varnothing", "∅"), ("\\emptyset", "∅"), ("\\mid", "|"), ("\\cdot", "·"), ("\\in", "∈"),
    ("\\sim", "∼"), ("\\exists", "∃"), ("\\infty", "∞"), ("\\mathcal{A}", "𝒜"), ("\\mathcal{G}", "𝒢"),
    ("\\tilde a", "ã"), ("\\tilde{a}", "ã"), ("\\mathbb{1}", "𝟙"), ("\\{", "{"), ("\\}", "}"),
    ("\\quad", " "), ("\\,", " "), ("\\ ", " "), ("\\%", "%")]


def simple_math(latex: str):
    """Inline math made only of symbols, sub/superscripts and relations -> list of
    (text, italic, position) runs in the body font; None when it needs real typesetting."""
    s = latex
    for a, b in _SIMPLE_MACROS:
        s = s.replace(a, b)
    if "\\" in s:
        return None
    out, i, n = [], 0, len(s)
    def take_group(j):  # returns (text, next index) for _x or _{...}
        if j < n and s[j] == "{":
            k = s.index("}", j)
            return s[j + 1:k], k + 1
        return s[j], j + 1
    while i < n:
        ch = s[i]
        if ch in "_^":
            g, i = take_group(i + 1)
            if "{" in g or "\\" in g:
                return None
            for piece in g:
                out.append((piece, piece.isalpha() and piece.isascii(), "sub" if ch == "_" else "sup"))
            continue
        out.append((ch, ch.isalpha() and ch.isascii(), ""))
        i += 1
    merged = []  # merge adjacent runs with the same style
    for txt, ital, pos in out:
        if merged and merged[-1][1] == ital and merged[-1][2] == pos:
            merged[-1] = (merged[-1][0] + txt, ital, pos)
        else:
            merged.append((txt, ital, pos))
    return merged

# ---------------------------------------------------------------- inline text -----------
NOLIG = re.compile(r"f(?=[fil])")
LIG_SPLIT = re.compile(r"(?<=f)(?=[fil])")


def nolig(t: str) -> str:
    """Break fi, fl and ff ligatures with a zero-width non-joiner, so PDF text copies as words."""
    return t  # zero-width joiners made line spacing uneven in Pages, ligatures are switched off in the styles instead


def _no_ligatures(d):
    """Ask the renderer not to form fi and fl ligatures, so the PDF text copies as words."""
    W14 = "http://schemas.microsoft.com/office/word/2010/wordml"
    tag = "{%s}ligatures" % W14
    xml = '<w14:ligatures xmlns:w14="%s" w14:val="none"/>' % W14
    rpr = d.styles.element.find(qn("w:docDefaults") + "/" + qn("w:rPrDefault") + "/" + qn("w:rPr"))
    if rpr is not None and rpr.find(tag) is None:
        rpr.append(parse_xml(xml))
    for st in d.styles.element.findall(qn("w:style")):
        srpr = st.find(qn("w:rPr"))
        if srpr is not None and srpr.find(tag) is None:
            srpr.append(parse_xml(xml))
    for r in d.element.body.iter(qn("w:r")):
        rp = r.find(qn("w:rPr"))
        if rp is None:
            rp = OxmlElement("w:rPr"); r.insert(0, rp)
        if rp.find(tag) is None:
            rp.append(parse_xml(xml))


def add_runs(par, text: str, size: float | None = None, italic_all: bool = False,
             color: RGBColor | None = None):
    for part in INLINE.split(text):
        if not part:
            continue
        bold = italic = code = sup = sub = False
        if part.startswith("\\(") and part.endswith("\\)"):
            simple = simple_math(part[2:-2])
            if simple is None:
                par._p.append(parse_xml(to_omml(part[2:-2])))
            else:  # plain symbols: body font, italic letters, real sub/superscripts
                for txt, ital, pos in simple:
                    rr = par.add_run(txt)
                    rr.italic = ital or None
                    if pos == "sub":
                        rr.font.subscript = True
                    elif pos == "sup":
                        rr.font.superscript = True
                    if size:
                        rr.font.size = Pt(size)
                    if GREEK.fullmatch(txt) or txt in "∅≥≤≠∈∼∃∞·|…{}":
                        rr.font.name = "Times New Roman"
            continue
        if part.startswith("**") and part.endswith("**"):
            part, bold = part[2:-2], True
        elif part.startswith("*") and part.endswith("*") and len(part) > 1:
            part, italic = part[1:-1], True
        elif part.startswith("`") and part.endswith("`"):
            part, code = part[1:-1], True
        elif part.startswith("^") and part.endswith("^"):
            part, sup = part[1:-1], True
        elif part.startswith("~") and part.endswith("~") and len(part) > 2:
            part, sub = part[1:-1], True
        # Old Standard TT has no Greek: set Greek letters in Times New Roman (same metrics)
        if not code:
            part = nolig(part)
        if not code and GREEK.search(part):
            for piece in GREEK.split(part):
                if piece:
                    rr = par.add_run(piece)
                    rr.bold, rr.italic = bold or None, (italic or italic_all) or None
                    if sup:
                        rr.font.superscript = True
                    if sub:
                        rr.font.subscript = True
                    if size:
                        rr.font.size = Pt(size)
                    if color is not None:
                        rr.font.color.rgb = color
                    if GREEK.fullmatch(piece):
                        rr.font.name = "Times New Roman"
            continue
        chunks = [part] if code else LIG_SPLIT.split(part)
        for ci, chunk in enumerate(chunks):
            if ci:  # a 1-pt space in its own run stops Pages from forming an fi, fl or ff ligature
                gap = par.add_run("\u200c")
                gap.font.size = Pt(1)
            if not chunk:
                continue
            run = par.add_run(chunk)
            run.bold = bold or None
            run.italic = (italic or italic_all) or None
            if code:
                run.font.name = "Courier New"
            if sup:
                run.font.superscript = True
            if sub:
                run.font.subscript = True
            if size:
                run.font.size = Pt(size)
            if color is not None:
                run.font.color.rgb = color
    return par


def fill(text: str, v: dict, missing: list) -> str:
    def sub(m):
        if m.group(1) in v:
            return str(v[m.group(1)])
        missing.append(m.group(1))
        return f"[{m.group(1)}]"

    return re.sub(r"\{\{([^{}]+)\}\}", sub, text)


# ---------------------------------------------------------------- template surgery ------
def set_cell_text_runs(paragraph, text):
    """Replace a paragraph's text while keeping its first run's formatting."""
    runs = paragraph.runs
    if not runs:
        paragraph.add_run(nolig(text))
        return
    runs[0].text = nolig(text)
    for r in runs[1:]:
        if r._element.find(qn("w:footnoteReference")) is None:
            r.text = ""


def fill_front_matter(d, title: str, authors: list[tuple[str, str]], abstract: str):
    t0 = d.tables[0]
    title_par = next(p for p in t0.cell(0, 0).paragraphs if "PROJECT TITLE" in p.text)
    set_cell_text_runs(title_par, title)
    # author slots live in nested tables: "Author name k" / "Affiliation" paragraphs
    slots = []
    for p in t0._element.iter(qn("w:p")):
        txt = "".join(t.text or "" for t in p.iter(qn("w:t")))
        if txt.startswith("Author name") or txt.strip() == "Affiliation":
            slots.append(p)
    pairs = [(slots[i], slots[i + 1]) for i in range(0, len(slots) - 1, 2)
             if "".join(t.text or "" for t in slots[i].iter(qn("w:t"))).startswith("Author name")]
    for k, (pn, pa) in enumerate(pairs):
        if k >= len(authors):  # drop unused author slots entirely (they leave blank lines)
            for p in (pn, pa):
                p.getparent().remove(p)
            continue
        name, aff = authors[k]
        for p in (pn, pa):  # centre the remaining author block
            ppr = p.find(qn("w:pPr"))
            if ppr is None:
                ppr = OxmlElement("w:pPr")
                p.insert(0, ppr)
            for tag in ("w:jc", "w:ind"):
                for el in ppr.findall(qn(tag)):
                    ppr.remove(el)
            jc = OxmlElement("w:jc")
            jc.set(qn("w:val"), "center")
            ppr.append(jc)
        for p, txt in ((pn, name), (pa, aff)):
            for tag in ("w:tab", "w:br"):  # template tabs and breaks shift the text off centre
                for el in list(p.iter(qn(tag))):
                    el.getparent().remove(el)
            ts = list(p.iter(qn("w:t")))
            if ts:
                ts[0].text = txt
                ts[0].set("{http://www.w3.org/XML/1998/namespace}space", "default")
                for t in ts[1:]:
                    t.text = ""
    # the template's author slot is a 3-column nested table (name | affiliation | empty) that
    # renders side by side; make it one wide cell with the name above the affiliation
    for pn, pa in pairs[: len(authors)]:
        tc_n = pn.getparent()
        tr = tc_n.getparent()
        tbl = tr.getparent()
        tc_n.append(pa)  # move the affiliation paragraph under the name
        for tc in list(tr.findall(qn("w:tc"))):
            if tc is not tc_n:
                tr.remove(tc)
        tblpr = tbl.find(qn("w:tblPr"))
        for w in tblpr.findall(qn("w:tblW")):
            w.set(qn("w:w"), "9000")
        grid = tbl.find(qn("w:tblGrid"))
        if grid is not None:
            for el in list(grid):
                grid.remove(el)
            gc = OxmlElement("w:gridCol")
            gc.set(qn("w:w"), "9000")
            grid.append(gc)
        tcw = tc_n.find(qn("w:tcPr")).find(qn("w:tcW"))
        if tcw is not None:
            tcw.set(qn("w:w"), "9000")
    # nested author tables left empty by the removed slots only add vertical space: drop them
    for tbl in list(t0._element.iter(qn("w:tbl")))[::-1]:
        if tbl is t0._element:
            continue
        if not "".join(x.text or "" for x in tbl.iter(qn("w:t"))).strip():
            tbl.getparent().remove(tbl)
    # abstract: nested 1x1 table whose text starts with "Summarize your project"
    for tbl in t0._element.iter(qn("w:tbl")):
        txt = "".join(t.text or "" for t in tbl.iter(qn("w:t")))
        if txt.startswith("Summarize your project"):
            cell = tbl.find(qn("w:tr")).find(qn("w:tc"))
            pars = cell.findall(qn("w:p"))
            for p in pars[1:]:
                cell.remove(p)
            p = docx.text.paragraph.Paragraph(pars[0], d)
            for r in list(p.runs):
                r._element.getparent().remove(r._element)
            add_runs(p, abstract, size=BODY_PT, italic_all=True)
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            break


def strip_body(d):
    """Remove the guidance info box and every body element after it (keep sectPr)."""
    body = d.element.body
    children = list(body.iterchildren())
    info = d.tables[1]._element
    start = children.index(info)
    for el in children[start:]:
        if el.tag != qn("w:sectPr"):
            body.remove(el)


# ---------------------------------------------------------------- body blocks -----------
def para(d, style="normal", align=None, space_after=3):
    p = d.add_paragraph(style=style)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.space_before = Pt(0)
    if align:
        p.alignment = align
    return p


def add_heading(d, text, level):
    p = d.add_paragraph(style="Heading 2" if level == 2 else "Heading 3")
    add_runs(p, text)
    p.paragraph_format.space_before = Pt(8 if level == 2 else 5)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True


def add_body(d, text):
    p = para(d, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    add_runs(p, text, size=BODY_PT)
    if text.startswith("**") and len(text) < 60:  # a run-in heading on its own line
        p.paragraph_format.keep_with_next = True


def add_list_item(d, marker, text, size=BODY_PT, level=1):
    p = para(d, space_after=2)
    numbered = marker[:-1].isdigit()
    indent = 0.3 if numbered and level == 1 else 0.28 * level
    p.paragraph_format.left_indent = Inches(indent)
    p.paragraph_format.first_line_indent = Inches(-0.3 if numbered and level == 1 else -0.2)
    add_runs(p, f"{marker}\t" if marker != "•" else "•\t", size=size)
    add_runs(p, text, size=size)
    p.paragraph_format.tab_stops.add_tab_stop(Inches(indent))


def add_figure(d, path, caption, width):
    p = para(d, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(REP / path), width=Inches(width))
    c = para(d, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_after=6)
    add_runs(c, caption, size=8.5, color=RGBColor(0x40, 0x40, 0x40))


def borders(table, color="BBBBBB"):
    tbl = table._element
    pr = tbl.tblPr
    b = OxmlElement("w:tblBorders")
    for edge in ("top", "bottom", "insideH"):
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:val"), "single")
        e.set(qn("w:sz"), "4")
        e.set(qn("w:color"), color)
        b.append(e)
    pr.append(b)


def add_table(d, rows):
    if d.paragraphs:  # keep a table's caption on the same page as the table
        d.paragraphs[-1].paragraph_format.keep_with_next = True
    t = d.add_table(rows=len(rows), cols=len(rows[0]))
    borders(t)
    # column widths in proportion to the longest entry (header counted less), within 6.3 in
    plain = [[re.sub(r"[*`~^{}]|\\\(|\\\)", "", c) for c in row] for row in rows]
    need = [max(max((len(r[j]) for r in plain[1:]), default=0),
                max(len(w) for w in plain[0][j].split()) + 1, 4) for j in range(len(rows[0]))]
    need = [n ** 0.85 for n in need]
    tot = sum(need)
    t.autofit = False
    for j, n in enumerate(need):
        w = Inches(6.3 * n / tot)
        t.columns[j].width = w
        for row in t.rows:
            row.cells[j].width = w
    tblpr = t._element.tblPr
    lay = OxmlElement("w:tblLayout")
    lay.set(qn("w:type"), "fixed")
    tblpr.append(lay)
    for i, r in enumerate(rows):
        for j, cell_text in enumerate(r):
            cell = t.cell(i, j)
            p = cell.paragraphs[0]
            add_runs(p, f"**{cell_text}**" if i == 0 else cell_text, size=8)
            p.paragraph_format.space_after = Pt(1)
            # keep the table on one page: every row stays with the next one
            p.paragraph_format.keep_with_next = i < len(rows) - 1
        trpr = t.rows[i]._tr.get_or_add_trPr()
        cant = OxmlElement("w:cantSplit")
        trpr.append(cant)
    if len(rows) > 9:  # long tables may break across pages (rows themselves stay whole)
        for i in range(len(rows)):
            for j in range(len(rows[0])):
                t.cell(i, j).paragraphs[0].paragraph_format.keep_with_next = i < 1
    para(d, space_after=4)


def add_box(d, title, lines, bullets=True):
    t = d.add_table(rows=1, cols=1)
    cell = t.cell(0, 0)
    tcpr = cell._element.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), "F3F1EA")
    tcpr.append(shd)
    borders(t, "999999")
    p = cell.paragraphs[0]
    add_runs(p, f"**{title}**", size=9)
    for ln in lines:
        q = cell.add_paragraph()
        q.paragraph_format.space_after = Pt(1)
        q.paragraph_format.left_indent = Inches(0.15)
        q.paragraph_format.first_line_indent = Inches(-0.12)
        add_runs(q, ("• " + ln.lstrip("- ").strip()) if bullets else ln.strip(), size=8.5)
    para(d, space_after=4)


def page_break(d):
    from docx.enum.text import WD_BREAK

    d.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


# ---------------------------------------------------------------- main ------------------
def render(text: str, v: dict):
    missing: list[str] = []
    text = fill(text, v, missing)
    lines = text.splitlines()
    title = next(ln[len("% TITLE:"):].strip() for ln in lines if ln.startswith("% TITLE:"))
    authors = [tuple(x.strip() for x in ln[len("% AUTHOR:"):].split("|"))
               for ln in lines if ln.startswith("% AUTHOR:")]
    a0 = lines.index("% ABSTRACT")
    a1 = lines.index("% END")
    abstract = " ".join(ln.strip() for ln in lines[a0 + 1:a1] if ln.strip())
    d = docx.Document(str(TEMPLATE))
    normal = d.styles["normal"]
    normal.font.size = Pt(BODY_PT)
    # the template's own text is set in Old Standard TT; use it for body, headings and captions
    for name in ("normal", "Heading 2", "Heading 3"):
        st = d.styles[name]
        st.font.name = FONT
        rfonts = st.element.get_or_add_rPr().get_or_add_rFonts()
        for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
            rfonts.set(qn(attr), FONT)
    fill_front_matter(d, title, authors, abstract)
    strip_body(d)
    i, body = 0, lines[a1 + 1:]
    buf: list[str] = []

    def flush():
        if buf:
            add_body(d, " ".join(buf))
            buf.clear()

    while i < len(body):
        ln = body[i].rstrip()
        if not ln.strip():
            flush()
        elif ln.startswith("### "):
            flush()
            add_heading(d, ln[4:], 3)
        elif ln.startswith("## "):
            flush()
            add_heading(d, ln[3:], 2)
        elif ln.startswith("!["):
            flush()
            m = re.match(r"!\[(.*)\]\((.+?)\)(\{([\d.]+)\})?", ln)
            add_figure(d, m.group(2), m.group(1), float(m.group(4) or 6.3))
        elif ln.startswith("|"):
            flush()
            rows = []
            while i < len(body) and body[i].startswith("|"):
                cells = [c.strip() for c in body[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            add_table(d, rows)
            continue
        elif ln.startswith("$$ "):
            flush()
            q = para(d, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=4)
            q._p.append(parse_xml(to_omml(ln[3:].strip(), display=True)))
        elif ln.startswith(":::box") or ln.startswith(":::algo"):
            flush()
            kind = "algo" if ln.startswith(":::algo") else "box"
            title_b, items = ln[len(":::" + kind):].strip(), []
            i += 1
            while not body[i].startswith(":::"):
                if body[i].strip():
                    items.append(body[i])
                i += 1
            add_box(d, title_b, items, bullets=kind == "box")
        elif ln.strip() == "<<pagebreak>>":
            flush()
            page_break(d)
        elif re.match(r"^\d+\. ", ln):
            flush()
            k, rest = ln.split(". ", 1)
            add_list_item(d, f"{k}.", rest)
        elif ln.startswith("- "):
            flush()
            add_list_item(d, "•", ln[2:])
        elif ln.startswith("  - "):
            flush()
            add_list_item(d, "–", ln[4:], level=2)
        elif ln.startswith("% "):
            pass
        else:
            buf.append(ln.strip())
        i += 1
    flush()
    return d, abstract, missing


def tighten_front_matter(d):
    """Drop the empty spacer paragraphs between the template's title block and the body text."""
    t0 = d.tables[0]._tbl
    nxt = t0.getnext()
    if nxt is not None and nxt.tag == qn("w:p") and not "".join(nxt.itertext()).strip():
        nxt.getparent().remove(nxt)
    cell = d.tables[0].rows[-1].cells[0]
    last = cell.paragraphs[-1]
    if not last.text.strip():
        last.paragraph_format.space_before = Pt(0)
        last.paragraph_format.space_after = Pt(0)
        last.paragraph_format.line_spacing = Pt(1)


def main():
    nj = REP / "numbers.json"
    v = json.loads(nj.read_text()) if nj.exists() else {}
    d, abstract, missing = render((REP / "content.md").read_text(), v)
    tighten_front_matter(d)
    OUT_DOCX.parent.mkdir(parents=True, exist_ok=True)
    _no_ligatures(d)
    d.save(str(OUT_DOCX))
    strict = len([w for w in re.split(r"[\s\-–—/]+", abstract) if re.search(r"\w", w)])
    print(f"abstract words: {len(abstract.split())} (strict, hyphens and dashes split: {strict})")
    # plain-text abstract for the submission form's "Project summary" field
    plain = re.sub(r"[*`~^]", "", abstract)
    (REP / "abstract_for_form.txt").write_text(plain + "\n")
    script = f'''with timeout of 180 seconds
tell application "Pages"
  set doc to open POSIX file "{OUT_DOCX}"
  export doc to POSIX file "{OUT_PDF}" as PDF
  close doc saving no
end tell
end timeout'''
    for attempt in range(3):  # Pages occasionally fails transiently
        r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, check=False)
        if r.returncode == 0:
            break
        print(f"Pages export attempt {attempt + 1} failed: {r.stderr.strip()[:200]}")
    else:
        raise RuntimeError("Pages export failed three times")
    # Pages takes only bitmaps from a .docx: swap the 600-dpi figures for their vector versions
    vec = subprocess.run(["uv", "run", "--with", "pikepdf", "python",
                          str(ROOT / "experiments" / "vectorize_pdf.py")],
                         capture_output=True, text=True, check=False, cwd=ROOT)
    print(vec.stdout.strip() if vec.returncode == 0 else f"VECTORIZE FAILED: {vec.stderr[-400:]}")
    info = subprocess.run(["pdfinfo", str(OUT_PDF)], capture_output=True, text=True, check=False)
    pages = re.search(r"Pages:\s+(\d+)", info.stdout)
    print(f"pages: {pages.group(1) if pages else '?'}  -> {OUT_PDF}")
    print(f"unfilled placeholders: {sorted(set(missing))}")


if __name__ == "__main__":
    main()

"""A small LaTeX-to-OMML converter for the report's equations (Word's native math, which Pages
renders as typeset math). It covers the subset the report uses: sub- and superscripts, fractions,
sums and products with limits, limits under max/min, accents, Greek letters, common relations,
upright text and blackboard/calligraphic letters.

    to_omml(r"e_{ij,t} = \\frac{q(a)}{\\sum_{x} q(x)}", display=True)  -> XML string
"""

from __future__ import annotations

from xml.sax.saxutils import escape

M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"

SYMBOLS = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "varepsilon": "ε",
    "theta": "θ", "lambda": "λ", "mu": "μ", "nu": "ν", "pi": "π", "rho": "ρ", "sigma": "σ",
    "tau": "τ", "phi": "φ", "psi": "ψ", "omega": "ω", "Gamma": "Γ", "Delta": "Δ", "Pi": "Π",
    "Sigma": "Σ", "Omega": "Ω", "mid": "∣", "cdot": "⋅", "in": "∈", "notin": "∉", "le": "≤",
    "leq": "≤", "ge": "≥", "geq": "≥", "ne": "≠", "neq": "≠", "sim": "∼", "times": "×",
    "dots": "…", "ldots": "…", "cdots": "⋯", "infty": "∞", "to": "→", "forall": "∀",
    "exists": "∃", "emptyset": "∅", "varnothing": "∅", "#": "#", "{": "{", "}": "}", "|": "∥",
    "lvert": "|", "rvert": "|", "approx": "≈", "propto": "∝", "star": "⋆", "ast": "∗",
    "setminus": "∖", "otimes": "⊗", "oplus": "⊕", "cup": "∪", "cap": "∩", "subseteq": "⊆", "Rightarrow": "⇒", "colon": ":",
    "%": "%", "_": "_", "&": "&", "langle": "⟨", "rangle": "⟩", "prime": "′",
    "quad": " ", "qquad": "  ", ",": " ", ";": " ", ":": " ",
    " ": " ", "!": "",
}
FUNCS = {"max", "min", "log", "exp", "Pr", "argmax", "arg", "sup", "inf", "mean"}
NARY = {"sum": "∑", "prod": "∏"}
ACCENTS = {"tilde": "̃", "hat": "̂", "bar": "̅", "dot": "̇"}
BB = {"E": "𝔼", "P": "ℙ", "R": "ℝ", "N": "ℕ", "1": "𝟙"}
CAL = {"A": "𝒜", "F": "ℱ", "G": "𝒢", "H": "ℋ", "P": "𝒫", "S": "𝒮", "O": "𝒪", "V": "𝒱"}


def _run(text: str, style: str | None = None) -> str:
    if not text:
        return ""
    rpr = f'<m:rPr><m:sty m:val="{style}"/></m:rPr>' if style else ""
    return f'<m:r>{rpr}<m:t xml:space="preserve">{escape(text)}</m:t></m:r>'


class _Parser:
    def __init__(self, src: str):
        self.s, self.i = src, 0

    def peek(self) -> str:
        return self.s[self.i] if self.i < len(self.s) else ""

    def skip_ws(self):
        while self.peek().isspace():
            self.i += 1

    def read_cmd(self) -> str:
        self.i += 1  # backslash
        if not self.peek().isalpha():
            c = self.peek()
            self.i += 1
            return c
        j = self.i
        while self.peek().isalpha():
            self.i += 1
        return self.s[j:self.i]

    def read_group_raw(self) -> str:
        """Raw text of a {...} group (for \\text and friends)."""
        self.skip_ws()
        if self.peek() != "{":
            c = self.peek()
            self.i += 1
            return c
        depth, j = 0, self.i
        while self.i < len(self.s):
            c = self.s[self.i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    self.i += 1
                    return self.s[j + 1:self.i - 1]
            self.i += 1
        raise ValueError(f"unbalanced braces in {self.s!r}")

    def arg(self) -> str:
        """One argument: a {...} group or a single atom, as OMML."""
        self.skip_ws()
        if self.peek() == "{":
            self.i += 1
            out = self.seq(stop="}")
            self.i += 1
            return out
        return self.atom() or ""

    def scripts(self, base: str) -> str:
        sub = sup = None
        while True:
            self.skip_ws()
            c = self.peek()
            if c == "_" and sub is None:
                self.i += 1
                sub = self.arg()
            elif c == "^" and sup is None:
                self.i += 1
                sup = self.arg()
            elif c == "'" and sup is None:
                self.i += 1
                sup = _run("′")
            else:
                break
        if sub is None and sup is None:
            return base
        if sub is not None and sup is not None:
            return (f"<m:sSubSup><m:e>{base}</m:e><m:sub>{sub}</m:sub>"
                    f"<m:sup>{sup}</m:sup></m:sSubSup>")
        if sub is not None:
            return f"<m:sSub><m:e>{base}</m:e><m:sub>{sub}</m:sub></m:sSub>"
        return f"<m:sSup><m:e>{base}</m:e><m:sup>{sup}</m:sup></m:sSup>"

    def atom(self) -> str | None:
        self.skip_ws()
        c = self.peek()
        if not c or c == "}":
            return None
        if c == "{":
            self.i += 1
            inner = self.seq(stop="}")
            self.i += 1
            return self.scripts(inner)
        if c == "\\":
            name = self.read_cmd()
            if name in NARY:
                return self.nary(NARY[name])
            if name in FUNCS:
                text = "arg max" if name == "argmax" else name
                return self.limits(_run(text, "p")) + _run("\u2009", "p")
            if name in ("frac", "tfrac", "dfrac"):
                num, den = self.arg(), self.arg()
                return self.scripts(f"<m:f><m:num>{num}</m:num><m:den>{den}</m:den></m:f>")
            if name == "sqrt":
                body = self.arg()
                return self.scripts(f'<m:rad><m:radPr><m:degHide m:val="1"/></m:radPr><m:deg/>'
                                    f"<m:e>{body}</m:e></m:rad>")
            if name == "binom":
                n, k = self.arg(), self.arg()
                f = (f'<m:f><m:fPr><m:type m:val="noBar"/></m:fPr><m:num>{n}</m:num>'
                     f"<m:den>{k}</m:den></m:f>")
                return self.scripts(f"<m:d><m:e>{f}</m:e></m:d>")
            if name == "tilde":  # precomposed letters render better than an accent object
                self.skip_ws()
                ch = self.peek()
                if ch in "anoANO":
                    self.i += 1
                    return self.scripts(_run({"a": "ã", "n": "ñ", "o": "õ", "A": "Ã", "N": "Ñ",
                                              "O": "Õ"}[ch]))
            if name in ACCENTS:
                body = self.arg()
                return self.scripts(f'<m:acc><m:accPr><m:chr m:val="{ACCENTS[name]}"/></m:accPr>'
                                    f"<m:e>{body}</m:e></m:acc>")
            if name in ("text", "mathrm", "operatorname", "textrm"):
                return self.scripts(_run(self.read_group_raw(), "p"))
            if name in ("mathbf", "boldsymbol"):
                return self.scripts(_run(self.read_group_raw(), "b"))
            if name == "mathbb":
                raw = self.read_group_raw()
                return self.scripts(_run("".join(BB.get(ch, ch) for ch in raw), "p"))
            if name == "mathcal":
                raw = self.read_group_raw()
                return self.scripts(_run("".join(CAL.get(ch, ch) for ch in raw), "p"))
            if name in ("left", "right", "big", "Big", "bigl", "bigr", "Bigl", "Bigr"):
                self.skip_ws()
                if self.peek() == "\\":
                    nxt = self.read_cmd()
                    return _run(SYMBOLS.get(nxt, nxt), "p")
                ch = self.peek()
                self.i += 1
                return _run("" if ch == "." else ch, "p")
            if name in SYMBOLS:
                upright = not name[0].isalpha() or name[0].isupper() or name in ("quad", "qquad")
                return self.scripts(_run(SYMBOLS[name], "p" if upright else None))
            raise ValueError(f"unsupported LaTeX command \\{name}")
        self.i += 1
        if c.isalpha():
            return self.scripts(_run(c))
        if c.isdigit() or c == ".":
            j = self.i - 1
            while self.peek().isdigit() or (self.peek() == "." and
                                             self.s[self.i + 1:self.i + 2].isdigit()):
                self.i += 1
            return self.scripts(_run(self.s[j:self.i]))
        if c == "~":
            return _run(" ", "p")
        return self.scripts(_run(c, "p"))

    def limits(self, base: str) -> str:
        self.skip_ws()
        if self.peek() == "_":
            self.i += 1
            lim = self.arg()
            return f"<m:limLow><m:e>{base}</m:e><m:lim>{lim}</m:lim></m:limLow>"
        return base

    def nary(self, chr_: str) -> str:
        sub = sup = None
        while True:
            self.skip_ws()
            if self.peek() == "_" and sub is None:
                self.i += 1
                sub = self.arg()
            elif self.peek() == "^" and sup is None:
                self.i += 1
                sup = self.arg()
            else:
                break
        body = self.atom() or ""
        hide = ("" if sub is not None else '<m:subHide m:val="1"/>') + \
               ("" if sup is not None else '<m:supHide m:val="1"/>')
        return (f'<m:nary><m:naryPr><m:chr m:val="{chr_}"/><m:limLoc m:val="undOvr"/>{hide}'
                f"</m:naryPr><m:sub>{sub or ''}</m:sub><m:sup>{sup or ''}</m:sup>"
                f"<m:e>{body}</m:e></m:nary>")

    def seq(self, stop: str = "") -> str:
        out = []
        while self.i < len(self.s):
            self.skip_ws()
            if stop and self.peek() == stop:
                break
            a = self.atom()
            if a is None:
                break
            out.append(a)
        return "".join(out)


def to_omml(latex: str, display: bool = False) -> str:
    body = _Parser(latex).seq()
    math = f'<m:oMath xmlns:m="{M_NS}">{body}</m:oMath>'
    if display:
        return (f'<m:oMathPara xmlns:m="{M_NS}"><m:oMathParaPr><m:jc m:val="center"/>'
                f"</m:oMathParaPr>{math}</m:oMathPara>")
    return math

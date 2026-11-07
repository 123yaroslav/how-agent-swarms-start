"""Tiny CSV tables and the three deterministic transforms of the cluster (clean, merge,
aggregate). Row ids (r-xxxxxx) and checksums (ck-xxxxxxxx) are the fingerprints that the
provenance log follows from artifact to artifact."""

from __future__ import annotations

import hashlib
import re

RAW_COLS = ("row_id", "region", "sku", "units", "price")
AGG_COLS = ("region", "units", "revenue")
FP_RX = re.compile(r"\b(?:r-[0-9a-f]{6}|ck-[0-9a-f]{8})\b")


def fingerprints(text: str) -> set[str]:
    return set(FP_RX.findall(text or ""))


def parse_table(text: str) -> tuple[list[str], list[list[str]]] | None:
    """(header, rows) of a CSV text with optional '#' comment lines, else None."""
    lines = [ln.strip() for ln in (text or "").splitlines()]
    lines = [ln for ln in lines if ln and not ln.startswith("#")]
    if len(lines) < 2:
        return None
    header = [c.strip() for c in lines[0].split(",")]
    if len(header) < 2:
        return None
    rows = [[c.strip() for c in ln.split(",")] for ln in lines[1:]]
    if any(len(r) != len(header) for r in rows):
        return None
    return header, rows


def render(comment: str, header: tuple[str, ...] | list[str], rows: list[list[str]],
           footer: str | None = None) -> str:
    out = f"# {comment}\n" + ",".join(header) + "\n" + "".join(",".join(r) + "\n" for r in rows)
    return out + (f"# {footer}\n" if footer else "")


def _row_level(header: list[str]) -> bool:
    return all(c in header for c in RAW_COLS)


def clean_rows(header: list[str], rows: list[list[str]]) -> list[list[str]]:
    """Drop rows with an empty field, upper-case the region, two decimals for the price."""
    ix = [header.index(c) for c in RAW_COLS]
    out = []
    for r in rows:
        v = [r[i] for i in ix]
        if any(x == "" for x in v):
            continue
        try:
            units, price = int(float(v[3])), float(v[4])
        except ValueError:
            continue
        out.append([v[0], v[1].upper(), v[2], str(units), f"{price:.2f}"])
    return sorted(out)


def checksum(row_ids: list[str]) -> str:
    return "ck-" + hashlib.sha256("|".join(sorted(row_ids)).encode()).hexdigest()[:8]


def aggregate_rows(rows: list[list[str]]) -> list[list[str]]:
    """rows: cleaned row-level rows. Group by region: total units and revenue."""
    acc: dict[str, list[float]] = {}
    for _, region, _, units, price in rows:
        a = acc.setdefault(region, [0, 0.0])
        a[0] += int(units)
        a[1] += int(units) * float(price)
    return [[r, str(int(u)), f"{rev:.2f}"] for r, (u, rev) in sorted(acc.items())]


class OpError(ValueError):
    pass


def run_op(op: str, inputs: list[tuple[str, str]]) -> tuple[str, int]:
    """inputs: (name, text). Returns (output text, number of rows). Raises OpError."""
    op = (op or "").strip().lower()
    if op not in ("clean", "merge", "aggregate"):
        raise OpError(f"unknown op {op!r}; ops are clean, merge, aggregate")
    tables = []
    for name, text in inputs:
        t = parse_table(text)
        if t is None:
            raise OpError(f"{name} is not a table")
        tables.append((name, *t))
    if op in ("clean", "aggregate") and len(tables) != 1:
        raise OpError(f"{op} takes exactly one input")
    if op == "merge" and len(tables) < 2:
        raise OpError("merge takes two or more inputs")
    for name, header, _ in tables:
        if not _row_level(header):
            raise OpError(f"{name} is not a row-level table ({','.join(RAW_COLS)})")
    if op == "clean":
        name, header, rows = tables[0]
        out = clean_rows(header, rows)
        return render(f"cleaned from {name}", RAW_COLS, out), len(out)
    if op == "merge":
        seen: dict[str, list[str]] = {}
        for _, header, rows in tables:
            for r in clean_rows(header, rows):
                seen.setdefault(r[0], r)
        out = sorted(seen.values())
        names = ", ".join(n for n, _, _ in tables)
        return render(f"merged from {names}", RAW_COLS, out), len(out)
    name, header, rows = tables[0]
    cleaned = clean_rows(header, rows)
    out = aggregate_rows(cleaned)
    return (render(f"aggregate by region of {name}", AGG_COLS, out,
                   checksum([r[0] for r in cleaned])), len(out))


def same_table(a: str, b: str) -> bool:
    """Same header and rows (comment lines ignored)."""
    ta, tb = parse_table(a), parse_table(b)
    return ta is not None and tb is not None and ta == tb

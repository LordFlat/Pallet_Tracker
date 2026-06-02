"""Turn a SAP production plan into structured rows.

Input is tab/pipe/comma/multi-space separated table text. The user can paste it
directly, or it can be produced from a screenshot by :mod:`job_sheets.ocr` and
then fed through :func:`parse_pasted_rows` — both paths converge here.

We auto-detect the delimiter and map columns either by a header row or by the
documented fixed order.

Required SAP columns: Material Description, Text, Work Center, Prod. Order,
Act.Qty. Rows whose work center is out of scope are dropped here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .lines import is_allowed

# Expected column order when no header row is present.
FIELD_ORDER = ["material_description", "text", "work_center", "prod_order", "act_qty"]

_HEADER_HINTS = {
    "material_description": ("material", "description"),
    "text": ("text",),
    "work_center": ("work center", "work centre", "work_center", "wc"),
    "prod_order": ("prod", "order"),
    "act_qty": ("act", "qty", "quantity"),
}


@dataclass
class RawRow:
    material_description: str
    text: str
    work_center: str
    prod_order: str
    act_qty: float
    act_qty_raw: str  # original cell, so we can flag OCR garbage for review
    # True if Act.Qty parsed as a real number (including an explicit 0). False
    # means the cell was empty/unreadable — only THAT needs review. An explicit
    # 0 is a valid business case (job not yet produced; paperwork still prepared).
    act_qty_ok: bool = True


def _split_line(line: str) -> list[str]:
    if "\t" in line:
        parts = line.split("\t")
    elif "|" in line:
        parts = line.split("|")
    elif "  " in line:  # two or more spaces -> column gap
        parts = re.split(r"\s{2,}", line)
    elif "," in line:
        parts = line.split(",")
    else:
        parts = [line]
    return [p.strip() for p in parts]


def _looks_like_header(cells: list[str]) -> dict[str, int] | None:
    mapping: dict[str, int] = {}
    for i, cell in enumerate(cells):
        low = cell.lower()
        for field, hints in _HEADER_HINTS.items():
            if field in mapping:
                continue
            if any(h in low for h in hints):
                mapping[field] = i
    # Need at least description + work center to trust it as a header.
    if "material_description" in mapping and "work_center" in mapping:
        return mapping
    return None


def _to_qty(value: str) -> tuple[float, str, bool]:
    """Parse an Act.Qty cell -> (value, raw, ok).

    ``ok`` is False only when the cell holds no parseable number (empty or OCR
    garbage). An explicit "0" parses fine and is ``ok=True``.
    """
    raw = (value or "").strip()
    cleaned = re.sub(r"[^\d.\-]", "", raw)
    try:
        return float(cleaned), raw, True
    except ValueError:
        return 0.0, raw, False


def parse_pasted_rows(text: str) -> list[RawRow]:
    """Parse pasted SAP table text into in-scope :class:`RawRow` objects."""
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return []

    mapping: dict[str, int] | None = None
    body = lines

    first_cells = _split_line(lines[0])
    header_map = _looks_like_header(first_cells)
    if header_map:
        mapping = header_map
        body = lines[1:]

    rows: list[RawRow] = []
    for line in body:
        cells = _split_line(line)
        if not any(cells):
            continue

        def cell(field: str) -> str:
            if mapping is not None:
                i = mapping.get(field)
            else:
                i = FIELD_ORDER.index(field)
            return cells[i].strip() if i is not None and i < len(cells) else ""

        wc = cell("work_center")
        if not is_allowed(wc):
            continue

        qty, qty_raw, qty_ok = _to_qty(cell("act_qty"))
        rows.append(
            RawRow(
                material_description=cell("material_description"),
                text=cell("text"),
                work_center=wc,
                prod_order=cell("prod_order"),
                act_qty=qty,
                act_qty_raw=qty_raw,
                act_qty_ok=qty_ok,
            )
        )

    return rows

"""Work-center filtering and line normalization.

Allowed work centers (MVP):
    FLOW 5, FLOW 7, LINERLESS,
    TOP SEAL and all numbered top sealers, written either "TOP SEAL n" or
    "TP SEAL n" (e.g. TP SEAL2, TP SEAL4, TP SEAL5, TP SEAL6, TP SEAL7).

Ignored: GIRO 3/4/5, MPACK, H/LINES, OFF SITE and everything else.

Output line normalization (the real top-sealer number is preserved):
    FLOW 5 -> F5, FLOW 7 -> F7, LINERLESS -> LL,
    TOP SEAL -> TS, TP SEAL2 -> TS2, TP SEAL4 -> TS4, TP SEAL5 -> TS5,
    TP SEAL6 -> TS6, TP SEAL7 -> TS7
"""

from __future__ import annotations

import re

from .text_utils import collapse_spaces

EXPLICIT_LINES = {
    "FLOW 5": "F5",
    "FLOW 7": "F7",
    "LINERLESS": "LL",
}

# Compact (space-stripped, upper-cased) prefixes for the top-sealer lines. SAP
# writes these as both "TOP SEAL n" and "TP SEAL n"; both are in scope and map
# to the same "TS" line code.
_TOP_SEAL_PREFIXES = ("TOPSEAL", "TPSEAL")


def _canonical(work_center: str) -> str:
    return collapse_spaces(work_center).upper()


def _compact(work_center: str) -> str:
    """Upper-cased, space-stripped form for tolerant prefix matching."""
    return _canonical(work_center).replace(" ", "")


def _flow_digit(compact: str) -> str | None:
    """First digit after 'FLOW' (the line number), or None if unreadable."""
    digits = re.findall(r"\d", compact[4:])
    return digits[0] if digits else None


def _seal_code(compact: str) -> str | None:
    """Top-sealer line code preserving its number: TS, TS2, TS4, TS5, TS6,
    TS7, ... or None.

    "TOP SEAL"/"TP SEAL" with no number -> "TS"; "TP SEAL5" -> "TS5", etc.
    """
    for prefix in _TOP_SEAL_PREFIXES:
        if compact.startswith(prefix):
            m = re.search(r"\d+", compact[len(prefix):])
            return "TS" + (m.group() if m else "")
    return None


def is_allowed(work_center: str) -> bool:
    """True if the work center is in scope (FLOW 5/7, LINERLESS, TOP SEAL*).

    Tolerant of OCR noise in the digit, e.g. "FLOW 5S" or "FLOW " with an
    unreadable number, so valid production rows are not silently dropped.
    """
    compact = _compact(work_center)
    if not compact:
        return False
    if compact.startswith("LINER"):  # LINERLESS (and OCR-truncated variants)
        return True
    if compact.startswith(_TOP_SEAL_PREFIXES):  # TOP SEAL / TP SEAL (+ numbers)
        return True
    if compact.startswith("FLOW"):
        d = _flow_digit(compact)
        # FLOW 5 / FLOW 7 only; an unreadable digit is allowed (defaulted later).
        return d in (None, "5", "7")
    return False


def normalize_line(work_center: str) -> str:
    """Map a SAP work center to its short PD080 line code (OCR-tolerant)."""
    compact = _compact(work_center)
    if compact.startswith("LINER"):
        return "LL"
    seal = _seal_code(compact)
    if seal is not None:  # TOP/TP SEAL with its real number preserved
        return seal
    if compact.startswith("FLOW"):
        return "F7" if _flow_digit(compact) == "7" else "F5"
    # Unknown but not filtered out elsewhere: fall back to the raw value.
    return _canonical(work_center)

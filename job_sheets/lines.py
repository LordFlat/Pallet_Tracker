"""Work-center filtering and line normalization.

Allowed work centers (MVP):
    FLOW 5, FLOW 7, LINERLESS,
    TOP SEAL, TOP SEAL 2, TOP SEAL 4, TOP SEAL 5, TOP SEAL 6, TOP SEAL 7

Ignored: GIRO 3/4/5, MPACK, H/LINES, OFF SITE and everything else.

Output line normalization:
    FLOW 5 -> F5, FLOW 7 -> F7, LINERLESS -> LL, any TOP SEAL* -> TS
"""

from __future__ import annotations

import re

from .text_utils import collapse_spaces

EXPLICIT_LINES = {
    "FLOW 5": "F5",
    "FLOW 7": "F7",
    "LINERLESS": "LL",
}


def _canonical(work_center: str) -> str:
    return collapse_spaces(work_center).upper()


def _compact(work_center: str) -> str:
    """Upper-cased, space-stripped form for tolerant prefix matching."""
    return _canonical(work_center).replace(" ", "")


def _flow_digit(compact: str) -> str | None:
    """First digit after 'FLOW' (the line number), or None if unreadable."""
    digits = re.findall(r"\d", compact[4:])
    return digits[0] if digits else None


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
    if compact.startswith("TOPSEAL"):  # TOP SEAL, TOP SEAL 2, ...
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
    if compact.startswith("TOPSEAL"):
        return "TS"
    if compact.startswith("FLOW"):
        return "F7" if _flow_digit(compact) == "7" else "F5"
    # Unknown but not filtered out elsewhere: fall back to the raw value.
    return _canonical(work_center)

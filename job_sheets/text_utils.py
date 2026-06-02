"""Text normalization helpers shared by the matcher and processor.

SAP descriptions are messy: extra digits, missing spaces, different x/X,
strange suffixes, random codes and OCR noise. We never rely on exact matching.
Pack-size numbers (e.g. ``8X800g``) are *kept* because they carry meaning.
"""

from __future__ import annotations

import re

# Pack multiplier, e.g. "8X800g" -> 8, "6 x 2" -> 6.
_MULTIPLIER_RE = re.compile(r"(\d+)\s*[xX]\s*\d+")

# Noise suffixes that frequently appear in SAP descriptions and should not
# influence matching. Kept conservative on purpose.
_SUFFIX_NOISE = [
    "perf ripe",
    "f/w",
    "dated",
    "dtd",
    "mid-tier",
    "mid tier",
    "h/s",
    "h/r",
    "promo",
]


def collapse_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


def normalize(value: str) -> str:
    """Normalize a description for fuzzy matching.

    - lowercase
    - unify x/X separators inside pack sizes (``8 X 800`` -> ``8x800``)
    - drop common suffix noise
    - strip punctuation that is safe to remove
    - keep digits (pack sizes matter)
    - collapse repeated whitespace
    """
    text = (value or "").lower()

    # Normalize the multiplier separator so "8 x 800" == "8x800".
    text = re.sub(r"(\d+)\s*x\s*(\d+)", r"\1x\2", text)

    for noise in _SUFFIX_NOISE:
        text = text.replace(noise, " ")

    # Remove bracketed clutter like "(dated)".
    text = re.sub(r"\([^)]*\)", " ", text)

    # Keep letters, digits, the x separator and spaces; drop other punctuation.
    text = re.sub(r"[^a-z0-9x ]+", " ", text)

    return collapse_spaces(text)


def first_multiplier(*descriptions: str) -> int | None:
    """Return the first pack multiplier found across the given descriptions.

    Tries each description in order (typically matched description first, then
    the raw SAP description) and returns the leading multiplier, e.g. for
    ``GRAPE SELECTION PACK 8X800g`` it returns ``8``.
    """
    for desc in descriptions:
        if not desc:
            continue
        m = _MULTIPLIER_RE.search(desc)
        if m:
            return int(m.group(1))
    return None


def last4(job_number: str) -> str:
    """Last 4 characters of a job/prod-order number (used for header suffixes)."""
    digits = re.sub(r"\D", "", str(job_number or ""))
    return digits[-4:] if digits else ""


# A SAP Prod.Order is a 6-9 digit number (real ones are typically 7-8, e.g.
# 1683047). We accept this range and reject anything else so OCR noise can never
# become a fake Job Number or a fake header suffix.
_PROD_ORDER_RE = re.compile(r"^\d{6,9}$")


def valid_prod_order(value: str) -> str:
    """Return the digits of ``value`` iff it looks like a real SAP Prod.Order.

    Job Number and header suffixes must come ONLY from Prod.Order, never from
    Sequence or OCR garbage. A value that is not 6-9 digits long returns ""
    (i.e. "no Prod.Order"), which leaves the Job Number blank and adds no suffix.
    """
    digits = re.sub(r"\D", "", str(value or ""))
    return digits if _PROD_ORDER_RE.fullmatch(digits) else ""

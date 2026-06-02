"""Core business logic: match, detect punnets, compute quantities, merge jobs.

Produces :class:`JobSheet` objects — one per PD080 sheet — ready for the review
page and the PDF generator.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from .lines import normalize_line
from .matcher import match_description
from .parser import RawRow
from .text_utils import collapse_spaces, first_multiplier, last4, normalize, valid_prod_order

TRACE_CODE = "47/"  # MVP: always 47/. No trace-code database yet.
PUNNET_BUFFER = 300  # added on top of multiplier * total qty

# Act.Qty values above this are treated as OCR failures (a number from another
# column — For.Qty/Cust.Qty/Yield/time — bleeding into Act.Qty). Such rows are
# flagged needs_review and never auto-included. Tune for your real run sizes.
ACT_QTY_MAX = 20000

# A SAP Text column triggers punnets if it mentions both a "loose/change"
# action and "punnet". Kept loose on purpose to survive OCR noise; this covers
# "Loose to Punnet(s)", "Change Punnet(s)" and "Change to Punnet(s)".
_PUNNET_ACTIONS = ("loose", "change")


def requires_punnets(text: str) -> bool:
    low = (text or "").lower()
    return "punnet" in low and any(action in low for action in _PUNNET_ACTIONS)


@dataclass
class JobSheet:
    id: str

    # Provenance / preview.
    raw_description: str
    work_center: str
    line: str
    text: str
    status: str  # "matched" | "needs_review"
    review_reason: str
    include: bool

    # Main product line.
    product: str
    job_number: str  # main job (largest Act.Qty)
    act_qty: float

    # Punnet line (optional).
    needs_punnets: bool
    punnet_material: str
    punnet_qty: int | None
    multiplier: int | None

    # Merge bookkeeping.
    additional_jobs: list[str] = field(default_factory=list)
    traceability: str = TRACE_CODE

    # Review bookkeeping.
    was_matched: bool = True  # confident fuzzy match at processing time
    confirmed: bool = False  # user ticked "confirm" for an uncertain match

    def revalidate(self) -> bool:
        """Recompute status from current (possibly edited) field values.

        Returns True if the sheet is safe to generate. Never auto-clears an
        uncertain fuzzy match unless the user explicitly confirmed it.
        """
        reasons: list[str] = []
        if not (self.product or "").strip():
            reasons.append("Missing product")
        # Job Number (Prod.Order) may legitimately be blank — not a blocker.
        # Act.Qty == 0 is a valid business case (operator-confirmed on review);
        # only a negative or implausibly large value is wrong.
        if self.act_qty < 0:
            reasons.append("Act.Qty is negative")
        elif self.act_qty > ACT_QTY_MAX:
            reasons.append(
                f"Act.Qty looks implausible (> {ACT_QTY_MAX}) — likely OCR error, verify"
            )
        # Punnet checks only matter when there's a quantity to pack (qty > 0).
        if self.needs_punnets and self.act_qty > 0 and not (self.punnet_material or "").strip():
            reasons.append("Punnet material missing")
        if self.needs_punnets and self.act_qty > 0 and self.punnet_qty is None:
            reasons.append("Punnet qty missing")
        if not self.was_matched and not self.confirmed:
            reasons.append("Unconfirmed fuzzy match — tick Confirm or exclude")

        self.review_reason = "; ".join(reasons)
        self.status = "needs_review" if reasons else "matched"
        return not reasons

    @property
    def header_suffix(self) -> str:
        """e.g. ' /3086 /3090' appended after the PD080 title."""
        parts = [f"/{last4(j)}" for j in self.additional_jobs if last4(j)]
        return (" " + " ".join(parts)) if parts else ""

    @property
    def title(self) -> str:
        return "Packaging Sign Out & In Record" + self.header_suffix

    def product_lines(self) -> list[dict]:
        """Rows to write into the PD080 grid (main + optional punnet)."""
        rows = [
            {
                "line": self.line,
                "product": self.product,
                "trace": self.traceability,
                "job": self.job_number,
                "qty": "",  # main row qty stays empty
            }
        ]
        if self.needs_punnets and self.punnet_material:
            rows.append(
                {
                    "line": self.line,
                    "product": self.punnet_material,
                    "trace": self.traceability,
                    "job": self.job_number,
                    "qty": "" if self.punnet_qty is None else str(self.punnet_qty),
                }
            )
        return rows


def _merge_key(line: str, product_norm: str, needs_punnets: bool) -> tuple:
    return (line, product_norm, "punnet" if needs_punnets else "standard")


def process_rows(raw_rows: list[RawRow]) -> list[JobSheet]:
    """Match, enrich and merge raw SAP rows into PD080 job sheets."""

    # 1) Enrich each raw row with match + punnet info.
    enriched = []
    for row in raw_rows:
        result = match_description(row.material_description)
        rule = result.rule if result.matched else None

        product = rule.using_name if rule else collapse_spaces(row.material_description)
        product_norm = normalize(product) or normalize(row.material_description)
        needs_pun = requires_punnets(row.text)
        line = normalize_line(row.work_center)
        mult = first_multiplier(
            rule.material_description if rule else "", row.material_description
        )
        punnet_material = (rule.punnets if rule else None) or ""

        enriched.append(
            {
                "row": row,
                "matched": result.matched,
                "score": result.score,
                "rule": rule,
                "product": product,
                "product_norm": product_norm,
                "needs_pun": needs_pun,
                "line": line,
                "mult": mult,
                "punnet_material": punnet_material,
            }
        )

    # 2) Merge rows that represent the same job/product/line/type.
    groups: dict[tuple, list[dict]] = {}
    order: list[tuple] = []
    for e in enriched:
        key = _merge_key(e["line"], e["product_norm"], e["needs_pun"])
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(e)

    sheets: list[JobSheet] = []
    for key in order:
        members = groups[key]
        total_qty = sum(m["row"].act_qty for m in members)

        # Main job = Prod.Order of the row with the largest Act.Qty.
        # Job Number comes ONLY from Prod.Order. Never substitute Sequence or any
        # other field; a value that isn't a real Prod.Order (or is missing) leaves
        # the job number blank.
        main = max(members, key=lambda m: m["row"].act_qty)
        main_job = valid_prod_order(main["row"].prod_order)

        # Additional job numbers must not be lost (distinct, order preserved).
        # Only genuine Prod.Order values become suffixes — no fake suffixes from
        # OCR noise or Sequence numbers.
        additional: list[str] = []
        for m in members:
            j = valid_prod_order(m["row"].prod_order)
            if j and j != main_job and j not in additional:
                additional.append(j)

        needs_pun = main["needs_pun"]
        mult = next((m["mult"] for m in members if m["mult"]), None)
        punnet_material = next(
            (m["punnet_material"] for m in members if m["punnet_material"]), ""
        )

        # Punnets are only quantified when there's a real quantity to pack.
        # Act.Qty == 0 is a valid case (job not produced yet): keep the punnet
        # qty blank — never default it to the buffer.
        punnet_qty: int | None = None
        if needs_pun and mult is not None and total_qty > 0:
            punnet_qty = int(mult * total_qty + PUNNET_BUFFER)

        # 3) Decide status. Never guess silently.
        any_unreadable = any(not m["row"].act_qty_ok for m in members)
        max_member_qty = max((m["row"].act_qty for m in members), default=0)

        reasons: list[str] = []
        if not main["matched"]:
            reasons.append(
                f"No confident match (best {main['score']:.0f}%) — check product"
            )
        # A missing Prod.Order is allowed: Job Number is simply left blank.
        # Act.Qty rules: unreadable -> review; an explicit 0 is acceptable.
        if any_unreadable:
            reasons.append("Act.Qty could not be read — verify")
        elif max_member_qty > ACT_QTY_MAX:
            # OCR column bleed: an impossibly large Act.Qty means another
            # column's value was read. Don't use it.
            reasons.append(
                f"Act.Qty looks implausible (> {ACT_QTY_MAX}) — likely OCR error, verify"
            )
        # Punnet checks only matter when there's a quantity to pack (qty > 0).
        if needs_pun and total_qty > 0 and not punnet_material:
            reasons.append("Punnets required but no punnet material in rules")
        if needs_pun and total_qty > 0 and mult is None:
            reasons.append("Punnets required but no pack multiplier found")

        status = "needs_review" if reasons else "matched"

        sheets.append(
            JobSheet(
                id=uuid.uuid4().hex[:8],
                raw_description=collapse_spaces(main["row"].material_description),
                work_center=main["row"].work_center,
                line=main["line"],
                text=collapse_spaces(main["row"].text),
                status=status,
                review_reason="; ".join(reasons),
                include=(status == "matched"),
                product=main["product"],
                job_number=main_job,
                act_qty=total_qty,
                needs_punnets=needs_pun,
                punnet_material=punnet_material,
                punnet_qty=punnet_qty,
                multiplier=mult,
                additional_jobs=additional,
                was_matched=main["matched"],
            )
        )

    return sheets

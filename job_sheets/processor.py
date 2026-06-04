"""Core business logic: match, detect punnets, compute quantities, merge jobs.

Produces :class:`JobSheet` objects — one per PD080 sheet — ready for the review
page and the PDF generator.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from .giro_rules import match_giro
from .lines import line_kind, normalize_line
from .matcher import match_description
from .parser import RawRow
from .text_utils import collapse_spaces, first_multiplier, last4, normalize, valid_prod_order

TRACE_CODE = "47/"  # MVP: always 47/. No trace-code database yet.
PUNNET_BUFFER = 300  # added on top of multiplier * total qty

# PD080 template data rows (see resources/templates/PD080.xlsx and the
# resources/example/*.xlsx golden files). The grid is rows 5-21; the footer
# starts at row 22. Layout differs per line family:
#   standard (TOP SEAL / FLOW / LINERLESS): main row 5, optional punnet row 12.
#   MPACK: the product name repeated on rows 5 and 6.
#   GIRO: Front on row 5, Back on row 12, Net on the penultimate row 19.
MAIN_ROW = 5
PUNNET_ROW = MAIN_ROW + 7  # 12 — 7-row gap leaves rows 6-11 blank for additions
MPACK_SECOND_ROW = 6
GIRO_ROWS = (5, 12, 19)

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

    # Layout family: "standard" (TOP SEAL / FLOW / LINERLESS) | "giro" | "mpack".
    kind: str = "standard"

    # GIRO-only values, taken verbatim from the GIRO rules file (never guessed
    # from the product description). ``giro_net`` is display-ready, e.g.
    # "Green Net".
    giro_front: str = ""
    giro_back: str = ""
    giro_net: str = ""

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

    def sheet_rows(self) -> list[dict]:
        """Rows to write into the PD080 grid, each pinned to a template row.

        This is the single source of truth for *where* each value lands, so the
        generator stays a dumb writer. The dicts carry the same keys as
        :meth:`product_lines` plus ``"row"`` (the 1-based template row number).
        """

        def make(row: int, product: str, qty: str = "") -> dict:
            return {
                "row": row,
                "line": self.line,
                "product": product,
                "trace": self.traceability,
                "job": self.job_number,
                "qty": qty,
            }

        if self.kind == "giro":
            front, back, net = GIRO_ROWS
            return [
                make(front, self.giro_front),
                make(back, self.giro_back),
                make(net, self.giro_net),
            ]
        if self.kind == "mpack":
            # The job/product name repeated on the first two template rows.
            return [make(MAIN_ROW, self.product), make(MPACK_SECOND_ROW, self.product)]

        # Standard TOP SEAL / FLOW / LINERLESS: main row + optional punnet row.
        rows = [make(MAIN_ROW, self.product)]
        if self.needs_punnets and self.punnet_material:
            rows.append(
                make(
                    PUNNET_ROW,
                    self.punnet_material,
                    "" if self.punnet_qty is None else str(self.punnet_qty),
                )
            )
        return rows


def _merge_key(kind: str, line: str, product_norm: str, needs_punnets: bool) -> tuple:
    return (kind, line, product_norm, "punnet" if needs_punnets else "standard")


# Special-case combine: the 230g and 140g Fig rows are prepared as a single
# PD080 ("Figs 230/140"). Both rules carry this Using Name; they are told apart
# by the pack size in the matched rule's Material Description.
FIG_COMBINED_NAME = "Figs 230/140"


def _is_fig_combined(enriched: dict) -> bool:
    rule = enriched.get("rule")
    return rule is not None and rule.using_name.strip().lower() == FIG_COMBINED_NAME.lower()


def _find_fig_pair(enriched_rows: list[dict]) -> tuple[dict | None, dict | None]:
    """Return (fig_230, fig_140) enriched rows if both are present, else Nones."""
    fig_230 = fig_140 = None
    for e in enriched_rows:
        if not _is_fig_combined(e):
            continue
        desc = (e["rule"].material_description or "")
        if "230" in desc and fig_230 is None:
            fig_230 = e
        elif "140" in desc and fig_140 is None:
            fig_140 = e
    return fig_230, fig_140


def _make_fig_sheet(fig_230: dict, fig_140: dict) -> JobSheet:
    """Build the combined Figs 230/140 sheet.

    - Job number: from the 230g row's Prod.Order.
    - Suffix: from the 140g row's Prod.Order.
    - Punnet qty: from the 230g row's Act.Qty ONLY (140g qty is ignored).
    - Line / work center / text: from the first Fig row in SAP order.
    """
    first = fig_230 if fig_230["idx"] <= fig_140["idx"] else fig_140

    main_job = valid_prod_order(fig_230["row"].prod_order)
    suffix_job = valid_prod_order(fig_140["row"].prod_order)
    additional = [suffix_job] if suffix_job and suffix_job != main_job else []

    # Punnet maths uses the 230g row exclusively.
    qty_230 = fig_230["row"].act_qty
    needs_pun = fig_230["needs_pun"]
    mult = fig_230["mult"]
    punnet_material = fig_230["punnet_material"]
    punnet_qty: int | None = None
    if needs_pun and mult is not None and qty_230 > 0:
        punnet_qty = int(mult * qty_230 + PUNNET_BUFFER)

    reasons: list[str] = []
    if not (fig_230["matched"] and fig_140["matched"]):
        reasons.append("Fig 230/140 match not confident — check product")
    if not fig_230["row"].act_qty_ok:
        reasons.append("230g Act.Qty could not be read — verify")
    if needs_pun and qty_230 > 0 and not punnet_material:
        reasons.append("Punnets required but no punnet material in rules")
    if needs_pun and qty_230 > 0 and mult is None:
        reasons.append("Punnets required but no pack multiplier found")
    status = "needs_review" if reasons else "matched"

    return JobSheet(
        id=uuid.uuid4().hex[:8],
        raw_description=(
            f'{collapse_spaces(fig_230["row"].material_description)} + '
            f'{collapse_spaces(fig_140["row"].material_description)}'
        ),
        work_center=first["row"].work_center,
        line=first["line"],
        text=collapse_spaces(first["row"].text),
        status=status,
        review_reason="; ".join(reasons),
        include=(status == "matched"),
        product=FIG_COMBINED_NAME,
        job_number=main_job,
        act_qty=qty_230,  # 230g drives the sheet (and punnet calc)
        needs_punnets=needs_pun,
        punnet_material=punnet_material,
        punnet_qty=punnet_qty,
        multiplier=mult,
        additional_jobs=additional,
        was_matched=fig_230["matched"] and fig_140["matched"],
    )


def process_rows(raw_rows: list[RawRow]) -> list[JobSheet]:
    """Match, enrich and merge raw SAP rows into PD080 job sheets."""

    # 1) Enrich each raw row with match + punnet info.
    enriched = []
    for idx, row in enumerate(raw_rows):
        kind = line_kind(row.work_center)
        line = normalize_line(row.work_center)

        # Defaults for the punnet machinery and GIRO Front/Back/Net. GIRO and
        # MPACK never carry punnets; GIRO supplies its three values from its
        # own rules file (the source of truth — never guessed from the SAP
        # description).
        needs_pun = False
        mult = None
        punnet_material = ""
        giro_front = giro_back = giro_net = ""

        if kind == "giro":
            gm = match_giro(row.material_description)
            rule = gm.rule
            matched = gm.matched
            score = gm.score
            product = (rule.using_name if rule else "") or collapse_spaces(
                row.material_description
            )
            if rule:
                giro_front = rule.front
                giro_back = rule.back
                giro_net = f"{rule.net} Net" if rule.net else ""
        else:
            result = match_description(row.material_description)
            rule = result.rule if result.matched else None
            matched = result.matched
            score = result.score
            product = (
                rule.using_name if rule else collapse_spaces(row.material_description)
            )
            if kind == "standard":
                # MPACK is intentionally simple (product name only); the punnet
                # machinery applies to standard TOP SEAL / FLOW / LINERLESS rows.
                needs_pun = requires_punnets(row.text)
                mult = first_multiplier(
                    rule.material_description if rule else "", row.material_description
                )
                punnet_material = (rule.punnets if rule else None) or ""

        product_norm = normalize(product) or normalize(row.material_description)

        enriched.append(
            {
                "idx": idx,  # original SAP order, used to keep output ordering
                "row": row,
                "kind": kind,
                "matched": matched,
                "score": score,
                "rule": rule,
                "product": product,
                "product_norm": product_norm,
                "needs_pun": needs_pun,
                "line": line,
                "mult": mult,
                "punnet_material": punnet_material,
                "giro_front": giro_front,
                "giro_back": giro_back,
                "giro_net": giro_net,
            }
        )

    # Special-case: combine the 230g + 140g Fig rows into one sheet (when both
    # are present). Those rows are then excluded from the generic merge below.
    fig_230, fig_140 = _find_fig_pair(enriched)
    fig_merge = fig_230 is not None and fig_140 is not None
    fig_excluded = {id(fig_230), id(fig_140)} if fig_merge else set()

    # 2) Merge rows that represent the same job/product/line/type.
    groups: dict[tuple, list[dict]] = {}
    order: list[tuple] = []
    for e in enriched:
        if id(e) in fig_excluded:
            continue
        key = _merge_key(e["kind"], e["line"], e["product_norm"], e["needs_pun"])
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(e)

    # Collect (sort_index, sheet) so the combined Fig sheet can be slotted back
    # into SAP order by its first source row.
    built: list[tuple[int, JobSheet]] = []
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

        kind = main["kind"]
        reasons: list[str] = []
        if not main["matched"]:
            where = "GIRO rule" if kind == "giro" else "match"
            reasons.append(
                f"No confident {where} (best {main['score']:.0f}%) — check product"
            )
        elif kind == "giro" and not (
            main["giro_front"] and main["giro_back"] and main["giro_net"]
        ):
            reasons.append("GIRO rule is missing a Front / Back / Net value — verify")
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

        sort_index = min(m["idx"] for m in members)
        built.append(
            (
                sort_index,
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
                    kind=kind,
                    giro_front=main["giro_front"],
                    giro_back=main["giro_back"],
                    giro_net=main["giro_net"],
                    needs_punnets=needs_pun,
                    punnet_material=punnet_material,
                    punnet_qty=punnet_qty,
                    multiplier=mult,
                    additional_jobs=additional,
                    was_matched=main["matched"],
                ),
            )
        )

    # Slot the combined Fig sheet in at its first source row's position.
    if fig_merge:
        built.append(
            (min(fig_230["idx"], fig_140["idx"]), _make_fig_sheet(fig_230, fig_140))
        )

    built.sort(key=lambda pair: pair[0])
    return [sheet for _, sheet in built]

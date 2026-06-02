"""Local OCR for SAP production-plan screenshots.

Uses Tesseract (via pytesseract) with Pillow + OpenCV preprocessing. No external
APIs. The table is reconstructed from word-level bounding boxes
(``image_to_data``): words are grouped into rows by Tesseract's own line
grouping, the header row fixes the column x-positions, and every other word is
binned into a column by its horizontal centre. This is far more robust for a
fixed-layout table than splitting flat OCR text on whitespace.

If the Tesseract executable is not installed, :func:`reconstruct_table` raises
:class:`TesseractNotInstalled` with a clear, user-facing message.
"""

from __future__ import annotations

import io
import os
import shutil
import statistics
from dataclasses import dataclass

import pytesseract
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

try:  # OpenCV is optional; we degrade to Pillow-only preprocessing without it.
    import cv2
    import numpy as np

    _HAS_CV2 = True
except Exception:  # pragma: no cover
    _HAS_CV2 = False

TESSERACT_MISSING_MSG = (
    "Tesseract OCR is not installed. Please install it or paste SAP table text."
)

# Mean word confidence (0-100) below which we treat the OCR as unreliable and
# ask the user to review/correct the extracted text instead of auto-processing.
CONF_THRESHOLD = 55.0

# Canonical output column order (matches the pasted-text parser's FIELD_ORDER).
# These are the ONLY columns we extract. Everything else (Material, Sequence,
# For.Qty, Cust.Qty, Yield, dates/times, ...) is ignored — though we still
# detect those headers, purely as boundary anchors (see _classify_header_cell).
COLUMNS = ["material_description", "text", "work_center", "prod_order", "act_qty"]
WANTED = set(COLUMNS)
COLUMN_LABELS = {
    "material_description": "Material Description",
    "text": "Text",
    "work_center": "Work Center",
    "prod_order": "Prod. Order",
    "act_qty": "Act.Qty",
}


class TesseractNotInstalled(RuntimeError):
    def __init__(self, message: str = TESSERACT_MISSING_MSG):
        super().__init__(message)


@dataclass
class OCRResult:
    text: str  # reconstructed tab-separated table (with header)
    mean_conf: float  # 0-100
    n_rows: int

    @property
    def low_confidence(self) -> bool:
        return self.mean_conf < CONF_THRESHOLD


# --------------------------------------------------------------------------- #
# Tesseract discovery
# --------------------------------------------------------------------------- #
def _candidate_paths() -> list[str]:
    paths = [
        os.environ.get("TESSERACT_CMD", ""),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
        shutil.which("tesseract") or "",
    ]
    return [p for p in paths if p]


def configure_tesseract() -> bool:
    """Point pytesseract at a Tesseract binary if one can be found."""
    for path in _candidate_paths():
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            return True
    # Maybe it is already resolvable on PATH under the default name.
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def tesseract_available() -> bool:
    return configure_tesseract()


# --------------------------------------------------------------------------- #
# Image preprocessing
# --------------------------------------------------------------------------- #
def preprocess(image_bytes: bytes, scale: int = 3) -> Image.Image:
    """Grayscale, upscale and sharpen the screenshot for better OCR."""
    img = Image.open(io.BytesIO(image_bytes))
    # Honour EXIF orientation, drop alpha.
    img = ImageOps.exif_transpose(img).convert("RGB")

    if _HAS_CV2:
        arr = np.array(img)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(
            gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )
        # Otsu binarisation copes well with clean UI screenshots.
        _, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return Image.fromarray(thresh)

    # Pillow-only fallback.
    gray = ImageOps.grayscale(img)
    gray = gray.resize((gray.width * scale, gray.height * scale), Image.LANCZOS)
    gray = ImageOps.autocontrast(gray)
    gray = ImageEnhance.Contrast(gray).enhance(1.5)
    gray = gray.filter(ImageFilter.SHARPEN)
    return gray


# --------------------------------------------------------------------------- #
# Table reconstruction (pure logic, unit-testable without Tesseract)
# --------------------------------------------------------------------------- #
def _classify_header_word(text: str) -> str | None:
    """Classify a single header *word* into a SAP column key.

    Returns a *wanted* key (one of ``COLUMNS``); an *ignored-but-anchor* key
    (prefixed ``_``), used only to bound the wanted columns so e.g. Act.Qty
    cannot absorb the neighbouring For.Qty/Sequence/Unt numbers; or ``None``.

    Matching is deliberately tolerant of OCR damage seen in real SAP captures,
    e.g. "Description" -> "Descnption" (matched by the ``desc`` prefix) and
    "Prod. Order" split into "Prod." + "Orde".
    """
    t = text.lower().strip()
    # Material Description — match both "Material" and the (often mangled)
    # "Description"/"Descnption". A standalone "Material" column is split off
    # later in _build_anchors using word geometry.
    if "desc" in t or t.startswith("material"):
        return "material_description"
    if "version" in t:
        return "_version"
    if t.startswith("text"):
        return "text"
    if "work" in t or "cent" in t:
        return "work_center"
    if "prod" in t or "orde" in t:
        return "prod_order"
    if "seq" in t:
        return "_sequence"
    if "act" in t:  # Act.Qty — checked before the generic *.Qty columns
        return "act_qty"
    if "for" in t and "qty" in t:
        return "_for_qty"
    if "cust" in t:
        return "_cust_qty"
    if "yield" in t:
        return "_yield"
    if "unt" in t or "unit" in t:
        return "_unit"
    if any(k in t for k in ("date", "time", "plan", "basic", "finish", "start", "sched")):
        return "_date"
    return None


def _cx(w: dict) -> float:
    return w["left"] + w.get("width", 0) / 2.0


def _cy(w: dict) -> float:
    return w["top"] + w.get("height", 0) / 2.0


def _cluster_rows(words: list[dict], threshold: float) -> list[list[dict]]:
    """Group words into visual rows by their vertical centre.

    A new row starts when a word's centre is more than ``threshold`` below the
    running mean centre of the current row. This reconstructs every table row
    independently of Tesseract's own (often unreliable) line grouping, so dense
    tables with many rows are not silently merged or dropped.
    """
    ordered = sorted(words, key=_cy)
    rows: list[list[dict]] = []
    current = [ordered[0]]
    current_cy = _cy(ordered[0])
    for w in ordered[1:]:
        cy = _cy(w)
        if cy - current_cy > threshold:
            rows.append(current)
            current = [w]
            current_cy = cy
        else:
            current.append(w)
            current_cy = sum(_cy(x) for x in current) / len(current)
    rows.append(current)
    return rows


def _build_anchors(header_words: list[dict], mh: float) -> dict[str, int]:
    """Map each detected column to its header LEFT edge (the column boundary).

    SAP plan columns are left-aligned, so a column spans from its own header's
    left edge to the next column's left edge. Using left edges (not label
    centres) keeps wide left-aligned text columns from being clipped by a short
    header label.

    A standalone "Material" (number) column is separated from "Material
    Description": the "Material" word(s) sitting far to the LEFT of the
    "Description" word become the ignored ``_material`` anchor; an adjacent
    "Material" (part of the "Material Description" label) stays with it.
    """
    classified: list[tuple[dict, str]] = []
    for w in header_words:
        key = _classify_header_word(w["text"])
        if key:
            classified.append((w, key))

    by_key: dict[str, list[dict]] = {}
    for w, key in classified:
        by_key.setdefault(key, []).append(w)

    anchors: dict[str, int] = {}

    md_words = by_key.pop("material_description", [])
    if md_words:
        desc = [w for w in md_words if "desc" in w["text"].lower()]
        if desc:
            desc_left = min(w["left"] for w in desc)
            near, far = [], []
            for w in md_words:
                if "desc" in w["text"].lower():
                    near.append(w)
                elif w["left"] + w.get("width", 0) >= desc_left - 1.5 * mh:
                    near.append(w)  # "Material" touching "Description" — same label
                else:
                    far.append(w)  # a distinct, ignored Material(-number) column
            anchors["material_description"] = min(w["left"] for w in near)
            if far:
                anchors["_material"] = min(w["left"] for w in far)
        else:
            anchors["material_description"] = min(w["left"] for w in md_words)

    for key, ws in by_key.items():
        anchors[key] = min(w["left"] for w in ws)

    return anchors


def reconstruct_from_words(words: list[dict]) -> OCRResult | None:
    """Rebuild the SAP table from word boxes using fixed column positions.

    Each ``word`` dict must have ``text``, ``left``, ``top``, ``width``,
    ``height`` and ``conf`` (0-100, -1 for non-text).

    Strategy (robust for a fixed-layout table with many rows):
    1. Cluster all words into visual rows by their vertical centre — every row
       is reconstructed independently, so dense tables aren't dropped/merged.
    2. Find the header row (the one classifying into the most wanted columns)
       and read each column's LEFT edge — including the ignored neighbour
       columns (Material, Version, Sequence, For.Qty, Unt, ...) as anchors.
    3. Bin every data word into a column by its x-centre against the column
       left-edge boundaries, then keep only the 5 wanted columns. Because
       Act.Qty is bounded on the left by Unt/For.Qty, big values from those
       columns can never leak into it.

    Returns ``None`` if the essential headers (Material Description, Work Center,
    Prod. Order) cannot be located, so the caller can fall back to raw text.
    """
    words = [w for w in words if (w.get("text") or "").strip()]
    if not words:
        return None

    heights = [w.get("height", 0) for w in words if w.get("height", 0) > 0]
    mh = statistics.median(heights) if heights else 12.0

    rows = _cluster_rows(words, threshold=0.6 * mh)

    # Header = the row whose words classify into the most wanted columns.
    best: tuple | None = None  # (n_wanted, idx)
    for i, row in enumerate(rows):
        wanted_keys = {
            k
            for w in row
            if (k := _classify_header_word(w["text"])) in WANTED
        }
        if best is None or len(wanted_keys) > best[0]:
            best = (len(wanted_keys), i)

    if best is None:
        return None
    header_idx = best[1]

    anchors = _build_anchors(rows[header_idx], mh)

    # Need the essentials to bin (and to filter by) reliably.
    if not {"material_description", "work_center", "prod_order"} <= anchors.keys():
        return None

    # Columns sorted left-to-right; a column spans [left_i, left_{i+1}).
    ordered = sorted(anchors.items(), key=lambda kv: kv[1])
    keys = [k for k, _ in ordered]
    boundaries = [left for _, left in ordered][1:]  # left edges of columns 1..n-1

    def _col_of(x: float) -> str:
        idx = 0
        for j, edge in enumerate(boundaries):
            if x >= edge:
                idx = j + 1
            else:
                break
        return keys[idx]

    header_top = min(w["top"] for w in rows[header_idx])

    out_rows: list[list[str]] = []
    confs: list[float] = []
    for i, row in enumerate(rows):
        if i == header_idx:
            continue
        if min(w["top"] for w in row) <= header_top:
            continue  # title / anything above the header

        cells: dict[str, list[str]] = {c: [] for c in COLUMNS}
        for w in sorted(row, key=lambda x: x["left"]):
            key = _col_of(_cx(w))
            if key in WANTED:
                cells[key].append(w["text"])
                if w.get("conf", -1) >= 0:
                    confs.append(float(w["conf"]))

        values = [" ".join(cells[c]).strip() for c in COLUMNS]
        if any(values):
            out_rows.append(values)

    if not out_rows:
        return None

    header_line = "\t".join(COLUMN_LABELS[c] for c in COLUMNS)
    body = "\n".join("\t".join(r) for r in out_rows)
    mean_conf = sum(confs) / len(confs) if confs else 0.0
    return OCRResult(text=f"{header_line}\n{body}", mean_conf=mean_conf, n_rows=len(out_rows))


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #
def _words_from_image(img: Image.Image) -> list[dict]:
    from pytesseract import Output

    data = pytesseract.image_to_data(img, config="--psm 6", output_type=Output.DICT)
    words = []
    n = len(data["text"])
    for i in range(n):
        words.append(
            {
                "text": data["text"][i],
                "left": int(data["left"][i]),
                "top": int(data["top"][i]),
                "width": int(data["width"][i]),
                "height": int(data["height"][i]),
                "conf": float(data["conf"][i]),
            }
        )
    return words


def reconstruct_table(image_bytes: bytes) -> OCRResult | None:
    """OCR a screenshot and rebuild the SAP table.

    Returns an :class:`OCRResult`, or ``None`` if the table layout could not be
    detected (caller should fall back to :func:`raw_text`). Raises
    :class:`TesseractNotInstalled` if the executable is missing.
    """
    if not configure_tesseract():
        raise TesseractNotInstalled()
    img = preprocess(image_bytes)
    return reconstruct_from_words(_words_from_image(img))


def raw_text(image_bytes: bytes) -> str:
    """Plain OCR text, used as the manual-correction fallback."""
    if not configure_tesseract():
        raise TesseractNotInstalled()
    return pytesseract.image_to_string(preprocess(image_bytes), config="--psm 6")

"""FastAPI routes for the Job Sheets module.

    GET  /job-sheets                      upload page
    POST /job-sheets/upload               parse + process -> review
    GET  /job-sheets/review/{batch_id}    preview / edit table
    POST /job-sheets/generate/{batch_id}  apply edits -> combined PD080 PDF (or xlsx)

Auth reuses the existing ``operator`` cookie convention from the core app.
"""

from __future__ import annotations

import io

from fastapi import APIRouter, Cookie, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from . import ocr, parser, pd080_generator, store
from .processor import process_rows

router = APIRouter(prefix="/job-sheets", tags=["job-sheets"])
templates = Jinja2Templates(directory="templates")


def _require_operator(operator: str | None):
    """Return None if logged in, else a redirect response to /login."""
    if not operator:
        return RedirectResponse("/login", status_code=303)
    return None


@router.get("", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
def upload_page(request: Request, operator: str | None = Cookie(default=None)):
    if (redir := _require_operator(operator)) is not None:
        return redir
    return templates.TemplateResponse(
        "job_sheets/upload.html",
        {"request": request, "error": "", "sap_text": "", "notice": ""},
    )


@router.post("/upload", response_class=HTMLResponse)
async def upload_submit(
    request: Request,
    sap_text: str = Form(""),
    screenshot: UploadFile | None = File(None),
    operator: str | None = Cookie(default=None),
):
    if (redir := _require_operator(operator)) is not None:
        return redir

    def render_upload(error: str = "", text: str = "", notice: str = ""):
        return templates.TemplateResponse(
            "job_sheets/upload.html",
            {"request": request, "error": error, "sap_text": text, "notice": notice},
        )

    # ---- Screenshot path: OCR -> reconstructed table text -------------------
    # On success with good confidence we go straight to review. On low
    # confidence (or undetectable layout) we drop the extracted text into the
    # textarea so the user can correct it and re-submit via the paste path.
    if screenshot is not None and screenshot.filename:
        content = await screenshot.read()
        if not content:
            return render_upload(error="The uploaded screenshot was empty.", text=sap_text)

        if not ocr.tesseract_available():
            return render_upload(error=ocr.TESSERACT_MISSING_MSG, text=sap_text)

        try:
            result = ocr.reconstruct_table(content)
        except ocr.TesseractNotInstalled:
            return render_upload(error=ocr.TESSERACT_MISSING_MSG, text=sap_text)
        except Exception:
            # Any OCR failure: try plain text so the user can fix it manually.
            try:
                fallback = ocr.raw_text(content)
            except Exception:
                fallback = sap_text
            return render_upload(
                notice="Couldn't read the screenshot automatically. "
                "Please correct the text below and press Process.",
                text=fallback,
            )

        if result is None:
            try:
                fallback = ocr.raw_text(content)
            except Exception:
                fallback = sap_text
            return render_upload(
                notice="Couldn't detect the SAP table layout in the screenshot. "
                "Please check/correct the text below and press Process.",
                text=fallback,
            )

        rows = parser.parse_pasted_rows(result.text)
        if not rows or result.low_confidence:
            return render_upload(
                notice=(
                    f"OCR confidence is {result.mean_conf:.0f}% — please review and "
                    "correct the extracted text below, then press Process."
                    if result.low_confidence
                    else "No in-scope production lines were detected. Please check "
                    "the extracted text below (and the Work Center column)."
                ),
                text=result.text,
            )

        sheets = process_rows(rows)
        batch_id = store.create_batch(sheets)
        return RedirectResponse(f"/job-sheets/review/{batch_id}", status_code=303)

    # ---- Pasted-text path (unchanged) ---------------------------------------
    if not sap_text.strip():
        return render_upload(
            error="Upload a screenshot or paste the SAP table text.", text=sap_text
        )

    raw_rows = parser.parse_pasted_rows(sap_text)
    if not raw_rows:
        return render_upload(
            error="No in-scope production lines found. Paste the SAP table "
            "(include the Work Center column).",
            text=sap_text,
        )

    sheets = process_rows(raw_rows)
    batch_id = store.create_batch(sheets)
    return RedirectResponse(f"/job-sheets/review/{batch_id}", status_code=303)


@router.get("/review/{batch_id}", response_class=HTMLResponse)
def review_page(
    request: Request,
    batch_id: str,
    error: str = "",
    operator: str | None = Cookie(default=None),
):
    if (redir := _require_operator(operator)) is not None:
        return redir

    sheets = store.get_batch(batch_id)
    if sheets is None:
        return RedirectResponse("/job-sheets", status_code=303)

    return templates.TemplateResponse(
        "job_sheets/review.html",
        {
            "request": request,
            "batch_id": batch_id,
            "sheets": sheets,
            "error": error,
            "pdf_ready": pd080_generator.pdf_available(),
        },
    )


def _to_int(value: str):
    value = (value or "").strip()
    if not value:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _to_float(value: str, default: float = 0.0) -> float:
    try:
        return float((value or "").strip())
    except ValueError:
        return default


@router.post("/generate/{batch_id}")
async def generate(
    batch_id: str,
    request: Request,
    operator: str | None = Cookie(default=None),
):
    if (redir := _require_operator(operator)) is not None:
        return redir

    sheets = store.get_batch(batch_id)
    if sheets is None:
        return RedirectResponse("/job-sheets", status_code=303)

    form = await request.form()
    fmt = form.get("fmt", "xlsx")

    # Apply edits back onto the stored sheets.
    for s in sheets:
        s.include = form.get(f"include_{s.id}") is not None
        s.confirmed = form.get(f"confirm_{s.id}") is not None
        s.product = (form.get(f"product_{s.id}") or s.product).strip()
        s.line = (form.get(f"line_{s.id}") or s.line).strip()
        # Job Number comes only from Prod.Order; an empty field stays empty.
        s.job_number = (form.get(f"job_{s.id}") or "").strip()
        s.act_qty = _to_float(form.get(f"act_qty_{s.id}"), s.act_qty)
        s.punnet_material = (form.get(f"punnet_material_{s.id}") or s.punnet_material).strip()
        pq = _to_int(form.get(f"punnet_qty_{s.id}"))
        s.punnet_qty = pq if pq is not None else s.punnet_qty
        s.revalidate()

    store.save_batch(batch_id, sheets)

    def review_error(message: str):
        return templates.TemplateResponse(
            "job_sheets/review.html",
            {
                "request": request,
                "batch_id": batch_id,
                "sheets": sheets,
                "error": message,
                "pdf_ready": pd080_generator.pdf_available(),
            },
        )

    included = [s for s in sheets if s.include]
    blocked = [s for s in included if s.status == "needs_review"]

    if not included:
        return review_error("Select at least one row to generate.")

    if blocked:
        names = ", ".join(s.product or s.raw_description for s in blocked)
        return review_error(
            "Cannot generate while these included rows still need review: "
            f"{names}. Fix the values (and tick Confirm) or exclude them."
        )

    if fmt == "pdf":
        pdf_bytes = pd080_generator.generate_pdf(included)
        if pdf_bytes is None:
            return review_error(
                "PDF conversion isn't available on this server (install LibreOffice "
                "or Excel). The print-ready workbook is available via "
                "“Download workbook (XLSX)”."
            )
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=PD080_job_sheets.pdf"},
        )

    xlsx_bytes = pd080_generator.generate_xlsx(included)
    return StreamingResponse(
        io.BytesIO(xlsx_bytes),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=PD080_job_sheets.xlsx"},
    )

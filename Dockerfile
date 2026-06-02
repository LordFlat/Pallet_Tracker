# Render deployment image for Pallet Tracker.
#
# System packages are installed FIRST (before pip), guaranteeing the binaries
# the app needs at runtime:
#   - tesseract-ocr -> /usr/bin/tesseract   (screenshot OCR via pytesseract)
#   - libreoffice   -> /usr/bin/soffice     (combined PDF generation)
#
# Detection in the app is path-based (shutil.which + /usr/bin fallbacks), so no
# Windows-specific configuration is needed here and local Windows runs are
# unaffected.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=10000

# 1) System dependencies FIRST.
#    libglib2.0-0 is the shared lib opencv-python-headless needs to import.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        libreoffice \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2) Python dependencies.
COPY requirements.txt .
RUN pip install -r requirements.txt

# 3) Application code.
COPY . .

EXPOSE 10000

# Render injects $PORT at runtime; bind to it (shell form so $PORT expands).
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]

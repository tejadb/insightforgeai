FROM python:3.13-slim

# --- Runtime env (better logs, smaller images) ---
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# --- System dependencies required by Unstructured hi_res OCR/PDF pipeline ---
# - tesseract-ocr: OCR engine required by unstructured (TesseractNotFoundError without it)
# - poppler-utils: PDF text extraction helpers (pdftoppm/pdftotext) commonly used by PDF pipelines
# - libmagic1: file type detection (python-magic)
# - libgl1/libglib2.0-0/libsm6/libxext6/libxrender1: common runtime libs for image/PDF tooling
# - ca-certificates: HTTPS
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    libmagic1 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# --- Python deps ---
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# --- App code ---
COPY . /app

# Render sets PORT; default for local docker runs
EXPOSE 8000

# Default command is the API; on Render, set the worker service to override this to:
#   python -m arq worker.WorkerSettings
CMD ["bash", "-lc", "python -m uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]


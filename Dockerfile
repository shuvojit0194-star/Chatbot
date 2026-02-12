# ── Base image ───────────────────────────────────────────────────
FROM python:3.11-slim

# ── System dependencies ──────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ──────────────────────────────────
# Copy requirements first (Docker layer caching — faster rebuilds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application files ───────────────────────────────────────
COPY main.py .
COPY scraped_urls.json .
COPY static/ ./static/

# ── Cloud Run listens on PORT env var (default 8080) ────────────
ENV PORT=8080

# ── Start the server ─────────────────────────────────────────────
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]

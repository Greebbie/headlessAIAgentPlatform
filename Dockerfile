# ── Stage 1: Build frontend ──────────────────────────
FROM node:20-alpine AS frontend

WORKDIR /app/console
COPY console/package*.json ./
RUN npm ci --no-audit --no-fund
COPY console/ ./
RUN npx vite build

# ── Stage 2: Python backend + serve frontend ────────
FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[rag]"

# Pre-download embedding model (~100MB) so first request is fast
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-zh-v1.5')"

# Copy backend source
COPY server/ server/

# Copy built frontend from stage 1
COPY --from=frontend /app/console/dist /app/static

# Create data directories
RUN mkdir -p data/vectors data/uploads data/cache

EXPOSE 8000

CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]

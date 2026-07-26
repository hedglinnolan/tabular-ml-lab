# Tabular ML Lab — Production Docker Image
# Supports: standalone, behind reverse proxy (Shibboleth/CAS/KeyCloak), with optional Ollama
#
# Build:  docker build -t tabular-ml-lab .
# Run:    docker run -p 8501:8501 tabular-ml-lab
# Or use: docker compose up

FROM python:3.12-slim AS builder

WORKDIR /build

# System deps for building wheels (numpy, scipy, torch, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY requirements.txt .

# Install core dependencies
RUN uv pip install --system --link-mode=copy -r requirements.txt

# Install optional dependencies (TDA, UMAP) — non-fatal if they fail
RUN uv pip install --system --link-mode=copy giotto-tda umap-learn 2>/dev/null || \
    echo "Optional: giotto-tda/umap-learn unavailable — TDA and UMAP features disabled"

# ── Production image ──────────────────────────────────────────────
FROM python:3.12-slim

# Non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser -d /app appuser

WORKDIR /app

# System runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=appuser:appuser . .

# Streamlit config for containerized deployment
RUN mkdir -p /app/.streamlit && cat > /app/.streamlit/config.toml <<'EOF'
[server]
port = 8501
address = "0.0.0.0"
headless = true
enableCORS = true
enableXsrfProtection = true
maxUploadSize = 50
maxMessageSize = 50

[browser]
gatherUsageStats = false
serverAddress = "localhost"

[theme]
primaryColor = "#667eea"
backgroundColor = "#f1f5f9"
secondaryBackgroundColor = "#e2e8f0"
textColor = "#0f172a"
font = "sans serif"
EOF

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true"]

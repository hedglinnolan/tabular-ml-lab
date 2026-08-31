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

# Streamlit config for containerized deployment — a DEFAULT, not an override.
#
# `COPY . .` above brings the repo's own `.streamlit/config.toml` into the image
# (`.dockerignore` excludes the docs and the venvs, not `.streamlit/`), and this
# heredoc used to overwrite it unconditionally. That silently discarded the one
# file UNIVERSITY_DEPLOYMENT.md tells a site admin to edit: an institution that
# raised the upload ceiling in their checkout built an image that ignored the
# change, with nothing in the build output to say so. The guard makes this a
# fallback for a tree that carries no `.streamlit/` at all, so an edited config
# now wins.
#
# Two RUN instructions rather than `mkdir -p ... && test -f ... || cat ...`:
# in `sh` that chain writes the default when the *mkdir* fails, which is
# operator precedence quietly deciding a configuration question.
#
# `address` is the only key that differs between the two configs ("0.0.0.0"
# here, "localhost" in the repo). Letting the repo's value win cannot strand the
# container: the ENTRYPOINT passes `--server.address=0.0.0.0` on the command
# line, and a Streamlit CLI flag outranks config.toml.
RUN mkdir -p /app/.streamlit
RUN test -f /app/.streamlit/config.toml || cat > /app/.streamlit/config.toml <<'EOF'
[server]
port = 8501
address = "0.0.0.0"
headless = true
enableCORS = true
enableXsrfProtection = true
# Kept in step with .streamlit/config.toml, which the guard above now lets win —
# these values apply only to a build whose tree carried no config of its own.
# The server is not the admission gate: at 50 it matched the app's own limit
# exactly and refused uploads with a generic 413 before any page code could
# explain itself. Shape decides now (utils/admission.py). maxMessageSize governs
# the other direction — what is rendered back to the browser — and stays far
# lower.
maxUploadSize = 2000
maxMessageSize = 500

# Streamlit's default is 120 s, which is a laptop lid or a VPN reconnect. See
# .streamlit/config.toml for the full reasoning and the memory tradeoff.
disconnectedSessionTTL = 1800

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

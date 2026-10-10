# syntax=docker/dockerfile:1
# TurboTab v2 in server mode, for a university server (docs/turbotab-next/DEPLOY.md).
#
#   docker build -f turbotab/deploy/Dockerfile -t turbotab .        # from the repository root
#
# Two stages: Node builds the interface, then a slim Python runs the server as a non-root user.
# No R: the R reference implementations are test-only (turbotab/server/requirements-dev-R.txt).
# This is v2's image; the repository root's Dockerfile is Classic's and is not this one.

# ── 1. the interface ─────────────────────────────────────────────────────────
FROM node:22-bookworm-slim AS interface
WORKDIR /src/turbotab/frontend
COPY turbotab/frontend/package.json turbotab/frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY turbotab/frontend/ ./
# tokens.css takes Inter and JetBrains Mono from the repository's static/fonts.
COPY static/fonts/ /src/static/fonts/
RUN npm run build

# ── 2. the server ────────────────────────────────────────────────────────────
FROM python:3.13-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
# libgomp1: the OpenMP runtime the boosted-tree families (lightgbm, xgboost) load.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 turbotab \
    && useradd --system --uid 10001 --gid turbotab --home-dir /data --no-create-home \
       --shell /usr/sbin/nologin turbotab
WORKDIR /app
# constraints.txt holds the versions the tests ran on (how to update it: its header).
COPY turbotab/server/requirements.txt turbotab/server/constraints.txt /tmp/
RUN pip install -c /tmp/constraints.txt -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt /tmp/constraints.txt

# The engine and the domain code it reuses (Classic's ml/, utils/, models/ by import, read-only).
COPY data_processor.py ./
COPY ml/ ml/
COPY utils/ utils/
COPY models/ models/
COPY static/fonts/ static/fonts/
COPY turbotab/ turbotab/
COPY --from=interface /src/turbotab/frontend/dist/ turbotab/frontend/dist/

# /data holds every user's workspace (a volume); /etc/turbotab holds the users file (mounted
# read-only; manage it with python -m turbotab.server.users).
RUN mkdir -p /data /etc/turbotab && chown turbotab:turbotab /data
ENV TURBOTAB_MODE=server \
    TURBOTAB_HOME=/data \
    TURBOTAB_USERS=/etc/turbotab/users.toml \
    PYTHONPATH=/app
VOLUME ["/data"]
USER turbotab
# A writable working folder: DuckDB spills large queries to ./.tmp; the code is found on PYTHONPATH.
WORKDIR /data
EXPOSE 8787

# /healthz needs no session and says only that the server answers.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8787/healthz', timeout=4)"]

CMD ["python", "-m", "turbotab.server", "--mode", "server", "--host", "0.0.0.0", "--port", "8787"]

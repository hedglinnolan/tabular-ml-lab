# Tabular ML Lab — Build & Verify
# ================================
#
# Test Pyramid:
#   Tier 1 (fast):   Unit + workflow tests         ~10s
#   Tier 2 (medium): Streamlit AppTest integration  ~30s
#   Tier 3 (slow):   Playwright E2E browser tests   ~2min (requires running server)
#
# Usage:
#   make test          Run Tier 1 (fast, use after every change)
#   make test-integration  Run Tier 1 + 2 (before pushing)
#   make test-all      Run Tier 1 + 2 + 3 (pre-deploy, requires server at localhost:8501)
#   make verify        Alias for test-integration (the CI target)
#   make ci            What GitHub Actions runs (Tier 1 + 2)

PYTHON := ./venv/bin/python
PYTEST := $(PYTHON) -m pytest
PYTEST_OPTS := --timeout=60 -q

.PHONY: test test-integration test-all verify ci lint clean help

# ── Tier 1: Unit + Workflow (~10s) ───────────────────────────────────
test:
	$(PYTEST) tests/ --ignore=tests/integration $(PYTEST_OPTS)

# ── Tier 2: Streamlit AppTest Integration (~30s) ─────────────────────
test-apptest:
	$(PYTEST) tests/integration $(PYTEST_OPTS)

# ── Tier 1 + 2 Combined (the standard pre-push check) ───────────────
test-integration: test test-apptest

# ── Tier 3: Playwright E2E (requires running server) ─────────────────
test-e2e:
	@echo "Checking if Streamlit is running on localhost:8501..."
	@curl -s -o /dev/null -w "%{http_code}" http://localhost:8501 | grep -q 200 || \
		(echo "❌ Streamlit not running. Start with: make serve" && exit 1)
	$(PYTHON) scripts/integration_test.py

# ── All tiers ────────────────────────────────────────────────────────
test-all: test-integration test-e2e

# ── Aliases ──────────────────────────────────────────────────────────
verify: test-integration
ci: test-integration

# ── Verbose variants ─────────────────────────────────────────────────
test-v:
	$(PYTEST) tests/ --ignore=tests/integration $(PYTEST_OPTS) -v

test-integration-v: 
	$(PYTEST) tests/ $(PYTEST_OPTS) -v

# ── Dev utilities ────────────────────────────────────────────────────
# `serve` is the STREAMLIT app (Classic). The legacy TurboTab door's `make
# turbotab` was retired with that app (BLUEPRINT §9.1); `make turbotab-next`
# below starts TurboTab v2.
serve:
	$(PYTHON) -m streamlit run app.py --server.port 8501

# ── TurboTab Next (v2: docs/turbotab-next/BLUEPRINT.md) ──────────────
#
# `make turbotab-next` builds the React frontend when its bundle is missing or
# older than any of its sources, then serves app and API together on
# $(NEXT_PORT) and opens a browser. `make turbotab-next-dev` prints how to run
# the Vite dev server (hot reload) beside the API instead.
.PHONY: turbotab-next turbotab-next-dev

NEXT_PORT ?= 8787
NEXT_FRONTEND := turbotab/frontend
NEXT_DIST := $(NEXT_FRONTEND)/dist/index.html
NEXT_SOURCES := $(shell find $(NEXT_FRONTEND)/src -type f 2>/dev/null) \
	$(wildcard $(NEXT_FRONTEND)/index.html $(NEXT_FRONTEND)/package.json \
	$(NEXT_FRONTEND)/package-lock.json $(NEXT_FRONTEND)/vite.config.ts $(NEXT_FRONTEND)/tsconfig*.json)

$(NEXT_DIST): $(NEXT_SOURCES)
	cd $(NEXT_FRONTEND) && { test -d node_modules || npm ci; } && npm run build

turbotab-next: $(NEXT_DIST)
	$(PYTHON) -m turbotab.server --port $(NEXT_PORT) --mode local --open

turbotab-next-dev:
	@echo "TurboTab Next with hot reload: two terminals, from the repository root."
	@echo ""
	@echo "  1. The API (FastAPI + SSE) on :8787, which the Vite dev server proxies /api to:"
	@echo "       $(PYTHON) -m turbotab.server --port 8787 --mode local"
	@echo ""
	@echo "  2. The frontend (Vite, hot reload) on :5173 — open http://localhost:5173/"
	@echo "       cd $(NEXT_FRONTEND) && npm install && npm run dev"
	@echo ""
	@echo "  No Python at hand? The frontend alone, against an in-browser mock API:"
	@echo "       cd $(NEXT_FRONTEND) && npm run dev:mock"
	@echo ""
	@echo "  After changing a server model, regenerate the contract types:"
	@echo "       $(PYTHON) -m turbotab.server.openapi --write && (cd $(NEXT_FRONTEND) && npm run gen:api)"

lint:
	$(PYTHON) -m py_compile app.py
	@for f in pages/*.py; do $(PYTHON) -m py_compile "$$f" && echo "  ✓ $$f"; done
	@echo "All pages compile cleanly."

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache

help:
	@echo "Tabular ML Lab — Test Targets"
	@echo ""
	@echo "  make test              Tier 1: unit + workflow tests (~10s)"
	@echo "  make test-apptest      Tier 2: Streamlit AppTest integration (~30s)"
	@echo "  make test-integration  Tier 1 + 2 combined (pre-push check)"
	@echo "  make verify            Alias for test-integration"
	@echo "  make test-e2e          Tier 3: Playwright browser tests (needs server)"
	@echo "  make test-all          All tiers"
	@echo "  make ci                What GitHub Actions runs"
	@echo "  make serve             Start Streamlit dev server (Classic)"
	@echo "  make turbotab-next     Build the v2 frontend if needed and serve TurboTab v2 on :8787"
	@echo "  make lint              Syntax-check all pages"
	@echo "  make clean             Remove caches"

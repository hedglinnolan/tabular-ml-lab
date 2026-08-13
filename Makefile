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

.PHONY: test test-integration test-all verify ci lint clean help turbotab turbotab-check

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
# `serve` is the STREAMLIT app — the parity reference, still. It is not
# TurboTab and it is deliberately not renamed: `make turbotab` below is the
# other door, and both need to be startable.
serve:
	$(PYTHON) -m streamlit run app.py --server.port 8501

# ── TurboTab ─────────────────────────────────────────────────────────
#
# `DRIVE-035`, and this target is the whole of what four human drives cost.
#
# There was no launch command. `make serve` starts Streamlit, `README.md` says
# nothing about TurboTab, and `turbotab/README.md` — written when TurboTab was
# a walking skeleton with no Train step — tells you to build `turbotab/.venv`
# from `turbotab/requirements.txt` and run uvicorn from THAT. That environment
# has no scikit-learn ON PURPOSE, and
# `tests/test_the_guided_door_installs_without_the_app.py` exists to keep it
# that way. So the documented way to start the app could not fit a model, and a
# test guarded the guarantee.
#
# `$(PYTHON)` is `./venv/bin/python` — named explicitly rather than left to
# whatever `python` resolves to, which is the mistake this exists to make
# impossible. The script checks the engine's stack IN THE INTERPRETER THAT WILL
# SERVE and refuses before binding the port, so a missing package is a legible
# refusal in the terminal instead of 21 characters of Internal Server Error at
# the Train step.
turbotab:
	$(PYTHON) scripts/serve_turbotab.py --port $(PORT)

# The environment check on its own, for a machine where the question is only
# *would this start*. Same code path, no port.
turbotab-check:
	$(PYTHON) scripts/serve_turbotab.py --check-only

PORT ?= 8777

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
	@echo "  make serve             Start Streamlit dev server (the parity reference)"
	@echo "  make turbotab          Start TurboTab on :8777 — refuses if the stack is missing"
	@echo "  make turbotab-check    Same environment check, without binding a port"
	@echo "  make lint              Syntax-check all pages"
	@echo "  make clean             Remove caches"

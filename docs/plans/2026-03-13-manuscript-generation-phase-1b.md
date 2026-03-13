# Manuscript Generation Phase 1B — Enforce Canonical Boundary

## Objective
Finish Phase 1 by actually enforcing the canonical manuscript/export boundary across markdown and LaTeX generation.

## Why this follow-on exists
Phase 1 made progress but did not clear acceptance. The remaining blockers are:
1. LaTeX still using full `model_results` / `bootstrap_results` instead of the manuscript-selected subset.
2. LaTeX still using base feature columns instead of the frozen manuscript feature set/count.
3. Methods generation still letting `best_model_by_metric` override the manuscript primary model.
4. Hidden `st.session_state` dependence still causing drift in methods generation.

## In scope
- `pages/10_Report_Export.py`
- `ml/publication.py`
- `ml/latex_report.py`
- add page/integration tests if needed (`tests/test_report_export.py`)

## Required fixes
### 1. Freeze manuscript-scoped model subset explicitly
Add to canonical context:
- `included_models`
- `selected_model_results`
- `selected_bootstrap_results`

Use those in BOTH:
- markdown methods/results generation
- LaTeX generation

### 2. Freeze canonical feature facts explicitly
Add to canonical context:
- `feature_names_for_manuscript`
- `feature_counts`

Ensure markdown and LaTeX both consume those, not `data_config.feature_cols` or raw live columns directly.

### 3. Make manuscript primary model authoritative for manuscript prose
In methods/results prose generation:
- if a manuscript primary model is selected, use that as the highlighted model in manuscript prose
- still expose best-by-metric as a separate fact when useful
- do not let metric-best silently override manuscript-primary in manuscript text

### 4. Reduce hidden session-state reads in methods generation
Make `generate_methods_section()` prefer explicit context/facts passed from export.
Session-state fallback is allowed only for backward compatibility where explicit inputs are unavailable.

### 5. Add integration-level tests
At minimum cover:
- markdown and LaTeX use same selected model subset
- markdown and LaTeX use same feature counts / feature names source
- manuscript primary model is preserved distinctly from metric-best model
- no drift when session state contains extra models not selected for manuscript export

## Acceptance bar
Phase 1B is successful if:
- markdown and LaTeX are driven by the same frozen model subset
- markdown and LaTeX are driven by the same frozen feature facts
- manuscript primary model policy is consistent across outputs
- focused tests pass
- no broader workflow regression is introduced

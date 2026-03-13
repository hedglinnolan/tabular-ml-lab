# Manuscript Generation Phase 1 — Canonical State Boundary

## Objective
Stabilize the manuscript/export state boundary so markdown and LaTeX consume the same frozen manuscript facts instead of deriving them differently from live session state.

## Why this phase first
All three analysis tracks converged on the same root problem:
- `pages/10_Report_Export.py`, `ml/publication.py`, and `ml/latex_report.py` still derive manuscript facts differently.
- This creates drift in feature counts, model subset scope, best-model highlighting, explainability coverage, and section content.
- The safest/highest-leverage move is to treat `build_export_context()` as the canonical freeze point and make downstream generators consume that same resolved manuscript context.

## Phase 1 scope
### In scope
1. Add a canonical manuscript-context builder at the export page boundary.
2. Resolve and freeze manuscript-scoped facts once:
   - selected manuscript model subset
   - primary model
   - best model by metric
   - metrics used
   - feature counts (original / engineered / candidate / selected)
   - preprocessing summaries and per-model preprocessing configs
   - explainability methods included
   - bootstrap/table1/custom-test availability
   - split/sample counts
3. Make methods generation consume this frozen context first.
4. Make LaTeX generation consume the same frozen context/model subset first.
5. Reduce direct dependence on raw `st.session_state` inside manuscript composition paths where practical.

### Out of scope
- Full rewrite of manuscript authoring
- Upstream workflow page refactors unless strictly required
- New export/manuscript features
- UI redesign
- Discussion/Introduction authorship generation

## Files most likely in scope
- `pages/10_Report_Export.py`
- `ml/publication.py`
- `ml/latex_report.py`
- `tests/test_publication.py`
- add `tests/test_report_export.py` if needed for context-freezing behavior

## Concrete requirements
### 1. Canonical manuscript context
Add a frozen manuscript context derived from `export_ctx` + current export selections.
It should explicitly carry:
- dataset facts
- split counts
- feature counts
- selected manuscript models
- best-model and primary-model semantics
- explainability inclusion set
- preprocessing summaries/configs
- table1/final table1 payload
- selected bootstrap results
- readiness/manuscript policy flags

### 2. Methods generation alignment
Refactor methods generation so it prefers canonical manuscript context values over live session-state probing.
Backward-compatible fallback behavior is acceptable, but frozen context should be authoritative.

### 3. LaTeX alignment
Ensure LaTeX uses:
- the same selected manuscript model subset
- the same feature counts
- the same manuscript primary model / best-model policy
- the same methods/results source where available

### 4. Results redundancy policy
Adopt the composition rule from the analysis:
- brief prose summary + structured table
- avoid full metric re-dump in prose when table exists

## Required tests
1. Feature-count consistency across methods + LaTeX
2. Selected-model subset consistency across methods + LaTeX
3. Primary/best-model consistency
4. Exact preprocessing parameter preservation still survives through canonical context
5. Results redundancy policy (brief prose + table, no duplicate metric dump)
6. Placeholder behavior remains where information is truly unavailable

## Acceptance bar
Phase 1 is successful if:
- markdown and LaTeX are driven by the same manuscript-scoped facts
- selected manuscript model subset is consistent across outputs
- feature counts are materially less drift-prone
- best-model / primary-model semantics are consistent across outputs
- tests pass
- no broader workflow regression is introduced

# Workflow-Derived Authoring Boundary Pass

## Goal
Tighten the methods/results generation so it documents what the user explicitly did and what the app explicitly computed in workflow, without opportunistically authoring recomputable-but-unrun analyses.

## Primary boundary rules
1. Do not generate manuscript/results prose for analyses the user did not explicitly run.
2. Prefer frozen export/manuscript context over live session-state drift.
3. Do not label a model as manuscript-primary unless that role is explicitly selected; otherwise describe it as best-by-metric.
4. Prefer user-facing workflow feature facts over internal transformed-matrix names when documenting predictors.

## Scope
### In scope
- `pages/10_Report_Export.py`
- `ml/publication.py`
- `ml/latex_report.py`
- publication/export tests

### Out of scope
- broad manuscript quality improvements beyond this boundary enforcement
- new analyses
- export feature expansion

## Concrete targets
1. Remove/restrict export prose for recomputed-but-unrun analyses (e.g. Bland–Altman if not explicitly run).
2. Reduce remaining live `st.session_state` dependence in methods generation where it can drift from frozen context.
3. Tighten manuscript-primary-model semantics.
4. Ensure predictor narration reflects user-facing workflow facts rather than internal transformed columns.

## Verification requirements
- `python3 -m py_compile` on changed files
- targeted tests for boundary rules
- diff inspection
- no commit

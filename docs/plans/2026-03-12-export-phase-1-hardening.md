# Export Phase 1 Hardening Plan

## Goal
Harden the Export surface so it more honestly and reliably reflects the workflow state already produced by the app, without redesigning the overall architecture.

## Constraints
- No changes to modeling, preprocessing, training, or explainability computation logic unless required for key-contract consistency.
- Preserve the single shared state/data-flow model.
- Prefer low-risk trust/completeness fixes over feature expansion.
- Any change must be verified with syntax checks and targeted regression review.

## In Scope
1. Normalize export-side key contracts
   - Align Export reads with upstream writes for explainability artifacts
   - Resolve `partial_dependence` vs `pdp_results`
   - Resolve SHAP detection inconsistencies (`shap_values` vs `shap_results`)
   - Make Bland-Altman handling explicit and truthful

2. Freeze export context
   - Build a single local export snapshot/context from session state
   - Render downstream export UI/artifacts from that context instead of live repeated lookups where practical

3. Add export readiness audit
   - Show users what is present, inferred, or missing before export
   - Distinguish draft-complete vs evidence-complete where useful

4. Improve reproducibility of ZIP export
   - Export per-model preprocessing artifacts where they exist
   - Add a manifest mapping model to pipeline/config artifacts

5. Clear/contain stale export-side derived artifacts
   - Ensure export-local derived state does not survive upstream workflow changes misleadingly
   - Prefer lightweight cleanup over architectural rewrite

6. Clarify best-model semantics where they surface in export
   - Differentiate best-by-metric from manuscript-primary model if needed
   - Avoid contradictory export outputs

## Out of Scope
- Full project artifact/versioning system
- Full TRIPOD evidence/provenance redesign
- End-to-end manuscript compiler / final PDF guarantees
- Major UX redesign outside Export

## Files Most Likely In Scope
- `pages/10_Report_Export.py`
- `ml/publication.py`
- `ml/latex_report.py`
- `utils/session_state.py`
- Any narrowly required helper touched for key normalization

## Verification Requirements
- `python3 -m py_compile` on all changed Python files
- targeted diff inspection
- app restart after edits
- reviewer pass focused on regressions and contract drift

## Success Criteria
- Export reflects actual upstream artifact keys consistently
- Export UI truthfully indicates what is available/missing
- ZIP bundle better matches per-model preprocessing reality
- No new data-flow branch is introduced
- No breakage in happy-path app usage

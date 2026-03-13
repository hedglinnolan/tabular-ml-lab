# Explainability + Export Correctness Plan

## Goal
Fix the correctness issues Nolan found during a real end-to-end run, without broad redesign.

## User-observed failures to treat as source-of-truth acceptance targets
1. SHAP failed for LASSO / HISTGB_REG / NN with missing-column errors.
2. Permutation importance failed with the same missing-column pattern.
3. Methods generation defaulted to LASSO as best model when it should not have.
4. Explainability methods selector only exposed calibration in the tested run.
5. Generated markdown / LaTeX contained quality defects:
   - malformed `\\subelection.`
   - awkward / duplicated section text
   - duplicated sensitivity entries
   - inconsistent feature counts relative to the actual workflow

## Scope
### In scope
- Explainability column-contract correctness for SHAP and permutation importance
- Best-model default / manuscript-primary-model consistency in export methods generation
- Explainability-method detection in export methods UI
- Manuscript generation bugs directly tied to the tested workflow
- Targeted fixes to counts / wording when they are clearly wrong due to state derivation bugs

### Out of scope
- New explainability features
- New export features
- Large refactor of methodology logging or publication architecture
- General copy polish not tied to the observed bugs

## Likely files in scope
- `pages/07_Explainability.py`
- `pages/10_Report_Export.py`
- `ml/publication.py`
- `ml/latex_report.py`
- Any narrowly required helper for feature-name or best-model reconciliation

## Implementation guidance
1. Reproduce the bug path from Nolan's report logically from code/state contracts.
2. Fix the explainability/preprocessing feature-name mismatch at the narrowest correct layer.
3. Ensure SHAP / permutation importance operate on the same transformed feature representation expected by fitted pipelines and models.
4. Make export methods best-model default consistent with actual metric-based winner unless the user explicitly overrides it.
5. Ensure explainability-method selector reflects the actual explainability artifacts created upstream.
6. Fix manuscript generation defects exposed by the sample output, but keep changes bounded to correctness rather than style rewriting.

## Verification requirements
- `python3 -m py_compile` on all changed Python files
- targeted diff inspection
- if feasible, a narrow code-path sanity check for explainability/export state contracts
- stop without commit

## Success criteria
- Missing-column SHAP/permutation bug is plausibly fixed in code
- Best-model default no longer obviously drifts from results
- Explainability methods detection reflects actual stored artifacts
- Exported markdown/LaTeX no longer contains the observed malformed section defects from the current generation logic
- Changes remain bounded and do not alter the overall workflow architecture

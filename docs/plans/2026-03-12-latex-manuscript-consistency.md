# LaTeX Manuscript Consistency Pass

## Goal
Resolve the remaining concrete defects in generated LaTeX so the manuscript draft is internally consistent, parameter-faithful, and free of obvious text corruption.

## Source-of-truth defects
1. Contradictory feature counts (e.g. 35 predictors vs reduced from 26 to 23).
2. Exact preprocessing parameters (e.g. 4th / 96th percentile clipping) not surviving into LaTeX.
3. Sensitivity-analysis duplication (e.g. same model/metric narrated twice with different seed counts).
4. Methodological text corruption (e.g. `due to theng pipeline) due to ...`).
5. Results-section redundancy between prose and table.

## Scope
### In scope
- `ml/publication.py`
- `ml/latex_report.py`
- `pages/10_Report_Export.py`
- any narrow helper needed to reconcile workflow counts or manuscript text assembly

### Out of scope
- broad export redesign
- new manuscript sections
- aesthetic rewriting beyond these defects
- new analytical features

## Implementation requirements
1. Establish one consistent source for original, engineered, candidate, and selected feature counts used in manuscript text.
2. Ensure exact known preprocessing parameters survive all the way into LaTeX.
3. Implement a clear sensitivity deduplication policy for manuscript text (recommended: latest run per model/metric/analysis type).
4. Eliminate methodology text corruption/duplication artifacts.
5. Make Results prose/table balance intentional and non-redundant.

## Verification requirements
- `python3 -m py_compile` on changed Python files
- targeted tests for:
  - feature-count consistency
  - parameter specificity in LaTeX/manuscript text
  - sensitivity deduplication behavior
  - text-corruption regression
- diff inspection
- stop without commit

## Success criteria
- feature counts are internally consistent
- exact known preprocessing parameters survive into LaTeX
- duplicate sensitivity entries are resolved by policy
- methodology text corruption is gone
- results narrative/table balance feels intentional
- no new workflow/export regressions are introduced

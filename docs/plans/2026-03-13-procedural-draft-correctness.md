# Procedural Draft Correctness Cleanup

## Goal
Fix the remaining high-impact correctness defects in the workflow-derived markdown/LaTeX draft generation, using Nolan's real saved-session path as the target acceptance case.

## Real-world source-of-truth defects
1. **Canonical feature-count narrative is still inconsistent**
   - e.g. 25 original predictors, 35 candidate predictors after feature engineering, 23 selected predictors should be narrated coherently.
2. **Exact preprocessing parameters are still being lost**
   - e.g. 4th / 96th percentile clipping still collapses to generic "percentile-based winsorization" in the generated procedural draft.
3. **Text corruption remains in generated methods/LaTeX handoff**
   - e.g. broken fused strings like `andautomaompared...`

## Scope
### In scope
- `ml/publication.py`
- `ml/latex_report.py`
- `pages/10_Report_Export.py`
- tests for publication/export text generation

### Out of scope
- broad manuscript-quality improvements beyond these defects
- new export features
- interpretation/discussion generation
- unrelated UI polishing

## Implementation requirements
1. Establish one canonical feature-count story for methods text:
   - original predictors
   - engineered candidate predictors
   - selected final predictors
2. Ensure exact preprocessing parameter values survive the actual methods-generation path used in the real workflow.
3. Eliminate text concatenation/corruption artifacts in the markdown→LaTeX / methods-generation path.
4. Keep the pass tightly scoped to procedural correctness.

## Verification requirements
- `python3 -m py_compile` on changed files
- targeted tests for:
  - feature-count narrative consistency
  - exact preprocessing parameter preservation
  - text-corruption regression
- diff inspection
- no commit

## Acceptance bar
- generated procedural draft tells one coherent feature-count story
- exact known preprocessing parameters survive into the generated draft
- no obvious text-corruption artifacts remain in the observed path
- no new regressions introduced

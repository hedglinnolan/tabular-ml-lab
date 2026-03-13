# Export Specificity Pass Plan

## Goal
Increase the specificity and fidelity of export/methods generation so known workflow parameters are preserved when they materially improve reproducibility and scientific usefulness.

## User-observed acceptance target
- If the workflow knows specific settings (example: outlier clipping at 4% / 96%), export should preserve that level of detail rather than collapse it into vague summaries like "percentile-based winsorization."

## Scope
### In scope
- Methods/export generation detail level
- Preprocessing parameter specificity
- Feature-selection / bootstrap / sensitivity / feature-engineering specificity where reliable state exists
- Tightening generated narrative where current wording throws away known parameters

### Out of scope
- Inventing details not present in state
- Broad manuscript-style rewriting
- New export features unrelated to specificity

## Likely files in scope
- `pages/10_Report_Export.py`
- `ml/publication.py`
- `ml/latex_report.py` (only if needed for preserving generated detail)
- any helper that formats preprocessing or methodological descriptions

## Implementation guidance
1. Identify where export currently generalizes known parameters too aggressively.
2. Preserve exact values when they are present and scientifically meaningful.
3. Prefer faithful parameter reporting over generalized summaries.
4. Never fabricate precision where state does not support it.
5. Keep the pass bounded to specificity/trust, not a full narrative rewrite.

## Verification requirements
- `python3 -m py_compile` on changed Python files
- targeted diff review
- if feasible, narrow sanity checks against representative formatting paths
- stop without commit

## Success criteria
- Export reflects exact workflow parameters when available (e.g. clipping thresholds)
- Generated methods feel more reproducible and less generic
- No invented detail is introduced
- Changes remain bounded and low-risk

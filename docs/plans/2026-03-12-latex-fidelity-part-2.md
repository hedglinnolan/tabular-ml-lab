# LaTeX Fidelity Part 2 Plan

## Goal
Make the LaTeX manuscript generator faithfully carry through the richer known workflow/methods details instead of falling back to thin generic scaffold sections when that information already exists.

## User-observed acceptance targets
1. If generated methods already contain substantive workflow detail, LaTeX should preserve it.
2. Placeholder-only sections should remain only where the app truly does not know the answer.
3. Known details should appear in LaTeX for areas like:
   - missing data handling
   - preprocessing/model development
   - feature engineering when present
   - interpretability/explainability when present
4. Minor text-hygiene defects like `PriorWork` should be fixed.

## Scope
### In scope
- Handoff from export/methods generation into LaTeX generation
- Results/methods splitting and reuse behavior
- Placeholder suppression when real content exists
- Minor text-hygiene fixes directly tied to current generated output

### Out of scope
- Full manuscript redesign
- New export/manuscript features
- General academic writing polish beyond preserving existing known detail

## Likely files in scope
- `ml/latex_report.py`
- `pages/10_Report_Export.py`
- `ml/publication.py` (only if required to improve handoff/fidelity)

## Implementation guidance
1. Trace exactly what methods/report text is available when `Generate LaTeX Manuscript` is clicked.
2. Ensure LaTeX generation uses the richest available generated methods content rather than unnecessary fallback placeholders.
3. Preserve detailed generated subsections when they exist, while avoiding section-structure collisions.
4. Fix the specific malformed text issue `PriorWork` → `Prior Work` and similar obvious fusion defects if found.
5. Keep the pass bounded to fidelity and correctness, not narrative enhancement.

## Verification requirements
- `python3 -m py_compile` on changed Python files
- targeted tests if feasible
- diff inspection
- stop without commit

## Success criteria
- LaTeX output reflects substantially more of the known workflow detail already present in generated methods
- Generic placeholders remain only where information is truly unavailable
- No regression to current export/manuscript generation

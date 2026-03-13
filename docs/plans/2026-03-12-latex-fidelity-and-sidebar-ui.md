# LaTeX Fidelity + Sidebar UI Plan

## Goal
Fix the remaining session-controls readability issue and make LaTeX manuscript generation faithfully carry through the richer generated workflow/methods details instead of collapsing into a thin generic scaffold.

## User-observed acceptance targets
1. Session management buttons in the sidebar must be readable without hover.
2. LaTeX output should carry through known workflow details when the app already has them.
3. Placeholder-only sections should remain only where the app truly lacks information.
4. Obvious unresolved artifacts like `[METRICS]` and malformed text such as `mainresults` should be fixed.

## Scope
### In scope
- Sidebar session-controls styling/readability
- LaTeX generation fidelity relative to methods/report content
- Placeholder substitution where reliable upstream data exists
- Minor manuscript text hygiene directly tied to current generation defects

### Out of scope
- Full manuscript-authoring redesign
- New export features
- Broad UI restyling outside session controls

## Likely files in scope
- `utils/session_manager.py`
- `pages/10_Report_Export.py`
- `ml/latex_report.py`
- `ml/publication.py` (if needed for data handoff / placeholder resolution)

## Implementation guidance
1. Fix the session-controls buttons by targeting the actual rendered sidebar button/download-button elements.
2. Inspect the handoff from generated methods/report text into LaTeX generation.
3. Preserve detailed generated workflow text where reliable instead of letting LaTeX fall back to generic placeholders.
4. Replace resolvable placeholders like `[METRICS]` when the app already knows the metrics used.
5. Fix minor text-hygiene defects in the generated LaTeX path (e.g. missing spaces like `mainresults`).
6. Keep the changes bounded: improve fidelity, not narrative perfection.

## Verification requirements
- `python3 -m py_compile` on changed Python files
- targeted tests if possible
- diff inspection
- stop without commit

## Success criteria
- Sidebar session buttons are visibly readable before hover
- LaTeX manuscript reflects meaningful workflow details already known to the app
- Resolvable placeholders are resolved
- No major regression to export/manuscript generation

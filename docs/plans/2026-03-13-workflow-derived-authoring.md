# Workflow-Derived Methods & Results Authoring Pass

## Goal
Improve the quality of generated methods/results text as **robotic documentation of user actions and computed outputs**, not as full academic authorship.

## Product boundary
The app should be strongest at writing from:
- workflow state
- user-selected options
- logged methodology actions
- computed model/validation outputs

The app should NOT attempt to author:
- scientific rationale
- literature framing
- interpretation of findings
- clinical/scientific implications
- conclusions beyond mechanical result summaries

## Scope
### In scope
- methods-writing quality for action-derived workflow documentation
- procedural results-summary quality grounded in computed outputs
- section ordering and readability
- reduction of robotic awkwardness / redundancy
- clearer boundary between generated procedural text and author-owned narrative

### Out of scope
- discussion generation
- introduction/background generation
- interpretive claims
- domain-specific significance claims
- broad AI writing expansion

## Likely files in scope
- `ml/publication.py`
- `ml/latex_report.py`
- `pages/10_Report_Export.py`
- tests around publication/export text generation

## Desired outcomes
1. Methods text reads like a strong procedural draft of what was actually done.
2. Results text is concise, factual, and table-aware.
3. Exact workflow parameters are preserved where they matter.
4. Generated text avoids overclaiming or pretending to interpret the science.
5. Placeholder boundaries are intentional and visible.

## Verification requirements
- `python3 -m py_compile` on changed files
- targeted tests for methods/results generation behavior
- diff inspection
- app restart after acceptance
- no commit

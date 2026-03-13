# EDA Information Architecture Pass

## Goal
Refactor the presentation of the EDA page so it feels intentionally structured rather than like a hodgepodge of accumulated analyses.

## Core design principle
EDA should prioritize attention before it offers exploration.

## Key constraint
For datasets with many features, the page must degrade gracefully:
- summarize before enumerating
- use top-N views where appropriate
- collapse heavy sections by default
- avoid flooding the user with equal-weight per-feature content

## Scope
### In scope
- information architecture of `pages/02_EDA.py`
- clearer section hierarchy
- progressive disclosure / collapsible deeper diagnostics
- large-feature-friendly presentation logic
- reduced visual duplication between decision hub, warnings, recommendations, and diagnostics

### Out of scope
- new EDA algorithms
- major charting rewrites unless required for hierarchy
- changes to modeling/preprocessing logic
- broad app-wide navigation redesign

## Desired page structure
1. Dataset Verdict
2. What Matters Most
3. Recommended Checks
4. Deep Dive Diagnostics
5. Handoff / Next Step

## Product goals
1. Make the page feel composed, not cumulative.
2. Reduce equal-weight clutter.
3. Preserve analytical power while lowering cognitive load.
4. Improve usability for wide datasets.
5. Keep existing EDA functionality available.

## Verification requirements
- `python3 -m py_compile pages/02_EDA.py`
- targeted diff inspection
- app restart after acceptance
- no commit

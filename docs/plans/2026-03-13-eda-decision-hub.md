# EDA Decision-Hub Refinement Plan

## Goal
Make the EDA page function as the central decision-making hub of the workflow rather than just a capable exploratory analysis page.

## Product rationale
The next highest-leverage refinement is not more export polish or navigation cleanup. It is helping users understand:
- what kind of dataset they have
- what risks are present
- what they should do next
- what they can safely skip
- how defensible the analysis path is

## Scope
### In scope
- top-of-page EDA triage / dataset verdict layer
- downstream branching guidance
- reviewer-risk framing
- better "so what?" interpretation of EDA outputs
- explicit handoff into next recommended workflow steps

### Out of scope
- new analytical algorithms
- major charting rewrites unless necessary for guidance
- changes to modeling/preprocessing logic
- broad UI redesign outside the EDA page

## Likely files in scope
- `pages/02_EDA.py`
- small shared UI helpers only if strictly needed

## Desired outcomes
1. Users quickly understand the state of their dataset.
2. EDA produces actionable downstream recommendations.
3. The page helps decide whether advanced steps are warranted.
4. Reviewer-facing risks are surfaced clearly and usefully.
5. The page feels like the product’s decision hub.

## Verification requirements
- `python3 -m py_compile` on changed files
- targeted diff inspection
- app restart after acceptance
- no commit

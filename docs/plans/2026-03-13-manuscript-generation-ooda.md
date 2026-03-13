# Manuscript Generation OODA Program

## Objective
Improve markdown and LaTeX generation so they become more internally consistent, section-aware, parameter-faithful, and trustworthy as a manuscript scaffold.

## OODA framing
### Observe
Current system is useful but still suffers from:
- inconsistent counts/metadata across sections
- uneven preservation of known workflow parameters
- weak section-aware composition (especially Results vs tables)
- over-reliance on live session state instead of one canonical manuscript source

### Orient
This is now primarily a manuscript-state + composition architecture problem, not just a copy-polish problem.

### Decide
Run three short analysis tracks first, then synthesize and execute implementation:
1. Manuscript state canonicalization analysis
2. Section composition rules analysis
3. Implementation planning / code audit

### Act
After synthesis, execute a bounded implementation batch with reviewer coverage and main-agent acceptance.

## Analysis deliverables
### Track A — Manuscript state canonicalization
Deliver:
- canonical manuscript/export state fields
- where current markdown and LaTeX diverge in state sourcing
- recommended `manuscript_context` shape

### Track B — Section composition rules
Deliver:
- rules for Methods / Results / Discussion placeholders / Supplementary sections
- prose-vs-table redundancy policy
- placeholder policy: where placeholders remain vs where generation should be specific

### Track C — Implementation planning / code audit
Deliver:
- exact files/functions to touch first
- safest implementation order
- required regression tests
- what to defer

## Execution constraints
- No broad rewrite before synthesis
- No parallel conflicting code edits during architecture phase
- After synthesis, execution may proceed without another user approval
- Verification required before acceptance

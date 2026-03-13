# Upload & Audit Simplification Plan

## Goal
Reduce first-contact friction on Upload & Audit so the page feels like the beginning of a guided research workflow rather than a project-management / ETL console, while preserving advanced/multi-dataset capability.

## Product rationale
This remains the highest-leverage product refinement after workflow architecture and manuscript/export stabilization because:
- first-run activation is still the biggest friction point
- the page is large and conceptually crowded
- advanced data setup competes with the default happy path too early

## Scope
### In scope
- visual/information hierarchy on Upload & Audit
- default single-dataset path emphasis
- demotion/collapse of advanced/multi-dataset controls where safe
- microcopy / headings / layout structure that support the quick workflow

### Out of scope
- changes to underlying data merge logic
- changes to modeling/preprocessing/training pipelines
- major persistence/session redesign

## Likely files in scope
- `pages/01_Upload_and_Audit.py`
- `utils/theme.py` if minor shared UI support is needed
- possibly small helper files if strictly required

## Desired outcomes
1. A first-time user immediately sees the shortest valid path.
2. Multi-dataset/advanced setup is still available but not primary.
3. The page reads as the start of analysis, not data operations overhead.
4. No breakage to existing upload/merge/project capabilities.

## Verification requirements
- `python3 -m py_compile` on changed files
- targeted diff inspection
- app restart after acceptance
- no commit

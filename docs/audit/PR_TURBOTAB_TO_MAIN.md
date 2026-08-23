# PR: TurboTab → main

Open at: https://github.com/hedglinnolan/tabular-ml-lab/compare/main...TurboTab

**Title:** Classic paper-risk sprint: all 52 findings fixed, adversarially verified; main back-merged

## What this is

The Classic (Streamlit) app cleared of the 52 confirmed paper-risk findings from `docs/audit/CLASSIC_PAPER_RISK.json`, ahead of the AJCN submission — plus the TurboTab product work this branch already carried, and a back-merge of main's 16 commits (PRs #148–151) with conflicts resolved both ways.

## The sprint, in numbers

- **52/52 findings FIXED** in the ledger (`docs/turbotab/data/findings.json`), each with a named revert-checked regression test in `tests/test_paper_risk_*.py` / `tests/test_paper_alignment.py` (~280 new tests).
- **One full adversarial verification cycle**: four read-only verifiers attacked every closure with the original repros plus variants; 38 held, 14 were impeached and repaired, with the verifiers' attacks becoming the pins.
- **Paper-alignment app fixes** so the manuscript's claims are true in code: CV on by default, ≥2-method consensus, coach CI detector, cross-model consensus highlight, plots+calibration export by default, normality-driven test defaults, all-models seed sensitivity, external validation persisted into provenance and the manuscript.
- Headline correctness work: split identity is labels end-to-end with refusal on mismatch; the lockbox measures repetition regardless of column name, asks for the subject column, counts its own opens, and survives archives whole; join/stack keys share one canonical space with value-based precision refusal; resets derive from the provenance schema; the report reads the realized split instead of re-deriving.

## Gate evidence

- Scoped suite: **2495 passed, 0 failed** (excluding the torch-dependent NN module tests — torch is deliberately optional per TEST-038; `models/nn_whuber` now imports without it).
- Meta-tests: suite order-independence + every-FIXED-row-names-a-running-test — **4 passed (36:42)**.
- Integration + workflow suites green (373); routing baseline formally adjudicated through L67.
- Ledger: `ledger.py check` clean, 494/1016 closed repo-wide.

## Notes for review

- `docs/turbotab/VALUE_CHECK_ADJUDICATION.md` L66/L67 entries record how main's diagnostics dedup and the new grain question moved the frozen routing baseline, and the rulings.
- Open ledger rows filed during the sprint (MISC-090..100) are tracked follow-ups, none blocking: dead-UI cleanups, the DAG declaration gap, the routing clause-attribution mechanism, and TRANSITION_PLAN §05's expired freeze wording.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01SFnnwVx67prTv6Dhzvs3sx

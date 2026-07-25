# Findings ledger — data import, multi-file combine, JSON

Why this file exists: a stress-test run confirmed 48 defects in the import and
join engines. The critical tier was fixed, a few systemic rules retired others,
and the rest were never checked off — because the list was never written down.
When the question "did you address all of those?" came back, the answer had to
be recovered from workflow journals on disk rather than read off a document.

So: every finding lands here with its disposition, and nothing is closed
without a regression test named after it. A finding is only `FIXED` when a test
in `tests/test_stress_regressions.py` would fail if the fix were reverted.

Status values: `FIXED` · `OPEN` · `WONTFIX` (with reason) · `NOT-A-DEFECT`

---

## Governing rule

The app may be **silent**, and it may **refuse**, but it must never **assert
something false**.

`high` confidence is the only tier the UI pre-selects, so `high` means the app
is asserting. Nothing uncertain may reach it. A confidently-wrong answer is
worse than a crash, because a crash gets reported and a wrong number gets
published.

---

## Closed in this wave

| # | Finding | Severity | Status | Test |
|---|---------|----------|--------|------|
| C1 | `age ↔ age` across two survey cycles rated `high` confidence; join delivered 144 Cartesian rows, and promised-equals-delivered so the existing guard passed | critical | FIXED | `TestMeasurementIsNotAKey` |
| 39 | Row counters (`row`, `index`, `id`, `n`, `Unnamed: 0`, `rownum`, `obs`) rated `high` when the column names matched — identical names defeated the guard (`1.0 >= 0.85`) | critical | FIXED | `TestRowCounterIsNotAKey` |
| NEW | Repeated-measures joins found **zero** key candidates: `_MIN_UNIQUENESS = 0.5` discarded any column repeating on >half its rows, and 3 visits/subject is 0.33 | critical | FIXED | `TestRepeatedMeasuresAreJoinable` |
| 13 | Duplicate column labels crashed `check_empty_rows_and_columns` and `check_constant_columns`; `diagnose()` swallowed it and showed a clean bill of health | major | FIXED | `TestDuplicateLabels` |
| 27 | `raw_numeric >= 0.99` skipped pure-numeric text, so a promoted-header frame stayed all-text, was declared clean, and `age.mean()` raised `TypeError` | major | FIXED | `TestNumericStoredAsText` |
| C2 | Step 2 could not express stack-then-link; the NHANES 2×2 shape had no correct path (400 rows or 144, never 200) | critical | FIXED | `TestMixedRelationships` |
| C3 | JSON `records_key` was a dead end — the loader told users to pick a key, with no widget and no parameter to carry it | major | FIXED | `TestJsonRowSetChoice` |
| C4 | Several wrapper keys present → resolved by `_JSON_WRAPPER_KEYS` iteration order, silently | major | FIXED | `TestJsonRowSetChoice::test_a_wrapper_guess_is_disclosed` |
| NEW | `__source_file_demo` / `__source_file_labs` leaked into the feature pool after stack-then-link, because reservation matched exactly rather than by prefix | major | FIXED | `TestSourceColumnIsNeverAPredictor` |
| NEW | A failing check reported nothing at all, so an uninspectable file looked clean | major | FIXED | `TestDuplicateLabels::test_a_failed_check_is_disclosed_not_swallowed` |
| NEW | Truncated JSON reported "read as JSON Lines" because the fallback was assumed, not attempted | minor | FIXED | `TestJsonRowSetChoice::test_truncated_json_is_not_mislabeled_as_json_lines` |

---

## Still open

Populated from the two audit runs in flight:

- `wf_70254f26-494` — fresh exhaustive hunt, ten lenses over the multi-file
  path and JSON, every finding independently reproduced and adversarially
  judged twice.
- `wf_5446c57b-3f6` — all 48 recovered findings re-run against current HEAD,
  every verdict adversarially rechecked in both directions.

Nothing is closed on a verdict alone. Each surviving finding gets a repro, a
fix, and a named test here.

---

## Where the original 48 live

Recovered from the stress-test run's journal and written to
`scratchpad/audit/orig48/finding_NN.md` (title, confirmed severity, verifier
reasoning, and the repro as recorded). The raw journal is at:

    subagents/workflows/wf_e5abb4fe-e32/journal.jsonl

102 raw findings across 8 families; 48 survived verification
(11 critical, 30 major, 7 minor).

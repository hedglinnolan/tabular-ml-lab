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

## Still open — now tracked in the live ledger

**This section used to point at two workflow IDs that never wrote results.** It
pointed at them for long enough that the paths went away underneath it:
`scratchpad/audit/orig48/` is empty, `subagents/` does not exist, and the
journal named below is gone. A pointer to a dead artifact is worse than no
pointer, because it reads as a plan.

The tail now lives in `docs/turbotab/data/findings.json` as the `IMPORT-*` rows,
worked through `docs/turbotab/tools/ledger.py` like everything else:

```bash
python docs/turbotab/tools/ledger.py stats
python docs/turbotab/tools/ledger.py check     # schema guard, before every commit
```

Recovered at L10, in two passes:

- **Thirteen from the tests that remember them** (`IMPORT-001` … `IMPORT-013`,
  `FIXED`). `tests/test_stress_regressions.py` has 21 classes and this file
  names 8; the other 13 guard defects whose text was lost. Each row is
  reconstructed from its regression class — what broke, what the fix asserts,
  what would regress — and says in its note that it is a reconstruction from
  executable evidence rather than a transcription.
- **Six re-derived by a fresh adversarial pass** (`IMPORT-014` … `IMPORT-019`,
  `OPEN`), each with a runnable reproduction in its evidence field. These may or
  may not be among the ~24 whose statements are gone; there is no way to tell,
  and the rows say so.

**Roughly two dozen of the original 48 have neither surviving text nor a
guard.** That gap is real and is not closed by the above. It is the reason the
freeze condition below is written the way it is.

---

## The freeze, and what lifts it

`TRANSITION_PLAN.md` §05 freezes `ml/import_doctor.py`, `ml/join_doctor.py`,
`utils/combine*.py` and `pages/01` as **engine-move-only** pending this tail.
The old condition — "until that ledger closes" — could not be evaluated, because
the ledger it referred to had no open items written down.

**The new condition: the freeze lifts when no `IMPORT-*` row is
un-dispositioned.** Every row is `OPEN`, `PARTIAL`, `FIXED` with a named test,
`NOT-A-DEFECT` with a reason, or `WONTFIX` with a reason — which is checkable by
`ledger.py check` rather than by memory.

That is a lower bar than "no open defects", deliberately: the product owner has
ruled that Guided ships multi-file assembly, and a build cannot start on an
engine whose defect state is *unknown*. Known and open is a backlog. Unknown is
a hazard, and a live preview asserts continuously rather than once.

---

## Where the original 48 lived

Recovered from the stress-test run's journal to
`scratchpad/audit/orig48/finding_NN.md`, with the raw journal at
`subagents/workflows/wf_e5abb4fe-e32/journal.jsonl`. **Both paths are gone.**
102 raw findings across 8 families; 48 survived verification (11 critical, 30
major, 7 minor) — the counts are all that is left of them, and they are recorded
here so the size of the loss stays legible.

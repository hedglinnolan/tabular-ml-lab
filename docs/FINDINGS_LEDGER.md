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

**Correction, L11.** This section said the two audit runs were "in flight". They
were not: both landed, and their output has been committed in `docs/audit/` the
whole time — `ORIGINAL_48_FINDINGS.md` (all 48 verbatim, with severity, verifier
reasoning and repros), `HUNT_FINDINGS.md` (run `wf_70254f26-494`), and
`ADJUDICATION.md` (run `wf_5446c57b-3f6`). `TRANSITION_PLAN.md` §05 points at
them two lines above the freeze rule.

What was gone was only the *scratchpad* copy this file pointed at
(`scratchpad/audit/orig48/`, now empty) — ephemeral storage standing in for a
durable artifact that already existed. A ledger asserting something false about
itself is the governing rule's own failure in the document that states the rule,
and it cost a loop of rediscovering findings that were already written down. The
process rule that came out of it is in `FEATURE_PARITY.md`: *a record that points
at ephemeral storage will eventually lie, and it lies toward "the work is gone."*

The tail now lives in `docs/turbotab/data/findings.json` as the `IMPORT-*` rows,
worked through `docs/turbotab/tools/ledger.py` like everything else:

```bash
python docs/turbotab/tools/ledger.py stats
python docs/turbotab/tools/ledger.py check     # schema guard, before every commit
```

Recovered and reconciled across L10 and L11, in three passes:

- **All 48 originals** are filed as `IMPORT-101` … `IMPORT-148`, keyed to the
  original numbering, each carrying the verifier's title and severity verbatim
  and a disposition against HEAD.
- **Thirteen recovered from the tests that guard them** (`IMPORT-001` …
  `IMPORT-013`) before `docs/audit/` was rediscovered. That work was not wasted:
  it is independent corroboration, and the test names turned out to be
  reconstructible statements of the findings — which is now a policy, not an
  accident (`FEATURE_PARITY.md`).
- **Nine re-derived by fresh adversarial probes** (`IMPORT-014` …
  `IMPORT-022`), each with a runnable reproduction. `IMPORT-014` turned out to
  duplicate a hunt finding, which is corroboration rather than waste;
  `IMPORT-020` and `IMPORT-022` are lockbox findings and drove constitution §03.

`docs/audit/HUNT_FINDINGS.md` is explicitly **unverified** — its adversarial
stage never ran — and `docs/audit/ADJUDICATION.md` measured against `413671a`,
before this branch's fixes. Per `docs/audit/RESUME.md`, treat that file as leads
and never as status: findings 5, 7, 11, 13 and 27 read cold will send you to
re-fix already-fixed code.

## The freeze, and what lifts it

**Defined once, in `TRANSITION_PLAN.md` §05.** Not restated here — the earlier
version of this section carried its own wording, `LOOP.md` carried a third, and
the three did not agree about what the freeze permitted.

In short: the freeze is **lifted for repair** — that is what this audit was for —
and new construction waits on three gates, all evaluable: every recorded finding
dispositioned against HEAD; all ten lenses run with no `critical` or `landmine`
`IMPORT-*` row left `OPEN`; and the seven guided-assembly requirements carrying
named tests. §05 has the list.

An earlier condition — *"no un-dispositioned rows"* — is superseded: it measured
what got filed rather than whether the audit was complete.

---

## Where the original 48 live

**`docs/audit/ORIGINAL_48_FINDINGS.md`** — all 48, verbatim, with the title, the
severity as confirmed, the verifier's reasoning and the repro as recorded. In
the repository, committed.

This section previously named `scratchpad/audit/orig48/` and a journal under
`subagents/`, both since deleted, and concluded that the findings were gone. The
durable copy already existed; only the pointer was ephemeral. 102 raw findings
across 8 families; 48 survived verification (11 critical, 30 major, 7 minor),
and every one of them is now an `IMPORT-1NN` row.

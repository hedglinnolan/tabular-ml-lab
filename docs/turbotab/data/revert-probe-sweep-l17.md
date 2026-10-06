# Revert-probe sweep — L17

**Question.** The ledger's integrity claim is that a `FIXED` row names a test that fails when the
fix is reverted. `tools/ledger.py check` can only verify the row *names* a test. This sweep
measures how often that test is actually load-bearing.

**Method.** Sampling frame: every `FIXED` row at `critical` or `landmine` severity carrying a test
— 92 rows. Sample: 24, drawn with `random.Random(1616).sample`, seeded so the draw is reproducible
and visibly not cherry-picked. For each row the fix was located from the row's own `note` and the
source, textually reverted, and the named test re-run.

Two pairs share one test (`IMPORT-103`/`106`, `IMPORT-104`/`140`), so 24 rows resolve to 22
distinct probes. For those pairs one mechanism was reverted; the sibling rides on the same probe
and is reported as guarded on that basis rather than independently.

## Result

| | rows | share |
|---|---:|---:|
| **Guarded** — test went red under the revert | 22 | 91.7% |
| **Not guarded** — test stayed green | 1 | 4.2% |
| **Unrunnable** — test skips in this environment | 1 | 4.2% |

## The two that failed

**`STATE-056` — the named test guards neither half.**
`test_the_ledger_round_trips_through_json` asserts `after == before` where both sides are
`InsightLedger.to_list()` output, so it compares the serializer against itself. Deleting
`manuscript_text` from `to_dict` drops it from both sides and the equality still holds — probed,
stayed green. Reverting `from_list`'s `upsert()` to `add()` also left it green. The upsert half
*is* guarded, by `test_a_save_file_with_two_entries_for_one_id_keeps_the_later_one` in the same
file — which the row's `note` names and its `test` field does not. Reopened `PARTIAL`.

**`MODELS-001` — the named test cannot run here.**
`test_clone_and_refit_now_raises_instead_of_answering_from_the_old_model` opens with
`pytest.importorskip("torch")`, and torch is deliberately not installed. It reports `SKIPPED`. The
fix may be correct; nothing in this environment demonstrates it. Reopened `PARTIAL`.

## The caveat on the headline number

**Five of the twenty-two reverts were wrong on my first attempt**, and every one produced a
plausible `NOT GUARDED` before it was corrected:

| row | the wrong revert | why it was wrong |
|---|---|---|
| `IMPORT-127` | loosened the `raw_numeric >= 0.99` gate | the defect was the branch *skipping*; the fix made it emit |
| `MINE-003` | renamed `compute_pca` | red with `ImportError`, which is not the defect |
| `TEST-001` | added a no-op function beside the real one | the real one was untouched — a no-op edit |
| `IMPORT-118` | renamed `_key_tokens` | `NameError`, not the sampling defect |
| `T0-PREREG-002` | an anchor appearing three times | not a unique edit |

So the true rate is bounded by the quality of the reconstruction, and a sweep that trusted its
first revert would have reported ~5 false failures against 1 true one. `NOT GUARDED` is a
hypothesis until the revert is confirmed to reintroduce the named defect.

## What the sample does and does not represent

Twelve of the twenty-four are `IMPORT-1xx` rows from the original 48-finding audit, most closed as
*does not reproduce* against the shared characterization file `tests/test_stress_regressions.py`.
Those were uniformly well guarded — that file was written as a characterization suite and it
behaves like one. The sample is therefore **weighted toward the best-tested corner of the ledger**,
and a sweep of `high`-severity rows, or of rows whose tests were written alongside a build rather
than as characterization, could plausibly come out worse.

## Is the self-referential round trip a class or a one-off?

Cheap to answer, so it was answered rather than assumed. An AST scan over every `test_*.py` for
an `assert a == b` whose **both** sides are assigned from the same serializer method found two
candidates:

| test | verdict |
|---|---|
| `test_the_manuscript_voice_survives_the_save_file.py::test_the_ledger_round_trips_through_json` | **real** — this is `STATE-056` |
| `tests/test_router.py::test_the_plan_is_a_pure_function_of_the_record` | **false positive** |

The router one compares two *different inputs* through the same serializer and asserts they agree,
which is a **determinism** claim, not a completeness one. A lossy `to_dict` would not undermine
it — both sides would lose the same field and the claim ("reordering the findings does not change
the plan") would still be true and still worth holding.

So the pattern is **rare rather than systemic**: one instance in the repository. Recorded because
the negative result is the useful one — it says the ledger does not need a sweep for this
specific shape, and the rule in `FEATURE_PARITY.md` is prevention rather than cleanup.

---

# L18 · The stratified re-sample, excluding the IMPORT-1xx family

The L17 result carried a stated caveat: twelve of twenty-four rows were `IMPORT-1xx` findings from
the original 48-file audit, closed against `tests/test_stress_regressions.py` — a characterization
suite, and it behaves like one. The sample was **weighted toward the ledger's best-tested corner**,
so the headline could not be read as the ledger's rate.

**Frame:** `FIXED` rows at `critical` or `landmine` severity, carrying a test, with `IMPORT-1*`
excluded — 52 rows across ten id families. **Sample:** 12, drawn round-robin across families so no
family dominates, seeded `random.Random(1818)`.

| | rows | share |
|---|---:|---:|
| **Guarded** | 12 | 100% |
| Not guarded | 0 | 0% |
| Unrunnable | 0 | 0% |

| row | the revert |
|---|---|
| `COACH-014` | the engine stops reading `auto_suggestable` and pre-selects everything |
| `CONTRACT-002` | page 03 hand-rolls its cascade again |
| `IMPORT-006` | `blank_cell_mask` flags every NaN, so it cannot tell the two blanks apart |
| `IMPORT-232` | drop the `_ORIGINAL_KEY` coalesce |
| `MINE-003` | reintroduce the constant-key cache |
| `MODELS-005` / `STATE-032` | the decorative cancel flag comes back |
| `RECORD-001` | `to_dict` drops `manuscript_text` |
| `STATE-055` | cohort runs restore as dicts, which `completed_runs()` filters out |
| `SWEEP-011` | reintroduce the global fallback pipeline slot |
| `T0-BUILD-004` | substring matching for clinical variables |
| `TEST-001` | the production cascade becomes a no-op |

## Reverts reconstructed twice: **zero of eleven**

Against **five of twenty-two** at L17. That difference is the interesting number, and it is not
luck — it is `FEATURE_PARITY.md`'s *"write the revert down in the row"* rule collecting.
`IMPORT-232`'s note says in as many words that reverting the obvious `drop(columns=[right_key])`
does **not** turn the test red and that the real guard is the `_ORIGINAL_KEY` coalesce. The row
told me where to look. Where a note recorded the guard, the first attempt worked; the L17 misses
were concentrated in rows whose notes recorded the *finding* and not the *fix*.

## Combined estimate

Three rows appear in both samples (`MINE-003`, `STATE-055`, `TEST-001`), so:

**31 of 33 distinct rows guarded — 93.9%**, with one genuinely weak guard (`STATE-056`) and one
unrunnable (`MODELS-001`, joined at L18 by `T0-LIVE-003`, the same torch test from the Tier-0 side).

The IMPORT-1xx corner is not the only well-guarded one. The two failures were both found in the
first sample and neither is of a kind the second sample could have surfaced — one is a
self-referential round trip, the other an environment skip — which is weak evidence that the
remaining failure modes are *categorical* rather than distributed, and that finding more of them
means enumerating shapes rather than drawing more rows.

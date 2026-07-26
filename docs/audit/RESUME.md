# Resume state — data import / multi-file / JSON audit wave

Written so this work survives the container being reclaimed. Everything needed
to continue is in the repository; nothing important is left in `/tmp` or in
agent transcripts.

## Where the work is

Branch: **`claude/tabular-ml-lab-review-nashh7-audit-fixes`**, branched from
`413671a` on `claude/tabular-ml-lab-review-nashh7`.

It is developed in a separate git worktree so the audits could keep measuring a
frozen tree while fixes landed:

    git worktree add <path> claude/tabular-ml-lab-review-nashh7-audit-fixes

That is an implementation detail — the branch is pushed and self-contained. To
resume anywhere, just check it out.

**Not yet merged.** The intended finish is a single merge into
`claude/tabular-ml-lab-review-nashh7` once the outstanding findings are fixed,
so the owner gets one coherent push rather than a half-finished one.

## The rule everything is judged against

The app may be **silent**, and it may **refuse**, but it must never **assert
something false**. `high` confidence is the only tier the UI pre-selects, so
`high` means the app is asserting; nothing uncertain may reach it. A
confidently-wrong answer is worse than a crash — a crash gets reported, a wrong
number gets published.

## Documents

| File | What it holds |
|------|---------------|
| `docs/FINDINGS_LEDGER.md` | Every finding and its disposition. The live checklist. |
| `docs/audit/ORIGINAL_48_FINDINGS.md` | The 48 confirmed findings verbatim, with repros. |
| `tests/test_stress_regressions.py` | One named test per closed finding. |

A finding is only `FIXED` when a test there would fail if the fix were
reverted. No finding is closed on a verdict alone.

## Verification commands

    python -m pytest tests/ -q                       # full suite
    python -m pytest tests/test_stress_regressions.py -q
    python -c "from streamlit.testing.v1 import AppTest; \
      at=AppTest.from_file('pages/01_Upload_and_Audit.py',default_timeout=120); \
      at.run(); print(at.exception)"                 # boot check

Driving the multi-file screen end to end needs project state injected into
`st.session_state['sp_projects']` — see `tests/integration/test_combine_ui.py`
for a working harness. `get_project_datasets` returns newest-first, so sort on
`upload_timestamp` when order matters.

## Audits that were in flight

Two background runs. If they did not finish, their findings are lost and the
hunt is worth re-running; the 48 are safe in `docs/audit/`.

- `wf_70254f26-494` — fresh exhaustive hunt: ten lenses over the multi-file
  path and JSON (key semantics, cardinality/fan-out, join modes, stacking, the
  Step 2 screen, JSON structure, JSON encoding, JSON→downstream, Import Doctor,
  state/scale). Every finding independently reproduced and adversarially judged
  twice.
- `wf_5446c57b-3f6` — re-runs all 48 against current HEAD and rules each
  fixed / still-broken / partially-fixed / changed-form, with an adversarial
  second opinion on every verdict in both directions.

Note both were launched against the FROZEN tree at `413671a`, so a
"still_broken" verdict on findings 5, 7, 11, 13 or 27 refers to the pre-fix
code — check the ledger before re-fixing.

## Known-good verification numbers

Useful as a smoke test that the fixes still hold:

- NHANES 2×2 (two cycles × two domains) through the real page:
  **200 rows, 200 distinct SEQN, 0 nulls in glucose**. Previously 400 (stack)
  or 144 (link on `age`).
- Repeated measures found at 2, 3, 4, 10 and 25 visits per subject; all `high`.
- Excel title-row scenario: `age.mean() == 36.5`, `subject_id` stays text.
- Key finder on 60×60 columns, 5k rows: ~1.5s.

## Still open

See the bottom of `docs/FINDINGS_LEDGER.md`. At the time of writing, the
outstanding work is whatever the two audit runs return, plus these known items
carried over from earlier:

- Page 01: transpose preview/apply consistency has a regression test gap
  (preview and commit now share one frame, but the Excel-sheet × transpose
  combination is untested).
- The remaining tail of the original 48 that neither I nor the adjudication has
  reached — the ledger table is the source of truth for which.

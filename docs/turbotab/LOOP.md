# Running this as an unsupervised loop

The ledger is designed to be worked by an agent without supervision. This file is the
operator's manual: the prompts to paste, the guardrails that make it safe to walk away,
and how to check what happened when you get back.

`data/findings.json` is the source of truth. `FINDINGS_LEDGER.md` is generated. The agent
edits the JSON through `tools/ledger.py` and regenerates the markdown — never the reverse.

```bash
python docs/turbotab/tools/ledger.py stats            # progress
python docs/turbotab/tools/ledger.py next --n 15      # next batch, as JSON
python docs/turbotab/tools/ledger.py set ID --status OPEN --note "..." --evidence "file:line"
python docs/turbotab/tools/ledger.py regen            # rewrite the markdown
python docs/turbotab/tools/ledger.py check            # schema guard; non-zero on violation
```

`check` enforces the rules that make the ledger trustworthy: no duplicate ids, no invalid
status, **no `FIXED` without a named regression test**, and no `PARTIAL` / `NOT-A-DEFECT` /
`WONTFIX` without a written reason. Run it before every commit.

---

## Loop 1 — verify Tier 1 (start here)

370 findings were produced by agents reading the repo at `fbe422a`, *before* PR #145 changed
`utils/test_lockbox.py` by +312 lines and added `utils/replay.py`. They are all marked
`UNVERIFIED`. Until they are re-checked they are research, not a backlog.

This loop **reads application code and writes only to `docs/turbotab/`**. That is what makes it
safe to run unattended. Paste this:

> You are working the TurboTab transition ledger on branch `TurboTab`. Read
> `docs/turbotab/README.md` and `docs/turbotab/LOOP.md` first.
>
> Your job is one thing only: **re-verify `UNVERIFIED` findings against the current code and
> record a disposition.** Do not fix anything. Do not refactor. Do not modify any file outside
> `docs/turbotab/`.
>
> Loop until `python docs/turbotab/tools/ledger.py stats` reports 0 `UNVERIFIED`:
>
> 1. `python docs/turbotab/tools/ledger.py next --n 15` to get the next batch.
> 2. For each finding: open the file(s) named in its evidence field and determine whether the
>    described problem still exists at HEAD. The evidence line numbers are from an older commit
>    — locate the code by symbol, not by line.
> 3. Record the disposition with `tools/ledger.py set`:
>    - `OPEN` — still exists. Update `--evidence` to current `file:line`.
>    - `PARTIAL` — partly addressed. `--note` must say precisely what remains.
>    - `NOT-A-DEFECT` — the finding was wrong. `--note` must say why.
>    - `FIXED` — no longer reproducible. `--test` must name the test that would catch a
>      regression. **If no such test exists, the finding is `OPEN`, not `FIXED`** — write the
>      test in a later loop.
> 4. If a finding is ambiguous or you cannot determine the answer from the code, mark it `OPEN`
>    with a note explaining the ambiguity. **Never guess `FIXED`.** A wrongly-closed finding is
>    worse than an open one.
> 5. After each batch: `tools/ledger.py regen`, then `tools/ledger.py check` (must exit 0), then
>    commit with a message naming the batch and the dispositions.
>
> Report at the end: how many of each disposition, and the three findings you consider most
> urgent.

**Expected shape:** ~25 batches. Each batch is a commit, so a crash costs at most one batch.

---

## Loop 2 — the live bugs

Only after Loop 1 finishes, and only if you want existing app code changed while you are away.
(If what you want is a *new app* to look at, skip to Loop 3.) Three
well-specified bugs, each with a clear gate. See `TRANSITION_PLAN.md` §01.

> On branch `TurboTab`, fix the three live bugs in `docs/turbotab/TRANSITION_PLAN.md` §01
> (`T0-LIVE-001`, `T0-LIVE-002`, `T0-LIVE-003`). Work them one at a time, smallest first.
>
> For each: write a failing regression test first, then the fix, then confirm the test passes and
> the existing suite still does. Then `tools/ledger.py set <ID> --status FIXED --test <test name>`,
> regen, check, and commit — one commit per bug, with the test and fix together.
>
> `T0-LIVE-001` gate: two different datasets in one session produce different PCA results.
> Reproduce the bug first and say so in the commit message — do not fix from the description alone.
>
> Do not touch anything in the freeze list in `TRANSITION_PLAN.md` §05.

---

## Loop 3 — the walking skeleton (this is how you get an app)

**Loops 1 and 2 do not produce an application.** Loop 1 produces a verified backlog; Loop 2 fixes
three bugs in the *existing* Streamlit app. Neither writes a line of TurboTab.

If the goal is to see the new design running against real data, that is a different job: a
**walking skeleton** — the thinnest end-to-end slice that uses the real engine, real uploaded data,
and the new interface, proving the architecture works before any of it is built out.

This is greenfield. It creates new files under `turbotab/` and touches no existing application
code, which makes it **safer to run unattended than Loop 2**, not riskier.

### Scope

One vertical slice: **upload a CSV → real structural diagnosis → real profile → real ranked
findings → recorded decisions**. No training, no jobs, no manuscript. Those come later; they are
not needed to prove the shape.

```
turbotab/
  engine.py     thin adapter over the real ml/ functions — import_doctor.diagnose,
                dataset_profile, triage.detect_task_type. No logic of its own.
  project.py    minimal AnalysisProject: dataframe handle, target, decisions[], findings[].
                Index LABELS as row identity (see TRANSITION_PLAN.md §02.2). Serializable.
  api.py        FastAPI: POST /project (upload), GET /project/{id},
                POST /project/{id}/decision, GET /project/{id}/findings
  web/index.html  the prototype, rewired to fetch from the API instead of its synthetic constants
  test_skeleton.py  upload a real CSV, assert findings come back non-empty and match a direct
                    engine call
```

### The prompt

> On branch `TurboTab`, build the walking skeleton described in `docs/turbotab/LOOP.md` §"Loop 3".
> Read `PRODUCT_VISION.md` and `ARCHITECTURE.md` first, and open
> `prototypes/interview-feed.html` in a browser to see the target interaction.
>
> Build it in this order, committing each step:
>
> 1. `turbotab/engine.py` — call the real `ml/` functions. Confirm first that they import with
>    Streamlit blocked (see the reproduce snippet in `ARCHITECTURE.md` §01). `ml.model_coach` is
>    transitively tainted via `utils/insight_ledger.py`'s `get_ledger()` singleton; cut that
>    singleton if you need the coach, and record it against `T0-*` in the ledger.
> 2. `turbotab/project.py` — minimal serializable project. Row identity is index **labels**.
> 3. `turbotab/test_skeleton.py` — a real CSV in, real findings out, asserted against a direct
>    engine call. Write this before the API.
> 4. `turbotab/api.py` — FastAPI over the project.
> 5. `turbotab/web/index.html` — copy the prototype and replace its synthetic constants with
>    `fetch()` calls. Keep the design language exactly; only the data source changes.
>
> **Gate:** I can start the server, drop my own CSV on it, and see real findings about *my* data
> rendered in the new interface. Put the exact run command in `turbotab/README.md`.
>
> Do not implement training, jobs, or manuscript generation. Do not modify anything outside
> `turbotab/` except the `insight_ledger` singleton if step 1 requires it. If the engine fights
> you, stop and write `docs/turbotab/BLOCKED.md` — that is a finding, not a failure.

### Why this is the right first build

- It **validates the riskiest assumption** — that the engine runs headless — with real code
  rather than a claim. The import test already found 7 tainted modules that grep missed.
- It is **demoable**. You can drop a real CSV on it and see the new design respond to your data.
- It **throws nothing away**: `engine.py`, `project.py` and the rewired frontend are all first
  drafts of production components, not scaffolding.
- It **finds the contract problems early**, while they cost a day instead of a month.

Expect it to be ugly and incomplete. That is the point of a skeleton — it walks, it does not run.

---

## Guardrails

Append this to any unsupervised prompt.

> **Hard rules.** Stay on branch `TurboTab`. Never push to `main`, never force-push, never open a
> pull request. Never modify `ml/import_doctor.py`, `ml/join_doctor.py`, `utils/combine*.py` or
> `pages/01_Upload_and_Audit.py` — they are frozen pending the open tail in
> `docs/FINDINGS_LEDGER.md`. Never edit `FINDINGS_LEDGER.md` by hand; it is generated. Never mark
> a finding `FIXED` without a regression test that would catch its return. Run
> `python docs/turbotab/tools/ledger.py check` before every commit and stop if it fails. Commit
> after every batch so nothing is lost. If you are blocked or something looks structurally wrong,
> stop and write what you found to `docs/turbotab/BLOCKED.md` rather than guessing.

The freeze exists because `docs/FINDINGS_LEDGER.md` has an unresolved tail on the multi-file
import path from two audit runs whose results were lost. Touching that code without those findings
means rebuilding on defects nobody has re-triaged.

---

## Checking in when you get back

```bash
git -C . log --oneline TurboTab | head -30
python docs/turbotab/tools/ledger.py stats
python docs/turbotab/tools/ledger.py check
git diff main...TurboTab --stat
cat docs/turbotab/BLOCKED.md 2>/dev/null
```

Three questions worth asking of the result:

1. **Does the `FIXED` count have tests behind it?** `check` enforces that a test is *named*; it
   cannot verify the test is any good. Spot-check two or three.
2. **How many went `NOT-A-DEFECT`?** A high rate means either the agents over-reported, or the
   verifier is being credulous. Read a few of those notes specifically — that is where a loop
   quietly goes wrong.
3. **Did anything land outside `docs/turbotab/` during Loop 1?** `git diff main...TurboTab --stat`
   answers it in one line. Loop 1 should touch nothing else.

---

## What not to hand an unsupervised loop

Some of the sequence in `TRANSITION_PLAN.md` §06 needs a human in the room:

- **S3, settling row identity.** Choosing labels over positions is a design decision with
  consequences across the whole project model. An agent can gather the evidence; it should not
  make the call alone.
- **S4, extracting the split block.** ~370 lines of untested, safety-critical logic. Extract it
  under supervision, with the characterization tests from S2 already in place.
- **S6, the Router.** New construction under a governing rule about what may be pre-selected.
  Design work, not loop work.

Loops are for verification, for well-specified fixes with clear gates, and for writing tests
against behaviour that already exists. They are not for decisions you would want to argue about.

But note the reframe in [`ROADMAP.md`](ROADMAP.md): those three items are not permanently
off-limits. They are blocked on **one decision each**. Make the decision and the execution behind
it becomes ordinary loop work. The roadmap lays out all twelve loops and the three decision gates
that separate them, plus a testable definition of done.

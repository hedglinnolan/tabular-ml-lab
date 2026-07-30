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

## Loop 1 — verify Tier 1 · ✓ DONE

**Completed 2026-07-27: 370 findings re-verified in 24 batches, 26 commits, nothing touched
outside `docs/turbotab/`.** Dispositions: 289 OPEN, 31 PARTIAL, 50 FIXED, 0 NOT-A-DEFECT.
The verifier cut four FIXEDs of its own on the no-test rule, corrected three over-reported
impacts in notes rather than closing them, refreshed four stale `classic-only` register
reasons, and flagged four stale Tier-0 rows outside its mandate (closed by the adjudicator
after verifying the named tests exist). The zero NOT-A-DEFECT rate was checked against the
three named over-reports — the verifier read the code, not the findings. The prompt below is
kept for the record.

**One-writer amendment, from this run:** the rule is now scoped to the shared data files
(`findings.json`, `register.json`, and their generated markdown). A docs-only commit from
another session may land mid-loop **only if** it touches none of those files and its commit
message says so — the L1 mid-loop design-language commit (`8667810`) is the precedent, and
the loop agent's clean rebase over it is the recovery path. Anything touching the data files
still waits for the loop to land. No exceptions there; that is the artifact this project has
already lost once.

370 findings were produced by agents reading the repo at `fbe422a`, *before* PR #145 changed
`utils/test_lockbox.py` by +312 lines and added `utils/replay.py`. They are all marked
`UNVERIFIED`. Until they are re-checked they are research, not a backlog.

This loop **reads application code and writes only to `docs/turbotab/`**. That is what makes it
safe to run unattended.

**Run it in a fresh session, not the builder's — verification and construction are different
hands.** This rule was added after L3–L9 landed, for two reasons that did not exist when the loop
was written. First, the builder now *wrote or moved* a large share of the code the findings
describe, and a builder verifying their own work reads intent where the job is to read code —
the same reason the workflow pattern never lets the finder be the judge. Second, both the loop
and the builder write the same data files (`findings.json`, `register.json`), and this project
has already lost a register to two writers on one artifact. One writer at a time: the loop runs
alone, the builder resumes after it lands.

Paste this into the fresh session:

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
> Two rules added after L4–L9 landed:
>
> - **"Guided avoids it" is never closure.** Streamlit never retires (Decision C), so a defect
>   that still exists in Classic stays `OPEN` even where the core or the Guided door has
>   structurally resolved it. Disposition those as `OPEN` with the note
>   `resolved-in-core; closes at L11 convergence of <page>` — that phrase is the queue for the
>   convergence loop, so use it verbatim.
> - **Tag, don't fix, siblings of known patterns.** If a finding is another instance of
>   "a blocker that only offers is not gating" (see `T0-ROUTE-001`), add
>   `sibling-of: T0-ROUTE-001` to its note and move on. They get one batched build, not
>   twenty inline ones.
>
> In the same pass, re-verify the register's `classic-only` claims (`data/register.json`) —
> several predate the Router and may be stale. Use `register.py set` with an updated reason;
> a `classic-only` row whose reason no longer holds is the register lying.
>
> Report at the end: how many of each disposition, the three findings you consider most
> urgent, and every row tagged `sibling-of`.

**Expected shape:** ~25 batches. Each batch is a commit, so a crash costs at most one batch.

---

## Loop 2 — the live bugs · ✓ DONE (folded into L7)

`T0-LIVE-001` through `T0-LIVE-005` are `FIXED` with named tests — see the ledger. The prompt
below is kept for the record only; do not run it.

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
> pull request. `ml/import_doctor.py`, `ml/join_doctor.py`, `utils/combine*.py` and
> `pages/01_Upload_and_Audit.py` are **frozen — see `TRANSITION_PLAN.md` §05 for the one statement
> of what that permits and the gates that lift it.** Never edit `FINDINGS_LEDGER.md` by hand; it is
> generated. Never mark a finding `FIXED` without a regression test **verified to fail when the fix
> is reverted** — see `FEATURE_PARITY.md`, "the revert probe". **First command in a fresh clone:**
> `git config core.hooksPath .githooks`. Commit after every batch so nothing is lost. If you are
> blocked or something looks structurally wrong, stop and write what you found to
> `docs/turbotab/BLOCKED.md` rather than guessing. **Domain science comes from
> `docs/turbotab/research/`, never from recollection** — where a research file and your memory
> disagree, the file wins, and a number marked `[verify-at-build]` may not ship as a hard-coded
> constant. See "Loops that build a domain pack" below.

**Run the documented setup path, or find out that nobody has.** `Makefile` names
`./venv/bin/python` and nothing had created it in a long time — long enough that
`tests/test_american_spelling.py`'s skip list had `.venv` and not `venv`, so the
gate died with a `UnicodeDecodeError` on a compiled dependency the first time
anybody followed the instructions. The gate was not wrong about the prose; it
never reached the prose.

A setup path is a claim like any other, and it decays the same way: silently,
while the people who already have working environments keep working. So the
documented commands are something to *run*, not only to keep accurate — and the
cheapest form of that is building both environments from scratch at the start of
a loop that touches either.

**The three gates are a hook, not an instruction.** `.githooks/pre-commit` runs `ledger.py check`,
`register.py check` and `tests/test_american_spelling.py`, and refuses the commit on any failure.

This replaced the line that used to sit in the guardrails above — *"run `ledger.py check` before
every commit and stop if it fails"* — which enforced nothing. Commit `8127101` went out with the
spelling test red because the agent ran all three gates, saw the third fail, and committed anyway:
the gate was chained to the commit with a newline instead of `&&`, so a non-zero exit did not stop
the sequence. The instruction was followed and the gate still did not hold. **An instruction a tired
agent can skip by punctuation is not a gate**, which is this project's own rule — *make silence a
test failure* — applied to its own process.

`core.hooksPath` is local config and git does not version `.git/hooks`, so the hook file is
committed under `.githooks/` and each clone points at it once. That one command is the only part
still carried by discipline, so it is the first line of the guardrails rather than a footnote.
Bypass with `git commit --no-verify`, and say why in the message.

The freeze exists because the multi-file import path had an untriaged defect tail. **Its results
were never lost** — they are in `docs/audit/`, committed, and were there the whole time this file
said otherwise. That error cost a loop of rediscovery, and it is the reason for the
ephemeral-pointer rule in `FEATURE_PARITY.md`.

The rule, its permissions and the three gates that lift it are stated **once**, in
`TRANSITION_PLAN.md` §05. Do not restate them here: this file said "never modify" while §05 said
"engine-move-only — no signature changes", which are different rules, and a reader following the
stricter one could not do the work §05 permits.

---

## Loops that build a domain pack

The four research threads in `docs/turbotab/research/` are **3,602 lines and are the authoritative
source** for every pack detector, coaching sentence, threshold and figure specification. They are not
background reading. A loop that builds pack content without citing them has invented its content,
which is the failure this whole apparatus exists to prevent.

Three problems have to be solved together, and they are solved differently.

### 1 · Volume — bounded reads, named in the task block

Nobody holds 3,602 lines. **The task block names the file and the section**, and the agent reads that
slice:

> Implement the Atwater reconstruction check as an import-doctor detector. **Source of truth:
> `docs/turbotab/research/NUTRITION_PACK.md` §01, "The Atwater reconstruction check" — read that
> section before writing anything.** The ratio table is normative, including the drift-with-total-energy
> row, which is the mixed-units-across-rows case.

This is cheap, it works with a nearly-full context window, and it makes the citation the *input* to the
work rather than a footnote added afterward. **A task block that says "build the nutrition detectors"
without section pointers is a malformed task block** — it has handed the agent the job of deciding what
the science is.

### 2 · Provenance — a field, and a gate

Every pack advisory, detector and figure spec carries a **`source`** field naming file and section, and
an **`evidence_status`** of `SETTLED` / `CONVENTION` / `DISPUTED`.

A checker verifies that the named section exists in the named file, and that `evidence_status` is
present and one of the three. **It runs in `.githooks/pre-commit` beside the other three gates.**

What this does and does not buy, stated plainly, because the ledger's own gate has the same shape:
`ledger.py check` enforces that a test is *named* and cannot verify the test is any good. This checker
enforces that a source is *named and resolvable* and cannot verify the claim is faithful to it. That is
still most of the value — it makes fabrication require deliberate effort rather than inattention, and
it makes spot-checking one line of work instead of a research project.

**The rule behind the field.** Where the research file and the model's recollection disagree, *the file
wins.* The files were built under a blocked egress proxy and they say so; a threshold in the file is a
recorded, checkable claim, and a threshold from memory is neither. This is the same reason the app
prefers a stated basis to a confident one.

### 3 · `[verify-at-build]` is a hard stop, not a comment

The research files mark specific numbers as unverified — QC-RSD thresholds, the D-ratio 50% criterion,
software defaults that change between versions, the Goldberg/Black algebra, the DRI tables.

**A `[verify-at-build]` number may not ship as a hard-coded constant until it is verified against the
primary source.** Until then it ships one of two ways: as an `offered` item the user chooses with the
uncertainty stated, or not at all. Shipping a wrong number is the worst failure mode a pack has — the
files say so themselves — and it is worse than shipping nothing, because a wrong number arrives with
the app's authority behind it.

Where a fact is genuinely unavailable, that is a `BLOCKED.md` entry, not a guess.

### 4 · Sequencing — one pack end-to-end, then three that are mostly content

The natural reading of "build each domain out fully" is four parallel verticals. **That is the wrong
shape and it will cost a loop.** The four packs share far more than they differ: the figure spec, the
annotation engine, the badge rendering, the checklist engine, the hard-stop class. Built four times,
they are built four different ways; built once at the end, the first three packs get rewritten.

So: **one pack end-to-end first, as the reference implementation.** Its job is to discover the
abstractions — that is the deliverable, alongside the pack. The remaining three are then mostly
content-loading against a proven spine, which is the regime where per-domain loops actually work.

**Nutrition goes first**, for a reason that has nothing to do with the science being easier:

- It is the product owner's own field, so **he can adjudicate whether the content is right** — and at
  the reference-implementation stage that review is the point, because we are finding out whether the
  pack architecture can carry real domain content at all.
- It is the NHANES data already being driven, so the fixtures are real rather than synthetic.
- It exercises the widest spread of primitives in one pack: complex survey design, repeated measures,
  compositional closure, an estimand that must be named, and — valuable and rare — **a refusal**, in
  the app declining to compute a prevalence of inadequacy from an AI.

That last one is worth stating separately. A pack that can only *add* findings has not been tested. The
refusal path is where the architecture either holds or doesn't.

### 5 · The anti-pattern audit runs ahead of all of it

Across the four files there are on the order of 150 named anti-patterns, each a specific checkable
behavior. They are content for the packs **and** a conformance suite against the engine that exists
today, and the first pass already found a live defect: the app recommends class weighting and SMOTE
(`ml/dataset_profile.py`, `ml/eda_recommender.py`) and asserts it into the generated manuscript
(`ml/narrative_engine.py`), against the primary literature. See `DOMAIN_SCIENCE.md` §03b.

**A defect the research already found in shipped code outranks a pack feature that has not been built
yet**, because the first is the app asserting something false and the second is only the app being less
useful than it could be. The audit is also good loop work by this file's own standard — well-specified,
gated, and testing behavior that already exists.

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
against behavior that already exists. They are not for decisions you would want to argue about.

But note the reframe in [`ROADMAP.md`](ROADMAP.md): those three items are not permanently
off-limits. They are blocked on **one decision each**. Make the decision and the execution behind
it becomes ordinary loop work. The roadmap lays out all twelve loops and the three decision gates
that separate them, plus a testable definition of done.

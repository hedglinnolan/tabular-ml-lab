# The road to done

`LOOP.md` covers the first three loops. This file covers all of them, and says what "done"
means so the finish line is a thing you can test rather than a feeling.

The organising idea: **loops are separated by decision gates.** Most of this work is loopable.
What is *not* loopable is a handful of design decisions — and once you make each one, the
execution behind it becomes ordinary loop work. So the plan is not "autonomous until it gets
hard," it is "autonomous between checkpoints, and here are the checkpoints."

---

## Definition of done

TurboTab is finished when all four are true:

1. **Parity, permanently.** For the same CSV and the same choices, both front doors produce the
   same numbers — same splits, same metrics, same lockbox rows, same manuscript claims. Enforced
   by the parity harness running in CI, not by inspection. This condition never retires, because
   Streamlit never retires.
2. **The interaction model is real.** One card per step, preview before apply, deferral that
   resurfaces, a stale cascade you can see, and flagged exhibits that land in the export. The
   prototype defines the target; the app must do it against live data.
3. **One core.** No domain logic remains in `pages/` or in the TurboTab frontend, enforced by the
   import-boundary test. Every rule is implemented once.
4. **The invariants hold.** Every rule in `ARCHITECTURE.md` §02 has a test that fails if it is
   broken. The lockbox in particular: sealed before exploration, sealed once, opened once.
5. **The ledger is closed.** Every finding is `FIXED` (with a named test), `NOT-A-DEFECT`, or
   `WONTFIX` (with a reason). Zero `UNVERIFIED`, zero `OPEN`.

Anything short of all five is a milestone, not a finish.

---

## The map

```
  L1  verify ledger ──────────┐
  L2  live bugs               ├─ can run in any order, no decisions needed
  L3  walking skeleton ───────┘
                     │
              ◆ DECISION A — row identity
                     │
  L4  characterization tests
  L5  AnalysisProject + invalidation DAG
  L6  extract split block + step state machine
  L7  detaint the engine, add the job queue
                     │
              ◆ DECISION B — Router gating policy
                     │
  L8  Router, EDA only
  L9  feed frontend, one step per loop  ← the long one
  L10 parity harness + manuscript chain
                     │
              ◆ DECISION C — ANSWERED: never delete
                     │
  L11 converge Streamlit onto the core (page by page, lazily)
  L12 packaging
```

**Rough scale.** L1–L3 are days. L4–L7 are weeks. L9 is the long pole — eleven steps of
interaction, each its own loop. L10–L12 are weeks. Treat this as a months-long project with
usable milestones throughout, not a sprint.

---

## Decision gates

Three decisions only you can make. Each is one question, answerable in an afternoon, and each
unblocks a run of autonomous work. **Decision C is already answered** — see below; it reshapes
the back half of the roadmap.

### ◆ Decision A — row identity

**Question:** are rows identified by index *labels* or by *positions*?

Today both conventions exist (`TRANSITION_PLAN.md` §02.2) and they agree only while the index is
a pristine `RangeIndex`. Everything in `AnalysisProject` depends on the answer.

**Recommendation: labels.** They survive filtering, cohort selection and row-dropping repairs;
positions do not. The cost is that every consumer must be audited for `.iloc` on a stored index.

**Unblocks:** L4, L5, L6.
**Evidence to gather first (loopable):** an agent can enumerate every site that stores or consumes
a row index and report which convention each assumes. Ask for that before deciding.

### ◆ Decision B — Router gating policy

**Question:** may the Router ever skip a question, or only reorder and recommend?

The governing rule says `high` confidence is the only tier that may pre-select, and
**auto-advancing an interview is pre-selection.** The coach today emits no `blocker` severity and
has no confidence tier of its own (`TRANSITION_PLAN.md` §02.5), so gating is not merely unbuilt —
its semantics are undefined.

**Options:** (a) reorder and recommend only, never skip — safest, most verbose; (b) skip only
where a `high`-confidence finding makes a question moot, with the skip visible and reversible in
the transcript; (c) full gating with a new severity model.

**Recommendation: (b).** It honours the rule, it is testable, and the transcript records the skip
so the manuscript can still describe what happened.

**Unblocks:** L8, and the shape of L9.

### ◆ Decision C — cutover · **ANSWERED: Streamlit is never deleted**

Users depend on it. That is a fixed constraint, not a preference, and it changes the shape of the
project — for the better, once the framing changes.

**The project is no longer "replace the app." It is "extract the core so the UI becomes a choice."**

That was always the more valuable objective. A shared core is what makes the desktop packaging
possible, what makes an All of Us deployment possible, and what makes TurboTab possible. Streamlit
stops being the thing being replaced and becomes **one of several front doors.**

But "keep both" is genuinely the failure mode that kills rewrites — *when both are full
implementations.* The failure is never two UIs; it is **two implementations of the domain logic**,
which drift until a number differs and nobody can say which is right. The rules in
"Two front doors" below are what prevent that, and they are non-negotiable.

**Consequences for this roadmap:**

- L11 changes from *cutover and delete* to *converge Streamlit onto the core*.
- The parity harness (L10) becomes **permanent CI infrastructure**, not a one-time gate.
- The extraction loops (L4–L7) become the highest-value work in the project rather than
  preparation for it — they are now the deliverable.
- Logic trapped in `pages/` becomes more urgent, not less: **logic in a UI layer cannot be shared,
  so every line still in `pages/` is a line that must be written twice.**

---

## Two front doors, one core

The rule that makes keeping both sustainable:

> **No domain logic lives in any UI layer. Both Streamlit and TurboTab are thin views over the
> same Engine, Project, Record and Router.**

Six policies enforce it.

1. **One core, no forks.** `ml/`, `models/`, `AnalysisProject`, the Record and the Router are
   shared. Neither UI may hold a private copy of a rule, a default, or a computation.
2. **Enforce it with a test, not a convention.** Add an import-boundary test asserting that
   nothing under `pages/` or the TurboTab frontend defines domain logic, and that the core never
   imports either UI. Conventions decay; a red test does not.
3. **Streamlit is feature-frozen at the UI level, and continuously improved underneath.** It gets
   no new screens. It inherits every core fix automatically — which means today's cache-poisoning
   bug and every landmine in the ledger get fixed once, for both.
4. **Pages migrate to the core lazily.** Do not schedule a 19,835-line refactor. When you touch a
   page for any reason, that page's logic moves to the core first. Touching it is the trigger.
5. **New capability lands core-first, TurboTab-second, Streamlit-only-if-cheap.** The core is
   where the feature lives; the front doors decide whether to expose it.
6. **Parity runs in CI forever.** Same CSV, same scripted choices, both front doors, diff the
   outputs. Not a cutover gate — a permanent guard against the drift that makes two apps
   unmaintainable.

**What this costs.** Converging Streamlit is more work than deleting it: eventually every page
must consume the core rather than `st.session_state`. It is also work you were partly doing
anyway — extraction is the migration — and policy 4 spreads it over the natural cadence of
maintenance rather than demanding a single dangerous refactor.

**What this buys.** Your existing users keep the app they know, and it gets *better* while they
keep it — because the fixes land underneath them. TurboTab becomes an opt-in second door rather
than a forced migration, and adoption decides its fate instead of a cutover date. And the same
core reaches a Workbench VM or a Docker deployment without a third implementation.

---

## The loops

### L4 · Characterization tests — autonomous

Pin current behaviour before anything moves. Golden-output tests on: the split block (all four
branches), `reset_downstream_results()` called for real, the lockbox seal/redraw signature, every
model wrapper against *real* wrapper objects rather than a bare `Ridge`, and the manuscript
composed from a fixed project.

Port `tests/test_insight_id_integrity.py` first with a non-zero-count assertion, so it cannot pass
vacuously after the renames that are coming.

**Gate:** deliberately break the cascade and the suite goes red.

### L5 · AnalysisProject and the invalidation DAG — autonomous after Decision A

The typed, serializable project model. Must carry: per-model pipeline **specs** (not fitted
objects), the active cohort filter as a first-class field, the lockbox, and a declarative DAG that
can express **partial** invalidation — `reset_downstream_results(clear_feature_engineering=False)`
is a real call.

**Gate:** the new DAG reproduces both existing cascade implementations, including the hand-rolled
one in `pages/03`, and round-trips through `to_dict`/`from_dict` with no loss.

### L6 · Extract the split block and the step state machine — autonomous, needs L4

`pages/06:380-760` becomes a real `ml/splits.py`. `utils/theme.py:685`'s ten predicates become the
project's readiness model.

**Gate:** a headless script runs CSV → trained models with `streamlit` blocked at import.

### L7 · Detaint the engine, add the job queue — autonomous

Cut the `get_ledger()` / `get_provenance()` singletons and the `st` reads in `test_lockbox`. That
alone detaints `ml.model_coach` and unblocks Router work (`ARCHITECTURE.md` §01). Then the job
queue, with an explicit RNG passed to every worker — the global-seed mutation in
`models/nn_whuber.py`, `utils/seed.py` and `utils/datasets.py` is safe single-threaded and
silently corrupting under a worker pool.

**Gate:** the import-blocker test passes for all 42 core modules; two concurrent jobs produce
identical results to two sequential ones.

### L8 · The Router, EDA only — autonomous after Decision B

New construction, not a move. Lift triggers out of the pages, add a severity model, implement the
gating policy from Decision B. Narrowest possible slice: exploration only.

**Gate:** for a fixed project, the chosen next question is derivable from the record alone, and
the same project always yields the same question.

### L9 · The feed frontend — autonomous, one step per loop

The long pole. Eleven steps of interaction, each its own loop with its own gate. Order by
specification quality, not by pipeline order: **Preprocess and Train first** (well-specified,
jobs pay off most), **Upload & Audit last** among the early candidates (its defect backlog is
open — see the freeze in `TRANSITION_PLAN.md` §05).

Each step's gate is the same shape: *this step works against real data, its decisions appear in
the transcript, and its exhibits reach the export.*

### L10 · Parity harness and the manuscript chain — autonomous

The harness is the most valuable thing in this list and should be built early if you can afford
it. Same CSV, same scripted choices, both apps, diff the outputs: splits, metrics, lockbox row
sets, manuscript text. Any divergence is a bug in exactly one of them, and the harness says which.

Also split `ml/publication.py` (1,887 loc, 32 `st` refs) into logic and delivery, which detaints
`latex_report` and the rest of the manuscript chain.

**Gate:** parity green on at least three real datasets of different shapes — wide, longitudinal,
and multi-file.

### L11 · Converge Streamlit onto the core — autonomous per page, supervised overall

Not a cutover. Page by page, replace each Streamlit page's private logic with calls into the
shared core, keeping its UI intact. Users see nothing change; the duplication disappears
underneath them.

Order by the freeze list and by how much logic each page traps: `06_Train_and_Compare` (382 logic
markers) and `10_Report_Export` (24 local functions) first, `01_Upload_and_Audit` only after
`docs/FINDINGS_LEDGER.md`'s open tail closes. `11_Theory_Reference` is content, not logic —
migrate it as data.

**Gate per page:** parity green for that page's outputs before and after, and the import-boundary
test still passes.

**Do not delete anything.** The one deletion worth making is `utils/dataset_db.py` — 797 lines,
zero importers, superseded — and even that goes in its own commit.

### L12 · Packaging — autonomous

Tauri or pywebview around the FastAPI backend. Signed installer (Apple Developer ID plus
notarization, Authenticode on Windows) so the Gatekeeper and SmartScreen prompts go away, and the
Python runtime bundled so the 600 MB first-launch download does too.

This is where the *original* complaint — launching the app is painful — actually gets fixed. It
was never a Streamlit problem; it was always a packaging problem.

---

## What stays supervised, and why

Not "too hard for an agent" — **decisions where being wrong is expensive and the agent has no way
to know it is wrong.**

- The three decision gates above.
- Any change inside the freeze list until `docs/FINDINGS_LEDGER.md`'s open tail closes.
- Deleting anything — and under Decision C, deletion is almost never the answer. An agent that
  removes `utils/theme.py` as "just styling" also removes the step state machine.
- Accepting parity failures. If L10 reports a divergence, a human decides which side is right.
  Sometimes it will be the new one.

## How to keep the loops honest

Three habits, all cheap:

- **Every loop ends with a gate that is a command, not an opinion.** If you cannot run it, it is
  not a gate.
- **Every loop updates the ledger.** `tools/ledger.py check` runs before every commit, and the
  `FIXED`-needs-a-test rule is what stops a loop from declaring victory.
- **Re-verify after any large merge.** PR #145 invalidated 370 findings overnight. The
  import-blocker snippet in `ARCHITECTURE.md` §01 and `tools/ledger.py stats` together take under
  a minute and tell you whether the ground moved.

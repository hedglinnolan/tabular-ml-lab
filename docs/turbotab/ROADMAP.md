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
  L1  verify ledger ──────────┐            [not started — 370 UNVERIFIED]
  L2  live bugs               ├─ ✓ DONE (folded into L7; T0-LIVE-001..004 FIXED)
  L3  walking skeleton ───────┘  ✓ DONE
                     │
              ◆ DECISION A — ANSWERED: identity barrier (reframed)
                     │
  L4  characterization tests   ✓ DONE
  L5  AnalysisProject + DAG    ✓ DONE (incl. serialization guard, archive port)
  L6  split block + readiness  ✓ DONE
  L7  detaint + job queue      ✓ DONE (45/45 headless; RNG serialization; honest cancel)
                     │
              ◆ DECISION B — ANSWERED: skip only when a high-confidence finding
                     │           makes the question moot; visible + reversible
  L8  Router, EDA only  ← IN PROGRESS — step 1 (baseline) ✓ DONE
              │
       ◆ ROUTING VALUE CHECK — criteria PRE-REGISTERED in
              │    VALUE_CHECK_PREREG.md, frozen before Router code;
              │    editing it after Router code exists = failing it
              │
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

### ◆ Decision A — row identity · **REFRAMED — the original question was insufficient**

I first posed this as *labels or positions?*, i.e. two components disagreeing on a convention.
The preview-engine loop found something the framing missed: **four of the nine repair kinds
renumber the rows.** `promote_header`, `drop_empty_rows` and `drop_rows` all end in
`.reset_index(drop=True)`; `melt_repeated` rebuilds the index outright.

That is not two components disagreeing. It is **one repair invalidating the convention
mid-analysis**, and no agreement between components prevents it. Picking labels does not help if
a repair renumbers the labels: a sealed lockbox survives the repair and afterwards names
*different rows*. PR #145 removed one such `reset_index` in `apply_plausibility_filter`; these
four are the rest of the class.

**The real question:** *what is a row's identity across a repair?*

Three answers, in increasing badness:

- **(a) Repairs preserve identity.** No operation after identity is established may renumber.
  Drop-style repairs keep the surviving rows' original labels; that is a one-line change per site.
- **(b) Renumbering repairs record an explicit remapping**, and every row-keyed artifact is
  rewritten through it. Correct but invasive — every consumer must remember to apply remaps, and
  forgetting is silent.
- **(c) Renumbering repairs invalidate row-keyed artifacts** and force a re-seal. This breaks the
  *sealed once* invariant, which is load-bearing for the manuscript's honesty claim.

**Recommendation: (a), with an explicit identity barrier.**

> There is a point in the pipeline where rows acquire identities — the moment the lockbox is
> sealed. Operations that *cannot* preserve identity (`promote_header`, `melt_repeated`: they
> change what a row *is*) are **pre-barrier structural repairs** and may only run before it.
> Operations that merely remove rows (`drop_empty_rows`, `drop_rows`) preserve survivor labels
> and may run on either side.

This turns a convention into a phase rule, which is testable rather than remembered:

- a test asserting no post-barrier operation changes the index of any surviving row;
- a test asserting the pre-barrier repairs are unreachable once the lockbox is sealed;
- the preview engine already reports renumbering **by content rather than by fix kind**, which is
  the right detector — a footer drop on a clean `RangeIndex` is genuinely safe, a mid-frame drop
  is not.

**Unblocks:** L4, L5, L6 — and it is now the single highest-value decision in the project,
because `AnalysisProject`, the lockbox and cohort runs all key by row.

### ◆ Decision B — Router gating policy

**Question:** may the Router ever skip a question, or only reorder and recommend?

The governing rule says `high` confidence is the only tier that may pre-select, and
**auto-advancing an interview is pre-selection.** The coach today emits no `blocker` severity and
has no confidence tier of its own (`TRANSITION_PLAN.md` §02.5), so gating is not merely unbuilt —
its semantics are undefined.

**Options:** (a) reorder and recommend only, never skip — safest, most verbose; (b) skip only
where a `high`-confidence finding makes a question moot, with the skip visible and reversible in
the transcript; (c) full gating with a new severity model.

**Recommendation: (b).** It honors the rule, it is testable, and the transcript records the skip
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
6. **Every capability has a register row.** `core` / `both` / `classic-only` / `guided-only`,
   with a reason. `guided-only` exists because capability flows both ways: preview-before-apply
   and undo are Guided-first, and Classic today applies repairs from a single button with no diff
   and no undo — the blind consent the vision argues against. **Convergence is bidirectional.**
   A capability with no row fails the register check — see [`FEATURE_PARITY.md`](FEATURE_PARITY.md).
   Without this, lazy migration plus per-feature exposure decisions lose features silently.
7. **Parity runs in CI forever.** Same CSV, same scripted choices, both front doors, diff the
   outputs. Not a cutover gate — it is how the product keeps a promise made to researchers, that
   a manuscript from either door describes the same science (`PRODUCT_VISION.md` §04b).

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

Pin current behavior before anything moves. Golden-output tests on: the split block (all four
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

**Design constraint from `PRODUCT_VISION.md` §04b:** the project model must be sufficient for
**both** doors, not shaped to the Guided door's convenience. If Guided needs a field Classic
cannot populate, the state model has forked and "same modeling process" stops being true.

**Gate:** the new DAG reproduces both existing cascade implementations, including the hand-rolled
one in `pages/03`, and round-trips through `to_dict`/`from_dict` with no loss. Stretch gate, and
the one that proves the architecture: **a project started in one door can be opened in the other,
mid-analysis, with no loss of state or change in results.**

**On the stretch gate and persistence — there is no conflict, and prior art exists.** Two
requirements were conflated:

- *Switching doors mid-analysis* needs no persistence at all. Both doors are views over one
  running core, so the project is in memory and switching is opening a different view of it.
- *Resuming tomorrow* needs durability — and `utils/session_manager.py` already does it correctly:
  a zip of decisions and inputs (config, widget state, ledger, provenance, **lockbox labels**,
  cohorts, FE recipe, probe results), with pickles refused and derived artifacts dropped.

`_NEVER_PERSIST` is that drop-list, not a no-disk rule. **Port the archive schema; do not redesign
it.** Add one test that turns the invariant into a guard: *no cell value from the loaded frame
appears anywhere in a serialized project.* That is what "never persist participant data" means
operationally, and it is checkable.

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

**This is the differentiator.** The Guided door's whole justification is that it asks better
questions in a better order (`PRODUCT_VISION.md` §04b). If routing is thin, TurboTab is a reskin
and the eleven loops of L9 are wasted on it.

**Gate:** for a fixed project, the chosen next question is derivable from the record alone, and
the same project always yields the same question.

**◆ ROUTING VALUE CHECK — do this before starting L9.** Take three real datasets of different
shapes and measure, against the Streamlit path:

| Claim | Measurement |
|---|---|
| Fewer irrelevant questions | count questions asked per dataset, both doors |
| Ordering by consequence | does the first question match what the coach ranks highest? |
| Findings drive disclosure | how many questions appear *because* of a finding rather than a pipeline stage? |
| Deferral closes | does every deferred item resurface at a step that can act on it? |

If the Guided door does not measurably win on at least the first and last, **stop and rethink the
Router before building the feed.** Discovering this after eleven step-loops is the single most
expensive mistake available in this plan.

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

**Gate per page:** parity green for that page's outputs before and after, the import-boundary test
still passes, and every capability the page held is registered `core` or `classic-only` with a
reason (`FEATURE_PARITY.md`).

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

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
  L1  verify ledger ──────────┐  ✓ DONE — 370 verified: 289 OPEN, 31 PARTIAL, 50 FIXED, 0 N-A-D
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
  L8  Router, EDA only      ✓ DONE — messy-clinic: coverage 1/9→9/9
              │                       with questions 34→9, irrelevant 25→0
       ◆ ROUTING VALUE CHECK — ✓ PASSED against the frozen prereg
              │    (one edge ambiguity adjudicated in
              │     VALUE_CHECK_ADJUDICATION.md; prereg unedited)
              │
  L9  feed frontend, one step per loop  ← NEXT — the long one.
              │    The routing harness becomes a STANDING rail: re-run at
              │    every L9 step. When the pull palette lands, the harness
              │    must distinguish pushed questions (thresholds bind) from
              │    pull affordances (offered, not asked, not counted).
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

**Refinement from the L8 implementation — the fact/choice distinction, now canonical:**

> A `high`-confidence finding can settle a question of **fact** — "is this categorical?" — because
> the engine is certain and the transcript can state it. It can never settle a question of
> **choice**. Whether to apply a repair is the user's decision however confident the engine is,
> because applying without preview is the blind consent the preview exists to end.

Repairs are always asked; only detected facts are skippable. And with the palette came the second
clause: **a pull affordance may never be skipped or deferred — ignoring one has to be free, or it
was never pull.**

**Third clause, from the Explore step's register dispositioning — blocker severity:**

> A `blocker`-severity finding is a question of **consequence**. It is always pushed, never
> offered — *a blocker that only offers is not gating.* The tool does not hard-refuse to proceed
> (the user may know the flagged column is legitimate), but passing an unresolved blocker requires
> an **explicit recorded acknowledgment**, and that acknowledgment flows into the record so the
> manuscript can carry it as a limitation. Silence past a blocker is impossible; overriding one is
> a decision the transcript owns.

Fact → skippable at `high` confidence. Choice → always asked. Consequence → always asked, and
exit past it unresolved is itself a recorded decision. This is the routing constitution in three
clauses, and `router.audit()` enforces all of it before any run is scored — a run that breaks a
rule has no number, it has a failure.

Two refinements from the T0-ROUTE-001 build, now binding:

- **Certainty does not make a question of consequence moot.** Being certain a column leaks is a
  reason to ask, not a reason to stay quiet — the one place where `high` confidence must not
  skip. This is why consequence could never fit under fact.
- **Blockers rank first.** A blocker third in a list of nine is a blocker in name only. Ordering
  is part of the gate.

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

---

## The lockbox constitution · **ANSWERED**

The routing constitution governs which questions get asked. This one governs **what the app is
allowed to know, and when.** It exists because the seal is the load-bearing claim of the whole
product: every held-out number, every manuscript metric, rests on the assertion that the test rows
were never seen. `IMPORT-020` proved that assertion could be false while a lock icon rendered
cleanly, which is the governing rule's own failure at the deepest point in the app.

Grounded in TRIPOD+AI (eligibility reporting), Harrell RMS (extrapolation), sklearn's pipeline
doctrine and Kapoor & Narayanan's leakage taxonomy (fold-local fitting), Sisk/Sperrin/van Smeden
and Groenwold (missingness under prediction vs inference), and Steyerberg (outcome excluded from
imputation).

### 01 · The pre-seal sequence is fixed

> **load → structural repairs and the impossibility pass → grain → eligibility → SEAL → EDA**

Nothing may be resequenced. Two of those steps are pre-seal for reasons that are easy to miss:

- **The impossibility pass**, not for leakage reasons — setting a physiologically impossible value
  to missing is row-local and leaks nothing — but because a stratified or grouped split computed
  over corrupted values is a worse split, and impossible entries are normally an exclusion that
  changes N, which belongs in the flow diagram before anything is sealed.
- **Grain**, because the seal cannot be drawn correctly without it. See §02.

### 02 · Grain is asked, never inferred

> **"Can one person appear in more than one row?"**

This is the same question multi-file assembly asks (`IMPORT-005`, `IMPORT-015`) and the same one
the lockbox needs. It is asked **once**, pre-seal, and both consumers read the one recorded answer:
a project that arrives through assembly has already answered it and the seal inherits it.

The heuristics (`detect_repeated_subjects`, `rank_grouping_candidates`) are **demoted from source
of truth to two lesser roles**: a *suggestion* offered to the human, and a *contradiction detector*
when the human's answer disagrees with the data's shape. A user who says "one row per person"
while a column repeats three times per value is evidence that somebody is wrong — that earns an
interruption, by the same rule that governs join drops: escalate on evidence of error, never on
the magnitude of a consequence.

Name lists and ratio bounds cannot close this and must not be tuned as though they could. The
engine was guessing at something the user simply knows.

### 03 · The seal states its own basis — three states, never two

> **Grouped by column X · repetition found but grouping abandoned · undetermined**

`undetermined` is first-class: persisted in the lockbox record (never as `group_col: None`, which
a consumer cannot tell from a verified cross-sectional seal), asserted by a test, and **never
rendered as a clean lock.** The failure `IMPORT-020` names is not that detection is hard — it is
that failure to detect was indistinguishable from success.

The asymmetry that settled it: `IMPORT-021` leaks too, and closes anyway, because it *says so*.
Leaking and disclosing is the governing rule's **refuse** branch. Leaking behind a lock icon is
its **assert something false** branch. An undetermined seal is an advisory with exploratory
labeling, not a hard block — a user who genuinely does not know their own data's shape should get
honest numbers, not a locked door.

### 04 · Eligibility and robustness trims are different objects

Two operations that look identical in a spreadsheet and are not:

| | Eligibility criterion | Robustness trim |
|---|---|---|
| What it says | who the model is *for* | how the fit is *stabilized* |
| Applied to | the whole dataset, **pre-seal** | the training partition only, **post-seal** |
| Changes N | yes — reported in the flow diagram with its reason | no |
| Test set | obeys it | never touched |

TRIPOD+AI names continuous-variable restrictions ("e.g. age range") as an eligibility item
reported in participant flow. The eligibility question is asked in **scientific terms** — *does
your research question restrict the outcome range?* — with the target's distribution **withheld**,
because an eligibility criterion comes from the research question and not from the histogram. If a
user needs to see the shape to decide where to cut, that is data-driven cohort selection, which is
its own publishable bias. The app may show what is needed to answer *"is this data corrupted?"*
(observed min/max, impossible-value flags) and not what is needed to answer *"where should I cut?"*

**"Also trim the test set to match" is permanently off the menu.** A user who truly wants the
narrower population is routed back to the pre-seal eligibility question, which requires a re-seal
and is therefore its own hard, logged decision.

### 05 · The extrapolation obligation fires at the report, not at the trim

A train-only trim is a **legitimate choice**, so it does not earn a blocker — friction is spent
where an operation is almost certainly an error, and this one is not. What is illegitimate is
reporting a single aggregate metric afterward as though nothing happened.

So the trim is a CHOICE that silently **arms a requirement**, and the blocker fires at export if
the stratified in-range / out-of-range breakdown is absent. Same protection, spent at the point
where the error actually occurs, and no tax on a researcher doing something defensible.

### 06 · Declaration and execution are separate, and execution is bound to a data scope

The litmus test, automatable:

> **Does this transform's output for row *i* depend on any other row?**

- **No — structural repair.** Row-local, deterministic, label-free: parse `True`/`False` to
  boolean, coerce a type, fix units, rename, split a delimited field. Zero leakage pathway, so it
  **executes immediately** on the working table and posts a receipt.
- **Yes — statistical transform.** Imputation, scaling, winsorizing, trimming, target encoding,
  feature selection all learn from a distribution. They are **recorded as decisions now and
  executed inside per-model pipelines fit on training folds only.** Materializing one on the
  working table pre-split is the canonical preprocessing leak.

**The router defaults to deferral when unsure.** The user still gets the immediate point-and-fix;
the decision sentence carries the timing as methods prose — *"Missing `age` will be imputed with
the training-fold median"* — which is simultaneously the receipt, the schedule, and the manuscript
line. Never hidden, never a lecture. Forcing a stateful transform to materialize early is a
blocker; a **read-only preview not persisted to the modeling table** is the only permitted
override, and it is labeled *preview, not applied*.

### 07 · Missingness routes by dtype **and** mechanism

Prediction is not inference, and the distinction is load-bearing: the missing-indicator method
discouraged for causal estimation is defensible and often helpful for prediction under informative
missingness.

- **Binary / categorical** — ask first whether the missingness is informative (*"could a blank here
  mean something?"*); in EHR data it usually is. Default to an explicit `Missing` category or a
  missing indicator, which preserve the signal. Imputing an informatively-missing field is a
  blocker with typed acknowledgment, and the **stability assumption** — that missingness means the
  same thing at deployment — is recorded as a methods assumption, because it may not hold across
  sites.
- **Numeric** — offer single vs multiple imputation and the strategy; fit **inside the fold**; and
  **never place the outcome in the imputation model**, which is a blocker in any configuration.

### 08 · What this does not settle

No source gives a missingness rate at which an indicator beats imputation; the app asks rather than
infers. Mechanism stability at deployment is unverifiable at build time and is recorded, not
checked. Whether a train-only trim is worth its extrapolation cost is a per-dataset judgment. And
whether a non-persisted preview biases the analyst's later model choice is an unstudied
cognitive-leakage question — previews stay conservative and labeled.

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

## How L9 is sequenced, and why it is not negotiable

The Guided steps are built **in the order a user meets them** — Data & Target, Explore, Features,
Preprocess, Train, Explain, Report — and each one must be **drivable end to end from upload on the
day it lands.** Not demoable in pieces: drivable.

There was a proposal to build a thin Train and Report first, to get a demonstrable spine sooner.
It was rejected, on two grounds:

1. **Sequential order is the data-dependency order.** A Train step built before Preprocess must
   invent its own preprocessing assumptions, and Preprocess-when-built then has to retrofit into
   decisions Train already hardcoded. Building out of order buys a demo and pays for it in rework.
2. **Driving the app is the design method, and it only works on a whole journey.** In the product
   owner's words: *design is an iterative process of taking the journey yourself through the app.*
   `GUIDED-001` through `GUIDED-008` — the binary-text detector, evidence inside finding cards, the
   impossibility band, expandable panels — came from one drive of a real dataset through the two
   built steps. None of them came from reading code. A step that cannot be driven produces no
   feedback, and a journey with a hole in the middle cannot be driven at all.

**The acceptance criterion for every L9 loop is therefore the same:** the product owner can open
the app, upload their own file, and reach the end of the newly built step without leaving the
Guided door. A step that needs a script, a fixture, or an explanation to exercise is not done.

### Correctness is scoped to the step, not to the ledger

Correctness work is not optional here — the app makes claims that end up in manuscripts, and a
wrong number that gets published is worse than a crash that gets reported. But the backlog must
not set the build's pace. Two rules keep both true:

- **Findings in the path of the step being built are worked. Findings outside it are parked.**
  Parked is not forgotten: the row exists, dispositioned, and comes back when its step arrives.
- **A step is not done while an `OPEN` critical or high sits in the code path it executes.** That
  is a per-step gate, and it is what makes the definition-of-done's zero-`OPEN` condition
  reachable by construction rather than by one enormous burn-down at the end.

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

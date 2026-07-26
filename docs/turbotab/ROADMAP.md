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

1. **Parity.** For the same CSV and the same choices, TurboTab produces the same numbers as the
   Streamlit app — same splits, same metrics, same lockbox rows, same manuscript claims. Enforced
   by the parity harness (L10), not by inspection.
2. **The interaction model is real.** One card per step, preview before apply, deferral that
   resurfaces, a stale cascade you can see, and flagged exhibits that land in the export. The
   prototype defines the target; the app must do it against live data.
3. **The invariants hold.** Every rule in `ARCHITECTURE.md` §02 has a test that fails if it is
   broken. The lockbox in particular: sealed before exploration, sealed once, opened once.
4. **The ledger is closed.** Every finding is `FIXED` (with a named test), `NOT-A-DEFECT`, or
   `WONTFIX` (with a reason). Zero `UNVERIFIED`, zero `OPEN`.

Anything short of all four is a milestone, not a finish.

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
              ◆ DECISION C — cutover
                     │
  L11 cutover and delete
  L12 packaging
```

**Rough scale.** L1–L3 are days. L4–L7 are weeks. L9 is the long pole — eleven steps of
interaction, each its own loop. L10–L12 are weeks. Treat this as a months-long project with
usable milestones throughout, not a sprint.

---

## Decision gates

Three decisions only you can make. Each is one question, answerable in an afternoon, and each
unblocks a run of autonomous work.

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

### ◆ Decision C — cutover

**Question:** does the Streamlit app get deleted, or kept as a fallback?

Ask it once parity is green (L10). Keeping both doubles maintenance forever; deleting is
irreversible in practice. The parity harness is what makes this decidable on evidence.

**Recommendation:** keep Streamlit read-only for one release cycle, then delete. Do not maintain
two feature sets in parallel — that is the failure mode that kills rewrites.

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

### L11 · Cutover — supervised

Ship TurboTab as the default, Streamlit read-only. Then, after a release cycle, delete
`pages/`, `app.py` and the `utils/*_ui.py` layer. Expect this to surface the last of the trapped
logic; that is what the freeze list and the ledger are for.

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
- Deleting anything. An agent that deletes `utils/theme.py` as "just styling" also deletes the
  step state machine.
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

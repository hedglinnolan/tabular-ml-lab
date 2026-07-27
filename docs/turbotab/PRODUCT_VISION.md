# TurboTab — the product thesis

> TurboTax for tabular research data. The app asks one question at a time, the answers
> accumulate into a document, and that document *is* the manuscript in embryo.

This is the design breakthrough the rebuild exists to serve. The architecture follows from it,
not the other way round.

---

## 01 · The problem, stated correctly

The presenting complaint was "Streamlit's design language is limited" — the app has a Ferrari
engine and the body of a Honda Civic. That is true but it is not the root cause.

The actual problem: **users arrive with heterogeneous datasets and goals, the app offers eleven
pages of freely-clickable pathways, and it degrades — visually and structurally — as soon as
someone wanders.** Read that back and the root cause is *combinatorial pathway explosion*.

This matters because it rules out the obvious fix. A more expressive design language does not
reduce combinatorics; it usually increases them. More freedom means more paths means more of
exactly the mess. Rebuilding the current information architecture in React would produce the same
maze, rendered better.

**The fix for heterogeneous users is more opinionation, not more flexibility.**

## 02 · The insight

Take a branching, intimidating process. Run it as an **interview**: one consequential question at
a time, with complexity disclosed only when an answer opens it. That is the TurboTax pattern, and
it is the correct shape for an audience of researchers who do not know what they should want to
see — the blank-page problem.

Render that interview as a **growing document**: answered questions collapse into prose sentences,
the page builds downward as you go, and scrolling up is reading what you already decided.

Then the payoff, which is what makes this specific to *this* app rather than a generic UX pattern:

> **The transcript the user scrolls and the manuscript they export are the same object at two
> levels of formality.**

"Here is what we saw, here is what we chose, here is why" *is* a methods section being drafted in
real time. The app is not a tool that produces a paper at the end; the app is the paper, being
written.

## 03 · Why this app in particular

The idea is not imported. It is the codebase's own architecture finally becoming visible.

`docs/ARCHITECTURE_SCROLLYTELLING_BRIEF.md` already describes the app as four planes. Map the
interview onto them:

| Plane | Existing implementation | Its role in TurboTab |
|---|---|---|
| **Data** | `utils/session_state.py`, `reset_downstream_results()` | What is currently true; the invalidation cascade |
| **Advisory** | `utils/insight_ledger.py` — `upsert / acknowledge / resolve / rollback` | The coach's noticings, with a lifecycle that *already models* "remind me later" |
| **Record** | `utils/workflow_provenance.py` — one slot per stage | The scroll transcript, played forward |
| **Narrative** | `ml/narrative_engine.py`, `ManuscriptDraft`, `[AUTHOR REQUIRED]` | The same record, formalised |

The coach popping in with *"I noticed this — want me to remind you downstream?"* is an Insight
with an `acknowledge` lifecycle. **Already built.** The transcript of decisions is
`WorkflowProvenance`. **Already built.** "Mark this for the manuscript" is the evidence map
feeding the narrative engine. **Already built.**

The Streamlit pages have been standing *in front of* this spine, hiding it. That is why the idea
feels right: it is more faithful to the existing codebase than the current UI is.

## 04 · The interaction model

### The loop

The app observes, orients the user, the user decides, the app acts and records. Every cycle
appends to the record; the record is the deliverable.

### Push the notable, pull the rest

The dilemma at EDA — *ask users what they want to see, or hand them everything?* — is a false
choice. Both are wrong. "What do you want to see?" is a bad question for researchers who don't
know yet. Handing them everything is the wall of plots that already exists.

Instead: the profiler runs the moment data lands, and the section **opens already answered** —
a ranked stack of findings, each carrying its plot, its plain-language consequence, and its
downstream action. Below that, a quiet palette for pull-based exploration.

The app's question is never "what do you want to see?" It is **"here is what matters — anything
else you want to look at?"**

### Preview before apply

An early prototype offered "Fix now" on each finding. That was wrong, and the reason generalizes:
**in a pipelined app you cannot earn trust by asking for blind consent.**

Every finding's primary action is now *"Show me what changes"*, which opens a before/after panel —
two table snippets with changed cells highlighted, a row of before→after statistics, and for
distributional fixes a pair of histograms. Only inside that panel do Apply / Defer / Dismiss
appear. The engine already works this way: the import doctor returns findings and proposed fixes
and applies nothing on its own.

### Deferral is a first-class disposition

Three answers to every noticing: handle it now, **hold it until the step where it belongs**, or
let it go. Deferred items queue in a rail dock and **resurface, pre-checked and attributed, at the
step they target**. That closes the loop between what the app noticed and what the user decided,
which is the whole point.

Interjection discipline matters: if the coach interrupts on every noticing it becomes Clippy.
Noticings accumulate quietly and surface at decision points — which is exactly what
`acknowledge → resolve` already models.

### The past is editable, never silently destroyed

Changing an earlier answer marks downstream work **stale** — visibly veiled, tagged, and
recoverable — rather than deleting it. The existing app answers this destructively via
`reset_downstream_results()`. Making the cascade *legible instead of silent* turns the app's
most important safety mechanism into something the user can see and trust.

This is also the single feature Streamlit most fundamentally cannot render.

### Work is visible

Anything over ~1s becomes an observable job with a name in plain language, progress, and a cancel.
"Did it break?" is the most common complaint about the current app, and it is a direct consequence
of Streamlit having nowhere to put long work.

## 04b · Two doors, one modeling process

TurboTab is not version two. It is a **second door onto the same analysis**, and the Streamlit app
is not deprecated by its existence.

> Here is the Streamlit version of the app, and here is a more dynamic version of the same
> modeling process — with more intelligent routing.

Three consequences follow, and they are load-bearing.

### Parity is a product promise, not a QA gate

"The same modeling process" is a claim made *to researchers*. It means a manuscript produced
through either door describes the same science, and a reviewer asking "which tool did you use?"
gets an answer that does not change the numbers. The parity harness is therefore not migration
scaffolding — it is the mechanism that keeps a public promise, and it should be stated in both
UIs, not hidden in CI.

### Routing is the entire differentiator

If the two doors differ only in appearance, the second one has no reason to exist. Streamlit's
design ceiling is real but it is not a product thesis. **What justifies TurboTab is that it asks
better questions, in a better order, and hides the ones that do not apply.**

That places the Router at the centre rather than at the end. It also names the failure mode
precisely: *a beautiful reskin with thin routing*. Everything else in this repo — the design
language, the interview feed, the preview panels — is packaging around that claim. If the routing
is not demonstrably better, the packaging does not save it.

Concretely, "more intelligent routing" has to mean at least:

- **Fewer irrelevant questions.** A single-file upload never sees join options. A regression
  target never sees class-balance questions.
- **Ordering by consequence.** What matters most for *this* dataset comes first, from the coach's
  ranking rather than from a filename prefix.
- **Disclosure driven by findings.** A question appears because something in the data raised it,
  not because the pipeline has a stage for it.
- **Deferral that closes.** Something noticed at exploration returns at the step that can act on
  it — which no page-ordered app can do.

Each of those is testable against a fixed dataset. Test them before building eleven steps of feed.

### Users should be able to switch doors mid-analysis

This falls out of the architecture almost for free, and it is the strongest argument for the
extraction: if both doors are thin views over one `AnalysisProject`, a user can begin in Guided,
switch to Classic for one fiddly step, and switch back — without losing state or changing results.

That imposes one design constraint worth stating now: **`AnalysisProject` must be sufficient for
both doors, not shaped to TurboTab's convenience.** If the Guided door needs a field the Classic
door cannot populate, the state model has forked and the promise breaks.

### Naming

Call them modes, not versions. "Classic" and "Guided" both read as legitimate choices; "v1" and
"v2" tell every existing user they are on the dying one. Whatever the words, the app should be
explicit that the two doors run the same engine.

---

## 05 · Why the Guided door required leaving Streamlit

Streamlit has no client. Every interaction is a round trip: the browser reports a widget change,
the server reruns the script top to bottom, the browser repaints. Four consequences, all of which
the vision depends on escaping:

1. **No instant visual confirmation.** State that responds to a click must live in the browser.
   Streamlit has nowhere to put it, so the app can only ever *reload*, never *respond*.
2. **Rented styling.** You can inject CSS, but you are styling a DOM you do not own, with
   generated class names that shift between releases.
3. **No home for background work.** A five-minute training run either blocks the page or fights
   the rerun loop, and a refresh loses it. This is the big one.
4. **No way to model a file roster.** Multi-file upload wants per-file cards with parse status,
   join relationships, and background parsing — an entity with rich client-side state.

Notice that **background jobs appear in three of the four**. That is what determined the shape:
not Streamlit → native app, but **Streamlit → client/server split**. A Python backend owning
project state and a job queue, wrapping the existing `ml/` core untouched, plus a real frontend.

Wrapped in Tauri or pywebview it is a double-click desktop app on Mac and Windows; served bare it
is the university Docker deployment and the eventual All of Us story. All three targets survive —
which a native rewrite would have foreclosed.

## 06 · Considered and rejected

**Native desktop (Qt, Flutter).** Rejected. The app serves three deployment targets — laptop
desktop, university Docker/OIDC, and research enclaves. A browser-served app runs in all three; a
native binary runs in one. You cannot install a Mac app inside a Workbench VM.

**Fork into All of Us.** Rejected *as a fork*, kept as a direction. All of Us individual-level data
cannot leave the Researcher Workbench, so the app must go to the data. Its Dataset Builder also
homogenises input shapes, which dissolves much of the heterogeneity problem, and it supplies a
motivated, homogeneous user population. But a fork means two diverging codebases and death by
maintenance. The right structure is one core with thin platform adapters. Note the Data and
Statistics Dissemination Policy forbids reporting counts under 20 participants — small-cell
suppression would be required, and is worth building generally.

**Redesign inside Streamlit.** Partially adopted. The hierarchy work — ordering by coach verdict,
attributes as chips, warnings attached to what they warn about, a primary action that is actually
primary — is achievable today and transfers to any future frontend because it is design decisions,
not code. It was worth doing first to prove the interaction model before paying for the rewrite.

## 07 · Design principles

Five commitments, ordered. When two collide, the earlier wins.

1. **Never assert falsely.** Inherited from the codebase's governing rule. The UI may be silent or
   may refuse, but every rendered statement traces to a recorded decision or a computed fact.
   Confidence below `high` never pre-selects — and **auto-advancing an interview is pre-selection.**
2. **The transcript is the artifact.** Answered questions collapse into prose. Export formalises
   what is already on screen; it never surprises.
3. **Push the notable, pull the rest.** Each section opens already oriented. The user is never
   asked "what do you want to see?"
4. **The past is editable, never silently destroyed.** Changes mark downstream work stale —
   visible, veiled, recoverable — and recomputation is an explicit, observable job.
5. **Only consequences move.** Motion shows cause and effect: sections settle when answered,
   arrive when opened, propagate when invalidated. No ambient animation.

## 08 · Open design questions

- **The coach's interruption budget.** How many noticings may a step raise at once? The prototype
  settled on two. Unbounded, "push the notable" collapses back into the wall of plots.
- **Recompute depth.** The prototype restates invalidated *numbers*. It does not re-derive *which
  findings the coach would raise* against changed data. In a real implementation that is the
  invalidation cascade doing genuine work, and it is the piece most worth prototyping against the
  real `ml/` core before committing.
- **Interview cadence.** One-question-at-a-time is right for true branches and infuriating for a
  power user's tenth dataset. Quick/Advanced should set how chatty the interview is — same spine,
  two verbosities.
- **Router gating.** The coach can order questions but cannot gate them (see `TRANSITION_PLAN.md`
  §02.5). What gating is legitimate, given that pre-selection requires `high` confidence?

## 09 · Revision to a stated rule

The design language originally said *"the viewport never auto-scrolls; the user's scroll position
is theirs."* Building the prototype proved that wrong: when a new question arrives below the fold,
not moving is disorienting.

**Revised rule:** new content is nudged into view *only when it sits below the viewport*, so the
page never yanks a user who has scrolled up to read. The prototype implements the revision.

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

### Guided is never the less capable door

Routing is the differentiator, but it is not a licence to ship a narrower product. The rule, in
the product owner's words:

> **Guided should be easier to understand and more dynamic, not less capable.**

Multi-file assembly is the case that settled it. Guided takes one table today, and the tempting
shortcut was to hand the user off to Classic whenever files need combining — cheap, honest, and
exactly backwards. Combining files is the interaction that *most* needs a dynamic surface,
because the user is empowered to decide correctly only when they can watch their working data
morph under each join, merge or split. Routing them to Streamlit for that is routing them to the
one place where the morph cannot be shown.

Two consequences follow, and they are design requirements rather than nice-to-haves:

- **The live preview answers the question the wizards had to ask.** Tableau, SPSS and Power Query
  interrogate the user about grain — *what does one row represent?* — because a static form cannot
  show them. A surface that animates the working table can let the user *see* one-row-per-patient
  become one-row-per-visit. Ask only what showing cannot answer.
- **Reshape is not a second release.** Cross-sectional versus longitudinal restructuring is part of
  "how does my data morph," not an advanced extra bolted on later.

The corollary binds the other way too: a capability Guided cannot yet do is a `classic-only`
register row with a dated reason, never a permanent scoping-down of what the second door is for.

### The shelf is never shortened — judgment is rendered as order, not as absence

The shape of the data changes how models are **ranked**. It never changes which models are
**available.** In the product owner's words:

> **The shape of the data changes the model shelf, but never to the extent that a user has no
> option to select a bad model. We do our best to fit based on their selection, but the app
> surfaces the concerns outright.**

Silently withholding a model is the app making a decision in the user's name, which §06 of the
design language forbids. Offering it at the bottom of the list with the reason stated is the app
doing its job. Ordering and prominence carry the judgment — which is precisely what Classic's
Train page failed to do: `model_coach` was already bucketing into Recommended / Worth Trying /
Not Recommended, and the layout rendered taxonomy instead.

**The three rungs, because "never remove the option" is not absolute.** The app already refuses
some things outright, so the line has to be stated rather than assumed:

| Rung | When | Example |
|---|---|---|
| **Refuse** | No legitimate use exists at all. Proceeding makes the app assert something false that no caveat repairs. | The outcome inside the imputation model. A post-seal eligibility restriction. |
| **Block and record** | A legitimate use exists but is rare, and the user may know something the engine cannot. | Keeping a leakage-suspect column. Imputing informatively-missing data. Contradicting the grain evidence. Typed acknowledgment, and the manuscript carries it as a limitation. |
| **Rank and state the concern** | A matter of judgment with a real cost. | SMOTE (documented calibration harm, but legitimate when only discrimination matters). PLS-DA on small *n*. A tree ensemble at p ≫ n. |

The test for the top rung is not severity — it is whether a competent researcher could have a
reason. There is no analysis in which the outcome belongs in the imputation model; there are many
in which a suspicious column is measured before the outcome and the researcher knows it.

### The export, and what a marked figure means — the product owner's rulings

Recorded when made, because both were answered in conversation and both decide work that is
already scoped (`GUIDED-107`).

**The manuscript is data before it is a document.** `draft.py` composes one structured document
and two thin renderers emit Markdown and LaTeX — the latter through `ml/latex_report.py`, which
is already detainted and imports headless, so neither door holds a private exporter. The larger
scope was chosen over shipping Markdown alone, and the reason is downstream: **L10's checklist
engine has to read the manuscript**, and a checklist cannot be run against prose. TRIPOD+AI,
STROBE-nut, COSMIN and mQACC are one artifact with two column types — what the app knows, and
what it must ask — and both column types need a document with structure to attach to.

**A marked figure is promoted as the author marked it.** No tier annotation is added on the way
in. An `EXPLORATORY` figure the modeler moves into Results appears there as they placed it.

> The manuscript is the author's document. The app drafts it; the researcher signs it.

The alternative — annotating the caption with the tier — was considered and rejected as the
second, uncalibrated layer of caution this project already forbids elsewhere: a caveat printed on
every promoted figure makes a real concern and a routine one read identically, which is the exact
failure the evidence badge exists to prevent.

**But the record is not laundered, and this is the part that makes the ruling safe.** The tier
stays on the figure in the record, and `ml/manuscript_validator.py` is the surface that reports
it — a cross-section check in the validation report, not a caveat in the prose. The author gets
the document they asked for and a separate, honest list of what a reviewer will notice. That
keeps `AUDIT-001`'s lesson intact: the defect there was the *generated document* asserting
something no section supported, and a validator flagging a promoted exploratory figure is that
machine doing its job rather than the app editorializing in the author's voice.

### The resolution statement — what the app may say about a study, as opposed to a method

**Status: specified, unbuilt.** The design guidance is here so it is not re-derived; the loop that
builds it is not yet scheduled.

Every finding the app has attaches to a **column** or a **decision**. Nothing attaches to the
**study**. The three-rung ladder above governs what the app says about a *method*. There is no
equivalent for the case where the honest observation is about the whole project — an assay with
n=80, or a nutrition exposure whose attenuation factor implies an eightfold sample-size penalty.

#### The obvious version is wrong, and wrong in a way the research already names

The tempting card reads *"this study is underpowered for the claim you've described."* It fails
twice.

**We do not hold the claim.** At the seal we know the target, task type, grain, eligibility and
purpose. We do not know the expected effect size, which predictor is the exposure of interest, or
what magnitude would be scientifically meaningful. Asserting a verdict on a claim we were never
told is exactly the overreach the governing rule forbids.

**And it is post-hoc power in a nicer suit** — listed flatly as an anti-pattern in
`research/METABOLOMICS_PACK.md` §10 and echoed in the nutrition and clinical threads. The app would
be committing a named error while presenting itself as the tool that catches them.

#### The correct form inverts it

State the **instrument's resolution** and let the researcher judge their claim against it.

> *"With n=80 and a typical per-metabolite CV of 25%, this study can detect roughly a 1.6-fold
> change at 80% power after FDR correction. Anything smaller is invisible here."*

> *"Your exposure's attenuation factor is 0.35. Relative to an error-free measurement, matching this
> study's power would take roughly eight times the sample size. That is a property of the
> instrument, not of your hypothesis."*

This asserts nothing about their science. It is arithmetic over quantities the app already holds,
and it is the same posture as asking eligibility in scientific terms with the outcome's distribution
withheld: **the app supplies what only it can compute, and withholds the judgment that is the
researcher's to make.**

#### Design guidance

**Inputs must be derivable at seal.** n, event count, candidate parameter count (*parameters, not
variables* — a 4-knot spline is 3), observed per-feature variance or CV, outcome prevalence, and
where repeated measures exist, λ from the variance components. Nothing that requires an effect size
the user has not stated.

**It is always available, and unprompted only when stark.** This is a pull surface by default —
`PRODUCT_VISION.md` §04's *push the notable, pull the rest*. It pushes only when the arithmetic is
unambiguous: fewer events than Riley's criteria require for the declared parameter count, or an
attenuation factor low enough that the implied penalty exceeds an order of magnitude. A resolution
statement that fires on every dataset is wallpaper.

**It never says "don't."** No refusal, no blocked action, no severity that gates a step. The shelf is
not shortened here either. A researcher who wants to model 12 samples per group may; the app states
what that study can see and records the sentence.

**It is a recorded decision, and it belongs in the manuscript.** *"This study was powered to detect
a 1.6-fold change"* is a limitations sentence a reviewer would otherwise compute themselves, and
the reporting standards in all four packs expect a sample-size justification or an explicit
statement that the work is hypothesis-generating. The statement is that sentence.

**Its natural home is the seal, extended.** The seal already states its basis in four states. A
fifth thing a seal can honestly report is what the sealed cohort can resolve — the sealed n is the
input, and the seal is the moment the cohort stops changing. It is a statement *beside* the basis,
not a fifth basis value.

**Per-domain forms, all already researched.** Metabolomics: detectable fold change given CV and FDR
(`METABOLOMICS_PACK.md` §08.4). Nutrition: λ, the 1/λ² penalty, and days-needed
(`NUTRITION_PACK.md` §03). Clinical: Riley's criteria-based minimum, counting candidate parameters
including those later dropped (`CLINICAL_SURVEY_PACK.md` §A5.4). Survey: attenuation by scale
reliability (`§B6`).

#### The related change it forces

**The holdout should track n.** A 20% test set at n=80 is 16 rows, and a C-statistic estimated on 16
rows has a confidence interval spanning most of the unit interval — an honest number that answers
nothing. The clinical thread is direct: a single train/test split is the weakest option at typical
sample sizes, and bootstrap optimism correction is preferred because it uses all the data.

So whether to draw a holdout at all is a **consequence card at the seal**, not a default — the same
treatment the constitution already gives every choice whose cost is invisible at the moment it is
made. And at small n the deliverable shifts with it: the **prediction-instability plot** and the
resolution statement stop being supplementary and become the headline, which is the app being
useful at small n by being loud about it rather than by refusing.

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

## 06b · Correct, surfaced, beautiful — the product owner's ruling, 2026-08-03

Recorded the turn it was made, in his words:

> *"The whole point of this app is to be a beautiful expression of how to conduct informed math
> modeling in the researcher's specific domain, with real tradecraft and results to back up their
> decisions, all presented to the user in a dynamic, easily digestible manner. **In addition to being
> correct, the engine must surface and it must be beautiful.**"*

**Three conditions, ordered, and all three required.** Correct is first and does not become optional;
what the ruling adds is that it was never sufficient. A capability that is right and unreachable has
not shipped, and a capability that is reachable and unreadable has not either.

**Why this needed saying now.** The three preceding loops found the same defect three times, at
increasing depth: the pull palette threw on every click, no pack finding from any of five packs had
ever been rendered, and `/recipes` — fifty fields — reaches nobody still. That is condition two,
measured for the first time, and `LOOP.md` §05's rule about capabilities and consumers is the
instrument for it.

**Condition three has no instrument at all, and this is the honest gap.** `DESIGN_LANGUAGE.md`
specifies the palette, the three-voice type rule, the component vocabulary, motion and the question
grammar — and nothing checks any of it. `pageharness.py` says so in its own docstring: it proves what
the controller renders and **cannot prove visibility** — that a card is on screen, unclipped, above
the fold, in a section that is not hidden. Nothing without layout can. The one measured fact we hold
about condition three is `DESIGN_LANGUAGE.md` §05.2's: the app has **no mechanism for animating a
change of content** — 92 `innerHTML` assignments against 22 node-owning writes, zero
`startViewTransition`, zero FLIP, zero WAAPI — which is precisely the *dynamic* half of the sentence
above.

### The consequence that is already live, and it arrived with the fix

**Surfacing created the beauty problem in the same commit.** `GUIDED-142` made five packs' worth of
findings visible at once, and the page renders them as `pf.concat(packf).map(findingCard)` — every
profile finding, then every pack finding, **unbounded and uncapped**. Measured on `clinical_labs.csv`
under the clinical lens: **twenty finding cards.** The prototype settled on two.

§08's first open question — *"the coach's interruption budget… unbounded, push-the-notable collapses
back into the wall of plots"* — is therefore no longer open in practice. It has been answered by
accident, and the answer is *all of them*. That is §01's root cause reassembled inside the new door:
the wall of plots was the thing this product exists to replace, and breadth of domain content is the
force that rebuilds it.

**So condition three is not a later polish phase.** It binds now, it binds hardest exactly where the
domain work is succeeding, and `GUIDED-149` is the row.

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

  **Split at L45, because it was two questions wearing one.** An **interruption** arrives at a
  decision point and takes the user off what they were doing; two is right for that, and §04's
  deferral machinery is the answer. A **stack** is the section opening already answered — it
  interrupts nothing, it *is* the content, and capping it at two on `clinical_labs.csv` would
  collapse ten of thirteen findings including five of the eight the clinical lens exists to
  produce. The stack half is answered: `turbotab/attention.py`, bound five, with the median of
  this repository's sixteen fixtures as the reason, everything reachable behind a counted and
  typed affordance, and nothing that gates a decision ever collapsed (`GUIDED-149`).
  **The interruption half is still open**, and it is the one this bullet was about.
  `prototypes/explore-stack.html` is where the number gets looked at.
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

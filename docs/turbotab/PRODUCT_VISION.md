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
under the clinical lens: **thirteen finding cards in one list.** The prototype settled on two.

*(This paragraph said **twenty**, and twenty was the number of findings **served**, not the number of
cards on the Explore list. Re-measured at L45 by driving the page: 21 served today — 5 profile, 8
pack, 8 structural — and the Explore list renders the first two streams, so thirteen. The eight
structural ones reach the user at the **Data** step, and on this fixture `structList` renders **zero**
of them because the bulk repair groups take all eight. The correction does not weaken the finding —
thirteen unbounded cards is the same wall — but a ruling document asserting a measurement it cannot
reproduce is the governing rule failing in the record layer, which is a mistake this project has made
before and written down twice.)*

§08's first open question — *"the coach's interruption budget… unbounded, push-the-notable collapses
back into the wall of plots"* — is therefore no longer open in practice. It has been answered by
accident, and the answer is *all of them*. That is §01's root cause reassembled inside the new door:
the wall of plots was the thing this product exists to replace, and breadth of domain content is the
force that rebuilds it.

**So condition three is not a later polish phase.** It binds now, it binds hardest exactly where the
domain work is succeeding, and `GUIDED-149` is the row.

## 06c · Explainability under the lens — the product owner's rulings, 2026-08-09

> **Revised 2026-08-09 after adversarial review.** Two reviewers with live literature search returned
> SERIOUS PROBLEMS on this section and **the rulings survived while several of the claims under them
> did not.** Corrected in place: the instrument's **name** (it is a multivariate forward marginal
> effect, not an ALE), its **support claim** (measurably false — a mask is mandatory), the assertion
> that the field cannot state an estimand for a black-box learner (`NUTRITION_PACK.md` §05 already
> could), **ruling 3** (fixed and variable totals are not one kind of thing), and **ruling 7** (all
> three named checks are broken as specified). One new gate was added that the review produced rather
> than refuted: **held-out performance gates the explainability surface.** Nothing here is left
> asserting what was refuted — that state persisted for one day and is recorded in `LOOP.md` §03.

**The reframe that produced these.** A generic tool's explainability answers *"which features
matter."* Under a domain lens the question is **"is this model's reasoning consistent with what is
known about this domain, and where does it disagree?"** The product owner's framing, and it is the
specification: *"SHAP in particular I have high hopes for being the vehicle by which a nutrition
modeler can understand the inductive bias each model brings to the table in solving their problem."*

**What made this conversation necessary.** The parity register says **four explain capabilities are
`classic-only`** — `explain-shap`, `sens-seed`, `sens-feature-dropout`, `sens-robustness-verdict` —
so the entire explainability and sensitivity suite is in the door being left. Guided has
`sklearn.inspection.permutation_importance` and nothing else, and **permutation importance cannot
answer the inductive-bias question**: it reports how much a fitted model's *metric* degrades when a
feature is destroyed, one number per feature. It cannot show that a tree found a threshold where a
linear model found a slope, which is the disagreement the modeler needs to see.

**Four layers, and layers 2 and 3 are the differentiated part:**

1. **Attribution.** SHAP as raw material — beeswarm, per-observation force plots.
2. **Inductive bias.** Each feature's SHAP value plotted against its raw value, **per model family,
   on shared axes.** A linear model draws a straight line by construction; a tree draws a step
   function. **The divergence between those curves IS the inductive bias**, and it only means
   anything when families are compared on one dataset — which is why no generic tool draws it.
3. **Domain consistency.** The packs hold established directions and plausibility bounds. Where the
   attribution disagrees with them, that is a finding.
4. **Stability.** The three `sens-*` capabilities. **An attribution whose sign flips under reseeding
   is not an explanation**, and layer 3 is dishonest without it.

### The instrument — the substitution curve, and what it actually is

**This subsection was refuted by two adversarial reviewers with live literature search on
2026-08-09 and is rewritten from their findings.** Both returned SERIOUS PROBLEMS. What follows keeps
the design and corrects the claims, because **the product owner's reframe outranks the review**:

> *"Just because a CRAN package exists in the world with a specific plot doesn't mean that would not
> still be a useful feature to ship in our app."*

**Read the review as a specification, not a verdict.** Nearly every objection converts to a build
requirement with a citation attached — and the existence of three reference implementations gives
every element of the figure a resolving `source`, which is what the evidence gate demands anyway.

**The problem §04 creates for attribution, stated once.** If the *specification* determines the
estimand, then an attribution computed on a fitted model **inherits that model's estimand** — and
SHAP neither knows nor says which one. A SHAP value on a nutrition model is a substitution effect
against an unnamed population-average mixture, using an estimand §04 calls *biased even absent
confounding*, and it renders as `fiber_g: +0.31`. **That is the governing rule broken in the step
this product cares most about.**

**What this section previously added to that, and it was false.** It said that because all five of
§04's models are regression specifications, *"for a tree ensemble the field has no established way to
state the estimand at all."* **`NUTRITION_PACK.md` §05 already contradicts it** — line 516 names
compositional data analysis with **isometric log-ratio (ilr) coordinates as covariates**. ilr is a
basis change; any learner fits on it, tree ensembles included. Two further routes were named:
the **leave-one-out parameterization**, and **g-computation with TMLE or double/debiased ML**. The
claim was an assertion of absence made without reading the pack section that answers it, which is the
failure `LOOP.md` §06 now carries a check for.

**The design, unchanged.** Make the substitution explicit and chosen rather than implicit and
averaged: move *k* units of the conserved total from donor A to recipient B, hold the total fixed,
and plot the model's predicted outcome against *k*. It is a substitution effect **that names its own
donor and recipient**, and that is why it is worth building.

#### What it is actually called, because the wrong name was shipped

**It is a multivariate forward marginal effect along the direction *d* = *e*_B − *e*_A, aggregated
over rows as an Average Marginal Effect.** Scholbeck et al. 2024, *Data Mining and Knowledge
Discovery* 38:2997–3042, which ships as the R package **`fmeffects`**.

**It is *not* "a 1-D ALE along a constrained direction," which is what this section said.** That
describes a different object: ALE accumulates *local* differences over the **conditional**
distribution, whereas shifting **every** row by the same *k* and averaging is **marginal**
averaging. The two coincide only when the shifted coordinates are independent of the rest, which
under a closed composition is exactly what they are not.

**Prior art this must position against, not claim novelty over:**

| Work | What it already does |
|---|---|
| Dumuid et al. 2019, *Stat Methods Med Res* 28(3):846–857 | The compositional isotemporal substitution model |
| CRAN `codaredistlm` (2022), `multilevelcoda` | Pairwise one-for-one reallocations **with confidence intervals** |
| Ho et al. 2021, *Lancet* | Non-linear isocaloric substitution |
| Mekary et al. 2009, *AJE* 170(4):519–527 | The substitution framing in nutritional epidemiology |
| Lundborg & Pfister 2025, arXiv:2311.18501 — **preprint only** | Defines the estimand, and **explicitly excludes random forests and boosted trees** |
| Fisher, Rudin & Dominici 2019, *JMLR* 20:177 | **Model Class Reliance** — the correct framing for what this section calls "inductive bias" |

**What has no precedent as a shipped tool is using it as the axis on which model families are
compared.** That claim survives the review intact, and it is the only novelty claim this section may
make.

#### The support claim was false, and it was measured false

This section said the curve *"stays inside the data's support where a partial-dependence plot would
evaluate the model at combinations that do not exist."* **On a synthetic conserved-energy composition,
a 300 kcal shift puts 22% of rows off-support; 500 kcal puts 64% off, with 1% of intakes going
negative.** Shifting every row by a fixed *k* walks off the simplex exactly like a PDP does.

**A support mask is therefore mandatory, not an enhancement.** The curve **stops** where the shifted
composition leaves the observed support, and it says why it stopped.

#### "The slope at zero" does not exist for a tree ensemble

A piecewise-constant model has derivative zero almost everywhere and undefined on its splits.
**Report a finite difference at a stated *k*, with *k* in the label** — *"+0.082 per 100 kcal"*, never
*"the slope."*

#### What the curve does not remedy, and the two-word error that hid it

**`NUTRITION_PACK.md` §04 carried a two-word error and it was load-bearing on this whole section.**
It said the standard and residual models are biased even absent confounding **"because"** the
substituted mixture is the population-average mixture. The phrase *"biased even absent confounding"*
is near-verbatim from Tomova et al. 2022 (*AJCN* 115(1):189–198, PMC8755101) and **is correct.** The
word **"because"** was not: the paper's mechanism is **composite variable bias** — information lost
when two or more components with distinct effects are collapsed into a total — and the
population-average mixture is the paper's *definition of the estimand*, in an adjacent sentence.
Corrected in the pack the same day.

**Why it mattered here.** With "because" in place, the substitution curve read as **a remedy for that
bias.** It is not. **The total is still in the model either way** — naming the donor and the recipient
makes the estimand *explicit and chosen*, which is a real and sufficient gain, but it does not undo
composite variable bias. This section must not claim it does, and the caption must not imply it.

*(The first reviewer said §04 had drifted from Tomova. It had not. A second reviewer, reading the
primary source, produced the narrower and sharper truth above — which is why a design proposal now
ships with a prior-art check the way a closure ships with a revert probe.)*

#### The five marks, recorded here because the drawn spec must not live only in a URL

A figure specification that points at ephemeral storage will eventually lie (`AGENT_ONBOARD.md` §07
trap 8), so the marks are enumerated in the repository:

| Mark | Exists because |
|---|---|
| Support mask and density strip | 300 kcal → 22% of rows off-support; 500 kcal → 64% off, 1% negative |
| Bootstrap uncertainty band | Every reference implementation ships intervals; shading between bare point estimates reads as significance |
| Linear-ilr null overlay, dashed | A linear model already produces curved, asymmetric substitution curves — without the null we sell coordinate geometry as learned structure |
| Stated *k*, not "the slope" | The ensemble is piecewise constant; the derivative is zero a.e. and undefined on splits |
| Dash pattern per series | Four categorical hues cannot pass CVD separation while teal, green, gold and red are semantically reserved — see `DESIGN_LANGUAGE.md` §05.2 |

**The pack supplies the constraint direction and the plausibility bounds; the core computes the
curve.** That division is unchanged and it is what makes one mechanism serve many domains.

### The rulings — the product owner's, 2026-08-09

**1 · A disagreement with the pack is a badged finding, never a verdict.** Show both directions,
label the disagreement, carry the pack's own `SETTLED` / `CONVENTION` / `DISPUTED` badge onto it.
**The app does not decide who is right** — a disagreement may be confounding, a coding error, or the
result. *"The model is wrong"* and *"the literature is wrong"* are both assertions it cannot support.

**2 · The unit of explanation is the model, rendered as the deck's face 3.** Not one explanation for
a chosen model with comparison as an extra step — **the comparison is the default view**, because the
inductive-bias question is a question about difference. `GUIDED-178` and `GUIDED-232` are **one
mechanism**, and face 3's *"the reorder is the comparison"* becomes literal.

**3 · The curve is core, with a pack-supplied budget — and the budget must declare whether it is
FIXED or VARIABLE.** Any lens that declares a conserved total gets the curve; a lens with no budget is
not offered it. **One mechanism, many domains**, which is the whole architecture of the packs.

**The original form of this ruling was unsafe, and the omission is the interesting part.** It listed
nutrition's kcal, metabolomics' total ion current and genomics' library size as though they were one
kind of thing. **All three are *variable* totals** — a person can eat more, a sample can carry more
ions, a library can be sequenced deeper. **24-hour time-use is a *fixed* total**, and the two behave
differently under substitution (Tomova et al. 2025, *BMC Med Res Methodol* 25:100). A fixed total
makes the reallocation exhaustive and the constraint hard; a variable total means "hold the total
fixed" is a *modeling choice the user is making*, not a property of the data, and the app must say so.

**And the case the method was built for is the one case the list omitted.** Isotemporal substitution
comes from time-use epidemiology — Dumuid 2019 is a 24-hour composition. So the budget declaration
carries a kind, the curve states which kind it is drawing under, and a variable total gets the
sentence that the total was held fixed by assumption.

**4 · Where the estimand cannot be determined, refuse the number and offer the curve.** No scalar
attribution is printed when the app cannot say what it is an effect *relative to*. The substitution
curve is still drawn, because **it carries its estimand explicitly in its own axes.** This is the
governing rule's strongest available reading and it costs the user nothing.

**5 · The app performs energy adjustment, inside the training fold.** §04's five specifications
become a preprocessing choice rather than an advisory. The in-fold requirement is §04's own and the
lockbox already enforces that class of constraint. **This is what unblocks the inference path**, and
it is the largest single piece of scope in this section.

**6 · Methodological thresholds are researched before they ship; performance budgets are measured.**
The two are different and conflating them is how an unsourced number enters. A correlation cut, a
rank-stability warning level, an FDR *q* — these change what a result **means** and each needs a
primary source or a `CONVENTION` badge naming it as practitioner default. A row count above which a
refit runs on demand rather than automatically changes only **how long the user waits**; it ships as
a measured budget with the measurement recorded beside it, which is `LOOP.md` §06.2's distinction
applied one level out.

**7 · The faithfulness harness runs automatically where it is cheap** — and **all three of the checks
this ruling originally named are broken as specified.** Each failure below was demonstrated by
simulation on 2026-08-09, not argued.

| Check as ruled | What the simulation showed | What replaces it |
|---|---|---|
| Label-permutation null | **Vacuous for normalized measures** — permuted-label impurity importances still sum to 1, so the null is satisfied by construction | **PIMP** — Altmann et al. 2010: fit the null distribution per feature and report a calibrated *p* |
| Fold-stability rank correlation | **Passes a stable-but-wrong explanation.** With *y* independent of *X*: held-out **R² = −0.136**, Kendall **τ = 0.867** — a model that learned nothing, agreeing with itself | Keep as a *diagnostic*, never as a gate; a gate on agreement rewards consistent nonsense |
| Deletion curve vs random-deletion baseline | **Nearly fails a correct explanation.** Deleting the only causal driver cost **0.027 R²**, because a 0.995-correlated copy substitutes for it | **ROAD** — Rong et al. 2022, ICML: retrain after removal so the substitute cannot stand in |

**The one genuinely new thing the review produced, and it is a gate.** This ruling gated on
calibration and gated nothing on generalization. **Held-out performance should gate the entire
explainability surface**: explaining a model that does not generalize is explaining noise with a
citation attached, and the fold-stability result above is exactly what that looks like from inside.
An explanation is offered only above a stated held-out floor, and below it the app says what it will
not explain and why — which is the governing rule's silent branch, not its refusing one.

Cheap checks run silently on small tables so most users see them without asking, and degrade to an
explicit action above the measured row count. **Stability of a single attribution stays on demand
everywhere**, because its cost is N× training rather than a bounded refit set.

**8 · The comparison is drawn as overlaid curves with the disagreement shaded** — adjudicator's call,
deferred by the product owner. A difference curve is the purer quantity and discards absolute scale,
so a reader cannot tell whether a divergence matters; a donor × recipient matrix cannot show whether
a line is straight or terraced at sparkline size, which is the entire point. Overlaying carries
shape, magnitude and divergence at once, and **shading the region between the curves renders the
difference curve inside the same frame** at no cost in views. **The matrix survives as the
navigator** — it is how a reader chooses which pair to open, and the shelf is never shortened.

### The domain-agnostic playbook is the mould, not the product

**The product owner's framing, 2026-08-09**, supplying a full agnostic explainability playbook:
*"I see it as a useful mould for what we actually want to form into our domain-aware playbook."*
That is the correct relationship and it decides the build. **The agnostic sequence is the substrate
— calibration before explaining, grouped permutation, ALE over PDP under correlation, centered ICE
with a heterogeneity badge, interaction ranking, then attribution. The packs are the layer on top**,
and an ALE plot becomes domain-aware when the pack marks the DRI and the tolerable upper limit on
its axis and flags a direction the literature already settled.

**Three things from it bind and are recorded here so they are not re-derived.** *Build the
faithfulness harness first, not last* — it is cheap, it is what makes every later attribution
credible, and it is the governing rule turned on the explanation methods themselves. *Refuse
probability-scale attribution until calibration is checked*, because explaining an uncalibrated model
explains a distorted probability. *Force the estimand choice before anything renders* — reached
independently from §04, which is why it is ruling 4 rather than an import.

**And one thing from it is rejected on the constitution.** A rule that **recommends** a model when it
lands within one standard deviation of the best black box shortens the shelf. Ranking it first and
saying why is the same information as an ordering rather than an absence, and §04b already settled
that judgment is rendered as order. The one-standard-deviation figure is also exactly the kind of
unsourced constant ruling 6 governs.

### What this does not settle

**Whether an unstable attribution may reach the manuscript.** Ruling 7 governs the in-app default;
the export is a durable claim and the export gate is a separate decision, unmade.

**Which primary sources establish the thresholds.** Ruling 6 says they are researched before they
ship; it does not say by whom or from where, and that is a pack-authoring loop with a source
requirement rather than a build.

**What an inference-mode explainability suite owes** beyond the estimand choice — DoubleML,
knockoffs, E-values and marginal effects are named in the mould and none is ruled on here. They wait
on the inference path itself (`GUIDED-231`).

**Where the held-out floor sits.** Ruling 7's new gate says generalization gates the surface; it does
not say at what value, and that number is `GUIDED-233`'s to establish with a source rather than the
adjudicator's to pick. Until it exists the gate is **specified and unbuilt**, which is the honest
state and is not the same as absent.

**Whether the substitution curve ships before `GUIDED-233`.** It now has a complete visual
specification — five marks, each traced above to a measurement — and **no pack section behind its
thresholds.** Ruling 6 forbids shipping the thresholds unsourced, so the figure is buildable and not
yet shippable, and that ordering is deliberate rather than an oversight.

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

## 09 · Revision to a stated rule — and the revision was itself revised

**The current rule is scoped, not absolute: the page never moves the viewport *unbidden*, and it may
scroll exactly once where the user pressed a control whose sole purpose is to go somewhere.**
`DESIGN_LANGUAGE.md` §05 is authoritative and carries the full history; `turbotab/web/index.html`
has no nudge, `DRIVE-006` deleted it, and
`turbotab/test_the_page_never_moves_the_viewport.py` pins the unbidden half at **zero** while
`test_the_rail_navigates_when_a_user_presses_it` requires the navigating half to exist.

**And this section said the absolute form in the present tense for a second time — L62 shipped the
scoped rule and amended neither document.** The paragraph below describes exactly that failure at L47
and it recurred, on the same rule, in the loop that changed it; the adjudicator wrote both documents
after the fact, which is later than the rule requires. **A ruling is not a ruling until the
authoritative text carries it**, and no test in this repository reads either paragraph, so nothing
could have reported the disagreement.

**This section said otherwise until L47**, and it said so in the present tense — *"new content is
nudged into view only when it sits below the viewport… The prototype implements the revision"* —
while the app deliberately had no nudge at all and three green tests held it that way. Nothing in
the repository resolved the disagreement: there was no supersession note in either direction. Two
loops reasoned from the wrong one, `GUIDED-173`'s note among them.

**The four positions, kept because the middle one is the part that generalizes:**

1. *The viewport never auto-scrolls; the user's scroll position is theirs.*
2. *New content is nudged into view only when it sits below the viewport.* Adopted after the
   prototype, where a section held two or three cards, so "below the viewport" meant "the next
   card" and the nudge landed on the thing that had just appeared.
3. **Back to (1), with no condition.** `metabolomics_untargeted.csv` produces nine structural
   findings, and revealing the Explore section scrolled the user *past the card they were reading*,
   every time.
4. **(1) scoped to unbidden motion, at L62.** Zero scrolls the user did not ask for; **exactly one**
   where they pressed a navigation control. Run 5 filed `DRIVE-047` — the rail highlighted the active
   step and did not go there — and the absolute form of (3) required the app to keep shipping a
   control that lied about being one. **This is not (2) returning**: (2) failed on a *size-dependent*
   condition and *"did the user press a navigation control"* is categorical, so it cannot be correct
   at one dataset size and wrong at another. Two human drives pointed opposite ways and the
   distinction that resolves them is **who asked** — `DRIVE-006`'s row is titled *"**Auto**-scroll
   skips past the noticed card."*

**Why (2) failed is the lesson, and it is not "the threshold was wrong."** The revised rule had a
**size-dependent condition** in it, so it was correct at one dataset size and incorrect at another,
and **nothing in the interface could tell which one it was in.** Prefer a rule with no free
parameter over a rule with a tuned one — and treat a lesson learned on synthetic data as a
hypothesis until a real dataset has seen it.

**What replaces the nudge is `DESIGN_LANGUAGE.md` §05's placement rule, ruled at L47**: a response
to a press renders **at the control**. That has no free parameter either — where the button is does
not move when the data does.

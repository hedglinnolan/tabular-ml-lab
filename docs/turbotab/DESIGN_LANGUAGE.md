# TurboTab — design language

The written system behind `prototypes/design-language.html` and `prototypes/interview-feed.html`.
Open those in a browser to see it working; this file is the specification.

---

## 01 · What was wrong with the old UI

Diagnosis first, because the fixes only make sense against it. From a screenshot of the existing
Train & Compare page:

1. **The smartest thing whispered; the dumbest thing shouted.** The coach's ranking appeared as
   tiny orange "Would use RF's" badges repeated five times in jargon, while a yellow warning box
   about one edge case was the loudest element on the page — sitting *above* a flat, default
   Train button. Hierarchy inverted: advisory > primary action, metadata > recommendation.
2. **Every model got equal billing.** Identical white rectangles at identical sizes regardless of
   what the coach thought of them for *this* dataset. The layout rendered taxonomy (Boosting,
   Margin, Neural Net) instead of judgment — while `model_coach` was already bucketing into
   Recommended / Worth Trying / Not Recommended.
3. **Orphaned checkboxes.** Bare 16px squares floating below cards, with the card looking like the
   click target. A pure Streamlit artifact — you cannot put the widget inside the card.
4. **Semicolon prose where structure belonged.** "Fast gradient boosting; Handles missing values;
   Good for large datasets" — three attributes wearing a sentence costume, unscannable and
   incomparable.
5. **Emoji as brand.** 🚀 Boosting, 🧠 Neural Net. For a tool whose tagline is "publication-grade
   ML for research," emoji headers actively undercut the credibility claim.
6. **Three UIs stacked in the sidebar.** A page list, a workflow-mode radio, and a progress
   checklist — with the page list and checklist describing the same journey twice in different
   vocabularies. The checklist was the good idea, buried at the bottom.

**Rendering:** 1, 2, 4, 5 and 6 are fixable inside Streamlit. 3, and anything requiring smooth
reflow or instant response, are the framework ceiling.

---

## 02 · Color — every hue is a claim

Five meanings, five hues, no exceptions. If a color appears, the user may rely on what it asserts.
Values are light / dark; all pairs meet WCAG AA on their grounds.

| Token | Light / Dark | Means |
|---|---|---|
| `--accent` | `#0E7368` / `#45BFAF` | **Now.** The open question, the primary action, the current position. One accent moment per viewport. |
| `--ok` | `#2F7D46` / `#5CB877` | **Recorded.** Sealed, decided, provenance-backed. The left keyline of every decision sentence. |
| `--warn` | `#9A6B0F` / `#E0B45C` | **The coach's voice.** Noticings, deferrals, staleness. Advisory only — never errors, never decoration. |
| `--stop` | `#8C2B2B` / `#D97A6C` | **Invalid downstream.** Appears in exactly one component: the blocker band (§09) and its unresolved recorded artifact. Not error styling, not validation, not emphasis — a claim that numbers below this point cannot be trusted until the blocker reaches a terminal state. |
| `--ink` / `--ground` | `#1C2B29` on `#F7F8F6` | **The page.** Teal-biased neutrals — cool paper, not default grey. Muted `#5B6B68`, hairlines `#DCE3E0`. |

`--stop` is the strongest claim in the palette, and it is reserved to the one component allowed
to make it. The ANSI Z535 evidence is that red uniquely conveys the top severity tier; the
redundant-coding doctrine (§09) is why the tier is *also* carried by signal word, silhouette,
and grammar — color is a reinforcing channel, never the sole one.

Charts use a separate categorical ramp so semantic color stays semantic:
`--c1 #0E7368` (teal), `--c2 #4A5FA5` (indigo), `--c3 #A85C3C` (clay), `--c4 #6B7F8C` (slate).

Both themes are first-class. Define the palette as custom properties on `:root`, redefine the
tokens under `@media (prefers-color-scheme: dark)`, then again under `:root[data-theme="dark"]`
and `:root[data-theme="light"]` so a viewer's toggle wins in both directions. Style components
through tokens only, never inside the media query.

---

## 03 · Type — three voices, three planes

The rule that makes the whole system legible:

> **If the app is speaking, it's serif. If the user is acting, it's sans. If the data is speaking,
> it's mono.**

A reader can tell narrative from control from evidence without reading a word.

| Role | Face | Used for |
|---|---|---|
| **Voice** | Charter / Iowan Old Style / Georgia | Questions (19px), recorded decisions (16px), finding titles. The narrative plane. Weight 700 for titles only. |
| **Action** | Seravek / Avenir Next / Segoe UI | Buttons, chips, labels, rationale. 13–14px, weight 600 on controls. Uppercase kickers at 10.5px, `.09em` tracking. |
| **Data** | SF Mono / Cascadia / Consolas | Column names, values, counts, state tags. Always `tabular-nums`; always in a bordered chip when inline with serif prose. |

Keep running text near 65 characters. Give headings `text-wrap: balance`.

---

## 04 · Component vocabulary

Eight components build every screen. Each owns its states; no page-specific variants.

| Component | Job | States |
|---|---|---|
| **Question block** | A branch point. Serif question, one line of rationale, 2–4 answer chips. Exactly one is visually primary when the coach has a recommendation. | `open` → collapses to Decision sentence |
| **Decision sentence** | An answered question as one serif sentence with mono values inline and a green keyline. The unit of the transcript *and* of the future methods section. | `recorded` · `stale` · `reopened` |
| **Finding card** | A coach noticing: thumbnail, serif claim, consequence, actions, evidence flag. | `open` · `deferred` · `dismissed` (undoable) · `applied` |
| **Before/after preview** | Two table snippets with changed cells highlighted, before→after statistics, optional distribution pair. Apply lives *inside* it. | `closed` · `open` |
| **Coach ledger** | Docked queue of deferred noticings. Never interrupts; badge count only. Items resurface inside the step they target. | `empty` · `queued(n)` |
| **Analysis map** | Sticky rail: one dot per section, tracks scroll, doubles as navigation. Carries the evidence tally. | `done` · `now` · `waiting` · `stale` |
| **Stale veil** | Downstream sections after an edit: desaturated, veiled, tagged, and `inert` — visibly recoverable, never deleted. | `stale` → `recomputing` → `fresh` |
| **Job chip** | Every computation over ~1s: spinner, plain-language label. Lives in the rail; the page never freezes silently. | `running` · `done` (auto-clears) |
| **Evidence flag** | One-click "this goes in the paper." Flagged items accrue to the tally and pre-fill Report Export. | `unmarked` · `marked` |

**One card per step.** A step is one card. Sub-questions stack *inside* it, with answered ones
collapsing to a compact attributed row at the top. A new card per sub-question breaks progressive
disclosure.

---

## 05 · Motion — cause and effect only

1. **Settle.** An answered question collapses into its decision sentence in 250ms ease-out. The
   collapse is the visual receipt that the choice was recorded.
2. **Arrive.** New sections grow downward from the point of consequence (300ms, 6px rise).
3. **Propagate.** Staleness sweeps downstream in document order, ≤150ms per section, so the user
   watches their edit's blast radius draw itself.
4. **Work is visible.** Anything over ~1s shows a job chip immediately, named in plain language —
   never a bare spinner.
5. **Nothing else moves.** No ambient animation, no hover theatrics beyond elevation, and
   `prefers-reduced-motion` collapses everything to instant state changes.

**Scroll: the page never moves the viewport.** Cards build downward and the user's scroll follows
them. There is no condition under which the interface scrolls on the user's behalf.

**This section implies a rendering requirement, and it was measured rather than asserted.** Rule 1
says the settle *is* the receipt that a choice was recorded. A renderer that rebuilds the page
wholesale on every change cannot honor it, and the reason is mechanical: **`transitioncancel` does
not fire when an element is removed** — the transition ends with its target and there is nothing
left to dispatch on. So a repaint cannot even *report* what it interrupted; it has to infer the
count by subtraction. A receipt that restarts from scratch mid-flight is not a receipt, and the
focus ring is lost the same way, because the focused node is gone and focus falls to `<body>`.

**So whatever renders this app must mutate in place, not repaint.** Measured in
`prototypes/recipe-lattice.html` (`L31`), on a case where moving the lens changes 6 of 16 cells and
a repaint rebuilds all 24 nodes to move those 6. **This is a requirement, not a stack choice** —
vanilla JS mutates in place perfectly well, and `GUIDED-073` remains open on whether a framework
earns its cost.

This is the third position on one rule, and the history is the argument (`DRIVE-006`). The original
rule was *never auto-scroll*. Building the prototype appeared to disprove it, and it was revised to
*new content is nudged into view only when it sits below the viewport — never yank a user who has
scrolled up to read.* That revision was written against a prototype where a section held two or
three cards, so "below the viewport" meant "the next card", the nudge landed on the thing that had
just appeared, and it read as helpful.

**The prototype's lesson did not survive a dataset with many findings.**
`metabolomics_untargeted.csv` produces nine structural findings; revealing the Explore section
scrolled the user from the middle of those findings to the top of a section below them — past the
card they were reading, every time. The drive became: read, get yanked, scroll back, repeat.

What is worth extracting is not "the threshold was wrong" but **why a threshold was there at all**.
The revised rule had a size-dependent condition in it, so it was correct at one dataset size and
incorrect at another, and nothing in the interface could tell which one it was in. The rule that
replaces it has no condition, which is why it cannot be wrong at the next scale. Prefer a rule with
no free parameter over a rule with a tuned one, and treat a lesson learned on synthetic data as a
hypothesis until a real dataset has seen it.

### 05.1 · Acknowledgment — the voice that reports

Settle and arrive describe *motion*. Neither says what the app **says**, and the product owner drove
the app and found controls labeled `Earmark it` and `Show me` that named themselves and nothing else
(`DRIVE-003`, `DRIVE-004`). So this is the missing rule, and it is a **distinct voice from the coach**:
§06.3's coach observes and waits. Acknowledgment reports. It never advises, never hedges, and never
appears before there is a fact to state.

**One sentence, seen at three moments.**

| | | |
|---|---|---|
| **Before** | The control states what it will do — on hover and to a screen reader. | *"Records this and brings it back at Explore."* |
| **During** | That same sentence is the job label, so what is happening is what was promised. | *"Recording this and bringing it back at Explore…"* |
| **After** | The subcard collapses, and the row that replaces it says what was done. | *"One row is one person; each person's records are combined into one before anything is held out."* |

Four rules govern it:

1. **A control whose effect cannot be stated in one sentence should not exist.** The effect table is
   therefore the test of whether a control has earned its place, not documentation of one that has.
2. **Never a bare verb.** `Earmark it`, `Apply`, `Show me` name the mechanism. The sentence names the
   *consequence* — what changes, where it lands, and whether it can be undone.
3. **The after-sentence is a quotation, never a composition.** It is the sentence the *record* holds,
   read back verbatim. This is not tidiness: it makes the acknowledgment and the transcript the same
   string by construction, so the interface cannot report an effect the record does not carry, and a
   promise the server did not keep surfaces as a visible disagreement instead of as a reassuring
   sentence the page made up.
4. **An action that disappears has not acknowledged anything.** An answered question leaves the plan,
   so without a collapsed row the card simply vanishes and the user infers success from an absence —
   which is the one thing §09 reserves green for.

Refusals are acknowledgments too, and they carry the reason: *"The test set is already sealed, so a
person's rows cannot be combined now."* An action that declines and says nothing is
indistinguishable from one that broke.

### 05.2 · Identity continuity — what the motion is *for*, and why none of it runs today

§05 states three motions and never says what they are **for**. That omission is why the section
reads as polish, and it is not polish. Recorded here because it is a design principle the product
owner named as load-bearing, and because it existed only in a conversation until this paragraph.

> **Motion's job is to preserve identity across a state change**, so the user never loses track of
> what became what.

That is the whole of it. Settle is not "a nice collapse" — it is the user watching *their question*
**become** *their decision sentence*, which is what makes the sentence feel earned rather than
issued. Arrive is the new section visibly caused by the answer above it. Propagate is the edit's
blast radius drawing itself across objects that persist while their meaning changes. In every case
the same object survives the transition wearing a new state, and the user's model of the document
survives with it. An object that is destroyed and replaced teaches nothing, however smoothly it
fades.

**None of the three can currently execute, and the reason is mechanical rather than aesthetic.**
Measured on `turbotab/web/index.html` at `58bab10`:

| | |
|---|---|
| `innerHTML =` assignments | **92** |
| node-owning writes (`ownChild` / `appendChild`) | **22** |
| `startViewTransition` | **0** |
| `getBoundingClientRect` (FLIP) | **0** |
| `.animate()` (Web Animations) | **0** |
| `transitionend` listeners | **0** |
| CSS `transition:` / `@keyframes` | 15 / 3 |

So the app has hover-and-state motion and **no mechanism at all for animating a change of
content.** Settle needs the answered question's node to survive its own collapse; a repaint destroys
it. Arrive needs the section to exist before it animates; a repaint creates it already-final.
Propagate needs downstream sections to persist while their appearance changes; `stale_downstream`
has no reader at all (`GUIDED-094`). The app today is a well-typeset document that redraws.

**And there is a browser primitive built for exactly this, which the codebase uses zero times.**
`document.startViewTransition()` snapshots the DOM before and after a change and morphs elements
matched by `view-transition-name`. The identity continuity comes from the browser rather than from
the renderer — which means **the wholesale repaint can stay.** That matters, because the repaint is
a defended architecture and not an accident: the server owns the record, so a full re-render cannot
desynchronize from it. If this holds, the choice `GUIDED-073` frames as *repaint vs mutate vs
framework* is really *repaint + View Transitions*, and no stack decision is required.

`[verify-at-build]` — **support has not been measured against our three deployment targets.** Chromium
111+ and Safari 18+ are the figures I hold and they are recollected, not read. A Tauri/WebView2
desktop build is almost certainly fine; a research enclave running an older browser may not be, and
that is the target that decides it. Feature-detect with an instant fallback either way. Measure
before building on it.

**Be judicious. This is the rule that keeps the principle from eating the app.** Identity
continuity is expensive attention — it tells the user *this thing is the same thing* — and an app
that says that about everything has said it about nothing. §05 rule 5 already forbids ambient
animation; this subsection must not be read as license to revisit that. Spend the continuity only
where **identity would otherwise be lost**, which is a short and closed list:

1. **Settle** — the question becoming its decision sentence.
2. **Arrive** — the section caused by the answer directly above it.
3. **Propagate** — staleness crossing objects that remain on screen.
4. **The working table under a reshape** — join, merge, split. `PRODUCT_VISION.md` §04b makes this
   the argument for why Guided must own multi-file assembly at all: *the user is empowered to decide
   correctly only when they can watch their working data morph.*

Everything else changes state instantly. A fourth item added to that list is a design decision, not
an implementation detail, and belongs in a loop prompt rather than in a renderer.

**Asked and ruled, 2026-08-03 — the collapsed-remainder expand does NOT get a slot.** `GUIDED-149`
bounds what the Explore stack pushes and collapses the rest into a counted, typed affordance, and
the question is whether that expand joins the closed list. **It does not, and the expand is
instant.**

Three reasons, in the order they bind. **The criterion is not met**: the four entries are all a
*decision and its consequence* — a question becoming its sentence, a section caused by the answer
above it, staleness crossing what remains on screen, a table morphing under a reshape the user
chose. An expand is **disclosure, not consequence**, and §05 rule 5 governs it directly: *only
consequences move.* **The scarcity argument is this section's own**: continuity is expensive
attention, an app that says *this is the same thing* about everything has said it about nothing, and
expanding a list is the most ordinary interaction the app has — spending the vocabulary's scarcest
signal there devalues it at the three places it is load-bearing. And **the mechanics agree**: the
app has no mechanism for animating a change of content, so a fifth slot pulls in `GUIDED-073`'s
stack decision, which is deliberately unbuilt.

Recorded here rather than in the loop prompt so the next reader meets the ruling where the list is,
not where one build happened to need it.

**One thing this section cannot yet source, stated rather than papered over.** Item 4 is the single
load-bearing design assertion in `PRODUCT_VISION.md` that resolves to no evidence — an empirical
claim about whether animating a transformation improves a viewer's ability to follow it, asserted
because it is intuitive. This project does not let a vitamin conversion factor ship on intuition and
should not make an exception here. The literature exists (animated transitions in statistical
graphics; Heer & Robertson, InfoVis 2007, is the anchor I would start from and have not read in
primary). **Proposed and unscheduled: `research/INTERACTION_PACK.md`**, built under the same
discipline as the four science packs — sourced claims, SETTLED / CONVENTION / DISPUTED,
`[verify-at-build]` on anything not read from primary text. It is D-track work, and it has nothing
to attach to until the journey has an end, so it sits behind Explain and Report.

**On Apple's HIG specifically**, since it was asked and the answer should not be re-derived: perhaps
a fifth of it transfers to a research tool in a browser, and this document has already derived most
of that fifth independently — deference and content-over-chrome as *one accent moment per viewport*
(§03), progressive disclosure as the interview, direct manipulation as preview-before-apply. The one
idea genuinely worth taking is the one above: Apple's transitions preserve object identity across
state changes, and that is the vocabulary this section was missing. Take the principle; do not
import the platform conventions, which are about navigation idioms this app does not have.

---

## 06 · Voice — how the app talks

1. **Questions are single and concrete.** One decision per block, named values, no compound asks.
   "Add the interaction `age × creatinine`?" — never "Configure feature engineering."
2. **Decisions are past-tense and exact.** "184 rows were sealed" — a count, a verb, no adjectives.
   If the sentence couldn't appear in a methods section, it isn't a decision sentence.
3. **The coach observes; it does not command.** Noticings state evidence, consequence, and option,
   then wait. Amber text never says "you should"; it says "this usually means."
4. **Uncertainty is stated as uncertainty.** Probe results say "on training folds only." The UI
   never borrows confidence the math doesn't have.
5. **The app never speaks in the user's name.** Interpretive claims are left as authored gaps —
   the manuscript's `[AUTHOR REQUIRED]` rule, applied to the interface.

---

## 07 · Figures — interactive in-app, journal-format on export

The duality that makes the export credible:

- **In-app**: hover tooltips on histogram bars, heatmap cells and ROC operating points; clickable
  legends to toggle series; visible axis labels.
- **Journal view**: a toggle re-renders the same figure as it will be published — serif type,
  greyscale, series distinguished by **dash pattern rather than color alone**, printed *r* values
  in correlation matrices, proper ticks, numbered caption.
- **Export emits the journal version** — SVG and PNG at 3×. Because journal rendering uses literal
  hex colors rather than CSS variables, the exported file is self-contained and rasterises cleanly.
- **Tables offer LaTeX** (booktabs) instead of image export, matching `ml/latex_report.py`.

What the user sees in Journal view is exactly what lands in the manuscript. That is the point.

---

## 08 · Build rules

- Structural devices must encode something true. Numbered markers only when the content genuinely
  is a sequence.
- Wide content (tables, diagrams, code) scrolls inside its own `overflow-x: auto` container; the
  page body never scrolls sideways.
- Lay out sibling groups with flex/grid and `gap`, not per-element margins.
- Use `inert` for interaction locks, not `pointer-events` alone — the latter leaves veiled controls
  in the tab order.
- Give keyboard focus a visible state; respect `prefers-reduced-motion`.
- Custom controls carry real `role` and `aria-*`, and toggle labels must stay in sync with state.

---

## 09 · Question grammar — three types, three moods, three silhouettes

The Router asks three kinds of question (`ROADMAP.md`, the routing constitution), and the user
must be able to tell which kind is in front of them **before reading a word**. Each type gets a
distinct grammatical mood, a distinct silhouette, and a distinct verb. Color marks *state*
(teal now, green recorded, amber advisory, `--stop` invalid-downstream), never type. No channel
carries the distinction alone — silhouette, grammar, typography, and color are redundant by
design, because any single channel fails (habituation, color-blindness, skimming).

Sources, for the record: GOV.UK warning text / question pages / check-your-answers; USWDS alert
taxonomy; ANSI Z535 signal words; aviation checklist doctrine (Degani & Wiener; Gawande's killer
items); the WHO surgical time-out; GitHub's danger zone; the MOJ interruption card; Jarrett's
question protocol; teach-back consent; the CDS hard-stop/attestation literature.

### FACT — mood: interrogative · silhouette: flat inline row · verb: answer

- The question is the heading: plain, sentence case, serif, data terms in mono.
  *Is `ward_id` categorical?*
- The lightest object on screen: no border, no icon, no background tint. Teal marker on the
  currently-asked one only.
- Every FACT carries a "Why we ask" disclosure that names who consumes the answer and what for.
  A FACT that cannot state its consumer is a question we have no right to ask.
- Answered → collapses to a recorded row: label, value, green keyline, Change link. (The
  existing Decision sentence, unchanged.)
- **Rendered skips are NOT green.** Green means a human recorded it. A skipped FACT is a muted
  neutral row: mono provenance clause, sans reopen affordance.
  *Not asked: `age` read as numeric — 100% parseable. — Ask me anyway.*
  Skips group together so their density reads as "machine work" at a glance. The
  "Ask me anyway" click-rate is trust telemetry; log it.

### CHOICE — mood: imperative proposal · silhouette: bordered before/after card · verb: decide

- Verb-first headline naming the operation and its object in mono, one serif consequence
  sentence beneath. *Impute missing `age` with the median (n=142).*
- The **only** element with a two-pane before/after body — that split is its silhouette.
  Neutral border while open; the whole card settles into a green recorded row after the
  decision, whichever way it went.
- Buttons are outcome-labeled and symmetric in weight: **Apply repair** / **Keep as is**.
  Never OK/Cancel; never a styled "yes" against a de-emphasized "no" — declining is as easy
  and as dignified as accepting.
- Amber may appear *inside* the preview when the engine annotates a caveat; never on the
  card frame.
- Never auto-answered, at any confidence — the engine may do the work, but the human confirms
  (do-confirm doctrine, and the constitution's pre-selection rule).
- Deliberately kept **below** the interruption hierarchy: inline in flow, no modal, no band.
  CHOICE cards recur; their frequency must never erode the blocker's authority. Habituation
  starts at the second exposure — the blocker treatment survives only if nothing else wears it.

### CONSEQUENCE — mood: declarative, then first-person · silhouette: full-width interruption · verb: resolve or attest

- Opens with a signal word that appears nowhere else in the product — small-caps **BLOCKER** —
  then a declarative statement of mechanism and concrete consequence, never a hedge:
  *`abx_escalation_score` may encode the outcome. If it is computed after the outcome occurs,
  every accuracy number downstream is invalid.* Then the question.
- A full-width band that breaks the page rhythm: heavy rule, larger type, the reserved `--stop`
  color. One reserved geometric glyph (a notched square), used by blockers alone — exclusivity
  is what makes a shape semantic.
- Ranked first, always pushed, never skipped at any confidence — the existing constitution,
  now with a matching costume.
- **Two exits, both terminal:**
  1. **Resolve** — spawns the relevant CHOICE card inline (e.g. drop the column).
  2. **Acknowledge and proceed** — requires typing a sentence that restates the specific risk
     with the specific object: *"I am keeping `abx_escalation_score` although it may leak the
     outcome."* Object-specific, not a generic "PROCEED" (generic strings habituate). Paste is
     allowed (blocking paste harms accessibility and adds nothing). The prompt is neutral and
     factual — no shaming, ever. Recorded verbatim with timestamp; surfaced afterward as a
     distinct `--stop`-flagged artifact in the review, never green.
- **A terminal state is guaranteed.** A blocker never re-fires on the same facts after
  resolution or acknowledgment. (TurboTax's own CompleteCheck loop is the documented
  anti-pattern: a flag that cannot be satisfied teaches contempt for all flags.)
- **Budgeted.** Blockers work only while rare and every one legitimate. One false-positive
  blocker that cannot be cleanly resolved costs more trust than ten missed advisories. New
  blocker classes are added to the Router the way killer items are added to a checklist: only
  steps dangerous to skip *and* sometimes skipped.

### The register rule

Serif/sans/mono and the three moods are reinforcing channels, not the sole discriminators.
Every type distinction must survive with typography removed (silhouette + grammar) and with
color removed (silhouette + signal word).

### The recorded-absence rule — "nothing to do here" is an answer, and it gets written down

**Wherever a question has a *nothing to do here* option, that option is a recorded answer with
its own sentence — never the absence of a record.** *The absence of a restriction is a claim, and
a claim needs a record.*

Three places it already holds, and the shape is identical each time:

| Question | The "nothing" answer | What it would otherwise be indistinguishable from |
|---|---|---|
| Eligibility (§04) | *No, the study is about everyone here* | the eligibility question never being asked |
| Feature selection (§06) | *Every column goes to the models* | the selection step never being reached |
| The seal's basis (§03) | `cross_sectional` — verified, not merely ungrouped | `undetermined`, which is a different claim entirely |

The seal is the sharpest case and the reason this is a rule rather than a habit. `group_col: None`
was the state a verified cross-sectional seal *and* a failed detection both produced, so a
consumer could not tell "we checked, and rows do not repeat" from "we could not tell". That is
not a missing feature; it is **two different claims rendering as one**, and the fix was to make
the confident answer say itself out loud rather than be inferred from an empty field.

The general form, and why it is a design rule rather than a data-model preference:

- **A reader of the methods section has to be able to tell.** *"No exclusion criteria were
  applied"* is a sentence a paper can carry. *"The eligibility question does not appear in the
  record"* is not, and nothing downstream can turn the second into the first.
- **Absence is not falsifiable.** A recorded `everyone` can be contradicted by evidence and can
  be revisited by the user; an empty field can only be guessed at. Constitution §03's whole
  argument — *leaking and disclosing is the refuse branch, leaking behind a lock icon is the
  assert-something-false branch* — is this rule applied to one clause.
- **It costs one decision entry.** The record already exists; the only work is refusing to treat
  a default as an answer.

So: when a step gains a question, ask what its *nothing* answer is, and give it a sentence. And
when reviewing a step, the check is *"can I tell a step that concluded nothing from a step nobody
reached?"* — if not, the record is missing a claim rather than missing a value.

---

## 10 · Education in the flow — model left, learn right

When does the user model, when do they learn, when both? One answer: **modeling is the left
column, learning is the right panel, both means both columns. There is no third place.**
Education never interrupts the interview and never lives in a modal.

Four layers, outermost first:

1. **The card is self-sufficient.** Every question carries the one sentence of rationale
   needed to answer it. A user who never opens anything else can finish a defensible analysis.
   (Expertise reversal: explanations that help novices actively slow experts — so the default
   surface is minimal.)
2. **"Why?" opens in place.** A disclosure on the card expands two or three sentences of the
   app's voice (serif) — why this question, what the answer changes. No navigation, no panel.
   (Split-attention: explanation adjacent to the thing explained, or it costs more than it
   teaches.)
3. **The side panel teaches on the user's data.** Opt-in, persistent, bound to the current
   step: the concept explained with *their* columns and *their* numbers, not abstractions.
   Productive-failure ordering where possible — show the consequence on their data first,
   name the concept second.
4. **The Theory Reference is the appendix.** Stable IDs (`theory_anchors` keys), linkable from
   any layer, never required. The registry pair gets the missing key-match test when it is
   extracted (`FEATURE_PARITY.md`, "two specific things to watch").

Two mechanics bind the layers to the flow:

- **"Save to my review list."** Any explanation can be earmarked without leaving the
  interview — same gesture as the evidence flag, different destination. The review list is
  the user's private syllabus; it never auto-surfaces.
- **Read-as-draft is a faded worked example.** The draft manuscript accumulating in the right
  panel *is* layer-3 education: it shows the user what their decisions look like in
  methods-section prose while they still have time to change them. Named here so the builder
  treats it as pedagogy, not decoration.

---

## 11 · The evidence badge — a token, and deliberately not a component

All four domain-research threads arrived at the same recommendation without being
asked for it: **surface the epistemic status of every claim the app makes.** The
clinical thread says why in one sentence — *"that single design decision is what would
make TurboTab trustworthy to a methodologist, because it makes the tool's epistemic
position legible rather than uniformly confident."*

**It is a token, not a card type, and that is load-bearing rather than tidy.**
`DOMAIN_PACKS.md` guard #1 says a pack may not add interview components; a badge that
arrived as a new card would be the packs buying breadth with the design language, which
is the thing guard #1 exists to prevent. This adds no silhouette, no mood, no verb. It
is a small mono chip rendered **inside** an advisory that already exists — beside the
`derived`/`convention` marker the card already carries, in the same row.

| Badge | The field's position | Rendering obligation |
|---|---|---|
| **SETTLED** | Methodological consensus. A tool asserting the opposite would be wrong. | May be a pre-selected default, **with its reason shown**. |
| **CONVENTION** | No strong evidence base, but field expectation. Deviating invites reviewer friction. | May be pre-selected, and **must be stated as convention, never as fact**. |
| **DISPUTED** | Live disagreement among competent methodologists. | **Never defaulted silently.** Both positions stated, and a sensitivity analysis offered. |

**Type and color follow the existing rules rather than extending them.** The badge is
mono, because it is *data about the claim* rather than the app's voice (§03). It takes
no new hue: SETTLED wears `--ok` because the field has recorded a position, CONVENTION
wears `--warn` because that is the coach's advisory voice and a convention is exactly
advisory, and DISPUTED wears **`--ink` on `--surface-2` — deliberately not `--stop`**.
§02 reserves `--stop` for the blocker band alone, and a disagreement among competent
methodologists is not an invalid downstream; treating it as one would spend the
strongest claim in the palette on the honest case.

**It sharpens the three markers rather than replacing them.** `derived`/`convention`/
`offered` describe **the app's** confidence; SETTLED/CONVENTION/DISPUTED describe **the
field's**. The second is the one a reviewer can check. They are not a translation — a
compatibility table, because `offered` admits all three: pooled QC rows are not
participants (SETTLED) and the app still only *offers* the exclusion, because acting on
a high-confidence detection whose consequence is irreversible if wrong is what every
pack's hard-stop list forbids. **Settled science and a withheld hand are compatible,
and that combination is one of the most important in the product.**

**The consequence for the governing rule.** *"The app may be silent, and it may refuse,
but it must never assert something false"* gains a fourth mode: **the app may state
that the field disagrees.** That is not hedging. On a DISPUTED claim it is the only true
sentence available, and the badge is what makes it a sentence the app can actually
write.

**Every badge names a source, and the source is checked.**
`docs/turbotab/tools/evidence.py check` runs in the pre-commit gate and resolves the
named file and the named section. Its limit is exactly `ledger.py check`'s and is stated
in the same breath: it verifies that a source is **named and resolvable**, never that
the claim is faithful to it.

## Open items recorded, not resolved

- **Attestation polymorphism.** Whether typed attestation stays meaningful for repeat users is
  unstudied. Do not vary the required sentence yet (it may read as hostile); revisit when there
  is telemetry.
- **Three-voice learnability.** Whether users internalize serif/sans/mono as a code over
  sessions is untested — which is why the register rule in §09 demands the system survive
  typography's failure.
- **Blocker budget number.** The literature says "few" and gives no number. Track
  blockers-per-session in the value-check harness; alarm on trend, not threshold.

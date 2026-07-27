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

**Scroll:** new content is nudged into view *only when it sits below the viewport*. Never yank a
user who has scrolled up to read. (This revises an earlier "never auto-scroll" rule that building
the prototype disproved.)

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

## Open items recorded, not resolved

- **Attestation polymorphism.** Whether typed attestation stays meaningful for repeat users is
  unstudied. Do not vary the required sentence yet (it may read as hostile); revisit when there
  is telemetry.
- **Three-voice learnability.** Whether users internalize serif/sans/mono as a code over
  sessions is untested — which is why the register rule in §09 demands the system survive
  typography's failure.
- **Blocker budget number.** The literature says "few" and gives no number. Track
  blockers-per-session in the value-check harness; alarm on trend, not threshold.

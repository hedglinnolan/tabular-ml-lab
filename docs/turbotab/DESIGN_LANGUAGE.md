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

Four meanings, four hues, no exceptions. If a color appears, the user may rely on what it asserts.
Values are light / dark; all pairs meet WCAG AA on their grounds.

| Token | Light / Dark | Means |
|---|---|---|
| `--accent` | `#0E7368` / `#45BFAF` | **Now.** The open question, the primary action, the current position. One accent moment per viewport. |
| `--ok` | `#2F7D46` / `#5CB877` | **Recorded.** Sealed, decided, provenance-backed. The left keyline of every decision sentence. |
| `--warn` | `#9A6B0F` / `#E0B45C` | **The coach's voice.** Noticings, deferrals, staleness. Advisory only — never errors, never decoration. |
| `--ink` / `--ground` | `#1C2B29` on `#F7F8F6` | **The page.** Teal-biased neutrals — cool paper, not default grey. Muted `#5B6B68`, hairlines `#DCE3E0`. |

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

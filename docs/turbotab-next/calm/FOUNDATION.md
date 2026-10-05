# The calm foundation

Nolan, 2026-10-05, after clicking three prototypes of the living methods section: *"all of this
feels incredibly busy even for me and I am developing the app. The colors and dashed lines and all
of it makes it really hard to know what I need to click and where."* Then, on a calm redraw of one
question: *"I like this version better since it is closer to the progressive disclosure I need."*
On fairness: *"we did not make a good faith effort to synchronize the design decisions made on each
part."*

This file is the shared foundation every presentation design builds on. It amends
`DESIGN_LANGUAGE.md` §02–§04 and the presentation parts of BLUEPRINT §11 and §11.4. The methods do
not change: decisions are still asked, stated or silent by their consequence; every option is
still previewed on the user's own data; every decision is still a sentence in the record. What
changes is how much is on screen at once and how loudly it is drawn.

References in this folder: `calm-screen.html` (the screen Nolan approved), `color-study.html` (the
color system on two real screens and a page of every role), `tokens.css` (the tokens, light and
dark), `system.json` (the same values as data).

## 1 · The test

**Within five seconds, a first-time user knows what to click.** Every screen and every review is
judged by this first. One primary action per screen, always in the same place (the bottom of the
card column, or the fixed footer on narrow screens). The choice color is reserved for the choice.

## 2 · The calm budget

Per screen:
- one focal region, the open question, plus the canvas that answers it;
- no dashed lines, hatching, or pills used as decoration;
- column names as plain text, never boxed chips;
- one type family (Source Sans 3) with tabular numbers; no serif/sans/mono "voices";
- at most one quiet label per option ("Recommended", "Common practice", "Not available yet");
- teaching behind one "Why does this matter?" disclosure;
- the record, the map and provenance one click away, never all on screen at once.

## 3 · Page zones

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ CHAIN   Data · Participants · Exposure · Confounders · Energy · Model · Results │
├────────┬──────────────────────┬──────────────────────────────────────────────┤
│ M      │ CARD                 │ CANVAS                                         │
│ a      │ Participants · step 2│ (the dynamic window: everything the open       │
│ n      │ Question             │  question does to the user's data)             │
│ u      │ options…             │                                                │
│ s      │ Why does this matter?│                                                │
│ c      │            Continue  │                                                │
└────────┴──────────────────────┴──────────────────────────────────────────────┘
```

- **Chain (top, one line).** The analysis as a sequence of stages, by the purpose's reporting
  guideline (STROBE-nut under inference, TRIPOD+AI under prediction). Done stages are marked,
  the current one is named, later ones wait. Clicking a done stage revisits it. This is the map:
  the provenance figure in its calmest form.
- **Card (left, 380–440 px).** Atop the card, the stage and its step ("Participants · step 2 of
  3"). Then the question, its options, the disclosure, and Continue.
- **Canvas (right, all remaining width; at least 60% of a 1440 px screen).** The dynamic window.
- **Manuscript (collapsible).** A slim rail at the far left, labeled "Manuscript", with the count
  of sentences. Opening it lays the manuscript over the card column, never over the canvas. On
  screens 1680 px and wider it can be pinned open as a third column. It updates with every
  recorded choice; the newest sentence is marked for a moment; asked slots show as blanks the user
  can click to answer; stated phrases can be clicked to change them.

## 4 · Color by role

| Role | Light | Dark | Means |
|---|---|---|---|
| Ground, surface, lines, text | warm paper | warm charcoal | the app is speaking |
| Hover, selected, Continue, focus | one indigo family (272°) | lighter indigo | the choice you are making |
| Canvas | cool lightbox | the deepest layer | your data is speaking |
| Data now | gray | gray | your data as it is |
| Data the choice touches | indigo | light indigo | exactly what the pointed or chosen option changes |
| Recorded | green, only beside saved sentences | | status, with words |
| Coach | amber, only for a noticing | | status, with words |
| Blocker | red, only for a real blocker | | status, with words |
| Comparisons | sage, plum, ochre, steel, clay, in that order | | model families, groups, lanes; never indigo |

Every pair meets its contrast target in both modes. The data pairs and the comparison palette
pass the color-blind checks. The color of the option you point at and the color of its effect on
the canvas are the same: that link is the lesson.

## 5 · The canvas grammar (visual routing)

The canvas must stay simple when a choice is simple and stay readable when it is not: a choice
can change several features at once, change what is fed where, or trade one good against another.
The engine already returns, for each option, up to three views from a closed vocabulary
(`row_flow`, `lineage`, `distribution`, `relationship`, `table_focus`, with storyboards), the first
primary, ranked by measured change (`turbotab/core/consequences.py`). The grammar adds one thing:
**a layout, chosen by the option's footprint.**

The footprint of an option, measured by the engine:
- **rows**: how many rows each flow step keeps or removes;
- **columns**: which columns change, each with a change magnitude;
- **routing**: which columns move between roles or into and out of the model;
- **angles**: the tradeoffs its method contract declares (§13), each a question with one view.

The layout follows from the footprint:

| Footprint | Layout | What the canvas shows |
|---|---|---|
| one column's values change | **Focus** | one large view (distribution or relationship) with the transform player |
| two or more columns change | **Strip** | a ranked strip of every changed column (top 12, then "and N more"), each with its before → after measure; the first is focused large; pointing or arrow keys move the focus |
| rows change | **Flow** | the participant flow with the changed step lit, then who leaves (one distribution), then how they differ from who stays |
| what feeds where changes | **Routing** | the lineage from raw columns to roles to the model's inputs; the re-routed paths drawn in the choice color; counts per role |
| the contract declares tradeoffs | **Angles** | two or three linked panels, each titled with the one question it answers, and a small option-by-question table that follows the pointer |

Rules:
1. The largest measured change is the primary view, and gets the most room.
2. At most three views are visible; any others sit behind "More angles".
3. One flip and one storyboard drive every view on the canvas at once ("Your data now" ⇄ "With
   this choice"); switching options morphs straight to the new result.
4. Views are linked: pointing at a column in the strip, or a step in the flow, highlights it in
   every view.
5. Gray is now, indigo is what the choice touches, in every layout.
6. **Under inference, no outcome-model estimate appears on the canvas before the plan is
   locked.** Seeing estimates while choosing invites choosing by the estimate. The canvas shows
   the data, its flow and its structure; Table 2 and "Which of my decisions mattered?" come after
   the lock.
7. A preview that shows nothing changing says so in one line rather than drawing an empty chart.

## 6 · Comparing structures fairly

Every competing structure (the Q&A card, the paper, the quest log, the map) is built from the same
kit: the same tokens, card, options, canvas, layouts, chain, manuscript, copy and data. The only
thing that differs is how the analysis is organized on the screen. A synchronization audit checks
this before Nolan sees them.

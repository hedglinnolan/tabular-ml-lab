# The calm foundation

Nolan, 2026-10-05, after clicking three prototypes of the living methods section: *"all of this
feels incredibly busy even for me and I am developing the app. The colors and dashed lines and all
of it makes it really hard to know what I need to click and where."* Then, on a calm redraw of one
question: *"I like this version better since it is closer to the progressive disclosure I need."*
On fairness: *"we did not make a good faith effort to synchronize the design decisions made on each
part."*

Nolan, 2026-10-06, after walking the four calm structures, on the quest log: *"the best option as
of right now, but we can do better to design it."* What it is for, before the fit: *"see what the
engine needs to surface that a user decides, dynamically build the tapestry to show the
consequences of their decision-making."* After the fit: *"show them the results and help them
interpret with the tapestry, let them decide what to include and not include in their manuscript
results section."*

This file is the shared foundation every presentation design builds on. It amends
`DESIGN_LANGUAGE.md` §02–§04 and the presentation parts of BLUEPRINT §11 and §11.4, and it is the
design the quest-log shell (SIZING P0.7) is built to. The methods do not change: the engine still
asks, states or keeps silent each decision by its consequence (those words never reach the screen,
§2); every option is still previewed on the user's own data; every decision is still a sentence in
the record. What changes is how much is on screen at once, how loudly it is drawn and, since
2026-10-06, how the analysis is organized: as a quest log of seven fixed stages.

**The tapestry** is Nolan's word for the canvas (2026-10-06): the large region on the right where
the user's data speaks. Before the fit it shows the consequences of each decision; after the fit,
the evidence behind each result. This file says "tapestry" throughout. The kit's code still names
the component `Canvas`, and other documents still cite §5 as "the canvas grammar".

References in this folder: `calm-screen.html` (the screen Nolan approved on 2026-10-05, from before
the quest log; still the reference for the card, its options and the tapestry), `color-study.html`
(the color system on two real screens and a page of every role), `tokens.css` (the tokens, light
and dark), `system.json` (the same values as data).

Rewritten on 2026-10-08 for the quest log, from Nolan's rulings of 2026-10-06 to 2026-10-08
(`HANDOFF.md`, "UI and scope discussion"; `V2_DEFINITION_OF_DONE.md`, the amendments of 2026-10-07
and 2026-10-08) and the crosswalk (`crosswalk/CROSSWALK.md`), and revised the same day after
review. At the end: what the rewrite replaced ("Superseded"), the one question still open for
Nolan, and the review's notes not taken.

## 1 · The test

**Within five seconds, a first-time user knows what to click.** Every screen and every review is
judged by this first. One primary action per screen, always in the same place: the foot of the card
column. Continue, "Confirm all 6" and Export all sit there. The one exception is Fit, which Nolan
placed on the analysis flowchart (2026-10-06 and 2026-10-08): on that screen it is the screen's one
primary action, at the flowchart's end, and the card's foot holds no button (§7). The choice color
is reserved for the choice.

## 2 · The calm budget

Per screen:
- one focal region, the open line on the card, plus the tapestry that answers it;
- one stage's lines on the card; the other stages' questions show only on hover (§3);
- no dashed lines, hatching, or pills used as decoration;
- column names as plain text, never boxed chips;
- one type family (Source Sans 3) with tabular numbers; no serif/sans/mono "voices";
- at most one quiet label per option ("Recommended", "Common practice", "Not available yet");
- **Two registers.** Nolan, 2026-10-05, on the kit's copy: *"a bit impenetrable… big on technical
  jargon"*. The card speaks plain language a researcher from another field understands at once:
  the question, the lede, each option's name and one-line consequence, the tapestry's caption, the
  readout, the coach line and the Angles questions ("Should sugar's calories replace other
  calories, or add to them?", not "a substitution or an addition"). The technical name of a method
  rides along quietly: an option's `term` sits on its top edge while it is pointed at or focused
  ("Known as Willett's cut-offs", "residual method", "substitution", "disjunctive cause
  criterion"), never as a second label. The manuscript keeps the methods register, because it is
  written for the paper. Every meaning and every number stays exact; where plain words would
  change a meaning, the term stays and is defined in place in at most 12 words ("Mediators are what
  sugar changes that in turn changes glucose").
  - **Three words stay quiet terms only** (2026-10-07): "exposure", "confounder" and "estimand".
    The card asks "What do you think affects glucose?"; the crosswalk writes *what you study* and
    *what you adjust for*.
  - **The engine's tiers never reach the screen.** Asked, stated and silent are the engine's words;
    Nolan found them "LLM-like" (2026-10-06). The screen says Decide · Confirm · For the record
    (§3).
  - **Every phenomenon gets its name in both registers** (2026-10-08): a plain sentence, with the
    technical name and its source riding quietly, so the researcher thinks *"huh, so that's what I
    call that problem when I'm discussing it with colleagues"* (`MODEL_FAMILY_CONTRACT.md` §0);
- teaching behind one "Why does this matter?" disclosure;
- the record, the flowcharts and provenance one click away, never all on screen at once.

## 3 · The quest log

**Desktop only (Nolan, 2026-10-05: "TurboTab will never be used on a phone").** Design for laptop
and desktop screens, from 1280 px wide up. A narrow window must not break: no lost controls and
no overlapping text. But no layout is designed for phones, and audits do not test phone width.

```
┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Your data · Your question · First look · Who's in · Models · Results · Write-up        ▰▰▰▱▱▱▱ │
├────┬───────────────────────────┬───────────────────────────────────────────────────────────────┤
│ M  │ CARD                      │ TAPESTRY                                                      │
│ a  │ Who's in                  │ (the dynamic window: before the fit, what the open line       │
│ n  │ Decide                    │  does to the user's data; after it, the evidence behind       │
│ u  │   answered lines          │  the open result)                                             │
│ s  │   the open line:          │                                                               │
│ c  │     question, options,    │                                                               │
│ r  │     Why does this matter? │                                                               │
│ i  │   lines still to come     │                                                               │
│ p  │ Confirm · 4               │                                                               │
│ t  │ For the record · 11       │                                                               │
│    │                Continue   │                                                               │
└────┴───────────────────────────┴───────────────────────────────────────────────────────────────┘
```

- **The quest line (top, one line).** The seven stages by name, and nothing else, except while a
  long fit runs: then Results reads "Results · fitting, about N min" (§7). The open stage is in the
  text color, the others quieter. Clicking a stage opens it on the card; pointing at it shows its
  questions (below).
- **The progress bar (top right).** Seven segments, one per stage (below).
- **Card (left, 380–440 px).** The open stage: its name, then its lines under their labels, the open
  line expanded, and the primary action at the foot.
- **Tapestry (right, all remaining width; at least 60% of a 1440 px screen).** The dynamic window
  (§5).
- **Manuscript (collapsible).** A slim rail at the far left (below).

**The seven stages** are fixed, the same for every analysis, in this order (Nolan, 2026-10-06/07):
1. **Your data:** the file, its joins and stacking, the codebook, the lens, which way round the
   table is, and what each column is. One ledger of every column with its proposed reading: Decide
   asks only the readings that change a number, and the rest sit in the Confirm sweep (ruled
   2026-10-08).
2. **Your question:** the outcome; then how the people were chosen and whether anything was
   assigned (observed as they were, by default), because that changes which goals and wordings are
   valid (disagreement 10); then the goal (Describe, Estimate an effect, Predict) worded around the
   outcome, and its shape, filtered by domain. "No single outcome" starts Describe (ruled
   2026-10-08). A second goal adds a track.
3. **First look:** shaped by the goal, a guide plus a gallery walked on the first visit. It shows
   what it notices and decides almost nothing itself: each noticing is decided in the stage whose
   answer it changes, and returns there as a line.
4. **Who's in:** who is kept, and what each blank means; under Predict, the drawing of the held-out
   rows. A Predict track whose outcome an earlier track has already read draws none: it validates
   by resampling the whole procedure instead (crosswalk, "Settled here"). It closes on the
   participant flowchart (§7).
5. **Models:** what is fit and how, including how the kept blanks are filled, in each track (ruled
   2026-10-08). Classic's feature engineering, feature selection and preprocessing fold in here,
   because they are per-model choices. It closes on the analysis flowchart, where Fit is pressed
   (§7).
6. **Results:** every result as an exhibit (§8).
7. **Write-up:** the manuscript at full width, and the export (§9).

**Dynamic sections.** The stages are fixed; what is inside them is not. Each stage's sections and
lines are built from the stage registry (SIZING P0.4) for this table, this goal and this lens. A
section appears only when the engine needs something from the user there: survey weights only when
the table has weights, usual intake only for recalls, QC drift only for omics. Under several goals,
the shared stages run once; Models and Results hold one section per track, in the tracks' order
(Estimate, Describe, Predict), and Write-up merges them. Sections come and go as answers change;
the stages and their order never do.

**Hover to see a stage's questions.** Pointing at a stage on the quest line, or focusing it from the
keyboard, opens a quiet panel under it that lists the stage's lines by label, each answered line
with its answer. Leaving closes it after the kit's 160 ms grace (`POINT_GRACE_MS`), so moving along
the line does not flicker. Clicking opens the stage on the card.
- **Any stage can be opened and read, ahead too.** A line whose earlier answers are missing reads
  "Waiting for: [the question]", with a link to that question, and cannot be answered yet
  (crosswalk disagreement 20). Pointing at it or clicking it shows only that line and its link: it
  lights nothing on the tapestry and previews nothing, since what it would show depends on the
  missing answer. The one exception is First look's "Decide now", which opens a later stage's
  question early when every answer it needs is in.
- **A stage not reached yet** lists the questions it expects so far, without a count, under one
  quiet line: "May change as you answer".

**The progress bar.** Seven segments at the top right, one per stage, so that a long analysis never
feels endless (Nolan: "so users don't get discouraged").
- **A segment fills with its stage's objectives** (Nolan, 2026-10-06): each Decide line counts one,
  open noticings included, and the stage's Confirm sweep counts one, however many defaults it
  holds. For the record never counts.
- **A stage not reached yet shows empty,** never "0 of N".
- **A stage reopened by an earlier change drops back and says why,** in one plain line beside the
  bar and atop the stage's card, naming the answer that changed: "Your data reopened: the join
  added 12 columns, so what each new column is gets read again."
- The bar is drawn in the app's own ink, never in the choice color (§4). Pointing at a segment opens
  the same panel as pointing at its stage.

**Decide · Confirm · For the record.** Every line carries one of three labels (Nolan, 2026-10-06).
They are the card's section heads, in plain type, never chips or colors.
- **Decide:** a question the user answers, or an open noticing they decide or dismiss. Each counts
  as one objective.
- **Confirm:** a default set for the user, listed only when another choice would change a number on
  this table. A stage's Confirm lines sit together in its one sweep (below), and the sweep counts
  as one objective. Nolan: *"They ultimately need to own their results, but we can make their lives
  easier."*
- **For the record:** what was read or done with no choice that matters here. It sits collapsed in
  one quiet disclosure at the foot of the stage and never counts.

Previews, views and refusals are not lines: they draw on the tapestry. The stage registry maps the
engine's tiers onto the labels: a question the engine asks is a Decide; a default it states is a
Confirm when another choice would change a number, and For the record otherwise; a silent one is
For the record, or appears only in the export.

**The Confirm sweep, last.** Each stage ends its lines with at most one Confirm sweep, after its
Decides, because the Decides change which defaults apply. It appears only when the stage holds a
default whose alternative would change a number. One line heads it ("Here are the 6 other choices
set for you"), each choice sits beneath with its reason and can be changed, and one primary action
clears it: "Confirm all 6".

Two departures, both under a Predict track that holds rows out and both within the display-order
rule (§10), because what the held-out rows read must be fixed before they are drawn or scored:
- **Who's in** runs its sweep just before the held-out rows are drawn, so the drawing is the
  stage's last Decide (disagreement 6).
- **Results** runs its sweep just before the open-noticings gate and the opening of the held-out
  rows, since the threshold range and the calibration horizon in it are read by the held-out score
  (disagreement 16). The opening is then the last Decide the held-out score reads; what follows it
  is reading and placing exhibits. The sweep's one other default, the explanations' curve method,
  does not depend on the opening and rides in the same sweep, so the stage still has one.

Write-up's sweep holds one default, the small-cell threshold, and appears only under the lenses
that report participant-level cells (dietary, clinical and survey).

**Three levels of disclosure.** Every line opens in three steps (Nolan, 2026-10-06, for the Confirm
sweep; the quest log uses them for every line):
1. **The line:** plain words, with its answer or default and, for a Confirm, a short reason.
2. **Pointing or keyboard focus:** the line elaborates "ever so slightly", by one clause on what it
   does to this table, and the tapestry lights what it touches in the choice color, as pointing at
   an option does. The technical name rides on the line's top edge here and only here. A "Waiting
   for" line elaborates and lights nothing (above).
3. **A click:** the card expands to that line's options, each previewed on the tapestry, with "Why
   does this matter?" and a way back ("Back to Who's in", or Escape) to the list as it was. A
   choice holds in the card, marked "changed by you", until the primary action records it
   (`RECIPES_AND_TUNING.md` §6.3). Nolan: *"It makes for a nice dynamic UI design."*

The open Decide line is already at the third level: the card shows it expanded, with Continue at
the foot. Answered lines collapse above it to their answers, in the text color (green stays beside
saved sentences in the manuscript, §4); lines still to come wait below in the quieter color.

**Not available yet** (ruled 2026-10-07). Not every option on a design question must be supported.
A rare design or analysis outside v2, such as a crossover trial, or "only the direct part" of an
effect, stays on the card as an option with the quiet label "Not available yet". Choosing it shows
its reason in one line and an exit that keeps the work (for the direct effect, the whole effect),
never a dead end. The option is never hidden, so no one has to misanswer to get past it.

**The manuscript rail.** A slim rail at the far left, labeled "Manuscript", with the count of
sentences. Opening it lays the manuscript over the card column, never over the tapestry. On screens
1680 px and wider it can be pinned open as a third column. It updates with every recorded choice;
the newest sentence is marked for a moment. A question not answered yet shows as a blank the user
can click to answer. Any phrase can be clicked to change it, which opens the card that asks it: the
methods text changes only through its decisions. A sentence shaped by a noticing ends with a quiet
"noticed" link back to First look. The rail is a view, not a stage, and fills no segment of the
bar. It and Write-up are one document (§9).

## 4 · Color by role

| Role | Light | Dark | Means |
|---|---|---|---|
| Ground, surface, lines, text | warm paper | warm charcoal | the app is speaking |
| Hover, selected, Continue, focus | one indigo family (272°) | lighter indigo | the choice you are making |
| Tapestry | cool lightbox | the deepest layer | your data is speaking |
| Data now | gray | gray | your data as it is |
| Data the choice touches | indigo | light indigo | exactly what the pointed option, or the pointed line, changes |
| Progress | the text color on the line color | the same | how far each stage has come; a drop-back is said in words |
| Recorded | green, only beside saved sentences | | status, with words |
| Coach | amber, only for a noticing | | status, with words |
| Blocker | red, only for a real blocker | | status, with words |
| Comparisons | sage, plum, ochre, steel, clay, in that order | | model families, groups, lanes; never indigo |

Every pair meets its contrast target in both modes. The data pairs and the comparison palette
pass the color-blind checks. The color of the option you point at and the color of its effect on
the tapestry are the same: that link is the lesson.

## 5 · The tapestry's grammar (visual routing)

**The pivot.** Before Fit, the tapestry shows the consequence of a choice. After Fit, it shows the
evidence behind a result (§8).

The tapestry must stay simple when a choice is simple and stay readable when it is not: a choice
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

| Footprint | Layout | What the tapestry shows |
|---|---|---|
| one column's values change | **Focus** | one large view (distribution or relationship) with the transform player |
| two or more columns change | **Strip** | a ranked strip of every changed column (top 12, then "and N more"), each with its before → after measure; the first is focused large; pointing or arrow keys move the focus |
| rows change | **Flow** | the participant flow with the changed step lit, then who leaves (one distribution), then how they differ from who stays |
| what feeds where changes | **Routing** | the lineage from raw columns to roles to the model's inputs; the re-routed paths drawn in the choice color; counts per role |
| the contract declares tradeoffs | **Angles** | two linked panels, each one question and one picture, and a small option table (the option and at most two short answers) that follows the pointer |

Rules:
1. The largest measured change is the primary view, and gets the most room.
2. At most three views are visible; any others sit behind "More angles".
3. One flip and one storyboard drive every view on the tapestry at once ("Your data now" ⇄ "With
   this choice"); switching options morphs straight to the new result.
4. Views are linked: pointing at a column in the strip, or a step in the flow, highlights it in
   every view.
5. Gray is now, indigo is what the choice touches, in every layout.
6. **No estimate and no score before Fit.** Seeing estimates while choosing invites choosing by the
   estimate.
   - Under Estimate and Describe, no estimate is served before the track's plan is locked, and
     pressing Fit locks it (§7). Under Predict, no score appears before Fit opens Results.
   - **The outcome's own views open at their gates** (ruled 2026-10-08, crosswalk question 1). Every
     such view is recorded.
     - Under Estimate, the outcome alone opens after Who's in, on the rows analyzed, and the outcome
       beside another column waits for the lock.
     - Under Describe, the same as Estimate, with the track's own lock.
     - Under Predict, both open after the held-out rows are drawn, on the training rows.
     - Under several goals, every shared view follows the strictest gate among the tracks: with an
       Estimate or Describe track, the outcome beside a column waits for that track's lock, whatever
       a Predict track would allow.
   - **Before its gate, the outcome shows its reading only:** its kind, unit, codes and level names,
     with the event lit. Its values, their spread and its counts stay out of every view: Your
     question's cards, First look's index and Your data's ledger alike. Two things read it
     earlier, because a stage cannot do its work without them. The participant flow counts the rows
     without an outcome, since Who's in decides who is kept by it. A structural check that reads it,
     such as a batch that matches the outcome exactly, is said as a one-line verdict and never
     drawn (`FIRST_LOOK_BRIEF.md` §6.1, O2). This narrows the crosswalk's Focus on the outcome in
     Your question, "its distribution or its levels", to its levels.
   - **Before the outcome is named,** Your data draws every column as a column, because its readings
     are settled on their values and no column is the outcome yet. Naming the outcome takes its
     values out of every view until its gate, and For the record says that its column was drawn in
     Your data before it was named. This is the one exception to the gates, and it is open for
     Nolan (below, "Open for Nolan").
   - The models' live ranking is outcome-blind: it reads what each model would be given, never a
     relationship with the outcome or a score; only the outcome's own counts, such as events, may
     enter (`MODEL_FAMILY_CONTRACT.md`, ruling 1).
   - Before Fit, the tapestry shows the data, its flow and its structure. Table 2 and "Which of my
     decisions mattered?" come after the lock.
7. A preview that shows nothing changing says so in one line, over your data now, rather than
   drawing an empty chart; so does an option that is not available.
8. **The tapestry is never empty.** At rest, before anything is pointed at, it shows "your data
   now" for the open line: the column or columns it is about, drawn in gray, in the layout its
   options use (the people in the analysis as a flow, the nutrients as a strip, what the models read
   as a lineage, the plan as it stands as angles), with one line saying what it is. Nolan wants most
   of the screen for the tapestry, and empty space wastes it. A view at rest obeys rule 6: an
   outcome whose gate is closed is drawn by its reading, with one line saying when its values open
   ("The outcome's values open after Who's in").
9. **New view kinds the closed vocabulary lacks:** table, forest, curve, calibration, decision
   curve, specification curve, overlap, embedding, matrix and page. They serve the exhibits after
   Fit, and some serve views before it too: overlap and weights in Models, before any estimate, and
   embeddings in First look. Each is designed once, as a recorded design decision with its purpose
   entry (SIZING P0.3b), and joins the vocabulary before any view draws it. No exhibit or view
   invents its own.

## 6 · One kit, and fairness

The four structures (the Q&A card, the paper, the quest log and the map) were built from one kit,
on one scenario, and a synchronization audit confirmed that only the structure differed before
Nolan saw them (2026-10-05). On 2026-10-06 he picked the quest log, to be designed further, and
this file now describes it.

The rule that made the comparison fair still holds. Every screen of the quest log, and any
alternative tried later, is built from the same kit: the same tokens, card, options, tapestry,
layouts, manuscript, copy and data. A new part enters the kit, never a single screen, so that
comparing two structures compares structures only. Any alternative tried later passes the same
synchronization audit before Nolan sees it. The four prototypes stay in the dev app as lab
routes (`/lab/calm`, and the static `calm.html`) for reference, beside the three earlier
living-methods prototypes (`/lab/methods`); a production build drops them all.

## 7 · The two flowcharts, and Fit

**Full-tapestry flowcharts** (Nolan, 2026-10-06). Two flowcharts close the stages before the fit.
Each is a confirmation of the whole stage at a glance: it takes the whole tapestry, can be saved as
a figure, and is placed like any exhibit (§8). A flowchart is not a line; it is its stage's last
screen, after the Confirm sweep.
- **The participant flow closes Who's in.** It counts the lens's own unit: people for dietary,
  clinical and survey data; for omics, two lanes, samples and features; for a trial, the CONSORT
  flow by arm. While Models answers can still change who is kept (under Estimate, complete cases
  are counted on the columns the adjustment set keeps), its counts are labeled provisional; they
  become final on the analysis flowchart (disagreement 7).
- **The analysis flowchart closes Models.** It is the user's own pipeline: what will be fit, on
  which rows, in what order, drawn as the flow of rows joined to the routing of columns. Under
  Estimate and Describe it comes after the open-noticings gate: every open noticing that feeds the
  plan is decided or dismissed first (Nolan, 2026-10-06). Under Predict that gate stands before the
  held-out rows open, in Results.

**Fit is pressed on the analysis flowchart** (Nolan, 2026-10-06: it "is where the Fit button
lives"; 2026-10-08: "Fit is pressed on the analysis flowchart, which shows progress and Cancel").
The flowchart fills the tapestry, and Fit sits at its end, where the rows and the columns meet the
models, with its estimate: "Fit · about 30 min". It is the screen's one primary action, and the
card's foot holds no button on this screen: the one exception to §1's fixed place. Pressing it is a
job command, not a decision (`RECIPES_AND_TUNING.md` §4.4).
- Under Estimate and Describe, pressing Fit locks the track's plan (the system's `lock_plan`
  record). Results opens with one line, "Plan fixed at 14:02", and the plan's SHA-256 fingerprint
  as its quiet label.
- Under Predict nothing locks. Results opens on the cross-validated comparison.
- A short fit may already be computed when Fit is pressed, and Results simply opens. Computing
  stays live; what waits is what is shown.

**A long fit runs as a server job** (ruled 2026-10-08).
- A fit expected to take over about 2 minutes does not start until Fit is pressed (ruled
  2026-10-06).
- It runs on the server as a job that survives closing the tab, and notifies the user when it is
  done.
- The analysis flowchart shows its progress and a Cancel where Fit was. The quest line reads
  "Results · fitting, about N min" (§3).
- The user can keep working on anything that does not depend on the fit.
- **Cancel before any estimate is served withdraws the lock.** The lock says the plan was declared
  before any estimate was shown, and none was. The record keeps the withdrawn lock with its time
  and fingerprint, so the history stays visible, and the next press of Fit records a new lock.
  This needs one engine change: `plan_lock.py` records the lock "once and never undone". Once any
  estimate has been served, the lock stands, and Cancel stops only the remaining work.
- **A change to the plan while the fit runs** cancels the job, after a consequence line, and asks
  for Fit again. Made before any estimate is served, it is an ordinary change, never a secondary.

## 8 · Results as exhibits

After Fit, the quest log turns from deciding to reading and placing. Results' lines are its
exhibits.
- **An exhibit** is a figure or table with its caption, the finding, and its interpretation, drawn
  on the tapestry. Results are exhibits, not text.
- **The three levels hold.** The line names the exhibit, with its placement at its right edge.
  Pointing lights it on the tapestry. A click opens it: the tapestry shows the evidence, and the
  card offers its wording and its placement.
- **Wording:** a drafted wording, one of them quietly labeled "Recommended", or "Write my own".
  Each draft is written at its claim strength: descriptive; association; estimated effect with its
  assumptions named (Estimate only, with the declared effect and the unmeasured-confounding exhibit
  beside it); causal (trials only); prediction performance; "describes the model" (explanations);
  labeled secondary; or inconclusive null.
- **Write my own** passes the same manuscript gate as a draft: every number in it traces to the
  record. Wording stronger than the exhibit's claim strength, such as an association worded as an
  effect or an explanation worded as what changing an intake would do, raises a noticing on the
  exhibit: the user rewords it, or keeps it and the keeping is recorded. It is never allowed
  silently. This is how the floor's wording rules (below) reach text the app checks but does not
  write.
- **Placement:** Results, Discussion or the Supplement, or left out. Each exhibit arrives with a
  default place (crosswalk, "The exhibits"). An exhibit left out keeps its analysis listed (the
  floor, below).
- **The floor** (the orchestrator's methods floor of 2026-10-06, as the crosswalk extends it), so
  that curating never becomes selective reporting:
  - under Estimate, the locked primary always stays in Results; under Predict, so does the
    held-out score, or the declared cross-validated result. Their placement is fixed;
  - every analysis that was run is listed in the exported supplement, whatever its exhibit's
    placement, with those left out of the paper under "Analyses left out"
    (`other:results_inventory`, `export:supplement-document`). The list cannot be turned off;
  - a declared family of exposures is never trimmed by p-value;
  - explanations are never worded as effects;
  - causal wording is offered only for trials.
- **Under Predict,** the final model, the threshold and the recalibration are fixed, Results'
  Confirm sweep is cleared (§3) and the open-noticings gate is cleared, before the held-out rows
  open, once.
- **Noticings born after the fit** label the exhibit they concern, or add a disclosure or a
  sensitivity analysis. A new analysis they ask for after the lock is a secondary.

**After results, never overwrite silently** (ruled 2026-10-08). Going back to an earlier stage after
results is allowed, and a consequence card comes first, in `RECIPES_AND_TUNING.md` §6.3's words.
Then:
- under Predict, before the held-out rows open, a change to a model's recipe or tuning keeps the
  earlier version in the comparison, and a change to a shared step keeps the earlier scores as a
  read-only row;
- under Predict, once the held-out rows are open, a change runs as a secondary analysis
  (`RECIPES_AND_TUNING.md` §6.3, "opened"). Drawing them again is a reseal, recorded as after their
  scores were seen, and the first opening stays the reported result;
- under Estimate and Describe, a change after the lock runs as a secondary analysis beside the
  locked primary, which stays.

**"Revised after first results" appears in two places only:** the comparison ("Boosted trees ·
first version") and the methods text. It is never on a stage, the quest line or the bar, and it is
worded neutrally, never "changed after seeing scores". The bar may still drop a stage back and say
why, in plain words about the change.

## 9 · Write-up: one document, two views

The manuscript rail and Write-up are one document in two views (ruled 2026-10-08). The rail is the
live draft in every stage, for glancing and changing a phrase. Write-up is the same draft at full
width, across the card column and the tapestry, with the finishing tools, in this order:
1. **What the export still waits for,** each item with a way forward: its exit opens the stage that
   owns it, and that stage drops back with the reason.
2. **The merge of tracks,** when the paper has several goals.
3. **The placements,** as reviewed.
4. **The Discussion drafts,** with limitations drafted from what was noticed: keep, edit or drop
   each.
5. **Author-only text** (the title, objectives, setting, ethics, funding): it never blocks the
   export, and exports as `\todo` until written.
6. **The export:** an Overleaf-ready LaTeX project and a Word document, rendered from one
   manuscript model. Every number traces to the record, every `\ref` resolves, and every DOI is
   checked against Crossref. The checklist as it stands now (STROBE-nut, TRIPOD+AI or CONSORT)
   sits under For the record.

Only these fill Write-up's segment; author-only text never counts. Export sits at the foot of the
left column, the fixed place of §1.

## 10 · The display-order rule

The quest log's order (the seven stages, and the lines inside them) is the order a person thinks
in. The engine's order is the order its answers depend on. They disagree in 20 places (crosswalk,
"Where the engine order and the stage order disagree"). **The screen order may differ from the
engine order only if** (the orchestrator, 2026-10-08):
1. nothing shown depends on an unanswered decision without saying so;
2. any answer the engine fills in is recorded and visible, in a Confirm sweep or For the record;
3. no view touches rows or the outcome before its gate opens.

Never "quietly". How each of the crosswalk's 20 fixes meets it, in the crosswalk's numbering:

| # | Where the orders disagree | What the quest log does | Rule |
|---|---|---|---|
| 1 | The roles answer waits behind Who's in | Your data asks no roles question: the engine records the answer once the readings settle, and the ledger's Confirm sweep shows it | 2 |
| 2 | Readings are asked where they are used | Your data's ledger asks only the readings that change a number and sweeps the rest; a column that enters later is asked where it enters | 1, 2 |
| 3 | The outcome's reading depends on Who's in | Which value counts is asked on the outcome card when rows repeat; a Who's in answer that changes the outcome's kind reopens Your question with the reason | 1 |
| 4 | First look comes before the draw, the engine explores after it | Outcome-free looks are computed from the oriented table; the outcome's views wait for their gates (§5, rule 6) | 3 |
| 5 | The split is two decisions in one | The draw in Who's in, the validation scheme in Models' Confirm sweep; under Estimate the split is For the record | 2 |
| 6 | "Confirm last" collides with the draw | Under Predict, Who's in's sweep runs before the draw (§3) | 2, 3 |
| 7 | Who's in depends on Models answers | The participant flowchart is provisional until the analysis flowchart; a Models answer that changes who is kept reopens Who's in with the reason | 1 |
| 8 | The clusters question decides a model term | Who's in asks only whether people are grouped, and by what; how the model handles it is a Models Confirm | 2 |
| 9 | Survey has no question under Predict | The survey question is asked under every goal; under Predict it asks whose performance the scores describe | 1 |
| 10 | The design has no slot | Your question asks it before the goal, set to "observed as they were" until the user says otherwise | 2 |
| 11 | Ten Models decisions have no Router key | The stage registry lists each in Models where it applies; one that waits on an earlier answer says so | 1 |
| 12 | Estimates are served, and the plan locks, before Fit | No estimate is served before the lock, and pressing Fit locks (§7) | 3 |
| 13 | Substitution is asked after the lock | The pair is declared in Models before Fit, with an outcome-free preview; Results draws the curve | 3 |
| 14 | Displays sanctioned after the fit count as plan changes | Explanations, a diagnostic's response and updating are recorded as companion displays, not as changes to the plan | 2 |
| 15 | A change after the lock overwrites the primary | The primary stays on the locked plan, and the change runs as a labeled secondary (§8) | never quietly |
| 16 | Intended use straddles three stages | Intended use is Predict's shape in Your question; under Predict, Results fixes the recalibration and runs its sweep (the threshold range, the calibration horizon) before the opening (§3) | 2, 3 |
| 17 | Reference rows and drift correction sit in two stages | Decided in Your data, before the outcome's kind is read, and drawn as the participant flow's first step | 1 |
| 18 | The Router's order inside Models | The quest log adopts the Router's order | none at stake: the orders agree |
| 19 | The methods text follows the guideline | The manuscript keeps the guideline's order; each sentence's "change" link opens its card through the stage registry | none at stake: the rail is a view |
| 20 | The Router answers in order | "Waiting for" on a line whose earlier answers are missing; First look's "Decide now" is the one exception (§3) | 1 |

Every stage's design checks its order against these three rules first.

## Superseded

Replaced by the rewrite of 2026-10-08:
- **The Q&A card as the baseline** (§3 and §6, written 2026-10-05). Nolan picked the quest log on
  2026-10-06.
- **The chain** (one top line, "Data · Participants · Columns · Exposure · Confounders · Energy ·
  Model …", ordered by the purpose's reporting guideline), replaced by the quest line of seven
  fixed stages and the progress bar. The guideline now orders only the manuscript
  (`export/methods.py:SECTION_OF`). Nolan had already ruled out the chain's label "Exposure" on
  2026-10-07, as jargon.
- **The card's head "Participants · step 2 of 3".** The card names its stage; progress lives in the
  bar.
- **The objective list by guideline section** (BLUEPRINT §11.4, item 5, and the calm-quest
  prototype's objective line, which lists the methods section's sections with their counts),
  replaced by the seven stages. The prototype stays in the lab as built.
- **The manuscript's "asked slots" and "stated phrases",** now "a question not answered yet" and
  "any phrase", since the tiers' names never reach the screen.
- **§1's "fixed footer on narrow screens",** dropped with the desktop-only ruling of 2026-10-05.
- **§5 rule 6, "Under inference, no outcome-model estimate appears on the canvas before the plan is
  locked",** widened on 2026-10-08 to every goal, to the outcome's own views and to the models'
  ranking, and tied to Fit.
- **"Canvas" in this file's prose,** now "the tapestry" (Nolan's word, 2026-10-06). The code keeps
  `Canvas`.
- **§2's "the record, the map and provenance",** now "the record, the flowcharts and provenance".
  The map was the chain, the provenance figure in its calmest form; the two flowcharts now show the
  analysis at a glance.

## Open for Nolan

- **The outcome's column in Your data, before it is named** (§5, rule 6). Your data settles every
  column's reading on its values, so it draws the eventual outcome as a column before anyone names
  it, on every row, including the rows a Predict track will later hold out. The design records that,
  and hides the outcome's values from every view once it is named. The stricter answer is that Your
  data draws no column's spread at all, only its reading, which makes units harder to confirm.
  Recommended: the design as written. A look at one column shows no association and no score, so
  it cannot steer a choice toward a result, and the record says it happened.

## Review notes not taken

From the review of 2026-10-08 (the merge `d9908297` and the first rewrite `d9747d1a`):
- **Splitting Results' sweep under Predict,** with the explanation defaults left last. Its one
  explanation default does not depend on the opening, so one sweep before the opening keeps one
  Confirm objective per stage (§3).
- **Updating the documents that still cite the old FOUNDATION** (`HANDOFF.md`'s note that §3 and
  §6 must be amended, two notes in `crosswalk/crosswalk.json`, SIZING P0.3a, and BLUEPRINT §11 and
  §11.4's objective list by guideline section), and the crosswalk's Focus on the outcome's
  distribution in Your question (narrowed in §5, rule 6). The review leaves them to the
  orchestrator at merge.

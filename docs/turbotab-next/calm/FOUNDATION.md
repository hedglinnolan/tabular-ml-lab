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
and 2026-10-08) and the crosswalk (`crosswalk/CROSSWALK.md`). What the rewrite replaced is listed
at the end, under "Superseded".

## 1 · The test

**Within five seconds, a first-time user knows what to click.** Every screen and every review is
judged by this first. One primary action per screen, always in the same place: the foot of the card
column. Continue, "Confirm all 6", Fit and Export all sit there. The choice color is reserved for
the choice.

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

- **The quest line (top, one line).** The seven stages by name, and nothing else. The open stage is
  in the text color, the others quieter. Clicking a stage opens it on the card; pointing at it
  shows its questions (below).
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
2. **Your question:** the outcome, then the goal (Describe, Estimate an effect, Predict) worded
   around it, and its shape, filtered by domain. "No single outcome" starts Describe (ruled
   2026-10-08). A second goal adds a track.
3. **First look:** shaped by the goal, a guide plus a gallery walked on the first visit. It shows
   what it notices and decides almost nothing itself: each noticing is decided in the stage whose
   answer it changes, and returns there as a line.
4. **Who's in:** who is kept, and what each blank means; under Predict, the drawing of the held-out
   rows. It closes on the participant flowchart (§7).
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
  (crosswalk disagreement 20).
- **A stage not reached yet** lists the questions it expects so far, without a count, since its
  lines depend on answers still to come.

**The progress bar.** Seven segments at the top right, one per stage, so that a long analysis never
feels endless (Nolan: "so users don't get discouraged").
- A segment fills with its stage's Decide and Confirm lines. For the record never counts.
- **A stage not reached yet shows empty,** never "0 of N".
- **A stage reopened by an earlier change drops back and says why,** in one plain line beside the
  bar and atop the stage's card, naming the answer that changed: "Your data reopened: the join
  added 12 columns, so their roles are read again."
- While a long fit runs, the Results segment reads "fitting, about N min" (§7).
- The bar is drawn in the app's own ink, never in the choice color (§4). Pointing at a segment opens
  the same panel as pointing at its stage.

**Decide · Confirm · For the record.** Every line carries one of three labels (Nolan, 2026-10-06).
They are the card's section heads, in plain type, never chips or colors.
- **Decide:** a question the user answers, or an open noticing they decide or dismiss. It counts
  toward progress.
- **Confirm:** a default set for the user, listed only when another choice would change a number on
  this table. It counts toward progress. Nolan: *"They ultimately need to own their results, but we
  can make their lives easier."*
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
clears it: "Confirm all 6". One departure, within the display-order rule (§10): under Predict, Who's
in runs its sweep just before the held-out rows are drawn, so the drawing is the stage's last Decide
(disagreement 6). Write-up's sweep is usually absent, since its one number-changing default is the
small-cell threshold.

**Three levels of disclosure.** Every line opens in three steps (Nolan, 2026-10-06, for the Confirm
sweep; the quest log uses them for every line):
1. **The line:** plain words, with its answer or default and, for a Confirm, a short reason.
2. **Pointing or keyboard focus:** the line elaborates "ever so slightly", by one clause on what it
   does to this table, and the tapestry lights what it touches in the choice color, as pointing at
   an option does. The technical name rides on the line's top edge here and only here.
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
   - The outcome's own views open at their gates (ruled 2026-10-08). Under Estimate, the outcome
     alone opens after Who's in, and the outcome beside another column waits for the lock. Under
     Predict, both open after the seal, on the training rows. Every such view is recorded.
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
   of the screen for the tapestry, and empty space wastes it.
9. **After Fit, exhibits need view kinds the closed vocabulary lacks:** table, forest, curve,
   calibration, decision curve, specification curve, overlap, embedding, matrix and page. Each is
   designed once, as a recorded design decision with its purpose entry (SIZING P0.3b), and joins
   the vocabulary before any exhibit draws it. No exhibit invents its own.

## 6 · One kit, and fairness

The four structures (the Q&A card, the paper, the quest log and the map) were built from one kit,
on one scenario, and a synchronization audit confirmed that only the structure differed before
Nolan saw them (2026-10-05). On 2026-10-06 he picked the quest log, to be designed further, and
this file now describes it.

The rule that made the comparison fair still holds. Every screen of the quest log, and any
alternative tried later, is built from the same kit: the same tokens, card, options, tapestry,
layouts, manuscript, copy and data. A new part enters the kit, never a single screen, so that
comparing two structures compares structures only. The four prototypes stay in the dev app as lab
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

**Fit is pressed on the analysis flowchart.** The flowchart fills the tapestry, and Fit sits where
every primary action sits, at the foot of the card column, with its estimate: "Fit · about 30
min". Pressing it is a job command, not a decision (`RECIPES_AND_TUNING.md` §4.4).
- Under Estimate and Describe, pressing Fit locks the track's plan (the system's `lock_plan`
  record). Results opens with one line: the lock's time and the plan's SHA-256.
- Under Predict nothing locks. Results opens on the cross-validated comparison.
- A short fit may already be computed when Fit is pressed, and Results simply opens. Computing
  stays live; what waits is what is shown.

**A long fit runs as a server job** (ruled 2026-10-08).
- A fit expected to take over about 2 minutes does not start until Fit is pressed (ruled
  2026-10-06).
- It runs on the server as a job that survives closing the tab, and notifies the user when it is
  done.
- The analysis flowchart shows its progress and a Cancel. The quest line reads "Results · fitting,
  about N min".
- The user can keep working on anything that does not depend on the fit.

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
- **Placement:** Results, Discussion or the Supplement, or left out. Each exhibit arrives with a
  default place (crosswalk, "The exhibits"). An exhibit left out is still listed, and every
  analysis that was run stays in the record.
- **The floor** (the orchestrator's methods floor of 2026-10-06, as the crosswalk extends it), so
  that curating never becomes selective reporting:
  - under Estimate, the locked primary always stays in Results; under Predict, so does the
    held-out score, or the declared cross-validated result. Their placement is fixed;
  - a declared family of exposures is never trimmed by p-value;
  - explanations are never worded as effects;
  - causal wording is offered only for trials.
- **Under Predict,** the final model, the threshold and the recalibration are fixed, and the
  open-noticings gate is cleared, before the held-out rows open, once.
- **Noticings born after the fit** label the exhibit they concern, or add a disclosure or a
  sensitivity analysis. A new analysis they ask for after the lock is a secondary.

**After results, never overwrite silently** (ruled 2026-10-08). Going back to an earlier stage after
results is allowed, and a consequence card comes first, in `RECIPES_AND_TUNING.md` §6.3's words.
Then:
- under Predict, a change to a model's recipe or tuning keeps the earlier version in the
  comparison, and a change to a shared step keeps the earlier scores as a read-only row;
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
left column, where every primary action sits.

## 10 · The display-order rule

The quest log's order (the seven stages, and the lines inside them) is the order a person thinks
in. The engine's order is the order its answers depend on. They disagree in 20 places (crosswalk,
"Where the engine order and the stage order disagree"). **The screen order may differ from the
engine order only if** (the orchestrator, 2026-10-08):
1. nothing shown depends on an unanswered decision without saying so;
2. any answer the engine fills in is recorded and visible, in a Confirm sweep or For the record;
3. no view touches rows or the outcome before its gate opens.

Never "quietly". How the crosswalk's fixes meet it, for example:
- Your data shows no roles question: the engine records the roles answer itself once the readings
  settle, and the ledger and its Confirm sweep show it (rule 2; disagreement 1).
- A line that waits on an earlier answer reads "Waiting for: [the question]" (rule 1; disagreement
  20).
- The participant flowchart is labeled provisional until the analysis flowchart (rule 1;
  disagreement 7).
- First look's outcome-free looks are computed from the oriented table, before the seal, and the
  outcome's views wait for their gates (rule 3; disagreement 4, question 1).
- Under Predict, Who's in's Confirm sweep comes before the drawing, which reads its answers (rules
  2 and 3; disagreement 6).

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

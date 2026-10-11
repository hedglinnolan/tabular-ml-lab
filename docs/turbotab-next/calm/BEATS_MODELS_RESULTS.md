# Beats: Models and Results

Written 2026-10-10 by the design owner, before any screen was built (FOUNDATION §0, ruling 6), and
revised twice the same day: on the critique of the first build (Round 1) and on the third critique,
of the revised build (Round 2). It is the contract the build of Direction A ("the question, drawn")
is held to: what the person understands and feels at each step from entering Models to "Put this in
my paper", what the card says, what the tapestry draws, what the manuscript gains, and what the
engine must serve. The brief is FOUNDATION §0; the critique that led to it is
`QUEST_LOG_CRITIQUE_2026-10-10.md`. The build is the lab page `/lab/quest-meaning` (static:
`npm run build:quest-meaning`).

**The journey.** NHANES, 21,849 adults, nine survey cycles (2001–2002 to 2017–2018). The outcome is
fasting glucose; the goal is Estimate an effect; nobody is excluded and nothing is missing. In
Models the person asks for total sugars in place of other carbohydrate at the same total calories.

**Where the numbers come from.** Every number below is the engine's, read from
`turbotab/frontend/src/explore/quest-log/fixture.json`, or computed in `capture/capture.py` and
`capture/build.py` by the engine's own functions. The section `meaning` holds what the beats draw;
its paths are given as `meaning.…`. The capture replays the journey through the real server, in
process, and replays it again with one answer changed for each refit (`capture.py --variants all`,
about three and a half minutes in all). Each number the engine does not serve yet is marked
**design-time**, with the engine requirement it stands for (§10).

**Names.** The project imported no codebook, so its columns have raw names such as `sugar`, `carb`
and `kcal`. The cards use the NHANES variable labels in plain words instead (`meaning.names`):
total sugars, carbohydrate, total calories, and so on. Those plain names are design-time
(requirement E13). So are the plain names of the earlier stages' lines in the quest line's hover
panel (`meaning.stage_lines[].plain`), each with its numbers read from the engine's own line.

**The two journeys in the fixture.** The *replayed* journey is the 2026-10-05 journey as recorded.
The *acted* journey is the same journey with the triage's routes taken before Fit (beat M6). The
two journeys' primary estimates agree to 13 significant digits. The acted journey's plan has its
own fingerprint, `ed8449eb5756`. The beats follow the acted journey.

## Round 2: what the third critique changed, and why

The third critique (2026-10-10, "Third look") found Models now hands the person's question back
(In place of what? and the model ladder were "the best work in the project"), but Results gave the
parts of the finding, not the finding, and the sequence had grown dense again. Round 2:

1. **R1 says the finding.** The locked primary stays the headline; "how big" against the standard
   deviation of a skewed outcome (skew 4.6) is gone. One plain reading follows: it is not a steady
   slope, and it is mostly not about the typical person; glucose is highest among people eating the
   least sugar, and two thirds of the difference comes from values in the diabetic range (18% of
   that fifth against 7% of the highest); that is the pattern people would leave if they cut sugar
   after a diagnosis, which the next screen checks. The tapestry draws glucose's own spread split
   at 126 mg/dL and the share past it by fifth of sugar. "Two thirds" is an exact split of the
   primary at the cut-off (§6), not a reading of the picture.
2. **The checks are sealed before the lock.** What Results will check (the other shape, the
   diabetic range, the split by a diagnosis on record, the implausible days, a hidden cause, each
   choice undone) is listed on the plan Fit seals and described in the Methods, so the paper
   reports them as planned secondary analyses. The only analysis decided after the lock is "Separate
   it further", labeled as such. The diagnosis noticed on Who is compared gets its own "Check it in
   your plan", at the same age and gender (4.9 points, not 45% against 25%).
3. **Reverse causation is read on the shape as well as the line, and said no stronger than its
   test.** On the straight line the association sits with people who have a diagnosis on record;
   on the shape, glucose is highest in the lowest fifth in both groups. The verdict is "In part",
   never "points to it", and the Discussion draft no longer says "near zero".
4. **Fewer screens, each saying one thing once.** Models goes from ten screens to seven (How
   calories are handled and A second swap fold into In place of what?, What you expect moves to
   the plan, the last thing before Fit) and Results from four to three (which choices mattered
   becomes the eighth question a reviewer will ask). After In place of what?, the comparison rides
   as a small emblem instead of a spine that builds into an accumulation. Every screen fits at
   1280 × 800.
5. **The paper.** Results is two paragraphs of at most 90 words: the primary with its declared
   checks and the second comparison, then the planned secondary analyses with the difference
   between the groups. The expectation and the reading as reverse causation are Discussion drafts.
   The Methods describe every analysis Results reports, carry no engine prose, and give the
   medicines one role: a marker of a diagnosis, never a covariate. The second comparison is an
   exact contrast of the primary model, not a bootstrap band.

§13 answers the critique point by point, with what was not followed and why.

## What the first critique changed (Round 1)

The first build (`e603a756`) was judged "much closer" but not yet magic: the meaning came from the
wrong answer (the partner was never asked), Results read the sign rather than the finding, and
nothing built up. Round 1 (`603994b2`) asked the partner, drew who is compared by what could explain
the link with a diagnosis noticed before any outcome is read, labeled each model by the comparison
it makes, gave Results the finding's size and shape and six standing reviewer questions, measured
every choice on one yardstick, and fixed the methods problems in §12's table.

## 1 · The arc

| | Beat | One decision | What the person feels |
|---|---|---|---|
| M1 | What you study | which nutrient (total sugars) | at home: this is my question |
| M2 | In place of what? | the partner (other carbohydrate); what it settles is said on the same card; also report (protein) | "so that's what my number will mean" |
| M3 | Who is compared | who is held fixed (confirmed, two edits); the diagnosis checked in the plan | careful, and warned |
| M4 | The shape | straight line or curve (straight); the model rides along | informed, responsible |
| M5 | Set for you | Confirm all 3 | respected, quick |
| M6 | What your data raised | the implausible days; two done for you; two noted | caught, and protected |
| M7 | Your plan | what you expect (higher), then Fit | ownership; a little suspense |
| R1 | What you found | none (reading) | surprise, then clarity: not steady, and in the tail |
| R2 | What a reviewer will ask | none (reading) | sobered, but armed |
| R3 | Your sentence | "Put this in my paper" | proud: it is defensible |

**The arc builds, then rests.** M1 draws the day; M2 draws the compared day, frames it with what is
held fixed once the partner is chosen, and sets the second comparison beside it. From M3 on, the
comparison rides as a small emblem at the tapestry's head, so each beat's own picture has the
canvas. M5 draws each model's own compared day. M7 shows the comparison with what you expect at the
head of the flowchart. R1 turns the same two bars into two groups of people with their difference.
The manuscript rail shows, at the card's foot, what each confirmation has just added to the paper.

**"What could fool you" is said three times, where each belongs:** in M3 as who is compared (with
the diagnosis noticed and checked in the plan, before any outcome is read), in M6 as the data's own
warnings at the gate, and in R2 as the questions a reviewer will ask, after the lock.

## 2 · How a beat is written

Each beat gives:

- what the person understands after it, and what they should feel;
- the one decision, or none;
- the card in final copy: the heading, the return sentence, the options, the quiet technical name,
  the button, and what sits behind "why?" and behind the alternatives link;
- the tapestry: which view kind, what it draws and from which data, what pointing at each option
  changes, and the caption;
- the sentence the manuscript gains, in the methods register;
- the engine: what it serves today (module and function) and what it must newly serve.

Card copy is final. The quiet technical name rides on an option's top edge while it is pointed at
(FOUNDATION §2), on one line (its full text is the element's title), never as a second label. The
card's kicker reads "Models · [the line's plain name]". Progress lives in the bar, never as "step 2
of 3" on the card. The card says; the tapestry draws; never both. A fact is said once per screen.

**Type.** One family, five sizes: 13 px (notes, quiet names, captions), 15 px (secondary lines), 17
px (body, and the paper's paragraphs), 19 px (the tapestry's lead) and 26 px (the card's heading, or
the finding, one weight).

**Budgets** (checks, not targets; FOUNDATION §0 ruling 5): about 120 words on a card, about 250 on
a screen, and nothing below the fold at 1280 × 800. Round 2's measures are in §13.

## 3 · Models

### M1 · What you study

**Understands:** "My question is about total sugars: 116 g on an average day here, 22% of its
calories." **Feels:** at home with the data.

**Decision:** what the person thinks affects fasting glucose. They choose total sugars. The engine
records this choice with the next one, as one `set_estimand`.

**Card**
- Heading: "What do you think affects fasting glucose?"
- Return: "Seven of your columns are nutrients that carry calories; pick the one your question is
  about."
- Options: total sugars · carbohydrate · protein · fat · saturated fat · monounsaturated fat ·
  polyunsaturated fat · "All seven, each in turn". Quiet names: "the exposure"; on "All seven",
  "an exposure family, tested seven times".
- Button: Continue. Alternatives link: "Something else in your table".

**Tapestry:** comparison view, the day as recorded.
- One calorie bar of the average day: total sugars 116 g, other carbohydrate 142 g, protein 81 g,
  fat 81 g, each as wide as its calories, and a thin last slice for alcohol and the rest (38 kcal:
  the recorded mean, 2,120 kcal, less the four sources' 2,082). Fat's three kinds sit under its
  slice. Pointing at an option lights its slice.
- Data: `meaning.day.days.recorded`, `meaning.day.fat_parts_grams`, `meaning.day.rest_of_energy`.
- Caption: the means of the 21,849 analyzed rows; "Until you press Fit, nothing here reads a
  glucose value." This is the one place Models says it; the rule itself is enforced in code.

**Manuscript:** nothing yet; the estimand sentence shows with its blank (FOUNDATION §3).

**Engine:** serves `estimand.estimand_card`, `methods.energy.energy_factor` and `omitted_energy`.
Must serve the comparison view's data (E3) and the codebook's names (E13).

### M2 · In place of what? (the comparison's home)

**Understands:** "My number will compare two kinds of day with the same calories: one with 25 g
more sugar and 25 g less of the other carbohydrate. So total calories and carbohydrate stay in the
model. Beside it, I can see sugar in place of protein, the same way round." **Feels:** "that is
what I mean." This is the core meaning moment, and Round 2 makes it the comparison's home: what the
partner settles (the energy model, formerly M4) and the second comparison (formerly M7) are said
on the same card.

**Decision:** what the extra sugar replaces (other carbohydrate), and what else to report (sugar in
place of protein).

**Card, before the partner is chosen** (`#/m2`)
- Heading: "More sugar, in place of what?"
- Return: "People here who eat more sugar eat more of everything (r = 0.67 with total calories), so
  your comparison needs a partner." First look's noticing returns as the reason
  (`diet-energy-carries-the-nutrient`), marked "Noticed in First look".
- Options, each a different question, each saying only what it gives up (the heading says "more
  sugar"):
  - "Other carbohydrate": "25 g less other carbohydrate; the rest of the day as it was." Quiet
    name: "substitution for other carbohydrate".
  - "Protein": "25 g less protein."
  - "Fat": "11 g less fat: the same 100 kcal."
  - "All other calories, as people here eat them": "Every other source gives up its share, as
    among people here who eat more sugar at the same calories." Quiet name: "substitution for total
    energy".
  - "Nothing: add it on top": "The day grows by 100 kcal." Quiet name: "addition (total energy not
    held)".
- Why?: each partner asks a different question (Tomova et al. 2022); "all other calories" is the
  mix people here eat (23% other carbohydrate, 17% protein, 33% fat and 27% alcohol and the rest);
  whichever is chosen, the number is sugar's whole effect as a difference in mean fasting glucose
  per 25 g (100 kcal). The Round 1 line "only the direct part: not available yet" is gone.
- Button: Continue (on a choice).

**Card, the partner chosen** (`#/m2-chosen`): the "because you said" form (FOUNDATION §0 ruling 2),
on the card where the answer was given.
- Kicker: "Because you said". Quote: "Other carbohydrate." (Models · In place of what?)
- Heading: "So total calories and carbohydrate stay in the model, beside protein and fat."
- Return: "Every gram of sugar the model adds is a gram of other carbohydrate it takes away, and the
  day's total does not move."
- Quiet name: "Known as the standard multivariate energy model (Willett, Howe & Kushi 1997)".
- "Also report, beside it": two boxes, in the same direction as the main comparison. "Sugar in place
  of protein" (ticked in this journey) and "Sugar in place of fat"; quiet name "a second
  comparison: a contrast in your model".
- Why?: the same-question methods (the residual method gives exactly this number,
  Frisch–Waugh–Lovell, by the gate's column-space test; without total calories it drifts;
  all-components is refused; the density models are a different measure), and "A second comparison
  asks one more question of the same model, and is labeled as one wherever it appears."
- Button: "Yes, that's my question". Left: "I meant a different comparison" (reopens the options).

**Tapestry:** comparison view.
- Before the choice: the recorded day and an empty second bar ("Choose a partner to draw it").
  Pointing at a partner draws its compared day: what the comparison adds is filled in the choice
  color at the gaining slice's end ("+25 g total sugars"); what it takes is outlined on the
  recorded day ("−25 g other carbohydrate"). "All other calories" draws the projection mix (other
  carbohydrate 136 g, protein 77 g, fat 77 g, alcohol and the rest 11 kcal) and the lead says the
  shares: 23% from other carbohydrate, 17% from protein, 33% from fat and 27% from alcohol and the
  rest (`capture.partners`, by `materiality._model_columns` and `inference_table`; outcome-blind).
- Once chosen: the two days framed by what is held fixed (a bracket over carbohydrate,
  "carbohydrate: unchanged"; one over the rest, "protein, fat and the rest: as they were"; the end
  line, "total calories: unchanged"), and each second comparison asked for as its own row, "Also
  reported: 25 g more sugar, 25 g less protein", with what it gives up drawn under its own bar so the
  recorded day carries only the main comparison's marks. Pointing at a box draws its row lit.
- Data: `meaning.day.days.{recorded,swap,protein,fat,all_other,add}`, `meaning.day.mix.all_other`.
- Caption: "From the columns' means and, for a mix, how each source moves with sugar at the same
  calories: an illustration of what your number compares, not an estimate."

**Manuscript** (three sentences arrive on "Yes, that's my question"): the estimand sentence ("The
analysis estimates the total effect of total sugars in place of the same energy from other
carbohydrate (total carbohydrate and total energy intake held fixed) on fasting glucose, as a
difference in mean fasting glucose per 25 g/day (100 kcal)."), the energy sentence ("Because the
comparison is a substitution with total carbohydrate held fixed, energy was adjusted by the
standard multivariate model (Willett, Howe & Kushi 1997)…"), and the second comparison: "A second
substitution, total sugars in place of protein at constant total energy intake, was estimated per
25 g/day (100 kcal) as a contrast of the primary model's coefficients, with its HC3 interval."

**Engine**
- Serves: the estimand card's contrasts; `estimand.substitution_words` (ME-04);
  `materiality.energy_noticing`; the substitution stage (a second swap, today as a bootstrap curve).
- Must serve (E17): the partner asked on the comparison; (E18) the partner's projection mix; (E1)
  the energy model settled by the partner, said on the partner's own card; (E7, amended) the
  second comparison as an exact contrast of the primary model, in the main comparison's direction;
  (E4) the comparison's step; (E2) the plain answer text.

### M3 · Who is compared

**Understands:** "My comparison already holds the day's calories and nutrients fixed. People who
eat the least sugar are older and more often women, and they more often have a diagnosis on record,
which can change what people eat; my plan will check it." **Feels:** careful, and warned. This is
the first "what could fool you", before any outcome is read.

**Decision:** who is held fixed, confirmed with two edits of the person's own ("changed by you":
survey cycle held fixed, the packs had no guess; the blood measures left out, the packs had proposed
the backup model); and the noticing, checked in the plan.

**Card**
- Heading: "Compared with people of the same age, gender and survey cycle" (a statement: the answer
  is already filled in).
- Lines, each with its group said quietly in front:
  - "Held fixed: age, gender and survey cycle · survey cycle changed by you. They shape both diet and
    glucose." Quiet name: "confounders".
  - "In a backup model: body size. Sugar may have changed it; Model 3 adds it." Quiet name: "a
    with-and-without analysis, for unknown timing".
  - "Left out: blood pressure, HDL cholesterol and triglycerides · changed by you. Sugar may change
    them, and they may change glucose." Quiet name: "mediators". The medicines are no longer on this
    line (Round 2): they mark a diagnosis, never a covariate.
  - The noticing, in the coach's amber: "Noticed here · The medicine questions mark a diagnosis on
    record: NHANES asks them only after one. People who eat the least sugar have one more often, 4.9
    points more at the same age and gender. A diagnosis can change what people eat." Its route:
    "Check it in your plan". Once taken (`#/m3-checked`): "In your plan: reported with and without
    one, and their difference. Undo". Quiet name: "reverse causation through a diagnosis (Shaper,
    Wannamethee & Walker 1988)".
- Why?: the disjunctive cause criterion (VanderWeele 2019); calories, carbohydrate, protein and fat
  are not on the list because they define the swap; "NHANES asks the medicine questions only of
  people told they have high blood pressure, or told to take medicine for their cholesterol (its
  Blood Pressure & Cholesterol questionnaire); a blank means the question was not asked. So the
  answers are not adjusted for: they mark the diagnosis."
- Button: Continue. Alternatives link: "Change who is held fixed", which opens each line's three
  questions. The blood measures' answers ("Could it cause sugar intake? no") no longer sit beside a
  noticing that says a diagnosis changes what people eat.

**Tapestry:** comparison view, who is compared, signed, with the comparison's emblem at the head.
- At rest: the fifth who eat the most sugar (165.5 g a day or more) against the fifth who eat the
  least (53.7 g or less), 4,371 people each, in two groups only: what is held fixed (age 0.59 left,
  51 against 41 years; men 0.41 right; survey cycle 0.32 left) and "A diagnosis on record" (answered
  a medicine question 25% against 45%, 0.42 left; on blood-pressure medicine 16% against 33%; on
  cholesterol medicine 10% against 21%), each marked by the coach's dot.
- Pointing at a line adds its own group (body size; the blood measures) and lights it.
- Pointing at the noticing lights the diagnosis rows and draws the share with a diagnosis by fifth
  of sugar intake (45%, 42%, 37%, 33%, 25%); the lead says the raw gap and then "Most of that
  19-point gap is age: at the same age and gender, it shrinks to 4.9 points."
- Caption: standardized differences before any adjustment; "Point at a line to add its group";
  calories, carbohydrate, protein and fat are not drawn because they define the swap (sugar itself
  is 72% of the carbohydrate gap).
- Data: `meaning.balance` (`materiality.smd`, Austin 2009, signed), `meaning.diagnosis`
  (`capture.diagnosis`: the engine's fifths; the adjusted difference by `inference_table` on the
  fit's own age and gender columns). Outcome-blind.

**Manuscript** (two sentences on Continue): the adjustment sentence (E14): "Total energy intake,
protein, carbohydrate and total, saturated, monounsaturated and polyunsaturated fat were held fixed
to define the substitution. By the disjunctive cause criterion (VanderWeele 2019), the analysis
adjusted for age, gender and survey cycle. It treated systolic and diastolic blood pressure, HDL
cholesterol and triglycerides as possible mediators and left them out. It added weight, height,
body mass index and waist circumference, of unknown timing, in a secondary model (Model 3); these
include values imputed in the source data, flagged there for 235, 248, 306 and 658 participants,
used as recorded." And the medicines' one role: "Answers to the blood-pressure and cholesterol
medicine questions, which NHANES asks only of participants told they had high blood pressure or
told to take medicine for their cholesterol, were not adjusted for; an answer to either marked a
diagnosis on record (7,946 participants)."

**Engine**
- Serves: `estimand.adjustment_card`, `estimand.guess_blocks`, `estimand.derive`; Who's in's
  reading "a blank may mean not asked" for the medicine columns; Your data's imputation flags.
- Must serve: the signed balance (E3); the comparison's own terms named as such (E17); the
  diagnosis marker from the skip pattern, its gradient before the lock and its check sealed in the
  plan (E19, amended); the composed sentences (E14).

### M4 · The shape (and the model, settled)

**Understands:** "A straight line gives one number per 25 g; a curve lets the step count differently
at low and high intakes. Whichever I choose, the other shape is checked in my plan." **Feels:**
informed, and responsible.

**Decision:** straight or curved. The person keeps the straight line; the engine's Recommended is
the curve (`meaning.cards.forms_sugar`).

**Card**
- Heading: "Should each extra 25 g count the same, whatever someone already eats?" (Round 2 drops the
  return line: the tapestry draws the intakes.)
- Options: "Let it bend" (Recommended; "Shows whether the first grams matter more than the last, and
  tests it."; quiet name "restricted cubic spline, 5 knots (Harrell 2015)"), "Keep it straight"
  ("One number per 25 g; if the truth bends, it averages over the bend."), "Fifths of intake, with a
  trend", and behind "Other shapes" the data-derived cut point (not sound, Altman & Royston 2006) and
  categories.
- The model, settled, in one line: "Settled: linear regression, for a difference in mean fasting
  glucose with each person once (set for you in Who's in)." Quiet name: "ordinary least squares,
  HC3 standard errors". Round 2 drops "Because you said" here: neither answer was said by the
  person (the measure rode as a quiet line; each person once was set for them).
- Why?: "…Whichever you choose, the other shape is set in your plan, and Results reports it beside
  your estimate."

**Tapestry:** sugar's one-day intakes (21,849 days; 5th percentile 24 g, median 99 g, 95th 262 g;
658 days above 300 g sit beyond the axis) with two equal 25 g steps (61 → 86 g and 150 → 175 g).
Pointing at the curve marks Harrell's knots; at fifths, the boundaries (54, 84, 116 and 166 g). The
comparison rides as the emblem at the head; the Round 1 spine, expectation axis and second row are
gone.

**Manuscript:** "Total sugars entered the models as a straight line, as did protein, carbohydrate,
total, saturated, monounsaturated and polyunsaturated fat, and age." and "The model was linear
regression with HC3 standard errors."

### M5 · Set for you

**Understands:** "Three conventions are set for me. Model 1 is the field's first model, and it asks
a different question from mine; the diagnosis split I asked for is already in." **Feels:**
respected, and quick.

**Decision:** "Confirm all 3".

**Card:** the three lines with their reasons. Model 1: "The field's usual first model. Its 25 g of
sugar replaces a mix of other calories (34% of them fat), so it asks a different question from
yours." "No other group.": "Also reported with and without a diagnosis on record, as you asked on
Who is compared. A group named after the estimates is labeled as suggested by the data." The main
model alone. Quiet names: "the model sequence", "effect modification", "double machine learning"
(Round 2 removes the internal document name from the first).

**Tapestry:** the ladder of comparisons. The average day once, labeled; then each declared model with
what it holds fixed, what it compares in one line, and its own compared day drawn bare (the marks
alone, under the labeled day):
- Unadjusted, holding nothing: "People eating 25 g more sugar, and with it 107 kcal more of
  everything else";
- Model 1, holding age, gender and total calories: "25 g more sugar in place of a mix of other
  calories"; pointing at it says the mix (23% from other carbohydrate, 17% from protein, 34% from
  fat and 27% from alcohol and the rest);
- Model 2, yours: "25 g more sugar in place of other carbohydrate";
- Model 3: "Your comparison, also at the same body size"; "Not a total effect: sugar may change body
  size."
- Pointing at the groups line: "Reported in each group: with a diagnosis on record (7,946) and
  without (13,903). Another group, for example women (11,195) and men (10,654), would be reported
  the same way."
- Caption: "…Nothing is estimated yet."

**Manuscript:** "The estimate is also reported in a declared sequence of models, each its own
comparison: unadjusted; Model 1, adjusted for age, gender and total energy intake, in which total
sugars replace the mix of other energy sources; Model 2, the primary; and Model 3, further adjusted
for weight, height, body mass index and waist circumference, possible mediators, which is not a
total effect." The engine's sweep sentence ("the primary model alone estimates the declared
effect") is engine prose and stays in the record.

**Engine:** must serve Model 1 as a Confirm line and the ladder before the fit (E6), each rung with
its comparison's mix (E18), and a modifier declared from a noticing's route (E19).

### M6 · What your data raised

**Understands:** "One thing to decide: the implausible days. Two mistakes in the file were fixed for
me and change no estimate; one day of recall becomes a limitation; the columns that move in step
change nothing." **Feels:** caught, and protected.

**Decision:** the implausible days (the one decision on the card, FOUNDATION §0 ruling 5). The zeros
and gender's coding change no estimate, so they are done for the person, recorded, with Undo.

**Card**
- "Decide before you fit": "1,614 days have implausible calories for one day, by Willett's cut-offs
  for each sex (1,419 by NHS/HPFS): they could move the estimate." (Round 2: the counts are the
  declared checks' own, not the engine's generic 500 to 5,000 kcal reading of 501 rows.) Options:
  "Keep everyone, and check without them" (Recommended: the triage's route), "Leave them out of your
  estimate", "Keep everyone, unchecked".
- "Done for you: no estimate moves": "152 misread SAS zeros (5.4 × 10⁻⁷⁹) read as 0. Undo";
  "Gender coded female 1 (11,195), male 0 (10,654). Undo".
- "Noted for your paper": "One day of recall per person: a limitation." "Calories, carbohydrate and
  fat move in step: a supplement line."
- Button: "Confirm all 5", enabled once the implausible days are decided ("Decide the implausible
  days first").

**Tapestry:** the view follows the pointed row: the participant flow with the checks' leavers and
calories by sex with their cut-offs (the implausible days and its options); the zeros by column
("152 values in 9 columns held the misread value instead of 0. 119 are diastolic pressures; blood
pressure is in no model, so how they are read moves nothing."); gender's levels; the one recall
day's lineage; what moves in step. At rest, the participant flow.

**Manuscript** (on Confirm all 5): "152 values holding 5.4 × 10⁻⁷⁹, a misread SAS transport zero,
were read as 0 (119 of them diastolic blood pressures, which enter no model)." "Gender was coded 1
for female and 0 for male." The declared checks (Banna et al. 2017). The total energy unit is a
reading and stays in the record (E14); the triage's sentence is engine prose and stays there too.

### M7 · Your plan: what you expect, then Fit

**Understands:** "This is my whole analysis, and what Results will check is sealed with it. Last, I
say which way I expect glucose to differ, before anything is estimated." **Feels:** ownership, and
a little suspense: the expectation is the last thing before the seal.

**Decision:** the expected direction (Round 2 moves it here from its own screen). In this journey,
"Higher". Design-time (E5).

**Card**
- Heading: "Last, before you fit: which way do you expect fasting glucose to differ?"
- Return: "Between people whose day had 25 g more sugar in place of other carbohydrate, at the same
  calories."
- Options: Higher · Lower · No different · "I'd rather not say" ("Results checks nothing against
  it.").
- Quiet: "Fit seals it with your plan and what Results will check, under one fingerprint, before any
  estimate is shown. Written after the estimate, it would not count."
- The card's foot holds no button (FOUNDATION §7).

**Tapestry:** the analysis flowchart.
- At its head, the comparison (the emblem, wide) beside "You expect": "Not said yet", then "Higher
  fasting glucose" once chosen, "among people of the same age, gender and survey cycle".
- Three boxes: People (People in your table 21,849 → fasting glucose recorded → nothing missing in
  the model) → What the model reads (what you study; part of your comparison; held fixed; backup
  model only) → The model, and what Fit produces (linear regression, HC3; the four models; "Also
  reports: sugar in place of protein").
- "Sealed with your plan: what Results will check" (E22): the other shape (a curve, and fifths of
  intake); the diabetic range (the part above 126 mg/dL); with and without a diagnosis on record
  (yours); without the implausible days, two ways (yours); what a hidden cause would need; each of
  your choices, undone.
- Fit at the end, disabled until the expectation is said or declined: "Say which way you expect
  glucose to differ, or that you'd rather not; then Fit." Everything fits above Fit at 1280 × 800.

**Manuscript:** on Fit, four sentences: the expectation, the planned secondary analyses, the hidden
cause and the choices, and the lock (§8).

## 4 · Results

### R1 · What you found

**Understands:** "People whose day had 25 g more sugar in place of other carbohydrate had 0.50 mg/dL
lower fasting glucose. It is not a steady slope, and mostly not the typical person: glucose is
highest among those eating the least sugar, and two thirds of the difference comes from values in
the diabetic range. That is what people who cut sugar after a diagnosis would leave behind."
**Feels:** surprise, then clarity.

**Card**
- Kicker: "Results · What you found · plan sealed at [the lock's time]".
- Finding (26 px): "At the same calories, people whose day had 25 g more sugar in place of other
  carbohydrate had 0.50 mg/dL lower fasting glucose." Then "95% CI 0.18 to 0.82 lower · 21,849
  adults".
- The reading (17 px), each clause by its check's verdict (§6): "It is not a steady slope, and it is
  mostly not about the typical person. Fasting glucose is highest among people eating the least sugar
  (under 54 g a day), and two thirds of the difference comes from values in the diabetic range: 18%
  of them are there, against 7% of those eating the most. That is the pattern people would leave if
  they cut sugar after a diagnosis; the next screen checks it."
- Quiet name: "Known as Model 2, the primary · HC3 95% interval · SHA-256 ed8449eb5756".
- Why?: per gram, per 25 g (the comparison's step), the model's terms, and the diabetic range's
  source (fasting glucose of 126 mg/dL or more, American Diabetes Association).
- Button: "Next: what a reviewer will ask".

**Tapestry:** two panels, nothing below the fold.
1. The two groups of people: "People whose day looked like this" and "people whose day had 25 g more
   sugar, 25 g less other carbohydrate", with the bracket "Fasting glucose, on average, 0.50 mg/dL
   lower (95% CI 0.18 to 0.82)". The difference belongs to the groups, never to an imagined day.
2. "Where it comes from: two thirds of the 0.50 mg/dL is in the diabetic range": a split bar (0.34
   from values in the diabetic range, 0.16 from the rest), glucose's own spread with the values at
   or above 126 mg/dL in ink ("126 mg/dL and above: 12% of people"), and the share in the diabetic
   range by fifth of sugar (18%, 14%, 10%, 9%, 7%), the lowest fifth in ink.
- Caption: "Between groups of people observed as they were: an association, not what would happen
  if anyone changed their diet. Glucose drawn once the plan was sealed; 333 values above 250 mg/dL
  sit beyond the axis."
- The curve moves to R2's shape question; the standard deviation is no longer a yardstick on screen.
- Data: `meaning.results.headline`, `meaning.results.tail` (`capture._tail_and_groups`, after the
  lock), `meaning.results.outcome.hist`.

**Manuscript:** the four sentences sealed by Fit arrive.

### R2 · What a reviewer will ask (and which choices mattered)

**Understands:** "Eight questions, each set in my plan. It runs against my expectation; it bends; it
lives mostly in the tail; reverse causation explains part of it but not all; a hidden cause cannot
be ruled out; one day of recall and the unweighted design cannot be checked here; only the shape
moved my estimate past its margin." **Feels:** sobered, but armed.

**Card:** the kicker "Results · Each set in your plan before Fit", the heading "What a reviewer will
ask", and eight lines, each a question and a one-line verdict:
1. "Did it go the way you expected?" "No: lower, not higher."
2. "Is a straight line fair?" "No: it bends at the lowest intakes."
3. "The typical person, or the tail?" "Mostly the tail: the diabetic range."
4. "Could it run the other way?" "In part: stronger with a diagnosis, yet there without one." A
   button, "Separate it further", opens one line in place: "Join NHANES's diabetes questionnaire in
   Your data, so a diabetes diagnosis is on record. Decided after the plan was sealed, it runs as an
   analysis labeled as such." This is the one analysis decided after the lock.
5. "Could a hidden cause explain it?" "Not ruled out: one tied to both about three times as strongly
   as survey cycle would erase it."
6. "Could one day of recall distort it?" "Cannot be checked: one day per person."
7. "Does it describe the US population?" "Not as analyzed: no survey weights."
8. "Did your choices move it?" "Only the shape moved it beyond its margin." (Round 1's R3, folded
   in.)

Quiet names: a prespecified direction, checked; functional form (Harrell 2015); the difference in
means split at a clinical cut-off (126 mg/dL); reverse causation through a diagnosis (Shaper,
Wannamethee & Walker 1988); robustness value 2.3% (Cinelli & Hazlett 2020); measurement error in
several intakes (Freedman et al. 2011); complex survey design, the design effect (Kish 1965); each
choice refit with its main alternative.

**Tapestry:** at rest, the one question that points somewhere: "Could it run the other way?" (its
line on the card lit to match). Lead: "In part: on your straight line it sits with people who have
a diagnosis on record; but on the shape, glucose is highest at the lowest intakes in both groups."
Two forests, signed, one under the other:
- your straight line, per 25 g: everyone −0.50 (−0.82 to −0.18); no diagnosis on record −0.16
  (−0.48 to 0.16), 13,903 people; a diagnosis on record −1.52 (−2.30 to −0.74), 7,946 people;
- the shape, the highest fifth against the lowest, mg/dL: everyone −8.2 (−10.6 to −5.8); no
  diagnosis −5.2 (−7.5 to −2.8); a diagnosis −13.5 (−19.0 to −7.9).
- Caption: "A planned check, not proof: a diagnosis can itself follow from diet, and the data hold
  no record of a diabetes diagnosis. Differences between the groups: −1.36 (−2.21 to −0.52) on the
  line, −8.3 (−14.3 to −2.2) in the highest fifth."

Pointing draws each other question's evidence: the expectation beside the interval; the curve
(sugar as a five-knot spline in the primary model, its pointwise band, each fifth against the
lowest with its interval, your straight line; F(3, 21,827) = 24.5, p = 8 × 10⁻¹⁶; "your straight line
stays the estimate"); the tail (yours −0.50 split into −0.34, −0.61 to −0.08, from values of 126 or
more and −0.16, −0.26 to −0.06, from values up to 126); the what-if forest, benchmarked on age,
gender and survey cycle only (survey cycle moves it half its margin; three times as strong erases
it); the recall lineage; the design effect of 2.44 at which the interval reaches zero; and the
choices forest, under the lead "Your sealed estimate stands; these are sensitivity analyses beside
it", its curve row "a different summary, not a corrected slope", and the implausible days' miss said
plainly: predicted before Fit to move it about a tenth of its margin, they moved it three quarters.
One sign convention: the leads say verdicts in words, the forests print signed numbers.

**Manuscript:** five Discussion drafts arrive: against the expectation; dietary change after a
diagnosis; a hidden cause; one day of recall; unweighted (§6).

### R3 · Your sentence, and "Put this in my paper"

**Understands:** "My paper says two paragraphs in Results: my plan's estimate with its declared
checks, then what the planned analyses found. The rest goes where it belongs." **Feels:** proud.

**Card**
- Heading: "How should your paper say it?"
- The Recommended wording, "An association, with what your checks found", as two paragraphs at 17
  px, each labeled quietly ("Results, paragraph 1 · Your plan's estimate"; "Results, paragraph 2 ·
  The planned secondary analyses"), each at most 90 words (a test holds them to it):
  > Each 25 g/day higher intake of total sugars in place of other carbohydrate, at constant total
  > energy intake, was associated with 0.50 mg/dL lower fasting glucose (95% CI 0.18 to 0.82;
  > n = 21,849; Table 2); in place of protein, with 0.84 mg/dL lower (95% CI 0.36 to 1.31).
  > Excluding participants with implausible energy intakes, as declared, it was 0.73 (95% CI 0.39 to
  > 1.06; Willett cut-offs) and 0.71 mg/dL lower (95% CI 0.38 to 1.03; NHS/HPFS cut-offs).

  > In planned secondary analyses, the association was not linear (p < 0.001): fasting glucose was
  > highest in the lowest fifth of intake, 4.9 to 8.2 mg/dL above the other fifths, and two thirds of
  > the difference per 25 g/day came from values of 126 mg/dL or more. Per 25 g/day it was 1.52 mg/dL
  > lower (95% CI 0.74 to 2.30) with a diagnosis on record and 0.16 lower (95% CI 0.48 lower to 0.16
  > higher) without (difference 1.36, 95% CI 0.52 to 2.21); the lowest fifth was highest in both
  > groups.
- Behind "Other wordings" (scrolled into view when opened): "An estimated effect, its assumptions
  named", not offered, with its computed reason ("its assumptions would include a straight line,
  which the planned check rejects (p = 8 × 10⁻¹⁶); no dietary change after a diagnosis, which the
  split by diagnosis is partly consistent with; and no hidden cause, which cannot be ruled out");
  and "Write my own".
- Button: "Put this in my paper".

**Tapestry:** Table 2 as it will print, with its "Compared with" row (nothing held · a mix of other
energy · other carbohydrate · other carbohydrate) and one footnote line; and where each part goes:
Results (the two paragraphs; Table 2, Model 3 among its columns), the Discussion's five drafts, the
Supplement's six analyses run.

**After "Put this in my paper"** (`#/r3-placed`): the rail opens over the card with the two
paragraphs under "Results · just placed". The primary, "Next: the exhibits", closes the rail and
opens the seven exhibits in place, each at its usual place: Table 2 in Results; the shape, the
diabetic range, the split by diagnosis, the checks without implausible days, what a hidden cause
would need (with a Discussion draft) and each choice refit, in the Supplement. "Confirm all 7"
places them; Write-up is the next design.

## 5 · The return grammar

When earlier answers settle a question, the person is not asked it again. They are shown what
their answers already decide.

**How the engine knows.** Each question that can be settled declares a rule,
`settled_by(state) → Settlement | None`. A question is settled when, after removing the options the
engine would refuse on the answers in force, every remaining option answers the same question, and
those options give the same estimate exactly (`materiality.invariance`) or all but one are
dominated. The energy model is the model case, and it follows from the partner (M2):
- other carbohydrate as the partner holds total carbohydrate and total energy, which admits
  `SUBSTITUTION_METHODS`;
- the standard and residual models span one space (`meaning.energy.invariance`);
- the energy-dropped residual is dominated; all-components is refused; the density models are a
  different measure.

The model family is settled when one family alone estimates the declared measure with an interval
(M4's settled line). A recoding is settled by its theorem.

**The card's form:** the kicker "Because you said"; the settling answer quoted with the card it was
given on; the so-sentence as the heading; the quiet name; one button, "Yes, that's my question"; the
alternatives behind "I meant a different comparison"; the same-question methods behind "why?".
Choosing an alternative reopens the question that settles it. **Round 2:** when the settling answer
is given on the same card (the partner settles the energy model), the confirmation replaces the
options on that card once the answer is given; it is never a screen of its own two screens later.
"Because you said" quotes only what the person said; a default set for them is said as "set for
you".

## 6 · Results' interpretation order, for any exhibit

1. **The finding**, as a headline in the person's own comparison and human unit, then **one plain
   reading** composed from the checks below (its shape, where in the outcome's spread it sits, and
   the reading the next screen checks). A size against the outcome's standard deviation is said only
   for an outcome that is roughly symmetric; for a skewed outcome with a clinical cut-off, the size is
   said on that cut-off's scale.
2. **What a reviewer will ask.** Standing questions, each set in the plan before the lock (E22) and
   checked after it, each with its computed evidence and a verdict no stronger than its test:
   - **Did it go the way you expected?** Only when an expectation was sealed before the lock. Its
     verdict: against it, when the 95% interval lies wholly on the other side; as expected; or
     cannot tell, when it includes zero. Only the exposure's sign is compared, never an adjustment
     term's (the Table 2 fallacy, Westreich & Greenland 2013).
   - **Is a straight line fair?** For a declared straight line, the spline's test for nonlinearity;
     "it bends" when p < 0.05.
   - **The typical person, or the tail?** (Round 2, E23.) Standing for an outcome with a clinical
     cut-off (a lens fact known before the lock: fasting glucose's 126 mg/dL, ADA). The primary is
     split exactly at the cut-off, `y = min(y, c) + max(y − c, 0)`: least squares is linear in `y`,
     so the two parts' coefficients add up to the primary's. "Mostly the tail" when the part above
     the cut-off is more than half of the estimate and its interval excludes zero; else
     "throughout". The share at or above the cut-off is reported beside it, per step and by fifth.
   - **Could it run the other way?** Standing whenever the exposure and the outcome come from the
     same visit. When a diagnosis marker exists (E19), the primary and the fifths are fit within the
     rows with and without it, and the groups' difference is reported beside their two estimates.
     Computed on the declared line **and** on the shape (the highest fifth against the lowest):
     "consistent with it" when, on both, the interval without the marker includes zero, the one
     with it lies on the primary's side and their difference excludes zero; "in part" when the
     groups differ on either but the association stands without the marker on one; else "not
     separated". Never "proof": a diagnosis can follow from diet. Never "points to it" on the line
     alone, and never "near zero" for an interval that reaches the overall estimate.
   - **Could a hidden cause explain it?** The robustness value beside the measured cause that moves
     the estimate most, said as a ratio of partial R²; never a pass or a fail (Cinelli & Hazlett
     2020). The benchmarks are the criterion's confounders only: the nutrients the comparison holds
     fixed define it and benchmark no hidden cause. The E-value is not reported for an outcome this
     skewed (skew 4.6): its conversion from a standardized difference is meant for a roughly normal
     outcome (VanderWeele & Ding 2017). It stays in the fixture.
   - **Could one day of recall distort it?** "Cannot be checked here" when λ is not measurable
     (the K3 noticing); with several intakes in the model, the direction is not guaranteed
     (Freedman et al. 2011).
   - **Does it describe the population?** For a survey table with no weights or design columns,
     "not as analyzed", with the design effect at which the interval reaches zero,
     `(|β| / half-width)²`.
   - **Did your choices move it?** Each of the person's choices refit with its main alternative and
     measured on one yardstick, the primary's half-width (τ₀ = 0.1, τ₁ = 0.5): a curve by its
     average step over the people's own intakes (a different summary, never a corrected slope), an
     edit of the adjustment set by its replay. The locked estimate stands; these are sensitivity
     analyses, and a prediction's miss is said.
3. **The wording**, computed from 1 and 2:
   - start at the design's floor: an association for an observational design;
   - offer "an estimated effect, its assumptions named" only when no question points elsewhere and
     no stated assumption is refuted; otherwise show it with its computed reason, not offered;
   - Results, paragraph 1: the locked primary with its n and Table 2, each declared second
     comparison, and each declared check that moved it at band 2, each with its interval;
   - Results, paragraph 2: the planned secondary analyses, labeled "In planned secondary analyses",
     each contrast with its interval, and for a split the difference between the groups beside the
     two estimates;
   - at most 90 words each; nothing decided after the lock, no expectation and no interpretation in
     either;
   - the Discussion drafts: the expectation's clause, the reading as reverse causation, the hidden
     cause with its robustness value, and each limitation;
   - the exhibits go to the Supplement; every analysis run is listed there.
4. **The placement.** The locked primary's placement is fixed; every analysis run is listed in the
   supplement (FOUNDATION §8).

## 7 · The comparison view across Models' question families

| Question family | The picture |
|---|---|
| What you study | the average day as one calorie bar, with the rest of the day; each candidate's slice lit |
| In place of what? (the partner) | the compared day: what sugar gains filled in the choice color, what each partner gives outlined on the recorded day; a mix by its projection |
| How calories are handled (settled by the partner, on its card) | the two days framed by what is held fixed; a method that asks the same question redraws nothing and says "exactly this comparison" |
| Also report (a second comparison) | a further compared day beside yours, what it gives up drawn under its own bar |
| Who is compared | the fifth eating the most against the fifth eating the least, signed, in what could explain the link and a marker of diagnosis; a pointed line adds its own group |
| The shape | the exposure's spread with two equal steps |
| The model sequence | each model's compared day, from the projection on that model's columns |
| Who the effect is for | the population split into the modifier's groups, with their counts |
| What you expect (at the seal) | the stated direction beside the comparison at the flowchart's head |
| Survey weights | the sample's average day beside the population's: the comparison is about whom |

After In place of what?, the comparison so far rides as a small emblem at the tapestry's head (M3,
M4) or at the flowchart's head (M7), never as a spine carried under each beat's own picture.

## 8 · The manuscript rail

**What it shows:** the manuscript in the guideline's order (STROBE-nut), its sentences in the
methods register with the codebook's names. Readings confirmations and the engine's mechanics live
in the record, not the paper (E14); no build notes ride on a sentence.

**How it grows.** Every confirmation adds its sentences, and the card's foot says so in one line in
the recorded green ("3 sentences added to Methods", the first words of the first), which opens the
rail. When the rail is open, the newest sentences carry the green rule. In this journey the count
runs 4 on entering Models (the study design, the participants, the outcome and the unweighted
design, each in the methods register, Round 2), 7 after the comparison, 9 after Who is compared,
11 after the shape, 12 after the sweep, 15 after the triage, 19 after Fit (the expectation, the
planned secondary analyses, the hidden cause and the choices, and the lock). From R2 the Discussion
gains its five drafts; from R3, Results gains its two paragraphs (the count reaches 21).

**The lock sentence:** "The analysis plan, with the expected direction and the planned secondary
analyses, was recorded in TurboTab before any estimate was displayed (SHA-256 ed8449eb5756); an
analysis decided afterwards is labeled as such." The fingerprint shown is the engine's for the
acted plan, which does not yet hold the expectation or the checks: it covers them once E5 and E22
are served.

## 9 · Where each beat's screen reads the engine

| Beat | Fixture moment | Design-time blocks |
|---|---|---|
| M1, M2 | `meaning.moments.enter` | `meaning.day` (the days, the mix: `capture.partners`), `meaning.energy.invariance` |
| M3 | `meaning.moments.adjust` | `meaning.balance` (signed), `meaning.diagnosis` |
| M4 | between `rest` and `gate` | `meaning.sugar`, `meaning.spreads` |
| M5 | `models.gate` | `meaning.day.days.{crude,model_1}` |
| M6 | `meaning.triage.before`, `.after_acting` | `meaning.derived` |
| M7 | `meaning.acted.flowchart` | `meaning.expectation` (E5) |
| R1 to R3 | `meaning.acted.fitted`, `meaning.results` | `meaning.results.{tail,fifths_by_group,second_comparison,diagnosis_split,spline,reviewer,mattered}`, `outcome.hist` (after the lock: `capture.checks_after_lock`, `_tail_and_groups`, `after_press`) |

## 10 · Engine requirements

Each names what the engine serves today and what it must newly serve. The design-time computation
for each is in `capture/capture.py` and `capture/build.py`.

- **E1. The return grammar:** `settled_by` per question, with `follows_from`, the quoted answer
  text and the same-question and different-question options; the energy model settled by the
  partner, said on the partner's own card.
- **E2. The answer text on every record**, so a quote is verbatim.
- **E3. The comparison view's data**, served outcome-blind per question family (§7), including the
  signed balance.
- **E4. The human unit.** The comparison carries a step (`step_kcal`, default 100 kcal); every
  estimate is served per step beside per unit.
- **E5. The expected direction** as a declaration sealed by the lock (`set_expected_direction`),
  asked at the plan.
- **E6. Model 1 in the sweep**, and the ladder served before the fit.
- **E7. The second comparison** (amended in Round 2): asked as "also report" on the partner's card,
  in the main comparison's direction; on a linear model served as the exact contrast of the
  primary's coefficients with its interval (`capture._tail_and_groups`: sugar in place of protein,
  −0.84, −1.31 to −0.36 per 25 g), never a bootstrap band; a fat partner moves the fat's kinds in
  their usual proportions; labeled everywhere.
- **E8. The model family as settled**, recorded with its basis.
- **E9. The triage acts:** a repair that changes no estimate is done for the person with Undo; a
  declared check counts as done; the repairs' predicted movement; readings asked inline; open rows
  on the flowchart.
- **E10. Which choices mattered:** the person's choices refit after the lock; a curve's band and
  its average step per the comparison's step, from the covariance the fit already computes
  (`InferenceTable.cov`, not serialized today); the ledger over declared checks, with its misses
  said.
- **E11. The interpretation checks** with §6's rules.
- **E12. The wording:** the two Results paragraphs and the Discussion drafts by §6's rule; the
  effect wording's refusal reason.
- **E13. Names:** codebook labels on every card, sentence and hover panel; the outcome's unit.
- **E14. The paper's sentences:** the adjustment set in one sentence, naming the comparison's own
  terms as such; the medicines' role as a marker; the study design and participants in the methods
  register (never the engine's purpose, sweep or triage sentences, never "rows" for participants);
  the readings out of the paper; the imputed values of a backup model's measures said.
- **E15. The flowchart** shows the comparison, the expectation, the declared checks, the second
  comparison, what Results will check, and any open row.
- **E16. Sources to verify into the registry:** Hutcheon, Chiolero & Hanley 2010; Austin 2009;
  Banna et al. 2017; Shaper, Wannamethee & Walker 1988; Frisch & Waugh 1933 and Lovell 1963; Altman &
  Royston 2006; Kish 1965; the American Diabetes Association's Standards of Care (fasting plasma
  glucose ≥ 126 mg/dL); the NHANES analytic guidelines (weights and design variables). **NHANES's
  Blood Pressure & Cholesterol questionnaire:** verified on the 2015–2016 codebook (BPQ_I): "No" to
  ever being told of high blood pressure (BPQ020) skips the blood-pressure medicine questions, and
  "No" to being told to take cholesterol medicine (BPQ090D) ends the section before BPQ100D. Which
  items the table's `meds_hbp` and `meds_chol` are, and the skip rules in the other eight cycles,
  are still to verify.
- **E17. The partner asked.** The comparison asks what the exposure replaces; the adjustment set's
  dietary terms follow from it and are named "part of your comparison", never confounders.
- **E18. The partner mix.** For a comparison that holds total energy but not the other nutrients,
  each source's movement with the exposure on the comparison's own columns, outcome-blind.
- **E19. A diagnosis marker** (amended in Round 2). A codebook skip pattern ("asked only after a
  diagnosis") read as a marker, never as a covariate (a blank there means "not asked", so as a
  covariate it drops everyone not asked); its gradient across the exposure's fifths before the
  lock, said at the same age and gender; its route "Check it in your plan", which declares the
  split (the primary and the fifths within each group, with their difference) as a modifier sealed
  by the lock.
- **E20. Reverse causation as a standing question** wherever the exposure and the outcome come from
  the same visit, its verdict computed on the line and on the shape (§6).
- **E21. The survey design question:** "unweighted" in the methods when the table has no design
  columns, and the design effect at which the interval reaches zero.
- **E22. What Results will check, sealed with the plan** (Round 2). The standing checks (the other
  shape, the tail for an outcome with a clinical cut-off, a hidden cause, each choice undone) and the
  person's own (a split from a noticing, the declared checks) are declared at Fit, covered by the
  fingerprint, listed on the flowchart, described in the Methods and reported as planned secondary
  analyses. An analysis decided after the lock is the only one labeled post hoc.
- **E23. The tail question** (Round 2). For an outcome with a clinical cut-off (a lens fact), the
  primary split exactly at the cut-off and the share past it, per step and by fifth; the size said
  on that scale, not against the standard deviation of a skewed outcome. A quantile family (the
  median's contrast) would add "the typical person" directly; the engine has none today, so the
  split is used.
- **E24. Repairs per column** (Round 2). The SAS-zero repair reads every column's misread zeros as
  0 in one option; a diastolic pressure of 0 mmHg wants its own reading (0 or missing). It changes
  no estimate here (blood pressure is in no model), so the design says so instead of choosing.

**Engine findings from this build** (the engine was not changed; each is for its owner):
- The packs' backup model for the clinical group puts the raw medicine answers in Model 3; a blank
  there means "not asked", so complete cases keep only the 2,996 rows that answered both questions
  (`adjust_clinical_backup`). The marker (E19) has no blanks. Round 2's choices row undoes the
  person's edit on the four blood measures alone (`adjust_blood_backup`: the primary unchanged; its
  backup model, with body size and the blood measures, −0.58, −0.89 to −0.28, on all 21,849).
- seq 28's sentence says the second swap's band comes from "bootstrap resamples of every analyzed
  row"; the band's own caption says 10,000 of 21,849 rows. Round 2 no longer uses the band (E7).
- seq 15's estimand sentence says "per unit of sugar" while every estimate is shown per step (E4).
- seq 12 says "no row is missing any predictor" while Model 3's body measures include values the
  source imputed (flagged in Your data); E14.
- seq 5's purpose sentence ("declared for `inference`: associations are estimated with their
  uncertainty"), seq 30's sweep sentence and seq 31's triage sentence are engine prose; Round 2
  keeps them in the record and out of the paper (E14).
- The engine's implausible-intake reading counts 501 rows outside a generic 500 to 5,000 kcal, while
  the screens it offers (by sex) remove 1,614 and 1,419; the triage row should say the screens'.
- Results counts five exhibits on this journey (`quest.served_exhibits`); three of them are Describe
  and Predict exhibits. The build keeps the engine's count.

## 11 · Rulings this asks for

1. The expected direction is asked at the plan, the last thing before Fit; it is one standing
   question among eight, never the trigger for the rest.
2. In place of what? asks the partner and says what it settles on the same card; a second
   comparison is "also report" there, in the same direction.
3. Reverse causation is a standing question for same-visit designs; a skip-pattern marker of
   diagnosis may be drawn before the lock (it reads no outcome) and its split sealed into the plan
   from the noticing's route.
4. What Results will check is sealed with the plan and reported as planned secondary analyses.
5. For an outcome with a clinical cut-off, the tail question is standing, and the size is said on
   that scale.
6. Model 1 moves into the Confirm sweep, and every model is labeled by the comparison it makes.
7. A repair that changes no estimate is done for the person with Undo; only a decision that could
   move the estimate is asked.
8. Results is two paragraphs: the primary with its declared checks, then the planned secondary
   analyses; the expectation and the interpretation are Discussion drafts.
9. Dark mode's tapestry is the ground's own temperature, one step lifted with a hairline edge, held
   to this page until Nolan sees it; then a kit token (FOUNDATION §6).

## 12 · How the first critique was answered (Round 1)

Beat numbers in this table are Round 1's (M4 How calories are handled, M5 What you expect, M7 A
second swap, M8 Set for you, M9 What your data raised, M10 Your plan, R3 Which choices mattered, R4
Your sentence).

| Critique point | What the build did |
|---|---|
| Deep 1: the meaning came from the wrong answer | M2 asks the partner; M3 draws what could explain the link; M4 quotes the partner; M8 and R1 label each model's comparison |
| Deep 2: Results read the sign | R1 said size and shape; R2 asked six standing questions with the diagnosis split; R3 one yardstick with the person's edits |
| Deep 3: nothing builds up | the spine carried each answer's mark; the ladder drew each model's day; arrivals at the card's foot |
| E-value per gram beside a per-25 g estimate | per 25 g by `e_values` with δ the step (Round 2 then stops reporting it: §6) |
| "The same comparison in each row" | removed; each row says what it compares |
| Carbohydrate and calories as confounders | "part of your comparison", in the card and the methods |
| Verdicts stronger than tests | computed by rules (§6) |
| No survey design | standing question with the design effect; "unweighted" in the methods and Table 2 |
| The shape exhibit | spline with its band from the covariance; the fifths with their intervals |
| R3 omitted the person's edits | replayed and ranked |
| Methods faults | one substitution studied, the second labeled; per 25 g/day throughout; no build notes |
| Cosmetics (quiet names, two yeses, legends, type sizes, 1280 px, dark mode) | fixed as listed in Round 1's commit |

## 13 · How the third critique was answered (Round 2)

Beat numbers in the critique are Round 1's; the right column gives Round 2's.

**Methods and reporting**

| Critique point | What Round 2 does |
|---|---|
| 4.1 The standing checks are not sealed, so the paper calls them post hoc | sealed on M7's flowchart ("Sealed with your plan: what Results will check"), in the Methods' planned-analyses sentence and the lock sentence; Results says "In planned secondary analyses"; "Separate it further" is the only analysis labeled as decided after the lock; "run after" and "decided after" are no longer confused (E22) |
| 4.2 "Points to it" too strong, on the rejected line | the verdict is computed on the line and on the shape (the highest fifth against the lowest, within each group): "In part"; the Discussion draft drops "near zero" and says the association was stronger with a diagnosis though glucose was highest in the lowest fifth without one |
| 4.3 R4's paragraph | two paragraphs, each at most 90 words (77 and 90): the primary with Table 2, the second comparison and the declared checks; then the planned analyses with the difference between the groups (1.36, 0.52 to 2.21) beside the two estimates; the expectation and "asked after a diagnosis" moved to the Discussion and Methods; the exhibits to the Supplement |
| 4.3 R3 not safe | the choices forest (now R2's eighth question) leads with "Your sealed estimate stands; these are sensitivity analyses beside it"; the curve's row says "a different summary, not a corrected slope" |
| 4.4 The medicines' contradictory roles | one role: a marker of a diagnosis, never a covariate (the Methods' marker sentence; M3's lines; the "Could it cause sugar intake? no" beside the noticing is gone); the 2,996 is reported as the engine finding it is, and the person's edit is undone on the blood measures alone |
| 4.5 Size on a skewed outcome | the standard-deviation size is gone from screen and paper; the size is said on the clinical scale: the share at or above 126 mg/dL by fifth, and the exact split of the primary at the cut-off (two thirds from the diabetic range); the E-value is not reported for this outcome |
| 4.6 Engine prose in the Methods | the purpose, sweep and triage sentences stay in the record; the study design and participants are said in the methods register; no "rows" for participants (a test checks) |
| 4.7 Unsourced claims | "asked only after one" sourced to the NHANES BPQ codebook (verified for 2015–2016; E16); "the people most likely to have been told to change their diet" removed; "they change glucose" now "they may change glucose" |
| 4.7 The second swap's bootstrap | the exact contrast of Model 2, −0.84 (−1.31 to −0.36) per 25 g, same direction as the primary |
| 4.7 Benchmarks include the comparison's own terms | benchmarked on age, gender and survey cycle only |
| 4.7 "No" for the population question | "Not as analyzed" |
| 4.7 Diastolic zeros | said plainly: 119 of the 152 are diastolic pressures, and blood pressure is in no model, so how they are read moves nothing; a per-column reading is an engine requirement (E24) |

**Deep and density**

| Critique point | What Round 2 does |
|---|---|
| Deep 1: Results gives the parts, not the finding | R1 is rebuilt around one reading and one picture (§4, R1); the curve moves to R2; the synthesis comes first, not last |
| Deep 2: the reverse-causation thread half-built | M3's noticing is said at the same age and gender, with "Check it in your plan"; M5's group line names the diagnosis; R2 draws the split on the line and the shape at rest; the Discussion is corrected |
| Deep 3: density | 7 Models and 3 Results screens; repeated lists gone (R2's table, R3's list); "No glucose value is read" once (M1); "after the plan was fixed" once (Separate it further); the partners no longer repeat "25 g more sugar"; nothing below the fold at 1280 × 800 (measured); type five sizes |
| Deep 4: the arc is long | the critic's table taken as given: M4 and M7 fused into M2, M5 moved to the plan, R3 folded into R2 |

**Cosmetic**

| Critique point | What Round 2 does |
|---|---|
| Internal document names on screen | removed (a test checks every state) |
| "Other wordings" opens out of sight | it scrolls into view when opened (`#/r3-alts`) |
| Bars drop their names at 1280 | a compact bar puts the name above the grams when one line does not fit; the ladder's rows are drawn bare under one labeled day |
| Seven beats hide 40 px or more at 1280 | none does (§2 budgets, measured) |
| M3's question heading over a filled-in answer | the heading is a statement |
| M9's "Confirm all 5" beside "Three not decided yet"; 501 against 1,614 and 1,419 | one decision; "Confirm all 5" waits for it; the counts are the screens' own |
| R3's "0.08" with no unit, and the ninefold miss | said in words: predicted about a tenth of its margin, moved three quarters |
| Two sign conventions on one screen | leads in words, forests signed |
| M7's two opposite-direction comparisons on one bar | the second comparison runs in the same direction and draws what it gives up under its own bar |
| Copy that fails read aloud | each line rewritten or removed ("only the direct part", the quiet names of the backup model and the mediators, "agree to 13 significant digits", "Because you said … each person appears once", "set for you on In place of what?.", "In Results: your plan's main estimate always is.") |
| Copy that sings | kept |

**Where the critique was not followed, and why**
- **The robustness value with the HC3 t (2.05% rather than 2.25%).** The robustness value that
  brings the estimate to zero, RV_q, is an algebraic function of the coefficient's partial R² with
  the outcome, t² / (t² + dof) from the classical t; it describes the point estimate, whatever
  standard error the interval uses (Cinelli & Hazlett 2020, §4; the engine's `LinearSensitivity`
  docstring). Only RV_{q,α}, which brings the interval to zero, depends on the standard error, and
  it is not reported beside an HC3 interval. Putting the HC3 t into the formula would not give a
  partial R². The 2.3% stays.
- **The median's contrast (about 2 mg/dL at the median).** The engine has no quantile family, and
  every number must come from the engine's functions. The tail question uses the exact split at the
  cut-off instead, which answers "typical person or tail" for the estimate itself (E23).
- **The with-and-without pair for the marker.** The critique notes that by the app's own criterion
  a variable of unknown timing gets the with-and-without pair (its check moves −0.50 to −0.52).
  The marker is not adjusted for: it is a proxy for diagnoses that may themselves follow the
  outcome's condition, and adjusting for it would ask whether it confounds the average slope, a
  different question from where the association lives. The plan's check is the split with the
  difference, which is the question reverse causation asks.
- **Pooling the four upper fifths ("the lowest fifth against the rest", 4.1 and 8.3).** That
  contrast was chosen after the curve was seen. The plan seals the fifths against the lowest; the
  verdict and the paper use the highest fifth against the lowest, the prespecified contrast
  (−5.2 and −13.5).
- **"Under 54 g a day" written as a number.** The cut point is 53.7 g; "under 54 g" is computed as
  the next whole gram above it (exact, since the lowest fifth holds its cut point).
- **R3's card above 120 words.** Its two paragraphs are the paper's own text (167 words between
  them); the card holds nothing else but the heading and the wording's name. The 120-word budget is a
  check (FOUNDATION §0 ruling 5); the screen's tapestry is held to Table 2 and one outline.

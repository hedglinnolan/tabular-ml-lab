# Beats: Predict, Models and Results (NHANES fasting glucose)

Written 2026-10-10 by the design owner, before any screen is built (FOUNDATION §0, ruling 6). It is
the contract W2's build ("Predict, drawn") is held to: what the person understands and feels at
each step from entering Models to "Put this in my paper", what the card says, what the tapestry
draws, what the manuscript gains, and what the engine must serve. Its form is the Estimate
journey's (`BEATS_MODELS_RESULTS.md`, round 2), and it is written against what that journey's three
critiques caught (`QUEST_LOG_CRITIQUE_2026-10-10.md`, and the two rounds in
`BEATS_MODELS_RESULTS.md` §12–§13): asking what is already settled, drawing mechanics instead of meaning, a sign instead of a finding, checks done after the
fact reported as planned, and density.

**The journey.** NHANES: 21,849 adults with a fasting glucose, nine survey cycles (2001–2002 to
2017–2018). The outcome is fasting glucose in mg/dL; the goal is Predict. It is Nolan's own
project, and two of its answers are his rulings of 2026-10-10:
- the model is used **at a visit with a fasting blood draw**, so it may read HDL cholesterol and
  triglycerides from the same draw, and nothing known only after the visit;
- whether it reads the **medicine answers** is recommended by the app from the intended use the
  person chooses.

This walk takes the intended use "Any adult it is used on, diagnosed or not" (beat Q); every beat
says what changes under the other use.

**Where the numbers come from.** A light live run; no number here is typed.
- `predict-capture/capture.py` drives the real server in process (`server_drive.local_server`, a
  fresh `TURBOTAB_HOME` under the scratchpad, two workers, two threads), answers the journey,
  presses Fit (`Drive.artifact` presses and releases), names the final model, opens the held-out
  rows once and exports the bundle. Families: least squares, ridge and the elastic net only, each
  continuous measure a curve (ruling 6). It took 3 min 20 s at `bf289d1c`: the fit 2 min 42 s
  (the engine had estimated 67 s), evaluation and explanations 17 s more.
- `predict-capture/extras.py` computes, with the engine's own functions on the same nine cycle
  folds, the Results numbers the engine does not serve yet (P4, P6, P10). Its pooled scores
  reproduce the engine's cross-validated MSE exactly (1,047.352 for ridge).
- Both write their JSON beside them (`capture.json`, `extras.json`), trimmed of per-row arrays.
- **The Models beats read no glucose value.** Their numbers come from `capture.before_fit` and the
  outcome-free counts. **Every number that reads glucose was computed after the held-out rows were
  opened** (`capture.after_seal`, `extras`) and appears in Results beats only. Under Predict the
  engine may show the outcome on the training rows once the held-out rows are drawn
  (`outcome_gate.alone_gate` and `beside_gate`, each view recorded as looked at); where a Models
  tapestry is designed to draw it, its numbers are marked ⟨engine-filled⟩.
- **No tuned trees.** A tuned shelf on these rows takes hours (the shelf's own estimates: random
  forest about 2 hours, XGBoost about 17 minutes, each tuned; boosted trees are untuned until C6a
  phase 3 lands). Wherever a tree family's number would appear, it is ⟨engine-filled⟩.
- **Three runs, one draw.** All three drew the same held-out rows (seed 0): straight lines (the
  engine's default); curves with least squares named final; curves with ridge named final. The
  last two differ only in the final model: the first form of ruling 4 named least squares, and the
  rule was refined from cross-validated evidence alone (least squares' body-size terms are not
  identified) before any held-out number was read. The third run is the one cited. The first two
  runs' held-out scores were computed by the engine at their openings and never read. Ruling 6
  (curves by default) was adopted after the first run's cross-validated benchmark showed curves
  ahead; it is a product default, set before Fit in the walk.

**Names.** The cards use the NHANES variable labels in plain words (P7, design-time): `hdl` is HDL
cholesterol, `bp_sys` systolic blood pressure, `meds_hbp` "now taking prescribed medicine for high
blood pressure", and so on.

## 1 · The arc

| | Beat | Stage | One decision | What the person feels |
|---|---|---|---|---|
| Q | Whom it is for | Your question | the intended use: "Any adult it is used on, diagnosed or not" | purposeful: the model has a job |
| W | The held-out rows | Who's in | how many stay unseen: one in five | protected |
| M1 | When it is used | Models | the moment of use: "At a visit with a fasting blood draw" | grounded: "that's the moment I mean" |
| M2 | The medicine answers | Models | read them, each blank as "not asked" (Recommended from Q) | caught, then trusted |
| M3 | Which kinds of model | Models | the families: the three that add up curves, in seconds | in control of the cost |
| M4 | Set for you | Models | Confirm all 4 | respected, quick |
| M5 | Your plan | Models | checked on survey cycles it never saw (Recommended), then Fit | ownership; a little suspense |
| R1 | How good it is | Results | none: reading | honest clarity |
| R2 | Where it misses | Results | none: reading | sobered, but armed |
| R3 | What drives it | Results | none: reading | curious: "so that's what it reads" |
| R4 | Your final model | Results | the final model (ridge, Recommended), then open the held-out rows | decisive: the moment of truth |
| R5 | Your sentence | Results | "Put this in my paper" | proud: defensible, and honest about its limits |

Five Models screens and five Results screens, as the reasoning needs (the Estimate journey ended at
seven and three: Predict has no comparison to build and no lock, and it has an opening). Q and W
are not new screens: they are Your question's intended-use card and Who's in's draw, written here
because Models stands on them.

**The arc builds, then rests.** M1 draws the visit as a timeline; from M2 on it rides as a small
emblem at the tapestry's head, each beat lighting what it settles. M5 draws the whole plan with the
nine cycles. R1 turns the timeline's last mark, glucose from the same draw, into the miss. R3 hands
three answers back: the lipids M1 let in, the medicine answers M2 kept, the curves M4 set.

**"What could fool you" is said three times, where each belongs:** in M1, as what the model may
not read (anything after the visit, and the period); in M2, as a blank that means "not asked"; in
R2, as where it misses (the diabetic range, the groups, the period's drift).

## 2 · How a beat is written

As `BEATS_MODELS_RESULTS.md` §2: what the person understands and feels; the one decision; the card
in final copy (the heading, the return sentence, the options with their one-line consequences, the
quiet technical name on the option's top edge, the button, what sits behind "Why?"); the tapestry
(what it draws, from which data, what pointing changes, its caption); the sentence the manuscript
gains, in the methods register and TRIPOD+AI's order; and the engine (what it serves, what it must
newly serve, what conflicts). One type family in five sizes; about 120 words on a card and 250 on a
screen, as checks; nothing below the fold at 1280 × 800. The card says; the tapestry draws.

Two additions for Predict:
- **A Models beat never quotes a number that reads glucose.** A Results beat says which rows it
  read: the out-of-cycle estimates of the 17,479 development participants, or the 4,370 held-out
  participants after the opening.
- **The miss is said in mg/dL.** The card says the average miss (the mean absolute error); the root
  of the squared miss (the RMSE, what the models are compared on) rides in the quiet register. R²
  is said as a share: "about a sixth of the differences between people".

## 3 · Before Models: the two answers it stands on

### Q · Whom it is for (Your question)

**Understands:** "My model estimates fasting glucose for any adult it is used on, diagnosed or not;
its accuracy will be reported for women and men, three age groups, and people with and without a
diagnosis on record." **Feels:** purposeful.

**Decision:** the intended use. Nothing in the data says what a model is for, so no option is
Recommended.

**Card** (Your question's intended-use card; W1-F builds today's version, this is its copy)
- Kicker: "Your question · What it is for"
- Heading: "Whose fasting glucose will it estimate?"
- Options:
  - "Any adult it is used on, diagnosed or not": "It estimates a value for everyone it meets; your
    paper reports how close it comes." Quiet name: "intended use: estimation in the whole population
    at the moment of use".
  - "Adults whose high glucose is not yet known": "It looks for undiagnosed high glucose; people
    already diagnosed are not its population." Quiet name: "intended use: screening for undiagnosed
    hyperglycemia".
  - "A yes-or-no call: whom to test", labeled "Not with this outcome": "A call needs a yes-or-no
    outcome: fasting glucose of 126 mg/dL or more." Its exit: "Change the outcome". Quiet name:
    "decision support (decision curve analysis, Vickers & Elkin 2006)".
- Set for you, beneath the options: "Its accuracy is reported for women and men, three age groups,
  and people with and without a diagnosis on record." Quiet name: "subgroup performance (TRIPOD+AI
  14, 23a)". One quiet line under it: "Not in your table: race or ethnicity, income, education. Your
  paper will say their groups could not be checked."
- Why?: "What the model is for decides whom it learns from, which answers it may read, and how it is
  judged. A model used only on people not yet diagnosed should learn from people like them; a model
  used on everyone may read what a diagnosis leaves behind, such as a medicine."
- Button: Continue.

**Tapestry:** who the model is for, from the table's own answers (outcome-free). One bar of the
21,849 people, with the 7,946 who have a diagnosis on record (they were asked a medicine question,
which NHANES asks only of people told to take the medicine) as one segment and the 13,903 without as
the other.
- Pointing at "Any adult it is used on" lights the whole bar.
- Pointing at "Adults whose high glucose is not yet known" leaves the bar gray, with one line:
  "Who already knows they have diabetes is not in your table (NHANES asks it in its diabetes
  questionnaire, not joined here), so the model would learn from people already diagnosed and
  treated."
- Pointing at the groups line splits the bar into its groups: women 11,195 and men 10,654; the
  development rows' age groups, 18 to 36 (5,891), 37 to 58 (5,833), 59 and over (5,755); the
  diagnosis on record.
- Caption: "Your table's people, by what it records. No glucose value is read."

**Manuscript.** Introduction (TRIPOD+AI 3b), drafted for the author: "The model is intended to
estimate fasting plasma glucose for adults seen at a visit with a fasting blood draw, whether or not
they have a diagnosis [author: the care pathway and its intended users]." Methods (TRIPOD+AI 14):
"Performance was reported overall and by gender, age group and whether a diagnosis was on record;
race and ethnicity, income and education were not recorded in the data, so performance could not be
examined across them."

**Engine.** Serves `decisions.SetIntendedUse` (`decisions.py:1771`); for a numeric outcome only
`risk_estimation` is accepted (`decision_curve._intended_use_fits`, `:473`, refuses decision
support with the exit "Risk estimation only"). The groups are scored in `stages/evaluation.py`
(`:323`, `decision_curve.subgroup_performance`, `:263`; an amount in thirds by
`subgroup_labels`). Must newly serve **P9** (the use's population, and the copy "estimation",
never "risk estimation" for a value) and, in **P4**, the derived group "a diagnosis on record" from
the skip pattern (shared with BE19).

### W · The held-out rows (Who's in)

**Understands:** "One person in five is set aside now, unseen until I name my final model; their
score is the one my paper reports." **Feels:** protected.

**Decision:** how many stay unseen. This journey keeps the engine's first: one in five.

**Card** (Who's in's last Decide under Predict, FOUNDATION §3)
- Heading: "How many people should stay unseen until the end?"
- Options, in the seal plan's order:
  - "One in five: 4,370 people" (Recommended: the seal plan's first, "At this size the held-out rows
    measure precisely, and they guard against the analyst's own overfitting"): "Their score will be
    known to about ±0.02 in R²." Quiet name: "a held-out test set (Steyerberg 2018)".
  - "One in ten: 2,185 people": "About ±0.03; more people to learn from."
  - "Three in ten: 6,555 people": "About ±0.02; fewer to learn from."
  - "None: cross-validation only": "Everyone trains and is scored in turn; no untouched final
    score."
  - "The latest survey cycle, whole (2,270 people)", labeled "Not available yet": "A later period
    held out; your plan already scores the latest cycle from earlier ones only." Its exit: "One in
    five, at random". Quiet name: "temporal validation".
- Why?: "They protect your result from your own choices: everything you try in Models and Results
  is judged without them, and they open once, after you name your final model."
- Button: Continue.

**Tapestry:** the participant flow, outcome-free: 21,849 people, then 4,370 set aside (one quiet
block, "unseen until you name your final model"), then 17,479 to build and check the models.
Beneath it, each cycle split the same way by chance (2001–2002: 530 of 2,501 set aside; 2017–2018:
444 of 2,270). Pointing at an option resizes the block; pointing at the latest cycle moves the whole
2017–2018 bar into it. Caption: "Drawn once, at random (seed 0). Their glucose values stay unread
until you open them."

**Manuscript** (TRIPOD+AI 12a): "Before model development, a random 20% of participants (n = 4,370)
was set aside and used once, to evaluate the final model; the remaining 17,479 were used for
development and internal validation."

**Engine.** Serves `seal.plan` (`seal.py:850`) and `split_offer` (`:825`), the draw
(`stages/rows.draw_split`), the sealed scores (`seal.serve_fit`), and the outcome's gates on the
training rows once drawn (`outcome_gate.py:95`, `:113`). Must newly serve **P21** (a holdout of
whole levels of a period column); "Not available yet" until then.

## 4 · Models

### M1 · When it is used

**Understands:** "The model is used at a visit with a fasting draw, so it reads the 20 things known
by then: who the person is, what they have been told about blood pressure and cholesterol,
yesterday's diet, body measures, blood pressure, and HDL and triglycerides from the same draw. It
reads nothing made after the visit, and not the survey cycle: a visit after my data is in none of
my nine." **Feels:** grounded: "that's the moment I mean".

**Decision:** the moment of use. In this journey, Nolan's ruling: "At a visit with a fasting blood
draw". No Recommended: the use decides.

**Card**
- Kicker: "Models · When it is used"
- Heading: "When will the model be used?"
- Lede: "It may read only what is known by then."
- Options:
  - "Before any blood is drawn": "It reads what is asked and measured at the visit: 18 things, not
    the lipids." Quiet name: "predictors available before laboratory testing".
  - "At a visit with a fasting blood draw": "It also reads HDL and triglycerides from the same draw:
    20 things." Quiet name: "predictors measured up to the fasting draw (TRIPOD+AI 9b)".
- Settled by the moment, one quiet line (not a choice): "Not read: the survey cycle (a later visit
  is in none of your nine; your plan uses them to check the model across periods), the respondent
  number, and six flags made in processing."
- Why?: "A model that reads something known only after its moment looks better in a paper than in
  use. Glucose comes from the same draw as the lipids, so the model is for a visit where glucose
  was not measured, or not yet."
- Button: Continue.

**Tapestry:** comparison view, the visit as a timeline (outcome-free).
- Three lanes, left to right:
  - **Before the visit** (the interview at home): age, gender; the two medicine questions, asked
    only of people told to take medicine for high blood pressure or high cholesterol.
  - **At the visit** (the exam): yesterday's diet, recalled once (total calories, protein,
    carbohydrate, total sugars, total fat and its three kinds); body measures (weight, height, body
    mass index, waist); blood pressure (systolic, diastolic); the fasting draw: HDL cholesterol,
    triglycerides, and fasting glucose, drawn in outline as what is estimated, by its name and unit
    only.
  - **After the visit:** "Nothing in your table", with the six processing flags in the quiet color.
- Beneath the lanes, one strip: the nine survey cycles, 2001–2002 to 2017–2018, and "A visit after
  these is in none of them."
- Pointing at "Before any blood is drawn" moves HDL and triglycerides to "not read", outlined in the
  choice color; the count reads 18. Pointing at "At a visit with a fasting blood draw" lights them,
  "+2 from the same draw"; the count reads 20.
- Caption: "When each column of your table is known, at the moment the model is used. No glucose
  value is read until you press Fit."

**Manuscript** (TRIPOD+AI 9a, 9b): "The model was intended for use at a visit with a fasting blood
draw. All variables available at that time were candidate predictors, with no selection before
modeling: age and gender; whether the participant was taking prescribed medicine for high blood
pressure and for high cholesterol; total energy, protein, carbohydrate, total sugars and total,
saturated, monounsaturated and polyunsaturated fat from one 24-hour dietary recall; weight, height,
body mass index and waist circumference; systolic and diastolic blood pressure; and HDL cholesterol
and triglycerides from the same fasting blood draw. Survey cycle was not a predictor."

**Engine.** Serves the roles (`decisions.SetRoles`; `PREDICTOR_ROLES`, `decisions.py:2821`).
Must newly serve **P3** (the moment of use: `q:moment_of_use`, each column's moment from the lens's
codebook, for NHANES interview, examination or laboratory, or asked; compiled to predictor roles;
the 9a and 9b sentences) and **P14** (a period column is a validation grouping under a later
moment, never a predictor). Conflict **C1**: the dietary lens proposes the role "exposure" for the
nutrients under Predict, and that fires the energy-model and substitution questions; the capture
gave them the role P3 compiles to (a predictor, the engine's covariate).

### M2 · The medicine answers

**Understands:** "Two questions in my table were asked only of people told to take medicine for
high blood pressure or high cholesterol, so a blank means 'not asked', not 'unknown'. For a model
used on any adult, the app reads them, each blank as 'not asked'. Had I said the model looks for
high glucose nobody knows about, it would leave them out." **Feels:** caught, then trusted: the app
knows what a blank means here.

**Decision:** whether the model reads the medicine answers. Recommended from Q; in this journey,
read them.

**Card**
- Kicker: "Models · The medicine questions"
- Heading: "Should the model read the medicine answers?"
- Return, the lede, in the "because you said" form: "Because you said" over the quote "Any adult it
  is used on, diagnosed or not." (Your question · What it is for), then "At the visit these answers
  are known, and they say whether a doctor has prescribed the medicine, so the app recommends
  reading them."
- Options:
  - "Read them, a blank as 'not asked'" (Recommended): "Three answers each: taking the medicine,
    told but not taking, not asked." Quiet name: "a skip-pattern blank kept as its own level".
  - "Leave them out": "It reads 18 things instead of 20."
- Why?: "NHANES asks whether someone now takes a prescribed medicine only after they were told to
  take it (its Blood Pressure and Cholesterol questionnaire). So 15,552 of your 21,849 people were
  never asked about blood-pressure medicine, and 17,204 never about cholesterol medicine. Filled
  with the most common answer, all of them would read as taking it; kept only where both were
  answered, 2,996 people would remain, chosen by their diagnoses. Had you said the model looks for
  high glucose not yet known, the app would recommend leaving them out: cholesterol medicine is
  recommended for most adults with diabetes, and blood-pressure medicine for those whose pressure
  is raised, so in your table part of what these answers carry is a diabetes diagnosis already made,
  which the people it screens do not have."
- Button: Continue.

**Under the other use** ("Adults whose high glucose is not yet known"): "Leave them out" is
Recommended, with the last sentence of Why? as its reason, and a noticing joins Results' notes as
"act on it": "People already diagnosed with diabetes are in your table and are not this model's
population; join NHANES's diabetes questionnaire in Your data to leave them out."

**Tapestry:** the two questions' answers, outcome-free.
- Two bars of 21,849: blood-pressure medicine (taking 5,527; told, not taking 770; not asked
  15,552) and cholesterol medicine (taking 3,644; told, not taking 1,001; not asked 17,204). "Not
  asked" carries its meaning: "never told to take it".
- One line beneath: "Asked at least one: 7,946 people (36%), from 27% of the 2001–2002 cycle to 44%
  of 2017–2018."
- Pointing at "Read them" lights each bar's three parts in the choice color, as the three answers
  the model reads. Pointing at "Leave them out" turns both bars to the quiet color, "not read".
- A permanent quiet line: "Filled with the most common answer, the 15,552 not asked would read as
  taking blood-pressure medicine."
- Caption: "What your table records for each question. No glucose value is read."

**Manuscript** (TRIPOD+AI 6c, 9b, 11): "Use of prescribed medicine for high blood pressure and for
high cholesterol was recorded only for participants who had been told to take it (NHANES Blood
Pressure and Cholesterol questionnaire). Each was entered as a predictor with three levels (taking,
told but not taking, and not asked), the last marking that the participant had not been told to
take it; no value was imputed."

**Engine.** Serves the blank as a level (`pipeline.MISSING_LEVEL`; "the honest reading of a column
like `meds_hbp`", `models/pipeline.py:16`, `level_columns` `:173`) and the findings
`binary_text__meds_*` ("a blank may mean not asked"). Must newly serve **P1** (W1-H, in flight:
the blank as its own level recommended for a skip pattern, never the most common answer), **P9**
(the recommendation read from the intended use, with its reason), and the skip fact as a lens fact
(the Estimate requirement E16's BPQ check, extended to all nine cycles). Conflict **C2**: today the
missing-values question ranks "Single fill in each training fold" first, and its fill for a category
writes "taking" into every "not asked".

### M3 · Which kinds of model

**Understands:** "Three kinds of model add up curves and take seconds here; the tree models can bend
and combine measures, and take from minutes to hours. I fit the fast three now; the trees can run
later, as a job." **Feels:** in control of the cost.

**Decision:** which families to fit. In this journey: least squares, ridge and the elastic net.

**Card**
- Kicker: "Models · Which kinds of model"
- Heading: "Which kinds of model should it try?"
- Lede: "Ranked for your 17,479 people and 20 measures, before any glucose value is read."
- Options (checkboxes in the shelf's order, each with what it can draw and its cost; the quiet name
  on the top edge):
  - Boosted trees: "Steps that can bend and combine measures. ⟨engine-filled: about N hours, once
    tuned⟩" Quiet name: "gradient-boosted trees".
  - Elastic net: "Curves pulled toward flat; it may drop weak measures. About 45 seconds." Quiet
    name: "the elastic net (Zou & Hastie 2005)".
  - Ridge: "Curves pulled toward flat together; it keeps every measure. About 15 seconds." Quiet
    name: "ridge regression (Hoerl & Kennard 1970)".
  - Random forest: "Many deep trees, averaged. About 2 hours."
  - XGBoost: "Steps, tuned. About 17 minutes."
  - Least squares: "Each measure a curve, added up. About 5 seconds." Quiet name: "linear regression
    with restricted cubic splines".
  - Robust regression: "People far from the rest count less. About 35 seconds." Quiet name: "Huber
    regression".
  - One quiet line for the rest: "Not for this table: a screened elastic net (nothing here needs
    screening), mixed and GEE models (no one appears twice), feature-wise tests (they make no
    predictions)."
- Under the options, the ticked families' cost in one quiet line: "about 1 minute in all" (the
  engine's estimate for the three, 67 s).
- Why?: "The ranking reads what each model would be given (your people, your measures, their curves)
  and never a glucose value. A tree family can find bends and combinations the others cannot;
  whether that is worth its cost here is a question Results answers once one is fitted."
- Button: Continue.

The shelf's own ranking is the order; no family is labeled Recommended, because the choice here is
the person's budget.

**Tapestry:** what each kind of model can draw, on one shared axis (an illustration, outcome-free):
a curve; a curve pulled toward flat (ridge); a curve with parts set to zero (the elastic net); a
staircase (trees); a smoothed staircase (the forest). Beneath, each family's cost as a dot on one
time axis, seconds to hours, with the hold's line at about 2 minutes: "Longer fits wait for Fit and
run as a job." Pointing at a family lights its drawing and its dot. Caption: "What each kind of
model can draw, not what it will find. No glucose value is read."

**Manuscript** (TRIPOD+AI 12c): "Three model families were developed: least-squares linear
regression, ridge regression and the elastic net. Tree-based families were not fitted [author: the
rationale]."

**Engine.** Serves the `shelf` stage (rank, fit, `inductive_bias`, the cost `estimate`, and Riley's
sample size), `models/cost.py`, and the hold (`fit_press.HOLD_SECONDS`, `:51`). Must newly serve
the live ranking on each family's own input (MC-6, planned) and **P15** (each family's own recipe:
the curves only for the families that draw lines). Conflict **C3**: `set_levers` reaches every
family, so the tree families are handed spline columns too; the forest's estimate rose from about 2
hours with straight lines to about 11 with curves. The costs above are the straight-line run's for
the trees and the curves run's for the others.

### M4 · Set for you

**Understands:** "Four conventions are set for me: each measure may bend; every measure is kept;
ridge and the elastic net choose how hard to pull inside each fold; and the four body measures,
which carry nearly one quantity, are all kept and read together." **Feels:** respected, quick.

**Decision:** "Confirm all 4".

**Card:** heading "Here are the 4 choices set for you"; each line with its reason, changeable
through the three levels of disclosure (FOUNDATION §3).
1. "Each measure may bend: a smooth curve with five knots at its own percentiles. Your 17,479
   people support the 71 terms this makes; 3,805 would be enough." Quiet name: "restricted cubic
   splines, knots by Harrell's rule (Harrell 2015); sample size by Riley et al. 2020". Changing it:
   "Straight lines change every estimate."
2. "Every measure is kept: none is dropped before fitting; the elastic net may set some to zero on
   its own." Quiet name: "no predictor selection (TRIPOD+AI 9a)".
3. "Ridge and the elastic net choose how hard to pull inside each training fold." Quiet name:
   "penalty tuned by cross-validation within each fold".
4. "Weight, height, body mass index and waist are all kept. Body mass index is weight over height
   squared, so the four carry nearly one quantity, and Results reads them as one group." Quiet
   name: "near-collinear predictors; grouped importance".
- For the record (collapsed, never counted): "Nutrients and total calories enter as recorded: under
  Predict there is no energy model to choose (ruling 5)." "Glucose stays on its own scale."
- Button: "Confirm all 4".

**Tapestry:** follows the pointed line; at rest, line 1.
1. Waist's spread over the development rows, with its five knots at 74.0, 88.0, 97.2, 106.9 and
   127.7 cm (Harrell's 5th, 27.5th, 50th, 72.5th and 95th percentiles), and one curve sketched
   through them as an illustration.
2. The 20 measures in a column, all lit.
3. A fold within a fold, drawn as a schematic: the training fold split again to choose the pull.
4. Weight against height for the development rows, shaded by body mass index, with "body mass index
   = weight / height²" (they agree to a correlation of 0.9998) and the four measures' correlations
   (weight with waist 0.89, body mass index with waist 0.90, weight with body mass index 0.88).
- Caption: "What each convention does to your measures. No glucose value is read."

**Manuscript** (TRIPOD+AI 12b, 9a, 12c): "Continuous predictors entered as restricted cubic splines
with five knots at Harrell's recommended percentiles, giving 71 model terms for 17,479 development
participants, above the 3,805 required for an expected shrinkage of at least 0.9 (Riley et al.
2020). No predictor selection preceded model fitting. The ridge and elastic-net penalties were
chosen by cross-validation within each training fold."

**Engine.** Serves the sweep (the quest log's Confirm lines `form`, `set_validation` and
`set_levers`; `sweep.py`), the curves (`methods/levers.RuleSplines`, `:139`), and Riley's sample
size (`models/sample_size.py`). Must newly serve **P15** (the curves as the line-drawing families'
default recipe when Riley's criteria hold) and **P16** (the body-measures noticing before Fit,
outcome-blind; today `models/linear.py:61` raises it only after the fit, on least squares). Conflict
**C8**: `set_selection` is a Decide under Predict; here it is line 2 of the sweep.

### M5 · Your plan

**Understands:** "It will be judged by how far its estimates miss, in mg/dL, beside a model that
estimates everyone at the average. Each survey cycle will be estimated by models built on the
other eight, so I will see whether it holds in a period it never saw. What Results will look at
(the diabetic range, my groups, calibration, what each of my choices added) is fixed now, before
any score. One person in five stays unseen until I name my final model." **Feels:** ownership, and
a little suspense.

**Decision:** how it is checked across periods. Recommended: each survey cycle in turn.

**Card**
- Kicker: "Models · Your plan"
- Heading: "Last, before you fit: should it be checked on survey cycles it never saw?"
- Set for you, one line above the options: "Judged by how far it misses, in mg/dL, beside a model
  that estimates everyone at the average; a big miss counts for more than a small one." Quiet name:
  "mean squared error, reported as the mean absolute and root mean squared errors (Gneiting 2011)".
- Options:
  - "Each survey cycle in turn, from the other eight" (Recommended): "Shows whether it holds in a
    period it never saw; the latest cycle is estimated from earlier ones only." Quiet name:
    "internal–external cross-validation by cycle (Steyerberg & Harrell 2016)".
  - "People from the same years, in five groups": "Mixes the years: how well it does for new people
    from these sixteen years." Quiet name: "5-fold cross-validation".
- The Recommended's reason, quietly under it: "Your data span sixteen years and the model is for a
  later visit; the share with a diagnosis on record rose from 27% to 44% over them."
- One quiet line: "Either way, the kinds of model are compared on 10 rounds of 5 groups, and
  choosing the best is corrected for being chosen."
- The card's foot holds no button (FOUNDATION §7).

**Tapestry:** the analysis flowchart.
- At its head, the timeline emblem.
- Boxes, left to right: **People** (21,849; 4,370 unseen until you name your final model; 17,479 to
  build and check). **What it reads** (20 measures: 4 before the visit, 16 at it, 2 of them from the
  draw; as curves, 71 terms; not read: the cycle, the respondent number, 6 processing flags).
  **The models** (least squares, ridge, elastic net, each with curves; beside them, a model that
  estimates everyone at the average). **How it is judged** (the nine cycles as a strip, each held
  out in turn once as it is drawn; the miss in mg/dL; "compared on 10 × 5 groups").
- "Fixed now: what Results will look at": the diabetic range (126 mg/dL or more) on its own; women
  and men, three age groups, with and without a diagnosis on record; each survey cycle; calibration;
  what each of your choices added (the lipids from the draw, the medicine answers, the curves).
- Fit at its end: "Fit · about 1 minute".
- Pointing at "People from the same years" redraws the strip as five mixed groups.
- Caption: "Your whole plan. Nothing is estimated yet."

**Manuscript** (three sentences arrive on Fit; TRIPOD+AI 12c, 12d, 12e, 14):
- "Performance was estimated by internal–external cross-validation across the nine survey cycles:
  each cycle was predicted by models developed on the other eight, and the cycles' results were
  summarized by random-effects meta-analysis with a 95% prediction interval for a new cycle."
- "Model families were compared on 10 repeats of 5-fold cross-validation by the corrected resampled
  t test, and the optimism of choosing the best was estimated by bootstrap bias-corrected
  cross-validation."
- "Performance was measured by the mean squared error and reported as the mean absolute and root
  mean squared errors in mg/dL beside a model predicting the development mean, with R² and the
  calibration intercept and slope. Declared before model fitting, performance was also assessed at
  fasting glucose of 126 mg/dL or more, within gender, age group and diagnosis on record, by survey
  cycle, and with each modeling choice undone."

**Engine.** Serves `validation.validation_plan` (`models/validation.py:359`), internal–external
validation (`:457`; the cycle folds in `stages/rows`), the comparison substrate
(`folds.comparison_folds`), BBC-CV (`selection.selection_optimism`, `:245`), the Fit estimate
(`FitLock.estimate_seconds`: 66.6 s), and the analysis plan (`GET /plan`, `declared_at` the press).
Must newly serve **P5** (the yardstick declared), **P12** (the evaluation plan fixed before Fit and
carried by the opening) and **P14** (the validation order reads the period column and the moment
of use; each cycle's signed error). Conflict **C9**: under Predict nothing is fixed at Fit, so a
check added after the scores are seen is not labeled.

## 5 · Results

### R1 · How good it is

**Understands:** "In survey cycles it never saw, my models missed fasting glucose by 16.9 mg/dL on
average; a model that estimates everyone at the average missed by 18.9. That is about a sixth of
the differences between people. Where it estimates a value, people average about that value; but
its estimates stay in a narrow band." **Feels:** honest clarity; a little sobered.

**Decision:** none (reading).

**Card**
- Kicker: "Results · How good it is"
- Finding (26 px): "In survey cycles it never saw, each model missed fasting glucose by about 17
  mg/dL on average; estimating everyone at the average missed by 19."
- Reading (17 px): "That is about a sixth of the differences between people. Where it estimates
  120, people average about 120: it is right on average at every level it reaches. But it reaches
  a narrow band: its estimates spread 14 mg/dL, against glucose's own 35."
- Quiet names: "mean absolute error 16.9 (95% CI 16.4 to 17.3), root mean squared error 32.4
  against 35.3, R² 0.16; calibration slope 0.98 (0.94 to 1.01), intercept −0.05 mg/dL (Van Calster
  et al. 2019)".
- Why?: "The average miss counts every mg/dL the same; the root of the squared miss counts a big
  miss more, and it is what the models are compared on. With glucose this skewed, 80% of the
  squared misses come from the 12% of people at 126 mg/dL or more: the next screen looks at them.
  An estimate of the average for people like this spreads less than the outcome, by about the
  square root of R² (0.40 × 35 ≈ 14)."
- Button: "Next: where it misses".

**Tapestry:** two panels.
1. "How far each missed, in mg/dL": a forest on one axis. Estimating everyone at the average 18.9
   (18.5 to 19.4); least squares 16.9 (16.5 to 17.4); ridge 16.9 (16.4 to 17.3); elastic net 16.8
   (16.4 to 17.2). A quiet second row of dots for the root of the squared miss: 35.3 against 32.4,
   32.4 and 32.4.
2. "Right on average, in a narrow band": the mean measured against the mean estimate in each tenth
   of ridge's estimates, on the diagonal (86.6 estimated, 91.4 measured, in the lowest tenth; 122.9
   and 121.3 in the ninth; 135.0 and 139.2 in the highest), with the smoothed calibration curve.
   Beside it, on the same vertical axis, glucose's own spread on the development rows with 126
   mg/dL marked and the band the estimates cover lit.
- Caption: "Out-of-cycle estimates of the 17,479 development participants: each cycle estimated by
  models built on the other eight. Glucose read after Fit, on these rows only; the 4,370 set aside
  stay unread."

**Manuscript:** nothing yet; the paper's Results report the held-out rows (R5).

**Engine.** Serves the fit's `cv` (RMSE, MAE, R², each with its interval), the `baseline` (its MSE,
1,249.0), `versus_baseline` (better; R² gain 0.16, 0.145 to 0.176), and `calibration` (intercept,
slope, curve, the ICI 1.7 mg/dL). Must newly serve, in **P4**, the no-predictor model's MAE and RMSE
beside each family's (today only its MSE, `Baseline.value`) and the calibration by tenth; in
**P18**, the "narrow band" verdict.

### R2 · Where it misses

**Understands:** "It misses most where it matters: of the 2,034 people at 126 mg/dL or more, it
places 776 there and misses them by 57 mg/dL on average, almost always too low. It does best for
young people and people with no diagnosis on record; above 58 or with a diagnosis on record, it is
no better than the average. Across cycles it ranks people about as well, but its level drifts: it
underestimates the latest cycle by 4.9 mg/dL." **Feels:** sobered, but armed: they know where not
to trust it.

**Decision:** none (reading).

**Card**
- Kicker: "Results · Where it misses"
- Heading: "Where it misses"
- Three lines, each a question and its verdict (§6):
  1. "Does it reach the diabetic range?" "Rarely: of 2,034 people at 126 mg/dL or more, it places
     776 there, and misses them by 57 mg/dL, too low." Quiet name: "error by range at the 126 mg/dL
     cut-off (American Diabetes Association)".
  2. "Is it as good for everyone it is for?" "Best from 18 to 36 (10.6 mg/dL) and with no diagnosis
     on record (12.4). Above 58 and with a diagnosis on record, no better than the average (22.0
     against 21.3; 24.6 against 23.5)." Quiet name: "subgroup performance (TRIPOD+AI 14, 23a)".
  3. "Does it hold in another period?" "It ranks people about as well in every cycle (R² 0.12 to
     0.21), but its level drifts: later cycles ran higher, and it underestimates the latest by 4.9
     mg/dL." Quiet name: "heterogeneity across clusters; calibration-in-the-large drift (TRIPOD+AI
     23b)".
- Why?: "It estimates the average glucose of people like each person. Most people with these
  measures are not in the diabetic range, so that average stays below it: for a group with a long
  upper tail, the average sits above most of the group and below its tail. The same pull makes its
  typical miss larger above 58 and with a diagnosis on record, where the tail is longer. The drift
  is the period's: glucose ran higher in later cycles, and the model does not read the cycle; a
  change of laboratory method may be part of it."
- Button: "Next: what drives it".

**Tapestry:** at rest, the first question.
- "Estimated against measured": the 17,479 out-of-cycle estimates against the measured values, both
  in mg/dL, with the 126 lines and the four parts counted: measured and estimated at 126 or more,
  776; measured at 126 or more but estimated below, 1,258; measured below but estimated at 126 or
  more, 1,100; both below, 14,345.
- Beside it, the signed miss by range: under 100 (9,020 people), 10.6 mg/dL too high; 100 to 125
  (6,425), 3.2 too high; 126 or more (2,034), 56.4 too low.
- Lead: "People at 126 or more average 179 mg/dL; it estimates them at 123."
- Pointing at question 2: the average miss by group beside estimating-at-the-average's: women 15.9
  and 19.3; men 17.9 and 18.5; 18 to 36, 10.6 and 16.5; 37 to 58, 18.2 and 19.1; 59 and over, 22.0
  and 21.3; a diagnosis on record, 24.6 and 23.5; none, 12.4 and 16.3; each with its share at 126 or
  more (from 2.2% of those 18 to 36 to 22.8% of those with a diagnosis on record).
- Pointing at question 3: the nine cycles' signed miss (+3.0, +3.8, +2.0, −2.5, +0.9, +0.2, +1.5,
  −4.0, −4.9 mg/dL) under their measured means (103.4 in 2001–2002 rising to 113.3 in 2017–2018),
  and the cycles' scores with the random-effects summary (root of the squared miss 31.7, 29.7 to
  33.5) and its 95% prediction interval for a new cycle (25.9 to 36.6).
- Caption: as R1's.

**Manuscript:** two Discussion drafts arrive, placed in R5 (TRIPOD+AI 25, 26): the tail, and the
drift.

**Engine.** Must newly serve **P4** (error by range and at the cut, on the headline's own folds and
on the held-out rows; the groups on the headline's folds, with the derived diagnosis group; each
cycle's signed error) and **P18** (the verdicts by §6's rules). Conflict **C4**: the engine scores
the groups on the comparison folds' first repeat (random 5-fold), not on the declared scheme.

### R3 · What drives it

**Understands:** "All three models read the same things most: HDL and triglycerides from the draw
move an estimate most, then age, body size and the medicine answers; yesterday's diet moves it
least. The lipids hardly change the average miss, but they are what let it reach high values:
without them it would place 657 of the 2,034 at 126 or more there, not 776." **Feels:** curious,
"so that's what it reads", and that their own choices mattered.

**Decision:** none (reading).

**Card**
- Kicker: "Results · What drives it"
- Heading: "What moves its estimates?"
- Finding (17 px): "In all three models, HDL and triglycerides from the draw move an estimate most
  (4.8 mg/dL on average), then age, body size and the medicine answers (3.6 to 3.7 each), then
  yesterday's diet (2.6), gender and blood pressure."
- What your choices added, three returns:
  - Because you said "At a visit with a fasting blood draw" (When it is used): "The lipids let it
    reach high values: 776 of the 2,034 at 126 or more placed there, 657 without them."
  - Because you said "Read them, a blank as 'not asked'" (The medicine questions): "776 against 662
    without them, and an average miss 0.2 mg/dL smaller."
  - Set for you, each measure may bend: "776 against 678 with straight lines."
- Quiet names: "grouped SHAP values (Lundberg & Lee 2017); accumulated local effects (Apley & Zhu
  2020); each choice undone and refit on the same folds (Lei et al. 2018)".
- Why?: "These describe how each model turns measures into estimates, not what changing a measure
  would do. Calories and the nutrients they are made of move together, as do the four body
  measures, so credit is given to each group; how one nutrient's share comes out depends on which
  model you ask. Three models as good as each other can read your data differently, the Rashomon
  effect (Breiman 2001): least squares gives height, weight and body mass index large shares that
  cancel, ridge gives them small ones, and their estimates are the same."
- Button: "Next: your final model".

**Tapestry:** three panels.
1. "What moves each model's estimates": each plain group's average push on one mg/dL axis, the
   three families side by side. Ridge: lipids 4.8, age 3.7, body size 3.7, medicine answers 3.6,
   diet 2.6, gender 2.1, blood pressure 1.9; least squares and the elastic net within a quarter of a mg/dL of these.
2. "How an estimate moves with one measure, the others as they are": accumulated local effects on
   shared axes for waist, triglycerides, HDL and age, the three families' curves overlaid (they
   nearly coincide). Waist: flat to about 92 cm, then rising to +12.5 mg/dL at 130 cm and +29 at
   161. Triglycerides: from −3.5 at 62 mg/dL to +7.4 at 304. HDL: from +6.0 at 23 mg/dL to −5.8 at
   97. Age: from −6.2 at 18 to +5.3 at 61, then down to −2.7 at 85. No curve where fewer than five
   people sit. Total calories behind "More angles".
3. "What your choices added": people at 126 or more placed there, 776 with every choice, 657
   without the lipids, 662 without the medicine answers, 678 with straight lines; the average miss
   moves by 0.2 mg/dL at most.
- Caption: "Described on the development rows: what the models do, not what changing a measure
  would do (Molnar et al. 2022)."

**Manuscript** (TRIPOD+AI 12c): "The fitted models were described by grouped SHAP values and by
accumulated local effects on shared axes, and the contribution of each modeling choice was
estimated by refitting without it on the same folds; these describe the models' predictions, not
causal effects."

**Engine.** Serves the `explain` stage (exact linear SHAP, `models/explain.py:117`; stability
across five refits, ρ 0.89 for ridge; accumulated local effects with their masks, `:626`; the
performance floor, `:1193`; each family's equation, `:947`). Must newly serve **P6** (each plain
group's net SHAP, the mean of |the sum of its inputs' SHAP values|, across families, and the curves
chosen from it) and **P10** (what each choice added). Conflict **C5**: the curves follow one
family's top three among the "exposure" columns (`_exposure_inputs`, `:1382`); on the straight-line
run they were total fat, total calories and waist, from least squares' unstable shares.

### R4 · Your final model, then the held-out rows

**Understands:** "The three are equally good; ridge is the simplest whose equation holds up, since
least squares cannot separate height, weight and body mass index. Naming it now, before the 4,370
open, makes its score the one my paper reports. My data's own notes go to the paper as limitations
or supplement lines." **Feels:** decisive: the moment of truth.

**Decision:** the final model. Recommended by ruling 4: ridge.

**Card**
- Kicker: "Results · Your final model"
- Heading: "Which one is your final model?"
- Return: "The three missed by the same amount: on the comparison folds no pair differs (their
  squared misses differ by less than 1, each interval across zero)."
- Options:
  - "Ridge" (Recommended): "As good as the best, and its equation is stable: it shares weight among
    measures that move together." Quiet name: "ridge regression (Hoerl & Kennard 1970)".
  - "Least squares": "As good, but it cannot tell height, weight and body mass index apart, so its
    equation would not hold up." Quiet name: "least squares; scaled condition number 3,969".
  - "Elastic net": "As good; how much it keeps of each measure changes from fold to fold." Quiet
    name: "the elastic net".
  - ⟨engine-filled: each tree family once fitted, with what the regression costs or gains against
    it (BBC-CV over the flexible families)⟩.
- One quiet line under the options: "Choosing the best of three flatters it by nothing measurable
  here." Quiet name: "bootstrap bias-corrected cross-validation (Tsamardinos et al. 2018): squared
  miss 1,039, 912 to 1,157".
- Before they open, two collapsed groups and one line:
  - "Noted for your paper · 3": "People treated for diabetes are in your table, and treatment lowers
    glucose." "No survey weights: it describes these 21,849 people, not the US population." "No
    race or ethnicity, income or education: their groups could not be checked." Each becomes a
    limitation in your Discussion.
  - "Done for you · 3", none of which moves an estimate: "Gender read as female and male." "The 501
    days outside 500 to 5,000 kcal are kept: the model will meet such days in use." "One day of
    diet, as at the visit: a supplement line."
  - Set for you: "No recalibration: its calibration slope is 0.98, and its interval (0.94 to 1.01)
    includes 1."
- Button: "Open the held-out rows". Under it: "4,370 people, unseen since Who's in. They open once;
  anything changed afterward is reported as decided after."

**Under the other use** (screening): the first noted line becomes "Act on it": "Leave out people
already diagnosed: join NHANES's diabetes questionnaire in Your data."

**Tapestry:** at rest, the three families' paired differences from the best on the comparison
folds, each crossing zero, and the held-out block, 4,370, still sealed. Pointing at "Least squares"
draws the reason: its four body measures' shares add to 14.8 mg/dL but cancel to 3.7, where ridge's
add to 9.0 and come to the same 3.7. Pointing at a noted line draws its evidence (the treated line:
22.8% of those with a diagnosis on record are at 126 or more). Caption: "Compared on 10 rounds of 5
groups of the development rows. The set-aside rows are still unread."

**Manuscript** (TRIPOD+AI 12c, 12f): "Ridge regression was declared the final model before the
held-out participants were analyzed: it was the simplest family not distinguishable from the best
on cross-validation whose coefficients were identified (in least squares, height, weight and body
mass index were nearly collinear). No recalibration was applied."

**Engine.** Serves the final-model refusal and its exits, each quoting its family's CV score
(`selection._a_final_model_is_declared`, `models/selection.py:657`), BBC-CV, the pairwise
`comparisons`, the near-singular concern (`models/linear.py:61`), `open_seal`, and the triage
(`GET /triage`; `sweep.triage`, `:644`). Must newly serve **P13** (the Recommended final model by
ruling 4, with its reason; one press confirms Results' sweep and the notes, then opens), **P2**
(W1-H: the open-noticings gate) and **P17** (Predict's notes and their dispositions). Conflicts
**C7** (today's triage under Predict) and **C8** (`set_updating` is a Decide).

### R5 · Your sentence, and "Put this in my paper"

**Understands:** "On the 4,370 people set aside from the start, ridge missed by 17.1 mg/dL on
average against 19.3 for estimating everyone at the average, and explained about a sixth of the
differences between people. My paper says that, then where it misses, then the period's drift; the
Discussion says what it is not for." **Feels:** proud: it is defensible, and honest about its
limits.

**Decision:** "Put this in my paper".

**Card**
- Kicker: "Results · Your sentence · opened at [the opening's time]"
- Finding (26 px): "On 4,370 people it never saw, your model missed fasting glucose by 17.1 mg/dL on
  average; estimating everyone at the average missed by 19.3."
- The Recommended wording, "Prediction performance, with where it misses", as two paragraphs at 17
  px, each labeled quietly and each at most 90 words (75 and 78 here):
  > Of 21,849 participants, 17,479 were used to develop the models and 4,370 were held out (Figure
  > 1). In the held-out participants, the final ridge model had a mean absolute error of 17.1 mg/dL
  > (95% CI 16.2 to 17.9) and a root mean squared error of 33.6 mg/dL (30.6 to 36.6), against 19.3
  > and 36.7 mg/dL for the development mean; R² was 0.16 (0.14 to 0.18) and the calibration slope
  > 0.98 (0.92 to 1.05) (Table 2).

  > As planned, in held-out participants with fasting glucose of 126 mg/dL or more (12.3%),
  > predictions were 57.0 mg/dL too low on average, and 36% were predicted at or above 126 mg/dL.
  > Errors were largest above age 58 (22.6 mg/dL) and with a diagnosis on record (25.6 mg/dL; Table
  > S2). Across survey cycles, each predicted by models developed on the other eight, R² ranged from
  > 0.12 to 0.21, and the latest cycle was underestimated by 4.9 mg/dL (Figure S3).
- "As planned" is earned by P12, which fixes the checks before Fit; without it the paragraph opens
  "In further analyses".
- Behind "Other wordings": "Write my own", which passes the same manuscript gate (every number traces
  to the record). No effect wording is offered under Predict: these are predictions and a
  description of the model.
- Button: "Put this in my paper".

**Tapestry:** Table 2 as it will print, and where each part goes.
- Table 2: the final model on the held-out rows (mean absolute error 17.1, 16.2 to 17.9; root mean
  squared error 33.6, 30.6 to 36.6; R² 0.16, 0.14 to 0.18; calibration slope 0.98, 0.92 to 1.05;
  intercept 0.75 mg/dL, −0.25 to 1.75), the development mean on the same rows (19.3, 18.4 to 20.2;
  36.7, 33.6 to 39.9), and, labeled secondary, the other two families on the held-out rows (least
  squares 17.1; elastic net 17.0) and all three out of cycle; one footnote line.
- Where each part goes: Results (the two paragraphs; Table 2; Figure 1, the participant flow); the
  Supplement (S1 the full equation, TRIPOD+AI 22; S2 the groups; S3 the cycles; S4 calibration; S5
  what drives it; S6 what each choice added); the Discussion (six drafts, below).
- The TRIPOD+AI checklist as it stands sits under For the record.

**Discussion drafts** (TRIPOD+AI 25, 26, 27c), each kept, edited or dropped in Write-up:
1. "The model is not suited to detecting fasting glucose in the diabetic range: it estimates the
   average for people like each person, which stays below that range for most."
2. "Fasting glucose ran higher in later survey cycles, and the model underestimated the latest;
   before use in a later period its level would need updating (Van Calster et al. 2019)."
3. "Participants treated for diabetes were included, and treatment lowers fasting glucose; the
   data held no record of a diabetes diagnosis."
4. "The analysis was unweighted and describes these participants, not the US population."
5. "Diet was recorded by one 24-hour recall, as at the intended visit; a model given a better diet
   measure would need recalibrating (Luijken et al. 2019)."
6. "Performance could not be examined by race or ethnicity, income or education, which the data
   did not record."

**After "Put this in my paper":** the rail opens with the two paragraphs under "Results · just
placed"; "Next: the exhibits" opens them in place, each at its usual place, and "Confirm all 8"
places them. Write-up is the next design.

**Manuscript:** the two paragraphs (Results), the six Discussion drafts, the full equation for the
Supplement (TRIPOD+AI 22, from the explanations' `architecture.equation`), and the usability
sentence (TRIPOD+AI 27a): "At use, a medicine question not asked enters as 'not asked'; the
development data held no missing values for any other measure, so the model was not evaluated with
any of them missing."

**Engine.** Serves the opened fit (`holdout`, `holdout_detail` with intervals and calibration,
`final_model`, `at_opening`), the export (methods in TRIPOD+AI sections, `export/methods.py:47`;
the checklist, today 4 items answered, 11 partly and 37 unanswered). Must newly serve **P19** (the
Results paragraphs and Discussion drafts, the methods in the register, and TRIPOD+AI rules for
items 6c, 9a, 9b, 10, 12d, 12f, 14, 15, 16, 20c, 22, 23b, 24 and 27a), **P8** (the sectioned
manuscript for the rail) and **P12** (to say "as planned").

## 6 · Results' interpretation order under Predict

Each verdict is computed; none is stronger than its evidence.

1. **How good it is.** The declared scheme's out-of-sample average miss (MAE) for each family and
   for the no-predictor model, in mg/dL with intervals; the root of the squared miss beside it,
   quietly; R² said as a share ("about a sixth" for 0.16: one over R², rounded).
2. **Calibration.** "Right on average at every level it reaches" when the intercept's interval
   includes 0, the slope's includes 1, and the ICI is under 2% of the outcome's mean (1.7 of 107
   here). Otherwise its direction: a slope below 1 is "its estimates are too spread out", above 1
   "not spread out enough". "A narrow band" when the estimates' standard deviation is under half
   the outcome's, said with both (14 and 35).
3. **The tail,** for an outcome with a clinical cut-off (a lens fact: fasting glucose 126 mg/dL,
   American Diabetes Association). "Rarely" when fewer than half of those at or above the cut are
   estimated there, "mostly" otherwise; the signed miss in the tail and its share of the squared
   misses.
4. **The groups** the intended use names. Each group's average miss beside the no-predictor
   model's in that group; "no better than the average" when the model's interval reaches the
   no-predictor model's estimate; best and worst by the average miss; each group's share in the
   tail beside it.
5. **The period.** The cycles' R² range, the random-effects summary and its prediction interval for
   a new cycle; "its level drifts" when the latest cycle's mean signed error has a 95% interval
   that excludes 0 (here −4.9 out of cycle, −4.7 on the held-out rows, standard error 1.6).
6. **What drives it.** By plain group, each group's net SHAP (the mean of |the sum of its inputs'
   SHAP values|), across families; "in all three models" when every family ranks the same group
   first. The Rashomon line is said when two families that perform alike give one group's inputs
   summed shares that differ by more than half while the group's net shares agree (body size:
   least squares 14.8, ridge 9.0, both netting 3.7).
7. **What each choice added.** The choice undone and refit on the same folds, said in people
   reached at the cut and in the average miss, each with its interval; a choice whose undoing moves
   neither beyond its interval is said as "made no difference here".
8. **Which to keep:** ruling 4.
9. **The wording.** Prediction performance; explanations "describe the model", never an effect.
   Paragraph 1 is the held-out evaluation; paragraph 2 is the checks fixed before Fit, on the
   held-out rows, with the cycle check from the development data; each at most 90 words; nothing
   decided after the opening. The Discussion drafts come from 3, 4 and 5 and the noted lines.

## 7 · The tapestry's pictures under Predict

| Question | The picture | View kind |
|---|---|---|
| Whom it is for | the people as one bar, by what the table records (a diagnosis on record, the groups) | comparison |
| The held-out rows | the participant flow with the unseen block; each cycle split the same way | row flow |
| When it is used | the visit as a timeline, each column at its moment; the period strip | comparison (new picture: the timeline) |
| The medicine answers | each question's answers as a bar, "not asked" named | comparison |
| Which kinds of model | what each kind can draw on one axis; cost on a time axis with the hold | curve |
| Set for you | the line's own evidence: knots on a spread; weight against height | distribution, relationship |
| Your plan | the analysis flowchart with the cycle strip and what Results will look at | lineage (the flowchart) |
| How good it is | the miss as a forest; calibration by tenth beside the outcome's spread | forest, calibration |
| Where it misses | estimated against measured with the cut lines; the groups; the cycles | relationship, forest |
| What drives it | grouped SHAP by family; accumulated local effects on shared axes; choices undone | table, curve, forest |
| Your final model | paired differences; the sealed block | forest |
| Your sentence | Table 2 and the placement map | table, page |

The timeline is the Predict counterpart of the Estimate journey's compared day: an outcome-free
picture of what a choice means, said once to be an illustration. From M2 to M5 it rides as a small
emblem at the tapestry's head.

## 8 · The manuscript rail under Predict

TRIPOD+AI's order (`export/methods.TRIPOD_SECTIONS`), the methods register, the codebook's names;
readings confirmations and the engine's mechanics stay in the record. The rail grows: entering
Models it holds the data, the participants, the outcome, the intended use (with its Introduction
draft), the missing-data sentence and the held-out draw; M1 adds the predictors (9a, 9b); M2 the
medicine answers (6c, 11); M3 the families (12c); M4 the curves, no selection and the tuning (12b);
Fit the validation, the comparison and the measures (12c, 12d, 12e); R3 the explanations; R4 the
final model and no recalibration (12f); R5 the Results paragraphs and the Discussion's six drafts.
The card's foot says what each confirmation added ("2 sentences added to Methods"), in the recorded
green, which opens the rail.

## 9 · Methods rulings proposed

Each is the design owner's, with its source; an independent adversarial check follows (the plan's
W1-I).

- **Ruling 1 · The medicine answers.** They are known at the moment of use, so M1 admits them. Their
  blanks mean "not asked" (the BPQ skip pattern), so they enter with three levels (taking, told but
  not taking, not asked): never filled with the most common answer, which would mark 15,552 people
  as taking blood-pressure medicine, and never complete cases, which would keep the 2,996 who
  answered both. Whether to read them follows the intended use. For estimation in everyone at the
  visit, read them, labeled as a marker of a diagnosis on record. For screening people not yet
  diagnosed, leave them out: in development data part of what they carry is a diabetes diagnosis
  already made, because cholesterol treatment has been recommended for adults with diabetes
  throughout these cycles (diabetes a coronary risk equivalent since NCEP ATP III 2001; statins for
  most adults with diabetes aged 40 to 75 since Stone et al. 2014; ADA Standards of Care §10). The
  model would then look better here than where it is used (a difference in case mix between
  development and use; Moons et al. 2012). Sources: the NHANES BPQ questionnaire (verified for
  2015–2016, to verify for the other eight cycles); Sperrin et al. 2020 and Sisk et al. 2023 on
  missing values in prediction, as the engine cites them.
- **Ruling 2 · The yardstick for a skewed outcome.** The mean squared error stays the score the families
  are compared and chosen on: every family on this shelf estimates the conditional mean, and squared
  error is a scoring function consistent for the mean (Gneiting 2011). The card says the average
  miss (MAE) in mg/dL, with the root of the squared miss quietly beside it and the no-predictor
  model always alongside. Because skew 4.6 puts 85% of the no-predictor model's squared misses in
  the 12% at 126 mg/dL or more, the tail is a standing check (the first question of beat R2),
  never hidden in the average. Two things are not done: choosing on the absolute error (consistent for the median,
  which no family here estimates), and a log scale (glucose's log is skewed too, 2.4: the tail is a
  group of people, not a scale, which is why `structural.scale_question` stays silent).
- **Ruling 3 · Validation across survey cycles.** For a model meant for a later visit, on data spanning
  sixteen years, the headline is internal–external cross-validation by survey cycle: each cycle
  estimated by models developed on the other eight, summarized by random-effects meta-analysis
  with a 95% prediction interval for a new cycle (Steyerberg & Harrell 2016; Riley et al. 2016;
  Collins et al. 2024, Box 4; TRIPOD+AI 12d, 23b). The latest cycle's own score comes from models
  developed on earlier cycles only, so it is said as the forward-in-time check. The families are
  still compared, and the choice corrected, on 10 × 5-fold cross-validation (the engine's MS6), and
  the random 20% stays the sealed test, which measures new people from the same years. The period's
  drift is reported, not corrected inside the development data: the held-out rows mix the same
  cycles, and recalibration belongs to the setting of use (Van Calster et al. 2019).
- **Ruling 4 · The final model.** Among the families whose paired difference from the best includes zero
  on the comparison folds (corrected resampled t; Nadeau & Bengio 2003), the simplest whose fit
  raised no concern its equation would carry into the paper (TRIPOD+AI 22 asks for the full model);
  the simplest of them all when every one raised one. Named before the held-out rows open, with
  BBC-CV's correction reported for the best (Tsamardinos et al. 2018). Here least squares' body-size
  terms are not identified (scaled condition number 3,969), and ridge, built for exactly that (Hoerl
  & Kennard 1970), is chosen. With a tree family fitted, the interpretable cost (BBC-CV over the
  flexible set) decides whether the regression's simplicity is worth its price.
- **Ruling 5 · Energy adjustment and the substitution under Predict: neither is asked.** The energy model
  decides what a nutrient's coefficient means, an Estimate question (Willett, Howe & Kushi 1997); a
  prediction model's nutrients enter as recorded beside total calories, said For the record (for
  least squares the standard and residual models span one column space and give the same
  predictions, Frisch–Waugh–Lovell). The substitution compares two imagined diets, an effect; under Predict its
  curve would describe the model under a change, which the floor forbids wording as an effect
  (FOUNDATION §8; Shmueli 2010). A person who wants the swap gets an Estimate track. The dietary
  energy noticing does not fire under Predict.
- **Ruling 6 · Curves by default for the families that draw lines.** Under Predict, each continuous
  predictor of least squares, ridge and the elastic net enters as a restricted cubic spline with
  knots by Harrell's rule when Riley's criteria hold for the terms it makes (Harrell 2015; Riley et
  al. 2019, 2020); straight lines stay one click away, and tree families get none. Prespecified
  flexibility, not a form chosen after seeing the scores. Here: 71 terms, 3,805 rows needed, 17,479
  available; undoing it places 98 fewer of the people at 126 or more in that range.
- **Ruling 7 · The period is never a predictor under a later moment of use.** A later visit falls in no
  level of the cycle, so the model cannot read it; the cycle is the validation's grouping (ruling 3).
- **Ruling 8 · What Results checks is fixed before Fit.** Under Predict nothing locks, but the yardstick,
  the scheme and the standing checks (the tail, the groups, the cycles, calibration, each choice
  undone) are declared with the plan at Fit, and the held-out rows are scored on exactly them, once;
  a check added later is labeled as added after the cross-validated scores were seen. This is what
  lets the paper say "as planned".
- **Ruling 9 · The groups follow the intended use** (TRIPOD+AI 14, 23a): gender, age in thirds and a
  diagnosis on record, the last derived from the skip pattern. The absent ones (race or ethnicity,
  income, education) are named as a limitation (TRIPOD+AI 3c, 26).

## 10 · Engine requirements

Sizes: S = 1, S–M = 2, M = 3, M–L = 5.5, L = 8. P1–P8 are the plan's (`WALKABLE_PLAN.md` §3),
confirmed and resized where the beats changed them; P9–P21 are new. "Phase 3" is C6a phase 3
(`decisions.py`, `quest.py`, `voice.py`, `service.py`); every item touching those files waits for
its merge.

| P | What | Status | Size | Wave | Files and conflicts |
|---|---|---|---|---|---|
| P1 | A skip-pattern blank: its own level recommended, never the most common answer; the "not asked → No" recode | NEW, in flight (W1-H) | 2 | W1 | `methods/missing.py`, `coach.py`, `repairs.py` |
| P2 | The open-noticings gate blocks the opening | NEW, in flight (W1-H) | 1 | W1 | `seal.py` |
| P3 | Moment of use: `q:moment_of_use`, each column's moment from the lens's codebook (NHANES interview, examination, laboratory) or asked, compiled to predictor roles; the 9a and 9b sentences | NEW | 3 | W2 | decisions, quest, voice (after phase 3); a lens table of moments |
| P4 | Where it misses: error by range and at a clinical cut-off on the declared scheme's own out-of-sample estimates and on the held-out rows; the groups on the same estimates; the derived group "a diagnosis on record"; the no-predictor model's MAE and RMSE; calibration by tenth; each cycle's signed error | NEW (resized from 2) | 3 | W2 | a new stage file and `stages/__init__`; not `evaluation.py` until MC-2b-3 |
| P5 | The yardstick declared: MSE compared and chosen on, MAE and RMSE said in mg/dL, the tail standing (ruling 2) | NEW | 2 | W2 | metrics declaration |
| P6 | Each plain group's net SHAP across families (mean of the summed SHAP's absolute value) and the curves chosen from it | PARTIAL (`explain.grouped`, `:459`) | 2 | W2 | `models/explain.py` (after phase 3) |
| P7 | Plain names for quest lines and NHANES columns | PARTIAL | 3 | W2 | quest, voice |
| P8 | Sectioned manuscript endpoint for the rail | NEW | 1 | W1 | routes, service (after phase 3) |
| P9 | Intended use for a numeric outcome: its population (everyone at the moment of use, or people not yet diagnosed), the copy "estimation", and the medicine recommendation read from it with its reason | NEW | 2 | W2 | decisions, quest, voice |
| P10 | What each choice added: the final family refit without a group or with the alternative recipe on the declared folds, paired; said in people reached at the cut and in the average miss | NEW (the Predict side of BE10) | 3 | W2 | a new stage file |
| P11 | Under Predict the energy-model and substitution questions and the energy noticing do not fire; nutrients are For the record (ruling 5) | NEW | 2 | W2 | `interview._not_applicable` (`:628`), the dietary pack's detector |
| P12 | The evaluation plan (yardstick, scheme, standing checks, groups) fixed at Fit, kept with `/plan`, carried by the opening record; a later check labeled (ruling 8) | PARTIAL (`/plan` records `declared_at`) | 3 | W2 | `plan_lock.py`, `seal.py` (the Predict counterpart of BE22) |
| P13 | The Recommended final model by ruling 4, with its reason; one press confirms Results' sweep and the notes, then opens | NEW | 2 | W2 | `models/selection.py`, sweep |
| P14 | The period column: never a predictor under a later moment of use; the validation order reads it and the moment of use (internal–external first); each cycle's signed error served | NEW | 2 | W2 | `models/validation.py`, `seal.py` |
| P15 | Each family's own recipe: the curves only for the families that draw lines, by default when Riley's criteria hold (ruling 6) | NEW (C6b's recipes) | 3 | C6b | `models/pipeline.py`, `methods/levers.py`, the shelf |
| P16 | The body-measures noticing before Fit, outcome-blind (the model matrix's condition number), placed in Models' sweep | NEW | 2 | can follow | `models/linear.py:61`, a noticing |
| P17 | Predict's notes and their dispositions: treated diabetes in the table (limitation; under screening, act on it), unweighted (limitation), absent groups (limitation), one day of recall at the moment of use (supplement line), implausible days kept (supplement line), gender's coding (done for you) | NEW | 3 | W2 | `sweep.recommend` (`:519`), `materiality.recommend` |
| P18 | Predict's verdicts by §6's rules: calibration words, the narrow band, the tail, "no better than the average", the drift, the Rashomon line | NEW | 3 | W2 | a new module beside the views |
| P19 | The Predict paper: the two Results paragraphs and the Discussion drafts (at most 90 words each, every number traced), the methods in the register, and TRIPOD+AI rules for 6c, 9a, 9b, 10, 12d, 12f, 14, 15, 16, 20c, 22, 23b, 24 and 27a | PARTIAL (export, checklist) | 5.5 | W2 | voice, `export/methods.py`, `export/checklists.py` (overlaps BE12 and BE14) |
| P20 | Under Predict the outcome-scale question reads the training rows, after the draw | NEW | 1 | can follow | `structural.scale_question` (`:126`), the task follow-up |
| P21 | A holdout of whole levels of a period column (the latest cycle) | NEW | 2 | can follow | `seal.py`, `stages/rows.py` |

- **Totals:** 50.5 units. W1 holds P1, P2 and P8 (4). W2's engine core is P3–P7, P9–P14, P17–P19
  (38.5). P15 rides with C6b's recipes (3); P16, P20 and P21 can follow (5).
- **Clear of phase 3:** P4 and P10 (new stage files), P15 (C6b), P21. Everything else touching
  `decisions.py`, `quest.py` or `voice.py` waits for it; P3 and P7 share `quest.py` and `voice.py`,
  so one owner takes both, as the plan says.
- **Shared with Estimate** (§12): P4's diagnosis group is BE19's marker; P4's cut-off is BE23's
  lens fact; P12 is BE22's Predict counterpart; P19 overlaps BE12 and BE14; P7 is BE13.

## 11 · Where the engine conflicts with the beats today

- **C1.** The dietary lens proposes the role "exposure" for the nutrients under Predict, so the
  energy-model and substitution questions fire (`interview._not_applicable`, `:628`), and Models
  under Predict carries the dietary energy noticing as an open Decide ("`fat_total` correlates 0.88
  with `kcal`…"). → P3, P11.
- **C2.** The missing-values question ranks "Single fill in each training fold" first; its fill for a
  category writes "taking" into all 15,552 and 17,204 "not asked". → P1 (W1-H).
- **C3.** `set_levers` applies its form to every family: tree families get spline columns, and the
  forest's estimate rises from about 2 hours to about 11. → P15.
- **C4.** The groups are scored on the comparison folds' first repeat (random 5-fold), not on the
  declared scheme (`stages/evaluation.py:323`), and so is the regression-with-splines benchmark. →
  P4.
- **C5.** The curves default to the best-scoring family's top three among the "exposure" columns
  (`explain._exposure_inputs`, `:1382`); under Predict that follows one family's unstable shares
  (the straight-line run drew total fat, total calories and waist). → P6.
- **C6.** The body measures' near-singularity is said only after Fit, as a concern on least squares
  (`models/linear.py:61`). → P16.
- **C7.** The triage under Predict lists gender's coding as "act on it" (an engine to-do, not a
  finding), the 501 implausible days as "act on it" though keeping them is right for prediction,
  and the energy noticing as "already answered" though its question did not apply. → P17.
- **C8.** `set_selection` is a Decide under Predict even when Riley's criteria hold, and
  `set_updating` is a Decide in Results though the slope's interval includes 1. → M4's sweep, P13.
- **C9.** Under Predict nothing is fixed at Fit: a check added after the cross-validated scores are
  seen is not labeled. → P12.
- **C10.** The outcome-scale question reads every row's glucose, the future held-out rows
  included, before the draw (`structural.scale_question`). → P20.
- **C11.** The methods are engine prose in the TRIPOD+AI sections (backticked names, "covariates",
  "risk estimation", the readings' sentences): the problem BE14 names under STROBE. → P19, P7.
- **C12.** Under Predict, Results counts no exhibit (the quest log reads 1 of 2 after the opening:
  the opening and the shrinkage question), and Write-up reads complete at 0 of 0. → P19 with C7a.

## 12 · What Predict shares with Estimate (one shell)

| Part | Estimate (`BEATS_MODELS_RESULTS.md`) | Predict (this file) | One component |
|---|---|---|---|
| Quest line, bar, card with tapestry, three levels, Decide · Confirm · For the record | the same | the same | the shell (W1-E) |
| The return | "Because you said" on the partner's card (M2) | M2 quotes Q; R3 hands back M1, M2 and M4 | one Return block: kicker, quote with its card, the so-sentence (BE1, BE2) |
| What a choice means, drawn | the compared day | the visit's timeline; the answers' bars | the comparison view kind |
| The emblem at the tapestry's head | the comparison | the timeline | one Emblem |
| The flowchart with Fit | sealed with the plan: what Results will check | fixed now: what Results will look at | one Plan view; BE22 and P12 feed it |
| The commitment | Fit locks the plan (SHA-256) | Fit fixes the evaluation plan; R4 opens the held-out rows | one Commitment line with two modes; "Open the held-out rows" is Predict's only extra primary action |
| What a reviewer will ask | R2's eight questions | R2's three questions | one Checks list (question, verdict, quiet name) |
| Pictures after Fit | forests, the curve, the tail | forests, calibration, curves, the density with cut lines | the view kinds of FOUNDATION §5 rule 9 |
| Wording and "Put this in my paper" | two paragraphs, at most 90 words, Discussion drafts | the same | one Wording card (BE12, P19) |
| Placement | exhibits, "Confirm all N" | the same | one Placement step |
| The diagnosis marker | the split by a diagnosis on record | the group "a diagnosis on record" and M2's recommendation | one derived marker (BE19, P4) |
| The clinical cut-off | the tail question at 126 mg/dL | the tail in R2 | one lens fact (BE23, P4) |

What differs: Estimate's finding is one estimate in the person's own comparison; Predict's is a miss
in mg/dL beside a baseline. Estimate decides its draw For the record; Predict draws it in Who's in
and opens it in R4. "What could fool you" is said three times in both.

## 13 · Where each beat reads the capture

| Beat | Capture (`predict-capture/`) |
|---|---|
| Q, W | `capture.before_fit.predictors_all_rows`; `.at_the_draw.seal_plan` (options, validation order); `.models_entry.split` |
| M1 | `capture.before_fit.state_at_end` (roles); the design (`matrix`) |
| M2 | `capture.before_fit.predictors_all_rows` (answers by level, asked share by cycle, the most common answer) |
| M3 | `capture.before_fit.shelf` (rank, inductive bias, cost, Riley's sample size); `straight-run-shelf.json` for the trees' cost |
| M4 | `capture.before_fit.shelf.sample_size`; outcome-free body-measure correlations and knots (computed from the development rows, no glucose) |
| M5 | `capture.before_fit.plan`; `.after_seal.quest_before_opening.fit` (the estimate, 66.6 s); `capture.timing` |
| R1 | `capture.after_seal.fit_before_opening.models[].cv`, `.calibration`, `.versus_baseline`; `extras.families.*.by_tenth`, `.prediction_sd` |
| R2 | `extras.families.ridge.by_band`, `.at_cut`; `extras.groups`; `extras.cycles`; the fit's `internal_external` |
| R3 | `extras.group_shap`; `capture.after_seal.explain_before_opening.curves`; `extras.added` and `extras.families["ridge:…"].at_cut` |
| R4 | `capture.final_rule`; `.after_seal.fit_before_opening.comparisons`, `.selection`; `.triage_before_opening` |
| R5 | `capture.after_seal.fit_opened` (held-out scores and calibration); `extras.held_out` (range, cut, groups, cycles); `.bundle` (methods, checklist) |

## 14 · Sources to verify into the registry

**Not yet in the registry** (`export/data/citations.json`), to verify with their DOIs: Gneiting
2011 (*J Am Stat Assoc* 106:746); Hoerl & Kennard 1970 (*Technometrics* 12:55); Zou & Hastie 2005
(*J R Stat Soc B* 67:301); Breiman 2001, "Statistical modeling: the two cultures" (*Stat Sci*
16:199; the registry holds his "Random forests" of the same year); Lei et al. 2018 (*J Am Stat
Assoc* 113:1094); Moons et al. 2012 (*Heart* 98:691); Shmueli 2010 (*Stat Sci* 25:289); NCEP ATP
III 2001 (*JAMA* 285:2486); Stone et al. 2014 (*Circulation* 129:S1); the American Diabetes
Association's Standards of Care (§2, the fasting cut-off of 126 mg/dL; §10, statins for adults with
diabetes); Riley et al. 2019 (*Stat Med* 38:1262), Riley et al. 2020 (*BMJ* 368:m441) and Sperrin
et al. 2020 (*J Clin Epidemiol* 125:183), which the engine cites but the registry lacks.

**Already in the registry:** Steyerberg & Harrell 2016, Riley et al. 2016, Collins et al. 2024, Van
Calster et al. 2019, Luijken et al. 2019, Vickers & Elkin 2006, Lundberg & Lee 2017, Apley & Zhu
2020, Molnar et al. 2022, Tsamardinos et al. 2018, Nadeau & Bengio 2003, Harrell 2015, Willett,
Howe & Kushi 1997, Sisk et al. 2023.

**NHANES facts to verify:** the BPQ skip rules for `meds_hbp` and `meds_chol` in all nine cycles
(2015–2016 is verified, `BEATS_MODELS_RESULTS.md` E16); the laboratory's fasting glucose methods
across cycles (the level steps up in 2007–2008 and again from 2015–2016); the fasting subsample's
own weights, which this unweighted table does not carry.

## 15 · Open for Nolan

1. **What his glucose model is for.** The beats walk "Any adult it is used on, diagnosed or not".
   If he means screening for high glucose nobody knows about, M2 recommends leaving the medicine
   answers out, and the treated-diabetes note becomes "act on it": join NHANES's diabetes
   questionnaire in Your data so that people already diagnosed can be left out.
2. **The tree families.** Fitting them tuned takes hours on his machine: as a job while he is away,
   or in CI. Until then their numbers on these screens are ⟨engine-filled⟩.

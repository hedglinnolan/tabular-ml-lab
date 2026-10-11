# Beats: Predict, Models and Results (NHANES fasting glucose)

Written 2026-10-10 by the design owner, before any screen is built (FOUNDATION §0, ruling 6), and
revised the same day on the independent critique (Round 1, below; §16 answers it point by point).
It is the contract W2's build ("Predict, drawn") is held to: what the person understands and feels
at each step from entering Models to "Put this in my paper", what the card says, what the tapestry
draws, what the manuscript gains, and what the engine must serve. Its form is the Estimate
journey's (`BEATS_MODELS_RESULTS.md`, round 2), and it is written against what that journey's three
critiques caught (`QUEST_LOG_CRITIQUE_2026-10-10.md`, and the two rounds in
`BEATS_MODELS_RESULTS.md` §12–§13): asking what is already settled, drawing mechanics instead of
meaning, a sign instead of a finding, checks done after the fact reported as planned, and density.

**The journey.** NHANES: 21,849 adults with a fasting glucose, nine survey cycles (2001–2002 to
2017–2018). The outcome is fasting glucose in mg/dL; the goal is Predict. It is Nolan's own
project, and two of its answers are his rulings of 2026-10-10:
- the model is used **at a visit with a fasting blood draw**, so it may read HDL cholesterol and
  triglycerides from the same draw, and nothing known only after the visit;
- whether it reads the **medicine answers** is recommended by the app from the intended use the
  person chooses.

This walk takes the intended use "Researchers, for everyone in a study that measured these but
not glucose" (beat Q): estimation in everyone, diagnosed or not, in research data whose fasting
draw did not measure glucose. Every beat says what changes under the other use: clinicians, for
patients not known to have diabetes, at a draw that left glucose out.

**Where the numbers come from.** A light live run; no number here is typed.
- `predict-capture/capture.py` applies beat D's answers to the table (the engine does not serve
  them yet: P22–P24), drives the real server in process (`server_drive.local_server`, a fresh
  `TURBOTAB_HOME` under the scratchpad, two workers, two threads), answers the journey, keeps the
  latest cycle sealed, presses Fit (`Drive.artifact` presses and releases), and names the final
  model by ruling 4 before anything reads the latest cycle. It took 4 min 29 s at `80c1e81e` with
  this round's script: the fit 3 min 40 s after the press, against the engine's estimate of 94 s.
- **The seal is emulated.** The engine cannot hold out a whole survey cycle (it refuses
  `set_temporal` when each person appears once: C13, P21). So the run keeps 2017–2018 out of the
  analysis by a rule, draws nothing at random, and `extras.py --open` opens 2017–2018 once,
  with the final model and the level the run declared; it refuses to run twice.
- `capture.py --forms none --shelf-only` stops at the families question: the straight-line shelf,
  for the tree families' costs (they get no curves: P15). Nothing is fitted.
- `extras.py` computes, with the engine's own functions on the same eight cycle folds, the Results
  numbers the engine does not serve yet (P4, P6, P10, P18, P25, P27). Its pooled scores reproduce
  the engine's cross-validated MSE exactly (1,051.830 for ridge). Each script writes its JSON beside
  it (`capture.json`, `extras.json`, `opened.json`, `straight-shelf.json`).
- **What reads glucose.** The Models beats quote no number that reads glucose. The engine's shelf
  does read glucose's mean and spread on the development rows, for one sample-size criterion that
  never binds here (C17, said in M3). Every Results beat says which rows it read: the out-of-cycle
  estimates of the 19,579 development participants (2001–2016), or the 2,270 in 2017–2018 after
  the opening.
- **No tuned trees.** A tuned shelf on these rows takes hours (the straight-line shelf's own
  estimates: random forest about 2 hours, XGBoost about 18 minutes; boosted trees are untuned
  until C6a phase 3 lands). Wherever a tree family's number would appear, it is ⟨engine-filled⟩;
  wherever a joined file's would, ⟨after the join⟩.
- **The history, declared.** This design was first captured three times on 2026-10-10, each with a
  random one-in-five seal drawn at seed 0, and the critique's recheck read those runs. So every
  cycle's glucose, the latest included, was read before this capture, and two rulings of this
  round (the latest cycle sealed whole, and the level for a later visit) were adopted after the
  critique computed a level update on 2017–2018 from those runs. This capture opens 2017–2018 once
  in its own project, but it is not a first look at those values: its numbers illustrate the
  design, and Nolan's own project opens its seal once. Ruling 6 (curves) was likewise adopted after
  the first run's cross-validated benchmark. The capture's three attempts are in the scratchpad:
  the first stopped on a script error after the fit, the second shared the machine (so its timing
  is not cited); none opened anything, and all three gave the same numbers.

**Names.** The cards use the NHANES variable labels in plain words (P7, design-time): `hdl` is HDL
cholesterol, `bp_sys` systolic blood pressure, `meds_hbp` "now taking prescribed medicine for high
blood pressure", and so on. The group the original beats called "a diagnosis on record" is now
said as what it is: people told to take a blood-pressure or cholesterol medicine (they were asked
a medicine question).

## Round 1: what the critique changed, and why

The critique found Models had real magic (the visit's timeline, the medicine answers' return) and
the numbers traced, but Results judged by three yardsticks after declaring one, read regression to
the mean as a bias, left the data's own warnings until after the scores, and let a random seal
argue against a level update the evidence asked for. Round 1:

1. **One yardstick, end to end.** Every verdict, return and choice is on the squared miss, said as
   the share of the differences explained and as the root mean squared error in mg/dL, with paired
   intervals; the average miss stays on the card as description only. The lipids become "the most
   valuable thing it reads" (R² 0.155 to 0.125 without them); the groups' verdict becomes "better
   than the average in every group".
2. **"Where it misses" reads ranking and calibration at the cut.** The diabetic range is judged by
   how well the estimate ranks it (c 0.82) and whether its high estimates are right (4 in 10 of
   those estimated at 126 mg/dL or more are there); the 57 mg/dL miss among people picked by their
   measured value moves to Why? as a lesson in regression to the mean (a perfectly calibrated
   version misses them by 56 too). No count at the cut is a verdict anywhere.
3. **The intended use names its care pathway, and the moment fits it.** Q's options say who uses
   the estimate, for whom, at a draw that did not measure glucose; M1 returns it. A new line in
   Your data (beat J) recommends joining NHANES's diabetes and demographics files under either use:
   a diagnosis is the decisive predictor for everyone, and the population's filter for screening.
   Diet follows the setting: research keeps the recall, a clinic has none (PROBAST 2.3).
4. **What your data raised comes before the draw** (beat D, First look): the 2,444 people with a
   value filled in before the table was made, the 119 diastolic zeros, the age top-codes and the
   glucose laboratory's changes with NHANES's published equations. Results keeps only limitations,
   and the opening is a press of its own.
5. **The seal matches the use, and Results returns to the plan.** The latest cycle is sealed whole
   (P21 promoted to W2); the plan declares, before Fit, that its level follows the latest cycle if
   the cycles shift (M5's one decision); R4 returns it (2015–2016 ran 4.0 mg/dL above its
   estimates, so its level rises 3.4), and the opening tests it: in 2017–2018 the level held
   (−0.5 mg/dL against −3.8 without the update), and the misses (33 mg/dL) fell inside what the
   other cycles predicted (24 to 42). The cycles' summary is on the log scale with the
   Hartung–Knapp–Sidik–Jonkman interval.
6. **Density.** "No glucose value is read" is said once (M1); R1–R3 cards carry at most six
   numbers each; M5's validation question becomes a "because" line; M3 and M4 draw what a choice
   means; R4 names the final model on one press and opens on another.
7. **The numbers.** 73 model columns (not 71) and Riley's minimum for them (3,916, the shrinkage
   criterion); the fit's real time beside the engine's estimate; BBC-CV's correction said as
   resampling noise; "poorly determined" for an ill-conditioned least squares; a stated margin for
   "equally good"; the history of openings declared above.

§16 answers the critique point by point, with what was not followed and why.

## 1 · The arc

| | Beat | Stage | One decision | What the person feels |
|---|---|---|---|---|
| Q | Whom it is for | Your question | the intended use, with its care pathway: "Researchers, for everyone in a study that measured these but not glucose" | purposeful: the model has a job |
| J | What your use needs | Your data | join NHANES's diabetes and demographics files (Recommended); this walk goes on without them | guided |
| D | What your data raised | First look | the 2,444 filled values: read as missing (Recommended); three set for you | caught, and protected |
| W | The held-out rows | Who's in | the latest survey cycle, sealed whole (Recommended) | protected |
| M1 | When it is used | Models | the moment of use: "At a visit with a fasting blood draw" | grounded: "that's the moment I mean" |
| M2 | The medicine answers | Models | read them, each blank as "not asked" (Recommended from Q) | caught, then trusted |
| M3 | Which kinds of model | Models | the three that print an equation, in about 2 minutes | in control of the cost |
| M4 | Set for you | Models | Confirm all 4 | respected, quick |
| M5 | Your plan | Models | its level follows the latest cycle if the cycles shift (Recommended), then Fit | ownership; a little suspense |
| R1 | How good it is | Results | none: reading | honest clarity |
| R2 | Where it misses | Results | none: reading | sobered, but armed |
| R3 | What drives it | Results | none: reading | curious: "so that's what it reads" |
| R4 | Your final model | Results | the final model (ridge, Recommended); then, on its own press, open the latest cycle | decisive: the moment of truth |
| R5 | Your sentence | Results | "Put this in my paper" | proud: defensible, and honest about its limits |

Five Models screens and five Results screens. Q, J, D and W are not new screens: they are Your
question's intended-use card, a line Q reopens in Your data, First look's noticings card and Who's
in's draw, written here because Models stands on them.

**The arc builds, then rests.** M1 draws the visit as a timeline; from M2 on it rides as a small
emblem at the tapestry's head, each beat lighting what it settles. M5 draws the whole plan with the
eight development cycles and the sealed ninth. R1 turns the timeline's last mark, glucose, into the
miss. R3 hands three answers back: the lipids M1 let in, the medicine answers M2 kept, the curves M4
set. R4 hands back M5's level, and the opening tests the plan's two expectations.

**"What could fool you" is said three times, where each belongs:** in D, as the data's own
warnings (values filled before the table, impossible zeros, a laboratory that changed); in M1, as
what the model may not read (anything after the visit, and the period); in R2, as where it misses
(the diabetic range, the groups, the period's level).

## 2 · How a beat is written

As `BEATS_MODELS_RESULTS.md` §2: what the person understands and feels; the one decision; the card
in final copy (the heading, the return sentence, the options with their one-line consequences, the
quiet technical name on the option's top edge, the button, what sits behind "Why?"); the tapestry
(what it draws, from which data, what pointing changes, its caption); the sentence the manuscript
gains, in the methods register and TRIPOD+AI's order; and the engine (what it serves, what it must
newly serve, what conflicts). One type family in five sizes; about 120 words on a card and 250 on a
screen, as checks; nothing below the fold at 1280 × 800. The card says; the tapestry draws.

Three additions for Predict:
- **A Models beat never quotes a number that reads glucose.** A Results beat says which rows it
  read.
- **One yardstick** (ruling 2). Every verdict is on the squared miss, said as the share of the
  differences between people explained (R²: "about a sixth" for 0.155) and as the root mean squared
  error in mg/dL ("its misses came to 32 mg/dL"), each beside the model that estimates everyone at
  the average, with paired intervals. The average miss (the mean absolute error) is description,
  in Why? or a quiet name, never a verdict.
- **At most six numbers on a card**; the rest are drawn.

## 3 · Before Models: the answers it stands on

### Q · Whom it is for (Your question)

**Understands:** "My model is for researchers: it estimates fasting glucose for everyone in a study
that measured these things, a fasting lipid panel included, but not glucose, diagnosed or not. Its
accuracy will be reported for women and men, three age groups, and people told or not told to take
a blood-pressure or cholesterol medicine." **Feels:** purposeful.

**Decision:** the intended use, with its care pathway (TRIPOD+AI 3b). Nothing in the data says what
a model is for, so no option is Recommended.

**Card** (Your question's intended-use card; W1-F builds today's version, this is its copy)
- Kicker: "Your question · What it is for"
- Heading: "Who will use its estimate, and for whom?"
- Options:
  - "Researchers, for everyone in a study that measured these but not glucose": "It fills in the
    glucose a study did not measure, for anyone in it, diagnosed or not; your paper reports how
    close it comes." Quiet name: "intended use: estimating an unmeasured outcome in research data".
  - "Clinicians, for patients not known to have diabetes, at a draw that left glucose out": "It
    estimates glucose for people whose draw measured the lipids only; people already diagnosed are
    not its population." Quiet name: "intended use: estimation to inform testing for undiagnosed
    hyperglycemia".
  - "A yes-or-no call: whom to test", labeled "Not with this outcome": "A call needs a yes-or-no
    outcome: fasting glucose of 126 mg/dL or more." Its exit: "Change the outcome". Quiet name:
    "decision support (decision curve analysis, Vickers & Elkin 2006)".
- Set for you, beneath the options: "Its accuracy is reported for women and men, three age groups,
  and people told or not told to take a blood-pressure or cholesterol medicine." Quiet name:
  "subgroup performance (TRIPOD+AI 14, 23a)".
- Why?: "What the model is for decides whom it learns from, which answers it may read, and how it is
  judged. A model used on everyone may read what a diagnosis leaves behind; a model for people not
  yet diagnosed must learn from people like them."
- Button: Continue.

**Tapestry:** who the model is for, from the table's own answers (outcome-free). One bar of the
21,849 people: 7,946 told to take a blood-pressure or cholesterol medicine (they were asked a
medicine question, which NHANES asks only of people told to take it) and 13,903 not.
- Pointing at the researchers' option lights the whole bar, and one line beneath: "What this use
  reads that your table lacks: whether each person has been told they have diabetes, the strongest
  thing known at the visit (NHANES asks it in every cycle)."
- Pointing at the clinicians' option leaves the bar gray: "Who already knows they have diabetes is
  not in your table, so the model would learn from people already diagnosed and treated."
- Pointing at the groups line splits the bar: women 11,195 and men 10,654; on the development rows,
  18 to 36 (6,696), 37 to 58 (6,519), 59 and over (6,364); told to take a medicine or not. One
  quiet line: "Race and ethnicity, income and education are not in your table; NHANES records
  them in its demographics file."
- Caption: "Your table's people, by what it records."

**Manuscript.** Introduction (TRIPOD+AI 3b), drafted for the author: "The model is intended for
researchers, to estimate fasting plasma glucose in adults whose study measured these predictors,
including a fasting lipid panel, but not glucose, whether or not they have diabetes." Methods
(TRIPOD+AI 14): "Performance was reported overall and by gender, age group, and whether
participants had been told to take medicine for high blood pressure or high cholesterol." With J
declined (as in this walk), the Discussion's last draft says which groups could not be examined.

**Engine.** Serves `decisions.SetIntendedUse` (`decisions.py:1771`); for a numeric outcome only
`risk_estimation` is accepted (`decision_curve._intended_use_fits`, `:473`). The groups are scored
in `stages/evaluation.py` (`:323`; age in thirds by `subgroup_labels`). Must newly serve **P9** (the
use's pathway and population, and the copy "estimation", never "risk estimation" for a value) and,
in **P4**, the derived group "told to take a blood-pressure or cholesterol medicine" from the skip
pattern (shared with BE19).

### J · What your use needs (Your data, reopened by Q)

**Understands:** "For everyone, diagnosed or not, the strongest thing known at the visit is whether
someone has been told they have diabetes, and my table doesn't have it; NHANES does, in every
cycle, beside race and ethnicity, income and education." **Feels:** guided: the app knows NHANES.

**Decision:** join the files, or go on without them. Recommended: join them. **In this walk: "Go on
without them"**, because the capture's folder holds the export alone; every number below is for
the table as it is, and the files' own numbers are ⟨after the join⟩.

**Card**
- Kicker: "Your data · What your use needs"
- Heading: "Should the model know who has been told they have diabetes?"
- Return, the lede: "Because you said" over "Researchers, for everyone in a study that measured
  these but not glucose" (Your question · What it is for), then "For everyone, diagnosed or not, the
  most informative answer at the visit is a diagnosis of diabetes. Your table does not have it;
  NHANES asks it in every cycle."
- Options:
  - "Join NHANES's diabetes and demographics files" (Recommended): "Adds a diagnosis of diabetes and
    its treatment as predictors, and race and ethnicity, income and education as groups to check."
    Quiet name: "the Diabetes (DIQ) and Demographics (DEMO) files, joined on the respondent number".
  - "Go on without them": "Your paper says the model does not know who has diabetes, and which
    groups could not be checked."
- Why?: "A diagnosis is known before the draw and changes glucose (its treatment lowers it), so for
  a model of everyone it is the strongest predictor at the visit. For the clinicians' use it
  decides who the model is for instead: people already diagnosed are left out."
- Button: Continue.

**Under the other use** (clinicians): the Recommended reads "Join the diabetes file first", and
"Go on without it" is labeled "Not for this use": "Without it, people already diagnosed stay in the
data your model learns from."

**Tapestry:** the Q bar with a third mark, drawn as unknown: "told they have diabetes: not in your
table". Pointing at "Join" draws the two files beside the table on the respondent number, with what
each adds. Caption: "What your table records about diagnoses and groups, and what it does not."

**Manuscript:** with the join, the predictors and the groups (TRIPOD+AI 9a, 14); without it, the
Discussion drafts 4 and 7 (R5).

**Engine.** The join exists in Your data (the client's `joins` hooks, unused; WALKABLE_PLAN §1).
Must newly serve **P28** (the NHANES lens names what an intended use needs that the table lacks,
DIQ and DEMO by cycle; the line in Your data with its Recommended; the derived predictors and
groups).

### D · What your data raised (First look)

**Understands:** "One thing to decide: 2,444 people have a body measure or blood pressure filled in
before my table was made, by a method nobody recorded, some impossibly (a waist of 323 cm). The app
reads them as missing and fills them again in each training fold, without glucose. Three
conventions are set for me: glucose's laboratory changes are bridged by NHANES's own equations,
diastolic zeros are missing, and 80 and over is one age everywhere. Two things go to my paper."
**Feels:** caught, and protected.

**Decision:** the filled values (the one decision on the card, FOUNDATION §0 ruling 5). The card
comes before the draw, since one of its conventions rewrites the outcome: `seal.py` refuses an
answer that rewrites the outcome once rows are drawn, and the draw waits for such a repair.

**Card**
- Kicker: "First look · What your data raised"
- "Decide before the draw": "2,444 people (11%) have a measure filled in before your table was
  made: 1,876 blood pressures, 658 waists, 306 body mass indexes, 248 heights and 235 weights. How
  they were filled is not recorded, and 13 filled waists are larger than any measured one (176 cm),
  up to 323 cm." Options:
  - "Read them as missing" (Recommended): "Each is filled again in each training fold, from the
    other people's values and never from glucose." Quiet name: "single imputation within training
    folds, without the outcome (Sisk et al. 2023)".
  - "Keep them: I know how they were filled": "Say how; if glucose was used, the model has read its
    own outcome." Quiet name: "the imputation method (TRIPOD+AI 11)".
  - "Leave those people out": "2,444 fewer people, and the model will meet such people in use."
    Quiet name: "complete cases".
- "Set for you · 3" (collapsed; each changeable through the three levels of disclosure):
  - "Glucose's earlier instruments are bridged to the ones that followed by NHANES's own equations;
    none exists for 2013's move to a new laboratory." Quiet name: "outcome harmonization across
    survey cycles (NHANES laboratory documentation, GLU_D, GLU_E, GLU_I)".
  - "119 diastolic pressures recorded as 0 mmHg are read as missing." Quiet name: "a SAS transport
    zero, 5.4 × 10⁻⁷⁹, read per column".
  - "80 and over is one age in every cycle; 2001–2006 recorded 85 and over as 85." Quiet name:
    "harmonized top-coding".
- "Noted for your paper · 2" (collapsed): "No survey weights: it describes these 21,849 people, not
  the US population." "No diabetes diagnosis, race and ethnicity, income or education: not joined
  (Your data)."
- For the record (collapsed, never counted): "Gender read as female and male." "The 501 days
  outside 500 to 5,000 kcal are kept: the model will meet such days in use." "One day of diet per
  person, as in the studies it is for."
- Button: "Confirm all 6", enabled once the filled values are decided ("Decide the filled values
  first").

**Tapestry:** the view follows the pointed line; at rest, the filled values.
- The filled values: each measure's spread with its filled values in the quiet color; waist's 13
  beyond the largest measured waist drawn past the axis ("up to 323 cm"); body mass index against
  weight over height squared, measured people on the line (within 0.05) and filled ones off it (up
  to 14.3 apart). Pointing at "Read them as missing" turns the filled marks into gaps: "filled in
  each training fold".
- The laboratory: the nine cycles as a strip under five laboratory setups (Missouri, Cobas Mira,
  2001–2004; Minnesota, Hitachi 911, 2005–2006; Minnesota, Modular P, 2007–2012; Missouri, Cobas
  C501, 2013–2014; Missouri, Cobas C311, 2015–2018), NHANES's equations as arrows between them
  (× 0.9815 + 3.57; + 1.15; × 1.023 − 0.51), and "no published equation" at 2013. Outcome-free:
  the setups and the equations, never a glucose value.
- The zeros: the diastolic pressures' spread with the 119 at 0 marked.
- The ages: the age axis by cycle, 85 marked in 2001–2006 (198 people) and 80 from 2007.
- Caption: "Each line's own evidence, from your table's columns and NHANES's documentation."

**Manuscript** (on Confirm all 6; TRIPOD+AI 7, 8a, 11):
- "Body measurements and blood pressures imputed before the analysis data set was assembled (2,444
  participants, 11%), by an unrecorded method, were set to missing; diastolic blood pressures
  recorded as 0 mmHg (n = 119) were set to missing. Missing predictor values were imputed within
  each training fold by the median of the training data, without the outcome. Age was top-coded at
  80 years in every cycle."
- "Fasting plasma glucose measured on earlier instruments was calibrated to the instrument that
  followed with the equations NHANES publishes (2001–2006 and 2013–2014); NHANES publishes none for
  the change of laboratory between 2011–2012 and 2013–2014."

**Engine.** Serves the `imputed_*` columns as flags, the SAS-zero repair (`repairs.SAS_ZERO`,
`:107`, one option: zero), the missing-values strategy (`pipeline.py:656`, the in-fold median),
and the triage (`GET /triage`). Must newly serve **P22** (a flag linked to its measure; "read as
missing" Recommended when the method is not recorded), **P23** (= BE24, a per-column SAS-zero
reading), **P24** (the NHANES lens's measurement facts, applied before the draw: glucose's
laboratories by cycle with the published equations, the age top-codes) and **P17** (Predict's
notes and their dispositions, here, before the draw). Conflicts **C7**, **C18**, **C19**.

### W · The held-out rows (Who's in)

**Understands:** "The latest survey cycle, 2017–2018, stays sealed until I name my final model. It
is scored as a later visit would be, by a model built only on earlier cycles, and its score is the
one my paper reports." **Feels:** protected.

**Decision:** which people stay unseen. Recommended: the latest cycle, whole.

**Card** (Who's in's last Decide under Predict, FOUNDATION §3)
- Heading: "Which people should stay unseen until the end?"
- Options, in the seal plan's order:
  - "The latest survey cycle, whole: 2,270 people" (Recommended: "Your model will be used after
    2018, and the latest cycle is the nearest thing to that"): "Scored as a later visit would be:
    by a model built on 2001–2016 only." Quiet name: "temporal validation (TRIPOD+AI 12a)".
  - "One in five, at random: 4,370 people": "New people from the same sixteen years; a later period
    is checked only by cross-validation." Quiet name: "a random held-out set".
  - "None: cross-validation only": "Everyone trains and is scored in turn; no untouched final
    score."
- Why?: "They protect your result from your own choices: everything you try in Models and Results
  is judged without them, and they open once, after you name your final model (Cawley & Talbot
  2010). Held out by period, they also show what time does to the model."
- Button: Continue.

**Tapestry:** the participant flow, outcome-free: 21,849 people, then 2,270 sealed (the 2017–2018
cycle, one quiet block at the strip's end, "unseen until you name your final model"), then 19,579
in eight cycles to build and check the models. Pointing at "One in five, at random" redraws the
block as 4,370 people spread across all nine cycles. Caption: "Their glucose values stay unread
until you open them."

**Manuscript** (TRIPOD+AI 12a): "Participants in the latest survey cycle (2017–2018, n = 2,270)
were set aside before model development and used once, to evaluate the final model; the 19,579
participants in 2001–2016 were used for development and internal–external validation."

**Engine.** Serves `seal.plan` (`seal.py:850`) and `split_offer` (`:825`), the draw
(`stages/rows.draw_split`, which already draws a chronological holdout and cycle folds together),
and the sealed scores (`seal.serve_fit`). Must newly serve **P21** (a holdout of whole levels of a
period column, Recommended under Predict when the table has one, and its opening). Conflict
**C13**: `set_temporal` is refused when each person appears once ("temporal prediction does not
arise"), so no seal by period can be drawn today; the capture emulates it.

## 4 · Models

### M1 · When it is used

**Understands:** "The model reads the 20 things known at a visit with a fasting draw: who the person
is, what they have been told about blood pressure and cholesterol, yesterday's diet, body measures,
blood pressure, and HDL and triglycerides from the draw. The draw it reads did not measure glucose.
It reads nothing made after the visit, and not the survey cycle: a later visit is in none of my
nine." **Feels:** grounded: "that's the moment I mean".

**Decision:** the moment of use. In this journey, Nolan's ruling: "At a visit with a fasting blood
draw". No Recommended: the use decides.

**Card**
- Kicker: "Models · When it is used"
- Heading: "When will the model be used?"
- Return, the lede: "Because you said" over "Researchers, for everyone in a study that measured
  these but not glucose" (Your question), then "The draw it reads measured the lipids, not glucose.
  It may read only what is known by then."
- Options:
  - "Before any blood is drawn": "It reads what is asked and measured at the visit: 18 things, not
    the lipids." Quiet name: "predictors available before laboratory testing".
  - "At a visit with a fasting blood draw": "It also reads HDL and triglycerides from the same draw:
    20 things." Quiet name: "predictors measured up to the fasting draw (TRIPOD+AI 9b)".
- Settled by the moment, one quiet line (not a choice): "Not read: the survey cycle (a later visit
  is in none of your nine), the respondent number, and the six flags of values filled before your
  table was made."
- Why?: "A model that reads something known only after its moment looks better in a paper than in
  use (PROBAST, Wolff et al. 2019, item 2.3). A fasting draw usually measures glucose too; your
  model is for the draws that did not."
- Button: Continue.

**Under the other use** (clinicians): the return reads "at a draw that left glucose out", and
yesterday's diet is not read: a clinic visit has no 24-hour recall (PROBAST 2.3). The counts become
10 before the draw and 12 with it.

**Tapestry:** comparison view, the visit as a timeline (outcome-free).
- Three lanes, left to right:
  - **Before the visit** (the interview at home): age, gender; the two medicine questions, asked
    only of people told to take medicine for high blood pressure or high cholesterol.
  - **At the visit** (the exam): yesterday's diet, recalled once (total calories, protein,
    carbohydrate, total sugars, total fat and its three kinds); body measures (weight, height, body
    mass index, waist); blood pressure (systolic, diastolic); the fasting draw: HDL cholesterol and
    triglycerides, and fasting glucose drawn in outline, "not in this draw", by its name and unit
    only.
  - **After the visit:** "Nothing in your table", with the six processing flags in the quiet color.
- Beneath the lanes, one strip: the nine survey cycles, and "A visit after these is in none of
  them."
- Pointing at "Before any blood is drawn" moves HDL and triglycerides to "not read", outlined in the
  choice color; the count reads 18. Pointing at "At a visit with a fasting blood draw" lights them,
  "+2 from the same draw"; the count reads 20. Under the clinicians' use the diet row is gray, "not
  recalled at a clinic visit".
- Caption: "When each column of your table is known, at the moment the model is used. No glucose
  value is read until you press Fit." (The one place this is said.)

**Manuscript** (TRIPOD+AI 9a, 9b): "The model was intended for use at a visit with a fasting blood
draw that did not measure glucose. All variables available at that time were candidate predictors,
with no selection before modeling: age and gender; whether the participant was taking prescribed
medicine for high blood pressure and for high cholesterol; total energy, protein, carbohydrate,
total sugars and total, saturated, monounsaturated and polyunsaturated fat from one 24-hour dietary
recall; weight, height, body mass index and waist circumference; systolic and diastolic blood
pressure; and HDL cholesterol and triglycerides from the same fasting blood draw. Survey cycle was
not a predictor."

**Engine.** Serves the roles (`decisions.SetRoles`; `PREDICTOR_ROLES`, `decisions.py:2821`). Must
newly serve **P3** (the moment of use: `q:moment_of_use`, each column's moment from the lens's
codebook, compiled to predictor roles, the 9a and 9b sentences, and its return of the intended
use's pathway) and **P14** (a period column is a validation grouping under a later moment, never a
predictor). Conflict **C1**: the dietary lens proposes the role "exposure" for the nutrients under
Predict; the capture gave them the role P3 compiles to (a predictor, the engine's covariate).

### M2 · The medicine answers

**Understands:** "Two questions in my table were asked only of people told to take medicine for
high blood pressure or high cholesterol, so a blank means 'not asked', not 'unknown'. For a model of
everyone, the app reads them, each blank as 'not asked'." **Feels:** caught, then trusted: the app
knows what a blank means here.

**Decision:** whether the model reads the medicine answers. Recommended from Q; in this journey,
read them.

**Card**
- Kicker: "Models · The medicine questions"
- Heading: "Should the model read the medicine answers?"
- Return, the lede: "Because you said" over "Researchers, for everyone in a study that measured
  these but not glucose" (Your question), then "At the visit these answers are known, and they say
  whether a doctor has prescribed the medicine, so the app recommends reading them."
- Options:
  - "Read them, a blank as 'not asked'" (Recommended): "Three answers each: taking the medicine,
    told but not taking, not asked." Quiet name: "a skip-pattern blank kept as its own level".
  - "Leave them out": "It reads 18 things instead of 20."
- Why?: "NHANES asks whether someone now takes a prescribed medicine only after they were told to
  take it. So 15,552 of your 21,849 people were never asked about blood-pressure medicine, and
  17,204 never about cholesterol medicine. Filled with the most common answer, all of them would
  read as taking it; kept only where both were answered, 2,996 people would remain, chosen by their
  diagnoses."
- Button: Continue.

**Under the other use** (clinicians): the return is "Join the diabetes file first" (beat J): it
decides who the model is for, and only then are the answers read. With people already diagnosed
left out, "Read them" stays Recommended: being told to take blood-pressure medicine is a question
of the standard diabetes risk scores (FINDRISC, Lindström & Tuomilehto 2003; the American Diabetes
Association's risk test, Bang et al. 2009). Without the join, the card's Recommended waits.

**Tapestry:** the two questions' answers, outcome-free.
- Two bars of 21,849: blood-pressure medicine (taking 5,527; told, not taking 770; not asked
  15,552) and cholesterol medicine (taking 3,644; told, not taking 1,001; not asked 17,204). "Not
  asked" carries its meaning: "never told to take it".
- One line beneath: "Told to take at least one: 7,946 people (36%), from 27% of the 2001–2002 cycle
  to 44% of 2017–2018."
- Pointing at "Read them" lights each bar's three parts in the choice color. Pointing at "Leave them
  out" turns both bars to the quiet color, "not read".
- A permanent quiet line: "Filled with the most common answer, the 15,552 not asked would read as
  taking blood-pressure medicine."
- Caption: "What your table records for each question."

**Manuscript** (TRIPOD+AI 6c, 9b, 11): "Use of prescribed medicine for high blood pressure and for
high cholesterol was recorded only for participants who had been told to take it (NHANES Blood
Pressure and Cholesterol questionnaire). Each was entered as a predictor with three levels (taking,
told but not taking, and not asked), the last marking that the participant had not been told to
take it; no value was imputed."

**Engine.** Serves the blank as a level (`pipeline.MISSING_LEVEL`, `models/pipeline.py:16`) and the
findings `binary_text__meds_*`. Must newly serve **P1** (W1-H, in flight), **P9** (the
recommendation read from the intended use, with its reason) and the skip fact as a lens fact (E16's
BPQ check, extended to all nine cycles). Conflict **C2**: today the missing-values question ranks
"Single fill in each training fold" first, and its fill for a category writes "taking" into every
"not asked".

### M3 · Which kinds of model

**Understands:** "Three kinds of model add up curves, take about 2 minutes together, and print an
equation my paper can carry. The tree kinds can bend and combine measures and take from a minute to
hours. I fit the three now; the trees can run later, as a job." **Feels:** in control of the cost.

**Decision:** which families to fit. In this journey: least squares, ridge and the elastic net.

**Card**
- Kicker: "Models · Which kinds of model"
- Heading: "Which kinds of model should it try?"
- Lede: "Ranked for your 19,579 people and 20 measures."
- Options (checkboxes in the shelf's order, each with what it can find and its cost; the quiet name
  on the top edge):
  - Boosted trees: "Steps that can bend and combine measures. About 1 minute untuned;
    ⟨engine-filled: tuned, once C6a phase 3 lands⟩." Quiet name: "gradient-boosted trees".
  - Elastic net: "Curves pulled toward flat; it may drop weak measures. About 1 minute." Quiet name:
    "the elastic net (Zou & Hastie 2005)".
  - Ridge: "Curves pulled toward flat together; it keeps every measure. About 20 seconds." Quiet
    name: "ridge regression (Hoerl & Kennard 1970)".
  - Random forest: "Many deep trees, averaged. About 2 hours, tuned."
  - XGBoost: "Steps, tuned. About 18 minutes."
  - Least squares: "Each measure a curve, added up. About 9 seconds." Quiet name: "linear
    regression with restricted cubic splines".
  - Robust regression: "People far from the rest count less. About 55 seconds." Quiet name: "Huber
    regression".
  - One quiet line for the rest: "Not for this table: a screened elastic net (nothing here needs
    screening), mixed and GEE models (no one appears twice), feature-wise tests (they make no
    predictions)."
- Under the options, the ticked families' cost in one quiet line: "about 2 minutes in all" (the
  engine's estimate, 94 s).
- Why?: "The ranking reads what each model would be given: your people, your measures, their
  curves. A tree family can find bends and combinations the others cannot; whether that is worth
  its cost here is a question Results answers once one is fitted. The three that add up curves give
  an equation your paper can print in full (TRIPOD+AI 22)."
- Button: Continue.

The shelf's own ranking is the order; no family is labeled Recommended, because the choice here is
the person's budget.

**Tapestry:** "What each kind can find, and what it costs" (comparison view, outcome-free): one row
per family against four plain columns (bends; combines two measures; keeps every measure; an
equation your paper can print), and each family's cost as a dot on one time axis, seconds to hours,
with the hold's line at about 2 minutes: "Longer fits wait for Fit and run as a job." Pointing at a
family lights its row and its dot. Caption: "What each kind of model can find, not what it will
find."

**Manuscript** (TRIPOD+AI 12c): "Three model families were developed: least-squares linear
regression, ridge regression and the elastic net; tree-based families were not fitted [author: the
rationale]."

**Engine.** Serves the `shelf` stage (rank, fit, `inductive_bias`, the cost `estimate`, Riley's
sample size), `models/cost.py`, and the hold (`fit_press.HOLD_SECONDS`, `:51`). The line families'
costs above are the curves run's; the trees' are the straight-line shelf's (P15 gives them no
curves). Must newly serve the live ranking on each family's own input (MC-6, planned), **P15** (each
family's own recipe), **P29** (the Fit estimate counts every fit) and **P30** (Riley's count and
criteria). Conflicts **C3** (tree families are handed spline columns: the forest's estimate is about
15 hours with curves, 2 without), **C16** (the estimate, 94 s, against the fit's 220 s; ridge's
share, 21 s, against its 100 s) and **C17** (the shelf reads glucose's mean and spread on the
development rows for Riley's intercept criterion, counts 71 parameters, and names that criterion
binding on a tie it does not decide).

### M4 · Set for you

**Understands:** "Four conventions are set for me: each measure may bend; every measure is kept;
ridge and the elastic net choose how hard to pull inside each fold; and the four body measures,
which carry nearly one quantity, are all kept and read together." **Feels:** respected, quick.

**Decision:** "Confirm all 4".

**Card:** heading "Here are the 4 choices set for you"; each line with its reason, changeable
through the three levels of disclosure (FOUNDATION §3).
1. "Each measure may bend: a smooth curve through its own percentiles. Your 19,579 people support
   the 73 terms this makes; 3,916 would be enough." Quiet name: "restricted cubic splines, five
   knots by Harrell's rule (Harrell 2015); sample size by Riley et al. 2020". Changing it:
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

**Tapestry:** follows the pointed line; at rest, line 1. Each draws what the convention means.
1. "Should each centimeter of waist count the same, whatever someone's waist already is?": waist's
   spread over the development rows, with two equal 10 cm steps, 80 to 90 cm (19% of people) and
   110 to 120 cm (12%): "A straight line moves the estimate the same for both steps; a curve lets
   them differ." Pointing at the quiet name marks Harrell's five knots (74.0, 87.6, 96.8, 106.4 and
   127.0 cm).
2. The 20 measures in a column, all lit.
3. A small fold within a fold: "The pull is chosen without the people it is scored on."
4. "One quantity, four measures": how much of each body measure the other three carry (weight 99%,
   height 99%, body mass index 99%, waist 86%), and "body mass index = weight / height²".
- Caption: "What each convention means for your measures."

**Manuscript** (TRIPOD+AI 12b, 9a, 12c, 10): "Continuous predictors entered as restricted cubic
splines with five knots at Harrell's recommended percentiles, giving 73 model terms for 19,579
development participants, above the 3,916 required for an expected shrinkage of at least 0.9
(Riley et al. 2020). No predictor selection preceded model fitting. The ridge and elastic-net
penalties were chosen by cross-validation within each training fold."

**Engine.** Serves the sweep (`sweep.py`; `form`, `set_validation`, `set_levers`), the curves
(`methods/levers.RuleSplines`, `:139`, knots placed in each fold) and Riley's sample size
(`models/sample_size.py`). Must newly serve **P15**, **P16** (the collinearity noticing before Fit,
outcome-blind: today `models/linear.py:61` raises it after the fit, and on this run it names the
triglycerides curve's pieces, not the body measures) and **P30**. Conflict **C8**: `set_selection` is
a Decide under Predict; here it is line 2.

### M5 · Your plan

**Understands:** "Each of my eight development cycles will be estimated by models built on the other
seven, so I see whether it holds in a period it never saw; the latest cycle stays sealed. It is
judged on its squared miss, said in mg/dL beside a model that estimates everyone at the average.
Last, I decide that its level follows the latest cycle if the cycles' levels shift. What Results
will look at, and what the opening will test, is fixed now." **Feels:** ownership, and a little
suspense.

**Decision:** the level for a later visit. Recommended: follow the latest cycle, if the cycles
shift.

**Card**
- Kicker: "Models · Your plan"
- Heading: "Last, before you fit: should its level follow the latest cycle?"
- Return, the lede: "Because the latest cycle is sealed (Who's in), each of the other eight is
  estimated by models built on the other seven; the kinds of model are compared on 10 rounds of 5
  groups, and choosing the best is corrected for being chosen."
- Set for you, one line: "Judged by its squared miss, said in mg/dL beside a model that estimates
  everyone at the average: a big miss counts for more than a small one." Quiet name: "mean squared
  error, reported as the root mean squared error and R² (Gneiting 2011)".
- Options:
  - "Follow the latest cycle, if the cycles shift" (Recommended): "If the latest development cycle
    runs above or below its estimates, its level is reset there before the sealed cycle opens."
    Quiet name: "temporal recalibration of the intercept (Booth et al. 2020)".
  - "Keep the level it learns from all eight": "Its level is the eight cycles' average."
- The Recommended's reason, quietly under it: "Your model is used after 2018, and glucose was
  measured by five laboratory setups over your sixteen years."
- The card's foot holds no button (FOUNDATION §7).

**Tapestry:** the analysis flowchart.
- At its head, the timeline emblem.
- Boxes, left to right: **People** (21,849; 2,270 sealed, 2017–2018; 19,579 to build and check).
  **What it reads** (20 measures, 73 terms; not read: the cycle, the respondent number, the six
  flags). **The models** (least squares, ridge, elastic net, each with curves; beside them, a model
  that estimates everyone at the average). **How it is judged** (the eight cycles as a strip, each
  held out in turn; the squared miss; "compared on 10 × 5 groups"). **Its level** ("follows
  2015–2016 if the cycles shift", or "the eight cycles' average").
- "Fixed now: what Results will look at": how well it ranks people at 126 mg/dL or more, and
  whether its high estimates are right on average; women and men, three age groups, told to take a
  medicine or not; each survey cycle's level and ranking; calibration, the ends included; what each
  of your choices added.
- "What the opening will test": whether the sealed cycle's misses fall where the other cycles
  predict, and whether its level holds.
- Fit at its end: "Fit · about 2 minutes".
- Pointing at "Keep the level" redraws the level box.
- Caption: "Your whole plan. Nothing is estimated yet."

**Manuscript** (four sentences arrive on Fit; TRIPOD+AI 12a, 12c, 12d, 12e, 12f, 14):
- "Performance was estimated by internal–external cross-validation across the eight development
  cycles: each cycle was predicted by models developed on the other seven, and the cycles' mean
  squared errors were summarized by random-effects meta-analysis on the log scale, with a
  Hartung–Knapp–Sidik–Jonkman confidence interval and a 95% prediction interval for a new cycle."
- "Model families were compared on 10 repeats of 5-fold cross-validation by the corrected resampled
  t test, and the optimism of choosing the best was estimated by bootstrap bias-corrected
  cross-validation."
- "Performance was measured by the mean squared error and reported as the root mean squared error
  in mg/dL and R², beside a model predicting the development mean, with the calibration intercept
  and slope; the mean absolute error is reported descriptively. Declared before model fitting, the
  c statistic for fasting glucose of 126 mg/dL or more and calibration among the highest
  predictions were assessed, with performance within gender, age group and medicine-question group,
  by survey cycle, and with each modeling choice undone."
- "Declared before model fitting, the final model's intercept was to be re-estimated on the latest
  development cycle (2015–2016) if that cycle's mean prediction error out of cycle had a 95%
  confidence interval excluding zero."

**Engine.** Serves `validation.validation_plan` (`models/validation.py:359`), internal–external
validation (`:457`), the comparison substrate (`folds.comparison_folds`), BBC-CV
(`selection.selection_optimism`, `:245`), the Fit estimate (`FitLock.estimate_seconds`: 94.1 s) and
the analysis plan (`GET /plan`, `declared_at` the press). Must newly serve **P5**, **P12** (the plan
fixed at Fit, with the level rule and the opening's tests), **P14**, **P25** (the level rule) and
**P27** (the log-scale summary). Conflicts **C9**, **C14**, **C15**.

## 5 · Results

### R1 · How good it is

**Understands:** "In survey cycles it never saw, my model accounted for about a sixth of the
differences in fasting glucose between people; its misses came to 32 mg/dL against 35 for the
average. Through the middle of its range people average what it estimates; at both ends it reads
about 4 mg/dL low. Its estimates stay in a narrow band." **Feels:** honest clarity; a little
sobered.

**Decision:** none (reading).

**Card**
- Kicker: "Results · How good it is"
- Finding (26 px): "In survey cycles it never saw, it accounted for about a sixth of the
  differences in fasting glucose between people."
- Reading (17 px): "Its misses came to 32 mg/dL, against 35 for estimating everyone at the average.
  It is right on average through the middle of its range and reads about 4 mg/dL low at both ends,
  and its estimates stay in a narrow band: they spread 14 mg/dL, against glucose's own 35."
- Quiet names: "R² 0.15 (0.14 to 0.17); root mean squared error 32.4 (31.0 to 33.8) against 35.3;
  for a new cycle, 24.4 to 42.0 (95% prediction interval); calibration slope 0.98 (0.95 to 1.01),
  E90 3.3 mg/dL (Van Calster et al. 2019)".
- Why?: "A big miss counts more on this yardstick, the one the models were compared on; the average
  miss, which counts every mg/dL alike, was 17 mg/dL. With glucose this skewed, 8 in 10 of the
  squared misses come from the 12% at 126 mg/dL or more: the next screen looks at them. An estimate
  of the average for people like each person spreads less than the outcome, by about the square
  root of R² (0.39 × 35 ≈ 14)."
- Button: "Next: where it misses".

**Tapestry:** two panels.
1. "How far each missed, counting a big miss more": the root mean squared error with its interval
   for estimating everyone at the average (35.3), least squares (32.5), ridge (32.4) and the elastic
   net (32.4). Beneath it, ridge in each of the eight cycles, their summary (32.0, 29.1 to 35.2) and
   the band a new cycle should fall in (24.4 to 42.0).
2. "Right on average, in a narrow band": the mean measured against the mean estimated in each tenth
   of ridge's estimates, with intervals, on the diagonal (87.4 estimated, 91.5 measured, in the
   lowest tenth; within about 2 mg/dL in the middle eight; 134.7 and 138.3 in the highest). Beside
   it, on the same vertical axis, glucose's own spread on the development rows with 126 mg/dL marked
   and the band the estimates cover lit.
- Caption: "Out-of-cycle estimates of the 19,579 development participants: each cycle estimated by
  models built on the other seven. Glucose read after Fit, on these rows only; 2017–2018 stays
  sealed."

**Manuscript:** the development estimate across cycles opens the paper's first paragraph (R5).

**Engine.** Serves the fit's `cv` (RMSE, MAE, R², each with its interval), the `baseline` (MSE
1,244.6), `versus_baseline` (R² gain 0.155, 0.139 to 0.170), `calibration` (intercept, slope,
curve, E90 3.3) and `internal_external` (each cycle's MSE; its summary on the identity scale, RMSE
31.8, prediction interval 23.7 to 38.2). Must newly serve, in **P4**, the no-predictor model's RMSE
beside each family's and calibration by tenth with intervals; **P27** (the summary on the log
scale, which here moves it from 31.8 to 32.0 and its prediction interval from 23.7–38.2 to
24.4–42.0: the cycles' MSE and its standard error correlate at 0.81); **P18** (the calibration words
and the narrow band).

### R2 · Where it misses

**Understands:** "It ranks the people in the diabetic range well, and 4 in 10 of those it estimates
at 126 or more are there. It does better than the average in every group, but its misses run from
23 mg/dL among the young to 42 among people told to take a medicine. It ranks people as well in
every cycle, but its level shifts: 2015–2016 ran 4 mg/dL above its estimates." **Feels:** sobered,
but armed: they know where not to trust it.

**Decision:** none (reading).

**Card**
- Kicker: "Results · Where it misses"
- Heading: "Where it misses"
- Three lines, each a question and its verdict (§6):
  1. "Does it pick out who is in the diabetic range?" "It ranks them well (c 0.82), and 4 in 10 of
     the people it estimates at 126 mg/dL or more are there." Quiet name: "c statistic for fasting
     glucose of 126 mg/dL or more (Hosmer, Lemeshow & Sturdivant 2013); positive predictive value
     of an estimate at 126 or more, not a decision rule (TRIPOD+AI 15)".
  2. "Is it as good for everyone it is for?" "Better than the average in every group; its misses run
     from 23 mg/dL among the youngest to 42 among people told to take a medicine." Quiet name:
     "paired difference in squared error against the no-predictor model; root mean squared error
     by group (TRIPOD+AI 14, 23a)".
  3. "Does it hold in another period?" "It ranks people as well in every cycle, but its level
     shifts: 2015–2016 ran 4 mg/dL above its estimates." Quiet name: "rank correlation 0.52 to 0.55
     by cycle; calibration-in-the-large (TRIPOD+AI 23b)".
- Why?: "People measured at 126 or more average 179 mg/dL, and it estimates them at 123. That is
  not a bias a recalibration would fix: anyone picked out by a high measurement sits above the
  average of people like them, so a perfectly calibrated version misses them by 56 too (Bland &
  Altman 1995). What it can do is rank them high and be right where it estimates high. NHANES's
  equations bridge the laboratory's recorded changes, and they do not explain the two cycles that
  still ran high, 2007–2008 and 2015–2016."
- Button: "Next: what drives it".

**Tapestry:** at rest, the first question.
- "Who it puts high": the share at 126 mg/dL or more in each tenth of ridge's estimates, from 0.5%
  in the lowest to 41% in the highest, with c = 0.82; and the 2,094 people it estimates at 126 or
  more: "4 in 10 are there; they average 137 mg/dL, and it estimates them at 134."
- Pointing at Why?'s first sentence: estimated against measured, both in mg/dL, with the 126 lines,
  and the people picked by their measured value (2,262): their mean estimate beside their mean
  measured value, for ridge (56.5 low) and for a perfectly calibrated version of the same estimates
  (55.6 low): "the same pull, calibrated or not."
- Pointing at question 2: for each group, its root mean squared error beside the average's, and the
  share of the group's own differences it explains: women 30.8 (34.0), 17%; men 34.0 (36.6), 13%; 18
  to 36, 22.8 (25.9), 5%; 37 to 58, 35.1 (37.7), 13%; 59 and over, 37.8 (40.9), 9%; told to take a
  medicine 42.0 (45.4), 8%; neither 25.7 (28.2), 12%. "Among the youngest it tells people apart
  least: their glucose is mostly normal, and its estimates mostly say they are young."
- Pointing at question 3: the eight cycles' level (measured less estimated, with intervals: 2001
  −0.4, 2003 −1.6, 2005 −1.6, 2007 +2.5, 2009 −1.9, 2011 −0.9, 2013 0.0, 2015 +4.0 mg/dL) under the
  laboratory bands from D, and each cycle's rank correlation (0.52 to 0.55).
- Caption: as R1's.

**Manuscript:** the planned checks open the paper's second paragraph, on the opened cycle (R5); two
Discussion drafts arrive (the diabetic range, the level).

**Engine.** Must newly serve **P4** (on the declared folds and on the opened rows: c with its
interval and the share at the cut among estimates at or above it; calibration among the highest
estimates; the groups with paired differences, R² against the no-predictor model and within each
group; each cycle's level with its interval and rank correlation) and **P18** (the verdicts by §6's
rules). Conflict **C4**: the engine scores the groups on the comparison folds' first repeat (random
5-fold), not on the declared scheme.

### R3 · What drives it

**Understands:** "All three models read the same groups most: HDL and triglycerides from the draw,
then age, the medicine answers and body size, then diet, gender and blood pressure. The lipids are
also the most valuable thing it reads: without them it would explain 12.5% of the differences, not
15.5%. The medicine answers, the curves, body size and age each add about a point." **Feels:**
curious, "so that's what it reads", and that their own choices mattered.

**Decision:** none (reading).

**Card**
- Kicker: "Results · What drives it"
- Heading: "What moves its estimates?"
- Finding (17 px): "In all three models, HDL and triglycerides from the draw move an estimate most
  (4.4 mg/dL on average), then age, the medicine answers and body size, then diet, gender and
  blood pressure."
- What your choices added, three returns, on the squared miss:
  - Because you said "At a visit with a fasting blood draw" (When it is used): "The lipids are the
    most valuable thing it reads: without them it would explain 12.5% of the differences, not
    15.5%."
  - Because you said "Read them, a blank as 'not asked'" (The medicine questions): "Without them,
    14.5%."
  - Set for you, each measure may bend: "With straight lines, 14.6%."
- Quiet names: "grouped SHAP values (Lundberg & Lee 2017); accumulated local effects (Apley & Zhu
  2020); each choice undone and refit on the same folds, paired (Lei et al. 2018)".
- Why?: "These describe how each model turns measures into estimates, not what changing a measure
  would do. Calories and the nutrients they are made of move together, as do the four body
  measures, so credit is given to each group: how it splits within a group is arbitrary. In ridge the
  diet group's eight inputs take shares that add to 17 mg/dL but net to 2.4, and one input's rank
  changes from refit to refit."
- Button: "Next: your final model".

**Tapestry:** three panels.
1. "What moves each model's estimates": each plain group's net push on one mg/dL axis, the three
   families side by side. Ridge: lipids 4.4, age 4.0, medicine answers 3.6, body size 2.9, diet
   2.4, gender 2.3, blood pressure 1.7; least squares within 0.2 of these; the elastic net lipids
   4.1, age 3.8, medicines 3.7, body size 2.5, gender 2.0, diet 1.9, blood pressure 1.7.
2. "How an estimate moves with one measure, the others as they are": accumulated local effects on
   shared axes for waist, triglycerides, HDL and age, the three families' curves overlaid (they
   nearly coincide). Ridge: waist flat at about −4 mg/dL to 88 cm, then rising to +1.9 at 110 cm,
   +8.8 at 129 and +20.8 at 160; triglycerides from −5.0 at 17 mg/dL to +7.4 at 289 and +15.4 at
   489; HDL from +5.1 at 23 mg/dL to −5.9 at 106; age from −6.7 at 18 to +5.2 at 64, then down to
   +0.3 at 80, which means 80 and over. No curve where fewer than five people sit. Total calories
   behind "More angles", with one line: "Calories cannot change while the nutrients they are made
   of hold still; read diet as one group."
3. "What your choices added": the share explained with every choice (15.5%) and with each undone,
   each with its paired interval: without the lipids 12.5%; without body size 14.5%; without the
   medicine answers 14.5%; without age 14.6%; straight lines 14.6%; without diet 14.9%; without
   blood pressure 15.2%; without gender 15.2%.
- Caption: "Described on the development rows: what the models do, not what changing a measure would
  do (Molnar et al. 2022)."

**Manuscript** (TRIPOD+AI 12c): "The fitted models were described by grouped SHAP values and by
accumulated local effects on shared axes, and the contribution of each modeling choice was
estimated by refitting without it on the same folds, compared by the paired difference in squared
error; these describe the models' predictions, not causal effects."

**Engine.** Serves the `explain` stage (exact linear SHAP, `models/explain.py:117`; stability across
five reseeds, rank correlation 0.48 for ridge, 0.56 for least squares, 0.63 for the elastic net;
accumulated local effects with their masks, `:626`; each family's equation, `:947`). Must newly
serve **P6** (each plain group's net SHAP across families, the curves chosen from it, and per-input
shares said unstable under a stated rank correlation) and **P10** (what each choice added, on the
squared miss). Conflict **C5**: the curves follow one family's top three among the "exposure"
columns (`_exposure_inputs`, `:1382`).

### R4 · Your final model, then the latest cycle

**Understands:** "The three are equally good within a stated margin; ridge is the simplest whose
equation holds up. Its level follows the latest cycle, as my plan said: 2015–2016 ran 4.0 mg/dL
above its estimates, so its level rises 3.4. I name it; then I open 2017–2018, which will show
whether its misses fall where the other cycles predicted and whether the level holds." **Feels:**
decisive: the moment of truth.

**Decision:** the final model (Recommended by ruling 4: ridge); then the opening, on its own press.

**Card**
- Kicker: "Results · Your final model"
- Heading: "Which one is your final model?"
- Return: "The three missed alike: on 10 rounds of 5 groups, each differs from the best by less than
  a twentieth of what the best adds over the average."
- Options:
  - "Ridge" (Recommended): "As good as the best, and its equation is stable: it shares weight among
    measures that move together." Quiet name: "ridge regression (Hoerl & Kennard 1970)".
  - "Least squares": "As good, but its curve for triglycerides is poorly determined: the curve's
    pieces nearly cancel one another, so its equation would not hold up." Quiet name: "least
    squares; scaled condition number 4,229".
  - "Elastic net": "As good; how much it keeps of each measure changes from fold to fold." Quiet
    name: "the elastic net".
  - ⟨engine-filled: each tree family once fitted, with what the regression costs or gains against
    it (BBC-CV over the flexible families)⟩.
- Because you said "Follow the latest cycle, if the cycles shift" (Your plan): "2015–2016 ran 4.0
  mg/dL above its estimates, so its level is raised by 3.4." Quiet name: "temporal recalibration
  (Booth et al. 2020): the intercept re-estimated on 2015–2016's 2,227 people, whose level the model
  had partly learned already".
- One quiet line: "Choosing the best of three flatters it by nothing measurable here." Quiet name:
  "bootstrap bias-corrected cross-validation (Tsamardinos et al. 2018)".
- Button: "Name ridge". Once pressed, the card's foot holds the second press: "Open the latest
  cycle", and under it: "2,270 people from 2017–2018, sealed since Who's in. They open once;
  anything changed afterward is reported as decided after. The opening checks what your plan fixed:
  whether its misses fall where the other cycles predict (24 to 42 mg/dL), and whether its level
  holds."

**Under the other use** (clinicians): unchanged; its population would differ (beat J).

**Tapestry:** at rest, the other two families' paired differences from the best on the comparison
folds (least squares 0.6 worse, −1.5 to 2.7; the elastic net 0.8 worse, −1.0 to 2.7, in squared
mg/dL) inside the margin's band (±9.6), and the sealed block, 2,270, still closed. Pointing at
"Least squares" draws the reason: the engine's concern in one line, and its triglycerides curve
beside ridge's (the same shape, from coefficients its fit cannot pin down). Pointing at the level
line draws the eight cycles' levels with 2015–2016 lit and the update as an arrow, and the forward
check on development cycles alone: a model built on 2001–2012 with its level set on 2013–2014 would
have missed 2015–2016's level by 3.4 mg/dL instead of 4.1. Caption: "Compared on 10 rounds of 5
groups of the development rows. 2017–2018 is still sealed."

**Manuscript** (TRIPOD+AI 12c, 12f): "Ridge regression was declared the final model before the
held-out cycle was analyzed: it was the simplest family whose difference from the best in
cross-validated mean squared error lay within a prespecified margin (5% of the best family's
improvement over the development mean) and whose coefficients were well determined. As
prespecified, its intercept was re-estimated on 2015–2016 (+3.4 mg/dL), whose mean prediction error
out of cycle, 4.0 mg/dL (2.4 to 5.6), excluded zero."

**Engine.** Serves the final-model refusal and its exits (`selection._a_final_model_is_declared`,
`models/selection.py:657`), BBC-CV, the pairwise `comparisons`, the near-singular concern
(`models/linear.py:61`) and `open_seal`. Must newly serve **P13** (the Recommended final model by
ruling 4, named on its own press; the opening its own press), **P25** (the level, applied before the
opening and carried by its record) and **P2** (W1-H: the open-noticings gate, here already cleared at
D). Conflicts **C7** and **C14** (`set_updating` knows "none" and "shrinkage" only).

### R5 · Your sentence, and "Put this in my paper"

**Understands:** "On 2,270 people from 2017–2018, sealed until now, my model missed by 33 mg/dL
against 37 for the average, inside what the other cycles predicted, and its level held. Its highest
estimates read low in that cycle, which had more people in the diabetic range. My paper says the
development estimate, then the opened cycle, then the planned checks; the Discussion says what it
is not for." **Feels:** proud: it is defensible, and honest about its limits.

**Decision:** "Put this in my paper".

**Card**
- Kicker: "Results · Your sentence · opened at [the opening's time]"
- Finding (26 px): "On 2,270 people from 2017–2018 it never saw, your model's misses came to 33
  mg/dL, against 37 for estimating everyone at the development average, and its level held."
- The plan's two tests, resolved, in one quiet line each: "Expected from the other cycles: 24 to 42
  mg/dL; found 33." "Its level, set on 2015–2016: within 1 mg/dL (3.8 low without it)."
- The Recommended wording, "Prediction performance, with the planned checks", as two paragraphs at
  17 px, each labeled quietly and each at most 90 words (84 and 84 here):
  > Of 21,849 participants, the 19,579 from 2001–2016 were used for development and the 2,270
  > from 2017–2018 were held out (Figure 1). In internal–external cross-validation across the
  > development cycles, the final ridge model's root mean squared error was 32.4 mg/dL (95% CI 31.0
  > to 33.8; prediction interval for a new cycle 24.4 to 42.0), against 35.3 for the development
  > mean. In 2017–2018 it was 33.1 mg/dL (29.8 to 36.4), with calibration slope 1.14 (1.04 to 1.24)
  > and calibration-in-the-large 0.5 mg/dL (−0.9 to 1.8) (Table 2).

  > As planned, in 2017–2018 the model ranked participants with fasting glucose of 126 mg/dL or
  > more with a c statistic of 0.81 (0.79 to 0.84), and 43% of those predicted at 126 mg/dL or more
  > were in that range. It improved on the development mean in every subgroup, with root mean
  > squared errors from 18.4 mg/dL (14.4 to 22.3) at age 36 years or younger to 40.9 mg/dL (36.4 to
  > 45.4) among participants told to take medicine for blood pressure or cholesterol (Table S2).
- "As planned" is earned by P12, which fixes the checks before Fit; without it the paragraph opens
  "In further analyses".
- Behind "Other wordings": "Write my own", which passes the same manuscript gate (every number
  traces to the record). No effect wording is offered under Predict.
- Button: "Put this in my paper".

**Tapestry:** Table 1, Table 2 and where each part goes.
- Table 1 (TRIPOD+AI 20b, 20c), development beside the opened cycle: women 51% and 52%; median age
  47 and 51.5; body mass index 27.6 and 28.4; waist 96.8 and 98.8 cm; systolic pressure 120 and
  122; HDL 51 and 51 mg/dL; triglycerides 106 and 92 mg/dL; taking blood-pressure medicine 25% and
  29%, cholesterol medicine 16% and 21%; values missing (after D) by measure; fasting glucose median
  99.9 and 104, at 126 mg/dL or more 11.6% and 15.8%. One quiet line: "Triglycerides ran lower in
  2017–2018 with no recorded change of laboratory method (NHANES TRIGLY_J)."
- Table 2: the final model as declared on 2017–2018 (root mean squared error 33.1, 29.8 to 36.4; R²
  0.20, 0.17 to 0.24, against the development mean; calibration slope 1.14, 1.04 to 1.24;
  calibration-in-the-large 0.5 mg/dL, −0.9 to 1.8; c at 126 mg/dL 0.81, 0.79 to 0.84); the
  development mean on the same rows (37.1, 33.6 to 40.6); without the level update (33.3; −3.8
  mg/dL, −5.2 to −2.5); labeled secondary, the other two families (least squares 33.3, the elastic
  net 33.5, neither updated); and the development estimate across cycles. The average miss (18.2)
  in a footnote, as description.
- Where each part goes: Results (the two paragraphs; Table 1; Table 2; Figure 1, the participant
  flow); the Supplement (S1 the full equation, TRIPOD+AI 22; S2 the groups; S3 the cycles; S4
  calibration by tenth on both; S5 what drives it; S6 what each choice added); the Discussion
  (seven drafts, below).
- The TRIPOD+AI checklist as it stands sits under For the record.

**Discussion drafts** (TRIPOD+AI 25, 26, 27c), each kept, edited or dropped in Write-up:
1. "The model's estimates are averages for people like each participant: they ranked people in the
   diabetic range well but cannot reach individual values in it, so the model is not suited to
   classifying who has diabetes; that use needs a model of the binary outcome, judged by decision
   curve analysis."
2. "Fasting glucose ran higher in 2015–2016 than the model estimated, beyond the instrument changes
   NHANES's equations bridge; as prespecified, the intercept was re-estimated on that cycle, which
   removed the error of level in 2017–2018. Before use elsewhere its level should be set on local
   data (Van Calster et al. 2019)."
3. "In 2017–2018 the highest predictions were too low (calibration slope 1.14); that cycle had more
   participants with fasting glucose of 126 mg/dL or more (15.8% against 11.6%)."
4. "Whether participants had been told they have diabetes, which NHANES records and which is known
   at a visit, was not used; for a model of all adults it would be expected to improve accuracy,
   most among people with treated diabetes."
5. "The analysis was unweighted and describes these participants, not the US population."
6. "Diet was recorded by one 24-hour recall; with a different dietary instrument the model would
   need recalibrating (Luijken et al. 2019)."
7. "Performance was not examined by race and ethnicity, income or education, which NHANES records
   but the analysis data did not include."

**After "Put this in my paper":** the rail opens with the two paragraphs under "Results · just
placed"; "Next: the exhibits" opens them in place, and "Confirm all 9" places them. Write-up is the
next design.

**Manuscript:** the two paragraphs, the seven Discussion drafts, the full equation for the
Supplement (TRIPOD+AI 22, from the explanations' `architecture.equation`), and the usability
sentence (TRIPOD+AI 27a): "At use, a medicine question not asked enters as 'not asked', and a
missing body measure or blood pressure is filled by the development data's median, as in
development."

**Engine.** Serves the export (methods in TRIPOD+AI sections, `export/methods.py:47`; the checklist,
today 4 items answered, 11 partly and 37 unanswered). Must newly serve **P19** (the Results
paragraphs and Discussion drafts, the methods in the register, and TRIPOD+AI rules for items 6c, 7,
8a, 9a, 9b, 10, 11, 12d, 12f, 14, 15, 16, 20b, 20c, 22, 23a, 23b, 24 and 27a), **P8**, **P12** (to
say "as planned") and **P21** (the opening of a period).

## 6 · Results' interpretation order under Predict

Each verdict is computed on the declared yardstick; none is stronger than its evidence.

1. **How good it is.** The declared scheme's out-of-sample MSE, said as the share of the differences
   explained (R²; "about a sixth" for 0.155) and the RMSE in mg/dL beside the no-predictor model's,
   each with its interval; the cycles' summary on the log scale with its prediction interval for a
   new cycle. The MAE is description, never a verdict.
2. **Calibration.** "Right on average through the middle of its range" when the slope's interval
   includes 1, the intercept's includes 0 and E90 is under 5% of the outcome's mean (3.3 of 108
   here); then the ends: each outer tenth whose measured-less-estimated interval excludes 0 is said
   ("reads about 4 mg/dL low at both ends"). A slope below 1 is "its estimates are too spread out",
   above 1 "not spread out enough". "A narrow band" when the estimates' standard deviation is under
   half the outcome's, said with both (14 and 35).
3. **The diabetic range,** for an outcome with a clinical cut-off (a lens fact: fasting glucose 126
   mg/dL, American Diabetes Association). Said by the c statistic with its interval ("ranks them
   well" at 0.80 or more, "fairly" from 0.70, "poorly" below; Hosmer, Lemeshow & Sturdivant 2013)
   and by the share at or above the cut among estimates at or above it, with their measured and
   estimated means. Never by a count placed at the cut (it rewards spread), and never by the miss
   among people picked by their measured value, which regression to the mean fixes for any
   calibrated estimate: that miss is a teaching line in Why?, beside a perfectly calibrated
   version's.
4. **The groups** the intended use names. "Better than the average in every group" when each
   group's paired difference in squared miss from the no-predictor model has an interval below 0;
   then the groups with the largest and smallest RMSE; a group whose within-group R² is under half
   the overall is said as the one "it tells people apart least".
5. **The period.** Each cycle's level (measured less estimated) with its interval, and its rank
   correlation; "ranks people as well in every cycle" when the rank correlations span under 0.05;
   "its level shifts" when the latest development cycle's level interval excludes 0. Never "drifts":
   a level is said beside the laboratory's recorded changes and NHANES's equations.
6. **What drives it.** By plain group, each group's net SHAP across families; "in all three models"
   when every family ranks the same group first; one input's share is said unstable when the
   reseeds' rank correlation is under 0.7; the line on arbitrary credit is said when a group's summed
   shares exceed twice its net.
7. **What each choice added.** The choice undone and refit on the same folds, the paired difference
   in squared miss with its interval, said as the share explained with and without; "made no
   difference here" when the interval includes 0.
8. **Which to keep:** ruling 4 (the margin, then the simplest without a concern), and ruling 10 (the
   level). BBC-CV's correction below the naive score is said as "nothing measurable".
9. **The opening.** It resolves the plan's tests: the opened cycle's RMSE inside or outside the
   cycles' prediction interval, and its level with and without the update; then calibration by
   tenth, the ends included; the final model as declared, the others secondary.
10. **The wording.** Prediction performance; explanations "describe the model", never an effect.
    Paragraph 1 is the development estimate across cycles, then the opened cycle; paragraph 2 is the
    checks fixed before Fit, on the opened cycle, with their intervals; each at most 90 words;
    nothing decided after the opening. The Discussion drafts come from 3, 5, 9 and the noted lines.

## 7 · The tapestry's pictures under Predict

| Question | The picture | View kind |
|---|---|---|
| Whom it is for | the people as one bar, by what the table records; what the use needs that it lacks | comparison |
| What your use needs | the bar with the unknown diagnosis; the two files on the respondent number | comparison |
| What your data raised | each line's evidence: the filled values off their measures; the laboratory's setups and equations; the zeros; the ages | distribution, lineage |
| The held-out rows | the participant flow with the sealed cycle at the strip's end | row flow |
| When it is used | the visit as a timeline, each column at its moment; the period strip | comparison (the timeline) |
| The medicine answers | each question's answers as a bar, "not asked" named | comparison |
| Which kinds of model | what each kind can find, against its cost on a time axis | comparison |
| Set for you | what each convention means: two equal waist steps; one quantity in four measures | distribution, comparison |
| Your plan | the analysis flowchart with the cycle strip, the level, what Results will look at and what the opening tests | lineage (the flowchart) |
| How good it is | the miss as a forest with the cycles' band; calibration by tenth beside the outcome's spread | forest, calibration |
| Where it misses | who it puts high; the same pull, calibrated or not; the groups; the cycles' levels under the laboratory | curve, relationship, forest |
| What drives it | grouped SHAP by family; accumulated local effects on shared axes; choices undone | table, curve, forest |
| Your final model | paired differences inside the margin; the cycles' levels with the update; the sealed block | forest |
| Your sentence | Table 1, Table 2 and the placement map | table, page |

The timeline is the Predict counterpart of the Estimate journey's compared day: an outcome-free
picture of what a choice means. From M2 to M5 it rides as a small emblem at the tapestry's head.

## 8 · The manuscript rail under Predict

TRIPOD+AI's order (`export/methods.TRIPOD_SECTIONS`), the methods register, the codebook's names;
readings confirmations and the engine's mechanics stay in the record. The rail grows: before Models
it holds the data, the participants, the outcome with its laboratory equations, the intended use
(with its Introduction draft), the data preparation and the held-out cycle; M1 adds the predictors
(9a, 9b); M2 the medicine answers (6c, 11); M3 the families (12c); M4 the curves, no selection and
the tuning (12b, 10); Fit the validation, the comparison, the measures and the level rule (12c, 12d,
12e, 12f); R3 the explanations; R4 the final model and its level (12f, 24); R5 the Results
paragraphs and the Discussion's drafts. The card's foot says what each confirmation added ("2
sentences added to Methods"), in the recorded green, which opens the rail.

## 9 · Methods rulings proposed

Each is the design owner's, with its source; an independent adversarial check follows (the plan's
W1-I). Rulings 1 to 9 are Round 0's, revised where the critique moved them; 10 to 13 are new.

- **Ruling 1 · The medicine answers.** They are known at the moment of use, so M1 admits them. Their
  blanks mean "not asked" (the BPQ skip pattern), so they enter with three levels (taking, told but
  not taking, not asked): never filled with the most common answer, which would mark 15,552 people
  as taking blood-pressure medicine, and never complete cases, which would keep the 2,996 who
  answered both. For estimation in everyone, read them. For people not known to have diabetes, the
  population is fixed first, by joining the diabetes questionnaire and leaving out people already
  diagnosed; then read them, since being told to take blood-pressure medicine is a standard
  screening predictor (FINDRISC, Lindström & Tuomilehto 2003; Bang et al. 2009). The case-mix
  problem Round 0 answered by dropping predictors is the population's, not the predictors' (Moons et
  al. 2012). The group they derive is said as what it is: told to take a blood-pressure or
  cholesterol medicine. Sources: the NHANES BPQ questionnaire (verified for 2015–2016, to verify for
  the other eight cycles); Sperrin et al. 2020 and Sisk et al. 2023 on missing values in prediction.
- **Ruling 2 · One yardstick for a skewed outcome.** The mean squared error is what the families are
  compared on, chosen on, and judged by: every family on this shelf estimates the conditional mean,
  and squared error is a scoring function consistent for the mean (Gneiting 2011). It is said as
  R² and as the RMSE in mg/dL beside the no-predictor model; every verdict, return and choice is a
  paired difference in squared miss with its interval. The MAE is description only: it is
  consistent for the median, and on this outcome it judges differently (the no-predictor constant
  sits near the median of older and treated people, so it "wins" there on MAE while losing 240 and
  303 squared mg/dL on MSE). At the clinical cut-off, the c statistic and calibration among the
  highest estimates; never a count placed at the cut, and never the miss conditioned on the
  measured value (Bland & Altman 1995). No log scale: glucose's log is skewed too (2.46 on the
  development rows).
- **Ruling 3 · Validation across survey cycles, and the seal by period.** For a model meant for a
  later visit, on data spanning sixteen years, the seal is the latest cycle, whole (temporal
  validation), and the development estimate is internal–external cross-validation over the other
  cycles, summarized by random-effects meta-analysis of the log MSE, with a
  Hartung–Knapp–Sidik–Jonkman interval and a prediction interval for a new cycle on t(k − 2)
  (Steyerberg & Harrell 2016; Riley et al. 2016; Higgins, Thompson & Spiegelhalter 2009; IntHout et
  al. 2014; Snell et al. 2018 on choosing the scale; Collins et al. 2024; TRIPOD+AI 12a, 12d, 23b).
  The identity scale is optimistic here, because the cycles with a larger MSE have a larger standard
  error (correlation 0.81) and so less weight. The families are still compared, and the choice
  corrected, on 10 × 5-fold cross-validation (the engine's MS6). A random holdout is offered for
  tables with no period column; its rationale is the lockbox against the analyst's own choices
  (Cawley & Talbot 2010; Gelman & Loken 2013), not Steyerberg 2018, which argues against random
  splitting.
- **Ruling 4 · The final model.** Among the families whose paired difference from the best on the
  comparison folds (corrected resampled t; Nadeau & Bengio 2003) lies within a stated margin, a
  twentieth of the best family's gain over the no-predictor model (here ±9.6 squared mg/dL, about
  0.15 mg/dL of RMSE), the simplest whose fit raised no concern its equation would carry into the
  paper (TRIPOD+AI 22); the simplest of them all when every one raised one. "Not distinguishable"
  is never read from an interval that crosses zero alone. Named on its own press, before the seal
  opens, with BBC-CV's correction reported (Tsamardinos et al. 2018). Here least squares' curve for
  triglycerides is poorly determined (scaled condition number 4,229), and ridge, built for exactly
  that (Hoerl & Kennard 1970), is chosen. With a tree family fitted, the interpretable cost decides
  whether the regression's simplicity is worth its price.
- **Ruling 5 · Energy adjustment and the substitution under Predict: neither is asked.** Unchanged
  from Round 0: the energy model decides what a nutrient's coefficient means, an Estimate question
  (Willett, Howe & Kushi 1997); a prediction model's nutrients enter as recorded beside total
  calories, For the record. The substitution compares two imagined diets, an effect (FOUNDATION §8;
  Shmueli 2010). The dietary energy noticing does not fire under Predict.
- **Ruling 6 · Curves by default for the families that draw lines.** Under Predict, each continuous
  predictor of least squares, ridge and the elastic net enters as a restricted cubic spline with
  knots by Harrell's rule, placed in each training fold, when Riley's criteria hold for the terms it
  makes (Harrell 2015; Riley et al. 2019, 2020); straight lines stay one click away, and tree
  families get none. As a product default it is fixed before any fit of a person's data. In this
  capture it was adopted after the first run's cross-validated benchmark, so a paper from this walk
  says the form was chosen on cross-validation. Here: 73 terms, 3,916 rows needed, 19,579 available;
  undoing it lowers the share explained from 15.5% to 14.6%.
- **Ruling 7 · The period is never a predictor under a later moment of use.** A later visit falls in
  no level of the cycle; the cycle is the draw's and the validation's grouping (ruling 3).
- **Ruling 8 · What Results checks, and what the opening tests, is fixed before Fit.** Under Predict
  nothing locks, but the yardstick, the scheme, the standing checks (the diabetic range by c and
  calibration at the top, the groups, the cycles, calibration by tenth, each choice undone), the
  level rule and the opening's two tests are declared with the plan at Fit, and the sealed cycle is
  scored on exactly them, once; a check added later is labeled as added after the cross-validated
  scores were seen. This is what lets the paper say "as planned".
- **Ruling 9 · The groups follow the intended use** (TRIPOD+AI 14, 23a): gender, age in thirds and
  told to take a medicine, the last derived from the skip pattern; race and ethnicity, income and
  education come with the demographics file (beat J). Unjoined, they are named as groups not
  examined, never as "not recorded" (TRIPOD+AI 3c, 26).
- **Ruling 10 · The level for a later visit (new).** When the use is later than the data and the
  table spans periods, the plan asks before Fit whether the model's level follows the latest
  period. If it does, and the latest development period's out-of-cycle mean error has a 95% interval
  excluding 0, the final model's intercept is re-estimated on that period, its coefficients kept
  (temporal recalibration, Booth et al. 2020; Janssen et al. 2008 on updating by the intercept), and
  the sealed period tests it. A forward check on development periods alone is reported beside it
  (here a level set on 2013–2014 would have cut 2015–2016's error of level from 4.1 to 3.4 mg/dL),
  never used to decide after the fact.
- **Ruling 11 · The intended use names its care pathway (new).** Who uses the estimate, for whom, and
  at what moment fit together: at a fasting draw, glucose is usually measured, so the model is for
  draws that did not measure it, in research data or in a clinic. What the use needs that the
  table lacks is said in Your data: a diabetes diagnosis under either use (TRIPOD+AI 3b; PROBAST,
  Wolff et al. 2019).
- **Ruling 12 · The data's own warnings are decided before the draw (new).** Values filled before the
  table was made, by an unrecorded method, are read as missing and filled in each training fold
  without the outcome (TRIPOD+AI 11; Sisk et al. 2023); a value no measure can take (a resting
  diastolic pressure of 0) is missing; top-codes are harmonized to the coarsest; an outcome measured
  by changing laboratories is bridged by the survey's published equations where they exist, and the
  gaps are said (NHANES laboratory documentation; Ingram et al. 2018; TRIPOD+AI 7, 8a).
- **Ruling 13 · A predictor is read only where the use records it (new).** A 24-hour recall belongs
  to research data; a clinic visit has none, so the clinicians' use leaves diet out (PROBAST 2.3).

## 10 · Engine requirements

Sizes: S = 1, S–M = 2, M = 3, M–L = 5.5, L = 8. P1–P8 are the plan's (`WALKABLE_PLAN.md` §3),
confirmed and resized where the beats changed them; P9–P21 are Round 0's, revised; P22–P30 are new
in Round 1. "Phase 3" is C6a phase 3 (`decisions.py`, `quest.py`, `voice.py`, `service.py`); every
item touching those files waits for its merge.

| P | What | Status | Size | Wave | Files and conflicts |
|---|---|---|---|---|---|
| P1 | A skip-pattern blank: its own level recommended, never the most common answer; the "not asked → No" recode | NEW, in flight (W1-H) | 2 | W1 | `methods/missing.py`, `coach.py`, `repairs.py` |
| P2 | The open-noticings gate blocks the opening; cleared at D, never by the opening's own press | NEW, in flight (W1-H) | 1 | W1 | `seal.py` |
| P3 | Moment of use: `q:moment_of_use`, each column's moment from the lens's codebook, compiled to predictor roles; the 9a and 9b sentences; its return of the intended use's pathway | NEW | 3 | W2 | decisions, quest, voice (after phase 3); a lens table of moments |
| P4 | Where it misses, on the declared folds and on the opened rows: c at a clinical cut-off with its interval; the share at the cut among estimates at or above it; calibration by tenth with intervals, E90 and the outer tenths; the groups with paired differences in squared miss, R² against the no-predictor model and within the group; each cycle's level with its interval and rank correlation; the no-predictor model's RMSE | NEW (resized from 3) | 5.5 | W2 | a new stage file and `stages/__init__`; not `evaluation.py` until MC-2b-3 |
| P5 | The yardstick declared: MSE compared, chosen and judged on; RMSE and R² said; MAE descriptive (ruling 2) | NEW | 2 | W2 | metrics declaration |
| P6 | Each plain group's net SHAP across families; the curves chosen from it; per-input shares said unstable under a stated reseed rank correlation | PARTIAL (`explain.grouped`, `:459`) | 2 | W2 | `models/explain.py` (after phase 3) |
| P7 | Plain names for quest lines and NHANES columns | PARTIAL | 3 | W2 | quest, voice |
| P8 | Sectioned manuscript endpoint for the rail | NEW | 1 | W1 | routes, service (after phase 3) |
| P9 | Intended use for a numeric outcome: its pathway and population, the copy "estimation", and the medicine recommendation read from it with its reason | NEW | 2 | W2 | decisions, quest, voice |
| P10 | What each choice added: the final family refit without a group or with the alternative recipe on the declared folds, paired on the squared miss, said as the share explained with and without | NEW (the Predict side of BE10) | 3 | W2 | a new stage file |
| P11 | Under Predict the energy-model and substitution questions and the energy noticing do not fire (ruling 5) | NEW | 2 | W2 | `interview._not_applicable` (`:628`), the dietary pack's detector |
| P12 | The evaluation plan fixed at Fit (yardstick, scheme, standing checks, groups, the level rule, the opening's tests), kept with `/plan`, carried by the opening record; a later check labeled (ruling 8) | PARTIAL (`/plan` records `declared_at`) | 3 | W2 | `plan_lock.py`, `seal.py` |
| P13 | The Recommended final model by ruling 4 (the margin, then the simplest without a concern), named on its own press; the opening its own press | NEW | 2 | W2 | `models/selection.py`, sweep |
| P14 | The period column: never a predictor under a later moment of use; the validation order reads it (internal–external first) | NEW | 2 | W2 | `models/validation.py`, `seal.py` |
| P15 | Each family's own recipe: the curves only for the families that draw lines, by default when Riley's criteria hold (ruling 6) | NEW (C6b's recipes) | 3 | C6b | `models/pipeline.py`, `methods/levers.py`, the shelf |
| P16 | The collinearity noticing before Fit, outcome-blind (the model matrix's condition number), placed in Models' sweep | NEW | 2 | can follow | `models/linear.py:61`, a noticing |
| P17 | Predict's notes and their dispositions, in First look before the draw: the filled values (decide); the laboratory, the zeros, the ages (set for you); unweighted and the unjoined files (noted); gender, the implausible days, the one day of recall (for the record) | NEW | 3 | W2 | `sweep.recommend` (`:519`), `materiality.recommend` |
| P18 | Predict's verdicts by §6's rules: calibration words with the ends, the narrow band, the diabetic range by c and the share at the top, "better than the average in every group", the level shift, arbitrary credit within a group, BBC-CV's noise | NEW | 3 | W2 | a new module beside the views |
| P19 | The Predict paper: two Results paragraphs and the Discussion drafts (at most 90 words each, every number traced), Table 1, the methods in the register, and TRIPOD+AI rules for 6c, 7, 8a, 9a, 9b, 10, 11, 12d, 12f, 14, 15, 16, 20b, 20c, 22, 23a, 23b, 24 and 27a | PARTIAL (export, checklist) | 5.5 | W2 | voice, `export/methods.py`, `export/checklists.py` (overlaps BE12 and BE14) |
| P20 | Under Predict the outcome-scale question reads the training rows, after the draw | NEW | 1 | can follow | `structural.scale_question` (`:126`) |
| P21 | A holdout of whole levels of a period column (the latest cycle), Recommended under Predict when the table has one; its opening | NEW (promoted from "can follow") | 3 | W2 | `seal.py`, `stages/rows.py` (the chronological draw exists; the period reading is new), decisions (after phase 3) |
| P22 | Values filled before the table was made: a flag column linked to its measure; "read as missing" Recommended when the method is not recorded | NEW | 2 | W2 | readings, `repairs.py`, `coach.py` |
| P23 | A per-column SAS-zero reading: a predictor's 0 read as missing where 0 is not a value it can take (BE24) | PARTIAL (`repairs.SAS_ZERO`) | 2 | W2 | `repairs.py` |
| P24 | The NHANES lens's measurement facts, applied before the draw: fasting glucose's laboratories by cycle with the published equations; the age top-codes harmonized | NEW | 3 | W2 | the NHANES lens tables, a repair family |
| P25 | The level for a later visit: `set_updating` "follow the latest period", declared in the plan; its trigger and update on the development rows; applied before the opening and carried by its record | NEW | 3 | W2 | `decision_curve.py` (`updating_sentence`, `:582`), `seal.py`, the evaluation stage (after phase 3) |
| P26 | Openings remembered across projects by the data's fingerprint and the draw, said at a new project's draw | NEW | 2 | can follow | `seal.py`, the workspace |
| P27 | The cycles' summary of MSE or RMSE on the log scale, with the Hartung–Knapp–Sidik–Jonkman interval | NEW | 1 | W2 | `performance.random_effects` (`:882`), `validation.internal_external` |
| P28 | What the intended use needs that the table lacks: the NHANES lens names DIQ and DEMO; the join line in Your data, Recommended; the derived predictors and groups | NEW | 3 | W2 | the NHANES lens, joins, quest |
| P29 | The Fit estimate counts every fit the fit stage makes (inferred from `models/cost.py`: it times the headline's folds only, not the comparison's 10 × 5 refits) | NEW | 2 | W2 | `models/cost.py`, `fit_press.fit_estimate` |
| P30 | Riley's count includes the level columns (73, not 71); C1 is computed as its own minimum, after Fit; a tie names the outcome-free criterion | NEW | 1 | W2 | `models/sample_size.py`, `stages/modeling.py:136` |

- **Totals:** 73 units. W1 holds P1, P2 and P8 (4). C6b holds P15 (3). P16, P20 and P26 can follow
  (5). W2's engine core is the rest: 61 units, against the plan's estimate of 35 for W2's engine
  and frontend together. The orchestrator rules on the cut (§15).
- **Clear of phase 3:** P4 and P10 (new stage files), P15 (C6b), P22, P23, P27, P29, P30. P3 and P7
  share `quest.py` and `voice.py`, so one owner takes both.
- **Shared with Estimate** (§12): P4's group is BE19's marker; P4's cut-off is BE23's lens fact; P12
  is BE22's Predict counterpart; P19 overlaps BE12 and BE14; P7 is BE13; P23 is BE24.

## 11 · Where the engine conflicts with the beats today

- **C1.** The dietary lens proposes the role "exposure" for the nutrients under Predict, so the
  energy-model and substitution questions fire, and Models carries the dietary energy noticing. →
  P3, P11.
- **C2.** The missing-values question ranks "Single fill in each training fold" first; its fill for
  a category writes "taking" into all 15,552 and 17,204 "not asked". → P1 (W1-H).
- **C3.** `set_levers` applies its form to every family: tree families get spline columns, and the
  forest's estimate rises from about 2 hours to about 15. → P15.
- **C4.** The groups are scored on the comparison folds' first repeat (random 5-fold), not on the
  declared scheme (`stages/evaluation.py:323`). → P4.
- **C5.** The curves default to the best family's top three among the "exposure" columns
  (`explain._exposure_inputs`, `:1382`). → P6.
- **C6.** The collinearity concern is said only after Fit, on least squares (`models/linear.py:61`).
  → P16.
- **C7.** The triage under Predict still lists gender's coding and the 501 implausible days as "act
  on it" and the energy noticing as "already answered"; in this arc they are D's lines, before the
  draw. → P17.
- **C8.** `set_selection` is a Decide under Predict even when Riley's criteria hold. → M4's sweep.
- **C9.** Under Predict nothing is fixed at Fit: a check added after the cross-validated scores are
  seen is not labeled. → P12.
- **C10.** The outcome-scale question reads every row's glucose before the draw, the sealed cycle's
  included (`structural.scale_question`). → P20.
- **C11.** The methods are engine prose in the TRIPOD+AI sections. → P19, P7.
- **C12.** Under Predict, Results counts no exhibit, and Write-up reads complete at 0 of 0. → P19
  with C7a.
- **C13 (new).** `set_temporal` is refused when each person appears once ("temporal prediction does
  not arise"), so a period cannot be sealed whole. → P21; the capture emulates it.
- **C14 (new).** `set_updating` knows "none" and "shrinkage" only, so the level rule cannot be
  declared, and the engine's opening scores the model without it. → P25.
- **C15 (new).** The cycles' summary pools MSE on the identity scale with a z interval
  (`performance.random_effects`). → P27.
- **C16 (new).** The Fit estimate was 94 s; the fit took 220 s, ridge alone 100 s against its 21. →
  P29.
- **C17 (new).** The shelf's Riley minimum counts 71 parameters (the model has 73: the two "not
  asked" levels are missing), reads glucose's mean and spread on the development rows before Fit for
  its intercept criterion (C1), starts that criterion at the others' maximum and so names it
  binding on a tie with the shrinkage criterion, which decides it. → P30.
- **C18 (new).** The SAS-zero repair has one option, zero, for every column. → P23.
- **C19 (new).** The `imputed_*` flags are read as flags but never linked to their values: the
  missing-values question sees no blank where 2,444 people's values were filled. → P22.

## 12 · What Predict shares with Estimate (one shell)

| Part | Estimate (`BEATS_MODELS_RESULTS.md`) | Predict (this file) | One component |
|---|---|---|---|
| Quest line, bar, card with tapestry, three levels, Decide · Confirm · For the record | the same | the same | the shell (W1-E) |
| The return | "Because you said" on the partner's card (M2) | J, M1 and M2 quote Q; R3 hands back M1, M2 and M4; R4 hands back M5 | one Return block: kicker, quote with its card, the so-sentence (BE1, BE2) |
| What a choice means, drawn | the compared day | the visit's timeline; the answers' bars; two equal waist steps | the comparison view kind |
| The emblem at the tapestry's head | the comparison | the timeline | one Emblem |
| The data's warnings before the estimate | M6 What your data raised, before the lock | D, before the draw | one noticings card (Decide, Set for you, Noted, For the record) |
| The flowchart with Fit | sealed with the plan: what Results will check | fixed now: what Results will look at, and what the opening tests | one Plan view; BE22 and P12 feed it |
| The commitment | Fit locks the plan (SHA-256) | Fit fixes the evaluation plan; R4 names, then opens | one Commitment line with two modes; "Open the latest cycle" is Predict's only extra primary action |
| What a reviewer will ask | R2's eight questions | R2's three questions | one Checks list (question, verdict, quiet name) |
| Wording and "Put this in my paper" | two paragraphs, at most 90 words, Discussion drafts | the same | one Wording card (BE12, P19) |
| The diagnosis marker | the split by a diagnosis on record | the group "told to take a medicine" and M2's recommendation | one derived marker (BE19, P4) |
| The clinical cut-off | the tail question at 126 mg/dL | R2's ranking and calibration at 126 | one lens fact (BE23, P4) |
| Per-column repairs | the 152 SAS zeros, moving no estimate | the 119 diastolic zeros, read as missing | BE24 = P23 |

What differs: Estimate's finding is one estimate in the person's own comparison; Predict's is a miss
in mg/dL beside a baseline. Estimate decides its draw For the record; Predict draws it in Who's in
and opens it in R4. "What could fool you" is said three times in both.

## 13 · Where each beat reads the capture

| Beat | Capture (`predict-capture/`) |
|---|---|
| Q, J | `capture.before_fit.predictors_all_rows`; `extras.groups.age` (the thirds' counts on the development rows) |
| D | `capture.before_fit.prepared` (`filled`, `bmi_identity_gap`, `waist_filled_above_measured_max`, `diastolic_zeros`, `age_*`, `laboratory`, `glucose_adjusted`) |
| W | `capture.before_fit.split_facts`; `.prepared.cycles` |
| M1 | `capture.before_fit.state_at_end` (roles); `.design` |
| M2 | `capture.before_fit.predictors_all_rows` (answers by level, asked share by cycle, the most common answer) |
| M3 | `capture.before_fit.shelf` (the curves run's costs and rank); `straight-shelf.json` `before_fit.shelf` (the trees' costs); `capture.after_seal.quest_before_opening.fit.estimate_seconds`; `capture.timing` |
| M4 | `capture.before_fit.riley_73`; `.measures_development` (knots, the two steps' shares, the body measures' shared R²) |
| M5 | `capture.before_fit.plan`; `capture.declared_before_fit.level_rule` |
| R1 | `extras.families.*.scores`, `.calibration`, `.by_tenth`, `.prediction_sd`; `extras.outcome`; `extras.summary_log`, `.summary_engine_identity`, `.corr_mse_se` |
| R2 | `extras.families.ridge.tail` (c, the share among estimates at 126 or more, `picked_by_measured`); `extras.groups`; `extras.cycles` |
| R3 | `extras.group_shap`; `capture.after_seal.explain_before_opening.curves` and `.families[].stability`; `extras.added` |
| R4 | `capture.final_rule`; `.after_seal.fit_before_opening.selection`, `.models[].concerns`; `capture.declared_before_opening`; `extras.level` |
| R5 | `opened.json` (`final_as_declared`, `final_without_level`, `tests`, `groups`, `table1`, `secondary`); `capture.after_seal.bundle` (methods, checklist) |

## 14 · Sources to verify into the registry

**Not yet in the registry** (`export/data/citations.json`), to verify with their DOIs: Gneiting 2011
(*J Am Stat Assoc* 106:746); Hoerl & Kennard 1970 (*Technometrics* 12:55); Zou & Hastie 2005 (*J R
Stat Soc B* 67:301); Lei et al. 2018 (*J Am Stat Assoc* 113:1094); Moons et al. 2012 (*Heart*
98:691); Shmueli 2010 (*Stat Sci* 25:289); Riley et al. 2019 (*Stat Med* 38:1262), Riley et al.
2020 (*BMJ* 368:m441) and Sperrin et al. 2020 (*J Clin Epidemiol* 125:183), which the engine cites
but the registry lacks; and, new in Round 1: Bland & Altman 1995, "Comparing methods of measurement:
why plotting difference against standard method is misleading" (*Lancet* 346:1085); Booth et al.
2020, temporal recalibration (*Int J Epidemiol* 49:1316); Janssen et al. 2008, updating methods (*J
Clin Epidemiol* 61:76); IntHout, Ioannidis & Borm 2014, the Hartung–Knapp–Sidik–Jonkman method (*BMC
Med Res Methodol* 14:25); Wolff et al. 2019, PROBAST (*Ann Intern Med* 170:51); Lindström &
Tuomilehto 2003, FINDRISC (*Diabetes Care* 26:725); Bang et al. 2009, the ADA risk test (*Ann
Intern Med* 151:775); the American Diabetes Association's Standards of Care (§2, the fasting
cut-off of 126 mg/dL).

**Already in the registry:** Steyerberg & Harrell 2016, Riley et al. 2016, Collins et al. 2024, Van
Calster et al. 2019, Luijken et al. 2019, Vickers & Elkin 2006, Lundberg & Lee 2017, Apley & Zhu
2020, Molnar et al. 2022, Tsamardinos et al. 2018, Nadeau & Bengio 2003, Harrell 2015, Willett, Howe
& Kushi 1997, Sisk et al. 2023; new in Round 1: Higgins, Thompson & Spiegelhalter 2009
(`higgins2009reeval`), Snell et al. 2018 (`snell2018scales`), Cawley & Talbot 2010
(`cawley2010overfitting`), Gelman & Loken 2013 (`gelman2013forking`), Hosmer, Lemeshow & Sturdivant
2013 (`hosmer2013logistic`), Ingram et al. 2018 (`ingram2018trends`) and the NHANES documentation
(`nhanes_documentation`). Steyerberg 2018 (*J Clin Epidemiol* 103:131) is no longer cited for the
holdout.

**NHANES facts.** Verified on 2026-10-10 from the laboratory documentation: glucose was measured at
the University of Missouri on a Roche Cobas Mira in 2001–2004, at the University of Minnesota on a
Roche/Hitachi 911 in 2005–2006 and a Roche Modular P from 2007, with "no changes" in 2009–2010
(GLU_F) and 2011–2012 (GLU_G); at the University of Missouri from 2013 ("changes to the lab method, lab equipment, and lab
site", GLU_H, no equation), on a Cobas C501 in 2013–2014 and a C311 from 2015 ("2% higher", GLU_I;
"no adjustments are needed" between 2015–2016 and 2017–2018, GLU_J). The equations: Hitachi 911 =
0.9815 × Cobas Mira + 3.5707 (GLU_D); Modular P = Hitachi 911 + 1.148 (GLU_E); C311 = 1.023 × C501 −
0.5108 (GLU_I). Triglycerides: "no changes" in 2017–2018 (TRIGLY_J). Still to verify: the BPQ skip
rules for `meds_hbp` and `meds_chol` in all nine cycles (2015–2016 is verified, E16); the DIQ and
DEMO variables to join (DIQ010, its treatment items, RIDRETH1, INDFMPIR, DMDEDUC2) in every cycle;
the fasting subsample's own weights.

## 15 · Open for Nolan, and for the orchestrator

For Nolan:
1. **What his glucose model is for.** The beats walk the researchers' use. If he means the clinic
   (patients not known to have diabetes, at a draw that left glucose out), J requires the diabetes
   file first, diet is not read, and the medicine answers are read after the join.
2. **The two NHANES files.** Joining DIQ and DEMO is Recommended under either use; the walk could
   not (the files are not in the folder). Their numbers are ⟨after the join⟩.
3. **How the filled values were made.** If he knows the method, and it did not use glucose, D's
   second option keeps them; otherwise they are read as missing, as here.
4. **The tree families.** Fitting them tuned takes hours on his machine: as a job while he is away,
   or in CI. Until then their numbers are ⟨engine-filled⟩.

For the orchestrator: the rulings listed in the return of this round (W2's size, the seal by period,
the level rule, the data preparation before the draw, the joins, and the one-yardstick rule).

## 16 · How the critique was answered (Round 1)

Section numbers are the critique's. "Done" means the beats and the capture now do what the
critique asked, with the numbers of this round's run.

**Methods problems (critique §4)**

| # | Critique point | What Round 1 does |
|---|---|---|
| 1 | Verdicts judged off the declared yardstick | Done: every verdict is a paired difference in squared miss with its interval (§2, §6, ruling 2). The groups: better than the average in every group (all intervals below 0, from −132 to −303 squared mg/dL). The lipids: the most valuable group (+36.7, 25.7 to 47.7; R² 0.155 to 0.125), where the MAE says +0.07 (−0.01 to 0.15). On MAE, curves and diet would look harmful (−0.08, −0.07); on the yardstick they add +10.8 and +7.2 |
| 2 | The tail conditioned on the measured value; a threshold statistic; c and top calibration missing | Done: R2's first line is c 0.82 and 4 in 10 of the estimates at 126 or more; the 56.5 mg/dL miss is a Why? line beside the perfectly calibrated version's 55.6; calibration by tenth with the ends (both 4 mg/dL low). §6 rule 3, P4, P10 and P18 revised |
| 3 | Upstream imputation in 11% of rows | Done: beat D, before the draw; read as missing and filled in each fold without glucose (P22); the false 27a sentence is rewritten. Whether glucose was used upstream cannot be known from the table; D's second option asks the author |
| 4 | "Any adult" without diabetes status; screening drops predictors | Done: beat J (DIQ under either use; DEMO for the groups); M2 under screening says "join DIQ first", then reads the answers (ruling 1). The capture cannot join (the files are absent): the walk declines, said in J |
| 5 | Random holdout headlines a later-use model; no level update; summary and prediction interval promised but unreported | Done: the latest cycle sealed whole (P21 promoted); the level rule declared before Fit (ruling 10), returned in R4, tested at the opening; paragraph 1 opens with the cycles' estimate and its prediction interval |
| 6 | Glucose laboratory changes not harmonized | Done: verified in NHANES's documentation (§14); bridged by its equations in D; the remaining gap (2013's new laboratory) said. Partly disputed (below) |
| 7 | SAS-zero diastolics entered as 0 mmHg | Done: read as missing (D, P23) |
| 8 | Three openings of the same rows; ruling 6's "prespecified" claim | Done: the cited run opens 2017–2018 once in its own project (`extras.py --open` refuses a second run), and the history is declared in the header; ruling 6 now says it was adopted after a cross-validated benchmark; P26 makes the engine remember openings |
| 9 | Random-effects summary on the identity scale, z interval | Done: log scale with the HKSJ interval and t(k − 2) prediction interval (RMSE 32.0, 29.1 to 35.2; a new cycle 24.4 to 42.0), beside the engine's identity-scale 31.8 (prediction interval 23.7 to 38.2); P27 |
| 10 | Race and ethnicity, income, education "not recorded" | Done: "not in your table; NHANES records them" (Q), a join in J, and a Discussion draft that says the groups were not examined |
| 11 | "Diagnosis on record" | Done: "told to take a blood-pressure or cholesterol medicine" throughout |
| 12 | Diet at a clinic visit | Done: M1 under the clinicians' use reads no diet (ruling 13) |
| 13 | Calibration overstated at the ends | Done: §6 rule 2 rules on E90 and the outer tenths; R1 says both ends; the opened cycle's slope 1.14 and top tenth (10.8 low) are reported and drafted for the Discussion |
| 14 | Steyerberg 2018 cited for the random holdout | Done: the lockbox's sources are Cawley & Talbot 2010 and Gelman & Loken 2013 (ruling 3) |
| 15 | Riley 71 vs 73; the shelf reads glucose; C1 binding | Done: 73 terms, 3,916 by the shrinkage criterion (computed with the engine's own function); the shelf's reading said in M3; C17, P30 |
| 16 | "Not identified"; "not measurably worse" by non-significance; Rashomon | Done: "poorly determined"; a stated margin (ruling 4); "credit within a group is arbitrary" (R3) |
| 17 | Table 1, subgroup intervals in ¶2 | Done: Table 1 in R5's tapestry and placement map; ¶2 carries the groups' intervals |

**Ranked failures (critique §3) and density (§6–§8)**

| Critique point | What Round 1 does |
|---|---|
| Deep 1: one yardstick declared, three used | ruling 2 end to end; MAE in Why? and the quiet names only |
| Deep 2: "where it misses" reads regression to the mean as failure | R2 rebuilt on ranking and calibration at the cut; the lesson kept as teaching |
| Deep 3: intended use and moment don't fit; the decisive predictor missing | Q names the pathway; M1 returns it; J joins DIQ; diet follows the setting |
| Deep 4: the data's warnings never reach the person before Fit | D before the draw; R4 keeps only the level and the opening; the gate is not the opening's press |
| Deep 5: Results never returns; the sealed number answers a different question | the seal by period; M5's level rule returned in R4 and tested at the opening; the opening resolves the plan's two tests |
| "No glucose value is read" eight times | once, M1's caption |
| R2 and R3 cards with 11–12 numbers | R1 6, R2 6, R3 5; the rest drawn |
| R4 bundles four commitments | the notes moved to D, the level to M5; naming and opening are two presses |
| M5 asks what M1 settled | M5's validation is a "because" line; its one decision is the level |
| M3 and M4 draw mechanics | M3 draws what each kind can find against its cost; M4 two equal waist steps and one quantity in four measures |
| The opening as a non-event | it tests the prediction interval (24 to 42; found 33) and the level (−0.5 with the update, −3.8 without) |
| "Ranks people about as well (R² 0.12 to 0.21)" | the rank correlation, 0.52 to 0.55 |
| "Diet moves it least" | the groups in order, diet with gender and blood pressure |
| "The final family" before R4 | R1–R3 say "the models" or ridge's numbers as one family's |
| M3's "before any glucose value is read" | removed; the shelf's reading is said |
| "71 terms"; "Fit · about 1 minute"; BBC-CV below the naive score | 73 terms; the estimate (94 s) beside the fit's 220 s; BBC-CV's −5.8 is resampling noise (200 resamples; the three families win 105, 36 and 59 of them) |

**Where the critique was not followed, or was wrong, and why**
- **"That is the laboratory's fingerprint" (Deep 4, the glucose medians).** Partly. NHANES records a
  change in 2005, 2007, 2013 and 2015, and publishes equations for three of them; it records "no
  changes" in 2009–2010 (GLU_F). So the 2007–2008 step and its return in 2009–2010 are not a change
  of method. After the equations, the cycles' medians run 99.3, 98.4, 98.1, 101.0, 99.0, 99.0, 100.8
  and 102.0 mg/dL; 2013–2014, the first cycle at the new laboratory with no equation, came out on
  target (0.0, −1.1 to 1.2), while 2007–2008 (+2.5) and 2015–2016 (+4.0) ran high beyond what the
  recorded changes explain (2015's new instrument is bridged by its equation). The verdict word is
  "shift", said beside the laboratory, not "the laboratory".
- **The coherent use is triage (Deep 3).** Not only. "Not measured" also describes research data
  that recorded a fasting lipid panel but not glucose, where "everyone, diagnosed or not" is
  coherent and diet is recorded; the walk takes that use and offers the clinic triage on the same
  card.
- **"Make the opening test the BBC-CV expectation" (§6, §5).** Replaced. With the latest cycle
  sealed, BBC-CV estimates performance in the development years' mix (RMSE 32.3, 30.6 to 33.9), not
  in a later period; the opening tests the cycles' prediction interval for a new cycle instead,
  which is the expectation for a later period (and the found 33.1 falls inside both).
- **"Weaker within the older and the treated groups" (§5.1).** Superseded by this run's numbers:
  within-group, the youngest are the weakest (R² 0.05 at 36 or younger; 0.08 told to take a
  medicine; 0.09 at 59 and over). R2 now says the misses' range in mg/dL and names the youngest as
  the group it tells apart least.
- **"Unbundle" the final model from the opening.** Kept on one card, as two presses: the opening is
  of the named model (`open_seal` carries the family), so separating them into two screens would add
  a screen without adding a decision.
- **Rashomon "misnamed".** Accepted, with a note: Breiman's Rashomon set does include different
  variable-level stories from equally accurate models, which is what least squares and ridge tell
  inside the body-size and diet groups; but at the level the beats report, the groups, they agree,
  so the precise phenomenon is collinearity, and R3 says "credit within a group is arbitrary".

## 17 · The orchestrator's rulings (2026-10-11)

1. **Scope.** W2's engine work grew from about 35 to 61 units, almost all of it correctness, so it is accepted and split into W2a and W2b. Three items move to "can follow", as the reviser offered:
   - P26 (openings remembered across projects);
   - P28 (DIQ and DEMO joins drawn from the intended use; the card still recommends the joins in words);
   - P24's age part.

   P29 (the fit estimate counts every fit) stays. The hold depends on it, and the capture's estimate of 94 s against 220 s actual shows why.
2. **The seal under Predict.** When a period column exists, sealing the latest period whole is Recommended. A random lockbox is used only otherwise. P21 moves to W2.
3. **The level rule.** Booth et al.'s in-sample re-estimate on the latest period (+3.4), as built. The out-of-cycle shift (+4.0) is a validation residual, and using it as the update would fit to the check.
4. **Data preparation in D.**
   - NHANES's published glucose equations are set for you. They are listed with their sources, and the 2013 gap is stated with its measured agreement.
   - Values filled before the table was made are read as missing, then filled in fold at the median without indicators. Their missingness belongs to the upstream process, not to the person, and does not exist at the moment of use.
5. **DIQ and DEMO joins are Recommended under either use.** DIQ carries the decisive predictor under estimation and the population filter under screening; DEMO carries the subgroup checks.
6. **One yardstick, as revised.**
   - Verdicts are paired differences in squared miss with intervals.
   - c-statistic words follow Hosmer and Lemeshow's bands.
   - "Equally good" uses the stated margin of 5% of the gain over the no-predictor model.
7. **R² on the opened cycle is computed against the cycle's own mean (0.18):** conventional and conservative. The comparison with a model a person could actually deploy is the RMSE against the development mean's RMSE. The calibration slope of 1.14 found after the opening stays a Discussion draft: nothing is recalibrated after the seal.
8. **Every glucose value in this fixture has been read many times.** Nolan's own project draws and opens its seal once, on its own data fingerprint.

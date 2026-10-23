"""The rulings that triage a journey's sentences no registered claim renders
(``research/registry/sentences.py``; CLAIMS_REGISTRY.md §7; R1_PLAN.md R1-B).

A sentence becomes a claim only if it states a number the app relies on, sets a default, a
ranking or a refusal, or teaches a method. Everything else is plain wording: a label, the
person's own answer said back, a reading of this table (its counts and values, what a column
holds), a number measured on these rows by a method stated elsewhere, or interface copy (what a
press does, where a question is, what an answer reopened).

Each :class:`Ruling` is a regular expression searched in a sentence (and, where ``where`` is given,
in one of the places it was shown, as "<source> <path>"), its verdict and why; the first that
matches decides, so the specific come before the general. Two sentences that state one rule must
get verdicts that agree (``sentences.disagreements``: alike but for their values, or one held whole
in the other), and the table refuses where they do not, so a catch-all cannot take a sentence that
holds a claim's. ``"register"`` is a claim to register:
its ``why`` says what it states and where the code writes it, so the next batch can register it
with its tests (the render site is named so a batch knows whose file it is: ``seal.py``,
``teaching/content.py``, ``export/replay.py``, ``models/pipeline.py`` and ``stages/rows.py`` were
the integration fix packages' when this package ran). ``"unrendered"`` is a registered claim's
rule said in other words: the claim is named, and its render site does not render through it.
``"plain"`` is plain wording.

The rulings were written for the NHANES glucose Predict journey (SWARM_PLAN §10 package 6a) by its
builder, the author's judgment: a verifier reads them against the sentences in the table.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Ruling:
    pattern: str
    verdict: str  # "register", "unrendered" or "plain"
    why: str
    claims: tuple[str, ...] = ()
    where: str = ""  # a pattern one of the sentence's places ("<source> <path>") must match


# The plain wording's reasons.
LABEL = "a label: it names an option, a line, a question or a heading"
ANSWER = "the person's own answer said back, or what it recorded"
READING = "a reading of this table: its counts, values or what a column holds"
MEASURED = "a number measured on these rows, by a method another sentence states"
COPY = "interface copy: what a press does, where a question is, or what waits"
STATUS = "says which answer made a result out of date"
EVIDENCE = "the evidence a reading of this table rests on, as measured here"
SOURCE = "names a field or a source, nothing it asserts"
PROVENANCE = "says what was read or written, and where"
OPTION = "what a repair option does on this table, or the sentence it records if chosen"
TAUGHT_OPTION = "what an option does if chosen, as its card says, and nothing it asserts of a method"

# The claims to register, by family: what each states and where it is written.
SPLIT_LABELS = ("the split question's customary and sound labels: a ranking of the validation "
                "schemes with its reasons and sources (custom_sound.SPLIT_ROWS)")
MISSING_LABELS = ("the missing-values question's customary and sound labels: a ranking with its "
                  "reasons and sources (methods/missing.py METHODS; custom_sound.missing)")
MI_LABELS = ("multiple imputation's sub-answers' labels: a ranking with its reasons "
             "(methods/missing.py)")
DETECTION_LABELS = ("the values below a detection limit: each option's customary and sound "
                    "labels, teaching a method with its source (methods/missing.py)")
SCREENS = ("the intake screens' bounds in kcal a day and their sources: numbers the exclusions "
           "rest on (custom_sound.EXCLUSION_ROWS, stages/proposals.py)")
EXPLORE_LEVERS = ("First look's levers: what is customary and what is sound for each, a ranking "
                  "with its reasons (stages/explore.py)")
FAMILIES = ("a model family's inductive bias as the shelf states it: teaches a method "
            "(models/<family>.py)")
WOULD_CHANGE = ("the sweep's rule that a default is a Confirm only where another answer would "
                "change a number here, said for this line (sweep.py, quest.py)")
SET_FOR_YOU = ("why a question is set for you on this table or under Predict: a default and its "
               "reason (quest.py, sweep.py)")
ROLES = ("the roles proposal's reason: a default role and why, read from the column "
         "(stages/roles.py, readings.py)")
ROLE_WAITS = ("BLUEPRINT §14 rule 2: a role proposed below high confidence waits for its own "
              "confirmation, and the fit waits for it (quest.py, readings.py)")
PLAUSIBILITY = ("the physiological limits and the unusual band of a clinical measure, and that "
                "impossible values are repaired, never excluded unasked: numbers and a rule "
                "(detectors/plausibility.py)")
FLAGS = ("an imputation flag describes the data, not the participant, and is no predictor: a rule "
         "with its reason (stages/finding_words.py)")
SUBSTRATE = ("the comparison's substrate, 5-fold cross-validation repeated 10 times, and the "
             "bootstrap's 500 resamples: default numbers (models/validation.py)")
RILEY = ("Riley et al.'s minimum sample size and its four criteria (shrinkage 0.9, optimism "
         "0.05, the mean and the residual SD within 10%): numbers (models/sample_size.py, "
         "models/selection.py)")
BBC = ("bootstrap bias-corrected cross-validation corrects the best family's score for the "
       "choice, by 200 resamples, and a family declared before the opening carries no such "
       "optimism: teaches a method, a number (models/selection.py)")
FINAL = ("the family declared final on cross-validation before the opening is the reported "
         "result, the others secondary: a rule (models/selection.py, voice.py)")
LOCK = ("the plan's lock and the scores withheld until Fit: under Predict nothing locks and no "
        "score is shown before the press, a rule (fit_press.py)")
SEAL_OPEN = ("the held-out rows open once, after the triage, so no noticing is decided after a "
             "held-out score is seen: a refusal (seal.py's open_seal)")
SEAL_PLAN = ("the seal plan's floor of 100 held-out rows, its holdout ranked first at or above "
             "it, and the precision of each share by its stated formula: numbers and a ranking "
             "(seal.py)")
NHANES_WEIGHTS = ("combining NHANES cycles' weights: teaches a method with its source "
                  "(stages/finding_words.py, teaching/content.py)")
SURVEY = ("without the survey's design columns the results describe these participants, not the "
          "surveyed population: teaches a method (stages/finding_words.py)")
SAS = ("a SAS transport zero misread as 16 to the power −65 and why it matters: teaches a method "
       "(repairs.py)")
INTAKE_MARKS = ("the plausible intake of 500 to 5,000 kcal a day the energy column is read "
                "against: numbers (repairs.INTAKE_MARKS, stages/finding_words.py, coach.py)")
BINARY = ("a yes/no column's coding and what it means for every coefficient: teaches a method "
          "(stages/finding_words.py, repairs.py)")
IDENTIFIER = ("an identifier is no predictor: a column unique on every row, or whole numbers too "
              "spread to repeat as a measurement would, is read as a unit's identifier "
              "(stages/finding_words.py, readings.py)")
CV_SCORE = ("what a cross-validated score, its pooling and its standard error estimate: "
            "teaches a method (models/metrics.py, models/validation.py)")
SINGLE_FILL = ("the single fill's defaults: the median, a nutrient from its line on total "
               "energy, the most frequent value, learned in each training fold without the "
               "outcome (methods/missing.py, voice.py)")
ENCODING = ("how categories enter each family: one indicator per level with a reference or every "
            "level, a blank as its own level: a default (models/pipeline.py)")
SCALING = "penalized families center and scale on the training fold: a default (models/pipeline.py)"
RIDGE = ("ridge's penalty chosen along a path of 50 strengths inside the training rows: a "
         "default number (models/ridge.py)")
USUAL_INTAKE = ("what a usual-intake distribution describes and why it changes no prediction: "
                "teaches a method (methods/usual_intake.py)")
ORIENTATION = ("which way round a table is, read from its shape and its first column: a rule "
               "with its threshold (stages/oriented.py)")
GROUPS = ("First look's groups for performance and missingness: named from the header, levels "
          "or thirds by the reading, the 5-point gap at which missingness is called out "
          "(stages/explore.py GROUP_GAP)")
EXPLORE_ROWS = ("First look reads the training rows only and offers each lever first as an "
                "in-fold rule: a rule (stages/explore.py)")
TRIAGE = ("the triage's recommended disposition and why: a ranking (sweep.recommend, "
          "materiality.py)")
NOISE = ("under Predict day-to-day noise in a predictor does not bias the score, so it is For "
         "the record: teaching and a default (sweep.noise_for_the_record, materiality.py)")
REPLAY = "the replay refuses a changed input and compares what it rebuilds: a rule (export/replay.py)"
EXPORT_RULE = ("how the bundle's methods are written: answers changed before anything was seen "
               "folded out, changes after the opening marked: a rule (export/methods.py)")
TASK = "the outcome's task read from its values: a default (stages/target.py)"
UNIT = ("an energy column's unit read by the Atwater identity, with its ratios: a rule "
        "(stages/proposals.py, readings.py)")
CALIBRATION = ("the calibration curve's smoother, lowess with span 2/3 and no robustness "
               "iterations as rms val.prob draws it: a default (models/performance.py)")
VALIDATION_SIZE = ("at or above validation.resampling_first's 20,000 units one cross-validation "
                   "run ranks first: a ranking, its threshold that claim's number "
                   "(models/validation.py, also typed there as RESAMPLE_BELOW)")
PURPOSE = "what the declared purpose means for how models are judged: teaches a method (voice.py)"
DESIGN_ROUTE = ("a family whose structure the table cannot use is fit as the plain model, or "
                "adds nothing to a prediction: a rule (models/repeated.py, the shelf)")
SPLIT_BY_ROW = "the seal's basis where each unit appears once: a rule (seal.decide_basis)"
CYCLE_ROLE = ("what the survey cycle does as a covariate or as a time column: teaches a method "
              "(stages/finding_words.py)")
HOLDOUT_INTERVAL = ("the held-out score's interval, by its influence function on the held-out rows: "
                    "teaches a method (models/baseline.py tolerance_basis, models/performance.py)")
OUTCOME_WITHHELD = ("the outcome's distribution is withheld until eligibility is answered, and no "
                    "eligibility rule reads the outcome, so no restriction is chosen by looking at "
                    "it: a refusal (outcome_gate.py, teaching/content.py)")
STATED_IN_METHODS = ("the methods' account of what the values settled holds the identifier rule "
                     "(IDENTIFIER), the roles proposal's reasons (ROLES) and the outcome's task read "
                     "from its values (TASK), each a claim to register (export/methods.py, "
                     "readings.py)")
CONSUMER = "names the parts of the analysis that read the answer: interface copy"
DEFINITION = "a term's definition that says what a word names, and asserts nothing of a method"

# The teaching entries (``GET /teaching``) of the questions the journey answered, and the repairs
# entry its noticings draw: all written in teaching/content.py (a fix package's file when this
# package ran), the families' options from models/<family>.py.
T_LENS = ("what a lens does: the checks it runs and the choices it puts first, never hiding an "
          "option: a default and a rule (teaching/content.py, lenses.py)")
T_ELIGIBILITY = ("eligibility is applied to every row before the seal and reported in the "
                 "participant flow with its count, and a cut chosen on the outcome is selection: "
                 "a rule and teaching (teaching/content.py, stages/rows.py)")
T_INTAKE = ("implausible intake and under-reporting: what they are, and that under-reporting "
            "concentrates in people with higher BMI: teaches a method (teaching/content.py, "
            "NUTRITION_PACK)")
T_MISSING = ("what complete cases, a single fill, multiple imputation and missing at random "
             "assume and cost: teaches a method (teaching/content.py)")
MI_DEFAULT = ("multiple imputation under inference: 20 imputations with the outcome, pooled by "
              "Rubin's rules: a default number (methods/missing.py, teaching/content.py)")
T_MODELS = ("what a family assumes and reports, why several are fit, shrinkage and calibration: "
            "teaches a method (teaching/content.py)")
FAMILY_OPTIONS = ("a model family's option on the models card: its inductive bias and what it "
                  "reports, teaching a method (teaching/content.py family_options, from "
                  "models/<family>.py)")
SEALED = ("the held-out scores are withheld until the seal is opened, each model scored once on "
          "them, and the scores fixed in the record once seen; every choice is tuned on the rest: "
          "a rule and teaching (seal.py, fit_press.py, teaching/content.py)")
T_PURPOSE = ("what each purpose means for how models are judged and reported, and the shelf's "
             "order it sets: teaches a method and sets a ranking (teaching/content.py, voice.py)")
T_REPAIRS = ("a repair before the seal reads only its own row, so it leaks nothing and runs on the "
             "table now, and what is learned from many rows waits for the training folds; a "
             "sentinel code is a missing answer: a rule and teaching (teaching/content.py, "
             "repairs.py)")
T_ROLES = ("which roles enter the models: three predictor roles, every other role kept out of the "
           "predictors while the record keeps it: a rule (teaching/content.py, stages/roles.py)")
T_SPLIT = ("the seal's and cross-validation's methods taught: a holdout's share and its trade-off, "
           "folds, repeats, a grouped split, optimism and a seed: teaches a method and states its "
           "numbers (teaching/content.py)")
T_TARGET = ("rows without the outcome leave the analysis first, so the outcome sets the participant "
            "flow's first count: a rule (stages/rows.py, teaching/content.py)")
T_ENERGY_OUTCOME = ("an energy-related outcome makes adjusting for reported energy need extra care: "
                    "teaches a method (teaching/content.py)")

RULINGS: tuple[Ruling, ...] = (
    # ── registered claims' rules in other words ──
    Ruling(r"^Sound: a near-collinear pair does not harm prediction on rows like these", "unrendered",
           "the collinear lever's soundness under Predict: the claim's rule in the lever option's "
           "words (stages/explore.py)", ("materiality.collinear_under_prediction",)),
    Ruling(r"^How many days each value spans is not settled: .*record it before screening by "
           r"`[^`]+`\);.*is in grams \(its kcal per unit is its Atwater factor\)", "unrendered",
           "the methods' account of what the values settled: the screens wait for the energy "
           "column's days and a macronutrient's grams convert by its Atwater factor, the claims' "
           "rules in other words; it holds the Atwater identity's ratios too (UNIT, to register) "
           "(export/methods.py, readings.py)",
           ("exclusions.screen_waits_for_days", "substitution.isocaloric_units")),
    Ruling(r"^How many days each value spans is not settled: .*record it before screening by "
           r"`[^`]+`\)(;|\.$)", "unrendered",
           "the methods' account of what the values settled: the screens wait for the energy "
           "column's days, the claim's refusal in other words (export/methods.py, readings.py)",
           ("exclusions.screen_waits_for_days",)),
    Ruling(r"^Under prediction, a fill learned in each training fold without the outcome lets|"
           r"^Under prediction: blanks filled in each training fold without the outcome", "unrendered",
           "a fill learned in each training fold without the outcome scores a new row: the claim's "
           "rule in the missing-values entry's words (teaching/content.py)",
           ("missing.fold_fill_first_under_prediction",)),
    Ruling(r"^The answer depends on the purpose: inference imputes with the outcome, prediction "
           r"without it", "unrendered",
           "the missing-values entry's one line: the two claims' rules, the fill without the "
           "outcome under prediction and multiple imputation with it only under inference "
           "(teaching/content.py)",
           ("missing.fold_fill_first_under_prediction", "missing.mi_refused_unless_inference")),
    Ruling(r"^Under inference, multiple imputation fills each blank many times from the other "
           r"variables, the outcome included", "unrendered",
           "multiple imputation uses the outcome, so it is offered only under inference: the "
           "claim's rule in the missing-values entry's words (teaching/content.py)",
           ("missing.mi_refused_unless_inference",)),
    Ruling(r"^Rows at or after a time value you name are sealed, each unit kept whole", "unrendered",
           "the cut-point's rule in the split entry's words (teaching/content.py); it says each "
           "unit is kept whole unconditionally, which holds only where the seal groups a unit's "
           "rows (seal.cut_point beside seal.cut_point_units): listed as a defect",
           ("seal.cut_point_units",)),
    Ruling(r"^The seal states its basis: grouped by an identifier, repeats found but not grouped, "
           r"or undetermined, which is labeled exploratory", "unrendered",
           "a seal that could not keep a unit's rows together is exploratory: the claim's rule in "
           "the split entry's words (teaching/content.py)", ("seal.by_row_exploratory",)),
    Ruling(r"^Rows at levels you name are excluded, as are blanks and refused or don't-know codes "
           r"named", "unrendered",
           "a refused or don't-know answer cannot confirm a row eligible: the claim's rule in the "
           "eligibility entry's option (teaching/content.py)", ("eligibility.unconfirmed_answer",)),
    Ruling(r"^Better only when the gain's 95% interval lies above zero", "unrendered",
           "the baseline floor's rule, composed beside the fold's tolerance "
           "(models/baseline.py tolerance_basis), not the claim's words",
           ("baseline.min_gain",)),
    Ruling(r"not distinguishable on these rows|^Each pairwise interval is descriptive|"
           r"^The interval is corrected for the \d+ folds sharing training rows|"
           r"whether two families differ is read from their paired interval", "unrendered",
           "the pairwise interval's method and verdict, which the claim's teaching states "
           "(models/validation.py, models/baseline.py)", ("validation.paired_interval",)),
    Ruling(r"by the corrected repeated k-fold t \(Nadeau & Bengio 2003; Bouckaert & Frank "
           r"2004\)", "unrendered",
           "the split's methods sentence names the comparison the claim teaches "
           "(voice._set_split)", ("validation.paired_interval",)),
    Ruling(r"^Sound for regression-type families; a near-interpolating one", "unrendered",
           "the bootstrap's exception for near-interpolating families, the claim's second "
           "clause (seal.py's validation option, models/validation.py)",
           ("validation.resampling_first",)),
    Ruling(r"^The model will be used on everyone, misreporters included|"
           r"^The people it removes are still among those the model will be used on", "unrendered",
           "the every-row option's and the screens' soundness under Predict: the claim's ranking "
           "in each option's words (custom_sound.EXCLUSION_ROWS)",
           ("exclusions.every_row_first_under_prediction",)),
    Ruling(r"^Valid but wasteful, and a new row with a blank cannot be scored", "unrendered",
           "complete cases' soundness under Predict: the claim's reason in the option's words "
           "(methods/missing.py)", ("missing.fold_fill_first_under_prediction",)),
    Ruling(r"^How many days each value spans is not settled: its median, [\d,.]+, is an adult's "
           r"day's intake \(a child's total over two days sits there too\); record it before "
           r"screening by `[^`]+`\.$|^Which rows are implausible depends on it, so none is counted "
           r"as a misreport until it is recorded, and the intake screens wait for it", "unrendered",
           "the screens wait for the energy column's days: the claim's refusal in the reading's "
           "words (stages/proposals.py, stages/finding_words.py)",
           ("exclusions.screen_waits_for_days",)),
    Ruling(r"blank on `[\d.]+%`: likely a question not asked|"
           r"is two-level text with `[\d,]+` of `[\d,]+` blank; a blank may mean not asked",
           "unrendered", "a blank answer read as not asked: the claim's reading in the coach's "
           "and the finding's words (coach.py, stages/finding_words.py)",
           ("missing.blank_not_asked",)),
    Ruling(r"^Whole numbers with `\d+` values: if they are codes for groups, one slope across "
           r"them means nothing", "unrendered",
           "whole numbers may be codes or amounts: the claim's rule in the roles card's words "
           "(stages/roles.py)", ("readings.whole_numbers_wait",)),
    Ruling(r"^Not here: it uses the outcome, which a new row does not have", "unrendered",
           "multiple imputation uses the outcome a new row lacks: the claim's refusal in the "
           "option's words (methods/missing.py)", ("missing.mi_refused_unless_inference",)),
    Ruling(r"^is in grams \(its kcal per unit is its Atwater factor\)", "unrendered",
           "a macronutrient's grams convert by its Atwater factor: the claim's rule in the "
           "reading's words (readings.py)", ("substitution.isocaloric_units",)),

    # ── claims to register ──
    Ruling(r"by analogy with the 100-event rule|at or above the floor of \d+ rows|"
           r"held-out rows: R² known to about ±|^Widths are approximate 95% intervals",
           "register", SEAL_PLAN),
    Ruling(r"^With [\d,]+ rows, one cross-validation run is precise", "register",
           VALIDATION_SIZE),
    Ruling(r"^Held out by row: every `[^`]+` appears once", "register", SPLIT_BY_ROW),
    Ruling(r"^At this size the held-out rows measure precisely", "register",
           "the holdout's verdict where the seal plan ranks it first: a ranking "
           "(custom_sound.split, from seal.plan's order)"),
    Ruling(r"refits of each family|10 × 5-fold repeats|compared on 5-fold cross-validation "
           r"repeated 10 times|^The folds are drawn 10 times|500 whole-pipeline refits",
           "register", SUBSTRATE),
    Ruling(r"^Every row trains and is scored by cross-validation|^One partition into folds|"
           r"^Each site,? (study )?or period scored by models fit on the others", "register",
           SPLIT_LABELS, where=r"^artifact:seal_plan "),
    Ruling(r".", "register", SPLIT_LABELS, where=r"^artifact:seal_plan labels\.options\[\]\."
           r"(customary\.text|sound\.reason)$"),
    Ruling(r"^lowess, span 2/3", "register", CALIBRATION),
    Ruling(r"^Riley et al\.'s minimum sample size|^an expected shrinkage of at least|"
           r"^an optimism of at most|^the (mean outcome|residual SD) within a \d+% margin",
           "register", RILEY),
    Ruling(r"flatters the winner|corrected for the choice by BBC-CV|^The held-out score of a "
           r"family declared before the seal is opened carries no such optimism", "register", BBC),
    Ruling(r"declared the final model on cross-validation|^The held-out rows score the final "
           r"model|its held-out score is the reported result", "register", FINAL),
    Ruling(r"^At the opening \(decision #\d+\)", "register",
           "the held-out score at the opening, said as the reported result: the final model's "
           "rule (FINAL) beside its number (models/selection.py, voice.py)"),
    Ruling(r"^The final model is not declared: the held-out rows are still sealed, so no held-out "
           r"score is the result yet|^Declare the final model on cross-validation, then open them "
           r"once", "register", FINAL),
    Ruling(r"^The interval is the held-out R²'s \(influence function", "register", HOLDOUT_INTERVAL),
    Ruling(r"^The outcome · opens after Who's in$", "register", OUTCOME_WITHHELD),
    Ruling(r"^The outcome · its distribution is shown in its own view$", "plain", COPY),
    Ruling(r"^Read from the values, no question asked: ", "register", STATED_IN_METHODS),
    Ruling(r"proposed as \w+ at \w+ confidence, below high, for one reason \(\"(Not a nutrient or "
           r"feature: read as a|A nutrient that carries energy;|Total energy intake: its values)",
           "register", "the roles proposal's reason, quoted in the answer said back: a default "
           "role and why (ROLES; stages/roles.py, voice.py)"),
    Ruling(r"^Read as binary, \w+ becomes 1 and \w+ becomes 0\.$", "register",
           "a two-level column's default coding, which level becomes 1: a default (BINARY; "
           "stages/finding_words.py, repairs.py)"),
    Ruling(r"^Cross-validated (MSE|R²|AUC|log loss|RMSE|MAE) [\d.]+ \(95% interval", "register",
           CV_SCORE),
    Ruling(r"^Cross-validated R², MSE, RMSE and MAE pool every out-of-fold prediction|"
           r"^Each standard error counts which rows were scored", "register", CV_SCORE),
    Ruling(r"^The fold values show the spread", "plain", COPY),
    Ruling(r"^Under prediction nothing locks|^Pressing Fit opens Results|^No score or estimate is "
           r"shown until Fit is pressed|^Nothing is locked and no estimate is shown|"
           r"^The held-out rows are sealed: no held-out score is served", "register", LOCK),
    Ruling(r"^Fit was pressed, so the scores are shown", "plain", COPY),
    Ruling(r"^The held-out rows open once, after every open noticing|^Act on each where it is "
           r"decided, or record it as a limitation|^Triage them in Results first", "register",
           SEAL_OPEN),
    Ruling(r"^(The held-out rows|The rows the model is scored on) carry the same day-to-day "
           r"noise|^Part of `[^`]+`'s spread is day-to-day noise", "register", NOISE),
    Ruling(r"^It concerns `[^`]+`, which the analysis reads; left as it is, it could bias the "
           r"score", "register", TRIAGE),
    Ruling(r"outside \d+–\d+ mmHg, limits no living adult is expected to show|"
           r"has `\d+` impossible values outside `\d+`–`\d+`|^This is different from the \d+ adult "
           r"values outside|records a diastolic pressure of 0 on purpose|holds impossible values "
           r"and unusual ones, and they are different categories|^Physiologically impossible and "
           r"statistically extreme are different categories|^The impossible values are a "
           r"data-quality repair|^Every limit here is a convention|^Excluding them is an "
           r"eligibility criterion|^Nothing is filtered unless you say so", "register",
           PLAUSIBILITY),
    Ruling(r"flags `[\d,]+` imputed values of `[^`]+`; it describes the data|^It marks where "
           r"`[^`]+` was imputed|^As a flag it stays in the record|^As a predictor, a flag lets a "
           r"model learn", "register", FLAGS),
    Ruling(r"^Named like a flag on `[^`]+`, but `[^`]+` has no blanks|^Named like a time, but no "
           r"unit repeats here|^Not a nutrient or feature: read as a|^A person's characteristic, "
           r"usually adjusted for|^A nutrient that carries energy|^Total energy intake: its values "
           r"follow", "register", ROLES),
    Ruling(r"^`[^`]+` is the NHANES respondent number|^`[^`]+` is unique on every row: it names a "
           r"participant and is not a predictor|^A value that appears once cannot tell a model|"
           r"^As a predictor it would be noise at best", "register", IDENTIFIER),
    Ruling(r"was proposed below high confidence and recorded with the other roles|"
           r"were proposed below high confidence for one reason and recorded with the other roles|"
           r"^As a flag on another column it enters no model, but the fit waits for it",
           "register", ROLE_WAITS),
    Ruling(r"^(A causal estimator|A modifier such as|A trial or a matched case-control|An in-fold "
           r"rule|Another form changes|Another reading changes|Another scheme|Who counts as one "
           r"person)", "register", WOULD_CHANGE),
    Ruling(r"^Each unit appears once, so|^every `[^`]+` appears once, so each person is one row|"
           r"^(Under prediction no coefficient is read as an effect|Under prediction no effect is "
           r"estimated|A substitution estimates the effect|The energy model decides what a "
           r"nutrient's coefficient means|The outcome is read as regression, so|No assay lens is "
           r"on|No column reads as a survey weight)|^no column reads as a site|^Under prediction "
           r"each predictor enters as each family takes it", "register", SET_FOR_YOU),
    Ruling(r"^Combining NHANES cycles|is the exception: its two-year weights", "register",
           NHANES_WEIGHTS),
    Ruling(r"^NHANES deliberately oversamples|results describe these participants, not the US "
           r"population|^If the export dropped them, export them again", "register", SURVEY),
    Ruling(r"^A SAS transport file stores numbers in IBM floating point|^Left as they are, these "
           r"are tiny positive amounts", "register", SAS),
    Ruling(r"rows fall outside `?\d+`?–`?[\d,]+`?|^These are possible days and poor estimates",
           "register", INTAKE_MARKS),
    Ruling(r"^Which level is the positive one is your call|^Coercing it to a number instead gives "
           r"the same digits|the column keeps its meaning: a rate, an odds ratio", "register",
           BINARY),
    Ruling(r"^As a covariate, the cycle absorbs level differences", "register", CYCLE_ROLE),
    Ruling(r"^Median for numbers; .* from their line on|^Blanks in categorical and yes/no "
           r"predictors were kept as a level of their own", "register", SINGLE_FILL),
    Ruling(r"become indicator columns", "register", ENCODING),
    Ruling(r"^Centers and scales all \d+ columns on the training fold", "register", SCALING),
    Ruling(r"^Pulls every coefficient toward zero by one penalty, chosen along a path of \d+",
           "register", RIDGE),
    Ruling(r"^Fits the straight-line effect of every column", "register", FAMILIES),
    Ruling(r".", "register", FAMILIES, where=r"^artifact:shelf families\[\]\.inductive_bias$"),
    Ruling(r"^No identifier repeats in these rows, so there is no|^Tests each study factor on its "
           r"own and makes no predictions|^With fewer features than rows the penalty alone",
           "register", DESIGN_ROUTE),
    Ruling(r"^Multiple imputation \(m at least \d+", "register", MISSING_LABELS),
    Ruling(r".", "register", MISSING_LABELS, where=r"^artifact:proposals (labels\.missing\.options"
           r"\[\]\.(customary\.text|sound\.reason)|missing\.methods\[\]\.(customary|sound))$"),
    Ruling(r".", "register", MI_LABELS,
           where=r"^artifact:proposals missing\.imputation_models\[\]\.(customary|sound)$"),
    Ruling(r".", "register", DETECTION_LABELS,
           where=r"^artifact:proposals missing\.below_detection\[\]\.(customary|sound)$"),
    Ruling(r"kcal a day|^Fixed kcal cut-offs|^Goldberg: energy over Schofield BMR|^implausible "
           r"(energy reports|intakes)", "register", SCREENS),
    Ruling(r".", "register", SCREENS,
           where=r"^artifact:proposals labels\.exclusions\.options\[\]\.(customary\.text|"
                 r"sound\.reason)$"),
    Ruling(r".", "register", EXPLORE_LEVERS,
           where=r"^artifact:explore findings\[\]\.lever\.options\[\]\.(customary|sound)$"),
    Ruling(r"^Explore read the [\d,]+ training rows only|^Checked on the training rows once "
           r"answered", "register", EXPLORE_ROWS),
    Ruling(r"^Missing predictors (are similar|differ by)|^Named like (age|sex): a proposal from "
           r"the header|^Its groups are its levels if its numbers are codes|'s groups wait on "
           r"whether its numbers are codes or amounts", "register", GROUPS),
    Ruling(r"^The outcome's scale and levels: answered with the outcome; a re-seal to change them",
           "register", "the outcome's scale changes after the seal only by a re-seal: a refusal "
           "(stages/explore.py hand levers)"),
    Ruling(r"^(A usual-intake distribution describes a population|An association between an "
           r"intake and an outcome is a different question|The distribution of usual intake|"
           r"Under prediction the model reads each person's recalls|Consumers only needs a column|"
           r"Whole population or consumers only)", "register", USUAL_INTAKE),
    Ruling(r"^Measured on a log scale, the row means spread|^The first column, `[^`]+`, holds a "
           r"different whole number on every row", "register", ORIENTATION),
    Ruling(r"^Continuous, with [\d,]+ distinct values — read as a regression outcome", "register",
           TASK),
    Ruling(r"^the Atwater identity holds with total energy", "register", UNIT),
    Ruling(r"^Replaying it with `python -m turbotab\.replay`", "register", REPLAY),
    Ruling(r"^_?Each sentence below is the decision record's own", "register", EXPORT_RULE),
    Ruling(r"^The analysis was declared for `prediction`: models are judged on rows they never "
           r"saw", "register", PURPOSE),

    # ── the teaching entries the journey's cards drew (GET /teaching) ──
    Ruling(r".", "plain", CONSUMER, where=r"^teaching:\w+ consumer$"),
    Ruling(r".", "plain", LABEL, where=r"^teaching:\w+ title$"),
    Ruling(r"^Several can apply at once|^Pick all that apply", "plain", COPY,
           where=r"^teaching:lens "),
    Ruling(r"^It changes what TurboTab checks and suggests, never what you can do|^A lens runs that "
           r"field's checks|^It never hides a column, a model or an option|^Each field reads a table "
           r"its own way|^A field's way of reading a table", "register", T_LENS),
    Ruling(r".", "register", T_LENS, where=r"^teaching:lens options\[\]\.consequence$"),
    Ruling(r"^A restriction comes from your research question, so the outcome's distribution is not "
           r"shown|^Rows outside a range you set on a numeric column, never the outcome",
           "register", OUTCOME_WITHHELD),
    Ruling(r"^No row is excluded; the record states that no exclusion was applied", "plain", TAUGHT_OPTION),
    Ruling(r"^Energy over estimated BMR outside the cut-offs|^A screen comparing reported energy "
           r"with estimated basal metabolic rate", "register", SCREENS),
    Ruling(r"^A reported day's energy intake too low or too high to be a real diet|^It is "
           r"systematic, concentrated in people with higher BMI|^Implausible-intake screens are "
           r"offered with the rows each removes", "register", T_INTAKE),
    Ruling(r"^Reporting less than was eaten", "plain", DEFINITION),
    Ruling(r"^Who the study is about, such as an age range: applied to every row before the seal|"
           r"^A criterion such as an age range changes N|^A cut chosen by looking at the outcome is "
           r"data-driven selection", "register", T_ELIGIBILITY),
    Ruling(r"^Rows missing any predictor are dropped; the participant flow shows how many", "plain",
           TAUGHT_OPTION),
    Ruling(r"^Under inference: each blank imputed \d+ times with the outcome", "register",
           MI_DEFAULT),
    Ruling(r"^Analyzing only rows with every variable measured; it can bias|^Filling a missing value "
           r"with an estimate learned from observed rows|^Filling each blank several times from its "
           r"predicted distribution|^Missingness that the observed columns explain|^Complete cases "
           r"keep only rows with every predictor measured", "register", T_MISSING),
    Ruling(r"^Each family brings its own assumptions|^Shrinking coefficients toward zero|^What a "
           r"model assumes before it sees data|^Whether predicted probabilities match observed "
           r"frequencies|^A linear model assumes straight-line, additive effects|^An elastic net "
           r"shrinks correlated nutrients together|^Boosted trees find curves and interactions|"
           r"^In one comparison of [\d,]+ models", "register", T_MODELS),
    Ruling(r".", "register", FAMILY_OPTIONS, where=r"^teaching:models options\[\]\.consequence$"),
    Ruling(r"^The held-out scores exist but are withheld until now|^Each model is scored once on the "
           r"held-out rows; those scores are fixed|^Held-out rows give one honest number only if|"
           r"^Once opened, the scores stay in the record|^Sealed rows are scored once, when you open "
           r"the seal|^The line between the held-out rows and the rest: drawn before modeling",
           "register", SEALED),
    Ruling(r"^The Results stay on cross-validation; the held-out rows stay unread|^No rows are "
           r"sealed; every score comes from cross-validation on all rows", "plain", TAUGHT_OPTION),
    Ruling(r"^A change made after the held-out scores were seen; the record and the manuscript mark|"
           r"^A later change still refits every model, but its sentence and the manuscript mark",
           "register", EXPORT_RULE),
    Ruling(r"^Prediction is judged on rows the model never saw|^Linear models report coefficients "
           r"with confidence intervals; the shelf favors|^Models are ranked by performance on rows "
           r"they never saw|^Estimating how the outcome relates to a study factor|^Estimating the "
           r"outcome for new people|^The exact quantity an analysis estimates|^Inference reports "
           r"coefficients with confidence intervals|^Some advice even flips|^The two lead to "
           r"different models", "register", T_PURPOSE),
    Ruling(r"^A repair fixes how a value is written, row by row|^A number standing for an answer "
           r"that is not a value|^Computed from one row's own cells, so it leaks nothing|^A sentinel "
           r"code such as \d+ on a|^Anything learned from many rows waits for the training folds|"
           r"^Repairs that look only at their own row run on the table now", "register", T_REPAIRS),
    Ruling(r"^Nothing changes; the record keeps that it was seen and set aside|^The changed cells "
           r"are written into the working table|^The finding waits inside the question it belongs "
           r"to", "plain", TAUGHT_OPTION),
    Ruling(r"^A column that names a row or a person rather than measuring anything|^An identifier "
           r"names a person and cannot generalize", "register", IDENTIFIER),
    Ruling(r"^Three roles are predictor roles|^None of them measures the participant, so each stays "
           r"out|^Survey weights and design columns describe how people were sampled|^A column "
           r"measured on each person that could predict the outcome|^A predictor that describes the "
           r"person", "register", T_ROLES),
    Ruling(r".", "register", T_ROLES, where=r"^teaching:roles options\[\]\.consequence$"),
    Ruling(r"^A small holdout measures little, so with few rows cross-validation alone comes first",
           "register", SEAL_PLAN),
    Ruling(r"^(Nearly a third|One row in (five|ten)) is sealed|^A split that keeps all of one "
           r"participant's rows|^Cross-validation drawn again on fresh folds|^Each site, study or "
           r"period held out in turn|^How much better a model scores on the rows it learned from|"
           r"^One of the equal parts cross-validation splits|^Rows set aside before any modeling and "
           r"scored once|^Splitting the training rows into folds|^The number that fixes the random "
           r"draw|^A model scored on the rows it was tuned on grades its own homework", "register",
           T_SPLIT),
    Ruling(r"^Rows without the outcome leave the analysis first", "register", T_TARGET),
    Ruling(r"^When the outcome is itself energy-related", "register", T_ENERGY_OUTCOME),
    Ruling(r"^Everything downstream is built around it|^It also decides the task, the metrics",
           "plain", CONSUMER),
    Ruling(r"^The column a model learns to predict or explain|^The count of rows at each step, from "
           r"the file to the analysis", "plain", DEFINITION),

    # ── the checklist and the methods' headings ──
    Ruling(r".", "plain", LABEL, where=r"^(quest:checklist title|methods sections\[\]\.title)$"),
    Ruling(r"(is|are) not answered yet; the methods would leave (it|them) out\.$|^The (fit|model matrix|"
           r"participant flow) (waits for .* before it can run|is still being computed for the "
           r"current answers)\.$", "plain", COPY),

    # ── plain wording ──
    Ruling(r"^Your change to .* made \d+ results? in .* out of date", "plain", STATUS),
    Ruling(r"^(Move|Confirm) `[^`]+` (to the|as a)|^Confirm these \d+ as|^Confirm each of the \d+ "
           r"as shown|^(Decide|Record as a limitation): |^Keep the recommended dispositions|"
           r"^Triage the open noticings in Results|^Score performance within `[^`]+`'s groups|"
           r"^A spline for `[^`]+` alone, by hand", "plain", LABEL),
    Ruling(r"^`[^`]+` (holds codes for categories \(one indicator per level|is a count or an "
           r"amount \(a slope|is left out of the models$|is in (kcal|kJ|kilograms|grams|m|in|cm)"
           r"(, one day's intake)?$|is a total over \d+ days, in \w+$|is a sample$|names the "
           r"features$)|^`gender`: ", "plain", LABEL),
    Ruling(r"^the (role|roles) of `|^the (question|held-out rows question|model families question|"
           r"rules repeated inside each fold)|^what your data settled|^The \d+ choices set for you|"
           r"^The choice set for you", "plain", LABEL),
    Ruling(r"^(proposed as a|is a flag on another column|is an amount \(one slope\)|is coded |"
           r"is in grams|is the unit's identifier)", "plain", READING),
    Ruling(r"^\d+ columns (confirmed|proposed) as characteristics at \w+ confidence", "plain",
           ANSWER),
    Ruling(r"^Confirm (all \d+|both) in one press|^Confirm it after reading why it was proposed|"
           r"^These columns were settled by their values|^Tell me about these columns|"
           r"^Say what it is first|^To act on it instead, change that answer there|^It is decided "
           r"(at|in) |^Confirm the same dietary methodology applies across cycles|^No earlier fit "
           r"is recorded|^Opens once you decide which rows to hold out|^This is a comparison of its "
           r"own|^The \d[\d,]* blank\(s\) are a separate question", "plain", COPY),
    Ruling(r"\?$", "plain", "a question the card asks"),
    Ruling(r"^(How the outcome moves with `[^`]+`|The outcome's distribution) on the training "
           r"rows; opening it is recorded", "plain",
           "names a view; that opening it is recorded is the seal's rule, said by the view's "
           "own claim when it is registered"),
    Ruling(r"^\(`[\d,]+` whole-number values from |^`[\d,]+` values with decimals that fill their "
           r"grid|^its median, [\d.,]+, is|^the labels spell the sexes|^`[^`]+` matches the energy "
           r"its macronutrients carry", "plain", EVIDENCE),
    Ruling(r"^`[^`]+` was confirmed as a|were each confirmed as a flag on another column|"
           r"^Confirmed together, each as the question showed it|were confirmed together as one "
           r"group|^Column roles were set for|^`[^`]+` was chosen as the outcome|^The table was "
           r"read through|^No predictor selection was applied|^No rows were excluded|^Two model "
           r"families were chosen|^The model was declared for estimating|^The open noticings were "
           r"triaged in Results|^Predictor forms: none|^the purpose is prediction", "plain", ANSWER),
    Ruling(r"blanks were recoded as `False`, on the assumption that a person not asked would have "
           r"answered no", "register",
           "the not-asked repair's assumption that a person not asked would have answered no: a "
           "default (repairs.py, stages/finding_words.py)"),
    Ruling(r".", "plain", OPTION, where=r"^artifact:findings findings\[\]\.repairs\[\]\."
                                        r"(consequence|sentence)$"),
    Ruling(r"^(`[^`]+` (has|holds|is|names|pools|becomes|leaves|was|were|flags|marks|reads)\b|"
           r"`[\d,]+` (cells|blanks|impossible|rows|physiologically|rows are held out)|`[\d,]+` "
           r"rows (holding|with)|\d+ of `[^`]+`'s values are 0|\d+ columns are recognized|Each "
           r"cycle holds|Observed range|It is a binary variable|Its values are true and false|The "
           r"blanks are why pandas|The days in `|This NHANES table has no survey weight|No survey "
           r"weights or design columns\.$|Read all [\d,]+ rows|Read [\d,]+ rows and \d+ columns|"
           r"Column summaries and lens hints read|Counted across all|Ranked for [\d,]+ training rows|"
           r"no value for the outcome|one row per unit$|Rows in the table)", "plain", READING),
    Ruling(r"^`[^`]+` and `[^`]+`: r = |^`\d+` columns hold SAS export zeros as", "plain", READING),
    Ruling(r"^one-hot, blank as a level$", "plain", LABEL),
    Ruling(r"^The analysis was run in TurboTab|^The supplementary provenance record holds|"
           r"^Sections follow TRIPOD\+AI", "plain", PROVENANCE),
    Ruling(r"^(Methods: |Missing data \(TRIPOD|Analytical methods \(TRIPOD)", "plain", LABEL),
    Ruling(r"^about [\d,]+ (seconds|minutes?|hours?), most of it tuning", "plain",
           "an estimate of the fit's time on these rows"),
    Ruling(r"^(Could bias the score|How well it measures)$", "plain", LABEL),
    Ruling(r"^female=(fe)?male", "plain", "a coding's value as the line records it"),
    Ruling(r"^(clinical prediction and machine learning|prediction from health records|survey and "
           r"health-record analysis)$", "plain", SOURCE),
    Ruling(r".", "plain", LABEL, where=r"\.label$"),
    Ruling(r".", "plain", LABEL, where=r"^artifact:findings findings\[\]\.lever_label$"),
    Ruling(r"^TRIPOD\+AI 23a: performance with intervals in key subgroups", "register",
           EXPLORE_LEVERS),
)

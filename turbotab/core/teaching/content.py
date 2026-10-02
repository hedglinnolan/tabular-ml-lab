"""What each interview question teaches. Sourced from ``docs/turbotab/research/``.

Every drawer section names its pack section and carries the status the pack gives the claim
(SETTLED · CONVENTION · DISPUTED). Where the pack gives a claim no status, the claim is not here.
Budgets: ``turbotab.core.teaching.BUDGETS`` (tested). American spelling.

One precision the packs leave implicit, stated here on purpose (energy adjustment, §04): the
residual and standard models give the same nutrient coefficient when energy is also in the model,
or when there are no other covariates. The residual method as this app fits it takes energy out of
the model, and then the two agree only when no other covariate correlates with energy.
"""
from __future__ import annotations

from typing import Any

_NUT = "research/NUTRITION_PACK.md"
_CLIN = "research/CLINICAL_SURVEY_PACK.md"
_GEN = "research/GENOMICS_PACK.md"
_MET = "research/METABOLOMICS_PACK.md"

NUT01 = f"{_NUT}#01 · Import and structural recognition"
NUT02 = f"{_NUT}#02 · Implausible intake exclusions"
NUT03 = f"{_NUT}#03 · Repeated recalls and measurement error"
NUT04 = f"{_NUT}#04 · Energy adjustment — the methodological signature"
NUT05 = f"{_NUT}#05 · Compositional structure and substitution modeling"
NUT06 = f"{_NUT}#06 · Missing data"
NUT08 = f"{_NUT}#08 · Feature selection and modeling"
CLIN_A12 = f"{_CLIN}#A1.2 · Reference ranges vs physiological plausibility"
CLIN_A13 = f"{_CLIN}#A1.3 · Lab value formats and censored values"
CLIN_A2 = f"{_CLIN}#A2 · Missing data"
CLIN_A41 = f"{_CLIN}#A4.1 · Participant flow diagram"
CLIN_A51 = f"{_CLIN}#A5.1 · Calibration first"
CLIN_A52 = f"{_CLIN}#A5.2 · Class imbalance"
CLIN_A53 = f"{_CLIN}#A5.3 · Discrimination vs calibration"
CLIN_A55 = f"{_CLIN}#A5.5 · Modeling practice"
CLIN_B11 = f"{_CLIN}#B1.1 · Detecting Likert blocks"
CLIN_B4 = f"{_CLIN}#B4 · Ordinal vs interval"
CLIN_B6 = f"{_CLIN}#B6 · Modeling"
GEN01 = f"{_GEN}#01 · Import and structure"
GEN08 = f"{_GEN}#08 · Modeling at p >> n"
MET01 = f"{_MET}#01 · Import and structure"
MET03 = f"{_MET}#03 · Missing data"


def ev(status: str, source: str) -> dict[str, str]:
    return {"status": status, "source": source}


def section(heading: str, body: str, status: str, source: str) -> dict[str, Any]:
    return {"heading": heading, "body": body, "evidence": ev(status, source)}


def option(value: str, label: str, consequence: str) -> dict[str, str]:
    return {"value": value, "label": label, "consequence": consequence}


def term(name: str, definition: str) -> dict[str, str]:
    return {"term": name, "definition": definition}


ESTIMAND = term(
    "estimand",
    "The exact quantity an analysis estimates, stated in words; two methods with different "
    "estimands answer different questions.")

# The energy card's nested-parts note, folded into a term card (M2_CONTRACT §6): the partition
# option's reason says which columns are nested, and this says what follows from it.
NESTED = term(
    "nested",
    "Part of another column's total, such as `fat_sat` within `fat_total`: a partition would "
    "count it twice, and a substitution moves it with its total.")

LENS = {
    "key": "lens",
    "title": "What kind of data this is",
    "question": "What kind of measurements are in this table?",
    "one_liner": "Pick all that apply. It changes what TurboTab checks and suggests, never what you "
                 "can do.",
    "why": "Each field reads a table its own way: an assay has more columns than rows, a recall "
           "file repeats each person, a lab column can hold two units. A lens runs that field's "
           "checks and puts its usual choices first. It never hides a column, a model or an option.",
    "consumer": "The findings, the column roles and the order of every suggestion read it.",
    "options": [
        option("dietary", "Dietary intake",
               "Checks energy units and implausible intakes; offers energy adjustment and "
               "substitution curves."),
        option("clinical", "Clinical measurements and labs",
               "Separates impossible values from abnormal ones; checks units and censored "
               "results."),
        option("metabolomics", "Metabolomics or proteomics",
               "Looks for QC injections, run order, drift and values below detection."),
        option("genomics", "Genomics or transcriptomics",
               "Reads what the values are, counts or normalized, and plans for few samples."),
        option("survey", "Survey or questionnaire instruments",
               "Finds shared response scales and codes outside them, such as 9 for no answer."),
    ],
    "terms": [
        term("lens", "A field's way of reading a table: which checks run and which choices are "
                     "offered first. Several can apply at once."),
    ],
    "drawer": {"sections": [
        section("Dietary intake",
                "Total energy is a strong determinant of every nutrient's intake, so nutrient "
                "associations are confounded by it; energy adjustment is the field's "
                "methodological signature. The dietary pack also checks energy units by the "
                "Atwater reconstruction (4·protein + 4·carbohydrate + 9·fat + 7·alcohol) and "
                "offers the implausible-intake screens.",
                "SETTLED", NUT04),
        section("Clinical measurements and labs",
                "A reference interval holds the central 95% of healthy people by construction and "
                "is for annotation only. Plausibility bounds flag values no living patient could "
                "have. The two are different categories, and no generic outlier rule tells them "
                "apart; the bounds themselves are institution-specific.",
                "CONVENTION", CLIN_A12),
        section("Metabolomics and proteomics",
                "Values below detection are usually filled with half the feature's minimum, the "
                "de-facto default. It is widely used and statistically criticized: it collapses "
                "every missing value to one point, deflating variance.",
                "CONVENTION", MET03),
        section("Genomics and transcriptomics",
                "With more features than samples, an unpenalized model is degenerate rather than "
                "merely overfit: infinitely many coefficient vectors fit the data exactly. "
                "Regularization is mandatory.",
                "SETTLED", GEN08),
        section("Survey instruments",
                "Whether a Likert item is ordinal or interval is genuinely disputed. Run the other "
                "treatment as a sensitivity analysis; if the conclusion holds under both, the "
                "dispute is moot for your paper.",
                "DISPUTED", CLIN_B4),
    ]},
    "evidence": None,
}

ORIENTATION = {
    "key": "orientation",
    "title": "Which way round the table is",
    "question": "Which way round is this table?",
    "one_liner": "Assay exports come both ways; if the columns are samples, every check reads "
                 "across the wrong axis.",
    "why": "In a table of samples, columns are analytes and differ by orders of magnitude while "
           "rows barely differ; here it reads the other way. A transposed table does not fail: "
           "it runs cleanly and means nothing. Turning it changes what a row is, so it is settled "
           "before anything is diagnosed or sealed.",
    "consumer": "The structural diagnosis, the findings, the outcome list and every later "
                "question read the turned table.",
    "options": [
        option("sample_major", "Rows are samples",
               "The table stays as supplied; the record notes it was checked."),
        option("feature_major", "Rows are features",
               "The table is turned: each sample becomes a row and each feature a column."),
    ],
    "terms": [
        term("feature-major", "Features such as metabolites or genes in rows and samples in "
                              "columns: the transpose of what modeling expects."),
        term("transposed", "Turned around, so that the rows become columns and the columns "
                           "become rows."),
    ],
    "drawer": {"sections": [
        section("Exports come both ways",
                "Vendor exports from XCMS, MZmine, MS-DIAL and similar tools are overwhelmingly "
                "features in rows; MetaboAnalyst's own table is samples in rows. A tool must "
                "never guess silently: it presents the reading with its evidence and asks.",
                "SETTLED", MET01),
        section("Genes in rows is the convention",
                "An expression matrix is stored with genes in rows and samples in columns, the "
                "inverse of what the rest of the app assumes. An undetected transpose does not "
                "error; it produces a PCA of genes labeled as samples.",
                "SETTLED", GEN01),
    ]},
    "evidence": ev("SETTLED", MET01),
}

REPAIRS = {
    "key": "repairs",
    "title": "Repairs before the outcome",
    "question": "Should this be repaired before anything is counted?",
    "one_liner": "A repair fixes how a value is written, row by row; it previews on your rows "
                 "first and is recorded.",
    "why": "A sentinel code such as 9 on a 1–5 item is a missing answer, not a strong opinion; a "
           "value impossible in a living patient is an entry error. Repairs that look only at "
           "their own row run on the table now. Anything learned from many rows waits for the "
           "training folds.",
    "consumer": "The working table, the findings, the outcome list and every count downstream "
                "read the repaired values.",
    "options": [
        option("apply", "Apply the repair",
               "The changed cells are written into the working table and listed in the record."),
        option("defer", "Ask me later",
               "The finding waits inside the question it belongs to, checked and attributed."),
        option("dismiss", "Dismiss it",
               "Nothing changes; the record keeps that it was seen and set aside."),
    ],
    "terms": [
        term("sentinel code", "A number standing for an answer that is not a value, such as 9 for "
                              "don't know or 999 for not measured."),
        term("row-local", "Computed from one row's own cells, so it leaks nothing about other "
                          "rows and can run before the seal."),
    ],
    "drawer": {"sections": [
        section("Sentinel codes",
                "In a 1–5 item, a 9 is not extremely agree; it is don't know or refused. Sentinels "
                "must be recoded, never automatically: some legitimate scales do run 0–9.",
                "SETTLED", CLIN_B11),
        section("Impossible is not abnormal",
                "Plausibility bounds flag values no living patient could have; a reference "
                "interval holds the central 95% of healthy people and is for annotation only. "
                "The bounds themselves are institution-specific.",
                "CONVENTION", CLIN_A12),
        section("Failures are not values",
                "TNTC and QNS are measurement failures, not censoring at a detection limit: treat "
                "them as missing, not as extreme values.",
                "SETTLED", CLIN_A13),
    ]},
    "evidence": None,
}

TARGET = {
    "key": "target",
    "title": "The outcome",
    "question": "Which column is the outcome you want to explain or predict?",
    "one_liner": "Everything downstream is built around it: who is counted, how rows are split, and "
                 "what the models learn.",
    "why": "Rows without the outcome leave the analysis first, so the outcome sets the first count "
           "in the participant flow. It also decides the task, the metrics and which findings "
           "apply. When the outcome is itself energy-related, such as weight, BMI or diabetes, "
           "adjusting for reported energy needs extra care.",
    "consumer": "Task detection, the participant flow, the split and every model read it.",
    "options": [],
    "terms": [
        term("outcome", "The column a model learns to predict or explain; also called the target "
                        "or the dependent variable."),
        term("participant flow", "The count of rows at each step, from the file to the analysis, "
                                 "with the reason each step removed rows."),
    ],
    "drawer": {"sections": [
        section("When the outcome is energy-related",
                "Total energy is plausibly on the causal path from diet composition to adiposity, "
                "and adiposity causes under-reporting of energy. Conditioning on reported energy "
                "can then be over-adjustment and collider bias at once. Present adjusted and "
                "unadjusted models and flag it as a limitation; the pack does not pick a side.",
                "DISPUTED", NUT04),
        section("Which N",
                "A participant-flow diagram with every dietary exclusion itemized, each box "
                "carrying an n and a reason, is the single most-checked figure in a nutrition "
                "methods review. The outcome sets its first step.",
                "SETTLED", NUT02),
    ]},
    "evidence": None,
}

EVENT = {
    "key": "event",
    "title": "The event",
    "question": "Which level of the outcome is the event?",
    "one_liner": "Models predict the event's probability; whether that is death or survival is "
                 "your research question.",
    "why": "A two-level outcome needs one level coded 1, and nothing in the file says which: "
           "alive or dead has no correct default. The event fixes the meaning of every predicted "
           "probability, odds ratio and calibration curve, and the methods section names it.",
    "consumer": "The models, the sign of every coefficient, the calibration and each predicted "
                "probability read it.",
    "options": [],
    "terms": [
        term("event", "The outcome level a binary model predicts the probability of, coded 1."),
        term("reference level", "The outcome level coded 0; odds ratios compare the event "
                                "against it."),
    ],
    "drawer": {"sections": [
        section("Probabilities, not just ranks",
                "Rank models on calibration and clinical utility, not on AUC alone: two models "
                "with identical AUC can differ enormously in whether their probabilities of the "
                "event are usable.",
                "SETTLED", CLIN_A51),
        section("A rare event is not a problem to resample away",
                "Undersampling, oversampling and SMOTE overestimate the probability of the "
                "minority class without improving discrimination. A rare event's real problem is "
                "small-sample overfitting, answered by penalization and sample size.",
                "SETTLED", CLIN_A52),
    ]},
    "evidence": None,
}

TASK = {
    "key": "task",
    "title": "The kind of outcome",
    "question": "What kind of outcome is this column?",
    "one_liner": "The task decides which models apply and how they are scored.",
    "why": "A quantity is regression, scored by R², RMSE and MAE. Two classes are binary, scored "
           "by AUC, Brier score and log loss. Several unordered classes are multiclass. An ordered "
           "score, such as a 1–5 rating, fits none of these well: regression assumes equal gaps, "
           "and classification discards the order.",
    "consumer": "The model shelf, the split's stratification and every metric read it.",
    "options": [
        option("regression", "Regression",
               "The outcome is a quantity; models predict its value, scored by R² and RMSE."),
        option("binary", "Binary",
               "Two classes; models predict the probability of one, scored by AUC and Brier "
               "score."),
        option("multiclass", "Multiclass",
               "Several unordered classes; models predict each class's probability, scored by "
               "accuracy."),
    ],
    "terms": [
        term("regression", "A model of a quantity: it predicts a number, and its errors are "
                           "distances from the true value."),
        term("classification", "A model of classes: it predicts the probability that a row "
                               "belongs to each one."),
        term("ordinal outcome", "Classes with an order but no fixed spacing, such as a 1–5 "
                                "rating; best modeled by a cumulative link model."),
    ],
    "drawer": {"sections": [
        section("Ordered outcomes",
                "For an ordinal outcome, a cumulative link (proportional odds) model uses the full "
                "ordering and handles ties; a linear model on the score, or a split into "
                "responders and non-responders, does not. This version fits no ordinal model, so "
                "say which approximation you chose.",
                "SETTLED", CLIN_B6),
        section("Scores built from many items",
                "A multi-item scale score with many categories and no floor or ceiling is often "
                "analyzed as a quantity; whether that is safe is disputed. Run the other treatment "
                "as a sensitivity analysis.",
                "DISPUTED", CLIN_B4),
    ]},
    "evidence": None,
}

PURPOSE = {
    "key": "purpose",
    "title": "Prediction or inference",
    "question": "Is this analysis for prediction or for inference?",
    "one_liner": "Prediction is judged on rows the model never saw; inference on estimates and "
                 "their uncertainty.",
    "why": "The two lead to different models, different checks and a different methods section. "
           "Some advice even flips: a missing-value indicator is legitimate for prediction and "
           "biased for inference. Inference reports coefficients with confidence intervals; "
           "prediction reports performance on held-out rows and leaves coefficients "
           "uninterpreted.",
    "consumer": "The model shelf's order, the coefficient intervals and the Results read it.",
    "options": [
        option("prediction", "Prediction",
               "Models are ranked by performance on rows they never saw; coefficients are not "
               "interpreted."),
        option("inference", "Inference",
               "Linear models report coefficients with confidence intervals; the shelf favors "
               "reportable models."),
    ],
    "terms": [
        term("prediction", "Estimating the outcome for new people as accurately as possible, "
                           "judged on rows the model never saw."),
        term("inference", "Estimating how the outcome relates to an exposure, with the "
                          "uncertainty of that estimate."),
        ESTIMAND,
    ],
    "drawer": {"sections": [
        section("The advice that flips",
                "For prediction, a missing-value indicator is legitimate and often improves "
                "performance, because the same indicator is observable when the model is used. "
                "For inference it gives biased estimates; multiple imputation or a principled "
                "model of the missingness is required.",
                "SETTLED", CLIN_A2),
        section("Hygiene for prediction",
                "Split by participant, not by row, and fit everything learned from data — "
                "energy-adjustment residuals, scaling, imputation — inside the training fold "
                "only.",
                "SETTLED", NUT08),
    ]},
    "evidence": None,
}

GROUPED_SPLIT = term("grouped split", "A split that keeps all of one participant's rows on the "
                                      "same side, so repeat measurements cannot leak across.")

GRAIN = {
    "key": "grain",
    "title": "Whether people repeat",
    "question": "Can one person appear in more than one row?",
    "one_liner": "You know this and the file only hints at it; the held-out rows cannot be drawn "
                 "correctly without it.",
    "why": "If one person's rows land on both sides of the split, the model is scored on people it "
           "has already seen, and every score is inflated. A column that repeats is a suggestion, "
           "never the answer: the app checks your answer against the data and says so when they "
           "disagree.",
    "consumer": "The seal's grouping, the repeats and unit questions, and every cross-validation "
                "fold read it.",
    "options": [
        option("one_row_per_unit", "One row each",
               "Each row is a different person; the split may draw rows freely."),
        option("repeated", "People repeat",
               "Rows sharing an identifier stay together; the next questions ask what repeats."),
        option("unknown", "I don't know",
               "Rows are held out one by one, and every held-out score is labeled exploratory."),
    ],
    "terms": [
        GROUPED_SPLIT,
        term("grain", "What one row of the table is: one person, one visit, one recall or one "
                      "sample."),
    ],
    "drawer": {"sections": [
        section("Split by participant",
                "With repeated recalls, a row-level split puts one person's days on both sides and "
                "inflates every score. Split by participant, not by row.",
                "SETTLED", NUT08),
        section("What repeats can look like",
                "Duplicate participant identifiers mean repeated measures; look for the occasion "
                "too, such as a day, visit or recall number, or a date. Twin columns such as "
                "`DR1TKCAL` and `DR2TKCAL` are the same structure in wide form.",
                "SETTLED", NUT01),
    ]},
    "evidence": ev("SETTLED", NUT08),
}

REPEAT_KIND = {
    "key": "repeat_kind",
    "title": "What repeats",
    "question": "Are these repeats or different time points?",
    "one_liner": "Replicates of one quantity can be averaged; time points carry change that "
                 "averaging would erase.",
    "why": "Two recalls days apart measure one usual diet twice, so their mean reduces "
           "day-to-day error. Clinic visits months apart measure a person changing, so the order "
           "is the signal. Dates with no visit structure suggest repeats; a visit label or "
           "spaced dates suggest time points. Where the evidence is thin, this is asked.",
    "consumer": "The aggregation menu, its recommended default and the temporal question read "
                "it.",
    "options": [
        option("repeats", "Repeats",
               "Repeated measurements of one quantity, such as two recalls; averaging is offered "
               "first."),
        option("time_points", "Time points",
               "Different moments in time; averaging is not offered first, and order matters."),
    ],
    "terms": [
        term("replicate", "A repeated measurement of the same quantity under the same "
                          "conditions, such as a second recall of usual diet."),
        term("time point", "A measurement at a distinct moment, such as a visit, where change "
                           "between moments is part of the data."),
    ],
    "drawer": {"sections": [
        section("Why two recalls exist",
                "A single 24-hour recall measures one day, not usual diet. Two or more "
                "non-consecutive recalls separate day-to-day variation from real differences "
                "between people; with one recall that separation is impossible.",
                "SETTLED", NUT03),
        section("Consecutive days are not independent",
                "Recalls on consecutive days have correlated errors, so within-person variation "
                "is underestimated and the data look more reliable than they are. NHANES uses "
                "non-consecutive days by design.",
                "SETTLED", NUT03),
    ]},
    "evidence": ev("SETTLED", NUT03),
}

UNIT = {
    "key": "unit",
    "title": "One row in the analysis",
    "question": "When you analyze this, what is one row?",
    "one_liner": "People appear more than once; that leaves two honest options, and they lead to "
                 "different analyses.",
    "why": "One row per person combines each person's records, and the next question asks how. "
           "One row per record keeps them as they are, and the split keeps each person's records "
           "on one side. There is no default: guessing here is how the leak this check exists "
           "to prevent begins.",
    "consumer": "The aggregation question, the working table, the participant flow and the seal "
                "read it.",
    "options": [
        option("unit", "One row per person",
               "Each person's records are combined into one row; the next question asks how."),
        option("row", "One row per record",
               "Records stay as they are; held-out people never appear in training."),
    ],
    "terms": [
        GROUPED_SPLIT,
        term("unit of analysis", "What one row of the modeled table stands for: a person, or one "
                                 "of their records."),
    ],
    "drawer": {"sections": [
        section("Leakage across a person's rows",
                "If a person contributes several recalls, rows from the same person must never "
                "be split across training and test folds; anything estimated from the data, such "
                "as a residual energy adjustment, is fit inside the training fold.",
                "SETTLED", NUT03),
    ]},
    "evidence": None,
}

AGGREGATION = {
    "key": "aggregation",
    "title": "Combining each person's rows",
    "question": "How should each person's rows be combined?",
    "one_liner": "The right summary depends on what repeats: averaging replicates reduces error, "
                 "averaging time points destroys the signal.",
    "why": "For repeated recalls, the mean is an acceptable exposure for ranking people: still "
           "attenuated, but unbiased in direction. For visits, choose the baseline, the last "
           "visit or the change by what the study asks. When the outcome itself varies within a "
           "person, say which value is the outcome.",
    "consumer": "The working table, the participant flow, the seal and every model read the "
                "combined rows.",
    "options": [
        option("mean", "Mean", "Each person's rows are averaged; replicates' day-to-day error "
                               "shrinks."),
        option("first", "First (baseline)", "Each person keeps their first row, such as the "
                                             "baseline visit."),
        option("last", "Last", "Each person keeps their most recent row."),
        option("change", "Change from baseline",
               "Each measurement becomes the last value minus the first: change over time."),
    ],
    "terms": [
        term("attenuation", "The shrinking of an association toward zero because the exposure "
                            "is measured with error; averaging more days reduces it."),
        term("usual intake", "A person's long-run average intake, which single days only "
                             "estimate; modeling it properly is more than a mean."),
    ],
    "drawer": {"sections": [
        section("When the mean is adequate",
                "To rank people for regression, classification or a predictive model, the mean of "
                "the available recalls is an acceptable exposure: attenuated, but unbiased in "
                "direction under classical error, and what most cohort analyses use.",
                "CONVENTION", NUT03),
        section("When the mean is not adequate",
                "Prevalence or percentile claims about usual intake, episodically consumed foods "
                "with many zero days, and exposure coefficients that must be unbiased in "
                "magnitude all need usual-intake modeling, which this version does not fit.",
                "SETTLED", NUT03),
        section("Cumulative averages over follow-up",
                "With repeated questionnaires, the cohort standard is the cumulative average up "
                "to each event; if diet changes because of preclinical disease, a lag is "
                "conventional.",
                "CONVENTION", NUT03),
    ]},
    "evidence": ev("CONVENTION", NUT03),
}

TEMPORAL = {
    "key": "temporal",
    "title": "Predicting forward in time",
    "question": "Are you predicting something later from measurements taken earlier?",
    "one_liner": "A random split is optimistic when the task looks forward; the held-out rows "
                 "should then be the latest.",
    "why": "With visits kept as rows, a random split lets the model learn from a person's later "
           "visits and be scored on their earlier ones. Validation in time, holding out the "
           "latest rows, is a distinct check from validation on random rows, and reporting "
           "guidelines treat it so.",
    "consumer": "The seal: chronological when yes, grouped by person either way.",
    "options": [
        option("true", "Yes, later from earlier",
               "The held-out rows are the latest, and each person's rows stay together."),
        option("false", "No",
               "The held-out rows are drawn at random, each person's rows kept together."),
    ],
    "terms": [
        GROUPED_SPLIT,
        term("chronological split", "A split that holds out the latest rows by date, so models "
                                    "are scored only on what came after their training data."),
    ],
    "drawer": {"sections": [
        section("Split by participant",
                "Split by participant, not by row, and fit everything learned from data inside "
                "the training fold only.",
                "SETTLED", NUT08),
        section("Resample the whole pipeline",
                "Internal validation must resample the entire modeling pipeline: imputation, "
                "transformation, selection and tuning.",
                "SETTLED", CLIN_A55),
    ]},
    "evidence": None,
}

ROLES = {
    "key": "roles",
    "title": "What each column is",
    "question": "What role does each column play in the analysis?",
    "one_liner": "Only exposures, covariates and the energy column can become predictors; every "
                 "other role stays out of the models.",
    "why": "An identifier names a person and cannot generalize. Survey weights and design columns "
           "describe how people were sampled, and a flag such as `imputed_weight` describes how a "
           "value was filled. None of them measures the participant, so each stays out of the "
           "predictors while the record keeps it.",
    "consumer": "The participant flow, the split's grouping, energy adjustment and the model "
                "matrix read it.",
    "options": [
        option("identifier", "Identifier",
               "Names a participant; never a predictor, and keeps a person's rows together in "
               "the split."),
        option("exposure", "Exposure",
               "What you are studying; a predictor, and energy adjustment applies to its "
               "nutrients."),
        option("energy", "Energy",
               "Total energy intake; energy adjustment is computed against it and decides whether "
               "it stays."),
        option("covariate", "Covariate",
               "Adjusted for, such as age or sex; enters the models beside the exposures."),
        option("design", "Survey design",
               "A survey weight, stratum or cluster; kept out of the predictors."),
        option("flag", "Flag",
               "Marks another column's values, such as imputed ones; kept out of the predictors."),
        option("time", "Time",
               "When a row was measured, such as a cycle or visit; kept out of the predictors."),
        option("excluded", "Excluded",
               "Left out of the analysis; the record keeps the column and the reason."),
    ],
    "terms": [
        term("exposure", "The variable whose relationship with the outcome the study is about, "
                         "such as a nutrient intake."),
        term("covariate", "A variable adjusted for so that the exposure's estimate is not "
                          "confounded by it, such as age or sex."),
        term("identifier", "A column that names a row or a person rather than measuring anything; "
                           "a value that appears once cannot generalize."),
    ],
    "drawer": {"sections": [
        section("Survey design columns",
                "NHANES oversamples some age, race and income groups on purpose. Without its "
                "weights, means are biased toward the oversampled groups; without strata and PSUs, "
                "standard errors are too small. Dietary analyses use `WTDRD1` for day 1 or "
                "`WTDR2D` for both days, not the examination weight `WTMEC2YR`. This version "
                "records design columns but does not yet weight its estimates.",
                "SETTLED", NUT01),
        section("Pooled cycles",
                "Combining NHANES cycles means dividing the two-year weights by the number of "
                "cycles combined and confirming the same dietary method applies across them.",
                "SETTLED", NUT01),
        section("Units decide everything downstream",
                "1 kcal = 4.184 kJ, and a column ending `_pct_kcal` is a share of energy, not an "
                "amount. Every implausibility screen, energy adjustment and substitution is a "
                "function of total energy, so a unit error there reaches every result.",
                "SETTLED", NUT01),
    ]},
    "evidence": None,
}

EXCLUSIONS = {
    "key": "exclusions",
    "title": "Who the study is about",
    "question": "Is your study restricted to part of this data?",
    "one_liner": "A restriction comes from your research question, so the outcome's distribution "
                 "is not shown here.",
    "why": "A criterion such as an age range changes N and is reported in the participant flow "
           "before any rows are sealed. A cut chosen by looking at the outcome is data-driven "
           "selection, its own bias. Implausible-intake screens are offered with the rows each "
           "removes; under-reporting concentrates in higher BMI, so many analyses keep everyone.",
    "consumer": "The participant flow, the seal and every model's rows read it.",
    "options": [
        option("none", "Keep every row",
               "No row is excluded; the record states that no exclusion was applied."),
        option("willett_by_sex", "Willett, by sex",
               "Women outside 500–3,500 and men outside 800–4,200 kcal a day are excluded."),
        option("sex_neutral_500_5000", "500–5,000 kcal a day",
               "Anyone outside 500–5,000 kcal a day is excluded, whatever their sex."),
        option("sex_neutral_500_3500", "500–3,500 kcal a day",
               "Anyone outside 500–3,500 kcal a day is excluded; stricter, so it removes more."),
        option("custom", "Your own range",
               "Rows outside a range you set on a numeric column, never the outcome, are excluded."),
    ],
    "terms": [
        term("eligibility criterion", "Who the study is about, such as an age range: applied to "
                                      "every row before the seal and reported with its count."),
        term("implausible intake", "A reported day's energy intake too low or too high to be a "
                                   "real diet, usually a reporting error."),
        term("under-reporting", "Reporting less than was eaten. It is systematic, concentrated in "
                                "people with higher BMI and in weight-conscious participants."),
        term("Goldberg cut-off", "A screen comparing reported energy with estimated basal "
                                 "metabolic rate; the field's standard for misreporting, not "
                                 "offered in this version."),
    ],
    "drawer": {"sections": [
        section("Each criterion its own box",
                "Track the rows through every filter, each eligibility criterion separately, with "
                "its reason and its count. A single excluded box tells a reader nothing about who "
                "is missing from the model's population.",
                "CONVENTION", CLIN_A41),
        section("The screens in circulation",
                "Willett and the Nurses' Health Study use 500–3,500 kcal a day for women and "
                "800–4,200 for men. Variants use 4,000 or 5,000 as men's upper bound, or a "
                "sex-neutral 500–5,000 or 500–3,500. The conventions genuinely differ across "
                "literatures, so show how N moves with the choice.",
                "CONVENTION", NUT02),
        section("What exclusion does not fix",
                "Excluding misreporters reduces bias in diet–outcome associations but does not "
                "remove it; in one evaluation only 14 of 24 nutrition–outcome pairs improved. "
                "That exclusion is insufficient is settled; whether to exclude at all is disputed.",
                "DISPUTED", NUT02),
        section("A common default",
                "Drop only unreliable recalls and impossible values, run the primary analysis on "
                "the full sample with energy adjustment, and present the misreporter-excluded "
                "analysis as a prespecified sensitivity analysis. A reviewer in an obesity journal "
                "may expect exclusion.",
                "CONVENTION", NUT02),
        section("Who is excluded",
                "Under-reporting is concentrated in people with higher BMI, so excluding "
                "under-reporters when the outcome is adiposity removes a non-random slice of "
                "exactly the population studied. Report the excluded group's characteristics.",
                "SETTLED", NUT02),
    ]},
    "evidence": ev("CONVENTION", NUT02),
}

MISSING = {
    "key": "missing",
    "title": "Missing predictor values",
    "question": "How should rows with missing predictor values be handled?",
    "one_liner": "Dropping incomplete rows can shrink and skew the sample; filling values keeps "
                 "rows but adds an assumption.",
    "why": "Complete cases keep only rows with every predictor measured, which can remove a large, "
           "non-random share: a predictor blank for 70% of people drops 70% of rows. Imputation keeps "
           "every row by filling values learned from training rows, assuming the blanks resemble "
           "what was observed. A blank can also mean the question was not asked.",
    "consumer": "The participant flow, each model's pipeline and the banner's row counts read it.",
    "options": [
        option("complete_case", "Complete cases",
               "Rows missing any predictor are dropped; the participant flow shows how many."),
        option("impute", "Impute",
               "Missing predictor values are filled in, learned from training rows only; every "
               "row stays."),
    ],
    "terms": [
        term("complete cases", "Analyzing only rows with every variable measured; it can bias "
                               "estimates when who is missing depends on the outcome."),
        term("imputation", "Filling a missing value with an estimate learned from observed rows; "
                           "a single fill understates the uncertainty."),
        term("missing at random", "Missingness that the observed columns explain; the assumption "
                                  "standard multiple imputation relies on."),
    ],
    "drawer": {"sections": [
        section("Blanks often mean something",
                "On a food-frequency questionnaire a blank frequently means never, especially "
                "among older participants. That missingness is not at random, and standard "
                "multiple imputation assumes it is: an improvement, not a solution.",
                "SETTLED", NUT06),
        section("A zero is not a blank",
                "A zero on a 24-hour recall is a zero for that day, not a person who never eats "
                "the food.",
                "SETTLED", NUT06),
        section("Filling with the mean or median",
                "Mean or median filling understates variance and distorts the distribution; the "
                "clinical pack calls it indefensible in a manuscript.",
                "SETTLED", CLIN_A2),
        section("The outcome belongs in the imputation model",
                "Imputing with the outcome left out of the imputation model biases associations "
                "toward the null.",
                "SETTLED", CLIN_A2),
    ]},
    "evidence": None,
}

SPLIT = {
    "key": "split",
    "title": "The seal: held-out rows",
    "question": "How many rows should be sealed for one final, untouched score?",
    "one_liner": "Sealed rows are scored once, when you open the seal at the end; every choice is "
                 "tuned by cross-validation on the rest.",
    "why": "A model scored on the rows it was tuned on grades its own homework. The seal states "
           "its basis: grouped by an identifier, repeats found but not grouped, or undetermined, "
           "which is labeled exploratory. A small holdout measures little, so with few rows "
           "cross-validation alone comes first.",
    "consumer": "The participant flow, every model's cross-validation and the held-out score read "
                "it.",
    "options": [
        option("0", "Cross-validation only",
               "No rows are sealed; every score comes from cross-validation on all rows."),
        option("0.1", "Hold out 10%",
               "One row in ten is sealed for the final score; more rows train."),
        option("0.2", "Hold out 20%",
               "One row in five is sealed for the final score."),
        option("0.3", "Hold out 30%",
               "Nearly a third is sealed: a steadier final score, fewer rows to train."),
    ],
    "terms": [
        term("holdout", "Rows set aside before any modeling and scored once, at the end; the "
                        "closest thing to new data you have."),
        term("seal", "The line between the held-out rows and the rest: drawn before modeling, "
                     "opened once, its basis stated."),
        term("cross-validation", "Splitting the training rows into folds, fitting on all but one "
                                 "and scoring on the one left out, in turn."),
        term("fold", "One of the equal parts cross-validation splits the training rows into; each "
                     "is scored by a model that never saw it."),
        term("grouped split", "A split that keeps all of one participant's rows on the same side, "
                              "so repeat measurements cannot leak across."),
        term("seed", "The number that fixes the random draw, so the same split can be drawn "
                     "again."),
    ],
    "drawer": {"sections": [
        section("Split by participant",
                "With repeated recalls, a row-level split puts one person's days on both sides and "
                "inflates every score. Split by participant, not by row.",
                "SETTLED", NUT08),
        section("Everything learned stays inside the fold",
                "Everything supervised — feature ranking, selection, thresholds, tuning — happens "
                "inside the training fold. Selecting features on all samples can report near-zero "
                "error when no signal exists at all.",
                "SETTLED", GEN08),
        section("A single split is weak",
                "Internal validation should resample the whole pipeline. Bootstrap optimism "
                "correction or repeated cross-validation is preferred; a single train/test split "
                "is the weakest option at typical clinical sample sizes.",
                "CONVENTION", CLIN_A55),
        section("Small samples",
                "Below about 50 rows, a single 5-fold estimate has a standard error large enough "
                "that a 0.05 AUC difference is noise. Repeat the cross-validation and report the "
                "spread.",
                "SETTLED", GEN08),
    ]},
    "evidence": ev("CONVENTION", CLIN_A55),
}

ENERGY_ADJUSTMENT = {
    "key": "energy_adjustment",
    "title": "Energy adjustment",
    "question": "How should nutrient intakes be adjusted for total energy?",
    "one_liner": "People who eat more eat more of everything; each method answers a different "
                 "question about a nutrient.",
    "why": "Total energy confounds every nutrient association, and the errors in reported "
           "nutrients and energy move together. Each method has its own estimand: the standard "
           "and residual methods ask about swapping calories between sources at fixed total "
           "energy, and partition asks about adding calories. Choose by the question; compare "
           "methods as a sensitivity analysis.",
    "consumer": "Each model's pipeline, the column lineage, the coefficients and the substitution "
                "curves read it.",
    "options": [
        option("none", "No adjustment",
               "Absolute intake: a nutrient's effect stays mixed with how much people eat "
               "overall."),
        option("standard", "Standard model",
               "Energy stays in the model: more of the nutrient in place of other calories."),
        option("residual", "Residual method",
               "Each nutrient's residual on energy: the same substitution, in the nutrient's own "
               "units."),
        option("density_multivariate", "Density plus energy",
               "Nutrient per calorie, with total energy as its own term: diet composition."),
        option("density", "Density alone",
               "Nutrient per calorie, energy dropped: a rescaled effect whose meaning is "
               "obscure."),
        option("partition", "Energy partition",
               "Calories from the nutrient and from everything else: adding calories, not "
               "substituting."),
    ],
    "terms": [
        ESTIMAND,
        term("residual method", "Replacing a nutrient by its residual from a regression on total "
                                "energy, re-centered at its mean, so it no longer tracks energy."),
        term("substitution", "More of one energy source in place of others at the same total "
                             "energy; what an energy-adjusted coefficient describes."),
        term("nutrient density", "A nutrient's amount per unit of energy, such as grams per "
                                 "1,000 kcal."),
        term("energy partition", "Splitting total energy into calories from the nutrient and "
                                 "calories from everything else, each its own term."),
        NESTED,
    ],
    "drawer": {"sections": [
        section("Why adjust at all",
                "Total energy drives nutrient intake through body size, activity and reporting "
                "scale; the hypothesis is usually about composition, not quantity; and correlated "
                "reporting errors in nutrients and energy partly cancel once energy is accounted "
                "for.",
                "SETTLED", NUT04),
        section("Residual and standard: when they agree",
                "The pack calls the two mathematically equivalent, and it holds in two cases. With "
                "total energy also in the model, swapping a nutrient for its residual only "
                "reparametrizes it, so the coefficient is identical, covariates or not. With no "
                "other covariates, the residual alone gives the standard model's coefficient too. "
                "The residual method here takes energy out of the model, so with covariates the "
                "two agree only when no covariate correlates with energy.",
                "SETTLED", NUT04),
        section("What the coefficient means",
                "An energy-adjusted coefficient is a substitution estimate: more of this nutrient "
                "and correspondingly less of everything else, at the same total intake — not the "
                "effect of simply eating more of it.",
                "SETTLED", NUT04),
        section("The biases they share",
                "Standard and residual models are biased even without confounding (composite "
                "variable bias), and all four models only partly account for confounding by "
                "common dietary causes; each evaluates a different estimand (Tomova et al. 2022, "
                "AJCN).",
                "SETTLED", NUT04),
        section("The field's default",
                "The Willett residual method, computed within the final analytic sample and within "
                "sex, on log-transformed intakes, with the predicted nutrient at the mean energy "
                "added back. It is the field default, but not uncontested.",
                "CONVENTION", NUT04),
        section("When the outcome is BMI or adiposity",
                "Energy may be on the causal path and a collider at once. Present adjusted and "
                "unadjusted models and flag it in the limitations; the pack does not pick a side.",
                "DISPUTED", NUT04),
        section("Fit inside the folds",
                "The nutrient-on-energy regression is learned from data, so it is fit on training "
                "rows only; fitting it on all rows before cross-validation leaks the held-out "
                "rows into every fold.",
                "SETTLED", NUT08),
    ]},
    "evidence": ev("CONVENTION", NUT04),
}

MODELS = {
    "key": "models",
    "title": "Model families",
    "question": "Which model families should be fit and compared?",
    "one_liner": "Each family brings its own assumptions about how nutrients relate to the "
                 "outcome; fitting several shows what depends on them.",
    "why": "A linear model assumes straight-line, additive effects and gives coefficients you can "
           "report. An elastic net shrinks correlated nutrients together, which suits diets where "
           "nutrients travel in foods. Boosted trees find curves and interactions but give no "
           "coefficients. In one comparison of 30,000 models, the algorithm mattered less than the "
           "endpoint and the analyst.",
    "consumer": "The fit, the Results and the substitution curves read it.",
    "options": [
        option("linear", "Linear model",
               "OLS or logistic regression: one reportable coefficient per predictor, with "
               "intervals for inference."),
        option("elastic_net", "Elastic net",
               "A penalized linear model: shrinks correlated nutrients together, tuned inside "
               "training folds."),
        option("boosted_trees", "Boosted trees",
               "Many shallow trees: finds curves and interactions and handles missing values; no "
               "coefficients."),
    ],
    "terms": [
        term("inductive bias", "What a model assumes before it sees data, such as straight lines "
                               "or interactions; it decides what the model can find."),
        term("penalization", "Shrinking coefficients toward zero to trade a little bias for less "
                             "variance; essential when predictors are many or correlated."),
        term("calibration", "Whether predicted probabilities match observed frequencies; two "
                            "models with the same AUC can differ greatly in it."),
    ],
    "drawer": {"sections": [
        section("Collinearity is the biology",
                "Nutrients travel together in foods. Automatic selection among correlated "
                "nutrients picks one marker of a shared source, not a cause; report how often "
                "each is selected across resamples rather than a single set.",
                "SETTLED", NUT08),
        section("The algorithm is rarely the lever",
                "In MAQC-II, across more than 30,000 models from 36 teams, performance depended "
                "mainly on the endpoint and the team's proficiency, and different algorithms "
                "performed similarly. Whether tree ensembles beat penalized linear models is "
                "disputed.",
                "DISPUTED", GEN08),
        section("More features than samples",
                "When predictors outnumber rows, an unpenalized model is degenerate and "
                "regularization is mandatory; the elastic net is the standard choice if one must "
                "be named.",
                "SETTLED", GEN08),
        section("Calibrate the trees",
                "Rank models on calibration as well as discrimination. Boosted trees often produce "
                "miscalibrated probabilities: report the calibration curve, or recalibrate on "
                "held-out data, never on training rows.",
                "SETTLED", CLIN_A51),
        section("Avoid stepwise selection",
                "Stepwise selection produces unstable variable sets, biased coefficients and "
                "intervals with the wrong coverage; prefer prespecification or penalization.",
                "SETTLED", CLIN_A55),
    ]},
    "evidence": None,
}

SUBSTITUTION = {
    "key": "substitution",
    "title": "Substitution curves",
    "question": "Which nutrient's calories should replace which, and in what steps?",
    "one_liner": "A substitution curve shows the predicted outcome as calories move from one "
                 "nutrient to another at fixed total energy.",
    "why": "Energy-adjusted effects are substitutions, so name what replaces what. The curve moves "
           "calories from the donor to the recipient through each fitted model, only as far as "
           "your data hold people with such diets. Both nutrients must carry energy; grams and "
           "servings do not make an isocaloric swap.",
    "consumer": "The substitution stage draws one curve per fitted model family.",
    "options": [
        option("50", "50 kcal steps",
               "Finer steps; the curve's labeled effect is a 50 kcal swap."),
        option("100", "100 kcal steps",
               "The curve's labeled effect is swapping 100 kcal from donor to recipient."),
        option("200", "200 kcal steps",
               "Coarser steps; the curve's labeled effect is a 200 kcal swap."),
    ],
    "terms": [
        term("donor", "The nutrient whose calories are taken away in a substitution."),
        term("recipient", "The nutrient that receives those calories, so total energy stays the "
                          "same."),
        term("isocaloric", "At the same total energy. A substitution is isocaloric only when both "
                           "parts are measured in calories."),
        term("support", "The range of diets actually present in your data; a curve beyond it is "
                        "extrapolation, so the curve stops there."),
        NESTED,
    ],
    "drawer": {"sections": [
        section("Adjusting for energy is not enough",
                "Adjusting for total energy does not by itself make a substitution isocaloric. The "
                "components must be measured and analyzed in calories; in grams or servings, the "
                "substitution reported is not the one intended.",
                "SETTLED", NUT05),
        section("Parts of a whole",
                "When shares of energy sum to 100%, one part is determined by the others, and a "
                "coefficient on percent energy from fat has no meaning until you say what the fat "
                "replaced.",
                "SETTLED", NUT05),
        section("Three legitimate designs",
                "A leave-one-out model (every component but one, plus total energy, in kcal), "
                "compositional log-ratios, or an explicit difference of coefficients. The first and "
                "last are conventions; log-ratios are emerging and less familiar to reviewers.",
                "CONVENTION", NUT05),
        section("Validated inputs",
                "A review of 100 substitution studies found 53% used unvalidated food-frequency "
                "variables; where validation was reported, correlations with reference methods "
                "ranged from 0.12 to 0.77.",
                "SETTLED", NUT05),
    ]},
    "evidence": ev("SETTLED", NUT05),
}

OPEN_SEAL = {
    "key": "open_seal",
    "title": "Opening the seal",
    "question": "Open the held-out rows and score the models once?",
    "one_liner": "The held-out scores exist but are withheld until now; opening the seal fixes them "
                 "in the record.",
    "why": "Held-out rows give one honest number only if nothing was tuned against them. Once "
           "opened, the scores stay in the record. A later change still refits every model, but "
           "its sentence and the manuscript mark it as made after the seal was opened.",
    "consumer": "The Results, the manuscript's performance table and the mark on any later "
                "change read it.",
    "options": [
        option("open", "Open the seal",
               "Each model is scored once on the held-out rows; those scores are fixed."),
        option("keep", "Keep it sealed",
               "The Results stay on cross-validation; the held-out rows stay unread."),
    ],
    "terms": [
        term("post-seal", "A change made after the held-out scores were seen; the record and the "
                          "manuscript mark it so."),
    ],
    "drawer": {"sections": [
        section("What to report from the held-out rows",
                "Discrimination ranks and calibration measures magnitude; you need both. Report "
                "the C-statistic with its interval, the calibration intercept and slope, the "
                "calibration curve and the Brier score.",
                "SETTLED", CLIN_A53),
        section("The whole pipeline, validated",
                "Internal validation must resample the entire modeling pipeline, imputation and "
                "tuning included. A single train and test split is the weakest option at typical "
                "clinical sample sizes.",
                "CONVENTION", CLIN_A55),
    ]},
    "evidence": None,
}

ENTRIES = [LENS, ORIENTATION, REPAIRS, TARGET, EVENT, TASK, PURPOSE, GRAIN, REPEAT_KIND, UNIT,
           AGGREGATION, TEMPORAL, ROLES, EXCLUSIONS, MISSING, SPLIT, ENERGY_ADJUSTMENT, MODELS,
           SUBSTITUTION, OPEN_SEAL]

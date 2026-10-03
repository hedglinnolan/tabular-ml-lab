"""What each interview question teaches. Sourced from ``docs/turbotab/research/``.

Every drawer section names its pack section and carries the status the pack gives the claim
(SETTLED · CONVENTION · DISPUTED). Where the pack gives a claim no status, the claim is not here.
Budgets: ``turbotab.core.teaching.BUDGETS`` (tested). American spelling.

One precision the packs leave implicit, stated here on purpose (energy adjustment, §04): the
residual and standard models give the same nutrient coefficient when energy is also in the model,
or when there are no other covariates. The residual method keeps total energy in the outcome model
by default (BLUEPRINT §12 ruling 1); its energy-dropped form agrees with the standard model only
when no other covariate correlates with energy.

Audit WP15 (claims): every sentence here matches its primary source and the computation. Where a
claims-ledger row was WRONG, OVERSTATED or SELF-CONTRADICTED, the corrected sentence sits beside a
comment quoting its source, and ``tests/acceptance/test_wp15_claims.py`` replays each one. A claim
two cards share is one section constant, so it carries one badge wherever it appears.
"""
from __future__ import annotations

from typing import Any

from turbotab.core.methods.missing import RULE

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

# Claims that several cards make are one section each, so a claim carries one badge wherever it
# appears (audit WP15 acceptance 2, G18: "resample the whole pipeline" read SETTLED in one card and
# CONVENTION in two others; the clinical pack, §A5.5: "SETTLED that the full pipeline must be
# inside the loop; the bootstrap-vs-CV preference is CONVENTION").
WHOLE_PIPELINE = section(
    "Resample the whole pipeline",
    "Internal validation must resample the entire modeling pipeline: imputation, transformation, "
    "selection and tuning.",
    "SETTLED", CLIN_A55)
SINGLE_SPLIT = section(
    "A single split is the weakest",
    "Bootstrap optimism correction is the recommended default and repeated cross-validation is "
    "acceptable; a single train and test split is the weakest option at typical clinical sample "
    "sizes.",
    "CONVENTION", CLIN_A55)
# Audit ledger #4 (OVERSTATED): "Regularization is mandatory" is true of least squares and logistic
# regression, whose fit at p ≥ n is not unique (rank(X) ≤ n < p, so XᵀX is singular); dimension
# reduction and one test per feature (this app's feature-wise family) are other ways through.
P_OVER_N = section(
    "More features than samples",
    "With more features than samples, an unpenalized least-squares or logistic model is "
    "degenerate rather than merely overfit: infinitely many coefficient vectors fit the data "
    "exactly. Penalization, dimension reduction or one test per feature are the ways through; the "
    "elastic net is the standard choice if one model must be named.",
    "SETTLED", GEN08)
# Audit ledger #12 (SELF-CONTRADICTED): the drawer said "rank models on calibration", while the
# families are ranked by AUC; each fit now reports calibration beside it (models/metrics.py).
# Van Calster et al. 2019 (BMC Med 17:230): "poor calibration may make an algorithm less clinically
# useful than a competitor algorithm that has a lower AUC but is well calibrated".
CALIBRATION_TOO = section(
    "Probabilities, not just ranks",
    "Judge models on calibration as well as discrimination: a well-calibrated model with a lower "
    "AUC can be more useful than a miscalibrated one with a higher AUC (Van Calster et al. 2019). "
    "Each fit here reports its calibration intercept and slope beside the AUC that ranks the "
    "families.",
    "SETTLED", CLIN_A51)

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
        # Audit IN-20 (ledger #43): adjusting changes the estimand (Tomova et al. 2022); "every
        # nutrient association is confounded by it" overstated what adjustment does.
        section("Dietary intake",
                "Total energy is a strong determinant of every nutrient's intake, so adjusting for "
                "it changes what a nutrient's coefficient means: with energy fixed, more of one "
                "source is less of another (Tomova et al. 2022). Energy adjustment is the field's "
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
        {**P_OVER_N, "heading": "Genomics and transcriptomics"},
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
        # Audit IN-23 (ledger #9, WRONG for TNTC). FDA Bacteriological Analytical Manual, ch. 3:
        # "When number of CFU per plate exceeds 250, for all dilutions, record the counts as too
        # numerous to count (TNTC)"; crowded plates are estimated "as greater than 100 times the
        # highest dilution plated". A count above the range is right-censored, not missing.
        section("Too many to count is a value",
                "TNTC, too numerous to count, means a plate held more colonies than its countable "
                "range, over 250 per plate in FDA's Bacteriological Analytical Manual: the count "
                "is above that limit, right-censored like a result above the upper limit of "
                "quantitation, not missing. QNS, quantity not sufficient, and a hemolyzed specimen "
                "are measurement failures with no value: treat those as missing.",
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
        # Audit G14 (ledger #11, OVERSTATED): the superlative had no source. STROBE-nut (Lachat et
        # al. 2016), nut-13: "Report the number of individuals excluded based on missing,
        # incomplete, or implausible dietary/nutritional data"; STROBE item 13(c): "Consider use of
        # a flow diagram."
        section("Which N",
                "Report how many people each dietary exclusion removed, and why: STROBE-nut asks "
                "for the number excluded for missing, incomplete or implausible dietary data, and "
                "STROBE suggests a flow diagram, each box carrying an n and a reason. The outcome "
                "sets its first step.",
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
        CALIBRATION_TOO,
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
           "score, such as a 1–5 rating, is ordinal: regression would assume equal gaps, and "
           "classification would discard the order. An event whose follow-up varies is time to "
           "event.",
    "consumer": "The model shelf, the split's stratification and every metric read it.",
    "options": [
        option("regression", "Regression",
               "The outcome is a quantity; models predict its value, scored by R² and RMSE."),
        option("binary", "Binary",
               "Two classes; models predict the probability of one, scored by AUC and Brier "
               "score."),
        # Audit G16 (ledger #14): the primary metric is log loss (models/metrics.py PRIMARY).
        option("multiclass", "Multiclass",
               "Several unordered classes; models predict each class's probability, scored by log "
               "loss."),
        option("ordinal", "Ordinal",
               "Ordered levels, such as a 1–5 rating; a proportional-odds model keeps their order."),
        option("time_to_event", "Time to event",
               "An event with each row's follow-up; a Cox model gives hazard ratios, scored by "
               "the C-index."),
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
                "responders and non-responders, does not.",
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
        # Audit ledger #16: Groenwold et al. 2012 (CMAJ 184:1265): the method "typically results in
        # biased estimates in nonrandomized studies", while "In randomized trials, the missing-
        # indicator method is a valid method to handle missing baseline covariate data".
        section("The advice that flips",
                "For prediction, a missing-value indicator is legitimate and often improves "
                "performance, because the same indicator is observable when the model is used. "
                "For inference in observational data it gives biased estimates (in a randomized "
                "trial it is valid for baseline covariates); multiple imputation, offered here, or "
                "a principled model of the missingness is required.",
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
    # Audit IN-22 (ledger #21): toward zero only for one error-prone exposure under classical error.
    "why": "For repeated recalls, the mean is an acceptable exposure for ranking people, still "
           "measured with error: alone it is attenuated toward zero, but beside other error-prone "
           "intakes it can be inflated or flip sign. For visits, choose the baseline, the last "
           "visit or the change. When the outcome itself varies within a person, say which value "
           "is the outcome.",
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
        term("attenuation", "The shrinking of a lone exposure's association toward zero because "
                            "it is measured with error; averaging more days reduces it."),
        term("usual intake", "A person's long-run average intake, which single days only "
                             "estimate; modeling it properly is more than a mean."),
    ],
    "drawer": {"sections": [
        # Keogh et al. 2020 (STRATOS Part 1, §3.1.3): with several error-prone covariates "the
        # estimated coefficients … may be larger or smaller than the true target values in a rather
        # unpredictable manner"; Freedman et al. 2011 (JNCI 103:1086): "with two or more
        # mismeasured exposures, estimated relative risks may become attenuated, inflated, or can
        # even change direction".
        section("When the mean is adequate",
                "To rank people for regression, classification or a predictive model, the mean of "
                "the available recalls is an acceptable exposure and what most cohort analyses "
                "use. Under classical error it is attenuated toward zero only as the model's one "
                "error-prone exposure: with several error-prone nutrients, or energy, in the "
                "model, a coefficient can be attenuated, inflated or change sign (Keogh et al. "
                "2020; Freedman et al. 2011).",
                "CONVENTION", NUT03),
        section("When the mean is not adequate",
                "Prevalence or percentile claims about usual intake, episodically consumed foods "
                "with many zero days, and exposure coefficients that must be unbiased in "
                "magnitude all need more than a mean. This version corrects energy-adjusted "
                "exposure coefficients by regression calibration from the repeated recalls; it "
                "fits no usual-intake distribution.",
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
    # Audit IN-24: whole people are held out by their last observation, so with repeated rows their
    # earlier visits can predate training rows; the words say what is drawn (seal.py).
    "one_liner": "A random split is optimistic when the task looks forward; the held-out people "
                 "should then be those seen last.",
    "why": "With visits kept as rows, a random split lets the model learn from a person's later "
           "visits and be scored on their earlier ones. Validation in time, holding out those "
           "seen last, is a distinct check from validation on random rows, and reporting "
           "guidelines treat it so.",
    "consumer": "The seal and the folds: ordered by time when yes, grouped by person either way.",
    "options": [
        option("true", "Yes, later from earlier",
               "Those seen last are held out whole, earlier visits included; folds run forward "
               "in time."),
        option("false", "No",
               "The held-out rows are drawn at random, each person's rows kept together."),
    ],
    "terms": [
        GROUPED_SPLIT,
        term("chronological split", "A split that holds out the people seen last, whole, by their "
                                    "latest date; their earlier rows can predate training rows."),
    ],
    "drawer": {"sections": [
        section("Split by participant",
                "Split by participant, not by row, and fit everything learned from data inside "
                "the training fold only.",
                "SETTLED", NUT08),
        WHOLE_PIPELINE,
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
                "`WTDR2D` for both days, not the examination weight `WTMEC2YR`. Under inference "
                "the survey question then asks whether the estimates describe the surveyed "
                "population (weighted, with design-based intervals) or these participants.",
                "SETTLED", NUT01),
        section("Pooled cycles",
                "Combining NHANES cycles from 2001–2002 on means dividing the two-year weights by "
                "the number of cycles combined. 1999–2000 is the exception: its two-year weights "
                "and 2001–2002's rest on different censuses, so the 1999–2002 rows take the "
                "four-year weight, doubled, before the division (NHANES Analytic Guidelines "
                "2011–2016 §3.1.3–3.1.4). Confirm the same dietary method applies across cycles.",
                "SETTLED", NUT01),
        section("Units decide everything downstream",
                "1 kcal = 4.184 kJ, and a column ending `_pct_kcal` is a share of energy, not an "
                "amount. Every implausibility screen, energy adjustment and substitution is a "
                "function of total energy, so a unit error there reaches every result.",
                "SETTLED", NUT01),
    ]},
    "evidence": None,
}

SURVEY = {
    "key": "survey",
    "title": "Whose estimate it is",
    "question": "Should the estimates describe the surveyed population, or these participants?",
    "one_liner": "The population answer weights each row and takes intervals from the strata and "
                 "PSUs; the other is unweighted, and says so.",
    "why": "A survey oversamples some groups on purpose, so an unweighted estimate describes the "
           "sample, not the population, and can even change sign. Rows sampled in the same PSU "
           "resemble each other, so intervals that ignore the design are too narrow. Restricting "
           "the analysis keeps every row in the design, as a domain.",
    "consumer": "The coefficient table, its intervals, the eligibility rows and the methods "
                "section read it.",
    "options": [
        option("population", "Surveyed population",
               "Weighted estimates with Taylor-linearized intervals over the strata and PSUs."),
        option("sample", "These participants",
               "Unweighted; the methods state a sample-only estimand whose intervals ignore the "
               "design."),
    ],
    "terms": [
        term("PSU", "Primary sampling unit: the cluster, such as a county, drawn first; people "
                    "from one PSU resemble each other."),
        term("domain", "The rows an analysis is about. The others stay in the survey design, so "
                       "the variance still sees every stratum and PSU."),
        term("Taylor linearization", "The design-based variance of an estimate, from how much the "
                                     "PSU totals of its scores vary within each stratum."),
        ESTIMAND,
    ],
    "drawer": {"sections": [
        section("Why the design matters",
                "Without the weights, estimates are biased toward the oversampled groups; without "
                "strata and PSUs, standard errors are too small because clustering within PSUs is "
                "ignored. NCHS states that variance estimates computed under a simple-random-sample "
                "assumption are generally too low for NHANES.",
                "SETTLED", NUT01),
        section("Restrict by domain, never by deleting rows",
                "To restrict to a subgroup, keep every row and mark the subgroup (`subset()` on a "
                "`svydesign` object, `DOMAIN` in SAS). Deleting rows drops PSUs and strata from the "
                "variance and gives wrong standard errors. Degrees of freedom are the PSUs minus "
                "the strata that hold the subgroup (NHANES Analytic Guidelines 2011–2016 "
                "§3.2.3).",
                "SETTLED", NUT01),
        section("A stratum with one PSU",
                "Taylor linearization estimates a stratum's variance from the spread between its "
                "PSUs, and one PSU has none. TurboTab centers such a stratum at the mean of all PSU "
                "totals (R's lonely.psu \"adjust\", Stata's singleunit(centered)), which is "
                "conservative, and names the stratum.",
                "SETTLED", NUT01),
    ]},
    "evidence": ev("SETTLED", NUT01),
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
        # Audit MI-02 (ledger #29): Willett 2013's men's range is 800–4,000 (Banna et al. 2017,
        # quoting Nutritional Epidemiology, 3rd ed.); 800–4,200 is the Health Professionals
        # Follow-up Study's (Pan et al. 2011), beside the Nurses' Health Study's 500–3,500.
        option("willett_2013_by_sex", "Willett 2013, by sex",
               "Women outside 500–3,500 and men outside 800–4,000 kcal a day are excluded."),
        option("nhs_hpfs_by_sex", "NHS/HPFS, by sex",
               "Women outside 500–3,500 (NHS) and men outside 800–4,200 (HPFS) kcal a day are "
               "excluded."),
        option("sex_neutral_500_5000", "500–5,000 kcal a day",
               "Anyone outside 500–5,000 kcal a day is excluded, whatever their sex."),
        option("sex_neutral_500_3500", "500–3,500 kcal a day",
               "Anyone outside 500–3,500 kcal a day is excluded; stricter, so it removes more."),
        option("goldberg_schofield", "Goldberg, energy over BMR",
               "Energy over estimated BMR outside the cut-offs for a stated PAL and recall days "
               "is excluded."),
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
        # Audit ledger #34: Yamamoto et al. 2023 (eLife 12:e83616): "Whether one uses Goldberg
        # cutoffs should therefore be decided based on research purposes and not general rules."
        term("Goldberg cut-off", "A screen comparing reported energy with estimated basal "
                                 "metabolic rate, within limits that widen as recall days fall; "
                                 "widely used, and chosen by research purpose."),
    ],
    "drawer": {"sections": [
        section("Each criterion its own box",
                "Track the rows through every filter, each eligibility criterion separately, with "
                "its reason and its count. A single excluded box tells a reader nothing about who "
                "is missing from the model's population.",
                "CONVENTION", CLIN_A41),
        # Banna et al. 2017 (Front Nutr 4:45), quoting Willett 2013: "an allowable range of
        # 800–4,000 kcal/day for men may be used"; Pan et al. 2011 (AJCN 94:1088), NHS and HPFS:
        # "daily energy intake <800 or >4200 kcal/d for men and <500 or >3500 kcal/d for women".
        section("The screens in circulation",
                "Willett's textbook (2013) gives 500–3,500 kcal a day for women and 800–4,000 for "
                "men. The Nurses' Health Study (women) and the Health Professionals Follow-up "
                "Study (men) use 500–3,500 and 800–4,200. Variants use 5,000 as men's upper bound, "
                "or a sex-neutral 500–5,000 or 500–3,500. The conventions genuinely differ across "
                "literatures, so show how N moves with the choice.",
                "CONVENTION", NUT02),
        # Audit G15 (ledger #31): the 14-of-24 evidence is about Goldberg cut-offs, from one
        # simulation; Yamamoto et al. 2023: bias "was reduced but not completely eliminated by
        # Goldberg cutoffs in 14 of 24 nutrition-outcome pairs; bias was not reduced for the
        # remaining 10 cases".
        section("What exclusion does not fix",
                "In one simulation built on a biomarker study, Goldberg cut-offs reduced but did "
                "not remove bias in 14 of 24 nutrient–outcome associations and did not reduce it "
                "in the other 10; fixed kcal screens were not evaluated (Yamamoto et al. 2023). "
                "Whether to exclude at all is disputed: decide by the research purpose, not a "
                "general rule.",
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
    "one_liner": "The answer depends on the purpose: inference imputes with the outcome, prediction "
                 "without it.",
    "why": "Complete cases keep only rows with every predictor measured, which can remove a large, "
           "non-random share. Under inference, multiple imputation fills each blank many times "
           "from the other variables, the outcome included, and pools the answers. Under "
           "prediction, a fill learned in each training fold without the outcome lets the model "
           "impute a new row the same way.",
    "consumer": "The participant flow, each model's pipeline and the banner's row counts read it.",
    "options": [
        option("multiple_imputation", "Multiple imputation",
               "Under inference: each blank imputed 20 times with the outcome; results pooled by "
               "Rubin's rules."),
        option("complete_case", "Complete cases",
               "Rows missing any predictor are dropped; the participant flow shows how many."),
        option("impute", "Fill in each fold",
               "Under prediction: blanks filled in each training fold without the outcome; every "
               "row stays."),
    ],
    "terms": [
        term("complete cases", "Analyzing only rows with every variable measured; it can bias "
                               "estimates when who is missing depends on the outcome."),
        term("multiple imputation", "Filling each blank several times from its predicted "
                                    "distribution, analyzing each copy and pooling the results "
                                    "by Rubin's rules."),
        term("imputation", "Filling a missing value with an estimate learned from observed rows; "
                           "a single fill understates the uncertainty."),
        term("missing at random", "Missingness that the observed columns explain; the assumption "
                                  "standard multiple imputation relies on."),
    ],
    "drawer": {"sections": [
        section("One rule, by purpose", RULE, "SETTLED", CLIN_A2),
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
                "For inference, mean or median filling understates variance and biases "
                "coefficients; the clinical pack calls it indefensible in a manuscript, and here "
                "it is blocked until recorded. For prediction, a fill learned in each training "
                "fold is the deployable choice.",
                "SETTLED", CLIN_A2),
        section("Values below detection",
                "A blank below a detection limit is small, not unknown: the median puts it in the "
                "middle of the distribution. Half the minimum is customary; a censoring-aware "
                "fill is sound for associations.",
                "SETTLED", MET03),
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
        # The validation answer (audit ME-11, E16; turbotab/core/models/validation.py).
        term("optimism", "How much better a model scores on the rows it learned from than on new "
                         "rows; the bootstrap estimates it, refitting everything, and subtracts "
                         "it."),
        term("repeated cross-validation", "Cross-validation drawn again on fresh folds and "
                                          "averaged, so the score rests on no single partition."),
        term("internal–external validation", "Each site, study or period held out in turn and "
                                             "scored by models fit on the others, showing how "
                                             "performance varies."),
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
        WHOLE_PIPELINE,
        SINGLE_SPLIT,
        # Audit ME-11/G10: the unsourced "below about 50 rows" threshold was 2–4× too low; the
        # spread is now computed on the user's own rows (models/validation.py).
        section("How precise the scores are here",
                "Each cross-validated score carries its standard error, and each pair of families "
                "an interval for their difference, computed on these rows. Read whether two models "
                "differ from that interval, not from a rule about sample size.",
                "SETTLED", CLIN_A53),
    ]},
    "evidence": ev("CONVENTION", CLIN_A55),
}

ENERGY_ADJUSTMENT = {
    "key": "energy_adjustment",
    "title": "Energy adjustment",
    "question": "How should nutrient intakes be adjusted for total energy?",
    "one_liner": "People who eat more eat more of everything; each method answers a different "
                 "question about a nutrient.",
    # Audit IN-20 (ledger #43): adjusting for total energy changes the estimand (Tomova et al. 2022).
    "why": "Total energy drives every nutrient's intake, and the errors in reported nutrients and "
           "energy move together. Adjusting for it changes the question: the standard model and "
           "the residual method with energy kept swap calories between sources at fixed total "
           "energy; partition adds calories; all components gives each source's added and average "
           "swapped calories. Choose by the question.",
    "consumer": "Each model's pipeline, the column lineage, the coefficients and the substitution "
                "curves read it.",
    "options": [
        option("none", "No adjustment",
               "Total energy leaves the model: absolute intake, mixed with how much people eat."),
        option("standard", "Standard model",
               "Energy stays in the model: more of the nutrient in place of other calories."),
        option("residual", "Residual, energy kept",
               "Each nutrient's residual on energy, energy kept: the standard model's swap, in "
               "its units."),
        option("residual_energy_dropped", "Residual, energy dropped",
               "Each nutrient's residual on energy, energy dropped: differs when covariates track "
               "energy."),
        # Audit IN-21 (ledger #45): Tomova et al. 2022, model 3b: "a more accurate estimate than
        # the (unadjusted) nutrient density model, but one which is still biased".
        option("density_multivariate", "Density plus energy",
               "Nutrient per calorie, energy its own term: obscure, and still biased (Tomova "
               "2022)."),
        option("density", "Density alone",
               "Nutrient per calorie, energy dropped: a rescaled effect whose meaning is "
               "obscure."),
        option("partition", "Energy partition",
               "Calories from the nutrient and from everything else: adding calories, not "
               "substituting."),
        option("all_components", "All components",
               "Every energy source its own term: added calories, and each one's average swap."),
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
        term("all-components model", "A partition with every energy source its own term; a "
                                     "source's average swap is its coefficient less the others', "
                                     "weighted by their share of energy."),
        term("omitted energy source", "An energy source not in the model; total energy carries it "
                                      "in one composite, so every swap is partly in its place."),
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
                "With total energy also in the model, swapping a nutrient for its residual only "
                "reparametrizes it, so the coefficient, its interval and p are identical to the "
                "standard model's, covariates or not; that is the residual method here unless you "
                "choose to drop energy (McCullough & Byrd 2023). With energy dropped, the "
                "coefficient is the standard model's only when no covariate correlates with "
                "energy, and its interval is wider even then; otherwise it differs, sometimes in "
                "sign.",
                "SETTLED", NUT04),
        section("What the coefficient means",
                "An energy-adjusted coefficient is a substitution estimate: more of this nutrient "
                "and correspondingly less of everything else, at the same total intake — not the "
                "effect of simply eating more of it. With several energy sources in the model, "
                "\"everything else\" is only the sources left out of it; name them.",
                "SETTLED", NUT04),
        section("All components",
                "Every energy source its own term in kcal: each coefficient is the total effect of "
                "adding that source's calories, and its average swap is the coefficient less the "
                "other sources', weighted by their share of the remaining energy (Tomova et al. "
                "2022). It avoids composite variable bias at a cost in precision, one term per "
                "source, and its use is disputed (Willett, Stampfer & Tobias 2022).",
                "DISPUTED", NUT04),
        section("The biases they share",
                "Standard and residual models are biased even without confounding (composite "
                "variable bias), and all four models only partly account for confounding by "
                "common dietary causes; each evaluates a different estimand (Tomova et al. 2022, "
                "AJCN). The density model's coefficient is an obscure quantity, and with energy "
                "added as a term it is more accurate but still biased; the all-components model "
                "is the paper's recommended route.",
                "SETTLED", NUT04),
        section("The field's default",
                "The Willett residual method, computed within the final analytic sample and within "
                "sex, on log-transformed intakes, with the predicted nutrient at the mean energy "
                "added back. It is the field default, but not uncontested. Here total energy stays "
                "in the outcome model beside it unless you choose the energy-dropped form.",
                "CONVENTION", NUT04),
        section("When the outcome is BMI or adiposity",
                "Energy may be on the causal path and a collider at once. Present adjusted and "
                "unadjusted models and flag it in the limitations; the pack does not pick a side. "
                "An outcome named as weight, BMI, waist, body fat or diabetes raises this on the "
                "card.",
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
        # Audit G16 (ledger #54): the shared missing-values step runs before every family, so the
        # trees never see a blank (models/pipeline.py shared_steps).
        option("boosted_trees", "Boosted trees",
               "Many shallow trees: finds curves and interactions; gives no coefficients."),
        option("featurewise", "Feature-wise tests",
               "Tests each exposure on its own, adjusted for the covariates, with "
               "Benjamini–Hochberg false-discovery control; no predictions."),
        option("proportional_odds", "Proportional-odds model",
               "Cumulative odds ratios for an ordered outcome, the same at every cut-point; "
               "Brant's test checks it."),
        option("mixed", "Mixed model",
               "A random intercept per unit: model-based intervals when rows repeat, even within "
               "few units."),
        option("gee", "GEE",
               "Population-average effects, with intervals robust to how a unit's repeated rows "
               "correlate."),
        option("cox", "Cox model",
               "Hazard ratios for a time-to-event outcome, using every row's follow-up, censored "
               "or not."),
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
        P_OVER_N,
        section("Calibrate the trees",
                "Boosted trees often produce miscalibrated probabilities: report the calibration "
                "curve, or recalibrate on held-out data, never on training rows.",
                "SETTLED", CLIN_A51),
        CALIBRATION_TOO,
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
                "A review of 100 substitution studies (Louie & Bhowmik 2026) found 53% used "
                "unvalidated food-frequency variables; where validation was reported, correlations "
                "with reference methods ranged from 0.12 to 0.77.",
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
        WHOLE_PIPELINE,
        SINGLE_SPLIT,
    ]},
    "evidence": None,
}

ENTRIES = [LENS, ORIENTATION, REPAIRS, TARGET, EVENT, TASK, PURPOSE, GRAIN, REPEAT_KIND, UNIT,
           AGGREGATION, TEMPORAL, ROLES, SURVEY, EXCLUSIONS, MISSING, SPLIT, ENERGY_ADJUSTMENT,
           MODELS, SUBSTITUTION, OPEN_SEAL]

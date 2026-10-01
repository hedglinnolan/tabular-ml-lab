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
NUT04 = f"{_NUT}#04 · Energy adjustment — the methodological signature"
NUT05 = f"{_NUT}#05 · Compositional structure and substitution modeling"
NUT06 = f"{_NUT}#06 · Missing data"
NUT08 = f"{_NUT}#08 · Feature selection and modeling"
CLIN_A12 = f"{_CLIN}#A1.2 · Reference ranges vs physiological plausibility"
CLIN_A2 = f"{_CLIN}#A2 · Missing data"
CLIN_A51 = f"{_CLIN}#A5.1 · Calibration first"
CLIN_A55 = f"{_CLIN}#A5.5 · Modeling practice"
CLIN_B4 = f"{_CLIN}#B4 · Ordinal vs interval"
CLIN_B6 = f"{_CLIN}#B6 · Modeling"
GEN08 = f"{_GEN}#08 · Modeling at p >> n"
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
    "title": "Rows to exclude",
    "question": "Should any rows be excluded as implausible before modeling?",
    "one_liner": "An exclusion changes N and is reported in the participant flow, so nothing is "
                 "removed unless you choose it.",
    "why": "A 300 kcal recall is under-reporting, not starvation. The fixed screens genuinely "
           "differ across literatures, so each is offered with the rows it would remove from your "
           "table. Under-reporting concentrates in people with higher BMI, so excluding it "
           "removes a non-random slice; many analyses keep everyone and treat exclusion as a "
           "sensitivity analysis.",
    "consumer": "The participant flow, the split and every model's rows read it.",
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
               "Rows outside a range you set on any numeric column are excluded."),
    ],
    "terms": [
        term("implausible intake", "A reported day's energy intake too low or too high to be a "
                                   "real diet, usually a reporting error."),
        term("under-reporting", "Reporting less than was eaten. It is systematic, concentrated in "
                                "people with higher BMI and in weight-conscious participants."),
        term("Goldberg cut-off", "A screen comparing reported energy with estimated basal "
                                 "metabolic rate; the field's standard for misreporting, not "
                                 "offered in this version."),
    ],
    "drawer": {"sections": [
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
    "title": "Held-out rows",
    "question": "How many rows should be held out for one final, untouched score?",
    "one_liner": "Held-out rows are sealed until the end; every choice is tuned by "
                 "cross-validation on the rest.",
    "why": "A model scored on the rows it was tuned on grades its own homework. Cross-validation "
           "reuses every training row for scoring, fold by fold; a holdout adds one honest final "
           "number at the cost of training rows. At typical clinical sample sizes a single split "
           "is the weakest check, so keep the holdout modest.",
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

ENTRIES = [LENS, TARGET, TASK, PURPOSE, ROLES, EXCLUSIONS, MISSING, SPLIT, ENERGY_ADJUSTMENT,
           MODELS, SUBSTITUTION]

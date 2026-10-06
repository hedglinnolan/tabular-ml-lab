"""The routing leash's method contracts (BLUEPRINT §13, §11.3; the package LEASH).

Wave 2's LEASH package tightens or loosens the routing leash where the routing verifier found it
wrong (``docs/turbotab-next/INBOX.md``, its "Leash:" notes). Each change enters through the one
registry (``turbotab.core.contracts``) with its slot, scope, needs, routing (question, options with
both labels, rungs), storyboard, sentence and relations, and each relation names the code that makes
it fire; the acceptance chain test (``acceptance/test_leash_chain.py``) asserts every one fires.

* **The adjustment card's guesses** (``turbotab/core/covariate_guesses.py``): clinical
  measurements, medications and lifestyle under the exposure–outcome pairing, covariates with the
  same guess in one block, a multi-select answer that settles exactly what it lists.
* **The grouping question by structure** (``turbotab/core/groupings.py``): any column that can
  group rows is asked under inference, with its guess.
* **Imputed copies are not repeats** (``turbotab/core/structural.py``): the repeats answer over
  rows read as imputed copies is blocked and recorded, the Rubin's-rules answer its first exit.
* **The E-value's SD** (MODELING_SEQUENCE §0 ruling 14): the design-weighted SD under the surveyed
  population, in the effects stage and the causal lane alike.
* **Fit statistics wait with the estimates** (MODELING_SEQUENCE §1 row 11): under inference a
  withheld fit serves no cross-validated R², RMSE or MAE.
* **Censored values below a detection limit** are the below-detection repair's, and a text predictor
  confirmed "amount" is checked as the numbers the fit reads.
"""
from __future__ import annotations

from typing import Any

from turbotab.core import contracts

PACKAGE = "LEASH"
_INFERENCE = {"inference": "recommended", "prediction": "not_offered"}
_NOT_UNDER_PREDICTION = "Not offered: under prediction no covariate's causal place is asked."


def _option(key: str, label: str, customary: str, sound: dict[str, str],
            rung: dict[str, str]) -> contracts.ContractOption:
    return contracts.ContractOption(key, label, customary, sound=sound,  # type: ignore[arg-type]
                                    rung=rung)


def _relation(id: str, kind: str, target: str, condition: str, says: str, enforced_by: str,
              rung: str | None = None, exits: tuple[str, ...] = (),
              purposes: tuple[str, ...] = ("inference",)) -> contracts.Relation:
    return contracts.Relation(kind, target, says,  # type: ignore[arg-type]
                              purposes=purposes, rung=rung, condition=condition, id=id,
                              enforced_by=enforced_by, exits=exits)


def _contract(**fields: Any) -> contracts.MethodContract:
    return contracts.MethodContract(package=PACKAGE, **fields)


CONTRACTS = tuple(contracts.register_contract(c) for c in (
    _contract(
        key="adjustment_guesses",
        label="The adjustment card's guesses, by exposure–outcome pairing",
        slot="model", scope="descriptive",
        scope_note="reads the covariates' names, never a row: each guess is a proposal the user "
                   "confirms, and the answers declare which covariates enter the model",
        needs=("a declared exposure", "the outcome", "the covariates' recorded roles"),
        question="adjustment", place="MODELING_SEQUENCE §1 step 3", decision="set_adjustment",
        stage="proposals", leash=_INFERENCE,
        storyboard=("each covariate read into its class", "the guess under the exposure–outcome "
                    "pairing, with its source", "covariates with the same guess in one block",
                    "one tap answers exactly the block's columns"),
        sentence="turbotab.core.voice:_set_adjustment",
        options=(
            _option("accept_block", "Accept a block's guess (one tap)",
                    "the field's nested models: Model 1 age, sex, energy; Model 2 demographics and "
                    "lifestyle; Model 3 body composition; Model 4 mutually adjusted nutrients "
                    "(NUTRITION_PACK §08)",
                    {"inference": "Sound when the guess fits the covariate: the criterion derives "
                                  "its role, and every guess shows its source",
                     "prediction": _NOT_UNDER_PREDICTION},
                    {"inference": "recommended", "prediction": "not_offered"}),
            _option("multi_select", "Answer chosen covariates together",
                    "uncommon: covariates are usually listed by hand",
                    {"inference": "Sound: one answer settles exactly the covariates it lists",
                     "prediction": _NOT_UNDER_PREDICTION},
                    {"inference": "available", "prediction": "not_offered"}),
            _option("one_by_one", "Answer each covariate",
                    "customary for a few covariates",
                    {"inference": "Sound, and light only for a few covariates",
                     "prediction": _NOT_UNDER_PREDICTION},
                    {"inference": "available", "prediction": "not_offered"}),
        ),
        relations=(
            _relation("guess-pair-for-measurements", "implies", "the with-and-without pair",
                      "a body measure, a clinical measurement or a medication measured with the "
                      "exposure, its block's guess accepted",
                      "the primary leaves it out and the declared secondary adds it (Model 3)",
                      "turbotab.core.covariate_guesses:guess"),
            _relation("guess-outcome-measure-out", "implies", "left out",
                      "a measurement of the outcome's own group taken at the same visit (HbA1c "
                      "beside glucose, HDL beside LDL)",
                      "it measures the state the outcome measures, so the criterion leaves it "
                      "out, and the card and the record call it another measure of the outcome's "
                      "own kind, never a consequence of the outcome or a possible collider",
                      "turbotab.core.covariate_guesses:guess"),
            _relation("guess-baseline-outcome-kind-pair", "implies", "the with-and-without pair",
                      "a baseline measurement of the outcome's own group beside an outcome over "
                      "follow-up (HbA1c beside incident diabetes, blood pressure beside incident "
                      "hypertension, LDL beside LDL at twelve months)",
                      "measured before the outcome it is no consequence of it, so it takes its "
                      "class's guess: declared without it and, beside, with it",
                      "turbotab.core.covariate_guesses:guess"),
            _relation("block-settles-listed", "implies", "exactly the listed covariates",
                      "a block's tap or a multi-select answer",
                      "the answer settles exactly the columns it lists, never another",
                      "turbotab.core.decisions:fold"),
            _relation("mediator-kept-blocked", "conflicts", "a mediator in a total-effect set",
                      "a covariate guessed a possible mediator answered to be kept in the primary",
                      "blocked and recorded: the exits leave it out with the secondary beside, or "
                      "keep it recorded as not a total effect",
                      "turbotab.core.estimand:_mediators_stay_out_of_a_total_effect",
                      rung="block_and_record",
                      exits=("leave it out, further adjusted for it beside",
                             "keep it, recorded: not a total effect")),
            _relation("bulk-roles-keep-confirmations", "implies",
                      "the settled roles and the adjustment answers",
                      "a bulk roles answer that re-records a settled role unchanged",
                      "a role confirmed one by one stays confirmed, so the adjustment set is not "
                      "reopened", "turbotab.core.decisions:_roles_record_what_rode_along"),
        ),
        sources=("VanderWeele 2019, Eur J Epidemiol 34:211–219",
                 "Schisterman, Cole & Platt 2009, Epidemiology 20:488",
                 "Tobin et al. 2005, Stat Med 24:2911–2935", "NUTRITION_PACK §08")),
    _contract(
        key="grouping_by_structure", label="The grouping question, asked by structure",
        slot="model", scope="descriptive",
        scope_note="reads every row's value counts, never the outcome, to decide whether the "
                   "question is asked; the user's answer decides the intervals",
        needs=("a column whose values repeat as labels do",), question="clusters",
        place="MODELING_SEQUENCE §2 (repeated units or clusters)", decision="set_clusters",
        stage="roles", leash={"inference": "recommended", "prediction": "available"},
        storyboard=("every column that can group rows", "the guess from its name or values",
                    "the user's answer", "the intervals clustered by it, or recorded as not"),
        sentence="turbotab.core.voice:_set_clusters",
        options=(
            _option("fixed_effects", "Its own intercept per group, and clustered intervals",
                    "customary for a few sites (a site term)",
                    {"inference": "Sound: between-group confounding removed, intervals CR2 with "
                                  "Bell–McCaffrey df",
                     "prediction": "Not offered: under prediction a grouping decides validation"},
                    {"inference": "recommended", "prediction": "not_offered"}),
            _option("cluster_only", "Clustered intervals only",
                    "customary (cluster-robust errors)",
                    {"inference": "Sound for the intervals; between-group confounding stays",
                     "prediction": "Not offered: under prediction a grouping decides validation"},
                    {"inference": "available", "prediction": "not_offered"}),
            _option("group", "Rows sharing it are one group: the folds keep each group whole",
                    "customary for a multicenter prediction model (internal–external "
                    "validation by center)",
                    {"inference": "Not offered: under inference a grouping is adjusted for or "
                                  "clusters the intervals",
                     "prediction": "Sound: no group trains and scores at once, so the score is a "
                                   "new group's; internal–external cross-validation folds by it "
                                   "(Steyerberg & Harrell 2016)"},
                    {"inference": "not_offered", "prediction": "recommended"}),
            _option("none", "Nothing groups the participants",
                    "customary when no site or household is named",
                    {"inference": "Unsound over a column that reads as a grouping: intervals too "
                                  "narrow",
                     "prediction": "Sound when validation need not keep groups whole"},
                    {"inference": "block_and_record", "prediction": "available"}),
        ),
        relations=(
            _relation("structure-asks", "implies", "the grouping question",
                      "a column with more than ten repeating values and several rows on each, "
                      "under inference, whatever its name or the shape of its counts (a "
                      "measurement's shape turns the guess to no; a measured word beside a "
                      "grouping word lets the values lead)",
                      "the grouping question is asked of it, with its guess; only a name read as a "
                      "measured quantity and no grouping scopes a column out",
                      "turbotab.core.groupings:candidates"),
            _relation("grouping-implies-cr2", "implies", "cluster-robust intervals",
                      "a grouping answered under inference",
                      "the intervals are CR2 by it with Bell–McCaffrey degrees of freedom",
                      "turbotab.core.models.inference:resolve_clusters"),
            _relation("none-over-grouping", "conflicts", "independent intervals over a grouping",
                      "\"nothing groups them\" over a column guessed to group the participants",
                      "blocked and recorded: the exits adjust and cluster, say that a column "
                      "guessed from its values alone marks no group (its reading confirmed no), "
                      "or record the answer",
                      "turbotab.core.estimand:_no_grouping_is_recorded", rung="block_and_record",
                      exits=("adjust for it and cluster by it",
                             "a column guessed from its values alone marks no group",
                             "record that they group nothing")),
        ),
        sources=("MODELING_SEQUENCE §2", "Bell & McCaffrey 2002, Surv Methodol 28:169",
                 "Steyerberg & Harrell 2016, J Clin Epidemiol 69:245–247")),
    _contract(
        key="copies_not_repeats", label="Rows read as imputed copies are not repeats",
        slot="reshape", scope="row_local",
        scope_note="which rows are copies of one record is read from the rows' own copy numbers; "
                   "nothing is learned from other rows or the outcome",
        needs=("rows read as imputed copies (a copy number such as NHANES's _MULT_)",),
        question="repeat_kind", place="the opening sequence (repeat kind)",
        decision="set_repeat_kind", stage="structure",
        leash={"inference": "block_and_record", "prediction": "available"},
        storyboard=("the rows read as copies of one record", "each copy analyzed as a completed "
                    "dataset", "the estimates pooled by Rubin's rules"),
        sentence="turbotab.core.voice:_set_repeat_kind",
        options=(
            _option("imputed_copies", "Imputed copies, pooled by Rubin's rules",
                    "NCHS's direction for its DXA files",
                    {"inference": "Sound: the imputation's variability enters the intervals",
                     "prediction": "Sound"},
                    {"inference": "recommended", "prediction": "available"}),
            _option("repeats", "Repeated measurements",
                    "customary for a unit's repeated rows",
                    {"inference": "Unsound over imputed copies: imputed values analyzed as "
                                  "measured, intervals too narrow",
                     "prediction": "The copies' concern is stated"},
                    {"inference": "block_and_record", "prediction": "available"}),
        ),
        relations=(
            _relation("copies-as-repeats-blocked", "conflicts", "repeats over imputed copies",
                      "the repeats or time-points answer over rows read as imputed copies, under "
                      "inference",
                      "blocked and recorded: the exits answer imputed copies (Rubin's rules) or "
                      "record the answer",
                      "turbotab.core.structural:copies_refusal", rung="block_and_record",
                      exits=("imputed copies, each analyzed and pooled by Rubin's rules",
                             "keep the answer, recorded: intervals too narrow")),
            _relation("copies-imply-pooling", "implies", "pooling by Rubin's rules",
                      "the imputed-copies answer under inference, each copy kept as a record",
                      "every estimate is pooled over the copies (MODELING_SEQUENCE §2)",
                      "turbotab.core.stages.modeling:imputed_copies_column"),
        ),
        sources=("CDC, NHANES 1999–2006 DXA, Multiple Imputation Details",)),
    _contract(
        key="evalue_sd", label="The E-value of a difference, standardized by the estimand's SD",
        slot="evaluation", scope="descriptive",
        scope_note="reads the analyzed rows' outcome (and their survey weights) to standardize "
                   "the reported estimate; informs no modeling choice",
        needs=("a reported difference", "the survey answer"), question="stated",
        place="MODELING_SEQUENCE §0 ruling 14", stage="effects",
        leash={"inference": "recommended", "prediction": "not_offered"},
        storyboard=("the reported difference", "the outcome's SD the estimand speaks of",
                    "d = estimate / SD", "RR ≈ exp(0.91 d), and its E-value"),
        sentence="turbotab.core.stages.effects:methods_sentence",
        options=(
            _option("design_weighted", "The surveyed population's SD",
                    "uncommon (most reports use the sample SD)",
                    {"inference": "Sound under the surveyed-population answer: the estimate is "
                                  "the population's",
                     "prediction": "Not offered: a prediction reports no effect"},
                    {"inference": "recommended", "prediction": "not_offered"}),
            _option("sample", "The analyzed rows' SD",
                    "customary (EValue::evalues.OLS's sd)",
                    {"inference": "Sound under the sample answer",
                     "prediction": "Not offered: a prediction reports no effect"},
                    {"inference": "available", "prediction": "not_offered"}),
        ),
        relations=(
            _relation("evalue-population-sd", "implies", "the design-weighted SD",
                      "the surveyed-population answer and a design-based difference",
                      "the effects stage and the causal lane standardize the difference by the "
                      "outcome's design-weighted SD, and the methods text says so",
                      "turbotab.core.models.effects:estimand_sd"),
            _relation("evalue-sample-sd", "implies", "the sample SD",
                      "the sample answer, or no survey design",
                      "the difference is standardized by the analyzed rows' own SD",
                      "turbotab.core.models.effects:estimand_sd"),
        ),
        sources=("VanderWeele & Ding 2017, Ann Intern Med 167:268",)),
    _contract(
        key="fit_statistics_withheld", label="Outcome-model fit statistics wait with the estimates",
        slot="evaluation", scope="model",
        scope_note="the cross-validated scores are the outcome model's own, fit on the analyzed "
                   "rows and their outcome",
        needs=("a fit under inference",), question="stated", place="MODELING_SEQUENCE §1 row 11",
        stage="fit", leash={"inference": "recommended", "prediction": "not_offered"},
        storyboard=("the plan open", "estimates and fit statistics withheld", "the plan answered",
                    "both shown"),
        sentence="turbotab.core.routing_leash:withheld_sentence",
        options=(
            _option("withheld", "Withheld until the plan is answered",
                    "uncommon: software reports R² beside every fit",
                    {"inference": "Sound: under inference an R² describes the outcome model, not "
                                  "the estimate, and reading it before the plan is a fork",
                     "prediction": "Not offered: under prediction the scores are the result"},
                    {"inference": "recommended", "prediction": "not_offered"}),
        ),
        relations=(
            _relation("plan-open-withholds-scores", "implies", "no fit statistics",
                      "under inference, a question the estimate rests on unanswered",
                      "the fit serves no cross-validated R², RMSE or MAE, nor any other score",
                      "turbotab.core.estimand:withhold"),
        ),
        sources=("Gelman & Loken 2013 (the garden of forking paths)",)),
    _contract(
        key="censored_below_detection",
        label="Values below a detection limit belong to the below-detection repair",
        slot="repairs", scope="row_local",
        scope_note="each value is read from its own cell (half its limit, or its limit over √2); "
                   "no other row is read",
        needs=("a text column with values below a detection limit (<0.20)",),
        question="findings", place="the repairs before the seal", decision="apply_repair",
        stage="findings", leash={"inference": "recommended", "prediction": "recommended"},
        storyboard=("the cell written <0.20", "its limit read", "the value set to a fraction of "
                    "it", "the column read as numbers"),
        sentence="turbotab.core.voice:_apply_repair",
        options=(
            _option("half_limit", "Half the limit", "customary (LOD/2)",
                    {"inference": "Conditional: biased as the censored share grows",
                     "prediction": "Conditional: a censoring indicator beside it helps"},
                    {"inference": "available", "prediction": "available"}),
            _option("limit_root2", "Limit over √2", "customary (LOD/√2)",
                    {"inference": "Conditional: biased as the censored share grows",
                     "prediction": "Conditional: a censoring indicator beside it helps"},
                    {"inference": "available", "prediction": "available"}),
        ),
        relations=(
            _relation("censored-left-to-repair", "implies", "the below-detection repair",
                      "a lab column's values below a detection limit that the app's own finding "
                      "reads",
                      "the lab pack's censored-values finding leaves them to that finding's "
                      "repair, never saying it has no control",
                      "turbotab.core.stages.finding_words:censored_rows",
                      purposes=("prediction", "inference")),
            _relation("text-amount-checked", "implies", "the plausibility checks on its numbers",
                      "a text predictor confirmed to hold amounts",
                      "its values are checked as the numbers the fit reads, before the seal",
                      "turbotab.core.repairs:ledger_amounts",
                      purposes=("prediction", "inference")),
        ),
        sources=("CLINICAL_SURVEY_PACK §A1.3",)),
))

def withheld_sentence(*_: Any) -> str:
    """What a withheld inference fit says about its scores (``estimand.SCORES_WITHHELD``)."""
    from turbotab.core.estimand import SCORES_WITHHELD

    return SCORES_WITHHELD


__all__ = ["CONTRACTS", "PACKAGE", "withheld_sentence"]

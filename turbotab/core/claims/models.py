"""The models area's claims (CLAIMS_REGISTRY.md §1, R0_PLAN.md §R0b): tuning, the shelf's rows,
the baseline verdict's floor, and two materiality rules, each as stage 1 left the code
(claims backtest A-001, A-029, A-124, C-170). Each number lives in its claim's parameters alone;
the engine constant that holds it reads it with ``claims.value``.

* ``tuning.small_sample_standard`` (A-001): below an effective-size threshold the plan draws no
  search, and boosted trees and XGBoost are fit once at a small-sample standard (RECIPES §4.6,
  ruled 2026-10-11). ``tuning.SMALL_N`` reads it, and so do the four family sentences that state
  it, boosted trees' and XGBoost's tuning reasons and descriptions. Its study scored predictions
  of a yes/no or numeric outcome, and the plan reads neither purpose nor task, so two fallbacks
  record where it applies the same rule as judgments: ``tuning.small_sample_standard_estimate``
  under Estimate, carried from Predict, and ``tuning.small_sample_standard_classes`` for several
  classes or ordered ones under either purpose, carried from a yes/no outcome. A searched family
  with no small-sample standard of its own (the random forest) draws no search below the same
  threshold either and is fit once at its standard settings: the judgment
  ``tuning.small_sample_no_search``, the last fallback.
* ``boosted_trees.small_rows`` and ``forest.small_rows``: the rows below which the shelf rates
  each family fair, the rows it ranks for (training rows under Predict, every analyzed row under
  Estimate or Describe, as the shelf stage counts them). Both modules named theirs ``SMALL_N``,
  beside the tuning's threshold of the same name, and their values are equal, but they are two
  claims, not one read by both:
  RECIPES §2.6 gives each family its own row of base scores, all conventions; it states the
  forest's rows and not boosted trees', which come from draft 2's shelf with a second, poor step
  and a reason of their own (they overfit easily). One claim would move the forest's rating
  whenever boosted trees' evidence moved.
* ``baseline.min_gain`` (A-029): the floor a gain over the no-predictor baseline must exceed, on
  its own scale for a score better when higher, as a share of the baseline's loss on a loss.
* ``materiality.rows_smd_floor`` (A-124): the energy screen's rows times SMD never says "doesn't
  change your numbers here". Its construction moves a coefficient under Estimate; the floor reads
  no purpose, so under Predict its fallback ``materiality.rows_smd_floor_prediction`` records it on
  the score, a judgment.
* ``materiality.collinear_under_prediction`` (C-170): under Predict the collinear line is a
  Confirm, both of a pair kept and no selection recorded; reading their importance together is
  the line's advice, since nothing groups it (BEATS_PREDICT M4's grouped importance is not built).

The regime predicates below read the context the caller passes, a mapping or an object with
those names, and nothing of the engine: each is written from the regime's words, never by calling
the code it guards (CLAIMS_REGISTRY §10, C-123). This module imports nothing but ``claims``.
"""
from __future__ import annotations

from typing import Any, Mapping

from turbotab.core.claims import (RETIRED, Basis, Claim, Fires, Param, Part, Regime, Silent,
                                  register)

HERE = "turbotab.core.claims.models"
AUTHOR = "R0b-models builder"
# The author of the claim the R0b finish added (tuning.small_sample_no_search), so that its grader
# above judgment is checked against the right name (CLAIMS_REGISTRY.md §2.4).
FINISH_AUTHOR = "R0b-finish builder"
TESTS = "turbotab/core/tests/test_claims_models.py"


# ── the regime predicates ────────────────────────────────────────────────────────────────────


def _field(ctx: Any, name: str) -> Any:
    if isinstance(ctx, Mapping):
        return ctx.get(name)
    return getattr(ctx, name, None)


def under_prediction(ctx: Any) -> bool:
    """The question is Predict."""
    return _field(ctx, "purpose") == "prediction"


def under_inference(ctx: Any) -> bool:
    """The question is Estimate or Describe."""
    return _field(ctx, "purpose") == "inference"


def under_either_purpose(ctx: Any) -> bool:
    """The question is Predict, or Estimate or Describe: a purpose is set."""
    return _field(ctx, "purpose") in ("prediction", "inference")


def standard_from_the_study(ctx: Any) -> bool:
    """The family's small-sample standard is the one notes/SMALL_SAMPLE_STANDARD.md chose, as its
    tuning declaration says (``small_standard_source``): the declaration is read, never the
    family's key (MODEL_FAMILY_CONTRACT C1)."""
    return "SMALL_SAMPLE_STANDARD.md" in str(_field(ctx, "small_standard_source") or "")


def searched_without_a_small_sample_standard(ctx: Any) -> bool:
    """The family's tuning declaration is a search (``kind``) and names no small-sample standard
    notes/SMALL_SAMPLE_STANDARD.md chose (``small_standard_source``): the declaration is read,
    never the family's key (MODEL_FAMILY_CONTRACT C1)."""
    return (_field(ctx, "kind") == "search"
            and "SMALL_SAMPLE_STANDARD.md" not in str(_field(ctx, "small_standard_source") or ""))


def yes_no_or_numeric(ctx: Any) -> bool:
    """The outcome is yes/no or a number, the two the small-sample study simulated."""
    return _field(ctx, "task") in ("binary", "regression")


def classes_or_ordered(ctx: Any) -> bool:
    """The outcome is one of several classes, or of several ordered ones: the two the small-sample
    study did not simulate."""
    return _field(ctx, "task") in ("multiclass", "ordinal")


def rated_before_any_fit(ctx: Any) -> bool:
    """The shelf's situation: the rows it ranks for and the columns counted, as whole numbers,
    before any model is fit. Which family is rated is its own ``assess``'s, the claim's
    enforcer."""
    rows, columns = _field(ctx, "n_rows"), _field(ctx, "n_features")
    return isinstance(rows, int) and isinstance(columns, int) and rows >= 0 and columns >= 0


def verdict_metric(ctx: Any) -> bool:
    """The score a verdict against the baseline is read on: R² for a number, log loss for a
    yes/no or class outcome, the ranked probability score for an ordered one, the Brier score at
    the horizon for time to event, and Harrell's C for a time to event with delayed entry."""
    return _field(ctx, "metric") in ("r2", "log_loss", "rps", "brier_t", "c_index")


def rows_times_smd(ctx: Any) -> bool:
    """The movement is predicted by rows changed times the leavers' largest SMD."""
    return _field(ctx, "instrument") == "rows_smd"


def yes_no_left_uncorrected(ctx: Any) -> bool:
    """A yes/no outcome whose fit applies no imbalance correction (``imbalance`` "none", or no
    levers answered)."""
    return _field(ctx, "task") == "binary" and _field(ctx, "imbalance") in (None, "none")


def held_out_by_a_cut_point(ctx: Any) -> bool:
    """The split holds rows out by a cut-point: it names a column and a value, and draws no share
    beside it (SWARM_PLAN §8's temporal seal by cut-point)."""
    cut = _field(ctx, "cut_point")
    return (cut is not None and bool(_field(cut, "column")) and _field(cut, "at") not in (None, "")
            and not _field(ctx, "holdout"))


def held_out_in_whole_units(ctx: Any) -> bool:
    """The seal keeps each unit's rows together, and some unit has more than one row: its
    ``grouped`` is true (``seal.cut_holdout``, ``voice._set_split``)."""
    return bool(_field(ctx, "grouped"))


def held_out_row_by_row(ctx: Any) -> bool:
    """The seal holds rows out one by one: one row per unit, or rows the seal does not group."""
    return not held_out_in_whole_units(ctx)


def drawn_by_row(ctx: Any) -> bool:
    """The seal could not keep a unit's rows together: its basis is ``abandoned`` (a unit's rows
    repeat, or were said to, and nothing groups them) or ``undetermined`` (whether they can repeat
    was answered as not known)."""
    return _field(ctx, "state") in ("abandoned", "undetermined")


def whole_numbers_asked(ctx: Any) -> bool:
    """Some predictor's numbers (whole, or written as text) are asked whether they are codes or
    amounts: the caller names them (``asked``)."""
    return bool(_field(ctx, "asked"))


def blank_on_a_yes_no_or_medication_question(ctx: Any) -> bool:
    """A yes/no column, or one named for a medication, blank on at least the claim's share of the
    rows with the outcome measured (``share``, ``yes_no``, ``medication``)."""
    share = _field(ctx, "share")
    return (bool(_field(ctx, "yes_no") or _field(ctx, "medication"))
            and isinstance(share, (int, float)) and not isinstance(share, bool)
            and share >= NOT_ASKED.value)


def energy_days_unsettled(ctx: Any) -> bool:
    """The energy column's unit is settled and how many days each value spans is not
    (``days_unsettled``, ``unit_settled``)."""
    return bool(_field(ctx, "days_unsettled")) and bool(_field(ctx, "unit_settled"))


# ── tuning (A-001) ───────────────────────────────────────────────────────────────────────────

SMALL_SAMPLE_STANDARD = register(Claim(
    id="tuning.small_sample_standard",
    kind="rule",
    card="Below an effective size of {small_n}, boosted trees and XGBoost are not tuned: each is "
         "fit once at a small-sample standard, which in simulation predicted about as well as "
         "tuning and stayed calibrated, where the libraries' defaults did not.",
    methods="Below an effective size of {small_n} (units for a numeric outcome; events, or the "
            "rarest class's count, for a yes/no or class outcome) the tuning plan drew no "
            "candidate settings for a searched family, and boosted trees and XGBoost were fit "
            "once at a small-sample standard chosen by simulation in place of the libraries' "
            "defaults; from {small_n} their settings were searched inside every training fold, "
            "the defaults among the candidates.",
    goal=("prediction",),
    regime=Regime(
        "boosted trees or XGBoost under Predict on a yes/no or numeric outcome, at any effective "
        "size, since both sentences are conditioned on the threshold they state; the standard "
        "below it rests on notes/SMALL_SAMPLE_STANDARD.md's simulation (ten normal "
        "predictors, 100 to 1,000 rows, a yes/no outcome chosen on and a numeric one checked), "
        "whose other limits the scope part names",
        requires=(f"{HERE}:standard_from_the_study", f"{HERE}:yes_no_or_numeric",
                  f"{HERE}:under_prediction")),
    decision="set_tuning",
    options=("automatic", "lighter", "standard", "manual"),
    verdict="recommended",
    parts=(
        Part("value", grade="judgment", basis=Basis(
            reason="RECIPES §4.6 calls 300 a convention, and the 2026-10-11 ruling kept it: the "
                   "small-sample study's cells all sat below it (at or above 300 in 1 of 680 "
                   "datasets), so the study does not locate the threshold.")),
        Part("direction", grade="judgment", basis=Basis(
            reason="notes/SMALL_SAMPLE_STANDARD.md: across 14 small-sample cells the libraries' "
                   "defaults were the worst arm in every one (regret 0.10 to 0.39 nats, "
                   "calibration slopes 0.2 to 0.5), and the standard matched nested tuning to the "
                   "K floor on loss with a third of its calibration spread. Its constants were "
                   "chosen with every cell in view, so those cells are not held out; "
                   "test_small_sample_standard.py checks it again on T1's signal generator at "
                   "n = 200.")),
        Part("scope", grade="judgment", basis=Basis(
            reason="The study fit the two boosting families on ten normal predictors with a "
                   "yes/no outcome, a numeric one checked; multiclass outcomes, categorical "
                   "predictors, p = 50 and NHANES subsamples are to be checked before release "
                   "(RECIPES §4.6), and ordered outcomes are not checked either. It scored "
                   "predictions alone. The plan reads "
                   "neither the task nor the purpose, so it applies the same rule under Estimate "
                   "and to several classes or ordered ones, each its fallback's judgment "
                   "(tuning.small_sample_standard_estimate, then "
                   "tuning.small_sample_standard_classes).")),
    ),
    params={"small_n": Param(300, unit="units of effective size (units, events or the rarest "
                                       "class's count, in one outer training fold)",
                             constant="turbotab.core.models.tuning:SMALL_N")},
    tests=(
        Fires("turbotab/core/tests/test_small_sample_standard.py::"
              "test_below_300_the_standard_fit_is_calibrated_and_predicts_as_well_as_nested_tuning",
              truth="independent:statsmodels"),
        Fires("turbotab/core/tests/test_boosted_trees_tuned.py::"
              "test_below_300_the_tuned_fit_is_a_direct_fit_at_the_small_sample_standard",
              truth="independent:scikit-learn"),
        Fires("turbotab/core/tests/test_xgboost_family.py::"
              "test_below_300_the_standard_is_the_small_sample_standard",
              truth="independent:xgboost"),
        Fires("turbotab/core/tests/test_boosted_trees_tuned.py::"
              "test_the_small_sample_standard_never_stops_early", truth="construction"),
        Fires(f"{TESTS}::test_below_an_effective_size_of_300_a_searched_family_is_fit_once_at_its_"
              "standard", truth="construction"),
        Silent(f"{TESTS}::test_from_an_effective_size_of_300_the_search_runs_with_the_library_"
               "defaults_first", truth="construction"),
        Silent("turbotab/core/tests/test_boosted_trees_tuned.py::"
               "test_the_standard_candidate_is_scikit_learns_own_defaults",
               truth="independent:scikit-learn"),
        Silent("turbotab/core/tests/test_xgboost_family.py::"
               "test_the_standard_settings_are_xgboosts_own_defaults",
               truth="independent:xgboost"),
    ),
    enforced_by=("turbotab.core.models.tuning:standard_for",
                 "turbotab.core.models.tuning:searched_size",
                 "turbotab.core.models.tuning:make_plan"),
    fallback="tuning.small_sample_standard_estimate",
    author=AUTHOR,
    origin="audit",
    ruled="RECIPES_AND_TUNING.md §4.6, ruled 2026-10-11 (built 2026-10-15)",
))

SMALL_SAMPLE_STANDARD_ESTIMATE = register(Claim(
    id="tuning.small_sample_standard_estimate",
    kind="rule",
    card="Under Estimate too, below an effective size of {small_n} boosted trees and XGBoost are "
         "not tuned but fit once at the small-sample standard chosen for Predict: a judgment, "
         "since the simulation that chose it scored predictions and their calibration, not the "
         "curves an estimate is read from.",
    methods="Under Estimate or Describe the tuning plan followed the same rule as under Predict, "
            "a judgment carried from it: below an effective size of {small_n} it drew no "
            "candidate settings, and boosted trees and XGBoost were fit once at the small-sample "
            "standard, which its simulation chose on predictive loss and calibration, not on the "
            "effect curves these families describe under Estimate; from {small_n} their settings "
            "were searched as under Predict, the defaults among the candidates.",
    goal=("inference",),
    regime=Regime(
        "boosted trees or XGBoost under Estimate or Describe on a yes/no or numeric outcome, at "
        "any effective size, since both sentences are conditioned on the threshold they state: "
        "the plan reads no purpose, and no study of its own covers this purpose",
        requires=(f"{HERE}:standard_from_the_study", f"{HERE}:yes_no_or_numeric",
                  f"{HERE}:under_inference"),
        disjoint_from=("tuning.small_sample_standard",)),
    decision="set_tuning",
    options=("automatic", "lighter", "standard", "manual"),
    verdict="recommended",
    parts=(
        Part("value", grade="judgment", basis=Basis(
            reason="The threshold under Predict, a convention (RECIPES §4.6, ruled "
                   "2026-10-11), read here unchanged: nothing locates one for an estimate.")),
        Part("direction", grade="judgment", basis=Basis(
            reason="Carried from Predict: notes/SMALL_SAMPLE_STANDARD.md §1 scored each arm by "
                   "its excess log loss and calibration slope on fresh rows, never an effect "
                   "curve's bias or coverage. That a calibrated fit which predicts as well as "
                   "tuning also draws a better curve than the overconfident defaults is "
                   "expected, not measured.")),
        Part("scope", grade="judgment", basis=Basis(
            reason="The same design limits as under Predict (ten normal predictors, a yes/no "
                   "outcome chosen on and a numeric one checked), and no effect estimated. Several "
                   "classes or ordered ones, to which the plan applies the same rule under "
                   "either purpose, are its fallback's.")),
    ),
    # The threshold under Predict, read from that claim here: one number in one place.
    params={"small_n": Param(SMALL_SAMPLE_STANDARD.params["small_n"].value,
                             unit=SMALL_SAMPLE_STANDARD.params["small_n"].unit)},
    tests=(
        Fires(f"{TESTS}::test_under_estimate_below_300_the_plan_is_one_fit_at_the_small_sample_"
              "standard", truth="construction"),
        Silent(f"{TESTS}::test_under_estimate_from_300_the_plan_searches_after_the_library_"
               "defaults", truth="independent:scikit-learn"),
    ),
    enforced_by=("turbotab.core.models.tuning:standard_for",
                 "turbotab.core.models.tuning:searched_size",
                 "turbotab.core.stages.modeling:tuning_plans"),
    fallback="tuning.small_sample_standard_classes",
    author=AUTHOR,
    origin="engine",
))

SMALL_SAMPLE_STANDARD_CLASSES = register(Claim(
    id="tuning.small_sample_standard_classes",
    kind="rule",
    card="For an outcome of several classes, ordered or not, below an effective size of "
         "{small_n} boosted trees and XGBoost are not tuned either but fit once at the "
         "small-sample standard: a judgment, since the simulation that chose it had a yes/no "
         "outcome and a numeric one, never several classes or an order.",
    methods="For a multiclass or ordinal outcome, under either purpose, the tuning plan followed "
            "the same rule, a judgment carried from the yes/no and numeric outcomes its "
            "simulation covered: below an effective size of {small_n}, counted as the rarest "
            "class's units, it drew no candidate settings, and boosted trees and XGBoost were "
            "fit once at the small-sample standard; from {small_n} their settings were searched, "
            "the defaults among the candidates.",
    goal=("prediction", "inference"),
    regime=Regime(
        "boosted trees or XGBoost under either purpose on an outcome of several classes or "
        "several ordered ones, at any effective size (the rarest class's units), since both "
        "sentences are conditioned on the threshold they state: the plan reads no task, and "
        "notes/SMALL_SAMPLE_STANDARD.md simulated neither",
        requires=(f"{HERE}:standard_from_the_study", f"{HERE}:classes_or_ordered",
                  f"{HERE}:under_either_purpose"),
        disjoint_from=("tuning.small_sample_standard", "tuning.small_sample_standard_estimate")),
    decision="set_tuning",
    options=("automatic", "lighter", "standard", "manual"),
    verdict="recommended",
    parts=(
        Part("value", grade="judgment", basis=Basis(
            reason="The threshold for a yes/no or numeric outcome, a convention (RECIPES §4.6, "
                   "ruled 2026-10-11), read here unchanged on the rarest class's units: nothing "
                   "locates one for several classes or an order.")),
        Part("direction", grade="judgment", basis=Basis(
            reason="Carried from a yes/no outcome: docs/turbotab-next/notes/SMALL_SAMPLE_"
                   "STANDARD.md chose the standard on yes/no cells and checked it on a numeric "
                   "outcome (its §2 and §4), and simulated neither several classes nor ordered "
                   "ones. That it also predicts about as well as tuning and stays calibrated "
                   "where the defaults do not, when the rarest of several classes is small, is "
                   "expected, not measured.")),
        Part("scope", grade="judgment", basis=Basis(
            reason="Every limit of the study, with no class or ordered outcome among its cells: "
                   "RECIPES §4.6 names multiclass among the shapes to check before release, and "
                   "ordered outcomes nowhere; under Estimate no effect was estimated either.")),
    ),
    # The threshold for a yes/no or numeric outcome, read from that claim: one number in one place.
    params={"small_n": Param(SMALL_SAMPLE_STANDARD.params["small_n"].value,
                             unit=SMALL_SAMPLE_STANDARD.params["small_n"].unit)},
    tests=(
        Fires(f"{TESTS}::test_for_a_class_or_ordered_outcome_below_300_the_plan_is_one_fit_at_"
              "the_standard", truth="construction"),
        Silent(f"{TESTS}::test_for_a_class_or_ordered_outcome_from_300_the_plan_searches_after_"
               "the_library_defaults", truth="independent:scikit-learn"),
    ),
    enforced_by=("turbotab.core.models.tuning:standard_for",
                 "turbotab.core.models.tuning:searched_size",
                 "turbotab.core.stages.modeling:tuning_plans"),
    fallback="tuning.small_sample_no_search",
    author=AUTHOR,
    origin="engine",
))

SMALL_SAMPLE_NO_SEARCH = register(Claim(
    id="tuning.small_sample_no_search",
    kind="rule",
    card="Below an effective size of {small_n}, a searched family with no small-sample standard of "
         "its own, as the random forest, is not tuned either: it is fit once at the standard "
         "settings it starts its search from at {small_n} and above. A judgment: settings chosen "
         "on so few units vary from sample to sample, and in the simulation behind the boosting "
         "families' standard, on a yes/no outcome, the forest's standard settings predicted "
         "about as well as tuning it.",
    methods="Below an effective size of {small_n} the tuning plan drew no candidate settings for "
            "a searched family with no small-sample standard of its own (the random forest), "
            "which was fit once at its standard settings, a judgment, since settings ranked on so "
            "few units vary from sample to sample; from {small_n} its settings were searched "
            "inside every training fold, the standard among the candidates.",
    goal=("prediction", "inference"),
    regime=Regime(
        "a searched family whose tuning declaration names no small-sample standard (the random "
        "forest), under either purpose and on any outcome, at any effective size, since both "
        "sentences are conditioned on the threshold they state: the plan draws no search below "
        "it for any searched family, and notes/SMALL_SAMPLE_STANDARD.md checked the forest's "
        "standard settings on a yes/no outcome under Predict alone",
        requires=(f"{HERE}:searched_without_a_small_sample_standard",
                  f"{HERE}:under_either_purpose"),
        disjoint_from=("tuning.small_sample_standard", "tuning.small_sample_standard_estimate",
                       "tuning.small_sample_standard_classes")),
    decision="set_tuning",
    options=("automatic", "lighter", "standard", "manual"),
    verdict="recommended",
    parts=(
        Part("value", grade="judgment", basis=Basis(
            reason="The threshold of tuning.small_sample_standard, read here unchanged: RECIPES "
                   "§4.6 calls 300 a convention, and nothing locates one for the forest.")),
        Part("direction", grade="judgment", basis=Basis(
            reason="RECIPES §4.6 opens: \"Below an effective size of 300, searched families use "
                   "their standard settings by default\", for the variance of settings chosen on "
                   "few rows, and its 2026-10-11 ruling, which gave boosted trees and XGBoost a "
                   "small-sample standard, says \"The forest keeps its standard.\" "
                   "notes/SMALL_SAMPLE_STANDARD.md §2, item 5: \"The forest needs nothing. Its "
                   "standard settings are within 0.009 nats of the best arm in every cell; nested "
                   "tuning gains at most 0.008 under the null and nothing with signal.\" Those "
                   "cells are a yes/no outcome on ten normal predictors, the forest's nested arm "
                   "run only up to 300 rows, and their constants were read with every cell in "
                   "view, so none is held out.")),
        Part("scope", grade="judgment", basis=Basis(
            reason="The plan reads neither the purpose nor the task, so the forest draws no search "
                   "below the threshold under Estimate too, and for a number, several classes or "
                   "ordered ones, which the study checked for the forest on no outcome but a "
                   "yes/no one: a judgment carried from its yes/no cells.")),
    ),
    # The threshold for the boosting families, read from that claim: one number in one place.
    params={"small_n": Param(SMALL_SAMPLE_STANDARD.params["small_n"].value,
                             unit=SMALL_SAMPLE_STANDARD.params["small_n"].unit)},
    tests=(
        Fires(f"{TESTS}::test_below_an_effective_size_of_300_a_searched_family_is_fit_once_at_its_"
              "standard", truth="construction"),
        Silent(f"{TESTS}::test_from_an_effective_size_of_300_the_search_runs_with_the_library_"
               "defaults_first", truth="construction"),
    ),
    enforced_by=("turbotab.core.models.tuning:standard_for",
                 "turbotab.core.models.tuning:searched_size",
                 "turbotab.core.models.tuning:make_plan"),
    author=FINISH_AUTHOR,
    origin="engine",
    ruled="RECIPES_AND_TUNING.md §4.6, ruled 2026-10-11: the forest keeps its standard",
))


# ── the shelf's rows ─────────────────────────────────────────────────────────────────────────

BOOSTED_TREES_SMALL_ROWS = register(Claim(
    id="boosted_trees.small_rows",
    kind="rule",
    card="Below {small_rows} rows the shelf rates boosted trees fair, and poor below "
         "{tiny_rows}: they overfit easily and their cross-validated score is noisy.",
    methods="On the shelf, boosted trees (and XGBoost, read through them) were rated fair below "
            "{small_rows} rows and poor below {tiny_rows}, a convention, counting the training "
            "rows under Predict and every analyzed row under Estimate or Describe.",
    goal=("prediction", "inference"),
    regime=Regime(
        "the shelf's pre-fit rating of boosted trees, and of XGBoost through it, under either "
        "purpose, read on the rows the shelf ranks for: the training rows under Predict, every "
        "analyzed row under Estimate or Describe",
        requires=(f"{HERE}:rated_before_any_fit",)),
    decision="models.families",
    options=("good", "fair", "poor"),
    verdict="rank_lower",
    parts=(
        Part("value", grade="judgment", basis=Basis(
            reason="A convention carried from draft 2's shelf scores: RECIPES §2.6 says no "
                   "validated rule predicts which family will perform best, and states boosted "
                   "trees' rows nowhere.")),
        Part("direction", grade="judgment", basis=Basis(
            reason="Fewer rows let a boosted ensemble fit noise and make a cross-validated score "
                   "noisier, a general expectation the shelf orders by, not a measured "
                   "threshold.")),
    ),
    params={"small_rows": Param(500, unit="rows the shelf ranks for",
                                constant="turbotab.core.models.boosted_trees:SMALL_N"),
            "tiny_rows": Param(200, unit="rows the shelf ranks for",
                               constant="turbotab.core.models.boosted_trees:TINY_N")},
    tests=(
        Fires(f"{TESTS}::test_below_500_rows_the_shelf_rates_boosted_trees_fair_and_below_200_"
              "poor", truth="construction"),
        Silent(f"{TESTS}::test_from_500_rows_boosted_trees_carry_no_small_table_concern",
               truth="construction"),
    ),
    enforced_by=("turbotab.core.models.boosted_trees:BoostedTrees.assess",),
    author=AUTHOR,
    origin="engine",
))

FOREST_SMALL_ROWS = register(Claim(
    id="forest.small_rows",
    kind="rule",
    card="Below {small_rows} rows the shelf rates a random forest fair: its cross-validated "
         "score is noisy.",
    methods="On the shelf, the random forest was rated fair below {small_rows} rows, a "
            "convention, counting the training rows under Predict and every analyzed row under "
            "Estimate or Describe.",
    goal=("prediction", "inference"),
    regime=Regime(
        "the shelf's pre-fit rating of the random forest under either purpose, read on the rows "
        "the shelf ranks for: the training rows under Predict, every analyzed row under Estimate "
        "or Describe",
        requires=(f"{HERE}:rated_before_any_fit",)),
    decision="models.families",
    options=("good", "fair"),
    verdict="rank_lower",
    parts=(
        Part("value", grade="judgment", basis=Basis(
            reason="RECIPES §2.6's table: 2.0 from 2,000 training rows, 1.5 from 500, fair at "
                   "1.0 below 500, a convention, since no validated rule predicts which family "
                   "will perform best.")),
        Part("direction", grade="judgment", basis=Basis(
            reason="Fewer rows make a cross-validated score noisier, a general expectation the "
                   "shelf orders by, not a measured threshold; the forest's rating has no poor "
                   "step below it, as boosted trees' does.")),
    ),
    params={"small_rows": Param(500, unit="rows the shelf ranks for",
                                constant="turbotab.core.models.forest:SMALL_N")},
    tests=(
        Fires(f"{TESTS}::test_below_500_rows_the_shelf_rates_a_random_forest_fair",
              truth="construction"),
        Silent(f"{TESTS}::test_from_500_rows_a_random_forest_carries_no_small_table_concern",
               truth="construction"),
    ),
    enforced_by=("turbotab.core.models.forest:RandomForest.assess",),
    author=AUTHOR,
    origin="engine",
))


# ── the baseline verdict's floor (A-029) ─────────────────────────────────────────────────────

BASELINE_FLOOR = register(Claim(
    id="baseline.min_gain",
    kind="rule",
    card="A model is called better than predicting without its predictors only when its gain's "
         "interval lies above zero and the gain exceeds {min_gain} on a score where higher is "
         "better, or {min_gain:.0%} of the baseline's own loss on a loss such as log loss.",
    methods="A family was called better than its no-predictor baseline only when the gain's "
            "corrected interval lay above zero and the gain exceeded a floor, a convention: "
            "{min_gain} on a score better when higher, and on a loss {min_gain:.0%} of the "
            "baseline's own cross-validated loss (a McFadden pseudo R squared, a ranked "
            "probability skill score or a scaled Brier score of {min_gain}).",
    goal=("prediction", "inference"),
    regime=Regime(
        "a family's cross-validated primary against its no-predictor baseline under either "
        "purpose, paired over the same folds: R² for a numeric outcome, log loss for a yes/no or "
        "class outcome, the ranked probability score for an ordered one, the Brier score at the "
        "horizon for time to event, Harrell's C for a time to event with delayed entry",
        requires=(f"{HERE}:verdict_metric",)),
    decision="baseline.verdict",
    options=("better", "no_better", "worse"),
    verdict="noted",
    parts=(
        Part("value", grade="judgment", basis=Basis(
            reason="A convention, not a cited rule (baseline.py): no source sets one smallest "
                   "gain that matters for every outcome.")),
        Part("scale", grade="judgment", basis=Basis(
            reason="R² is already the MSE's gain as a share of the baseline's, and a McFadden R², "
                   "a ranked probability skill score and a scaled Brier score are the same share "
                   "of log loss, the RPS and the Brier score at the horizon, so one share reads "
                   "alike across them. A fixed 0.01 on a loss's own scale called a rare "
                   "outcome's true risks, gaining 6% to 12% of the baseline's loss, better by too "
                   "little to matter (claims backtest A-029; the silent test's construction).")),
    ),
    params={"min_gain": Param(0.01, unit="on a score better when higher, its own scale; on a "
                                         "loss, a share of the baseline's loss",
                              constant="turbotab.core.models.baseline:MIN_GAIN")},
    tests=(
        Fires(f"{TESTS}::test_a_real_gain_under_a_hundredth_of_the_baseline_loss_is_too_little_"
              "to_matter", truth="construction"),
        Fires(f"{TESTS}::test_a_gain_under_a_hundredth_on_a_score_better_when_higher_is_too_"
              "little_to_matter", truth="construction"),
        Silent(f"{TESTS}::test_a_rare_outcomes_true_risks_beat_the_baseline_by_enough_to_matter",
               truth="construction"),
        Silent(f"{TESTS}::test_a_gain_of_three_hundredths_on_a_score_better_when_higher_is_"
               "better", truth="construction"),
    ),
    enforced_by=("turbotab.core.models.baseline:min_gain",
                 "turbotab.core.models.baseline:versus_baseline"),
    author=AUTHOR,
    origin="audit",
))


# ── materiality (A-124, C-170) ───────────────────────────────────────────────────────────────

ROWS_SMD_FLOOR = register(Claim(
    id="materiality.rows_smd_floor",
    kind="rule",
    card="The share of rows an energy screen removes, times how far they differ on what the model "
         "adjusts for, is a screening convention: it can say the screen could bias the estimate "
         "or that you should act on it, never that it doesn't change your numbers here.",
    methods="Rows changed times the leavers' largest standardized mean difference on the "
            "adjustment set served as an outcome-blind screening convention floored at could "
            "bias, not as a predictor of how far the estimate moves: what moves a correctly "
            "specified coefficient is what the leavers report of the exposure, not their "
            "imbalance on what is adjusted for.",
    goal=("inference",),
    regime=Regime(
        "the energy screen's noticing (diet-implausible-reporters) under Estimate, before the "
        "lock: the predicted movement of keeping every row, read as rows changed times the "
        "leavers' largest SMD on the adjustment set, whatever its calibration cases",
        requires=(f"{HERE}:rows_times_smd", f"{HERE}:under_inference")),
    decision="materiality.diet_implausible_reporters",
    options=("no_change", "could_bias", "act_on_it"),
    verdict="noted",
    parts=(
        Part("direction", grade="certified", basis=Basis(
            tests=("turbotab/core/tests/test_materiality.py::test_a_narrow_screen_whose_leavers_"
                   "misreport_what_you_study_is_never_called_below_noise",),
            reason="Known truth against the proxy: four over-reporters out of 10,000 put rows "
                   "times SMD below the committed cut-off, yet keeping them moves sugar's "
                   "least-squares coefficient by more than half its t-interval half-width, an "
                   "assertion that would fail if the product tracked the movement. The verifier "
                   "rebuilt the construction (statsmodels OLS, pooled-SD SMD by hand) on seeds 0 "
                   "to 5: the proxy stayed below the cut-off and the coefficient moved more than "
                   "half a half-width in each, so it is not one seed's artifact (claims backtest "
                   "A-124)."),
            graded_by="R0b-models verifier, 2026-10-16"),
        Part("scope", grade="judgment", basis=Basis(
            reason="The floor is the convention's own (its floored reason in "
                   "materiality_calibration.json): a proxy without it may still be calibrated, "
                   "and band 2 stays band 2.")),
    ),
    tests=(
        Fires("turbotab/core/tests/test_materiality.py::"
              "test_a_narrow_screen_whose_leavers_misreport_what_you_study_is_never_called_below_"
              "noise", truth="independent:numpy"),
        Fires(f"{TESTS}::test_rows_times_smd_is_never_calibrated_and_never_below_could_bias",
              truth="construction"),
        Silent(f"{TESTS}::test_rows_times_smd_past_its_band_two_convention_is_act_on_it",
               truth="construction"),
        Silent(f"{TESTS}::test_a_proxy_whose_convention_is_not_floored_may_say_below_noise",
               truth="construction"),
    ),
    enforced_by=("turbotab.core.materiality:band_of", "turbotab.core.materiality:calibrate",
                 "turbotab.core.materiality:recommend"),
    fallback="materiality.rows_smd_floor_prediction",
    author=AUTHOR,
    origin="audit",
))

ROWS_SMD_FLOOR_PREDICTION = register(Claim(
    id="materiality.rows_smd_floor_prediction",
    kind="rule",
    card="Under Predict too, the share of rows an energy screen removes, times how far they differ "
         "on what the model adjusts for (its predictors other than one column marked as what you "
         "study), is a screening convention: it can say the screen could bias the score or that "
         "you should act on it, never that it doesn't change your numbers here.",
    methods="Under Predict, rows changed times the leavers' largest standardized mean difference "
            "on what the model adjusts for (the predictors by role, less the outcome, those left "
            "out and one exposure: the estimand's where one is declared, else the first column "
            "with the exposure role) served as the same outcome-blind screening convention, "
            "floored at could bias: a judgment carried from Estimate, where a constructed case "
            "showed the product does not track how far a coefficient moves; whether it tracks "
            "how far a cross-validated score moves was not measured.",
    goal=("prediction",),
    regime=Regime(
        "the energy screen's noticing (diet-implausible-reporters) under Predict, before the "
        "lock: the predicted movement of keeping every row, read as rows changed times the "
        "leavers' largest SMD on what the model adjusts for, as under Estimate (its predictors "
        "by role less the outcome, those left out and one column of what you study, the declared "
        "comparison's or else the first column marked as what you study), whatever its "
        "calibration cases",
        requires=(f"{HERE}:rows_times_smd", f"{HERE}:under_prediction"),
        disjoint_from=("materiality.rows_smd_floor",)),
    decision="materiality.diet_implausible_reporters",
    options=("no_change", "could_bias", "act_on_it"),
    verdict="noted",
    parts=(
        Part("direction", grade="judgment", basis=Basis(
            reason="Carried from Estimate: the certified construction there (four over-reporters "
                   "out of 10,000) moves a least-squares coefficient, not a cross-validated "
                   "score. That the product can sit below its cut-off while the score moves is "
                   "expected from it, with no case of its own.")),
        Part("scope", grade="judgment", basis=Basis(
            reason="The floor is the convention's own (its floored reason in "
                   "materiality_calibration.json), and band_of reads no purpose: the same floor "
                   "acts on the score as on the estimate.")),
    ),
    tests=(
        Fires(f"{TESTS}::test_under_predict_rows_times_smd_is_never_below_could_bias_the_score",
              truth="construction"),
        Silent(f"{TESTS}::test_under_predict_the_floor_never_caps_and_is_its_conventions_own",
               truth="construction"),
    ),
    enforced_by=("turbotab.core.materiality:band_of", "turbotab.core.materiality:recommend"),
    author=AUTHOR,
    origin="engine",
))

COLLINEAR_UNDER_PREDICTION = register(Claim(
    id="materiality.collinear_under_prediction",
    kind="rule",
    card="Under Predict, both of a nearly collinear pair are kept and keeping them records no "
         "selection, a convention set for you to confirm: such a pair does not harm prediction "
         "on rows like these. Reading their importance together is the line's advice to you.",
    methods="Under Predict, both predictors of a nearly collinear pair were kept, a convention the "
            "analyst confirmed rather than a decision: keeping both recorded no predictor "
            "selection, taking back the line's own elastic-net filter where it was recorded on a "
            "table with fewer candidate predictors than rows (on a wide table the width's lever "
            "offers the same filter, so it stays), and leaving any other selection as it stood, "
            "since collinearity does not harm "
            "prediction on rows like the training rows; the elastic-net selection filter, a step "
            "before every family, was offered but not recommended. The line advised reading the "
            "pair's importance together; no importance was grouped for the pair.",
    goal=("prediction",),
    regime=Regime(
        "Predict, a pair of numeric predictors nearly collinear on the training rows (explore's "
        "pair scan), scored on new rows like the training rows",
        requires=(f"{HERE}:under_prediction",)),
    decision="explore.collinear",
    options=("keep_both", "elastic_net", "by_hand"),
    verdict="recommended",
    parts=(
        Part("direction", grade="judgment", basis=Basis(
            reason="Collinearity inflates the variance of the pair's separate coefficients, not "
                   "of predictions on rows like the training rows; the fires test's construction "
                   "(a pair at r = 0.95, ten draws) shows the elastic-net filter keeps both, "
                   "leaves least squares where it was and harms the forest. New rows where the "
                   "pair comes apart are outside it.")),
        Part("tier", grade="judgment", basis=Basis(
            reason="BEATS_PREDICT M4 sets keeping the pair as a convention for the person to "
                   "confirm; no materiality instrument grades it under Predict.")),
    ),
    tests=(
        Fires("turbotab/core/tests/test_collinear_adjustment.py::"
              "test_under_predict_a_collinear_pair_is_kept_and_the_selection_filter_is_not_"
              "recommended", truth="independent:scikit-learn"),
        Fires(f"{TESTS}::test_under_predict_the_collinear_line_is_a_confirm_whatever_the_pair_"
              "holds", truth="construction"),
        Silent(f"{TESTS}::test_under_estimate_a_pair_holding_what_you_study_is_decided_at_the_"
               "adjustment_question", truth="construction"),
        Silent("turbotab/core/tests/test_collinear_adjustment.py::"
               "test_the_collinear_line_is_decide_only_where_what_you_study_is_in_it",
               truth="construction"),
        Fires(f"{TESTS}::test_the_collinear_methods_sentence_says_where_keeping_both_takes_back_"
              "the_filter", truth="construction"),
    ),
    enforced_by=("turbotab.core.materiality:collinear_label",
                 "turbotab.core.stages.explore:_collinear_lever",
                 "turbotab.core.stages.explore:_keep_both"),
    author=AUTHOR,
    origin="audit",
    ruled="calm/BEATS_PREDICT.md M4, its fourth line (near-collinear predictors, all kept; the "
          "grouped importance it rules for Results is not built)",
))

# ── R1-F's first batch (R1_PLAN.md R1-F; claims backtest FIXES.md) ───────────────────────────
#
# The models area's sentences an analyst reads in teaching, on a card or in a fit's reading, each
# fixed test first: which validation leads, and for which families (D-018), what the pairwise
# interval says (D-021), the event's share and whether anything was resampled (A-106), and the
# seal drawn by row (C-110). Each part is the author's judgment: no grader other than the author
# has checked one yet (CLAIMS_REGISTRY.md §2.4). The seal's part on the folds is retired: repair
# round 1 found it false under internal-external validation.

RF1_AUTHOR = "RF-1 builder"
RF1_TESTS = "turbotab/core/tests/test_rf1_sentences.py"

VALIDATION_RESAMPLING_FIRST = register(Claim(
    id="validation.resampling_first",
    kind="teaching",
    card="Under prediction, for regression-type models below about {resample_below:,} units, "
         "where the folds need not follow time, bootstrap optimism correction is the recommended "
         "default and repeated cross-validation is acceptable; boosted trees, XGBoost and random "
         "forests keep their cross-validated score, since the bootstrap overstates them. A single "
         "train and test split is the weakest option.",
    goal=("prediction",),
    regime=Regime("Predict, folds that need not follow time, the units validated below the "
                  "number at which models/validation.py's plan ranks resampling first",
                  requires=(f"{HERE}:under_prediction",)),
    decision="set_validation",
    params={"resample_below": Param(20_000, "units")},
    parts=(
        Part("threshold", grade="judgment", basis=Basis(
            reason="Harrell (2017, Split-sample model validation) finds data splitting unstable "
                   "below about twenty thousand subjects, and the plan ranks resampling first "
                   "below it (models/validation.py); the clinical pack's order, bootstrap then "
                   "repeated cross-validation, is a convention (CLINICAL_SURVEY_PACK §A5.5). The "
                   "threshold holds only where the folds need not follow time: where the answers "
                   "ask for time order the plan forward-chains the folds by whole unit and ranks "
                   "the bootstrap last, whatever the units, since resamples would ignore time "
                   "(the RF-1 verifier's P4).")),
        Part("trees", grade="judgment", basis=Basis(
            reason="Coley et al. 2023 found the bootstrap's optimism correction did not validate "
                   "a random forest, and the repair round replicated boosted trees' overstated "
                   "score (models/validation.py), so each near-interpolating family declares "
                   "bootstrap_optimism False and keeps its cross-validated score.")),
    ),
    tests=(
        Fires(f"{RF1_TESTS}::test_resampling_leads_under_prediction_below_harrells_twenty_"
              "thousand", truth="independent:Harrell 2017, Split-sample model validation"),
        Silent(f"{RF1_TESTS}::test_the_teaching_names_every_family_the_bootstrap_does_not_"
               "correct", truth="construction"),
        Silent(f"{RF1_TESTS}::test_with_time_ordered_folds_the_bootstrap_comes_last_and_the_"
               "teaching_says_where_it_leads", truth="construction"),
    ),
    enforced_by=("turbotab.core.models.validation:validation_plan",
                 "turbotab.core.teaching.content:SINGLE_SPLIT"),
    author=RF1_AUTHOR,
    origin="audit",
))

VALIDATION_PAIRED_INTERVAL = register(Claim(
    id="validation.paired_interval",
    kind="teaching",
    card="Under prediction, each cross-validated score carries its standard error, and each pair "
         "of families an interval for their difference by the corrected repeated k-fold t "
         "(Nadeau & Bengio 2003; Bouckaert & Frank 2004), computed on these rows. The interval "
         "is approximate and usually conservative, and with several pairs each is descriptive: "
         "one that excludes zero says a family is ahead on these rows; one that includes zero "
         "says these rows cannot tell the two apart, not that they are equal.",
    goal=("prediction",),
    regime=Regime("Predict, two or more families compared on the repeated k-fold substrate",
                  requires=(f"{HERE}:under_prediction",)),
    decision="models.comparisons",
    parts=(
        Part("method", grade="judgment", basis=Basis(
            cites=("nadeau2003variance", "bouckaert2004"),
            reason="The comparison is the corrected resampled t over r repeats of K folds "
                   "(models/validation.py corrected_t), which the first test writes out by "
                   "hand.")),
        Part("conservative", grade="judgment", basis=Basis(
            reason="On a null built here (two exchangeable least-squares models, 10 × 5 folds) "
                   "the interval calls one ahead in at most the nominal share of tables, and the "
                   "claims backtest's adjudicator found 0 to 1% at n = 30 to 200 (D-021); "
                   "unstable learners at small n are outside both.")),
        Part("verdict", grade="judgment", basis=Basis(
            reason="With several pairs and no correction for their number each interval is "
                   "descriptive, and one that includes zero says the rows cannot tell the two "
                   "apart (family_differences: 'not distinguishable on these rows'), never that "
                   "they are equal.")),
    ),
    tests=(
        Fires(f"{RF1_TESTS}::test_the_pairwise_interval_is_the_corrected_repeated_k_fold_t_"
              "written_out", truth="independent:scipy"),
        Fires(f"{RF1_TESTS}::test_under_a_true_null_the_interval_rarely_calls_a_family_ahead",
              truth="construction"),
        Fires(f"{RF1_TESTS}::test_the_teaching_says_an_interval_with_zero_cannot_tell_two_"
              "families_apart", truth="construction"),
    ),
    enforced_by=("turbotab.core.models.validation:corrected_t",
                 "turbotab.core.models.validation:family_differences",
                 "turbotab.core.teaching.content:SPLIT"),
    author=RF1_AUTHOR,
    origin="audit",
))

MODELING_NO_RESAMPLING = register(Claim(
    id="modeling.no_resampling",
    kind="rule",
    card="No resampling (SMOTE, over- or undersampling) or class weighting was applied: customary "
         "in machine-learning papers, resampling overestimates the rarer class's probability "
         "without raising the AUC (van den Goorbergh et al. 2022); a decision moves the "
         "threshold instead.",
    regime=Regime("a yes/no outcome whose fit applies no imbalance correction (the levers' "
                  "imbalance answer none, or no levers answered)",
                  requires=(f"{HERE}:yes_no_left_uncorrected",)),
    decision="set_levers.imbalance",
    options=("none", "weights", "undersample", "oversample"),
    verdict="recommended",
    parts=(
        Part("default", grade="judgment", basis=Basis(
            cites=("vandengoorbergh2022imbalance",),
            reason="Van den Goorbergh et al. 2022 found random under- and oversampling and SMOTE "
                   "overestimate the minority class's probability without a higher AUC (quoted "
                   "in stages/modeling.py); both referees and the audit agreed (claims backtest "
                   "A-106).")),
        Part("scope", grade="judgment", basis=Basis(
            reason="The sentence says nothing was resampled only where the fit applied no "
                   "correction: the levers' correction wraps the model whenever it is answered "
                   "(models.pipeline, methods.levers.wrap_model), and the sentence then says the "
                   "imbalance was corrected.")),
    ),
    tests=(
        Fires(f"{RF1_TESTS}::test_a_fit_with_no_imbalance_correction_says_none_was_applied",
              truth="construction"),
        Silent(f"{RF1_TESTS}::test_a_fit_with_class_weights_never_says_none_was_applied",
               truth="construction"),
    ),
    enforced_by=("turbotab.core.stages.modeling:imbalance_sentence",),
    author=RF1_AUTHOR,
    origin="audit",
))

SEAL_BY_ROW = register(Claim(
    id="seal.by_row_exploratory",
    kind="rule",
    card="treat held-out scores as exploratory",
    regime=Regime("a seal that could not keep a unit's rows together (abandoned or undetermined), "
                  "whatever the validation: the seal is drawn before it is chosen",
                  requires=(f"{HERE}:drawn_by_row",)),
    decision="set_split.basis",
    verdict="noted",
    parts=(
        Part("folds", grade=RETIRED, graded_by="RF-1 builder, repair round 1, 2026-10-18",
             regraded="The seal's sentence no longer says the folds were drawn by row: it is "
                      "drawn before the validation is chosen, and under internal-external "
                      "validation each fold is one level of the grouping (rows.draw_split's "
                      "clusters), so the RF-1 verifier's P3 found it false there. The split's "
                      "note, written after the validation is chosen, holds the folds' sentence "
                      "(models.validation.within_unit_sentence, MS6), and was not right under "
                      "every validation either: it said the folds were drawn by row under "
                      "internal-external validation too (the verifier's round-2 P3), and then "
                      "that every score was within-unit where one unit of forty spanned two folds "
                      "(the RF-1b verifier's P3), and under k-fold that every score was "
                      "within-unit where no unit, or only some, had rows in two folds (the RF-1b "
                      "verifier's round-2 V2). It now reads the drawn folds, each draw of them, "
                      "under every validation wherever a column names the unit: it says nothing of "
                      "units where no unit's rows fall in two folds, says every score is "
                      "within-unit only where every unit's training rows do in every draw, and "
                      "otherwise counts them (their range over the draws of repeated k-fold) and "
                      "reads only their rows' scores as within-unit; where no column can tell, it "
                      "says a fold may split a unit.",
             basis=Basis(
                 reason="The folds are drawn with the seal's groups (seal.split_args, "
                        "rows.draw_split), so where the held-out rows are drawn by row the folds "
                        "are too, unless internal-external validation makes each fold a level of "
                        "the grouping.")),
        Part("direction", grade="judgment", basis=Basis(
            cites=("saeb2017",),
            reason="A unit on both sides of the held-out draw scores high: record-wise splits "
                   "overestimate performance on new units (Saeb et al. 2017), so the claims "
                   "backtest's 'conservative' (C-110) was backwards; the scores are labeled "
                   "exploratory rather than corrected.")),
    ),
    tests=(
        Fires(f"{RF1_TESTS}::test_a_seal_drawn_by_row_over_a_repeating_identifier_labels_its_held_"
              "out_scores_exploratory", truth="construction"),
        Silent(f"{RF1_TESTS}::test_a_grouped_seal_says_nothing_of_exploratory_scores",
               truth="construction"),
        Silent(f"{RF1_TESTS}::test_under_internal_external_validation_the_folds_are_the_sites_and_"
               "the_seal_says_nothing_of_them", truth="construction"),
    ),
    enforced_by=("turbotab.core.seal:decide_basis",),
    author=RF1_AUTHOR,
    origin="audit",
))

# ── SWARM_PLAN §10 package 4: the temporal seal by cut-point ─────────────────────────────────
# The seal's sentence leads with the cut-point's column, value and counts of the analyzed rows and
# then says the claim's card (seal.cut_holdout); the split's methods sentence names the column and
# value and then says its methods register (voice._set_split). A seal that keeps units whole says
# seal.cut_point_units instead, so a table of one row per unit is never told of units (the
# verifier's P14). The author grades no part above judgment.

P4_AUTHOR = "swarm package 4 builder"
P4_REPAIR = "swarm package 4 builder, repair round 1, 2026-10-22"
P4_CUT_TESTS = "turbotab/core/tests/test_cut_point_seal.py"

BOUNDARY = Part("boundary", grade="judgment", basis=Basis(
    reason="At or after: a row at the cut-point itself is held out, so a later cycle's first "
           "value (2017-March 2020's, SWARM_PLAN §5) is all held out; the fires test puts rows at "
           "the value and a hair before it. Every other analyzed row trains, one with no readable "
           "time among them (seal.cut_point_undated). A row an exclusion leaves out is sealed "
           "with the rest where it falls at or after the cut-point, so an exclusion relaxed later "
           "brings it back held out, but it never trains; so the seal counts the analyzed rows, "
           "as the split's own counts do (the package 4 verifier's P1)."))

SEAL_CUT_POINT = register(Claim(
    id="seal.cut_point",
    kind="rule",
    card="A cut-point holds out every row at or after it, and every other analyzed row trains.",
    methods="held out every row at or after it for one final score",
    regime=Regime("a split that holds rows out by a cut-point on a column that reads as numbers "
                  "or dates, in place of a share, whatever the folds' order, where the seal holds "
                  "out rows one by one (one row per unit, or rows it does not group)",
                  requires=(f"{HERE}:held_out_by_a_cut_point", f"{HERE}:held_out_row_by_row"),
                  disjoint_from=("seal.cut_point_units",)),
    decision="set_split.cut_point",
    verdict="noted",
    parts=(
        BOUNDARY,
        Part("whole_units", grade=RETIRED, graded_by=P4_REPAIR,
             regraded="The sentence no longer says a unit's earlier rows are held out: on a table "
                      "of one row per unit it named units that do not exist (the package 4 "
                      "verifier's P14). Where the seal keeps units whole, seal.cut_point_units "
                      "says it.",
             basis=Basis(
                 reason="A unit with rows on both sides is held out whole, by its last "
                        "observation, as the chronological draw ranks units (GUIDED-143: a unit "
                        "split across the boundary trades one leak for another), so its earlier "
                        "rows are held out too and the seal counts them (audit IN-24).")),
        Part("folds_free", grade="judgment", basis=Basis(
            reason="The training folds' order is the split's own answer under a cut-point "
                   "(SWARM_PLAN §10's watch note on package 3), so the same held-out rows can be "
                   "scored with random repeated folds or time-ordered ones; random folds there "
                   "are the analyst's call, which the methods sentence states.")),
    ),
    tests=(
        Fires(f"{P4_CUT_TESTS}::test_every_row_at_or_after_the_cut_point_is_held_out_and_every_"
              "earlier_row_trains", truth="construction"),
        Fires(f"{P4_CUT_TESTS}::test_with_rows_excluded_the_seal_counts_the_analyzed_rows_it_holds_"
              "out_and_trains", truth="independent:pandas"),
        Silent(f"{P4_CUT_TESTS}::test_the_draw_by_share_still_orders_its_folds_by_time",
               truth="construction"),
        Fires(f"{P4_CUT_TESTS}::test_the_same_held_out_rows_take_random_repeated_folds_or_time_"
              "ordered_ones", truth="construction"),
        Fires(f"{P4_CUT_TESTS}::test_the_split_sentence_says_what_the_cut_point_held_out",
              truth="construction"),
        Silent(f"{P4_CUT_TESTS}::test_the_split_sentence_names_units_only_where_the_seal_keeps_"
               "units_whole", truth="construction"),
        Fires(f"{P4_CUT_TESTS}::test_the_seal_sentence_agrees_with_its_counts_row_by_row",
              truth="construction"),
    ),
    enforced_by=("turbotab.core.seal:cut_holdout", "turbotab.core.seal:_cut",
                 "turbotab.core.voice:_set_split"),
    author=P4_AUTHOR,
    origin="engine",
))

SEAL_CUT_POINT_UNITS = register(Claim(
    id="seal.cut_point_units",
    kind="rule",
    card="A cut-point holds out every row at or after it, with the earlier rows of any unit it "
         "holds out, and every other analyzed row trains.",
    methods="held out every row at or after it for one final score, with the earlier rows of any "
            "unit it held out",
    regime=Regime("a split that holds rows out by a cut-point on a column that reads as numbers "
                  "or dates, in place of a share, where the seal keeps each unit's rows together "
                  "and some unit has more than one row",
                  requires=(f"{HERE}:held_out_by_a_cut_point", f"{HERE}:held_out_in_whole_units"),
                  disjoint_from=("seal.cut_point",)),
    decision="set_split.cut_point",
    verdict="noted",
    parts=(
        BOUNDARY,
        Part("whole_units", grade="judgment", basis=Basis(
            reason="A unit with rows on both sides is held out whole, by its last observation, as "
                   "the chronological draw ranks units (GUIDED-143: a unit split across the "
                   "boundary trades one leak for another), so its earlier rows are held out too "
                   "and the seal counts them (audit IN-24).")),
    ),
    tests=(
        Fires(f"{P4_CUT_TESTS}::test_a_unit_seen_on_both_sides_is_held_out_whole_and_its_earlier_"
              "rows_are_counted", truth="independent:pandas"),
        Fires(f"{P4_CUT_TESTS}::test_a_grouped_preview_says_whole_units_are_held_out_with_their_"
              "earlier_rows", truth="independent:pandas"),
        Fires(f"{P4_CUT_TESTS}::test_the_split_sentence_names_units_only_where_the_seal_keeps_"
              "units_whole", truth="construction"),
        Silent(f"{P4_CUT_TESTS}::test_every_row_at_or_after_the_cut_point_is_held_out_and_every_"
               "earlier_row_trains", truth="construction"),
        Fires(f"{P4_CUT_TESTS}::test_the_seal_sentence_agrees_with_its_counts_in_whole_units",
              truth="independent:pandas"),
    ),
    enforced_by=("turbotab.core.seal:cut_holdout", "turbotab.core.voice:_set_split"),
    author=P4_AUTHOR,
    origin="engine",
))

SEAL_CUT_POINT_UNDATED = register(Claim(
    id="seal.cut_point_undated",
    kind="rule",
    card="rows with no readable time would train, though any of them may fall at or after the "
         "cut-point, so a cut-point is refused where more than {max_undated:.0%} of the rows with "
         "the outcome recorded have none",
    regime=Regime("a split that holds rows out by a cut-point on a column some of whose rows do "
                  "not read as a time; the chronological draw by share reads the same ceiling "
                  "(seal.MAX_UNDATED_SHARE), which this claim does not cover",
                  requires=(f"{HERE}:held_out_by_a_cut_point",)),
    decision="set_split.cut_point_undated",
    verdict="refused",
    params={"max_undated": Param(0.1, unit="share of the rows with the outcome recorded",
                                 constant="turbotab.core.seal:MAX_UNDATED_SHARE")},
    parts=(
        Part("undated_train", grade="judgment", basis=Basis(
            reason="A row with no readable time cannot be placed before or after the cut-point, "
                   "so it is never held out and trains, as under the chronological draw by share; "
                   "the seal counts such rows.")),
        Part("ceiling", grade="judgment", basis=Basis(
            reason="Above a tenth, rows that train though they may fall at or after the "
                   "cut-point are too many for the held-out rows to stand for the later period; "
                   "the tenth is the chronological draw's own ceiling (MAX_UNDATED_SHARE), kept "
                   "so both draws by time refuse alike. It is our call, not a published "
                   "threshold.")),
    ),
    tests=(
        Fires(f"{P4_CUT_TESTS}::test_more_than_a_tenth_of_the_rows_undated_refuses_the_cut_point",
              truth="construction"),
        Silent(f"{P4_CUT_TESTS}::test_a_tenth_of_the_rows_undated_is_drawn_and_the_undated_rows_"
               "train", truth="construction"),
        Fires(f"{P4_CUT_TESTS}::test_the_undated_share_is_counted_on_the_rows_with_the_outcome_"
              "recorded", truth="independent:pandas"),
        Fires(f"{P4_CUT_TESTS}::test_one_undated_row_of_too_few_is_said_in_the_singular",
              truth="construction"),
    ),
    enforced_by=("turbotab.core.seal:cut_problem", "turbotab.core.seal:cut_holdout"),
    author=P4_AUTHOR,
    origin="engine",
))

SEAL_CUT_POINT_ANALYZED = register(Claim(
    id="seal.cut_point_analyzed",
    kind="rule",
    card="A cut-point is drawn only where some analyzed rows are held out and some before it "
         "train.",
    regime=Regime("a split that holds rows out by a cut-point that holds rows out and leaves some "
                  "to train among the rows with the outcome recorded, drawn where the cohort's "
                  "rows (the analyzed ones) are known: the split stage and its preview",
                  requires=(f"{HERE}:held_out_by_a_cut_point",)),
    decision="set_split.cut_point_analyzed",
    verdict="refused",
    parts=(
        Part("refused", grade="judgment", basis=Basis(
            reason="The cut-point is drawn over the rows with the outcome recorded, and the "
                   "answers that choose the analyzed rows (an exclusion, the missing-values "
                   "answer under complete cases: NHANES's later cycle can lack a predictor "
                   "outright) can leave none of them on one side. Drawn so, the split would hold "
                   "out no analyzed row, or train on none, while the opening, the Router and the "
                   "compared-families guard read a cut-point as rows held out "
                   "(decisions.holds_out); so the draw is refused, naming the answers that leave "
                   "them out, and never drawn empty (the package 4 verifier's V2). The answers "
                   "come after the split in the interview, so this is checked at the draw, not "
                   "at the split's answer.")),
    ),
    tests=(
        Fires(f"{P4_CUT_TESTS}::test_an_exclusion_that_leaves_no_analyzed_row_at_or_after_the_"
              "cut_point_refuses_the_draw", truth="independent:pandas"),
        Fires(f"{P4_CUT_TESTS}::test_an_exclusion_that_leaves_no_analyzed_row_before_the_cut_"
              "point_refuses_the_draw", truth="independent:pandas"),
        Fires(f"{P4_CUT_TESTS}::test_complete_cases_that_leave_no_analyzed_row_held_out_refuse_"
              "the_draw", truth="construction"),
        Silent(f"{P4_CUT_TESTS}::test_an_exclusion_that_thins_a_side_of_the_cut_point_is_drawn",
               truth="independent:pandas"),
        Fires(f"{P4_CUT_TESTS}::test_the_draws_refusal_agrees_with_one_row_left_out",
              truth="construction"),
    ),
    enforced_by=("turbotab.core.seal:_cut_leaves_the_analysis", "turbotab.core.seal:_cut"),
    author=P4_AUTHOR,
    origin="engine",
))

# ── SWARM_PLAN §10 package 6a: the NHANES glucose journey's claims (R1_PLAN.md R1-B) ────────
#
# The walkthrough's journey (NHANES fasting glucose under Predict) captured by
# research/registry/sentences.py, each sentence it shows indexed to its claim or triaged
# (research/registry/journeys/). These are the refusals and defaults it shows that render where the
# project's claims render log reads them: in a stage's artifact or a refusal of an answer. Each
# renders its words as the code wrote them before, through claims.card, inside the stage or the
# answer that shows it, so a swarm record's ``claims`` lists it. The author grades no part above
# judgment (CLAIMS_REGISTRY.md §2.4): a grader other than the author raises a part with its
# evidence.

P6A_AUTHOR = "swarm package 6a builder"
P6A_TESTS = "turbotab/core/tests/test_journey_claims.py"

WHOLE_NUMBERS_WAIT = register(Claim(
    id="readings.whole_numbers_wait",
    kind="rule",
    card="which may be codes for categories (one indicator per level) or amounts (one slope); the "
         "fit waits for the answer for each.",
    goal=("prediction", "inference"),
    regime=Regime("a predictor whose values do not settle whether they are codes or amounts (whole "
                  "numbers with three values or more, numbers written as text, or decimals the "
                  "values do not read as an amount: readings.code_question), whose reading nobody "
                  "has answered, at the fit",
                  requires=(f"{HERE}:whole_numbers_asked",)),
    decision="code_or_count",
    options=("code", "amount"),
    verdict="refused",
    parts=(
        Part("both_codings", grade="judgment", basis=Basis(
            reason="Whole numbers fit both readings, and the two give different models: five "
                   "values as four indicators or as one slope fit different values to the same "
                   "rows (the fires test, by numpy's least squares), so the values cannot settle "
                   "which the column is, and a fit on a guess would enter it one way unasked "
                   "(BLUEPRINT §14.3; readings.code_or_count_reading is never settled by the "
                   "values).")),
        Part("two_values", grade="judgment", basis=Basis(
            reason="Under the fit a column of exactly two values is not asked: its code's one "
                   "indicator is the amount shifted to start at zero and scaled, so beside an "
                   "intercept both span the same columns and fit the same values (the silent "
                   "test, by numpy's least squares; readings.code_question). Combining a unit's "
                   "rows still asks it, since a mean and a most frequent value differ.")),
        Part("waits", grade="judgment", basis=Basis(
            reason="The fit is refused until each column is answered (readings.predictors_or_ask "
                   "raises Unsettled with its ways forward), and an answer recorded is not asked "
                   "again.")),
    ),
    tests=(
        Fires(f"{P6A_TESTS}::test_a_whole_number_predictor_of_five_values_waits_for_its_reading_"
              "since_the_two_codings_fit_apart", truth="independent:numpy"),
        Silent(f"{P6A_TESTS}::test_a_two_valued_predictor_is_not_asked_since_either_coding_fits_"
               "alike", truth="independent:numpy"),
        Silent(f"{P6A_TESTS}::test_a_whole_number_predictor_read_as_an_amount_is_not_asked_again",
               truth="construction"),
    ),
    enforced_by=("turbotab.core.readings:predictors_or_ask", "turbotab.core.readings:code_question"),
    author=P6A_AUTHOR,
    origin="engine",
))

NOT_ASKED = Param(0.5, unit="share of the rows with the outcome measured",
                  constant="turbotab.core.stages.proposals:NOT_ASKED_SHARE")

BLANK_NOT_ASKED = register(Claim(
    id="missing.blank_not_asked",
    kind="rule",
    card="blank usually means not asked.",
    goal=("prediction", "inference"),
    regime=Regime("a yes/no column, or one named for a medication, blank on at least that share of "
                  "the rows with the outcome measured",
                  requires=(f"{HERE}:blank_on_a_yes_no_or_medication_question",)),
    decision="set_missing.not_asked",
    verdict="noted",
    params={"not_asked_share": NOT_ASKED},
    parts=(
        Part("threshold", grade="judgment", basis=Basis(
            reason="Half the rows with the outcome measured is our call, not a published "
                   "threshold: a skip pattern blanks a question for everyone it does not route "
                   "to it, which for NHANES's medication questions is most adults (meds_chol "
                   "blank on 79%, meds_hbp on 71%), while blanks at random rarely reach half (the "
                   "WP7 fixture's quarter, methods.missing.blanks_mean_not_asked).")),
        Part("scope", grade="judgment", basis=Basis(
            reason="Only a yes/no answer or a medication question is read so: a measurement's "
                   "blanks say nothing of a question not asked, so a measurement blank on most "
                   "rows is never read as not asked (the second silent test), and the reason says "
                   "nothing says so. The share counts the rows with the outcome measured, not "
                   "every row: blank on half of them reads as not asked though on under half of "
                   "every row (the second fires test), and blank on half of every row but few of "
                   "the measured ones does not (the third silent test).")),
    ),
    tests=(
        Fires(f"{P6A_TESTS}::test_a_yes_no_answer_blank_on_half_the_rows_reads_as_not_asked",
              truth="construction"),
        Silent(f"{P6A_TESTS}::test_a_yes_no_answer_blank_on_one_row_fewer_than_half_does_not",
               truth="construction"),
        Silent(f"{P6A_TESTS}::test_a_measurement_blank_on_most_rows_is_never_read_as_not_asked",
               truth="construction"),
        Fires(f"{P6A_TESTS}::test_a_yes_no_answer_blank_on_half_the_rows_with_the_outcome_measured_"
              "reads_as_not_asked", truth="construction"),
        Silent(f"{P6A_TESTS}::test_a_yes_no_answer_blank_on_half_of_every_row_but_few_with_the_"
               "outcome_measured_does_not", truth="construction"),
    ),
    enforced_by=("turbotab.core.stages.proposals:missing_reading",
                 "turbotab.core.stages.proposals:_blank_reason"),
    author=P6A_AUTHOR,
    origin="engine",
))

FOLD_FILL_FIRST = register(Claim(
    id="missing.fold_fill_first_under_prediction",
    kind="rule",
    card="Complete cases are the field's habit, but a new row with a blank cannot be scored; a "
         "fill learned in each training fold without the outcome can (Sisk et al. 2023, Stat "
         "Methods Med Res 32:1461).",
    goal=("prediction",),
    regime=Regime("Predict: the missing-values question's options, where a fill in each training "
                  "fold leads and complete cases, the field's first choice, do not",
                  requires=(f"{HERE}:under_prediction",)),
    decision="set_missing",
    options=("impute", "missing_category", "indicators", "complete_case", "multiple_imputation"),
    verdict="rank_lower",
    parts=(
        Part("unscorable", grade="judgment", basis=Basis(
            reason="A model fit on complete cases learns no rule for a blank, so a new row with "
                   "one cannot be scored: scikit-learn's least squares refuses it, while the same "
                   "model behind a fill learned on the training rows scores it (the fires test).")),
        Part("without_outcome", grade="judgment", basis=Basis(
            cites=("sisk2023missing",),
            reason="Sisk et al. 2023 recommend, for a model to be used on new rows, a fill that "
                   "leaves the outcome out, learned on the development rows and applied as it "
                   "was learned: the fills in each training fold (impute, missing_category and "
                   "indicators in methods/missing.py), which never read the outcome.")),
        Part("ranking", grade="judgment", basis=Basis(
            reason="Under Predict a fill in each training fold ranks first, complete cases after "
                   "it, and multiple imputation, which uses the outcome, last "
                   "(custom_sound.missing, methods.missing.METHODS' order); under Estimate "
                   "multiple imputation leads and the tension gives its own reason (the silent "
                   "test).")),
    ),
    tests=(
        Fires(f"{P6A_TESTS}::test_under_prediction_a_fill_learned_in_each_fold_ranks_above_complete_"
              "cases_and_scores_a_new_row_with_a_blank", truth="independent:scikit-learn"),
        Silent(f"{P6A_TESTS}::test_under_estimation_multiple_imputation_leads_and_the_tension_names_"
               "its_own_reason", truth="construction"),
    ),
    enforced_by=("turbotab.core.custom_sound:missing", "turbotab.core.methods.missing:methods_for"),
    author=P6A_AUTHOR,
    origin="engine",
))

EVERY_ROW_FIRST = register(Claim(
    id="exclusions.every_row_first_under_prediction",
    kind="rule",
    card="Screens are the field's habit; under prediction the people they remove will still be "
         "scored, so keeping every row comes first.",
    goal=("prediction",),
    regime=Regime("Predict: the eligibility question's intake screens offered on this table, "
                  "beside keeping every row",
                  requires=(f"{HERE}:under_prediction",)),
    decision="set_exclusions",
    options=("keep_every_row", "goldberg_schofield", "willett_2013_by_sex", "nhs_hpfs_by_sex",
             "sex_neutral_500_5000", "sex_neutral_500_3500"),
    verdict="recommended",
    parts=(
        Part("ranking", grade="judgment", basis=Basis(
            reason="A prediction model is used on everyone it meets, misreporters included, so a "
                   "screen changes who it is developed on and not who it scores: under Predict "
                   "keeping every row ranks first and every screen after it "
                   "(custom_sound.EXCLUSION_ROWS); Banna et al. 2017 (Front Nutr 4:45) ask that "
                   "the every-row analysis be reported beside any screen.")),
        Part("scope", grade="judgment", basis=Basis(
            reason="Under Estimate a screen's people are not scored, so a screen ranks first and "
                   "the every-row analysis is reported beside it (the silent test).")),
    ),
    tests=(
        Fires(f"{P6A_TESTS}::test_under_prediction_keeping_every_row_ranks_first_among_the_screens",
              truth="construction"),
        Silent(f"{P6A_TESTS}::test_under_estimation_a_screen_ranks_first_and_every_row_is_"
               "reported_beside_it", truth="construction"),
    ),
    enforced_by=("turbotab.core.custom_sound:exclusions",),
    author=P6A_AUTHOR,
    origin="engine",
))

SCREEN_WAITS_FOR_DAYS = register(Claim(
    id="exclusions.screen_waits_for_days",
    kind="rule",
    card="so whether each value is one day's intake or a total over several is not settled, and "
         "the screen's bounds are read in a day's.",
    goal=("prediction", "inference"),
    regime=Regime("an intake screen on a total-energy column whose unit is settled and whose days "
                  "are not",
                  requires=(f"{HERE}:energy_days_unsettled",)),
    decision="set_exclusions.energy_days",
    verdict="refused",
    parts=(
        Part("per_day", grade="judgment", basis=Basis(
            reason="The screens' bounds are kcal a day (NUTRITION_PACK §02), and a value that is "
                   "a total over two days is twice its day's intake: a two-day total of 6,000 "
                   "kcal is outside 500–5,000 read as one day and inside it read as two (the fires "
                   "test's arithmetic), so who a screen removes turns on the days.")),
        Part("refused", grade="judgment", basis=Basis(
            reason="While the days are unsettled the screen is refused, never drawn on a guess "
                   "(proposals.unit_refusal), and once they are recorded it is offered (the "
                   "silent test).")),
    ),
    tests=(
        Fires(f"{P6A_TESTS}::test_a_screen_waits_while_the_energy_columns_days_are_unsettled",
              truth="construction"),
        Silent(f"{P6A_TESTS}::test_a_screen_is_offered_once_the_days_are_recorded",
               truth="construction"),
    ),
    enforced_by=("turbotab.core.stages.proposals:unit_refusal",),
    author=P6A_AUTHOR,
    origin="engine",
))

COLLINEAR_PAIR = register(Claim(
    id="explore.collinear_pair",
    kind="parameter",
    card="move almost together (|r| ≥ {collinear:g})",
    goal=("prediction", "inference"),
    regime=Regime("First look's scan of two to five hundred numeric predictors on the rows it "
                  "reads: the training rows under Predict, every analyzed row under Estimate"),
    decision="explore.collinear_scan",
    verdict="noted",
    params={"collinear": Param(0.9, unit="|r|, Pearson's, between two numeric predictors",
                               constant="turbotab.core.stages.explore:COLLINEAR")},
    parts=(
        Part("value", grade="judgment", basis=Basis(
            reason="A convention, not a cited rule (stages/explore.py): a pair this close is "
                   "named in First look so that its importance is read together "
                   "(materiality.collinear_under_prediction), whatever the threshold; a pair at "
                   "the threshold is named (at or above).")),
        Part("both_signs", grade="judgment", basis=Basis(
            reason="|r| is read, so a pair that moves almost together in opposite directions is "
                   "named too (the fires test builds one of each sign).")),
    ),
    tests=(
        Fires(f"{P6A_TESTS}::test_a_pair_correlated_just_above_nine_tenths_either_way_is_named",
              truth="independent:numpy"),
        Silent(f"{P6A_TESTS}::test_a_pair_correlated_just_under_nine_tenths_is_not",
               truth="independent:numpy"),
    ),
    enforced_by=("turbotab.core.stages.explore:_collinear",),
    author=P6A_AUTHOR,
    origin="engine",
))

P6A_CLAIMS = (WHOLE_NUMBERS_WAIT, BLANK_NOT_ASKED, FOLD_FILL_FIRST, EVERY_ROW_FIRST,
              SCREEN_WAITS_FOR_DAYS, COLLINEAR_PAIR)

CLAIMS = (SMALL_SAMPLE_STANDARD, SMALL_SAMPLE_STANDARD_ESTIMATE, SMALL_SAMPLE_STANDARD_CLASSES,
          SMALL_SAMPLE_NO_SEARCH, BOOSTED_TREES_SMALL_ROWS, FOREST_SMALL_ROWS, BASELINE_FLOOR, ROWS_SMD_FLOOR,
          ROWS_SMD_FLOOR_PREDICTION, COLLINEAR_UNDER_PREDICTION, VALIDATION_RESAMPLING_FIRST,
          VALIDATION_PAIRED_INTERVAL, MODELING_NO_RESAMPLING, SEAL_BY_ROW, SEAL_CUT_POINT,
          SEAL_CUT_POINT_UNITS, SEAL_CUT_POINT_UNDATED, SEAL_CUT_POINT_ANALYZED)

__all__ = ["AUTHOR", "BASELINE_FLOOR", "BOOSTED_TREES_SMALL_ROWS", "CLAIMS",
           "COLLINEAR_UNDER_PREDICTION", "FOREST_SMALL_ROWS", "MODELING_NO_RESAMPLING",
           "ROWS_SMD_FLOOR", "ROWS_SMD_FLOOR_PREDICTION", "SEAL_BY_ROW", "SEAL_CUT_POINT",
           "SEAL_CUT_POINT_ANALYZED", "SEAL_CUT_POINT_UNDATED", "SEAL_CUT_POINT_UNITS",
           "SMALL_SAMPLE_STANDARD",
           "SMALL_SAMPLE_STANDARD_CLASSES", "SMALL_SAMPLE_STANDARD_ESTIMATE",
           "SMALL_SAMPLE_NO_SEARCH", "VALIDATION_PAIRED_INTERVAL", "VALIDATION_RESAMPLING_FIRST",
           "classes_or_ordered", "drawn_by_row", "held_out_by_a_cut_point",
           "held_out_in_whole_units", "held_out_row_by_row", "rated_before_any_fit",
           "rows_times_smd",
           "searched_without_a_small_sample_standard", "standard_from_the_study",
           "under_either_purpose", "under_inference", "under_prediction", "verdict_metric",
           "yes_no_left_uncorrected", "yes_no_or_numeric"]

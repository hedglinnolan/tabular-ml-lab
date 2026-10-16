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

from turbotab.core.claims import Basis, Claim, Fires, Param, Part, Regime, Silent, register

HERE = "turbotab.core.claims.models"
AUTHOR = "R0b-models builder"
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
                   "yes/no outcome, a numeric one checked; multiclass and ordered outcomes, "
                   "categorical predictors, p = 50 and NHANES subsamples are not yet checked "
                   "(RECIPES §4.6, before release). It scored predictions alone. The plan reads "
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
                   "outcome chosen on and a numeric one checked), and no estimand. Several "
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
                   "ordered outcomes nowhere; under Estimate there is no estimand either.")),
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
    author=AUTHOR,
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
))

ROWS_SMD_FLOOR_PREDICTION = register(Claim(
    id="materiality.rows_smd_floor_prediction",
    kind="rule",
    card="Under Predict too, the share of rows an energy screen removes, times how far they differ "
         "on what the model adjusts for (its predictors other than one with the exposure role), "
         "is a screening convention: it can say the screen could bias the score or that you "
         "should act on it, never that it doesn't change your numbers here.",
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
        "by role less the outcome, those left out and one exposure, the estimand's or else the "
        "first column with the exposure role), whatever its calibration cases",
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
))

COLLINEAR_UNDER_PREDICTION = register(Claim(
    id="materiality.collinear_under_prediction",
    kind="rule",
    card="Under Predict, both of a nearly collinear pair are kept and keeping them records no "
         "selection, a convention set for you to confirm: such a pair does not harm prediction "
         "on rows like these. Reading their importance together is the line's advice to you.",
    methods="Under Predict, both predictors of a nearly collinear pair were kept, a convention the "
            "analyst confirmed rather than a decision: keeping both recorded no predictor "
            "selection, taking back the line's own elastic-net filter where it was recorded and "
            "leaving any other selection as it stood, since collinearity does not harm "
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
    ),
    enforced_by=("turbotab.core.materiality:collinear_label",
                 "turbotab.core.stages.explore:_collinear_lever"),
    author=AUTHOR,
    ruled="calm/BEATS_PREDICT.md M4, its fourth line (near-collinear predictors, all kept; the "
          "grouped importance it rules for Results is not built)",
))

CLAIMS = (SMALL_SAMPLE_STANDARD, SMALL_SAMPLE_STANDARD_ESTIMATE, SMALL_SAMPLE_STANDARD_CLASSES,
          SMALL_SAMPLE_NO_SEARCH, BOOSTED_TREES_SMALL_ROWS, FOREST_SMALL_ROWS, BASELINE_FLOOR, ROWS_SMD_FLOOR,
          ROWS_SMD_FLOOR_PREDICTION, COLLINEAR_UNDER_PREDICTION)

__all__ = ["AUTHOR", "BASELINE_FLOOR", "BOOSTED_TREES_SMALL_ROWS", "CLAIMS",
           "COLLINEAR_UNDER_PREDICTION", "FOREST_SMALL_ROWS", "ROWS_SMD_FLOOR",
           "ROWS_SMD_FLOOR_PREDICTION", "SMALL_SAMPLE_STANDARD", "SMALL_SAMPLE_STANDARD_CLASSES",
           "SMALL_SAMPLE_STANDARD_ESTIMATE", "SMALL_SAMPLE_NO_SEARCH", "classes_or_ordered",
           "rated_before_any_fit", "rows_times_smd", "searched_without_a_small_sample_standard",
           "standard_from_the_study", "under_either_purpose", "under_inference",
           "under_prediction", "verdict_metric", "yes_no_or_numeric"]

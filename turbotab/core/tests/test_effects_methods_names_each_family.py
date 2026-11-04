"""The effects stage's methods text says each chosen family's sequence under its own name wherever
two families report an estimate from different models, whichever order they were chosen in (the
verifier's V4a3-2; the orchestrator's ruling (3) on the swarm's tail).

On the base it named the families only where a later family reported from a model the first did
not, so with the family that reports more chosen first, its unscoped sequence stood for the whole
analysis: under [feature-wise, linear] the linear model's singular Models 2 and 3 went unsaid,
and under [linear, feature-wise] with the estimand `grp` feature-wise regression's refusal of
every model did.

Truth from construction (``test_featurewise_family_counts_its_tests``' fixtures): `fib` = 2 ×
`activity` + 1, so numpy finds the linear model's Model 2 and Model 3 designs singular while
feature-wise regression tests `fiber` in each; `grp` is a category, which feature-wise regression
adjusts for and does not test, while the linear model reports its indicators.

Two families that report from the same models but say different things of them are said apart
too (the verifier's FW-V1): the linear model fits an exposure family's members together, each
adjusted for the other, where feature-wise regression fits each alone, so the first family's
sequence was false of the second whichever came first. The silent side: families whose models and
words agree (a single exposure, fit the same way under both) keep the first family's text,
unscoped.
"""
from __future__ import annotations

import statsmodels.api as sm
import pytest

from turbotab.core.tests import test_featurewise_family_counts_its_tests as T
from turbotab.core.tests.acceptance import estimand_fixtures as ef

# The linear model holds both members in each model, each adjusted for the other (statsmodels'
# glucose ~ fiber + fib reproduces its unadjusted rows), so its unadjusted model is said with them
# fit together and Model 2 with the covariates and the other exposure, never a member adjusted for
# itself (pp2-family-effect-words; the verifier's FW-V3).
LINEAR_FIB = (f"Under the linear model, the estimate of each exposure is reported only from the "
              f"model without the covariates, with `fiber` and `fib` fit together, each adjusted "
              f"for the other, fit on all 400 analyzed rows. Model 2, the primary, adjusted for "
              f"`age`, `sex`, `smoking`, `activity` and the other exposure, reports none: "
              f"{T.SINGULAR} Model 3, further adjusted for `bmi`, reports none: {T.SINGULAR}")


def test_with_the_family_that_reports_more_chosen_first_each_is_said_by_name(tmp_path):
    """[feature-wise, linear], the family {`fiber`, `fib`}: feature-wise regression reports in the
    unadjusted model, Model 2 and Model 3, the linear model in the unadjusted model alone (its
    Models 2 and 3 singular). The methods text was feature-wise regression's alone; now each
    family's under its name, the linear model's refusals in their words, then the family's
    multiplicity sentence, as in the other order."""
    frame = T._fib_frame()
    out = ef.run(frame, tmp_path, T._family({"fiber": "exposure", "fib": "exposure"},
                                            ["featurewise", "linear"], amounts=T.FIB_AMOUNTS))
    assert T._rank_deficient(frame, ["fiber", "fib", *T.MODEL_2])  # the premise
    linear = T._sequence(out, "linear")
    for key in ("model_2", "model_3"):
        assert linear[key]["effects"] is None
        assert linear[key]["inference"]["refused"] == T.SINGULAR
    together = sm.OLS(frame["glucose"].to_numpy(float), T._design(frame, ["fiber", "fib"])).fit()
    for row in linear["crude"]["effects"]:
        assert row["estimate"] == pytest.approx(float(together.params[row["feature"]]), rel=1e-9)
    assert out["effects"]["methods"] == (
        f"Under the feature-wise regression, {T.SEQUENCE}{T.MODEL_3}{T.APART}{T.BENCHMARKED} "
        f"{LINEAR_FIB} {T.FIB_FAMILY}")


def test_with_the_family_that_reports_nothing_chosen_second_each_is_said_by_name(tmp_path):
    """[linear, feature-wise], the estimand `grp`: the linear model reports `grp`'s indicators in
    Model 2 (statsmodels' glucose ~ `grp`'s indicators + `fiber` + the covariates), feature-wise
    regression in no model. The methods text was the linear model's alone; now it ends with
    feature-wise regression's refusal, in its words, under its name."""
    frame = T._frame()
    out = ef.run(frame, tmp_path, T._state({"fiber": "exposure", "grp": "exposure"},
                                           ["linear", "featurewise"], exposure="grp"))
    rows = {r["feature"]: r for r in T._sequence(out, "linear")["model_2"]["effects"]}
    fit = sm.OLS(frame["glucose"].to_numpy(float),
                 T._design(frame, [*T.GRP, *T.MODEL_2, "fiber"])).fit()
    for c in T.GRP:
        assert rows[c]["estimate"] == pytest.approx(float(fit.params[c]), rel=1e-9)
    assert T._sequence(out)["model_2"]["inference"]["refused"] == T.GRP_ALONE  # the premise
    methods = out["effects"]["methods"]
    assert methods.startswith("Under the linear model, the estimate of `grp` is reported across a "
                              "declared sequence of models fit on all 400 analyzed rows: "
                              "unadjusted; Model 2, the primary, adjusted for `age`, `sex`, "
                              "`smoking`, `activity` and `fiber`;")
    assert methods.endswith(f" Under the feature-wise regression, no estimate of `grp` is "
                            f"reported: {T.GRP_ALONE}")


def test_families_that_report_from_the_same_models_with_different_words_are_each_said_by_name(
        tmp_path):
    """The numeric family {`fiber`, `sugar`} under [feature-wise, linear]: each family reports in
    every model, and the methods text was feature-wise regression's alone, unscoped, so it said the
    linear model's unadjusted model was "unadjusted" and its Model 2 adjusted for the covariates
    alone, where each of its rows is statsmodels' fit of both members together, each adjusted for
    the other (the verifier's FW-V1). Now each family's text is said under its name, then the
    family's sentence."""
    frame = T._frame()
    out = ef.run(frame, tmp_path, T._family({"fiber": "exposure", "sugar": "exposure"},
                                            ["featurewise", "linear"]))
    for key, adjust in (("crude", []), ("model_2", T.MODEL_2)):
        together = sm.OLS(frame["glucose"].to_numpy(float),
                          T._design(frame, ["fiber", "sugar", *adjust])).fit()
        for row in T._sequence(out, "linear")[key]["effects"]:
            assert row["estimate"] == pytest.approx(float(together.params[row["feature"]]),
                                                    rel=1e-9)
        for row in T._sequence(out)[key]["effects"]:
            assert row["estimate"] == pytest.approx(T._ols(frame, row["feature"], adjust)[0],
                                                    rel=1e-9)
    assert out["effects"]["methods"] == (
        f"Under the feature-wise regression, {T.SEQUENCE}{T.MODEL_3}{T.APART}{T.BENCHMARKED} "
        f"Under the linear model, {T.LINEAR_NUMERIC} {T.BOTH_TWO}")


def test_families_whose_models_and_words_agree_keep_the_first_familys_text(tmp_path):
    """Silent: the single exposure `fiber` under [feature-wise, linear]. Each fits `fiber` in each
    model as statsmodels' glucose ~ `fiber` (and the covariates) does, so they report from the same
    models and say the same of them, and the methods text is the first family's sequence,
    unscoped, under no family's name."""
    frame = T._frame()
    out = ef.run(frame, tmp_path, T._state({"fiber": "exposure"}, ["featurewise", "linear"]))
    for family in ("featurewise", "linear"):
        sequence = T._sequence(out, family)
        for key, adjust in (("crude", []), ("model_2", T.MODEL_2),
                            ("model_3", [*T.MODEL_2, "bmi"])):
            (row,) = sequence[key]["effects"]
            assert row["estimate"] == pytest.approx(T._ols(frame, "fiber", adjust)[0], rel=1e-9)
    assert [(s["label"], s["adjusted_for"]) for s in T._sequence(out, "featurewise").values()] == [
        (s["label"], s["adjusted_for"]) for s in T._sequence(out, "linear").values()]
    methods = out["effects"]["methods"]
    assert methods.startswith(
        "The estimate of `fiber` is reported across a declared sequence of models fit on all 400 "
        "analyzed rows: unadjusted; Model 2, the primary, adjusted for `age`, `sex`, `smoking` "
        "and `activity`; Model 3, further adjusted for `bmi`, a possible mediator, so not a total "
        "effect.")
    assert "Under the" not in methods

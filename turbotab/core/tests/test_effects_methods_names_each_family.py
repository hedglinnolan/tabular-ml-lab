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
adjusts for and does not test, while the linear model reports its indicators. The silent side:
families that report from the same models keep the first family's text, byte for byte.
"""
from __future__ import annotations

import statsmodels.api as sm
import pytest

from turbotab.core.tests import test_featurewise_family_counts_its_tests as T
from turbotab.core.tests.acceptance import estimand_fixtures as ef

LINEAR_FIB = (f"Under the linear model, the estimate of each exposure is reported from the "
              f"unadjusted model alone, fit on all 400 analyzed rows. Model 2, the primary, "
              f"adjusted for `age`, `sex`, `smoking` and `activity`, reports none: {T.SINGULAR} "
              f"Model 3, further adjusted for `bmi`, reports none: {T.SINGULAR}")


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


def test_families_that_report_from_the_same_models_keep_the_first_familys_text(tmp_path):
    """Silent, byte for byte: the numeric family {`fiber`, `sugar`} under [feature-wise, linear]
    (each reports in every model) reads as feature-wise regression's text alone, unscoped, as on
    the base."""
    out = ef.run(T._frame(), tmp_path, T._family({"fiber": "exposure", "sugar": "exposure"},
                                                 ["featurewise", "linear"]))
    assert out["effects"]["methods"] == (
        f"T{T.SEQUENCE[1:]}{T.MODEL_3}{T.APART}{T.BENCHMARKED} {T.ALL_TWO}")

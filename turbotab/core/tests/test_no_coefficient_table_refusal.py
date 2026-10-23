"""Under Estimate, the declared models (``effects``) and the model further adjusted beside the
primary (``secondary``) are refit only for a family with a coefficient table. Where no chosen
family has one (elastic net, ridge, boosted trees, random forest, XGBoost: the crash sweep's
cases on the walking_bp table) the stage is refused, typed as a fit-time refusal (SWARM_PLAN §8)
with its code and ways forward, so the record classes it as refused, not a crash
(``jobs.failure_of``), and with the words a person reads unchanged.

Each way forward adds one family whose table these rows and these answers give.
Expected values come from the construction: the families registered for a numeric outcome whose
declarations give a coefficient table under inference are the linear model, the feature-wise
regression, the random-intercept mixed model and GEE (``models/linear.py``, ``featurewise.py``,
``repeated.py``), the linear model first; for a yes/no outcome the same less the mixed model. The
mixed model and GEE model the unit, and in ``ef.cohort`` the identifier `pid` takes a value of its
own on every row, so no unit repeats and their tables are refused (``repeated._no_units``); with
`pid` repeated three times over (100 units of 3 rows) both are offered and fit. The feature-wise
tests pool no imputed copies, so under multiple imputation with `activity` blank on 112 rows
their table is refused (``omics.unpooled_refusal``) and only the linear model is offered. Of the
four, only the linear model declares a design-based estimator (``InferenceDecl.design_based``), so
under the surveyed population the feature-wise table is blocked (``survey.no_design_estimator``)
and only the linear model is offered; under the "these participants" answer both are. With the
first way forward taken, the stage fits the linear model.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import turbotab.core.models  # noqa: F401 - registers the families
from turbotab.core import decisions as d
from turbotab.core.decisions import parse_decision
from turbotab.core.graph import StageRefusal
from turbotab.core.jobs import failure_of
from turbotab.core.tests.acceptance import estimand_fixtures as ef

# Read off the family declarations by hand, in their registration order (the shelf's).
WITH_A_TABLE = ["linear", "featurewise", "mixed", "gee"]
# Of those, the ones these rows give a table for where no identifier repeats (the mixed model and
# GEE model the unit), and under multiple imputation with a blank (the feature-wise tests pool no
# copies).
NO_UNIT_REPEATS = ["linear", "featurewise"]
UNDER_IMPUTATION = ["linear"]
DESIGN_BASED = ["linear"]
UNITS = 100  # `pid` = row // 3 on 300 rows
LABELS = {"linear": "Linear model", "featurewise": "Feature-wise regression",
          "mixed": "Random-intercept mixed model", "gee": "GEE (exchangeable)"}
EFFECTS_WORDS = ("None of the chosen model families has a coefficient table, so there is no "
                 "effect to report; choose the linear model.")
SECONDARY_WORDS = ("None of the chosen model families has a coefficient table, so there is no "
                   "estimate to set beside the primary's; choose the linear model.")


def _run(tmp_path: Path, models: list[str], *, secondary: bool, imputed: bool = False,
         repeated: bool = False, binary: bool = False, survey: str | None = None) -> dict:
    """``ef.cohort(300)`` under Estimate (the mean difference in `glucose` per unit of `fiber`, or
    with ``binary`` the odds ratio of `dm`; `bmi` of unknown timing, so a secondary model is
    declared beside the primary), the chosen ``models``: design, then the effects stage, or the
    secondary stage. ``imputed``: `activity` blank on 112 rows, under multiple imputation (m 20).
    ``repeated``: `pid` repeats over 100 units of 3 rows, the grain answer says so, and the split
    keeps each unit whole. ``survey``: a weight, 4 strata and 2 PSUs in each, the survey answer
    ``"population"`` or ``"sample"``."""
    from turbotab.core.stages.effects import effects_stage
    from turbotab.core.stages.modeling import design_stage
    from turbotab.core.stages.secondary import secondary_stage
    from turbotab.core.tests import modeling_fixtures as mf

    frame = ef.cohort(300, seed=5)
    answers: dict = {}
    if imputed:
        frame.loc[frame.index[:112], "activity"] = np.nan
        answers["missing"] = {"strategy": "multiple_imputation", "m": 20}
    if repeated:
        frame["pid"] = np.arange(len(frame)) // 3
        answers["grain"] = d.GrainSpec(grain="repeated", id_column="pid")
    if survey is not None:
        rng = np.random.default_rng(5)
        frame["sampling_weight"] = rng.uniform(0.5, 3.0, len(frame))
        frame["stratum"] = rng.integers(1, 5, len(frame))
        frame["psu"] = frame["stratum"] * 10 + rng.integers(1, 3, len(frame))
        answers["roles"] = {**ef.ROLES, "sampling_weight": "design", "stratum": "design",
                            "psu": "design"}
        answers["survey"] = d.SurveySpec(estimand=survey, weight="sampling_weight",
                                         strata="stratum", psu="psu")
        answers["amounts"] = {**ef.AMOUNTS, "code_or_count:stratum": "code",
                              "code_or_count:psu": "code"}
    outcome = ({"target": "dm", "task": "binary", "event": "yes", "measure": "odds_ratio"}
               if binary else {"target": "glucose", "task": "regression",
                               "measure": "mean_difference"})
    st = ef.state(models=models, **outcome, **answers)
    paths = mf.ingest_frame(frame, tmp_path)
    split = (mf.split_bundle(np.arange(len(frame)), holdout=0.0, seed=st.split.seed,
                             groups=frame["pid"].to_numpy(), grouped_by="pid")
             if repeated else mf.split_bundle(np.arange(len(frame)), holdout=0.0,
                                              seed=st.split.seed))
    info = mf.target_info(st.task, st.target)
    design = design_stage(mf.context(st, {"split": split, "target_info": info}, paths))
    stage = secondary_stage if secondary else effects_stage
    inputs = {"design": design, "split": split, "target_info": info}
    return stage(mf.context(st, inputs, paths)).data


def _check_refusal(exc: StageRefusal, words: str, chosen: list[str], offered: list[str]) -> None:
    assert isinstance(exc, ValueError)
    shown, refusal = failure_of(exc)
    assert shown == f"ValueError: {words}"
    assert refusal is not None and refusal["code"] == "no_coefficient_table"
    assert [e["label"] for e in refusal["exits"]] == [
        f"Add {LABELS[k]}, which has a coefficient table" for k in offered]
    for key, way in zip(offered, refusal["exits"]):
        decision = parse_decision(way["decision"])
        assert decision.kind == "select_models" and decision.models == [*chosen, key]


@pytest.mark.parametrize("family", ["elastic_net", "random_forest"])
def test_the_declared_models_with_no_family_that_has_a_table_are_refused_typed(tmp_path, family):
    """Elastic net reports shrunk coefficients and no inference table; a random forest reports
    none: under Estimate the effects stage has nothing to refit, and refuses in words, typed, with
    a way forward for each family whose table these rows give, where it raised a bare
    ``ValueError``. No identifier repeats, so neither the mixed model nor GEE is offered: each
    would refit the declared models to a table refused for want of a unit."""
    with pytest.raises(StageRefusal) as refused:
        _run(tmp_path, [family], secondary=False)
    _check_refusal(refused.value, EFFECTS_WORDS, [family], NO_UNIT_REPEATS)


@pytest.mark.parametrize("family", ["elastic_net", "random_forest"])
def test_the_secondary_model_with_no_family_that_has_a_table_is_refused_typed(tmp_path, family):
    """The model further adjusted for `bmi` beside the primary: the same refusal, in its words."""
    with pytest.raises(StageRefusal) as refused:
        _run(tmp_path, [family], secondary=True)
    _check_refusal(refused.value, SECONDARY_WORDS, [family], NO_UNIT_REPEATS)


def test_for_a_yes_no_outcome_gee_is_not_offered_where_no_identifier_repeats(tmp_path):
    """Ridge on `dm`: the families with a table for a yes/no outcome are the linear model (the
    logistic), the feature-wise regression and GEE; with no identifier repeating, GEE is left
    out."""
    with pytest.raises(StageRefusal) as refused:
        _run(tmp_path, ["ridge"], secondary=False, binary=True)
    _check_refusal(refused.value, EFFECTS_WORDS, ["ridge"], NO_UNIT_REPEATS)


@pytest.mark.parametrize("secondary", [False, True])
def test_under_multiple_imputation_the_ways_forward_leave_out_the_unpooled_table(tmp_path,
                                                                                  secondary):
    """`activity` blank on 112 rows under multiple imputation: the feature-wise table pools no
    copies, so the fit refuses it (the design's impute line says so), and the only way forward is
    the linear model, where the refusal offered the feature-wise regression too."""
    with pytest.raises(StageRefusal) as refused:
        _run(tmp_path, ["elastic_net"], secondary=secondary, imputed=True)
    _check_refusal(refused.value, SECONDARY_WORDS if secondary else EFFECTS_WORDS,
                   ["elastic_net"], UNDER_IMPUTATION)


@pytest.mark.parametrize("secondary", [False, True])
def test_where_the_identifier_repeats_the_mixed_model_and_gee_are_offered(tmp_path, secondary):
    """`pid` repeats over 100 units of 3 rows: every family with a table is offered, the mixed
    model and GEE among them."""
    with pytest.raises(StageRefusal) as refused:
        _run(tmp_path, ["elastic_net"], secondary=secondary, repeated=True)
    _check_refusal(refused.value, SECONDARY_WORDS if secondary else EFFECTS_WORDS,
                   ["elastic_net"], WITH_A_TABLE)


@pytest.mark.parametrize("secondary", [False, True])
@pytest.mark.parametrize(("survey", "offered"), [("population", DESIGN_BASED),
                                                 ("sample", NO_UNIT_REPEATS)])
def test_under_the_surveyed_population_only_a_design_based_table_is_offered(tmp_path, secondary,
                                                                            survey, offered):
    """Under the surveyed population a family with no design-based estimator is blocked, so the
    feature-wise regression is left out, where it was offered; for these participants alone it is
    not blocked, and is offered."""
    with pytest.raises(StageRefusal) as refused:
        _run(tmp_path, ["elastic_net"], secondary=secondary, survey=survey)
    _check_refusal(refused.value, SECONDARY_WORDS if secondary else EFFECTS_WORDS,
                   ["elastic_net"], offered)


@pytest.mark.parametrize(("key", "answers"), [
    ("featurewise", {}), ("linear", {"imputed": True}), ("linear", {"survey": "population"}),
    ("featurewise", {"survey": "sample"}), ("mixed", {"repeated": True}),
    ("gee", {"repeated": True})])
def test_a_way_forward_taken_is_fit(tmp_path, key, answers):
    """Each way forward taken where it is offered: the feature-wise regression under complete
    cases and for these participants, the linear model under multiple imputation (pooled over the
    copies) and under the surveyed population (over the design), the mixed model and GEE on the
    100 units of 3 rows (by those 100 units). Model 2's table and both of the secondary stage's
    are fit, with an interval for `fiber`, not refused."""
    effects = _run(tmp_path / "effects", ["elastic_net", key], secondary=False, **answers)
    assert [f["family"] for f in effects["families"]] == [key]
    model_2 = ef.sequence(effects)["model_2"]
    secondary = _run(tmp_path / "secondary", ["elastic_net", key], secondary=True, **answers)
    [family] = secondary["families"]
    assert family["family"] == key
    tables = [(model_2["inference"], model_2["effects"]),
              *[(fit["inference"], fit["coefficients"]) for fit in family["fits"]]]
    assert len(tables) == 3
    for inference, rows in tables:
        assert inference["refused"] is None
        if answers.get("repeated"):
            assert f"G = {UNITS} " in inference["caption"]
        if answers.get("imputed"):
            assert inference["missing"]["method"] == "multiple_imputation"
        if answers.get("survey") == "population":
            assert inference["covariance"] == "design"
        [fiber] = rows
        assert fiber["feature"] == "fiber"
        assert fiber["ci_low"] < fiber["estimate"] < fiber["ci_high"]


def test_with_the_first_way_forward_taken_the_linear_model_is_refit(tmp_path):
    """The first way forward adds the linear model to elastic net: the effects stage refits the
    linear model's declared models (Model 2 reports the exposure's estimate), and the secondary
    stage fits the linear model with and without `bmi`; elastic net takes no part in either."""
    effects = _run(tmp_path / "effects", ["elastic_net", "linear"], secondary=False)
    assert [f["family"] for f in effects["families"]] == ["linear"]
    model_2 = ef.sequence(effects)["model_2"]
    assert [r["feature"] for r in model_2["effects"]] == ["fiber"]
    secondary = _run(tmp_path / "secondary", ["elastic_net", "linear"], secondary=True)
    [linear] = secondary["families"]
    assert linear["family"] == "linear"
    assert [[r["feature"] for r in f["coefficients"]] for f in linear["fits"]] == [["fiber"],
                                                                                   ["fiber"]]

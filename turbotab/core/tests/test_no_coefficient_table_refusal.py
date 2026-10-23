"""Under Estimate, the declared models (``effects``) and the model further adjusted beside the
primary (``secondary``) are refit only for a family with a coefficient table. Where no chosen
family has one (elastic net, ridge, boosted trees, random forest, XGBoost: the crash sweep's
cases on the walking_bp table) the stage is refused, typed as a fit-time refusal (SWARM_PLAN §8)
with its code and ways forward, so the record classes it as refused, not a crash
(``jobs.failure_of``), and with the words a person reads unchanged where the linear model is
offered first; the words name the first family offered, or say that none is. The sensitivity
stage, which runs under either goal, is refused the same way where no chosen family reports
coefficients (it raised a bare ``ValueError``).

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
and only the linear model is offered; under the "these participants" answer both are. Below the
unit floor every table that rests on cluster-robust intervals reports none (see the floor's
section), so only the mixed model is offered. With the first way forward taken, the stage fits
the linear model.
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
         repeated: bool = False, binary: bool = False, survey: str | None = None,
         units: int | None = None) -> dict:
    """``ef.cohort(300)`` under Estimate (the mean difference in `glucose` per unit of `fiber`, or
    with ``binary`` the odds ratio of `dm`; `bmi` of unknown timing, so a secondary model is
    declared beside the primary), the chosen ``models``: design, then the effects stage, or the
    secondary stage. ``imputed``: `activity` blank on 112 rows, under multiple imputation (m 20).
    ``repeated``: `pid` repeats over 100 units of 3 rows, the grain answer says so, and the split
    keeps each unit whole. ``units``: `pid` repeats over that many units of equal size (60 rows
    each for 5), the grain answer says so, and the split keeps each unit whole from 8 units and
    is drawn by row below (as the seal draws it). ``survey``: a weight, 4 strata and 2 PSUs in
    each, the survey answer ``"population"`` or ``"sample"``."""
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
    if units is not None:
        frame["pid"] = np.arange(len(frame)) * units // len(frame)
        assert frame["pid"].nunique() == units
        answers["grain"] = d.GrainSpec(grain="repeated", id_column="pid")
    grouped = repeated or (units is not None and units >= 8)
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
             if grouped else mf.split_bundle(np.arange(len(frame)), holdout=0.0,
                                             seed=st.split.seed))
    info = mf.target_info(st.task, st.target)
    design = design_stage(mf.context(st, {"split": split, "target_info": info}, paths))
    stage = secondary_stage if secondary else effects_stage
    inputs = {"design": design, "split": split, "target_info": info}
    return stage(mf.context(st, inputs, paths)).data


def _check_refusal(exc: StageRefusal, words: str, chosen: list[str], offered: list[str],
                   has: str = "has a coefficient table") -> None:
    assert isinstance(exc, ValueError)
    shown, refusal = failure_of(exc)
    assert shown == f"ValueError: {words}"
    assert refusal is not None and refusal["code"] == "no_coefficient_table"
    assert [e["label"] for e in refusal["exits"]] == [f"Add {LABELS[k]}, which {has}"
                                                      for k in offered]
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
    ("gee", {"repeated": True}), ("mixed", {"units": 5})])
def test_a_way_forward_taken_is_fit(tmp_path, key, answers):
    """Each way forward taken where it is offered: the feature-wise regression under complete
    cases and for these participants, the linear model under multiple imputation (pooled over the
    copies) and under the surveyed population (over the design), the mixed model and GEE on the
    100 units of 3 rows (by those 100 units), and the mixed model on 5 units of 60 rows, below the
    unit floor, where it is the only one offered. Model 2's table and both of the secondary
    stage's are fit, with an interval for `fiber`, not refused."""
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
        if answers.get("repeated") or answers.get("units"):
            assert f"G = {answers.get('units') or UNITS} " in inference["caption"]
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


# ── below the unit floor (the verifier's V2) ─────────────────────────────────
#
# With rows clustered by fewer units than the floor (``inference.min_clusters``, the seal's own),
# every table that rests on cluster-robust intervals reports none: the linear model's, the
# feature-wise tests' and GEE's (each runs ``inference.floor_refusal``; written out here from
# ``models/inference.py``, ``featurewise.py`` and ``repeated.py``). The mixed model's intervals
# are model-based (REML, on t with Satterthwaite degrees of freedom), so it is the one way forward
# for a numeric outcome; a yes/no outcome has no mixed model, so none is offered. At the floor
# itself every family with a table is offered, as with 100 units.

BELOW_FLOOR, AT_FLOOR = 5, 8
FLOOR_TABLES = ["linear", "featurewise", "gee"]


def _floor_holds() -> None:
    from turbotab.core.models.inference import min_clusters

    assert BELOW_FLOOR < min_clusters() <= AT_FLOOR


@pytest.mark.parametrize("secondary", [False, True])
def test_below_the_unit_floor_only_the_mixed_model_is_offered_and_named(tmp_path, secondary):
    """5 units of 60 rows: the linear model, the feature-wise regression and GEE are left out,
    and the words name the mixed model, offered, where they said "choose the linear model"."""
    _floor_holds()
    with pytest.raises(StageRefusal) as refused:
        _run(tmp_path, ["elastic_net"], secondary=secondary, units=BELOW_FLOOR)
    words = (SECONDARY_WORDS if secondary else EFFECTS_WORDS).replace(
        "choose the linear model", "choose the random-intercept mixed model")
    _check_refusal(refused.value, words, ["elastic_net"], ["mixed"])
    assert "linear model" not in str(refused.value)


@pytest.mark.parametrize("secondary", [False, True])
def test_below_the_unit_floor_a_yes_no_outcome_is_offered_no_family_and_says_so(tmp_path,
                                                                                 secondary):
    """Ridge on `dm` with 5 units: the families with a table for a yes/no outcome (the linear
    model, the feature-wise regression, GEE) each rest on cluster-robust intervals, so none is
    offered, and the words say no family gives a table here, naming none."""
    _floor_holds()
    with pytest.raises(StageRefusal) as refused:
        _run(tmp_path, ["ridge"], secondary=secondary, units=BELOW_FLOOR, binary=True)
    nothing = "no estimate to set beside the primary's" if secondary else "no effect to report"
    words = (f"None of the chosen model families has a coefficient table, so there is {nothing}; "
             f"no model family TurboTab offers gives one on these rows under these answers.")
    _check_refusal(refused.value, words, ["ridge"], [])
    assert "linear model" not in str(refused.value)


@pytest.mark.parametrize("secondary", [False, True])
def test_at_the_unit_floor_every_family_with_a_table_is_offered(tmp_path, secondary):
    _floor_holds()
    with pytest.raises(StageRefusal) as refused:
        _run(tmp_path, ["elastic_net"], secondary=secondary, units=AT_FLOOR)
    _check_refusal(refused.value, SECONDARY_WORDS if secondary else EFFECTS_WORDS,
                   ["elastic_net"], WITH_A_TABLE)


@pytest.mark.parametrize("key", FLOOR_TABLES)
def test_below_the_unit_floor_each_family_left_out_has_its_table_refused(tmp_path, key):
    """Why they are left out, from the stage itself: each taken with 5 units gives Model 2's table
    refused for want of units, its interval withheld."""
    _floor_holds()
    effects = _run(tmp_path, ["elastic_net", key], secondary=False, units=BELOW_FLOOR)
    model_2 = ef.sequence(effects)["model_2"]
    refused = model_2["inference"]["refused"] or ""
    assert refused.startswith(f"`pid` has {BELOW_FLOOR} units, fewer than the "), refused
    assert "TurboTab requires for cluster-robust intervals" in refused
    assert all(r["ci_low"] is None for r in model_2["effects"] or [])


# ── the sensitivity stage (its bare ValueError, typed) ───────────────────────

# Read off the family declarations by hand: under prediction the sensitivity stage reports each
# family's coefficients as it fits them, so a way forward is a family that reports coefficients
# and serves prediction, in the shelf's order. The mixed model and GEE model the unit, and no
# identifier repeats in ``ef.cohort``, so neither is offered.
PREDICT_COEFFICIENTS = ["linear", "elastic_net", "ridge", "huber", "screened_elastic_net"]
LABELS.update({"elastic_net": "Elastic net", "ridge": "Ridge", "huber": "Robust linear regression",
               "screened_elastic_net": "Screened elastic net"})


def _sensitivity(tmp_path: Path, models: list[str], purpose: str) -> dict:
    """``ef.cohort(300)`` with the chosen ``models`` under ``purpose``: design, then the
    sensitivity stage (the primary analysis alone: no rule excludes a row)."""
    from turbotab.core.datastore import DataStore
    from turbotab.core.stages.modeling import design_stage
    from turbotab.core.stages.sensitivity import sensitivity_stage
    from turbotab.core.tests import modeling_fixtures as mf

    frame = ef.cohort(300, seed=5)
    st = ef.state(target="glucose", task="regression", measure="mean_difference", models=models)
    st = st.model_copy(update={"purpose": purpose})
    paths = mf.ingest_frame(frame, tmp_path)
    split = mf.split_bundle(np.arange(len(frame)), holdout=0.0, seed=st.split.seed)
    inputs = {"split": split, "target_info": mf.target_info(st.task, st.target)}
    design = design_stage(mf.context(st, inputs, paths))
    with DataStore(paths["data"], 1 << 30) as store:
        ingest = store.info().to_dict()
    return sensitivity_stage(mf.context(st, {**inputs, "design": design, "ingest": ingest},
                                        paths)).data


@pytest.mark.parametrize("purpose", ["inference", "prediction"])
def test_the_sensitivity_stage_with_no_family_that_has_coefficients_is_refused_typed(tmp_path,
                                                                                      purpose):
    """A random forest reports no coefficients: the sensitivity stage, which runs under either
    goal, raised a bare ``ValueError``. It is refused typed, in words true under each goal: under
    Estimate the secondary stage's, each way forward a family with a coefficient table these rows
    give; under Predict, where each family's coefficients are reported as fit, a family that
    reports coefficients and serves prediction."""
    with pytest.raises(StageRefusal) as refused:
        _sensitivity(tmp_path, ["random_forest"], purpose)
    if purpose == "inference":
        _check_refusal(refused.value, SECONDARY_WORDS, ["random_forest"], NO_UNIT_REPEATS)
        return
    words = ("None of the chosen model families reports coefficients, so there is no estimate to "
             "set beside the primary's; choose the linear model.")
    _check_refusal(refused.value, words, ["random_forest"], PREDICT_COEFFICIENTS,
                   has="reports coefficients")


@pytest.mark.parametrize("purpose", ["inference", "prediction"])
def test_with_the_first_way_forward_taken_the_sensitivity_stage_is_fit(tmp_path, purpose):
    """The same stage with the linear model beside the forest: no refusal, and the linear model's
    coefficients of the primary analysis are reported (the forest has none)."""
    out = _sensitivity(tmp_path, ["random_forest", "linear"], purpose)
    [linear] = out["families"]
    assert linear["family"] == "linear"
    [primary] = linear["fits"]
    assert primary["label"] == "Primary" and primary["coefficients"]

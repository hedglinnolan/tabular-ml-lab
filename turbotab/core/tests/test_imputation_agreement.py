"""The missing-values answer refuses what the fit would refuse (PP2, pp2-imputation-agreement).

Before this package the check at the answer ran the sampler's refusals before drawing only where
the width rule refused the model, so a plan the sampler refuses for another reason (a column
recorded once on a model of 80 columns or fewer, a logged zero, a 0/1 column recorded once or a
text column recorded on no analyzed row, which the rule does not hold) was locked and then refused
by the fit. Now the answer and the lock refuse every plan the sampler refuses before it draws a
copy, at any width, in the fit's words, with the fit's ways forward, each where the record accepts
it when taken; a roles answer's ways forward are checked against the state after it too. The
leave-out ways never name a column the energy model is computed with, a missing-values answer
leaving one out is refused, and the log residual's zero is a refusal in words at the design. The
scales stage refuses the items' copies in the fit's words with its ways forward.

Every expected value is from construction (the rows recorded, the columns built), pandas' counts,
or R (``nnet::multinom``'s coefficient matrix); each guard is tested where it fires and where it
stays silent.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import pytest

from turbotab.core.decisions import (EnergyAdjustment, ExclusionRule, MissingSpec, ProjectState,
                                     Refusal, SetMissing, SetRoles, parse_decision, validate)
from turbotab.core.methods import smcfcs  # importing it registers the check at the answer
from turbotab.core.methods.imputation import ImputationRefused
from turbotab.core.methods.smcfcs import Variable, impute
from turbotab.core.tests.stage_harness import Ingested
from turbotab.core.tests.test_imputation_width import (FIT, KCAL_ONCE, LEAD, ONCE, ZEROS,
                                                       ZEROS_WAY, _fit_table, _kcal_once, _more,
                                                       _surveyed, ctx_of, numbers, opening, wide,
                                                       wide_state)

NO_LOG = "Adjust for energy without the log"
COMPLETE = "Complete cases, with their assumption stated"
ZEROS_CODE = "zeros_cannot_be_logged"  # a recorded zero a log would meet


def _labels(refused: Any) -> list[str]:
    return [e["label"] for e in refused.exits]


# ── a narrow model: what the width rule never reached ────────────────────────


def _narrow(folder: Any, q_rows: int) -> tuple[pd.DataFrame, Any, ProjectState]:
    """300 rows: `x` (what is studied, about 10% blank), five complete covariates and a covariate
    `q` recorded on its first ``q_rows`` rows (amounts), under multiple imputation: 8 model columns
    (`x`, the five, `q` and the outcome's term), which the width rule draws whatever its rows."""
    rng = np.random.default_rng(4)
    n = 300
    x = rng.normal(size=n)
    x[rng.random(n) < 0.1] = np.nan
    cols = {f"c{j}": rng.normal(size=n) for j in range(1, 6)}
    q = np.full(n, np.nan)
    q[:q_rows] = rng.normal(size=q_rows)
    table = pd.DataFrame({"id": np.arange(n), "y": rng.normal(size=n), "x": x, **cols, "q": q})
    path = folder / "narrow.csv"
    table.to_csv(path, index=False)
    roles = {"id": "identifier", "x": "exposure", **{c: "covariate" for c in cols},
             "q": "covariate"}
    state = wide_state(roles=roles, missing=MissingSpec(strategy="multiple_imputation"), **FIT)
    return table, path, state


def test_on_a_narrow_model_a_column_recorded_once_is_refused_at_the_answer_as_the_fit_refuses_it(
        tmp_path):
    """`q` recorded on one of 300 rows of a model 8 columns wide, which the width rule draws: the
    sampler refuses that `q` has fewer than two values, so the lock and the missing-values answer
    are refused in those words, with the one way the fit offers (leaving `q` out, any column but
    the exposure), which the record accepts; complete cases keep one row, which it refuses. The
    fit says the same, with the same way."""
    table, path, state = _narrow(tmp_path, 1)
    assert int(table["q"].notna().sum()) == 1
    ingested = Ingested(path, tmp_path / "answer")
    ctx = ctx_of(state, ingested)
    found = smcfcs.imputation_at(state, ctx)
    assert found.sized.width == 1 + 5 + 1 + 1 and not found.sized.refused()
    with pytest.raises(Refusal) as answered:
        validate({"kind": "lock_plan"}, ctx)
    assert answered.value.code == smcfcs.VALUES_REFUSAL
    assert answered.value.message == LEAD + ONCE
    assert _labels(answered.value) == ["Leave `q` out"]
    left = answered.value.exits[0]["decision"]
    assert left["strategy"] == "multiple_imputation" and left["drop_columns"] == ["q"]
    validate(left, ctx)  # taken, it is accepted
    before = state.model_copy(update={"missing": None})
    with pytest.raises(Refusal) as refused:
        validate(SetMissing(strategy="multiple_imputation"), ctx_of(before, ingested))
    assert refused.value.message == LEAD + ONCE and _labels(refused.value) == ["Leave `q` out"]
    said = _fit_table(path, tmp_path / "project", state)
    assert said["caption"] == f"No coefficients: {LEAD}{ONCE}"
    assert [e["label"] for e in said["exits"]] == ["Leave `q` out"]
    assert said["exits"][0]["decision"]["drop_columns"] == ["q"]


def test_on_a_narrow_model_a_column_recorded_on_280_rows_is_accepted_and_drawn(tmp_path):
    """The guard above the other way: `q` recorded on 280 of the 300 rows, the lock is accepted
    and the fit pools the copies drawn with `q` in them."""
    _, path, state = _narrow(tmp_path, 280)
    validate({"kind": "lock_plan"}, ctx_of(state, Ingested(path, tmp_path / "answer")))
    said = _fit_table(path, tmp_path / "project", state)
    assert said["refused"] is None and said["caption"].startswith("Multiple imputation, m = ")
    assert "q" in said["missing"]["variables"]


def _fat_narrow(folder: Any, method: str, zeros: int) -> tuple[pd.DataFrame, Any, ProjectState]:
    """150 rows: the exposure `fat` (about 10% blank, zero on ``zeros`` rows), total energy `kcal`
    and five covariates, adjusted for energy (a density, or the log residual method), under
    multiple imputation: the analysis takes `fat`'s log, on a model the width rule draws."""
    rng = np.random.default_rng(3)
    n = 150
    fat = np.exp(rng.normal(3, 0.3, n))
    fat[:zeros] = 0.0
    fat[zeros:zeros + 15] = np.nan
    cols = {f"c{j}": rng.normal(size=n) for j in range(1, 6)}
    table = pd.DataFrame({"id": np.arange(n), "y": rng.normal(size=n), "fat": fat,
                          "kcal": np.exp(rng.normal(7.5, 0.2, n)), **cols})
    path = folder / "fat_narrow.csv"
    table.to_csv(path, index=False)
    roles = {"id": "identifier", "fat": "exposure", "kcal": "energy",
             **{c: "covariate" for c in cols}}
    adjustment = EnergyAdjustment(method=method, energy_column="kcal", nutrients=["fat"],
                                  log_transform=method == "residual")
    state = wide_state(roles=roles, energy_adjustment=adjustment,
                       missing=MissingSpec(strategy="multiple_imputation"), **FIT)
    return table, path, state


def test_on_a_narrow_model_a_logged_zero_is_refused_at_the_answer_as_the_fit_refuses_it(tmp_path):
    """`fat` as a density, zero on one of 150 rows (pandas) on a narrow model: the lock is refused
    with the sampler's words for the zeros and its code, with the one way the fit offers (how the
    zeros are read), and the fit says the same. With no zero the lock is accepted and the fit
    pools the copies."""
    table, path, state = _fat_narrow(tmp_path, "density", 1)
    assert int((table["fat"] <= 0).sum()) == 1
    ctx = ctx_of(state, Ingested(path, tmp_path / "answer"))
    assert not smcfcs.imputation_at(state, ctx).sized.refused()
    with pytest.raises(Refusal) as answered:
        validate({"kind": "lock_plan"}, ctx)
    assert answered.value.code == ZEROS_CODE
    assert answered.value.message == LEAD + ZEROS.format(k="1 row")
    assert [(e["label"], e["decision"]) for e in answered.value.exits] == [(ZEROS_WAY, None)]
    said = _fit_table(path, tmp_path / "project", state)
    assert said["caption"] == f"No coefficients: {answered.value.message}"
    assert [(e["label"], e["decision"]) for e in said["exits"]] == [(ZEROS_WAY, None)]
    (tmp_path / "clean").mkdir()
    _, clean, kept = _fat_narrow(tmp_path / "clean", "density", 0)
    validate({"kind": "lock_plan"}, ctx_of(kept, Ingested(clean, tmp_path / "clean" / "answer")))
    pooled = _fit_table(clean, tmp_path / "clean" / "project", kept)
    assert pooled["refused"] is None and pooled["caption"].startswith("Multiple imputation, m = ")


def test_under_the_log_residual_on_a_narrow_model_the_answer_offers_the_model_without_the_log(
        tmp_path):
    """Under the log residual method `fat`'s zero on a narrow model: the lock is refused with the
    zeros' words, and the way without the log, which the record accepts here (its plan logs
    nothing, and the width rule draws it), is offered first as the fit's table words it, with
    the energy adjustment recorded but its log; then how the zeros are read."""
    _, path, state = _fat_narrow(tmp_path, "residual", 1)
    ctx = ctx_of(state, Ingested(path, tmp_path / "answer"))
    with pytest.raises(Refusal) as answered:
        validate({"kind": "lock_plan"}, ctx)
    assert answered.value.message == LEAD + ZEROS.format(k="1 row")
    assert _labels(answered.value) == [NO_LOG, ZEROS_WAY]
    without = answered.value.exits[0]["decision"]
    assert without == parse_decision({"kind": "set_energy_adjustment",
                                      **state.energy_adjustment.model_dump(),
                                      "log_transform": False}).model_dump(mode="json")
    validate(without, ctx)  # taken, it is accepted


# ── above 80 columns: what the width rule does not hold ──────────────────────


def _broad(folder: Any, extra: dict[str, Any], rules: list[Any] | None = None, n: int = 400
           ) -> tuple[pd.DataFrame, Any, ProjectState]:
    """``n`` rows: `x` (what is studied, about 10% blank) beside 86 complete numbers, the outcome
    and the covariates ``extra``, under multiple imputation: 88 model columns and the extra ones,
    `x`'s model with its rows (about 360 recorded on 88 parameters, 4.1 each)."""
    rng = np.random.default_rng(17)
    cols = numbers(n, 86, rng)
    x = rng.normal(size=n)
    x[rng.random(n) < 0.1] = np.nan
    table = pd.DataFrame({"id": np.arange(n), "y": rng.normal(size=n), "x": x, **cols, **extra})
    path = folder / "broad.csv"
    table.to_csv(path, index=False)
    roles = {"id": "identifier", "x": "exposure", **{c: "covariate" for c in cols},
             **{c: "covariate" for c in extra}}
    state = wide_state(roles=roles, exclusions=list(rules or []),
                       missing=MissingSpec(strategy="multiple_imputation"), **FIT)
    return table, path, state


@pytest.mark.slow
def test_above_80_columns_a_0_1_column_recorded_once_is_refused_at_the_answer_as_at_the_fit(
        tmp_path):
    """A 0/1 covariate `flag` recorded once beside 86 numbers (89 columns): the width rule does not
    hold a model of one recorded level, so it does not refuse the plan, but the sampler refuses
    that `flag` has fewer than two values. The lock is refused in those words with the fit's way
    (leaving `flag` out), which the record accepts, and the fit says the same. Recorded on two
    rows, of one level, the sampler draws it and the lock is accepted."""
    flag = np.full(400, np.nan)
    flag[5] = 1.0
    table, path, state = _broad(tmp_path, {"flag": flag})
    ctx = ctx_of(state, Ingested(path, tmp_path / "answer"))
    found = smcfcs.imputation_at(state, ctx)
    assert found.sized.width == 89 and not found.sized.refused()
    once = "`flag` has fewer than two observed values, so it cannot be imputed."
    with pytest.raises(Refusal) as answered:
        validate({"kind": "lock_plan"}, ctx)
    assert answered.value.message == LEAD + once
    assert _labels(answered.value) == ["Leave `flag` out"]
    validate(answered.value.exits[0]["decision"], ctx)  # taken, it is accepted
    said = _fit_table(path, tmp_path / "project", state)
    assert said["caption"] == f"No coefficients: {LEAD}{once}"
    assert [e["label"] for e in said["exits"]] == ["Leave `flag` out"]
    twice = flag.copy()
    twice[6] = 1.0
    (tmp_path / "twice").mkdir()
    _, path2, state2 = _broad(tmp_path / "twice", {"flag": twice})
    ctx2 = ctx_of(state2, Ingested(path2, tmp_path / "twice" / "answer"))
    validate({"kind": "lock_plan"}, ctx2)
    plan = smcfcs.imputation_at(state2, ctx2).plan
    impute(plan.data, plan.variables, kinds=plan.kinds, m=1, iterations=1)  # drawn


@pytest.mark.slow
def test_above_80_columns_a_text_column_recorded_on_no_analyzed_row_is_refused_at_the_answer(
        tmp_path):
    """A text covariate `site` recorded only on the last 50 of 400 rows, which an eligibility rule
    leaves out: the chain would start its draws from no recorded value, which the sampler refuses
    after the width rule (88 columns drawn). The lock is refused in the chain's words, with the
    fit's way (leaving `site` out), and the fit says the same. Recorded on one analyzed row too,
    the chain draws its one level and the lock is accepted."""
    rule = ExclusionRule(column="id", high=349.0, reason="the last 50 rows")
    site = np.full(400, None, dtype=object)
    site[-50:] = np.random.default_rng(2).choice(["n", "s", "w"], 50)
    table, path, state = _broad(tmp_path, {"site": site}, [rule])
    assert int(table.loc[table["id"] <= 349, "site"].notna().sum()) == 0
    ctx = ctx_of(state, Ingested(path, tmp_path / "answer"))
    assert not smcfcs.imputation_at(state, ctx).sized.refused()
    none = "`site` has no recorded value to start its imputations from."
    with pytest.raises(Refusal) as answered:
        validate({"kind": "lock_plan"}, ctx)
    assert answered.value.code == smcfcs.VALUES_REFUSAL
    assert answered.value.message == LEAD + none
    assert _labels(answered.value) == ["Leave `site` out"]
    validate(answered.value.exits[0]["decision"], ctx)  # taken, it is accepted
    said = _fit_table(path, tmp_path / "project", state)
    assert said["caption"] == f"No coefficients: {LEAD}{none}"
    assert [e["label"] for e in said["exits"]] == ["Leave `site` out"]
    once = site.copy()
    once[0] = "n"
    (tmp_path / "once").mkdir()
    _, path2, state2 = _broad(tmp_path / "once", {"site": once}, [rule])
    ctx2 = ctx_of(state2, Ingested(path2, tmp_path / "once" / "answer"))
    validate({"kind": "lock_plan"}, ctx2)
    plan = smcfcs.imputation_at(state2, ctx2).plan
    impute(plan.data, plan.variables, kinds=plan.kinds, m=1, iterations=1)  # drawn


def test_the_sampler_refuses_a_category_with_no_recorded_value_before_its_chain_starts():
    """The sampler's own order: a text column the chain draws with no recorded value is refused
    before any chain starts (:func:`smcfcs.refusal_before_drawing`), in the chain's words, after
    the width (on 150 of :func:`wide`'s rows the width rule refuses first), and a number with none
    is refused as having fewer than two values; one recorded value is drawn as its level."""
    rng = np.random.default_rng(8)
    n = 120
    frame = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n),
                          "site": np.full(n, None, dtype=object), "__outcome": rng.normal(size=n)})
    frame.loc[::7, "x"] = np.nan
    variables = [Variable("x", "numeric"), Variable("site", "categorical")]
    found = smcfcs.refusal_before_drawing(frame, variables)
    assert str(found) == "`site` has no recorded value to start its imputations from."
    assert found.columns == ["site"] and found.width_refused is False
    with pytest.raises(ImputationRefused) as refused:
        impute(frame, variables, m=1, iterations=1)
    assert str(refused.value) == str(found)
    numeric = frame.assign(site=np.nan)
    assert str(smcfcs.refusal_before_drawing(numeric, [variables[0], Variable("site", "numeric")])
               ) == "`site` has fewer than two observed values, so it cannot be imputed."
    frame.loc[0, "site"] = "n"
    assert smcfcs.refusal_before_drawing(frame, variables) is None
    drawn = impute(frame, variables, m=1, iterations=1)
    assert set(drawn.frames[0]["site"]) == {"n"}


# ── the energy model's columns are never left out ────────────────────────────


def _kcal_narrow(folder: Any, adjusted: bool, kcal_once: bool = True
                 ) -> tuple[pd.DataFrame, Any, ProjectState]:
    """probe_p1's table: 300 rows, `x` (the exposure, about 10% blank), five covariates and total
    energy `kcal` (the energy role), recorded once or on every row; with ``adjusted`` the standard
    energy model names `kcal`, under multiple imputation on a model the width rule draws."""
    rng = np.random.default_rng(4)
    n = 300
    x = np.exp(rng.normal(3, 0.3, size=n))
    x[rng.random(n) < 0.1] = np.nan
    cols = {f"c{j}": rng.normal(size=n) for j in range(1, 6)}
    path = folder / "kcal.csv"
    table = pd.DataFrame({"id": np.arange(n), "y": rng.normal(size=n), "x": x, **cols})
    if kcal_once:
        table = _kcal_once(table, path)
    else:
        table = table.assign(kcal=np.exp(rng.normal(7.5, 0.2, n)))
        table.to_csv(path, index=False)
    roles = {"id": "identifier", "x": "exposure", **{c: "covariate" for c in cols},
             "kcal": "energy"}
    extra = ({"energy_adjustment": EnergyAdjustment(method="standard", energy_column="kcal",
                                                    nutrients=["x"])} if adjusted else {})
    state = wide_state(roles=roles, missing=MissingSpec(strategy="multiple_imputation"), **FIT,
                       **extra)
    return table, path, state


def test_an_energy_column_the_energy_model_names_is_never_offered_to_leave_out(tmp_path):
    """`kcal`, total energy in the standard energy model, recorded on one of 300 rows of a narrow
    model: the lock is refused because `kcal` has fewer than two values, and no way forward
    leaves it out (the design could not compute the standard model without it: on the base the
    fit offered it, and taking it raised "The energy column kcal is not in this table"); complete
    cases keep one row, which the record refuses. The fit says the same with no way forward."""
    from turbotab.core.stages.modeling import _leavable

    _, path, state = _kcal_narrow(tmp_path, adjusted=True)
    ctx = ctx_of(state, Ingested(path, tmp_path / "answer"))
    assert _leavable(state, ["kcal", "c1", "x"]) == ["c1"]
    assert "kcal" not in smcfcs.imputation_at(state, ctx).leavable
    with pytest.raises(Refusal) as answered:
        validate({"kind": "lock_plan"}, ctx)
    assert answered.value.message == LEAD + KCAL_ONCE
    assert answered.value.exits == []
    said = _fit_table(path, tmp_path / "project", state)
    assert said["caption"] == f"No coefficients: {LEAD}{KCAL_ONCE}"
    assert said["exits"] == []
    plain = state.model_copy(update={"energy_adjustment": None})
    assert _leavable(plain, ["kcal", "c1", "x"]) == ["kcal", "c1"]


LEFT_OUT = ("`{c}` is part of the energy adjustment (standard (multivariate) model), which cannot "
            "be computed without it, so it cannot be left out of the predictors.")


def test_a_missing_values_answer_leaving_out_a_column_the_energy_model_needs_is_refused(tmp_path):
    """Under the standard energy model on `kcal` and `x`, a missing-values answer whose columns
    left out name `kcal` (or the nutrient `x`) is refused, with the same answer leaving out only
    the rest, which the record accepts; one leaving out a covariate is accepted, and with no
    energy model, leaving out `kcal` is accepted."""
    _, path, state = _kcal_narrow(tmp_path, adjusted=True, kcal_once=False)
    state = state.model_copy(update={"missing": None})
    ctx = ctx_of(state, Ingested(path, tmp_path / "answer"))
    with pytest.raises(Refusal) as refused:
        validate(SetMissing(strategy="complete_case", drop_columns=["kcal"]), ctx)
    assert refused.value.code == smcfcs.ENERGY_LEFT_OUT
    assert refused.value.message == LEFT_OUT.format(c="kcal")
    assert _labels(refused.value) == ["Keep every predictor"]
    assert refused.value.exits[0]["decision"]["drop_columns"] == []
    with pytest.raises(Refusal) as refused:
        validate(SetMissing(strategy="complete_case", drop_columns=["c1", "x"]), ctx)
    assert refused.value.message == LEFT_OUT.format(c="x")
    assert _labels(refused.value) == ["Leave out only `c1`"]
    validate(refused.value.exits[0]["decision"], ctx)
    validate(SetMissing(strategy="complete_case", drop_columns=["c1"]), ctx)
    plain = state.model_copy(update={"energy_adjustment": None})
    validate(SetMissing(strategy="complete_case", drop_columns=["kcal"]), ctx_of(
        plain, Ingested(path, tmp_path / "plain")))


def test_the_log_residual_s_zero_is_a_refusal_in_words_with_the_model_without_the_log(tmp_path):
    """The log residual on a `fat` recorded at zero: the energy step refuses in words (a stage's
    refusal, its error text the words it raised before), offering the same model without the log,
    made the recorded answer by the engine; the design stage then serves it as a refusal where it
    raised an untyped error. With no zero it fits."""
    from turbotab.core.graph import StageRefusal
    from turbotab.core.jobs import failure_of
    from turbotab.core.methods.energy import EnergyAdjuster
    from turbotab.core.stages.modeling import design_stage
    from turbotab.core.tests import modeling_fixtures as mf

    rng = np.random.default_rng(5)
    n = 120
    frame = pd.DataFrame({"fat": np.exp(rng.normal(3, 0.3, n)),
                          "kcal": np.exp(rng.normal(7.5, 0.2, n)), "c1": rng.normal(size=n)})
    frame.loc[3, "fat"] = 0.0
    words = ("log_transform needs positive values, and fat is zero or negative in 1 of the "
             "fitting rows. Adding a small constant to zeros changes the answer and would have to "
             "be reported; adjust fat without the log, or decide how its zeros are handled first.")
    step = EnergyAdjuster("residual", "kcal", ["fat"], log_transform=True)
    with pytest.raises(StageRefusal) as refused:
        step.fit(frame)
    assert refused.value.code == ZEROS_CODE and str(refused.value) == words
    assert _labels(refused.value) == [NO_LOG]
    recorded = EnergyAdjustment(method="residual", energy_column="kcal", nutrients=["fat"],
                                log_transform=True, strata="c1")
    made = refused.value.on_answers(ProjectState(energy_adjustment=recorded)).exits[0]["decision"]
    assert made == {"kind": "set_energy_adjustment", **recorded.model_dump(mode="json"),
                    "log_transform": False}
    EnergyAdjuster("residual", "kcal", ["fat"], log_transform=True).fit(
        frame.assign(fat=frame["fat"].replace(0.0, 1.0)))
    table = frame.assign(y=rng.normal(size=n))
    roles = {"fat": "exposure", "kcal": "energy", "c1": "covariate"}
    st = mf.state(roles=roles, role_confirmations=dict(roles), target="y", task="regression",
                  models=["linear"], purpose="inference", lens=["clinical"],
                  missing=MissingSpec(strategy="complete_case"),
                  energy_adjustment=EnergyAdjustment(method="residual", energy_column="kcal",
                                                     nutrients=["fat"], log_transform=True))
    paths = mf.ingest_frame(table, tmp_path)
    split = mf.split_bundle(np.arange(n), holdout=0.0, seed=0)
    with pytest.raises(StageRefusal) as at_design:
        design_stage(mf.context(st, {"split": split, "target_info": mf.target_info(
            "regression", "y")}, paths))
    text, refusal = failure_of(at_design.value)
    assert text == f"ValueError: {words}"
    assert refusal["code"] == ZEROS_CODE and [e["label"] for e in refusal["exits"]] == [NO_LOG]


# ── a roles answer's ways forward, against the state after it ────────────────


def test_a_roles_answer_s_ways_forward_are_those_the_record_accepts_with_the_answer_recorded(
        tmp_path):
    """`q` recorded once beside :func:`_broad`'s 86 numbers, while no role holds it: the roles
    answer making `q` a covariate is refused (`q` has fewer than two values). On the base it
    offered complete cases, accepted now but leaving 1 row once `q` is a covariate, which the row
    floor then refuses; leaving `q` out is accepted with the answer recorded but refused now (`q`
    is no predictor yet). So neither is offered."""
    q = np.full(400, np.nan)
    q[0] = 1.5
    _, path, state = _broad(tmp_path, {"q": q})
    roles = dict(state.roles)
    before = state.model_copy(update={"roles": {c: r for c, r in roles.items() if c != "q"}})
    ctx = ctx_of(before, Ingested(path, tmp_path / "answer"))
    validate(SetMissing(strategy="complete_case"), ctx)  # now, complete cases are accepted
    with pytest.raises(Refusal) as refused:
        validate(SetRoles(roles=roles), ctx)
    assert refused.value.message == LEAD + ONCE
    assert refused.value.exits == []
    after = ctx_of(state, Ingested(path, tmp_path / "after"))
    with pytest.raises(Refusal) as floor:
        validate(SetMissing(strategy="complete_case"), after)
    assert floor.value.code == "too_few_rows"


def test_a_roles_answer_s_complete_cases_are_offered_where_accepted_now_and_after_it(tmp_path):
    """The guard above the other way: on 150 of :func:`wide`'s rows with 79 numbers and `smoke`
    out by their roles, the roles answer bringing them back is refused by the width rule, and
    complete cases, accepted now and with the answer recorded (about 128 rows), are offered."""
    from turbotab.core.tests.test_imputation_width import WIDE_ROLES

    table = wide(150)
    path = tmp_path / "wide.csv"
    table.to_csv(path, index=False)
    out = {**{f"c{j:02d}": "excluded" for j in range(6, 85)}, "smoke": "excluded"}
    narrow = wide_state(roles={**WIDE_ROLES, **out},
                        missing=MissingSpec(strategy="multiple_imputation"))
    with pytest.raises(Refusal) as refused:
        validate(SetRoles(roles=dict(WIDE_ROLES)), ctx_of(narrow, Ingested(path, tmp_path / "a")))
    assert refused.value.code == smcfcs.WIDTH_REFUSAL
    assert _labels(refused.value) == [COMPLETE]


# ── the width refusal's own way forward, against the record ──────────────────


def _surveyed_project(folder: Any) -> tuple[ProjectState, Ingested]:
    """70 strata of two rows, two PSUs each, a weight, under the population answer: the design's
    terms are most of the imputation model's 81 columns, and the sample answer is offered."""
    from turbotab.core.decisions import SurveySpec

    rng = np.random.default_rng(8)
    n = 140
    x = rng.normal(size=n)
    x[::10] = np.nan
    table = pd.DataFrame({"y": rng.normal(size=n), "x": x, **numbers(n, 8, rng),
                          "strat": np.repeat(np.arange(101, 171), 2), "psu": np.tile([1, 2], 70),
                          "wt": rng.uniform(1000, 9000, n)})
    path = folder / "surveyed.csv"
    table.to_csv(path, index=False)
    roles = {"x": "exposure", **{f"c{j:02d}": "covariate" for j in range(1, 9)},
             "strat": "design", "psu": "design", "wt": "design"}
    state = ProjectState(target="y", task="regression", purpose="inference",
                         grain={"grain": "one_row_per_unit"}, roles=roles, exclusions=[],
                         missing=MissingSpec(strategy="multiple_imputation"),
                         survey=SurveySpec(estimand="population", weight="wt", strata="strat",
                                           psu="psu"))
    return state, Ingested(path, folder)


def test_the_sample_answer_is_offered_only_where_the_record_accepts_it(tmp_path, monkeypatch):
    """The width refusal's own way forward (the sample answer) is asked of the record as the
    missing-values answers are: offered where it accepts it, and not where it refuses it (the
    record's check made to refuse every survey answer; on the base it was offered unasked)."""
    from turbotab.core import row_floor
    from turbotab.core.models.survey import SAMPLE_EXIT

    state, ingested = _surveyed_project(tmp_path)
    with pytest.raises(Refusal) as offered:
        validate({"kind": "lock_plan"}, ctx_of(state, ingested))
    assert _labels(offered.value) == [COMPLETE, SAMPLE_EXIT]
    real = row_floor.accepted_exit
    monkeypatch.setattr(row_floor, "accepted_exit", lambda d, ctx: None if row_floor._dump(d).get(
        "kind") == "set_survey" else real(d, ctx))
    with pytest.raises(Refusal) as refused:
        validate({"kind": "lock_plan"}, ctx_of(state, ingested))
    assert refused.value.message == offered.value.message
    assert _labels(refused.value) == [COMPLETE]


# ── a check that cannot be made is said, not silent ──────────────────────────


@pytest.mark.parametrize("raised, logged", [(KeyError("a column"), True),
                                            (ValueError("a refusal in words"), False)])
def test_an_engine_defect_in_the_check_is_logged_and_a_refusal_in_words_is_not(
        tmp_path, monkeypatch, caplog, raised, logged):
    """Where foreseeing the fit's imputation fails, the answer is accepted as before the check,
    but an engine defect (a KeyError) is logged with its traceback, where a refusal in words on
    the way (a ValueError: a reading asked, a stage's refusal) is the fit's to say."""
    _, path, state = _narrow(tmp_path, 280)

    def fails(*_: Any) -> Any:
        raise raised

    monkeypatch.setattr(smcfcs, "_imputation_at", fails)
    with caplog.at_level(logging.ERROR, logger=smcfcs.__name__):
        validate({"kind": "lock_plan"}, ctx_of(state, Ingested(path, tmp_path / "answer")))
    records = [r for r in caplog.records if r.name == smcfcs.__name__]
    assert bool(records) is logged
    if logged:
        assert records[0].exc_info is not None and records[0].exc_info[0] is KeyError


# ── a unit-level column every unit records ───────────────────────────────────


def test_a_unit_level_column_every_unit_records_is_not_held_to_the_rule():
    """Clustered rows (50 units of two) and a unit-level `u` recorded on each unit's first row:
    every unit records it, so the chain fits no model of it (each blank holds its unit's value),
    and the width rule holds none, though the 87 columns are wider than 80 and 50 units would be
    short of 86 parameters. Where one unit records it on no row, its model is held to the rule on
    the 49 units that record it, and refused."""
    rng = np.random.default_rng(31)
    units = np.repeat(np.arange(50), 2)
    n = len(units)
    unit_u = rng.normal(size=50)
    u = np.repeat(unit_u, 2)
    u[1::2] = np.nan
    frame = pd.DataFrame({**numbers(n, 85, rng), "u": u, "__outcome": rng.normal(size=n)})
    variables = [Variable("u", "numeric", unit_level=True)]
    sized = smcfcs.model_width(frame, variables, units=units)
    assert sized.width == 85 + 1 + 1 and sized.models == []
    drawn = impute(frame, variables, units=units, m=2, iterations=1)
    for copy in drawn.frames:
        np.testing.assert_array_equal(copy["u"].to_numpy(), np.repeat(unit_u, 2))
    frame.loc[0, "u"] = np.nan  # unit 0 now records it on no row
    [model] = smcfcs.model_width(frame, variables, units=units).models
    assert (model.rows, model.parameters, model.units) == (49, 86, True)
    with pytest.raises(ImputationRefused):
        impute(frame, variables, units=units, m=1, iterations=1)


# ── the docs and tests items ─────────────────────────────────────────────────


@pytest.mark.parametrize("value, said", [(0.00995, "0.010"), (0.00999, "0.010"),
                                         (0.0099, "0.0099"), (1 / 90, "0.011"),
                                         (0.0496, "0.050"), (0.000999, "0.0010")])
def test_a_figure_below_0_05_is_said_to_two_significant_digits(value, said):
    """Two significant digits, also where rounding reaches the next power of ten (0.00996 to
    0.010, where the base said 0.0100): the decimal expansions are the construction."""
    assert smcfcs._below(value, 2) == said


def test_the_design_said_to_be_most_of_the_width_is_more_than_half_of_it_not_half():
    """The boundary of "most of the width" (``2 · design > width``), both sides: 45 strata of two
    rows and 36 more numbers give the design 44 + 1 + 1 = 46 of 92 columns, exactly half, and the
    refusal says nothing of the design and offers no sample answer; 46 strata give it 47 of 93,
    and the refusal counts them and offers the sample answer (without them, 46 columns are
    drawn)."""
    from turbotab.core.methods.missing import WITHOUT_THE_DESIGN, sample_way

    half = _more(_surveyed(45), 36)
    with pytest.raises(ImputationRefused) as refused:
        impute(half, [Variable("x", "numeric")], m=1, iterations=1)
    assert str(refused.value).startswith(opening(92))
    assert "survey design" not in str(refused.value)
    assert getattr(refused.value, "exits", None) is None
    most = _more(_surveyed(46), 36)
    with pytest.raises(ImputationRefused) as refused:
        impute(most, [Variable("x", "numeric")], m=1, iterations=1)
    assert str(refused.value).startswith(opening(93))
    assert ("Of the imputation model's 93 columns, 47 are the survey design's: 45 indicators of "
            f"its strata, one indicator of its PSUs within their strata and its weight. "
            f"{WITHOUT_THE_DESIGN}") in str(refused.value)
    assert refused.value.exits == [sample_way()]


@pytest.mark.parametrize("a, b", [(5, 95), (50, 50), (1, 2), (809, 1620)])
def test_the_effective_size_of_two_levels_is_3ab_over_n(a, b):
    """n − Σn³/n² for two levels of a and b rows is (n³ − a³ − b³)/n² = 3ab(a + b)/n² = 3ab/n."""
    from turbotab.core.models.ordinal import effective_rows

    assert effective_rows([a, b]) == pytest.approx(3 * a * b / (a + b), rel=1e-12)


# ── the scales stage refuses the items' copies as the fit refuses them ───────


def _scaled(folder: Any, q_rows: int) -> tuple[ProjectState, Any, Any, Any, Any]:
    """A reflective 8-item scale (about 2% of each item's answers blank), two covariates and a
    covariate `q` recorded on ``q_rows`` of 300 rows, corrected by regression calibration under
    multiple imputation: the state, its paths, the split, the target and the design stage's
    artifact."""
    from turbotab.core import decisions as d
    from turbotab.core.stages.modeling import design_stage
    from turbotab.core.tests import modeling_fixtures as mf
    from turbotab.core.tests.acceptance.scales_fixtures import linear_scale_table

    items = [f"sat_{j}" for j in range(1, 9)]
    frame = linear_scale_table(n=300, missing=0.02)
    q = np.full(300, np.nan)
    q[:q_rows] = np.random.default_rng(9).normal(size=q_rows)
    frame["q"] = q
    roles = {"age": "covariate", "bmi": "covariate", "q": "covariate",
             **{c: "covariate" for c in items}}
    st = mf.state(purpose="inference", target="sbp", task="regression", models=["linear"],
                  roles=roles, role_confirmations=dict(roles), lens=["clinical"],
                  split=d.SplitSpec(holdout=0.0, seed=7, folds=5), shape_confirmations={},
                  missing=MissingSpec(strategy="multiple_imputation"),
                  scales=[d.ScaleSpec(name="sat_score", items=items, reverse=["sat_3", "sat_6"],
                                      low=1, high=5, kind="reflective",
                                      correction="regression_calibration", n_boot=50)])
    paths = mf.ingest_frame(frame, folder)
    split = mf.split_bundle(np.arange(len(frame)), holdout=0.0, seed=7)
    ti = mf.target_info("regression", "sbp")
    design = design_stage(mf.context(st, {"split": split, "target_info": ti}, paths))
    return st, paths, split, ti, design


def test_the_scales_stage_refuses_the_items_copies_in_the_fit_s_words_with_its_ways(tmp_path):
    """`q` recorded once: the sampler refuses the copies the correction needs. The scales stage
    says why in the fit's table's words (its lead and the sampler's), where the base said "answer
    multiple imputation" to a plan that had; it offers the fit's way forward (leaving `q` out), the
    same decision, and its methods sentence says the coefficient was not corrected and why."""
    from turbotab.core.stages.modeling import fit_stage
    from turbotab.core.stages.scales import scales_stage
    from turbotab.core.tests import modeling_fixtures as mf

    st, paths, split, ti, design = _scaled(tmp_path, 1)
    inputs = {"design": design, "split": split, "target_info": ti}
    [scale] = scales_stage(mf.context(st, inputs, paths)).data["scales"]
    assert scale["not_corrected"] == LEAD + ONCE
    assert scale["correction"] is None and scale["imputation"] is None
    assert [e["label"] for e in scale["exits"]] == ["Leave `q` out"]
    assert scale["methods"].endswith(
        f"Its coefficient was not corrected for measurement error. {LEAD}{ONCE}")
    table = fit_stage(mf.context(st, inputs, paths)).data["models"][0]["inference"]
    assert table["caption"] == f"No coefficients: {LEAD}{ONCE}"
    assert [(e["label"], parse_decision(e["decision"])) for e in table["exits"]] == \
        [(e["label"], parse_decision(e["decision"])) for e in scale["exits"]]


def test_the_scales_stage_hands_the_sampler_the_fit_s_columns_and_draws_where_it_draws(
        tmp_path, monkeypatch):
    """The guard above the other way: `q` recorded on 280 rows, the items are imputed and no
    refusal is said; the sampler is handed the columns an exit may leave out that the fit's table
    hands it (every settled covariate here: none is studied, and no energy model), so a refusal
    would name what the fit's names. (The correction itself, tested in ``test_ms8_scales``, is
    skipped here.)"""
    from turbotab.core.methods import missing
    from turbotab.core.stages import scales as S
    from turbotab.core.tests import modeling_fixtures as mf

    st, paths, split, ti, design = _scaled(tmp_path, 280)
    handed: list[Any] = []
    real = missing.impute_for_inference

    def spy(*args: Any, **kwargs: Any) -> Any:
        handed.append(kwargs.get("leavable"))
        return real(*args, **kwargs)

    monkeypatch.setattr(missing, "impute_for_inference", spy)
    monkeypatch.setattr(S, "correct_together", lambda *a, **k: ({}, {}))
    [scale] = S.scales_stage(mf.context(st, {"design": design, "split": split,
                                             "target_info": ti}, paths)).data["scales"]
    assert scale["not_corrected"] is None and scale["exits"] == []
    assert scale["imputation"] is not None and scale["imputation"]["m"] >= 20
    assert sorted(handed[0]) == sorted(["age", "bmi", "q", *[f"sat_{j}" for j in range(1, 9)]])

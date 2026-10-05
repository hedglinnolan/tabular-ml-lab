"""FORM · 6 · an exposure with a mass at zero (non-consumers): non-consumers as their own category
beside a spline among consumers with knots at consumers' percentiles, or the consumers-only domain
recorded as an estimand change (MODELING_SEQUENCE §1 rows 2 and 5, §2; STROBE-nut nut-11 and
nut-14).

The fixture's ``fish_g`` is a food 60% of participants never eat (``form_fixtures.diet``). The
independent references: R's ``Hmisc::rcspline.eval`` on the consumers' values for the knots (and
Harrell's percentiles written out with NumPy), least squares with HC3 written out for the two Wald
tests on a design built here by hand (the consumer indicator, the intake, Harrell's basis), and the
participant flow's counts by NumPy.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from turbotab.core import decisions as d
from turbotab.core.methods import exposure_form as ef
from turbotab.core.tests import modeling_fixtures as mf
from turbotab.core.tests.acceptance import estimand_fixtures as est_f
from turbotab.core.tests.acceptance import form_fixtures as ff
from turbotab.core.tests.acceptance.r_reference import needs_r, run_r

ROLES = {"pid": "identifier", "age": "covariate", "sex": "covariate", "smoking": "covariate",
         "activity": "covariate", "fish_g": "exposure", "kcal": "excluded",
         "protein_g": "excluded"}
ANSWERS = {"age": est_f.CONFOUNDER, "sex": est_f.CONFOUNDER, "smoking": est_f.CONFOUNDER,
           "activity": est_f.PRECISION}


def _state(**slots: Any) -> d.ProjectState:
    base: dict[str, Any] = dict(
        roles=dict(ROLES), role_confirmations=dict(ROLES), target="glucose", task="regression",
        purpose="inference", models=["linear"], lens=["dietary"],
        split=d.SplitSpec(holdout=0.0, seed=0, folds=5),
        shape_confirmations={"code_or_count:smoking": "amount", "code_or_count:activity": "amount"},
        estimand=d.EstimandSpec(exposure="fish_g", measure="mean_difference"),
        adjustment=est_f.answers_for("fish_g", ANSWERS),
        grain=d.GrainSpec(grain="one_row_per_unit", id_column="pid"))
    return mf.state(**{**base, **slots})


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return ff.diet(n=900, seed=43)


def _info(frame: pd.DataFrame) -> dict:
    return {c: {"dtype": "numeric" if pd.api.types.is_numeric_dtype(frame[c]) else "text",
                "n_unique": int(frame[c].nunique())} for c in frame.columns}


def test_6_the_card_sees_the_mass_at_zero_and_leads_with_non_consumers_apart(frame):
    st = _state()
    card = ef.forms_card(st, frame, _info(frame), frame["glucose"], "regression")
    fish = next(n for n in card["needs"] if n["column"] == "fish_g")
    share = float(np.mean(frame["fish_g"].to_numpy() == 0))
    assert fish["zero_share"] == pytest.approx(round(share, 4), abs=1e-12) and share > 0.5
    assert fish["mass_at_zero"] is True
    assert [o["value"] for o in fish["options"]][:2] == ["zero_spline", "consumers_only"]
    assert fish["options"][1]["sound"] == ("Answers a different question: the effect among "
                                           "consumers, not in everyone.")
    assert fish["proposal"] == {"form": "zero_spline", "knots": 5, "knots_rule": "harrell"}


@pytest.fixture(scope="module")
def zero_run(frame, tmp_path_factory) -> dict[str, Any]:
    from turbotab.core.stages.modeling import design_stage, fit_stage

    st = _state(exposure_forms={"fish_g": d.ExposureFormSpec(form="zero_spline", knots=4)})
    folder = tmp_path_factory.mktemp("zero")
    paths = mf.ingest_frame(frame, folder)
    split = mf.split_bundle(np.arange(len(frame)), holdout=0.0)
    info = mf.target_info("regression", "glucose")
    design = design_stage(mf.context(st, {"split": split, "target_info": info}, paths))
    fit = fit_stage(mf.context(st, {"design": design, "split": split, "target_info": info}, paths))
    return {"model": fit.data["models"][0], "design": design.data}


def test_6_the_knots_are_at_consumers_percentiles_numpy(frame, zero_run):
    consumers = frame["fish_g"].to_numpy(float)
    consumers = consumers[consumers > 0]
    tests = {t["test"]: t for t in zero_run["model"]["exposure_tests"]}
    assert np.allclose(tests["overall"]["knots"], ff.harrell_knots(consumers, 4), rtol=0,
                       atol=1e-10)
    assert {"fish_g_consumer", "fish_g", "fish_g'", "fish_g''"} <= {
        r["feature"] for r in zero_run["model"]["coefficients"]}


@needs_r
def test_6_the_knots_are_rcspline_evals_on_the_consumers_values(frame, zero_run, tmp_path):
    found = run_r("""
suppressMessages(library(Hmisc))
d <- read.csv(frame_csv)
x <- d$fish_g[d$fish_g > 0]
out(list(knots = Hmisc::rcspline.eval(x, nk = 4, knots.only = TRUE)))
""", {"frame": frame[["fish_g"]]}, tmp_path)
    tests = {t["test"]: t for t in zero_run["model"]["exposure_tests"]}
    assert np.allclose(tests["overall"]["knots"], found["knots"], rtol=0, atol=1e-10)


def test_6_the_overall_and_consumers_nonlinearity_tests_agree_with_numpy(frame, zero_run):
    tests = {t["test"]: t for t in zero_run["model"]["exposure_tests"]}
    knots = np.asarray(tests["overall"]["knots"])
    x = frame["fish_g"].to_numpy(float)
    X = est_f.design_matrix(frame, ["age", "sex", "smoking", "activity"])
    X["fish_g_consumer"] = (x > 0).astype(float)
    X["fish_g"] = x
    from turbotab.core.tests.acceptance.test_form_3_5_tests_and_quintiles import _rcs_by_hand

    basis = _rcs_by_hand(x, knots)
    X["fish_g'"], X["fish_g''"] = basis[:, 0], basis[:, 1]
    assert np.allclose(basis[x == 0], 0.0)  # every spline term is zero at zero intake
    beta, cov = ff.ols_hc3(X.to_numpy(float), frame["glucose"].to_numpy(float))
    names = list(X.columns)
    df = len(frame) - X.shape[1]
    idx = [names.index(n) for n in ("fish_g_consumer", "fish_g", "fish_g'", "fish_g''")]
    F, p = ff.wald_f(beta, cov, idx, df)
    assert tests["overall"]["df_num"] == 4
    assert tests["overall"]["statistic"] == pytest.approx(F, rel=1e-8)
    assert tests["overall"]["p"] == pytest.approx(p, rel=1e-8, abs=1e-300)
    F_nl, p_nl = ff.wald_f(beta, cov, idx[2:], df)
    assert tests["nonlinear"]["df_num"] == 2
    assert tests["nonlinear"]["statistic"] == pytest.approx(F_nl, rel=1e-8)
    assert tests["nonlinear"]["caption"].startswith(
        "Wald test that the 2 nonlinear terms of `fish_g` among consumers are zero")


def test_6_the_record_states_non_consumers_as_the_reference():
    from turbotab.core.voice import sentence_for

    st = _state()
    decision = d.SetExposureForm(column="fish_g", form="zero_spline", knots=4)
    assert sentence_for(decision, st) == (
        "Non-consumers of `fish_g` (`fish_g` = 0) entered the models as their own category, the "
        "reference, beside a restricted cubic spline among consumers with `4` knots at the 5, 35, "
        "65 and 95 percentiles of consumers' values (`fish_g` above 0) in the rows each model was fit "
        "on (STROBE-nut nut-11); the test of association is the Wald test that every term is zero, "
        "and nonlinearity was tested by a Wald test that its nonlinear terms are zero, a "
        "non-significant result never refitting a straight line.")


class _Store:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame
        self.columns = list(frame.columns)

    def materialize(self, columns, rows=None):
        return self.frame[list(columns)]


def test_6_a_column_without_zeros_has_no_non_consumers_to_set_apart(frame):
    ctx = {"state": _state(), "columns": list(frame.columns), "column_info": _info(frame),
           "store": lambda: _Store(frame), "analyzed": lambda: None}
    with pytest.raises(d.Refusal) as refused:
        d.validate({"kind": "set_exposure_form", "column": "age", "form": "zero_spline"}, ctx)
    assert refused.value.code == "no_mass_at_zero"
    assert refused.value.message == ("`age` has no value at zero, so there are no non-consumers "
                                     "to set apart.")
    d.validate({"kind": "set_exposure_form", "column": "fish_g", "form": "zero_spline"}, ctx)


# ── the consumers-only domain: an estimand change ────────────────────────────


def test_6_the_consumers_only_domain_is_an_estimand_change_in_the_flow_and_the_caption(frame):
    from turbotab.core.decisions import DecisionRecord, fold
    from turbotab.core.estimand import caption
    from turbotab.core.stages.rows import cohort_flow, domain_of
    from turbotab.core.voice import sentence_for

    base = _state()
    decision = d.SetExposureForm(column="fish_g", form="spline", knots=4, domain="consumers")
    from datetime import datetime, timezone

    records = [DecisionRecord(id="a" * 32, seq=1, at=datetime.now(timezone.utc), decision=decision)]
    folded = fold(records)
    assert folded.form_domains == {"fish_g": "consumers"}
    st = base.model_copy(update={"exposure_forms": folded.exposure_forms,
                                 "form_domains": folded.form_domains})
    assert domain_of(st) == ["fish_g"]
    steps, kept = cohort_flow(frame.set_index(frame["pid"]), target="glucose", rules=[],
                              missing="complete_case", predictor_columns=[], domain=["fish_g"])
    step = next(s for s in steps if s["key"] == "domain:fish_g")
    zeros = int(np.sum(frame["fish_g"].to_numpy() == 0))
    assert step["dropped"] == zeros and step["n"] == len(frame) - zeros == len(kept)
    assert step["label"] == "`fish_g` above 0: consumers only"
    assert step["reason"] == ("a non-consumer of `fish_g` (`fish_g` = 0): the analysis is "
                              "restricted to consumers, an estimand change (STROBE-nut nut-14)")
    assert caption(st).startswith("The total effect of `fish_g` on `glucose` among consumers of "
                                  "`fish_g`, as a difference in the mean outcome per unit of "
                                  "`fish_g`")
    assert sentence_for(decision, base) == (
        "`fish_g` entered the models as a restricted cubic spline with `4` knots at the 5, 35, 65 "
        "and 95 percentiles of its values in the rows each model was fit on (Harrell's "
        "placement); the test of association is the Wald test that every term is zero, and "
        "nonlinearity was tested by a Wald test that its nonlinear terms are zero, a "
        "non-significant result never refitting a straight line; quintiles were reported beside "
        "it, their boundaries and reference stated, with the p for linear trend (customary) "
        "across quintile medians; the analysis was restricted to consumers of `fish_g` (`fish_g` "
        "above 0): an estimand change, the effect among consumers, not in the whole population "
        "(STROBE-nut (Lachat et al. 2016) nut-14).")
    # A form answer for another column, or one keeping everyone, leaves the flow alone.
    everyone = fold(records + [DecisionRecord(
        id="b" * 32, seq=2, at=datetime.now(timezone.utc),
        decision=d.SetExposureForm(column="fish_g", form="spline", knots=4))])
    assert everyone.form_domains is None


def test_6_the_consumers_only_domain_is_refused_under_prediction(frame):
    st = _state(purpose="prediction")
    with pytest.raises(d.Refusal) as refused:
        d.validate({"kind": "set_exposure_form", "column": "fish_g", "form": "spline",
                    "domain": "consumers"}, {"state": st, "columns": list(frame.columns)})
    assert refused.value.code == "domain_for_inference"
    assert refused.value.exits[0]["decision"]["form"] == "zero_spline"

"""E3 (SWARM_ENGINE_PLAN): which decisions mattered under Estimate, the ``mattered`` stage
(``turbotab/core/mattered.py``) over the real design and fit stages.

The fixture is the ESTIMAND cohort (``acceptance/estimand_fixtures.cohort``) with blanks in
``activity`` (an adjusted covariate) and ``bmi`` (the covariate of unknown timing the adjustment
answers leave out), and two exclusion rules: ``age`` within 30–75, and ``bmi`` at most 33 or not
recorded. Every expected number is statsmodels' (least squares with HC3 on t(n − p), logistic
regression with Wald intervals) on the rows and columns each specification's definition gives,
written out here from the table as ingested: never the engine's.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from turbotab.core import decisions as d
from turbotab.core import mattered as M
from turbotab.core.methods import spec_curve as S
from turbotab.core.tests import modeling_fixtures as mf
from turbotab.core.tests.acceptance import estimand_fixtures as ef

ADJUSTED = ["age", "sex", "smoking", "activity"]  # the answers' confounders and precision
RULES = [{"column": "age", "low": 30, "high": 75, "reason": "the study's age range"},
         {"column": "bmi", "high": 33, "reason": "an implausible BMI", "missing": "keep"}]
ORDER = ["exclusion:0", "exclusion:1", *[f"adjusted:{c}" for c in ADJUSTED], "left_out:bmi"]


def cohort() -> pd.DataFrame:
    frame = ef.cohort(n=800, seed=7)
    rng = np.random.default_rng(11)
    frame.loc[rng.choice(len(frame), 25, replace=False), "activity"] = np.nan
    frame.loc[rng.choice(len(frame), 30, replace=False), "bmi"] = np.nan
    return frame


def state(**slots: Any) -> d.ProjectState:
    base: dict[str, Any] = {"target": "glucose", "task": "regression",
                            "measure": "mean_difference", "plan_locked": True,
                            "exclusions": [d.ExclusionRule(**r) for r in RULES]}
    base.update(slots)
    return ef.state(**base)


class Run:
    """The table ingested, the participant flow, and the design, fit and mattered stages as a
    worker runs them (every analyzed row, no held-out rows: inference). ``grouped_by``: the
    column the split keeps whole, each row's unit its value there (row id = position)."""

    def __init__(self, frame: pd.DataFrame, folder: Path, st: d.ProjectState, *,
                 fit: Any = None, grouped_by: str | None = None):
        from turbotab.core.datastore import DataStore, ingest
        from turbotab.core.stages.modeling import design_stage, fit_stage
        from turbotab.core.stages.rows import compute_cohort

        (folder / "data").mkdir(parents=True, exist_ok=True)
        source = folder / "table.csv"
        frame.to_csv(source, index=False)
        parquet = folder / "data" / "raw.parquet"
        self.info = ingest(source, parquet).to_dict()
        self.table = pd.read_csv(source)  # the values the engine reads, for every expectation
        self.paths = {"source": str(source), "data": str(parquet)}
        self.state = st
        with DataStore(parquet, 1 << 30) as store:
            _, kept, _ = compute_cohort(store, st, self.info)
        self.kept = np.asarray(kept)
        units = (None if grouped_by is None
                 else self.table[grouped_by].to_numpy()[self.kept])
        split = mf.split_bundle(self.kept, holdout=0.0, seed=st.split.seed, groups=units,
                                grouped_by=grouped_by)
        base = {"ingest": self.info, "split": split, "target_info": mf.target_info(st.task,
                                                                                    st.target)}
        self.design = design_stage(mf.context(st, base, self.paths))
        self.fit = fit if fit is not None else fit_stage(
            mf.context(st, {**base, "design": self.design}, self.paths))
        self.inputs = {**base, "design": self.design, "fit": self.fit}

    def context(self, st: d.ProjectState | None = None) -> Any:
        return mf.context(st or self.state, self.inputs, self.paths)

    def mattered(self, st: d.ProjectState | None = None) -> dict[str, Any]:
        return M.mattered_stage(self.context(st))

    def reported(self, exposure: str = "fiber") -> dict[str, Any]:
        model = next(m for m in self.fit.data["models"] if m["family"] == "linear")
        return next(r for r in model["coefficients"] if r["feature"] == exposure)


def rows_of(table: pd.DataFrame, *, rules: tuple[int, ...], columns: list[str]) -> np.ndarray:
    """A specification's rows by its definition: the rules it applies, then every column it reads
    recorded."""
    keep = np.ones(len(table), dtype=bool)
    if 0 in rules:
        keep &= table["age"].between(30, 75).to_numpy()
    if 1 in rules:
        keep &= (table["bmi"].isna() | (table["bmi"] <= 33)).to_numpy()
    return keep & table[columns].notna().all(axis=1).to_numpy()


def statsmodels_fit(table: pd.DataFrame, *, outcome: str, rules: tuple[int, ...],
                    covariates: list[str], binary: bool = False) -> tuple[float, float, float, int]:
    keep = rows_of(table, rules=rules, columns=[outcome, "fiber", *covariates])
    part = table[keep]
    X = ef.design_matrix(part, ["fiber", *covariates])
    if binary:
        fit = sm.Logit((part[outcome] == "yes").astype(float), X).fit(disp=0)
    else:
        fit = sm.OLS(part[outcome].astype(float), X).fit(cov_type="HC3", use_t=True)
    low, high = fit.conf_int(alpha=0.05).loc["fiber"]
    return float(fit.params["fiber"]), float(low), float(high), int(keep.sum())


def departures() -> dict[str, tuple[tuple[int, ...], list[str]]]:
    """Each specification of the curve by its key: (the rules it applies, its covariates)."""
    def key(changed: str | None = None, pick: str = "") -> str:
        return "|".join(f"{c}={pick if c == changed else M.RECORDED}" for c in ORDER)

    out = {key(): ((0, 1), ADJUSTED)}
    out[key("exclusion:0", "dropped")] = ((1,), ADJUSTED)
    out[key("exclusion:1", "dropped")] = ((0,), ADJUSTED)
    for c in ADJUSTED:
        out[key(f"adjusted:{c}", "dropped")] = ((0, 1), [x for x in ADJUSTED if x != c])
    out[key("left_out:bmi", "added")] = ((0, 1), [*ADJUSTED, "bmi"])
    return out


@pytest.fixture(scope="module")
def numeric(tmp_path_factory) -> Run:
    return Run(cohort(), tmp_path_factory.mktemp("numeric"), state())


@pytest.fixture(scope="module")
def binary(tmp_path_factory) -> Run:
    return Run(cohort(), tmp_path_factory.mktemp("binary"),
               state(target="dm", task="binary", measure="odds_ratio", event="yes"))


# ── each specification against statsmodels ───────────────────────────────────


def test_each_specification_is_least_squares_with_hc3_as_statsmodels_fits_it(numeric):
    art = numeric.mattered()
    assert art["refused"] is None, art["refused"]
    curve = art["curve"]
    assert curve["crossing"] == "one_at_a_time" and curve["sealed"] is None
    assert [c["key"] for c in curve["choices"]] == ORDER
    got = {s["key"]: s for s in curve["specs"]}
    want = departures()
    assert set(got) == set(want) and curve["not_estimated"] == []
    for k, (rules, covariates) in want.items():
        est, low, high, n = statsmodels_fit(numeric.table, outcome="glucose", rules=rules,
                                            covariates=covariates)
        s = got[k]
        assert s["estimate"] == pytest.approx(est, rel=1e-9), k
        assert s["low"] == pytest.approx(low, rel=1e-9) and s["high"] == pytest.approx(high,
                                                                                         rel=1e-9)
        assert s["n"] == n and s["covariance"] == "HC3", k
    # The departures move the rows as their definitions say: the blanks of `activity` return when
    # it is dropped, those of `bmi` leave when it is added.
    primary = next(s for s in curve["specs"] if s["primary"])
    by = {k: got[k]["n"] for k in got}
    assert by[[k for k in got if "adjusted:activity=dropped" in k][0]] > primary["n"]
    assert by[[k for k in got if "left_out:bmi=added" in k][0]] < primary["n"]
    assert "one at a time" in art["methods"] and "No joint test" in art["methods"]
    # No specification's rows repeat here, so none is clustered and the sentence says none is.
    assert curve["clustered"] is False and "cluster-robust" not in art["methods"]


def test_each_logistic_specification_has_the_wald_interval_statsmodels_gives(binary):
    art = binary.mattered()
    assert art["refused"] is None, art["refused"]
    curve = art["curve"]
    # The table's `yes` and `no` are read as True and False, so the event is named `True`, as the
    # fit's own captions name it (``stages.modeling.outcome_levels``).
    assert curve["estimate_label"] == ("Difference in log-odds of `dm` being `True` per unit of "
                                       "`fiber`")
    got = {s["key"]: s for s in curve["specs"]}
    for k, (rules, covariates) in departures().items():
        est, low, high, n = statsmodels_fit(binary.table, outcome="dm", rules=rules,
                                            covariates=covariates, binary=True)
        s = got[k]
        assert s["estimate"] == pytest.approx(est, rel=1e-6), k
        assert s["low"] == pytest.approx(low, rel=1e-6) and s["high"] == pytest.approx(high,
                                                                                         rel=1e-6)
        assert s["n"] == n, k


def test_under_the_odds_ratio_the_primary_is_worded_as_the_log_of_the_reported_ratio(binary):
    """The fit reports the odds ratio exp(β); the curve follows β. Its sentence says the primary is
    the log of the reported odds ratio and quotes both, statsmodels' numbers rounded as shown."""
    art = binary.mattered()
    est, *_ = statsmodels_fit(binary.table, outcome="dm", rules=(0, 1), covariates=ADJUSTED,
                              binary=True)
    log_odds = f"{est:.3g}".replace("-", "−")
    assert (f"the primary, {log_odds}, is the log of the reported odds ratio, "
            f"{math.exp(est):.3g}") in art["says"]
    assert "is the reported estimate" not in art["says"]
    assert art["reported"]["ratio"] == pytest.approx(math.exp(est), rel=1e-6)
    assert "the log of its odds ratio" in art["methods"]


def test_the_primary_is_the_reported_estimate_and_the_fit_is_unchanged(numeric):
    before = numeric.reported()
    art = numeric.mattered()
    primary = next(s for s in art["curve"]["specs"] if s["primary"])
    reported = numeric.reported()
    assert reported == before  # computing the curve moves nothing the fit reports
    assert art["reported"]["estimate"] == reported["estimate"]
    assert art["reported"]["ci_low"] == reported["ci_low"]
    assert art["reported"]["ci_high"] == reported["ci_high"]
    assert primary["estimate"] == pytest.approx(reported["estimate"], rel=1e-12)
    assert primary["low"] == pytest.approx(reported["ci_low"], rel=1e-12)
    assert primary["n"] == len(numeric.kept)  # the participant flow's own rows
    assert "is the reported estimate" in art["says"]
    assert art["reported"]["ratio"] is None and "odds ratio" not in art["methods"]


# ── the lock ─────────────────────────────────────────────────────────────────


def test_before_the_lock_nothing_is_fitted_and_the_curve_says_when_it_opens(numeric, monkeypatch):
    calls: list[Any] = []
    real = S.linear_fitter
    monkeypatch.setattr(S, "linear_fitter", lambda *a, **k: calls.append(1) or real(*a, **k))
    sealed = numeric.mattered(numeric.state.model_copy(update={"plan_locked": None}))
    assert calls == [] and sealed["refused"] is None
    assert sealed["curve"]["sealed"] == S.SEALED and sealed["curve"]["specs"] == []
    assert sealed["says"] == S.SEALED and sealed["methods"] is None
    assert sealed["reported"] is None
    assert [c["key"] for c in sealed["curve"]["choices"]] == ORDER
    opened = numeric.mattered()
    assert calls == [1] and opened["curve"]["sealed"] is None and opened["curve"]["specs"]


# ── the guards: each fires where it should and stays silent where it should ──


def test_a_primary_that_is_not_the_reported_estimate_is_refused(numeric, tmp_path):
    """The reported row moved by ten of its standard errors: the curve's refit no longer
    reproduces it, so no specification is shown. The fixture's own fit passes (the tests above).

    The refusal quotes no estimate, neither the refit's nor the reported one: it is served where
    no estimate is (the route's refusal marks no lock shown, and a withheld artifact keeps no
    refusal), so a number in it would be an estimate seen with nothing recording it."""
    import copy

    from turbotab.core.graph import Bundle

    data = copy.deepcopy(numeric.fit.data)
    model = next(m for m in data["models"] if m["family"] == "linear")
    row = next(r for r in model["coefficients"] if r["feature"] == "fiber")
    for k in ("estimate", "ci_low", "ci_high"):
        row[k] += 10 * row["se"]
    moved = Run(cohort(), tmp_path, numeric.state, fit=Bundle(data=data))
    art = moved.mattered()
    assert art["curve"] is None and art["says"] is None
    message = art["refused"]["message"]
    assert "would not be fitted as the primary is" in message
    assert "does not reproduce the reported estimate" in message
    assert not re.search(r"\d", message), message  # the same rows: no number at all
    with pytest.raises(S.SpecCurveRefused, match="does not reproduce the reported estimate"):
        M.served(art, status="fresh", fresh=True, key="k", lock_id="r1")  # served, not withheld
    reason = "a question the estimates rest on is open"
    held = M.withheld(art, reason)
    assert held["refused"] is None and held["withheld"] == reason
    assert numeric.mattered()["refused"] is None  # the fit's own row: silent


def test_a_withheld_curve_carries_none_of_its_estimates(numeric):
    art = numeric.mattered()
    numbers = {repr(v) for s in art["curve"]["specs"] for v in (s["estimate"], s["low"], s["high"])}
    numbers |= {repr(art["reported"][k]) for k in ("estimate", "ci_low", "ci_high")}
    assert any(n in json.dumps(art) for n in numbers)  # served: the estimates are there
    held = M.withheld(art, "a question the estimates rest on is open")
    text = json.dumps(held)
    assert not [n for n in numbers if n in text]
    assert held["curve"]["specs"] == [] and held["curve"]["summary"] is None
    assert held["says"] is None and held["methods"] is None and held["reported"] is None
    assert held["refused"] is None and not M.shows_estimates(held)


def test_agreement_is_judged_against_the_standard_error_both_ways():
    rep = M.MatteredReported(family="linear", feature="x", estimate=1.0, ci_low=0.5, ci_high=1.5, se=0.25,
                     n_rows=100, covariance="HC3")
    near = 0.5 * M.AGREEMENT * 0.25
    assert M.agrees(S.Spec("p", {}, True, 1.0 + near, 0.5 - near, 1.5, 100), rep)
    far = 2 * M.AGREEMENT * 0.25
    assert not M.agrees(S.Spec("p", {}, True, 1.0 + far, 0.5, 1.5, 100), rep)
    assert not M.agrees(S.Spec("p", {}, True, 1.0, 0.5, 1.5 + far, 100), rep)
    assert not M.agrees(S.Spec("p", {}, True, 1.0, 0.5, 1.5, 99), rep)  # other rows


FILL = d.MissingSpec(strategy="impute", acknowledged=True, reason="a test of the curve's fill")


def filled_fit(table: pd.DataFrame, *, rules: tuple[int, ...], covariates: list[str],
               basis: dict[str, Any] | None = None) -> tuple[float, float, float, int]:
    """statsmodels' least squares with HC3 on t(n − p) on a specification's rows under a single
    fill: every row its rules keep, each blank its column's median on those rows (the fill, done
    here by hand). ``basis``: a column's spline terms beside it, by name (:func:`rcs_terms`)."""
    part = table[rows_of(table, rules=rules, columns=["glucose"])].copy()
    for c in ["fiber", *covariates]:
        if c != "sex" and part[c].isna().any():
            part[c] = part[c].fillna(float(np.nanmedian(part[c].to_numpy(dtype=float))))
    X = ef.design_matrix(part, ["fiber", *covariates])
    for c, terms in (basis or {}).items():
        if c in covariates:
            for j, column in enumerate(terms(part[c].to_numpy(dtype=float)).T):
                X[f"{c}_rcs{j}"] = column
    fit = sm.OLS(part["glucose"].astype(float), X).fit(cov_type="HC3", use_t=True)
    low, high = fit.conf_int(alpha=0.05).loc["fiber"]
    return float(fit.params["fiber"]), float(low), float(high), int(len(part))


def rcs_terms(x: np.ndarray, k: int = 4) -> np.ndarray:
    """Harrell's restricted cubic spline's k − 2 nonlinear terms (Regression Modeling Strategies,
    2nd ed., §2.4.5, eq. 2.25), its knots at the 5th, 35th, 65th and 95th percentiles of ``x`` by
    R's type-7 quantile, as ``rms::rcs`` places four; written out here from the formula."""
    t = np.quantile(x, [0.05, 0.35, 0.65, 0.95], method="linear")

    def cube(u: np.ndarray) -> np.ndarray:
        return np.clip(u, 0.0, None) ** 3

    out = [cube(x - t[j]) - cube(x - t[k - 2]) * (t[k - 1] - t[j]) / (t[k - 1] - t[k - 2])
           + cube(x - t[k - 1]) * (t[k - 2] - t[j]) / (t[k - 1] - t[k - 2]) for j in range(k - 2)]
    return np.column_stack(out) / (t[k - 1] - t[0]) ** 2


def test_under_a_single_fill_each_specification_is_fitted_through_the_fill(tmp_path, numeric):
    """A single fill is a step the raw refit does not repeat, so each specification is fitted as
    its plan: every row its rules keep, its blanks filled on those rows. Each is statsmodels' fit
    on the rows filled here by hand (medians on each specification's own rows); on the base the
    curve was refused by name. Under complete cases (the tests above) no fill is a step."""
    run = Run(cohort(), tmp_path, state(missing=FILL))
    art = run.mattered()
    assert art["refused"] is None, art["refused"]
    curve = art["curve"]
    got = {s["key"]: s for s in curve["specs"]}
    assert set(got) == set(departures()) and curve["not_estimated"] == []
    for k, (rules, covariates) in departures().items():
        est, low, high, n = filled_fit(run.table, rules=rules, covariates=covariates)
        s = got[k]
        assert s["n"] == n, k
        assert (s["estimate"], s["low"], s["high"]) == pytest.approx((est, low, high), rel=1e-9), k
    assert curve["through"] == ["impute"]
    assert "(the fill of missing values)" in art["methods"]
    assert "multiple imputation" not in art["methods"] and "Rubin" not in art["methods"]
    plain = numeric.mattered()  # complete cases: every specification the raw refit, as before
    assert plain["curve"]["through"] == [] and "through its plan" not in plain["methods"]


def test_under_a_fill_a_departure_whose_added_column_has_blanks_is_fitted_through_the_fill(
        tmp_path):
    """A single fill with blanks only in `bmi`, which the answers leave out: the primary's columns
    are complete on its rows, so the fill changes nothing there and the primary is the raw refit;
    the departure adding `bmi` alone has its blanks filled by its plan, and is fitted through the
    fill (on the base it was not estimated). Every specification is statsmodels' fit, `bmi`'s blanks
    filled here by hand."""
    frame = ef.cohort(n=800, seed=7)
    frame.loc[np.random.default_rng(11).choice(len(frame), 30, replace=False), "bmi"] = np.nan
    run = Run(frame, tmp_path, state(missing=FILL))
    art = run.mattered()
    assert art["refused"] is None, art["refused"]
    curve = art["curve"]
    assert curve["not_estimated"] == []
    got = {s["key"]: s for s in curve["specs"]}
    assert set(got) == set(departures())
    for k, (rules, covariates) in departures().items():
        est, low, high, n = filled_fit(run.table, rules=rules, covariates=covariates)
        assert got[k]["estimate"] == pytest.approx(est, rel=1e-9) and got[k]["n"] == n, k
    added = next(s for s in curve["specs"] if s["picks"]["left_out:bmi"] == "added")
    assert added["n"] == len(run.table[rows_of(run.table, rules=(0, 1), columns=["glucose"])])
    assert curve["through"] == ["impute"]  # the bmi departure's fill alone


def test_a_spline_on_a_covariate_is_fitted_through_and_one_on_the_exposure_stays_refused(tmp_path):
    """Truth (c), both ways. A spline declared on `age`, a covariate: each specification is fitted
    through the form, its knots placed on its own rows, and is statsmodels' fit with Harrell's
    restricted cubic spline terms written out here (on the base the curve was refused). A spline on
    `fiber`, what the curve follows, splits it into three coefficients: there is no one estimate to
    follow, so the curve stays refused, in words and with no exit, before anything is fitted."""
    from turbotab.core.decisions import ExposureFormSpec

    formed = state(exposure_forms={"age": ExposureFormSpec(form="spline", knots=4, scale="raw")})
    run = Run(cohort(), tmp_path / "age", formed)
    assert run.design.objects["spec"]["exposure_forms"]  # the design applies it
    art = run.mattered()
    assert art["refused"] is None, art["refused"]
    curve = art["curve"]
    got = {s["key"]: s for s in curve["specs"]}
    assert set(got) == set(departures()) and curve["through"] == ["form"]
    for k, (rules, covariates) in departures().items():
        est, low, high, n = statsmodels_fit(run.table, outcome="glucose", rules=rules,
                                            covariates=covariates)
        if "age" in covariates:  # the spline's terms on the specification's own complete cases
            part = run.table[rows_of(run.table, rules=rules,
                                     columns=["glucose", "fiber", *covariates])]
            X = ef.design_matrix(part, ["fiber", *covariates])
            for j, column in enumerate(rcs_terms(part["age"].to_numpy(dtype=float)).T):
                X[f"age_rcs{j}"] = column
            fit = sm.OLS(part["glucose"].astype(float), X).fit(cov_type="HC3", use_t=True)
            est = float(fit.params["fiber"])
            low, high = (float(v) for v in fit.conf_int(alpha=0.05).loc["fiber"])
        assert got[k]["n"] == n, k
        assert (got[k]["estimate"], got[k]["low"], got[k]["high"]) == pytest.approx(
            (est, low, high), rel=1e-8), k
    assert "(a declared form other than a straight line)" in art["methods"]

    on_fiber = state(exposure_forms={"fiber": ExposureFormSpec(form="spline", knots=4,
                                                               scale="raw")})
    refused = Run(cohort(), tmp_path / "fiber", on_fiber).mattered()
    assert refused["curve"] is None and refused["refused"]["exits"] == []
    assert refused["refused"]["message"] == M.FORMED_EXPOSURE.format(
        exposure="fiber", n=3, names="`fiber`, `fiber'` and `fiber''`")


def test_a_one_column_form_on_the_exposure_is_refused_and_one_on_a_covariate_is_fitted_through(
        tmp_path):
    """V3 (PP2 C1 repair), both ways. One cut point declared on `fiber`, what the curve follows,
    or one data-derived cut point: the linear model fits it as one column (`fiber_C2`,
    `fiber_above`), a contrast of the groups either side of the cut, not a difference per unit of
    `fiber`. Refused in words that end with what happens (the declared cut's sentence written out,
    VR2-2), with no exit, before anything is fitted, as truth (c) refuses an
    exposure split by its own form (on the base refused as a step the refit did not repeat; before
    this repair drawn, labeled per unit of `fiber_C2`). One cut point declared on `age`, a
    covariate, is fitted through: each specification is statsmodels' fit with `age` replaced by
    its indicator above 50 (the form's groups: at most the cut, and above it), written out here
    (on the base the curve was refused)."""
    from turbotab.core.decisions import ExposureFormSpec

    for form, name in ((ExposureFormSpec(form="categories", cuts=[20.0]), "fiber_C2"),
                       (ExposureFormSpec(form="optimal"), "fiber_above")):
        run = Run(cohort(), tmp_path / form.form, state(exposure_forms={"fiber": form}))
        model = next(m for m in run.fit.data["models"] if m["family"] == "linear")
        assert [r["feature"] for r in model["coefficients"] if "fiber" in r["feature"]] == [name]
        art = run.mattered()
        assert art["curve"] is None and art["reported"] is None
        assert art["refused"] == {"message": M.FORMED_CONTRAST.format(exposure="fiber", name=name),
                                  "exits": []}, form.form
        if form.form == "categories":  # the sentence the user reads, ending with what happens
            assert art["refused"]["message"] == (
                "The plan fits `fiber` by a declared form other than a straight line: the linear "
                "model's one coefficient of it (`fiber_C2`) compares the groups on either side of "
                "the form's cut point, and the curve follows only a difference per unit, so it is "
                "not drawn.")

    run = Run(cohort(), tmp_path / "age",
              state(exposure_forms={"age": ExposureFormSpec(form="categories", cuts=[50.0])}))
    art = run.mattered()
    assert art["refused"] is None, art["refused"]
    curve = art["curve"]
    assert curve["through"] == ["form"] and curve["not_estimated"] == []
    assert curve["estimate_label"] == "Difference in mean `glucose` per unit of `fiber`"
    got = {s["key"]: s for s in curve["specs"]}
    assert set(got) == set(departures())
    for k, (rules, covariates) in departures().items():
        part = run.table[rows_of(run.table, rules=rules,
                                 columns=["glucose", "fiber", *covariates])]
        X = ef.design_matrix(part.assign(age=(part["age"] > 50.0).astype(float)),
                             ["fiber", *covariates])
        fit = sm.OLS(part["glucose"].astype(float), X).fit(cov_type="HC3", use_t=True)
        low, high = fit.conf_int(alpha=0.05).loc["fiber"]
        assert got[k]["n"] == len(part), k
        assert (got[k]["estimate"], got[k]["low"], got[k]["high"]) == pytest.approx(
            (float(fit.params["fiber"]), float(low), float(high)), rel=1e-9), k


def test_an_omics_normalization_refuses_the_curve_by_name_and_no_step_of_the_others_does(
        tmp_path, numeric):
    """An omics normalization (log2) of `fiber` leaves its column's name on the normalized scale,
    which the curve's label does not state, so the curve is not fitted through it: refused by name,
    with no exit, before anything is fitted. The curves drawn above hold none of these steps."""
    frame = cohort()
    frame["fiber"] = frame["fiber"].clip(lower=1.0)  # intensities: positive
    st = state(findings={"omics_scale": d.FindingDisposition(
        action="applied", option="log2", params={"kind": "intensities", "columns": ["fiber"]})})
    run = Run(frame, tmp_path, st)
    assert run.design.objects["spec"]["normalization"]["columns"] == ["fiber"]  # the premise
    art = run.mattered()
    assert art["curve"] is None and art["refused"]["exits"] == []
    assert art["refused"]["message"] == M.NOT_FOLLOWED_REFUSAL.format(
        steps="the omics normalization")
    assert numeric.mattered()["refused"] is None


BELOW_DETECTION = d.MissingSpec(strategy="complete_case", below_detection="half_minimum",
                                censored_columns=["bmi"])


def test_a_departure_whose_plan_holds_a_step_the_curve_does_not_follow_is_not_estimated(
        tmp_path, numeric):
    """V2 (PP2 C1 repair), both ways. `bmi`'s blanks declared values below detection, filled at
    half its minimum: a step the curve does not fit a specification through (``detect``,
    :data:`M.NOT_FOLLOWED`). Only the departure adding `bmi`, which the adjustment answers leave
    out, reads it, so that one specification alone is not estimated, the step named, and every
    other is statsmodels' fit on its own complete cases, as with no declaration. With no
    declaration (the numeric fixture) the same departure is estimated."""
    run = Run(cohort(), tmp_path, state(missing=BELOW_DETECTION))
    art = run.mattered()
    assert art["refused"] is None, art["refused"]
    curve = art["curve"]
    added = [k for k in departures() if "left_out:bmi=added" in k]
    assert [s["key"] for s in curve["not_estimated"]] == added
    assert curve["not_estimated"][0]["reason"] == (
        "With this choice changed, the plan would apply the fill of values below detection, which "
        "the curve does not fit a specification through, so it is not estimated.")
    assert curve["through"] == []
    got = {s["key"]: s for s in curve["specs"]}
    assert set(got) == set(departures()) - set(added)
    for k, s in got.items():
        rules, covariates = departures()[k]
        est, low, high, n = statsmodels_fit(run.table, outcome="glucose", rules=rules,
                                            covariates=covariates)
        assert s["n"] == n, k
        assert (s["estimate"], s["low"], s["high"]) == pytest.approx((est, low, high),
                                                                     rel=1e-9), k
    assert set(added) <= {s["key"] for s in numeric.mattered()["curve"]["specs"]}


def test_the_hold_counts_the_specifications_the_curve_fits_and_the_fit_s_own_seconds(numeric):
    """What the hold counts (``fit_press.curve_estimate``): the specifications the curve fits for
    the state, the eight this curve draws, times the followed family's seconds: the fit's measured
    ``fit_seconds`` once it is made, the shelf's ``estimate_seconds`` before. None where the curve is
    not offered or no followed family is timed."""
    curve = numeric.mattered()["curve"]
    assert M.specifications_of(numeric.state) == len(curve["specs"]) + len(curve["not_estimated"])
    assert M.specifications_of(numeric.state) == len(departures()) == 8
    seconds = next(m["fit_seconds"] for m in numeric.fit.data["models"] if m["family"] == "linear")
    assert M.expected_seconds(numeric.state, numeric.fit) == pytest.approx(8 * seconds)
    shelf = {"families": [{"key": "elastic_net", "estimate_seconds": 900.0},
                          {"key": "linear", "estimate_seconds": 2.5}]}
    assert M.expected_seconds(numeric.state, shelf) == 20.0
    assert M.expected_seconds(numeric.state, {"families": [{"key": "elastic_net",
                                                            "estimate_seconds": 9.0}]}) is None
    predict = numeric.state.model_copy(update={"purpose": "prediction"})
    assert M.specifications_of(predict) is None and M.expected_seconds(predict, shelf) is None


def energy_cohort() -> pd.DataFrame:
    frame = cohort()
    rng = np.random.default_rng(5)
    frame["kcal"] = 2000 + 40 * (frame["fiber"] - 20) + rng.normal(0, 300, len(frame))
    return frame


def test_a_residual_energy_model_is_fitted_through_and_the_curve_follows_its_column(tmp_path):
    """The residual model fits ``fiber_adj`` in place of ``fiber``: the curve follows that column,
    the fit's own reported row, and fits each specification through the energy model on its own
    rows. Each is statsmodels' fit of `fiber_adj` written out here (Willett's residual of `fiber` on
    `kcal` on the specification's rows, plus `fiber`'s mean, `kcal` kept). The departure that drops
    `kcal` leaves the energy model without its total energy: it is not estimated, in words, while
    every other departure is. On the base the curve was refused by naming the energy model."""
    st = state(roles={**ef.ROLES, "kcal": "energy"}, answers={**ef.ANSWERS, "kcal": ef.CONFOUNDER},
               energy_adjustment=d.EnergyAdjustment(method="residual", energy_column="kcal",
                                                    nutrients=["fiber"]))
    run = Run(energy_cohort(), tmp_path, st)
    model = next(m for m in run.fit.data["models"] if m["family"] == "linear")
    assert "fiber_adj" in [r["feature"] for r in model["coefficients"]]  # the premise
    art = run.mattered()
    assert art["refused"] is None, art["refused"]
    curve = art["curve"]
    assert curve["estimate_label"] == "Difference in mean `glucose` per unit of `fiber_adj`"
    assert art["reported"]["feature"] == "fiber_adj" and curve["through"] == ["energy"]
    assert {c["key"] for c in curve["choices"]} == {*ORDER, "adjusted:kcal"}
    assert [s["picks"]["adjusted:kcal"] for s in curve["not_estimated"]] == ["dropped"]
    assert curve["not_estimated"][0]["reason"] == ("With this choice changed, "
                                                   + M.ENERGY_READS.format(column="kcal"))
    assert len(curve["specs"]) == len(departures())  # every other specification is estimated
    for s in curve["specs"]:
        picks = s["picks"]
        rules = tuple(i for i in (0, 1) if picks[f"exclusion:{i}"] == M.RECORDED)
        covariates = [c for c in ADJUSTED if picks[f"adjusted:{c}"] == M.RECORDED] \
            + (["bmi"] if picks["left_out:bmi"] == "added" else [])
        part = run.table[rows_of(run.table, rules=rules,
                                 columns=["glucose", "fiber", "kcal", *covariates])]
        e, n_ = part["kcal"].to_numpy(dtype=float), part["fiber"].to_numpy(dtype=float)
        slope = float(np.cov(e, n_, ddof=1)[0, 1] / np.var(e, ddof=1))
        X = ef.design_matrix(part, ["kcal", *covariates])
        X["fiber_adj"] = n_ - slope * (e - e.mean())
        fit = sm.OLS(part["glucose"].astype(float), X).fit(cov_type="HC3", use_t=True)
        low, high = fit.conf_int(alpha=0.05).loc["fiber_adj"]
        assert s["n"] == len(part), s["key"]
        assert (s["estimate"], s["low"], s["high"]) == pytest.approx(
            (float(fit.params["fiber_adj"]), float(low), float(high)), rel=1e-8), s["key"]
    assert "(the energy model)" in art["methods"]
    # The label quotes the column the curve follows: the reported row's.
    assert M.reported_of(run.fit, "fiber", "regression",
                         predictors=run.design.objects["spec"]["predictors"]).feature == "fiber_adj"


def strata_cohort() -> pd.DataFrame:
    """The cohort with total energy 200 kcal higher in men, so a residual model within the levels
    of `sex` is not the pooled one. `sex` has no blank."""
    frame = cohort()
    rng = np.random.default_rng(5)
    frame["kcal"] = (2000 + 40 * (frame["fiber"] - 20) + 200 * (frame["sex"] == "male")
                     + rng.normal(0, 300, len(frame)))
    return frame


def energy_state(*, answers: dict[str, dict[str, str]] | None = None,
                 **energy: Any) -> d.ProjectState:
    """`kcal` total energy, a confounder, and the residual energy model of `fiber` on it;
    ``answers`` replace the adjustment answers of the columns they name."""
    return state(roles={**ef.ROLES, "kcal": "energy"},
                 answers={**ef.ANSWERS, "kcal": ef.CONFOUNDER, **(answers or {})},
                 energy_adjustment=d.EnergyAdjustment(method="residual", energy_column="kcal",
                                                      nutrients=["fiber"], **energy))


def within_strata(part: pd.DataFrame) -> np.ndarray:
    """`fiber_adj` by the stratified residual model's definition (``StratifiedEnergyAdjuster``):
    N − b_s (E − Ē_s) − N̄_s + N̄ in each level s of `sex`, a blank `sex` a level of its own, with
    b_s, Ē_s and N̄_s that level's slope of `fiber` on `kcal` and means on these rows, and N̄
    `fiber`'s mean over all of them. (A level under ``energy.MIN_LEVEL_ROWS`` rows would take the
    pooled slope instead; each level here has more.)"""
    from turbotab.core.methods.energy import MIN_LEVEL_ROWS

    e, n_ = part["kcal"].to_numpy(dtype=float), part["fiber"].to_numpy(dtype=float)
    levels = part["sex"].astype(object).where(part["sex"].notna(), "(blank)")
    out = np.empty(len(part))
    for level in pd.unique(levels):
        m = (levels == level).to_numpy()
        assert m.sum() >= MIN_LEVEL_ROWS, level  # the definition above, not the pooled slope's
        slope = float(np.cov(e[m], n_[m], ddof=1)[0, 1] / np.var(e[m], ddof=1))
        out[m] = n_[m] - slope * (e[m] - e[m].mean()) - n_[m].mean() + n_.mean()
    return out


def test_a_departure_dropping_the_strata_column_keeps_it_as_the_energy_model_s_strata(tmp_path):
    """V3-1 (PP2 C1 repair round 1), both ways. The residual energy model within the levels of
    `sex`, an adjusted covariate. The departure dropping `sex` from the adjustment is the plan with
    `sex` out of the outcome model and its energy model still within `sex`, the column kept among
    its inputs as the design keeps a strata column that is no predictor
    (``pipeline.input_columns``). It is estimated: statsmodels' fit of `fiber_adj` with `sex` out
    of the model (on the base it was not estimated, said to be a plan that could not be fitted).
    A column a departure drops that no step reads (`age`) leaves its inputs: each other
    specification is the same fit with its own covariates, `fiber_adj` the stratified model's
    definition written out here (:func:`within_strata`). The departure dropping `kcal` leaves the
    energy model without its total energy and is not estimated, as with no strata."""
    run = Run(strata_cohort(), tmp_path, energy_state(strata="sex"))
    assert run.design.objects["spec"]["energy"]["strata"] == "sex"  # the premise
    art = run.mattered()
    assert art["refused"] is None, art["refused"]
    curve = art["curve"]
    assert curve["through"] == ["energy"]
    assert [s["picks"]["adjusted:kcal"] for s in curve["not_estimated"]] == ["dropped"]
    assert curve["not_estimated"][0]["reason"] == ("With this choice changed, "
                                                   + M.ENERGY_READS.format(column="kcal"))
    assert len(curve["specs"]) == len(departures())
    every = rows_of(run.table, rules=(0, 1), columns=["glucose", "fiber", "kcal", *ADJUSTED])
    assert [s["n"] for s in curve["specs"] if s["picks"]["adjusted:sex"] == "dropped"] == [
        int(every.sum())]  # `sex` has no blank: the primary's rows
    for s in curve["specs"]:
        picks = s["picks"]
        rules = tuple(i for i in (0, 1) if picks[f"exclusion:{i}"] == M.RECORDED)
        covariates = [c for c in ADJUSTED if picks[f"adjusted:{c}"] == M.RECORDED] \
            + (["bmi"] if picks["left_out:bmi"] == "added" else [])
        part = run.table[rows_of(run.table, rules=rules,
                                 columns=["glucose", "fiber", "kcal", *covariates])]
        X = ef.design_matrix(part, ["kcal", *covariates])  # `sex` only where it is adjusted for
        X["fiber_adj"] = within_strata(part)
        fit = sm.OLS(part["glucose"].astype(float), X).fit(cov_type="HC3", use_t=True)
        low, high = fit.conf_int(alpha=0.05).loc["fiber_adj"]
        assert s["n"] == len(part), s["key"]
        assert (s["estimate"], s["low"], s["high"]) == pytest.approx(
            (float(fit.params["fiber_adj"]), float(low), float(high)), rel=1e-8), s["key"]


NOT_A_CAUSE = {"causes_exposure": "no", "causes_outcome": "no", "after_exposure": "no"}


def slopes_cohort(blank_sex: int = 0) -> pd.DataFrame:
    """The cohort with total energy rising 70 kcal per gram of `fiber` in men and 30 in women,
    and 350 kcal higher in men, so the residual model within the levels of `sex` is not the
    pooled one; ``blank_sex`` rows, drawn at random, with `sex` blank."""
    frame = cohort()
    rng = np.random.default_rng(23)
    male = (frame["sex"] == "male").to_numpy()
    frame["kcal"] = (1900 + np.where(male, 70.0, 30.0) * (frame["fiber"] - 20) + 350 * male
                     + rng.normal(0, 250, len(frame)))
    if blank_sex:
        frame.loc[rng.choice(len(frame), blank_sex, replace=False), "sex"] = np.nan
    return frame


def strata_fit(table: pd.DataFrame, picks: dict[str, str]) -> tuple[int, float, float, float]:
    """A specification under the residual model within `sex`, by its definition: the rules its
    picks apply; its covariates, each column they adjust for or add; its rows, those with the
    outcome, `fiber`, `kcal` and its covariates recorded (`sex` among them only where it is a
    covariate, so a blank `sex` keeps its row elsewhere); and statsmodels' least squares with HC3
    on t(n − p) of `glucose` on its covariates and `fiber_adj` (:func:`within_strata` on these
    rows). Returns (n, estimate, low, high)."""
    rules = tuple(i for i in (0, 1) if picks[f"exclusion:{i}"] == M.RECORDED)
    covariates = [k.split(":", 1)[1] for k, v in picks.items()
                  if (k.startswith("adjusted:") and v == M.RECORDED)
                  or (k.startswith("left_out:") and v == "added")]
    part = table[rows_of(table, rules=rules, columns=["glucose", "fiber", "kcal", *covariates])]
    X = ef.design_matrix(part, covariates)
    X["fiber_adj"] = within_strata(part)
    fit = sm.OLS(part["glucose"].astype(float), X).fit(cov_type="HC3", use_t=True)
    low, high = fit.conf_int(alpha=0.05).loc["fiber_adj"]
    return len(part), float(fit.params["fiber_adj"]), float(low), float(high)


def test_a_strata_column_the_plan_never_adjusts_for_stays_the_energy_model_s_strata(tmp_path):
    """VR1-1 (PP2 C1), both ways. The residual energy model within the levels of `sex`, where the
    plan never adjusts for `sex` (a cause of neither the exposure nor the outcome): `sex` is an
    input only, the model's strata, no predictor (``pipeline.input_columns``). Each departure's
    plan keeps it so. Every specification but the one dropping `kcal` is estimated, each
    statsmodels' fit of `fiber_adj` within `sex` with its own covariates (:func:`strata_fit`),
    the departure adding `sex` to the adjustment among them, and a column a departure drops that
    no step reads (`age`) leaves it. A departure's inputs taken from the primary's predictors
    instead of its inputs fail here: each departure but the one adding `sex` then stops on its
    strata column missing.

    Then with 40 rows' `sex` blank, under complete cases, `sex` adjusted for: the specifications
    that adjust for `sex` lose the rows with it blank, and the departure dropping `sex` keeps
    them, a blank `sex` a level of its own in the energy model (``StratifiedEnergyAdjuster``).
    Its n is the rows of the primary's rules with every column it reads recorded, and its
    estimate is statsmodels' fit written out here and R's: lm with sandwich's HC3 on the same
    rows, `fiber_adj` written out with the blanks their own level, -0.517709353575527 (R 4.6.1,
    sandwich 3.1.3, read to 15 digits)."""
    never = Run(slopes_cohort(), tmp_path / "never",
                energy_state(strata="sex", answers={"sex": NOT_A_CAUSE}))
    spec = never.design.objects["spec"]
    assert spec["energy"]["strata"] == "sex"  # the premise
    assert "sex" in spec["inputs"] and "sex" not in spec["predictors"]
    art = never.mattered()
    assert art["refused"] is None, art["refused"]
    curve = art["curve"]
    assert curve["through"] == ["energy"]
    assert [s["picks"]["adjusted:kcal"] for s in curve["not_estimated"]] == ["dropped"]
    assert curve["not_estimated"][0]["reason"] == ("With this choice changed, "
                                                   + M.ENERGY_READS.format(column="kcal"))
    changed = [[k for k, v in s["picks"].items() if v != M.RECORDED] for s in curve["specs"]]
    assert changed == [[], ["exclusion:0"], ["exclusion:1"], ["adjusted:age"],
                       ["adjusted:smoking"], ["adjusted:activity"], ["left_out:sex"],
                       ["left_out:bmi"]]
    assert curve["specs"][6]["picks"]["left_out:sex"] == "added"
    for s in curve["specs"]:
        n, estimate, low, high = strata_fit(never.table, s["picks"])
        assert s["n"] == n, s["key"]
        assert (s["estimate"], s["low"], s["high"]) == pytest.approx(
            (estimate, low, high), rel=1e-8), s["key"]

    blank = Run(slopes_cohort(blank_sex=40), tmp_path / "blank", energy_state(strata="sex"))
    assert int(blank.table["sex"].isna().sum()) == 40  # the premise
    assert blank.state.missing.strategy == "complete_case"
    curve = blank.mattered()["curve"]
    assert [s["picks"]["adjusted:kcal"] for s in curve["not_estimated"]] == ["dropped"]
    assert len(curve["specs"]) == len(departures())
    for s in curve["specs"]:
        n, estimate, low, high = strata_fit(blank.table, s["picks"])
        assert s["n"] == n, s["key"]
        assert (s["estimate"], s["low"], s["high"]) == pytest.approx(
            (estimate, low, high), rel=1e-8), s["key"]
    every = rows_of(blank.table, rules=(0, 1),
                    columns=["glucose", "fiber", "kcal", *[c for c in ADJUSTED if c != "sex"]])
    blanks = int(blank.table["sex"][every].isna().sum())
    [primary] = [s for s in curve["specs"] if s["primary"]]
    [dropped] = [s for s in curve["specs"] if s["picks"]["adjusted:sex"] == "dropped"]
    assert dropped["n"] == int(every.sum()) == primary["n"] + blanks and blanks > 0
    assert dropped["estimate"] == pytest.approx(-0.517709353575527, rel=1e-10)  # R's


REFIT_ERROR = ("With this choice changed, the curve's refit of this specification stopped on an "
               "error, not on a refusal of the plan, so it is not estimated: ")  # in words


def test_an_error_in_the_curve_s_own_refit_is_never_said_to_be_a_plan_that_could_not_be_fitted(
        tmp_path, monkeypatch):
    """V3-1 (PP2 C1 repair round 1), both ways. Where the curve's refit of a departure through its
    plan raises an error the engine did not type as a refusal (constructed here: building each
    departure's pipeline raises one), the failure is the curve's own: each such departure is not
    estimated, said in those words, never as a plan that could not be fitted. Where it raises the
    engine's typed refusal of a plan (``graph.StageRefusal``), the plan is refused: said to be a
    plan that could not be fitted, with the refusal's words. The primary is fitted by the design's
    own pipeline either way, and the departure dropping `kcal` is not estimated before any build."""
    from turbotab.core.graph import StageRefusal
    from turbotab.core.models import pipeline as P

    run = Run(energy_cohort(), tmp_path, energy_state())
    for raised, words in ((RuntimeError("a constructed error"), REFIT_ERROR),
                          (StageRefusal("constructed", "A constructed refusal."),
                           "With this choice changed, the plan could not be fitted: ")):
        def build(*args: Any, raised: Exception = raised, **kwargs: Any) -> Any:
            raise raised

        monkeypatch.setattr(P, "build_pipeline", build)
        art = run.mattered()
        assert art["refused"] is None, art["refused"]
        curve = art["curve"]
        assert [s["primary"] for s in curve["specs"]] == [True]
        held = [s for s in curve["not_estimated"] if s["picks"]["adjusted:kcal"] != "dropped"]
        assert len(held) == len(departures()) - 1
        for s in held:
            assert s["reason"] == words + str(raised), s["key"]
        if isinstance(raised, RuntimeError):
            assert not [s for s in held if "could not be fitted" in s["reason"]]


def test_a_departure_says_no_estimate_where_its_plan_has_none_and_no_interval_where_it_has_one(
        tmp_path, monkeypatch):
    """V3-2 (PP2 C1 repair round 1), both ways. Passive multiple imputation with a spline declared
    on `bmi`, the covariate the adjustment answers leave out: the departure adding `bmi` is the plan
    with `bmi` adjusted for, whose fit is blocked with no estimate at all (the plan's own fit
    shows it), so the departure is not estimated, said to have no estimate, with the block's words
    (on the base it was said to have no interval). Where a departure's plan has an estimate and no
    interval (constructed here: under a single fill, the `bmi` departure's table with `fiber`'s
    interval removed, with a reason or none), it is said to have no interval; where the row's
    estimate is removed too and the table gives no reason, no estimate, and never that no
    interval could be computed."""
    from turbotab.core.decisions import ExposureFormSpec
    from turbotab.core.stages import sensitivity

    passive = d.MissingSpec(strategy="multiple_imputation", imputation_model="passive")
    formed = {"bmi": ExposureFormSpec(form="spline", knots=4, scale="raw")}
    run = Run(cohort(), tmp_path / "curve", state(missing=passive, exposure_forms=formed))
    art = run.mattered()
    assert art["refused"] is None, art["refused"]
    [held] = art["curve"]["not_estimated"]
    assert held["picks"]["left_out:bmi"] == "added"
    plan = Run(cohort(), tmp_path / "plan", state(missing=passive, exposure_forms=formed,
                                                  answers={**ef.ANSWERS, "bmi": ef.CONFOUNDER}))
    linear = next(m for m in plan.fit.data["models"] if m["family"] == "linear")
    why = linear["inference"]["refused"]
    assert why and not [r for r in linear["coefficients"] or []
                        if r["feature"] == "fiber" and r.get("estimate") is not None]
    assert held["reason"] == "With this choice changed, the fit would report no estimate: " + why

    real = sensitivity.fit_on_rows
    fill = Run(cohort(), tmp_path / "fill", state(missing=FILL))
    blank = {"interval": ("ci_low", "ci_high"), "estimate": ("estimate", "ci_low", "ci_high")}
    for lacks, reason, words in (
            ("interval", "a constructed reason", "no interval: a constructed reason"),
            ("interval", None, "no interval: no interval could be computed"),
            ("estimate", None, "no estimate: no estimate could be computed")):
        def table(*args: Any, lacks: str = lacks, reason: str | None = reason,
                  **kwargs: Any) -> Any:
            fitted, (rows, info), concerns = real(*args, **kwargs)
            if "bmi" in args[4]:  # the inputs: the departure adding `bmi` alone
                rows = [{**r, **dict.fromkeys(blank[lacks])} if r["feature"] == "fiber" else r
                        for r in rows]
                info = {**info, "refused": reason}
            return fitted, (rows, info), concerns

        monkeypatch.setattr(sensitivity, "fit_on_rows", table)
        [held] = fill.mattered()["curve"]["not_estimated"]
        assert held["picks"]["left_out:bmi"] == "added"
        assert held["reason"] == "With this choice changed, the fit would report " + words


@pytest.mark.parametrize("measure, words", [("risk_difference", "marginal risk difference"),
                                            ("risk_ratio", "marginal risk ratio")])
def test_a_marginal_measure_refuses_the_curve_by_name_and_the_odds_ratio_does_not(binary, measure,
                                                                                  words):
    """The reported estimate under a marginal measure is standardized over the analyzed rows from
    the logistic model; the curve refits the logistic coefficient, a conditional log odds ratio,
    so its primary would not be the reported estimate. Refused by name, before the lock too."""
    st = binary.state.model_copy(update={
        "estimand": binary.state.estimand.model_copy(update={"measure": measure})})
    art = binary.mattered(st)
    assert art["curve"] is None and art["says"] is None and art["reported"] is None
    assert words in art["refused"]["message"]
    before = st.model_copy(update={"plan_locked": None})
    assert not M.offered(st).offered and not M.offered(before).offered
    assert words in M.offered(before).says
    # The odds ratio, the logistic model's own measure: the curve is drawn.
    assert binary.state.estimand.measure == "odds_ratio" and M.offered(binary.state).offered
    assert binary.mattered()["refused"] is None


AFTER_EXPOSURE = {"causes_exposure": "no", "causes_outcome": "yes", "after_exposure": "yes"}


def with_statin() -> pd.DataFrame:
    """The cohort with `statin`, two-valued with 120 blanks, which lowers glucose."""
    frame = cohort()
    rng = np.random.default_rng(3)
    statin = (rng.random(len(frame)) < 0.4).astype(float)
    frame["glucose"] = frame["glucose"] - 4 * statin
    statin[rng.choice(len(frame), 120, replace=False)] = np.nan
    frame["statin"] = statin
    return frame


def test_an_added_column_the_plan_would_code_otherwise_is_fitted_through_that_coding(tmp_path):
    """`statin` is two-valued with blanks, and the answer keeps a category's blanks as a level of
    their own, so a plan adjusting for it codes its blanks as a level (``level_columns``); the
    adjustment answers leave it out, so only the curve's departure adds it. That departure is
    fitted as its plan: its blank rows kept (complete cases set a level column aside, as the flow
    does) and coded as a level, statsmodels' fit with `statin`'s two indicators (1, and blank)
    written out here. On the base it was not estimated. `bmi`, a number with no step of its own,
    is added and estimated as statsmodels fits it, by the raw refit."""
    st = state(roles={**ef.ROLES, "statin": "covariate"},
               answers={**ef.ANSWERS, "statin": AFTER_EXPOSURE},
               missing=d.MissingSpec(strategy="complete_case", categorical="missing_category",
                                     acknowledged=True, reason="a test of the guard"))
    run = Run(with_statin(), tmp_path, st)
    art = run.mattered()
    assert art["refused"] is None, art["refused"]
    curve = art["curve"]
    assert "left_out:statin" in [c["key"] for c in curve["choices"]]
    assert curve["not_estimated"] == [] and curve["through"] == ["levels"]
    statin = next(s for s in curve["specs"] if s["picks"]["left_out:statin"] == "added")
    part = run.table[rows_of(run.table, rules=(0, 1), columns=["glucose", "fiber", *ADJUSTED])]
    assert part["statin"].isna().any()  # the premise: blank rows kept
    X = ef.design_matrix(part, ["fiber", *ADJUSTED])
    X["statin_1"] = (part["statin"] == 1).astype(float)
    X["statin_blank"] = part["statin"].isna().astype(float)
    fit = sm.OLS(part["glucose"].astype(float), X).fit(cov_type="HC3", use_t=True)
    low, high = fit.conf_int(alpha=0.05).loc["fiber"]
    assert statin["n"] == len(part)
    assert (statin["estimate"], statin["low"], statin["high"]) == pytest.approx(
        (float(fit.params["fiber"]), float(low), float(high)), rel=1e-8)
    bmi = next(s for s in curve["specs"] if s["picks"]["left_out:bmi"] == "added")
    est, low, high, n = statsmodels_fit(run.table, outcome="glucose", rules=(0, 1),
                                        covariates=[*ADJUSTED, "bmi"])
    assert bmi["estimate"] == pytest.approx(est, rel=1e-9) and bmi["n"] == n
    assert len(curve["specs"]) == 1 + len(curve["choices"])
    assert "could not be estimated" not in art["says"]
    assert "(the coding of blanks as a level of their own)" in art["methods"]


def test_an_added_column_with_a_declared_form_is_fitted_through_the_form(numeric, tmp_path):
    """A spline declared on `bmi`, which the adjustment answers leave out: a plan adjusting for it
    fits the spline, so the departure that adds it is fitted through the form, its knots on its own
    rows: statsmodels' fit with Harrell's terms of `bmi` written out here (on the base it was not
    estimated). Every other specification is the raw refit, as without the form (the numeric
    fixture)."""
    from turbotab.core.decisions import ExposureFormSpec

    formed = state(exposure_forms={"bmi": ExposureFormSpec(form="spline", knots=4, scale="raw")})
    run = Run(cohort(), tmp_path, formed)
    art = run.mattered()
    assert art["refused"] is None, art["refused"]
    curve = art["curve"]
    assert curve["not_estimated"] == [] and curve["through"] == ["form"]
    assert {s["key"] for s in curve["specs"]} == set(departures())
    added = next(s for s in curve["specs"] if s["picks"]["left_out:bmi"] == "added")
    columns = ["glucose", "fiber", *ADJUSTED, "bmi"]
    part = run.table[rows_of(run.table, rules=(0, 1), columns=columns)]
    X = ef.design_matrix(part, ["fiber", *ADJUSTED, "bmi"])
    for j, column in enumerate(rcs_terms(part["bmi"].to_numpy(dtype=float)).T):
        X[f"bmi_rcs{j}"] = column
    fit = sm.OLS(part["glucose"].astype(float), X).fit(cov_type="HC3", use_t=True)
    low, high = fit.conf_int(alpha=0.05).loc["fiber"]
    assert added["n"] == len(part)
    assert (added["estimate"], added["low"], added["high"]) == pytest.approx(
        (float(fit.params["fiber"]), float(low), float(high)), rel=1e-8)
    plain = {s["key"]: s for s in numeric.mattered()["curve"]["specs"]}
    for s in curve["specs"]:
        if s["key"] != added["key"]:
            assert s["estimate"] == pytest.approx(plain[s["key"]]["estimate"], rel=1e-12)


def with_codes() -> pd.DataFrame:
    """The cohort with `eth`, whole numbers 1, 2, 3, 4, 6 and 7 (RIDRETH3's codes) whose groups 3
    and 6 shift glucose, so six groups and one straight line give the fiber slope differently."""
    frame = cohort()
    rng = np.random.default_rng(9)
    eth = rng.choice([1, 2, 3, 4, 6, 7], len(frame))
    frame["eth"] = eth.astype(float)
    frame["glucose"] = frame["glucose"] + np.select([eth == 3, eth == 6], [5.0, -4.0], 0.0)
    return frame


def eth_fit(table: pd.DataFrame, *, codes: bool) -> tuple[float, float, float, int]:
    """statsmodels' fit of the primary's model with `eth` added: one indicator per code (the first
    the reference) or one slope, HC3 on t(n − p), on both rules' rows with every column present."""
    keep = rows_of(table, rules=(0, 1), columns=["glucose", "fiber", *ADJUSTED, "eth"])
    part = table[keep]
    X = ef.design_matrix(part, ["fiber", *ADJUSTED])
    if codes:
        X = X.join(pd.get_dummies(part["eth"].astype(int), prefix="eth", drop_first=True,
                                  dtype=float))
    else:
        X["eth"] = part["eth"].astype(float)
    fit = sm.OLS(part["glucose"].astype(float), X).fit(cov_type="HC3", use_t=True)
    low, high = fit.conf_int(alpha=0.05).loc["fiber"]
    return float(fit.params["fiber"]), float(low), float(high), int(keep.sum())


# The fit's own ask for an unanswered whole-number predictor (``readings.predictors_or_ask``), as a
# plan adjusting for `eth` refuses with it.
ETH_ASK = ("`eth` holds whole numbers, which may be codes for categories (one indicator per level) "
           "or amounts (one slope); the fit waits for the answer for each.")
# A departure adding `eth` waits as that plan would, and says which reading is unanswered and where
# it is answered (Your data's readings card), never the fit's question, which the curve cannot take.
ETH_WAITS = ("With this choice changed, the fit would first wait for an answer, so it is not "
             "estimated: whether `eth` holds codes for categories or amounts is not answered yet; "
             "it is answered in Your data.")


@pytest.mark.parametrize("answer", [None, "code", "amount"])
def test_an_added_column_of_whole_numbers_waits_for_its_code_or_amount_reading(tmp_path, answer):
    """`eth` is a covariate the adjustment answers leave out (after the exposure), its whole numbers
    never answered as codes or amounts. A plan adjusting for it waits for that answer, so the
    departure adding it does too: unanswered it is not estimated, its reason the reading that is
    unanswered and where it is answered, with no question; answered as codes it is six groups, as
    an amount one straight line, each statsmodels' fit. An examined alternative adding it is judged
    the same way, and so is one adding it while its role puts it in no model (`eth` with no
    role)."""
    from turbotab.core.decisions import left_out
    from turbotab.core.readings import Unsettled, predictors_or_ask

    amounts = {**ef.AMOUNTS, **({"code_or_count:eth": answer} if answer else {})}
    st = state(roles={**ef.ROLES, "eth": "covariate"},
               answers={**ef.ANSWERS, "eth": AFTER_EXPOSURE}, amounts=amounts)
    run = Run(with_codes(), tmp_path, st)
    art = run.mattered()
    assert art["refused"] is None, art["refused"]
    curve = art["curve"]
    codes, line = eth_fit(run.table, codes=True), eth_fit(run.table, codes=False)
    assert codes[0] != pytest.approx(line[0], rel=1e-4)  # the coding moves the estimate
    want = None if answer is None else codes if answer == "code" else line
    examined = [M.MatteredExamined(key="eth", label="with eth", add=["eth"])]
    free = st.model_copy(update={
        "roles": {c: r for c, r in st.roles.items() if c != "eth"},
        "role_confirmations": {c: r for c, r in st.role_confirmations.items() if c != "eth"},
        "adjustment": {c: a for c, a in st.adjustment.items() if c != "eth"}})
    looks = [(curve["specs"], curve["not_estimated"], "left_out:eth", "added")]
    for s in (st, free):
        drawn, _ = M.curve_for(run.context(s), examined=examined)
        looks.append(([vars(x) for x in drawn.specs], [vars(x) for x in drawn.not_estimated],
                      "examined:eth", "examined"))
    for specs, missing, key, option in looks:
        got = [x for x in specs if x["picks"].get(key) == option]
        held = [x for x in missing if x["picks"].get(key) == option]
        if want is None:
            assert not got and len(held) == 1, key
            assert held[0]["reason"] == ETH_WAITS, key
            assert "?" not in held[0]["reason"] and "Tell me" not in held[0]["reason"], key
        else:
            assert not held and len(got) == 1, (key, held)
            est, low, high, n = want
            assert got[0]["estimate"] == pytest.approx(est, rel=1e-9), key
            assert got[0]["low"] == pytest.approx(low, rel=1e-9), key
            assert got[0]["high"] == pytest.approx(high, rel=1e-9) and got[0]["n"] == n, key
    # Every other departure is drawn either way (the guard is silent where nothing waits).
    assert len(curve["specs"]) == len(curve["choices"]) + (0 if answer is None else 1)
    # The fit's own gate over the plan adjusting for `eth` waits exactly where the departure does,
    # and asks the question the departure's reason leaves to Your data.
    from turbotab.core.datastore import DataStore

    with DataStore(Path(run.paths["data"]), 1 << 30) as store:
        summaries = {c.name: {"dtype": c.dtype, "n_unique": c.n_unique}
                     for c in store.info().columns}
        drop = [c for c in left_out(st) if c != "eth"]
        if answer is None:
            with pytest.raises(Unsettled) as asked:
                predictors_or_ask(st, summaries, drop=drop, store=store)
            assert str(asked.value).startswith(ETH_ASK) and "?" in str(asked.value)
        else:
            assert "eth" in predictors_or_ask(st, summaries, drop=drop, store=store)


def test_a_plan_with_no_recorded_choice_is_refused_with_no_exit(tmp_path, numeric):
    """No exclusion rule and no covariate: every alternative is derived from the record, so there
    is nothing to change and nothing to declare (no route accepts a declared one yet), before the
    lock and after it. The fixture's recorded choices are drawn."""
    st = ef.state(target="glucose", task="regression", measure="mean_difference",
                  roles={"pid": "identifier", "fiber": "exposure"}, plan_locked=True)
    run = Run(cohort()[["pid", "fiber", "glucose"]], tmp_path, st)
    for s in (st, st.model_copy(update={"plan_locked": None})):
        art = run.mattered(s)
        assert art["curve"] is None
        assert art["refused"] == {"message": M.NOTHING_TO_VARY, "exits": []}
    assert numeric.mattered()["refused"] is None


REFIT_FOLLOWS = ("The curve refits one coefficient of what you study per specification, by least "
                 "squares for a numeric outcome or logistic regression for a yes/no one; ")


@pytest.mark.parametrize("task, measure, said", [
    ("multiclass", "relative_risk_ratio", "`grade` is an outcome of several classes"),
    ("ordinal", "cumulative_odds_ratio", "`grade` is an outcome of ordered levels"),
    ("time_to_event", "hazard_ratio", "`grade` with its follow-up is a time to an event")])
def test_an_outcome_the_refit_does_not_follow_is_refused_by_name(task, measure, said):
    """The refit follows least squares or logistic regression only, so a multiclass, ordinal or
    time-to-event outcome is refused by naming it, with no exit (the outcome is the question),
    before the lock too; a numeric or yes/no outcome is offered."""
    def estimate(**slots: Any) -> d.ProjectState:
        return d.ProjectState(purpose="inference", target="grade",
                              estimand=d.EstimandSpec(exposure="fiber",
                                                      measure=slots.pop("measure")), **slots)

    refused = M.offered(estimate(task=task, measure=measure))
    assert not refused.offered and refused.exits == ()
    assert refused.says == REFIT_FOLLOWS + said + ", so the curve is not drawn for it."
    assert M.offered(estimate(task="regression", measure="mean_difference")).offered
    assert M.offered(estimate(task="binary", measure="odds_ratio")).offered


def test_a_time_to_event_outcome_is_named_with_its_follow_up_never_as_the_event_column_alone():
    """Under a time-to-event outcome the outcome column is the event and its follow-up is its own
    answer (``set_follow_up``), so the refusal names the two together as the time to an event;
    the event column alone is not one."""
    st = d.ProjectState(purpose="inference", target="died", task="time_to_event",
                        follow_up=d.FollowUpSpec(time_column="years"),
                        estimand=d.EstimandSpec(exposure="fiber", measure="hazard_ratio"))
    refused = M.offered(st)
    assert refused.says == (REFIT_FOLLOWS + "`died` with its follow-up `years` is a time to an "
                            "event, so the curve is not drawn for it.")
    assert "`died` is" not in refused.says


def test_a_multiclass_outcome_is_refused_in_the_stage_with_no_scale_exit(tmp_path):
    """Through the stage, with the task recorded and with it read from the outcome's detection
    (``target_info``): the curve's own line, never the reported row's "no single coefficient" with
    an exit about scales, which cannot help an outcome with one coefficient per class."""
    frame = cohort()
    frame["grade"] = pd.cut(frame["glucose"], 3, labels=["low", "mid", "high"]).astype(str)
    st = state(target="grade", task="multiclass", measure="relative_risk_ratio")
    run = Run(frame, tmp_path, st)
    line = (REFIT_FOLLOWS + "`grade` is an outcome of several classes, so the curve is not drawn "
            "for it.")
    for s in (st, st.model_copy(update={"task": None})):
        art = run.mattered(s)
        assert art["curve"] is None and art["refused"] == {"message": line, "exits": []}


def test_more_specifications_than_the_cap_are_refused_with_no_exit():
    """One at a time, k derived choices give 1 + k specifications; past ``spec_curve.CAP`` the
    curve is refused in its own words with no exit (every alternative is derived, so none can be
    declared away). At the cap it is silent, as spec_curve's own enumeration is."""
    under = M.adjustment_choices([f"c{i}" for i in range(S.CAP - 1)], [])
    over = M.adjustment_choices([f"c{i}" for i in range(S.CAP)], [])
    M.varied(under)
    S.enumerate_specs(under, crossing=M.CROSSING)
    with pytest.raises(S.SpecCurveRefused) as refused:
        M.varied(over)
    assert str(refused.value) == M.CAP_REFUSAL.format(n=S.CAP + 1, cap=S.CAP)
    assert f"{S.CAP + 1:,}" in str(refused.value) and refused.value.exits == []
    with pytest.raises(S.SpecCurveRefused):
        S.enumerate_specs(over, crossing=M.CROSSING)


def test_a_departure_the_fit_would_not_cluster_is_not_estimated(numeric, monkeypatch):
    """Which rows belong together is read on each specification's own rows
    (``inference.resolve_clusters``), and where that reading refuses, the fit reports no interval,
    so the departure is not estimated with the refusal as its reason. The refusal is constructed
    here, on the rows the age rule would exclude: today a reading that waits needs a role that
    waits, which stops the design stage first. The fixture's own curve reads every
    specification's rows and refuses none."""
    from turbotab.core.models import inference as I

    real = I.resolve_clusters
    said = "`pid` repeats here, and whether its rows belong together is asked, not assumed."
    seen: list[int] = []

    def resolve(st: Any, frame: pd.DataFrame, also: Any = ()) -> Any:
        seen.append(len(frame))
        if (~numeric.table.loc[frame.index, "age"].between(30, 75)).any():
            return I.Clusters(refusal=said)
        return real(st, frame, also)

    monkeypatch.setattr(I, "resolve_clusters", resolve)
    curve = numeric.mattered()["curve"]
    held = [s for s in curve["not_estimated"] if s["picks"]["exclusion:0"] == "dropped"]
    assert len(held) == 1 and said in held[0]["reason"]
    assert {s["key"] for s in curve["specs"]} == set(departures()) - {held[0]["key"]}
    assert len(set(seen)) > 2  # each specification's own rows, not the primary's alone
    monkeypatch.setattr(I, "resolve_clusters", real)
    assert numeric.mattered()["curve"]["not_estimated"] == []


def test_an_exposure_family_is_refused_and_one_exposure_is_not(numeric):
    roles = {**ef.ROLES, "supplement": "exposure"}
    family = numeric.state.model_copy(update={
        "roles": roles, "role_confirmations": dict(roles),
        "estimand": d.EstimandSpec(family=True, measure="mean_difference",
                                   multiplicity="fdr_bh")})
    art = numeric.mattered(family)
    assert art["curve"] is None and art["refused"]["message"] == M.ONE_EXPOSURE
    assert art["refused"]["exits"][0]["label"] == M.ONE_EXPOSURE_EXIT["label"]
    assert numeric.mattered()["refused"] is None  # the declared single exposure


def test_the_curve_is_refused_outside_estimate_and_offered_under_it():
    predict = d.ProjectState(purpose="prediction")
    describe = d.ProjectState(purpose="inference")
    unanswered = d.ProjectState()
    estimate = d.ProjectState(purpose="inference",
                              estimand=d.EstimandSpec(exposure="fiber", measure="mean_difference"))
    assert not M.offered(predict).offered and M.offered(predict).exits
    assert not M.offered(describe).offered and M.offered(describe).exits
    assert not M.offered(unanswered).offered and M.offered(unanswered).says == M.NOT_OFFERED_YET
    assert M.offered(estimate).offered


def test_the_reported_estimate_is_the_linear_family_s_and_refused_without_one():
    row = {"feature": "fiber", "estimate": 0.5, "ci_low": 0.1, "ci_high": 0.9, "se": 0.2}
    fit = {"models": [{"family": "linear", "coefficients": [row],
                       "inference": {"n_rows": 50, "covariance": "HC3", "refused": None}}]}
    assert M.reported_of(fit, "fiber").estimate == 0.5
    with pytest.raises(S.SpecCurveRefused, match="not among the chosen families"):
        M.reported_of({"models": [{**fit["models"][0], "family": "elastic_net"}]}, "fiber")
    with pytest.raises(S.SpecCurveRefused, match="no single coefficient of `fiber_adj`"):
        M.reported_of(fit, "fiber_adj")
    refused = {"models": [{**fit["models"][0], "inference": {"refused": "the survey question",
                                                             "exits": []}}]}
    with pytest.raises(S.SpecCurveRefused, match="the survey question"):
        M.reported_of(refused, "fiber")


@pytest.mark.parametrize("task", ["regression", "binary"])
def test_the_followed_family_is_the_tasks_declared_default_inference_family(task):
    """The curve follows the family the task's declaration names its default inference family
    (``inference_decl.default_for``; MODEL_FAMILY_CONTRACT C1: read from the declaration, never a
    switch on a family's key), least squares or logistic regression: the linear family under
    either task, its row reported with the family's key."""
    from turbotab.core.models.base import inference_default

    row = {"feature": "fiber", "estimate": 0.5, "ci_low": 0.1, "ci_high": 0.9, "se": 0.2}
    model = {"coefficients": [row], "inference": {"n_rows": 50, "covariance": "HC3",
                                                  "refused": None}}
    fit = {"models": [{**model, "family": "ridge"}, {**model, "family": "linear"}]}
    assert inference_default(task).key == "linear"
    got = M.reported_of(fit, "fiber", task)
    assert (got.family, got.estimate) == ("linear", 0.5)
    with pytest.raises(S.SpecCurveRefused, match="not among the chosen families"):
        M.reported_of({"models": [{**model, "family": "ridge"}]}, "fiber", task)


# ── the analyst's examined alternatives ──────────────────────────────────────


def test_an_examined_alternative_is_one_more_departure_and_none_adds_nothing(numeric):
    assert M.examined_of(numeric.state) == []
    plain = numeric.mattered()
    assert not [c for c in plain["curve"]["choices"] if c["key"].startswith("examined:")]
    examined = [M.MatteredExamined(key="lean", label="without smoking, with bmi", add=["bmi"],
                           drop=["smoking"])]
    curve, _ = M.curve_for(numeric.context(), examined=examined)
    assert [c.key for c in curve.specs if c.picks.get("examined:lean") == "examined"]
    s = next(s for s in curve.specs if s.picks.get("examined:lean") == "examined")
    est, low, high, n = statsmodels_fit(numeric.table, outcome="glucose", rules=(0, 1),
                                        covariates=["age", "sex", "activity", "bmi"])
    assert s.estimate == pytest.approx(est, rel=1e-9) and s.n == n
    assert s.low == pytest.approx(low, rel=1e-9) and s.high == pytest.approx(high, rel=1e-9)
    assert len(curve.specs) == len(plain["curve"]["specs"]) + 1


def test_the_methods_sentence_names_each_alternative_the_curve_derived(numeric):
    """The alternatives are the engine's own (each rule dropped, each adjusted covariate dropped,
    each left-out covariate added): the sentence names the kinds the curve holds, and says
    "declared" only of an analyst's examined alternative."""
    said = numeric.mattered()["methods"]
    assert ("each exclusion rule dropped, each covariate in the adjustment set dropped and each "
            "covariate the adjustment answers left out added") in said
    assert "declared" not in said and "one at a time" in said and "No joint test" in said

    def sentence(choices: list[S.Choice]) -> str:
        curve = S.specification_curve(choices, lambda picks: S.SpecFit(1.0, 0.5, 1.5, 50),
                                      lock="r1", estimate_label="x", crossing=M.CROSSING)
        return M.methods_sentence(curve, task="regression")

    only = sentence(M.adjustment_choices(["age"], []))
    assert "each covariate in the adjustment set dropped" in only
    assert "exclusion rule" not in only and "left out" not in only and "declared" not in only
    examined = sentence(M.examined_choices([M.MatteredExamined(key="lean", label="with bmi",
                                                               add=["bmi"])]))
    assert "each alternative the analyst declared as examined" in examined
    assert "exclusion rule" not in examined and "adjustment set dropped" not in examined


def test_the_methods_sentence_gives_each_specification_the_interval_the_fit_gives_its_rows(
        numeric, binary):
    """Each specification is refitted by the primary's estimator on its own rows, with the interval
    the fit gives those rows, which need not be the primary's: HC3 (a yes/no outcome: Wald) where
    they do not repeat within a unit, CR2 by the unit where they do. So the sentence says which
    intervals the specifications have, and never that each was fitted as the primary was. Every
    numeric specification here is HC3 and every yes/no one Wald (statsmodels' fits above); a curve
    whose specifications are HC3 and CR2 names both, each where the fit gives it."""
    said = numeric.mattered()["methods"]
    assert "as the primary" not in said and "CR2" not in said
    assert ("; each specification estimated was refitted by least squares on its own rows, and "
            "the primary remains the reported estimate. Each specification's interval is the one "
            "the fit gives its rows: heteroskedasticity-robust (HC3) where they do not repeat "
            "within a unit. No joint test") in said
    said = binary.mattered()["methods"]
    assert "as the primary" not in said and "CR2" not in said
    assert ("; each specification estimated was refitted by logistic regression on its own rows, "
            "and the primary remains the reported estimate (the log of its odds ratio). Each "
            "specification's interval is the one the fit gives its rows: the logistic model's Wald "
            "interval where they do not repeat within a unit and no column separates the "
            "outcome. No joint test") in said

    def sentence(covariances: list[str]) -> str:
        kinds = iter(covariances)
        curve = S.specification_curve(
            M.adjustment_choices(["age", "sex"], []),
            lambda picks: S.SpecFit(1.0, 0.5, 1.5, 50, covariance=next(kinds)),
            lock="r1", estimate_label="x", crossing=M.CROSSING)
        return M.methods_sentence(curve, task="regression")

    assert ("Each specification's interval is the one the fit gives its rows: heteroskedasticity-"
            "robust (HC3) where they do not repeat within a unit; cluster-robust (CR2) by the unit "
            "where they repeat within one.") in sentence(["CR2", "HC3", "CR2"])
    assert ("Each specification's interval is the one the fit gives its rows: cluster-robust "
            "(CR2) by the unit where they repeat within one.") in sentence(["CR2"] * 3)


# ── the stage: registered, keyed on what it reads, recomputed when a choice changes ──


def test_the_stage_is_registered_after_the_fit_and_locks_nothing():
    from turbotab.core.estimand import ESTIMATE_STAGES
    from turbotab.core.plan_lock import plan_slots
    from turbotab.core.stages import build_graph

    graph = build_graph()
    stage = graph[M.STAGE]
    assert {"fit", "design", "cohort", "working", "split", "target_info"} <= set(stage.deps)
    assert stage.heavy and set(M.MATTERED_REQUIRES) == set(stage.requires)
    assert "plan_locked" in stage.requires and "plan_locked" in stage.reads
    # Not an estimate stage of its own: it adds no slot to the plan the lock records.
    assert M.STAGE not in ESTIMATE_STAGES and "plan_locked" not in plan_slots()


def recorded_reads(run: Run) -> set[str]:
    """Every field of the state the stage's code reads for ``run``, through every helper."""
    seen: set[str] = set()
    fields = set(d.ProjectState.model_fields)

    class Recording(d.ProjectState):
        def __getattribute__(self, name: str) -> Any:
            if name in fields:
                seen.add(name)
            return super().__getattribute__(name)

    st = Recording(**{k: getattr(run.state, k) for k in fields})
    art = run.mattered(st)
    assert art["refused"] is None and art["curve"]["specs"]
    assert seen, "nothing was recorded"
    return seen


def test_the_key_covers_every_slot_the_stage_reads(numeric, tmp_path):
    """A recording state: every field the stage's code reads, through every helper it calls, is
    among the slots its key reads (``MATTERED_READS``): the raw refit's, and a plan's fit's (its
    copies drawn with the split's seed under multiple imputation, a spline's knots)."""
    from turbotab.core.decisions import ExposureFormSpec

    allowed = set(M.MATTERED_READS) | set(M.MATTERED_REQUIRES)
    seen = recorded_reads(numeric)
    assert seen <= allowed, seen - allowed
    plan = Run(cohort(), tmp_path, state(
        missing=d.MissingSpec(strategy="multiple_imputation"),
        exposure_forms={"age": ExposureFormSpec(form="spline", knots=4, scale="raw")}))
    seen = recorded_reads(plan)
    assert seen <= allowed, seen - allowed
    assert "split" in seen  # the seed its copies are drawn with


def test_a_changed_choice_changes_the_key_and_a_look_does_not():
    from turbotab.core.graph import stage_keys
    from turbotab.core.stages import build_graph

    graph = build_graph()
    order = graph.order()
    st = state()

    def key(s: d.ProjectState) -> str | None:
        return stage_keys(order, s.model_dump(mode="json"), "fingerprint")[M.STAGE]

    assert key(st) is not None
    assert key(st.model_copy(update={"plan_locked": None})) is None  # blocked before the lock
    narrower = st.model_copy(update={"exclusions": [d.ExclusionRule(
        column="age", low=35, high=70, reason="a narrower range")]})
    assert key(narrower) != key(st)
    looked = st.model_copy(update={"outcome_views": {"distribution:glucose": d.OutcomeViewSpec(
        view="distribution", column="glucose")}})
    assert key(looked) == key(st)


def test_a_stale_curve_is_recomputed_for_the_changed_choice(numeric, tmp_path):
    narrower = numeric.state.model_copy(update={"exclusions": [
        d.ExclusionRule(column="age", low=35, high=70, reason="a narrower range"),
        d.ExclusionRule(**RULES[1])]})
    again = Run(cohort(), tmp_path, narrower)
    art = again.mattered()
    primary = next(s for s in art["curve"]["specs"] if s["primary"])
    assert primary["estimate"] == pytest.approx(again.reported()["estimate"], rel=1e-12)
    keep = (again.table["age"].between(35, 70)
            & (again.table["bmi"].isna() | (again.table["bmi"] <= 33))
            & again.table[["glucose", "fiber", *ADJUSTED]].notna().all(axis=1))
    assert primary["n"] == int(keep.sum()) != next(
        s for s in numeric.mattered()["curve"]["specs"] if s["primary"])["n"]

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
    worker runs them (every analyzed row, no held-out rows: inference)."""

    def __init__(self, frame: pd.DataFrame, folder: Path, st: d.ProjectState, *,
                 fit: Any = None):
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
        split = mf.split_bundle(self.kept, holdout=0.0, seed=st.split.seed)
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
    reproduces it, so no specification is shown. The fixture's own fit passes (the tests above)."""
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
    assert "would not be fitted as the primary is" in art["refused"]["message"]
    assert numeric.mattered()["refused"] is None  # the fit's own row: silent


def test_agreement_is_judged_against_the_standard_error_both_ways():
    rep = M.MatteredReported(family="linear", feature="x", estimate=1.0, ci_low=0.5, ci_high=1.5, se=0.25,
                     n_rows=100, covariance="HC3")
    near = 0.5 * M.AGREEMENT * 0.25
    assert M.agrees(S.Spec("p", {}, True, 1.0 + near, 0.5 - near, 1.5, 100), rep)
    far = 2 * M.AGREEMENT * 0.25
    assert not M.agrees(S.Spec("p", {}, True, 1.0 + far, 0.5, 1.5, 100), rep)
    assert not M.agrees(S.Spec("p", {}, True, 1.0, 0.5, 1.5 + far, 100), rep)
    assert not M.agrees(S.Spec("p", {}, True, 1.0, 0.5, 1.5, 99), rep)  # other rows


def test_a_fill_of_missing_values_refuses_the_curve_and_complete_cases_do_not(tmp_path):
    """Under complete cases each specification's rows are its own complete cases, as the flow
    judges the primary's (the tests above have blanks); a single fill would fill them instead,
    which the refit does not, so it refuses by name before anything is fitted."""
    filled = state(missing=d.MissingSpec(strategy="impute", acknowledged=True,
                                         reason="a test of the refusal"))
    run = Run(cohort(), tmp_path, filled)
    art = run.mattered()
    assert art["curve"] is None
    assert "the fill of missing values" in art["refused"]["message"]


def test_a_declared_spline_refuses_the_curve_by_name(tmp_path):
    from turbotab.core.decisions import ExposureFormSpec

    formed = state(exposure_forms={"age": ExposureFormSpec(form="spline", knots=4, scale="raw")})
    run = Run(cohort(), tmp_path, formed)
    assert run.design.objects["spec"]["exposure_forms"]  # the design applies it
    art = run.mattered()
    assert art["curve"] is None
    assert "a declared form other than a straight line" in art["refused"]["message"]


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


def test_the_key_covers_every_slot_the_stage_reads(numeric):
    """A recording state: every field the stage's code reads, through every helper it calls, is
    among the slots its key reads (``MATTERED_READS``)."""
    seen: set[str] = set()
    fields = set(d.ProjectState.model_fields)

    class Recording(d.ProjectState):
        def __getattribute__(self, name: str) -> Any:
            if name in fields:
                seen.add(name)
            return super().__getattribute__(name)

    st = Recording(**{k: getattr(numeric.state, k) for k in fields})
    art = numeric.mattered(st)
    assert art["refused"] is None and art["curve"]["specs"]
    assert seen, "nothing was recorded"
    assert seen <= set(M.MATTERED_READS) | set(M.MATTERED_REQUIRES), seen - set(M.MATTERED_READS)


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

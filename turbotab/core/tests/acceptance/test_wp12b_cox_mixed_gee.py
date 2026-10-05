"""WP12b · Methods a reviewer expects: the Cox model, the random-intercept mixed model and GEE
(docs/turbotab-next/audit/AUDIT_REPORT.md §5 WP12, acceptance tests 3 and 4; the family RO-03
routes to, and the exits WP2 left for MA-01's repeated and few-unit designs).

The package's two acceptance tests, each with the checks that make its reference result a
property of the method rather than of one fixture:

* **3 · Cox on the staggered-entry fixture: HR 1.012 (0.997–1.027).** The audit's cohort
  (I/make_tte_null.py, seed 7: enrolment spread over 14 years, later enrollees eat more fiber and
  are followed for less time, fiber has no effect on the hazard) through the real split, design and
  fit stages, against lifelines' ``CoxPHFitter`` (Efron ties, the default of R's ``coxph``). The
  engine is also checked against lifelines with delayed entry and heavily tied times, its
  cluster-robust intervals against the infinitesimal jackknife built from lifelines' weighted refits
  (the definition of R's ``coxph(cluster = …)`` variance), its proportional-hazards check against
  lifelines' ``proportional_hazard_test``, and its C-index against lifelines'
  ``concordance_index``.
* **4 · Random-intercept mixed model on the 6 × 40 fixture: sodium p ≈ 0.85.** The audit's server
  scenario (E/scenA.py, seed 11), driven through the real HTTP API: the linear model's table is
  refused below the unit floor, its exit selects the mixed model, and the mixed model reports
  p = 0.85 (statsmodels ``MixedLM``, REML, on the same rows: 0.852). The REML fit is checked against
  the likelihood written out with dense matrices, the Satterthwaite degrees of freedom against the
  classical ANOVA degrees of freedom they must reproduce in a balanced design (derived below) and
  against the definition computed densely on an unbalanced one, and the intervals' type-I error at
  six units by Monte Carlo.

GEE, the other exit, is checked against statsmodels' own GEE sandwich and the CR2 definition
(``references.cr2_by_definition``) on a differently whitened working model, refused below the unit
floor with the mixed model as its exit, and held to its type-I error at twelve units.

R is not installed. Where the package names R (``survival::coxph``, ``lme4``/``lmerTest``), the
reference is lifelines, statsmodels, a closed form, or the published definition computed by a
different route, as each test's docstring says. Monte Carlo runs are seeded; each states its
Monte Carlo standard error against the bound it checks.
"""
from __future__ import annotations

import math
import time
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from turbotab.core.decisions import (FollowUpSpec, GrainSpec, ProjectState, Refusal, SelectModels,
                                     SplitSpec, validate)
from turbotab.core.models import Situation, rank
from turbotab.core.models.inference import Clusters, bread_of, cr2, floor_refusal, min_clusters
from turbotab.core.models.repeated import (fit_gee, fit_random_intercept, gee_table, gee_working,
                                           mixed_table)
from turbotab.core.models.survival import (concordance, cox_fit, cox_table, ph_test,
                                           survival_outcome)
from turbotab.core.stages.modeling import design_stage, fit_stage, resampled_units, shelf_stage
from turbotab.core.stages.rows import split_stage
from turbotab.core.tests import modeling_fixtures as mf
from turbotab.core.tests.acceptance import references as ref
from turbotab.core.tests.acceptance.server_drive import answer_plan


def _clusters(codes: Any, column: str = "pid") -> Clusters:
    codes = pd.factorize(pd.Series(np.asarray(codes)))[0].astype(np.int64)
    return Clusters(column=column, codes=codes, n_clusters=int(codes.max()) + 1)


def _row(rows: list[dict[str, Any]], feature: str) -> dict[str, Any]:
    return next(r for r in rows if r["feature"] == feature)


def _stages(frame: pd.DataFrame, st: ProjectState, predictors: list[str], task: str,
            tmp_path: Any) -> tuple[Any, Any, Any]:
    """The real split, shelf, design and fit stages over these rows, as a worker runs them."""
    paths = mf.ingest_frame(frame, tmp_path)
    ids = np.arange(len(frame))
    ti = mf.target_info(task, st.target)
    cohort = mf.cohort_bundle(ids, predictors)
    split = split_stage(mf.context(st, {"cohort": cohort, "target_info": ti}, paths))
    shelf = shelf_stage(mf.context(st, {"cohort": cohort, "split": split, "target_info": ti}, paths))
    design = design_stage(mf.context(st, {"split": split, "target_info": ti}, paths))
    fit = fit_stage(mf.context(st, {"design": design, "split": split, "target_info": ti}, paths))
    return shelf, design, fit


# ── 3 · the Cox model ────────────────────────────────────────────────────────


def _staggered_entry_cohort() -> pd.DataFrame:
    """The audit's fixture (I/make_tte_null.py, seed 7), written as its CSV writes it: 3,000 people
    enrolled over years 0–14 of a study that closes at year 15, so later enrollees are followed for
    less time; fiber rises with the enrolment year, and has no effect on the hazard (HR = 1)."""
    rng = np.random.default_rng(7)
    n = 3000
    entry = rng.uniform(0, 14, n)
    fiber = (15 + 1.0 * entry + rng.normal(0, 4, n)).clip(1)
    age = rng.normal(55, 8, n)
    hazard = 0.03 * np.exp(0.04 * (age - 55))
    t_event = rng.exponential(1 / hazard)
    closes = 15 - entry
    return pd.DataFrame({"participant_id": [f"P{i:05d}" for i in range(n)], "age": age.round(1),
                         "fiber_g": fiber.round(2),
                         "followup_years": np.minimum(t_event, closes).round(3),
                         "cvd_event": (t_event <= closes).astype(int)})


def _lifelines(frame: pd.DataFrame, duration: str, event: str, covariates: list[str], *,
               entry: str | None = None, weights: str | None = None, cluster: str | None = None) -> Any:
    from lifelines import CoxPHFitter

    columns = [*covariates, duration, event] + [c for c in (entry, weights, cluster) if c]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # lifelines asks for robust=True with non-integer weights
        return CoxPHFitter().fit(frame[columns], duration, event, entry_col=entry,
                                 weights_col=weights, cluster_col=cluster, robust=False,
                                 fit_options={"precision": 1e-12, "max_steps": 200})


def test_3_cox_on_the_staggered_entry_fixture_matches_lifelines(tmp_path):
    """AUDIT_REPORT §5 WP12 test 3: Cox on the staggered-entry fixture, HR 1.012 (0.997–1.027).

    Through the real split, shelf, design and fit stages under inference (every eligible row
    estimates, BLUEPRINT §12 ruling 3), the outcome `cvd_event` with its follow-up
    `followup_years` (``set_follow_up``). Reference: lifelines ``CoxPHFitter`` on the same rows
    (Efron ties; lifelines documents its estimates as matching R's ``coxph``), estimates and
    standard errors to 10⁻⁶ relative; the 95% Wald interval exp(β ± 1.96·SE) as ``coxph`` prints
    it. The shelf offers no other family: a censored outcome is never served as a log-odds."""
    frame = _staggered_entry_cohort()
    st = ProjectState(lens=["clinical"], target="cvd_event", task="time_to_event",
                      purpose="inference", missing="complete_case",
                      roles={"participant_id": "identifier", "fiber_g": "exposure",
                             "age": "covariate", "followup_years": "time"},
                      split=SplitSpec(holdout=0.0, seed=0, folds=5), models=["cox"],
                      grain=GrainSpec(grain="one_row_per_unit"),
                      follow_up=FollowUpSpec(time_column="followup_years"))
    shelf, _, fit = _stages(frame, st, ["fiber_g", "age"], "time_to_event", tmp_path)
    assert [f["key"] for f in shelf["families"]] == ["cox"]
    assert "658 with the event" in shelf["basis"]
    model = fit.data["models"][0]
    info = model["inference"]
    assert info["covariance"] == "model" and "Efron ties" in info["caption"]
    assert "log hazard ratios" in info["caption"]
    assert fit.data["n_train"] == 3000 and fit.data["primary_metric"] == "c_index"

    reference = _lifelines(frame, "followup_years", "cvd_event", ["fiber_g", "age"])
    for name in ("fiber_g", "age"):
        row = _row(model["coefficients"], name)
        assert row["estimate"] == pytest.approx(reference.params_[name], rel=1e-6)
        assert row["se"] == pytest.approx(reference.standard_errors_[name], rel=1e-6)
        assert row["p"] == pytest.approx(reference.summary.loc[name, "p"], rel=1e-5)
    fiber = _row(model["coefficients"], "fiber_g")
    hr, low, high = (math.exp(fiber[k]) for k in ("estimate", "ci_low", "ci_high"))
    print(f"\nfiber HR {hr:.4f} ({low:.4f}–{high:.4f}), p = {fiber['p']:.3f}; lifelines "
          f"{reference.summary.loc['fiber_g', 'exp(coef)']:.4f} "
          f"({reference.summary.loc['fiber_g', 'exp(coef) lower 95%']:.4f}–"
          f"{reference.summary.loc['fiber_g', 'exp(coef) upper 95%']:.4f})")
    assert (round(hr, 3), round(low, 3), round(high, 3)) == (1.012, 0.997, 1.027)
    assert low == pytest.approx(reference.summary.loc["fiber_g", "exp(coef) lower 95%"], rel=1e-6)
    assert high == pytest.approx(reference.summary.loc["fiber_g", "exp(coef) upper 95%"], rel=1e-6)
    # Repair round (WP12b × WP8): the table declares the hazard-ratio scale, drawn on a log axis,
    # and carries each hazard ratio and its interval, lifelines' exp(coef) and its 95% limits.
    assert info["scale"] == "hazard_ratio" and info["axis"] == "log"
    assert "Hazard ratio of `cvd_event`" in info["effect"]
    assert fiber["ratio"] == pytest.approx(reference.summary.loc["fiber_g", "exp(coef)"], rel=1e-6)
    assert fiber["ratio_low"] == pytest.approx(low, rel=1e-12)
    assert fiber["ratio_high"] == pytest.approx(high, rel=1e-12)
    assert fiber["p"] == pytest.approx(0.12, abs=0.01)
    # Cross-validated Harrell's C beats one risk for everyone (age carries the hazard).
    assert model["cv"]["c_index"]["estimate"] > 0.55
    assert model["baseline"]["value"] == 0.5 and model["versus_baseline"]["verdict"] == "better"


def test_3_a_time_to_event_outcome_gets_only_a_family_that_declares_it():
    """RO-03 at the method layer: the families that model a yes/no outcome do not model a
    time-to-event one. The shelf offers only the Cox model, and choosing the linear (logistic)
    family for such an outcome is refused, so the audit's p ≈ 10⁻¹³ log-odds cannot be fit."""
    ranked = rank(Situation(task="time_to_event", purpose="inference", n_rows=3000, n_features=2,
                            n_events=658))
    assert [f.key for f, _ in ranked] == ["cox"]
    with pytest.raises(Refusal) as refused:
        validate(SelectModels(models=["linear", "cox"]), {"task": "time_to_event"})
    assert "cannot model a time-to-event outcome" in refused.value.message
    assert validate(SelectModels(models=["cox"]), {"task": "time_to_event"})


def test_3_a_follow_up_is_refused_while_the_outcome_is_not_a_time_to_event():
    """Repair round (verifier, RO-03 reproduced): the task question is skipped as binary for a 0/1
    outcome, and ``set_follow_up`` was accepted beside it; the record then said "`cvd_event` was
    analyzed as a time to event" over a logistic fit (fiber log-odds −0.069, p = 1.7 × 10⁻¹⁷).

    The follow-up is refused while the task, answered or detected, is anything but time to event,
    with the exit that sets it; the sentences say what is used when a follow-up stands beside
    another task. Reference: the task itself (the families that read a follow-up are the
    time-to-event ones, test above)."""
    from turbotab.core import voice
    from turbotab.core.decisions import SetFollowUp, SetTask, parse_decision

    follow = {"kind": "set_follow_up", "column": "cvd_event", "time_column": "followup_years"}
    for task in ("binary", "regression"):
        with pytest.raises(Refusal) as refused:
            validate(follow, {"target": "cvd_event", "task": task})
        assert refused.value.code == "not_time_to_event"
        exit_ = parse_decision(refused.value.exits[0]["decision"])
        assert isinstance(exit_, SetTask) and exit_.task == "time_to_event"
        assert exit_.column == "cvd_event"
    validate(follow, {"target": "cvd_event", "task": "time_to_event"})

    binary = ProjectState(target="cvd_event", task="binary")
    said = voice.sentence_for(SetFollowUp(**{k: v for k, v in follow.items() if k != "kind"}), binary)
    assert "analyzed as a time to event" not in said and "not used" in said
    standing = ProjectState(target="cvd_event", task="time_to_event",
                            follow_up=FollowUpSpec(time_column="followup_years"))
    said = voice.sentence_for(SetTask(column="cvd_event", task="binary"), standing)
    assert "`followup_years` recorded for it is not used" in said


def test_3_the_cox_model_is_reached_through_the_server(tmp_path):
    """The same fixture driven through the real HTTP API: the outcome is declared a time to event
    (``set_task``), its event level named (the event question, asked for a time-to-event outcome as
    for a binary one), its follow-up named (``set_follow_up``, refused while it would name the
    outcome itself), and the Cox model chosen. The fit artifact carries the hazard ratio the
    stages computed above, and the decision record says what was done in a methods sentence."""
    frame = _staggered_entry_cohort()
    client, path = _server(tmp_path, frame, "cohort_tte_null")
    with client:
        r = client.post("/api/projects", json={"path": str(path)})
        assert r.status_code == 200, r.text
        d = _Drive(client, r.json()["id"], _age_truth())
        d.artifact("ingest")
        d.decide({"kind": "set_lens", "lenses": ["clinical"]})
        d.reach("target")
        d.decide({"kind": "set_target", "column": "cvd_event"})
        d.reach("task")
        assert d.artifact("target_info")["task"] == "binary"  # 0/1: detected as yes/no
        early = client.post(f"/api/projects/{d.pid}/decisions",
                            json={"kind": "set_follow_up", "column": "cvd_event",
                                  "time_column": "followup_years"})
        assert early.status_code == 409 and "not_time_to_event" in early.text
        d.decide({"kind": "set_task", "column": "cvd_event", "task": "time_to_event"})
        d.answer("event", {"kind": "set_event", "column": "cvd_event", "level": "1"})
        wrong = client.post(f"/api/projects/{d.pid}/decisions",
                            json={"kind": "set_follow_up", "column": "cvd_event",
                                  "time_column": "cvd_event"})
        assert wrong.status_code == 409 and "follow_up_is_the_outcome" in wrong.text
        d.decide({"kind": "set_follow_up", "column": "cvd_event", "time_column": "followup_years"})
        d.reach("purpose")
        d.decide({"kind": "set_purpose", "purpose": "inference"})
        d.answer("grain", {"kind": "set_grain", "grain": "one_row_per_unit"})
        d.reach("roles")
        d.decide({"kind": "set_roles", "roles": {"participant_id": "identifier",
                                                 "fiber_g": "exposure", "age": "covariate",
                                                 "followup_years": "time"}})
        d.reach("exclusions")
        d.decide({"kind": "set_exclusions", "rules": []})
        d.reach("missing")
        d.decide({"kind": "set_missing", "strategy": "complete_case"})
        d.reach("split")
        d.decide({"kind": "set_split", "holdout": 0.0, "seed": 0, "folds": 5})
        # WP17, after the split (MODELING_SEQUENCE §1 steps 2–3): the exposure, its effect and the
        # adjustment set
        answer_plan(d, "fiber_g")
        d.answer("energy_adjustment", {"kind": "set_energy_adjustment", "method": "none"})
        d.reach("models")
        refused = client.post(f"/api/projects/{d.pid}/decisions",
                              json={"kind": "select_models", "models": ["linear"]})
        assert refused.status_code == 409 and "time-to-event" in refused.text
        d.decide({"kind": "select_models", "models": ["cox"]})
        fit = d.artifact("fit")
        view = d.view()
    assert view["state"]["follow_up"] == {"time_column": "followup_years", "entry_column": None}
    said = next(r["sentence"] for r in view["decisions"]
                if r["decision"]["kind"] == "set_follow_up")
    assert said.startswith("`cvd_event` was analyzed as a time to event, each row followed until "
                           "`followup_years`"), said
    model = fit["models"][0]
    assert fit["task"] == "time_to_event" and model["family"] == "cox"
    fiber = _row(model["coefficients"], "fiber_g")
    assert round(math.exp(fiber["estimate"]), 3) == 1.012
    assert (round(math.exp(fiber["ci_low"]), 3), round(math.exp(fiber["ci_high"]), 3)) == (0.997, 1.027)
    assert model["inference"]["scale"] == "hazard_ratio" and round(fiber["ratio"], 3) == 1.012


def _tied_delayed_entry(seed: int = 3, n: int = 800, G: int = 40) -> pd.DataFrame:
    """Rows clustered in G units with a shared frailty, half entering late (left truncation), and
    times and entries rounded to half-years so many events tie and entries fall on event times."""
    rng = np.random.default_rng(seed)
    g = rng.integers(0, G, n)
    x1 = rng.normal(size=n) + rng.normal(size=G)[g]
    x2 = rng.binomial(1, 0.4, n).astype(float)
    frailty = np.exp(0.5 * rng.normal(size=G))[g]
    t = rng.exponential(1 / (0.1 * frailty * np.exp(0.3 * x1 - 0.4 * x2)))
    entry = rng.uniform(0, 3, n) * (rng.random(n) < 0.5)
    censor = entry + rng.uniform(0.5, 12, n)
    keep = t > entry  # observed only if the event comes after entry
    frame = pd.DataFrame({"x1": x1, "x2": x2, "unit": g, "t": t, "entry": entry,
                          "censor": censor})[keep]
    frame["time"] = np.ceil(np.minimum(frame["t"], frame["censor"]) * 2) / 2
    frame["event"] = (frame["t"] <= frame["censor"]).astype(int)
    frame["entry"] = np.floor(frame["entry"] * 2) / 2
    return frame[frame["time"] > frame["entry"]].reset_index(drop=True)


def test_3_delayed_entry_and_tied_times_match_lifelines():
    """A row entering late is never at risk before it entered (risk set: entry < t ≤ time), and
    tied event times follow Efron's approximation. References: lifelines ``CoxPHFitter`` with
    ``entry_col`` (left truncation) on the same rows, partial log-likelihood, estimates and model
    standard errors to 10⁻⁶ relative; and statsmodels' ``PHReg`` (``entry``, Efron ties), whose
    risk set counts a row at an event time equal to its entry, so it is given each entry plus
    10⁻⁷ to state the same risk sets (unshifted, it differs in the third decimal here)."""
    frame = _tied_delayed_entry()
    assert frame["time"][frame["event"] == 1].duplicated().mean() > 0.9  # heavily tied
    assert (frame["entry"] > 0).mean() > 0.3
    X = frame[["x1", "x2"]].to_numpy(float)
    y = survival_outcome(frame["event"], frame["time"], frame["entry"])
    fit = cox_fit(X, y)
    reference = _lifelines(frame, "time", "event", ["x1", "x2"], entry="entry")
    assert fit.converged
    np.testing.assert_allclose(fit.beta, reference.params_.to_numpy(), rtol=1e-6)
    np.testing.assert_allclose(np.sqrt(np.diag(fit.cov)), reference.standard_errors_.to_numpy(),
                               rtol=1e-6)
    assert fit.loglik == pytest.approx(reference.log_likelihood_, rel=1e-9)
    from statsmodels.duration.hazard_regression import PHReg

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        other = PHReg(frame["time"].to_numpy(), X, status=frame["event"].to_numpy(),
                      entry=frame["entry"].to_numpy() + 1e-7, ties="efron").fit()
    np.testing.assert_allclose(fit.beta, other.params, rtol=1e-8)
    np.testing.assert_allclose(np.sqrt(np.diag(fit.cov)), other.bse, rtol=1e-6)


def test_3_cluster_robust_intervals_are_the_infinitesimal_jackknife():
    """Rows that repeat within a unit get the Lin–Wei sandwich by unit (Lin & Wei 1989), R
    ``coxph(cluster = …)``'s robust variance: D'D, with D the units' dfbetas (the inverse
    information times each unit's summed score residuals). That is the infinitesimal jackknife:
    D's rows are the derivatives of the estimate with respect to each unit's case weight.

    Reference: those derivatives computed outside the engine, by central differences of lifelines'
    weighted Efron fits (each unit's weight moved by ±10⁻⁴; lifelines gives a tied event set the
    mean weight of its rows), with delayed entry and heavy ties, where lifelines' own robust
    variance does not apply (its score-residual code carries the note "doesn't handle ties" and
    takes no entry times). On untied rows with no late entry, lifelines' own clustered variance is
    a second reference. The table uses t(G − 1) (Cameron & Miller 2015)."""
    frame = _tied_delayed_entry()
    X = frame[["x1", "x2"]].to_numpy(float)
    y = survival_outcome(frame["event"], frame["time"], frame["entry"])
    clusters = _clusters(frame["unit"], "unit")
    table = cox_table(frame[["x1", "x2"]], y, clusters)
    se = np.array([_row(table.rows, n)["se"] for n in ("x1", "x2")])

    base = frame[["x1", "x2", "time", "event", "entry"]].copy()
    codes = clusters.codes
    h = 1e-4
    dfbeta = np.zeros((clusters.n_clusters, 2))
    for k in range(clusters.n_clusters):
        moved = []
        for step in (h, -h):
            base["w"] = 1.0 + step * (codes == k)
            moved.append(_lifelines(base, "time", "event", ["x1", "x2"], entry="entry",
                                    weights="w").params_.to_numpy())
        dfbeta[k] = (moved[0] - moved[1]) / (2 * h)
    jackknife = np.sqrt(np.diag(dfbeta.T @ dfbeta))
    print(f"\nLin–Wei SE {se}; infinitesimal jackknife (lifelines refits) {jackknife}")
    np.testing.assert_allclose(se, jackknife, rtol=1e-6)
    G = clusters.n_clusters
    info = table.info
    assert info["covariance"] == "CR0" and f"G = {G:,}" in info["caption"]
    assert f"t({G - 1:,})" in info["caption"]
    x1 = _row(table.rows, "x1")
    q = stats.t.ppf(0.975, G - 1)
    assert x1["ci_high"] - x1["estimate"] == pytest.approx(q * x1["se"], rel=1e-12)
    assert x1["df"] == G - 1

    # Untied, no late entry: lifelines' own clustered variance.
    rng = np.random.default_rng(11)
    simple = _tied_delayed_entry(seed=4).assign(time=lambda f: f["time"] + rng.uniform(0, 0.4, len(f)),
                                               entry=0.0)
    table = cox_table(simple[["x1", "x2"]], survival_outcome(simple["event"], simple["time"]),
                      _clusters(simple["unit"], "unit"))
    reference = _lifelines(simple, "time", "event", ["x1", "x2"], cluster="unit")
    np.testing.assert_allclose([_row(table.rows, n)["se"] for n in ("x1", "x2")],
                               reference.standard_errors_.to_numpy(), rtol=1e-6)


def _crossing_hazards(seed: int, n: int = 1500, ph: bool = True) -> pd.DataFrame:
    """A binary exposure whose hazard ratio is 2 throughout (``ph``), or 4 early and ¼ late."""
    rng = np.random.default_rng(seed)
    x = rng.binomial(1, 0.5, n).astype(float)
    age = rng.normal(size=n)
    base = 0.2 * np.exp(0.3 * age)
    if ph:
        t = rng.exponential(1 / (base * np.where(x == 1, 2.0, 1.0)))
    else:  # piecewise: before t = 2 the exposed have 4× the hazard, after it ¼
        early = rng.exponential(1 / (base * np.where(x == 1, 4.0, 1.0)))
        late = 2.0 + rng.exponential(1 / (base * np.where(x == 1, 0.25, 1.0)))
        t = np.where(early < 2.0, early, late)
    censor = rng.uniform(1, 10, n)
    return pd.DataFrame({"x": x, "age": age, "time": np.minimum(t, censor),
                         "event": (t <= censor).astype(int)})


@pytest.mark.parametrize("ph", [True, False], ids=["proportional", "crossing"])
def test_3_the_proportional_hazards_check_matches_lifelines_and_speaks_only_when_it_fails(ph):
    """Grambsch & Therneau's (1994) approximate test on scaled Schoenfeld residuals against the
    event order. Reference: lifelines ``proportional_hazard_test(time_transform="rank")``, the
    same statistic, p-values to 10⁻⁶ relative. The table raises a concern on the crossing-hazards
    fixture and stays silent when the hazard ratio is constant; the C-index equals lifelines'
    ``concordance_index`` (Harrell's C) on the same risk score."""
    from lifelines.statistics import proportional_hazard_test
    from lifelines.utils import concordance_index

    frame = _crossing_hazards(seed=21, ph=ph)
    X = frame[["x", "age"]].to_numpy(float)
    y = survival_outcome(frame["event"], frame["time"])
    fit = cox_fit(X, y)
    reference = _lifelines(frame, "time", "event", ["x", "age"])
    test = proportional_hazard_test(reference, frame[["x", "age", "time", "event"]],
                                    time_transform="rank")
    np.testing.assert_allclose(ph_test(X, y, fit), test.summary.loc[["x", "age"], "p"].to_numpy(),
                               rtol=1e-6)
    table = cox_table(frame[["x", "age"]], y, Clusters())
    flagged = [c for c in table.concerns if "may change over follow-up" in c]
    assert bool(flagged) is (not ph), table.concerns
    if flagged:
        assert "`x`" in flagged[0]
    risk = X @ fit.beta
    assert concordance(frame["time"], frame["event"], risk) == pytest.approx(
        concordance_index(frame["time"], -risk, frame["event"]), abs=1e-12)


def test_3_harrells_c_counts_ties_as_lifelines_does():
    """Tied times, tied risk scores and censoring at an event's time: the C-index equals lifelines'
    ``concordance_index`` (which takes a survival-ordered score, so the negated risk), and one
    risk score for everyone is ½."""
    from lifelines.utils import concordance_index

    rng = np.random.default_rng(8)
    time_ = rng.integers(1, 12, 400).astype(float)
    event = rng.binomial(1, 0.6, 400)
    risk = rng.integers(0, 5, 400).astype(float) - 0.1 * time_
    assert concordance(time_, event, risk) == pytest.approx(
        concordance_index(time_, -risk, event), abs=1e-12)
    assert concordance(time_, event, np.zeros(400)) == pytest.approx(0.5)


# ── 4 · the random-intercept mixed model ─────────────────────────────────────


def _six_by_forty() -> pd.DataFrame:
    """The audit's server scenario (E/scenA.py, seed 11): 6 participants × 40 visits, strong
    participant intercepts, sodium varying mostly between participants, and no sodium effect."""
    rng = np.random.default_rng(11)
    units, per = 6, 40
    pid = np.repeat(np.arange(units) + 1001, per)
    u_x = rng.normal(0, 1, units)
    u_y = rng.normal(0, 2, units)
    x = np.repeat(u_x, per) + rng.normal(0, 0.3, units * per)
    age = np.repeat(rng.integers(30, 70, units), per).astype(float)
    y = 5 + np.repeat(u_y, per) + rng.normal(0, 1, units * per)
    return pd.DataFrame({"participant_id": pid, "sodium_mg": x * 500 + 3000, "age": age,
                         "sbp": y * 3 + 110})


# These fixtures' truth (BLUEPRINT §14.3): age is drawn in whole years, an amount.
def _age_truth() -> Any:
    from turbotab.core.tests.truths import Truth

    # WP17: age is drawn before the exposure and causes the outcome (and, in the cohort, the hazard).
    return Truth({"code_or_count:age": "amount", "adjust:age": "yes,yes,no"},
                 fixture="the audit's server scenario")


class _Drive:
    """Answers the opening sequence through the real HTTP API, in the Router's order."""

    def __init__(self, client: Any, pid: str, truth: Any = None):
        from turbotab.core.tests.truths import Truth

        self.c, self.pid = client, pid
        # The fixture's declared truth answers the readings asked (BLUEPRINT §14.3); the roles
        # the test records are its truth for the role readings.
        self.truth = truth if truth is not None else Truth(fixture="this fixture")

    def view(self) -> dict[str, Any]:
        return self.c.get(f"/api/projects/{self.pid}").json()

    def decide(self, body: dict[str, Any]) -> None:
        from turbotab.core.tests.acceptance.server_drive import settle_post

        if body.get("kind") == "set_roles":
            for column, role in body["roles"].items():
                self.truth.setdefault(f"role:{column}", role)
        r = settle_post(self.c, self.pid, body, self.truth)  # the readings asked, from the truth
        assert r.status_code == 200, (body["kind"], r.text[:600])

    def reach(self, key: str, timeout: float = 120.0) -> dict[str, Any]:
        end = time.monotonic() + timeout
        while True:
            steps = self.view()["interview"]
            step = next(s for s in steps if s["key"] == key)
            first = next((s for s in steps if s["status"] in ("open", "waiting")), None)
            ready = step["status"] not in ("open", "waiting") or first is None or first["key"] == key
            if ready and not (step["status"] == "waiting" and step.get("waiting_on")):
                return step
            assert time.monotonic() < end, f"{key} held behind {first}"
            time.sleep(0.05)

    def answer(self, key: str, body: dict[str, Any]) -> None:
        """Answer ``key`` when the Router asks it (it may skip it, or find it not applicable)."""
        if self.reach(key)["status"] in ("open", "waiting"):
            self.decide(body)

    def artifact(self, stage: str, timeout: float = 120.0) -> dict[str, Any]:
        end = time.monotonic() + timeout
        while True:
            status = self.view()["stages"][stage]
            if status["status"] == "fresh":
                return self.c.get(f"/api/projects/{self.pid}/stages/{stage}").json()["artifact"]
            assert status["status"] != "error", status
            assert time.monotonic() < end, f"{stage} never fresh: {status}"
            time.sleep(0.05)


def _server(tmp_path: Any, frame: pd.DataFrame, name: str) -> tuple[Any, Any]:
    from fastapi.testclient import TestClient

    from turbotab.core.config import Settings
    from turbotab.server.app import create_app

    path = tmp_path / f"{name}.csv"
    frame.to_csv(path, index=False)
    settings = Settings(home=tmp_path / "home", mode="local", workers=2, memory_budget_bytes=1 << 30)
    client = TestClient(create_app(settings, frontend_dist=tmp_path / "none"),
                        base_url="http://127.0.0.1")
    return client, path


def test_4_the_six_unit_exit_is_a_mixed_model_with_sodium_p_085(tmp_path):
    """AUDIT_REPORT §5 WP12 test 4: random-intercept mixed model on the 6 × 40 fixture, sodium
    p ≈ 0.85; and WP2 test 2's "mixed model, once WP12 adds it", wired.

    Driven through the real server as the audit drove it. The linear model's table is refused
    below the unit floor; one of its exits is a decision, ``select_models: ["mixed"]``, and
    recording it fits the mixed model, whose table reports sodium p = 0.85 on t with Satterthwaite
    degrees of freedom. References on the same 240 rows: statsmodels ``MixedLM`` fit by REML (its
    Wald z test gives 0.852; its fixed effects and variance components match to its optimizer's
    tolerance); independent-row OLS, the table the audit found, p ≈ 4 × 10⁻¹⁴."""
    import statsmodels.formula.api as smf

    frame = _six_by_forty()
    client, path = _server(tmp_path, frame, "six_by_forty")
    with client:
        r = client.post("/api/projects", json={"path": str(path)})
        assert r.status_code == 200, r.text
        d = _Drive(client, r.json()["id"], _age_truth())
        d.artifact("ingest")
        d.decide({"kind": "set_lens", "lenses": ["clinical"]})
        d.reach("target")
        d.decide({"kind": "set_target", "column": "sbp"})
        d.answer("task", {"kind": "set_task", "column": "sbp", "task": "regression"})
        d.reach("purpose")
        d.decide({"kind": "set_purpose", "purpose": "inference"})
        d.reach("grain")
        d.decide({"kind": "set_grain", "grain": "repeated", "id_column": "participant_id"})
        d.answer("repeat_kind", {"kind": "set_repeat_kind", "repeat_kind": "repeats"})
        d.answer("unit", {"kind": "set_unit", "unit": "row"})
        d.answer("temporal", {"kind": "set_temporal", "temporal": False})
        d.reach("roles")
        d.decide({"kind": "set_roles", "roles": {"participant_id": "identifier",
                                                 "sodium_mg": "exposure", "age": "covariate"}})
        d.reach("exclusions")
        d.decide({"kind": "set_exclusions", "rules": []})
        d.reach("missing")
        d.decide({"kind": "set_missing", "strategy": "complete_case"})
        d.reach("split")
        d.decide({"kind": "set_split", "holdout": 0.0, "seed": 0, "folds": 5})
        # WP17, after the split (MODELING_SEQUENCE §1 steps 2–3): the exposure, its effect and the
        # adjustment set
        answer_plan(d, "sodium_mg")
        d.answer("energy_adjustment", {"kind": "set_energy_adjustment", "method": "none"})
        d.reach("models")
        shelf = d.artifact("shelf")
        d.decide({"kind": "select_models", "models": ["linear"]})
        refused = d.artifact("fit")["models"][0]["inference"]
        exit_ = next(e for e in refused["exits"] if e["decision"] is not None)
        d.decide(exit_["decision"])
        fit = d.artifact("fit")
    # The shelf put the exit first, and said why the linear model's table is empty.
    keys = [f["key"] for f in shelf["families"]]
    assert keys[0] == "mixed", keys
    linear = next(f for f in shelf["families"] if f["key"] == "linear")
    assert linear["fit"] == "poor" and any("6 units" in c for c in linear["concerns"])
    assert refused["covariance"] == "none" and "6 units" in refused["refused"]
    assert exit_["decision"] == {"kind": "select_models", "models": ["mixed"]}
    assert "mixed model" in exit_["label"]

    model = fit["models"][0]
    assert model["family"] == "mixed"
    info = model["inference"]
    assert info["covariance"] == "model" and info["n_clusters"] == 6
    assert "REML" in info["caption"] and "Satterthwaite" in info["caption"]
    sodium = _row(model["coefficients"], "sodium_mg")
    mixed = smf.mixedlm("sbp ~ sodium_mg + age", frame, groups=frame["participant_id"]).fit(reml=True)
    naive = smf.ols("sbp ~ sodium_mg + age", frame).fit()
    print(f"\nmixed model sodium p {sodium['p']:.3f} (t, {sodium['df']:.1f} df); statsmodels "
          f"MixedLM {mixed.pvalues['sodium_mg']:.3f} (z); independent-row OLS "
          f"{naive.pvalues['sodium_mg']:.2g}")
    assert sodium["p"] == pytest.approx(0.85, abs=0.01)
    assert mixed.pvalues["sodium_mg"] == pytest.approx(0.85, abs=0.01)
    assert naive.pvalues["sodium_mg"] < 1e-12
    for name in ("sodium_mg", "age"):
        assert _row(model["coefficients"], name)["estimate"] == pytest.approx(
            mixed.fe_params[name], rel=1e-4)
    # The interval is the t interval on the coefficient's own Satterthwaite df.
    q = stats.t.ppf(0.975, sodium["df"])
    assert sodium["ci_high"] - sodium["estimate"] == pytest.approx(q * sodium["se"], rel=1e-9)
    assert sodium["ci_low"] < 0 < sodium["ci_high"]


def _dense_reml(X: np.ndarray, y: np.ndarray, codes: np.ndarray) -> dict[str, Any]:
    """REML for the random-intercept model from its definition, with every matrix written out:
    ``ℓ_R(θ) = −½[log|V| + log|XᵀV⁻¹X| + rᵀV⁻¹r]`` (Harville 1977), ``V = σ²_e I + σ²_u ZZᵀ``,
    ``r = y − Xβ̂(θ)``, maximized over log σ² by Nelder–Mead (no profile, no low-rank identity)."""
    from scipy.optimize import minimize

    n = len(y)
    Z = (codes[:, None] == np.arange(codes.max() + 1)[None, :]).astype(float)
    ZZ = Z @ Z.T

    def parts(phi: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        s2u, s2e = np.exp(np.clip(phi, -30, 30))  # the optimizer's far trial points stay finite
        V = s2e * np.eye(n) + s2u * ZZ
        Vi = np.linalg.inv(V)
        A = X.T @ Vi @ X
        beta = np.linalg.solve(A, X.T @ Vi @ y)
        r = y - X @ beta
        # numpy's slogdet reports divide-by-zero here even on a well-conditioned V (log|V| = 14
        # at the optimum below), from floating-point flags it did not raise; every value returned
        # is finite and is checked against the engine.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ll = -0.5 * (np.linalg.slogdet(V)[1] + np.linalg.slogdet(A)[1] + r @ Vi @ r)
        return float(ll), beta, np.linalg.inv(A)

    start = np.log([max(np.var(y) / 2, 1e-3)] * 2)
    best = minimize(lambda p: -parts(p)[0], start, method="Nelder-Mead",
                    options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20_000})
    ll, beta, cov = parts(best.x)
    assert np.isfinite(ll)
    return {"phi": best.x, "loglik": ll, "beta": beta, "cov": cov, "parts": parts}


def _unbalanced(seed: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    sizes = rng.integers(2, 15, 9)
    codes = np.repeat(np.arange(9), sizes)
    n = len(codes)
    z = rng.normal(size=9)[codes]
    x = rng.normal(size=n) + 0.5 * z
    y = 2 + 0.3 * x + rng.normal(size=9)[codes] * 1.2 + rng.normal(size=n)
    return np.column_stack([np.ones(n), x, z]), y, codes


def test_4_the_reml_fit_and_its_covariance_are_the_definition():
    """On an unbalanced design (unit sizes 2–14): the variance components maximize the REML
    likelihood written out with dense matrices (to 10⁻⁵ relative), the fixed effects are the GLS
    estimate at them, and the reported covariance is ``(XᵀV̂⁻¹X)⁻¹``, as lme4 and lmerTest report
    it (to 10⁻⁶). statsmodels ``MixedLM`` (REML) agrees on the estimates and variance components
    to its optimizer's tolerance; its standard errors are not used as a reference, since it
    inverts the Hessian over every parameter (fixed effects and variance components together)."""
    import statsmodels.api as sm

    X, y, codes = _unbalanced()
    fit = fit_random_intercept(X, y, codes)
    dense = _dense_reml(X, y, codes)
    s2u, s2e = np.exp(dense["phi"])
    assert fit.sigma2_u == pytest.approx(s2u, rel=1e-5)
    assert fit.sigma2_e == pytest.approx(s2e, rel=1e-5)
    # At the engine's own estimate the dense likelihood is at its maximum, and GLS gives β and C.
    ll, beta, cov = dense["parts"](np.log([fit.sigma2_u, fit.sigma2_e]))
    assert ll >= dense["loglik"] - 1e-9
    np.testing.assert_allclose(fit.beta, beta, rtol=1e-8)
    np.testing.assert_allclose(fit.cov, cov, rtol=1e-6)
    statsmodels = sm.MixedLM(y, X, groups=codes).fit(reml=True)
    np.testing.assert_allclose(fit.beta, statsmodels.fe_params, rtol=1e-4)
    assert fit.sigma2_e == pytest.approx(statsmodels.scale, rel=1e-3)
    assert fit.sigma2_u == pytest.approx(float(np.asarray(statsmodels.cov_re)[0, 0]), rel=1e-3)


def test_4_satterthwaite_df_reproduce_the_balanced_anova_df():
    """In a balanced design the Satterthwaite df are the classical ANOVA df. The derivation: with
    G units of m rows, a covariate constant within units (``z``) and one centered within units
    (``w``), REML splits into the between-unit stratum, the unit means regressed on (1, z) with
    G − 2 residual df and variance θ_B = σ²_e + mσ²_u, and the within stratum, deviations from the
    unit means regressed on w with N − G − 1 residual df and variance σ²_e. At an interior optimum
    θ̂_B and σ̂²_e are those strata's mean squares, each a scaled χ² whose REML variance is
    2θ²/df. A between-unit coefficient's variance is a multiple of θ_B alone, so Satterthwaite's
    df = 2C²/Var(Ĉ) = G − 2; a within-unit coefficient's is a multiple of σ²_e, so N − G − 1."""
    rng = np.random.default_rng(5)
    G, m = 9, 7
    codes = np.repeat(np.arange(G), m)
    z = rng.normal(size=G)[codes]
    w = rng.normal(size=G * m)
    w -= pd.Series(w).groupby(codes).transform("mean").to_numpy()
    y = 1 + rng.normal(size=G)[codes] * 1.5 + rng.normal(size=G * m)
    fit = fit_random_intercept(np.column_stack([np.ones(G * m), z, w]), y, codes)
    assert not fit.boundary
    np.testing.assert_allclose(fit.df, [G - 2, G - 2, G * m - G - 1], rtol=1e-4)


def test_4_satterthwaite_df_are_the_definition_on_an_unbalanced_design():
    """Satterthwaite (Giesbrecht & Burns 1985), as lmerTest computes it: ``ν_j = 2 C_jj² /
    (∇C_jjᵀ 𝒜 ∇C_jj)``, with 𝒜 the inverse of the negative Hessian of the REML log-likelihood in
    the variance parameters and ∇C_jj the gradient of ``(XᵀV⁻¹X)⁻¹_jj`` in them. Reference: both
    computed from the dense definition (:func:`_dense_reml`) by finite differences in (σ²_u, σ²_e)
    themselves, where the engine differentiates analytically and takes the Hessian in log σ² (ν is
    the same in any parameterization at the optimum). To 10⁻³ relative."""
    X, y, codes = _unbalanced()
    fit = fit_random_intercept(X, y, codes)
    dense = _dense_reml(X, y, codes)
    theta = np.array([fit.sigma2_u, fit.sigma2_e])

    def at(t: np.ndarray) -> tuple[float, np.ndarray]:
        ll, _, cov = dense["parts"](np.log(t))
        return ll, np.diag(cov)

    h = 1e-4 * theta
    H = np.zeros((2, 2))
    grad = np.zeros((2, X.shape[1]))
    for i in range(2):
        e_i = np.eye(2)[i] * h[i]
        grad[i] = (at(theta + e_i)[1] - at(theta - e_i)[1]) / (2 * h[i])
        for j in range(2):
            e_j = np.eye(2)[j] * h[j]
            H[i, j] = (at(theta + e_i + e_j)[0] - at(theta + e_i - e_j)[0]
                       - at(theta - e_i + e_j)[0] + at(theta - e_i - e_j)[0]) / (4 * h[i] * h[j])
    A = np.linalg.inv(-H)
    C = at(theta)[1]
    df = 2 * C ** 2 / np.einsum("aj,ab,bj->j", grad, A, grad)
    print(f"\nSatterthwaite df {fit.df}; by definition {df}")
    np.testing.assert_allclose(fit.df, df, rtol=1e-3)


MIXED_REPS = 400  # Monte Carlo SE of a 0.05 rejection rate: 0.011, so 0.07 sits 1.8 SE above 0.05


def test_4_type_one_error_at_six_units():
    """The exit is honest where the audit found the table dishonest: 6 participants × 40 visits as
    in the audit's scenario, no effect of sodium (which varies within and between participants) or
    of a participant-level covariate. Over 400 replicates the mixed model's 5%-level tests reject
    at most 7% of the time for each; independent-row OLS, the table the audit found, rejects the
    participant-level covariate in most replicates (the audit's Monte Carlo: 0.44–0.67). Luke
    (2017, *Behav Res Methods* 49:1494): "Type 1 error rates are closest to .05 when models are
    fitted using REML and p-values are derived using the Kenward-Roger or Satterthwaite
    approximations"."""
    rng = np.random.default_rng(20261002)
    units, per = 6, 40
    codes = np.repeat(np.arange(units), per)
    rejected = np.zeros(2)
    ols_rejected = 0
    for _ in range(MIXED_REPS):
        sodium = rng.normal(size=units)[codes] + rng.normal(0, 0.3, units * per)
        level = rng.normal(size=units)[codes]  # a participant-level covariate, like age
        y = rng.normal(0, 2, units)[codes] + rng.normal(size=units * per)
        X = np.column_stack([np.ones(units * per), sodium, level])
        fit = fit_random_intercept(X, y, codes)
        se = np.sqrt(np.diag(fit.cov))[1:]
        p = 2 * stats.t.sf(np.abs(fit.beta[1:] / se), fit.df[1:])
        rejected += p < 0.05
        ols = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ ols[0]
        cov = np.linalg.inv(X.T @ X) * (resid @ resid) / (len(y) - 3)
        ols_rejected += 2 * stats.t.sf(abs(ols[0][2] / math.sqrt(cov[2, 2])), len(y) - 3) < 0.05
    rate = rejected / MIXED_REPS
    print(f"\nmixed model type-I error: sodium {rate[0]:.3f}, participant-level {rate[1]:.3f}; "
          f"independent-row OLS, participant-level {ols_rejected / MIXED_REPS:.3f}")
    assert np.all(rate <= 0.07), rate
    assert ols_rejected / MIXED_REPS > 0.3


def test_4_with_no_between_unit_variance_the_model_is_least_squares_and_says_so():
    """When REML puts the between-unit variance at zero (the boundary), the random intercept adds
    nothing: the estimates are ordinary least squares (statsmodels OLS), the df are N − P, and a
    concern says the intervals then treat the rows as independent."""
    import statsmodels.api as sm

    rng = np.random.default_rng(1)
    codes = np.repeat(np.arange(10), 6)
    x = rng.normal(size=60)
    y = 0.5 * x + rng.normal(size=60)
    y -= pd.Series(y).groupby(codes).transform("mean").to_numpy() - y.mean()  # no unit spread
    fit = fit_random_intercept(np.column_stack([np.ones(60), x]), y, codes)
    assert fit.boundary and fit.sigma2_u == 0.0
    ols = sm.OLS(y, sm.add_constant(x)).fit()
    np.testing.assert_allclose(fit.beta, ols.params, rtol=1e-10)
    np.testing.assert_allclose(np.sqrt(np.diag(fit.cov)), ols.bse, rtol=1e-8)
    np.testing.assert_allclose(fit.df, 58, rtol=1e-8)
    table = mixed_table(pd.DataFrame({"x": x}), y, _clusters(codes))
    assert any("estimated at zero" in c for c in table.concerns)


def test_4_under_prediction_the_folds_are_told_each_rows_unit(tmp_path):
    """The units reach every fit, not only the inference table: under prediction the mixed model
    is cross-validated with each fold's fit told the rows' units (its between-unit variance is
    estimated, so its fold predictions differ from least squares'), and the final fit's fixed
    effects equal the REML fit by the definition (:func:`_dense_reml`)."""
    frame = _six_by_forty()
    st = ProjectState(lens=["clinical"], target="sbp", task="regression", purpose="prediction",
                      roles={"participant_id": "identifier", "sodium_mg": "exposure",
                             "age": "covariate"}, missing="complete_case",
                      split=SplitSpec(holdout=0.0, seed=0, folds=5), models=["linear", "mixed"],
                      grain=GrainSpec(grain="repeated", id_column="participant_id"),
                      shape_confirmations={"code_or_count:age": "amount"})  # the truth
    _, _, fit = _stages(frame, st, ["sodium_mg", "age"], "regression", tmp_path)
    linear, mixed = fit.data["models"]
    assert mixed["cv"]["r2"]["folds"] != pytest.approx(linear["cv"]["r2"]["folds"], rel=1e-6)
    final = fit.objects["fitted"]["mixed"][-1]
    assert final.sigma2_u_ > 0
    X = np.column_stack([np.ones(240), frame[["sodium_mg", "age"]].to_numpy(float)])
    dense = _dense_reml(X / np.r_[1.0, 500.0, 10.0], frame["sbp"].to_numpy(float),
                        pd.factorize(frame["participant_id"])[0])
    np.testing.assert_allclose(np.r_[final.intercept_, final.coef_],
                               dense["beta"] / np.r_[1.0, 500.0, 10.0], rtol=1e-4)


def test_4_a_band_refit_counts_each_resampled_unit_as_its_own():
    """A substitution band resamples whole units; the k-th copy of a unit is a unit of its own,
    as a cluster bootstrap counts it, so a refit's random intercept does not merge the copies."""
    units = pd.Series(["a", "a", "b", "b"], index=[10, 11, 12, 13])
    labels = resampled_units(units, [10, 11, 10, 11, 12, 13])
    assert labels.tolist() == ["a#0", "a#0", "a#1", "a#1", "b#0", "b#0"]


# ── GEE, the other exit ──────────────────────────────────────────────────────


def _exchangeable(seed: int, G: int, m: int, binary: bool) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    sizes = rng.integers(max(2, m - 3), m + 4, G)
    codes = np.repeat(np.arange(G), sizes)
    n = len(codes)
    x = rng.normal(size=n) + rng.normal(size=G)[codes]
    z = rng.binomial(1, 0.5, G)[codes].astype(float)
    eta = 0.3 * x - 0.2 * z + rng.normal(size=G)[codes]
    y = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(float) if binary else eta + rng.normal(size=n)
    return pd.DataFrame({"x": x, "z": z}), y, codes


@pytest.mark.parametrize("binary", [False, True], ids=["identity", "logit"])
def test_gee_sandwich_is_statsmodels_and_its_cr2_is_the_definition(binary):
    """GEE's estimates are statsmodels' ``GEE`` (exchangeable working correlation). Its variance
    goes through a working linear model, ``W_g = R_g^(−½)A_g^(½)X_g`` and the whitened residuals,
    built with a closed-form inverse square root of the exchangeable matrix. References: with no
    small-sample adjustment, that sandwich is statsmodels' own robust covariance (to 10⁻⁶); with
    it, the CR2 (bias-reduced linearization, McCaffrey & Bell 2006) covariance and Bell–McCaffrey
    df equal ``references.cr2_by_definition`` on the model whitened instead by the symmetric
    square root of the dense working covariance, eigen-decomposed (to 10⁻⁸): CR2 does not depend
    on which square root whitens a unit."""
    import statsmodels.api as sm

    M, y, codes = _exchangeable(seed=6, G=14, m=6, binary=binary)
    X = sm.add_constant(M).to_numpy(float)
    task = "binary" if binary else "regression"
    fit = fit_gee(X, y, codes, task)
    family = sm.families.Binomial() if binary else sm.families.Gaussian()
    reference = sm.GEE(y, X, groups=codes, family=family,
                       cov_struct=sm.cov_struct.Exchangeable()).fit(maxiter=200, ctol=1e-10)
    np.testing.assert_allclose(fit.beta, reference.params, rtol=1e-10)
    W, r = gee_working(X, y, fit.mu, codes, fit.alpha, task)
    V0, _ = cr2(W, r, codes, bread_of(W), adjust=False)
    np.testing.assert_allclose(V0, reference.cov_robust, rtol=1e-6)

    # The symmetric root of each unit's dense working covariance, an independent whitening.
    v = fit.mu * (1 - fit.mu) if binary else np.ones(len(y))
    Wd, rd = np.zeros_like(X), np.zeros(len(y))
    for g in np.unique(codes):
        idx = np.flatnonzero(codes == g)
        R = (1 - fit.alpha) * np.eye(len(idx)) + fit.alpha
        cov = np.sqrt(v[idx])[:, None] * R * np.sqrt(v[idx])[None, :]
        lam, vec = np.linalg.eigh(cov)
        inv_root = vec @ np.diag(lam ** -0.5) @ vec.T
        D = X[idx] * v[idx][:, None]  # ∂μ/∂β for the canonical link
        Wd[idx] = inv_root @ D
        rd[idx] = inv_root @ (y[idx] - fit.mu[idx])
    V_ref, df_ref = ref.cr2_by_definition(Wd, rd, codes)
    V, df = cr2(W, r, codes, bread_of(W))
    np.testing.assert_allclose(V, V_ref, rtol=1e-8, atol=1e-14)
    np.testing.assert_allclose(df, df_ref, rtol=1e-8)
    table = gee_table(M, y, _clusters(codes), task, [0.0, 1.0] if binary else None)
    np.testing.assert_allclose([row["se"] for row in table.rows], np.sqrt(np.diag(V_ref)), rtol=1e-8)
    assert "exchangeable working correlation" in table.info["caption"]
    assert table.info["covariance"] == "CR2"


def test_gee_a_binary_table_is_on_the_odds_ratio_scale_through_the_fit_stage(tmp_path):
    """Repair round (WP12b × WP8, ME-07 again): a binary GEE table declared the difference scale,
    gave no odds ratios, and its caption never said log-odds. Through the real stages under
    inference, the outcome spelled "case"/"control" with "case" named the event: the estimates
    are statsmodels' ``GEE`` (binomial, exchangeable) log-odds, each row's ratio is exp(estimate),
    the table says odds ratio of `event` being `case` rather than `control`, on a log axis."""
    import statsmodels.api as sm

    M, y, codes = _exchangeable(seed=6, G=40, m=6, binary=True)
    frame = M.assign(pid=codes, event=np.where(y == 1, "case", "control"))
    st = ProjectState(lens=["clinical"], target="event", task="binary", event="case",
                      purpose="inference", roles={"pid": "identifier", "x": "exposure",
                                                  "z": "covariate"},
                      missing="complete_case", split=SplitSpec(holdout=0.0, seed=0, folds=5),
                      models=["gee"], grain=GrainSpec(grain="repeated", id_column="pid"))
    _, _, fit = _stages(frame, st, ["x", "z"], "binary", tmp_path)
    model = fit.data["models"][0]
    info = model["inference"]
    reference = sm.GEE(y, sm.add_constant(M).to_numpy(float), groups=codes,
                       family=sm.families.Binomial(),
                       cov_struct=sm.cov_struct.Exchangeable()).fit(maxiter=200, ctol=1e-10)
    x = _row(model["coefficients"], "x")
    assert x["estimate"] == pytest.approx(reference.params[1], rel=1e-6)
    assert info["scale"] == "odds_ratio" and info["axis"] == "log"
    assert info["event"] == "case" and info["reference"] == "control"
    assert info["caption"].startswith("Estimates are log-odds of `case` against `control`")
    assert x["ratio"] == pytest.approx(math.exp(reference.params[1]), rel=1e-6)
    assert x["ratio_low"] == pytest.approx(math.exp(x["ci_low"]), rel=1e-12)


def test_gee_below_the_unit_floor_refuses_and_names_the_mixed_model_exit():
    """GEE's intervals are a sandwich, so they keep the unit floor the linear family's keep
    (``inference.min_clusters``, the seal's 8): at 6 units the table reports estimates only, and
    for a continuous outcome its exit is the random-intercept mixed model, whose decision the
    ``select_models`` validator accepts. For a binary outcome there is no mixed-model exit yet, and
    the table says so rather than offering one."""
    frame = _six_by_forty()
    clusters = _clusters(frame["participant_id"], "participant_id")
    table = gee_table(frame[["sodium_mg", "age"]], frame["sbp"].to_numpy(float), clusters,
                      "regression")
    assert table.info["covariance"] == "none" and all(r["p"] is None for r in table.rows)
    decisions = [e["decision"] for e in table.info["exits"] if e["decision"]]
    assert decisions == [{"kind": "select_models", "models": ["mixed"]}]
    assert validate(SelectModels(**{k: v for k, v in decisions[0].items() if k != "kind"}),
                    {"task": "regression"})
    _, exits = floor_refusal(Clusters(column="pid", codes=np.repeat(np.arange(6), 5), n_clusters=6),
                             task="binary")
    assert all(e["decision"] is None for e in exits)
    assert any("not yet in TurboTab" in e["label"] for e in exits)
    assert min_clusters() == 8


GEE_REPS = 300  # Monte Carlo SE of a 0.05 rejection rate: 0.013; 0.08 sits 2.3 SE above 0.05


def test_gee_type_one_error_at_twelve_units():
    """Twelve units of about eight rows, a unit-level binary exposure with no effect and strong
    within-unit correlation: GEE's CR2 t tests on Bell–McCaffrey df reject at most 8% of the time
    at the 5% level over 300 replicates. The uncorrected sandwich on z, the reference McCaffrey &
    Bell (2006) improve on, rejects more often on the same replicates."""
    import statsmodels.api as sm

    rng = np.random.default_rng(12)
    rejected = naive = 0
    for _ in range(GEE_REPS):
        G = 12
        codes = np.repeat(np.arange(G), rng.integers(6, 11, G))
        n = len(codes)
        z = np.isin(np.arange(G), rng.choice(G, G // 2, replace=False))[codes].astype(float)
        x = rng.normal(size=n)
        y = rng.normal(0, 1.2, G)[codes] + 0.4 * x + rng.normal(size=n)
        table = gee_table(pd.DataFrame({"z": z, "x": x}), y, _clusters(codes), "regression")
        rejected += _row(table.rows, "z")["p"] < 0.05
        X = sm.add_constant(np.column_stack([z, x]))
        fit = fit_gee(X, y, codes, "regression")
        W, r = gee_working(X, y, fit.mu, codes, fit.alpha, "regression")
        V0, _ = cr2(W, r, codes, bread_of(W), adjust=False)
        naive += 2 * stats.norm.sf(abs(fit.beta[1] / math.sqrt(V0[1, 1]))) < 0.05
    print(f"\nGEE CR2 type-I error at 12 units {rejected / GEE_REPS:.3f}; uncorrected sandwich on z "
          f"{naive / GEE_REPS:.3f}")
    assert rejected / GEE_REPS <= 0.08
    assert naive > rejected


# ── the shelf: ordered by soundness for the declared purpose ─────────────────


def test_the_shelf_orders_the_exits_by_soundness_for_the_purpose():
    """Under inference with a continuous outcome repeated within 6 units, the mixed model ranks
    first and the linear and GEE families are marked poor with the reason (their intervals would be
    refused); with 40 units, the CR2 linear model, the mixed model and GEE all stand. When no
    identifier repeats, neither repeated-rows family is recommended. Under prediction they are
    "fair": they predict a new unit from population-level effects, as the linear model does."""
    def keys(**kw: Any) -> list[tuple[str, str]]:
        base = dict(task="regression", purpose="inference", n_rows=240, n_features=2)
        base.update(kw)
        return [(f.key, a.fit) for f, a in rank(Situation(**base))]

    few = keys(n_units=6)
    assert few[0] == ("mixed", "good")
    assert ("linear", "poor") in few and ("gee", "poor") in few
    many = dict(keys(n_units=40, n_rows=400))
    assert many["linear"] == "good" and many["mixed"] == "good" and many["gee"] == "good"
    alone = keys()
    assert alone[0][0] == "linear" and dict(alone)["mixed"] == "poor" and dict(alone)["gee"] == "poor"
    predicting = dict(keys(purpose="prediction", n_units=6))
    assert predicting["mixed"] == "fair" and predicting["gee"] == "fair"
    binary = dict(keys(task="binary", n_units=6, n_events=100))
    assert "mixed" not in binary and binary["gee"] == "poor" and binary["linear"] == "poor"

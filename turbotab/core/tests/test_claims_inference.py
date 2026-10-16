"""The inference area's claims (``turbotab/core/claims/inference.py``; R0_PLAN.md R0b): the known-
truth tests each claim names beside stage 1's, chiefly the silent side of each guard, and the
checks that the area's module constants read their claims and its predicates hold where the
claims say.

Each expected value comes from the construction, an independent library or the source a claim
cites, never from the claim or the constant it fills (CLAIMS_REGISTRY.md §2.1; meta-test 2 holds
this file to it): Li & Redden (2015) simulated from ten clusters; Cameron & Miller (2015, §VI)
put "few" below fifty in every reading; Kutner, Nachtsheim & Neter's collinearity line is a
variance inflation factor of ten; a group with events among the others' has a finite Cox
maximum, which lifelines finds; a logistic fit statsmodels converges on separates no one.
"""
from __future__ import annotations

import ast
import importlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from turbotab.core import claims
from turbotab.core.claims import inference as area
from turbotab.core.claims.inference import (cluster_randomized_trial, cox_partial_likelihood,
                                            design_unanswered,
                                            drawn_by_outcome_analyzed_as_observational,
                                            numeric_exposure, proportional_odds_cluster_sandwich,
                                            randomized_trial, rows_repeat, yes_no_outcome)

# ── the trial core's few clusters (Li & Redden 2015) ─────────────────────────


def _cluster_trial(K: int, seed: int = 7) -> pd.DataFrame:
    """A cluster-randomized trial of ``K`` clusters (4 to 39 people each), alternate clusters
    treated, a cluster effect, a covariate and a yes/no outcome."""
    rng = np.random.default_rng(seed)
    m = rng.integers(4, 40, K)
    df = pd.DataFrame({"cluster": np.repeat(np.arange(1, K + 1), m)})
    df["arm"] = df.cluster.map({c: ("ctl" if c % 2 else "trt") for c in range(1, K + 1)})
    u = rng.normal(0, 0.8, K + 1)
    df["x"] = rng.normal(0, 1, len(df))
    lp = 0.6 * u[df.cluster] + 0.5 * (df.arm == "trt") + 0.3 * df.x - 0.2
    df["event"] = (rng.random(len(df)) < 1 / (1 + np.exp(-lp))).astype(int)
    return df


def _gee_spec():
    from turbotab.core.methods import trials as T

    return T.TrialSpec(arm="arm", control="ctl", outcome="event",
                       trial_design="cluster_randomized_trial", cluster="cluster",
                       covariates=("x",))


def _few(concerns) -> list[str]:
    return [c for c in concerns if "clusters were randomized" in c]


@pytest.mark.parametrize("K", [8, 9])
def test_a_cluster_trial_of_eight_clusters_names_the_few_clusters(K):
    """Li & Redden (2015) simulated the corrected sandwich of the estimating equations for a
    yes/no outcome from 10 clusters, so with 8 or 9 the trial's concerns name the count and where
    the evidence ends, with the estimate still reported."""
    from turbotab.core.methods import trials as T

    r = T.estimate(_cluster_trial(K), _gee_spec())
    [said] = _few(r.concerns)
    assert said == (f"Only {K} clusters were randomized; the intervals rest on the small-sample "
                    f"correction, and the fewest clusters Li & Redden (2015) studied was 10.")
    assert r.contrast(measure="risk_difference").se > 0


@pytest.mark.parametrize("K", [10, 12])
def test_a_cluster_trial_of_twelve_clusters_names_no_few_clusters(K):
    """From 10 clusters, within what Li & Redden (2015) simulated, no concern names the count;
    and with no group's risk at 0% or 100% the estimating equations are reported, not refused
    for separation."""
    from turbotab.core.methods import trials as T

    r = T.estimate(_cluster_trial(K), _gee_spec())
    assert _few(r.concerns) == [], r.concerns
    assert not any("Li & Redden" in c for c in r.concerns), r.concerns
    rd = r.contrast(measure="risk_difference")
    assert np.isfinite(rd.estimate) and rd.lower < rd.estimate < rd.upper


# ── the trial's separation (claims backtest B-281) ───────────────────────────


def test_a_cluster_trial_whose_treated_clusters_all_had_the_event_is_refused():
    """Everyone in the treated clusters had the event, so the arm predicts it exactly and the
    logistic model's risk there runs to 100%: the estimating equations are refused for
    separation, with no risk difference reported."""
    from turbotab.core.methods import trials as T

    df = _cluster_trial(12)
    df.loc[df.arm == "trt", "event"] = 1
    with pytest.raises(T.TrialRefused) as r:
        T.estimate(df, _gee_spec())
    assert r.value.term == "separation in the logistic model", str(r.value)
    assert "no finite estimate exists" not in str(r.value)


def test_a_trial_no_one_is_separated_in_is_reported_as_statsmodels_fits_it():
    """A pH of 7.4 ± 0.02 with a log odds ratio of 60 per unit: the fitted coefficient passes 25,
    a sign of separation, but statsmodels' ``Logit`` converges on everyone, so no direction
    separates anyone (Albert & Anderson 1984). The trial is reported, its standardized risk
    difference the g-computation on statsmodels' fit (each person's risk with the arm set to
    each arm, averaged), with no note that the data separate anyone."""
    import statsmodels.api as sm

    from turbotab.core.methods import trials as T

    rng = np.random.default_rng(91000)
    df = pd.DataFrame({"arm": rng.choice(["ctl", "trt"], 150), "site": rng.choice(["a", "b"], 150),
                       "ph": 7.4 + rng.normal(0, 0.02, 150)})
    lp = -0.3 + 0.7 * (df.arm == "trt") + 60 * (df.ph - 7.4)
    df["event"] = (rng.random(150) < 1 / (1 + np.exp(-lp))).astype(float)
    X = np.column_stack([np.ones(150), df.arm == "trt", df.site == "b", df.ph]).astype(float)
    fit = sm.Logit(df.event.to_numpy(), X).fit(method="newton", tol=1e-12, disp=0)
    assert fit.mle_retvals["converged"] and fit.params[-1] > 25

    def risk(arm: float) -> np.ndarray:
        Xa = X.copy()
        Xa[:, 1] = arm
        return 1 / (1 + np.exp(-Xa @ fit.params))

    truth = float(np.mean(risk(1.0)) - np.mean(risk(0.0)))
    spec = T.TrialSpec(arm="arm", control="ctl", outcome="event", strata=("site",),
                       covariates=("ph",))
    r = T.estimate(df, spec)
    got = r.contrast(measure="risk_difference")
    assert abs(got.estimate - truth) <= 1e-6 * got.se, (got.estimate, truth, got.se)
    assert not any("separate" in c for c in r.concerns), r.concerns


# ── the proportional-odds table's cluster sandwich (claims backtest A-102) ───


def _units(seed: int, G: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """G units of lognormal size (median 20, at least 2), a column that varies only between
    units, and a proportional-odds outcome on it with β = 0.5 and cut-points −1, 0, 1."""
    rng = np.random.default_rng(seed)
    unit = np.repeat(np.arange(G), np.maximum(2, np.round(rng.lognormal(np.log(20), 1.0, G))
                                                .astype(int)))
    x = rng.normal(size=G)[unit]
    y = np.searchsorted([-1.0, 0.0, 1.0], 0.5 * x + rng.logistic(size=len(unit)))
    return pd.DataFrame({"x": x}), y, unit


def test_an_ordinal_table_over_sixty_units_says_nothing_of_narrow_intervals():
    """Sixty units are not few in any reading of Cameron & Miller (2015, §VI), who put "few"
    below twenty to below fifty: the clustered proportional-odds table still says its intervals
    are CR1 on t(59), and no line says they can be too narrow."""
    from turbotab.core.models.linear import as_clusters
    from turbotab.core.models.ordinal import ordinal_table

    X, y, unit = _units(0, 60)
    table = ordinal_table(X, y, None, as_clusters(unit))
    assert table.info["covariance"] == "CR1" and "t(59)" in table.info["caption"]
    assert any("(CR1, G = 60)" in c for c in table.concerns), table.concerns
    assert not any("too narrow" in c or "Only" in c for c in table.concerns), table.concerns


# ── a sample drawn by its outcome (claims backtest B-211) ────────────────────


def test_a_trial_analyzed_as_observational_keeps_its_risk_difference_and_predict():
    """A parallel trial is not drawn by its outcome, so taking its refusal's first exit (analyzed
    as an observational study) keeps the marginal risk difference and ratio, the prevalence and
    Predict: the record names the trial and says nothing of odds ratios only."""
    from turbotab.core import decisions as d
    from turbotab.core import voice
    from turbotab.core.decisions import DecisionRecord, ProjectState, parse_decision
    from turbotab.core.estimand import estimand_card

    with pytest.raises(d.Refusal) as caught:
        d.validate({"kind": "set_design", "design": "parallel_trial"}, {})
    first = caught.value.exits[0]
    assert first["label"] == "Analyze it as an observational study"
    taken = d.validate(first["decision"], {})
    at = datetime(2026, 10, 16, tzinfo=timezone.utc)
    state = d.fold([DecisionRecord(id="r1", seq=1, at=at,
                                   decision=parse_decision(first["decision"]))])
    said = voice.sentence_for(taken, ProjectState(), None)
    assert "randomized to parallel groups" in said.lower(), said
    assert "odds ratio" not in said and "sampled separately" not in said, said

    roles = {"pid": "identifier", "x": "exposure", "age": "covariate"}
    estimate = state.model_copy(update=dict(lens=["clinical"], target="case", task="binary",
                                            purpose="inference", roles=roles))
    for measure in ("risk_difference", "risk_ratio"):
        assert d.validate({"kind": "set_estimand", "exposure": "x", "measure": measure},
                          {"state": estimate}).measure == measure
    card = estimand_card(estimate, "binary", prevalence=0.2)
    assert card["prevalence"] == pytest.approx(0.2)
    assert "risk_difference" in [m["measure"] for m in card["measures"] if m["fitted"]]
    assert d.validate({"kind": "set_purpose", "purpose": "prediction"},
                      {"state": state}).purpose == "prediction"


# ── a numeric exposure's R² (claims backtest B-416) ──────────────────────────


def _positivity(r2: float) -> dict:
    from turbotab.core import causal

    [row] = [a for a in causal.assumption_card(
        exposure="dose", exposure_kind="continuous", adjusted=["age"], timing_unknown=[],
        overlap=None, variation={"r2": r2, "residual_sd": 1.0}, contrast=None, level=None)
        if a["key"] == "positivity"]
    return row


def test_an_exposure_explained_to_ninety_one_percent_names_its_widening():
    """At R² 0.91 the variance inflation factor is 1/0.09 ≈ 11, past Kutner, Nachtsheim & Neter's
    line of 10, and by Frisch–Waugh the standard error is about √11.1 ≈ 3.3 times what it would
    be were the exposure unrelated to the covariates: the row names that widening as precision,
    and its positivity stays a stated assumption."""
    row = _positivity(0.91)
    assert row["status"] == "stated", row
    factor, vif = (1 - 0.91) ** -0.5, 1 / (1 - 0.91)
    assert (f"{factor:.1f}", f"{vif:.0f}") == ("3.3", "11")
    assert "about 3.3 times what it would be" in row["diagnostic"], row["diagnostic"]
    assert "(a variance inflation factor of 11)" in row["diagnostic"], row["diagnostic"]
    assert "a matter of precision, not of positivity" in row["diagnostic"]


def test_an_exposure_explained_to_eighty_five_percent_names_no_widening():
    """At R² 0.85 the variance inflation factor is about 6.7, below Kutner, Nachtsheim & Neter's
    line of 10: the row gives the R² and the residual SD and names no widening, and the
    positivity is a stated assumption as at any R²."""
    row = _positivity(0.85)
    assert row["status"] == "stated", row
    assert "85% of `dose`'s variance" in row["diagnostic"], row["diagnostic"]
    assert "times what it would be" not in row["diagnostic"], row["diagnostic"]
    assert "precision" not in row["diagnostic"], row["diagnostic"]


# ── the Cox fit's monotone likelihood (claims backtest B-435) ────────────────


def test_a_cox_group_with_two_events_is_estimated_as_lifelines_fits_it():
    """A binary group (30% of rows) with two events, beside an age that carries the hazard: its
    events fall among the other rows' at risk, so no column orders every risk set and the
    partial likelihood has a finite maximum. The table is reported, its log hazard ratios and
    standard errors lifelines' to 10⁻⁶ relative."""
    from lifelines import CoxPHFitter

    from turbotab.core.models.inference import Clusters
    from turbotab.core.models.survival import cox_table, survival_outcome

    rng = np.random.default_rng(4)
    n = 200
    g = rng.binomial(1, 0.3, n).astype(float)
    age = rng.normal(size=n)
    t = rng.exponential(1 / (0.2 * np.exp(0.3 * age)))
    censor = rng.uniform(1, 10, n)
    event = (t <= censor).astype(int)
    time_ = np.minimum(t, censor)
    group_events = np.flatnonzero((g == 1) & (event == 1))
    event[group_events[2:]] = 0  # the group keeps two events; the rest are censored there
    frame = pd.DataFrame({"g": g, "age": age, "time": time_, "event": event})
    assert frame.loc[frame.g == 1, "event"].sum() == 2 and frame.event.sum() > 20

    table = cox_table(frame[["g", "age"]], survival_outcome(frame.event, frame.time), Clusters())
    assert not table.info.get("refused"), table.info.get("refused")
    reference = CoxPHFitter().fit(frame, duration_col="time", event_col="event", robust=False,
                                  fit_options={"precision": 1e-12, "max_steps": 200})
    rows = {r["feature"]: r for r in table.rows}
    np.testing.assert_allclose([rows[c]["estimate"] for c in ("g", "age")],
                               reference.params_[["g", "age"]].to_numpy(), rtol=1e-6)
    np.testing.assert_allclose([rows[c]["se"] for c in ("g", "age")],
                               reference.standard_errors_[["g", "age"]].to_numpy(), rtol=1e-6)


# ── the area's constants and predicates ──────────────────────────────────────


def _assignment(module: str, name: str) -> ast.expr:
    source = Path(importlib.import_module(module).__file__).read_text(encoding="utf-8")
    [value] = [node.value for node in ast.parse(source).body if isinstance(node, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)]
    return value


def test_each_constant_the_area_s_claims_name_reads_a_claim_holding_its_number():
    """Every engine constant a claim here names is assigned once, by ``claims.value`` on a claim of
    this area that holds the same parameter, and holds that parameter's value."""
    held = {c.id: c for c in area.CLAIMS}
    named = [(c, key, p) for c in area.CLAIMS for key, p in c.params.items() if p.constant]
    assert len(named) >= 7
    for c, key, p in named:
        module, _, name = p.constant.partition(":")
        assert getattr(importlib.import_module(module), name) == p.value, p.constant
        call = _assignment(module, name)
        assert isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
        assert call.func.attr == "value" and isinstance(call.func.value, ast.Name)
        assert call.func.value.id == "claims", p.constant
        read_id, read_key = (a.value for a in call.args)
        assert held[read_id].params[read_key] == p, (p.constant, read_id, read_key)


def test_the_claims_module_imports_nothing_of_the_engine():
    tree = ast.parse(Path(area.__file__).read_text(encoding="utf-8"))
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    imported |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert {m for m in imported if m and m.startswith("turbotab")} == {"turbotab.core.claims"}


def test_each_predicate_holds_in_its_regime_and_not_outside_it():
    """Each returns a bool, from the context alone."""
    cases = [
        (rows_repeat, {"cluster_codes": np.array([0, 0, 1])}, {"cluster_codes": [0, 1, 2]}),
        (proportional_odds_cluster_sandwich, {"model": "proportional_odds", "covariance": "CR1"},
         {"model": "linear", "covariance": "CR2"}),
        (cluster_randomized_trial, {"trial_design": "cluster_randomized_trial"},
         {"trial_design": "parallel_trial"}),
        (randomized_trial, {"trial_design": "parallel_trial"}, {"trial_design": "observational"}),
        (yes_no_outcome, {"outcome_values": [0.0, 1.0, 1.0]}, {"outcome_values": [0.0, 1.0, 2.0]}),
        (design_unanswered, {"design": None}, {"design": "observational"}),
        (drawn_by_outcome_analyzed_as_observational,
         {"design": "observational", "design_declared": "matched_sets"},
         {"design": "observational", "design_declared": "parallel_trial"}),
        (numeric_exposure, {"exposure_kind": "continuous"}, {"exposure_kind": "binary"}),
        (cox_partial_likelihood, {"model": "cox"}, {"model": "proportional_odds"}),
    ]
    for predicate, inside, outside in cases:
        assert predicate(inside) is True, predicate.__name__
        assert predicate(outside) is False, predicate.__name__
        # An empty context names nothing, so only "unanswered" holds there.
        assert predicate({}) is (predicate is design_unanswered), predicate.__name__


def test_the_proportional_odds_sentence_refuses_a_linear_model_s_context_in_strict_mode():
    """Claims backtest A-102 is the case: the proportional-odds table had shown the linear
    family's concern line. A claim whose regime is the proportional-odds model's cluster sandwich
    raises in strict mode when the context is a linear model's CR2, and renders in its own."""
    [c] = [c for c in area.CLAIMS
           if f"{area.HERE}:proportional_odds_cluster_sandwich" in c.regime.requires]
    with pytest.raises(claims.OutsideRegime):
        claims.card(c.id, {"model": "linear", "covariance": "CR2"}, strict=True)
    said = claims.card(c.id, {"model": "proportional_odds", "covariance": "CR1"}, strict=True)
    assert said.endswith("can still be too narrow"), said

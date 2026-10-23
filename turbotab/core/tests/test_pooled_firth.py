"""A logistic table pooled over multiple imputation where a column separates the outcome
(``models.inference.for_pooling`` and ``pooled_firth``; ``stages.modeling.pooled_table``;
``stages.effects``; claim ``inference.pooled_separation``).

Rubin's rules read each completed copy's estimate and standard error and make the interval and
p-value themselves (``imputation.pool_rows``), so a Firth copy computes no profile, and the pooled
estimates and standard errors are the ones today's profiled copies pool to. The pooled numbers are
Wald intervals on t from Firth's standard errors, and the table says so; a coefficient that
separates the outcome in any copy keeps its estimate and has its interval and p-value withheld.

Every expected value comes from the construction (a column that separates the outcome by how the
rows are drawn), or from Firth's penalized likelihood maximized by scipy's general-purpose
optimizer on the penalized log-likelihood written out in numpy, its standard errors the inverse
information there, pooled by Rubin's rules written out here (Rubin 1987: ν = (m − 1)/λ², the
complete-data degrees of freedom infinite for a likelihood fit).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import optimize, stats

import turbotab.core.models  # noqa: F401 - registers the families
from turbotab.core.decisions import MissingSpec, ProjectState
from turbotab.core.methods.imputation import pool_rows
from turbotab.core.methods.missing import impute_for_inference
from turbotab.core.models import get_family
from turbotab.core.models import inference as I
from turbotab.core.models.inner_cv import fit_pipeline
from turbotab.core.models.pipeline import build_pipeline, design_spec
from turbotab.core.stages.modeling import _inference_table, pooled_table

LINEAR = get_family("linear")


def _separated_rows(n: int = 150, seed: int = 3, separate: bool = True) -> tuple[pd.DataFrame, np.ndarray]:
    """``x`` a 0/1 column; with ``separate`` every row with x = 1 has the event (so x's
    maximum-likelihood log-odds is infinite in every completed copy, whatever ``z`` is imputed
    to), else the event follows x and z alike. ``z`` is blank on about 30% of the rows."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    x = (rng.random(n) < 0.2).astype(float)
    p = 1 / (1 + np.exp(-(-0.5 + 0.8 * z + (0.0 if separate else 0.7) * x)))
    y = (rng.random(n) < p).astype(float)
    if separate:
        y = np.where(x == 1, 1.0, y)
    frame = pd.DataFrame({"x": x, "z": np.where(rng.random(n) < 0.3, np.nan, z)})
    return frame, y


def _pooled(frame: pd.DataFrame, y: np.ndarray) -> dict[str, Any]:
    """The engine's imputations of ``z`` under inference and the pooled logistic table, as the
    fit stage pools it, with every call of ``firth_profile`` counted."""
    st = ProjectState(target="y", task="binary", purpose="inference",
                      roles={"x": "exposure", "z": "covariate"},
                      missing=MissingSpec(strategy="multiple_imputation", m=20))
    spec = design_spec(st, frame, ["x", "z"])
    imputations = impute_for_inference(spec, frame, y, "binary", seed=1)
    pipe = build_pipeline(spec, LINEAR, "binary", "inference", len(frame), 2)
    collected: dict[str, Any] = {}
    table, rows, _, _ = pooled_table(LINEAR, pipe, imputations, y, task="binary",
                                     clusters=I.INDEPENDENT, outcome=I.Outcome(name="y"),
                                     survey=None, design=None, spec=spec, rows="all",
                                     fit=lambda model, X_k: fit_pipeline(model, X_k, y),
                                     collect=collected)
    return {"table": table, "rows": {r["feature"]: r for r in rows}, "spec": spec,
            "imputations": imputations, "fits": collected["fits"],
            "tables": collected["tables"]}


@pytest.fixture(scope="module")
def separated() -> dict[str, Any]:
    frame, y = _separated_rows()
    calls = {"n": 0}
    real = I.firth_profile

    def counting(*a: Any, **k: Any) -> Any:
        calls["n"] += 1
        return real(*a, **k)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(I, "firth_profile", counting)
        out = _pooled(frame, y)
        out["profiles"] = calls["n"]
        # Today's per-copy work, for the comparison: each copy's table made as an unpooled table
        # is (every coefficient profiled), then pooled by Rubin's rules.
        calls["n"] = 0
        full = [_inference_table(LINEAR, f, X_k, y, task="binary", clusters=I.INDEPENDENT,
                                 outcome=I.Outcome(name="y"), rows="all")
                for f, X_k in zip(out["fits"], out["imputations"].frames)]
        out["full_profiles"] = calls["n"]
    out["full_tables"] = full
    out["full"] = {r["feature"]: r for r in pool_rows([t.rows for t in full])}
    out["frame"], out["y"] = frame, y
    return out


def _firth_by_scipy(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Firth's estimate, the maximum of l(β) + ½ log |XᵀWX| (Firth 1993) written out in numpy and
    maximized by scipy's BFGS, and the inverse information XᵀWX at it."""
    def negative(beta: np.ndarray) -> float:
        eta = X @ beta
        mu = 1 / (1 + np.exp(-eta))
        w = mu * (1 - mu)
        sign, logdet = np.linalg.slogdet(X.T @ (X * w[:, None]))
        return -(float(np.sum(y * eta - np.logaddexp(0.0, eta))) + 0.5 * logdet)

    found = optimize.minimize(negative, np.zeros(X.shape[1]), method="BFGS",
                              options={"gtol": 1e-10, "maxiter": 10_000})
    mu = 1 / (1 + np.exp(-(X @ found.x)))
    return found.x, np.linalg.inv(X.T @ (X * (mu * (1 - mu))[:, None]))


def _rubin_by_hand(frames: list[pd.DataFrame], y: np.ndarray, j: int) -> dict[str, float]:
    """Coefficient ``j`` of [intercept, x, z] over the completed copies, each fit by
    :func:`_firth_by_scipy`, pooled by Rubin's rules written out (ν = (m − 1)/λ²)."""
    q, u = [], []
    for X_k in frames:
        X = np.column_stack([np.ones(len(X_k)), X_k["x"].to_numpy(float), X_k["z"].to_numpy(float)])
        beta, cov = _firth_by_scipy(X, y)
        q.append(beta[j])
        u.append(cov[j, j])
    m = len(q)
    qbar, ubar, b = float(np.mean(q)), float(np.mean(u)), float(np.var(q, ddof=1))
    t = ubar + (1 + 1 / m) * b
    lam = (1 + 1 / m) * b / t
    nu = (m - 1) / lam ** 2
    half = float(stats.t.ppf(0.975, nu)) * math.sqrt(t)
    return {"estimate": qbar, "se": math.sqrt(t), "ci_low": qbar - half, "ci_high": qbar + half,
            "p": float(2 * stats.t.sf(abs(qbar) / math.sqrt(t), nu)), "df": nu}


# ── F2: no profile per copy, and the same pooled numbers ─────────────────────


def test_under_imputation_no_copy_profiles_and_the_pooled_estimates_and_errors_hold(separated):
    """Each copy's Firth table is made for Rubin's rules, so ``firth_profile`` is never called
    (today's per-copy tables call it for every coefficient of every copy), and the pooled
    estimates and standard errors are those today's profiled tables pool to, to 1e-12; both are
    Firth's estimate and Rubin's total variance, recomputed by scipy and numpy."""
    m = separated["imputations"].m
    assert m >= 20 and all(t.info["estimator"] == "Firth-penalized logistic regression"
                            for t in separated["tables"])
    assert separated["profiles"] == 0
    assert separated["full_profiles"] == 3 * m  # the base's work: three coefficients a copy
    for name in ("(intercept)", "x", "z"):
        ours, base = separated["rows"][name], separated["full"][name]
        assert ours["estimate"] == pytest.approx(base["estimate"], rel=1e-12, abs=1e-12), name
        assert ours["se"] == pytest.approx(base["se"], rel=1e-12, abs=1e-12), name
    frames = separated["imputations"].frames
    for j, name in enumerate(("(intercept)", "x", "z")):
        truth = _rubin_by_hand(frames, separated["y"], j)
        ours = separated["rows"][name]
        assert ours["estimate"] == pytest.approx(truth["estimate"], abs=1e-6), name
        assert ours["se"] == pytest.approx(truth["se"], rel=1e-5), name


def test_a_table_not_pooled_keeps_its_profile_intervals(separated):
    """Outside the pooling (complete cases, or any one table) the Firth table is as before: each
    coefficient's profile interval and likelihood-ratio p-value, and the caption that says so."""
    for table in separated["full_tables"][:3]:
        assert all(r["ci_low"] is not None and r["p"] is not None for r in table.rows
                   if r["feature"] != "x")
        x = next(r for r in table.rows if r["feature"] == "x")
        assert x["ci_low"] is not None and x["ci_low"] < x["estimate"] and x["p"] is not None
        assert "95% profile penalized-likelihood intervals" in table.info["caption"]


# ── F3: the pooled table's words, and the separated coefficient withheld ─────


def test_a_coefficient_separated_in_the_copies_keeps_its_pooled_estimate_and_withholds_its_interval(
        separated):
    """x separates the outcome in every copy by construction: its pooled estimate and standard
    error stand (Firth's, pooled), its interval, p-value and odds-ratio interval are withheld, and
    the table names it, says why and gives the ways forward, complete cases a decision."""
    m = separated["imputations"].m
    x = separated["rows"]["x"]
    truth = _rubin_by_hand(separated["imputations"].frames, separated["y"], 1)
    assert x["estimate"] == pytest.approx(truth["estimate"], abs=1e-6)
    assert x["se"] == pytest.approx(truth["se"], rel=1e-5)
    assert (x["ci_low"], x["ci_high"], x["p"]) == (None, None, None)
    assert x["ratio"] == pytest.approx(math.exp(truth["estimate"]), rel=1e-5)
    assert (x["ratio_low"], x["ratio_high"]) == (None, None)
    info = separated["table"].info
    assert info["separated"] == ["x"] and info["covariance"] == "model"
    [said] = [c for c in separated["table"].concerns if "not reported" in c]
    assert said.startswith(f"`x` separates the outcome in {m} of the {m} imputed copies, so its "
                           f"pooled interval and p-value are not reported, only its estimate: a "
                           f"Wald interval on a separated coefficient's Firth estimate is not "
                           f"certified")
    assert "(Heinze & Schemper 2002)" in said
    assert said.endswith("To report it, read the column another way or combine its sparse "
                         "levels, or analyze complete cases, where one table is fit and none is "
                         "pooled.")
    first, cases = info["exits"]
    assert first == {"label": "Read `x` another way, or combine its sparse levels",
                     "decision": None}
    assert cases["decision"]["kind"] == "set_missing"
    assert cases["decision"]["strategy"] == "complete_case"
    from turbotab.core.decisions import parse_decision

    parse_decision(cases["decision"])  # a decision the decision model reads


def test_a_coefficient_no_copy_separates_keeps_rubins_wald_interval_on_t(separated):
    """z separates in no copy: its pooled interval and p-value are reported, and are Rubin's Wald
    interval on t from each copy's Firth estimate and standard error, computed by scipy and
    numpy (the intercept likewise)."""
    for j, name in ((2, "z"), (0, "(intercept)")):
        truth = _rubin_by_hand(separated["imputations"].frames, separated["y"], j)
        row = separated["rows"][name]
        assert row["ci_low"] == pytest.approx(truth["ci_low"], rel=1e-5, abs=1e-6), name
        assert row["ci_high"] == pytest.approx(truth["ci_high"], rel=1e-5, abs=1e-6), name
        assert row["p"] == pytest.approx(truth["p"], rel=1e-4, abs=1e-8), name
        assert row["df"] == pytest.approx(truth["df"], rel=1e-3), name
    z = separated["rows"]["z"]
    assert z["ratio_low"] == pytest.approx(math.exp(z["ci_low"]), rel=1e-12)


def test_the_pooled_caption_says_its_intervals_are_wald_ones_from_firths_standard_errors(separated):
    """The pooled numbers are Wald intervals on t from each copy's Firth estimate and standard
    error; the caption says that, and no longer says profile intervals or likelihood-ratio
    p-values, which no copy computed."""
    m = separated["imputations"].m
    caption = separated["table"].info["caption"]
    assert caption == (
        f"Multiple imputation, m = {m}: each completed table analyzed as follows, then pooled by "
        f"Rubin's rules, each interval on t with Barnard–Rubin degrees of freedom. Estimates are "
        f"log-odds of `1.0` against `0.0`; each odds ratio is exp(estimate). Firth-penalized "
        f"logistic regression, because the maximum-likelihood estimate does not exist: each "
        f"estimate with its standard error from the information at the penalized estimate, which "
        f"Rubin's rules pool into Wald intervals and p-values; no profile interval is computed. "
        f"Estimated from all 150 analyzed rows.")
    assert "profile penalized-likelihood intervals" not in caption
    assert "likelihood-ratio" not in caption


def test_a_table_no_copy_separates_withholds_nothing():
    """With no separation every copy is a maximum-likelihood fit: every pooled interval is
    reported, no concern of a withheld one, and the caption is the logistic model's Wald one."""
    frame, y = _separated_rows(separate=False, seed=8)
    out = _pooled(frame, y)
    assert all(t.info["estimator"] == "logistic regression (maximum likelihood)"
               for t in out["tables"])
    assert all(r["ci_low"] is not None and r["p"] is not None for r in out["rows"].values())
    info = out["table"].info
    assert info["separated"] == [] and info["covariance"] == "model" and info["exits"] == []
    assert not [c for c in out["table"].concerns if "not reported" in c]
    assert info["caption"].endswith("95% Wald intervals from the logistic model's information. "
                                    "Estimated from all 150 analyzed rows.")


def _table(estimator: str, separated: list[str], est: float) -> I.InferenceTable:
    rows = [{"feature": "(intercept)", "estimate": 0.1, "se": 0.2, "df": None},
            {"feature": "x", "estimate": est, "se": 1.0, "df": None},
            {"feature": "z", "estimate": 0.5, "se": 0.2, "df": None}]
    caption = (I.POOLED_FIRTH_CAPTION if estimator == I.FIRTH else
               "95% Wald intervals from the logistic model's information.")
    return I.InferenceTable(rows, {"estimator": estimator, "separated": separated,
                                   "caption": caption}, [])


def test_copies_of_both_kinds_say_how_many_are_maximum_likelihood_and_withhold_the_separated():
    """Two copies, x separated in one: its interval is withheld (the construction's one copy of
    two), z's kept, and the caption is the Firth copy's with the count of each kind; without a
    missing-values answer (the data's own copies) complete cases are not offered."""
    tables = [_table("logistic regression (maximum likelihood)", [], 2.0), _table(I.FIRTH, ["x"], 4.0)]
    rows = pool_rows([t.rows for t in tables])
    found = I.pooled_firth(tables, rows)
    by = {r["feature"]: r for r in rows}
    assert (by["x"]["ci_low"], by["x"]["p"]) == (None, None) and by["x"]["estimate"] == 3.0
    assert by["z"]["ci_low"] is not None and by["z"]["p"] is not None
    assert found.caption == (f"{I.POOLED_FIRTH_CAPTION} In 1 of the 2 copies a column separates "
                             f"the outcome and the copy is fit so; the other 1 is a "
                             f"maximum-likelihood fit, its standard errors from the logistic "
                             f"model's information.")
    assert found.concern.startswith("`x` separates the outcome in 1 of the 2 imputed copies")
    assert found.concern.endswith("To report it, read the column another way or combine its "
                                  "sparse levels.")
    assert [e["label"] for e in found.info["exits"]] == [
        "Read `x` another way, or combine its sparse levels"]


# ── the effects stage's pooled tables ────────────────────────────────────────


def test_the_effects_pooled_tables_withhold_a_separated_term_and_say_wald_from_firth(tmp_path):
    """Through the design and effects stages: every smoker has the event, so `smoking` separates
    the outcome in every copy; under multiple imputation Model 2's pooled appendix term for it
    keeps its estimate and has no interval or p-value, the model says why, its caption says the
    intervals are Wald ones from Firth's standard errors, and the exposure's interval stands."""
    from turbotab.core.tests.acceptance import estimand_fixtures as ef

    frame = ef.cohort(400, seed=12)
    frame.loc[frame["smoking"] == 1, "dm"] = "yes"
    frame.loc[np.random.default_rng(4).random(len(frame)) < 0.15, "age"] = np.nan
    roles = {c: r for c, r in ef.ROLES.items() if c != "bmi"}
    answers = {c: a for c, a in ef.ANSWERS.items() if c != "bmi"}
    st = ef.state(target="dm", task="binary", event="yes", measure="odds_ratio", roles=roles,
                  answers=answers, missing={"strategy": "multiple_imputation", "m": 20})
    run = ef.run(frame, tmp_path / "mi", st, fit=False)
    two = ef.sequence(run["effects"])["model_2"]
    [fiber] = two["effects"]
    assert fiber["feature"] == "fiber" and fiber["ci_low"] is not None and fiber["p"] is not None
    [appendix] = [a for a in run["effects"]["families"][0]["appendix"] if a["key"] == "model_2"]
    [smoking] = [t for t in appendix["terms"] if t["feature"] == "smoking"]
    assert smoking["estimate"] is not None and smoking["se"] is not None
    assert (smoking["ci_low"], smoking["ci_high"], smoking["p"]) == (None, None, None)
    assert two["inference"]["separated"] == ["smoking"]
    assert two["inference"]["covariance"] == "model"
    assert ("Firth-penalized logistic regression, because the maximum-likelihood estimate does not "
            "exist: each estimate with its standard error from the information at the penalized "
            "estimate, which Rubin's rules pool into Wald intervals and p-values; no profile "
            "interval is computed.") in two["inference"]["caption"]
    assert two["inference"]["caption"].startswith("Multiple imputation, m = ")
    assert any(c.startswith("`smoking` separates the outcome in ") and "not reported" in c
               for c in two["concerns"])

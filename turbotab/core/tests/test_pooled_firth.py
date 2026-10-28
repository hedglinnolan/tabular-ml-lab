"""A logistic table pooled over multiple imputation where a column separates the outcome
(``models.inference.for_pooling`` and ``pooled_firth``; ``stages.modeling.pooled_table``;
``stages.effects``; claim ``inference.pooled_separation``).

Rubin's rules read each completed copy's estimate and standard error and make the interval and
p-value themselves (``imputation.pool_rows``), so a Firth copy computes no profile, nor do its form
tests' refits (the companion quintiles beside a declared spline, the trend), and the pooled
estimates are the ones the copies' own tables pool to (a table not pooled serves its estimates
alone where the data separate, T5d), the standard errors Firth's. The pooled numbers are
Wald intervals on t from Firth's standard errors, and the table says so; a coefficient that
separates the outcome in any copy keeps its estimate alone, its standard error, degrees of freedom
and fraction of missing information withheld with its interval and p-value (with them the
withheld interval is estimate ± t·se), and the copies' own separation concern, which says the
table reports Firth's fit, is replaced by the pooled one.

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
# The claim's card (``inference.pooled_separation``), which the concern gives as its reason.
CARD = ("a Wald interval on a separated coefficient's Firth estimate is not certified, and the "
        "profile penalized-likelihood interval advised in its place is not one Rubin's rules pool")


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


def _pooled(frame: pd.DataFrame, y: np.ndarray,
            roles: dict[str, str] | None = None) -> dict[str, Any]:
    """The engine's imputations of ``z`` under inference and the pooled logistic table, as the
    fit stage pools it, with every call of ``firth_profile`` counted."""
    roles = roles or {"x": "exposure", "z": "covariate"}
    st = ProjectState(target="y", task="binary", purpose="inference", roles=roles,
                      missing=MissingSpec(strategy="multiple_imputation", m=20))
    spec = design_spec(st, frame, list(roles))
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
        # For the comparison, each copy's table made as an unpooled table is (the base profiled
        # every coefficient there; since T5d none, the data separating), then pooled by Rubin's
        # rules.
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
    maximized by scipy's BFGS on its gradient, Firth's modified score Xᵀ(y − μ + h(½ − μ)) with h
    the hat matrix's diagonal, and the inverse information XᵀWX at it."""
    def negative(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = X @ beta
        mu = 1 / (1 + np.exp(-eta))
        w = mu * (1 - mu)
        info = X.T @ (X * w[:, None])
        sign, logdet = np.linalg.slogdet(info)
        h = w * np.einsum("ij,jk,ik->i", X, np.linalg.inv(info), X)
        score = X.T @ (y - mu + h * (0.5 - mu))
        return -(float(np.sum(y * eta - np.logaddexp(0.0, eta))) + 0.5 * logdet), -score

    found = optimize.minimize(negative, np.zeros(X.shape[1]), method="BFGS", jac=True,
                              options={"gtol": 1e-10, "maxiter": 10_000})
    mu = 1 / (1 + np.exp(-(X @ found.x)))
    return found.x, np.linalg.inv(X.T @ (X * (mu * (1 - mu))[:, None]))


def _x_and_z(X_k: pd.DataFrame) -> np.ndarray:
    return np.column_stack([np.ones(len(X_k)), X_k["x"].to_numpy(float), X_k["z"].to_numpy(float)])


def _rubin_by_hand(frames: list[pd.DataFrame], y: np.ndarray, j: int,
                   design: Any = _x_and_z) -> dict[str, float]:
    """Coefficient ``j`` of ``design(copy)`` ([intercept, x, z] unless given) over the completed
    copies, each fit by :func:`_firth_by_scipy`, pooled by Rubin's rules written out."""
    q, u = [], []
    for X_k in frames:
        beta, cov = _firth_by_scipy(design(X_k), y)
        q.append(beta[j])
        u.append(cov[j, j])
    return _rubin(q, u)


def _rubin(q: list[float], u: list[float]) -> dict[str, float]:
    """Rubin's rules written out (Rubin 1987: ν = (m − 1)/λ², the complete-data df infinite)."""
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
    (the base's per-copy tables called it for every coefficient of every copy; since T5d a table
    not pooled calls it for none, the data separating), and Rubin's rules over the copies give the
    estimates the copies' own tables pool to, to 1e-12, and the standard errors Rubin's total
    variance gives: Firth's estimate and Rubin's total variance, recomputed by scipy and numpy.
    The table reports them, but for `x`'s standard error, which is withheld (it separates)."""
    m = separated["imputations"].m
    assert m >= 20 and all(t.info["estimator"] == "Firth-penalized logistic regression"
                            for t in separated["tables"])
    assert separated["profiles"] == 0
    assert separated["full_profiles"] == 0  # the base profiled three coefficients a copy
    rubin = {r["feature"]: r for r in pool_rows([t.rows for t in separated["tables"]])}
    for name in ("(intercept)", "x", "z"):
        ours, base = rubin[name], separated["full"][name]
        assert ours["estimate"] == pytest.approx(base["estimate"], rel=1e-12, abs=1e-12), name
        shown = separated["rows"][name]
        assert shown["estimate"] == pytest.approx(base["estimate"], rel=1e-12, abs=1e-12), name
        assert shown["se"] == (None if name == "x" else pytest.approx(ours["se"], rel=1e-12)), name
    frames = separated["imputations"].frames
    for j, name in enumerate(("(intercept)", "x", "z")):
        truth = _rubin_by_hand(frames, separated["y"], j)
        assert rubin[name]["estimate"] == pytest.approx(truth["estimate"], abs=1e-6), name
        assert rubin[name]["se"] == pytest.approx(truth["se"], rel=1e-5), name


def test_a_table_not_pooled_serves_its_estimates_alone(separated):
    """Outside the pooling (complete cases, or any one table) the Firth table on separated rows
    serves its penalized estimates alone, refusing every profile interval and p-value in words
    that name `x`, which separates (T5d, V5c2-1: held at a value the penalized likelihood can then
    have more than one maximum; before, it served each coefficient's profile interval)."""
    for table in separated["full_tables"][:3]:
        assert table.info["refused"].startswith("`x` separates the outcome: a cut on it puts "
                                                "every row on its outcome's side")
        assert all(r["estimate"] is not None and r["ci_low"] is None and r["p"] is None
                   for r in table.rows)
        assert "profile penalized-likelihood intervals" not in table.info["caption"]


# ── F3: the pooled table's words, and the separated coefficient withheld ─────


def test_a_coefficient_separated_in_the_copies_keeps_its_pooled_estimate_and_withholds_its_interval(
        separated):
    """x separates the outcome in every copy by construction: its pooled estimate stands
    (Firth's, pooled) with its odds ratio and its Monte Carlo error (the copies' estimates' spread,
    √(B/m)), and its standard error, degrees of freedom and fraction of missing information are
    withheld with its interval, p-value and odds-ratio interval, since estimate ± t·se would give
    the withheld interval back; the table names it, says why and gives the ways forward, complete
    cases a decision."""
    m = separated["imputations"].m
    x = separated["rows"]["x"]
    truth = _rubin_by_hand(separated["imputations"].frames, separated["y"], 1)
    assert x["estimate"] == pytest.approx(truth["estimate"], abs=1e-6)
    assert (x["se"], x["df"], x["fmi"]) == (None, None, None)
    assert (x["ci_low"], x["ci_high"], x["p"]) == (None, None, None)
    assert x["ratio"] == pytest.approx(math.exp(truth["estimate"]), rel=1e-5)
    assert (x["ratio_low"], x["ratio_high"]) == (None, None)
    q = [_firth_by_scipy(_x_and_z(X_k), separated["y"])[0][1]
         for X_k in separated["imputations"].frames]
    assert x["mc_se"] == pytest.approx(float(np.std(q, ddof=1)) / math.sqrt(m), rel=1e-4)
    info = separated["table"].info
    assert info["separated"] == ["x"] and info["covariance"] == "model"
    [said] = [c for c in separated["table"].concerns if "not reported" in c]
    assert said == (f"`x` separates the outcome in every one of the {m} imputed copies, so its "
                    f"pooled standard error, interval and p-value are not reported, only its "
                    f"estimate: {CARD} (Heinze & Schemper 2002). To report it, read `x` another "
                    f"way or combine its sparse levels, or analyze complete cases, where one table "
                    f"is fit and none is pooled; where the complete cases separate the outcome "
                    f"too, that table reports no standard error, interval or p-value either.")
    # The copies' own concern says the table reports Firth's fit; the pooled one replaces it.
    assert I.separation_concern(["x"]) in separated["tables"][0].concerns
    assert I.separation_concern(["x"]) not in separated["table"].concerns
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
    numpy. The intercept likewise: only x's log-odds is infinite (the rows with x = 0 hold both
    outcomes at every z), so no copy finds the intercept infinite, and its interval stands."""
    assert all(t.infinite == ["x"] for t in separated["tables"])
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
    two), z's kept, and the caption says the maximum-likelihood estimate does not exist in the
    Firth copy alone, which is fit by Firth's penalized likelihood, and the other is a
    maximum-likelihood fit; without a missing-values answer (the data's own copies) complete
    cases are not offered."""
    tables = [_table("logistic regression (maximum likelihood)", [], 2.0), _table(I.FIRTH, ["x"], 4.0)]
    rows = pool_rows([t.rows for t in tables])
    found = I.pooled_firth(tables, rows)
    by = {r["feature"]: r for r in rows}
    assert (by["x"]["ci_low"], by["x"]["p"]) == (None, None) and by["x"]["estimate"] == 3.0
    assert by["z"]["ci_low"] is not None and by["z"]["p"] is not None
    assert found.caption == (
        "In 1 of the 2 copies a column separates the outcome, so the maximum-likelihood estimate "
        "does not exist in it and it is fit by Firth's penalized likelihood, its standard errors "
        "from the information at the penalized estimate; the other 1 is a maximum-likelihood fit, "
        "its standard errors from the logistic model's information. Rubin's rules pool each "
        "copy's estimates and standard errors into Wald intervals and p-values; no profile "
        "interval is computed.")
    assert "because the maximum-likelihood estimate does not exist" not in found.caption
    assert found.concern.startswith("`x` separates the outcome in 1 of the 2 imputed copies, so ")
    assert (by["x"]["se"], by["x"]["df"], by["x"]["fmi"]) == (None, None, None)
    assert by["z"]["se"] is not None
    assert found.concern.endswith("To report it, read `x` another way or combine its sparse "
                                  "levels.")
    assert [e["label"] for e in found.info["exits"]] == [
        "Read `x` another way, or combine its sparse levels"]


def test_the_copies_a_coefficient_separates_in_are_counted_as_they_are():
    """Three copies: where every named coefficient separates in the same number of copies the
    concern says that number once ("every one of" when it is all of them); where the numbers
    differ it gives each one's; and with several Firth copies the caption says "each is"."""
    ml = "logistic regression (maximum likelihood)"
    their = "so their pooled standard errors, intervals and p-values are not reported, only their "
    cases = [
        ([_table(I.FIRTH, ["x", "z"], 4.0)] * 3,
         f"`x` and `z` separate the outcome in every one of the 3 imputed copies, {their}"),
        ([_table(ml, [], 2.0)] + [_table(I.FIRTH, ["x", "z"], 4.0)] * 2,
         f"`x` and `z` separate the outcome in 2 of the 3 imputed copies, {their}"),
        ([_table(I.FIRTH, ["x"], 4.0)] + [_table(I.FIRTH, ["x", "z"], 4.0)] * 2,
         "`x` and `z` separate the outcome in some of the 3 imputed copies (`x` in 3, `z` in 2), "
         f"{their}"),
    ]
    for tables, start in cases:
        found = I.pooled_firth(tables, pool_rows([t.rows for t in tables]))
        assert found.concern.startswith(start), found.concern
        assert found.concern.endswith("To report them, read `x` and `z` another way or combine "
                                      "their sparse levels.")
    tables = cases[1][0]
    assert I.pooled_firth(tables, pool_rows([t.rows for t in tables])).caption.startswith(
        "In 2 of the 3 copies a column separates the outcome, so the maximum-likelihood estimate "
        "does not exist in them and each is fit by Firth's penalized likelihood, its standard "
        "errors from the information at the penalized estimate; the other 1 is a "
        "maximum-likelihood fit, its standard errors")


# ── the intercept: infinite beside a separated categorical column ────────────


def _reference_level_all_events(n: int = 240, seed: int = 5) -> tuple[pd.DataFrame, np.ndarray]:
    """``g`` with levels a, b and c, a the reference; every row at ``a`` has the event, so in
    every completed copy the intercept's maximum-likelihood log-odds is +∞ and ``g_b``'s and
    ``g_c``'s −∞: the direction (1, −1, −1, 0) on [intercept, g_b, g_c, z] sends the ``a`` rows'
    log-odds to +∞ and leaves every other row's as it is. ``z`` is blank on about 30% of rows,
    and no direction moves it (the b and c rows hold both outcomes at every z)."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    g = rng.choice(["a", "b", "c"], n, p=[0.15, 0.45, 0.40])
    y = (rng.random(n) < 1 / (1 + np.exp(-(-0.3 + 0.6 * z)))).astype(float)
    y = np.where(g == "a", 1.0, y)
    frame = pd.DataFrame({"g": g, "z": np.where(rng.random(n) < 0.3, np.nan, z)})
    return frame, y


def _g_and_z(X_k: pd.DataFrame) -> np.ndarray:
    return np.column_stack([np.ones(len(X_k)), (X_k["g"] == "b").to_numpy(float),
                            (X_k["g"] == "c").to_numpy(float), X_k["z"].to_numpy(float)])


@pytest.fixture(scope="module")
def categorical() -> dict[str, Any]:
    frame, y = _reference_level_all_events()
    out = _pooled(frame, y, roles={"g": "exposure", "z": "covariate"})
    out["y"] = y
    return out


def test_an_intercept_infinite_in_the_copies_keeps_its_estimate_and_withholds_its_interval(
        categorical):
    """The intercept's log-odds is infinite in every copy with `g_b`'s and `g_c`'s, by
    construction: its pooled estimate stands (Firth's, pooled, recomputed by scipy and numpy), its
    standard error, interval and p-value are withheld as theirs are, and the concern says so; the
    copies' own concern still names the columns alone. `z`, which no direction moves, keeps
    Rubin's Wald interval on t."""
    m = categorical["imputations"].m
    frames, y, rows = categorical["imputations"].frames, categorical["y"], categorical["rows"]
    assert all(t.info["separated"] == ["g_b", "g_c"] for t in categorical["tables"])
    for j, name in enumerate(("(intercept)", "g_b", "g_c")):
        truth = _rubin_by_hand(frames, y, j, _g_and_z)
        row = rows[name]
        assert row["estimate"] == pytest.approx(truth["estimate"], abs=1e-6), name
        assert (row["se"], row["df"], row["fmi"]) == (None, None, None), name
        assert (row["ci_low"], row["ci_high"], row["p"]) == (None, None, None), name
    truth = _rubin_by_hand(frames, y, 3, _g_and_z)
    z = rows["z"]
    assert z["ci_low"] == pytest.approx(truth["ci_low"], rel=1e-5, abs=1e-6)
    assert z["ci_high"] == pytest.approx(truth["ci_high"], rel=1e-5, abs=1e-6)
    assert z["p"] == pytest.approx(truth["p"], rel=1e-4, abs=1e-8)
    [said] = [c for c in categorical["table"].concerns if "not reported" in c]
    assert said == (
        f"`g_b` and `g_c` separate the outcome in every one of the {m} imputed copies, so their "
        f"pooled standard errors, intervals and p-values are not reported, only their estimates: "
        f"{CARD} (Heinze & Schemper 2002). The standard error, interval and p-value of the "
        f"intercept are withheld too: its maximum-likelihood value is infinite in every one of "
        f"the {m} imputed copies. To report them, read `g` another way or combine its sparse "
        f"levels, or analyze complete cases, where one table is fit and none is pooled; where "
        f"the complete cases separate the outcome too, that table reports no standard error, "
        f"interval or p-value either.")


def test_the_way_forward_names_the_column_a_separated_indicator_comes_from(categorical):
    """`g_b` and `g_c` are indicators of `g`, with no levels of their own: the way forward names
    `g`, the column the researcher can read another way or whose sparse levels combine."""
    first, cases = categorical["table"].info["exits"]
    assert first == {"label": "Read `g` another way, or combine its sparse levels",
                     "decision": None}
    assert cases["decision"]["strategy"] == "complete_case"


# ── the all-components relative effects ──────────────────────────────────────

SOURCES = {"protein_g": 4.0, "carb_g": 4.0, "fat_g": 9.0, "alc_g": 7.0}  # kcal per gram


def _energy_frame(separate: bool, n: int = 600, seed: int = 91) -> pd.DataFrame:
    """Four energy sources in grams, total energy `kcal` (their kcal and some of other sources),
    a yes/no outcome `y`, and `age` blank on about a quarter of the rows. About 12% of the rows
    drink; with ``separate`` every drinker has the event."""
    rng = np.random.default_rng(seed)
    age = rng.normal(50, 12, n)
    size = 2100 + 450 * rng.normal(size=n) - 9 * (age - 50)
    kp = size * np.clip(rng.normal(0.16, 0.03, n), 0.05, 0.4)
    kf = size * np.clip(rng.normal(0.34, 0.05, n), 0.1, 0.6)
    kc = size * np.clip(rng.normal(0.40, 0.05, n), 0.1, 0.7)
    ka = np.where(rng.random(n) < 0.12, size * rng.uniform(0.02, 0.08, n), 0.0)
    other = size * rng.uniform(0.02, 0.1, n)
    eta = 0.002 * kp + 0.001 * kc + 0.03 * (age - 50)
    y = (rng.random(n) < 1 / (1 + np.exp(-(eta - eta.mean())))).astype(int)
    if separate:
        y = np.where(ka > 0, 1, y)
    frame = pd.DataFrame({"protein_g": kp / 4, "carb_g": kc / 4, "fat_g": kf / 9, "alc_g": ka / 7,
                          "kcal": kp + kf + kc + ka + other, "age": age, "y": y})
    frame.loc[rng.random(n) < 0.25, "age"] = np.nan
    return frame


def _energy_fit(folder: Any, separate: bool) -> dict[str, Any]:
    """The design and fit stages under the all-components model and multiple imputation."""
    from turbotab.core.decisions import EnergyAdjustment, SplitSpec
    from turbotab.core.stages.modeling import design_stage, fit_stage
    from turbotab.core.tests import modeling_fixtures as mf

    frame = _energy_frame(separate)
    roles = {"protein_g": "exposure", "carb_g": "exposure", "fat_g": "exposure",
             "alc_g": "exposure", "kcal": "energy", "age": "covariate"}
    paths = mf.ingest_frame(frame, folder)
    st = mf.state(roles=roles, target="y", task="binary", models=["linear"], purpose="inference",
                  energy_adjustment=EnergyAdjustment(method="all_components", energy_column="kcal",
                                                     nutrients=list(SOURCES)),
                  missing=MissingSpec(strategy="multiple_imputation"),
                  split=SplitSpec(holdout=0.0, seed=0, folds=5), column_units=mf.grams(*SOURCES))
    split = mf.split_bundle(np.arange(len(frame)), holdout=0.0)
    info = mf.target_info("binary", "y")
    design = design_stage(mf.context(st, {"split": split, "target_info": info}, paths))
    fit = fit_stage(mf.context(st, {"design": design, "split": split, "target_info": info}, paths))
    [model] = [mm for mm in fit.data["models"] if mm["family"] == "linear"]
    imputations = fit.objects["imputations"]
    return {"model": model, "rows": {r["feature"]: r for r in model["coefficients"]},
            "fits": imputations["fits"]["linear"], "frames": imputations["frames"],
            "y": frame["y"].to_numpy(float)}


@pytest.fixture(scope="module")
def drinkers_all_events(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return _energy_fit(tmp_path_factory.mktemp("drinkers"), separate=True)


def test_a_relative_effect_resting_on_a_separated_source_keeps_its_estimate_and_withholds_its_interval(
        drinkers_all_events):
    """Every drinker has the event, so `kcal_from_alc_g`'s log-odds is +∞ in every copy, along
    its own direction alone (the non-drinkers hold both outcomes at every value of the rest).
    Each relative effect θ_j = β_j − Σ_k w_k β_k (Tomova et al. 2022) reads β_alc with weight 1
    (j = alc) or −w_alc (the drinkers' kcal's share of the other sources', above zero), so each
    is infinite along it too: every relative row keeps its pooled estimate and has its standard
    error, interval, p-value and odds-ratio interval withheld, and the concern says so, naming
    `alc_g` as the column to read another way. The other coefficients' intervals stand."""
    rows, model = drinkers_all_events["rows"], drinkers_all_events["model"]
    m = len(drinkers_all_events["frames"])
    relative = [f"{n}_relative" for n in SOURCES]
    for name in [*relative, "kcal_from_alc_g"]:
        row = rows[name]
        assert row["estimate"] is not None and row["se"] is None, name
        assert (row["ci_low"], row["ci_high"], row["p"]) == (None, None, None), name
        assert (row["ratio_low"], row["ratio_high"]) == (None, None), name
    for name in ("kcal_from_protein_g", "kcal_from_carb_g", "kcal_from_fat_g", "kcal_from_other",
                 "age"):
        assert rows[name]["ci_low"] is not None and rows[name]["p"] is not None, name
    [said] = [c for c in model["concerns"] if "not reported" in c]
    assert said.startswith(f"`kcal_from_alc_g` separates the outcome in every one of the {m} "
                           f"imputed copies, so its pooled standard error, interval and p-value "
                           f"are not reported")
    assert (f"The standard errors, intervals and p-values of `protein_g_relative`, "
            f"`carb_g_relative`, `fat_g_relative` and `alc_g_relative` are withheld too: their "
            f"maximum-likelihood values are infinite in every one of the {m} imputed copies. To "
            f"report them, read `alc_g` another way or combine its sparse levels") in said


def test_relative_effects_no_copy_separates_keep_rubins_wald_interval_on_t(tmp_path):
    """No source separates the outcome: every relative row keeps its pooled interval, and each is
    Rubin's Wald interval on t from each copy's θ_j = β_j − Σ_k w_k β_k, per gram (× its kcal per
    gram), with variance c'Vc, from statsmodels' logistic fit on the copy's model matrix and
    pooled by Rubin's rules written out."""
    import statsmodels.api as sm

    from turbotab.core.models.linear import model_matrix

    found = _energy_fit(tmp_path, separate=False)
    assert not [c for c in found["model"]["concerns"] if "not reported" in c]
    per: dict[str, tuple[list[float], list[float]]] = {n: ([], []) for n in SOURCES}
    for fitted, X_k in zip(found["fits"], found["frames"]):
        M = model_matrix(fitted, X_k).astype(float)
        logit = sm.Logit(found["y"], sm.add_constant(M, has_constant="add")).fit(disp=0,
                                                                                 maxiter=200)
        names = list(logit.params.index)
        kcal = [c for c in M.columns if str(c).startswith("kcal_from_")]
        for n, factor in SOURCES.items():
            j = f"kcal_from_{n}"
            others = [k for k in kcal if k != j]
            total = float(M[others].mean().sum())
            c = np.zeros(len(names))
            c[names.index(j)] = factor
            for k in others:
                c[names.index(k)] = -factor * float(M[k].mean()) / total
            per[n][0].append(float(c @ logit.params.to_numpy()))
            per[n][1].append(float(c @ logit.cov_params().to_numpy() @ c))
    for n in SOURCES:
        truth, row = _rubin(*per[n]), found["rows"][f"{n}_relative"]
        for key in ("estimate", "se", "ci_low", "ci_high"):
            assert row[key] == pytest.approx(truth[key], rel=1e-5, abs=1e-9), (n, key)
        assert row["p"] == pytest.approx(truth["p"], rel=1e-4, abs=1e-10), n


# ── the effects stage's pooled tables ────────────────────────────────────────


def test_the_effects_pooled_tables_withhold_a_separated_term_and_say_wald_from_firth(tmp_path):
    """Through the design and effects stages: every smoker has the event, so `smoking` separates
    the outcome in every copy; under multiple imputation Model 2's pooled appendix term for it
    keeps its estimate and has no standard error, interval or p-value, the model says why (and no
    longer that the table reports Firth's fit, which its copies say), its caption says the
    intervals are Wald ones on t from Firth's standard errors, and the exposure's interval
    stands."""
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
    assert smoking["estimate"] is not None and (smoking["se"], smoking["df"]) == (None, None)
    assert (smoking["ci_low"], smoking["ci_high"], smoking["p"]) == (None, None, None)
    assert two["inference"]["separated"] == ["smoking"]
    assert two["inference"]["covariance"] == "model"
    assert ("Firth-penalized logistic regression, because the maximum-likelihood estimate does not "
            "exist: each estimate with its standard error from the information at the penalized "
            "estimate, which Rubin's rules pool into Wald intervals and p-values; no profile "
            "interval is computed.") in two["inference"]["caption"]
    m = two["inference"]["missing"]["m"]
    assert two["inference"]["caption"].startswith(
        f"Multiple imputation, m = {m}: each completed copy analyzed as follows, then pooled by "
        f"Rubin's rules, each interval on t with Barnard–Rubin degrees of freedom. ")
    assert any(c.startswith("`smoking` separates the outcome in ") and "not reported" in c
               for c in two["concerns"])
    assert I.separation_concern(["smoking"]) not in two["concerns"]


def test_the_effects_relative_effect_resting_on_a_separated_source_is_withheld(tmp_path):
    """Through the design and effects stages under the all-components model: every drinker has
    the event, so in every copy `kcal_from_alc_g` is infinite, and with it each relative effect
    (as the fit's table, above). Model 2's `protein_g_relative`, the exposure's substitution, keeps
    its estimate and has no interval or p-value, as do the other sources' relative rows in the
    appendix, and the model says why; `kcal_from_protein_g`'s interval stands."""
    from turbotab.core import decisions as d
    from turbotab.core.tests import modeling_fixtures as mf
    from turbotab.core.tests.acceptance import estimand_fixtures as ef

    frame = _energy_frame(separate=True).assign(pid=lambda f: np.arange(len(f)))
    open_timing = {"causes_exposure": "unknown", "causes_outcome": "unknown",
                   "after_exposure": "no"}
    roles = {"pid": "identifier", "protein_g": "exposure", "carb_g": "covariate",
             "fat_g": "covariate", "alc_g": "covariate", "kcal": "energy", "age": "covariate"}
    answers = {"age": ef.CONFOUNDER, "carb_g": open_timing, "fat_g": open_timing,
               "alc_g": open_timing}
    st = ef.state(target="y", task="binary", event="1", exposure="protein_g",
                  measure="odds_ratio", roles=roles, answers=answers, lens=["dietary"],
                  missing={"strategy": "multiple_imputation"}, column_units=mf.grams(*SOURCES),
                  energy_adjustment=d.EnergyAdjustment(method="all_components",
                                                       energy_column="kcal",
                                                       nutrients=list(SOURCES)))
    run = ef.run(frame, tmp_path, st, fit=False)
    two = ef.sequence(run["effects"])["model_2"]
    shown = {r["feature"]: r for r in two["effects"]}
    assert set(shown) == {"kcal_from_protein_g", "protein_g_relative"}
    assert shown["kcal_from_protein_g"]["ci_low"] is not None
    assert shown["protein_g_relative"]["estimate"] is not None
    assert (shown["protein_g_relative"]["ci_low"], shown["protein_g_relative"]["p"]) == (None, None)
    [appendix] = [a for a in run["effects"]["families"][0]["appendix"] if a["key"] == "model_2"]
    terms = {t["feature"]: t for t in appendix["terms"]}
    for name in ("carb_g_relative", "fat_g_relative", "alc_g_relative", "kcal_from_alc_g"):
        assert terms[name]["estimate"] is not None, name
        assert (terms[name]["ci_low"], terms[name]["p"]) == (None, None), name
    assert any("are withheld too" in c and "`protein_g_relative`" in c for c in two["concerns"])


# ── copies of both kinds: the pooled concerns whichever comes first ──────────


def _sometimes_separated(n: int = 240, seed: int = 7) -> tuple[pd.DataFrame, np.ndarray]:
    """``x`` 0/1, every row recording x = 1 has the event; ``x`` is blank on 24 rows, four of them
    without the event, so a copy that imputes x = 1 on any of those four is not separated by x
    and one that imputes x = 0 on all four is. ``z`` is blank on about 20% of the rows."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    x = (rng.random(n) < 0.2).astype(float)
    y = (rng.random(n) < 1 / (1 + np.exp(-(-0.3 + 0.8 * z)))).astype(float)
    y = np.where(x == 1, 1.0, y)
    blank = rng.choice(np.where(y == 0)[0], 4, replace=False)
    x[blank] = np.nan
    x[rng.choice(np.setdiff1d(np.arange(n), blank), 20, replace=False)] = np.nan
    return pd.DataFrame({"x": x, "z": np.where(rng.random(n) < 0.2, np.nan, z)}), y


@pytest.mark.parametrize("first", ["Firth", "maximum likelihood"])
def test_copies_of_both_kinds_leave_out_the_copies_own_separation_concern_whichever_is_first(
        first):
    """In the copies that impute x = 0 on every blank row without the event, every x = 1 row has
    the event and x separates (counted here from the copies themselves); in the rest it does not.
    With the copies put in either order, a Firth copy first or a maximum-likelihood one, the
    pooled table's concerns name the separated count and never repeat the Firth copy's own
    concern, which says the table reports Firth's fit (true of the copy, false of a table pooled
    over both kinds)."""
    import dataclasses

    frame, y = _sometimes_separated()
    roles = {"x": "exposure", "z": "covariate"}
    st = ProjectState(target="y", task="binary", purpose="inference", roles=roles,
                      missing=MissingSpec(strategy="multiple_imputation", m=20))
    spec = design_spec(st, frame, list(roles))
    imputations = impute_for_inference(spec, frame, y, "binary", seed=1)
    m = imputations.m
    separated = [bool(np.all(y[X_k["x"].to_numpy() == 1] == 1)) for X_k in imputations.frames]
    k = sum(separated)
    assert 0 < k < m
    wanted = first == "Firth"
    order = sorted(range(m), key=lambda j: separated[j] != wanted)
    copies = dataclasses.replace(imputations, frames=[imputations.frames[j] for j in order])
    pipe = build_pipeline(spec, LINEAR, "binary", "inference", len(frame), 2)
    collected: dict[str, Any] = {}
    table, _, _, _ = pooled_table(LINEAR, pipe, copies, y, task="binary", clusters=I.INDEPENDENT,
                                  outcome=I.Outcome(name="y"), survey=None, design=None, spec=spec,
                                  rows="all", fit=lambda model, X_k: fit_pipeline(model, X_k, y),
                                  collect=collected)
    kinds = [t.info["estimator"] == I.FIRTH for t in collected["tables"]]
    assert kinds == [separated[j] for j in order] and kinds[0] == wanted
    assert not [c for c in table.concerns if "the table reports Firth" in c]
    assert I.separation_concern(["x"]) not in table.concerns
    [said] = [c for c in table.concerns if "not reported" in c]
    assert said.startswith(f"`x` separates the outcome in {k} of the {m} imputed copies, so its "
                           f"pooled standard error, interval and p-value are not reported, only "
                           f"its estimate: ")
    assert f"In {k} of the {m} copies a column separates the outcome" in table.info["caption"]


# ── the companion quintiles beside a declared spline ─────────────────────────


def _companion_fit(folder: Any) -> dict[str, Any]:
    """The design and fit stages under multiple imputation (`age` blank on about 15% of the
    rows), `fiber` the declared exposure as a 4-knot spline, so its companion quintiles are fit
    beside it, and every row above `fiber`'s 80th percentile has the event: the companion's
    quintile 5 separates the outcome in every copy (`fiber` is never blank), the spline does
    not. Every call of ``firth_profile`` is counted."""
    from turbotab.core import decisions as d
    from turbotab.core.stages.modeling import design_stage, fit_stage
    from turbotab.core.tests import modeling_fixtures as mf
    from turbotab.core.tests.acceptance import estimand_fixtures as ef

    frame = ef.cohort(400, seed=12)
    cut = float(np.quantile(frame["fiber"], 0.8))
    frame.loc[frame["fiber"] > cut + 1e-9, "dm"] = "yes"
    frame.loc[np.random.default_rng(4).random(len(frame)) < 0.15, "age"] = np.nan
    roles = {c: r for c, r in ef.ROLES.items() if c != "bmi"}
    answers = {c: a for c, a in ef.ANSWERS.items() if c != "bmi"}
    st = ef.state(target="dm", task="binary", event="yes", measure="odds_ratio", roles=roles,
                  answers=answers, missing={"strategy": "multiple_imputation", "m": 20},
                  exposure_forms={"fiber": d.ExposureFormSpec(form="spline", knots=4)})
    paths = mf.ingest_frame(frame, folder)
    split = mf.split_bundle(np.arange(len(frame)), holdout=0.0, seed=st.split.seed)
    info = mf.target_info(st.task, st.target)
    design = design_stage(mf.context(st, {"split": split, "target_info": info}, paths))
    calls = {"n": 0}
    real = I.firth_profile

    def counting(*a: Any, **k: Any) -> Any:
        calls["n"] += 1
        return real(*a, **k)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(I, "firth_profile", counting)
        fit = fit_stage(mf.context(st, {"design": design, "split": split, "target_info": info},
                                   paths))
    [model] = [mm for mm in fit.data["models"] if mm["family"] == "linear"]
    imputations = fit.objects["imputations"]
    return {"model": model, "tests": {t["test"]: t for t in model["exposure_tests"]},
            "fits": imputations["fits"]["linear"], "frames": imputations["frames"],
            "y": (frame["dm"] == "yes").to_numpy(float), "cut": cut, "profiles": calls["n"]}


@pytest.fixture(scope="module")
def companion(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return _companion_fit(tmp_path_factory.mktemp("companion"))


def _companion_designs(found: dict[str, Any]) -> list[tuple[pd.DataFrame, np.ndarray]]:
    """Each copy's model matrix with `fiber`'s spline terms dropped, and `fiber`'s quintile group
    (0 to 4) on cut points at its 20th, 40th, 60th and 80th percentiles (numpy's linear
    quantiles, `fiber` recorded on every row), each group holding the values above the cut point
    below it and up to its own (the boundaries the caption states)."""
    from turbotab.core.models.linear import model_matrix

    out = []
    for fitted, X_k in zip(found["fits"], found["frames"]):
        M = model_matrix(fitted, X_k).astype(float)
        rest = M[[c for c in M.columns if not str(c).startswith("fiber")]]
        x = X_k["fiber"].to_numpy(float)
        cuts = np.quantile(x, [0.2, 0.4, 0.6, 0.8])
        assert cuts[-1] == pytest.approx(found["cut"], abs=1e-12)
        out.append((rest, np.digitize(x, cuts, right=True)))
    return out


def test_under_imputation_the_companion_refits_compute_no_profile(companion):
    """The companion quintiles' refit is Firth's in every copy (quintile 5 separates), and is made
    for Rubin's rules as the copy's table is: no copy calls ``firth_profile`` (the base profiled
    each of the refit's nine coefficients in each of the 20 copies, 180 calls)."""
    assert len(companion["frames"]) == 20
    assert companion["profiles"] == 0
    assert {"companion_q2", "companion_q5", "companion_trend"} <= set(companion["tests"])


def test_a_companion_quintile_separated_in_the_copies_keeps_its_estimate_alone(companion):
    """Quintile 5 of `fiber` has the event on every row, in every copy: its pooled estimate is
    Rubin's mean of the copies' Firth estimates (recomputed by scipy on the companion's design),
    and its statistic, interval and p-value are not reported, with the reason and complete cases
    as the way forward. No copy's profile interval or p-value is quoted beside it."""
    m = len(companion["frames"])
    q, u = [], []
    for rest, group in _companion_designs(companion):
        X = np.column_stack([np.ones(len(rest)), rest.to_numpy(float),
                             *[(group == g).astype(float) for g in range(1, 5)]])
        beta, cov = _firth_by_scipy(X, companion["y"])
        q.append(beta[-1])
        u.append(cov[-1, -1])
    truth = _rubin(q, u)
    q5 = companion["tests"]["companion_q5"]
    assert q5["estimate"] == pytest.approx(truth["estimate"], abs=1e-6)
    assert (q5["ci_low"], q5["ci_high"], q5["p"], q5["statistic"], q5["df_den"]) == (
        None, None, None, None, None)
    assert "infinite" not in q5
    assert q5["caption"] == (
        f"Pooled over {m} imputations by Rubin's rules: quintile 5 of `fiber` against the lowest "
        f"{q5['estimate']:+.4g}. It separates the outcome in every one of the {m} imputed copies, "
        f"so its statistic, interval and p-value are not reported, only its estimate: {CARD} "
        f"(Heinze & Schemper 2002). To report them, analyze complete cases, where one table is "
        f"fit and none is pooled; where the complete cases separate the outcome too, that table "
        f"reports no standard error, interval or p-value either.")


def test_a_companion_quintile_no_copy_separates_keeps_rubins_wald_interval_on_t(companion):
    """Quintiles 2 to 4 hold both outcomes: each keeps Rubin's Wald interval on t and its p-value,
    from each copy's Firth estimate and standard error on the companion's design (recomputed by
    scipy and numpy), and the copy's words beside it quote no p-value or ratio interval, which
    no copy computes. The trend on the quintile medians separates in no copy, so its refit is
    the logistic model's: its pooled interval is Rubin's over statsmodels' fits."""
    import statsmodels.api as sm

    designs = _companion_designs(companion)
    for g in (2, 3, 4):
        q, u = [], []
        for rest, group in designs:
            X = np.column_stack([np.ones(len(rest)), rest.to_numpy(float),
                                 *[(group == h).astype(float) for h in range(1, 5)]])
            beta, cov = _firth_by_scipy(X, companion["y"])
            q.append(beta[g - 6])  # quintiles 2 to 5 are the last four columns
            u.append(cov[g - 6, g - 6])
        truth, test = _rubin(q, u), companion["tests"][f"companion_q{g}"]
        for key in ("estimate", "ci_low", "ci_high"):
            assert test[key] == pytest.approx(truth[key], rel=1e-5, abs=1e-6), (g, key)
        assert test["p"] == pytest.approx(truth["p"], rel=1e-4, abs=1e-10), g
        within = test["caption"].split("Within each imputation: ")[1]
        assert " p = " not in within and "ratio" not in within, within
    q, u = [], []
    for (rest, group), X_k in zip(designs, companion["frames"]):
        x = X_k["fiber"].to_numpy(float)
        medians = np.array([np.median(x[group == g]) for g in range(5)])
        X = np.column_stack([np.ones(len(rest)), rest.to_numpy(float), medians[group]])
        logit = sm.Logit(companion["y"], X).fit(disp=0, maxiter=200)
        q.append(float(logit.params[-1]))
        u.append(float(logit.bse[-1]) ** 2)
    truth, trend = _rubin(q, u), companion["tests"]["companion_trend"]
    for key in ("estimate", "ci_low", "ci_high"):
        assert trend[key] == pytest.approx(truth[key], rel=1e-5, abs=1e-8), key
    assert trend["p"] == pytest.approx(truth["p"], rel=1e-4, abs=1e-12)


def _cut_test(estimate: float, se: float, infinite: bool | None) -> dict[str, Any]:
    """One copy's cut-point test, as ``exposure_form.exposure_tests`` makes it from the copy's
    own row (its statistic the estimate over its standard error), marked as a copy's is."""
    test = {"column": "x", "form": "optimal", "test": "cut", "statistic": estimate / se,
            "df_num": 1, "df_den": None, "distribution": "z", "p": None, "estimate": estimate,
            "ci_low": None, "ci_high": None, "knots": None, "medians": None, "label": None,
            "boundaries": [1.0], "reference": "`x` ≤ 1", "what": "`x` above 1 against below it",
            "caption": "`x` above 1 against below it: a cut point the data chose."}
    if infinite is not None:
        test["infinite"] = infinite
    return test


def test_a_cut_point_infinite_in_one_copy_keeps_its_estimate_alone_and_otherwise_rubins():
    """A data-derived cut point's coefficient pooled over three copies: where one copy's table
    finds it infinite, only its pooled estimate (the copies' mean) is reported, and without the
    missing-values answer (the data's own copies) no way forward is named; where none does, its
    interval and p-value are Rubin's, written out here. The copies' mark never reaches the
    pooled test."""
    from turbotab.core.stages.modeling import _pool_form_tests

    copies = [(1.2, 0.5), (1.6, 0.6), (0.9, 0.4)]
    tables = [I.InferenceTable([], {"covariance": "model"}, []) for _ in copies]
    for marks, withheld in (([False, True, False], True), ([False, False, False], False)):
        tests = [[_cut_test(b, s, mark)] for (b, s), mark in zip(copies, marks)]
        [pooled] = _pool_form_tests(tests, tables, [None] * 3, 3)
        assert "infinite" not in pooled
        assert pooled["estimate"] == pytest.approx(np.mean([b for b, _ in copies]), rel=1e-12)
        if withheld:
            assert (pooled["ci_low"], pooled["ci_high"], pooled["p"], pooled["statistic"]) == (
                None, None, None, None)
            assert pooled["caption"] == (
                "Pooled over 3 imputations by Rubin's rules: `x` above 1 against below it +1.233. "
                "It separates the outcome in 1 of the 3 imputed copies, so its statistic, interval "
                f"and p-value are not reported, only its estimate: {CARD} (Heinze & Schemper "
                "2002).")
            continue
        truth = _rubin([b for b, _ in copies], [s ** 2 for _, s in copies])
        for key in ("ci_low", "ci_high", "p"):
            assert pooled[key] == pytest.approx(truth[key], rel=1e-10), key

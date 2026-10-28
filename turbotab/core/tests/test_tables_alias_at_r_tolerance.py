"""Nearly dependent columns: every coefficient table of a family whose linear predictor is Xβ
reads a column as aliased where, centered, its part outside the span of the intercept and the
columns before it is under R's tolerance, 10⁻⁷ of its length (``lm``'s ``qr(x, tol = 1e-07)``,
LINPACK's limited pivoting), and refuses by row each coefficient in that dependency; above the
tolerance it keeps the column, and the least-squares tables' HC3 and CR2 standard errors come from
the QR of the design, not from ``XᵀX`` (T1V-3, T1V2-2; the methods lead's ruling of 2026-10-27).

The columns ``x`` and ``x2 = x + ε·d`` span what ``x`` and ``d`` do for every ε ≠ 0, so `v`'s
estimate and its HC3 and CR2 standard errors are exact invariants, and the table on
``(1, x, x2, v)`` is the table on ``(1, x, d, v)`` reparametrized: ``β = T⁻¹β_d`` and
``V = T⁻¹ V_d T⁻ᵀ`` for the map ``T`` that takes ``(1, x, d, v)`` to ``(1, x, x2, v)`` (HC3 and
CR2 are equivariant under a nonsingular map of the columns, their leverages and residuals
unchanged; a Bell–McCaffrey df is a combination's own). On the base, at ε from 10⁻⁸ to 10⁻¹², no
dependency was found at working precision and the tables were served, with `v`'s HC3 standard
error off by up to +2.6 times (statsmodels' leverages are ``diag(X(XᵀX)⁻¹Xᵀ)``, whose rounding is
the condition number squared) and its CR2 one by −1.3·10⁻³ (the pseudo-inverse of ``XᵀX`` dropped
the near-null direction).

Truth: statsmodels' OLS with HC3 on the well-conditioned ``(1, x, d, v)``, and CR2 with its
Bell–McCaffrey df written out densely from the definition on it, mapped by ``T⁻¹``; below the
tolerance, `v` and the intercept are those of the fit with `x2` left out (R's ``lm``), the same
two references on ``(1, x, v)``; the other families' `v` rows are their own tables on ``(x, v)``
(the identity that a coefficient in no dependency does not depend on which aliased column is left
out), anchored at statsmodels for the logistic and Cox tables.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from turbotab.core.models.effects import ALIAS_TOL
from turbotab.core.models.inference import INDEPENDENT, Clusters, inference_table
from turbotab.core.models.ordinal import ordinal_table
from turbotab.core.models.repeated import gee_table, mixed_table
from turbotab.core.models.survey import SurveyDesign, survey_table
from turbotab.core.models.survival import cox_table, survey_cox_table, survival_outcome

N, G = 400, 80
ABOVE = (1e-4, 1e-6)
BELOW = (1e-8, 1e-9, 1e-10, 1e-11, 1e-12)


def _data():
    rng = np.random.default_rng(8)
    codes = np.repeat(np.arange(G), N // G).astype(np.int64)
    x, d, v = rng.normal(size=(3, N))
    eta = 0.5 * x + 0.3 * d - 0.4 * v
    y = eta + rng.normal(size=N) * (1 + 0.5 * np.abs(v)) + 0.5 * rng.normal(size=G)[codes]
    yb = (rng.random(N) < 1 / (1 + np.exp(-eta))).astype(int)
    yo = np.digitize(eta + rng.logistic(size=N), [-0.7, 0.7])
    ys = survival_outcome(rng.random(N) < 0.7, rng.exponential(np.exp(-eta)))
    return x, d, v, codes, y, yb, yo, ys


def _clusters(codes: np.ndarray) -> Clusters:
    return Clusters(column="pid", codes=codes, n_clusters=G, n_missing=0)


def _near(eps: float) -> pd.DataFrame:
    x, d, v, *_ = _data()
    return pd.DataFrame({"x": x, "x2": x + eps * d, "v": v})


def _dense_cr2(X: np.ndarray, e: np.ndarray, codes: np.ndarray, C: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray]:
    """CR2's covariance on the design ``X`` with residuals ``e``, and the Bell–McCaffrey df of each
    combination in the columns of ``C``, written out from their definition (Bell & McCaffrey 2002;
    Pustejovsky & Tipton 2018): ``A_g = (I − H_gg)^(−½)``, ``V = B Σ_g X_gᵀA_g e_g e_gᵀA_g X_g B``;
    ``p_g = A_g X_g B c`` on cluster g's rows, ``u_g = (I − H) p_g``, ``Ω_gh = u_gᵀu_h`` and
    ν = tr(Ω)² / tr(Ω²)."""
    B = np.linalg.inv(X.T @ X)
    H = X @ B @ X.T
    roots = {}
    meat = np.zeros((X.shape[1], X.shape[1]))
    for g in range(G):
        at = codes == g
        lam, Q = np.linalg.eigh(np.eye(at.sum()) - H[np.ix_(at, at)])
        roots[g] = Q @ np.diag(1.0 / np.sqrt(lam)) @ Q.T
        s = X[at].T @ roots[g] @ e[at]
        meat += np.outer(s, s)
    V = B @ meat @ B
    df = []
    for c in C.T:
        U = np.zeros((len(X), G))
        for g in range(G):
            at = codes == g
            p = np.zeros(len(X))
            p[at] = roots[g] @ X[at] @ B @ c
            U[:, g] = p - H @ p
        omega = U.T @ U
        df.append(float(np.trace(omega) ** 2 / np.sum(omega ** 2)))
    return V, np.array(df)


def _reference(design: np.ndarray, kind: str, T: np.ndarray | None = None
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each coefficient's estimate, standard error and df on ``design`` (with its intercept), by
    HC3 (statsmodels) or CR2 (dense), on the coefficients ``T⁻¹β`` where ``T`` is given."""
    _, _, _, codes, y, *_ = _data()
    fit = sm.OLS(y, design).fit()
    P = design.shape[1]
    Tinv = np.eye(P) if T is None else np.linalg.inv(T)
    if kind == "HC3":
        V = np.asarray(fit.get_robustcov_results(cov_type="HC3").cov_params())
        df = np.full(P, float(N - P))
    else:
        V, df = _dense_cr2(design, np.asarray(fit.resid), codes, Tinv.T)
    V = Tinv @ V @ Tinv.T
    return Tinv @ np.asarray(fit.params), np.sqrt(np.diag(V)), df


def _table(kind: str, matrix: pd.DataFrame, **kw):
    _, _, _, codes, y, *_ = _data()
    clusters = INDEPENDENT if kind == "HC3" else _clusters(codes)
    return inference_table("regression", matrix, y, None, clusters, **kw)


@pytest.mark.parametrize("kind", ["HC3", "CR2"])
@pytest.mark.parametrize("eps", ABOVE)
def test_above_the_tolerance_the_table_is_the_reparametrized_one(eps, kind):
    """`x2` is kept and every row is the table on ``(1, x, d, v)`` mapped by ``T⁻¹``: estimate,
    standard error and df, `x2`'s standard error ``1/ε`` times `d`'s. On the base, at 10⁻⁶,
    `v`'s HC3 standard error was off by −1.8·10⁻⁶ and its CR2 one by +4.2·10⁻⁷."""
    x, d, v, *_ = _data()
    table = _table(kind, _near(eps), study=["v"])
    assert table.info["refused"] is None and table.unidentified == []
    T = np.eye(4)
    T[1, 2], T[2, 2] = 1.0, eps  # (1, x, d, v) T = (1, x, x + εd, v)
    est, se, df = _reference(np.column_stack([np.ones(N), x, d, v]), kind, T)
    rows = table.rows
    assert [r["feature"] for r in rows] == ["(intercept)", "x", "x2", "v"]
    np.testing.assert_allclose([r["estimate"] for r in rows], est, rtol=1e-8)
    np.testing.assert_allclose([r["se"] for r in rows], se, rtol=1e-8)
    np.testing.assert_allclose([r["df"] for r in rows], df, rtol=1e-6)


def _near_words(named: list[str], left_out: str) -> str:
    listed = " and ".join(f"`{c}`" for c in named)
    return (f"On these {N:,} rows the model matrix is singular to within 1 part in "
            f"{round(1 / ALIAS_TOL):,} of a column's variation about its mean (the tolerance of "
            f"R's `lm`): {listed} are fixed combinations of the other columns and the intercept "
            f"to that precision, so their coefficients are not identified at it. "
            f"No estimate, interval or p-value is reported for {listed}; the other coefficients, "
            f"which these rows identify, are estimated with `{left_out}` left out. To report "
            f"every coefficient, leave one of the dependent columns out.")


@pytest.mark.parametrize("kind", ["HC3", "CR2"])
@pytest.mark.parametrize("eps", BELOW)
def test_below_the_tolerance_the_dependency_is_refused_by_row_and_the_rest_are_lm(eps, kind):
    """`x2`, centered, is within 10⁻⁷ of `x`'s span: `x` and `x2` are refused by row in words that
    name them and the tolerance, and the intercept and `v` are the fit's with `x2` left out (R's
    ``lm`` aliases `x2` there): estimate, standard error and df. On the base each table was served
    whole, `v`'s HC3 standard error off by up to +2.6 times and its CR2 one by −1.3·10⁻³."""
    x, d, v, *_ = _data()
    table = _table(kind, _near(eps), study=["v"])
    assert table.info["refused"] is None
    assert table.unidentified == ["x", "x2"]
    assert table.concerns[0] == _near_words(["x", "x2"], "x2")
    est, se, df = _reference(np.column_stack([np.ones(N), x, v]), kind)
    rows = {r["feature"]: r for r in table.rows}
    for j, name in ((0, "(intercept)"), (2, "v")):
        assert rows[name]["estimate"] == pytest.approx(est[j], rel=1e-9), name
        assert rows[name]["se"] == pytest.approx(se[j], rel=1e-9), name
        assert rows[name]["df"] == pytest.approx(df[j], rel=1e-6), name
    for name in ("x", "x2"):
        assert all(rows[name][k] is None for k in ("estimate", "se", "ci_low", "p")), name


@pytest.mark.parametrize("study", [["x2"], None])
def test_below_the_tolerance_a_study_factor_in_the_dependency_refuses_the_table_whole(study):
    """The whole table is refused, with no coefficient, where what is studied is in the near
    dependency or nothing studied is named, in words that name `x` and `x2` and the tolerance."""
    table = _table("HC3", _near(1e-9), study=study)
    assert table.rows == []
    assert table.info["refused"] == (
        f"On these {N:,} rows the model matrix is singular to within 1 part in "
        f"{round(1 / ALIAS_TOL):,} of a column's variation about its mean (the tolerance of R's "
        f"`lm`): `x` and `x2` are fixed combinations of the other columns and the intercept to "
        f"that precision, so their "
        f"coefficients are not identified at it. No estimate, interval or p-value is reported; "
        f"leave one of the dependent columns out.")


def _families():
    x, d, v, codes, y, yb, yo, ys = _data()
    cl = _clusters(codes)
    index = pd.RangeIndex(N)
    strata = np.repeat(np.arange(10), N // 10)
    design = SurveyDesign(row_ids=np.arange(N), weight=np.random.default_rng(3).uniform(1, 3, N),
                          stratum=strata.astype(np.int64),
                          psu=(strata * 4 + np.arange(N) % 4).astype(np.int64),
                          weight_column="w", strata_column="s", psu_column="p")
    S = {"study": ["v"]}
    return index, {
        "logistic": lambda m: inference_table("binary", m, yb, [0, 1], **S),
        "logistic (CR2)": lambda m: inference_table("binary", m, yb, [0, 1], cl, **S),
        "multinomial": lambda m: inference_table("multiclass", m, yo, [0, 1, 2], **S),
        "proportional odds": lambda m: ordinal_table(m, yo, [0, 1, 2], INDEPENDENT, **S),
        "survey least squares": lambda m: survey_table("regression", m, y, None, design, **S),
        "survey logistic": lambda m: survey_table("binary", m, yb, [0, 1], design, **S),
        "Cox": lambda m: cox_table(m, ys, INDEPENDENT, **S),
        "mixed model": lambda m: mixed_table(m, y, cl, **S),
        "GEE": lambda m: gee_table(m, y, cl, "regression", **S),
        "GEE (logit)": lambda m: gee_table(m, yb, cl, "binary", [0, 1], **S),
        "survey Cox": lambda m: survey_cox_table(m, ys, design, **S),
    }


@pytest.mark.parametrize("family", list(_families()[1]))
def test_every_family_aliases_below_the_tolerance_and_keeps_above_it(family):
    """At ε = 10⁻⁹ each family refuses `x` and `x2` by row and serves `v`'s row as its own table on
    ``(x, v)`` does, every field; at ε = 10⁻² (the silent side) it keeps `x2`, refusing nothing by
    row, and serves the table: not refused whole, which an empty dependency also satisfies, in
    the Cox, survey Cox, mixed, GEE and GEE logit families too, whose near-singular refusal stands
    above a centered condition number of 1,000 (the methods lead's ruling 5 of 2026-10-27,
    T1V3-6). On the base the families found no dependency at 10⁻⁹."""
    index, families = _families()
    build = families[family]
    x, d, v, *_ = _data()
    table = build(_near(1e-9).set_axis(index))
    reference = build(pd.DataFrame({"x": x, "v": v}, index=index))
    assert reference.info["refused"] is None, reference.info["refused"]
    assert table.info["refused"] is None, table.info["refused"]
    refused = [r["feature"] for r in table.rows if r["feature"].split(" [")[0] in ("x", "x2")]
    assert refused and sorted(table.unidentified) == sorted(refused)
    ref = {r["feature"]: r for r in reference.rows}
    for row in table.rows:
        if row["feature"].split(" [")[0] != "v":
            continue
        for key in ("estimate", "se", "ci_low", "ci_high", "df", "p"):
            want = ref[row["feature"]][key]
            assert (row[key] is None and want is None) or row[key] == pytest.approx(
                want, rel=1e-7), (row["feature"], key)
    kept = build(_near(1e-2).set_axis(index))
    assert kept.unidentified == [] and kept.dependency is None
    assert kept.info["refused"] is None, kept.info["refused"]
    assert any(r["feature"].split(" [")[0] == "x2" and r["se"] is not None for r in kept.rows)


def test_the_logistic_and_cox_rows_below_the_tolerance_are_statsmodels_without_x2():
    """The anchor: below the tolerance `v` in the logistic table is statsmodels' Logit on
    ``(1, x, v)`` and in the Cox table PHReg's (Efron ties) on ``(x, v)``."""
    x, d, v, codes, y, yb, yo, ys = _data()
    matrix = _near(1e-10)
    logit = sm.Logit(yb, np.column_stack([np.ones(N), x, v])).fit(disp=0)
    row = {r["feature"]: r for r in inference_table("binary", matrix, yb, [0, 1],
                                                    study=["v"]).rows}["v"]
    assert row["estimate"] == pytest.approx(logit.params[2], rel=1e-7)
    assert row["se"] == pytest.approx(logit.bse[2], rel=1e-6)
    phreg = sm.PHReg(ys["time"], np.column_stack([x, v]), status=ys["event"].astype(int),
                     ties="efron").fit()
    row = {r["feature"]: r for r in cox_table(matrix, ys, INDEPENDENT, study=["v"]).rows}["v"]
    assert row["estimate"] == pytest.approx(phreg.params[1], rel=1e-7)
    assert row["se"] == pytest.approx(phreg.bse[1], rel=1e-6)


# The families whose covariance is the inverse of an information, now taken on the QR of the
# working design (the logistic, multinomial and survey tables) or evaluated on the QR's Q (the
# proportional-odds tables) and carried back by R⁻¹.
INFORMATION = ("logistic", "multinomial", "proportional odds", "proportional odds (CR1)",
               "survey least squares", "survey logistic", "survey multinomial",
               "survey multinomial (ordered levels)", "survey proportional odds")


def _information_tables():
    x, d, v, codes, y, yb, yo, ys = _data()
    index, families = _families()
    cl = _clusters(codes)
    design = SurveyDesign(row_ids=np.arange(N), weight=np.random.default_rng(3).uniform(1, 3, N),
                          stratum=np.repeat(np.arange(10), N // 10).astype(np.int64),
                          psu=(np.repeat(np.arange(10), N // 10) * 4 + np.arange(N) % 4
                               ).astype(np.int64),
                          weight_column="w", strata_column="s", psu_column="p")
    from turbotab.core.models.ordinal import survey_ordinal_table

    S = {"study": ["v"]}
    extra = {
        "proportional odds (CR1)": lambda m: ordinal_table(m, yo, [0, 1, 2], cl, **S),
        "survey multinomial": lambda m: survey_table("multiclass", m, yo, [0, 1, 2], design, **S),
        # the linear family's table for an ordered outcome, which ignores the levels' order
        "survey multinomial (ordered levels)": lambda m: survey_table("ordinal", m, yo, [0, 1, 2],
                                                                      design, **S),
        "survey proportional odds": lambda m: survey_ordinal_table(m, yo, [0, 1, 2], design, **S),
    }
    return index, {k: extra.get(k) or families[k] for k in INFORMATION}


@pytest.mark.parametrize("eps", [1e-4, 1e-6, 3e-7])
@pytest.mark.parametrize("family", INFORMATION)
def test_above_the_tolerance_an_information_table_is_right_or_refused(family, eps):
    """Kept `x2` and `v` are the table on ``(x, d, v)`` reparametrized: `x2`'s estimate and
    standard error are `d`'s divided by ε and `v`'s are `v`'s; or the table is refused whole (a
    logistic or multinomial fit that does not converge). The standard errors to 10⁻⁹ at
    ε = 10⁻⁴ (a condition number near 2·10⁴, whose square was the rounding of the base's
    inverse: the multinomial's was off by 2·10⁻⁷) and to 10⁻⁶ below it, where the base's were off
    by 3·10⁻⁴ to 4·10⁻³ at ε = 10⁻⁶ and by up to 2·10⁻² at 3·10⁻⁷; the estimates to 10⁻⁵, the
    proportional-odds Newton fit stopping within 10⁻⁶ of `x2`'s at 3·10⁻⁷."""
    index, families = _information_tables()
    build = families[family]
    x, d, v, *_ = _data()
    table = build(_near(eps).set_axis(index))
    if table.info["refused"] is not None:
        assert all(r.get("ci_low") is None and r.get("p") is None for r in table.rows)
        return
    assert table.unidentified == []
    reference = {r["feature"]: r for r in build(pd.DataFrame({"x": x, "d": d, "v": v},
                                                             index=index)).rows}
    for row in table.rows:
        base, _, suffix = row["feature"].partition(" [")
        suffix = f" [{suffix}" if suffix else ""
        if base == "x2":
            other, factor = reference[f"d{suffix}"], eps
        elif base == "v":
            other, factor = reference[f"v{suffix}"], 1.0
        else:
            continue
        assert row["estimate"] * factor == pytest.approx(other["estimate"], rel=1e-5), (
            row["feature"])
        assert row["se"] * factor == pytest.approx(other["se"], rel=1e-9 if eps >= 1e-4 else 1e-6
                                                   ), row["feature"]

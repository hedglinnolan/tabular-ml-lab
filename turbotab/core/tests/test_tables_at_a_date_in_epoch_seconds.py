"""A date in epoch seconds spanning an hour or a day, beside a standard-normal `x`: every coefficient
table of a family whose linear predictor is Xβ is right or refused (T1V-2, 2026-10-27).

The first round fit each table on its columns scaled to a root mean square of one. Scaling does not
center: such a date, ``1.7·10⁹ + 86400·s``, scaled by its root mean square stays within 10⁻⁵ of one,
nearly a copy of the intercept, so statsmodels' logistic and multinomial fits stopped short of
their maximum (statsmodels reported ``converged=False`` after 200 steps, which the table never
read) and served `x`'s log-odds as 59.05 with a standard error of 23.98 and p = 0.014, where R's
``glm`` on the same columns gives 1.0566 (0.1118); at an hour the logistic, CR2, multinomial,
survey and proportional-odds tables were off by 10⁻³ to 10⁻². Each table is now fit on its columns
centered at their means and scaled to an SD of one (``inference.by_identified``), its slopes scaled
back, and its intercept (or cut-points) mapped back to the columns' own origin as the linear
combination it is, with that combination's standard error, degrees of freedom (CR2's
Bell–McCaffrey df are the combination's own) and interval; a logistic or multinomial fit that does
not converge is refused, with no estimate.

Truth from construction: the date is the affine map ``t = T0 + span·s`` of ``s``, so the table on
``(x, t)`` is the table on ``(x, s)`` with `x`'s row unchanged, the date's divided by ``span``, and
the intercept moved by ``−b_s·T0/span`` (each cut-point by ``+b_s·T0/span``); the table on
``(x, s)`` is anchored at statsmodels (OLS with HC3, Logit, MNLogit, OrderedModel), whose
covariance gives the moved intercept's standard error, and its CR2 df are the Bell–McCaffrey df of
the moved combination, written out densely from their definition below. The GEE, Cox and mixed
tables refused such a date until their near-singular concern read the columns centered, as each is
fit (T1V2-1, 2026-10-27): they are now right there too.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel

from turbotab.core.models import inference as I
from turbotab.core.models.inference import INDEPENDENT, Clusters, inference_table
from turbotab.core.models.ordinal import ordinal_table, survey_ordinal_table
from turbotab.core.models.repeated import gee_table, mixed_table
from turbotab.core.models.survey import SurveyDesign, survey_table
from turbotab.core.models.survival import cox_table, survey_cox_table, survival_outcome

N, G = 600, 150
T0 = 1.7e9
SPANS = {"an hour": 3600.0, "a day": 86400.0}


def _data():
    rng = np.random.default_rng(4)
    x = rng.normal(size=N)
    s = rng.uniform(0, 1, N)
    codes = np.repeat(np.arange(G), N // G).astype(np.int64)
    yr = x + 0.5 * s + rng.normal(size=N)
    yb = (x + s + rng.logistic(size=N) > 0.5).astype(int)
    yo = np.digitize(x + s + rng.logistic(size=N), [0, 1.0])
    ys = survival_outcome(np.ones(N, bool), rng.exponential(np.exp(-(x + s))))
    strata = np.repeat(np.arange(10), N // 10)
    design = SurveyDesign(row_ids=np.arange(N), weight=rng.uniform(1, 3, N),
                          stratum=strata.astype(np.int64),
                          psu=(strata * 3 + np.arange(N) % 3).astype(np.int64),
                          weight_column="w", strata_column="s", psu_column="p")
    return x, s, codes, yr, yb, yo, ys, design


def _clusters(codes: np.ndarray) -> Clusters:
    return Clusters(column="c", codes=codes, n_clusters=G, n_missing=0)


def _tables():
    _, _, codes, yr, yb, yo, ys, design = _data()
    cl = _clusters(codes)
    return {
        "least squares (HC3)": lambda m: inference_table("regression", m, yr, None),
        "least squares (CR2)": lambda m: inference_table("regression", m, yr, None, cl),
        "logistic": lambda m: inference_table("binary", m, yb, [0, 1]),
        "logistic (CR2)": lambda m: inference_table("binary", m, yb, [0, 1], cl),
        "multinomial": lambda m: inference_table("multiclass", m, yo, [0, 1, 2]),
        "multinomial (CR2)": lambda m: inference_table("multiclass", m, yo, [0, 1, 2], cl),
        "proportional odds": lambda m: ordinal_table(m, yo, [0, 1, 2], INDEPENDENT),
        "proportional odds (CR1)": lambda m: ordinal_table(m, yo, [0, 1, 2], cl),
        "survey least squares": lambda m: survey_table("regression", m, yr, None, design),
        "survey logistic": lambda m: survey_table("binary", m, yb, [0, 1], design),
        "survey multinomial": lambda m: survey_table("multiclass", m, yo, [0, 1, 2], design),
        "survey proportional odds": lambda m: survey_ordinal_table(m, yo, [0, 1, 2], design),
        "Cox": lambda m: cox_table(m, ys, INDEPENDENT),
        "Cox (clustered)": lambda m: cox_table(m, ys, cl),
        "survey Cox": lambda m: survey_cox_table(m, ys, design),
        "mixed model": lambda m: mixed_table(m, yr, cl),
        "GEE": lambda m: gee_table(m, yr, cl, "regression"),
        "GEE (logit)": lambda m: gee_table(m, yb, cl, "binary", [0, 1]),
    }


def _frames(span: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    x, s, *_ = _data()
    index = pd.RangeIndex(N)
    return (pd.DataFrame({"x": x, "t": T0 + span * s}, index=index),
            pd.DataFrame({"x": x, "t": s}, index=index))


def _base(feature: str) -> str:
    return feature.split(" [")[0]


def _close(a, b, rel: float) -> bool:
    if a is None or b is None:
        return a is b
    return a == pytest.approx(b, rel=rel, abs=1e-12)


@pytest.mark.parametrize("span", list(SPANS))
@pytest.mark.parametrize("family", list(_tables()))
def test_each_table_is_the_affine_map_of_the_date_in_its_own_units_or_refused(family, span):
    """`x`'s rows are the table's on ``(x, s)``, every field; the date's estimate, standard error
    and interval are its rows' on ``s`` divided by the span, its df and p-value the same; the
    intercept's (or each cut-point's) estimate is moved by the date's coefficient times its origin.
    The GEE, Cox and mixed tables were refused here as nearly singular before their concern read
    the columns centered (T1V2-1)."""
    span_s = SPANS[span]
    build = _tables()[family]
    epoch, plain = _frames(span_s)
    table, reference = build(epoch), build(plain)
    assert reference.info["refused"] is None
    assert table.info["refused"] is None, table.info["refused"]
    ref = {r["feature"]: r for r in reference.rows}
    slope = {}  # the date's coefficient on `s`, per class
    for feature, row in ref.items():
        if _base(feature) == "t":
            slope[feature[len("t"):]] = row["estimate"]
    for row in table.rows:
        feature, other = row["feature"], ref[row["feature"]]
        if _base(feature) == "x":
            for key in ("estimate", "se", "ci_low", "ci_high", "df", "p"):
                assert _close(row[key], other[key], 1e-7), (feature, key)
        elif _base(feature) == "t":
            for key in ("estimate", "se", "ci_low", "ci_high"):
                assert row[key] * span_s == pytest.approx(other[key], rel=1e-7), (feature, key)
            for key in ("df", "p"):
                assert _close(row[key], other[key], 1e-7), (feature, key)
        elif feature.startswith("(intercept)"):
            moved = other["estimate"] - slope[feature[len("(intercept)"):]] * T0 / span_s
            assert row["estimate"] == pytest.approx(moved, rel=1e-8), feature
        else:  # a cut-point, compared with −Xβ
            moved = other["estimate"] + slope[""] * T0 / span_s
            assert row["estimate"] == pytest.approx(moved, rel=1e-8), feature


def _moved(span: float) -> np.ndarray:
    """The intercept on the date's own origin as a combination of ``(1, x, s)``'s coefficients."""
    return np.array([1.0, 0.0, -T0 / span])


@pytest.mark.parametrize("span", list(SPANS))
def test_the_moved_intercept_is_statsmodels_on_the_plain_date(span):
    """The intercept's estimate and standard error on the epoch date are statsmodels' combination
    ``cᵀβ`` and ``√(cᵀVc)`` on ``(1, x, s)`` (OLS with HC3, Logit, MNLogit per class), and its
    interval and p-value t(n − 3) or normal ones on them; the proportional-odds model's first
    cut-point is OrderedModel's first threshold moved by ``+b_s·T0/span``."""
    from scipy import stats

    span_s = SPANS[span]
    x, s, _, yr, yb, yo, *_ = _data()
    epoch, plain = _frames(span_s)
    exog = sm.add_constant(plain)
    c = _moved(span_s)
    for task, y, fit in (("regression", yr, sm.OLS(yr, exog).fit(cov_type="HC3")),
                         ("binary", yb, sm.Logit(yb, exog).fit(disp=0))):
        table = inference_table(task, epoch, y, None if task == "regression" else [0, 1])
        row = table.rows[0]
        b = float(c @ fit.params)
        se = float(np.sqrt(c @ np.asarray(fit.cov_params()) @ c))
        q = (stats.t.ppf(0.975, N - 3) if task == "regression" else stats.norm.ppf(0.975))
        assert row["feature"] == "(intercept)"
        assert row["estimate"] == pytest.approx(b, rel=1e-8)
        assert row["se"] == pytest.approx(se, rel=1e-6)
        assert row["ci_low"] == pytest.approx(b - q * se, rel=1e-6)
    mnl = sm.MNLogit(yo, exog).fit(disp=0)
    C = np.asarray(mnl.cov_params())
    table = {r["feature"]: r for r in inference_table("multiclass", epoch, yo, [0, 1, 2]).rows}
    for k in (1, 2):
        ck = np.zeros(6)
        ck[3 * (k - 1):3 * k] = c
        row = table[f"(intercept) [{k}]"]
        assert row["estimate"] == pytest.approx(float(c @ np.asarray(mnl.params)[:, k - 1]),
                                                rel=1e-7)
        assert row["se"] == pytest.approx(float(np.sqrt(ck @ C @ ck)), rel=1e-5)
    po = OrderedModel(yo, plain, distr="logit").fit(method="newton", disp=0, maxiter=200)
    cut = next(r for r in ordinal_table(epoch, yo, [0, 1, 2], INDEPENDENT).rows
               if r["feature"].startswith("(cut-point"))
    d = np.array([T0 / span_s if n == "t" else 0.0 for n in po.params.index[:2]] + [1.0, 0.0])
    assert cut["estimate"] == pytest.approx(float(d @ po.params), rel=1e-6)
    assert cut["se"] == pytest.approx(float(np.sqrt(d @ np.asarray(po.cov_params()) @ d)),
                                      rel=1e-4)


def _bell_mccaffrey(X: np.ndarray, codes: np.ndarray, c: np.ndarray) -> float:
    """The Bell–McCaffrey df of ``cᵀβ`` on the working design ``X``, written out densely from the
    definition (Bell & McCaffrey 2002; Pustejovsky & Tipton 2018): ``A_g = (I − H_gg)^(−½)``,
    ``p_g = A_g X_g (XᵀX)⁻¹ c`` placed on cluster g's rows, ``u_g = (I − H) p_g``, ``Ω_gh =
    u_gᵀu_h`` and ν = tr(Ω)² / tr(Ω²)."""
    B = np.linalg.inv(X.T @ X)
    H = X @ B @ X.T
    U = np.zeros((len(X), G))
    for g in range(G):
        at = codes == g
        lam, Q = np.linalg.eigh(np.eye(at.sum()) - H[np.ix_(at, at)])
        p = np.zeros(len(X))
        p[at] = Q @ np.diag(1.0 / np.sqrt(lam)) @ Q.T @ X[at] @ B @ c
        U[:, g] = p - H @ p
    omega = U.T @ U
    return float(np.trace(omega) ** 2 / np.sum(omega ** 2))


@pytest.mark.parametrize("span", list(SPANS))
def test_the_moved_intercepts_cr2_df_are_its_own_bell_mccaffrey_df(span):
    """Clustered least squares and logistic: the intercept on the epoch date has the
    Bell–McCaffrey df of its own combination on ``(1, x, s)``, not the centered intercept's,
    each written out densely (the logistic one on statsmodels' working design ``W^½X``); `x`'s
    df are its own the same way."""
    span_s = SPANS[span]
    x, s, codes, yr, yb, *_ = _data()
    epoch, plain = _frames(span_s)
    X = sm.add_constant(plain).to_numpy(dtype=float)
    mu = sm.Logit(yb, X).fit(disp=0).predict(X)
    working = {"regression": X, "binary": X * np.sqrt(mu * (1 - mu))[:, None]}
    for task, y in (("regression", yr), ("binary", yb)):
        table = inference_table(task, epoch, y, None if task == "regression" else [0, 1],
                                _clusters(codes))
        rows = {r["feature"]: r for r in table.rows}
        assert rows["(intercept)"]["df"] == pytest.approx(
            _bell_mccaffrey(working[task], codes, _moved(span_s)), rel=1e-6), task
        assert rows["x"]["df"] == pytest.approx(
            _bell_mccaffrey(working[task], codes, np.array([0.0, 1.0, 0.0])), rel=1e-6), task


def test_columns_at_their_own_origin_keep_the_intercept_unmoved():
    """The silent side of the map: columns already centered (each mean exactly zero) leave the
    intercept's row as the fit gives it, statsmodels' at ordinary scales."""
    x, s, _, yr, *_ = _data()
    plain = pd.DataFrame({"x": x - x.mean(), "t": s - s.mean()})
    fit = sm.OLS(yr, sm.add_constant(plain)).fit(cov_type="HC3")
    row = inference_table("regression", plain, yr, None).rows[0]
    assert row["estimate"] == pytest.approx(float(fit.params["const"]), rel=1e-10)
    assert row["se"] == pytest.approx(float(fit.bse["const"]), rel=1e-8)


@pytest.mark.parametrize("task", ["binary", "multiclass"])
def test_a_fit_that_stops_before_converging_is_refused_with_no_estimate(monkeypatch, task):
    """The guard both ways: allowed a single Newton step, statsmodels' logistic and multinomial
    fits stop short of the maximum and say so, and the table reports no estimate, interval or
    p-value, in words that count the steps; at the default 200 the same rows converge and are
    served. On the base a fit that stopped short was served, its numbers where the steps
    stopped."""
    x, s, _, _, yb, yo, *_ = _data()
    plain = pd.DataFrame({"x": x, "t": s})
    y, classes = (yb, [0, 1]) if task == "binary" else (yo, [0, 1, 2])
    served = inference_table(task, plain, y, classes)
    assert served.info["refused"] is None and all(r["ci_low"] is not None for r in served.rows)
    monkeypatch.setattr(I, "MAX_NEWTON", 1)
    table = inference_table(task, plain, y, classes)
    model = "logistic" if task == "binary" else "multinomial"
    assert table.info["refused"] == (
        f"The {model} model's maximum-likelihood fit did not converge in 1 Newton step, so no "
        f"estimate, interval or p-value is reported.")
    assert all(r["estimate"] is None and r["ci_low"] is None for r in table.rows)

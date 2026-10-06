"""WP12a · Exposure form: restricted cubic splines and quintiles (AUDIT_REPORT §5 WP12, test 1).

Closes ME-17 ("Exposure–response is straight-line only"). §5 WP12 acceptance test 1: "Restricted
cubic splines with the pack's knot percentiles match R ``rms::rcs`` to 10⁻⁶, with a nonlinearity
test; quintile estimates with a trend test on quintile medians."

R is not installed, so ``rms::rcs`` is matched through its documentation and source, by paths that
share nothing with the engine (``turbotab/core/methods/exposure_form.py``):

* **Knots.** ``rms::rcs`` places knots by calling ``Hmisc::rcspline.eval(x, nk = nknots, inclx =
  TRUE, pc = pc, fractied = fractied)`` (rms/R/rms.trans.s; ``fractied`` defaults to 0.05). The
  ``rcspline.eval`` documentation: "For 3 knots, the outer quantiles used are 0.10 and 0.90. For
  4-6 knots, the outer quantiles used are 0.05 and 0.95 … The knots are equally spaced between
  these on the quantile scale. For fewer than 100 non-missing values of x, the outer knots are the
  5th smallest and largest x", and ``fractied``: "If the fraction of observations tied at the
  lowest and/or highest values of x is greater than or equal to fractied, the algorithm attempts
  to use a different algorithm for knot finding based on quantiles of x after excluding the one or
  two values with excessive ties." :func:`_rcspline_knots` below transliterates that source
  (Hmisc/R/rcspline.eval.s, read 2026-10-02) with pandas' type-7 quantile (R's default); for
  ``x = 1…100`` and ``1…50`` the knots are also written out from the rule by hand.
* **Basis.** The ``norm`` argument: "2 to normalize by the square of the spacing between the first
  and last knots (the default)", applied to the Devlin & Weeks (1986) truncated-power terms
  (Harrell 2015, *Regression Modeling Strategies*, 2nd ed., eq. 2.25):
  ``X_j = (x − t_j)₊³ − (x − t_{k−1})₊³ (t_k − t_j)/(t_k − t_{k−1}) + (x − t_k)₊³ (t_{k−1} − t_j)/(t_k − t_{k−1})``.
  :func:`_harrell_closed_form` writes that and divides by ``(t_k − t_1)²``. A second library checks
  the space the basis spans: patsy's natural cubic regression spline ``cr`` with the same knots
  gives the same least-squares fit.
* **Tests.** statsmodels' OLS with HC3 covariance and ``f_test`` (the regression table's
  covariance), and statsmodels' ``Logit`` with ``wald_test`` (a binary outcome's information-based
  covariance), on the same columns.
* **Quintiles.** ``pandas.qcut`` for the groups, ``groupby().median()`` for the trend score, and
  statsmodels for the estimates: NUTRITION_PACK §08, "p for trend using the median of each
  quintile as a continuous score, not the quintile number".
"""
from __future__ import annotations

import warnings
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import pytest
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from turbotab.core.decisions import ExposureFormSpec
from turbotab.core.methods import exposure_form as ef
from turbotab.core.models import get_family
from turbotab.core.models.artifacts import FitArtifact
from turbotab.core.models.inference import INDEPENDENT
from turbotab.core.stages.modeling import design_stage, fit_stage
from turbotab.core.tests import modeling_fixtures as mf

REPO = Path(__file__).resolve().parents[4]


# ── the independent references ───────────────────────────────────────────────


def _rcspline_knots(x: np.ndarray, nk: int, fractied: float = 0.05) -> np.ndarray:
    """``Hmisc::rcspline.eval``'s default knot placement, transliterated from its R source.

    ``quantile`` is R's type 7, which is pandas' ``interpolation="linear"``; ``table(xx) / n``
    is ``value_counts(normalize=True)`` sorted by value.
    """
    xx = pd.Series(np.asarray(x, dtype=float)).dropna()
    n = len(xx)
    xu = np.sort(xx.unique())
    nxu = len(xu)
    if nxu - 2 <= nk:
        knots = list(xu[1:-1])
    else:
        outer = 0.05 if nk > 3 else 0.1
        if nk > 6:
            outer = 0.025
        nke = nk
        firstknot: list[float] = []
        lastknot: list[float] = []
        override_first = override_last = False
        f = xx.value_counts(normalize=True).sort_index()
        inner = f.iloc[1:-1]
        if (inner.max() if len(inner) else -np.inf) < fractied:
            if f.iloc[0] >= fractied:
                firstknot = [float(xx[xx > xx.min()].min())]
                xx = xx[xx > firstknot[0]]
                nke -= 1
                override_first = True
            if f.iloc[-1] >= fractied:
                lastknot = [float(xx[xx < xx.max()].max())]
                xx = xx[xx < lastknot[0]]
                nke -= 1
                override_last = True
        if nke == 1:
            knots = [float(xx.median())]
        else:
            if nxu <= nke:
                knots = list(xu)
            else:
                p = (np.linspace(0.5, 1 - outer, nke) if nke == 2
                     else np.linspace(outer, 1 - outer, nke))
                knots = list(xx.quantile(p, interpolation="linear").to_numpy())
                assert len(set(knots)) >= min(nke, 3), "these fixtures avoid the alternate algorithm"
            if len(xx) < 100:
                ordered = np.sort(xx.to_numpy())
                if not override_first:
                    knots[0] = ordered[4]  # R: xx[5]
                if not override_last:
                    knots[nke - 1] = ordered[len(ordered) - 5]  # R: xx[length(xx) - 4]
        knots = firstknot + knots + lastknot
    return np.unique(np.asarray(knots, dtype=float))


def _harrell_closed_form(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Harrell (2015) eq. 2.25, the Devlin–Weeks terms, divided by ``(t_k − t_1)²`` (norm = 2)."""
    k = len(t)

    def plus3(u):
        return np.where(u > 0, u, 0.0) ** 3

    cols = []
    for j in range(k - 2):
        term = (plus3(x - t[j])
                - plus3(x - t[k - 2]) * (t[k - 1] - t[j]) / (t[k - 1] - t[k - 2])
                + plus3(x - t[k - 1]) * (t[k - 2] - t[j]) / (t[k - 1] - t[k - 2]))
        cols.append(term / (t[k - 1] - t[0]) ** 2)
    return np.column_stack(cols)


# ── fixtures ─────────────────────────────────────────────────────────────────


def _intake(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.gamma(4.0, 5.0, n)


def _zero_inflated(n: int, rng: np.random.Generator, zeros: float = 0.2) -> np.ndarray:
    """Alcohol-like: a share of exact zeros (non-drinkers), the rest continuous."""
    x = rng.gamma(1.5, 8.0, n)
    x[rng.random(n) < zeros] = 0.0
    return x


def _curved(n: int, seed: int) -> pd.DataFrame:
    """Glucose-like outcome on age and a fiber intake whose effect flattens (log-shaped)."""
    rng = np.random.default_rng(seed)
    age = rng.uniform(20, 80, n)
    fiber = _intake(n, rng)
    y = 5 + 0.03 * age + 2.0 * np.log(fiber) + rng.normal(0, 1.0 + 0.01 * fiber, n)
    return pd.DataFrame({"age": age, "fiber_g": fiber, "y": y})


def _run(frame: pd.DataFrame, folder: Path, form: ExposureFormSpec, *, task: str = "regression",
         target: str = "y", holdout: float = 0.0) -> tuple[dict, pd.Index]:
    """Ingest, design and fit under inference with ``fiber_g`` in ``form``; the linear model's
    entry in the validated fit artifact, and the training rows' positions."""
    paths = mf.ingest_frame(frame, folder)
    roles = {"age": "covariate", "fiber_g": "exposure"}
    st = mf.state(roles=roles, target=target, models=["linear"], purpose="inference",
                  exposure_forms={"fiber_g": form})
    split = mf.split_bundle(np.arange(len(frame)), holdout=holdout)
    ti = mf.target_info(task, target)
    design = design_stage(mf.context(st, {"split": split, "target_info": ti}, paths))
    fit = fit_stage(mf.context(st, {"design": design, "split": split, "target_info": ti}, paths))
    artifact = FitArtifact.model_validate(fit.data)
    assignment = split.frames["assignment"]
    train = pd.Index(assignment.loc[assignment["partition"] == "train", "row_id"].to_numpy())
    return artifact.models[0].model_dump(), train


def _spline_frame(frame: pd.DataFrame, knots: np.ndarray) -> pd.DataFrame:
    X = frame[["age", "fiber_g"]].copy()
    basis = _harrell_closed_form(X["fiber_g"].to_numpy(), knots)
    for j in range(basis.shape[1]):
        X[f"s{j + 1}"] = basis[:, j]
    return X


# ── 1 · knots: rms::rcs's placement ──────────────────────────────────────────


@pytest.mark.parametrize("nk, expected", [
    # type-7 quantile of 1…100 at p is 1 + 99p; with 100 values no outer knot moves
    (3, [10.9, 50.5, 90.1]),
    (4, [5.95, 35.65, 65.35, 95.05]),
    (5, [5.95, 28.225, 50.5, 72.775, 95.05]),
])
def test_1_knots_for_1_to_100_are_the_documented_quantiles(nk, expected):
    knots, notes = ef.rcs_knots(np.arange(1, 101), nk)
    np.testing.assert_allclose(knots, expected, atol=1e-9)
    assert notes == []


def test_1_below_100_values_the_outer_knots_are_the_5th_smallest_and_largest():
    # 1…50, 4 knots: inner knots at 1 + 49·0.35 = 18.15 and 1 + 49·0.65 = 32.85; outer 5 and 46
    knots, notes = ef.rcs_knots(np.arange(1, 51), 4)
    np.testing.assert_allclose(knots, [5.0, 18.15, 32.85, 46.0], atol=1e-9)
    assert any("5th smallest and 5th largest" in n for n in notes)


@pytest.mark.parametrize("nk", [3, 4, 5])
@pytest.mark.parametrize("case", ["gamma_n500", "gamma_n60", "zeros_n1000", "zeros_n80",
                                  "rounded_n400"])
def test_1_knots_match_rcspline_eval_to_1e_6(case, nk):
    rng = np.random.default_rng(zlib.crc32(case.encode()))
    x = {
        "gamma_n500": lambda: _intake(500, rng),
        "gamma_n60": lambda: _intake(60, rng),
        "zeros_n1000": lambda: _zero_inflated(1000, rng),
        "zeros_n80": lambda: _zero_inflated(80, rng, zeros=0.3),
        "rounded_n400": lambda: np.round(_intake(400, rng), 1),
    }[case]()
    knots, _ = ef.rcs_knots(x, nk)
    reference = _rcspline_knots(x, nk)
    assert len(knots) == len(reference)
    np.testing.assert_allclose(knots, reference, rtol=0, atol=1e-6)


def test_1_a_zero_heavy_intake_gets_rcspline_evals_tied_value_knot():
    """20% exact zeros: ``fractied`` puts the first knot at the smallest positive value, and the
    note says why."""
    x = _zero_inflated(1000, np.random.default_rng(7))
    knots, notes = ef.rcs_knots(x, 4)
    assert knots[0] == pytest.approx(x[x > 0].min())
    assert any("lowest one" in n for n in notes)


# ── 1 · basis: Harrell's closed form, and the space it spans ─────────────────


@pytest.mark.parametrize("nk", [3, 4, 5])
def test_1_basis_matches_harrells_closed_form_to_1e_6(nk):
    rng = np.random.default_rng(nk)
    x = _intake(2_000, rng)
    knots, _ = ef.rcs_knots(x, nk)
    grid = np.concatenate([x, np.linspace(x.min() - 10, x.max() + 10, 101)])
    ours = ef.rcs_basis(grid, knots)
    reference = _harrell_closed_form(grid, knots)
    assert ours.shape == (len(grid), nk - 2)
    np.testing.assert_allclose(ours, reference, rtol=1e-9, atol=1e-6)


@pytest.mark.parametrize("nk", [3, 4, 5])
def test_1_the_spline_spans_patsys_natural_cubic_spline_and_is_linear_in_the_tails(nk):
    rng = np.random.default_rng(10 + nk)
    x = _intake(1_000, rng)
    y = np.sin(x / 8) + rng.normal(0, 0.2, len(x))
    knots, _ = ef.rcs_knots(x, nk)
    inside = (x >= knots[0]) & (x <= knots[-1])
    ours = np.column_stack([np.ones(inside.sum()), x[inside], ef.rcs_basis(x[inside], knots)])
    natural = np.asarray(patsy.dmatrix(
        f"cr(x, knots={list(knots[1:-1])}, lower_bound={knots[0]}, upper_bound={knots[-1]}) - 1",
        {"x": x[inside]}))
    assert natural.shape[1] == ours.shape[1] == nk
    fit_ours = ours @ np.linalg.lstsq(ours, y[inside], rcond=None)[0]
    fit_natural = natural @ np.linalg.lstsq(natural, y[inside], rcond=None)[0]
    np.testing.assert_allclose(fit_ours, fit_natural, atol=1e-6)
    # Beyond the outer knots every column is a straight line: zero second differences.
    for tail in (np.linspace(knots[-1], knots[-1] + 50, 41), np.linspace(knots[0] - 50, knots[0], 41)):
        second = np.diff(ef.rcs_basis(tail, knots), n=2, axis=0)
        assert np.abs(second).max() < 1e-9


def test_1_columns_are_named_as_rms_prints_them():
    assert ef.spline_names("fiber_g", 4) == ["fiber_g", "fiber_g'", "fiber_g''"]


def test_1_knots_are_learned_on_the_rows_each_fit_sees():
    """Cross-validation refits the form on each training fold: knots come from the fitting rows."""
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({"fiber_g": _intake(600, rng), "age": rng.uniform(20, 80, 600)})
    y = rng.normal(size=600)
    half = frame.iloc[:300]
    pipe = Pipeline([("form", ef.ExposureForms({"fiber_g": {"form": "spline", "knots": 4}})),
                     ("model", LinearRegression())]).fit(half, y[:300])
    learned = pipe.named_steps["form"].knots_["fiber_g"]
    np.testing.assert_allclose(learned, _rcspline_knots(half["fiber_g"].to_numpy(), 4), atol=1e-9)
    assert not np.allclose(learned, _rcspline_knots(frame["fiber_g"].to_numpy(), 4))


@pytest.mark.parametrize("method, strata", [("residual", None), ("residual", "gender"),
                                            ("density", None), ("density_multivariate", None),
                                            ("standard", None)])
def test_1_a_formed_nutrient_is_shaped_after_its_energy_adjustment(method, strata):
    """The pack's spline is "of outcome vs energy-adjusted intake" (§07G): the form step receives
    the energy step's output for the nutrient, whatever that step names it, and places its knots
    on those values."""
    from turbotab.core.models.steps import energy_step

    frame = mf.nhanes_like(400, seed=2)
    predictors = ["age", "gender", "kcal", "protein", "carb", "fat_total"]
    adjustment = mf.energy(method, strata=strata)
    X = frame[predictors]
    energy = energy_step(adjustment, predictors).fit(X)
    adjusted = energy.transform(X)
    name = ef.formed_name("protein", adjustment)
    assert name in list(energy.get_feature_names_out())
    pipe = Pipeline([("energy", energy_step(adjustment, predictors)),
                     ("form", ef.ExposureForms(ef.adjusted_forms(
                         {"protein": {"form": "spline", "knots": 4}}, adjustment)))]).fit(X)
    np.testing.assert_allclose(pipe.named_steps["form"].knots_[name],
                               _rcspline_knots(adjusted[name].to_numpy(dtype=float), 4), atol=1e-9)
    assert ef.spline_names(name, 4)[1] in pipe.transform(X).columns


# ── 1 · the nonlinearity test ────────────────────────────────────────────────


@pytest.mark.parametrize("nk", [3, 4, 5])
def test_1_spline_tests_match_statsmodels_hc3_f_tests(tmp_path, nk):
    """Regression under inference: the overall and nonlinear Wald tests equal statsmodels'
    ``f_test`` on an HC3 OLS fit of the same columns (the closed-form basis, reference knots)."""
    frame = _curved(800, seed=nk)
    entry, train = _run(frame, tmp_path, ExposureFormSpec(form="spline", knots=nk))
    training = frame.iloc[train]
    knots = _rcspline_knots(training["fiber_g"].to_numpy(), nk)
    X = _spline_frame(training, knots)
    reference = sm.OLS(training["y"].to_numpy(), sm.add_constant(X)).fit(cov_type="HC3", use_t=True)
    terms = ["fiber_g", *[f"s{j + 1}" for j in range(nk - 2)]]
    overall = reference.f_test(np.eye(len(X.columns) + 1)[[1 + list(X.columns).index(c) for c in terms]])
    nonlinear = reference.f_test(np.eye(len(X.columns) + 1)[[1 + list(X.columns).index(c) for c in terms[1:]]])

    tests = {t["test"]: t for t in entry["exposure_tests"]}
    assert set(tests) == {"overall", "nonlinear"}
    for kind, ref in (("overall", overall), ("nonlinear", nonlinear)):
        t = tests[kind]
        assert t["distribution"] == "F"
        assert (t["df_num"], t["df_den"]) == (int(ref.df_num), float(ref.df_denom))
        assert t["statistic"] == pytest.approx(float(np.squeeze(ref.fvalue)), rel=1e-6)
        assert t["p"] == pytest.approx(float(ref.pvalue), rel=1e-6, abs=1e-300)
        np.testing.assert_allclose(t["knots"], knots, atol=1e-6)
    # The coefficient rows are rms's columns, on rms's scale.
    rows = {r["feature"]: r for r in entry["coefficients"]}
    for ours, theirs in zip(ef.spline_names("fiber_g", nk), terms):
        assert rows[ours]["estimate"] == pytest.approx(float(reference.params[theirs]), rel=1e-6)


def test_1_spline_tests_on_a_binary_outcome_match_statsmodels_logit_wald(tmp_path):
    frame = _curved(1_500, seed=21)
    frame["high"] = (frame["y"] > frame["y"].median()).astype(int)
    frame = frame.drop(columns="y")
    entry, train = _run(frame, tmp_path, ExposureFormSpec(form="spline", knots=4), task="binary",
                        target="high")
    training = frame.iloc[train]
    knots = _rcspline_knots(training["fiber_g"].to_numpy(), 4)
    X = sm.add_constant(_spline_frame(training, knots))
    reference = sm.Logit(training["high"].to_numpy(), X).fit(disp=False, method="newton", tol=1e-12)
    tests = {t["test"]: t for t in entry["exposure_tests"]}
    for kind, names in (("overall", ["fiber_g", "s1", "s2"]), ("nonlinear", ["s1", "s2"])):
        R = np.eye(X.shape[1])[[list(X.columns).index(c) for c in names]]
        ref = reference.wald_test(R, use_f=False, scalar=True)
        assert tests[kind]["distribution"] == "chi2"
        assert tests[kind]["df_num"] == len(names)
        assert tests[kind]["statistic"] == pytest.approx(float(ref.statistic), rel=1e-5)
        assert tests[kind]["p"] == pytest.approx(float(ref.pvalue), rel=1e-4)


def _nonlinear_p(frame: pd.DataFrame) -> float:
    """The app's nonlinearity p-value through its own pipeline step, inference table and test."""
    linear = get_family("linear")
    X, y = frame[["age", "fiber_g"]], frame["y"].to_numpy()
    pipe = Pipeline([("form", ef.ExposureForms({"fiber_g": {"form": "spline", "knots": 4}})),
                     ("model", LinearRegression())]).fit(X, y)
    table = linear.inference(pipe, X, y, task="regression", clusters=INDEPENDENT)
    tests, _ = ef.exposure_tests(linear, pipe, X, y, task="regression", clusters=INDEPENDENT,
                                 table=table)
    return next(t["p"] for t in tests if t["test"] == "nonlinear")


def test_1_the_nonlinearity_test_holds_its_size_and_finds_a_curve():
    """n = 300, errors whose SD grows with intake. Straight-line truth, 600 datasets: rejections
    at 0.05 ≤ 0.08 (nominal 0.05, Monte Carlo SE 0.009, so 0.08 is 3.4 SE above; a classical-
    covariance F test on this design rejects about 0.09, measured over 1,000 datasets while writing
    this test, which is why the table's HC3 covariance is the one tested). Log-shaped truth (the
    fixture's), 100 datasets: rejections ≥ 0.90."""
    n = 300
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        size = []
        for r in range(600):
            rng = np.random.default_rng(1_000 + r)
            age, fiber = rng.uniform(20, 80, n), _intake(n, rng)
            noise = rng.normal(0, 1.0 + 0.03 * fiber, n)
            line = pd.DataFrame({"age": age, "fiber_g": fiber, "y": 0.03 * age + 0.1 * fiber + noise})
            size.append(_nonlinear_p(line) < 0.05)
        power = [_nonlinear_p(_curved(n, seed=5_000 + r)) < 0.05 for r in range(100)]
    assert np.mean(size) <= 0.08, np.mean(size)
    assert np.mean(power) >= 0.90, np.mean(power)


# ── 1 · quintiles and the trend on quintile medians ──────────────────────────


def test_1_quintile_estimates_and_trend_match_statsmodels_on_pandas_qcut(tmp_path):
    frame = _curved(1_000, seed=4)
    entry, train = _run(frame, tmp_path, ExposureFormSpec(form="quintiles"))
    training = frame.iloc[train].reset_index(drop=True)
    group = pd.qcut(training["fiber_g"], 5, labels=False)
    X = training[["age"]].copy()
    for g in range(1, 5):
        X[f"Q{g + 1}"] = (group == g).astype(float)
    reference = sm.OLS(training["y"].to_numpy(), sm.add_constant(X)).fit(cov_type="HC3", use_t=True)
    ci = reference.conf_int()
    rows = {r["feature"]: r for r in entry["coefficients"]}
    for g in range(2, 6):
        row = rows[f"fiber_g_Q{g}"]
        assert row["estimate"] == pytest.approx(float(reference.params[f"Q{g}"]), rel=1e-8)
        assert row["ci_low"] == pytest.approx(float(ci.loc[f"Q{g}", 0]), rel=1e-8)
        assert row["ci_high"] == pytest.approx(float(ci.loc[f"Q{g}", 1]), rel=1e-8)

    # The trend: each row scored by its quintile's median (not 1…5), one continuous term.
    medians = training.groupby(group)["fiber_g"].median()
    score = group.map(medians).astype(float)
    Xt = pd.DataFrame({"age": training["age"], "score": score})
    trend_ref = sm.OLS(training["y"].to_numpy(), sm.add_constant(Xt)).fit(cov_type="HC3", use_t=True)
    (trend,) = [t for t in entry["exposure_tests"] if t["test"] == "trend"]
    np.testing.assert_allclose(trend["medians"], medians.to_numpy(), rtol=1e-12)
    assert trend["estimate"] == pytest.approx(float(trend_ref.params["score"]), rel=1e-8)
    assert trend["p"] == pytest.approx(float(trend_ref.pvalues["score"]), rel=1e-6, abs=1e-300)
    assert trend["ci_low"] == pytest.approx(float(trend_ref.conf_int().loc["score", 0]), rel=1e-8)
    assert trend["distribution"] == "t" and trend["df_den"] == pytest.approx(trend_ref.df_resid)
    # Scoring by the quintile number instead gives a different coefficient: the medians are used.
    by_rank = sm.OLS(training["y"].to_numpy(),
                     sm.add_constant(pd.DataFrame({"age": training["age"], "q": group + 1.0}))).fit()
    assert abs(float(by_rank.params["q"]) - trend["estimate"]) > 0.01


def test_1_a_quintile_cut_holds_its_value_in_the_lower_group_as_qcut_does():
    values = np.arange(1.0, 101.0)
    cuts = ef.quantile_cuts(values)
    np.testing.assert_array_equal(ef.quantile_group(values, cuts),
                                  pd.qcut(values, 5, labels=False))


# ── the options, labeled and ordered for the purpose (north star 5) ──────────


def test_1_under_inference_the_spline_ranks_first_and_quintiles_are_tagged_customary():
    for purpose in ("inference", "prediction"):
        options = ef.options(purpose)
        # FORM widened the menu (the shelf is never shortened): declared categories and the
        # data-derived cut point follow, ranked lower (blocked and recorded under inference).
        assert [o["value"] for o in options] == ["spline", "linear", "quintiles", "categories",
                                                 "optimal"]
        for option in options:
            assert option["customary"] and option["sound"]
    quintiles = ef.options("inference")[2]
    assert quintiles["customary"].startswith("Customary") and "quintiles remain expected" in quintiles["customary"]
    assert quintiles["sound"].startswith("Weaker for inference")
    pack = (REPO / "docs" / "turbotab-next" / "reference" / "research" / "NUTRITION_PACK.md").read_text()
    assert "now near-default; quintiles remain expected alongside" in pack
    assert "using the median of each quintile as a continuous score, not the quintile number" in pack


def test_1_the_labeled_options_are_served_with_the_proposals(tmp_path):
    """Repair round (verifier minor 3): the exposure-form options with their "customary in" and
    "sound for" labels were never served by the API. The proposals stage serves them, ordered for
    the declared purpose, in the published shape (``ProposalsArtifact``)."""
    from turbotab.core.decisions import ProjectState
    from turbotab.core.stages.proposals import proposals_stage
    from turbotab.core.tests.stage_harness import Ingested
    from turbotab.server.schemas import ProposalsArtifact

    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"age": rng.uniform(20, 80, 200), "protein_g": rng.gamma(9, 9, 200),
                          "energy_kcal": rng.normal(2000, 300, 200), "y": rng.normal(size=200)})
    source = tmp_path / "t.csv"
    frame.to_csv(source, index=False)
    table = Ingested(source, tmp_path / "i")
    for purpose in ("inference", "prediction"):
        st = ProjectState(lens=["dietary"], target="y", purpose=purpose,
                          roles={"age": "covariate", "protein_g": "exposure",
                                 "energy_kcal": "energy"})
        served = ProposalsArtifact.model_validate(table.run(proposals_stage, st, {"roles": None}))
        assert [o.model_dump() for o in served.exposure_forms] == ef.options(purpose)

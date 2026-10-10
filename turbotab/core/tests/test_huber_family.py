"""RT-5c · robust linear regression (Huber), against references that are not the code's own output
(RECIPES_AND_TUNING §2.2 and T11; MODEL_FAMILY_CONTRACT §3.2; WAVE_C6A_PLAN §3).

* **The fit** against an iteratively reweighted least squares written out below from Holland &
  Welsch (1977): start at least squares; each step re-estimates the scale as the MAD,
  median|r| / 0.6745, weights each row by min(1, t / |r/s|) with t = 1.345 (Huber 1964), and solves
  the weighted normal equations; to 1e-6.
* **R's MASS::rlm** on Brownlee's stack-loss data, its printed coefficients copied in with their
  source, to the 4 decimals it prints; and the same call run to convergence, to 1e-6.
* **Invariance** (C3, C13): any invertible linear map of the inputs keeps the predictions; a log of a
  column changes them.
* **SHAP** in closed form, φ_ij = β_j (z_ij − z̄_j) (Lundberg & Lee 2017), with β from the IRLS here.
* **Purposes** are declared, never checked by key (V2X_SEAMS row 8): the shelf reads ``purposes``.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

T_STANDARD = 1.345  # Huber (1964): 95% of least squares' efficiency under normal errors
MAD_CONSTANT = 0.6745  # Holland & Welsch (1977) and MASS::rlm: the normal's upper quartile, rounded

# Brownlee's stack-loss data (R's ``datasets::stackloss``, 21 rows): Air.Flow, Water.Temp,
# Acid.Conc., stack.loss, printed by R 4.6.1.
STACKLOSS = np.array([
    (80, 27, 89, 42), (80, 27, 88, 37), (75, 25, 90, 37), (62, 24, 87, 28), (62, 22, 87, 18),
    (62, 23, 87, 18), (62, 24, 93, 19), (62, 24, 93, 20), (58, 23, 87, 15), (58, 18, 80, 14),
    (58, 18, 89, 14), (58, 17, 88, 13), (58, 18, 82, 11), (58, 19, 93, 12), (50, 18, 89, 8),
    (50, 18, 86, 7), (50, 19, 72, 8), (50, 19, 79, 8), (50, 20, 80, 9), (56, 20, 82, 15),
    (70, 20, 91, 15)], dtype=float)
STACK_COLUMNS = ["Air.Flow", "Water.Temp", "Acid.Conc."]
# ``summary(rlm(stack.loss ~ ., stackloss))``, the example on MASS's ``?rlm`` page, as R 4.6.1 with
# MASS 7.3-65 prints it ("Value" column): (Intercept), Air.Flow, Water.Temp, Acid.Conc. The same
# four values are statsmodels' own R reference (statsmodels/robust/tests/results/results_rlm.py,
# class Huber). MASS stops at its default ``acc = 1e-4``, so these sit within about 1e-4 of the
# exact solution: they are compared at the 4 decimals printed.
MASS_PRINTED = (-41.0265, 0.8294, 0.9261, -0.1278)
# The same call with ``acc = 1e-14, maxit = 500`` (R 4.6.1, MASS 7.3-65), printed to 15 digits, and
# its scale ``$s``: the exact solution, which MASS reaches with the scale median|r| / 0.6745.
MASS_CONVERGED = (-41.026485373295010, 0.829385770253855, 0.926059415548607, -0.127846317965416)
MASS_CONVERGED_SCALE = 2.4404890459928


def _irls(X: np.ndarray, y: np.ndarray, t: float = T_STANDARD) -> tuple[np.ndarray, float]:
    """Huber's M-estimator by IRLS, as Holland & Welsch (1977) set it out: (intercept and
    coefficients, the final scale)."""
    A = np.column_stack([np.ones(len(y)), X])
    beta = np.linalg.lstsq(A, y, rcond=None)[0]
    for _ in range(2000):
        r = y - A @ beta
        s = np.median(np.abs(r)) / MAD_CONSTANT
        u = np.abs(r / s)
        w = np.where(u <= t, 1.0, t / np.maximum(u, 1e-300))
        new = np.linalg.solve(A.T @ (w[:, None] * A), A.T @ (w * y))
        if np.max(np.abs(new - beta)) < 1e-13 * max(1.0, np.max(np.abs(beta))):
            beta = new
            break
        beta = new
    else:  # pragma: no cover - the fixtures converge
        raise AssertionError("the reference IRLS did not converge")
    r = y - A @ beta
    return beta, float(np.median(np.abs(r)) / MAD_CONSTANT)


def _data(seed: int = 11, n: int = 240) -> tuple[pd.DataFrame, np.ndarray]:
    """Three positive predictors, heavy-tailed errors and a few gross outliers."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.uniform(1, 5, n), "b": rng.uniform(1, 10, n),
                      "c": rng.lognormal(0, 0.5, n)})
    y = 2.0 + 1.5 * X["a"] - 0.7 * X["b"] + 0.4 * X["c"] + rng.standard_t(2, n)
    y = y.to_numpy()
    y[:8] += rng.choice([-1, 1], 8) * rng.uniform(15, 30, 8)
    return X, y


def _family():
    from turbotab.core.models import get_family

    return get_family("huber")


def _fit(X: pd.DataFrame, y: np.ndarray, **params):
    model = _family().build("regression", "prediction", len(X), X.shape[1])
    return model.set_params(**params).fit(X, y)


# ── the fit ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("t", [T_STANDARD, 2.0])
def test_the_fit_is_holland_and_welschs_irls(t: float) -> None:
    X, y = _data()
    beta, scale = _irls(X.to_numpy(), y, t=t)
    model = _fit(X, y, t=t)
    np.testing.assert_allclose(model.intercept_, beta[0], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(model.coef_, beta[1:], rtol=1e-6, atol=1e-6)
    # the scale is median|r| / 0.6745, as Holland & Welsch and MASS write it (statsmodels' own
    # default divides by the exact quartile 0.67449, a relative 1.5e-5 away)
    np.testing.assert_allclose(model.scale_, scale, rtol=1e-8)
    assert model.converged_ and list(model.feature_names_in_) == ["a", "b", "c"]
    np.testing.assert_allclose(model.predict(X), beta[0] + X.to_numpy() @ beta[1:], rtol=1e-6,
                               atol=1e-6)


def test_the_stack_loss_fit_is_mass_rlms() -> None:
    X = pd.DataFrame(STACKLOSS[:, :3], columns=STACK_COLUMNS)
    y = STACKLOSS[:, 3]
    model = _fit(X, y)
    found = np.array([model.intercept_, *model.coef_])
    assert tuple(np.round(found, 4)) == MASS_PRINTED
    np.testing.assert_allclose(found, MASS_CONVERGED, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(model.scale_, MASS_CONVERGED_SCALE, rtol=1e-6)
    # and the hand IRLS agrees with R, so the two references are one
    np.testing.assert_allclose(_irls(STACKLOSS[:, :3], y)[0], MASS_CONVERGED, rtol=1e-8)


def test_a_fit_that_runs_out_of_steps_says_so() -> None:
    X, y = _data()
    with pytest.warns(Warning, match="did not converge"):
        model = _fit(X, y, max_iter=2)
    assert not model.converged_


@pytest.mark.parametrize("params, says", [
    ({"t": 0.0}, "threshold"), ({"t": float("nan")}, "threshold"),
    # the declaration sets the threshold by hand in [1, 3] (RECIPES §4.1): outside it, no fit
    ({"t": 0.5}, "threshold"), ({"t": 5.0}, "threshold"),
    ({"scale": "Huber"}, "MAD"), ({"penalty": "l2"}, "no penalty")])
def test_a_setting_outside_the_declaration_is_refused(params: dict, says: str) -> None:
    X, y = _data(n=60)
    with pytest.raises(ValueError, match=says):
        _fit(X, y, **params)


def test_a_near_exact_fit_converges_without_a_warning() -> None:
    """Most rows on the line put the MAD at floating-point dust, not 0: that fit is exact, so it
    converges and says nothing. The line y = x is the hand answer."""
    x = np.concatenate([np.zeros(40), np.arange(10.0)])
    X = pd.DataFrame({"a": x})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = _fit(X, x.copy())
    assert model.converged_
    np.testing.assert_allclose([model.intercept_, *model.coef_], [0.0, 1.0], atol=1e-12)


# ── invariance (C3) ──────────────────────────────────────────────────────────


def test_a_linear_map_keeps_the_predictions_and_a_log_changes_them() -> None:
    X, y = _data()
    base = _fit(X, y).predict(X)
    rng = np.random.default_rng(5)
    A = rng.normal(size=(3, 3)) + 3 * np.eye(3)
    assert abs(np.linalg.det(A)) > 1
    shift = np.array([4.0, -2.0, 7.5])
    mapped = pd.DataFrame(X.to_numpy() @ A + shift, columns=["u", "v", "w"])
    np.testing.assert_allclose(_fit(mapped, y).predict(mapped), base, rtol=1e-9, atol=1e-9)
    logged = X.assign(b=np.log(X["b"]))
    moved = np.abs(_fit(logged, y).predict(logged) - base)
    assert moved.max() > 0.05


# ── SHAP in closed form (C10) ────────────────────────────────────────────────


def test_its_shap_values_are_the_closed_form() -> None:
    from sklearn.pipeline import Pipeline

    from turbotab.core.models import explain as E

    X, y = _data()
    model = _family().build("regression", "prediction", len(X), 3)
    pipe = Pipeline([("model", model)]).set_output(transform="pandas").fit(X, y)
    anat = E.anatomy(pipe, list(X.columns), "regression")
    assert E.model_kind("huber", anat) == "linear"
    got = E.attributions(anat, X, X, kind="linear", family="huber")
    beta, _ = _irls(X.to_numpy(), y)
    mean = X.to_numpy().mean(axis=0)
    by_hand = (X.to_numpy() - mean) * beta[1:]
    np.testing.assert_allclose(got.phi.to_numpy(), by_hand, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(got.expected, beta[0] + beta[1:] @ mean, rtol=1e-6)
    # local accuracy: the values and the expected value add up to the prediction
    np.testing.assert_allclose(got.phi.sum(axis=1) + got.expected, pipe.predict(X), rtol=1e-10,
                               atol=1e-10)


# ── the declaration ──────────────────────────────────────────────────────────


def test_it_declares_its_recipe_and_a_threshold_set_only_by_hand() -> None:
    from turbotab.core.models import families
    from turbotab.core.models import tuning as T
    from turbotab.core.models.base import contract_problems

    f = _family()
    assert contract_problems(f) == []
    assert (f.tasks, f.purposes, f.label) == (("regression",), ("prediction",),
                                              "Robust linear regression")
    assert (f.invariances, f.curve_shape, f.attribution, f.output, f.inference_decl) == (
        ("linear_maps",), "straight", "linear", "value", None)
    assert (f.flexible, f.bootstrap_optimism, f.linear_in_values) == (False, True, True)
    assert {s.key for s in f.sources} == {"huber1964", "holland1977irls"}
    decl = f.tuning
    assert (decl.kind, decl.dimensions, dict(decl.standard), decl.standard_source,
            dict(decl.fixed), decl.space_version) == (
        "none", (), {"t": T_STANDARD}, "Huber 1964", {"scale": "MAD", "penalty": "none"},
        "huber/1")
    (t,) = decl.by_hand
    assert (t.name, t.low, t.high, t.scale) == ("t", 1, 3, "linear")
    assert [k.setting for k in f.complexity] == ["t"]
    # the plan fits once, at the standard threshold, or at one set by hand
    plan = T.make_plan(f, task="regression", loss="mse", n_plan=400, plan_rows=400, unit="units",
                       split_seed=3)
    assert ([dict(c.values) for c in plan.candidates], plan.inner_k, plan.fits()) == (
        [{"t": T_STANDARD}], 0, 1)
    by_hand = T.make_plan(f, task="regression", loss="mse", n_plan=400, plan_rows=400,
                          unit="units", split_seed=3, mode="manual", manual={"t": 2.0})
    params = T.estimator_params(f, "regression", by_hand.candidates[0].values, n_units=400,
                                n_rows=400)
    assert params == {"t": 2.0, "scale": "MAD", "penalty": "none"}
    X, y = _data()
    fitted = f.build("regression", "prediction", len(X), 3).set_params(**params).fit(X, y)
    np.testing.assert_allclose(fitted.coef_, _irls(X.to_numpy(), y, t=2.0)[0][1:], rtol=1e-6)
    # it registers after linear, so a tie on the shelf goes to linear
    keys = [g.key for g in families()]
    assert keys.index("linear") < keys.index("huber")


def test_its_coefficients_are_the_fitted_equation_without_intervals() -> None:
    from sklearn.pipeline import Pipeline

    X, y = _data()
    pipe = Pipeline([("model", _family().build("regression", "prediction", len(X), 3))])
    pipe.set_output(transform="pandas").fit(X, y)
    rows = _family().coefficients(pipe, X, y, task="regression", purpose="prediction")
    beta, _ = _irls(X.to_numpy(), y)
    assert [r["feature"] for r in rows] == ["(intercept)", "a", "b", "c"]
    np.testing.assert_allclose([r["estimate"] for r in rows], beta, rtol=1e-6)
    assert all(r["ci_low"] is None and r["p"] is None for r in rows)


def test_its_purposes_are_read_from_the_declaration_never_its_key() -> None:
    """V2X_SEAMS row 8: under inference the shelf says why it is not offered, reading
    ``purposes``; the same family declaring inference too is judged as any other."""
    from turbotab.core.models.base import Situation, assessment
    from turbotab.core.models.huber import Huber

    situation = Situation(task="regression", purpose="inference", n_rows=400, n_features=3)
    said = assessment(_family(), situation)
    assert said.fit == "poor" and any("prediction" in c for c in said.concerns)
    widened = type("Widened", (Huber,), {"purposes": ("prediction", "inference")})()
    other = assessment(widened, situation)
    assert other.fit != "poor" and not any("prediction" in c for c in other.concerns)
    under_prediction = assessment(_family(), Situation(task="regression", purpose="prediction",
                                                       n_rows=400, n_features=3))
    assert under_prediction.fit == "good"
    # RECIPES_AND_TUNING §2.2's shelf table: "Robust linear | 1.5 (numeric only; after linear on
    # ties)". Under prediction it scores as linear does on the same rows (1.5, a point less when
    # the rows fall short); linear registers first, so a tie goes to linear.
    assert under_prediction.score == 1.5
    from turbotab.core.models import families, get_family

    linear = get_family("linear")
    for purpose in ("prediction", "inference", None):
        for rows, columns in ((400, 3), (40, 12), (20, 30)):
            s = Situation(task="regression", purpose=purpose, n_rows=rows, n_features=columns)
            mine, theirs = assessment(_family(), s).score, assessment(linear, s).score
            assert mine <= theirs, (purpose, rows, columns)
            if purpose != "inference":
                assert mine == theirs, (purpose, rows, columns)
    keys = [g.key for g in families()]
    assert keys.index("linear") < keys.index("huber")


def test_no_warning_escapes_an_ordinary_fit() -> None:
    X, y = _data()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _fit(X, y)


def test_choosing_it_under_inference_is_refused() -> None:
    from turbotab.core.decisions import Refusal, SelectModels, validate

    with pytest.raises(Refusal):
        validate(SelectModels(models=["huber"]), {"task": "regression", "purpose": "inference"})

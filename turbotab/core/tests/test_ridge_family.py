"""RT-5b · the ridge family (RECIPES_AND_TUNING §2.2, §4.1, T11; MODEL_FAMILY_CONTRACT §3.2, C7, C13).

Every reference here is computed in the test, independently of the family's code:

* least squares: the closed form on the centered matrix with the intercept unpenalized,
  β(λ) = (ZcᵀZc + nλI)⁻¹Zcᵀ(y − ȳ), b = ȳ − z̄ᵀβ, solved by numpy at every grid point; where
  LAPACK's gesdd is made to fail, also the augmented least squares [Zc; √(nλ)I]β = [y − ȳ; 0] by
  Householder QR, with no SVD;
* df(λ): the trace of the explicit hat matrix Zc(ZcᵀZc + nλI)⁻¹Zcᵀ (ESL §3.4.1, eq. 3.50);
* logistic ridge: scipy's L-BFGS on the penalized log-likelihood written out here,
  Σᵢ ℓᵢ + (nλ/2)‖W‖², the intercepts unpenalized, for a yes/no outcome and for classes;
* linear SHAP: βⱼ(zᵢⱼ − z̄ⱼ) on the columns standardized by hand, β from the references above;
* invariance (C3, C13): an orthogonal rotation after the scaling keeps every prediction, and a
  shear changes them;
* logistic ridge's knob factors (C7): the eigenvalues of the penalized likelihood's Hessian at the
  fit, built here from [1, Z] and the fitted probabilities, its intercept profiled out (a Schur
  complement), so df = tr[(S + κI)⁻¹S] with κ = 1/C;
* the shelf (RECIPES §2.6): the first two predicting families with ridge on the shelf are those of
  the same shelf with ridge taken off, the families the reference journeys pick today;
* the card's words: RECIPES §6.5's vocabulary row for λ, read from the spec.

T2(a) (the pooled argmin on the diabetes data) waits for the engine (phase 3).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from turbotab.core.models import get_family
from turbotab.core.models.base import contract_problems
from turbotab.core.models.formulas import FORMULAS
from turbotab.core.models.sources import SOURCES
from turbotab.core.models.tuning import estimator_params, path_grid, tuning_for

N, P = 160, 5
REPO = Path(__file__).resolve().parents[3]


def _family():
    import turbotab.core.models.ridge  # noqa: F401 - registers

    return get_family("ridge")


def _data(seed: int = 0, n: int = N, p: int = P) -> tuple[np.ndarray, np.ndarray]:
    """Correlated columns on unequal scales, and a linear signal with noise."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, p))
    mix = np.eye(p) + 0.6 * np.tri(p, p, -1)
    Z = base @ mix.T * np.array([1.0, 3.0, 0.5, 10.0, 2.0])[:p] + np.arange(p)
    beta = np.array([1.0, -0.5, 2.0, 0.05, 0.0])[:p]
    y = Z @ beta + rng.normal(scale=2.0, size=n) + 4.0
    return Z, y


def _closed_form(Z: np.ndarray, y: np.ndarray, lam: float,
                 w: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Ridge by the normal equations, the intercept unpenalized: (weighted) centering, then
    (ZcᵀWZc + nλI)β = ZcᵀW(y − ȳ)."""
    n, p = Z.shape
    w = np.ones(n) if w is None else np.asarray(w, dtype=float)
    zbar = w @ Z / w.sum()
    ybar = w @ y / w.sum()
    Zc, yc = Z - zbar, y - ybar
    beta = np.linalg.solve(Zc.T @ (w[:, None] * Zc) + n * lam * np.eye(p), Zc.T @ (w * yc))
    return beta, float(ybar - zbar @ beta)


def _penalized_logistic(Z: np.ndarray, y: np.ndarray, lam: float,
                        w: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The minimizer of Σᵢ wᵢ·(−log softmax(Wzᵢ + b)[yᵢ]) + (nλ/2)‖W‖²_F by scipy's L-BFGS (wᵢ = 1
    unless given): for a yes/no outcome one row (the logit of class 1), for K classes K rows, as
    scikit-learn's multinomial parameterizes it. Returns (coef (K, p), intercept (K,))."""
    n, p = Z.shape
    w = np.ones(n) if w is None else np.asarray(w, dtype=float)
    classes = np.unique(y)
    K = 1 if len(classes) == 2 else len(classes)
    Y = (y == classes[1]).astype(float)[:, None] if K == 1 else (
        y[:, None] == classes[None, :]).astype(float)

    def unpack(theta):
        return theta[:K * p].reshape(K, p), theta[K * p:]

    def objective(theta):
        W, b = unpack(theta)
        eta = Z @ W.T + b
        if K == 1:
            nll = np.sum(w * (np.logaddexp(0.0, eta[:, 0]) - Y[:, 0] * eta[:, 0]))
            resid = w[:, None] * (1.0 / (1.0 + np.exp(-eta)) - Y)
        else:
            m = eta.max(axis=1, keepdims=True)
            lse = m[:, 0] + np.log(np.exp(eta - m).sum(axis=1))
            nll = np.sum(w * (lse - np.sum(Y * eta, axis=1)))
            resid = w[:, None] * (np.exp(eta - lse[:, None]) - Y)
        grad_W = resid.T @ Z + n * lam * W
        grad_b = resid.sum(axis=0)
        return nll + 0.5 * n * lam * np.sum(W ** 2), np.concatenate([grad_W.ravel(), grad_b])

    fit = minimize(objective, np.zeros(K * p + K), jac=True, method="L-BFGS-B",
                   options={"gtol": 1e-12, "ftol": 1e-15, "maxiter": 50_000, "maxcor": 30})
    return unpack(fit.x)


def _fitted(family, task: str, Z: np.ndarray, y: np.ndarray, lam: float):
    est = family.build(task, "prediction", len(y), Z.shape[1])
    est.set_params(**estimator_params(family, task, {"lambda": lam}, n_units=len(y),
                                      n_rows=len(y), y=y, Z=Z))
    return est.fit(Z, y)


# ── the declaration ──────────────────────────────────────────────────────────


def test_ridge_registers_with_the_declarations_the_plan_rules():
    """WAVE_C6A_PLAN §2 and §7, RECIPES §2.2 and §4.1, MODEL_FAMILY_CONTRACT §3.2, written out."""
    f = _family()
    assert contract_problems(f) == []
    assert f.tasks == ("regression", "binary", "multiclass", "ordinal")
    decl = tuning_for(f, "regression")
    assert decl is not None and decl.kind == "path" and decl.space_version == "ridge/1"
    assert all(tuning_for(f, t) is decl for t in f.tasks)
    (dim,) = decl.dimensions
    assert (dim.name, dim.low, dim.high, dim.scale, dim.points) == ("lambda", 1e-5, 1e2, "log", 50)
    # RECIPES §4.1: 50 points from 10² down to 10⁻⁵, log-spaced, the strongest penalty first
    assert np.allclose(path_grid(decl)["lambda"], 10.0 ** np.linspace(2, -5, 50), rtol=1e-12)
    assert f.path is not None and f.settings is not None and f.trees is None
    assert f.updating == ()  # §5 ruling 7: a ridge's calibration-slope shrinkage is not sourced
    assert f.attribution == "linear" and f.output == "margin"
    assert f.invariances == ("rotation_after_scaling",) and f.curve_shape == "straight"
    assert f.inference_decl.table == "shrunk_no_intervals" and f.inference is None
    assert (f.flexible, f.bootstrap_optimism, f.needs_scaling) == (False, True, True)
    assert f.inductive_bias == ("Straight-line effects all shrunk toward zero together; "
                                "correlated predictors share weight; none is dropped.")
    assert f.consequence == ("A penalized straight-line model that shrinks every effect and keeps "
                             "every predictor.")
    cited = {s.key for s in (*f.sources, *(t.source for t in f.bias_terms),
                             *(k.source for k in f.complexity if k.source),
                             *(q.source for q in f.sample_efficiency if q.source))}
    assert {"hastie2009", "ng2004l1l2", "kobak2020ridge", "probst2019tunability"} <= cited
    assert cited <= set(SOURCES)
    (knob,) = f.complexity
    assert (knob.setting, knob.more_means, knob.formula) == ("lambda", "simpler", "ridge")


def test_settings_turn_a_per_row_penalty_into_each_estimators_own():
    """α = nλ for least squares; C = 1/(nλ) for logistic ridge, with l1_ratio 0 (all L2)."""
    f = _family()
    lam, n = 0.03, 250
    reg = estimator_params(f, "regression", {"lambda": lam}, n_units=n, n_rows=n)
    assert reg == {"alpha": pytest.approx(7.5, rel=1e-15)}
    for task in ("binary", "multiclass", "ordinal"):
        got = estimator_params(f, task, {"lambda": lam}, n_units=n, n_rows=n)
        assert got == {"C": pytest.approx(1 / 7.5, rel=1e-15)}
        assert f.build(task, "prediction", n, 3).get_params()["l1_ratio"] == 0.0


@pytest.mark.parametrize("p,n_rows,want", [(5, 2_500, 2.0), (5, 2_000, 2.0), (5, 1_999, 1.0),
                                           (5, 120, 1.0), (300, 120, 3.0), (3_000, 2_500, 3.0)])
def test_the_shelf_scores_are_recipes_conventions(p, n_rows, want):
    """RECIPES §2.6: 2.0 from 2,000 training rows; at p ≥ n, 3.0 (below the elastic net's 4.0);
    below 2,000 rows 1.0, under the 1.5 of the families the reference journeys pick second there."""
    from turbotab.core.models.base import Situation

    s = Situation(task="regression", purpose="prediction", n_rows=n_rows, n_features=p)
    assert _family().assess(s).score == want


# ── T11: the closed form, df(λ), logistic ridge ──────────────────────────────


def test_the_path_is_the_closed_form_at_every_grid_point():
    f = _family()
    Z, y = _data()
    grid = path_grid(tuning_for(f, "regression"))
    lams = 10.0 ** np.linspace(2, -5, 50)
    fit = f.path(Z, y, grid, task="regression")
    assert fit.coefs.shape == (1, 50, 1, P) and fit.intercepts.shape == (1, 50, 1)
    assert np.allclose(fit.values[0], lams, rtol=1e-12)
    for g, lam in enumerate(lams):
        beta, b = _closed_form(Z, y, lam)
        assert np.max(np.abs(fit.coefs[0, g, 0] - beta)) < 1e-8, lam
        assert abs(fit.intercepts[0, g, 0] - b) < 1e-8, lam
        # the refit at that point (α = nλ) is the same fit
        est = _fitted(f, "regression", Z, y, lam)
        assert np.max(np.abs(est.coef_ - beta)) < 1e-8 and abs(est.intercept_ - b) < 1e-8, lam


def test_a_weighted_path_is_the_weighted_closed_form_at_weights_of_mean_one():
    """Weights are relative, scaled to mean 1 as glmnet scales them to sum to n, so λ keeps its
    per-row meaning: weights summing to a population total give the same path, and the refit at
    α = nλ with ``sample_weight`` w/mean(w) is the path's point."""
    f = _family()
    Z, y = _data(1)
    w = np.random.default_rng(5).uniform(0.2, 3.0, size=len(y))
    relative = w / w.mean()
    grid = {"lambda": np.array([10.0, 0.1, 1e-4])}
    fit = f.path(Z, y, grid, task="regression", weights=w)
    total = f.path(Z, y, grid, task="regression", weights=w * 25_000.0)
    for g, lam in enumerate(grid["lambda"]):
        beta, b = _closed_form(Z, y, lam, relative)
        assert np.max(np.abs(fit.coefs[0, g, 0] - beta)) < 1e-8
        assert abs(fit.intercepts[0, g, 0] - b) < 1e-8
        assert np.max(np.abs(total.coefs[0, g, 0] - beta)) < 1e-8
        est = f.build("regression", "prediction", len(y), P)
        est.set_params(**estimator_params(f, "regression", {"lambda": lam}, n_units=len(y),
                                          n_rows=len(y)))
        est.fit(Z, y, sample_weight=relative)
        assert np.max(np.abs(est.coef_ - beta)) < 1e-8 and abs(est.intercept_ - b) < 1e-8


def test_df_is_the_trace_of_the_explicit_hat_matrix():
    """ESL eq. 3.50 with scikit-learn's α = nλ: df(λ) = tr Zc(ZcᵀZc + nλI)⁻¹Zcᵀ, from rank(Z) at 0
    toward 0, to 1e-10; and the fitted values are that hat matrix applied to y − ȳ."""
    f = _family()
    Z, y = _data(2)
    n = len(y)
    Zc = Z - Z.mean(axis=0)
    formula = FORMULAS[f.complexity[0].formula]
    for lam in (1e-5, 1e-2, 1.0, 1e2):
        alpha = estimator_params(f, "regression", {"lambda": lam}, n_units=n, n_rows=n)["alpha"]
        hat = Zc @ np.linalg.inv(Zc.T @ Zc + n * lam * np.eye(P)) @ Zc.T
        assert formula(Z, alpha).df == pytest.approx(np.trace(hat), abs=1e-10)
        est = _fitted(f, "regression", Z, y, lam)
        assert np.max(np.abs(est.predict(Z) - (y.mean() + hat @ (y - y.mean())))) < 1e-8
    assert formula(Z, 0.0).df == pytest.approx(P, abs=1e-10)


@pytest.mark.parametrize("task,classes", [("binary", 2), ("multiclass", 3)])
def test_logistic_ridge_is_the_penalized_likelihood_minimum(task, classes):
    f = _family()
    Z, signal = _data(3)
    Zs = (Z - Z.mean(axis=0)) / Z.std(axis=0)
    rng = np.random.default_rng(11)
    noisy = signal + rng.normal(scale=4.0, size=len(signal))
    y = np.digitize(noisy, np.quantile(noisy, np.linspace(0, 1, classes + 1)[1:-1]))
    grid = path_grid(tuning_for(f, task))
    picks = [0, 20, 35, 49]  # λ = 100, about 0.27, about 0.0038, 1e-5
    fit = f.path(Zs, y, {"lambda": grid["lambda"][picks]}, task=task)
    K = 1 if classes == 2 else classes
    assert fit.coefs.shape == (1, len(picks), K, P)
    for g, lam in enumerate(grid["lambda"][picks]):
        W, b = _penalized_logistic(Zs, y, lam)
        est = _fitted(f, task, Zs, y, lam)
        assert np.max(np.abs(est.coef_ - W)) < 1e-6, lam
        assert np.max(np.abs(est.intercept_ - b)) < 1e-6, lam
        assert np.max(np.abs(fit.coefs[0, g] - W)) < 1e-6, lam
        assert np.max(np.abs(fit.intercepts[0, g] - b)) < 1e-6, lam


# ── C10: linear SHAP; C3: rotation after scaling ─────────────────────────────


def _pipeline(f, task: str, n: int, p: int, lam: float, between=None) -> Pipeline:
    est = f.build(task, "prediction", n, p)
    est.set_params(**estimator_params(f, task, {"lambda": lam}, n_units=n, n_rows=n))
    steps = [("scale", StandardScaler())]
    if between is not None:
        steps.append(("turn", FunctionTransformer(lambda A: A @ between)))
    return Pipeline([*steps, ("model", est)])


@pytest.mark.parametrize("task", ["regression", "binary"])
def test_linear_shap_is_the_closed_form(task):
    from turbotab.core.models import explain as E

    f = _family()
    Z, signal = _data(4)
    y = signal if task == "regression" else (signal > np.median(signal)).astype(int)
    X = pd.DataFrame(Z, columns=[f"x{j}" for j in range(P)])
    lam = 0.05
    pipe = _pipeline(f, task, N, P, lam).set_output(transform="pandas").fit(X, y)
    anat = E.anatomy(pipe, list(X.columns), task)
    assert E.model_kind("ridge", anat) == "linear"
    got = E.attributions(anat, X, X, kind="linear", family="ridge")
    # by hand: the columns standardized (population SD, as the scaler), β from the references
    Zs = (Z - Z.mean(axis=0)) / Z.std(axis=0)
    if task == "regression":
        beta, b = _closed_form(Zs, y, lam)
    else:
        W, bb = _penalized_logistic(Zs, y, lam)
        beta, b = W[0], float(bb[0])
    want = beta * (Zs - Zs.mean(axis=0))
    tol = 1e-8 if task == "regression" else 1e-6
    assert np.max(np.abs(got.phi.to_numpy() - want)) < tol
    assert abs(got.expected - (b + beta @ Zs.mean(axis=0))) < tol
    raw = pipe.predict(X) if task == "regression" else pipe.decision_function(X)
    assert np.max(np.abs(got.expected + got.phi.to_numpy().sum(axis=1) - raw)) < 1e-10


@pytest.mark.parametrize("task", ["regression", "binary", "multiclass"])
def test_a_rotation_after_scaling_keeps_predictions_and_a_shear_changes_them(task):
    f = _family()
    Z, signal = _data(6)
    y = (signal if task == "regression" else
         np.digitize(signal, np.quantile(signal, [0.5] if task == "binary" else [1 / 3, 2 / 3])))
    rng = np.random.default_rng(9)
    Q, _ = np.linalg.qr(rng.normal(size=(P, P)))  # orthogonal
    shear = np.eye(P)
    shear[0, 1] = 0.8
    lam = 0.02

    def raw(between=None):
        pipe = _pipeline(f, task, N, P, lam, between).fit(Z, y)
        return pipe.predict(Z) if task == "regression" else pipe.decision_function(Z)

    plain = raw()
    tol = 1e-8 if task == "regression" else 1e-6
    assert np.max(np.abs(raw(Q) - plain)) < tol
    assert np.max(np.abs(raw(shear) - plain)) > 1e-3


# ── the shelf: the reference journeys' first two stay first (RECIPES §2.6) ───


def _first_two(situation) -> list[str]:
    """``journeys.first_models``' rule under prediction: the first two predicting families."""
    from turbotab.core.models.base import rank

    return [f.key for f, _ in rank(situation) if f.predicts][:2]


@pytest.mark.parametrize("task", ["regression", "binary"])
def test_ridge_never_displaces_the_first_two_families_the_journeys_pick(task, monkeypatch):
    """RECIPES §2.6: "The two families the reference journeys pick (`first_models`) stay first."
    Every prediction journey is a number or a yes/no outcome, so the reference is the shelf for
    those tasks with ridge taken off, at sizes from 30 to 10,000 rows, narrow and wide; with ridge
    on it, the first two predicting families are the same."""
    import turbotab.core.methods.omics  # noqa: F401 - registers the screened elastic net
    from turbotab.core.models import base

    _family()
    events = (None, None) if task == "regression" else (None, 10)  # 10%: least squares "fair"
    situations = [base.Situation(task=task, purpose="prediction", n_rows=n, n_features=p,
                                 n_events=None if share is None else max(1, n * share // 100))
                  for n in (30, 100, 300, 800, 1_500, 1_999, 2_000, 2_500, 10_000)
                  for p in (5, 40, n, 3 * n) for share in events]
    with_ridge = [_first_two(s) for s in situations]
    # a registry without ridge for the comparison, the original restored after it in its own
    # order (deleting and restoring the key would move ridge to the end of the registration order
    # for every later test in this process)
    monkeypatch.setattr(base, "_REGISTRY", {k: f for k, f in base._REGISTRY.items() if k != "ridge"})
    without = [_first_two(s) for s in situations]
    assert with_ridge == without
    # the survey journey's pick at 300 rows (its capture) is among the shelves compared
    assert ["elastic_net", "screened_elastic_net"] in without


# ── the card's words (RECIPES §6.5) ──────────────────────────────────────────


def test_the_penalty_takes_recipes_card_and_quiet_words():
    """RECIPES §6.5's vocabulary: "ridge and elastic net λ" reads "how strongly coefficients are
    pulled toward zero" on the card and "penalty (λ)" as the quiet term."""
    text = (REPO / "docs" / "turbotab-next" / "RECIPES_AND_TUNING.md").read_text(encoding="utf-8")
    row = re.search(r"^\| ridge and elastic net λ \| (.+?) \| (.+?) \|$", text, re.M)
    assert row is not None
    (dim,) = tuning_for(_family(), "regression").dimensions
    assert (dim.label, dim.term) == (row[1], row[2])


# ── C7: logistic ridge's knob factors ────────────────────────────────────────


def _profiled_hessian(Z: np.ndarray, prob: np.ndarray) -> np.ndarray:
    """The log-likelihood's Hessian in (b, β) at the fitted probabilities, [1, Z]ᵀdiag(p(1 − p))[1, Z],
    with the unpenalized intercept profiled out: the Schur complement of its (b, b) entry."""
    X = np.column_stack([np.ones(len(prob)), Z])
    H = X.T @ ((prob * (1 - prob))[:, None] * X)
    return H[1:, 1:] - np.outer(H[1:, 0], H[0, 1:]) / H[0, 0]


def test_logistic_ridge_knob_is_the_penalized_hessians_shrinkage():
    """MODEL_FAMILY_CONTRACT C7 (logistic ridge, C = 1/(nλ)): the factors use the weighted matrix at
    the fitted probabilities, labeled an approximation. On the fit's own Hessian S (built here) the
    shrink factors are the eigenvalues e of S over e + κ, κ = 1/C, and df = tr[(S + κI)⁻¹S]."""
    f = _family()
    Z, signal = _data(7)
    Zs = (Z - Z.mean(axis=0)) / Z.std(axis=0)
    y = (signal + np.random.default_rng(3).normal(scale=3.0, size=N) > np.median(signal)).astype(int)
    formula = FORMULAS[f.complexity[0].formula]
    for lam in (1e-3, 0.05, 1.0):
        C = estimator_params(f, "binary", {"lambda": lam}, n_units=N, n_rows=N)["C"]
        W, b = _penalized_logistic(Zs, y, lam)
        prob = 1.0 / (1.0 + np.exp(-(Zs @ W[0] + b[0])))
        S = _profiled_hessian(Zs, prob)
        kappa = N * lam
        got = formula(Zs, C=C, probabilities=prob)
        assert got.approximate
        assert got.df == pytest.approx(np.trace(np.linalg.solve(S + kappa * np.eye(P), S)),
                                       abs=1e-10)
        assert np.allclose(got.spread, np.sort(np.linalg.eigvalsh(S))[::-1], rtol=1e-10)
        # scikit-learn's two-column probabilities read the same
        two = formula(Zs, C=C, probabilities=np.column_stack([1 - prob, prob]))
        assert two.df == pytest.approx(got.df, abs=1e-12)
    # least squares stays exact, by alpha as before
    assert not formula(Zs, 3.0).approximate
    with pytest.raises(ValueError, match="probabilities"):
        formula(Zs, C=1.0)
    with pytest.raises(ValueError, match="one of"):
        formula(Zs, 1.0, C=1.0)
    with pytest.raises(ValueError, match="yes/no"):
        formula(Zs, C=1.0, probabilities=np.full((N, 3), 1 / 3))


# ── weighted logistic ridge ──────────────────────────────────────────────────


@pytest.mark.parametrize("task,classes", [("binary", 2), ("multiclass", 3)])
def test_a_weighted_logistic_path_and_refit_are_the_weighted_likelihood_minimum(task, classes):
    """Σᵢ (wᵢ/w̄)ℓᵢ + (nλ/2)‖W‖² by scipy at weights scaled to mean 1; the path at weights summing to
    a population total is the same; the refit with ``sample_weight`` w/w̄ is the path's point."""
    f = _family()
    Z, signal = _data(8)
    Zs = (Z - Z.mean(axis=0)) / Z.std(axis=0)
    noisy = signal + np.random.default_rng(12).normal(scale=4.0, size=N)
    y = np.digitize(noisy, np.quantile(noisy, np.linspace(0, 1, classes + 1)[1:-1]))
    w = np.random.default_rng(6).uniform(0.2, 3.0, size=N)
    relative = w / w.mean()
    lams = np.array([1.0, 0.01, 1e-4])
    fit = f.path(Zs, y, {"lambda": lams}, task=task, weights=w)
    total = f.path(Zs, y, {"lambda": lams}, task=task, weights=w * 25_000.0)
    for g, lam in enumerate(lams):
        W, b = _penalized_logistic(Zs, y, lam, relative)
        assert np.max(np.abs(fit.coefs[0, g] - W)) < 1e-6, lam
        assert np.max(np.abs(fit.intercepts[0, g] - b)) < 1e-6, lam
        assert np.max(np.abs(total.coefs[0, g] - W)) < 1e-6, lam
        est = f.build(task, "prediction", N, P)
        est.set_params(**estimator_params(f, task, {"lambda": lam}, n_units=N, n_rows=N))
        est.fit(Zs, y, sample_weight=relative)
        assert np.max(np.abs(est.coef_ - W)) < 1e-6 and np.max(np.abs(est.intercept_ - b)) < 1e-6


# ── R1 (PP2): an SVD that converges on any finite design ─────────────────────


def _full_coded(seed: int = 0, n: int = 240) -> tuple[np.ndarray, np.ndarray]:
    """The failing NHANES design's shape in small: every level of three categories its own column
    (full coding, so each category's columns sum to one, a dependency per category), three skewed
    numbers and one constant column, standardized as the pipeline's scaler does (a constant
    column's scale is 1, so it is all zero): rank-deficient, as the inner fold whose SVD failed
    (3,382 × 97, ``ridge._thin_svd``)."""
    rng = np.random.default_rng(seed)
    cols = [(rng.integers(0, k, size=n)[:, None] == np.arange(k)[None, :]).astype(float)
            for k in (2, 3, 4)]
    cols += [rng.lognormal(size=(n, 3)), np.ones((n, 1))]
    raw = np.hstack(cols)
    scale = raw.std(axis=0)
    Z = (raw - raw.mean(axis=0)) / np.where(scale > 0, scale, 1.0)
    y = Z @ rng.normal(size=Z.shape[1]) + rng.normal(scale=2.0, size=n) + 3.0
    return Z, y


def _gesdd_fails(monkeypatch) -> list[str]:
    """LAPACK's gesdd failing as Accelerate's did on glucose's inner folds (run_36 and run_37,
    2,720 to 3,787 rows × 96 to 99 columns): every wrapper that asks it for the factors raises,
    numpy's svd and pinv and scipy's svd (gesdd is its default driver) and pinv, while scipy's
    gesvd and the singular values alone still converge, as they did on those folds. A fallback to
    gesdd by any of these routes fails here as it failed there. The list names each refused call."""
    import inspect

    import scipy.linalg

    refused: list[str] = []

    def refuse(name: str, real, when):
        signature = inspect.signature(real)

        def wrapper(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            if when(bound.arguments):
                refused.append(name)
                raise np.linalg.LinAlgError("SVD did not converge")
            return real(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(np.linalg, "svd", refuse(
        "numpy svd", np.linalg.svd, lambda a: a["compute_uv"] and not a["hermitian"]))
    monkeypatch.setattr(np.linalg, "pinv", refuse(
        "numpy pinv", np.linalg.pinv, lambda a: not a["hermitian"]))
    monkeypatch.setattr(scipy.linalg, "svd", refuse(
        "scipy svd", scipy.linalg.svd,
        lambda a: a["compute_uv"] and a["lapack_driver"] == "gesdd"))
    monkeypatch.setattr(scipy.linalg, "pinv", refuse("scipy pinv", scipy.linalg.pinv,
                                                     lambda a: True))
    return refused


@pytest.mark.parametrize("design,weighted", [("correlated", False), ("correlated", True),
                                             ("full_coded", False)])
def test_where_gesdd_fails_the_path_is_still_the_closed_form(design, weighted, monkeypatch):
    """R1: with gesdd refusing through every wrapper (``_gesdd_fails``), every grid point is still
    the normal equations' ridge solved by numpy (weighted at weights of mean 1), and
    scikit-learn's Ridge at α = nλ, to 1e-8: the path falls back to a driver that converges, with
    the same coefficients, rank-deficient or not."""
    from sklearn.linear_model import Ridge as SklearnRidge

    f = _family()
    Z, y = _data(3) if design == "correlated" else _full_coded()
    n = len(y)
    w = np.random.default_rng(7).uniform(0.2, 3.0, size=n) if weighted else None
    lams = 10.0 ** np.linspace(2, -5, 50)
    refused = _gesdd_fails(monkeypatch)
    fit = f.path(Z, y, {"lambda": lams}, task="regression", weights=w)
    assert refused, "the path never asked gesdd"
    relative = None if w is None else w / w.mean()
    for g, lam in enumerate(lams):
        beta, b = _closed_form(Z, y, lam, relative)
        assert np.max(np.abs(fit.coefs[0, g, 0] - beta)) < 1e-8, lam
        assert abs(fit.intercepts[0, g, 0] - b) < 1e-8, lam
        ref = SklearnRidge(alpha=n * lam, solver="cholesky").fit(Z, y, sample_weight=relative)
        assert np.max(np.abs(fit.coefs[0, g, 0] - ref.coef_)) < 1e-8, lam
        assert abs(fit.intercepts[0, g, 0] - ref.intercept_) < 1e-8, lam


@pytest.mark.parametrize("design,weighted", [("full_coded", True), ("wide", False),
                                             ("wide", True)])
def test_where_gesdd_fails_in_every_wrapper_the_path_is_the_augmented_least_squares(
        design, weighted, monkeypatch):
    """R1: with gesdd failing through every wrapper (``_gesdd_fails``), scipy's own default
    included, the path at λ from the grid's top to its bottom is still ridge's augmented least
    squares, [√w(Z − z̄); √(nλ)I]β = [√w(y − ȳ); 0] with b = ȳ − z̄ᵀβ (the means weighted, the
    weights scaled to mean 1), solved by Householder QR with no SVD at all, and scikit-learn's
    Ridge at α = nλ, to 1e-8 of the coefficients' size: rank-deficient and weighted, or with more
    columns than rows. It pins the coefficients, not which driver gave them."""
    from scipy.linalg import solve_triangular
    from sklearn.linear_model import Ridge as SklearnRidge

    f = _family()
    rng = np.random.default_rng(12)
    if design == "full_coded":
        Z, y = _full_coded(2)
    else:
        Z = rng.normal(size=(30, 60)) * rng.uniform(0.2, 5.0, size=60)
        y = Z @ rng.normal(size=60) + rng.normal(size=30) + 4.0
    n, p = Z.shape
    w = rng.uniform(0.2, 3.0, size=n) if weighted else None
    lams = np.array([1e2, 1.0, 1e-2, 1e-4, 1e-5])
    _gesdd_fails(monkeypatch)
    fit = f.path(Z, y, {"lambda": lams}, task="regression", weights=w)
    relative = np.ones(n) if w is None else w / w.mean()
    zbar, ybar = relative @ Z / relative.sum(), float(relative @ y / relative.sum())
    root = np.sqrt(relative)
    for g, lam in enumerate(lams):
        Q, R = np.linalg.qr(np.vstack([root[:, None] * (Z - zbar), np.sqrt(n * lam) * np.eye(p)]))
        beta = solve_triangular(R, Q.T @ np.concatenate([root * (y - ybar), np.zeros(p)]))
        size = max(1.0, float(np.max(np.abs(beta))))
        assert np.max(np.abs(fit.coefs[0, g, 0] - beta)) < 1e-8 * size, lam
        assert abs(fit.intercepts[0, g, 0] - (ybar - zbar @ beta)) < 1e-8 * size, lam
        ref = SklearnRidge(alpha=n * lam, solver="cholesky").fit(Z, y, sample_weight=relative)
        assert np.max(np.abs(fit.coefs[0, g, 0] - ref.coef_)) < 1e-8 * size, lam
        assert abs(fit.intercepts[0, g, 0] - ref.intercept_) < 1e-8 * size, lam


@pytest.mark.parametrize("design", ["correlated", "full_coded"])
def test_where_gesdd_converges_the_path_is_its_own_result(design, monkeypatch):
    """R1, the fallback silent: where gesdd converges, scipy's driver is never asked and the path
    is gesdd's own factors in the closed form, bit for bit (ESL eq. 3.47 on the centered matrix)."""
    import scipy.linalg

    f = _family()
    Z, y = _data(4) if design == "correlated" else _full_coded(1)
    n = len(y)
    lams = 10.0 ** np.linspace(2, -5, 50)
    asked: list[int] = []
    gesvd = scipy.linalg.svd

    def spy(*args, **kwargs):
        asked.append(1)
        return gesvd(*args, **kwargs)

    monkeypatch.setattr(scipy.linalg, "svd", spy)
    fit = f.path(Z, y, {"lambda": lams}, task="regression")
    assert not asked
    # the path's own arithmetic, unit weights included, so the bits can match
    w = np.ones(n)
    zbar, ybar = w @ Z / w.sum(), float(w @ y / w.sum())
    U, d, Vt = np.linalg.svd(w[:, None] * (Z - zbar), full_matrices=False)
    shrink = d[None, :] / (d[None, :] ** 2 + n * lams[:, None])
    coefs = (shrink * (U.T @ (w * (y - ybar)))[None, :]) @ Vt
    assert np.array_equal(fit.coefs[0, :, 0], coefs)
    assert np.array_equal(fit.intercepts[0, :, 0], ybar - coefs @ zbar)


def _not_finite(bad: int, rows: int, cols: int) -> str:
    """The refusal of a design with ``bad`` values that are not finite numbers, word for word."""
    verb = "is not a finite number" if bad == 1 else "are not finite numbers"
    return (rf"^Ridge is not fit: {bad:,} of the {rows * cols:,} values in one inner fold's "
            rf"training rows as the model receives them \({rows:,} rows × {cols:,} columns\) "
            rf"{verb}\.$")


def _not_factored(rows: int, cols: int) -> str:
    """The refusal of a design neither SVD driver factors, word for word."""
    return (rf"^Ridge is not fit: on one inner fold's training rows as the model receives them "
            rf"\({rows:,} rows × {cols:,} columns\), the matrix factorization its penalty path "
            rf"needs did not converge \(an SVD: LAPACK's gesdd, then gesvd\)\. Ridge can be "
            rf"left out of the chosen model families\.$")


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_a_design_that_is_not_finite_is_refused_in_words(bad):
    """R1: a design with a value that is not a finite number is refused, the count and the matrix's
    size named, before any SVD runs (gesdd said only "SVD did not converge"); a finite one fits
    (every test above)."""
    f = _family()
    Z, y = _data(5)
    Z[[3, 9], 1] = bad
    with pytest.raises(ValueError, match=_not_finite(2, N, P)):
        f.path(Z, y, {"lambda": np.array([1.0, 0.01])}, task="regression")
    Z[9, 1] = 0.5
    with pytest.raises(ValueError, match=_not_finite(1, N, P)):
        f.path(Z, y, {"lambda": np.array([1.0, 0.01])}, task="regression")


def test_where_both_drivers_fail_the_path_is_refused_in_words(monkeypatch):
    """R1: where gesvd fails too (LAPACK documents that it can), the path is refused in plain
    words, the matrix's size and both drivers named, with a way forward, never a number; where
    gesvd converges it is silent (the test above where gesdd fails)."""
    import scipy.linalg

    f = _family()
    Z, y = _data(6)
    _gesdd_fails(monkeypatch)

    def fails(*args, **kwargs):
        raise np.linalg.LinAlgError("SVD did not converge")

    monkeypatch.setattr(scipy.linalg, "svd", fails)
    with pytest.raises(np.linalg.LinAlgError, match=_not_factored(N, P)):
        f.path(Z, y, {"lambda": np.array([1.0, 0.01])}, task="regression")


@pytest.mark.parametrize("failure", ["not_finite", "not_factored"])
def test_a_refusal_names_the_inner_folds_rows_the_path_search_fit(failure, monkeypatch):
    """R1 (V1): ridge's path is fit only in the penalty search, on each inner fold's training rows
    as the model receives them (``tuning._path_searched``), so a refusal's sizes are that fold's,
    as it says: here 300 rows, 5 inner folds, so each fold trains on 240, and the count of values
    that are not finite is the fold's own, counted here in the rows the split drew."""
    import warnings

    import scipy.linalg

    from turbotab.core.models import tuning as T
    from turbotab.core.models.inner_cv import fit_pipeline
    from turbotab.core.models.tuning import TunedPipeline, make_plan

    f = _family()
    n, p = 300, 4
    rng = np.random.default_rng(8)
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"x{j}" for j in range(p)],
                     index=pd.Index(np.arange(n) + 7_000, name="row_id"))
    y = X.to_numpy() @ np.array([1.0, -0.5, 0.2, 0.0]) + rng.normal(size=n)
    if failure == "not_finite":
        X.iloc[::15, 1] = np.nan  # 20 rows; the scaler keeps a missing value missing
    else:
        _gesdd_fails(monkeypatch)

        def fails(*args, **kwargs):
            raise np.linalg.LinAlgError("SVD did not converge")

        monkeypatch.setattr(scipy.linalg, "svd", fails)
    plan = make_plan(f, task="regression", loss="mse", n_plan=n, plan_rows=n, unit="units",
                     split_seed=11)
    assert plan is not None and plan.kind == "path" and plan.inner_k == 5
    pipe = TunedPipeline([("scale", StandardScaler()),
                          ("model", f.build("regression", "prediction", n, p))],
                         search=plan).set_output(transform="pandas")
    drawn: list = []
    with pytest.raises((ValueError, np.linalg.LinAlgError)) as caught, \
            T.observing(drawn.append), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit_pipeline(pipe, X, y, seed=11)
    fold = [d for d in drawn if d.kind == "inner"][-1]  # the split whose path was refused
    rows = len(fold.train)
    assert rows == n - n // 5 < n
    if failure == "not_finite":
        bad = int(X.loc[fold.train].isna().to_numpy().sum())
        assert 0 < bad < 20  # some of the 20 sit in the fold's validation rows
        assert caught.type is ValueError and re.match(_not_finite(bad, rows, p), str(caught.value))
    else:
        assert caught.type is np.linalg.LinAlgError
        assert re.match(_not_factored(rows, p), str(caught.value))

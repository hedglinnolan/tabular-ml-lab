"""The path families' search, against references computed here (RECIPES_AND_TUNING §4.3, §8 T2(a)
and T2(d); RT-5f: F4 and F5 fixed).

* **T2(a) · Ridge** on scikit-learn's diabetes data (442 rows, 10 columns): the engine's pooled
  curve over its own inner splits equals numpy's closed form, (ZᵀZ + nλI)⁻¹Zᵀ(y − ȳ) on each
  split's centered and scaled training rows, at every λ of the grid written out here, to 1e-10;
  the choice is the pooled squared error's rounded argmin, index exact; and the scorer is the
  pooled loss, not the folds' mean (the folds here differ in size, and the two curves differ).
* **T2(d) · The elastic net**: the engine's pooled curve equals independent scikit-learn
  ``ElasticNet`` and ``LogisticRegression`` fits over the same ratios and splits, each penalty
  r·λ_max of the split's own training rows, with λ_max from its definition (the largest gradient of
  the loss at zero, the intercepts fit, over ρ); the choice is the rounded argmin; the refit is
  ``ElasticNet``'s (``LogisticRegression``'s) at r·λ_max of the fit's own rows.
* **λ_max is each split's own**, and it is the edge it claims: at λ_max an independent
  ``ElasticNet`` keeps no coefficient, a hair below it keeps one.
* **The logistic penalty is per row** (F5): C = 1/(n·r·λ_max), so a table with every row twice
  is fit to the same coefficients, where the fixed C grid it replaced weakened the penalty per row.
* **The survey-weighted inner loss** (F5, RECIPES §4.3): under the population answer the pooled
  inner loss is Σw·e²/Σw over the validation rows, by hand, the models fit unweighted.
* **The screen is refit in every inner split** (F4): the screened elastic net's screen is fit on
  each inner split's training rows and on the fit's rows, never on rows that score its penalty.
* **The shrinkage path reads the tuning record**: its chosen penalty is the refit's, its penalties
  the chosen mix's ratios times λ_max of the fit's rows, its coefficients at the chosen point the
  fitted model's.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from turbotab.core.models import tuning as T
from turbotab.core.models.base import get_family
from turbotab.core.models.inner_cv import fit_pipeline
from turbotab.core.models.tuning import FitDesign, TunedPipeline, make_plan

# RECIPES §4.1's grids, written out here: 100 ratios of λ_max for a number, 8 for classes, from 1
# down to 10⁻³, log-spaced, the strongest first; the mixes in their order.
RATIOS = np.exp(np.linspace(np.log(1.0), np.log(1e-3), 100))
LOGISTIC_RATIOS = np.exp(np.linspace(np.log(1.0), np.log(1e-3), 8))
MIXES = (0.1, 0.5, 0.7, 0.9, 0.95, 1.0)
LOGISTIC_MIXES = (0.2, 0.6, 1.0)
SEED = 7


def _rounded_argmin(losses) -> int:
    """The lowest loss after rounding each to 10⁻⁹ of the smallest, the first of a tie."""
    v = np.asarray(losses, dtype=float)
    return int(np.argmin(np.round(v / (1e-9 * v.min()))))


def _scaled(train: pd.DataFrame, *others: pd.DataFrame) -> list[np.ndarray]:
    """``train`` and ``others`` centered and scaled by ``train``'s mean and SD (ddof 0)."""
    m = train.to_numpy(dtype=float).mean(axis=0)
    s = train.to_numpy(dtype=float).std(axis=0)
    s[s == 0] = 1.0
    return [(f.to_numpy(dtype=float) - m) / s for f in (train, *others)]


def _lambda_max(Z: np.ndarray, y: np.ndarray, mix: float, classes=None) -> float:
    """The definition: the largest |gradient| of the mean loss at zero coefficients with the
    intercepts fit, over ρ. A number: Zcᵀ(y − ȳ)/n; classes: Zᵀ(Y − Ȳ)/n, Y each class's
    indicator (the second class's alone for two)."""
    n = len(Z)
    Zc = Z - Z.mean(axis=0)
    if classes is None:
        R = (np.asarray(y, dtype=float) - np.mean(y))[:, None]
    else:
        Y = (np.asarray(y)[:, None] == np.asarray(classes)[None, :]).astype(float)
        R = Y[:, 1:] if len(classes) == 2 else Y
        R = R - R.mean(axis=0)
    return float(np.max(np.abs(Zc.T @ R))) / (n * mix)


def _tuned(key: str, task: str, X: pd.DataFrame, y, *, steps=None, **plan_args) -> TunedPipeline:
    family = get_family(key)
    unit = {"regression": "units", "binary": "events"}.get(task, "rarest_class")
    loss = "mse" if task == "regression" else "log_loss"
    plan = make_plan(family, task=task, loss=loss, n_plan=len(y), plan_rows=len(y), unit=unit,
                     split_seed=SEED, **plan_args)
    assert plan is not None and plan.kind == "path", f"{key} is not a path family for {task}"
    model = family.build(task, "prediction", len(y), X.shape[1])
    return TunedPipeline([*(steps or [("scale", StandardScaler())]), ("model", model)],
                         search=plan).set_output(transform="pandas")


def _fit(pipe, X, y, **kw):
    drawn: list[T.Drawn] = []
    with T.observing(drawn.append), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = fit_pipeline(pipe, X, y, seed=SEED, **kw)
    return fitted, [d for d in drawn if d.kind == "inner"]


def _table(n: int, p: int = 6, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Correlated columns, three of them real; row ids that are not positions."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, p))
    X = Z + 0.6 * Z[:, [0]]
    y = X[:, :3] @ np.array([0.6, -0.4, 0.25]) + rng.normal(0, 1.2, n)
    frame = pd.DataFrame(X, columns=[f"x{i}" for i in range(p)],
                         index=pd.Index(np.arange(n) + 5_000, name="row_id"))
    return frame, y


# ── T2(a): ridge on the diabetes data ────────────────────────────────────────


def test_t2a_ridges_path_search_is_the_closed_form_and_the_pooled_argmin():
    from sklearn.datasets import load_diabetes

    data = load_diabetes()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target.astype(float)
    pipe = _tuned("ridge", "regression", X, y)
    assert pipe.search.inner_k == 5
    fitted, inner = _fit(pipe, X, y)
    record = fitted.tuning_
    assert len(inner) == 5 and len({len(d.validation) for d in inner}) > 1  # unequal folds
    lambdas = np.exp(np.linspace(np.log(1e2), np.log(1e-5), 50))  # RECIPES §4.1, strongest first
    sse = np.zeros(len(lambdas))
    by_fold = np.zeros((len(inner), len(lambdas)))
    for f, d in enumerate(inner):
        Zt, Zv = _scaled(X.loc[d.train], X.loc[d.validation])
        yt, yv = y[X.index.get_indexer(d.train)], y[X.index.get_indexer(d.validation)]
        zbar, ybar = Zt.mean(axis=0), yt.mean()
        Zc = Zt - zbar
        for k, lam in enumerate(lambdas):
            beta = np.linalg.solve(Zc.T @ Zc + len(yt) * lam * np.eye(Zt.shape[1]),
                                   Zc.T @ (yt - ybar))
            errors = yv - (ybar + (Zv - zbar) @ beta)
            sse[k] += float(errors @ errors)
            by_fold[f, k] = float(np.mean(errors ** 2))
    pooled = sse / sum(len(d.validation) for d in inner)
    assert np.max(np.abs(np.asarray(record.losses) - pooled) / pooled) <= 1e-10
    assert record.chosen == _rounded_argmin(pooled)
    # the scorer is the pooled loss, which the folds' mean is not on these unequal folds
    assert np.max(np.abs(by_fold.mean(axis=0) - pooled) / pooled) > 1e-6
    Z = _scaled(X)[0]
    assert fitted[-1].alpha == pytest.approx(len(y) * lambdas[record.chosen], rel=1e-12)


# ── T2(d): the elastic net against independent scikit-learn fits ──────────────


def test_t2d_the_least_squares_path_search_is_elastic_nets_on_the_same_ratios_and_splits():
    from sklearn.linear_model import ElasticNet

    X, y = _table(243)  # five unequal inner folds
    pipe = _tuned("elastic_net", "regression", X, y)
    assert pipe.search.inner_k == 5 and len(pipe.search.candidates) == len(MIXES) * len(RATIOS)
    fitted, inner = _fit(pipe, X, y)
    record = fitted.tuning_
    assert record.path.names == ("l1_ratio", "ratio") and len(inner) == 5
    sse = np.zeros((len(MIXES), len(RATIOS)))
    for d in inner:
        Zt, Zv = _scaled(X.loc[d.train], X.loc[d.validation])
        yt, yv = y[X.index.get_indexer(d.train)], y[X.index.get_indexer(d.validation)]
        for m, mix in enumerate(MIXES):
            model = ElasticNet(l1_ratio=mix, tol=1e-13, max_iter=1_000_000, warm_start=True)
            top = _lambda_max(Zt, yt, mix)
            for k, r in enumerate(RATIOS):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.set_params(alpha=r * top).fit(Zt, yt)
                sse[m, k] += float(((yv - model.predict(Zv)) ** 2).sum())
    pooled = (sse / sum(len(d.validation) for d in inner)).ravel()
    assert np.max(np.abs(np.asarray(record.losses) - pooled) / pooled) <= 1e-9
    chosen = _rounded_argmin(pooled)
    assert record.chosen == chosen
    m, k = divmod(chosen, len(RATIOS))
    Z = _scaled(X)[0]
    alpha = RATIOS[k] * _lambda_max(Z, y, MIXES[m])
    refit = fitted[-1]
    assert type(refit).__name__ == "ExactElasticNet" and not hasattr(refit, "alpha_")
    assert refit.l1_ratio == MIXES[m] and refit.alpha == pytest.approx(alpha, rel=1e-12)
    assert record.chosen_params["alpha"] == refit.alpha
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        theirs = ElasticNet(alpha=alpha, l1_ratio=MIXES[m], tol=1e-14,
                            max_iter=1_000_000).fit(Z, y)
    assert np.max(np.abs(fitted.predict(X) - theirs.predict(Z))) <= 1e-9 * float(np.std(y))


@pytest.mark.parametrize("classes", [2, 3], ids=["yes-no", "classes"])
def test_t2d_the_logistic_path_search_is_logistic_regressions_at_c_per_row(classes):
    from sklearn.linear_model import LogisticRegression

    X, signal = _table(240, p=5, seed=classes)
    if classes == 2:
        y = (signal > np.median(signal)).astype(int)
    else:
        y = np.digitize(signal, np.quantile(signal, [1 / 3, 2 / 3]))
    task = "binary" if classes == 2 else "multiclass"
    pipe = _tuned("elastic_net", task, X, y)
    fitted, inner = _fit(pipe, X, y)
    record = fitted.tuning_
    labels = np.arange(classes)
    log_loss = np.zeros((len(LOGISTIC_MIXES), len(LOGISTIC_RATIOS)))

    def saga(C, mix, Z, target):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return LogisticRegression(C=C, l1_ratio=mix, solver="saga", tol=1e-12,
                                      max_iter=1_000_000, random_state=0).fit(Z, target)

    for d in inner:
        Zt, Zv = _scaled(X.loc[d.train], X.loc[d.validation])
        yt, yv = y[X.index.get_indexer(d.train)], y[X.index.get_indexer(d.validation)]
        for m, mix in enumerate(LOGISTIC_MIXES):
            top = _lambda_max(Zt, yt, mix, labels)
            for k, r in enumerate(LOGISTIC_RATIOS):
                fit = saga(1.0 / (len(yt) * r * top), mix, Zt, yt)
                proba = fit.predict_proba(Zv)
                if k == 0:  # at λ_max no coefficient, so the intercepts alone: the training
                    # rows' class shares (saga stops before it has fit them, 10⁻⁸ off)
                    assert np.max(np.abs(fit.coef_)) <= 1e-12
                    proba = np.tile(np.bincount(yt, minlength=classes) / len(yt), (len(yv), 1))
                log_loss[m, k] -= float(np.log(proba[np.arange(len(yv)), yv]).sum())
    pooled = (log_loss / sum(len(d.validation) for d in inner)).ravel()
    assert np.max(np.abs(np.asarray(record.losses) - pooled) / pooled) <= 1e-10
    chosen = _rounded_argmin(pooled)
    assert record.chosen == chosen
    m, k = divmod(chosen, len(LOGISTIC_RATIOS))
    Z = _scaled(X)[0]
    C = 1.0 / (len(y) * LOGISTIC_RATIOS[k] * _lambda_max(Z, y, LOGISTIC_MIXES[m], labels))
    refit = fitted[-1]
    assert type(refit).__name__ == "ExactLogisticRegression" and not hasattr(refit, "C_")
    assert refit.l1_ratio == LOGISTIC_MIXES[m] and refit.C == pytest.approx(C, rel=1e-12)
    want = saga(C, LOGISTIC_MIXES[m], Z, y).predict_proba(Z)
    assert np.max(np.abs(fitted.predict_proba(X) - want)) <= 1e-8


# ── each split's own λ_max, and the edge it is ────────────────────────────────


def test_each_split_reads_its_own_lambda_max_and_it_is_where_every_coefficient_leaves():
    from sklearn.linear_model import ElasticNet

    family = get_family("elastic_net")
    X, y = _table(200)
    Z = _scaled(X)[0]
    grid = {"l1_ratio": np.asarray(MIXES), "ratio": RATIOS}
    for rows in (np.arange(200), np.arange(0, 200, 2), np.arange(60, 200)):
        out = family.path(Z[rows], y[rows], grid, task="regression")
        tops = np.array([_lambda_max(Z[rows], y[rows], mix) for mix in MIXES])
        assert np.allclose(out.values, tops[:, None] * RATIOS[None, :], rtol=1e-12, atol=0)
        assert out.coefs.shape == (len(MIXES), len(RATIOS), 1, Z.shape[1])
        assert not out.coefs[:, 0].any()  # r = 1: every coefficient zero
    first = family.path(Z, y, grid, task="regression").values
    half = family.path(Z[::2], y[::2], grid, task="regression").values
    assert not np.allclose(first, half)  # its own rows' λ_max, not one grid for every split
    for mix in (0.5, 1.0):
        top = _lambda_max(Z, y, mix)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            at = ElasticNet(alpha=top, l1_ratio=mix, tol=1e-14, max_iter=1_000_000).fit(Z, y)
            below = ElasticNet(alpha=top * (1 - 1e-6), l1_ratio=mix, tol=1e-14,
                               max_iter=1_000_000).fit(Z, y)
        assert not at.coef_.any() and below.coef_.any()


def test_the_logistic_penalty_is_per_row_so_every_row_twice_fits_the_same_coefficients():
    family = get_family("elastic_net")
    X, signal = _table(150, p=4, seed=2)
    y = (signal > 0).astype(int)
    Z = _scaled(X)[0]
    values = {"l1_ratio": 0.6, "ratio": 0.05}
    once = family.settings(values, task="binary", n_units=150, n_rows=150, y=y, Z=Z)
    twice_Z, twice_y = np.vstack([Z, Z]), np.concatenate([y, y])
    twice = family.settings(values, task="binary", n_units=300, n_rows=300, y=twice_y, Z=twice_Z)
    by_hand = 1.0 / (150 * 0.05 * _lambda_max(Z, y, 0.6, [0, 1]))
    assert once["C"] == pytest.approx(by_hand, rel=1e-12) and once["l1_ratio"] == 0.6
    assert twice["C"] == pytest.approx(by_hand / 2, rel=1e-12)  # C Σℓ: half as much per row
    a = family.build("binary", "prediction", 150, 4).set_params(**once).fit(Z, y)
    b = family.build("binary", "prediction", 300, 4).set_params(**twice).fit(twice_Z, twice_y)
    assert np.max(np.abs(a.coef_ - b.coef_)) <= 1e-9 * max(1.0, float(np.max(np.abs(a.coef_))))
    regression = family.settings({"l1_ratio": 0.5, "ratio": 0.1}, task="regression", n_units=150,
                                 n_rows=150, y=signal, Z=Z)
    assert regression == {"l1_ratio": 0.5,
                          "alpha": pytest.approx(0.1 * _lambda_max(Z, signal, 0.5), rel=1e-12)}


# ── the survey-weighted inner loss ────────────────────────────────────────────


def test_under_the_population_answer_the_inner_loss_is_weighted_by_hand_and_the_fits_are_not():
    from sklearn.linear_model import ElasticNet

    X, y = _table(240, p=4, seed=5)
    rng = np.random.default_rng(9)
    strata = np.repeat(np.arange(8), 30)
    psu = np.tile(np.repeat([1, 2], 15), 8)
    weights = rng.uniform(0.2, 5.0, len(y))
    design = FitDesign(strata=strata, psu=psu, weights=weights)
    pipe = _tuned("elastic_net", "regression", X, y, psus=16, weighted=True,
                  manual={"l1_ratio": 0.5})
    fitted, inner = _fit(pipe, X, y, design=design)
    record = fitted.tuning_
    assert len(inner) == 5
    w_sse = np.zeros(len(RATIOS))
    sse = np.zeros(len(RATIOS))
    w_sum, n_val = 0.0, 0
    for d in inner:
        Zt, Zv = _scaled(X.loc[d.train], X.loc[d.validation])
        tr, va = X.index.get_indexer(d.train), X.index.get_indexer(d.validation)
        top = _lambda_max(Zt, y[tr], 0.5)  # the fits are unweighted
        for k, r in enumerate(RATIOS):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = ElasticNet(alpha=r * top, l1_ratio=0.5, tol=1e-13,
                                 max_iter=1_000_000).fit(Zt, y[tr])
            e2 = (y[va] - fit.predict(Zv)) ** 2
            w_sse[k] += float(weights[va] @ e2)
            sse[k] += float(e2.sum())
        w_sum += float(weights[va].sum())
        n_val += len(va)
    weighted = w_sse / w_sum
    assert np.max(np.abs(np.asarray(record.losses) - weighted) / weighted) <= 1e-9
    assert np.max(np.abs(weighted - sse / n_val) / weighted) > 1e-3  # the weights matter here
    assert record.chosen == _rounded_argmin(weighted)
    Z = _scaled(X)[0]
    alpha = RATIOS[record.chosen] * _lambda_max(Z, y, 0.5)  # unweighted on every row
    assert fitted[-1].alpha == pytest.approx(alpha, rel=1e-12)


# ── F4: the screen refit in every inner split ─────────────────────────────────


def test_the_screened_elastic_nets_screen_is_refit_on_every_inner_splits_training_rows(monkeypatch):
    from turbotab.core.contracts import contracts
    from turbotab.core.methods import omics

    contracts()  # the omics chain registers the screened elastic net
    rng = np.random.default_rng(12)
    n, p = 120, 60
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"g{i}" for i in range(p)],
                     index=pd.Index(np.arange(n) + 900, name="row_id"))
    y = X.iloc[:, :3].to_numpy() @ np.array([1.0, -0.8, 0.6]) + rng.normal(size=n)
    seen: list[frozenset] = []
    original = omics.UnivariateScreen.fit

    def spy(self, frame, target=None):
        seen.append(frozenset(frame.index.tolist()))
        return original(self, frame, target)

    monkeypatch.setattr(omics.UnivariateScreen, "fit", spy)
    screen = omics.UnivariateScreen(list(X.columns))
    pipe = _tuned("screened_elastic_net", "regression", X, y,
                  steps=[("screen", screen), ("scale", StandardScaler())])
    fitted, inner = _fit(pipe, X, y)
    trains = [frozenset(d.train.tolist()) for d in inner]
    assert len(inner) == 5
    assert seen == [*trains, frozenset(X.index.tolist())]  # each split, then the refit
    for d in inner:
        assert frozenset(d.train.tolist()).isdisjoint(d.validation.tolist())
    assert fitted["screen"].size_ == omics.sis_size(n)


def test_a_screened_net_built_wide_refits_the_screened_matrix_as_its_inner_paths_do():
    """The verifier's finding: production builds the screened net at the table's width (120 rows ×
    2,000 columns: the wide build, which refit in single precision at 10⁻⁴), while every inner
    split's path ran exactly on the ~n/log n columns the screen keeps, so the refit was not the
    path's point. Built at the table's width as the design stage builds it, the refit now reads
    the matrix it is handed: on the screened, scaled columns it equals scikit-learn's own
    float64 ``ElasticNet`` run to 10⁻¹⁴ at the chosen penalty (1e-8 of the largest coefficient),
    and the family's path at that ratio on the same rows (1e-10)."""
    from sklearn.linear_model import ElasticNet

    from turbotab.core.contracts import contracts
    from turbotab.core.methods import omics
    from turbotab.core.models.wide import WideElasticNet

    contracts()
    family = get_family("screened_elastic_net")
    rng = np.random.default_rng(21)
    n, p = 120, 2000
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"g{i}" for i in range(p)],
                     index=pd.Index(np.arange(n) + 900, name="row_id"))
    y = X.iloc[:, :4].to_numpy() @ np.array([1.0, -0.8, 0.6, 0.5]) + rng.normal(size=n)
    built = family.build("regression", "prediction", n, p)
    assert type(built) is WideElasticNet  # the wide build production makes
    plan = make_plan(family, task="regression", loss="mse", n_plan=n, plan_rows=n, unit="units",
                     split_seed=SEED)
    pipe = TunedPipeline([("screen", omics.UnivariateScreen(list(X.columns))),
                          ("scale", StandardScaler()), ("model", built)],
                         search=plan).set_output(transform="pandas")
    fitted, _ = _fit(pipe, X, y)
    model = fitted[-1]
    Z = fitted[:-1].transform(X).to_numpy(dtype=float)
    assert Z.shape[1] == omics.sis_size(n) < 500  # within the exact path's reach
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        theirs = ElasticNet(alpha=model.alpha, l1_ratio=model.l1_ratio, tol=1e-14,
                            max_iter=1_000_000).fit(Z, y)
    scale = float(np.max(np.abs(theirs.coef_)))
    assert scale > 0
    assert np.max(np.abs(model.coef_ - theirs.coef_)) <= 1e-8 * scale
    values = fitted.tuning_.plan.candidates[fitted.tuning_.chosen].values
    point = family.path(Z, y, {"l1_ratio": [values["l1_ratio"]], "ratio": [values["ratio"]]},
                        task="regression")
    assert np.max(np.abs(np.asarray(point.coefs)[0, 0, 0] - model.coef_)) <= 1e-10 * scale


def test_past_the_exact_path_the_refit_runs_the_paths_own_solver_whatever_it_was_built_for():
    """Past the exact path's 500 columns the path runs coordinate descent (``wide.coordinate_path``,
    single precision at 10⁻⁴ while the matrix is wide). The refit runs the same solver on the
    matrix it is handed whichever class it was built as, so the narrow build and the wide build
    give the same coefficients bit for bit, and both are scikit-learn's float64 ``ElasticNet``
    (run to 10⁻¹²) to 10⁻³ of the largest coefficient."""
    from sklearn.linear_model import ElasticNet

    from turbotab.core.models.elastic_net import ELASTIC_NET, ExactElasticNet
    from turbotab.core.models.wide import WideElasticNet

    rng = np.random.default_rng(5)
    n, p = 150, 600
    Z = rng.normal(size=(n, p))
    Z = (Z - Z.mean(axis=0)) / Z.std(axis=0)
    y = Z[:, :5] @ np.array([1.0, -0.7, 0.5, 0.4, -0.3]) + rng.normal(size=n)
    narrow = ELASTIC_NET.build("regression", "prediction", 1000, 50)
    wide = ELASTIC_NET.build("regression", "prediction", n, p)
    assert type(narrow) is ExactElasticNet and type(wide) is WideElasticNet
    alpha = 0.1 * float(np.max(np.abs(Z.T @ (y - y.mean())))) / (n * 0.9)
    fits = [m.set_params(alpha=alpha, l1_ratio=0.9).fit(Z, y) for m in (narrow, wide)]
    assert np.array_equal(fits[0].coef_, fits[1].coef_)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        theirs = ElasticNet(alpha=alpha, l1_ratio=0.9, tol=1e-12, max_iter=100_000).fit(Z, y)
    scale = float(np.max(np.abs(theirs.coef_)))
    assert np.max(np.abs(fits[0].coef_ - theirs.coef_)) <= 1e-3 * scale


# ── the shrinkage path reads the record ──────────────────────────────────────


@pytest.mark.parametrize("task", ["regression", "binary"])
def test_the_shrinkage_path_reads_the_tuning_record(task):
    from sklearn.linear_model import ElasticNet

    from turbotab.core.models import explain as E

    X, signal = _table(200, p=5, seed=4)
    y = signal if task == "regression" else (signal > np.median(signal)).astype(int)
    fitted, _ = _fit(_tuned("elastic_net", task, X, y), X, y)
    record = fitted.tuning_
    anat = E.anatomy(fitted, list(X.columns), task)
    path = E.shrinkage_path(anat, X, y, record=record)
    assert E.shrinkage_path(anat, X, y) is None  # no record, no path
    values = record.plan.candidates[record.chosen].values
    ratios = RATIOS if task == "regression" else LOGISTIC_RATIOS
    k = int(np.argmin(np.abs(ratios - values["ratio"])))
    Z = _scaled(X)[0]
    classes = None if task == "regression" else [0, 1]
    top = _lambda_max(Z, y, values["l1_ratio"], classes)
    refit = fitted[-1]
    assert path.l1_ratio == values["l1_ratio"] and path.penalties[path.chosen_index] == path.chosen
    if task == "regression":
        assert path.penalty_name == "alpha" and path.chosen == refit.alpha
        want = [r * top for r in ratios]
    else:
        assert path.penalty_name == "C" and path.chosen == refit.C
        want = [1.0 / (len(y) * r * top) for r in ratios]
    assert all(any(abs(p - w) <= 1e-12 * w for w in want) for p in path.penalties)
    assert path.penalties == sorted(path.penalties, reverse=task == "regression")
    assert any(abs(p - want[k]) <= 1e-12 * want[k] for p in [path.chosen])
    coef = dict(zip([str(c) for c in X.columns], np.ravel(refit.coef_)))
    for line in path.lines:
        assert line.coefficients[path.chosen_index] == pytest.approx(coef[line.column], abs=1e-10)
    if task == "regression":  # a point that is not the chosen one, by scikit-learn
        j = 0 if path.chosen_index else len(path.penalties) - 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            theirs = ElasticNet(alpha=path.penalties[j], l1_ratio=path.l1_ratio, tol=1e-14,
                                max_iter=1_000_000).fit(Z, y)
        for line in path.lines:
            got = line.coefficients[j]
            assert got == pytest.approx(theirs.coef_[list(X.columns).index(line.column)],
                                        abs=1e-8)

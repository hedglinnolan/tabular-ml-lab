"""The causal lane's estimators (MODELING_SEQUENCE §0 ruling 1, rung (d); V2 definition of done §2,
"causal inference (shortest leash)").

Data-adaptive adjustment with valid inference over an adjustment set chosen from subject knowledge
(the disjunctive cause criterion's answers, ``turbotab/core/estimand.py``). Three estimators, each
written here from its primary source and checked against an independent reference
(``turbotab/core/tests/acceptance/test_causal_lane.py``):

* **Double/debiased machine learning** (Chernozhukov, Chetverikov, Demirer, Duflo, Hansen, Newey &
  Robins 2018, *Econometrics J* 21:C1–C68), as the R package DoubleML 1.0.2 computes it (Bach,
  Chernozhukov, Kurz & Spindler 2024, *J Stat Softw* 108(3)):

  - the **partially linear model** ``Y = θD + g(X) + ε`` for a continuous exposure, with the
    "partialling out" score ``ψ = (Y − ℓ(X) − θ(D − m(X)))(D − m(X))``;
  - the **interactive model** for a yes/no exposure, with the doubly robust scores of the average
    effect (ATE) and of the effect among the exposed (ATT);
  - each nuisance function learned by a flexible learner and **cross-fitted** (K folds: every row's
    nuisance values come from models that never saw it), the score solved over all folds at once
    (``dml2``), the variance ``mean(ψ²)/J²/n`` with ``J = mean(ψ_a)``; repeated over S random
    sample splits and aggregated by the median: ``θ̃ = median θ_s`` and
    ``se = sqrt(median(n·se_s² + (θ_s − θ̃)²)/n)`` (DoubleML's ``agg_cross_fit``).

* **Targeted maximum likelihood** (van der Laan & Rubin 2006, *Int J Biostat* 2(1); Gruber & van der
  Laan 2012, *J Stat Softw* 51(13)) for a yes/no exposure and a numeric or yes/no outcome, as the R
  package tmle 2.1.1 computes it with a given Q and g model (``Qform``, ``gform``,
  ``cvQinit = FALSE``): a numeric outcome mapped to [0, 1] by its range, the initial Q bounded to
  [0.0005, 0.9995] and put on the logit scale, g truncated below at ``5/(√n ln n)`` for each level
  (Gruber, Phillips, Lee & van der Laan 2022, *Am J Epidemiol* 191:1640), one logistic fluctuation
  with the two clever covariates ``(1 − A)/g₀`` and ``A/g₁``, and the influence-curve variance
  ``var(IC)/n``. With a flexible learner, Q and g are cross-fitted on the same sample splits as the
  machine-learning estimators (the cross-validated TMLE's pooled form; Zheng & van der Laan 2011).

* **Post-double-selection lasso** (Belloni, Chernozhukov & Hansen 2014, *Rev Econ Stud*
  81:608–650): a lasso of the outcome on the candidates and a lasso of the exposure on them, each
  with the plug-in penalty ``λ = 2c√n Φ⁻¹(1 − γ/(2p))`` (c = 1.1, γ = 0.1/ln n) and penalty
  loadings iterated from the post-lasso residuals (Belloni, Chen, Chernozhukov & Hansen 2012,
  *Econometrica* 80:2369, Algorithm A.1, as R's hdm ``rlasso`` sets it); then least squares of the
  outcome on the exposure and the union of both selections, with HC3 standard errors (the app's
  heteroskedasticity-robust standard for a linear model; Long & Ervin 2000). "Standard
  post-model selection estimators fail to provide uniform inference even in simple cases with a
  small, fixed number of controls"; double selection gives "confidence intervals that are valid
  uniformly across a large class of models" (BCH 2014, abstract).

**A survey design** (MODELING_SEQUENCE §0 ruling 6): the analysis weights enter every nuisance fit
(``sample_weight``) and the estimating equation (``Σ w_i ψ_i(θ) = 0``); the folds keep each PSU's
rows together; the variance is Taylor linearization of the weighted score's total over strata and
PSUs (:func:`turbotab.core.models.survey.total_variance`, R's ``svyglm`` and ``svymean``).
**Clusters** (a site, a household, a person's repeated rows) are the same computation with equal
weights and the cluster as the PSU. Without either, the variance is DoubleML's and tmle's own.

Nothing here decides whether an estimator may be used: the leash (assumptions declared, positivity,
the design) is :mod:`turbotab.core.causal`'s, and the stage that runs these is
:mod:`turbotab.core.stages.causal`.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Sequence

import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin

Learner = Literal["linear", "lasso", "random_forest", "boosted_trees"]
LEARNERS: tuple[str, ...] = ("linear", "lasso", "random_forest", "boosted_trees")
LEARNER_WORDS: dict[str, str] = {
    "linear": "main-terms linear and logistic regression",
    "lasso": "the cross-validated lasso (L1-penalized linear and logistic regression)",
    "random_forest": "random forests (200 trees, at least 5 rows per leaf)",
    "boosted_trees": "histogram gradient boosting (up to 100 trees)",
}
LEVEL = 0.95
TMLE_ALPHA = 0.9995  # tmle's ``alpha``: Q is bounded to [1 − α, α] on the [0, 1] scale
_NEWTON_TOL = 1e-12
_NEWTON_MAX = 200


def tmle_gbound(n: int) -> float:
    """tmle's default truncation level for g: ``5/(√n ln n)`` (Gruber et al. 2022)."""
    return 5.0 / math.sqrt(n) / math.log(n)


# ── designs and the variance of a weighted score ─────────────────────────────


@dataclass(frozen=True)
class Design:
    """How the rows were drawn, for the variance and the folds.

    ``weight``: the analysis weights (a survey), or None for equal weights. ``stratum`` and ``psu``:
    integer codes per row (a PSU unique across strata), or None. A cluster answer (a site, a
    household, a unit's repeated rows) is a design with equal weights, one stratum and the cluster
    as its PSU. ``df`` is the design's degrees of freedom: PSUs minus strata, counting those that
    hold the analysis rows.

    ``full`` and ``positions``: a survey design over every row of the working table (a
    :class:`turbotab.core.models.survey.SurveyDesign`) and where each analysis row sits in it. The
    analysis rows are then a *domain*: every other row of the design keeps its stratum and PSU in
    the variance with a score of zero (NHANES Analytic Guidelines 2011–2016 §3.2.3.1), as the app's
    survey tables do."""

    weight: np.ndarray | None = None
    stratum: np.ndarray | None = None
    psu: np.ndarray | None = None
    label: str = ""
    full: Any = None
    positions: np.ndarray | None = None

    @property
    def groups(self) -> np.ndarray | None:
        return self.psu

    @property
    def df(self) -> int | None:
        if self.psu is None:
            return None
        strata = 1 if self.stratum is None else len(np.unique(self.stratum))
        return int(len(np.unique(self.psu)) - strata)

    def subset(self, keep: np.ndarray) -> "Design":
        return Design(weight=None if self.weight is None else self.weight[keep],
                      stratum=None if self.stratum is None else self.stratum[keep],
                      psu=None if self.psu is None else self.psu[keep], label=self.label,
                      full=self.full,
                      positions=None if self.positions is None else self.positions[keep])


def _weights(design: Design | None, n: int) -> np.ndarray:
    if design is None or design.weight is None:
        return np.ones(n)
    return np.asarray(design.weight, dtype=float)


def total_variance(u: np.ndarray, design: Design) -> float:
    """The design-based variance of ``Σ_i u_i``: ``Σ_h n_h/(n_h − 1) Σ_j (z_hj − z̄_h)²`` over the
    PSU totals ``z_hj`` within strata (with-replacement PSUs, no finite-population correction; a
    stratum holding one PSU is centered at the grand mean, R's ``survey.lonely.psu = "adjust"``).
    The app's survey tables use the same function (:mod:`turbotab.core.models.survey`); a domain's
    scores are zero on the design's other rows."""
    from turbotab.core.models.survey import SurveyDesign
    from turbotab.core.models.survey import total_variance as design_total

    u = np.asarray(u, dtype=float)
    n = len(u)
    if design.full is not None and design.positions is not None:
        full = design.full
        scores = np.zeros(full.n_rows)
        np.add.at(scores, np.asarray(design.positions, dtype=np.int64), u)
        domain = np.zeros(full.n_rows, dtype=bool)
        domain[np.asarray(design.positions, dtype=np.int64)] = True
        return float(design_total(scores[:, None], full, domain).meat[0, 0])
    psu = (np.arange(n, dtype=np.int64) if design.psu is None
           else np.asarray(design.psu, dtype=np.int64))
    stratum = (np.zeros(n, dtype=np.int64) if design.stratum is None
               else np.asarray(design.stratum, dtype=np.int64))
    psu = np.unique(psu, return_inverse=True)[1].astype(np.int64)
    stratum = np.unique(stratum, return_inverse=True)[1].astype(np.int64)
    survey = SurveyDesign(row_ids=np.arange(n, dtype=np.int64), weight=_weights(design, n),
                          stratum=stratum, psu=psu, weight_column=None, strata_column=None,
                          psu_column=None)
    found = design_total(u[:, None], survey, np.ones(n, dtype=bool))
    return float(found.meat[0, 0])


# ── sample splits ────────────────────────────────────────────────────────────

Split = tuple[np.ndarray, np.ndarray]  # (train rows, test rows)


def sample_splits(n: int, folds: int = 5, repetitions: int = 1, seed: int = 0,
                  groups: np.ndarray | None = None) -> list[list[Split]]:
    """``repetitions`` random partitions of the rows into ``folds`` test folds, reproducible from
    ``seed``. With ``groups`` (PSUs or clusters), whole groups are dealt to folds, so no group's
    rows are on both sides of a split."""
    if folds < 2:
        raise ValueError("Cross-fitting needs at least two folds.")
    rng = np.random.default_rng(seed)
    out: list[list[Split]] = []
    rows = np.arange(n)
    for _ in range(repetitions):
        if groups is None:
            order = rng.permutation(n)
            fold_of = np.empty(n, dtype=np.int64)
            fold_of[order] = np.arange(n) % folds
        else:
            codes = np.unique(np.asarray(groups), return_inverse=True)[1]
            G = int(codes.max()) + 1
            if G < folds:
                raise ValueError(f"Only {G} groups for {folds} folds: each fold needs whole groups.")
            order = rng.permutation(G)
            group_fold = np.empty(G, dtype=np.int64)
            group_fold[order] = np.arange(G) % folds
            fold_of = group_fold[codes]
        out.append([(rows[fold_of != k], rows[fold_of == k]) for k in range(folds)])
    return out


# ── learners ─────────────────────────────────────────────────────────────────


def _design_matrix(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    return np.column_stack([np.ones(len(X)), X])


class _OLS:
    """Least squares with an intercept (R's ``lm``; DoubleML's ``regr.lm``), weighted when asked."""

    def fit(self, X: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> "_OLS":
        A = _design_matrix(X)
        root = np.ones(len(A)) if w is None else np.sqrt(np.asarray(w, dtype=float))
        self.coef_ = np.linalg.lstsq(A * root[:, None], np.asarray(y, float) * root, rcond=None)[0]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return _design_matrix(X) @ self.coef_


def logistic_fit(A: np.ndarray, y: np.ndarray, w: np.ndarray | None = None,
                 offset: np.ndarray | None = None) -> np.ndarray:
    """The maximum-likelihood logistic coefficients of ``y`` (in [0, 1], a share allowed, as R's
    ``glm(family = binomial)`` takes it) on the columns of ``A`` (no intercept added), weighted by
    ``w``, with an ``offset`` on the logit scale. Newton–Raphson with step halving, to a score
    norm below 1e-12 of the weight total."""
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.ones(len(y)) if w is None else np.asarray(w, dtype=float)
    off = np.zeros(len(y)) if offset is None else np.asarray(offset, dtype=float)
    beta = np.zeros(A.shape[1])

    def loglik(b: np.ndarray) -> float:
        eta = off + A @ b
        return float(np.sum(w * (y * eta - np.logaddexp(0.0, eta))))

    current = loglik(beta)
    scale = max(1.0, float(np.sum(w)))
    for _ in range(_NEWTON_MAX):
        mu = 0.5 * (1.0 + np.tanh(0.5 * (off + A @ beta)))
        score = A.T @ (w * (y - mu))
        if float(np.max(np.abs(score))) < _NEWTON_TOL * scale:
            return beta
        info = A.T @ (A * (w * mu * (1.0 - mu))[:, None])
        step = np.linalg.lstsq(info, score, rcond=None)[0]
        for _half in range(60):
            trial = beta + step
            value = loglik(trial)
            if value >= current - 1e-14 * max(1.0, abs(current)):
                break
            step = step / 2.0
        if not np.all(np.isfinite(trial)):
            break
        beta, current = trial, value
    mu = 0.5 * (1.0 + np.tanh(0.5 * (off + A @ beta)))
    if float(np.max(np.abs(A.T @ (w * (y - mu))))) > 1e-6 * scale:
        raise ValueError("The logistic model did not converge: one level is predicted perfectly "
                         "by the covariates (complete separation), so no probability is estimable.")
    return beta


class _Logit:
    """Unpenalized logistic regression with an intercept (R's ``glm``; DoubleML's
    ``classif.log_reg``), weighted when asked."""

    def fit(self, X: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> "_Logit":
        self.coef_ = logistic_fit(_design_matrix(X), y, w)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        eta = _design_matrix(X) @ self.coef_
        return 0.5 * (1.0 + np.tanh(0.5 * eta))


class _Sklearn:
    """A scikit-learn estimator behind the same interface; ``predict`` is a classifier's
    probability of the level coded 1."""

    def __init__(self, estimator: Any, classifier: bool):
        self.estimator = estimator
        self.classifier = classifier

    def fit(self, X: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> "_Sklearn":
        from sklearn.base import clone

        model = clone(self.estimator)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if self.classifier:
            y = y.astype(int)
            if len(np.unique(y)) < 2:
                raise ValueError("A training fold holds one level only; the propensity cannot be "
                                 "learned from it.")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if w is None:
                model.fit(X, y)
            else:
                _fit_weighted(model, X, y, np.asarray(w, dtype=float))
        self.model_ = model
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.classifier:
            proba = self.model_.predict_proba(X)
            classes = list(self.model_.classes_)
            return proba[:, classes.index(1)] if 1 in classes else np.zeros(len(X))
        return np.asarray(self.model_.predict(X), dtype=float)


def _fit_weighted(model: Any, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> None:
    """Fit with sample weights, routed to every step of a pipeline that takes them."""
    from sklearn.pipeline import Pipeline

    if isinstance(model, Pipeline):
        params = {f"{name}__sample_weight": w for name, step in model.steps
                  if "sample_weight" in _fit_parameters(step)}
        model.fit(X, y, **params)
    else:
        model.fit(X, y, sample_weight=w)


def _fit_parameters(step: Any) -> set[str]:
    import inspect

    try:
        return set(inspect.signature(step.fit).parameters)
    except (TypeError, ValueError):
        return set()


class L1LogisticCV(ClassifierMixin, BaseEstimator):
    """The L1-penalized logistic regression with its penalty chosen by 5-fold cross-validated
    deviance over glmnet's path: 100 values from the smallest penalty that zeroes every coefficient
    down to 10⁻⁴ of it (Friedman, Hastie & Tibshirani 2010, *J Stat Softw* 33(1)), as
    ``C = 1/(nλ)``."""

    def __init__(self, seed: int = 0, n_lambdas: int = 100, ratio: float = 1e-4):
        self.seed = seed
        self.n_lambdas = n_lambdas
        self.ratio = ratio

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> "L1LogisticCV":
        from sklearn.linear_model import LogisticRegressionCV
        from sklearn.model_selection import StratifiedKFold

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        yy = (y == np.max(y)).astype(float)
        top = float(np.max(np.abs(X.T @ (w * (yy - np.average(yy, weights=w)))))) / float(np.sum(w))
        top = max(top, 1e-12)
        lams = np.geomspace(top, top * self.ratio, self.n_lambdas)
        model = LogisticRegressionCV(
            Cs=list(1.0 / (len(y) * lams)), l1_ratios=(1.0,), solver="saga",
            scoring="neg_log_loss", cv=StratifiedKFold(5, shuffle=True, random_state=self.seed),
            max_iter=5000, tol=1e-4, random_state=self.seed, use_legacy_attributes=False)
        model.fit(X, y, sample_weight=sample_weight)
        self.model_ = model
        self.classes_ = model.classes_
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        return self.model_.predict_proba(np.asarray(X, dtype=float))

    def predict(self, X: Any) -> np.ndarray:
        return self.model_.predict(np.asarray(X, dtype=float))


def make_learner(name: str, classifier: bool, seed: int = 0) -> Any:
    """A fresh nuisance learner. ``linear`` is deterministic (least squares, logistic regression);
    the others are flexible and seeded by ``seed``."""
    if name == "linear":
        return _Logit() if classifier else _OLS()
    from sklearn.model_selection import KFold, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if name == "lasso":
        if classifier:
            est = make_pipeline(StandardScaler(), L1LogisticCV(seed=seed))
        else:
            from sklearn.linear_model import LassoCV

            est = make_pipeline(StandardScaler(), LassoCV(
                alphas=100, eps=1e-4, cv=KFold(5, shuffle=True, random_state=seed),
                max_iter=50_000, tol=1e-7, random_state=seed))
        return _Sklearn(est, classifier)
    if name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        cls = RandomForestClassifier if classifier else RandomForestRegressor
        return _Sklearn(cls(n_estimators=200, min_samples_leaf=5, random_state=seed, n_jobs=1),
                        classifier)
    if name == "boosted_trees":
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

        cls = HistGradientBoostingClassifier if classifier else HistGradientBoostingRegressor
        return _Sklearn(cls(random_state=seed), classifier)
    raise ValueError(f"Unknown learner {name!r}; the causal lane learns with {', '.join(LEARNERS)}.")


Factory = Callable[[bool, int], Any]  # (classifier, seed) -> learner


def learner_factory(name: str) -> Factory:
    return lambda classifier, seed: make_learner(name, classifier, seed)


def cross_fit(factory: Factory, classifier: bool, X: np.ndarray, y: np.ndarray,
              splits: Sequence[Split], w: np.ndarray | None = None, seed: int = 0,
              train_mask: np.ndarray | None = None,
              predict_at: Callable[[np.ndarray], list[np.ndarray]] | None = None) -> list[np.ndarray]:
    """Out-of-fold predictions: each test fold predicted by a learner fit on its training rows
    (restricted to ``train_mask`` where given, as DoubleML's ``get_cond_samples`` restricts an
    outcome model to one exposure level). ``predict_at(X_test)`` lists the matrices to predict
    each test fold at (default: as observed); one array is returned per matrix."""
    n = len(y)
    outs: list[np.ndarray] | None = None
    for k, (train, test) in enumerate(splits):
        rows = train if train_mask is None else train[train_mask[train]]
        model = factory(classifier, seed + k)
        model.fit(X[rows], y[rows], None if w is None else w[rows])
        at = [X[test]] if predict_at is None else predict_at(X[test])
        if outs is None:
            outs = [np.full(n, np.nan) for _ in at]
        for out, Xt in zip(outs, at):
            out[test] = model.predict(Xt)
    return outs or [np.full(n, np.nan)]


# ── results ──────────────────────────────────────────────────────────────────


@dataclass
class Estimate:
    """One estimate with its uncertainty, as every estimator here reports it."""

    method: str  # dml_plr | dml_irm_ate | dml_irm_att | tmle_ate | pds_lasso
    estimate: float
    se: float
    ci: tuple[float, float]
    p_value: float
    n: int
    df: int | None = None  # the design's degrees of freedom (t intervals), else None (normal)
    repetitions: list[dict[str, float]] = field(default_factory=list)  # per sample split
    extra: dict[str, Any] = field(default_factory=dict)


def _interval(theta: float, se: float, df: int | None) -> tuple[tuple[float, float], float]:
    if df is not None and df > 0:
        q = float(stats.t.ppf(0.5 + LEVEL / 2, df))
        p = float(2 * stats.t.sf(abs(theta / se), df)) if se > 0 else float("nan")
    else:
        q = float(stats.norm.ppf(0.5 + LEVEL / 2))
        p = float(2 * stats.norm.sf(abs(theta / se))) if se > 0 else float("nan")
    return (theta - q * se, theta + q * se), p


def aggregate(thetas: Sequence[float], ses: Sequence[float], n: int) -> tuple[float, float]:
    """DoubleML's aggregation over S sample splits: the median estimate, and
    ``se = sqrt(median(n·se_s² + (θ_s − θ̃)²)/n)``."""
    t = np.asarray(thetas, dtype=float)
    s = np.asarray(ses, dtype=float)
    theta = float(np.median(t))
    se = float(math.sqrt(float(np.median(n * s ** 2 + (t - theta) ** 2)) / n))
    return theta, se


def _solve_score(psi_a: np.ndarray, psi_b: np.ndarray, w: np.ndarray, design: Design | None,
                 n: int) -> tuple[float, float]:
    """θ solving ``Σ w (ψ_a θ + ψ_b) = 0`` and its standard error: DoubleML's ``mean(ψ²)/J²/n``
    without a design, else linearization of the weighted score's total over the design."""
    theta = -float(np.sum(w * psi_b) / np.sum(w * psi_a))
    psi = psi_a * theta + psi_b
    if design is None or (design.weight is None and design.psu is None):
        J = float(np.mean(psi_a))
        var = float(np.mean(psi ** 2)) / J ** 2 / n
    else:
        u = w * psi / float(np.sum(w * psi_a))
        var = total_variance(u, design)
    return theta, math.sqrt(var)


def _finish(method: str, thetas: list[float], ses: list[float], n: int, design: Design | None,
            extra: dict[str, Any]) -> Estimate:
    theta, se = aggregate(thetas, ses, n)
    df = design.df if design is not None and design.psu is not None else None
    ci, p = _interval(theta, se, df)
    return Estimate(method=method, estimate=theta, se=se, ci=ci, p_value=p, n=n, df=df,
                    repetitions=[{"estimate": t, "se": s} for t, s in zip(thetas, ses)],
                    extra=extra)


def _check(y: np.ndarray, d: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    d = np.asarray(d, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if X.shape[1] == 0:
        X = np.zeros((len(y), 1))  # no covariate: every nuisance learns a constant
    if not (len(y) == len(d) == len(X)):
        raise ValueError("The outcome, the exposure and the covariates need one row each.")
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(d)) and np.all(np.isfinite(X))):
        raise ValueError("The causal lane needs complete rows: a value is missing.")
    return y, d, X


def is_binary(values: np.ndarray) -> bool:
    v = np.unique(np.asarray(values, dtype=float))
    return len(v) == 2 and set(v.tolist()) == {0.0, 1.0}


# ── double/debiased machine learning ─────────────────────────────────────────


def dml_plr(y: Any, d: Any, X: Any, *, learner: str | Factory = "linear",
            splits: Sequence[Sequence[Split]], design: Design | None = None,
            outcome_binary: bool = False, seed: int = 0) -> Estimate:
    """The partially linear model's θ (DoubleML's ``DoubleMLPLR``, score "partialling out",
    ``dml2``): ``ℓ(X) = E[Y|X]`` and ``m(X) = E[D|X]`` cross-fitted, then
    ``θ = Σ w (D − m)(Y − ℓ) / Σ w (D − m)²`` per split, aggregated by the median."""
    y, d, X = _check(y, d, X)
    n = len(y)
    w = _weights(design, n)
    factory = learner_factory(learner) if isinstance(learner, str) else learner
    # ℓ for a yes/no outcome: a flexible learner's probability; least squares stays DoubleML's
    # ``regr.lm`` (the linear-probability form).
    l_classifier = bool(outcome_binary and learner != "linear" and isinstance(learner, str))
    d_binary = is_binary(d)
    thetas, ses, r2 = [], [], []
    for s, rep in enumerate(splits):
        [l_hat] = cross_fit(factory, l_classifier, X, y, rep, None if design is None else design.weight,
                            seed + 1000 * s)
        # m for a yes/no exposure: a flexible learner's probability (least squares stays
        # DoubleML's ``regr.lm``).
        m_classifier = bool(d_binary and learner != "linear" and isinstance(learner, str))
        [m_hat] = cross_fit(factory, m_classifier, X, d, rep,
                            None if design is None else design.weight, seed + 1000 * s + 500)
        v = d - m_hat
        u = y - l_hat
        theta, se = _solve_score(-v * v, v * u, w, design, n)
        thetas.append(theta)
        ses.append(se)
        r2.append(1.0 - float(np.sum(w * v ** 2) / np.sum(w * (d - np.average(d, weights=w)) ** 2)))
    extra = {"exposure_r2": float(np.median(r2)), "exposure_binary": d_binary}
    return _finish("dml_plr", thetas, ses, n, design, extra)


def dml_irm(y: Any, d: Any, X: Any, *, learner: str | Factory = "linear",
            splits: Sequence[Sequence[Split]], score: Literal["ATE", "ATT"] = "ATE",
            bound: float = 1e-12, design: Design | None = None, outcome_binary: bool = False,
            seed: int = 0) -> Estimate:
    """The interactive model's average effect (``score = "ATE"``) or the effect among the exposed
    (``"ATT"``) (DoubleML's ``DoubleMLIRM``, ``dml2``, propensities truncated to
    ``[bound, 1 − bound]``). The outcome models ``g₀``, ``g₁`` are fit on the training rows at each
    exposure level; for the ATT, ``p`` is the exposed share of each test fold."""
    y, d, X = _check(y, d, X)
    if not is_binary(d):
        raise ValueError("The interactive model needs a yes/no exposure coded 0 and 1.")
    n = len(y)
    w = _weights(design, n)
    weight = None if design is None else design.weight
    factory = learner_factory(learner) if isinstance(learner, str) else learner
    thetas, ses = [], []
    propensities = []
    for s, rep in enumerate(splits):
        [m_hat] = cross_fit(factory, True, X, d, rep, weight, seed + 1000 * s)
        [g0] = cross_fit(factory, outcome_binary, X, y, rep, weight, seed + 1000 * s + 100,
                         train_mask=(d == 0))
        g1 = None
        if score == "ATE":
            [g1] = cross_fit(factory, outcome_binary, X, y, rep, weight, seed + 1000 * s + 200,
                             train_mask=(d == 1))
        propensities.append(m_hat.copy())
        m = np.clip(m_hat, bound, 1.0 - bound)
        u0 = y - g0
        if score == "ATE":
            psi_b = g1 - g0 + d * (y - g1) / m - (1 - d) * u0 / (1 - m)
            psi_a = -np.ones(n)
        else:
            p = np.empty(n)
            for _, test in rep:
                p[test] = float(np.sum(w[test] * d[test]) / np.sum(w[test]))
            psi_b = d * u0 / p - m * (1 - d) * u0 / (p * (1 - m))
            psi_a = -d / p
        theta, se = _solve_score(psi_a, psi_b, w, design, n)
        thetas.append(theta)
        ses.append(se)
    extra = {"propensity": np.median(np.vstack(propensities), axis=0), "bound": bound,
             "score": score}
    return _finish("dml_irm_ate" if score == "ATE" else "dml_irm_att", thetas, ses, n, design, extra)


# ── targeted maximum likelihood ──────────────────────────────────────────────


def _plogis(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def _qlogis(p: np.ndarray) -> np.ndarray:
    return np.log(p) - np.log1p(-p)


@dataclass
class TMLEParts:
    """What one TMLE fit leaves: the targeted Q, the bounded g, and the influence curves."""

    ate: float
    var_ate: float
    ic_ate: np.ndarray
    mu1: float
    mu0: float
    g1: np.ndarray  # g₁ bounded
    propensity: np.ndarray  # g as estimated (before truncation)
    epsilon: np.ndarray
    rr: float | None = None
    var_log_rr: float | None = None
    odds_ratio: float | None = None
    var_log_or: float | None = None
    ic_log_rr: np.ndarray | None = None
    ic_log_or: np.ndarray | None = None


def _tmle_once(y: np.ndarray, a: np.ndarray, W: np.ndarray, *, family: str, gbound: float,
               w: np.ndarray, Q: np.ndarray | None, g: np.ndarray | None) -> TMLEParts:
    """tmle 2.1.1's point treatment estimate, given the initial ``Q`` (n × 3 on the [0, 1] scale:
    QAW, Q0W, Q1W; None: fit ``glm(Y ~ A + W)``) and ``g`` (P(A = 1 | W); None: fit
    ``glm(A ~ W, binomial)``), with observation weights ``w`` summing to n."""
    n = len(y)
    # .initStage1: map to [0, 1]
    if family == "binomial":
        ab = (0.0, 1.0)
        ystar = np.clip(y, 0.0, 1.0)
    else:
        lo, hi = float(np.min(y)), float(np.max(y))
        lo, hi = lo - 0.01 * abs(lo), hi + 0.01 * abs(hi)
        ystar = np.clip(y, lo, hi)
        ab = (float(np.min(ystar)), float(np.max(ystar)))
        ystar = (ystar - ab[0]) / (ab[1] - ab[0])
    qb = (1.0 - TMLE_ALPHA, TMLE_ALPHA)
    if Q is None:
        A = np.column_stack([np.ones(n), a, W])
        A0 = np.column_stack([np.ones(n), np.zeros(n), W])
        A1 = np.column_stack([np.ones(n), np.ones(n), W])
        if family == "binomial":
            beta = logistic_fit(A, ystar, w)
            Q = np.column_stack([_plogis(A @ beta), _plogis(A0 @ beta), _plogis(A1 @ beta)])
        else:
            root = np.sqrt(w)
            beta = np.linalg.lstsq(A * root[:, None], ystar * root, rcond=None)[0]
            Q = np.column_stack([A @ beta, A0 @ beta, A1 @ beta])
    Q = _qlogis(np.clip(Q, min(qb), max(qb)))
    if g is None:
        G = np.column_stack([np.ones(n), W])
        g = _plogis(G @ logistic_fit(G, a, w))
    g1 = np.clip(g, gbound, 1.0)
    g0 = np.clip(1.0 - g, gbound, 1.0)
    H0 = (1 - a) / g0
    H1 = a / g1
    eps = logistic_fit(np.column_stack([H0, H1]), ystar, w, offset=Q[:, 0])
    Qs = Q + np.column_stack([eps[0] * H0 + eps[1] * H1, eps[0] / g0, eps[1] / g1])
    Qs = _plogis(Qs) * (ab[1] - ab[0]) + ab[0]
    yo = ystar * (ab[1] - ab[0]) + ab[0]
    mu1 = float(np.mean(w * Qs[:, 2]))
    mu0 = float(np.mean(w * Qs[:, 1]))
    ate = mu1 - mu0
    resid = yo - Qs[:, 0]
    ic = w * ((a / g1 - (1 - a) / g0) * resid + Qs[:, 2] - Qs[:, 1] - ate)
    parts = TMLEParts(ate=ate, var_ate=float(np.var(ic, ddof=1)) / n, ic_ate=ic, mu1=mu1, mu0=mu0,
                      g1=g1, propensity=g, epsilon=eps)
    if family == "binomial":
        ic_rr = w * (1 / mu1 * (a / g1 * resid + Qs[:, 2] - mu1)
                     - 1 / mu0 * ((1 - a) / g0 * resid + Qs[:, 1] - mu0))
        ic_or = w * (1 / (mu1 * (1 - mu1)) * (a / g1 * resid + Qs[:, 2])
                     - 1 / (mu0 * (1 - mu0)) * ((1 - a) / g0 * resid + Qs[:, 1]))
        parts.rr = mu1 / mu0
        parts.var_log_rr = float(np.var(ic_rr, ddof=1)) / n
        parts.odds_ratio = mu1 / (1 - mu1) / (mu0 / (1 - mu0))
        parts.var_log_or = float(np.var(ic_or, ddof=1)) / n
        parts.ic_log_rr = ic_rr
        parts.ic_log_or = ic_or
    return parts


def tmle(y: Any, a: Any, W: Any, *, learner: str | Factory = "linear",
         family: Literal["gaussian", "binomial"] = "gaussian", gbound: float | None = None,
         design: Design | None = None, splits: Sequence[Sequence[Split]] | None = None,
         seed: int = 0) -> Estimate:
    """The average effect of a yes/no exposure by targeted maximum likelihood (tmle 2.1.1).

    ``learner = "linear"``: Q is ``glm(Y ~ A + W)`` and g ``glm(A ~ W, binomial)``, each fit on
    every row (tmle with ``Qform``, ``gform`` and ``cvQinit = FALSE``). A flexible learner's Q and
    g are cross-fitted on ``splits`` and the fluctuation is pooled; repeated splits are aggregated
    by the median, as the machine-learning estimators are. Under a design the weights enter Q, g,
    the fluctuation and the means (tmle's ``obsWeights``), and the variance is linearization of
    the influence curve's weighted total over the design."""
    y, a, W = _check(y, a, W)
    if not is_binary(a):
        raise ValueError("TMLE here needs a yes/no exposure coded 0 and 1.")
    if family == "binomial" and not is_binary(y):
        raise ValueError("A yes/no outcome must be coded 0 and 1.")
    n = len(y)
    bound = tmle_gbound(n) if gbound is None else float(gbound)
    w = _weights(design, n)
    w = w / np.sum(w) * n  # tmle: obsWeights/sum(obsWeights) * n
    designed = design is not None and (design.weight is not None or design.psu is not None)

    def finish(parts: TMLEParts) -> tuple[float, float]:
        if designed:
            return parts.ate, math.sqrt(total_variance(parts.ic_ate / n, design))
        return parts.ate, math.sqrt(parts.var_ate)

    if learner == "linear" and splits is None:
        parts = _tmle_once(y, a, W, family=family, gbound=bound, w=w, Q=None, g=None)
        theta, se = finish(parts)
        df = design.df if designed and design.psu is not None else None
        ci, p = _interval(theta, se, df)
        return Estimate(method="tmle_ate", estimate=theta, se=se, ci=ci, p_value=p, n=n, df=df,
                        extra=_tmle_extra(parts, bound, family, designed, design))
    factory = learner_factory(learner) if isinstance(learner, str) else learner
    if splits is None:
        raise ValueError("A flexible learner's Q and g are cross-fitted: give the sample splits.")
    thetas, ses, last = [], [], None
    weight = None if design is None else design.weight
    for s, rep in enumerate(splits):
        AW = np.column_stack([a, W])

        def at_levels(Xt: np.ndarray) -> list[np.ndarray]:
            X0, X1 = Xt.copy(), Xt.copy()
            X0[:, 0], X1[:, 0] = 0.0, 1.0
            return [Xt, X0, X1]

        # Q is learned on the scale _tmle_once maps the outcome to: [0, 1] by its range.
        target = y if family == "binomial" else (y - y.min()) / (y.max() - y.min())
        Q = np.column_stack(cross_fit(factory, family == "binomial", AW, target, rep, weight,
                                      seed + 1000 * s, predict_at=at_levels))
        [g] = cross_fit(factory, True, W, a, rep, weight, seed + 1000 * s + 500)
        parts = _tmle_once(y, a, W, family=family, gbound=bound, w=w, Q=Q, g=g)
        theta, se = finish(parts)
        thetas.append(theta)
        ses.append(se)
        last = parts
    est = _finish("tmle_ate", thetas, ses, n, design if designed else None,
                  _tmle_extra(last, bound, family, designed, design))
    return est


def _tmle_extra(parts: TMLEParts, bound: float, family: str, designed: bool,
                design: Design | None) -> dict[str, Any]:
    extra: dict[str, Any] = {"propensity": parts.propensity, "bound": bound, "mu1": parts.mu1,
                             "mu0": parts.mu0, "epsilon": parts.epsilon.tolist()}
    if family == "binomial" and parts.rr is not None:
        n = len(parts.ic_ate)
        var_rr = (total_variance(parts.ic_log_rr / n, design) if designed else parts.var_log_rr)
        q = float(stats.norm.ppf(0.5 + LEVEL / 2))
        se = math.sqrt(var_rr)
        extra["risk_ratio"] = {"estimate": parts.rr, "se_log": se,
                               "ci": (math.exp(math.log(parts.rr) - q * se),
                                      math.exp(math.log(parts.rr) + q * se))}
        var_or = (total_variance(parts.ic_log_or / n, design) if designed else parts.var_log_or)
        se_or = math.sqrt(var_or)
        extra["odds_ratio"] = {"estimate": parts.odds_ratio, "se_log": se_or,
                               "ci": (math.exp(math.log(parts.odds_ratio) - q * se_or),
                                      math.exp(math.log(parts.odds_ratio) + q * se_or))}
    return extra


# ── post-double-selection lasso ──────────────────────────────────────────────

PDS_C = 1.1
PDS_ITERATIONS = 15
PDS_TOL = 1e-5


def plugin_lambda(n: int, p: int, c: float = PDS_C, gamma: float | None = None) -> float:
    """The plug-in penalty level ``2c√n Φ⁻¹(1 − γ/(2p))``, ``γ = 0.1/ln n`` (hdm's default)."""
    g = 0.1 / math.log(n) if gamma is None else gamma
    return 2.0 * c * math.sqrt(n) * float(stats.norm.ppf(1.0 - g / (2.0 * p)))


@dataclass
class LassoSelection:
    selected: list[int]
    lam: float
    loadings: np.ndarray
    iterations: int


def rigorous_lasso(X: np.ndarray, y: np.ndarray, *, c: float = PDS_C, gamma: float | None = None,
                   iterations: int = PDS_ITERATIONS, tol: float = PDS_TOL) -> LassoSelection:
    """The heteroskedastic plug-in lasso's selection (Belloni et al. 2012, Algorithm A.1; hdm's
    ``rlasso`` with ``post = TRUE``): on centered data, minimize
    ``(1/n)‖y − Xβ‖² + (λ/n) Σ ψ_j|β_j|``; start from ``ψ_j = sqrt(mean(x_j² (y − ȳ)²))``, and
    after each fit reset ``ψ_j = sqrt(mean(x_j² ε̂²))`` from the post-lasso residuals, until the
    loadings move by less than ``tol`` (Euclidean) or ``iterations`` fits. The selection is the last
    fit's support."""
    from sklearn.linear_model import Lasso

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    lam = plugin_lambda(n, p, c, gamma)
    loadings = np.sqrt(np.mean(Xc ** 2 * (yc ** 2)[:, None], axis=0))
    support: list[int] = []
    done = 0
    for done in range(1, iterations + 1):
        usable = loadings > 0
        Z = np.zeros_like(Xc)
        Z[:, usable] = Xc[:, usable] / loadings[usable]
        model = Lasso(alpha=lam / (2.0 * n), fit_intercept=False, tol=1e-12, max_iter=1_000_000,
                      selection="cyclic")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(Z, yc)
        coef = np.where(usable, model.coef_, 0.0)
        support = [int(j) for j in np.flatnonzero(np.abs(coef) > 0)]
        if support:
            B = Xc[:, support]
            beta = np.linalg.lstsq(B, yc, rcond=None)[0]
            resid = yc - B @ beta
        else:
            resid = yc
        new = np.sqrt(np.mean(Xc ** 2 * (resid ** 2)[:, None], axis=0))
        moved = float(np.sqrt(np.sum((new - loadings) ** 2)))
        loadings = new
        if moved < tol:
            break
    return LassoSelection(selected=support, lam=lam, loadings=loadings, iterations=done)


def hc3_ols(A: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares coefficients and their HC3 covariance (MacKinnon & White 1985)."""
    beta = np.linalg.lstsq(A, y, rcond=None)[0]
    resid = y - A @ beta
    bread = np.linalg.pinv(A.T @ A)
    lever = np.einsum("ij,jk,ik->i", A, bread, A)
    scaled = resid / (1.0 - lever)
    meat = A.T @ (A * (scaled ** 2)[:, None])
    return beta, bread @ meat @ bread


def pds_lasso(y: Any, d: Any, X: Any, *, c: float = PDS_C, gamma: float | None = None) -> Estimate:
    """The post-double-selection estimate of the exposure's coefficient (BCH 2014): the union of
    the outcome lasso's and the exposure lasso's selections, then least squares of the outcome on
    the exposure and that union (intercept included), with HC3 standard errors."""
    y, d, X = _check(y, d, X)
    n, p = X.shape
    on_y = rigorous_lasso(X, y, c=c, gamma=gamma)
    on_d = rigorous_lasso(X, d, c=c, gamma=gamma)
    union = sorted(set(on_y.selected) | set(on_d.selected))
    A = np.column_stack([np.ones(n), d, X[:, union]])
    beta, cov = hc3_ols(A, y)
    theta, se = float(beta[1]), float(math.sqrt(cov[1, 1]))
    ci, pval = _interval(theta, se, None)
    return Estimate(method="pds_lasso", estimate=theta, se=se, ci=ci, p_value=pval, n=n,
                    extra={"selected_outcome": on_y.selected, "selected_exposure": on_d.selected,
                           "selected": union, "candidates": p, "lambda": on_y.lam,
                           "iterations": [on_y.iterations, on_d.iterations]})


# ── the diagnostics shown before any estimate ────────────────────────────────

OVERLAP_BINS = 20
# The lane's line for a practical positivity violation: 1% or more of the rows beyond the bound the
# estimator caps propensities at. A few such rows change little (their weights are capped); a
# share this large means the estimate leans on extrapolation for a visible part of the sample.
# A convention of this app, stated wherever it is used; Petersen et al. 2012 (*Stat Methods Med
# Res* 21:31) set no universal threshold.
POSITIVITY_SHARE = 0.01


def kish_ess(weights: np.ndarray) -> float:
    """Kish's effective sample size ``(Σ w)² / Σ w²`` (Kish 1965, *Survey Sampling*)."""
    w = np.asarray(weights, dtype=float)
    if len(w) == 0 or float(np.sum(w ** 2)) == 0:
        return 0.0
    return float(np.sum(w) ** 2 / np.sum(w ** 2))


@dataclass
class Overlap:
    """What the propensity says about positivity, for a yes/no exposure (outcome-free)."""

    bins: list[float]  # OVERLAP_BINS + 1 edges on [0, 1]
    exposed: list[float]  # (weighted) row counts per bin among the exposed
    unexposed: list[float]
    bound: float  # the truncation level b: propensities outside [b, 1 − b] are capped
    n_outside: int  # rows whose propensity lies outside [b, 1 − b]
    share_outside: float
    extreme_weight_share: float  # share of rows whose inverse-probability weight exceeds 1/b
    max_weight: float
    ess_exposed: float
    ess_unexposed: float
    n_exposed: int
    n_unexposed: int
    min_exposed: float
    max_unexposed: float
    violated: bool


def overlap(propensity: np.ndarray, exposure: np.ndarray, bound: float,
            weight: np.ndarray | None = None, target: Literal["ATE", "ATT"] = "ATE") -> Overlap:
    """The overlap of the exposed and unexposed rows' propensities (Austin & Stuart 2015, *Stat
    Med* 34:3661: the propensity distributions, extreme weights and the effective sample size),
    and whether positivity is practically violated: :data:`POSITIVITY_SHARE` or more of the rows
    beyond the bound, for the average effect a propensity outside ``[b, 1 − b]``, for the effect
    among the exposed one above ``1 − b`` (a row with no comparable unexposed one)."""
    g = np.asarray(propensity, dtype=float)
    a = np.asarray(exposure, dtype=float)
    sw = np.ones(len(g)) if weight is None else np.asarray(weight, dtype=float)
    edges = np.linspace(0.0, 1.0, OVERLAP_BINS + 1)
    exposed = np.histogram(g[a == 1], bins=edges, weights=sw[a == 1])[0]
    unexposed = np.histogram(g[a == 0], bins=edges, weights=sw[a == 0])[0]
    outside = (g < bound) | (g > 1 - bound) if target == "ATE" else (g > 1 - bound)
    ipw = np.where(a == 1, 1.0 / np.clip(g, 1e-300, None), 1.0 / np.clip(1 - g, 1e-300, None))
    extreme = ipw > 1.0 / bound
    w1 = (sw * ipw)[a == 1]
    w0 = (sw * ipw)[a == 0]
    return Overlap(
        bins=[float(e) for e in edges], exposed=[float(v) for v in exposed],
        unexposed=[float(v) for v in unexposed], bound=float(bound),
        n_outside=int(outside.sum()), share_outside=float(np.sum(sw * outside) / np.sum(sw)),
        extreme_weight_share=float(np.sum(sw * extreme) / np.sum(sw)),
        max_weight=float(np.max(ipw)) if len(ipw) else 0.0,
        ess_exposed=kish_ess(w1), ess_unexposed=kish_ess(w0),
        n_exposed=int((a == 1).sum()), n_unexposed=int((a == 0).sum()),
        min_exposed=float(g[a == 1].min()) if (a == 1).any() else float("nan"),
        max_unexposed=float(g[a == 0].max()) if (a == 0).any() else float("nan"),
        violated=bool(np.sum(sw * outside) / np.sum(sw) >= POSITIVITY_SHARE))


# A continuous exposure has no propensity to bound; its practical-positivity reading is how much of
# its variance the covariates explain. R² ≥ 0.9 is a variance inflation factor of 10, the
# conventional line past which a coefficient rests on little of its own variation (Kutner,
# Nachtsheim & Neter 2004, *Applied Linear Regression Models* §10.5).
CONTINUOUS_R2_LINE = 0.9


@dataclass
class Variation:
    """What is left of a continuous exposure's variation once the covariates are known."""

    r2: float  # the cross-fitted share of the exposure's variance the covariates explain
    residual_sd: float
    violated: bool


def variation(exposure: np.ndarray, fitted: np.ndarray, weight: np.ndarray | None = None) -> Variation:
    d = np.asarray(exposure, dtype=float)
    w = np.ones(len(d)) if weight is None else np.asarray(weight, dtype=float)
    resid = d - np.asarray(fitted, dtype=float)
    total = float(np.sum(w * (d - np.average(d, weights=w)) ** 2))
    r2 = 1.0 - float(np.sum(w * resid ** 2)) / total if total > 0 else 1.0
    return Variation(r2=r2, residual_sd=float(math.sqrt(np.average(resid ** 2, weights=w))),
                     violated=bool(r2 >= CONTINUOUS_R2_LINE))


def trim_rows(propensity: np.ndarray, at: float) -> np.ndarray:
    """The rows kept by trimming at ``at``: propensity in ``[at, 1 − at]`` (Crump, Hotz, Imbens &
    Mitnik 2009, *Biometrika* 96:187; 0.1 is their rule of thumb)."""
    g = np.asarray(propensity, dtype=float)
    return (g >= at) & (g <= 1 - at)


__all__ = [
    "CONTINUOUS_R2_LINE", "POSITIVITY_SHARE", "Design", "Estimate", "LEARNERS", "LEARNER_WORDS",
    "LassoSelection",
    "Overlap", "Variation", "aggregate", "cross_fit", "dml_irm", "dml_plr", "hc3_ols", "is_binary",
    "kish_ess", "learner_factory", "logistic_fit", "make_learner", "overlap", "pds_lasso",
    "plugin_lambda", "rigorous_lasso", "sample_splits", "tmle", "tmle_gbound", "total_variance",
    "trim_rows", "variation",
]

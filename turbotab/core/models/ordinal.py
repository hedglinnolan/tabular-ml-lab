"""``proportional_odds``: the cumulative-logit (proportional-odds) model of an ordered outcome.

Audit ME-19 ("No ordinal outcome model") and RO-10 (ordered levels read as unordered classes).
CLINICAL_SURVEY_PACK §B6, as the task card quotes it: "For an ordinal outcome, a cumulative link
(proportional odds) model uses the full ordering and handles ties; a linear model on the score, or
a split into responders and non-responders, does not."

**The model** (McCullagh 1980, *J R Stat Soc B* 42:109), in the parameterization of R's
``MASS::polr`` and statsmodels' ``OrderedModel``, for levels 1 < … < K and cut-points
θ₁ < … < θ_{K−1}:

    logit P(Y ≤ j | x) = θ_j − xᵀβ

so a positive β makes higher levels more likely, and exp(β) is the cumulative odds ratio: the
odds of being above any cut-point, per unit of x, the same at every cut-point. It is fit by
Newton–Raphson on the exact log-likelihood (analytic gradient and Hessian; the log-likelihood is
concave in (θ, β) for the logistic link: Pratt 1981, *J Am Stat Assoc* 76:103), with step halving
that keeps the cut-points increasing.

**Intervals.** Independent rows: Wald intervals from the inverse observed information, as
``polr``'s ``summary`` and statsmodels report. Rows that repeat by a unit: the sandwich
``(G/(G − 1)) · I⁻¹ (Σ_g s_g s_gᵀ) I⁻¹`` over the units' summed scores, on t(G − 1), which is
Stata's ``vce(cluster)`` for maximum-likelihood models (Cameron & Miller 2015, *J Hum Resour*
50:317, §VI: "at a minimum one should use the T(G − 1) distribution"); below the unit floor no
interval is reported, as for every family (:func:`~turbotab.core.models.inference.floor_refusal`).

**The proportional-odds check** is Brant's (1990, *Biometrics* 46:1171): the K − 1 binary logistic
regressions of 1{Y > j} on x are fit separately, and a Wald test asks whether their slopes are
equal, using the covariance of the stacked estimates (for j < l,
``Cov(β̂_j, β̂_l) = (XᵀW_jX)⁻¹ XᵀW_{jl}X (XᵀW_lX)⁻¹``, ``W_{jl} = diag(π_l (1 − π_j))``), overall on
(K − 2)·p degrees of freedom and per column on K − 2. Below 0.05 it is a stated concern, with the
columns whose odds ratio changes across cut-points named.

The outcome reaches the family as integer codes 0…K − 1 in the declared order
(:func:`ordinal_outcome`), so every family on the shelf, every metric and the baseline read the
same order; the level names travel with the fitted model (``level_names_``) for the cut-points'
labels.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from turbotab.core.decisions import Purpose, Task
from turbotab.core.models.base import (
    Assessment,
    FamilyBase,
    Situation,
    coefficient_rows,
    register_family,
)

BRANT_ALPHA = 0.05
PER_PARAMETER = 10  # effective rows per predictor, the events-per-variable rule (Peduzzi 1996)


def _expit(z: np.ndarray) -> np.ndarray:
    """The logistic function, exact at ±∞ and without overflow."""
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(z, dtype=float)))


# ── the outcome's order ──────────────────────────────────────────────────────


def ordinal_outcome(y: Any, order: Sequence[Any] | None = None, *, column: str = "the outcome"
                    ) -> tuple[np.ndarray, list[str]]:
    """The outcome as codes 0…K − 1 in its order, and the level names in that order.

    ``order``: the declared order (``set_outcome_order``), lowest first, matched to the values as
    the event level is (:func:`turbotab.core.stages.rows._level_key`). Without one, numbers are
    ordered by value; text cannot be ordered by anything but the user, so it is refused.
    """
    from turbotab.core.stages.rows import _level_key

    values = pd.Series(np.asarray(y, dtype=object))
    if values.isna().any():
        raise ValueError(f"{int(values.isna().sum()):,} rows have no {column}.")
    if order:
        rank = {_level_key(level): i for i, level in enumerate(order)}
        keys = values.map(_level_key)
        unknown = sorted({str(v) for v, k in zip(values, keys) if k not in rank})
        if unknown:
            raise ValueError(f"{', '.join(unknown)} {'is' if len(unknown) == 1 else 'are'} not "
                             f"placed in the declared order of {column}'s levels.")
        return keys.map(rank).to_numpy(dtype=np.int64), [str(level) for level in order]
    numbers = pd.to_numeric(values, errors="coerce")
    if numbers.isna().any():
        raise ValueError(f"`{column}` is text, so the order of its levels must be declared "
                         f"(lowest first) before it can be modeled as ordinal; it is never "
                         f"guessed from the labels.")
    levels = np.unique(numbers.to_numpy(dtype=float))
    names = [str(int(v)) if float(v).is_integer() else f"{v:g}" for v in levels]
    return np.searchsorted(levels, numbers.to_numpy(dtype=float)).astype(np.int64), names


# ── the estimator ────────────────────────────────────────────────────────────


def _parts(beta: np.ndarray, theta: np.ndarray, X: np.ndarray, codes: np.ndarray
           ) -> tuple[np.ndarray, ...]:
    """Per row: log p, ∂ℓ/∂a, ∂ℓ/∂b, ∂²ℓ/∂a², ∂²ℓ/∂b², ∂²ℓ/∂a∂b, at a = θ_y − η, b = θ_{y−1} − η."""
    eta = X @ beta
    cuts = np.concatenate([[-np.inf], theta, [np.inf]])
    a = cuts[codes + 1] - eta
    b = cuts[codes] - eta
    Fa, Fb = _expit(a), _expit(b)
    # P(Y = y) = F(a) − F(b), from the upper tail when both cut-points are above zero.
    p = np.where(b > 0, _expit(-b) - _expit(-a), Fa - Fb)
    p = np.clip(p, 1e-300, None)
    fa, fb = Fa * (1.0 - Fa), Fb * (1.0 - Fb)  # densities: 0 at ±∞
    dfa, dfb = fa * (1.0 - 2.0 * Fa), fb * (1.0 - 2.0 * Fb)
    ga, gb = fa / p, -fb / p
    haa = dfa / p - ga * ga
    hbb = -dfb / p - gb * gb
    hab = -ga * gb
    return np.log(p), ga, gb, haa, hbb, hab


def loglik_gradient_hessian(beta: np.ndarray, theta: np.ndarray, X: np.ndarray, codes: np.ndarray,
                            w: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """The weighted log-likelihood, its gradient and Hessian over (β, θ), and each row's score.

    Parameters are ordered β (p) then θ (K − 1), as ``OrderedModel`` orders them.
    """
    n, P = X.shape
    q = len(theta)
    logp, ga, gb, haa, hbb, hab = _parts(beta, theta, X, codes)
    upper = codes  # the row's upper cut-point is θ_y (none for the top level)
    lower = codes - 1  # its lower cut-point is θ_{y−1} (none for the bottom level)
    has_up = upper < q
    has_low = lower >= 0
    scores = np.zeros((n, P + q))
    scores[:, :P] = -X * (ga + gb)[:, None]
    rows = np.arange(n)
    scores[rows[has_up], P + upper[has_up]] += ga[has_up]
    scores[rows[has_low], P + lower[has_low]] += gb[has_low]
    grad = (w[:, None] * scores).sum(axis=0)
    H = np.zeros((P + q, P + q))
    H[:P, :P] = X.T @ (X * (w * (haa + 2.0 * hab + hbb))[:, None])
    for j in range(q):
        up = upper == j
        low = lower == j
        cross = -(X[up].T @ (w[up] * (haa[up] + hab[up]))) - (X[low].T @ (w[low] * (hab[low] + hbb[low])))
        H[:P, P + j] = H[P + j, :P] = cross
        H[P + j, P + j] = float(np.sum(w[up] * haa[up]) + np.sum(w[low] * hbb[low]))
        if j + 1 < q:
            between = codes == j + 1  # a = θ_{j+1}, b = θ_j
            H[P + j, P + j + 1] = H[P + j + 1, P + j] = float(np.sum(w[between] * hab[between]))
    return float(np.sum(w * logp)), grad, H, scores


class ProportionalOddsRegression(ClassifierMixin, BaseEstimator):
    """The proportional-odds model by Newton–Raphson; scikit-learn's classifier interface.

    ``classes_`` are the sorted outcome values, which the fit stage makes the codes of the
    declared order. Fitted: ``coef_`` (β), ``thresholds_`` (θ), ``loglik_``, ``information_``
    (the observed information over (β, θ)), ``n_iter_``, ``converged_``.
    """

    def __init__(self, max_iter: int = 100, tol: float = 1e-10):
        self.max_iter = max_iter
        self.tol = tol

    def _matrix(self, X: Any) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X.to_numpy(dtype=float, na_value=np.nan)
        return np.asarray(X, dtype=float)

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> "ProportionalOddsRegression":
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray([str(c) for c in X.columns], dtype=object)
        A = self._matrix(X)
        if A.ndim != 2 or not np.all(np.isfinite(A)):
            raise ValueError("The proportional-odds model needs a complete numeric matrix.")
        self.n_features_in_ = A.shape[1]
        self.classes_ = np.unique(np.asarray(y))
        K = len(self.classes_)
        if K < 2:
            raise ValueError("An ordinal outcome needs at least two levels in the fitting rows.")
        codes = np.searchsorted(self.classes_, np.asarray(y)).astype(np.int64)
        w = np.ones(len(codes)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        counts = np.bincount(codes, weights=w, minlength=K)
        cumulative = np.cumsum(counts)[:-1] / counts.sum()
        theta = np.log(cumulative / (1.0 - cumulative))
        beta = np.zeros(A.shape[1])
        P = len(beta)
        ll, grad, H, _ = loglik_gradient_hessian(beta, theta, A, codes, w)
        converged = False
        it = 0
        for it in range(1, int(self.max_iter) + 1):
            info = -H
            try:
                step = np.linalg.solve(info, grad)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(info, grad, rcond=None)[0]
            decrement = float(grad @ step)
            if decrement < float(self.tol):
                converged = True
                break
            size = 1.0
            for _ in range(60):
                b_new, t_new = beta + size * step[:P], theta + size * step[P:]
                if np.all(np.diff(t_new) > 0):
                    trial = loglik_gradient_hessian(b_new, t_new, A, codes, w)
                    if trial[0] >= ll - 1e-12 * max(1.0, abs(ll)):
                        break
                size /= 2.0
            else:
                break
            beta, theta = b_new, t_new
            ll, grad, H, _ = trial
        self.coef_ = beta
        self.thresholds_ = theta
        self.loglik_ = ll
        self.information_ = -H
        self.n_iter_ = it
        self.converged_ = converged
        return self

    def decision_function(self, X: Any) -> np.ndarray:
        """xᵀβ: larger means higher levels are more likely."""
        return self._matrix(X) @ self.coef_

    def predict_proba(self, X: Any) -> np.ndarray:
        eta = self.decision_function(X)
        cumulative = _expit(self.thresholds_[None, :] - eta[:, None])
        n = len(eta)
        full = np.column_stack([np.zeros(n), cumulative, np.ones(n)])
        return np.clip(np.diff(full, axis=1), 0.0, 1.0)

    def predict(self, X: Any) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# ── the proportional-odds check ──────────────────────────────────────────────


def _logistic_fit(X: np.ndarray, z: np.ndarray, max_iter: int = 100) -> tuple[np.ndarray, np.ndarray] | None:
    """Binary logistic regression by Newton (X with its intercept column): β and fitted π."""
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        pi = _expit(X @ beta)
        W = pi * (1.0 - pi)
        info = X.T @ (X * W[:, None])
        grad = X.T @ (z - pi)
        try:
            step = np.linalg.solve(info, grad)
        except np.linalg.LinAlgError:
            return None
        beta = beta + step
        if float(grad @ step) < 1e-12:
            break
    if not np.all(np.isfinite(beta)) or np.abs(beta).max() > 30:
        return None  # separated: the binary fit at this cut-point has no finite estimate
    return beta, _expit(X @ beta)


def brant_test(X: np.ndarray, codes: np.ndarray, names: Sequence[str]) -> dict[str, Any] | None:
    """Brant's Wald test of proportional odds: overall and per column (see the module docstring).

    ``X`` without an intercept; ``codes`` 0…K − 1. None when K < 3 or a cut-point's binary fit
    does not exist.
    """
    from scipy import stats

    K = int(codes.max()) + 1 if len(codes) else 0
    n, p = X.shape
    if K < 3 or p == 0:
        return None
    Xt = np.column_stack([np.ones(n), X])
    fits = []
    for j in range(K - 1):
        fit = _logistic_fit(Xt, (codes > j).astype(float))
        if fit is None:
            return None
        fits.append(fit)
    inverse = [np.linalg.inv(Xt.T @ (Xt * (pi * (1 - pi))[:, None])) for _, pi in fits]
    m = K - 1
    V = np.zeros((m * p, m * p))
    for j in range(m):
        for l in range(j, m):
            if j == l:
                block = inverse[j]
            else:
                cross = fits[l][1] * (1.0 - fits[j][1])
                block = inverse[j] @ (Xt.T @ (Xt * cross[:, None])) @ inverse[l]
            V[j * p:(j + 1) * p, l * p:(l + 1) * p] = block[1:, 1:]
            V[l * p:(l + 1) * p, j * p:(j + 1) * p] = block[1:, 1:].T
    stacked = np.concatenate([b[1:] for b, _ in fits])
    D = np.zeros(((m - 1) * p, m * p))
    for r in range(m - 1):
        D[r * p:(r + 1) * p, :p] = np.eye(p)
        D[r * p:(r + 1) * p, (r + 1) * p:(r + 2) * p] = -np.eye(p)

    def wald(rows: np.ndarray) -> tuple[float, int, float]:
        d = rows @ stacked
        S = rows @ V @ rows.T
        stat = float(d @ np.linalg.pinv(S) @ d)
        df = int(np.linalg.matrix_rank(S))
        return stat, df, float(stats.chi2.sf(stat, df)) if df else float("nan")

    overall = wald(D)
    per = {}
    for k, name in enumerate(names):
        pick = D[[r * p + k for r in range(m - 1)]]
        per[str(name)] = wald(pick)
    return {"statistic": overall[0], "df": overall[1], "p": overall[2],
            "columns": {c: {"statistic": s, "df": d, "p": pv} for c, (s, d, pv) in per.items()},
            "slopes": [b[1:].tolist() for b, _ in fits]}


# ── the inference table ──────────────────────────────────────────────────────


def cut_names(levels: Sequence[Any]) -> list[str]:
    return [f"(cut-point {levels[j]} | {levels[j + 1]})" for j in range(len(levels) - 1)]


def ordinal_table(matrix: pd.DataFrame, y: Any, levels: Sequence[Any] | None, clusters: Any) -> Any:
    """The proportional-odds inference table on the model matrix: coefficients (log cumulative
    odds ratios), then the cut-points; intervals as the module docstring says."""
    from turbotab.core.models.inference import (
        InferenceTable,
        _cluster_concerns,
        _info,
        _refused,
        _t_rows,
        _z_rows,
        floor_refusal,
        format_p,
    )

    X = matrix.to_numpy(dtype=float)
    codes = np.asarray(y).astype(np.int64)
    if clusters.clustered and len(clusters.codes) != len(X):
        raise ValueError(f"The clusters cover {len(clusters.codes):,} rows but the model matrix has "
                         f"{len(X):,}.")
    model = ProportionalOddsRegression().fit(X, codes)
    K = len(model.classes_)
    names_levels = list(levels) if levels is not None and len(levels) == K else \
        [str(c) for c in model.classes_]
    features = [str(c) for c in matrix.columns]
    names = features + cut_names(names_levels)
    est = np.concatenate([model.coef_, model.thresholds_])
    estimator = "proportional-odds (cumulative logit) model, maximum likelihood"
    scale = (f"Coefficients are log cumulative odds ratios: exp(β) multiplies the odds of a higher "
             f"level, per unit, at every cut-point.")
    concerns: list[str] = []
    if not model.converged_:
        concerns.append("The proportional-odds fit stopped before converging; treat these numbers "
                        "with care.")
    refusal = floor_refusal(clusters)
    if refusal:
        table = _refused(names, est, estimator, clusters, *refusal)
        table.concerns.extend(concerns)
        return table
    bread = np.linalg.pinv(model.information_)
    if clusters.clustered:
        _, _, _, scores = loglik_gradient_hessian(model.coef_, model.thresholds_, X, codes,
                                                  np.ones(len(codes)))
        G = clusters.n_clusters
        summed = np.zeros((G, scores.shape[1]))
        np.add.at(summed, clusters.codes, scores)
        V = (G / (G - 1.0)) * bread @ (summed.T @ summed) @ bread
        V = (V + V.T) / 2.0
        se = np.sqrt(np.clip(np.diag(V), 0, None))
        rows = _t_rows(names, est, se, np.full(len(est), float(G - 1)))
        caption = (f"95% intervals cluster-robust by `{clusters.column}` (sandwich over G = {G:,} "
                   f"units with the G/(G − 1) correction, Stata's vce(cluster)), on t({G - 1:,}). "
                   f"{scale}")
        info = _info(estimator, "CR1", caption, clusters)
        concerns = _cluster_concerns(clusters) + concerns
        concerns.append("The proportional-odds check (Brant) is not computed when rows repeat: it "
                        "assumes independent rows.")
        return InferenceTable(rows, info, concerns, cov=V)
    se = np.sqrt(np.clip(np.diag(bread), 0, None))
    rows = _z_rows(names, est, se)
    caption = f"95% Wald intervals from the model's information. {scale}"
    info = _info(estimator, "model", caption, clusters)
    if clusters.note:
        concerns.insert(0, clusters.note)
    brant = brant_test(X, codes, features)
    if brant is None:
        concerns.append("The proportional-odds check (Brant) could not be computed: a binary fit at "
                        "some cut-point does not exist (a column separates it).")
    elif brant["p"] < BRANT_ALPHA:
        off = [c for c, r in brant["columns"].items() if r["p"] < BRANT_ALPHA]
        named = (f"; it fails for {', '.join(f'`{c}`' for c in off)}" if off else "")
        concerns.append(f"The proportional-odds assumption is in doubt (Brant test χ²({brant['df']}) "
                        f"= {brant['statistic']:.1f}, p = {format_p(brant['p'])}){named}: an odds "
                        f"ratio that differs across cut-points is averaged into one. A partial "
                        f"proportional-odds or multinomial model is the usual alternative.")
    info["brant"] = None if brant is None else {
        "statistic": brant["statistic"], "df": brant["df"], "p": brant["p"],
        "columns": {c: {"statistic": r["statistic"], "df": r["df"], "p": r["p"]}
                    for c, r in brant["columns"].items()}}
    return InferenceTable(rows, info, concerns, cov=bread)


def on_cumulative_scale(table: Any, outcome: Any = None, levels: Sequence[Any] | None = None) -> Any:
    """Declare the table's scale as every inference table does (AUDIT_REPORT §5 WP8, ME-07): each
    coefficient is a log cumulative odds ratio, so its row carries exp(β) and exp of its interval's
    ends, drawn on a log axis; the cut-points are thresholds on the log-odds scale and carry none."""
    from turbotab.core.models.inference import _ratio

    name = getattr(outcome, "name", None)
    who = f"`{name}`" if name else "the outcome"
    lowest = f" (lowest `{levels[0]}`)" if levels is not None and len(levels) else ""
    table.info.update(scale="odds_ratio", axis="log", event=None, reference=None,
                      effect=f"Cumulative odds ratio of a higher level of {who}{lowest} rather "
                             f"than a lower one, the same at every cut-point, per unit of each "
                             f"input, holding the others.")
    for row in table.rows:
        if str(row["feature"]).startswith("(cut-point"):
            row.update(ratio=None, ratio_low=None, ratio_high=None)
            continue
        row.update(ratio=_ratio(row.get("estimate")), ratio_low=_ratio(row.get("ci_low")),
                   ratio_high=_ratio(row.get("ci_high")))
    return table


# ── the family ───────────────────────────────────────────────────────────────


def effective_rows(counts: Sequence[int]) -> float:
    """An ordinal outcome's effective sample size, n − Σ n_k³ / n² (Whitehead 1993, as Harrell's
    *Regression Modeling Strategies* §4.4 gives it): n for a continuous outcome, the smaller class
    count's equivalent for two levels."""
    n = float(sum(counts))
    return n - sum(float(c) ** 3 for c in counts) / (n * n) if n else 0.0


class ProportionalOdds(FamilyBase):
    key = "proportional_odds"
    label = "Proportional-odds model"
    tasks: tuple[Task, ...] = ("ordinal",)
    ordered_levels = True  # it models the levels' order (``base.rank`` reads this)
    inductive_bias = ("Each predictor shifts the odds of a higher level by one factor, the same "
                      "at every cut-point.")
    strengths = (
        "Keeps the order of the levels without assuming they are equally spaced.",
        "Coefficients are cumulative odds ratios, with intervals.",
    )
    cautions = (
        "Assumes one odds ratio at every cut-point (proportional odds); the Brant test checks it.",
        "Misses curves and interactions it is not given.",
    )
    needs_scaling = False
    handles_missing = False

    def build(self, task: Task, purpose: Purpose | None, n_rows: int, n_features: int) -> Any:
        return ProportionalOddsRegression()

    def describe(self, task: Task, purpose: Purpose | None) -> tuple[str, str]:
        detail = ("Fits one cumulative log-odds effect per column, the same at every cut-point "
                  "between levels.")
        if purpose == "inference":
            detail += (" Intervals are Wald, or cluster-robust when a unit's rows repeat; the Brant "
                       "test checks the proportional-odds assumption.")
        return "Proportional-odds (cumulative logit) model", detail

    def coefficients(self, pipeline: Any, X: Any, y: Any, *, task: Task,
                     purpose: Purpose | None, groups: Any = None) -> list[dict[str, Any]] | None:
        from turbotab.core.models.linear import as_clusters

        model = pipeline[-1]
        if purpose != "inference":
            features = [str(f) for f in getattr(model, "feature_names_in_", [])]
            levels = getattr(model, "level_names_", None) or [str(c) for c in model.classes_]
            rows = coefficient_rows(features, model.coef_)
            rows += coefficient_rows(cut_names(levels), model.thresholds_)
            return rows
        return self.inference(pipeline, X, y, task=task, clusters=as_clusters(groups)).rows

    def inference(self, pipeline: Any, X: Any, y: Any, *, task: Task, clusters: Any,
                  outcome: Any = None, rows: Any = None) -> Any:
        """The proportional-odds table on the matrix the model saw, on the cumulative odds-ratio
        scale (WP8): ``outcome`` names the outcome, ``rows`` says which rows ``X`` holds."""
        from turbotab.core.models.inference import _on_rows
        from turbotab.core.models.linear import model_matrix

        levels = getattr(pipeline[-1], "level_names_", None)
        table = on_cumulative_scale(ordinal_table(model_matrix(pipeline, X), y, levels, clusters),
                                    outcome, levels)
        return _on_rows(table, len(X), rows)

    def inference_matrix(self, matrix: pd.DataFrame, y: Any, *, task: Task,
                         classes: Sequence[Any] | None, clusters: Any, outcome: Any = None) -> Any:
        """The inference table on a given model matrix (the quintile trend test's refit)."""
        return on_cumulative_scale(ordinal_table(matrix, y, classes, clusters), outcome, classes)

    def assess(self, s: Situation) -> Assessment:
        concerns: list[str] = []
        if s.n_features >= s.n_rows:
            concerns.append(f"{s.n_features:,} predictors for {s.n_rows:,} rows: the model has no "
                            f"unique solution.")
            return Assessment(0.0, "poor", tuple(concerns))
        fit = "good"
        counts = list(getattr(s, "class_counts", None) or [])
        if counts and s.n_features:
            effective = effective_rows(counts)
            per = effective / s.n_features
            if per < PER_PARAMETER:
                concerns.append(f"An effective sample size of {effective:,.0f} (n − Σn³/n², for "
                                f"{len(counts)} levels) for {s.n_features:,} predictors ({per:.1f} "
                                f"each); a common rule of thumb asks for {PER_PARAMETER}.")
                fit = "poor" if per < PER_PARAMETER / 2 else "fair"
            if min(counts) < 5:
                concerns.append(f"The rarest level has {min(counts):,} rows, so its cut-point is "
                                f"barely estimated; merging it with a neighbor is the usual remedy.")
                fit = "fair" if fit == "good" else fit
        score = 3.5 if s.purpose == "inference" else 3.0
        if fit == "fair":
            score -= 1.0
        elif fit == "poor":
            score = 0.5
        return Assessment(score, fit, tuple(concerns))


PROPORTIONAL_ODDS = register_family(ProportionalOdds())

__all__ = ["BRANT_ALPHA", "PROPORTIONAL_ODDS", "ProportionalOdds", "ProportionalOddsRegression",
           "brant_test", "cut_names", "effective_rows", "loglik_gradient_hessian",
           "ordinal_outcome", "ordinal_table"]

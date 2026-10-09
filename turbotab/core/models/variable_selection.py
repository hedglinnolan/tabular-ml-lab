"""The selection menu (MODELING_SEQUENCE §1 row 8; §0 ruling 1; §4 "Selection outside the
resampling"; TRIPOD+AI 9a): which predictors the model keeps, chosen inside the resampling.

**Under prediction every option is a pipeline step** (:class:`Selector`), fitted on each training
fold after outcome-free imputation and construction (MODELING_SEQUENCE §1.1: "construction; filters
and selection; tuning; the fit"), so the cross-validation and the bootstrap repeat it. Ambroise &
McLachlan (PNAS 2002;99:6562): "the same feature-selection method must be implemented in training
the rule on the M − 1 subsets combined at each stage of an (external) cross-validation". The menu,
soundest first:

* **none** — every candidate predictor (a penalized family shrinks on its own);
* **elastic net** — the predictors an elastic net (mixing 0.5, its penalty by inner
  cross-validation) keeps nonzero on the fold;
* **stability selection**, *for a short reproducible panel*, ranked after the elastic net:
  complementary pairs (Shah & Samworth, JRSS B 2013;75:55), each of B random halves and its
  complement running the lasso path to its first q variables; a predictor is kept when its
  selection proportion over the 2B halves is at least τ. With q chosen so that Meinshausen &
  Bühlmann's bound q²/((2τ − 1)p) on the expected number of false selections is one (which Shah &
  Samworth show complementary pairs keep, under the same exchangeability assumption), it "controls
  false selections, not prediction error" (the review's label);
* **in-fold screening** at p ≫ n — the m predictors most correlated with the outcome on the fold
  (m = ⌊n / ln n⌋ by default; Fan & Lv's sure independence screening, JRSS B 2008;70:849);
* **customary options, ranked lower, with their instability stated**: backward elimination by AIC
  (``MASS::stepAIC``'s rule), univariable screens (p < 0.25: Hosmer, Lemeshow & Sturdivant 2013,
  *Applied Logistic Regression* 3rd ed. §4.2), and VIP > 1 from a two-component PLS (Wold's rule).
  Heinze, Wallisch & Dunkler (Biom J 2018;60:431): "Variable selection … can compromise stability of
  a final model, unbiasedness of regression coefficients, and validity of p-values or confidence
  intervals". Their inclusion frequencies across folds and resamples show it (the ``evaluation``
  stage).

**Selection outside the resampling is refused** (``where = "outside"``): a performance estimate
computed after choosing predictors on every row is "a false performance number"; the exit is "run
it in-fold". TRIPOD+AI 9a asks whether any predictor was pre-selected before modeling: a
pre-selection on these rows' outcome is the same false number, refused with the same exit.

**Under inference selection is only a labeled sensitivity analysis** (ruling 1, rungs a and b):
backward elimination by Wald tests among the declared covariates, the exposure kept, at α = 0.157
(the AIC-equivalent level Heinze et al. 2018 recommend for backward elimination), each test pooled
across the imputed copies by Rubin's rules (one coefficient: the pooled estimate over its total
variance with Barnard & Rubin's degrees of freedom; several: Li, Raghunathan & Rubin's D1), as
Wood, White & Royston (Stat Med 2008;27:3227) recommend for selection with multiple imputation.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

PREDICTION_METHODS = ("none", "elastic_net", "stability", "screening", "stepwise", "univariable",
                      "vip")
CUSTOMARY = ("stepwise", "univariable", "vip")
SUPPORTED_TASKS = ("regression", "binary")
UNIVARIABLE_P = 0.25  # Hosmer, Lemeshow & Sturdivant 2013, §4.2
STABILITY_TAU = 0.6  # the lower end of Meinshausen & Bühlmann's (0.6, 0.9)
STABILITY_PAIRS = 50  # Shah & Samworth's B
EXPECTED_FALSE = 1.0  # the error bound q is chosen for
VIP_COMPONENTS = 2
INFERENCE_ALPHA = 0.157  # Heinze et al. 2018: AIC's equivalent significance level
SOURCES = {
    "ambroise": "Ambroise & McLachlan, PNAS 2002;99:6562",
    "heinze": "Heinze, Wallisch & Dunkler, Biom J 2018;60:431",
    "shah": "Shah & Samworth, JRSS B 2013;75:55",
    "fan": "Fan & Lv, JRSS B 2008;70:849",
    "hosmer": "Hosmer, Lemeshow & Sturdivant 2013, Applied Logistic Regression §4.2",
    "wood": "Wood, White & Royston, Stat Med 2008;27:3227",
}
LABELS = {
    "none": "No selection: every candidate predictor",
    "elastic_net": "Elastic net, in each training fold",
    "stability": "Stability selection, for a short reproducible panel",
    "screening": "In-fold screening by correlation with the outcome",
    "stepwise": "Backward elimination by AIC (stepwise)",
    "univariable": "Univariable screen, p < 0.25",
    "vip": "VIP > 1 from PLS",
}
INSTABILITY = ("its selected set changes from sample to sample (Heinze et al. 2018); the inclusion "
               "frequencies across folds and resamples show by how much")


# ── the fold's matrix ────────────────────────────────────────────────────────


def outcome01(task: str, y: Any) -> np.ndarray:
    """The outcome as numbers: as recorded for a numeric outcome; 1 for the second level (sorted as
    text, which puts the coded event 1 after 0) of a yes/no one."""
    values = np.asarray(y)
    if task != "binary":
        return values.astype(float)
    levels = sorted(pd.unique(values).tolist(), key=lambda v: str(v))
    return (values == levels[-1]).astype(float)


def standardized(X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(Z, usable): each column centered and divided by its SD (ddof = 1) on these rows, blanks at
    the column mean (0 after centering); a column with no spread is not usable."""
    values = X.to_numpy(dtype=float)
    mean = np.nanmean(values, axis=0)
    sd = np.nanstd(values, axis=0, ddof=1)
    usable = np.isfinite(sd) & (sd > 0)
    Z = (values - mean) / np.where(usable, sd, 1.0)
    Z = np.where(np.isfinite(Z), Z, 0.0)
    return Z, usable


# ── the methods (module docstring) ───────────────────────────────────────────


def screening_scores(Z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """|Pearson correlation| of each standardized column with the outcome (the point-biserial
    correlation for a yes/no outcome)."""
    yc = y - y.mean()
    denom = math.sqrt(float(yc @ yc)) * np.sqrt((Z * Z).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (Z.T @ yc) / denom
    return np.abs(np.where(np.isfinite(r), r, 0.0))


def screening_keep(n: int, keep: int | None) -> int:
    """How many predictors screening keeps: ``keep``, else ⌊n / ln n⌋ (Fan & Lv 2008)."""
    return int(keep) if keep else max(1, int(n / math.log(max(n, 3))))


def _ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """(β, residual sum of squares, (XᵀX)⁻¹) by least squares."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    rss = float(((y - X @ beta) ** 2).sum())
    return beta, rss, np.linalg.pinv(X.T @ X)


def _loglik(X: np.ndarray, y: np.ndarray) -> float | None:
    from turbotab.core.models.performance import logistic_fit

    fitted = logistic_fit(X, y)
    if fitted is None:
        return None
    eta = X @ fitted[0]
    return float(np.sum(y * eta - np.logaddexp(0.0, eta)))


def univariable_p(Z: np.ndarray, y: np.ndarray, task: str,
                  groups: Sequence[Sequence[int]] | None = None) -> np.ndarray:
    """Each term's univariable p-value (a term: one column, or several, ``groups``): for one column
    the slope's t test of least squares or the Wald z test of a logistic regression, as R's ``lm``
    and ``glm`` report them; for several, the F test of least squares (``anova``) or the
    likelihood-ratio test of a logistic regression (``anova(…, test = "LRT")``)."""
    from scipy import stats

    from turbotab.core.models.performance import logistic_fit

    n = len(y)
    terms = [list(g) for g in groups] if groups is not None else [[j] for j in range(Z.shape[1])]
    out = np.ones(len(terms))
    ones = np.ones((n, 1))
    for t, cols in enumerate(terms):
        X = np.hstack([ones, Z[:, cols]])
        k = len(cols)
        if task == "binary":
            if k == 1:
                fitted = logistic_fit(X, y)
                if fitted is None:
                    continue
                se = math.sqrt(fitted[1][1, 1])
                out[t] = float(2 * stats.norm.sf(abs(fitted[0][1]) / se)) if se > 0 else 1.0
                continue
            full, null = _loglik(X, y), _loglik(ones, y)
            if full is not None and null is not None:
                out[t] = float(stats.chi2.sf(2.0 * (full - null), k))
            continue
        beta, rss, inv = _ols(X, y)
        df = n - k - 1
        if k == 1:
            s2 = rss / df
            se = math.sqrt(s2 * inv[1, 1]) if s2 > 0 else 0.0
            out[t] = float(2 * stats.t.sf(abs(beta[1]) / se, df)) if se > 0 else 0.0
            continue
        rss0 = float(((y - y.mean()) ** 2).sum())
        f = ((rss0 - rss) / k) / (rss / df) if rss > 0 else float("inf")
        out[t] = float(stats.f.sf(f, k, df))
    return out


def aic(Z: np.ndarray, y: np.ndarray, task: str) -> float:
    """``extractAIC`` as ``MASS::stepAIC`` reads it: n·ln(RSS/n) + 2·edf for least squares;
    −2·log-likelihood + 2·edf for a logistic regression. ``Z`` holds the intercept."""
    n, k = Z.shape
    if task == "binary":
        from turbotab.core.models.performance import logistic_fit

        fitted = logistic_fit(Z, y)
        if fitted is None:
            return float("inf")
        eta = Z @ fitted[0]
        loglik = float(np.sum(y * eta - np.logaddexp(0.0, eta)))
        return -2.0 * loglik + 2.0 * k
    _, rss, _ = _ols(Z, y)
    return n * math.log(rss / n) + 2.0 * k if rss > 0 else float("-inf")


def backward_aic(Z: np.ndarray, y: np.ndarray, task: str,
                 groups: Sequence[Sequence[int]] | None = None) -> np.ndarray:
    """Backward elimination by AIC from every term (``MASS::stepAIC(direction = "backward")``; a
    term is one column, or several, ``groups``, dropped together as ``stepAIC`` drops a factor's or
    a spline's columns): drop the term whose removal lowers AIC most, while that lowers it by more
    than 10⁻⁷ (R's ``step`` stops when ``bAIC >= AIC + 1e-07``). Returns the kept mask over terms."""
    n = Z.shape[0]
    terms = [list(g) for g in groups] if groups is not None else [[j] for j in range(Z.shape[1])]
    keep = np.ones(len(terms), dtype=bool)
    ones = np.ones((n, 1))

    def columns(mask: np.ndarray) -> list[int]:
        return [c for t, on in zip(terms, mask) if on for c in t]

    current = aic(np.hstack([ones, Z[:, columns(keep)]]), y, task)
    while keep.any():
        best, best_j = float("inf"), -1
        for j in np.flatnonzero(keep):
            trial = keep.copy()
            trial[j] = False
            value = aic(np.hstack([ones, Z[:, columns(trial)]]), y, task)
            if value < best:
                best, best_j = value, j
        if best >= current - 1e-7:
            break
        keep[best_j] = False
        current = best
    return keep


def pls_vip(Z: np.ndarray, y: np.ndarray, components: int = VIP_COMPONENTS) -> np.ndarray:
    """Variable importance in projection of a PLS1 fit by NIPALS on standardized ``Z`` and the
    centered outcome: VIP_j = √(p · Σ_a SS_a w_aj² / Σ_a SS_a), w_a the unit weight vector and SS_a
    = c_a² t_aᵀt_a the outcome variance component a explains (Wold; Mehmood et al. 2012)."""
    X = Z.copy()
    yy = y - y.mean()
    p = X.shape[1]
    A = max(1, min(int(components), p, X.shape[0] - 1))
    W, SS = [], []
    for _ in range(A):
        w = X.T @ yy
        norm = float(np.linalg.norm(w))
        if norm == 0:
            break
        w = w / norm
        t = X @ w
        tt = float(t @ t)
        if tt == 0:
            break
        load = X.T @ t / tt
        c = float(yy @ t) / tt
        X = X - np.outer(t, load)
        yy = yy - c * t
        W.append(w)
        SS.append(c * c * tt)
    if not W:
        return np.zeros(p)
    W_ = np.asarray(W)
    S_ = np.asarray(SS)
    return np.sqrt(p * (S_[:, None] * W_ ** 2).sum(axis=0) / S_.sum())


def lasso_first(Z: np.ndarray, y: np.ndarray, q: int) -> np.ndarray:
    """The first ``q`` variables to enter the lasso path (least squares on the outcome as numbers;
    the yes/no outcome as 0/1), as Shah & Samworth's base procedure takes them: the active set at
    the first knot of the LARS–lasso path where at least ``q`` are nonzero."""
    from sklearn.linear_model import lars_path

    yc = y - y.mean()
    _, _, coefs = lars_path(Z, yc, method="lasso", max_iter=max(4 * q, 50))
    chosen = np.zeros(Z.shape[1], dtype=bool)
    for j in range(coefs.shape[1]):
        nonzero = coefs[:, j] != 0
        if nonzero.sum() >= q:
            return nonzero
        chosen = nonzero
    return chosen


def stability_q(p: int, tau: float, expected_false: float = EXPECTED_FALSE) -> int:
    """The q that makes Meinshausen & Bühlmann's bound q²/((2τ − 1)p) equal ``expected_false``."""
    return max(1, int(math.floor(math.sqrt(expected_false * (2 * tau - 1) * p))))


def stability_bound(q: int, tau: float, p: int) -> float:
    """E(V) ≤ q² / ((2τ − 1)p): the expected number of false selections (Meinshausen & Bühlmann
    2010, Theorem 1; Shah & Samworth 2013 for complementary pairs)."""
    return q * q / ((2 * tau - 1) * p)


def complementary_pairs(Z: np.ndarray, y: np.ndarray, *, q: int, pairs: int,
                        rng: np.random.Generator) -> np.ndarray:
    """Each column's selection proportion over 2B complementary halves (module docstring). Half b
    is ``rng.permutation(n)[:⌊n/2⌋]`` and its complement the next ⌊n/2⌋ positions."""
    n, p = Z.shape
    half = n // 2
    counts = np.zeros(p)
    for _ in range(int(pairs)):
        order = rng.permutation(n)
        for rows in (order[:half], order[half: 2 * half]):
            counts += lasso_first(Z[rows], y[rows], q)
    return counts / (2.0 * pairs)


def elastic_net_support(Z: np.ndarray, y: np.ndarray, task: str, cv: Any, seed: int) -> np.ndarray:
    """The columns an elastic net (mixing 0.5, its penalty by inner cross-validation) keeps. The
    penalty is chosen as the elastic-net family chooses it (``elastic_net.PooledElasticNetCV`` and
    ``PooledLogisticRegressionCV``: the exact path, the pooled inner loss rounded), with
    coordinate descent at its tolerance on a matrix too wide for the exact path and scikit-learn's
    on a wide one (``wide.WIDE_TOL``). A yes/no outcome's penalty was chosen by accuracy (scikit-
    learn's default score), which is not a proper score; it is the log loss now, as the family's."""
    import warnings

    from turbotab.core.models.elastic_net import (SOLVER_MAX_ITER, SOLVER_TOL, PooledElasticNetCV,
                                                  PooledLogisticRegressionCV)
    from turbotab.core.models.wide import WIDE_TOL, is_wide

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if task == "binary":
            model = PooledLogisticRegressionCV(Cs=10, l1_ratios=[0.5], solver="saga", cv=cv,
                                               scoring="neg_log_loss", max_iter=5000,
                                               random_state=seed, use_legacy_attributes=False)
            model.fit(Z, y.astype(int))
            return np.abs(model.coef_[0]) > 1e-10
        wide = is_wide(*np.shape(Z))
        model = PooledElasticNetCV(l1_ratio=0.5, cv=cv, random_state=seed,
                                   max_iter=10000 if wide else SOLVER_MAX_ITER,
                                   tol=WIDE_TOL if wide else SOLVER_TOL)
        model.fit(Z, y)
        return np.abs(model.coef_) > 1e-10


class Selector(TransformerMixin, BaseEstimator):
    """The selection menu's in-fold step (module docstring). It keeps or drops whole **terms**: a
    term is the columns one input became (a spline's ``x``, ``x'``, ``x''``; a category's
    indicators; :func:`term_groups` over ``inputs``), so no spline loses its linear part and no
    category one of its levels. A method that scores columns (screening, the elastic net,
    stability selection, VIP) keeps a term when it keeps any of its columns; screening counts terms
    against ``keep``; backward elimination and the univariable screen test terms. ``columns``: the
    candidates (None: every column); a column with no spread on the fold is never kept. ``cv`` is
    set by the fit (``models.inner_cv``) to whole-unit or time-ordered inner splits."""

    def __init__(self, method: str = "none", task: str | None = "regression",
                 keep: int | None = None, threshold: float | None = None, q: int | None = None,
                 pairs: int = STABILITY_PAIRS, cv: Any = 5, seed: int = 0,
                 columns: Sequence[str] | None = None, inputs: Sequence[str] | None = None):
        self.method = method
        self.task = task
        self.keep = keep
        self.threshold = threshold
        self.q = q
        self.pairs = pairs
        self.cv = cv
        self.seed = seed
        self.columns = columns
        self.inputs = inputs

    def fit(self, X: pd.DataFrame, y: Any = None) -> "Selector":
        self.feature_names_in_ = np.asarray([str(c) for c in X.columns], dtype=object)
        self.n_features_in_ = X.shape[1]
        names = ([str(c) for c in X.columns] if self.columns is None
                 else [c for c in self.columns if c in X.columns])
        self.terms_ = term_groups(names, list(self.inputs) if self.inputs else names)
        self.candidates_ = list(self.terms_)
        if self.method == "none" or not names:
            self.selected_ = list(self.terms_)
        else:
            if y is None:
                raise ValueError("Selection needs the outcome on the fold.")
            from turbotab.core.methods.levers import infer_task

            self.task_ = infer_task(self.task, y)
            yy = outcome01(self.task_, y)
            Z, usable = standardized(X[names])
            position = {c: i for i, c in enumerate(c for c, u in zip(names, usable) if u)}
            groups = {t: [position[c] for c in cols if c in position]
                      for t, cols in self.terms_.items()}
            groups = {t: g for t, g in groups.items() if g}
            kept = self._kept(Z[:, usable], yy, list(groups.values()))
            self.selected_ = [t for t, k in zip(groups, kept) if k]
        dropped = {c for t, cols in self.terms_.items() if t not in set(self.selected_)
                   for c in cols}
        self.kept_ = [str(c) for c in X.columns if str(c) not in dropped]
        return self

    def _kept(self, Z: np.ndarray, y: np.ndarray, groups: Sequence[Sequence[int]]) -> np.ndarray:
        """The kept mask over ``groups`` (the terms' column positions in ``Z``)."""
        p = Z.shape[1]
        if p == 0 or not groups:
            return np.zeros(len(groups), dtype=bool)
        m = self.method

        def lifted(columns: np.ndarray) -> np.ndarray:
            return np.asarray([bool(np.any(columns[list(g)])) for g in groups])

        if m == "screening":
            scores = screening_scores(Z, y)
            term_scores = np.asarray([float(np.max(scores[list(g)])) for g in groups])
            take = min(screening_keep(len(y), self.keep), len(groups))
            order = np.argsort(-term_scores, kind="stable")[:take]
            out = np.zeros(len(groups), dtype=bool)
            out[order] = True
            return out
        if m == "elastic_net":
            from turbotab.core.methods.levers import inner_folds

            cv = (self.cv if isinstance(self.cv, (list, tuple))
                  else inner_folds(len(y), self.cv, int(self.seed)))
            return lifted(elastic_net_support(Z, y, self.task_, cv, int(self.seed)))
        if m == "stability":
            tau = float(self.threshold or STABILITY_TAU)
            q = int(self.q or stability_q(p, tau))
            self.q_ = min(q, p)
            self.bound_ = stability_bound(self.q_, tau, p)
            freq = complementary_pairs(Z, y, q=self.q_, pairs=int(self.pairs),
                                       rng=np.random.default_rng(int(self.seed)))
            self.frequencies_ = freq
            return lifted(freq >= tau)
        if m == "stepwise":
            if p >= len(y) - 1:
                # The answer is refused where the cohort shows it (``_stepwise_has_rows``); this is
                # the backstop when a later answer added columns, and it names the way forward.
                raise ValueError(
                    f"Backward elimination starts from all {p:,} candidate columns, and these "
                    f"training rows number {len(y):,}: a model with every column and an intercept "
                    f"needs more rows than that. Choose the elastic net or in-fold screening in the "
                    f"selection question.")
            return backward_aic(Z, y, self.task_, groups)
        if m == "univariable":
            return univariable_p(Z, y, self.task_, groups) < float(self.threshold or UNIVARIABLE_P)
        if m == "vip":
            return lifted(pls_vip(Z, y) > 1.0)
        raise ValueError(f"Unknown selection method {m!r}.")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "kept_"):
            raise ValueError("Selector is not fitted yet.")
        keep = set(self.kept_)
        return X.loc[:, [c for c in X.columns if str(c) in keep]]

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(self.kept_, dtype=object)

    def lineage(self) -> list[dict[str, Any]]:
        return [{"output": c, "inputs": [c], "operation": f"kept by {LABELS[self.method].lower()}",
                 "formula": None} for c in self.kept_]


def selection_step(spec: Mapping[str, Any] | None, task: str | None, seed: int = 0,
                   inputs: Sequence[str] | None = None) -> list[tuple[str, Any]]:
    """The in-fold step a ``set_selection`` answer adds under prediction (none for "none", for an
    inference sensitivity analysis, or for selection outside the resampling, which is refused).
    ``inputs``: the predictors as the step receives their names, which name the terms."""
    if not spec or spec.get("sensitivity") or spec.get("where") == "outside":
        return []
    method = spec.get("method") or "none"
    if method == "none":
        return []
    return [("select", Selector(method, task, spec.get("keep"), spec.get("threshold"),
                                spec.get("q"), STABILITY_PAIRS, 5, seed, None,
                                list(inputs) if inputs else None))]


# ── inclusion frequencies (Heinze et al. 2018) ────────────────────────────────


def inclusion_frequencies(pipeline: Any, X: pd.DataFrame, y: Any,
                          pairs: Sequence[tuple[int, np.ndarray, np.ndarray]],
                          fit: Any = None) -> dict[str, Any] | None:
    """How often each candidate was kept over the training folds of ``pairs``: the pipeline up to
    and including its selection step, refit on each fold's training rows. None without one."""
    from sklearn.base import clone

    names = [n for n, _ in pipeline.steps]
    if "select" not in names:
        return None
    head = clone(pipeline[: names.index("select") + 1])
    counts: dict[str, int] = {}
    folds = 0
    y = np.asarray(y)
    for _, fit_rows, _ in pairs:
        part = clone(head)
        rows = np.flatnonzero(fit_rows)
        if fit is not None:
            part = fit(part, X.iloc[rows], y[rows], fit_rows)
        else:
            part.fit(X.iloc[rows], y[rows])
        step = part.steps[-1][1]
        for c in step.candidates_:
            counts.setdefault(c, 0)
        for c in step.selected_:
            counts[c] = counts.get(c, 0) + 1
        folds += 1
    if not folds:
        return None
    rows = sorted(({"column": c, "kept": n, "share": n / folds} for c, n in counts.items()),
                  key=lambda r: (-r["share"], r["column"]))
    return {"folds": folds, "columns": rows}


# ── under inference: a sensitivity analysis by pooled Wald tests ──────────────


def _fit_copy(task: str, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """(β, covariance) of least squares (σ̂²(XᵀX)⁻¹, as ``lm``) or a logistic regression (the inverse
    information, as ``glm``); ``X`` holds the intercept."""
    if task == "binary":
        from turbotab.core.models.performance import logistic_fit

        return logistic_fit(X, y)
    beta, rss, inv = _ols(X, y)
    n, k = X.shape
    return beta, rss / (n - k) * inv


def term_p(task: str, copies: Sequence[tuple[np.ndarray, np.ndarray]], columns: Sequence[int],
           df_com: float) -> tuple[float, float, float | None]:
    """A term's Wald test over the copies: (p, statistic, denominator df). One coefficient: Rubin's
    rules with Barnard & Rubin's df (``imputation.pool_scalar``; one copy: its own t or z test);
    several: D1 (``imputation.pooled_wald``)."""
    from turbotab.core.methods.imputation import pool_scalar, pooled_wald

    fits = [_fit_copy(task, X, y) for X, y in copies]
    if any(f is None for f in fits):
        return float("nan"), float("nan"), None
    idx = list(columns)
    if len(idx) == 1:
        j = idx[0]
        pooled = pool_scalar([f[0][j] for f in fits], [f[1][j, j] for f in fits], df_com)
        stat = pooled.estimate / math.sqrt(pooled.total) if pooled.total > 0 else float("nan")
        return (pooled.p if pooled.p is not None else float("nan")), stat, pooled.df
    Q = np.asarray([f[0][idx] for f in fits])
    U = np.asarray([f[1][np.ix_(idx, idx)] for f in fits])
    if len(fits) == 1:
        from scipy import stats

        q, u = Q[0], U[0]
        stat = float(q @ np.linalg.solve(u, q)) / len(idx)
        dist = stats.f(len(idx), df_com) if task != "binary" else None
        p = float(dist.sf(stat)) if dist is not None else float(stats.chi2.sf(stat * len(idx), len(idx)))
        return p, stat, df_com if task != "binary" else None
    tested = pooled_wald(Q, U, df_com)
    if tested is None:
        return float("nan"), float("nan"), None
    return float(tested["p"]), float(tested["statistic"]), tested.get("df_den")


def backward_wald(task: str, frames: Sequence[pd.DataFrame], outcomes: Sequence[Any],
                  terms: Mapping[str, Sequence[str]], forced: Sequence[str], *,
                  alpha: float = INFERENCE_ALPHA) -> dict[str, Any]:
    """Backward elimination by Wald tests pooled over the imputed copies (module docstring).

    ``frames``: each copy's model matrix (named columns, no intercept); ``outcomes`` its outcome;
    ``terms``: each term's columns; ``forced`` terms are never dropped (the exposure). Each step
    tests every remaining unforced term and drops the one with the largest p-value while it is
    above ``alpha``. Returns the path ``[{step, tested: {term: p}, dropped, p}]`` and the terms
    kept."""
    remaining = [t for t in terms]
    path: list[dict[str, Any]] = []
    ys = [outcome01(task, y) for y in outcomes]
    while True:
        columns = [c for t in remaining for c in terms[t]]
        copies = [(np.column_stack([np.ones(len(f)), f[columns].to_numpy(dtype=float)]), y)
                  for f, y in zip(frames, ys)]
        n, k = copies[0][0].shape
        df_com = float(n - k)
        tested: dict[str, float] = {}
        for t in remaining:
            if t in forced:
                continue
            idx = [1 + columns.index(c) for c in terms[t]]
            tested[t] = term_p(task, copies, idx, df_com)[0]
        candidates = {t: p for t, p in tested.items() if math.isfinite(p)}
        worst = max(candidates, key=lambda t: (candidates[t], t)) if candidates else None
        step = {"step": len(path) + 1, "tested": tested, "dropped": None, "p": None}
        if worst is None or candidates[worst] <= alpha:
            path.append(step)
            break
        step["dropped"], step["p"] = worst, candidates[worst]
        path.append(step)
        remaining.remove(worst)
    return {"path": path, "kept": remaining, "alpha": alpha, "copies": len(frames)}


def term_groups(columns: Sequence[str], inputs: Sequence[str]) -> dict[str, list[str]]:
    """Each model-matrix column grouped under the raw input it came from: the longest input name it
    equals or begins, followed by ``_`` or a prime (``race_2``, ``fiber'``, ``protein_adj``); a
    column no input names is a term of its own."""
    ordered = sorted(inputs, key=len, reverse=True)
    out: dict[str, list[str]] = {}
    for c in columns:
        owner = next((i for i in ordered if c == i or c.startswith(i + "_") or c.startswith(i + "'")),
                     c)
        out.setdefault(owner, []).append(c)
    return out


# ── the contract, the leash and the sentence ─────────────────────────────────


def _register_contract() -> None:
    from turbotab.core.contracts import (CONTRACTS, ContractOption, MethodContract, Relation,
                                        register_contract)

    if "variable_selection" in CONTRACTS:
        return
    here = "turbotab.core.models.variable_selection"

    def option(key: str, customary: str, prediction: str, inference: str,
               rungs: tuple[str, str], order: tuple[int, int]) -> ContractOption:
        return ContractOption(key, LABELS.get(key, key), customary,
                              {"prediction": prediction, "inference": inference},
                              {"prediction": rungs[0], "inference": rungs[1]},
                              {"prediction": order[0], "inference": order[1]})

    not_reported = ("Not offered for the reported model (ruling 1); as a labeled sensitivity "
                    "analysis only")
    register_contract(MethodContract(
        key="variable_selection", label="Selection of predictors", slot="in_fold",
        scope="model", package="EXPLORE", decision="set_selection", stage="design",
        place="8 · Selection", run_order=8.0,
        scope_note=("It reads the outcome on the rows it is fitted on, so it is fitted in each "
                    "training fold, after construction and before the fit."),
        needs=("the candidate predictors", "the outcome on the training fold"),
        question="Which predictors does the model keep, chosen inside the resampling?",
        options=(
            option("none", "Fitting every candidate predictor is common in clinical prediction",
                   "Sound: no selection to correct; a penalized family shrinks instead",
                   "The declared model: the adjustment set is declared, not selected",
                   ("recommended", "recommended"), (0, 0)),
            option("elastic_net", "Penalized selection is common in omics prediction",
                   "Sound in-fold: the penalty is tuned on the fold; the resampling repeats it",
                   not_reported, ("recommended", "not_offered"), (1, 5)),
            option("stability", "Used in omics to discover biomarker panels",
                   "For a short reproducible panel: controls false selections, not prediction "
                   "error; usually predicts no better than the penalized fit it wraps",
                   not_reported, ("available", "not_offered"), (2, 6)),
            option("screening", "Sure-independence and variance screens are customary in genomics",
                   "Sound in-fold at p ≫ n (Ambroise & McLachlan 2002)", not_reported,
                   ("available", "not_offered"), (3, 4)),
            option("stepwise", "Stepwise selection is customary in nutrition and clinical papers",
                   f"Customary, unsound: {INSTABILITY}",
                   "A labeled sensitivity analysis: backward elimination by Wald tests pooled by "
                   "Rubin's rules (Wood et al. 2008)", ("rank_lower", "rank_lower"), (4, 1)),
            option("univariable", "Univariable p-value screens are customary (p < 0.2–0.25)",
                   f"Customary, unsound: {INSTABILITY}", not_reported,
                   ("rank_lower", "not_offered"), (5, 2)),
            option("vip", "VIP > 1 from PLS-DA is customary in metabolomics",
                   f"Customary, unsound: {INSTABILITY}", not_reported,
                   ("rank_lower", "not_offered"), (6, 3)),
        ),
        storyboard=("take the training fold", "standardize the candidates on it",
                    "select on the fold's outcome", "fit the model on the kept predictors",
                    "score the held-out fold"),
        relations=(
            Relation("conflicts", "selection_outside_resampling",
                     "selection on every row before the resampling gives a false performance "
                     "number, so it is refused", purposes=("prediction",), rung="refused",
                     exits=("Run it in-fold",), condition="where = outside, or a pre-selection on "
                     "these rows' outcome (TRIPOD+AI 9a)",
                     enforced_by=f"{here}:_selection_runs_in_fold", id="selection_outside_refused"),
            Relation("conflicts", "too_few_rows_for_every_column",
                     "backward elimination starts from every candidate column and an intercept, "
                     "so it is refused where the smallest training fold has too few rows to fit "
                     "them", purposes=("prediction",), rung="refused", when=("stepwise",),
                     exits=("Elastic net, in each training fold",
                            "In-fold screening by correlation with the outcome", "No selection",
                            "Keep every predictor linear"),
                     condition="candidate columns at least the smallest training fold's rows less "
                               "one", enforced_by=f"{here}:_stepwise_has_rows",
                     id="stepwise_needs_rows"),
            Relation("implies", "inclusion_frequencies",
                     "each candidate's inclusion frequency across the folds and resamples is "
                     "reported", purposes=("prediction",),
                     condition="any selection method",
                     enforced_by=f"{here}:inclusion_frequencies", id="inclusion_frequencies"),
            Relation("conflicts", "selection_as_primary",
                     "under inference selection is not offered for the reported model; it runs "
                     "as a labeled sensitivity analysis only", purposes=("inference",),
                     rung="refused", exits=("Run it as a labeled sensitivity analysis",),
                     condition="a selection answer under inference without sensitivity",
                     enforced_by=f"{here}:_selection_is_sensitivity_under_inference",
                     id="selection_only_sensitivity"),
            Relation("implies", "rubins_rules_wald",
                     "the sensitivity analysis's Wald tests are pooled across the imputed copies "
                     "by Rubin's rules (D1 for several coefficients)", purposes=("inference",),
                     condition="multiple imputation", enforced_by=f"{here}:backward_wald",
                     id="selection_pooled_wald"),
        ),
        short="the selection",
        sources=(SOURCES["ambroise"], SOURCES["heinze"], SOURCES["shah"], SOURCES["fan"],
                 SOURCES["hosmer"], SOURCES["wood"], "TRIPOD+AI item 9a"),
        sentence=f"{here}:decision_sentence"))


_register_contract()


def _ctx_state(ctx: Any) -> Any:
    from turbotab.core.decisions import _state

    return _state(ctx)


def _selection_runs_in_fold(decision: Any, ctx: Any) -> None:
    """MODELING_SEQUENCE §4: "Selection outside the resampling: refuse (a false performance
    number)", with the exit "run it in-fold"; TRIPOD+AI 9a's pre-selection on these rows is the
    same."""
    from turbotab.core.decisions import Refusal, SetSelection

    state = _ctx_state(ctx)
    if state is not None and getattr(state, "purpose", None) == "inference":
        return
    method = decision.method if decision.method != "none" else "screening"
    in_fold = SetSelection(method=method, where="in_fold", keep=decision.keep,
                           threshold=decision.threshold, q=decision.q,
                           pre_selected=("no" if decision.pre_selected == "yes"
                                         else decision.pre_selected))
    if decision.where == "outside":
        raise Refusal(
            "selection_outside_resampling",
            "Choosing predictors on every row before the resampling gives a false performance "
            "number: the scores would not include the optimism of the choice (Ambroise & McLachlan "
            "2002). Run it in-fold, so every training fold repeats it.",
            exits=[{"label": "Run it in-fold", "decision": in_fold}])
    if decision.pre_selected == "yes":
        raise Refusal(
            "pre_selected_on_these_rows",
            "Predictors chosen by their association with the outcome in these rows before this "
            "analysis are a selection outside the resampling: no score here could include its "
            "optimism (TRIPOD+AI 9a). Run it in-fold: start from every candidate predictor and "
            "choose here.",
            exits=[{"label": "Run it in-fold", "decision": in_fold},
                   {"label": "The pre-selection used other data",
                    "decision": decision.model_copy(update={"pre_selected": "other_data"})}])


def _selection_is_sensitivity_under_inference(decision: Any, ctx: Any) -> None:
    """Ruling 1 (a, b): under inference selection is not offered for the reported model; backward
    elimination by Wald tests runs as a labeled sensitivity analysis only."""
    from turbotab.core.decisions import Refusal

    state = _ctx_state(ctx)
    if state is None or getattr(state, "purpose", None) != "inference" or decision.method == "none":
        return
    exit_ = {"label": "Run it as a labeled sensitivity analysis",
             "decision": decision.model_copy(update={"method": "stepwise", "sensitivity": True,
                                                     "where": "in_fold"})}
    if not decision.sensitivity:
        raise Refusal(
            "selection_not_for_inference",
            "Under inference the model is declared, not selected from the rows it reports on "
            "(choosing by significance is refused; stepwise selection is customary but unsound "
            "as the primary model). Backward elimination can run as a labeled sensitivity analysis "
            "beside the declared model.", exits=[exit_])
    if decision.method != "stepwise":
        raise Refusal(
            "sensitivity_is_backward_wald",
            "Under inference the selection sensitivity analysis is backward elimination by Wald "
            "tests, pooled across imputed copies by Rubin's rules (Wood et al. 2008).",
            exits=[exit_])


def _selection_fits_the_task(decision: Any, ctx: Any) -> None:
    """The selection methods here read a numeric or yes/no outcome; stepwise needs more rows than
    predictors."""
    from turbotab.core.decisions import Refusal, SetSelection, _ctx

    if decision.method == "none":
        return
    state = _ctx_state(ctx)
    task = getattr(state, "task", None) if state is not None else None
    if task is None:
        task = _ctx(ctx, "task") or _ctx(ctx, "detected_task")
    if task is not None and task not in SUPPORTED_TASKS:
        raise Refusal(
            "selection_task",
            f"The selection methods here read a numeric or yes/no outcome; a {str(task).replace('_', '-')} "
            f"outcome keeps every candidate (a penalized family shrinks them instead).",
            exits=[{"label": "No selection", "decision": SetSelection(method="none")}])
    if decision.method == "stepwise" and not decision.sensitivity \
            and getattr(state, "purpose", None) != "inference":
        _stepwise_has_rows(decision, ctx, state, task)


def stepwise_room(state: Any, ctx: Any, task: str | None) -> tuple[int, int] | None:
    """(candidate columns, rows of the smallest training fold) for backward elimination under
    prediction, from the cohort's predictors and the table's summaries; None when they are not
    known yet. The columns are those the selection step receives: a number one, a category of k
    levels k − 1, a declared spline its k − 1, a form rule's splines theirs
    (``stages.modeling.rule_spline_terms``, k read on the fold's rows, never fewer than its
    effective size gives), and a missing-value indicator per predictor with blanks when indicators
    are kept. The fold holds (K − 1)/K of the training rows, the holdout drawn first."""
    from turbotab.core.decisions import _ctx
    from turbotab.core.methods.exposure_form import model_terms
    from turbotab.core.readings import confirmed_codes
    from turbotab.core.sequence import artifact
    from turbotab.core.stages.modeling import _summary, predictor_parameters, rule_spline_terms

    cohort = artifact(ctx, "cohort") or {}
    info = _ctx(ctx, "column_info") or {}
    n = cohort.get("n_final")
    predictors = [c for c in cohort.get("predictors") or [] if c in info]
    if not n or not predictors or state is None:
        return None
    split = getattr(state, "split", None)
    holdout = float(getattr(split, "holdout", 0.0) or 0.0)
    folds = int(getattr(split, "folds", 5) or 5)
    n_train = int(math.floor(int(n) * (1.0 - holdout)))
    n_fold = int(math.floor(n_train * (folds - 1) / folds))
    columns = (predictor_parameters(predictors, info, confirmed_codes(state))
               + model_terms(predictors, getattr(state, "exposure_forms", None)) - len(predictors)
               + rule_spline_terms(state, predictors, info, n_fold))
    missing = getattr(state, "missing", None)
    if getattr(missing, "indicators", False):
        columns += sum(1 for c in predictors if int(_summary(info[c], "n_missing") or 0) > 0)
    return columns, n_fold


def _stepwise_has_rows(decision: Any, ctx: Any, state: Any, task: str | None) -> None:
    """Backward elimination starts from the model with every candidate column and an intercept,
    which least squares or the logistic likelihood can fit only with fewer columns than rows less
    one (``MASS::stepAIC`` stops there too: "AIC is -infinity for this model"). Where the smallest
    training fold has too few rows, the answer is refused with the selections that work at p ≫ n
    (MODELING_SEQUENCE §1 row 8)."""
    from turbotab.core.decisions import Refusal

    room = stepwise_room(state, ctx, task)
    if room is None:
        return
    columns, n_fold = room
    if columns < n_fold - 1:
        return
    raise Refusal(
        "stepwise_needs_rows",
        f"Backward elimination starts from all {columns:,} candidate columns, and the smallest "
        f"training fold has {n_fold:,} rows: a model with every column and an intercept needs more "
        f"rows than that, so it cannot start. At p ≫ n a penalized or screening selection comes "
        f"first.",
        exits=[{"label": "Elastic net, in each training fold",
                "decision": decision.model_copy(update={"method": "elastic_net"})},
               {"label": "In-fold screening by correlation with the outcome",
                "decision": decision.model_copy(update={"method": "screening"})},
               {"label": "No selection", "decision": decision.model_copy(update={"method": "none"})}])


def _register_validators() -> None:
    from turbotab.core.decisions import register_validator

    register_validator("set_selection", _selection_runs_in_fold)
    register_validator("set_selection", _selection_is_sensitivity_under_inference)
    register_validator("set_selection", _selection_fits_the_task)


_register_validators()


def decision_sentence(d: Any, state: Any) -> str:
    """The ``set_selection`` record's methods sentence."""
    from turbotab.core.voice import tick

    pre = {"no": " No predictor was pre-selected on these rows' outcome (TRIPOD+AI 9a).",
           "other_data": " Candidate predictors were pre-selected on other data (TRIPOD+AI 9a).",
           "unknown": " Whether any predictor was pre-selected before this analysis is unknown "
                      "(TRIPOD+AI 9a).",
           None: ""}[getattr(d, "pre_selected", None)]
    if getattr(d, "sensitivity", False):
        return (f"As a labeled sensitivity analysis, covariates were removed by backward "
                f"elimination at α = {tick(INFERENCE_ALPHA)}, each Wald test pooled across the "
                f"imputed copies by Rubin's rules (Wood et al. 2008), the exposure kept; the "
                f"declared model is the reported one.{pre}")
    m = d.method
    if m == "none":
        return f"No predictor selection was applied: every candidate predictor entered the models.{pre}"
    how = {
        "elastic_net": "an elastic net (mixing 0.5, its penalty by inner cross-validation) kept "
                       "the predictors with nonzero coefficients",
        "stability": (f"stability selection with complementary pairs (Shah & Samworth 2013; "
                      f"{tick(STABILITY_PAIRS)} pairs of halves, threshold "
                      f"{tick(d.threshold or STABILITY_TAU)}) kept a short reproducible panel"),
        "screening": (f"the {tick(d.keep) if d.keep else 'n / ln n'} predictors most correlated "
                      f"with the outcome were kept"),
        "stepwise": "backward elimination by AIC kept its predictors (customary; unstable)",
        "univariable": (f"predictors with a univariable p below {tick(d.threshold or UNIVARIABLE_P)} "
                        f"were kept (customary; unstable)"),
        "vip": "predictors with a PLS VIP above 1 were kept (customary; unstable)",
    }[m]
    return (f"Within each training fold, {how}, so every fold and resample repeated the selection "
            f"(Ambroise & McLachlan 2002); each predictor's inclusion frequency is reported.{pre}")


__all__ = ["CUSTOMARY", "INFERENCE_ALPHA", "INSTABILITY", "LABELS", "PREDICTION_METHODS",
           "STABILITY_PAIRS", "STABILITY_TAU", "Selector", "UNIVARIABLE_P", "aic", "backward_aic",
           "backward_wald", "complementary_pairs", "decision_sentence", "elastic_net_support",
           "inclusion_frequencies", "lasso_first", "outcome01", "pls_vip", "screening_keep",
           "screening_scores", "selection_step", "stability_bound", "stability_q", "standardized",
           "term_groups", "term_p", "univariable_p"]

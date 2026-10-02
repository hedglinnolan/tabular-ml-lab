"""``featurewise``: each exposure tested on its own, adjusted for the covariates, with
Benjamini–Hochberg false-discovery control (AUDIT_REPORT §5 WP11; closes ME-18).

Omics inference has no sound family on the shelf without it: at p ≥ n least squares has no unique
fit and elastic net's shrunk coefficients carry no confidence intervals. The field's workhorse is
per-feature testing with a multiple-testing correction — METABOLOMICS_PACK §08, SETTLED: "per-feature
testing with multiple-testing correction is expected, and its absence is a fatal flaw in review …
covariate-adjusted → linear model on log values … Benjamini–Hochberg at q < 0.05 is the field
standard" — and limma's design for expression data is the same linear model, one per gene.

The model, one per exposure ``j`` (the predictors with the role exposure; every other column of the
model matrix — covariates, energy, one-hot levels — is the adjustment set ``Z``):

* **numeric outcome** — ``outcome ~ x_j + Z`` by least squares; the estimate is the change in the
  outcome per unit of ``x_j`` (per log2 unit once the values are normalized to log scale);
* **two-level outcome** — ``x_j ~ event + Z`` by least squares, the limma design; the estimate is
  the difference in ``x_j``'s mean between the event and the other level, adjusted for ``Z`` (a
  log2 fold change on log2 values).

Standard errors: each test's own residual variance (classical least squares, limma's linear model
without its variance moderation; Smyth 2004), on t(n − rank[1, Z] − 1). Under the null that the
exposure is unrelated to the outcome given ``Z``, the residuals do not depend on the exposure, so
this t-test is exact for normal errors whatever the exposure's distribution — and the null is what
a false-discovery rate is about. The linear family's HC3 is not used here: Benjamini–Hochberg
works in the far tail (a p-value near 0.05/m), where an HC3 t-statistic is much heavier-tailed than
its t reference. Simulated (n = 40, 500 null exposures, one covariate; the acceptance suite,
``test_wp11_omics.py``, keeps the comparison): the chance of any false discovery after
Benjamini–Hochberg at q = 0.05 was 0.11 with HC3 on normal exposures and 0.74 on log-normal ones
(200 replicates), against 0.054 and 0.051 with these standard errors (2,000 replicates, Monte
Carlo error 0.005). The price: an interval around a real effect assumes equal residual variance.

When a unit's rows repeat: CR2 with Bell–McCaffrey degrees of freedom, the math layer's estimator
(up to :data:`CR2_MAX_FEATURES` exposures; beyond, the table is refused rather than reported as if
the rows were independent). p-values are two-sided; ``q`` is the Benjamini–Hochberg adjusted
p-value over the exposures tested, and a discovery is ``q <`` :data:`FDR`.

Every test is computed by Frisch–Waugh–Lovell: ``Z`` is projected out once, so 20,000 tests cost a
few matrix products. The family makes no predictions (``predicts = False``): the fit stage reports
its table and no cross-validated score.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from turbotab.core.decisions import Purpose, Task
from turbotab.core.models.base import Assessment, FamilyBase, Situation, register_family

FDR = 0.05
LEVEL = 0.95
CR2_MAX_FEATURES = 200  # CR2 is one design per exposure; beyond this the table is refused when rows repeat
_BLOCK_FLOATS = 4_000_000  # n × exposures per block of the vectorized tests (32 MB)
INDICATOR = "missingindicator_"
OMICS = ("genomics", "metabolomics")
PLS_DA_ABSENT = ("PLS-DA is not offered: customary in metabolomics, it is sound only with nested "
                 "cross-validation and permutation tests, and VIP > 1 ranks features without "
                 "testing them.")


def bh_adjust(p: Sequence[float]) -> np.ndarray:
    """Benjamini & Hochberg (1995) adjusted p-values: ``q_(i) = min_{k ≥ i} m p_(k) / k``, capped at
    1; a missing p stays missing and does not count among the m tests."""
    p = np.asarray(p, dtype=float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if not m:
        return q
    order = np.argsort(p[ok])
    ranked = p[ok][order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(ranked, 1.0)
    q[ok] = out
    return q


def _basis(Z: np.ndarray) -> np.ndarray:
    """An orthonormal basis of the column space of ``Z`` (rank-revealing: collinear columns add
    nothing)."""
    U, s, _ = np.linalg.svd(Z, full_matrices=False)
    tol = s.max() * max(Z.shape) * np.finfo(float).eps if s.size else 0.0
    return U[:, s > tol]


def split_columns(columns: Sequence[str], features: Sequence[str] | None) -> tuple[list[str], list[str]]:
    """(tested, adjustment): the exposures among ``columns``, and every other column but a tested
    exposure's own missing indicator."""
    names = [str(c) for c in columns]
    if features is None:
        return names, []
    wanted = set(features)
    tested = [c for c in names if c in wanted]
    mine = set(tested)
    adjust = [c for c in names if c not in mine
              and not (c.startswith(INDICATOR) and c[len(INDICATOR):] in mine)]
    return tested, adjust


def event_indicator(y: Any) -> tuple[np.ndarray, Any, Any]:
    """(0/1 event indicator, event level, other level) of a two-level outcome: 1 where the outcome
    is 1 when it is coded 0/1 (the fit stage codes the named event 1; the other level is then
    None, its name unknown here), else its last level."""
    values = pd.Series(np.asarray(y, dtype=object))
    levels = sorted(values.dropna().unique().tolist(), key=lambda v: str(v))
    if set(levels) <= {0, 1, 0.0, 1.0, True, False} and len(levels) == 2:
        return values.astype(float).to_numpy(), 1, None  # coded: the other level's name is not here
    if len(levels) != 2:
        raise ValueError(f"A two-level outcome needs two levels; this one has {len(levels)}.")
    return (values == levels[1]).to_numpy(dtype=float), levels[1], levels[0]


def classical_tests(response: np.ndarray, regressor: np.ndarray, Q: np.ndarray, df: float, *,
                    response_is_matrix: bool) -> tuple[np.ndarray, np.ndarray]:
    """(estimates, standard errors) of the regressor's coefficient in ``response ~ regressor +
    span(Q)``, one test per column, by Frisch–Waugh–Lovell.

    ``response_is_matrix``: the responses are the columns of ``response`` (n × m) and ``regressor``
    is one vector (the event); otherwise ``response`` is one vector (the outcome) and the regressors
    are the columns of ``regressor``. With the residualized regressor ``r`` and the full model's
    residuals ``e``: ``β = Σ r_i ỹ_i / Σ r_i²`` and ``Var = (Σ e_i² / df) / Σ r_i²``, the
    least-squares coefficient and its classical variance, ``df = n − rank[1, Z] − 1``.
    """
    if response_is_matrix:
        g = regressor - Q @ (Q.T @ regressor)
        sgg = float(g @ g)
        Y = response - Q @ (Q.T @ response)
        if sgg <= 0:
            return np.full(Y.shape[1], np.nan), np.full(Y.shape[1], np.nan)
        beta = (g @ Y) / sgg
        e = Y - np.outer(g, beta)
        return beta, np.sqrt(np.sum(e ** 2, axis=0) / df / sgg)
    X = regressor - Q @ (Q.T @ regressor)
    yt = response - Q @ (Q.T @ response)
    sxx = np.sum(X ** 2, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = (yt @ X) / sxx
        e = yt[:, None] - X * beta
        var = np.sum(e ** 2, axis=0) / df / sxx
    # A column with nothing left once Z is projected out (constant, or a copy of a covariate)
    # cannot be tested: no estimate, rather than a number from rounding error.
    flat = sxx <= 1e-12 * np.maximum(np.sum(regressor ** 2, axis=0), 1e-300)
    return np.where(flat, np.nan, beta), np.sqrt(np.where(flat, np.nan, var))


def _p_and_ci(est: np.ndarray, se: np.ndarray, df: np.ndarray) -> tuple[np.ndarray, ...]:
    from scipy import stats

    ok = np.isfinite(est) & np.isfinite(se) & (se > 0) & np.isfinite(df) & (df > 0)
    p = np.full(est.shape, np.nan)
    lo = np.full(est.shape, np.nan)
    hi = np.full(est.shape, np.nan)
    if ok.any():
        t = est[ok] / se[ok]
        p[ok] = 2.0 * stats.t.sf(np.abs(t), df[ok])
        q = stats.t.ppf(0.5 + LEVEL / 2, df[ok])
        lo[ok] = est[ok] - q * se[ok]
        hi[ok] = est[ok] + q * se[ok]
    return p, lo, hi


def _clean(x: float) -> float | None:
    return float(x) if np.isfinite(x) else None


def featurewise_table(matrix: pd.DataFrame, y: Any, task: str, features: Sequence[str] | None,
                      clusters: Any = None, event: str | None = None) -> Any:
    """The feature-wise inference table (an ``InferenceTable``) on the model matrix."""
    from turbotab.core.models.inference import (INDEPENDENT, InferenceTable, _info, floor_refusal,
                                                format_p)

    clusters = INDEPENDENT if clusters is None else clusters
    tested, adjust = split_columns(list(matrix.columns), features)
    if not tested:
        raise ValueError("No exposure to test: feature-wise regression tests the columns with the "
                         "exposure role, one at a time, adjusted for the rest.")
    values = matrix.to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(values).all():
        raise ValueError("The model matrix has missing values; feature-wise tests need complete "
                         "rows (choose complete cases or imputation).")
    n = len(matrix)
    Z = np.column_stack([np.ones(n), matrix[adjust].to_numpy(dtype=float)]) if adjust else np.ones((n, 1))
    Q = _basis(Z)
    df_resid = float(n - Q.shape[1] - 1)
    X = matrix[tested].to_numpy(dtype=float)
    binary = task == "binary"
    if binary:
        g, level, other = event_indicator(y)
        against = f"`{other}`" if other is not None else "the other level"
        estimand = f"the difference in each exposure's mean between `{event or level}` and {against}"
    elif task == "regression":
        yv = np.asarray(y, dtype=float)
        estimand = "the change in the outcome per unit of each exposure"
    else:
        raise ValueError("Feature-wise regression models a numeric or a two-level outcome.")
    m = len(tested)
    estimator = "feature-wise least squares"
    refusal = floor_refusal(clusters)
    if refusal is None and clusters.clustered and m > CR2_MAX_FEATURES:
        refusal = (f"`{clusters.column}` repeats, and cluster-robust intervals for {m:,} separate "
                   f"tests are beyond what TurboTab computes ({CR2_MAX_FEATURES:,} at most); "
                   f"intervals that treated the rows as independent would be too narrow, so none "
                   f"is reported.",
                   ({"label": f"Combine each `{clusters.column}`'s rows into one (the unit question, "
                              f"before the seal)", "decision": None},))
    est = np.full(m, np.nan)
    se = np.full(m, np.nan)
    df = np.full(m, df_resid)
    if refusal is None and df_resid < 1:
        refusal = (f"{n:,} rows leave no residual degrees of freedom after {Q.shape[1]:,} adjustment "
                   f"columns, so no test can be made.", ())
    if not clusters.clustered or refusal is not None:
        step = max(1, _BLOCK_FLOATS // max(n, 1))
        for lo in range(0, m, step):
            block = slice(lo, lo + step)
            if refusal is not None and df_resid < 1:
                break  # no residual degrees of freedom: nothing can be estimated
            if binary:
                est[block], se[block] = classical_tests(X[:, block], g, Q, df_resid,
                                                        response_is_matrix=True)
            else:
                est[block], se[block] = classical_tests(yv, X[:, block], Q, df_resid,
                                                        response_is_matrix=False)
    else:
        est, se, df = _cr2_tests(X, g if binary else yv, Z, clusters.codes, binary)
    concerns = [clusters.note] if clusters.note else []
    if refusal is not None:
        reason, exits = refusal
        rows = [{"feature": f, "estimate": _clean(b), "ci_low": None, "ci_high": None, "p": None,
                 "se": None, "df": None, "q": None} for f, b in zip(tested, est)]
        info = _info(estimator, "none", f"No intervals: {reason}", clusters, refused=reason,
                     exits=[dict(e) for e in exits])
        return InferenceTable(rows, info, [reason, *concerns])
    p, low, high = _p_and_ci(est, se, df)
    q = bh_adjust(p)
    found = int(np.sum(q < FDR))
    rows = [{"feature": f, "estimate": _clean(b), "ci_low": _clean(a), "ci_high": _clean(c),
             "p": _clean(pv), "se": _clean(s), "df": _clean(d), "q": _clean(qv)}
            for f, b, a, c, pv, s, d, qv in zip(tested, est, low, high, p, se, df, q)]
    n_adjust = Q.shape[1] - 1
    adjusted = (f"with {n_adjust:,} adjustment column{'s' if n_adjust != 1 else ''}" if n_adjust
                else "with no adjustment columns")
    if n_adjust:
        estimand += ", adjusted for them"
    if clusters.clustered:
        how = (f"cluster-robust (CR2) by `{clusters.column}`, G = {clusters.n_clusters:,}, on t with "
               f"Bell–McCaffrey degrees of freedom")
        covariance = "CR2"
    else:
        how = (f"from each test's own residual variance (classical least squares, exact under "
               f"the null for normal errors), on t({df_resid:,.0f})")
        covariance = "model"
    caption = (f"Each of {m:,} exposures tested on its own {adjusted}; the estimate is {estimand}. "
               f"95% intervals {how}. Benjamini–Hochberg q < {FDR:g}: {found:,} of {m:,}.")
    smallest = np.nanmin(q) if np.isfinite(q).any() else float("nan")
    if found == 0 and np.isfinite(smallest):
        concerns.append(f"No exposure passes the false-discovery threshold (smallest q = "
                        f"{format_p(float(smallest))}): {m:,} tests at p < 0.05 would give about "
                        f"{0.05 * m:,.0f} false positives by chance alone.")
    return InferenceTable(rows, _info(estimator, covariance, caption, clusters), concerns)


def _cr2_tests(X: np.ndarray, other: np.ndarray, Z: np.ndarray, codes: np.ndarray,
               binary: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One CR2 fit per exposure, by the math layer's estimator (``inference.cr2``)."""
    from turbotab.core.models.inference import bread_of, cr2

    m = X.shape[1]
    est, se, df = np.full(m, np.nan), np.full(m, np.nan), np.full(m, np.nan)
    for j in range(m):
        if binary:
            D, resp = np.column_stack([Z, other]), X[:, j]
        else:
            D, resp = np.column_stack([Z, X[:, j]]), other
        bread = bread_of(D)
        beta = bread @ (D.T @ resp)
        e = resp - D @ beta
        V, dfs = cr2(D, e, codes, bread)
        est[j] = beta[-1]
        se[j] = float(np.sqrt(max(V[-1, -1], 0.0)))
        df[j] = float(dfs[-1]) if dfs is not None else float(int(codes.max()) + 1 - 1)
    return est, se, df


class FeatureWiseTests(BaseEstimator):
    """The model step: fitting it computes the feature-wise table on independent rows (the fit
    stage asks the family for the clustered one). It makes no predictions."""

    def __init__(self, task: str = "regression", features: Sequence[str] | None = None):
        self.task = task
        self.features = features

    def fit(self, X: pd.DataFrame, y: Any) -> "FeatureWiseTests":
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.feature_names_in_ = np.asarray([str(c) for c in frame.columns], dtype=object)
        self.n_features_in_ = frame.shape[1]
        self.tested_, self.adjust_ = split_columns(list(frame.columns), self.features)
        self.table_ = featurewise_table(frame, y, self.task, self.features)
        if self.task == "binary":
            _, event, other = event_indicator(y)
            self.classes_ = np.asarray([0 if other is None else other, event], dtype=object)
        return self


class FeatureWise(FamilyBase):
    key = "featurewise"
    label = "Feature-wise regression"
    tasks = ("regression", "binary")
    purposes = ("inference",)
    predicts = False
    linear_in_values = True
    inductive_bias = ("Each exposure tested on its own with the covariates; Benjamini–Hochberg "
                      "holds the false-discovery rate.")
    strengths = (
        "Valid tests with more features than rows.",
        "Holds the false-discovery rate across thousands of features (Benjamini–Hochberg).",
    )
    cautions = (
        "Each estimate ignores the other exposures: one separate question per feature.",
        "Tests only: it makes no predictions and has no cross-validated score.",
        "Intervals assume equal residual variance; robust ones fail at the thresholds it tests.",
    )
    needs_scaling = False
    handles_missing = False

    def build(self, task: Task, purpose: Purpose | None, n_rows: int, n_features: int) -> Any:
        return FeatureWiseTests(task=task, features=None)

    def build_for(self, spec: Any, task: Task, purpose: Purpose | None, n_rows: int,
                  n_features: int) -> Any:
        """The model step for this design: the exposures are tested, everything else adjusts."""
        exposures = [c for c in spec.predictors if spec.roles.get(c) == "exposure"]
        return FeatureWiseTests(task=task, features=exposures)

    def describe(self, task: Task, purpose: Purpose | None) -> tuple[str, str]:
        if task == "binary":
            what = ("Regresses each exposure on the event and the covariates (the limma design): "
                    "the estimate is the adjusted difference in its mean.")
        else:
            what = ("Regresses the outcome on each exposure and the covariates, one exposure at a "
                    "time: the estimate is per unit of that exposure.")
        return "Feature-wise least squares", (f"{what} Classical least-squares intervals, CR2 when "
                                              f"a unit's rows repeat; Benjamini–Hochberg q-values "
                                              f"across the exposures.")

    def _matrix(self, pipeline: Any, X: Any) -> pd.DataFrame:
        from turbotab.core.models.linear import model_matrix

        return model_matrix(pipeline, X)

    def coefficients(self, pipeline: Any, X: Any, y: Any, *, task: Task,
                     purpose: Purpose | None, groups: Any = None) -> list[dict[str, Any]] | None:
        from turbotab.core.models.linear import as_clusters

        return self.inference(pipeline, X, y, task=task, clusters=as_clusters(groups)).rows

    def inference(self, pipeline: Any, X: Any, y: Any, *, task: Task, clusters: Any,
                  event: str | None = None) -> Any:
        """The feature-wise table on the matrix the model step saw, clustered when rows repeat."""
        model = pipeline[-1]
        return featurewise_table(self._matrix(pipeline, X), y, task,
                                 getattr(model, "features", None), clusters, event)

    def assess(self, s: Situation) -> Assessment:
        if s.purpose != "inference":
            return Assessment(0.0, "poor", ("Tests each exposure on its own and makes no "
                                            "predictions, so it adds nothing to a prediction.",))
        lenses = tuple(getattr(s, "lenses", ()) or ())
        omics = any(lens in OMICS for lens in lenses)
        if omics or s.n_features >= s.n_rows:
            # AUDIT_REPORT ME-18: PLS-DA is not on the shelf, and the one line says why.
            pls = (PLS_DA_ABSENT,) if "metabolomics" in lenses else ()
            return Assessment(3.5, "good", pls)
        return Assessment(2.0, "fair", ("Each exposure is adjusted for the covariates but not for the "
                                        "other exposures: a separate question for each.",))


FEATUREWISE = register_family(FeatureWise())

__all__ = ["CR2_MAX_FEATURES", "FDR", "FEATUREWISE", "FeatureWise", "FeatureWiseTests", "bh_adjust",
           "classical_tests", "event_indicator", "featurewise_table", "split_columns"]

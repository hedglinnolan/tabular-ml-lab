"""Scale scores, their reliability, and the correction for their measurement error (MS8;
MODELING_SEQUENCE §0 ruling 8, §2, §6 chain 4; V2 definition of done §2, survey instruments).

**Ruling 8** (2026-10-03, after the adversarial review): "Scale reliability is ω, not α (omega-total
for a unidimensional scale; omega-hierarchical for a multidimensional one). α is shown only with its
customary label. Disattenuation by α or ω is refused for formative indices (diet-quality scores,
FFQ-derived scores). It needs test–retest ICC or a calibration substudy. For reflective scales, the
correction is conditional on the covariates (done by regression calibration) and labeled as omitting
transient error."

**Scoring** (row-local: a row's score reads only its own answers). The items of a declared scale,
the reverse-coded ones turned over on the instrument's own response scale (``low + high − x``), are
summed or averaged. Which items are reversed comes from the instrument's published key, declared by
the user and never inferred: a negative item–rest correlation has four explanations (the item needs
reversing, it was already reversed, it loads on a method factor, it does not belong) and the values
cannot tell them apart (CLINICAL_SURVEY_PACK §B1.2). Codes outside the response scale ("don't know",
"refused") must be recoded to missing first (the sentinel repair); the scale is refused until they
are. Under inference a missing item is multiply imputed at the item level before the score is formed
in each completed copy: Eekhout et al. 2014 (*J Clin Epidemiol* 67:335): "when a large percentage of
subjects had missing items (>25%), MI methods applied to the items outperformed methods applied to
the total score". Under prediction the in-fold fill runs before the score, as every other step.

**Reliability.** McNeish 2018 (*Psychol Methods* 23:412): "Cronbach's alpha is riddled with problems
stemming from unrealistic assumptions. In many circumstances, violating these assumptions yields
estimates of reliability that are too small". The coefficients are McDonald's ω, computed as R's
``psych::omega`` computes them (Revelle; Zinbarg, Revelle, Yovel & Li 2005, *Psychometrika* 70:123):

* the items' correlation matrix R;
* a minimum-residual (unweighted least squares) factor analysis with ``f`` factors (Harman & Jones
  1966, *Psychometrika* 31:351): the uniquenesses Ψ minimize the residuals of R − Ψ after its best
  rank-f fit, started at 1 − SMC and kept within [0.005, max(1, SMC)]; the loadings are the leading
  eigenvectors of R − Ψ times the square roots of their eigenvalues;
* with one factor, ω_t = (V_t − Σu²)/V_t, where V_t = ΣR is the variance of the composite and
  u²_j = 1 − h²_j the item's uniqueness (h² its communality);
* with f ≥ 2 group factors: varimax (Kaiser 1958, Kaiser-normalized as ``stats::varimax``), then the
  oblique quartimin rotation by gradient projection from that start (Bernaards & Jennrich 2005,
  *Educ Psychol Meas* 65:676; ``GPArotation::oblimin``), then the Schmid–Leiman transformation
  (Schmid & Leiman 1957, *Psychometrika* 22:53): the factors' correlations Φ are themselves
  factored (one factor; with f = 2 its two loadings are set equal, √|φ₁₂|, ``psych``'s default, which
  needs a caution), and each item's general loading is g = F γ. ω_h = (Σg)²/V_t.

ω above is the standardized-item composite's, the number ``psych::omega`` prints. A score is a sum
of the items in their own units, so the reliability *of the score* rescales the same factor solution
by each item's standard deviation s_j: ω_t = 1 − Σ s²_j u²_j / ΣC and ω_h = (Σ s_j g_j)² / ΣC, with C
the items' covariance matrix (the two agree when the items' standard deviations are equal). α is
Cronbach's (1951) on the same covariance matrix, ``psych::alpha``'s ``raw_alpha``, shown labeled
*customary*.

**The correction** (inference only; a declared secondary analysis beside the uncorrected estimate).
Keogh, Shaw & Gustafson 2020 (STRATOS Part 1, *Stat Med* 39:2197, §3.1.2): with covariates Z, "the
relationship β*X = λβX still holds but now λ = αX|Z var(X|Z) / (α²X|Z var(X|Z) + var(U))": the slope is
attenuated by the reliability *conditional on Z*, not by the scale's marginal one, so β/ω
under-corrects whenever the score correlates with the covariates. The correction is therefore
regression calibration with every other column of the model's matrix in the calibration equation
(Boe et al. 2023, *Am J Epidemiol* 192:1406: "the calibration equation should include all confounders
included in the outcome model"):

* the score W is regressed on Z by least squares; s²_{W|Z} = RSS/(n − 1), the Schur complement of the
  (n − 1) covariance matrix of (W, Z);
* σ²_U is the score's error variance: (1 − ω)·var(W) from the internal consistency (for a
  multidimensional scale ω-hierarchical, so the group factors' variance counts as error about the
  general construct), the two-way residual mean square of a repeat administration (test–retest), or,
  with a calibration substudy, no σ²_U at all: the reference measure is regressed on W and Z in the
  substudy's rows;
* λ = (s²_{W|Z} − σ²_U)/s²_{W|Z}, and the calibrated score is fitted + λ·(W − fitted);
* the outcome model is refit with the calibrated score in place of W. For least squares the corrected
  coefficient is exactly the uncorrected one divided by λ (R's ``mecor`` with ``MeasErrorRandom``
  computes the same); for a logistic or proportional-odds outcome the substitution is the usual
  approximation (Carroll, Ruppert, Stefanski & Crainiceanu 2006, *Measurement Error in Nonlinear
  Models*, §4.2), and the result says so.

**Several scores corrected in one model** are calibrated jointly (multivariate regression
calibration; Rosner, Spiegelman & Willett 1990, *Am J Epidemiol* 132:734; MODELING_SEQUENCE §0
ruling 7 and §2: "multivariate calibration when several intakes are error-prone"). Freedman et al.
2011: with two or more error-prone exposures, estimates "may become attenuated, inflated, or can
even change direction", which a correction of each score on its own, the others read as exact,
does not undo. With W the k corrected scores and Z every other column of the model's matrix:

* each score is regressed on Z by least squares; S = RᵀR/(n − 1) is the k × k covariance of the
  residuals R (the Schur complement of the (n − 1) covariance matrix of (W, Z));
* a score j whose error variance σ²_j is known (ω or a repeat administration) has
  E[X_j | W, Z] = fitted_j + (S − D)_j S⁻¹ rᵢ, where D = diag(σ²) over those scores: with classical
  errors independent of one another, of the true scores and of Z, cov(X_j, W_l | Z) = S_jl − δ_jl σ²_j;
* a score with a calibration substudy has its reference measure regressed on every W and Z in the
  substudy's rows, and that prediction for every row;
* the outcome model is refit with every calibrated score in place of its W. Each score's λ reported
  is its own slope in that calibration (the j-th diagonal of (S − D) S⁻¹), the univariate λ when
  k = 1, where the method reduces to the one above.

Its interval comes from refitting on bootstrap replicates, the reliability (ω and the whole factor
analysis, the test–retest mean square, or the calibration regression) re-estimated in every
replicate: the estimate ± 1.96 bootstrap standard errors. The replicates resample independent rows,
or, when rows are grouped (a household, a site, a person's repeated rows: the clusters the
coefficient table's intervals are robust to), whole clusters, so that the variance keeps each
cluster's rows together (MODELING_SEQUENCE §0 ruling 7: "a bootstrap over the whole chain,
resampling PSUs within strata, or clusters"; a population design, whose PSUs and strata it would
resample, is blocked before any correction is computed). Under multiple imputation each completed
copy is corrected and bootstrapped on its own, and the copies are combined by Rubin's rules with
each copy's bootstrap variance (``turbotab.core.stages.scales``). The test of no association stays the
uncorrected model's: for one error-prone exposure "the usual statistical test of the null hypothesis
(no exposure effect) remains theoretically valid even though the estimated relative risk is
attenuated" (Freedman et al. 2011, *J Natl Cancer Inst* 103:1086).

**Formative indices.** A diet-quality score is defined by its components; it is not their common
cause. Reedy et al. 2018 (*J Acad Nutr Diet* 118:1622) on the HEI-2015: "the standardized Cronbach's
alpha was .67" and "The components demonstrated multidimensionality when examined with a scree plot
(at least four dimensions)". Internal consistency is not the error that dilutes the regression, so
dividing by α or ω over-corrects: that answer is refused, with two exits, a test–retest ICC from a
repeat administration or a calibration substudy against a reference measure.

**Transient error.** Schmidt, Le & Ilies 2003 (*Psychol Methods* 8:206): "the nearly universal use of
the coefficient of equivalence (Cronbach's alpha …), which fails to assess transient error, leads to
overestimates of reliability and undercorrections for biases due to measurement error." A correction
from α or ω is labeled as omitting transient error; one from a repeat administration includes it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

Z_95 = 1.959963984540054  # the standard normal's 97.5th percentile
PSI_LOWER = 0.005  # psych's lower bound on a uniqueness
MIN_ITEMS = 3  # ω needs three items a factor to be identified; α two
ITEMS_PER_FACTOR = 3

MCNEISH = "McNeish 2018"
STRATOS = "Keogh, Shaw & Gustafson 2020"
BOE = "Boe et al. 2023"
SCHMIDT = "Schmidt, Le & Ilies 2003"
REEDY = "Reedy et al. 2018"
EEKHOUT = "Eekhout et al. 2014"
ZINBARG = "Zinbarg et al. 2005"
FREEDMAN = "Freedman et al. 2011"
CARROLL = "Carroll et al. 2006"
ROSNER = "Rosner, Spiegelman & Willett 1990"

Scoring = Literal["sum", "mean"]


class ScaleRefused(ValueError):
    """The data cannot support the computation; the message says why. ``which``: the position,
    among the scores corrected together, of the one whose data refused (None: all of them)."""

    def __init__(self, message: str, which: int | None = None):
        super().__init__(message)
        self.which = which


# ── scoring ──────────────────────────────────────────────────────────────────


def reverse_code(values: Any, low: float, high: float) -> np.ndarray:
    """An item turned over on its response scale: ``low + high − x`` (1–5: 1 ↔ 5, 2 ↔ 4)."""
    return float(low) + float(high) - np.asarray(values, dtype=float)


def keyed_items(frame: pd.DataFrame, items: Sequence[str], reverse: Sequence[str],
                low: float, high: float) -> pd.DataFrame:
    """The items as the score reads them: each reverse-coded item turned over, the others as
    recorded. Missing answers stay missing."""
    out = pd.DataFrame(index=frame.index)
    turned = set(reverse)
    for c in items:
        x = pd.to_numeric(frame[c], errors="coerce").to_numpy(dtype=float)
        out[c] = reverse_code(x, low, high) if c in turned else x
    return out


def score_items(keyed: pd.DataFrame | np.ndarray, scoring: Scoring) -> np.ndarray:
    """The sum or the mean of the keyed items; a row with any item missing has no score (the
    imputation, or the complete-case rows, decide what reaches here)."""
    M = np.asarray(keyed, dtype=float)
    total = M.sum(axis=1)  # NaN wherever an item is NaN
    return total / M.shape[1] if scoring == "mean" else total


class ScaleScorer(TransformerMixin, BaseEstimator):
    """The pipeline step that scores each declared scale: its items, keyed by the instrument
    (reverse-coded items turned over on ``low``–``high``), become one column, ``name``, the sum or
    the mean, in the place of the first item; every other column passes through. Row-local: a row's
    score reads its own answers only, so nothing is learned in ``fit``.

    It runs after the in-fold fill (prediction) and on each completed copy of the multiple
    imputation (inference), so a blank item is filled at the item level before the score is formed.
    An imputed answer may lie between or beyond the response scale's points; it is summed as drawn,
    since rounding a normal-model draw biases the estimates (the score is a sum, and its coefficient
    the estimand)."""

    def __init__(self, scales: Sequence[Mapping[str, Any]] = ()):
        self.scales = scales

    def fit(self, X: pd.DataFrame, y: Any = None) -> "ScaleScorer":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ScaleScorer needs a pandas DataFrame with named columns.")
        names = [str(c) for c in X.columns]
        for s in self.scales:
            missing = [c for c in s["items"] if c not in names]
            if missing:
                raise ValueError(f"`{s['name']}` is scored from {', '.join(missing)}, which "
                                 f"{'is' if len(missing) == 1 else 'are'} not among the "
                                 f"predictors; give each item the exposure or covariate role.")
        self.feature_names_in_ = np.asarray(names, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.outputs_ = self._outputs(names)
        return self

    def _outputs(self, names: Sequence[str]) -> list[str]:
        first = {s["items"][0]: s["name"] for s in self.scales}
        used = {c for s in self.scales for c in s["items"]}
        return [first[c] if c in first else c for c in names if c in first or c not in used]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "outputs_"):
            raise ValueError("ScaleScorer is not fitted yet.")
        scores = {}
        for s in self.scales:
            keyed = keyed_items(X, s["items"], s.get("reverse") or [], s["low"], s["high"])
            scores[s["name"]] = score_items(keyed, s.get("scoring") or "sum")
        parts = {name: (pd.Series(scores[name], index=X.index) if name in scores else X[name])
                 for name in self.outputs_}
        return pd.DataFrame(parts, index=X.index)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(self.outputs_, dtype=object)

    def lineage(self) -> list[dict[str, Any]]:
        """``{output, inputs, operation, formula}`` per output, as the energy step's."""
        out = []
        by_name = {s["name"]: s for s in self.scales}
        for name in self.outputs_:
            s = by_name.get(name)
            if s is None:
                out.append({"output": name, "inputs": [name], "operation": "pass-through"})
                continue
            turned = list(s.get("reverse") or [])
            terms = [f"({s['low']} + {s['high']} − {c})" if c in turned else c for c in s["items"]]
            formula = " + ".join(terms)
            if (s.get("scoring") or "sum") == "mean":
                formula = f"({formula}) / {len(terms)}"
            out.append({"output": name, "inputs": list(s["items"]),
                        "operation": f"scale score ({s.get('scoring') or 'sum'})",
                        "formula": f"{name} = {formula}"})
        return out


def describe(scales: Sequence[Mapping[str, Any]]) -> str:
    """The score step's line in the pipeline panel."""
    parts = []
    for s in scales:
        turned = list(s.get("reverse") or [])
        flip = (f", {len(turned)} reverse-coded on {s['low']}–{s['high']}" if turned else "")
        parts.append(f"{s['name']} is the {s.get('scoring') or 'sum'} of {len(s['items'])} "
                     f"items{flip}")
    return ("; ".join(parts) + ". Computed from each row's own answers, after blanks are filled, "
            "so a missing item is filled before its score is formed.")


# ── internal consistency: α ──────────────────────────────────────────────────


def cronbach_alpha(items: np.ndarray) -> float:
    """Cronbach's α on the items' covariance matrix (``psych::alpha``'s ``raw_alpha``):
    k/(k − 1) · (1 − tr C / ΣC)."""
    M = np.asarray(items, dtype=float)
    k = M.shape[1]
    if k < 2:
        raise ScaleRefused("α needs at least two items.")
    C = np.cov(M, rowvar=False)
    total = float(C.sum())
    if not total > 0:
        raise ScaleRefused("The items' covariances sum to zero or less, so α is undefined.")
    return k / (k - 1) * (1.0 - float(np.trace(C)) / total)


# ── the factor analysis behind ω ─────────────────────────────────────────────


def smc(R: np.ndarray) -> np.ndarray:
    """Squared multiple correlations, 1 − 1/diag(R⁻¹) (pseudo-inverse), clipped to [0, 1] as
    ``psych::smc`` clips them."""
    inv = np.linalg.pinv(R)
    out = 1.0 - 1.0 / np.diag(inv)
    return np.clip(out, 0.0, 1.0)


def _leading(M: np.ndarray, nf: int) -> np.ndarray:
    """The leading ``nf`` eigenvectors of ``M`` times the square roots of their eigenvalues (a
    negative one counts as zero): ``psych``'s ``FAout.wls``."""
    w, v = np.linalg.eigh(M)
    order = np.argsort(w)[::-1][:nf]
    return v[:, order] * np.sqrt(np.maximum(w[order], 0.0))


def minres(R: np.ndarray, nf: int) -> tuple[np.ndarray, np.ndarray]:
    """``(loadings, uniquenesses)`` of the minimum-residual (unweighted least squares) factor
    analysis of ``R`` with ``nf`` factors, as ``psych::fa(fm = "minres")`` finds them: Ψ minimizes
    ‖R − Ψ − LLᵀ‖² (L the best rank-nf fit of R − Ψ), whose gradient is −2·diag(R − Ψ − LLᵀ), from
    1 − SMC within [0.005, max(1, SMC)]. At an interior optimum this is the off-diagonal residual
    minimum ``psych`` reports (Harman & Jones 1966). Columns are signed to sum positive."""
    from scipy.optimize import minimize

    R = np.asarray(R, dtype=float)
    p = R.shape[0]
    s = smc(R)
    start = np.diag(R) - s
    upper = max(float(s.max()), 1.0)

    def objective(psi: np.ndarray) -> tuple[float, np.ndarray]:
        M = R - np.diag(psi)
        L = _leading(M, nf)
        resid = M - L @ L.T
        return float((resid ** 2).sum()), -2.0 * np.diag(resid)

    result = minimize(objective, np.clip(start, PSI_LOWER, upper), jac=True, method="L-BFGS-B",
                      bounds=[(PSI_LOWER, upper)] * p,
                      options={"ftol": 1e-16, "gtol": 1e-12, "maxiter": 20_000, "maxcor": 30})
    psi = np.asarray(result.x, dtype=float)
    L = _leading(R - np.diag(psi), nf)
    signs = np.sign(L.sum(axis=0))
    signs[signs == 0] = 1.0
    return L * signs, psi


def varimax(L: np.ndarray, normalize: bool = True, eps: float = 1e-5) -> np.ndarray:
    """``stats::varimax``: Kaiser's criterion by the iterated SVD, rows normalized to unit length
    first (Kaiser normalization), stopping when the criterion grows by less than a factor 1 + eps."""
    x = np.asarray(L, dtype=float)
    nc = x.shape[1]
    if nc < 2:
        return x.copy()
    sc = np.sqrt((x ** 2).sum(axis=1)) if normalize else np.ones(x.shape[0])
    sc[sc == 0] = 1.0
    x = x / sc[:, None]
    p = x.shape[0]
    T = np.eye(nc)
    d = 0.0
    for _ in range(1000):
        z = x @ T
        B = x.T @ (z ** 3 - z @ np.diag((z ** 2).sum(axis=0)) / p)
        u, sv, vt = np.linalg.svd(B)
        T = u @ vt
        past, d = d, float(sv.sum())
        if d < past * (1 + eps):
            break
    return (x @ T) * sc[:, None]


def _quartimin(L: np.ndarray) -> tuple[float, np.ndarray]:
    """The oblimin criterion with γ = 0 (quartimin) and its gradient, ``GPArotation``'s
    ``vgQ.oblimin``: X = rowSums(L²) − L², f = Σ L²X / 4, ∂f/∂L = L ∘ X."""
    L2 = L ** 2
    X = L2.sum(axis=1, keepdims=True) - L2
    return float((L2 * X).sum() / 4.0), L * X


def oblimin(A: np.ndarray, eps: float = 1e-9, maxit: int = 5_000,
            window: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """``(pattern loadings, factor correlations)`` of the oblique quartimin rotation of ``A`` by
    gradient projection (Bernaards & Jennrich 2005), as ``GPArotation::GPFoblq`` runs it today
    (``algorithm = "bb"``): started at the identity, L = A (Tᵀ)⁻¹ with T's columns of unit length,
    a Barzilai–Borwein step and a non-monotone line search over the last ``window`` criterion
    values, Φ = TᵀT. Converged far tighter than ``GPArotation``'s default (a projected gradient
    below 1e-9 against 1e-5), from the same start, so it reaches the same stationary point."""
    A = np.asarray(A, dtype=float)
    k = A.shape[1]
    T = np.eye(k)
    Tinv = np.linalg.inv(T)
    L = A @ Tinv.T
    f, Gq = _quartimin(L)
    G = -(L.T @ Gq @ Tinv).T
    alpha = 1.0
    history: list[float] = []
    T_prev = Gp_prev = None
    for _ in range(maxit + 1):
        Gp = G - T @ np.diag((T * G).sum(axis=0))
        s = float(np.sqrt((Gp ** 2).sum()))
        history.append(f)
        if s < eps:
            break
        if T_prev is not None:
            dT, dGp = T - T_prev, Gp - Gp_prev
            if float((dGp ** 2).sum()) > 0:
                alpha = float((dT ** 2).sum()) / abs(float((dT * dGp).sum()))
                alpha = max(1e-10, min(alpha, 20.0))
        else:
            alpha *= 2.0
        target = max(history[-window:])
        for _ in range(11):
            X = T - alpha * Gp
            Tt = X / np.sqrt((X ** 2).sum(axis=0))
            Ttinv = np.linalg.inv(Tt)
            Lt = A @ Ttinv.T
            ft, Gqt = _quartimin(Lt)
            if target - ft > 0.5 * s ** 2 * alpha:
                break
            alpha /= 2.0
        T_prev, Gp_prev = T, Gp
        T, Tinv, L, f, Gq = Tt, Ttinv, Lt, ft, Gqt
        G = -(L.T @ Gq @ Tinv).T
    return L, T.T @ T


@dataclass(frozen=True)
class Omega:
    """ω of one set of items, on the standardized items (as ``psych::omega`` prints it) and for the
    score as summed in the items' own units."""

    nfactors: int
    omega_total_standardized: float
    omega_h_standardized: float
    omega_total: float  # of the score: the item units
    omega_h: float
    alpha: float  # Cronbach's, on the covariance matrix (customary)
    general: np.ndarray = field(repr=False)  # each item's general-factor loading (standardized)
    communality: np.ndarray = field(repr=False)  # h², from the f-factor solution
    phi: np.ndarray | None = field(repr=False, default=None)  # the group factors' correlations
    caution: str | None = None

    def coefficient(self, structure: str) -> tuple[str, float]:
        """Ruling 8: ω-total of a unidimensional scale, ω-hierarchical of a multidimensional one."""
        if structure == "multidimensional":
            return "omega_hierarchical", self.omega_h
        return "omega_total", self.omega_total


def omega(items: np.ndarray, nfactors: int = 1) -> Omega:
    """McDonald's ω of the items (rows complete), as ``psych::omega(items, nfactors, flip =
    FALSE)`` computes it, plus the score's own (unstandardized) ω and Cronbach's α. ``nfactors``
    is the number of group factors: 1 for a unidimensional scale."""
    M = np.asarray(items, dtype=float)
    n, p = M.shape
    if p < MIN_ITEMS:
        raise ScaleRefused(f"ω needs at least {MIN_ITEMS} items; this scale has {p}.")
    if nfactors < 1 or (nfactors > 1 and p < ITEMS_PER_FACTOR * nfactors):
        raise ScaleRefused(f"{nfactors} group factors need at least {ITEMS_PER_FACTOR * nfactors} "
                           f"items; this scale has {p}.")
    if n <= p:
        raise ScaleRefused(f"{n:,} complete rows cannot estimate the correlations of {p} items.")
    C = np.cov(M, rowvar=False)
    sd = np.sqrt(np.diag(C))
    if np.any(~(sd > 0)):
        raise ScaleRefused("An item has the same answer on every row, so it has no correlations.")
    R = C / np.outer(sd, sd)
    L, _ = minres(R, nfactors)
    h2 = (L ** 2).sum(axis=1)
    u2 = 1.0 - h2
    phi = None
    caution = None
    if nfactors == 1:
        g = L[:, 0]
    else:
        orth = varimax(L)
        signs = np.sign(orth.sum(axis=0))
        signs[signs == 0] = 1.0
        orth = orth * signs
        order = np.argsort(-(orth ** 2).sum(axis=0), kind="stable")
        orth = orth[:, order]
        F, phi = oblimin(orth)
        if nfactors == 2:
            r = float(phi[0, 1])
            gamma = np.array([math.sqrt(abs(r)), math.copysign(math.sqrt(abs(r)), r)])
            caution = ("With two group factors the general factor is identified only by setting its "
                       "loadings on them equal (psych's default), so ω-hierarchical rests on that "
                       "constraint.")
        else:
            gl, _ = minres(phi, 1)
            gamma = gl[:, 0]
        g = F @ gamma
    Vt_std = float(R.sum())
    Vt = float(C.sum())
    return Omega(
        nfactors=int(nfactors),
        omega_total_standardized=(Vt_std - float(u2.sum())) / Vt_std,
        omega_h_standardized=float(g.sum()) ** 2 / Vt_std,
        omega_total=1.0 - float((sd ** 2 * u2).sum()) / Vt,
        omega_h=float((sd * g).sum()) ** 2 / Vt,
        alpha=cronbach_alpha(M),
        general=g, communality=h2, phi=phi, caution=caution)


# ── test–retest ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Retest:
    """Two administrations of the score: the two-way (subjects × occasions) analysis of
    variance, ICC(3,1) (Shrout & Fleiss 1979, consistency of a single measurement), and its
    residual mean square, the error variance of one administration."""

    icc: float
    error_variance: float  # MS_E
    n: int  # people with both administrations


def retest_reliability(first: np.ndarray, second: np.ndarray) -> Retest:
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    both = np.isfinite(a) & np.isfinite(b)
    a, b = a[both], b[both]
    n = int(both.sum())
    if n < 3:
        raise ScaleRefused("Fewer than three people have both administrations, so the test–retest "
                           "error cannot be estimated.")
    X = np.column_stack([a, b])
    grand = float(X.mean())
    ss_r = 2.0 * float(((X.mean(axis=1) - grand) ** 2).sum())
    ss_c = n * float(((X.mean(axis=0) - grand) ** 2).sum())
    ss_t = float(((X - grand) ** 2).sum())
    ms_r = ss_r / (n - 1)
    ms_e = (ss_t - ss_r - ss_c) / (n - 1)
    return Retest(icc=(ms_r - ms_e) / (ms_r + ms_e), error_variance=ms_e, n=n)


# ── regression calibration, conditional on the covariates ────────────────────


Fit = Callable[[np.ndarray, np.ndarray], np.ndarray]


def ols_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    from turbotab.core.methods.calibration import ols_fit as fit

    return fit(X, y)


def logistic_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    from turbotab.core.methods.calibration import logistic_fit as fit

    return fit(X, y)


def ordinal_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The proportional-odds slopes (``MASS::polr``'s parameterization), the outcome as codes."""
    from turbotab.core.models.ordinal import ProportionalOddsRegression

    return ProportionalOddsRegression().fit(np.asarray(X, dtype=float), np.asarray(y)).coef_


FITS: dict[str, Fit] = {"regression": ols_fit, "binary": logistic_fit, "ordinal": ordinal_fit}
APPROXIMATE = ("binary", "ordinal")  # the substitution of E[X | W, Z] is an approximation there


@dataclass(frozen=True)
class Calibrated:
    calibrated: np.ndarray  # E[X | W, Z] per row
    attenuation: float  # λ = slope of X on W given Z
    residual_variance: float  # s²_{W|Z}
    error_variance: float | None  # σ²_U (None for a calibration substudy)


@dataclass(frozen=True)
class JointCalibrated:
    """E[X | W, Z] for k scores calibrated together (multivariate regression calibration)."""

    calibrated: np.ndarray  # n × k
    attenuation: np.ndarray  # k: each score's own slope in its calibration
    residual_covariance: np.ndarray  # k × k: S, the scores' covariance given Z
    error_variances: tuple[float | None, ...]  # σ²_j (None for a calibration substudy)


def _design(Z: np.ndarray | None, n: int) -> np.ndarray:
    Zm = np.empty((n, 0)) if Z is None else np.asarray(Z, dtype=float).reshape(n, -1)
    return np.column_stack([np.ones(n), Zm])


def calibrate_jointly(W: np.ndarray, Z: np.ndarray | None, error_variances: Sequence[float | None],
                      references: Sequence[np.ndarray | None]) -> JointCalibrated:
    """E[X | W, Z] for the k columns of ``W`` (Rosner, Spiegelman & Willett 1990). Score j has a
    known error variance ``error_variances[j]`` (its ``references[j]`` None) or a calibration
    substudy's reference measure ``references[j]`` (blank outside the substudy). With classical
    errors independent of one another, of the true scores and of Z: each W on Z by least squares,
    S = RᵀR/(n − 1), D = diag(σ²_j) over the scores with a known error, and

    * a known error: E[X_j | W, Z] = fitted_j + (S − D)_j S⁻¹ r, its λ the j-th entry of that row;
    * a substudy: the reference regressed on (1, W, Z) in its rows, predicted for every row, its λ
      the coefficient on W_j.

    With k = 1 this is the univariate calibration: λ = (s²_{W|Z} − σ²_U)/s²_{W|Z}."""
    W = np.asarray(W, dtype=float)
    n = len(W)
    W = W.reshape(n, -1)
    k = W.shape[1]
    D_z = _design(Z, n)
    coef, *_ = np.linalg.lstsq(D_z, W, rcond=None)
    fitted = D_z @ coef
    resid = W - fitted
    S = resid.T @ resid / (n - 1)
    for j in range(k):
        if not S[j, j] > 0:
            raise ScaleRefused("The covariates predict the score exactly, so no part of it is left "
                               "to calibrate.", which=j)
    known = [j for j in range(k) if references[j] is None]
    D = np.zeros((k, k))
    for j in known:
        D[j, j] = float(error_variances[j])  # type: ignore[arg-type]
        if not S[j, j] - D[j, j] > 0:
            raise ScaleRefused(
                "The score's error variance is at least as large as its variance left after the "
                "covariates, so the true score's variance given them is not positive and no "
                "attenuation factor exists.", which=j)
    if len(known) > 1 and np.linalg.eigvalsh((S - D)[np.ix_(known, known)]).min() <= 0:
        raise ScaleRefused("Given the covariates, the scores' error variances leave their true "
                           "scores no positive-definite covariance, so they cannot be calibrated "
                           "together.")
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        raise ScaleRefused("Given the covariates, one score is an exact combination of the others, "
                           "so they cannot be calibrated together.") from None
    out = np.empty_like(W)
    lam = np.empty(k)
    for j in known:
        row = (S - D)[j] @ S_inv
        out[:, j] = fitted[:, j] + resid @ row
        lam[j] = row[j]
    Zm = D_z[:, 1:]
    for j in (j for j in range(k) if references[j] is not None):
        x = np.asarray(references[j], dtype=float)
        D_j = np.column_stack([np.ones(n), W, Zm])
        have = np.isfinite(x)
        if int(have.sum()) <= D_j.shape[1]:
            raise ScaleRefused(
                f"{int(have.sum()):,} rows have the reference measure, too few for a calibration "
                f"regression on the score{'s' if k > 1 else ''} and {Zm.shape[1]} covariates.",
                which=j)
        c, *_ = np.linalg.lstsq(D_j[have], x[have], rcond=None)
        if not c[1 + j] > 0:
            raise ScaleRefused("In the calibration substudy the reference measure does not rise "
                               "with the score once the covariates are allowed for, so there is no "
                               "attenuation factor to correct by.", which=j)
        out[:, j] = D_j @ c
        lam[j] = c[1 + j]
    return JointCalibrated(out, lam, S, tuple(None if references[j] is not None
                                              else float(error_variances[j])  # type: ignore[arg-type]
                                              for j in range(k)))


def calibrate_known_error(W: np.ndarray, Z: np.ndarray | None, error_variance: float) -> Calibrated:
    """E[X | W, Z] when W = X + U with a known error variance σ²_U: W on Z by least squares,
    λ = (s²_{W|Z} − σ²_U) / s²_{W|Z} with s²_{W|Z} = RSS/(n − 1), and fitted + λ·residual."""
    cal = calibrate_jointly(np.asarray(W, dtype=float)[:, None], Z, [error_variance], [None])
    return Calibrated(cal.calibrated[:, 0], float(cal.attenuation[0]),
                      float(cal.residual_covariance[0, 0]), float(error_variance))


def calibrate_reference(W: np.ndarray, Z: np.ndarray | None, reference: np.ndarray) -> Calibrated:
    """E[X | W, Z] from a calibration substudy: the reference measure X, observed on the substudy's
    rows, regressed on W and Z there; the prediction for every row."""
    cal = calibrate_jointly(np.asarray(W, dtype=float)[:, None], Z, [None], [reference])
    return Calibrated(cal.calibrated[:, 0], float(cal.attenuation[0]),
                      float(cal.residual_covariance[0, 0]), None)


@dataclass(frozen=True)
class Source:
    """Where a replicate's error variance comes from: ``reliability(rows) -> (value, σ²_U)`` for the
    internal consistency or a repeat administration, else a calibration substudy's ``reference``."""

    kind: Literal["internal_consistency", "test_retest", "calibration_substudy"]
    reliability: Callable[[np.ndarray], tuple[float, float]] | None = None
    reference: np.ndarray | None = None


@dataclass
class Correction:
    """One score's uncorrected and corrected coefficient, with the parts."""

    naive: float
    estimate: float
    se: float | None
    ci_low: float | None
    ci_high: float | None
    attenuation: float  # λ, conditional on the covariates (and on the other corrected scores)
    reliability: float  # the marginal reliability (ω, the ICC, or the calibration slope)
    error_variance: float | None
    n: int
    n_boot: int
    n_boot_ok: int
    boot: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))
    n_clusters: int | None = None  # the clusters the replicates resampled (None: rows)


def bootstrap_draws(n: int, n_boot: int, seed: int,
                    groups: np.ndarray | None = None) -> list[np.ndarray]:
    """The replicates' row indices, ``rng = numpy.random.default_rng(seed)``:

    * independent rows (``groups`` None): ``rng.integers(0, n, n)`` per replicate;
    * clustered rows (``groups``: each row's cluster, coded 0 … G − 1): ``rng.integers(0, G, G)``
      per replicate draws G clusters with replacement, and the replicate's rows are each drawn
      cluster's rows (in table order), the clusters in the order drawn."""
    rng = np.random.default_rng(seed)
    if groups is None:
        return [rng.integers(0, n, n) for _ in range(int(n_boot))]
    codes = np.asarray(groups, dtype=np.int64)
    if len(codes) != n:
        raise ValueError(f"The clusters cover {len(codes):,} rows, not {n:,}.")
    G = int(codes.max()) + 1 if n else 0
    order = np.argsort(codes, kind="stable")
    members = np.split(order, np.cumsum(np.bincount(codes, minlength=G))[:-1])
    return [np.concatenate([members[g] for g in rng.integers(0, G, G)]) for _ in range(int(n_boot))]


def _matrix(W: np.ndarray, Z: np.ndarray, js: Sequence[int]) -> np.ndarray:
    """The model's matrix with the k score columns ``W`` back at their positions ``js``."""
    n, k = W.shape
    M = np.empty((n, Z.shape[1] + k))
    placed = set(js)
    M[:, [i for i in range(M.shape[1]) if i not in placed]] = Z
    M[:, list(js)] = W
    return M


def _replicate(W: np.ndarray, Z: np.ndarray, y: np.ndarray, js: Sequence[int], fit: Fit,
               sources: Sequence[Source], rows: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float], list[float | None]]:
    """(naive, corrected, λ, reliabilities, σ²) of every score on ``rows`` (indices, with repeats)."""
    Wr, Zr, yr = W[rows], Z[rows], y[rows]
    errors: list[float | None] = []
    references: list[np.ndarray | None] = []
    found: list[float | None] = []
    for j, source in enumerate(sources):
        if source.kind == "calibration_substudy":
            errors.append(None)
            references.append(np.asarray(source.reference, dtype=float)[rows])
            found.append(None)
            continue
        try:
            value, error_variance = source.reliability(rows)  # type: ignore[misc]
        except ScaleRefused as refused:
            raise ScaleRefused(str(refused), which=j) from None
        errors.append(error_variance)
        references.append(None)
        found.append(float(value))
    cal = calibrate_jointly(Wr, Zr, errors, references)
    # A calibration substudy's reliability is its calibration slope.
    reliabilities = [float(cal.attenuation[j]) if r is None else r for j, r in enumerate(found)]
    naive = np.asarray(fit(_matrix(Wr, Zr, js), yr), dtype=float)[list(js)]
    corrected = np.asarray(fit(_matrix(cal.calibrated, Zr, js), yr), dtype=float)[list(js)]
    return naive, corrected, cal.attenuation, reliabilities, list(cal.error_variances)


def correct_jointly(W: np.ndarray, Z: np.ndarray, y: np.ndarray, js: Sequence[int],
                    sources: Sequence[Source], *, fit: Fit = ols_fit, n_boot: int = 200,
                    seed: int = 0, draws: Sequence[np.ndarray] | None = None,
                    groups: np.ndarray | None = None) -> list[Correction]:
    """The k scores ``W`` (columns ``js`` of the model's matrix, every other column in ``Z``)
    corrected together by regression calibration, each score's reliability re-estimated in every
    one of ``n_boot`` bootstrap replicates (:func:`bootstrap_draws`; ``groups``, each row's cluster,
    resamples whole clusters). ``draws`` (a list of row-index arrays) fixes the replicates. A
    score whose data refuse on every row raises :class:`ScaleRefused` naming it (``which``)."""
    W = np.asarray(W, dtype=float)
    n = len(W)
    W = W.reshape(n, -1)
    Z = np.asarray(Z, dtype=float).reshape(n, -1)
    y = np.asarray(y)
    k = W.shape[1]
    if len(js) != k or len(sources) != k:
        raise ValueError("Each corrected score needs its column and its source.")
    naive, estimate, lam, reliability, error_variance = _replicate(W, Z, y, js, fit, sources,
                                                                   np.arange(n))
    if draws is None:
        draws = bootstrap_draws(n, n_boot, seed, groups)
    boots = []
    for rows in draws:
        try:
            _, b, *_ = _replicate(W, Z, y, js, fit, sources, np.asarray(rows, dtype=np.int64))
        except (ScaleRefused, np.linalg.LinAlgError, ValueError):
            continue
        if np.all(np.isfinite(b)):
            boots.append(b)
    boot = np.asarray(boots, dtype=float).reshape(len(boots), k)
    n_clusters = None if groups is None else int(np.asarray(groups).max()) + 1
    out = []
    for j in range(k):
        se = float(np.std(boot[:, j], ddof=1)) if len(boot) >= 2 else None
        out.append(Correction(
            naive=float(naive[j]), estimate=float(estimate[j]), se=se,
            ci_low=None if se is None else float(estimate[j]) - Z_95 * se,
            ci_high=None if se is None else float(estimate[j]) + Z_95 * se,
            attenuation=float(lam[j]), reliability=float(reliability[j]),
            error_variance=error_variance[j], n=n, n_boot=int(len(draws)),
            n_boot_ok=int(len(boot)), boot=boot[:, j].copy(), n_clusters=n_clusters))
    return out


def correct(W: np.ndarray, Z: np.ndarray, y: np.ndarray, j: int, source: Source, *,
            fit: Fit = ols_fit, n_boot: int = 200, seed: int = 0,
            draws: Sequence[np.ndarray] | None = None,
            groups: np.ndarray | None = None) -> Correction:
    """The score ``W`` (column ``j`` of the model's matrix, every other column in ``Z``) corrected
    by regression calibration, the reliability re-estimated in each of ``n_boot`` bootstrap
    replicates. ``draws`` fixes the replicates; by default they are :func:`bootstrap_draws`'s
    (rows ``numpy.random.default_rng(seed).integers(0, n, n)`` per replicate; whole clusters when
    ``groups`` is given)."""
    (out,) = correct_jointly(np.asarray(W, dtype=float)[:, None], Z, y, [j], [source], fit=fit,
                             n_boot=n_boot, seed=seed, draws=draws, groups=groups)
    return out


def internal_consistency(keyed: np.ndarray, scoring: Scoring, structure: str,
                         nfactors: int) -> Callable[[np.ndarray], tuple[float, float]]:
    """A replicate's reliability and error variance from the items: ω (the whole factor analysis
    refit on the replicate's rows) and σ²_U = (1 − ω)·var(W)."""
    K = np.asarray(keyed, dtype=float)

    def of(rows: np.ndarray) -> tuple[float, float]:
        M = K[rows]
        est = omega(M, nfactors)
        _, value = est.coefficient(structure)
        W = score_items(M, scoring)
        return value, (1.0 - value) * float(np.var(W, ddof=1))

    return of


def retest_source(first: np.ndarray, second: np.ndarray) -> Callable[[np.ndarray], tuple[float, float]]:
    """A replicate's test–retest ICC and error variance (the two-way residual mean square), from
    the replicate's rows that have both administrations."""
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)

    def of(rows: np.ndarray) -> tuple[float, float]:
        r = retest_reliability(a[rows], b[rows])
        return r.icc, r.error_variance

    return of


__all__ = [
    "APPROXIMATE", "Calibrated", "Correction", "FITS", "JointCalibrated", "Omega", "Retest",
    "ScaleRefused", "ScaleScorer", "Source", "Z_95", "bootstrap_draws", "calibrate_jointly",
    "calibrate_known_error", "calibrate_reference", "correct", "correct_jointly",
    "cronbach_alpha", "describe",
    "internal_consistency", "keyed_items", "logistic_fit", "minres", "oblimin", "ols_fit", "omega",
    "ordinal_fit", "retest_reliability", "retest_source", "reverse_code", "score_items", "smc",
    "varimax",
]

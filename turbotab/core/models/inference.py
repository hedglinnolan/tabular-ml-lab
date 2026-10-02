"""Intervals that match how the rows were sampled (AUDIT_REPORT §5 WP2: MA-01, MA-06, MA-07, MA-08).

Under inference the linear family's coefficient table is made here, in four parts:

* **Which rows belong together** (:func:`resolve_clusters`) is read from the grain answer, the
  identifier roles and the analysis rows themselves, never from how the seal was drawn. A seal
  abandoned for too few units, or a grain answered "I don't know", still leaves an identifier that
  repeats, and intervals that ignore it are too narrow (Cameron & Miller 2015, *J Hum Resour*
  50:317). A missing identifier is its own unit, as the split maps it
  (``turbotab/core/seal.py::seal_inputs``).
* **Rows that repeat** get CR2 standard errors (bias-reduced linearization: Bell & McCaffrey 2002,
  *Surv Methodol* 28:169) and t reference distributions whose degrees of freedom are Bell and
  McCaffrey's Satterthwaite approximation under the working model of independent rows
  (Pustejovsky & Tipton 2018, *J Bus Econ Stat* 36:672; clubSandwich's ``test = "Satterthwaite"``).
  One engine, :func:`cr2`, serves OLS, logistic and multinomial models through their working
  linear models (McCaffrey & Bell 2006, *Stat Med* 25:4081). Below the unit floor no interval is
  reported: the table says why and offers the exits.
* **Independent rows** of a least-squares model get HC3 standard errors with t(n − p) reference
  distributions (Long & Ervin 2000, *Am Stat* 54:217), and a Breusch–Pagan check in Koenker's
  studentized form says when the residual spread is not constant.
* **Separation** in a logistic model is found by linear programming (Konis 2007, *Linear
  programming algorithms for detecting separated data in binary logistic regression models*,
  DPhil thesis, Oxford), named, and the table becomes Firth's penalized likelihood (Firth 1993,
  *Biometrika* 80:27) with profile penalized-likelihood intervals and likelihood-ratio p-values
  (Heinze & Schemper 2002, *Stat Med* 21:2409), whose estimates are finite.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd

LEVEL = 0.95
# Cameron & Miller (2015) §VI: "'few' may range from less than 20 clusters to less than 50
# clusters in the balanced case". Below this many, a concern names the number of clusters.
FEW_CLUSTERS = 30
# The Breusch–Pagan check speaks below this p-value. HC3 is used whatever it says; the check only
# tells the reader the spread is not constant, so it is kept strict to stay silent on clean data.
HETERO_ALPHA = 0.01
HETERO_MATERIAL = 0.10  # …and only when classical standard errors would be off by this much
FIRTH_MAX_STEP = 5.0  # the largest Newton step on any coefficient (logistf's ``maxstep``)
_BATCH = 1_000_000  # floats per block of clusters in the CR2 engine (8 MB)
# The Bell–McCaffrey df cost about G·P³ operations (6.8 × 10⁹ took 0.3 s). Beyond this (say
# 100,000 clusters and 100 columns) they are not computed: the intervals fall back to t(G − 1),
# and the caption says so.
BM_BUDGET = 1e11
_EIG_FLOOR = 1e-10  # 1 − λ below this is a zero eigenvalue of I − H_gg: pseudo-inverted (clubSandwich)

# CR1: the cluster sandwich with the G/(G − 1) correction, on t(G − 1), for a model with no
# working linear form here (the proportional-odds family, ``turbotab/core/models/ordinal.py``).
Covariance = Literal["HC3", "CR2", "CR1", "model", "profile", "none"]


def min_clusters() -> int:
    """The unit floor: the seal's own (``turbotab/core/seal.py::min_groups``), so the grain
    question, the seal and the intervals agree about how few units are too few."""
    from turbotab.core.seal import min_groups

    return min_groups()


# ── which rows belong together ───────────────────────────────────────────────


@dataclass(frozen=True)
class Clusters:
    """The clustering the intervals use, decided independently of the seal's grouping."""

    column: str | None = None
    codes: np.ndarray | None = None  # one integer per analysis row (aligned with them), or None
    n_clusters: int = 0
    n_missing: int = 0  # rows with no identifier: each is a unit of its own
    note: str | None = None  # the answers and the data disagree, said plainly
    refusal: str | None = None  # the rows repeat but nothing says which belong together
    exits: tuple[dict[str, Any], ...] = ()

    @property
    def clustered(self) -> bool:
        return self.codes is not None


INDEPENDENT = Clusters()


def cluster_columns(state: Any, available: Sequence[str], also: Sequence[str | None] = ()) -> list[str]:
    """Columns that may name the unit: the grain's named column first, then every identifier,
    then ``also`` (the column the split grouped by, when it is none of these)."""
    grain = getattr(state, "grain", None)
    named = getattr(grain, "id_column", None) if grain is not None else None
    identifiers = [c for c, r in (getattr(state, "roles", None) or {}).items() if r == "identifier"]
    have = set(available)
    wanted = [*([named] if named else []), *identifiers, *[c for c in also if c]]
    return [c for c in dict.fromkeys(wanted) if c in have]


def _missing(values: pd.Series) -> np.ndarray:
    return values.isna().to_numpy(dtype=bool)


def resolve_clusters(state: Any, frame: pd.DataFrame, also: Sequence[str | None] = ()) -> Clusters:
    """How the analysis rows cluster, for the intervals: by the unit column whenever one repeats.

    ``frame`` holds the candidate columns (:func:`cluster_columns`) over the analysis rows,
    indexed by row id. The grain's named column is used when it repeats; otherwise the repeating
    identifier with the fewest units (the coarsest grouping, as the seal reads it). A missing
    identifier is its own unit, keyed by its row id, exactly as the split keys it.
    """
    n = int(len(frame))
    grain = getattr(state, "grain", None)
    answer = getattr(grain, "grain", None) if grain is not None else None
    named = getattr(grain, "id_column", None) if grain is not None else None
    repeating: list[tuple[int, str]] = []
    for column in cluster_columns(state, list(frame.columns), also):
        values = frame[column]
        units = int(values.nunique(dropna=True))
        if 0 < units < int(values.notna().sum()):
            repeating.append((units, column))
    repeating.sort()
    chosen = next((c for _, c in repeating if c == named), repeating[0][1] if repeating else None)
    if chosen is not None:
        units = next(u for u, c in repeating if c == chosen)
        values = frame[chosen].astype(object)
        missing = _missing(values)
        keys = np.asarray(values, dtype=object).copy()
        keys[missing] = [f"__missing_{rid}" for rid in np.asarray(frame.index)[missing]]
        codes = pd.factorize(pd.Series(keys, dtype=object))[0].astype(np.int64)
        seen = f"`{chosen}` repeats ({units:,} values over {n:,} rows)"
        so = f"so the intervals are cluster-robust by `{chosen}`."
        note = None
        if answer == "unknown":
            note = (f"Whether a unit can appear in more than one row was answered as not known, "
                    f"but {seen}, {so}")
        elif answer == "one_row_per_unit":
            note = f"Each row was said to be a different unit, but {seen}, {so}"
        elif answer is None:
            note = f"{seen[0].upper()}{seen[1:]}, {so}"
        return Clusters(column=chosen, codes=codes, n_clusters=int(codes.max()) + 1 if n else 0,
                        n_missing=int(missing.sum()), note=note)
    if answer == "repeated" and not _combined(state) and (named is None or named not in frame.columns):
        said = f"by `{named}`, but the table has no such column" if named else \
            "but no column names the unit"
        return Clusters(
            refusal=(f"The rows were said to repeat {said}, so which rows belong together is "
                     f"unknown, and intervals that treat them as independent would be too narrow."),
            exits=({"label": "Name the column that identifies the unit (the grain question)",
                    "decision": None},))
    if answer == "unknown":
        return Clusters(note=("Whether a unit can appear in more than one row was answered as not "
                              "known, and no identifier repeats, so the intervals assume every row "
                              "is a different unit."))
    return INDEPENDENT


def _combined(state: Any) -> bool:
    """Each unit's rows were combined into one before the analysis (the unit question)."""
    return getattr(state, "unit", None) == "unit" and getattr(state, "aggregation", None) is not None


def floor_refusal(clusters: Clusters) -> tuple[str, tuple[dict[str, Any], ...]] | None:
    """Why no interval can be reported for these clusters, and the ways forward; None if one can."""
    if clusters.refusal:
        return clusters.refusal, clusters.exits
    if not clusters.clustered or clusters.n_clusters >= min_clusters():
        return None
    column = clusters.column
    return (
        f"`{column}` has {clusters.n_clusters} units, fewer than the {min_clusters()} TurboTab "
        f"requires for cluster-robust intervals: with so few, an interval rests on a handful of "
        f"unit-level residuals and its accuracy depends on how balanced the units are, while one "
        f"that ignores the repetition is far too narrow. No interval or p-value is reported.",
        ({"label": f"Combine each `{column}`'s rows into one (the unit question, before the seal)",
          "decision": None},
         {"label": "A random-intercept mixed model or GEE (not yet in TurboTab)", "decision": None}),
    )


# ── CR2 with Bell–McCaffrey degrees of freedom ───────────────────────────────


def cr2(X: np.ndarray, e: np.ndarray, codes: np.ndarray, bread: np.ndarray,
        *, adjust: bool = True) -> tuple[np.ndarray, np.ndarray | None]:
    """The CR2 covariance of a working linear model, and each coefficient's Bell–McCaffrey df.

    ``X`` (N × P) is the working design and ``e`` (N) the working residuals, both already on the
    scale where the working model is independent rows of equal variance (for OLS, the design and
    the residuals; for a GLM, ``W^½ X`` and the Pearson residuals). ``bread`` is ``(XᵀX)⁻¹``
    (a pseudo-inverse when singular). ``codes`` says which cluster each row is in.

    For each cluster ``A_g = (I − H_gg)^(−½)``, with ``H_gg = X_g (XᵀX)⁻¹ X_gᵀ`` (a pseudo-inverse
    square root when ``I − H_gg`` is singular, as clubSandwich does), and
    ``V = B (Σ_g X_gᵀ A_g e_g e_gᵀ A_g X_g) B``. ``H_gg`` has rank at most P, so ``A_g`` is
    applied through the eigenvectors of its P-dimensional part: no ``n_g × n_g`` matrix is formed.

    The degrees of freedom for coefficient j: with ``p_g = A_g X_g B e_j`` and the working model's
    residual-maker ``I − H``, the variance estimate is ``Σ_g (u_gᵀ ε)²`` with ``u_g = (I − H) p_g``;
    Satterthwaite's ν = tr(Ω)² / tr(Ω²) for ``Ω_gh = u_gᵀu_h = δ_gh |p_g|² − w_gᵀ B w_h``,
    ``w_g = X_gᵀ p_g``. ``adjust=False`` gives CR0 (``A_g = I``), for checks. The df are None
    when computing them would cost more than :data:`BM_BUDGET` (the caller then uses t(G − 1)).
    """
    X = np.asarray(X, dtype=float)
    e = np.asarray(e, dtype=float)
    N, P = X.shape
    blocks = _size_blocks(np.asarray(codes), P)
    # Pass 1: A_g X_g for every cluster, kept as one N × P array (the size of the design itself).
    AX = X.copy()
    meat = np.zeros((P, P))
    for block in blocks:
        Xg = X[block]  # (k, m, P)
        if adjust:
            Q, R = np.linalg.qr(Xg)  # X_g = Q R, Q with orthonormal columns
            Bg = R @ bread @ np.swapaxes(R, 1, 2)
            lam, Z = np.linalg.eigh((Bg + np.swapaxes(Bg, 1, 2)) / 2)
            gap = 1.0 - lam
            f = np.where(gap > _EIG_FLOOR, 1.0 / np.sqrt(np.clip(gap, _EIG_FLOOR, None)) - 1.0, -1.0)
            U = Q @ Z  # eigenvectors of H_gg (k, m, r)
            AX[block] = Xg + U @ (f[:, :, None] * (np.swapaxes(U, 1, 2) @ Xg))
        s = np.einsum("kmp,km->kp", AX[block], e[block])
        meat += s.T @ s
    V = bread @ meat @ bread
    G = sum(len(b) for b in blocks)
    if G * P ** 3 > BM_BUDGET:
        return (V + V.T) / 2, None
    # Pass 2: the Satterthwaite pieces, a slice of coefficients at a time (S_j is P × P each).
    T = AX @ bread  # row i, column j: p_g's entry for coefficient j
    df = np.full(P, np.nan)
    width = max(1, min(P, _BATCH // max(1, P * P)))
    for j0 in range(0, P, width):
        cols = slice(j0, min(P, j0 + width))
        J = cols.stop - cols.start
        sum_d = np.zeros(J)
        sum_k = np.zeros(J)
        sum_dd = np.zeros(J)
        sum_dk = np.zeros(J)
        S = np.zeros((J, P, P))  # S[j] = Σ_g w_g w_gᵀ
        for block in blocks:
            Tg = T[block][:, :, cols]  # (k, m, J)
            d = np.einsum("kmj,kmj->kj", Tg, Tg)
            Wg = np.swapaxes(X[block], 1, 2) @ Tg  # (k, P, J): w_g = X_gᵀ p_g
            k_gg = np.einsum("kaj,kaj->kj", Wg, bread @ Wg)
            sum_d += d.sum(0)
            sum_k += k_gg.sum(0)
            sum_dd += (d * d).sum(0)
            sum_dk += (d * k_gg).sum(0)
            Wj = np.transpose(Wg, (2, 0, 1))  # (J, k, P)
            S += np.swapaxes(Wj, 1, 2) @ Wj
        BS = bread[None, :, :] @ S
        tr_k2 = np.einsum("jab,jba->j", BS, BS)
        numerator = (sum_d - sum_k) ** 2
        denominator = sum_dd - 2.0 * sum_dk + tr_k2
        with np.errstate(divide="ignore", invalid="ignore"):
            df[cols] = np.where(denominator > 0, numerator / denominator, np.nan)
    return (V + V.T) / 2, df


def _size_blocks(codes: np.ndarray, P: int) -> list[np.ndarray]:
    """The rows of each cluster as (k, m) index arrays, clusters of one size m together, at most
    :data:`_BATCH` floats of design per array."""
    N = len(codes)
    order = np.argsort(codes, kind="stable")
    ranked = codes[order]
    starts = np.flatnonzero(np.r_[True, ranked[1:] != ranked[:-1]]) if N else np.zeros(0, int)
    sizes = np.diff(np.r_[starts, N])
    out = []
    for m in np.unique(sizes):
        rows = order[starts[sizes == m][:, None] + np.arange(m)[None, :]]
        per = max(1, _BATCH // max(1, m * P + P * P))
        out.extend(rows[lo:lo + per] for lo in range(0, len(rows), per))
    return out


def bread_of(X: np.ndarray) -> np.ndarray:
    return np.linalg.pinv(X.T @ X)


def logistic_working(X: np.ndarray, y: np.ndarray, mu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A logistic fit's working linear model at its estimate: ``W^½ X`` and the Pearson residuals
    ``(y − μ)/√w`` (w = μ(1 − μ)). Its ``(XᵀX)⁻¹`` is the inverse information, and its meat is the
    sum of score outer products, so :func:`cr2` gives the logistic sandwich."""
    root = np.sqrt(np.clip(mu * (1.0 - mu), 1e-300, None))
    return X * root[:, None], (y - mu) / root


def multinomial_working(X: np.ndarray, outcome: np.ndarray, probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A multinomial logit's working linear model: each row becomes q = K − 1 rows,
    ``W_i^½ (I_q ⊗ x_iᵀ)`` and ``W_i^(−½) (y_i − π_i)`` with ``W_i = diag(π_i) − π_i π_iᵀ`` over the
    non-reference classes. Coefficients are ordered class by class (each class's P together).
    ``outcome`` holds class codes 0…K − 1 (0 the reference); ``probs`` is n × K."""
    n, P = X.shape
    q = probs.shape[1] - 1
    pi = probs[:, 1:]
    W = np.einsum("ia,ab->iab", pi, np.eye(q)) - pi[:, :, None] * pi[:, None, :]
    lam, vec = np.linalg.eigh(W)
    lam = np.clip(lam, 1e-300, None)
    half = vec @ (np.sqrt(lam)[:, :, None] * np.swapaxes(vec, 1, 2))
    inv_half = vec @ ((1.0 / np.sqrt(lam))[:, :, None] * np.swapaxes(vec, 1, 2))
    onehot = (np.asarray(outcome)[:, None] == np.arange(1, q + 1)[None, :]).astype(float)
    Xw = np.einsum("iab,ij->iabj", half, X).reshape(n * q, q * P)
    ew = np.einsum("iab,ib->ia", inv_half, onehot - pi).reshape(n * q)
    return Xw, ew


# ── separation and Firth ─────────────────────────────────────────────────────


def separated_columns(X: np.ndarray, y: np.ndarray) -> list[int]:
    """Indices of the coefficients whose maximum-likelihood estimate is infinite; [] when none.

    Konis (2007): the data are separated when some β ≠ 0 has ``s_i x_iᵀβ ≥ 0`` for every row
    (``s_i = ±1`` for an event or not) with ``Xβ ≠ 0``. With every column scaled to at most one in
    absolute value and β boxed in [−1, 1], one linear program maximizing ``Σ s_i x_iᵀβ`` finds
    such a direction when one exists; coefficient j is infinite when some direction in that cone
    moves it (one linear program each way). Every direction a solver returns is checked again on
    the rows before it is believed. With a rank-deficient design, directions with ``Xβ = 0`` make
    the per-coefficient reading undefined, so nothing is reported (the collinearity concern says
    why the table cannot be computed).
    """
    from scipy.optimize import linprog

    X = np.asarray(X, dtype=float)
    s = np.where(np.asarray(y, dtype=float) > 0.5, 1.0, -1.0)
    n, P = X.shape
    scale = np.abs(X).max(axis=0)
    scale[scale == 0] = 1.0
    Xs = X / scale
    if n <= P or np.linalg.matrix_rank(Xs) < P:
        return []
    A = -(s[:, None] * Xs)
    b = np.zeros(n)
    bounds = [(-1.0, 1.0)] * P

    def direction(c: np.ndarray) -> np.ndarray | None:
        result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
        if result.status != 0 or result.x is None:
            return None
        beta = np.where(np.abs(result.x) < 1e-9, 0.0, result.x)
        margins = s * (Xs @ beta)
        return beta if margins.min() >= -1e-7 and margins.max() > 1e-6 else None

    if direction(-(s @ Xs)) is None:
        return []
    infinite = []
    for j in range(P):
        for sign in (1.0, -1.0):
            c = np.zeros(P)
            c[j] = -sign
            beta = direction(c)
            if beta is not None and sign * beta[j] > 1e-6:
                infinite.append(j)
                break
    return infinite


@dataclass
class FirthFit:
    beta: np.ndarray
    loglik: float  # the penalized log-likelihood at beta
    cov: np.ndarray  # the inverse Fisher information at beta
    iterations: int
    converged: bool


def _firth_parts(X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> tuple[Any, ...]:
    eta = X @ beta
    mu = 0.5 * (1.0 + np.tanh(0.5 * eta))  # the logistic function, without overflow
    w = mu * (1.0 - mu)
    info = X.T @ (X * w[:, None])
    sign, logdet = np.linalg.slogdet(info)
    loglik = float(np.sum(y * eta - np.logaddexp(0.0, eta)))
    penalized = loglik + 0.5 * logdet if sign > 0 else -np.inf
    return mu, w, info, penalized


def firth_fit(X: np.ndarray, y: np.ndarray, *, fixed: dict[int, float] | None = None,
              start: np.ndarray | None = None, max_iter: int = 300, tol: float = 1e-8) -> FirthFit:
    """Firth's bias-reduced logistic regression: the maximum of ``l(β) + ½ log |I(β)|``.

    Newton steps on the modified score ``Xᵀ(y − μ + h (½ − μ))`` (``h`` the hat values of the
    weighted fit), which is the exact gradient of the penalized log-likelihood, with steps capped
    at :data:`FIRTH_MAX_STEP` and halved until the penalized log-likelihood does not fall
    (logistf's algorithm). ``fixed`` holds coefficients at given values (the profile). Converged
    when the Newton decrement ``Uᵀ I⁻¹ U`` (invariant to the columns' units) is below ``tol``².
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    P = X.shape[1]
    beta = np.zeros(P) if start is None else np.array(start, dtype=float)
    free = np.ones(P, dtype=bool)
    for j, value in (fixed or {}).items():
        free[j] = False
        beta[j] = value
    mu, w, info, ll = _firth_parts(X, y, beta)
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        inv = np.linalg.pinv(info)
        h = w * np.einsum("ij,jk,ik->i", X, inv, X)
        score = X.T @ (y - mu + h * (0.5 - mu))
        delta = np.zeros(P)
        delta[free] = np.linalg.lstsq(info[np.ix_(free, free)], score[free], rcond=None)[0]
        if float(score[free] @ delta[free]) < tol * tol:
            converged = True
            break
        biggest = float(np.abs(delta).max()) if P else 0.0
        if biggest > FIRTH_MAX_STEP:
            delta *= FIRTH_MAX_STEP / biggest
        for _ in range(40):
            trial = beta + delta
            parts = _firth_parts(X, y, trial)
            if parts[3] >= ll - 1e-12 * max(1.0, abs(ll)):
                break
            delta /= 2.0
        beta = trial
        mu, w, info, ll = parts
    return FirthFit(beta=beta, loglik=float(ll), cov=np.linalg.pinv(info), iterations=it,
                    converged=converged)


def firth_profile(X: np.ndarray, y: np.ndarray, fit: FirthFit, j: int, *,
                  level: float = LEVEL) -> tuple[float | None, float | None, float]:
    """Coefficient j's profile penalized-likelihood interval and likelihood-ratio p-value.

    The interval is every b with ``2 (l*(β̂) − max_{β: β_j = b} l*(β)) ≤ χ²₁(level)``
    (Heinze & Schemper 2002); a bound that is not reached within a wide search is None.
    """
    from scipy.optimize import brentq
    from scipy.stats import chi2

    crit = 0.5 * float(chi2.ppf(level, 1))
    centre = float(fit.beta[j])
    warm = {"beta": fit.beta.copy()}

    def drop(b: float) -> float:
        profile = firth_fit(X, y, fixed={j: b}, start=warm["beta"])
        warm["beta"] = profile.beta
        return fit.loglik - profile.loglik

    se = math.sqrt(max(float(fit.cov[j, j]), 1e-12))
    bounds: list[float | None] = []
    for side in (-1.0, 1.0):
        warm["beta"] = fit.beta.copy()
        inner, reach = centre, 2.0 * se
        bound = None
        for _ in range(40):
            outer = centre + side * reach
            if drop(outer) >= crit:
                bound = brentq(lambda b: drop(b) - crit, min(inner, outer), max(inner, outer),
                               xtol=1e-10, rtol=1e-10)
                break
            inner, reach = outer, reach * 2.0
        bounds.append(bound)
    warm["beta"] = fit.beta.copy()
    statistic = max(0.0, 2.0 * drop(0.0))
    return bounds[0], bounds[1], float(chi2.sf(statistic, 1))


# ── the table ────────────────────────────────────────────────────────────────


@dataclass
class InferenceTable:
    """The coefficient rows, how their intervals were made (the ``inference`` artifact field), and
    the concerns the fit states first."""

    rows: list[dict[str, Any]]
    info: dict[str, Any]
    concerns: list[str] = field(default_factory=list)
    # The covariance the intervals rest on, over the rows in order (None when refused or when the
    # intervals are not Wald: Firth's profile ones). Joint tests read it (WP12a: a spline's
    # nonlinearity, ``turbotab/core/methods/exposure_form.py``); it is never serialized.
    cov: Any = None


def _clean(value: Any) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _row(name: str, estimate: float, se: float | None = None, df: float | None = None,
         low: float | None = None, high: float | None = None, p: float | None = None) -> dict[str, Any]:
    return {"feature": name, "estimate": _clean(estimate), "ci_low": None if low is None else _clean(low),
            "ci_high": None if high is None else _clean(high), "p": None if p is None else _clean(p),
            "se": None if se is None else _clean(se), "df": None if df is None else _clean(df)}


def _t_rows(names: Sequence[str], est: np.ndarray, se: np.ndarray, df: np.ndarray) -> list[dict[str, Any]]:
    """Rows with t intervals and two-sided p-values on each coefficient's own df."""
    from scipy import stats

    rows = []
    for name, b, s, v in zip(names, est, se, df):
        if not (np.isfinite(s) and s > 0 and np.isfinite(v) and v > 0):
            rows.append(_row(name, b, s if np.isfinite(s) else None))
            continue
        q = float(stats.t.ppf(0.5 + LEVEL / 2, v))
        p = float(2.0 * stats.t.sf(abs(b / s), v))
        rows.append(_row(name, b, s, v, b - q * s, b + q * s, p))
    return rows


def _z_rows(names: Sequence[str], est: np.ndarray, se: np.ndarray) -> list[dict[str, Any]]:
    from scipy import stats

    q = float(stats.norm.ppf(0.5 + LEVEL / 2))
    rows = []
    for name, b, s in zip(names, est, se):
        if not (np.isfinite(s) and s > 0):
            rows.append(_row(name, b, None))
            continue
        rows.append(_row(name, b, s, None, b - q * s, b + q * s, float(2.0 * stats.norm.sf(abs(b / s)))))
    return rows


def _info(estimator: str, covariance: Covariance, caption: str, clusters: Clusters,
          **extra: Any) -> dict[str, Any]:
    return {"estimator": estimator, "covariance": covariance, "caption": caption,
            "grouped_by": clusters.column, "n_clusters": clusters.n_clusters if clusters.clustered else None,
            "n_missing_ids": clusters.n_missing, "separated": [], "refused": None, "exits": [], **extra}


def _cluster_caption(clusters: Clusters, bell_mccaffrey: bool = True) -> str:
    missing = (f", {clusters.n_missing:,} rows with no `{clusters.column}` counted as units of their "
               f"own" if clusters.n_missing else "")
    reference = ("on t with Bell–McCaffrey degrees of freedom" if bell_mccaffrey else
                 f"on t({clusters.n_clusters - 1:,}): Bell–McCaffrey degrees of freedom are too costly "
                 f"to compute at this size")
    return (f"95% intervals cluster-robust (CR2) by `{clusters.column}`, G = "
            f"{clusters.n_clusters:,} clusters{missing}, {reference}.")


def _clustered(names: Sequence[str], est: np.ndarray, estimator: str, X: np.ndarray, e: np.ndarray,
               codes: np.ndarray, clusters: Clusters) -> InferenceTable:
    """The CR2 table of a working linear model (``X``, ``e``), rows clustered by ``codes``."""
    V, df = cr2(X, e, codes, bread_of(X))
    exact = df is not None
    if df is None:
        df = np.full(len(est), float(clusters.n_clusters - 1))
    rows = _t_rows(names, est, np.sqrt(np.clip(np.diag(V), 0, None)), df)
    return InferenceTable(rows, _info(estimator, "CR2", _cluster_caption(clusters, exact), clusters),
                          _cluster_concerns(clusters), cov=V)


def _cluster_concerns(clusters: Clusters) -> list[str]:
    out = [clusters.note] if clusters.note else []
    G = clusters.n_clusters
    if clusters.note is None:
        out.append(f"Intervals are cluster-robust by `{clusters.column}` (CR2, G = {G:,}), because "
                   f"its rows repeat.")
    if G < FEW_CLUSTERS:
        out.append(f"Only {G} `{clusters.column}` clusters: the intervals use the CR2 small-sample "
                   f"correction with Bell–McCaffrey degrees of freedom, which Cameron & Miller "
                   f"(2015) recommend when clusters are few; a random-intercept mixed model (not "
                   f"yet in TurboTab) is the usual alternative.")
    return out


def _refused(names: Sequence[str], est: np.ndarray, estimator: str, clusters: Clusters,
             reason: str, exits: Sequence[dict[str, Any]]) -> InferenceTable:
    rows = [_row(n, b) for n, b in zip(names, est)]
    info = _info(estimator, "none", f"No intervals: {reason}", clusters,
                 refused=reason, exits=[dict(e) for e in exits])
    return InferenceTable(rows, info, [reason])


def format_p(p: float) -> str:
    """A p-value as a reader writes it: 0.004, or 3 × 10⁻⁸ below 0.001."""
    if p >= 0.001:
        return f"{p:.3f}"
    if p <= 0:
        return "< 10⁻³⁰⁰"
    exponent = math.floor(math.log10(p))
    mantissa = p / 10 ** exponent
    sup = str(exponent).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{mantissa:.0f} × 10{sup}"


def heteroskedasticity_concern(resid: np.ndarray, X: np.ndarray, names: Sequence[str],
                               classical: np.ndarray, robust: np.ndarray) -> str | None:
    """A sentence when the residual spread changes with the predictors and that matters here:
    Koenker's studentized Breusch–Pagan test below :data:`HETERO_ALPHA`, and classical standard
    errors off from the HC3 ones by at least :data:`HETERO_MATERIAL` for some coefficient."""
    from statsmodels.stats.diagnostic import het_breuschpagan

    if X.shape[1] < 2:
        return None
    try:
        _, p, _, _ = het_breuschpagan(resid, X, robust=True)
    except Exception:  # noqa: BLE001 - a check that cannot run is not a verdict
        return None
    if not np.isfinite(p) or p >= HETERO_ALPHA:
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.asarray(classical, float) / np.asarray(robust, float)
    keep = [j for j, n in enumerate(names) if n != "(intercept)" and np.isfinite(ratio[j]) and ratio[j] > 0]
    if not keep:
        return None
    j = max(keep, key=lambda i: abs(math.log(ratio[i])))
    change = ratio[j] - 1.0
    if abs(change) < HETERO_MATERIAL:
        return None
    return (f"The residual spread is not constant (Breusch–Pagan p = {format_p(float(p))}): "
            f"classical standard errors would be {abs(change):.0%} "
            f"{'larger' if change > 0 else 'smaller'} for `{names[j]}`; these intervals use HC3 "
            f"robust standard errors, which allow for it.")


def inference_table(task: str, matrix: pd.DataFrame, y: Any, classes: Sequence[Any] | None,
                    clusters: Clusters = INDEPENDENT) -> InferenceTable:
    """The inference coefficient table for the linear family on the model matrix it saw."""
    import statsmodels.api as sm

    if clusters.clustered and len(clusters.codes) != len(matrix):
        raise ValueError(f"The clusters cover {len(clusters.codes):,} rows but the model matrix has "
                         f"{len(matrix):,}.")
    # The fits are handed the indexed frame, so which rows they saw stays checkable.
    exog = sm.add_constant(matrix.astype(float), has_constant="add")
    names = ["(intercept)" if c == "const" else str(c) for c in exog.columns]
    if task == "regression":
        return _least_squares(names, exog, np.asarray(y, dtype=float), clusters)
    if task == "binary":
        if classes is None or len(classes) != 2:
            raise ValueError("A binary outcome needs exactly two classes.")
        event = (np.asarray(y) == classes[1]).astype(float)
        return _logistic(names, exog, event, clusters)
    return _multinomial(names, exog, np.asarray(y), list(classes or []), clusters)


def _least_squares(names: list[str], exog: pd.DataFrame, y: np.ndarray,
                   clusters: Clusters) -> InferenceTable:
    import statsmodels.api as sm

    X = exog.to_numpy(dtype=float)
    fit = sm.OLS(y, exog).fit()
    est = np.asarray(fit.params, dtype=float)
    estimator = "ordinary least squares"
    refusal = floor_refusal(clusters)
    if refusal:
        return _refused(names, est, estimator, clusters, *refusal)
    if clusters.clustered:
        return _clustered(names, est, estimator, X, np.asarray(fit.resid, dtype=float),
                          clusters.codes, clusters)
    hc3 = fit.get_robustcov_results(cov_type="HC3", use_t=True)
    se = np.asarray(hc3.bse, dtype=float)
    dof = float(hc3.df_resid)
    rows = _t_rows(names, est, se, np.full(len(est), dof))
    concerns = [clusters.note] if clusters.note else []
    hetero = heteroskedasticity_concern(np.asarray(fit.resid, float), X, names,
                                        np.asarray(fit.bse, float), se)
    if hetero:
        concerns.append(hetero)
    caption = f"95% intervals from HC3 heteroskedasticity-robust standard errors, on t({dof:,.0f})."
    return InferenceTable(rows, _info(estimator, "HC3", caption, clusters), concerns,
                          cov=np.asarray(hc3.cov_params(), dtype=float))


def _named(names: Sequence[str], idx: Sequence[int]) -> list[str]:
    return [names[j] for j in idx if names[j] != "(intercept)"]


def _and(items: Sequence[str]) -> str:
    quoted = [f"`{i}`" for i in items]
    return quoted[0] if len(quoted) == 1 else f"{', '.join(quoted[:-1])} and {quoted[-1]}"


def _logistic(names: list[str], exog: pd.DataFrame, y: np.ndarray, clusters: Clusters) -> InferenceTable:
    import statsmodels.api as sm

    X = exog.to_numpy(dtype=float)
    separated = _named(names, separated_columns(X, y))
    if separated:
        return _firth_table(names, X, y, clusters, separated)
    fit = sm.Logit(y, exog).fit(disp=0, maxiter=200)
    est = np.asarray(fit.params, dtype=float)
    estimator = "logistic regression (maximum likelihood)"
    refusal = floor_refusal(clusters)
    if refusal:
        return _refused(names, est, estimator, clusters, *refusal)
    if clusters.clustered:
        Xw, ew = logistic_working(X, y, np.asarray(fit.predict(X), dtype=float))
        return _clustered(names, est, estimator, Xw, ew, clusters.codes, clusters)
    rows = _z_rows(names, est, np.asarray(fit.bse, dtype=float))
    concerns = [clusters.note] if clusters.note else []
    return InferenceTable(rows, _info(estimator, "model",
                                      "95% Wald intervals from the logistic model's information.",
                                      clusters), concerns,
                          cov=np.asarray(fit.cov_params(), dtype=float))


def _firth_table(names: list[str], X: np.ndarray, y: np.ndarray, clusters: Clusters,
                 separated: list[str]) -> InferenceTable:
    fit = firth_fit(X, y)
    est = fit.beta
    estimator = "Firth-penalized logistic regression"
    verb = "separates" if len(separated) == 1 else "separate"
    concern = (f"{_and(separated)} {verb} the outcome (the maximum-likelihood log-odds is infinite), "
               f"so a Wald interval or p-value for {'it' if len(separated) == 1 else 'them'} would be "
               f"meaningless; the table reports Firth-penalized logistic regression instead "
               f"(Heinze & Schemper 2002).")
    extra = {"separated": separated}
    refusal = floor_refusal(clusters)
    if refusal is None and clusters.clustered:
        refusal = (f"{_and(separated)} {verb} the outcome and the rows repeat by `{clusters.column}`: "
                   f"Firth's profile intervals assume independent rows, so no interval or p-value "
                   f"is reported.",
                   ({"label": f"Combine each `{clusters.column}`'s rows into one (the unit question, "
                              f"before the seal)", "decision": None},
                    {"label": f"Leave {_and(separated)} out, or merge its rare levels", "decision": None}))
    if refusal:
        table = _refused(names, est, estimator, clusters, *refusal)
        table.info.update(extra)
        table.concerns.insert(0, concern)
        return table
    se = np.sqrt(np.clip(np.diag(fit.cov), 0, None))
    rows = []
    for j, name in enumerate(names):
        low, high, p = firth_profile(X, y, fit, j)
        rows.append(_row(name, est[j], se[j], None, low, high, p))
    caption = ("95% profile penalized-likelihood intervals of a Firth-penalized logistic regression, "
               "because the maximum-likelihood estimate does not exist; p-values from penalized "
               "likelihood-ratio tests.")
    concerns = [concern] + ([clusters.note] if clusters.note else [])
    if not fit.converged:
        concerns.append("Firth's penalized fit stopped before converging; treat these numbers with care.")
    return InferenceTable(rows, _info(estimator, "profile", caption, clusters, **extra), concerns)


def _multinomial(names: list[str], exog: pd.DataFrame, y: np.ndarray, classes: list[Any],
                 clusters: Clusters) -> InferenceTable:
    import statsmodels.api as sm

    X = exog.to_numpy(dtype=float)
    codes = pd.Categorical(y, categories=classes).codes
    fit = sm.MNLogit(codes, exog).fit(disp=0, maxiter=200)
    B = np.asarray(fit.params, dtype=float)  # (P, K − 1), against the first class
    q = B.shape[1]
    labels = [f"{n} [{classes[k + 1] if classes else k + 1}]" for k in range(q) for n in names]
    est = B.ravel(order="F")  # class by class: the order of every vector below
    estimator = "multinomial logistic regression (maximum likelihood)"
    refusal = floor_refusal(clusters)
    if refusal:
        return _refused(labels, est, estimator, clusters, *refusal)
    if clusters.clustered:
        Xw, ew = multinomial_working(X, codes, np.asarray(fit.predict(X), dtype=float))
        return _clustered(labels, est, estimator, Xw, ew, np.repeat(clusters.codes, q), clusters)
    se = np.asarray(fit.bse, dtype=float).ravel(order="F")
    concerns = [clusters.note] if clusters.note else []
    return InferenceTable(_z_rows(labels, est, se),
                          _info(estimator, "model",
                                "95% Wald intervals from the multinomial model's information.",
                                clusters), concerns)


__all__ = [
    "Clusters", "FEW_CLUSTERS", "FirthFit", "INDEPENDENT", "InferenceTable", "bread_of",
    "cluster_columns", "cr2", "logistic_working", "multinomial_working",
    "firth_fit", "firth_profile", "floor_refusal", "format_p", "heteroskedasticity_concern",
    "inference_table", "min_clusters", "resolve_clusters", "separated_columns",
]

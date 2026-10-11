"""``cox``: the Cox proportional hazards model, for a time-to-event outcome (AUDIT_REPORT §5 WP12;
the family RO-03 routes to).

A time-to-event outcome is three columns: the event (the outcome column, coded 1 for the event
the user named, ``set_event``), the time at which each row's follow-up ended, at the event or at
censoring, and, with staggered or delayed entry, the time at which each row came under
observation on the same scale (``set_follow_up``). A row is at risk at time t when
``entry < t ≤ time``, so a row entering late is never counted at risk before it was observed
(left truncation; "staggered entry"). PROBAST (Moons et al. 2019, *Ann Intern Med* 170:W1,
item 4.6): "For prognostic models to predict long-term outcomes in which censoring occurs, a
time-to-event analysis, such as a Cox regression, should be used."

The model is fit here, by Newton–Raphson on Cox's partial likelihood with Efron's handling of
tied event times (Efron 1977, *J Am Stat Assoc* 72:557), the default of R's ``coxph`` and of
lifelines. Every risk-set sum is a cumulative sum over rows sorted by time (and by entry), so a
fit costs O(n log n + n·P²). Under inference:

* independent rows get Wald intervals from the partial likelihood's information, as ``coxph``
  reports them;
* rows that repeat within a unit get the Lin–Wei sandwich by unit (Lin & Wei 1989, *J Am Stat
  Assoc* 84:1074; ``coxph(cluster = …)``) from Efron-consistent score residuals, on t(G − 1) as
  Cameron & Miller (2015, *J Hum Resour* 50:317) ask of any cluster-robust interval at a minimum,
  refused below the unit floor as every sandwich interval here is;
* each coefficient's proportional-hazards assumption is checked by Grambsch & Therneau's (1994,
  *Biometrika* 81:515) approximate score test on scaled Schoenfeld residuals against the event
  order (lifelines' ``proportional_hazard_test(time_transform="rank")``).

Estimates are log hazard ratios: exp(estimate) is the hazard ratio per unit of the column.
Performance is Harrell's concordance index (Harrell et al. 1996, *Stat Med* 15:361): among pairs
whose order of events is known, the share the model's risk score orders correctly, ties in the
score counting one half.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from turbotab.core.decisions import Purpose, Task
from turbotab.core.models.base import (
    Assessment,
    FamilyBase,
    Identity,
    InferenceDecl,
    Situation,
    coefficient_rows,
    register_family,
)

OUTCOME_DTYPE = np.dtype([("event", bool), ("time", float), ("entry", float)])
MAX_ITER = 60
# The Newton decrement UᵀI⁻¹U (on standardized columns) at which the fit has converged: β is then
# within about √TOL of the maximum. Rounding in U is far below this even at 10⁵ events.
TOL = 1e-18
LOOSE_TOL = 1e-10  # after MAX_ITER steps, still accepted as converged below this
DIVERGED = 20.0  # |β_j| · SD(x_j) beyond this: a log hazard ratio of 20 per SD is an infinite one
PH_ALPHA = 0.01  # the proportional-hazards check speaks below this, shared over the columns
EPV_RULE = 10  # events per predictor, a rule of thumb for Cox models (Peduzzi et al. 1995)


# ── the outcome ──────────────────────────────────────────────────────────────


def survival_outcome(event: Any, time: Any, entry: Any = None) -> np.ndarray:
    """The outcome as one structured array per row: ``event`` (bool), ``time``, ``entry``."""
    event = np.asarray(event)
    out = np.empty(len(event), dtype=OUTCOME_DTYPE)
    out["event"] = event.astype(float) > 0.5
    out["time"] = np.asarray(time, dtype=float)
    out["entry"] = 0.0 if entry is None else np.asarray(entry, dtype=float)
    return out


def follow_up_columns(state: Any) -> list[str]:
    """The follow-up time column, and the entry column when one was named."""
    spec = getattr(state, "follow_up", None)
    if spec is None:
        return []
    return [spec.time_column] + ([spec.entry_column] if spec.entry_column else [])


def time_to_event_outcome(state: Any, frame: pd.DataFrame, event: Any) -> np.ndarray:
    """The structured outcome of the analysis rows: ``event`` coded 0/1 (the event the user named,
    as the fit stage codes a binary outcome), with the follow-up the state names.

    Raises ValueError, saying what to do, when the follow-up is not named, a row has no follow-up
    time, an entry is not before its row's end of follow-up, or the event is not coded.
    """
    spec = getattr(state, "follow_up", None)
    target = getattr(state, "target", None)
    if spec is None:
        raise ValueError(f"`{target}` is a time-to-event outcome, but no column was named as its "
                         f"follow-up time (the follow-up question).")
    values = pd.Series(np.asarray(event, dtype=object))
    coded = pd.to_numeric(values, errors="coerce")
    if coded.isna().any() or not set(coded.unique()) <= {0, 1}:
        levels = ", ".join(f"`{v}`" for v in sorted({str(v) for v in values.unique()})[:4])
        raise ValueError(f"`{target}` holds {levels}: name which level is the event (the event "
                         f"question), so the event can be coded 1.")
    time = pd.to_numeric(frame[spec.time_column], errors="coerce").to_numpy(dtype=float)
    entry = (pd.to_numeric(frame[spec.entry_column], errors="coerce").to_numpy(dtype=float)
             if spec.entry_column else np.zeros(len(time)))
    coded_event = coded.to_numpy(dtype=float)
    horizon = getattr(spec, "horizon", None)
    if horizon is not None:
        # Follow-up ends at the horizon: an event after it is censored there.
        coded_event = np.where(time > float(horizon), 0.0, coded_event)
        time = np.minimum(time, float(horizon))
    landmark = getattr(spec, "landmark", None)
    if landmark is not None:
        # Follow-up counted from the landmark: every row is at risk from it (the cohort has left
        # out the rows whose follow-up ended by then; ``stages.rows.cohort_flow``).
        entry = np.maximum(entry, float(landmark))
    unknown = ~np.isfinite(time) | ~np.isfinite(entry)
    if unknown.any():
        named = f"`{spec.time_column}`" + (f" or `{spec.entry_column}`" if spec.entry_column else "")
        raise ValueError(f"{int(unknown.sum()):,} analysis rows have no {named}: a row whose "
                         f"follow-up is unknown cannot be placed in any risk set. Exclude them "
                         f"(an eligibility rule on {named} that drops unknown values).")
    bad = entry >= time
    if bad.any():
        what = (f"end their follow-up by the landmark `{float(landmark):g}`" if landmark is not None
                else f"start at or after their `{spec.time_column}`" if spec.entry_column
                else f"have a `{spec.time_column}` of zero or less")
        raise ValueError(f"{int(bad.sum()):,} analysis rows {what}, so they are never at risk.")
    return survival_outcome(coded_event, time, entry)


# ── the partial likelihood (Efron ties, delayed entry) ───────────────────────


@dataclass
class _Risk:
    """The risk-set sums at each event slot (one slot per event, Efron's l = 0 … d − 1)."""

    order: np.ndarray  # event rows, by time
    k_of: np.ndarray  # each event row's distinct event time (index into ``times``)
    times: np.ndarray  # distinct event times
    frac: np.ndarray  # each slot's l / d
    S0: np.ndarray  # (d,)
    S1: np.ndarray  # (d, P)
    loglik: float
    # Each slot's weight: the mean case weight of its time's tied events (1 without weights), as
    # R's coxph weights Efron's slots (``meanwt``).
    mw: np.ndarray | None = None


def _suffix(values: np.ndarray, sorted_keys: np.ndarray, at: np.ndarray, side: str) -> np.ndarray:
    """Σ values over rows whose key is ≥ ``at`` (``side="left"``) — keys sorted ascending."""
    csum = np.concatenate([np.cumsum(values[::-1], axis=0)[::-1],
                           np.zeros((1,) + values.shape[1:])], axis=0)
    return csum[np.searchsorted(sorted_keys, at, side=side)]


def _risk(X: np.ndarray, y: np.ndarray, beta: np.ndarray, w: np.ndarray | None = None) -> _Risk:
    """The risk-set sums at ``beta``. With case weights ``w`` (Binder's pseudo-likelihood): every
    risk-set sum is weighted, ``S0 = Σ w r``, ``S1 = Σ w r x``, and each of a time's tied events
    counts its time's mean weight ``w̄`` in the log-likelihood,
    ``Σ_events w_i η_i − Σ_slots w̄ log(S0_slot)`` (R's ``coxph``, ``coxfit6``/``agfit4``)."""
    t, e, s = y["time"], y["event"], y["entry"]
    eta = X @ beta
    shift = float(eta.max()) if len(eta) else 0.0
    r = np.exp(eta - shift)
    wr = r if w is None else w * r
    rX = wr[:, None] * X
    by_t = np.argsort(t, kind="stable")
    by_s = np.argsort(s, kind="stable")
    events = np.flatnonzero(e)
    order = events[np.argsort(t[events], kind="stable")]
    times, k_of = np.unique(t[order], return_inverse=True)
    # At risk at τ: time ≥ τ, less those that had not yet entered (entry ≥ τ).
    S0_k = _suffix(wr[by_t], t[by_t], times, "left") - _suffix(wr[by_s], s[by_s], times, "left")
    S1_k = _suffix(rX[by_t], t[by_t], times, "left") - _suffix(rX[by_s], s[by_s], times, "left")
    d_k = np.bincount(k_of, minlength=len(times)).astype(float)
    D0_k = np.bincount(k_of, weights=wr[order], minlength=len(times))
    D1_k = np.zeros_like(S1_k)
    np.add.at(D1_k, k_of, rX[order])
    # Efron: the l-th of d tied events sees the risk set less l/d of the tied events' weight.
    first = np.searchsorted(k_of, np.arange(len(times)))
    l = np.arange(len(order)) - first[k_of]
    frac = l / d_k[k_of]
    S0 = S0_k[k_of] - frac * D0_k[k_of]
    S1 = S1_k[k_of] - frac[:, None] * D1_k[k_of]
    if w is None:
        loglik = float(eta[order].sum() - (np.log(S0) + shift).sum())
        return _Risk(order=order, k_of=k_of, times=times, frac=frac, S0=S0, S1=S1, loglik=loglik)
    mw = (np.bincount(k_of, weights=w[order], minlength=len(times)) / d_k)[k_of]
    loglik = float(w[order] @ eta[order] - mw @ (np.log(S0) + shift))
    return _Risk(order=order, k_of=k_of, times=times, frac=frac, S0=S0, S1=S1, loglik=loglik, mw=mw)


def _at_risk_sums(y: np.ndarray, rk: _Risk, per_slot: np.ndarray, own: np.ndarray) -> np.ndarray:
    """For every row, Σ over the event slots at which it is at risk of ``per_slot`` (shape (d, …)),
    with a row's own event time weighted (1 − l/d) per slot as Efron weights it (``own`` holds each
    slot's value times (1 − l/d))."""
    t, s, e = y["time"], y["entry"], y["event"]
    K = len(rk.times)
    shape = per_slot.shape[1:]
    by_k = np.zeros((K,) + shape)
    np.add.at(by_k, rk.k_of, per_slot)
    own_k = np.zeros((K,) + shape)
    np.add.at(own_k, rk.k_of, own)
    cum = np.concatenate([np.zeros((1,) + shape), np.cumsum(by_k, axis=0)], axis=0)
    # slots with entry < τ_k ≤ time: cum up to time minus cum up to entry
    total = cum[np.searchsorted(rk.times, t, side="right")] - cum[np.searchsorted(rk.times, s, side="right")]
    k_row = np.full(len(t), -1)
    k_row[rk.order] = rk.k_of
    died = e & (k_row >= 0)
    total[died] -= by_k[k_row[died]] - own_k[k_row[died]]
    return total


def _score_info(X: np.ndarray, y: np.ndarray, beta: np.ndarray, w: np.ndarray | None = None
                ) -> tuple[float, np.ndarray, np.ndarray, _Risk]:
    """The log-likelihood, score and information at ``beta``; weighted by ``w`` as :func:`_risk`
    says: ``U = Σ_events w_i x_i − Σ_slots w̄ x̄`` and ``I = Σ_slots w̄ (S2/S0 − x̄ x̄ᵀ)``."""
    rk = _risk(X, y, beta, w)
    xbar = rk.S1 / rk.S0[:, None]
    eta = X @ beta
    r = np.exp(eta - (float(eta.max()) if len(eta) else 0.0))
    if w is None:
        U = X[rk.order].sum(axis=0) - xbar.sum(axis=0)
        c = _at_risk_sums(y, rk, 1.0 / rk.S0, (1.0 - rk.frac) / rk.S0)
        info = (X * (r * c)[:, None]).T @ X - xbar.T @ xbar
        return rk.loglik, U, (info + info.T) / 2, rk
    U = w[rk.order] @ X[rk.order] - rk.mw @ xbar
    hazard = rk.mw / rk.S0
    c = _at_risk_sums(y, rk, hazard, (1.0 - rk.frac) * hazard)
    info = (X * (w * r * c)[:, None]).T @ X - (xbar * rk.mw[:, None]).T @ xbar
    return rk.loglik, U, (info + info.T) / 2, rk


@dataclass
class CoxFit:
    beta: np.ndarray
    cov: np.ndarray  # inverse of the partial likelihood's information
    loglik: float
    loglik0: float  # at β = 0
    iterations: int
    converged: bool
    n_events: int


def cox_fit(X: np.ndarray, y: np.ndarray, *, max_iter: int = MAX_ITER, tol: float = TOL,
            weights: np.ndarray | None = None) -> CoxFit:
    """Maximize Efron's partial likelihood by Newton–Raphson, halving a step that lowers it.

    The columns are centered and scaled for the iterations (the coefficients are mapped back), so
    the Newton decrement convergence test does not depend on their units. Not converged when the
    decrement stays above ``tol`` or a coefficient runs past :data:`DIVERGED` SDs (monotone
    likelihood: the estimate is infinite).

    ``weights`` (case weights, positive) fit Binder's pseudo-likelihood, each row weighted in every
    risk set and Efron's tied slots by their time's mean weight (:func:`_risk`), as R's ``coxph``
    with ``weights`` does; ``cov`` is then the inverse of the weighted information.
    """
    X = np.asarray(X, dtype=float)
    n, P = X.shape
    w = None if weights is None else np.asarray(weights, dtype=float)
    if w is not None and (len(w) != n or not np.all(np.isfinite(w)) or np.any(w <= 0)):
        raise ValueError("Case weights must be positive and finite, one per row.")
    n_events = int(np.sum(y["event"]))
    if n_events == 0:
        raise ValueError("No row has the event, so the hazard ratios cannot be estimated.")
    mean = X.mean(axis=0) if n else np.zeros(P)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Z = (X - mean) / sd
    beta = np.zeros(P)
    ll, U, info, _ = _score_info(Z, y, beta, w)
    ll0 = ll
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        step = np.linalg.lstsq(info, U, rcond=None)[0]
        if float(U @ step) < tol:
            converged = True
            break
        for _ in range(40):
            trial = beta + step
            parts = _score_info(Z, y, trial, w)
            if parts[0] >= ll - 1e-12 * max(1.0, abs(ll)):
                break
            step /= 2.0
        beta = trial
        ll, U, info, _ = parts
        if np.any(np.abs(beta) > DIVERGED):
            break
    else:
        converged = float(U @ np.linalg.lstsq(info, U, rcond=None)[0]) < max(tol, LOOSE_TOL)
    cov_z = np.linalg.pinv(info)
    return CoxFit(beta=beta / sd, cov=cov_z / np.outer(sd, sd), loglik=ll, loglik0=ll0,
                  iterations=it, converged=converged and not np.any(np.abs(beta) > DIVERGED),
                  n_events=n_events)


def score_residuals(X: np.ndarray, y: np.ndarray, beta: np.ndarray,
                    weights: np.ndarray | None = None) -> np.ndarray:
    """Each row's contribution to the score at ``beta`` (n × P), consistent with Efron's
    likelihood: ``δ_i (x_i − x̄_i) − r_i Σ_slots φ (x_i − x̄_slot) dΛ_slot``, where x̄_i is the mean
    of the x̄ over the slots of row i's tied event time, φ is 1, or 1 − l/d at a row's own event
    time, and ``dΛ = w̄ / S0`` the slot's hazard (``w̄`` 1 without weights). They sum to the score,
    so to zero at the estimate (Therneau & Grambsch 2000, §7); with ``weights`` they are R's
    unweighted ``residuals(fit, "score")`` of the weighted fit, and ``w_i`` times them sum to its
    score."""
    X = np.asarray(X, dtype=float)
    w = None if weights is None else np.asarray(weights, dtype=float)
    rk = _risk(X, y, beta, w)
    xbar = rk.S1 / rk.S0[:, None]
    eta = X @ beta
    r = np.exp(eta - (float(eta.max()) if len(eta) else 0.0))
    hazard = 1.0 / rk.S0 if w is None else rk.mw / rk.S0
    c = _at_risk_sums(y, rk, hazard, (1.0 - rk.frac) * hazard)
    q = _at_risk_sums(y, rk, xbar * hazard[:, None], xbar * ((1.0 - rk.frac) * hazard)[:, None])
    out = -r[:, None] * (X * c[:, None] - q)
    K = len(rk.times)
    mean_k = np.zeros((K, X.shape[1]))
    np.add.at(mean_k, rk.k_of, xbar)
    mean_k /= np.bincount(rk.k_of, minlength=K)[:, None]
    out[rk.order] += X[rk.order] - mean_k[rk.k_of]
    return out


def schoenfeld_residuals(X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each event's Schoenfeld residual ``x_i − x̄`` (x̄ averaged over its time's Efron slots), and
    the event rows in time order."""
    X = np.asarray(X, dtype=float)
    rk = _risk(X, y, beta)
    xbar = rk.S1 / rk.S0[:, None]
    K = len(rk.times)
    mean_k = np.zeros((K, X.shape[1]))
    np.add.at(mean_k, rk.k_of, xbar)
    mean_k /= np.bincount(rk.k_of, minlength=K)[:, None]
    return X[rk.order] - mean_k[rk.k_of], rk.order


def ph_test(X: np.ndarray, y: np.ndarray, fit: CoxFit) -> np.ndarray:
    """Each coefficient's p-value for proportional hazards: Grambsch & Therneau's approximate test
    ``T_j = (Σ_k (g_k − ḡ) s*_kj)² / (d V_jj Σ_k (g_k − ḡ)²)`` on χ²₁, with scaled Schoenfeld
    residuals ``s* = d · s V`` and ``g_k`` the event's rank in time."""
    from scipy.stats import chi2

    s, _ = schoenfeld_residuals(X, y, fit.beta)
    d = len(s)
    scaled = d * s @ fit.cov
    g = np.arange(1, d + 1, dtype=float)
    g -= g.mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        T = (g @ scaled) ** 2 / (d * np.diag(fit.cov) * float(g @ g))
    return chi2.sf(T, 1)


def concordance(time: Any, event: Any, risk: Any) -> float:
    """Harrell's C: over pairs (i, j) with i's event before j's end of follow-up (or at the same
    time when j is censored), the share with risk_i > risk_j, ties in risk counting one half.
    NaN when no pair is comparable. O(n log n), by a Fenwick tree over the risk ranks."""
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=bool)
    risk = np.asarray(risk, dtype=float)
    n = len(time)
    if n == 0 or not event.any():
        return float("nan")
    rank = np.unique(risk, return_inverse=True)[1] + 1  # 1-based ranks; ties share one
    size = int(rank.max())
    tree = [0] * (size + 1)

    def add(i: int) -> None:
        while i <= size:
            tree[i] += 1
            i += i & -i

    def below(i: int) -> int:  # rows in the tree with rank ≤ i
        total = 0
        while i > 0:
            total += tree[i]
            i -= i & -i
        return total

    # Visit rows latest first; at one time, censored rows join the tree before the events are
    # counted (a row censored at an event's time outlived it), tied events join after.
    key = np.lexsort((event.astype(int), -time))
    concordant = tied = pairs = 0.0
    inserted = 0
    i = 0
    while i < n:
        j = i
        while j < n and time[key[j]] == time[key[i]]:
            j += 1
        block = key[i:j]
        censored = block[~event[block]]
        died = block[event[block]]
        for row in censored:
            add(int(rank[row]))
            inserted += 1
        for row in died:
            lo = below(int(rank[row]) - 1)
            at = below(int(rank[row])) - lo
            concordant += lo
            tied += at
            pairs += inserted
        for row in died:
            add(int(rank[row]))
            inserted += 1
        i = j
    return float((concordant + 0.5 * tied) / pairs) if pairs else float("nan")


# ── the sklearn estimators ───────────────────────────────────────────────────


def _matrix(X: Any) -> tuple[np.ndarray, list[str]]:
    if isinstance(X, pd.DataFrame):
        return X.to_numpy(dtype=float), [str(c) for c in X.columns]
    M = np.asarray(X, dtype=float)
    return M, [f"x{j}" for j in range(M.shape[1])]


class CoxRegressor(BaseEstimator):
    """Cox proportional hazards on a structured outcome (:func:`survival_outcome`); ``predict``
    returns the risk score, the log relative hazard ``Xβ``."""

    def fit(self, X: Any, y: Any) -> "CoxRegressor":
        M, names = _matrix(X)
        fit = cox_fit(M, np.asarray(y))
        self.coef_ = fit.beta
        self.converged_ = fit.converged
        self.feature_names_in_ = np.asarray(names, dtype=object)
        self.n_features_in_ = len(names)
        return self

    def predict(self, X: Any) -> np.ndarray:
        return _matrix(X)[0] @ self.coef_


class ConstantRisk(BaseEstimator):
    """The baseline: one risk score for every row, which orders no pair (C = 0.5)."""

    def fit(self, X: Any, y: Any) -> "ConstantRisk":
        return self

    def predict(self, X: Any) -> np.ndarray:
        return np.zeros(len(X))


# ── the inference table ──────────────────────────────────────────────────────


def cox_table(matrix: pd.DataFrame, y: np.ndarray, clusters: Any) -> Any:
    """The Cox model's coefficient table (log hazard ratios) on the matrix the model saw."""
    from scipy import stats

    from turbotab.core.models.inference import (FEW_CLUSTERS, InferenceTable, _and, _info,
                                                _refused, _row, _z_rows, floor_refusal, format_p)
    from turbotab.core.models.linear import collinearity_concern

    names = [str(c) for c in matrix.columns]
    X = matrix.to_numpy(dtype=float)
    y = np.asarray(y)
    estimator = "Cox proportional hazards (partial likelihood, Efron ties)"
    # The scale (log hazard ratios, each hazard ratio exp(estimate)) is declared by the family's
    # ``inference`` through ``inference._on_scale``, as every inference table's is (WP8).
    try:
        singular = collinearity_concern(matrix)
    except Exception:  # noqa: BLE001 - a diagnostic that cannot run is not a verdict
        singular = None
    fit = cox_fit(X, y)
    if singular:
        return _refused(names, fit.beta, estimator, clusters, singular, ())
    if not fit.converged:
        sd = X.std(axis=0)
        runaway = [n for n, b, s in zip(names, fit.beta, sd) if abs(b * s) > DIVERGED] or names
        reason = (f"The partial likelihood has no maximum for {_and(runaway)} (in one of its groups "
                  f"every row has the event, or none does), so the hazard ratio is infinite and no "
                  f"interval or p-value is reported.")
        return _refused(names, fit.beta, estimator, clusters, reason,
                        ({"label": f"Leave {_and(runaway)} out, or merge its rare levels",
                          "decision": None},))
    concerns = [clusters.note] if clusters.note else []
    refusal = floor_refusal(clusters, task="time_to_event")
    if refusal:
        return _refused(names, fit.beta, estimator, clusters, *refusal)
    if clusters.clustered:
        U = score_residuals(X, y, fit.beta)
        G = clusters.n_clusters
        by_unit = np.zeros((G, X.shape[1]))
        np.add.at(by_unit, clusters.codes, U)
        V = fit.cov @ (by_unit.T @ by_unit) @ fit.cov
        se = np.sqrt(np.clip(np.diag(V), 0, None))
        q = float(stats.t.ppf(0.975, G - 1))
        rows = []
        for name, b, s in zip(names, fit.beta, se):
            if not (np.isfinite(s) and s > 0):
                rows.append(_row(name, b, None))
                continue
            rows.append(_row(name, b, s, float(G - 1), b - q * s, b + q * s,
                             float(2.0 * stats.t.sf(abs(b / s), G - 1))))
        missing = (f", {clusters.n_missing:,} rows with no `{clusters.column}` counted as units of "
                   f"their own" if clusters.n_missing else "")
        caption = (f"95% intervals from the Lin–Wei cluster-robust sandwich by `{clusters.column}`, "
                   f"G = {G:,} clusters{missing}, on t({G - 1:,}).")
        if clusters.note is None:
            concerns.append(f"Intervals are cluster-robust by `{clusters.column}` (Lin–Wei, G = "
                            f"{G:,}), because its rows repeat.")
        if G < FEW_CLUSTERS:
            concerns.append(f"Only {G} `{clusters.column}` clusters: the sandwich rests on {G} "
                            f"unit-level sums, and t({G - 1}) is the least allowance Cameron & "
                            f"Miller (2015) ask for.")
        covariance = "CR0"
        cov = V
    else:
        rows = _z_rows(names, fit.beta, np.sqrt(np.clip(np.diag(fit.cov), 0, None)))
        caption = "95% Wald intervals from the Cox partial likelihood's information (Efron ties)."
        covariance = "model"
        cov = fit.cov
    p = ph_test(X, y, fit)
    threshold = PH_ALPHA / max(1, len(names))
    flagged = [(pj, n) for pj, n in zip(p, names) if np.isfinite(pj) and pj < threshold]
    if flagged:
        pj, name = min(flagged)
        concerns.append(f"The hazard ratio for `{name}` may change over follow-up (Grambsch–Therneau "
                        f"p = {format_p(float(pj))}, against {PH_ALPHA:g} shared over "
                        f"{len(names)} column{'s' if len(names) != 1 else ''}): it is then an "
                        f"average over follow-up, not one constant ratio.")
    # MS3: the covariance rides with the table, so a spline's Wald tests (and D1 over imputations)
    # can read it.
    return InferenceTable(rows, _info(estimator, covariance, caption, clusters), concerns, cov=cov)


SURVEY_ESTIMATOR = ("survey-weighted Cox proportional hazards (Binder's pseudo-likelihood, Efron "
                    "ties)")


def survey_cox_table(matrix: pd.DataFrame, y: np.ndarray, design: Any) -> Any:
    """The design-based Cox table over a survey design (MODELING_SEQUENCE §0 ruling 6; MS4).

    Binder (1992, *Int Stat Rev* 60:249): the population's partial-likelihood score is estimated
    by weighting each analysis row by its survey weight in every risk set, and β̂ solves the
    weighted score (:func:`cox_fit` with ``weights``). Its variance is the design-based variance
    of the total of the weighted score residuals ``w_i U_i(β̂)``, sandwiched by the inverse of the
    weighted information (:func:`~turbotab.core.models.survey.design_table`), which is R's
    ``survey::svycoxph``: ``svyrecvar`` of ``residuals(fit, "dfbeta", weighted = TRUE)``. The
    domain is the matrix's rows placed in the design with a positive weight; every other design
    row keeps its stratum and PSU at a score of zero.

    The proportional-hazards check is Grambsch & Therneau's on these participants, unweighted: a
    diagnostic of the model's form, never a design-based test, and said so.
    """
    from turbotab.core.models.inference import _and, format_p
    from turbotab.core.models.linear import collinearity_concern
    from turbotab.core.models.survey import _refused, design_table, domain_of

    names = [str(c) for c in matrix.columns]
    domain = domain_of(matrix.index, design)
    X = matrix.to_numpy(dtype=float)[domain.keep]
    yy = np.asarray(y)[domain.keep]
    estimator = SURVEY_ESTIMATOR
    n_events = int(np.sum(yy["event"])) if len(yy) else 0
    if n_events == 0 or domain.n <= X.shape[1]:
        return _refused(names, None, estimator, design, domain.n,
                        f"Only {domain.n:,} analysis rows ({n_events:,} with the event) carry a "
                        f"positive weight and a place in the design, too few for {X.shape[1]} "
                        f"coefficients.")
    try:
        singular = collinearity_concern(pd.DataFrame(X, columns=names))
    except Exception:  # noqa: BLE001 - a diagnostic that cannot run is not a verdict
        singular = None
    fit = cox_fit(X, yy, weights=domain.weight)
    if singular:
        return _refused(names, fit.beta, estimator, design, domain.n, singular)
    if not fit.converged:
        sd = X.std(axis=0)
        runaway = [n for n, b, s in zip(names, fit.beta, sd) if abs(b * s) > DIVERGED] or names
        reason = (f"The weighted partial likelihood has no maximum for {_and(runaway)} (in "
                  f"one of its groups every row has the event, or none does), so the hazard ratio "
                  f"is infinite and no interval or p-value is reported.")
        return _refused(names, fit.beta, estimator, design, domain.n, reason,
                        ({"label": f"Leave {_and(runaway)} out, or merge its rare levels",
                          "decision": None},))
    U = score_residuals(X, yy, fit.beta, domain.weight) * domain.weight[:, None]
    table = design_table(names, fit.beta, U, None, design, domain, estimator, bread=fit.cov)
    if table.info.get("refused"):
        return table
    plain = cox_fit(X, yy)
    if plain.converged:
        p = ph_test(X, yy, plain)
        threshold = PH_ALPHA / max(1, len(names))
        flagged = [(pj, n) for pj, n in zip(p, names) if np.isfinite(pj) and pj < threshold]
        if flagged:
            pj, name = min(flagged)
            table.concerns.append(
                f"Among these participants, unweighted, the hazard ratio for `{name}` may change "
                f"over follow-up (Grambsch–Therneau p = {format_p(float(pj))}, against "
                f"{PH_ALPHA:g} shared over {len(names)} column{'s' if len(names) != 1 else ''}): "
                f"a check of the model's form, not a design-based test. If it changes in the "
                f"population too, the hazard ratio is an average over follow-up.")
    return table


# ── the family ───────────────────────────────────────────────────────────────


class Cox(FamilyBase):
    key = "cox"
    label = "Cox proportional hazards"
    tasks: tuple[Task, ...] = ("time_to_event",)
    inductive_bias = ("Each predictor multiplies the hazard by a constant ratio over all of "
                      "follow-up; effects add on the log scale.")
    strengths = (
        "Uses every row's follow-up, censored or not, and rows that enter late.",
        "Hazard ratios with intervals; no assumption about the baseline hazard's shape.",
    )
    cautions = (
        "Assumes each hazard ratio is constant over follow-up; a check says when it is not.",
        "Misses curves and interactions it is not given.",
    )
    needs_scaling = False
    handles_missing = False
    # MODEL_FAMILY_CONTRACT §1 (§3.1's row for it). No attribution yet: the explain stage refuses
    # a time-to-event outcome until MC-7 widens it.
    identity = Identity(kind="estimator", library="turbotab", estimator="CoxRegressor")
    purposes = ("prediction", "inference")
    predicts = True
    flexible = False
    bootstrap_optimism = True
    inference_decl = InferenceDecl(table="intervals", intervals=("model", "sandwich", "Taylor"),
                                   design_based=True, product_terms=True, matrix_table=True,
                                   words="Cox regression (Binder's pseudo-likelihood, Efron ties)")
    invariances = ("linear_maps",)
    curve_shape = "straight"
    diagnostics = ("collinearity", "proportional_hazards")
    raw_scale = {"time_to_event": "log_hazard"}
    review_lenses = ("shared",)
    consequence = ("Hazard ratios for a time-to-event outcome, using every row's follow-up, "
                   "censored or not.")
    card_label = "Cox model"

    def methods_label(self, task: Task | None) -> str:
        return "Cox proportional hazards"

    def build(self, task: Task, purpose: Purpose | None, n_rows: int, n_features: int) -> Any:
        return CoxRegressor()

    def describe(self, task: Task, purpose: Purpose | None) -> tuple[str, str]:
        detail = ("Fits the log hazard ratio of every column by partial likelihood, Efron ties, "
                  "each row at risk from its entry to its end of follow-up.")
        if purpose == "inference":
            detail += (" Intervals are Wald, or Lin–Wei cluster-robust when a unit's rows repeat, "
                       "or, for the surveyed population, survey-weighted (Binder) with Taylor "
                       "linearization over the design; proportional hazards are checked.")
        return "Cox proportional hazards", detail

    def coefficients(self, pipeline: Any, X: Any, y: Any, *, task: Task,
                     purpose: Purpose | None, groups: Any = None) -> list[dict[str, Any]] | None:
        model = pipeline[-1]
        return coefficient_rows([str(f) for f in model.feature_names_in_], model.coef_)

    def inference(self, pipeline: Any, X: Any, y: Any, *, task: Task, clusters: Any,
                  outcome: Any = None, rows: Any = None, survey: Any = None) -> Any:
        """The Cox table on the hazard-ratio scale (WP8's ``_on_scale``: each row's hazard ratio
        and its interval, drawn on a log axis, the event named), and the rows it was fit on. With
        ``survey`` (a :class:`~turbotab.core.models.survey.SurveyDesign`, the "surveyed
        population" answer) the table is design-based (:func:`survey_cox_table`)."""
        from turbotab.core.models.linear import model_matrix

        return self.inference_matrix(model_matrix(pipeline, X), y, task=task, classes=[0, 1],
                                     clusters=clusters, outcome=outcome, rows=rows, survey=survey)

    def inference_matrix(self, matrix: pd.DataFrame, y: Any, *, task: Task, classes: Any,
                         clusters: Any, outcome: Any = None, rows: Any = None,
                         survey: Any = None) -> Any:
        """The Cox table on a given model matrix (a quintile trend test's refit), design-based
        under ``survey``, on the hazard-ratio scale either way."""
        from turbotab.core.models.inference import _on_rows, _on_scale

        if survey is None:
            table = cox_table(matrix, y, clusters)
            return _on_rows(_on_scale(table, "time_to_event", [0, 1], outcome), len(matrix), rows)
        table = survey_cox_table(matrix, y, survey)
        if table.rows:
            table = _on_scale(table, "time_to_event", [0, 1], outcome)
        n_domain = (table.info.get("survey") or {}).get("n_domain")
        if n_domain is not None and int(n_domain) < len(matrix):
            table.info.update(n_rows=int(n_domain), rows=rows)
            table.info["caption"] += (f" Estimated from the {int(n_domain):,} of the "
                                      f"{len(matrix):,} analyzed rows with a positive weight and a "
                                      f"place in the design.")
            return table
        return _on_rows(table, len(matrix), rows)

    def assess(self, s: Situation) -> Assessment:
        from turbotab.core.models.inference import min_clusters

        concerns: list[str] = []
        events = s.n_events
        if events is not None and s.n_features >= events:
            return Assessment(0.0, "poor", (f"{events:,} events for {s.n_features:,} predictors: "
                                            f"the partial likelihood cannot separate them.",))
        fit = "good"
        if events is not None and s.n_features:
            epv = events / s.n_features
            if epv < EPV_RULE:
                concerns.append(f"{events:,} events for {s.n_features:,} predictors ({epv:.1f} "
                                f"each); a common rule of thumb asks for {EPV_RULE}.")
                fit = "poor" if epv < EPV_RULE / 2 else "fair"
        units = getattr(s, "n_units", None)
        if s.purpose == "inference" and units is not None and units < min_clusters():
            concerns.append(f"Rows repeat within only {units} units: no interval can be reported.")
            fit = "poor"
        score = {"good": 3.0, "fair": 2.0, "poor": 0.5}[fit]
        return Assessment(score, fit, tuple(concerns))


COX = register_family(Cox())

__all__ = [
    "COX", "ConstantRisk", "Cox", "CoxFit", "CoxRegressor", "OUTCOME_DTYPE", "SURVEY_ESTIMATOR",
    "concordance", "cox_fit", "cox_table", "follow_up_columns", "ph_test", "schoenfeld_residuals",
    "score_residuals", "survey_cox_table", "survival_outcome", "time_to_event_outcome",
]

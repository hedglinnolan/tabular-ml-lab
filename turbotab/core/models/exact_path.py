"""The elastic net's penalty path solved exactly, for least squares and the logistic loss.

The family chooses its penalty on the pooled inner loss rounded to 10⁻⁹ of the smallest
(``elastic_net.lowest_rounded``; RECIPES_AND_TUNING §4.2), which only means something if each
point of the curve is computed far closer than that. An iterative solver gets there slowly or not
at all:

* **Least squares.** scikit-learn's coordinate descent at its tolerance (10⁻⁴) leaves the inner
  curve 10⁻⁵ to 10⁻³ off; at 10⁻¹² it is close enough but four to twelve times slower (400 × 25:
  0.27 s to 1.25 s; 1,000 × 200 correlated: 2.4 s to 29 s), since each sweep gains a fixed share
  of the remaining error and collinear columns make that share small.
* **Logistic.** scikit-learn fits an elastic-net logistic regression only with ``saga``, whose
  stopping rule (the largest coefficient change in an epoch, relative to the largest coefficient,
  the intercept left out) left the family's inner log loss 2 × 10⁻³ off at its tolerance and still
  at 10⁻¹⁰: a penalty that zeros every coefficient stops it after an epoch or two with the
  intercept unfitted (the null model's 0.69320 for 0.69315). Tightening it only made it slower.

**Feature-sign search** (Lee, Battle, Raina & Ng 2007, NIPS 19) minimizes a quadratic plus an l1
penalty exactly: it holds a set of coefficients nonzero with their signs, solves the linear system
those give, and moves to the best of that solution and the points where a held coefficient reaches
zero; a zero coefficient whose gradient exceeds the penalty joins with the sign that lowers the
objective. It stops in finitely many steps at the exact minimizer, whatever the conditioning.

* **Least squares** is that quadratic: one search per penalty, started at the last solution.
* **Logistic** is proximal Newton (Lee, Sun & Saunders 2014, SIAM J. Optim. 24:1420): the log loss
  replaced by its second-order expansion, exact Hessian included, that penalized quadratic solved
  by feature-sign search, then a backtracking line search on the true objective. Near the solution
  the full step is taken and the convergence is quadratic, so a full step smaller than 10⁻¹⁰ of
  the largest coefficient is the last: what remains is rounding.

It costs about what scikit-learn's defaults did, timed at one thread over a whole inner
cross-validation: least squares on the NHANES-like 400 × 27, 0.15 s (coordinate descent at 10⁻⁴,
0.18 s; at 10⁻¹², 1.05 s); 120 × 100 correlated, 0.9 s (2.7 s; 18.6 s); logistic on the NHANES-like
400 × 19, 0.05 s (``saga`` at 10⁻³, 0.28 s).

The problems are scikit-learn's own. Least squares: ``(1/2n)‖y − Xw − b‖² + αρ‖w‖₁ +
α(1 − ρ)/2 ‖w‖²``. Logistic: ``C Σ sᵢ ℓᵢ + ρ‖W‖₁ + (1 − ρ)/2 ‖W‖²``, divided here by ``C Σ sᵢ``;
two classes have one row of coefficients (the second's logit), more have one row per class, the
last intercept held at 0 while solving (one number added to every intercept changes no
probability) and the intercepts then centered. A direction the quadratic cannot see (a multinomial
lasso's feature nonzero for every class, or columns that repeat one another under a pure lasso) is
followed until a coefficient reaches zero, since the penalty falls along it.
"""
from __future__ import annotations

import warnings
from typing import Any, Sequence

import numpy as np
from scipy.special import expit, logsumexp

MAX_NEWTON = 100     # proximal Newton steps for one penalty (quadratic convergence needs < 15)
STEP_TOL = 1e-10     # a full Newton step this small, relative to the largest coefficient, is the last
NULL_EIGEN = 1e-12   # eigenvalues below this share of the largest are a flat direction
ARMIJO = 1e-4


def _warn(what: str) -> None:
    from sklearn.exceptions import ConvergenceWarning

    warnings.warn(f"The elastic net's {what} did not converge.", ConvergenceWarning, stacklevel=3)


# ── the penalized quadratic ──────────────────────────────────────────────────


def _solve_active(A: np.ndarray, r: np.ndarray, z: np.ndarray, definite: bool
                  ) -> tuple[np.ndarray, bool]:
    """The minimizer of ½zᵀAz − rᵀz (A positive semidefinite) or, along a flat direction of A
    that ``r`` descends, that direction: ``(point or direction, is_direction)``.

    A Cholesky solve when A is known definite (a ridge part) or its factor shows no pivot at the
    flat threshold (a singular A has one at rounding, or fails); else the eigen-decomposition,
    ten to forty times dearer, which finds the flat directions."""
    from scipy.linalg import LinAlgError, cho_factor, cho_solve

    try:
        factor = cho_factor(A, check_finite=False)
        pivots = np.abs(np.diag(factor[0]))
        if definite or float(pivots.min()) ** 2 > NULL_EIGEN * max(float(np.max(np.diag(A))),
                                                                   np.finfo(float).tiny):
            return cho_solve(factor, r, check_finite=False), False
    except LinAlgError:
        pass
    vals, vecs = np.linalg.eigh(A)
    top = float(np.max(np.abs(vals))) if len(vals) else 0.0
    flat = vals <= NULL_EIGEN * max(top, np.finfo(float).tiny)
    proj = vecs.T @ r
    if flat.any():
        along = vecs[:, flat] @ proj[flat]
        if float(along @ r) > NULL_EIGEN * max(float(r @ r), np.finfo(float).tiny):
            return along, True
    solid = ~flat
    point = vecs[:, solid] @ (proj[solid] / vals[solid])
    if flat.any():
        point = point + vecs[:, flat] @ (vecs[:, flat].T @ z)  # stays where it is along the flat
    return point, False


def feature_sign(A: np.ndarray, c: np.ndarray, gamma: float, penalized: np.ndarray,
                 z0: np.ndarray, definite: bool) -> tuple[np.ndarray, bool]:
    """``argmin ½zᵀAz − cᵀz + γ Σ_{penalized} |z_j|`` exactly, started at ``z0`` (Lee et al. 2007,
    feature-sign search, with the unpenalized coordinates always held): ``(z, settled)``;
    ``definite``: A is known positive definite (a ridge part), so its Cholesky factor needs no
    check for a flat direction."""
    m = len(c)
    z = np.array(z0, dtype=float)
    if gamma <= 0.0:
        point, direction = _solve_active(A, c, z, definite)
        return (z if direction else point), True
    sign = np.where(penalized, np.sign(z), 0.0)
    held = ~penalized | (z != 0.0)
    step_due = True
    for _ in range(50 * m + 200):
        if step_due:
            a = np.flatnonzero(held)
            step_due = False
            if not len(a):
                continue
            Aa, ca, za, pen = A[np.ix_(a, a)], c[a], z[a], penalized[a]
            target, direction = _solve_active(Aa, ca - gamma * sign[a], za, definite)
            d = target if direction else target - za
            with np.errstate(divide="ignore", invalid="ignore"):
                cross = -za / d
            crossing = pen & (za != 0.0) & (np.sign(za) == -np.sign(d)) & (cross > 0.0)
            ts = cross[crossing]
            if direction:
                if not len(ts):
                    return z, False  # unbounded along a flat: not a problem the family poses
                t = float(np.min(ts))
            else:
                ts = np.concatenate([ts[ts < 1.0], [1.0]])
                Ad = Aa @ d
                lin, quad = float(za @ Ad - ca @ d), float(d @ Ad)
                l1 = np.abs(za[pen][None, :] + ts[:, None] * d[pen][None, :]).sum(axis=1)
                t = float(ts[int(np.argmin(ts * lin + 0.5 * ts * ts * quad + gamma * l1))])
            new = za + t * d
            reached = crossing & (np.abs(cross - t) <= 1e-12 * t)
            new[reached] = 0.0  # a coefficient that reaches zero there is zero
            # The linear system held the signs it was given: where one changed or reached zero,
            # its solution is no longer the active set's, and the search steps again.
            step_due = bool(direction or t != 1.0 or (crossing & (cross < t)).any() or reached.any())
            z[a] = new
            held &= ~(penalized & (z == 0.0))
            sign = np.where(penalized, np.sign(z), 0.0)
            continue
        grad = A @ z - c
        idle = penalized & ~held
        if not idle.any():
            return z, True
        excess = np.where(idle, np.abs(grad) - gamma, -np.inf)
        j = int(np.argmax(excess))
        # a gradient past the penalty by no more than rounding is no reason to move
        if excess[j] <= 1e-9 * gamma + 1e-13 * max(1.0, float(np.max(np.abs(c)))):
            return z, True
        held[j] = True
        sign[j] = -np.sign(grad[j])
        step_due = True
    return z, False


# ── least squares ────────────────────────────────────────────────────────────


def alpha_grid(X: np.ndarray, y: np.ndarray, l1_ratio: float, n_alphas: int = 100,
               eps: float = 1e-3, fit_intercept: bool = True) -> np.ndarray:
    """scikit-learn's grid for ``ElasticNetCV`` (``_alpha_grid``, unweighted): from the penalty
    that keeps every coefficient at zero down to ``eps`` of it, evenly on the log scale."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if fit_intercept:
        y = y - np.average(y, axis=0)
        x_mean = np.average(X, axis=0)
        Xy = X.T @ y - x_mean * np.sum(y, axis=0)
    else:
        Xy = X.T @ y
    alpha_max = np.sqrt(np.max(Xy[:, None] ** 2)) / (X.shape[0] * l1_ratio)
    if alpha_max <= np.finfo(np.float64).resolution:
        return np.full(n_alphas, np.finfo(np.float64).resolution)
    return np.geomspace(alpha_max, alpha_max * eps, num=n_alphas)


def gaussian_path(X: np.ndarray, y: np.ndarray, alphas: Sequence[float], l1_ratio: float,
                  fit_intercept: bool = True, start: np.ndarray | None = None
                  ) -> tuple[np.ndarray, np.ndarray, bool]:
    """The least-squares elastic net at each of ``alphas`` (warm-started in their order):
    ``(coefs p × k, intercepts k, every point settled)``."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    if fit_intercept:
        x_mean, y_mean = X.mean(axis=0), float(y.mean())
        Xc, yc = X - x_mean, y - y_mean
    else:
        x_mean, y_mean = np.zeros(p), 0.0
        Xc, yc = X, y
    G = Xc.T @ Xc / n
    c = Xc.T @ yc / n
    pen = np.ones(p, dtype=bool)
    w = np.zeros(p) if start is None else np.array(start, dtype=float)
    coefs = np.zeros((p, len(alphas)))
    every = True
    for k, alpha in enumerate(alphas):
        ridge = float(alpha) * (1.0 - l1_ratio)
        A = G + ridge * np.eye(p) if ridge > 0 else G
        w, ok = feature_sign(A, c, float(alpha) * l1_ratio, pen, w, ridge > 0)
        every &= ok
        coefs[:, k] = w
    if not every:
        _warn("least-squares path")
    return coefs, y_mean - x_mean @ coefs, every


# ── the logistic loss ────────────────────────────────────────────────────────


class _Logistic:
    """One penalized logistic fit's data: ``X`` (n × p), the outcome as class codes 0…K−1, the
    weights (scaled to sum to 1)."""

    def __init__(self, X: np.ndarray, codes: np.ndarray, n_classes: int, weights: np.ndarray,
                 fit_intercept: bool):
        self.X = np.ascontiguousarray(X, dtype=float)
        self.n, self.p = self.X.shape
        self.K = int(n_classes)
        self.binary = self.K == 2
        self.rows = 1 if self.binary else self.K
        self.w = np.asarray(weights, dtype=float) / float(np.sum(weights))
        self.fit_intercept = bool(fit_intercept)
        self.codes = np.asarray(codes, dtype=np.int64)
        self.Y = np.zeros((self.n, self.K))
        self.Y[np.arange(self.n), self.codes] = 1.0
        self.n_free = (1 if self.binary else self.K - 1) if self.fit_intercept else 0
        self.m = self.rows * self.p + self.n_free
        self.penalized = np.zeros(self.m, dtype=bool)
        self.penalized[: self.rows * self.p] = True
        self.Z = np.hstack([self.X, np.ones((self.n, 1))]) if self.fit_intercept else self.X
        q = self.Z.shape[1]
        self.order = np.asarray([k * q + j for k in range(self.rows) for j in range(self.p)]
                                + [k * q + self.p for k in range(self.n_free)], dtype=np.int64)

    def unpack(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """θ = [the coefficients row by row, the free intercepts] as ``(W, b)``."""
        W = theta[: self.rows * self.p].reshape(self.rows, self.p)
        b = np.zeros(self.rows)
        b[: self.n_free] = theta[self.rows * self.p:]
        return W, b

    def scores(self, theta: np.ndarray) -> np.ndarray:
        W, b = self.unpack(theta)
        return self.X @ W.T + b

    def loss(self, eta: np.ndarray) -> float:
        if self.binary:
            e = eta[:, 0]
            return float(self.w @ (np.logaddexp(0.0, e) - self.Y[:, 1] * e))
        return float(self.w @ (logsumexp(eta, axis=1) - eta[np.arange(self.n), self.codes]))

    def gradient_hessian(self, eta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Z, q = self.Z, self.Z.shape[1]
        if self.binary:
            prob = expit(eta[:, 0])
            g = Z.T @ (self.w * (prob - self.Y[:, 1]))
            H = (Z * (self.w * prob * (1.0 - prob))[:, None]).T @ Z
        else:
            P = np.exp(eta - logsumexp(eta, axis=1, keepdims=True))
            g = (Z.T @ ((P - self.Y) * self.w[:, None])).T.ravel()  # class k: g[k*q:(k+1)*q]
            H = np.empty((self.K * q, self.K * q))
            for k in range(self.K):
                for j in range(k, self.K):
                    block = (Z * (self.w * P[:, k] * ((k == j) - P[:, j]))[:, None]).T @ Z
                    H[k * q:(k + 1) * q, j * q:(j + 1) * q] = block
                    H[j * q:(j + 1) * q, k * q:(k + 1) * q] = block.T
        return g[self.order], H[np.ix_(self.order, self.order)]

    def start(self) -> np.ndarray:
        """No coefficient, and the null model's intercepts (the weighted class shares)."""
        theta = np.zeros(self.m)
        if self.n_free:
            share = np.clip(self.w @ self.Y, 1e-300, None)
            theta[self.rows * self.p:] = (np.log(share[1] / share[0]) if self.binary
                                          else np.log(share[:-1] / share[-1]))
        return theta


def _logistic_solve(problem: _Logistic, lam: float, l1_ratio: float, theta: np.ndarray
                    ) -> tuple[np.ndarray, int, bool]:
    """The penalized fit at ``λ`` (the penalty per unit of weight), mix ``ρ``, started at
    ``theta``: ``(θ, Newton steps, converged)``."""
    gamma, ridge = lam * l1_ratio, lam * (1.0 - l1_ratio)
    pen = problem.penalized
    shift = np.diag(np.where(pen, ridge, 0.0))

    def objective(t: np.ndarray, eta: np.ndarray) -> float:
        coef = t[pen]
        return problem.loss(eta) + gamma * float(np.abs(coef).sum()) + 0.5 * ridge * float(coef @ coef)

    eta = problem.scores(theta)
    value = objective(theta, eta)
    for step in range(1, MAX_NEWTON + 1):
        g, H = problem.gradient_hessian(eta)
        g = g + ridge * np.where(pen, theta, 0.0)
        A = H + shift
        z, settled = feature_sign(A, A @ theta - g, gamma, pen, theta, ridge > 0.0)
        d = z - theta
        scale = max(1.0, float(np.max(np.abs(theta))))
        decrease = float(g @ d) + gamma * float(np.abs(z[pen]).sum() - np.abs(theta[pen]).sum())
        if settled and (float(np.max(np.abs(d))) <= STEP_TOL * scale
                        or decrease >= -4 * np.finfo(float).eps * max(1.0, abs(value))):
            return z, step, True  # the last Newton step: what remains is rounding
        t = 1.0
        while True:
            trial = theta + t * d
            eta_t = problem.scores(trial)
            v = objective(trial, eta_t)
            if v <= value + ARMIJO * t * decrease or t < 1e-10:
                break
            t *= 0.5
        theta, eta, value = trial, eta_t, v
    return theta, MAX_NEWTON, False


def logistic_path(X: np.ndarray, codes: np.ndarray, n_classes: int, Cs: Sequence[float],
                  l1_ratio: float, sample_weight: Any = None, fit_intercept: bool = True,
                  start: np.ndarray | None = None
                  ) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray, bool]:
    """``(coef, intercept)`` at each of ``Cs`` (warm-started in their order), the Newton steps
    each took, and whether every one converged. ``coef`` has one row for two classes and one per
    class otherwise; a multinomial's intercepts are centered."""
    weights = (np.ones(len(codes)) if sample_weight is None
               else np.broadcast_to(np.asarray(sample_weight, dtype=float), (len(codes),)))
    problem = _Logistic(X, codes, n_classes, weights, fit_intercept)
    total = float(np.sum(weights))
    theta = problem.start() if start is None else np.array(start, dtype=float)
    out: list[tuple[np.ndarray, np.ndarray]] = []
    steps = np.zeros(len(Cs), dtype=np.int64)
    every = True
    for i, C in enumerate(Cs):
        theta, steps[i], ok = _logistic_solve(problem, 1.0 / (float(C) * total), l1_ratio, theta)
        every &= ok
        W, b = problem.unpack(theta)
        if not problem.binary and problem.fit_intercept:
            b = b - b.mean()
        out.append((W.copy(), b.copy()))
    if not every:
        _warn("logistic path")
    return out, steps, every


def log_loss_rows(X: np.ndarray, codes: np.ndarray, coef: np.ndarray, intercept: np.ndarray
                  ) -> np.ndarray:
    """Each row's log loss under ``coef`` and ``intercept`` (one row of ``coef``: the logit)."""
    eta = np.asarray(X, dtype=float) @ np.atleast_2d(coef).T + intercept
    codes = np.asarray(codes, dtype=np.int64)
    if eta.shape[1] == 1:
        e = eta[:, 0]
        return np.logaddexp(0.0, e) - (codes == 1) * e
    return logsumexp(eta, axis=1) - eta[np.arange(len(codes)), codes]


__all__ = ["alpha_grid", "feature_sign", "gaussian_path", "log_loss_rows", "logistic_path"]

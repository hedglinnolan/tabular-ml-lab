"""Independent references for the EXPLORE package's acceptance tests (wave 2): each a definition
written out from its source, by hand in NumPy, never through the app's own code. Where R holds the
reference (``dcurves``, ``survey``, ``MASS``, ``Hmisc``, ``mice``, ``pROC``) the tests call it through
``r_reference.run_r``; these are the definitions R does not hold here, or the loops a test replays.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import pandas as pd


def caret_near_zero(values: Sequence[Any]) -> bool:
    """caret::nearZeroVar's rule with its defaults, as its documentation states it: "the frequency
    ratio of the most prevalent value over the second most frequent value … greater than
    ``freqCut`` (95/5)" and "the percent of unique values … less than ``uniqueCut`` (10)"; or one
    distinct value (zero variance)."""
    s = pd.Series(list(values)).dropna()
    counts = sorted(s.value_counts().tolist(), reverse=True)
    if len(counts) <= 1:
        return True
    return counts[0] / counts[1] > 95 / 5 and 100.0 * len(counts) / len(s) <= 10


def harrell_k(n: int) -> int:
    """RMS 2nd ed. §2.4.6 as written: "If the sample size is small (n < 30, say) k = 3 … If the
    sample is large (n > 100), k = 5", else k = 4."""
    return 3 if n < 30 else (5 if n > 100 else 4)


def rcs_columns(x: np.ndarray, knots: Sequence[float]) -> np.ndarray:
    """Harrell's restricted cubic spline basis (RMS 2nd ed. eq. 2.25, norm = 2), x first, written
    out from the formula: (x − t_j)₊³ − (x − t_{k−1})₊³ (t_k − t_j)/(t_k − t_{k−1})
    + (x − t_k)₊³ (t_{k−1} − t_j)/(t_k − t_{k−1}), all over (t_k − t_1)²."""
    t = np.asarray(knots, dtype=float)
    k = len(t)
    scale = (t[-1] - t[0]) ** 2
    cols = [np.asarray(x, dtype=float)]
    for j in range(k - 2):
        a = np.maximum(x - t[j], 0) ** 3
        b = np.maximum(x - t[k - 2], 0) ** 3 * (t[k - 1] - t[j]) / (t[k - 1] - t[k - 2])
        c = np.maximum(x - t[k - 1], 0) ** 3 * (t[k - 2] - t[j]) / (t[k - 1] - t[k - 2])
        cols.append((a - b + c) / scale)
    return np.column_stack(cols)


def harrell_knots(x: np.ndarray, k: int) -> np.ndarray:
    """Harrell's default knot percentiles for n ≥ 100 values and untied extremes (``rcspline.eval``:
    outer 0.10 for 3 knots, 0.05 for 4 and 5), as R's type-7 quantiles."""
    outer = 0.1 if k == 3 else 0.05
    return np.quantile(np.asarray(x, dtype=float), np.linspace(outer, 1 - outer, k),
                       method="linear")


def net_benefit(event: np.ndarray, risk: np.ndarray, t: float) -> float:
    """Vickers & Elkin (2006): TP/n − FP/n · t/(1 − t), positive when risk ≥ t."""
    pos = risk >= t
    n = len(event)
    return float(((pos & (event == 1)).sum() - (pos & (event == 0)).sum() * t / (1 - t)) / n)


def youden(event: np.ndarray, risk: np.ndarray, grid: Sequence[float]) -> float:
    """The grid threshold maximizing sensitivity + specificity − 1, the lowest among ties."""
    best, best_t = -np.inf, None
    for t in grid:
        pos = risk >= t
        j = pos[event == 1].mean() + (~pos[event == 0]).mean() - 1
        if j > best + 1e-15:
            best, best_t = j, float(t)
    return best_t


def interpretable_bbc(y: np.ndarray, preds: dict[str, np.ndarray], interpretable: str,
                      flexible: Sequence[str], *, task: str, B: int, seed: int,
                      units: np.ndarray | None = None) -> tuple[float, float, float]:
    """The cost or gain of the interpretable model by BBC-CV over the flexible set, replayed by
    hand: draw units with ``default_rng(seed).integers(0, U, U)``; choose the flexible family with
    the lowest in-bag mean squared error (numeric) or log loss (yes/no); difference = its out-of-bag
    loss − the interpretable model's (positive favors the interpretable one). Mean and 2.5/97.5
    percentiles."""
    n = len(y)
    if units is None:
        codes = np.arange(n)
    else:
        codes = pd.factorize(pd.Series(units).astype(str))[0]
    U = int(codes.max()) + 1

    def loss(p: np.ndarray, rows: np.ndarray) -> float:
        if task == "regression":
            return float(np.mean((y[rows] - p[rows]) ** 2))
        q = np.clip(p[rows], 1e-15, 1 - 1e-15)
        yy = y[rows]
        return float(-np.mean(yy * np.log(q) + (1 - yy) * np.log(1 - q)))

    rng = np.random.default_rng(seed)
    diffs = []
    rows_all = np.arange(n)
    for _ in range(B):
        draw = rng.integers(0, U, U)
        times = np.bincount(draw, minlength=U)[codes]
        if times.all():
            continue
        drawn = np.repeat(rows_all, times)
        out = rows_all[times == 0]
        if len(out) < 2:
            continue
        inbag = {k: loss(preds[k], drawn) for k in flexible}
        chosen = min(inbag, key=inbag.get)
        diffs.append(loss(preds[chosen], out) - loss(preds[interpretable], out))
    return float(np.mean(diffs)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def vip_from_sklearn(Z: np.ndarray, y: np.ndarray, components: int = 2) -> np.ndarray:
    """VIP from scikit-learn's PLSRegression (its own NIPALS): Wold's formula over its x-weights,
    x-scores and y-loadings (Mehmood et al. 2012, eq. 1)."""
    from sklearn.cross_decomposition import PLSRegression

    pls = PLSRegression(n_components=components, scale=False).fit(Z, y - y.mean())
    W, T, Q = pls.x_weights_, pls.x_scores_, pls.y_loadings_
    ss = (Q[0] ** 2) * (T ** 2).sum(axis=0)
    p = Z.shape[1]
    Wn = W / np.linalg.norm(W, axis=0)
    return np.sqrt(p * (Wn ** 2 @ ss) / ss.sum())


def weighted_fold_means(losses: np.ndarray, weights: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """Each fold's Σ wℓ / Σ w."""
    return np.asarray([float((weights[folds == f] * losses[folds == f]).sum()
                             / weights[folds == f].sum()) for f in np.unique(folds)])


def corrected_t(d: np.ndarray, share: float) -> tuple[float, float]:
    """Nadeau & Bengio's corrected resampled t over the paired differences ``d``: (mean, se)."""
    k = len(d)
    return float(d.mean()), math.sqrt((1 / k + share) * float(d.var(ddof=1)))


__all__ = ["caret_near_zero", "corrected_t", "harrell_k", "harrell_knots", "interpretable_bbc",
           "net_benefit", "rcs_columns", "vip_from_sklearn", "weighted_fold_means", "youden"]

"""Independent references for the MS6 acceptance tests (prediction validation).

Each function computes a published quantity from its definition by a route of its own, without
importing the engine: the corrected repeated k-fold t written from Bouckaert & Frank's formula,
BBC-CV written from Tsamardinos et al.'s algorithm with explicit loops, log loss and the ranked
probability score from their sums, Graf et al.'s censoring-weighted Brier score with the censoring
distribution's Kaplan–Meier product written out. :func:`run_r` runs an R script (test-only; the
app never calls R) on CSV files and reads back the JSON it writes.
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

RSCRIPT = shutil.which("Rscript")


def run_r(script: str, frames: dict[str, pd.DataFrame], timeout: int = 600) -> dict[str, Any]:
    """Write each frame to ``<name>.csv`` in a fresh folder, run ``script`` there with ``Rscript``
    (it must leave its answer in ``out.json``), and return the parsed JSON."""
    if RSCRIPT is None:
        raise RuntimeError("Rscript is not installed")
    with tempfile.TemporaryDirectory() as folder:
        for name, frame in frames.items():
            frame.to_csv(Path(folder) / f"{name}.csv", index=False)
        (Path(folder) / "script.R").write_text(script)
        done = subprocess.run([RSCRIPT, "--vanilla", "script.R"], cwd=folder, capture_output=True,
                              text=True, timeout=timeout)
        if done.returncode != 0:
            raise RuntimeError(f"R failed:\n{done.stdout}\n{done.stderr}")
        return json.loads((Path(folder) / "out.json").read_text())


# ── the corrected repeated k-fold t (Bouckaert & Frank 2004) ─────────────────


def corrected_repeated_t(a: Sequence[float], b: Sequence[float], n_train: Sequence[int],
                         n_test: Sequence[int], *, lower_is_better: bool) -> tuple[float, int, float]:
    """(t, df, se) for r repeats of K folds: ``x_ij = a_ij − b_ij`` (``b − a`` when lower is
    better), ``t = mean(x) / √((1/(rK) + n₂/n₁)·σ̂²)`` with σ̂² the sample variance of the rK
    differences and ``n₂/n₁`` the mean over folds of rows tested over rows trained; df ``rK − 1``.
    Written as sums over an explicit loop."""
    rk = len(a)
    x = []
    for i in range(rk):
        x.append((b[i] - a[i]) if lower_is_better else (a[i] - b[i]))
    mean = sum(x) / rk
    var = sum((v - mean) ** 2 for v in x) / (rk - 1)
    ratio = sum(n_test[i] / n_train[i] for i in range(rk)) / rk
    se = math.sqrt((1.0 / rk + ratio) * var)
    return mean / se, rk - 1, se


# ── per-row losses written out ───────────────────────────────────────────────


def log_loss(y: np.ndarray, proba: np.ndarray, classes: Sequence[Any]) -> float:
    """Mean of −log p(observed class), p clipped to [eps, 1 − eps] and renormalized, as
    scikit-learn clips (the engine's convention), summed in a loop."""
    eps = np.finfo(float).eps
    total = 0.0
    for i in range(len(y)):
        row = np.clip(proba[i], eps, 1 - eps)
        row = row / row.sum()
        total += -math.log(row[list(classes).index(y[i])])
    return total / len(y)


def auc(positive: np.ndarray, score: np.ndarray) -> float:
    """The Mann–Whitney AUC by the m × n comparison matrix, ties one half."""
    pos, neg = score[positive], score[~positive]
    if not len(pos) or not len(neg):
        return float("nan")
    k = (pos[:, None] > neg[None, :]).astype(float) + 0.5 * (pos[:, None] == neg[None, :])
    return float(k.mean())


# ── BBC-CV (Tsamardinos, Greasidou & Borboudakis 2018, algorithm 5) ──────────


def bbc_cv(y: np.ndarray, predictions: dict[str, np.ndarray], classes: Sequence[Any], *,
           metric: str, B: int, seed: int, units: np.ndarray | None = None,
           extra: str | None = None) -> dict[str, Any]:
    """BBC-CV over fixed out-of-fold predictions: B times, draw U units with replacement
    (``default_rng(seed).integers(0, U, U)``, units in order of first appearance; rows when no
    units), choose the configuration with the best ``metric`` on the drawn units' rows (ties: the
    first listed), score it on the rows of the units not drawn. Returns the mean, the 2.5th and
    97.5th percentiles, the wins, and the same for ``extra`` on the same choices."""
    n = len(y)
    labels = list(range(n)) if units is None else [str(u) for u in units]
    order: list[str] = []
    rows_of: dict[str, list[int]] = {}
    for i, u in enumerate(labels):
        key = str(u)
        if key not in rows_of:
            rows_of[key] = []
            order.append(key)
        rows_of[key].append(i)
    unit_rows = [rows_of[u] for u in order]
    U = len(unit_rows)
    rng = np.random.default_rng(seed)
    keys = list(predictions)

    def value(name: str, key: str, rows: list[int]) -> float:
        yy, pp = y[rows], predictions[key][rows]
        if name == "log_loss":
            return log_loss(yy, pp, classes)
        if name == "auc":
            return auc(yy == classes[1], pp[:, 1])
        raise KeyError(name)

    lower = metric == "log_loss"
    scores, extras = [], []
    wins = {k: 0 for k in keys}
    for _ in range(B):
        draw = rng.integers(0, U, U)
        drawn = [r for u in draw for r in unit_rows[u]]
        left = [r for u in range(U) if u not in set(draw.tolist()) for r in unit_rows[u]]
        if not left:
            continue
        best, best_value = None, None
        for k in keys:
            v = value(metric, k, drawn)
            if best is None or (v < best_value if lower else v > best_value):
                best, best_value = k, v
        scores.append(value(metric, best, left))
        wins[best] += 1
        if extra:
            extras.append(value(extra, best, left))
    out = {"corrected": float(np.mean(scores)), "low": float(np.percentile(scores, 2.5)),
           "high": float(np.percentile(scores, 97.5)), "wins": wins, "replicates": len(scores)}
    if extra:
        out["extra"] = {"corrected": float(np.mean(extras)),
                        "low": float(np.percentile(extras, 2.5)),
                        "high": float(np.percentile(extras, 97.5))}
    return out


# ── Graf et al.'s Brier score at a horizon ───────────────────────────────────


def censoring_km_left(time: np.ndarray, event: np.ndarray, at: float, *, left: bool) -> float:
    """Ĝ(at) (or its left limit Ĝ(at−)): the product over censoring times s ≤ at (s < at) of
    (1 − c_s/n_s), n_s every row whose time is ≥ s, written as a loop over the distinct times."""
    G = 1.0
    for s in sorted(set(time[~event].tolist())):
        if (s >= at) if left else (s > at):
            break
        c = int(np.sum((time == s) & ~event))
        n_s = int(np.sum(time >= s))
        G *= 1.0 - c / n_s
    return G


def graf_brier(time: np.ndarray, event: np.ndarray, risk: np.ndarray, horizon: float) -> float:
    """(1/n) Σ [S_i² 1{T_i ≤ h, δ_i}/Ĝ(T_i−) + F_i² 1{T_i > h}/Ĝ(h)], F = predicted risk by h,
    S = 1 − F (Graf, Schmoor, Sauerbrei & Schumacher, Stat Med 1999;18:2529)."""
    total = 0.0
    G_h = censoring_km_left(time, event, horizon, left=False)
    for i in range(len(time)):
        if time[i] <= horizon and event[i]:
            total += (1.0 - risk[i]) ** 2 / censoring_km_left(time, event, time[i], left=True)
        elif time[i] > horizon:
            total += risk[i] ** 2 / G_h
    return total / len(time)


def breslow_risk(train_time: np.ndarray, train_event: np.ndarray, train_lp: np.ndarray,
                 lp: np.ndarray, horizon: float) -> np.ndarray:
    """1 − exp(−Λ₀(h) e^{lp}) with Breslow's Λ₀(h) = Σ_{event times s ≤ h} d_s / Σ_{T_j ≥ s} e^{lp_j},
    written as a loop (no delayed entry)."""
    H = 0.0
    for s in sorted(set(train_time[train_event].tolist())):
        if s > horizon:
            break
        d = int(np.sum((train_time == s) & train_event))
        H += d / float(np.sum(np.exp(train_lp[train_time >= s])))
    return 1.0 - np.exp(-H * np.exp(lp))


# ── the ranked probability score ─────────────────────────────────────────────


def rps(codes: np.ndarray, proba: np.ndarray) -> float:
    """Mean over rows of (1/(K − 1)) Σ_{k < K−1} (Σ_{j ≤ k} p_j − 1{y ≤ k})² (Epstein 1969)."""
    K = proba.shape[1]
    total = 0.0
    for i in range(len(codes)):
        cum = 0.0
        row = 0.0
        for k in range(K - 1):
            cum += proba[i, k]
            row += (cum - (1.0 if codes[i] <= k else 0.0)) ** 2
        total += row / (K - 1)
    return total / len(codes)

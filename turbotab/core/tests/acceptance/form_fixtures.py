"""Fixtures for the FORM acceptance tests (MODELING_SEQUENCE §1 rows 5 and 7).

A dietary cohort whose generator declares its truth: ``protein_g`` (an energy-bearing exposure, g)
and ``kcal`` (total energy) drive ``glucose``, the protein effect curved; ``age`` and ``sex``,
``smoking`` cause the exposure and the outcome (confounders); ``activity`` (eight whole values)
causes the outcome only. ``fish_g`` is a food most people do not eat (a mass at zero). Every
expected number in the tests comes from these columns by an independent path (R, NumPy written out
by hand, or the generator's truth), never from the engine.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

ROLES = {"pid": "identifier", "age": "covariate", "sex": "covariate", "smoking": "covariate",
         "activity": "covariate", "kcal": "energy", "protein_g": "exposure",
         "fish_g": "excluded"}
TRUTH = {"code_or_count:smoking": "amount", "code_or_count:activity": "amount",
         "code_or_count:pid": "amount", "code_or_count:kcal": "amount", "day_count:kcal": "1",
         "unit:protein_g": "g", "unit:kcal": "kcal",
         "adjust:age": "yes,yes,no", "adjust:sex": "yes,yes,no", "adjust:smoking": "yes,yes,no",
         "adjust:activity": "no,yes,no"}


def diet(n: int = 900, seed: int = 41) -> pd.DataFrame:
    """The dietary cohort: glucose curves in protein (a plateau above 90 g), rises with age and
    smoking, falls with activity."""
    rng = np.random.default_rng(seed)
    age = rng.normal(52, 9, n).round(1)
    sex = rng.choice(["female", "male"], n)
    smoking = rng.binomial(1, 0.3, n)
    activity = rng.integers(0, 8, n)
    kcal = rng.normal(2100, 400, n).round(0)
    protein_g = (0.035 * kcal + 0.15 * (age - 52) - 4 * smoking + rng.normal(0, 12, n)).round(1)
    fish_g = np.where(rng.random(n) < 0.6, 0.0, rng.gamma(2.0, 30.0, n)).round(1)
    curve = -0.08 * np.minimum(protein_g, 90) - 0.00003 * (kcal - 2100) ** 2 / 100
    glucose = (110 + curve + 0.3 * (age - 52) + 5 * smoking - 1.1 * (activity - 3.5)
               + 1.5 * (sex == "male") - 0.02 * fish_g + rng.normal(0, 6, n)).round(2)
    return pd.DataFrame({"pid": np.arange(1, n + 1), "age": age, "sex": sex, "smoking": smoking,
                         "activity": activity, "kcal": kcal, "protein_g": protein_g,
                         "fish_g": fish_g, "glucose": glucose})


def open_inference(drive: Any, *, roles: dict[str, str] | None = None, target: str = "glucose",
                   task: str = "regression") -> None:
    """Lens, outcome, purpose, grain, roles, eligibility, complete cases and no holdout."""
    drive.decide({"kind": "set_lens", "lenses": ["dietary"]})
    drive.reach("target")
    drive.decide({"kind": "set_target", "column": target})
    drive.answer("task", {"kind": "set_task", "column": target, "task": task})
    drive.reach("purpose")
    drive.decide({"kind": "set_purpose", "purpose": "inference"})
    drive.answer("grain", {"kind": "set_grain", "grain": "one_row_per_unit", "id_column": "pid"})
    drive.reach("roles")
    drive.decide_roles(dict(ROLES if roles is None else roles))
    drive.answer("exclusions", {"kind": "set_exclusions", "rules": []})
    drive.answer("missing", {"kind": "set_missing", "strategy": "complete_case"})
    drive.answer("split", {"kind": "set_split", "holdout": 0.0, "seed": 0, "folds": 5})


def harrell_knots(x: Any, k: int) -> np.ndarray:
    """Harrell's percentiles written out (n ≥ 100, no tied extreme): R type-7 quantiles at
    0.10/0.50/0.90 (k = 3), 0.05/0.35/0.65/0.95 (k = 4), 0.05/0.275/0.5/0.725/0.95 (k = 5)."""
    p = {3: [0.10, 0.50, 0.90], 4: [0.05, 0.35, 0.65, 0.95],
         5: [0.05, 0.275, 0.50, 0.725, 0.95]}[k]
    v = np.sort(np.asarray(x, dtype=float))
    h = (len(v) - 1) * np.asarray(p)
    lo = np.floor(h).astype(int)
    return v[lo] + (h - lo) * (v[np.minimum(lo + 1, len(v) - 1)] - v[lo])


def ols_hc3(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least squares and its HC3 covariance written out (MacKinnon & White 1985)."""
    inv = np.linalg.inv(X.T @ X)
    beta = inv @ X.T @ y
    e = y - X @ beta
    h = np.einsum("ij,jk,ik->i", X, inv, X)
    meat = (X * (e / (1 - h))[:, None]).T @ (X * (e / (1 - h))[:, None])
    return beta, inv @ meat @ inv


def wald_f(beta: np.ndarray, cov: np.ndarray, idx: list[int], df: float) -> tuple[float, float]:
    """F = b'V⁻¹b / q on (q, df) and its p, written out."""
    from scipy import stats

    b = beta[idx]
    V = cov[np.ix_(idx, idx)]
    F = float(b @ np.linalg.solve(V, b)) / len(idx)
    return F, float(stats.f.sf(F, len(idx), df))


__all__ = ["ROLES", "TRUTH", "diet", "harrell_knots", "ols_hc3", "open_inference", "wald_f"]

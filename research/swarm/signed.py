"""The signed change in a fit's primary score, from the per-fold scores the record keeps (SWARM_PLAN
§7, "Material, in the loss's units").

Two fits scored on the same rows in the same folds (``GET …/fits``' ``folds`` digest is the same)
are **paired**: over the J = r·K fold scores of r repeats of K folds, the differences d = after −
before have the corrected repeated k-fold variance of Nadeau & Bengio (2003), as Bouckaert & Frank
(2004) apply it, (1/J + n_test/n_train)·s²_d, with n_test/n_train = 1/(K − 1), on J − 1 degrees of
freedom.

A decision that changes the analyzed rows (an exclusion, an eligibility rule, complete cases
against a fill) leaves no shared folds, so the two fits' folds differ: the change is **unpaired**,
the difference of the two means, each arm's variance of its mean inflated by the same factor over
its own scores, (1/J_arm + 1/(K_arm − 1))·s²_arm, since repeated folds are correlated; the interval
is Welch's over those two variances, its degrees of freedom by the Welch–Satterthwaite rule.

The change is signed on the metric's own scale (``difference``: after less before), and
``better`` says whether the later fit scored better, by the metric's direction (a loss is better
lower). Beside it is half the earlier fit's own interval width (``half_width``), the size §7's rule
compares the change with; whether a change is material is the pre-registration's to say
(``research/swarm/prereg.json``), not this module's. A change with fewer than two fold scores in
an arm, a paired change whose arms are not the same folds, or scores that are not k-fold means
has no number, and says why.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

LEVEL = 0.95


def _finite(values: Sequence[Any]) -> list[float]:
    out = []
    for v in values:
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(x):
            out.append(x)
    return out


def _variance(xs: Sequence[float]) -> float:
    m = sum(xs) / len(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)


def _t(df: float, level: float) -> float:
    from scipy import stats

    return float(stats.t.ppf(0.5 + level / 2, df))


def folds_per_repeat(score: Mapping[str, Any]) -> int | None:
    """K, the folds in each repeat of a k-fold score as the fit record serves it (its ``folds``
    list over its ``repeats``), or None where the score is no k-fold mean of at least two folds."""
    folds = score.get("folds")
    repeats = int(score.get("repeats") or 1)
    if not isinstance(folds, list) or repeats < 1 or len(folds) % repeats:
        return None
    if score.get("estimator") not in (None, "fold_mean"):
        return None
    k = len(folds) // repeats
    return k if k >= 2 else None


def paired(after: Sequence[Any], before: Sequence[Any], k: int, *,
           level: float = LEVEL) -> dict[str, Any]:
    """The paired change over the same folds, in order (the module docstring): fold pairs where
    either score is missing are left out, as the engine's own comparison leaves them."""
    if len(after) != len(before):
        return {"why": f"The two fits hold {len(after)} and {len(before)} fold scores, so they "
                       f"are not the same folds."}
    d = [float(a) - float(b) for a, b in zip(after, before)
         if _finite([a]) and _finite([b])]
    if len(d) < 2:
        return {"why": "Fewer than two folds hold both fits' scores."}
    j = len(d)
    mean = sum(d) / j
    se = math.sqrt((1.0 / j + 1.0 / (k - 1)) * _variance(d))
    q = _t(j - 1, level)
    return {"difference": mean, "se": se, "df": float(j - 1), "ci_low": mean - q * se,
            "ci_high": mean + q * se, "folds": [j, j]}


def unpaired(after: Sequence[Any], before: Sequence[Any], k_after: int, k_before: int, *,
             level: float = LEVEL) -> dict[str, Any]:
    """The unpaired change between two fits on different rows or folds (the module docstring)."""
    a, b = _finite(after), _finite(before)
    if len(a) < 2 or len(b) < 2:
        return {"why": "An arm holds fewer than two fold scores."}
    va = (1.0 / len(a) + 1.0 / (k_after - 1)) * _variance(a)
    vb = (1.0 / len(b) + 1.0 / (k_before - 1)) * _variance(b)
    mean = sum(a) / len(a) - sum(b) / len(b)
    se = math.sqrt(va + vb)
    denominator = (va ** 2 / (len(a) - 1) if va else 0.0) + (vb ** 2 / (len(b) - 1) if vb else 0.0)
    if se == 0 or denominator == 0:
        return {"difference": mean, "se": 0.0, "df": None, "ci_low": mean, "ci_high": mean,
                "folds": [len(a), len(b)]}
    df = (va + vb) ** 2 / denominator
    q = _t(df, level)
    return {"difference": mean, "se": se, "df": df, "ci_low": mean - q * se,
            "ci_high": mean + q * se, "folds": [len(a), len(b)]}


def change(after: Mapping[str, Any], before: Mapping[str, Any], *, same_folds: bool,
           higher_is_better: bool, level: float = LEVEL) -> dict[str, Any]:
    """The signed change from ``before``'s score to ``after``'s (each a metric's summary as the
    fit record serves it: ``folds``, ``repeats``, ``estimator``, ``ci_low``, ``ci_high``), paired
    where the fits scored the same folds (``same_folds``), else unpaired; each key of
    ``record.SignedChange`` but its fit and family."""
    k_after, k_before = folds_per_repeat(after), folds_per_repeat(before)
    out: dict[str, Any] = {"paired": bool(same_folds), "difference": None, "better": None,
                           "se": None, "df": None, "ci_low": None, "ci_high": None,
                           "excludes_zero": None, "folds": None, "half_width": None, "why": None}
    low, high = before.get("ci_low"), before.get("ci_high")
    if _finite([low]) and _finite([high]):
        out["half_width"] = (float(high) - float(low)) / 2
    if k_after is None or k_before is None:
        out["why"] = "A fit's score is no k-fold mean of at least two folds."
        return out
    if same_folds:
        got = paired(after["folds"], before["folds"], k_after, level=level) \
            if k_after == k_before else {"why": "The two fits' folds are not the same scheme."}
    else:
        got = unpaired(after["folds"], before["folds"], k_after, k_before, level=level)
    out.update(got)
    if out["difference"] is not None:
        diff = out["difference"]
        out["better"] = None if diff == 0 else (diff > 0) == higher_is_better
        out["excludes_zero"] = out["ci_low"] > 0 or out["ci_high"] < 0
    return out


__all__ = ["LEVEL", "change", "folds_per_repeat", "paired", "unpaired"]

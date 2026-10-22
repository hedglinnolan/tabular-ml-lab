"""The signed change in a fit's primary score, from the per-fold scores the record keeps (SWARM_PLAN
§7, "Material, in the loss's units").

Two fits scored on the same rows in the same folds (``GET …/fits``' ``folds`` digest is the same)
are **paired**: over the J = r·K fold scores of r repeats of K folds, the differences d = after −
before have the corrected repeated k-fold variance of Nadeau & Bengio (2003), as Bouckaert & Frank
(2004) apply it, (1/J + n_test/n_train)·s²_d, on J − 1 degrees of freedom: the engine's own
corrected t (``models.validation.corrected_t``), with its measured n_test/n_train
(``metrics.CrossValidated.test_share``: the mean over folds of the rows a fold scores over the
rows it is fit on).

**Which folds it covers.** The fit record serves no fold's size, so the share is read from the
facts the split stage served for the fit's folds (:func:`test_share`; the driver keeps them beside
each fit's live markers): random K-fold, once or repeated (under the bootstrap's validation too,
whose optimism is estimated beside a headline scored on the split's one run of K folds), over rows
no unit groups, whose K folds hold ⌊N/K⌋ or ⌈N/K⌉ of the N training rows each (scikit-learn's
``KFold`` and ``StratifiedKFold``, which the split draws them with), so the share is the mean of
n_j/(N − n_j) over those sizes, the engine's measured one. Time-ordered folds (each fold fit on
the folds before it), internal–external validation (one fold per cluster) and folds grouped by
unit have sizes those facts do not fix: there the change has no number, and says why, as it has
none where the facts were not read.

A decision that changes the analyzed rows (an exclusion, an eligibility rule, complete cases
against a fill) leaves no shared folds, so the two fits' folds differ: the change is **unpaired**,
the difference of the two means, each arm's variance of its mean inflated by the same factor over
its own scores, (1/J_arm + share_arm)·s²_arm, since repeated folds are correlated; the interval is
Welch's over those two variances, its degrees of freedom by the Welch–Satterthwaite rule.

**A pooled score.** A numeric outcome's MSE (and MAE) is pooled over every out-of-fold prediction
(``metrics.summarize``: ``estimator`` "pooled"), the folds' scores weighted by their sizes, which
is the mean of the fold scores only where every fold holds the same rows (N a multiple of K): the
change is computed there, and has no number where the sizes differ. A pooled score that is no
weighted mean of the fold scores (the RMSE, R²) has none either.

The change is signed on the metric's own scale (``difference``: after less before), and
``better`` says whether the later fit scored better, by the metric's direction (a loss is better
lower). Beside it is half the earlier fit's own interval width (``half_width``), the size §7's rule
compares the change with, and each arm's share it used (``test_share``, the later fit's first);
whether a change is material is
the pre-registration's to say (``research/swarm/prereg.json``), not this module's. A change with
fewer than two fold scores in an arm, a paired change whose arms are not the same folds, or scores
that are not k-fold scores has no number, and says why.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

LEVEL = 0.95
# The validations whose headline is scored on random K-fold (``decisions.Validation``): once,
# repeated, or once beside the bootstrap's optimism (``stages.rows.split_stage`` draws its folds
# as for "kfold"; ``stages.modeling`` scores the headline on them).
RANDOM_KFOLD = ("kfold", "repeated_kfold", "bootstrap")
# The pooled scores that are their folds' size-weighted mean (``metrics.pooled``: SSE/n, SAE/n).
WEIGHTED_MEAN = ("mse", "mae")
# The split stage's facts a share is read from (``stages.rows.draw_split``'s, and its validation).
SPLIT_FACTS = ("n_train", "folds", "repeats", "fold_scheme", "validation", "grouped_by")
NO_SPLIT = ("The facts of the folds this fit was scored on were not read (the driver keeps them at "
            "each fit), so the rows each fold scores are not known.")


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


def split_facts(split: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The facts of :data:`SPLIT_FACTS` from the split stage's served artifact, or None."""
    if not isinstance(split, Mapping):
        return None
    return {k: split.get(k) for k in SPLIT_FACTS}


def fold_sizes(split: Mapping[str, Any] | None) -> tuple[list[int] | None, str | None]:
    """The rows each of the K folds of one repeat scores, where the split's facts fix them (the
    module docstring), largest first; else None and why."""
    if not isinstance(split, Mapping):
        return None, NO_SPLIT
    validation = split.get("validation") or "kfold"
    if split.get("fold_scheme") == "time_ordered":
        return None, ("The folds are time-ordered, each scored by a model fit on the folds before "
                      "it, and the record holds no fold's size, so the share of rows each fold "
                      "scores is not known.")
    if split.get("fold_scheme") not in (None, "random"):
        return None, f"The folds' scheme is {split.get('fold_scheme')!r}, not random K-fold."
    if validation not in RANDOM_KFOLD:
        return None, (f"The validation is {validation!r}, not random K-fold, and the record holds "
                      f"no fold's size, so the share of rows each fold scores is not known.")
    if split.get("grouped_by"):
        return None, (f"The folds keep each `{split['grouped_by']}`'s rows together, so their "
                      f"sizes are the units', which the record does not hold.")
    try:
        n, k = int(split["n_train"]), int(split["folds"])
    except (KeyError, TypeError, ValueError):
        return None, "The split served no count of its training rows or folds."
    if k < 2 or n < 2 * k:
        return None, f"{n:,} training rows in {k} folds leave a fold too small to fit or score."
    q, r = divmod(n, k)
    return [q + 1] * r + [q] * (k - r), None


def test_share(split: Mapping[str, Any] | None) -> tuple[float | None, str | None]:
    """n_test/n_train as the engine measures it on these folds (the module docstring): the mean
    over the folds of the rows each scores over the rows it is fit on; else None and why."""
    sizes, why = fold_sizes(split)
    if sizes is None:
        return None, why
    n = sum(sizes)
    return sum(s / (n - s) for s in sizes) / len(sizes), None


def folds_per_repeat(score: Mapping[str, Any]) -> int | None:
    """K, the folds in each repeat of a k-fold score as the fit record serves it (its ``folds``
    list over its ``repeats``), or None where the score is no k-fold score of at least two folds
    (a mean over folds, or pooled over them)."""
    folds = score.get("folds")
    repeats = int(score.get("repeats") or 1)
    if not isinstance(folds, list) or repeats < 1 or len(folds) % repeats:
        return None
    if score.get("estimator") not in (None, "fold_mean", "pooled"):
        return None
    k = len(folds) // repeats
    return k if k >= 2 else None


def paired(after: Sequence[Any], before: Sequence[Any], share: float, *,
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
    se = math.sqrt((1.0 / j + share) * _variance(d))
    q = _t(j - 1, level)
    return {"difference": mean, "se": se, "df": float(j - 1), "ci_low": mean - q * se,
            "ci_high": mean + q * se, "folds": [j, j]}


def unpaired(after: Sequence[Any], before: Sequence[Any], share_after: float,
             share_before: float, *, level: float = LEVEL) -> dict[str, Any]:
    """The unpaired change between two fits on different rows or folds (the module docstring)."""
    a, b = _finite(after), _finite(before)
    if len(a) < 2 or len(b) < 2:
        return {"why": "An arm holds fewer than two fold scores."}
    va = (1.0 / len(a) + share_after) * _variance(a)
    vb = (1.0 / len(b) + share_before) * _variance(b)
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


def _arm(score: Mapping[str, Any], split: Mapping[str, Any] | None, metric: str | None
         ) -> tuple[float | None, str | None]:
    """One fit's share, or why its scores give no change: its folds' facts, and a pooled score
    that is the mean of its fold scores (the module docstring)."""
    k = folds_per_repeat(score)
    if k is None:
        return None, "A fit's score is no k-fold score of at least two folds."
    share, why = test_share(split)
    if share is None:
        return None, why
    sizes, _ = fold_sizes(split)
    repeats = int(score.get("repeats") or 1)
    if len(sizes) != k or int(split.get("repeats") or 1) != repeats:
        return None, (f"The fit's scores are of {repeats} × {k} folds, and the split it was read "
                      f"with draws {int(split.get('repeats') or 1)} × {len(sizes)}.")
    if score.get("estimator") == "pooled":
        name = metric or "score"
        if metric not in WEIGHTED_MEAN:
            return None, (f"The {name} is pooled over every out-of-fold prediction and is no "
                          f"weighted mean of the fold scores, so their mean is not the score.")
        if len(set(sizes)) > 1:
            return None, (f"The {name} is pooled over every out-of-fold prediction, the folds' "
                          f"size-weighted mean, and these {sum(sizes):,} training rows in {k} "
                          f"folds give folds of {max(sizes):,} and {min(sizes):,} rows, so the "
                          f"mean of the fold scores is not the score.")
    return share, None


def _empty(before: Mapping[str, Any], same_folds: bool) -> dict[str, Any]:
    out: dict[str, Any] = {"paired": bool(same_folds), "difference": None, "better": None,
                           "se": None, "df": None, "ci_low": None, "ci_high": None,
                           "excludes_zero": None, "folds": None, "half_width": None,
                           "test_share": None, "why": None}
    low, high = before.get("ci_low"), before.get("ci_high")
    if _finite([low]) and _finite([high]):
        out["half_width"] = (float(high) - float(low)) / 2
    return out


def from_shares(after: Mapping[str, Any], before: Mapping[str, Any], *, same_folds: bool,
                higher_is_better: bool, shares: tuple[float, float],
                level: float = LEVEL) -> dict[str, Any]:
    """:func:`change` once each arm's share is known (``shares``: the later fit's, then the
    earlier's), and both scores are k-fold scores it covers."""
    out = _empty(before, same_folds)
    if same_folds:  # one split's facts for both, each arm's scores of its folds (``_arm``)
        got = paired(after["folds"], before["folds"], shares[0], level=level)
    else:
        got = unpaired(after["folds"], before["folds"], shares[0], shares[1], level=level)
    out.update(got)
    if out["difference"] is not None:
        diff = out["difference"]
        out["test_share"] = [float(shares[0]), float(shares[1])]
        out["better"] = None if diff == 0 else (diff > 0) == higher_is_better
        out["excludes_zero"] = out["ci_low"] > 0 or out["ci_high"] < 0
    return out


def change(after: Mapping[str, Any], before: Mapping[str, Any], *, same_folds: bool,
           higher_is_better: bool, metric: str | None = None,
           after_split: Mapping[str, Any] | None = None,
           before_split: Mapping[str, Any] | None = None, level: float = LEVEL) -> dict[str, Any]:
    """The signed change from ``before``'s score to ``after``'s (each a metric's summary as the
    fit record serves it: ``folds``, ``repeats``, ``estimator``, ``ci_low``, ``ci_high``), paired
    where the fits scored the same folds (``same_folds``), else unpaired; each fit's share read
    from the facts of the split its folds were drawn by (``after_split``, ``before_split``: the
    split stage's, :data:`SPLIT_FACTS`); ``metric`` names the score. Each key of
    ``record.SignedChange`` but its fit and family."""
    if same_folds:
        after_split = after_split if after_split is not None else before_split
        before_split = after_split
    share_after, why = _arm(after, after_split, metric)
    share_before, why_before = _arm(before, before_split, metric) if why is None else (None, why)
    if share_after is None or share_before is None:
        out = _empty(before, same_folds)
        out["why"] = why or why_before
        return out
    return from_shares(after, before, same_folds=same_folds, higher_is_better=higher_is_better,
                       shares=(share_after, share_before), level=level)


__all__ = ["LEVEL", "NO_SPLIT", "RANDOM_KFOLD", "SPLIT_FACTS", "WEIGHTED_MEAN", "change",
           "fold_sizes", "folds_per_repeat", "from_shares", "paired", "split_facts", "test_share",
           "unpaired"]

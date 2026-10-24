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
(``metrics.summarize``: ``estimator`` "pooled"), in each repeat its folds' scores weighted by their
sizes, and the score the mean over the repeats. Where every fold holds the same rows (N a multiple
of K) that is the mean of the fold scores, and the change is computed on them as on any. Where the
sizes differ, the change is computed on the size-weighted mean of the fold scores, each fold
weighted by the rows the split's facts give it (:func:`fold_sizes`, in the order the fit scores
the folds), which is the pooled score: the difference of the two weighted means (paired, the
weighted mean of the differences), with the corrected variance on the fold scores as for any
change, (1/J + n_test/n_train)·s². A weighted mean that is not the pooled score the fit serves
(``estimate``, where it serves one) to :data:`AGREES` of it has no number, and says so: the folds
were weighted by sizes that are not theirs. A pooled score that is no weighted mean of the fold
scores (the RMSE, R²) has none either.

The change is signed on the metric's own scale (``difference``: after less before), and
``better`` says whether the later fit scored better, by the metric's direction (a loss is better
lower). Beside it is half the earlier fit's own interval width (``half_width``), the size §7's rule
compares the change with, and each arm's share it used (``test_share``, the later fit's first);
whether a change is material is
the pre-registration's to say (``research/swarm/prereg.json``), not this module's. A change with
fewer than two fold scores in an arm, a paired change whose arms are not the same folds, or scores
that are not k-fold scores has no number, and says why; so has a change with no fold-to-fold
spread, its differences (paired) or both its arms (unpaired) each one value but for float64's
rounding of the scores (:data:`ROUNDING`), whose interval would have no width and certify nothing
(the engine's own paired movement, ``runs.movement``, refuses the same). Whether two fits scored
the same in every fold but for that rounding is :func:`same_scores`'s to say (a sweep reads such a
change as zero, not as no number).
"""
from __future__ import annotations

import math
import sys
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
# Values that differ by no more than this many units of float64's precision of the largest score
# are one value: what rounding each score once, and their difference once, can part.
ROUNDING = 8
# A size-weighted mean of fold scores is the pooled score the fit serves where the two agree to
# this share of the score: float64 summing the same squared errors in another order parts them by
# far less, and a fold weighted by another fold's size by more wherever two folds' scores differ
# by more than this share of the score times the training rows.
AGREES = 1e-9
WEIGHTED_UNSCORED = ("A fold holds no score, so the size-weighted mean of the fold scores is not "
                     "the pooled score.")


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


def _no_spread(xs: Sequence[float], scores: Sequence[float]) -> bool:
    """Whether ``xs`` are one value but for float64's rounding of the ``scores`` they come from
    (:data:`ROUNDING`): their variance is then zero, or rounding's, and an interval on it has no
    width it can certify."""
    scale = max((abs(float(x)) for x in scores), default=0.0)
    return max(xs) - min(xs) <= ROUNDING * sys.float_info.epsilon * scale


def same_scores(after: Sequence[Any] | None, before: Sequence[Any] | None) -> bool:
    """Whether ``after``'s fold scores are ``before``'s, fold by fold, but for float64's rounding
    of each (:data:`ROUNDING` units of the precision of the largest score): the same count of
    folds, at least two, each scored in both."""
    if not isinstance(after, Sequence) or not isinstance(before, Sequence) \
            or isinstance(after, str) or isinstance(before, str):
        return False
    a, b = _finite(after), _finite(before)
    if len(after) != len(before) or len(a) != len(after) or len(b) != len(before) or len(a) < 2:
        return False
    scale = max(abs(x) for x in (*a, *b))
    return all(abs(x - y) <= ROUNDING * sys.float_info.epsilon * scale for x, y in zip(a, b))


def _mean(xs: Sequence[float], weights: Sequence[float] | None) -> float:
    """The mean of ``xs``, or their mean weighted by ``weights``."""
    if weights is None:
        return sum(xs) / len(xs)
    return sum(w * x for w, x in zip(weights, xs)) / sum(weights)


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
           level: float = LEVEL, weights: Sequence[float] | None = None) -> dict[str, Any]:
    """The paired change over the same folds, in order (the module docstring): fold pairs where
    either score is missing are left out, as the engine's own comparison leaves them. With
    ``weights`` (each fold score's rows: a pooled score on folds of different sizes) the change
    is the weighted mean of the differences, every fold scored in both."""
    if len(after) != len(before):
        return {"why": f"The two fits hold {len(after)} and {len(before)} fold scores, so they "
                       f"are not the same folds."}
    pairs = [(float(a), float(b)) for a, b in zip(after, before) if _finite([a]) and _finite([b])]
    if weights is not None and (len(pairs) != len(after) or len(weights) != len(after)):
        return {"why": WEIGHTED_UNSCORED}
    d = [a - b for a, b in pairs]
    if len(d) < 2:
        return {"why": "Fewer than two folds hold both fits' scores."}
    if _no_spread(d, [x for pair in pairs for x in pair]):
        # The engine's own paired movement refuses the same (``runs.movement``).
        moved = (f"changed by the same {d[0]:+.4g} from the earlier fit's" if d[0] else
                 "is the earlier fit's")
        return {"why": f"Every fold's score {moved}, so the change has no fold-to-fold spread and "
                       f"its corrected interval, of no width, is not defined."}
    j = len(d)
    mean = _mean(d, weights)
    se = math.sqrt((1.0 / j + share) * _variance(d))
    if se == 0:  # a spread too small for float64 to square
        return {"why": "The differences' spread is too small for float64 to give the change an "
                       "interval."}
    q = _t(j - 1, level)
    return {"difference": mean, "se": se, "df": float(j - 1), "ci_low": mean - q * se,
            "ci_high": mean + q * se, "folds": [j, j]}


def unpaired(after: Sequence[Any], before: Sequence[Any], share_after: float,
             share_before: float, *, level: float = LEVEL,
             weights: tuple[Sequence[float] | None, Sequence[float] | None] = (None, None)
             ) -> dict[str, Any]:
    """The unpaired change between two fits on different rows or folds (the module docstring):
    each arm's mean its weighted mean where its ``weights`` are given (the later fit's, then the
    earlier's: each fold score's rows, a pooled score on folds of different sizes), every fold of
    it scored."""
    a, b = _finite(after), _finite(before)
    if len(a) < 2 or len(b) < 2:
        return {"why": "An arm holds fewer than two fold scores."}
    for scores, kept, w in ((after, a, weights[0]), (before, b, weights[1])):
        if w is not None and (len(kept) != len(scores) or len(w) != len(scores)):
            return {"why": WEIGHTED_UNSCORED}
    if _no_spread(a, a) and _no_spread(b, b):
        return {"why": f"Each fit scored the same in every fold ({a[0]:.4g} and {b[0]:.4g}), so "
                       f"the change has no fold-to-fold spread and its interval, of no width, is "
                       f"not defined."}
    # An arm with no spread but rounding's adds none (Welch's interval rests on the other's).
    va = 0.0 if _no_spread(a, a) else (1.0 / len(a) + share_after) * _variance(a)
    vb = 0.0 if _no_spread(b, b) else (1.0 / len(b) + share_before) * _variance(b)
    mean = _mean(a, weights[0]) - _mean(b, weights[1])
    se = math.sqrt(va + vb)
    denominator = (va ** 2 / (len(a) - 1) if va else 0.0) + (vb ** 2 / (len(b) - 1) if vb else 0.0)
    if se == 0 or denominator == 0:  # spreads too small for float64 to square
        return {"why": "The fold scores' spread is too small for float64 to give the change an "
                       "interval."}
    df = (va + vb) ** 2 / denominator
    q = _t(df, level)
    return {"difference": mean, "se": se, "df": df, "ci_low": mean - q * se,
            "ci_high": mean + q * se, "folds": [len(a), len(b)]}


def _arm(score: Mapping[str, Any], split: Mapping[str, Any] | None, metric: str | None
         ) -> tuple[float | None, list[float] | None, str | None]:
    """One fit's share and, for a pooled score on folds of different sizes, each fold score's
    weight (its fold's rows, in the order the fit scores the folds, repeat after repeat); or why
    its scores give no change: its folds' facts, and a pooled score that is a weighted mean of its
    fold scores, the one the fit serves where it serves one (the module docstring)."""
    k = folds_per_repeat(score)
    if k is None:
        return None, None, "A fit's score is no k-fold score of at least two folds."
    share, why = test_share(split)
    if share is None:
        return None, None, why
    sizes, _ = fold_sizes(split)
    repeats = int(score.get("repeats") or 1)
    if len(sizes) != k or int(split.get("repeats") or 1) != repeats:
        return None, None, (f"The fit's scores are of {repeats} × {k} folds, and the split it "
                            f"was read with draws {int(split.get('repeats') or 1)} × "
                            f"{len(sizes)}.")
    if score.get("estimator") != "pooled":
        return share, None, None
    name = metric or "score"
    if metric not in WEIGHTED_MEAN:
        return None, None, (f"The {name} is pooled over every out-of-fold prediction and is no "
                            f"weighted mean of the fold scores, so their mean is not the score.")
    if len(set(sizes)) == 1:
        return share, None, None  # folds of one size: the pooled score is their scores' mean
    weights = [float(s) for s in sizes] * repeats
    folds = list(score.get("folds") or [])
    scored = _finite(folds)
    if len(scored) != len(folds):
        return None, None, WEIGHTED_UNSCORED
    served = score.get("estimate")
    if _finite([served]):
        weighted, served = _mean(scored, weights), float(served)
        if abs(weighted - served) > AGREES * max(abs(weighted), abs(served)):
            return None, None, (
                f"The {name}'s fold scores, each weighted by its fold's rows as the split's facts "
                f"give them ({', '.join(f'{s:,}' for s in sizes)} of {sum(sizes):,} in each "
                f"of {repeats} repeat{'s' if repeats != 1 else ''}), average to {weighted:.10g}, "
                f"which is not the {name} it serves, {served:.10g}: the folds are not weighted "
                f"by their own sizes, so the change has no number.")
    return share, weights, None


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
                level: float = LEVEL,
                weights: tuple[Sequence[float] | None, Sequence[float] | None] = (None, None)
                ) -> dict[str, Any]:
    """:func:`change` once each arm's share is known (``shares``: the later fit's, then the
    earlier's), and both scores are k-fold scores it covers; ``weights``, each arm's fold scores'
    weights where its score is pooled on folds of different sizes (``_arm``)."""
    out = _empty(before, same_folds)
    if same_folds:  # one split's facts for both, each arm's scores of its folds (``_arm``)
        got = paired(after["folds"], before["folds"], shares[0], level=level,
                     weights=weights[0])
    else:
        got = unpaired(after["folds"], before["folds"], shares[0], shares[1], level=level,
                       weights=weights)
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
    share_after, weights_after, why = _arm(after, after_split, metric)
    share_before, weights_before, why_before = _arm(before, before_split, metric) \
        if why is None else (None, None, why)
    if share_after is None or share_before is None:
        out = _empty(before, same_folds)
        out["why"] = why or why_before
        return out
    return from_shares(after, before, same_folds=same_folds, higher_is_better=higher_is_better,
                       shares=(share_after, share_before), level=level,
                       weights=(weights_after, weights_before))


__all__ = ["AGREES", "LEVEL", "NO_SPLIT", "RANDOM_KFOLD", "ROUNDING", "SPLIT_FACTS",
           "WEIGHTED_MEAN", "WEIGHTED_UNSCORED", "change", "fold_sizes", "folds_per_repeat",
           "from_shares", "paired", "same_scores", "split_facts", "test_share", "unpaired"]

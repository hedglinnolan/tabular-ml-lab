"""Whether a model does better than predicting without its predictors, with an honest interval.

M1 said when a family scores *worse* than its baseline (the outcome's training-fold mean, the
class prior, or for a time-to-event outcome one risk for everyone); M2 added "no better". The
comparison is paired, fold by fold, on the primary metric and the gain is the difference of the
two reported cross-validated scores (for R², the pooled estimate,
:mod:`turbotab.core.models.metrics`), signed so that a gain is an improvement: for a metric that
is better when higher (R², AUC, the C-index)

    d_k = model_k − baseline_k            gain = model CV score − baseline CV score

and the other way round for one that is better when lower (log loss, the multiclass primary).
Under repeated k-fold every repeat's folds enter, so with r repeats of K folds this is Bouckaert &
Frank's corrected repeated k-fold t (PAKDD 2004): ``1/(rK)`` in place of ``1/K``, ``rK − 1`` df.

**The interval corrects for the folds sharing training rows.** Fold scores are not independent:
any two folds' models share most of their training rows, so the spread of ``d_k`` understates the
uncertainty of their mean (Bengio & Grandvalet, JMLR 2004;5:1089–1105: "naive estimators … grossly
underestimate variance"). The corrected resampled t of Nadeau & Bengio (Machine Learning
2003;52:239–281), as Bouckaert & Frank (PAKDD 2004) apply it to k-fold cross-validation, inflates
the variance by the share of rows scored to rows fit:

    se = √( (1/K + n₂/n₁) · s²_d )        95% interval: gain ± t₀.₉₇₅,K−1 · se

**The verdict.** ``better`` when the whole 95% interval lies above zero and the gain is at least
the floor (:func:`min_gain`); ``worse`` when the gain is below zero; ``no_better`` otherwise. The
floor is a convention, not a cited rule: a gain below it is too small to matter whatever the
folds say. On a score that is better when higher it is ``MIN_GAIN``, 0.01 on the score's own
scale: R² (already the MSE's gain as a share of the baseline's, which is why a regression's
verdict is read on it), the AUC, Harrell's C. On a loss (log loss, the ranked probability score,
the Brier score at the horizon) it is ``MIN_GAIN`` of the baseline's own loss: a McFadden R², a
ranked probability skill score or a scaled Brier score of 0.01. A fixed 0.01 on a loss's own
scale called a rare outcome's model better by too little to matter at AUC 0.73 to 0.79 and 1% to
10% risk, where it gains 6% to 12% of the baseline's loss (claims backtest A-029).

Requiring the two-sided interval to clear zero is a one-sided test at 2.5%: the verdict and the
interval the reader sees never disagree. The rule it replaces — a gain above one fold
standard error — called a model "better" on 25–31% of binary datasets with no signal at all
(audit MA-10).
"""
from __future__ import annotations

import math
from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict

MIN_GAIN = 0.01  # the floor on a score better when higher; on a loss, this share of the baseline's
LEVEL = 0.95
# What a loss's floor is as a scaled score, 1 − loss / the baseline's loss (module docstring).
SKILL = {"log_loss": "a McFadden R²", "rps": "a ranked probability skill score",
         "brier_t": "a scaled Brier score", "brier": "a scaled Brier score", "mse": "an R²"}
Verdict = Literal["better", "no_better", "worse"]


class VersusBaseline(BaseModel):
    """A family's primary metric against its baseline's, paired over the same folds."""

    model_config = ConfigDict(extra="forbid", json_schema_serialization_defaults_required=True)

    verdict: Verdict
    metric: str
    gain: float | None  # the model's cross-validated score minus the baseline's
    tolerance: float  # the gain must exceed this to be "better": max(min_gain, t · se)
    tolerance_basis: str  # the rule in words, both of its parts named
    # The gain's 95% interval, corrected for the folds' shared training rows (Nadeau & Bengio).
    se: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    df: int | None = None  # folds − 1
    level: float = LEVEL


def min_gain(metric: str, baseline: float | None) -> float:
    """The floor a gain on ``metric`` must reach to matter (module docstring): ``MIN_GAIN`` on a
    score that is better when higher; on a loss, ``MIN_GAIN`` times ``baseline``, the baseline's
    cross-validated loss (``MIN_GAIN`` without a positive one)."""
    from turbotab.core.models.metrics import higher_is_better

    if (higher_is_better(metric) or baseline is None or not math.isfinite(float(baseline))
            or float(baseline) <= 0):
        return MIN_GAIN
    return MIN_GAIN * float(baseline)


def _spoken(metric: str) -> str:
    from turbotab.core.models.metrics import LABELS, SPOKEN

    return SPOKEN.get(metric, LABELS.get(metric, metric))


def _share_of(metric: str, against: str | None) -> str:
    """"1% of the class prior's log loss": a loss's floor as a share (``against``: the baseline as
    the sentence names it; the baseline otherwise)."""
    spoken = _spoken(metric).removeprefix("the ")
    against = against or "the baseline"
    whose = f"{against}'s {spoken}" if against.startswith("the ") else f"the {spoken} of {against}"
    return f"{MIN_GAIN:.0%} of {whose}"


def floor_words(metric: str, floor: float, against: str | None = None) -> str:
    """The floor as a sentence states it: "0.01 in R²" on a score better when higher; on a loss,
    its value and what it is a share of, "0.0061, 1% of the class prior's log loss"."""
    from turbotab.core.models.metrics import higher_is_better

    if higher_is_better(metric):
        return f"{floor:g} in {_spoken(metric)}"
    return f"{floor:.2g}, {_share_of(metric, against)}"


def _t(level: float, df: int) -> float:
    from scipy import stats

    return float(stats.t.ppf(1 - (1 - level) / 2, df))


def versus_baseline(metric: str, model_folds: Sequence[float | None],
                    baseline_folds: Sequence[float | None], *, gain: float | None = None,
                    test_share: float | None = None,
                    baseline: float | None = None) -> VersusBaseline:
    """The paired comparison and its verdict (see the module docstring).

    ``gain``: the improvement of the reported cross-validated score over the baseline's, signed
    as the per-fold differences are (default: their mean). ``test_share``: rows scored over rows
    fit, per fold, on average (default ``1/(K − 1)``, a K-fold cross-validation's). ``baseline``:
    the baseline's reported cross-validated score, of which a loss's floor is a share
    (:func:`min_gain`; default: the mean of its folds).
    """
    from turbotab.core.models.metrics import higher_is_better

    sign = 1.0 if higher_is_better(metric) else -1.0
    pairs = [(float(m), float(b)) for m, b in zip(model_folds, baseline_folds)
             if m is not None and b is not None and math.isfinite(float(m)) and math.isfinite(float(b))]
    diffs = [sign * (m - b) for m, b in pairs]
    k = len(diffs)
    if gain is None or not math.isfinite(gain):
        gain = sum(diffs) / k if k else None
    if baseline is None and pairs:
        baseline = sum(b for _, b in pairs) / len(pairs)
    floor = min_gain(metric, baseline)
    if higher_is_better(metric):
        at_least = f"at least {MIN_GAIN:g} (a convention)"
    else:
        at_least = (f"at least {_share_of(metric, None)}, {floor:.2g} here "
                    f"({SKILL.get(metric, 'a skill score')} of {MIN_GAIN:g}, a convention)")
    se = ci_low = ci_high = None
    df = None
    if k > 1 and gain is not None:
        mean = sum(diffs) / k
        var = sum((d - mean) ** 2 for d in diffs) / (k - 1)
        share = test_share if test_share is not None and test_share > 0 else 1.0 / (k - 1)
        se = math.sqrt((1.0 / k + share) * var)
        df = k - 1
        half = _t(LEVEL, df) * se
        ci_low, ci_high = gain - half, gain + half
        tolerance = max(floor, half)
        above = f"{tolerance:.3f}" if higher_is_better(metric) else f"{tolerance:.3g}"
        basis = (f"Better only when the gain's {LEVEL:.0%} interval lies above zero and the gain is "
                 f"{at_least}: a gain above {above}. The interval is corrected for the {k} folds "
                 f"sharing training rows (Nadeau & Bengio 2003).")
    else:
        tolerance = floor
        basis = (f"With {k} fold{'s' if k != 1 else ''} the gain has no interval, so the model is "
                 f"never called better than its baseline.")
    if gain is None:
        verdict: Verdict = "no_better"
    elif gain < 0:
        verdict = "worse"
    elif se is not None and gain > tolerance:
        verdict = "better"
    else:
        verdict = "no_better"
    return VersusBaseline(verdict=verdict, metric=metric, gain=gain, tolerance=tolerance,
                          tolerance_basis=basis, se=se, ci_low=ci_low, ci_high=ci_high, df=df)


def compare(task: str, model: Any, baseline: Any,
            summary: dict[str, dict[str, Any]] | None = None
            ) -> tuple[dict[str, dict[str, Any]], VersusBaseline]:
    """A family's cross-validated summary and its verdict against the baseline's, as the fit
    stage reports them. ``model`` and ``baseline`` are ``metrics.CrossValidated`` on the same
    fold pairs; ``summary`` is the model's, when the caller has made it (with its units)."""
    from turbotab.core.models.metrics import PRIMARY, higher_is_better

    primary = PRIMARY[task]
    summary = summary if summary is not None else model.summary(task)
    estimate = summary[primary]["estimate"]
    base = baseline.summary(task)[primary]["estimate"]
    sign = 1.0 if higher_is_better(primary) else -1.0
    gain = sign * (estimate - base) if estimate is not None and base is not None else None
    versus = versus_baseline(primary, [f[primary] for f in model.per_fold],
                             [b[primary] for b in baseline.per_fold], gain=gain,
                             test_share=model.test_share, baseline=base)
    return summary, versus


def _signed(x: float, places: int = 3) -> str:
    return f"{x:.{places}f}".replace("-", "−")


def no_better_concern(task: str, metric_label: str, versus: VersusBaseline,
                      model: float | None, base: float | None, fmt: Any) -> str | None:
    """A plain sentence when a family is not shown to beat its baseline (``worse`` has M1's own)."""
    if versus.verdict != "no_better" or model is None or base is None or versus.gain is None:
        return None
    m, b = fmt(model, base)
    against = {"regression": "the outcome's average", "ordinal": "the level prior",
               "time_to_event": "one risk for everyone"}.get(task, "the class prior")
    if versus.ci_low is None or versus.ci_high is None:
        return f"Not shown to beat {against}: CV {metric_label} {m} against {b}, from one fold."
    if versus.se == 0 and versus.gain == 0:  # e.g. a penalty that removed every predictor
        return f"Scores the same as {against} in every fold: CV {metric_label} {m} against {b}."
    # a loss's floor can sit below a thousandth: the gain is shown to two places past its first digit
    places = max(3, 1 - math.floor(math.log10(versus.tolerance))) if versus.tolerance > 0 else 3
    interval = f"{_signed(versus.ci_low, places)} to {_signed(versus.ci_high, places)}"
    if versus.ci_low > 0:  # so the tolerance is the floor (the interval's half-width is smaller)
        return (f"Better than {against} by too little to matter: CV {metric_label} {m} against "
                f"{b}, a gain of {_signed(versus.gain, places)} ({LEVEL:.0%} interval {interval}), "
                f"under {floor_words(versus.metric, versus.tolerance, against)}.")
    return (f"Not distinguishable from {against}: CV {metric_label} {m} against {b}; the gain, "
            f"{_signed(versus.gain, places)}, has a {LEVEL:.0%} interval of {interval}.")


__all__ = ["LEVEL", "MIN_GAIN", "SKILL", "VersusBaseline", "compare", "floor_words", "min_gain",
           "no_better_concern", "versus_baseline"]

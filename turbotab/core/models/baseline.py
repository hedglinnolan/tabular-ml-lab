"""Whether a model does better than predicting without its predictors, with an honest interval.

M1 said when a family scores *worse* than its baseline (the outcome's training-fold mean, or the
class prior); M2 added "no better". The comparison is paired, fold by fold, on the primary metric
(higher is better for R², AUC and macro-F1), and the gain is the difference of the two reported
cross-validated scores (for R², the pooled estimate, :mod:`turbotab.core.models.metrics`):

    d_k = model_k − baseline_k            gain = model CV score − baseline CV score

**The interval corrects for the folds sharing training rows.** Fold scores are not independent:
any two folds' models share most of their training rows, so the spread of ``d_k`` understates the
uncertainty of their mean (Bengio & Grandvalet, JMLR 2004;5:1089–1105: "naive estimators … grossly
underestimate variance"). The corrected resampled t of Nadeau & Bengio (Machine Learning
2003;52:239–281), as Bouckaert & Frank (PAKDD 2004) apply it to k-fold cross-validation, inflates
the variance by the share of rows scored to rows fit:

    se = √( (1/K + n₂/n₁) · s²_d )        95% interval: gain ± t₀.₉₇₅,K−1 · se

**The verdict.** ``better`` when the whole 95% interval lies above zero and the gain is at least
``MIN_GAIN`` (0.01 on the metric's own scale: a convention, not a cited rule — a gain below it is
too small to matter whatever the folds say); ``worse`` when the gain is below zero; ``no_better``
otherwise. Requiring the two-sided interval to clear zero is a one-sided test at 2.5%: the verdict
and the interval the reader sees never disagree. The rule it replaces — a gain above one fold
standard error — called a model "better" on 25–31% of binary datasets with no signal at all
(audit MA-10).
"""
from __future__ import annotations

import math
from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict

MIN_GAIN = 0.01
LEVEL = 0.95
Verdict = Literal["better", "no_better", "worse"]


class VersusBaseline(BaseModel):
    """A family's primary metric against its baseline's, paired over the same folds."""

    model_config = ConfigDict(extra="forbid", json_schema_serialization_defaults_required=True)

    verdict: Verdict
    metric: str
    gain: float | None  # the model's cross-validated score minus the baseline's
    tolerance: float  # the gain must exceed this to be "better": max(MIN_GAIN, t · se)
    tolerance_basis: str  # the rule in words, both of its parts named
    # The gain's 95% interval, corrected for the folds' shared training rows (Nadeau & Bengio).
    se: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    df: int | None = None  # folds − 1
    level: float = LEVEL


def _t(level: float, df: int) -> float:
    from scipy import stats

    return float(stats.t.ppf(1 - (1 - level) / 2, df))


def versus_baseline(metric: str, model_folds: Sequence[float | None],
                    baseline_folds: Sequence[float | None], *, gain: float | None = None,
                    test_share: float | None = None) -> VersusBaseline:
    """The paired comparison and its verdict (see the module docstring).

    ``gain``: the difference of the reported cross-validated scores (default: the mean per-fold
    difference). ``test_share``: rows scored over rows fit, per fold, on average (default
    ``1/(K − 1)``, a K-fold cross-validation's).
    """
    pairs = [(float(m), float(b)) for m, b in zip(model_folds, baseline_folds)
             if m is not None and b is not None and math.isfinite(float(m)) and math.isfinite(float(b))]
    diffs = [m - b for m, b in pairs]
    k = len(diffs)
    if gain is None or not math.isfinite(gain):
        gain = sum(diffs) / k if k else None
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
        tolerance = max(MIN_GAIN, half)
        basis = (f"Better only when the gain's {LEVEL:.0%} interval lies above zero and the gain is "
                 f"at least {MIN_GAIN:g} (a convention): a gain above {tolerance:.3f}. The interval "
                 f"is corrected for the {k} folds sharing training rows (Nadeau & Bengio 2003).")
    else:
        tolerance = MIN_GAIN
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


def compare(task: str, model: Any, baseline: Any) -> tuple[dict[str, dict[str, Any]], VersusBaseline]:
    """A family's cross-validated summary and its verdict against the baseline's, as the fit
    stage reports them. ``model`` and ``baseline`` are ``metrics.CrossValidated`` on the same
    fold pairs."""
    from turbotab.core.models.metrics import PRIMARY

    primary = PRIMARY[task]
    summary = model.summary(task)
    estimate = summary[primary]["estimate"]
    base = baseline.summary(task)[primary]["estimate"]
    gain = estimate - base if estimate is not None and base is not None else None
    versus = versus_baseline(primary, [f[primary] for f in model.per_fold],
                             [b[primary] for b in baseline.per_fold], gain=gain,
                             test_share=model.test_share)
    return summary, versus


def _signed(x: float, places: int = 3) -> str:
    return f"{x:.{places}f}".replace("-", "−")


def no_better_concern(task: str, metric_label: str, versus: VersusBaseline,
                      model: float | None, base: float | None, fmt: Any) -> str | None:
    """A plain sentence when a family is not shown to beat its baseline (``worse`` has M1's own)."""
    if versus.verdict != "no_better" or model is None or base is None or versus.gain is None:
        return None
    m, b = fmt(model, base)
    against = {"regression": "the outcome's average", "binary": "the class prior",
               "ordinal": "the level prior"}.get(task, "always guessing the most common class")
    if versus.ci_low is None or versus.ci_high is None:
        return f"Not shown to beat {against}: CV {metric_label} {m} against {b}, from one fold."
    if versus.se == 0 and versus.gain == 0:  # e.g. a penalty that removed every predictor
        return f"Scores the same as {against} in every fold: CV {metric_label} {m} against {b}."
    interval = f"{_signed(versus.ci_low)} to {_signed(versus.ci_high)}"
    if versus.ci_low > 0:
        return (f"Better than {against} by too little to matter: CV {metric_label} {m} against "
                f"{b}, a gain of {_signed(versus.gain)} ({LEVEL:.0%} interval {interval}), under "
                f"{MIN_GAIN:g}.")
    return (f"Not distinguishable from {against}: CV {metric_label} {m} against {b}; the gain, "
            f"{_signed(versus.gain)}, has a {LEVEL:.0%} interval of {interval}.")


__all__ = ["LEVEL", "MIN_GAIN", "VersusBaseline", "compare", "no_better_concern", "versus_baseline"]

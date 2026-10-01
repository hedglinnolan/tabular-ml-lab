"""Whether a model does better than predicting without its predictors, within a stated tolerance.

M1 said when a family scores *worse* than its baseline (the outcome's training-fold mean, or the
class prior). That misses the more common failure: a family that beats the baseline by less than the
folds disagree with each other has learned nothing a reader should trust. The comparison is paired,
fold by fold, on the primary metric (higher is better for R², AUC and macro-F1):

    gain_k = model_k − baseline_k        gain = mean_k gain_k        se = sd_k(gain_k) / √K

    tolerance = max(MIN_GAIN, se)

``worse`` when gain < 0, ``no_better`` when 0 ≤ gain ≤ tolerance, ``better`` otherwise. Both parts
of the tolerance are stated with the verdict. ``MIN_GAIN`` (0.01 on the metric's own scale) is a
convention, not a cited rule: a gain below it is too small to matter whatever the folds say. One
standard error across folds is lenient on purpose — fold scores share training rows, so their
spread understates the uncertainty (Bengio & Grandvalet, JMLR 2004;5:1089–1105) — and a family
inside it is still called no better.
"""
from __future__ import annotations

import math
from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict

MIN_GAIN = 0.01
Verdict = Literal["better", "no_better", "worse"]


class VersusBaseline(BaseModel):
    """A family's primary metric against its baseline's, paired over the same folds."""

    model_config = ConfigDict(extra="forbid", json_schema_serialization_defaults_required=True)

    verdict: Verdict
    metric: str
    gain: float | None  # the mean per-fold gain over the baseline
    tolerance: float  # the larger of MIN_GAIN and one standard error of the per-fold gain
    tolerance_basis: str  # the tolerance in words, both of its parts named


def versus_baseline(metric: str, model_folds: Sequence[float | None],
                    baseline_folds: Sequence[float | None]) -> VersusBaseline:
    pairs = [(float(m), float(b)) for m, b in zip(model_folds, baseline_folds)
             if m is not None and b is not None and math.isfinite(float(m)) and math.isfinite(float(b))]
    gains = [m - b for m, b in pairs]
    k = len(gains)
    gain = sum(gains) / k if k else None
    se = 0.0
    if k > 1 and gain is not None:
        se = math.sqrt(sum((g - gain) ** 2 for g in gains) / (k - 1)) / math.sqrt(k)
    tolerance = max(MIN_GAIN, se)
    basis = (f"The larger of {MIN_GAIN:g} (a convention) and one standard error of the gain across "
             f"the {k} folds ({se:.3f}).")
    if gain is None:
        verdict: Verdict = "no_better"
    elif gain < 0:
        verdict = "worse"
    elif gain <= tolerance:
        verdict = "no_better"
    else:
        verdict = "better"
    return VersusBaseline(verdict=verdict, metric=metric, gain=gain, tolerance=tolerance,
                          tolerance_basis=basis)


def no_better_concern(task: str, metric_label: str, versus: VersusBaseline,
                      model: float | None, base: float | None, fmt: Any) -> str | None:
    """A plain sentence when a family ties its baseline (``worse`` has its own, M1's)."""
    if versus.verdict != "no_better" or model is None or base is None or versus.gain is None:
        return None
    m, b = fmt(model, base)
    against = {"regression": "the outcome's average", "binary": "the class prior"}.get(
        task, "always guessing the most common class")
    return (f"No better than {against}: CV {metric_label} {m} against {b}, a gain within the "
            f"tolerance of {versus.tolerance:.3f}.")


__all__ = ["MIN_GAIN", "VersusBaseline", "no_better_concern", "versus_baseline"]

"""The substitution curve: move k kcal from a donor to a recipient and watch the prediction.

Source: ``docs/turbotab/PRODUCT_VISION.md`` §06c, "The instrument". What it is: a
multivariate forward marginal effect along ``d = e_recipient - e_donor``, aggregated
over rows as an Average Marginal Effect (Scholbeck et al. 2024, DMKD 38:2997-3042;
the R package ``fmeffects``). It is not an ALE.

For each k (kcal, k >= 0) every row gives up ``k / kcal_per_unit[donor]`` units of the
donor and gains ``k / kcal_per_unit[recipient]`` units of the recipient. Every other
column, total energy included, is left as it is. The shifted rows go through the
whole fitted pipeline on the RAW inputs, so an energy-adjustment step inside the
pipeline recomputes its outputs from the shifted values.

The marks this computes, and why each exists (§06c "The five marks"):

* Support mask. Shifting every row by the same k walks off the observed
  composition exactly like a PDP does (300 kcal put 22% of rows off-support in the
  §06c measurement). A row is off-support at k when a shifted value leaves that
  column's observed [min, max] or goes negative, and the average uses on-support
  rows only. The curve stops at the first k where fewer than ``min_support`` of
  the rows remain. That floor is a practitioner convention, not a sourced threshold.
* Bootstrap band over rows (``n_boot > 0``), with the fitted model held fixed.
* A stated k. A tree ensemble is piecewise constant, so "the slope" does not
  exist; the label is a finite difference at a named k ("+0.082 per 100 kcal at
  k = 100").

What it does not do: remedy composite variable bias. Naming the donor and the
recipient makes the estimand explicit and chosen; the total is in the model either
way (§06c, "What the curve does not remedy").
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = ["substitution_curve"]

TotalKind = Literal["fixed", "variable"]


def _signed(value: float) -> str:
    """A signed number with three significant digits and no scientific notation."""
    if value == 0:
        return "0"
    return np.format_float_positional(float(value), precision=3, unique=False, fractional=False,
                                      trim="-", sign=True)


def _plain(value: float) -> str:
    return np.format_float_positional(float(value), precision=6, unique=True, fractional=False, trim="-")


def substitution_curve(predict: Callable[[pd.DataFrame], Any], X: pd.DataFrame, *, donor: str,
                       recipient: str, kcal_per_unit: Mapping[str, float], ks: Sequence[float],
                       total_kind: TotalKind = "variable", min_support: float = 0.5,
                       n_boot: int = 0, random_state: int = 0,
                       label_k: Optional[float] = None) -> Dict[str, Any]:
    """The average change in prediction when k kcal move from ``donor`` to ``recipient``.

    Parameters
    ----------
    predict : the fitted pipeline's prediction function, called on raw-input frames.
        For a classifier pass e.g. ``lambda df: model.predict_proba(df)[:, 1]``.
    X : the rows to average over (raw inputs, as the pipeline expects them). Their
        observed range also defines the support.
    donor, recipient : the columns energy moves from and to.
    kcal_per_unit : kcal per unit of each column (9 for fat in grams, 1 for a kcal column).
    ks : kcal moved, all >= 0 (swap donor and recipient to move energy the other way).
        Returned sorted and de-duplicated.
    total_kind : "variable" (total energy can change in reality; holding it fixed is an
        assumption) or "fixed" (a conserved budget, e.g. 24-hour time use).
    min_support : the curve stops at the first k where the on-support fraction drops
        below this. A practitioner convention, not a sourced threshold.
    n_boot : bootstrap resamples of rows for a 95% percentile band (0 = none).
    label_k : the k the effect label reports; default the supported k > 0 nearest 100.

    Returns a dict with ``donor, recipient, ks, delta, on_support_fraction, stopped_at,
    ci_low, ci_high, total_kind, effect_label, note`` plus ``effect_sentence, label_k,
    n_on_support, n_rows, min_support, n_boot, units_moved``. ``delta`` and the band are
    ``None`` at and beyond ``stopped_at``.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"X must be a pandas DataFrame, not {type(X).__name__}.")
    if donor == recipient:
        raise ValueError("The donor and the recipient must be different columns.")
    for col in (donor, recipient):
        if col not in X.columns:
            raise ValueError(f"{col} is not a column of X.")
        if col not in kcal_per_unit:
            raise ValueError(f"kcal_per_unit has no factor for {col}; energy cannot be moved "
                             f"without knowing how many kcal one unit of it carries.")
        factor = float(kcal_per_unit[col])
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError(f"kcal_per_unit[{col!r}] must be a positive number, not {factor!r}.")
        if not pd.api.types.is_numeric_dtype(X[col]):
            raise TypeError(f"{col} must be numeric to move energy through it.")
    if total_kind not in ("fixed", "variable"):
        raise ValueError(f"total_kind must be 'fixed' or 'variable', not {total_kind!r}.")
    if not 0.0 <= float(min_support) <= 1.0:
        raise ValueError(f"min_support is a fraction of rows and must lie in [0, 1], not {min_support!r}.")
    if int(n_boot) < 0:
        raise ValueError("n_boot must be 0 or a positive number of bootstrap resamples.")
    k_values = np.asarray(list(ks), dtype=float)
    if k_values.size == 0 or not np.all(np.isfinite(k_values)):
        raise ValueError("ks must be a non-empty sequence of finite kcal amounts.")
    if np.any(k_values < 0):
        raise ValueError("Every k must be >= 0; to move energy the other way, swap donor and recipient.")
    k_values = np.unique(k_values)

    per_donor = 1.0 / float(kcal_per_unit[donor])
    per_recipient = 1.0 / float(kcal_per_unit[recipient])
    n_rows = int(len(X))
    d = X[donor].to_numpy(dtype=float, na_value=np.nan)
    r = X[recipient].to_numpy(dtype=float, na_value=np.nan)
    finite_d, finite_r = np.isfinite(d), np.isfinite(r)
    if not finite_d.any() or not finite_r.any():
        raise ValueError(f"{donor} or {recipient} has no observed values in X.")
    d_lo, d_hi = float(np.min(d[finite_d])), float(np.max(d[finite_d]))
    r_lo, r_hi = float(np.min(r[finite_r])), float(np.max(r[finite_r]))

    # A row that cannot be shifted (missing, or already negative) is off-support at every k.
    valid = finite_d & finite_r & (d >= 0) & (r >= 0)
    rows = np.flatnonzero(valid)
    base_frame = X.iloc[rows]
    base = _predictions(predict, base_frame)
    dv, rv = d[rows], r[rows]

    n_valid = rows.size
    diffs = np.full((k_values.size, n_valid), np.nan)
    masks = np.zeros((k_values.size, n_valid), dtype=bool)
    fractions: List[float] = []
    n_on: List[int] = []
    stopped_at: Optional[float] = None
    for i, k in enumerate(k_values):
        d_k = dv - k * per_donor
        r_k = rv + k * per_recipient
        mask = ((d_k >= d_lo) & (d_k <= d_hi) & (r_k >= r_lo) & (r_k <= r_hi)
                & (d_k >= 0) & (r_k >= 0))
        count = int(mask.sum())
        fractions.append(count / n_rows if n_rows else 0.0)
        n_on.append(count)
        if stopped_at is None and fractions[-1] < min_support:
            stopped_at = float(k)
        if stopped_at is not None or count == 0:
            continue
        shifted = base_frame.iloc[np.flatnonzero(mask)].copy()
        shifted[donor] = d_k[mask]
        shifted[recipient] = r_k[mask]
        diffs[i, mask] = _predictions(predict, shifted) - base[mask]
        masks[i] = mask

    live = np.array([stopped_at is None or k < stopped_at for k in k_values]) & masks.any(axis=1)
    delta: List[Optional[float]] = [
        float(np.mean(diffs[i, masks[i]])) if live[i] else None for i in range(k_values.size)]

    ci_low: Optional[List[Optional[float]]] = None
    ci_high: Optional[List[Optional[float]]] = None
    if n_boot > 0:
        ci_low, ci_high = _bootstrap(diffs, masks, live, int(n_boot), random_state)

    chosen = _label_k(k_values, delta, label_k)
    ks_out = [float(k) for k in k_values]
    if chosen is None:
        effect_label = None
        effect_sentence = (f"No k > 0 kept enough rows on support to report an effect of moving "
                           f"energy from {donor} to {recipient}.")
    else:
        value = delta[ks_out.index(chosen)]
        effect_label = f"{_signed(value * 100.0 / chosen)} per 100 kcal at k = {_plain(chosen)}"
        share = fractions[ks_out.index(chosen)]
        effect_sentence = (f"Moving {_plain(chosen)} kcal from {donor} to {recipient} changes the "
                           f"average prediction by {_signed(value)}, over the {share:.0%} of rows "
                           f"whose shifted intakes stay within the observed range.")

    return {
        "donor": donor,
        "recipient": recipient,
        "ks": ks_out,
        "delta": delta,
        "on_support_fraction": fractions,
        "n_on_support": n_on,
        "n_rows": n_rows,
        "stopped_at": stopped_at,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "total_kind": total_kind,
        "effect_label": effect_label,
        "effect_sentence": effect_sentence,
        "label_k": chosen,
        "min_support": float(min_support),
        "n_boot": int(n_boot),
        "units_moved": {donor: -per_donor, recipient: per_recipient},  # per kcal moved
        "note": _note(donor, recipient, total_kind, float(min_support), stopped_at, fractions,
                      ks_out, int(n_boot)),
    }


def _predictions(predict: Callable[[pd.DataFrame], Any], frame: pd.DataFrame) -> np.ndarray:
    if len(frame) == 0:
        return np.empty(0)
    values = np.asarray(predict(frame), dtype=float)
    if values.ndim > 1:
        if values.shape[1:] != (1,) * (values.ndim - 1):
            raise ValueError("predict must return one number per row (for a classifier, pass a "
                             "function that returns the probability of one class).")
        values = values.reshape(len(frame))
    if values.shape != (len(frame),):
        raise ValueError(f"predict returned {values.size} values for {len(frame)} rows.")
    return values


def _bootstrap(diffs: np.ndarray, masks: np.ndarray, live: np.ndarray, n_boot: int,
               random_state: int) -> tuple:
    """95% percentile band: resample rows, recompute the on-support average at each k."""
    n = diffs.shape[1]
    weights_by_k = masks.astype(float)
    totals = np.where(masks, diffs, 0.0)
    rng = np.random.default_rng(random_state)
    draws = np.full((n_boot, diffs.shape[0]), np.nan)
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, n, size=n), minlength=n).astype(float) if n else np.zeros(0)
        den = weights_by_k @ w
        with np.errstate(invalid="ignore", divide="ignore"):
            draws[b] = np.where(den > 0, (totals @ w) / den, np.nan)
    low: List[Optional[float]] = []
    high: List[Optional[float]] = []
    for i in range(diffs.shape[0]):
        column = draws[:, i][np.isfinite(draws[:, i])]
        if not live[i] or column.size == 0:
            low.append(None)
            high.append(None)
        else:
            low.append(float(np.percentile(column, 2.5)))
            high.append(float(np.percentile(column, 97.5)))
    return low, high


def _label_k(ks: np.ndarray, delta: List[Optional[float]], requested: Optional[float]) -> Optional[float]:
    supported = [float(k) for k, v in zip(ks, delta) if k > 0 and v is not None]
    if requested is not None:
        requested = float(requested)
        if requested not in supported:
            raise ValueError(f"label_k = {requested:g} is not a supported k > 0 on this curve.")
        return requested
    if not supported:
        return None
    return min(supported, key=lambda k: (abs(k - 100.0), k))


def _note(donor: str, recipient: str, total_kind: str, min_support: float,
          stopped_at: Optional[float], fractions: List[float], ks: List[float], n_boot: int) -> str:
    if total_kind == "variable":
        parts = ["Total energy was held fixed by assumption: intake can rise or fall, so keeping "
                 "the total constant is a modeling choice, not a property of the data."]
    else:
        parts = ["The total is a fixed budget, so moving energy from one component to another is "
                 "an exhaustive reallocation."]
    parts.append(
        f"Rows whose shifted {donor} or {recipient} would leave the observed range or go negative "
        f"are left out at that k, and the curve stops at the first k where fewer than "
        f"{min_support:.0%} of rows remain; that {min_support:.0%} floor (min_support) is a "
        f"practitioner convention, not a sourced threshold.")
    if stopped_at is not None:
        share = fractions[ks.index(stopped_at)]
        parts.append(f"The curve stopped at k = {_plain(stopped_at)} kcal, where {share:.0%} of rows "
                     f"remained on support.")
    if n_boot > 0:
        parts.append(f"The band is a 95% percentile bootstrap over rows ({n_boot} resamples) with "
                     f"the fitted model held fixed; it does not include the uncertainty of fitting "
                     f"the model.")
    return " ".join(parts)

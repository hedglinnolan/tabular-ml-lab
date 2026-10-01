"""The substitution curve: move k kcal from a donor to a recipient and watch the prediction.

Source: ``docs/turbotab/PRODUCT_VISION.md`` §06c, "The instrument". What it is: a
multivariate forward marginal effect along ``d = e_recipient - e_donor``, aggregated
over rows as an Average Marginal Effect (Scholbeck et al. 2024, DMKD 38:2997-3042;
the R package ``fmeffects``). It is not an ALE.

For each k (kcal, k >= 0) every row gives up ``k / kcal_per_unit[donor]`` units of the
donor and gains ``k / kcal_per_unit[recipient]`` units of the recipient. Every other
column, total energy included, is left as it is — except the parts of a moved total
(``nested``): a total carries its parts in proportion, so their shares of it hold, and a
moved part carries its total by the same amount. The shifted rows go through the whole
fitted pipeline on the RAW inputs, so an energy-adjustment step inside the pipeline
recomputes its outputs from the shifted values.

The marks this computes, and why each exists (§06c "The five marks"):

* Support mask. Shifting every row by the same k walks off the observed
  composition exactly like a PDP does (300 kcal put 22% of rows off-support in the
  §06c measurement). A row is off-support at k when a shifted value leaves that
  column's observed [min, max] or goes negative, and the average uses on-support
  rows only. The curve stops at the first k where fewer than ``min_support`` of
  the rows remain. That floor is a practitioner convention, not a sourced threshold.
* An uncertainty band from refitting (:func:`refit_band`): the model is refit on
  bootstrap resamples of the rows, so the band carries the uncertainty of fitting it.
  A band from resampling rows through one fitted model has zero width for a linear
  model and reads as certainty, so there is none here.
* A stated k. A tree ensemble is piecewise constant, so "the slope" does not
  exist; the label is a finite difference at a named k ("+0.082 per 100 kcal at
  k = 100").

What it does not do: remedy composite variable bias. Naming the donor and the
recipient makes the estimand explicit and chosen; the total is in the model either
way (§06c, "What the curve does not remedy").
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = ["Shift", "refit_band", "substitution_curve"]

TotalKind = Literal["fixed", "variable"]


def _signed(value: float) -> str:
    """A signed number with three significant digits and no scientific notation."""
    if value == 0:
        return "0"
    return np.format_float_positional(float(value), precision=3, unique=False, fractional=False,
                                      trim="-", sign=True)


def _plain(value: float) -> str:
    return np.format_float_positional(float(value), precision=6, unique=True, fractional=False, trim="-")


def _values(frame: pd.DataFrame, column: str) -> np.ndarray:
    return frame[column].to_numpy(dtype=float, na_value=np.nan)


def _names(columns: Sequence[str]) -> str:
    columns = list(columns)
    return columns[0] if len(columns) == 1 else ", ".join(columns[:-1]) + " and " + columns[-1]


class Shift:
    """What moving k kcal from ``donor`` to ``recipient`` does to a row, and where it stays on support.

    ``reference`` fixes the observed range of every column the move touches (the support).
    ``nested`` (child -> parent, from :mod:`turbotab.core.methods.nesting`) makes the move
    coherent when the table carries a total beside its parts:

    * a moved total carries its parts in proportion (each part's share of the total holds);
    * a moved part carries its total by the same amount (the other parts stay as they were).

    Pairing a total with its own part is refused: kcal moved between them go nowhere.
    """

    def __init__(self, reference: pd.DataFrame, *, donor: str, recipient: str,
                 kcal_per_unit: Mapping[str, float], nested: Optional[Mapping[str, str]] = None):
        if not isinstance(reference, pd.DataFrame):
            raise TypeError(f"X must be a pandas DataFrame, not {type(reference).__name__}.")
        if donor == recipient:
            raise ValueError("The donor and the recipient must be different columns.")
        for col in (donor, recipient):
            if col not in reference.columns:
                raise ValueError(f"{col} is not a column of X.")
            if col not in kcal_per_unit:
                raise ValueError(f"kcal_per_unit has no factor for {col}; energy cannot be moved "
                                 f"without knowing how many kcal one unit of it carries.")
            factor = float(kcal_per_unit[col])
            if not np.isfinite(factor) or factor <= 0:
                raise ValueError(f"kcal_per_unit[{col!r}] must be a positive number, not {factor!r}.")
            if not pd.api.types.is_numeric_dtype(reference[col]):
                raise TypeError(f"{col} must be numeric to move energy through it.")
        nested = {str(c): str(p) for c, p in (nested or {}).items()
                  if c in reference.columns and p in reference.columns}
        if nested.get(donor) == recipient or nested.get(recipient) == donor:
            child, parent = (donor, recipient) if nested.get(donor) == recipient else (recipient, donor)
            raise ValueError(f"{child} is part of {parent}, so moving energy between them moves "
                             f"nothing; choose two separate nutrients.")
        self.donor, self.recipient = donor, recipient
        self.per_donor = 1.0 / float(kcal_per_unit[donor])
        self.per_recipient = 1.0 / float(kcal_per_unit[recipient])
        self.nested = nested
        self.parts: Dict[str, List[str]] = {}
        for child, parent in nested.items():
            self.parts.setdefault(parent, []).append(child)
        moved = [donor, recipient]
        self.carried: List[str] = []  # partners that move because the donor or the recipient does
        for column in moved:
            parent = nested.get(column)
            if parent is not None and parent not in moved and parent not in self.carried:
                self.carried.append(parent)
            for child in self.parts.get(column, []):
                if child not in moved and child not in self.carried:
                    self.carried.append(child)
        self.ranges: Dict[str, tuple] = {}
        for column in moved + self.carried:
            if not pd.api.types.is_numeric_dtype(reference[column]):
                raise TypeError(f"{column} must be numeric to move energy through it.")
            v = _values(reference, column)
            v = v[np.isfinite(v)]
            if column in moved and not v.size:
                raise ValueError(f"{donor} or {recipient} has no observed values in X.")
            self.ranges[column] = (float(v.min()), float(v.max())) if v.size else (np.nan, np.nan)

    def valid(self, frame: pd.DataFrame) -> np.ndarray:
        """Rows that can be shifted at all: the donor and the recipient recorded and not negative."""
        d, r = _values(frame, self.donor), _values(frame, self.recipient)
        return np.isfinite(d) & np.isfinite(r) & (d >= 0) & (r >= 0)

    def values(self, frame: pd.DataFrame, k: float) -> Dict[str, np.ndarray]:
        """The shifted values of every column the move touches, for every row of ``frame``."""
        new: Dict[str, np.ndarray] = {
            self.donor: _values(frame, self.donor) - k * self.per_donor,
            self.recipient: _values(frame, self.recipient) + k * self.per_recipient,
        }
        for column in (self.donor, self.recipient):
            old = _values(frame, column)
            parent = self.nested.get(column)
            if parent is not None and parent in self.carried:
                base = new.get(parent, _values(frame, parent))
                new[parent] = base + (new[column] - old)
            for child in self.parts.get(column, []):
                if child in self.carried:
                    with np.errstate(invalid="ignore", divide="ignore"):
                        ratio = np.where(old > 0, new[column] / np.where(old > 0, old, 1.0), 1.0)
                    new[child] = _values(frame, child) * ratio
        return new

    def apply(self, frame: pd.DataFrame, k: float) -> tuple:
        """(the shifted frame, the on-support mask) over ``frame``'s rows."""
        new = self.values(frame, k)
        mask = np.ones(len(frame), dtype=bool)
        for column, v in new.items():
            lo, hi = self.ranges[column]
            with np.errstate(invalid="ignore"):
                inside = (v >= lo) & (v <= hi) & (v >= 0)
            if column in (self.donor, self.recipient):
                mask &= inside
            else:  # a partner the table does not record on this row stays unrecorded
                mask &= inside | ~np.isfinite(v)
        shifted = frame.copy()
        for column, v in new.items():
            shifted[column] = v
        return shifted, mask

    def note(self) -> Optional[str]:
        """What moved besides the donor and the recipient, in one plain sentence."""
        parts = []
        for column in (self.donor, self.recipient):
            kids = [c for c in self.parts.get(column, []) if c in self.carried]
            if kids:
                one = len(kids) == 1
                parts.append(f"{_names(kids)} {'is a part' if one else 'are parts'} of {column} and "
                             f"moved with it in proportion, so {'its share' if one else 'their shares'} "
                             f"of {column} held")
            parent = self.nested.get(column)
            if parent is not None and parent in self.carried:
                parts.append(f"{column} is part of {parent}, so {parent} moved with it by the same "
                             f"amount while its other parts held")
        return ("On every row, " + "; ".join(parts) + ".") if parts else None


def substitution_curve(predict: Callable[[pd.DataFrame], Any], X: pd.DataFrame, *, donor: str,
                       recipient: str, kcal_per_unit: Mapping[str, float], ks: Sequence[float],
                       total_kind: TotalKind = "variable", min_support: float = 0.5,
                       nested: Optional[Mapping[str, str]] = None,
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
    nested : child -> parent for columns that are parts of another (see :class:`Shift`).
    label_k : the k the effect label reports; default the supported k > 0 nearest 100.

    Returns a dict with ``donor, recipient, ks, delta, on_support_fraction, stopped_at,
    total_kind, effect_label, note`` plus ``live, effect_sentence, label_k, n_on_support,
    n_rows, min_support, units_moved, carried``. ``delta`` is ``None`` at and beyond
    ``stopped_at``. The band, when one is wanted, is :func:`refit_band`'s.
    """
    shift = Shift(X, donor=donor, recipient=recipient, kcal_per_unit=kcal_per_unit, nested=nested)
    if total_kind not in ("fixed", "variable"):
        raise ValueError(f"total_kind must be 'fixed' or 'variable', not {total_kind!r}.")
    if not 0.0 <= float(min_support) <= 1.0:
        raise ValueError(f"min_support is a fraction of rows and must lie in [0, 1], not {min_support!r}.")
    k_values = _ks(ks)

    n_rows = int(len(X))
    rows = np.flatnonzero(shift.valid(X))  # a row that cannot be shifted is off-support at every k
    base_frame = X.iloc[rows]
    base = _predictions(predict, base_frame)

    n_valid = rows.size
    diffs = np.full((k_values.size, n_valid), np.nan)
    masks = np.zeros((k_values.size, n_valid), dtype=bool)
    fractions: List[float] = []
    n_on: List[int] = []
    stopped_at: Optional[float] = None
    for i, k in enumerate(k_values):
        shifted, mask = shift.apply(base_frame, float(k))
        count = int(mask.sum())
        fractions.append(count / n_rows if n_rows else 0.0)
        n_on.append(count)
        if stopped_at is None and fractions[-1] < min_support:
            stopped_at = float(k)
        if stopped_at is not None or count == 0:
            continue
        diffs[i, mask] = _predictions(predict, shifted.iloc[np.flatnonzero(mask)]) - base[mask]
        masks[i] = mask

    live = np.array([stopped_at is None or k < stopped_at for k in k_values]) & masks.any(axis=1)
    delta: List[Optional[float]] = [
        float(np.mean(diffs[i, masks[i]])) if live[i] else None for i in range(k_values.size)]

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
        "live": [bool(v) for v in live],
        "on_support_fraction": fractions,
        "n_on_support": n_on,
        "n_rows": n_rows,
        "stopped_at": stopped_at,
        "total_kind": total_kind,
        "effect_label": effect_label,
        "effect_sentence": effect_sentence,
        "label_k": chosen,
        "min_support": float(min_support),
        "units_moved": {donor: -shift.per_donor, recipient: shift.per_recipient},  # per kcal moved
        "carried": list(shift.carried),
        "note": _note(donor, recipient, total_kind, float(min_support), stopped_at, fractions,
                      ks_out, shift.note()),
    }


def _ks(ks: Sequence[float]) -> np.ndarray:
    k_values = np.asarray(list(ks), dtype=float)
    if k_values.size == 0 or not np.all(np.isfinite(k_values)):
        raise ValueError("ks must be a non-empty sequence of finite kcal amounts.")
    if np.any(k_values < 0):
        raise ValueError("Every k must be >= 0; to move energy the other way, swap donor and recipient.")
    return np.unique(k_values)


def refit_band(fit: Callable[[pd.DataFrame, np.ndarray], Callable[[pd.DataFrame], Any]],
               X: pd.DataFrame, y: Any, *, shift: Shift, ks: Sequence[float], live: Sequence[bool],
               n_boot: int, groups: Any = None, random_state: int = 0, level: float = 0.95,
               center: Optional[Sequence[Optional[float]]] = None,
               progress: Optional[Callable[[int, int], None]] = None) -> Dict[str, Any]:
    """A percentile band for the curve from ``n_boot`` refits on bootstrap resamples of ``X``.

    Each resample draws rows of ``X`` with replacement (whole groups when ``groups`` is given, so
    a participant's rows stay together), refits the model on it — ``fit(X_b, y_b)`` returns the
    refit's prediction function — and recomputes the average change at each live k over the
    resample's own on-support rows. The band is the ``level`` percentile interval of those
    curves at each k. Because the model is refit, the band carries the uncertainty of fitting
    it, which a band through one fitted model never can.

    ``center`` (the point curve's deltas) draws the band around the curve: the interval's
    distances below and above the refits' median, placed at the curve. When ``X`` is a
    subsample of the rows the curve was fit on, the refits' own center differs from the curve's
    by that subsample's luck; their spread is what the band reports.

    ``live`` is the point curve's own (False at and beyond where it stopped); the band is None
    there. ``progress(done, n_boot)`` is called after each refit and may raise to stop early (a
    cancelled job does). A refit that fails (say, a resample holding one class) is skipped and
    counted. Returns ``{ci_low, ci_high, n_boot, n_ok, failed, seconds}``.
    """
    if int(n_boot) <= 0:
        raise ValueError("n_boot must be a positive number of refits.")
    k_values = _ks(ks)
    live = np.asarray(list(live), dtype=bool)
    if live.size != k_values.size:
        raise ValueError("live must say, for every k, whether the point curve reached it.")
    y = np.asarray(y)
    if len(y) != len(X):
        raise ValueError("X and y must have the same rows.")
    started = time.perf_counter()
    rows = np.flatnonzero(shift.valid(X))
    frame = X.iloc[rows]
    shifted = {i: shift.apply(frame, float(k)) for i, k in enumerate(k_values) if live[i]}
    rng = np.random.default_rng(random_state)
    members: Optional[List[np.ndarray]] = None
    if groups is not None:
        codes, _ = pd.factorize(pd.Series(np.asarray(groups, dtype=object)))
        members = [np.flatnonzero(codes == u) for u in range(codes.max() + 1)]
    position = np.full(len(X), -1)
    position[rows] = np.arange(rows.size)
    draws = np.full((int(n_boot), k_values.size), np.nan)
    failed = 0
    for b in range(int(n_boot)):
        if members is not None:
            picked = rng.integers(0, len(members), size=len(members))
            idx = np.concatenate([members[j] for j in picked])
        else:
            idx = rng.integers(0, len(X), size=len(X))
        try:
            predict = fit(X.iloc[idx], y[idx])
            base = _predictions(predict, frame)
            at = position[idx]
            weight = np.bincount(at[at >= 0], minlength=rows.size).astype(float)
            for i, (moved, mask) in shifted.items():
                on = mask & (weight > 0)
                if on.any():
                    diff = _predictions(predict, moved.iloc[np.flatnonzero(on)]) - base[on]
                    draws[b, i] = float(np.average(diff, weights=weight[on]))
        except (ValueError, IndexError):
            failed += 1
            draws[b] = np.nan
        if progress is not None:
            progress(b + 1, int(n_boot))
    tail = (1.0 - float(level)) / 2.0 * 100.0
    low: List[Optional[float]] = []
    high: List[Optional[float]] = []
    centers = list(center) if center is not None else None
    for i in range(k_values.size):
        column = draws[:, i][np.isfinite(draws[:, i])]
        if not live[i] or column.size < 2:
            low.append(None)
            high.append(None)
            continue
        lo, hi = float(np.percentile(column, tail)), float(np.percentile(column, 100.0 - tail))
        if centers is not None and centers[i] is not None:
            mid = float(np.median(column))
            lo, hi = float(centers[i]) - (mid - lo), float(centers[i]) + (hi - mid)
        low.append(lo)
        high.append(hi)
    return {"ci_low": low, "ci_high": high, "n_boot": int(n_boot), "n_ok": int(n_boot) - failed,
            "failed": failed, "seconds": round(time.perf_counter() - started, 3)}


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
          stopped_at: Optional[float], fractions: List[float], ks: List[float],
          carried: Optional[str]) -> str:
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
    if carried:
        parts.append(carried)
    return " ".join(parts)

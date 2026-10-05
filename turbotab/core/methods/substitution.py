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
  §06c measurement). A row is off-support at k when either of two checks fails:

  - the amount check: a shifted column leaves that column's observed [min, max] or goes
    negative;
  - the composition check (when the total-energy column is named): a shifted column's
    share of total energy leaves the observed range of that share. Moving energy at a
    fixed total changes the diet's composition, not only its amounts: a row can keep
    every amount inside its range while its fat share falls below any share observed
    (audit MA-13; Scholbeck et al., arXiv:2201.08837, App. A.2, on extrapolation as
    "predictions in areas of the feature space with a low density of training points").

  The rows each check removes are counted at every k. The average uses on-support rows
  only, and the curve stops at the first k where fewer than ``min_support`` of the rows
  remain. That floor is a practitioner convention, not a sourced threshold.
* A fixed-population curve. Because each k averages over its own on-support rows, the
  rows behind the curve change with k and the curve mixes the effect of k with which rows
  remain (audit B17). The fixed-population curve averages over the same rows at every k:
  those on support at every k the curve reached.
* An uncertainty band from refitting (:func:`refit_band`): the model is refit on
  bootstrap resamples of the rows it was fit on, so the band carries the uncertainty of
  fitting it. A band from resampling rows through one fitted model has zero width for a
  linear model and reads as certainty, so there is none here.
* A stated k. A tree ensemble is piecewise constant, so "the slope" does not
  exist; the label is a finite difference at a named k ("+0.082 per 100 kcal at
  k = 100").

What it does not do: remedy composite variable bias. Naming the donor and the
recipient makes the estimand explicit and chosen; the total is in the model either
way (§06c, "What the curve does not remedy").
"""
from __future__ import annotations

import math
import time
from statistics import NormalDist
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = ["Shift", "refit_band", "substitution_curve", "PERCENTILE_MIN_REFITS",
           "MIN_REFIT_SHARE"]

TotalKind = Literal["fixed", "variable"]
Scale = Literal["kcal", "percent_energy"]
# The amount an effect label is stated per: 100 kcal, or 5% of energy (the field's figure,
# NUTRITION_PACK §05: "5% of energy from X replaced by Y").
PER_UNIT: Dict[str, float] = {"kcal": 100.0, "percent_energy": 5.0}
Interval = Literal["auto", "normal", "percentile"]

# A percentile interval reads its endpoints off the refits' own tails, so it needs many refits:
# "For 90-95 per cent confidence intervals, most practitioners ... suggest that B should be
# between 1000 and 2000" (Carpenter & Bithell 2000, Stat Med 19:1141). With fewer, the band is the
# normal interval from the refits' standard error, which settles with far fewer: "Very seldom are
# more than B = 200 replications needed for estimating a standard error" (Efron & Tibshirani
# 1993, An Introduction to the Bootstrap, §6.4).
PERCENTILE_MIN_REFITS = 1000
# A band is drawn only when at least this share of its refits succeeded. Leaving failed refits out
# conditions the band on the resamples a model could be fit on; when most succeed that barely
# moves it, when many fail it describes some other population of resamples. A practitioner
# convention, not a sourced threshold.
MIN_REFIT_SHARE = 0.9


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

    ``total`` names the total-energy column. With it, the support also checks composition: each
    moved column's share of total energy (its kcal over the total) must stay inside the range of
    that share in ``reference``. The total may be in kcal or in any unit proportional to it (kJ):
    a share's range and its shifted value scale together, so the check is the same. A row whose
    total is not recorded (or is not positive) cannot have its composition checked and is never
    on support.

    Pairing a total with its own part is refused: kcal moved between them go nowhere.
    """

    def __init__(self, reference: pd.DataFrame, *, donor: str, recipient: str,
                 kcal_per_unit: Mapping[str, float], nested: Optional[Mapping[str, str]] = None,
                 total: Optional[str] = None):
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
        # kcal per unit of every column the move touches: a partner carries the nutrient it moves with
        self.factors: Dict[str, float] = {c: float(kcal_per_unit[c]) for c in moved}
        for column in moved:
            parent = nested.get(column)
            if parent is not None and parent not in moved and parent not in self.carried:
                self.carried.append(parent)
                self.factors[parent] = self.factors[column]
            for child in self.parts.get(column, []):
                if child not in moved and child not in self.carried:
                    self.carried.append(child)
                    self.factors[child] = self.factors[column]
        self.ranges: Dict[str, tuple] = {}
        for column in moved + self.carried:
            if not pd.api.types.is_numeric_dtype(reference[column]):
                raise TypeError(f"{column} must be numeric to move energy through it.")
            v = _values(reference, column)
            v = v[np.isfinite(v)]
            if column in moved and not v.size:
                raise ValueError(f"{donor} or {recipient} has no observed values in X.")
            self.ranges[column] = (float(v.min()), float(v.max())) if v.size else (np.nan, np.nan)
        self.total: Optional[str] = None
        self.share_ranges: Dict[str, tuple] = {}
        if total is not None:
            if total not in reference.columns:
                raise ValueError(f"{total} is not a column of X, so shares of energy cannot be checked.")
            if total in moved:
                raise ValueError(f"{total} is the total energy; it cannot also be the donor or the "
                                 f"recipient.")
            if not pd.api.types.is_numeric_dtype(reference[total]):
                raise TypeError(f"{total} must be numeric to take shares of it.")
            energy = _values(reference, total)
            usable = np.isfinite(energy) & (energy > 0)
            if not usable.any():
                raise ValueError(f"{total} has no positive recorded values, so no share of energy "
                                 f"can be taken.")
            self.total = total
            for column in moved + self.carried:
                if column == total:
                    continue
                share = self._share(column, _values(reference, column), energy)
                share = share[usable & np.isfinite(share)]
                self.share_ranges[column] = ((float(share.min()), float(share.max())) if share.size
                                             else (np.nan, np.nan))

    def _share(self, column: str, values: np.ndarray, energy: np.ndarray) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return values * self.factors[column] / energy

    def valid(self, frame: pd.DataFrame) -> np.ndarray:
        """Rows that can be shifted at all: the donor and the recipient recorded and not negative,
        and, when shares are checked, a positive total energy recorded."""
        d, r = _values(frame, self.donor), _values(frame, self.recipient)
        ok = np.isfinite(d) & np.isfinite(r) & (d >= 0) & (r >= 0)
        if self.total is not None:
            energy = _values(frame, self.total)
            ok &= np.isfinite(energy) & (energy > 0)
        return ok

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

    def checks(self, frame: pd.DataFrame, k: float) -> tuple:
        """(the shifted frame, the amount check, the composition check) over ``frame``'s rows.

        Each check is a boolean array, True where the row passes. The composition check is all
        True when no total-energy column was named.
        """
        new = self.values(frame, k)
        amount = np.ones(len(frame), dtype=bool)
        for column, v in new.items():
            lo, hi = self.ranges[column]
            with np.errstate(invalid="ignore"):
                inside = (v >= lo) & (v <= hi) & (v >= 0)
            if column in (self.donor, self.recipient):
                amount &= inside
            else:  # a partner the table does not record on this row stays unrecorded
                amount &= inside | ~np.isfinite(v)
        composition = np.ones(len(frame), dtype=bool)
        if self.total is not None:
            energy = new.get(self.total, _values(frame, self.total))
            for column, v in new.items():
                if column == self.total:
                    continue
                lo, hi = self.share_ranges[column]
                share = self._share(column, v, energy)
                with np.errstate(invalid="ignore"):
                    inside = (share >= lo) & (share <= hi)
                if column in (self.donor, self.recipient):
                    composition &= inside
                else:
                    composition &= inside | ~np.isfinite(v)
        shifted = frame.copy()
        for column, v in new.items():
            shifted[column] = v
        return shifted, amount, composition

    def apply(self, frame: pd.DataFrame, k: float) -> tuple:
        """(the shifted frame, the on-support mask) over ``frame``'s rows: both checks passed."""
        shifted, amount, composition = self.checks(frame, k)
        return shifted, amount & composition

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
                       label_k: Optional[float] = None, total: Optional[str] = None,
                       scale: Scale = "kcal", shift: Optional[Shift] = None,
                       weights: Optional[Sequence[float]] = None) -> Dict[str, Any]:
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
    total : the total-energy column of ``X``; with it the support also checks each moved
        column's share of energy (see :class:`Shift`). Without it, composition is not checked
        and the note says so.
    scale : ``"kcal"`` (k is kcal on every row) or ``"percent_energy"`` (k is percentage points
        of each row's own total energy; :mod:`turbotab.core.methods.percent_energy`), which
        sets the effect label's unit ("per 5% of energy") and the k it reports (nearest 5).
    shift : a prepared :class:`Shift` (a ``PercentEnergyShift`` for the percent scale) used in
        place of one built from the arguments above.
    weights : one per row of ``X``: each k's average is weighted by them (a survey weight, the
        people a row stands for: the curve over a surveyed population, MS4). The support's
        fractions and the stopping rule still count rows.

    Returns a dict with ``donor, recipient, ks, delta, on_support_fraction, stopped_at,
    total_kind, effect_label, note`` plus ``live, effect_sentence, label_k, n_on_support,
    n_rows, min_support, units_moved, carried`` and the support's accounting: ``total``,
    ``n_not_recorded`` (rows never on support: donor, recipient or total not recorded),
    ``n_off_amount`` and ``n_off_share`` (per k: rows the amount check removed, then the further
    rows the composition check removed) and ``fixed_population`` (``n_rows``, ``through`` and
    ``delta``: the curve over the rows on support at every k the curve reached). ``delta`` is
    ``None`` at and beyond ``stopped_at``. The band, when one is wanted, is :func:`refit_band`'s.
    """
    if scale not in ("kcal", "percent_energy"):
        raise ValueError(f"scale must be 'kcal' or 'percent_energy', not {scale!r}.")
    if shift is None:
        shift = Shift(X, donor=donor, recipient=recipient, kcal_per_unit=kcal_per_unit,
                      nested=nested, total=total)
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
    off_amount: List[int] = []
    off_share: List[int] = []
    stopped_at: Optional[float] = None
    for i, k in enumerate(k_values):
        shifted, amount, composition = shift.checks(base_frame, float(k))
        mask = amount & composition
        off_amount.append(int((~amount).sum()))
        off_share.append(int((amount & ~composition).sum()))
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
    # Under a surveyed population (MS4) each row counts as the people its survey weight stands for.
    w = None if weights is None else np.asarray(weights, dtype=float)[rows]

    def mean(values: np.ndarray, where: np.ndarray) -> float:
        return float(np.mean(values[where])) if w is None else float(np.average(values[where],
                                                                                weights=w[where]))

    delta: List[Optional[float]] = [
        mean(diffs[i], masks[i]) if live[i] else None for i in range(k_values.size)]
    fixed = _fixed_population(masks, live)
    fixed_delta: List[Optional[float]] = [
        mean(diffs[i], fixed) if live[i] and fixed.any() else None
        for i in range(k_values.size)]
    through = float(k_values[np.flatnonzero(live)[-1]]) if live.any() else None

    per = PER_UNIT[scale]
    chosen = _label_k(k_values, delta, label_k, per)
    ks_out = [float(k) for k in k_values]
    within = ("whose shifted intakes and shares of energy stay within the observed range"
              if shift.total is not None else "whose shifted intakes stay within the observed range")
    if chosen is None:
        effect_label = None
        effect_sentence = (f"No k > 0 kept enough rows on support to report an effect of moving "
                           f"energy from {donor} to {recipient}.")
    else:
        value = delta[ks_out.index(chosen)]
        effect_label = (f"{_signed(value * per / chosen)} per {_amount(per, scale)} at "
                        f"k = {_plain(chosen)}")
        share = fractions[ks_out.index(chosen)]
        effect_sentence = (f"Moving {_amount(chosen, scale)} from {donor} to {recipient} changes "
                           f"the average prediction by {_signed(value)}, over the {share:.0%} of "
                           f"rows {within}.")

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
        "total": shift.total,
        "n_not_recorded": int(n_rows - n_valid),
        "n_off_amount": off_amount,
        "n_off_share": off_share,
        "fixed_population": {"n_rows": int(fixed.sum()), "through": through, "delta": fixed_delta},
        "note": _note(donor, recipient, total_kind, float(min_support), stopped_at, fractions,
                      ks_out, shift.note(), shift.total, int(fixed.sum()), through, scale),
        "scale": scale,
    }


def _fixed_population(masks: np.ndarray, live: np.ndarray) -> np.ndarray:
    """Rows on support at every live k (the rows a fixed-population curve averages over)."""
    if not live.any():
        return np.zeros(masks.shape[1], dtype=bool)
    return masks[np.flatnonzero(live)].all(axis=0)


def _ks(ks: Sequence[float]) -> np.ndarray:
    k_values = np.asarray(list(ks), dtype=float)
    if k_values.size == 0 or not np.all(np.isfinite(k_values)):
        raise ValueError("ks must be a non-empty sequence of finite kcal amounts.")
    if np.any(k_values < 0):
        raise ValueError("Every k must be >= 0; to move energy the other way, swap donor and recipient.")
    return np.unique(k_values)


def _units(groups: Any) -> List[np.ndarray]:
    """Row positions of each resampling unit (a whole group).

    A row whose group is not recorded is a unit of its own (it cannot be matched to anyone),
    as the split and the inner cross-validation treat it.
    """
    codes, _ = pd.factorize(pd.Series(np.asarray(groups, dtype=object)))
    codes = codes.copy()
    missing = np.flatnonzero(codes < 0)
    codes[missing] = codes.max(initial=-1) + 1 + np.arange(missing.size)
    order = np.argsort(codes, kind="stable")
    bounds = np.flatnonzero(np.diff(codes[order])) + 1
    return np.split(order, bounds)


def refit_band(fit: Callable[[pd.DataFrame, np.ndarray], Callable[[pd.DataFrame], Any]],
               X: pd.DataFrame, y: Any, *, shift: Shift, ks: Sequence[float], live: Sequence[bool],
               n_boot: int, groups: Any = None, random_state: int = 0, level: float = 0.95,
               center: Optional[Sequence[Optional[float]]] = None,
               progress: Optional[Callable[[int, int], None]] = None,
               curve_rows: Optional[Sequence[int]] = None, resample_size: Optional[int] = None,
               max_rows: Optional[int] = None, interval: Interval = "auto",
               min_ok_share: float = MIN_REFIT_SHARE,
               fixed_center: Optional[Sequence[Optional[float]]] = None) -> Dict[str, Any]:
    """A ``level`` band for the curve from ``n_boot`` refits on bootstrap resamples of ``X``.

    ``X``/``y`` are the rows the model was fit on. Each resample draws ``resample_size`` units
    with replacement: rows, or whole groups when ``groups`` is given, so a participant's rows stay
    together. The default draws as many units as there are (the ordinary bootstrap); ``max_rows``
    instead bounds a refit's rows: when ``X`` has more, each resample draws the share of units
    that holds about that many rows. ``fit(X_b,
    y_b)`` refits the model on the resample and returns its prediction function, and the change
    at each live k is averaged over the curve's rows (``curve_rows``, positions in ``X``; default
    every row) as often as each was drawn and only where it is on support at that k. Because the
    model is refit, the band carries the uncertainty of fitting it, which a band through one
    fitted model never can.

    **Fewer units than there are (an m-out-of-n bootstrap).** A refit on m of n units spreads
    more than the fit on all n. For an estimate whose error shrinks with the square root of the
    sample, the spread of m-unit refits times sqrt(m/n) estimates the spread at n (Bickel, Götze
    & van Zwet 1997, Statistica Sinica 7:1-31). The band rescales by that factor and returns it
    as ``scale``. When the curve averages over a subset of the rows, the rescaling also holds for
    the averaging: a curve row is drawn m/n times on average, so its weight's variance shrinks by
    the same factor the rescaling restores.

    **The interval.** ``"normal"`` is the curve plus or minus z standard errors of the refits;
    ``"percentile"`` reads the refits' own tails, placed around the curve (their distances below
    and above the refits' median). ``"auto"`` takes the percentile interval from
    :data:`PERCENTILE_MIN_REFITS` successful refits and the normal one below that; see the
    sources at that constant.

    ``center`` (the point curve's deltas) places the band around the curve; it is required when
    the band is rescaled, because then the refits' own center is that of a smaller sample.
    ``fixed_center`` does the same for the fixed-population curve (the rows on support at every
    live k); its band is returned as ``fixed_ci_low``/``fixed_ci_high``.

    A refit that fails (say, a resample holding one class) is skipped and counted. The band is
    drawn only when at least ``min_ok_share`` of the ``n_boot`` refits succeeded at that k; else
    it is None there and ``refused`` says why. ``live`` is the point curve's own (False at and
    beyond where it stopped); the band is None there. ``progress(done, n_boot)`` is called after
    each refit and may raise to stop early (a cancelled job does).

    Returns ``{ci_low, ci_high, fixed_ci_low, fixed_ci_high, n_boot, n_ok, failed, seconds,
    interval, level, n_units, resample_size, scale, min_ok_share, refused}``.
    """
    n_boot = int(n_boot)
    if n_boot <= 0:
        raise ValueError("n_boot must be a positive number of refits.")
    if interval not in ("auto", "normal", "percentile"):
        raise ValueError(f"interval must be 'auto', 'normal' or 'percentile', not {interval!r}.")
    if not 0.0 < float(level) < 1.0:
        raise ValueError(f"level must lie strictly between 0 and 1, not {level!r}.")
    if not 0.0 <= float(min_ok_share) <= 1.0:
        raise ValueError(f"min_ok_share must lie in [0, 1], not {min_ok_share!r}.")
    k_values = _ks(ks)
    live = np.asarray(list(live), dtype=bool)
    if live.size != k_values.size:
        raise ValueError("live must say, for every k, whether the point curve reached it.")
    y = np.asarray(y)
    if len(y) != len(X):
        raise ValueError("X and y must have the same rows.")
    started = time.perf_counter()
    n = len(X)
    curve = np.arange(n) if curve_rows is None else np.asarray(curve_rows, dtype=np.int64)
    if curve.size and (curve.min() < 0 or curve.max() >= n or np.unique(curve).size != curve.size):
        raise ValueError("curve_rows must be distinct positions of rows of X.")
    rows = curve[shift.valid(X.iloc[curve])]
    frame = X.iloc[rows]
    # Each live k's on-support rows and their shifted values, fixed across refits.
    prepared: Dict[int, tuple] = {}
    masks = np.zeros((k_values.size, rows.size), dtype=bool)
    for i, k in enumerate(k_values):
        if live[i]:
            moved, mask = shift.apply(frame, float(k))
            masks[i] = mask
            on = np.flatnonzero(mask)
            prepared[i] = (on, moved.iloc[on])
    fixed = _fixed_population(masks, live)

    units = _units(groups) if groups is not None else None
    if units is not None and len(np.asarray(groups)) != n:
        raise ValueError("groups must name a unit for every row of X.")
    n_units = len(units) if units is not None else n
    if resample_size is not None and max_rows is not None:
        raise ValueError("Give resample_size or max_rows, not both.")
    if max_rows is not None:
        if int(max_rows) < 1:
            raise ValueError(f"max_rows must be a positive number of rows, not {max_rows!r}.")
        share = math.ceil(n_units * int(max_rows) / n)
        m = n_units if n <= int(max_rows) else max(2, min(n_units, share))
    else:
        m = n_units if resample_size is None else int(resample_size)
    if not 1 <= m <= n_units:
        raise ValueError(f"resample_size must be between 1 and the {n_units} units, "
                         f"not {resample_size!r}.")
    scale = math.sqrt(m / n_units)
    if m < n_units and center is None:
        raise ValueError("A band from resamples smaller than the sample needs the curve's own "
                         "deltas as its center.")

    rng = np.random.default_rng(random_state)
    position = np.full(n, -1)
    position[rows] = np.arange(rows.size)
    draws = np.full((n_boot, k_values.size), np.nan)
    fixed_draws = np.full((n_boot, k_values.size), np.nan)
    failed = 0
    for b in range(n_boot):
        picked = rng.integers(0, n_units, size=m)
        idx = picked if units is None else np.concatenate([units[j] for j in picked])
        try:
            predict = fit(X.iloc[idx], y[idx])
            base = _predictions(predict, frame)
            at = position[idx]
            weight = np.bincount(at[at >= 0], minlength=rows.size).astype(float)
            for i, (on, moved_on) in prepared.items():
                w = weight[on]
                if w.sum() <= 0:
                    continue
                diff = _predictions(predict, moved_on) - base[on]
                draws[b, i] = float(np.average(diff, weights=w))
                wf = w * fixed[on]
                if wf.sum() > 0:
                    fixed_draws[b, i] = float(np.average(diff, weights=wf))
        except (ValueError, IndexError):
            failed += 1
            draws[b] = np.nan
            fixed_draws[b] = np.nan
        if progress is not None:
            progress(b + 1, n_boot)

    n_ok = n_boot - failed
    needed = max(2, math.ceil(float(min_ok_share) * n_boot))
    method = interval
    if method == "auto":
        method = "percentile" if n_ok >= PERCENTILE_MIN_REFITS else "normal"
    refused = None
    if n_ok < needed:
        refused = (f"{n_ok} of {n_boot} refits succeeded; a band needs at least "
                   f"{float(min_ok_share):.0%} of them")
    low, high = _interval(draws, live, center, needed, method, float(level), scale)
    if m < n_units and fixed_center is None:  # a rescaled band has no center of its own
        fixed_low = fixed_high = [None] * k_values.size
    else:
        fixed_low, fixed_high = _interval(fixed_draws, live, fixed_center, needed, method,
                                          float(level), scale)
    return {"ci_low": low, "ci_high": high, "fixed_ci_low": fixed_low, "fixed_ci_high": fixed_high,
            "n_boot": n_boot, "n_ok": n_ok, "failed": failed,
            "seconds": round(time.perf_counter() - started, 3), "interval": method,
            "level": float(level), "n_units": n_units, "resample_size": m, "scale": scale,
            "min_ok_share": float(min_ok_share), "refused": refused}


def _interval(draws: np.ndarray, live: np.ndarray, center: Optional[Sequence[Optional[float]]],
              needed: int, method: str, level: float, scale: float) -> tuple:
    """Per k: the band's (low, high) lists from the refits' draws, None where it is not drawn."""
    tail = (1.0 - level) / 2.0
    z = NormalDist().inv_cdf(1.0 - tail)
    centers = list(center) if center is not None else None
    low: List[Optional[float]] = []
    high: List[Optional[float]] = []
    for i in range(draws.shape[1]):
        column = draws[:, i][np.isfinite(draws[:, i])]
        mid = centers[i] if centers is not None else None
        if not live[i] or column.size < needed or (centers is not None and mid is None):
            low.append(None)
            high.append(None)
            continue
        if method == "normal":
            at = float(mid) if mid is not None else float(np.mean(column))
            half = z * scale * float(np.std(column, ddof=1))
            low.append(at - half)
            high.append(at + half)
            continue
        lo, hi = (float(v) for v in np.percentile(column, [100.0 * tail, 100.0 * (1.0 - tail)]))
        median = float(np.median(column))
        at = float(mid) if mid is not None else median
        low.append(at - scale * (median - lo))
        high.append(at + scale * (hi - median))
    return low, high


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


def _label_k(ks: np.ndarray, delta: List[Optional[float]], requested: Optional[float],
             per: float = 100.0) -> Optional[float]:
    supported = [float(k) for k, v in zip(ks, delta) if k > 0 and v is not None]
    if requested is not None:
        requested = float(requested)
        if requested not in supported:
            raise ValueError(f"label_k = {requested:g} is not a supported k > 0 on this curve.")
        return requested
    if not supported:
        return None
    return min(supported, key=lambda k: (abs(k - per), k))


def _amount(k: float, scale: str) -> str:
    """``100 kcal`` or ``5% of energy``."""
    return f"{_plain(k)}% of energy" if scale == "percent_energy" else f"{_plain(k)} kcal"


def _note(donor: str, recipient: str, total_kind: str, min_support: float,
          stopped_at: Optional[float], fractions: List[float], ks: List[float],
          carried: Optional[str], total: Optional[str] = None, fixed_rows: int = 0,
          through: Optional[float] = None, scale: str = "kcal") -> str:
    if total_kind == "variable":
        parts = ["Total energy was held fixed by assumption: intake can rise or fall, so keeping "
                 "the total constant is a modeling choice, not a property of the data."]
    else:
        parts = ["The total is a fixed budget, so moving energy from one component to another is "
                 "an exhaustive reallocation."]
    if total is not None:
        support = (f"Rows whose shifted {donor} or {recipient} would leave the observed range or go "
                   f"negative, or whose share of energy from either (of {total}) would leave the "
                   f"range of that share observed, are left out at that k")
    else:
        support = (f"Rows whose shifted {donor} or {recipient} would leave the observed range or go "
                   f"negative are left out at that k; no total-energy column was named, so the "
                   f"diet's composition (each nutrient's share of energy) was not checked")
    parts.append(
        f"{support}, and the curve stops at the first k where fewer than {min_support:.0%} of "
        f"rows remain; that {min_support:.0%} floor (min_support) is a practitioner convention, "
        f"not a sourced threshold.")
    if stopped_at is not None:
        share = fractions[ks.index(stopped_at)]
        parts.append(f"The curve stopped at k = {_amount(stopped_at, scale)}, where {share:.0%} of "
                     f"rows remained on support.")
    if through is not None and through > 0:
        parts.append(f"Each k averages over its own on-support rows, so the rows behind the curve "
                     f"change with k; the fixed-population curve averages over the same {fixed_rows:,} "
                     f"rows, those on support at every k up to {_amount(through, scale)}.")
    if carried:
        parts.append(carried)
    return " ".join(parts)

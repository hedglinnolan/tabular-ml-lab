"""The substitution curve: move k kcal from a donor to a recipient and watch the prediction.

Source: ``docs/turbotab-next/reference/PRODUCT_VISION.md`` §06c, "The instrument". What it is: a
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

**A multiclass outcome** (V2 definition of done §2, "Dietary, extended": multiclass substitution
curves, one per class). A model of an outcome in K unordered classes predicts K probabilities that
sum to one on every row. :func:`class_curves` follows all K: at each k, class c's curve is the
average change in its predicted probability, over the same on-support rows (and, under a surveyed
population, the same weights) for every class. Because each row's probabilities sum to one before
and after the move, the K changes sum to zero on every row, so the K curves sum to zero at every k:
energy moved between nutrients moves probability between classes and creates none. That identity
is checked on every curve the app draws (:func:`check_sums_to_zero`): a set of curves that does
not sum to zero is a defect, raised, never drawn. It is the average marginal effect of a multinomial
model on the probability scale (the "average discrete change" of each outcome's probability, whose
values sum to zero across the outcomes: Long & Freese 2014, *Regression Models for Categorical
Dependent Variables Using Stata*, 3rd ed., ch. 8, nominal outcomes; Stata's ``margins`` and R's
``marginaleffects`` compute it per outcome level), here for the isocaloric move of k kcal. Its
band, by refits (:func:`class_refit_band`), is drawn class by class; the bands need not sum to
anything. Under multiple imputation each copy's class curves are pooled at each k by Rubin's rules
(:func:`pool_class_curves`; MODELING_SEQUENCE §2: "pooling of every estimate shown under inference,
including substitution curves (per copy, pooled per k)"). Under a surveyed population the curves
come from the survey-weighted multinomial fit, averaged with the weights, and their band is the
design's linearization (:func:`design_class_curves`; Graubard & Korn 1999).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = ["Shift", "refit_band", "substitution_curve", "PERCENTILE_MIN_REFITS",
           "MIN_REFIT_SHARE", "CLASS_SUM_TOLERANCE", "check_sums_to_zero", "class_curves",
           "class_estimand", "class_refit_band", "design_class_curves", "level_name",
           "pool_class_curves", "class_clause", "CLASS_CONTRACT"]

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

    walk = _walk(lambda frame: _predictions(predict, frame)[:, None], X, shift, k_values,
                 float(min_support), 1)
    n_rows, rows, n_valid = walk.n_rows, walk.rows, walk.n_valid
    masks, live, stopped_at = walk.masks, walk.live, walk.stopped_at
    fractions, n_on = walk.fractions, walk.n_on
    off_amount, off_share = walk.off_amount, walk.off_share
    delta, fixed_delta, fixed = _averages(walk, 0, weights)
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


@dataclass
class _Walk:
    """Every row's change at every k, for each of a prediction's ``n_out`` outputs (one for a
    number or a probability; one per class for a multiclass model), and the support's accounting.

    ``rows`` are the positions in ``X`` of the rows that can be shifted at all; ``diffs`` is
    k × those rows × outputs (NaN off support); ``masks`` is k × those rows."""

    n_rows: int
    rows: np.ndarray
    n_valid: int
    diffs: np.ndarray
    masks: np.ndarray
    fractions: List[float]
    n_on: List[int]
    off_amount: List[int]
    off_share: List[int]
    stopped_at: Optional[float]
    live: np.ndarray


def _walk(predict_matrix: Callable[[pd.DataFrame], np.ndarray], X: pd.DataFrame, shift: "Shift",
          k_values: np.ndarray, min_support: float, n_out: int) -> _Walk:
    """Move every row k at a time and record each output's change where the row is on support.

    ``predict_matrix(frame)`` returns len(frame) × ``n_out`` predictions. The curve stops at the
    first k where fewer than ``min_support`` of the rows of ``X`` are on support."""
    n_rows = int(len(X))
    rows = np.flatnonzero(shift.valid(X))  # a row that cannot be shifted is off-support at every k
    base_frame = X.iloc[rows]
    base = predict_matrix(base_frame)

    n_valid = rows.size
    diffs = np.full((k_values.size, n_valid, n_out), np.nan)
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
        diffs[i, mask] = predict_matrix(shifted.iloc[np.flatnonzero(mask)]) - base[mask]
        masks[i] = mask

    live = np.array([stopped_at is None or k < stopped_at for k in k_values]) & masks.any(axis=1)
    return _Walk(n_rows, rows, n_valid, diffs, masks, fractions, n_on, off_amount, off_share,
                 stopped_at, live)


def _averages(walk: _Walk, output: int, weights: Optional[Sequence[float]]) -> tuple:
    """(the curve, the fixed-population curve, the fixed population's mask) of one output.

    Each k averages over its own on-support rows; under a surveyed population (MS4) each row counts
    as the people its survey weight stands for."""
    w = None if weights is None else np.asarray(weights, dtype=float)[walk.rows]
    diffs = walk.diffs[:, :, output]

    def mean(values: np.ndarray, where: np.ndarray) -> float:
        return float(np.mean(values[where])) if w is None else float(np.average(values[where],
                                                                                weights=w[where]))

    K = diffs.shape[0]
    delta: List[Optional[float]] = [
        mean(diffs[i], walk.masks[i]) if walk.live[i] else None for i in range(K)]
    fixed = _fixed_population(walk.masks, walk.live)
    fixed_delta: List[Optional[float]] = [
        mean(diffs[i], fixed) if walk.live[i] and fixed.any() else None for i in range(K)]
    return delta, fixed_delta, fixed


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
    started = time.perf_counter()
    n_boot, k_values, live = _band_arguments(n_boot, interval, level, min_ok_share, ks, live)
    drawn = _refit_draws(fit, X, y, lambda predict, frame: _predictions(predict, frame)[:, None], 1,
                         shift=shift, k_values=k_values, live=live, n_boot=n_boot, groups=groups,
                         random_state=random_state, center=center, progress=progress,
                         curve_rows=curve_rows, resample_size=resample_size, max_rows=max_rows)
    n_units, m, scale = drawn.n_units, drawn.m, drawn.scale
    n_ok, needed, method, refused = _band_verdict(n_boot, drawn.failed, interval, min_ok_share)
    low, high = _interval(drawn.draws[:, :, 0], live, center, needed, method, float(level), scale)
    if m < n_units and fixed_center is None:  # a rescaled band has no center of its own
        fixed_low = fixed_high = [None] * k_values.size
    else:
        fixed_low, fixed_high = _interval(drawn.fixed_draws[:, :, 0], live, fixed_center, needed,
                                          method, float(level), scale)
    return {"ci_low": low, "ci_high": high, "fixed_ci_low": fixed_low, "fixed_ci_high": fixed_high,
            "n_boot": n_boot, "n_ok": n_ok, "failed": drawn.failed,
            "seconds": round(time.perf_counter() - started, 3), "interval": method,
            "level": float(level), "n_units": n_units, "resample_size": m, "scale": scale,
            "min_ok_share": float(min_ok_share), "refused": refused}


def _band_arguments(n_boot: int, interval: str, level: float, min_ok_share: float,
                    ks: Sequence[float], live: Sequence[bool]) -> tuple:
    """(n_boot, the ks, live) checked: what every band of refits needs."""
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
    return n_boot, k_values, live


def _band_verdict(n_boot: int, failed: int, interval: str, min_ok_share: float) -> tuple:
    """(refits that succeeded, the count a band needs, the interval drawn, why none is drawn)."""
    n_ok = n_boot - failed
    needed = max(2, math.ceil(float(min_ok_share) * n_boot))
    method = interval
    if method == "auto":
        method = "percentile" if n_ok >= PERCENTILE_MIN_REFITS else "normal"
    refused = None
    if n_ok < needed:
        refused = (f"{n_ok} of {n_boot} refits succeeded; a band needs at least "
                   f"{float(min_ok_share):.0%} of them")
    return n_ok, needed, method, refused


@dataclass
class _Draws:
    """Each refit's curve at each k, for each output (refits × k × outputs; NaN where a refit
    failed or a k is not live), its fixed-population twin, and how the resamples were drawn."""

    draws: np.ndarray
    fixed_draws: np.ndarray
    failed: int
    n_units: int
    m: int
    scale: float


def _refit_draws(fit: Callable[[pd.DataFrame, np.ndarray], Any], X: pd.DataFrame, y: Any,
                 values: Callable[[Any, pd.DataFrame], np.ndarray], n_out: int, *, shift: Shift,
                 k_values: np.ndarray, live: np.ndarray, n_boot: int, groups: Any,
                 random_state: int, center: Optional[Sequence[Any]],
                 progress: Optional[Callable[[int, int], None]],
                 curve_rows: Optional[Sequence[int]], resample_size: Optional[int],
                 max_rows: Optional[int]) -> _Draws:
    """The refits behind a band (:func:`refit_band`; :func:`class_refit_band`): ``values(predict,
    frame)`` turns a refit's prediction function into len(frame) × ``n_out`` predictions."""
    y = np.asarray(y)
    if len(y) != len(X):
        raise ValueError("X and y must have the same rows.")
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
    draws = np.full((n_boot, k_values.size, n_out), np.nan)
    fixed_draws = np.full((n_boot, k_values.size, n_out), np.nan)
    failed = 0
    for b in range(n_boot):
        picked = rng.integers(0, n_units, size=m)
        idx = picked if units is None else np.concatenate([units[j] for j in picked])
        try:
            predict = fit(X.iloc[idx], y[idx])
            base = values(predict, frame)
            at = position[idx]
            weight = np.bincount(at[at >= 0], minlength=rows.size).astype(float)
            for i, (on, moved_on) in prepared.items():
                w = weight[on]
                if w.sum() <= 0:
                    continue
                diff = values(predict, moved_on) - base[on]
                wf = w * fixed[on]
                for c in range(n_out):
                    draws[b, i, c] = float(np.average(diff[:, c], weights=w))
                    if wf.sum() > 0:
                        fixed_draws[b, i, c] = float(np.average(diff[:, c], weights=wf))
        except (ValueError, IndexError):
            failed += 1
            draws[b] = np.nan
            fixed_draws[b] = np.nan
        if progress is not None:
            progress(b + 1, n_boot)
    return _Draws(draws, fixed_draws, failed, n_units, m, scale)


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


# ── a multiclass outcome: one curve per class ─────────────────────────────────

# Each row's predicted probabilities sum to one to within a few units in the last place, and an
# average adds no more, so the class curves sum to zero to within this (relative to the size of the
# changes). A rounding tolerance, not a threshold of judgment.
CLASS_SUM_TOLERANCE = 1e-9


def level_name(value: Any) -> str:
    """A class as a sentence names it: ``high``, ``2`` (never ``2.0``), ``True``."""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return _plain(float(value))
    return str(value)


def check_sums_to_zero(curves: Sequence[Sequence[Optional[float]]], what: str = "the class curves"
                       ) -> float:
    """Raise unless ``curves`` (one per class, each a value or None at every k) sum to zero at
    every k where any is drawn; return the largest absolute sum.

    Every class's probability changes on the same rows, and each row's changes sum to zero, so the
    curves do too. A set that does not (a class drawn where another is not, or a sum off zero) is a
    defect in what predicted the probabilities, raised rather than drawn."""
    curves = [list(c) for c in curves]
    if not curves:
        return 0.0
    largest = 0.0
    for i in range(len(curves[0])):
        values = [c[i] for c in curves]
        drawn = [v for v in values if v is not None]
        if not drawn:
            continue
        if len(drawn) != len(values):
            raise ArithmeticError(f"{what[0].upper()}{what[1:]} are drawn for some classes and not "
                                  f"others at the k numbered {i}; every class's curve averages over "
                                  f"the same rows.")
        total = float(math.fsum(drawn))
        size = max(1.0, float(math.fsum(abs(v) for v in drawn)))
        if not math.isfinite(total) or abs(total) > CLASS_SUM_TOLERANCE * size:
            raise ArithmeticError(f"{what[0].upper()}{what[1:]} sum to {total:.3g} at the k "
                                  f"numbered {i}, not to zero: the class probabilities they follow "
                                  f"do not sum to one.")
        largest = max(largest, abs(total))
    return largest


def _class_predictions(predict: Callable[[pd.DataFrame], Any], frame: pd.DataFrame,
                       n_classes: int) -> np.ndarray:
    if len(frame) == 0:
        return np.empty((0, n_classes))
    values = np.asarray(predict(frame), dtype=float)
    if values.shape != (len(frame), n_classes):
        raise ValueError(f"predict must return the probability of each of the {n_classes} classes "
                         f"for every row, not an array of shape {values.shape}.")
    return values


def class_curves(predict_proba: Callable[[pd.DataFrame], Any], X: pd.DataFrame, *,
                 classes: Sequence[Any], donor: str, recipient: str,
                 kcal_per_unit: Mapping[str, float], ks: Sequence[float],
                 total_kind: TotalKind = "variable", min_support: float = 0.5,
                 nested: Optional[Mapping[str, str]] = None, label_k: Optional[float] = None,
                 total: Optional[str] = None, scale: Scale = "kcal", shift: Optional[Shift] = None,
                 weights: Optional[Sequence[float]] = None) -> Dict[str, Any]:
    """One substitution curve per class of a multiclass outcome (module docstring).

    ``predict_proba(frame)`` returns len(frame) × len(``classes``) probabilities, in the order of
    ``classes``. Every other argument is :func:`substitution_curve`'s: the support, the stopping
    rule and the weights are the same for every class, so every class's curve averages over the
    same rows at each k, and the curves sum to zero there (checked: :func:`check_sums_to_zero`).

    Returns :func:`substitution_curve`'s support accounting (``ks, live, on_support_fraction,
    n_on_support, n_rows, stopped_at, total_kind, label_k, min_support, units_moved, carried,
    total, n_not_recorded, n_off_amount, n_off_share, note, scale``; ``fixed_population`` holds
    ``n_rows`` and ``through``) and ``classes``: per class, in order, ``{level, name, delta,
    fixed_delta, effect_label, effect_sentence}``, each label the change in that class's
    probability per 100 kcal (or 5% of energy) at the stated k; ``class_sum`` is the largest
    absolute sum of the curves over k (zero up to rounding).
    """
    classes = list(classes)
    if len(classes) < 3:
        raise ValueError("A multiclass curve follows three or more classes; a yes/no outcome's "
                         "single curve follows the probability of its event.")
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
    C = len(classes)
    walk = _walk(lambda frame: _class_predictions(predict_proba, frame, C), X, shift, k_values,
                 float(min_support), C)
    per_class = []
    fixed = np.zeros(walk.n_valid, dtype=bool)
    for c in range(C):
        delta, fixed_delta, fixed = _averages(walk, c, weights)
        per_class.append((delta, fixed_delta))
    largest = check_sums_to_zero([d for d, _ in per_class], "the class curves")
    check_sums_to_zero([f for _, f in per_class], "the fixed-population class curves")
    live = walk.live
    through = float(k_values[np.flatnonzero(live)[-1]]) if live.any() else None
    per = PER_UNIT[scale]
    chosen = _label_k(k_values, per_class[0][0], label_k, per)
    ks_out = [float(k) for k in k_values]
    within = ("whose shifted intakes and shares of energy stay within the observed range"
              if shift.total is not None else "whose shifted intakes stay within the observed range")
    out = []
    for level, (delta, fixed_delta) in zip(classes, per_class):
        name = level_name(level)
        if chosen is None:
            label = None
            sentence = (f"No k > 0 kept enough rows on support to report an effect of moving "
                        f"energy from {donor} to {recipient}.")
        else:
            value = delta[ks_out.index(chosen)]
            label = (f"{_signed(value * per / chosen)} in the probability of {name} per "
                     f"{_amount(per, scale)} at k = {_plain(chosen)}")
            share = walk.fractions[ks_out.index(chosen)]
            sentence = (f"Moving {_amount(chosen, scale)} from {donor} to {recipient} changes the "
                        f"average predicted probability of {name} by {_signed(value)}, over the "
                        f"{share:.0%} of rows {within}.")
        out.append({"level": level, "name": name, "delta": delta, "fixed_delta": fixed_delta,
                    "effect_label": label, "effect_sentence": sentence})
    return {
        "donor": donor,
        "recipient": recipient,
        "ks": ks_out,
        "live": [bool(v) for v in live],
        "on_support_fraction": walk.fractions,
        "n_on_support": walk.n_on,
        "n_rows": walk.n_rows,
        "stopped_at": walk.stopped_at,
        "total_kind": total_kind,
        "label_k": chosen,
        "min_support": float(min_support),
        "units_moved": {donor: -shift.per_donor, recipient: shift.per_recipient},
        "carried": list(shift.carried),
        "total": shift.total,
        "n_not_recorded": int(walk.n_rows - walk.n_valid),
        "n_off_amount": walk.off_amount,
        "n_off_share": walk.off_share,
        "fixed_population": {"n_rows": int(fixed.sum()), "through": through},
        "note": _note(donor, recipient, total_kind, float(min_support), walk.stopped_at,
                      walk.fractions, ks_out, shift.note(), shift.total, int(fixed.sum()), through,
                      scale),
        "scale": scale,
        "classes": out,
        "class_sum": largest,
    }


def class_refit_band(fit: Callable[[pd.DataFrame, np.ndarray], Callable[[pd.DataFrame], Any]],
                     X: pd.DataFrame, y: Any, *, classes: Sequence[Any], shift: Shift,
                     ks: Sequence[float], live: Sequence[bool], n_boot: int, groups: Any = None,
                     random_state: int = 0, level: float = 0.95,
                     centers: Optional[Sequence[Sequence[Optional[float]]]] = None,
                     fixed_centers: Optional[Sequence[Sequence[Optional[float]]]] = None,
                     progress: Optional[Callable[[int, int], None]] = None,
                     curve_rows: Optional[Sequence[int]] = None,
                     resample_size: Optional[int] = None, max_rows: Optional[int] = None,
                     interval: Interval = "auto",
                     min_ok_share: float = MIN_REFIT_SHARE) -> Dict[str, Any]:
    """Each class's band from ``n_boot`` refits on bootstrap resamples of ``X``.

    :func:`refit_band`, class by class: every refit's prediction function (``fit(X_b, y_b)``)
    returns each class's probability in the order of ``classes`` (a resample missing a class
    cannot, so its refit fails and is counted), each refit's curve is taken for every class over
    the same on-support rows, and each class's interval is read from its own refits. ``centers``
    and ``fixed_centers`` are the class curves' own deltas (one list per class). Every resample,
    the interval rule, the rescaling of an m-out-of-n bootstrap and the share of refits a band
    needs are :func:`refit_band`'s.

    Returns ``{classes, n_boot, n_ok, failed, seconds, interval, level, n_units, resample_size,
    scale, min_ok_share, refused}``; ``classes`` holds per class ``{level, ci_low, ci_high,
    fixed_ci_low, fixed_ci_high, se, fixed_se}``, ``se`` each k's bootstrap standard error (the
    refits' spread, rescaled), None where no band is drawn.
    """
    started = time.perf_counter()
    classes = list(classes)
    C = len(classes)
    n_boot, k_values, live = _band_arguments(n_boot, interval, level, min_ok_share, ks, live)
    if centers is not None and len(centers) != C:
        raise ValueError("centers must hold one curve per class.")
    drawn = _refit_draws(fit, X, y, lambda predict, frame: _class_predictions(predict, frame, C),
                         C, shift=shift, k_values=k_values, live=live, n_boot=n_boot,
                         groups=groups, random_state=random_state, center=centers,
                         progress=progress, curve_rows=curve_rows, resample_size=resample_size,
                         max_rows=max_rows)
    n_units, m, scale = drawn.n_units, drawn.m, drawn.scale
    n_ok, needed, method, refused = _band_verdict(n_boot, drawn.failed, interval, min_ok_share)
    out = []
    for c, name in enumerate(classes):
        center = centers[c] if centers is not None else None
        low, high = _interval(drawn.draws[:, :, c], live, center, needed, method, float(level),
                              scale)
        if m < n_units and fixed_centers is None:
            fixed_low = fixed_high = [None] * k_values.size
        else:
            fixed_low, fixed_high = _interval(drawn.fixed_draws[:, :, c], live,
                                              fixed_centers[c] if fixed_centers is not None
                                              else None, needed, method, float(level), scale)
        out.append({"level": name, "ci_low": low, "ci_high": high, "fixed_ci_low": fixed_low,
                    "fixed_ci_high": fixed_high,
                    "se": _spread(drawn.draws[:, :, c], live, needed, scale, low),
                    "fixed_se": _spread(drawn.fixed_draws[:, :, c], live, needed, scale,
                                        fixed_low)})
    return {"classes": out, "n_boot": n_boot, "n_ok": n_ok, "failed": drawn.failed,
            "seconds": round(time.perf_counter() - started, 3), "interval": method,
            "level": float(level), "n_units": n_units, "resample_size": m, "scale": scale,
            "min_ok_share": float(min_ok_share), "refused": refused}


def _spread(draws: np.ndarray, live: np.ndarray, needed: int, scale: float,
            drawn: Sequence[Optional[float]]) -> List[Optional[float]]:
    """Per k: the refits' standard deviation times ``scale`` where a band is drawn, else None."""
    out: List[Optional[float]] = []
    for i in range(draws.shape[1]):
        column = draws[:, i][np.isfinite(draws[:, i])]
        if not live[i] or column.size < needed or drawn[i] is None:
            out.append(None)
            continue
        out.append(scale * float(np.std(column, ddof=1)))
    return out


def pool_class_curves(copies: Sequence[Mapping[str, Any]],
                      se: Optional[Sequence[Sequence[Sequence[Optional[float]]]]] = None,
                      fixed_se: Optional[Sequence[Sequence[Sequence[Optional[float]]]]] = None,
                      df_com: Optional[float] = None, level: float = 0.95) -> Dict[str, Any]:
    """Each class's curve pooled over imputed copies at each k (MS3; Rubin 1987).

    ``copies`` are each completed copy's :func:`class_curves` (the same classes and ks). At each k
    where every copy's curve is drawn, the pooled curve is the mean of the copies' (Q̄). With
    ``se`` (``se[j][c][i]``: copy j's within-copy standard error for class c at the k numbered i, a
    bootstrap's or the design's), the interval is Rubin's: total variance T = Ū + (1 + 1/m)B on
    Barnard & Rubin's (1999) degrees of freedom, ``df_com`` the complete-data df (the design's
    under a survey; None: large-sample). ``fixed_se`` does the same for the fixed-population curves.
    The pooled curves are means of curves that sum to zero, so they sum to zero (checked).

    Returns ``{classes, class_sum}``; ``classes`` per class ``{level, delta, ci_low, ci_high, df,
    fixed_delta, fixed_ci_low, fixed_ci_high}``.
    """
    from turbotab.core.methods.imputation import pool_scalar

    copies = list(copies)
    if not copies:
        raise ValueError("Pooling needs at least one imputed copy's curves.")
    m = len(copies)
    C = len(copies[0]["classes"])
    n_k = len(copies[0]["ks"])
    if any(len(c["classes"]) != C or len(c["ks"]) != n_k for c in copies):
        raise ValueError("Every copy's curves must follow the same classes at the same ks.")

    def pooled(values: List[Optional[float]], errors: Optional[List[Optional[float]]]) -> tuple:
        if any(v is None for v in values):
            return None, None, None, None
        q = [float(v) for v in values]
        if errors is None or any(e is None for e in errors):
            return float(np.mean(q)), None, None, None
        p = pool_scalar(q, [float(e) ** 2 for e in errors], df_com, level)
        return p.estimate, p.ci_low, p.ci_high, p.df

    out = []
    for c in range(C):
        delta, low, high, dfs = [], [], [], []
        fixed, fixed_low, fixed_high = [], [], []
        for i in range(n_k):
            d = pooled([copy["classes"][c]["delta"][i] for copy in copies],
                       [se[j][c][i] for j in range(m)] if se is not None else None)
            f = pooled([copy["classes"][c]["fixed_delta"][i] for copy in copies],
                       [fixed_se[j][c][i] for j in range(m)] if fixed_se is not None else None)
            delta.append(d[0])
            low.append(d[1])
            high.append(d[2])
            dfs.append(d[3])
            fixed.append(f[0])
            fixed_low.append(f[1])
            fixed_high.append(f[2])
        out.append({"level": copies[0]["classes"][c]["level"], "delta": delta, "ci_low": low,
                    "ci_high": high, "df": dfs, "fixed_delta": fixed, "fixed_ci_low": fixed_low,
                    "fixed_ci_high": fixed_high})
    largest = check_sums_to_zero([c["delta"] for c in out], "the pooled class curves")
    check_sums_to_zero([c["fixed_delta"] for c in out], "the pooled fixed-population class curves")
    return {"classes": out, "class_sum": largest}


def _class_jacobian(X: np.ndarray, P: np.ndarray, c: int) -> np.ndarray:
    """∂π_c/∂β for each row: ``π_c (1[c = a] − π_a) x`` for each non-reference class a = 1…K − 1,
    class by class (the order of :func:`~turbotab.core.models.survey.weighted_multinomial`)."""
    q = P.shape[1] - 1
    blocks = [(P[:, c] * ((c == a) - P[:, a]))[:, None] * X for a in range(1, q + 1)]
    return np.concatenate(blocks, axis=1)


def design_class_curves(matrix_of: Callable[[pd.DataFrame], pd.DataFrame], X: pd.DataFrame,
                        y: Any, classes: Sequence[Any], design: Any, domain: Any, *,
                        level: float = 0.95, **curve_args: Any) -> Dict[str, Any]:
    """Each class's curve over the surveyed population, and its band by linearization (MS4).

    ``X`` holds the domain's kept rows (every analyzed row placed in the design with a positive
    weight, in the order of ``domain``; :func:`~turbotab.core.models.survey.domain_of`), ``y``
    their classes, and ``matrix_of`` the fitted pipeline up to its model step. The multinomial
    logit is refit on that matrix with each row's survey weight (pseudo-maximum likelihood,
    :func:`~turbotab.core.models.survey.weighted_multinomial`); each class's curve is the weighted
    mean of each row's change in that class's probability (:func:`class_curves` with the weights).

    The band at each k is Graubard & Korn's (1999, *Biometrics* 55:652) linearization of each
    class's predictive margin, as :func:`~turbotab.core.models.survey.design_curve` takes it for a
    yes/no outcome: ``z_i = w_i m_i (Δ_ic − θ_ck)/Σ w m + ψ_iᵀ g_ck``, ``ψ_i`` row i's influence on
    the coefficients (the bread times its weighted score) and ``g_ck`` the weighted mean of
    ``∂Δ_ic/∂β``; its variance is the design's variance of a total
    (:func:`~turbotab.core.models.survey.total_variance`: PSUs within strata, the domain's rows
    scored and every other row zero), and the interval is on t with the design's degrees of
    freedom. ``curve_args`` are :func:`class_curves`'s (``shift`` among them).

    Raises ValueError, saying why, where no design-based fit exists: a class with no weighted row,
    too few rows for the coefficients, or a fit that does not converge (a column separating the
    classes). Returns ``{curves, band}``: the :func:`class_curves` dict, and ``band`` with per class
    ``{level, ci_low, ci_high, se, fixed_ci_low, fixed_ci_high, fixed_se}``, the ``df`` and the
    ``variance`` (:class:`~turbotab.core.models.survey.DesignVariance`).
    """
    from scipy import stats

    from turbotab.core.models.survey import _softmax, total_variance, weighted_multinomial

    classes = list(classes)
    K = len(classes)
    if len(X) != domain.n:
        raise ValueError("The curve's rows are the domain's kept rows.")
    labels = pd.Series(np.asarray(y, dtype=object))
    codes = pd.Categorical(labels, categories=classes).codes.astype(np.int64)
    if (codes < 0).any():
        raise ValueError("Some analyzed rows hold a class the model was not fit on, or none.")
    empty = [level_name(classes[c]) for c in range(K) if not (codes == c).any()]
    if empty:
        raise ValueError(f"No analyzed row in the survey design is in the class {empty[0]}, so its "
                         f"probability has no weighted fit.")
    matrix = matrix_of(X)
    Xd = np.column_stack([np.ones(len(matrix)), matrix.to_numpy(dtype=float)])
    P_cols = Xd.shape[1]
    if domain.n <= (K - 1) * P_cols:
        raise ValueError(f"Only {domain.n:,} analysis rows carry a positive weight and a place in "
                         f"the design, too few for {(K - 1) * P_cols} coefficients.")
    fit = weighted_multinomial(Xd, codes, K, domain.weight)
    if not fit.converged:
        raise ValueError("The survey-weighted multinomial fit did not converge: a column may "
                         "separate the classes among the weighted rows, so a log-odds is "
                         "infinite.")
    B = fit.estimate.reshape(K - 1, P_cols).T
    influence = fit.scores @ np.linalg.pinv(fit.information)

    def design_of(frame: pd.DataFrame) -> np.ndarray:
        values = matrix_of(frame).to_numpy(dtype=float)
        return np.column_stack([np.ones(len(values)), values])

    def proba(frame: pd.DataFrame) -> np.ndarray:
        return _softmax(design_of(frame) @ B)

    shift = curve_args["shift"]
    curves = class_curves(proba, X, classes=classes, weights=domain.raw, **curve_args)
    k_values = np.asarray(curves["ks"], dtype=float)
    live = np.asarray(curves["live"], dtype=bool)
    valid = np.flatnonzero(shift.valid(X))
    base_frame = X.iloc[valid]
    X0 = design_of(base_frame)
    P0 = _softmax(X0 @ B)
    w = domain.raw[valid]
    n_k = len(k_values)
    checked: Dict[int, tuple] = {}
    for i, k in enumerate(k_values):
        if live[i]:
            shifted, amount, composition = shift.checks(base_frame, float(k))
            checked[i] = (shifted, amount & composition)
    fixed = (np.logical_and.reduce([on for _, on in checked.values()]) if checked
             else np.zeros(len(valid), dtype=bool))

    def slot(half: int, c: int, i: int) -> int:  # the curve's, then the fixed population's
        return (half * K + c) * n_k + i

    z = np.zeros((domain.n, 2 * K * n_k))
    theta = np.full(2 * K * n_k, np.nan)
    for i, (shifted, on) in checked.items():
        rows = np.flatnonzero(on)
        if not len(rows):
            continue
        X1 = design_of(shifted.iloc[rows])
        P1 = _softmax(X1 @ B)
        diff = P1 - P0[rows]
        for c in range(K):
            grad = _class_jacobian(X1, P1, c) - _class_jacobian(X0[rows], P0[rows], c)
            for half, chosen in ((0, np.ones(len(rows), dtype=bool)), (1, fixed[rows])):
                if not chosen.any():
                    continue
                wk = w[rows] * chosen
                total = float(wk.sum())
                value = float(wk @ diff[:, c]) / total
                s = slot(half, c, i)
                theta[s] = value
                z[valid[rows], s] += wk * (diff[:, c] - value) / total
                z[:, s] += influence @ ((wk @ grad) / total)
    u = np.zeros((design.n_rows, 2 * K * n_k))
    u[domain.at] = z
    var = total_variance(u, design, domain.mask(design))
    se = np.sqrt(np.clip(np.diag(var.meat), 0, None))
    t = float(stats.t.ppf(0.5 + level / 2, var.df)) if var.df >= 1 else float("nan")

    def band(half: int, c: int) -> tuple:
        low, high, errors = [], [], []
        for i in range(n_k):
            s = slot(half, c, i)
            if not np.isfinite(theta[s]) or not np.isfinite(t):
                low.append(None)
                high.append(None)
                errors.append(None)
                continue
            low.append(float(theta[s] - t * se[s]))
            high.append(float(theta[s] + t * se[s]))
            errors.append(float(se[s]))
        return low, high, errors

    out = []
    for c in range(K):
        low, high, errors = band(0, c)
        fixed_low, fixed_high, fixed_errors = band(1, c)
        out.append({"level": classes[c], "ci_low": low, "ci_high": high, "se": errors,
                    "fixed_ci_low": fixed_low, "fixed_ci_high": fixed_high,
                    "fixed_se": fixed_errors})
    return {"curves": curves, "band": {"classes": out, "df": int(var.df), "variance": var,
                                       "level": float(level)}}


# ── what the curves estimate, in words ───────────────────────────────────────


def class_estimand(target: str, classes: Sequence[Any], donor: str, recipient: str, *,
                   population: str, scale: str = "kcal", energy_out: bool = False,
                   carried: bool = False) -> str:
    """The estimand of a multiclass outcome's curves, as the artifact states it: the probability
    scale, the isocaloric move, and the population the curves average over (``population``: the
    surveyed population, these analyzed rows, or the training rows)."""
    names = [level_name(c) for c in classes]
    listed = ", ".join(names[:-1]) + " and " + names[-1]
    moved = ("k percent of each row's own total energy moves" if scale == "percent_energy"
             else "k kcal move")
    others = ("every other input left as it was (total energy is not in this model; only the swap "
              "itself keeps it fixed)" if energy_out else
              "every other input, total energy included, left as it was")
    parts = ", their parts or totals moving with them," if carried else ""
    return (f"For each class of {target} ({listed}), the average change in its predicted "
            f"probability, on the probability scale, when {moved} from {donor} to "
            f"{recipient}{parts} at the same total energy (isocaloric), with {others}, over "
            f"{population}. The class curves sum to zero at every k, because each row's class "
            f"probabilities sum to one.")


def class_clause(state: Any) -> Optional[str]:
    """The ``set_substitution`` sentence's clause for a multiclass outcome (its contract's
    sentence), or None. It reads the recorded task only, so the methods text restates it whole."""
    if getattr(state, "task", None) != "multiclass":
        return None
    from turbotab.core.voice import tick

    target = getattr(state, "target", None)
    outcome = f"the outcome {tick(target)}" if target else "the outcome"
    return (f"{outcome} has unordered classes, so there is one curve per class, each the average "
            f"change in that class's predicted probability, and the curves sum to zero at every k")


# ── the method contract (BLUEPRINT §13) ──────────────────────────────────────

CLASS_CONTRACT = "multiclass_substitution"


def _contract_clause(run: Mapping[str, Any]) -> Optional[str]:
    """The contract's clause of the methods paragraph, from ``donor``, ``recipient``,
    ``step_kcal`` and how the bands were made (``estimand`` "population", ``m`` imputations,
    ``n_boot`` refits)."""
    donor, recipient = run.get("donor"), run.get("recipient")
    if not donor or not recipient:
        return None
    step = run.get("step_kcal")
    steps = f" in steps of {_plain(float(step))} kcal" if step else ""
    text = (f"for the multiclass outcome, one substitution curve per class was drawn, the average "
            f"change in that class's predicted probability as energy moved from `{donor}` to "
            f"`{recipient}`{steps} at the same total energy, the curves summing to zero at every k")
    m = run.get("m")
    if run.get("estimand") == "population":
        text += ("; each class's curve came from the survey-weighted multinomial fit, averaged "
                 "with the weights, with its band by Taylor linearization over the survey design "
                 "(Graubard & Korn 1999)")
        if m:
            text += f", pooled over {int(m)} imputations by Rubin's rules at each k"
    elif m:
        text += f"; each copy's curves were pooled over {int(m)} imputations by Rubin's rules at each k"
    elif run.get("n_boot"):
        text += (f"; each class's band came from {int(run['n_boot']):,} refits on bootstrap "
                 f"resamples")
    return text


def _register_contract() -> None:
    from turbotab.core.contracts import ContractOption as Option
    from turbotab.core.contracts import MethodContract, Relation, register_contract

    register_contract(MethodContract(
        key=CLASS_CONTRACT, label="Substitution curves for a multiclass outcome, one per class",
        slot="evaluation", scope="model", package="MULTISUB", run_order=2.0,
        scope_note=("Each curve follows the fitted outcome model, so a row's predicted change moves "
                    "with the outcome and with every row the model was fit on. Lockbox §06's test: "
                    "row i's change in each class's probability moves when the outcome does."),
        needs=("a multiclass outcome (three or more unordered classes)",
               "two energy-bearing exposures whose kcal per unit is settled",
               "a fitted family that predicts each class's probability",
               "the substitution answer (donor, recipient and step)"),
        question=("Which energy substitution, in what steps? (the substitution question; a "
                  "multiclass outcome draws one curve per class)"),
        place=("MODELING_SEQUENCE §1 step 11's displays: beside the coefficient table under "
               "inference, beside the comparison under prediction"),
        decision="set_substitution", stage="substitution",
        options=(
            Option("class_curves",
                   "One curve per class: the average change in each class's predicted probability",
                   "Average discrete changes of a multinomial model on each outcome's probability "
                   "(Long & Freese 2014, ch. 8; Stata margins and R marginaleffects per outcome)",
                   {"inference": "Sound: a marginal, isocaloric contrast on the probability scale "
                                 "over the stated population; the curves sum to zero at every k, "
                                 "as each person's probabilities sum to one.",
                    "prediction": "Sound as a model contrast: what the fitted model predicts for "
                                  "each class when k kcal move."},
                   {"inference": "recommended", "prediction": "recommended"}),
            Option("reference_ratios",
                   "Relative-risk ratios against the reference class per kcal swapped",
                   "The multinomial coefficients' contrast, as categorical-outcome analyses "
                   "report them",
                   {"inference": "Conditional on every covariate and against one reference class: "
                                 "it says how the odds of a class against the reference move, not "
                                 "how any class's probability does, and it is one number only when "
                                 "every energy source is a linear term. The coefficient table "
                                 "reports each class's ratios per unit.",
                    "prediction": "Not a prediction: the class curves say what the model "
                                  "predicts."},
                   {"inference": "not_offered", "prediction": "not_offered"}),
        ),
        leash={"inference": "recommended", "prediction": "recommended"},
        storyboard=("Predict every class's probability on every row",
                    "Move k kcal from the donor to the recipient on every row on support",
                    "Predict again, and take each class's change",
                    "Average each class's change over the same rows (weighted under a surveyed "
                    "population)",
                    "Check that the class curves sum to zero at every k",
                    "Band each class: refits on bootstrap resamples, Rubin's rules over the "
                    "imputations, or the survey design's linearization"),
        relations=(
            Relation("implies", "class_probabilities_sum_to_one",
                     "The class curves sum to zero at every k, because each row's class "
                     "probabilities sum to one before and after the move; a set that does not is a "
                     "defect, raised and never drawn.",
                     enforced_by="turbotab.core.methods.substitution:check_sums_to_zero",
                     id="curves_sum_to_zero"),
            Relation("implies", "refit_band",
                     "Each class's band comes from refits of the model on bootstrap resamples of "
                     "the rows it was fit on (whole units when rows repeat), its interval read "
                     "from that class's refits.",
                     enforced_by="turbotab.core.methods.substitution:class_refit_band",
                     condition="a band asked for (n_boot > 0) with no surveyed population",
                     id="refit_band"),
            Relation("implies", "multiple_imputation_compatible",
                     "Each completed copy's class curves are drawn on that copy's rows and fit, "
                     "and pooled at each k by Rubin's rules; a curve from one fill is never shown.",
                     purposes=("inference",),
                     enforced_by="turbotab.core.methods.substitution:pool_class_curves",
                     condition="multiple imputation under inference", id="pooled_per_k"),
            Relation("implies", "survey_population",
                     "Each class's curve comes from the survey-weighted multinomial fit, averaged "
                     "with the weights, and its band is Taylor linearization over the survey "
                     "design; refits on bootstrap resamples of rows are not drawn.",
                     purposes=("inference",),
                     enforced_by="turbotab.core.methods.substitution:design_class_curves",
                     condition="the survey answer \"the surveyed population\"",
                     id="design_based"),
            Relation("conflicts", "survey_population",
                     "A family with no design-based estimator draws no class curves under the "
                     "surveyed population: blocked and recorded.",
                     purposes=("inference",), rung="block_and_record",
                     exits=("the design-based family in its place, every other chosen family kept",
                            "the sample-only attestation"),
                     enforced_by="turbotab.core.stages.class_substitution:population_blocked",
                     condition="the survey answer \"the surveyed population\" and a family with "
                               "no design-based estimator",
                     id="blocked_family"),
            Relation("conflicts", "omitted_energy_sources",
                     "Energy sources left out of the model, above the stated share of total "
                     "energy, block the swap under inference until it is recorded: the curves "
                     "carry the confounding of the sources total energy holds as one composite.",
                     purposes=("inference",), rung="block_and_record",
                     exits=("add each missing energy source to the model as an exposure",
                            "Keep this swap; the curve carries their confounding",
                            "Choose another swap"),
                     enforced_by="turbotab.core.decisions:_substitution_has_every_energy_source",
                     condition="energy sources left out above MAX_OMITTED_SHARE of total energy",
                     id="omitted_sources"),
            Relation("implies", "omitted_energy_sources",
                     "Under prediction the curves are model contrasts: the sources the model "
                     "leaves out are named as a concern, never blocked.",
                     purposes=("prediction",),
                     enforced_by="turbotab.core.methods.energy:omitted_sentence",
                     condition="energy sources left out of the model", id="omitted_stated"),
            Relation("implies", "estimand_label",
                     "The estimand names the probability scale, the isocaloric move and the "
                     "population the curves average over; each class's label is the change in "
                     "its probability at the stated k, an average over that population, since a "
                     "multinomial model's change depends on k and on each person's intake "
                     "(MODELING_SEQUENCE §2).",
                     enforced_by="turbotab.core.methods.substitution:class_estimand",
                     id="estimand_label"),
        ),
        sources=("Long & Freese 2014, Regression Models for Categorical Dependent Variables Using "
                 "Stata, 3rd ed., ch. 8",
                 "Graubard & Korn 1999, Biometrics 55:652",
                 "Rubin 1987, Multiple Imputation for Nonresponse in Surveys",
                 "Tomova, Gilthorpe & Tennant 2022 (substitution models; PMC9630885)",
                 "MODELING_SEQUENCE §2, §4"),
        clause=_contract_clause, sentence="turbotab.core.methods.substitution:class_clause"))


_register_contract()

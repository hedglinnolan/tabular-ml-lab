"""Substitution in percent of energy: "5% of energy from X replaced by Y" (audit B24, D19).

NUTRITION_PACK §05 names the field's expected figure: "**★ Isocaloric substitution forest
plot** — … Rows of the form '5% of energy from X replaced by Y', estimate + 95% CI", and the
leave-one-out model behind it: "all components except one, plus total energy … Each coefficient
is the effect of substituting that component for the omitted one." The kcal curve
(:mod:`turbotab.core.methods.substitution`) moves the same k kcal on every row; this moves the
same *share* of each row's own total energy:

* through an amount (grams, kcal): ``k/100 · E_i`` kcal on row i, divided by the column's kcal
  per unit (Atwater: 9 for fat in grams);
* through a column already in percent of energy (``fat_pct_kcal``): k percentage points.

Total energy is left as it was, so the swap is isocaloric on every row. For a linear model on
shares of energy with every source but one plus total energy in it, the change is exactly
``k (β_Y − β_X)``, the pack's "explicit difference-in-coefficients" (way (c)); through amounts it
is ``k/100 · mean(E) · (β_Y/f_Y − β_X/f_X)`` over the rows on support. For any other family it is
the model contrast the kcal curve is, averaged over rows on support.

Support: each shifted column must stay inside its observed range and not go below zero, and its
share of energy inside the observed range of that share (a share-of-energy column is its own
share). A row whose total energy is not recorded cannot be shifted through an amount.
"""
from __future__ import annotations

import re
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from turbotab.core.methods.substitution import Shift, _values

# Share-of-energy suffixes: the percent members of the energy module's density suffixes. A
# per-1,000-kcal density (``_per1000kcal``) is grams per energy, not a percentage, and is not one.
PERCENT_SUFFIX = r".*(_pct_energy|_pct_kcal|_percent_energy)$"
UNIT = "% of energy"


def is_percent_of_energy(column: str) -> bool:
    """True when ``column``'s name says it is a macronutrient's percent of total energy."""
    from turbotab.core.methods.energy import nutrient_role

    if not re.fullmatch(PERCENT_SUFFIX, str(column).lower()):
        return False
    try:
        return nutrient_role(column) is not None
    except ValueError:  # names two macronutrients
        return False


def check_percent_values(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    """Refuse a share-of-energy column whose values are not percentages of energy."""
    for column in columns:
        values = _values(frame, column)
        values = values[np.isfinite(values)]
        if not values.size:
            raise ValueError(f"{column} has no recorded values.")
        if values.min() < 0 or values.max() > 100:
            raise ValueError(f"{column} reads as a percent of energy, but its values run from "
                             f"{values.min():g} to {values.max():g}, outside 0 to 100.")
        if values.max() <= 1:
            raise ValueError(f"{column} reads as a percent of energy, but every value is at most 1: "
                             f"it holds fractions, so a 5-point step would be 500% of energy.")


class PercentEnergyShift(Shift):
    """Move k percent of each row's total energy from ``donor`` to ``recipient``.

    ``percent``: the moved columns already in percent of energy. Every other moved column is an
    amount with ``kcal_per_unit``. ``total`` (the total-energy column) is required when an amount
    moves, since the kcal moved on a row is ``k/100`` of its own total.
    """

    def __init__(self, reference: pd.DataFrame, *, donor: str, recipient: str,
                 kcal_per_unit: Mapping[str, float], percent: Sequence[str] = (),
                 nested: Optional[Mapping[str, str]] = None, total: Optional[str] = None):
        self.percent = {str(c) for c in percent}
        unknown = self.percent - {donor, recipient}
        if unknown:
            raise ValueError(f"{', '.join(sorted(unknown))} is not moved by this substitution.")
        amounts = [c for c in (donor, recipient) if c not in self.percent]
        if amounts and total is None:
            raise ValueError(f"Moving a share of energy through {' and '.join(amounts)} needs each "
                             f"row's total energy, and no total-energy column is among the "
                             f"model's inputs.")
        nested = dict(nested or {})
        tangled = [c for c in self.percent if c in nested or c in nested.values()]
        if tangled:
            raise ValueError(f"{', '.join(tangled)} is a share of energy nested with another "
                             f"column; moving shares through parts and totals is not supported.")
        check_percent_values(reference, sorted(self.percent))
        # A share column's own factor is not a constant (one point is E_i/100 kcal); the parent
        # only needs a positive placeholder for it, and every use of it is overridden here.
        factors = {**{c: 1.0 for c in self.percent}, **{c: float(kcal_per_unit[c]) for c in amounts}}
        super().__init__(reference, donor=donor, recipient=recipient, kcal_per_unit=factors,
                         nested=nested, total=total)

    def _share(self, column: str, values: np.ndarray, energy: np.ndarray) -> np.ndarray:
        if column in self.percent:
            return values / 100.0
        return super()._share(column, values, energy)

    def kcal_moved(self, frame: pd.DataFrame, k: float) -> np.ndarray:
        """kcal moved on each row at k percent of energy (NaN where total energy is not recorded)."""
        if self.total is None:
            return np.full(len(frame), np.nan)
        return float(k) / 100.0 * _values(frame, self.total)

    def values(self, frame: pd.DataFrame, k: float) -> Dict[str, np.ndarray]:
        moved = self.kcal_moved(frame, k)
        new: Dict[str, np.ndarray] = {}
        for column, sign in ((self.donor, -1.0), (self.recipient, 1.0)):
            old = _values(frame, column)
            step = float(k) if column in self.percent else moved / self.factors[column]
            new[column] = old + sign * step
        for column in (self.donor, self.recipient):
            if column in self.percent:
                continue
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


__all__ = ["PERCENT_SUFFIX", "PercentEnergyShift", "UNIT", "check_percent_values",
           "is_percent_of_energy"]

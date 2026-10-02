"""Pipeline steps the model families share, beyond scikit-learn's own.

``energy_step`` turns an ``EnergyAdjustment`` slot into its pipeline step. The residual method
within the levels of a strata column (NUTRITION_PACK §04: "within sex") is
:class:`~turbotab.core.methods.energy.StratifiedEnergyAdjuster`, which lives with the rest of the
energy algebra and is re-exported here.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from turbotab.core.methods.energy import (
    MIN_LEVEL_ROWS,
    RESIDUAL_METHODS,
    EnergyAdjuster,
    StratifiedEnergyAdjuster,
)


def energy_step(adjustment: Any, predictors: Sequence[str],
                roles: Mapping[str, str] | None = None) -> Any | None:
    """The energy-adjustment pipeline step for an ``EnergyAdjustment`` slot, or None.

    None when nothing was answered: the predictors then reach the model as their roles give them.
    "none" is a step whenever a total-energy column is among the predictors (its own
    ``energy_column``, or any column with the energy role in ``roles``): the step takes it out of
    the model, because "no energy adjustment" means total energy is not in it (audit ME-02).

    The strata column stratifies the residual methods only; it leaves the output when it is not
    itself a predictor. Either way the adjusted nutrient carries no difference between levels
    (one constant for every level), so leaving the strata column out of the model is sound.
    """
    if adjustment is None:
        return None
    if adjustment.method == "none":
        energy = [c for c in dict.fromkeys(
            [adjustment.energy_column, *(c for c, r in (roles or {}).items() if r == "energy")])
            if c and c in predictors]
        if not energy:
            return None
        return EnergyAdjuster("none", energy[0], [], leave_out=energy[1:])
    if adjustment.strata and adjustment.method in RESIDUAL_METHODS:
        return StratifiedEnergyAdjuster(
            energy_column=adjustment.energy_column, nutrient_columns=list(adjustment.nutrients),
            strata=adjustment.strata, log_transform=adjustment.log_transform,
            drop_strata=adjustment.strata not in predictors, method=adjustment.method,
        )
    return EnergyAdjuster(adjustment.method, adjustment.energy_column, list(adjustment.nutrients),
                          log_transform=adjustment.log_transform)


__all__ = ["MIN_LEVEL_ROWS", "StratifiedEnergyAdjuster", "energy_step"]

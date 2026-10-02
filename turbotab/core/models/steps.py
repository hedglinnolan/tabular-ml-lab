"""Pipeline steps the model families share, beyond scikit-learn's own.

``energy_step`` turns an ``EnergyAdjustment`` slot into its pipeline step. The residual method
within the levels of a strata column (NUTRITION_PACK §04: "within sex") is
:class:`~turbotab.core.methods.energy.StratifiedEnergyAdjuster`, which lives with the rest of the
energy algebra and is re-exported here.
"""
from __future__ import annotations

from typing import Any, Sequence

from turbotab.core.methods.energy import MIN_LEVEL_ROWS, EnergyAdjuster, StratifiedEnergyAdjuster


def energy_step(adjustment: Any, predictors: Sequence[str]) -> Any | None:
    """The energy-adjustment pipeline step for an ``EnergyAdjustment`` slot, or None for none.

    The strata column stratifies the residual method only; it leaves the output when it is not
    itself a predictor. Either way the adjusted nutrient carries no difference between levels
    (one constant for every level), so leaving the strata column out of the model is sound.
    """
    if adjustment is None or adjustment.method == "none":
        return None
    if adjustment.strata and adjustment.method == "residual":
        return StratifiedEnergyAdjuster(
            energy_column=adjustment.energy_column, nutrient_columns=list(adjustment.nutrients),
            strata=adjustment.strata, log_transform=adjustment.log_transform,
            drop_strata=adjustment.strata not in predictors,
        )
    return EnergyAdjuster(adjustment.method, adjustment.energy_column, list(adjustment.nutrients),
                          log_transform=adjustment.log_transform)


__all__ = ["MIN_LEVEL_ROWS", "StratifiedEnergyAdjuster", "energy_step"]

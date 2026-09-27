"""Statistical methods for TurboTab Next: energy adjustment and substitution curves."""
from turbotab.core.methods.energy import (
    METHOD_TABLE,
    METHODS,
    EnergyAdjuster,
    EnergyAdjustmentNotApplicable,
    EnergyMethod,
    applicable_methods,
    default_atwater,
    describe_method,
    energy_factor,
)
from turbotab.core.methods.substitution import substitution_curve

__all__ = [
    "METHOD_TABLE",
    "METHODS",
    "EnergyAdjuster",
    "EnergyAdjustmentNotApplicable",
    "EnergyMethod",
    "applicable_methods",
    "default_atwater",
    "describe_method",
    "energy_factor",
    "substitution_curve",
]

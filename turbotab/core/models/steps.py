"""Pipeline steps the model families share, beyond scikit-learn's own.

``StratifiedEnergyAdjuster`` is the residual method computed within each level of a strata
column (NUTRITION_PACK §04: "within sex"): one ``N ~ E`` regression per level, each fit on that
level's fitting rows, and the adjusted value is ``N − b_s × (E − mean E_s)`` — the residual plus
the predicted intake at the level's own mean energy. A level with too few fitting rows, and a row
whose level is missing or was never seen in fit, uses the pooled regression instead; ``lineage()``
says which levels those were.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from turbotab.core.methods.energy import METHOD_TABLE, EnergyAdjuster

MIN_LEVEL_ROWS = 3  # EnergyAdjuster's own floor for one residual regression


def _fmt(value: float) -> str:
    return np.format_float_positional(float(value), precision=4, unique=False, fractional=False,
                                      trim="-")


class StratifiedEnergyAdjuster(TransformerMixin, BaseEstimator):
    """The residual method within each level of ``strata``. Every other column passes through.

    ``drop_strata``: the strata column is an input only (it is not a predictor), so it leaves
    the output.
    """

    def __init__(self, energy_column: str = "energy_kcal", nutrient_columns: Sequence[str] = (),
                 strata: str = "sex", log_transform: bool = False, drop_strata: bool = False,
                 atwater: Optional[Mapping[str, float]] = None):
        self.energy_column = energy_column
        self.nutrient_columns = nutrient_columns
        self.strata = strata
        self.log_transform = log_transform
        self.drop_strata = drop_strata
        self.atwater = atwater

    def _adjuster(self) -> EnergyAdjuster:
        return EnergyAdjuster("residual", self.energy_column, list(self.nutrient_columns),
                              log_transform=self.log_transform, atwater=self.atwater)

    def fit(self, X: pd.DataFrame, y: Any = None) -> "StratifiedEnergyAdjuster":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("StratifiedEnergyAdjuster needs a pandas DataFrame with named columns.")
        if self.strata not in X.columns:
            raise ValueError(f"The strata column {self.strata} is not among the inputs.")
        self.feature_names_in_ = np.asarray([str(c) for c in X.columns], dtype=object)
        self.n_features_in_ = X.shape[1]
        self.pooled_ = self._adjuster().fit(X)
        levels = X[self.strata]
        self.by_level_: dict[Any, EnergyAdjuster] = {}
        self.pooled_levels_: list[Any] = []
        self.level_rows_: dict[Any, int] = {}
        for level in sorted(levels.dropna().unique(), key=str):
            rows = X[levels == level]
            complete = rows[[self.energy_column, *self.nutrient_columns]].notna().all(axis=1).sum()
            self.level_rows_[level] = int(len(rows))
            if complete < MIN_LEVEL_ROWS:
                self.pooled_levels_.append(level)
                continue
            self.by_level_[level] = self._adjuster().fit(rows)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "pooled_")
        out = self.pooled_.transform(X)
        levels = X[self.strata]
        for level, adjuster in self.by_level_.items():
            mask = (levels == level).to_numpy()
            if mask.any():
                part = adjuster.transform(X.loc[mask])
                out.loc[mask, part.columns] = part
        if self.drop_strata:
            out = out.drop(columns=[self.strata])
        return out

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        check_is_fitted(self, "pooled_")
        names = [str(n) for n in self.pooled_.get_feature_names_out()]
        if self.drop_strata:
            names = [n for n in names if n != self.strata]
        return np.asarray(names, dtype=object)

    def lineage(self) -> list[dict[str, Any]]:
        """``{output, inputs, operation, formula[, params]}`` per output, as EnergyAdjuster's."""
        check_is_fitted(self, "pooled_")
        E, s = self.energy_column, self.strata
        entries = []
        for entry in self.pooled_.lineage():
            if self.drop_strata and entry["output"] == s:
                continue
            if entry["operation"] != "residual":
                entries.append(entry)
                continue
            n = entry["inputs"][0]
            per_level = {str(level): a.params_[n] for level, a in self.by_level_.items()}
            slopes = ", ".join(f"{level}: b = {_fmt(p['slope'])}, mean {E} {_fmt(p['reference_energy'])}"
                               for level, p in per_level.items())
            pooled = (f"; {', '.join(map(str, self.pooled_levels_))} use the pooled fit"
                      if self.pooled_levels_ else "")
            log = "log " if self.log_transform else ""
            formula = (f"{n}_adj = the residual of {log}{n} ~ {log}{E} within each level of {s}, plus "
                       f"the predicted {n} at that level's mean {E} ({slopes}{pooled})")
            entries.append({**entry, "inputs": [n, E, s], "formula": formula,
                            "params": {"strata": s, "levels": per_level,
                                       "pooled_levels": [str(v) for v in self.pooled_levels_],
                                       "pooled": entry.get("params")}})
        return entries

    def estimand(self) -> str:
        return METHOD_TABLE["residual"]["estimand"]


def energy_step(adjustment: Any, predictors: Sequence[str]) -> Any | None:
    """The energy-adjustment pipeline step for an ``EnergyAdjustment`` slot, or None for none.

    The strata column stratifies the residual method only; it leaves the output when it is not
    itself a predictor.
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


__all__ = ["StratifiedEnergyAdjuster", "energy_step"]

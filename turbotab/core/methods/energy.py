"""Energy adjustment: the five models of NUTRITION_PACK §04 as an in-fold sklearn step.

Source: ``docs/turbotab/research/NUTRITION_PACK.md`` §04, "The five models, formally".
Let N be a nutrient, E total energy, Y the outcome, C other covariates.

====================  =================================  ==================================
method                what the model sees                what a nutrient coefficient means
====================  =================================  ==================================
none                  N, E as given                      absolute intake (confounded by E)
standard              N + E                              substitution for the average of all
                                                         other energy sources
residual              N_adj  (E leaves the model)        the same substitution as standard
density_multivariate  N/E + E                            composition, energy as its own term
density               N/E    (E leaves the model)        rescaled relative effect; obscure
partition             kcal from N + kcal from other      addition, not substitution
====================  =================================  ==================================

Residual: ``N ~ E`` by OLS on the fitting rows (optionally both logged), then
``N_adj = residual + N_hat(mean E of the fitting rows)``, which simplifies to
``N_adj = N - b * (E - mean E)``. Under ``log_transform`` the regression is
``log N ~ log E`` and the adjusted value is back-transformed to N's units,
``N_adj = exp(log N - b * (log E - mean log E))``, i.e. N at the geometric mean
energy of the fitting rows.

**Everything learned here is learned in ``fit`` from the rows ``fit`` is given.**
Put this step inside the Pipeline that is cross-validated and the residual
regression sees training-fold rows only. That is the pack's own anti-pattern
("fitting the N ~ E residual regression on train+test before cross-validation").

What this module does not do yet: stratified residuals (the pack's default is
"within sex"); that needs a strata column and is a separate decision.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

try:  # Reuse the nutrition pack's constants and column recognition when present.
    from turbotab import nutrition as _nutrition
except Exception:  # pragma: no cover - a server deploy without Classic's ml/ package
    _nutrition = None

__all__ = [
    "EnergyMethod",
    "METHODS",
    "METHOD_TABLE",
    "EnergyAdjuster",
    "EnergyAdjustmentNotApplicable",
    "FactorReading",
    "applicable_methods",
    "default_atwater",
    "partition_refusal",
    "describe_method",
    "energy_factor",
    "nutrient_role",
    "unit_of",
]

EnergyMethod = Literal["none", "standard", "residual", "density_multivariate", "density", "partition"]
METHODS: Tuple[str, ...] = ("none", "standard", "residual", "density_multivariate", "density", "partition")

SOURCE = "docs/turbotab/research/NUTRITION_PACK.md#04 · Energy adjustment"
_TOMOVA = "Tomova et al. 2022, AJCN 115(1):189-198"
_PARTIAL = f"Only partially accounts for confounding by common dietary causes ({_TOMOVA})."

# The pack's table, one row per method. `estimand` is the plain-language sentence the
# UI prints next to a coefficient; `kind` is the short label for a forest-plot row.
METHOD_TABLE: Dict[str, Dict[str, Any]] = {
    "none": {
        "label": "No energy adjustment",
        "specification": "Y ~ N + C",
        "kind": "absolute",
        "estimand": (
            "Not energy-adjusted: a nutrient coefficient describes absolute intake, which "
            "total energy confounds through body size, physical activity, metabolic "
            "efficiency and reporting scale."),
        "standing": None,
        "caveats": [],
    },
    "standard": {
        "label": "Standard (multivariate) model",
        "specification": "Y ~ N + E + C",
        "kind": "substitution",
        "estimand": (
            "More of the nutrient with total energy held fixed, which is implicitly a "
            "substitution for the average of all other energy sources."),
        "standing": "CONVENTION",
        "caveats": [f"Biased even absent confounding: composite variable bias ({_TOMOVA}).",
                    _PARTIAL],
    },
    "residual": {
        "label": "Willett residual model",
        "specification": "N ~ E (OLS); N_adj = residual + N_hat(mean E); Y ~ N_adj + C",
        "kind": "substitution",
        "estimand": (
            "The same substitution as the standard model: more of the nutrient at the same "
            "total energy, in place of the average of all other energy sources, and not the "
            "effect of simply eating more of it."),
        "standing": "CONVENTION",
        "caveats": [f"Biased even absent confounding: composite variable bias ({_TOMOVA}).",
                    _PARTIAL],
    },
    "density_multivariate": {
        "label": "Multivariate nutrient density model",
        "specification": "Y ~ (N/E) + E + C",
        "kind": "composition",
        "estimand": (
            "Diet composition: the nutrient's amount per unit of energy, with total energy "
            "as a separate term."),
        "standing": "CONVENTION",
        "caveats": [],
    },
    "density": {
        "label": "Nutrient density alone",
        "specification": "Y ~ (N/E) + C",
        "kind": "obscure",
        "estimand": (
            "A rescaled relative effect of the nutrient's amount per unit of energy, whose "
            "interpretation is obscure without a total-energy term."),
        "standing": "CONVENTION (weakest)",
        "caveats": [f"Interpretation obscure ({_TOMOVA}).", _PARTIAL],
    },
    "partition": {
        "label": "Energy partition model",
        "specification": "Y ~ E_from_N + E_from_other + C (all kcal)",
        "kind": "addition",
        "estimand": (
            "The effect of adding calories from the nutrient while calories from every other "
            "source are held fixed, which is an addition and not a substitution."),
        "standing": "CONVENTION",
        "caveats": [
            "Estimates the total causal effect; unbiased only when there is no confounding "
            f"or all other nutrients have equal effects ({_TOMOVA}).",
            _PARTIAL],
    },
}


def describe_method(method: str) -> Dict[str, Any]:
    """The pack's row for one method, with its source, as a fresh dict."""
    _check_method(method)
    return {"method": method, **METHOD_TABLE[method], "source": SOURCE}


class EnergyAdjustmentNotApplicable(ValueError):
    """The chosen energy-adjustment model cannot be computed on these columns or rows."""

    def __init__(self, method: str, reason: str):
        super().__init__(f"{METHOD_TABLE.get(method, {}).get('label', method)} cannot be used: {reason}")
        self.method = method
        self.reason = reason


# ── Atwater factors and column recognition ───────────────────────────────────

_FALLBACK_ATWATER = {"protein": 4.0, "carbohydrate": 4.0, "fat": 9.0, "alcohol": 7.0, "fiber": 2.0}
KCAL_PER_KJ: float = float(getattr(_nutrition, "KCAL_PER_KJ", 4.184))

_FALLBACK_ROLE_PATTERNS = {
    "protein": r"prot",
    "carbohydrate": r"carb|cho\b",
    "fat": r"\bfat|lipid|tfat",
    "alcohol": r"alco|etoh",
}
_ROLE_PATTERNS: Dict[str, str] = {
    role: pattern
    for role, pattern in (getattr(_nutrition, "_NAME_PATTERNS", None) or _FALLBACK_ROLE_PATTERNS).items()
    if role != "energy"
}
_ROLE_PATTERNS["fiber"] = r"fib"
# Sugars and starch are carbohydrate at the same 4 kcal/g (NUTRITION_PACK §01): a `sugar` column
# carries energy and is adjusted with the rest. It is a part of total carbohydrate (nesting.py),
# so a partition never takes it beside its total.
_ROLE_PATTERNS["carbohydrate"] = f"(?:{_ROLE_PATTERNS['carbohydrate']})|sugar|starch"

# Unit suffixes, NUTRITION_PACK §01 signal 3. Density is tested first so that
# `protein_pct_kcal` is a share of energy and not an amount in kcal.
_DENSITY_SUFFIX = (r".*(_pct_energy|_pct_kcal|_percent_energy|_per1000kcal|"
                   r"_per_1000_kcal|_density)$")
_UNIT_SUFFIXES = (
    ("density", _DENSITY_SUFFIX),
    ("grams", r".*(_g|_gram|_grams)$"),
    ("kcal", r".*_kcal$"),
    ("kj", r".*_kj$"),
    ("milligrams", r".*_mg$"),
    ("micrograms", r".*(_mcg|_ug)$"),
    ("IU", r".*_iu$"),
)


def default_atwater() -> Dict[str, float]:
    """kcal per gram by nutrient role.

    Protein, carbohydrate, fat and alcohol come from ``turbotab.nutrition.ATWATER``
    (4/4/9/7, SETTLED in NUTRITION_PACK §01). Fiber at 2 kcal/g is the value "some
    systems" use (the pack's §01 wording), kept so fiber can be partitioned.
    """
    factors = dict(_FALLBACK_ATWATER)
    factors.update({k: float(v) for k, v in getattr(_nutrition, "ATWATER", {}).items()})
    return factors


def unit_of(column: str) -> str:
    """The unit a column name declares by its suffix: grams, kcal, kj, density, … or unmarked."""
    name = str(column).lower()
    for unit, pattern in _UNIT_SUFFIXES:
        if re.fullmatch(pattern, name):
            return unit
    return "unmarked"


def nutrient_role(column: str) -> Optional[str]:
    """Which energy-bearing macronutrient a column name refers to, or None.

    Uses the nutrition pack's name patterns (plus fiber), with ``_`` read as a word
    break so that ``total_fat_g`` and ``sat_fat_g`` are fat. Returns None when no
    role matches and raises ``ValueError`` when more than one does.
    """
    name = re.sub(r"[_\-.]+", " ", str(column))
    roles = [role for role, pattern in _ROLE_PATTERNS.items() if re.search(pattern, name, re.I)]
    if len(roles) > 1:
        raise ValueError(f"{column} matches more than one nutrient ({' and '.join(roles)})")
    return roles[0] if roles else None


@dataclass(frozen=True)
class FactorReading:
    """kcal per unit of one column, and where that number came from."""

    column: str
    factor: Optional[float]
    role: Optional[str]
    unit: str
    declared: bool  # the unit is stated (suffix or an explicit factor), not assumed
    reason: str


def energy_factor(column: str, atwater: Optional[Mapping[str, float]] = None) -> FactorReading:
    """kcal per unit of ``column``: the conversion both partition and substitution need.

    ``atwater`` entries override the defaults; a key may be a nutrient role
    (``"fat"``) or an exact column name (``"sfa_g"``). An exact column key is the
    user declaring that column's unit and factor, so it is accepted as given.
    """
    factors = default_atwater()
    if atwater:
        factors.update({str(k): float(v) for k, v in atwater.items()})
    unit = unit_of(column)
    if str(column) in factors:
        factor = factors[str(column)]
        return FactorReading(column, factor, None, unit, True,
                             f"{column}: {factor:g} kcal per unit, as given for this column")
    if unit == "density":
        return FactorReading(column, None, None, unit, True,
                             f"{column} is already a share of energy, not an amount that carries energy")
    if unit == "kcal":
        return FactorReading(column, 1.0, None, unit, True, f"{column} is already in kcal")
    if unit == "kj":
        return FactorReading(column, 1.0 / KCAL_PER_KJ, None, unit, True,
                             f"{column} is in kilojoules, converted at {KCAL_PER_KJ} kJ per kcal")
    try:
        role = nutrient_role(column)
    except ValueError as err:
        return FactorReading(column, None, None, unit, False, f"{err}, so its energy factor is ambiguous")
    if role is None or role not in factors:
        return FactorReading(column, None, None, unit, False,
                             f"{column} carries no energy: no Atwater factor is known for it")
    if unit in ("milligrams", "micrograms", "IU"):
        return FactorReading(column, None, role, unit, True,
                             f"{column} is in {unit}, and the Atwater factors are per gram")
    factor = factors[role]
    note = " (fiber at 2 kcal/g is the value some systems use)" if role == "fiber" else ""
    if unit == "grams":
        return FactorReading(column, factor, role, unit, True,
                             f"{column} is {role} in grams: {factor:g} kcal/g{note}")
    return FactorReading(column, factor, role, unit, False,
                         f"{column} reads as {role} but does not say it is in grams; "
                         f"at {factor:g} kcal/g that needs the Atwater reconstruction to confirm it{note}")


# ── Which methods can run on these columns ───────────────────────────────────

def _check_method(method: str) -> str:
    if method not in METHOD_TABLE:
        raise ValueError(f"Unknown energy-adjustment method {method!r}; expected one of {', '.join(METHODS)}")
    return method


def _and(items: Sequence[str]) -> str:
    items = list(items)
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


def _as_list(columns: Any) -> List[str]:
    if columns is None:
        return []
    if isinstance(columns, str):
        return [columns]
    return [str(c) for c in columns]


def applicable_methods(columns: Sequence[str], energy_column: Optional[str],
                       nutrient_columns: Sequence[str], *,
                       atwater: Optional[Mapping[str, float]] = None) -> Dict[str, Dict[str, Any]]:
    """For each method: ``{ok, reason}``, judged from column names alone.

    Partition also checks the data when it is fitted (the Atwater reconstruction),
    so ``ok`` here means "can be tried", and the fit can still refuse.
    """
    cols = [str(c) for c in columns]
    nutrients = _as_list(nutrient_columns)
    E = str(energy_column) if energy_column else None
    out: Dict[str, Dict[str, Any]] = {
        "none": {"ok": True, "reason": "Nutrients enter the model as absolute intakes; nothing is adjusted."}}

    common: Optional[str] = None
    if not E:
        common = ("No total-energy column has been chosen, and every energy adjustment is "
                  "computed against total energy intake.")
    elif E not in cols:
        common = f"The energy column {E} is not in this table."
    elif not nutrients:
        common = "No nutrient columns have been chosen to adjust."
    elif E in nutrients:
        common = f"{E} is the total-energy column and cannot also be a nutrient to adjust."
    elif len(set(nutrients)) != len(nutrients):
        twice = sorted({n for n in nutrients if nutrients.count(n) > 1})
        common = f"{_and(twice)} {'is' if len(twice) == 1 else 'are'} listed more than once."
    else:
        absent = [n for n in nutrients if n not in cols]
        if absent:
            common = f"These nutrient columns are not in this table: {', '.join(absent)}."
    if common:
        for method in METHODS[1:]:
            out[method] = {"ok": False, "reason": common}
        return out

    listed = _and(nutrients)
    out["standard"] = {"ok": True, "reason": f"{E} stays in the model beside {listed} (Y ~ N + E + C)."}

    densities = [n for n in nutrients if unit_of(n) == "density"]
    twice = None
    if densities:
        twice = (f"{_and(densities)} {'is' if len(densities) == 1 else 'are'} already expressed per "
                 f"unit of energy; adjusting again puts energy into the model twice.")
    out["residual"] = {"ok": not twice, "reason": twice or (
        f"Each of {listed} is regressed on {E} within the fitting rows; the model sees the "
        f"adjusted nutrients and not {E} itself.")}
    out["density_multivariate"] = {"ok": not twice, "reason": twice or (
        f"Each of {listed} is divided by {E}, and {E} stays in the model as its own term.")}
    out["density"] = {"ok": not twice, "reason": twice or (
        f"Each of {listed} is divided by {E}, and {E} leaves the model.")}

    readings = [energy_factor(n, atwater) for n in nutrients]
    refused = [r.reason for r in readings if r.factor is None]
    if unit_of(E) == "kj":
        refused.insert(0, f"{E} is in kilojoules, and the partition subtracts kcal from it")
    elif unit_of(E) not in ("kcal", "unmarked"):
        refused.insert(0, f"{E} does not hold energy in kcal")
    if refused:
        out["partition"] = {"ok": False, "reason": (
            "Energy partition splits total energy into kcal from each chosen nutrient and kcal "
            "from everything else, so every chosen nutrient must carry energy in a known unit. "
            + "; ".join(refused) + ".")}
    else:
        pending = [r.column for r in readings if not r.declared]
        tail = (f" {', '.join(pending)} will be read as grams only if the Atwater reconstruction "
                f"confirms it when the model is fitted." if pending else "")
        out["partition"] = {"ok": True, "reason": (
            f"{E} splits into kcal from {listed} and kcal from everything else; {E} itself "
            f"leaves the model.{tail}")}
    return out


PARTITION_NEGATIVE_SHARE = 0.10  # rows on which the nutrients out-weigh total energy …
PARTITION_SLACK = 0.05  # … by more than 5% of it (general Atwater factors' own error)


def partition_refusal(frame: pd.DataFrame, energy_column: str, nutrient_columns: Sequence[str], *,
                      nested: Optional[Mapping[str, str]] = None,
                      atwater: Optional[Mapping[str, float]] = None) -> Optional[Dict[str, Any]]:
    """Why the energy partition cannot be fitted on these columns and rows, or None when it can.

    The checks the fit itself makes (unit suffixes and the Atwater reconstruction on ``frame``),
    and two it cannot make on its own:

    * a total with its parts (``nested``: child -> parent, e.g. ``fat_sat`` -> ``fat_total``)
      counts the parts' energy twice;
    * nutrients that out-weigh total energy by more than :data:`PARTITION_SLACK` of it on more
      than :data:`PARTITION_NEGATIVE_SHARE` of the rows leave a negative "kcal from everything
      else", which reads as overlap or a wrong unit.

    Returns ``{"reason": str, "nutrients": list | None}``; ``nutrients`` is the totals-only list
    the partition can take instead (it passed every check), when nesting was the cause.
    """
    nutrients = _as_list(nutrient_columns)
    nested = dict(nested or {})
    parts = [n for n in nutrients if nested.get(n) in nutrients]
    if parts:
        parents = list(dict.fromkeys(nested[n] for n in parts))
        roles = []
        for p in parents:
            try:
                roles.append(nutrient_role(p) or p)
            except ValueError:
                roles.append(p)
        whose = _and([f"{r}'s" for r in dict.fromkeys(roles)])
        reason = (f"{_and(parts)} {'is a part' if len(parts) == 1 else 'are parts'} of "
                  f"{_and(parents)}, so a partition would count {whose} energy twice.")
        totals = [n for n in nutrients if n not in parts]
        alternative = totals if totals and _partition_problem(frame, energy_column, totals,
                                                              atwater) is None else None
        return {"reason": reason, "nutrients": alternative}
    problem = _partition_problem(frame, energy_column, nutrients, atwater)
    return None if problem is None else {"reason": problem, "nutrients": None}


def _partition_problem(frame: pd.DataFrame, energy_column: str, nutrients: Sequence[str],
                       atwater: Optional[Mapping[str, float]]) -> Optional[str]:
    columns = [c for c in dict.fromkeys([energy_column, *nutrients]) if c in frame.columns]
    verdict = applicable_methods(list(frame.columns), energy_column, nutrients, atwater=atwater)
    if not verdict["partition"]["ok"]:
        return str(verdict["partition"]["reason"])
    rows = frame[columns].apply(pd.to_numeric, errors="coerce").dropna()
    if len(rows) < 3:
        return None  # nothing to judge on: the fit decides
    step = EnergyAdjuster(method="partition", energy_column=energy_column,
                          nutrient_columns=list(nutrients), atwater=atwater)
    try:
        step.fit(rows)
    except EnergyAdjustmentNotApplicable as err:
        return err.reason[:1].upper() + err.reason[1:]
    except (ValueError, TypeError) as err:
        return str(err)
    e = rows[energy_column].to_numpy(dtype=float)
    other = e - sum(step.factors_[n] * rows[n].to_numpy(dtype=float) for n in nutrients)
    # General Atwater factors run a few percent off a food table's specific ones, so a slightly
    # negative remainder is factor error; well past it, the nutrients overlap or are not grams.
    below = int(np.sum(other < -PARTITION_SLACK * np.abs(e)))
    if len(e) and below / len(e) > PARTITION_NEGATIVE_SHARE:
        return (f"{_and(list(nutrients))} carry more energy than {energy_column} itself on "
                f"{below:,} of {len(e):,} rows, so they overlap or are not in grams.")
    return None


# ── The transformer ──────────────────────────────────────────────────────────

def _fmt(value: float) -> str:
    return np.format_float_positional(float(value), precision=6, unique=True, fractional=False, trim="-")


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    series = frame[column]
    if not (pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series)):
        raise TypeError(f"{column} must be numeric for energy adjustment; it has dtype {series.dtype}")
    return series.to_numpy(dtype=float, na_value=np.nan)


class EnergyAdjuster(TransformerMixin, BaseEstimator):
    """One of NUTRITION_PACK §04's five energy-adjustment models, as a Pipeline step.

    Parameters
    ----------
    method : "none" | "standard" | "residual" | "density_multivariate" | "density" | "partition"
    energy_column : the total-energy column.
    nutrient_columns : the nutrient columns to adjust. Every other column passes through.
    log_transform : residual method only: regress log N on log E and back-transform.
    atwater : kcal per gram by nutrient role or exact column name; overrides the defaults
        (``default_atwater()``). Used by the partition method.

    Missing values stay missing: the residual regression is fit on the rows where both
    the nutrient and energy are present, and an adjusted value is missing wherever
    either input is.
    """

    def __init__(self, method: EnergyMethod = "residual", energy_column: Optional[str] = "energy_kcal",
                 nutrient_columns: Sequence[str] = (), log_transform: bool = False,
                 atwater: Optional[Mapping[str, float]] = None):
        self.method = method
        self.energy_column = energy_column
        self.nutrient_columns = nutrient_columns
        self.log_transform = log_transform
        self.atwater = atwater

    # -- fitting -----------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: Any = None) -> "EnergyAdjuster":
        X = self._frame(X)
        method = _check_method(self.method)
        nutrients = _as_list(self.nutrient_columns)
        E = self.energy_column
        verdict = applicable_methods(list(X.columns), E, nutrients, atwater=self.atwater)[method]
        if not verdict["ok"]:
            raise EnergyAdjustmentNotApplicable(method, verdict["reason"])
        if self.log_transform and method != "residual":
            raise ValueError("log_transform applies to the residual method only "
                             f"(the chosen method is {method!r}).")

        self.feature_names_in_ = np.asarray([str(c) for c in X.columns], dtype=object)
        self.n_features_in_ = X.shape[1]
        self.n_fit_rows_ = int(len(X))
        self.method_ = method
        self.nutrients_ = nutrients if method != "none" else []
        self.params_: Dict[str, Dict[str, Any]] = {}
        self.factors_: Dict[str, float] = {}
        self.factor_notes_: Dict[str, str] = {}
        self.atwater_check_: Optional[Dict[str, Any]] = None

        if method != "none":
            e = _numeric(X, E)
            for n in nutrients:
                _numeric(X, n)
            if method in ("density", "density_multivariate"):
                self._check_positive_energy(e, "fitting rows")
            elif method == "residual":
                for n in nutrients:
                    self.params_[n] = self._fit_residual(_numeric(X, n), e, n)
            elif method == "partition":
                self._fit_partition(X, e)

        self.dropped_columns_ = [E] if method in ("residual", "density") else []
        self._plan = self._build_plan()
        names = [entry["output"] for entry in self._plan]
        clashes = sorted({name for name in names if names.count(name) > 1})
        if clashes:
            raise ValueError(f"The adjusted columns would collide with existing columns: {', '.join(clashes)}")
        return self

    def _fit_residual(self, n: np.ndarray, e: np.ndarray, name: str) -> Dict[str, Any]:
        E = self.energy_column
        rows = np.isfinite(n) & np.isfinite(e)
        if self.log_transform:
            bad = rows & ((n <= 0) | (e <= 0))
            if bad.any():
                which = name if (rows & (n <= 0)).any() else E
                raise ValueError(
                    f"log_transform needs positive values, and {which} is zero or negative in "
                    f"{int(bad.sum())} of the fitting rows. Adding a small constant to zeros "
                    f"changes the answer and would have to be reported; adjust {name} without "
                    f"the log, or decide how its zeros are handled first.")
            x, yv = np.log(e[rows]), np.log(n[rows])
        else:
            x, yv = e[rows], n[rows]
        if x.size < 3:
            raise ValueError(f"The residual regression of {name} on {E} needs at least 3 rows with "
                             f"both values present; the fitting rows have {x.size}.")
        xbar, ybar = float(x.mean()), float(yv.mean())
        dx, dy = x - xbar, yv - ybar
        sxx, syy = float(dx @ dx), float(dy @ dy)
        if sxx == 0.0:
            raise ValueError(f"{E} is constant in the fitting rows, so {name} cannot be regressed on it.")
        slope = float(dx @ dy) / sxx
        intercept = ybar - slope * xbar
        adjusted = yv - slope * dx
        da = adjusted - adjusted.mean()
        r_after = float(dx @ da) / np.sqrt(sxx * float(da @ da)) if float(da @ da) > 0 else 0.0
        return {
            "intercept": intercept,
            "slope": slope,
            "scale": "log" if self.log_transform else "linear",
            "reference_energy": float(np.exp(xbar)) if self.log_transform else xbar,
            "reference_kind": "geometric mean" if self.log_transform else "mean",
            "center": xbar,  # mean of the regressor on its own (possibly log) scale
            "constant_added": intercept + slope * xbar,  # N_hat at the reference energy
            "r2": 1.0 - float((dy - slope * dx) @ (dy - slope * dx)) / syy if syy > 0 else float("nan"),
            "r_before": float(dx @ dy) / np.sqrt(sxx * syy) if syy > 0 else float("nan"),
            "r_after": r_after,
            "n_fit": int(x.size),
        }

    def _fit_partition(self, X: pd.DataFrame, e: np.ndarray) -> None:
        E = self.energy_column
        readings = {n: energy_factor(n, self.atwater) for n in self.nutrients_}
        check = _atwater_reading(X, E)
        # The reconstruction 4P + 4C + 9F (+ 7A) only means something when protein,
        # carbohydrate and fat are all there: without one of them the ratio drifts with
        # diet composition and reads as a multi-source merge that is not there.
        complete = check is not None and set(_RECONSTRUCTION_ROLES) <= set(check.macro_columns)
        if check is not None:
            self.atwater_check_ = {"verdict": check.verdict, "ratio": check.ratio,
                                   "columns": dict(check.macro_columns), "sentence": check.sentence,
                                   "used": complete}
        if complete:
            if check.verdict in ("energy_in_kj", "energy_inverse", "mixed_units"):
                raise EnergyAdjustmentNotApplicable(
                    "partition", f"the partition subtracts kcal from {E}, and the Atwater "
                                 f"reconstruction on the fitting rows says: {check.sentence}")
            overlap = [n for n in self.nutrients_ if n in check.macro_columns.values()]
            if check.verdict == "macros_not_grams" and overlap:
                raise EnergyAdjustmentNotApplicable(
                    "partition", f"{_and(overlap)} would be converted as grams, and the "
                                 f"Atwater reconstruction says: {check.sentence}")
        for n, reading in readings.items():
            if not reading.declared:
                confirmed = complete and check.verdict == "pass" and n in check.macro_columns.values()
                if not confirmed:
                    why = (check.sentence if complete else
                           "the table does not carry protein, carbohydrate and fat columns "
                           "for the reconstruction to check.")
                    raise EnergyAdjustmentNotApplicable(
                        "partition", f"{n} does not say what unit it is in, and the Atwater "
                                     f"reconstruction could not confirm it is grams: {why} If it "
                                     f"is in grams, a _g suffix on its name says so.")
            self.factors_[n] = float(reading.factor)
            self.factor_notes_[n] = reading.reason if reading.declared else (
                f"{n} is {reading.role}; the Atwater reconstruction on the fitting rows confirms "
                f"grams, so {reading.factor:g} kcal/g")
        other = e - sum(self.factors_[n] * _numeric(X, n) for n in self.nutrients_)
        with np.errstate(divide="ignore", invalid="ignore"):
            share = other / np.abs(e)
        finite = share[np.isfinite(share)]
        self.params_["__other__"] = {
            "rows_below_zero": int(np.sum(other < 0)),
            # Below zero by more than general Atwater factors' own error: overlap or a wrong unit.
            "rows_well_below_zero": int(np.sum(other < -PARTITION_SLACK * np.abs(e))),
            "median_share_of_energy": float(np.median(finite)) if finite.size else None,
            "n_fit": int(np.isfinite(other).sum())}

    def _check_positive_energy(self, e: np.ndarray, where: str) -> None:
        bad = np.isfinite(e) & (e <= 0)
        if bad.any():
            raise ValueError(
                f"{self.energy_column} is zero or negative in {int(bad.sum())} of the {where}, "
                f"where a nutrient density N/E is undefined; implausible intakes need to be "
                f"excluded first (NUTRITION_PACK §02).")

    # -- the output plan (names, inputs, formulas) --------------------------

    def _build_plan(self) -> List[Dict[str, Any]]:
        method, E = self.method_, self.energy_column
        nutrients = set(self.nutrients_)
        plan: List[Dict[str, Any]] = []
        for col in self.feature_names_in_:
            col = str(col)
            if method == "none" or (col not in nutrients and col != E):
                plan.append({"output": col, "inputs": [col], "operation": "pass-through",
                             "formula": f"{col} (unchanged)"})
            elif method == "standard":
                plan.append({"output": col, "inputs": [col], "operation": "kept", "formula": (
                    f"{col} (unchanged; total energy stays in the model as a covariate)" if col == E else
                    f"{col} (unchanged; enters the model beside {E}: Y ~ N + E + C)")})
            elif col == E:
                if method == "density_multivariate":
                    plan.append({"output": col, "inputs": [col], "operation": "kept", "formula": (
                        f"{col} (unchanged; total energy enters as its own term: Y ~ N/E + E + C)")})
                elif method == "partition":
                    terms = " + ".join(f"{_fmt(self.factors_[n])} × {n}" for n in self.nutrients_)
                    plan.append({"output": "kcal_from_other", "inputs": [E, *self.nutrients_],
                                 "operation": "partition-other",
                                 "formula": f"kcal_from_other = {E} − ({terms})",
                                 "params": {**self.params_["__other__"],
                                            "atwater_check": self.atwater_check_}})
                # residual and density: energy leaves the model (see dropped_columns_)
            elif method == "residual":
                plan.append(self._residual_entry(col))
            elif method in ("density", "density_multivariate"):
                plan.append({"output": f"{col}_per_{E}", "inputs": [col, E], "operation": "density",
                             "formula": f"{col}_per_{E} = {col} / {E}"})
            elif method == "partition":
                plan.append({"output": f"kcal_from_{col}", "inputs": [col], "operation": "partition",
                             "formula": f"kcal_from_{col} = {_fmt(self.factors_[col])} × {col}  "
                                        f"({self.factor_notes_[col]})",
                             "params": {"kcal_per_unit": self.factors_[col]}})
        return plan

    def _residual_entry(self, n: str) -> Dict[str, Any]:
        E, p = self.energy_column, self.params_[n]
        b, c = _fmt(p["slope"]), _fmt(p["center"])
        if p["scale"] == "log":
            formula = (f"{n}_adj = exp(log {n} − {b} × (log {E} − {c})): the residual of "
                       f"log {n} ~ log {E} (OLS on {p['n_fit']} fitting rows) plus the predicted "
                       f"log {n} at their mean log {E}, back-transformed to {n}'s units "
                       f"(reference {E} = geometric mean {_fmt(p['reference_energy'])})")
        else:
            formula = (f"{n}_adj = {n} − {b} × ({E} − {c}): the residual of {n} ~ {E} (OLS on "
                       f"{p['n_fit']} fitting rows) plus the predicted {n} at their mean {E} "
                       f"({_fmt(p['constant_added'])} added)")
        return {"output": f"{n}_adj", "inputs": [n, E], "operation": "residual",
                "formula": formula, "params": dict(p)}

    # -- transforming --------------------------------------------------------

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "method_")
        X = self._frame(X)
        expected = [str(c) for c in self.feature_names_in_]
        missing = [c for c in expected if c not in X.columns]
        extra = [str(c) for c in X.columns if str(c) not in expected]
        if missing or extra:
            parts = []
            if missing:
                parts.append(f"missing {', '.join(missing)}")
            if extra:
                parts.append(f"not seen in fit: {', '.join(extra)}")
            raise ValueError(f"transform needs the columns fit saw ({'; '.join(parts)}).")

        E = self.energy_column
        e = _numeric(X, E) if self.method_ != "none" else None
        if self.method_ in ("density", "density_multivariate"):
            self._check_positive_energy(e, "rows being transformed")
        out: Dict[str, Any] = {}
        for entry in self._plan:
            op, name = entry["operation"], entry["output"]
            if op in ("pass-through", "kept"):
                out[name] = X[entry["inputs"][0]]
            elif op == "residual":
                out[name] = self._apply_residual(_numeric(X, entry["inputs"][0]), e, entry["inputs"][0])
            elif op == "density":
                out[name] = _numeric(X, entry["inputs"][0]) / e
            elif op == "partition":
                n = entry["inputs"][0]
                out[name] = self.factors_[n] * _numeric(X, n)
            elif op == "partition-other":
                out[name] = e - sum(self.factors_[n] * _numeric(X, n) for n in self.nutrients_)
        return pd.DataFrame(out, index=X.index)

    def _apply_residual(self, n: np.ndarray, e: np.ndarray, name: str) -> np.ndarray:
        p = self.params_[name]
        if p["scale"] == "log":
            bad = (np.isfinite(n) & (n <= 0)) | (np.isfinite(e) & (e <= 0))
            if bad.any():
                raise ValueError(f"log_transform needs positive values, and {name} or "
                                 f"{self.energy_column} is zero or negative in {int(bad.sum())} rows "
                                 f"being transformed.")
            return np.exp(np.log(n) - p["slope"] * (np.log(e) - p["center"]))
        return n - p["slope"] * (e - p["center"])

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        check_is_fitted(self, "method_")
        return np.asarray([entry["output"] for entry in self._plan], dtype=object)

    # -- what the UI shows ----------------------------------------------------

    def lineage(self) -> List[Dict[str, Any]]:
        """How each output column was produced: ``{output, inputs, operation, formula[, params]}``."""
        check_is_fitted(self, "method_")
        return [{**entry, "inputs": list(entry["inputs"]),
                 **({"params": dict(entry["params"])} if "params" in entry else {})}
                for entry in self._plan]

    def estimand(self) -> str:
        """What a nutrient coefficient means under this method, in one sentence (from the pack)."""
        return METHOD_TABLE[_check_method(self.method)]["estimand"]

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _frame(X: Any) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("EnergyAdjuster needs a pandas DataFrame with named columns, "
                            f"not {type(X).__name__}.")
        return X


_RECONSTRUCTION_ROLES = ("protein", "carbohydrate", "fat")


def _atwater_reading(X: pd.DataFrame, energy_column: str) -> Any:
    """The nutrition pack's Atwater reconstruction on these rows, if it reads this energy column."""
    if _nutrition is None:
        return None
    try:
        reading = _nutrition.atwater(X)
    except Exception:  # pragma: no cover - a diagnostic that cannot run is not a verdict
        return None
    if reading is None or reading.energy_column != energy_column:
        return None
    return reading

"""Energy adjustment: the models of NUTRITION_PACK §04 as an in-fold sklearn step.

Source: ``docs/turbotab/research/NUTRITION_PACK.md`` §04, "The five models, formally", with the
two forms of the residual method (McCullough & Byrd 2023) and the all-components model (Tomova et
al. 2022) beside them. Let N be a nutrient, E total energy, Y the outcome, C other covariates.

=======================  ==============================  =====================================
method                   what the model sees             what a nutrient coefficient means
=======================  ==============================  =====================================
none                     N  (E leaves the model)         absolute intake (confounded by E)
standard                 N + E                           substitution: for the average of all
                                                         other sources with one nutrient; for
                                                         the sources not in the model with more
residual                 N_adj + E                       the standard model's, identically
residual_energy_dropped  N_adj  (E leaves the model)     the standard model's only when no
                                                         covariate correlates with E
density_multivariate     N/E + E                         an obscure quantity (Tomova 2022)
density                  N/E    (E leaves the model)     rescaled relative effect; obscure
partition                kcal from N + kcal from other   addition, not substitution
all_components           kcal from every source          addition per source; the average
                                                         relative effect by a weighted contrast
=======================  ==============================  =====================================

The label equals the model fitted: :func:`describe_model` composes the estimand and each
coefficient's meaning from the columns the fitted model actually sees (which energy sources are
in it, which are left out, a total beside its own parts), never from the method's name alone
(audit ME-02, ME-03, ME-04, ME-15). Ruling 1 (BLUEPRINT §12): the residual method keeps total
energy in the outcome model by default; the energy-dropped form is offered under its own name and
estimand.

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

Within strata (the pack's "within sex"), :class:`StratifiedEnergyAdjuster` fits
``N ~ E`` within each level of a strata column and adds back **one** constant for
every level: the predicted nutrient at the mean energy of all fitting rows
(NUTRITION_PACK §04, "with the predicted nutrient at the cohort mean energy added
back"). Each level's own constant would give the adjusted nutrient the levels'
differences by construction, and a model without the strata column would read
them as the nutrient's effect (audit MA-02).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

try:  # Reuse the nutrition pack's constants and column recognition when present.
    from turbotab import nutrition as _nutrition
except Exception:  # pragma: no cover - a server deploy without Classic's ml/ package
    _nutrition = None

__all__ = [
    "ENERGY_SOURCES",
    "EnergyMethod",
    "EnergyTerm",
    "LOG_ESTIMAND",
    "MAX_OMITTED_SHARE",
    "METHODS",
    "METHOD_TABLE",
    "ModelEstimand",
    "OMITTED_ROW_SHARE",
    "omitted_sentence",
    "PARTITION_METHODS",
    "RANKING",
    "RESIDUAL_METHODS",
    "TENSION",
    "EnergyAdjuster",
    "EnergyAdjustmentNotApplicable",
    "FactorReading",
    "MIN_LEVEL_ROWS",
    "StratifiedEnergyAdjuster",
    "applicable_methods",
    "coefficient_gap",
    "default_atwater",
    "describe_model",
    "energy_terms",
    "fiber_beside_carbohydrate",
    "omitted_energy",
    "partition_refusal",
    "reads_as_total_energy",
    "total_energy_columns",
    "describe_method",
    "energy_factor",
    "nutrient_role",
    "rank_methods",
    "relative_effect_rows",
    "unit_of",
]

EnergyMethod = Literal["none", "standard", "residual", "residual_energy_dropped",
                       "density_multivariate", "density", "partition", "all_components"]
METHODS: Tuple[str, ...] = ("none", "standard", "residual", "residual_energy_dropped",
                            "density_multivariate", "density", "partition", "all_components")
RESIDUAL_METHODS: Tuple[str, ...] = ("residual", "residual_energy_dropped")
PARTITION_METHODS: Tuple[str, ...] = ("partition", "all_components")

SOURCE = "docs/turbotab/research/NUTRITION_PACK.md#04 · Energy adjustment"
_TOMOVA = "Tomova et al. 2022, AJCN 115(1):189-198"
_MCCULLOUGH = "McCullough & Byrd 2023, AJE 192(11):1801-1805"
_DISPUTE = "Willett, Stampfer & Tobias 2022, AJCN 116(2):608-609"
_PARTIAL = f"Only partially accounts for confounding by common dietary causes ({_TOMOVA})."
# Tomova et al. 2022, Discussion: "this strategy does introduce a trade-off between minimizing bias
# (by including the largest number of components at the finest level of detail) and maximizing
# precision (by having to estimate many parameters, i.e., 1 for each additional dietary component)."
_PRECISION = (f"Trades precision for bias: one term per energy source ({_TOMOVA}).")

# The energy sources a diet's total is made of, by Atwater factor (NUTRITION_PACK §01). Whatever
# total energy holds beyond the sources in a model is named "other" (fiber, polyols, organic acids,
# food-table rounding, and any source the table has no column for).
ENERGY_SOURCES: Tuple[str, ...] = ("protein", "carbohydrate", "fat", "alcohol")

# The pack's table, one row per method. ``estimand`` is the base sentence (one energy-bearing
# nutrient; :func:`describe_model` composes the sentence for the model actually fitted); ``kind``
# is the short label for a forest-plot row; ``customary`` and ``sound`` are north star 5's two
# labels (AUDIT_REPORT §3.1): where the field uses it, and why it is or is not sound per purpose.
METHOD_TABLE: Dict[str, Dict[str, Any]] = {
    "none": {
        "label": "No energy adjustment",
        "specification": "Y ~ N + C (total energy leaves the model)",
        "kind": "absolute",
        "estimand": (
            "Not energy-adjusted: total energy is not in the model, so a nutrient coefficient "
            "describes absolute intake, which total energy confounds through body size, physical "
            "activity, metabolic efficiency and reporting scale."),
        "standing": None,
        "caveats": [],
        "customary": "As the unadjusted model beside an adjusted one (NUTRITION_PACK §04).",
        "sound": {"inference": "As a stated crude or sensitivity model, beside an adjusted one.",
                  "prediction": "Rarely: it drops total energy, often the strongest predictor."},
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
        "customary": "Yes (NUTRITION_PACK §04; Tomova et al. 2022).",
        "sound": {"inference": "Acceptable with the omitted energy sources named.",
                  "prediction": "Good: keeps total energy."},
    },
    "residual": {
        "label": "Willett residual model, total energy kept",
        "specification": "N ~ E (OLS); N_adj = residual + N_hat(mean E); Y ~ N_adj + E + C",
        "kind": "substitution",
        "estimand": (
            "The standard model's substitution, with the nutrient in its own units: the "
            "residual and total energy span the same model as the nutrient and total energy, "
            "so the coefficient is the standard model's exactly (the Willett–Stampfer variant "
            f"with a term for total energy; {_MCCULLOUGH}): more of the nutrient at the same "
            "total energy, in place of the average of all other energy sources."),
        "standing": "CONVENTION",
        "caveats": [f"Biased even absent confounding: composite variable bias ({_TOMOVA}).",
                    _PARTIAL],
        "customary": f"Yes ({_MCCULLOUGH}).",
        "sound": {"inference": "The sound form of the residual method.",
                  "prediction": "Fine: keeps total energy."},
    },
    "residual_energy_dropped": {
        "label": "Willett residual model, total energy left out",
        "specification": "N ~ E (OLS); N_adj = residual + N_hat(mean E); Y ~ N_adj + C",
        "kind": "substitution only without energy-correlated covariates",
        "estimand": (
            "Total energy is not in the outcome model: the coefficient equals the standard "
            "model's substitution only when no other covariate correlates with energy; "
            "otherwise it is a different number, which can differ in sign, and its interval "
            "is wider."),
        "standing": "CONVENTION",
        "caveats": [f"Biased even absent confounding: composite variable bias ({_TOMOVA}).",
                    _PARTIAL],
        "customary": "Yes, the field default (NUTRITION_PACK §04).",
        "sound": {"inference": "Avoid in this form: covariates that track energy change it.",
                  "prediction": "Avoid: discards total energy's signal."},
    },
    "density_multivariate": {
        "label": "Multivariate nutrient density model",
        "specification": "Y ~ (N/E) + E + C",
        "kind": "obscure",
        "estimand": (
            "The nutrient's amount per unit of energy, with total energy as a separate term. "
            "Its coefficient is not clean diet composition: Tomova et al. (2022) find the "
            "density model's coefficient \"an obscure quantity that conflates both the effect of "
            "the nutrient exposure and that of the reciprocal of total energy\", and with total "
            "energy added \"a more accurate estimate than the (unadjusted) nutrient density model, "
            "but one which is still biased\"; the all-components model is the paper's recommended "
            "route."),
        "standing": "CONVENTION",
        # Audit IN-21: Tomova et al. 2022, model 3b, "returns a more accurate estimate than the
        # (unadjusted) nutrient density model, but one which is still biased"; the paper
        # recommends "the all-components model as the more intuitive and transparent option".
        "caveats": [f"Interpretation obscure ({_TOMOVA}).",
                    f"Still biased even without confounding; the all-components model is the "
                    f"source's recommended route ({_TOMOVA}).",
                    _PARTIAL],
        "customary": "Yes (NUTRITION_PACK §04).",
        "sound": {"inference": "Below standard and all components, with its caveat.",
                  "prediction": "Fine: keeps total energy."},
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
        "customary": "Yes, the weakest (NUTRITION_PACK §04).",
        "sound": {"inference": "Avoid: an obscure estimand, severely biased (Tomova 2022).",
                  "prediction": "Rank low: drops total energy."},
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
        "customary": "Yes (NUTRITION_PACK §04).",
        "sound": {"inference": "For total-effect questions.",
                  "prediction": "Fine: carries total energy in its parts."},
    },
    "all_components": {
        "label": "All-components model",
        "specification": ("Y ~ kcal from each energy source (protein, carbohydrate, fat, alcohol) "
                          "+ kcal from other + C"),
        "kind": "addition and relative effect",
        "estimand": (
            "Every energy source is its own term in kcal. A source's coefficient is the total "
            "effect of adding its calories with every other source fixed; its average relative "
            "effect, the coefficient less the other sources' coefficients weighted by their "
            "share of the remaining energy, is the substitution for the average of the others "
            f"({_TOMOVA}: the \"all-components model\")."),
        "standing": "EMERGING (disputed)",
        "caveats": [_PRECISION, f"Its use is disputed ({_DISPUTE})."],
        "customary": f"Emerging ({_TOMOVA}); disputed ({_DISPUTE}).",
        "sound": {"inference": "Sound for total and average relative effects, at a precision cost.",
                  "prediction": "Information-equivalent to keeping total energy."},
    },
}

# The log variant (``log_transform``) of each residual form estimates something else again: it
# regresses log N on log E and rescales the nutrient to the geometric-mean energy, N × (E/G)^(−b).
LOG_ESTIMAND: Dict[str, str] = {
    "residual": (
        "An energy-elasticity adjustment: log N is regressed on log E and the nutrient is "
        "rescaled to the geometric-mean energy, N × (E/G)^(−b). With total energy also in the "
        "model, the coefficient is per unit of that rescaled nutrient at fixed energy; it is not "
        "the linear residual's or the standard model's coefficient, because the rescaling "
        "differs with each row's energy."),
    "residual_energy_dropped": (
        "An energy-elasticity adjustment with total energy not in the outcome model: log N is "
        "regressed on log E and the nutrient rescaled to the geometric-mean energy, "
        "N × (E/G)^(−b). With an elasticity b near 1 that is close to a nutrient density "
        "(N/E, rescaled), so the coefficient reads like the density model's rescaled relative "
        "effect, not the linear residual's substitution."),
}

# Soundness for the declared purpose orders the menu (north star 5; AUDIT_REPORT §3.1 and §4).
# Ruling 2 (BLUEPRINT §12): under inference the all-components model ranks first for substitution
# questions; under prediction the energy-model choice matters little among models that keep energy.
RANKING: Dict[str, Tuple[str, ...]] = {
    "inference": ("all_components", "standard", "residual", "partition", "density_multivariate",
                  "none", "residual_energy_dropped", "density"),
    "prediction": ("standard", "residual", "density_multivariate", "partition", "all_components",
                   "none", "residual_energy_dropped", "density"),
}
TENSION: Dict[str, str] = {
    "inference": (
        "All components ranks first for substitution questions: no composite-variable bias "
        "(Tomova 2022), at a precision cost of one term per source; disputed by Willett, "
        "Stampfer and Tobias (2022)."),
    "prediction": (
        "For prediction the energy model matters little: models that keep total energy predict "
        "alike, and dropping energy discards its signal."),
}

# The mean share of total energy a model's energy sources may leave unaccounted before a
# substitution under inference is blocked and recorded (audit ME-05). Below it the remainder is
# within the general Atwater factors' own error (PARTITION_SLACK, a few percent of total energy);
# above it a real energy source is missing from the model. A stated threshold, not a sourced one.
MAX_OMITTED_SHARE = 0.05
OMITTED_ROW_SHARE = 0.10  # a row's remainder counted as large (the audit's NHANES reading)


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
    if not items:
        return ""
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
        "none": {"ok": True, "reason": (
            f"{E} leaves the model, and nutrients enter as absolute intakes." if E and E in cols
            else "Nutrients enter the model as absolute intakes; nothing is adjusted.")}}

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
        f"adjusted nutrients with {E} beside them.")}
    out["residual_energy_dropped"] = {"ok": not twice, "reason": twice or (
        f"Each of {listed} is regressed on {E} within the fitting rows; {E} then leaves the "
        f"outcome model.")}
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
    out["all_components"] = _all_components_verdict(cols, E, nutrients, out["partition"], atwater)
    return out


def _all_components_verdict(columns: Sequence[str], E: str, nutrients: Sequence[str],
                            partition: Mapping[str, Any],
                            atwater: Optional[Mapping[str, float]]) -> Dict[str, Any]:
    """The all-components model is a partition over every energy source the table holds: each of
    protein, carbohydrate and fat its own term, and alcohol too when the table has it (Tomova et
    al. 2022: "simultaneously adjusting for all dietary components")."""
    if not partition["ok"]:
        return {"ok": False, "reason": str(partition["reason"]).replace(
            "Energy partition splits", "The all-components model splits", 1)}
    covered = {_role_or_none(n) for n in nutrients}
    missing = [role for role in ENERGY_SOURCES[:3] if role not in covered]
    if missing:
        return {"ok": False, "reason": (
            f"The all-components model gives every energy source its own term, and "
            f"{_and(missing)} {'is' if len(missing) == 1 else 'are'} not among the chosen "
            f"nutrients.")}
    left = [c for c in columns if c not in nutrients and c != E and _role_or_none(c) == "alcohol"
            and energy_factor(c, atwater).factor is not None]
    if left:
        return {"ok": False, "reason": (
            f"The all-components model gives every energy source its own term, and {_and(left)} "
            f"carries alcohol's energy but is not among the chosen nutrients.")}
    listed = _and(list(nutrients))
    return {"ok": True, "reason": (
        f"{E} splits into kcal from {listed}, each its own term, and kcal from everything else; "
        f"each nutrient's average relative effect is computed from them.")}


def _role_or_none(column: str) -> Optional[str]:
    try:
        return nutrient_role(column)
    except ValueError:
        return None


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
    """One of the energy-adjustment models (NUTRITION_PACK §04 and beside it), as a Pipeline step.

    Parameters
    ----------
    method : one of :data:`METHODS`.
    energy_column : the total-energy column.
    nutrient_columns : the nutrient columns to adjust. Every other column passes through.
    log_transform : the residual methods only: regress log N on log E and back-transform.
    atwater : kcal per gram by nutrient role or exact column name; overrides the defaults
        (``default_atwater()``). Used by the partition and all-components methods.
    leave_out : "none" only: further total-energy columns (the energy role) that leave the model
        with ``energy_column``. Under "none" total energy is not in the model at all: that is
        what "no energy adjustment" means (audit ME-02).

    Which columns leave the model: total energy under "none", "residual_energy_dropped" and
    "density"; under "partition" and "all_components" it is replaced by kcal from everything
    else. "residual" keeps total energy beside the adjusted nutrients (the Willett–Stampfer
    variant; McCullough & Byrd 2023), so its coefficient is the standard model's.

    Missing values stay missing: the residual regression is fit on the rows where both
    the nutrient and energy are present, and an adjusted value is missing wherever
    either input is.
    """

    def __init__(self, method: EnergyMethod = "residual", energy_column: Optional[str] = "energy_kcal",
                 nutrient_columns: Sequence[str] = (), log_transform: bool = False,
                 atwater: Optional[Mapping[str, float]] = None, leave_out: Sequence[str] = ()):
        self.method = method
        self.energy_column = energy_column
        self.nutrient_columns = nutrient_columns
        self.log_transform = log_transform
        self.atwater = atwater
        self.leave_out = leave_out

    # -- fitting -----------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: Any = None) -> "EnergyAdjuster":
        X = self._frame(X)
        method = _check_method(self.method)
        nutrients = _as_list(self.nutrient_columns)
        E = self.energy_column
        verdict = applicable_methods(list(X.columns), E, nutrients, atwater=self.atwater)[method]
        if not verdict["ok"]:
            raise EnergyAdjustmentNotApplicable(method, verdict["reason"])
        if self.log_transform and method not in RESIDUAL_METHODS:
            raise ValueError("log_transform applies to the residual methods only "
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
        self.other_term_ = True  # partition methods: whether kcal_from_other is a term

        if method != "none":
            e = _numeric(X, E)
            for n in nutrients:
                _numeric(X, n)
            if method in ("density", "density_multivariate"):
                self._check_positive_energy(e, "fitting rows")
            elif method in RESIDUAL_METHODS:
                for n in nutrients:
                    self.params_[n] = self._fit_residual(_numeric(X, n), e, n)
            elif method in PARTITION_METHODS:
                self._fit_partition(X, e)
                # The all-components model: when the named sources make up total energy exactly
                # (a recall's energy computed from them, or a simulation), kcal from everything
                # else is zero on every row, a term with nothing in it; it leaves the model.
                if method == "all_components":
                    other = e - sum(self.factors_[n] * _numeric(X, n) for n in self.nutrients_)
                    finite = np.isfinite(other) & np.isfinite(e)
                    scale = float(np.nanmax(np.abs(e[finite]))) if finite.any() else 0.0
                    self.other_term_ = not (finite.any() and
                                            float(np.max(np.abs(other[finite]))) <= 1e-9 * max(scale, 1.0))

        if method == "none":
            self.dropped_columns_ = [c for c in dict.fromkeys([E, *_as_list(self.leave_out)])
                                     if c and c in X.columns]
        elif method in ("residual_energy_dropped", "density"):
            self.dropped_columns_ = [E]
        else:
            self.dropped_columns_ = []
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
                    self.method_, f"the partition subtracts kcal from {E}, and the Atwater "
                                 f"reconstruction on the fitting rows says: {check.sentence}")
            overlap = [n for n in self.nutrients_ if n in check.macro_columns.values()]
            if check.verdict == "macros_not_grams" and overlap:
                raise EnergyAdjustmentNotApplicable(
                    self.method_, f"{_and(overlap)} would be converted as grams, and the "
                                 f"Atwater reconstruction says: {check.sentence}")
        for n, reading in readings.items():
            if not reading.declared:
                confirmed = complete and check.verdict == "pass" and n in check.macro_columns.values()
                if not confirmed:
                    why = (check.sentence if complete else
                           "the table does not carry protein, carbohydrate and fat columns "
                           "for the reconstruction to check.")
                    raise EnergyAdjustmentNotApplicable(
                        self.method_, f"{n} does not say what unit it is in, and the Atwater "
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
            if method == "none":
                if col not in self.dropped_columns_:  # total energy leaves the model (ME-02)
                    plan.append({"output": col, "inputs": [col], "operation": "pass-through",
                                 "formula": f"{col} (unchanged)"})
            elif col not in nutrients and col != E:
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
                elif method == "residual":
                    plan.append({"output": col, "inputs": [col], "operation": "kept", "formula": (
                        f"{col} (unchanged; total energy stays in the outcome model beside the "
                        f"adjusted nutrients: Y ~ N_adj + E + C)")})
                elif method in PARTITION_METHODS and self.other_term_:
                    terms = " + ".join(f"{_fmt(self.factors_[n])} × {n}" for n in self.nutrients_)
                    plan.append({"output": "kcal_from_other", "inputs": [E, *self.nutrients_],
                                 "operation": "partition-other",
                                 "formula": f"kcal_from_other = {E} − ({terms})",
                                 "params": {**self.params_["__other__"],
                                            "atwater_check": self.atwater_check_}})
                # residual_energy_dropped and density: energy leaves the model (dropped_columns_);
                # all-components whose sources make up total energy exactly: no other term.
            elif method in RESIDUAL_METHODS:
                plan.append(self._residual_entry(col))
            elif method in ("density", "density_multivariate"):
                plan.append({"output": f"{col}_per_{E}", "inputs": [col, E], "operation": "density",
                             "formula": f"{col}_per_{E} = {col} / {E}"})
            elif method in PARTITION_METHODS:
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
        """What a nutrient coefficient means under this method with one adjusted nutrient. The
        log variant of a residual form has its own (:data:`LOG_ESTIMAND`). For the model a design
        actually fits, several nutrients and all, :func:`describe_model` composes the sentence."""
        method = _check_method(self.method)
        if self.log_transform and method in LOG_ESTIMAND:
            return LOG_ESTIMAND[method]
        return METHOD_TABLE[method]["estimand"]

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _frame(X: Any) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("EnergyAdjuster needs a pandas DataFrame with named columns, "
                            f"not {type(X).__name__}.")
        return X


# ── Within strata (NUTRITION_PACK §04: "within sex") ─────────────────────────

MIN_LEVEL_ROWS = 30
"""Fitting rows (nutrient and energy both present) a level needs for a slope of its own.

Below it, the level's residual uses the pooled slope, still centered on the level's own means,
and the lineage says so. A stated floor, not a sourced one: the audit's proposal (AUDIT_REPORT
§5, WP3) for a slope that a handful of rows cannot pin down.
"""


class _BlankLevel:
    """The level of a row whose strata value is missing: a level of its own, like any other.

    It pickles by reference and equals any copy of itself, so a fitted step reloaded from the
    stage cache still finds its blank rows.
    """

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _BlankLevel)

    def __hash__(self) -> int:
        return hash(_BlankLevel.__name__)

    def __repr__(self) -> str:
        return "(blank)"

    __str__ = __repr__

    def __reduce__(self) -> str:
        return "BLANK_LEVEL"


BLANK_LEVEL = _BlankLevel()


def _pearson(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """Pearson's r, or None when it is undefined (under 3 rows, or either side constant)."""
    if a.size < 3:
        return None
    da, db = a - a.mean(), b - b.mean()
    saa, sbb = float(da @ da), float(db @ db)
    if saa <= 0.0 or sbb <= 0.0:
        return None
    return float(da @ db) / float(np.sqrt(saa * sbb))


def _correlation_ratio(values: np.ndarray, codes: np.ndarray) -> Optional[float]:
    """η, the correlation of ``values`` with a category: sqrt(between-level SS / total SS).

    With two levels it is the size of the point-biserial r with either level's indicator. None
    when it is undefined (fewer than two levels, or ``values`` constant).
    """
    groups = np.unique(codes)
    if groups.size < 2 or values.size < 3:
        return None
    grand = float(values.mean())
    total = float(((values - grand) ** 2).sum())
    if total <= 0.0:
        return None
    between = sum(float((codes == g).sum()) * (float(values[codes == g].mean()) - grand) ** 2
                  for g in groups)
    return float(np.sqrt(between / total))


def _level_fit(v: np.ndarray, e: np.ndarray, pooled_slope: float, log: bool) -> Optional[Dict[str, Any]]:
    """One level's residual regression ``v ~ e`` (both on the regression's scale), or None
    when the level has no row with both present."""
    rows = np.isfinite(v) & np.isfinite(e)
    m = int(rows.sum())
    if m == 0:
        return None
    x, yv = e[rows], v[rows]
    xbar, ybar = float(x.mean()), float(yv.mean())
    dx, dy = x - xbar, yv - ybar
    sxx, syy = float(dx @ dx), float(dy @ dy)
    if m < MIN_LEVEL_ROWS:
        slope, source, reason = float(pooled_slope), "pooled", f"fewer than {MIN_LEVEL_ROWS} fitting rows"
    elif sxx == 0.0:
        slope, source, reason = float(pooled_slope), "pooled", "energy is constant in this level"
    else:
        slope, source, reason = float(dx @ dy) / sxx, "level", None
    return {
        "slope": slope,
        "slope_from": source,
        "reason": reason,
        "scale": "log" if log else "linear",
        "center": xbar,  # the level's mean of the regressor, on its own (possibly log) scale
        "level_mean": ybar,  # the level's mean nutrient: what its residual is centered on
        "reference_energy": float(np.exp(xbar)) if log else xbar,
        "reference_kind": "geometric mean" if log else "mean",
        "r_before": float(dx @ dy) / float(np.sqrt(sxx * syy)) if sxx > 0 and syy > 0 else None,
        "n_fit": m,
    }


def _r3(value: Optional[float]) -> str:
    if value is None or not np.isfinite(value):
        return "undefined"
    return "0.000" if abs(value) < 0.0005 else f"{value:.3f}".replace("-", "−")


def _slope(value: float) -> str:
    return np.format_float_positional(float(value), precision=4, unique=False, fractional=False,
                                      trim="-")


class StratifiedEnergyAdjuster(TransformerMixin, BaseEstimator):
    """The residual method within each level of ``strata``, with one constant for every level.

    For a nutrient N and energy E (both logged under ``log_transform``), in level s::

        N_adj = N − b_s × (E − Ē_s) − N̄_s + N̄

    ``b_s``, ``Ē_s`` and ``N̄_s`` are level s's slope and means over its fitting rows. ``N̄`` is
    the predicted nutrient at the mean energy of **all** fitting rows, the pooled regression's
    ``N̂(Ē)``, which is the nutrient's mean over them (NUTRITION_PACK §04: "with the predicted
    nutrient at the cohort mean energy added back"). Each level's residual is uncorrelated with
    energy and averages zero, so on the fitting rows N_adj is uncorrelated with energy and with
    the strata column, pooled as well as within levels, whether or not the strata column is a
    predictor. (Adding each level's own ``N̄_s`` instead hands N_adj the levels' differences, and
    a model without the strata column reads them as the nutrient's effect: audit MA-02.)

    * A level with fewer than :data:`MIN_LEVEL_ROWS` fitting rows, or with constant energy, uses
      the pooled slope ``b``, still centered on its own means.
    * Rows whose strata value is missing are a level of their own, :data:`BLANK_LEVEL`.
    * A row from a level never seen in fit uses the pooled regression, ``N − b × (E − Ē)``.

    ``drop_strata``: the strata column is an input only (it is not a predictor), so it leaves
    the output. Every other column passes through as the pooled :class:`EnergyAdjuster` passes
    it. ``lineage()`` reports each level's slope and rows, the levels on the pooled slope, and
    r(N_adj, E) and r(N_adj, strata) on the fitting rows, pooled and per level.

    ``method``: "residual" keeps total energy in the outcome model beside the adjusted nutrients
    (the default, BLUEPRINT §12 ruling 1); "residual_energy_dropped" lets it leave.
    """

    def __init__(self, energy_column: str = "energy_kcal", nutrient_columns: Sequence[str] = (),
                 strata: str = "sex", log_transform: bool = False, drop_strata: bool = False,
                 atwater: Optional[Mapping[str, float]] = None, method: str = "residual"):
        self.energy_column = energy_column
        self.nutrient_columns = nutrient_columns
        self.strata = strata
        self.log_transform = log_transform
        self.drop_strata = drop_strata
        self.atwater = atwater
        self.method = method

    # -- fitting -----------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: Any = None) -> "StratifiedEnergyAdjuster":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("StratifiedEnergyAdjuster needs a pandas DataFrame with named columns.")
        if self.strata not in X.columns:
            raise ValueError(f"The strata column {self.strata} is not among the inputs.")
        self.feature_names_in_ = np.asarray([str(c) for c in X.columns], dtype=object)
        self.n_features_in_ = X.shape[1]
        self.nutrients_ = _as_list(self.nutrient_columns)
        if self.method not in RESIDUAL_METHODS:
            raise ValueError(f"Strata stratify the residual methods only, not {self.method!r}.")
        self.pooled_ = EnergyAdjuster(self.method, self.energy_column, self.nutrients_,
                                      log_transform=self.log_transform, atwater=self.atwater).fit(X)
        strata = X[self.strata]
        levels: List[Any] = sorted(pd.unique(strata.dropna().to_numpy(dtype=object)), key=str)
        if strata.isna().any():
            levels.append(BLANK_LEVEL)
        self.level_index_: Dict[Any, int] = {level: i for i, level in enumerate(levels)}
        codes = self._codes(X)
        e = self._scaled(_numeric(X, self.energy_column))
        self.levels_: Dict[str, Dict[Any, Dict[str, Any]]] = {}
        for n in self.nutrients_:
            v = self._scaled(_numeric(X, n))
            pooled_slope = self.pooled_.params_[n]["slope"]
            fits = {level: _level_fit(v[codes == i], e[codes == i], pooled_slope, self.log_transform)
                    for level, i in self.level_index_.items()}
            self.levels_[n] = {level: fit for level, fit in fits.items() if fit is not None}
        self.check_ = self._check(X, codes)
        return self

    def _scaled(self, values: np.ndarray) -> np.ndarray:
        if not self.log_transform:
            return values
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log(values)

    def _codes(self, X: pd.DataFrame) -> np.ndarray:
        """Each row's level as its index in ``level_index_``; -1 for a level fit never saw."""
        values = pd.Series(X[self.strata].to_numpy(dtype=object))
        codes = values.map(self.level_index_).to_numpy(dtype=float, na_value=np.nan)
        codes[values.isna().to_numpy()] = self.level_index_.get(BLANK_LEVEL, -1)
        codes[np.isnan(codes)] = -1
        return codes.astype(np.int64)

    def _check(self, X: pd.DataFrame, codes: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """r(N_adj, E) and r(N_adj, strata) on the fitting rows, as the model sees N_adj.

        Pooled: Pearson's r with energy, and the correlation ratio η with the strata column.
        Per level: r with energy within the level, and r with the level's indicator over every
        row (within one level the strata column is constant, so it has no r of its own there).
        """
        out = self.transform(X)
        e = _numeric(X, self.energy_column)
        check: Dict[str, Dict[str, Any]] = {}
        for n in self.nutrients_:
            adj = out[f"{n}_adj"].to_numpy(dtype=float, na_value=np.nan)
            ok = np.isfinite(adj) & np.isfinite(e) & (codes >= 0)
            per = {}
            for level, i in self.level_index_.items():
                inside = ok & (codes == i)
                per[str(level)] = {
                    "rows": int(inside.sum()),
                    "r_adj_energy": _pearson(adj[inside], e[inside]),
                    "r_adj_indicator": _pearson(adj[ok], (codes[ok] == i).astype(float)),
                }
            check[n] = {
                "pooled": {"rows": int(ok.sum()),
                           "r_adj_energy": _pearson(adj[ok], e[ok]),
                           "r_adj_strata": _correlation_ratio(adj[ok], codes[ok]),
                           "strata_measure": "correlation ratio (η)"},
                "levels": per,
            }
        return check

    # -- transforming --------------------------------------------------------

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "pooled_")
        out = self.pooled_.transform(X)  # the pooled regression for every row, to start
        codes = self._codes(X)
        e = self._scaled(_numeric(X, self.energy_column))
        for n in self.nutrients_:
            constant = float(self.pooled_.params_[n]["constant_added"])
            v = self._scaled(_numeric(X, n))
            column = out[f"{n}_adj"].to_numpy(dtype=float, na_value=np.nan).copy()
            for level, p in self.levels_[n].items():
                rows = codes == self.level_index_[level]
                if rows.any():
                    adjusted = v[rows] - p["slope"] * (e[rows] - p["center"]) - p["level_mean"] + constant
                    column[rows] = np.exp(adjusted) if self.log_transform else adjusted
            out[f"{n}_adj"] = column
        if self.drop_strata:
            out = out.drop(columns=[self.strata])
        return out

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        check_is_fitted(self, "pooled_")
        names = [str(n) for n in self.pooled_.get_feature_names_out()]
        if self.drop_strata:
            names = [n for n in names if n != self.strata]
        return np.asarray(names, dtype=object)

    # -- what the UI shows ----------------------------------------------------

    def lineage(self) -> List[Dict[str, Any]]:
        """``{output, inputs, operation, formula[, params]}`` per output, as EnergyAdjuster's."""
        check_is_fitted(self, "pooled_")
        entries = []
        for entry in self.pooled_.lineage():
            if self.drop_strata and entry["output"] == self.strata:
                continue
            if entry["operation"] != "residual":
                entries.append(entry)
                continue
            n = entry["inputs"][0]
            entries.append({**entry, "inputs": [n, self.energy_column, self.strata],
                            "formula": self._formula(n), "params": self._params(n, entry.get("params"))})
        return entries

    def _params(self, n: str, pooled: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        fits = self.levels_[n]
        return {
            "strata": self.strata,
            "constant": float(self.pooled_.params_[n]["constant_added"]),
            "constant_is": "the predicted nutrient at the mean energy of all fitting rows",
            "min_level_rows": MIN_LEVEL_ROWS,
            "levels": {str(level): dict(p) for level, p in fits.items()},
            "pooled_slope_levels": [str(level) for level, p in fits.items() if p["slope_from"] == "pooled"],
            "check": self.check_[n],
            "pooled": dict(pooled) if pooled else None,
        }

    def _formula(self, n: str) -> str:
        E, s = self.energy_column, self.strata
        pooled = self.pooled_.params_[n]
        constant = _fmt(pooled["constant_added"])
        lg = "log " if self.log_transform else ""
        if self.log_transform:
            head = (f"{n}_adj = exp(log {n} − b_s × (log {E} − mean log {E}_s) − mean log {n}_s "
                    f"+ {constant})")
            # Audit IN-25: under log the reference is the geometric mean, not the arithmetic one.
            back = (f", back-transformed to {n}'s units: {n} at the geometric-mean {E} "
                    f"({_fmt(pooled['reference_energy'])})")
        else:
            head = f"{n}_adj = {n} − b_s × ({E} − mean {E}_s) − mean {n}_s + {constant}"
            back = ""
        check = self.check_[n]
        slopes = []
        for level, p in self.levels_[n].items():
            if p["slope_from"] == "level":
                slopes.append(f"{level} b = {_slope(p['slope'])} ({p['n_fit']:,} rows)")
            else:
                slopes.append(f"{level} the pooled slope b = {_slope(p['slope'])} on its own means "
                              f"({p['n_fit']:,} rows; {p['reason']})")
        within = ", ".join(f"{_r3(c['r_adj_energy'])} within {level}" for level, c in check["levels"].items())
        whole = check["pooled"]
        return (f"{head}: the residual of {lg}{n} ~ {lg}{E} within each level s of {s}, plus one "
                f"constant for every level, the predicted {lg}{n} at the mean {lg}{E} of all "
                f"{pooled['n_fit']:,} fitting rows{back}. Slopes: {'; '.join(slopes)}. On the "
                f"fitting rows, r({n}_adj, {E}) is {_r3(whole['r_adj_energy'])} pooled, {within}; "
                f"r({n}_adj, {s}) is {_r3(whole['r_adj_strata'])} pooled (correlation ratio).")

    def notes(self, rows_word: str = "training rows") -> List[str]:
        """Plain statements for the design's warnings: the levels on the pooled slope.
        ``rows_word`` names the rows the step was fit on (every analyzed row under inference)."""
        check_is_fitted(self, "pooled_")
        by_level: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for n, fits in self.levels_.items():
            for level, p in fits.items():
                if p["slope_from"] == "pooled":
                    entry = by_level.setdefault((str(level), p["reason"]), {"rows": 0, "nutrients": []})
                    entry["rows"] = max(entry["rows"], p["n_fit"])
                    entry["nutrients"].append(n)
        out = []
        for (level, reason), entry in by_level.items():
            who = _and(entry["nutrients"])
            verb = "is" if len(entry["nutrients"]) == 1 else "are"
            rows = entry["rows"]
            why = (f"{self.energy_column} is constant in level {level} of {self.strata}"
                   if reason.startswith("energy") else
                   f"level {level} of {self.strata} has {rows:,} "
                   f"{rows_word if rows != 1 else rows_word[:-1]}, "
                   f"fewer than the {MIN_LEVEL_ROWS} a residual regression of its own needs")
            out.append(f"{why[0].upper()}{why[1:]}, so {who} {verb} adjusted there with the pooled "
                       f"slope, centered on the level's own means.")
        return out

    def estimand(self) -> str:
        method = self.method if self.method in RESIDUAL_METHODS else "residual"
        if self.log_transform:
            return LOG_ESTIMAND[method]
        return METHOD_TABLE[method]["estimand"]


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


# ── What the fitted model estimates (audit ME-02, ME-03, ME-04, ME-14, ME-15) ────


@dataclass(frozen=True)
class EnergyTerm:
    """A predictor that carries a diet's energy: its column, its source, its kcal per unit.

    ``share``: the column is the source's percent of total energy (``fat_pct_kcal``), so it
    carries ``value/100`` of each row's own energy and has no constant kcal per unit (``factor``
    is NaN)."""

    column: str
    source: str  # protein | carbohydrate | fat | alcohol | fiber, else the column's own name
    factor: float
    share: bool = False


# A column named like total energy (``energy_kcal``, ``DR1TKCAL``, ``total_kj``): the proposals'
# recognizer (``stages.proposals.is_energy_name``), less what names an expenditure or a target.
_TOTAL_ENERGY_NAME = re.compile(r"kcal|(?:^|[^a-z])kj(?:$|[^a-z])|energy|calor", re.I)
_NOT_INTAKE = re.compile(r"expend|\btee\b|\bree\b|\bbmr\b|\bpal\b|burn|basal|requirement|"
                         r"goal|target|^kcal from\b", re.I)


def reads_as_total_energy(column: str) -> bool:
    """Whether ``column``'s name reads as a total energy intake (audit ME-02: the estimand is read
    off the fitted matrix, so total energy in it is recognized whatever its role).

    Not a macronutrient's amount or share (``protein_kcal``, ``fat_pct_kcal``), and not an energy
    expenditure, requirement or basal rate."""
    name = str(column)
    if not _TOTAL_ENERGY_NAME.search(name) or unit_of(name) == "density":
        return False
    if _NOT_INTAKE.search(re.sub(r"[_\-.]+", " ", name)):
        return False
    try:
        return nutrient_role(name) is None
    except ValueError:
        return False


def total_energy_columns(predictors: Sequence[str], roles: Mapping[str, str]) -> List[str]:
    """The predictors that are total energy: the energy role's, then any exposure or covariate
    whose name reads as total energy intake (:func:`reads_as_total_energy`)."""
    named = [str(c) for c in predictors if roles.get(c) == "energy"]
    by_name = [str(c) for c in predictors if roles.get(c) in ("exposure", "covariate")
               and str(c) not in named and reads_as_total_energy(c)]
    return named + by_name


def _is_share(column: str) -> bool:
    from turbotab.core.methods.percent_energy import is_percent_of_energy

    return is_percent_of_energy(column)


def energy_terms(predictors: Sequence[str], roles: Mapping[str, str],
                 energy_columns: Sequence[str] = (),
                 atwater: Optional[Mapping[str, float]] = None) -> List[EnergyTerm]:
    """The predictors that carry energy in a known unit, in the order given.

    Exposures with an energy factor, and covariates whose name states their unit (``fat_g``): a
    covariate's name alone (``body_fat``) is not read as an energy source. An exposure in percent
    of energy (``fat_pct_kcal``) carries its share of each row's energy (``share``; audit B24).
    Total-energy columns (``energy_columns`` and the energy role) are not sources.
    """
    skip = set(_as_list(energy_columns))
    out: List[EnergyTerm] = []
    for c in predictors:
        role = roles.get(c)
        if c in skip or role not in ("exposure", "covariate"):
            continue
        if role == "exposure" and _is_share(c):
            out.append(EnergyTerm(str(c), _role_or_none(c) or str(c), float("nan"), share=True))
            continue
        reading = energy_factor(c, atwater)
        if reading.factor is None or (role == "covariate" and not reading.declared):
            continue
        out.append(EnergyTerm(str(c), _role_or_none(c) or str(c), float(reading.factor)))
    return out


def _omitted(sources: Sequence[str]) -> List[str]:
    """The energy sources not among ``sources``, then "other" (always: what total energy holds
    beyond any listed source)."""
    return [s for s in ENERGY_SOURCES if s not in sources] + ["other"]


def _in_place_of(named: bool, omitted: Sequence[str]) -> str:
    if not named:
        return "in place of the average of all other energy sources"
    return f"in place of {_and([*omitted[:-1], 'other energy'])}"


@dataclass
class ModelEstimand:
    """What the fitted model estimates, composed from the columns it sees.

    ``form`` is the energy model actually fitted (a method's name never overrides its matrix);
    ``text`` the estimand sentence (None when the table has no energy question); ``terms`` each
    model-matrix column's meaning, for its coefficient row; ``omitted`` the energy sources not in
    the model ("other" last); ``sources`` those in it.
    """

    form: str
    text: Optional[str]
    terms: Dict[str, str] = field(default_factory=dict)
    omitted: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    energy_in_model: bool = False
    nested: Dict[str, List[str]] = field(default_factory=dict)  # total -> its parts, both in it


def _form(method: Optional[str], energy_in: bool) -> str:
    if method in (None, "none", "standard"):
        return "standard" if energy_in else "none"
    if method in RESIDUAL_METHODS:
        return "residual" if energy_in else "residual_energy_dropped"
    return str(method)


def _matrix_name(column: str, form: str, adjusted: Sequence[str], E: Optional[str]) -> str:
    """The model-matrix column an energy step makes of a raw column (EnergyAdjuster's names)."""
    if column not in adjusted:
        return column
    if form in RESIDUAL_METHODS:
        return f"{column}_adj"
    if form in ("density", "density_multivariate"):
        return f"{column}_per_{E}"
    if form in PARTITION_METHODS:
        return f"kcal_from_{column}"
    return column


def _slopes(step: Any) -> Dict[str, float]:
    pooled = getattr(step, "pooled_", step)
    params = getattr(pooled, "params_", {}) or {}
    return {n: float(p["slope"]) for n, p in params.items()
            if isinstance(p, Mapping) and p.get("scale") == "log"}


def describe_model(adjustment: Any, predictors: Sequence[str], roles: Mapping[str, str],
                   matrix_columns: Sequence[str], *, nested: Optional[Mapping[str, str]] = None,
                   step: Any = None, atwater: Optional[Mapping[str, float]] = None) -> ModelEstimand:
    """The estimand of the model a design fits, and each coefficient's meaning, from its matrix.

    ``adjustment`` is the recorded ``EnergyAdjustment`` (or None: unanswered); ``predictors`` and
    ``roles`` the design's; ``matrix_columns`` the columns the model sees after the shared steps;
    ``nested`` child -> parent among the predictors; ``step`` the fitted energy step (for the log
    variant's fitted elasticity).

    * Whether total energy is in the model is read off the matrix, so "no adjustment" with energy
      still in it would read as the standard model (it no longer can: ME-02), and a residual whose
      energy left reads as the energy-dropped form (ME-03).
    * With two or more energy sources in the model, each coefficient is a substitution for the
      sources not in it, which are listed (NUTRITION_PACK §05a: "Each coefficient is the effect of
      substituting that component for the omitted one. Name the omitted component in your
      results."), not for "the average of all other sources" (ME-04).
    * A total beside its own parts estimates the remainder not in any part (ME-15).
    """
    roles = dict(roles or {})
    matrix = [str(c) for c in matrix_columns]
    in_matrix = set(matrix)
    method = getattr(adjustment, "method", None)
    # Total energy by its role, or by its name when it sits in the model as a covariate or an
    # exposure (ME-02's recommendation: the estimand comes from the fitted matrix, not the method's
    # name nor the role alone, so an energy column kept as a covariate reads as the standard model).
    energy_cols = total_energy_columns(predictors, roles)
    E = getattr(adjustment, "energy_column", None) or (energy_cols[0] if energy_cols else None)
    if E and E not in energy_cols:
        energy_cols = [E, *energy_cols]
    terms = energy_terms(predictors, roles, energy_cols, atwater)
    energy_in = [c for c in energy_cols if c in in_matrix]
    if method is None and not terms:
        # No nutrient carries energy: there is no energy estimand to state.
        return ModelEstimand(form="standard" if energy_in else "none", text=None,
                             energy_in_model=bool(energy_in))
    form = _form(method, bool(energy_in))
    adjusted = (list(getattr(adjustment, "nutrients", None) or [])
                if form not in ("none", "standard") else [])
    log = bool(getattr(adjustment, "log_transform", False)) and form in RESIDUAL_METHODS

    present = [(t, _matrix_name(t.column, form, adjusted, E)) for t in terms]
    present = [(t, m) for t, m in present if m in in_matrix]
    raw_present = {t.column for t, _ in present}
    groups: Dict[str, List[str]] = {}
    for child, parent in dict(nested or {}).items():
        if child in raw_present and parent in raw_present:
            groups.setdefault(parent, []).append(child)
    children = {c for kids in groups.values() for c in kids}
    # The main sources first (protein, carbohydrate, fat, alcohol), then any other energy-bearing
    # source the model holds (fiber at 2 kcal/g): each is "in the model" and not left to "other".
    held = list(dict.fromkeys(t.source for t, _ in present))
    sources = [s for s in ENERGY_SOURCES if s in held] + [s for s in held if s not in ENERGY_SOURCES]
    omitted = _omitted(sources)

    # Two energy-bearing terms that are not one another's parts: each coefficient is then a swap
    # for the sources left out of the model, not for the average of all the others (ME-04).
    named = len([t for t, _ in present if t.column not in children]) >= 2
    swap = _in_place_of(named, omitted)
    who = (_and(sources) if len(sources) >= 2 else
           _and([t.column for t, _ in present if t.column not in children]))
    clause = (f"with {who} all in the model, each coefficient is a substitution in place "
              f"of the energy sources not in the model: {', '.join(omitted)}." if named else "")
    if log:
        text = LOG_ESTIMAND[form]
        slopes = _slopes(step)
        if slopes:
            text += " Fitted elasticities: " + ", ".join(
                f"b = {_slope(b)} for {n}" for n, b in slopes.items()) + "."
    elif form == "standard" and named:
        text = ("More of each nutrient with total energy held fixed: " + clause +
                " It is not a swap for the average of all other sources (NUTRITION_PACK §05a: "
                "name the omitted component).")
    elif form == "residual" and named:
        text = ("The standard model's substitution, in each nutrient's own units: the residuals "
                "and total energy span the same model as the nutrients and total energy, so each "
                f"coefficient is the standard model's exactly ({_MCCULLOUGH}). "
                + clause[0].upper() + clause[1:])
    else:
        text = METHOD_TABLE[form]["estimand"]
        if form == "residual_energy_dropped" and named:
            text += f" Where it does equal the standard model, {clause}"
    if form in PARTITION_METHODS:
        if "kcal_from_other" in in_matrix:
            text += (f" kcal_from_other is one term for the energy sources not named: "
                     f"{', '.join(omitted)}.")
        elif form == "all_components":
            text += (" The named sources make up total energy on every row, so there is no "
                     "other term.")
        if form == "all_components":
            text += (" Each nutrient's average relative effect is reported beside its "
                     "coefficient, per unit of the nutrient.")
    shares = [t.column for t, _ in present if t.share]
    if shares and not energy_in:
        text = ("Nutrients enter as shares of energy with total energy not in the model: a share "
                "is a nutrient density, so each coefficient mixes a swap between energy sources "
                f"with any effect of total energy, an obscure quantity ({_TOMOVA}).")
    elif shares:
        # The field's model in percent of energy leaves exactly one source out, the reference
        # (Hu et al. 1997; NUTRITION_PACK §05): the estimand names it, and says so when the model
        # is not that one (the methods gate, item C).
        left = [s for s in omitted if s != "other"]
        text += (f" {_and(shares)} {'is a share' if len(shares) == 1 else 'are shares'} of energy: "
                 "a coefficient is per percentage point of energy.")
        if len(left) == 1:
            text += (f" With every source but {left[0]} in the model, plus total energy, it is the "
                     f"field's leave-one-out model (Hu et al. 1997, NEJM 337:1491), with {left[0]} "
                     f"as the reference: each coefficient is a point of energy from its source in "
                     f"place of {left[0]}.")
        elif left:
            text += (f" {_and(left)} are left out, so no one source is the reference: each "
                     f"coefficient is a point of energy in place of their mix, not the field's "
                     f"leave-one-out model (Hu et al. 1997, NEJM 337:1491), which leaves out one.")
        else:
            text += (" Every named source is in the model, so the reference is the energy from no "
                     "named source; where the shares sum to 100% there is none, and only "
                     "differences between their coefficients are estimable. The field's "
                     "leave-one-out model leaves one source out as the reference (Hu et al. 1997, "
                     "NEJM 337:1491).")
    for parent, kids in groups.items():
        source = next((t.source for t, _ in present if t.column == parent), parent)
        text += (f" {parent} sits beside its own parts {_and(kids)}, so its coefficient is "
                 f"the remaining {source} (holding {', '.join(kids)} fixed): the {source} in "
                 f"none of them, not total {source}.")

    meanings: Dict[str, str] = {}
    parent_of = {c: p for p, kids in groups.items() for c in kids}
    for t, m in present:
        if t.share:
            # One percentage point of energy from the source (audit B24): with total energy in the
            # model it comes from the sources not in it; without, energy is not held fixed.
            if energy_in and form not in PARTITION_METHODS:
                meanings[m] = f"one point of energy from {t.source} {swap}, total energy fixed"
            else:
                meanings[m] = (f"{t.source}'s share of energy; total energy not in the model "
                               f"(a nutrient density)")
            continue
        if t.column in parent_of:
            meanings[m] = (f"{t.column} in place of the rest of {parent_of[t.column]} "
                           f"({parent_of[t.column]} fixed)")
            continue
        what = (f"remaining {t.source} (holding {', '.join(groups[t.column])} fixed)"
                if t.column in groups else t.source)
        if form in ("standard", "residual"):
            meanings[m] = f"{what} {swap}, total energy fixed"
        elif form == "residual_energy_dropped":
            meanings[m] = f"energy-adjusted {what}; total energy not in the outcome model"
        elif form == "none":
            meanings[m] = f"absolute intake of {what}; total energy not in the model"
        elif form == "density_multivariate":
            meanings[m] = f"{what} per kcal, total energy fixed (an obscure quantity: Tomova 2022)"
        elif form == "density":
            meanings[m] = f"{what} per kcal; total energy not in the model (obscure)"
        else:
            meanings[m] = f"adding kcal from {what}, every other source fixed (total effect)"
    rest = _and([*omitted[:-1], "other energy"])
    only_shares = bool(present) and all(t.share for t, _ in present)
    for c in energy_in:
        if only_shares and form in ("standard", "residual"):
            # Every share fixed, more total energy scales every source alike (audit B24).
            meanings[c] = "more total energy with each nutrient's share of it fixed"
        elif form == "standard" and sources:
            meanings[c] = (f"more energy from {rest} (the sources not in the model), the named "
                           f"nutrients fixed")
        elif form == "residual":
            meanings[c] = ("more energy with each adjusted nutrient fixed: the nutrients rise with "
                           "it as they do on average")
        elif form == "density_multivariate":
            meanings[c] = "more energy with each nutrient's share of it fixed"
    if "kcal_from_other" in in_matrix:
        meanings["kcal_from_other"] = f"adding kcal from {rest}, every named source fixed"
    return ModelEstimand(form=form, text=text, terms=meanings, omitted=omitted, sources=sources,
                         energy_in_model=bool(energy_in), nested=groups)


def omitted_energy(frame: pd.DataFrame, energy_column: Optional[str], columns: Sequence[str], *,
                   nested: Optional[Mapping[str, str]] = None,
                   atwater: Optional[Mapping[str, float]] = None) -> Optional[Dict[str, Any]]:
    """How much of total energy the model's energy-bearing columns leave out, on ``frame``'s rows.

    Each row's remainder is ``(E − Σ kcal per unit × amount) / E − Σ share / 100`` over
    ``columns`` that carry energy, as an amount or as a percent of energy (``fat_pct_kcal``, audit
    B24), a part beside its own total counted once, through the total. Returns the sources in the
    model, the ones left out ("other" last), the mean remainder, and the rows whose remainder
    exceeds :data:`OMITTED_ROW_SHARE`; None without a usable energy column. (Shares alone do not
    read it: their remainder is ``1 − Σ share / 100`` on every row.)
    """
    if not energy_column or energy_column not in frame.columns:
        return None
    nested = dict(nested or {})
    cols = [str(c) for c in columns if c in frame.columns and c != energy_column]
    shares = [c for c in cols if _is_share(c) and nested.get(c) not in cols]
    readings = {c: energy_factor(c, atwater) for c in cols if c not in shares}
    amounts = [c for c in readings if readings[c].factor is not None and nested.get(c) not in cols]
    carriers = [c for c in cols if c in amounts or c in shares]
    n = len(frame)
    e = pd.to_numeric(frame[energy_column], errors="coerce").to_numpy(dtype=float)
    named = np.zeros(n)
    for c in amounts:
        named = named + float(readings[c].factor) * pd.to_numeric(
            frame[c], errors="coerce").to_numpy(dtype=float)
    in_shares = np.zeros(n)
    for c in shares:
        in_shares = in_shares + pd.to_numeric(frame[c], errors="coerce").to_numpy(dtype=float) / 100.0
    if amounts:
        ok = np.isfinite(e) & np.isfinite(named) & np.isfinite(in_shares) & (e > 0)
        share = (e[ok] - named[ok]) / e[ok] - in_shares[ok]
    else:
        ok = np.isfinite(in_shares)
        share = 1.0 - in_shares[ok]
    sources = list(dict.fromkeys(r for r in (_role_or_none(c) for c in carriers)
                                 if r in ENERGY_SOURCES))
    return {
        "energy_column": str(energy_column),
        "columns": carriers,
        "sources": sources,
        "omitted": _omitted(sources),
        "mean_share": float(share.mean()) if share.size else None,
        "rows_over": int(np.sum(share > OMITTED_ROW_SHARE)),
        "n_rows": int(share.size),
    }


def omitted_sentence(reading: Mapping[str, Any]) -> Optional[str]:
    """The concern a substitution states when energy sources are missing from the model (ME-05)."""
    share = reading.get("mean_share")
    if share is None:
        return None
    omitted = list(reading["omitted"])
    named = _and([*omitted[:-1], "other energy"])
    held = _and(list(reading["columns"])) if reading["columns"] else "no energy source"
    over = int(reading.get("rows_over") or 0)
    rows = (f", and more than {OMITTED_ROW_SHARE:.0%} on {over:,} of {int(reading['n_rows']):,} rows"
            if over else "")
    return (f"The model holds {held}; {named} make up the rest of total energy, "
            f"{share:.0%} of it on average{rows}. Total energy carries them as one composite, so "
            f"the curve carries their confounding (Tomova, Gilthorpe & Tennant 2022: \"wherever ≥2 "
            f"components are involved in the substitution, there is scope for composite variable "
            f"bias unless the individual effects are estimated\"); add the remaining sources to "
            f"the model to remove it.")


def fiber_beside_carbohydrate(columns: Sequence[str]) -> Optional[str]:
    """A warning when fiber and total carbohydrate are both energy-bearing predictors (B19).

    Carbohydrate "by difference" (USDA's, and the NHANES total) already holds fiber, so fiber's
    energy then counts twice in any kcal accounting (a partition's "other", the omitted share of
    energy) and a swap through fiber also moves carbohydrate's. Food tables differ, so the data
    cannot say which it is (``nesting`` does not read fiber as a part of carbohydrate).
    """
    def bearing(c: str, role: str) -> bool:
        return _role_or_none(c) == role and energy_factor(c).factor is not None

    fiber = [c for c in columns if bearing(c, "fiber")]
    carbs = [c for c in columns if bearing(c, "carbohydrate")
             and not re.search(r"sugar|starch|sucrose|fructose|lactose", str(c), re.I)]
    if not fiber or not carbs:
        return None
    return (f"{_and(fiber)} sits beside {_and(carbs)}: if carbohydrate is by difference (total "
            f"carbohydrate, as in USDA and NHANES tables), it already holds fiber, so fiber's "
            f"energy counts twice in kcal accounting and a swap through fiber also moves "
            f"carbohydrate.")


def rank_methods(purpose: Optional[str],
                 applicability: Optional[Mapping[str, Mapping[str, Any]]] = None) -> Dict[str, Any]:
    """The energy methods in order of soundness for ``purpose`` (applicable ones first), with the
    one line that names the tension between custom and soundness. No declared purpose: the
    method table's own order and no line."""
    order = list(RANKING.get(str(purpose), METHODS))
    if applicability:
        ok = [m for m in order if m == "none" or applicability.get(m, {}).get("ok")]
        order = ok + [m for m in order if m not in ok]
    line = TENSION.get(str(purpose))
    if purpose == "inference" and order and order[0] != "all_components":
        # The line never says all components ranks first when it cannot run here and the order
        # lists it last (repair round): it says which method leads instead, and why.
        lead = METHOD_TABLE[order[0]]["label"]
        line = (f"All components would rank first for substitution questions (Tomova 2022), but it "
                f"cannot run on these columns; the {lead[0].lower() + lead[1:]} leads among those "
                f"that can.")
    return {"order": order, "first": order[0] if order else None, "line": line}


def _unit_word(column: str) -> str:
    return {"grams": "g", "kcal": "kcal", "kj": "kJ"}.get(unit_of(column), "unit")


def relative_effect_rows(matrix: pd.DataFrame, nutrients: Sequence[str],
                         factors: Mapping[str, float], *,
                         table: Optional[Callable[[pd.DataFrame], Sequence[Mapping[str, Any]]]] = None,
                         coefficients: Optional[Sequence[Mapping[str, Any]]] = None
                         ) -> List[Dict[str, Any]]:
    """Each nutrient's average relative causal effect from the all-components model (ME-14).

    For nutrient j with kcal term x_j and every other energy term x_k (``kcal_from_other``
    included), ``θ_j = β_j − Σ_k w_k β_k`` with ``w_k = mean(x_k) / Σ_k' mean(x_k')`` over the rows
    the model was fit on: Tomova et al. (2022) subtract "a weighted average of the estimated effects
    of all other individual component sources of energy", w_i being "the proportion of the
    remaining energy intake contributed by each component".

    ``θ_j`` is exactly the coefficient of x_j once each x_k is replaced by ``x_k + w_k x_j``: an
    invertible change of the model's columns, with the same fit, residuals and leverages. So
    ``table(matrix)`` (the family's own coefficient table, whatever its covariance: HC3, CR2,
    Firth) gives ``θ_j`` its interval as it gives any coefficient one. Without ``table`` (no
    intervals: prediction) the estimate is read off ``coefficients``. Reported per unit of the
    nutrient: × its kcal per unit (``factors``).
    """
    kcal = [f"kcal_from_{n}" for n in nutrients if f"kcal_from_{n}" in matrix.columns]
    if "kcal_from_other" in matrix.columns:
        kcal.append("kcal_from_other")
    if len(kcal) < 2:
        return []
    means = matrix[kcal].astype(float).mean()
    by_feature = {str(r["feature"]): r for r in (coefficients or [])}
    rows: List[Dict[str, Any]] = []
    for n in nutrients:
        j = f"kcal_from_{n}"
        if j not in kcal:
            continue
        others = [k for k in kcal if k != j]
        total = float(means[others].sum())
        if not np.isfinite(total) or total <= 0:
            continue
        w = {k: float(means[k]) / total for k in others}
        f = float(factors.get(n, 1.0))
        shares = ", ".join(f"{k.replace('kcal_from_', '')} {w[k]:.0%}" for k in others)
        meaning = (f"{n} in place of the other energy sources, weighted by their share of the "
                   f"remaining energy ({shares}): the average relative effect, per "
                   f"{_unit_word(n)}")

        def scaled(v: Any, _f: float = f) -> Optional[float]:
            return None if v is None else float(v) * _f

        if table is not None:
            moved = matrix.copy()
            for k in others:
                moved[k] = matrix[k].astype(float) + w[k] * matrix[j].astype(float)
            found = next((r for r in table(moved) if str(r["feature"]) == j), None)
            if found is None:
                continue
            row = {"feature": f"{n}_relative", "estimate": scaled(found.get("estimate")),
                   "ci_low": scaled(found.get("ci_low")),
                   "ci_high": scaled(found.get("ci_high")), "p": found.get("p"),
                   "se": scaled(found.get("se")), "df": found.get("df"),
                   "meaning": meaning}
            if "ratio" in found:
                # On the table's ratio scale (WP8): exp of the per-unit estimate and its ends, or
                # None where that is missing or overflows, as the table's own rows do.
                def ratio(v: Optional[float]) -> Optional[float]:
                    return (None if v is None or not np.isfinite(v) or v > 700
                            else float(np.exp(v)))

                row.update(ratio=ratio(row["estimate"]), ratio_low=ratio(row["ci_low"]),
                           ratio_high=ratio(row["ci_high"]))
            rows.append(row)
            continue
        if j not in by_feature or any(k not in by_feature for k in others):
            continue
        b = {k: by_feature[k].get("estimate") for k in [j, *others]}
        if any(v is None for v in b.values()):
            continue
        theta = float(b[j]) - sum(w[k] * float(b[k]) for k in others)
        rows.append({"feature": f"{n}_relative", "estimate": theta * f, "ci_low": None,
                     "ci_high": None, "p": None, "meaning": meaning})
    return rows


def coefficient_gap(task: str, y: Any, dropped: pd.DataFrame, kept: pd.DataFrame,
                    pairs: Sequence[Tuple[str, str]]) -> List[Dict[str, Any]]:
    """Each adjusted nutrient's coefficient with total energy left out of the outcome model and
    with it kept (the standard model), fit on the same rows (ME-03: "show the gap").

    ``dropped`` and ``kept`` are the two model matrices; ``pairs`` (column in ``dropped``, column
    in ``kept``) name the nutrient in each. Least squares for a regression, logistic regression
    for a binary outcome (coded 0/1); [] for anything else or a fit that fails.
    """
    import statsmodels.api as sm

    if task not in ("regression", "binary"):
        return []
    yv = pd.Series(np.asarray(y, dtype=float), index=dropped.index)
    ok = yv.notna() & dropped.notna().all(axis=1) & kept.notna().all(axis=1)
    if int(ok.sum()) <= max(dropped.shape[1], kept.shape[1]) + 1:
        return []

    def fit(matrix: pd.DataFrame) -> Any:
        X = sm.add_constant(matrix.loc[ok].astype(float), has_constant="add")
        if task == "regression":
            return sm.OLS(yv[ok], X).fit()
        return sm.Logit(yv[ok], X).fit(disp=0, maxiter=200)

    try:
        a, b = fit(dropped), fit(kept)
    except Exception:  # noqa: BLE001 - a gap that cannot be computed is not shown
        return []
    return [{"nutrient": d, "dropped": float(a.params[d]), "standard": float(b.params[k]),
             "n_rows": int(ok.sum())}
            for d, k in pairs if d in a.params.index and k in b.params.index]

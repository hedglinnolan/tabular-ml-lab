"""Goldberg cut-offs for reported energy intake against basal metabolic rate (AUDIT_REPORT §5 WP12).

A screen for implausible energy reports that is individual where the fixed kcal screens are not:
each person's reported energy intake (EI) is divided by their estimated basal metabolic rate (BMR),
and the ratio is compared with the physical activity level (PAL) the population is expected to
have, within confidence limits that allow for day-to-day variation in intake, the error of the BMR
estimate and the spread of PAL between people.

**The cut-offs** (Goldberg et al. 1991, *Eur J Clin Nutr* 45:569; restated by Black 2000, *Int J
Obes* 24:1119)::

    lower = PAL × exp(−SD × (S / 100) / √n)      upper = PAL × exp(+SD × (S / 100) / √n)
    S = √(CV²_wEI / d + CV²_wB + CV²_tP)

with SD = 2 for the 95% limits ("SDmin is −2 for the 95% lower confidence limit, SDmax is +2 for
the 95% upper confidence limit": EFSA 2014, EU Menu guidance Appendix 8.2.1, §4.2), d the number of
days of intake the reported EI averages, and n = 1 when individuals are screened. Black's values
(abstract of Black 2000): "The suggested value for average within-subject variation in energy
intake is 23% (unchanged) … For within-subject variation in measured and estimated BMR, 4% and
8.5% respectively are suggested … and for total between-subject variation in PAL, the suggested
value is 15%". With BMR estimated from an equation, CV_wB = 8.5%. Black also: "the PAL value of
1.55 x BMR is not necessarily the value of choice", so PAL is always stated, never assumed.

Banna et al. 2017 (*Front Nutr* 4:45) on the same screen: "Regardless of which method is used, for
the time being, analyses in the total sample without exclusion of participants should also be
conducted and reported" (the primary-plus-sensitivity answer, ``stages/sensitivity.py``), and
"body weight is included in both the calculation of implausible rEI and the outcome variable …
which could artificially elevate the association" (the screen reads weight; a rule that reads the
outcome is refused, ``decisions._on_the_outcome``).

**BMR equations.** Three, each named on the rule, each in its own published units:

* ``schofield`` and ``schofield_height``: Schofield 1985 (*Hum Nutr Clin Nutr* 39 Suppl 1:5), the
  equations Goldberg and Black used and the ones Black's CV_wB of 8.5% was estimated for. MJ/day,
  weight in kg, height in m; ages 10 and over. The coefficients are Schofield's as reproduced in the
  literature; they agree, within its rounding, with EFSA's kcal/day rendering of the same equations
  (EFSA 2014 Appendix 8.2.1, the "Schofield equations for estimating BMR (kcal/d)" tables), which
  the acceptance test checks them against. Children under 10 are not screened: no Schofield
  equation for them was read from a primary source here.
* ``henry`` and ``henry_height``: the Oxford equations (Henry 2005, *Public Health Nutr* 8:1133,
  Tables 12 and 15), MJ/day, weight in kg, height in m, every age. EFSA's guidance names Schofield
  or Henry. Black's CV_wB was estimated for Schofield, so with Henry it is a convention, stated.
* ``mifflin``: Mifflin et al. 1990 (*Am J Clin Nutr* 51:241), resting energy expenditure in kcal/day
  from weight (kg), height (cm) and age: "REE (males) = 10 x weight (kg) + 6.25 x height (cm) - 5 x
  age (y) + 5; REE (females) = 10 x weight (kg) + 6.25 x height (cm) - 5 x age (y) - 161". Derived
  on ages 19–78; applied to ages 18 and over. It predicts resting, not basal, expenditure.

Nothing here reads the outcome. Rows whose inputs are missing, whose sex or activity level has no
equation or PAL, or whose age is outside the equation's bands are *not screened*, and the rule's
``missing`` answer decides whether they stay (the participant flow counts them on lines of their
own, as it does for range rules).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

BmrEquation = Literal["schofield", "schofield_height", "henry", "henry_height", "mifflin"]
BMR_EQUATIONS: tuple[str, ...] = ("schofield", "schofield_height", "henry", "henry_height", "mifflin")

KCAL_PER_MJ = 1000.0 / 4.184  # 1 kcal = 4.184 kJ (the thermochemical calorie)
KJ_PER_KCAL = 4.184

# Black 2000 (abstract): CV_wEI 23%, CV_wB 8.5% for estimated BMR (4% for measured), CV_tP 15%.
CV_WEI = 23.0
CV_WB_ESTIMATED = 8.5
CV_WB_MEASURED = 4.0
CV_TP = 15.0
SD_95 = 2.0  # EFSA 2014 App. 8.2.1 §4.2: "SDmin is -2 … SDmax is +2" for the 95% limits


@dataclass(frozen=True)
class Band:
    """One age band of a BMR equation: ``low <= age < high`` (``high`` None: no upper bound)."""

    low: float
    high: float | None
    weight: float
    height: float = 0.0  # per metre (Schofield, Henry) or per cm (Mifflin)
    age: float = 0.0
    constant: float = 0.0


@dataclass(frozen=True)
class Equation:
    key: str
    label: str
    unit: Literal["MJ", "kcal"]
    height_unit: Literal["m", "cm"] | None  # None: weight only
    bands: Mapping[str, tuple[Band, ...]]  # "female" / "male"
    source: str


def _bands(rows: Sequence[tuple[float, float | None, float, float, float]], *, height: bool) -> tuple[Band, ...]:
    if height:
        return tuple(Band(lo, hi, w, h, 0.0, c) for lo, hi, w, h, c in rows)
    return tuple(Band(lo, hi, w, 0.0, 0.0, c) for lo, hi, w, _, c in rows)


# Schofield 1985, MJ/day; W kg, H m. Ages 10–18, 18–30, 30–60, 60 and over.
_SCHOFIELD_W = {
    "male": _bands([(10, 18, 0.074, 0, 2.754), (18, 30, 0.063, 0, 2.896),
                    (30, 60, 0.048, 0, 3.653), (60, None, 0.049, 0, 2.459)], height=False),
    "female": _bands([(10, 18, 0.056, 0, 2.898), (18, 30, 0.062, 0, 2.036),
                      (30, 60, 0.034, 0, 3.538), (60, None, 0.038, 0, 2.755)], height=False),
}
_SCHOFIELD_WH = {
    "male": _bands([(10, 18, 0.068, 0.574, 2.157), (18, 30, 0.063, -0.042, 2.953),
                    (30, 60, 0.048, -0.011, 3.670), (60, None, 0.038, 4.068, -3.491)], height=True),
    "female": _bands([(10, 18, 0.035, 1.948, 0.837), (18, 30, 0.057, 1.184, 0.411),
                      (30, 60, 0.034, 0.006, 3.530), (60, None, 0.033, 1.917, 0.074)], height=True),
}
# Henry 2005, Table 12 (weight) and Table 15 (weight and height), MJ/day; W kg, H m.
_HENRY_W = {
    "male": _bands([(0, 3, 0.255, 0, -0.141), (3, 10, 0.0937, 0, 2.15), (10, 18, 0.0769, 0, 2.43),
                    (18, 30, 0.0669, 0, 2.28), (30, 60, 0.0592, 0, 2.48),
                    (60, None, 0.0563, 0, 2.15)], height=False),
    "female": _bands([(0, 3, 0.246, 0, -0.0965), (3, 10, 0.0842, 0, 2.12), (10, 18, 0.0465, 0, 3.18),
                      (18, 30, 0.0546, 0, 2.33), (30, 60, 0.0407, 0, 2.90),
                      (60, None, 0.0424, 0, 2.38)], height=False),
}
_HENRY_WH = {
    "male": _bands([(0, 3, 0.118, 3.59, -1.55), (3, 10, 0.0632, 1.31, 1.28),
                    (10, 18, 0.0651, 1.11, 1.25), (18, 30, 0.0600, 1.31, 0.473),
                    (30, 60, 0.0476, 2.26, -0.574), (60, None, 0.0478, 2.26, -1.07)], height=True),
    "female": _bands([(0, 3, 0.127, 2.94, -1.20), (3, 10, 0.0666, 0.878, 1.46),
                      (10, 18, 0.0393, 1.04, 1.93), (18, 30, 0.0433, 2.57, -1.18),
                      (30, 60, 0.0342, 2.10, -0.0486), (60, None, 0.0356, 1.76, 0.0448)], height=True),
}
# Mifflin et al. 1990 (abstract), kcal/day; W kg, H cm, age in years. Adults (derived on 19–78).
_MIFFLIN = {
    "male": (Band(18, None, 10.0, 6.25, -5.0, 5.0),),
    "female": (Band(18, None, 10.0, 6.25, -5.0, -161.0),),
}

EQUATIONS: dict[str, Equation] = {
    "schofield": Equation("schofield", "Schofield (weight)", "MJ", None, _SCHOFIELD_W,
                          "Schofield 1985, Hum Nutr Clin Nutr 39 Suppl 1:5"),
    "schofield_height": Equation("schofield_height", "Schofield (weight and height)", "MJ", "m",
                                 _SCHOFIELD_WH, "Schofield 1985, Hum Nutr Clin Nutr 39 Suppl 1:5"),
    "henry": Equation("henry", "Henry's Oxford equations (weight)", "MJ", None, _HENRY_W,
                      "Henry 2005, Public Health Nutr 8:1133, Table 12"),
    "henry_height": Equation("henry_height", "Henry's Oxford equations (weight and height)", "MJ", "m",
                             _HENRY_WH, "Henry 2005, Public Health Nutr 8:1133, Table 15"),
    "mifflin": Equation("mifflin", "Mifflin–St Jeor", "kcal", "cm", _MIFFLIN,
                        "Mifflin et al. 1990, Am J Clin Nutr 51:241"),
}


# ── the cut-offs ─────────────────────────────────────────────────────────────


def goldberg_s(days: Any, *, cv_wei: float = CV_WEI, cv_wb: float = CV_WB_ESTIMATED,
               cv_tp: float = CV_TP) -> Any:
    """Black's S (in %): √(CV²_wEI / d + CV²_wB + CV²_tP). ``days`` may be an array."""
    d = np.asarray(days, dtype=float)
    s = np.sqrt(cv_wei ** 2 / d + cv_wb ** 2 + cv_tp ** 2)
    return float(s) if s.ndim == 0 else s


def goldberg_cutoffs(pal: Any, days: Any, n: Any = 1, *, cv_wei: float = CV_WEI,
                     cv_wb: float = CV_WB_ESTIMATED, cv_tp: float = CV_TP,
                     sd: float = SD_95) -> tuple[Any, Any]:
    """The lower and upper Goldberg cut-offs for EI:BMR (Black 2000).

    ``pal``, ``days`` and ``n`` broadcast against each other; ``n = 1`` screens individuals, a larger
    n tests a group's mean EI:BMR. Returns floats for scalar inputs, arrays otherwise.
    """
    s = np.asarray(goldberg_s(days, cv_wei=cv_wei, cv_wb=cv_wb, cv_tp=cv_tp), dtype=float)
    half = sd * (s / 100.0) / np.sqrt(np.asarray(n, dtype=float))
    p = np.asarray(pal, dtype=float)
    lower, upper = p * np.exp(-half), p * np.exp(half)
    if lower.ndim == 0:
        return float(lower), float(upper)
    return lower, upper


# ── BMR ──────────────────────────────────────────────────────────────────────


def bmr_kcal(equation: str, sex: Any, age: Any, weight_kg: Any, height: Any = None, *,
             height_unit: Literal["cm", "m"] = "cm") -> np.ndarray:
    """Estimated BMR (kcal/day) per row; NaN where the equation has no band for the row.

    ``sex`` holds "female" / "male" (anything else is NaN); ``height`` is in ``height_unit`` and is
    converted to the equation's own unit.
    """
    eq = EQUATIONS[equation]
    sex = np.asarray(sex, dtype=object)
    age = np.asarray(age, dtype=float)
    w = np.asarray(weight_kg, dtype=float)
    n = len(w)
    h = np.full(n, np.nan) if height is None else np.asarray(height, dtype=float)
    if eq.height_unit is not None:
        if height_unit != eq.height_unit:
            h = h / 100.0 if eq.height_unit == "m" else h * 100.0
    out = np.full(n, np.nan)
    for who, bands in eq.bands.items():
        mine = sex == who
        for b in bands:
            at = mine & (age >= b.low) & ((age < b.high) if b.high is not None else True)
            value = b.weight * w[at] + b.age * age[at] + b.constant
            if eq.height_unit is not None:
                value = value + b.height * h[at]
            out[at] = value
    if eq.unit == "MJ":
        out = out * KCAL_PER_MJ
    with np.errstate(invalid="ignore"):
        out[~(out > 0)] = np.nan
    return out


def age_bands(equation: str) -> tuple[float, float | None]:
    """The youngest and oldest ages (``None``: no upper bound) the equation covers."""
    bands = [b for bs in EQUATIONS[equation].bands.values() for b in bs]
    highs = [b.high for b in bands]
    return min(b.low for b in bands), (None if any(h is None for h in highs) else max(highs))


# ── one rule over a frame ────────────────────────────────────────────────────


def level_key(value: Any) -> str | None:
    """One spelling per level (``2``, ``2.0`` and ``"2"`` are one level); None for a missing value.
    The same spelling as ``stages.rows._level_key``."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value).strip()
    if math.isfinite(number) and number.is_integer():
        return str(int(number))
    return str(value).strip()


def rule_columns(rule: Any) -> list[str]:
    """Every column a Goldberg rule reads, in a stable order."""
    cols = [rule.column, rule.sex, rule.age, rule.weight]
    if EQUATIONS[rule.equation].height_unit is not None and rule.height:
        cols.append(rule.height)
    if rule.pal_by is not None:
        cols.append(rule.pal_by.column)
    if rule.days_column:
        cols.append(rule.days_column)
    return list(dict.fromkeys(cols))


LB_TO_KG = 0.45359237  # the international yard and pound agreement (1959): 1 lb = 0.45359237 kg


def screen(frame: pd.DataFrame, rule: Any) -> pd.DataFrame:
    """Each row's screen under a Goldberg ``rule``, indexed like ``frame``.

    Columns: ``sex`` ("female"/"male"/None), ``bmr`` (in the energy column's unit per day),
    ``ratio`` (EI:BMR), ``pal``, ``days``, ``lower``, ``upper``, ``recorded`` (every input has a
    value), ``screened`` (an equation, a PAL and a day count apply), ``inside`` (the ratio is within
    the cut-offs the rule screens by) and ``status``: "not recorded", "not screened", "under",
    "plausible" or "over".
    """
    index = frame.index
    num = {c: pd.to_numeric(frame[c], errors="coerce").astype(float)
           for c in (rule.column, rule.age, rule.weight) if c in frame.columns}
    energy, age, weight = num[rule.column], num[rule.age], num[rule.weight]
    if getattr(rule, "weight_unit", "kg") == "lb":
        weight = weight * LB_TO_KG  # the international pound, exactly
    needs_height = EQUATIONS[rule.equation].height_unit is not None
    height = (pd.to_numeric(frame[rule.height], errors="coerce").astype(float)
              if needs_height and rule.height else pd.Series(np.nan, index=index))
    female = {level_key(v) for v in rule.female}
    male = {level_key(v) for v in rule.male}
    sex_keys = frame[rule.sex].map(level_key)
    sex = pd.Series(np.where(sex_keys.isin(female), "female",
                             np.where(sex_keys.isin(male), "male", None)), index=index, dtype=object)

    recorded = energy.notna() & age.notna() & weight.notna() & sex_keys.notna()
    if needs_height:
        recorded &= height.notna()

    if rule.pal_by is not None:
        # A level with a PAL of its own takes it; any other row takes the rule's own PAL when it
        # has one (a fallback), else it is not screened. A missing level with no fallback is an
        # input not recorded.
        levels = frame[rule.pal_by.column].map(level_key)
        table = {level_key(k): float(v) for k, v in rule.pal_by.values.items()}
        pal = levels.map(table).astype(float)
        if rule.pal is not None:
            pal = pal.fillna(float(rule.pal))
        else:
            recorded &= levels.notna()
    else:
        pal = pd.Series(float(rule.pal), index=index)

    if rule.days_column:
        days = pd.to_numeric(frame[rule.days_column], errors="coerce").astype(float)
        recorded &= days.notna()
    else:
        days = pd.Series(float(rule.days), index=index)

    bmr = pd.Series(bmr_kcal(rule.equation, sex.to_numpy(), age.to_numpy(), weight.to_numpy(),
                             height.to_numpy(), height_unit=rule.height_unit), index=index)
    if rule.energy_unit == "kj":
        bmr = bmr * KJ_PER_KCAL
    screened = sex.notna() & bmr.notna() & pal.notna() & (days >= 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = energy / bmr
    lower, upper = goldberg_cutoffs(pal.to_numpy(), days.clip(lower=1).to_numpy(),
                                    cv_wei=rule.cv_wei, cv_wb=rule.cv_wb, cv_tp=rule.cv_tp)
    lower, upper = pd.Series(lower, index=index), pd.Series(upper, index=index)
    under = (ratio < lower).fillna(False)
    over = (ratio > upper).fillna(False)
    inside = ~((under if rule.exclude in ("both", "under") else False)
               | (over if rule.exclude in ("both", "over") else False))
    inside = pd.Series(inside, index=index).astype(bool)
    status = np.where(~recorded, "not recorded",
                      np.where(~screened, "not screened",
                               np.where(under, "under", np.where(over, "over", "plausible"))))
    return pd.DataFrame({"sex": sex, "bmr": bmr, "ratio": ratio, "pal": pal, "days": days,
                         "lower": lower, "upper": upper, "recorded": recorded.astype(bool),
                         "screened": (recorded & screened).astype(bool), "inside": inside,
                         "status": status}, index=index)


def parts(frame: pd.DataFrame, rule: Any) -> tuple[pd.Series, pd.Series, pd.Series]:
    """``(recorded, screened, inside)`` as the participant flow reads a rule (``stages.rows``)."""
    s = screen(frame, rule)
    return s["recorded"], s["screened"] | ~s["recorded"], s["inside"]


def label(rule: Any) -> str:
    """What the rows the rule keeps are, for the participant flow: true of every one of them."""
    eq = EQUATIONS[rule.equation].label
    pal = (f"PAL by `{rule.pal_by.column}`" if rule.pal_by is not None
           else f"PAL `{_num(rule.pal)}`")
    days = f"`{rule.days_column}` days" if rule.days_column else (
        f"`{_num(rule.days)}` {'day' if float(rule.days) == 1 else 'days'}")
    side = {"both": "within", "under": "not below", "over": "not above"}[rule.exclude]
    text = f"`{rule.column}`:BMR {side} the Goldberg cut-offs ({eq}, {pal}, {days})"
    return text + (", or not screened" if rule.missing == "keep" else "")


def describe(rule: Any) -> str:
    """The rule in a methods section's words (Black 2000's values named when they are the ones used)."""
    eq = EQUATIONS[rule.equation]
    black = (rule.cv_wei, rule.cv_wb, rule.cv_tp) == (CV_WEI, CV_WB_ESTIMATED, CV_TP)
    cvs = ("Black's (2000) coefficients of variation (intake 23%, BMR 8.5%, PAL 15%)" if black else
           f"coefficients of variation of {_num(rule.cv_wei)}% (intake), {_num(rule.cv_wb)}% (BMR) "
           f"and {_num(rule.cv_tp)}% (PAL)")
    pal = (f"a PAL set by `{rule.pal_by.column}`" if rule.pal_by is not None
           else f"a PAL of {_num(rule.pal)}")
    days = (f"the days in `{rule.days_column}`" if rule.days_column
            else f"{_num(rule.days)} {'day' if float(rule.days) == 1 else 'days'} of intake")
    side = {"both": "under- and over-reporters", "under": "under-reporters",
            "over": "over-reporters"}[rule.exclude]
    return (f"{side.capitalize()} were identified by the Goldberg cut-offs for energy intake to "
            f"BMR at 95% confidence (n = 1), with BMR from {eq.label} ({eq.source}), {pal}, "
            f"{days} and {cvs}")


def _num(value: Any) -> str:
    value = float(value)
    return f"{int(value)}" if value.is_integer() else f"{value:g}"


__all__ = [
    "BMR_EQUATIONS", "Band", "BmrEquation", "CV_TP", "CV_WB_ESTIMATED", "CV_WB_MEASURED", "CV_WEI",
    "EQUATIONS", "Equation", "KCAL_PER_MJ", "SD_95", "age_bands", "bmr_kcal", "describe",
    "goldberg_cutoffs", "goldberg_s", "label", "level_key", "parts", "rule_columns", "screen",
]

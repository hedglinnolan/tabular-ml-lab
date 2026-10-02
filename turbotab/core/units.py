"""The outcome's unit (mg/dL, mmol/L, %…), so every outcome quantity can say what it is in.

M2_CONTRACT §6: "Outcome units wherever an outcome quantity is shown, read from the column name or
the pack." Two readings, in order:

1. **The name.** A suffix that states a unit: ``glucose_mgdl``, ``ldl_mmol_l``, ``sbp_mmhg``,
   ``weight_kg``, ``hba1c_pct``.
2. **The pack.** The clinical pack's analytes (``CLINICAL_SURVEY_PACK.md`` §A1.1 and §A1.2): a
   name that reads as an analyte with one usual unit (blood pressure in mmHg, BMI in kg/m²) takes
   it; an analyte reported in two units (glucose in mg/dL or mmol/L) takes the one whose typical
   adult value is nearest the column's median on a log scale — the conversion factors are large
   (18 for glucose, 38.67 for cholesterol), so the two readings sit far apart.

Anything else is unknown, and the app says nothing rather than guess (asking once is inboxed).
"""
from __future__ import annotations

import math
import re
from typing import Any

# The last one or two name tokens that state a unit.
_SUFFIX: dict[str, str] = {
    "mgdl": "mg/dL", "mg/dl": "mg/dL", "mmoll": "mmol/L", "mmol": "mmol/L", "umoll": "µmol/L",
    "umol": "µmol/L", "gdl": "g/dL", "mmolmol": "mmol/mol", "mmhg": "mmHg", "kg": "kg",
    "cm": "cm", "lb": "lb", "lbs": "lb", "pct": "%", "percent": "%", "bpm": "beats/min",
    "kcal": "kcal", "kj": "kJ", "mcg": "µg", "ug": "µg", "mg": "mg", "g": "g", "ngml": "ng/mL",
    "pgml": "pg/mL", "iu": "IU", "years": "years", "yrs": "years", "yr": "years", "kgm2": "kg/m²",
}

# (name pattern, [(unit, typical adult value)]) — CLINICAL_SURVEY_PACK §A1.1's conversion table
# and §A1.2's plausibility table; the typical values sit inside the reference intervals.
_ANALYTES: list[tuple[str, list[tuple[str, float]]]] = [
    (r"hba1c|\ba1c\b|glycohemoglobin|glycated", [("%", 5.7), ("mmol/mol", 39.0)]),
    (r"gluc|\bglc\b", [("mg/dL", 100.0), ("mmol/L", 5.5)]),
    (r"triglyc|\btg\b|\btrig\b", [("mg/dL", 130.0), ("mmol/L", 1.5)]),
    (r"chol|\bldl\b|\bhdl\b", [("mg/dL", 120.0), ("mmol/L", 3.1)]),
    (r"creat", [("mg/dL", 0.9), ("µmol/L", 80.0)]),
    (r"bilirubin|\btbili\b", [("mg/dL", 0.7), ("µmol/L", 12.0)]),
    (r"hemoglobin|haemoglobin|\bhgb\b", [("g/dL", 14.0), ("g/L", 140.0)]),
    (r"\bbp\b|\bsbp\b|\bdbp\b|systolic|diastolic|blood pressure|\bbp (sys|di|dia)", [("mmHg", 120.0)]),
    (r"\bbmi\b", [("kg/m²", 27.0)]),
    (r"\bwaist\b|\bhip\b", [("cm", 95.0), ("in", 37.0)]),
    (r"height|stature", [("cm", 168.0), ("in", 66.0)]),
    (r"weight|\bwt\b", [("kg", 78.0), ("lb", 172.0)]),
    (r"\bage\b", [("years", 45.0)]),
    (r"heart rate|\bhr\b|pulse", [("beats/min", 72.0)]),
]


def _words(column: str) -> str:
    return re.sub(r"[_\-.]+", " ", str(column).lower()).strip()


def from_name(column: str) -> str | None:
    """The unit the column's name states by its suffix, or None."""
    tokens = [t for t in re.split(r"[^a-z0-9]+", str(column).lower()) if t]
    if len(tokens) == 1 and tokens[0] in ("kcal", "kj"):
        return _SUFFIX[tokens[0]]
    if len(tokens) < 2:
        return None
    for n in (3, 2, 1):
        if len(tokens) > n:
            unit = _SUFFIX.get("".join(tokens[-n:]))
            if unit is not None:
                return unit
    return None


def _median(values: Any) -> float | None:
    if values is None:
        return None
    try:
        import numpy as np
        import pandas as pd

        x = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
        x = x[np.isfinite(x) & (x > 0)]
        return float(np.median(x)) if len(x) else None
    except Exception:  # noqa: BLE001 - no values to judge by: the pack only names single units
        return None


def from_pack(column: str, values: Any = None) -> str | None:
    """The clinical pack's unit for an analyte the name reads as, judged by magnitude when two
    units are in use; None when the name is no analyte or the values cannot decide."""
    name = _words(column)
    for pattern, units in _ANALYTES:
        if not re.search(pattern, name):
            continue
        if len(units) == 1:
            return units[0][0]
        median = _median(values)
        if median is None:
            return None
        return min(units, key=lambda u: abs(math.log(median) - math.log(u[1])))[0]
    return None


def outcome_unit(column: str, values: Any = None) -> tuple[str | None, str | None]:
    """``(unit, source)`` with source ``"name"`` or ``"pack"``; ``(None, None)`` when unknown."""
    unit = from_name(column)
    if unit is not None:
        return unit, "name"
    unit = from_pack(column, values)
    return (unit, "pack") if unit is not None else (None, None)


def with_unit(text: str, unit: str | None) -> str:
    """``"1.23"`` → ``"1.23 mg/dL"``; a percent sticks to its number."""
    if not unit:
        return text
    return f"{text}%" if unit == "%" else f"{text} {unit}"


__all__ = ["from_name", "from_pack", "outcome_unit", "with_unit"]


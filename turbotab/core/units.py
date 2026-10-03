"""The outcome's unit (mg/dL, mmol/L, %…), so every outcome quantity can say what it is in.

M2_CONTRACT §6: "Outcome units wherever an outcome quantity is shown." Audit IN-05 found the unit
guessed from the name and the values and written into the record and the estimand: dietary choline
in mg/day became "mg/dL", gestational age in weeks "years", birth weight in grams "lb", sleep hours
"beats/min". CLINICAL_SURVEY_PACK §A1.1 says what the app must do instead: "Please confirm units
per analyte against the source data dictionary — TurboTab will not guess", and "Detect, propose,
require explicit confirmation."

So there are two readings, and only one of them is ever stated:

1. **Stated** (:func:`outcome_unit`): a unit the user recorded (``set_outcome_unit``), else one the
   name spells out in full as its suffix (``glucose_mg_dl``, ``ldl_mmol_l``, ``sbp_mmhg``,
   ``weight_kg``). A sentence carries a unit only from here.
2. **Proposed** (:func:`proposed_unit`): the clinical pack's analytes (§A1.1 and §A1.2), matched as
   whole words, never in an intake (``dietary_``, ``_intake``, ``per day``) or a specimen the
   reference values do not describe (urine). Where an analyte has one usual unit (BMI in kg/m²)
   that unit is proposed; where it has two, the one whose typical adult value is nearest the
   column's median on a log scale, but only when the two sit at least :data:`MIN_FACTOR` apart:
   kg and lb (2.2×) or cm and in (2.54×) are listed as candidates and never picked by magnitude.
   A proposal is offered for the user's decision; it is never written into a sentence.
"""
from __future__ import annotations

import math
import re
from typing import Any

# A full unit spelled by the name's last one, two or three words (joined without separators).
_SUFFIX: dict[str, str] = {
    "mgdl": "mg/dL", "mmoll": "mmol/L", "umoll": "µmol/L", "nmoll": "nmol/L", "pmoll": "pmol/L",
    "gdl": "g/dL", "gl": "g/L", "mgl": "mg/L", "mmolmol": "mmol/mol", "mmhg": "mmHg", "kg": "kg",
    "cm": "cm", "lb": "lb", "lbs": "lb", "pct": "%", "percent": "%", "bpm": "beats/min",
    "kcal": "kcal", "kj": "kJ", "mcg": "µg", "ug": "µg", "mg": "mg", "g": "g", "ngml": "ng/mL",
    "pgml": "pg/mL", "iu": "IU", "iul": "IU/L", "ul": "U/L", "years": "years", "yrs": "years",
    "yr": "years", "kgm2": "kg/m²", "mgday": "mg/day", "mgd": "mg/day", "gday": "g/day",
    "mcgday": "µg/day", "ugday": "µg/day", "kcalday": "kcal/day", "kjday": "kJ/day",
    "weeks": "weeks", "wks": "weeks", "days": "days", "months": "months", "hours": "hours",
    "hrs": "hours", "minutes": "min", "mins": "min",
}
# A bare amount ("mmol", "umol") says neither per litre nor per mole, so it is not a full unit:
# ``hba1c_mmol`` is mmol/mol, ``glucose_mmol`` mmol/L. Such a name is proposed, never stated.

# (whole-word name pattern, [(unit, typical adult value)]) — CLINICAL_SURVEY_PACK §A1.1's
# conversion table and §A1.2's plausibility table; the typical values sit inside the reference
# intervals.
_ANALYTES: list[tuple[str, list[tuple[str, float]]]] = [
    (r"\b(hba1c|a1c|glycohemoglobin|ghb)\b|\bglycated\b", [("%", 5.7), ("mmol/mol", 39.0)]),
    (r"\b(glucose|glu|glc|fpg)\b", [("mg/dL", 100.0), ("mmol/L", 5.5)]),
    (r"\b(triglycerides?|tg|trig|trigs)\b", [("mg/dL", 130.0), ("mmol/L", 1.5)]),
    (r"\b(cholesterol|chol|tc|ldl|hdl|ldlc|hdlc)\b", [("mg/dL", 120.0), ("mmol/L", 3.1)]),
    (r"\b(creatinine|creat|scr)\b", [("mg/dL", 0.9), ("µmol/L", 80.0)]),
    (r"\b(bilirubin|tbili)\b", [("mg/dL", 0.7), ("µmol/L", 12.0)]),
    (r"\b(hemoglobin|haemoglobin|hgb|hb)\b", [("g/dL", 14.0), ("g/L", 140.0)]),
    (r"^(bp|sbp|dbp|map)$|\b(systolic|diastolic)\b|\bbp (sys|dia|systolic|diastolic)\b|"
     r"\bblood pressure\b", [("mmHg", 120.0)]),
    (r"\bbmi\b", [("kg/m²", 27.0)]),
    (r"\b(waist|hip)\b", [("cm", 95.0), ("in", 37.0)]),
    (r"\b(height|stature)\b", [("cm", 168.0), ("in", 66.0)]),
    (r"\b(weight|wt)\b", [("kg", 78.0), ("lb", 172.0)]),
    (r"^age$|^age (years|yrs)$", [("years", 45.0)]),
    (r"\bheart rate\b|^(hr|pulse|pulse rate)$", [("beats/min", 72.0)]),
]
# Two units this close are not told apart by magnitude: a heavy cohort's kg is a light one's lb.
MIN_FACTOR = 3.0
# Words that make a name an intake, a specimen the reference values do not describe, or a
# quantity the analyte reference is not about (birth weight, gestational age).
_NOT_BLOOD = {"dietary", "diet", "intake", "intakes", "consumption", "consumed", "per", "daily",
              "supplement", "supplements", "food", "foods", "urine", "urinary", "csf", "saliva",
              "birth", "gestational", "fetal", "infant", "newborn", "length", "telomere", "sleep",
              "steps", "activity", "recreational", "exercise", "kinase", "change", "delta"}


def _words(column: str) -> str:
    spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", str(column))
    return re.sub(r"[^a-z0-9]+", " ", spaced.lower()).strip()


def from_name(column: str) -> str | None:
    """The unit the column's name spells out in full as its suffix, or None."""
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


def proposed_unit(column: str, values: Any = None) -> dict[str, Any] | None:
    """The clinical pack's proposal for an analyte the name reads as: ``{unit, candidates, source}``
    with ``unit`` None when the values cannot decide between candidates; None when the name is no
    analyte. A proposal is for the user's decision (``set_outcome_unit``), never a sentence."""
    name = _words(column)
    if set(name.split()) & _NOT_BLOOD:
        return None
    for pattern, units in _ANALYTES:
        if not re.search(pattern, name):
            continue
        candidates = [u for u, _ in units]
        if len(units) == 1:
            return {"unit": units[0][0], "candidates": candidates, "source": "pack"}
        factor = max(u[1] for u in units) / min(u[1] for u in units)
        median = _median(values)
        if median is None or factor < MIN_FACTOR:
            return {"unit": None, "candidates": candidates, "source": "pack"}
        best = min(units, key=lambda u: abs(math.log(median) - math.log(u[1])))[0]
        return {"unit": best, "candidates": candidates, "source": "pack"}
    return None


def from_pack(column: str, values: Any = None) -> str | None:
    """The pack's proposed unit alone (:func:`proposed_unit`), or None. A proposal, not a fact."""
    proposal = proposed_unit(column, values)
    return proposal["unit"] if proposal is not None else None


def outcome_unit(column: str, values: Any = None,
                 recorded: str | None = None) -> tuple[str | None, str | None]:
    """The unit a sentence may state, ``(unit, source)``: the user's recorded unit (source
    ``"decision"``), else a full unit the name spells out (``"name"``), else ``(None, None)``.
    ``values`` are accepted for the callers' convenience and never read: a unit is not guessed."""
    if recorded:
        return str(recorded), "decision"
    unit = from_name(column)
    if unit is not None:
        return unit, "name"
    return None, None


def recorded_unit(state: Any, column: str | None) -> str | None:
    """The unit ``set_outcome_unit`` recorded for ``column``, while it is the outcome."""
    unit = getattr(state, "outcome_unit", None)
    if not unit or column is None or getattr(state, "target", None) != column:
        return None
    return str(unit)


def with_unit(text: str, unit: str | None) -> str:
    """``"1.23"`` → ``"1.23 mg/dL"``; a percent sticks to its number."""
    if not unit:
        return text
    return f"{text}%" if unit == "%" else f"{text} {unit}"


__all__ = ["MIN_FACTOR", "from_name", "from_pack", "outcome_unit", "proposed_unit",
           "recorded_unit", "with_unit"]

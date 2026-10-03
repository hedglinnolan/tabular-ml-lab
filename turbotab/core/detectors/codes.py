"""Numeric missing-value codes in one column: 999, -9, or 7/9 in a coded question (audit IN-04, H19).

``ml.import_doctor.check_numeric_sentinels`` (Classic's, not edited) kept a candidate code when
fewer than max(3, 1% of n) real values lay on *either* side of it. On clean simulated clinical
columns that flagged a diastolic pressure of 99 or a glucose of 66 in 2–15.5% of columns, and the
Next app offered a one-click repair that blanked the most extreme, and so the sickest, patients.
This module is the reading the findings stage now serves in its place.

A value is read as a code only when the data leave room for no other reading:

* **It lies beyond every other value on its side.** A positive code (7, 9, 99, 999 …) must be
  above the column's largest other value; a negative code (-9, -99 …) below its smallest. A code
  sits outside the observations, never among them.
* **A gap separates it from the rest.** Either the gap is wide (*far*: at least ten typical
  spacings between neighbouring values and at least half the range of the rest), or, in a coded
  question (whole numbers, at most 15 of them), at least one unused value separates it from the
  answers *and* the top answer is common enough that an unused next value is itself evidence of a
  scale end (*near*): the count expected at the next value, from the decay of the top two answers,
  is at least 5, so a real continuation would be missing with probability below e^-5 ≈ 0.7%.
  A decaying count tail (children, admissions, drinks) does not pass: its top values are rare,
  so a gap above them says nothing.
* **It recurs.** At least two rows hold it.
* **A negative code in a column of non-negative values** is never a real observation, whatever
  the distance (the -9 in a 0–100 score).

Columns that describe the survey design (strata, PSUs, weights) are not read: NHANES stratum
``999`` is a real stratum that another finding already reads as one (H19).

The finding is a **warning**: the numbers alone cannot tell a code from an extreme value. It is
**critical** only when the table is an NHANES export and the code is one the NHANES codebooks
define (7 or 9 for refused and don't know in a single-digit question, 77/99, 777/999 … in wider
ones), because then the codebook corroborates the reading.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

import numpy as np
import pandas as pd

#: The codes a value must equal to be read as one. Classic's list, read and not copied.
from ml.import_doctor import NUMERIC_SENTINELS

CODED_MAX_DISTINCT = 15   # whole-number columns with at most this many values are coded questions
MIN_COUNT = 2             # rows that must hold a code
FAR_SPACINGS = 10.0       # a far code is at least this many typical spacings beyond the rest …
FAR_RANGE = 0.5           # … and at least this share of the rest's range
NEAR_EXPECTED = 5.0       # count expected at the unused next value, for a gap to mean a scale end

#: The codes the NHANES codebooks define: 7/9 ("Refused", "Don't know") and their wider repdigits.
NHANES_CODES = frozenset({7.0, 9.0, 77.0, 99.0, 777.0, 999.0, 7777.0, 9999.0, 77777.0, 99999.0,
                          777777.0, 999999.0})

_DESIGN = re.compile(r"^(?:wt[a-z0-9]*|sdmv\w*|sddsrvyr)$|(?:^|_)(?:pweight|sampweight|"
                     r"sampling_weight|survey_weight|svy_weight|weight_svy|strata|stratum|psu|"
                     r"cluster|fpc)(?:$|_)", re.I)


def design_column(name: str) -> bool:
    """A column that describes the survey design (a weight, stratum or PSU), by its name."""
    return bool(_DESIGN.search(str(name)))


def _nhanes_like(columns: Iterable[str]) -> bool:
    lowered = {str(c).lower() for c in columns}
    return bool({"seqn", "sddsrvyr", "riagendr", "ridageyr"} & lowered
                or any(c.startswith(("dr1t", "dr2t", "bmx", "bpx", "lbx", "sdmv", "wtdr", "wtmec"))
                       for c in lowered))


def _spacing(values: np.ndarray) -> float:
    """The typical gap between neighbouring distinct values (their median), 1 when undefined."""
    if len(values) < 2:
        return 1.0
    gaps = np.diff(np.sort(values))
    gaps = gaps[gaps > 0]
    return float(np.median(gaps)) if len(gaps) else 1.0


def _num(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else f"{v:g}"


def read(series: pd.Series) -> dict[str, Any] | None:
    """The codes this column holds, with why each was read as one, or None.

    ``{"values": {code: count}, "far": [codes], "near": [codes], "real_low", "real_high",
    "spacing", "coded"}``.
    """
    x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return None
    distinct, counts = np.unique(x, return_counts=True)
    count = dict(zip(distinct.tolist(), counts.tolist()))
    integral = bool(np.all(np.mod(distinct, 1) == 0))
    coded = integral and len(distinct) <= CODED_MAX_DISTINCT
    codes = {float(v) for v in NUMERIC_SENTINELS}
    flagged: dict[float, int] = {}
    far: list[float] = []
    near: list[float] = []

    # ── the upper side: positive codes above every other value ──
    # The values at the top that are codes, lowest first. The cut is the lowest of them that a gap
    # separates from everything below it (the values below it, codes or not, are then the answers):
    # in 1–9 the 7, 8 and 9 are answers, in 1–5 with 7 and 9 both are codes.
    block: list[float] = []
    for v in distinct[::-1]:
        if v > 0 and v in codes:
            block.append(float(v))
            continue
        break
    block.sort()
    for j, v in enumerate(block):
        rest = distinct[distinct < v]
        if not len(rest):
            continue
        hi, lo = float(rest.max()), float(rest.min())
        spacing = _spacing(rest)
        gap = v - hi
        if gap >= max(FAR_SPACINGS * spacing, FAR_RANGE * (hi - lo)):
            kind = far
        elif coded and gap >= 2 * spacing and _top_is_populated(rest, count):
            kind = near
        else:
            continue
        for w in block[j:]:
            if count[w] >= MIN_COUNT:
                flagged[w] = int(count[w])
                kind.append(w)
        break

    # ── the lower side: negative codes below every other value ──
    block = []
    for v in distinct:
        if v < 0 and v in codes:
            block.append(float(v))
            continue
        break
    rest = distinct[distinct > max(block)] if block else distinct
    if block and len(rest):
        lo = float(rest.min())
        hi = float(rest.max())
        spacing = _spacing(rest)
        for v in sorted(block):
            if count[v] < MIN_COUNT:
                continue
            if lo >= 0 or (lo - v) >= max(FAR_SPACINGS * spacing, FAR_RANGE * (hi - lo)):
                flagged[v] = int(count[v])
                far.append(v)

    if not flagged:
        return None
    real = x[~np.isin(x, list(flagged))]
    return {"values": {float(v): n for v, n in sorted(flagged.items())},
            "far": sorted(far), "near": sorted(near),
            "real_low": float(real.min()), "real_high": float(real.max()),
            "coded": coded, "n": int(len(x))}


def _top_is_populated(rest: np.ndarray, count: dict[float, int]) -> bool:
    """Whether a value just above the top answer would be expected at least ``NEAR_EXPECTED``
    times, extrapolating the per-step decay across the top three answers (never growing): one
    noisy count at the top of a decaying tail does not make it look like a scale end."""
    ordered = np.sort(rest)
    c = [float(count.get(float(v), 0)) for v in ordered[-3:]]
    c_top = c[-1]
    if len(c) == 1:
        return c_top >= NEAR_EXPECTED
    ratio = min(1.0, (c_top / c[0]) ** (1.0 / (len(c) - 1))) if c[0] else 1.0
    return c_top * ratio >= NEAR_EXPECTED


def findings(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """``sentinel_missing__<column>`` findings, in the structural stream's shape."""
    nhanes = _nhanes_like(frame.columns)
    out: list[dict[str, Any]] = []
    for col in frame.columns:
        s = frame[col]
        if isinstance(s, pd.DataFrame) or not pd.api.types.is_numeric_dtype(s) \
                or pd.api.types.is_bool_dtype(s) or design_column(str(col)):
            continue
        reading = read(s)
        if reading is None:
            continue
        values = reading["values"]
        listed = ", ".join(f"{_num(v)} ({n}×)" for v, n in values.items())
        above = [v for v in values if v > 0]
        below = [v for v in values if v < 0]
        where = []
        if above:
            where.append(f"above every other value (the largest is {_num(reading['real_high'])})")
        if below:
            where.append(f"below every other value (the smallest is {_num(reading['real_low'])})")
        how = []
        if any(v in reading["far"] for v in values):
            how.append("far beyond the rest, with nothing between")
        if reading["near"]:
            how.append("past an unused value at the top of a coded question whose top answer is "
                       "common, which is where a scale ends and its codes begin")
        corroborated = nhanes and all(v in NHANES_CODES for v in values)
        detail = (f"Found {listed}: {' and '.join(where)}, {'; '.join(how)}. These are "
                  f"conventional codes for a missing answer, and nothing in the numbers separates "
                  f"a code from a real extreme value, so the reading is a proposal.")
        if corroborated:
            detail += (" The table is an NHANES export, whose codebooks define 7 and 9 (and 77/99, "
                       "777/999 in wider questions) as 'Refused' and 'Don't know'.")
        out.append({
            "id": f"sentinel_missing__{col}",
            "source": "structure",
            "severity": "critical" if corroborated else "warning",
            "confidence": "high" if corroborated else "medium",
            "title": f"'{col}' may use numeric codes for missing values",
            "detail": detail,
            "why_it_matters": ("Survey and clinical exports code missing answers as 999, -9, or "
                               "7/8/9 in coded questions. Left as numbers they are averaged into "
                               "your results; recoded wrongly, they remove real extreme values."),
            "fix_label": f"Treat {', '.join(_num(v) for v in values)} as missing in '{col}'",
            "fix_kind": "recode_missing",
            "auto_suggestable": False,
            "affected_columns": [str(col)],
            "params": {"column": str(col), "values": [float(v) for v in values],
                       "rule": "beyond_every_other_value_with_a_gap",
                       "far": reading["far"], "near": reading["near"]},
            "suggested_actions": [],
        })
    return out


#: Findings after which the structural diagnosis reports nothing per column (a header stuck in a
#: later row, or two columns sharing a label): the per-column reading waits for them, as Classic's.
MASKING = ("header_in_later_row", "duplicate_columns")


def supersede(structural: list[dict[str, Any]], frame: pd.DataFrame) -> list[dict[str, Any]]:
    """The structural stream with Classic's code findings replaced by this module's."""
    kept = [f for f in structural if not str(f.get("id", "")).startswith("sentinel_missing__")]
    if any(str(f.get("id")) in MASKING for f in kept):
        return kept
    return kept + findings(frame)


__all__ = ["CODED_MAX_DISTINCT", "NHANES_CODES", "design_column", "findings", "read", "supersede"]

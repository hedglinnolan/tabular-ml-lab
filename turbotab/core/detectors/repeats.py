"""Repeats or time points: stated only when the evidence is unambiguous, asked otherwise
(audit IN-12; OPENING_SEQUENCE §03 question 4).

``turbotab.repeats.read`` (shared with the legacy app, not edited) decided from visit spacing alone,
with two unsourced cut-offs (14 days; a gap CV of 0.35). A 10-day feeding study with glucose
falling every day read "repeats", stated, with the sentence "too uneven to be a visit schedule"
at a gap CV of 0, and the aggregation menu then recommended averaging the time course away; a
weekly four-period crossover read "repeats"; four 24-hour recalls a quarter apart read "time
points: averaging them destroys the signal"; and a bare visit index 1, 2, 3 was stated as
replicates although the module's own docstring says such an index "says there is an ORDER and
says nothing about what the order means".

This reading keeps the legacy one's measurements and changes what it may state:

* **Spacing is weak evidence.** It decides only where nothing points the other way.
* **Time-point evidence**: a measured value that trends within units (Spearman ρ with the order
  of at least 0.5 in absolute value, in the same direction in at least 70% of units with three or
  more records), or a column naming a treatment, arm, period or phase that changes within units
  (a crossover). Either one turns a "repeats" reading into a question.
* **Replicate evidence under the dietary lens**: an index or a name that says the rows are
  recalls (``recall_number``, ``recall``, ``day_of_recall``, ``24h``). Recalls are replicates
  of usual intake (NUTRITION_PACK), so a recall index without dates is stated as repeats; recalls
  that the dates space like a schedule are asked, never stated as time points.
* **A short regular schedule** (daily, weekly) is a schedule, not "too uneven to be a visit
  schedule": it is asked.
* **A bare index** (``visit`` 1, 2, 3, no dates) is asked: it orders the rows and says nothing about
  what the order means.
"""
from __future__ import annotations

import re
from typing import Any, Sequence

import numpy as np
import pandas as pd

TREND_RHO = 0.5          # within-unit |Spearman ρ| a measured value must reach …
TREND_UNITS = 0.7        # … in the same direction in this share of units with ≥ 3 records
TREND_MIN_RECORDS = 3
_PERIOD = re.compile(r"(?:^|_)(?:treatment|treat|trt|arm|period|phase|condition|intervention|"
                     r"regimen|sequence|diet_period|allocation)(?:$|_)", re.I)
_RECALL = re.compile(r"(?:^|_)(?:recall|recalls|recall_number|recall_no|recall_day|24h|24hr|"
                     r"day_of_recall|replicate|rep)(?:$|_|\d)", re.I)


def _order_column(df: pd.DataFrame, reading: dict[str, Any]) -> str | None:
    spacing = reading.get("spacing") or {}
    return spacing.get("column") or reading.get("replicate_index")


def trend(df: pd.DataFrame, unit: str, order: str | None) -> dict[str, Any] | None:
    """A measured column that rises or falls within units along ``order``, or None."""
    from scipy import stats

    if order is None or order not in df.columns:
        return None
    key = df[order]
    if not pd.api.types.is_numeric_dtype(key):
        parsed = pd.to_datetime(key, errors="coerce", format="mixed")
        key = (parsed - pd.Timestamp("1970-01-01")).dt.total_seconds()
    key = pd.to_numeric(key, errors="coerce")
    best = None
    for c in df.columns:
        if c in (unit, order) or not pd.api.types.is_numeric_dtype(df[c]) \
                or pd.api.types.is_bool_dtype(df[c]):
            continue
        rhos = []
        for _, block in pd.DataFrame({"k": key, "v": df[c], "u": df[unit]}).dropna().groupby("u"):
            if len(block) < TREND_MIN_RECORDS or block["v"].nunique() < 2 or block["k"].nunique() < 2:
                continue
            rho = stats.spearmanr(block["k"], block["v"])[0]
            if np.isfinite(rho):
                rhos.append(rho)
        if len(rhos) < 5:
            continue
        rhos_a = np.array(rhos)
        sign = np.sign(np.median(rhos_a))
        share = float(np.mean((np.abs(rhos_a) >= TREND_RHO) & (np.sign(rhos_a) == sign)))
        if sign != 0 and share >= TREND_UNITS and (best is None or share > best["share"]):
            best = {"column": str(c), "share": share, "direction": "falls" if sign < 0 else "rises",
                    "median_rho": float(np.median(rhos_a)), "n_units": len(rhos)}
    return best


def period_column(df: pd.DataFrame, unit: str) -> str | None:
    """A column naming a treatment, arm or period that changes within units (a crossover)."""
    for c in df.columns:
        if c == unit or not _PERIOD.search(str(c)):
            continue
        varying = df.groupby(unit, dropna=True)[c].nunique(dropna=True)
        if len(varying) and float((varying > 1).mean()) >= 0.5:
            return str(c)
    return None


def recall_evidence(df: pd.DataFrame, reading: dict[str, Any]) -> str | None:
    """A column whose name says the rows are dietary recalls."""
    index = reading.get("replicate_index")
    if index and _RECALL.search(str(index)):
        return str(index)
    spacing = reading.get("spacing") or {}
    if spacing.get("column") and _RECALL.search(str(spacing["column"])):
        return str(spacing["column"])
    return next((str(c) for c in df.columns if _RECALL.search(str(c))), None)


def read(df: pd.DataFrame, unit: str | None, lens: Sequence[str] | None = None) -> dict[str, Any]:
    """Question 4's reading: ``turbotab.repeats.read``'s measurements, stated only when
    unambiguous (see the module docstring)."""
    from turbotab import repeats

    out = repeats.read(df, unit)
    if not unit or unit not in df.columns:
        return out
    dietary = "dietary" in (lens or [])
    gaps = out.get("spacing")
    index = out.get("replicate_index")
    order = _order_column(df, out)
    moving = trend(df, unit, order)
    period = period_column(df, unit)
    recalls = recall_evidence(df, out) if dietary else None
    evidence: list[str] = []
    reading, stated = None, False
    if gaps is not None and gaps["all_identical"]:
        reading, stated = repeats.REPEATS, True
        evidence.append(f"every one of a unit's records carries the same date in `{gaps['column']}`")
    elif gaps is not None:
        days = "day" if round(gaps["median_days"]) == 1 else "days"
        spaced = (f"a unit's records in `{gaps['column']}` are {gaps['median_days']:.0f} {days} "
                  f"apart at the median, ranging {gaps['min_days']:.0f} to {gaps['max_days']:.0f}")
        regular = gaps["cv"] <= repeats._SCHEDULE_MAX_CV
        if regular and gaps["median_days"] >= repeats._SCHEDULE_MIN_DAYS:
            evidence.append(spaced + ", regular enough to be a schedule")
            reading, stated = repeats.TIME_POINTS, recalls is None
            if recalls:
                evidence.append(f"`{recalls}` says they are dietary recalls, which are repeated "
                                f"measures of usual intake, so the spacing alone does not decide")
        elif regular:
            evidence.append(spaced + ": a short, regular schedule, which a time course and "
                                     "same-week replicates both follow")
        elif gaps["median_days"] < repeats._SCHEDULE_MIN_DAYS:
            evidence.append(spaced + ", close together and irregular")
            reading, stated = repeats.REPEATS, True
        else:
            evidence.append(spaced + ", which fits unscheduled encounters as well as repeats")
    elif index:
        evidence.append(f"`{index}` numbers each unit's records 1, 2, 3 and there is no date, "
                        f"which orders the records and says nothing about what the order means")
        if recalls:
            reading, stated = repeats.REPEATS, True
            evidence.append(f"under the dietary lens `{recalls}` reads as a recall number, and "
                            f"recalls are repeated measures of usual intake")
    if moving:
        evidence.append(f"`{moving['column']}` {moving['direction']} within units (Spearman ρ "
                        f"{moving['median_rho']:+.2f} at the median; {moving['share']:.0%} of "
                        f"{moving['n_units']} units)")
    if period:
        evidence.append(f"`{period}` changes within units, as a crossover's periods do")
    if (moving or period) and reading != repeats.TIME_POINTS:
        reading, stated = None, False   # time-point evidence against a repeats reading: asked
    out.update(reading=reading if stated else None, stated=stated,
               confidence=("medium" if stated else None), evidence=evidence,
               trend=moving, period_column=period, recall_column=recalls)
    out["sentence"] = repeats._sentence({**out, "reading": out["reading"], "evidence": evidence})
    if not stated:
        lead = "; ".join(evidence)
        out["sentence"] = (f"Asked rather than stated: {lead}. Which of the two it is decides "
                           f"whether averaging a unit's rows is correct." if lead else
                           "Asked rather than stated: there is no date column and nothing "
                           "numbering a unit's records, so nothing here says what varies between "
                           "them.")
    return out


__all__ = ["period_column", "read", "recall_evidence", "trend"]

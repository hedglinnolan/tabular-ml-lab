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

* **Spacing is weak evidence.** It never states a reading alone (audit WP14 repair: a pre/post
  design whose blood pressure fell 10 mmHg within every person, visits 10 ± 4 days apart, was
  stated "repeats, close together and irregular"). Records close together are asked about; only
  same-day records, or recalls under the dietary lens, state repeats.
* **Time-point evidence**: a measured value that trends within units (Spearman ρ with the order
  of at least 0.5 in absolute value, in the same direction in at least 70% of units with three or
  more records), or, with two records per unit, a change in the same direction in at least 70% of
  units that a two-sided sign test puts below 1%; or a column naming a treatment, arm, period or
  phase that changes within units (a crossover). Either one turns a "repeats" reading into a
  question.
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
PAIRED_SIGN_P = 0.01     # two records per unit: the sign test's two-sided p a change must reach
_PERIOD = re.compile(r"(?:^|_)(?:treatment|treat|trt|arm|period|phase|condition|intervention|"
                     r"regimen|sequence|diet_period|allocation)(?:$|_)", re.I)
_RECALL = re.compile(r"(?:^|_)(?:recall|recalls|recall_number|recall_no|recall_day|24h|24hr|"
                     r"day_of_recall|replicate|rep)(?:$|_|\d)", re.I)


def _order_column(df: pd.DataFrame, reading: dict[str, Any]) -> str | None:
    spacing = reading.get("spacing") or {}
    return spacing.get("column") or reading.get("replicate_index")


def _orders_rows(column: str) -> bool:
    """A column whose name says it orders or dates the rows (``recall_number``, ``visit``,
    ``visit_id``): it rises within units by construction, so it is no measured trend."""
    from turbotab.core.recognizers import id_kind, reads_as_time

    return reads_as_time(column) or id_kind(column) == "visit"


def trend(df: pd.DataFrame, unit: str, order: str | None) -> dict[str, Any] | None:
    """A measured column that rises or falls within units along ``order``, or None. Columns that
    order or date the rows themselves are not measurements and are skipped."""
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
                or pd.api.types.is_bool_dtype(df[c]) or _orders_rows(str(c)):
            continue
        rhos, changes = [], []
        for _, block in pd.DataFrame({"k": key, "v": df[c], "u": df[unit]}).dropna().groupby("u"):
            if block["v"].nunique() < 2 or block["k"].nunique() < 2:
                continue
            if len(block) == 2:  # two records: the change from the first to the second
                first, second = block.sort_values("k")["v"].to_numpy(dtype=float)
                changes.append(second - first)
                continue
            if len(block) < TREND_MIN_RECORDS:
                continue
            rho = stats.spearmanr(block["k"], block["v"])[0]
            if np.isfinite(rho):
                rhos.append(rho)
        found = None
        if len(rhos) >= 5 and len(rhos) >= len(changes):
            rhos_a = np.array(rhos)
            sign = np.sign(np.median(rhos_a))
            share = float(np.mean((np.abs(rhos_a) >= TREND_RHO) & (np.sign(rhos_a) == sign)))
            if sign != 0 and share >= TREND_UNITS:
                found = {"column": str(c), "share": share,
                         "direction": "falls" if sign < 0 else "rises",
                         "median_rho": float(np.median(rhos_a)), "n_units": len(rhos)}
        elif len(changes) >= 5:
            d = np.array(changes)
            d = d[d != 0]
            if len(d) >= 5:
                k = int((d > 0).sum())
                sign = 1.0 if k * 2 > len(d) else -1.0 if k * 2 < len(d) else 0.0
                share = float(max(k, len(d) - k) / len(d))
                p = float(stats.binomtest(k, len(d), 0.5).pvalue)
                if sign != 0 and share >= TREND_UNITS and p < PAIRED_SIGN_P:
                    # The median ρ of a two-record unit is ±1: the direction, said as a ρ.
                    found = {"column": str(c), "share": share,
                             "direction": "falls" if sign < 0 else "rises",
                             "median_rho": float(sign), "n_units": int(len(d)),
                             "paired": True, "sign_test_p": p}
        if found and (best is None or found["share"] > best["share"]):
            best = found
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


def _p(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"= {p:.3f}"


# A clock or an elapsed time within the day: ``time_min``, ``minutes``, ``hour``, ``tp`` (audit WP13
# gate repair: an OGTT drawn at 0–120 minutes on one morning was stated "repeated measurements",
# every record carrying the same date, and its mean recommended).
_CLOCK_WORDS = {"min", "mins", "minute", "minutes", "hr", "hrs", "hour", "hours", "sec", "secs",
                "second", "seconds", "time", "timepoint", "tp", "clock", "draw", "sample", "sampling",
                "postprandial", "elapsed"}


def within_day_order(df: pd.DataFrame, unit: str, date_column: str) -> str | None:
    """What orders a unit's same-date records within the day, or None: the date column's own
    clock times (records minutes or hours apart, which a day-resolution spacing reads as one
    date), or a column named as a time or an elapsed time that changes within units."""
    parsed = pd.to_datetime(df[date_column], errors="coerce", format="mixed")
    seconds = []
    for _, block in parsed.groupby(df[unit], dropna=True):
        values = block.dropna().sort_values()
        if len(values) >= 2:
            seconds.extend(np.diff(values.to_numpy()).astype("timedelta64[s]").astype(float))
    gaps = np.array(seconds, dtype=float)
    if len(gaps) and float(np.mean(gaps > 0)) >= 0.5:
        minutes = float(np.median(gaps[gaps > 0])) / 60.0
        return (f"`{date_column}` holds clock times {minutes:,.0f} minutes apart at the median "
                f"within a unit's day")
    from turbotab.core.recognizers import tokens

    for c in df.columns:
        if c in (unit, date_column) or not pd.api.types.is_numeric_dtype(df[c]) \
                or pd.api.types.is_bool_dtype(df[c]):
            continue
        if not set(tokens(c)) & _CLOCK_WORDS:
            continue
        varying = df.groupby(unit, dropna=True)[c].nunique(dropna=True)
        if len(varying) and float((varying > 1).mean()) >= 0.5:
            values = sorted(pd.to_numeric(df[c], errors="coerce").dropna().unique())
            shown = ", ".join(f"{v:g}" for v in values[:7]) + (" …" if len(values) > 7 else "")
            return f"`{c}` orders them within the day ({shown})"
    return None


_OCCASION_WORDS = {"visit", "visits", "wave", "waves", "followup", "round", "session", "timepoint",
                   "occasion", "cycle", "exam", "examination"}


def occasion_column(df: pd.DataFrame, unit: str) -> str | None:
    """A column that names each record's occasion as a visit, wave or follow-up (``visit``,
    ``visit_date``, ``wave``) and changes within units: the study's own word for a schedule of
    time points (BLUEPRINT §14: "a time column varies within units"; a ``wave`` the same on every
    row of a unit names no occasion of its rows)."""
    from turbotab.core.recognizers import TIME_VARIES, id_kind, tokens, within_unit_variation

    for c in df.columns:
        if c == unit:
            continue
        if id_kind(c) == "visit" or set(tokens(c)) & _OCCASION_WORDS:
            share = within_unit_variation(df[c], df[unit])
            if share is not None and share >= TIME_VARIES:
                return str(c)
    return None


def constant_dates(df: pd.DataFrame, unit: str) -> list[str]:
    """Date columns the same on every row of a unit (``dob``, ``randomization_date``): they date
    the unit, not its rows, and are never spacing evidence (BLUEPRINT §14 rule 1; the gate: the
    legacy reader kept the first date column on a tie and stated "repeats" from ``dob``)."""
    from turbotab import repeats
    from turbotab.core.recognizers import TIME_CONSTANT, within_unit_variation

    out = []
    for c in repeats._date_columns(df.drop(columns=[unit])):
        share = within_unit_variation(df[c], df[unit])
        if share is not None and share <= TIME_CONSTANT:
            out.append(str(c))
    return out


def intake_varies(df: pd.DataFrame, unit: str) -> str | None:
    """A dietary intake (total energy or a nutrient, by the one recognizer) that changes within
    units: the repeated rows are reports of intake, which the dietary lens reads as replicates of
    usual intake as readily as time points (NUTRITION_PACK)."""
    from turbotab.core.recognizers import is_nutrient, reads_as_total_energy

    for c in df.columns:
        if c == unit or not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if not (reads_as_total_energy(c) or is_nutrient(c)):
            continue
        varying = df.groupby(unit, dropna=True)[c].nunique(dropna=True)
        if len(varying) and float((varying > 1).mean()) >= 0.5:
            return str(c)
    return None


def read(df: pd.DataFrame, unit: str | None, lens: Sequence[str] | None = None) -> dict[str, Any]:
    """Question 4's reading: ``turbotab.repeats.read``'s measurements, stated only when
    unambiguous (see the module docstring)."""
    from turbotab import repeats

    if not unit or unit not in df.columns:
        return repeats.read(df, unit)
    dated_units = constant_dates(df, unit)
    # Same date is not the same moment: what orders a unit's records within that day (an OGTT's
    # ``time_min``) is read before the date leaves the spacing evidence.
    within_day = within_day_order(df, unit, dated_units[0]) if dated_units else None
    if dated_units:
        df = df.drop(columns=dated_units)
    out = repeats.read(df, unit)
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
        same = f"every one of a unit's records carries the same date in `{gaps['column']}`"
        within = within_day_order(df, unit, gaps["column"])
        if within:
            # Same date is not the same moment: a time course within the day (an OGTT, a meal
            # test) is as likely as duplicate measurements, so it is asked.
            evidence.append(f"{same}, but {within}, which a time course within the day follows "
                            f"as well as repeated measurements")
        else:
            reading, stated = repeats.REPEATS, True
            evidence.append(same)
    elif gaps is not None:
        days = "day" if round(gaps["median_days"]) == 1 else "days"
        spaced = (f"a unit's records in `{gaps['column']}` are {gaps['median_days']:.0f} {days} "
                  f"apart at the median, ranging {gaps['min_days']:.0f} to {gaps['max_days']:.0f}")
        regular = gaps["cv"] <= repeats._SCHEDULE_MAX_CV
        if regular and gaps["median_days"] >= repeats._SCHEDULE_MIN_DAYS:
            evidence.append(spaced + ", regular enough to be a schedule")
            # Spacing alone never states time points (audit WP13 gate repair: four 24-hour recalls
            # a season apart, the SEASONS design, were stated "time points: averaging them
            # destroys the signal"). The study naming its occasions as visits or waves is the second
            # signal, unless the dietary lens's repeated rows are intakes, replicates of usual
            # intake as readily as time points.
            occasion = occasion_column(df, unit)
            intake = intake_varies(df, unit) if dietary else None
            if recalls:
                evidence.append(f"`{recalls}` says they are dietary recalls, which are repeated "
                                f"measures of usual intake, so the spacing alone does not decide")
            elif intake:
                evidence.append(f"`{intake}` is an intake reported on each record, which the dietary "
                                f"lens reads as a replicate of usual intake as readily as a time "
                                f"point")
            elif occasion:
                reading, stated = repeats.TIME_POINTS, True
                evidence.append(f"`{occasion}` names each record's occasion, as a schedule of visits "
                                f"does")
            else:
                evidence.append("nothing names the records' occasions, so the spacing alone does "
                                "not decide")
        elif regular:
            evidence.append(spaced + ": a short, regular schedule, which a time course and "
                                     "same-week replicates both follow")
        elif gaps["median_days"] < repeats._SCHEDULE_MIN_DAYS:
            evidence.append(spaced + ", close together and irregular")
            if recalls:
                # Recalls days apart are the textbook replicates of usual intake (NUTRITION_PACK):
                # the name, not the spacing, states it.
                reading, stated = repeats.REPEATS, True
                evidence.append(f"under the dietary lens `{recalls}` reads as a recall number, "
                                f"and recalls are repeated measures of usual intake")
        else:
            evidence.append(spaced + ", which fits unscheduled encounters as well as repeats")
    elif index:
        evidence.append(f"`{index}` numbers each unit's records 1, 2, 3 and there is no date, "
                        f"which orders the records and says nothing about what the order means")
        occasion = occasion_column(df.drop(columns=[index]), unit)
        if recalls and occasion:
            # BLUEPRINT §14 rule 3: recalls numbered across occasions the study names (two at
            # baseline, two at month 6) are replicates within an occasion and time points across
            # them; averaging all four erases the change (the gate: an arm's 470 kcal fall).
            evidence.append(f"`{recalls}` reads as a recall number, but `{occasion}` names each "
                            f"record's occasion and changes within units, as time points do")
        elif recalls:
            reading, stated = repeats.REPEATS, True
            evidence.append(f"under the dietary lens `{recalls}` reads as a recall number, and "
                            f"recalls are repeated measures of usual intake")
    if moving and moving.get("paired"):
        evidence.append(f"`{moving['column']}` {moving['direction']} from a unit's first record to "
                        f"its second in {moving['share']:.0%} of {moving['n_units']} units (sign "
                        f"test p {_p(moving['sign_test_p'])})")
    elif moving:
        evidence.append(f"`{moving['column']}` {moving['direction']} within units (Spearman ρ "
                        f"{moving['median_rho']:+.2f} at the median; {moving['share']:.0%} of "
                        f"{moving['n_units']} units)")
    if period:
        evidence.append(f"`{period}` changes within units, as a crossover's periods do")
    if dated_units:
        listed = " and ".join(f"`{c}`" for c in dated_units[:3])
        same = (f"{listed} {'is' if len(dated_units) == 1 else 'are'} the same on every row of a "
                f"unit")
        evidence.append(f"{same}, but {within_day}, which a time course within the day follows as "
                        f"well as repeated measurements" if within_day else
                        f"{same}, which dates the unit (a birth, enrollment or visit day), not the "
                        f"order of its rows")
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


__all__ = ["constant_dates", "occasion_column", "period_column", "read", "recall_evidence",
           "trend"]

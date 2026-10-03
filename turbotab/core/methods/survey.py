"""The survey answer, applied: the analysis weight for pooled cycles, and the design the fit uses
(AUDIT_REPORT §5 WP10; the question and its refusals are :mod:`turbotab.core.survey`).

**Pooled cycles** (the minor D20/G20). NHANES Analytic Guidelines 2011–2016 §3.1.3: "When combining
two or more 2-year cycles from 2001–2002 onward, new multi-year sample weights can be computed by
simply dividing the 2-year sample weights by the number of 2-year cycles in the analysis." §3.1.4:
"When combining data from the 1999-2000 NHANES cycle with other cycles, it is recommended that the
4-year sample weights be used for 1999-2002 and the 2-year sample weights be used for other cycles.
To use both the 4-year sample weight for 1999-2002 and 2-year sample weights for other cycles, the
4-year sample weight needs to be doubled prior to analysis … then divide the doubled 4-year
1999-2002 sample weight and the 2-year weights for the 2003-2004 cycles, by 3, the number of
cycles; the resulting sample weight will be a 6-year weight." Table F: "4 years 1999-2002 Provided
on the Public-use Data Files". :func:`analysis_weights` builds exactly that weight.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from turbotab.core.survey import (
    GUIDELINES,
    DesignReading,
    cycle_number,
    offered,
    reading_of,
    sample_concern,
)


@dataclass
class PooledWeight:
    weights: np.ndarray  # per row, aligned with the frame
    note: str | None  # how it was built, for the caption and the sentence
    refusal: str | None = None
    cycles: list[Any] = field(default_factory=list)


def _plain(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return int(value)
    return value


def analysis_weights(frame: pd.DataFrame, weight: str, cycle: str | None = None,
                     four_year: str | None = None) -> PooledWeight:
    """The weight the estimate uses, by the Analytic Guidelines §3.1.3–3.1.4.

    One cycle: the weight as it is. ``k`` pooled cycles from 2001–2002 on: the 2-year weight ÷ k.
    With 1999–2000 among them: ``2 × four_year ÷ k`` on the 1999–2000 and 2001–2002 rows and the
    2-year weight ÷ k on the others (which is the four-year weight itself when only 1999–2002 is
    pooled). 1999–2000 pooled without the four-year weight is refused: "the 2-year weights for
    1999–2000 and 2001–2002 are not comparable" (§3.1.4).
    """
    w2 = pd.to_numeric(frame[weight], errors="coerce").to_numpy(dtype=float)
    if cycle is None or cycle not in frame.columns:
        return PooledWeight(w2, None)
    values = frame[cycle]
    levels = sorted(values.dropna().unique().tolist(), key=lambda v: (str(type(v)), v))
    k = len(levels)
    if k <= 1:
        return PooledWeight(w2, None, cycles=levels)
    first_two = np.array([cycle_number(v, cycle) in (1, 2) for v in values], dtype=bool)
    shown = ", ".join(f"`{_plain(v)}`" for v in levels[:6]) + (" …" if k > 6 else "")
    if not any(cycle_number(v, cycle) == 1 for v in levels):
        return PooledWeight(w2 / k, f"`{weight}` ÷ {k} over the {k} pooled cycles ({shown})",
                            cycles=levels)
    if four_year is None or four_year not in frame.columns:
        return PooledWeight(
            w2 / k, None, cycles=levels,
            refusal=(f"`{cycle}` pools 1999–2000 with other cycles, and the 2-year weights for "
                     f"1999–2000 and 2001–2002 are not comparable ({GUIDELINES} §3.1.4): the "
                     f"1999–2002 rows need their four-year weight."))
    w4 = pd.to_numeric(frame[four_year], errors="coerce").to_numpy(dtype=float)
    missing = first_two & ~np.isfinite(w4)
    if missing.any():
        return PooledWeight(
            w2 / k, None, cycles=levels,
            refusal=(f"`{four_year}` is missing on {int(missing.sum()):,} of the 1999–2002 rows, so "
                     f"their four-year weight cannot be used ({GUIDELINES} §3.1.4)."))
    out = np.where(first_two, 2.0 * w4 / k, w2 / k)
    if first_two[values.notna().to_numpy()].all():
        note = f"`{four_year}`, the four-year weight NCHS provides for 1999–2002 ({GUIDELINES} §3.1.4)"
    else:
        note = (f"2 × `{four_year}` ÷ {k} on the 1999–2002 rows and `{weight}` ÷ {k} on the "
                f"others, over the {k} pooled cycles ({shown}; {GUIDELINES} §3.1.4)")
    return PooledWeight(out, note, cycles=levels)


# ── the fit's view of the answer ─────────────────────────────────────────────

UNANSWERED = ("Survey design columns are in this table, and whether the estimates describe the "
              "surveyed population or these participants is not answered, so no coefficient is "
              "reported.")


@dataclass
class FitSurvey:
    """What the fit does with the survey answer, under inference."""

    answer: str | None  # "population" | "sample" | None (not answered)
    design: Any = None  # turbotab.core.models.survey.SurveyDesign, for "population"
    refusal: str | None = None
    exits: tuple[dict[str, Any], ...] = ()
    reading: DesignReading = field(default_factory=DesignReading)

    def concern(self) -> str | None:
        return sample_concern(self.reading) if self.answer == "sample" else None


def for_fit(state: Any, store: Any, unit_column: str | None = None) -> FitSurvey | None:
    """The survey answer as the fit applies it under inference; None when there is no design.

    ``store`` is the working table's DataStore: a population design is read over every row of it,
    so rows outside the analysis keep their strata and PSUs (a domain). ``unit_column`` names which
    rows belong together (a person's repeated rows), for a design with no PSU column: each unit is
    then one sampling unit rather than each row.
    """
    from turbotab.core.models.survey import build_design

    reading = reading_of(state)
    spec = getattr(state, "survey", None)
    if spec is None:
        if not reading.present:
            return None
        return FitSurvey(None, refusal=UNANSWERED, reading=reading,
                         exits=({"label": "Answer the survey question: the surveyed population, "
                                          "or these participants", "decision": None},))
    if spec.estimand == "sample":
        return FitSurvey("sample", reading=reading)
    unit = unit_column if (unit_column and unit_column in set(getattr(store, "columns", ()) or ())) \
        else None
    columns = [c for c in (spec.weight, spec.strata, spec.psu, spec.cycle, spec.four_year_weight,
                           unit) if c]
    frame = store.materialize(list(dict.fromkeys(columns)), None)
    if unit is not None and spec.psu is not None:
        # A unit's rows are one person's: the PSU totals hold all of them only if they share a PSU.
        placed = frame.dropna(subset=[unit, spec.psu, *([spec.strata] if spec.strata else [])])
        keys = placed[spec.psu].astype(str)
        if spec.strata:
            keys = placed[spec.strata].astype(str) + "\x1f" + keys
        spread = keys.groupby(placed[unit]).nunique()
        if (spread > 1).any():
            return FitSurvey(
                "population", reading=reading,
                refusal=(f"{int((spread > 1).sum()):,} `{unit}` units have rows in more than one "
                         f"PSU, so the design cannot hold a unit's rows together; the strata and "
                         f"PSU columns or the unit column are not what they seem."),
                exits=({"label": "Check the survey question's strata and PSU", "decision": None},))
        unit = None  # the PSU already holds each unit's rows together
    pooled = analysis_weights(frame, spec.weight, spec.cycle, spec.four_year_weight)
    if pooled.refusal:
        return FitSurvey("population", refusal=pooled.refusal, reading=reading,
                         exits=({"label": "Name the four-year weight (the survey question)",
                                 "decision": None},))
    unit_codes = None
    if unit is not None:
        keys = frame[unit].astype(object).to_numpy().copy()
        missing = frame[unit].isna().to_numpy()
        keys[missing] = [f"__missing_{rid}" for rid in np.asarray(frame.index)[missing]]
        unit_codes = pd.factorize(pd.Series(keys, dtype=object))[0]
    design = build_design(frame, pooled.weights, weight_column=spec.weight,
                          strata_column=spec.strata, psu_column=spec.psu, unit_codes=unit_codes,
                          unit_column=unit, weight_note=pooled.note)
    return FitSurvey("population", design=design, reading=reading)


# ── what the question offers on this table ───────────────────────────────────


def proposal(state: Any, store: Any) -> dict[str, Any] | None:
    """The survey question's options on this table (the ``proposals`` artifact's ``survey``), or
    None when no column reads as a survey weight. A cycle column holding more than one cycle is
    named in each population option, with the four-year weight its rule needs."""
    reading = reading_of(state)
    if not reading.present:
        return None
    pooled_cycle = None
    cycles: list[Any] = []
    for c in reading.cycles:
        if c in set(getattr(store, "columns", ()) or ()):
            values = store.materialize([c], None)[c].dropna().unique().tolist()
            if len(values) > 1:
                pooled_cycle = c
                cycles = sorted((_plain(v) for v in values), key=lambda v: (str(type(v)), v))
                break
    frame = None
    columns = set(getattr(store, "columns", ()) or ())
    roles = getattr(state, "roles", None) or {}
    target = getattr(state, "target", None)
    read = [c for c in [*([target] if target else []),
                        *(c for c, r in roles.items() if r in ("exposure", "covariate", "energy")),
                        *reading.weights] if c in columns]
    if reading.weights and 0 < len(read) <= 500:
        # The values place a subsample's variables the names cannot (the least-common-denominator
        # rule; audit WP13 gate repair).
        frame = store.materialize(list(dict.fromkeys(read)), None)
    placed = None
    unplaced = [c for c in reading.unplaced if c in columns]
    if unplaced:
        # Design columns the user set by role, which no name reads: their values say which is a
        # weight, and which are codes (BLUEPRINT §14.1).
        from turbotab.core.survey import place_design

        frame = store.materialize(unplaced, None)
        placed = place_design(frame, unplaced)
    return {"weights": reading.weights or list((placed or {}).get("weights") or []),
            "strata": reading.strata, "psu": reading.psu,
            "cycle": pooled_cycle, "cycles": [str(v) for v in cycles],
            "four_year": reading.four_year,
            "options": offered(state, pooled_cycle, frame, placed)}


__all__ = ["FitSurvey", "PooledWeight", "UNANSWERED", "analysis_weights", "for_fit", "proposal"]

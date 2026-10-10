"""Stacking NHANES cycles into one table with one survey design (C1b; engine only until C1 asks it).

Each NHANES release is a nationally representative sample on its own, and any set of releases can be
stacked to estimate over a longer period or to follow a measure across them (D6,
:mod:`turbotab.core.methods.cycle_trends`). NHANES Analytic Guidelines 2011–2016 §2.6 names what a
stack must do: "be aware of sample design changes", "verify that data items collected in all
combined years are comparable in wording, methods, and inclusion/exclusions", "select the proper
weight to use for the combined dataset", and "examine the inherent assumption of no trend in the
estimate over the time period being combined". This module does the first three, refuses what it
cannot do, and leaves the fourth to the trend tests.

**Which release each file is.** A cycle is named by its release code (``SDDSRVYR``) or its years.
The codes are NCHS's (the DEMO documentation of each release, and Table F of the Guidelines): 1 for
1999–2000, 2 for 2001–2002, and so on two years at a time to 10 for 2017–2018; 66 for the
2017–March 2020 prepandemic file; 12 for August 2021–August 2023 (the DEMO_L documentation: "NHANES
August 2021-August 2023 public release"). A two-year label ("2015–2016") is a two-year cycle. Any
other cycle (a code not listed, a span that is not two years) is refused until its length in years
is given: the pooled weight needs it. A cycle whose years are given but whose start is not known is
stacked with a concern: whether it overlaps another cycle or leaves a gap cannot be checked.

**August 2021–August 2023 stands alone.** NCHS's weighting tutorial: "It is generally not
recommended to combine the August 2021-August 2023 cycle with other cycles given the 1.5-year gap
between this cycle and the 2017-March 2020 cycle", a gap in which the pandemic disrupted health
care, work and schooling. Stacked with any other cycle it is refused, with two exits: analyze it
alone, or leave it out.

**The pooled weight.** Each cycle's weight W represents that cycle's period; stacked, each is scaled
by the share of the stacked years its cycle covers, W × y_c ÷ T (y_c the cycle's years, T their
sum). This is NCHS's rule written once:

* two-year cycles from 2001–2002 on: "new multi-year sample weights can be computed by simply
  dividing the 2-year sample weights by the number of 2-year cycles in the analysis" (Guidelines
  §3.1.3; 2 ÷ 2k = 1 ÷ k);
* 1999–2000 with any other cycle: the 2-year weights of 1999–2000 and 2001–2002 "are not
  comparable", so the 1999–2002 rows take the four-year weight, doubled, and every weight is then
  divided by the number of cycles (§3.1.4; Table F's "If sddsrvyr in (1,2) then MEC6YR = 2/3 *
  WTMEC4YR": 2 × WTMEC4YR × 2 ÷ 6). Without the four-year weight it is refused;
* the 2017–March 2020 file covers 3.2 years: "If SDDSRVYR = 9 then MEC52Y = (2/5.2) • WTMEC2YR; If
  SDDSRVYR = 66 then MEC52Y = (3.2/5.2) • WTMECPRP" (Akinbami et al. 2022, "Combining Survey
  Cycles"). Its weights are the ``…PRP`` / ``…PP`` columns; a two-year weight named for every
  cycle is read as its prepandemic counterpart there (``WTMEC2YR`` → ``WTMECPRP``), and the
  substitution is recorded.

The 2017–March 2020 file holds the 2017–2018 participants (Akinbami et al. 2022: the 2019–March
2020 data "were combined with the data from the 2017–2018 cycle"), renumbered: its ``SEQN`` run
from 109263, past 2017–2018's 93703–102956 (the public DEMO files). Stacked with 2017–2018 it
would count them twice, and the identifiers would not show it, so cycles whose periods overlap are
refused; identifiers (``SEQN``) that appear in two cycles are refused too. A weight of another
kind in one cycle (the interview weight beside the examination weight) describes another sample
and is refused; the four-year weight the 1999–2002 rows take is checked with the others, so an
interview four-year weight beside examination two-year weights is refused too. A weight name the
registry does not know cannot be checked, and the stack says so.

Rows of a cycle with a blank stratum, PSU or weight have no place in the survey design; they are
stacked, counted, and the stack says how many every estimate will leave out.

**Strata and PSUs stay distinct across cycles.** A stratum is the pair (cycle, stratum), a PSU the
triple (cycle, stratum, PSU), coded afresh, so two cycles that reuse a stratum's number are never
merged into one stratum. The public DEMO files number the masked variance strata afresh in each
release (1–13 in 1999–2000, 14–28 in 2001–2002, …, 134–148 in 2017–2018, 149–172 in 2017–March
2020), so a number seen in two cycles is noticed: the table is not as NCHS released it, or the
files are not NHANES.

**Variables across cycles.** A variable is stacked under one name. A rename the caller declares
(``renames``) and a conversion the caller declares (``recodes``: a value map, or a factor for a
change of unit) are applied and recorded. What the data show is flagged, never changed: a variable
missing from a cycle (with a likely earlier name, when one cycle has a similar name no other cycle
has), codes that differ between cycles (a recode?), and a typical value that differs by a factor of
five or more (a change of unit?). A measurement the caller declares incompatible across cycles
(``incompatible``: a changed assay, question or eligibility, from the release documentation) is
refused unless a conversion for it is declared; a variable asked for that a cycle lacks is refused
too. Each refusal says why in plain words and gives its exits: leave the variable out, stack only
the cycles that have it alike, or declare the rename or the conversion.

Nothing here learns from the data: the weights are a formula of the cycles' lengths, the codes are
labels, and the flags change no value. The design over the stacked table is
:meth:`Stacked.design`.
"""
from __future__ import annotations

import difflib
import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

SOURCES = {
    "guidelines": "NHANES Analytic Guidelines 2011–2016, §2.6 and §3.1.3–3.1.4 (NCHS)",
    "prepandemic": ("Akinbami et al. 2022, Vital Health Stat 2(190), \"Combining Survey Cycles\" "
                    "(doi:10.15620/cdc:115434)"),
    "documentation": "NHANES DEMO documentation, SDDSRVYR (the data release cycle)",
    "tutorial": ("NHANES weighting tutorial, Constructing Weights for Combined NHANES Survey "
                 "Cycles (NCHS)"),
}

# SDDSRVYR -> (label, first year, years covered). 1–9: Table F of the Guidelines; 10, 66: the DEMO
# documentation of 2017–2018 ("NHANES 2017-2018 public release") and of the prepandemic file
# ("NHANES 2017-2020 public release"; 3.2 years, Akinbami et al. 2022); 12: the DEMO_L
# documentation ("NHANES August 2021-August 2023 public release"), a two-year period from August.
RELEASES: dict[int, tuple[str, float, float]] = {
    **{k: (f"{1999 + 2 * (k - 1)}–{2000 + 2 * (k - 1)}", float(1999 + 2 * (k - 1)), 2.0)
       for k in range(1, 11)},
    66: ("2017–March 2020", 2017.0, 3.2),
    12: ("August 2021–August 2023", 2021.0 + 7 / 12, 2.0),
}
PREPANDEMIC = 66
# The release NCHS advises against combining with any other (the weighting tutorial).
STANDS_ALONE = 12
# The 1999–2002 releases, whose rows take the four-year weight when 1999–2000 is stacked.
FOUR_YEAR_RELEASES = (1, 2)
# A two-year weight and its counterpart on the prepandemic file (the names NCHS publishes;
# :data:`turbotab.core.survey.NHANES_WEIGHTS`).
PREPANDEMIC_WEIGHT = {"WTMEC2YR": "WTMECPRP", "WTINT2YR": "WTINTPRP", "WTDRD1": "WTDRD1PP",
                      "WTDR2D": "WTDR2DPP", "WTSAF2YR": "WTSAFPRP"}
# What sample a weight describes, for the check that every cycle uses one kind.
WEIGHT_KINDS = {
    "WTMEC2YR": "examination", "WTMEC4YR": "examination", "WTMECPRP": "examination",
    "WTINT2YR": "interview", "WTINT4YR": "interview", "WTINTPRP": "interview",
    "WTDRD1": "dietary day-one", "WTDR4YR": "dietary day-one", "WTDRD1PP": "dietary day-one",
    "WTDR2D": "dietary two-day", "WTDR2DPP": "dietary two-day",
    "WTSAF2YR": "fasting subsample", "WTSAF4YR": "fasting subsample", "WTSAFPRP": "fasting subsample",
}
# The 1999–2002 four-year weight of each kind, offered when the one named is of another kind.
FOUR_YEAR_OF_KIND = {"examination": "WTMEC4YR", "interview": "WTINT4YR",
                     "dietary day-one": "WTDR4YR", "fasting subsample": "WTSAF4YR"}
SCALE_FACTOR = 5.0  # a typical value this many times another cycle's is flagged as a unit change
MAX_CODES = 12  # a whole-number variable with at most this many values in every cycle reads as codes

# The columns the stacked table adds.
CYCLE = "cycle"
RELEASE = "cycle_release"
START = "cycle_start"
YEARS = "cycle_years"
MIDPOINT = "cycle_midpoint"
WEIGHT = "pooled_weight"
SOURCE_WEIGHT = "cycle_weight"
STRATUM = "stratum_id"
PSU = "psu_id"
ADDED = (CYCLE, RELEASE, START, YEARS, MIDPOINT, WEIGHT, SOURCE_WEIGHT, STRATUM, PSU)


class StackRefused(ValueError):
    """The cycles cannot be stacked as asked; the message says why in plain words (the technical
    term last, as a quiet label), ``exits`` the ways forward, each the change to the call it makes."""

    def __init__(self, message: str, exits: Sequence[Mapping[str, Any]] = ()):
        super().__init__(message)
        self.exits = [dict(e) for e in exits]


# ── which cycle a file is ────────────────────────────────────────────────────


@dataclass(frozen=True)
class Cycle:
    key: Any  # as the caller named it
    label: str
    start: float  # the first year
    years: float  # the years its weight represents
    release: int | None = None  # SDDSRVYR, when known

    @property
    def end(self) -> float:
        return self.start + self.years

    @property
    def midpoint(self) -> float:
        """The cycle's midpoint in years: NCHS's time value for intervals (Ingram et al. 2018,
        Guideline 4: 1999–2000 is 2000)."""
        return self.start + self.years / 2


_SPAN = re.compile(r"^\s*(\d{4})\s*[-–—_/ ]\s*(?:([A-Za-z]+)\.?\s+)?(\d{2,4})\s*(.*)$")
_AUG2021 = re.compile(r"^\s*(?:Aug(?:ust)?\.?\s+)?2021\s*[-–—_/]\s*(?:Aug(?:ust)?\.?\s+)?(?:20)?23\b",
                      re.I)


def cycle_of(key: Any, years: float | None = None) -> Cycle:
    """The cycle ``key`` names: a release code (``SDDSRVYR``) or its years ("2015–2016",
    "2017–March 2020", "2017–2020 prepandemic"). ``years`` gives the length of a cycle the code
    or the label does not fix; a cycle whose length is not known is refused."""
    if isinstance(key, Cycle):
        return key
    code = _code(key)
    if code is not None:
        if code in RELEASES:
            label, start, span = RELEASES[code]
            return Cycle(key, label, start, float(years) if years else span, code)
        if years:
            return Cycle(key, f"release {code}", math.nan, float(years), code)
        raise StackRefused(
            f"Release {code} is not one whose length is known here, and the pooled weight needs "
            f"how many years each cycle covers. (Survey term: SDDSRVYR {code}.)",
            ({"label": f"Say how many years release {code} covers", "years": {key: None}},))
    text = str(key).strip()
    if re.search(r"pre-?pandemic|^P$", text, re.I) or re.match(r"^2017\s*[-–—]\s*(?:March\s+)?2020",
                                                               text, re.I):
        label, start, span = RELEASES[PREPANDEMIC]
        return Cycle(key, label, start, float(years) if years else span, PREPANDEMIC)
    if _AUG2021.match(text):
        label, start, span = RELEASES[STANDS_ALONE]
        return Cycle(key, label, start, float(years) if years else span, STANDS_ALONE)
    m = _SPAN.match(text)
    if m:
        first, month, last = int(m.group(1)), m.group(2), m.group(3)
        last_year = int(last) if len(last) == 4 else 100 * (first // 100) + int(last)
        release = next((c for c, (_, s, _y) in RELEASES.items()
                        if c != PREPANDEMIC and s == first and last_year == first + 1), None)
        if years:
            return Cycle(key, text, float(first), float(years), release)
        if last_year == first + 1 and not month:
            return Cycle(key, f"{first}–{last_year}", float(first), 2.0, release)
        raise StackRefused(
            f"`{text}` does not span the two years of a standard NHANES cycle, so how many years "
            "its weight represents is not known; the pooled weight needs it.",
            ({"label": f"Say how many years `{text}` covers", "years": {key: None}},))
    if years:
        first = re.search(r"(?<!\d)(\d{4})(?!\d)", text)
        return Cycle(key, text, float(first.group(1)) if first else math.nan, float(years))
    raise StackRefused(
        f"`{text}` names no NHANES cycle (a release code such as 9, or years such as 2015–2016).",
        ({"label": f"Say which cycle `{text}` is and how many years it covers",
          "years": {key: None}},))


def _code(key: Any) -> int | None:
    if isinstance(key, (bool, np.bool_)):
        return None
    if isinstance(key, (int, np.integer)):
        return int(key)
    if isinstance(key, (float, np.floating)) and float(key).is_integer():
        return int(key)
    if isinstance(key, str) and re.fullmatch(r"\s*\d{1,3}\s*", key):
        return int(key)
    return None


# ── what the stack records ───────────────────────────────────────────────────


@dataclass(frozen=True)
class Flag:
    """What the stack noticed or did about one variable across the cycles."""

    column: str  # its stacked name
    kind: str  # renamed · converted · absent · codes_differ · scale_differs
    cycles: tuple[str, ...]  # the cycles it concerns, by label
    says: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class Stacked:
    """The stacked table and how it was made. ``frame``'s rows are numbered anew (0 … n − 1), in
    cycle order; it holds the stacked variables and the columns :data:`ADDED` names."""

    frame: pd.DataFrame
    cycles: tuple[Cycle, ...]
    total_years: float
    factors: dict[str, float]  # cycle label -> what its weight W was multiplied by
    weights: dict[str, str]  # cycle label -> the weight column it took
    four_year: str | None = None  # the four-year weight the 1999–2002 rows took, if any
    variables: tuple[str, ...] = ()
    flags: list[Flag] = field(default_factory=list)  # renames, conversions and what was noticed
    left_out: list[Flag] = field(default_factory=list)  # variables not in every cycle, not asked
    notes: list[str] = field(default_factory=list)
    concerns: list[str] = field(default_factory=list)

    @property
    def labels(self) -> list[str]:
        return [c.label for c in self.cycles]

    def design(self) -> Any:
        """The survey design over every stacked row: the pooled weight, the strata and PSUs kept
        distinct by cycle (:class:`~turbotab.core.models.survey.SurveyDesign`)."""
        from turbotab.core.models.survey import build_design

        return build_design(self.frame, self.frame[WEIGHT].to_numpy(dtype=float),
                            weight_column=WEIGHT, strata_column=STRATUM, psu_column=PSU,
                            weight_note=self.weight_note)

    @property
    def weight_note(self) -> str:
        parts = []
        for c in self.cycles:
            share = f"({_num(c.years)}/{_num(self.total_years)})"
            if self.four_year and c.release in FOUR_YEAR_RELEASES:
                parts.append(f"{share} × 2 × `{self.four_year}` on {c.label}")
            else:
                parts.append(f"{share} × `{self.weights[c.label]}` on {c.label}")
        return (f"{'; '.join(parts)} (each cycle's weight × its years ÷ the {_num(self.total_years)} "
                "years stacked)")

    def sentence(self) -> str:
        return stack_sentence(self)


# ── stacking ─────────────────────────────────────────────────────────────────


def stack_cycles(files: Mapping[Any, pd.DataFrame], *, weight: str | Mapping[Any, str],
                 strata: str = "SDMVSTRA", psu: str = "SDMVPSU",
                 four_year: str | Mapping[Any, str] | None = None,
                 variables: Sequence[str] | None = None, exclude: Sequence[str] = (),
                 renames: Mapping[Any, Mapping[str, str]] | None = None,
                 recodes: Mapping[Any, Mapping[str, Any]] | None = None,
                 incompatible: Mapping[str, Mapping[Any, str]] | None = None,
                 years: Mapping[Any, float] | None = None,
                 id_column: str | None = "SEQN") -> Stacked:
    """Stack ``files`` (cycle -> its table) into one table with one design (module docstring).

    ``weight`` is the weight column, one name for every cycle or one per cycle; ``four_year`` the
    four-year weight the 1999–2002 rows take when 1999–2000 is stacked. ``variables`` are the
    columns to stack (default: every column every cycle has, after ``renames``), ``exclude`` those
    to leave out. ``renames`` maps a cycle to {its name: the stacked name}; ``recodes`` maps a
    cycle to {stacked name: a value map, or a factor}; ``incompatible`` maps a stacked name to
    {cycle: why it was measured differently there}. ``years`` gives the length of a cycle its key
    does not fix. Raises :class:`StackRefused` with its exits."""
    if not files:
        raise ValueError("no files to stack")
    years = dict(years or {})
    cycles = [cycle_of(k, years.get(k)) for k in files]
    keyed = dict(zip([c.label for c in cycles], files))
    if len(set(keyed)) < len(cycles):
        same = sorted({c.label for c in cycles if [d.label for d in cycles].count(c.label) > 1})
        raise StackRefused(f"Two files are the same cycle ({', '.join(same)}); a cycle is "
                           "stacked once.", ({"label": "Keep one file per cycle"},))
    order = sorted(range(len(cycles)),
                   key=lambda i: (math.isnan(cycles[i].start),
                                  0.0 if math.isnan(cycles[i].start) else cycles[i].start,
                                  cycles[i].years))
    cycles = [cycles[i] for i in order]
    keys = [list(files)[i] for i in order]
    _refuse_overlaps(cycles, keys)
    _refuse_standing_alone(cycles, keys)
    renames = renames or {}
    recodes = recodes or {}
    incompatible = incompatible or {}
    tables: list[pd.DataFrame] = []
    for c, k in zip(cycles, keys):
        frame = files[k]
        if not isinstance(frame, pd.DataFrame):
            raise ValueError(f"the file for {c.label} is not a table")
        tables.append(frame.rename(columns=dict(renames.get(k, {}))))
    flags: list[Flag] = []
    for c, k in zip(cycles, keys):
        for old, new in dict(renames.get(k, {})).items():
            flags.append(Flag(new, "renamed", (c.label,),
                              f"`{old}` in {c.label} is stacked as `{new}` (a rename declared for "
                              "this stack).", {"from": old}))
    weights_used, notes = _weights_by_cycle(weight, cycles, keys, tables)
    _refuse_missing_design(cycles, tables, strata, psu)
    four = _four_year(four_year, cycles, keys, tables)
    unchecked = _refuse_weight_kinds(cycles, tables, weights_used, four)
    if id_column:
        _refuse_repeated_ids(cycles, keys, tables, id_column)
    total = float(sum(c.years for c in cycles))
    design_cols = {strata, psu, *weights_used.values(), *([four] if four else []),
                   *([id_column] if id_column else []), "SDDSRVYR"}
    present = [set(t.columns) for t in tables]
    everywhere = set.intersection(*present) - design_cols
    excluded = set(exclude)
    if variables is None:
        stacked_vars = [v for v in tables[0].columns if v in everywhere and v not in excluded]
        others = sorted((set.union(*present) - everywhere - design_cols - excluded),
                        key=lambda v: (v not in tables[0].columns, v))
        left_out = [_absent_flag(v, cycles, keys, tables, present) for v in others]
    else:
        stacked_vars = [v for v in dict.fromkeys(variables) if v not in excluded]
        clash = [v for v in stacked_vars if v in ADDED]
        if clash:
            raise ValueError(f"{clash} are names the stacked table adds; rename them first")
        for v in stacked_vars:
            if v in design_cols:
                continue
            missing = [i for i, p in enumerate(present) if v not in p]
            if missing:
                _refuse_absent(v, cycles, keys, tables, present, missing)
        stacked_vars = [v for v in stacked_vars if v not in design_cols]
        left_out = []
    for v in [v for v in stacked_vars if v in ADDED]:
        raise ValueError(f"`{v}` is a name the stacked table adds; rename it first")
    tables, converted = _convert(cycles, keys, tables, recodes, stacked_vars)
    flags.extend(converted)
    _refuse_incompatible(cycles, keys, stacked_vars, incompatible, recodes)
    flags.extend(_noticed(cycles, tables, stacked_vars,
                          {(f.column, f.cycles[0]) for f in converted}))
    frame, factors, collided, unplaced = _assemble(cycles, tables, stacked_vars, weights_used,
                                                   four, strata, psu, id_column, total)
    concerns = []
    if unchecked:
        shown = "; ".join(f"{lab}: `{name}`" for lab, name in unchecked.items())
        concerns.append(f"The weights {shown} are not among the NHANES weights known here, so "
                        "whether every cycle's weight describes the same sample (the examination, "
                        "the interview, a dietary or a fasting subsample) could not be checked. "
                        "Check that each is the same kind of weight. (Survey term: weight kind "
                        "unverified.)")
    if unplaced:
        n = sum(unplaced.values())
        by = "; ".join(f"{lab}: {k:,}" for lab, k in unplaced.items())
        concerns.append(f"{n:,} row{'s have' if n != 1 else ' has'} a blank survey layer, cluster or "
                        f"weight ({by}), so {'they have' if n != 1 else 'it has'} no place in the "
                        "survey design and every estimate over the stack will leave "
                        f"{'them' if n != 1 else 'it'} out. (Survey terms: blank `{strata}`, "
                        f"`{psu}` or weight.)")
    unknown_start = [c.label for c in cycles if math.isnan(c.start)]
    if unknown_start and len(cycles) > 1:
        concerns.append(f"When {_listed(unknown_start)} began is not known here, so whether "
                        f"{'it overlaps' if len(unknown_start) == 1 else 'they overlap'} another "
                        "cycle (and would count the same people or years twice) or leaves a gap "
                        "could not be checked. Check the release documentation.")
    if collided:
        collided = sorted(collided, key=lambda v: (str(type(v)), v))
        shown = ", ".join(str(_plain(v)) for v in collided[:6]) + (" …" if len(collided) > 6 else "")
        concerns.append(f"Survey layers numbered {shown} appear in more than one cycle. NCHS "
                        "numbers the layers afresh in each release, so these files may not be as "
                        "NCHS released them; each cycle's layers are kept apart all the same. "
                        f"(Survey term: `{strata}` values, the masked variance strata, repeat "
                        "across cycles.)")
    gaps = [(a, b) for a, b in zip(cycles, cycles[1:]) if b.start - a.end > 1e-9]
    if gaps:
        between = "; ".join(f"between {a.label} and {b.label}" for a, b in gaps)
        concerns.append(f"The cycles are not adjacent ({between}). NCHS describes stacking adjacent "
                        f"cycles; a pooled estimate here averages over the {_num(total)} years the "
                        f"cycles cover, not over the years between them ({SOURCES['guidelines']}).")
    if len(cycles) > 1:
        notes.append("A single estimate over the stacked cycles assumes the measure did not change "
                     "across them (NHANES Analytic Guidelines 2011–2016 §2.6): the trend tests "
                     "across cycles check it.")
    return Stacked(frame=frame, cycles=tuple(cycles), total_years=total, factors=factors,
                   weights=weights_used, four_year=four, variables=tuple(stacked_vars),
                   flags=flags, left_out=left_out, notes=notes, concerns=concerns)


def _refuse_overlaps(cycles: list[Cycle], keys: list[Any]) -> None:
    for i, a in enumerate(cycles):
        for b in cycles[i + 1:]:
            if math.isnan(a.start) or math.isnan(b.start):
                continue
            if a.start < b.end - 1e-9 and b.start < a.end - 1e-9:
                whole = a if a.years >= b.years else b
                part = b if whole is a else a
                raise StackRefused(
                    f"{whole.label} and {part.label} cover the same years: "
                    + (f"the {whole.label} file already holds the {part.label} participants, so "
                       "stacking both would count them twice."
                       if PREPANDEMIC in (whole.release, part.release) else
                       "stacking both would count the same period twice."),
                    ({"label": f"Leave {part.label} out", "drop_cycles": [keys[cycles.index(part)]]},
                     {"label": f"Leave {whole.label} out",
                      "drop_cycles": [keys[cycles.index(whole)]]}))


def _weights_by_cycle(weight: str | Mapping[Any, str], cycles: list[Cycle], keys: list[Any],
                      tables: list[pd.DataFrame]) -> tuple[dict[str, str], list[str]]:
    used: dict[str, str] = {}
    notes: list[str] = []
    for c, k, t in zip(cycles, keys, tables):
        name = weight.get(k, weight.get(c.label)) if isinstance(weight, Mapping) else weight
        if name is None:
            raise StackRefused(f"No weight is named for {c.label}.",
                               ({"label": f"Name the weight column for {c.label}",
                                 "weight": {k: None}},))
        if name not in t.columns and c.release == PREPANDEMIC and name in PREPANDEMIC_WEIGHT \
                and PREPANDEMIC_WEIGHT[name] in t.columns:
            notes.append(f"{c.label} has no `{name}`; its counterpart on the prepandemic file, "
                         f"`{PREPANDEMIC_WEIGHT[name]}`, is used (Akinbami et al. 2022).")
            name = PREPANDEMIC_WEIGHT[name]
        if name not in t.columns:
            offered = [col for col in t.columns if re.match(r"^WT", str(col), re.I)]
            raise StackRefused(
                f"The {c.label} file has no `{name}` column, so its rows have no weight."
                + (f" Its weight columns: {', '.join(f'`{o}`' for o in offered[:6])}." if offered
                   else ""),
                tuple({"label": f"Use `{o}` for {c.label}", "weight": {k: o}} for o in offered[:3])
                + ({"label": f"Leave {c.label} out", "drop_cycles": [k]},))
        values = pd.to_numeric(t[name], errors="coerce")
        if (values < 0).any():
            raise StackRefused(f"`{name}` in {c.label} has weights below zero; a survey weight "
                               "counts people and cannot be negative.",
                               ({"label": f"Check `{name}` in {c.label}"},))
        used[c.label] = name
    return used, notes


def _refuse_missing_design(cycles: list[Cycle], tables: list[pd.DataFrame], strata: str,
                           psu: str) -> None:
    for c, t in zip(cycles, tables):
        lacking = [col for col in (strata, psu) if col not in t.columns]
        if lacking:
            raise StackRefused(
                f"The {c.label} file has no {' or '.join(f'`{x}`' for x in lacking)}, so its rows "
                "cannot be placed in the survey design and their standard errors could not be "
                "computed. (Survey terms: the masked variance strata and PSUs.)",
                ({"label": "Join the demographics file of that cycle first",
                  "stage": "ingest"},))


def _refuse_weight_kinds(cycles: list[Cycle], tables: list[pd.DataFrame], used: dict[str, str],
                         four: str | None) -> dict[str, str]:
    """Refuse weights of different samples, the four-year weight of the 1999–2002 rows among them;
    return the weights (cycle label -> name) whose kind is not known, which could not be checked."""
    taken = {c.label: (four if four and c.release in FOUR_YEAR_RELEASES else used[c.label])
             for c in cycles}
    kinds = {lab: WEIGHT_KINDS.get(str(name).upper()) for lab, name in taken.items()}
    known = {k for k in kinds.values() if k}
    if len(known) > 1:
        listed = "; ".join(f"{c.label}: `{taken[c.label]}` ({kinds[c.label] or 'unknown'})"
                           for c in cycles)
        exits: list[dict[str, Any]] = []
        if four:
            early = [t for c, t in zip(cycles, tables) if c.release in FOUR_YEAR_RELEASES]
            later = {kinds[c.label] for c in cycles if c.release not in FOUR_YEAR_RELEASES} - {None}
            for k in sorted(later):
                match = FOUR_YEAR_OF_KIND.get(k)
                if match and match != four and all(match in t.columns for t in early):
                    exits.append({"label": f"Use the {k} four-year weight `{match}` on 1999–2002",
                                  "four_year": match})
        exits.extend({"label": f"Use the {k} weight in every cycle", "weight_kind": k}
                     for k in sorted(known))
        raise StackRefused(
            "The cycles' weights describe different samples (" + listed + "): stacked, the "
            "estimate would mix them. Every cycle needs the weight of the same sample"
            + (", the four-year weight of the 1999–2002 rows included." if four else ".")
            + " (Survey term: weight kinds differ.)", exits)
    return {lab: name for lab, name in taken.items() if kinds[lab] is None}


def _refuse_standing_alone(cycles: list[Cycle], keys: list[Any]) -> None:
    """August 2021–August 2023 with any other cycle: refused, as NCHS advises (the weighting
    tutorial)."""
    if len(cycles) < 2:
        return
    alone = [(c, k) for c, k in zip(cycles, keys) if c.release == STANDS_ALONE]
    if not alone:
        return
    c, k = alone[0]
    others = [kk for cc, kk in zip(cycles, keys) if cc.release != STANDS_ALONE]
    raise StackRefused(
        f"NCHS advises against combining {c.label} with other cycles: a year and a half passed "
        "between the end of the 2017–March 2020 file and its start, in which the pandemic "
        "disrupted health care, work and schooling, so a stack would assume those unobserved "
        "months looked like the observed ones (NHANES weighting tutorial, NCHS). "
        "(Survey term: combining across the 2020–2021 gap.)",
        ({"label": f"Analyze {c.label} alone", "drop_cycles": others},
         {"label": f"Leave {c.label} out", "drop_cycles": [k]}))


def _refuse_repeated_ids(cycles: list[Cycle], keys: list[Any], tables: list[pd.DataFrame],
                         id_column: str) -> None:
    seen: dict[Any, str] = {}
    for c, k, t in zip(cycles, keys, tables):
        if id_column not in t.columns:
            continue
        for v in pd.unique(t[id_column].dropna()):
            if v in seen and seen[v] != c.label:
                other = seen[v]
                raise StackRefused(
                    f"`{id_column}` {_plain(v)} appears in both {other} and {c.label}: the same "
                    "participant would be counted twice.",
                    ({"label": f"Leave {c.label} out", "drop_cycles": [k]},
                     {"label": f"Leave {other} out",
                      "drop_cycles": [keys[[d.label for d in cycles].index(other)]]}))
            seen.setdefault(v, c.label)


def _four_year(four_year: str | Mapping[Any, str] | None, cycles: list[Cycle], keys: list[Any],
               tables: list[pd.DataFrame]) -> str | None:
    """The four-year weight the 1999–2002 rows take: only when 1999–2000 is stacked with another
    cycle (Guidelines §3.1.4)."""
    if len(cycles) < 2 or not any(c.release == 1 for c in cycles):
        return None
    early = [(c, k, t) for c, k, t in zip(cycles, keys, tables) if c.release in FOUR_YEAR_RELEASES]
    names = set()
    for c, k, t in early:
        name = four_year.get(k, four_year.get(c.label)) if isinstance(four_year, Mapping) else \
            four_year
        if name is None or name not in t.columns:
            raise StackRefused(
                "1999–2000 is stacked with other cycles, and its two-year weights and those of "
                "2001–2002 rest on different censuses, so they cannot be pooled "
                f"({SOURCES['guidelines']}): the 1999–2002 rows need the four-year weight"
                + (f", and the {c.label} file has no `{name}`." if name else "."),
                ({"label": "Name the four-year weight (WTMEC4YR or WTINT4YR)", "four_year": None},
                 {"label": "Leave 1999–2000 out",
                  "drop_cycles": [kk for cc, kk in zip(cycles, keys) if cc.release == 1]}))
        values = pd.to_numeric(t[name], errors="coerce")
        lacking = int(values.isna().sum())
        if lacking:
            raise StackRefused(
                f"`{name}` is blank on {lacking:,} rows of {c.label}, so their four-year weight "
                f"cannot be used ({SOURCES['guidelines']}).",
                ({"label": "Leave 1999–2000 out",
                  "drop_cycles": [kk for cc, kk in zip(cycles, keys) if cc.release == 1]},))
        names.add(name)
    if len(names) > 1:
        raise StackRefused("The 1999–2000 and 2001–2002 files name different four-year weights "
                           f"({', '.join(sorted(names))}); they are one weight over both cycles.",
                           ({"label": "Name one four-year weight", "four_year": None},))
    return names.pop()


def _similar(column: str, candidates: Sequence[str]) -> str | None:
    best, score = None, 0.0
    for c in candidates:
        s = difflib.SequenceMatcher(None, column.upper(), str(c).upper()).ratio()
        if s > score:
            best, score = str(c), s
    return best if score >= 0.75 else None


def _candidate(column: str, i: int, tables: list[pd.DataFrame], present: list[set]) -> str | None:
    """A column only cycle ``i`` has whose name is like ``column``: an earlier name?"""
    others = set().union(*(p for j, p in enumerate(present) if j != i))
    return _similar(column, [c for c in tables[i].columns if c not in others])


def _absent_flag(column: str, cycles: list[Cycle], keys: list[Any], tables: list[pd.DataFrame],
                 present: list[set]) -> Flag:
    missing = [i for i, p in enumerate(present) if column not in p]
    hints = {cycles[i].label: h for i in missing if (h := _candidate(column, i, tables, present))}
    shown = ", ".join(cycles[i].label for i in missing)
    says = f"`{column}` is not in {shown}, so it is not stacked"
    if hints:
        says += " (" + "; ".join(f"{lab} has `{h}`: an earlier name?" for lab, h in hints.items()) + ")"
    return Flag(column, "absent", tuple(cycles[i].label for i in missing), says + ".",
                {"candidates": hints})


def _refuse_absent(column: str, cycles: list[Cycle], keys: list[Any], tables: list[pd.DataFrame],
                   present: list[set], missing: list[int]) -> None:
    flag = _absent_flag(column, cycles, keys, tables, present)
    hints = flag.detail["candidates"]
    exits: list[dict[str, Any]] = []
    for i in missing:
        hint = hints.get(cycles[i].label)
        if hint:
            exits.append({"label": f"Stack `{hint}` in {cycles[i].label} as `{column}`",
                          "renames": {keys[i]: {hint: column}}})
    exits.append({"label": f"Leave `{column}` out", "exclude": [column]})
    having = [keys[i] for i in range(len(cycles)) if i not in missing]
    if len(having) >= 1:
        exits.append({"label": "Stack only the cycles that have it",
                      "drop_cycles": [keys[i] for i in missing]})
    raise StackRefused(f"`{column}` is not in {', '.join(cycles[i].label for i in missing)}"
                       + (" (" + "; ".join(f"{lab} has `{h}`: an earlier name?"
                                           for lab, h in hints.items()) + ")" if hints else "")
                       + ", so a stacked column would be blank for a whole cycle.", exits)


def _convert(cycles: list[Cycle], keys: list[Any], tables: list[pd.DataFrame],
             recodes: Mapping[Any, Mapping[str, Any]], variables: Sequence[str]
             ) -> tuple[list[pd.DataFrame], list[Flag]]:
    out, flags = [], []
    for c, k, t in zip(cycles, keys, tables):
        plan = dict(recodes.get(k, recodes.get(c.label, {})) or {})
        if not plan:
            out.append(t)
            continue
        t = t.copy()
        for column, how in plan.items():
            if column not in t.columns:
                raise ValueError(f"a conversion is declared for `{column}` in {c.label}, which "
                                 "has no such column")
            if isinstance(how, Mapping):
                before = t[column]
                t[column] = before.map(lambda v, m=dict(how): m.get(v, v) if pd.notna(v) else v)
                shown = ", ".join(f"{_plain(a)} → {_plain(b)}" for a, b in list(how.items())[:6])
                flags.append(Flag(column, "converted", (c.label,),
                                  f"`{column}` in {c.label} was recoded ({shown}) to match the "
                                  "other cycles (a conversion declared for this stack).",
                                  {"map": {str(a): b for a, b in how.items()}}))
            elif isinstance(how, (int, float, np.integer, np.floating)) and math.isfinite(how):
                t[column] = pd.to_numeric(t[column], errors="coerce") * float(how)
                flags.append(Flag(column, "converted", (c.label,),
                                  f"`{column}` in {c.label} was multiplied by {_num(float(how))} to "
                                  "the other cycles' unit (a conversion declared for this stack).",
                                  {"factor": float(how)}))
            else:
                raise ValueError(f"the conversion of `{column}` in {c.label} is neither a value "
                                 "map nor a finite factor")
        out.append(t)
    return out, flags


def _refuse_incompatible(cycles: list[Cycle], keys: list[Any], variables: Sequence[str],
                         incompatible: Mapping[str, Mapping[Any, str]],
                         recodes: Mapping[Any, Mapping[str, Any]]) -> None:
    labels = {k: c.label for c, k in zip(cycles, keys)}
    labels.update({c.label: c.label for c in cycles})
    for column in variables:
        declared = dict(incompatible.get(column, {}) or {})
        unresolved = {}
        for where, why in declared.items():
            if where not in labels:
                continue
            k = next((kk for c, kk in zip(cycles, keys) if where in (kk, c.label)), where)
            plan = recodes.get(k, recodes.get(labels[where], {})) or {}
            if column not in plan:
                unresolved[labels[where]] = why
        if not unresolved:
            continue
        reasons = "; ".join(f"{lab}: {why}" for lab, why in unresolved.items())
        alike = [kk for c, kk in zip(cycles, keys) if c.label not in unresolved]
        apart = [kk for c, kk in zip(cycles, keys) if c.label in unresolved]
        exits = [{"label": f"Leave `{column}` out", "exclude": [column]}]
        if alike:
            exits.append({"label": "Stack only the cycles measured alike", "drop_cycles": apart})
        exits.append({"label": f"Declare the documented conversion of `{column}`",
                      "recodes": {kk: {column: None} for kk in apart}})
        raise StackRefused(
            f"`{column}` was not measured the same way in every cycle ({reasons}), so values from "
            "different cycles would not mean the same thing side by side. (Measurement not "
            "comparable across cycles.)", exits)


def _noticed(cycles: list[Cycle], tables: list[pd.DataFrame], variables: Sequence[str],
             converted: set[tuple[str, str]]) -> list[Flag]:
    """Codes that differ between cycles, and a typical value that moves by a factor of five or more:
    flagged, never changed."""
    flags: list[Flag] = []
    for column in variables:
        if any((column, c.label) in converted for c in cycles):
            continue
        series = [pd.to_numeric(t[column], errors="coerce") if column in t.columns else
                  pd.Series(dtype=float) for t in tables]
        if not all(s.notna().any() for s in series):
            continue
        whole = all(np.allclose(s.dropna(), np.round(s.dropna())) for s in series)
        codes = [set(np.round(s.dropna()).astype(np.int64).tolist()) for s in series]
        if whole and all(len(c) <= MAX_CODES for c in codes):
            union = set().union(*codes)
            if any(c != union for c in codes):
                detail = {c.label: sorted(code) for c, code in zip(cycles, codes)}
                odd = [cy.label for cy, code in zip(cycles, codes) if code != union]
                flags.append(Flag(column, "codes_differ", tuple(odd),
                                  f"`{column}` takes different codes in different cycles ("
                                  + "; ".join(f"{lab}: {', '.join(map(str, v))}"
                                              for lab, v in detail.items())
                                  + "): check the release documentation for a recode.",
                                  {"codes": detail}))
            continue
        medians = [float(s.median()) for s in series]
        if all(m > 0 for m in medians) and max(medians) / min(medians) >= SCALE_FACTOR:
            hi = cycles[int(np.argmax(medians))].label
            lo = cycles[int(np.argmin(medians))].label
            flags.append(Flag(column, "scale_differs", (hi, lo),
                              f"The typical `{column}` in {hi} is {_num(max(medians) / min(medians))} "
                              f"times that in {lo}: a change of unit? Check the release "
                              "documentation.", {"medians": dict(zip([c.label for c in cycles],
                                                                     medians))}))
    return flags


def _assemble(cycles: list[Cycle], tables: list[pd.DataFrame], variables: Sequence[str],
              weights: dict[str, str], four: str | None, strata: str, psu: str,
              id_column: str | None, total: float
              ) -> tuple[pd.DataFrame, dict[str, float], list[Any], dict[str, int]]:
    parts, factors = [], {}
    unplaced: dict[str, int] = {}
    seen_strata: dict[Any, str] = {}
    collided: list[Any] = []
    for c, t in zip(cycles, tables):
        keep = [col for col in dict.fromkeys([*([id_column] if id_column and id_column in t.columns
                                                else []), *variables, strata, psu])]
        part = t[keep].copy()
        if four is not None and c.release in FOUR_YEAR_RELEASES:
            w = 2.0 * pd.to_numeric(t[four], errors="coerce").to_numpy(dtype=float)
            source = four
            factor = 2.0 * c.years / total
        else:
            w = pd.to_numeric(t[weights[c.label]], errors="coerce").to_numpy(dtype=float)
            source = weights[c.label]
            factor = c.years / total
        factors[c.label] = factor
        part[CYCLE] = c.label
        part[RELEASE] = c.release if c.release is not None else np.nan
        part[START] = c.start
        part[YEARS] = c.years
        part[MIDPOINT] = c.midpoint
        part[SOURCE_WEIGHT] = pd.to_numeric(t[source], errors="coerce").to_numpy(dtype=float)
        part[WEIGHT] = w * (c.years / total)
        blank = int((part[strata].isna() | part[psu].isna() | np.isnan(w)).sum())
        if blank:
            unplaced[c.label] = blank
        for v in pd.unique(t[strata].dropna()):
            if v in seen_strata and seen_strata[v] != c.label and v not in collided:
                collided.append(v)
            seen_strata.setdefault(v, c.label)
        parts.append(part)
    frame = pd.concat(parts, ignore_index=True)
    placed = frame[strata].notna() & frame[psu].notna()
    stratum_keys = pd.MultiIndex.from_arrays([frame.loc[placed, CYCLE],
                                              frame.loc[placed, strata].astype(object)])
    psu_keys = pd.MultiIndex.from_arrays([frame.loc[placed, CYCLE],
                                          frame.loc[placed, strata].astype(object),
                                          frame.loc[placed, psu].astype(object)])
    frame[STRATUM] = np.nan
    frame[PSU] = np.nan
    frame.loc[placed, STRATUM] = pd.factorize(stratum_keys)[0]
    frame.loc[placed, PSU] = pd.factorize(psu_keys)[0]
    return frame, factors, collided, unplaced


# ── words ────────────────────────────────────────────────────────────────────


def _num(v: float) -> str:
    if float(v).is_integer():
        return f"{int(v)}"
    return f"{v:.4g}"


def _plain(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return int(value)
    return value


def _listed(items: Sequence[str]) -> str:
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def stack_sentence(s: Stacked | None = None) -> str:
    """The methods sentence of a stack of NHANES cycles: which cycles, the pooled weight and its
    rule, the strata kept apart, and what was renamed or converted."""
    if s is None:
        return ("NHANES cycles were stacked into one sample; each cycle's weight was multiplied by "
                "the share of the stacked years its cycle covers (NCHS Analytic Guidelines "
                "2011–2016 §3.1.3–3.1.4; Akinbami et al. 2022 for the 3.2-year 2017–March 2020 "
                "file), and strata and PSUs were kept distinct by cycle.")
    years = _num(s.total_years)
    sentence = (f"NHANES {_listed(s.labels)} were stacked into one sample of {len(s.frame):,} rows "
                f"covering {years} years")
    if len(s.cycles) > 1:
        sentence += (f"; each cycle's weight was multiplied by its years over the {years} stacked "
                     "(NCHS Analytic Guidelines 2011–2016 §3.1.3")
        if s.four_year:
            sentence += f", with the doubled four-year weight `{s.four_year}` on 1999–2002 (§3.1.4)"
        sentence += (")" if not any(c.release == PREPANDEMIC for c in s.cycles) else
                     "; Akinbami et al. 2022 for the 3.2-year 2017–March 2020 file)")
    sentence += ", and strata and PSUs were kept distinct by cycle."
    renamed = [f for f in s.flags if f.kind in ("renamed", "converted")]
    if renamed:
        sentence += (" Harmonized across cycles: "
                     + "; ".join(f.says.rstrip(".").split(" (a ")[0] for f in renamed) + ".")
    return sentence


# ── the contract ─────────────────────────────────────────────────────────────


def _register_contract() -> None:
    from turbotab.core.contracts import (CONTRACTS, ContractOption, MethodContract, Relation,
                                        register_contract)

    if "stack_cycles" in CONTRACTS:
        return
    here = "turbotab.core.methods.cycles"
    both = ("prediction", "inference")

    register_contract(MethodContract(
        key="stack_cycles", label="Stacking NHANES cycles into one sample", slot="ingest",
        scope="row_local", package="C1b", run_order=0.5,
        scope_note=("A stacked row's values are its own (renamed or converted by a rule declared "
                    "for its cycle); its pooled weight is its own weight times its cycle's share of "
                    "the stacked years, a formula of the cycles' lengths, never of another row's "
                    "values. What it flags (codes or a typical value that differ between cycles) "
                    "reads every row but changes none and informs no modeling choice."),
        needs=("two or more NHANES files, each one cycle, with the same weight kind",
               "each file's masked variance strata and PSUs (SDMVSTRA, SDMVPSU)",
               "the four-year weight, when 1999–2000 is stacked with another cycle",
               "optional: renames and conversions declared from the release documentation, and "
               "measurements declared incompatible across cycles"),
        question="Stack these NHANES cycles into one sample?",
        place="before the opening sequence (the ingest stage builds the table)",
        options=(
            ContractOption(
                "stack", "Stack the cycles, each weight scaled by its cycle's share of the years",
                "NCHS's rule for pooled cycles (NHANES Analytic Guidelines 2011–2016 §3.1.3–3.1.4; "
                "Akinbami et al. 2022)",
                dict.fromkeys(both, "Sound when every stacked variable was measured alike in "
                                    "every cycle; the cycles' strata and PSUs stay distinct, so "
                                    "the standard errors keep each cycle's design"),
                dict.fromkeys(both, "recommended"), dict.fromkeys(both, 0)),
            ContractOption(
                "one_cycle", "Analyze one cycle alone",
                "Customary when a measurement changed between cycles",
                dict.fromkeys(both, "Sound always; fewer rows, so wider intervals"),
                dict.fromkeys(both, "available"), dict.fromkeys(both, 1)),
        ),
        storyboard=("name each file's cycle and its years", "refuse cycles that overlap or repeat "
                    "a participant", "rename and convert as declared, flag what differs",
                    "scale each weight by its cycle's share of the stacked years",
                    "keep each cycle's strata and PSUs distinct"),
        relations=(
            Relation("implies", "pooled_weights",
                     "each cycle's weight is multiplied by its years over the stacked years: 1 ÷ k "
                     "for k two-year cycles, and 3.2 over the stacked years for the 3.2-year "
                     "prepandemic file",
                     when=("stack",), condition="two or more cycles",
                     enforced_by=f"{here}:stack_cycles", id="pooled_weight", purposes=both),
            Relation("implies", "strata_by_cycle",
                     "strata and PSUs are coded by cycle, so no two cycles share one",
                     when=("stack",), condition="always", enforced_by=f"{here}:stack_cycles",
                     id="strata_kept_apart", purposes=both),
            Relation("conflicts", "four_year_weight_needed",
                     "1999–2000 stacked with another cycle needs the four-year weight on the "
                     "1999–2002 rows", when=("stack",), rung="refused",
                     exits=("name the four-year weight", "leave 1999–2000 out"),
                     condition="1999–2000 and another cycle, no four-year weight",
                     enforced_by=f"{here}:stack_cycles", id="four_year", purposes=both),
            Relation("conflicts", "overlapping_cycles",
                     "cycles covering the same years (2017–2018 and the prepandemic file, which "
                     "holds its participants) are refused: the same participants would count "
                     "twice", when=("stack",), rung="refused",
                     exits=("leave one of them out",), condition="two cycles whose years overlap, "
                     "or an identifier in two cycles", enforced_by=f"{here}:stack_cycles",
                     id="overlap", purposes=both),
            Relation("conflicts", "weight_kinds_differ",
                     "weights of different samples (examination and interview) are refused, the "
                     "four-year weight of the 1999–2002 rows among them; a weight whose kind is not "
                     "known is stacked with a concern that it could not be checked",
                     when=("stack",), rung="refused",
                     exits=("use one kind of weight in every cycle",
                            "name the four-year weight of the same kind"),
                     condition="two kinds of weight among the cycles, the four-year weight included",
                     enforced_by=f"{here}:stack_cycles", id="weight_kind", purposes=both),
            Relation("conflicts", "combined_across_the_pandemic_gap",
                     "the 2021–2023 cycle (release 12) stacked with any other cycle is refused, as "
                     "the NHANES weighting tutorial advises: a year and a half of unobserved "
                     "pandemic months lies between it and the prepandemic file",
                     when=("stack",), rung="refused",
                     exits=("analyze the 2021–2023 cycle alone", "leave it out"),
                     condition="the 2021–2023 cycle and another cycle",
                     enforced_by=f"{here}:stack_cycles", id="stands_alone", purposes=both),
            Relation("conflicts", "measurement_not_comparable",
                     "a variable declared to be measured differently in a cycle is refused unless "
                     "its conversion is declared", when=("stack",), rung="refused",
                     exits=("leave the variable out", "stack only the cycles measured alike",
                            "declare the documented conversion"),
                     condition="a measurement declared incompatible across cycles",
                     enforced_by=f"{here}:stack_cycles", id="incompatible", purposes=both),
            Relation("conflicts", "variable_missing_in_a_cycle",
                     "a variable asked for that a cycle lacks is refused, with a likely earlier name "
                     "when one is found", when=("stack",), rung="refused",
                     exits=("declare the rename", "leave the variable out",
                            "stack only the cycles that have it"),
                     condition="a requested variable absent from a cycle",
                     enforced_by=f"{here}:stack_cycles", id="absent", purposes=both),
            Relation("implies", "differences_flagged",
                     "codes that differ between cycles and a typical value that moves by a factor "
                     "of five or more are flagged, never changed", when=("stack",),
                     condition="always", enforced_by=f"{here}:stack_cycles", id="flags",
                     purposes=both),
            Relation("enables", "cycle_trends",
                     "a stack of three or more cycles can be tested for a trend across them, which "
                     "checks the no-trend assumption a pooled estimate makes", when=("stack",),
                     condition="two or more cycles stacked", enforced_by=f"{here}:stack_cycles",
                     id="trends", purposes=both),
        ),
        sources=tuple(SOURCES.values()),
        sentence=f"{here}:stack_sentence"))


_register_contract()

__all__ = ["ADDED", "Cycle", "Flag", "RELEASES", "StackRefused", "Stacked", "cycle_of",
           "stack_cycles", "stack_sentence"]

"""Findings that can be acted on (M2_CONTRACT §4): the repair registry, its SQL, its validators,
its previews, and the memory that lets a finding say it was answered.

The registry
------------
A *family* is one kind of repair and the findings it answers. Each finding of a family carries
its options in the ``findings`` artifact (``finding["repairs"]``)::

    RepairOption { key, label (≤ 4 words), consequence (≤ 16 words), row_local, effect,
                   sentence, decision }

``decision`` is the ``apply_repair`` the option records, its ``params`` filled in from the data,
so the client previews and records it as it is. ``sentence`` is the methods sentence the option
would earn. ``effect`` says what the repair changes:

==================  =========================================  =================================
family              findings                                   options (effect)
==================  =========================================  =================================
impossible values   ``pack::clinical::impossible_vs_extreme``  ``set_missing`` (values) ·
                                                               ``exclude_rows`` (rows) ·
                                                               ``unusable`` (columns)
sentinel codes      ``sentinel_missing__<col>``,               ``set_missing`` (values)
                    ``pack::survey::sentinel_codes``
SAS zeros           ``sas_zeros`` (detected here)              ``zero`` (values)
binary text         ``binary_text__<col>``, written as text,   ``level``: one option per level,
                    not the outcome                            which becomes 1 (values)
energy in kJ        ``pack::dietary::atwater``, verdict        ``to_kcal`` (values)
                    ``energy_in_kj``
text numbers        ``text_numbers__<col>`` (detected here)    ``read_numbers``, or ``thousands``
                                                               and ``decimal_comma`` (values)
below detection     ``below_detection__<col>`` (detected       ``half_limit``, ``limit_root2``
                    here)                                      (values)
infinite values     ``infinite_values`` (detected here)        ``set_missing`` (values)
date reading        ``ambiguous_dates`` (detected here)        ``month_first``, ``day_first``
                                                               (values)
==================  =========================================  =================================

Where each effect executes
--------------------------
* **values** — :func:`column_expressions` gives one DuckDB SQL expression per column, which the
  ``working`` stage selects in place of the column (M2_CONTRACT §2). Two repairs on one column
  compose in a fixed order: text read as numbers, then SAS zeros, then codes, then units, then
  levels, infinities and impossible bands (the families' ``priority``).
* **rows** — :func:`exclusion_rules`: the impossible values become range rules the participant
  flow applies after the eligibility answer's own, so the rows leave in the flow diagram before
  the seal, and no row id moves.
* **columns** — :func:`unusable_columns`: the columns leave the predictors
  (``decisions.left_out``).

Every family here passes the constitution's litmus (ROADMAP lockbox constitution §06: does row
*i*'s output depend on any other row? No), so ``row_local`` is true and each executes on the data
as soon as it is recorded. A statistical repair (``row_local: false``) would be recorded and run
in-fold instead; none is in M2.

A disposition is self-contained: its ``params`` say exactly what it does, so a repair the Record
says was applied stays applied when a later answer (a lens) stops raising its finding.

Memory
------
:func:`annotate` adds ``disposition`` and ``answered_by`` (a record id) to each finding when the
server serves the artifact: a finding is answered by the record of its own disposition (applied,
deferred or dismissed), else by the answer to the question it routes to when that answer matches
it, else by another finding's applied repair that already does what one of its options would.
Deferred findings resurface in the interview step they were deferred to
(``InterviewStep.deferred_findings``, :func:`deferred_to`).

Importing this module registers the validators, the completion, the preview builder for
``apply_repair`` and the key view of the ``findings`` slot.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from turbotab.core.decisions import (
    ROW_ID,
    ApplyRepair,
    DeferFinding,
    DismissFinding,
    ExclusionRule,
    FindingDisposition,
    Refusal,
    register_completion,
    register_validator,
)

LABEL_WORDS = 4
CONSEQUENCE_WORDS = 16
NO_LEVER = "No control for this yet."

# A zero in a SAS transport (XPT) file is an IBM float; read back without converting it, it is
# 16 ** -65 = 5.397605346934028e-79. The band is that value and nothing else a table measures.
SAS_ZERO = 5.397605346934028e-79
SAS_LOW, SAS_HIGH = 5.3976e-79, 5.3977e-79
KJ_PER_KCAL = 4.184
# NUTRITION_PACK §02's plausible-intake cut-offs, marked on the converted energy distribution.
INTAKE_MARKS = (500.0, 5000.0)

Effect = Literal["values", "rows", "columns"]


class RepairOption(BaseModel):
    """One way to act on a finding, and the ``apply_repair`` it records."""

    model_config = ConfigDict(extra="forbid", json_schema_serialization_defaults_required=True)

    key: str
    label: str  # ≤ 4 words
    consequence: str  # ≤ 16 words, on this table's own numbers
    row_local: bool = True  # executes on the data now; False: recorded and run in-fold
    effect: Effect  # what it changes: cell values, which rows are in, which columns are predictors
    sentence: str  # the methods sentence it would earn
    decision: ApplyRepair


# ── small SQL and text helpers ───────────────────────────────────────────────


def ident(name: str) -> str:
    """A quoted SQL identifier; any column name survives it."""
    return '"' + str(name).replace('"', '""') + '"'


def lit_str(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def lit_num(value: Any) -> str:
    x = float(value)
    if not math.isfinite(x):
        raise ValueError(f"{value!r} is not a finite number")
    return repr(x)


def _tick(value: Any) -> str:
    return f"`{value}`"


def _count(n: int) -> str:
    return f"`{int(n):,}`"


def _num(value: Any) -> str:
    from turbotab.core.voice import number

    return number(value)


def _amount(value: Any) -> str:
    """A number for a mark or a caption: ``30000.0`` → ``30,000``, ``0.4`` → ``0.4``."""
    v = float(value)
    return f"{int(v):,}" if v.is_integer() else f"{v:,.6g}"


def _plural(n: int, one: str, many: str | None = None) -> str:
    return one if n == 1 else (many or one + "s")


def _names(columns: Sequence[str], limit: int = 2) -> str:
    """```a```, ```a`` and ``b```, or ```a`` and 3 more columns``."""
    shown = [_tick(c) for c in columns]
    if len(shown) == 1:
        return shown[0]
    if len(shown) <= limit:
        return ", ".join(shown[:-1]) + " and " + shown[-1]
    rest = len(shown) - (limit - 1)
    return ", ".join(shown[: limit - 1]) + f" and {rest} more"


def _words(text: str) -> int:
    return len(str(text).split())


def is_sas_zero(values: Any) -> np.ndarray:
    a = np.abs(np.asarray(values, dtype=float))
    with np.errstate(invalid="ignore"):
        return (a >= SAS_LOW) & (a <= SAS_HIGH)


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float)


# ── the families ─────────────────────────────────────────────────────────────


@dataclass
class OfferContext:
    """What an offer may read: the table the finding was computed on, and the outcome."""

    frame: pd.DataFrame
    target: str | None = None

    @property
    def n_rows(self) -> int:
        return int(len(self.frame))

    def has(self, column: str | None) -> bool:
        return column is not None and column in self.frame.columns


Wrap = Callable[[str], str]  # a SQL expression of the column so far -> the repaired expression


@dataclass(frozen=True)
class Family:
    key: str
    priority: int  # the order two repairs on one column compose in, lowest first
    offer: Callable[[dict[str, Any], dict[str, Any], OfferContext], list[RepairOption]]
    effects: dict[str, Effect]  # option key -> effect
    values: Callable[[str, Mapping[str, Any]], dict[str, Wrap]] = lambda option, params: {}
    rows: Callable[[str, Mapping[str, Any]], list[ExclusionRule]] = lambda option, params: []
    columns: Callable[[str, Mapping[str, Any]], list[str]] = lambda option, params: []
    # What an option does, as (column, token) pairs: two options with the same marks do the same.
    marks: Callable[[str, Mapping[str, Any]], set[tuple[str, str]]] = lambda option, params: set()


def _option(finding: dict[str, Any], key: str, label: str, consequence: str, sentence: str,
            effect: Effect, params: dict[str, Any]) -> RepairOption:
    from turbotab.core.voice import finish

    return RepairOption(
        key=key, label=finish(label, terminal=False), consequence=finish(consequence),
        effect=effect, sentence=finish(sentence),
        decision=ApplyRepair(finding_id=str(finding["id"]), option=key, params=params),
    )


# impossible values ─────────────────────────────────────────────────────────


def _bands(params: Mapping[str, Any]) -> dict[str, tuple[float | None, float | None]]:
    out = {}
    for column, band in dict(params.get("bands") or {}).items():
        lo, hi = (list(band) + [None, None])[:2]
        out[str(column)] = (None if lo is None else float(lo), None if hi is None else float(hi))
    return out


def _outside(values: pd.Series, lo: float | None, hi: float | None) -> pd.Series:
    x = _numeric(values)
    out = pd.Series(False, index=values.index)
    if lo is not None:
        out |= x < lo
    if hi is not None:
        out |= x > hi
    return out


def _offer_impossible(finding: dict[str, Any], p: dict[str, Any], oc: OfferContext) -> list[RepairOption]:
    entries = [e for e in p.get("columns") or [] if isinstance(e, Mapping)
               and oc.has(e.get("column")) and e.get("impossible_band")]
    if not entries:
        return []
    bands = {str(e["column"]): [float(e["impossible_band"][0]), float(e["impossible_band"][1])]
             for e in entries}
    units = {str(e["column"]): str(e["unit"]) for e in entries if e.get("unit")}
    cols = list(bands)
    outside = {c: _outside(oc.frame[c], *bands[c]) for c in cols}
    n_values = int(sum(int(m.sum()) for m in outside.values()))
    n_rows = int(pd.concat(outside.values(), axis=1).any(axis=1).sum())
    n_abnormal = int(sum(int(e.get("n_abnormal_but_possible") or 0) for e in entries))
    params = {"bands": bands, **({"units": units} if units else {})}
    named = _names(cols)
    real = (f"{_count(n_abnormal)} abnormal but real ones stay" if n_abnormal
            else "every possible value stays")
    holding = (f"an impossible {named.replace(' and ', ' or ')} value" if len(cols) <= 2
               else f"an impossible value in {_count(len(cols))} columns")
    options = [
        _option(finding, "set_missing", "Set to missing",
                f"{_count(n_values)} impossible {_plural(n_values, 'value')} in {named} become blank; {real}.",
                f"{_count(n_values)} physiologically impossible {_plural(n_values, 'value')} in {named} "
                f"{_plural(n_values, 'was', 'were')} set to missing.",
                "values", params),
        _option(finding, "exclude_rows", "Exclude those rows",
                f"{_count(n_rows)} {_plural(n_rows, 'row')} holding {holding} leave the analysis "
                f"before the seal.",
                f"{_count(n_rows)} {_plural(n_rows, 'row')} with a physiologically impossible value in "
                f"{named} {_plural(n_rows, 'was', 'were')} excluded.",
                "rows", params),
    ]
    usable = [c for c in cols if c != oc.target]
    if usable:
        kept = {"bands": {c: bands[c] for c in usable},
                **({"units": {c: units[c] for c in usable if c in units}} if units else {})}
        one = len(usable) == 1
        options.append(_option(
            finding, "unusable", f"Mark {_plural(len(usable), 'column')} unusable",
            f"{_names(usable)} {'leaves' if one else 'leave'} the predictors; "
            f"{'its' if one else 'their'} values stay in the table, unused.",
            f"{_names(usable)} {'was' if one else 'were'} set aside as unusable for holding "
            f"physiologically impossible values.",
            "columns", kept))
    return options


def _impossible_values(option: str, params: Mapping[str, Any]) -> dict[str, Wrap]:
    if option != "set_missing":
        return {}
    out: dict[str, Wrap] = {}
    for column, (lo, hi) in _bands(params).items():
        def wrap(x: str, lo: float | None = lo, hi: float | None = hi) -> str:
            test = " OR ".join(
                [f"TRY_CAST({x} AS DOUBLE) < {lit_num(lo)}"] * (lo is not None)
                + [f"TRY_CAST({x} AS DOUBLE) > {lit_num(hi)}"] * (hi is not None)) or "FALSE"
            return f"CASE WHEN {test} THEN NULL ELSE {x} END"
        out[column] = wrap
    return out


def _impossible_rows(option: str, params: Mapping[str, Any]) -> list[ExclusionRule]:
    if option != "exclude_rows":
        return []
    # A blank is not an impossible value: these rules keep the rows they cannot check.
    return [ExclusionRule(column=c, low=lo, high=hi, reason="a physiologically impossible value",
                          missing="keep")
            for c, (lo, hi) in _bands(params).items()]


def _impossible_columns(option: str, params: Mapping[str, Any]) -> list[str]:
    return list(_bands(params)) if option == "unusable" else []


def _impossible_marks(option: str, params: Mapping[str, Any]) -> set[tuple[str, str]]:
    return {(c, f"{option}:{lo}:{hi}") for c, (lo, hi) in _bands(params).items()}


# sentinel codes ────────────────────────────────────────────────────────────


def _code_values(params: Mapping[str, Any]) -> dict[str, list[float]]:
    return {str(c): sorted({float(v) for v in vs}) for c, vs in dict(params.get("values") or {}).items()}


def _offer_sentinels(finding: dict[str, Any], p: dict[str, Any], oc: OfferContext) -> list[RepairOption]:
    from turbotab.core.stages.finding_words import family

    if family(finding["id"]) == "pack::survey::sentinel_codes":
        codes = {str(e["item"]): [float(v) for v in e.get("sentinel_values") or []]
                 for e in p.get("items") or [] if isinstance(e, Mapping) and oc.has(e.get("item"))}
    else:
        column = p.get("column") or (finding.get("affected_columns") or [None])[0]
        codes = {str(column): [float(v) for v in p.get("values") or []]} if oc.has(column) else {}
    codes = {c: sorted(set(vs)) for c, vs in codes.items() if vs}
    if not codes:
        return []
    hits = {c: int(_numeric(oc.frame[c]).isin(vs).sum()) for c, vs in codes.items()}
    total = sum(hits.values())
    params = {"values": codes}
    if len(codes) == 1:
        (column, vs), = codes.items()
        x = _numeric(oc.frame[column])
        kept = x[~x.isin(vs)]
        said = _names([_num(v) for v in vs], limit=3)
        move = ""
        if x.notna().any() and kept.notna().any():
            move = (f"; the mean moves {_tick(f'{x.mean():.2f}')} → {_tick(f'{kept.mean():.2f}')}")
        consequence = (f"{_count(total)} {_plural(total, 'cell')} of {said} in {_tick(column)} "
                       f"become blank{move}.")
        sentence = (f"{said} in {_tick(column)} {_plural(len(vs), 'is a code', 'are codes')} for a "
                    f"missing answer; {_count(total)} {_plural(total, 'cell was', 'cells were')} "
                    f"recoded as missing.")
    else:
        lead = max(codes, key=lambda c: (hits[c], c))
        consequence = (f"{_count(total)} {_plural(total, 'code')} in {_count(len(codes))} items become "
                       f"blank, such as {_tick(_num(codes[lead][0]))} in {_tick(lead)}.")
        sentence = (f"Missing-answer codes in {_count(len(codes))} items were recoded as missing "
                    f"({_count(total)} cells).")
    return [_option(finding, "set_missing", "Treat as missing", consequence, sentence, "values",
                    params)]


def _sentinel_values(option: str, params: Mapping[str, Any]) -> dict[str, Wrap]:
    out: dict[str, Wrap] = {}
    for column, vs in _code_values(params).items():
        listed = ", ".join(lit_num(v) for v in vs)

        def wrap(x: str, listed: str = listed) -> str:
            # TRY_CAST: a text column's codes ("7", "9" beside "refused") are compared as the
            # numbers the preview counted, where a plain IN raised a conversion error (audit B18).
            return f"CASE WHEN TRY_CAST({x} AS DOUBLE) IN ({listed}) THEN NULL ELSE {x} END"
        out[column] = wrap
    return out


def _sentinel_marks(option: str, params: Mapping[str, Any]) -> set[tuple[str, str]]:
    return {(c, f"code:{v!r}") for c, vs in _code_values(params).items() for v in vs}


# SAS zeros ─────────────────────────────────────────────────────────────────


def sas_zero_counts(frame: pd.DataFrame) -> dict[str, int]:
    """Column -> how many of its cells hold a misread SAS transport zero (float columns only)."""
    floats = [c for c in frame.columns if frame[c].dtype.kind == "f"]
    if not floats:
        return {}
    hits = is_sas_zero(frame[floats].to_numpy(dtype=float, na_value=np.nan)).sum(axis=0)
    return {str(c): int(n) for c, n in zip(floats, hits) if n}


def sas_zero_finding(frame: pd.DataFrame, target: str | None = None) -> tuple[dict, dict] | None:
    """The app's own finding for SAS transport zeros, as ``(raw, finding)``; None when there are none."""
    counts = sas_zero_counts(frame)
    if not counts:
        return None
    cols = sorted(counts, key=lambda c: (-counts[c], list(frame.columns).index(c)))
    total = sum(counts.values())
    n = len(cols)
    each = ", ".join(f"{_tick(c)} ({counts[c]:,})" for c in cols[:6]) + (
        f" and {n - 6} more" if n > 6 else "")
    finding = {
        "id": "sas_zeros",
        "severity": "warning",
        "title": f"{_count(n)} {_plural(n, 'column holds', 'columns hold')} SAS export zeros as `5.4e-79`.",
        "detail": (f"{_count(total)} {_plural(total, 'cell holds', 'cells hold')} `5.397605e-79`: {each}. "
                   f"A SAS transport file stores numbers in IBM floating point, and a zero read back "
                   f"without converting it becomes 16 to the power −65."),
        "why_it_matters": ("Left as they are, these are tiny positive amounts where the record says "
                           "none: a log or a ratio of them is enormous, and a rule such as 'above "
                           "zero' counts them in."),
        "affected_columns": cols,
        "source": "structural",
        "lens": None,
        "evidence": None,
        "summary": (f"{_count(total)} {_plural(total, 'cell')} in {_names(cols)} hold `5.4e-79`: a "
                    f"zero in a SAS transport file, misread."),
        "routes_to": None,
        "lever_label": None,
        "group": None,
    }
    return {"params": {"columns": counts}, "confidence": "high"}, finding


def _offer_sas(finding: dict[str, Any], p: dict[str, Any], oc: OfferContext) -> list[RepairOption]:
    counts = {str(c): int(n) for c, n in dict(p.get("columns") or {}).items() if oc.has(c)}
    if not counts:
        return []
    cols = list(counts)
    total = sum(counts.values())
    return [_option(
        finding, "zero", "Read as zero",
        f"{_count(total)} {_plural(total, 'cell')} holding `5.4e-79` become `0` in {_names(cols)}.",
        f"{_count(total)} {_plural(total, 'cell')} holding `5.4e-79`, a misread SAS transport zero, "
        f"{_plural(total, 'was', 'were')} read as `0`.",
        "values", {"columns": cols})]


def _sas_values(option: str, params: Mapping[str, Any]) -> dict[str, Wrap]:
    def wrap(x: str) -> str:
        return (f"CASE WHEN abs({x}) BETWEEN {lit_num(SAS_LOW)} AND {lit_num(SAS_HIGH)} "
                f"THEN 0.0 ELSE {x} END")
    return {str(c): wrap for c in params.get("columns") or []}


def _sas_marks(option: str, params: Mapping[str, Any]) -> set[tuple[str, str]]:
    return {(str(c), "sas_zero") for c in params.get("columns") or []}


# binary text ───────────────────────────────────────────────────────────────


def _offer_binary(finding: dict[str, Any], p: dict[str, Any], oc: OfferContext) -> list[RepairOption]:
    column = p.get("column") or (finding.get("affected_columns") or [None])[0]
    levels = [str(v) for v in p.get("levels") or []]
    if (p.get("written_as") != "object_text" or not oc.has(column) or column == oc.target
            or len(levels) != 2):
        return []
    counts = {str(k): int(v) for k, v in dict(p.get("counts") or {}).items()}
    order = levels
    if p.get("positive_known") and p.get("positive") in levels:  # the usual reading first
        order = [str(p["positive"])] + [v for v in levels if v != p["positive"]]
    out = []
    for one in order:
        zero = next(v for v in levels if v != one)
        out.append(_option(
            finding, "level", f"{_tick(one)} counts as 1",
            f"{_tick(column)} becomes `1` for {_tick(one)} ({_count(counts.get(one, 0))} rows) and "
            f"`0` for {_tick(zero)} ({_count(counts.get(zero, 0))}).",
            f"{_tick(column)} was recoded with {_tick(one)} as `1` and {_tick(zero)} as `0`.",
            "values", {"column": str(column), "one": one, "zero": zero}))
    return out


# DuckDB's trim takes the characters to strip; Python's str.strip() strips these four as well.
_BLANKS = "' ' || chr(9) || chr(10) || chr(13)"


def _binary_values(option: str, params: Mapping[str, Any]) -> dict[str, Wrap]:
    column, one, zero = params.get("column"), params.get("one"), params.get("zero")
    if column is None or one is None or zero is None:
        return {}

    def wrap(x: str) -> str:
        token = f"lower(trim(CAST({x} AS VARCHAR), {_BLANKS}))"
        return (f"CASE {token} WHEN {lit_str(str(one).lower())} THEN 1 "
                f"WHEN {lit_str(str(zero).lower())} THEN 0 ELSE NULL END")
    return {str(column): wrap}


def _binary_marks(option: str, params: Mapping[str, Any]) -> set[tuple[str, str]]:
    return {(str(params.get("column")), f"level:{params.get('one')}")}


# energy in kilojoules ──────────────────────────────────────────────────────


def _offer_kj(finding: dict[str, Any], p: dict[str, Any], oc: OfferContext) -> list[RepairOption]:
    column = p.get("energy_column")
    if p.get("verdict") != "energy_in_kj" or not oc.has(column):
        return []
    x = _numeric(oc.frame[column]).dropna()
    move = ""
    if len(x):
        before, after = float(x.median()), float(x.median()) / KJ_PER_KCAL
        move = f": its median moves from {_tick(f'{before:,.0f}')} to {_tick(f'{after:,.0f}')} kcal"
    return [_option(
        finding, "to_kcal", "Convert to kcal",
        f"{_tick(column)} is divided by {_tick(KJ_PER_KCAL)}{move}.",
        f"{_tick(column)} was converted from kilojoules to kcal by dividing by {_tick(KJ_PER_KCAL)}.",
        "values", {"column": str(column), "factor": KJ_PER_KCAL})]


def _kj_values(option: str, params: Mapping[str, Any]) -> dict[str, Wrap]:
    column = params.get("column")
    if column is None:
        return {}
    factor = float(params.get("factor") or KJ_PER_KCAL)

    def wrap(x: str) -> str:
        return f"CAST({x} AS DOUBLE) / {lit_num(factor)}"
    return {str(column): wrap}


def _kj_marks(option: str, params: Mapping[str, Any]) -> set[tuple[str, str]]:
    return {(str(params.get("column")), "kj_to_kcal")}


# numbers written as text (audit MA-16) ─────────────────────────────────────
#
# A SAS or Stata "." for missing, a spreadsheet error ("#DIV/0!"), a trailing space or a decimal
# comma made a whole numeric column text: age became a 61-level category and total energy "free
# text", and no repair existed. ``text_numbers`` reads such a column as numbers and blanks the
# values that are not numbers, counted and named; values below a detection limit ("<0.2") are not
# "not numbers" but values below a limit, and go to their own family, ``below_detection``.

NUMBER_SHARE = 0.8  # a text column is numbers written as text when this share of it reads as numbers
_LOD = r"<\s*([0-9]*\.?[0-9]+)"
# Spellings of "no value" in research exports: SAS and Stata's "." (and their lettered special
# missing values, ".a" to ".z"), spreadsheet errors, and the NaN spellings of C runtimes.
_MISSING_SPELLINGS = re.compile(r"^(\.[A-Za-z_]?|-+|\?|#[A-Z0-9/!?]+[!?]?|n/?a|nan|-nan|null|none|"
                                r"missing|[-+]?1\.#(IND|QNAN|INF)\w*)$", re.IGNORECASE)
_SPREADSHEET = re.compile(r"^#[A-Z0-9/!?]+[!?]?$", re.IGNORECASE)


def _text_of(x: str, decimal: str = ".", thousands: bool = False) -> str:
    """SQL: the text of ``x``, trimmed, with its decimal mark made a point (thousands marks out)."""
    t = f"trim(CAST({x} AS VARCHAR))"
    if decimal == ",":
        return f"replace(replace({t}, '.', ''), ',', '.')"
    if thousands:
        return f"replace({t}, ',', '')"
    return t


def number_sql(x: str, decimal: str = ".", thousands: bool = False,
               lod_factor: float | None = None) -> str:
    """SQL reading text ``x`` as a DOUBLE: NULL where it is not a number (a NaN spelling included).

    ``lod_factor``: a value below a detection limit (``<0.2``) becomes that fraction of its limit;
    without it, such a value is NULL like any other text.
    """
    t = _text_of(x, decimal, thousands)
    plain = f"TRY_CAST({t} AS DOUBLE)"
    plain = f"CASE WHEN isnan({plain}) THEN NULL ELSE {plain} END"
    if lod_factor is None:
        return plain
    limit = f"TRY_CAST(regexp_extract({t}, '^{_LOD}$', 1) AS DOUBLE)"
    return (f"CASE WHEN regexp_full_match({t}, '{_LOD}') THEN {limit} * {lit_num(lod_factor)} "
            f"ELSE {plain} END")


def _spelling_kind(value: str) -> str:
    if _SPREADSHEET.match(value):
        return "a spreadsheet error"
    if value.startswith(".") and len(value) <= 2:
        return "a missing value in SAS or Stata"
    if _MISSING_SPELLINGS.match(value):
        return "a spelling of missing"
    return "not a number"


def read_text_numbers(values: pd.Series) -> dict[str, Any] | None:
    """How a text column reads as numbers, over every value; None when it is not numbers.

    ``decimal`` and ``thousands`` say how its marks read (``ambiguous_comma`` when every comma could
    split thousands: ``1,234``); ``numbers`` the values that read; ``below`` the values below a
    detection limit, by limit; ``unparsed`` every other spelling, with its count, most frequent
    first. Text whose numbers carry leading zeros ("007") is an identifier, never numbers.
    """
    import duckdb
    import pyarrow as pa

    from turbotab.core.datastore import DECIMAL_COMMA, THOUSANDS_COMMA

    text = [None if v is None or (isinstance(v, float) and math.isnan(v)) else str(v)
            for v in values.tolist()]
    con = duckdb.connect()
    try:
        con.register("__v", pa.table({"v": pa.array(text, pa.string())}))
        t = "trim(v)"
        n, with_comma, comma_ok, plain_decimal, leading = con.execute(
            f"SELECT count(v), count_if(contains(v, ',')), "
            f"count_if(contains(v, ',') AND regexp_full_match({t}, '{DECIMAL_COMMA}')), "
            f"count_if(contains(v, ',') AND regexp_full_match({t}, '{DECIMAL_COMMA}') AND NOT "
            f"regexp_full_match({t}, '{THOUSANDS_COMMA}')), "
            f"count_if(regexp_full_match({t}, '[+-]?0[0-9]+')) FROM __v").fetchone()
        n = int(n or 0)
        if n == 0 or int(leading or 0):
            return None
        decimal, thousands, ambiguous = ".", False, False
        if int(with_comma or 0) and int(comma_ok or 0) == int(with_comma):
            if int(plain_decimal or 0):
                decimal = ","
            else:
                thousands, ambiguous = True, True
        x = "v"
        number = number_sql(x, decimal, thousands)
        lod = f"regexp_full_match({_text_of(x, decimal, thousands)}, '{_LOD}')"
        numbers = int(con.execute(f"SELECT count({number}) FROM __v").fetchone()[0] or 0)
        below = con.execute(
            f"SELECT regexp_extract({_text_of(x, decimal, thousands)}, '^{_LOD}$', 1) AS l, count(*) "
            f"FROM __v WHERE {lod} GROUP BY l ORDER BY count(*) DESC, l").fetchall()
        unparsed = con.execute(
            f"SELECT {t} AS s, count(*) FROM __v WHERE v IS NOT NULL AND ({number}) IS NULL AND NOT "
            f"{lod} GROUP BY s ORDER BY count(*) DESC, s").fetchall()
    finally:
        con.close()
    n_below = sum(int(c) for _, c in below)
    if numbers == 0 or (numbers + n_below) < NUMBER_SHARE * n:
        return None
    return {"n_values": n, "numbers": numbers, "decimal": decimal, "thousands": thousands,
            "ambiguous_comma": ambiguous, "below": {str(l): int(c) for l, c in below},
            "unparsed": {str(s): int(c) for s, c in unparsed}}


def _named_values(unparsed: Mapping[str, int], limit: int = 10) -> str:
    """```.``` (`90`), ```#DIV/0!``` (`3`)…: every spelling with its count, the first ``limit``
    named and the rest counted (the recorded decision holds them all)."""
    items = list(unparsed.items())
    shown = ", ".join(f"{_tick(v)} ({_count(n)})" for v, n in items[:limit])
    if len(items) > limit:
        rest = sum(n for _, n in items[limit:])
        shown += f" and {_count(len(items) - limit)} other spellings ({_count(rest)})"
    return shown


def text_number_findings(frame: pd.DataFrame) -> list[tuple[dict, dict]]:
    """The app's own findings for text columns that hold numbers (``text_numbers__<column>``) and
    for values below a detection limit (``below_detection__<column>``), as ``(raw, finding)``."""
    out: list[tuple[dict, dict]] = []
    for column in frame.columns:
        s = frame[column]
        if s.dtype != object:
            continue
        # A cheap look at the first values first: only a column that starts out as numbers is
        # read in full (a table may hold thousands of text columns).
        head = s.dropna().astype(str).head(50).str.strip().str.lstrip("<").str.replace(",", ".")
        if head.empty or pd.to_numeric(head, errors="coerce").notna().mean() < 0.5:
            continue
        reading = read_text_numbers(s)
        if reading is None:
            continue
        c = str(column)
        n, numbers, unparsed, below = (reading["n_values"], reading["numbers"], reading["unparsed"],
                                       reading["below"])
        n_bad, n_below = sum(unparsed.values()), sum(below.values())
        lead = next(iter(unparsed), None)
        how = (" Commas split thousands or mark decimals; nothing in the column says which."
               if reading["ambiguous_comma"] else
               " Its decimals are written with a comma." if reading["decimal"] == "," else "")
        spelled = (f" The values that are not numbers: {_named_values(unparsed)}." if unparsed else "")
        limited = (f" {_count(n_below)} {_plural(n_below, 'value is', 'values are')} below a detection "
                   f"limit, such as {_tick('<' + next(iter(below)))}: that is its own finding."
                   if below else "")
        what = f"; {_count(n_bad)} are not" if n_bad else ""
        summary = (f"{_count(numbers)} of {_count(n)} values of {_tick(c)} read as numbers{what}"
                   + (f", such as {_tick(lead)}, {_spelling_kind(lead)}" if lead else "") + ".")
        out.append(({"params": {"column": c, **reading}, "confidence": "high"}, {
            "id": f"text_numbers__{c}", "severity": "warning",
            "title": f"{_tick(c)} holds numbers stored as text.",
            "detail": (f"{_count(numbers)} of its {_count(n)} values read as numbers.{how}{spelled}"
                       f"{limited}"),
            "why_it_matters": ("As text the column cannot be modeled, correlated or plotted: it "
                               "would sit out of the analysis while looking complete."),
            "affected_columns": [c], "source": "structural", "lens": None, "evidence": None,
            "summary": summary, "routes_to": None, "lever_label": None, "group": None,
        }))
        if below:
            first = next(iter(below))
            out.append(({"params": {"column": c, **reading}, "confidence": "high"}, {
                "id": f"below_detection__{c}", "severity": "warning",
                "title": (f"{_tick(c)} has {_count(n_below)} {_plural(n_below, 'value')} below a "
                          f"detection limit."),
                "detail": (f"{_named_values({'<' + k: v for k, v in below.items()})}. Each is a value "
                           f"somewhere below its limit, not a missing one."),
                "why_it_matters": ("Read as blanks they are left out as if unknown, though each is "
                                   "known to be small; replaced by one value they all become that "
                                   "value, which narrows their spread."),
                "affected_columns": [c], "source": "structural", "lens": None, "evidence": None,
                "summary": (f"{_count(n_below)} {_plural(n_below, 'value')} of {_tick(c)}, such as "
                            f"{_tick('<' + first)}, lie below a detection limit."),
                "routes_to": None, "lever_label": None, "group": None,
            }))
    return out


def _parse_spec(params: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(c): dict(spec) for c, spec in dict(params.get("columns") or {}).items()}


def _offer_text_numbers(finding: dict[str, Any], p: dict[str, Any], oc: OfferContext) -> list[RepairOption]:
    column = p.get("column")
    if not oc.has(column):
        return []
    c = str(column)
    n, numbers = int(p.get("n_values") or 0), int(p.get("numbers") or 0)
    unparsed = {str(k): int(v) for k, v in dict(p.get("unparsed") or {}).items()}
    n_below = sum(int(v) for v in dict(p.get("below") or {}).values())
    n_bad = sum(unparsed.values())
    blank = n_bad + n_below
    named = _named_values(unparsed) if unparsed else ""
    below = (f"{_count(n_below)} below a detection limit" if n_below else "")
    gone = " and ".join(x for x in (f"{_count(n_bad)} that are not numbers" if n_bad else "", below) if x)

    def option(key: str, label: str, decimal: str, thousands: bool, example: str = "") -> RepairOption:
        spec = {"decimal": decimal, "thousands": thousands}
        consequence = (f"{_count(numbers)} values of {_tick(c)} become numbers{example}"
                       + (f"; {_count(blank)} become blank." if blank else "."))
        sentence = (f"{_tick(c)} was read as numbers{example}"
                    + (f"; {gone} {_plural(blank, 'was', 'were')} set to missing" if blank else "")
                    + (f" ({named})" if named else "") + ".")
        return _option(finding, key, label, consequence, sentence, "values", {"columns": {c: spec}})

    if p.get("ambiguous_comma"):
        return [option("thousands", "Commas split thousands", ".", True, ", `1,234` as `1234`"),
                option("decimal_comma", "Commas mark decimals", ",", False, ", `1,234` as `1.234`")]
    return [option("read_numbers", "Read as numbers", str(p.get("decimal") or "."),
                   bool(p.get("thousands")))]


def _text_number_values(option: str, params: Mapping[str, Any]) -> dict[str, Wrap]:
    out: dict[str, Wrap] = {}
    for column, spec in _parse_spec(params).items():
        def wrap(x: str, spec: dict[str, Any] = spec) -> str:
            return number_sql(x, str(spec.get("decimal") or "."), bool(spec.get("thousands")))
        out[column] = wrap
    return out


def _text_number_marks(option: str, params: Mapping[str, Any]) -> set[tuple[str, str]]:
    return {(c, "parse") for c in _parse_spec(params)}


LOD_FACTORS = {"half_limit": 0.5, "limit_root2": 1 / math.sqrt(2)}


def _offer_below_detection(finding: dict[str, Any], p: dict[str, Any], oc: OfferContext) -> list[RepairOption]:
    column = p.get("column")
    below = {str(k): int(v) for k, v in dict(p.get("below") or {}).items()}
    if not oc.has(column) or not below or p.get("ambiguous_comma"):
        return []
    c = str(column)
    n_below = sum(below.values())
    n_bad = sum(int(v) for v in dict(p.get("unparsed") or {}).values())
    first = next(iter(below))
    spec = {"decimal": str(p.get("decimal") or "."), "thousands": bool(p.get("thousands"))}
    out = []
    for key, label, words in (("half_limit", "Half the limit", "half"),
                              ("limit_root2", "Limit over √2", "the limit over √2")):
        factor = LOD_FACTORS[key]
        example = float(first) * factor
        rest = f"; {_count(n_bad)} other text values become blank" if n_bad else ""
        out.append(_option(
            finding, key, label,
            f"{_count(n_below)} values below a limit become {words} of it: {_tick('<' + first)} is "
            f"{_tick(f'{example:.4g}')}{rest}.",
            f"{_count(n_below)} {_plural(n_below, 'value')} of {_tick(c)} below a detection limit "
            f"{_plural(n_below, 'was', 'were')} set to {words} of {_plural(n_below, 'its', 'their')} "
            f"limit, and the column was read as numbers.",
            "values", {"columns": {c: {**spec, "factor": factor}}}))
    return out


def _below_detection_values(option: str, params: Mapping[str, Any]) -> dict[str, Wrap]:
    out: dict[str, Wrap] = {}
    for column, spec in _parse_spec(params).items():
        def wrap(x: str, spec: dict[str, Any] = spec) -> str:
            return number_sql(x, str(spec.get("decimal") or "."), bool(spec.get("thousands")),
                              float(spec.get("factor") or 0.5))
        out[column] = wrap
    return out


def _below_detection_marks(option: str, params: Mapping[str, Any]) -> set[tuple[str, str]]:
    return {m for c, spec in _parse_spec(params).items()
            for m in ((c, "parse"), (c, f"lod:{float(spec.get('factor') or 0.5):.6f}"))}


# dates that read two ways (audit MA-05) ─────────────────────────────────────


def ambiguous_date_finding(frame: pd.DataFrame) -> tuple[dict, dict] | None:
    """The app's own finding for text date columns that read month-first and day-first alike."""
    from turbotab.core import dates
    from turbotab.core.stages.working import _DATEISH

    found: dict[str, dict[str, Any]] = {}
    for column in frame.columns:
        s = frame[column]
        if s.dtype != object:
            continue
        head = s.dropna().astype(str).head(20)
        if head.empty or head.map(lambda v: bool(_DATEISH.search(v))).mean() < 0.5:
            continue
        reading = dates.read_series(s, column=str(column))
        if reading.kind == "ambiguous" and reading.pair:
            found[str(column)] = {"pair": list(reading.pair), "examples": reading.examples,
                                  "n": reading.n}
    if not found:
        return None
    cols = list(found)
    lead = found[cols[0]]
    rows = "; ".join(f"{_tick(e['text'])} is {e[dates.MONTH_FIRST]} month-first or "
                     f"{e[dates.DAY_FIRST]} day-first" for e in lead["examples"])
    finding = {
        "id": "ambiguous_dates", "severity": "warning",
        "title": f"{_names(cols)} {'holds' if len(cols) == 1 else 'hold'} dates that read two ways.",
        "detail": (f"No day in {_names(cols)} is above 12, so nothing in the file says whether the "
                   f"month or the day comes first: {rows}. Until you say which, "
                   f"{'it stays' if len(cols) == 1 else 'they stay'} text."),
        "why_it_matters": ("Read the wrong way, each date moves by up to eleven months: quarterly "
                           "visits read as days apart, and every order and interval built on them "
                           "is wrong."),
        "affected_columns": cols, "source": "structural", "lens": None, "evidence": None,
        "summary": f"{_names(cols)} could be month-first or day-first; the file does not say which.",
        "routes_to": None, "lever_label": None, "group": None,
    }
    return {"params": {"columns": found}, "confidence": "high"}, finding


def _offer_dates(finding: dict[str, Any], p: dict[str, Any], oc: OfferContext) -> list[RepairOption]:
    from turbotab.core import dates

    columns = {str(c): v for c, v in dict(p.get("columns") or {}).items() if oc.has(c)}
    if not columns:
        return []
    cols = list(columns)
    lead = columns[cols[0]]
    example = (lead.get("examples") or [None])[0]
    out = []
    for key, label, at in (("month_first", "Month first", 0), ("day_first", "Day first", 1)):
        formats = {c: v["pair"][at] for c, v in columns.items()}
        shown = (f"{_tick(example['text'])} reads as {_tick(example[dates.MONTH_FIRST if at == 0 else dates.DAY_FIRST])}"
                 if example else "dates read")
        out.append(_option(
            finding, key, label,
            f"{shown}: {'month' if at == 0 else 'day'} first, in every row of {_names(cols)}.",
            f"{_names(cols)} {'was' if len(cols) == 1 else 'were'} read as dates written "
            f"{'month' if at == 0 else 'day'} first ({_tick(formats[cols[0]])}).",
            "values", {"formats": formats}))
    return out


def _date_formats(params: Mapping[str, Any]) -> dict[str, str]:
    return {str(c): str(f) for c, f in dict(params.get("formats") or {}).items()}


def _date_values(option: str, params: Mapping[str, Any]) -> dict[str, Wrap]:
    from turbotab.core import dates

    out: dict[str, Wrap] = {}
    for column, fmt in _date_formats(params).items():
        def wrap(x: str, fmt: str = fmt) -> str:
            parsed = dates.parse_sql(x, [fmt])
            return parsed if dates.has_time(fmt) else f"CAST({parsed} AS DATE)"
        out[column] = wrap
    return out


def _date_marks(option: str, params: Mapping[str, Any]) -> set[tuple[str, str]]:
    return {(c, f"date:{f}") for c, f in _date_formats(params).items()}


def date_formats(state: Any) -> dict[str, str]:
    """Column -> the format an applied date-reading repair chose (month or day first)."""
    out: dict[str, str] = {}
    for fam, _, option, params in _applied(state):
        if fam.key == "date_reading":
            out.update(_date_formats(params))
    return out


# infinite values (audit MA-18) ─────────────────────────────────────────────


def infinite_counts(frame: pd.DataFrame) -> dict[str, int]:
    floats = [c for c in frame.columns if frame[c].dtype.kind == "f"]
    if not floats:
        return {}
    hits = np.isinf(frame[floats].to_numpy(dtype=float, na_value=np.nan)).sum(axis=0)
    return {str(c): int(n) for c, n in zip(floats, hits) if n}


def infinite_finding(frame: pd.DataFrame) -> tuple[dict, dict] | None:
    """The app's own finding for infinite values (a ratio over zero), as ``(raw, finding)``."""
    counts = infinite_counts(frame)
    if not counts:
        return None
    cols = sorted(counts, key=lambda c: (-counts[c], list(frame.columns).index(c)))
    total = sum(counts.values())
    each = ", ".join(f"{_tick(c)} ({counts[c]:,})" for c in cols[:6]) + (
        f" and {len(cols) - 6} more" if len(cols) > 6 else "")
    finding = {
        "id": "infinite_values", "severity": "warning",
        "title": (f"{_count(len(cols))} {_plural(len(cols), 'column holds', 'columns hold')} "
                  f"infinite values."),
        "detail": (f"{_count(total)} {_plural(total, 'cell is', 'cells are')} infinite: {each}. A "
                   f"ratio whose denominator is zero gives one, such as protein per 1,000 kcal on a "
                   f"day recorded as 0 kcal."),
        "why_it_matters": ("An infinite value is not a measurement: no mean, model or plot can use "
                           "it. The summaries count it and leave it out."),
        "affected_columns": cols, "source": "structural", "lens": None, "evidence": None,
        "summary": f"{_count(total)} {_plural(total, 'cell')} in {_names(cols)} {_plural(total, 'is', 'are')} infinite.",
        "routes_to": None, "lever_label": None, "group": None,
    }
    return {"params": {"columns": counts}, "confidence": "high"}, finding


def _offer_infinite(finding: dict[str, Any], p: dict[str, Any], oc: OfferContext) -> list[RepairOption]:
    counts = {str(c): int(n) for c, n in dict(p.get("columns") or {}).items() if oc.has(c)}
    if not counts:
        return []
    cols, total = list(counts), sum(counts.values())
    return [_option(
        finding, "set_missing", "Set to missing",
        f"{_count(total)} infinite {_plural(total, 'value')} in {_names(cols)} become blank; the "
        f"finite values stay.",
        f"{_count(total)} infinite {_plural(total, 'value')} in {_names(cols)} (a ratio over zero) "
        f"{_plural(total, 'was', 'were')} set to missing.",
        "values", {"columns": cols})]


def _infinite_values(option: str, params: Mapping[str, Any]) -> dict[str, Wrap]:
    def wrap(x: str) -> str:
        return f"CASE WHEN isinf(CAST({x} AS DOUBLE)) THEN NULL ELSE {x} END"
    return {str(c): wrap for c in params.get("columns") or []}


def _infinite_marks(option: str, params: Mapping[str, Any]) -> set[tuple[str, str]]:
    return {(str(c), "infinite") for c in params.get("columns") or []}


FAMILIES: dict[str, Family] = {}
_BY_FINDING: dict[str, str] = {}


def register_family(family: Family, finding_families: Iterable[str]) -> Family:
    """Declare ``family`` and the finding families (``finding_words.family(id)``) it answers."""
    FAMILIES[family.key] = family
    for name in finding_families:
        _BY_FINDING[name] = family.key
    return family


# The order two repairs on one column compose in: text is read as numbers (values below a limit
# first, so the plain reading finds them already read), then SAS zeros, then codes, which are
# written in the file's own units, before any conversion of units (a 99999 code divided by 4.184
# was no longer 99999: audit B18), then levels, infinities and impossible bands. Dates stand alone.
register_family(Family("below_detection", 0, _offer_below_detection,
                       {"half_limit": "values", "limit_root2": "values"},
                       values=_below_detection_values, marks=_below_detection_marks),
                ["below_detection"])
register_family(Family("text_numbers", 1, _offer_text_numbers,
                       {"read_numbers": "values", "thousands": "values", "decimal_comma": "values"},
                       values=_text_number_values, marks=_text_number_marks), ["text_numbers"])
register_family(Family("sas_zeros", 2, _offer_sas, {"zero": "values"},
                       values=_sas_values, marks=_sas_marks), ["sas_zeros"])
register_family(Family("sentinel_codes", 3, _offer_sentinels, {"set_missing": "values"},
                       values=_sentinel_values, marks=_sentinel_marks),
                ["sentinel_missing", "pack::survey::sentinel_codes"])
register_family(Family("energy_kj", 4, _offer_kj, {"to_kcal": "values"},
                       values=_kj_values, marks=_kj_marks), ["pack::dietary::atwater"])
register_family(Family("binary_text", 5, _offer_binary, {"level": "values"},
                       values=_binary_values, marks=_binary_marks), ["binary_text"])
register_family(Family("infinite_values", 6, _offer_infinite, {"set_missing": "values"},
                       values=_infinite_values, marks=_infinite_marks), ["infinite_values"])
register_family(Family("impossible_values", 7, _offer_impossible,
                       {"set_missing": "values", "exclude_rows": "rows", "unusable": "columns"},
                       values=_impossible_values, rows=_impossible_rows, columns=_impossible_columns,
                       marks=_impossible_marks),
                ["pack::clinical::impossible_vs_extreme"])
register_family(Family("date_reading", 8, _offer_dates, {"month_first": "values", "day_first": "values"},
                       values=_date_values, marks=_date_marks), ["ambiguous_dates"])


def family_for(finding_id: str) -> Family | None:
    from turbotab.core.stages.finding_words import family

    key = _BY_FINDING.get(family(str(finding_id)))
    return FAMILIES.get(key) if key else None


def option_columns(finding_id: str, option: str, params: Mapping[str, Any],
                   effects: Iterable[Effect] = ("values", "rows", "columns")) -> set[str]:
    """The columns a repair option acts on, by the effects asked for: the columns whose values it
    rewrites, the columns its row rules read, the columns it sets aside."""
    fam = family_for(finding_id)
    if fam is None or option not in fam.effects:
        return set()
    wanted = set(effects)
    out: set[str] = set()
    if "values" in wanted:
        out |= set(fam.values(option, params))
    if "rows" in wanted:
        out |= {rule.column for rule in fam.rows(option, params)}
    if "columns" in wanted:
        out |= set(fam.columns(option, params))
    return out


# ── offering: the findings stage attaches the options ───────────────────────


def offer(finding: dict[str, Any], raw: Mapping[str, Any] | None, oc: OfferContext) -> list[RepairOption]:
    fam = family_for(finding["id"])
    if fam is None:
        return []
    params = dict((raw or {}).get("params") or {})
    try:
        return fam.offer(finding, params, oc)
    except Exception:  # noqa: BLE001 - a finding never fails for want of a repair
        import logging

        logging.getLogger(__name__).exception("no repair offered for %s", finding["id"])
        return []


def attach(pairs: Iterable[tuple[Mapping[str, Any] | None, dict[str, Any]]], frame: pd.DataFrame,
           target: str | None = None) -> None:
    """Give every finding its ``repairs`` (possibly none), in place.

    A finding with a repair has a control after all, so its summary stops saying it has none.
    """
    oc = OfferContext(frame=frame, target=target)
    for raw, finding in pairs:
        options = offer(finding, raw, oc)
        finding["repairs"] = [o.model_dump(mode="json") for o in options]
        summary = str(finding.get("summary") or "")
        if options and summary.endswith(NO_LEVER):
            finding["summary"] = summary[: -len(NO_LEVER)].rstrip()


# ── executing: values, rows, columns ────────────────────────────────────────


def _findings_of(artifact: Any) -> list[dict[str, Any]]:
    data = getattr(artifact, "data", artifact)
    if isinstance(data, Mapping):
        return [f for f in data.get("findings") or [] if isinstance(f, Mapping)]
    return []


findings_of = _findings_of  # the findings an artifact (or its Bundle) holds


def _finding(artifact: Any, finding_id: str) -> dict[str, Any] | None:
    return next((dict(f) for f in _findings_of(artifact) if f.get("id") == finding_id), None)


def _disposition(value: Any) -> FindingDisposition:
    return value if isinstance(value, FindingDisposition) else FindingDisposition.model_validate(value)


def _dispositions(source: Any) -> dict[str, FindingDisposition]:
    """``state.findings`` (from a ProjectState, or the mapping itself) as dispositions."""
    if source is None:
        return {}
    value = getattr(source, "findings", source) if not isinstance(source, Mapping) else source
    if not value:
        return {}
    return {str(k): _disposition(v) for k, v in dict(value).items()}


def _params(finding_id: str, d: FindingDisposition, findings: Any) -> dict[str, Any]:
    """The disposition's own params, else those of the one option of that key the finding offers."""
    if d.params:
        return dict(d.params)
    found = _finding(findings, finding_id) if findings is not None else None
    same = [o for o in (found or {}).get("repairs") or [] if o.get("key") == d.option]
    return dict(same[0]["decision"]["params"]) if len(same) == 1 else {}


def _applied(dispositions: Any, findings: Any = None) -> list[tuple[Family, str, str, dict[str, Any]]]:
    """``(family, finding id, option, params)`` of every applied repair, in composition order."""
    out = []
    for fid, d in _dispositions(dispositions).items():
        fam = family_for(fid)
        if d.action != "applied" or fam is None or d.option not in fam.effects:
            continue
        out.append((fam, fid, str(d.option), _params(fid, d, findings)))
    return sorted(out, key=lambda item: (item[0].priority, item[1]))


def column_expressions(findings: Any, dispositions: Any) -> dict[str, str]:
    """Column -> the DuckDB SQL expression that replaces it on the working table.

    ``findings`` is the findings artifact (or None); ``dispositions`` is ``state.findings`` (or the
    state). Only applied repairs whose effect is on values contribute; several on one column
    compose in the families' order. Columns no repair touches are absent.
    """
    wraps: dict[str, list[Wrap]] = {}
    done: set[tuple[str, str]] = set()
    for fam, _, option, params in _applied(dispositions, findings):
        marks = fam.marks(option, params)
        for column, wrap in fam.values(option, params).items():
            mine = {m for m in marks if m[0] == column}
            if mine and mine <= done:
                continue  # another finding's repair already does exactly this to this column
            done |= mine
            wraps.setdefault(column, []).append(wrap)
    out = {}
    for column, chain in wraps.items():
        sql = ident(column)
        for wrap in chain:
            sql = wrap(sql)
        out[column] = sql
    return out


def exclusion_rules(dispositions: Any, findings: Any = None) -> list[ExclusionRule]:
    """Range rules for the rows applied repairs exclude, after the eligibility answer's own."""
    rules: list[ExclusionRule] = []
    for fam, _, option, params in _applied(dispositions, findings):
        rules.extend(fam.rows(option, params))
    return rules


def unusable_columns(dispositions: Any, findings: Any = None) -> list[str]:
    """Columns applied repairs marked unusable: they leave the predictors."""
    out: list[str] = []
    for fam, _, option, params in _applied(dispositions, findings):
        out.extend(c for c in fam.columns(option, params) if c not in out)
    return out


def evaluate(parquet: str | Path, columns: Sequence[str], expressions: Mapping[str, str],
             row_ids: Any = None) -> pd.DataFrame:
    """``columns`` of an ingested table as DuckDB computes them under ``expressions``, by row id.

    The same SQL the working table is made with, so a preview and the working table it foretells
    cannot disagree. ``row_ids`` (default: every row) are returned in row-id order.
    """
    import duckdb

    rel = f"read_parquet({lit_str(str(parquet))})"
    select = ", ".join([ROW_ID, *(f"{expressions.get(c, ident(c))} AS {ident(c)}" for c in columns)])
    con = duckdb.connect()
    try:
        if row_ids is None:
            table = con.execute(f"SELECT {select} FROM {rel} ORDER BY {ROW_ID}").to_arrow_table()
        else:
            import pyarrow as pa

            con.register("__repair_ids", pa.table({"id": np.asarray(row_ids, dtype=np.int64)}))
            table = con.execute(f"SELECT {select} FROM {rel} WHERE {ROW_ID} IN "
                                f"(SELECT id FROM __repair_ids) ORDER BY {ROW_ID}").to_arrow_table()
    finally:
        con.close()
    frame = table.to_pandas(date_as_object=False).set_index(ROW_ID)
    frame.index.name = "row_id"
    return frame


# ── validating and completing apply / defer / dismiss ───────────────────────

_UNKNOWN = object()


def _ctx(ctx: Any, name: str) -> Any:
    if ctx is None:
        return None
    return ctx.get(name) if isinstance(ctx, Mapping) else getattr(ctx, name, None)


def _artifact(ctx: Any) -> Any:
    """The fresh findings artifact as ``ctx`` knows it; ``_UNKNOWN`` when ``ctx`` cannot say."""
    reader = _ctx(ctx, "artifact")
    if not callable(reader):
        return _UNKNOWN
    try:
        return reader("findings")
    except Exception:  # noqa: BLE001 - an unreadable artifact checks nothing
        return None


def _known_finding(decision: Any, ctx: Any) -> dict[str, Any] | None | object:
    """The finding the decision names; ``_UNKNOWN`` when the findings cannot be read now."""
    artifact = _artifact(ctx)
    if artifact is _UNKNOWN or artifact is None:
        return _UNKNOWN
    found = _finding(artifact, decision.finding_id)
    if found is None:
        raise Refusal(
            "unknown_finding",
            f"There is no finding {decision.finding_id!r} on this table now.",
            exits=[{"label": "Choose one of the findings listed now", "decision": None}],
        )
    return found


def admits(offered: Mapping[str, Any], given: Mapping[str, Any]) -> bool:
    """True when ``given`` params are ``offered`` ones or a part of them.

    A per-column mapping may name fewer columns; a list may hold fewer of its values (some codes
    and not others); everything else must be equal. Nothing may be added.
    """
    for key, value in given.items():
        if key not in offered:
            return False
        want = offered[key]
        if isinstance(want, Mapping):
            if not isinstance(value, Mapping) or not value or not set(value) <= set(want):
                return False
            for column, part in value.items():
                if isinstance(want[column], list):
                    if not isinstance(part, list) or not part or not _subset(part, want[column]):
                        return False
                elif not _same(part, want[column]):
                    return False
        elif isinstance(want, list):
            if not isinstance(value, list) or not value or not _subset(value, want):
                return False
        elif not _same(value, want):
            return False
    return True


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return float(a) == float(b)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def _subset(part: Sequence[Any], whole: Sequence[Any]) -> bool:
    return all(any(_same(p, w) for w in whole) for p in part)


def _exits(options: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{"label": str(o["label"]), "decision": o["decision"]} for o in options]


def _repair_is_offered(decision: ApplyRepair, ctx: Any) -> None:
    found = _known_finding(decision, ctx)
    if found is _UNKNOWN:
        if not decision.params and callable(_ctx(ctx, "artifact")):
            raise Refusal(
                "findings_not_ready",
                "The findings are still being worked out; choose the repair once they are listed.",
            )
        return
    options = found.get("repairs") or []
    title = str(found.get("title") or decision.finding_id).rstrip(".")
    if not options:
        exits = []
        if found.get("routes_to"):
            exits.append({"label": "Hold it for the question it belongs to",
                          "decision": DeferFinding(finding_id=decision.finding_id, to=found["routes_to"])})
        exits.append({"label": "Dismiss it", "decision": DismissFinding(finding_id=decision.finding_id)})
        raise Refusal("no_repair", f"{title}: there is no repair to apply to it here.", exits=exits)
    same = [o for o in options if o.get("key") == decision.option]
    if not same:
        raise Refusal("unknown_repair",
                      f"{decision.option!r} is not one of the repairs offered for this finding.",
                      exits=_exits(options))
    if decision.params:
        if not any(admits(o["decision"]["params"], decision.params) for o in same):
            raise Refusal("repair_params",
                          "Those settings are not ones this finding offers; the data decide them.",
                          exits=_exits(same))
    elif len(same) > 1:
        raise Refusal("choose_repair", "This repair has more than one form; choose one.",
                      exits=_exits(same))


def _fill_params(decision: ApplyRepair, ctx: Any) -> ApplyRepair:
    """An option named without its params gets the ones the finding offers, so every recorded
    disposition says exactly what it does."""
    if decision.params:
        return decision
    artifact = _artifact(ctx)
    found = _finding(artifact, decision.finding_id) if artifact not in (_UNKNOWN, None) else None
    same = [o for o in (found or {}).get("repairs") or [] if o.get("key") == decision.option]
    if len(same) != 1:
        return decision
    return decision.model_copy(update={"params": dict(same[0]["decision"]["params"])})


def _question_keys() -> tuple[str, ...]:
    from turbotab.core.interview import QUESTION_KEYS

    return tuple(QUESTION_KEYS)


def _deferral_has_a_place(decision: DeferFinding, ctx: Any) -> None:
    keys = _question_keys()
    found = _known_finding(decision, ctx)
    route = found.get("routes_to") if isinstance(found, dict) else None
    if decision.to not in keys:
        exits = ([{"label": "Hold it for the question it belongs to",
                   "decision": DeferFinding(finding_id=decision.finding_id, to=route)}]
                 if route in keys else [])
        raise Refusal("unknown_question", f"There is no question {decision.to!r} to hold it for.",
                      exits=exits + [{"label": "Dismiss it",
                                      "decision": DismissFinding(finding_id=decision.finding_id)}])


def _dismissal_names_a_finding(decision: DismissFinding, ctx: Any) -> None:
    _known_finding(decision, ctx)


register_validator("apply_repair", _repair_is_offered)
register_completion("apply_repair", _fill_params)
register_validator("defer_finding", _deferral_has_a_place)
register_validator("dismiss_finding", _dismissal_names_a_finding)


# ── memory: dispositions, answered by, deferred to ──────────────────────────

_FINDING_KINDS = ("apply_repair", "defer_finding", "dismiss_finding")


def disposition_writers(records: Sequence[Any]) -> dict[str, str]:
    """Finding id -> the live record that wrote its disposition."""
    from turbotab.core.decisions import reverted

    ordered = sorted(records, key=lambda r: r.seq)
    try:
        cancelled = reverted(ordered)
    except Refusal:
        cancelled = {}
    out: dict[str, str] = {}
    for record in ordered:
        d = record.decision
        if record.id not in cancelled and d.kind in _FINDING_KINDS:
            out[str(d.finding_id)] = record.id
    return out


_IDENTIFIER_FAMILIES = ("voice::identifier", "voice::repeats", "pack::metabolomics::repeated_subjects")
_EXCLUDED_FAMILIES = ("unnamed_columns", "constant_columns")
_PREDICTOR_ROLES = ("exposure", "covariate", "energy")


def matching_answer(finding: Mapping[str, Any], state: Any) -> bool:
    """Whether the answer to the question ``finding`` routes to settles it.

    Roles: the columns it is about have the role it asks for (an identifier for a repeating or
    unique id, a flag for a flag, left out for an unnamed or constant column, any role otherwise).
    Exclusions: a rule names one of its columns. Any other question: it is answered.
    """
    from turbotab.core.stages.finding_words import family

    route = finding.get("routes_to")
    if not route or getattr(state, str(route), None) is None:
        return False
    cols = [c for c in finding.get("affected_columns") or [] if c != getattr(state, "target", None)]
    fam = family(str(finding.get("id")))
    if route == "roles":
        roles = state.roles or {}
        if fam in _IDENTIFIER_FAMILIES:
            return bool(cols) and roles.get(cols[0]) == "identifier"
        if fam == "voice::flag":
            return bool(cols) and roles.get(cols[0]) == "flag"
        if fam in _EXCLUDED_FAMILIES:
            return all(c in roles and roles[c] not in _PREDICTOR_ROLES for c in cols)
        return all(c in roles for c in cols)
    if route == "exclusions":
        if finding.get("affected_columns") and not cols:
            return False  # about the outcome alone: no eligibility rule may act on it (RO-01)
        named = {rule.column for rule in state.exclusions or []}
        return not cols or bool(named & set(cols))
    return True


def annotate(artifact: Any, state: Any, records: Sequence[Any]) -> Any:
    """The findings artifact as served: each finding with its ``disposition`` and ``answered_by``.

    ``answered_by`` is the record that wrote the finding's disposition; else, the answer to its
    routed question when that answer settles it (:func:`matching_answer`); else, an applied repair
    of another finding that already does what one of this finding's options would.
    """
    if not isinstance(artifact, Mapping) or not isinstance(artifact.get("findings"), list):
        return artifact
    from turbotab.core.interview import _live_writer

    dispositions = _dispositions(state)
    writers = disposition_writers(records)
    slot_writer = _live_writer(records, state)
    done: dict[tuple[str, str], str] = {}  # what applied repairs do -> the record that applied it
    for fam, fid, option, params in _applied(dispositions, artifact):
        if fid in writers:
            for mark in fam.marks(option, params):
                done.setdefault(mark, writers[fid])
    out = []
    for finding in artifact["findings"]:
        f = dict(finding)
        f.setdefault("repairs", [])
        d = dispositions.get(str(f.get("id")))
        f["disposition"] = d.model_dump(mode="json") if d is not None else None
        answered = writers.get(str(f.get("id"))) if d is not None else None
        if answered is None and matching_answer(f, state):
            answered = slot_writer.get(str(f["routes_to"]))
        if answered is None:
            answered = _covered(f, done)
        f["answered_by"] = answered
        out.append(f)
    return {**artifact, "findings": out}


def _covered(finding: Mapping[str, Any], done: Mapping[tuple[str, str], str]) -> str | None:
    fam = family_for(str(finding.get("id")))
    if fam is None or not done:
        return None
    for option in finding.get("repairs") or []:
        marks = fam.marks(option["key"], option["decision"]["params"])
        if marks and all(m in done for m in marks):
            return done[sorted(marks)[0]]
    return None


def deferred_to(state: Any) -> dict[str, list[str]]:
    """Question key -> the findings deferred to it, in the order they were recorded."""
    out: dict[str, list[str]] = {}
    for fid, d in _dispositions(state).items():
        if d.action == "deferred" and d.to:
            out.setdefault(str(d.to), []).append(fid)
    return out


# ── the key view: only applied repairs change data ──────────────────────────


def _applied_only(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    kept = {k: v for k, v in value.items() if isinstance(v, Mapping) and v.get("action") == "applied"}
    return kept or None


def _register_key_view() -> None:
    from turbotab.core.graph import register_key_view

    register_key_view("findings", _applied_only)


_register_key_view()


# ── preview before apply ────────────────────────────────────────────────────

MAX_ROWS = 8
MAX_COLUMNS = 12


def _base_parquet(ctx: Any) -> Path:
    """The table findings are computed on: the oriented table when the graph has one, else the
    store's own (before the working table exists, they are the same file)."""
    try:
        oriented = ctx.artifact("oriented")
    except Exception:  # noqa: BLE001 - no oriented stage: the store's table is the base
        oriented = None
    files = getattr(oriented, "files", None) or {}
    path = files.get("table.parquet")
    return Path(path) if path else Path(ctx.datastore.parquet)


def repair_views(decision: Any, ctx: Any) -> list[Any]:
    """What applying a repair would change, before it is recorded (M2_CONTRACT §4).

    Values: a table_focus of the cells that change, and the affected column's distribution with
    labeled marks (the codes, the impossible band, zero, the intake cut-offs). Rows: the
    participant flow with the rows leaving, and the column with its band marked. Columns: the
    lineage with the column leaving the predictors. Values are read on every row, as the finding
    read them (it answers "is this data corrupted?"), and the basis says so.
    """
    findings = ctx.artifact("findings")
    finding = _finding(findings, decision.finding_id)
    fam = family_for(decision.finding_id)
    if finding is None or fam is None or decision.option not in fam.effects:
        ctx.read["note"] = "This finding is not listed now, so its repair cannot be shown."
        return []
    params = dict(decision.params) or _params(decision.finding_id, FindingDisposition(
        action="applied", option=decision.option), findings)
    current = {k: v for k, v in _dispositions(ctx.state).items() if k != decision.finding_id}
    proposed = {**current, decision.finding_id: FindingDisposition(
        action="applied", option=decision.option, params=params)}
    effect = fam.effects[decision.option]
    if effect == "values":
        return _value_views(fam, decision.option, params, findings, current, proposed, ctx)
    if effect == "rows":
        return _row_views(params, proposed, ctx)
    return _column_views(params, ctx)


def _value_views(fam: Family, option: str, params: Mapping[str, Any], findings: Any,
                 current: Mapping[str, Any], proposed: Mapping[str, Any], ctx: Any) -> list[Any]:
    from turbotab.core.consequences import CAPTION_WORDS, TITLE_WORDS, _table_view, clip_words

    store = ctx.datastore
    columns = [c for c in fam.values(option, params) if c in store.columns][:MAX_COLUMNS]
    if not columns:
        ctx.read["note"] = "The columns this repair rewrites are not in the table now."
        return []
    parquet = _base_parquet(ctx)
    before = evaluate(parquet, columns, column_expressions(findings, current))
    after = evaluate(parquet, columns, column_expressions(findings, proposed))
    n = len(before)
    ctx.read["basis"] = f"Every one of the {n:,} rows, as the finding read them."
    changed = {c: _changed(before[c], after[c]) for c in columns}
    ranked = sorted(columns, key=lambda c: (-int(changed[c].sum()), columns.index(c)))
    ranked = [c for c in ranked if changed[c].any()]
    if not ranked:
        ctx.read["note"] = "No cell changes: the values this repair rewrites are not in the table now."
        return []
    focus = _table_view(before, after, ranked)
    views: list[Any] = []
    if focus is not None:
        n_cells = int(sum(int(changed[c].sum()) for c in ranked))
        caption = (f"{_count(n_cells)} {_plural(n_cells, 'cell changes', 'cells change')} in "
                   f"{_names(ranked)}; the rows with most changes are shown.")
        views.append(focus.model_copy(update={
            "title": clip_words("The cells this repair changes", TITLE_WORDS),
            "caption": clip_words(caption, CAPTION_WORDS)}))
    top = ranked[0]
    dist = _distribution(fam, option, params, top, before[top], after[top], changed[top])
    if dist is not None:
        views.append(dist)
    return views


def _changed(before: pd.Series, after: pd.Series) -> pd.Series:
    both_na = before.isna() & after.isna()
    if before.dtype.kind in "biuf" and after.dtype.kind in "biuf":
        x = before.to_numpy(dtype=float, na_value=np.nan)
        y = after.to_numpy(dtype=float, na_value=np.nan)
        with np.errstate(invalid="ignore"):
            same = (x == y) | (np.isnan(x) & np.isnan(y))
        return pd.Series(~same, index=before.index)
    return ~((before.astype(object) == after.astype(object)) | both_na)


def _marks(fam: Family, option: str, params: Mapping[str, Any], column: str) -> list[Any]:
    from turbotab.core.consequences import Mark

    shown = _amount
    if fam.key == "impossible_values":
        lo, hi = _bands(params).get(column, (None, None))
        unit = dict(params.get("units") or {}).get(column)
        tail = f" {unit}" if unit else ""
        return ([Mark(value=lo, label=f"floor {shown(lo)}{tail}")] if lo is not None else []) + (
            [Mark(value=hi, label=f"ceiling {shown(hi)}{tail}")] if hi is not None else [])
    if fam.key == "sentinel_codes":
        return [Mark(value=v, label=f"code {shown(v)}") for v in _code_values(params).get(column, [])]
    if fam.key == "sas_zeros":
        return [Mark(value=0.0, label="zero")]
    if fam.key == "energy_kj":
        return [Mark(value=v, label=f"{v:,.0f} kcal") for v in INTAKE_MARKS]
    return []


def _distribution(fam: Family, option: str, params: Mapping[str, Any], column: str,
                  before: pd.Series, after: pd.Series, changed: pd.Series) -> Any | None:
    from turbotab.core.consequences import CAPTION_WORDS, TITLE_WORDS, DistributionView, _histogram_pair, clip_words

    if (before.dtype.kind not in "biuf" or after.dtype.kind not in "biuf"
            or fam.key in ("binary_text", "infinite_values")):
        return None  # an infinity has no place on a histogram's axis
    x = before.to_numpy(dtype=float, na_value=np.nan)
    y = after.to_numpy(dtype=float, na_value=np.nan)
    hist_before, hist_after = _histogram_pair(x, y)
    n = int(changed.sum())
    labels = ("Now", "With this repair")
    if fam.key == "sentinel_codes":
        fx, fy = x[np.isfinite(x)], y[np.isfinite(y)]
        caption = (f"{_tick(column)}: mean {_tick(f'{fx.mean():.2f}')} with the codes, "
                   f"{_tick(f'{fy.mean():.2f}')} without; {_count(n)} {_plural(n, 'value')} go blank.")
    elif fam.key == "sas_zeros":
        caption = (f"{_tick(column)}: {_count(n)} {_plural(n, 'value')} at `5.4e-79` become exactly "
                   f"`0`; the shape barely moves.")
    elif fam.key == "energy_kj":
        fy = y[np.isfinite(y)]
        inside = int(((fy >= INTAKE_MARKS[0]) & (fy <= INTAKE_MARKS[1])).sum())
        caption = (f"In kcal, {_count(inside)} of {_count(len(fy))} values of {_tick(column)} fall "
                   f"between `500` and `5,000`.")
        labels = ("Now, in kJ", "With this repair, in kcal")
    else:
        caption = (f"{_tick(column)}: {_count(n)} {_plural(n, 'value')} outside the possible range "
                   f"go blank; the rest stay.")
    return DistributionView(
        title=clip_words(f"How `{column}` changes", TITLE_WORDS),
        caption=clip_words(caption, CAPTION_WORDS),
        emphasis=[column],
        column=column,
        before=hist_before,
        after=hist_after,
        before_label=labels[0],
        after_label=labels[1],
        marks=_marks(fam, option, params, column),
    )


def _row_views(params: Mapping[str, Any], proposed: Mapping[str, Any], ctx: Any) -> list[Any]:
    from turbotab.core.consequences import CAPTION_WORDS, TITLE_WORDS, DistributionView, RowFlowView, _histogram_pair, clip_words
    from turbotab.core.row_previews import _flows, _pool, _sealed, _steps
    from turbotab.core.stages.rows import rule_keep

    bands = {c: b for c, b in _bands(params).items() if c in ctx.datastore.columns}
    if not bands:
        ctx.read["note"] = "The columns this repair reads are not in the table now."
        return []
    after_state = ctx.state.model_copy(update={"findings": dict(proposed)})
    _, _, [(before, _), (after, _)] = _flows(ctx, [ctx.state, after_state])
    n_before, n_after = int(before[-1]["n"]), int(after[-1]["n"])
    gone = n_before - n_after
    named = _names(list(bands)).replace(" and ", " or ")
    caption = (f"{_count(gone)} {_plural(gone, 'row')} with an impossible {named} would leave; "
               f"{_count(n_after)} remain.")
    views: list[Any] = [RowFlowView(
        title=clip_words("Rows this repair excludes", TITLE_WORDS),
        caption=clip_words(caption, CAPTION_WORDS),
        emphasis=[s["key"] for s in after if str(s["key"]).startswith("repair:")][:3],
        before=_steps(before, _sealed(ctx)),
        after=_steps(after, _sealed(ctx)),
    )]
    values = ctx.datastore.materialize(list(bands), _pool(ctx))
    outside = {c: int((~rule_keep(values, ExclusionRule(column=c, low=lo, high=hi, reason="r",
                                                        missing="keep"))).sum())
               for c, (lo, hi) in bands.items()}
    column = max(bands, key=lambda c: (outside[c], -list(bands).index(c)))  # the one that excludes most
    lo, hi = bands[column]
    rule = ExclusionRule(column=column, low=lo, high=hi, reason="a physiologically impossible value",
                         missing="keep")
    x = _numeric(values[column]).to_numpy(dtype=float, na_value=np.nan)
    kept = rule_keep(values, rule).to_numpy()
    hist_before, hist_after = _histogram_pair(x, np.where(kept, x, np.nan))
    hist_after = hist_after.model_copy(update={"n_missing": int(np.isnan(x[kept]).sum())})
    n_out = int((~kept).sum())
    fam = FAMILIES["impossible_values"]
    views.append(DistributionView(
        title=clip_words(f"Impossible values of `{column}`", TITLE_WORDS),
        caption=clip_words(f"{_count(n_out)} {_plural(n_out, 'value')} of {_tick(column)} lie outside "
                           f"{_tick(_amount(lo))}–{_tick(_amount(hi))}; their rows leave.", CAPTION_WORDS),
        emphasis=[column],
        column=column,
        before=hist_before,
        after=hist_after,
        before_label="Now",
        after_label="With this repair",
        marks=_marks(fam, "exclude_rows", params, column),
    ))
    return views


def _column_views(params: Mapping[str, Any], ctx: Any) -> list[Any]:
    from turbotab.core.consequences import CAPTION_WORDS, TITLE_WORDS, LineageView, clip_words
    from turbotab.core.row_previews import _order, _roles_lineage

    columns = [c for c in _bands(params) if c in ctx.datastore.columns]
    roles = dict(ctx.state.roles or {})
    if not columns:
        ctx.read["note"] = "The columns this repair sets aside are not in the table now."
        return []
    if not roles:
        ctx.read["note"] = (f"{_names(columns)} will leave the predictors once the column roles are "
                            f"confirmed; nothing else changes.")
        return []
    order = _order(ctx)
    after = {c: ("excluded" if c in columns else r) for c, r in roles.items()}
    one = len(columns) == 1
    in_model = [c for c in columns if roles.get(c) in _PREDICTOR_ROLES]
    if in_model:
        caption = (f"{_names(in_model)} {'leaves' if len(in_model) == 1 else 'leave'} the model "
                   f"matrix; {'its' if one else 'their'} values stay in the table.")
    else:
        caption = f"{_names(columns)} {'is' if one else 'are'} not a predictor now, so the matrix keeps its columns."
    return [LineageView(
        title=clip_words("Columns this repair sets aside", TITLE_WORDS),
        caption=clip_words(caption, CAPTION_WORDS),
        emphasis=columns[:MAX_COLUMNS],
        before=_roles_lineage(order, roles, set(columns)),
        after=_roles_lineage(order, after, set(columns)),
    )]


def _register_previews() -> None:
    from turbotab.core.consequences import register_consequence

    register_consequence("apply_repair", repair_views)


_register_previews()


__all__ = [
    "CONSEQUENCE_WORDS", "FAMILIES", "Family", "KJ_PER_KCAL", "LABEL_WORDS", "LOD_FACTORS",
    "NUMBER_SHARE", "OfferContext", "RepairOption", "SAS_ZERO", "admits", "ambiguous_date_finding",
    "annotate", "attach", "column_expressions", "date_formats", "deferred_to",
    "disposition_writers", "evaluate", "exclusion_rules", "family_for", "infinite_counts",
    "infinite_finding", "is_sas_zero", "matching_answer", "number_sql", "offer",
    "read_text_numbers", "register_family", "repair_views", "sas_zero_counts", "sas_zero_finding",
    "text_number_findings", "unusable_columns",
]

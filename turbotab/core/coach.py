"""The coach: data-grounded notes on the stage's pictures and one line on a decision card.

Nolan's ruling (2026-10-01, M2_CONTRACT §6): a consequence view carries at most two coach notes,
each at most 12 words, pointing at part of the picture — a column, a range on the value axis, some
points, a row-flow step — and drawn in the coach's amber. A decision card carries at most one coach
line. Every note is a fact about the user's own rows ("`194` rows below `500` kcal: likely
under-reporting"). None names an option as the answer, and none pre-selects anything: the coach
says what the data shows, and the user decides.

| Where                      | Notes                                                              |
|----------------------------|--------------------------------------------------------------------|
| ``set_exclusions`` preview | rows under the floor and over the ceiling, on the cut; who leaves  |
| ``set_missing`` preview    | blanks that likely mean "not asked"; what they alone cost a        |
|                            | complete-case analysis                                             |
| ``set_energy_adjustment``  | how tightly the nutrient tracks energy now, and what is left after |
| ``set_aggregation``        | what combining does to replicates or time points; units it cannot  |
|                            | combine; an outcome that varies within a unit                      |
| finding evidence           | implausible intakes, energy's grip on a nutrient, repeats, blanks  |
| the decision card          | ``proposals.coach``: exclusions and missing values                 |

Held-out discipline: a preview's notes read the preview's own pool (every row before the split,
every row but the held-out ones after), never a sealed row. Evidence notes read what the finding
read. The card lines are made with the proposals, which are pre-seal answers about the data as
loaded (their basis says so); the energy question's card gets no line, because it is a modeling
choice made after the seal and the proposals read every row.

The eligibility question withholds the outcome's distribution (ROADMAP lockbox constitution §04),
so no exclusions note ever describes the outcome's values.

Annotators never fail a preview: one that raises is logged and skipped.
"""
from __future__ import annotations

import logging
import math
import re
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from turbotab.core.consequences import (
    COACH_WORDS, MAX_COACH, CoachAnchor, CoachNote, PreviewContext,
)
from turbotab.core.voice import count, finish, listing, tick, words

log = logging.getLogger(__name__)

# ── notes ────────────────────────────────────────────────────────────────────


def note(text: str, kind: str, ref: Any) -> CoachNote | None:
    """A finished note, or None when it would break the budget (a note is never cut mid-claim)."""
    text = finish(text)
    if words(text) > COACH_WORDS:
        log.warning("coach note over %d words, left out: %s", COACH_WORDS, text)
        return None
    if isinstance(ref, (list, tuple)):
        ref = [float(x) for x in ref]
    return CoachNote(text=text, anchor=CoachAnchor(kind=kind, ref=ref))


def attach(view: Any, notes: Iterable[CoachNote | None]) -> None:
    """Add notes to a view, keeping at most :data:`MAX_COACH` (the first ones win)."""
    current = list(getattr(view, "coach", []) or [])
    for n in notes:
        if n is not None and len(current) < MAX_COACH and n.text not in {c.text for c in current}:
            current.append(n)
    view.coach = current


def r_text(r: float) -> str:
    """A correlation as written: two decimals, never ``-0.00``."""
    return tick("0.00" if abs(r) < 0.005 else f"{r:.2f}")


def pct(share: float) -> str:
    return tick(f"{share:.0%}" if share >= 0.01 or share == 0 else "<1%")


def value(x: float) -> str:
    """A cut-off or a summary as written: whole numbers with separators, else three figures."""
    x = float(x)
    if x.is_integer() and abs(x) < 1e15:
        return tick(f"{int(x):,}")
    return tick(f"{x:,.3g}" if abs(x) < 1000 else f"{x:,.0f}")


# ── the registry ─────────────────────────────────────────────────────────────

Annotator = Callable[[Any, list, PreviewContext], None]
_ANNOTATORS: dict[str, list[Annotator]] = {}


def register_coach(kind: str, fn: Annotator) -> Annotator:
    """``fn(decision, views, ctx)`` adds notes to ``views`` (with :func:`attach`) for ``kind``."""
    _ANNOTATORS.setdefault(kind, []).append(fn)
    return fn


def annotate(decision: Any, views: list, ctx: PreviewContext) -> list:
    """Run every annotator for ``decision.kind`` over the planned views. Never raises."""
    for fn in _ANNOTATORS.get(decision.kind, ()):
        try:
            fn(decision, views, ctx)
        except Exception:  # noqa: BLE001 - a note is a courtesy; the preview stands without it
            log.exception("coach annotator %s failed for %s", getattr(fn, "__name__", fn), decision.kind)
    return views


# ── shared readings ──────────────────────────────────────────────────────────


def _pool(ctx: PreviewContext) -> Any:
    """The preview's pool: every row before the split, every row but the held-out ones after."""
    if ctx.sealed_row_ids is None:
        return None
    return ctx.unsealed_row_ids()


def _read(ctx: PreviewContext, columns: Sequence[str]) -> pd.DataFrame:
    wanted = [c for c in dict.fromkeys(columns) if c and c in ctx.datastore.columns]
    return ctx.datastore.materialize(wanted, _pool(ctx))


def _measured(frame: pd.DataFrame, target: str | None) -> pd.Series:
    if target and target in frame.columns:
        return frame[target].notna()
    return pd.Series(True, index=frame.index)


def _is_energy(column: str) -> bool:
    from turbotab.core.stages.proposals import is_energy_name

    try:
        return is_energy_name(column)
    except Exception:  # noqa: BLE001 - a name that cannot be read is not energy
        return False


def _unit_suffix(column: str) -> str:
    from turbotab.core.stages.rows import _unit

    unit = _unit(column)
    return f" {unit}" if unit else ""


def _first(views: Sequence[Any], kind: str) -> Any:
    return next((v for v in views if getattr(v, "kind", None) == kind), None)


def _below_above(frame: pd.DataFrame, rule: Any, base: pd.Series) -> tuple[int, int]:
    """Rows under the rule's floor and over its ceiling (each judged by its own level's range)."""
    from turbotab.core.stages.proposals import rule_excludes

    lows = rule.model_copy(update={
        "high": None,
        "by": None if rule.by is None else rule.by.model_copy(update={
            "ranges": {k: (lo, None) for k, (lo, _) in rule.by.ranges.items()}})})
    highs = rule.model_copy(update={
        "low": None,
        "by": None if rule.by is None else rule.by.model_copy(update={
            "ranges": {k: (None, hi) for k, (_, hi) in rule.by.ranges.items()}})})
    below = int((rule_excludes(frame, lows) & base).sum()) if _has(lows) else 0
    above = int((rule_excludes(frame, highs) & base).sum()) if _has(highs) else 0
    return below, above


def _has(rule: Any) -> bool:
    if rule.low is not None or rule.high is not None:
        return True
    return bool(rule.by and any(lo is not None or hi is not None for lo, hi in rule.by.ranges.values()))


def _bounds(rule: Any) -> tuple[list[float], list[float]]:
    lows = [b for b in [rule.low] if b is not None]
    highs = [b for b in [rule.high] if b is not None]
    if rule.by is not None:
        lows += [lo for lo, _ in rule.by.ranges.values() if lo is not None]
        highs += [hi for _, hi in rule.by.ranges.values() if hi is not None]
    return [float(x) for x in lows], [float(x) for x in highs]


def range_notes(column: str, below: int, above: int, lows: Sequence[float], highs: Sequence[float],
                axis: tuple[float, float], *, by: str | None = None) -> list[CoachNote | None]:
    """"`194` rows below `500` kcal: likely under-reporting" and its mirror, anchored to the tails."""
    energy = _is_energy(column)
    suffix = _unit_suffix(column)
    lo_axis, hi_axis = axis
    out: list[CoachNote | None] = []
    if below and lows:
        where = (f"under their {tick(by)} floor" if by and len(set(lows)) > 1
                 else f"below {value(max(lows))}{suffix}")
        text = f"{count(below)} {'row' if below == 1 else 'rows'} {where}"
        text += ": likely under-reporting" if energy else ""
        out.append(note(text, "range", [min(lo_axis, min(lows)), max(lows)]))
    if above and highs:
        where = (f"over their {tick(by)} ceiling" if by and len(set(highs)) > 1
                 else f"above {value(min(highs))}{suffix}")
        text = f"{count(above)} {'row' if above == 1 else 'rows'} {where}"
        text += ": likely over-reporting" if energy else ""
        out.append(note(text, "range", [min(highs), max(hi_axis, max(highs))]))
    return out


def _axis(view: Any) -> tuple[float, float]:
    edges = list(getattr(getattr(view, "before", None), "edges", []) or [])
    return (float(edges[0]), float(edges[-1])) if edges else (-math.inf, math.inf)


# ── set_exclusions (eligibility) ─────────────────────────────────────────────

_BMI = re.compile(r"(^|[^a-z])bmi([^a-z]|$)", re.I)


def exclusions_coach(decision: Any, views: list, ctx: PreviewContext) -> None:
    rules = list(decision.rules)
    target = ctx.state.target
    if not rules:
        return
    cut = _first(views, "distribution")
    flow = _first(views, "row_flow")
    rule_columns = {r.column for r in rules} | {r.by.column for r in rules if r.by is not None}
    bmi = next((c for c in ctx.datastore.columns if _BMI.search(str(c)) and c != target
                and c not in rule_columns), None)
    frame = _read(ctx, [*rule_columns, target, bmi])
    base = _measured(frame, target)
    if cut is not None and cut.column != target:  # the outcome's values are withheld here
        rule = next((r for r in rules if r.column == cut.column), None)
        if rule is not None:
            below, above = _below_above(frame, rule, base)
            lows, highs = _bounds(rule)
            attach(cut, range_notes(rule.column, below, above, lows, highs, _axis(cut),
                                    by=rule.by.column if rule.by is not None else None))
    if flow is not None and bmi is not None and bmi in frame.columns:
        from turbotab.core.stages.proposals import rule_excludes

        out = pd.Series(False, index=frame.index)
        for rule in rules:
            out |= rule_excludes(frame, rule)
        values = pd.to_numeric(frame[bmi], errors="coerce")
        gone, kept = values[out & base].dropna(), values[~out & base].dropna()
        step = next((s.key for s in flow.after if s.key.startswith("exclusion:")), None)
        if len(gone) >= 10 and len(kept) >= 10 and step is not None:
            attach(flow, [note(f"Excluded rows' median {tick(bmi)} is {value(round(gone.median(), 1))}, "
                               f"against {value(round(kept.median(), 1))} kept", "step", step)])


# ── set_missing ──────────────────────────────────────────────────────────────


def _not_asked(ctx: PreviewContext) -> list[str]:
    proposals = ctx.artifact("proposals")
    data = getattr(proposals, "data", proposals)
    reading = (data.get("missing") or {}) if isinstance(data, Mapping) else {}
    return [str(e["column"]) for e in reading.get("columns") or [] if e.get("likely_not_asked")]


def missing_coach(decision: Any, views: list, ctx: PreviewContext) -> None:
    if not ctx.state.roles:
        return
    target = ctx.state.target
    table = _first(views, "table_focus")
    flow = _first(views, "row_flow")
    likely = _not_asked(ctx)
    drop = set(getattr(decision, "drop_columns", []) or [])
    shown = list(table.columns_before) if table is not None else []
    about = [c for c in likely if c in shown] or ([c for c in shown if c not in drop][:1] if shown else [])
    frame = _read(ctx, [target, *about, *likely])
    base = _measured(frame, target)
    n_base = int(base.sum())
    if not n_base:
        return
    if table is not None:
        notes = []
        for c in about[:MAX_COACH]:
            share = float((frame[c].isna() & base).sum()) / n_base
            if not share:
                continue
            text = (f"{tick(c)} blank on {pct(share)}: likely a question not asked" if c in likely
                    else f"{tick(c)} is blank on {pct(share)} of these rows")
            notes.append(note(text, "column", c))
        attach(table, notes)
    keep = [c for c in likely if c in frame.columns and c not in drop]
    if flow is not None and decision.strategy == "complete_case" and keep:
        n = int((frame[keep].isna().any(axis=1) & base).sum())
        if n:
            attach(flow, [note(f"{listing(keep, limit=2)} {'is' if len(keep) == 1 else 'are'} blank "
                               f"on {count(n)} of these rows", "step", "complete_cases")])


# ── set_energy_adjustment ────────────────────────────────────────────────────


def energy_coach(decision: Any, views: list, ctx: PreviewContext) -> None:
    rel = _first(views, "relationship")
    if rel is None or rel.r_before is None:
        return
    nutrient, energy = rel.y_label_before, rel.x_label
    r0, r1 = float(rel.r_before), rel.r_after
    if abs(r0) >= 0.3:
        notes = [note(f"{tick(nutrient)} tracks {tick(energy)} at r {r_text(r0)}: mostly how much "
                      f"people eat", "column", nutrient)]
    else:  # an adjustment already on record: the picture starts from its result
        notes = [note(f"Now r {r_text(r0)}: {tick(nutrient)} barely tracks {tick(energy)}",
                      "column", nutrient)]
    if decision.method == "none":
        notes.append(note(f"Unadjusted, {tick(nutrient)} still carries how much people eat",
                          "column", energy))
    elif r1 is not None:
        r1 = float(r1)
        if abs(r1 - r0) < 0.05:
            notes.append(note(f"r stays {r_text(r0)}: {tick(energy)} enters the model beside it",
                              "column", energy))
        elif abs(r1) < 0.1:
            notes.append(note(f"With this method r {r_text(r1)}: what is left is composition",
                              "column", energy))
        else:
            notes.append(note(f"With this method r {r_text(r1)}: some of energy's signal remains",
                              "column", energy))
    attach(rel, notes)


# ── set_aggregation ──────────────────────────────────────────────────────────


def _unit_column(state: Any) -> str | None:
    grain = getattr(state, "grain", None)
    return getattr(grain, "id_column", None) if grain is not None else None


def _structure(ctx: PreviewContext) -> dict[str, Any] | None:
    """The structure artifact (the stated repeats reading), when it is fresh."""
    try:
        found = ctx.artifact("structure")
    except Exception:  # noqa: BLE001 - no reading: only the answers speak
        return None
    found = getattr(found, "data", found)
    return found if isinstance(found, dict) else None


def _time_column(state: Any, structure: Mapping[str, Any] | None = None) -> str | None:
    """The column that orders a unit's rows, as the working stage reads it: the answers' time
    column, else the one the stated reading spaced the rows by."""
    from turbotab.core.stages.working import time_column

    return time_column(state, structure)


def _anchor_for(view: Any, unit: str) -> tuple[str, Any]:
    if getattr(view, "kind", None) == "row_flow" and view.after:
        return "step", view.after[-1].key
    if getattr(view, "kind", None) == "distribution":
        return "column", view.column
    return "column", unit


def _before_combining(ctx: PreviewContext, columns: Sequence[str]) -> pd.DataFrame | None:
    """These columns of the table before any rows are combined: the oriented table, every row.

    Once an aggregation is recorded the working table has one row per unit, so a note read from
    it would find no replicates to speak of (M2_CONTRACT §12.4). Aggregation is pre-seal (Decision
    A refuses it after), so every row is the pool. Falls back to the datastore's table.
    """
    from turbotab.core.stages.working import _bundle_table

    try:
        oriented = ctx.artifact("oriented")
    except Exception:  # noqa: BLE001 - no oriented table: read what the datastore holds
        oriented = None
    path = _bundle_table(oriented) if oriented is not None else None
    if path is None:
        return None
    import pyarrow.parquet as pq

    present = set(pq.read_schema(path).names)
    wanted = [c for c in dict.fromkeys(columns) if c and c in present]
    return pq.read_table(path, columns=wanted).to_pandas() if wanted else None


def aggregation_coach(decision: Any, views: list, ctx: PreviewContext) -> None:
    unit = _unit_column(ctx.state)
    if not views or not unit:
        return
    target = ctx.state.target
    frame = _before_combining(ctx, [unit, target])
    if frame is None:
        if unit not in ctx.datastore.columns:
            return
        frame = _read(ctx, [unit, target])
    if unit not in frame.columns:
        return
    sizes = frame[unit].dropna().value_counts()
    if sizes.empty:
        return
    k = int(np.median(sizes.to_numpy()))
    singles = int((sizes == 1).sum())
    from turbotab.core.stages.working import effective_repeat_kind

    structure = _structure(ctx)
    kind = effective_repeat_kind(ctx.state, structure)  # answered, else stated (a skip)
    primary = views[0]
    anchor = _anchor_for(primary, unit)
    notes: list[CoachNote | None] = []
    if decision.method == "mean" and k >= 2:
        if kind == "repeats":
            notes.append(note(f"Averaging {tick(k)} replicates cuts within-person variance "
                              f"{tick(k)}-fold", *anchor))
        elif kind == "time_points":
            notes.append(note(f"Averaging {tick(k)} time points erases change between them", *anchor))
    if decision.method == "change" and singles:
        notes.append(note(f"{count(singles)} {tick(unit)} values have one row: no change to take",
                          "column", unit))
    if decision.method in ("first", "last", "change") and not _time_column(ctx.state, structure):
        notes.append(note("No time column is named, so order is file order", "column", unit))
    if target and target in frame.columns and decision.outcome is None:
        varies = frame.dropna(subset=[target]).groupby(unit)[target].nunique()
        n_vary = int((varies > 1).sum())
        if n_vary:
            notes.append(note(f"{tick(target)} differs within {count(n_vary)} of {count(len(sizes))} "
                              f"{tick(unit)} values", "column", target))
    attach(primary, notes)


register_coach("set_exclusions", exclusions_coach)
register_coach("set_missing", missing_coach)
register_coach("set_energy_adjustment", energy_coach)
register_coach("set_aggregation", aggregation_coach)


# ── finding evidence ─────────────────────────────────────────────────────────

EvidenceAnnotator = Callable[[dict, list, Any], None]
_EVIDENCE: dict[str, EvidenceAnnotator] = {}


def register_evidence_coach(family: str, fn: EvidenceAnnotator) -> EvidenceAnnotator:
    _EVIDENCE[family] = fn
    return fn


def annotate_evidence(family: str, finding: dict, views: list, ctx: Any) -> list:
    """Notes on a finding's evidence views (its family's annotator, else the generic one)."""
    fn = _EVIDENCE.get(family, generic_evidence_coach)
    try:
        fn(finding, views, ctx)
    except Exception:  # noqa: BLE001 - evidence stands without its notes
        log.exception("evidence coach failed for %s", family)
    return views


def implausible_evidence_coach(finding: dict, views: list, ctx: Any) -> None:
    cut = _first(views, "distribution")
    if cut is None or not cut.marks:
        return
    lows = [m.value for m in cut.marks if m.value <= float(np.median([m.value for m in cut.marks]))]
    highs = [m.value for m in cut.marks if m.value not in lows]
    values = pd.to_numeric(ctx.datastore.materialize([cut.column])[cut.column], errors="coerce")
    below = int((values < min(lows)).sum()) if lows else 0
    above = int((values > max(highs)).sum()) if highs else 0
    attach(cut, range_notes(cut.column, below, above, lows[:1], highs[-1:], _axis(cut)))


def energy_evidence_coach(finding: dict, views: list, ctx: Any) -> None:
    rel = _first(views, "relationship")
    if rel is None or len(rel.points_before) < 20:
        return
    pts = np.asarray(rel.points_before, dtype=float)
    e, n = pts[:, 0], pts[:, 1]
    lo, hi = np.quantile(e, 0.1), np.quantile(e, 0.9)
    bottom, top = n[e <= lo], n[e >= hi]
    if not len(bottom) or not len(top) or np.mean(bottom) <= 0:
        return
    ratio = float(np.mean(top) / np.mean(bottom))
    if ratio < 1.2:
        return
    attach(rel, [note(f"Top tenth by {tick(rel.x_label)} has {tick(f'{ratio:.1f}×')} the "
                      f"{tick(rel.y_label_before)} of the bottom", "range", [float(hi), float(e.max())])])


def repeats_evidence_coach(finding: dict, views: list, ctx: Any) -> None:
    table = _first(views, "table_focus")
    columns = [c for c in finding.get("affected_columns") or [] if c in ctx.datastore.columns]
    if table is None or not columns:
        return
    sizes = ctx.datastore.materialize([columns[0]])[columns[0]].dropna().value_counts()
    if sizes.empty:
        return
    k, most = int(np.median(sizes.to_numpy())), int(sizes.max())
    text = (f"Most {tick(columns[0])} values have {tick(k)} rows; up to {tick(most)}" if most > k
            else f"Every {tick(columns[0])} value has {tick(k)} rows")
    attach(table, [note(text, "column", columns[0])])


def generic_evidence_coach(finding: dict, views: list, ctx: Any) -> None:
    table = _first(views, "table_focus")
    if table is None or not table.columns_before:
        return
    frame = ctx.datastore.materialize(list(table.columns_before))
    blank = frame.isna().mean()
    worst = str(blank.idxmax()) if len(blank) else None
    if worst is None or not blank[worst]:
        return
    attach(table, [note(f"{tick(worst)} is blank on {pct(float(blank[worst]))} of rows",
                        "column", worst)])


register_evidence_coach("pack::dietary::implausible_intake", implausible_evidence_coach)
register_evidence_coach("pack::dietary::energy_adjustment", energy_evidence_coach)
register_evidence_coach("voice::repeats", repeats_evidence_coach)
register_evidence_coach("voice::flag", lambda finding, views, ctx: None)  # the caption says it
register_evidence_coach("voice::identifier", lambda finding, views, ctx: None)


# ── the decision card ────────────────────────────────────────────────────────


def card_lines(frame: pd.DataFrame, *, target: str | None, energy: str | None, unit: str,
               missing: Mapping[str, Any]) -> dict[str, CoachNote]:
    """At most one line per decision card, from the proposals' own reading of the data as loaded.

    ``exclusions``: rows outside the pack's plausible-intake range (the detector's own bounds).
    ``missing``: the blankest column whose blanks likely mean "not asked".
    """
    out: dict[str, CoachNote] = {}
    base = _measured(frame, target)
    if energy and energy in frame.columns:
        try:
            from turbotab.packs import _PLAUSIBLE_KCAL as kcal  # the detector's own range
        except ImportError:  # pragma: no cover - the legacy pack always defines it
            kcal = (500.0, 5000.0)
        factor = 4.184 if unit == "kj" else 1.0
        low, high = kcal[0] * factor, kcal[1] * factor
        e = pd.to_numeric(frame[energy], errors="coerce")
        below, above = int(((e < low) & base).sum()), int(((e > high) & base).sum())
        word = "kJ" if unit == "kj" else "kcal"
        if below:
            line = note(f"{count(below)} {'row' if below == 1 else 'rows'} below {value(round(low))} "
                        f"{word}: likely under-reporting", "column", energy)
        elif above:
            line = note(f"{count(above)} {'row' if above == 1 else 'rows'} above {value(round(high))} "
                        f"{word}: likely over-reporting", "column", energy)
        else:
            line = None
        if line is not None:
            out["exclusions"] = line
    likely = [e for e in missing.get("columns") or [] if e.get("likely_not_asked")]
    if likely:
        top = likely[0]
        line = note(f"{tick(top['column'])} blank on {pct(float(top['share']))}: likely a question "
                    f"not asked", "column", str(top["column"]))
        if line is not None:
            out["missing"] = line
    return out


__all__ = [
    "annotate", "annotate_evidence", "attach", "card_lines", "note", "range_notes",
    "register_coach", "register_evidence_coach",
]

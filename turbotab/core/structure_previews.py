"""Previews for the structural answers that change what a row is (M2_CONTRACT §1–§2).

* ``set_unit`` — the participant flow with and without combining: ``600`` records → ``300`` rows.
* ``set_aggregation`` — that flow, and the reshape itself on two units: their records, then the one
  row each becomes (a ``table_focus`` whose story is *the records* → *combined*). The combined
  values come from the working stage's own SQL (``stages.working.combine_sql``) run on those units,
  so a preview and the table it foretells cannot disagree.

Aggregation is pre-seal (Decision A refuses it after), so nothing here reads held-out rows.

The other structural answers, found without a picture on the M2 journeys:

* ``set_orientation`` — the table's corner as supplied, then turned to one row per sample.
* ``set_grain`` — two units' rows side by side, and how many rows each unit has.
* ``set_repeat_kind`` — a note: what the combining menu and the temporal question become.
* ``set_temporal`` — the seal's own chronological draw at the usual size, or a random one.
* ``set_event`` — a note: which level becomes 1, with its rows.
"""
from __future__ import annotations

from typing import Any

from turbotab.core.consequences import (
    DistributionView,
    FrameRow,
    PreviewContext,
    RowFlowView,
    RowStep,
    TableFocusView,
    TableFrame,
    TableRow,
    register_consequence,
)

SHOWN_UNITS = 2
SHOWN_NUMBERS = 4
# The reshape shows up to this many units, as long as their records fit a table_focus (≤ 8 rows),
# so rows can be seen meeting their partners (M2_CONTRACT §11).
RESHAPE_UNITS = 4
RESHAPE_ROWS = 8
# What each combining rule (stages.working.column_rule) does, for the reshape's caption.
_RULE_CAPTIONS = {
    "mean": "amounts averaged",
    "change": "amounts as last minus first",
    "first": "the first record kept",
    "last": "the last record kept",
    "mode": "codes by their most frequent value",
    "constant": "unchanging columns as they are",
}


def _data(value: Any) -> Any:
    return getattr(value, "data", value)


def _flow(ctx: PreviewContext, combine: bool) -> tuple[list[RowFlowView], dict[str, Any]] | None:
    oriented = ctx.artifact("oriented")
    structure = _data(ctx.artifact("structure")) or {}
    units = structure.get("units")
    if oriented is None or not units:
        return None
    n_rows = int(_data(oriented)["n_rows"])
    key, n_units = str(units["column"]), int(units["n_units"])
    loaded = RowStep(key="loaded", label="Records in the table", n=n_rows)
    ctx.read["counted"] = n_rows
    if combine:
        after = [loaded, RowStep(key="combined", label=f"One row per `{key}`", n=n_units,
                                 dropped=n_rows - n_units,
                                 reason=f"each `{key}`'s records combined into one row")]
        view = RowFlowView(
            title=f"One row per {key}",
            caption=f"`{n_rows:,}` records become `{n_units:,}` rows, one per `{key}`.",
            emphasis=["combined"], before=[loaded], after=after)
    else:
        view = RowFlowView(
            title="Records stay as rows",
            caption=f"`{n_rows:,}` records stay rows; each `{key}`'s rows are held out together.",
            before=[loaded], after=[loaded])
    return [view], {"key": key, "n_rows": n_rows, "n_units": n_units, "structure": structure}


def unit_views(decision: Any, ctx: PreviewContext) -> list[Any]:
    flow = _flow(ctx, decision.unit == "unit")
    return flow[0] if flow else []


def _reshape(decision: Any, ctx: PreviewContext, facts: dict[str, Any]) -> TableFocusView | None:
    import duckdb

    from turbotab.core.consequences import CAPTION_WORDS, clip_words
    from turbotab.core.datastore import json_safe
    from turbotab.core.decisions import AggregationSpec
    from turbotab.core.stages.working import (
        _K,
        _M,
        _UNIT,
        ROW_ID,
        StructureError,
        _bundle_table,
        _ident,
        _lit,
        aggregation_plan,
        combine_sql,
        prepare_combine,
        rank_units,
        repair_expressions,
        source_sql,
    )

    oriented = ctx.artifact("oriented")
    info = _data(oriented)
    names = [str(c["name"]) for c in info["columns"]]
    state = ctx.state.model_copy(update={"aggregation": AggregationSpec(
        method=decision.method, outcome=decision.outcome, columns=dict(decision.columns))})
    plan = aggregation_plan(state, set(names), facts["structure"])
    if plan is None:
        return None
    key = plan["id_column"]
    source = _bundle_table(oriented)
    repairs = repair_expressions(_data(ctx.artifact("findings")), state.findings)
    q = _ident(key)
    con = duckdb.connect()
    try:
        src = source_sql(source, names, repairs)
        try:  # the stage's own decisions, over the whole table: the order, each column's rule
            prep = prepare_combine(con, src, plan, state.target)
        except StructureError as exc:
            ctx.read["note"] = str(exc)
            return None
        candidates = con.execute(
            f"SELECT CAST({q} AS VARCHAR) AS u, count(*) AS n FROM read_parquet({_lit(source)}) "
            f"WHERE {q} IS NOT NULL GROUP BY u HAVING count(*) > 1 ORDER BY min({ROW_ID}) "
            f"LIMIT {RESHAPE_UNITS}").fetchall()
        picked, rows_shown = [], 0
        for u, n in candidates:
            if len(picked) >= SHOWN_UNITS and rows_shown + int(n) > RESHAPE_ROWS:
                break
            picked.append(u)
            rows_shown += int(n)
        if not picked:
            return None
        chosen = ", ".join(_lit(u) for u in picked)
        subset = f"(SELECT * FROM {src} WHERE CAST({q} AS VARCHAR) IN ({chosen}))"
        rank_units(con, subset, plan, prep["order_expr"])
        target = prep["target"]
        select = combine_sql(prep["names"], prep["physical"], plan, target, prep["varies"],
                             prep["rules"])
        combined = con.execute(select).df()
        records = con.execute(f"SELECT * FROM ranked ORDER BY {_UNIT}, {_K}").df()
    finally:
        con.close()

    rules, kinds = prep["rules"], prep["kinds"]
    numeric = [c for c in prep["others"] if prep["numeric"][c]]
    numeric.sort(key=lambda c: not kinds[c]["varied"])  # the ones combining changes first
    shown = [c for c in dict.fromkeys([key, plan["time_column"], target]) if c]
    shown += [c for c in numeric if c not in shown][:SHOWN_NUMBERS]
    unit_of = records.groupby(_UNIT).ngroup().to_numpy()
    rows: list[TableRow] = []
    changed: list[tuple[int, str]] = []
    record_frames: list[FrameRow] = []
    label_of = {int(u): str(json_safe(v)) for u, v in zip(records[_UNIT], records[key])}
    for i, record in records.iterrows():
        rid = int(record[ROW_ID])
        before = {c: json_safe(record[c]) for c in shown}
        after = {c: json_safe(combined.iloc[int(unit_of[i])][c]) for c in shown}
        rows.append(TableRow(row_id=rid, before=before, after=after))
        record_frames.append(FrameRow(row_id=rid, values=before, unit=label_of[int(record[_UNIT])],
                                      sources=[rid]))
        changed += [(rid, c) for c in shown if before[c] != after[c]]
    # The row map, made visible: a combined row stands for all of its unit's records; a kept one
    # (first, last) is that record, so it keeps the record's identity (M2_CONTRACT §11).
    combined_frames: list[FrameRow] = []
    for j, (u, group) in enumerate(records.groupby(_UNIT, sort=True)):
        ids = [int(x) for x in group.sort_values(_K)[ROW_ID]]
        if decision.method in ("first", "last"):  # the dated record kept; undated ones never are
            at = group[_K] == (1 if decision.method == "first" else group[_M])
            ids = [int(x) for x in group.loc[at, ROW_ID]] or ids[:1]
        combined_frames.append(FrameRow(row_id=ids[0] if len(ids) == 1 else min(ids),
                                        values={c: json_safe(combined.iloc[j][c]) for c in shown},
                                        unit=label_of[int(u)], sources=sorted(ids)))
    used = {rules[c] for c in prep["others"]}
    parts = [_RULE_CAPTIONS[r] for r in ("mean", "change", "first", "last", "mode", "constant")
             if r in used]
    return TableFocusView(
        title=f"{len(picked)} units' records, combined",
        caption=clip_words("Each column by what it holds: " + "; ".join(parts) + ".", CAPTION_WORDS),
        emphasis=shown[1:], columns_before=shown, columns_after=shown, rows=rows, changed=changed,
        n_affected_columns=len(numeric),
        story=[TableFrame(label=f"Each {key}'s records", columns=shown, rows=record_frames),
               TableFrame(label="Combined into one row each", columns=shown, rows=combined_frames)])


# The spread of one column, record by record and unit by unit (the /lab/m2 bar): what combining
# does to the values the model will see. Only columns that vary within a unit are candidates, and
# only the first few numeric ones, so a wide table costs what a narrow one does.
SPREAD_CANDIDATES = 12
_SPREAD_WORDS = {"mean": "mean", "first": "first record", "last": "last record",
                 "mode": "most frequent value"}


def _spread(decision: Any, ctx: PreviewContext, facts: dict[str, Any]) -> DistributionView | None:
    """Each record's value against each unit's combined value, on one axis.

    Never the outcome (eligibility has not been asked yet; constitution §04), never an identifier
    or the time column. The column is the one whose within-unit share of the variance is largest:
    the one combining changes most. Read over every row, and the basis says so.
    """
    import numpy as np
    import pandas as pd

    from turbotab.core.consequences import CAPTION_WORDS, TITLE_WORDS, _histogram_pair, clip_words
    from turbotab.core.decisions import AggregationSpec
    from turbotab.core.stages.working import NUMERIC, ROW_ID, _bundle_table, _ident, aggregation_plan
    from turbotab.core.stages.working import repair_expressions, source_sql

    if decision.method == "change":
        return None  # last minus first is a change, on another axis than the records
    oriented = ctx.artifact("oriented")
    info = _data(oriented)
    names = [str(c["name"]) for c in info["columns"]]
    dtypes = {str(c["name"]): str(c["dtype"]) for c in info["columns"]}
    state = ctx.state.model_copy(update={"aggregation": AggregationSpec(
        method=decision.method, outcome=decision.outcome, columns=dict(decision.columns))})
    plan = aggregation_plan(state, set(names), facts["structure"])
    if plan is None:
        return None
    key, order = plan["id_column"], plan["time_column"]
    roles = state.roles or {}
    structure = facts["structure"]
    # Not a measurement: the unit, its dates and visit or replicate indices, identifiers, the outcome.
    skip = {key, order, state.target, *(structure.get("time_columns") or []),
            (structure.get("repeats") or {}).get("replicate_index"),
            *(c for c, r in roles.items() if r in ("identifier", "ignore"))}
    candidates = [c for c in names if dtypes.get(c) in NUMERIC and c not in skip][:SPREAD_CANDIDATES]
    if not candidates:
        return None
    import duckdb

    repairs = repair_expressions(_data(ctx.artifact("findings")), state.findings)
    wanted = list(dict.fromkeys([key, *([order] if order else []), *candidates]))
    src = source_sql(_bundle_table(oriented), names, repairs)
    con = duckdb.connect()
    try:
        frame = con.execute(f"SELECT {', '.join(_ident(c) for c in wanted)}, {ROW_ID} FROM {src} "
                            f"ORDER BY {ROW_ID}").df()
    finally:
        con.close()
    # A unit is the rows sharing the id; a row with no id is a unit of its own (as `working` does).
    unit = frame[key].astype(object).where(frame[key].notna(),
                                           "__row_" + frame[ROW_ID].astype(str))
    best, share = None, 0.0
    for c in candidates:
        v = pd.to_numeric(frame[c], errors="coerce")
        total = float(v.var(ddof=0)) if v.notna().sum() > 1 else 0.0
        if not total or not np.isfinite(total):
            continue
        within = float((v - v.groupby(unit).transform("mean")).pow(2).mean()) / total
        if within > share + 1e-12:
            best, share = c, within
    if best is None or share < 0.01:
        return None  # nothing varies within a unit: combining changes no value's spread
    if "dietary" in (state.lens or []):
        # The dietary lens reads total energy first (plausibility, adjustment): its spread, when
        # combining changes it, is the one the user is about to reason with.
        from turbotab.core.stages.proposals import energy_column

        energy = energy_column({str(c["name"]): c for c in info["columns"]}, roles)
        if energy in candidates:
            v = pd.to_numeric(frame[energy], errors="coerce")
            if float((v - v.groupby(unit).transform("mean")).pow(2).mean()) > 0.01 * float(v.var(ddof=0)):
                best = energy
    values = pd.to_numeric(frame[best], errors="coerce")
    # Each unit's value as the working stage computes it: its order, its rule for this column.
    from turbotab.core.stages.working import _UNIT, StructureError, _rule_sql, prepare_combine

    con = duckdb.connect()
    try:
        try:
            prep = prepare_combine(con, src, plan, state.target)
        except StructureError as exc:
            ctx.read["note"] = str(exc)
            return None
        rule = prep["rules"].get(best)
        if rule not in _SPREAD_WORDS:
            return None  # a change is on another axis than the records; a constant does not move
        combined = con.execute(
            f"SELECT {_rule_sql(best, rule, prep['physical'][best])} AS v FROM ranked "
            f"GROUP BY {_UNIT} ORDER BY {_UNIT}").df()["v"]
    finally:
        con.close()
    combined = pd.to_numeric(combined, errors="coerce")
    before, after = _histogram_pair(values.to_numpy(), combined.to_numpy())
    n_rows, n_units = int(len(values)), int(len(combined))
    ctx.read["counted"] = n_rows
    noun = _SPREAD_WORDS[rule]
    sd0, sd1 = float(values.std()), float(combined.std())
    caption = (f"`{n_rows:,}` records become `{n_units:,}` values of `{best}`, one per `{key}`: "
               f"SD `{sd0:,.3g}` → `{sd1:,.3g}`.")
    return DistributionView(
        title=clip_words(f"`{best}` per record and per {key}", TITLE_WORDS),
        caption=clip_words(caption, CAPTION_WORDS),
        emphasis=[best], column=best, before=before, after=after,
        before_label="Each record", after_label=f"Each {key}'s {noun}")


def aggregation_views(decision: Any, ctx: PreviewContext) -> list[Any]:
    flow = _flow(ctx, True)
    if flow is None:
        return []
    views: list[Any] = list(flow[0])
    reshape = _reshape(decision, ctx, flow[1])
    if reshape is not None:
        views.insert(0, reshape)
    spread = _spread(decision, ctx, flow[1])
    if spread is not None:
        views.append(spread)
    return views


# ── the other structural answers (found missing on the M2 journeys) ─────────────
# Orientation, grain and temporal change what a row is or how the seal is drawn, so each shows
# its picture before it is recorded; the event says, in a note, which rows become 1.

SHOWN_COLUMNS = 5
SHOWN_ROWS = 4
# The turn shows the table's corner at a table_focus's limits (≤ 8 rows either way round), so the
# picture reads as a table turning rather than a few cells (M2_CONTRACT §11, the /lab/m2 bar).
TURN_FEATURES = 8
TURN_SAMPLES = 8


def _oriented_source(ctx: PreviewContext) -> tuple[Any, Any] | None:
    from turbotab.core.stages.working import _bundle_table

    oriented = ctx.artifact("oriented")
    if oriented is None:
        return None
    path = _bundle_table(oriented)
    return (path, _data(oriented)) if path is not None else None


def grain_views(decision: Any, ctx: PreviewContext) -> list[Any]:
    """Repeated: two units' records side by side, and how many rows each unit has; one row per
    unit: the rows stay, each its own unit, and the held-out rows are drawn row by row."""
    import duckdb

    from turbotab.core.consequences import CAPTION_WORDS, clip_words
    from turbotab.core.datastore import json_safe
    from turbotab.core.stages.working import ROW_ID, _ident, _lit

    found = _oriented_source(ctx)
    if found is None:
        return []
    source, info = found
    n_rows = int(info["n_rows"])
    loaded = RowStep(key="loaded", label="Rows in the table", n=n_rows)
    ctx.read["counted"] = n_rows
    key = decision.id_column
    if decision.grain == "unknown":
        # "I don't know": the rows stay, and the seal that follows is drawn row by row and labeled
        # exploratory (its undetermined basis); the picture says what that answer costs.
        return [RowFlowView(
            title="Held out row by row",
            caption=clip_words(f"The `{n_rows:,}` rows are held out one by one; every held-out "
                               f"score is labeled exploratory.", CAPTION_WORDS),
            before=[loaded], after=[loaded])]
    if decision.grain != "repeated" or not key:
        # attested over a contradiction: the data repeats, so only the answer is said
        said = f" and no `{key}` repeats" if key and not decision.acknowledged else ""
        return [RowFlowView(
            title="Each row its own unit",
            caption=clip_words(f"Each of the `{n_rows:,}` rows is a different unit{said}; the "
                               f"held-out rows are drawn row by row.", CAPTION_WORDS),
            before=[loaded], after=[loaded])]
    names = [str(c["name"]) for c in info["columns"]]
    if key not in names:
        return []
    numeric = [str(c["name"]) for c in info["columns"]
               if str(c["dtype"]) in ("numeric", "integer") and str(c["name"]) != key]
    target = ctx.state.target if ctx.state.target in names else None
    shown = list(dict.fromkeys([key, *([target] if target else []), *numeric]))[:SHOWN_COLUMNS]
    q = _ident(key)
    rel = f"read_parquet({_lit(str(source))})"
    con = duckdb.connect()
    try:
        n_units, most = con.execute(
            f"SELECT count(*), max(n) FROM (SELECT count(*) AS n FROM {rel} "
            f"WHERE {q} IS NOT NULL GROUP BY {q})").fetchone()
        picked = [r[0] for r in con.execute(
            f"SELECT {q} FROM {rel} WHERE {q} IS NOT NULL GROUP BY {q} HAVING count(*) > 1 "
            f"ORDER BY min({ROW_ID}) LIMIT {SHOWN_UNITS}").fetchall()]
        records = (con.execute(
            f"SELECT {ROW_ID}, {', '.join(_ident(c) for c in shown)} FROM {rel} WHERE {q} IN "
            f"(SELECT {q} FROM {rel} WHERE {q} IS NOT NULL GROUP BY {q} HAVING count(*) > 1 "
            f"ORDER BY min({ROW_ID}) LIMIT {SHOWN_UNITS}) ORDER BY {ROW_ID} LIMIT 8").df()
                   if picked else None)
    finally:
        con.close()
    caption = (f"`{n_rows:,}` rows from `{int(n_units or 0):,}` `{key}` values, at most "
               f"`{int(most or 0)}` each; each one's rows are held out together.")
    if records is None or not len(records):
        return [RowFlowView(title=f"Rows per {key}", caption=clip_words(caption, CAPTION_WORDS),
                            before=[loaded], after=[loaded])]
    # One view: the units' rows, captioned with the counts (a second view repeating the caption
    # would answer nothing new; BLUEPRINT §11.2).
    rows = [TableRow(row_id=int(r[ROW_ID]), before={c: json_safe(r[c]) for c in shown},
                     after={c: json_safe(r[c]) for c in shown}) for _, r in records.iterrows()]
    frames = [FrameRow(row_id=row.row_id, values=row.before) for row in rows]
    return [TableFocusView(
        title=f"{len(picked)} {key} values, each in several rows",
        caption=clip_words(caption, CAPTION_WORDS), emphasis=[key],
        columns_before=shown, columns_after=shown, rows=rows, changed=[], n_affected_columns=0,
        story=[TableFrame(label=f"Each {key}'s rows", columns=shown, rows=frames)])]


def temporal_views(decision: Any, ctx: PreviewContext) -> list[Any]:
    """Yes: the seal's own chronological draw at the usual size (the latest units held out);
    no: a random draw, grouped by the unit."""
    import numpy as np

    from turbotab.core import seal
    from turbotab.core.consequences import CAPTION_WORDS, clip_words
    from turbotab.core.decisions import TemporalSpec

    store = ctx.datastore
    cohort = ctx.artifact("cohort")
    frames = getattr(cohort, "frames", None) or {}
    measured = frames.get("measured")
    universe = (measured["row_id"].to_numpy(dtype=np.int64) if measured is not None
                else np.arange(int(store.n_rows), dtype=np.int64))
    probe = ctx.state.model_copy(update={"temporal": TemporalSpec(
        temporal=decision.temporal, time_column=decision.time_column)})
    usual = getattr(seal, "USUAL", 0.2)
    draw = seal.seal_inputs(probe, universe, store, ctx.state.task, holdout=usual, seed=0,
                            structure=_data(ctx.artifact("structure")))
    n = int(len(universe))
    loaded = RowStep(key="loaded", label="Rows with the outcome", n=n)
    ctx.read["counted"] = n
    if draw.refusal:
        ctx.read["note"] = draw.refusal
        return []
    chron = draw.chronology
    if decision.temporal and chron is not None and chron.drawn and draw.held is not None:
        held = int(np.asarray(draw.held, dtype=bool).sum())
        after = [loaded, RowStep(key="holdout", label="Held out: the latest", n=n - held,
                                 dropped=held, reason=chron.sentence)]
        return [RowFlowView(title="The latest rows held out",
                            caption=clip_words(chron.sentence, CAPTION_WORDS),
                            emphasis=["holdout"], before=[loaded], after=after)]
    group = f", keeping each `{draw.grouped_by}`'s rows together" if draw.grouped_by else ""
    return [RowFlowView(title="Held-out rows drawn at random",
                        caption=clip_words(f"About `{usual:.0%}` of the `{n:,}` rows are held out "
                                           f"at random{group}.", CAPTION_WORDS),
                        before=[loaded], after=[loaded])]


def orientation_views(decision: Any, ctx: PreviewContext) -> list[Any]:
    """Feature-major: the table's corner as supplied, then turned (one row per sample)."""
    import duckdb

    from turbotab.core.consequences import CAPTION_WORDS, clip_words
    from turbotab.core.stages.working import ROW_ID, _ident, _lit

    found = _oriented_source(ctx)
    if found is None:
        return []
    source, info = found
    if _data(ctx.artifact("oriented")).get("transposed"):
        ctx.read["note"] = "The table is turned already; this preview reads it as supplied."
        return []
    n_rows = int(info["n_rows"])
    loaded = RowStep(key="loaded", label="Rows as supplied", n=n_rows)
    ctx.read["counted"] = n_rows
    if decision.orientation != "feature_major":
        return [RowFlowView(title="The table as supplied",
                            caption=clip_words(f"The `{n_rows:,}` rows stay as they are, one per "
                                               f"sample.", CAPTION_WORDS),
                            before=[loaded], after=[loaded])]
    from turbotab.core.stages.working import turn_plan, turned_corner

    con = duckdb.connect()
    try:
        # The stage's own partition and block functions (audit MA-04): the label, the feature
        # annotations kept beside the features, and the samples; never a guess of its own.
        plan = turn_plan(con, source, info, getattr(ctx.state, "feature_table", None))
        if plan["refusal"]:
            ctx.read["note"] = plan["refusal"]
            return []
        turned_frame = turned_corner(con, source, info, plan, TURN_SAMPLES, TURN_FEATURES)
        label, samples = plan["label_column"], list(plan["samples"][:TURN_SAMPLES])
        shown = [c for c in [label, *plan["annotations"][:2], *samples] if c]
        corner = con.execute(
            f"SELECT {ROW_ID}, {', '.join(_ident(c) for c in shown)} "
            f"FROM read_parquet({_lit(str(source))}) ORDER BY {ROW_ID} LIMIT {TURN_FEATURES}").df()
    finally:
        con.close()
    supplied = [FrameRow(row_id=int(r[ROW_ID]), values={c: _plain(r[c]) for c in shown})
                for _, r in corner.iterrows()]
    turned_cols = [str(c) for c in turned_frame.columns]
    turned = [FrameRow(row_id=i, values={c: _plain(r[c]) for c in turned_cols})
              for i, (_, r) in enumerate(turned_frame.iterrows())]
    kept = (f"; {', '.join(f'`{c}`' for c in plan['annotations'][:3])} stay beside the features"
            if plan["annotations"] else "")
    caption = (f"`{n_rows:,}` feature rows become columns: `{plan['n_samples']:,}` rows, one per "
               f"sample, named by the header{kept}.")
    rows = [TableRow(row_id=f.row_id, before=f.values, after=f.values) for f in turned]
    # One view: the corner, as supplied and then turned; the caption carries the counts.
    return [TableFocusView(
        title="The table turned round", caption=clip_words(caption, CAPTION_WORDS),
        emphasis=["sample_id"], columns_before=shown, columns_after=turned_cols,
        rows=rows, changed=[], n_affected_columns=0,
        story=[TableFrame(label="As supplied: features in rows", columns=shown, rows=supplied),
               TableFrame(label="Turned: one row per sample", columns=turned_cols, rows=turned)])]


def _plain(value: Any) -> Any:
    from turbotab.core.datastore import json_safe

    return json_safe(value)


def event_views(decision: Any, ctx: PreviewContext) -> list[Any]:
    """Which rows become 1 and which 0, in a note: the levels and their counts."""
    from turbotab.core.stages.rows import _level_key

    info = _data(ctx.artifact("target_info")) or {}
    classes = info.get("classes") or [] if info.get("column") == decision.column else []
    event = _level_key(decision.level)
    hit = [c for c in classes if _level_key(c.get("value")) == event]
    rest = [c for c in classes if _level_key(c.get("value")) != event]
    if not hit:
        return []
    others = " and ".join(f"`{_level_key(c['value'])}` (`{int(c['count']):,}` rows)" for c in rest)
    ctx.read["note"] = (f"`{event}` (`{int(hit[0]['count']):,}` rows) becomes 1 and {others} 0; "
                        f"scores and coefficients are about `{event}`.")
    return []


def repeat_kind_views(decision: Any, ctx: PreviewContext) -> list[Any]:
    """What the answer changes downstream, in a note: the combining menu and the temporal question."""
    if decision.repeat_kind == "repeats":
        ctx.read["note"] = ("As repeats of one measurement, combining a unit's rows recommends their "
                            "mean, and no temporal question follows.")
    else:
        ctx.read["note"] = ("As time points, combining has no default (first, last or change), and "
                            "kept as rows they raise the temporal question.")
    return []


register_consequence("set_unit", unit_views)
register_consequence("set_aggregation", aggregation_views)
register_consequence("set_repeat_kind", repeat_kind_views)
register_consequence("set_grain", grain_views)
register_consequence("set_temporal", temporal_views)
register_consequence("set_orientation", orientation_views)
register_consequence("set_event", event_views)

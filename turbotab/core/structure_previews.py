"""Previews for the structural answers that change what a row is (M2_CONTRACT §1–§2).

* ``set_unit`` — the participant flow with and without combining: ``600`` records → ``300`` rows.
* ``set_aggregation`` — that flow, and the reshape itself on two units: their records, then the one
  row each becomes (a ``table_focus`` whose story is *the records* → *combined*). The combined
  values come from the working stage's own SQL (``stages.working.combine_sql``) run on those units,
  so a preview and the table it foretells cannot disagree.

Aggregation is pre-seal (Decision A refuses it after), so nothing here reads held-out rows.
"""
from __future__ import annotations

from typing import Any

from turbotab.core.consequences import (
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
_METHOD_WORDS = {
    "mean": "are averaged",
    "first": "keep the first record",
    "last": "keep the last record",
    "change": "become last minus first",
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

    from turbotab.core.datastore import json_safe
    from turbotab.core.decisions import AggregationSpec
    from turbotab.core.stages.working import (
        _K,
        _UNIT,
        NUMERIC,
        ROW_ID,
        _bundle_table,
        _ident,
        _lit,
        aggregation_plan,
        combine_sql,
        rank_units,
        repair_expressions,
        source_sql,
    )

    oriented = ctx.artifact("oriented")
    info = _data(oriented)
    names = [str(c["name"]) for c in info["columns"]]
    dtypes = {str(c["name"]): str(c["dtype"]) for c in info["columns"]}
    physical = {str(c["name"]): str(c["physical_type"]) for c in info["columns"]}
    state = ctx.state.model_copy(update={"aggregation": AggregationSpec(
        method=decision.method, outcome=decision.outcome)})
    plan = aggregation_plan(state, set(names), facts["structure"])
    if plan is None:
        return None
    key = plan["id_column"]
    source = _bundle_table(oriented)
    repairs = repair_expressions(_data(ctx.artifact("findings")), state.findings)
    q = _ident(key)
    con = duckdb.connect()
    try:
        picked = [r[0] for r in con.execute(
            f"SELECT CAST({q} AS VARCHAR) AS u FROM read_parquet({_lit(source)}) "
            f"WHERE {q} IS NOT NULL GROUP BY u HAVING count(*) > 1 ORDER BY min({ROW_ID}) "
            f"LIMIT {SHOWN_UNITS}").fetchall()]
        if not picked:
            return None
        chosen = ", ".join(_lit(u) for u in picked)
        subset = (f"(SELECT * FROM {source_sql(source, names, repairs)} "
                  f"WHERE CAST({q} AS VARCHAR) IN ({chosen}))")
        rank_units(con, subset, plan, physical)
        outcome = facts["structure"].get("outcome") or {}
        target = state.target if state.target in dtypes else None
        varies = bool(outcome.get("varies")) and outcome.get("column") == target
        select, numeric, _ = combine_sql(info, plan, target, varies)
        combined = con.execute(select).df()
        records = con.execute(f"SELECT * FROM ranked ORDER BY {_UNIT}, {_K}").df()
    finally:
        con.close()

    shown = [c for c in dict.fromkeys([key, plan["time_column"], target]) if c]
    shown += [c for c in numeric if c not in shown and dtypes.get(c) in NUMERIC][:SHOWN_NUMBERS]
    unit_of = records.groupby(_UNIT).ngroup().to_numpy()
    rows: list[TableRow] = []
    changed: list[tuple[int, str]] = []
    record_frames: list[FrameRow] = []
    for i, record in records.iterrows():
        rid = int(record[ROW_ID])
        before = {c: json_safe(record[c]) for c in shown}
        after = {c: json_safe(combined.iloc[int(unit_of[i])][c]) for c in shown}
        rows.append(TableRow(row_id=rid, before=before, after=after))
        record_frames.append(FrameRow(row_id=rid, values=before))
        changed += [(rid, c) for c in shown if before[c] != after[c]]
    first_rows = records.groupby(_UNIT)[ROW_ID].min().to_numpy()
    combined_frames = [FrameRow(row_id=int(first_rows[j]), values={c: json_safe(combined.iloc[j][c])
                                                                   for c in shown})
                       for j in range(len(combined))]
    words = _METHOD_WORDS[decision.method]
    kept = "last" if decision.method == "last" else "first"
    return TableFocusView(
        title=f"{len(picked)} units' records, combined",
        caption=f"Numeric columns {words}; other columns keep the {kept} record.",
        emphasis=shown[1:], columns_before=shown, columns_after=shown, rows=rows, changed=changed,
        n_affected_columns=len(numeric),
        story=[TableFrame(label=f"Each {key}'s records", columns=shown, rows=record_frames),
               TableFrame(label="Combined into one row each", columns=shown, rows=combined_frames)])


def aggregation_views(decision: Any, ctx: PreviewContext) -> list[Any]:
    flow = _flow(ctx, True)
    if flow is None:
        return []
    views: list[Any] = list(flow[0])
    reshape = _reshape(decision, ctx, flow[1])
    if reshape is not None:
        views.insert(0, reshape)
    return views


register_consequence("set_unit", unit_views)
register_consequence("set_aggregation", aggregation_views)

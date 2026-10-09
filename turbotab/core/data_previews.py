"""Consequence previews for what the data are (V2 definition of done §1: data in; BLUEPRINT §14, the
readings ledger) and for the answers that act on a drawn result.

| Kind                  | Views (primary first)                                                    |
|-----------------------|--------------------------------------------------------------------------|
| ``set_column_unit``   | distribution of the column, the pack's intake screen marked in the unit  |
|                       | answered (energy), or the values in years, story: as recorded (an age)   |
| ``join_files``        | row flow: the table's rows → those with a partner → the joined rows      |
| ``import_codebook``   | table focus of what it settles: each settled column's cells, as read now |
|                       | and as the codebook documents them                                       |
| ``confirm_reading``,  | the same table focus, for the readings confirmed (a role: the lineage of |
| ``confirm_readings``  | which columns the models read)                                           |
| ``confirm_role``      | lineage: which columns the models read once the role is settled          |
| ``set_feature_table`` | the turned table's corner (the orientation's own preview, under this     |
|                       | label and these annotations)                                             |
| ``set_substitution``  | table focus, the swap on real rows: each person's two intakes as         |
|                       | recorded and with one step moved                                         |
| ``set_explain``       | relationship: where the explanation evaluates the model, against the     |
|                       | rows that exist (partial dependence's grid, or accumulated local         |
|                       | effects' local moves)                                                    |
| ``revert``            | the preview of the answer the revert restores                            |

Importing this module registers the builders.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Sequence

import numpy as np
import pandas as pd

from turbotab.core.consequences import (
    MAX_VIEWS, DistributionFrame, DistributionView, FrameRow, LineageView, Mark, PreviewContext,
    RelationshipView, RowFlowFrame, RowFlowView, RowStep, TableFocusView, TableFrame, TableRow,
    after_state, fmt_count, fmt_value, register_consequence,
)
from turbotab.core.plan_previews import (
    caption, frame_label, histogram, names, num, points, pool, population_block, shared_edges,
    task_of, tick, title,
)

MAX_ROWS = 8
MAX_COLUMNS = 12


def _json(value: Any) -> Any:
    from turbotab.core.datastore import json_safe

    return json_safe(value)


# ── set_column_unit ──────────────────────────────────────────────────────────

AGE_FACTORS = {"years": 1.0, "months": 12.0, "weeks": 365.25 / 7.0, "days": 365.25}


def energy_screen(frame: pd.DataFrame, energy: str, target: str | None, unit: str,
                  days: int) -> dict[str, Any] | None:
    """The pack's sex-neutral 500–5,000 kcal-a-day intake screen read in the answered unit, as the
    proposals stage reads it (``stages.proposals.exclusion_proposals``: the same bounds and the same
    rows, every row with the outcome measured), with the rows it would remove."""
    from turbotab.core.decisions import ColumnUnitSpec
    from turbotab.core.stages.proposals import energy_unit_reading, exclusion_proposals

    base = (frame[target].notna() if target and target in frame.columns
            else pd.Series(True, index=frame.index))
    reading = energy_unit_reading(frame, energy, ColumnUnitSpec(unit=unit, days=days))
    offers = exclusion_proposals(frame, energy=energy, unit=reading["unit"], sex=None,
                                 sex_levels={}, base=base, days=int(reading.get("days") or 1),
                                 unit_reading=reading)
    found = next((o for o in offers if o["key"] == "sex_neutral_500_5000"), None)
    if found is None:
        return None
    rule = found["rule"]
    return {"low": float(rule["low"]), "high": float(rule["high"]),
            "affected": int(found["affected"]), "n": int(base.sum()), "unit": reading["unit"],
            "days": int(reading.get("days") or 1)}


def column_unit_views(decision: Any, ctx: PreviewContext) -> list[Any]:
    """Total energy: the column read in the answered unit, step by step to kcal a day, with the
    pack's 500–5,000 kcal-a-day screen marked and the rows it would remove counted as the proposals
    count them. An age: the column read in years."""
    column = decision.column
    if column not in ctx.datastore.columns:
        return []
    target = getattr(ctx.state, "target", None)
    ids = ctx.sample_row_ids(ctx.unsealed_row_ids())
    wanted = [column, *([target] if target and target in ctx.datastore.columns and target != column
                        else [])]
    frame = ctx.datastore.materialize(wanted, ids)
    x = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(x).any():
        return []
    if decision.unit in ("kcal", "kj"):
        found = energy_screen(frame, column, target, decision.unit, decision.days)
        if found is None:
            return []
        word = "kJ" if found["unit"] == "kj" else "kcal"
        days = max(found["days"], 1)
        over = f" over {days} days" if days > 1 else " a day"
        ctx.read["column_unit"] = found
        kcal = x / (4.184 if found["unit"] == "kj" else 1.0)
        per_day = kcal / days
        edges = shared_edges(x, per_day)
        story = []
        if found["unit"] == "kj" and days > 1:
            story = [DistributionFrame(label=frame_label(f"Divided by 4.184: kcal over {days} days"),
                                       hist=histogram(kcal, edges=edges))]
        converted = found["unit"] == "kj" or days > 1
        steps = " ÷ 4.184" if found["unit"] == "kj" else ""
        steps += f" ÷ {days}" if days > 1 else ""
        return [DistributionView(
            title=title(f"{tick(column)} read in {word}{over}"),
            caption=caption(f"The 500–5,000 kcal-a-day screen is {num(found['low'])}–"
                            f"{num(found['high'])} {word}{over}: {fmt_count(found['affected'])} of "
                            f"{fmt_count(found['n'])} rows fall outside."),
            emphasis=[column],
            column=column,
            before=histogram(x, edges=edges),
            after=histogram(per_day if converted else x, edges=edges),
            before_label=f"{column} as recorded, {word}{over}",
            after_label=(f"{column}{steps}: kcal a day" if converted
                         else f"{column} as recorded, kcal a day"),
            marks=[Mark(value=500.0, label="500 kcal a day"),
                   Mark(value=5000.0, label="5,000 kcal a day")],
            story=story,
        )]
    factor = AGE_FACTORS.get(decision.unit)
    if factor is None:
        return []
    years = x / factor
    median = float(np.nanmedian(x))
    ctx.read["column_unit"] = {"median": median, "median_years": median / factor}
    edges = shared_edges(x, years)
    return [DistributionView(
        title=title(f"{tick(column)} read in {decision.unit}"),
        caption=caption(f"Median {num(median)} {decision.unit}: {num(median / factor)} years; "
                        f"every age-banded check reads it so."),
        emphasis=[column],
        column=column,
        before=histogram(x, edges=edges),
        after=histogram(years, edges=edges),
        before_label=f"{column} as recorded, in {decision.unit}",
        after_label=f"{column} in years",
    )]


# ── join_files ───────────────────────────────────────────────────────────────


def join_views(decision: Any, ctx: PreviewContext) -> list[Any]:
    """The join's own counts (``assembly.plan``), which the server fills in before any preview
    (``join_files``'s completion): what is recorded is what was previewed."""
    if decision.counts is not None:
        counts = decision.counts.model_dump()
    else:
        from turbotab.core.assembly import plan_for

        project_dir = ctx.settings.get("project_dir")
        if not project_dir:
            return []
        try:
            counts = plan_for(decision, {"project_dir": str(project_dir),
                                         "state": ctx.state}).counts()
        except Exception:  # noqa: BLE001 - the validator says why a join cannot run
            return []
    table, rows = int(counts["table_rows"]), int(counts["rows"])
    unmatched = int(counts["table_unmatched"])
    matched = table - unmatched
    file_rows, file_unmatched = int(counts["file_rows"]), int(counts["file_unmatched"])
    name = decision.name or decision.file
    start = RowStep(key="table", label="Rows in the table", n=table)
    partner = RowStep(key="matched",
                      label=f"With a partner in “{name}” on `{decision.on}`",
                      n=matched if decision.how == "inner" else table,
                      dropped=unmatched if decision.how == "inner" else 0,
                      reason=("no row of the file shares its identifier" if decision.how == "inner"
                              else None))
    steps = [start, partner]
    relation = str(counts["relation"])
    if rows != partner.n:
        steps.append(RowStep(key="joined", label=f"After the join ({relation})", n=rows,
                             reason=None))
    blanks = (f"; {fmt_count(unmatched)} keep blanks in its columns"
              if decision.how == "left" and unmatched else "")
    text = (f"{fmt_count(matched)} of {fmt_count(table)} rows match{blanks}; "
            f"{fmt_count(file_unmatched)} of the file's {fmt_count(file_rows)} find no row.")
    ctx.read["join"] = {"rows": rows, "matched": matched, "table": table}
    ctx.read["basis"] = (f"Counted on every row of the table and of “{name}”, matched on "
                         f"`{decision.on}`.")
    return [RowFlowView(
        title=title(f"Joining “{name}” on {tick(decision.on)}"),
        caption=caption(text),
        emphasis=["matched"],
        before=[start],
        after=steps,
        story=[RowFlowFrame(label=frame_label("The rows matched on the identifier"),
                            steps=[start, RowStep(key="matched", label="With a partner",
                                                  n=matched, dropped=table - matched,
                                                  reason="no partner in the file")])],
    )]


# ── what a codebook or a confirmation settles ────────────────────────────────

READING_WORDS = {
    "unit": "unit", "day_count": "days per value", "code_or_count": "codes or amounts",
    "role": "role", "cluster": "groups the rows", "time_column": "orders each unit's rows",
    "nested_in": "part of", "sex_coding": "sex coding",
}


def read_as(items: Sequence[Any], cell: Any) -> Any:
    """A cell as the settled readings of its column read it, in words: ``2 = female`` (a sex
    coding), ``2,206 kcal`` (a unit), ``2,206 kcal over 2 days``, ``1 (a code)``."""
    if cell is None or (isinstance(cell, float) and not np.isfinite(cell)):
        return cell
    found = {i.reading: str(i.value) for i in items}
    shown = num(cell) if isinstance(cell, (int, float)) and not isinstance(cell, bool) else cell
    if "sex_coding" in found:
        pairs = dict(p.split("=", 1)[::-1] for p in found["sex_coding"].split(",") if "=" in p)
        return f"{_plain(cell)} = {pairs.get(str(_plain(cell)), '?')}"
    if "unit" in found:
        word = {"kj": "kJ", "pct_energy": "% of energy"}.get(found["unit"], found["unit"])
        text = f"{shown} {word}"
        if found.get("day_count") not in (None, "1"):
            text += f" over {found['day_count']} days"
        return text
    if found.get("day_count") not in (None, "1"):
        return f"{shown} over {found['day_count']} days"
    if found.get("code_or_count") == "code":
        return f"{_plain(cell)} (a code)"
    return cell


def _plain(cell: Any) -> Any:
    if isinstance(cell, float) and cell.is_integer():
        return int(cell)
    return cell


def settles_views(items: Sequence[Any], ctx: PreviewContext, *, source: str, step: str,
                  asked: int = 0, kept: int = 0) -> list[Any]:
    """The table narrowed to the columns ``items`` settle: each cell as read now and as the
    settled reading reads it, the story's step the readings themselves (``step`` labels it). A
    role confirmation is the lineage's (:func:`roles_views`)."""
    from turbotab.core.readings import confirmation

    state = ctx.state
    by_column: dict[str, list[Any]] = {}
    for item in items:
        if item.reading == "role":
            continue
        by_column.setdefault(item.column, []).append(item)
    columns = [c for c in by_column if c in ctx.datastore.columns][:MAX_COLUMNS]
    changed_items = [i for i in items if confirmation(state, i.reading, i.column) != i.value]
    if not columns:
        return []
    ids = ctx.sample_row_ids(ctx.unsealed_row_ids(), n=MAX_ROWS)
    frame = ctx.datastore.materialize(columns, ids)
    rows, changed = [], []
    for rid, row in frame.iterrows():
        before = {c: _json(row[c]) for c in columns}
        after = {}
        for c in columns:
            after[c] = read_as(by_column[c], before[c])
            if after[c] != before[c]:
                changed.append((int(rid), c))
        rows.append(TableRow(row_id=int(rid), before=before, after=after))
    labels = [", ".join(f"{READING_WORDS.get(i.reading, i.reading)}: {i.value}"
                        for i in by_column[c]) for c in columns]
    n_columns = len({i.column for i in items})
    text = (f"{source} settles {fmt_count(len(items))} "
            f"{'reading' if len(items) == 1 else 'readings'} of "
            f"{fmt_count(n_columns)} {'column' if n_columns == 1 else 'columns'}"
            + (f"; {fmt_count(asked)} asked, values disagree" if asked else "")
            + (f"; {fmt_count(kept)} answers stand" if kept else "") + ".")
    ctx.read["settles"] = {"items": [(i.reading, i.column, i.value) for i in items],
                           "changed": [(i.reading, i.column) for i in changed_items]}
    return [TableFocusView(
        title=title("What it settles, cell by cell"),
        caption=caption(text),
        emphasis=columns,
        columns_before=columns,
        columns_after=columns,
        rows=rows,
        changed=changed,
        n_affected_columns=len(by_column),
        story=[TableFrame(label=frame_label(step), columns=["column", "reading"],
                          rows=[FrameRow(row_id=i, values={"column": c, "reading": label})
                                for i, (c, label) in enumerate(zip(columns, labels))])],
    )]


def codebook_views(decision: Any, ctx: PreviewContext) -> list[Any]:
    name = decision.name or decision.codebook
    views = settles_views(decision.items, ctx, source=f"“{name}”",
                          step="What the codebook documents", asked=len(decision.asked),
                          kept=len(decision.kept))
    roles = [i for i in decision.items if i.reading == "role"]
    if roles:
        views += roles_views(roles, ctx)
    if not views and decision.items:
        ctx.read["note"] = (f"“{name}” settles {len(decision.items)} readings of columns this "
                            f"table no longer holds.")
    elif not decision.items:
        ctx.read["note"] = (f"“{name}” settles no reading: its labels only strengthen the guesses "
                            f"each question leads with.")
    return views[:MAX_VIEWS]


def confirm_reading_views(decision: Any, ctx: PreviewContext) -> list[Any]:
    items = list(decision.items) if decision.kind == "confirm_readings" else [decision]
    views = settles_views(items, ctx, source="The confirmation", step="The readings confirmed")
    roles = [i for i in items if i.reading == "role"]
    if roles:
        views += roles_views(roles, ctx)
    return views[:MAX_VIEWS]


def roles_views(items: Sequence[Any], ctx: PreviewContext) -> list[Any]:
    """Which columns the models read once these roles are settled (a role that rode along
    unconfirmed puts no column in a model: BLUEPRINT §14.1), from names alone."""
    from turbotab.core import decisions
    from turbotab.core.readings import settled_roles
    from turbotab.core.row_previews import _order, _roles_lineage
    from turbotab.core.stages.rows import predictors

    state = ctx.state
    after = state
    for item in items:
        after = decisions.fold_onto(after, decisions.ConfirmRole(column=item.column,
                                                                 role=str(item.value)))
    old, new = dict(settled_roles(state)), dict(settled_roles(after))
    order = _order(ctx)
    touched = {i.column for i in items}
    now_in, then_in = predictors(old, order), predictors(new, order)
    arrive = [c for c in then_in if c not in now_in]
    leave = [c for c in now_in if c not in then_in]
    if arrive or leave:
        bits = []
        if arrive:
            bits.append(f"{names(arrive, 2)} {'enters' if len(arrive) == 1 else 'enter'} the models")
        if leave:
            bits.append(f"{names(leave, 2)} {'leaves' if len(leave) == 1 else 'leave'} them")
        text = "; ".join(bits) + f"; {fmt_count(len(then_in))} predictors in all."
    else:
        text = (f"{names(sorted(touched), 2)} settled as recorded; the same "
                f"{fmt_count(len(then_in))} predictors.")
    return [LineageView(
        title=title("Which columns the models read"),
        caption=caption(text),
        emphasis=[*arrive, *leave, *touched][:12],
        before=_roles_lineage(order, old, touched) if old else None,
        after=_roles_lineage(order, new, touched),
    )]


def confirm_role_views(decision: Any, ctx: PreviewContext) -> list[Any]:
    item = type("Item", (), {"reading": "role", "column": decision.column,
                             "value": decision.role})()
    return roles_views([item], ctx)


# ── set_feature_table ────────────────────────────────────────────────────────


def feature_table_views(decision: Any, ctx: PreviewContext) -> list[Any]:
    """The orientation's own preview (``structure_previews.orientation_views``: the stage's turn
    plan and corner), under the label and annotations this answer names."""
    from turbotab.core.decisions import SetOrientation
    from turbotab.core.structure_previews import orientation_views

    shadow = dataclasses.replace(ctx, state=after_state(decision, ctx), read={})
    views = orientation_views(SetOrientation(orientation="feature_major"), shadow)
    ctx.read.update(shadow.read)
    return views


# ── set_substitution ─────────────────────────────────────────────────────────


def substitution_views(decision: Any, ctx: PreviewContext) -> list[Any]:
    """The swap on real rows: each person's donor and recipient as recorded, and with one step
    moved (``step_kcal`` of energy, each source converted by the kcal per unit the readings ledger
    settled, as the substitution stage converts it)."""
    from turbotab.core.readings import kcal_per_unit

    donor, recipient = decision.donor, decision.recipient
    if donor not in ctx.datastore.columns or recipient not in ctx.datastore.columns:
        return []
    ids = pool(ctx)
    if ids is None:
        return []
    try:
        fd = kcal_per_unit(ctx.state, donor, store=ctx.datastore)
        fr = kcal_per_unit(ctx.state, recipient, store=ctx.datastore)
    except Exception:  # noqa: BLE001 - an unsettled unit is the validator's to ask
        return []
    fd = getattr(fd, "factor", fd)
    fr = getattr(fr, "factor", fr)
    if not fd or not fr:
        return []
    # Under the surveyed population the substitution stage draws each family's curve over the
    # design, or blocks and records it (MODELING_SEQUENCE §4): every curve where the design itself
    # is refused, a family's where it has no design-based estimator. The swap says the same.
    curves = "substitution curves" if task_of(ctx, ctx.state) == "multiclass" else \
        "substitution curve"
    population_block(ctx, after_state(decision, ctx), decision, what=curves, marginal=False)
    frame = ctx.datastore.materialize([donor, recipient], ids[:MAX_ROWS])
    if decision.scale == "percent_energy":
        ctx.read["note"] = (f"Each step moves {decision.step_percent:g}% of each person's energy "
                            f"from {tick(donor)} to {tick(recipient)}.")
        return []
    k = float(decision.step_kcal)
    moved_d, moved_r = k / float(fd), k / float(fr)
    rows, changed = [], []
    for rid, row in frame.iterrows():
        before = {donor: _json(row[donor]), recipient: _json(row[recipient])}
        after = {donor: _json(float(row[donor]) - moved_d) if pd.notna(row[donor]) else None,
                 recipient: (_json(float(row[recipient]) + moved_r) if pd.notna(row[recipient])
                             else None)}
        rows.append(TableRow(row_id=int(rid), before=before, after=after))
        changed += [(int(rid), donor), (int(rid), recipient)]
    return [TableFocusView(
        title=title(f"{num(k)} kcal from {tick(donor)} to {tick(recipient)}"),
        caption=caption(f"One step: {num(moved_d)} of {tick(donor)} out, {num(moved_r)} of "
                        f"{tick(recipient)} in, total energy unchanged."),
        emphasis=[donor, recipient],
        columns_before=[donor, recipient],
        columns_after=[donor, recipient],
        rows=rows,
        changed=changed,
        n_affected_columns=2,
    )]


# ── set_explain ──────────────────────────────────────────────────────────────

GRID = 10


def explain_views(decision: Any, ctx: PreviewContext) -> list[Any]:
    """Where the explanation evaluates the model, against the rows that exist: partial dependence
    sets every row's exposure to each grid value (combinations no row may have, when the exposure
    tracks another column); accumulated local effects move each row only within its own bin."""
    from turbotab.core.models.pipeline import model_predictors

    ids = pool(ctx)
    if ids is None:
        return []
    predictors = model_predictors(ctx.state)
    exposures = list(decision.exposures) or [c for c, r in (ctx.state.roles or {}).items()
                                             if r == "exposure" and c in predictors]
    exposure = next((c for c in exposures if c in ctx.datastore.columns), None)
    if exposure is None:
        return []
    numeric = [c for c in predictors if c != exposure and c in ctx.datastore.columns]
    frame = ctx.datastore.materialize([exposure, *numeric], ids)
    x = pd.to_numeric(frame[exposure], errors="coerce").to_numpy(dtype=float)
    best, partner = 0.0, None
    for c in numeric:
        y = pd.to_numeric(frame[c], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() > 3 and np.std(x[ok]) > 0 and np.std(y[ok]) > 0:
            r = float(np.corrcoef(x[ok], y[ok])[0, 1])
            if abs(r) > abs(best):
                best, partner = r, c
    if partner is None:
        return []
    y = pd.to_numeric(frame[partner], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    edges = np.unique(np.quantile(x, np.linspace(0, 1, GRID + 1)))
    if decision.curves == "partial_dependence":
        grid = edges
        gx = np.repeat(grid, len(x))
        gy = np.tile(y, len(grid))
        label = "partial dependence's grid"
        text = (f"{tick(exposure)} and {tick(partner)} correlate {fmt_value(round(best, 2))}: "
                f"partial dependence also reads combinations no row has.")
    else:
        bins = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
        gx = np.concatenate([edges[bins], edges[bins + 1]])
        gy = np.concatenate([y, y])
        label = "accumulated local effects' moves"
        text = (f"{tick(exposure)} and {tick(partner)} correlate {fmt_value(round(best, 2))}: "
                f"each row moves only across its own bin.")
    return [RelationshipView(
        title=title(f"Where {tick(exposure)}'s curve reads the model"),
        caption=caption(text),
        emphasis=[exposure, partner],
        x_label=exposure,
        y_label_before=partner,
        y_label_after=f"{partner}, at the points the curve reads ({label})",
        points_before=points(x, y),
        points_after=points(gx, gy),
        r_before=best,
        r_after=None,
    )]


# ── revert ───────────────────────────────────────────────────────────────────


def revert_views(decision: Any, ctx: PreviewContext) -> list[Any]:
    """The answer the revert restores, previewed as that answer: the latest earlier record of the
    same question that still stands (``decisions.fold`` keeps the latest write of a slot). With no
    earlier answer, the question opens again, which the note says."""
    from turbotab.core import consequences, decisions

    records = ctx.settings.get("records")
    if records is None:
        return []
    records = list(records() if callable(records) else records)
    by_id = {r.id: r for r in records}
    target = by_id.get(decision.decision_id)
    if target is None or target.decision.kind not in decisions.SLOTS:
        return []
    undone = target.decision
    cancelled = decisions.reverted(records)

    def same(d: Any) -> bool:
        if d.kind not in decisions.SLOTS or decisions.SLOTS[d.kind] != decisions.SLOTS[undone.kind]:
            return False
        key = decisions._KEYS.get(undone.kind)
        other = decisions._KEYS.get(d.kind)
        return key is None or (other is not None and other(d) == key(undone))

    earlier = [r for r in records
               if r.seq < target.seq and r.id not in cancelled and same(r.decision)]
    if not earlier:
        ctx.read["note"] = "Undoing it leaves the question unanswered, so it is asked again."
        return []
    restored = earlier[-1].decision
    # The restored answer written onto the state as it stands (the revert's own effect), the
    # preview's reads kept in this context's record of them.
    shadow = dataclasses.replace(ctx, settings={k: v for k, v in ctx.settings.items()
                                                if k != "records"}, read=ctx.read)
    views: list[Any] = []
    for _, builder in consequences._BUILDERS.get(restored.kind, []):
        views.extend(builder(restored, shadow))
    ctx.caution = shadow.caution
    return views[:MAX_VIEWS]


register_consequence("set_column_unit", column_unit_views)
register_consequence("join_files", join_views)
register_consequence("import_codebook", codebook_views)
register_consequence("confirm_reading", confirm_reading_views)
register_consequence("confirm_readings", confirm_reading_views)
register_consequence("confirm_role", confirm_role_views)
register_consequence("set_feature_table", feature_table_views)
register_consequence("set_substitution", substitution_views)
register_consequence("set_explain", explain_views)
register_consequence("revert", revert_views)

__all__ = ["energy_screen", "settles_views"]

"""Consequence previews for the row-side decisions (M1_CONTRACT.md §4; owner: the rows agent).

| Kind             | Views (primary first)                                          |
|------------------|----------------------------------------------------------------|
| ``set_exclusions`` | row_flow · distribution of the excluded column, cuts marked  |
| ``set_missing``    | row_flow · table_focus of the rows and cells affected (with  |
|                    | columns left out: the rows that leaving them out saves)      |
| ``set_split``      | row_flow: the train / held-out fork                          |
| ``set_roles``      | lineage: which columns enter the model                        |

Counts are exact over the pool: every row before the split exists, and every row but the
held-out ones once it does. Rows an exclusion removed are still in the pool, so a preview can
show them coming back; the held-out rows are never read. The one exception is the split preview
itself, which draws a split as the split stage would — it reads the outcome's classes and the
identifier, which is what defining the held-out rows means, and nothing that informs a model.

Importing this module registers the builders.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from turbotab.core.consequences import (
    CAPTION_WORDS, TITLE_WORDS, Caution, CautionExit, DistributionView, LineageView, Mark,
    PreviewContext, RowFlowView, RowStep, TableFocusView, TableRow, _histogram_pair, clip_words,
    fmt_count, fmt_value, lineage_of, register_consequence,
)
from turbotab.core.decisions import ROW_ID, MissingSpec, ProjectState, missing_strategy
from turbotab.core.stages.rows import (
    PREDICTOR_ROLES, _missing_mask, cohort_flow, cohort_inputs, draw_split, predictors, repair_rules,
    rule_keep,
)

MAX_FOCUS_COLUMNS = 12
MAX_FOCUS_ROWS = 8


# ── shared ───────────────────────────────────────────────────────────────────


def _ingest(ctx: PreviewContext) -> dict[str, Any]:
    return {"columns": [c.to_dict() for c in ctx.datastore.info().columns]}


def _order(ctx: PreviewContext) -> list[str]:
    return [c for c in ctx.datastore.columns if c != ROW_ID]


def sealed_rows(split: Any) -> Any:
    """Every held-out row of a split Bundle, including those the cohort now excludes."""
    if "sealed" in split.frames:
        return split.frames["sealed"]["row_id"].to_numpy(dtype=np.int64)
    frame = split.frames["assignment"]
    return frame.loc[frame["partition"] == "holdout", "row_id"].to_numpy(dtype=np.int64)


def _pool(ctx: PreviewContext) -> Any | None:
    """Every row but the held-out ones once a split exists; None (every row) before.

    The held-out rows are the newest split's, even while it recomputes: an answer that changes
    the cohort never moves a row across the seal, so the stale split's sealed rows still hold.
    """
    if ctx.sealed_row_ids is None:
        ctx.read["counted"] = int(ctx.datastore.n_rows)
        return None
    pool = ctx.unsealed_row_ids()
    ctx.read["counted"] = int(len(pool))
    return pool


def _sealed(ctx: PreviewContext) -> bool:
    return ctx.sealed_row_ids is not None


def _steps(steps: Sequence[dict[str, Any]], sealed: bool) -> list[RowStep]:
    out = [RowStep(**s) for s in steps]
    if sealed and out:
        out[0] = out[0].model_copy(update={"label": "Rows not held out"})
    return out


def _flows(ctx: PreviewContext, states: Sequence[ProjectState]) -> tuple[Any, Any, list[tuple[list, Any]]]:
    """Materialize once what every state's flow reads, on the pool; run each flow."""
    ingest = _ingest(ctx)
    needed: list[str] = []
    gappy: list[str] = []
    for st in states:
        n, _, g = cohort_inputs(st, ingest)
        needed += n
        gappy += g
    needed, gappy = list(dict.fromkeys(needed)), list(dict.fromkeys(gappy))
    frame = ctx.datastore.materialize(needed, _pool(ctx))
    mask = _missing_mask(ctx.datastore, gappy, frame.index) if gappy else None
    results = []
    for st in states:
        _, _, g = cohort_inputs(st, ingest)
        steps, kept = cohort_flow(frame, target=st.target, rules=st.exclusions,
                                  missing=missing_strategy(st), predictor_columns=g,
                                  missing_frame=mask, repairs=repair_rules(st))
        results.append((steps, kept))
    return frame, mask, results


def _pool_words(ctx: PreviewContext) -> str:
    return "rows"


# ── set_exclusions ───────────────────────────────────────────────────────────


def exclusions_views(decision: Any, ctx: PreviewContext) -> list[Any]:
    state = ctx.state
    proposed = state.model_copy(update={"exclusions": list(decision.rules)})
    frame, _, [(before, _), (after, _)] = _flows(ctx, [state, proposed])
    training = _sealed(ctx)
    entering = next((s["n"] for s in after if s["key"] == "outcome_measured"), after[0]["n"])
    excluded = sum(s["dropped"] for s in after if s["key"].startswith("exclusion:"))
    now = sum(s["dropped"] for s in before if s["key"].startswith("exclusion:"))
    if not decision.rules:
        caption = f"No rule excludes a row: all {fmt_count(entering)} {_pool_words(ctx)} pass"
        caption += f"; the current rules exclude {fmt_count(now)}." if now else "."
    else:
        caption = (f"{fmt_count(excluded)} of {fmt_count(entering)} {_pool_words(ctx)} fall outside "
                   f"these ranges and would leave; {fmt_count(entering - excluded)} stay.")
    views: list[Any] = [RowFlowView(
        title="Who these rules would exclude",
        caption=clip_words(caption, CAPTION_WORDS),
        emphasis=[s["key"] for s in after if s["key"].startswith("exclusion:")],
        before=_steps(before, training),
        after=_steps(after, training),
    )]
    rules = list(decision.rules) or list(state.exclusions or [])
    if rules:
        drops = [s["dropped"] for s in (after if decision.rules else before) if s["key"].startswith("exclusion:")]
        rule = rules[int(np.argmax(drops))] if drops else rules[0]
        views.append(_cut_view(frame, state.target, rule, removing=bool(decision.rules), ctx=ctx))
    return views


def _cut_view(frame: Any, target: str | None, rule: Any, *, removing: bool, ctx: PreviewContext) -> DistributionView:
    """The rule's column over the pool, and what the rule keeps of it, on shared bins."""
    import pandas as pd

    rows = frame if target is None else frame[frame[target].notna()]
    values = pd.to_numeric(rows[rule.column], errors="coerce").astype(float)
    kept = values[rule_keep(rows, rule)]
    marks = rule_marks(rule)
    cuts = sorted({m.value for m in marks})
    everything, inside = _histogram_pair(
        np.concatenate([values.to_numpy(), np.asarray(cuts, dtype=float)]), kept.to_numpy())
    # The cuts only widen the axis; they are not values.
    everything = everything.model_copy(update={"counts": _histogram_counts(values.to_numpy(), everything.edges),
                                               "n_missing": int(values.isna().sum())})
    out = int(len(values.dropna()) - len(kept.dropna()))
    col = f"`{rule.column}`"
    if rule.low is not None and rule.high is not None:
        where = f"below {fmt_value(rule.low)} or above {fmt_value(rule.high)}"
    elif rule.low is not None:
        where = f"below {fmt_value(rule.low)}"
    elif rule.high is not None:
        where = f"above {fmt_value(rule.high)}"
    else:
        where = f"outside its range for each `{rule.by.column}`"
    share = out / max(1, len(values.dropna()))
    caption = (f"{col} {where} is cut: {fmt_count(out)} {_pool_words(ctx)}, "
               f"`{round(100 * share, 1):g}%` of those measured.")
    return DistributionView(
        title=clip_words(f"Where the cut falls on {col}", TITLE_WORDS),
        caption=clip_words(caption, CAPTION_WORDS),
        emphasis=[rule.column],
        column=rule.column,
        before=everything,
        after=inside,
        before_label="Every measured value",
        after_label="Kept by the rule" if removing else "Kept by the current rule",
        marks=marks,
    )


def rule_marks(rule: Any) -> list[Mark]:
    """A rule's bounds as labeled marks ("500 kcal"); a by-level bound names its level."""
    from turbotab.core.stages.rows import _fmt_num, _unit

    unit = _unit(rule.column)
    suffix = f" {unit}" if unit else ""

    def mark(value: float, group: str | None) -> Mark:
        return Mark(value=float(value), label=f"{_fmt_num(value)}{suffix}", group=group)

    out = [mark(b, None) for b in (rule.low, rule.high) if b is not None]
    if rule.by is not None:
        for level, (lo, hi) in rule.by.ranges.items():
            out += [mark(b, str(level)) for b in (lo, hi) if b is not None]
    return out


def _histogram_counts(values: Any, edges: Sequence[float]) -> list[int]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(edges):
        return []
    counts, _ = np.histogram(finite, np.asarray(edges, dtype=float))
    return counts.astype(int).tolist()


# ── set_missing ──────────────────────────────────────────────────────────────


def missing_views(decision: Any, ctx: PreviewContext) -> list[Any]:
    state = ctx.state
    if not state.roles:
        return []
    drop = list(getattr(decision, "drop_columns", []) or [])
    proposed = state.model_copy(update={"missing": MissingSpec(**decision.model_dump(exclude={"kind"}))})
    # Which predictors can be missing at all (nothing left out), and plain complete cases.
    probe = state.model_copy(update={"missing": MissingSpec(strategy="complete_case")})
    frame, mask, [(before, _), (after, kept_after), (plain, kept_plain)] = _flows(
        ctx, [state, proposed, probe])
    training = _sealed(ctx)
    order = _order(ctx)
    if drop:
        return _left_out_views(ctx, decision, drop, before, after, plain, kept_after, kept_plain,
                               training, order)
    preds = predictors(state.roles, order)  # the option leaves nothing out
    # Rows that reach the missing-values step: past the outcome and the rules.
    reach_ids = _kept_before_missing(frame, state)
    gaps: dict[str, int] = {}
    if mask is not None:
        m = mask.loc[reach_ids]
        gaps = {c: int(m[c].isna().sum()) for c in m.columns if c in preds and m[c].isna().any()}
    # Blanks that become their own "Missing" level are values: the strategy does not act on them.
    levels = [c for c in _level_columns(ctx, proposed, preds) if c in gaps]
    levels = sorted(levels, key=lambda c: (-gaps[c], order.index(c) if c in order else 0))
    ranked = sorted((c for c in gaps if c not in levels),
                    key=lambda c: (-gaps[c], order.index(c) if c in order else 0))
    gaps = {c: gaps[c] for c in ranked}
    n_reach = len(reach_ids)
    views: list[Any] = []
    if decision.strategy == "complete_case":
        dropped = next((s["dropped"] for s in after if s["key"] == "complete_cases"), 0)
        caption = (f"{fmt_count(dropped)} of {fmt_count(n_reach)} {_pool_words(ctx)} miss a predictor and "
                   f"would leave" + (f"; `{ranked[0]}` is missing most." if ranked else "."))
        if not dropped:
            caption = f"No {_pool_words(ctx)} miss a predictor, so none would leave."
    else:
        cells = sum(gaps.values())
        caption = (f"No rows leave; {fmt_count(cells)} missing cells in {fmt_count(len(gaps))} predictors "
                   f"would be filled.") if cells else "No predictor has a missing value, so nothing is filled."
        if ranked:
            fills = _fill_values(ctx.datastore.materialize(ranked[:MAX_FOCUS_COLUMNS], reach_ids),
                                 ranked[:MAX_FOCUS_COLUMNS])
            if len(ranked) <= 2:
                said = ", ".join(f"`{c}` → `{_shown(fills.get(c))}`" for c in ranked)
                caption = f"No rows leave; blanks are filled: {said}."
            ctx.caution = _not_asked_caution(ctx, ranked, gaps, fills)
    if levels and not ranked:  # every blank left is a level: say so rather than "no blanks"
        caption = f"No rows leave: blanks in {_names(levels)} stay as a level of their own."
    views.append(RowFlowView(
        title="Rows kept when values are missing",
        caption=clip_words(caption, CAPTION_WORDS),
        emphasis=["complete_cases"],
        before=_steps(before, training),
        after=_steps(after, training),
    ))
    if ranked:
        views.append(_gaps_view(ctx, reach_ids, ranked, gaps, decision.strategy))
    if levels:
        views.insert(1, _levels_view(ctx, reach_ids, levels))
    return views[:3]


def _level_columns(ctx: PreviewContext, state: ProjectState, preds: Sequence[str]) -> list[str]:
    from turbotab.core.models.pipeline import level_columns

    info = {c.name: c.to_dict() for c in ctx.datastore.info().columns}
    return level_columns(state, preds, info)


def _levels_view(ctx: PreviewContext, reach_ids: Any, levels: list[str]) -> TableFocusView:
    """The rows whose blanks become a ``Missing`` level (M2_CONTRACT §4: missingness by mechanism)."""
    from turbotab.core.datastore import json_safe
    from turbotab.core.models.pipeline import MISSING_LEVEL

    columns = levels[:MAX_FOCUS_COLUMNS]
    values = ctx.datastore.materialize(columns, reach_ids)
    missing = values.isna()
    per_row = missing.sum(axis=1)
    top = sorted(per_row[per_row > 0].sort_values(ascending=False, kind="stable").index[:MAX_FOCUS_ROWS])
    rows, cells = [], []
    for r in top:
        before = {c: json_safe(values.at[r, c]) for c in columns}
        after = {c: (MISSING_LEVEL if missing.at[r, c] else before[c]) for c in columns}
        rows.append(TableRow(row_id=int(r), before=before, after=after))
        cells += [(int(r), c) for c in columns if bool(missing.at[r, c])]
    n_cells = int(missing.to_numpy().sum())
    caption = (f"{fmt_count(n_cells)} blanks in {_names(columns)} become a `{MISSING_LEVEL}` level the "
               f"models can use; no row leaves for them.")
    return TableFocusView(
        title="Blanks kept as their own level",
        caption=clip_words(caption, CAPTION_WORDS),
        emphasis=columns,
        columns_before=columns,
        columns_after=columns,
        rows=rows,
        changed=cells,
        n_affected_columns=len(levels),
    )


def _left_out_views(ctx: PreviewContext, decision: Any, drop: list[str], before: list, after: list,
                    plain: list, kept_after: Any, kept_plain: Any, training: bool,
                    order: list[str]) -> list[Any]:
    """Leaving columns out first: the rows that saves, against complete cases on every predictor."""
    names = _names(drop)
    if decision.strategy == "complete_case":
        saved = np.setdiff1d(kept_after, kept_plain, assume_unique=True)
        n_left = next((s["n"] for s in after if s["key"] == "complete_cases"), len(kept_after))
        if len(saved):
            caption = (f"Leaving out {names} keeps {fmt_count(len(saved))} rows complete cases would "
                       f"drop; {fmt_count(n_left)} remain.")
        else:
            caption = f"Leaving out {names} saves no rows: {fmt_count(n_left)} remain either way."
    else:
        saved = np.asarray([], dtype=np.int64)
        caption = f"{names} leave the predictors; the rest are imputed and no row leaves."
    views: list[Any] = [RowFlowView(
        title="Rows kept when values are missing",
        caption=clip_words(caption, CAPTION_WORDS),
        emphasis=["complete_cases"],
        before=_steps(before, training),
        after=_steps(after, training),
    )]
    if len(saved):
        views.append(_saved_view(ctx, saved, drop, order))
    return views


def _names(columns: Sequence[str]) -> str:
    shown = [f"`{c}`" for c in columns[:2]]
    if len(columns) > 2:
        return f"{', '.join(shown)} and {len(columns) - 2:,} more"
    return " and ".join(shown)


def _saved_view(ctx: PreviewContext, saved: Any, drop: list[str], order: list[str]) -> TableFocusView:
    """A few of the saved rows: blank in the columns left out, complete in every other predictor."""
    from turbotab.core.datastore import json_safe

    preds = predictors(ctx.state.roles, order)
    others = [c for c in preds if c not in drop]
    columns = (drop + others)[:MAX_FOCUS_COLUMNS]
    shown = np.sort(np.asarray(saved, dtype=np.int64))[:MAX_FOCUS_ROWS]
    values = ctx.datastore.materialize(columns, shown)
    rows, cells = [], []
    for r in values.index:
        before = {c: json_safe(values.at[r, c]) for c in columns}
        after = {c: before[c] for c in columns if c not in drop}
        rows.append(TableRow(row_id=int(r), before=before, after=after))
        cells += [(int(r), c) for c in drop if c in columns and before[c] is None]
    caption = (f"{fmt_count(len(saved))} rows are blank only in {_names(drop)}; without those "
               f"columns, each is complete.")
    return TableFocusView(
        title="The rows leaving them out saves",
        caption=clip_words(caption, CAPTION_WORDS),
        emphasis=[c for c in drop if c in columns],
        columns_before=columns,
        columns_after=[c for c in columns if c not in drop],
        rows=rows,
        changed=cells,
        n_affected_columns=len(drop),
    )


def _kept_before_missing(frame: Any, state: ProjectState) -> Any:
    _, kept = cohort_flow(frame, target=state.target, rules=state.exclusions, missing=None,
                          predictor_columns=[], repairs=repair_rules(state))
    return kept


def _fill_values(values: Any, columns: Sequence[str]) -> dict[str, Any]:
    """What imputation writes into each column's blanks, as the pipeline fits it: the median of a
    number, the most common value of anything else (``pipeline.py``'s SimpleImputers)."""
    fills: dict[str, Any] = {}
    for c in columns:
        s = values[c].dropna()
        if s.empty:
            fills[c] = None
        elif s.dtype.kind in "biuf" and s.dtype.kind != "b":
            fills[c] = float(s.median())
        else:
            fills[c] = s.mode().iloc[0]
    return fills


def _shown(value: Any) -> str:
    from turbotab.core.datastore import json_safe

    value = json_safe(value)
    if isinstance(value, float):
        return fmt_value(value)
    return str(value)


def _not_asked_caution(ctx: PreviewContext, ranked: list[str], gaps: dict[str, int],
                       fills: dict[str, Any]) -> Any:
    """Filling a column whose blanks mean "not asked" asserts an answer nobody gave
    (DRIVE_RUBRIC §4): say what it would write, and offer to leave the columns out instead."""
    from turbotab.core.consequences import Caution, CautionExit
    from turbotab.core.decisions import SetMissing

    proposals = ctx.artifact("proposals")
    data = getattr(proposals, "data", proposals)
    reading = (data.get("missing") or {}) if isinstance(data, dict) else {}
    likely = [str(e["column"]) for e in reading.get("columns") or [] if e.get("likely_not_asked")]
    likely = [c for c in likely if c in gaps]
    if not likely:
        return None
    named = likely[:3]
    parts = [f"`{c}` becomes `{_shown(fills.get(c))}` on {fmt_count(gaps[c])} rows" for c in named]
    joined = parts[0] if len(parts) == 1 else f"{', '.join(parts[:-1])} and {parts[-1]}"
    one = len(likely) == 1
    text = (f"Imputing asserts answers nobody gave: blank {joined}, its most common value, "
            f"though a blank there likely means the question was not asked.") if one else (
            f"Imputing asserts answers nobody gave: blank {joined}, the most common value, though "
            f"a blank there likely means the question was not asked.")
    leave = SetMissing(strategy="complete_case", drop_columns=likely)  # the question's own offer
    exits = [CautionExit(label=f"Leave {'it' if one else 'them'} out first",
                         decision=leave.model_dump(mode="json"))]
    probe = ctx.state.model_copy(update={"missing": MissingSpec(strategy="impute",
                                                                categorical="missing_category")})
    if set(likely) <= set(_level_columns(ctx, probe, likely)):
        level = SetMissing(strategy="impute", categorical="missing_category")
        exits.append(CautionExit(label="Keep blanks as their own level",
                                 decision=level.model_dump(mode="json")))
    return Caution(text=text, exits=exits)


def _gaps_view(ctx: PreviewContext, reach_ids: Any, ranked: list[str], gaps: dict[str, int],
               strategy: str) -> TableFocusView:
    """The rows with the most missing predictors, and what each strategy does to them."""
    from turbotab.core.datastore import json_safe

    columns = ranked[:MAX_FOCUS_COLUMNS]
    values = ctx.datastore.materialize(columns, reach_ids)
    missing = values.isna()
    per_row = missing.sum(axis=1)
    top = per_row[per_row > 0].sort_values(ascending=False, kind="stable").index[:MAX_FOCUS_ROWS]
    top = sorted(top)
    fills: dict[str, Any] = _fill_values(values, columns) if strategy == "impute" else {}
    rows = []
    cells = []
    for r in top:
        before = {c: json_safe(values.at[r, c]) for c in columns}
        if strategy == "impute":
            after = {c: (json_safe(fills[c]) if missing.at[r, c] else before[c]) for c in columns}
        else:
            after = {}  # the row leaves the analysis
        rows.append(TableRow(row_id=int(r), before=before, after=after))
        cells += [(int(r), c) for c in columns if bool(missing.at[r, c])]
    n_cells = sum(gaps.values())
    if strategy == "impute":
        shown = [c for c in columns[:2]]
        said = " and ".join(f"`{c}` with `{_shown(fills.get(c))}`" for c in shown)
        caption = (f"{fmt_count(n_cells)} cells would be filled: {said}"
                   f"{', and the rest likewise' if len(columns) > 2 else ''}.")
        title = "The cells that would be filled"
    else:
        caption = (f"Each row with a gap leaves whole; {fmt_count(n_cells)} missing cells across "
                   f"{fmt_count(len(gaps))} predictors.")
        title = "The rows that would leave"
    return TableFocusView(
        title=title,
        caption=clip_words(caption, CAPTION_WORDS),
        emphasis=columns,
        columns_before=columns,
        columns_after=columns if strategy == "impute" else [],
        rows=rows,
        changed=cells,
        n_affected_columns=len(gaps),
    )


# ── set_split ────────────────────────────────────────────────────────────────


def _fork(info: dict[str, Any]) -> list[RowStep]:
    if not info["n_holdout"]:
        return [RowStep(key="train", label="Training rows (cross-validation only)", n=info["n_train"])]
    return [
        RowStep(key="train", label="Training rows", n=info["n_train"], dropped=info["n_holdout"],
                reason="held out for the final check"),
        RowStep(key="holdout", label="Held-out rows", n=info["n_holdout"]),
    ]


def split_views(decision: Any, ctx: PreviewContext) -> list[Any]:
    state = ctx.state
    if state.target is None:
        return []
    cohort = ctx.artifact("cohort")
    if cohort is not None and "measured" in cohort.frames:
        steps = cohort.data["steps"]
        rows = cohort.frames["rows"]["row_id"].to_numpy(dtype=np.int64)
        universe = cohort.frames["measured"]["row_id"].to_numpy(dtype=np.int64)
    else:
        from turbotab.core.stages.rows import compute_cohort

        measured: list[Any] = []
        steps, rows, _ = compute_cohort(ctx.datastore, state, _ingest(ctx), measured=measured)
        universe = measured[0]
    target_info = ctx.artifact("target_info")
    task = state.task or (target_info.get("task") if isinstance(target_info, dict) else None)
    # The seal's basis, its chronological draw and what a holdout of this size can measure
    # (turbotab/core/seal.py, M2_CONTRACT §3).
    from turbotab.core import seal

    draw = seal.seal_inputs(state, universe, ctx.datastore, task, holdout=decision.holdout,
                            seed=decision.seed)
    if draw.refusal:
        ctx.read["note"] = draw.refusal
        return []
    assignment, info = draw_split(rows, holdout=decision.holdout, seed=decision.seed,
                                  folds=decision.folds, universe=universe, **draw.split_args())
    before = [RowStep(**s) for s in steps]
    current = ctx.artifact("split")
    if current is not None:
        before += _fork(current.data)
    after = [RowStep(**s) for s in steps] + _fork(info)
    held_counts = None
    if draw.labels is not None:  # the classes among the rows a held-out score would be made on
        held = assignment.loc[assignment["partition"] == "holdout", "row_id"].to_numpy()
        in_held = np.isin(np.asarray(universe, dtype=np.int64), held)
        held_counts = seal.class_counts(np.asarray(draw.labels, dtype=object)[in_held])
    measures, below = seal.measure(task, int(info["n_holdout"]), held_counts)
    if info["n_holdout"]:
        how = (f"the latest by `{draw.chronology.time_column}`" if draw.chronology and draw.chronology.drawn
               else f"grouped by `{info['grouped_by']}`" if info["grouped_by"] else "by row")
        caption = f"{fmt_count(info['n_holdout'])} rows held out, {how}. {measures}"
    else:
        caption = (f"All {fmt_count(info['n_train'])} rows train, scored by "
                   f"{fmt_count(info['folds'])}-fold cross-validation.")
    if draw.exploratory and decision.holdout > 0:
        basis = draw.basis
        exits = []
        if basis.state == "abandoned" and basis.source == "grain" and basis.column:
            # Each row was said to be a unit while an identifier repeats: the lever is the grain.
            exits.append(CautionExit(label=f"Keep each `{basis.column}`'s rows together",
                                     decision={"kind": "set_grain", "grain": "repeated",
                                               "id_column": basis.column}))
        ctx.caution = Caution(text=basis.sentence if basis.exploratory else draw.chronology.sentence,
                              exits=exits)
    elif below and decision.holdout > 0:
        floor = seal.floor_for(task)
        ctx.caution = Caution(
            text=f"{measures} {floor.text}",
            exits=[CautionExit(label="Cross-validation only",
                               decision={"kind": "set_split", "holdout": 0.0, "seed": decision.seed,
                                         "folds": decision.folds})])
    return [RowFlowView(
        title="Where the held-out rows come from",
        caption=clip_words(caption, CAPTION_WORDS),
        emphasis=["train", "holdout"],
        before=before,
        after=after,
    )]


# ── set_roles ────────────────────────────────────────────────────────────────


def _roles_lineage(order: list[str], roles: dict[str, str], touched: set[str]) -> Any:
    columns = [c for c in order if c in roles]
    preds = [c for c in columns if roles[c] in PREDICTOR_ROLES]
    return lineage_of(
        [("raw", c, None) for c in columns] + [("matrix", c, None) for c in preds],
        [(f"raw:{c}", f"matrix:{c}", "kept") for c in preds],
        touched=touched,
        roles=dict(roles),
        max_nodes=30 if not touched else MAX_FOCUS_COLUMNS,
    )


def roles_views(decision: Any, ctx: PreviewContext) -> list[Any]:
    order = _order(ctx)
    new = dict(decision.roles)
    old = dict(ctx.state.roles or {})
    changed = [c for c in order if old.get(c) != new.get(c) and (c in old or c in new)] if old else []
    touched = set(changed[:MAX_FOCUS_COLUMNS])
    before = _roles_lineage(order, old, touched) if old else None
    after = _roles_lineage(order, new, touched)
    now_in = set(predictors(old))
    then_in = predictors(new, order)
    entering = [c for c in then_in if c not in now_in]
    staying = set(then_in)  # built once: inside the test it cost 6 s at 20,000 columns
    leaving = [c for c in predictors(old, order) if c not in staying]

    def names(cols: list[str]) -> str:
        shown = ", ".join(f"`{c}`" for c in cols[:2])
        return shown + (f" and {len(cols) - 2:,} more" if len(cols) > 2 else "")

    if not old:
        counts = {}
        for c in then_in:
            counts[new[c]] = counts.get(new[c], 0) + 1
        plural = {"exposure": "exposures", "covariate": "covariates"}
        parts = ", ".join(f"{fmt_count(n)} {plural.get(role, role) if n != 1 else role}"
                          for role, n in counts.items())
        caption = f"{fmt_count(len(then_in))} of {fmt_count(len(new))} columns enter the model: {parts}."
    elif entering or leaving:
        bits = []
        if entering:
            bits.append(f"{names(entering)} {'enters' if len(entering) == 1 else 'enter'} the model")
        if leaving:
            bits.append(f"{names(leaving)} {'leaves' if len(leaving) == 1 else 'leave'} it")
        caption = "; ".join(bits) + f"; {fmt_count(len(then_in))} predictors in all."
    elif changed:
        c = changed[0]
        caption = (f"The same {fmt_count(len(then_in))} columns enter the model; `{c}` becomes "
                   f"{new.get(c, 'unassigned')}.")
    else:
        caption = f"Nothing changes: the same {fmt_count(len(then_in))} columns enter the model."
    return [LineageView(
        title="Which columns enter the model",
        caption=clip_words(caption, CAPTION_WORDS),
        emphasis=(entering + leaving + changed)[:MAX_FOCUS_COLUMNS],
        before=before,
        after=after,
    )]


register_consequence("set_exclusions", exclusions_views)
register_consequence("set_missing", missing_views)
register_consequence("set_split", split_views)
register_consequence("set_roles", roles_views)

__all__ = ["exclusions_views", "missing_views", "roles_views", "sealed_rows", "split_views"]

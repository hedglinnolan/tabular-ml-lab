"""Consequence previews for the row-side decisions (M1_CONTRACT.md §4; owner: the rows agent).

| Kind             | Views (primary first)                                          |
|------------------|----------------------------------------------------------------|
| ``set_exclusions`` | row_flow · distribution of the excluded column, cut marked   |
| ``set_missing``    | row_flow · table_focus of the rows and cells affected        |
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
    CAPTION_WORDS, TITLE_WORDS, DistributionView, LineageView, PreviewContext, RowFlowView,
    RowStep, TableFocusView, TableRow, _histogram_pair, clip_words, fmt_count, fmt_value,
    lineage_of, register_consequence,
)
from turbotab.core.decisions import ROW_ID, ProjectState
from turbotab.core.stages.rows import (
    PREDICTOR_ROLES, _missing_mask, cohort_flow, cohort_inputs, draw_split, predictors, rule_keep,
    split_inputs,
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
    """Every row but the held-out ones once the split exists; None (every row) before."""
    split = ctx.artifact("split")
    if split is None:
        return None
    held = sealed_rows(split)
    return np.setdiff1d(np.arange(ctx.datastore.n_rows, dtype=np.int64), held, assume_unique=True)


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
        steps, kept = cohort_flow(frame, target=st.target, rules=st.exclusions, missing=st.missing,
                                  predictor_columns=g, missing_frame=mask)
        results.append((steps, kept))
    return frame, mask, results


def _pool_words(ctx: PreviewContext) -> str:
    return "rows"


# ── set_exclusions ───────────────────────────────────────────────────────────


def exclusions_views(decision: Any, ctx: PreviewContext) -> list[Any]:
    state = ctx.state
    proposed = state.model_copy(update={"exclusions": list(decision.rules)})
    frame, _, [(before, _), (after, _)] = _flows(ctx, [state, proposed])
    training = ctx.artifact("split") is not None
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
    cuts = sorted({float(b) for b in (rule.low, rule.high) if b is not None}
                  | {float(b) for lo_hi in (rule.by.ranges.values() if rule.by else []) for b in lo_hi
                     if b is not None})
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
        cuts=cuts,
    )


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
    proposed = state.model_copy(update={"missing": decision.strategy})
    probe = state.model_copy(update={"missing": "complete_case"})  # which predictors can be missing
    frame, mask, [(before, _), (after, _), (_, _)] = _flows(ctx, [state, proposed, probe])
    training = ctx.artifact("split") is not None
    order = _order(ctx)
    preds = predictors(state.roles, order)
    # Rows that reach the missing-values step: past the outcome and the rules.
    reach_ids = _kept_before_missing(frame, state)
    gaps: dict[str, int] = {}
    if mask is not None:
        m = mask.loc[reach_ids]
        gaps = {c: int(m[c].isna().sum()) for c in m.columns if c in preds and m[c].isna().any()}
    ranked = sorted(gaps, key=lambda c: (-gaps[c], order.index(c) if c in order else 0))
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
    views.append(RowFlowView(
        title="Rows kept when values are missing",
        caption=clip_words(caption, CAPTION_WORDS),
        emphasis=["complete_cases"],
        before=_steps(before, training),
        after=_steps(after, training),
    ))
    if ranked:
        views.append(_gaps_view(ctx, reach_ids, ranked, gaps, decision.strategy))
    return views


def _kept_before_missing(frame: Any, state: ProjectState) -> Any:
    _, kept = cohort_flow(frame, target=state.target, rules=state.exclusions, missing=None,
                          predictor_columns=[])
    return kept


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
    fills: dict[str, Any] = {}
    if strategy == "impute":
        for c in columns:
            s = values[c].dropna()
            if s.empty:
                fills[c] = None
            elif s.dtype.kind in "biuf" and s.dtype.kind != "b":
                fills[c] = float(s.median())
            else:
                fills[c] = s.mode().iloc[0]
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
        caption = (f"{fmt_count(n_cells)} cells would be filled; shown with each column's median or "
                   f"most common value on these rows.")
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
    inputs = split_inputs(state, universe, ctx.datastore, task)
    _, info = draw_split(rows, holdout=decision.holdout, seed=decision.seed, folds=decision.folds,
                         universe=universe, **inputs)
    before = [RowStep(**s) for s in steps]
    current = ctx.artifact("split")
    if current is not None:
        before += _fork(current.data)
    after = [RowStep(**s) for s in steps] + _fork(info)
    if info["n_holdout"]:
        caption = (f"{fmt_count(info['n_holdout'])} rows held out, {fmt_count(info['n_train'])} train "
                   f"in {fmt_count(info['folds'])} folds")
    else:
        caption = f"All {fmt_count(info['n_train'])} rows train, scored by {fmt_count(info['folds'])}-fold cross-validation"
    if info["grouped_by"]:
        caption += f", grouped by `{info['grouped_by']}`"
    caption += "."
    if 0 < info["n_holdout"] < 30:
        caption = (f"Only {fmt_count(info['n_holdout'])} rows held out: a score on so few rows swings "
                   f"widely.")
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
    leaving = [c for c in predictors(old, order) if c not in set(then_in)]

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

"""Consequence previews for the first three questions: the lens, the outcome and the purpose.

None of them changes a value, so the generic diff has nothing to show; each still changes what the
app does with this table, and the stage shows that on the user's own columns (BLUEPRINT §11 and
DRIVE_RUBRIC §5.14: every answer visibly changes something downstream, or says plainly that it
does not):

| Kind          | Views (primary first)                                                     |
|---------------|---------------------------------------------------------------------------|
| ``set_lens``    | lineage: the role each column is read as under this lens; a note naming   |
|               | what it recognizes (total energy, the nutrients) and the questions it adds |
| ``set_target``  | row_flow: the rows with the outcome recorded · distribution of the outcome |
| ``set_purpose`` | a note: the shelf's order under this purpose and what the Results report   |

The outcome's counts are made over the preview pool (every row before the split exists, every row
but the held-out ones after); the lens reads column names and summaries only.

Importing this module registers the builders.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from turbotab.core.consequences import (
    CAPTION_WORDS, TITLE_WORDS, DistributionView, LineageView, PreviewContext, RowFlowView, RowStep,
    _histogram_pair, clip_words, fmt_count, lineage_of, register_consequence,
)
from turbotab.core.decisions import ROW_ID
from turbotab.core.plan_previews import specifies_the_model

MAX_LINEAGE = 30
LENS_NOUN = {
    "dietary": "dietary",
    "metabolomics": "metabolomics",
    "genomics": "genomics",
    "clinical": "clinical",
    "survey": "survey",
}


def _summaries(ctx: PreviewContext) -> list[dict[str, Any]]:
    """The profile's column summaries (fresh), else the ingest's column records."""
    profile = ctx.artifact("profile")
    data = getattr(profile, "data", profile)
    if isinstance(data, dict) and data.get("columns"):
        return [dict(c) for c in data["columns"] if not str(c.get("name", "")).startswith("__")]
    return [c.to_dict() for c in ctx.datastore.info().columns if c.name != ROW_ID]


_ROLES: dict[tuple[Any, ...], dict[str, str]] = {}


def _roles_under(ctx: PreviewContext, lens: list[str] | None) -> dict[str, str]:
    """The role each column is read as under ``lens`` (remembered per table, lens and outcome:
    on a 500-column assay the reading takes half a second, and arrow keys flip lenses fast)."""
    from turbotab.core.stages.rows import _acquisition_columns, _energy_reading, propose_roles

    store = ctx.datastore
    key = (ctx.project_id, int(store.n_rows), tuple(store.columns), tuple(sorted(lens or [])),
           ctx.state.target)
    if key not in _ROLES:
        columns = _summaries(ctx)
        proposals = propose_roles(columns, lens=lens, target=ctx.state.target,
                                  n_rows=int(store.n_rows),
                                  energy_column=_energy_reading(columns),
                                  acquisition=_acquisition_columns(columns))
        _ROLES[key] = {str(p["column"]): str(p["proposed"]) for p in proposals}
        while len(_ROLES) > 64:
            _ROLES.pop(next(iter(_ROLES)))
    return dict(_ROLES[key])


def _names(columns: list[str], limit: int = 3) -> str:
    shown = [f"`{c}`" for c in columns[:limit]]
    if len(columns) > limit:
        return f"{', '.join(shown)} and {len(columns) - limit:,} more"
    return shown[0] if len(shown) == 1 else f"{', '.join(shown[:-1])} and {shown[-1]}"


_NUMBER = {3: "three", 4: "four", 5: "five"}


def _and(items: list[str]) -> str:
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


# The pack's checks read the whole table, so the lens preview runs them only on a table small
# enough to answer within a preview's time (NHANES, 21,849 × 29, is 0.6 M cells); a wider one says
# the checks run without counting what they raise.
LENS_CHECK_CELLS = 2_000_000
_FOUND: dict[tuple[Any, ...], int] = {}


def _small_table(ctx: PreviewContext) -> Any:
    """The whole table when it is small enough for the pack checks, else None.

    Before the split only: the checks read every row, as the findings do, and once rows are sealed
    no preview reads them. The basis then says every row was counted.
    """
    store = ctx.datastore
    if ctx.sealed_row_ids is not None:
        return None
    n_cols = len([c for c in store.columns if c != ROW_ID])
    if int(store.n_rows) * max(1, n_cols) > LENS_CHECK_CELLS:
        return None
    ctx.read["counted"] = int(store.n_rows)
    return store.materialize([c for c in store.columns if c != ROW_ID]).reset_index(drop=True)


def _reads_feature_major(lens: list[str], table: Any) -> bool:
    """Whether the orientation question would be asked (the interview's rule, audit WP14)."""
    from turbotab.core.detectors import orientation

    try:
        return orientation.asks(lens, orientation.read_frame(table))
    except Exception:  # noqa: BLE001 - a reading that fails asks nothing
        return False


def _pack_findings(ctx: PreviewContext, lens: list[str], table: Any) -> int | None:
    """How many findings the lens's packs raise on this table (cached per table and lens)."""
    if table is None:
        return None
    key = (ctx.project_id, int(ctx.datastore.n_rows), tuple(table.columns), tuple(sorted(lens)))
    if key not in _FOUND:
        from turbotab.core import detectors

        try:  # the findings stage's own pack reading (audit WP14), so the two counts agree
            _FOUND[key] = len(detectors.pack_findings(table, lens))
        except Exception:  # noqa: BLE001 - the count is a courtesy; the lineage is the preview
            return None
        while len(_FOUND) > 64:
            _FOUND.pop(next(iter(_FOUND)))
    return _FOUND[key]


def lens_views(decision: Any, ctx: PreviewContext) -> list[Any]:
    from turbotab.core.stages.proposals import energy_bearing
    from turbotab.core.stages.rows import PREDICTOR_ROLES

    lens = list(decision.lenses)
    if lens == ["other"]:  # audit RO-11 (WP18): no field's lens, the generic checks only
        new = _roles_under(ctx, [])
        order = [c for c in ctx.datastore.columns if c != ROW_ID and c in new]
        n_pred = sum(1 for c in order if new[c] in PREDICTOR_ROLES)
        ctx.read["note"] = ("Only the generic checks run on this table; no field's defaults are "
                            "applied, and the record says so.")
        return [LineageView(
            title=clip_words("Columns as the generic rules read them", TITLE_WORDS),
            caption=clip_words(f"With no field's lens, {fmt_count(n_pred)} columns are read as "
                               f"predictors.", CAPTION_WORDS),
            emphasis=[],
            before=None,
            after=lineage_of([("raw", c, None) for c in order]
                             + [("matrix", c, None) for c in order if new[c] in PREDICTOR_ROLES],
                             [(f"raw:{c}", f"matrix:{c}", "kept") for c in order
                              if new[c] in PREDICTOR_ROLES],
                             touched=set(), roles=dict(new), max_nodes=MAX_LINEAGE),
        )]
    new = _roles_under(ctx, lens)
    old = _roles_under(ctx, list(ctx.state.lens)) if ctx.state.lens else None
    order = [c for c in ctx.datastore.columns if c != ROW_ID and c in new]
    changed = [c for c in order if old is not None and old.get(c) != new.get(c)]
    energy = [c for c in order if new[c] == "energy"]
    nutrients = [c for c in order if new[c] == "exposure" and energy_bearing(c)]
    touched = set((changed or energy + nutrients)[:12])

    def lineage(roles: dict[str, str]) -> Any:
        preds = [c for c in order if roles.get(c) in PREDICTOR_ROLES]
        return lineage_of(
            [("raw", c, None) for c in order] + [("matrix", c, None) for c in preds],
            [(f"raw:{c}", f"matrix:{c}", "kept") for c in preds],
            touched=touched, roles=dict(roles), max_nodes=MAX_LINEAGE)

    said = []
    if energy:
        said.append(f"`{energy[0]}` reads as total energy")
    if nutrients:
        said.append(f"{fmt_count(len(nutrients))} nutrients that carry energy as exposures")
    names = [LENS_NOUN.get(k, k) for k in lens]
    noun = _and(names) if len(names) <= 2 else f"{_NUMBER.get(len(names), len(names))}"
    lenses = f"the {noun} lens" if len(names) == 1 else (
        f"the {noun} lenses" if len(names) == 2 else f"these {noun} lenses")
    if said:
        caption = f"Under {lenses}, " + " and ".join(said) + "."
    else:
        n_pred = sum(1 for c in order if new[c] in PREDICTOR_ROLES)
        caption = f"Under {lenses}, {fmt_count(n_pred)} columns are read as predictors."
    adds = []
    if "dietary" in lens and energy and len(nutrients) >= 1:
        adds.append("energy adjustment")
    if "dietary" in lens and len(nutrients) >= 2:
        adds.append("substitution curves")
    table = _small_table(ctx)
    if table is not None and _reads_feature_major(lens, table):
        adds.insert(0, "which way round the table is")
    # The caption says what the lens reads; the note says what it then does (no repeat): the
    # questions it adds and the findings its pack raises on this table.
    found = _pack_findings(ctx, lens, table)
    whose = f"{noun} pack's" if len(names) == 1 else f"{noun} packs'"
    checks = (f"The {whose} checks raise {fmt_count(found)} "
              f"{'finding' if found == 1 else 'findings'} on this table" if found is not None
              else f"The {whose} checks run on this table")
    ctx.read["note"] = checks + (f", and {_and(adds)} {'joins' if len(adds) == 1 else 'join'} "
                                 f"the questions." if adds else ".")
    return [LineageView(
        title=clip_words(f"Columns as {lenses} read them" if len(names) > 1
                         else f"Columns as the {noun} lens reads them", TITLE_WORDS),
        caption=clip_words(caption, CAPTION_WORDS),
        emphasis=sorted(touched),
        before=lineage(old) if old is not None else None,
        after=lineage(new),
    )]


def target_views(decision: Any, ctx: PreviewContext) -> list[Any]:
    import pandas as pd

    from turbotab.core.row_previews import _pool, _sealed

    column = decision.column
    if column not in ctx.datastore.columns:
        return []
    pool = _pool(ctx)
    frame = ctx.datastore.materialize([column], pool)
    values = frame[column]
    n = len(frame)
    measured = int(values.notna().sum())
    first = "Rows not held out" if _sealed(ctx) else "Rows in the table"
    before = [RowStep(key="loaded", label=first, n=n)]
    after = before + [RowStep(key="outcome_measured", label=f"`{column}` recorded", n=measured,
                              dropped=n - measured, reason=f"no value for the outcome `{column}`")]
    if measured == n:
        caption = f"Every one of the {fmt_count(n)} rows records `{column}`, so none leaves."
    else:
        caption = (f"{fmt_count(measured)} of {fmt_count(n)} rows record `{column}`; the "
                   f"{fmt_count(n - measured)} without it leave the analysis.")
    views: list[Any] = [RowFlowView(
        title=clip_words(f"Rows with `{column}` recorded", TITLE_WORDS),
        caption=clip_words(caption, CAPTION_WORDS),
        emphasis=["outcome_measured"],
        before=before,
        after=after,
    )]
    numeric = pd.to_numeric(values, errors="coerce") if values.dtype.kind in "biuf" else None
    # The outcome card shows only what its own questions need: for a number, its observed range and
    # recorded unit (the unit and impossible-value questions read them). Its distribution, a
    # histogram or a median and middle half, waits for the outcome alone's gate (CROSSWALK, "The
    # outcome card"; lockbox constitution §04: "is this data corrupted?", not "where to cut?").
    if numeric is not None and values.dtype.kind != "b" and numeric.notna().sum() > 1 \
            and numeric.nunique() > 2:
        from turbotab.core.units import outcome_unit, recorded_unit, with_unit

        x = numeric.to_numpy(dtype=float, na_value=np.nan)
        finite = x[np.isfinite(x)]
        # Only a recorded unit is stated (BLUEPRINT §14.3: a header's letters are a name).
        unit, _ = outcome_unit(column, finite, recorded=recorded_unit(ctx.state, column))
        low = with_unit(f"{finite.min():,.4g}", unit)
        high = with_unit(f"{finite.max():,.4g}", unit)
        when = ("once you decide which rows to hold out"
                if getattr(ctx.state, "purpose", None) == "prediction" else "after Who's in")
        ctx.read["note"] = (f"`{column}` runs from {low} to {high} over {fmt_count(len(finite))} "
                            f"rows; its distribution is shown {when}.")
    else:
        ctx.read["note"] = _levels_note(column, values)
    return views


def _levels_note(column: str, values: Any) -> str | None:
    """A class outcome's levels and their counts, and what the next question asks of them."""
    counts = values.dropna().value_counts()
    if counts.empty:
        return None
    from turbotab.core.datastore import json_safe
    from turbotab.core.voice import number

    shown = [f"`{number(json_safe(level))}` on {fmt_count(n)}" for level, n in counts.head(3).items()]
    listed = shown[0] if len(shown) == 1 else f"{', '.join(shown[:-1])} and {shown[-1]}"
    if len(counts) == 2:
        return f"Two levels: {listed} rows; the next question asks which is the event."
    more = f", and {len(counts) - 3:,} more levels" if len(counts) > 3 else ""
    return f"{len(counts):,} levels, most common {listed} rows{more}."


def purpose_views(decision: Any, ctx: PreviewContext) -> list[Any]:
    """What the purpose changes downstream, said in a note: no value changes, so no view."""
    purpose = decision.purpose
    # What the purpose does to the shelf, as a policy rather than a promised order: the shelf is
    # ranked later with what is known then (the roles, the events per predictor), so a plain
    # linear model can still rank last under inference when the rows cannot support it.
    shelf = ("the shelf favors the linear model" if purpose == "inference" else
             "the shelf favors penalized and flexible models")
    # Which later questions and checks change (OPENING_SEQUENCE §2.5: the advice inverts in
    # several places): the shelf's order, the Results, and the missing-value indicator.
    if purpose == "inference":
        text = (f"With inference, {shelf}; the Results report coefficients with 95% intervals, and "
                f"a missing-value indicator would bias them.")
    else:
        text = (f"With prediction, {shelf}; the Results lead with scores on unseen rows, and "
                f"missing-value indicators are legitimate, since they exist at deployment.")
    ctx.read["note"] = text
    return []


register_consequence("set_lens", lens_views)
# The outcome itself, and whether its model is estimated at all (inference): beside each, what
# the surveyed population blocks of that model (MODELING_SEQUENCE §4;
# ``plan_previews.specifies_the_model``).
register_consequence("set_target", specifies_the_model(target_views))
register_consequence("set_purpose", specifies_the_model(purpose_views))

__all__ = ["lens_views", "purpose_views", "target_views"]

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


def _roles_under(ctx: PreviewContext, lens: list[str] | None) -> dict[str, str]:
    from turbotab.core.stages.rows import _acquisition_columns, _energy_reading, propose_roles

    columns = _summaries(ctx)
    proposals = propose_roles(columns, lens=lens, target=ctx.state.target,
                              n_rows=int(ctx.datastore.n_rows),
                              energy_column=_energy_reading(columns),
                              acquisition=_acquisition_columns(columns))
    return {str(p["column"]): str(p["proposed"]) for p in proposals}


def _names(columns: list[str], limit: int = 3) -> str:
    shown = [f"`{c}`" for c in columns[:limit]]
    if len(columns) > limit:
        return f"{', '.join(shown)} and {len(columns) - limit:,} more"
    return shown[0] if len(shown) == 1 else f"{', '.join(shown[:-1])} and {shown[-1]}"


def lens_views(decision: Any, ctx: PreviewContext) -> list[Any]:
    from turbotab.core.stages.proposals import energy_bearing
    from turbotab.core.stages.rows import PREDICTOR_ROLES

    lens = list(decision.lenses)
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
    noun = " and ".join(LENS_NOUN.get(k, k) for k in lens)
    if said:
        caption = f"Under the {noun} lens, " + " and ".join(said) + "."
    else:
        n_pred = sum(1 for c in order if new[c] in PREDICTOR_ROLES)
        caption = f"Under the {noun} lens, {fmt_count(n_pred)} columns are read as predictors."
    adds = []
    if "dietary" in lens and energy and len(nutrients) >= 1:
        adds.append("energy adjustment")
    if "dietary" in lens and len(nutrients) >= 2:
        adds.append("substitution curves")
    # The caption says what the lens reads; the note says what it then does (no repeat).
    ctx.read["note"] = (f"The {noun} pack's checks run on this table" +
                        (f", and {' and '.join(adds)} join the questions." if adds else "."))
    return [LineageView(
        title=clip_words(f"Columns as the {noun} lens reads them", TITLE_WORDS),
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
    if numeric is not None and values.dtype.kind != "b" and numeric.notna().sum() > 1 \
            and numeric.nunique() > 2:
        x = numeric.to_numpy(dtype=float, na_value=np.nan)
        hist, _ = _histogram_pair(x, x)
        finite = x[np.isfinite(x)]
        views.append(DistributionView(
            title=clip_words(f"Values of `{column}`", TITLE_WORDS),
            caption=clip_words(f"`{column}`: median {np.median(finite):,.4g}, middle half "
                               f"{np.percentile(finite, 25):,.4g}–{np.percentile(finite, 75):,.4g}, "
                               f"over {fmt_count(len(finite))} rows.", CAPTION_WORDS),
            emphasis=[column],
            column=column,
            before=hist,
            after=hist,
            before_label="Every recorded value",
            after_label="Every recorded value",
        ))
    return views


def purpose_views(decision: Any, ctx: PreviewContext) -> list[Any]:
    """What the purpose changes downstream, said in a note: no value changes, so no view."""
    from turbotab.core.models import Situation, rank

    purpose = decision.purpose
    state = ctx.state
    info = ctx.artifact("target_info")
    data = getattr(info, "data", info)
    task = state.task or (data.get("task") if isinstance(data, dict) else None) or "regression"
    cohort = ctx.artifact("cohort")
    cdata = getattr(cohort, "data", cohort)
    n_rows = int(cdata["n_final"]) if isinstance(cdata, dict) and "n_final" in cdata \
        else int(ctx.datastore.n_rows)
    roles = state.roles or {}
    if roles:
        n_features = sum(1 for r in roles.values() if r in ("exposure", "covariate", "energy"))
    else:
        n_features = max(1, len(ctx.datastore.columns) - 2)
    try:
        ranked = rank(Situation(task=task, purpose=purpose, n_rows=n_rows, n_features=n_features))
        order = [f.label.lower() for f, _ in ranked]
    except Exception:  # noqa: BLE001 - the order is a courtesy; the report below is the point
        order = []
    shelf = f"the shelf leads with the {order[0]}" if order else "the shelf is reordered"
    if purpose == "inference":
        text = (f"With inference, {shelf}, and the Results report each exposure's coefficient with a "
                f"95% confidence interval and p-value, estimated on the training rows.")
    else:
        text = (f"With prediction, {shelf}, and the Results lead with the score on rows the models "
                f"never saw; coefficients are shown but not interpreted.")
    ctx.read["note"] = text
    return []


register_consequence("set_lens", lens_views)
register_consequence("set_target", target_views)
register_consequence("set_purpose", purpose_views)

__all__ = ["lens_views", "purpose_views", "target_views"]

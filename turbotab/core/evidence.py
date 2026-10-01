"""Finding evidence: the views that show why a finding was raised (M1_CONTRACT §12.3).

``GET /api/projects/{pid}/findings/{fid}/evidence`` answers a :class:`PreviewResult` built from
the same closed vocabulary of views as the consequence previews, so the stage draws a finding's
evidence exactly as it draws an option's consequence. Nothing is recorded.

Builders register per finding family (``family(finding_id)``: ``voice::flag__imputed_bmi`` →
``voice::flag``), and a finding with no builder of its own gets the generic one:

=================================  ===============================================================
family                             views
=================================  ===============================================================
``pack::dietary::energy_adjustment``  the nutrient most correlated with energy, against energy
``pack::dietary::implausible_intake`` energy's distribution with the rule's bounds marked, and
                                   what the rule keeps
``voice::flag``                    the rows the flag marks, with the values it marks highlighted
``voice::identifier``              the identifier beside the first columns: one value per row
``voice::repeats``                 the rows of the unit that repeats most
anything else                      the affected columns (blank cells highlighted), and the
                                   distribution of the first numeric one
=================================  ===============================================================

Held-out discipline (Nolan's ruling, F:GUIDED-096): evidence that answers "is this data what it
says?" (an implausible value, a flag, an identifier) reads every row, as the finding did; evidence
that informs a modeling choice (energy adjustment) reads only rows outside the held-out ones once
the split exists. A view whose before and after are the same shows the data as it is: evidence
has no option to flip to, except where a finding's lever has a picture of its own (the rule's cut).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from turbotab.core.consequences import (
    CAPTION_WORDS, TITLE_WORDS, DistributionView, HistogramData, PreviewResult, RelationshipView,
    TableFocusView, TableRow, _histogram_pair, clip_words, fmt_count, fmt_value,
)
from turbotab.core.decisions import ROW_ID

MAX_COLUMNS = 12
MAX_ROWS = 8
POINTS = 800
_TRUTHY = {"true", "1", "1.0", "yes", "y", "t"}


@dataclass
class EvidenceContext:
    """What an evidence builder may read. The server constructs it per request."""

    state: Any  # ProjectState
    datastore: Any  # DataStore
    sealed: Any | None = None  # the newest split's held-out row ids, once a split exists
    sample_size: int = 5_000
    # The rows the models train on (the cohort outside the seal), once a split exists: modeling
    # evidence reads the same rows as the modeling previews, so their numbers agree.
    training: Any | None = None

    @property
    def n_rows(self) -> int:
        return int(self.datastore.n_rows)

    def _modeling_pool(self) -> tuple[Any, str]:
        if self.training is not None:
            return np.asarray(self.training, dtype=np.int64), "training rows"
        pool = np.arange(self.n_rows, dtype=np.int64)
        if self.sealed is not None and len(self.sealed):
            pool = np.setdiff1d(pool, np.asarray(self.sealed, dtype=np.int64), assume_unique=True)
            return pool, "rows not held out"
        return pool, "rows"

    def modeling_rows(self) -> Any:
        """Rows a modeling-choice view may read: the training rows (else every row but the held-out
        ones), sampled exactly as the previews sample them (``PreviewContext.sample_row_ids``)."""
        pool, _ = self._modeling_pool()
        if len(pool) > self.sample_size:
            pool = np.sort(np.random.default_rng(0).choice(pool, size=self.sample_size, replace=False))
        return pool

    def modeling_basis(self, n: int, points: int | None = None) -> str:
        """The rows a modeling view read, worded as the previews word theirs (service.preview_basis)."""
        held = self.sealed is not None and len(self.sealed)
        pool, words = self._modeling_pool()
        where = f"all {len(pool):,} {words}" if n >= len(pool) else (
            f"a sample of {n:,} of the {len(pool):,} {words}")
        drawn = f"; the scatter draws {points:,} of them" if points is not None and points < n else ""
        tail = "; held-out rows stay sealed" if held else ""
        return f"Values on {where}{drawn}{tail}."

    def all_rows_basis(self) -> str:
        return f"Every one of the {self.n_rows:,} rows, as the finding read them."

    def columns(self) -> list[str]:
        return [c for c in self.datastore.columns if c != ROW_ID]


Builder = Callable[[dict, EvidenceContext], "tuple[list[Any], str] | None"]
_BUILDERS: dict[str, Builder] = {}


def register_evidence(family: str, builder: Builder) -> Builder:
    """``builder(finding, ctx) -> (views, basis) | None`` for findings of ``family``.

    None (or no views) falls through to the generic builder.
    """
    _BUILDERS[family] = builder
    return builder


def family_of(finding_id: str) -> str:
    from turbotab.core.stages.finding_words import family

    return family(finding_id)


def evidence(finding: dict, ctx: EvidenceContext) -> PreviewResult:
    """The views that show why ``finding`` was raised, primary first, at most three."""
    fam = family_of(str(finding["id"]))
    built = None
    builder = _BUILDERS.get(fam)
    if builder is not None:
        built = builder(finding, ctx)
    if not built or not built[0]:
        built = generic(finding, ctx)
    views, basis = built
    views = views[:3]
    from turbotab.core import coach

    coach.annotate_evidence(fam, finding, views, ctx)  # ≤ 2 data-grounded notes per view
    note = None if views else ("This finding is about columns the table does not have, so there is "
                               "nothing of it to draw.")
    return PreviewResult(kind=fam, views=views, basis=basis, note=note)


# ── helpers ──────────────────────────────────────────────────────────────────


def _json(value: Any) -> Any:
    from turbotab.core.datastore import json_safe

    return json_safe(value)


def _numeric(series: Any) -> bool:
    return series.dtype.kind in "biuf" and series.dtype.kind != "b"


def _table(frame: Any, columns: list[str], rows: Sequence[Any], cells: list[tuple[int, str]], *,
           title: str, caption: str, n_affected: int, emphasis: list[str]) -> TableFocusView:
    out = []
    for r in rows:
        values = {c: _json(frame.at[r, c]) for c in columns}
        out.append(TableRow(row_id=int(r), before=values, after=dict(values)))
    return TableFocusView(
        title=clip_words(title, TITLE_WORDS),
        caption=clip_words(caption, CAPTION_WORDS),
        emphasis=emphasis,
        columns_before=columns,
        columns_after=columns,
        rows=out,
        changed=cells,
        n_affected_columns=n_affected,
    )


def _histogram(values: Any) -> HistogramData:
    hist, _ = _histogram_pair(values, values)
    return hist


def _distribution(column: str, values: Any, caption: str, *, title: str | None = None) -> DistributionView:
    hist = _histogram(values)
    return DistributionView(
        title=clip_words(title or f"Values of `{column}`", TITLE_WORDS),
        caption=clip_words(caption, CAPTION_WORDS),
        emphasis=[column],
        column=column,
        before=hist,
        after=hist,
        before_label="Every recorded value",
        after_label="Every recorded value",
    )


# ── the builders ─────────────────────────────────────────────────────────────


def energy_evidence(finding: dict, ctx: EvidenceContext) -> tuple[list[Any], str] | None:
    """The nutrient most correlated with total energy, against it, on rows outside the seal."""
    import pandas as pd

    from turbotab.core.stages.proposals import energy_bearing, energy_column

    info = {c.name: c.to_dict() for c in ctx.datastore.info().columns}
    roles = dict(ctx.state.roles or {})
    E = energy_column(info, roles)
    if E is None:
        return None
    affected = [c for c in finding.get("affected_columns") or [] if c in info and c != E]
    pool = affected or list(info)
    nutrients = [c for c in pool if c != ctx.state.target and energy_bearing(c)
                 and info[c].get("dtype") in ("numeric", "integer")
                 and roles.get(c) in (None, "exposure")]
    if not nutrients:
        return None
    rows = ctx.modeling_rows()
    frame = ctx.datastore.materialize([E, *nutrients], rows)
    e = pd.to_numeric(frame[E], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    # The nutrient the finding names (it chose it on every row as loaded); else the closest here.
    named = re.findall(r"`([^`]+)`", str(finding.get("summary") or ""))
    best, best_r = None, 0.0
    for n in nutrients:
        x = pd.to_numeric(frame[n], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        ok = np.isfinite(x) & np.isfinite(e)
        if ok.sum() < 3 or np.std(x[ok]) == 0 or np.std(e[ok]) == 0:
            continue
        r = float(np.corrcoef(x[ok], e[ok])[0, 1])
        if n in named[:1]:
            best, best_r = n, r
            break
        if best is None or abs(r) > abs(best_r):
            best, best_r = n, r
    if best is None:
        return None
    y = pd.to_numeric(frame[best], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    ok = np.flatnonzero(np.isfinite(y) & np.isfinite(e))
    if len(ok) > POINTS:
        ok = np.sort(np.random.default_rng(0).choice(ok, size=POINTS, replace=False))
    points = [(float(e[i]), float(y[i])) for i in ok]
    caption = (f"On these rows `{best}` correlates {fmt_value(round(best_r, 2))} with `{E}`: more of "
               f"it mostly means more food.")
    view = RelationshipView(
        title=clip_words(f"`{best}` against `{E}`", TITLE_WORDS),
        caption=clip_words(caption, CAPTION_WORDS),
        emphasis=[best, E],
        x_label=E,
        y_label_before=best,
        y_label_after=best,
        points_before=points,
        points_after=list(points),
        r_before=best_r,
        r_after=best_r,
    )
    return [view], ctx.modeling_basis(len(frame), points=len(points))


def implausible_evidence(finding: dict, ctx: EvidenceContext) -> tuple[list[Any], str] | None:
    """Energy's distribution with the pack's plausible range marked, and what that range keeps."""
    import pandas as pd

    from turbotab.core.decisions import ExclusionRule
    from turbotab.core.row_previews import rule_marks

    columns = [c for c in finding.get("affected_columns") or [] if c in ctx.columns()]
    if not columns:
        return None
    column = columns[0]
    try:
        from turbotab.packs import _PLAUSIBLE_KCAL as bounds  # the detector's own range
    except ImportError:  # pragma: no cover - the legacy pack always defines it
        bounds = (500.0, 5000.0)
    low, high = float(bounds[0]), float(bounds[1])
    rule = ExclusionRule(column=column, low=low, high=high, reason="implausible intake")
    values = pd.to_numeric(ctx.datastore.materialize([column])[column], errors="coerce").astype(float)
    finite = values[np.isfinite(values)]
    kept = finite[(finite >= low) & (finite <= high)]
    before, after = _histogram_pair(np.concatenate([finite.to_numpy(), [low, high]]), kept.to_numpy())
    edges = np.asarray(before.edges, dtype=float)
    counts, _ = np.histogram(finite.to_numpy(), edges)  # the bounds only widened the axis
    before = before.model_copy(update={"counts": counts.astype(int).tolist(),
                                       "n_missing": int(values.isna().sum())})
    out = int(len(finite) - len(kept))
    caption = (f"{fmt_count(out)} of {fmt_count(len(values))} rows report `{column}` below "
               f"{fmt_value(low)} or above {fmt_value(high)}.")
    view = DistributionView(
        title=clip_words(f"`{column}` against the plausible range", TITLE_WORDS),
        caption=clip_words(caption, CAPTION_WORDS),
        emphasis=[column],
        column=column,
        before=before,
        after=after,
        before_label="Every recorded value",
        after_label="Inside the plausible range",
        marks=rule_marks(rule),
    )
    return [view], ctx.all_rows_basis()


def flag_evidence(finding: dict, ctx: EvidenceContext) -> tuple[list[Any], str] | None:
    """The rows a flag marks, the flagged values of its base column highlighted."""
    columns = [c for c in finding.get("affected_columns") or [] if c in ctx.columns()]
    if not columns:
        return None
    flag, base = columns[0], (columns[1] if len(columns) > 1 else None)
    shown = [flag] + ([base] if base else [])
    frame = ctx.datastore.materialize(shown)
    on = frame[flag].map(lambda v: str(v).strip().lower() in _TRUTHY)
    marked = frame.index[on.to_numpy()]
    rows = sorted(marked[:MAX_ROWS])
    cells = [(int(r), base) for r in rows] if base else [(int(r), flag) for r in rows]
    what = f"`{base}` was filled in, not measured" if base else "the row was flagged"
    caption = f"`{flag}` is true on {fmt_count(len(marked))} rows; on each, {what}."
    view = _table(frame, shown, rows, cells, title=f"Rows `{flag}` marks", caption=caption,
                  n_affected=len(shown), emphasis=[base or flag])
    return [view], ctx.all_rows_basis()


def identifier_evidence(finding: dict, ctx: EvidenceContext) -> tuple[list[Any], str] | None:
    """The identifier beside the table's first columns: one value per row."""
    names = ctx.columns()
    columns = [c for c in finding.get("affected_columns") or [] if c in names]
    if not columns:
        return None
    column = columns[0]
    context = [c for c in names if c != column][:3]
    frame = ctx.datastore.materialize([column, *context])
    values = frame[column].dropna()
    rows = list(frame.index[:MAX_ROWS])
    caption = (f"`{column}` takes {fmt_count(values.nunique())} different values in "
               f"{fmt_count(len(frame))} rows: it names rows, not traits.")
    view = _table(frame, [column, *context], rows, [(int(r), column) for r in rows],
                  title=f"`{column}` names each row", caption=caption, n_affected=1,
                  emphasis=[column])
    return [view], ctx.all_rows_basis()


def repeats_evidence(finding: dict, ctx: EvidenceContext) -> tuple[list[Any], str] | None:
    """The rows of the unit that repeats most: one participant, several rows."""
    names = ctx.columns()
    columns = [c for c in finding.get("affected_columns") or [] if c in names]
    if not columns:
        return None
    column = columns[0]
    context = [c for c in names if c != column][:3]
    frame = ctx.datastore.materialize([column, *context])
    counts = frame[column].dropna().value_counts()
    if counts.empty:
        return None
    unit = counts.index[0]
    rows = list(frame.index[(frame[column] == unit).to_numpy()][:MAX_ROWS])
    caption = (f"`{column}` repeats: {fmt_count(len(frame[column].dropna()))} rows from "
               f"{fmt_count(len(counts))} units; one unit's rows are shown.")
    view = _table(frame, [column, *context], rows, [(int(r), column) for r in rows],
                  title=f"One `{column}`, several rows", caption=caption, n_affected=1,
                  emphasis=[column])
    return [view], ctx.all_rows_basis()


def generic(finding: dict, ctx: EvidenceContext) -> tuple[list[Any], str]:
    """The affected columns, blank cells highlighted, and the first numeric one's distribution."""
    names = ctx.columns()
    columns = [c for c in finding.get("affected_columns") or [] if c in names][:MAX_COLUMNS]
    if not columns:
        return [], ctx.all_rows_basis()
    frame = ctx.datastore.materialize(columns)
    blank = frame.isna()
    gappy = blank.any(axis=1)
    if gappy.any():
        rows = list(frame.index[gappy.to_numpy()][:MAX_ROWS])
        cells = [(int(r), c) for r in rows for c in columns if bool(blank.at[r, c])]
        worst = max(columns, key=lambda c: int(blank[c].sum()))
        caption = (f"{fmt_count(int(blank[worst].sum()))} of {fmt_count(len(frame))} rows are blank "
                   f"in `{worst}`; a few are shown.")
    else:
        rows = list(frame.index[:MAX_ROWS])
        cells = []
        caption = f"The first rows of {', '.join(f'`{c}`' for c in columns[:3])}" + (
            f" and {len(columns) - 3} more." if len(columns) > 3 else ".")
    views: list[Any] = [_table(frame, columns, rows, cells, title="The columns this is about",
                               caption=caption, n_affected=len(columns), emphasis=columns)]
    numeric = [c for c in columns if _numeric(frame[c])]
    if numeric:
        c = numeric[0]
        v = frame[c].to_numpy(dtype=float, na_value=np.nan)
        finite = v[np.isfinite(v)]
        if finite.size:
            caption = (f"`{c}` over {fmt_count(len(v))} rows: {fmt_count(len(v) - finite.size)} blank, "
                       f"from {fmt_value(finite.min())} to {fmt_value(finite.max())}.")
            views.append(_distribution(c, v, caption))
    return views, ctx.all_rows_basis()


register_evidence("pack::dietary::energy_adjustment", energy_evidence)
register_evidence("pack::dietary::implausible_intake", implausible_evidence)
register_evidence("voice::flag", flag_evidence)
register_evidence("voice::identifier", identifier_evidence)
register_evidence("voice::repeats", repeats_evidence)

__all__ = ["EvidenceContext", "evidence", "family_of", "generic", "register_evidence"]

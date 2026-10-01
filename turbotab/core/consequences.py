"""Consequence previews — what a choice would do to the user's own data, before it is recorded.

BLUEPRINT §11 ("guidance without walls"): the explanation of an option is its effect on the user's
table, not a paragraph. When the user hovers or focuses an option, the client asks
``POST /api/projects/{pid}/preview`` with the decision that option would record, and the planner
returns **at most three views — the first is primary** — chosen by what that option actually
changes. Nothing is recorded and nothing is cached as a stage artifact.

Rules every builder keeps:

- **Bounded and fast.** Work on a sample (``ctx.sample_row_ids``) and only the affected columns;
  a preview answers in well under a second on a laptop. Say what it was computed on in ``basis``.
- **Held-out rows stay sealed.** Once the split exists, anything that informs a modeling choice is
  computed on training rows only (Nolan's ruling, F:GUIDED-096). ``ctx.training_row_ids`` is the
  pool; before the split exists, only row-descriptive previews (exclusions, missing values) run.
- **Relevance, not coverage.** Pick the column that changes most, the nutrient most correlated with
  energy, the step that loses most rows — never every column. Wide tables show the affected
  columns only, with a count of the rest.
- **Captions are facts about their data** ("protein_g correlates 0.71 with kcal; after adjustment,
  0.00"), ≤ 20 words; titles ≤ 8 words. A test enforces the budgets.

**The scaling strategy — a closed vocabulary plus a generic diff.** Every modeling decision, now or
in any later milestone, changes some combination of five things: which rows are in, which columns
exist (and where they came from), a column's values, how columns relate, and what a model outputs.
Each has one view type here; later milestones may add ``embedding`` (clustering, PCA) and
``metric``/``curve`` — and nothing else without a design decision. A new decision kind therefore
only has to say how to apply itself to a sample (:func:`register_transform`); the generic diff
(:func:`diff_views`) compares before and after and picks what to show by measured change. Domain
builders (:func:`register_consequence`) override only where a specific picture teaches better —
energy adjustment's nutrient-against-energy scatter *is* the method.

Builders register per decision kind; the M1 agents own the builders for their kinds
(M1_CONTRACT.md §8).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any, Callable, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

from turbotab.core.decisions import ProjectState, Role

TITLE_WORDS = 8
CAPTION_WORDS = 20
MAX_VIEWS = 3


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_serialization_defaults_required=True)


# ── shared pieces (also used by stage artifacts) ─────────────────────────────


class RowStep(_Model):
    """One step of the participant flow (CONSORT/STROBE style)."""

    key: str
    label: str
    n: int
    dropped: int = 0
    reason: str | None = None
    decision_id: str | None = None


class HistogramData(_Model):
    edges: list[float]
    counts: list[int]
    n_missing: int = 0


class LineageNode(_Model):
    id: str
    column: str | None  # None for a collapsed group node
    lane: Literal["raw", "adjusted", "matrix"]
    role: Role | None = None
    label: str
    formula: str | None = None  # mono, e.g. "protein_g − ĝ(kcal)"; None when unchanged
    group: str | None = None  # the role group a collapsed node stands for
    count: int = 1  # columns this node stands for (> 1 only for a collapsed group)


class LineageLink(_Model):
    source: str
    target: str
    operation: str  # "kept", "energy-adjusted (residual)", "scaled", "one-hot", "dropped", ...


class Lineage(_Model):
    nodes: list[LineageNode]
    links: list[LineageLink]
    collapsed: bool = False  # True when groups were collapsed to keep the picture readable


# ── storyboards ──────────────────────────────────────────────────────────────
# A view's ``story`` is the method's own labeled intermediate states between its before and its
# after (BLUEPRINT §11.1): real, computed states, never an interpolation. The client's flip plays
# before → story… → after, and a save can capture any of them. Empty: a direct before ⇄ after.
# Each frame has the data its view kind draws, and a ``label`` of at most :data:`FRAME_WORDS`.

FRAME_WORDS = 8


class FitLine(_Model):
    slope: float
    intercept: float


class RelationshipFrame(_Model):
    label: str
    points: list[tuple[float, float]]
    r: float | None
    fit_line: FitLine | None = None  # y = intercept + slope · x, on this frame's axes
    y_label: str | None = None  # the y axis when it is not the view's before label


class DistributionFrame(_Model):
    label: str
    hist: HistogramData
    x_label: str | None = None  # the axis when it is not the view's column


class LineageFrame(_Model):
    label: str
    lineage: Lineage


class FrameRow(_Model):
    row_id: int
    values: dict[str, Any]


class TableFrame(_Model):
    label: str
    columns: list[str]
    rows: list[FrameRow]


class RowFlowFrame(_Model):
    label: str
    steps: list[RowStep]


# ── the views ────────────────────────────────────────────────────────────────


class _View(_Model):
    title: str
    caption: str
    emphasis: list[str] = Field(default_factory=list)  # columns or step keys to highlight


class RowFlowView(_View):
    kind: Literal["row_flow"] = "row_flow"
    before: list[RowStep]
    after: list[RowStep]
    story: list[RowFlowFrame] = Field(default_factory=list)


class LineageView(_View):
    kind: Literal["lineage"] = "lineage"
    before: Lineage | None
    after: Lineage
    story: list[LineageFrame] = Field(default_factory=list)


class TableRow(_Model):
    row_id: int
    before: dict[str, Any]
    after: dict[str, Any]


class TableFocusView(_View):
    """The working table narrowed to the columns this choice touches (≤ 12 shown, ≤ 8 rows)."""

    kind: Literal["table_focus"] = "table_focus"
    columns_before: list[str]
    columns_after: list[str]
    rows: list[TableRow]
    changed: list[tuple[int, str]]  # (row_id, column) cells whose value changes (evidence: the cells it is about)
    n_affected_columns: int
    story: list[TableFrame] = Field(default_factory=list)


class Mark(_Model):
    """A labeled value on a distribution's axis, e.g. an exclusion cut-off ("500 kcal")."""

    value: float
    label: str
    group: str | None = None  # the level a by-level cut applies to (e.g. "female"); None = all rows


class DistributionView(_View):
    kind: Literal["distribution"] = "distribution"
    column: str
    before: HistogramData
    after: HistogramData
    before_label: str
    after_label: str
    marks: list[Mark] = Field(default_factory=list)  # labeled values on the axis (e.g. a rule's bounds)
    story: list[DistributionFrame] = Field(default_factory=list)


class RelationshipView(_View):
    """A before/after scatter, e.g. a nutrient against energy before and after adjustment."""

    kind: Literal["relationship"] = "relationship"
    x_label: str
    y_label_before: str
    y_label_after: str
    points_before: list[tuple[float, float]]  # ≤ 800, sampled
    points_after: list[tuple[float, float]]
    r_before: float | None
    r_after: float | None
    story: list[RelationshipFrame] = Field(default_factory=list)


ConsequenceView = Annotated[
    Union[RowFlowView, LineageView, TableFocusView, DistributionView, RelationshipView],
    Field(discriminator="kind"),
]


class PreviewResult(_Model):
    kind: str  # the decision kind previewed
    views: list[ConsequenceView]  # ≤ MAX_VIEWS; the first is primary
    basis: str  # e.g. "5,000 of 17,478 training rows"
    note: str | None = None  # e.g. why there is nothing to show yet


# ── the planner ──────────────────────────────────────────────────────────────


@dataclass
class PreviewContext:
    """What a builder may read. The server constructs it per request."""

    project_id: str
    state: ProjectState  # the state as recorded — the previewed decision is NOT applied
    datastore: Any  # turbotab.core.datastore.DataStore
    artifact: Callable[[str], Any]  # stage name -> its fresh artifact (whole Bundle), or None
    training_row_ids: Any | None  # numpy array once the split exists, else None
    cohort_row_ids: Any | None  # numpy array once the cohort exists, else None
    settings: dict[str, Any] = field(default_factory=dict)
    sample_size: int = 5000

    before: Callable[[], Any] | None = None  # the server supplies: the sampled working frame

    def before_frame(self) -> Any:
        """The sampled working frame the previewed decision would act on (server-supplied)."""
        if self.before is None:
            raise RuntimeError("this PreviewContext was built without a working frame")
        return self.before()

    def sample_row_ids(self, pool: Any | None = None, n: int | None = None, seed: int = 0) -> Any:
        """A deterministic sample of ``pool`` (default: training rows, else cohort, else all)."""
        import numpy as np

        if pool is None:
            pool = self.training_row_ids if self.training_row_ids is not None else self.cohort_row_ids
        if pool is None:
            pool = np.arange(self.datastore.n_rows)
        pool = np.asarray(pool)
        n = self.sample_size if n is None else n
        if len(pool) <= n:
            return pool
        return np.sort(np.random.default_rng(seed).choice(pool, size=n, replace=False))


Builder = Callable[[Any, PreviewContext], list[Any]]
_BUILDERS: dict[str, list[tuple[int, Builder]]] = {}


def register_consequence(kind: str, builder: Builder, *, priority: int = 0) -> Builder:
    """Register ``builder(decision, ctx) -> [view, ...]`` for decisions of ``kind``.

    Builders for one kind are consulted in ascending ``priority``; their views are concatenated in
    that order and capped at :data:`MAX_VIEWS`, so a builder returns its views most relevant first.
    """
    _BUILDERS.setdefault(kind, []).append((priority, builder))
    _BUILDERS[kind].sort(key=lambda item: item[0])
    return builder


Transform = Callable[[Any, Any, PreviewContext], Any]
_TRANSFORMS: dict[str, Transform] = {}


def register_transform(kind: str, transform: Transform) -> Transform:
    """Register how decisions of ``kind`` change a sample of the working data.

    ``transform(decision, before, ctx) -> after`` takes the sampled frame (indexed by row id, the
    columns the current state would feed the models) and returns the frame the decision would
    produce: rows may be dropped, columns added, removed, renamed or changed. Fitting inside a
    transform uses ``before`` only — which is training rows once the split exists. ``before`` is
    shared between requests (the server caches it): build ``after`` from a copy, never in place.
    """
    _TRANSFORMS[kind] = transform
    return transform


def diff_views(before: Any, after: Any, ctx: PreviewContext) -> list[Any]:
    """The generic diff: compare two frames and return the views that show what changed most.

    Ranked by measured change, capped at :data:`MAX_VIEWS`:
      * rows dropped            -> a ``row_flow`` view (before / after counts with the drop);
      * columns added/removed   -> a ``lineage`` view (kept, changed, new, dropped columns);
      * values changed          -> a ``distribution`` view of the column whose distribution moved
                                   most (standardized Wasserstein distance), and a ``table_focus``
                                   of the changed cells over ≤ 12 affected columns;
      * relationships changed   -> a ``relationship`` view of the pair whose correlation moved most
                                   (only when that change is itself large, |Δr| ≥ 0.2).
    Owner: the M1 "rows" agent (M1_CONTRACT.md §8). Must stay fast on 20,000-column frames:
    compare only columns whose values actually differ.

    Scores, so views of different kinds can be ranked together: rows → the share of rows dropped;
    columns → a quarter plus the share of columns added or removed; values → the standardized
    shift of the most-moved column (capped at 1), with its table a step behind; relationships →
    |Δr|. Views scoring zero are left out.
    """
    import numpy as np

    scored: list[tuple[float, int, Any]] = []
    n_before, n_after = len(before), len(after)
    gone = before.index.difference(after.index)
    if len(gone):
        scored.append((len(gone) / max(1, n_before), 0, _rows_view(n_before, n_after, len(gone))))

    before_cols, after_cols = list(before.columns), list(after.columns)
    bset, aset = set(before_cols), set(after_cols)
    added = [c for c in after_cols if c not in bset]
    removed = [c for c in before_cols if c not in aset]
    common = [c for c in before_cols if c in aset]
    rows = before.index.intersection(after.index, sort=False)
    same_rows = len(rows) == n_before and len(rows) == n_after and before.index.equals(after.index)
    b = before if same_rows else before.loc[rows]
    a = after if same_rows else after.loc[rows]
    changed = _changed_columns(b, a, common)

    if added or removed:
        share = (len(added) + len(removed)) / max(1, len(before_cols))
        scored.append((min(1.0, 0.25 + share), 1, _columns_view(before_cols, after_cols, added, removed, changed)))

    if changed:
        numeric = [c for c in changed if _is_number(b[c]) and _is_number(a[c])]
        shifts = _shifts(b, a, numeric)
        if shifts:
            top = max(shifts, key=lambda c: (shifts[c], -numeric.index(c)))
            score = min(1.0, shifts[top])
            if score > 0:
                scored.append((score, 2, _distribution_view(b[top], a[top], top, shifts[top])))
        ranked = sorted(changed, key=lambda c: -shifts.get(c, 0.0)) if shifts else list(changed)
        focus = _table_view(b, a, ranked)
        if focus is not None:
            scored.append((0.9 * min(1.0, max(shifts.values(), default=0.5)), 3, focus))
        pair = _relationship_change(b, a, [c for c in ranked if c in shifts][:3], changed)
        if pair is not None:
            scored.append((pair[0], 4, pair[1]))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [view for score, _, view in scored if score > 0 and np.isfinite(score)][:MAX_VIEWS]


# ── the generic diff's pieces ────────────────────────────────────────────────

MAX_FOCUS_COLUMNS = 12
MAX_FOCUS_ROWS = 8
MAX_POINTS = 800
RANK_COLUMNS = 500  # at most this many changed columns are ranked by shift
DIFF_CHUNK = 2_000  # columns compared at once
RELATIONSHIP_DELTA = 0.2


def clip_words(text: str, limit: int) -> str:
    """``text`` cut to ``limit`` words, keeping backticks balanced. A safety net, not a style."""
    parts = text.split()
    if len(parts) <= limit:
        return text
    out = " ".join(parts[:limit]).rstrip(".,;:") + "…"
    return out + "`" if out.count("`") % 2 else out


def fmt_count(n: int) -> str:
    return f"`{int(n):,}`"


def fmt_value(x: float) -> str:
    import math

    if x is None or not math.isfinite(float(x)):
        return "`–`"
    x = float(x)
    if x.is_integer() and abs(x) < 1e15:
        return f"`{int(x):,}`"
    return f"`{x:.3g}`" if abs(x) < 1000 else f"`{x:,.0f}`"


def _is_number(series: Any) -> bool:
    return series.dtype.kind in "biuf"


def _changed_columns(before: Any, after: Any, columns: list[str]) -> list[str]:
    """Columns (in ``columns``) whose values differ between two row-aligned frames.

    Numbers are compared a block of columns at a time as arrays (NaN equals NaN); everything
    else column by column. The frames are never copied whole.
    """
    import numpy as np

    kinds_b = before.dtypes
    kinds_a = after.dtypes
    numeric = [c for c in columns if kinds_b[c].kind in "biuf" and kinds_a[c].kind in "biuf"]
    numset = set(numeric)
    changed: set[str] = set()
    for start in range(0, len(numeric), DIFF_CHUNK):
        chunk = numeric[start:start + DIFF_CHUNK]
        x = before[chunk].to_numpy(dtype=float, na_value=np.nan)
        y = after[chunk].to_numpy(dtype=float, na_value=np.nan)
        with np.errstate(invalid="ignore"):
            same = (x == y) | (np.isnan(x) & np.isnan(y))
        differs = ~same.all(axis=0)
        changed.update(c for c, d in zip(chunk, differs) if d)
    for c in columns:
        if c in numset:
            continue
        left, right = before[c], after[c]
        if left.dtype != right.dtype:
            left, right = left.astype(object), right.astype(object)
        if not left.reset_index(drop=True).equals(right.reset_index(drop=True)):
            changed.add(c)
    return [c for c in columns if c in changed]


def _shifts(before: Any, after: Any, columns: list[str]) -> dict[str, float]:
    """Standardized Wasserstein distance per column: W1(before, after) / sd(before)."""
    import numpy as np

    if len(columns) > RANK_COLUMNS:  # a transform that touches everything: rank a fixed subset
        step = len(columns) / RANK_COLUMNS
        columns = [columns[int(i * step)] for i in range(RANK_COLUMNS)]
    grid = np.linspace(0.0, 1.0, 101)
    out: dict[str, float] = {}
    for c in columns:
        x = before[c].to_numpy(dtype=float, na_value=np.nan)
        y = after[c].to_numpy(dtype=float, na_value=np.nan)
        x, y = x[np.isfinite(x)], y[np.isfinite(y)]
        if not len(x) or not len(y):
            out[c] = 1.0 if len(x) != len(y) else 0.0
            continue
        w = float(np.mean(np.abs(np.quantile(x, grid) - np.quantile(y, grid))))
        sd = float(np.std(x)) or float(np.std(y)) or (abs(float(np.mean(x))) or 1.0)
        out[c] = w / sd
    return out


def _histogram_pair(x: Any, y: Any, bins: int = 30) -> tuple[HistogramData, HistogramData]:
    """Two histograms on shared edges, so their bars line up."""
    import numpy as np

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    fx, fy = x[np.isfinite(x)], y[np.isfinite(y)]
    both = np.concatenate([fx, fy])
    if not len(both):
        empty = HistogramData(edges=[], counts=[], n_missing=0)
        return (empty.model_copy(update={"n_missing": int(len(x))}),
                empty.model_copy(update={"n_missing": int(len(y))}))
    lo, hi = float(both.min()), float(both.max())
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5
    edges = np.linspace(lo, hi, bins + 1)
    cx, _ = np.histogram(fx, edges)
    cy, _ = np.histogram(fy, edges)
    return (
        HistogramData(edges=edges.tolist(), counts=cx.astype(int).tolist(), n_missing=int(len(x) - len(fx))),
        HistogramData(edges=edges.tolist(), counts=cy.astype(int).tolist(), n_missing=int(len(y) - len(fy))),
    )


def _rows_view(n_before: int, n_after: int, dropped: int) -> RowFlowView:
    start = RowStep(key="before", label="Rows now", n=n_before)
    return RowFlowView(
        title="Rows this choice removes",
        caption=clip_words(f"{fmt_count(dropped)} of {fmt_count(n_before)} rows would leave; "
                           f"{fmt_count(n_after)} remain.", CAPTION_WORDS),
        emphasis=["after"],
        before=[start],
        after=[start, RowStep(key="after", label="Rows with this choice", n=n_after, dropped=dropped)],
    )


def _names(columns: list[str], limit: int = 3) -> str:
    shown = ", ".join(f"`{c}`" for c in columns[:limit])
    more = len(columns) - limit
    return shown + (f" and {more:,} more" if more > 0 else "")


def lineage_of(
    lanes: list[tuple[str, str, str | None]],
    links: list[tuple[str, str, str]],
    *,
    touched: set[str],
    roles: dict[str, str] | None = None,
    group_of: Callable[[str], str] | None = None,
    max_nodes: int = MAX_FOCUS_COLUMNS,
) -> Lineage:
    """A Lineage from ``(lane, column, formula)`` nodes and ``(source, target, operation)`` links.

    When there are more than ``max_nodes`` columns per lane, only ``touched`` columns (≤ 12)
    keep their own node; the rest collapse into one count node per (lane, group, operation), and
    their links collapse with them.
    """
    roles = roles or {}
    group_of = group_of or (lambda c: roles.get(c) or "other")
    per_lane: dict[str, int] = {}
    for lane, _, _ in lanes:
        per_lane[lane] = per_lane.get(lane, 0) + 1
    collapse = any(n > max_nodes for n in per_lane.values())
    keep = set(list(touched)[:max_nodes]) if collapse else None
    op_of: dict[tuple[str, str], str] = {}
    for src, dst, op in links:
        op_of[("src", src)] = op
        op_of[("dst", dst)] = op
    nodes: dict[str, LineageNode] = {}
    node_of: dict[tuple[str, str], str] = {}
    for lane, column, formula in lanes:
        if keep is None or column in keep:
            nid = f"{lane}:{column}"
            nodes[nid] = LineageNode(id=nid, column=column, lane=lane, role=roles.get(column),
                                     label=column, formula=formula)
        else:
            group = group_of(column)
            op = op_of.get(("src", f"{lane}:{column}")) or op_of.get(("dst", f"{lane}:{column}")) or ""
            nid = f"{lane}:group:{group}:{op}"
            node = nodes.get(nid)
            if node is None:
                nodes[nid] = LineageNode(id=nid, column=None, lane=lane, role=roles.get(column),
                                         label=group, group=group, count=1)
            else:
                nodes[nid] = node.model_copy(update={"count": node.count + 1})
        node_of[(lane, column)] = nid
    for node_id, node in list(nodes.items()):
        if node.column is None:
            nodes[node_id] = node.model_copy(update={"label": f"{node.count:,} {node.group} columns"
                                                     if node.count != 1 else f"1 {node.group} column"})
    seen: set[tuple[str, str, str]] = set()
    out_links: list[LineageLink] = []
    for src, dst, op in links:
        s_lane, s_col = src.split(":", 1)
        d_lane, d_col = dst.split(":", 1)
        key = (node_of.get((s_lane, s_col), src), node_of.get((d_lane, d_col), dst), op)
        if key in seen:
            continue
        seen.add(key)
        out_links.append(LineageLink(source=key[0], target=key[1], operation=op))
    return Lineage(nodes=list(nodes.values()), links=out_links, collapsed=collapse)


def _columns_view(before_cols: list[str], after_cols: list[str], added: list[str],
                  removed: list[str], changed: list[str]) -> LineageView:
    touched = set(added) | set(removed) | set(changed)
    status = {**{c: "kept" for c in before_cols}, **{c: "changed" for c in changed},
              **{c: "dropped" for c in removed}, **{c: "new" for c in added}}
    group_of = status.get  # collapsed groups say what happened to their columns
    before_lineage = lineage_of(
        [("raw", c, None) for c in before_cols],
        [], touched=touched, group_of=lambda c: group_of(c) or "kept")
    after_lineage = lineage_of(
        [("raw", c, None) for c in before_cols] + [("matrix", c, None) for c in after_cols],
        [(f"raw:{c}", f"matrix:{c}", "changed" if c in set(changed) else "kept")
         for c in before_cols if c in set(after_cols)],
        touched=touched, group_of=lambda c: group_of(c) or "kept")
    parts = []
    if added:
        parts.append(f"{fmt_count(len(added))} added ({_names(added, 2)})")
    if removed:
        parts.append(f"{fmt_count(len(removed))} removed ({_names(removed, 2)})")
    caption = "Columns: " + "; ".join(parts) + "."
    return LineageView(
        title="Columns in and out",
        caption=clip_words(caption, CAPTION_WORDS),
        emphasis=(added + removed)[:MAX_FOCUS_COLUMNS],
        before=before_lineage,
        after=after_lineage,
    )


def _distribution_view(x: Any, y: Any, column: str, shift: float) -> DistributionView:
    import numpy as np

    before_h, after_h = _histogram_pair(x, y)
    xv = x.to_numpy(dtype=float, na_value=np.nan)
    yv = y.to_numpy(dtype=float, na_value=np.nan)
    with np.errstate(invalid="ignore"):
        moved = int((~((xv == yv) | (np.isnan(xv) & np.isnan(yv)))).sum())
    caption = (f"`{column}` moves {fmt_value(round(shift, 2))} standard deviations; "
               f"{fmt_count(moved)} of {fmt_count(len(xv))} values change.")
    return DistributionView(
        title=clip_words(f"How `{column}` shifts", TITLE_WORDS),
        caption=clip_words(caption, CAPTION_WORDS),
        emphasis=[column],
        column=column,
        before=before_h,
        after=after_h,
        before_label="Now",
        after_label="With this choice",
    )


def _json(value: Any) -> Any:
    from turbotab.core.datastore import json_safe

    return json_safe(value)


def _table_view(before: Any, after: Any, ranked: list[str]) -> TableFocusView | None:
    import numpy as np
    import pandas as pd

    columns = ranked[:MAX_FOCUS_COLUMNS]
    if not columns:
        return None
    b = before[columns]
    a = after[columns]
    diff = pd.DataFrame(False, index=b.index, columns=columns)
    for c in columns:
        left, right = b[c], a[c]
        if _is_number(left) and _is_number(right):
            x = left.to_numpy(dtype=float, na_value=np.nan)
            y = right.to_numpy(dtype=float, na_value=np.nan)
            with np.errstate(invalid="ignore"):
                diff[c] = ~((x == y) | (np.isnan(x) & np.isnan(y)))
        else:
            diff[c] = ~((left.astype(object) == right.astype(object)) | (left.isna() & right.isna()))
    per_row = diff.sum(axis=1)
    n_cells = int(per_row.sum())
    if not n_cells:
        return None
    top = per_row[per_row > 0].sort_values(ascending=False, kind="stable").index[:MAX_FOCUS_ROWS]
    top = [top[i] for i in np.argsort(b.index.get_indexer(top), kind="stable")]  # table order
    rows = [
        TableRow(
            row_id=int(r),
            before={c: _json(b.at[r, c]) for c in columns},
            after={c: _json(a.at[r, c]) for c in columns},
        )
        for r in top
    ]
    cells = [(int(r), c) for r in top for c in columns if bool(diff.at[r, c])]
    caption = (f"{fmt_count(n_cells)} cells change in {fmt_count(len(ranked))} "
               f"column{'s' if len(ranked) != 1 else ''}; the rows with most changes are shown.")
    return TableFocusView(
        title="The cells that change",
        caption=clip_words(caption, CAPTION_WORDS),
        emphasis=columns,
        columns_before=columns,
        columns_after=columns,
        rows=rows,
        changed=cells,
        n_affected_columns=len(ranked),
    )


def _centered(m: Any) -> tuple[Any, Any]:
    """Columns centered (NaN read as the mean, i.e. 0 after centering) and their norms."""
    import numpy as np

    with np.errstate(invalid="ignore", divide="ignore"):
        if np.isfinite(m).all():
            m = m - m.mean(axis=0)
            return m, np.sqrt((m * m).sum(axis=0))
        m = np.where(np.isfinite(m), m, np.nan)
        m = m - np.nanmean(m, axis=0)
        norms = np.sqrt(np.nansum(m * m, axis=0))
        return np.nan_to_num(m), norms


def _corr_block(frame: Any, targets: Any, numeric: list[str]) -> Any:
    """Pearson r of each column of ``targets`` (n x k) with each column in ``numeric`` (p x k)."""
    import numpy as np

    zt, zn = _centered(np.asarray(targets, dtype=float))
    out = np.zeros((len(numeric), zt.shape[1]))
    for start in range(0, len(numeric), DIFF_CHUNK):
        chunk = numeric[start:start + DIFF_CHUNK]
        m, norms = _centered(frame[chunk].to_numpy(dtype=float, na_value=np.nan))
        with np.errstate(invalid="ignore", divide="ignore"):
            r = (m.T @ zt) / np.outer(norms, zn)
        out[start:start + len(chunk)] = np.nan_to_num(r)
    return out


def _pair_r(x: Any, y: Any) -> float | None:
    import numpy as np

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return None
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def _points(x: Any, y: Any) -> list[tuple[float, float]]:
    import numpy as np

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.flatnonzero(np.isfinite(x) & np.isfinite(y))
    if len(ok) > MAX_POINTS:
        ok = np.sort(np.random.default_rng(0).choice(ok, size=MAX_POINTS, replace=False))
    return [(float(x[i]), float(y[i])) for i in ok]


def _relationship_change(before: Any, after: Any, candidates: list[str],
                         changed: list[str]) -> tuple[float, RelationshipView] | None:
    """The pair whose correlation moved most, when |Δr| ≥ 0.2."""
    import numpy as np

    if not candidates:
        return None
    kinds_b, kinds_a = before.dtypes, after.dtypes
    numeric = [c for c in before.columns if c in kinds_a.index
               and kinds_b[c].kind in "biuf" and kinds_a[c].kind in "biuf"]
    if len(numeric) < 2:
        return None
    # One pass over the unchanged table: each candidate's old and new values against every
    # column as it was. Partners that changed too are then corrected with their new values.
    olds = before[candidates].to_numpy(dtype=float, na_value=np.nan)
    news = after[candidates].to_numpy(dtype=float, na_value=np.nan)
    r = _corr_block(before, np.hstack([olds, news]), numeric)
    k = len(candidates)
    r_old, r_new = r[:, :k], r[:, k:]
    moved = [i for i, c in enumerate(numeric) if c in set(changed)]
    if moved:
        r_new[moved] = _corr_block(after, news, [numeric[i] for i in moved])
    delta = np.abs(r_new - r_old)
    position = {c: i for i, c in enumerate(numeric)}
    for j, c in enumerate(candidates):
        if c in position:
            delta[position[c], j] = 0.0
    i, j = np.unravel_index(int(np.argmax(delta)), delta.shape)
    best: tuple[float, str, str] | None = (float(delta[i, j]), candidates[j], numeric[i])
    if best is None or best[0] < RELATIONSHIP_DELTA:
        return None
    _, column, other = best
    xb, yb = before[other].to_numpy(dtype=float, na_value=np.nan), before[column].to_numpy(dtype=float, na_value=np.nan)
    xa, ya = after[other].to_numpy(dtype=float, na_value=np.nan), after[column].to_numpy(dtype=float, na_value=np.nan)
    r_before, r_after = _pair_r(xb, yb), _pair_r(xa, ya)
    if r_before is None or r_after is None or abs(r_after - r_before) < RELATIONSHIP_DELTA:
        return None
    caption = (f"`{column}` and `{other}` correlate {fmt_value(round(r_before, 2))} now, "
               f"{fmt_value(round(r_after, 2))} with this choice.")
    view = RelationshipView(
        title=clip_words(f"How `{column}` relates to `{other}`", TITLE_WORDS),
        caption=clip_words(caption, CAPTION_WORDS),
        emphasis=[column, other],
        x_label=other,
        y_label_before=column,
        y_label_after=f"{column} with this choice",
        points_before=_points(xb, yb),
        points_after=_points(xa, ya),
        r_before=r_before,
        r_after=r_after,
    )
    return abs(r_after - r_before), view


def plan(decision: Any, ctx: PreviewContext, basis: str) -> PreviewResult:
    """Domain builders first (they know what teaches), then the generic diff fills the rest."""
    views: list[Any] = []
    for _, builder in _BUILDERS.get(decision.kind, []):
        views.extend(builder(decision, ctx))
        if len(views) >= MAX_VIEWS:
            break
    transform = _TRANSFORMS.get(decision.kind)
    if len(views) < MAX_VIEWS and transform is not None:
        before = ctx.before_frame()
        after = transform(decision, before, ctx)
        taken = {v.kind for v in views}
        views.extend(v for v in diff_views(before, after, ctx) if v.kind not in taken)
    note = None if views else "Nothing about this choice can be shown on your data yet."
    return PreviewResult(kind=decision.kind, views=views[:MAX_VIEWS], basis=basis, note=note)


def words(text: str) -> int:
    return len(text.split())


__all__ = [
    "CAPTION_WORDS", "FRAME_WORDS", "MAX_VIEWS", "TITLE_WORDS", "ConsequenceView",
    "DistributionFrame", "DistributionView", "FitLine", "FrameRow", "HistogramData", "Lineage",
    "LineageFrame", "LineageLink", "LineageNode", "LineageView", "Mark", "PreviewContext",
    "PreviewResult", "RelationshipFrame", "RelationshipView", "RowFlowFrame", "RowFlowView",
    "RowStep", "TableFocusView", "TableFrame", "TableRow",
    "clip_words", "diff_views", "fmt_count", "fmt_value", "lineage_of", "plan",
    "register_consequence", "register_transform", "words",
]

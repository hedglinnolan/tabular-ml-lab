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


# ── the views ────────────────────────────────────────────────────────────────


class _View(_Model):
    title: str
    caption: str
    emphasis: list[str] = Field(default_factory=list)  # columns or step keys to highlight


class RowFlowView(_View):
    kind: Literal["row_flow"] = "row_flow"
    before: list[RowStep]
    after: list[RowStep]


class LineageView(_View):
    kind: Literal["lineage"] = "lineage"
    before: Lineage | None
    after: Lineage


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
    changed: list[tuple[int, str]]  # (row_id, column) cells whose value changes
    n_affected_columns: int


class DistributionView(_View):
    kind: Literal["distribution"] = "distribution"
    column: str
    before: HistogramData
    after: HistogramData
    before_label: str
    after_label: str


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
    transform uses ``before`` only — which is training rows once the split exists.
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
    """
    raise NotImplementedError("M1: the generic diff is not built yet")


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
    "CAPTION_WORDS", "MAX_VIEWS", "TITLE_WORDS", "ConsequenceView", "DistributionView",
    "HistogramData", "Lineage", "LineageLink", "LineageNode", "LineageView", "PreviewContext",
    "PreviewResult", "RelationshipView", "RowFlowView", "RowStep", "TableFocusView", "TableRow",
    "diff_views", "plan", "register_consequence", "register_transform", "words",
]

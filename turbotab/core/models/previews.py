"""Consequence previews for the modeling agent's decision kinds (M1_CONTRACT §4).

``set_energy_adjustment``: the nutrient most correlated with energy, plotted against energy
before and after the method (the picture *is* the method), then the lineage of the model matrix,
then the nutrient's distribution before and after.

``select_models``: one lineage view per chosen family, newly added families first, each traced
from the family's own pipeline — so a family registered later previews itself with no code here.

Both fit only on a sample of ``ctx.training_row_ids``; before the split exists they show nothing.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from turbotab.core.consequences import (
    CAPTION_WORDS,
    MAX_VIEWS,
    TITLE_WORDS,
    DistributionView,
    HistogramData,
    LineageView,
    PreviewContext,
    RelationshipView,
    register_consequence,
)

POINTS = 800
BINS = 30
STEP_PHRASES: dict[str, str] = {"scale": "each standardized on training rows"}


def register_step_phrase(step: str, phrase: str) -> None:
    """How a select_models preview describes a family-specific step, e.g. ``"spline"``."""
    STEP_PHRASES[step] = phrase


# ── text helpers ─────────────────────────────────────────────────────────────


def fit_words(text: str, budget: int) -> str:
    """``text`` if it is within ``budget`` words, else its first clause that is, else a cut."""
    words = text.split()
    if len(words) <= budget:
        return text
    for mark in (";", ":", ","):
        head = text.split(mark)[0].strip()
        if head and len(head.split()) <= budget:
            return head.rstrip(".") + "."
    return " ".join(words[: budget - 1]) + " …"


def _num(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return np.format_float_positional(float(value), precision=3, unique=False, fractional=False,
                                      trim="-")


def _r(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _names(columns: Sequence[str], limit: int = 2) -> str:
    shown = [f"`{c}`" for c in columns[:limit]]
    rest = len(columns) - limit
    if rest > 0:
        return f"{', '.join(shown)} and {rest} more"
    return " and ".join(shown) if len(shown) == 2 else shown[0]


def _corr(x: np.ndarray, y: np.ndarray) -> float | None:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return None
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def _histogram(values: np.ndarray) -> HistogramData:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return HistogramData(edges=[], counts=[], n_missing=int(values.size))
    counts, edges = np.histogram(finite, bins=BINS)
    return HistogramData(edges=[float(e) for e in edges], counts=[int(c) for c in counts],
                         n_missing=int(values.size - finite.size))


def _points(x: np.ndarray, y: np.ndarray, keep: np.ndarray) -> list[tuple[float, float]]:
    return [(float(a), float(b)) for a, b in zip(x[keep], y[keep]) if np.isfinite(a) and np.isfinite(b)]


# ── shared reading ───────────────────────────────────────────────────────────


def _energy_reading(decision: Any, ctx: PreviewContext) -> tuple[str | None, list[str]]:
    """The energy column and nutrients the picture is about, for any option including "none"."""
    from turbotab.core.methods.energy import energy_factor

    state = ctx.state
    current = state.energy_adjustment
    roles = state.roles or {}
    E = decision.energy_column or (current.energy_column if current else None)
    if not E:
        E = next((c for c, r in roles.items() if r == "energy"), None)
    nutrients = list(decision.nutrients) or (list(current.nutrients) if current else [])
    if not nutrients:
        nutrients = [c for c, r in roles.items()
                     if r == "exposure" and c != E and energy_factor(c).factor is not None]
    return E, nutrients


def _read(ctx: PreviewContext, columns: Sequence[str]) -> Any:
    from turbotab.core.models.pipeline import modeling_frame

    ids = ctx.sample_row_ids(ctx.training_row_ids)
    available = set(ctx.datastore.columns)
    return modeling_frame(ctx.datastore, [c for c in dict.fromkeys(columns) if c in available], ids)


def _lineage(state: Any, frame: Any, predictors: Sequence[str], adjustment: Any) -> Any:
    from turbotab.core.models.lineage import missing_counts, trace
    from turbotab.core.models.pipeline import design_spec, shared_steps, transformer

    spec = design_spec(state, frame, predictors, energy=adjustment)
    fitted = transformer(shared_steps(spec)).fit(frame[spec.inputs])
    return trace(fitted.steps, spec.inputs, spec.roles, missing_counts(frame[spec.inputs]))


def _matrix(lineage: Any) -> list[str]:
    return [n.column for n in lineage.nodes if n.lane == "matrix" and n.column is not None]


def lineage_caption(before: Sequence[str], after: Sequence[str]) -> str:
    leave = [c for c in before if c not in after]
    arrive = [c for c in after if c not in before]
    parts = []
    if leave:
        parts.append(f"{_names(leave)} {'leaves' if len(leave) == 1 else 'leave'}")
    if arrive:
        parts.append(f"{_names(arrive)} {'arrives' if len(arrive) == 1 else 'arrive'}")
    n = len(after)
    columns = f"{n} column{'s' if n != 1 else ''}"
    if not parts:
        return f"The model matrix keeps the same {columns}."
    return fit_words(f"{'; '.join(parts)}. The model sees {columns}.", CAPTION_WORDS)


# ── set_energy_adjustment ────────────────────────────────────────────────────


def _relationship_caption(method: str, n: str, E: str, out: str, r0: float | None,
                          r1: float | None, strata: str | None) -> str:
    if method == "none":
        text = f"`{n}` enters unadjusted; it correlates {_r(r0)} with `{E}`, so energy confounds it."
    elif method == "standard":
        text = f"`{n}` stays as recorded (r = {_r(r0)} with `{E}`); `{E}` enters the model beside it."
    elif method == "residual":
        within = f" within `{strata}`" if strata else ""
        text = (f"`{n}` correlates {_r(r0)} with `{E}`; after residual adjustment{within}, "
                f"{_r(r1)}.")
    elif method in ("density", "density_multivariate"):
        stays = "stays as its own term" if method == "density_multivariate" else "leaves the model"
        text = f"`{out}` correlates {_r(r1)} with `{E}`, down from {_r(r0)}; `{E}` {stays}."
    else:
        text = f"`{n}` becomes kcal in `{out}`; `{E}` leaves, split into nutrient and other kcal."
    return fit_words(text, CAPTION_WORDS)


def energy_adjustment_preview(decision: Any, ctx: PreviewContext) -> list[Any]:
    from turbotab.core.decisions import EnergyAdjustment
    from turbotab.core.models.pipeline import input_columns, predictors_from_roles
    from turbotab.core.models.steps import energy_step

    if ctx.training_row_ids is None:
        return []
    state = ctx.state
    E, nutrients = _energy_reading(decision, ctx)
    if not E or not nutrients:
        return []
    after_adj = EnergyAdjustment(**decision.model_dump(exclude={"kind"}))
    before_adj = state.energy_adjustment
    predictors = predictors_from_roles(state.roles, state.target)
    predictors = predictors + [c for c in (E, *nutrients) if c not in predictors]
    frame = _read(ctx, [*input_columns(predictors, after_adj), *input_columns(predictors, before_adj)])
    if E not in frame.columns:
        return []
    nutrients = [n for n in nutrients if n in frame.columns]
    e = frame[E].to_numpy(dtype=float, na_value=np.nan)
    rs = {n: _corr(frame[n].to_numpy(dtype=float, na_value=np.nan), e) for n in nutrients}
    ranked = sorted((n for n in nutrients if rs[n] is not None), key=lambda n: -abs(rs[n]))
    if not ranked:
        return []
    n = ranked[0]
    raw = frame[n].to_numpy(dtype=float, na_value=np.nan)

    out_name, after = n, raw
    problem = None
    step = energy_step(after_adj, predictors) if after_adj.method != "none" else None
    if step is not None:
        inputs = input_columns(predictors, after_adj)
        try:
            fitted = step.fit(frame[inputs])
            entry = next(x for x in fitted.lineage() if x["inputs"][0] == n
                         and x["operation"] != "partition-other")
            out_name = str(entry["output"])
            after = fitted.transform(frame[inputs])[out_name].to_numpy(dtype=float, na_value=np.nan)
        except (ValueError, TypeError) as exc:
            problem = str(getattr(exc, "reason", None) or exc)
    r0 = rs[n]
    r1 = None if problem else _corr(after, e)
    keep = np.zeros(len(e), dtype=bool)
    keep[np.random.default_rng(0).permutation(len(e))[:POINTS]] = True
    method = after_adj.method
    caption = (fit_words(problem, CAPTION_WORDS) if problem else
               _relationship_caption(method, n, E, out_name, r0, r1, after_adj.strata))
    views: list[Any] = [RelationshipView(
        title=fit_words(f"{n} against {E}, before and after", TITLE_WORDS),
        caption=caption,
        emphasis=[n, E],
        x_label=E,
        y_label_before=n,
        y_label_after=out_name,
        points_before=_points(e, raw, keep),
        points_after=[] if problem else _points(e, after, keep),
        r_before=r0,
        r_after=r1,
    )]
    if problem:
        return views

    lineage_before = lineage_after = None
    try:
        lineage_after = _lineage(state, frame, predictors, after_adj)
        lineage_before = _lineage(state, frame, predictors, before_adj)
    except (ValueError, TypeError):
        pass  # the recorded method may not run on these rows; the option's own lineage still shows
    if lineage_after is not None:
        cols_after = _matrix(lineage_after)
        cols_before = _matrix(lineage_before) if lineage_before is not None else cols_after
        changed = [c for c in cols_after if c not in cols_before] + [c for c in cols_before
                                                                      if c not in cols_after]
        views.append(LineageView(
            title="Columns the model will see",
            caption=(lineage_caption(cols_before, cols_after) if lineage_before is not None
                     else f"The model will see {len(cols_after)} columns."),
            emphasis=changed[:12],
            before=lineage_before,
            after=lineage_after,
        ))
    if out_name != n or method in ("density", "density_multivariate", "partition"):
        finite0, finite1 = raw[np.isfinite(raw)], after[np.isfinite(after)]
        caption = (f"Mean {_num(finite0.mean())} → {_num(finite1.mean())}; SD {_num(finite0.std())} → "
                   f"{_num(finite1.std())}, over {len(raw):,} training rows.")
        views.append(DistributionView(
            title=fit_words(f"Values of {n}, before and after", TITLE_WORDS),
            caption=fit_words(caption, CAPTION_WORDS),
            emphasis=[n],
            column=n,
            before=_histogram(raw),
            after=_histogram(after),
            before_label=f"{n} as recorded",
            after_label=out_name,
        ))
    return views[:MAX_VIEWS]


# ── select_models ────────────────────────────────────────────────────────────


def models_preview(decision: Any, ctx: PreviewContext) -> list[Any]:
    from turbotab.core.models import get_family
    from turbotab.core.models.lineage import missing_counts, trace
    from turbotab.core.models.pipeline import (
        design_spec,
        family_steps,
        input_columns,
        predictors_from_roles,
        shared_steps,
        transformer,
    )

    if ctx.training_row_ids is None:
        return []
    state = ctx.state
    predictors = predictors_from_roles(state.roles, state.target)
    if not predictors:
        return []
    frame = _read(ctx, input_columns(predictors, state.energy_adjustment))
    spec = design_spec(state, frame, predictors)
    X = frame[spec.inputs]
    missing = missing_counts(X)
    try:
        shared = transformer(shared_steps(spec)).fit(X)
    except (ValueError, TypeError):
        return []
    base = trace(shared.steps, spec.inputs, spec.roles, missing)
    width = len(_matrix(base)) if not base.collapsed else sum(
        n.count for n in base.nodes if n.lane == "matrix")
    has_missing = bool(X.isna().to_numpy().any())
    current = set(state.models or [])
    ordered = [k for k in decision.models if k not in current] + [k for k in decision.models if k in current]
    shared_names = {name for name, _ in shared.steps}
    views: list[Any] = []
    for key in ordered:
        try:
            family = get_family(key)
        except KeyError:
            continue
        steps = family_steps(spec, family)
        extra = [name for name, _ in steps if name not in shared_names]
        fitted = transformer(steps).fit(X) if extra else shared
        after = trace(fitted.steps, spec.inputs, spec.roles, missing) if extra else base
        phrases = [STEP_PHRASES.get(name, name.replace("_", " ")) for name in extra]
        how = ", ".join(phrases) if phrases else "taken as they are, unscaled"
        tail = ""
        if has_missing and not spec.impute:
            tail = "; missing values allowed" if family.handles_missing else "; rows need complete values"
        views.append(LineageView(
            title=fit_words(f"Model matrix for {family.label.lower()}", TITLE_WORDS),
            caption=fit_words(f"{family.label}: {width} columns, {how}{tail}.", CAPTION_WORDS),
            emphasis=[],
            before=base,
            after=after,
        ))
        if len(views) >= MAX_VIEWS:
            break
    return views


register_consequence("set_energy_adjustment", energy_adjustment_preview)
register_consequence("select_models", models_preview)

__all__ = ["energy_adjustment_preview", "fit_words", "lineage_caption", "models_preview",
           "register_step_phrase"]

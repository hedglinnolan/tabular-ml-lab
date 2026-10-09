"""Consequence previews for the modeling agent's decision kinds (M1_CONTRACT §4).

``set_energy_adjustment``: the nutrient most correlated with energy, plotted against energy
before and after the method (the picture *is* the method), then the lineage of the model matrix,
then the nutrient's distribution before and after. Each method also tells its own storyboard
(M1_CONTRACT §12.1), the real intermediate states between before and after:

* residual — fit each nutrient on energy (the scatter with its fitted line), then keep what
  energy does not explain (the residuals, centered at 0); the after adds the average back. The
  distribution steps in time with the scatter.
* partition — total energy split into its parts, told on the lineage: each nutrient becomes its
  kcal, then the rest of energy becomes kcal from everything else; the after lets energy go.
* density, standard and none — no intermediate step: before ⇄ after directly.

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
    DistributionFrame,
    DistributionView,
    FitLine,
    HistogramData,
    Lineage,
    LineageFrame,
    LineageLink,
    LineageNode,
    LineageView,
    PreviewContext,
    RelationshipFrame,
    RelationshipView,
    estimates_unseen,
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
    """A correlation to two places; a residual's r of -1.9e-17 is 0.00, never "-0.00"."""
    if value is None:
        return "n/a"
    return "0.00" if abs(value) < 0.005 else f"{value:.2f}"


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
    from turbotab.core.decisions import left_out

    gone = set(left_out(state))  # left out with the missing values: not in the model to adjust
    return (None if E in gone else E), [n for n in nutrients if n not in gone]


def _closest_told(ctx: PreviewContext, candidates: Sequence[str]) -> str | None:
    """The nutrient the proposals name as most correlated with energy, if it is a candidate."""
    proposals = ctx.artifact("proposals")
    data = getattr(proposals, "data", proposals)
    reading = data.get("energy") if isinstance(data, dict) else None
    rs = (reading or {}).get("r_with_energy") or {}
    told = [c for c in sorted(rs, key=lambda c: -abs(float(rs[c]))) if c in candidates]
    return told[0] if told else None


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
                          r1: float | None, strata: str | None,
                          gap: dict[str, Any] | None = None) -> str:
    if method == "none":
        text = f"`{n}` enters unadjusted and `{E}` leaves the model; r = {_r(r0)}, so energy confounds it."
    elif method == "standard":
        text = f"`{n}` stays as recorded (r = {_r(r0)} with `{E}`); `{E}` enters the model beside it."
    elif method == "residual":
        within = f" within `{strata}`" if strata else ""
        text = (f"`{n}` correlates {_r(r0)} with `{E}`; after residual adjustment{within}, "
                f"{_r(r1)}, with `{E}` kept in the model.")
    elif method == "residual_energy_dropped" and gap is not None:
        # The card shows the gap the energy-dropped form opens on these rows (audit ME-03).
        text = (f"`{E}` leaves the outcome model: `{out}` coefficient {_gap_number(gap['dropped'])}, "
                f"against {_gap_number(gap['standard'])} with `{E}` kept.")
    elif method == "residual_energy_dropped":
        text = (f"`{n}` correlates {_r(r0)} with `{E}`; after residual adjustment, {_r(r1)}; "
                f"`{E}` leaves the outcome model.")
    elif method in ("density", "density_multivariate"):
        stays = "stays as its own term" if method == "density_multivariate" else "leaves the model"
        text = f"`{out}` correlates {_r(r1)} with `{E}`, down from {_r(r0)}; `{E}` {stays}."
    elif method == "all_components":
        text = f"`{n}` becomes kcal in `{out}`; every energy source is its own term, the rest other kcal."
    else:
        text = f"`{n}` becomes kcal in `{out}`; `{E}` leaves, split into nutrient and other kcal."
    return fit_words(text, CAPTION_WORDS)


def _gap_number(value: float) -> str:
    """A coefficient on the card: three significant digits, signed, a true minus."""
    return f"{value:+.3g}".replace("-", "−")


def _residual_gap(ctx: PreviewContext, state: Any, frame: Any, predictors: Sequence[str],
                  adjustment: Any, out_name: str) -> dict[str, Any] | None:
    """The energy-dropped residual's coefficient for the pictured nutrient against the standard
    model's, on the preview's training rows: what leaving energy out costs (audit ME-03).

    The outcome is read for the same training rows the picture uses (never a held-out row). None
    when the outcome or task cannot carry it (multiclass, no outcome yet) or a fit fails. The
    caller never asks for it under inference before the analysis plan is locked: both numbers are
    the outcome model's estimates (``consequences.estimates_unseen``).
    """
    from turbotab.core.methods.energy import coefficient_gap
    from turbotab.core.models.pipeline import design_spec, shared_steps, transformer
    from turbotab.core.stages.modeling import coded_outcome

    target = getattr(state, "target", None)
    task = getattr(state, "task", None)
    if task is None:
        info = ctx.artifact("target_info")
        info = getattr(info, "data", info)
        task = (info or {}).get("task") if isinstance(info, dict) else None
    if not target or task not in ("regression", "binary") or target not in ctx.datastore.columns:
        return None
    try:
        y = ctx.datastore.materialize([target], frame.index.to_numpy())[target]
        coded = coded_outcome(task, y.reindex(frame.index).to_numpy(), getattr(state, "event", None))
        mats = []
        for method in ("residual_energy_dropped", "residual"):
            spec = design_spec(state, frame, predictors,
                               energy=adjustment.model_copy(update={"method": method}))
            mats.append(transformer(shared_steps(spec)).fit_transform(frame[spec.inputs]))
        gaps = coefficient_gap(task, coded, mats[0], mats[1], [(out_name, out_name)])
    except (ValueError, TypeError, KeyError):
        return None
    return gaps[0] if gaps else None


# ── storyboards ──────────────────────────────────────────────────────────────


def _residual_story(fitted: Any, n: str, E: str, e: np.ndarray, raw: np.ndarray, after: np.ndarray,
                    keep: np.ndarray, strata: str | None) -> tuple[list[Any], list[Any]]:
    """Residual's two steps, for the scatter and the distribution: the fit, then the residuals.

    The residual is the adjusted value less what the method adds back (the nutrient predicted at
    the mean energy of all fitting rows, one constant even under strata), so it is exactly
    ``N − N̂(E)`` on the fitting rows' own regression, per level under strata. On the log scale
    it is the residual of ``log N`` on ``log E``.
    """
    pooled = getattr(fitted, "pooled_", fitted)
    params = pooled.params_[n]
    log = params["scale"] == "log"
    added = np.full(len(after), float(params["constant_added"]))
    with np.errstate(invalid="ignore", divide="ignore"):
        residual = (np.log(after) if log else after) - added
    line = None
    if not log and strata is None:
        line = FitLine(slope=float(params["slope"]), intercept=float(params["intercept"]))
    if strata is not None:
        fit_label = fit_words(f"Fit {n} on {E} within each {strata}", 8)
    elif log:
        fit_label = fit_words(f"Fit log {n} on log {E}", 8)
    else:
        fit_label = fit_words(f"Fit {n} on {E}", 8)
    keep_label = "Keep what energy does not explain"
    y_resid = f"log {n} residual" if log else f"{n} residual"
    relationship = [
        RelationshipFrame(label=fit_label, points=_points(e, raw, keep), r=_corr(raw, e),
                          fit_line=line),
        RelationshipFrame(label=keep_label, points=_points(e, residual, keep),
                          r=_corr(residual, e), fit_line=FitLine(slope=0.0, intercept=0.0),
                          y_label=y_resid),
    ]
    distribution = [
        DistributionFrame(label=fit_label, hist=_histogram(raw)),
        DistributionFrame(label=keep_label, hist=_histogram(residual), x_label=y_resid),
    ]
    return relationship, distribution


def _without(lineage: Lineage, gone: set[str]) -> tuple[list[LineageNode], list[LineageLink]]:
    nodes = [node for node in lineage.nodes if node.id not in gone]
    links = [link for link in lineage.links if link.source not in gone and link.target not in gone]
    return nodes, links


def _partition_story(after: Lineage, E: str, nutrients: Sequence[str]) -> list[LineageFrame]:
    """Partition's two steps on the lineage: each nutrient becomes its kcal, then the rest of
    energy becomes kcal from everything else. Energy itself stays in the picture until the after
    lets it go."""
    ids = {node.id: node for node in after.nodes}
    others = [nid for nid, node in ids.items() if node.column == "kcal_from_other"]
    if not others or f"raw:{E}" not in ids:
        return []
    still = [LineageNode(id=f"adj:{E}", column=E, lane="adjusted", role="energy", label=E),
             LineageNode(id=f"mx:{E}", column=E, lane="matrix", role="energy", label=E)]
    kept = [LineageLink(source=f"raw:{E}", target=f"adj:{E}", operation="kept"),
            LineageLink(source=f"adj:{E}", target=f"mx:{E}", operation="kept")]
    first_nodes, first_links = _without(after, set(others))
    first = Lineage(nodes=first_nodes + still, links=first_links + kept, collapsed=after.collapsed)
    second = Lineage(nodes=list(after.nodes) + still, links=list(after.links) + kept,
                     collapsed=after.collapsed)
    one = "Each nutrient becomes its kcal" if len(nutrients) != 1 else f"{nutrients[0]} becomes its kcal"
    return [LineageFrame(label=fit_words(one, 8), lineage=first),
            LineageFrame(label=fit_words(f"The rest of {E} becomes other kcal", 8), lineage=second)]


def energy_adjustment_preview(decision: Any, ctx: PreviewContext) -> list[Any]:
    # Ruling 3: under inference the rows are every analyzed row, not training rows.
    rows_word = "analyzed" if getattr(ctx.state, "purpose", None) == "inference" else "training"
    from turbotab.core.decisions import EnergyAdjustment
    from turbotab.core.models.pipeline import input_columns, model_predictors
    from turbotab.core.models.steps import energy_step

    if ctx.training_row_ids is None:
        from turbotab.core.plan_previews import rows_not_ready

        ctx.read.setdefault("note", rows_not_ready(ctx))
        return []
    state = ctx.state
    E, nutrients = _energy_reading(decision, ctx)
    if not E or not nutrients:
        return []
    from turbotab.core.consequences import after_state
    from turbotab.core.plan_previews import asks_first

    # BLUEPRINT §14: where the fit asks for a reading first, the preview offers that ask. The
    # method's own picture (the recorded values of the nutrient and energy) still stands, but the
    # model matrix, which rests on every predictor's reading, is not drawn on a guess.
    asked = asks_first(ctx, after_state(decision, ctx))
    after_adj = EnergyAdjustment(**decision.model_dump(exclude={"kind"}))
    before_adj = state.energy_adjustment
    predictors = model_predictors(state)
    predictors = predictors + [c for c in (E, *nutrients) if c not in predictors]
    frame = _read(ctx, [*input_columns(predictors, after_adj), *input_columns(predictors, before_adj)])
    if E not in frame.columns:
        return []
    nutrients = [n for n in nutrients if n in frame.columns]
    e = frame[E].to_numpy(dtype=float, na_value=np.nan)
    rs = {n: _corr(frame[n].to_numpy(dtype=float, na_value=np.nan), e) for n in nutrients}
    # The nutrient the question names as tracking energy most closely (every row as loaded), so
    # the question, the finding and this picture are about the same column; else the closest here.
    told = _closest_told(ctx, [n for n in nutrients if rs[n] is not None])
    ranked = sorted((n for n in nutrients if rs[n] is not None), key=lambda n: -abs(rs[n]))
    if not ranked:
        return []
    n = told or ranked[0]
    raw = frame[n].to_numpy(dtype=float, na_value=np.nan)

    from turbotab.core.models.pipeline import _energy_factors

    def adjusted(adj: Any) -> tuple[str, np.ndarray, Any, str | None]:
        """The nutrient as ``adj`` leaves it: its output column, values, fitted step, or why not.
        A partition method converts each source by the kcal per unit the readings ledger settled
        (never its name; the routing gate), so an unsettled unit shows why, not a number."""
        step = (energy_step(adj, predictors, factors=_energy_factors(state, adj, frame, None))
                if adj is not None and adj.method != "none" else None)
        if step is None:
            return n, raw, None, None
        inputs = input_columns(predictors, adj)
        try:
            fitted_step = step.fit(frame[inputs])
            entry = next(x for x in fitted_step.lineage() if x["inputs"][0] == n
                         and x["operation"] != "partition-other")
            name = str(entry["output"])
            values = fitted_step.transform(frame[inputs])[name].to_numpy(dtype=float, na_value=np.nan)
            return name, values, fitted_step, None
        except (ValueError, TypeError, StopIteration) as exc:
            return n, raw, None, str(getattr(exc, "reason", None) or exc)

    out_name, after, fitted, problem = adjusted(after_adj)
    # "Your data now" is the data as recorded: once an adjustment is on record, the flip morphs
    # the recorded method's result into this option's (residual ⇄ density), not raw into it.
    now_name, now, _, now_problem = adjusted(before_adj)
    reopened = before_adj is not None and before_adj.method != "none" and now_problem is None
    if not reopened:
        now_name, now = n, raw
    r0 = _corr(now, e) if reopened else rs[n]
    r1 = None if problem else _corr(after, e)
    keep = np.zeros(len(e), dtype=bool)
    keep[np.random.default_rng(0).permutation(len(e))[:POINTS]] = True
    method = after_adj.method
    dropped = method == "residual_energy_dropped"
    # The gap is the outcome model's own coefficients: under inference it waits for the lock, as
    # every estimate does (calm/FOUNDATION §5 rule 6); the option still says energy leaves.
    gap = (_residual_gap(ctx, state, frame, predictors, after_adj, out_name)
           if dropped and not problem and not estimates_unseen(state) else None)
    if problem:
        caption = fit_words(problem, CAPTION_WORDS)
    elif gap is not None:  # the gap is this option's own consequence, whatever is on record now
        caption = _relationship_caption(method, n, E, out_name, r0, r1, after_adj.strata, gap)
    elif reopened:
        leaves = f"; `{E}` leaves the outcome model" if dropped else ""
        caption = fit_words(f"Recorded now: `{now_name}` correlates {_r(r0)} with `{E}`; with this "
                            f"choice, `{out_name}` correlates {_r(r1)}{leaves}.", CAPTION_WORDS)
    else:
        caption = _relationship_caption(method, n, E, out_name, r0, r1, after_adj.strata)
    scatter_story: list[Any] = []
    hist_story: list[Any] = []
    if (not problem and not reopened and method in ("residual", "residual_energy_dropped")
            and fitted is not None):
        strata = after_adj.strata if after_adj.strata in frame.columns else None
        scatter_story, hist_story = _residual_story(fitted, n, E, e, raw, after, keep, strata)
    views: list[Any] = [RelationshipView(
        title=fit_words(f"{n} against {E}, now and with this choice" if reopened else
                        f"{n} against {E}, before and after", TITLE_WORDS),
        caption=caption,
        emphasis=[n, E],
        x_label=E,
        y_label_before=now_name,
        y_label_after=out_name,
        points_before=_points(e, now, keep),
        points_after=[] if problem else _points(e, after, keep),
        r_before=r0,
        r_after=r1,
        story=scatter_story,
    )]
    if problem:
        return views

    lineage_before = lineage_after = None
    try:
        if asked is None:
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
            story=_partition_story(lineage_after, E, nutrients) if method == "partition" else [],
        ))
    if out_name != now_name or method in ("density", "density_multivariate", "partition"):
        finite0, finite1 = now[np.isfinite(now)], after[np.isfinite(after)]
        caption = (f"Mean {_num(finite0.mean())} → {_num(finite1.mean())}; SD {_num(finite0.std())} → "
                   f"{_num(finite1.std())}, on the sampled {rows_word} rows.")
        views.append(DistributionView(
            title=fit_words(f"Values of {n}, before and after", TITLE_WORDS),
            caption=fit_words(caption, CAPTION_WORDS),
            emphasis=[n],
            column=n,
            before=_histogram(now),
            after=_histogram(after),
            before_label=now_name if reopened else f"{n} as recorded",
            after_label=out_name,
            story=hist_story,
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
        model_predictors,
        shared_steps,
        transformer,
    )

    if ctx.training_row_ids is None:
        from turbotab.core.plan_previews import rows_not_ready

        ctx.read.setdefault("note", rows_not_ready(ctx))
        return []
    state = ctx.state
    predictors = model_predictors(state)
    if not predictors:
        return []
    from turbotab.core.consequences import after_state
    from turbotab.core.plan_previews import asks_first

    # BLUEPRINT §14: the fit asks for a reading its predictors rest on before it builds any matrix,
    # so the preview offers that ask rather than a matrix built on a guess.
    if asks_first(ctx, after_state(decision, ctx)) is not None:
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
        if getattr(state, "purpose", None) == "inference":
            phrases = [ph.replace("on training rows", "on every analyzed row") for ph in phrases]
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
    _say_the_cost(decision, ctx, int(len(ctx.training_row_ids)), len(predictors))
    from turbotab.core.plan_previews import population_block

    # Under the surveyed population a family with no design-based estimator is blocked and
    # recorded (MODELING_SEQUENCE §4): the preview says so with the fit's own words and exits.
    population_block(ctx, after_state(decision, ctx), decision)
    return views


def _say_the_cost(decision: Any, ctx: PreviewContext, n_rows: int, n_columns: int) -> None:
    """The note says what fitting the chosen families will take, when it is long enough to matter
    (M2_CONTRACT §12.6): the shelf's measured estimates, summed, and the family that takes most."""
    from turbotab.core.models.cost import NOTEWORTHY_SECONDS, say

    shelf = ctx.artifact("shelf")
    shelf = getattr(shelf, "data", shelf)
    if not isinstance(shelf, dict):
        return
    timed = {f["key"]: (f.get("estimate_seconds"), f.get("label")) for f in shelf.get("families") or []}
    chosen = [timed[k] for k in decision.models if k in timed and timed[k][0] is not None]
    total = float(sum(s for s, _ in chosen))
    if not chosen or total < NOTEWORTHY_SECONDS:
        return
    note = f"Fitting {'it' if len(decision.models) == 1 else 'these'} takes {say(total, n_rows, n_columns)}"
    slowest_seconds, slowest = max(chosen, key=lambda c: c[0])
    if len(chosen) > 1 and slowest_seconds >= 0.5 * total:
        note += f", most of it {str(slowest).lower()}"
    ctx.read["note"] = note + "."


register_consequence("set_energy_adjustment", energy_adjustment_preview)
register_consequence("select_models", models_preview)

__all__ = ["energy_adjustment_preview", "fit_words", "lineage_caption", "models_preview",
           "register_step_phrase"]

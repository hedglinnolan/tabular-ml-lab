"""Consequence previews for the modeling sequence's declarations (MODELING_SEQUENCE §1 steps 2–5 and
row 11; §3, "what the canvas shows at each step"): what each answer does to the model the user is
building, drawn on their own rows in the closed vocabulary (BLUEPRINT §11 rule 2).

| Kind                  | Views (primary first)                                                    |
|-----------------------|--------------------------------------------------------------------------|
| ``set_estimand``      | lineage, the exposure's columns emphasized; the caption is the one-line  |
|                       | estimand                                                                 |
| ``set_adjustment``    | lineage: each covariate's derived role in the adjusted lane (confounder, |
|                       | mediator, collider, …), and only the adjusted ones reaching the matrix;  |
|                       | story: the roles derived, then the set                                   |
| ``set_model_sequence``| lineage per declared model, as a story: unadjusted → Model 1 → Model 2 → |
|                       | Model 3                                                                  |
| ``set_multiplicity``  | relationship: each test's threshold against its rank (flat, or BH's line)|
| ``set_exposure_form`` | relationship: the exposure against the terms the model sees, a frame per |
|                       | term · distribution with the knots or cut points marked · lineage        |
| ``set_clusters``      | distribution of the rows per group · lineage (fixed effects)             |
| ``set_survey``        | distribution of the exposure, unweighted and weighted · row flow of the  |
|                       | rows the design holds                                                    |
| ``set_follow_up``     | row flow with the landmark · distribution of follow-up with the horizon  |
| ``set_outcome_scale`` | distribution of the outcome, as recorded and on the log scale            |
| ``set_categorical``   | lineage: each coded column into its indicators                           |
| ``set_outcome_order`` | table focus: each level and the rank the model gives it                  |
| ``set_task``          | distribution or table of the outcome as the task reads it                |
| ``set_sensitivity``   | row flow per analysis                                                    |
| ``respond_diagnostic``| row flow without the influential rows (the other responses: a note)      |

**Preview = what will happen.** Each builder computes with the functions the downstream stage
calls, on the answer's state (:func:`consequences.after_state`) and the rows the purpose allows: a
sample of the training rows under prediction, of every analyzed row under inference
(``ctx.training_row_ids``, BLUEPRINT §12 ruling 3). The acceptance tests
(``tests/acceptance/test_previews_kinds.py``) hold each headline number to the stage's value
after the answer is recorded.

**Outcome-blind where the leash says so.** Under inference nothing here reads the outcome's relation
to anything: the functional form is drawn as the terms the model sees, never as a curve fitted to
the outcome, and the multiplicity threshold is drawn without a p-value (choosing the method after
seeing what survives it is a forking path; Gelman & Loken 2013).

Importing this module registers the builders.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from turbotab.core.consequences import (
    CAPTION_WORDS, FRAME_WORDS, MAX_VIEWS, TITLE_WORDS, DistributionView, FitLine, HistogramData,
    Lineage, LineageFrame, LineageLink, LineageNode, LineageView, Mark, PreviewContext,
    RelationshipFrame, RelationshipView, RowFlowView, RowStep, TableFocusView,
    TableRow, after_state, clip_words, fmt_count, fmt_value, register_consequence,
)

POINTS = 800


# ── words ────────────────────────────────────────────────────────────────────


def tick(value: Any) -> str:
    return f"`{value}`"


def names(columns: Sequence[str], limit: int = 3) -> str:
    """``a``, ``a and b``, ``a, b and c``, ``a, b and 4 more`` (ticked)."""
    shown = [tick(c) for c in columns]
    if len(shown) > limit:
        rest = len(shown) - (limit - 1)
        shown = shown[: limit - 1] + [f"{rest} more"]
    if len(shown) <= 1:
        return "".join(shown)
    return ", ".join(shown[:-1]) + " and " + shown[-1]


def num(x: float) -> str:
    """A value as a caption writes it: three figures, separators past a thousand."""
    x = float(x)
    if not np.isfinite(x):
        return "–"
    if x.is_integer() and abs(x) < 1e15:
        return f"{int(x):,}"
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    return f"{x:.3g}"


def series(values: Sequence[float]) -> str:
    """``1.2, 3.4 and 5.6``."""
    shown = [num(v) for v in values]
    return shown[0] if len(shown) == 1 else ", ".join(shown[:-1]) + " and " + shown[-1]


def caption(text: str) -> str:
    return clip_words(text, CAPTION_WORDS)


def title(text: str) -> str:
    return clip_words(text, TITLE_WORDS)


def frame_label(text: str) -> str:
    return clip_words(text, FRAME_WORDS)


def rows_word(ctx: PreviewContext) -> str:
    """The pool's rows, as the basis and captions name them (BLUEPRINT §12 ruling 3)."""
    return "analyzed" if ctx.training_kind == "analyzed" else "training"


def drawn(ctx: PreviewContext) -> str:
    """``"sampled "`` when the builders read a sample smaller than their pool, else nothing."""
    found = ctx.read.get("sample")
    return "sampled " if found is not None and found[2] < found[1] else ""


def agree(items: Sequence[Any], one: str, many: str) -> str:
    return one if len(items) == 1 else many


def moved(arrive: Sequence[str], leave: Sequence[str]) -> str:
    """``a`` leaves; ``b`` and ``c`` arrive."""
    parts = []
    if leave:
        parts.append(f"{names(leave, 2)} {agree(leave, 'leaves', 'leave')}")
    if arrive:
        parts.append(f"{names(arrive, 3)} {agree(arrive, 'arrives', 'arrive')}")
    return ("; ".join(parts) + ".") if parts else "The model sees the same columns."


# ── the model matrix on the preview's rows ───────────────────────────────────


@dataclass
class Design:
    """The design stage's shared steps, fitted on the preview's rows under one state."""

    state: Any
    spec: Any
    frame: pd.DataFrame  # the raw inputs (row id index)
    fitted: Any  # the fitted shared transformer
    matrix: pd.DataFrame
    lineage: Lineage

    @property
    def columns(self) -> list[str]:
        return [str(c) for c in self.matrix.columns]

    def sources(self) -> dict[str, tuple[set[str], set[str]]]:
        """Each matrix column's (raw sources, adjusted-lane sources), as the effects stage reads
        them (``stages.effects.matrix_sources``)."""
        from turbotab.core.stages.effects import matrix_sources

        class _Steps:  # matrix_sources drops the last step (the model's): give it one
            steps = [*self.fitted.steps, ("model", None)]

        return matrix_sources(_Steps(), self.spec.inputs)


def pool(ctx: PreviewContext) -> Any:
    """A sample of the rows the purpose allows (training rows, or every analyzed row under
    inference), or None before the split exists: a modeling choice previews on no other rows."""
    if ctx.training_row_ids is None:
        return None
    return ctx.sample_row_ids(ctx.training_row_ids)


def fit_design(ctx: PreviewContext, state: Any, ids: Any, extra: Sequence[str] = ()) -> Design | None:
    """The design stage's shared steps (``stages.modeling.design_stage``: the same spec, the same
    column summaries, the same energy factors) fitted on ``ids``; None when the state names no
    predictor yet, or a reading the fit needs is unsettled (the fit asks for it first)."""
    from turbotab.core.decisions import left_out
    from turbotab.core.methods.batch import batch_inputs
    from turbotab.core.models.lineage import missing_counts, trace
    from turbotab.core.models.pipeline import (design_spec, input_columns, model_predictors,
                                               modeling_frame, shared_steps, transformer)
    from turbotab.core.readings import Unsettled, predictors_or_ask

    store = ctx.datastore
    info_cols = store.info().columns
    try:
        predictors_or_ask(state, {c.name: {"dtype": c.dtype, "n_unique": c.n_unique}
                                  for c in info_cols}, drop=left_out(state), store=store)
    except Unsettled:
        return None
    predictors = model_predictors(state)
    if not predictors:
        return None
    adj = state.energy_adjustment
    available = set(store.columns)
    columns = [c for c in dict.fromkeys([*input_columns(predictors, adj), *batch_inputs(state),
                                         *extra]) if c in available]
    frame = modeling_frame(store, columns, ids)
    info = {c.name: c for c in info_cols}
    factors = None
    from turbotab.core.methods.energy import PARTITION_METHODS

    if adj is not None and adj.method in PARTITION_METHODS:
        from turbotab.core.readings import energy_source_factors

        factors = energy_source_factors(state, list(adj.nutrients), store=store)
    from turbotab.core.methods.qc_drift import working_qc_sd

    working = ctx.artifact("working")
    try:
        spec = design_spec(state, frame, predictors, column_info=info,
                           qc=working_qc_sd(working) if working is not None else None,
                           energy_factors=factors)
        fitted = transformer(shared_steps(spec)).fit(frame[spec.inputs])
        matrix = fitted.transform(frame[spec.inputs])
    except (ValueError, TypeError, KeyError) as exc:
        ctx.read.setdefault("note", str(getattr(exc, "reason", None) or exc))
        return None
    lineage = trace(fitted.steps, spec.inputs, spec.roles, missing_counts(frame[spec.inputs]))
    return Design(state=state, spec=spec, frame=frame, fitted=fitted, matrix=matrix,
                  lineage=lineage)


def matrix_of(lineage: Lineage) -> list[str]:
    return [n.column for n in lineage.nodes if n.lane == "matrix" and n.column is not None]


def lineage_change(before: Lineage | None, after: Lineage) -> tuple[list[str], list[str]]:
    """Matrix columns that arrive and that leave between two lineages."""
    now = matrix_of(before) if before is not None else []
    then = matrix_of(after)
    return [c for c in then if c not in now], [c for c in now if c not in then]


def points(x: np.ndarray, y: np.ndarray, keep: np.ndarray | None = None) -> list[tuple[float, float]]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.flatnonzero(np.isfinite(x) & np.isfinite(y) & (True if keep is None else keep))
    if len(ok) > POINTS:
        ok = np.sort(np.random.default_rng(0).choice(ok, size=POINTS, replace=False))
    return [(float(x[i]), float(y[i])) for i in ok]


def histogram(values: Any, bins: int = 30, edges: Sequence[float] | None = None) -> HistogramData:
    """Counts of the finite values, on ``edges`` when given (two pictures on one axis)."""
    v = np.asarray(values, dtype=float)
    finite = v[np.isfinite(v)]
    if edges is None:
        if not len(finite):
            return HistogramData(edges=[], counts=[], n_missing=int(len(v)))
        counts, found = np.histogram(finite, bins=bins)
        return HistogramData(edges=[float(e) for e in found], counts=[int(c) for c in counts],
                             n_missing=int(len(v) - len(finite)))
    e = np.asarray(edges, dtype=float)
    counts, _ = np.histogram(np.clip(finite, e[0], e[-1]), bins=e)
    return HistogramData(edges=[float(x) for x in e], counts=[int(c) for c in counts],
                         n_missing=int(len(v) - len(finite)))


def shared_edges(*arrays: Any, bins: int = 30) -> list[float]:
    both = np.concatenate([np.asarray(a, dtype=float) for a in arrays])
    both = both[np.isfinite(both)]
    if not len(both):
        return [0.0, 1.0]
    lo, hi = float(both.min()), float(both.max())
    if hi <= lo:
        hi = lo + 1.0
    return [float(x) for x in np.linspace(lo, hi, bins + 1)]


# ── set_estimand ─────────────────────────────────────────────────────────────


def exposure_features(state: Any, design: Design) -> list[str]:
    """The matrix columns that carry the declared exposure's effect (every family member's), as
    the served fit names them (``estimand.annotate_fit``: ``primary_features`` over the fit's
    coefficient names)."""
    from turbotab.core import estimand as est

    spec = est.current_estimand(state)
    if spec is None:
        return []
    predictors = list(est.predictor_roles(state))
    return sorted({f for e in est.exposures_of(state, spec)
                   for f in est.primary_features(design.columns, e, predictors)})


def estimand_line(state: Any) -> str | None:
    """The estimand in one line: whose effect, on what, on which scale."""
    from turbotab.core import estimand as est

    spec = est.current_estimand(state)
    if spec is None:
        return None
    effect = "Direct" if spec.effect == "direct" else "Total"
    measure = est.MEASURE_WORDS.get(str(spec.measure), str(spec.measure))
    target = tick(state.target)
    if spec.family:
        family = est.family_exposures(state)
        return (f"{effect} effect of each of {fmt_count(len(family))} exposures on {target}: "
                f"{measure}.")
    contrast = {"substitution": " in place of other calories",
                "addition": " added to the diet"}.get(str(spec.contrast), "")
    return f"{effect} effect of {tick(spec.exposure)}{contrast} on {target}: {measure}."


def estimand_views(decision: Any, ctx: PreviewContext) -> list[Any]:
    from turbotab.core import estimand as est

    ids = pool(ctx)
    if ids is None or getattr(ctx.state, "purpose", None) != "inference":
        return []
    after = after_state(decision, ctx)
    if est.current_estimand(after) is None:
        return []
    then = fit_design(ctx, after, ids)
    if then is None:
        return []
    now = fit_design(ctx, ctx.state, ids)
    feats = exposure_features(after, then)
    exposures = est.exposures_of(after, est.current_estimand(after))
    line = estimand_line(after)
    ctx.read["estimand"] = {"line": line, "features": feats, "exposures": exposures}
    views: list[Any] = [LineageView(
        title=title("The exposure in the model"),
        caption=caption(line or ""),
        emphasis=list(dict.fromkeys([*exposures, *feats]))[:12],
        before=now.lineage if now is not None else None,
        after=then.lineage,
    )]
    # MODELING_SEQUENCE §2: a declared exposure invalidates the adjustment answers given for
    # another one; they are asked again, and until then the covariates they left out are back.
    reopened = [c for c in est.adjustment_left_out(ctx.state)
                if c not in est.adjustment_left_out(after)]
    if reopened:
        ctx.read["note"] = (f"The adjustment answers were given for another exposure, so they are "
                            f"asked again; until then {names(reopened)} "
                            f"{'is' if len(reopened) == 1 else 'are'} back in the model.")
    return views


# ── set_adjustment ───────────────────────────────────────────────────────────

ROLE_SHORT = {
    "mediator_confounder": "mediator–outcome confounder",
    "confounder": "confounder",
    "exposure_cause": "cause of the exposure",
    "precision": "precision",
    "proxy": "proxy",
    "mediator": "mediator",
    "collider": "possible collider",
    "instrument": "instrument",
    "timing_unknown": "timing unknown",
    "not_a_cause": "cause of neither",
}


def adjustment_lineage(exposures: Sequence[str], covariates: Sequence[str],
                       derived: Mapping[str, Any], roles: Mapping[str, str],
                       *, decided: bool = True) -> Lineage:
    """The adjustment set as a lineage: each covariate (raw lane) to its derived role (the adjusted
    lane, its node grouped by the role), and on to the primary model's matrix only when the role
    adjusts for it; the exposure (each of a family's) straight through. ``decided`` False draws the
    roles alone (the story's first step); a covariate with no answer yet stands as "not
    answered"."""
    nodes, links = [], []
    for e in exposures:
        nodes += [LineageNode(id=f"raw:{e}", column=e, lane="raw", role=roles.get(e), label=e),
                  LineageNode(id=f"mx:{e}", column=e, lane="matrix", role=roles.get(e), label=e)]
        links.append(LineageLink(source=f"raw:{e}", target=f"mx:{e}", operation="the exposure"))
    for c in covariates:
        found = derived.get(c)
        role = ROLE_SHORT.get(found.role, found.role) if found is not None else "not answered"
        kept = found.adjusted if found is not None else True
        nodes.append(LineageNode(id=f"raw:{c}", column=c, lane="raw", role=roles.get(c), label=c))
        nodes.append(LineageNode(id=f"adj:{c}", column=c, lane="adjusted", role=roles.get(c),
                                 label=f"{c}: {role}", group=role,
                                 formula=(None if found is None else
                                          f"{role}: {'adjusted' if kept else 'left out'}")))
        links.append(LineageLink(source=f"raw:{c}", target=f"adj:{c}", operation=role))
        if decided and kept:
            nodes.append(LineageNode(id=f"mx:{c}", column=c, lane="matrix", role=roles.get(c),
                                     label=c))
            links.append(LineageLink(source=f"adj:{c}", target=f"mx:{c}", operation="adjusted"))
    return Lineage(nodes=nodes, links=links)


def adjustment_views(decision: Any, ctx: PreviewContext) -> list[Any]:
    from turbotab.core import estimand as est

    if getattr(ctx.state, "purpose", None) != "inference":
        return []
    after = after_state(decision, ctx)
    spec = est.current_estimand(after)
    if spec is None:
        return []
    exposures = est.exposures_of(after, spec)
    covariates = est.asked_covariates(after)
    roles = dict(est.predictor_roles(after))
    derived_now = (est.derived_roles(ctx.state) if est.current_estimand(ctx.state) is not None
                   else {})
    derived = est.derived_roles(after)
    before = adjustment_lineage(exposures, covariates, derived_now, roles) if derived_now else None
    after_l = adjustment_lineage(exposures, covariates, derived, roles)
    story = [LineageFrame(label=frame_label("Each covariate's role, from your answers"),
                          lineage=adjustment_lineage(exposures, covariates, derived, roles,
                                                     decided=False))]
    adjusted = [c for c in covariates if c in derived and derived[c].adjusted]
    out = [c for c in covariates if c in derived and not derived[c].adjusted]
    by_role: dict[str, list[str]] = {}
    for c in out:
        by_role.setdefault(ROLE_SHORT.get(derived[c].role, derived[c].role), []).append(c)
    parts = [f"{fmt_count(len(adjusted))} adjusted ({names(adjusted, 3)})" if adjusted
             else "none adjusted"]
    for role, cols in by_role.items():
        plural = "s" if len(cols) > 1 and not role.endswith("s") else ""
        parts.append(f"{names(cols, 2)} left out as {role}{plural}")
    waiting = [c for c in covariates if c not in derived]
    if waiting:
        parts.append(f"{fmt_count(len(waiting))} not answered yet")
    changed = [c for c in covariates if (derived_now.get(c) is None) != (derived.get(c) is None)
               or (c in derived and c in derived_now
                   and (derived[c].role, derived[c].adjusted)
                   != (derived_now[c].role, derived_now[c].adjusted))]
    return [LineageView(
        title=title("Which covariates the model adjusts for"),
        caption=caption("; ".join(parts) + "."),
        emphasis=(changed or list(decision.answers))[:12],
        before=before,
        after=after_l,
        story=story,
    )]


# ── set_model_sequence ───────────────────────────────────────────────────────

SEQUENCE_LABELS = {"crude": "Unadjusted", "model_1": "Model 1", "model_2": "Model 2 (primary)",
                   "model_3": "Model 3"}


def sequence_columns(state: Any, design: Design) -> dict[str, list[str]]:
    """Each declared model's adjusted-for columns, read as the effects stage reads them
    (``stages.effects._Run.family_block``): the unadjusted model is the exposure alone; Model 1 the
    declared columns (with "energy" bringing every term the energy model made); Model 2 every
    column of the primary's matrix; Model 3 adds the columns the answers set beside it."""
    from turbotab.core import estimand as est
    from turbotab.core.stages.effects import energy_outputs

    spec = est.current_estimand(state)
    exposures = set(est.exposures_of(state, spec))
    feats = set(exposure_features(state, design))
    sources = design.sources()
    declared = est.current_model_sequence(state)
    model_one = set(declared.model_1) if declared is not None else None

    class _Steps:
        steps = [*design.fitted.steps, ("model", None)]
        named_steps = dict(design.fitted.steps)

    energy = energy_outputs(_Steps())
    energy_named = bool(model_one and model_one & set(est.energy_terms_columns(state)))

    def raw(columns: Sequence[str]) -> list[str]:
        found: set[str] = set()
        for c in columns:
            if c in feats:
                continue
            found |= sources.get(str(c), ({str(c)}, set()))[0] - exposures
        order = [*design.spec.inputs, *sorted(found)]
        return [c for c in dict.fromkeys(order) if c in found]

    out = {"crude": [], "model_2": raw(design.columns)}
    if model_one is not None:
        def in_one(c: str) -> bool:
            src, adj = sources.get(str(c), ({str(c)}, {str(c)}))
            return bool(src & model_one) or (energy_named and bool(adj & energy))

        out["model_1"] = raw([c for c in design.columns if c in feats or in_one(c)])
    further = est.secondary_columns(state)
    if further:
        out["model_3"] = [*out["model_2"], *[c for c in further if c not in out["model_2"]]]
    return {k: out[k] for k in ("crude", "model_1", "model_2", "model_3") if k in out}


def _sequence_lineage(exposure_cols: Sequence[str], adjusted: Sequence[str],
                      roles: Mapping[str, str]) -> Lineage:
    nodes, links = [], []
    for c in [*exposure_cols, *adjusted]:
        nodes.append(LineageNode(id=f"raw:{c}", column=c, lane="raw", role=roles.get(c), label=c))
        nodes.append(LineageNode(id=f"mx:{c}", column=c, lane="matrix", role=roles.get(c), label=c))
        links.append(LineageLink(source=f"raw:{c}", target=f"mx:{c}",
                                 operation="the exposure" if c in exposure_cols else "adjusted"))
    return Lineage(nodes=nodes, links=links)


def model_sequence_views(decision: Any, ctx: PreviewContext) -> list[Any]:
    from turbotab.core import estimand as est

    ids = pool(ctx)
    if ids is None or getattr(ctx.state, "purpose", None) != "inference":
        return []
    after = after_state(decision, ctx)
    spec = est.current_estimand(after)
    if spec is None:
        return []
    design = fit_design(ctx, after, ids)
    if design is None:
        return []
    models = sequence_columns(after, design)
    exposures = est.exposures_of(after, spec)
    roles = dict(est.predictor_roles(after))
    frames = [(k, _sequence_lineage(exposures, cols, roles)) for k, cols in models.items()]
    last_key, last = frames[-1]
    story = []
    previous: list[str] = []
    for key, lineage in frames[:-1]:
        cols = models[key]
        added = [c for c in cols if c not in previous]
        alone = exposures[0] if len(exposures) == 1 else "exposures"
        label = (f"{SEQUENCE_LABELS[key]}: {alone} alone" if key == "crude"
                 else f"{SEQUENCE_LABELS[key]}: +{len(added)} columns")
        story.append(LineageFrame(label=frame_label(label), lineage=lineage))
        previous = cols
    one = models.get("model_1")
    parts = []
    if one is not None:
        parts.append(f"Model 1 adjusts for {names(one, 3) if one else 'nothing'}")
    two = models["model_2"]
    extra = [c for c in two if c not in (one or [])]
    parts.append(f"Model 2 adds {names(extra, 3)}" if one is not None and extra
                 else f"Model 2 adjusts for {fmt_count(len(two))} columns")
    if "model_3" in models:
        parts.append(f"Model 3 adds {names([c for c in models['model_3'] if c not in two], 2)}")
    return [LineageView(
        title=title("The declared models, one by one"),
        caption=caption("; ".join(parts) + "."),
        emphasis=list(one or [])[:12],
        before=frames[0][1],
        after=last,
        story=story,
    )]


# ── set_multiplicity ─────────────────────────────────────────────────────────

ALPHA = 0.05


def multiplicity_views(decision: Any, ctx: PreviewContext) -> list[Any]:
    """The threshold each of the family's tests is held to, by its rank among the p-values: flat at
    0.05 unadjusted (the count stated, or none), or Benjamini–Hochberg's line i × q / m. The family
    is every settled exposure, each tested in turn (``estimand.family_exposures``, the effects
    stage's and the feature-wise fit's). Drawn without the p-values: the method is chosen before
    they are seen (Gelman & Loken 2013, forking paths)."""
    from turbotab.core import estimand as est

    after = after_state(decision, ctx)
    if getattr(after, "purpose", None) != "inference":
        return []
    m = len(est.family_exposures(after))
    if m < 2:
        return []
    ranks = np.arange(1, m + 1, dtype=float)
    keep = np.unique(np.linspace(0, m - 1, min(m, POINTS)).round().astype(int))

    def line(method: str | None) -> list[tuple[float, float]]:
        if method == "bh":
            return [(float(ranks[i]), ALPHA * ranks[i] / m) for i in keep]
        return [(float(ranks[i]), ALPHA) for i in keep]

    now = (est.family_multiplicity(ctx.state)[0] if est.current_estimand(ctx.state) is not None
           else None)
    if decision.method == "bh":
        text = (f"{fmt_count(m)} tests: the i-th smallest p must fall below i × 0.05 / {m:,}; "
                f"the smallest below {fmt_value(ALPHA / m)}.")
    elif decision.method == "stated_count":
        text = (f"{fmt_count(m)} tests, each held to 0.05; the count is stated beside every "
                f"p-value.")
    else:
        text = f"{fmt_count(m)} tests, each held to 0.05, with no adjustment."
    ctx.read["note"] = ("Drawn without the p-values: the method is declared before the results "
                        "are seen.")
    ctx.read["multiplicity"] = {"m": m}
    return [RelationshipView(
        title=title("The threshold each test must clear"),
        caption=caption(text),
        emphasis=[],
        x_label="rank of the p-value (smallest first)",
        y_label_before="threshold for p",
        y_label_after="threshold for p",
        points_before=line("bh" if now in ("bh", "fdr_bh") else None),
        points_after=line("bh" if decision.method == "bh" else None),
        r_before=None,
        r_after=None,
    )]


# ── set_exposure_form ────────────────────────────────────────────────────────


def _form_step(design: Design) -> Any:
    from turbotab.core.methods.exposure_form import form_step

    return form_step(design.fitted)


def form_facts(design: Design, column: str) -> dict[str, Any] | None:
    """What the fitted form step learned for ``column`` on the preview's rows: its formed name,
    its form, the knots or cut points, and the terms it puts in the matrix."""
    from turbotab.core.methods.exposure_form import formed_name

    name = formed_name(column, design.spec.energy_adjustment())
    step = _form_step(design)
    if step is None or name not in step._plan():
        return {"name": name, "form": "linear", "terms": [name], "knots": [], "cuts": []}
    form, _ = step._plan()[name]
    return {"name": name, "form": form, "terms": [str(t) for t in step._outputs(name)],
            "knots": [float(k) for k in step.knots_.get(name, [])],
            "cuts": [float(c) for c in step.cuts_.get(name, [])],
            "notes": list(step.knot_notes_.get(name, [])),
            "medians": [float(m) for m in step.medians_.get(name, [])]}


def _received(design: Design, name: str, column: str) -> np.ndarray:
    """The values the form step receives for ``name``: the raw inputs through every step before
    it (the energy step's output when one replaced the column)."""
    X = design.frame[design.spec.inputs]
    for step_name, step in design.fitted.steps:
        if step_name == "form":
            break
        X = step.transform(X)
    if name in X.columns:
        return X[name].to_numpy(dtype=float)
    return design.frame[column].to_numpy(dtype=float)


def form_terms(design: Design, facts: Mapping[str, Any],
               x: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """The terms the form puts in the matrix, each as ``(name, values per row)``: the column itself
    (linear); the spline's straight-line and nonlinear columns; or, for quintiles, the group (1–5)
    and the trend score (each row's quintile median)."""
    if facts["form"] == "spline":
        return [(t, design.matrix[t].to_numpy(dtype=float)) for t in facts["terms"]]
    if facts["form"] == "quintiles":
        from turbotab.core.methods.exposure_form import quantile_group

        group = quantile_group(x, np.asarray(facts["cuts"]))
        medians = np.asarray(facts["medians"])
        score = np.where(group < 0, np.nan, medians[np.clip(group, 0, len(medians) - 1)])
        return [("quintile (1–5)", np.where(group < 0, np.nan, group + 1.0)),
                (f"{facts['name']} (quintile median)", score)]
    return [(facts["name"], x)]


def exposure_form_views(decision: Any, ctx: PreviewContext) -> list[Any]:
    from turbotab.core.methods.exposure_form import KNOT_PERCENTILES

    ids = pool(ctx)
    if ids is None:
        return []
    after = after_state(decision, ctx)
    then = fit_design(ctx, after, ids)
    if then is None:
        return []
    now = fit_design(ctx, ctx.state, ids)
    facts = form_facts(then, decision.column)
    name = facts["name"]
    x = _received(then, name, decision.column)
    terms = form_terms(then, facts, x)
    current = form_facts(now, decision.column) if now is not None else None
    now_terms = (form_terms(now, current, _received(now, current["name"], decision.column))
                 if current is not None else [(name, x)])
    form = facts["form"]
    marks: list[Mark] = []
    story: list[RelationshipFrame] = []
    if form == "spline":
        k = len(facts["knots"])
        pct = KNOT_PERCENTILES.get(k) if not facts.get("notes") else None
        for i, value in enumerate(facts["knots"]):
            marks.append(Mark(value=float(value),
                              label=f"knot {i + 1}" + (f" · {pct[i] * 100:g}th pct" if pct else "")))
        story = [RelationshipFrame(label=frame_label(f"{terms[0][0]}: the straight-line term"),
                                   points=points(x, terms[0][1]), r=None,
                                   fit_line=FitLine(slope=1.0, intercept=0.0), y_label=terms[0][0])]
        story += [RelationshipFrame(label=frame_label(f"{t}: bends past knot {j}"),
                                    points=points(x, v), r=None, y_label=t)
                  for j, (t, v) in enumerate(terms[1:-1], start=1)]
        text = (f"{tick(name)} enters as {len(terms)} terms, a restricted cubic spline with "
                f"knots at {series(facts['knots'])}.")
    elif form == "quintiles":
        marks = [Mark(value=float(c), label=f"Q{i + 1} | Q{i + 2}")
                 for i, c in enumerate(facts["cuts"])]
        story = [RelationshipFrame(label=frame_label("Cut into fifths of the rows"),
                                   points=points(x, terms[0][1]), r=None, y_label=terms[0][0])]
        text = (f"{tick(name)} enters as 4 indicators against its lowest fifth; cut at "
                f"{series(facts['cuts'])}.")
    else:
        text = f"{tick(name)} enters as itself: one straight-line term, one coefficient per unit."
    ctx.read["form"] = {"column": decision.column, "name": name, "form": form,
                        "knots": facts["knots"], "cuts": facts["cuts"],
                        "terms": list(facts["terms"])}
    views: list[Any] = [RelationshipView(
        title=title(f"How {tick(decision.column)} enters the model"),
        caption=caption(text),
        emphasis=[decision.column],
        x_label=name,
        y_label_before=now_terms[-1][0],
        y_label_after=terms[-1][0],
        points_before=(points(x, now_terms[-1][1]) if len(now_terms[-1][1]) == len(x)
                       else points(x, x)),
        points_after=points(x, terms[-1][1]),
        r_before=None,
        r_after=None,
        story=story,
    )]
    if marks:
        hist = histogram(x)
        views.append(DistributionView(
            title=title(f"Where the {'knots' if form == 'spline' else 'cut points'} fall"),
            caption=caption(f"Learned from {fmt_count(int(np.isfinite(x).sum()))} {drawn(ctx)}"
                            f"{rows_word(ctx)} rows of {tick(name)}; the fit learns them again on "
                            f"its own rows."),
            emphasis=[decision.column],
            column=name, before=hist, after=hist, before_label=name, after_label=name,
            marks=marks,
        ))
    arrive, leave = lineage_change(now.lineage if now is not None else None, then.lineage)
    if arrive or leave:
        views.append(LineageView(
            title=title("Columns the model will see"),
            caption=caption(moved(arrive, leave)),
            emphasis=[*arrive, *leave][:12],
            before=now.lineage if now is not None else None,
            after=then.lineage,
        ))
    return views[:MAX_VIEWS]


# ── set_clusters ─────────────────────────────────────────────────────────────


def clusters_views(decision: Any, ctx: PreviewContext) -> list[Any]:
    """The grouping the intervals will cluster by, read as the fit reads it
    (``models.inference.resolve_clusters`` under the answer's state): how many groups, and how many
    rows each holds; under fixed effects, the indicators the group adds to the matrix."""
    from turbotab.core.models.inference import cluster_columns, min_clusters, resolve_clusters

    after = after_state(decision, ctx)
    named = decision.column or next(iter(decision.none_of or []), None)
    if named is None or named not in ctx.datastore.columns:
        return []
    ids = ctx.sample_row_ids(ctx.training_row_ids) if ctx.training_row_ids is not None \
        else ctx.sample_row_ids()
    units = cluster_columns(after, ctx.datastore.columns)
    frame = ctx.datastore.materialize(list(dict.fromkeys([*units, named])), ids)
    clusters = resolve_clusters(after, frame[units]) if units else None
    column = clusters.column if clusters is not None and clusters.clustered else None
    shown = column or named
    sizes = frame[shown].dropna().value_counts().to_numpy(dtype=float)
    if not len(sizes):
        return []
    g = int(clusters.n_clusters) if column is not None else int(len(sizes))
    hist = histogram(sizes, bins=min(20, max(5, len(sizes))))
    floor = min_clusters()
    if column is None:
        text = (f"No grouping: the {fmt_count(len(frame))} sampled rows are read as independent, "
                f"{tick(named)}'s {fmt_count(len(sizes))} levels ignored.")
    elif decision.adjust == "fixed_effects" and getattr(after, "purpose", None) == "inference":
        text = (f"{fmt_count(g)} groups of {num(sizes.min())}–{num(sizes.max())} rows: one "
                f"intercept each, intervals clustered by {tick(column)}.")
    else:
        text = (f"{fmt_count(g)} groups of {num(sizes.min())}–{num(sizes.max())} rows; intervals "
                f"are clustered by {tick(column)}.")
    ctx.read["clusters"] = {"column": column, "n_clusters": g if column else None}
    views: list[Any] = [DistributionView(
        title=title(f"Rows in each {tick(shown)} group"),
        caption=caption(text),
        emphasis=[shown],
        column=shown, before=hist, after=hist,
        before_label="rows per group", after_label="rows per group",
    )]
    if column is not None and g < floor:
        ctx.read["note"] = (f"{g} groups is fewer than the {floor} a cluster-robust interval needs, "
                            f"so no interval is reported.")
    if decision.adjust == "fixed_effects" and ctx.training_row_ids is not None:
        rows = ctx.sample_row_ids(ctx.training_row_ids)
        then = fit_design(ctx, after, rows)
        now = fit_design(ctx, ctx.state, rows)
        if then is not None:
            arrive, _ = lineage_change(now.lineage if now is not None else None, then.lineage)
            if arrive:
                views.append(LineageView(
                    title=title(f"{tick(named)} as fixed effects"),
                    caption=caption(f"{tick(named)} enters as {fmt_count(len(arrive))} "
                                    f"indicators, one per group after the first."),
                    emphasis=[named, *arrive][:12],
                    before=now.lineage if now is not None else None,
                    after=then.lineage,
                ))
    return views


# ── set_survey ───────────────────────────────────────────────────────────────


def survey_column(state: Any) -> str | None:
    """The column the weighted picture is drawn on: the declared exposure, else the first
    numeric predictor."""
    from turbotab.core import estimand as est

    spec = est.current_estimand(state)
    if spec is not None and not spec.family:
        return str(spec.exposure)
    roles = est.predictor_roles(state)
    return next((c for c, r in roles.items() if r == "exposure"), next(iter(roles), None))


def weighted_counts(values: np.ndarray, weights: np.ndarray, edges: Sequence[float]) -> list[int]:
    """Each bin's share of the total weight, as rows of the same sample (rounded): the surveyed
    population's distribution on the sample's scale."""
    e = np.asarray(edges, dtype=float)
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    w, _ = np.histogram(np.clip(values[ok], e[0], e[-1]), bins=e, weights=weights[ok])
    total = float(w.sum())
    return whole(w / total * int(ok.sum()) if total > 0 else w)


def whole(shares: Any) -> list[int]:
    """Real counts as whole ones that keep their total (the largest remainders round up)."""
    x = np.asarray(shares, dtype=float)
    floor = np.floor(x)
    short = int(round(float(x.sum()) - float(floor.sum())))
    if short > 0:
        floor[np.argsort(-(x - floor), kind="stable")[:short]] += 1
    return [int(v) for v in floor]


def survey_views(decision: Any, ctx: PreviewContext) -> list[Any]:
    from turbotab.core.methods.survey import analysis_weights
    from turbotab.core.models.survey import build_design, domain_of

    column = survey_column(after_state(decision, ctx))
    if column is None or column not in ctx.datastore.columns:
        return []
    ids = ctx.sample_row_ids(ctx.training_row_ids) if ctx.training_row_ids is not None \
        else ctx.sample_row_ids()
    x = pd.to_numeric(ctx.datastore.materialize([column], ids)[column],
                      errors="coerce").to_numpy(dtype=float)
    edges = shared_edges(x)
    plain = histogram(x, edges=edges)
    if decision.estimand == "sample" or not decision.weight:
        return [DistributionView(
            title=title(f"{tick(column)} as these participants show it"),
            caption=caption(f"Unweighted: the estimates describe these {fmt_count(len(x))} "
                            f"sampled rows' participants, not the surveyed population."),
            emphasis=[column], column=column, before=plain, after=plain,
            before_label="these participants (unweighted)",
            after_label="these participants (unweighted)",
        )]
    design_cols = [c for c in (decision.weight, decision.strata, decision.psu, decision.cycle,
                               decision.four_year_weight) if c]
    if any(c not in ctx.datastore.columns for c in design_cols):
        return []
    # The design over every row of the working table (a domain keeps its strata and PSUs), as
    # ``methods.survey.for_fit`` reads it.
    whole = ctx.datastore.materialize(list(dict.fromkeys(design_cols)), None)
    pooled = analysis_weights(whole, decision.weight, decision.cycle, decision.four_year_weight)
    if pooled.refusal:
        ctx.read["note"] = pooled.refusal
        return []
    design = build_design(whole, pooled.weights, weight_column=decision.weight,
                          strata_column=decision.strata, psu_column=decision.psu)
    position = pd.Index(design.row_ids).get_indexer(np.asarray(ids, dtype=np.int64))
    w = np.where(position >= 0, design.weight[np.clip(position, 0, None)], np.nan)
    weighted = HistogramData(edges=edges, counts=weighted_counts(x, w, edges),
                             n_missing=int((~np.isfinite(x)).sum()))
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    mean_u = float(np.mean(x[np.isfinite(x)])) if np.isfinite(x).any() else float("nan")
    mean_w = float(np.average(x[ok], weights=w[ok])) if ok.any() else float("nan")
    pool_ids = ctx.training_row_ids if ctx.training_row_ids is not None else ctx.unsealed_row_ids()
    domain = domain_of(pool_ids, design)
    at = domain.at
    df = int(len(np.unique(design.psu[at])) - len(np.unique(design.stratum[at])))
    text = (f"Mean {tick(column)} {num(mean_u)} unweighted, {num(mean_w)} weighted by "
            f"{tick(decision.weight)}; {fmt_count(df)} design degrees of freedom.")
    views: list[Any] = [DistributionView(
        title=title(f"{tick(column)}: these participants and the population"),
        caption=caption(text),
        emphasis=[column, decision.weight],
        column=column, before=plain, after=weighted,
        before_label="these participants (unweighted)",
        after_label=f"the surveyed population (weighted by {decision.weight})",
    )]
    n_pool = int(len(pool_ids))
    held = int(domain.keep.sum())
    if held < n_pool:
        start = RowStep(key="analyzed", label="Rows analyzed", n=n_pool)
        steps = [start, RowStep(key="in_design", label="With a stratum, PSU and positive weight",
                                n=held, dropped=n_pool - held,
                                reason="outside the survey design: no stratum, PSU or weight")]
        views.append(RowFlowView(
            title=title("Rows the population estimate reads"),
            caption=caption(f"{fmt_count(n_pool - held)} of {fmt_count(n_pool)} rows have no "
                            f"stratum, PSU or positive weight."),
            emphasis=["in_design"], before=[start], after=steps,
        ))
    ctx.read["survey"] = {"df": df, "n_psu": design.n_psu, "n_strata": design.n_strata,
                          "domain": held, "mean_unweighted": mean_u, "mean_weighted": mean_w}
    sampled = ctx.read.get("sample")
    values = (f"a sample of {sampled[2]:,} of the {sampled[1]:,} {rows_word(ctx)} rows"
              if sampled is not None and sampled[2] < sampled[1] else
              f"all {len(ids):,} {rows_word(ctx)} rows")
    ctx.read["basis"] = (f"The design's weights, strata and PSUs read on every row of the table, as "
                         f"the fit reads them; `{column}` on {values}.")
    return views


# ── set_follow_up ────────────────────────────────────────────────────────────


def follow_up_views(decision: Any, ctx: PreviewContext) -> list[Any]:
    from turbotab.core import row_previews as rp
    from turbotab.core.models.survival import time_to_event_outcome
    from turbotab.core.stages.modeling import coded_outcome

    after = after_state(decision, ctx)
    if getattr(after, "task", None) != "time_to_event" or after.follow_up is None:
        return []
    time_col = decision.time_column
    if time_col not in ctx.datastore.columns or after.target not in ctx.datastore.columns:
        return []
    views: list[Any] = []
    frame, _, [(before, _), (steps, kept)] = rp._flows(ctx, [ctx.state, after])
    if decision.landmark is not None:
        cut = next((s for s in steps if s["key"] == "landmark"), None)
        if cut is not None:
            views.append(RowFlowView(
                title=title("Rows at risk at the landmark"),
                caption=caption(f"{fmt_count(cut['dropped'])} rows' follow-up ended by "
                                f"{num(decision.landmark)}: not at risk then; "
                                f"{fmt_count(cut['n'])} remain."),
                emphasis=["landmark"],
                before=rp._steps(before, rp._sealed(ctx)),
                after=rp._steps(steps, rp._sealed(ctx)),
            ))
    # Follow-up on the rows the cohort keeps, as the fit codes it (``time_to_event_outcome``).
    rows = np.asarray(kept, dtype=np.int64)
    if not len(rows):
        return views
    cols = [c for c in dict.fromkeys([after.target, time_col,
                                      *([decision.entry_column] if decision.entry_column else [])])]
    data = ctx.datastore.materialize(cols, rows)
    raw_time = pd.to_numeric(data[time_col], errors="coerce").to_numpy(dtype=float)
    try:
        coded = coded_outcome("time_to_event", data[after.target].to_numpy(), after.event)
        y = time_to_event_outcome(after, data, coded)
        open_ended = after.follow_up.model_copy(update={"horizon": None})
        plain = time_to_event_outcome(after.model_copy(update={"follow_up": open_ended}), data,
                                      coded)
    except ValueError as exc:
        ctx.read["note"] = str(exc)
        return views
    events_all, events = int(plain["event"].sum()), int(y["event"].sum())
    edges = shared_edges(raw_time)
    marks = []
    for value, label in ((decision.landmark, "landmark"), (decision.prediction_horizon,
                                                           "prediction horizon"),
                         (decision.horizon, "horizon")):
        if value is not None:
            marks.append(Mark(value=float(value), label=f"{label} {num(value)}"))
    if decision.horizon is not None:
        text = (f"{fmt_count(events_all - events)} events after {num(decision.horizon)} become "
                f"censored there; {fmt_count(events)} of {fmt_count(len(rows))} rows keep "
                f"the event.")
    else:
        text = f"{fmt_count(events)} of {fmt_count(len(rows))} rows end with the event."
    ctx.read["follow_up"] = {"rows": int(len(rows)), "events": events, "events_all": events_all}
    views.append(DistributionView(
        title=title(f"Follow-up time, {tick(time_col)}"),
        caption=caption(text),
        emphasis=[time_col],
        column=time_col,
        before=histogram(raw_time, edges=edges),
        after=histogram(y["time"], edges=edges),
        before_label="follow-up as recorded",
        after_label="follow-up as the model reads it",
        marks=marks,
    ))
    return views[:MAX_VIEWS]


# ── set_outcome_scale ────────────────────────────────────────────────────────


def skew_text(value: float | None) -> str:
    return tick("–" if value is None else f"{value:.2f}".replace("-", "−"))


def outcome_scale_views(decision: Any, ctx: PreviewContext) -> list[Any]:
    from turbotab.core.structural import skewness

    column = decision.column
    if column not in ctx.datastore.columns:
        return []
    ids = ctx.sample_row_ids()
    y = pd.to_numeric(ctx.datastore.materialize([column], ids)[column],
                      errors="coerce").to_numpy(dtype=float)
    y = y[np.isfinite(y)]
    if not len(y):
        return []
    positive = y > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        logged = np.where(positive, np.log(np.where(positive, y, 1.0)), np.nan)
    skew = skewness(y)
    skew_log = skewness(logged[np.isfinite(logged)])
    if decision.scale == "log":
        text = (f"{tick(column)} skewness {skew_text(skew)}; on the log scale, "
                f"{skew_text(skew_log)}: effects become ratios of geometric means.")
        after_label = f"ln_{column}"
        after_hist = histogram(logged)
        story: list[Any] = []
    else:
        text = (f"{tick(column)} stays as recorded (skewness {skew_text(skew)}): effects are "
                f"differences in means.")
        after_label, after_hist, story = column, histogram(y), []
    ctx.read["outcome_scale"] = {"skewness": skew, "log_skewness": skew_log, "n": int(len(y))}
    return [DistributionView(
        title=title(f"{tick(column)} on the scale analyzed"),
        caption=caption(text),
        emphasis=[column],
        column=column,
        before=histogram(y),
        after=after_hist,
        before_label=f"{column} as recorded",
        after_label=after_label,
        story=story,
    )]


# ── set_categorical ──────────────────────────────────────────────────────────


def categorical_views(decision: Any, ctx: PreviewContext) -> list[Any]:
    ids = pool(ctx)
    if ids is None:
        ids = ctx.sample_row_ids()
    after = after_state(decision, ctx)
    then = fit_design(ctx, after, ids)
    if then is None:
        return []
    now = fit_design(ctx, ctx.state, ids)
    arrive, leave = lineage_change(now.lineage if now is not None else None, then.lineage)
    coded = [c for c in decision.columns if c in then.spec.inputs]
    if not coded:
        ctx.read["note"] = (f"{names(list(decision.columns))} "
                            f"{agree(decision.columns, 'is', 'are')} not among the model's "
                            f"predictors, so nothing the model sees changes.")
        return []
    per = {c: [m for m in arrive if m.startswith(f"{c}_")] for c in coded}
    text = "; ".join(f"{tick(c)} becomes {len(per[c])} indicators" for c in coded[:2])
    return [LineageView(
        title=title("Codes enter as categories"),
        caption=caption(text + ", one per level after the first."),
        emphasis=[*coded, *arrive][:12],
        before=now.lineage if now is not None else None,
        after=then.lineage,
    )]


# ── set_outcome_order ────────────────────────────────────────────────────────


def outcome_order_views(decision: Any, ctx: PreviewContext) -> list[Any]:
    column = decision.column
    if column not in ctx.datastore.columns:
        return []
    ids = ctx.sample_row_ids()
    values = ctx.datastore.materialize([column], ids)[column]
    counts = values.dropna().astype(str).value_counts()
    rank = {lv: i + 1 for i, lv in enumerate(decision.levels)}
    current = getattr(ctx.state, "outcome_order", None) or []
    rank_now = {lv: i + 1 for i, lv in enumerate(current)}
    rows = []
    for i, lv in enumerate(decision.levels[:8]):
        rows.append(TableRow(row_id=i, before={column: lv, "rank": rank_now.get(lv),
                                               "rows": int(counts.get(lv, 0))},
                             after={column: lv, "rank": rank[lv], "rows": int(counts.get(lv, 0))}))
    changed = [(r.row_id, "rank") for r in rows if r.before["rank"] != r.after["rank"]]
    return [TableFocusView(
        title=title(f"The order of {tick(column)}'s levels"),
        caption=caption(f"Lowest {tick(decision.levels[0])} to highest {tick(decision.levels[-1])}: "
                        f"each cut-point model compares the levels above with those below."),
        emphasis=[column],
        columns_before=[column, "rank", "rows"],
        columns_after=[column, "rank", "rows"],
        rows=rows,
        changed=changed,
        n_affected_columns=1,
    )]


# ── set_task ─────────────────────────────────────────────────────────────────


def task_views(decision: Any, ctx: PreviewContext) -> list[Any]:
    column = decision.column
    if column not in ctx.datastore.columns:
        return []
    ids = ctx.sample_row_ids()
    values = ctx.datastore.materialize([column], ids)[column]
    present = values.dropna()
    if decision.task == "regression":
        y = pd.to_numeric(present, errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(y).any():
            return []
        h = histogram(y)
        return [DistributionView(
            title=title(f"{tick(column)} as a quantity"),
            caption=caption(f"{fmt_count(int(np.isfinite(y).sum()))} values from {num(np.nanmin(y))} "
                            f"to {num(np.nanmax(y))}: the model predicts its mean."),
            emphasis=[column], column=column, before=h, after=h,
            before_label=column, after_label=column,
        )]
    counts = present.astype(str).value_counts()
    levels = list(counts.index[:8])
    word = {"binary": "two classes", "multiclass": "unordered classes",
            "ordinal": "ordered levels", "time_to_event": "an event with its follow-up"}.get(
        decision.task, decision.task)
    rows = [TableRow(row_id=i, before={column: lv, "rows": int(counts[lv])},
                     after={column: lv, "rows": int(counts[lv])}) for i, lv in enumerate(levels)]
    return [TableFocusView(
        title=title(f"{tick(column)} as {word}"),
        caption=caption(f"{fmt_count(len(counts))} levels read as {word}; the largest, "
                        f"{tick(levels[0])}, holds {fmt_count(int(counts.iloc[0]))} rows."),
        emphasis=[column], columns_before=[column, "rows"], columns_after=[column, "rows"],
        rows=rows, changed=[], n_affected_columns=1,
    )]


# ── set_sensitivity ──────────────────────────────────────────────────────────


def sensitivity_views(decision: Any, ctx: PreviewContext) -> list[Any]:
    """One row flow per analysis beside the primary: the rows its own rules keep."""
    from turbotab.core import row_previews as rp

    if not decision.analyses:
        return []
    states = [ctx.state] + [ctx.state.model_copy(update={"exclusions": list(a.rules)})
                            for a in decision.analyses]
    _, _, flows = rp._flows(ctx, states)
    primary = flows[0][0]
    views = []
    for analysis, (steps, _) in zip(decision.analyses, flows[1:]):
        n_after, n_primary = steps[-1]["n"], primary[-1]["n"]
        views.append(RowFlowView(
            title=title(f"Rows in “{analysis.label}”"),
            caption=caption(f"{fmt_count(n_after)} rows, against {fmt_count(n_primary)} in the "
                            f"primary analysis."),
            emphasis=[steps[-1]["key"]],
            before=rp._steps(primary, rp._sealed(ctx)),
            after=rp._steps(steps, rp._sealed(ctx)),
        ))
    return views[:MAX_VIEWS]


# ── respond_diagnostic ───────────────────────────────────────────────────────


def diagnostic_views(decision: Any, ctx: PreviewContext) -> list[Any]:
    """What the recorded response shows beside the estimate, read from the effects stage's own
    diagnostics (shown with the estimates, so nothing here is seen for the first time)."""
    effects = ctx.artifact("effects")
    data = getattr(effects, "data", effects)
    if not isinstance(data, dict):
        return []
    found = None
    for fam in data.get("families") or []:
        for diag in fam.get("diagnostics") or []:
            if diag.get("check") == decision.check:
                found = diag
                break
        if found is not None:
            break
    if found is None:
        return []
    n = int(found.get("n") or 0)
    if decision.action == "without_influential" and found.get("flagged") is not None:
        flagged = int(found["flagged"])
        start = RowStep(key="analyzed", label="Rows analyzed", n=n)
        return [RowFlowView(
            title=title("The primary refit without influential rows"),
            caption=caption(f"{fmt_count(flagged)} rows above Cook's distance "
                            f"{fmt_value(found.get('threshold') or 0)} leave the refit shown "
                            f"beside the estimate."),
            emphasis=["influential"],
            before=[start],
            after=[start, RowStep(key="influential", label="Without the influential rows",
                                  n=n - flagged, dropped=flagged,
                                  reason="Cook's distance above the threshold")],
        )]
    if decision.action == "keep_labeled":
        ctx.read["note"] = ("The estimate stays as fitted, labeled with the failed check; nothing "
                            "else changes.")
        return []
    ctx.read["note"] = ("The exposure's hazard ratio is shown before and after the median event "
                        "time, beside the average over follow-up.")
    return []


register_consequence("set_estimand", estimand_views)
register_consequence("set_adjustment", adjustment_views)
register_consequence("set_model_sequence", model_sequence_views)
register_consequence("set_multiplicity", multiplicity_views)
register_consequence("set_exposure_form", exposure_form_views)
register_consequence("set_clusters", clusters_views)
register_consequence("set_survey", survey_views)
register_consequence("set_follow_up", follow_up_views)
register_consequence("set_outcome_scale", outcome_scale_views)
register_consequence("set_categorical", categorical_views)
register_consequence("set_outcome_order", outcome_order_views)
register_consequence("set_task", task_views)
register_consequence("set_sensitivity", sensitivity_views)
register_consequence("respond_diagnostic", diagnostic_views)

__all__ = ["Design", "exposure_features", "fit_design", "form_facts", "sequence_columns",
           "estimand_line", "survey_column"]

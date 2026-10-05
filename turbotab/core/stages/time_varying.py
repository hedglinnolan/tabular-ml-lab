"""The ``time_varying`` stage: an exposure that changes over time, estimated by g-methods.

The routing is ``turbotab/core/time_varying.py`` and the computation ``models/time_varying.py``.
This stage reads the working table's analyzed rows (every eligible row under inference, BLUEPRINT
§12 ruling 3) and returns, in order:

1. **the setting**. These facts come from the values: the unit (the grain answer's column) and the
   settled time column (BLUEPRINT §14.1), the number of time points, whether each unit's rows run
   from the first time point without a gap, whether the exposure is 0 or 1 and changes within units,
   and whether each covariate changes within units. The proposal splits the adjustment set's
   covariates and every confounder affected by prior exposure into time-varying confounders and
   baseline covariates. The Router reads the setting: an exposure fixed within every unit does not
   ask the question.
2. **the diagnostics**, before any estimate. For the weights: their distribution overall and at each
   time point, the stabilized mean's check, and each truncation option's effect on them. For both
   g-methods: positivity at each time point. None of these reads the outcome.
3. **the estimates**, only once the lane is complete. Under the weights, that means after the
   truncation is declared. The estimates are the marginal structural model's coefficient with its
   unit-clustered interval, or the g-formula's risks under "always" and "never" with their bootstrap
   intervals, and the E-value of either (MODELING_SEQUENCE §0 ruling 10).

The methods sentence is the §13 contract's clause (``contracts.paragraph``). It states what
was done and the diagnostics, never the estimates.
"""
from __future__ import annotations

import math
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from turbotab.core.graph import Bundle, StageContext

SEED = 20261005  # the g-formula's Monte Carlo and bootstrap draws: fixed, so a rerun reproduces them
MEAN_FLAG = 0.1  # a stabilized mean farther than this from 1 is flagged (the app's convention)


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_serialization_defaults_required=True)


class Candidate(_Model):
    column: str
    varies: bool  # changes within at least one unit
    role: str | None  # the role the adjustment answers derive (``energy``: total energy)
    adjusted: bool  # in the adjustment set the answers declare
    affected: bool  # a confounder affected by prior exposure


class Proposal(_Model):
    confounders: list[str]
    baseline: list[str]


class Setting(_Model):
    exposure: str
    unit: str
    time_column: str
    time_points: int
    units: int
    rows: int
    late_units: int  # units whose first row is not the first time point
    gap_units: int  # units missing a time point between their first and last
    repeat_units: int  # units with two rows at one time point
    exposure_binary: bool
    exposure_levels: int
    exposure_varies: bool
    exposure_changers: int  # units whose exposure changes
    exposure_stops: int  # units whose exposure goes from 1 to 0
    outcome_kind: Literal["event", "repeated_binary", "measure", "other"]
    units_after_event: int  # units with rows after their first event
    candidates: list[Candidate]
    varies: dict[str, bool]
    proposal: Proposal


class WeightSummary(_Model):
    n: int
    mean: float
    sd: float
    min: float
    p1: float
    p25: float
    median: float
    p75: float
    p99: float
    max: float


class WeightsAtTime(WeightSummary):
    time: int


class TruncationOption(_Model):
    key: Literal["none", "p1_p99", "p5_p95"]
    label: str
    customary: str
    sound: str
    summary: WeightSummary  # the weights the outcome model would read under this option
    chosen: bool


class PositivityAtTime(_Model):
    time: int
    rows: int
    exposed: int
    unexposed: int
    p_min: float
    p_max: float
    near_zero: int
    near_one: int


class Diagnostics(_Model):
    weights: WeightSummary | None = None  # the product of exposure and censoring weights
    weights_by_time: list[WeightsAtTime] = []
    exposure_weights: WeightSummary | None = None
    censoring_weights: WeightSummary | None = None
    truncation: list[TruncationOption] = []
    positivity: list[PositivityAtTime] = []
    concerns: list[str] = []


class EstimateRow(_Model):
    label: str
    measure: str
    estimate: float
    ci_low: float | None = None
    ci_high: float | None = None
    se: float | None = None  # on the scale the interval is built on (log for a ratio)
    p: float | None = None


class RiskCurve(_Model):
    strategy: Literal["always", "never", "natural", "observed"]
    label: str
    risks: list[float]  # by the end of each time point
    mc_se: float | None = None


class EValue(_Model):
    reads: str  # what ratio it reads, and how
    rr: float
    lo: float
    hi: float
    point: float
    ci: float


class Estimates(_Model):
    rows: list[EstimateRow]
    curves: list[RiskCurve] = []
    e_value: EValue | None = None
    n_rows: int
    n_units: int
    simulations: int | None = None
    bootstrap: int | None = None
    failed_resamples: int | None = None


class LaneOption(_Model):
    key: str
    label: str
    customary: str
    sound: str
    rung: str
    order: int


class TimeVaryingArtifact(_Model):
    purpose: Literal["inference"]
    applies: bool
    reason: str | None = None  # why it does not apply, or why nothing is estimated
    exposure: str | None = None
    method: str | None = None
    setting: Setting | None = None
    options: list[LaneOption] = []
    affected: list[str] = []
    diagnostics: Diagnostics | None = None
    estimates: Estimates | None = None
    withheld: str | None = None
    # The estimates blocked and recorded (the surveyed population with no design-based estimator,
    # MS4), and the ways forward.
    exits: list[dict[str, Any]] = []
    relations: list[str] = []  # what the lane implies, as fired here ("because you chose …")
    methods: str = ""


TIME_VARYING_READS = ("purpose", "target", "task", "event", "grain", "repeat_kind", "unit",
                      "temporal", "estimand", "adjustment", "clusters", "time_varying", "lens",
                      "categorical", "roles", "roles_unconfirmed", "role_confirmations",
                      "reading_confirmations", "shape_confirmations", "missing", "survey")

# MS4 (MODELING_SEQUENCE §0 ruling 6, §4 "population estimand without a design-based estimator"):
# the weights, the outcome model and the simulation read these rows as sampled, so under the
# surveyed population the estimates are blocked and recorded, the sample-only attestation the exit.
# The diagnostics read no outcome and describe these rows; they are shown as they are.
POPULATION = ("The g-methods here have no design-based estimator: their estimates would describe "
              "these participants, not the surveyed population the survey answer names, and their "
              "intervals would ignore the strata and PSUs.")


def population_block(state: Any) -> tuple[str, list[dict[str, Any]]] | None:
    """The block and its exit under the surveyed population, else None."""
    from turbotab.core.models.survey import SAMPLE_EXIT

    survey = getattr(state, "survey", None)
    if survey is None or getattr(survey, "estimand", None) != "population":
        return None
    return POPULATION, [{"label": SAMPLE_EXIT,
                         "decision": {"kind": "set_survey", "estimand": "sample"}}]


# ── the stage ────────────────────────────────────────────────────────────────


def _dump(artifact: TimeVaryingArtifact) -> Bundle:
    return Bundle(data=artifact.model_dump(mode="json"))


def lane_options(affected: Sequence[str], binary: bool = True) -> list[dict[str, Any]]:
    """The contract's options for inference, soundest first. With no confounder affected by prior
    exposure, standard regression is sound and leads; with one, it is block and record. An exposure
    that is not 0 or 1 has no g-method here (V2 scope): each is refused, with the reason."""
    from turbotab.core.contracts import options_for
    from turbotab.core.time_varying import KEY

    out = options_for(KEY, "inference")
    if not binary:
        for o in out:
            if o["key"] != "standard":
                o.update(rung="refused", order=o["order"] + 10,
                         sound="not available here: the weights and the always-or-never "
                               "strategies need an exposure of 0 or 1 at each time point")
    if not affected:
        for o in out:
            if o["key"] == "standard":
                o.update(rung="recommended", order=-1,
                         sound="sound here: no time-varying confounder is affected by prior "
                               "exposure, so adjusting for each at each time point blocks no part "
                               "of the effect")
            elif o["rung"] == "recommended":
                o["rung"] = "available"
    out.sort(key=lambda o: o["order"])
    return out


def time_varying_stage(ctx: StageContext) -> Bundle:
    from turbotab.core.time_varying import (affected_confounders, current_lane, exposure_of,
                                            lane_gate)

    state = ctx.state
    structure = getattr(ctx.inputs.get("structure"), "data", ctx.inputs.get("structure"))
    gate = lane_gate(state, structure)
    exposure = exposure_of(state)
    if gate is not None and gate[0] == "not_applicable":
        return _dump(TimeVaryingArtifact(purpose="inference", applies=False, reason=gate[1],
                                         exposure=exposure))
    if exposure is None:
        return _dump(TimeVaryingArtifact(purpose="inference", applies=False,
                                         reason="No single exposure is declared yet."))
    affected = affected_confounders(state)
    options = [LaneOption(**o) for o in lane_options(affected)]
    base = dict(purpose="inference", applies=True, exposure=exposure, options=options,
                affected=affected)
    try:
        frame, setting = read_setting(ctx, exposure, affected, structure)
    except NotReady as why:
        return _dump(TimeVaryingArtifact(**base, reason=str(why)))
    lane = current_lane(state)
    base["setting"] = setting
    base["options"] = [LaneOption(**o) for o in lane_options(affected, setting.exposure_binary)]
    if lane is None:
        return _dump(TimeVaryingArtifact(
            **base, withheld="No estimate until the lane is chosen: a marginal structural model, "
                             "the parametric g-formula, or standard regression."))
    base["method"] = lane.method
    if lane.method == "standard":
        present = {"estimand", *(["standard_adjustment_for_affected_confounders"]
                                 if affected else [])}
        return _dump(TimeVaryingArtifact(
            **base, relations=_fired(lane, present),
            reason="Standard regression: the estimate is the fitted model's."))
    try:
        if lane.method == "msm_iptw":
            return _dump(_msm(ctx, frame, setting, lane, affected, base))
        return _dump(_gformula(ctx, frame, setting, lane, affected, base))
    except NotReady as why:
        return _dump(TimeVaryingArtifact(**base, reason=str(why)))


class NotReady(Exception):
    """Nothing can be estimated yet, and the artifact says why (a refusal in the data's words)."""


# ── reading the data ─────────────────────────────────────────────────────────


def _codes(state: Any) -> set[str]:
    from turbotab.core.readings import confirmed_codes

    return set(confirmed_codes(state)) | set(getattr(state, "categorical", None) or [])


def _time_points(values: pd.Series, levels: Sequence[str] | None) -> np.ndarray | None:
    """Each row's time point as its rank (0 … K − 1): by the declared order of text labels, else by
    value (numbers or dates). None when text has no declared order."""
    from turbotab.core.models.time_varying import time_index

    if levels:
        position = {str(v): i for i, v in enumerate(levels)}
        mapped = values.astype(str).map(position)
        if mapped.isna().any():
            return None
        return time_index(mapped.to_numpy(dtype=float))
    if pd.api.types.is_numeric_dtype(values) or pd.api.types.is_datetime64_any_dtype(values):
        if values.isna().any():
            return None
        return time_index(values.to_numpy())
    return None


def candidate_columns(state: Any) -> list[str]:
    """The covariates the lane may adjust for: the adjustment question's covariates, every
    confounder affected by prior exposure, and total energy (settled energy roles)."""
    from turbotab.core.estimand import asked_covariates, predictor_roles
    from turbotab.core.time_varying import affected_confounders, exposure_of

    exposure = exposure_of(state)
    energy = [c for c, r in predictor_roles(state).items() if r == "energy"]
    out = [*asked_covariates(state), *affected_confounders(state), *energy]
    return [c for c in dict.fromkeys(out) if c != exposure]


def read_setting(ctx: StageContext, exposure: str, affected: Sequence[str],
                 structure: Any) -> tuple[pd.DataFrame, Setting]:
    """The analyzed rows sorted by unit and time, with ``__unit``, ``__time`` and the outcome coded,
    and what their values say."""
    from turbotab.core.estimand import derived_roles, predictor_roles
    from turbotab.core.models.pipeline import modeling_frame
    from turbotab.core.models.time_varying import check_schedule
    from turbotab.core.stages.data import open_store
    from turbotab.core.stages.modeling import coded_outcome, read_assignment
    from turbotab.core.stages.working import declared_levels, effective_grain, time_column

    state = ctx.state
    grain = effective_grain(state, structure)
    unit = getattr(grain, "id_column", None) if grain is not None else None
    if not unit:
        raise NotReady("No column names the unit whose rows repeat (the grain answer names it).")
    order = time_column(state, structure)
    if order is None:
        raise NotReady("No column that orders each unit's rows is settled; the weights and the "
                       "g-formula read each unit's history in time order.")
    target = state.target
    lane = getattr(state, "time_varying", None)
    censoring = getattr(lane, "censoring", None) if lane is not None else None
    covariates = candidate_columns(state)
    rows = read_assignment(ctx.inputs["split"]).index.to_numpy()
    wanted = [c for c in dict.fromkeys([unit, order, exposure, target, *covariates,
                                        *([censoring] if censoring else [])]) if c]
    with open_store(ctx) as store:
        missing = [c for c in wanted if c not in store.columns]
        if missing:
            raise NotReady(f"The working table has no column named {', '.join(missing)}.")
        frame = modeling_frame(store, wanted, rows, outcome=target)
    t = _time_points(frame[order], declared_levels(state, order))
    if t is None:
        raise NotReady(f"`{order}`'s values cannot be put in order: text labels need their order "
                       f"declared with the repeats answer, and no time point may be blank.")
    frame = frame.assign(__unit=frame[unit].to_numpy(), __time=t)
    frame = frame.sort_values(["__unit", "__time"], kind="mergesort").reset_index(drop=True)
    task = str(state.task or ctx.inputs["target_info"]["task"])
    y = coded_outcome(task, frame[target].to_numpy(), state.event)
    frame["__y"] = pd.to_numeric(pd.Series(np.asarray(y, dtype=object)),
                                 errors="coerce").to_numpy(float)
    sched = check_schedule(frame, "__unit", "__time")

    a = pd.to_numeric(frame[exposure], errors="coerce")
    levels = int(a.nunique(dropna=True))
    binary = bool(a.notna().all() and set(a.unique()) <= {0, 1})
    g = frame.assign(__a=a).groupby("__unit", sort=False)["__a"]
    changers = int((g.nunique(dropna=True) > 1).sum())
    stops = int((g.diff() < 0).groupby(frame["__unit"], sort=False).any().sum()) if binary else 0

    yv = frame["__y"]
    if task == "binary" and set(yv.dropna().unique()) <= {0, 1}:
        seen = yv.fillna(0).groupby(frame["__unit"], sort=False).cumsum() - yv.fillna(0)
        after = int((seen > 0).groupby(frame["__unit"], sort=False).any().sum())
        kind = "event" if after == 0 else "repeated_binary"
    elif task == "regression":
        after, kind = 0, "measure"
    else:
        after, kind = 0, "other"

    varies = {c: bool((frame.groupby("__unit", sort=False)[c].nunique(dropna=True) > 1).any())
              for c in covariates}
    derived = derived_roles(state)
    roles = predictor_roles(state)
    candidates = []
    for c in covariates:
        d = derived.get(c)
        role = "energy" if roles.get(c) == "energy" else (d.role if d is not None else None)
        adjusted = role == "energy" or bool(d is not None and d.adjusted)
        candidates.append(Candidate(column=c, varies=varies[c], role=role, adjusted=adjusted,
                                    affected=c in affected))
    proposal = Proposal(
        confounders=[c.column for c in candidates if c.varies and (c.adjusted or c.affected)],
        baseline=[c.column for c in candidates if not c.varies and c.adjusted and not c.affected])
    setting = Setting(
        exposure=exposure, unit=unit, time_column=order, time_points=sched["time_points"],
        units=sched["units"], rows=int(len(frame)), late_units=sched["late"],
        gap_units=sched["gaps"], repeat_units=sched["repeats"], exposure_binary=binary,
        exposure_levels=levels, exposure_varies=changers > 0, exposure_changers=changers,
        exposure_stops=stops, outcome_kind=kind, units_after_event=after, candidates=candidates,
        varies=varies, proposal=proposal)
    return frame, setting


def _design_columns(frame: pd.DataFrame, columns: Sequence[str], codes: set[str]
                    ) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Numeric model columns: a number as itself; a code or a text column one indicator per level
    but the first (sorted). Returns the columns and, per source column, its design names."""
    out: dict[str, np.ndarray] = {}
    names: dict[str, list[str]] = {}
    for c in columns:
        values = frame[c]
        if c in codes or not pd.api.types.is_numeric_dtype(values):
            levels = sorted(pd.unique(values.dropna().astype(str)))
            names[c] = []
            for level in levels[1:]:
                name = f"{c}={level}"
                out[name] = (values.astype(str) == level).to_numpy(dtype=float)
                names[c].append(name)
        else:
            out[c] = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
            names[c] = [c]
    return pd.DataFrame(out, index=frame.index), names


def _complete(frame: pd.DataFrame, setting: Setting, lane: Any) -> None:
    """The lane needs every value at each time point it reads, and a regular schedule."""
    needed = [setting.exposure, *lane.confounders, *lane.baseline,
              *([lane.censoring] if lane.censoring else [])]
    blanks = {c: int(frame[c].isna().sum()) for c in needed if frame[c].isna().any()}
    out_blank = int(frame["__y"].isna().sum())
    if out_blank:
        blanks["the outcome"] = out_blank
    if blanks:
        said = ", ".join(f"`{c}` ({n:,} rows)" if c != "the outcome" else f"the outcome ({n:,} rows)"
                         for c, n in blanks.items())
        raise NotReady(f"Values are missing in {said}. The weights and the g-formula read each time "
                       f"point's values in order, so a missing one breaks the unit's history; "
                       f"multiple imputation of long data for g-methods is not offered here.")
    if setting.late_units or setting.gap_units or setting.repeat_units:
        raise NotReady(
            f"Each unit needs one row at each time point from the first, with none missing in "
            f"between: {setting.late_units:,} units start later, {setting.gap_units:,} skip a time "
            f"point and {setting.repeat_units:,} have two rows at one.")
    if not setting.exposure_binary:
        raise NotReady(f"`{setting.exposure}` is not 0 or 1 at every time point.")
    if lane.pattern == "initiation" and setting.exposure_stops:
        raise NotReady(f"`{setting.exposure}` stops after starting in {setting.exposure_stops:,} "
                       f"units, so it is not an exposure that, once started, stays.")


def _fired(lane: Any, present: set[str]) -> list[str]:
    from turbotab.core.contracts import fired
    from turbotab.core.time_varying import KEY

    return [f.says for f in fired({KEY: lane.method}, "inference", consequences=sorted(present))
            if f.relation.target in present]


def _summary(values: Mapping[str, Any]) -> WeightSummary:
    return WeightSummary(**{k: values[k] for k in WeightSummary.model_fields})


# ── the marginal structural model ────────────────────────────────────────────


def weight_terms(lane: Any, names: Mapping[str, list[str]]) -> dict[str, list[str]]:
    """The weight models' terms (design names): the numerator reads the baseline covariates, the
    exposure's previous value (an exposure that can stop and restart) and time; the denominator
    also the time-varying confounders. The censoring models read the current exposure too."""
    from turbotab.core.models.time_varying import LAG, TIME, TIME2

    V = [n for c in lane.baseline for n in names[c]]
    L = [n for c in lane.confounders for n in names[c]]
    prior = [LAG + "__a"] if lane.pattern == "switches" else []
    return {"numerator": [*V, *prior, TIME, TIME2], "denominator": [*V, *L, *prior, TIME, TIME2],
            "censoring_numerator": [*V, "__a", TIME, TIME2],
            "censoring_denominator": [*V, *L, "__a", TIME, TIME2]}


def _msm(ctx: StageContext, frame: pd.DataFrame, setting: Setting, lane: Any,
         affected: Sequence[str], base: dict[str, Any]) -> TimeVaryingArtifact:
    from turbotab.core.models import time_varying as tv

    _complete(frame, setting, lane)
    if setting.outcome_kind not in ("event", "repeated_binary", "measure"):
        raise NotReady("The marginal structural model here needs a yes/no outcome at each time "
                       "point or a repeated measure.")
    ctx.progress(0.1, "Modeling the exposure at each time point")
    design, names = _design_columns(frame, [*lane.confounders, *lane.baseline], _codes(ctx.state))
    d = pd.concat([frame[["__unit", "__time", "__y"]], design], axis=1)
    d["__a"] = pd.to_numeric(frame[setting.exposure]).to_numpy(float)
    d = tv.with_history(d, "__unit", "__time", ["__a"])
    terms = weight_terms(lane, names)
    family = "gaussian" if setting.outcome_kind == "measure" else "binomial"
    try:
        exposure_w = tv.ipw_weights(d, id="__unit", time="__time", indicator="__a",
                                    numerator=terms["numerator"], denominator=terms["denominator"],
                                    kind="first" if lane.pattern == "initiation" else "all")
        censor_w = None
        if lane.censoring:
            d["__c"] = pd.to_numeric(frame[lane.censoring]).to_numpy(float)
            # A row with the event ends follow-up, so it is not at risk of being lost after it.
            at_risk = (d["__y"] == 0).to_numpy() if setting.outcome_kind == "event" else None
            censor_w = tv.censoring_weights(d, id="__unit", time="__time", indicator="__c",
                                            numerator=terms["censoring_numerator"],
                                            denominator=terms["censoring_denominator"],
                                            at_risk=at_risk)
    except tv.NotEstimable as exc:
        raise NotReady(f"The weights cannot be estimated: {exc}.") from None
    w = exposure_w.weights * (censor_w.weights if censor_w is not None else 1.0)
    keep = np.ones(len(d), dtype=bool)  # every analyzed row's outcome was seen
    level = tv.TRUNCATIONS[lane.truncation] if lane.truncation else None
    used = tv.truncate(w[keep], level) if lane.truncation else w[keep]
    t = d["__time"].to_numpy()
    concerns = []
    summary = tv.summarize(w[keep])
    if abs(summary["mean"] - 1.0) > MEAN_FLAG:
        concerns.append(
            f"The stabilized weights' mean is {summary['mean']:.2f}, not near 1: Cole & Hernán "
            f"(2008) read a mean far from one as nonpositivity or a misspecified weight model.")
    positivity = tv.positivity(t, d["__a"], exposure_w.p_event, exposure_w.modeled)
    empty = [p["time"] for p in positivity if not p["exposed"] or not p["unexposed"]]
    if empty:
        concerns.append(f"At time point{'s' if len(empty) > 1 else ''} {', '.join(map(str, empty))} "
                        f"no row is exposed, or none unexposed: the estimate there rests on the "
                        f"model alone.")
    near = sum(p["near_zero"] + p["near_one"] for p in positivity)
    if near:
        concerns.append(f"{near:,} rows have a fitted probability of exposure within {tv.NEAR} of "
                        f"0 or 1 (near-violations of positivity).")
    diagnostics = Diagnostics(
        weights=_summary(summary),
        weights_by_time=[WeightsAtTime(**x) for x in tv.by_time(w[keep], t[keep])],
        exposure_weights=_summary(tv.summarize(exposure_w.weights[keep])),
        censoring_weights=(_summary(tv.summarize(censor_w.weights[keep]))
                           if censor_w is not None else None),
        truncation=[TruncationOption(**o, summary=_summary(tv.summarize(tv.truncate(
            w[keep], tv.TRUNCATIONS[o["key"]]))), chosen=o["key"] == lane.truncation)
            for o in TRUNCATION_OPTIONS],
        positivity=[PositivityAtTime(**p) for p in positivity], concerns=concerns)
    present = {"weight_diagnostics_before_estimates", "positivity_by_time_point", "estimand"}
    if lane.censoring:
        present.add("censoring_weighted")
    run = {"lane": lane, "setting": setting, "names": names, "summary": tv.summarize(used),
           "family": family, "affected": affected, "target": ctx.state.target}
    if lane.truncation is None:
        return TimeVaryingArtifact(
            **base, diagnostics=diagnostics, relations=_fired(lane, present),
            withheld="Read the weights and positivity first: no estimate is computed until the "
                     "truncation is declared (none is an answer).",
            methods=methods_paragraph(run))
    blocked = population_block(ctx.state)
    if blocked is not None:
        return TimeVaryingArtifact(**base, diagnostics=diagnostics, relations=_fired(lane, present),
                                   reason=blocked[0], exits=blocked[1],
                                   methods=methods_paragraph(run))
    ctx.progress(0.6, "Fitting the weighted outcome model")
    summary_term = "__cum" if lane.summary == "cumulative" else "__a"
    d["__cum"] = d.groupby("__unit", sort=False)["__a"].cumsum().to_numpy(float)
    V = [n for c in lane.baseline for n in names[c]]
    X, xnames = tv.design(d.loc[keep], [summary_term, *V, tv.TIME, tv.TIME2])
    try:
        fit = tv.fit_msm(X, xnames, d.loc[keep, "__y"], used, d.loc[keep, "__unit"], family)
    except tv.NotEstimable as exc:
        raise NotReady(f"The weighted outcome model cannot be fit: {exc}.") from None
    r = fit.row(summary_term)
    per = (f"per time point of `{setting.exposure}` so far" if lane.summary == "cumulative"
           else f"for current `{setting.exposure}`")
    yv = d.loc[keep, "__y"].to_numpy(float)
    if family == "binomial":
        rare = float(yv.mean()) < tv.RARE
        est, lo, hi = math.exp(r["estimate"]), math.exp(r["ci_low"]), math.exp(r["ci_high"])
        row = EstimateRow(label=f"Odds ratio {per}", measure="odds_ratio", estimate=est,
                          ci_low=lo, ci_high=hi, se=r["se"], p=r["p"])
        ev = tv.e_value_or(est, lo, hi, rare)
        reads = ("the odds ratio as a risk ratio (the outcome is rare, below 15% of rows)" if rare
                 else "the square root of the odds ratio as a risk ratio (the outcome is common)")
    else:
        sd = float(np.std(yv, ddof=1))
        row = EstimateRow(label=f"Mean difference {per}", measure="mean_difference",
                          estimate=r["estimate"], ci_low=r["ci_low"], ci_high=r["ci_high"],
                          se=r["se"], p=r["p"])
        ev = tv.e_value_md(r["estimate"] / sd, r["se"] / sd)
        reads = "the mean difference in outcome SDs, as exp(0.91 d)"
    estimates = Estimates(rows=[row], e_value=EValue(reads=reads, **ev), n_rows=fit.n_rows,
                          n_units=fit.n_units)
    present |= {"intervals_by_unit", "unmeasured_confounding_sensitivity"}
    return TimeVaryingArtifact(**base, diagnostics=diagnostics, estimates=estimates,
                               relations=_fired(lane, present), methods=methods_paragraph(run))


TRUNCATION_OPTIONS = (
    {"key": "none", "label": "Keep the weights",
     "customary": "customary when the weights are well behaved",
     "sound": "unbiased if the weight models are right; extreme weights widen the interval"},
    {"key": "p1_p99", "label": "Truncate at the 1st and 99th percentiles",
     "customary": "customary (Cole & Hernán 2008)",
     "sound": "trades a little bias for precision: each step of truncation adds bias and removes "
              "variance"},
    {"key": "p5_p95", "label": "Truncate at the 5th and 95th percentiles",
     "customary": "less common",
     "sound": "more bias for more precision; Cole & Hernán found the precision gained outweighed "
              "by the bias"},
)


# ── the parametric g-formula ─────────────────────────────────────────────────


def gformula_spec(lane: Any, names: Mapping[str, list[str]], kinds: Mapping[str, str]) -> Any:
    """Every model's terms (``gfoRmula``'s defaults made explicit): each time-varying confounder
    given the baseline covariates, the confounders before it at that time point, every variable's
    previous value and time; the exposure and the outcome given the confounders too."""
    from turbotab.core.models import time_varying as tv

    V = tuple(n for c in lane.baseline for n in names[c])
    L = [c for c in lane.confounders]
    lagged = ("__a", *L)
    history = (*(tv.LAG + c for c in lagged), tv.TIME, tv.TIME2)
    covariates = tuple(tv.Covariate(c, kinds[c], (*V, *L[:i], *history))  # type: ignore[arg-type]
                       for i, c in enumerate(L))
    return tv.GFormulaSpec(
        id="__unit", time="__time", exposure="__a", outcome="__y", covariates=covariates,
        exposure_terms=(*V, *L, *history), outcome_terms=("__a", *L, *V, *history), baseline=V,
        lagged=lagged, absorbing=lane.pattern == "initiation")


def _gformula(ctx: StageContext, frame: pd.DataFrame, setting: Setting, lane: Any,
              affected: Sequence[str], base: dict[str, Any]) -> TimeVaryingArtifact:
    from turbotab.core.models import time_varying as tv

    _complete(frame, setting, lane)
    if setting.outcome_kind != "event":
        raise NotReady("The parametric g-formula here needs an event: a yes/no outcome at each time "
                       "point, with no rows after a unit's event.")
    codes = _codes(ctx.state)
    several = [c for c in lane.confounders if c in codes or
               not pd.api.types.is_numeric_dtype(frame[c])]
    several = [c for c in several if frame[c].nunique(dropna=True) > 2]
    if several:
        raise NotReady(f"{', '.join(f'`{c}`' for c in several)} {'has' if len(several) == 1 else 'have'} "
                       f"several categories; the g-formula here simulates a time-varying confounder "
                       f"that is 0 or 1, or a number. The marginal structural model reads it.")
    design, names = _design_columns(frame, [*lane.confounders, *lane.baseline], codes)
    for c in lane.confounders:  # a two-level code's one indicator stands for the confounder
        if names[c] != [c]:
            design[c] = design[names[c][0]] if names[c] else 0.0
            names[c] = [c]
    d = pd.concat([frame[["__unit", "__time", "__y"]], design], axis=1)
    d["__a"] = pd.to_numeric(frame[setting.exposure]).to_numpy(float)
    kinds = {c: ("binary" if set(d[c].dropna().unique()) <= {0, 1} else "normal")
             for c in lane.confounders}
    spec = gformula_spec(lane, names, kinds)
    ctx.progress(0.05, "Reading positivity at each time point")
    h = tv.with_history(d, "__unit", "__time", ["__a"])
    terms = weight_terms(lane, names)
    try:
        exposure_w = tv.ipw_weights(h, id="__unit", time="__time", indicator="__a",
                                    numerator=None, denominator=terms["denominator"],
                                    kind="first" if lane.pattern == "initiation" else "all")
    except tv.NotEstimable as exc:
        raise NotReady(f"The exposure model cannot be estimated: {exc}.") from None
    positivity = tv.positivity(h["__time"].to_numpy(), h["__a"], exposure_w.p_event,
                               exposure_w.modeled)
    concerns = []
    empty = [p["time"] for p in positivity if not p["exposed"] or not p["unexposed"]]
    if empty:
        concerns.append(f"At time point{'s' if len(empty) > 1 else ''} {', '.join(map(str, empty))} "
                        f"no row is exposed, or none unexposed: the simulated risks there "
                        f"extrapolate from the models.")
    diagnostics = Diagnostics(positivity=[PositivityAtTime(**p) for p in positivity],
                              concerns=concerns)
    blocked = population_block(ctx.state)
    if blocked is not None:
        return TimeVaryingArtifact(**base, diagnostics=diagnostics,
                                   relations=_fired(lane, {"positivity_by_time_point", "estimand"}),
                                   reason=blocked[0], exits=blocked[1])
    ctx.progress(0.1, "Simulating the strategies")
    try:
        result = tv.gformula(d, spec, n_sim=lane.simulations, seed=SEED)
    except tv.NotEstimable as exc:
        raise NotReady(f"The g-formula's models cannot be fit: {exc}.") from None
    boot = _bootstrap(ctx, d, spec, lane)
    iv = boot["intervals"]
    always, never = result.final("always"), result.final("never")
    rows = [EstimateRow(label="Risk if always exposed", measure="risk", estimate=always,
                        ci_low=iv["always"][0], ci_high=iv["always"][1]),
            EstimateRow(label="Risk if never exposed", measure="risk", estimate=never,
                        ci_low=iv["never"][0], ci_high=iv["never"][1]),
            EstimateRow(label="Risk difference (always − never)", measure="risk_difference",
                        estimate=always - never, ci_low=iv["difference"][0],
                        ci_high=iv["difference"][1]),
            EstimateRow(label="Risk ratio (always / never)", measure="risk_ratio",
                        estimate=always / never if never > 0 else math.nan,
                        ci_low=iv["ratio"][0], ci_high=iv["ratio"][1])]
    curves = [RiskCurve(strategy="always", label="Always exposed", risks=result.risks["always"],
                        mc_se=result.mc_se["always"]),
              RiskCurve(strategy="never", label="Never exposed", risks=result.risks["never"],
                        mc_se=result.mc_se["never"]),
              RiskCurve(strategy="natural", label="Natural course (simulated)",
                        risks=result.risks["natural"], mc_se=result.mc_se["natural"]),
              RiskCurve(strategy="observed", label="Observed", risks=result.observed)]
    ev = None
    if never > 0 and all(math.isfinite(x) for x in iv["ratio"]):
        ev = EValue(reads="the risk ratio", **tv.e_value(always / never, *iv["ratio"]))
    estimates = Estimates(rows=rows, curves=curves, e_value=ev, n_rows=int(len(d)),
                          n_units=setting.units, simulations=result.n_sim, bootstrap=boot["reps"],
                          failed_resamples=boot["failed"])
    present = {"positivity_by_time_point", "intervals_by_unit", "risks_under_always_and_never",
               "estimand"}
    if ev is not None:
        present.add("unmeasured_confounding_sensitivity")
    run = {"lane": lane, "setting": setting, "names": names, "kinds": kinds,
           "affected": affected, "target": ctx.state.target, "n_sim": result.n_sim,
           "mc": max(result.mc_se["always"], result.mc_se["never"])}
    return TimeVaryingArtifact(**base, diagnostics=diagnostics, estimates=estimates,
                               relations=_fired(lane, present), methods=methods_paragraph(run))


def _bootstrap(ctx: StageContext, d: pd.DataFrame, spec: Any, lane: Any) -> dict[str, Any]:
    """The percentile bootstrap over units, in chunks so the progress bar moves."""
    from turbotab.core.models import time_varying as tv

    reps, chunk = int(lane.bootstrap), 25
    draws: list[dict[str, Any]] = []
    done = 0
    while done < reps:
        n = min(chunk, reps - done)
        draws.append(tv.gformula_bootstrap(d, spec, reps=n, n_sim=None, seed=SEED + 1 + done))
        done += n
        ctx.progress(0.1 + 0.85 * done / reps, f"Bootstrap resample {done:,} of {reps:,}")
    merged: dict[str, list[float]] = {}
    for part in draws:
        for k, v in part["draws"].items():
            merged.setdefault(k, []).extend(v)
    intervals = {k: (float(np.nanquantile(v, 0.025)), float(np.nanquantile(v, 0.975)))
                 for k, v in merged.items()}
    return {"intervals": intervals, "reps": reps, "failed": sum(p["failed"] for p in draws)}


# ── the methods sentence ─────────────────────────────────────────────────────


def _tick(c: Any) -> str:
    return f"`{c}`"


def _listing(items: Sequence[Any]) -> str:
    """Phrases joined as a list (``a, b and c``); column names are ticked by the caller."""
    from turbotab.core.voice import listing

    return listing(list(items), limit=8, ticked=False)


def methods_paragraph(run: Mapping[str, Any]) -> str:
    """The lane's methods text through its §13 contract (``contracts.paragraph``)."""
    from turbotab.core.contracts import paragraph
    from turbotab.core.time_varying import KEY

    lane = run["lane"]
    lead, details = (_msm_text(run) if lane.method == "msm_iptw" else _gformula_text(run))
    return paragraph({KEY: lane.method}, {"clause": lead}, "inference", details=details)


def _assumptions(x: str) -> str:
    return (f"The estimate assumes no unmeasured confounding at each time point given the history, "
            f"positivity, and that each time point's {x} precedes the outcome it is paired with, "
            f"as declared.")


def _affected_text(affected: Sequence[str], x: str) -> list[str]:
    if not affected:
        return []
    one = len(affected) == 1
    return [f"{_listing(list(map(_tick, affected)))} "
            f"{'is a time-varying confounder' if one else 'are time-varying confounders'} "
            f"affected by prior {x}, which standard regression cannot adjust for."]


def _msm_text(run: Mapping[str, Any]) -> tuple[str, list[str]]:
    from turbotab.core.time_varying import CITE, TRUNCATION_WORDS

    lane, setting = run["lane"], run["setting"]
    x, y, unit = _tick(lane.exposure), _tick(run["target"]), _tick(setting.unit)
    lead = (f"the effect of {x} on {y} was estimated by a marginal structural model with stabilized "
            f"inverse-probability weights ({CITE['robins_2000']})")
    given = [*(["its previous value"] if lane.pattern == "switches" else []),
             *map(_tick, lane.baseline), "time and its square"]
    weights = (f"The weights came from pooled logistic models of {x} at each of "
               f"{setting.time_points} time points"
               + (" up to the one it started (once started, it stays)"
                  if lane.pattern == "initiation" else "")
               + f", given {_listing(given)} (numerator)"
               + (f" and also {_listing(list(map(_tick, lane.confounders)))} (denominator)"
                  if lane.confounders else " and the same (denominator)"))
    if lane.censoring:
        weights += (f"; loss to follow-up ({_tick(lane.censoring)}: 1 on a unit's last time point "
                    f"before it was lost) was weighted by the inverse probability of having stayed "
                    f"through the previous time point, from the same models given current {x} as "
                    f"well")
    if lane.truncation:
        s = run["summary"]
        weights += (f"; the weights were {TRUNCATION_WORDS[lane.truncation]} (mean {s['mean']:.2f}, "
                    f"from {s['min']:.2f} to {s['max']:.2f})")
    weights += "."
    summary = ("the number of time points exposed so far" if lane.summary == "cumulative"
               else f"current {x}")
    kind = "pooled logistic" if run["family"] == "binomial" else "linear"
    on = _listing([summary, *map(_tick, lane.baseline), "time and its square"])
    outcome = (f"The outcome model was a weighted {kind} model of {y} on {on}, with a variance "
               f"clustered by {unit} that treats the weights as known, so its interval is "
               f"conservative ({CITE['hernan_2000']}).")
    diag = ("The weights' distribution and positivity at each time point were read before any "
            "estimate was shown.")
    return lead, [weights, outcome, *_affected_text(run["affected"], x), diag, _assumptions(x)]


def _gformula_text(run: Mapping[str, Any]) -> tuple[str, list[str]]:
    from turbotab.core.time_varying import CITE

    lane, setting = run["lane"], run["setting"]
    x, y = _tick(lane.exposure), _tick(run["target"])
    lead = (f"the effect of {x} on {y} was estimated by the parametric g-formula "
            f"({CITE['robins_1986']}; {CITE['mcgrath']})")
    kinds = run["kinds"]
    each = _listing([f"{_tick(c)} ({'logistic' if kinds[c] == 'binary' else 'linear'})"
                     for c in lane.confounders])
    base = _listing(list(map(_tick, lane.baseline))) if lane.baseline else "none"
    drawn = (", a linear one drawn within its observed range"
             if any(kinds[c] == "normal" for c in lane.confounders) else "")
    models = ((f"Each time-varying confounder, {each}, was modeled given the baseline covariates "
               f"({base}), the confounders before it at that time point, every variable's previous "
               f"value, time and its square{drawn}; "
               if lane.confounders else
               f"With no time-varying confounder, ")
              + f"{y} at each time point was modeled by a pooled logistic model given {x}, the "
                f"confounders, the baseline covariates, the previous values, time and its square.")
    sim = (f"The risks of {y} by the last of {setting.time_points} time points had every unit "
           f"always, and never, been exposed were simulated for {run['n_sim']:,} units drawn from "
           f"the observed first time point (Monte Carlo error at most {run['mc']:.4f}), with 95% "
           f"intervals from {lane.bootstrap:,} bootstrap resamples of units (percentiles)"
           + (f"; units lost to follow-up ({_tick(lane.censoring)}) contribute the time points they "
              f"were seen, so the risks are those had no unit been lost, assuming loss depends only "
              f"on the measured history" if lane.censoring else "") + ".")
    check = ("The natural course's simulated risk was compared with the observed risk as a check "
             "of the models.")
    diag = "Positivity at each time point was read before any estimate was shown."
    return lead, [models, sim, *_affected_text(run["affected"], x), check, diag, _assumptions(x)]


__all__ = [
    "Diagnostics", "Estimates", "POPULATION", "Setting", "TIME_VARYING_READS",
    "TimeVaryingArtifact", "gformula_spec", "lane_options", "methods_paragraph",
    "population_block", "read_setting", "time_varying_stage", "weight_terms",
]

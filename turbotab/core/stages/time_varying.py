"""The ``time_varying`` stage: an exposure that changes over time, estimated by g-methods.

The routing is ``turbotab/core/time_varying.py`` and the computation ``models/time_varying.py``.
This stage reads the working table's analyzed rows (every eligible row under inference, BLUEPRINT
§12 ruling 3) and returns, in order:

1. **the setting**. These facts come from the values: the unit (the grain answer's column) and the
   settled time column (BLUEPRINT §14.1; numbers, dates or labels in a declared order), the number of
   time points, whether each unit's rows run from the first time point without a gap, how many
   units' rows end before the last time point (loss to follow-up), whether the exposure is 0 or 1
   and changes within units, and whether each covariate changes within units. The proposal splits
   the adjustment set's covariates and every confounder affected by prior exposure into
   time-varying confounders and baseline covariates, and names an excluded 0/1 column whose values
   read as a loss-to-follow-up indicator, if exactly one does. The Router reads the setting: an
   exposure fixed within every unit does not ask the question.
2. **the diagnostics**, before any estimate. For the weights: their distribution overall and at each
   time point, the stabilized mean's check, and each truncation option's effect on them. For both
   g-methods: positivity at each time point. For the g-formula: what the simulation will take,
   measured on these rows (V2 gate 5). They describe the exposure (and, under an event, the rows still
   at risk of loss, which the event ends); none relates the exposure to the outcome. With them, the
   key of what they were computed from (:func:`diagnostics_key`).
3. **the estimates**, only once the lane is complete *and* its declaration came after these very
   diagnostics: the truncation (or the simulation's size) carries the key of the diagnostics it was
   declared after, and an estimate is computed only when that is the key computed now. The estimates
   are the marginal structural model's coefficient with its CR2 interval clustered by unit, or the
   g-formula's risks under "always" and "never" with their bootstrap intervals, and the E-value of
   either (MODELING_SEQUENCE §0 ruling 10). Below the unit floor no interval is reported; when more
   than 1% of the bootstrap resamples cannot fit the g-formula's models, none is either.

The methods sentence is the §13 contract's clause (``contracts.paragraph``). It states what
was done and the diagnostics, never the estimates.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from turbotab.core.graph import Bundle, StageContext

SEED = 20261005  # the g-formula's Monte Carlo and bootstrap draws: fixed, so a rerun reproduces them
MEAN_FLAG = 0.1  # a stabilized mean farther than this from 1 is flagged (the app's convention)
# More than this share of bootstrap resamples whose models cannot be fit, and no interval is
# reported: a resample fails when the units drawn hold too few events, so the failures are not at
# random and an interval from the rest is too narrow. The app's convention, not a test.
BOOT_FAILED_MAX = 0.01


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
    # The one excluded 0/1 column whose values read as a loss-to-follow-up indicator (1 only on a
    # unit's last row, never with the event), else None. A proposal only: the lane's ``censoring``
    # is the user's answer.
    censoring: str | None = None


class Setting(_Model):
    exposure: str
    unit: str
    time_column: str
    time_kind: Literal["numbers", "dates", "labels"]
    time_points: int
    units: int
    rows: int
    late_units: int  # units whose first row is not the first time point
    gap_units: int  # units missing a time point between their first and last
    repeat_units: int  # units with two rows at one time point
    ending_early: int  # units whose rows end before the last time point (without the event)
    censoring_candidates: list[str]
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


class SimulationCost(_Model):
    """What the g-formula will take at the default size, measured on these rows, and the parts
    that scale with each count (``models.time_varying.gformula_cost``)."""

    simulations: int
    bootstrap: int
    seconds: float
    text: str
    fit_seconds: float
    unit_seconds: float  # per simulated unit, per strategy
    resample_seconds: float  # per bootstrap resample


class Diagnostics(_Model):
    weights: WeightSummary | None = None  # the product of exposure and censoring weights
    weights_by_time: list[WeightsAtTime] = []
    exposure_weights: WeightSummary | None = None
    censoring_weights: WeightSummary | None = None
    truncation: list[TruncationOption] = []
    positivity: list[PositivityAtTime] = []
    cost: SimulationCost | None = None
    concerns: list[str] = []


class Diagnosed(_Model):
    """What the diagnostics were computed for: the lane's parts and the key of those parts and the
    rows (:func:`diagnostics_key`). A declaration made after them carries the key."""

    key: str
    exposure: str
    method: str
    confounders: list[str]
    baseline: list[str]
    censoring: str | None
    pattern: str


class EstimateRow(_Model):
    label: str
    measure: str
    estimate: float
    ci_low: float | None = None
    ci_high: float | None = None
    se: float | None = None  # on the scale the interval is built on (log for a ratio)
    df: float | None = None  # the t reference's degrees of freedom (Bell–McCaffrey), if any
    p: float | None = None


class RiskCurve(_Model):
    strategy: Literal["always", "never", "natural", "observed"]
    label: str
    risks: list[float]  # by the end of each time point
    mc_se: float | None = None


class EValue(_Model):
    reads: str  # what ratio it reads, and how
    rr: float
    lo: float | None = None
    hi: float | None = None
    point: float
    ci: float | None = None  # None without an interval


class Estimates(_Model):
    rows: list[EstimateRow]
    curves: list[RiskCurve] = []
    e_value: EValue | None = None
    n_rows: int
    n_units: int
    simulations: int | None = None
    bootstrap: int | None = None
    failed_resamples: int | None = None
    concerns: list[str] = []  # what limits the estimates: no interval, and why


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
    diagnosed: Diagnosed | None = None
    estimates: Estimates | None = None
    withheld: str | None = None
    # The ways forward when the estimates are blocked and recorded (the surveyed population with no
    # design-based estimator, MS4), or withheld until the declaration is made again.
    exits: list[dict[str, Any]] = []
    relations: list[str] = []  # what the lane implies, as fired here ("because you chose …")
    methods: str = ""


TIME_VARYING_READS = ("purpose", "target", "task", "event", "grain", "repeat_kind", "unit",
                      "temporal", "estimand", "adjustment", "clusters", "time_varying", "lens",
                      "categorical", "roles", "roles_unconfirmed", "role_confirmations",
                      "reading_confirmations", "shape_confirmations", "missing", "survey",
                      # the date-reading repair's format, which a text time column is read by
                      "findings")

# MS4 (MODELING_SEQUENCE §0 ruling 6, §4 "population estimand without a design-based estimator"):
# the weights, the outcome model and the simulation read these rows as sampled, so under the
# surveyed population the estimates are blocked and recorded, the sample-only attestation the exit.
# The diagnostics relate no exposure to the outcome and describe these rows; they are shown as
# they are.
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
                                 if affected else []),
                   *(["concurrent_exposure_and_outcome"]
                     if lane.ordering != "exposure_precedes_outcome" else [])}
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


def _as_dates(values: pd.Series, declared: str | None) -> tuple[pd.Series | None, bool]:
    """``values`` as datetime64 when they are dates: datetimes as the working table holds them
    (object Timestamps once ``normalize_frame`` has passed them), or text that the app's own date
    reader (``turbotab.core.dates``, never pandas' lenient guess) reads in full; else None. And
    whether the text reads as dates two ways (month or day first), which only the date-reading
    repair settles."""
    if pd.api.types.is_datetime64_any_dtype(values):
        return pd.to_datetime(values), False
    if values.dtype != object:
        return None, False
    kind = pd.api.types.infer_dtype(values, skipna=True)
    if kind in ("datetime", "datetime64", "date"):
        return pd.to_datetime(values), False
    if kind == "string":
        from turbotab.core import dates

        reading = dates.read_series(values, declared, column=str(values.name))
        if reading.kind == "dates" and reading.parsed == reading.n:
            stamps = dates.to_timestamps(values, reading.formats)
            return (stamps if not stamps.isna().any() else None), False
        return None, reading.kind in ("ambiguous", "mixed")
    return None, False


def _time_points(values: pd.Series, levels: Sequence[str] | None,
                 declared: str | None = None) -> tuple[np.ndarray | None, str]:
    """Each row's time point as its rank (0 … K − 1), and what the column holds: dates by date;
    text labels by their declared order; numbers by value. None when text has no declared order,
    dates read two ways (``"ambiguous"``) or a time point is blank."""
    from turbotab.core.models.time_varying import time_index

    if values.isna().any():
        return None, "labels"
    stamps, ambiguous = _as_dates(values, declared)
    if stamps is not None:
        return time_index(stamps.to_numpy(dtype="datetime64[ns]").astype("int64")), "dates"
    if ambiguous:
        return None, "ambiguous"
    if levels:
        position = {str(v): i for i, v in enumerate(levels)}
        mapped = values.astype(str).map(position)
        if mapped.isna().any():
            return None, "labels"
        return time_index(mapped.to_numpy(dtype=float)), "labels"
    if pd.api.types.is_numeric_dtype(values):
        return time_index(values.to_numpy()), "numbers"
    return None, "labels"


def candidate_columns(state: Any) -> list[str]:
    """The covariates the lane may adjust for: the adjustment question's covariates, every
    confounder affected by prior exposure, and total energy (settled energy roles)."""
    from turbotab.core.estimand import asked_covariates, predictor_roles
    from turbotab.core.time_varying import affected_confounders, exposure_of

    exposure = exposure_of(state)
    energy = [c for c, r in predictor_roles(state).items() if r == "energy"]
    out = [*asked_covariates(state), *affected_confounders(state), *energy]
    return [c for c in dict.fromkeys(out) if c != exposure]


def _loss_indicators(frame: pd.DataFrame, columns: Sequence[str], last: np.ndarray,
                     event: np.ndarray) -> list[str]:
    """The columns whose values read as a loss-to-follow-up indicator: 0 or 1 in every row, 1 in
    some, and every 1 on its unit's last row and never on a row with the event."""
    out = []
    for c in columns:
        v = pd.to_numeric(frame[c], errors="coerce")
        if v.isna().any() or not set(v.unique()) <= {0, 1} or not (v == 1).any():
            continue
        ones = (v == 1).to_numpy()
        if (ones & ~last).any() or (ones & event).any():
            continue
        out.append(str(c))
    return out


def read_setting(ctx: StageContext, exposure: str, affected: Sequence[str],
                 structure: Any) -> tuple[pd.DataFrame, Setting]:
    """The analyzed rows sorted by unit and time, with ``__unit``, ``__time`` and the outcome coded,
    and what their values say."""
    from turbotab.core.estimand import derived_roles, predictor_roles
    from turbotab.core.models.pipeline import modeling_frame
    from turbotab.core.models.time_varying import check_schedule
    from turbotab.core.stages.data import open_store
    from turbotab.core.stages.modeling import coded_outcome, read_assignment
    from turbotab.core.stages.working import (declared_date_formats, declared_levels,
                                              effective_grain, time_column)

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
    taken = {unit, order, exposure, target, *covariates}
    excluded = [c for c, r in (getattr(state, "roles", None) or {}).items()
                if r == "excluded" and c not in taken]
    rows = read_assignment(ctx.inputs["split"]).index.to_numpy()
    wanted = [c for c in dict.fromkeys([unit, order, exposure, target, *covariates,
                                        *([censoring] if censoring else [])]) if c]
    with open_store(ctx) as store:
        missing = [c for c in wanted if c not in store.columns]
        if missing:
            raise NotReady(f"The working table has no column named {', '.join(missing)}.")
        excluded = [c for c in excluded if c in store.columns and c not in wanted]
        frame = modeling_frame(store, [*wanted, *excluded], rows, outcome=target)
    t, time_kind = _time_points(frame[order], declared_levels(state, order),
                                declared_date_formats(state).get(order))
    if t is None and time_kind == "ambiguous":
        raise NotReady(f"`{order}`'s dates read two ways, month first or day first, so their order "
                       f"is not known; the date-reading repair says which.")
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
    # Loss to follow-up, as the rows show it: a unit whose last row is before the last time point
    # (and, for an event, has no event) was lost.
    times = frame.groupby("__unit", sort=False)["__time"]
    last = (frame["__time"] == times.transform("max")).to_numpy()
    event = (yv == 1).to_numpy() if kind == "event" else np.zeros(len(frame), dtype=bool)
    early = last & (frame["__time"].to_numpy() < sched["time_points"] - 1) & ~event
    ending_early = int(early.sum())
    indicators = _loss_indicators(frame, excluded, last, event) if kind in ("event", "measure") \
        else []

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
        baseline=[c.column for c in candidates if not c.varies and c.adjusted and not c.affected],
        censoring=indicators[0] if len(indicators) == 1 else None)
    setting = Setting(
        exposure=exposure, unit=unit, time_column=order, time_kind=time_kind,
        time_points=sched["time_points"], units=sched["units"], rows=int(len(frame)),
        late_units=sched["late"], gap_units=sched["gaps"], repeat_units=sched["repeats"],
        ending_early=ending_early, censoring_candidates=indicators, exposure_binary=binary,
        exposure_levels=levels, exposure_varies=changers > 0, exposure_changers=changers,
        exposure_stops=stops, outcome_kind=kind, units_after_event=after, candidates=candidates,
        varies=varies, proposal=proposal)
    return frame, setting


def diagnostics_key(frame: pd.DataFrame, setting: Setting, lane: Any) -> str:
    """What the lane's diagnostics were computed from: its parts (``time_varying.lane_parts``) and
    the values of every column they read, row by row (the unit, the time point, the exposure, the
    covariates, the loss indicator and the outcome). Two runs share a key exactly when their
    diagnostics are the same computation on the same values."""
    from turbotab.core.time_varying import lane_parts

    columns = list(dict.fromkeys(["__unit", "__time", setting.exposure, *lane.confounders,
                                  *lane.baseline, *([lane.censoring] if lane.censoring else []),
                                  "__y"]))
    digest = hashlib.sha256(json.dumps(lane_parts(lane), sort_keys=True).encode())
    digest.update(pd.util.hash_pandas_object(frame[columns], index=False).to_numpy().tobytes())
    return digest.hexdigest()[:24]


def _diagnosed(frame: pd.DataFrame, setting: Setting, lane: Any) -> Diagnosed:
    from turbotab.core.time_varying import lane_parts

    return Diagnosed(key=diagnostics_key(frame, setting, lane), **lane_parts(lane))


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
        dates = (f" `{setting.time_column}` holds dates, and a time point here is one date the "
                 f"units share; when each unit has its own dates, a column that numbers each "
                 f"unit's visits is the time column (the repeats question)."
                 if setting.time_kind == "dates" else "")
        raise NotReady(
            f"Each unit needs one row at each time point from the first, with none missing in "
            f"between: {setting.late_units:,} units start later, {setting.gap_units:,} skip a time "
            f"point and {setting.repeat_units:,} have two rows at one.{dates}")
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


def _loss_concern(setting: Setting, lane: Any) -> str | None:
    """Units lost to follow-up with no indicator declared: the assumption, and the column the
    values read as one."""
    n = setting.ending_early
    if lane.censoring or not n:
        return None
    end = " without the event" if setting.outcome_kind == "event" else ""
    found = setting.proposal.censoring
    named = (f" `{found}` reads as one: 1 only on a unit's last row, never with the event."
             if found else "")
    return (f"{n:,} {'unit' if n == 1 else 'units'}' rows end before the last time point{end}, and "
            f"no loss-to-follow-up indicator is declared: the estimate assumes "
            f"{'its' if n == 1 else 'their'} loss is unrelated to the outcome.{named}")


def _floor() -> int:
    from turbotab.core.models.inference import min_clusters

    return min_clusters()


def _few_units(setting: Setting, how: str) -> str:
    return (f"`{setting.unit}` has {setting.units:,} units, fewer than the {_floor():,} {how}: no "
            f"interval or p-value is reported.")


def _stale(lane: Any, what: str) -> tuple[str, list[dict[str, Any]]]:
    """The estimate withheld because these diagnostics are not the ones the declaration came after,
    and the exit that declares it again, now that they are shown."""
    from turbotab.core.time_varying import TRUNCATION_WORDS

    spec = {"kind": "set_time_varying", **lane.model_dump(mode="json")}
    spec["diagnostics_seen"] = None
    if lane.method == "msm_iptw":
        declared = "truncation"
        label = f"Keep the weights {TRUNCATION_WORDS[lane.truncation]}, having read these"
    else:
        declared = "simulation's size"
        label = (f"Keep {lane.simulations:,} simulated units and {lane.bootstrap:,} resamples, "
                 f"having read these")
    return (f"These {what} are not the ones read when the {declared} was declared: the data, or "
            f"an answer they read, changed since. Read them, then declare it again; no estimate "
            f"is computed before that.", [{"label": label, "decision": spec}])


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
                                    kind="first" if lane.pattern == "initiation" else "all",
                                    label=f"`{setting.exposure}`")
        censor_w = None
        if lane.censoring:
            d["__c"] = pd.to_numeric(frame[lane.censoring]).to_numpy(float)
            # A row with the event ends follow-up, so it is not at risk of being lost after it.
            at_risk = (d["__y"] == 0).to_numpy() if setting.outcome_kind == "event" else None
            censor_w = tv.censoring_weights(d, id="__unit", time="__time", indicator="__c",
                                            numerator=terms["censoring_numerator"],
                                            denominator=terms["censoring_denominator"],
                                            at_risk=at_risk,
                                            label=f"loss to follow-up (`{lane.censoring}`)")
    except tv.NotEstimable as exc:
        raise NotReady(f"The weights cannot be estimated: {exc}.") from None
    w = exposure_w.weights * (censor_w.weights if censor_w is not None else 1.0)
    level = tv.TRUNCATIONS[lane.truncation] if lane.truncation else None
    used = tv.truncate(w, level) if lane.truncation else w
    t = d["__time"].to_numpy()
    concerns = []
    summary = tv.summarize(w)
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
    lost = _loss_concern(setting, lane)
    if lost:
        concerns.append(lost)
    if setting.units < _floor():
        concerns.append(_few_units(setting, "a cluster-robust interval needs"))
    diagnostics = Diagnostics(
        weights=_summary(summary),
        weights_by_time=[WeightsAtTime(**x) for x in tv.by_time(w, t)],
        exposure_weights=_summary(tv.summarize(exposure_w.weights)),
        censoring_weights=(_summary(tv.summarize(censor_w.weights))
                           if censor_w is not None else None),
        truncation=[TruncationOption(**o, summary=_summary(tv.summarize(tv.truncate(
            w, tv.TRUNCATIONS[o["key"]]))), chosen=o["key"] == lane.truncation)
            for o in TRUNCATION_OPTIONS],
        positivity=[PositivityAtTime(**p) for p in positivity], concerns=concerns)
    diagnosed = _diagnosed(frame, setting, lane)
    base = {**base, "diagnostics": diagnostics, "diagnosed": diagnosed}
    present = {"weight_diagnostics_before_estimates", "positivity_by_time_point", "estimand"}
    if lane.censoring:
        present.add("censoring_weighted")
    if lost:
        present.add("loss_assumed_independent")
    declared = lane.truncation is not None and lane.diagnostics_seen == diagnosed.key
    run = {"lane": lane, "setting": setting, "names": names, "summary": tv.summarize(used),
           "family": family, "affected": affected, "target": ctx.state.target,
           "declared_after": declared, "floor": _floor()}
    if lane.truncation is None:
        return TimeVaryingArtifact(
            **base, relations=_fired(lane, present),
            withheld="Read the weights and positivity first: no estimate is computed until the "
                     "truncation is declared (none is an answer).",
            methods=methods_paragraph(run))
    if not declared:
        withheld, exits = _stale(lane, "weights")
        return TimeVaryingArtifact(**base, relations=_fired(lane, present), withheld=withheld,
                                   exits=exits, methods=methods_paragraph(run))
    blocked = population_block(ctx.state)
    if blocked is not None:
        return TimeVaryingArtifact(**base, relations=_fired(lane, present), reason=blocked[0],
                                   exits=blocked[1], methods=methods_paragraph(run))
    ctx.progress(0.6, "Fitting the weighted outcome model")
    summary_term = "__cum" if lane.summary == "cumulative" else "__a"
    d["__cum"] = d.groupby("__unit", sort=False)["__a"].cumsum().to_numpy(float)
    V = [n for c in lane.baseline for n in names[c]]
    X, xnames = tv.design(d, [summary_term, *V, tv.TIME, tv.TIME2])
    yv = d["__y"].to_numpy(float)
    try:
        fit = tv.fit_msm(X, xnames, yv, used, d["__unit"], family, variance="CR2")
    except tv.NotEstimable as exc:
        events = (f" (`{ctx.state.target}` has {int(yv.sum()):,} events in {len(yv):,} rows)"
                  if family == "binomial" else "")
        return TimeVaryingArtifact(
            **base, relations=_fired(lane, present), methods=methods_paragraph(run),
            reason=f"The weighted outcome model cannot be fit: {exc}{events}.")
    r = fit.row(summary_term)
    # Below the unit floor the estimate stands alone: no interval, SE, df or p-value.
    interval = fit.n_units >= _floor()
    lo, hi = (r["ci_low"], r["ci_high"]) if interval else (None, None)
    test: dict[str, Any] = {"se": r["se"], "df": r["df"], "p": r["p"]} if interval else {}
    est_concerns = [] if interval else [_few_units(setting, "a cluster-robust interval needs")]
    per = (f"per time point of `{setting.exposure}` so far" if lane.summary == "cumulative"
           else f"for current `{setting.exposure}`")
    if family == "binomial":
        est = math.exp(r["estimate"])
        lo, hi = (math.exp(lo), math.exp(hi)) if interval else (None, None)
        row = EstimateRow(label=f"Odds ratio {per}", measure="odds_ratio", estimate=est,
                          ci_low=lo, ci_high=hi, **test)
        ev, reads = _ratio_e_value(setting, d, est, lo, hi)
    else:
        sd = float(np.std(yv, ddof=1))
        row = EstimateRow(label=f"Mean difference {per}", measure="mean_difference",
                          estimate=r["estimate"], ci_low=lo, ci_high=hi, **test)
        ev = tv.e_value_md(r["estimate"] / sd, r["se"] / sd if interval else None)
        reads = "the mean difference in outcome SDs, as exp(0.91 d)"
    estimates = Estimates(rows=[row], e_value=EValue(reads=reads, **ev), n_rows=fit.n_rows,
                          n_units=fit.n_units, concerns=est_concerns)
    present |= {"intervals_by_unit", "unmeasured_confounding_sensitivity"}
    return TimeVaryingArtifact(**base, estimates=estimates, relations=_fired(lane, present),
                               methods=methods_paragraph(run))


def _ratio_e_value(setting: Setting, d: pd.DataFrame, est: float, lo: float | None,
                   hi: float | None) -> tuple[dict[str, Any], str]:
    """The E-value of the pooled logistic model's odds ratio. For an event at each time point that
    model is a discrete-time hazard model, so the ratio is read as a hazard ratio and the outcome's
    rarity is its cumulative risk by the end of follow-up (VanderWeele & Ding 2017); for a repeated
    yes/no outcome, an odds ratio whose rarity is the share of rows."""
    from turbotab.core.models import time_varying as tv

    if setting.outcome_kind == "event":
        risk = tv.cumulative_risk(d["__time"], d["__y"])[-1]
        rare = risk < tv.RARE
        reads = (f"the pooled logistic odds ratio as a hazard ratio, the outcome rare by the end of "
                 f"follow-up ({risk:.1%} cumulative risk, below 15%)" if rare else
                 f"the pooled logistic odds ratio as a hazard ratio of an outcome common by the end "
                 f"of follow-up ({risk:.1%} cumulative risk), as (1 − 0.5^√HR)/(1 − 0.5^√(1/HR))")
        return tv.e_value_hr(est, lo, hi, rare), reads
    share = float(np.nanmean(d["__y"].to_numpy(float)))
    rare = share < tv.RARE
    reads = (f"the odds ratio as a risk ratio (the outcome is rare: {share:.1%} of rows, below 15%)"
             if rare else f"the square root of the odds ratio as a risk ratio (the outcome is "
                          f"common: {share:.1%} of rows)")
    return tv.e_value_or(est, lo, hi, rare), reads


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
    from turbotab.core.models.cost import duration
    from turbotab.core.time_varying import BOOTSTRAP, SIMULATIONS

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
                                    kind="first" if lane.pattern == "initiation" else "all",
                                    label=f"`{setting.exposure}`")
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
    lost = _loss_concern(setting, lane)
    if lost:
        concerns.append(lost)
    resampled = setting.units >= _floor()
    if not resampled:
        concerns.append(_few_units(setting, "an interval resampling whole units needs"))
    present = {"positivity_by_time_point", "positivity_and_time_before_estimates", "estimand"}
    if lost:
        present.add("loss_assumed_independent")
    diagnosed = _diagnosed(frame, setting, lane)
    if lane.simulations is None or lane.bootstrap is None:
        ctx.progress(0.2, "Timing the simulation on these rows")
        cost = None
        reason = None
        try:
            timed = tv.gformula_cost(d, spec, n_sim=SIMULATIONS,
                                     reps=BOOTSTRAP if resampled else 0, seed=SEED)
            cost = SimulationCost(simulations=SIMULATIONS, bootstrap=BOOTSTRAP,
                                  text=duration(timed["seconds"]), **timed)
        except tv.NotEstimable as exc:
            reason = f"The g-formula's models cannot be fit: {exc}."
        diagnostics = Diagnostics(positivity=[PositivityAtTime(**p) for p in positivity],
                                  cost=cost, concerns=concerns)
        return TimeVaryingArtifact(
            **base, diagnostics=diagnostics, diagnosed=diagnosed, reason=reason,
            relations=_fired(lane, present),
            withheld=None if reason else (
                "Read positivity and what the simulation will take first: no risk is simulated "
                "until its size is declared."))
    diagnostics = Diagnostics(positivity=[PositivityAtTime(**p) for p in positivity],
                              concerns=concerns)
    base = {**base, "diagnostics": diagnostics, "diagnosed": diagnosed}
    if lane.diagnostics_seen != diagnosed.key:
        withheld, exits = _stale(lane, "diagnostics")
        return TimeVaryingArtifact(**base, relations=_fired(lane, present), withheld=withheld,
                                   exits=exits)
    blocked = population_block(ctx.state)
    if blocked is not None:
        return TimeVaryingArtifact(**base, relations=_fired(lane, present), reason=blocked[0],
                                   exits=blocked[1])
    ctx.progress(0.1, "Simulating the strategies")
    try:
        result = tv.gformula(d, spec, n_sim=lane.simulations, seed=SEED)
    except tv.NotEstimable as exc:
        return TimeVaryingArtifact(**base, relations=_fired(lane, present),
                                   reason=f"The g-formula's models cannot be fit: {exc}.")
    boot = _bootstrap(ctx, d, spec, lane) if resampled else None
    iv = (boot or {}).get("intervals")

    def ci(key: str) -> dict[str, float | None]:
        lo, hi = iv[key] if iv and key in iv else (None, None)
        return {"ci_low": lo, "ci_high": hi}

    always, never = result.final("always"), result.final("never")
    rows = [EstimateRow(label="Risk if always exposed", measure="risk", estimate=always,
                        **ci("always")),
            EstimateRow(label="Risk if never exposed", measure="risk", estimate=never,
                        **ci("never")),
            EstimateRow(label="Risk difference (always − never)", measure="risk_difference",
                        estimate=always - never, **ci("difference")),
            EstimateRow(label="Risk ratio (always / never)", measure="risk_ratio",
                        estimate=always / never if never > 0 else math.nan, **ci("ratio"))]
    curves = [RiskCurve(strategy="always", label="Always exposed", risks=result.risks["always"],
                        mc_se=result.mc_se["always"]),
              RiskCurve(strategy="never", label="Never exposed", risks=result.risks["never"],
                        mc_se=result.mc_se["never"]),
              RiskCurve(strategy="natural", label="Natural course (simulated)",
                        risks=result.risks["natural"], mc_se=result.mc_se["natural"]),
              RiskCurve(strategy="observed", label="Observed", risks=result.observed)]
    ev = None
    ratio = ci("ratio")
    if never > 0 and always > 0:
        lo, hi = ratio["ci_low"], ratio["ci_high"]
        finite = lo is not None and hi is not None and math.isfinite(lo) and math.isfinite(hi)
        ev = EValue(reads="the risk ratio", **tv.e_value(always / never, lo if finite else None,
                                                         hi if finite else None))
    est_concerns = ([] if resampled else
                    [_few_units(setting, "an interval resampling whole units needs")])
    if boot is not None and boot["concern"]:
        est_concerns.append(boot["concern"])
    estimates = Estimates(rows=rows, curves=curves, e_value=ev, n_rows=int(len(d)),
                          n_units=setting.units, simulations=result.n_sim,
                          bootstrap=boot["reps"] if boot else None,
                          failed_resamples=boot["failed"] if boot else None,
                          concerns=est_concerns)
    present |= {"risks_under_always_and_never"}
    if boot is not None and iv:
        present.add("intervals_by_unit")
    if ev is not None:
        present.add("unmeasured_confounding_sensitivity")
    run = {"lane": lane, "setting": setting, "names": names, "kinds": kinds,
           "affected": affected, "target": ctx.state.target, "n_sim": result.n_sim,
           "mc": max(result.mc_se["always"], result.mc_se["never"]), "boot": boot,
           "floor": _floor()}
    return TimeVaryingArtifact(**base, estimates=estimates, relations=_fired(lane, present),
                               methods=methods_paragraph(run))


def bootstrap_verdict(reps: int, failed: int) -> tuple[bool, str | None]:
    """Whether an interval is reported from ``reps`` resamples of which ``failed`` could not fit
    the models, and the concern that says so. A resample fails when the units drawn hold too few
    events for the models, so the failures are the resamples with the fewest events: an interval
    from the rest is too narrow. Up to :data:`BOOT_FAILED_MAX` of them it is reported and the
    count named; beyond, none is."""
    if not failed:
        return True, None
    if failed <= BOOT_FAILED_MAX * reps and failed < reps:
        return True, (f"{failed:,} of {reps:,} bootstrap resamples could not fit the models and are "
                      f"left out: a resample fails when the units drawn hold too few events, so the "
                      f"interval from the other {reps - failed:,} may be slightly too narrow.")
    return False, (f"{failed:,} of {reps:,} bootstrap resamples could not fit the models: a "
                   f"resample fails when the units drawn hold too few events for them, so the "
                   f"failures are not at random and an interval from the rest would be too narrow. "
                   f"No interval is reported; a model with fewer terms, or more units, is the way "
                   f"to one.")


def _bootstrap(ctx: StageContext, d: pd.DataFrame, spec: Any, lane: Any) -> dict[str, Any]:
    """The percentile bootstrap over units, in chunks so the progress bar moves; the resamples
    that cannot fit the models counted, and the interval reported only as
    :func:`bootstrap_verdict` allows."""
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
    failed = sum(p["failed"] for p in draws)
    reported, concern = bootstrap_verdict(reps, failed)
    intervals = ({k: (float(np.nanquantile(v, 0.025)), float(np.nanquantile(v, 0.975)))
                  for k, v in merged.items() if v} if reported else None)
    return {"intervals": intervals, "reps": reps, "failed": failed, "concern": concern}


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


def _loss_text(setting: Setting, lane: Any) -> str | None:
    """With no loss-to-follow-up indicator declared, what the rows show and the assumption."""
    n = setting.ending_early
    if lane.censoring or not n:
        return None
    end = " without the event" if setting.outcome_kind == "event" else ""
    return (f"No loss-to-follow-up indicator was declared: {n:,} "
            f"{'unit' if n == 1 else 'units'}' rows end before the last time point{end}, and the "
            f"estimate assumes {'its' if n == 1 else 'their'} loss is unrelated to the outcome.")


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
    if setting.units >= run["floor"]:
        variance = (f"with a CR2 variance clustered by {unit} and Bell–McCaffrey degrees of freedom "
                    f"({CITE['bell_mccaffrey']}) that treats the weights as known, so its interval "
                    f"is conservative ({CITE['hernan_2000']})")
    else:
        variance = (f"and no interval: {unit} has {setting.units:,} units, fewer than the "
                    f"{run['floor']:,} a cluster-robust interval needs")
    outcome = f"The outcome model was a weighted {kind} model of {y} on {on}, {variance}."
    diag = ("The truncation was declared after the weights' distribution and positivity at each "
            "time point were read, and no estimate was computed before it."
            if run["declared_after"] else
            "The truncation is declared after the weights' distribution and positivity at each "
            "time point are read; no estimate is computed before it.")
    loss = _loss_text(setting, lane)
    return lead, [weights, outcome, *([loss] if loss else []), *_affected_text(run["affected"], x),
                  diag, _assumptions(x)]


def _interval_text(lane: Any, setting: Setting, boot: Mapping[str, Any] | None, floor: int) -> str:
    if boot is None:
        return (f"; no interval is reported: {setting.units:,} units are fewer than the {floor:,} an "
                f"interval resampling whole units needs")
    reps, failed = int(boot["reps"]), int(boot["failed"])
    if boot["intervals"] is None:
        return (f"; no interval is reported, because {failed:,} of {reps:,} bootstrap resamples of "
                f"units could not fit the models")
    if failed:
        return (f", with 95% intervals from the {reps - failed:,} of {reps:,} bootstrap resamples of "
                f"units whose models could be fit (percentiles)")
    return f", with 95% intervals from {reps:,} bootstrap resamples of units (percentiles)"


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
    n = setting.ending_early
    if lane.censoring:
        lost = (f"; units lost to follow-up ({_tick(lane.censoring)}) contribute the time points "
                f"they were seen, so the risks are those had no unit been lost, assuming loss "
                f"depends only on the measured history")
    elif n:
        lost = (f"; {n:,} {'unit' if n == 1 else 'units'}' rows end before the last time point "
                f"without the event (no loss-to-follow-up indicator was declared), so the risks are "
                f"those had no unit been lost, assuming loss depends only on the measured history")
    else:
        lost = ""
    sim = (f"The risks of {y} by the last of {setting.time_points} time points had every unit "
           f"always, and never, been exposed were simulated for {run['n_sim']:,} units drawn from "
           f"the observed first time point (Monte Carlo error at most {run['mc']:.4f})"
           + _interval_text(lane, setting, run["boot"], run["floor"]) + lost + ".")
    check = ("The natural course's simulated risk was compared with the observed risk as a check "
             "of the models.")
    diag = ("The simulation's size was declared after positivity at each time point and what the "
            "simulation would take were read, and no risk was simulated before it.")
    return lead, [models, sim, *_affected_text(run["affected"], x), check, diag, _assumptions(x)]


__all__ = [
    "BOOT_FAILED_MAX", "Diagnosed", "Diagnostics", "Estimates", "POPULATION", "Setting",
    "SimulationCost", "TIME_VARYING_READS", "TimeVaryingArtifact", "bootstrap_verdict",
    "diagnostics_key", "gformula_spec", "lane_options", "methods_paragraph", "population_block",
    "read_setting", "time_varying_stage", "weight_terms",
]

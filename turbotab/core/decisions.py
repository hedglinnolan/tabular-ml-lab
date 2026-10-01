"""Decisions, event-sourced (docs/turbotab-next/BLUEPRINT.md §3).

Every user answer is a :data:`Decision`. Decisions are appended to
``decisions.jsonl`` as :class:`DecisionRecord` lines and never edited; the
project's :class:`ProjectState` is a fold over that log.

Adding a decision kind is two steps in one place, never an if-chain:

1. define its model (a ``kind`` literal plus its payload) and add it to the
   :data:`Decision` union below, so the API and the log can parse it;
2. call :func:`register_kind` with the slot it writes (and, when the payload has
   more than one field, a ``value=`` function; when the answer is about another
   slot's value, a ``holds=`` condition), plus :func:`register_validator` for
   any refusal that needs project context.

Revert semantics (Tier A, pinned by ``tests/test_decisions.py``): a reverted
record is folded as though it had never been made, so ``revert(X)`` restores
X's slot to the value it had before X (when X was that slot's latest write);
reverting a revert reinstates what it reverted. A revert may target only an
earlier record that writes a slot, or an earlier revert; anything else is
refused with ``unknown_decision`` and nothing is recorded.

Conditional writes (Tier A, same file): a ``set_task`` answers for the column
it names, so the ``task`` slot holds the latest ``set_task`` for the *current*
target and is unset while no answer names it. Choosing another target never
carries one column's task over to the next.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Callable, Iterable, Literal, Mapping, Sequence, Union

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, TypeAdapter, field_validator

try:  # POSIX only; on Windows the in-process lock is the whole of it.
    import fcntl
except ImportError:  # pragma: no cover - exercised on Windows only
    fcntl = None  # type: ignore[assignment]

log = logging.getLogger(__name__)

Lens = Literal["metabolomics", "genomics", "dietary", "clinical", "survey"]
Task = Literal["regression", "binary", "multiclass"]
Purpose = Literal["prediction", "inference"]

ROW_ID = "__row_id"  # the stable row identity the datastore adds (BLUEPRINT §2)


def _kind_is_required(schema: dict[str, Any]) -> None:
    """Keep ``kind`` required in the JSON schema even though it has a default.

    The default lets Python write ``SetLens(lenses=[...])``; the schema must
    still say ``kind`` is always present, or generated TypeScript types make the
    discriminant optional and narrowing breaks.
    """
    required = schema.setdefault("required", [])
    if "kind" not in required:
        required.insert(0, "kind")


class _DecisionModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        json_schema_extra=_kind_is_required,
        json_schema_serialization_defaults_required=True,
    )


class SetLens(_DecisionModel):
    kind: Literal["set_lens"] = "set_lens"
    lenses: list[Lens] = Field(min_length=1)

    @field_validator("lenses")
    @classmethod
    def _unique(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("each lens may be named only once")
        return value


class SetTarget(_DecisionModel):
    kind: Literal["set_target"] = "set_target"
    column: str = Field(min_length=1)


class SetTask(_DecisionModel):
    """An override of task detection, for the outcome column it names.

    It stands only while ``column`` is the target (see :func:`fold`).
    """

    kind: Literal["set_task"] = "set_task"
    column: str = Field(min_length=1)
    task: Task


class SetPurpose(_DecisionModel):
    kind: Literal["set_purpose"] = "set_purpose"
    purpose: Purpose


class Revert(_DecisionModel):
    kind: Literal["revert"] = "revert"
    decision_id: str = Field(min_length=1)


# ── M1 kinds (docs/turbotab-next/M1_CONTRACT.md) ─────────────────────────────
# The shapes are fixed here so every M1 agent builds on one definition; their
# validators, sentences and the stages that read them are the agents' work.

Role = Literal["identifier", "exposure", "energy", "covariate", "design", "flag", "time", "excluded"]
EnergyMethod = Literal["none", "standard", "residual", "density_multivariate", "density", "partition"]
MissingStrategy = Literal["complete_case", "impute"]


class _Value(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True,
                              json_schema_serialization_defaults_required=True)


class RangeByLevel(_Value):
    """Different plausible ranges per level of another column (e.g. by sex)."""

    column: str = Field(min_length=1)
    ranges: dict[str, tuple[float | None, float | None]]


class ExclusionRule(_Value):
    kind: Literal["range"] = "range"
    column: str = Field(min_length=1)
    low: float | None = None
    high: float | None = None
    by: RangeByLevel | None = None
    reason: str = Field(min_length=1)


class EnergyAdjustment(_Value):
    method: EnergyMethod
    energy_column: str | None = None
    nutrients: list[str] = Field(default_factory=list)
    log_transform: bool = False
    strata: str | None = None


class SplitSpec(_Value):
    holdout: float = Field(ge=0.0, le=0.4)
    seed: int = 0
    folds: int = Field(default=5, ge=2, le=10)

    @field_validator("holdout")
    @classmethod
    def _holdout(cls, value: float) -> float:
        if value != 0.0 and value < 0.1:
            raise ValueError("a holdout is 0 (cross-validation only) or between 0.1 and 0.4")
        return value


MAX_BOOT = 500


class SubstitutionSpec(_Value):
    donor: str = Field(min_length=1)
    recipient: str = Field(min_length=1)
    step_kcal: float = Field(default=100.0, gt=0)
    n_boot: int = Field(default=0, ge=0, le=MAX_BOOT)


class MissingSpec(_Value):
    """The missing-values answer: columns left out of the predictors, then a strategy for the rest."""

    strategy: MissingStrategy
    drop_columns: list[str] = Field(default_factory=list)
    categorical: Literal["missing_category", "impute"] = "impute"
    indicators: bool = False


class SetRoles(_DecisionModel):
    """The confirmed reading of every column except the outcome."""

    kind: Literal["set_roles"] = "set_roles"
    roles: dict[str, Role] = Field(min_length=1)


class SetEnergyAdjustment(_DecisionModel):
    kind: Literal["set_energy_adjustment"] = "set_energy_adjustment"
    method: EnergyMethod
    energy_column: str | None = None
    nutrients: list[str] = Field(default_factory=list)
    log_transform: bool = False
    strata: str | None = None


class SetExclusions(_DecisionModel):
    """Row exclusions; an empty list is the answer "keep every row"."""

    kind: Literal["set_exclusions"] = "set_exclusions"
    rules: list[ExclusionRule]


class SetMissing(_DecisionModel):
    """How missing predictor values are handled.

    ``drop_columns`` leave the predictors first (a mostly blank column that means "not asked"),
    and the strategy applies to the predictors that remain; the cohort and the design read both.
    """

    kind: Literal["set_missing"] = "set_missing"
    strategy: MissingStrategy
    drop_columns: list[str] = Field(default_factory=list)
    # M2 — routed by dtype and mechanism (ROADMAP lockbox constitution §07): binary/categorical
    # blanks may become their own "Missing" level, which keeps the signal when a blank means
    # "not asked"; numeric columns may carry a missing indicator beside the in-fold imputation.
    categorical: Literal["missing_category", "impute"] = "impute"
    indicators: bool = False

    @field_validator("drop_columns")
    @classmethod
    def _unique(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("each column may be left out only once")
        return value


class SetSplit(_DecisionModel):
    kind: Literal["set_split"] = "set_split"
    holdout: float = Field(ge=0.0, le=0.4)
    seed: int = 0
    folds: int = Field(default=5, ge=2, le=10)


class SelectModels(_DecisionModel):
    kind: Literal["select_models"] = "select_models"
    models: list[str] = Field(min_length=1)

    @field_validator("models")
    @classmethod
    def _unique(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("each model family may be named only once")
        return value


class SetSubstitution(_DecisionModel):
    """The substitution to draw; ``n_boot > 0`` adds a band from that many bootstrap refits."""

    kind: Literal["set_substitution"] = "set_substitution"
    donor: str = Field(min_length=1)
    recipient: str = Field(min_length=1)
    step_kcal: float = Field(default=100.0, gt=0)
    n_boot: int = Field(default=0, ge=0, le=MAX_BOOT)


# ── M2 kinds (docs/turbotab-next/M2_CONTRACT.md) ─────────────────────────────
# The opening sequence (OPENING_SEQUENCE.md), the seal, and findings with dispositions.

Orientation = Literal["sample_major", "feature_major"]
RepeatKind = Literal["repeats", "time_points"]
AggregationMethod = Literal["mean", "first", "last", "change"]
FindingAction = Literal["applied", "deferred", "dismissed"]


class GrainSpec(_Value):
    grain: Literal["one_row_per_unit", "repeated"]
    id_column: str | None = None  # the column naming the unit (person, sample) when repeated


class RepeatSpec(_Value):
    repeat_kind: RepeatKind
    time_column: str | None = None


class AggregationSpec(_Value):
    method: AggregationMethod
    outcome: Literal["mean", "first", "last"] | None = None  # when the outcome varies within a unit


class TemporalSpec(_Value):
    temporal: bool
    time_column: str | None = None


class FindingDisposition(_Value):
    action: FindingAction
    option: str | None = None  # the repair option applied
    params: dict[str, Any] = Field(default_factory=dict)
    to: str | None = None  # the question key a deferral resurfaces at
    reason: str | None = None  # why it was dismissed, when given


class SetOrientation(_DecisionModel):
    """Which way round an assay table is; feature-major is transposed before any diagnosis."""

    kind: Literal["set_orientation"] = "set_orientation"
    orientation: Orientation


class SetEvent(_DecisionModel):
    """Which level of a binary outcome is the event. Stands only while ``column`` is the target."""

    kind: Literal["set_event"] = "set_event"
    column: str = Field(min_length=1)
    level: str = Field(min_length=1)


class SetGrain(_DecisionModel):
    kind: Literal["set_grain"] = "set_grain"
    grain: Literal["one_row_per_unit", "repeated"]
    id_column: str | None = None


class SetRepeatKind(_DecisionModel):
    kind: Literal["set_repeat_kind"] = "set_repeat_kind"
    repeat_kind: RepeatKind
    time_column: str | None = None


class SetUnit(_DecisionModel):
    """When a unit repeats: is one row of the analysis a unit (combined) or a record?"""

    kind: Literal["set_unit"] = "set_unit"
    unit: Literal["unit", "row"]


class SetAggregation(_DecisionModel):
    kind: Literal["set_aggregation"] = "set_aggregation"
    method: AggregationMethod
    outcome: Literal["mean", "first", "last"] | None = None


class SetTemporal(_DecisionModel):
    kind: Literal["set_temporal"] = "set_temporal"
    temporal: bool
    time_column: str | None = None


class OpenSeal(_DecisionModel):
    """Open the held-out rows: once, at the end. Held-out scores are withheld until then."""

    kind: Literal["open_seal"] = "open_seal"


class ApplyRepair(_DecisionModel):
    """Apply one of a finding's repair options. Row-local repairs rewrite the working table now;
    statistical ones are recorded and executed in-fold (ROADMAP lockbox constitution §06)."""

    kind: Literal["apply_repair"] = "apply_repair"
    finding_id: str = Field(min_length=1)
    option: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)


class DeferFinding(_DecisionModel):
    """Hold a finding until the question it belongs to; it resurfaces there, attributed."""

    kind: Literal["defer_finding"] = "defer_finding"
    finding_id: str = Field(min_length=1)
    to: str = Field(min_length=1)


class DismissFinding(_DecisionModel):
    kind: Literal["dismiss_finding"] = "dismiss_finding"
    finding_id: str = Field(min_length=1)
    reason: str | None = None


Decision = Annotated[
    Union[
        SetLens, SetTarget, SetTask, SetPurpose, Revert,
        SetRoles, SetEnergyAdjustment, SetExclusions, SetMissing, SetSplit,
        SelectModels, SetSubstitution,
        SetOrientation, SetEvent, SetGrain, SetRepeatKind, SetUnit, SetAggregation, SetTemporal,
        OpenSeal, ApplyRepair, DeferFinding, DismissFinding,
    ],
    Field(discriminator="kind"),
]
DECISION_ADAPTER: TypeAdapter[Any] = TypeAdapter(Decision)


def parse_decision(obj: Any) -> Any:
    """Validate a dict (or a decision model) into a :data:`Decision`."""
    return DECISION_ADAPTER.validate_python(obj)


class DecisionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_serialization_defaults_required=True)

    id: str
    seq: int = Field(ge=1)
    at: AwareDatetime
    note: str | None = None
    # The sentence the Record shows, authored by the server when the decision is
    # recorded (DESIGN_LANGUAGE §05.1: the receipt is a quotation, never a
    # composition). Backticks mark data values. None only for M0-era records.
    sentence: str | None = None
    decision: Decision

    @field_validator("at")
    @classmethod
    def _utc(cls, value: datetime) -> datetime:
        return value.astimezone(timezone.utc)


class ProjectState(BaseModel):
    """The fold of the decision log. Later milestones add slots as fields."""

    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    lens: list[Lens] | None = None
    target: str | None = None
    task: Task | None = None
    purpose: Purpose | None = None
    roles: dict[str, Role] | None = None
    energy_adjustment: EnergyAdjustment | None = None
    exclusions: list[ExclusionRule] | None = None
    missing: MissingSpec | None = None
    split: SplitSpec | None = None
    models: list[str] | None = None
    substitution: SubstitutionSpec | None = None
    # M2
    orientation: Orientation | None = None
    event: str | None = None  # the event level of a binary target (holds while its column is the target)
    grain: GrainSpec | None = None
    repeat_kind: RepeatSpec | None = None
    unit: Literal["unit", "row"] | None = None
    aggregation: AggregationSpec | None = None
    temporal: TemporalSpec | None = None
    seal_opened: bool | None = None
    findings: dict[str, FindingDisposition] | None = None  # finding id -> its disposition

    @field_validator("missing", mode="before")
    @classmethod
    def _strategy_alone(cls, value: Any) -> Any:
        """A bare strategy (``"impute"``) is the answer with no column left out."""
        return {"strategy": value} if isinstance(value, str) else value


def missing_strategy(state: Any) -> MissingStrategy | None:
    """The missing-values strategy the state holds, or None while unanswered."""
    spec = getattr(state, "missing", None)
    return None if spec is None else spec.strategy


def left_out(state: Any) -> list[str]:
    """Columns the missing-values answer left out of the predictors."""
    spec = getattr(state, "missing", None)
    return list(spec.drop_columns) if spec is not None else []


class Refusal(Exception):
    """The "refuse" rung: HTTP 409 ``{error: {code, message, exits}}``, never recorded.

    ``exits`` are ways forward, each ``{label, decision|None}``; a decision may
    be given as a model or a dict and is stored as its JSON form.
    """

    def __init__(self, code: str, message: str, exits: Iterable[Mapping[str, Any]] = ()):
        super().__init__(message)
        self.code = code
        self.message = message
        self.exits: list[dict[str, Any]] = [_normalize_exit(e) for e in exits]

    def to_dict(self) -> dict[str, Any]:
        return {"error": {"code": self.code, "message": self.message, "exits": self.exits}}

    def __repr__(self) -> str:
        return f"Refusal({self.code!r}, {self.message!r}, exits={self.exits!r})"


def _normalize_exit(item: Mapping[str, Any]) -> dict[str, Any]:
    decision = item.get("decision")
    if decision is not None:
        decision = parse_decision(decision).model_dump(mode="json")
    return {"label": str(item["label"]), "decision": decision}


# ── the registry: one handler per kind ──────────────────────────────────────

SLOTS: dict[str, str] = {}
"""kind -> the ProjectState slot it writes. ``revert`` writes no slot."""

_SLOT_VALUE: dict[str, Callable[[Any], Any]] = {}
_HOLDS: dict[str, Callable[[Any, Mapping[str, Any]], bool]] = {}
_KEYS: dict[str, Callable[[Any], str]] = {}
_VALIDATORS: dict[str, list[Callable[[Any, Any], None]]] = {}


def kind_of(model_cls: type[BaseModel]) -> str:
    field = model_cls.model_fields.get("kind")
    if field is None or not isinstance(field.default, str):
        raise TypeError(f"{model_cls.__name__} needs a `kind: Literal[...] = ...` field")
    return field.default


def register_kind(
    model_cls: type[BaseModel],
    slot: str,
    *,
    value: Callable[[Any], Any] | None = None,
    holds: Callable[[Any, Mapping[str, Any]], bool] | None = None,
    key: Callable[[Any], str] | None = None,
) -> type[BaseModel]:
    """Declare that decisions of ``model_cls`` write ``slot``.

    ``value(decision)`` gives the slot's new value; by default it is the model's
    single payload field. ``holds(decision, slots)``, when given, makes the write
    conditional: it stands only while it is true of the slots that
    unconditional kinds wrote, and the slot holds its latest write that stands.
    ``key(decision)``, when given, makes the slot a mapping: each decision writes
    one entry (the latest write per key wins, and a revert restores that key's
    earlier entry). Several kinds may write one keyed slot — apply, defer and
    dismiss all write a finding's disposition under its finding id.
    Returns the class, so it also works as a decorator via ``functools.partial``.
    """
    kind = kind_of(model_cls)
    if kind == "revert":
        raise ValueError("revert is built in and writes no slot")
    if slot not in ProjectState.model_fields:
        raise ValueError(f"ProjectState has no slot {slot!r}; add the field first")
    if value is None:
        payload = [name for name in model_cls.model_fields if name != "kind"]
        if len(payload) != 1:
            raise ValueError(f"{model_cls.__name__} has fields {payload}; pass value=")
        name = payload[0]

        def value(decision: Any, _name: str = name) -> Any:
            return getattr(decision, _name)

    SLOTS[kind] = slot
    _SLOT_VALUE[kind] = value
    if key is not None:
        _KEYS[kind] = key
    else:
        _KEYS.pop(kind, None)
    if holds is not None:
        _HOLDS[kind] = holds
    else:
        _HOLDS.pop(kind, None)
    return model_cls


def register_validator(kind: str, fn: Callable[[Any, Any], None]) -> None:
    """Add a check for ``kind``. ``fn(decision, ctx)`` raises :class:`Refusal`."""
    _VALIDATORS.setdefault(kind, []).append(fn)


def validate(decision: Any, ctx: Any = None) -> Any:
    """Parse ``decision`` and run its kind's validators; returns the parsed decision.

    ``ctx`` is whatever the caller knows about the project (for M0: an object or
    mapping with ``columns`` and ``target``); what it does not name is not checked. Revert targets are checked by
    :meth:`DecisionLog.append`, which holds the log.
    """
    decision = parse_decision(decision)
    for fn in _VALIDATORS.get(decision.kind, ()):
        fn(decision, ctx)
    return decision


def _columns_of(ctx: Any) -> set[str] | None:
    if ctx is None:
        return None
    columns = ctx.get("columns") if isinstance(ctx, Mapping) else getattr(ctx, "columns", None)
    if columns is None:
        return None
    names: set[str] = set()
    for column in columns:
        if isinstance(column, str):
            names.add(column)
        elif isinstance(column, Mapping):
            names.add(str(column["name"]))
        else:
            names.add(str(getattr(column, "name")))
    return names


def _target_is_a_column(decision: SetTarget, ctx: Any) -> None:
    if decision.column == ROW_ID:
        raise Refusal(
            "row_identity",
            "The row identity column numbers the rows; it cannot be the target.",
            exits=[{"label": "Choose one of the dataset's own columns", "decision": None}],
        )
    columns = _columns_of(ctx)
    if columns is not None and decision.column not in columns:
        raise Refusal(
            "unknown_column",
            f"There is no column named {decision.column!r} in this dataset.",
            exits=[{"label": "Choose one of the dataset's columns", "decision": None}],
        )


_UNKNOWN = object()


def _target_of(ctx: Any) -> Any:
    """The current target as ``ctx`` knows it (None: unset), or ``_UNKNOWN``."""
    if ctx is None:
        return _UNKNOWN
    if isinstance(ctx, Mapping):
        return ctx.get("target", _UNKNOWN)
    return getattr(ctx, "target", _UNKNOWN)


def _task_is_for_the_target(decision: SetTask, ctx: Any) -> None:
    target = _target_of(ctx)
    if target is _UNKNOWN:
        return
    if target is None:
        raise Refusal(
            "no_target",
            "Choose the outcome first; the task describes it.",
            exits=[{"label": "Choose the outcome", "decision": None}],
        )
    if decision.column != target:
        raise Refusal(
            "not_the_target",
            f"The outcome is {target!r}, not {decision.column!r}; a task answers for the outcome.",
            exits=[{"label": f"Answer the task question for {target}", "decision": None}],
        )


register_kind(SetLens, "lens")
register_kind(SetTarget, "target")
register_kind(
    SetTask,
    "task",
    value=lambda decision: decision.task,
    holds=lambda decision, slots: slots.get("target") == decision.column,
)
register_kind(SetPurpose, "purpose")
register_kind(SetRoles, "roles")
register_kind(SetEnergyAdjustment, "energy_adjustment",
              value=lambda d: EnergyAdjustment(**d.model_dump(exclude={"kind"})))
register_kind(SetExclusions, "exclusions")
register_kind(SetMissing, "missing", value=lambda d: MissingSpec(**d.model_dump(exclude={"kind"})))
register_kind(SetSplit, "split", value=lambda d: SplitSpec(**d.model_dump(exclude={"kind"})))
register_kind(SelectModels, "models")
register_kind(SetSubstitution, "substitution",
              value=lambda d: SubstitutionSpec(**d.model_dump(exclude={"kind"})))
register_kind(SetOrientation, "orientation")
register_kind(SetEvent, "event", value=lambda d: d.level,
              holds=lambda d, slots: slots.get("target") == d.column)
register_kind(SetGrain, "grain", value=lambda d: GrainSpec(**d.model_dump(exclude={"kind"})))
register_kind(SetRepeatKind, "repeat_kind",
              value=lambda d: RepeatSpec(**d.model_dump(exclude={"kind"})))
register_kind(SetUnit, "unit")
register_kind(SetAggregation, "aggregation",
              value=lambda d: AggregationSpec(**d.model_dump(exclude={"kind"})))
register_kind(SetTemporal, "temporal",
              value=lambda d: TemporalSpec(**d.model_dump(exclude={"kind"})))
register_kind(OpenSeal, "seal_opened", value=lambda d: True)
register_kind(ApplyRepair, "findings", key=lambda d: d.finding_id,
              value=lambda d: FindingDisposition(action="applied", option=d.option, params=d.params))
register_kind(DeferFinding, "findings", key=lambda d: d.finding_id,
              value=lambda d: FindingDisposition(action="deferred", to=d.to))
register_kind(DismissFinding, "findings", key=lambda d: d.finding_id,
              value=lambda d: FindingDisposition(action="dismissed", reason=d.reason))
register_validator("set_target", _target_is_a_column)
register_validator("set_task", _task_is_for_the_target)


# ── M1 validators (M1_CONTRACT.md §2) ────────────────────────────────────────
# Each reads only what ``ctx`` names; what it does not name is not checked. The
# server's ctx (turbotab/server/service.py DecisionContext) carries ``columns``,
# ``column_info`` ({name: {dtype, n_unique, n_missing}}), ``state`` (the state
# before this decision), ``task`` (the answered or detected task) and ``target``.

NUMERIC_DTYPES = ("numeric", "integer")
MAX_CLASSES_MULTICLASS = 20
MAX_STRATA_LEVELS = 10
PREDICTOR_ROLES = ("exposure", "covariate", "energy")


def _ctx(ctx: Any, name: str, default: Any = None) -> Any:
    if ctx is None:
        return default
    if isinstance(ctx, Mapping):
        return ctx.get(name, default)
    return getattr(ctx, name, default)


def _info(ctx: Any, column: str) -> Mapping[str, Any] | None:
    info = _ctx(ctx, "column_info")
    if not info:
        return None
    return info.get(column)


def _state(ctx: Any) -> "ProjectState | None":
    return _ctx(ctx, "state")


def _and(items: Sequence[str]) -> str:
    items = [f"`{i}`" for i in items]
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


def _task_fits_the_outcome(decision: SetTask, ctx: Any) -> None:
    info = _info(ctx, decision.column)
    if info is None:
        return
    n_unique = int(info.get("n_unique") or 0)
    numeric = info.get("dtype") in NUMERIC_DTYPES
    fits: list[str] = []
    if n_unique == 2:
        fits.append("binary")
    if 2 < n_unique <= MAX_CLASSES_MULTICLASS:
        fits.append("multiclass")
    if numeric and n_unique > 2:
        fits.append("regression")
    exits = [{"label": f"Treat it as {task}", "decision": SetTask(column=decision.column, task=task)}
             for task in fits]
    if decision.task == "binary" and n_unique != 2:
        raise Refusal(
            "task_mismatch",
            f"`{decision.column}` has {n_unique:,} distinct values; a binary outcome has exactly 2.",
            exits=exits or [{"label": "Choose another outcome", "decision": None}],
        )
    if decision.task == "multiclass" and n_unique > MAX_CLASSES_MULTICLASS:
        raise Refusal(
            "task_mismatch",
            f"`{decision.column}` has {n_unique:,} distinct values; multiclass takes at most "
            f"{MAX_CLASSES_MULTICLASS}.",
            exits=exits or [{"label": "Choose another outcome", "decision": None}],
        )


def _roles_name_real_columns(decision: SetRoles, ctx: Any) -> None:
    columns = _columns_of(ctx)
    if columns is not None:
        unknown = [c for c in decision.roles if c not in columns or c == ROW_ID]
        if unknown:
            rest = {c: r for c, r in decision.roles.items() if c not in unknown}
            exits = [{"label": "Keep the roles of the real columns", "decision": SetRoles(roles=rest)}] \
                if rest else []
            raise Refusal(
                "unknown_column",
                f"This dataset has no column named {_and(unknown)}.",
                exits=exits + [{"label": "Choose roles for the dataset's columns", "decision": None}],
            )
    target = _target_of(ctx)
    if target is not _UNKNOWN and target is not None and target in decision.roles:
        rest = {c: r for c, r in decision.roles.items() if c != target}
        raise Refusal(
            "target_has_role",
            f"`{target}` is the outcome, so it cannot also be {decision.roles[target]!s}.",
            exits=[{"label": "Leave the outcome out of the roles", "decision": SetRoles(roles=rest)}]
            if rest else [],
        )
    if not any(role in PREDICTOR_ROLES for role in decision.roles.values()):
        raise Refusal(
            "no_predictors",
            "No column is an exposure, a covariate or energy, so the models would have nothing to use.",
            exits=[{"label": "Mark at least one column as an exposure or a covariate", "decision": None}],
        )


def _rule_without(decision: SetExclusions, index: int) -> SetExclusions:
    return SetExclusions(rules=[r for i, r in enumerate(decision.rules) if i != index])


def _exclusions_are_ranges_on_numbers(decision: SetExclusions, ctx: Any) -> None:
    columns = _columns_of(ctx)
    for i, rule in enumerate(decision.rules):
        drop = {"label": f"Drop the rule on `{rule.column}`", "decision": _rule_without(decision, i)}
        for name in [rule.column] + ([rule.by.column] if rule.by is not None else []):
            if columns is not None and (name not in columns or name == ROW_ID):
                raise Refusal("unknown_column", f"This dataset has no column named `{name}`.", exits=[drop])
        info = _info(ctx, rule.column)
        if info is not None and info.get("dtype") not in NUMERIC_DTYPES:
            raise Refusal(
                "not_numeric",
                f"`{rule.column}` is {info.get('dtype')}, not numbers, so it has no range to keep.",
                exits=[drop],
            )
        if rule.low is None and rule.high is None and not (rule.by and rule.by.ranges):
            raise Refusal("no_bounds", f"The rule on `{rule.column}` sets no lower or upper bound.",
                          exits=[drop])
        bounds = [(None, rule.low, rule.high)]
        if rule.by is not None:
            bounds += [(level, lo, hi) for level, (lo, hi) in rule.by.ranges.items()]
        for level, low, high in bounds:
            if low is not None and high is not None and low >= high:
                where = f" for `{level}`" if level is not None else ""
                exits = [drop]
                if level is None and low > high:
                    swapped = list(decision.rules)
                    swapped[i] = rule.model_copy(update={"low": high, "high": low})
                    exits.insert(0, {"label": "Swap the bounds", "decision": SetExclusions(rules=swapped)})
                raise Refusal(
                    "empty_range",
                    f"The range on `{rule.column}`{where} runs from {low:g} to {high:g}; "
                    f"the lower bound must be below the upper.",
                    exits=exits,
                )


def _energy_adjustment_fits_the_roles(decision: SetEnergyAdjustment, ctx: Any) -> None:
    if decision.method == "none":
        return
    state = _state(ctx)
    columns = _columns_of(ctx)
    if state is None:
        return
    roles = state.roles or {}
    base = decision.model_dump(exclude={"kind"})

    def with_(**changes: Any) -> SetEnergyAdjustment:
        return SetEnergyAdjustment(**{**base, **changes})

    if not roles:
        raise Refusal(
            "roles_first",
            "Energy adjustment works on the column roles; confirm them first.",
            exits=[{"label": "Confirm the column roles", "decision": None},
                   {"label": "Do not adjust for energy", "decision": with_(method="none")}],
        )
    energy_columns = [c for c, r in roles.items() if r == "energy"]
    if decision.energy_column is None or roles.get(decision.energy_column) != "energy":
        exits = [{"label": f"Adjust against `{c}`", "decision": with_(energy_column=c)} for c in energy_columns]
        named = f"`{decision.energy_column}` does not" if decision.energy_column else "No column was named that"
        raise Refusal(
            "energy_role",
            f"{named} has the energy role, and adjustment is computed against total energy.",
            exits=exits + [{"label": "Give a column the energy role", "decision": None}],
        )
    exposures = [n for n in decision.nutrients if roles.get(n) == "exposure"]
    if not decision.nutrients or len(exposures) != len(decision.nutrients):
        others = [n for n in decision.nutrients if n not in exposures]
        message = ("Name the nutrients to adjust." if not decision.nutrients else
                   f"{_and(others)} {'is' if len(others) == 1 else 'are'} not an exposure; only exposures are adjusted.")
        exits = [{"label": "Adjust only the exposures", "decision": with_(nutrients=exposures)}] if exposures else []
        raise Refusal("nutrients_not_exposures", message,
                      exits=exits + [{"label": "Choose the nutrients to adjust", "decision": None}])
    gone = [c for c in (decision.energy_column, *decision.nutrients) if c in left_out(state)]
    if gone:
        kept = [n for n in decision.nutrients if n not in gone]
        exits = ([{"label": "Adjust only the nutrients still in the model", "decision": with_(nutrients=kept)}]
                 if kept and decision.energy_column not in gone else [])
        raise Refusal(
            "left_out",
            f"{_and(gone)} {'was' if len(gone) == 1 else 'were'} left out of the predictors with the "
            f"missing values, so energy adjustment cannot use {'it' if len(gone) == 1 else 'them'}.",
            exits=exits + [{"label": "Do not adjust for energy", "decision": with_(method="none")}],
        )
    from turbotab.core.methods.energy import METHOD_TABLE, applicable_methods

    table = applicable_methods(sorted(columns or roles), decision.energy_column, decision.nutrients)
    verdict = table.get(decision.method, {"ok": True})
    if not verdict["ok"]:
        exits = [{"label": METHOD_TABLE[m]["label"], "decision": with_(method=m)}
                 for m, v in table.items() if v["ok"] and m != decision.method]
        raise Refusal("method_not_applicable", str(verdict["reason"]), exits=exits)
    if decision.method == "partition":
        _partition_runs_on_the_data(decision, ctx, with_, table)
    if decision.strata is not None:
        info = _info(ctx, decision.strata)
        known = columns is None or decision.strata in columns
        levels = int(info.get("n_unique") or 0) if info else None
        categorical = info is None or info.get("dtype") in ("categorical", "boolean", "integer")
        if not known or not categorical or (levels is not None and levels > MAX_STRATA_LEVELS):
            raise Refusal(
                "strata_not_categorical",
                f"Strata need a category with at most {MAX_STRATA_LEVELS} levels; "
                f"`{decision.strata}` is not one.",
                exits=[{"label": "Adjust without strata", "decision": with_(strata=None)}],
            )


PARTITION_CHECK_ROWS = 5_000


def _partition_runs_on_the_data(decision: SetEnergyAdjustment, ctx: Any,
                                with_: Callable[..., SetEnergyAdjustment],
                                table: Mapping[str, Mapping[str, Any]]) -> None:
    """Refuse a partition the fit would refuse, or one that counts a total and its parts twice.

    The same checks the fit makes (energy.partition_refusal), on a sample of the rows outside the
    held-out set: a unit question, asked of the data before the answer is recorded, so the design
    never fails on it afterwards.
    """
    import numpy as np

    from turbotab.core.methods.energy import METHOD_TABLE, partition_refusal
    from turbotab.core.methods.nesting import nested_components

    opener = _ctx(ctx, "store")
    try:
        store = opener() if callable(opener) else None
    except Exception:  # noqa: BLE001 - no data to check: the fit decides
        store = None
    E, nutrients = decision.energy_column, list(decision.nutrients)
    if store is None or E is None or not {E, *nutrients} <= set(store.columns):
        return
    pool = np.arange(int(store.n_rows), dtype=np.int64)
    sealed_of = _ctx(ctx, "sealed")
    sealed = sealed_of() if callable(sealed_of) else None
    if sealed is not None and len(sealed):
        pool = np.setdiff1d(pool, np.asarray(sealed, dtype=np.int64), assume_unique=True)
    if len(pool) > PARTITION_CHECK_ROWS:
        pool = np.sort(np.random.default_rng(0).choice(pool, size=PARTITION_CHECK_ROWS, replace=False))
    frame = store.materialize([E, *nutrients], pool)
    nested = nesting_of(ctx) or nested_components(frame, nutrients)
    refused = partition_refusal(frame, E, nutrients, nested=nested)
    if refused is None:
        return
    exits: list[dict[str, Any]] = []
    if refused["nutrients"]:
        exits.append({"label": f"Partition {_and(refused['nutrients'])}",
                      "decision": with_(nutrients=refused["nutrients"])})
    exits += [{"label": METHOD_TABLE[m]["label"], "decision": with_(method=m)}
              for m in ("residual", "standard") if table.get(m, {}).get("ok")]
    raise Refusal("method_not_applicable", _ticked(str(refused["reason"]), [E, *nutrients]),
                  exits=exits)


def _ticked(text: str, columns: Sequence[str]) -> str:
    """Wrap these column names in backticks (data chips), longest first, whole words only."""
    import re

    for c in sorted(set(columns), key=len, reverse=True):
        text = re.sub(rf"(?<![`\w]){re.escape(c)}(?![`\w])", f"`{c}`", text)
    return text


def model_families() -> dict[str, set[str]]:
    """Model family key -> the tasks it can model, from ``turbotab.core.models`` when present.

    The registry is the modeling agent's (M1_CONTRACT.md §7); until it is importable the M1
    families stand in, so a validator never refuses a family that exists.
    """
    fallback = {key: {"regression", "binary", "multiclass"}
                for key in ("linear", "elastic_net", "boosted_trees")}
    try:
        import importlib

        module = importlib.import_module("turbotab.core.models")
    except Exception:
        return fallback
    found: Any = None
    for name in ("FAMILIES", "REGISTRY", "families", "all_families", "registry"):
        value = getattr(module, name, None)
        if value is None:
            continue
        found = value() if callable(value) and not isinstance(value, Mapping) else value
        break
    if found is None:
        return fallback
    items = found.values() if isinstance(found, Mapping) else found
    out: dict[str, set[str]] = {}
    for family in items:
        key = _ctx(family, "key")
        if key:
            out[str(key)] = {str(t) for t in (_ctx(family, "tasks") or ())}
    return out or fallback


def _models_can_fit_the_task(decision: SelectModels, ctx: Any) -> None:
    families = model_families()
    unknown = [m for m in decision.models if m not in families]
    known = [m for m in decision.models if m in families]
    if unknown:
        raise Refusal(
            "unknown_model",
            f"There is no model family called {_and(unknown)}.",
            exits=([{"label": "Keep the families that exist", "decision": SelectModels(models=known)}]
                   if known else []) + [{"label": "Choose from the shelf", "decision": None}],
        )
    task = _ctx(ctx, "task")
    if task is None:
        return
    unable = [m for m in decision.models if families[m] and task not in families[m]]
    if unable:
        able = [m for m in decision.models if m not in unable]
        raise Refusal(
            "model_cannot_fit_task",
            f"{_and(unable)} cannot model a {task} outcome.",
            exits=([{"label": "Keep the families that can", "decision": SelectModels(models=able)}]
                   if able else []) + [{"label": "Choose from the shelf", "decision": None}],
        )


def _substitution_swaps_energy(decision: SetSubstitution, ctx: Any) -> None:
    if decision.donor == decision.recipient:
        raise Refusal(
            "same_nutrient",
            f"A substitution swaps one nutrient for another; `{decision.donor}` cannot replace itself.",
            exits=[{"label": "Choose a different recipient", "decision": None}],
        )
    state = _state(ctx)
    if state is None:
        return
    from turbotab.core.stages.rows import energy_bearing

    roles = state.roles or {}
    candidates = [c for c, r in roles.items() if r == "exposure" and energy_bearing(c)]
    bad = [c for c in (decision.donor, decision.recipient) if c not in candidates]
    if bad:
        base = decision.model_dump(exclude={"kind"})
        exits = []
        for other in candidates:
            if other not in (decision.donor, decision.recipient) and len(exits) < 3:
                fixed = {**base, **({"donor": other} if decision.donor in bad else {"recipient": other})}
                if fixed["donor"] != fixed["recipient"] and fixed["donor"] in candidates \
                        and fixed["recipient"] in candidates:
                    exits.append({"label": f"Swap `{fixed['donor']}` for `{fixed['recipient']}`",
                                  "decision": SetSubstitution(**fixed)})
        raise Refusal(
            "not_energy_bearing",
            f"{_and(bad)} {'is' if len(bad) == 1 else 'are'} not an energy-bearing nutrient in the "
            f"model; a substitution moves kcal between two of them.",
            exits=exits + [{"label": "Choose two energy-bearing nutrients", "decision": None}],
        )


def nesting_of(ctx: Any) -> dict[str, str]:
    """Child column -> the column it is part of, as the design (else the roles stage) found it."""
    artifact = _ctx(ctx, "artifact")
    if not callable(artifact):
        return {}
    for stage in ("design", "roles"):
        try:
            data = artifact(stage)
        except Exception:  # noqa: BLE001 - a missing artifact checks nothing
            data = None
        if not isinstance(data, Mapping):
            continue
        if stage == "design" and data.get("nested") is not None:
            return {str(e["column"]): str(e["parent"]) for e in data["nested"]}
        if stage == "roles":
            return {str(e["column"]): str(e["nested_in"]) for e in data.get("columns") or []
                    if e.get("nested_in")}
    return {}


def _substitution_moves_between_separate_nutrients(decision: SetSubstitution, ctx: Any) -> None:
    nested = nesting_of(ctx)
    d, r = decision.donor, decision.recipient
    if nested.get(d) == r or nested.get(r) == d:
        child, parent = (d, r) if nested.get(d) == r else (r, d)
        raise Refusal(
            "part_of_the_other",
            f"`{child}` is part of `{parent}`, so moving kcal between them moves nothing; a "
            f"substitution swaps two separate nutrients.",
            exits=[{"label": "Choose two separate nutrients", "decision": None}],
        )


def _left_out_columns_are_predictors(decision: SetMissing, ctx: Any) -> None:
    if not decision.drop_columns:
        return
    columns = _columns_of(ctx)
    keep = {"label": "Keep every predictor", "decision": SetMissing(strategy=decision.strategy)}
    if columns is not None:
        unknown = [c for c in decision.drop_columns if c not in columns or c == ROW_ID]
        if unknown:
            raise Refusal("unknown_column", f"This dataset has no column named {_and(unknown)}.",
                          exits=[keep])
    state = _state(ctx)
    roles = (state.roles if state is not None else None) or {}
    if not roles:
        return
    preds = [c for c, r in roles.items() if r in PREDICTOR_ROLES]
    outside = [c for c in decision.drop_columns if c not in preds]
    if outside:
        rest = [c for c in decision.drop_columns if c not in outside]
        exits = ([{"label": f"Leave out only {_and(rest)}",
                   "decision": SetMissing(strategy=decision.strategy, drop_columns=rest)}]
                 if rest else []) + [keep]
        raise Refusal(
            "not_a_predictor",
            f"{_and(outside)} {'is' if len(outside) == 1 else 'are'} not a predictor, so there is "
            f"nothing to leave out.",
            exits=exits,
        )
    if not [c for c in preds if c not in decision.drop_columns]:
        raise Refusal(
            "no_predictors",
            "Leaving out every predictor would leave the models nothing to use.",
            exits=[keep],
        )


register_validator("set_task", _task_fits_the_outcome)
register_validator("set_roles", _roles_name_real_columns)
register_validator("set_missing", _left_out_columns_are_predictors)
register_validator("set_substitution", _substitution_moves_between_separate_nutrients)
register_validator("set_exclusions", _exclusions_are_ranges_on_numbers)
register_validator("set_energy_adjustment", _energy_adjustment_fits_the_roles)
register_validator("select_models", _models_can_fit_the_task)
register_validator("set_substitution", _substitution_swaps_energy)


# ── the fold ─────────────────────────────────────────────────────────────────

def _check_revertable(target_id: str, earlier: Mapping[str, DecisionRecord]) -> None:
    target = earlier.get(target_id)
    if target is None or not (
        isinstance(target.decision, Revert) or target.decision.kind in SLOTS
    ):
        raise Refusal(
            "unknown_decision",
            f"There is no earlier decision {target_id!r} that can be reverted.",
        )


def reverted(records: Sequence[DecisionRecord]) -> dict[str, str]:
    """Map each currently reverted record id to the id of the revert cancelling it.

    Raises :class:`Refusal` (``unknown_decision``) if a revert names a record
    that does not precede it or writes no slot.
    """
    ordered = sorted(records, key=lambda r: r.seq)
    earlier: dict[str, DecisionRecord] = {}
    for record in ordered:
        if isinstance(record.decision, Revert):
            _check_revertable(record.decision.decision_id, earlier)
        earlier[record.id] = record
    # Latest first: a revert only counts if nothing later has reverted it.
    cancelled: dict[str, str] = {}
    for record in reversed(ordered):
        if record.id in cancelled:
            continue
        if isinstance(record.decision, Revert):
            cancelled.setdefault(record.decision.decision_id, record.id)
    return cancelled


def fold(records: Sequence[DecisionRecord]) -> ProjectState:
    """The project state: each slot holds its latest write that is not reverted.

    A conditional write (``register_kind(..., holds=)``) counts only while its
    condition holds on the unconditional slots: the ``task`` slot is the latest
    ``set_task`` naming the current target, or unset.
    """
    ordered = sorted(records, key=lambda r: r.seq)
    cancelled = reverted(ordered)
    live = [r.decision for r in ordered if r.id not in cancelled and r.decision.kind in SLOTS]
    slots: dict[str, Any] = {}
    for decision in live:
        if decision.kind in _HOLDS:
            continue
        keyed = _KEYS.get(decision.kind)
        if keyed is not None:
            entries = dict(slots.get(SLOTS[decision.kind]) or {})
            entries[keyed(decision)] = _SLOT_VALUE[decision.kind](decision)
            slots[SLOTS[decision.kind]] = entries
        else:
            slots[SLOTS[decision.kind]] = _SLOT_VALUE[decision.kind](decision)
    base = dict(slots)
    for decision in live:
        holds = _HOLDS.get(decision.kind)
        if holds is not None and holds(decision, base):
            slots[SLOTS[decision.kind]] = _SLOT_VALUE[decision.kind](decision)
    return ProjectState(**slots)


def _check_revert(existing: Sequence[DecisionRecord], decision: Revert) -> None:
    _check_revertable(decision.decision_id, {r.id: r for r in existing})
    cancelled = reverted(existing)
    by = cancelled.get(decision.decision_id)
    if by is not None:
        raise Refusal(
            "already_reverted",
            "That decision has already been reverted.",
            exits=[{"label": "Undo the earlier revert instead", "decision": Revert(decision_id=by)}],
        )


# ── the log ──────────────────────────────────────────────────────────────────

class DecisionLog:
    """Append-only ``decisions.jsonl``. Safe across threads and processes.

    Each append is one ``write`` of one line under an exclusive ``flock``,
    followed by ``fsync``. A torn line left by a crash is skipped on read (with
    a warning) rather than making the project unreadable.
    """

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._cache: tuple[tuple[int, int], list[DecisionRecord]] | None = None

    def records(self) -> list[DecisionRecord]:
        with self._lock:
            return list(self._load())

    def state(self) -> ProjectState:
        return fold(self.records())

    def append(
        self,
        decision: Any,
        note: str | None = None,
        sentence: Callable[[Any, ProjectState], str | None] | None = None,
    ) -> DecisionRecord:
        """Append ``decision``; ``sentence(decision, state_before)`` authors the record's sentence.

        ``state_before`` is the fold of the log as it stands under the lock, just before this
        record. A sentence that fails is logged and left unset: the answer is still recorded.
        """
        decision = parse_decision(decision)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o644)
            try:
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_EX)
                existing = self._load()
                if isinstance(decision, Revert):
                    _check_revert(existing, decision)
                text: str | None = None
                if sentence is not None:
                    try:
                        text = sentence(decision, fold(existing))
                    except Exception:  # noqa: BLE001 - a missing sentence never loses an answer
                        log.exception("no sentence for a %s decision", decision.kind)
                        text = None
                record = DecisionRecord(
                    id=uuid.uuid4().hex,
                    seq=max((r.seq for r in existing), default=0) + 1,
                    at=datetime.now(timezone.utc),
                    note=note,
                    sentence=text if isinstance(text, str) and text.strip() else None,
                    decision=decision,
                )
                data = (record.model_dump_json() + "\n").encode("utf-8")
                size = os.fstat(fd).st_size
                if size and os.pread(fd, 1, size - 1) != b"\n":
                    data = b"\n" + data  # never glue onto a torn line
                view = memoryview(data)
                while view:
                    view = view[os.write(fd, view):]
                os.fsync(fd)
            finally:
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
            self._cache = None
            return record

    def _load(self) -> list[DecisionRecord]:
        try:
            st = self.path.stat()
        except FileNotFoundError:
            return []
        stamp = (st.st_size, st.st_mtime_ns)
        if self._cache is not None and self._cache[0] == stamp:
            return self._cache[1]
        records: list[DecisionRecord] = []
        text = self.path.read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                log.warning("%s:%d is a torn line; skipped", self.path, number)
                continue
            records.append(DecisionRecord.model_validate(raw))
        records.sort(key=lambda r: r.seq)
        self._cache = (stamp, records)
        return records

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


Decision = Annotated[
    Union[SetLens, SetTarget, SetTask, SetPurpose, Revert],
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
) -> type[BaseModel]:
    """Declare that decisions of ``model_cls`` write ``slot``.

    ``value(decision)`` gives the slot's new value; by default it is the model's
    single payload field. ``holds(decision, slots)``, when given, makes the write
    conditional: it stands only while it is true of the slots that
    unconditional kinds wrote, and the slot holds its latest write that stands.
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
register_validator("set_target", _target_is_a_column)
register_validator("set_task", _task_is_for_the_target)


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
        if decision.kind not in _HOLDS:
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

    def append(self, decision: Any, note: str | None = None) -> DecisionRecord:
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
                record = DecisionRecord(
                    id=uuid.uuid4().hex,
                    seq=max((r.seq for r in existing), default=0) + 1,
                    at=datetime.now(timezone.utc),
                    note=note,
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

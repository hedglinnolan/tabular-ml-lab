"""The surfacing registry (SURFACING_POLICY §1.6, recommendation 2): one executable protocol for
everything the quest log can surface, so a fuzzer has code to fuzz rather than prose.

The policy's candidate set is the union of five shapes the engine already has, and each now
answers the same questions in code (:class:`Surfaceable`):

* **questions**: the Router's keys (``interview.QUESTION_KEYS``). ``fires`` is the Router's own
  gate (``interview.applicability``: asked, stated or answered, never "does not apply"); ``reads``
  every earlier question (the Router asks in its dependency order, CROSSWALK disagreement 18);
  ``stands`` is ``quest.answer_holds``.
* **decisions**: every decision kind the log accepts (``decisions.SLOTS``). A kind that answers a
  Router question is that question's; a declaration (``quest.DECLARATIONS``) is its own, its
  ``applies`` and ``reads``; every other kind has a predicate here (:data:`KIND_FIRES`), read from
  its crosswalk item's ``fires_when``. "Confirm all" is one per stage sweep. A ``revert`` is the
  record it undoes (``quest.FOLLOWS_WHAT_IT_UNDOES``), so it has none of its own.
* **defaults**: the defaults the engine states today (the Router's stated gates, Estimate's split,
  what the values settled) and those whose condition is one slot.
* **noticings**: the engine's current findings, by family (``stages/finding_words.FAMILIES`` and
  the rest the engine names), and Explore's (``quest.EXPLORE_FINDINGS``). Each fires when the
  findings artifact (or Explore's) holds one of its family.
* **results**: each compute stage the quest log shows (``quest.COMPUTE``). It fires when the graph
  can compute it (every slot it and its upstream require is set, as ``graph.Engine`` reads
  ``requires``).

Each item declares ``fires(state, facts)``, ``reads`` (the questions or declarations whose answers
it needs), ``holds`` (the slots its answer writes: what it changes; a result writes none),
``consumer`` (the first compute stage that reads what it holds, in the graph's order; None when
only a reader reads it) and ``stage`` (its quest stage) with ``item`` (its crosswalk card).

**The crosswalk's rules.** ``crosswalk.json`` items gain a ``rule: "module:function"`` field naming
their predicate (:func:`crosswalk_rules`); the function is resolved here by name
(``fires_<item id>``), true when any engine object that item carries fires. An item whose
``fires_when`` is still prose has no rule and is counted UNWIRED by the registry test.

**Modes** (§4.4): an expertise mode changes disclosure (the level a line is drawn at), never tier;
:func:`disclosure` is the one place a mode is read, so the fuzzer can hold I10 over it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Iterable, Literal, Mapping, Protocol, Sequence, runtime_checkable

from turbotab.core.quest import Facts

Shape = Literal["question", "decision", "default", "noticing", "result"]
SHAPES: tuple[Shape, ...] = ("question", "decision", "default", "noticing", "result")
RULE_MODULE = "turbotab.core.surfacing"
RULE_PREFIX = "fires_"


@runtime_checkable
class Surfaceable(Protocol):
    """One thing the quest log can surface, executable (SURFACING_POLICY §1.6)."""

    key: str
    shape: str
    stage: str  # its quest stage
    item: str  # its crosswalk card ("" when the crosswalk has none yet)
    reads: tuple[str, ...]  # the questions (or declarations) whose answers it needs
    holds: frozenset[str]  # the slots its answer writes: what it changes
    consumer: str | None  # the first compute stage that reads what it holds

    def fires(self, state: Any, facts: Facts) -> bool: ...

    def stands(self, state: Any) -> bool: ...


@dataclass(frozen=True)
class Item:
    """The one implementation of :class:`Surfaceable`: a predicate and its declared edges."""

    key: str
    shape: Shape
    stage: str
    item: str
    when: Callable[[Any, Facts], bool]
    reads: tuple[str, ...] = ()
    holds: frozenset[str] = frozenset()
    consumer: str | None = None
    standing: Callable[[Any], bool] | None = field(default=None, compare=False)

    def fires(self, state: Any, facts: Facts) -> bool:
        return bool(self.when(state, facts))

    def stands(self, state: Any) -> bool:
        """Whether a recorded answer still stands (``quest.Declaration.holds``,
        ``quest.answer_holds``); True when nothing but its slot says so."""
        return True if self.standing is None else bool(self.standing(state))


# ── small readers ────────────────────────────────────────────────────────────


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _lens(state: Any, *lenses: str) -> bool:
    return any(lens in (getattr(state, "lens", None) or ()) for lens in lenses)


def _listed(artifact: Any, name: str = "findings") -> list[Mapping[str, Any]]:
    data = _get(artifact, "data", artifact)
    found = data if isinstance(data, list) else _get(data, name) or []
    return [f for f in found if isinstance(f, Mapping)]


def _readings(facts: Facts) -> list[Mapping[str, Any]]:
    return [r for r in facts.artifacts.get("readings") or () if isinstance(r, Mapping)]


def _unsettled(facts: Facts, *kinds: str) -> bool:
    return any(not r.get("settled", True) and (not kinds or r.get("kind") in kinds)
               for r in _readings(facts))


def _target_info(state: Any, facts: Facts) -> Mapping[str, Any] | None:
    info = facts.artifacts.get("target_info")
    info = _get(info, "data", info)
    if isinstance(info, Mapping) and info.get("column") == getattr(state, "target", None):
        return info
    return None


def _task_followup(state: Any, facts: Facts) -> str | None:
    from turbotab.core.structural import task_followup

    return task_followup(state, facts.artifacts.get("target_info"))


# ── questions ────────────────────────────────────────────────────────────────


def question_fires(key: str, state: Any, facts: Facts) -> bool:
    """A Router question fires unless its gate finds it not applicable (asked, stated or
    answered), as ``interview.route`` reads the same gates."""
    from turbotab.core.interview import applicability

    return applicability(key, state, facts.artifacts) is None


def _question(key: str) -> Callable[[Any, Facts], bool]:
    return lambda state, facts: question_fires(key, state, facts)


# ── decision kinds with no Router key and no declaration ─────────────────────


def _findings_on_the_table(state: Any, facts: Facts) -> bool:
    """decision:finding-disposition: "Fires on every finding with repairs"; a finding with none is
    still disposed of (dismissed, or held for a question)."""
    return bool(_listed(facts.artifacts.get("findings")))


def _readings_ask(state: Any, facts: Facts) -> bool:
    """decision:readings-ask-card: a reading a consumer reads is unsettled, or a role was proposed
    below high confidence."""
    return _unsettled(facts) or bool(getattr(state, "roles_unconfirmed", None))


def _role_unconfirmed(state: Any, facts: Facts) -> bool:
    """decision:confirm_role: a role proposed below high confidence is waiting."""
    return bool(getattr(state, "roles_unconfirmed", None))


def _code_or_amount(state: Any, facts: Facts) -> bool:
    """reading:code-or-amount: a whole-valued column whose code-or-amount reading is unsettled, or
    one recorded as codes."""
    return _unsettled(facts, "code_or_amount", "categorical") or bool(
        getattr(state, "categorical", None))


def _energy_unit(state: Any, facts: Facts) -> bool:
    """reading:energy-unit-and-days: under the dietary lens, total energy's unit or days is only
    proposed (unsettled), or a unit is recorded."""
    return _lens(state, "dietary") and (_unsettled(facts, "energy_unit", "unit", "days")
                                        or bool(getattr(state, "column_units", None)))


def _feature_table(state: Any, facts: Facts) -> bool:
    """decision:set_feature_table: a table turned around (features in rows)."""
    return getattr(state, "orientation", None) == "feature_major"


def _outcome_scale(state: Any, facts: Facts) -> bool:
    """q:outcome_scale: the task question's follow-up asks the outcome's scale, or one is recorded
    for this outcome."""
    spec = getattr(state, "outcome_scale", None)
    return _task_followup(state, facts) == "scale" or (
        spec is not None and _get(spec, "column") == getattr(state, "target", None))


def _outcome_order(state: Any, facts: Facts) -> bool:
    """q:outcome_order: an ordinal text outcome's levels are unordered, or an order is recorded."""
    return _task_followup(state, facts) == "order" or bool(getattr(state, "outcome_order", None))


def _outcome_unit(state: Any, facts: Facts) -> bool:
    """q:outcome_unit: a regression outcome (its unit is asked when no codebook documents it)."""
    if getattr(state, "target", None) is None:
        return False
    info = _target_info(state, facts) or {}
    return (getattr(state, "task", None) or info.get("task")) == "regression"


def _plan_locks(state: Any, facts: Facts) -> bool:
    """other:plan-lock: under Estimate and Describe the server records the lock when Fit is
    pressed (``fit_press``); never under prediction, never with no goal."""
    return getattr(state, "purpose", None) == "inference"


def _after_the_fit(state: Any, facts: Facts) -> bool:
    """decision:respond_diagnostic: a diagnostic of the fit is there to respond to."""
    return facts.artifacts.get("fit") is not None


def _seal_opened(state: Any, facts: Facts) -> bool:
    """decision:reseal: only as an exit after the held-out rows were opened."""
    return bool(getattr(state, "seal_opened", None))


KIND_FIRES: dict[str, Callable[[Any, Facts], bool]] = {
    "apply_repair": _findings_on_the_table,
    "defer_finding": _findings_on_the_table,
    "dismiss_finding": _findings_on_the_table,
    "confirm_reading": _readings_ask,
    "confirm_readings": _readings_ask,
    "confirm_role": _role_unconfirmed,
    "set_categorical": _code_or_amount,
    "set_column_unit": _energy_unit,
    "set_feature_table": _feature_table,
    "set_outcome_scale": _outcome_scale,
    "set_outcome_order": _outcome_order,
    "set_outcome_unit": _outcome_unit,
    "lock_plan": _plan_locks,
    "respond_diagnostic": _after_the_fit,
    "reseal": _seal_opened,
}


def _sweep_fires(stage: str) -> Callable[[Any, Facts], bool]:
    """"Confirm all" is offered where the stage's sweep holds a stated default
    (``sweep.sweep_of``); from the state alone, whenever a goal or an outcome is set (the design
    is stated observational from the start, the split under Estimate, the task on a settled
    reading). First look sets nothing for the person."""
    return lambda state, facts: (getattr(state, "target", None) is not None
                                 or getattr(state, "purpose", None) is not None)


# ── defaults ─────────────────────────────────────────────────────────────────


def _stated(key: str) -> Callable[[Any, Facts], bool]:
    def fires(state: Any, facts: Facts) -> bool:
        from turbotab.core.interview import stated

        if key in ("grain", "repeat_kind") and getattr(state, key, None) is not None:
            return False
        if key == "form" and getattr(state, "purpose", None) != "prediction":
            return False
        return stated(key, state, facts.artifacts) is not None
    return fires


def _task_settled(state: Any, facts: Facts) -> bool:
    """default:task_settled: the outcome's kind is read at high confidence and nothing about it is
    still asked (``interview.route``'s skip of the task)."""
    info = _target_info(state, facts)
    return (info is not None and info.get("confidence") == "high"
            and getattr(state, "task", None) is None and _task_followup(state, facts) is None)


def _outcome_steps_not_applicable(state: Any, facts: Facts) -> bool:
    """default:outcome_steps_not_applicable: the task makes the event or the follow-up
    inapplicable."""
    if getattr(state, "target", None) is None:
        return False
    return not question_fires("event", state, facts) or not question_fires("follow_up", state, facts)


def _slot(name: str, attr: str | None = None, *values: Any) -> Callable[[Any, Facts], bool]:
    def fires(state: Any, facts: Facts) -> bool:
        value = getattr(state, name, None)
        if attr is not None:
            value = _get(value, attr)
        return value in values if values else bool(value)
    return fires


DEFAULTS: tuple[tuple[str, str, Callable[[Any, Facts], bool], tuple[str, ...], frozenset[str]], ...] = (
    # (crosswalk item, its stage, when it is stated, what it waits for, what it stands in for)
    ("default:design_observational", "question", _stated("design"), (), frozenset({"design"})),
    ("default:task_settled", "question", _task_settled, ("target",), frozenset({"task"})),
    ("default:outcome_steps_not_applicable", "question", _outcome_steps_not_applicable,
     ("task",), frozenset({"event", "follow_up"})),
    ("default:grain_stated", "whos_in", _stated("grain"), (), frozenset({"grain"})),
    ("default:repeat_kind_stated", "whos_in", _stated("repeat_kind"), ("grain",),
     frozenset({"repeat_kind"})),
    ("default:split_under_inference", "whos_in", _slot("purpose", None, "inference"),
     ("purpose",), frozenset({"split"})),
    ("default:missing_m", "whos_in", _slot("missing", "strategy", "multiple_imputation"),
     ("missing",), frozenset({"missing"})),
    ("default:complete_case_predictors", "whos_in", _slot("missing", "strategy", "complete_case"),
     ("missing",), frozenset({"missing"})),
    ("default:imputed_copies_pooled", "whos_in",
     _slot("repeat_kind", "repeat_kind", "imputed_copies"), ("repeat_kind",),
     frozenset({"repeat_kind"})),
    ("default:every_row_analysis", "whos_in",
     lambda s, f: bool(getattr(s, "exclusions", None)) and bool(getattr(s, "sensitivity", None)),
     ("exclusions", "set_sensitivity"), frozenset({"sensitivity"})),
    ("default:form-under-prediction", "models", _stated("form"), ("purpose",),
     frozenset({"exposure_forms"})),
    ("default:fit_statistics_withheld", "results", _slot("purpose", None, "inference"),
     ("purpose",), frozenset()),
    ("default:bootstrap_optimism", "results", _slot("split", "validation", "bootstrap"),
     ("split",), frozenset({"split"})),
    ("default:plan-wording", "writeup", _slot("plan_locked"), (), frozenset({"plan_locked"})),
    ("default:read-from-data", "data", lambda s, f: bool(_readings(f)), (),
     frozenset({"reading_confirmations"})),
)


# ── findings ─────────────────────────────────────────────────────────────────

# The question each family's voice routes it to (``finding_words``: the ``Voice``'s second
# argument, read from each family's sayer); a family with none is decided by its repair in Your
# data. The energy finding's lever is the energy question's.
FINDING_QUESTIONS: dict[str, str] = {
    "binary_text": "missing",
    "boolean_as_text": "missing",
    "positive_class": "event",
    "unnamed_columns": "roles",
    "constant_columns": "roles",
    "pack::clinical::impossible_vs_extreme": "exclusions",
    "pack::dietary::compositional": "roles",
    "pack::dietary::implausible_intake": "exclusions",
    "pack::dietary::energy_adjustment": "energy_adjustment",
    "pack::dietary::survey_weights": "roles",
    "pack::dietary::partial_design": "roles",
    "pack::metabolomics::acquisition_design": "roles",
    "pack::metabolomics::left_censored": "missing",
    "pack::metabolomics::repeated_subjects": "roles",
    "pack::genomics::counts_p_over_n": "models",
}


def finding_families() -> tuple[str, ...]:
    """Every finding family the engine names: the voice table's, its groups', the identifier
    families the repairs read, the flag family and the omics normalization's."""
    from turbotab.core.quest import NORMALIZATION_FINDING
    from turbotab.core.repairs import _IDENTIFIER_FAMILIES
    from turbotab.core.stages.finding_words import FAMILIES, GROUPS

    names = [*FAMILIES, *GROUPS, *_IDENTIFIER_FAMILIES, "voice::flag", NORMALIZATION_FINDING]
    return tuple(dict.fromkeys(names))


def _finding_fires(fam: str) -> Callable[[Any, Facts], bool]:
    def fires(state: Any, facts: Facts) -> bool:
        from turbotab.core.stages.finding_words import family

        return any(family(str(f.get("id"))) == fam for f in _listed(facts.artifacts.get("findings")))
    return fires


def _explore_fires(kind: str) -> Callable[[Any, Facts], bool]:
    return lambda state, facts: any(f.get("kind") == kind
                                    for f in _listed(facts.artifacts.get("explore")))


# ── results ──────────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def _order() -> tuple[Any, ...]:
    from turbotab.core.quest import _graph

    return tuple(_graph().order())


def blocked(state: Any) -> dict[str, tuple[str, ...]]:
    """Each compute stage -> the slots it (or a stage upstream) requires that are unset here, as
    the scheduler reads ``requires`` (``graph.Engine._compute_keys``); empty when it can
    compute."""
    values = state.model_dump(mode="json") if hasattr(state, "model_dump") else dict(state)
    out: dict[str, tuple[str, ...]] = {}
    for stage in _order():
        missing = [slot for slot in stage.requires if values.get(slot) is None]
        for dep in stage.deps:
            missing.extend(out.get(dep, ()))
        out[stage.name] = tuple(dict.fromkeys(missing))
    return out


def computable(state: Any) -> frozenset[str]:
    """The compute stages the graph can compute on this state."""
    return frozenset(name for name, missing in blocked(state).items() if not missing)


def _result_fires(name: str) -> Callable[[Any, Facts], bool]:
    return lambda state, facts: not blocked(state).get(name)


@lru_cache(maxsize=None)
def _writer_of(slot: str) -> str:
    """The question (or the declaration) whose answer writes ``slot``."""
    from turbotab.core.decisions import SLOTS
    from turbotab.core.interview import QUESTION_KEYS, SLOT_OF
    from turbotab.core.quest import DECLARATIONS

    for key in QUESTION_KEYS:
        if SLOT_OF.get(key, key) == slot:
            return key
    for decl in DECLARATIONS:
        if SLOTS[decl.kind] == slot:
            return decl.kind
    return slot


def _result_reads(name: str) -> tuple[str, ...]:
    upstream: list[str] = []
    seen: set[str] = set()

    def walk(stage_name: str) -> None:
        if stage_name in seen:
            return
        seen.add(stage_name)
        stage = next(s for s in _order() if s.name == stage_name)
        for dep in stage.deps:
            walk(dep)
        upstream.extend(stage.requires)

    walk(name)
    return tuple(dict.fromkeys(_writer_of(slot) for slot in upstream))


# ── consumers ────────────────────────────────────────────────────────────────


def first_consumer(slots: Iterable[str]) -> str | None:
    """The first compute stage, in the graph's order, that reads or requires one of ``slots``."""
    wanted = set(slots)
    return next((s.name for s in _order() if wanted & (set(s.reads) | set(s.requires))), None)


def _downstream(name: str) -> str | None:
    return next((s.name for s in _order() if name in s.deps), None)


def _kind_writes(kind: str) -> frozenset[str]:
    """The slots a kind writes, as far as its registration says without a decision in hand: its
    slot and the slots it writes beside it (``register_kind``'s ``also``)."""
    from turbotab.core import decisions as d

    out = {d.SLOTS[kind]} if kind in d.SLOTS else set()
    out |= set(d._ALSO.get(kind, {}))
    return frozenset(out)


# ── the registry ─────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def registry() -> dict[str, Item]:
    """Every surfaceable item, by key (``<shape>:<name>``)."""
    from turbotab.core import quest
    from turbotab.core.decisions import SLOTS
    from turbotab.core.interview import QUESTION_KEYS
    from turbotab.core.sequence import question_of

    out: dict[str, Item] = {}

    def add(entry: Item) -> None:
        assert entry.key not in out, entry.key
        out[entry.key] = entry

    questions: dict[str, Item] = {}
    for i, key in enumerate(QUESTION_KEYS):
        place = quest.QUESTIONS[key]
        holds = frozenset().union(*(_kind_writes(k) for k in quest.answering_kinds(key)))
        entry = Item(key=f"question:{key}", shape="question", stage=place.stage, item=place.item,
                     when=_question(key), reads=QUESTION_KEYS[:i], holds=holds,
                     consumer=first_consumer(holds), standing=quest.answer_holds(key))
        questions[key] = entry
        add(entry)

    declarations = {decl.kind: decl for decl in quest.DECLARATIONS}
    for kind in sorted(SLOTS):
        holds = _kind_writes(kind)
        consumer = first_consumer(holds)
        if kind == quest.FOLLOWS_ITS_STAGE:
            for stage, item in quest.SWEEP_ITEMS.items():
                add(Item(key=f"decision:{kind}:{stage}", shape="decision", stage=stage, item=item,
                         when=_sweep_fires(stage), holds=holds, consumer=consumer))
            continue
        if kind in declarations:
            decl = declarations[kind]
            place = decl.place
            add(Item(key=f"decision:{kind}", shape="decision", stage=place.stage, item=place.item,
                     when=decl.applies, reads=decl.reads, holds=holds, consumer=consumer,
                     standing=decl.holds))
            continue
        question = question_of(kind)
        if kind in quest.OTHER_KINDS and kind in KIND_FIRES:
            place = quest.OTHER_KINDS[kind]
            reads = (question,) if question is not None else ()
            add(Item(key=f"decision:{kind}", shape="decision", stage=place.stage, item=place.item,
                     when=KIND_FIRES[kind], reads=reads, holds=holds, consumer=consumer))
            continue
        # A kind that answers a Router question (its slot's, or beside it: ``ALSO_ANSWERS``).
        answered = question or next((k for k, kinds in quest.ALSO_ANSWERS.items()
                                     if kind in kinds), None)
        if answered is None:
            raise KeyError(f"the surfacing registry has no declaration for the kind {kind!r}")
        mirror = questions[answered]
        place = quest.kind_place(kind)
        add(Item(key=f"decision:{kind}", shape="decision", stage=place.stage, item=place.item,
                 when=mirror.when, reads=mirror.reads, holds=holds, consumer=consumer,
                 standing=mirror.standing))

    for item, stage, when, reads, holds in DEFAULTS:
        add(Item(key=f"default:{item.split(':', 1)[1]}", shape="default", stage=stage, item=item,
                 when=when, reads=reads, holds=holds, consumer=first_consumer(holds)))

    for fam in finding_families():
        route = FINDING_QUESTIONS.get(fam)
        place, _question_key = quest.finding_place({"id": fam, "routes_to": route})
        add(Item(key=f"noticing:{fam}", shape="noticing", stage=place.stage, item=place.item,
                 when=_finding_fires(fam), reads=(route,) if route else (),
                 holds=frozenset({"findings"}), consumer=first_consumer({"findings"})))
    for kind, place in quest.EXPLORE_FINDINGS.items():
        add(Item(key=f"noticing:explore::{kind}", shape="noticing", stage=place.stage,
                 item=place.item, when=_explore_fires(kind), reads=("split",),
                 holds=frozenset(), consumer=None))

    for name, (stage, card) in quest.COMPUTE.items():
        add(Item(key=f"result:{name}", shape="result", stage=stage, item=card or "",
                 when=_result_fires(name), reads=_result_reads(name), holds=frozenset(),
                 consumer=_downstream(name)))
    return out


def by_shape(shape: str) -> list[Item]:
    return [i for i in registry().values() if i.shape == shape]


def for_kind(kind: str) -> Item:
    """The item a decision kind is surfaced by (one per stage for "Confirm all": its first)."""
    found = registry().get(f"decision:{kind}")
    if found is None:
        found = next((i for k, i in registry().items() if k.startswith(f"decision:{kind}:")), None)
    if found is None:
        raise KeyError(kind)
    return found


# ── the crosswalk's rules ────────────────────────────────────────────────────


def rule_name(item_id: str) -> str:
    """``q:lens`` -> ``fires_q_lens``; ``decision:finding-disposition`` ->
    ``fires_decision_finding_disposition``."""
    return RULE_PREFIX + re.sub(r"[^a-z0-9]+", "_", item_id.lower()).strip("_")


@lru_cache(maxsize=1)
def carriers() -> dict[str, tuple[str, ...]]:
    """Each crosswalk item -> the registry keys that carry it."""
    out: dict[str, list[str]] = {}
    for key, entry in registry().items():
        if entry.item:
            out.setdefault(entry.item, []).append(key)
    return {item: tuple(keys) for item, keys in sorted(out.items())}


def crosswalk_rules() -> dict[str, str]:
    """Each wired crosswalk item -> its ``rule`` (``module:function``)."""
    return {item: f"{RULE_MODULE}:{rule_name(item)}" for item in carriers()}


def _rule(item_id: str) -> Callable[[Any, Facts], bool]:
    keys = carriers()[item_id]

    def fires(state: Any, facts: Facts | None = None) -> bool:
        facts = facts if facts is not None else Facts()
        return any(registry()[k].fires(state, facts) for k in keys)

    fires.__name__ = fires.__qualname__ = rule_name(item_id)
    fires.__doc__ = f"Whether the crosswalk item {item_id!r} fires: any of {', '.join(keys)}."
    return fires


@lru_cache(maxsize=1)
def _rules_by_name() -> dict[str, str]:
    return {rule_name(item): item for item in carriers()}


def __getattr__(name: str) -> Any:
    if name.startswith(RULE_PREFIX):
        item = _rules_by_name().get(name)
        if item is not None:
            fn = _rule(item)
            globals()[name] = fn
            return fn
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ── disclosure, the one place a mode is read (§4.4) ──────────────────────────

MODES: tuple[str, ...] = ("standard", "knows_the_field")


@dataclass(frozen=True)
class Shown:
    stage: str
    id: str
    key: str
    label: str
    level: int


def disclosure(log: Any, mode: str = "standard", remembered: Sequence[str] = ()) -> list[Shown]:
    """How each line of a quest log is drawn under ``mode`` (§1.5, §4.4): the one open Decide of
    the first reached stage still asking at level 3, the rest at 1. "I know this field" starts a
    Decide whose answer is ``remembered`` (the same on earlier projects in this lens) collapsed at
    level 1, still listed and counted: a mode changes the level, never the label."""
    if mode not in MODES:
        raise ValueError(mode)
    out: list[Shown] = []
    opened = False
    for stage in log.stages:
        for line in stage.lines:
            level = 1
            if (stage.reached and not opened and line.label == "Decide" and line.status == "open"
                    and line.counted):
                opened = True
                level = 1 if mode == "knows_the_field" and line.key in remembered else 3
            out.append(Shown(stage=stage.key, id=line.id, key=line.key, label=line.label,
                             level=level))
    return out


__all__ = [
    "DEFAULTS", "FINDING_QUESTIONS", "Item", "KIND_FIRES", "MODES", "SHAPES", "Shown",
    "Surfaceable", "blocked", "by_shape", "carriers", "computable", "crosswalk_rules",
    "disclosure", "finding_families", "first_consumer", "for_kind", "question_fires", "registry",
    "rule_name",
]

"""What each open declaration line offers (SWARM_PLAN §10 package 1; the sixth review's §0): the
decisions with no Router question that the quest log lists as lines of their own
(``quest.DECLARATIONS``), each composed from what the engine itself defines for it, so the driver's
``ask`` lists them and ``preview`` and ``answer`` take them by id like a question's options.

**Where the options come from**, never a list kept here:

* **the decision's own model** (``turbotab/core/decisions.py``): the values each field accepts (a
  ``Literal``'s values, ``true`` or ``false``, a number inside the field's bounds, the table's
  columns for a column or a list of columns), its default, and the answer recorded for the line
  now (its state slot), which each field starts at, so an option changes only what it names;
* **the method contracts that record it** (``core/contracts.py``): a contract whose ``decision`` is
  the kind, or one with no decision of its own whose every option is a value of one of the
  kind's fields (the batch and multiplicity contracts); each contract option that is a value (or
  the name) of a field no card holds (**Held**, below) is an option, with its label, its
  customary and sound labels and its leash rung for the purpose, in the contract's order;
* **what the engine serves for the line**: the seal plan's validation schemes (each with its
  words and, for internal–external validation, its cluster, or that it needs one), the proposals'
  model sequence card, the sensitivity stage's every-row analysis
  (``stages.sensitivity.EVERY_ROW``), the columns whose names read as a batch
  (``methods.batch.batch_columns_of``).

**Options.** One per value of the field the contracts' options are values of (else the model's
first field with a fixed set of values), each recording the answer as it stands with that value:
its id is ``<kind>:<value>`` (``set_selection:elastic_net``), or ``<kind>:<field>:<value>`` where
another field's value took that id first (the levers' ``none``). Values the contracts do not label
are offered too, unlabeled; two that would record the same decision are one. An option the answer
as it stands leaves a needed field unset for records nothing by itself and lists the fields it
``needs``: a field its model requires, one the seal plan says a scheme needs (internal–external
validation's cluster), one without which the model does not read the option at all, or one the
engine's answer-time check sets in its way forward when it refuses the option as it stands
(:func:`needs_said`: the variance filter's ``top`` needs how many to keep). The scales question
offers no scale (``set_scales:none``) and one scale of one's own (``set_scales:custom``, its fields
the scale's); the sensitivity analyses, the every-row analysis or none.

**Fields.** Every other field the model takes from a client (``fields``; ``answer <id> --set
name=value``): a choice (``choices``), a number (``type``, with its ``bounds``), a column
(``columns``) or a comma list of them (``type: "columns"``), or a text of one's own (``type:
"text"``: a scale's name). A field the server fills itself (the calibration's adjustment set,
``SetMeasurementError``'s docstring) is not offered. The decision is read by the model before it
is posted, so an answer missing a field the model needs says which here.

**Held** (:func:`held`), offered neither as an option nor as a field, and recorded by every option
at one value: a field the shell's own cards never ask and always post at one value (the cards'
fixture's ``fixed``, :data:`CARDS`: the selection's ``where`` and its sensitivity analysis, the
explanation as an effect), at that value; and an attestation that keeps a blocked answer
(``acknowledged``), at its default, false, since the engine offers it only as a way forward of its
refusal of the answer as it stands, which ``answer`` takes by its id. An answer recorded with a
held field set is posted with the held value.

**Not composed:** a join and a data dictionary need a file the driver cannot hand over
(:data:`NEEDS_A_FILE`); the outcome views are recorded by the views themselves (``only_recorded``).
"""
from __future__ import annotations

import json
import math
import typing
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from research.swarm.options import Unanswerable, _art, _option, canonical

Served = Callable[[str], Any]

# The declarations whose decision names a file the person adds: the driver hands over none.
NEEDS_A_FILE = {"join_files": "A join adds a second file, which the driver cannot hand over.",
                "import_codebook": "A data dictionary is a file, which the driver cannot hand over."}
# A field the server fills from the state, never the client (the model's own docstring).
SERVER_FILLED = {"set_measurement_error": ("adjustment",)}
ROW_ID = "__row_id"
# What the shell's Predict declaration cards post: its ``fixed`` holds the fields a card never
# asks and always sends at one value (``core/tests/test_w1_declaration_cards.py`` holds them to the
# cards' payloads and to decisions.py's defaults).
CARDS = Path(__file__).resolve().parents[2] / (
    "turbotab/frontend/src/quest/cards/predict/payloads.fixture.json")


def held_by_the_cards() -> dict[str, dict[str, Any]]:
    """Each declaration's fields the shell's cards hold fixed, with the value they post: read from
    the cards' own fixture (:data:`CARDS`), never kept here. A fixture that cannot be read refuses
    to compose (the driver says so on the line) rather than offer them."""
    fixed = json.loads(CARDS.read_text("utf-8"))["fixed"]
    return {str(k): dict(v) for k, v in fixed.items()}


def attestations(model: Any) -> list[str]:
    """The fields of a decision model that keep a blocked answer: decisions.py names each
    ``acknowledged`` (or ``<what>_acknowledged``), false unless the engine's refusal of the answer
    as it stands offers it as a way forward, which ``answer`` takes by its id."""
    return [n for n in model.model_fields if n == "acknowledged" or n.endswith("_acknowledged")]


def held(kind: str) -> dict[str, Any]:
    """The fields of ``kind`` no option or ``--set`` offers, each with the value every option
    records: those the shell's cards hold fixed, at the value they post, and its attestations, at
    the model's default (false)."""
    model = models()[kind]
    out = {n: _default(model.model_fields[n]) for n in attestations(model)}
    out.update(held_by_the_cards().get(kind, {}))
    return out


def models() -> dict[str, Any]:
    """Each decision kind's model, from the engine's own union (``decisions.Decision``)."""
    from turbotab.core import decisions

    members = typing.get_args(typing.get_args(decisions.Decision)[0])
    return {m.model_fields["kind"].default: m for m in members}


def kinds() -> list[str]:
    """The declarations this module composes: every one the quest log lists that is not only
    recorded and needs no file (the module docstring), in the quest log's order."""
    from turbotab.core.quest import DECLARATIONS

    return [d.kind for d in DECLARATIONS if not d.only_recorded and d.kind not in NEEDS_A_FILE]


# ── a field's type, read from its annotation ─────────────────────────────────


def _parts(annotation: Any) -> list[Any]:
    origin = typing.get_origin(annotation)
    if origin in (typing.Union, getattr(__import__("types"), "UnionType", None)):
        return [a for a in typing.get_args(annotation) if a is not type(None)]
    return [annotation]


def _literal(annotation: Any) -> list[Any] | None:
    values: list[Any] = []
    for part in _parts(annotation):
        if typing.get_origin(part) is typing.Literal:
            values += list(typing.get_args(part))
    return values or None


def _bounds(info: Any) -> dict[str, float]:
    out: dict[str, float] = {}
    for m in getattr(info, "metadata", ()):
        for name in ("ge", "gt", "le", "lt"):
            value = getattr(m, name, None)
            if value is not None:
                out[name] = value
    return out


def field_type(name: str, info: Any) -> dict[str, Any] | None:
    """What a field takes: ``choice`` (``values``), ``bool``, ``int`` or ``float`` (``bounds``),
    ``column``, ``columns``, ``text``, or ``model`` / ``models`` (a nested answer); None for one no
    client sets here."""
    parts = _parts(info.annotation)
    values = _literal(info.annotation)
    if values is not None:
        return {"kind": "choice", "values": values, "optional": type(None) in
                typing.get_args(info.annotation)}
    if parts == [bool]:
        return {"kind": "bool"}
    if parts in ([int], [float]):
        return {"kind": parts[0].__name__, "bounds": _bounds(info)}
    if parts == [str]:
        return {"kind": "column" if name not in ("name", "instrument", "label") else "text"}
    origin = typing.get_origin(parts[0])
    if origin is list:
        inner = typing.get_args(parts[0])
        if inner and inner[0] is str:
            return {"kind": "columns"}
        if inner and hasattr(inner[0], "model_fields"):
            return {"kind": "models", "model": inner[0]}
    return None


def _default(info: Any) -> Any:
    from pydantic_core import PydanticUndefined

    if info.default is not PydanticUndefined:
        return json.loads(json.dumps(info.default, default=str))
    if info.default_factory is not None:
        return json.loads(json.dumps(info.default_factory(), default=str))
    return None


# ── the engine's options for a kind ──────────────────────────────────────────


def contract_options(kind: str, fields: Mapping[str, dict[str, Any]], purpose: str | None
                     ) -> list[tuple[str, Any, dict[str, Any]]]:
    """Each contract option of ``kind`` that is a field's value: ``(field, value, labels)`` in
    the contracts' order, the labels the contract's for ``purpose`` (the module docstring)."""
    from turbotab.core import contracts

    out = []
    for c in contracts.contracts().values():
        keys = [o.key for o in c.options]
        # The field every option of the contract is a value of, where one is (the imbalance
        # correction's "none" is the imbalance field's, though the forms field takes "none" too).
        home = next((n for n, t in fields.items() if set(keys) <= set(t.get("values") or ())),
                    None)
        mine = c.decision == kind or (not c.decision and home is not None)
        if not mine:
            continue
        for o in c.options_for(purpose or "prediction"):
            field = home if home is not None else next(
                (n for n, t in fields.items() if o["key"] in (t.get("values") or ())), None)
            if field is not None:
                out.append((field, o["key"], o))
            elif fields.get(o["key"], {}).get("kind") == "bool":
                out.append((o["key"], True, o))
    return out


def contract_question(kind: str) -> str | None:
    """The question the one contract that records ``kind`` asks, in its words; None where the
    contracts state it rather than ask it, or where several record it (each asks of its own part:
    the levers' four contracts)."""
    from turbotab.core import contracts

    recording = [c for c in contracts.contracts().values() if c.decision == kind]
    if len(recording) != 1:
        return None
    question = recording[0].question
    if not question or question.startswith("(") or question.strip() == "stated":
        return None
    return question


class Ctx:
    """What a composer reads: the project's view and the served artifacts."""

    def __init__(self, view: Mapping[str, Any], served: Served):
        self.view, self.served = view, served
        self.state: dict[str, Any] = dict(view.get("state") or {})

    def columns(self) -> list[str]:
        cols = _art(self.served, "oriented").get("columns") or _art(self.served, "ingest").get("columns")
        return [str(c["name"]) for c in cols or [] if isinstance(c, Mapping) and c.get("name")
                and c["name"] != ROW_ID]

    def recorded(self, kind: str) -> Mapping[str, Any]:
        """The answer recorded for ``kind`` now, as its state slot holds it (``decisions.SLOTS``)."""
        from turbotab.core.decisions import SLOTS

        value = self.state.get(SLOTS.get(kind, ""))
        return value if isinstance(value, Mapping) else {}


def _field_entry(kind: str, name: str, t: dict[str, Any], initial: Any, ctx: Ctx,
                 applies: list[str] | None = None, into: str | None = None) -> dict[str, Any]:
    entry: dict[str, Any] = {"name": name, "initial": initial, "_type": t, "_into": into}
    if t["kind"] == "choice":
        entry["choices"] = [{"value": v} for v in t["values"]]
    elif t["kind"] == "bool":
        entry["choices"] = [{"value": "true"}, {"value": "false"}]
    elif t["kind"] in ("int", "float"):
        entry.update(type="number", bounds=t["bounds"] or None)
    elif t["kind"] == "column":
        entry["choices"] = [{"value": c} for c in ctx.columns()]
    elif t["kind"] == "columns":
        entry.update(type="columns", choices=[{"value": c} for c in ctx.columns()])
    elif t["kind"] == "text":
        entry["type"] = "text"
    if applies:
        entry["applies_to"] = applies
    return entry


def compose(kind: str, view: Mapping[str, Any], served: Served) -> dict[str, Any]:
    """The options, fields and served data of the declaration ``kind`` (the module docstring)."""
    ctx = Ctx(view, served)
    model = models()[kind]
    purpose = ctx.state.get("purpose")
    skip = set(SERVER_FILLED.get(kind, ())) | {"kind"}
    fixed = held(kind)
    types = {n: t for n, f in model.model_fields.items() if n not in skip
             and (t := field_type(n, f)) is not None}
    recorded = ctx.recorded(kind)
    base: dict[str, Any] = {"kind": kind}
    for n, f in model.model_fields.items():
        if n in fixed:
            base[n] = fixed[n]
        elif n in types:
            base[n] = json.loads(json.dumps(recorded[n], default=str)) if n in recorded \
                else _default(f)
    # The fields an option or --set may set: none held (:func:`held`).
    types = {n: t for n, t in types.items() if n not in fixed}
    special = SPECIAL.get(kind)
    out = special(ctx, kind, base, types) if special is not None else None
    if out is None:
        out = _generic(ctx, kind, base, types, purpose)
    words = contract_question(kind)
    if words:
        out["words"] = {"question": words}
    out.setdefault("data", {})
    out["data"] = {k: v for k, v in out["data"].items() if v not in (None, [], {}, "")}
    return out


def _required(model: Any, base: Mapping[str, Any], types: Mapping[str, Any]) -> list[str]:
    """The fields the model needs that the answer as it stands leaves unset."""
    return [n for n, f in model.model_fields.items() if n in types and f.is_required()
            and base.get(n) is None]


def _needs_to_read(decision: Mapping[str, Any], types: Mapping[str, dict[str, Any]], ctx: Ctx
                   ) -> list[str]:
    """The one unset field without which the decision's own model does not read ``decision``,
    where setting it to a value it offers makes the model read it (internal–external validation's
    cluster, with no seal plan to say so); else none."""
    from pydantic import ValidationError

    from turbotab.core.decisions import parse_decision

    try:
        parse_decision(dict(decision))
        return []
    except ValidationError:
        pass
    for name, t in types.items():
        if decision.get(name) is not None:
            continue
        values = (t.get("values") if t["kind"] == "choice"
                  else ctx.columns() if t["kind"] == "column" else None)
        for value in [v for v in values or () if v is not None][:1]:
            try:
                parse_decision({**decision, name: value})
            except ValidationError:
                continue
            return [name]
    return []


def needs_said(decision: Mapping[str, Any]) -> list[str]:
    """The fields the engine's own answer-time check says ``decision`` needs set: where it refuses
    the decision on what the decision alone shows (``decisions.validate`` with no context checks
    only that) and a way forward sets fields the decision leaves unset, keeping every other field
    as it is (the variance filter's ``top`` with no ``keep``: ``keep_how_many``), those fields;
    else none."""
    from turbotab.core.decisions import Refusal, validate

    try:
        validate(dict(decision), None)
    except Refusal as exc:
        for way in exc.exits or ():
            filled = way.get("decision") if isinstance(way, Mapping) else None
            if not isinstance(filled, Mapping):
                continue
            filled = json.loads(json.dumps(filled, default=str))
            unset = [n for n, v in filled.items() if decision.get(n) is None and v is not None]
            if unset and all(filled.get(n) == v for n, v in decision.items() if n not in unset):
                return unset
    except Exception:  # noqa: BLE001 - a decision the model does not read says so when answered
        return []
    return []


def _generic(ctx: Ctx, kind: str, base: dict[str, Any], types: dict[str, dict[str, Any]],
             purpose: str | None) -> dict[str, Any]:
    offered = contract_options(kind, types, purpose)
    primary = offered[0][0] if offered else next(
        (n for n, t in types.items() if t["kind"] == "choice"), None)
    options: list[dict[str, Any]] = []
    seen: set[str] = set()
    required = _required(models()[kind], base, types)

    def offer(field: str, value: Any, labels: Mapping[str, Any] | None) -> None:
        decision = {**base, field: value}
        key = canonical(decision)
        if key in seen:
            return
        seen.add(key)
        oid = f"{kind}:{field if value is True else value}"
        if any(o["id"] == oid for o in options):  # the same value of another field: named by both
            oid = f"{kind}:{field}:{value}"
        extra = ({"labels": {"customary": labels.get("customary"), "sound": labels.get("sound")},
                  "rung": labels.get("rung")} if labels else {})
        unset = ([n for n in required if n != field] or _needs_to_read(decision, types, ctx)
                 or needs_said(decision))
        options.append(_option(oid, None if unset else decision, label=(labels or {}).get("label"),
                               _base=decision if unset else None, needs=unset or None,
                               **extra))

    for field, value, labels in offered:
        offer(field, value, labels)
    if primary is not None:
        for value in types[primary].get("values") or ():
            offer(primary, value, None)
    if not options:  # no field with a fixed set of values: one answer, its fields set by hand
        options.append(_option(f"{kind}:answer", None if required else dict(base),
                               _base=dict(base) if required else None,
                               needs=required or None))
    # Options that each set a different field leave every field open to --set; otherwise the
    # field the options are values of is chosen by the option.
    several = len({f for f, _, _ in offered} | ({primary} if primary else set())) > 1
    fields = [_field_entry(kind, n, t, base.get(n), ctx) for n, t in types.items()
              if t["kind"] != "models" and (n != primary or several)]
    return {"options": options, "fields": fields}


# ── the lines whose answer the engine serves, or whose shape is its own ─────


def _validation(ctx: Ctx, kind: str, base: dict[str, Any], types: dict[str, dict[str, Any]]
                ) -> dict[str, Any] | None:
    """The seal plan's validation schemes, each with its words; internal–external validation
    with the cluster the plan names."""
    plan = _art(ctx.served, "seal_plan")
    schemes = [dict(v) for v in (plan.get("validation") or {}).get("options") or []
               if isinstance(v, Mapping) and v.get("validation")]
    if not schemes:
        return None
    options, folding = [], []
    for v in schemes:
        decision = {**base, "validation": v["validation"], "cluster": v.get("cluster")}
        oid = f"{kind}:{v['validation']}"
        # The plan says a scheme needs the cluster column whose levels are its folds
        # (``ValidationOption.needs_cluster``); one the plan names already needs nothing more.
        needs = ["cluster"] if v.get("needs_cluster") and not v.get("cluster") else None
        if v.get("needs_cluster"):
            folding.append(oid)
        options.append(_option(oid, None if needs else decision, label=v.get("label"),
                               consequence=v.get("measures"), cost=v.get("cost"),
                               caution=v.get("caution"), _base=decision if needs else None,
                               needs=needs))
    fields = [_field_entry(kind, n, t, base.get(n), ctx,
                           applies=folding if n == "cluster" and folding else None)
              for n, t in types.items() if n != "validation"]
    for f in fields:
        if f["name"] == "cluster":  # a column whose levels fold, never the outcome (the shell's card)
            f["choices"] = [c for c in f.get("choices") or []
                            if c.get("value") != ctx.state.get("target")]
    data = {"reason": (plan.get("validation") or {}).get("reason")}
    return {"options": options, "fields": fields, "data": data}


def _batch(ctx: Ctx, kind: str, base: dict[str, Any], types: dict[str, dict[str, Any]]
           ) -> dict[str, Any]:
    """The batch methods (the contract's, and that the column is not a batch), on the columns
    whose names read as a batch, a run or a plate, or the one already answered."""
    from turbotab.core.methods.batch import batch_columns_of

    read = batch_columns_of(ctx.columns(), ctx.state.get("target"))
    answered = (ctx.state.get("batch") or {}).get("column")
    candidates = list(dict.fromkeys([*([answered] if answered else []), *read]))
    if base.get("column") is None and len(candidates) == 1:
        base = {**base, "column": candidates[0]}
    out = _generic(ctx, kind, base, types, ctx.state.get("purpose"))
    for f in out["fields"]:
        if f["name"] == "column":
            f["choices"] = [{"value": c} for c in candidates]
    out["data"] = {"batch_columns": candidates}
    return out


def _sensitivity(ctx: Ctx, kind: str, base: dict[str, Any], types: dict[str, dict[str, Any]]
                 ) -> dict[str, Any]:
    """The analysis beside the primary on every row (the stage's own, Banna et al. 2017), or none."""
    from turbotab.core.stages.sensitivity import BANNA, EVERY_ROW

    return {"options": [
        _option(f"{kind}:every_row", {"kind": kind, "analyses": [{"label": EVERY_ROW, "rules": []}]},
                label=EVERY_ROW, consequence=BANNA),
        _option(f"{kind}:none", {"kind": kind, "analyses": []})], "fields": []}


def _model_sequence(ctx: Ctx, kind: str, base: dict[str, Any], types: dict[str, dict[str, Any]]
                    ) -> dict[str, Any] | None:
    """The proposals' model sequence card, as a person taking its offer records it, with Model 1's
    columns as a field."""
    card = _art(ctx.served, "proposals").get("model_sequence") or {}
    decision = card.get("decision")
    if not isinstance(decision, Mapping):
        return None
    fields = [_field_entry(kind, "model_1", types["model_1"], decision.get("model_1"), ctx)]
    return {"options": [_option(f"{kind}:proposed", dict(decision), label=card.get("label"))],
            "fields": fields,
            "data": {k: card.get(k) for k in ("exposure", "columns", "reason") if card.get(k)}}


def _scales(ctx: Ctx, kind: str, base: dict[str, Any], types: dict[str, dict[str, Any]]
            ) -> dict[str, Any]:
    """No scale scored, or one scale of one's own: its fields the scale's own (``ScaleSpec``)."""
    sub = types["scales"]["model"]
    inner = {n: t for n, f in sub.model_fields.items() if (t := field_type(n, f)) is not None
             and t["kind"] not in ("models",)}
    spec = {n: _default(f) for n, f in sub.model_fields.items() if n in inner}
    options = [_option(f"{kind}:none", {"kind": kind, "scales": []}),
               _option(f"{kind}:custom", None, _base={"kind": kind, "scales": [spec]},
                       needs=[n for n, f in sub.model_fields.items()
                                  if n in inner and f.is_required()])]
    fields = [_field_entry(kind, n, t, spec.get(n), ctx, applies=[f"{kind}:custom"],
                           into="scales") for n, t in inner.items()]
    return {"options": options, "fields": fields}


SPECIAL: dict[str, Callable[..., dict[str, Any] | None]] = {
    "set_validation": _validation, "set_batch": _batch, "set_sensitivity": _sensitivity,
    "set_model_sequence": _model_sequence, "set_scales": _scales,
}


# ── an answer from an id and fields ──────────────────────────────────────────


def owner(oid: str, declared: Mapping[str, Any]) -> str | None:
    """The declaration an option id belongs to (its ``<kind>:`` prefix), or None."""
    kind = oid.split(":", 1)[0]
    return kind if kind in declared and ":" in oid else None


def _value(field: Mapping[str, Any], raw: str) -> Any:
    t = field["_type"]
    allowed = [str(c.get("value")) for c in field.get("choices") or []]
    if t["kind"] == "bool":
        if raw not in ("true", "false"):
            raise Unanswerable(f"{field['name']} is true or false, not {raw!r}.")
        return raw == "true"
    if t["kind"] == "choice":
        if raw not in allowed:
            raise Unanswerable(f"{field['name']} is one of {', '.join(allowed)}, not {raw!r}.")
        return next(v for v in t["values"] if str(v) == raw)
    if t["kind"] in ("int", "float"):
        try:
            value = float(raw)
        except ValueError:
            value = math.nan
        b = t["bounds"]
        whole = t["kind"] == "int"
        ok = math.isfinite(value) and (not whole or value.is_integer()) \
            and ("ge" not in b or value >= b["ge"]) and ("gt" not in b or value > b["gt"]) \
            and ("le" not in b or value <= b["le"]) and ("lt" not in b or value < b["lt"])
        if not ok:
            edges = ", ".join(f"{k} {v:g}" for k, v in b.items())
            raise Unanswerable(f"{field['name']} takes {'a whole number' if whole else 'a number'}"
                               f"{f' ({edges})' if edges else ''}; not {raw!r}.")
        return int(value) if whole else value
    if t["kind"] == "column":
        if raw not in allowed:
            raise Unanswerable(f"{field['name']} is one of the table's columns, not {raw!r}.")
        return raw
    if t["kind"] == "columns":
        chosen = [c.strip() for c in raw.split(",") if c.strip()]
        stray = [c for c in chosen if c not in allowed]
        if stray:
            raise Unanswerable(f"{field['name']} takes the table's columns, comma-separated; "
                               f"{', '.join(stray)} {'is' if len(stray) == 1 else 'are'} not one.")
        return chosen
    if not raw.strip():
        raise Unanswerable(f"{field['name']} takes words of one's own; none were given.")
    return raw.strip()


def resolve(composed: Mapping[str, Any], ids: list[str], sets: Iterable[str]) -> dict[str, Any]:
    """The decision ``ids`` (one) and ``sets`` name among a declaration's options and fields, read
    by its model before it is posted (:class:`options.Unanswerable` says what does not read)."""
    from pydantic import ValidationError

    from turbotab.core.decisions import parse_decision

    if len(ids) != 1:
        raise Unanswerable("A declaration takes one option at a time.")
    options = {str(o["id"]): o for o in composed.get("options") or []}
    chosen = options.get(ids[0])
    if chosen is None:
        raise Unanswerable(f"No option {ids[0]} is offered now; ask lists each open line's "
                           f"options.")
    decision = json.loads(json.dumps(chosen.get("decision") or chosen.get("_base") or {}))
    fields = {str(f["name"]): f for f in composed.get("fields") or []}
    given: set[str] = set()
    for item in sets:
        name, eq, raw = item.partition("=")
        name = name.strip()
        if not eq:
            raise Unanswerable(f"--set takes name=value, not {item!r}.")
        field = fields.get(name)
        if field is None:
            raise Unanswerable(f"No field {name!r} is offered for this line; ask lists its fields.")
        applies = field.get("applies_to")
        if applies and ids[0] not in applies:
            raise Unanswerable(f"The field {name} does not apply to option {ids[0]}.")
        value = _value(field, raw.strip())
        if field.get("_into"):
            decision[field["_into"]][0][name] = value
        else:
            decision[name] = value
        given.add(name)
    missing = [n for n in chosen.get("needs") or () if n not in given]
    if missing:
        raise Unanswerable(f"Option {ids[0]} takes {', '.join(missing)}: set "
                           f"{'it' if len(missing) == 1 else 'each'} with --set.")
    try:
        parse_decision(decision)
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first.get("loc") or () if p != decision["kind"])
        raise Unanswerable(f"This answer does not read as one: {where or 'the answer'}: "
                           f"{first.get('msg')}.") from None
    return decision


__all__ = ["CARDS", "NEEDS_A_FILE", "SERVER_FILLED", "attestations", "compose", "contract_options",
           "field_type", "held", "held_by_the_cards", "kinds", "models", "owner", "resolve"]

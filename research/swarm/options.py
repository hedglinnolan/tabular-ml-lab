"""What the open question offers, read from what the server serves (DRIVER_PLAN D1).

The driver's ``ask`` shows the Router's open question as the quest log and its ask card serve it,
and ``preview`` and ``answer`` take one of its options by id. This module composes those options
from the artifacts that offer them, the ones ``turbotab/core/reference/journeys.py`` reads for its
answers (the proposals' labeled options, the seal plan, the shelf, the estimand, adjustment and
form cards, the outcome's reading) and the question's teaching entry (``GET /teaching``). It lists
every option those artifacts offer, never only the first, and chooses none.

**Ids.** An option's id is the one the server serves for it: a labeled option's ``key``, a
teaching option's ``value``, a family's ``key``, a column's name, a group's ``key``, an outcome
level. A way forward (a refusal's exit, an ask card's exit, a quest line's press) carries a label
and a decision but no id of its own, and its label holds backticks the analyst's shell may not
pass, so it is named by its decision: ``x-`` and the first 8 hex digits of the SHA-256 of the
decision's canonical JSON (:func:`exit_id`), the same for the same decision.

**Words.** Every label, consequence and note is the server's, as served. Where the server serves
no label (a column, a level), the option has none; the driver adds no words to a question. The
strings it writes into a decision are the app's own: the reasons its exclusions card records for
a range of plausible values and for levels of one's own (:data:`CUSTOM_RANGE`,
:data:`CUSTOM_LEVELS`), and for a range of who the study is about the reason the teaching content
states (``content.ELIGIBILITY_RANGE_REASON``, :func:`eligibility_range`), which the card records
too.

**Fields.** A question answered with one value per column (the roles, the adjustment answers, the
forms) or with a setting beside its option (the split's validation, the estimand's measure, the
exclusions' ranges of one's own, ``custom`` and ``eligibility``: a numeric column, and ``low``,
``high`` or both, as numbers; its levels of one's own: ``level_column``, any column but the
outcome, and the levels it excludes and the codes that record a refused or a don't-know answer,
``exclude``, ``refused`` and ``dont_know``, each a list by commas; and for each, ``missing``, what
the rule does with a row it cannot confirm, one of the handlings the engine's rule models take)
lists ``fields``;
``answer <option> --set <name>=<value>`` sets one, from the field's served ``choices``, as a
number (``"type": "number"``) or as a list (``"type": "list"``: ``--set exclude=1,3``, each level
kept as written, for the server to read in its one spelling). Each field's ``initial`` is what the
card itself shows chosen (a proposal, the first-ranked measure, the card's own handling of a row a
rule cannot confirm), so an option's ``decision`` is exactly what pressing it records; a field the
card leaves unchosen starts empty.

**Several at once.** The lens and the models take more than one option (``multi``): ``answer a b``
records them together. So do the exclusions, a rule for each option named, in the order named
(``answer custom levels levels``: a range of one's own and two level rules), each offered screen
adding its own rule; keeping every row is answered alone. A setting goes to the one option named
that takes it; where more than one does, its place among the options named is written before it
(``--set 2.level_column=DIQ010 --set 3.level_column=DIQ050``), so each rule has its own column,
levels and handling of a row it cannot confirm. A way forward is answered alone, never beside
the question's own options.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Callable, Iterable, Mapping

Served = Callable[[str], Any]  # stage name -> its artifact as GET /stages/<stage> serves it, or None


# ── ids for what carries none ────────────────────────────────────────────────


def canonical(decision: Any) -> str:
    return json.dumps(decision, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def exit_id(decision: Mapping[str, Any]) -> str:
    """A way forward's id: its decision's digest (the module docstring)."""
    return "x-" + hashlib.sha256(canonical(decision).encode("utf-8")).hexdigest()[:8]


def exits(items: Iterable[Mapping[str, Any]] | None, **extra: Any) -> list[dict[str, Any]]:
    """Ways forward as the driver lists them: the server's label and decision, with the decision's
    id; one without a decision (a "wait for the fit") has no id and records nothing."""
    out = []
    for item in items or []:
        if not isinstance(item, Mapping):
            continue
        decision = item.get("decision")
        out.append({"id": exit_id(decision) if isinstance(decision, Mapping) else None,
                    "label": item.get("label"), "decision": decision, **extra})
    return out


def public(obj: Any) -> Any:
    """``obj`` without the driver's private keys (``_apply`` and the like), for printing."""
    if isinstance(obj, dict):
        return {k: public(v) for k, v in obj.items() if not (isinstance(k, str) and k[:1] == "_")}
    if isinstance(obj, list):
        return [public(v) for v in obj]
    return obj


# ── the question's own words ─────────────────────────────────────────────────


def _art(served: Served, stage: str) -> dict[str, Any]:
    value = served(stage)
    return value if isinstance(value, dict) else {}


def _taught(entry: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    return [dict(o) for o in (entry or {}).get("options") or [] if isinstance(o, Mapping)]


def _option(oid: Any, decision: dict[str, Any] | None, *, label: Any = None,
            consequence: Any = None, **extra: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"id": str(oid), "label": label, "decision": decision}
    if consequence is not None:
        out["consequence"] = consequence
    out.update({k: v for k, v in extra.items() if v is not None})
    return out


def _from_teaching(entry: Mapping[str, Any] | None, build: Callable[[str], dict[str, Any] | None],
                   only: Iterable[str] | None = None) -> list[dict[str, Any]]:
    keep = None if only is None else {str(v) for v in only}
    return [_option(o["value"], build(str(o["value"])), label=o.get("label"),
                    consequence=o.get("consequence"))
            for o in _taught(entry) if keep is None or str(o.get("value")) in keep]


def _labeled(question: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """A LabeledQuestion's options (``custom_sound``), soundest first for the purpose."""
    return [dict(o) for o in (question or {}).get("options") or [] if isinstance(o, Mapping)]


def _two_labels(option: Mapping[str, Any]) -> dict[str, Any] | None:
    customary, sound = option.get("customary"), option.get("sound")
    if customary is None and sound is None:
        return None
    return {"customary": customary, "sound": sound}


def _num(value: Any) -> str:
    """A holdout share as its id: ``0.2`` → ``"0.2"``, ``0.0`` → ``"0"``."""
    return format(float(value), "g")


# ── the composers, one per Router question ───────────────────────────────────


class Context:
    def __init__(self, key: str, step: Mapping[str, Any], view: Mapping[str, Any], served: Served,
                 entry: Mapping[str, Any] | None):
        self.key, self.step, self.view, self.served, self.entry = key, step, view, served, entry
        self.state: dict[str, Any] = dict(view.get("state") or {})

    @property
    def target(self) -> str | None:
        return self.state.get("target")

    def target_info(self) -> dict[str, Any]:
        info = _art(self.served, "target_info")
        return info if info.get("column") in (None, self.target) else {}


Composed = dict[str, Any]


def _lens(ctx: Context) -> Composed:
    hints = _art(ctx.served, "profile").get("lens_hints")
    return {"options": _from_teaching(ctx.entry, lambda v: {"kind": "set_lens", "lenses": [v]}),
            "multi": "lenses", "data": {"lens_hints": hints} if hints else {}}


def _simple(kind: str, field: str, convert: Callable[[str], Any] = str) -> Callable[[Context], Composed]:
    def compose(ctx: Context) -> Composed:
        return {"options": _from_teaching(ctx.entry,
                                          lambda v: {"kind": kind, field: convert(v)})}
    return compose


def _target(ctx: Context) -> Composed:
    columns = _art(ctx.served, "oriented").get("columns") or _art(ctx.served, "ingest").get("columns")
    return {"options": [_option(c["name"], {"kind": "set_target", "column": c["name"]},
                                dtype=c.get("dtype"), n_missing=c.get("n_missing"),
                                n_unique=c.get("n_unique"))
                        for c in columns or [] if isinstance(c, Mapping) and c.get("name")]}


def _event(ctx: Context) -> Composed:
    info = ctx.target_info()
    return {"options": [_option(c.get("value"), {"kind": "set_event", "column": ctx.target,
                                                 "level": str(c.get("value"))},
                                count=c.get("count"))
                        for c in info.get("classes") or [] if isinstance(c, Mapping)],
            "data": {k: info.get(k) for k in ("task", "reason") if info.get(k) is not None}}


def _task(ctx: Context) -> Composed:
    info = ctx.target_info()
    follow = ctx.step.get("followup")
    if follow == "scale":
        q = info.get("scale_question") or {}
        return {"options": exits(q.get("options")),
                "data": {k: q.get(k) for k in ("question", "evidence", "skewness", "log_skewness",
                                               "min", "median", "max", "log_column")}}
    if follow == "order":
        q = info.get("order_question") or {}
        return {"options": exits(q.get("options")) + exits(q.get("order_options")),
                "data": {k: q.get(k) for k in ("question", "levels", "proposed_order", "evidence")}}
    fits = info.get("fits")
    return {"options": _from_teaching(
                ctx.entry, lambda v: {"kind": "set_task", "column": ctx.target, "task": v},
                only=fits if fits else None),
            "data": {k: info.get(k) for k in ("detected_task", "confidence", "reason")
                     if info.get(k) is not None}}


def _follow_up(ctx: Context) -> Composed:
    info = ctx.target_info()
    candidates = [c for c in info.get("follow_up") or [] if isinstance(c, Mapping)]
    task = ctx.state.get("task") or info.get("task")
    if task == "time_to_event":
        named = [str(c["column"]) for c in candidates]
        others = [str(c) for c in info.get("follow_up_options") or [] if str(c) not in named]
        return {"options": [_option(f"time:{c}", {"kind": "set_follow_up", "column": ctx.target,
                                                  "time_column": c}, label=c)
                            for c in [*named, *others]],
                "data": {"follow_up": candidates}}
    build = {"same": {"kind": "set_censoring", "column": ctx.target},
             "varies": {"kind": "set_task", "column": ctx.target, "task": "time_to_event"}}
    return {"options": _from_teaching(ctx.entry, lambda v: build.get(v)),
            "data": {"follow_up": candidates} if candidates else {}}


def _grain(ctx: Context) -> Composed:
    grain = _art(ctx.served, "structure").get("grain") or {}
    suggested = [s.get("column") if isinstance(s, Mapping) else s
                 for s in grain.get("suggested") or []]
    fields = []
    if suggested:
        fields.append({"name": "id_column", "choices": [{"value": str(s)} for s in suggested if s],
                       "initial": None, "applies_to": ["repeated"], "_apply": ("set", "id_column")})
    return {"options": _from_teaching(ctx.entry, lambda v: {"kind": "set_grain", "grain": v}),
            "fields": fields, "data": {"grain": grain} if grain else {}}


def _repeat_kind(ctx: Context) -> Composed:
    structure = _art(ctx.served, "structure")
    time_column = structure.get("time_column")
    return {"options": _from_teaching(ctx.entry, lambda v: {
                "kind": "set_repeat_kind", "repeat_kind": v,
                "time_column": time_column if v == "time_points" else None}),
            "data": {k: structure.get(k) for k in ("repeats", "time_column", "time_columns")
                     if structure.get(k)}}


def _aggregation(ctx: Context) -> Composed:
    menu = _art(ctx.served, "structure").get("aggregation") or {}
    return {"options": _from_teaching(ctx.entry, lambda v: {"kind": "set_aggregation", "method": v},
                                      only=menu.get("options") or None),
            "data": {k: menu.get(k) for k in ("recommended", "reason", "from_pack")
                     if menu.get(k) is not None}}


def _roles(ctx: Context) -> Composed:
    art = _art(ctx.served, "roles")
    rows = [dict(c) for c in art.get("columns") or [] if isinstance(c, Mapping) and c.get("column")]
    proposed = {str(c["column"]): c.get("proposed") for c in rows}
    choices = [{"value": o["value"], "label": o.get("label")} for o in _taught(ctx.entry)]
    fields = [{"name": column, "choices": choices, "initial": role,
               "_apply": ("map", "roles", column)} for column, role in proposed.items()]
    keep = ("column", "proposed", "confidence", "reason", "unit", "linked_to", "nested_in")
    return {"options": [_option("proposed", {"kind": "set_roles", "roles": dict(proposed)})]
            if proposed else [],
            "fields": fields,
            "data": {"columns": [{k: c.get(k) for k in keep if c.get(k) is not None} for c in rows],
                     **{k: art.get(k) for k in ("needs_confirmation", "categorical", "groupings")
                        if art.get(k)}}}


def _clusters(ctx: Context) -> Composed:
    from turbotab.core.estimand import cluster_candidates

    roles = _art(ctx.served, "roles")
    named = cluster_candidates(ctx.state, roles)
    # Under inference the grouping card also asks about every column that can structurally group
    # rows, each with its guess (``estimand.grouping_candidates``).
    card = _art(ctx.served, "proposals").get("grouping") or {}
    asked = [c for c in card.get("columns") or [] if isinstance(c, Mapping) and c.get("column")]
    found = list(dict.fromkeys([*named, *(str(c["column"]) for c in asked)]))
    taught = {o["value"]: o for o in _taught(ctx.entry)}
    inference = ctx.state.get("purpose") == "inference"

    def set_(column: str | None, adjust: str | None) -> dict[str, Any]:
        return {"kind": "set_clusters", "column": column, "adjust": adjust,
                "acknowledged": column is None and bool(found), "none_of": []}

    options = []
    for c in found:
        if inference:
            for adjust in ("fixed_effects", "cluster_only"):
                t = taught.get(adjust) or {}
                options.append(_option(f"{adjust}:{c}", set_(c, adjust), label=t.get("label"),
                                       consequence=t.get("consequence"), column=c))
        else:
            options.append(_option(f"group:{c}", set_(c, None), column=c))
    t = taught.get("none") or {}
    options.append(_option("none", set_(None, None), label=t.get("label"),
                           consequence=t.get("consequence")))
    reasons = {str(p.get("column")): p.get("reason") for p in roles.get("columns") or []
               if isinstance(p, Mapping)}
    return {"options": options,
            "data": {"groupings": [{"column": c, "reason": reasons.get(c)} for c in named],
                     "asked": [{k: c.get(k) for k in ("column", "guess", "why") if c.get(k)}
                               for c in asked]}}


def _survey(ctx: Context) -> Composed:
    card = _art(ctx.served, "proposals").get("survey") or {}
    return {"options": [_option(o.get("key"), o.get("decision"), label=o.get("label"),
                                consequence=o.get("consequence"))
                        for o in card.get("options") or [] if isinstance(o, Mapping)],
            "data": {k: card.get(k) for k in ("weights", "strata", "psu", "cycle", "cycles")
                     if card.get(k)}}


# The range of plausible values of one's own as the app's card records it (``ExclusionsAsk``'s
# ``customRule``, turbotab/frontend/src/components/record/ask/ChoiceQuestions.tsx): by no level,
# the card's reason, and a row whose value is not recorded left out as not confirmed eligible (the
# server's default).
CUSTOM_RANGE = {"kind": "range", "column": None, "low": None, "high": None, "by": None,
                "reason": "implausible values (your own range)", "missing": "exclude"}


def eligibility_range() -> dict[str, Any]:
    """The range of who the study is about as the card records it (``ExclusionsAsk``'s
    ``eligibilityRule``, the same file): the range above with the reason the teaching content
    gives it, read here when the option is composed, never typed in this module (the final fix
    round's V1)."""
    from turbotab.core.teaching import content

    return {**CUSTOM_RANGE, "reason": content.ELIGIBILITY_RANGE_REASON}


# The levels of one's own as the card records them (``ExclusionsAsk``'s ``levelRule``, the same
# file; ``decisions.LevelRule``, SWARM_PLAN §5): a column, the levels it excludes and the codes
# that record a refused or a don't-know answer, each a list as typed, the card's reason, and a
# blank, refused or don't-know answer left out as not confirmed eligible (the server's default).
CUSTOM_LEVELS = {"kind": "levels", "column": None, "exclude": [], "refused": [], "dont_know": [],
                 "reason": "not in the study population (your own levels)", "missing": "exclude"}
LEVEL_LISTS = ("exclude", "refused", "dont_know")
NUMERIC = ("numeric", "integer")
ROW_ID = "__row_id"


def handlings() -> tuple[str, ...]:
    """What a rule of one's own may do with a row it cannot confirm (a blank; for a level rule also
    a refused or a don't-know answer), read off the engine's rule models (``decisions``'
    ``ExclusionRule`` and ``LevelRule``, their ``missing``): the handlings either model takes,
    each once, in the models' order."""
    from typing import get_args

    from turbotab.core.decisions import ExclusionRule, LevelRule

    return tuple(dict.fromkeys(v for model in (ExclusionRule, LevelRule)
                               for v in get_args(model.model_fields["missing"].annotation)))


def _columns(ctx: Context) -> list[Mapping[str, Any]]:
    """The table's columns as it stands (``AskFor.tsx``'s ``columns``), never the row id."""
    columns = _art(ctx.served, "oriented").get("columns") or _art(ctx.served, "ingest").get("columns")
    return [c for c in columns or [] if isinstance(c, Mapping) and c.get("name")
            and c["name"] != ROW_ID]


def _numeric_columns(ctx: Context) -> list[str]:
    """The columns a range is kept on, as the card lists them (``AskFor.tsx``'s
    ``numericColumns``): the table's numeric columns as it stands, never the outcome."""
    return [str(c["name"]) for c in _columns(ctx)
            if c.get("dtype") in NUMERIC and c["name"] != ctx.target]


def _level_columns(ctx: Context) -> list[str]:
    """The columns a level rule reads, as the card lists them (``AskFor.tsx``'s
    ``levelColumns``): every column of the table as it stands but the outcome, codes or text."""
    return [str(c["name"]) for c in _columns(ctx) if c["name"] != ctx.target]


def _exclusions(ctx: Context) -> Composed:
    """The card's options (``ExclusionsAsk``): keeping every row (``none``), each screen offered
    on this table with the rule it records, a range of one's own on a numeric column, of
    plausible values (``custom``) or of who the study is about (``eligibility``), each with its
    own reason, and levels of one's own (``levels``): the rows at the levels of one column it
    names are excluded, a code that records a refused or a don't-know answer named on its own,
    each in the teaching entry's words; the two labels the server gives each, where it gives them
    (keeping every row's under ``keep_every_row``). The ranges' fields are ``column``, ``low``
    and ``high``; the levels' are ``level_column`` and the lists ``exclude``, ``refused`` and
    ``dont_know``, each comma-separated, as the card's fields take them; every rule of one's own
    takes ``missing``, from the engine's handlings (:func:`handlings`), starting at the card's
    own. Several options are one answer, a rule each (``multi``, the module docstring)."""
    proposals = _art(ctx.served, "proposals")
    offered = [e for e in proposals.get("exclusions") or [] if isinstance(e, Mapping) and e.get("key")]
    labeled = {str(o.get("key")): o for o in _labeled((proposals.get("labels") or {}).get("exclusions"))}
    taught = {str(o.get("value")): o for o in _taught(ctx.entry)}
    none, custom = taught.get("none") or {}, taught.get("custom") or {}
    eligibility, levels = taught.get("eligibility") or {}, taught.get("levels") or {}
    options = [_option("none", {"kind": "set_exclusions", "rules": []}, label=none.get("label"),
                       consequence=none.get("consequence"),
                       labels=_two_labels(labeled.get("keep_every_row") or {}))]
    for e in offered:
        key, words = str(e["key"]), taught.get(str(e["key"])) or {}
        options.append(_option(key, {"kind": "set_exclusions", "rules": [e["rule"]]}
                               if e.get("rule") else None,
                               label=words.get("label") or e.get("label"),
                               consequence=words.get("consequence"), affected=e.get("affected"),
                               labels=_two_labels(labeled.get(key) or {}), note=e.get("refused")))
    options.append(_option("custom", None, label=custom.get("label"),
                           consequence=custom.get("consequence"),
                           _base={"kind": "set_exclusions", "rules": [dict(CUSTOM_RANGE)]},
                           _range=True))
    options.append(_option("eligibility", None, label=eligibility.get("label"),
                           consequence=eligibility.get("consequence"),
                           _base={"kind": "set_exclusions", "rules": [eligibility_range()]},
                           _range=True))
    options.append(_option("levels", None, label=levels.get("label"),
                           consequence=levels.get("consequence"),
                           _base={"kind": "set_exclusions",
                                  "rules": [json.loads(json.dumps(CUSTOM_LEVELS))]},
                           _levels=True))
    columns = _numeric_columns(ctx)
    ranges = ["custom", "eligibility"]
    fields = [{"name": "column", "choices": [{"value": c} for c in columns], "initial": None,
               "applies_to": ranges, "_apply": ("range", "column")},
              *({"name": bound, "type": "number", "initial": None, "applies_to": ranges,
                 "_apply": ("range", bound)} for bound in ("low", "high")),
              {"name": "level_column", "choices": [{"value": c} for c in _level_columns(ctx)],
               "initial": None, "applies_to": ["levels"], "_apply": ("levels", "column")},
              *({"name": name, "type": "list", "initial": None, "applies_to": ["levels"],
                 "_apply": ("levels", name)} for name in LEVEL_LISTS),
              {"name": "missing", "choices": [{"value": v} for v in handlings()],
               "initial": CUSTOM_LEVELS["missing"], "applies_to": [*ranges, "levels"],
               "_apply": ("rule", "missing")}]
    tension = ((proposals.get("labels") or {}).get("exclusions") or {}).get("tension")
    return {"options": options, "fields": fields, "multi": "rules",
            "data": {"tension": tension, "n_base": proposals.get("n_base") if offered else None}}


def _missing(ctx: Context) -> Composed:
    from turbotab.core.reference.journeys import MISSING_BODIES  # what each option records

    labeled = _labeled((_art(ctx.served, "proposals").get("labels") or {}).get("missing"))
    if labeled:
        options = [_option(o.get("key"), {"kind": "set_missing", **MISSING_BODIES[o["key"]]}
                           if o.get("key") in MISSING_BODIES else None,
                           label=o.get("label"), labels=_two_labels(o)) for o in labeled]
    else:
        options = _from_teaching(ctx.entry, lambda v: {"kind": "set_missing", **MISSING_BODIES[v]}
                                 if v in MISSING_BODIES else None)
    return {"options": options}


# The columns a cut-point is read on, as the seal's card lists them (``SealAsk``'s ``timeColumns``,
# turbotab/frontend/src/components/record/ask/SealQuestions.tsx): numbers or dates, never the
# outcome. The folds' order under it is the split's own answer, unchosen until set
# (``decisions.FoldOrder``).
TIME_DTYPES = ("numeric", "integer", "datetime", "text")
FOLD_ORDERS = ("random", "time")


def _time_columns(ctx: Context) -> list[str]:
    columns = _art(ctx.served, "oriented").get("columns") or _art(ctx.served, "ingest").get("columns")
    return [str(c["name"]) for c in columns or [] if isinstance(c, Mapping) and c.get("name")
            and c.get("dtype") in TIME_DTYPES and c["name"] not in (ROW_ID, ctx.target)]


def _split(ctx: Context) -> Composed:
    """The seal card's options (``SealAsk``): each holdout the seal plan offers, with the scheme
    as a field (``validation``), and a cut-point of one's own (``cut``, SWARM_PLAN §8): every row
    at or after a value of a time column held out, set by ``cut_column`` and ``cut_at`` (a number
    or a date, as the column reads), its folds' order, ``fold_order``, random or time, and its
    scheme, ``cut_validation``. The plan words its schemes for the folds the held-out draw orders
    (``time_ordered_folds``), and a cut-point's folds' order is its own answer, so each of the
    cut-point's schemes carries the plan's words only under the order they were made for
    (``with_fold_order``): random, where the plan's folds are not time-ordered; else none (the
    package 4 verifier's V5)."""
    plan = _art(ctx.served, "seal_plan")
    validation = [dict(v) for v in (plan.get("validation") or {}).get("options") or []
                  if isinstance(v, Mapping)]
    first = validation[0] if validation else {"validation": "kfold"}
    labels = {str(o.get("key")): o for o in _labeled(plan.get("labels"))}

    def body(holdout: Any, scheme: Mapping[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {"kind": "set_split", "holdout": float(holdout),
                               "validation": scheme.get("validation") or "kfold"}
        if scheme.get("cluster"):
            out["cluster"] = scheme["cluster"]
        return out

    options = [_option(_num(o["holdout"]), body(o["holdout"], first), label=o.get("label"),
                       consequence=o.get("measures"), n_holdout=o.get("n_holdout"),
                       below_floor=o.get("below_floor"),
                       labels=_two_labels(labels.get(_num(o["holdout"])) or {}))
               for o in plan.get("options") or [] if isinstance(o, Mapping) and "holdout" in o]
    fields: list[dict[str, Any]] = []
    words = ("label", "measures", "cost", "caution")
    schemes = {v.get("validation"): v for v in validation}
    if validation:
        fields.append({"name": "validation",
                       "choices": [{k: v.get(k) for k in ("validation", *words, "cluster")
                                    if v.get(k) is not None}
                                   for v in validation],
                       "initial": first.get("validation"),
                       "applies_to": [o["id"] for o in options],
                       "_apply": ("validation", schemes)})
    if plan:
        cut = {str(o.get("value")): o for o in _taught(ctx.entry)}.get("cut") or {}
        options.append(_option("cut", None, label=cut.get("label"),
                               consequence=cut.get("consequence"),
                               _base=body(0.0, first)))
        made_for = "time" if plan.get("time_ordered_folds") else "random"
        fields += [
            {"name": "cut_column", "choices": [{"value": c} for c in _time_columns(ctx)],
             "initial": None, "applies_to": ["cut"], "_apply": ("map", "cut_point", "column")},
            {"name": "cut_at", "initial": None, "applies_to": ["cut"],
             "_apply": ("map", "cut_point", "at")},
            {"name": "fold_order", "choices": [{"value": v} for v in FOLD_ORDERS], "initial": None,
             "applies_to": ["cut"], "_apply": ("set", "fold_order")}]
        if validation:
            fields.append({
                "name": "cut_validation",
                "choices": [{"validation": v.get("validation"),
                             **({"cluster": v["cluster"]} if v.get("cluster") else {}),
                             **({"with_fold_order": {"random": {k: v[k] for k in words
                                                                if v.get(k) is not None}}}
                                if made_for == "random" else {})}
                            for v in validation],
                "initial": first.get("validation"), "applies_to": ["cut"],
                "_apply": ("validation", schemes)})
    keep = ("reason", "precision_note", "refusal", "exploratory", "n_measured", "n_analyzed")
    data = {k: plan.get(k) for k in keep if plan.get(k) not in (None, "")}
    if (plan.get("floor") or {}).get("text"):
        data["floor"] = plan["floor"]["text"]
    if (plan.get("validation") or {}).get("reason"):
        data["validation_reason"] = plan["validation"]["reason"]
    return {"options": options, "fields": fields, "data": data}


def _estimand(ctx: Context) -> Composed:
    card = _art(ctx.served, "proposals").get("estimand") or {}
    if not card:
        return {"options": []}
    fitted = sorted((m for m in card.get("measures") or [] if m.get("fitted")),
                    key=lambda m: m.get("rank") if m.get("rank") is not None else 99)
    effects = [e for e in card.get("effects") or [] if isinstance(e, Mapping)]
    effect = (effects[0].get("effect") if effects else None) or "total"
    family = card.get("family") or None
    family_fitted = [m for m in (family or {}).get("measures") or [] if m.get("fitted")]
    energy = [e["column"] for e in card.get("exposures") or [] if e.get("energy_contrast")]
    options = []
    for e in card.get("exposures") or []:
        decision = {"kind": "set_estimand", "exposure": e["column"], "family": False,
                    "effect": effect, "contrast": None,
                    "measure": fitted[0]["measure"] if fitted else None}
        options.append(_option(e["column"], decision, energy_contrast=e.get("energy_contrast")))
    if family:
        options.append(_option("__family", {
            "kind": "set_estimand", "exposure": None, "family": True, "effect": effect,
            "contrast": None, "measure": family_fitted[0]["measure"] if family_fitted else None,
            "multiplicity": None}, consequence=family.get("consequence"), n=family.get("n")))
    fields = [{"name": "effect", "choices": [{"value": c.get("effect") or "total",
                                              "label": c.get("label"),
                                              "consequence": c.get("consequence")} for c in effects],
               "initial": effect, "_apply": ("set", "effect")}]
    if card.get("contrasts") and (energy or (family or {}).get("energy_contrast")):
        fields.append({"name": "contrast",
                       "choices": [{"value": c.get("contrast"), "label": c.get("label"),
                                    "consequence": c.get("consequence")}
                                   for c in card.get("contrasts") or []],
                       "initial": None, "applies_to": [*energy, *(["__family"] if (family or {})
                                                                  .get("energy_contrast") else [])],
                       "_apply": ("set", "contrast")})
    if fitted:
        fields.append({"name": "measure", "choices": [{"value": m["measure"], "label": m.get("label"),
                                                       "reason": m.get("reason")} for m in fitted],
                       "initial": fitted[0]["measure"],
                       "applies_to": [e["column"] for e in card.get("exposures") or []],
                       "_apply": ("set", "measure")})
    if family and family_fitted:
        fields.append({"name": "family_measure",
                       "choices": [{"value": m["measure"], "label": m.get("label")}
                                   for m in family_fitted],
                       "initial": family_fitted[0]["measure"], "applies_to": ["__family"],
                       "_apply": ("set", "measure")})
    multiplicity = (family or {}).get("multiplicity")
    if multiplicity:
        fields.append({"name": "multiplicity",
                       "choices": [{"value": o.get("key"), "label": o.get("label"),
                                    "labels": _two_labels(o)} for o in multiplicity.get("options") or []],
                       "initial": None, "applies_to": ["__family"], "_apply": ("set", "multiplicity")})
    named = [{"measure": m.get("measure"), "label": m.get("label"), "reason": m.get("reason")}
             for m in card.get("measures") or [] if not m.get("fitted")]
    return {"options": options, "fields": fields,
            "data": {"named_not_fitted": named} if named else {}}


ADJUSTMENT_ASKED = ("causes_exposure", "causes_outcome", "after_exposure")
ANSWER3 = ("yes", "no", "unknown")


def _adjustment(ctx: Context) -> Composed:
    card = _art(ctx.served, "proposals").get("adjustment") or {}
    if not card:
        return {"options": []}
    options, columns = [], []
    for g in card.get("groups") or []:
        if not isinstance(g, Mapping):
            continue
        columns += [str(c) for c in g.get("columns") or []]
        if g.get("decision"):
            options.append(_option(g.get("key"), g["decision"], label=g.get("label"),
                                   consequence=g.get("reason"), columns=g.get("columns"),
                                   guess=g.get("guess"), derived_words=g.get("derived_words"),
                                   note=g.get("estimand_note")))
    options.append(_option("answers", None, columns=columns))
    # Each column's answers, one per question the card asks, in this order (``card.questions``).
    fields = [{"name": c, "parts": list(ADJUSTMENT_ASKED), "each": list(ANSWER3),
               "initial": None, "applies_to": ["answers"],
               "_apply": ("answers", card.get("exposure"), c)} for c in columns]
    keep = ("exposure", "effect", "questions", "answered", "adjusted", "left_out", "secondary",
            "estimand_note")
    return {"options": options, "fields": fields,
            "data": {k: card.get(k) for k in keep if card.get(k) not in (None, [], {})}}


def _time_varying(ctx: Context) -> Composed:
    art = _art(ctx.served, "time_varying")
    exposure = art.get("exposure") or (ctx.state.get("estimand") or {}).get("exposure")
    options = [_option(o.get("key"), {"kind": "set_time_varying", "exposure": exposure,
                                      "method": o.get("key"), "ordering": None},
                       label=o.get("label"),
                       labels={"customary": o.get("customary"), "sound": o.get("sound")})
               for o in art.get("options") or [] if isinstance(o, Mapping)]
    from typing import get_args

    from turbotab.core.decisions import TimeOrdering

    fields = [{"name": "ordering", "choices": [{"value": v} for v in get_args(TimeOrdering)],
               "initial": None, "_apply": ("set", "ordering")}]
    return {"options": options, "fields": fields,
            "data": {k: art.get(k) for k in ("reason", "affected", "relations", "withheld")
                     if art.get(k)}}


def _energy_adjustment(ctx: Context) -> Composed:
    proposals = _art(ctx.served, "proposals")
    reading = proposals.get("energy") or {}
    applicability = reading.get("applicability") or {}
    labeled = _labeled((proposals.get("labels") or {}).get("energy_adjustment"))

    def body(method: str) -> dict[str, Any]:
        out: dict[str, Any] = {"kind": "set_energy_adjustment", "method": method}
        if method != "none":
            out.update({"energy_column": reading.get("energy_column"),
                        "nutrients": list(reading.get("nutrients") or [])})
        return out

    source = labeled or [{"key": o["value"], "label": o.get("label")} for o in _taught(ctx.entry)]
    options = []
    for o in source:
        fit = applicability.get(o.get("key")) or {}
        options.append(_option(o.get("key"), body(str(o.get("key"))), label=o.get("label"),
                               labels=_two_labels(o), applicable=fit.get("ok"),
                               note=fit.get("reason")))
    return {"options": options,
            "data": {k: reading.get(k) for k in ("energy_column", "nutrients")
                     if reading.get(k)}}


SPLINES = ("spline", "zero_spline")


def _form(ctx: Context) -> Composed:
    card = _art(ctx.served, "forms")
    needs = [dict(n) for n in card.get("needs") or [] if isinstance(n, Mapping)]
    fields = []
    for n in needs:
        fields.append({"name": n["column"],
                       "choices": [{"value": o.get("value"), "label": o.get("label"),
                                    "consequence": o.get("consequence"), "rung": o.get("rung")}
                                   for o in n.get("options") or []],
                       "initial": (n.get("proposal") or {}).get("form"),
                       "_apply": ("form", n["column"], n.get("proposal") or {}, n.get("rule_knots"))})
    keep = ("column", "role", "unit", "scale", "rule", "mass_at_zero", "label")
    return {"options": [_option("proposed", card["answer"])] if card.get("answer") else [],
            "fields": fields,
            "data": {"needs": [{k: n.get(k) for k in keep if n.get(k) not in (None, "")}
                               for n in needs],
                     **{k: card.get(k) for k in ("stated", "waiting", "beside") if card.get(k)}}}


def _causal(ctx: Context) -> Composed:
    exposure = (ctx.state.get("estimand") or {}).get("exposure")
    return {"options": _from_teaching(ctx.entry, lambda v: {"kind": "set_causal",
                                                            "exposure": exposure, "method": v})}


def _models(ctx: Context) -> Composed:
    shelf = _art(ctx.served, "shelf")
    options = [_option(f["key"], {"kind": "select_models", "models": [f["key"]]},
                       label=f.get("label"), rank=f.get("rank"), fit=f.get("fit"),
                       concerns=f.get("concerns") or None, inductive_bias=f.get("inductive_bias"),
                       estimate=f.get("estimate"))
               for f in sorted((f for f in shelf.get("families") or [] if isinstance(f, Mapping)),
                               key=lambda f: f.get("rank", 0))]
    data = {k: shelf.get(k) for k in ("basis",) if shelf.get(k)}
    if (shelf.get("sample_size") or {}).get("sentence"):
        data["sample_size"] = shelf["sample_size"]["sentence"]
    return {"options": options, "multi": "models", "data": data}


def _substitution(ctx: Context) -> Composed:
    pairs = [p for p in _art(ctx.served, "design").get("substitution_pairs") or []
             if isinstance(p, Mapping)]
    steps = [{"value": o["value"], "label": o.get("label"), "consequence": o.get("consequence")}
             for o in _taught(ctx.entry)]
    return {"options": [_option(f"{p['donor']}:{p['recipient']}",
                                {"kind": "set_substitution", "donor": p["donor"],
                                 "recipient": p["recipient"]}) for p in pairs],
            "fields": [{"name": "step_kcal", "choices": steps, "initial": None,
                        "_apply": ("number", "step_kcal")}] if steps else []}


def _open_seal(ctx: Context) -> Composed:
    return {"options": [_option(f, {"kind": "open_seal", "family": f})
                        for f in ctx.state.get("models") or []],
            "seal": True}


COMPOSERS: dict[str, Callable[[Context], Composed]] = {
    "lens": _lens,
    "orientation": _simple("set_orientation", "orientation"),
    "target": _target,
    "event": _event,
    "task": _task,
    "follow_up": _follow_up,
    "design": _simple("set_design", "design"),
    "purpose": _simple("set_purpose", "purpose"),
    "grain": _grain,
    "repeat_kind": _repeat_kind,
    "unit": _simple("set_unit", "unit"),
    "aggregation": _aggregation,
    "temporal": _simple("set_temporal", "temporal", lambda v: v == "true"),
    "roles": _roles,
    "clusters": _clusters,
    "survey": _survey,
    "exclusions": _exclusions,
    "missing": _missing,
    "split": _split,
    "estimand": _estimand,
    "adjustment": _adjustment,
    "time_varying": _time_varying,
    "energy_adjustment": _energy_adjustment,
    "form": _form,
    "causal": _causal,
    "models": _models,
    "substitution": _substitution,
    "open_seal": _open_seal,
}


def compose(key: str, step: Mapping[str, Any], view: Mapping[str, Any], served: Served,
            entry: Mapping[str, Any] | None) -> Composed:
    """The options, fields and served data of the open question ``key``. A question this module
    composes no options for (``modification``) lists its teaching entry's options with no
    decision: the analyst answers it through the ways forward the server offers, if any."""
    composer = COMPOSERS.get(key)
    if composer is None:
        out: Composed = {"options": _from_teaching(entry, lambda v: None)}
    else:
        out = composer(Context(key, step, view, served, entry))
    out.setdefault("fields", [])
    out.setdefault("data", {})
    out["data"] = {k: v for k, v in out["data"].items() if v not in (None, [], {}, "")}
    return out


# ── an answer from ids and fields ────────────────────────────────────────────


class Unanswerable(ValueError):
    """An id or a field the open question does not offer, in words the analyst reads."""


def _set_value(field: Mapping[str, Any], raw: str) -> Any:
    how = field["_apply"]
    if how == ("range", "column"):
        allowed = [str(c.get("value")) for c in field.get("choices") or []]
        if raw not in allowed:
            raise Unanswerable(f"column is one of the numeric columns, {', '.join(allowed) or 'none here'}"
                               f"; not {raw!r}.")
        return raw
    if how[0] == "range":
        try:
            value = float(raw)
        except ValueError:
            value = math.nan
        if not math.isfinite(value):
            raise Unanswerable(f"{field['name']} takes a finite number, such as 18 or 2.5; not {raw!r}.")
        return value
    if how == ("levels", "column"):
        allowed = [str(c.get("value")) for c in field.get("choices") or []]
        if raw not in allowed:
            raise Unanswerable(f"{field['name']} is one of the columns but the outcome, "
                               f"{', '.join(allowed) or 'none here'}; not {raw!r}.")
        return raw
    if how[0] == "levels":  # as the card's ``parseLevels``: by commas, each trimmed, blanks dropped
        return [p.strip() for p in raw.split(",") if p.strip()]
    if how[0] == "answers":
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) != 3 or any(p not in ANSWER3 for p in parts):
            raise Unanswerable(f"{field['name']} takes three answers, each yes, no or unknown, "
                               f"in this order: {', '.join(ADJUSTMENT_ASKED)} (for example "
                               f"yes,yes,no); not {raw!r}.")
        return dict(zip(ADJUSTMENT_ASKED, parts))
    if how[0] == "number":
        allowed = [str(c.get("value")) for c in field.get("choices") or []]
        if raw not in allowed:
            raise Unanswerable(f"{field['name']} is one of {', '.join(allowed)}, not {raw!r}.")
        return float(raw)
    value_key = "validation" if how[0] == "validation" else "value"
    allowed = [str(c.get(value_key)) for c in field.get("choices") or []]
    if allowed and raw not in allowed:
        raise Unanswerable(f"{field['name']} is one of {', '.join(allowed)}, not {raw!r}.")
    return raw


def _apply(decision: dict[str, Any], field: Mapping[str, Any], value: Any, place: int = 0) -> None:
    """``value`` set into ``decision`` as ``field`` says; a rule's own field into the rule at
    ``place`` among the options named (0 for the first)."""
    how = field["_apply"]
    if how[0] in ("set", "number"):
        decision[how[1]] = value
    elif how[0] == "map":
        decision.setdefault(how[1], {})[how[2]] = value
    elif how[0] == "answers":
        decision.update({"kind": "set_adjustment", "exposure": how[1]})
        decision.setdefault("answers", {})[how[2]] = value
    elif how[0] == "validation":
        scheme = how[1].get(value) or {}
        decision["validation"] = value
        decision.pop("cluster", None)
        if scheme.get("cluster"):
            decision["cluster"] = scheme["cluster"]
    elif how[0] in ("range", "levels", "rule"):
        decision["rules"][place][how[1]] = value
    elif how[0] == "form":
        _, column, proposal, knots = how
        if value == proposal.get("form"):
            spec = dict(proposal)
        else:
            spec = {"form": value}
            if value in SPLINES and knots:
                spec["knots"] = int(knots)
        decision.setdefault("forms", {})[column] = spec


def _range_set(named: str, at: str, rule: Mapping[str, Any]) -> None:
    """A range of one's own records only as the card records it: a column and a bound (the card's
    ``customRule`` is null otherwise). ``named`` names the option, ``at`` its place before a
    field's name (``"2."``; empty where one option is named)."""
    if rule["column"] is None:
        raise Unanswerable(f"{named} keeps the rows inside a range on one numeric column: set "
                           f"{at}column, and {at}low, {at}high or both.")
    if rule["low"] is None and rule["high"] is None:
        raise Unanswerable(f"{named} takes {at}low, {at}high or both, as numbers: neither is set.")


def _levels_set(named: str, at: str, rule: Mapping[str, Any]) -> None:
    """Levels of one's own record only as the card records them: a column and a level excluded
    (the card's ``levelRule`` is null otherwise); ``named`` and ``at`` as :func:`_range_set`'s."""
    if rule["column"] is None:
        raise Unanswerable(f"{named} excludes rows at levels of one column: set {at}level_column, "
                           f"and {at}exclude, its levels by commas.")
    if not rule["exclude"]:
        raise Unanswerable(f"{named} takes {at}exclude, the levels to exclude by commas, such "
                           f"as 1 or 1,3: none is set.")


def _engine_takes(named: str, rule: Mapping[str, Any]) -> None:
    """A rule of one's own as the engine's own rule models read it (``decisions.RULE_ADAPTER``):
    each level named once, in the one spelling the server compares levels by, and a handling of a
    row it cannot confirm that its model takes; the model's refusal is said."""
    from pydantic import ValidationError

    from turbotab.core.decisions import RULE_ADAPTER

    try:
        RULE_ADAPTER.validate_python(rule)
    except ValidationError as exc:
        why = "; ".join(str(e.get("msg", "")).removeprefix("Value error, ") for e in exc.errors())
        raise Unanswerable(f"{named} cannot record this rule: {why}.") from None


def _place(name: str, applies: list[str], ids: list[str], place: int | None) -> int:
    """The option named that a setting of ``name`` goes to (0 for the first): the place written
    before it (``--set 2.name=...``), else the one option named that takes it."""
    if place is None:
        takers = [i for i, oid in enumerate(ids) if oid in applies]
        if len(ids) == 1 or not takers:
            place = 0
        elif len(takers) == 1:
            return takers[0]
        else:
            raise Unanswerable(f"{len(takers)} of the options named take {name}: write its place "
                               f"among them before it, "
                               f"{' or '.join(f'{i + 1}.{name}' for i in takers)} (--set "
                               f"{takers[-1] + 1}.{name}=... sets option {takers[-1] + 1}'s).")
    if not 0 <= place < len(ids):
        raise Unanswerable(f"--set {place + 1}.{name} names option {place + 1}, and {len(ids)} "
                           f"{'is' if len(ids) == 1 else 'are'} named.")
    if ids[place] not in applies:
        raise Unanswerable(f"The field {name} does not apply to option {ids[place]}.")
    return place


def resolve(composed: Composed, ids: list[str], sets: list[str],
            ways: Iterable[Mapping[str, Any]] = ()) -> dict[str, Any]:
    """The decision ``ids`` and ``sets`` name among what the open question offers (``composed``)
    and the ways forward on hand (``ways``: the ask card's, the quest lines', the last refusal's).
    Several ids, where the question takes them (``multi``), record what each records, in their
    order; a setting goes to the option named that its place names, else to the one that takes it
    (the module docstring). Raises :class:`Unanswerable` for an id or a field not offered."""
    if not ids:
        raise Unanswerable("Name an option by its id, as ask lists it.")
    options = {str(o["id"]): o for o in composed.get("options") or [] if o.get("id") is not None}
    for way in ways:
        if way.get("id") is not None:
            options.setdefault(str(way["id"]), dict(way))
    missing = [i for i in ids if i not in options]
    if missing:
        raise Unanswerable(f"No option {', '.join(missing)} is offered now; ask lists the open "
                           f"question's options and its ways forward.")
    chosen = [options[i] for i in ids]
    multi = composed.get("multi")
    if len(chosen) > 1 and not multi:
        raise Unanswerable("This question takes one option at a time.")
    own_ids = {str(o["id"]) for o in composed.get("options") or [] if o.get("id") is not None}
    ways_named = list(dict.fromkeys(i for i in ids if i not in own_ids))
    if len(chosen) > 1 and ways_named:
        raise Unanswerable(
            f"Option {ways_named[0]} is a way forward: it is answered alone, and only this "
            f"question's own options are named together." if len(ways_named) == 1 else
            f"Options {', '.join(ways_named[:-1])} and {ways_named[-1]} are ways forward: each is "
            f"answered alone, and only this question's own options are named together.")
    blank = next((i for i, o in zip(ids, chosen) if not isinstance(o.get("decision"), Mapping)),
                 None)
    if blank is not None and not sets:
        raise Unanswerable(f"Option {blank} records nothing by itself: "
                           f"{'set its fields with --set' if composed.get('fields') else 'it says what to do'}.")

    def own(option: Mapping[str, Any]) -> dict[str, Any]:
        return json.loads(json.dumps(option.get("decision") or option.get("_base") or {}))

    decision = own(chosen[0])
    offsets = [0]  # where each option's own items start in ``decision[multi]``
    if multi and len(chosen) > 1:
        alone = next((i for i, o in zip(ids, chosen) if not own(o).get(multi)), None)
        if alone is not None:
            raise Unanswerable(f"Option {alone} records no {multi}: it is answered alone.")
        items = [own(o)[multi] for o in chosen]
        offsets = [sum(len(x) for x in items[:i]) for i in range(len(items))]
        decision[multi] = [v for x in items for v in x]
    fields = {str(f["name"]): f for f in composed.get("fields") or []}
    for item in sets:
        name, eq, raw = item.partition("=")
        if not eq:
            raise Unanswerable(f"--set takes name=value, not {item!r}.")
        name, place = name.strip(), None
        field = fields.get(name)
        if field is None:  # a setting by its option's place: ``2.level_column``
            head, dot, rest = name.partition(".")
            if dot and head.isdigit() and (fields.get(rest) or {}).get("applies_to"):
                field, place, name = fields[rest], int(head) - 1, rest
        if field is None:
            raise Unanswerable(f"No field {name!r} is offered now; ask lists the fields.")
        applies = field.get("applies_to")
        if applies:
            place = _place(name, applies, ids, place)
        _apply(decision, field, _set_value(field, raw.strip()), offsets[place or 0])
    for place, (oid, option) in enumerate(zip(ids, chosen)):
        if not (option.get("_range") or option.get("_levels")):
            continue
        named, at = (f"Option {oid}", "") if len(ids) == 1 else (f"Option {place + 1} ({oid})",
                                                                 f"{place + 1}.")
        rule = decision["rules"][offsets[place]]
        (_range_set if option.get("_range") else _levels_set)(named, at, rule)
        _engine_takes(named, rule)
    if not decision.get("kind"):
        raise Unanswerable(f"Option {ids[0]} records nothing as it stands.")
    return decision

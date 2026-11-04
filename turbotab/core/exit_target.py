"""The card a way forward opens (G12; FOUNDATION §9 item 1: each item's exit opens the stage that
owns it).

A way forward that names no decision (``{label, decision: None}``) is said in its label ("Answer
the missing-values question first"); its ``target`` names where that is done: the quest stage and
the quest line whose card it opens, in the keys the quest log serves (``quest.QuestLine.key``,
``quest.QuestStage.key``), so a client opens that card on a press. A way forward with no target is
one that opens nothing: keeping the answers as they are, or waiting for work under way.

Served as ``{"stage": <stage key>, "key": <line key> | None}``; a key of None opens the stage on
its open line. A site names it with :func:`toward`; the refusals' normalizers
(``decisions.Refusal``, ``graph.StageRefusal``) read it with :func:`exit_target`, which also takes a
name as :func:`toward` does, and refuse a target that names no stage or line the quest log serves,
as they refuse a decision the decision model does not read.
"""

from __future__ import annotations

from typing import Any, Mapping

TARGET = "target"


def toward(name: str | None = None, *, stage: str | None = None,
           key: str | None = None) -> dict[str, str | None]:
    """The target of a way forward, as the quest log keys it.

    * ``toward("missing")``: a Router question (``quest.QUESTIONS``), its line in its stage;
    * ``toward("set_scales")``: a decision kind, the line it answers (a declaration's own line,
      or the Router question's whose answer it records or that asks it, ``quest.ALSO_ANSWERS``),
      else its stage;
    * ``toward(stage="results")``: a stage, on its open line;
    * ``toward(stage="first_look", key=finding_id)``: one line of a stage, by the key the quest log
      serves it under (a noticing's, a column's role line).

    A stage key names its stage only where ``name`` is no question (the Models stage is
    ``toward(stage="models")``; ``toward("models")`` is the models question). Raises ``ValueError``
    for a name or stage the quest log does not serve."""
    from turbotab.core.quest import ALSO_ANSWERS, DECLARATIONS, QUESTIONS, STAGE_NAMES, kind_place

    if name is None:
        if stage not in STAGE_NAMES:
            raise ValueError(f"a way forward's target names no stage the quest log serves: {stage!r}")
        return {"stage": stage, "key": None if key is None else str(key)}
    if stage is not None or key is not None:
        raise ValueError("a way forward's target is a name, or a stage with its line, not both")
    if name in QUESTIONS:
        return {"stage": QUESTIONS[name].stage, "key": name}
    if name in STAGE_NAMES:
        return {"stage": name, "key": None}
    try:
        place = kind_place(name)
    except KeyError:
        raise ValueError(f"a way forward's target names no question, decision kind or stage the "
                         f"quest log serves: {name!r}") from None
    if any(d.kind == name for d in DECLARATIONS):
        return {"stage": place.stage, "key": name}
    from turbotab.core.sequence import question_of

    # A kind its question asks beside the one its slot names (the task question's scale and order).
    also = [q for q, kinds in ALSO_ANSWERS.items() if name in kinds]
    question = also[0] if also else question_of(name)
    if question is not None and question in QUESTIONS:
        return {"stage": QUESTIONS[question].stage, "key": question}
    return {"stage": place.stage, "key": None}


def exit_target(value: Any) -> dict[str, str | None] | None:
    """A way forward's ``target`` as it is served: None where it names none; a name read as
    :func:`toward` reads it; a mapping with its ``stage`` (and ``key``) checked as served. Raises
    ``ValueError`` for anything else, or a stage the quest log does not serve."""
    if value is None:
        return None
    if isinstance(value, str):
        return toward(value)
    if isinstance(value, Mapping) and set(value) <= {"stage", "key"}:
        key = value.get("key")
        if key is not None and not isinstance(key, str):
            raise ValueError(f"a way forward's target names its line by a string key: {key!r}")
        return toward(stage=value.get("stage"), key=key)
    raise ValueError(f"a way forward's target is a name or a stage with its line: {value!r}")


def with_target(item: Mapping[str, Any], out: dict[str, Any]) -> dict[str, Any]:
    """``out`` (a normalized way forward) with ``item``'s target, as served, where it names one;
    a way forward that names none is served as before, with no ``target``."""
    target = exit_target(item.get(TARGET))
    if target is not None:
        out[TARGET] = target
    return out


__all__ = ["TARGET", "exit_target", "toward", "with_target"]

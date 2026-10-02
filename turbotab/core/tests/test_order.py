"""Tier A: answers in the Router's order, and the seal's grain (M2_CONTRACT §12.2).

An answer to a question still waiting behind an earlier unanswered one is refused (409 ``not_yet``,
with "Answer <question> first"); changing an answered question is always allowed, whatever is
unanswered before it; the seal is never drawn without a grain, answered or stated.
"""
from __future__ import annotations

from itertools import permutations

import pytest

from turbotab.core import decisions as d
from turbotab.core import sequence
from turbotab.core.decisions import ProjectState, Refusal
from turbotab.core.interview import QUESTION_KEYS, first_unanswered, route

ALL_FRESH = {s: {"status": "fresh"} for s in (
    "ingest", "oriented", "profile", "findings", "structure", "working", "target_info", "roles",
    "proposals", "cohort", "split", "shelf", "design", "fit", "substitution", "seal_plan")}
DIET_ROLES = {"participant_id": "identifier", "age": "covariate", "energy_kcal": "energy",
              "protein_g": "exposure", "fat_g": "exposure", "carbohydrate_g": "exposure"}

# Every question answered: dietary recalls kept as rows, nothing combined.
FULL = dict(
    lens=["dietary"], target="hba1c", task="regression", purpose="prediction",
    grain={"grain": "repeated", "id_column": "participant_id"}, repeat_kind={"repeat_kind": "repeats"},
    unit="row", roles=DIET_ROLES, exclusions=[], missing="complete_case", split={"holdout": 0.2},
    energy_adjustment={"method": "none"}, models=["linear"],
    substitution={"donor": "fat_g", "recipient": "protein_g"}, seal_opened=True,
)
SLOT = {"open_seal": "seal_opened"}

# One decision per question (only its kind is read by the order check).
ANSWER = {
    "lens": d.SetLens(lenses=["dietary"]),
    "orientation": d.SetOrientation(orientation="sample_major"),
    "target": d.SetTarget(column="hba1c"),
    "event": d.SetEvent(column="hba1c", level="1"),
    "task": d.SetTask(column="hba1c", task="regression"),
    "purpose": d.SetPurpose(purpose="prediction"),
    "grain": d.SetGrain(grain="repeated", id_column="participant_id"),
    "repeat_kind": d.SetRepeatKind(repeat_kind="repeats"),
    "unit": d.SetUnit(unit="row"),
    "aggregation": d.SetAggregation(method="mean"),
    "temporal": d.SetTemporal(temporal=False),
    "roles": d.SetRoles(roles=DIET_ROLES),
    "exclusions": d.SetExclusions(rules=[]),
    "missing": d.SetMissing(strategy="complete_case"),
    "split": d.SetSplit(holdout=0.2),
    "energy_adjustment": d.SetEnergyAdjustment(method="none"),
    "models": d.SelectModels(models=["linear"]),
    "substitution": d.SetSubstitution(donor="fat_g", recipient="protein_g"),
    "open_seal": d.OpenSeal(),
}
ANSWERED_IN_FULL = [k for k in QUESTION_KEYS if FULL.get(SLOT.get(k, k)) is not None]


def ctx_for(state: ProjectState) -> dict:
    return {"state": state, "interview": lambda: route(state, ALL_FRESH)}


def without(*questions: str) -> ProjectState:
    return ProjectState(**{k: v for k, v in FULL.items() if k not in {SLOT.get(q, q) for q in questions}})


def test_every_question_has_an_answer_kind_and_every_answer_kind_its_question():
    assert set(ANSWER) == set(QUESTION_KEYS)
    for key, decision in ANSWER.items():
        assert sequence.question_of(decision.kind) == key
    for kind in ("revert", "apply_repair", "defer_finding", "dismiss_finding"):
        assert sequence.question_of(kind) is None  # not a question: never held back by the order


@pytest.mark.parametrize("unanswered", ANSWERED_IN_FULL)
def test_changing_an_answered_question_is_never_refused_as_out_of_order(unanswered):
    """Whatever earlier question is unanswered, every answered question can still be changed."""
    state = without(unanswered)
    steps = {s.key: s for s in route(state, ALL_FRESH)}
    for key in ANSWERED_IN_FULL:
        if key == unanswered or steps[key].status != "answered":
            continue
        sequence._answers_in_order(ANSWER[key], ctx_for(state))  # never raises


@pytest.mark.parametrize("earlier,later", [
    (a, b) for a, b in permutations(ANSWERED_IN_FULL, 2)
    if QUESTION_KEYS.index(a) < QUESTION_KEYS.index(b)
])
def test_a_question_still_waiting_behind_an_unanswered_one_is_refused_with_the_way_forward(earlier, later):
    state = without(earlier, later)
    steps = route(state, ALL_FRESH)
    step = next(s for s in steps if s.key == later)
    first = first_unanswered(steps)
    held = step.status in ("open", "waiting") and first is not None and first.key != later
    if not held:
        sequence._answers_in_order(ANSWER[later], ctx_for(state))
        return
    with pytest.raises(Refusal) as refused:
        sequence._answers_in_order(ANSWER[later], ctx_for(state))
    assert refused.value.code == "not_yet"
    [exit_] = refused.value.exits
    from turbotab.core.voice import question_name

    assert exit_["label"] == f"Answer {question_name(first.key)} first"
    assert exit_["decision"] is None


def test_the_open_question_and_a_stated_skip_are_never_held_back():
    state = ProjectState(lens=["dietary"])
    sequence._answers_in_order(ANSWER["target"], ctx_for(state))  # the open question
    with pytest.raises(Refusal):
        sequence._answers_in_order(ANSWER["purpose"], ctx_for(state))
    # "Ask me anyway" on a stated skip: the task skipped while an earlier event is still open
    info = {"column": "y", "task": "binary", "confidence": "high", "reason": "two levels",
            "classes": [{"value": "0"}, {"value": "1"}]}
    binary = ProjectState(lens=["dietary"], target="y")
    steps = {s.key: s for s in route(binary, ALL_FRESH, {"target_info": info})}
    assert steps["event"].status == "open" and steps["task"].status == "skipped"
    ctx = {"state": binary, "interview": lambda: route(binary, ALL_FRESH, {"target_info": info})}
    sequence._answers_in_order(d.SetTask(column="y", task="binary"), ctx)


def test_the_order_check_runs_inside_validate_and_outranks_other_checks():
    state = ProjectState(lens=["dietary"])
    with pytest.raises(Refusal) as refused:
        d.validate({"kind": "set_unit", "unit": "row"}, ctx_for(state))
    assert refused.value.code == "not_yet"  # not the unit's own "say whether units repeat first"
    # without the Router (a caller that has no project) nothing is checked against it
    d.validate({"kind": "set_purpose", "purpose": "prediction"}, {"state": state})


# ── the seal requires a grain ────────────────────────────────────────────────

STATED = {"grain": {"stated": {"column": "SEQN", "n_rows": 10, "sentence": "every `SEQN` appears once."}}}


def seal_ctx(state: ProjectState, structure: dict | None) -> dict:
    return {"state": state, "artifact": lambda stage: structure if stage == "structure" else None}


def test_the_seal_requires_a_grain_answered_or_stated():
    split = d.SetSplit(holdout=0.2)
    bare = ProjectState(**{k: v for k, v in FULL.items() if k != "grain"})
    with pytest.raises(Refusal) as refused:
        sequence._seal_needs_grain(split, seal_ctx(bare, {"grain": {"stated": None}}))
    assert refused.value.code == "not_yet"
    assert refused.value.exits[0]["label"] == "Answer the question of whether people repeat first"
    with pytest.raises(Refusal):  # the structure reading not ready: no grain can be stated yet
        sequence._seal_needs_grain(split, seal_ctx(bare, None))
    sequence._seal_needs_grain(split, seal_ctx(bare, STATED))  # stated from a unique identifier
    for answer in ("repeated", "one_row_per_unit", "unknown"):
        state = bare.model_copy(update={"grain": d.GrainSpec(grain=answer, id_column="participant_id")})
        sequence._seal_needs_grain(split, seal_ctx(state, None))
    # it holds a changed split too: a seal drawn before any grain answer cannot be drawn again
    legacy = bare.model_copy(update={"split": d.SplitSpec(holdout=0.3)})
    with pytest.raises(Refusal):
        sequence._seal_needs_grain(split, seal_ctx(legacy, {"grain": {"stated": None}}))

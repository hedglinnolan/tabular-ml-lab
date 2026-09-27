"""Decisions are an append-only log and state is its fold (BLUEPRINT §3). Tier A."""
from __future__ import annotations

import json
import multiprocessing as mp
import threading
from datetime import timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from turbotab.core import decisions as d
from turbotab.core.decisions import (
    DecisionLog,
    ProjectState,
    Refusal,
    Revert,
    SetLens,
    SetPurpose,
    SetTarget,
    SetTask,
    fold,
    parse_decision,
    validate,
)


@pytest.fixture
def log(tmp_path: Path) -> DecisionLog:
    return DecisionLog(tmp_path / "decisions.jsonl")


def test_later_writes_win_and_each_kind_writes_its_own_slot(log):
    log.append(SetTarget(column="a"))
    log.append(SetLens(lenses=["dietary", "clinical"]))
    log.append(SetTarget(column="b"))
    log.append({"kind": "set_task", "column": "b", "task": "binary"})
    log.append(SetPurpose(purpose="inference"))
    assert log.state() == ProjectState(
        lens=["dietary", "clinical"], target="b", task="binary", purpose="inference"
    )


def test_revert_restores_the_value_before_and_revert_of_revert_reapplies(log):
    log.append(SetTarget(column="a"))
    b = log.append(SetTarget(column="b"))
    undo = log.append(Revert(decision_id=b.id))
    assert log.state().target == "a"
    redo = log.append(Revert(decision_id=undo.id))
    assert log.state().target == "b"
    log.append(Revert(decision_id=redo.id))
    assert log.state().target == "a"


def test_a_task_answer_stands_only_while_its_column_is_the_target(log):
    log.append(SetTarget(column="sex"))
    log.append(SetTask(column="sex", task="multiclass"))
    assert log.state().task == "multiclass"
    kcal = log.append(SetTarget(column="energy_kcal"))
    # The answer was about sex: energy_kcal gets detection (task None), not multiclass.
    assert (log.state().target, log.state().task) == ("energy_kcal", None)
    undo = log.append(Revert(decision_id=kcal.id))
    assert (log.state().target, log.state().task) == ("sex", "multiclass")
    log.append(Revert(decision_id=undo.id))
    assert (log.state().target, log.state().task) == ("energy_kcal", None)

    log.append(SetTask(column="energy_kcal", task="regression"))
    assert log.state().task == "regression"
    log.append(SetTarget(column="sex"))  # back to sex: its own latest answer applies
    assert log.state().task == "multiclass"
    log.append(SetTarget(column="age"))  # never answered for age
    assert log.state().task is None


def test_a_task_answer_must_name_the_current_target():
    task = {"kind": "set_task", "column": "sex", "task": "binary"}
    assert validate(task, {"target": "sex"}) == SetTask(column="sex", task="binary")
    with pytest.raises(Refusal) as refused:
        validate(task, {"target": "energy_kcal"})
    assert refused.value.code == "not_the_target"
    with pytest.raises(Refusal) as refused:
        validate(task, {"target": None})
    assert refused.value.code == "no_target"
    validate(task, {"columns": ["sex"]})  # a context that does not know the target
    with pytest.raises(ValidationError):
        parse_decision({"kind": "set_task", "task": "binary"})  # the column is required


def test_reverting_the_only_write_unsets_the_slot(log):
    only = log.append(SetPurpose(purpose="prediction"))
    log.append(Revert(decision_id=only.id))
    assert log.state().purpose is None


def test_reverting_a_superseded_write_leaves_the_later_one(log):
    a = log.append(SetTarget(column="a"))
    log.append(SetTarget(column="b"))
    log.append(Revert(decision_id=a.id))
    assert log.state().target == "b"


def test_a_revert_of_an_unknown_id_is_refused_and_not_recorded(log):
    log.append(SetTarget(column="a"))
    with pytest.raises(Refusal) as refused:
        log.append(Revert(decision_id="no-such-decision"))
    assert refused.value.code == "unknown_decision"
    assert len(log.records()) == 1


def test_a_revert_of_a_record_that_writes_no_slot_is_refused(log, monkeypatch):
    purpose = log.append(SetPurpose(purpose="prediction"))
    monkeypatch.delitem(d.SLOTS, "set_purpose")  # as a future non-slot kind would be
    with pytest.raises(Refusal) as refused:
        log.append(Revert(decision_id=purpose.id))
    assert refused.value.code == "unknown_decision"


def test_reverting_twice_is_refused_with_a_way_out(log):
    a = log.append(SetTarget(column="a"))
    undo = log.append(Revert(decision_id=a.id))
    with pytest.raises(Refusal) as refused:
        log.append(Revert(decision_id=a.id))
    body = refused.value.to_dict()["error"]
    assert body["code"] == "already_reverted"
    assert body["exits"] == [
        {"label": "Undo the earlier revert instead", "decision": {"kind": "revert", "decision_id": undo.id}}
    ]


def test_fold_refuses_a_log_whose_revert_names_nothing(tmp_path):
    record = d.DecisionRecord(
        id="r1", seq=1, at="2026-09-27T12:00:00Z", decision=Revert(decision_id="ghost")
    )
    with pytest.raises(Refusal) as refused:
        fold([record])
    assert refused.value.code == "unknown_decision"


def test_records_are_json_lines_with_utc_times_and_survive_reopening(tmp_path):
    path = tmp_path / "p" / "decisions.jsonl"
    first = DecisionLog(path)
    rec = first.append(SetTarget(column="a"), note="the outcome")
    first.append(SetTask(column="a", task="regression"))
    line = json.loads(path.read_text().splitlines()[0])
    assert line == {
        "id": rec.id,
        "seq": 1,
        "at": line["at"],
        "note": "the outcome",
        "decision": {"kind": "set_target", "column": "a"},
    }
    assert line["at"].endswith("Z")
    again = DecisionLog(path).records()
    assert [r.seq for r in again] == [1, 2]
    assert again[0] == rec and again[0].at.tzinfo == timezone.utc


def test_concurrent_appends_from_threads_get_distinct_sequence_numbers(log):
    threads = [
        threading.Thread(target=lambda i=i: log.append(SetTarget(column=f"c{i}")))
        for i in range(16)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(r.seq for r in log.records()) == list(range(1, 17))


def _append_many(path: str, n: int) -> None:
    other = DecisionLog(path)
    for i in range(n):
        other.append(SetTarget(column=f"p{i}"))


def test_concurrent_appends_from_processes_get_distinct_sequence_numbers(tmp_path):
    path = str(tmp_path / "decisions.jsonl")
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_append_many, args=(path, 15)) for _ in range(2)]
    for p in procs:
        p.start()
    _append_many(path, 15)
    for p in procs:
        p.join(30)
    assert sorted(r.seq for r in DecisionLog(path).records()) == list(range(1, 46))


def test_a_torn_last_line_is_skipped_and_the_next_append_starts_clean(tmp_path):
    path = tmp_path / "decisions.jsonl"
    log = DecisionLog(path)
    log.append(SetTarget(column="a"))
    with open(path, "a") as f:
        f.write('{"id": "torn", "seq": 2, "at"')  # a crash mid-write
    assert [r.decision.column for r in log.records()] == ["a"]
    log.append(SetTarget(column="b"))
    assert [r.decision.column for r in DecisionLog(path).records()] == ["a", "b"]


def test_lens_must_be_non_empty_and_unique():
    with pytest.raises(ValidationError):
        SetLens(lenses=[])
    with pytest.raises(ValidationError):
        SetLens(lenses=["dietary", "dietary"])
    with pytest.raises(ValidationError):
        parse_decision({"kind": "set_lens", "lenses": ["astrology"]})
    with pytest.raises(ValidationError):
        parse_decision({"kind": "set_target", "column": "a", "extra": 1})


def test_validators_refuse_a_target_that_is_not_a_column():
    columns = {"columns": [{"name": "bmi"}, {"name": "age"}]}
    assert validate({"kind": "set_target", "column": "bmi"}, columns) == SetTarget(column="bmi")
    with pytest.raises(Refusal) as refused:
        validate({"kind": "set_target", "column": "weight"}, columns)
    assert refused.value.code == "unknown_column"
    with pytest.raises(Refusal) as refused:
        validate(SetTarget(column="__row_id"))
    assert refused.value.code == "row_identity"
    validate(SetTarget(column="anything"))  # no context: nothing to check against


def test_register_kind_rejects_a_slot_the_state_does_not_have():
    with pytest.raises(ValueError, match="no slot"):
        d.register_kind(SetTarget, "not_a_slot")

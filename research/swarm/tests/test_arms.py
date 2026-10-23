"""The mechanical arms (SWARM_PLAN §6; ``research/swarm/arms.py``): the card's question, the default
path's and the random path's answers on asks built here, the walk on a session that serves those
asks (and what stops it, classed by what the engine said), and, slow, each path through the
driver's own commands on a small dataset that is not one of the pilot's six
(``clinical_readmission_predict``, 480 rows) to a record that validates.

Known truth: the event level from the card's own words; each default from what the ask offers in
its order and the app's own guess and detection, built here; each random draw from numpy's
generator at the same seed, drawn here; the record from the schema (``record.validate``)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from research.swarm import arms


# ── the card's question ──────────────────────────────────────────────────────


@pytest.mark.parametrize("dataset, target, purpose, exposure", [
    ("clinical_readmission_predict", "readmit_30d", "prediction", None),
    ("survey_support_estimate", "sought_support", "inference", "age"),
])
def test_the_question_is_the_cards(dataset, target, purpose, exposure):
    card = arms.card_of(dataset)
    q = arms.Question.of_card(card)
    assert (q.target, q.purpose, q.exposure) == (target, purpose, exposure)
    assert q.target == card["outcome"]["column"]


@pytest.mark.parametrize("meaning, levels, event", [
    ("readmitted within 30 days (1 yes, 0 no)", ["0", "1"], "1"),
    ("fasting plasma glucose of 126 mg/dL or more (1) or below (0)", ["0", "1"], "1"),
    ("an outcome", ["0", "1"], "1"),
    ("an outcome", ["no", "yes"], "no"),
    ("bought (Y yes, N no)", ["N", "Y"], "Y"),
    ("a level not offered (7 yes)", ["0", "1"], "1"),
], ids=["yes_code", "bare_code", "zero_one", "first", "letters", "not_offered"])
def test_the_event_level_is_read_from_the_cards_words(meaning, levels, event):
    q = arms.Question(target="y", purpose="prediction", meaning=meaning)
    assert q.event(levels) == event
    assert q.event([]) is None


def test_a_card_whose_purpose_is_neither_is_refused():
    with pytest.raises(ValueError, match="Predict or Estimate"):
        arms.Question.of_card({"purpose": "Describe", "outcome": {"column": "y"}})


# ── the default path's answers ───────────────────────────────────────────────

Q = arms.Question(target="y", purpose="prediction", meaning="(1 yes, 0 no)")


def opt(oid, decision, **extra):
    return {"id": oid, "label": oid, "decision": decision, **extra}


def ask(key, options, **extra):
    return {"question": {"key": key, "status": "open"}, "options": options, **extra}


def no_read(path):
    return None


def test_the_card_answers_the_outcome_its_level_and_the_goal():
    d = arms.Default(Q)
    target = ask("target", [opt("x", {"kind": "set_target", "column": "x"}),
                            opt("y", {"kind": "set_target", "column": "y"})])
    assert d.answer("target", target, no_read)[0] == ["y"]
    event = ask("event", [opt("0", {}), opt("1", {})])
    assert d.answer("event", event, no_read)[0] == ["1"]
    purpose = ask("purpose", [opt("inference", {}), opt("prediction", {})])
    assert d.answer("purpose", purpose, no_read)[0] == ["prediction"]


def test_the_task_is_the_apps_detection_and_the_lens_its_hint_else_the_one_that_claims_none():
    d = arms.Default(Q)
    task = ask("task", [opt("time_to_event", {"kind": "set_task", "task": "time_to_event"}),
                        opt("binary", {"kind": "set_task", "task": "binary"})],
               data={"detected_task": "binary"})
    assert d.answer("task", task, no_read)[0] == ["binary"]  # not the first offered
    lenses = [opt(v, {"kind": "set_lens", "lenses": [v]}) for v in
              ("dietary", "clinical", "survey", "other")]
    assert d.answer("lens", ask("lens", lenses, data={"lens_hints": [{"lens": "survey"}]}),
                    no_read)[0] == ["survey"]
    assert d.answer("lens", ask("lens", lenses), no_read)[0] == ["other"]


def test_any_other_question_takes_the_first_option_that_records_something_by_itself():
    d = arms.Default(Q)
    q = ask("grain_like", [opt("needs", None, needs=["column"]),
                           opt("a", {"kind": "k", "v": 1}), opt("b", {"kind": "k", "v": 2})])
    assert d.answer("grain_like", q, no_read)[0] == ["a"]
    with pytest.raises(arms.Stuck):
        d.answer("grain_like", ask("grain_like", [opt("needs", None, needs=["x"])]), no_read)


def way(oid, decision):
    return {"id": oid, "label": oid, "decision": decision}


def test_a_reading_is_answered_at_the_apps_guess_else_its_first_way():
    code = way("x-1", {"kind": "confirm_reading", "reading": "code_or_count", "column": "age",
                       "value": "code"})
    amount = way("x-2", {"kind": "confirm_reading", "reading": "code_or_count", "column": "age",
                         "value": "amount"})
    guessed = {"readings": {"exits": [code, amount], "groups": [
        {"kind": "code_or_count", "columns": ["age"], "guess": "amount"}]}}
    d = arms.Default(Q)
    assert d.settle(guessed)[0] == "x-2"
    unguessed = {"readings": {"exits": [code, amount], "groups": []}}
    assert d.settle(unguessed)[0] == "x-1"
    assert d.settle({"readings": {"exits": []}}) is None


def test_a_role_is_confirmed_as_proposed_and_a_finding_takes_its_first_repair():
    d = arms.Default(Q, proposed={"x": "covariate"})
    role = {"also_open": [{"items": [{"options": [
        way("x-r1", {"kind": "confirm_role", "column": "x", "role": "exposure"}),
        way("x-r2", {"kind": "confirm_role", "column": "x", "role": "covariate"})]}]}]}
    assert d.settle(role)[0] == "x-r2"
    finding = {"also_open": [{"source": "finding", "key": "f1", "options": [
        way("x-d", {"kind": "dismiss_finding", "finding_id": "f1"}),
        way("x-a", {"kind": "apply_repair", "finding_id": "f1"}),
        way("x-b", {"kind": "apply_repair", "finding_id": "f1"})]}]}
    assert d.settle(finding)[0] == "x-a"


def test_the_default_path_records_no_declaration_and_declares_the_first_family_final():
    d = arms.Default(Q)
    lines = {"also_open": [{"source": "declaration", "key": "set_selection", "options": [
        opt("set_selection:none", {"kind": "set_selection", "method": "none"})]}]}
    assert d.declarations(lines) == []
    assert d.family(["linear", "random_forest"]) == "linear"
    assert d.way([{"id": None, "decision": None}, {"id": "x-9", "decision": {"kind": "k"}}]) \
        == "x-9"


# ── the random path's draws ──────────────────────────────────────────────────


def test_the_random_path_draws_as_numpy_does_at_its_seed_among_what_records_something():
    options = [opt(f"o{i}", {"kind": "k", "v": i}) for i in range(4)] + [
        opt("needs", None, needs=["x"])]
    r = arms.Random(Q, seed=7)
    got = [r.answer("missing", ask("missing", options), no_read)[0][0] for _ in range(40)]
    rng = np.random.default_rng(7)
    assert got == [f"o{int(rng.integers(4))}" for _ in range(40)]
    assert "needs" not in got and set(got) == {"o0", "o1", "o2", "o3"}
    again = arms.Random(Q, seed=7)
    assert [again.answer("missing", ask("missing", options), no_read)[0][0]
            for _ in range(40)] == got
    other = arms.Random(Q, seed=8)
    assert [other.answer("missing", ask("missing", options), no_read)[0][0]
            for _ in range(40)] != got


def test_the_random_path_keeps_the_cards_question_and_answers_each_declaration_once():
    r = arms.Random(Q, seed=1)
    target = ask("target", [opt("x", {}), opt("y", {})])
    assert r.answer("target", target, no_read)[0] == ["y"]
    lines = {"also_open": [
        {"source": "declaration", "key": "set_selection", "options": [
            opt("set_selection:none", {"kind": "set_selection", "method": "none"}),
            opt("set_selection:stability", {"kind": "set_selection", "method": "stability"})]},
        {"source": "declaration", "key": "set_batch", "options": [
            opt("set_batch:custom", None, needs=["column"])]},
        {"source": "finding", "key": "f", "options": []}]}
    drawn = r.declarations(lines)
    assert [ids[0].split(":")[0] for ids, _, _ in drawn] == ["set_selection"]


# ── the walk ─────────────────────────────────────────────────────────────────


class Scripted:
    """A session that serves the asks given, one per ask, and keeps every answer."""

    def __init__(self, asks, refuse=None, fail=None):
        self.asks, self.answers = list(asks), []
        self.refuse, self.fail = refuse or {}, fail or set()

    def ask(self):
        return self.asks.pop(0)

    def read(self, path):
        return None

    def answer(self, ids, sets, reason, marker=arms.DEFAULT_MARKER):
        self.answers.append((list(ids), list(sets)))
        if ids[0] in self.fail:
            return arms.Said("failed", {"refused": {"status": 500}}, code="http_500")
        if ids[0] in self.refuse:
            return arms.Said("refused", {}, by="server", code="no", exits=self.refuse[ids[0]])
        return arms.Said("ok", {"recorded": {"id": "r"}})


def test_a_walk_settles_a_reading_first_answers_each_question_and_stops_at_fit():
    reading = way("x-2", {"kind": "confirm_reading", "reading": "code_or_count",
                          "column": "age", "value": "amount"})
    asks = [{**ask("purpose", [opt("prediction", {"kind": "set_purpose"})]),
             "readings": {"exits": [reading], "groups": []}},
            ask("purpose", [opt("prediction", {"kind": "set_purpose"})]),
            ask("missing", [opt("impute", {"kind": "set_missing"})]),
            {"question": None, "fit_reached": True}]
    s = Scripted(asks)
    steps: list[arms.Step] = []
    q = arms.walk(s, arms.Default(Q), steps=steps)
    assert q["fit_reached"] is True
    assert s.answers == [(["x-2"], []), (["prediction"], []), (["impute"], [])]
    assert [st.at for st in steps] == ["way", "question", "question"]


def test_a_refused_answer_takes_its_way_forward_and_a_failed_one_stops_the_walk():
    forward = [{"id": "x-f", "decision": {"kind": "set_missing"}}]
    asks = [ask("missing", [opt("impute", {"kind": "set_missing"})]),
            {"question": None, "fit_reached": True}]
    s = Scripted(asks, refuse={"impute": forward})
    arms.walk(s, arms.Default(Q))
    assert s.answers == [(["impute"], []), (["x-f"], [])]
    failing = Scripted([ask("missing", [opt("impute", {"kind": "set_missing"})])],
                       fail={"impute"})
    with pytest.raises(arms.Stuck, match="failed"):
        arms.walk(failing, arms.Default(Q))


# ── each path through the driver, to its record ──────────────────────────────


@pytest.mark.slow
@pytest.mark.parametrize("path, seed", [("default", None), ("random", 7)])
def test_each_path_runs_through_the_driver_to_a_record_that_validates(tmp_path, path, seed):
    from research.swarm import driver
    from research.swarm.record import validate

    out = arms.run_arm("clinical_readmission_predict", path, seed=seed, runs=tmp_path / "runs",
                       corpus=tmp_path / "corpus")
    assert out["error"] is None and out["valid"], (out["error"], out["record"])
    assert driver.RUN_ID.fullmatch(out["run"]) and out["run"].startswith("run_")
    rec = validate(json.loads(open(out["record"]).read()))
    assert rec.run.analyst_model == ("mechanical-default" if path == "default"
                                     else f"mechanical-random-{seed}")
    assert rec.run.persona is None and rec.dataset.id == "clinical_readmission_predict"
    assert rec.dataset.control == "known_answer"
    assert rec.question.target == "readmit_30d" and rec.question.purpose == "prediction"
    assert rec.result.opening is not None and rec.result.scores  # the seal opened, once
    assert rec.export.reached is True
    commands = [json.loads(x)["command"] for x in
                (tmp_path / "runs" / out["run"] / "commands.jsonl").read_text().splitlines()]
    assert commands[0] == "start" and "seal" in commands and commands[-2:] == ["export", "stop"]
    analyst = [d for d in rec.decisions if d.source == "analyst"]
    assert analyst and all(d.tag is not None for d in analyst if d.kind not in
                           ("open_seal", "revert"))
    if path == "default":
        assert not [d for d in rec.decisions if d.kind.startswith("set_selection")]


# ── what the engine said stops the walk, classed ─────────────────────────────


def stage_view(**errors):
    """A project view whose stages ``errors`` are in error, each ``(words, refusal or None)``."""
    stages = {"working": {"status": "fresh", "key": "w1"}}
    for name, (words, refusal) in errors.items():
        stages[name] = {"status": "error", "key": f"{name}1", "error": words,
                        "refusal": refusal}
    return {"stages": stages}


def waiting_on(view, *names):
    """The ask the driver serves for a question waiting on ``names`` (``driver.the_question``:
    ``failed`` from ``failed_on``, ``fit_failed`` from ``failures`` over the same view)."""
    from research.swarm import driver

    step = {"key": "models", "status": "waiting", "waiting_on": list(names)}
    return {"question": {"key": "models", "status": "waiting", "waiting_on": list(names)},
            "failed": driver.failed_on(view, step),
            "fit_failed": driver.failures(view, [*names, "fit"])}


REFUSAL = {"stage": "cohort", "code": "too_few_rows",
           "message": "The answers recorded leave 3 of the 60 rows.",
           "exits": [{"id": "x-c", "label": "Fill the blanks",
                      "decision": {"kind": "set_missing", "strategy": "impute"}}]}


def test_a_question_waiting_on_a_stage_that_failed_untyped_stops_the_walk_failed():
    view = stage_view(shelf=("RuntimeError: boom", None))
    with pytest.raises(arms.Met) as got:
        arms.walk(Scripted([waiting_on(view, "shelf")]), arms.Default(Q))
    assert got.value.kind == "failed"
    assert got.value.detail["code"] == "stage_failed" and got.value.detail["stage"] == "shelf"
    assert got.value.detail["message"] == "RuntimeError: boom"


def test_a_question_waiting_on_a_stage_that_refused_in_words_stops_the_walk_refused():
    view = stage_view(cohort=("ValueError: The answers recorded leave 3 of the 60 rows.",
                              REFUSAL))
    with pytest.raises(arms.Met) as got:
        arms.walk(Scripted([waiting_on(view, "cohort")]), arms.Default(Q))
    detail = got.value.detail
    assert got.value.kind == "refused" and detail["by"] == "engine" and detail["at"] == "stage"
    assert (detail["stage"], detail["code"], detail["message"]) == (
        "cohort", "too_few_rows", "The answers recorded leave 3 of the 60 rows.")
    assert detail["exits"] == [{"label": "Fill the blanks",
                                "decision": {"kind": "set_missing", "strategy": "impute"}}]


def test_one_untyped_stage_among_typed_ones_is_a_failure():
    view = stage_view(cohort=("ValueError: refused", REFUSAL), shelf=("KeyError: 'x'", None))
    with pytest.raises(arms.Met) as got:
        arms.walk(Scripted([waiting_on(view, "cohort", "shelf")]), arms.Default(Q))
    assert got.value.kind == "failed" and got.value.detail["stage"] == "shelf"
    assert [s["kind"] for s in got.value.detail["stages"]] == ["refused", "failed"]


def test_an_answer_refused_with_no_way_forward_and_no_other_option_stops_the_walk_refused():
    asks = [ask("missing", [opt("impute", {"kind": "set_missing"})]),
            ask("missing", [opt("impute", {"kind": "set_missing"})])]
    s = Scripted(asks, refuse={"impute": []})
    with pytest.raises(arms.Met) as got:
        arms.walk(s, arms.Default(Q))
    detail = got.value.detail
    assert got.value.kind == "refused" and detail["code"] == "no" and detail["by"] == "server"
    assert detail["at"] == "walk" and detail["question"] == "missing"
    assert detail["answered"] == ["impute"]


def test_the_cards_own_answer_refused_with_no_way_forward_stops_the_walk_refused():
    asks = [ask("target", [opt("y", {"kind": "set_target", "column": "y"}),
                           opt("x", {"kind": "set_target", "column": "x"})])] * 2
    s = Scripted(asks, refuse={"y": []})
    with pytest.raises(arms.Met) as got:
        arms.walk(s, arms.Default(Q))
    assert got.value.kind == "refused" and got.value.detail["question"] == "target"
    assert s.answers == [(["y"], [])]  # never another of the card's answers


def test_a_walk_that_meets_its_own_limit_stops_unclassed():
    """A question the path cannot answer (every option takes a field) is the walk's own limit,
    not what the engine said."""
    s = Scripted([ask("grain_like", [opt("needs", None, needs=["column"])])])
    with pytest.raises(arms.Stuck) as got:
        arms.walk(s, arms.Default(Q))
    assert not isinstance(got.value, arms.Met)


def test_an_answer_that_failed_stops_the_walk_failed():
    failing = Scripted([ask("missing", [opt("impute", {"kind": "set_missing"})])],
                       fail={"impute"})
    with pytest.raises(arms.Met) as got:
        arms.walk(failing, arms.Default(Q))
    assert got.value.kind == "failed" and got.value.detail["code"] == "http_500"

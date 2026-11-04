"""The mechanical arms (SWARM_PLAN §6; ``research/swarm/arms.py``): the card's question, the default
path's and the random path's answers on asks built here, the walk on a session that serves those
asks (and what stops it, classed by what the engine said), a local session's posts keeping a
refusal's ways forward until an answer is recorded as the driver's do, a copy dropped once it
stops computing and, where a late write lands in its removal, removed again once the server can
no longer read it, else kept, never raising, a copy's fit cancelled through the server's Cancel
and, only where the copy stays busy, each job it holds; a press of Fit whose wait ran out,
waited out with reads alone, on a server modeled here on the one the
driver's ``fit`` meets (:class:`Server`): Fit pressed once, and the press kept, the fit record's
press and what the arm says those of the driver's own ``fit`` whose wait outlasts the fit,
however the fit ends; each path through a driver modeled here, its Fit the driver's own on that
server, waited out to the export or stopped at the run's time budget, and a fit blocked, held,
refused or failed stopping the arm as before; under Estimate, a press the open noticings' triage
holds taking its way forward "Keep the recommended dispositions" and pressed again, by either path
(on the server's refusal built from a triage written here) and by a local session on the walking
table walked to Fit, while a blocking noticing's refusal, and any other, is the answer as it came;
and, slow, each path through the driver's own commands on a small dataset that is not one of the
pilot's (``clinical_readmission_predict``, 480 rows) to a record that validates.

Known truth: the event level from the card's own words; each default from what the ask offers in
its order and the app's own guess and detection, built here; each random draw from numpy's
generator at the same seed, drawn here, the adjustment's columns each drawn in or out as the
driver's composer takes them; the press kept, built here as ``ProjectService.press_fit`` builds
it (the seq of the last record before it, the shelf's estimate where the shelf has measured it),
and what the driver's own ``fit`` says when its wait outlasts the fit; the time waited from a hand
clock that only the waits move, derived here from the budget and the wait; the arm's commands
against those of the same arm whose fit takes 10 s; the record from the schema
(``record.validate``); a late write in a copy's removal from the OS's own removal and the error
it raises (ENOTEMPTY) on a file written into the folder after the removal listed it, and the
server's read of a copy as ``Workspace.get`` reads it, from its ``project.json``, and its 404
for a project it cannot find (``errors.not_found``)."""
from __future__ import annotations

import json
from pathlib import Path

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


IN = {"causes_exposure": "yes", "causes_outcome": "yes", "after_exposure": "no"}
OUT = {"causes_exposure": "no", "causes_outcome": "no", "after_exposure": "no"}


def adjustment_ask(columns):
    """The adjustment question as the driver composes it: a guessed group's one tap, and the
    answers option over ``columns``, each column's three answers its fields."""
    from research.swarm import options as opts

    groups = [{"key": "confounder", "label": "Confounders", "columns": columns[:1],
               "decision": {"kind": "set_adjustment", "exposure": "walks",
                            "answers": {columns[0]: dict(IN)}}} if columns else None,
              {"key": "unguessed", "label": "No guess", "columns": columns[1:],
               "decision": None}]
    composed = opts.compose("adjustment", {"key": "adjustment", "status": "open"}, {"state": {}},
                            {"proposals": {"adjustment": {"exposure": "walks", "groups": [
                                g for g in groups if g]}}}.get, None)
    return composed, ask("adjustment", composed["options"])


def test_the_random_path_draws_each_adjustment_column_in_or_out_as_numpy_does_at_its_seed():
    """F4: the random path answers the adjustment question with every column the answers option
    takes, each drawn in (a confounder's answers) or out (a cause of neither's), one draw of
    numpy's generator at the seed a column, in the option's order; the driver's composer takes the
    answer (``options.resolve``). Drawn again at the same seed, the same; at others, both ways
    for each column."""
    from research.swarm import options as opts

    columns = ["age", "chronic_pain", "height_cm", "health_concern_score"]
    composed, q = adjustment_ask(columns)
    ids, sets, _ = arms.Random(Q, seed=7).answer("adjustment", q, no_read)
    rng = np.random.default_rng(7)
    drawn = [bool(rng.integers(2)) for _ in columns]
    words = {True: ",".join(IN[k] for k in opts.ADJUSTMENT_ASKED),
             False: ",".join(OUT[k] for k in opts.ADJUSTMENT_ASKED)}
    assert ids == ["answers"] and sets == [f"{c}={words[d]}" for c, d in zip(columns, drawn)]
    resolved = opts.resolve(composed, ids, sets)
    assert resolved == {"kind": "set_adjustment", "exposure": "walks",
                        "answers": {c: IN if d else OUT for c, d in zip(columns, drawn)}}
    assert arms.Random(Q, seed=7).answer("adjustment", q, no_read)[:2] == (ids, sets)
    seen = {c: set() for c in columns}
    for seed in range(40):
        _, drew, _ = arms.Random(Q, seed=seed).answer("adjustment", q, no_read)
        for c, s in zip(columns, drew):
            seen[c].add(s.split("=", 1)[1])
    assert all(len(v) == 2 for v in seen.values())


def test_with_no_column_to_answer_the_random_path_draws_among_the_options_as_before():
    """The guard's other side: an adjustment question whose answers option takes no column draws
    among the options that record something by themselves, as for any question."""
    options = [opt("confounder", {"kind": "set_adjustment", "exposure": "walks",
                                  "answers": {"age": dict(IN)}}), opt("other", {"kind": "k"}),
               opt("answers", None, columns=[])]
    q = ask("adjustment", options)
    got = [arms.Random(Q, seed=s).answer("adjustment", q, no_read)[0][0] for s in range(20)]
    rng_draws = [["confounder", "other"][int(np.random.default_rng(s).integers(2))]
                 for s in range(20)]
    assert got == rng_draws


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


def test_a_local_post_keeps_a_refusals_ways_until_an_answer_is_recorded_as_the_driver_does(
        tmp_path, monkeypatch):
    """The driver's ``answer`` and ``revisit`` keep a refusal's ways forward for the next ask
    (``last_refusal_exits``) and clear them once an answer is recorded; a :class:`arms.Local`
    session's posts do the same, so a way the walk took once is not served again as the last
    refusal's."""
    from research.swarm import driver

    served = []
    exits = [{"label": "Fill the blanks", "decision": {"kind": "set_missing",
                                                       "strategy": "impute"}}]
    refusal = driver.Response(409, json.dumps({"error": {"code": "not_here", "message": "No.",
                                                         "exits": exits}}).encode(), {})
    monkeypatch.setattr(driver, "post_decision", lambda run, client, decision, note: served.pop(0))
    monkeypatch.setattr(driver, "recorded", lambda view, kind: {"recorded": {"kind": kind}})
    local = arms.Local(tmp_path / "home")
    local.use("p1")
    local.client = None  # no server: each post is served above
    for kind in ("set_missing", "revert"):
        served[:] = [refusal, driver.Response(200, b"{}", {})]
        assert local.decide({"kind": kind}, "a test").kind == "refused"
        kept = local.run.kept_exits()
        assert [e["decision"] for e in kept] == [exits[0]["decision"]] and kept[0]["id"]
        assert local.decide({"kind": kind}, "a test").ok
        assert local.run.kept_exits() == []


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


def test_a_copy_still_written_to_after_its_stages_go_idle_is_removed_on_a_retry_or_kept(
        tmp_path, monkeypatch):
    """A late write can land in a copy between the idle check and its removal ("Directory not
    empty", as the readmission crash sweep met on 2026-10-23): drop waits for idle again and
    retries, and where the copy still cannot be removed it is kept and drop says so, so the sweep
    goes on rather than stopping on its own housekeeping."""
    import shutil

    session = arms.Local(tmp_path)
    (tmp_path / "projects" / "p1").mkdir(parents=True)
    monkeypatch.setattr(session, "idle", lambda pid=None, wait=0.0: True)
    real, calls = shutil.rmtree, []

    def flaky(path, *a, **k):
        calls.append(path)
        if len(calls) == 1:
            raise OSError(66, "Directory not empty", str(path))
        return real(path, *a, **k)

    monkeypatch.setattr(shutil, "rmtree", flaky)
    made = {"p1"}
    assert session.drop("p1", made) is True
    assert not (tmp_path / "projects" / "p1").exists() and made == set()

    (tmp_path / "projects" / "p2").mkdir(parents=True)

    def stuck(path, *a, **k):
        if "projects" in str(path) and not k.get("ignore_errors"):
            raise OSError(66, "Directory not empty", str(path))
        return real(path, *a, **k)

    monkeypatch.setattr(shutil, "rmtree", stuck)
    made = {"p2"}
    assert session.drop("p2", made) is False
    assert (tmp_path / "projects" / "p2").exists() and made == {"p2"}


# A copy's stages as the server serves them (``Engine._status``): none queued or running
# (``Local.BUSY``), or its fit running.
IDLE = {"ingest": {"status": "fresh", "job_id": None}, "fit": {"status": "stale", "job_id": None}}
BUSY = {"ingest": {"status": "fresh", "job_id": None}, "fit": {"status": "running", "job_id": "j1"}}


class Disk:
    """The server as a dropped copy's reads meet it (``GET /projects/{pid}``): the project read
    from its folder's ``project.json`` as ``Workspace.get`` reads it, and, where that is gone, the
    404 the server answers for a project it cannot find (``errors.not_found``); where it is there,
    the next of ``answers`` (the last again once they run out): stages served, or a status the
    server answers instead (404, its own; 500, an unhandled exception as the server serves it)."""

    def __init__(self, home, answers):
        self.home, self.answers, self.reads = home, list(answers), []

    def get(self, path, **query):
        from research.swarm import driver
        from turbotab.server.errors import not_found

        pid = path.rsplit("/", 1)[-1]
        self.reads.append(path)
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if not (self.home / "projects" / pid / "project.json").is_file() or answer == 404:
            return driver.Response(404, json.dumps(not_found(pid).body()).encode("utf-8"), {})
        if answer == 500:
            return driver.Response(500, b"Internal Server Error", {})
        return driver.Response(200, json.dumps({"stages": answer}).encode("utf-8"), {})


def a_copy(home, pid):
    """A copy's folders as ``Local.copy`` and ``Local.use`` leave them: its project folder (its
    ``project.json``, its decision log and a stage computed) and its driver folder."""
    folder = home / "projects" / pid
    (folder / "cache" / "ingest" / "k1").mkdir(parents=True)
    (folder / "project.json").write_text(json.dumps({"id": pid}), "utf-8")
    (folder / "decisions.jsonl").write_text("{}\n", "utf-8")
    (folder / "cache" / "ingest" / "k1" / "meta.json").write_text("{}", "utf-8")
    (home / "driver" / pid).mkdir(parents=True)
    return folder


def removing(monkeypatch, folder, *, refused=0, late=None):
    """The copy's removal, the OS's own (``shutil.rmtree``), with what lands in ``folder`` as it
    runs: the first ``refused`` files it would unlink refused (EACCES), so that removal stops
    having removed nothing; and a late write, a file written at the folder's top after the
    removal listed it, once it unlinks the ``project.json`` (``late="project.json"``) or after each
    file it unlinks (``late="each"``), so that it finds the folder not empty, as the OS says
    (ENOTEMPTY: 66 on macOS), as the step-4 sweeps' removals did. The errno of each removal that
    failed, in order."""
    import errno
    import os
    import shutil

    real_unlink, real_rmtree = os.unlink, shutil.rmtree
    raised, unlinked = [], []

    def unlink(name, *, dir_fd=None):
        if len(unlinked) < refused:
            unlinked.append(None)
            raise PermissionError(errno.EACCES, "Permission denied", name)
        real_unlink(name, dir_fd=dir_fd)
        unlinked.append(os.path.basename(name))
        if late == "each" or late == unlinked[-1]:
            (folder / f"late-{len(unlinked)}").write_text("late", "utf-8")

    def rmtree(path, *a, **k):
        try:
            return real_rmtree(path, *a, **k)
        except OSError as exc:
            raised.append(exc.errno)
            raise

    monkeypatch.setattr(os, "unlink", unlink)
    monkeypatch.setattr(shutil, "rmtree", rmtree)
    return raised


def test_a_copy_a_late_write_lands_in_as_it_is_removed_is_removed_again_once_it_cannot_be_read(
        tmp_path, monkeypatch):
    """A late write lands in a copy after the removal took its ``project.json`` ("Directory not
    empty", as both step-4 sweeps of the readmission dataset met on 2026-11-03), so the server can
    no longer read it (404): a copy it cannot read has stopped computing, so drop pauses and
    removes it again, its driver folder too, and says it was removed, where it once ended the sweep
    on the 404 (``DriverError``)."""
    import errno

    session, clock = arms.Local(tmp_path), Clock()
    folder = a_copy(tmp_path, "p1")
    session.client = Disk(tmp_path, [IDLE])
    monkeypatch.setattr(arms, "time", clock)
    raised = removing(monkeypatch, folder, late="project.json")
    made = {"p1", "p2"}
    assert session.drop("p1", made) is True
    assert raised == [errno.ENOTEMPTY]  # the first removal met the late write
    assert not folder.exists() and not (tmp_path / "driver" / "p1").exists() and made == {"p2"}
    assert session.client.reads == ["/projects/p1"] * 2  # idle before; its 404 after
    assert clock.now == arms.DROP_PAUSE  # one pause, then the removal that took it


@pytest.mark.parametrize("answers, refused", [([BUSY], 0), ([IDLE, BUSY], 1)],
                         ids=["before_its_removal", "after_a_removal_that_failed"])
def test_a_copy_still_computing_is_kept_and_drop_says_so(tmp_path, monkeypatch, answers, refused):
    """A copy whose stages stay computing, before its removal or once a removal failed having
    removed nothing, is waited for, :data:`arms.DROP_WAIT` seconds, and kept, whole, and drop
    says it was not removed: the other way of the copy that cannot be read."""
    session, clock = arms.Local(tmp_path), Clock()
    folder = a_copy(tmp_path, "p1")
    session.client = Disk(tmp_path, answers)
    monkeypatch.setattr(arms, "time", clock)
    raised = removing(monkeypatch, folder, refused=refused)
    made = {"p1"}
    assert session.drop("p1", made) is False
    assert len(raised) == refused and made == {"p1"}
    assert sorted(p.name for p in folder.iterdir()) == ["cache", "decisions.jsonl", "project.json"]
    assert (tmp_path / "driver" / "p1").is_dir()
    assert clock.now == arms.DROP_WAIT  # the wait for the copy that stayed busy, no more


def test_a_copy_that_late_writes_keep_landing_in_is_kept_after_its_removals_never_raising(
        tmp_path, monkeypatch):
    """Where a late write lands in each removal, the copy is removed :data:`arms.DROP_TRIES` times,
    a pause of :data:`arms.DROP_PAUSE` seconds between each two, and then kept, and drop says it
    was not removed; it never raises, though the server cannot read the copy."""
    import errno

    session, clock = arms.Local(tmp_path), Clock()
    folder = a_copy(tmp_path, "p1")
    session.client = Disk(tmp_path, [IDLE])
    monkeypatch.setattr(arms, "time", clock)
    raised = removing(monkeypatch, folder, late="each")
    made = {"p1"}
    assert session.drop("p1", made) is False
    assert raised == [errno.ENOTEMPTY] * arms.DROP_TRIES
    assert folder.is_dir() and (tmp_path / "driver" / "p1").is_dir() and made == {"p1"}
    assert clock.now == arms.DROP_PAUSE * (arms.DROP_TRIES - 1)


@pytest.mark.parametrize("answer, removed", [(404, True), (500, False)],
                         ids=["no_such_project", "a_server_failure"])
def test_a_copy_the_server_cannot_read_after_a_failed_removal_is_removed_only_on_its_404(
        tmp_path, monkeypatch, answer, removed):
    """A removal that failed having removed nothing (its ``project.json`` there), the server's
    read of the copy decides: its own 404 (there is no such project) says the copy cannot be read,
    so it is removed again; any other failure says nothing of whether the copy still computes, so
    it is kept, whole; either way drop never raises."""
    session, clock = arms.Local(tmp_path), Clock()
    folder = a_copy(tmp_path, "p1")
    session.client = Disk(tmp_path, [IDLE, answer])
    monkeypatch.setattr(arms, "time", clock)
    raised = removing(monkeypatch, folder, refused=1)
    made = {"p1"}
    assert session.drop("p1", made) is removed
    assert len(raised) == 1 and folder.exists() is not removed and (made == set()) is removed
    assert (folder / "project.json").is_file() is not removed
    assert clock.now == (arms.DROP_PAUSE if removed else 0.0)


class Served:
    """The server as a copy's cancel meets it: every post answered 200 and kept, and the project
    read with ``stages``."""

    def __init__(self, stages):
        self.stages, self.posts = stages, []

    def post(self, path, data=None, **query):
        from research.swarm import driver

        self.posts.append(path)
        return driver.Response(200, b"{}", {})

    def get(self, path, **query):
        from research.swarm import driver

        return driver.Response(200, json.dumps({"stages": self.stages}).encode("utf-8"), {})


BUSY_STAGES = {"design": {"status": "queued", "job_id": "j3"},
               "split": {"status": "running", "job_id": "j4"},
               "explore": {"status": "running", "job_id": None},  # a light stage: no job to stop
               "fit": {"status": "stale", "cancelled": True, "job_id": None},
               "shelf": {"status": "fresh", "job_id": "j0"}}
BUSY_JOBS = [{"stage": "design", "job": "j3", "status": 200},
             {"stage": "split", "job": "j4", "status": 200}]
CANCELS = ["/projects/p1/fit/cancel", "/projects/p1/jobs/j3/cancel", "/projects/p1/jobs/j4/cancel"]


@pytest.mark.parametrize("idle, posts, jobs", [
    ([True], CANCELS[:1], []), ([False, True], CANCELS, BUSY_JOBS),
    ([False, False], CANCELS, BUSY_JOBS),
], ids=["idle_after_the_fits_cancel", "idle_after_its_jobs_cancel", "never_idle"])
def test_a_copys_fit_is_cancelled_and_where_it_stays_busy_each_job_it_holds_too(
        tmp_path, monkeypatch, idle, posts, jobs):
    """The fit cancelled as Cancel on the flowchart cancels it (``POST …/fit/cancel``) and the copy
    waited for; only where it is still computing then is each job its stages hold queued or
    running cancelled too, and the copy waited for again: both ways."""
    session = arms.Local(tmp_path)
    session.client = Served(BUSY_STAGES)
    answers, waits = list(idle), []
    monkeypatch.setattr(session, "idle",
                        lambda pid=None, wait=0.0: (waits.append((pid, wait)), answers.pop(0))[1])
    done = session.cancel("p1")
    assert session.client.posts == posts
    assert done == {"fit": 200, "jobs": jobs, "idle": idle[-1], "wait": arms.CANCEL_WAIT}
    assert waits == [("p1", arms.CANCEL_WAIT)] * len(idle) and answers == []




# ── a fit that outlasts one press's wait ─────────────────────────────────────


RID = "run_01-" + "0" * 32
PID = "p1"
RECORDS = 28  # the decision records before Fit is pressed: the press keeps the last one's seq
ESTIMATE = 900.0  # the shelf's measured estimate of the fit, in seconds


class Clock:
    """The clock the arms and the driver read, by hand: their waits move it on (``time.sleep``),
    and nothing else does."""

    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


WAY = {"label": "Keep every row", "decision": {"kind": "exclusion", "rules": []}}
WORDS = "The answers leave 3 of 60 rows."
# How the fit ends, as the engine serves its stage (``Engine._status``); "never", it runs on.
ENDS = {
    "fresh": {"status": "fresh", "fresh": True},
    "refused": {"status": "error", "error": WORDS,
                "refusal": {"stage": "fit", "code": "too_few_rows", "message": WORDS,
                            "exits": []}},
    "refused_with_a_way": {"status": "error", "error": WORDS,
                           "refusal": {"stage": "fit", "code": "too_few_rows", "message": WORDS,
                                       "exits": [WAY]}},
    "failed": {"status": "error", "error": "ValueError: no rows"},
    "blocked": {"status": "blocked"},
    "held": {"status": "idle", "held": "fit"},
    "never": None,
}


class Server:
    """A run's server as the driver's ``fit`` meets it, modeled here on ``ProjectService`` and the
    engine, on the hand clock. A press (``POST …/fit``, :meth:`press`) keeps the press as
    ``press_fit`` does (``fit_press.record_press``: the outcome, the shelf's estimate where the
    shelf has measured it, else None, and the seq of the last record before it; the newest press
    replaces the last) and, under Estimate, then records the plan's lock once. The first press
    releases the fit, which waits while the shelf measures (``held: "estimate"``, until
    ``shelf_at``), runs ``seconds`` and ends as ``end`` names (:data:`ENDS`); a press while it runs
    or after it ended starts nothing (the engine keeps a stage's job, and its error, for a key the
    answers leave as it was). A fit that ends fresh is recorded as it ends, with the press as it
    stands then (``ProjectService._record``'s ``press``). ``GET /projects/{pid}`` serves the
    fit's status, and ``GET …/fits`` the records. Each press is kept, with when it came
    (``posts``); :meth:`refit` stands for answers that changed the fit's key."""

    def __init__(self, clock, *, purpose="prediction", seconds=600.0, end="fresh", shelf_at=0.0):
        self.clock, self.purpose, self.shelf_at = clock, purpose, shelf_at
        self.seqs = list(range(1, RECORDS + 1))
        self.kept: dict | None = None  # the press file, ``fit_pressed.json``
        self.locked = False
        self.posts: list[float] = []
        self.records: list[dict] = []
        self.made = 1
        self.fit = {"key": "k1", "from": None, "seconds": seconds, "end": end, "recorded": False}

    def _begins(self):
        f = self.fit
        return None if f["from"] is None else max(f["from"], self.shelf_at)

    def settle(self):
        """The fit's record, written as the fit ended fresh: before anything that came after."""
        f, begins = self.fit, self._begins()
        if begins is not None and f["end"] == "fresh" and not f["recorded"] \
                and self.clock.now >= begins + f["seconds"]:
            f["recorded"] = True
            self.records.append({"fit_key": f["key"], "seconds": f["seconds"],
                                 "press": dict(self.kept or {})})

    def status(self):
        f, now, begins = self.fit, self.clock.now, self._begins()
        common = {"stage": "fit", "key": f["key"]}
        if begins is None:
            return common | {"status": "idle", "held": "fit"}
        if now < begins:
            return common | {"status": "idle", "held": "estimate"}
        if ENDS[f["end"]] is None or now < begins + f["seconds"]:
            return common | {"status": "running",
                             "progress": round(min((now - begins) / f["seconds"], 0.999), 3)}
        return common | ENDS[f["end"]]

    def press(self):
        self.settle()
        self.posts.append(self.clock.now)
        measured = self.clock.now >= self.shelf_at
        self.kept = {"target": "y", "seconds": ESTIMATE if measured else None,
                     "seq": max(self.seqs)}
        if self.purpose != "prediction" and not self.locked:
            self.seqs.append(max(self.seqs) + 1)  # lock_plan, after the press
            self.locked = True
        if self.fit["from"] is None:
            self.fit["from"] = self.clock.now
        return {"purpose": self.purpose, "locks": self.purpose != "prediction",
                "locked": self.locked, "pressed": True}

    def refit(self, seconds, end="fresh"):
        """Answers that changed the fit's key: the fit at the new key starts at once, Fit having
        been pressed for the outcome with an estimate it is within (``fit_press.holds``)."""
        self.settle()
        self.made += 1
        self.fit = {"key": f"k{self.made}", "from": self.clock.now if self.kept else None,
                    "seconds": seconds, "end": end, "recorded": False}

    def view(self):
        self.settle()
        return {"state": {"purpose": self.purpose, "target": "y", "models": ["logistic"],
                          "seal_opened": False},
                "stages": {"fit": self.status()}, "decisions": [], "interview": []}


class Wire:
    """The driver's client (``driver.Client``) to a modeled :class:`Server`: the project, its fit
    records and the press of Fit, as JSON; any other path a 404 in the server's words."""

    def __init__(self, server):
        self.server = server

    @staticmethod
    def response(status, body):
        from research.swarm import driver

        return driver.Response(status, json.dumps(body).encode("utf-8"), {})

    def get(self, path, **query):
        if path == f"/projects/{PID}":
            return self.response(200, self.server.view())
        if path == f"/projects/{PID}/fits":
            self.server.settle()
            return self.response(200, {"records": list(self.server.records)})
        return self.response(404, {"error": {"code": "not_found", "message": "Not modeled."}})

    def post(self, path, data=None, **query):
        assert path == f"/projects/{PID}/fit", path
        return self.response(200, self.server.press())


def wired(tmp_path, monkeypatch, server):
    """The run ``RID`` in ``tmp_path / "runs"`` as ``start`` leaves it (its ``driver.json``: the
    project, the fit budget, a scripted analyst), the driver's client wired to ``server``, and the
    driver's and the arms' clock the server's; the runs' home."""
    from research.swarm import driver

    runs = tmp_path / "runs"
    (runs / RID).mkdir(parents=True)
    (runs / RID / "driver.json").write_text(json.dumps({
        "run": RID, "pid": None, "port": 9, "project": PID, "analyst": "scripted",
        "budget": {"fits": 40, "seconds": 4 * 3600.0}, "ended": None}), encoding="utf-8")
    monkeypatch.setattr(driver, "Client", lambda port, timeout=None: Wire(server))
    monkeypatch.setattr(driver, "time", server.clock)
    monkeypatch.setattr(arms, "time", server.clock)
    monkeypatch.setenv(driver.RUNS_ENV, str(runs))
    return runs


def arm_pressed(tmp_path, monkeypatch, server, budget=4 * 3600.0):
    """Fit pressed by a run of the driver's own (``arms.Driven.fit``) on ``server``, with ``budget``
    seconds of the run's time budget left; what it said, and the run's folder."""
    runs = wired(tmp_path, monkeypatch, server)
    session = arms.Driven(runs)
    session.run, session.budget_seconds = RID, budget
    session.until = server.clock.now + budget
    return session.fit(), runs / RID


def long_pressed(tmp_path, monkeypatch, server):
    """Fit pressed by the driver's own ``fit`` with a wait long enough for the fit (its cap,
    ``WAIT_MAX``, lifted for it): what it said, classed (``arms.said_of``), and the run's folder."""
    from research.swarm import driver

    runs = wired(tmp_path, monkeypatch, server)
    monkeypatch.setattr(driver, "WAIT_MAX", 1e9)
    out: list[str] = []
    code = driver.main(["fit", "--run", RID, "--wait", "1e8"], out=out.append)
    return arms.said_of(code, json.loads(out[-1])), runs / RID


def kept_beside(folder):
    """What the driver keeps of a press beside the run: the ways forward kept for ``answer``
    (``exits.json``), and its last markers' snapshot but its time (``markers.jsonl``)."""
    exits = json.loads((folder / "exits.json").read_text("utf-8"))
    snaps = [json.loads(s) for s in (folder / "markers.jsonl").read_text("utf-8").splitlines()]
    return exits, {k: v for k, v in snaps[-1].items() if k != "at"}


def told(said):
    """What a press says, classed (``arms.said_fields``), and what it printed."""
    return arms.said_fields(said), said.payload


def logged(folder):
    """The run's commands log (``commands.jsonl``), each line."""
    return [json.loads(s) for s in (folder / "commands.jsonl").read_text("utf-8").splitlines()]


def surfaced(folder):
    """Each command's name in the run's commands log, and what the record reads of it
    (``assemble.surfacings_of``: each refusal the run met, but when it came, and each fit that
    failed)."""
    from types import SimpleNamespace

    from research.swarm import assemble

    commands = logged(folder)
    got = assemble.surfacings_of(commands, SimpleNamespace(body=None))
    return ([c["command"] for c in commands],
            [{k: v for k, v in r.items() if k != "at"} for r in got["refusals"]],
            got["failed_fits"])


@pytest.mark.parametrize("purpose", ["inference", "prediction"])
@pytest.mark.parametrize("shelf_at, seconds", [(0.0, ESTIMATE), (100.0, None)],
                         ids=["shelf_measured", "shelf_measuring"])
def test_a_fit_waited_out_leaves_the_one_press_a_press_that_waited_long_enough_leaves(
        tmp_path, monkeypatch, purpose, shelf_at, seconds):
    """P1: the fit takes 600 s once the shelf has measured it, so the arm's press comes back with
    it still computing (at 240 s), and the arm waits on with reads alone. Fit is pressed once, as
    by the driver's own ``fit`` whose wait outlasts the fit; the press kept and the fit record's
    press are that one press's, built here as ``press_fit`` builds it: the seq of the last of the
    28 records before it (under Estimate the lock comes after, at 29), and the shelf's estimate
    where the shelf had measured it at the press, else None; and the arm says what that driver's
    ``fit`` printed, the markers and the ways kept beside the run, and the press's line in the
    commands log, the same, with what the press printed when its wait ran out beside it."""
    waited = Server(Clock(), purpose=purpose, shelf_at=shelf_at)
    said, folder = arm_pressed(tmp_path / "arm", monkeypatch, waited)
    once = Server(Clock(), purpose=purpose, shelf_at=shelf_at)
    long, long_folder = long_pressed(tmp_path / "long", monkeypatch, once)
    press = {"target": "y", "seconds": seconds, "seq": RECORDS}
    assert [r["press"] for r in waited.records] == [r["press"] for r in once.records] == [press]
    assert waited.kept == once.kept == press
    assert waited.posts == once.posts == [0.0]
    assert said.ok and told(said) == told(long)
    assert kept_beside(folder) == kept_beside(long_folder)
    assert surfaced(folder) == surfaced(long_folder) == (["fit"], [], [])
    line, long_line = logged(folder)[-1], logged(long_folder)[-1]
    assert {k: v for k, v in line["summary"].items() if k != "ran_out"} == long_line["summary"]
    assert line["summary"]["ran_out"]["fit"]["status"] == (
        "running" if shelf_at < arms.WAIT else "idle")
    assert waited.clock.now == once.clock.now == shelf_at + 600.0  # each waited the fit out


@pytest.mark.parametrize("end, kind, code", [
    ("refused_with_a_way", "refused", "too_few_rows"), ("failed", "failed", "fit_failed"),
    ("blocked", "refused", "fit_not_reached"), ("held", "refused", "fit_not_reached"),
], ids=["refused_in_words", "failed", "blocked", "held_for_a_press"])
@pytest.mark.parametrize("purpose", ["inference", "prediction"])
def test_a_fit_that_ends_otherwise_after_the_wait_ran_out_is_said_as_a_long_enough_press_says(
        tmp_path, monkeypatch, purpose, end, kind, code):
    """Both ways: the fit ends otherwise than fresh after the press's wait ran out (refused in
    words with a way forward, failed, blocked, or, its shelf measuring until 300 s, held for a
    press then). The arm says what the driver's own ``fit`` whose wait outlasts it says, with the
    refusal's ways forward kept for ``answer`` as it keeps them, and the record reads the same of
    the commands log: the refusal met, or the fit that failed; Fit is pressed once, the press
    kept the same."""
    shelf_at, seconds = (300.0, 0.0) if end == "held" else (0.0, 600.0)
    waited = Server(Clock(), purpose=purpose, seconds=seconds, end=end, shelf_at=shelf_at)
    said, folder = arm_pressed(tmp_path / "arm", monkeypatch, waited)
    once = Server(Clock(), purpose=purpose, seconds=seconds, end=end, shelf_at=shelf_at)
    long, long_folder = long_pressed(tmp_path / "long", monkeypatch, once)
    assert (said.kind, said.code) == (kind, code) and told(said) == told(long)
    assert kept_beside(folder) == kept_beside(long_folder)
    assert surfaced(folder) == surfaced(long_folder)
    commands, refusals, failed = surfaced(folder)
    assert (commands, [r["code"] for r in refusals], [f["error"] for f in failed]) == (
        ["fit"], ["too_few_rows"] if kind == "refused" and code == "too_few_rows" else [],
        [ENDS[end]["error"]] if ENDS[end]["status"] == "error" else [])
    assert waited.posts == once.posts == [0.0] and waited.kept == once.kept
    assert waited.records == once.records == []
    if end == "refused_with_a_way":
        assert [w["label"] for w in said.exits] == [WAY["label"]] and said.exits[0]["id"]


NOT_FRESH = (1, {"refused": {"by": "server", "status": 409, "code": "fit_not_fresh",
                             "message": "The held-out rows open on a current fit.", "exits": []}})


class Modeled:
    """The driver's answers to an arm's commands (``driver.main``'s exit status and JSON), modeled
    here but for Fit, which is the driver's own ``fit`` on a modeled :class:`Server`
    (:func:`wired`): the start takes ``start`` seconds of the clock; every ask finds Fit reached
    with nothing open; the seal answers the next of ``seals`` (recorded, once they run out); each
    of ``then`` (a command's name, a call) is called once after that command; an answer is
    recorded, and the results, the export and the stop are given. Each command is kept
    (``argv``)."""

    VIEW = {"state": {"purpose": "prediction", "models": ["logistic"]}, "stages": {}}

    def __init__(self, clock, seals=(), start=0.0, then=None):
        self.clock, self.seals, self.start = clock, list(seals), start
        self.then = dict(then or {})
        self.argv: list[list[str]] = []

    def __call__(self, *argv):  # set on the class, it is not bound: no session comes first
        from research.swarm import driver

        self.argv.append(list(argv))
        command = argv[0]
        if command == "start":
            self.clock.sleep(self.start)
            said = 0, {"started": {"run": RID}}
        elif command == "ask":
            said = 0, {"question": None, "fit_reached": True, "options": [], "also_open": []}
        elif command == "fit":
            out: list[str] = []
            said = driver.main(list(argv), out=out.append), json.loads(out[-1])
        elif command == "seal" and self.seals:
            said = self.seals.pop(0)
        else:
            said = 0, {"seal": {"recorded": {"id": "r9", "seq": 9,
                                             "decision": {"kind": "open_seal"}}},
                       "answer": {"recorded": {"id": "r30", "seq": RECORDS + 2,
                                               "decision": WAY["decision"]}},
                       "results": {"results": {}}, "export": {"exported": {"path": "export.zip"}},
                       "stop": {"stopped": {"run": RID}}}[command]
        if command in self.then:
            self.then.pop(command)()
        return said

    def waits(self):
        """The wait each press of Fit was given."""
        return [a[a.index("--wait") + 1] for a in self.argv if a[0] == "fit"]


def armed(tmp_path, monkeypatch, path, server, *, seals=(), start=0.0, then=None,
          budget_seconds=4 * 3600.0):
    """``path`` (the default arm, or the random at seed 3) run by ``arms.run_arm`` on the Predict
    card ``clinical_readmission_predict`` through the driver modeled here (:class:`Modeled`, its
    Fit the driver's own on ``server``), its record assembled by a stub; what the arm returned,
    and the model."""
    from research.swarm import assemble

    wired(tmp_path, monkeypatch, server)
    model = Modeled(server.clock, seals, start, then)
    monkeypatch.setattr(arms.Driven, "command", model)
    monkeypatch.setattr(arms.Driven, "read", lambda self, p: Modeled.VIEW if p is None else None)
    monkeypatch.setattr(assemble, "assemble",
                        lambda folder, meta, corpus: (Path(corpus) / "record.json", True))
    out = arms.run_arm("clinical_readmission_predict", path, seed=3 if path == "random" else None,
                       runs=tmp_path / "runs", corpus=tmp_path / "corpus",
                       budget_seconds=budget_seconds, data=tmp_path / "t.csv", meta={})
    return out, model


ARMS = ["default", "random"]


@pytest.mark.parametrize("path", ARMS)
def test_a_fit_still_running_when_its_press_comes_back_is_waited_out_to_the_export(
        tmp_path, monkeypatch, path):
    """W1: the fit takes 1000 s, so the press of Fit comes back with it still running (its wait
    ran out, as on the readmission fixture on 2026-10-29); the arm waits on with reads alone and
    goes on to the seal and the export, with the commands, the one press, the press kept and the
    fit record's press of the arm whose fit takes 10 s."""
    slow = Server(Clock(), seconds=1000.0)
    out, model = armed(tmp_path / "slow", monkeypatch, path, slow)
    assert out["error"] is None and out["valid"] is True
    assert slow.clock.now >= 1000.0
    quick = Server(Clock(), seconds=10.0)
    at_once, once = armed(tmp_path / "quick", monkeypatch, path, quick)
    assert at_once["error"] is None and at_once["steps"] == out["steps"]
    assert model.argv[1:] == once.argv[1:]  # after the start, whose table is each run's own
    assert [a[0] for a in model.argv][-2:] == ["export", "stop"]
    assert "seal" in [a[0] for a in model.argv]
    assert slow.posts == quick.posts == [0.0] and slow.kept == quick.kept
    assert [r["press"] for r in slow.records] == [r["press"] for r in quick.records]


@pytest.mark.parametrize("server, still", [
    (lambda: Server(Clock(), end="never"), "running"),
    (lambda: Server(Clock(), shelf_at=float("inf")), "idle, held for estimate,"),
], ids=["running", "held_while_the_shelf_measures"])
@pytest.mark.parametrize("path", ARMS)
def test_a_fit_that_outlasts_the_runs_time_budget_stops_the_arm_saying_how_long_it_waited(
        tmp_path, monkeypatch, path, server, still):
    """W1, the other way: the fit is never done. The run is given 1000 s from its start, and its
    start takes 100 s, so Fit is pressed 100 s in, once, and its fit waited for to the budget, 900
    s after the press; the arm stops there, in its own words; nothing after Fit is given, and the
    record is still assembled."""
    budget, start = 1000.0, 100.0
    s = server()
    out, model = armed(tmp_path, monkeypatch, path, s, start=start, budget_seconds=budget)
    assert model.waits() == [format(arms.WAIT, "g")] and s.posts == [start]
    assert out["error"] == (f"The fit outlasted the run's time budget of {budget:g} s from its "
                            f"start: it was still {still} {budget - start:g} s after Fit was "
                            f"pressed.")
    assert s.clock.now == budget
    assert [a[0] for a in model.argv][-2:] == ["fit", "stop"] and out["valid"] is True


def test_a_budget_spent_before_fit_is_pressed_leaves_the_press_its_own_wait_and_no_more(
        tmp_path, monkeypatch):
    """The press waits its whole wait, as before, whatever is left of the budget; where the budget
    is spent by then (the start took all 100 s of it), the fit is read once more, and the arm says
    so, :data:`arms.WAIT` seconds after the press."""
    s = Server(Clock(), end="never")
    out, model = armed(tmp_path, monkeypatch, "default", s, start=100.0, budget_seconds=100.0)
    assert model.waits() == [format(arms.WAIT, "g")] and s.posts == [100.0]
    assert out["error"] == (f"The fit outlasted the run's time budget of 100 s from its start: "
                            f"it was still running {arms.WAIT:g} s after Fit was pressed.")


@pytest.mark.parametrize("end, said", [
    ("blocked", "fit_not_reached: The fit is blocked."),
    ("held", "fit_not_reached: The fit is idle, held for fit."),
    ("refused", f"too_few_rows: {WORDS}"),
    ("failed", "fit_failed: ValueError: no rows"),
], ids=["blocked", "held_for_a_press", "refused_by_the_engine", "failed"])
@pytest.mark.parametrize("path", ARMS)
def test_a_fit_blocked_held_refused_or_failed_stops_the_arm_at_its_one_press_as_before(
        tmp_path, monkeypatch, path, end, said):
    """Both ways: a press that comes back for any reason but its wait running out (the fit
    blocked, held for a press of Fit, refused by the engine with no way forward, or failed) is
    the answer, nothing waited, and the arm stops on it in the words it stopped in before."""
    s = Server(Clock(), seconds=0.0, end=end)
    out, model = armed(tmp_path, monkeypatch, path, s)
    assert out["error"] == f"Fit was not made: {said}"
    assert model.waits() == [format(arms.WAIT, "g")] and s.posts == [0.0]
    assert s.clock.now == 0.0


@pytest.mark.parametrize("path", ARMS)
def test_the_fit_the_seal_waits_for_is_waited_out_as_the_first_is(tmp_path, monkeypatch, path):
    """The seal refused for a fit that is not current (``fit_not_fresh``; answers that changed the
    fit's key, :meth:`Server.refit`, to a fit of 1000 s): the press it waits for comes back with
    that fit still running, the arm waits it out with reads alone, and the seal opens and the
    export is given (``arms.seal_through``), each fit recorded once."""
    s = Server(Clock(), seconds=10.0)
    out, model = armed(tmp_path, monkeypatch, path, s, seals=[NOT_FRESH],
                       then={"seal": lambda: s.refit(1000.0)})
    assert out["error"] is None
    commands = [a[0] for a in model.argv]
    seal = commands.index("seal")
    assert commands[seal:] == ["seal", "fit", "seal", "results", "export", "stop"]
    assert len(s.posts) == 2 and [r["fit_key"] for r in s.records] == ["k1", "k2"]


# ── a press of Fit the open noticings' triage holds (under Estimate) ────────

QE = arms.Question(target="sbp_mmhg", purpose="inference", exposure="walks_daily",
                   meaning="systolic blood pressure (mmHg)")
# The way forward "Keep the recommended dispositions" records, written out by hand: the triage's
# Confirm all at the gate before the lock (the server fills in each noticing's disposition).
KEEP = {"kind": "confirm_sweep", "stage": "models", "sweep": "noticings", "lines": []}


def triage_held(tmp_path, *, blocking):
    """A press of Fit as the driver says the server's refusal of it while the open noticings'
    triage is not confirmed (``fit_press.triage_refusal`` over a triage written here, one noticing
    open, blocking or not; served as the server serves a refusal, ``Refusal.to_dict``; read by
    ``driver.refused``, each way forward with its id), classed (``arms.said_of``)."""
    from research.swarm import driver
    from turbotab.core import fit_press
    from turbotab.core.sweep import Triage, TriageItem

    item = TriageItem(id="shared-collinear-predictors", line="noticing:shared-collinear-predictors",
                      stage="models", summary="`age` and `height_cm` move almost in step",
                      severity="noticing", recommended="no_change",
                      label="Doesn't change your numbers here", reason="the reason",
                      blocker=blocking)
    refusal = fit_press.triage_refusal(Triage(stage="models", gate="gate", items=[item],
                                              blockers=int(blocking), confirmable=not blocking))
    served = driver.Response(409, json.dumps(refusal.to_dict()).encode("utf-8"), {})
    run = driver.Run(tmp_path / "run", {"project": PID})
    run.folder.mkdir(parents=True, exist_ok=True)
    return arms.said_of(1, driver.refused(served, run, "fit"))


class Presses:
    """A session whose presses of Fit say each of ``said`` in turn, and whose answers are each
    recorded; every press and answer kept."""

    def __init__(self, *said):
        self.left, self.presses, self.answers = list(said), 0, []

    def fit(self):
        self.presses += 1
        return self.left.pop(0)

    def answer(self, ids, sets, reason, marker=arms.DEFAULT_MARKER):
        self.answers.append((list(ids), list(sets), reason))
        return arms.Said("ok", {"answer": {"recorded": {"decision": {"kind": "confirm_sweep"}}}})


FITTED = arms.Said("ok", {"pressed": {"locked": True}, "fit": {"status": "fresh"}})


def policy_of(path):
    return arms.Default(QE) if path == "default" else arms.Random(QE, 3)


@pytest.mark.parametrize("path", ARMS)
def test_a_press_the_open_noticings_triage_holds_keeps_the_recommended_and_presses_again(
        tmp_path, path):
    """V2 (the mechanical arm stopped at Fit): under Estimate the server refuses a press of Fit
    while the open noticings' triage is unconfirmed; the arm takes the refusal's way forward "Keep
    the recommended dispositions" (its ``confirm_sweep``, by the id the driver gave it), as a
    person pressing Confirm all does, whichever path it walks, and presses Fit again: that press
    is the answer."""
    held = triage_held(tmp_path, blocking=False)
    assert (held.kind, held.by, held.code) == ("refused", "server", "triage_open")
    (way,) = [w for w in held.exits if w.get("decision")]
    assert (way["label"], way["decision"]) == ("Keep the recommended dispositions", KEEP)
    assert way["id"]
    session, steps = Presses(held, FITTED), []
    said = arms.fit_through(session, policy_of(path), steps)
    assert said is FITTED and session.presses == 2
    name = policy_of(path).name
    assert session.answers == [
        ([way["id"]], [], f"{name}: the fit's way forward, Keep the recommended dispositions.")]
    assert [(s.at, s.key, s.chosen, s.said, s.code) for s in steps] == [
        ("fit", "fit", [way["id"]], "refused", "triage_open")]


@pytest.mark.parametrize("path", ARMS)
def test_a_press_a_blocking_noticing_or_another_refusal_holds_is_the_answer_unchanged(
        tmp_path, path):
    """Both ways: a blocking noticing's refusal offers no way that records the triage (each is
    decided on its own), and a server refusal of another kind, even one with a way that records
    something (``fit_not_yet``), is not the triage's: either is the answer as it came, Fit pressed
    once and nothing answered."""
    from research.swarm import driver

    blocked = triage_held(tmp_path / "blocked", blocking=True)
    assert (blocked.kind, blocked.by, blocked.code) == ("refused", "server", "triage_open")
    assert [w.get("decision") for w in blocked.exits] == [None]
    other = arms.said_of(1, {"refused": {
        "by": "server", "status": 409, "code": "fit_not_yet", "message": "Fit waits.",
        "exits": [{"id": "x-1", "label": "Keep every row",
                   "decision": {"kind": "set_exclusions", "rules": []}}]}})
    for held in (blocked, other):
        session, steps = Presses(held), []
        assert arms.fit_through(session, policy_of(path), steps) is held
        assert session.presses == 1 and session.answers == [] and steps == []
    assert driver.Run(tmp_path / "blocked" / "run", {}).kept_exits() == blocked.exits


def test_under_estimate_a_local_press_the_triage_holds_records_the_triage_and_then_the_lock(
        tmp_path, monkeypatch):
    """V2 (the crash sweep stopped at the plan's own refit): a :class:`arms.Local` press of Fit,
    as the sweep presses each refit's (``Sweep._fit``), on the walking table under Estimate, walked
    by the default path. The premise as the server serves it: one noticing open, none blocking.
    Made to block (the triage served with that noticing a blocker, written here), the press is the
    server's refusal unchanged, no way forward taken and nothing recorded; as served, the press
    takes "Keep the recommended dispositions" and Fit is pressed again: the triage recorded, each
    noticing at its recommended disposition, and then the lock, the fit fresh."""
    from research.swarm.tests.test_sweep_engine import walking_table
    from turbotab.server.service import ProjectService

    walking_table(tmp_path / "walking.csv")
    with arms.Local(tmp_path / "home") as session:
        session.open(tmp_path / "walking.csv")
        arms.walk(session, arms.Default(QE))
        t = session.client.get(f"/projects/{session.pid}/triage").json()
        open_items = [i for i in t["items"] if i["recorded"] is None]
        assert t["stage"] == "models" and t["blockers"] == 0 and open_items
        before = [r["decision"]["kind"] for r in session.view()["decisions"]]
        assert "lock_plan" not in before and "confirm_sweep" not in before

        real = ProjectService.triage

        def blocking(self, pid):
            served = real(self, pid)
            return served.model_copy(update={
                "items": [i.model_copy(update={"blocker": True}) for i in served.items],
                "blockers": len(served.items), "confirmable": False})

        monkeypatch.setattr(ProjectService, "triage", blocking)
        raw = session.client.post(f"/projects/{session.pid}/fit")  # the server's own answer
        assert raw.status == 409 and raw.error()["code"] == "triage_open"
        held = session.fit(600.0)
        assert (held.kind, held.by, held.code) == ("refused", "server", "triage_open")
        assert held.message == raw.error()["message"]
        assert [{k: w.get(k) for k in ("label", "decision")} for w in held.exits] == \
            raw.error()["exits"] == [{"label": f"Decide: {i['summary']}", "decision": None}
                                     for i in open_items]
        assert [r["decision"]["kind"] for r in session.view()["decisions"]] == before
        monkeypatch.undo()

        said = session.fit(600.0)
        assert said.ok, (said.kind, said.code, said.message)
        assert said.payload["fit"]["status"] == "fresh"
        view = session.view()
        records = view["decisions"]
        assert [r["decision"]["kind"] for r in records] == [*before, "confirm_sweep", "lock_plan"]
        confirm = records[-2]["decision"]
        assert {k: confirm[k] for k in ("stage", "sweep")} == {"stage": "models",
                                                              "sweep": "noticings"}
        assert {line["key"]: line["value"] for line in confirm["lines"]} == {
            i["id"]: i["recommended"] for i in open_items}
        assert view["state"]["plan_locked"] is True
        assert records[-1]["decision"]["plan"]["triage"] == confirm["lines"]

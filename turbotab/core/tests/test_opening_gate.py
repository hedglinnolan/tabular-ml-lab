"""W1-H · the opening gate (WALKABLE_PLAN §3 P2): the held-out rows open once, so ``open_seal`` is
refused while the triage of open noticings (``turbotab/core/sweep.py``) holds a noticing with no
recorded disposition, or a blocker, with a plain reason and an exit to the triage; once the triage
is confirmed the opening goes through.

The states are built by hand: one open noticing about a column the model reads, under Predict
(where the triage's gate is the opening, ``sweep.gate_stage``), with rows held out and a fresh fit.
"""
from __future__ import annotations

import pytest

from turbotab.core import decisions, seal, sweep
from turbotab.core.decisions import ProjectState, Refusal, SplitSpec
from turbotab.core.tests.test_confirm_sweep import Ctx, log_of, planned
from turbotab.core.tests.test_stage_registry import record

FRESH = {"n_holdout": 120, "models": [{"family": "linear", "holdout": None}]}

OPEN_NOTICING = {"findings": [
    {"id": "heaping__age", "severity": "warning", "summary": "Ages heap on round numbers",
     "affected_columns": ["age"], "routes_to": None, "repairs": [{"key": "map"}],
     "answered_by": None},
]}
BLOCKER = {"findings": [
    *OPEN_NOTICING["findings"],
    {"id": "batch_confounded__run", "severity": "critical", "summary": "Run is the outcome",
     "affected_columns": ["run"], "routes_to": None, "repairs": [{"key": "x"}],
     "answered_by": None},
]}


def predicting(records=()) -> ProjectState:
    state = planned(purpose="prediction", split=SplitSpec(holdout=0.2, seed=3, folds=5))
    return state.model_copy(update={"sweeps": decisions.fold(list(records)).sweeps}) if records \
        else state


def triaged(findings, records=()):
    state = predicting(records)
    log = log_of(state, list(records), findings=findings)
    return state, log, sweep.triage(state, log, findings, list(records))


def opening_ctx(state, t):
    return {"state": state, "artifact": lambda stage: FRESH, "triage": lambda: t}


def test_the_opening_is_refused_while_a_noticing_is_open_and_says_where_to_triage_it():
    state, log, t = triaged(OPEN_NOTICING)
    assert t.stage == "results" and not t.answered and not t.passed  # the gate is the opening
    with pytest.raises(Refusal) as refused:
        seal._the_opening_waits_for_the_triage(decisions.OpenSeal(family="linear"),
                                               opening_ctx(state, t))
    r = refused.value
    assert r.code == "triage_open"
    assert "held-out rows open once" in r.message and "1 is still open" in r.message
    assert "Results" in r.message
    to_triage, keep = r.exits
    assert to_triage == {"label": "Triage the open noticings in Results", "decision": None}
    # The other way through is the triage's own "Confirm all", and it is accepted as it stands.
    assert keep["decision"] == {"kind": "confirm_sweep", "stage": "results", "sweep": "noticings",
                                "lines": []}
    taken = decisions.validate(keep["decision"], Ctx(log, t))
    assert {l.key: l.value for l in taken.lines} == {"heaping__age": "could_bias"}
    # Registered: validating an opening runs the gate after the fresh-fit checks.
    checks = decisions._VALIDATORS["open_seal"]
    assert checks.index(seal._the_opening_waits_for_the_triage) > \
        checks.index(seal._open_seal_once_on_a_fresh_fit)


def test_the_opening_is_accepted_once_the_triage_is_confirmed():
    state, log, t = triaged(OPEN_NOTICING)
    confirmed = decisions.validate({"kind": "confirm_sweep", "stage": "results",
                                    "sweep": "noticings"}, Ctx(log, t))
    records = [record(1, confirmed.model_dump(mode="json"))]
    state, _log, after = triaged(OPEN_NOTICING, records)
    assert after.answered and after.confirmed_by == "r1"
    seal._the_opening_waits_for_the_triage(decisions.OpenSeal(family="linear"),
                                           opening_ctx(state, after))
    seal._open_seal_once_on_a_fresh_fit(decisions.OpenSeal(family="linear"),
                                        opening_ctx(state, after))


def test_a_blocker_is_named_and_decided_on_its_own_before_the_opening():
    state, _log, t = triaged(BLOCKER)
    assert t.blockers == 1
    with pytest.raises(Refusal) as refused:
        seal._the_opening_waits_for_the_triage(decisions.OpenSeal(family="linear"),
                                               opening_ctx(state, t))
    assert refused.value.code == "triage_open"
    assert "Run is the outcome" in refused.value.message
    assert [e["label"] for e in refused.value.exits] == ["Decide: Run is the outcome"]


def test_nothing_open_or_no_triage_to_read_refuses_nothing():
    state, _log, t = triaged({"findings": []})
    assert not t.items and t.answered
    seal._the_opening_waits_for_the_triage(decisions.OpenSeal(family="linear"),
                                           opening_ctx(state, t))
    seal._the_opening_waits_for_the_triage(decisions.OpenSeal(family="linear"),
                                           {"state": state, "artifact": lambda s: FRESH})


# ── the repair round: the gate is no rubber stamp, names what it holds, and fails closed ──

TO_ACT_ON = {"findings": [
    {"id": "binary_text__age", "severity": "warning", "summary": "Age is written as text",
     "affected_columns": ["age"], "routes_to": None, "repairs": [{"key": "level"}],
     "answered_by": None},
]}


def _refusal(state, t) -> Refusal:
    with pytest.raises(Refusal) as refused:
        seal._the_opening_waits_for_the_triage(decisions.OpenSeal(family="linear"),
                                               opening_ctx(state, t))
    return refused.value


def _confirm(log, t, lines=()):
    body = {"kind": "confirm_sweep", "stage": "results", "sweep": "noticings",
            "lines": list(lines)}
    return decisions.validate(body, Ctx(log, t))


def test_the_refusal_names_each_open_noticing_with_its_recommended_disposition():
    state, _log, t = triaged({"findings": [*OPEN_NOTICING["findings"], *TO_ACT_ON["findings"]]})
    by_id = {i.id: i for i in t.items}
    # The dispositions by hand, from the triage's rules (``sweep.recommend``): a column the
    # model reads with no repair here could bias the score; one with a repair is acted on.
    assert by_id["heaping__age"].recommended == "could_bias"
    assert by_id["binary_text__age"].recommended == "act_on_it"
    r = _refusal(state, t)
    assert r.code == "triage_open" and "2 are still open" in r.message
    assert "Ages heap on round numbers (could bias the score)" in r.message
    assert "Age is written as text (act on it)" in r.message


def test_keeping_an_act_on_it_disposition_does_not_open_the_rows_until_it_is_settled():
    """Recording "act on it" is not acting on it: the one-click confirm records the dispositions,
    and the rows stay sealed while one marked to act on is still open, named, with its own way
    through (decide it, or record it as a limitation, which the triage lets the person choose)."""
    state, log, t = triaged(TO_ACT_ON)
    kept = _confirm(log, t)
    assert [(l.key, l.value) for l in kept.lines] == [("binary_text__age", "act_on_it")]
    records = [record(1, kept.model_dump(mode="json"))]
    state, log, after = triaged(TO_ACT_ON, records)
    assert after.answered  # every noticing has a recorded disposition: the old gate opened here
    r = _refusal(state, after)
    assert r.code == "triage_open"
    assert "1 marked \"act on it\" is not acted on yet: Age is written as text" in r.message
    decide, limit = r.exits
    assert decide == {"label": "Decide: Age is written as text", "decision": None}
    assert limit["label"] == "Record as a limitation: Age is written as text"
    # The exit is a disposition of that noticing alone, accepted as it stands.
    taken = _confirm(log, after, limit["decision"]["lines"])
    assert [(l.key, l.value) for l in taken.lines] == [("binary_text__age", "could_bias")]
    records.append(record(2, taken.model_dump(mode="json")))
    state, _log, settled = triaged(TO_ACT_ON, records)
    assert [i.recorded for i in settled.items] == ["could_bias"]
    seal._the_opening_waits_for_the_triage(decisions.OpenSeal(family="linear"),
                                           opening_ctx(state, settled))


def test_a_triage_that_cannot_be_read_keeps_the_rows_sealed():
    state, _log, _t = triaged(OPEN_NOTICING)

    def broken():
        raise RuntimeError("the materiality measure failed")

    with pytest.raises(Refusal) as refused:
        seal._the_opening_waits_for_the_triage(
            decisions.OpenSeal(family="linear"),
            {"state": state, "artifact": lambda stage: FRESH, "triage": broken})
    assert refused.value.code == "triage_unreadable"
    assert "stay sealed" in refused.value.message

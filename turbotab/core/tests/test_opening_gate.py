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

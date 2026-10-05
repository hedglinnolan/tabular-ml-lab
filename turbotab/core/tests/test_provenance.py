"""The seal's openings by outcome, the disclosure on each record, and the methods text built from
the log (audit WP16): pure, on a decision log alone. The server paths are in
``acceptance/test_wp16_after_opening_and_plan_lock.py``."""
from __future__ import annotations

from turbotab.core import voice
from turbotab.core.decisions import (DecisionLog, LockPlan, OpenSeal, Reseal, Revert,
                                     SetExclusions, SetPurpose, SetSplit, SetTarget, disclose)
from turbotab.core.provenance import methods_text


def _log(tmp_path) -> DecisionLog:
    return DecisionLog(tmp_path / "decisions.jsonl")


def _said(log: DecisionLog, decision) -> object:
    return log.append(decision, sentence=lambda d, before: voice.sentence_for(d, before, None))


def test_an_opening_stands_for_its_own_outcome_and_a_reseal_withdraws_it(tmp_path):
    log = _log(tmp_path)
    log.append(SetTarget(column="a"))
    log.append(SetSplit(holdout=0.2, seed=0))
    log.append(OpenSeal(family="linear", target="a"))
    assert log.state().seal_opened is True
    moved = log.append(SetTarget(column="b"))
    assert log.state().seal_opened is None  # a new outcome starts its own seal
    assert moved.post_seal is True  # ...made after held-out scores were seen
    log.append(SetTarget(column="a"))
    assert log.state().seal_opened is True  # back to the outcome whose seal was opened
    reseal = log.append(Reseal(target="a"))
    assert log.state().seal_opened is None and reseal.post_seal is True
    log.append(OpenSeal(family="linear", target="a"))
    assert log.state().seal_opened is True
    # An opening recorded before the rule names no outcome and stands for every one.
    old = _log(tmp_path / "old")
    old.append(SetTarget(column="a"))
    old.append(OpenSeal(family="linear"))
    old.append(SetTarget(column="b"))
    assert old.state().seal_opened is True


def test_the_disclosure_leads_each_sentence_once():
    text = "`177` rows were excluded."
    assert disclose(text, post_seal=False, after_estimates=False) == text
    assert disclose(text, post_seal=True, after_estimates=False) == (
        "After the held-out rows were opened, `177` rows were excluded.")
    assert disclose(text, post_seal=False, after_estimates=True) == (
        "After the estimates were seen, `177` rows were excluded.")
    both = disclose("Rows were excluded.", post_seal=True, after_estimates=True)
    assert both == "After the estimates were seen and the held-out rows were opened, rows were excluded."
    assert disclose(both, post_seal=True, after_estimates=True) == both
    assert disclose("NHANES weights were used.", post_seal=False, after_estimates=True).endswith(
        "NHANES weights were used.")


def test_the_methods_text_keeps_every_fork_after_the_estimates_were_seen(tmp_path):
    """Superseded before the lock: folded out. In force at the lock, then changed: kept as the plan
    declared. Changed after the lock, then withdrawn: kept, with the withdrawal, each marked."""
    log = _log(tmp_path)

    def rule(high: float) -> SetExclusions:
        return SetExclusions(rules=[{"column": "fiber_g", "high": high, "reason": "a plausible intake"}])

    _said(log, SetTarget(column="bmi"))
    _said(log, SetPurpose(purpose="inference"))
    tried = _said(log, rule(50))
    declared = _said(log, rule(60))
    lock = _said(log, LockPlan(plan={"exclusions": []}, digest="ab" * 32))
    forked = _said(log, rule(70))
    undone = _said(log, Revert(decision_id=forked.id))
    assert forked.after_estimates and undone.after_estimates and not declared.after_estimates
    assert forked.sentence.startswith("After the estimates were seen, ")

    text = methods_text(log.records())
    kept = [line.record_id for line in text.lines]
    assert tried.id not in kept  # superseded before anything was seen
    assert [declared.id, lock.id, forked.id, undone.id] == kept[-4:]
    in_force = {line.record_id: line.in_force for line in text.lines}
    assert in_force[declared.id] is True  # the revert brought it back
    assert in_force[forked.id] is False and in_force[undone.id] is False
    assert text.seen_from == lock.seq
    assert text.text.endswith(undone.sentence)

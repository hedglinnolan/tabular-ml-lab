"""The seal's openings by outcome, the disclosure on each record, and the methods text built from
the log (audit WP16): pure, on a decision log alone. The server paths are in
``acceptance/test_wp16_after_opening_and_plan_lock.py``."""
from __future__ import annotations

from turbotab.core import voice
from turbotab.core.decisions import (DecisionLog, LockPlan, OpenSeal, Reseal, Revert,
                                     SetExclusions, SetFeatureTable, SetOrientation, SetPurpose,
                                     SetSplit, SetTarget, disclose)
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


def test_the_plan_locks_on_an_estimate_in_view_and_never_on_a_refusal_or_a_withheld_table():
    """What locks the inference plan (WP16) once WP17 withholds estimates: a table with
    coefficients, an exposure's test or an unrefused inference table; never a refusal with nothing
    estimated, nor a fit WP17 withheld (``estimand.withhold``); and the "further adjusted for"
    model WP17 adds is an estimate like the sensitivity analysis's."""
    from turbotab.core.estimand import withhold
    from turbotab.core.plan_lock import ESTIMATE_STAGES, shows_estimates

    shown = {"feature": "fiber_g", "estimate": -0.2}
    fit = {"models": [{"coefficients": [shown], "inference": {"covariance": "HC3"},
                       "exposure_tests": []}]}
    assert shows_estimates("fit", fit)
    refused = {"models": [{"coefficients": None, "exposure_tests": [],
                           "inference": {"covariance": "none", "refused": "Answer the survey."}}]}
    assert not shows_estimates("fit", refused)
    assert not shows_estimates("fit", withhold("fit", fit, {"reason": "Which exposure?"}))
    trend = {"models": [{"coefficients": None, "inference": None,
                         "exposure_tests": [{"column": "fiber_g", "p": 0.01}]}]}
    assert shows_estimates("fit", trend)
    assert "secondary" in ESTIMATE_STAGES
    secondary = {"families": [{"fits": [{"coefficients": [shown], "inference": None}]}]}
    assert shows_estimates("secondary", secondary)
    assert not shows_estimates("secondary", withhold("secondary", secondary, {"reason": "…"}))


def _feature_lines(log: DecisionLog) -> list:
    return [line for line in methods_text(log.records()).lines if line.kind == "set_feature_table"]


def test_a_feature_table_reading_is_in_force_only_while_the_table_is_turned(tmp_path):
    """FT1: the features-in-rows reading (a turn check's exit) says how a turned table reads, and
    the oriented stage turns the table only on `feature_major` (``stages.working.oriented_stage``).
    Recorded on a table no answer turned, or one answered as one row per sample, it is not in
    force: no methods sentence, no sentence of the plan, none of a fit's records in force (``runs``
    reads them from ``in_force``). Turned, it is; a later "one row per sample" folds it out again."""
    from turbotab.core.plan_lock import plan_document
    from turbotab.core.provenance import in_force

    log = _log(tmp_path)
    reading = _said(log, SetFeatureTable(label="feature_id", annotations=["mz"]))
    assert reading.sentence.startswith("For the features-in-rows table, ")

    def said() -> tuple[bool, bool, bool]:
        records = log.records()
        return (reading.id in in_force(records), any(line.record_id == reading.id
                                                     for line in methods_text(records).lines),
                reading.sentence in plan_document(records).sentences)

    assert said() == (False, False, False)  # never turned
    _said(log, SetOrientation(orientation="sample_major"))
    assert said() == (False, False, False)  # rows are samples
    _said(log, SetOrientation(orientation="feature_major"))
    assert said() == (True, True, True)  # turned: the reading is how the table reads
    assert [line.sentence for line in _feature_lines(log)] == [reading.sentence]
    _said(log, SetOrientation(orientation="sample_major"))
    assert said() == (False, False, False)  # turned back: folded out again


def test_after_the_lock_a_feature_table_reading_on_an_unturned_table_changes_nothing(tmp_path):
    """After the lock nothing is folded out, but a features-in-rows reading recorded while no answer
    turns the table, and turning none since, changed nothing the analysis is (the oriented stage
    reads it only to turn the table), so it is no fork and writes no sentence, nor does its revert
    (``provenance.never_applied``). Recorded while the table is turned, it is a change made after
    the estimates were seen, and kept so; one the table is turned for later is in force."""
    def locked(orientation: str, at: str) -> DecisionLog:
        log = _log(tmp_path / at)
        _said(log, SetTarget(column="bmi"))
        _said(log, SetPurpose(purpose="inference"))
        _said(log, SetOrientation(orientation=orientation))
        _said(log, LockPlan(plan={}, digest="ab" * 32))
        return log

    plain = locked("sample_major", "plain")
    reading = _said(plain, SetFeatureTable(label="feature_id"))
    assert reading.after_estimates and _feature_lines(plain) == []
    undone = _said(plain, Revert(decision_id=reading.id))
    kept = [line.record_id for line in methods_text(plain.records()).lines]
    assert reading.id not in kept and undone.id not in kept

    turned = locked("feature_major", "turned")
    forked = _said(turned, SetFeatureTable(label="feature_id"))
    assert [(line.record_id, line.in_force)
            for line in _feature_lines(turned)] == [(forked.id, True)]
    assert forked.sentence.startswith("After the estimates were seen, for the features-in-rows")

    later = locked("sample_major", "later")
    first = _said(later, SetFeatureTable(label="feature_id"))
    _said(later, SetOrientation(orientation="feature_major"))
    assert [(line.record_id, line.in_force) for line in _feature_lines(later)] == [(first.id, True)]

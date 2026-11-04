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


def _locked(path, orientation: str, *before) -> DecisionLog:
    """A log under inference with the table answered ``orientation``, ``before`` recorded, and the
    plan then locked."""
    log = _log(path)
    _said(log, SetTarget(column="bmi"))
    _said(log, SetPurpose(purpose="inference"))
    _said(log, SetOrientation(orientation=orientation))
    for decision in before:
        _said(log, decision)
    _said(log, LockPlan(plan={}, digest="ab" * 32))
    return log


AFTER = "After the estimates were seen, for the features-in-rows table, "


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
    (``provenance.never_read``). Recorded while the table is turned, it is a change made after
    the estimates were seen, and kept so; one the table is turned for later is in force."""
    def locked(orientation: str, at: str) -> DecisionLog:
        return _locked(tmp_path / at, orientation)

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


def test_after_the_lock_a_feature_table_reading_read_at_any_moment_is_a_fork(tmp_path):
    """A features-in-rows reading recorded after the lock changed the analysis when the table was
    turned at some moment while it was the reading in force, whatever the orientation when it was
    recorded or now (``provenance.never_read``): recorded on a turned table and then answered back
    to one row per sample, or recorded unturned and turned and turned back, it is a change made
    after the estimates were seen, kept so and not in force, and so is its revert. One superseded
    before any answer turned the table was read at no moment, so it writes no sentence beside the
    reading that was. Truth: the orientation answers each log is built with (the oriented stage
    turns the table only on `feature_major`, ``stages.working.oriented_stage``)."""
    def said(log: DecisionLog) -> list[tuple[str, bool, bool]]:
        return [(line.record_id, line.in_force, line.sentence.startswith(AFTER))
                for line in _feature_lines(log)]

    then = _locked(tmp_path / "then", "feature_major")
    read_then = _said(then, SetFeatureTable(label="feature_id"))
    _said(then, SetOrientation(orientation="sample_major"))
    assert said(then) == [(read_then.id, False, True)]

    turned_first = _locked(tmp_path / "turned_first", "sample_major")
    _said(turned_first, SetOrientation(orientation="feature_major"))
    read_turned = _said(turned_first, SetFeatureTable(label="feature_id"))
    _said(turned_first, SetOrientation(orientation="sample_major"))
    assert said(turned_first) == [(read_turned.id, False, True)]

    between = _locked(tmp_path / "between", "sample_major")
    read_between = _said(between, SetFeatureTable(label="feature_id"))
    _said(between, SetOrientation(orientation="feature_major"))  # the table turns with it
    _said(between, SetOrientation(orientation="sample_major"))
    assert said(between) == [(read_between.id, False, True)]

    superseded = _locked(tmp_path / "superseded", "sample_major")
    _said(superseded, SetFeatureTable(label="feature_id"))  # replaced before any turn
    read_last = _said(superseded, SetFeatureTable(label="other"))
    _said(superseded, SetOrientation(orientation="feature_major"))
    assert said(superseded) == [(read_last.id, True, True)]

    withdrawn = _locked(tmp_path / "withdrawn", "feature_major")
    read_withdrawn = _said(withdrawn, SetFeatureTable(label="feature_id"))
    undone = _said(withdrawn, Revert(decision_id=read_withdrawn.id))
    assert undone.sentence and said(withdrawn) == [(read_withdrawn.id, False, True)]
    assert undone.id in [line.record_id for line in methods_text(withdrawn.records()).lines]


MADE = "made after the estimates were seen, listed with it"
NOW = "nothing in the current analysis."


def _plan(log: DecisionLog) -> tuple[list[int], list[int], str]:
    """The seqs the exported plan lists as made after the estimates were seen, the seqs of the
    methods' lines so marked, and the plan's text."""
    from turbotab.core.plan_lock import plan_document

    records = log.records()
    document = plan_document(records)
    return ([entry["seq"] for entry in document.after_estimates],
            [line.seq for line in methods_text(records).lines if line.after_estimates],
            document.text)


def test_the_plan_export_counts_every_later_decision_and_names_those_the_methods_leave_out(
        tmp_path):
    """V2-1: the exported plan lists and counts every decision recorded after the estimates were
    seen, as the Record holds them, and says which the methods state (``plan_lock.plan_text``): a
    features-in-rows reading read at no moment (``provenance.never_read``), or its revert, is
    listed, counted and named as left out of the methods, never "no decision since". Only a
    decision the methods state: counted and "marked so in the methods", as before; none at all: "no
    decision". Truth: the decisions each log is built with after its lock, and their orientation
    answers (the oriented stage turns the table only on `feature_major`)."""
    listed, stated, text = _plan(_locked(tmp_path / "none", "sample_major"))
    assert listed == stated == []
    assert text.endswith(" No decision has been made since the estimates were seen.")

    plain = _locked(tmp_path / "plain", "sample_major")
    reading = _said(plain, SetFeatureTable(label="feature_id"))
    listed, stated, text = _plan(plain)
    assert (listed, stated) == ([reading.seq], [])
    assert text.endswith(f" 1 decision was {MADE}; decision #{reading.seq} is left out of the "
                         f"methods, as it changes {NOW}")
    undone = _said(plain, Revert(decision_id=reading.id))
    listed, stated, text = _plan(plain)
    assert (listed, stated) == ([reading.seq, undone.seq], [])
    assert text.endswith(f" 2 decisions were {MADE}; decisions #{reading.seq} and #{undone.seq} "
                         f"are left out of the methods, as they change {NOW}")

    # An unread reading beside a decision the methods state: both counted, the reading named.
    mixed = _locked(tmp_path / "mixed", "sample_major")
    unread = _said(mixed, SetFeatureTable(label="feature_id"))
    rule = _said(mixed, SetExclusions(rules=[{"column": "fiber_g", "high": 70,
                                               "reason": "a plausible intake"}]))
    listed, stated, text = _plan(mixed)
    assert (listed, stated) == ([unread.seq, rule.seq], [rule.seq])
    assert text.endswith(f" 2 decisions were {MADE}; 1 of them is marked so in the methods, and "
                         f"decision #{unread.seq} is left out of them, as it changes {NOW}")

    # Silent: every later decision is in the methods.
    turned = _locked(tmp_path / "turned", "feature_major")
    forked = _said(turned, SetFeatureTable(label="feature_id"))
    listed, stated, text = _plan(turned)
    assert listed == stated == [forked.seq]
    assert text.endswith(f" 1 decision was {MADE} and marked so in the methods.")

    superseded = _locked(tmp_path / "superseded", "sample_major")
    first = _said(superseded, SetFeatureTable(label="feature_id"))  # replaced before any turn
    read = _said(superseded, SetFeatureTable(label="other"))
    turn = _said(superseded, SetOrientation(orientation="feature_major"))
    listed, stated, text = _plan(superseded)
    assert (listed, stated) == ([first.seq, read.seq, turn.seq], [read.seq, turn.seq])
    assert text.endswith(f" 3 decisions were {MADE}; 2 of them are marked so in the methods, and "
                         f"decision #{first.seq} is left out of them, as it changes {NOW}")


def test_a_reading_withdrawn_after_the_lock_is_a_change_only_if_it_was_read_from_the_lock_on(
        tmp_path):
    """V2-2: a features-in-rows reading recorded before the lock and reverted after it. In force
    at the lock, it is in the plan as declared, and its revert is a change made after the estimates
    were seen: said, listed and marked so. In force when it was recorded, but the table answered as
    one row per sample before the lock and since, the analysis read it at no moment from the lock
    on and the methods state it nowhere, so its revert writes no sentence; the plan lists and
    counts the revert and names it as left out. Truth: the orientation answers each log is built
    with (the oriented stage turns the table only on `feature_major`)."""
    def withdrawn(at: str, *before) -> tuple[DecisionLog, object, object]:
        log = _locked(tmp_path / at, "feature_major", SetFeatureTable(label="feature_id"), *before)
        reading = next(r for r in log.records() if r.decision.kind == "set_feature_table")
        return log, reading, _said(log, Revert(decision_id=reading.id))

    log, reading, undone = withdrawn("declared")
    lines = {line.record_id: line for line in methods_text(log.records()).lines}
    assert reading.id in lines and undone.id in lines and lines[undone.id].after_estimates
    listed, stated, text = _plan(log)
    assert listed == stated == [undone.seq]
    assert text.endswith(f" 1 decision was {MADE} and marked so in the methods.")

    log, reading, undone = withdrawn("orphan", SetOrientation(orientation="sample_major"))
    kept = [line.record_id for line in methods_text(log.records()).lines]
    assert reading.id not in kept and undone.id not in kept
    listed, stated, text = _plan(log)
    assert (listed, stated) == ([undone.seq], [])
    assert text.endswith(f" 1 decision was {MADE}; decision #{undone.seq} is left out of the "
                         f"methods, as it changes {NOW}")

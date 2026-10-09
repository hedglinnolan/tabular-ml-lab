"""ESTIMAND · 7 · the analysis-plan export (MODELING_SEQUENCE §1 row 12: "The lock is **never
described as 'prespecified' or 'preregistered'**: it says what was declared in the software before
estimates were displayed. The plan exports with a timestamp and a content hash for external
registration").

Source check (the review record, read 2026-10-05): Gelman & Loken 2013 — "we are often analyzing
public data … that have already been studied by others many times before, and it would be close to
meaningless to consider preregistration for data with which we are already so familiar." So the
export says what was declared and when, and that it is no registration.

The decision log is a real one on disk (``decisions.DecisionLog``), each record's sentence the
record's own (``voice``). The checks are independent of the export's code: the hash is recomputed
here with Python's ``json`` and ``hashlib`` under the canonical form the export states (keys
sorted, no spaces, UTF-8); "replay" re-reads the log from disk, and round-trips every record through
its JSON form.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from turbotab.core import decisions as d, plan_lock, voice
from turbotab.core.decisions import DecisionLog, DecisionRecord

FORBIDDEN = ("prespecified", "pre-specified", "preregistered", "pre-registered", "pre registered",
             "pre specified")


def _say(decision, state):
    return voice.sentence_for(decision, state, None)


def _log(tmp_path, *, lock: bool = True, after: bool = True) -> DecisionLog:
    log = DecisionLog(tmp_path / "decisions.jsonl")
    roles = {"pid": "identifier", "fiber": "exposure", "age": "covariate", "sex": "covariate",
             "kcal": "energy"}
    for decision in (
        d.SetLens(lenses=["dietary"]), d.SetTarget(column="dm"),
        d.SetTask(column="dm", task="binary"), d.SetEvent(column="dm", level="yes"),
        d.SetPurpose(purpose="inference"), d.SetRoles(roles=roles),
        d.SetEstimand(exposure="fiber", contrast="substitution", measure="risk_difference"),
        d.SetAdjustment(exposure="fiber", answers={c: d.CovariateAnswers(
            causes_exposure="yes", causes_outcome="yes", after_exposure="no")
            for c in ("age", "sex")}),
        d.SetModelSequence(exposure="fiber", model_1=["age", "sex", "kcal"]),
        d.SetEnergyAdjustment(method="standard", energy_column="kcal", nutrients=["fiber"]),
        d.SelectModels(models=["linear"]),
    ):
        log.append(decision, sentence=_say)
    if lock:
        log.append(d.validate({"kind": "lock_plan"}, {"state": log.state()}), sentence=_say)
    if after:
        log.append(d.SetModelSequence(exposure="fiber", model_1=["age", "sex"]), sentence=_say)
    return log


def test_7_the_export_is_timestamped_hashed_and_byte_identical_on_replay(tmp_path):
    log = _log(tmp_path)
    records = log.records()
    first = plan_lock.plan_export(records)
    assert plan_lock.plan_export(records) == first
    # replay: the log read again from disk, and every record through its JSON form
    assert plan_lock.plan_export(DecisionLog(tmp_path / "decisions.jsonl").records()) == first
    copied = [DecisionRecord.model_validate_json(r.model_dump_json()) for r in records]
    assert plan_lock.plan_export(list(reversed(copied))) == first
    doc = json.loads(first)
    # canonical JSON: keys sorted, no spaces, UTF-8
    assert json.dumps(doc, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8") == first
    # the content hash, recomputed here, over every field but the hash and the text that quotes it
    content = {k: v for k, v in doc.items() if k not in ("sha256", "text")}
    expected = hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":"),
                                         ensure_ascii=False).encode("utf-8")).hexdigest()
    assert doc["sha256"] == expected and expected in doc["text"]
    # the plan is the one the lock recorded, with the lock's own digest and time
    lock = next(r for r in records if r.decision.kind == "lock_plan")
    assert doc["status"] == "locked" and doc["plan"] == lock.decision.plan
    assert doc["plan_sha256"] == lock.decision.digest == hashlib.sha256(json.dumps(
        doc["plan"], sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    assert doc["declared_at"] == lock.at.isoformat().replace("+00:00", "Z")
    assert doc["through_record"] == lock.seq
    assert doc["plan"]["estimand"]["measure"] == "risk_difference"
    assert doc["plan"]["model_sequence"] == {"exposure": "fiber", "model_1": ["age", "sex", "kcal"]}
    assert set(doc["plan"]["adjustment"]) == {"age", "sex"}
    # what came after the estimates were seen is listed, as the record words it
    [later] = doc["after_estimates"]
    assert later["kind"] == "set_model_sequence"
    assert later["sentence"].startswith("After the estimates were seen, ")


def test_7_the_text_says_what_was_declared_and_never_that_it_was_registered(tmp_path):
    for lock in (True, False):
        log = _log(tmp_path / str(lock), lock=lock, after=lock)
        raw = plan_lock.plan_export(log.records()).decode("utf-8").lower()
        assert not any(word in raw for word in FORBIDDEN), lock
        doc = plan_lock.plan_document(log.records())
        assert "declared in TurboTab" in doc.text
        assert "it is not a registration with any outside registry" in doc.text
    locked = plan_lock.plan_document(_log(tmp_path / "again").records())
    assert locked.text.startswith("This is the analysis plan as declared in TurboTab before any "
                                  "estimate was displayed; it was locked on ")
    assert ", before the first estimate was shown." in locked.text
    assert locked.text.endswith("1 decision was made after the estimates were seen, listed with "
                                "it and marked so in the methods.")


def test_7_before_the_lock_the_plan_so_far_is_exported_and_any_change_changes_its_hash(tmp_path):
    log = _log(tmp_path, lock=False, after=False)
    records = log.records()
    doc = plan_lock.plan_document(records)
    assert doc.status == "declared" and doc.through_record == records[-1].seq
    assert doc.declared_at == records[-1].at.isoformat().replace("+00:00", "Z")
    assert doc.text.startswith("This is the analysis plan as declared in TurboTab so far, through "
                               "the decision recorded on ")
    log.append(d.SetModelSequence(exposure="fiber", model_1=["age"]), sentence=_say)
    changed = plan_lock.plan_document(log.records())
    assert changed.sha256 != doc.sha256 and changed.plan_sha256 != doc.plan_sha256
    assert changed.plan["model_sequence"]["model_1"] == ["age"]
    with pytest.raises(RuntimeError):
        plan_lock.plan_text({"status": "locked", "declared_at": "x"}, "prespecified")

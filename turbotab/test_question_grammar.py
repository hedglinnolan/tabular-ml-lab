"""§09 question grammar, on the steps that are built.

Three types, three moods, three silhouettes, and the register rule: every
distinction has to survive with typography removed (silhouette + grammar) and
with color removed (silhouette + signal word). No channel carries the
distinction alone, because every single channel fails — habituation,
color-blindness, skimming.

The blocker's costume, the skip's muted provenance row and the CHOICE card's
symmetric buttons are asserted in `test_guided_drive.py`, where they landed with
the drive batch that needed them. This file covers the FACT treatment and the
exclusivity rules that hold the whole grammar together.

`DESIGN_LANGUAGE.md` §09.
"""
# The tests here that drove the retired legacy app (turbotab/api.py, the page, the
# figure and manuscript modules, the gates) were removed with it, BLUEPRINT §9.1; git keeps them.
from __future__ import annotations

from pathlib import Path

import pytest

from ml import router

REPO_ROOT = Path(__file__).resolve().parent.parent


def data_plan(**kw):
    detection = kw.pop("detection", {"detected": "classification",
                                     "confidence": "medium",
                                     "reasons": ["12 distinct values."]})
    return router.plan([], target=kw.pop("target", "y"), detection=detection,
                       step="data", **kw)


# ═══════════════════════════════════════════════════════════════════════════
# FACT — a question we have the right to ask
# ═══════════════════════════════════════════════════════════════════════════

def test_every_pushed_fact_names_who_consumes_the_answer():
    """Clause: `assembly-04`"""
    for q in data_plan(answered=[]):
        if q.kind in router.FACT_KINDS and q.mode == "push":
            assert q.consumer, f"{q.key} asks without naming its consumer"
            assert len(q.consumer) > 40, (
                f"{q.key}'s consumer text does not say what changes")


def test_the_target_question_names_what_reads_the_answer():
    q = next(x for x in data_plan(target=None, detection=None)
             if x.key == "choose_target")
    consumer = q.consumer.lower()
    assert "detect_task_type" in consumer
    assert "lockbox" in consumer, (
        "the consumer text does not mention that the held-out test set is drawn "
        "against this column")


def test_the_task_type_question_names_what_changes():
    q = next(x for x in data_plan() if x.key == "confirm_task_type")
    consumer = q.consumer.lower()
    assert "metric" in consumer and "stratified" in consumer
    assert "does not raise an error" in consumer, (
        "the consumer text does not say that getting it wrong is silent")


def test_the_audit_refuses_a_fact_with_no_consumer():
    """The rule, enforced where a new question would have to pass it.

    Clause: `assembly-04`
    """
    q = router.Question(key="invented_fact", title="Is this a thing?",
                        why="", step="data", kind="task_type", mode="push")
    with pytest.raises(router.RouterError) as exc:
        router.audit([q])
    assert "no right to ask" in str(exc.value)


def test_a_skipped_fact_needs_no_consumer_because_it_is_not_asked():
    plan = data_plan(detection={"detected": "classification",
                                "confidence": "high",
                                "reasons": ["Two distinct values."]})
    skipped = [q for q in plan if q.status == "skipped"]
    assert skipped, "the high-confidence path no longer skips"
    router.audit(plan)


# ═══════════════════════════════════════════════════════════════════════════
# The three types stay distinguishable
# ═══════════════════════════════════════════════════════════════════════════

def test_the_three_kinds_are_named_in_the_router_not_only_in_the_page():
    assert router.FACT_KINDS and router.CHOICE_KINDS and router.CONSEQUENCE_KINDS
    assert not (router.FACT_KINDS & router.CHOICE_KINDS)
    assert not (router.CHOICE_KINDS & router.CONSEQUENCE_KINDS)
    assert not (router.FACT_KINDS & router.CONSEQUENCE_KINDS)



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
from __future__ import annotations

import re
from pathlib import Path

import pytest

from ml import router

REPO_ROOT = Path(__file__).resolve().parent.parent
PAGE = (REPO_ROOT / "turbotab" / "web" / "index.html").read_text(encoding="utf-8")
BODY = PAGE[PAGE.index("</style>"):]


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
    """The rule, enforced where a new question would have to pass it."""
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


def test_the_page_shows_the_disclosure_and_never_invents_the_text():
    assert 'class="whyask"' in BODY and "Why we ask" in BODY
    assert "consumerFor" in BODY and "LAST_DATA_PLAN" in BODY, (
        "the page composes its own consumer text instead of quoting the Router")
    # An empty disclosure is hidden, not shown blank: a "Why we ask" that
    # answers nothing is worse than no disclosure at all.
    assert 'box.style.display = "none"' in BODY


def test_the_fact_is_the_lightest_object_on_screen():
    rules = BODY[BODY.index(".fact-mark{"):BODY.index(".skips{")]
    for heavy in ("box-shadow", "border-radius:1", "background:var(--warn",
                  "background:var(--stop"):
        assert heavy not in rules, (
            f"the FACT treatment carries {heavy!r}; a question of fact has no "
            "border, no icon and no background tint of its own")


def test_the_teal_marker_sits_on_one_question_only():
    assert 'id="targetMark"' in BODY
    assert 'mark.style.display = P && P.target ? "none" : ""' in BODY, (
        "the current-question marker is not cleared once the question is "
        "answered, so more than one thing reads as 'now'")


# ═══════════════════════════════════════════════════════════════════════════
# The three types stay distinguishable
# ═══════════════════════════════════════════════════════════════════════════

def test_the_three_kinds_are_named_in_the_router_not_only_in_the_page():
    assert router.FACT_KINDS and router.CHOICE_KINDS and router.CONSEQUENCE_KINDS
    assert not (router.FACT_KINDS & router.CHOICE_KINDS)
    assert not (router.CHOICE_KINDS & router.CONSEQUENCE_KINDS)
    assert not (router.FACT_KINDS & router.CONSEQUENCE_KINDS)


def test_only_the_consequence_is_pushed_as_a_band():
    """Silhouette, not color, is the primary channel.

    A flat inline row (FACT), a bordered before/after card (CHOICE) and a
    full-width band (CONSEQUENCE). The band's rule is the one worth asserting:
    exactly one component breaks the page rhythm.
    """
    bands = re.findall(r"^\s*\.(\w[\w-]*)\{[^}]*margin:4px -30px", BODY,
                       re.MULTILINE)
    assert bands == ["blocker"], (
        f"{bands} break the page rhythm; the interruption silhouette is the "
        "blocker's alone")


def test_the_choice_card_stays_below_the_interruption_hierarchy():
    """CHOICE cards recur; their frequency must never erode the blocker."""
    choice_rules = BODY[BODY.index(".choice-acts{"):BODY.index(".ev{")]
    assert "var(--stop" not in choice_rules
    assert "small-caps" not in choice_rules
    assert "-30px" not in choice_rules, (
        "a CHOICE card breaks the page rhythm; it is deliberately inline")


def test_the_signal_word_appears_nowhere_else():
    """"a signal word that appears nowhere else in the product"."""
    rendered = re.findall(r">\s*Blocker\s*<", BODY)
    assert len(rendered) == 1, (
        f"the signal word is rendered {len(rendered)} times; a word used twice "
        "is not a signal word")


def test_the_grammar_survives_color_removal():
    """Silhouette plus signal word must carry the tier with color gone."""
    assert "blocker-word" in BODY and "blocker-glyph" in BODY
    band = BODY[BODY.index(".blocker{"):BODY.index(".blocker-head{")]
    assert "border-top:4px" in band, (
        "with color removed the band has no heavier rule than anything else")


def test_the_grammar_survives_typography_removal():
    """Silhouette plus grammar. The verbs differ per type."""
    assert "Keep as is" in BODY                      # CHOICE: decide
    assert "Ask me anyway" in BODY                   # FACT: answer
    assert "Keep it and record why" in BODY          # CONSEQUENCE: attest
    assert "Drop " in BODY                           # CONSEQUENCE: resolve

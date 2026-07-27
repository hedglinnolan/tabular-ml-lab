"""The Router's gates: Decision B, determinism, and no silent skips.

Separate from the value check. These say the Router obeys the governing rules;
the value check says whether obeying them produces a better interview. A Router
can pass every test here and still fail the check, which is exactly why the
check exists.
"""
from __future__ import annotations

import pathlib

import pandas as pd
import pytest

from ml import router
from ml.router import Question, RouterError
from turbotab import engine

DATA = pathlib.Path(__file__).resolve().parents[1] / "turbotab" / "sample_data"


@pytest.fixture(scope="module")
def messy():
    df = engine.read_table((DATA / "clinic_visits.csv").read_bytes(), "clinic.csv")
    findings = engine.rank_findings(engine.diagnose(df), None)
    detection = engine.detect_task_type(df, "outcome")
    return df, findings, detection


# ── determinism ──────────────────────────────────────────────────────────

def test_the_same_record_yields_the_same_next_question(messy):
    """The gate: same project and record, same next question."""
    _, findings, detection = messy
    args = dict(target="outcome", detection=detection, step="data")

    first = router.next_question(findings, **args)
    for _ in range(5):
        again = router.next_question(findings, **args)
        assert again.key == first.key
        assert again.title == first.title


def test_the_plan_is_a_pure_function_of_the_record(messy):
    """No clock, no RNG, no session state — replayable from the record alone."""
    _, findings, detection = messy
    a = [q.to_dict() for q in router.plan(findings, target="outcome",
                                          detection=detection, step="data")]
    shuffled = list(reversed(findings))
    b = [q.to_dict() for q in router.plan(shuffled, target="outcome",
                                          detection=detection, step="data")]
    assert a == b, "the plan depends on the order findings arrive in"


def test_answering_a_question_removes_it_and_nothing_else(messy):
    _, findings, detection = messy
    before = router.plan(findings, target="outcome", detection=detection, step="data")
    asked = [q.key for q in before if q.status == "asked"]

    after = router.plan(findings, target="outcome", detection=detection,
                        step="data", answered=[asked[0]])
    assert [q.key for q in after if q.status == "asked"] == asked[1:]


# ── Decision B ───────────────────────────────────────────────────────────

def test_a_repair_is_never_skipped_however_confident_the_engine_is(messy):
    """A repair is a question of choice. Skipping it applies a change the user
    never saw, which is the blind consent the preview exists to end."""
    _, findings, detection = messy
    high = [f for f in findings if f.get("confidence") == "high"
            and f.get("fix_kind") not in (None, "none")]
    assert high, "the fixture has no high-confidence repairs to check"

    plan = router.plan(findings, target="outcome", detection=detection, step="data")
    for q in plan:
        if q.kind == "repair":
            assert q.status != "skipped", (
                f"{q.key} was skipped — a repair is a choice, not a fact")


def test_a_high_confidence_detection_may_be_stated_rather_than_asked(messy):
    """The one legitimate skip: a question of fact the engine is certain of."""
    _, findings, detection = messy
    assert detection["confidence"] == "high"

    plan = router.plan(findings, target="outcome", detection=detection, step="data")
    q = next(q for q in plan if q.key == "confirm_task_type")
    assert q.status == "skipped"
    assert q.skip_reason and "high confidence" in q.skip_reason
    assert "change it there" in q.skip_reason, "the skip must be reversible"


def test_a_low_confidence_detection_is_always_asked(messy):
    _, findings, _ = messy
    for conf in ("medium", "low", None):
        detection = {"detected": "regression", "confidence": conf, "reasons": ["x"]}
        plan = router.plan(findings, target="bp_1", detection=detection, step="data")
        q = next(q for q in plan if q.key == "confirm_task_type")
        assert q.status == "asked", f"a {conf!r}-confidence detection auto-advanced"


def test_audit_rejects_a_skip_that_breaks_decision_b():
    """The rule is asserted, not trusted."""
    bad = Question(key="repair::x", title="t", why="w", step="data", kind="repair",
                   confidence="high", status="skipped", skip_reason="because")
    with pytest.raises(RouterError, match="Decision B"):
        router.audit([bad])

    silent = Question(key="confirm_task_type", title="t", why="w", step="data",
                      kind="task_type", confidence="high", status="skipped")
    with pytest.raises(RouterError, match="visible"):
        router.audit([silent])


def test_audit_rejects_a_deferral_with_nowhere_to_go():
    orphan = Question(key="repair::x", title="t", why="w", step="data",
                      kind="repair", status="deferred")
    with pytest.raises(RouterError, match="never resurface"):
        router.audit([orphan])


def test_a_clean_plan_passes_its_own_audit(messy):
    _, findings, detection = messy
    router.audit(router.plan(findings, target="outcome", detection=detection,
                             step="data"))


# ── what becomes a question at all ───────────────────────────────────────

def test_only_repairable_findings_become_questions(messy):
    """`fix_kind == 'none'` is the engine refusing to guess — a report, not a
    fork. `info` findings are noticings; one question each is the wall of
    prompts an interview replaces."""
    _, findings, detection = messy
    plan = router.plan(findings, target="outcome", detection=detection, step="data")
    asked = {q.triggering_finding for q in plan if q.kind == "repair"}

    for f in findings:
        repairable = (f["severity"] in ("critical", "warning")
                      and f.get("fix_label")
                      and f.get("fix_kind") not in (None, "none"))
        if repairable:
            assert f["id"] in asked, f"{f['id']} is repairable but never asked"
        else:
            assert f["id"] not in asked, f"{f['id']} became a question it should not"


def test_every_repair_question_cites_the_finding_that_raised_it(messy):
    """"Push the notable" is only true if the question says what it is pushing."""
    _, findings, detection = messy
    for q in router.plan(findings, target="outcome", detection=detection, step="data"):
        if q.kind == "repair":
            assert q.is_findings_driven
            assert q.why, f"{q.key} cites a finding but says nothing about why"


def test_questions_are_ranked_by_the_engines_severity(messy):
    _, findings, detection = messy
    plan = [q for q in router.plan(findings, target="outcome", detection=detection,
                                   step="data") if q.kind == "repair"]
    ranks = [router._SEVERITY_RANK.get(q.severity, 99) for q in plan]
    assert ranks == sorted(ranks), "repairs are not in the engine's severity order"


# ── deferral is a disposition, not a discard ─────────────────────────────

def test_a_deferred_question_resurfaces_at_the_step_it_names(messy):
    """The promise deferral makes. Without this it is a discard with manners."""
    _, findings, detection = messy
    first = router.plan(findings, target="outcome", detection=detection, step="data")
    victim = next(q.key for q in first if q.kind == "repair")

    # Deferred at `data`, targeting `explore`.
    deferred = {victim: "explore"}

    at_data = router.plan(findings, target="outcome", detection=detection,
                          step="data", deferred=deferred)
    moved = next(q for q in at_data if q.key == victim)
    assert moved.status == "deferred" and moved.defer_target == "explore"

    at_explore = router.plan(findings, target="outcome", detection=detection,
                             step="explore", deferred=deferred, answered=["choose_target"])
    back = next((q for q in at_explore if q.key == victim), None)
    assert back is not None, "a deferred question never resurfaced"
    assert back.status == "asked"
    assert back.deferred_from == "data"


def test_router_imports_without_streamlit():
    src = open(router.__file__, encoding="utf-8").read()
    assert "streamlit" not in src

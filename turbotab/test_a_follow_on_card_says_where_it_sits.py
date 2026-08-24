"""`GUIDED-040` — a `?` where a position belongs, and a card rendered twice.

Two defects in one function, both found by driving the page rather than by
reading it.

## The marker

Every dedicated card renders `01` in its kicker slot. The generic channel
rendered

    esc(q.clause ? "§" : "?")

so every follow-on card carried a literal question mark in the slot a reader has
been taught holds a position. DESIGN_LANGUAGE §08 says *structural devices must
encode something true*; a `?` encodes that the page did not know.

It did not, and it did not have to: the pre-seal sequence is **fixed** —
`OPENING_SEQUENCE.md` §01, constitution clause 01, *nothing may be resequenced*
— so every question in it has a position, and the position belongs to the module
that owns the sequence. `ml/router.SEQUENCE` is that table, and
`test_the_marker_is_the_sequence_the_document_states` reads the numbers back out
of `OPENING_SEQUENCE.md` so the two cannot drift apart in silence.

A question **outside** the sequence gets a word, not a number. The survey pack's
reverse-coding question is not a step of the pre-seal agreement — it is the one
question a pack is allowed to add — and numbering it would assert an ordering
the constitution does not contain.

## The duplicate

`renderAsked` filtered on `HANDLED_QUESTION_KEYS` alone. The prefixes whose
count is data — `repair::`, `blocker::`, `missingness::` — lived only in the
coverage test's own `HANDLED_PREFIXES`, so every repair question the Router
served was rendered **twice**: once as its finding card in `structList`, and
once more as a generic card whose buttons had no `ANSWERABLE` entry and
therefore did nothing at all.

On `metabolomics_untargeted.csv` that is nine dead cards. The coverage test
could not see it, because it asks *is this key renderable somewhere* and the
answer was yes, twice.

The fix is `FEATURE_PARITY.md`'s principle-locality rule: the list lives in the
page, the test reads it from there, and a prefix added to one is added to both.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml import router                                                 # noqa: E402
from turbotab import engine, pageharness as H                         # noqa: E402

DATA = Path(__file__).resolve().parent / "sample_data"
DOC = (Path(__file__).resolve().parents[1] / "docs" / "turbotab" /
       "OPENING_SEQUENCE.md")


def test_no_question_the_page_renders_carries_a_bare_question_mark():
    """The defect, as the property it broke.

    Driven over every fixture and both grain branches rather than asserted of
    one card, because a marker is a rendering rule and a rule with one witness
    is a coincidence.
    """
    seen = 0
    for name, target in (("metabolomics_untargeted", "responder"),
                         ("clinical_longitudinal", "progressed"),
                         ("survey_instrument", "sought_support")):
        df = pd.read_csv(DATA / f"{name}.csv")
        ranked = engine.rank_findings(engine.diagnose(df, target=target), None)
        block = None
        if name == "survey_instrument":
            from turbotab import packs as P
            block = P.likert_block(df)
        for step in router.STEPS:
            plan = router.plan(ranked, target=target, detection=None, step=step,
                               deferred={}, answered=[], recommendations=[],
                               signals=None, missing_columns=[],
                               lens_block=block)
            for q in plan:
                d = q.to_dict()
                assert d["seq"], (
                    f"{d['key']} would render a placeholder marker; every "
                    f"question knows either its position in the pre-seal "
                    f"sequence or the step that raised it")
                assert d["seq"] not in ("?", "§"), d["key"]
                seen += 1
    # An anti-vacuity floor, not a target. The number FALLS as the interview
    # gets better — `DRIVE-002`'s grouping took it from 24 to 22 by asking
    # fewer questions about the same findings — so the bar is "did this drive
    # actually enumerate questions", and a bound tight enough to track the
    # count would fail every time the product improves.
    assert seen >= 15, f"the drive covered only {seen} questions"


def test_the_marker_is_the_sequence_the_document_states():
    """The numbers are read back out of `OPENING_SEQUENCE.md` §01's own table.

    An expiring-guarantee guard in the shape `FEATURE_PARITY.md` prescribes:
    *name the expiry condition in the artifact.* The document says the sequence
    is fixed and nothing may be resequenced; if somebody moves a row in that
    table and not in `SEQUENCE`, the interface would number the questions one
    way while the constitution numbered them another, and neither would say so.
    """
    rows = re.findall(r"^\|\s*([0-9.]+)\s*\|\s*\*\*(.+?)\*\*", DOC.read_text(),
                      re.MULTILINE)
    assert len(rows) >= 8, (
        f"§01's table could not be read; it has {len(rows)} numbered rows")
    documented = {n for n, _ in rows}
    served = {v for k, v in router.SEQUENCE.items()}
    # `02` covers the target and the task-type row inside it, so the served set
    # is compared as positions and not as a count of questions.
    normalized = {n.zfill(2) if n.isdigit() else n for n in documented}
    assert normalized == served, (
        "the interface and the constitution disagree about the pre-seal "
        f"sequence.\n  document: {sorted(normalized)}\n  router:   {sorted(served)}")


def test_a_question_outside_the_sequence_gets_a_word_and_not_a_number():
    """The survey pack's question is not a step of the pre-seal agreement.

    Numbering it would assert an ordering the constitution does not contain,
    which is §08's rule broken in the other direction — a structural device
    encoding something false rather than nothing.
    """
    from turbotab import packs as P
    df = pd.read_csv(DATA / "survey_instrument.csv")
    plan = router.plan([], target="sought_support", detection=None, step="data",
                       deferred={}, answered=["state_lens", "choose_target",
                                              "state_grain"],
                       recommendations=[], signals=None, missing_columns=[],
                       lens_block=P.likert_block(df))
    rc = next(q.to_dict() for q in plan if q.key == "state_reverse_coding")
    assert rc["seq"] == "pack", rc["seq"]
    assert rc["seq"] not in router.SEQUENCE.values()


# ── the duplicate ────────────────────────────────────────────────────────────

def _js_array(name):
    text = H.PAGE.read_text(encoding="utf-8")
    start = text.index(f"var {name} = [")
    return re.findall(r'"([^"]+)"', text[start:text.index("];", start)])


def test_a_repair_is_rendered_once_and_not_twice():
    """Read back off the render.

    Nine binary-text repairs on `metabolomics_untargeted.csv`. Each has a
    finding card; each ALSO had a generic card with three buttons that no-op,
    because `ANSWERABLE` has no `repair::` entry and `submitAnswer` returns on a
    missing spec. A control that silently does nothing is the thing GUIDED-006
    exists to forbid, and here there were twenty-seven of them.
    """
    if not H.available():
        pytest.skip("no JS engine on this machine")
    from fastapi.testclient import TestClient

    from turbotab import api
    client = TestClient(api.app)
    # `clinic_visits`, not `metabolomics_untargeted`, since `DRIVE-002` landed:
    # every binary-text repair on the metabolomics fixture is now covered by one
    # group, so `repair::` questions no longer exist there and the assertion
    # below would pass on an empty set. This fixture still serves both kinds —
    # two groups AND three ungrouped repairs — which is the case worth guarding.
    with open(DATA / "clinic_visits.csv", "rb") as fh:
        project = client.post("/project", files={
            "file": ("c.csv", fh, "text/csv")}).json()
    pid = project["id"]
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_target", "payload": {"column": "outcome"}})
    project = client.get(f"/project/{pid}").json()
    plan = client.get(f"/project/{pid}/interview?step=data").json()
    served = [q["key"] for q in plan["questions"]
              if q["mode"] == "push" and q["status"] == "asked"
              and q["key"].startswith("repair::")]
    grouped = [q["key"] for q in plan["questions"]
               if q["key"].startswith("repair_bulk::")]
    assert served, "no repair questions on this fixture; the test proves nothing"
    assert grouped, "no repair GROUPS on this fixture; half the check is idle"

    html = H.run("__emit(__harness.html('askedQuestions'));", routes={
        f"/project/{pid}": project,
        f"/project/{pid}/interview?step=data": plan,
        f"/project/{pid}/interview?step=explore": {"questions": []},
        f"/project/{pid}/evidence/missingness": {"cards": []},
    }, search=f"?project={pid}")

    leaked = sorted({b["data-answer-key"] for b in H.elements(html)
                     if b.get("data-answer-key", "").startswith("repair")})
    assert not leaked, (
        f"{len(leaked)} repair question(s) rendered a second time in the "
        f"generic channel, with buttons that do nothing: {leaked[:4]}")


def test_the_page_and_its_coverage_test_read_one_prefix_list():
    """Principle-locality, made executable.

    The prefix list was in the test and not in the page, which is how the page
    came to render cards the test had already accounted for elsewhere. One
    list, two readers.
    """
    from turbotab import test_the_page_asks_what_the_router_serves as COV
    assert tuple(_js_array("HANDLED_QUESTION_PREFIXES")) == COV.HANDLED_PREFIXES, (
        "the page and the coverage test hold different prefix lists, so a "
        "question can be accounted for in one and orphaned or duplicated in "
        "the other")

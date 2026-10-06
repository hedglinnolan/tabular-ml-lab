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
# The tests here that drove the retired legacy app (turbotab/api.py, the page, the
# figure and manuscript modules, the gates) were removed with it, BLUEPRINT §9.1; git keeps them.
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml import router                                                 # noqa: E402
from turbotab import engine                                             # noqa: E402

DATA = Path(__file__).resolve().parent / "sample_data"
DOC = (Path(__file__).resolve().parents[1] / "docs" / "turbotab-next" /
       "reference" / "OPENING_SEQUENCE.md")


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

"""`AUDIT-034` — the recorded reverse-coding sentence, and the two clauses in it.

## What the sentence used to promise

`set_reverse_coding` recorded *"{n} item(s) were declared reverse-coded and will
be flipped before the scale is scored"*, and for the empty answer *"the scale is
scored with every item in the direction it was recorded"*. The record panel
renders that sentence and `draft.py` folds it into the **Data preparation**
section of the exported methods, so it is a claim that leaves the building.

Two things in it were false and they failed differently.

**The flip.** Nothing flips the declared columns in the table any other analysis
reads. `project.working_table` applies the cohort filter and nothing else, so a
declared item is byte-identical to the column that was uploaded.

**The scoring step.** There is no scale score anywhere in this repository. The
only row-wise sums are `survey.py`'s internal rest score and `figure_specs.py`'s
histogram total, and neither is a scale score the user could read, export or
model. So *"before the scale is scored"* names a step that does not exist at
all, which is the harder half: the flip could be built, the scoring step was
never on the roadmap.

## Reverse-coding is `DOMAIN_SCIENCE.md` §01's hard-stop class

*Detect, declare, never execute.* The declaration comes from the instrument's
published scoring key — `survey.py` §B1.2 is explicit that TurboTab will not
infer it from correlations, because a negative item–rest correlation has four
incompatible causes and no correlational signature separates them. So the honest
sentence is not *"we flipped them"*; it is *"you declared them, here is the one
place that reads the declaration, and nothing else is scored from it."*

## What the sentence must therefore assert, and this file drives all three

1. **The declaration was recorded, per the scoring key** — the list, verbatim.
2. **The reverse-coding audit recomputes each item's correlation with the rest
   of its scale with the reversal applied.** This is a claim about a computation
   and it is TRUE — `survey.audit` applies `(min+max)−x` to the declared columns
   at `survey.py:417-419` and reports `item_rest_r_after_reversal` beside the
   raw one. `test_the_audit_clause_is_true_and_the_reversal_moves_the_number`
   observes the number move rather than reading the sentence.
3. **This app computes no scale score**, so the declaration is not applied to
   the table any other analysis reads.

`AGENT_ONBOARD.md` §07 trap #3b is why claim 2 is driven rather than grepped: a
test named *the audit applies the reversal* that only asserted the sentence
would be the guard whose name carries a consequence its body never checks.

## `GUIDED-097` — the fixture rule

Both survey fixtures, `survey_instrument.csv` (no sentinels by construction) and
`survey_sentinels.csv` (the codebook's missing codes written in), because the
audit excludes sentinels before correlating and a claim made only against the
clean return would be a claim about their absence.

**The shape not covered is said out loud at the bottom of this file.**
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbotab import api                                              # noqa: E402

DATA = Path(__file__).resolve().parent / "sample_data"

#: `GUIDED-097`. Two survey returns of different shape.
FIXTURES = ("survey_instrument.csv", "survey_sentinels.csv")

#: The codebook's reversals, from `make_fixtures.REVERSE_CODED`.
DECLARED = ["item_05", "item_11"]

#: Every promise the old sentence made that the app does not keep. Each is a
#: fragment of the text this row was filed against, so a revert puts one of them
#: back and this file names which.
FALSE_PROMISES = (
    "will be flipped before the scale is scored",
    "be flipped before the scale is scored",
    "the scale is scored with every item",
    "before combining them",
)


@pytest.fixture(scope="module")
def client():
    return TestClient(api.app)


def _survey_project(client, fixture: str, columns):
    """A survey project with the lens on and one recorded declaration."""
    with open(DATA / fixture, "rb") as fh:
        pid = client.post("/project", files={
            "file": (fixture, fh, "text/csv")}).json()["id"]
    r = client.post(f"/project/{pid}/decision",
                    json={"kind": "set_lens", "payload": {"lens": ["survey"]}})
    assert r.status_code == 200, r.text[:300]
    r = client.post(f"/project/{pid}/decision",
                    json={"kind": "set_reverse_coding",
                          "payload": {"columns": list(columns)}})
    assert r.status_code == 200, r.text[:300]
    return pid, r.json()


def _recorded_sentence(payload) -> str:
    hits = [d.get("text") or "" for d in payload.get("decisions", [])
            if d.get("kind") == "set_reverse_coding"]
    assert len(hits) == 1, f"expected one recorded declaration, got {len(hits)}"
    return hits[0]


# ─────────────────────────────────────────────────────────────────────────────
# 1 · the sentence promises nothing the app does not do
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("fixture", FIXTURES)
def test_the_recorded_sentence_promises_no_flip_and_no_scoring(client, fixture):
    pid, payload = _survey_project(client, fixture, DECLARED)
    text = _recorded_sentence(payload)
    for promise in FALSE_PROMISES:
        assert promise not in text, (
            f"the recorded reverse-coding sentence still promises "
            f"{promise!r}: {text}")
    # And it says the true thing rather than going silent — the shelf is not
    # shortened, the claim is corrected.
    assert "no scale score" in text, text
    for column in DECLARED:
        assert f"`{column}`" in text, text


@pytest.mark.parametrize("fixture", FIXTURES)
def test_the_empty_answer_is_recorded_and_promises_nothing_either(client, fixture):
    """The empty list is an ANSWER — `api.py` stores it rather than treating it
    as no answer — so its sentence is exported too and must be true as well."""
    pid, payload = _survey_project(client, fixture, [])
    text = _recorded_sentence(payload)
    for promise in FALSE_PROMISES:
        assert promise not in text, (
            f"the empty-answer sentence still promises {promise!r}: {text}")
    assert "No items were declared reverse-coded" in text, text
    assert "no scale score" in text, text


# ─────────────────────────────────────────────────────────────────────────────
# 2 · what the app actually does to a reverse-coded item
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("fixture", FIXTURES)
def test_nothing_downstream_reads_a_flipped_column(client, fixture):
    """The claim *the declaration is not applied to the table any other
    analysis reads*, observed rather than asserted."""
    pid, _ = _survey_project(client, fixture, DECLARED)
    project = api.STORE.get(pid)
    for column in DECLARED:
        before = project.df[column]
        after = project.working_table[column]
        assert after.equals(before.loc[after.index]), (
            f"{column} differs between the uploaded table and the working "
            f"table; the sentence says the declaration is not applied to it")


@pytest.mark.parametrize("fixture", FIXTURES)
def test_no_scale_score_is_constructed_anywhere_the_project_exposes(client, fixture):
    """*This app computes no scale score.* If one ever appears, this sentence
    stops being true and the row reopens — which is what this assertion is for."""
    pid, _ = _survey_project(client, fixture, DECLARED)
    project = api.STORE.get(pid)
    scored = [c for c in project.working_table.columns
              if "scale_score" in str(c).lower() or "total_score" in str(c).lower()]
    assert not scored, f"a scale score appeared in the working table: {scored}"


@pytest.mark.parametrize("fixture", FIXTURES)
def test_the_audit_clause_is_true_and_the_reversal_moves_the_number(client, fixture):
    """Trap #3b. The sentence claims the audit *recomputes each item's
    correlation with the reversal applied*; this observes the recomputation.

    A declared item's item–rest correlation after reversal must differ from its
    raw one — that is the reversal being applied inside the audit — and an
    undeclared item's must not move materially, which is what makes the first
    difference attributable to the declaration rather than to the audit's own
    arithmetic.
    """
    pid, _ = _survey_project(client, fixture, DECLARED)
    body = client.get(f"/project/{pid}/evidence/reverse-coding").json()
    rows = {r["item"]: r for r in body["rows"]}
    assert set(body["declared_reversed"]) == set(DECLARED), body["declared_reversed"]

    for column in DECLARED:
        row = rows[column]
        assert row["reversal_declared"] is True, row
        raw, after = row["item_rest_r_raw"], row["item_rest_r_after_reversal"]
        assert raw is not None and after is not None, row
        assert abs(after - raw) > 0.05, (
            f"{column} was declared reverse-coded and its item-rest "
            f"correlation did not move when the audit applied the reversal "
            f"({raw} → {after}); the recorded sentence says it does")

    undeclared = [r for r in body["rows"] if not r["reversal_declared"]
                  and r["item_rest_r_raw"] is not None][:5]
    assert undeclared, "no undeclared item to compare against"
    for row in undeclared:
        assert abs(row["item_rest_r_after_reversal"]
                   - row["item_rest_r_raw"]) < 0.05, (
            f"{row['item']} was not declared and its correlation moved anyway; "
            f"the difference above is then not the declaration's doing")


# ─────────────────────────────────────────────────────────────────────────────
# 3 · the artifact that leaves the building
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("fixture", FIXTURES)
def test_the_draft_carries_the_corrected_sentence_and_not_the_promise(client, fixture):
    """`draft.py`'s fall-through folds the recorded text into **Data
    preparation**, which is the exported methods section. The row was filed
    because the false clause reached here."""
    pid, _ = _survey_project(client, fixture, DECLARED)
    draft = client.get(f"/project/{pid}/draft").json()
    lines = [s["text"] for sec in draft["sections"] for s in sec["sentences"]
             if "revers" in s["text"].lower()]
    assert lines, "the declaration reached no draft section at all"
    for line in lines:
        for promise in FALSE_PROMISES:
            assert promise not in line, (
                f"the exported methods draft still promises {promise!r}: {line}")
    assert any("no scale score" in line for line in lines), lines


def test_the_question_card_promises_only_what_the_audit_does(client):
    """`ml/router.py`'s `consumer` is the second place the promise was made —
    the card the user reads BEFORE answering. A correction that fixed the
    receipt and left the question is the same false claim one screen earlier."""
    with open(DATA / FIXTURES[0], "rb") as fh:
        pid = client.post("/project", files={
            "file": (FIXTURES[0], fh, "text/csv")}).json()["id"]
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_lens", "payload": {"lens": ["survey"]}})
    plan = client.get(f"/project/{pid}/interview").json()
    cards = [q for q in (plan.get("questions") or [])
             if q.get("key") == "state_reverse_coding"]
    assert len(cards) == 1, f"expected the one pack-added card, got {len(cards)}"
    consumer = cards[0]["consumer"]
    for promise in FALSE_PROMISES:
        assert promise not in consumer, (
            f"the question card still promises {promise!r}: {consumer}")
    assert "computes no scale score" in consumer, consumer


#: NOT COVERED, said out loud — `GUIDED-097`'s second clause.
#:
#: A TARGET-BEARING SURVEY RETURN. Both fixtures here are answered before a
#: target is chosen, because the reverse-coding card is a `step="data"` question
#: the survey lens raises at the front of the journey and the declaration is
#: recorded there. The shape not covered is a survey project SEALED with a
#: target, where a scale score would be the thing a model is fitted on — which
#: is precisely the state in which "no scale score exists" would be most costly
#: if it stopped being true. `AUDIT-035` is the row that lives in that state
#: (the purpose question claiming to decide scale-level vs item-level), and it
#: is open.
#:
#: A NON-LIKERT TABLE. `read_sentinels` returns no block, `survey.audit` returns
#: `None`, and the card is never raised — so there is no sentence to be false.
#: That is a refusal to compute rather than an uncovered claim.

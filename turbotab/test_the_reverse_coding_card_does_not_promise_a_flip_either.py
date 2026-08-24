"""`AUDIT-034` · the flip promise on the question card, and the last export hop.

## What was still there

`AUDIT-034` was filed against *"{n} item(s) were declared reverse-coded and will
be flipped before the scale is scored"* — a flip and a scoring step the app never
performs, recorded on `set_reverse_coding` and folded into the exported methods.
`L52` corrected the receipt (`turbotab/api.py`), the disposition preview
(`turbotab/web/index.html`) and the question card's `consumer`
(`ml/router.py`), and `test_the_reverse_coding_sentence_says_what_the_app_does`
guards those.

**The card's `why` was not corrected and no guard covered it.** It read:

    40 columns share one 5-point response scale. If some of them are worded so
    that agreeing means the opposite, **they have to be flipped before the
    scale means anything.**

An obligation in the passive voice, with no agent, on the card that collects the
declaration — one line above a `consumer` that says the app computes no scale
score. The four phrases the existing guard sweeps for are the *receipt's*
phrasings; none of them matches this one, so a user reading the card before
answering was still told a flip was coming. Corrected to name who flips, which
keeps `research/CLINICAL_SURVEY_PACK.md` §B1.2's domain fact — reverse-worded
items must be recoded before a scale is scored — and states that this app is not
the thing that does it.

## The hop the row names and nothing drove

`AUDIT-034`'s `ev` ends *"and it is exported into the manuscript"*. The existing
guard stops at `/draft`. `turbotab/draft.py`'s fall-through is what puts the
recorded sentence into **Data preparation**, and `turbotab/manuscript.py` renders
that section into the markdown a user exports, so
`test_no_flip_promise_survives_into_the_exported_manuscript` drives the last hop.

**That test is not load-bearing for this file's change** and says so: the
sentence it reads is composed in `turbotab/api.py`, which this chunk does not
own and which is uncommitted in another chunk's hands. It closes the *export*
clause of the finding by observation, not by correction.

## `GUIDED-097` — two fixtures of different target shape

The card is a journey step (`step="data"`, raised by the survey lens), so every
claim runs against `survey_instrument.csv` sealed on **`sought_support`, binary
numeric** and `survey_sentinels.csv` sealed on **`education`, multiclass
string** — which also differ in whether the block carries sentinel codes.

**The shape not covered is said out loud at the bottom of this file.**
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbotab import api                                          # noqa: E402
from turbotab import manuscript as _manuscript                    # noqa: E402

DATA = Path(__file__).resolve().parent / "sample_data"

#: `GUIDED-097`. Two target shapes on the one journey step.
#: `(fixture, target column, target shape)`.
SHAPES = {
    "binary_numeric": ("survey_instrument.csv", "sought_support"),
    "multiclass_string": ("survey_sentinels.csv", "education"),
}

#: The codebook's reversals, from `make_fixtures.REVERSE_CODED`.
DECLARED = ["item_05", "item_11"]

#: Every phrasing of the promise this app does not keep. The first four are the
#: receipt's, kept here so a revert of either surface lands on a named phrase;
#: the last is the card's own, which no guard covered.
FLIP_PROMISES = (
    "will be flipped before the scale is scored",
    "be flipped before the scale is scored",
    "the scale is scored with every item",
    "before combining them",
    "flipped before the scale means anything",
)


@pytest.fixture(scope="module")
def client():
    return TestClient(api.app)


def _survey_project(client, shape, columns=DECLARED, declare=True):
    """A survey project with the lens on, a target set, and a declaration."""
    fixture, target = SHAPES[shape]
    with open(DATA / fixture, "rb") as fh:
        pid = client.post("/project", files={
            "file": (fixture, fh, "text/csv")}).json()["id"]
    r = client.post(f"/project/{pid}/decision",
                    json={"kind": "set_lens", "payload": {"lens": ["survey"]}})
    assert r.status_code == 200, r.text[:300]
    r = client.post(f"/project/{pid}/decision",
                    json={"kind": "set_target", "payload": {"column": target}})
    assert r.status_code == 200, r.text[:300]
    if declare:
        r = client.post(f"/project/{pid}/decision",
                        json={"kind": "set_reverse_coding",
                              "payload": {"columns": list(columns)}})
        assert r.status_code == 200, r.text[:300]
    return pid


def _card(client, shape):
    pid = _survey_project(client, shape, declare=False)
    plan = client.get(f"/project/{pid}/interview").json()
    cards = [q for q in (plan.get("questions") or [])
             if q.get("key") == "state_reverse_coding"]
    return pid, cards


# ═══════════ 1 · the card the user reads BEFORE answering ═══════════

@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_the_card_does_not_tell_the_user_a_flip_is_coming(client, shape):
    """`ml/router.py`'s `why`. The load-bearing claim of this file."""
    _, cards = _card(client, shape)

    # POSITIVE CONTROL — the card is raised at all on this shape, so an absence
    # below is about its wording rather than about a card nobody sees.
    assert len(cards) == 1, (
        f"{shape}: expected the one pack-added card, got {len(cards)}; the "
        f"sentence under test never reaches a user")
    why = cards[0].get("why") or ""
    assert why.strip(), f"{shape}: the reverse-coding card carries no why line"

    for promise in FLIP_PROMISES:
        assert promise not in why, (
            f"{shape}: the reverse-coding card still tells the user "
            f"{promise!r} before they answer. Nothing in this app flips a "
            f"declared column and nothing scores a scale — `working_table` "
            f"applies the cohort filter and nothing else. AUDIT-034. The card "
            f"read: {why!r}")

    # AUDIT-028's model: corrected, not deleted. The domain fact stays and the
    # app's non-participation is stated.
    assert "does not score it" in why, (
        f"{shape}: the card dropped the flip clause instead of saying who "
        f"flips. §B1.2's fact — reverse-worded items must be recoded before a "
        f"scale is scored — is true and belongs on this card; what did not "
        f"belong was leaving the agent unnamed. The card read: {why!r}")


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_the_cards_two_lines_do_not_contradict_each_other(client, shape):
    """The `why` and the `consumer` are read as one card, so they are one
    claim. A `consumer` saying no scale is scored under a `why` implying one is
    about to be is the contradiction this row was filed against."""
    _, cards = _card(client, shape)
    assert cards, f"{shape}: no reverse-coding card"
    card = cards[0]
    consumer = card.get("consumer") or ""
    why = card.get("why") or ""

    # POSITIVE CONTROL — the consumer really does make the no-scale-score claim,
    # so the agreement asserted below is between two live sentences.
    assert "computes no scale score" in consumer, (
        f"{shape}: the card's consumer no longer says the app computes no "
        f"scale score; the agreement checked here would be vacuous. "
        f"The consumer read: {consumer!r}")
    assert "does not score it" in why, (
        f"{shape}: the card's two lines disagree about whether this app "
        f"scores the scale. why={why!r} consumer={consumer!r}")


# ═══════════ 2 · nothing was flipped, driven ═══════════

@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_the_declared_columns_are_untouched_in_every_downstream_read(client, shape):
    """Why the card may not promise a flip: there is not one. Observed on the
    project the record panel and the draft both read from."""
    pid = _survey_project(client, shape)
    project = api.STORE.get(pid)
    for column in DECLARED:
        after = project.working_table[column]
        before = project.df[column].loc[after.index]
        assert after.equals(before), (
            f"{shape}: {column} differs between the uploaded table and the "
            f"working table, so something IS applying the declaration and the "
            f"card's wording is the wrong end of this row")


# ═══════════ 3 · and the export hop the row names ═══════════

@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_no_flip_promise_survives_into_the_exported_manuscript(client, shape):
    """`AUDIT-034`'s *"and it is exported into the manuscript"*, driven.

    **GREEN WITHOUT THIS FILE'S CHANGE, and that is stated rather than left to
    a probe to discover.** The sentence read here is composed in
    `turbotab/api.py` and folded by `turbotab/draft.py`; `L52` corrected the
    first and this chunk does not own it. This closes the export clause by
    observation.
    """
    pid = _survey_project(client, shape)
    project = api.STORE.get(pid)
    doc = _manuscript.structure(project.to_dict())
    rendered = _manuscript.to_markdown(doc)
    text = "\n".join(str(v) for v in rendered.values())

    # POSITIVE CONTROL (`GUIDED-045`) — the declaration reaches the export at
    # all. An absence swept over a manuscript that never carried the sentence
    # would be an absence about nothing.
    assert "reverse-coded" in text, (
        f"{shape}: the reverse-coding declaration reached no exported section, "
        f"so this sweep is measuring a manuscript that never carried it")
    for column in DECLARED:
        assert column in text, (
            f"{shape}: the exported methods lost `{column}` from the declared "
            f"list, which is §B7's reviewer-checklist item 4")

    for promise in FLIP_PROMISES:
        assert promise not in text, (
            f"{shape}: the exported manuscript still promises {promise!r}. "
            f"AUDIT-034 — this is the artifact that leaves the building.")


def test_the_empty_declaration_exports_without_a_scoring_claim(client):
    """The empty list is an ANSWER, so its sentence is exported too."""
    pid = _survey_project(client, "binary_numeric", columns=[])
    project = api.STORE.get(pid)
    rendered = _manuscript.to_markdown(_manuscript.structure(project.to_dict()))
    text = "\n".join(str(v) for v in rendered.values())

    # POSITIVE CONTROL — the empty answer reached the export.
    assert "No items were declared reverse-coded" in text, (
        "the empty answer reached no exported section, so the sweep below is "
        "about a manuscript that never carried it")
    for promise in FLIP_PROMISES:
        assert promise not in text, (
            f"the empty-answer sentence still promises {promise!r} in the "
            f"exported manuscript. AUDIT-034.")


#: NOT COVERED, said out loud — `GUIDED-097`'s second clause.
#:
#: A CONTINUOUS TARGET. The card is gated on the survey lens block and reads
#: neither the target nor the task, so a regression target exercises no branch
#: the two shapes above do not — the `why` is composed from the block's column
#: count and its response scale only. Named rather than assumed.
#:
#: A NON-LIKERT TABLE. `read_sentinels` returns no block, the card is never
#: raised and `survey.audit` returns `None`. There is no sentence to be false
#: there; that is a refusal to compute rather than an uncovered claim.
#:
#: THE SEALED SURVEY PROJECT. Both shapes above answer the card after a target
#: and before a seal. A survey project sealed with a target is the state in
#: which "this app does not score the scale" would be most costly if it stopped
#: being true, and `AUDIT-035` — the purpose question claiming to decide
#: scale-level versus item-level entry — is the open row that lives there.

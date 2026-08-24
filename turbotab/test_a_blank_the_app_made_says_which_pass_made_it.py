"""`GUIDED-166` — the impossibility pass makes blanks and nothing marked them.

## What the drive showed

`clinical_labs.csv`, lens `clinical`, target `readmitted`. The plausibility
report flags **4 entries of `sbp` outside 40.0–300.0 mmHg** as impossible (and
`glucose`, whose whole reading is suspect, which is a different card). The
column has **0 blanks** before the press. `set_impossible_missing` is real since
L47 — it sets those 4 entries to missing — so afterwards the column has **4**.

Two cards later the app asks *"could a blank in `sbp` mean something?"* — §07's
mechanism question, the fork the whole missingness step is built on. It was
being asked about four blanks **the app had written itself**, and no honest
answer to it is right for them: they are not *not asked*, not *not applicable*,
and not an accident of collection. They mean *this app judged 4 recorded values
impossible and removed them*, which the record already knew and the question
never saw.

## What is fixed here, and what is deliberately not

**Provenance is recorded and legible.** `set_impossible_missing` now writes the
row labels it blanked, and both doors' missingness surfaces carry a per-column
block saying how many of a column's blanks are the app's own, which rows, and —
quoting the decision's own sentence rather than composing a second one — why.

**The eligibility route is opened.** The product owner named three instincts.
`set-to-missing` is clause §06's row-local repair and was the only one offered.
`exclude-the-rows` is clause §04's ELIGIBILITY CRITERION, and
`project.set_eligibility` has always implemented it correctly — pre-seal,
changing N, reported in participant flow — while being unreachable from this
card. `ml/pipeline.apply_plausibility_filter` is the same idea on the Streamlit
door and is reached from `pages/05_Preprocess.py`: `MISC-014` — **unrouted is
not absent**, and it is not reported here as missing. `mark-the-column-
corrupted` is `GUIDED-096`'s split and is **not built**; it is named on the card
with its reason rather than left off, because a shelf silently holding two of
three tells the user the third is not a thing one may want.

**What is NOT claimed** *(superseded at L49-C — kept because the reason it was
written is the reason it could be retired)*. The remainder of a column's blanks
was reported as *not recorded as made here*, never as *blank in the file*,
because `coerce_numeric` also turned values into `NaN` and filed no provenance.
Every writer files now, through `project._install`, so the field is
`n_blank_in_the_file` and the hedge moved to the case that earns it — a pass
that rebuilt the row index. See `test_every_writer_that_can_blank_a_cell_files_it.py`.

`GUIDED-097`: two target shapes — `clinical_labs.csv` / `readmitted` (binary
classification) and `clinical_longitudinal.csv` / `hba1c` (continuous). The
shape not covered is a multiclass target: no fixture in `sample_data` has both
a multiclass outcome and a repairable impossible block.
"""
from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbotab import api                                              # noqa: E402
from turbotab import eligibility as E                                 # noqa: E402
from turbotab import missingness as M                                 # noqa: E402

DATA = Path(__file__).resolve().parent / "sample_data"

#: `(file, lens, target, grain-column)`. The grain answer is a precondition of
#: the eligibility route and both fixtures repeat, so it is part of the setup
#: rather than a thing the test discovers.
SHAPES = [
    pytest.param("clinical_labs.csv", "clinical", "readmitted", "patient_id",
                 id="classification-target"),
    pytest.param("clinical_longitudinal.csv", "clinical", "hba1c",
                 "subject_id", id="regression-target"),
]


@pytest.fixture(scope="module")
def client():
    return TestClient(api.app)


def _project(client, fixture: str, lens: str, target: str) -> str:
    with open(DATA / fixture, "rb") as fh:
        pid = client.post("/project", files={
            "file": (fixture, fh, "text/csv")}).json()["id"]
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_lens", "payload": {"lens": [lens]}})
    answer = client.post(f"/project/{pid}/decision",
                         json={"kind": "set_target",
                               "payload": {"column": target}})
    assert answer.status_code == 200, answer.text
    return pid


def _repairable_block(client, pid: str):
    report = client.get(f"/project/{pid}/evidence/plausibility").json()
    block = next((b for b in report.get("impossible", [])
                  if not b["whole_column_suspect"]), None)
    assert block is not None, (
        "no repairable impossible block on this fixture, so nothing below is "
        "being driven")
    return block


def _survey_row(client, pid: str, column: str):
    rows = client.get(f"/project/{pid}/preprocess").json()["columns"]
    return next((r for r in rows if r["column"] == column), None)


# ═══════════ (a) the blanks the app made say so ═════════════════════════════

@pytest.mark.parametrize("fixture,lens,target,person", SHAPES)
def test_the_survey_counts_the_blanks_the_impossibility_pass_created(
        client, fixture, lens, target, person):
    """The whole defect, driven: zero blanks, press, N blanks, and the survey
    row can now say all N are the app's own and name the rows."""
    pid = _project(client, fixture, lens, target)
    block = _repairable_block(client, pid)
    column, n_flagged = block["column"], int(block["n_flagged"])

    before = _survey_row(client, pid, column)
    assert before is None or before["n_missing"] == 0, (
        f"`{column}` already has blanks, so this cannot show that the pass "
        f"created them")

    answer = client.post(
        f"/project/{pid}/decision",
        json={"kind": "set_impossible_missing", "subject": column,
              "payload": {"column": column}})
    assert answer.status_code == 200, answer.text

    after = _survey_row(client, pid, column)
    assert after is not None and after["n_missing"] == n_flagged, (
        f"the pass reports {n_flagged} flagged and the column now holds "
        f"{after and after['n_missing']} blanks")
    prov = after["provenance"]
    assert prov is not None, (
        f"`{column}` holds {n_flagged} blanks this app wrote and the survey "
        f"row says nothing about where they came from — so the mechanism "
        f"question is about to be asked of blanks the app made")
    assert prov["n_created_by_the_app"] == n_flagged
    assert prov["n_blank_in_the_file"] == 0
    assert len(prov["rows"]) == n_flagged, (
        "the count and the row list disagree, so one of them is derived and "
        "not recorded")


@pytest.mark.parametrize("fixture,lens,target,person", SHAPES)
def test_the_provenance_sentence_quotes_the_decision_that_made_the_blanks(
        client, fixture, lens, target, person):
    """One composer. The `why` is the recorded decision's own sentence, so the
    card and the transcript cannot describe one event two ways — which is what
    `GUIDED-098` cost a loop one question over."""
    pid = _project(client, fixture, lens, target)
    block = _repairable_block(client, pid)
    column = block["column"]
    answer = client.post(
        f"/project/{pid}/decision",
        json={"kind": "set_impossible_missing", "subject": column,
              "payload": {"column": column}})
    assert answer.status_code == 200, answer.text

    recorded = [d for d in client.get(f"/project/{pid}").json()["decisions"]
                if d["kind"] == "set_impossible_missing"]
    assert recorded, "nothing was recorded, so there is no sentence to quote"
    sentence = _survey_row(client, pid, column)["provenance"]["sentence"]
    assert recorded[-1]["text"] in sentence, (
        f"the provenance note paraphrases the decision instead of quoting it:\n"
        f"  decision: {recorded[-1]['text']!r}\n  note: {sentence!r}")
    assert str(block["low"]) in sentence and str(block["high"]) in sentence, (
        "the note does not carry the band the blanks were made against")


@pytest.mark.parametrize("fixture,lens,target,person", SHAPES)
def test_a_column_the_app_never_blanked_carries_no_provenance_block(
        client, fixture, lens, target, person):
    """The negative control. A block on every column would be noise, and worse
    — *"0 of these are ours"* on a column the pass never touched is a labeled
    region asserting a finding of nothing."""
    pid = _project(client, fixture, lens, target)
    block = _repairable_block(client, pid)
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_impossible_missing",
                      "subject": block["column"],
                      "payload": {"column": block["column"]}})
    rows = client.get(f"/project/{pid}/preprocess").json()["columns"]
    others = [r for r in rows if r["column"] != block["column"]]
    for row in others:
        assert row["provenance"] is None, (
            f"{row['column']} carries a provenance block and the app blanked "
            f"nothing in it")


def test_the_two_doors_read_one_provenance(client):
    """The Explore card and the Preprocess survey say the same thing about
    where a blank came from, because both read `missingness.provenance` — the
    standing rule after `GUIDED-090` / `091` / `098`.

    The Explore card has a share threshold that four blanks in 288 rows do not
    cross, so the card composer is run at `threshold=0.0` against the REAL
    project rather than the drive being skipped: the claim is that the two
    surfaces agree, not that both happen to render on this fixture.
    """
    from ml import missingness_plan as MP

    pid = _project(client, "clinical_labs.csv", "clinical", "readmitted")
    block = _repairable_block(client, pid)
    column = block["column"]
    answer = client.post(
        f"/project/{pid}/decision",
        json={"kind": "set_impossible_missing", "subject": column,
              "payload": {"column": column}})
    assert answer.status_code == 200, answer.text

    survey = _survey_row(client, pid, column)["provenance"]
    assert survey is not None
    project = api._project(pid)
    cards = MP.missingness_cards(project.working_table, threshold=0.0,
                                 provenance=project.blank_provenance())
    card = next(c for c in cards if c["column"] == column)
    assert card["provenance"] == survey, (
        "the Explore card and the Preprocess survey describe the same blanks "
        "differently")


def test_the_served_missingness_cards_carry_the_provenance_field(client):
    """The wire, separately from the composer. `/evidence/missingness` has to
    pass the project's reading into the card builder, and a card that never
    carries the key is the field composed and never sent."""
    pid = _project(client, "clinic_visits.csv", "clinical", "outcome")
    cards = client.get(f"/project/{pid}/evidence/missingness").json()["cards"]
    assert cards, "no card on this fixture, so this asserts nothing"
    for card in cards:
        assert "provenance" in card, (
            f"{card['column']}'s card does not carry `provenance`, so the "
            f"Explore door cannot say where a blank came from")
        assert card["provenance"] is None, (
            "nothing blanked these columns and the card claims otherwise")


def test_the_reading_refuses_to_decompose_what_it_cannot(client):
    """Trap #9 at the arithmetic layer. If a later step fills blanks that the
    impossibility pass made, `n_missing - n_created` goes negative — and the
    honest report is that the decomposition does not hold, not a clamped zero
    that would assert the app made every blank in the column."""
    made = {"column": "sbp", "n": 4, "rows": [1, 2, 3, 4],
            "by": [{"kind": "set_impossible_missing", "n": 4,
                    "pass": "set_impossible_missing",
                    "sentence": "4 entries of `sbp` were set to missing."}]}
    reading = M.provenance("sbp", 1, made)
    assert reading["n_created_by_the_app"] == 4
    assert reading["n_blank_in_the_file"] is None, (
        "a negative remainder was reported as a number")
    assert "cannot be reconciled" in reading["sentence"]
    assert M.provenance("sbp", 0, None) is None
    assert M.provenance("sbp", 0, {"n": 0, "rows": [], "by": []}) is None

    # THE MIXED BRANCH, and it is reached by a call rather than by a drive
    # because no fixture in `sample_data` produces a column that both arrives
    # with blanks and gets more from the impossibility pass. Said out loud
    # rather than left as coverage nobody counted.
    #
    # **The claim flipped at L49-C and that is the finding, not a relaxation.**
    # This asserted `"not recorded as made here"` and `"came with the file" not
    # in …`, because one writer filed provenance and the remainder could only
    # be described by what the record had failed to see. Every writer files
    # now, so the weaker sentence would be the app under-claiming. The hedge
    # did not go away — `test_every_writer_that_can_blank_a_cell_files_it.py`
    # holds it on the branch that still earns it.
    mixed = M.provenance("sbp", 10, made)
    assert mixed["n_created_by_the_app"] == 4
    assert mixed["n_blank_in_the_file"] == 6
    assert "came with the file" in mixed["sentence"]
    assert "set_impossible_missing" in mixed["sentence"], (
        "the note does not say which pass made the blanks")


# ═══════════ (b) all three instincts, and which are built ═══════════════════

@pytest.mark.parametrize("fixture,lens,target,person", SHAPES)
def test_the_card_carries_all_three_instincts_and_says_which_are_built(
        client, fixture, lens, target, person):
    """The shelf is never shortened. Two routes are built and the third is
    named with the row it belongs to, rather than being absent — an absence
    would say that distrusting the column is not a thing you may want."""
    block = _repairable_block(client, _project(client, fixture, lens, target))
    routes = {r["id"]: r for r in block["routes"]}
    assert set(routes) == {"set_to_missing", "exclude_the_rows",
                           "mark_the_column_corrupted"}
    assert routes["set_to_missing"]["built"] is True
    assert routes["exclude_the_rows"]["built"] is True
    assert routes["mark_the_column_corrupted"]["built"] is False
    assert "GUIDED-096" in routes["mark_the_column_corrupted"]["not_built_reason"]
    assert routes["mark_the_column_corrupted"]["decision"] is None, (
        "an unbuilt route ships a decision payload, so a client would post it "
        "and get an error instead of the reason it does not exist")
    assert block["withheld"] is None


def test_a_column_whose_reading_is_in_doubt_says_why_it_has_no_routes(client):
    """An empty list with no explanation asserts that nothing can be done.
    `glucose` on `clinical_labs.csv` reads as a unit or coding problem, where
    repairing entries would delete real data and leave the reading wrong."""
    pid = _project(client, "clinical_labs.csv", "clinical", "readmitted")
    report = client.get(f"/project/{pid}/evidence/plausibility").json()
    suspect = [b for b in report["impossible"] if b["whole_column_suspect"]]
    assert suspect, "no suspect-reading column on this fixture"
    for block in suspect:
        assert block["routes"] == []
        assert block["withheld"] and "reading" in block["withheld"], (
            f"{block['column']} offers nothing and says nothing about why")


@pytest.mark.parametrize("fixture,lens,target,person", SHAPES)
def test_the_exclude_the_rows_route_can_be_taken_and_changes_n(
        client, fixture, lens, target, person):
    """**The routing, observed as a consequence rather than as an import.**

    The route's own `decision` payload is posted — with the one field the app
    refuses to write filled in by this test standing in for the user — and what
    is asserted is that N changed and that participant flow carries the count
    and the reason. `apply_plausibility_filter` is the Streamlit door's
    implementation of the same idea and is NOT what this drives; §04's
    `set_eligibility` is the Guided door's, and it was already correct and
    unreachable from this card.
    """
    pid = _project(client, fixture, lens, target)
    block = _repairable_block(client, pid)
    route = next(r for r in block["routes"] if r["id"] == "exclude_the_rows")
    n_before = client.get(f"/project/{pid}").json()["n_rows"]

    grain = client.post(f"/project/{pid}/decision",
                        json={"kind": "set_grain",
                              "payload": {"answer": "people_repeat",
                                          "column": person}})
    assert grain.status_code == 200, grain.text

    body = copy.deepcopy(route["decision"])
    body["payload"][route["typed"]["field"]] = (
        "Values outside the survivable range are entry errors and those visits "
        "are not analyzable.")
    answer = client.post(f"/project/{pid}/decision", json=body)
    assert answer.status_code == 200, answer.text

    after = client.get(f"/project/{pid}").json()
    assert after["n_rows"] == n_before - int(block["n_flagged"]), (
        f"N went {n_before} -> {after['n_rows']} and {block['n_flagged']} rows "
        f"were flagged; an eligibility criterion that does not change N is not "
        f"one")
    recorded = [d for d in after["decisions"] if d["kind"] == "set_eligibility"]
    assert recorded, "the exclusion left no record"
    flow = recorded[-1]["payload"]
    assert flow["n_before"] == n_before
    assert flow["n_excluded"] == int(block["n_flagged"])
    assert flow["n_after"] == after["n_rows"]
    assert body["payload"]["reason"] in recorded[-1]["text"], (
        "participant flow reports the count and not the reason")


@pytest.mark.parametrize("fixture,lens,target,person", SHAPES)
def test_the_app_will_not_write_the_reason_participant_flow_reports(
        client, fixture, lens, target, person):
    """The route ships `reason: ""` on purpose, and says which field the user
    owns. A reason the app invented would be a methods sentence nobody wrote,
    and `build_criterion` refuses it — so the payload names the gap rather than
    filling it or 400ing without explanation."""
    pid = _project(client, fixture, lens, target)
    block = _repairable_block(client, pid)
    route = next(r for r in block["routes"] if r["id"] == "exclude_the_rows")
    assert route["decision"]["payload"]["reason"] == ""
    assert route["typed"]["field"] == "reason"
    assert route["typed"]["prompt"] == E.ELIGIBILITY_NEEDS_A_REASON

    client.post(f"/project/{pid}/decision",
                json={"kind": "set_grain",
                      "payload": {"answer": "people_repeat",
                                  "column": person}})
    answer = client.post(f"/project/{pid}/decision", json=route["decision"])
    assert answer.status_code == 400
    assert "reason" in answer.text and "participant flow" in answer.text.lower()


@pytest.mark.parametrize("fixture,lens,target,person", SHAPES)
def test_the_two_built_routes_do_different_things_to_the_same_rows(
        client, fixture, lens, target, person):
    """The distinction the card exists to draw, measured. Setting to missing
    blanks N cells and keeps every other value on those rows; excluding removes
    the rows entirely. A card that offered one of these as the other would be
    the governing rule failing at the decision the user is standing on."""
    block_column = None
    n_rows, n_flagged = {}, None
    for route_id in ("set_to_missing", "exclude_the_rows"):
        pid = _project(client, fixture, lens, target)
        block = _repairable_block(client, pid)
        block_column, n_flagged = block["column"], int(block["n_flagged"])
        route = next(r for r in block["routes"] if r["id"] == route_id)
        body = copy.deepcopy(route["decision"])
        if route_id == "exclude_the_rows":
            client.post(f"/project/{pid}/decision",
                        json={"kind": "set_grain",
                              "payload": {"answer": "people_repeat",
                                          "column": person}})
            body["payload"]["reason"] = "Entry errors; not analyzable visits."
        answer = client.post(f"/project/{pid}/decision", json=body)
        assert answer.status_code == 200, answer.text
        n_rows[route_id] = client.get(f"/project/{pid}").json()["n_rows"]

    assert n_rows["set_to_missing"] - n_rows["exclude_the_rows"] == n_flagged, (
        f"the two routes moved N the same way on `{block_column}`, so the card "
        f"is drawing a distinction the app does not make: {n_rows}")


# ═══════════ the consumer, which landed in the same commit ══════════════════
#
# This carried `xfail(strict=True)` while it was written — the honest form of
# *a capability ships with its consumer, or with a FAILING test naming the one
# it lacks* — because the page is serialized through one writer and the agent
# that built the server half could not make the edit. The reader landed with it
# and the marker came off in the same change, which is the whole point of the
# marker being strict: it would have XPASSed and failed if anyone had forgotten.
def test_the_page_renders_the_routes_the_server_composes(client):
    """`AGENT_ONBOARD.md` §07 trap #6, declared rather than discovered later.

    A capability ships with its consumer or with a failing test naming the
    missing consumer. This is the second form: it names the file, the two
    attributes that are there, and the one that is not.
    """
    page = (Path(__file__).resolve().parent / "web" / "index.html").read_text(
        encoding="utf-8")
    assert "data-impossible" in page and "data-keep-impossible" in page, (
        "the two controls this is measured against are gone from the page")
    # All three halves, so a partly-applied edit is caught rather than reading
    # as done: the renderer, the delegation selector, and the withheld sentence
    # on the branch that has no routes at all.
    assert "data-plaus-route" in page, (
        "the page renders no control for the routes the plausibility payload "
        "composes, so `exclude_the_rows` and the unbuilt third route are on "
        "the wire and invisible")
    assert "plausRoutes(b)" in page, "the routes renderer is not called"
    assert "b.withheld" in page, (
        "a column whose reading is in doubt renders no routes and no reason "
        "for having none")

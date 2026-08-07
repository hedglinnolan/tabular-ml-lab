"""L52-C — the reporting checklist, and the column that depends on nothing.

`DOMAIN_SCIENCE.md` §01.7 calls the checklist *"a checklist-shaped artifact with
two column types: what the app knows, and what it must ask."* The first waits on
L53. **The second waits on nothing**, and this file is what makes that claim
checkable rather than asserted.

## What is actually guarded here

1. **No item is invented.** Every one carries an exact run of words from
   `research/CLINICAL_SURVEY_PACK.md` §A6 and the test resolves each against the
   file. This is the gate the adjudicator named, and it is the only one that
   could not be satisfied by a plausible-sounding table.
2. **The fill-states are §09's four**, resolved against
   `research/NUTRITION_PACK.md`, not a boolean and not a local invention.
3. **Nothing renders blank or `None`.** `GUIDED-179` is the row where this exact
   surface put Python `None` in front of a user; an unfilled cell says it is
   unfilled and says why.
4. **The gap is stated.** Twelve rows under the name of a twenty-seven-item
   instrument is a completeness claim unless the artifact says otherwise —
   `GUIDED-195`'s rule, arriving at a table instead of a list.
5. **It reaches the Report step.** A capability ships with its consumer.

## What is NOT covered, and it is the loop's own boundary

**Auto-population.** Every `auto_filled` is `None` by design, so the tests below
assert the ABSENCE is honest rather than that any text is right. When L53 fills
the column, `test_nothing_renders_as_a_blank_or_a_none` keeps working and
`test_the_artifact_says_auto_population_is_not_built` is the one that must go
red — it is written to fail loudly the day the premise changes, rather than
passing quietly over a column that started working.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "docs" / "turbotab" / "research"
DATA = Path(__file__).resolve().parent / "sample_data"


def _normalized(text: str) -> str:
    """Markdown emphasis and line wrapping are not content.

    §A6 wraps mid-phrase and bolds fragments, so *"handling of missing data
    with the mechanism discussed"* is split across a newline with a `**` inside
    it. Normalizing both is what lets a trace be an EXACT run of words rather
    than a fuzzy match — and a fuzzy match is what would let an invented item
    through.
    """
    return re.sub(r"\s+", " ", text.replace("**", "").replace("*", ""))


# ── 1 · nothing is invented ──────────────────────────────────────────────────

def test_every_item_traces_to_the_pack():
    """The gate. Each item's `traces_to` must appear in §A6, verbatim."""
    from turbotab import reporting_checklist as CL

    pack = _normalized((RESEARCH / "CLINICAL_SURVEY_PACK.md").read_text(
        encoding="utf-8"))
    # THE POSITIVE CONTROL, `GUIDED-045`: this is an all-must-be-present
    # assertion, and it would pass loudest on an empty ITEMS list.
    assert len(CL.ITEMS) >= 12, (
        f"only {len(CL.ITEMS)} items — the assertion below is satisfied by an "
        f"empty checklist, which is what makes this line load-bearing")
    assert "TRIPOD+AI" in pack and "27-item checklist" in _normalized(pack), (
        "§A6 is not where this test thinks it is; the trace resolution below "
        "would then be checking a file that says nothing")

    orphans = [i.key for i in CL.ITEMS
               if not i.traces_to or _normalized(i.traces_to) not in pack]
    assert not orphans, (
        f"these items do not resolve to a run of words in {CL.SOURCE}: "
        f"{orphans}. An item written from recollection of the BMJ paper is "
        f"domain science nobody sourced, which is the one thing this project "
        f"does not do — read the section and quote it, or do not ship the row")


def test_the_trace_check_would_actually_fail():
    """The detector, on a planted item.

    `test_every_item_traces_to_the_pack` reports orphans, and an empty orphan
    list is equally consistent with twelve sourced items and a matcher that
    matches anything. This plants one that must not resolve.
    """
    pack = _normalized((RESEARCH / "CLINICAL_SURVEY_PACK.md").read_text(
        encoding="utf-8"))
    invented = "calibration slope reported to three decimal places in the abstract"
    assert _normalized(invented) not in pack, (
        "the planted phrase is genuinely in the pack, so it cannot show that "
        "the matcher rejects anything")


def test_the_fill_states_are_the_packs_four():
    """§09's vocabulary, resolved — and specifically not a boolean.

    *the app detects it and you confirm* is a real third state. A two-value
    column either claims a fact the user never agreed to or asks for one the
    app already has.
    """
    from turbotab import reporting_checklist as CL

    pack = _normalized((RESEARCH / "NUTRITION_PACK.md").read_text(encoding="utf-8"))
    for state in CL.FILL_STATES:
        assert state in pack, (
            f"the fill-state {state!r} is not in {CL.FILL_STATE_SOURCE}, so it "
            f"is this module's invention rather than the field's vocabulary")
    assert len(CL.FILL_STATES) == 4, CL.FILL_STATES
    used = {i.fills for i in CL.ITEMS}
    assert used <= set(CL.FILL_STATES), f"unknown fill-state in use: {used}"
    assert len(used) >= 3, (
        f"only {len(used)} of the four states are used: {used}. A checklist "
        f"where every item is `app` or every item is `user` has collapsed the "
        f"distinction §09's table exists to make")


# ── 2 · the column that depends on nothing ───────────────────────────────────

def test_every_item_says_what_it_needs_from_the_author():
    """The deliverable of this loop, asserted as content rather than presence.

    §01.7's second column type is *what it must ask*. An item that cannot say
    what it wants is an item nobody thought about, so a placeholder passes a
    presence check and fails this one.
    """
    from turbotab import reporting_checklist as CL

    thin = [i.key for i in CL.ITEMS if len((i.needs_from_you or "").split()) < 20]
    assert not thin, (
        f"{thin} ask the author for something in fewer than twenty words. This "
        f"is the column that depends on nothing and is therefore the column "
        f"with no excuse for being thin")
    # And it must be about THIS item rather than a house sentence repeated.
    texts = [i.needs_from_you for i in CL.ITEMS]
    assert len(set(texts)) == len(texts), (
        "two items ask for the same thing word for word, which means at least "
        "one of them is boilerplate")


# ── 3 · nothing renders as a blank or a None ─────────────────────────────────

#: `GUIDED-179` is about a Python `None` reaching a screen — *"Expected
#: analysis N=None"* — and not about the English word. The first draft of the
#: check below was a bare `"None" not in value` and it fired on *"None of them
#: is a property of the data"*, which is the matcher-fires-on-prose failure this
#: project keeps meeting one level down from wherever it is looking. A rendered
#: `None` appears either AS the whole value or after an assignment, a colon or
#: an opening bracket; English never does.
_RENDERED_NONE = re.compile(r"[=:(\[]\s*None\b|\bNone\s*[,)\]]")


def _leaks_a_none(value: str) -> bool:
    return value.strip() == "None" or bool(_RENDERED_NONE.search(value))


def test_the_none_detector_knows_prose_from_a_leak():
    """The control for the check below, because it already got this wrong once."""
    assert _leaks_a_none("None")
    assert _leaks_a_none("Expected analysis N=None, abstract N=None")
    assert _leaks_a_none("where addressed: None")
    assert not _leaks_a_none("All five. None of them is a property of the data.")
    assert not _leaks_a_none("None but the author can say what the setting was.")


def test_nothing_renders_as_a_blank_or_a_none():
    """`GUIDED-179`, and it was found on this exact surface.

    The reviewer checklist rendered Python `None` at a user. Every cell here is
    either content or a NAMED absence carrying its reason.
    """
    from turbotab import reporting_checklist as CL

    out = CL.render()
    assert out["rows"], "the artifact rendered no rows at all"
    for row in out["rows"]:
        for column in ("item", "where_addressed_text", "auto_filled_text",
                       "needs_from_you", "fills", "source"):
            value = row.get(column)
            assert isinstance(value, str) and value.strip(), (
                f"{row['key']}.{column} renders as {value!r}")
            assert not _leaks_a_none(value), (
                f"{row['key']}.{column} puts a rendered `None` on a screen: "
                f"{value!r} — `GUIDED-179` exactly")
        if not row["auto_filled"]:
            assert row["auto_filled_text"] == CL.NOT_YET_FILLED
            assert row["not_filled_because"], (
                f"{row['key']} is unfilled and does not say WHY. A blank cell "
                f"reads as a rendering fault; a named absence reads as the app "
                f"declining to state a quantity it does not have")


def test_the_artifact_says_how_many_items_it_is_not_showing():
    """`GUIDED-195` at a table. Twelve rows under a twenty-seven-item name."""
    from turbotab import reporting_checklist as CL

    cov = CL.render()["coverage"]
    assert cov["instrument_items"] == 27, cov
    assert cov["enumerated_here"] == len(CL.ITEMS)
    assert cov["not_yet_enumerated"] == 27 - len(CL.ITEMS)
    assert str(cov["not_yet_enumerated"]) in cov["why"], (
        "the gap is computed and not said — the number has to be IN the "
        "sentence a reader sees, or the table still looks complete")
    assert cov["how_to_close"], "a stated gap with no way to close it is a shrug"


def test_the_artifact_says_auto_population_is_not_built():
    """Written to go RED the day L53 lands, which is the point.

    A test that passes quietly over a column that started working is how a
    deliberate absence becomes an accidental one.
    """
    from turbotab import reporting_checklist as CL

    out = CL.render()
    assert out["auto_population_built"] is False, (
        "auto-population reports itself as built. If L53 has landed, this "
        "test is the one that says so — fill the `where addressed` and "
        "`auto-filled text` columns and delete this")
    assert all(r["auto_filled"] is None for r in out["rows"])


def test_probast_is_a_sentence_and_not_a_second_checklist():
    """§A6 carries PROBAST; `DOMAIN_SCIENCE.md` §05 item 6 says one instrument."""
    from turbotab import reporting_checklist as CL

    out = CL.render()
    note = out["probast"]["note"]
    assert "20 signaling questions" in note and "4 domains" in note, note
    assert "systematic review" in note, (
        "the line worth surfacing is the one about what happens to the paper "
        "later, and it is missing")
    pack = _normalized((RESEARCH / "CLINICAL_SURVEY_PACK.md").read_text(
        encoding="utf-8"))
    assert _normalized("this is the instrument that will be applied to it") in pack
    assert "rows" not in out["probast"], (
        "PROBAST has grown rows. §05 item 6 seeds ONE checklist; a second one "
        "is a decision, not a drift")


# ── 4 · it reaches the Report step ───────────────────────────────────────────

def _project(fixture="clinical_labs.csv", target="readmitted"):
    from fastapi.testclient import TestClient

    from turbotab import api

    client = TestClient(api.app)
    with (DATA / fixture).open("rb") as handle:
        pid = client.post("/project", files={
            "file": (fixture, handle, "text/csv")}).json()["id"]
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_target", "payload": {"column": target}})
    return client, pid


def test_the_route_serves_it_and_404s_for_a_project_that_is_not_there():
    """Half a consumer. The route exists and it is about a real project."""
    client, pid = _project()
    got = client.get(f"/project/{pid}/checklist")
    assert got.status_code == 200, got.text[:300]
    body = got.json()
    assert body["instrument"] == "TRIPOD+AI"
    assert len(body["rows"]) == 12
    assert client.get("/project/not-a-real-id/checklist").status_code == 404, (
        "the route serves a checklist for a study that does not exist, which "
        "means it never looked at the project and the id is decoration")


def test_the_checklist_reaches_the_report_step():
    """`GUIDED-119`'s rule: a capability ships with its consumer.

    Driven through the page rather than asserted about the HTML, because
    *the server composes it* and *the interface renders it* are trap #6's two
    halves and only the second one is the feature.
    """
    from turbotab import pageharness as PH

    if not PH.available():
        pytest.skip("no JS engine on this machine")

    client, pid = _project()
    routes = {f"/project/{pid}": client.get(f"/project/{pid}").json()}
    for path in ("interview?step=data", "interview?step=explore",
                 "interview?step=features", "interview?step=preprocess",
                 "capabilities", "features", "recipes", "preprocess", "figures",
                 "draft", "manuscript", "models", "training", "instability",
                 "explain", "sensitivity", "checklist",
                 "evidence/plausibility", "evidence/missingness"):
        got = client.get(f"/project/{pid}/{path}")
        routes[f"/project/{pid}/{path}"] = got.json() if got.status_code == 200 else {}

    out = PH.run(
        "for (var i = 0; i < 12; i++) "
        "  await new Promise(function(r){ setTimeout(r, 0); });\n"
        "__emit({report: __harness.html('reportBox')});",
        routes=routes, search=f"?project={pid}")

    report = out["report"] or ""
    assert report, "the Report step rendered nothing at all"
    assert "TRIPOD+AI checklist" in report, (
        "the checklist is composed by the server and never reaches the "
        "interface — trap #6, and the capability has no consumer")
    assert "Needs your input" in report, (
        "the four-column head is not rendered, so the column this loop built "
        "is not the one on screen")
    from turbotab import reporting_checklist as CL
    assert CL.ITEMS[0].needs_from_you[:40] in report, (
        "the header rendered and the asking column did not, which is the "
        "table shape without the content")

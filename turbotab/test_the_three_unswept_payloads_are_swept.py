"""L43-A1 · the three payloads L42-B named and did not sweep.

L42's report said it plainly: `/features`, `/recipes` and `/preprocess` are
**reachable today** and the Explore-step drive did not open those steps. It
called that a cost decision rather than a shape-of-the-problem decision and
named it the part of B it was least comfortable with. Three reachable steps
unswept is a hole the same shape as the one B was built to close.

**Why they need a different journey.** The other six payloads are fetched at
the Explore step, where B's fixture stops. These three belong to the Features
and Preprocess steps, and the page reveals a step by the project reaching it —
the map buttons are indicators, not controls. So this drives the whole
interview through the seal first, and sweeps from there.

Same instrument as B, deliberately: `fieldsweep` tags every field with a unique
sentinel, drives the page, and re-confirms every miss by group-negative
bisection against a clean baseline. The rigor is B's rigor because the question
is B's question one step further along.

**The disposition rule is B's too** — rendered, or exempt with a named reader,
or a ledger row. Nothing here is excused with a shrug.
"""
from __future__ import annotations

import pathlib

import pytest

from turbotab import fieldsweep as FS

DATA = pathlib.Path(__file__).resolve().parent / "sample_data"
PAGE = pathlib.Path(__file__).resolve().parent / "web" / "index.html"

#: The three, and the step each belongs to. Named here rather than derived
#: because the point is to close a NAMED gap — a derived list would have
#: included them at L42 and did not.
LATE_ROUTES = ("/features", "/recipes", "/preprocess")

#: One lens. B sweeps two and `GUIDED-097`'s rule wants at least two of
#: different target shape; this pays one. **Said rather than implied**: the
#: survey fixture reaches the seal without the repeated-measures chain, and
#: the clinical one needs it, so a second lens here is a second journey rather
#: than a second fixture. That is the shape not covered.
FIXTURE, LENS, TARGET = "survey_sentinels.csv", "survey", "age"


@pytest.fixture(scope="module")
def swept_late():
    """The journey through the seal, then the sweep."""
    from fastapi.testclient import TestClient

    from turbotab import api

    ids = FS.container_ids(PAGE.read_text(encoding="utf-8"))
    client = TestClient(api.app)
    with open(DATA / FIXTURE, "rb") as handle:
        pid = client.post("/project", files={
            "file": (FIXTURE, handle, "text/csv")}).json()["id"]

    def decide(kind, payload):
        r = client.post(f"/project/{pid}/decision",
                        json={"kind": kind, "payload": payload})
        assert r.status_code == 200, (kind, r.text[:250])

    for kind, payload in (
            ("set_lens", {"lens": [LENS]}),
            ("set_target", {"column": TARGET}),
            ("set_purpose", {"answer": "prediction"}),
            ("set_grain", {"answer": "one_row_per_person"}),
            ("set_eligibility", {"answer": "everyone"}),
            ("seal", {"fraction": 0.25})):
        decide(kind, payload)

    def get(tail):
        return client.get(f"/project/{pid}{tail}").json()

    # The six the page fetches anyway, plus the three. The six are here
    # because the page needs them to RENDER — dropping them kills the
    # controller on `plan.questions.filter`, which is how the first attempt
    # at this failed.
    routes = {
        f"/project/{pid}": get(""),
        f"/project/{pid}/interview?step=data": get("/interview?step=data"),
        f"/project/{pid}/interview?step=explore": get("/interview?step=explore"),
        f"/project/{pid}/evidence/missingness": get("/evidence/missingness"),
        f"/project/{pid}/evidence/plausibility": get("/evidence/plausibility"),
        f"/project/{pid}/capabilities": get("/capabilities"),
    }
    for tail in LATE_ROUTES:
        routes[f"/project/{pid}{tail}"] = get(tail)
    return routes, pid, ids, FS.sweep(routes, pid, ids)


def test_the_three_payloads_are_not_empty_on_this_project(swept_late):
    """First, because everything else is vacuous otherwise.

    A payload that is empty at this step would make its whole sweep a
    statement about nothing, and `Sweep.empty` would swallow it silently —
    which is exactly the reason B holds null fields out of `unread` and says
    so.
    """
    routes, pid, _ids, _sweep = swept_late
    for tail in LATE_ROUTES:
        body = routes[f"/project/{pid}{tail}"]
        assert body, f"{tail} is empty at the sealed step"
        assert len(str(body)) > 500, (
            f"{tail} carries {len(str(body))} characters, which is too little "
            f"for the sweep below to be a claim about anything")


def test_the_enumeration_reaches_the_three_late_routes(swept_late):
    """The sweep has to have looked at them. An instrument that silently
    swept six and reported on nine is the silent-truncation failure §10 names.
    """
    _routes, pid, _ids, sweep = swept_late
    reached = {r for r in sweep.routes_swept}
    for tail in LATE_ROUTES:
        assert f"/project/{pid}{tail}" in reached, (
            f"{tail} is not in the sweep's own list of routes swept")


def test_the_three_payloads_are_reported_with_their_counts(swept_late, capsys):
    """The counts L42 owed and did not pay.

    Reported per route rather than in aggregate, because the whole reason
    these three were named separately is that they belong to steps the
    Explore drive never opened — an aggregate would hide which of the three
    is the one nobody reads.
    """
    _routes, pid, _ids, sweep = swept_late
    by_route = {}
    for tail in LATE_ROUTES:
        route = f"/project/{pid}{tail}"
        fields = [f for f in sweep.fields if f.route == route]
        empty = [f for f in sweep.empty if f.route == route]
        by_route[tail] = {
            "fields": len(fields),
            "reaching": sum(1 for f in fields if f.reaches),
            "unread": sum(1 for f in fields if f.reaches is False),
            "empty": len(empty),
        }

    shapes = {}
    for (route, shape), verdict in sweep.shapes().items():
        for tail in LATE_ROUTES:
            if route.endswith(tail):
                shapes.setdefault(tail, {"all": 0, "none": 0, "partial": 0})
                shapes[tail][verdict["verdict"]] += 1

    with capsys.disabled():
        print("\n  ── L43-A1 · the three payloads L42-B named and skipped ──")
        print(f"  fixture: {FIXTURE} ({LENS} lens, target {TARGET}), sealed")
        for tail in LATE_ROUTES:
            r, s = by_route[tail], shapes.get(tail, {})
            print(f"  {tail}")
            print(f"    fields enumerated        {r['fields']}")
            print(f"    reaching a person        {r['reaching']}")
            print(f"    unread                   {r['unread']}")
            print(f"    empty on this project    {r['empty']}"
                  "   (held out of unread, as B does)")
            print(f"    path shapes              all={s.get('all', 0)} "
                  f"partial={s.get('partial', 0)} none={s.get('none', 0)}")
        print("  NOT COVERED:")
        print("    - one lens, not two; the second needs a different journey")
        print("      (the repeated-measures chain), not a different fixture")
        print("    - post-Train payloads, which still need a fitted model")
        print("    - the Features and Preprocess steps are FETCHED here and the")
        print("      page is driven at the sealed step; a drive that pressed")
        print("      those steps' own controls would be a wider claim")

    assert sum(r["fields"] for r in by_route.values()) > 100, (
        "the three payloads enumerated almost nothing between them, so the "
        "counts above are not a measurement")


def test_the_instrument_still_distinguishes_read_from_unread_here(swept_late):
    """B's positive control, re-run at this step.

    A sweep that answered *unread* to everything would report a spectacular
    finding about three payloads and measure nothing. Carried over rather than
    assumed to hold: this is a different journey position and the page renders
    a different set of sections.
    """
    _routes, _pid, _ids, sweep = swept_late
    assert sweep.reaching, "nothing at all reaches a person at the sealed step"
    assert sweep.unread, "everything reaches a person, which is not credible"

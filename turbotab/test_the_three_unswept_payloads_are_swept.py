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


def _family(route: str, shape: str) -> tuple:
    """`(route tail, top-level shape)` — L42-B's own function, not a second one.

    Two vocabularies for one question are two things to drift, which is the
    argument L42-B makes for reusing `READERS`. This reuses its grouping.
    """
    tail = route.rsplit("/", 1)[-1].split("?")[0]
    return (tail, shape.split("[")[0].split(".")[0])


#: Families that are fully dark here and have a KNOWN reader, with the reason.
#: `GUIDED-197`.
DECLARED = {
    ("preprocess", "columns"): (
        "`prepStrategyHTML` reads `col.strategies` on every render — it is the "
        "list the strategy buttons are built from. It reports dark for the "
        "reason L42-B's own `explore_stack` entry gives: this instrument "
        "detects a read by stamping a SENTINEL VALUE and looking for it in the "
        "DOM, and these entries are used as LOOKUP KEYS — `prepStrategy(branch, "
        "key)` — so a sentinel key resolves to nothing and renders nothing. A "
        "field read as a key is invisible to a sweep that watches for values."),
    ("features", "column_levels"): (
        "`GUIDED-198`'s new payload, and **the gate caught it on the first run "
        "after it landed** — which is what this file exists for. "
        "`featParamHTML` reads `FEAT.column_levels` to build the "
        "`ordinal_declared` order control, and `featColumnsFor` reads it again "
        "to decide which columns that transform may be offered over. It reports "
        "dark for the reason `preprocess.columns` does: the entries are read as "
        "LOOKUP KEYS and as `<option value=…>`, so a sentinel value resolves to "
        "nothing and renders nothing. The reader is named, known and tested — "
        "`test_the_transform_that_needs_a_parameter_gets_a_control_for_it.py"
        "::test_the_order_offered_is_the_chosen_columns_own_levels`."),
    ("preprocess", "receipt"): (
        "`receipt.n_applied_now`, `n_attested`, `n_deferred`, `n_left`, "
        "`n_mixed` and `n_unanswered` are the arithmetic the server composes "
        "`receipt.headline` FROM, and `prepReceiptHTML` renders the headline. "
        "They travel beside the sentence so a consumer never has to parse "
        "prose for a count — trap #7's rule kept, which is the opposite of a "
        "gap. The sentence is read; the numbers under it are its structured "
        "form."),
}

#: Families that are fully dark and for which NO reader was found. Filed, not
#: excused. L42-B's words: *a reason I cannot substantiate is the shrug the
#: table exists to stop.*
FILED = {
    # `/features`, and the four the gate found on its very first run — which is
    # the whole argument for the gate. `row_local` and `deferred` are dark on
    # ONE leaf each, `needs`: the parameter descriptor no control reads, which
    # is `GUIDED-198` exactly. When that lands they stop being dark and the
    # `stale` half of the gate below says so, rather than letting this table go
    # on excusing something that has since been wired.
    ("features", "row_local"): "GUIDED-198",
    ("features", "deferred"): "GUIDED-198",
    ("features", "all_columns"): "GUIDED-198",
    ("features", "identifiers"): "GUIDED-203",
    ("preprocess", "strategies"): "GUIDED-190",
    ("recipes", "operations"): "GUIDED-202",
    ("recipes", "pack_defaults"): "GUIDED-202",
    ("recipes", "n_choices_suppressed"): "GUIDED-202",
    ("recipes", "n_rows_seen"): "GUIDED-202",
    ("recipes", "n_rows_withheld"): "GUIDED-202",
    # `DRIVE-045`, and it joins its two siblings rather than being excused
    # separately. `n_rows_seen` narrowed to the analysis population at `L62`,
    # which silently turned "withheld" from *sealed* into *sealed or unusable*;
    # this is the split that keeps the two reasons apart. It is dark for the
    # same reason they are and under the same row.
    ("recipes", "n_rows_without_an_outcome"): "GUIDED-202",
    # `DRIVE-051`, and these two join the same family for the same reason. The
    # three counts above were served as a breakdown of the table and did not
    # partition it: `n_rows_without_an_outcome` counts the whole frame, so it
    # overlaps `n_rows_withheld` wherever a sealed row lost its outcome.
    # `n_rows_available_without_an_outcome` is the disjoint remainder and
    # `n_rows_withheld_without_an_outcome` is the overlap, served as a number
    # rather than left for a reader to infer from an arithmetic failure.
    #
    # THEY ARE ADDED RATHER THAN RENAMED INTO, and that is deliberate. A rename
    # trips BOTH assertions below — `undeclared` on the new key and `stale` on
    # the old one, because `_family` keys on the raw payload key — so the table
    # would have to move in two directions in one commit. Keeping
    # `n_rows_without_an_outcome` with the meaning its name already states
    # avoids inventing a second true-but-differently-scoped field under the
    # same word, which is the confusion `DRIVE-045` was filed for.
    ("recipes", "n_rows_available_without_an_outcome"): "GUIDED-202",
    ("recipes", "n_rows_withheld_without_an_outcome"): "GUIDED-202",
}


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


# ═══════════ THE GATE, WHICH THIS FILE DID NOT HAVE ═══════════════════════════

def test_every_dark_family_here_names_a_reader_or_a_row(swept_late, capsys):
    """`GUIDED-197`. A sweep that reports rather than fails is a document.

    This file enumerated three payloads, printed 226 fields and 39 unread for
    `/features` alone, and **asserted nothing about any of them** — so
    `GUIDED-171` (a two-column formula previewing one operand, with `inputs`
    and `all_columns` served and unread) sat inside its output for four loops
    and the file stayed green the whole time. L42-B's
    `test_every_unread_family_names_a_reader_or_a_row` is the pattern this was
    missing, and it is applied here rather than reinvented: three dispositions
    and no fourth.

    **A count is not a guard.** The distinction is `GUIDED-180`'s exactly, one
    surface over — `unchecked` and `checked and found to be fine` were rendering
    as the same output, which was *a number printed to stdout*.
    """
    _routes, _pid, _ids, sweep = swept_late
    dark = {_family(route, shape)
            for (route, shape), v in sweep.shapes().items()
            if v["verdict"] == "none"
            and any(route.endswith(t) for t in LATE_ROUTES)}
    undeclared = sorted(f for f in dark if f not in DECLARED and f not in FILED)
    stale = sorted((set(DECLARED) | set(FILED)) - dark)

    with capsys.disabled():
        print("\n  ── L49-A2 · the late sweep is a gate now ──")
        print(f"  families fully dark at this step    {len(dark)}")
        print(f"    with a named reader (DECLARED)    "
              f"{len([f for f in dark if f in DECLARED])}")
        print(f"    filed against a row (FILED)       "
              f"{len([f for f in dark if f in FILED])}")
        print(f"    neither                           {len(undeclared)}")
        print(f"  table entries no longer dark        {len(stale)}")
        for f in stale:
            print(f"      {f}")
        # THE LEAVES, not only the families. A family is dark because of a
        # particular field, and a reader who gets only the family name has to
        # re-run a twelve-minute sweep to learn which — which is most of why
        # the counts this file printed for four loops were never acted on.
        print("  the dark leaves, per family:")
        for (route, shape), v in sorted(sweep.shapes().items()):
            if v["verdict"] != "none":
                continue
            if not any(route.endswith(t) for t in LATE_ROUTES):
                continue
            print(f"      {_family(route, shape)[0]:<12} {shape}")

    assert not undeclared, (
        "these families are composed by the server at the sealed step and read "
        "by nothing in the Guided door, and neither a reader nor a row is "
        f"named for them:\n  {undeclared}\n"
        "Name the reader in DECLARED, or file a row and put it in FILED. A "
        "printed count is not a disposition.")
    assert not stale, (
        f"{stale} are named in DECLARED/FILED and are no longer dark. A table "
        f"that keeps excusing what has since been wired reports coverage "
        f"nothing provides — remove them.")


def test_the_gate_has_something_to_gate(swept_late):
    """The positive control, and this file is the reason it is written down.

    An empty `dark` set makes the gate above pass on any page, including an
    emptied one. It is the same failure the gate exists to fix, one level up.
    """
    _routes, _pid, _ids, sweep = swept_late
    verdicts = [v["verdict"] for (route, _s), v in sweep.shapes().items()
                if any(route.endswith(t) for t in LATE_ROUTES)]
    assert verdicts, "the sweep produced no verdicts at all for the late routes"
    assert "all" in verdicts, (
        "nothing at these three routes reaches a person, which is not credible "
        "and means the instrument is reporting darkness rather than measuring it")
    assert "none" in verdicts, (
        "every field at these three routes reaches a person. That would be a "
        "remarkable result and it is more likely the sentinel stopped working")

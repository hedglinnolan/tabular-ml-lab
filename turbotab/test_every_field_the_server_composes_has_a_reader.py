"""L42-B — the field granularity. `turbotab/fieldsweep.py` is the instrument.

The door guards one granularity and it is the coarse one.
`test_every_server_surface_names_its_reader` asserts every **route** is fetched
or declared unread with its reader named. It could not see either of L41's two
`critical` reachability defects, because `/project/{project_id}` *is* fetched on
every render — so the route "has a reader" while a whole `source` class inside
its payload was rendered by nothing.

This extends it downward and **does not replace it**: that check works at its
own level, its exemptions are correctly reasoned, and its twice-written comment
is a record worth keeping.

## What the sweep does, and why the number is what it is

Every field of every payload the door fetches is tagged with a sentinel, the
page is **driven** — bootstrapped, then every control it rendered pressed — and
the DOM is searched. See `fieldsweep`'s own docstring for the two passes and
for why a group negative is sound.

**The reportable unit is the path SHAPE, and it has three verdicts.**
`findings[0].title` and `findings[7].title` are one shape, and collapsing them
to *read* would hide `GUIDED-142` exactly. So a shape is `all`, `none`, or
**`partial`** — and `partial` is the interesting one.

## Three dispositions and no fourth

Every family that CONTAINS a fully-unread shape is either `DECLARED` here with
its reader named, or `FILED` against a ledger row. `READERS` is reused from the
standing check rather than restated, because two vocabularies for one question
are two things to drift.

**"Contains" rather than "is", and the docstring says so because the table
would otherwise read as a stronger claim than it is.** `project.findings` flags
while the door renders findings, because its per-finding `params` bookkeeping
reaches nobody.

**Families rather than shapes, and that is a real limit stated rather than
smoothed.** There are 830 shapes and 747 of them are unread; a per-shape
exemption table would be an artifact nobody re-reads, which is the failure mode
`register.py`'s docstring exists because of. A family is the unit somebody can
actually dispose of. What that costs is resolution: a single unread shape inside
an otherwise-read family is invisible here, and `partial` is what catches the
version of that which matters.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from turbotab import fieldsweep as FS

# ONE VOCABULARY, reused. The standing route check already decided what counts
# as a reader, and a second list would be a second thing to drift.
from turbotab.test_the_page_says_what_the_record_says import READERS

DATA = Path(__file__).resolve().parent / "sample_data"
PAGE = Path(__file__).resolve().parent / "web" / "index.html"

#: `GUIDED-097`. Two lenses and two target shapes, because `GUIDED-142` would
#: have passed a sweep written against a single pack — the defect was that a
#: whole `source` class never rendered, and one lens produces one pack's worth
#: of that class.
SWEPT = {
    "clinical, binary target": ("clinical_labs.csv", "clinical", "readmitted"),
    "survey, continuous target": ("survey_sentinels.csv", "survey", "age"),
}

#: NOT SWEPT, said out loud — the payloads this loop did not reach.
#:
#: The loop's own scope note: *narrow the payloads rather than the rigor*. Six
#: payloads are swept properly per lens. These are the ones a driven project
#: also fetches and this sweep does not enumerate, each because it needs a
#: project further down the journey than an Explore-step drive reaches.
NOT_SWEPT = [
    "/figures — needs a fitted model, so the payload is empty at Explore",
    "/manuscript — needs a run; the fields are the export's, not the door's",
    "/explain, /sensitivity, /instability — all post-Train",
    "/features, /recipes, /preprocess — the Features and Preprocess steps, "
    "which the Explore-step drive does not open",
    "/draft — fetched on every render and its fields are draft.py's, which "
    "test_the_manuscript_is_checked already covers at its own granularity",
]

#: Families that CONTAIN a fully-unread shape, with the reader named.
#:
#: **The predicate is "contains", not "is", and the difference matters.**
#: `project.findings` is in the unread set and the door plainly renders
#: findings — it flags because some shape *inside* it, the per-finding
#: `params` and `evidence.claims` bookkeeping, reaches nobody. A family verdict
#: is a pointer to look, not a claim that the family is dark; `partial` and
#: `test_no_finding_class_is_entirely_dark` are what carry the stronger claims.
#:
#: Same shape and the same rule as the standing check's `NOT_READ_BY_THE_DOOR`:
#: there is no fourth option, and a reason that names neither a reader nor a
#: row is not a reason.
DECLARED = {
    ("project", "profile"): (
        "The dataset profile's per-column block. The Guided door renders the "
        "summary rows and the target profile; the remaining per-column "
        "statistics are read by the Streamlit door at `pages/02` and by the "
        "export path, which is why `ml/dataset_profile.py` computes them. "
        "GUIDED-095's rule applies: a field a second door reads is not "
        "unread, it is read elsewhere."),
    ("project", "sample"): (
        "The first rows of the table, rendered by `renderSample` behind the "
        "sample pill — which this sweep presses, so the family is read and the "
        "unread shapes inside it are per-cell values in rows the preview caps. "
        "The full frame is the Streamlit door's, at `pages/01`; the project "
        "payload carries eight rows for the preview and no more."),
    ("project", "readiness"): (
        "Read by the Streamlit door's readiness gate at `pages/02`. The "
        "Guided door asks the same questions through `/interview`, which is "
        "the same relationship `/grain` and `/lens` have in "
        "NOT_READ_BY_THE_DOOR."),
    ("project", "decisions"): (
        "The transcript. Rendered through `/draft`, which composes the "
        "sentences rather than echoing the decision objects — so the "
        "`decisions` array reaches a person as prose and its raw fields do "
        "not. `project payload` carries it for the export path."),
    ("project", "name"): (
        "Duplicated at the top level and inside `profile`. `renderSample` "
        "reads `P.name` behind the sample pill and the profile table prefers "
        "`profile.*` — see GUIDED-146, which is the row for the duplication "
        "rather than for the field."),
    ("project", "created_at"): (
        "Read by the export path; the door shows no timestamps. Carried on "
        "the project payload because the archive round-trips it."),
    ("project", "n_rows"): (
        "GUIDED-146: duplicated inside `profile`, and the page prefers the "
        "profile copy through a ternary whose fallback arm the server's own "
        "payload makes unreachable."),
    ("project", "n_columns"): (
        "GUIDED-146, same as `n_rows` — the profile carries `n_features` and "
        "the page reads that."),
    ("project", "row_identity"): (
        "Read by the Streamlit door and by the archive. The Guided door states "
        "row identity in the sample preview's badge rather than from this "
        "field."),
    ("interview", "next"): (
        "The plan's next-step block. Read by the interview plan itself to "
        "order what it serves; the door renders the questions rather than the "
        "planner's own bookkeeping."),
    ("interview", "project_id"): (
        "Echoed so a response can be matched to its request. Read by the "
        "project payload's own bootstrap, not rendered."),
    ("interview", "step"): (
        "Echoed for the same reason as `project_id` — read by the interview "
        "plan to key its own response. The door already knows which step it "
        "asked for."),
    ("interview", "steps"): (
        "The step map. Read by the interview plan; the door's own step map is "
        "driven by `setMap`, from the project payload."),
    ("interview", "n_asked"): (
        "Counts the plan computes for the routing value check, which is the "
        "export path's measurement rather than a sentence anybody reads."),
    ("interview", "n_offered"): (
        "Same as `n_asked` — the routing value check's measurement, read by "
        "the export path rather than rendered."),
            ("interview", "questions"): (
        "Partly read, and the unread members are questions the plan serves "
        "as ANSWERED or SKIPPED — their `why`, `consumer` and option lists "
        "are not rendered a second time once the answer is on the record. "
        "The `partial` verdict below is what would catch a whole class going "
        "dark, which is what `GUIDED-142` was."),
    ("plausibility", "impossible"): (
        "Rendered by the plausibility pull, which this sweep presses. What "
        "stays unread is the per-entry bookkeeping — `all_rows`, "
        "`reading_evidence`, `scale_factor` — which `ml/card_evidence.py` "
        "computes for the Streamlit door's entry table."),
    ("plausibility", "improbable"): (
        "Same as `impossible` — rendered by the plausibility pull, with the "
        "per-entry bookkeeping read by the Streamlit door's entry table."),
    ("capabilities", "pulls"): (
        "The chip table. `built`, `title`, `endpoint` and "
        "`not_built_reason` render; `label`, `why` and the gate's `limit` and "
        "`n_features` are read by the Streamlit door's palette and by "
        "`/capabilities`' own consumers."),
}

#: The **third** disposition, and it is a ledger row rather than a reason.
#:
#: `DECLARED` above is for families whose reader is known. These are the ones
#: the sweep found and for which **no reader was found** — filed rather than
#: excused, which is the rule and not a fallback. Naming them here rather than
#: writing sixteen invented reasons is the honest form: a reason I cannot
#: substantiate is the shrug the table exists to stop.
FILED = {
    ('project', 'columns'): "GUIDED-147",
    ('project', 'disclosures'): "GUIDED-147",
    ('project', 'features_settled'): "GUIDED-147",
    ('project', 'findings'): "GUIDED-147",
    ('project', 'findings_stale'): "GUIDED-147",
    ('project', 'fingerprint'): "GUIDED-147",
    ('project', 'lens'): "GUIDED-147",
    ('project', 'n_working_rows'): "GUIDED-146",
    ('project', 'preprocess_settled'): "GUIDED-147",
    ('project', 'workflow_mode'): "GUIDED-147",
    ('plausibility', 'n_impossible'): "GUIDED-147",
    ('plausibility', 'n_improbable'): "GUIDED-147",
    ('plausibility', 'n_suspect_columns'): "GUIDED-147",
    ('plausibility', 'reference_version'): "GUIDED-147",
    ('capabilities', 'n_numeric'): "GUIDED-147",
    ('capabilities', 'not_built_reason'): "GUIDED-147",
}


@pytest.fixture(scope="module")
def swept():
    """Every sweep, run once. Two lenses at ~100 s each is the cost of the
    instrument, and running it per test would multiply it by the test count."""
    from fastapi.testclient import TestClient

    from turbotab import api

    ids = FS.container_ids(PAGE.read_text(encoding="utf-8"))
    out = {}
    for label, (fixture, lens, target) in sorted(SWEPT.items()):
        client = TestClient(api.app)
        with open(DATA / fixture, "rb") as handle:
            pid = client.post("/project", files={
                "file": (fixture, handle, "text/csv")}).json()["id"]
        for kind, payload in (("set_lens", {"lens": [lens]}),
                              ("set_target", {"column": target})):
            ok = client.post(f"/project/{pid}/decision",
                             json={"kind": kind, "payload": payload})
            assert ok.status_code == 200, (kind, ok.text[:200])

        def get(tail):
            return client.get(f"/project/{pid}{tail}").json()

        routes = {
            f"/project/{pid}": get(""),
            f"/project/{pid}/interview?step=data": get("/interview?step=data"),
            f"/project/{pid}/interview?step=explore": get("/interview?step=explore"),
            f"/project/{pid}/evidence/missingness": get("/evidence/missingness"),
            f"/project/{pid}/evidence/plausibility": get("/evidence/plausibility"),
            f"/project/{pid}/capabilities": get("/capabilities"),
        }
        out[label] = (routes, pid, FS.sweep(routes, pid, ids))
    return out


def _family(route: str, shape: str) -> tuple:
    tail = route.rsplit("/", 1)[-1].split("?")[0]
    if tail.startswith("interview"):
        tail = "interview"
    elif "/project/" in route and route.count("/") == 2:
        tail = "project"
    return (tail, shape.split("[")[0].split(".")[0])


# ═══════════ THE INSTRUMENT WORKS ═══════════

def test_the_enumeration_is_derived_and_not_a_list(swept):
    """**The check the adjudicator said it would run**: add a field, and it
    must appear. A hand-list is what `register.py`'s docstring exists because
    of — the register was a markdown table until a merge blind-copied an older
    one over it and a section vanished silently."""
    routes, _pid, sweep = next(iter(swept.values()))
    route = next(iter(routes))
    before = {f.path for f in FS.leaves(routes[route], route)}
    grown = dict(routes)
    grown[route] = {**routes[route], "a_field_nobody_added_yet": "x"}
    after = {f.path for f in FS.leaves(grown[route], route)}
    assert after - before == {"a_field_nobody_added_yet"}, (
        "a field added to the payload does not appear in the enumeration, so "
        "the sweep is reading a list rather than the response")
    assert len(before) > 100, "the enumeration found almost nothing"


def test_the_sweep_can_tell_a_read_field_from_an_unread_one(swept):
    """The positive control, and the file is worthless without it. A sweep
    that answered *unread* to everything would report a spectacular finding and
    measure nothing."""
    for label, (_routes, _pid, sweep) in swept.items():
        assert sweep.reaching, f"{label}: nothing at all reaches a person"
        assert sweep.unread, f"{label}: everything reaches a person"
        paths = {f.path for f in sweep.reaching}
        assert any(p.startswith("profile.n_rows") for p in paths), (
            f"{label}: the row count does not reach a person, which it "
            f"visibly does — the instrument is broken, not the page")


def test_a_path_the_writer_cannot_reach_stops_the_sweep(swept):
    """**The bug this module shipped with, pinned.** `poke` returned `False`
    for every nested-array path — the tokenizer reassembled
    `questions[0].title` as `['questions', 'title', 0]` — and the sweep
    reported 2,408 of 2,450 fields unread. A silently-dropped write is this
    module's own subject arriving inside it."""
    assert FS._steps("questions[0].title") == ["questions", 0, "title"]
    assert FS._steps("a.b[2].c[10]") == ["a", "b", 2, "c", 10]
    payload = {"questions": [{"title": "x"}]}
    assert FS.poke(payload, "questions[0].title", "y") is True
    assert payload["questions"][0]["title"] == "y"
    assert FS.poke(payload, "questions[9].title", "y") is False


# ═══════════ THE DISPOSITIONS ═══════════

@pytest.mark.parametrize("label", sorted(SWEPT))
def test_every_unread_family_names_a_reader_or_a_row(label, swept):
    """**Three dispositions and no fourth**, on the standing check's own
    pattern."""
    _routes, _pid, sweep = swept[label]
    unread = {_family(route, shape)
              for (route, shape), v in sweep.shapes().items()
              if v["verdict"] == "none"}
    undeclared = sorted(f for f in unread
                        if f not in DECLARED and f not in FILED)
    assert not undeclared, (
        "these families are composed by the server and read by nothing in the "
        f"Guided door, and neither a reader nor a row is named for them:\n  "
        f"{undeclared}\n\n"
        "Either wire them, or add them to DECLARED with the reader named, or "
        "put them in FILED against a row. A field nobody reads is a promise "
        "nobody keeps.")


def test_every_declaration_names_a_reader_or_a_ledger_row():
    """A reason that names neither is a shrug, which is what this table exists
    to stop. Same rule and the same `READERS` vocabulary as the route check."""
    for family, reason in DECLARED.items():
        assert len(reason) > 20, f"{family}: the reason is a shrug: {reason!r}"
        assert "GUIDED-" in reason or any(r in reason for r in READERS), (
            f"{family}: names neither a reader from {READERS} nor a row")


def test_the_declaration_has_no_stale_entries(swept):
    """A family declared unread that the page has since started rendering is a
    stale excuse, and stale excuses are how a list like this stops meaning
    anything. `GUIDED-108`'s rule: an exception for a thing nobody excluded is
    a decision about nothing."""
    live = set()
    for _label, (_routes, _pid, sweep) in swept.items():
        for (route, shape), v in sweep.shapes().items():
            if v["verdict"] == "none":
                live.add(_family(route, shape))
    stale = sorted(f for f in list(DECLARED) + list(FILED) if f not in live)
    assert not stale, (
        f"these are declared or filed as unread and something reads them now: "
        f"{stale}. Delete the entry, or close the row.")


# ═══════════ THE COUNTS ═══════════

def test_the_sweep_reports_its_own_coverage(swept, capsys):
    """**The counts are the deliverable.** A sweep that reports only its hits
    has not reported its coverage."""
    with capsys.disabled():
        print("\n  ── L42-B · field-granularity sweep ──")
        for label in sorted(swept):
            _routes, _pid, sweep = swept[label]
            shapes = sweep.shapes()
            verdicts = {v: sum(1 for s in shapes.values() if s["verdict"] == v)
                        for v in ("all", "partial", "none")}
            counts = sweep.counts()
            print(f"  {label}")
            print(f"    payloads swept            {counts['routes']}")
            print(f"    fields enumerated         {counts['fields']}")
            print(f"    fields reaching a person  {counts['reaching']}")
            print(f"    fields unread             {counts['unread']}")
            print(f"    drives                    {counts['renders']}")
            print(f"    path shapes               {len(shapes)}  "
                  f"all={verdicts['all']} partial={verdicts['partial']} "
                  f"none={verdicts['none']}")
            if sweep.truncated_arrays:
                print(f"    arrays over {FS.MAX_ELEMENTS} elements   "
                      f"{len(sweep.truncated_arrays)} (first {FS.MAX_ELEMENTS} "
                      f"swept, and that cap is the reason this line exists)")
        print("  payloads NOT swept:")
        for line in NOT_SWEPT:
            print(f"    - {line}")
        print(f"  unread families declared    {len(DECLARED)}")
        print(f"  unread families FILED       {len(FILED)}"
              f"   <- the finding")
        from collections import Counter
        for row, n in sorted(Counter(FILED.values()).items()):
            print(f"      {row}  {n}")

    for _label, (_routes, _pid, sweep) in swept.items():
        assert sweep.counts()["fields"] > 500


def test_partial_is_reported_because_it_is_what_guided_142_looked_like(swept):
    """A shape where some elements render and some do not is **the** signal.
    `findings[0].title` rendered and `findings[7].title` did not, because the
    seventh was a pack finding and the filter dropped its whole class.

    Collapsing shapes without this verdict would have hidden the defect the
    module exists to catch, so its presence is asserted rather than assumed.
    """
    seen = False
    for _label, (_routes, _pid, sweep) in swept.items():
        shapes = sweep.shapes()
        partial = {k: v for k, v in shapes.items() if v["verdict"] == "partial"}
        for key, v in partial.items():
            assert 0 < v["n_reaching"] < v["n"]
            assert v["unread_indices"]
        seen = seen or bool(partial)
    assert seen, (
        "no shape is partially read on either fixture. Either every array is "
        "uniform — which no real payload is — or the shape collapsing lost the "
        "verdict that catches GUIDED-142")


def test_no_finding_class_is_entirely_dark(swept):
    """`GUIDED-142` as a standing check at this granularity: a finding whose
    every field is unread is a finding nothing renders."""
    for label, (routes, pid, sweep) in swept.items():
        project = routes[f"/project/{pid}"]
        read = {f.path.split(".")[0] for f in sweep.reaching
                if f.path.startswith("findings[")}
        every = {f.path.split(".")[0] for f in sweep.fields
                 if f.path.startswith("findings[")}
        dark = sorted(every - read)
        assert not dark, (
            f"{label}: these findings are served and no field of them reaches "
            f"a person: "
            + str([project["findings"][int(p[9:-1])]["id"] for p in dark]))


# ═══════════ `GUIDED-147` IS A LIST, AND THE LIST IS THE ROW'S CONTENT ═══════

def test_the_class_row_counts_exactly_the_families_filed_against_it():
    """`GUIDED-147`'s condition 2, mechanically.

    The row's whole content is *these families, and nobody has looked at
    them*. So the count in the row and the number of `FILED` entries pointing
    at it are one fact written twice, and this is what keeps them one fact.

    It was already wrong once. The row said sixteen while the table filed
    fifteen to it, because `project.n_working_rows` moved to `GUIDED-146` — a
    duplicated summary count rather than an unexamined field — and the `ev`
    list was not re-cut. In a row whose entire content is a count, that is the
    whole content being wrong, and it survived a loop.

    **The splitting rule this enforces**: a family leaves `GUIDED-147` the
    moment it gets a disposition, and the row closes when the list is empty —
    never on a verdict over the class. A row closed by judging fifteen
    unexamined things in aggregate is the shrug arriving one loop later.
    """
    import json as _json
    import re as _re

    ledger = (Path(__file__).resolve().parents[1]
              / "docs" / "turbotab" / "data" / "findings.json")
    rows = _json.loads(ledger.read_text(encoding="utf-8"))
    row = next(r for r in rows if r["id"] == "GUIDED-147")

    filed_here = sorted(f for f, r in FILED.items() if r.startswith("GUIDED-147"))
    assert filed_here, "nothing is filed against GUIDED-147; the table moved"

    # The count, written as a word because the row is prose. If the row is
    # ever closed this test should not be the thing that fails, so a closed
    # row is out of scope rather than an offense.
    if row["status"] not in ("OPEN", "PARTIAL"):
        assert not filed_here, (
            f"GUIDED-147 is {row['status']} and {len(filed_here)} families are "
            f"still filed against it: {filed_here}. The row closes when the "
            f"list is empty, not on a verdict over the class.")
        return

    words = {9: "nine", 10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen",
             14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen"}
    n = len(filed_here)
    said = words.get(n, str(n))

    # BOTH PLACES, SEPARATELY, and that is the point. The first version of
    # this check asked whether the right word appeared *anywhere* in
    # `item + ev`, and a revert probe put `THESE SIXTEEN DO NOT` back into the
    # evidence and came back GREEN — because `item` still said "Fifteen" and
    # the word was therefore present. A check for the presence of the true
    # count cannot see a false one sitting beside it.
    #
    # Anchored on the two phrases that carry the claim rather than on a
    # word-count over the prose, because the evidence legitimately contains
    # other numerals — "six payloads", "two lenses", and the sentence
    # recording that the FILED table has sixteen entries against two rows.
    leading = _re.match(r"\s*(\w+)\s+families", row["item"], _re.I)
    assert leading, (
        f"GUIDED-147's item no longer opens with '<count> families', so the "
        f"count cannot be checked: {row['item'][:80]!r}")
    assert leading.group(1).lower() == said, (
        f"{n} families are filed against GUIDED-147 and its item says "
        f"'{leading.group(1)} families'. The row's whole content is a count.")

    claim = _re.search(r"THESE\s+(\w+)\s+DO NOT", row["ev"])
    assert claim, (
        "GUIDED-147's evidence no longer carries its 'THESE <count> DO NOT' "
        "sentence, which is where the list is introduced")
    assert claim.group(1).lower() == said, (
        f"{n} families are filed against GUIDED-147 and its evidence says "
        f"'THESE {claim.group(1)} DO NOT'. This is exactly how the row went "
        f"wrong the first time: a family left for GUIDED-146 and the evidence "
        f"list was not re-cut. Filed: {filed_here}")

    # And every family the row NAMES is one the table still files here — the
    # other direction, which is what went stale last time.
    named = set(_re.findall(r"\b(?:project|plausibility|capabilities)\.(\w+)",
                            row["ev"]))
    filed_leaves = {f[1] for f in filed_here}
    moved_on = sorted(named - filed_leaves - {"n_working_rows"})
    assert not moved_on, (
        f"GUIDED-147's evidence still lists {moved_on}, and the FILED table no "
        f"longer files them here. A family that got a disposition has to leave "
        f"the row's list, or the list stops being the row's content.")

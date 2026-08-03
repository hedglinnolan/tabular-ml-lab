"""L42-A — two rulings: `GUIDED-143`'s false assertion, and `MISC-018`'s name.

## A1 · The record described a chronology nobody drew

Answering *yes, I am predicting a later outcome from earlier measurements*
recorded `strategy: chronological_grouped` and the sentence *"The held-out rows
are the latest ones … at times after the ones it trained on"*, and the draft
carried it into the methods section. `engine.draw_holdout` never read
`temporal_prediction`.

**And it was false in a second, independent way that the L41 report did not
have.** Driven at L42: `ml/splits.py` has five strategies —
`grouped | chronological | lockbox | stratified | random` — and **no combined
one**. `choose_strategy` returns `grouped` when both a group column and a time
column are given, by explicit design: *"Grouping outranks time."* So
`chronological_grouped` is a strategy name that exists in `turbotab/repeats.py`
and in two test files and **in no splitter anywhere**. The consumer sentence the
user reads before answering promised `ml/splits.py` would route on this answer;
that routing could not have produced what it claimed even if it had existed.

### The pattern is the lockbox constitution's own

Clause §03 — *the seal states its own basis, three states, never two* — where
`undetermined` is first-class, persisted, asserted by a test, and **never
rendered as a clean lock.** A temporal answer the draw cannot honor is the
identical shape: the app knows what was asked, it cannot draw it, and the honest
rendering is a basis that says so.

`IMPORT-020`'s asymmetry decides the branch. Leaking and disclosing is *refuse*;
leaking behind a lock icon is *assert something false*. A split described as
chronological and drawn at random is the second.

**The question is not removed** — that is the shelf being shortened, and
judgment renders as ranking or as a stated basis, never as absence. **The
chronological grouped draw is not built here**, so `GUIDED-143` stays `OPEN`;
this closes the false assertion and nothing else.

## A2 · `MISC-018` — the core named a band after a concept it does not hold

`get_reference_interval` returned a `p01`/`p99` pair. A reference interval is
the central **95%** — 2.5th to 97.5th percentile, CLSI EP28-A3c, minimum
reference sample n=120 (§A1.2). `p01`/`p99` is the central 98%, and for
`bp_sys` it is `90–200 mmHg` where §A1.2's own table gives the adult reference
interval as `90–120`. **Wrong by a category, not by calibration** — and
`get_impossibility_band`'s docstring already had the right sentence with the
wrong noun in it.

Renamed in **core**, not aliased: an alias leaves the false name importable and
lets the next reader trust it. `GUIDED-120` and `GUIDED-124` are the precedent —
a shared-core defect is corrected once in core.

**No `p025`/`p975` pair was added.** That is reference data under D4 and must be
read from primary sources; a wrong reference interval is worse than none,
because a clinician reads that name and believes it.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from turbotab import repeats as R

DATA = Path(__file__).resolve().parent / "sample_data"

#: `GUIDED-097`. The temporal question needs time points surviving as rows, so
#: both arms are longitudinal — but the two fixtures have different repeat
#: structure and different target shape, which is what the rule asks for.
TEMPORAL_FIXTURES = {
    "a scheduled visit series, continuous target": (
        "clinical_longitudinal.csv", "subject_id", "sbp"),
    "an irregular visit series, categorical target": (
        "longitudinal_visits.csv", "subject_id", "outcome"),
}

#: NOT COVERED, said out loud.
#:
#: THE HONORED BRANCH. `CHRONOLOGICAL_GROUPED` is named and unreachable —
#: `DRAWS_CHRONOLOGICALLY` is `False` and the draw has no time parameter, so no
#: fixture can produce it. That is the point of `GUIDED-143` staying `OPEN`, and
#: the branch is asserted unreachable below rather than left to look covered.
#:
#: A NON-REPEATED TABLE. Question 7 refuses before it is asked when rows are not
#: time points, so there is no cross-sectional arm to run.
#:
#: A REAL REFERENCE INTERVAL. No `p025`/`p975` pair exists to compare against;
#: adding one is D4 reference data read from primary sources.
SHAPES_NOT_COVERED = [
    "the honored chronological branch — unreachable until the draw is built "
    "(GUIDED-143 stays OPEN), and asserted unreachable rather than skipped",
    "a cross-sectional table — question 7 refuses before it is asked",
    "a real p025/p975 reference interval — D4 reference data, not recollected",
]


def _driven(fixture, temporal):
    """Upload → the full repeated-measures chain → seal, through the routes."""
    from fastapi.testclient import TestClient

    from turbotab import api

    name, group, target = TEMPORAL_FIXTURES[fixture]
    client = TestClient(api.app)
    with open(DATA / name, "rb") as handle:
        pid = client.post("/project", files={
            "file": (name, handle, "text/csv")}).json()["id"]
    for kind, payload in (
            ("set_target", {"column": target}),
            ("set_purpose", {"answer": "prediction"}),
            ("set_grain", {"answer": "people_repeat", "group_col": group}),
            ("set_repeat_kind", {"kind": "time_points"}),
            ("set_unit_of_analysis", {"unit": "record"}),
            ("set_temporal_prediction", {"temporal": temporal}),
            ("set_eligibility", {"answer": "everyone"}),
            ("seal", {})):
        ok = client.post(f"/project/{pid}/decision",
                         json={"kind": kind, "payload": payload})
        assert ok.status_code == 200, (kind, ok.text[:300])
    return client, pid


# ═══════════ A1 · THREE STATES, NEVER TWO ═══════════

def test_the_basis_set_names_the_state_the_app_cannot_reach():
    """**The `undetermined` property, applied one question over.**

    A basis set that omitted the honorable state could not say the app is
    *missing* it — only that the app never tries. So `CHRONOLOGICAL_GROUPED` is
    named, and this asserts it is currently unreachable rather than letting its
    presence read as coverage.
    """
    assert set(R.SPLIT_BASES) == {R.GROUPED, R.CHRONOLOGICAL_GROUPED,
                                  R.CHRONOLOGICAL_NOT_DRAWN}

    # **L43-C BUILT THE DRAW, AND THE THIRD STATE DID NOT BECOME DEAD.** That
    # is the property this test now guards, and it is the more interesting
    # one. `chronological_requested_not_drawn` was written at L42 for an app
    # that could not draw chronologically at all; the temptation on building
    # the draw is to delete it. It stays, because the app still cannot draw
    # chronologically when no time column has been recorded — and a user who
    # has no clean date column must still be able to seal.
    reached = {R.split_strategy(t, u, time_col=c)["strategy"]
               for t in (True, False)
               for u in (R.UNIT_RECORD, R.UNIT_PERSON, R.UNIT_NOT_DESCRIBED)
               for c in (None, "visit_date")}
    assert reached == {R.GROUPED, R.CHRONOLOGICAL_GROUPED,
                       R.CHRONOLOGICAL_NOT_DRAWN}, (
        f"the basis set has a dead state: {sorted(set(R.SPLIT_BASES) - reached)} "
        f"is named and unreachable. Three states, never two — and never two "
        f"dressed as three.")
    assert R.split_strategy(True, R.UNIT_RECORD, time_col=None)["strategy"] \
        == R.CHRONOLOGICAL_NOT_DRAWN, (
        "a temporal task with no time column recorded reports the honorable "
        "basis, which is the assertion GUIDED-143 was filed for")


def test_the_capability_flag_matches_what_the_draw_can_do():
    """A bare boolean's failure mode is being flipped without the capability
    arriving. This pins it to the draw's own signature: `draw_holdout` has no
    time parameter, so the flag cannot go `True` while the draw stays random.
    """
    import inspect

    from turbotab import engine

    params = set(inspect.signature(engine.draw_holdout).parameters)
    takes_time = bool(params & {"datetime_col", "time_col", "temporal",
                                "order_by"})
    assert takes_time is R.DRAWS_CHRONOLOGICALLY, (
        f"`DRAWS_CHRONOLOGICALLY` is {R.DRAWS_CHRONOLOGICALLY} and "
        f"`draw_holdout` takes {sorted(params)}. If the chronological draw has "
        f"landed, flip the flag and close GUIDED-143; if it has not, the flag "
        f"is asserting a capability that is not there.")


def test_the_honored_flag_is_beside_the_sentence_and_not_only_inside_it():
    """Trap #7 — the machine-readable form lossier than the prose. The
    manuscript, the page and the validator all read the payload."""
    asked = R.split_strategy(True, R.UNIT_RECORD)
    assert asked["honored"] is False
    assert asked["strategy"] == R.CHRONOLOGICAL_NOT_DRAWN
    plain = R.split_strategy(False, R.UNIT_RECORD)
    assert plain["honored"] is True and plain["strategy"] == R.GROUPED


@pytest.mark.parametrize("fixture", sorted(TEMPORAL_FIXTURES))
def test_the_record_no_longer_claims_the_held_out_rows_are_the_latest(fixture):
    """`GUIDED-143`'s own evidence sentence, inverted into an assertion."""
    client, pid = _driven(fixture, temporal=True)
    import json

    everything = json.dumps(client.get(f"/project/{pid}").json())
    draft = json.dumps(client.get(f"/project/{pid}/draft").json())
    for lie in ("latest ones", "times after the ones it trained on"):
        assert lie not in everything, f"the record still says {lie!r}"
        assert lie not in draft, f"the draft still says {lie!r}"

    assert "not drawn that way" in draft, (
        "the draft says nothing about the split it did not draw, so the "
        "assertion was removed and nothing replaced it — which is silence "
        "where a disclosure belongs")
    assert "optimistic" in draft


@pytest.mark.parametrize("fixture", sorted(TEMPORAL_FIXTURES))
def test_the_seal_carries_the_temporal_basis_because_it_is_the_draw(fixture):
    """**Question 7 can only say what was asked; the seal is the only place
    that can say what was done.** Beside the basis rather than folded into
    `seal_basis`, on `resolution`'s precedent."""
    client, pid = _driven(fixture, temporal=True)
    lockbox = client.get(f"/project/{pid}").json()["lockbox"]
    assert lockbox["temporal_requested"] is True
    assert lockbox["temporal_honored"] is False
    assert lockbox["temporal_basis"] == R.CHRONOLOGICAL_NOT_DRAWN
    assert lockbox["temporal_sentence"]
    # It does not become a fifth seal basis.
    assert lockbox["seal_basis"] == "grouped"


def test_a_non_temporal_answer_leaves_the_seal_clean():
    """The negative control. A disclosure on every seal would be the second
    uncalibrated layer of caution this project forbids — it makes a real
    concern and a routine one read identically."""
    client, pid = _driven("a scheduled visit series, continuous target",
                          temporal=False)
    body = client.get(f"/project/{pid}").json()
    assert body["lockbox"]["temporal_honored"] is True
    assert body["lockbox"]["temporal_basis"] == R.GROUPED
    assert body["disclosures"]["exploratory"] is False
    assert "not drawn that way" not in body["disclosures"]["seal"]


@pytest.mark.parametrize("fixture", sorted(TEMPORAL_FIXTURES))
def test_it_is_never_rendered_as_a_clean_lock(fixture):
    """**The clause the ruling turns on.** `IMPORT-020`'s asymmetry: leaking
    and disclosing is the refuse branch; leaking behind a lock icon is the
    assert-something-false branch.

    Driven through the page's own controller, because the band is the thing a
    reader takes the held-out number to mean.
    """
    from turbotab import pageharness as PH

    if not PH.available():
        pytest.skip("no JS engine on this machine")

    client, pid = _driven(fixture, temporal=True)
    body = client.get(f"/project/{pid}").json()
    assert body["disclosures"]["exploratory"] is True, (
        "the seal reports itself clean over a split that did not honor the "
        "validation the user asked for")

    routes = {
        f"/project/{pid}": body,
        f"/project/{pid}/interview?step=data":
            client.get(f"/project/{pid}/interview?step=data").json(),
        f"/project/{pid}/interview?step=explore": {"questions": [], "steps": []},
        f"/project/{pid}/evidence/missingness": {"cards": []},
    }
    out = PH.run(
        "__emit({html: (__harness.html('disclosuresBox') || '').slice(0, 8000)});",
        routes=routes, search=f"?project={pid}")
    html = out["html"]
    assert "not a verified clean split" in html, (
        "the disclosure band renders `sealed` over a split that trains on rows "
        "from after the rows it is scored on")
    assert "is-exploratory" in html and "is-sealed" not in html


def test_the_effect_preview_no_longer_promises_the_draw():
    """The one place a user reads this *before* answering. It said *"the
    held-out rows become the latest ones rather than a random draw"* — the same
    false claim the record carried, in the control that sets the expectation.

    **Comments stripped first, and that is the whole difficulty.** Trap #5
    reserves grep for claims that are *genuinely about the file*, and this one
    is — *does the shipped code still contain this string*. But the old phrase
    is deliberately still in the file, quoted in the comment that records what
    it used to say, and a bare grep cannot tell the record from the claim.

    Driving it was tried first and does not work: `EFFECTS` is a `var` inside
    the page's closure, so the harness cannot reach it from an injected body —
    the same wall `PULL_RENDER` hit at L41-C, where the answer was to click the
    control rather than call the function. There is no control that renders an
    effect preview in isolation, so this is the honest instrument available.
    """
    page = (Path(__file__).resolve().parent / "web" / "index.html").read_text(
        encoding="utf-8")
    import re
    code = re.sub(r"/\*.*?\*/", " ", page, flags=re.S)
    assert "held-out rows become the latest ones" not in code, (
        "the effect preview still promises a draw no splitter here performs")
    assert "held-out rows become the latest ones" in page, (
        "the comment recording what this used to say is gone, so the next "
        "reader has no record of why the wording is what it is")
    assert "still drawn at random within whole people" in code


def test_no_splitter_anywhere_implements_the_strategy_the_app_records():
    """**The second falsity, and it is why a note beside the sentence would
    not have done.**

    `chronological_grouped` appears in `turbotab/repeats.py`, where it is
    composed, and nowhere else in the engine. `ml/splits.py` knows five
    strategies and none of them is it — and its router puts grouping above time
    by design, so a repeated-measures project could not reach the chronological
    branch even if the routing existed.
    """
    from ml.splits import SplitSpec, choose_strategy

    frame = pd.read_csv(DATA / "clinical_longitudinal.csv")
    spec = SplitSpec(use_group_split=True, entity_id_col="subject_id",
                     use_time_split=True, datetime_col="visit_date")
    assert choose_strategy(frame, spec) == "grouped", (
        "ml/splits now prefers time over grouping, which changes what "
        "GUIDED-143's second half says")

    splits = (Path(__file__).resolve().parents[1] / "ml" / "splits.py").read_text()
    assert R.CHRONOLOGICAL_GROUPED not in splits, (
        "ml/splits.py now knows the combined strategy; if the draw is wired "
        "too, GUIDED-143 closes")


def test_the_consumer_sentence_says_what_actually_happens():
    """A hover the user reads before answering is a claim like any other."""
    assert "drawn at random within whole people" in R.TEMPORAL_CONSUMER
    assert "picks grouped when both apply" in R.TEMPORAL_CONSUMER
    for lie in ("selects the chronological split",
                "reads this to choose between its chronological"):
        assert lie not in R.TEMPORAL_CONSUMER


def test_guided_143_closed_only_because_the_draw_exists():
    """L42 pinned this row OPEN; L43-C built the draw and it closes.

    **The successor, not the deletion.** The property worth keeping is the
    same one in both directions: the row's status and the app's capability are
    one fact. L42 asserted *open, because the draw does not exist*; this
    asserts *closed, and the draw exists* — so closing it again on a tree
    where the draw was removed fails here rather than passing quietly.
    """
    import inspect
    import json as _json

    from turbotab import engine

    ledger = (Path(__file__).resolve().parents[1]
              / "docs" / "turbotab" / "data" / "findings.json")
    row = next(r for r in _json.loads(ledger.read_text(encoding="utf-8"))
               if r["id"] == "GUIDED-143")

    takes_time = bool(set(inspect.signature(engine.draw_holdout).parameters)
                      & {"time_col", "datetime_col", "temporal"})
    if row["status"] in ("OPEN", "PARTIAL"):
        assert not takes_time, (
            "GUIDED-143 is open and `draw_holdout` takes a time argument — "
            "either the draw landed and the row should close, or the "
            "signature is asserting a capability that is not wired")
        return

    assert takes_time, (
        "GUIDED-143 is closed and `draw_holdout` has no time parameter, so "
        "the record is claiming a capability again — which is the defect this "
        "row is about, arriving through the ledger instead of the manuscript")
    assert R.DRAWS_CHRONOLOGICALLY is True, (
        "the row is closed and the composer still says the app cannot draw "
        "chronologically")
    assert row.get("test"), "a FIXED row with no named regression test"

def test_the_false_name_is_gone_from_core_and_has_no_alias():
    """**Renamed, not aliased.** An alias leaves the false name importable and
    lets a future reader trust it, which is the whole of what `MISC-018` is
    about — the code knew what it held and the name did not."""
    from ml import physiology_reference as PR

    assert hasattr(PR, "get_improbability_band")
    assert not hasattr(PR, "get_reference_interval"), (
        "the old name is still importable, so nothing stops a caller from "
        "trusting it")
    assert not hasattr(PR, "band_is_wider_than_interval")
    assert hasattr(PR, "impossibility_contains_improbability")


def test_no_module_still_imports_the_false_name():
    """Checked over the tree rather than over the one module, because a rename
    that left one importer would fail at import time in whichever door reached
    it first — and `pages/` is a door this suite does not collect."""
    root = Path(__file__).resolve().parents[1]
    offenders, carries_the_note = [], []
    for path in sorted(root.rglob("*.py")):
        parts = set(path.parts)
        if parts & {"venv", ".venv", "__pycache__", "node_modules"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for dead in ("get_reference_interval", "band_is_wider_than_interval"):
            # The rename note in `physiology_reference` quotes the old name on
            # purpose; that is the record of what it used to be.
            if dead not in text:
                continue
            (carries_the_note if "MISC-018" in text else offenders).append(
                f"{path.relative_to(root)} :: {dead}")

    # THE POSITIVE CONTROL, and it is the reason this test is not vacuous.
    # Everything above is an absence claim, so a sweep that read nothing —
    # `root` one level off, a glob that matched no file, an encoding that
    # silently emptied every read — would report a clean rename it never
    # checked. `physiology_reference` is the one module that legitimately
    # still contains both old names, in the note recording what they were.
    # If the sweep cannot find those, it did not look at the tree.
    # Anchored on the module rather than on a count: this file names both dead
    # names too, and a count would move every time another file cites the row.
    assert all(f"ml/physiology_reference.py :: {dead}" in carries_the_note
               for dead in ("get_reference_interval",
                            "band_is_wider_than_interval")), (
        "the sweep did not find the two old names where they are SUPPOSED to "
        f"survive — in `ml/physiology_reference`'s MISC-018 rename note. Got "
        f"{carries_the_note}. Nothing below this line means anything if the "
        "sweep is reading an empty tree.")

    assert not offenders, offenders


def test_what_the_band_returns_is_the_central_98_percent_not_95():
    """The arithmetic behind the rename, said as a number.

    `p01`/`p99` spans 98% of the reference population. A reference interval is
    the central 95%, 2.5th to 97.5th (CLSI EP28-A3c). They are different
    quantities and the old name asserted the second while returning the first.
    """
    from ml.physiology_reference import (get_improbability_band,
                                         load_reference_bundle)

    ref = load_reference_bundle()["nhanes"]
    entry = ref["variables"]["bp_sys"]
    assert (entry["p01"], entry["p99"]) == (90, 200)
    assert get_improbability_band(ref, "bp_sys") == (90.0, 200.0, "mmHg")
    # §A1.2's own table gives the adult SBP reference interval as 90–120. The
    # upper bound is wrong by a category, not by calibration.
    assert get_improbability_band(ref, "bp_sys")[1] != 120


def test_the_tiers_still_nest_under_the_new_name():
    """The predicate the rename carried with it, run over every variable."""
    from ml.physiology_reference import (impossibility_contains_improbability,
                                         load_nhanes_reference)

    ref = load_nhanes_reference()
    for key in ref["variables"]:
        assert impossibility_contains_improbability(ref, key), key


def test_the_pack_card_is_unchanged_and_still_says_which_band_it_counted():
    """`GUIDED-144` was ruled: the conservative count stays and the card
    already discloses its band. The rename must not silently move a number —
    it renames what the app calls the band, not what the band is."""
    from turbotab import clinical as C

    finding = C.impossible_vs_extreme_finding(
        pd.read_csv(DATA / "clinical_labs.csv"))
    sbp = next(c for c in finding["params"]["columns"] if c["column"] == "sbp")
    assert sbp["n_impossible"] == 4
    assert sbp["n_abnormal_but_possible"] == 31, (
        "the count moved; MISC-018 is a rename and GUIDED-144's ruling is that "
        "the number moves only when a real reference interval is added")
    assert sbp["normal_band"] == [90.0, 200.0]


# ═══════ `GUIDED-143` · A PRAGMA IS A CLAIM WITH NO GUARD (L43-A3) ═══════════

def test_the_unreachable_basis_is_asserted_unreachable_and_not_annotated():
    """`GUIDED-143`, ruled at the L42 adjudication.

    `chronological_grouped` stays: the lockbox constitution §03 is *three
    states, never two*, and `chronological_requested_not_drawn` is only
    meaningful against an honorable state that exists as a value — a basis set
    omitting it cannot say the app is *missing* it.

    What does not stay is the `# pragma: no cover` that used to excuse the
    branch. **A pragma is a claim with no guard**, and this project has been
    bitten by one already: `GUIDED-134`'s sat on a line its own test executed,
    so the annotation asserted something false about the code beside it.

    So the claim gets a test. This one, and it is deliberately written to stop
    being true the moment the draw lands rather than to pin `False` forever —
    a guard that has to be edited by the loop that fixes the thing is a guard
    that gets edited without being read.
    """
    import inspect

    from turbotab import engine

    # Two conditions now, not one: the draw has to EXIST and a time column
    # has to be recorded. L43-C flipped the first; the second is per-project
    # and is what keeps the not-drawn basis alive.
    reachable = R.DRAWS_CHRONOLOGICALLY
    drawn = R.split_strategy(temporal=True, unit=R.UNIT_RECORD,
                             time_col="visit_date")

    if not reachable:
        assert drawn["strategy"] == R.CHRONOLOGICAL_NOT_DRAWN, (
            "the flag says the chronological draw does not exist and the "
            "composer returned something other than the not-drawn basis")
        assert drawn["honored"] is False
        # And the flag is not free-floating: it agrees with the draw's own
        # signature, which is what stops it being flipped without the
        # capability arriving.
        assert not (set(inspect.signature(engine.draw_holdout).parameters)
                    & {"datetime_col", "time_col", "temporal", "order_by"}), (
            "`draw_holdout` takes a time argument but DRAWS_CHRONOLOGICALLY "
            "is False — the flag is understating what the draw can do")
        # THE ROW IS THE DATED REASON. §05's second clause: an unreachable
        # state ships with a test that names the row keeping it open.
        import json as _json
        ledger = (Path(__file__).resolve().parents[1]
                  / "docs" / "turbotab" / "data" / "findings.json")
        row = next(r for r in _json.loads(ledger.read_text(encoding="utf-8"))
                   if r["id"] == "GUIDED-143")
        assert row["status"] in ("OPEN", "PARTIAL"), (
            f"GUIDED-143 is {row['status']} and `DRAWS_CHRONOLOGICALLY` is "
            f"still False. Either the draw landed and the flag was not "
            f"flipped, or the row was closed without the draw.")
    else:
        assert drawn["strategy"] == R.CHRONOLOGICAL_GROUPED, (
            "the flag says the chronological draw exists and the composer "
            "still returns the not-drawn basis")
        assert drawn["honored"] is True


def test_the_honorable_branch_is_executed_rather_than_excused():
    """The other half, and the reason the pragma had to go rather than move.

    A state that is unreachable *and never executed* is two unverified claims
    stacked: that it cannot happen, and that it would be right if it did. The
    first is asserted above. This one runs the branch — by setting the flag,
    which is the only thing standing in front of it — and checks the payload
    it produces is the one the seal, the manuscript and the page would read.

    When the draw lands and the flag flips for real, this test does not
    change. That is the point of writing it this way.
    """
    import unittest.mock as _mock

    with _mock.patch.object(R, "DRAWS_CHRONOLOGICALLY", True):
        got = R.split_strategy(temporal=True, unit=R.UNIT_RECORD,
                               time_col="visit_date")

    assert got["strategy"] == R.CHRONOLOGICAL_GROUPED
    assert got["honored"] is True, (
        "the honorable basis reports itself unhonored, so the disclosure "
        "would fire on a split that was drawn correctly")
    assert "latest ones" in got["sentence"], got["sentence"]
    assert "not drawn" not in got["sentence"].lower(), (
        "the honorable branch is carrying the dishonorable sentence")

    # And the value is in the declared basis set — trap #3's rule, that a
    # stand-in must resolve in the real registry.
    assert got["strategy"] in R.SPLIT_BASES, (
        f"{got['strategy']!r} is not in SPLIT_BASES, so the seal would carry "
        f"a basis nothing downstream can interpret")


def test_no_pragma_excuses_a_branch_in_the_split_composer():
    """The standing form. `GUIDED-134` is the precedent: its `# pragma: no
    cover` sat on a line its own test executed, which is an annotation
    asserting something false about the code it annotates.

    Scoped to this module rather than the tree, because a blanket ban is a
    different and larger argument — several pragmas in this repository sit on
    genuinely defensive `except` arms and are correct.
    """
    source = (Path(__file__).resolve().parents[1]
              / "turbotab" / "repeats.py").read_text(encoding="utf-8")
    # A line that *starts* with `#` is prose about the pragma, not a pragma:
    # coverage only honors it as a trailing annotation on a statement. Without
    # this the check flags the comment in `repeats.py` that records why the
    # pragma was removed, which is the guard failing on its own documentation.
    offenders = [ln.strip() for ln in source.splitlines()
                 if "pragma: no cover" in ln and not ln.strip().startswith("#")]
    assert not offenders, (
        f"`repeats.py` excuses a branch with a pragma rather than asserting "
        f"what it claims: {offenders}. GUIDED-143's ruling is that the "
        f"unreachable basis value stays and the pragma does not.")

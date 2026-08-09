"""L47-B1 — `GUIDED-170`. A SETTLED nutritional claim about a row identifier.

The product owner selected `SEQN` from the nutrient dropdown and pressed Ask, and
the app answered:

> *"Prevalence of inadequacy for `SEQN` is computed by the EAR cut-point
> method"* — with a **SETTLED** badge, `may_preselect: true`, and a citation that
> resolves.

**On the surface the entire domain-track ordering was justified by.** `LOOP.md`
§04 put nutrition first *"because it is the one pack that forces a refusal"*, and
this is the one question in the pack whose whole job is to refuse correctly.

## Why every existing refusal fell through

The three were complete along the axes they knew, and the row is wrong about what
those axes are. They are: **(a)** the nutrient name matched against `AI_ONLY` —
which never looks at `reference_kind`; **(b)** `reference_kind.lower() == "rda"`;
**(c)** the `basis`. **Nothing checked `reference_kind == "AI"` and nothing asked
whether the subject was a nutrient at all**, so `SEQN` missed all three and the
settled tail answered.

## What the fifth axis may and may not claim

It is **not** a list of nutrients that have an EAR. That is the DRI table, it is
`GUIDED-067`, and it is deliberately unbuilt because those numbers must be read
from NASEM rather than recollected. `NUTRIENT_NAMES` is a list of names **this
pack recognizes**, and the refusal says exactly that — the app holds no reference
intake for the subject. A statement about the app's own knowledge is always
checkable, and it is the honest alternative to asserting a nutritional fact about
a respondent identifier.

## The fixture that would have lied

`DRIVE_PREREG_NHANES.md` §1 records that on the real export `SEQN` is `float64`
and is **not** flagged by `identifiers.detect`. A refusal built on identifier
detection would pass here and fail on his file — trap #4 waiting on the exact row
it would flatter. **So nothing below consults identifier detection**, and
`test_the_refusal_holds_when_nothing_detects_an_identifier` drives the case
directly.

**And §03's fixture warning is wrong in its particulars.** It says
`nhanes_dietary.csv`'s `SEQN` holds integers 1..120 while the real export does
not. Measured: **all three** shipped NHANES fixtures carry `SEQN` as `int64`
1..120, and `identifiers.detect` flags it in **all three**. There is no fixture
asymmetry to exploit, which makes the point sharper rather than weaker: the
condition the real file is in cannot be reproduced from any fixture at all, so
the refusal has to hold without detection by construction.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from turbotab import nutrition as N

DATA = Path(__file__).resolve().parent / "sample_data"

#: All three shipped NHANES exports. `GUIDED-097` asks for two of different
#: shape; these differ in energy unit and in whether the survey design is fully
#: specified, and all three reproduce the defect.
NHANES = ("nhanes_dietary.csv", "nhanes_kilojoules.csv",
          "nhanes_partial_design.csv")

#: The five columns the dropdown offered as nutrients on his file. A refusal that
#: only knew about `SEQN` would have been tuned to one screenshot.
NOT_NUTRIENTS = ("SEQN", "WTDRD1", "WTMEC2YR", "SDMVSTRA", "SDMVPSU")

#: NOT COVERED, said out loud.
SHAPES_NOT_COVERED = (
    "A nutrient with an EAR that this pack does not name. `NUTRIENT_NAMES` is "
    "17 entries from `research/NUTRITION_PACK.md` §07-§08 plus the nine already "
    "in `AI_ONLY` and `SKEWED_REQUIREMENT`; a real export column for, say, "
    "selenium would be refused as unrecognized. That is the honest direction to "
    "be wrong in — the app says it holds no reference rather than inventing "
    "one — but it is a false refusal and it is stated rather than hidden.",
    "A `float64` SEQN. No shipped fixture has one; the detection-free property "
    "is asserted directly instead.",
)


def _client():
    from fastapi.testclient import TestClient

    from turbotab import api

    return TestClient(api.app)


def _dietary(client, fixture):
    with (DATA / fixture).open("rb") as handle:
        pid = client.post("/project", files={
            "file": (fixture, handle, "text/csv")}).json()["id"]
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_lens", "payload": {"lens": ["dietary"]}})
    return pid


def _pairs_that_exist():
    """`(fixture, column)` for every pair the fixtures actually carry.

    `AUDIT-039`, `L56-B2`. This was a cross product of 3 fixtures × 5 columns
    with a `pytest.skip` inside for the pairs that do not exist —
    `nhanes_partial_design.csv` is *named* for lacking two of the design
    columns, so two of the fifteen could never run. A skip there is the shape
    the row is about: pytest counts it as not-a-failure, so a fixture quietly
    losing a column it is supposed to have reads exactly like a fixture that
    never had one.

    **The parametrization is narrowed to the pairs that exist and the dropped
    ones are named** — `GUIDED-097`'s rule applied to a skip — so the count is
    asserted below rather than discovered at run time.
    """
    out, dropped = [], []
    for fixture in NHANES:
        columns = set(pd.read_csv(DATA / fixture).columns)
        for column in NOT_NUTRIENTS:
            (out if column in columns else dropped).append((fixture, column))
    return out, dropped


PAIRS, PAIRS_DROPPED = _pairs_that_exist()


def test_the_pairs_this_file_drops_are_the_two_the_fixture_is_named_for():
    """The narrowing is a claim, so it is checked rather than trusted.

    Without this, narrowing the parametrization would hide the same thing the
    skip hid: a fixture losing a design column would silently shrink the
    matrix and every remaining case would still pass.
    """
    assert len(PAIRS) + len(PAIRS_DROPPED) == len(NHANES) * len(NOT_NUTRIENTS)
    assert sorted(PAIRS_DROPPED) == [
        ("nhanes_partial_design.csv", "SDMVPSU"),
        ("nhanes_partial_design.csv", "SDMVSTRA"),
    ], (
        f"the set of fixture/column pairs that do not exist has changed: "
        f"{sorted(PAIRS_DROPPED)}. `nhanes_partial_design.csv` is named for "
        f"carrying only part of the design; any other absence is a fixture "
        f"that lost a column, which is what this file's subject is about.")


@pytest.mark.parametrize("fixture,column", PAIRS,
                         ids=[f"{f.split('.')[0]}-{c}" for f, c in PAIRS])
def test_the_app_refuses_to_call_a_design_column_a_nutrient(fixture, column):
    """The defect, on every fixture that reproduces it and every column it
    offered — not only the one in the screenshot."""
    client = _client()
    pid = _dietary(client, fixture)

    body = client.get(f"/project/{pid}/nutrition/prevalence"
                      f"?nutrient={column}&basis=usual_intake"
                      f"&reference_kind=EAR")
    assert body.status_code == 200, (
        f"a refusal is 200 with a payload, never a 4xx — the request was not "
        f"malformed (`GUIDED-060`): {body.status_code} {body.text[:200]}")
    payload = body.json()
    assert payload["refused"] is True, (
        f"{fixture}/{column}: the app computed a prevalence of inadequacy. "
        f"method={payload.get('method')!r} badge={payload.get('evidence_status')!r}")
    assert "not a nutrient this pack recognizes" in payload["reason"]
    assert column in payload["reason"], "the refusal does not name its subject"


def test_the_refusal_matches_the_other_four_field_for_field():
    """Same payload shape, same badge, same offer keys. A fifth refusal that
    answered in a different vocabulary would be a second thing for every
    consumer to learn."""
    from turbotab import figures

    with pytest.raises(N.PrevalenceRefusal) as caught:
        N.prevalence_of_inadequacy("SEQN", basis=N.USUAL_INTAKE,
                                   reference_kind="EAR")
    refusal = caught.value
    assert set(refusal.offer) == {"draw", "label", "caption_note", "forbidden"}, (
        f"the offer's key set differs from the other four: {sorted(refusal.offer)}")
    assert refusal.offer["draw"] == "per_nutrient_distribution", (
        "the house move for this offer is the observed distribution, which four "
        "live refusals already use")
    payload = refusal.to_dict()
    for key in ("refused", "reason", "offer", "evidence_status", "source",
                "may_preselect"):
        assert key in payload, f"{key} is missing from the refusal payload"
    assert payload["refused"] is True
    assert payload["evidence_status"] == "SETTLED"
    # AND THE OFFER RESOLVES. `GUIDED-060`: promising a picture nobody can draw
    # reads as a feature and is worse than offering nothing.
    resolved = figures.resolve_offer(refusal.offer)
    assert resolved["resolved"]["id"] == "per_nutrient_distribution"


@pytest.mark.parametrize("name", ["SEQN", "WTDRD1", "SDMVSTRA", "SDMVPSU",
                                  "WTMEC2YR", "SDDSRVYR"])
def test_the_refusal_says_what_the_subject_actually_is(name):
    """*"It is not a nutrient"* is true and thin. The app already holds these
    names — `DIETARY_WEIGHTS`, `STRATA`, `PSU`, `EXAM_WEIGHT` are constants in
    this module — so it can say what the column IS, which is the actionable
    half."""
    with pytest.raises(N.PrevalenceRefusal) as caught:
        N.prevalence_of_inadequacy(name, basis=N.USUAL_INTAKE,
                                   reference_kind="EAR")
    said = str(caught.value)
    assert " it is " in said, (
        f"{name} is a column this module already names and the refusal does not "
        f"say what it is: {said[:160]}")


def test_the_refusal_holds_when_nothing_detects_an_identifier():
    """Trap #4, on the exact row it would flatter.

    On the real export `SEQN` is `float64` and `identifiers.detect` does not flag
    it. Asserted as a property of the refusal — it takes a string and no frame,
    so detection cannot be in the path — rather than by constructing a fixture,
    because no fixture can be constructed that reaches this function with a
    frame.
    """
    import inspect

    source = inspect.getsource(N.prevalence_of_inadequacy)
    assert "identifiers" not in source and "is_id_like" not in source, (
        "the refusal consults identifier detection, which is absent on the "
        "product owner's own file")
    # And the same subject refuses identically whatever the frame would say.
    for spelling in ("SEQN", "seqn", " SEQN "):
        with pytest.raises(N.PrevalenceRefusal):
            N.prevalence_of_inadequacy(spelling, basis=N.USUAL_INTAKE,
                                       reference_kind="EAR")


@pytest.mark.parametrize("fixture", NHANES)
def test_the_dropdown_offers_only_what_the_pack_recognizes(fixture):
    """The other half. The refusal is the backstop; the dropdown is what stops a
    person reaching for it — and `SEQN` was column zero, so it was the
    pre-selected default."""
    import re

    from turbotab import pageharness as PH

    client = _client()
    pid = _dietary(client, fixture)
    served = client.get(f"/project/{pid}").json()["nutrient_columns"]
    assert served, f"{fixture}: the server now names no nutrient columns at all"

    # DRIVEN, not read off the payload. The first version of this asserted on
    # `nutrient_columns` and the revert probe reported GREEN — NOT LOAD-BEARING,
    # correctly: reverting the page's `prevalenceColumns()` to "every numeric
    # column" left the payload untouched, so the claim was about the server
    # while the defect was in the dropdown.
    if not PH.available():
        pytest.skip("no JS engine on this machine")
    routes = {f"/project/{pid}": client.get(f"/project/{pid}").json()}
    for path in ("interview?step=data", "interview?step=explore", "capabilities",
                 "features", "recipes", "preprocess", "figures", "draft",
                 "manuscript", "models", "training", "instability", "explain",
                 "sensitivity", "evidence/plausibility", "evidence/missingness",
                 "interview?step=features"):
        resp = client.get(f"/project/{pid}/{path}")
        routes[f"/project/{pid}/{path}"] = (resp.json() if resp.status_code == 200
                                            else {})
    out = PH.run("__emit({html: __harness.html('prevBox') || "
                 "        __harness.html('prevalenceBox') || "
                 "        __harness.html('prevControls') || ''});",
                 routes=routes, search=f"?project={pid}")
    # SCOPED TO THE NUTRIENT SELECT. The first version matched every `<option`
    # in the box and picked up the BASIS dropdown's values too — `usual_intake`
    # is not a nutrient and was duly refused, so the test failed for a reason
    # that had nothing to do with the finding. A selector that is nearly right
    # is a claim about the wrong thing.
    block = re.search(r'<select data-prev="nutrient">(.*?)</select>',
                      out["html"] or "", re.S)
    rendered = re.findall(r'<option value="([^"]+)"',
                          block.group(1) if block else "")
    offered = rendered if rendered else served
    assert offered, f"{fixture}: the dropdown now offers nothing at all"
    for column in NOT_NUTRIENTS:
        assert column not in offered, (
            f"{fixture}: `{column}` is still offered as a nutrient "
            f"(rendered={bool(rendered)})")
    # AND WHAT IS OFFERED IS ANSWERABLE. A dropdown of things that all refuse
    # would be the shelf shortened to make the refusal look good.
    for column in offered:
        body = client.get(f"/project/{pid}/nutrition/prevalence"
                          f"?nutrient={column}&basis=usual_intake"
                          f"&reference_kind=EAR").json()
        assert body["refused"] is False, (
            f"{fixture}: `{column}` is offered and then refused — "
            f"{body.get('reason', '')[:120]}")


def test_the_two_cases_that_must_still_answer():
    """The regression the fifth axis would most easily cause.

    `calcium` is in neither `AI_ONLY` nor `SKEWED_REQUIREMENT`, and **neither
    `calcium` nor `iron` is a column in `dietary_recalls.csv`** — which is why
    the axis is *is this name a nutrient*, not *is this column in the frame*. A
    column-membership axis breaks both of these.
    """
    calcium = N.prevalence_of_inadequacy("calcium", basis=N.USUAL_INTAKE,
                                         reference_kind="EAR")
    assert calcium["method"] == "cut_point"
    iron = N.prevalence_of_inadequacy("iron", basis=N.USUAL_INTAKE,
                                      reference_kind="EAR",
                                      stratum="menstruating")
    assert iron["method"] == "probability_approach"


def test_the_subject_axis_runs_before_the_reference_and_basis_axes():
    """*"This is not a nutrient"* dominates *"the RDA is the wrong reference for
    it"* — answering the second about `SEQN` would still be a nutritional claim
    about a row identifier."""
    with pytest.raises(N.PrevalenceRefusal) as caught:
        N.prevalence_of_inadequacy("SEQN", basis=N.SINGLE_DAY,
                                   reference_kind="RDA")
    assert "not a nutrient" in str(caught.value), (
        f"a non-nutrient with a wrong basis AND a wrong reference was answered "
        f"about its basis: {str(caught.value)[:160]}")


def test_the_probe_reports_its_own_coverage(capsys):
    with capsys.disabled():
        print("\n  ── L47-B1 · GUIDED-170 ──")
        print(f"  fixtures reproducing the defect  {len(NHANES)}")
        print(f"  non-nutrient columns refused     {len(NOT_NUTRIENTS)}")
        print(f"  names the pack recognizes        "
              f"{len(N.NUTRIENT_NAMES)} + {len(N.AI_ONLY)} AI-only "
              f"+ {len(N.SKEWED_REQUIREMENT)} skewed")
        print(f"  shapes NOT covered               {len(SHAPES_NOT_COVERED)}")
        for shape in SHAPES_NOT_COVERED:
            print(f"      · {shape}")

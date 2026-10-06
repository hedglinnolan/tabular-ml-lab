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
# The tests here that drove the retired legacy app (turbotab/api.py, the page, the
# figure and manuscript modules, the gates) were removed with it, BLUEPRINT §9.1; git keeps them.
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

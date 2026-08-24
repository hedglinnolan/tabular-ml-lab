"""L41-B — eight clinical import detectors, and the upload that reaches them.

`research/CLINICAL_SURVEY_PACK.md` §A1.1, §A1.2 and §A1.3, plus
`DOMAIN_SCIENCE.md` §03b's clinical rows. The clinical pack held **one prior and
zero detectors** against a 1,209-line research file, under a comment arguing the
thinness was the point because physiologic bounds and unit harmonization already
live in the core.

**That argument was half right and the half that was wrong is the whole part.**
It is true of §A1.2 — this pack reads `ml/physiology_reference.py`'s bands
rather than carrying its own, and `impossible_vs_extreme` is a *correction to a
reading the core already produces* rather than a second set of bounds. It was
never true of §A1.3, which specifies censoring tokens, detection limits inferred
from the data, and result columns carrying a qualifier inside the value. Nothing
in this repository did any of that.

## Hardest-first, and what the order bought

`LOOP.md` §02: *judge hardest by what is most likely to break the abstraction,
not by effort.* `censored_values` went first because it produces a **per-analyte
table** where every other detector in this repository produces one reading. It
did not break the contract — the payload shape was already free — and it did
find the thing the ordering exists to find: `atwater_finding`'s rule that a
varying finding id cannot be bound to anything, arriving on a detector that
wanted eight ids and gets one.

**What the ordering actually caught was two defects in the seventh detector**,
and both were found by running it against a fixture it was not written for:

- `temporal_implausibility` read `clinical_longitudinal.csv`'s seeded
  `height_cm = 0.0` as *"an adult changed height by 164.7 cm."* Every word
  arithmetic and none of it a trajectory. It now drops the cells the atemporal
  check already calls impossible, **and reports how many** — see
  `test_a_trajectory_is_not_traced_through_an_entry_error`.
- It measured `max − min` across all visits where §A1.2 says *between visits*.
  Four visits of 2 cm measurement noise sum to 6 and report a jump nobody made.

And the same lens caught one in the eighth: `impossible_vs_extreme` would have
told a user that **120 correctly-measured glucose values are physiologically
impossible entry errors, set them to missing** — because 120 of them are in
mmol/L. The core already knew, and had said so in a field nothing was reading.

## `GUIDED-097` — the fixture rule

Two clinical fixtures of different shape, plus the four the pack must stay
silent on. `SHAPES_NOT_COVERED` names what neither reaches.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from turbotab import clinical as C
from turbotab import packs as P

DATA = Path(__file__).resolve().parent / "sample_data"

#: `GUIDED-097`. Two clinical tables of deliberately different shape — a lab
#: extract carrying every §A1.3 format problem, and the generic clinic table
#: that carries exactly one of them. The second is the more useful arm: a
#: detector suite verified only against the file it was written for is a suite
#: verified against its own fixture.
CLINICAL_FIXTURES = {
    "a messy multi-site lab extract": ("clinical_labs", "readmitted"),
    "the generic clinic table": ("clinic_visits", "outcome"),
}

#: The four the clinical pack describes nothing about. Silence here is the
#: guard `DOMAIN_PACKS.md` §05 calls the one that matters: a pack firing on the
#: wrong data asserts something false authoritatively, and the lens is answered
#: before anything else in the interview.
NOT_CLINICAL = ("metabolomics_untargeted", "survey_instrument",
                "genomics_expression", "dietary_recalls")

#: NOT COVERED, said out loud. A sweep that reports only what it covered has not
#: reported its coverage.
#:
#: SCIENTIFIC NOTATION AS A PARSE FAILURE. `bnp` is written `6.4E+01` and pandas
#: parses it natively, so the column arrives `float64` and the format detector
#: never sees it. §A1.3 lists it beside separators and decimal commas as though
#: the three fail the same way; here they do not. The pattern is kept because it
#: is reachable on a column that is text for another reason, and the limit is
#: stated rather than engineered around — making it fire would have meant
#: re-reading the source file from inside a detector whose contract is a
#: DataFrame.
#:
#: A REAL PEDIATRIC COHORT. §A1.2 forbids adult height bounds on growth data and
#: the detector excludes rows under 20 and reports the count. Every patient in
#: both fixtures is an adult, so the exclusion is exercised by a constructed
#: frame below rather than by a fixture.
#:
#: LABS TIMESTAMPED AFTER DEATH. Neither fixture carries a death date — both are
#: live outpatient cohorts. Same treatment: constructed below, named here.
#:
#: AN ANALYTE WHOSE CONVERSION FACTOR DEPENDS ON MOLECULAR WEIGHT. That is
#: exactly the class `CONVERSIONS` refuses to hold, so there is nothing to cover
#: — the absence is the position rather than a gap.
#:
#: A CENSORED COLUMN REACHING THE FIT. `blocks_substitution` is the second
#: consumer of the recorded purpose and it is a **capability with a failing
#: test**, not a wired path: nothing in the missingness route yet asks how a
#: censored column should be handled. See the xfail below.
SHAPES_NOT_COVERED = [
    "scientific notation on a column that is otherwise clean — pandas parses "
    "it, so it arrives numeric and this detector never sees it",
    "a pediatric cohort — both fixtures are adults; the age exclusion is "
    "covered by a constructed frame",
    "labs timestamped after death — neither fixture carries a death date",
    "an analyte whose conversion factor depends on molecular weight — "
    "deliberately absent from CONVERSIONS, which is the hard stop rather than "
    "a gap",
    "a censored column reaching the fit — blocks_substitution has no consumer "
    "in the missingness route yet, and the xfail below names it",
]


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / f"{name}.csv")


# ═══════════ EVERY DETECTOR REACHES AN UPLOAD ═══════════

def test_every_detector_fires_from_an_upload_and_not_from_its_own_test():
    """**`GUIDED-058`'s class, checked before it can recur.**

    L27 built four nutrition detectors and a refusal, all correct, all tested,
    and imported by nothing but their own tests. The rule that came out of it is
    that a capability ships with the path that consumes it, and the check is not
    *does something import this* — it is *does an upload reach it.*

    Driven through the API: a file, a lens answer, and the findings the project
    serves. Not `clinical.findings(df)`, which would prove the module and prove
    nothing about the app.
    """
    from fastapi.testclient import TestClient

    from turbotab import api

    client = TestClient(api.app)
    with open(DATA / "clinical_labs.csv", "rb") as handle:
        project_id = client.post("/project", files={
            "file": ("clinical_labs.csv", handle, "text/csv")}).json()["id"]
    answered = client.post(f"/project/{project_id}/decision", json={
        "kind": "set_lens", "payload": {"lens": [P.CLINICAL]}})
    assert answered.status_code == 200, answered.text

    served = client.get(f"/project/{project_id}").json()["findings"]
    reached = {f["id"] for f in served if f["id"].startswith("pack::clinical::")}
    declared = {f"pack::clinical::{name}" for name in (
        "censored_values", "text_numeric", "mixed_result_type", "mixed_units",
        "default_value_mass", "temporal_implausibility", "number_format",
        "impossible_vs_extreme")}
    assert reached == declared, (
        f"these detectors are registered and no upload reaches them: "
        f"{sorted(declared - reached)}")

    # AND EVERY ONE CARRIES ITS BADGE AND ITS SOURCE ON THE WIRE. The evidence
    # gate checks the call site; this checks that the boundary did not drop it,
    # which is `DRIVE-001`'s class and the reason `PackRefusal` has one
    # serializer.
    for finding in served:
        if not finding["id"].startswith("pack::clinical::"):
            continue
        badge = finding["evidence"]
        assert badge["evidence_status"] in ("SETTLED", "CONVENTION", "DISPUTED")
        assert badge["source"].startswith("research/CLINICAL_SURVEY_PACK.md#")


#: `GUIDED-142`. Every pack that has detectors, with a fixture that fires them
#: and the target that opens Explore. Parametrized over all of them rather than
#: over the clinical one, because the defect was never clinical: nothing
#: rendered ANY pack's findings, and a test naming one pack would have passed
#: with the other four still invisible.
PACKS_WITH_DETECTORS = {
    "clinical": ("clinical_labs.csv", "clinical", "readmitted", 8),
    "dietary": ("nhanes_kilojoules.csv", "dietary", "DR1TKCAL", 4),
    # 3 -> 5 at L50-D. `METABOLOMICS_PACK.md` §01's three diagnostic families
    # were filled out and the pack went from three detectors to thirteen; two
    # more of them fire on this fixture, the role census and the acquisition
    # inventory. The number is an INVENTORY of what the fixture produces rather
    # than a behavior being pinned, which is why moving it is a correction and
    # not trap #3c — the property this test asserts, that every pack finding
    # reaches a person, is unchanged and is now asserted over five.
    "metabolomics": ("metabolomics_untargeted.csv", "metabolomics",
                     "responder", 5),
    "survey": ("survey_sentinels.csv", "survey", "sought_support", 2),
    # TWO SINCE L50-B: `GENOMICS_PACK.md` §02's data-type reading joined the
    # p/n one. The count is written out rather than derived for the reason the
    # rest of this table is — a detector that stops firing has to change a
    # number here, and a length read off the pack would move with it silently.
    "genomics": ("genomics_expression.csv", "genomics", "condition", 2),
}


@pytest.mark.parametrize("lens", sorted(PACKS_WITH_DETECTORS))
def test_every_pack_finding_reaches_a_person_and_carries_its_badge(lens):
    """**`GUIDED-142`, and it is the largest instance of trap #6 this door has
    had.**

    `bySource("profile")` and `bySource("structure")` were the only two callers
    in the page. Every finding a LENS produces — the Atwater reconstruction, the
    pooled-QC rows, the mixed-unit analyte, the sentinel codes, all eight of
    L41-B's — was computed correctly, served correctly on `/project/{id}`, and
    **rendered nowhere.** Five packs, eighteen detectors.

    That is `GUIDED-058`'s class one layer past where L28 closed it: the L28
    fix made the detectors reachable from an upload through the API, and the
    test that closed it never drove the page. `GUIDED-075` is the same story
    about `/figures` and cost two loops.

    **And the badge travels with them**, because a pack claim without one is
    the uniform confidence `DOMAIN_SCIENCE.md` §01.1 exists to end — and the
    badge is nested on a finding, which is why the flat renderer did not pick
    it up for free.
    """
    from fastapi.testclient import TestClient

    from turbotab import api
    from turbotab import pageharness as PH

    if not PH.available():
        pytest.skip("no JS engine on this machine")

    fixture, key, target, expected = PACKS_WITH_DETECTORS[lens]
    client = TestClient(api.app)
    with open(DATA / fixture, "rb") as handle:
        pid = client.post("/project", files={
            "file": (fixture, handle, "text/csv")}).json()["id"]
    for kind, payload in (("set_lens", {"lens": [key]}),
                          ("set_target", {"column": target})):
        ok = client.post(f"/project/{pid}/decision",
                         json={"kind": kind, "payload": payload})
        assert ok.status_code == 200, (kind, ok.text[:300])

    project = client.get(f"/project/{pid}").json()
    served = [f for f in project["findings"] if f["source"] == "pack"]
    assert len(served) == expected, (
        f"{lens} serves {len(served)} pack findings, not {expected}: "
        f"{[f['id'] for f in served]}")

    routes = {
        f"/project/{pid}": project,
        f"/project/{pid}/interview?step=data":
            client.get(f"/project/{pid}/interview?step=data").json(),
        f"/project/{pid}/interview?step=explore":
            client.get(f"/project/{pid}/interview?step=explore").json(),
        f"/project/{pid}/evidence/missingness": {"cards": []},
        f"/project/{pid}/capabilities":
            client.get(f"/project/{pid}/capabilities").json(),
    }
    # THE STACK IS BOUNDED SINCE `GUIDED-149`, so "reaches a person" is no
    # longer "is in `profList`". A finding that ranked below the bound reaches a
    # person THROUGH THE COUNTED AFFORDANCE, and the honest form of this claim is
    # to press it — which makes the assertion stronger than it was, because a
    # collapsed group whose expand did not work would now fail here rather than
    # pass on a card nobody can open.
    #
    # `LOOP.md` trap #3c in the direction it is usually not read: this test went
    # red against a correct change, its NAME states the property worth keeping,
    # and what had to move was the locus of the assertion rather than the claim.
    out = PH.run(
        "var shut = (__harness.html('profList') || '');\n"
        "__harness.dispatch('click', __harness.target("
        "{'data-stack-more':'1','aria-expanded':'false'}));\n"
        "__emit({shut: shut.slice(0, 90000),"
        " open: ((__harness.html('profList') || '') +"
        "        (__harness.html('profRest') || '')).slice(0, 200000),"
        " more: (__harness.html('profMore') || '')});",
        routes=routes, search=f"?project={pid}")
    html = out["open"]
    assert out["shut"], "the Explore findings list rendered nothing at all"

    missing = [f["id"] for f in served if f["title"][:28] not in html]
    assert not missing, (
        f"the {lens} pack computes {missing} and the page never shows them, "
        f"pushed or collapsed. Server-composed and never rendered is the class "
        f"this door has already paid for at six surfaces.")

    # AND THE BOUND MAY NOT SWALLOW ONE SILENTLY. If any pack finding is only
    # reachable behind the affordance, the affordance has to have said so — the
    # count it states is the count behind it, which is the property `GUIDED-149`
    # turns on and the one an off-by-one would break invisibly.
    stack = project["explore_stack"]
    behind = [f["id"] for f in served if f["id"] in stack["collapsed"]]
    if behind:
        assert str(stack["remainder"]["n"]) in out["more"], (
            f"{len(behind)} {lens} findings are behind an affordance that does "
            f"not state its count: {out['more'][:200]}")

    # THE BADGE, because a pack claim without one is the app being uniformly
    # confident — and the finding's is NESTED, so a renderer written for the
    # question's flat shape shows nothing and raises nothing.
    statuses = set(re.findall(r'class="badge (\w+)"', html))
    expected_statuses = {f["evidence"]["evidence_status"].lower()
                         for f in served}
    assert expected_statuses <= statuses, (
        f"these badge statuses are on the wire and not on the page: "
        f"{sorted(expected_statuses - statuses)}")


def test_no_detector_offers_a_repair():
    """§A1.1's design rule — *detect, propose, require explicit confirmation* —
    and `DOMAIN_SCIENCE.md` §01.2's litmus: **can the data distinguish the
    causes of what I just detected?** For all eight it cannot.

    Checked as a property of the payload rather than of any one detector,
    because `router._is_repairable` reads `fix_kind` and a single detector that
    grew one would turn a report into a question the interview asks.
    """
    for shape in CLINICAL_FIXTURES:
        name, _target = CLINICAL_FIXTURES[shape]
        for finding in C.findings(load(name)):
            assert finding["fix_kind"] == "none", (
                f"{finding['id']} offers `{finding['fix_kind']}` on {name}; "
                f"the pack detects and declares, and the user executes")
            assert finding["fix_label"] == ""


@pytest.mark.parametrize("fixture", NOT_CLINICAL)
def test_the_pack_is_silent_on_a_table_it_does_not_describe(fixture):
    """The nearest misses are deliberate: `metabolomics_untargeted.csv` carries
    left-censored missingness, which is the same *word* as §A1.3's censoring and
    a different thing — there the values are blank, here they are text carrying
    an operator."""
    found = C.findings(load(fixture))
    assert found == [], [f["id"] for f in found]


# ═══════════ 1 · §A1.3 · CENSORING, AND THE PER-ANALYTE TABLE ═══════════

def test_the_censoring_summary_is_one_row_per_analyte():
    """§A1.3 specifies the table's columns exactly — *analyte | n | % below LOD
    | LOD value(s) | % above ULOQ* — and the per-analyte shape is the content.

    Folding two analytes into one fraction would report a detection limit
    neither of them has, because a limit belongs to an assay rather than to a
    study.
    """
    finding = C.censored_values_finding(load("clinical_labs"))
    assert finding is not None
    analytes = {row["analyte"]: row for row in finding["params"]["analytes"]}

    crp = analytes["hs_crp"]
    assert crp["detection_limit"] == 0.3
    assert crp["n_below_lod"] == 55
    assert crp["pct_below_lod"] == pytest.approx(19.1, abs=0.1)
    assert crp["n_above_uloq"] == 0

    ferritin = analytes["ferritin"]
    assert ferritin["n_above_uloq"] == 22
    assert ferritin["n_below_lod"] == 0

    assert finding["params"]["worst_censored_fraction"] == pytest.approx(0.191,
                                                                        abs=0.001)


def test_tntc_and_qns_are_measurement_failures_and_never_censoring():
    """§A1.3, verbatim: *"`TNTC` and `QNS` are not censoring at a detection
    limit — they are measurement failures. Treat them as missing, not as
    extreme values."*

    Counted in their own field. A summary that folded them into `n_below_lod`
    would assert a detection limit that was never reached, and one that dropped
    them would lose 14 unusable specimens.
    """
    finding = C.censored_values_finding(load("clinical_labs"))
    wbc = next(row for row in finding["params"]["analytes"]
               if row["analyte"] == "wbc")
    assert wbc["n_measurement_failure"] == 14
    assert wbc["n_below_lod"] == 0 and wbc["n_above_uloq"] == 0
    assert set(wbc["measurement_failure_tokens"]) == {"qns", "tntc"}
    assert "not** censoring" in finding["detail"] or "**not**" in finding["detail"]


def test_the_detection_limit_is_the_modal_value_and_survives_a_typo():
    """§A1.3: *"usually inferable as the modal `<X` value."*

    Modal rather than minimum, and this is the reason: one mistyped `<0.03`
    beside two hundred `<0.3` moves a minimum by an order of magnitude and moves
    a mode not at all. The typo still travels, in `detection_limits_seen`,
    because two limits in one analyte is an assay change mid-study and that is a
    real finding rather than a rounding problem.
    """
    values = ["<0.3"] * 60 + ["<0.03"] + [f"{v:.2f}" for v in np.linspace(0.4, 9, 60)]
    reading = C.read_censoring(pd.Series(values, name="hs_crp"))
    assert reading.detection_limit == 0.3
    assert reading.limits_seen == (0.03, 0.3)
    assert reading.n_left == 61


def test_all_three_positions_travel_and_none_is_collapsed():
    """**The three §A1.3 states, carried separately.** Collapsing them is what
    the evidence badge exists to prevent: a proven cutoff, a live dispute and a
    well-argued convention rendered identically is uniform confidence wearing a
    badge that says otherwise.
    """
    finding = C.censored_values_finding(load("clinical_labs"))
    claims = {c["key"]: c for c in finding["evidence"]["claims"]}
    assert claims["threshold"]["evidence_status"] == "CONVENTION"
    assert claims["substitution"]["evidence_status"] == "DISPUTED"
    assert claims["substitution"]["both_sides"], (
        "a DISPUTED claim with one position stated is the app picking a side "
        "while wearing a badge that says it has not")
    assert claims["purpose_asymmetry"]["evidence_status"] == "CONVENTION"

    # THE NUMBER IS BESIDE THE THRESHOLD. A threshold with no arithmetic next to
    # it is a threshold nobody can check.
    assert finding["params"]["warn_threshold"] == 0.10
    assert finding["params"]["over_warn_threshold"] is True
    assert "19.1%" in finding["detail"]


def test_the_substitution_band_moves_with_the_censored_fraction():
    """DISPUTED with a shape, not a shrug. Three fractions, three sentences,
    and the arithmetic is stated in each."""
    low = C.substitution_position(0.03)["band"]
    middle = C.substitution_position(0.19)["band"]
    high = C.substitution_position(0.35)["band"]
    assert "below the roughly 5%" in low
    assert "contested band" in middle and "Both positions are live" in middle
    assert "indefensible" in high and "censored regression" in high
    assert len({low, middle, high}) == 3


def test_the_prediction_inference_asymmetry_is_stated_on_the_finding():
    """§A1.3's asymmetry, and it is CONVENTION rather than SETTLED because the
    research marks it *well-argued, not formally settled*."""
    finding = C.censored_values_finding(load("clinical_labs"))
    said = finding["params"]["prediction_asymmetry"]
    assert "available at deployment" in said
    assert "opposite answer" in said
    assert "hs_crp" in said


def test_the_recorded_purpose_is_the_second_consumer():
    """`purpose` was recorded at `GUIDED-048` and read in exactly one place —
    `project.declare_missingness`, for the missing-indicator. This is the
    second, and it is the same shape for the same reason.

    **Unanswered blocks nothing**, which is the load-bearing clause: the app
    does not get to infer an objective and then hold somebody to it.
    """
    from turbotab import purpose as _purpose

    assert C.blocks_substitution(_purpose.INFERENCE, 0.19) is True
    assert C.blocks_substitution(_purpose.PREDICTION, 0.19) is False
    assert C.blocks_substitution(None, 0.19) is False, (
        "an unanswered purpose blocked a handling; the app inferred an "
        "objective and then held the user to it")
    # AND IT IS GATED ON THE FRACTION TOO. The research does not say
    # substitution is always wrong for inference — it says the bias grows with
    # the fraction and that below ~5% it rarely matters. Blocking at 2% would
    # make the app more confident than its own source.
    assert C.blocks_substitution(_purpose.INFERENCE, 0.02) is False

    blocker = C.substitution_blocker("hs_crp", 0.19)
    assert blocker["evidence_status"] == "CONVENTION"
    assert blocker["acknowledgment_kind"] == "typed"
    assert {e["kind"] for e in blocker["exits"]} == {"resolve", "attest"}, (
        "§09's CONSEQUENCE is resolve-or-attest; a block with one exit is a "
        "refusal wearing a block's clothes")


@pytest.mark.xfail(strict=True, reason=(
    "GUIDED-138. `blocks_substitution` is a capability with no consumer: "
    "`project.declare_missingness` routes a blank and nothing in it knows a "
    "column was censored rather than empty, so the purpose contraindication "
    "cannot fire on a real project. Shipped as a failing test naming the "
    "consumer it lacks, per LOOP.md §05."))
def test_a_censored_column_reaches_the_missingness_route():
    from turbotab.project import AnalysisProject

    df = load("clinical_labs")
    project = AnalysisProject.from_dataframe(df, "clinical_labs.csv")
    project.target, project.task_type = "readmitted", "classification"
    project.set_lens([P.CLINICAL])
    project.set_purpose("inference")
    # There is no route that says "this column is censored at 0.3, and here is
    # what that means for your objective". When there is, it lands here.
    assert hasattr(project, "declare_censoring")


# ═══════════ 2 · §A1.3 · TEXT THAT IS MOSTLY NUMBERS ═══════════

def test_the_blocking_values_are_the_finding():
    """A column above 80% numeric-parseable is *near-certain evidence of an
    embedded qualifier*, and the values that stop the parse are what the column
    is recording besides a number. Reporting only the rate would leave a reader
    knowing something is wrong and not what."""
    finding = C.text_numeric_finding(load("clinical_labs"))
    by_column = {c["column"]: c for c in finding["params"]["columns"]}
    assert set(by_column) == {"wbc", "ferritin", "troponin", "hs_crp"}
    assert set(by_column["wbc"]["blocking_values"]) == {"QNS", "TNTC"}
    assert by_column["wbc"]["numeric_parse_rate"] > 0.80
    assert "<0.3" in by_column["hs_crp"]["blocking_values"]


def test_a_genuinely_categorical_column_is_not_flagged():
    """The negative control. `sex` is `F`/`M` and parses as a number in 0% of
    its rows; a detector that flagged it would be recommending that a category
    be read as a measurement."""
    finding = C.text_numeric_finding(load("clinical_labs"))
    named = {c["column"] for c in finding["params"]["columns"]}
    assert "sex" not in named and "site" not in named
    assert "patient_id" not in named


# ═══════════ 3 · §A1.3 · A NUMBER AND A VERDICT IN ONE FIELD ═══════════

def test_the_troponin_column_is_read_as_two_kinds_of_result():
    """§A1.3's own example, and the one a generic profiler gets backwards."""
    finding = C.mixed_result_finding(load("clinical_labs"))
    lead = finding["params"]["columns"][0]
    assert lead["column"] == "troponin"
    assert lead["n_qualitative"] == 41
    assert lead["n_quantitative"] == 247
    assert set(lead["qualitative_values"]) == {"negative", "positive"}
    assert "one-hot" in finding["why_it_matters"]


def test_a_wholly_qualitative_column_is_left_alone():
    """A column of nothing but `negative`/`positive` IS a categorical, and
    reading it as a broken quantitative one would be this detector making the
    profiler's mistake in the opposite direction."""
    series = pd.Series(["negative"] * 40 + ["positive"] * 20, name="hiv_screen")
    frame = pd.DataFrame({"hiv_screen": series})
    assert C.mixed_result_finding(frame) is None


# ═══════════ 4 · §A1.1 · THE HARD STOP ═══════════

def test_the_mixed_unit_column_is_detected_and_never_converted():
    """`DOMAIN_SCIENCE.md` §01.2's first hard stop. High-confidence detection,
    irreversible-if-wrong action, and no signal in the data that resolves the
    ambiguity — so the app declares and the user executes."""
    finding = C.mixed_units_finding(load("clinical_labs"))
    assert finding["severity"] == "critical"
    lead = finding["params"]["columns"][0]
    assert lead["column"] == "glucose"
    assert lead["tabled_factor"] == 18.0
    assert lead["observed_ratio"] == pytest.approx(18.0, abs=1.5)
    assert lead["implied_unit_low"] == "mmol/L"
    assert lead["implied_unit_high"] == "mg/dL"

    # THE HARD STOP IS IN THE PAYLOAD, not only in the prose. `GUIDED-064`'s
    # class: the machine-readable form must not be lossier than the sentence,
    # and *never auto-convert* is the whole content of this finding.
    assert finding["params"]["hard_stop"] == "never_auto_convert"
    assert "molecular weight" in finding["params"]["hard_stop_because"]
    assert finding["fix_kind"] == "none"

    # AND THE TABLE IS UNTOUCHED. Asserted rather than assumed, because a
    # detector that repaired in passing would be the embarrassment §A1.1 names.
    before = load("clinical_labs")["glucose"]
    C.mixed_units_finding(load("clinical_labs"))
    pd.testing.assert_series_equal(before, load("clinical_labs")["glucose"])


def test_a_single_population_analyte_is_not_split():
    """`clinical_risk.csv` carries `creatinine_mg_dl` in one unit. A detector
    that found a conversion in every skewed lab column would be worse than no
    detector, because a `critical` finding is the one a user acts on."""
    assert C.mixed_units_finding(load("clinical_risk")) is None


def test_temperature_is_refused_rather_than_approximated():
    """°F = °C × 1.8 + 32 is **affine, not a ratio**, so a log-scale ratio test
    does not apply to it. It is in `CONVERSIONS` with a `None` factor and
    skipped, rather than given an approximate multiplier that would produce a
    number nobody can check."""
    entry = next(c for c in C.CONVERSIONS if c[0] == "temperature")
    assert entry[2] is None
    # And a temperature column with both scales present is not reported by this
    # detector — the absence is deliberate and is what the `None` encodes.
    frame = pd.DataFrame({"temp": [37.0] * 60 + [98.6] * 60})
    assert C.mixed_units_finding(frame) is None


# ═══════════ 5 · §03b · DEFAULT VALUES ═══════════

def test_the_spike_is_measured_against_the_columns_own_runner_up():
    """120 is also a perfectly ordinary systolic reading, so *how much* mass is
    excess cannot be a fixed count. Measured against the column's second-most-
    common value, and the threshold is **this module's own** — §03b names the
    artifact and states no cut point, exactly as `nutrition.DRIFT_LIMIT` found.
    """
    finding = C.default_value_mass_finding(load("clinical_labs"))
    by_column = {(h["column"], h["value"]): h for h in finding["params"]["values"]}
    assert ("sbp", 120.0) in by_column
    assert ("dbp", 80.0) in by_column
    assert ("temp_f", 98.6) in by_column
    spike = by_column[("dbp", 80.0)]
    assert spike["share"] > spike["next_most_common_share"] * 2
    assert finding["params"]["thresholds_are_this_apps_own"] is True, (
        "an invented threshold shipped without saying so; no bare-number scan "
        "can tell one from a cited one, which is why it is said in the payload")


def test_an_ordinary_vitals_column_has_no_spike():
    """`clinical_risk.csv` has no vitals at all and `clinic_visits.csv`'s
    `bp_1`/`bp_2`/`bp_3` are drawn continuously. Neither should report."""
    assert C.default_value_mass_finding(load("clinical_risk")) is None
    assert C.default_value_mass_finding(load("clinic_visits")) is None


# ═══════════ 6 · §A1.2 · THE TRAJECTORY, NOT THE VALUE ═══════════

def test_the_trajectory_checks_find_what_the_fixture_seeded():
    finding = C.temporal_implausibility_finding(load("clinical_labs"))
    jumps = finding["params"]["height_jumps"]
    assert [j["person"] for j in jumps] == ["PT0007"]
    assert jumps[0]["change_cm"] == pytest.approx(9.0, abs=0.1)

    swings = finding["params"]["weight_swings"]
    assert [s["person"] for s in swings] == ["PT0021"]
    assert swings[0]["days"] == 21
    assert swings[0]["change"] < -C.WEIGHT_CHANGE_FRACTION


def test_a_trajectory_is_not_traced_through_an_entry_error():
    """**The defect this detector shipped with, and the fixture that found it.**

    Run against `clinical_longitudinal.csv`, which seeds `height_cm = 0.0` as an
    entry error, the first version reported *"an adult changed height by
    164.7 cm, from 164.7 to 0."* Arithmetic, and not a trajectory: one cell is an
    entry error the atemporal check already flags, and dressing it as a
    trajectory reports one defect twice in more alarming language.

    Kahn et al.'s split is the reason it matters — atemporal and temporal
    plausibility have different remedies, and a temporal claim is only about
    values that are individually believable.
    """
    df = load("clinical_longitudinal")
    assert (df["height_cm"] == 0.0).any(), "the fixture no longer seeds one"
    finding = C.temporal_implausibility_finding(df)
    if finding is not None:
        for jump in finding["params"]["height_jumps"]:
            assert jump["to"] > 0 and jump["from"] > 0, jump
            assert jump["change_cm"] < 100, (
                "a 100 cm height change is an entry error being read as growth")


def test_the_height_rule_is_between_visits_and_not_across_all_of_them():
    """§A1.2 says *"adult height changing >5 cm between visits."* A spread over
    four visits is a different quantity: 2 cm of measurement noise each visit
    sums to 6 and reports a jump nobody made."""
    creeping = pd.DataFrame({
        "patient_id": ["A"] * 4,
        "visit_date": ["2024-01-01", "2024-04-01", "2024-07-01", "2024-10-01"],
        "age": [55] * 4,
        "height_cm": [170.0, 172.0, 174.0, 176.0],     # 6 cm spread, 2 cm steps
    })
    assert C.temporal_implausibility_finding(creeping) is None

    jumped = creeping.copy()
    jumped["height_cm"] = [170.0, 170.0, 179.0, 179.0]  # one 9 cm step
    finding = C.temporal_implausibility_finding(jumped)
    assert finding is not None
    assert finding["params"]["height_jumps"][0]["change_cm"] == pytest.approx(9.0)


def test_a_growing_child_is_excluded_and_the_exclusion_is_reported():
    """§A1.2, explicit: *never apply adult bounds to pediatric or growth data.*
    A 9 cm year is a finding in a 60-year-old and a normal year in a 12-year-old.
    """
    child = pd.DataFrame({
        "patient_id": ["K"] * 3 + ["A"] * 3,
        "visit_date": ["2024-01-01", "2024-07-01", "2025-01-01"] * 2,
        "age": [12, 12, 13, 61, 61, 62],
        "height_cm": [142.0, 148.0, 154.0, 170.0, 170.2, 170.1],
    })
    finding = C.temporal_implausibility_finding(child)
    assert finding is None, (
        "the child's 6 cm growth was reported as implausible, which is the "
        "error §A1.2 names by name")

    both = child.copy()
    both.loc[both["patient_id"] == "A", "height_cm"] = [170.0, 179.5, 179.5]
    finding = C.temporal_implausibility_finding(both)
    assert [j["person"] for j in finding["params"]["height_jumps"]] == ["A"]
    assert finding["params"]["pediatric_rows_excluded"] == 3
    assert "below age 20" in finding["detail"]


def test_a_lab_after_death_is_found():
    """The branch neither fixture reaches, constructed and named in
    `SHAPES_NOT_COVERED` rather than left silent."""
    frame = pd.DataFrame({
        "patient_id": ["A", "A", "B", "B"],
        "visit_date": ["2024-01-05", "2024-06-05", "2024-01-05", "2024-02-05"],
        "date_of_death": ["2024-03-01", "2024-03-01", None, None],
        "age": [70, 70, 66, 66],
        "creat": [1.1, 1.2, 0.9, 1.0],
    })
    finding = C.temporal_implausibility_finding(frame)
    assert finding is not None
    after = finding["params"]["measurements_after_death"]
    assert [row["person"] for row in after] == ["A"]
    assert after[0]["measured"] == "2024-06-05"


def test_a_cross_section_refuses_rather_than_reporting_zero():
    """**A refusal to compute is not a clean result.** A table with one row per
    patient has no trajectory, and coming back with zero implausible ones would
    say the check ran."""
    assert C.read_temporal(load("clinical_risk")) is None
    assert C.temporal_implausibility_finding(load("clinical_risk")) is None


# ═══════════ 7 · §A1.3 · NUMBER FORMATS ═══════════

def test_the_decimal_comma_is_called_out_as_the_ambiguous_one():
    """Separate from `text_numeric` deliberately. A thousands separator makes
    `float()` fail loudly; a decimal comma can parse **successfully and be wrong
    by a factor of a hundred**, and reporting both as *some values did not
    parse* would lose that."""
    finding = C.number_format_finding(load("clinical_labs"))
    by_column = {c["column"]: c["format"] for c in finding["params"]["columns"]}
    assert by_column["creatinine"] == "european_decimal_comma"
    assert by_column["platelets"] == "thousands_separator"
    assert "a hundred times" in finding["detail"]


def test_scientific_notation_is_absent_because_pandas_parses_it():
    """The limit, asserted so it stays true rather than living in a comment.

    `bnp` is written `6.4E+01` in the file and arrives as `float64`, so this
    detector never sees it. Making it fire would have meant re-reading the
    source file from inside a detector whose contract is a DataFrame.
    """
    df = load("clinical_labs")
    assert pd.api.types.is_numeric_dtype(df["bnp"]), (
        "bnp is no longer parsed as a number, so this test is measuring "
        "something else")
    finding = C.number_format_finding(df)
    formats = {c["format"] for c in finding["params"]["columns"]}
    assert "scientific_notation" not in formats

    # AND THE PATTERN IS REACHABLE, on a column that is text for another
    # reason — which is the only case where it says anything a reader needs.
    mixed = pd.DataFrame({"bnp": ["1.2E3"] * 40 + ["QNS"] * 5})
    reached = C.number_format_finding(mixed)
    assert reached is not None
    assert reached["params"]["columns"][0]["format"] == "scientific_notation"


# ═══════════ 8 · §03b · THE COACHING SENTENCE ═══════════

def test_the_two_counts_travel_in_one_sentence():
    """§03b's line, and the two counts together are the whole content:

    > *"4 systolic values are below 30 mmHg — physiologically impossible…
    > **This is different from the 812 values above 140 mmHg, which are abnormal
    > but real and must be kept.**"*

    Reported separately they read as two severities of the same thing, which is
    exactly the reading that deletes the sickest patients.
    """
    finding = C.impossible_vs_extreme_finding(load("clinical_labs"))
    assert finding is not None
    sbp = next(c for c in finding["params"]["columns"] if c["column"] == "sbp")
    assert sbp["n_impossible"] == 4
    assert sbp["n_abnormal_but_possible"] > 0

    detail = finding["detail"]
    assert "physiologically impossible" in detail
    assert "abnormal but real and must be kept" in detail
    assert "sickest patients" in detail
    assert str(sbp["n_impossible"]) in detail
    assert f"{sbp['n_abnormal_but_possible']:,}" in detail

    assert "no generic outlier rule" in finding["why_it_matters"]
    assert "3 SD" in finding["why_it_matters"] or "±3 SD" in finding["why_it_matters"]


def test_a_mixed_unit_column_is_set_aside_rather_than_called_entry_errors():
    """**The defect the ordering caught, and it would have been the worst one
    here.**

    120 of `glucose`'s values are in mmol/L, so 42% of the column falls below
    the mg/dL floor. Without this filter the coaching card would have said *120
    glucose values are physiologically impossible and are almost certainly entry
    errors, set them to missing* — about 120 correctly-measured values.

    The core already knew: `plausibility_report` sets `whole_column_suspect` and
    says the reading is about the column rather than its entries. Nothing was
    reading it. `AUDIT-008` arriving inside a detector written to avoid it.
    """
    finding = C.impossible_vs_extreme_finding(load("clinical_labs"))
    named = {c["column"] for c in finding["params"]["columns"]}
    assert "glucose" not in named, (
        "the coaching card is telling a user to delete 120 correctly-measured "
        "values because the column is in two units")
    assert finding["params"]["columns_the_core_could_not_read"] == ["glucose"], (
        "the column was set aside silently; what was not checked is a count, "
        "not an omission")

    # AND THE UNIT FINDING IS THE ONE THAT SPEAKS FOR IT, at `critical`.
    units = C.mixed_units_finding(load("clinical_labs"))
    assert units["severity"] == "critical"
    assert "glucose" in units["affected_columns"]


def test_it_reads_the_cores_bands_and_never_carries_its_own():
    """A pack whose physiologic bounds disagreed with the core's would be worse
    than a pack with none. The bands in the payload are the ones
    `engine.plausibility` reported, and the reference version travels with
    them."""
    finding = C.impossible_vs_extreme_finding(load("clinical_labs"))
    assert finding["params"]["reference_version"]

    from turbotab import engine
    report = engine.plausibility(load("clinical_labs"))
    core_bands = {row["column"]: [row["low"], row["high"]]
                  for row in report["impossible"]}
    for column in finding["params"]["columns"]:
        assert column["impossible_band"] == core_bands[column["column"]]

    # AND NO BOUNDS TABLE OF ITS OWN. Asserted over the module source, because
    # the guarantee is a subtraction: a second table would be a number in this
    # file that the core also holds.
    import ast
    tree = ast.parse(open("turbotab/clinical.py").read())
    names = {t.id for node in ast.walk(tree)
             if isinstance(node, ast.Assign)
             for t in node.targets if isinstance(t, ast.Name)}
    for banned in ("PLAUSIBILITY_BOUNDS", "REFERENCE_INTERVALS",
                   "PHYSIOLOGIC_BOUNDS"):
        assert banned not in names, (
            f"{banned} is a second bounds table beside the core's")

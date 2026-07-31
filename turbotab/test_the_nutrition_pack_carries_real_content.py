"""`GUIDED-055`–`057` — the nutrition pack, and the refusal that justified it.

This is the loop that tests whether the pack architecture carries real domain
content, and nutrition went first for one reason: **it is the pack with a
refusal in it.** A pack that can only add findings has not been tested. §07's
figure E is the path that says whether the architecture holds when the correct
answer is *no*.

## The two detectors

**The Atwater reconstruction** (§01). `E_hat = 4P + 4C + 9F + 7A` against the
declared energy column. Nothing else in the app can infer an energy unit — the
impossibility bands know physiology, not arithmetic — and the row that matters
most is the last one in the research's table: *the ratio drifts with total
energy* means mixed units across rows, which is a hard fail rather than a
conversion.

**The order of those two readings is itself the finding.** A table with mixed
units also has a median ratio, and it will often sit near one of the clean
factors. Reading the median first and the spread second is exactly how a
multi-source merge gets "converted" by a factor that describes some of its rows.
So the spread is the gate and no factor is proposed until it passes.

**NHANES survey design** (§01). The dietary weights rather than the examination
weight; a weight with no strata or PSU as a partially specified design; and a
stratum with one PSU, which does not degrade Taylor-series variance estimation
but breaks it — one PSU has no between-PSU spread, so the contribution is
undefined rather than small.

## What the refusal has to do

Four refusals, each for a different reason, and each **offering what it can
draw** — because a refusal that offers nothing is indistinguishable from a
missing feature, and the user still has a real question.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbotab import figure_specs as FS                               # noqa: E402
from turbotab import nutrition as N                                   # noqa: E402
from turbotab import packs as P                                       # noqa: E402


def _nhanes(n=400, energy_factor=1.0, drift=False, seed=0, design=True):
    """An NHANES-shaped table: SEQN, DR1T* nutrients, design variables.

    Built to the schema §01 names rather than to generic column names, because
    the detector's first signal is that schema and a fixture using `kcal` and
    `protein_g` would test a different path.
    """
    rng = np.random.default_rng(seed)
    protein = rng.gamma(9, 9, n)
    carb = rng.gamma(9, 28, n)
    fat = rng.gamma(7, 11, n)
    alcohol = rng.gamma(1, 4, n)
    reconstructed = 4 * protein + 4 * carb + 9 * fat + 7 * alcohol
    if drift:
        # A multi-source merge: half the rows in kJ, half in kcal.
        factor = np.where(np.arange(n) < n // 2, 1.0, N.KCAL_PER_KJ)
    else:
        factor = np.full(n, energy_factor)
    df = pd.DataFrame({
        "SEQN": np.arange(1, n + 1),
        "DR1TKCAL": reconstructed * factor,
        "DR1TPROT": protein,
        "DR1TCARB": carb,
        "DR1TTFAT": fat,
        "DR1TALCO": alcohol,
    })
    if design:
        df["WTDRD1"] = rng.gamma(4, 6000, n)
        df["WTMEC2YR"] = rng.gamma(4, 6000, n)
        df["SDMVSTRA"] = rng.integers(100, 115, n)
        df["SDMVPSU"] = rng.integers(1, 3, n)
    return df


# ── the Atwater reconstruction ──────────────────────────────────────────────

def test_a_consistent_table_reconstructs_and_raises_nothing():
    reading = N.atwater(_nhanes())
    assert reading is not None and reading.verdict == "pass"
    assert N.PASS_LOW <= reading.ratio <= N.PASS_HIGH
    assert N.atwater_finding(_nhanes()) is None, (
        "a table whose units are consistent produced a finding")


def test_an_energy_column_in_kilojoules_is_named_as_such():
    reading = N.atwater(_nhanes(energy_factor=N.KCAL_PER_KJ))
    assert reading.verdict == "energy_in_kj"
    assert abs(reading.ratio - N.KCAL_PER_KJ) < 0.05
    finding = N.atwater_finding(_nhanes(energy_factor=N.KCAL_PER_KJ))
    assert "kilojoule" in finding["detail"]
    assert finding["severity"] == "warning"


def test_the_inverse_mislabeling_is_a_different_verdict():
    reading = N.atwater(_nhanes(energy_factor=1.0 / N.KCAL_PER_KJ))
    assert reading.verdict == "energy_inverse"


def test_macros_as_percent_of_energy_are_recognized():
    df = _nhanes()
    total = df["DR1TKCAL"]
    for col, kcal in (("DR1TPROT", 4), ("DR1TCARB", 4), ("DR1TTFAT", 9),
                      ("DR1TALCO", 7)):
        df[col] = df[col] * kcal / total * 100.0
    reading = N.atwater(df)
    assert reading.verdict == "macros_not_grams", reading.verdict


def test_mixed_units_across_rows_is_a_hard_fail_and_offers_no_repair():
    """**The row that matters most.** A multi-source merge has a median ratio
    too, and it will often sit near a clean factor — so the drift is the gate
    and no conversion is proposed until it passes."""
    reading = N.atwater(_nhanes(drift=True))
    assert reading.verdict == "mixed_units", reading.verdict
    assert reading.drift > N.DRIFT_LIMIT

    finding = N.atwater_finding(_nhanes(drift=True))
    assert finding["severity"] == "critical"
    assert finding["fix_kind"] == "none", (
        "a repair was offered for a table with no single factor to apply — it "
        "would corrupt every row it did not describe")
    assert not finding["fix_label"]
    assert "separated by source" in finding["detail"]


def test_the_drift_gate_runs_before_any_factor_is_proposed():
    """The ordering, asserted as behavior rather than as a comment.

    A table half in kJ and half in kcal has a median ratio near 2.6 — outside
    every clean band — but the point is that the app must not even look. A
    fixture whose mixed factors AVERAGE to a clean one is the test that would
    catch reading the median first.
    """
    df = _nhanes(drift=True, seed=7)
    # Two-thirds kcal, one-third kJ: the median lands at 1.0, squarely in the
    # PASS band, and only the spread reveals it.
    n = len(df)
    reconstructed = (4 * df["DR1TPROT"] + 4 * df["DR1TCARB"]
                     + 9 * df["DR1TTFAT"] + 7 * df["DR1TALCO"])
    factor = np.where(np.arange(n) < int(n * 0.6), 1.0, N.KCAL_PER_KJ)
    df["DR1TKCAL"] = reconstructed * factor
    reading = N.atwater(df)
    assert 0.9 <= reading.ratio <= 1.1, (
        f"the fixture's median ratio is {reading.ratio}; it must sit in the "
        f"PASS band or this test does not exercise the ordering")
    assert reading.verdict == "mixed_units", (
        "the median ratio passed and the app called the table consistent — "
        "reading the median before the spread is how a mixed merge gets "
        "silently converted")


def test_a_table_without_enough_macronutrients_reads_nothing():
    """A reconstruction resting on one macronutrient would make a passing ratio
    mean nothing, so it is not attempted."""
    df = _nhanes()[["SEQN", "DR1TKCAL", "DR1TPROT"]]
    assert N.atwater(df) is None


# ── the survey design ───────────────────────────────────────────────────────

def test_the_dietary_weights_are_named_and_the_exam_weight_is_named_as_wrong():
    findings = {f["id"]: f for f in N.design_findings(_nhanes())}
    weights = findings["pack::dietary::survey_weights"]
    assert "WTDRD1" in weights["detail"]
    assert "WTMEC2YR" in weights["detail"]
    assert "not the right one here" in weights["detail"]
    assert weights["params"]["use"] == ["WTDRD1"]
    assert weights["params"]["not"] == ["WTMEC2YR"]


def test_a_weight_with_no_strata_or_psu_is_a_partially_specified_design():
    df = _nhanes().drop(columns=["SDMVSTRA", "SDMVPSU"])
    ids = {f["id"] for f in N.design_findings(df)}
    assert "pack::dietary::partial_design" in ids
    partial = next(f for f in N.design_findings(df)
                   if f["id"] == "pack::dietary::partial_design")
    assert set(partial["params"]["missing"]) == {"strata", "PSU"}
    assert "too narrow" in partial["detail"]


def test_a_stratum_with_one_psu_is_flagged_as_a_break_not_a_warning():
    """Taylor-series linearization estimates a stratum's variance from the
    spread BETWEEN its PSUs. One PSU has no spread, so the contribution is
    undefined rather than small — the estimator breaks."""
    df = _nhanes(seed=2)
    df.loc[df.index[:8], "SDMVSTRA"] = 999
    df.loc[df.index[:8], "SDMVPSU"] = 1          # a stratum with one PSU
    findings = {f["id"]: f for f in N.design_findings(df)}
    lonely = findings["pack::dietary::lonely_psu"]
    assert lonely["severity"] == "critical"
    assert "999" in lonely["detail"]
    assert "does not degrade, it breaks" in lonely["detail"]
    # And it offers the standard remedies without choosing one.
    for remedy in ("collapse", "centre", "certainty"):
        assert remedy in lonely["detail"]
    assert lonely["fix_kind"] == "none"


def test_a_table_with_no_weights_raises_no_design_findings():
    """Guard #2: a pack must not fire on data it does not describe."""
    assert N.design_findings(_nhanes(design=False)) == []


def test_every_nutrition_finding_offers_no_automatic_repair():
    """All four readings are things the app must detect and must not act on —
    `DOMAIN_SCIENCE.md` §01.2's class. Detection is easy, the action is
    irreversible if wrong, and nothing in the data resolves the ambiguity."""
    df = _nhanes(drift=True)
    df.loc[df.index[:8], "SDMVSTRA"] = 999
    df.loc[df.index[:8], "SDMVPSU"] = 1
    produced = [f for f in [N.atwater_finding(df)] if f] + N.design_findings(df)
    assert len(produced) >= 2
    for finding in produced:
        assert finding["fix_kind"] == "none", finding["id"]
        assert finding["pack"] == N.DIETARY


def test_every_advisory_carries_its_badge_and_a_resolvable_source():
    """`GUIDED-047`, applied to the pack's new content."""
    root = Path(__file__).resolve().parents[1] / "docs" / "turbotab"
    for evidence in (N.ATWATER_EVIDENCE, N.DESIGN_EVIDENCE,
                     N.PREVALENCE_EVIDENCE):
        assert evidence.status in P.EVIDENCE_STATUSES
        filename, _, section = evidence.source.partition("#")
        path = root / filename
        assert path.exists(), filename
        import re
        headings = {m.group(1).strip() for m in
                    re.finditer(r"^#{1,6}\s+(.*?)\s*$", path.read_text(), re.M)}
        assert section in headings, (filename, section)


# ── Part D · the refusal ────────────────────────────────────────────────────

def test_a_prevalence_of_inadequacy_is_refused_against_an_adequate_intake():
    """*"Fiber has an AI, not an EAR. I can show the distribution against the
    AI, but I cannot compute a prevalence of inadequacy from an AI, and neither
    can anyone else."*"""
    for nutrient in ("fiber", "dietary fibre", "potassium"):
        with pytest.raises(N.PrevalenceRefusal) as caught:
            N.prevalence_of_inadequacy(nutrient, basis=N.USUAL_INTAKE,
                                       reference_kind="AI")
        assert "Adequate Intake" in str(caught.value)
        assert "neither can anyone else" in str(caught.value)
        offer = caught.value.offer
        assert offer["draw"] == "distribution_against_ai"
        assert "NOT a prevalence of inadequacy" in offer["caption_note"]


def test_it_is_refused_against_the_rda_because_that_is_an_individual_target():
    """The RDA covers 97–98% of requirements. Counting everyone below it as
    inadequate counts most of an adequately-nourished population as
    deficient."""
    with pytest.raises(N.PrevalenceRefusal) as caught:
        N.prevalence_of_inadequacy("calcium", basis=N.USUAL_INTAKE,
                                   reference_kind="RDA")
    assert "individual-level target" in str(caught.value)
    offer = caught.value.offer
    assert offer["draw"] == "distribution_against_ear_and_rda"
    assert "no area is shaded against it" in offer["caption_note"]


@pytest.mark.parametrize("basis", [N.SINGLE_DAY, N.NAIVE_MEAN])
def test_it_is_refused_from_single_day_and_from_a_naive_mean(basis):
    """*"We averaged the two 24-hour recalls to obtain usual intake"* followed
    by a prevalence claim is a documented failure, not a simplification."""
    with pytest.raises(N.PrevalenceRefusal) as caught:
        N.prevalence_of_inadequacy("calcium", basis=basis, reference_kind="EAR")
    message = str(caught.value)
    assert "usual-intake distribution" in message
    assert "overstated, in both" in message
    assert caught.value.offer["draw"] == "shrinkage", (
        "the refusal offers nothing to look at, which is indistinguishable "
        "from a missing feature")


def test_every_refusal_offers_something_it_can_draw():
    """A refusal that offers nothing is indistinguishable from a missing
    feature, and the user still has a real question.

    **This test used to assert the offer strings were non-empty**, which is the
    shape of the claim and not its resolution — `GUIDED-060`. Two of the four
    offers named a figure that does not exist, and three truthy strings passed
    every time. The target is RESOLVED now: registered, or declared pending
    with what it needs and the row blocking it. An id in neither raises.
    """
    from turbotab import figures
    from turbotab import figure_specs                                 # noqa: F401

    cases = [
        ("fiber", N.USUAL_INTAKE, "AI"),
        ("calcium", N.USUAL_INTAKE, "RDA"),
        ("calcium", N.SINGLE_DAY, "EAR"),
        ("calcium", N.NAIVE_MEAN, "EAR"),
    ]
    for nutrient, basis, kind in cases:
        with pytest.raises(N.PrevalenceRefusal) as caught:
            N.prevalence_of_inadequacy(nutrient, basis=basis,
                                       reference_kind=kind)
        offer = caught.value.offer
        assert offer["draw"] and offer["label"] and offer["caption_note"]
        assert offer["forbidden"], "the refusal does not name what it refused"
        assert len(str(caught.value)) > 120, "the refusal states no reason"

        resolved = figures.resolve(offer["draw"])
        assert resolved["status"] in (figures.REGISTERED_STATUS,
                                      figures.PENDING_STATUS)
        if resolved["status"] == figures.PENDING_STATUS:
            # A pending target is honest only while it says what is missing.
            assert len(resolved["needs"]) > 60, offer["draw"]
            assert resolved["blocked_by"].startswith(("GUIDED-", "DRIVE-")), (
                f"{offer['draw']} is pending and names no ledger row, so "
                f"nothing tracks it back into existence")


def test_a_nutrient_name_is_matched_exactly_and_never_by_substring():
    """The hazard `test_every_substring_match_against_a_name_is_declared`
    caught, in the one code path whose whole job is to refuse correctly.

    The first version asked `any(a in name for a in AI_ONLY)`. `"iron" in
    "environment"` is True, and `"fiber"` matches anything containing it — so a
    refusal would have fired on columns that have nothing to do with either, and
    the probability approach would have been selected for a nutrient that is not
    iron. Exact key or declared alias; an unrecognized name yields silence.
    """
    from ml import name_registry as registry

    # The substring that would have matched, and now does not.
    assert registry.match("environment", N.SKEWED_REQUIREMENT) is None
    assert registry.match("fiber_supplement_user", N.AI_ONLY) is None
    # The real names, and their declared NHANES spellings, still do.
    assert registry.match("iron", N.SKEWED_REQUIREMENT) == "iron"
    assert registry.match("DR1TIRON", N.SKEWED_REQUIREMENT) == "iron"
    assert registry.match("Dietary Fibre", N.AI_ONLY) == "fiber"
    assert registry.match("DR1TFIBE", N.AI_ONLY) == "fiber"

    # And the refusal follows the registry rather than the spelling: a column
    # merely CONTAINING a name is computed, not refused.
    result = N.prevalence_of_inadequacy("fiber_supplement_user",
                                        basis=N.USUAL_INTAKE,
                                        reference_kind="EAR")
    assert result["method"] == "cut_point"


def test_the_valid_case_computes_and_says_which_method():
    result = N.prevalence_of_inadequacy("calcium", basis=N.USUAL_INTAKE,
                                        reference_kind="EAR")
    assert result["method"] == "cut_point"
    assert result["reference_kind"] == "EAR"
    assert result["evidence_status"] == "SETTLED" and result["source"]


def test_iron_in_menstruating_women_takes_the_probability_approach():
    """A skewed requirement distribution, so the cut-point method does not
    apply. Routing rather than refusal — the question has an answer, by a
    different route."""
    result = N.prevalence_of_inadequacy("iron", basis=N.USUAL_INTAKE,
                                        reference_kind="EAR",
                                        stratum="menstruating")
    assert result["method"] == "probability_approach"
    assert "skewed requirement" in result["note"]
    # And iron elsewhere is the ordinary cut-point.
    assert N.prevalence_of_inadequacy(
        "iron", basis=N.USUAL_INTAKE, reference_kind="EAR",
        stratum="men")["method"] == "cut_point"


# ── Part C · the shrinkage plot ─────────────────────────────────────────────

def _three_series(n=500, seed=3):
    rng = np.random.default_rng(seed)
    usual = rng.lognormal(2.6, 0.28, n)
    d1 = usual * rng.lognormal(0, 0.45, n)
    d2 = usual * rng.lognormal(0, 0.45, n)
    return {"single_day": d1, "mean_of_days": (d1 + d2) / 2,
            "usual_intake": usual}


def test_the_shrinkage_checklist_passes_and_the_narrowing_is_real():
    payload = FS.shrinkage_payload(_three_series(), nutrient="calcium",
                                   unit="mg", n_days=2)
    failed = [r for r in FS.SHRINKAGE.score(payload) if not r["passed"]]
    assert not failed, [(r["id"], r["because"]) for r in failed]
    # The claim, as arithmetic: each step narrows.
    assert (payload["spread_usual_intake"] < payload["spread_mean_of_days"]
            < payload["spread_single_day"])


def test_two_series_is_refused_rather_than_drawn():
    """The claim is the narrowing ACROSS three. Two of them is a different
    figure making a weaker claim while wearing this one's caption."""
    series = _three_series()
    del series["usual_intake"]
    with pytest.raises(ValueError, match="needs all three series"):
        FS.shrinkage_payload(series, nutrient="calcium")


def test_the_caption_says_the_third_distribution_is_modeled():
    """Reporting modeled individual predictions as measured usual intakes is a
    named failure in this field."""
    caption = FS.SHRINKAGE.caption(
        FS.shrinkage_payload(_three_series(), nutrient="calcium", unit="mg",
                             n_days=2))
    assert "MODELED" in caption
    assert "not measured usual intakes" in caption
    assert "overstated in both tails" in caption or "both tails" in caption


def test_the_narrowing_item_can_fail():
    """`GUIDED-045`'s axis: an item that cannot fail is not a check.

    If the modeled distribution is not narrower, the figure does not make the
    argument it is drawn to make — and saying so is more useful than drawing it
    anyway.
    """
    payload = FS.shrinkage_payload(_three_series(), nutrient="calcium")
    item = next(i for i in FS.SHRINKAGE.checklist
                if i.id == "narrowing_is_visible")
    assert item.check(payload) is True
    assert item.check(dict(payload, spread_usual_intake=1e9)) is False


def test_the_per_series_annotations_are_keyed_and_not_generalized():
    """What the third figure cost the abstraction, asserted so the note is not
    just prose.

    Every annotation until this figure named ONE number. Here the same
    annotation exists three times, once per series, and it is keyed per series
    rather than by adding an axis to `Annotation` — a generalization on one
    example is how a field becomes a taxonomy nobody can apply.
    """
    keys = {a.key for a in FS.SHRINKAGE.annotations}
    assert keys == {f"{s}_{k}" for k in FS.SERIES for s in ("p05", "p95")}
    assert len(keys) == 6
    # `Annotation` itself is unchanged: still one key, one label, one source.
    for annotation in FS.SHRINKAGE.annotations:
        assert annotation.key and annotation.label and annotation.source


def test_the_third_figure_did_not_need_a_third_tier():
    """The one thing that genuinely did not fit, recorded rather than resolved.

    The shrinkage plot is EXPLORATORY by the two-tier logic — it sees no group
    labels and makes no group claim — but it is the *argument for a method*,
    which is neither exploration nor confirmation. Adding a tier on one example
    is how a two-value distinction becomes a taxonomy nobody can apply, so it
    is filed (`GUIDED-056`) rather than built.
    """
    assert FS.SHRINKAGE.tier == FS.EXPLORATORY
    from turbotab.figures import TIERS
    assert len(TIERS) == 2, (
        "a third tier was added; if that is right it needs more than one "
        "example behind it")

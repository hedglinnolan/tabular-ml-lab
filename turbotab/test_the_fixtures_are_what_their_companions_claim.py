"""A fixture that drifts from its companion turns a drive into a comparison
against a lie.

The five domain fixtures ship with a `.md` stating what the app should surface
and what it should not. That document is the whole reason a drive produces
evidence rather than an impression — *"is this missing finding a bug?"* becomes
*"the companion says it should fire, and it did not"*.

Which makes the companion load-bearing, and a load-bearing document that nothing
checks is `FEATURE_PARITY.md`'s expiring guarantee: true when written, false
later, unannounced. So the numbers the companions quote are asserted here. This
file is the executable half of five markdown documents.

**Two things it is not.**

It does not check the prose. It checks the measured properties the prose quotes —
counts, correlations, spacing, tiers — because *"assert on the object, not on its
rendering"*. A companion whose sentences drift while its numbers hold is a
documentation problem; a companion whose numbers drift is a broken fixture.

And it is not a test of the packs. Nothing here needs a lens, and every assertion
below was true before Task 3 existed. What the lens does with these properties is
tested where the lens lives.

Run:  turbotab/.venv/bin/python -m pytest turbotab/test_the_fixtures_are_what_their_companions_claim.py -q
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbotab import engine, grain as G                              # noqa: E402

DATA = Path(__file__).resolve().parent / "sample_data"

# Every fixture, its companion, and the target a driver would choose.
FIXTURES = {
    "metabolomics_untargeted": "responder",
    "dietary_recalls": "hba1c",
    "clinical_longitudinal": "progressed",
    "survey_instrument": "sought_support",
    "genomics_expression": "condition",
    "clinic_visits": "outcome",
}


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / f"{name}.csv")


def structural(name: str):
    return {f.id: f for f in engine.diagnose(load(name), target=FIXTURES[name])}


# ── every fixture has a companion, and the companion names it ────────────────

@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_every_fixture_ships_with_a_companion_that_names_it(name):
    """A fixture with no stated expectation is a fixture nobody can drive.

    The check is deliberately dumb about quality and precise about existence,
    for the same reason `test_every_clause_is_tracked.py` is: counting the
    quality of an expectation would reward the wrong thing.
    """
    csv = DATA / f"{name}.csv"
    doc = DATA / f"{name}.md"
    assert csv.exists(), f"{csv.name} is missing"
    assert doc.exists(), (
        f"{csv.name} has no companion. The drive compares against a stated "
        f"expectation; without one it is a guess.")
    text = doc.read_text()
    assert f"{name}.csv" in text, "the companion does not name its fixture"
    for heading in ("Must surface", "Must NOT surface", "control"):
        if heading.lower() in text.lower():
            break
    else:                                                # pragma: no cover
        pytest.fail("the companion states neither what must surface nor what must not")


# ── 1 · metabolomics ─────────────────────────────────────────────────────────

def test_the_metabolomics_fixture_is_left_censored_and_drifts_with_run_order():
    """The four properties the metabolomics companion quotes.

    Ordered most diagnostic first: if the shape is wrong nothing below it means
    anything, and a probe reads the first assertion to fire.
    """
    df = load("metabolomics_untargeted")
    assert (len(df), len(df.columns)) == (80, 400)

    feats = [c for c in df.columns if c.startswith("mz_")]
    assert len(feats) == 392

    qc = df[df["sample_type"] == "pooled_qc"]
    assert len(qc) == 8, "eight pooled QC injections"
    assert list(qc["run_order"]) == [1, 11, 21, 31, 41, 51, 61, 71], (
        "QC rows sit at every tenth injection, which is the convention that "
        "makes them recognizable at all")
    # They look exactly like participants except for what they do not carry.
    assert qc[["age", "bmi", "responder"]].isna().all().all()

    # LEFT CENSORING. Missingness is ordered by abundance, not scattered.
    missing_rate = df[feats].isna().mean()
    log_abundance = np.log(df[feats].mean())
    rho = float(pd.Series(missing_rate).corr(pd.Series(log_abundance),
                                             method="spearman"))
    assert rho < -0.9, (
        f"missingness should track abundance rank (measured {rho:.2f}); this is "
        f"the property that makes a median imputation wrong here")
    assert int((missing_rate > 0).sum()) > 250
    assert missing_rate.max() <= 0.6

    # INSTRUMENT DRIFT along acquisition order, on the participant rows.
    part = df[df["sample_type"] == "participant"]
    r = np.array([abs(np.corrcoef(part["run_order"],
                                  np.log(part[c].fillna(part[c].median())))[0, 1])
                  for c in feats])
    share = float((r > 0.3).mean())
    assert share > 0.4, (
        f"only {share:.0%} of features track run order; the run-order finding "
        f"has nothing to detect below about 40%")

    # NOT a clinical table: nothing here is physiologically impossible.
    assert engine.plausibility(df)["impossible"] == []

    # NOT a repeated-measures table: no column repeats like a roster.
    assert G.suggestion(df)["columns"] == []


def test_the_metabolomics_fixture_reproduces_the_wide_shape_false_alarm():
    """`DOMAIN_PACKS.md` §01's first payoff needs something to pay off against.

    Not a pinning test of wrong behavior: `wide_repeated_measures` is `info` and
    is an OFFER, which the constitution permits. What is asserted is that the
    fixture still triggers it — a fixture that stopped would quietly lose the
    demonstration the lens exists to give, and nothing would say so.
    """
    f = structural("metabolomics_untargeted").get("wide_repeated_measures")
    assert f is not None, (
        "the assay fixture no longer trips the wide-shape reading, so the lens "
        "has nothing to correct here")
    assert f.fix_kind == "melt_repeated"
    assert f.severity == "info"


# ── 2 · dietary ──────────────────────────────────────────────────────────────

def test_the_dietary_fixture_is_compositional_with_irregular_short_gaps():
    df = load("dietary_recalls")
    assert (len(df), len(df.columns)) == (600, 17)
    assert df["participant_id"].nunique() == 300
    assert set(df.groupby("participant_id").size().unique()) == {2}

    # THE EVIDENCE QUESTION 4 TURNS ON: short, irregular, no schedule.
    gaps = df.groupby("participant_id")["recall_date"].apply(
        lambda s: (pd.to_datetime(s.iloc[1]) - pd.to_datetime(s.iloc[0])).days)
    assert gaps.min() >= 3 and gaps.max() <= 14
    cv = float(gaps.std() / gaps.mean())
    assert cv > 0.3, f"gaps are too regular (CV {cv:.2f}) to read as replicates"

    # COMPOSITIONAL: four parts of a whole.
    pct = ["protein_pct_kcal", "fat_pct_kcal",
           "carbohydrate_pct_kcal", "alcohol_pct_kcal"]
    total = df[pct].sum(axis=1)
    assert float(total.min()) >= 99.9 and float(total.max()) <= 100.1
    assert df["protein_pct_kcal"].corr(df["carbohydrate_pct_kcal"]) < -0.2, (
        "correlation between parts of a whole is negatively biased by "
        "construction; that bias is the finding")

    # THE OUTCOME IS MEASURED ONCE. Aggregation raises no "which outcome?" here.
    assert (df.groupby("participant_id")["hba1c"].nunique() == 1).all()


def test_the_dietary_implausible_intakes_are_advisory_and_not_impossible():
    """The distinction the whole dietary pack turns on.

    An implausible intake is a bad ESTIMATE of a possible day. It is an
    eligibility criterion the user states, not an entry error the app repairs —
    so it must land in the advisory tier and never in the impossibility tier.
    Confusing the two would turn a decision that changes N into a silent filter.
    """
    df = load("dietary_recalls")
    assert int((df["energy_kcal"] < 500).sum()) == 12
    assert int((df["energy_kcal"] > 5000).sum()) == 8

    report = engine.plausibility(df)
    impossible_columns = [r["column"] for r in report["impossible"]]
    assert "energy_kcal" not in impossible_columns, (
        "257 kcal is a possible day and a bad estimate; the impossibility band "
        "is 100-30,000 and nothing here is outside it")
    assert report["impossible"] == []
    assert "energy_kcal" in [r["column"] for r in report["improbable"]]


def test_the_dietary_fixture_offers_the_person_column_first():
    s = G.suggestion(load("dietary_recalls"))
    assert s["columns"][0] == "participant_id"
    assert s["from_name_heuristic"] == ["participant_id"]


# ── 3 · clinical ─────────────────────────────────────────────────────────────

def test_the_clinical_fixture_is_on_a_schedule_with_a_per_visit_outcome():
    df = load("clinical_longitudinal")
    assert (len(df), len(df.columns)) == (600, 13)
    assert df["subject_id"].nunique() == 200
    assert set(df.groupby("subject_id").size().unique()) == {3}

    # THE EVIDENCE QUESTION 4 TURNS ON, pointing the other way: a schedule.
    gaps = np.concatenate(df.groupby("subject_id")["visit_date"].apply(
        lambda s: np.diff(np.sort(pd.to_datetime(s).values).astype("datetime64[D]")
                          .astype(int))).values)
    assert 80 <= float(np.median(gaps)) <= 100
    cv = float(gaps.std() / gaps.mean())
    assert cv < 0.15, (
        f"visit spacing must read as a schedule (CV {cv:.3f}); irregular "
        f"spacing is what makes dietary read as replicates instead")

    # THE OUTCOME VARIES WITHIN A PERSON. This is why target precedes
    # aggregation: combining rows requires deciding which outcome.
    varies = int((df.groupby("subject_id")["progressed"].nunique() > 1).sum())
    assert varies > 100, f"only {varies} of 200 people change outcome"

    assert G.suggestion(df)["columns"][0] == "subject_id"


def test_the_clinical_impossible_values_are_all_outside_the_impossibility_band():
    """14 cells, 5 columns, each earning a repair proposal rather than an
    advisory. Asserted per column, because a total would pass if one column's
    seeds moved into another's."""
    report = engine.plausibility(load("clinical_longitudinal"))
    flagged = {r["column"]: r["n_flagged"] for r in report["impossible"]}
    assert flagged == {"dbp": 4, "sbp": 3, "weight_kg": 2,
                       "glucose": 2, "height_cm": 3}
    assert sum(flagged.values()) == 14


# ── 4 · survey ───────────────────────────────────────────────────────────────

def test_the_survey_reverse_coded_items_would_reward_an_inference_that_is_wrong():
    """The fixture is built so the forbidden shortcut looks correct.

    The eight reverse-coded items correlate at about -0.56 with the rest, so
    inferring reverse-coding from correlation would get all eight right HERE.
    That is the trap: two subscales measuring opposing constructs produce the
    same evidence, and nothing in the numbers separates them. The app must ask.
    """
    df = load("survey_instrument")
    assert (len(df), len(df.columns)) == (300, 45)

    items = [c for c in df.columns if c.startswith("item_")]
    assert len(items) == 40
    assert sorted(df[items[0]].dropna().unique().tolist()) == [1, 2, 3, 4, 5]

    from turbotab.sample_data.make_fixtures import REVERSE_CODED
    assert len(REVERSE_CODED) == 8
    forward = [c for c in items if c not in REVERSE_CODED]
    scale = df[forward].mean(axis=1)

    reverse_r = float(np.mean([df[c].corr(scale) for c in REVERSE_CODED]))
    forward_r = float(np.mean([df[c].corr(scale) for c in forward]))
    assert reverse_r < -0.4, f"reverse items correlate {reverse_r:.2f}"
    assert forward_r > 0.4, f"forward items correlate {forward_r:.2f}"

    # The outcome is external. A scale total would leak through every item.
    assert "sought_support" in df.columns
    assert df["sought_support"].corr(scale) < 0.85


def test_the_survey_fixture_no_longer_reports_a_critical_it_had():
    """`IMPORT-267`, closed — and this is what the `KNOWN_GAP_` prefix is for.

    This was `test_KNOWN_GAP_a_column_of_education_levels_is_reported_as_mixing_units`,
    pinning the defect so the gap stayed visible: `education` holds `High
    school`, `Some college`, `Bachelors`, `Graduate`, there is no number in it,
    and the app asserted at CRITICAL severity that it mixes measurement units,
    because `_KNOWN_UNITS` contains the bare letters `l` and `s`.

    `FEATURE_PARITY.md` says what happens next in as many words: *"the day it
    goes red is the day the row closes. A `KNOWN_GAP_` test failing is not a
    regression and must not be 'fixed' by editing the assertion. It is the
    signal to update the finding and the test together."* The freeze is lifted
    for repair of dispositioned `IMPORT-*` rows, so the fix landed in
    `_units_present` — a unit needs a measurement in front of it — and the
    pinning test is inverted rather than deleted.

    The behavioral guards live in
    `tests/test_a_unit_needs_a_measurement_in_front_of_it.py`, including the
    ones asserting that a column which GENUINELY mixes `mg/dL` and `mmol/L`
    still says so. This one keeps the fixture honest.
    """
    found = structural("survey_instrument")
    assert "mixed_units__education" not in found, (
        "IMPORT-267 has reappeared: a column with no number in it is being "
        "read as mixing measurement units")
    assert [f.id for f in found.values() if f.severity == "critical"] == []


# ── 5 · genomics ─────────────────────────────────────────────────────────────

def test_the_genomics_fixture_is_counts_at_p_over_n_with_batch_confounded_depth():
    df = load("genomics_expression")
    assert (len(df), len(df.columns)) == (60, 500)

    genes = [c for c in df.columns if c.startswith("gene_")]
    assert len(genes) == 495
    assert len(genes) / len(df) > 8.0, "p/n must be well past 1"

    # COUNTS, not concentrations. The whole pack turns on this.
    assert all(df[g].dtype.kind in "iu" for g in genes)

    # Library size varies, and varies WITH batch — which is why a normalization
    # that ignores batch and a batch correction that ignores normalization are
    # both incomplete, and why the pack declines to pick one.
    lib = df[genes].sum(axis=1)
    assert float(lib.max() / lib.min()) > 2.5
    by_batch = df.assign(_lib=lib).groupby("batch")["_lib"].median()
    assert float(by_batch.max() / by_batch.min()) > 1.1

    assert set(df["batch"]) == {"B1", "B2", "B3"}
    assert set(df["condition"]) == {"case", "control"}


def test_KNOWN_GAP_low_expression_genes_are_read_as_missing_value_codes():
    """A count matrix read as a set of coded survey items.

    `check_numeric_sentinels` treats an integral column with few distinct values
    as CODED, which makes single-digit sentinels such as 7, 8 and 9 credible.
    119 of the 495 genes have a maximum count below 10, so they look exactly
    like short integer scales, and ten of them clear the distance test and are
    reported at `critical`.

    The detector is right about shape and wrong about kind, which is precisely
    what a lens fixes — nothing new is asked, ten criticals become a stated
    fact. Pinned here because until the genomics lens is SET, the app really
    does say this.

    **PASSED means the misreading is still there.**
    """
    found = [k for k in structural("genomics_expression")
             if k.startswith("sentinel_missing__gene_")]
    assert len(found) >= 5, (
        "the count fixture no longer trips the sentinel reading; if the "
        "detector changed, update the companion and this test together")

    df = load("genomics_expression")
    genes = [c for c in df.columns if c.startswith("gene_")]
    tiny = sum(1 for g in genes if int(df[g].max()) < 10)
    assert tiny > 50, f"only {tiny} low-expression genes; the shape is gone"


# ── the control ──────────────────────────────────────────────────────────────

def test_the_control_fixture_has_a_genuine_wide_repeated_measures_family():
    """The same finding is correct here and wrong on three other fixtures.

    `bp_1`, `bp_2`, `bp_3` really are one quantity measured three times, so
    `wide_repeated_measures` is a true reading on `clinic_visits.csv` and a
    false alarm on `mz_0001` and `gene_0001`. The only thing that separates the
    cases is the lens, which is the entire argument for asking the question —
    and this assertion is what stops a pack from suppressing the finding
    globally and calling that a fix.
    """
    f = structural("clinic_visits").get("wide_repeated_measures")
    assert f is not None
    families = f.params["families"]
    assert "bp" in families
    assert sorted(families["bp"]) == ["bp_1", "bp_2", "bp_3"]

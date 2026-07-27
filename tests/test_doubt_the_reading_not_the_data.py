"""Escalate on evidence that the interpretation is wrong — not on the size of
the consequence.

The specific case: a diastolic blood pressure of ~0 is an entry error and gets a
repair proposal. Seventy-five values of `hba1c_proxy` below the HbA1c floor are
*not* seventy-five entry errors; they are one column that is not HbA1c. The
difference is not how many rows a repair would touch — reasoning from that would
make the tool hesitate exactly where it is most useful, and would let a
one-in-a-thousand naming collision through because it is small.

The difference is evidence. `ml.card_evidence.interpretation_verdict` is the
named predicate: it consumes three kinds of signal and returns which reading is
in doubt, with the signals that fired listed by name so the verdict is arguable.

  * **derived name** — the column carries a modifier beside the reference key.
    `match_variable_key` matches by substring (`T0-BUILD-003`), so a derived
    column inherits its parent's intervals wholesale.
  * **rescued by a known unit** — a factor the variable is actually recorded in
    puts the whole column inside its reference interval. The strongest signal
    available: it does not merely say the reading is wrong, it names the right
    one.
  * **coherence** — the violations are one-sided and numerous. Entry errors
    scatter; a systematic misreading does not.

Only when none fire are the entries the likely fault, and only then is a
per-entry repair proposed.

Findings: GUIDED-004 (the tier), T0-BUILD-003 (the substring match this
contains).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.card_evidence import (READING_ENTRIES, READING_IDENTITY,
                              READING_UNCLEAR, READING_UNITS,
                              WHOLE_COLUMN_SUSPECT_SHARE, derived_from,
                              interpretation_verdict, known_scale_factors,
                              plausibility_report, rescues_the_column)


def verdict_for(column, var_key, values, low, high, reference=None):
    s = pd.Series(values, dtype=float)
    flagged = s[(s < low) | (s > high)]
    return interpretation_verdict(column, var_key, s, low, high, flagged,
                                  reference=reference)


# ── the entries really are the fault ─────────────────────────────────────

def test_two_impossible_pressures_among_eighty_are_entry_errors():
    rng = np.random.default_rng(3)
    values = np.concatenate([rng.normal(78, 8, 78), [1.5e-15, 301.0]])
    v = verdict_for("bp_di", "bp_di", values, 15.0, 220.0, reference=(50, 120))
    assert v.reading == READING_ENTRIES
    assert v.suspect is False
    assert v.statement is None


def test_a_repair_is_still_proposed_for_scattered_errors():
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"bp_di": np.concatenate([rng.normal(78, 8, 78),
                                                [1.5e-15, 301.0]])})
    rep = plausibility_report(df)
    block = next(b for b in rep["impossible"] if b["column"] == "bp_di")
    assert block["reading"] == READING_ENTRIES
    assert block["whole_column_suspect"] is False
    assert rep["n_impossible"] == 2, (
        "the two entry errors stopped earning a repair proposal — the guard is "
        "now suppressing the case it exists to allow")


def test_the_magnitude_of_the_consequence_is_not_a_signal():
    """A large column of scattered errors still gets a repair proposal.

    Ten thousand rows, a hundred of them impossible and on both sides. A
    hundred deletions is a big consequence and no evidence at all that the
    reading is wrong, so the verdict must not move.
    """
    rng = np.random.default_rng(9)
    good = rng.normal(78, 8, 9900)
    bad = np.concatenate([np.full(50, 1e-12), np.full(50, 400.0)])
    v = verdict_for("bp_di", "bp_di", np.concatenate([good, bad]),
                    15.0, 220.0, reference=(50, 120))
    assert v.reading == READING_ENTRIES


# ── the unit is the fault, and the predicate can name it ─────────────────

def test_glucose_in_the_wrong_unit_is_read_as_a_unit_problem():
    rng = np.random.default_rng(4)
    v = verdict_for("glucose", "glucose", rng.normal(5.4, 0.6, 80),
                    10.0, 2000.0, reference=(70, 200))
    assert v.reading == READING_UNITS
    assert v.factor == pytest.approx(18.0)
    assert any(e.startswith("rescued-by:") for e in v.evidence)
    assert "18" in v.statement and "unit" in v.statement
    assert "error" in v.statement, (
        "the correction does not say that nothing here is proposed as an error")


def test_the_rescue_check_uses_units_the_variable_actually_has():
    factors = known_scale_factors("glucose")
    assert 18.0 in factors, "the mg/dL <-> mmol/L factor is not among the candidates"
    assert 10.0 in factors, "powers of ten are always candidates"

    values = pd.Series(np.full(50, 5.4))
    assert rescues_the_column(values, (70, 200), factors) == pytest.approx(18.0)


def test_a_column_already_in_range_is_never_rescued():
    """The rescue check must not fire on a column that is where it belongs."""
    values = pd.Series(np.full(50, 95.0))
    assert rescues_the_column(values, (70, 200), known_scale_factors("glucose")) is None


def test_no_known_unit_rescues_a_genuinely_broken_column():
    values = pd.Series([1e-14] * 40 + [5e5] * 40)
    assert rescues_the_column(values, (70, 200),
                              known_scale_factors("glucose")) is None


# ── the variable is the fault ────────────────────────────────────────────

@pytest.mark.parametrize("column, var_key, marker", [
    ("hba1c_proxy", "hba1c", "proxy"),
    ("bp_sys_delta", "bp_sys", "delta"),
    ("weight_change", "weight", "change"),
    ("height_percentile", "height", "percentile"),
    ("glucose_flag", "glucose", "flag"),
    ("bmi_zscore", "bmi", "zscore"),
])
def test_a_derived_column_is_not_the_variable(column, var_key, marker):
    assert derived_from(column, var_key) == marker


@pytest.mark.parametrize("column, var_key", [
    ("glucose", "glucose"),
    ("bp_di", "bp_di"),
    ("serum_glucose", "glucose"),
    ("glucose_mgdl", "glucose"),
])
def test_a_plain_or_qualified_name_is_not_treated_as_derived(column, var_key):
    """Only a *closed* list of modifiers counts. Guessing that an unfamiliar
    segment means 'derived' would suppress real findings on odd names."""
    assert derived_from(column, var_key) is None


def test_a_derived_name_alone_settles_the_reading():
    """Even a handful of violations on a derived column is the wrong question.

    Three impossible values out of eighty is well under the share threshold, so
    only the name evidence fires — and it is enough, because the finding is
    that the intervals do not describe this column at all.
    """
    rng = np.random.default_rng(5)
    values = np.concatenate([rng.normal(8.0, 0.4, 77), [0.1, 0.2, 0.3]])
    v = verdict_for("hba1c_proxy", "hba1c", values, 2.0, 30.0, reference=(4, 15))
    assert v.reading == READING_IDENTITY
    assert v.evidence == ["derived-name:proxy"]
    assert "proxy" in v.statement


# ── neither can be named, and the predicate says so ──────────────────────

def test_a_wholly_out_of_band_column_with_no_other_evidence_is_unclear():
    values = np.full(80, 1e-9)
    v = verdict_for("bp_di", "bp_di", values, 15.0, 220.0, reference=(50, 120))
    assert v.reading == READING_UNCLEAR
    assert "probably not in the units we assumed" in v.statement
    assert any(e.startswith("share:") for e in v.evidence)


def test_the_share_threshold_is_a_fallback_not_the_rule():
    """Just under the threshold, with no interpretation evidence, stays entries."""
    rng = np.random.default_rng(6)
    n = 200
    n_bad = int(n * WHOLE_COLUMN_SUSPECT_SHARE) - 5
    values = np.concatenate([rng.normal(78, 6, n - n_bad),
                             rng.uniform(1e-12, 1e-9, n_bad // 2),
                             np.full(n_bad - n_bad // 2, 400.0)])
    v = verdict_for("bp_di", "bp_di", values, 15.0, 220.0, reference=(50, 120))
    assert v.reading == READING_ENTRIES


# ── what the verdict licenses ────────────────────────────────────────────

def test_only_the_entries_reading_licenses_a_repair():
    for reading in (READING_UNITS, READING_IDENTITY, READING_UNCLEAR):
        from ml.card_evidence import InterpretationVerdict
        assert InterpretationVerdict(reading).suspect is True
    from ml.card_evidence import InterpretationVerdict
    assert InterpretationVerdict(READING_ENTRIES).suspect is False


def test_the_advisory_tier_never_doubts_its_own_reading():
    """Improbable is advisory and proposes nothing, so it needs no verdict.

    Running the predicate there would put a 'check the unit' correction beside
    a claim that never asserted anything strong enough to be wrong about.
    """
    rng = np.random.default_rng(7)
    df = pd.DataFrame({"hba1c_proxy": rng.normal(0.4, 0.1, 80)})
    rep = plausibility_report(df)
    for block in rep["improbable"]:
        assert block["reading"] == READING_ENTRIES
        assert block["reading_statement"] is None


def test_the_reported_count_excludes_every_doubted_column():
    rng = np.random.default_rng(8)
    df = pd.DataFrame({
        "hba1c_proxy": rng.normal(0.4, 0.1, 80),
        "glucose": rng.normal(5.4, 0.6, 80),
        "bp_di": np.concatenate([rng.normal(78, 8, 78), [1.5e-15, 301.0]]),
    })
    rep = plausibility_report(df)
    assert rep["n_suspect_columns"] == 2
    assert rep["n_impossible"] == 2, (
        "the two real entry errors are the only entries that earn a repair")
    readings = {b["column"]: b["reading"] for b in rep["impossible"]}
    assert readings == {"hba1c_proxy": READING_IDENTITY,
                        "glucose": READING_UNITS,
                        "bp_di": READING_ENTRIES}

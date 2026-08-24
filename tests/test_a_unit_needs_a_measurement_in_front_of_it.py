"""`IMPORT-267` — a column of education levels reported as mixing units.

**The governing rule's own failure at the highest severity the app has.**
`check_numeric_stored_as_text` reached its mixed-units branch on a column
containing no numbers, and `_KNOWN_UNITS` holds the bare letters `g`, `m`, `l`
and `s` with no requirement of a preceding digit. So `{'High school', 'Some
college', 'Bachelors', 'Graduate'}` yielded `{'l', 's'}` — the `l` of *school*
and the `s` of *Bachelors* — and the app asserted, in a card headed CRITICAL,
that a column of education levels mixes measurement units.

Not a hypothetical: `turbotab/sample_data/survey_instrument.csv` reproduced it,
and it was the only critical that file produced.

`ml/import_doctor.py` is frozen, and `TRANSITION_PLAN.md` §05 is the one
statement of what that permits: *"the freeze is lifted for REPAIR. Fixes to
dispositioned `IMPORT-*` findings may land, with a named regression test each,
as any other fix would."* This is that test.

**The repair has to be narrow in both directions**, which is why half this file
is about what must still fire. A gate that killed the false alarm by killing the
check would trade a critical that is wrong for a critical that is missing, and
`mg/dL` beside `mmol/L` in one column is two different measurements — coercing
them onto one scale is the defect `IMPORT-129` was filed for.
"""
from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.import_doctor import _units_present, diagnose        # noqa: E402


# ── the false alarm, gone ────────────────────────────────────────────────────

@pytest.mark.parametrize("values", [
    ["High school", "Some college", "Bachelors", "Graduate"],
    ["Nursing", "Bachelors"],                       # 'g' and 's'
    ["Farm", "Berlin"],                             # 'm' and 'in'
    ["Fasting", "Random"],                          # 'g' and 'm'
    ["Control", "Treatment arm"],                   # 'l' and 'm'
])
def test_a_category_with_no_number_in_it_reports_no_units(values):
    """The class, not just the instance. Four spellings of the same defect."""
    assert _units_present(pd.Series(values)) == set()


def test_a_column_of_education_levels_is_not_reported_as_mixing_units():
    """End to end through `diagnose`, because `_units_present` returning empty
    is only half the claim — the other half is that no finding is emitted."""
    df = pd.DataFrame({
        "education": ["High school", "Some college", "Bachelors", "Graduate"] * 4,
        "age": [31, 44, 52, 67] * 4,
        "outcome": [0, 1, 0, 1] * 4,
    })
    assert not [f for f in diagnose(df) if f.id.startswith("mixed_units__")]


def test_the_survey_fixture_no_longer_produces_a_critical():
    """The fixture that reproduced it, and the reason it was left reproducing
    it rather than designed around: a fixture that avoids the app's live
    defects tests an app that does not exist."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(os.path.join(here, "turbotab", "sample_data",
                                  "survey_instrument.csv"))
    criticals = [f.id for f in diagnose(df) if f.severity == "critical"]
    assert criticals == [], criticals


# ── and the true alarm, intact ───────────────────────────────────────────────

def test_a_column_that_genuinely_mixes_units_still_says_so():
    """`IMPORT-129`'s defect is the opposite one and must stay closed.

    `mg/dL` beside `mmol/L` in one column is two different measurements, and
    coercing them onto one scale produces numbers no model or statistic can
    interpret. A repair that silenced this would trade a critical that is wrong
    for a critical that is missing.
    """
    assert _units_present(pd.Series(["5.5 mg/dL", "7 mmol/L", "6.1 mg/dL"])) == \
        {"mg/dl", "mmol/l"}

    df = pd.DataFrame({
        "glucose": ["5.5 mg/dL", "7 mmol/L", "6.1 mg/dL", "5.9 mmol/L"] * 4,
        "age": [31, 44, 52, 67] * 4,
        "outcome": [0, 1, 0, 1] * 4,
    })
    found = [f for f in diagnose(df) if f.id == "mixed_units__glucose"]
    assert found, "the genuine mixed-unit reading was lost with the false one"
    assert found[0].severity == "critical"
    assert set(found[0].params["units"]) == {"mg/dl", "mmol/l"}


@pytest.mark.parametrize("values,expected", [
    (["107 kg", "98 kg"], {"kg"}),
    (["107 kg", "230 lb"], {"kg", "lb"}),
    (["12mo", "18 months"], {"mo", "months"}),
    (["120 mmHg", "80mmHg"], {"mmhg"}),
])
def test_a_measurement_keeps_its_unit(values, expected):
    """Every form the digit requirement has to survive: a space before the
    unit, none, a decimal, a comma, and a unit that is itself a word."""
    assert _units_present(pd.Series(values)) == expected


def test_the_control_fixture_reads_exactly_as_it_did():
    """`clinic_visits.csv` carries `weight` as `'107 kg'` and `income` as
    `'110,000'`, so it exercises both branches — a unit that must be found and a
    number that must not acquire one."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(os.path.join(here, "turbotab", "sample_data",
                                  "clinic_visits.csv"))
    ids = {f.id for f in diagnose(df)}
    assert "numeric_as_text__weight" in ids
    assert "numeric_as_text__income" in ids
    assert not [i for i in ids if i.startswith("mixed_units__")]

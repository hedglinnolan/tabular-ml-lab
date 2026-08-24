"""A column must not inherit a unit conversion from a variable it resembles.

`ml/clinical_units.infer_unit` picked its variable with `if var_name in
col_lower`. Unlike the same substring match in `ml/physiology_reference.py`,
which only flagged values falsely, this one **converts them**:
`ml/pipeline.py:60` feeds the returned `conversion_factor` into the positional
list `UnitHarmonizer` multiplies by inside the fitted pipeline.

So `weight_change` matched `weight` and came back as kilograms at **high**
confidence. A column of kilogram differences, already in the canonical unit,
was eligible to be rescaled by 0.453592 the moment its values happened to sit in
the pounds hypothesis range — a wrong number, with no error, in the data that
gets modeled and published.

The rule is ruling 4's, applied where it bites hardest: **exact key or declared
alias, nothing else.** An unrecognized name gets no unit and no conversion.
Silence is a gap; an inherited conversion is a corrupted column.

Finding: T0-BUILD-004.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.clinical_units import CLINICAL_VARIABLES, infer_unit, match_clinical_variable
from ml.physiology_reference import load_nhanes_reference


def values(n=60, loc=70.0, seed=1):
    return pd.Series(np.random.default_rng(seed).normal(loc, 5, n))


# ── the defect ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("column", [
    "weight_change", "weight_delta", "weight_pct",
    "hba1c_v2", "hba1c_imputed", "hba1c_lab2", "hba1c_proxy",
    "glucose_flag", "glucose_missing", "bmi_zscore",
    "bp_sys_delta", "height_percentile", "waist_ratio",
])
def test_a_resembling_name_earns_no_conversion(column):
    """The names a denylist of modifier words could never have covered."""
    assert match_clinical_variable(column) is None
    info = infer_unit(column, values())
    assert info["conversion_factor"] is None, (
        f"{column} inherited a conversion factor from a variable it merely "
        "resembles; ml/pipeline.py would multiply the column by it")
    assert info["inferred_unit"] is None
    assert info["confidence"] is None


def test_the_pipeline_gets_a_neutral_factor_for_an_unrecognized_column():
    """What the fix actually buys, at the site that applies it."""
    from ml.pipeline import build_unit_harmonization_config

    df = pd.DataFrame({"weight": values(loc=70.0), "weight_change": values(loc=70.0)})
    config = build_unit_harmonization_config(df, ["weight", "weight_change"])
    factors = dict(zip(["weight", "weight_change"], config["conversion_factors"]))

    assert factors["weight_change"] == 1.0, (
        "a derived column is still being rescaled inside the pipeline")
    assert config["inferred_units"]["weight_change"] is None


def test_a_pounds_column_named_for_a_variable_is_still_converted():
    """Silence must not become the answer to everything.

    The capability this protects: a genuine `weight` column in pounds still
    earns its 0.453592.

    The values are deliberately above 200: the kilogram hypothesis spans
    (30, 200), so anything lighter fits both readings and the engine correctly
    prefers the canonical unit. A test that used 150-200 would be asserting
    against a genuine ambiguity rather than against this fix.
    """
    info = infer_unit("weight", pd.Series([210.0, 250.0, 300.0, 350.0, 400.0] * 12))
    assert info["conversion_factor"] == pytest.approx(0.453592)
    assert info["inferred_unit"] == "lb"


# ── aliases are how a real column earns its unit ─────────────────────────

@pytest.mark.parametrize("column, expected", [
    ("weight", "weight"), ("wt", "weight"), ("BMXWT", "weight"),
    ("body_weight", "weight"),
    ("hba1c", "hba1c"), ("a1c", "hba1c"), ("HbA1c", "hba1c"),
    ("glucose", "glucose"), ("serum_glucose", "glucose"), ("LBXGLU", "glucose"),
    ("bp_di", "bp_di"), ("diastolic", "bp_di"), ("dbp", "bp_di"),
    ("kcal", "kcal"), ("energy", "kcal"),
])
def test_an_exact_key_or_declared_alias_still_matches(column, expected):
    assert match_clinical_variable(column) == expected


def test_every_alias_is_unique_to_one_variable():
    owner = {}
    for key, config in CLINICAL_VARIABLES.items():
        for name in [key] + list(config.get("aliases") or []):
            assert name not in owner, (
                f"{name!r} is claimed by both {owner.get(name)} and {key}")
            owner[name] = key


# ── the two registries must not drift ────────────────────────────────────

def test_the_two_name_registries_agree():
    """`CLINICAL_VARIABLES` and the NHANES reference name the same variables.

    Two tables for eleven variables will drift, and drift here means a column
    that earns physiologic bounds but no unit conversion, or the reverse — the
    plausibility check silently comparing raw values against converted bounds.
    """
    units = CLINICAL_VARIABLES
    physio = load_nhanes_reference()["variables"]
    assert set(units) == set(physio), (
        "the unit table and the physiology reference name different variables: "
        f"units only {sorted(set(units) - set(physio))}, "
        f"physiology only {sorted(set(physio) - set(units))}")

    for key in units:
        a = set(units[key].get("aliases") or [])
        b = set(physio[key].get("aliases") or [])
        assert a == b, (
            f"{key}: the two registries publish different aliases — "
            f"units only {sorted(a - b)}, physiology only {sorted(b - a)}")

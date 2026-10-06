"""The product designer's NHANES drive, as tests.

Eight findings came out of one hour with the Guided door on a real NHANES
extract (GUIDED-001 … GUIDED-008), plus the §09 reskin of the built steps. Each
is pinned here against the thing that was actually wrong, so the fix cannot
regress into the shape the drive found.

The frontend assertions read `web/index.html` rather than driving a browser.
That is a real limit and it is the honest one: it can prove the treatment is
present and exclusive, and it cannot prove it renders. What can be checked
end-to-end goes over HTTP against the real engine.

Run:  venv/bin/python -m pytest turbotab/test_guided_drive.py -v
"""
# The tests here that drove the retired legacy app (turbotab/api.py, the page, the
# figure and manuscript modules, the gates) were removed with it, BLUEPRINT §9.1; git keeps them.
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from turbotab import engine

REPO_ROOT = Path(__file__).resolve().parent.parent


def nhanes_like(n: int = 160) -> pd.DataFrame:
    """A table with each of the drive's artifacts in it."""
    rng = np.random.default_rng(4)
    df = pd.DataFrame({
        "seqn": np.arange(1, n + 1),
        "age": rng.integers(20, 80, n).astype(float),
        "bmi": np.round(rng.normal(28, 5, n), 1),
        "bp_di": np.round(rng.normal(78, 9, n), 1),
        "glucose": np.round(rng.normal(101, 18, n), 1),
        # True/False text with blanks — the GUIDED-001 column.
        "meds_chol": rng.choice(["True", "False", ""], n, p=[0.4, 0.35, 0.25]),
        "diabetes": rng.integers(0, 2, n),
    })
    df.loc[3, "bp_di"] = 1.5e-15        # not a patient: an entry error
    df.loc[11, "bp_di"] = 301.0
    df["leaky_score"] = df["diabetes"] * 5.0 + rng.normal(0, 0.02, n)
    df.loc[rng.choice(n, 60, replace=False), "bmi"] = np.nan
    return df


@pytest.fixture(scope="module")
def raw() -> bytes:
    return nhanes_like().to_csv(index=False).encode()


# ═══════════════════════════════════════════════════════════════════════════
# GUIDED-001 — the binary reading outranks numeric coercion
# ═══════════════════════════════════════════════════════════════════════════


def test_an_unknown_binary_pair_does_not_pretend_to_know_the_positive_level():
    from ml.binary_text import binary_text_finding
    s = pd.Series(["alpha", "beta", "alpha", "beta", "alpha", "beta", "alpha"])
    f = binary_text_finding("arm", s)
    assert f is not None
    assert f.confidence == "medium", "an arbitrary 0/1 assignment is not high confidence"
    assert "your call" in f.why_it_matters


def test_a_three_level_column_is_not_called_binary():
    from ml.binary_text import read_as_binary_plan
    s = pd.Series(["yes", "no", "unknown", "yes", "no", "unknown", "yes"])
    assert read_as_binary_plan(s) is None


# ═══════════════════════════════════════════════════════════════════════════
# GUIDED-002 — missingness names its features, routes by dtype, states timing
# ═══════════════════════════════════════════════════════════════════════════


def test_a_categorical_column_is_offered_an_explicit_missing_level():
    from ml.missingness_plan import missingness_cards
    df = pd.DataFrame({"site": ["a", "b", None, None, "c", None, "b", "a", None, "c"]})
    card = missingness_cards(df)[0]
    assert card["dtype_route"] == "categorical"
    keys = {o["key"] for o in card["options"]}
    # The card speaks the RECORD's vocabulary since `GUIDED-090`: one table
    # decides what both doors offer, so the option key IS the declaration key.
    assert "explicit_category" in keys and "impute_mode" in keys
    assert "leave" in keys, (
        "the categorical branch permits leaving the blanks and the card does "
        "not offer it — judgment renders as ranking, never as absence")


# ═══════════════════════════════════════════════════════════════════════════
# GUIDED-003 / GUIDED-005 — the evidence is on the table
# ═══════════════════════════════════════════════════════════════════════════


def test_the_gallery_and_the_matrix_are_gated_on_feature_count():
    from ml.card_evidence import (MAX_FEATURES_FOR_GALLERY, correlation_matrix,
                                  histogram_gallery)
    rng = np.random.default_rng(1)
    wide = pd.DataFrame(rng.normal(size=(60, MAX_FEATURES_FOR_GALLERY + 5)))
    wide.columns = [f"f{i}" for i in range(wide.shape[1])]

    gallery = histogram_gallery(wide)
    matrix = correlation_matrix(wide)
    assert gallery["available"] is False and matrix["available"] is False
    assert str(MAX_FEATURES_FOR_GALLERY) in gallery["reason"]
    assert gallery["plots"] == [] and matrix["matrix"] == []

    narrow = wide.iloc[:, :8]
    assert histogram_gallery(narrow)["available"] is True
    assert correlation_matrix(narrow)["available"] is True


# ═══════════════════════════════════════════════════════════════════════════
# GUIDED-004 — impossible is not a kind of outlier
# ═══════════════════════════════════════════════════════════════════════════

def test_the_impossibility_band_contains_the_improbability_band():
    """The tiers must nest. An impossibility band inside the improbability one
    would call ordinary values impossible and propose deleting them.

    **`MISC-018` renamed this test with what it checks.** It was
    `..._contains_the_reference_interval`, and the p01/p99 pair it compares
    against is not a reference interval — the central 98%, where CLSI EP28-A3c
    defines the interval as the central 95%.
    """
    from ml.physiology_reference import (impossibility_contains_improbability,
                                         load_nhanes_reference)
    ref = load_nhanes_reference()
    for key in ref["variables"]:
        assert impossibility_contains_improbability(ref, key), (
            f"{key}'s impossibility band is narrower than its improbability band")


def test_a_variable_with_no_published_band_returns_none_not_the_interval():
    from ml.physiology_reference import get_impossibility_band
    ref = {"variables": {"widget": {"unit": "u", "p01": 1, "p99": 9}}}
    assert get_impossibility_band(ref, "widget") is None, (
        "falling back to the reference interval would promote improbable values "
        "to impossible ones and propose deleting real data")


def test_a_column_that_is_mostly_impossible_is_read_as_a_unit_problem():
    """The predicate escalates on evidence, not on how much a repair would cost.

    A glucose column recorded in mmol/L reads as entirely impossible against
    mg/dL bounds. It is not: multiplying by 18 — a unit glucose is actually
    recorded in — puts the whole column inside its reference interval, so the
    reading is wrong and no entry is.

    (`hba1c_proxy`, which used to be this test's fixture, no longer reaches the
    predicate at all: `match_variable_key` matches exact keys and declared
    aliases only, so a column merely named like a variable now gets no bounds.
    That is asserted in
    `tests/test_doubt_the_reading_not_the_data.py::test_an_unknown_suffix_yields_silence_rather_than_inherited_bounds`.)
    """
    from ml.card_evidence import READING_UNITS, plausibility_report
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"glucose": rng.normal(5.4, 0.6, 80)})
    rep = plausibility_report(df)
    block = next(b for b in rep["impossible"] if b["column"] == "glucose")

    assert block["reading"] == READING_UNITS
    assert block["whole_column_suspect"] is True
    assert any(e.startswith("rescued-by:") for e in block["reading_evidence"])
    assert rep["n_impossible"] == 0, (
        "a suspect column inflated the count of entries that earn a repair")
    assert block["entries"], "the values are still shown; silence is the other lie"


# ═══════════════════════════════════════════════════════════════════════════
# The preview's count and its highlighting must mean the same thing
# ═══════════════════════════════════════════════════════════════════════════

def test_the_change_count_and_the_highlighted_cells_agree():
    """Found while building the binary reading, described by no finding.

    The count came from a value comparison and the highlighting from a text
    comparison, so `"1200"` -> `1200` reported eight changed cells over an
    unmarked table, and `True` -> `1` reported none over a marked one.
    """
    from ml import import_doctor

    df = pd.DataFrame({"id": range(8),
                       "dose": ["1200", "2400", "3100", "4050",
                                "5900", "6300", "7700", "8100"]})
    coerce = next(f for f in import_doctor.diagnose(df)
                  if f.fix_kind == "coerce_numeric")
    pv = engine.preview_fix(df, coerce)
    marked = sum(1 for r in pv["sample"]["rows"] for c in r["changed"] if c)
    assert pv["changed_cells"] == 0 and marked == 0
    assert any(s["key"].startswith("dtype of") for s in pv["stats"]), (
        "a type-only change must still be reported, in the statistics")

    df2 = pd.DataFrame({"id": range(8),
                        "meds": [True, False, True, False, True, True, False, True]})
    binary = next(f for f in engine.diagnose(df2) if f.fix_kind == "read_as_binary")
    pv2 = engine.preview_fix(df2, binary)
    marked2 = sum(1 for r in pv2["sample"]["rows"] for c in r["changed"] if c)
    assert pv2["changed_cells"] == 8 and marked2 == 8

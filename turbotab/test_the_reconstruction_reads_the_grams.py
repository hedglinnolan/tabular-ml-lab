"""`GUIDED-068` — the matcher that took the first hit.

`_match` compiled one regex per role and returned the first numeric column that
matched. `dietary_recalls.csv` carries `protein_pct_kcal` **and** `protein_g`,
in that order, so the Atwater reconstruction read the percentages and the gram
columns beside them were never seen. Not false about the columns it named — they
are percentages — and misleading about the table, because the finding warned
that every downstream step would be computing on the wrong quantity while the
right quantity sat one column over.

`NUTRITION_PACK.md` §01 opens with *match on three signals jointly, never names
alone* and lists the three: name patterns, the NHANES schema, and **unit
suffixes** — `_g`, `_mg`, `_mcg`, `_ug`, `_iu`, `_kcal`, `_kj`, `_per1000kcal`,
`_pct_energy`. The table already says which column is which; nothing was reading
it.

## Three signals, and the third one is the arithmetic

1. **The name** finds the candidates.
2. **The unit suffix** ranks them, and groups them into families — a set per
   unit rather than a cartesian product, because the defect's shape is one
   nutrient carried twice, and the question is which family to reconstruct from
   and never whether to mix them.
3. **The reconstruction itself** chooses between the families, because passing
   IS the definition of *these are the grams*. §01 says so directly: the Atwater
   check is a second signal, not just a verdict.

That is not hunting for a verdict the app likes. There is a fact — which columns
are grams — and the check identifies it. The alternative is reading whichever
column the file happened to list first, which is what this repairs.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbotab import nutrition as N                                   # noqa: E402
from turbotab import packs as P                                       # noqa: E402

DATA = Path(__file__).resolve().parent / "sample_data"


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / f"{name}.csv")


# ── the gate ────────────────────────────────────────────────────────────────

def test_a_table_carrying_grams_and_percentages_reconstructs_from_the_grams():
    """The defect, as its own assertion. `dietary_recalls.csv` has both."""
    df = load("dietary_recalls")
    reading = N.atwater(df)
    assert reading is not None
    assert reading.unit_class == N.GRAMS
    assert set(reading.macro_columns.values()) == {
        "protein_g", "carbohydrate_g", "fat_g"}
    assert reading.verdict == "pass", reading.verdict
    assert N.PASS_LOW <= reading.ratio <= N.PASS_HIGH
    # And the finding disappears, because there is nothing wrong with the table.
    assert N.atwater_finding(df) is None, (
        "a table whose gram columns reconstruct cleanly still produced a "
        "finding about its percentage columns")


def test_the_percentages_are_still_recognized_when_they_are_the_only_ones():
    """The other half of the gate. Ranking grams first must not make the
    `macros_not_grams` verdict unreachable — a table that carries ONLY
    percent-of-energy columns is exactly the case §01's ratio table names."""
    df = load("dietary_recalls").drop(
        columns=["protein_g", "carbohydrate_g", "fat_g", "fiber_g"])
    reading = N.atwater(df)
    assert reading.unit_class == N.DENSITY
    assert reading.verdict == "macros_not_grams", reading.verdict
    finding = N.atwater_finding(df)
    assert finding["title"] == "The macronutrient columns look like percentages"


def test_the_columns_it_did_not_use_travel_with_the_finding():
    """*Misleading about the table* was the defect, so the reading has to be
    able to say what else the table carries. A unit error in the gram columns
    of a table that also has percentages names both."""
    df = load("dietary_recalls").copy()
    df["energy_kcal"] = df["energy_kcal"] * N.KCAL_PER_KJ
    finding = N.atwater_finding(df)
    assert finding["params"]["unit_class"] == N.GRAMS
    assert set(finding["params"]["set_aside"]) == {
        "protein_pct_kcal", "carbohydrate_pct_kcal", "fat_pct_kcal",
        "alcohol_pct_kcal"}
    assert "same nutrients in another unit" in finding["detail"]
    assert "protein_g" in finding["detail"]


# ── signal 3 · the unit suffixes §01 names ─────────────────────────────────

@pytest.mark.parametrize("column,expected", [
    ("protein_g", N.GRAMS),
    ("carbohydrate_grams", N.GRAMS),
    ("protein_pct_kcal", N.DENSITY),
    ("fat_pct_energy", N.DENSITY),
    ("calcium_per1000kcal", N.DENSITY),
    ("sodium_mg", N.OTHER_UNIT),
    ("vitamin_d_iu", N.OTHER_UNIT),
    ("energy_kcal", N.OTHER_UNIT),
    ("DR1TPROT", N.BARE),
    ("protein", N.BARE),
])
def test_the_unit_suffix_is_read_from_the_name(column, expected):
    assert N._unit_class(column) == expected


def test_a_macronutrient_share_is_not_the_energy_column():
    """A second collision the first-match order was hiding.

    `protein_pct_kcal` matches the energy name pattern (`kcal`) as well as the
    protein one, and it only escaped notice because `energy_kcal` happened to
    come first in the file. Ranking by unit suffix without an exclusion
    promoted the protein share to the energy column, which would have made the
    reconstruction a ratio between two quantities that are both protein.
    """
    df = load("dietary_recalls")
    assert N._energy_column(df) == "energy_kcal"
    # Even with the energy column removed, a macro share must not stand in.
    without = df.drop(columns=["energy_kcal"])
    assert N._energy_column(without) is None
    assert N.atwater(without) is None


def test_the_families_are_sets_rather_than_a_cartesian_product():
    """One nutrient carried twice is the defect's shape. Mixing `protein_g`
    with `fat_pct_kcal` would be a reconstruction of nothing."""
    families = dict(N._macro_sets(load("dietary_recalls")))
    assert set(families) == {N.GRAMS, N.DENSITY}
    for unit, columns in families.items():
        suffixes = {N._unit_class(c) for c in columns.values()}
        assert suffixes == {unit}, (unit, columns)


# ── the arithmetic is what chooses ─────────────────────────────────────────

def _two_family_frame(n: int = 200, energy_factor: float = 1.0,
                      seed: int = 1) -> pd.DataFrame:
    """Grams and percent-of-energy for the same four macronutrients."""
    rng = np.random.default_rng(seed)
    protein = rng.gamma(9, 9, n)
    carb = rng.gamma(9, 28, n)
    fat = rng.gamma(7, 11, n)
    alcohol = rng.gamma(1, 4, n)
    energy = 4 * protein + 4 * carb + 9 * fat + 7 * alcohol
    return pd.DataFrame({
        # THE DENSITY COLUMNS FIRST, which is the order that produced the bug.
        "protein_pct_kcal": 4 * protein / energy * 100,
        "carbohydrate_pct_kcal": 4 * carb / energy * 100,
        "fat_pct_kcal": 9 * fat / energy * 100,
        "alcohol_pct_kcal": 7 * alcohol / energy * 100,
        "energy_kcal": energy * energy_factor,
        "protein_g": protein,
        "carbohydrate_g": carb,
        "fat_g": fat,
        "alcohol_g": alcohol,
    })


def test_the_gram_family_wins_because_it_passes_not_because_it_is_first():
    reading = N.atwater(_two_family_frame())
    assert reading.verdict == "pass" and reading.unit_class == N.GRAMS


def test_a_unit_error_in_the_grams_is_still_reported_rather_than_routed_around():
    """The failure mode of *try families until one passes*: if no family
    passes, the best-ranked one's reading stands. The app does not go looking
    for a set of columns that makes the problem disappear."""
    reading = N.atwater(_two_family_frame(energy_factor=N.KCAL_PER_KJ))
    assert reading.unit_class == N.GRAMS
    assert reading.verdict == "energy_in_kj", reading.verdict
    assert abs(reading.ratio - N.KCAL_PER_KJ) < 0.05


def test_a_mixed_unit_merge_in_the_grams_is_not_rescued_by_the_percentages():
    """The `critical` verdict is the one it would be worst to lose to a
    family swap, because a multi-source merge in the grams is still a
    multi-source merge whatever else the table carries."""
    frame = _two_family_frame(n=400)
    factor = np.where(np.arange(len(frame)) < len(frame) // 2, 1.0,
                      N.KCAL_PER_KJ)
    frame["energy_kcal"] = frame["energy_kcal"] * factor
    reading = N.atwater(frame)
    assert reading.verdict == "mixed_units", reading.verdict
    assert reading.unit_class == N.GRAMS
    finding = N.atwater_finding(frame)
    assert finding["severity"] == "critical"
    assert finding["fix_kind"] == "none"


# ── the promise the fix made unkeepable, and the fixture that keeps it ─────

def test_the_atwater_promise_is_kept_by_a_fixture_with_a_real_unit_error():
    """Fixing the matcher made `dietary_recalls.csv` PASS, which is correct and
    left `pack::dietary::atwater` promised by the dietary pack's hover and
    emitted by no fixture. A promise nobody keeps is the app announcing it will
    look for something it will not — the worse half of the key match."""
    ids = {f["id"] for f in P.findings(load("nhanes_kilojoules"), [P.DIETARY])}
    assert "pack::dietary::atwater" in ids
    finding = N.atwater_finding(load("nhanes_kilojoules"))
    assert finding["params"]["verdict"] == "energy_in_kj"
    assert abs(finding["params"]["ratio"] - N.KCAL_PER_KJ) < 0.05
    assert "kilojoule" in finding["detail"]


def test_the_new_fixture_raises_nothing_else_it_should_not():
    """A fixture built to trigger one detector must not quietly trigger
    others — that is how a fixture stops being evidence about the thing it was
    built for."""
    df = load("nhanes_kilojoules")
    assert N.partial_design_finding(df) is None
    assert N.lonely_psu_finding(df) is None
    assert N.survey_weights_finding(df) is not None

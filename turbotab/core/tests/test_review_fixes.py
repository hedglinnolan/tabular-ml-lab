"""Tier A: the statistics the M1 review found wrong, checked against known answers.

* A partition never counts a total and its parts twice: the validator refuses it before it is
  recorded (so the design never fails on it afterwards) and offers the totals-only partition, and
  a partition the fit's own unit check would refuse is refused the same way.
* Sugar is carbohydrate at 4 kcal/g: it carries energy and is adjusted with the other nutrients.
* Energy shares that sum to 100% beside an intercept: the linear family says the design is
  nearly singular, the roles propose leaving one share out, and the design warns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from turbotab.core.datastore import DataStore, ingest
from turbotab.core.decisions import ProjectState, Refusal, parse_decision, validate
from turbotab.core.methods.energy import (
    EnergyAdjuster,
    EnergyAdjustmentNotApplicable,
    energy_factor,
    partition_refusal,
)
from turbotab.core.methods.nesting import compositions, nested_components, reference_share
from turbotab.core.models.linear import collinearity_concern

SAMPLES = pytest.importorskip("pathlib").Path(__file__).resolve().parents[3] / "turbotab" / "sample_data"


def nested_diet(n: int = 600, seed: int = 0, grams: bool = True) -> pd.DataFrame:
    """Energy that is exactly 4P + 4C + 9F + 7A, with fat split into three parts inside its total."""
    rng = np.random.default_rng(seed)
    protein = rng.gamma(9.0, 9.0, n)
    carb = rng.gamma(9.0, 27.0, n)
    sugar = carb * rng.uniform(0.2, 0.5, n)
    fat = rng.gamma(9.0, 8.0, n)
    share = rng.dirichlet([4, 4, 2], n) * 0.95  # the parts fill 95% of the total
    alcohol = rng.gamma(0.4, 6.0, n)
    kcal = 4 * protein + 4 * carb + 9 * fat + 7 * alcohol
    sfx = "_g" if grams else ""
    return pd.DataFrame({
        "kcal": kcal, f"protein{sfx}": protein, f"sugar{sfx}": sugar, f"carb{sfx}": carb,
        f"fat_total{sfx}": fat, f"fat_sat{sfx}": fat * share[:, 0], f"fat_mon{sfx}": fat * share[:, 1],
        f"fat_poly{sfx}": fat * share[:, 2], "y": rng.normal(size=n),
    })


# ── the partition ────────────────────────────────────────────────────────────


def test_a_total_and_its_parts_double_count_and_the_totals_alone_do_not():
    df = nested_diet()
    nutrients = ["protein_g", "sugar_g", "carb_g", "fat_total_g", "fat_sat_g", "fat_mon_g", "fat_poly_g"]
    nested = nested_components(df, nutrients)
    assert nested == {"sugar_g": "carb_g", "fat_sat_g": "fat_total_g", "fat_mon_g": "fat_total_g",
                      "fat_poly_g": "fat_total_g"}
    # The known answer: with the parts in, "everything else" is alcohol less the parts' kcal again.
    step = EnergyAdjuster("partition", "kcal", nutrients).fit(df)
    other = step.transform(df)["kcal_from_other"]
    parts_kcal = 4 * df["sugar_g"] + 9 * (df["fat_sat_g"] + df["fat_mon_g"] + df["fat_poly_g"])
    alcohol_kcal = df["kcal"] - 4 * df["protein_g"] - 4 * df["carb_g"] - 9 * df["fat_total_g"]
    np.testing.assert_allclose(other, alcohol_kcal - parts_kcal)
    assert (other < 0).mean() > 0.9  # counted twice, the remainder goes negative

    refused = partition_refusal(df, "kcal", nutrients, nested=nested)
    assert refused is not None
    assert refused["reason"].startswith("sugar_g, fat_sat_g, fat_mon_g and fat_poly_g are parts of "
                                        "carb_g and fat_total_g")
    assert refused["nutrients"] == ["protein_g", "carb_g", "fat_total_g"]
    # The way out partitions cleanly: its remainder is exactly the alcohol's kcal.
    totals = EnergyAdjuster("partition", "kcal", refused["nutrients"]).fit(df).transform(df)
    np.testing.assert_allclose(totals["kcal_from_other"], alcohol_kcal)
    assert partition_refusal(df, "kcal", refused["nutrients"], nested=nested) is None


def test_overlapping_columns_are_refused_even_without_a_name_that_says_so():
    """Nutrients that out-weigh energy by more than factor error, on many rows, overlap."""
    df = nested_diet()
    df["fat_again_g"] = df["fat_total_g"]  # a duplicate the names do not tie to its total
    refused = partition_refusal(df, "kcal", ["protein_g", "carb_g", "fat_total_g", "fat_again_g"])
    assert refused is not None and "carry more energy than kcal itself" in refused["reason"]
    # A few percent of factor error (food tables' specific Atwater factors) is not refused.
    df["kcal"] = df["kcal"] * 0.97
    assert partition_refusal(df, "kcal", ["protein_g", "carb_g", "fat_total_g"]) is None


class _Store:
    """The slice of DataStore a validator reads."""

    def __init__(self, frame: pd.DataFrame):
        self.frame = frame.reset_index(drop=True)
        self.columns = list(self.frame.columns)
        self.n_rows = len(self.frame)
        self.read: set[int] = set()

    def materialize(self, columns, row_ids=None):
        rows = self.frame.index if row_ids is None else np.asarray(row_ids)
        self.read.update(int(i) for i in rows)
        return self.frame.loc[rows, list(columns)]


def _context(df: pd.DataFrame, store: _Store, sealed=None):
    roles = {c: ("energy" if c == "kcal" else "exposure") for c in df.columns if c != "y"}
    info = {c: {"dtype": "numeric", "n_unique": 500} for c in df.columns}
    return {"columns": list(df.columns), "column_info": info, "target": "y", "task": "regression",
            "state": ProjectState(lens=["dietary"], target="y", roles=roles),
            "store": lambda: store, "sealed": (lambda: sealed) if sealed is not None else None}


@pytest.mark.parametrize("grams", [True, False])
def test_the_validator_refuses_the_partition_before_it_is_recorded(grams):
    """NHANES names its nutrients without a unit; with the _g suffix the fit would run and count
    the parts twice. Either way the answer is refused at the question, with the totals as exit."""
    df = nested_diet(grams=grams)
    sfx = "_g" if grams else ""
    nutrients = [f"{n}{sfx}" for n in ("protein", "sugar", "carb", "fat_total", "fat_sat",
                                         "fat_mon", "fat_poly")]
    sealed = np.arange(0, len(df), 5)
    store = _Store(df)
    decision = {"kind": "set_energy_adjustment", "method": "partition", "energy_column": "kcal",
                "nutrients": nutrients}
    with pytest.raises(Refusal) as info:
        validate(decision, _context(df, store, sealed))
    refusal = info.value
    assert refusal.code == "method_not_applicable"
    assert f"are parts of `carb{sfx}` and `fat_total{sfx}`" in str(refusal)
    exit_ = refusal.exits[0]
    assert exit_["decision"]["nutrients"] == [f"protein{sfx}", f"carb{sfx}", f"fat_total{sfx}"]
    assert exit_["label"] == f"Partition `protein{sfx}`, `carb{sfx}` and `fat_total{sfx}`"
    assert not store.read & set(sealed.tolist()), "the unit check read a held-out row"
    # The exit itself is accepted: the reconstruction confirms the unmarked columns are grams.
    validate(parse_decision(exit_["decision"]), _context(df, _Store(df), sealed))
    EnergyAdjuster("partition", "kcal", exit_["decision"]["nutrients"]).fit(df)


def test_a_partition_the_fit_would_refuse_is_refused_at_the_question():
    """No protein column: the reconstruction cannot vouch for unmarked grams, so the fit refuses;
    the validator now refuses first, with the fit's own reason."""
    df = nested_diet(grams=False).drop(columns=["protein", "sugar", "fat_sat", "fat_mon", "fat_poly"])
    with pytest.raises(EnergyAdjustmentNotApplicable):
        EnergyAdjuster("partition", "kcal", ["carb", "fat_total"]).fit(df)
    decision = {"kind": "set_energy_adjustment", "method": "partition", "energy_column": "kcal",
                "nutrients": ["carb", "fat_total"]}
    with pytest.raises(Refusal) as info:
        validate(decision, _context(df, _Store(df)))
    assert "could not confirm it is grams" in str(info.value)
    assert "atwater=" not in str(info.value)  # no machinery in the words


def test_sugar_is_carbohydrate_at_four_kcal_a_gram():
    for column in ("sugar", "sugar_g", "added_sugars_g", "starch_g"):
        reading = energy_factor(column)
        assert (reading.role, reading.factor) == ("carbohydrate", 4.0), column
    assert energy_factor("fiber_g").factor == 2.0  # fiber keeps its own factor


# ── shares that sum to 100% ──────────────────────────────────────────────────


def test_shares_that_sum_to_one_hundred_make_the_linear_design_nearly_singular():
    df = pd.read_csv(SAMPLES / "dietary_recalls.csv")
    shares = ["protein_pct_kcal", "fat_pct_kcal", "carbohydrate_pct_kcal", "alcohol_pct_kcal"]
    assert compositions(df, shares) == shares
    assert reference_share(df, shares) == "carbohydrate_pct_kcal"  # the largest on average
    matrix = df[["age", "bmi", *shares, "protein_g", "fat_g"]].astype(float)
    concern = collinearity_concern(matrix)
    assert concern is not None and concern.startswith("The model matrix is nearly singular")
    assert all(s in concern for s in shares), concern
    # Leave one share out (the reference): the rest are identified, and nothing is said.
    assert collinearity_concern(matrix.drop(columns="carbohydrate_pct_kcal")) is None
    # Units do not count: a well-posed design in wildly different units stays quiet.
    rng = np.random.default_rng(0)
    wide = pd.DataFrame({"a": rng.normal(0, 1e-3, 500), "b": rng.normal(5e6, 1e5, 500),
                         "c": rng.normal(size=500)})
    assert collinearity_concern(wide) is None


def test_the_roles_and_the_design_say_so_on_the_recall_fixture(tmp_path):
    from turbotab.core.models.pipeline import design_spec, warnings_for
    from turbotab.core.stages.rows import _composition_reference

    dest = tmp_path / "recalls.parquet"
    ingest(SAMPLES / "dietary_recalls.csv", dest)
    shares = ["protein_pct_kcal", "fat_pct_kcal", "carbohydrate_pct_kcal", "alcohol_pct_kcal"]
    with DataStore(dest, 2 << 30) as store:
        proposals = [{"column": c, "proposed": "exposure"} for c in [*shares, "protein_g"]]
        assert _composition_reference(store, proposals) == ("carbohydrate_pct_kcal", 4)
    df = pd.read_csv(SAMPLES / "dietary_recalls.csv")
    roles = {c: "exposure" for c in shares} | {"age": "covariate"}
    state = ProjectState(lens=["dietary"], target="hba1c", roles=roles)
    spec = design_spec(state, df, [*shares, "age"])
    said = warnings_for(spec, df, [], {})
    assert any(w.startswith(", ".join(shares) + " sum to 100% on every row") for w in said), said

"""The proposals stage (M1_CONTRACT §3): the pack's exclusion screens with their counts, and the
energy reading — offered, never pre-selected. Counts are Tier A: they reach a participant flow."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from turbotab.core.decisions import ExclusionRule, ProjectState, RangeByLevel
from turbotab.core.graph import Bundle
from turbotab.core.stages.proposals import proposals_stage, roles_from, rule_excludes
from turbotab.core.tests.stage_harness import NHANES, SAMPLES, Ingested
from turbotab.server.schemas import ProposalsArtifact

KEYS = ["willett_by_sex", "sex_neutral_500_5000", "sex_neutral_500_3500"]


def pandas_counts(df: pd.DataFrame, energy: str, sex: str, women: str, men: str,
                  base: pd.Series) -> dict[str, int]:
    """The three screens written out longhand, the way a reader would check them."""
    e = df[energy]
    willett = ((df[sex] == women) & ((e < 500) | (e > 3500))) | ((df[sex] == men) & ((e < 800) | (e > 4200)))
    return {
        "willett_by_sex": int((willett & base).sum()),
        "sex_neutral_500_5000": int((((e < 500) | (e > 5000)) & base).sum()),
        "sex_neutral_500_3500": int((((e < 500) | (e > 3500)) & base).sum()),
    }


@pytest.fixture(scope="module")
def nhanes(tmp_path_factory):
    if not NHANES.is_file():
        pytest.skip("the real NHANES export is not on this machine")
    table = Ingested(NHANES, tmp_path_factory.mktemp("nhanes"))
    artifact = table.run(proposals_stage, ProjectState(lens=["dietary", "clinical"], target="glucose"))
    return table, artifact


@pytest.fixture(scope="module")
def recalls(tmp_path_factory):
    table = Ingested(SAMPLES / "dietary_recalls.csv", tmp_path_factory.mktemp("recalls"))
    artifact = table.run(proposals_stage, ProjectState(lens=["dietary"], target="hba1c"))
    return table, artifact


def test_the_nhanes_screens_count_what_pandas_counts(nhanes):
    table, artifact = nhanes
    df = pd.read_csv(table.source)
    expected = pandas_counts(df, "kcal", "gender", "female", "male", df["glucose"].notna())
    got = {p["key"]: p["affected"] for p in artifact["exclusions"]}
    assert got == expected
    assert got["sex_neutral_500_5000"] == 501  # the pack finding's own count, on every row


def test_the_recall_screens_count_what_pandas_counts(recalls):
    table, artifact = recalls
    df = pd.read_csv(table.source)
    expected = pandas_counts(df, "energy_kcal", "sex", "F", "M", df["hba1c"].notna())
    assert {p["key"]: p["affected"] for p in artifact["exclusions"]} == expected


def test_counts_are_among_rows_with_the_outcome_measured(recalls, tmp_path):
    """The flow's denominator: an outcome missing on half the rows halves what a screen can remove."""
    table, _ = recalls
    df = pd.read_csv(table.source)
    df.loc[df.index % 2 == 0, "hba1c"] = np.nan
    path = tmp_path / "half.csv"
    df.to_csv(path, index=False)
    half = Ingested(path, tmp_path / "t")
    artifact = half.run(proposals_stage, ProjectState(lens=["dietary"], target="hba1c"))
    expected = pandas_counts(df, "energy_kcal", "sex", "F", "M", df["hba1c"].notna())
    assert {p["key"]: p["affected"] for p in artifact["exclusions"]} == expected
    assert f"`{int(df['hba1c'].notna().sum()):,}` rows with `hba1c` measured" in artifact["basis"]


def test_each_screen_is_offered_with_its_badge_and_none_is_chosen(nhanes):
    _, artifact = nhanes
    ProposalsArtifact.model_validate(artifact)  # the contract's shape
    assert [p["key"] for p in artifact["exclusions"]] == KEYS
    for p in artifact["exclusions"]:
        assert p["evidence"] == {"status": "CONVENTION",
                                 "source": "research/NUTRITION_PACK.md#02 · Implausible intake exclusions"}
        assert set(p) == {"key", "rule", "label", "affected", "evidence"}  # no "selected", no default
        ExclusionRule.model_validate(p["rule"])
    willett = ExclusionRule.model_validate(artifact["exclusions"][0]["rule"])
    assert willett.by.column == "gender"
    assert willett.by.ranges == {"female": (500.0, 3500.0), "male": (800.0, 4200.0)}


def test_the_nhanes_energy_reading(nhanes):
    table, artifact = nhanes
    energy = artifact["energy"]
    assert energy["energy_column"] == "kcal"
    # Sugar is carbohydrate at 4 kcal/g: adjusted with the rest, never left out silently.
    assert energy["nutrients"] == ["protein", "sugar", "carb", "fat_total", "fat_sat", "fat_mon",
                                   "fat_poly"]
    assert energy["strata_candidates"][0] == "gender"
    # A total beside its parts counts their energy twice: the partition refuses, saying why.
    partition = energy["applicability"]["partition"]
    assert not partition["ok"]
    assert ("`sugar`, `fat_sat`, `fat_mon` and `fat_poly` are parts of `carb` and `fat_total`"
            in partition["reason"]), partition["reason"]
    assert set(energy["applicability"]) == {"none", "standard", "residual", "density_multivariate",
                                            "density", "partition"}
    assert energy["usual"] == "residual"
    assert energy["usual_evidence"]["status"] == "CONVENTION"
    df = pd.read_csv(table.source)
    assert math.isclose(energy["r_with_energy"]["fat_total"], df["kcal"].corr(df["fat_total"]), abs_tol=1e-3)
    note = " ".join(energy["notes"])
    assert "`fat_sat`, `fat_mon` and `fat_poly` are parts of `fat_total`" in note


def test_the_recall_energy_reading_offers_grams_not_shares(recalls):
    _, artifact = recalls
    energy = artifact["energy"]
    assert energy["energy_column"] == "energy_kcal"
    assert energy["nutrients"] == ["protein_g", "fat_g", "carbohydrate_g", "fiber_g"]
    assert energy["strata_candidates"][0] == "sex"
    assert all(v["ok"] for v in energy["applicability"].values())
    # Nothing is left out silently: each exposure it does not adjust is named, with why.
    left = {e["column"]: e["reason"] for e in energy["not_adjusted"]}
    assert left["sodium_mg"] == "carries no energy"
    assert left["protein_pct_kcal"] == "already a share of energy"


def test_kilojoules_are_compared_in_kilojoules(tmp_path):
    table = Ingested(SAMPLES / "nhanes_kilojoules.csv", tmp_path)
    artifact = table.run(proposals_stage, ProjectState(lens=["dietary"]))
    neutral = next(p for p in artifact["exclusions"] if p["key"] == "sex_neutral_500_5000")
    assert (neutral["rule"]["low"], neutral["rule"]["high"]) == (2092.0, 20920.0)
    assert "kJ" in neutral["label"]
    df = pd.read_csv(table.source)
    assert neutral["affected"] == int(((df.DR1TKCAL < 2092) | (df.DR1TKCAL > 20920)).sum())


def test_nothing_is_proposed_without_the_dietary_lens(recalls):
    table, _ = recalls
    artifact = table.run(proposals_stage, ProjectState(lens=["clinical"], target="hba1c"))
    assert artifact["exclusions"] == [] and artifact["energy"] is None


def test_confirmed_roles_decide_the_energy_column_and_the_nutrients(recalls):
    table, _ = recalls
    roles = {"energy_kcal": "energy", "protein_g": "exposure", "fat_g": "covariate",
             "carbohydrate_g": "exposure", "fiber_g": "excluded", "sex": "covariate"}
    artifact = table.run(proposals_stage, ProjectState(lens=["dietary"], target="hba1c", roles=roles))
    assert artifact["energy"]["nutrients"] == ["protein_g", "carbohydrate_g"]


def test_the_roles_stage_proposal_is_read_when_nothing_is_confirmed():
    data = {"columns": [{"column": "kcal", "proposed": "energy"}, {"column": "SEQN", "proposed": "identifier"}]}
    assert roles_from(None, Bundle(data=data)) == {"kcal": "energy", "SEQN": "identifier"}
    assert roles_from(None, data) == {"kcal": "energy", "SEQN": "identifier"}
    assert roles_from({"kcal": "covariate"}, data) == {"kcal": "covariate"}
    assert roles_from(None, None) == {}


# ── what a rule removes (Tier A) ─────────────────────────────────────────────

def test_a_range_keeps_its_bounds_and_never_removes_a_missing_value():
    frame = pd.DataFrame({"kcal": [499.9, 500, 5000, 5000.1, np.nan]})
    rule = ExclusionRule(column="kcal", low=500, high=5000, reason="implausible intakes")
    assert rule_excludes(frame, rule).tolist() == [True, False, False, True, False]


def test_a_one_sided_rule():
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    assert rule_excludes(frame, ExclusionRule(column="x", high=2, reason="r")).tolist() == [False, False, True]
    assert rule_excludes(frame, ExclusionRule(column="x", low=2, reason="r")).tolist() == [True, False, False]


def test_a_rule_by_level_judges_each_level_by_its_own_range_and_the_rest_by_the_default():
    frame = pd.DataFrame({"sex": ["F", "M", "F", "M", "X", None],
                          "kcal": [3600, 3600, 450, 790, 450, 450]})
    rule = ExclusionRule(column="kcal", low=400, high=None, reason="r",
                         by=RangeByLevel(column="sex", ranges={"F": (500, 3500), "M": (800, 4200)}))
    assert rule_excludes(frame, rule).tolist() == [True, False, True, True, False, False]


def test_numeric_levels_match_however_they_were_written():
    frame = pd.DataFrame({"RIAGENDR": [1.0, 2.0, 1.0], "kcal": [4300.0, 4300.0, 700.0]})
    rule = ExclusionRule(column="kcal", reason="r",
                         by=RangeByLevel(column="RIAGENDR", ranges={"1": (800, 4200), "2": (500, 3500)}))
    assert rule_excludes(frame, rule).tolist() == [True, True, True]

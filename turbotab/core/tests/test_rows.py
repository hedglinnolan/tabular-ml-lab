"""Row identity (Tier A): the cohort's counts, the split's seal, the role proposals.

Cohort counts are checked against an independent pandas computation over the source CSV, so a
wrong count cannot hide behind the code that produced it.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from turbotab.core.datastore import DataStore, ingest
from turbotab.core.decisions import ExclusionRule, ProjectState, RangeByLevel
from turbotab.core.stages.rows import (
    compute_cohort, draw_split, energy_bearing, propose_roles, rule_keep, split_inputs,
)

REPO = Path(__file__).resolve().parents[3]
SAMPLES = REPO / "turbotab" / "sample_data"
DIETARY = SAMPLES / "dietary_recalls.csv"


def _nhanes() -> Path | None:
    """The real NHANES export: untracked, beside the main checkout (a worktree looks up)."""
    candidates = [os.environ.get("TURBOTAB_NHANES", ""), REPO / "_tt_tmp_nhanes.csv"]
    if REPO.parent.name == "worktrees":
        candidates.append(REPO.parents[2] / "_tt_tmp_nhanes.csv")
    for c in candidates:
        if c and Path(c).is_file():
            return Path(c)
    return None


NHANES = _nhanes()
needs_nhanes = pytest.mark.skipif(NHANES is None, reason="the NHANES export is not on this machine")


@pytest.fixture(scope="module")
def stores(tmp_path_factory):
    root = tmp_path_factory.mktemp("rows")
    out = {}
    for name, src in (("dietary", DIETARY), ("nhanes", NHANES)):
        if src is None:
            continue
        dest = root / f"{name}.parquet"
        ingest(src, dest)
        out[name] = (DataStore(dest, 2 << 30), pd.read_csv(src))
    yield out
    for store, _ in out.values():
        store.close()


def _ingest_info(store: DataStore) -> dict:
    return {"columns": [c.to_dict() for c in store.info().columns]}


def _pandas_cohort(df: pd.DataFrame, target: str, rules: list[ExclusionRule], missing: str | None,
                   predictors: list[str]) -> list[int]:
    """The flow's counts, computed independently: n after each step."""
    counts = [len(df)]
    keep = df[target].notna()
    counts.append(int(keep.sum()))
    for rule in rules:
        v = pd.to_numeric(df[rule.column], errors="coerce")
        lo = pd.Series(rule.low if rule.low is not None else -np.inf, index=df.index, dtype=float)
        hi = pd.Series(rule.high if rule.high is not None else np.inf, index=df.index, dtype=float)
        if rule.by is not None:
            for level, (a, b) in rule.by.ranges.items():
                at = df[rule.by.column].astype(str) == level
                lo[at] = a if a is not None else -np.inf
                hi[at] = b if b is not None else np.inf
        keep &= v.isna() | ((v >= lo) & (v <= hi))
        counts.append(int(keep.sum()))
    if missing == "complete_case":
        keep &= df[predictors].notna().all(axis=1)
        counts.append(int(keep.sum()))
    return counts


def _roles(predictors: list[str], identifier: str | None = None) -> dict[str, str]:
    roles = {c: "covariate" for c in predictors}
    if identifier:
        roles[identifier] = "identifier"
    return roles


KCAL = ExclusionRule(column="kcal", low=500, high=5000, reason="implausible intake")
NHANES_CASES = [
    ([], "impute", ["age", "bmi", "kcal"]),
    ([KCAL], "complete_case", ["age", "bmi", "kcal", "protein", "meds_hbp"]),
    ([KCAL, ExclusionRule(column="bmi", high=60, reason="implausible BMI")], "complete_case",
     ["age", "bmi", "kcal", "meds_hbp", "meds_chol"]),
    ([ExclusionRule(column="kcal", low=500, high=5000, reason="by sex",
                    by=RangeByLevel(column="gender", ranges={"male": (800, 4200), "female": (600, 3500)}))],
     "complete_case", ["age", "kcal", "meds_chol"]),
    ([ExclusionRule(column="kcal", low=600, reason="under-reporting")], None, ["age"]),
]


@needs_nhanes
@pytest.mark.parametrize("rules,missing,preds", NHANES_CASES)
def test_the_nhanes_cohort_counts_match_pandas(stores, rules, missing, preds):
    store, df = stores["nhanes"]
    state = ProjectState(target="glucose", roles=_roles(preds, "SEQN"), exclusions=rules, missing=missing)
    steps, kept, got_preds = compute_cohort(store, state, _ingest_info(store))
    assert [s["n"] for s in steps] == _pandas_cohort(df, "glucose", rules, missing, preds)
    assert len(kept) == steps[-1]["n"] and len(set(kept.tolist())) == len(kept)
    assert got_preds == [c for c in df.columns if c in set(preds)]  # table order
    for before, after in zip(steps, steps[1:]):
        assert after["dropped"] == before["n"] - after["n"]


DIET_CASES = [
    ([ExclusionRule(column="energy_kcal", low=500, high=5000, reason="implausible")], "complete_case",
     ["age", "bmi", "energy_kcal", "fat_g"]),
    ([ExclusionRule(column="energy_kcal", low=500, high=5000, reason="implausible"),
      ExclusionRule(column="age", low=30, reason="adults over 30")], "impute", ["age"]),
    ([], "complete_case", ["sodium_mg", "protein_g"]),
]


@pytest.mark.parametrize("rules,missing,preds", DIET_CASES)
def test_the_dietary_cohort_counts_match_pandas(stores, rules, missing, preds):
    store, df = stores["dietary"]
    state = ProjectState(target="hba1c", roles=_roles(preds, "participant_id"), exclusions=rules,
                         missing=missing)
    steps, kept, _ = compute_cohort(store, state, _ingest_info(store))
    assert [s["n"] for s in steps] == _pandas_cohort(df, "hba1c", rules, missing, preds)
    expected_ids = np.flatnonzero(pd.Series(True, index=df.index).to_numpy())  # file order = row id
    assert set(kept.tolist()) <= set(expected_ids.tolist())


def test_a_missing_value_is_kept_by_a_range_rule():
    frame = pd.DataFrame({"x": [np.nan, 1.0, 10.0, 5.0]})
    keep = rule_keep(frame, ExclusionRule(column="x", low=2, high=8, reason="r"))
    assert keep.tolist() == [True, False, False, True]


# ── the split ────────────────────────────────────────────────────────────────


def _dietary_split(stores, **spec):
    store, df = stores["dietary"]
    state = ProjectState(target="hba1c", roles=_roles(["age"], "participant_id"))
    rows = np.arange(len(df))
    inputs = split_inputs(state, rows, store, "regression")
    return df, draw_split(rows, **spec, **inputs)


def test_no_participant_is_on_both_sides_and_folds_are_grouped(stores):
    df, (frame, info) = _dietary_split(stores, holdout=0.2, seed=7, folds=5)
    assert info["grouped_by"] == "participant_id" and info["n_groups"] == 300
    people = df["participant_id"].to_numpy()[frame["row_id"].to_numpy()]
    sides = pd.DataFrame({"p": people, "part": frame["partition"], "fold": frame["fold"]})
    assert sides.groupby("p")["part"].nunique().max() == 1
    train = sides[sides["part"] == "train"]
    assert train.groupby("p")["fold"].nunique().max() == 1
    assert set(train["fold"]) == set(range(5)) and (sides.loc[sides["part"] == "holdout", "fold"] == -1).all()
    assert info["n_holdout"] == 120 and info["n_train"] == 480  # 60 of 300 people, 2 rows each


def test_the_same_seed_gives_the_same_split_and_another_seed_another(stores):
    _, (a, _) = _dietary_split(stores, holdout=0.2, seed=3, folds=5)
    _, (b, _) = _dietary_split(stores, holdout=0.2, seed=3, folds=5)
    _, (c, _) = _dietary_split(stores, holdout=0.2, seed=4, folds=5)
    pd.testing.assert_frame_equal(a, b)
    assert not a["partition"].equals(c["partition"])


def test_a_zero_holdout_is_cross_validation_only(stores):
    _, (frame, info) = _dietary_split(stores, holdout=0.0, seed=0, folds=4)
    assert (frame["partition"] == "train").all() and info["n_holdout"] == 0
    assert set(frame["fold"]) == {0, 1, 2, 3}


def test_stratified_held_out_proportions_are_within_one_row():
    rng = np.random.default_rng(11)
    y = rng.choice(["a", "b", "c"], size=997, p=[0.6, 0.3, 0.1])
    rows = np.arange(997)
    frame, info = draw_split(rows, holdout=0.25, seed=5, folds=5, y=y)
    assert info["stratified"]
    held = frame["partition"].to_numpy() == "holdout"
    for label in "abc":
        n = int((y == label).sum())
        assert abs(int((y[held] == label).sum()) - 0.25 * n) <= 1, label
    # folds are stratified too
    train = ~held
    for f in range(5):
        in_fold = train & (frame["fold"].to_numpy() == f)
        share = (y[in_fold] == "c").mean()
        assert abs(share - (y[train] == "c").mean()) < 0.03


def test_removing_rows_never_moves_another_row_across_the_seal():
    rng = np.random.default_rng(2)
    universe = np.arange(2000)
    groups = rng.integers(0, 700, size=2000)
    y = rng.choice([0, 1], size=2000)
    full, _ = draw_split(universe, holdout=0.2, seed=1, folds=5, y=y, groups=groups, universe=universe)
    kept = np.sort(rng.choice(universe, size=1300, replace=False))
    part, info = draw_split(kept, holdout=0.2, seed=1, folds=5, y=y, groups=groups, universe=universe)
    was = full.set_index("row_id")["partition"]
    assert (part.set_index("row_id")["partition"] == was.loc[kept]).all()
    assert set(info["sealed"].tolist()) == set(full.loc[full["partition"] == "holdout", "row_id"])


def test_a_tiny_table_is_split_without_refusal():
    frame, info = draw_split(np.arange(3), holdout=0.4, seed=0, folds=5)
    assert info["n_holdout"] >= 1 and info["n_train"] >= 1 and info["folds"] <= 2


# ── role proposals ───────────────────────────────────────────────────────────


def _proposals(store: DataStore, lens: list[str], target: str) -> dict[str, dict]:
    from turbotab.core.stages.rows import _acquisition_columns, _energy_reading

    columns = [c for c in store.summaries() if not c["name"].startswith("__")]
    out = propose_roles(columns, lens=lens, target=target, n_rows=store.n_rows,
                        energy_column=_energy_reading(columns), acquisition=_acquisition_columns(columns))
    return {p["column"]: p for p in out}


def test_dietary_roles_read_the_fixture(stores):
    store, _ = stores["dietary"]
    roles = _proposals(store, ["dietary"], "hba1c")
    assert "hba1c" not in roles
    assert roles["participant_id"]["proposed"] == "identifier"
    assert roles["energy_kcal"]["proposed"] == "energy" and roles["energy_kcal"]["unit"] == "kcal"
    assert {roles[c]["proposed"] for c in ("protein_g", "fat_g", "carbohydrate_g", "sodium_mg")} == {"exposure"}
    assert roles["recall_date"]["proposed"] == "time" and roles["age"]["proposed"] == "covariate"
    assert all(len(p["reason"].split()) <= 16 for p in roles.values())


@needs_nhanes
def test_nhanes_roles_link_flags_and_name_the_respondent(stores):
    store, _ = stores["nhanes"]
    roles = _proposals(store, ["dietary"], "glucose")
    assert roles["SEQN"]["proposed"] == "identifier" and roles["SEQN"]["confidence"] == "high"
    assert roles["kcal"]["proposed"] == "energy"
    assert roles["imputed_bmi"]["proposed"] == "flag" and roles["imputed_bmi"]["linked_to"] == "bmi"
    assert roles["weight"]["proposed"] == "covariate"  # body weight, not a survey weight
    assert {roles[c]["proposed"] for c in ("protein", "carb", "fat_total", "fat_sat", "sugar")} == {"exposure"}
    assert roles["cycle_begin_year"]["proposed"] == "time"
    assert all(len(p["reason"].split()) <= 16 for p in roles.values())


def test_energy_bearing_means_an_amount_that_carries_kcal():
    assert all(energy_bearing(c) for c in ("protein_g", "fat_total", "carb", "DR1TPROT", "fiber_g"))
    assert not any(energy_bearing(c) for c in ("protein_pct_kcal", "kcal", "sodium_mg", "sugar", "age"))

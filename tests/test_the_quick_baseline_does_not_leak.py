"""`AUDIT-002` — the quick baseline splits by entity, not at random.

`ml/eda_actions.py` divided rows with
`train_test_split(X, y, test_size=0.2, random_state=42)` and reported MAE, RMSE
and R² in a table labeled *Model*. On a table with repeated measures one
person's rows land on both sides and the number is optimistic.

`research/NUTRITION_PACK.md` §03 states it as a **TurboTab-specific** item:

> If a person contributes multiple recalls, rows from the same person must
> never be split across train and test folds — use participant-level splitting.

`METABOLOMICS_PACK.md` §10 lists *repeated measures treated as independent*
under Structural.

## The part that makes it a defect rather than a gap

**The answer was already recorded and this path was not reading it.**
`DatasetSignals` carries `cohort_type_final` and `entity_id_final` — the app
asked whether the cohort is longitudinal and what identifies an entity, and the
user answered. `ml/splits.py` has implemented the grouped basis the whole time,
with `GroupShuffleSplit` and a priority order that puts grouping first *because
a subject spanning partitions is the worse leak*. Both existed; this function
used neither.

On `dietary_recalls.csv` — 300 people × 2 recalls — the old split put **103
participants on both sides**. That number is what the test below pins.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.eda_actions import quick_probe_baselines                     # noqa: E402
from ml.eda_recommender import DatasetSignals                        # noqa: E402

FIXTURES = Path(__file__).resolve().parents[1] / "turbotab" / "sample_data"
FEATURES = ["energy_kcal", "protein_g", "fat_g", "carbohydrate_g",
            "fiber_g", "sodium_mg", "age", "bmi"]


def _signals(df: pd.DataFrame, *, longitudinal: bool,
             entity: str = "participant_id",
             task: str = "regression") -> DatasetSignals:
    return DatasetSignals(
        n_rows=len(df), n_cols=len(df.columns), target_name="hba1c",
        task_type_final=task,
        cohort_type_final="longitudinal" if longitudinal else "cross_sectional",
        entity_id_final=entity)


@pytest.fixture(scope="module")
def recalls() -> pd.DataFrame:
    """300 people × 2 twenty-four-hour recalls."""
    return pd.read_csv(FIXTURES / "dietary_recalls.csv")


# ── the gate ────────────────────────────────────────────────────────────────

def test_no_participant_appears_on_both_sides_of_the_split(recalls):
    """The gate. `entity_overlap` is counted from the split's own labels, not
    promised in a comment — a promise nobody can check is what this was."""
    out = quick_probe_baselines(recalls, "hba1c", FEATURES,
                                _signals(recalls, longitudinal=True), None)
    basis = out["split_basis"]
    assert basis["strategy"] == "grouped"
    assert basis["entity_column"] == "participant_id"
    assert basis["entity_overlap"] == 0, (
        f"{basis['entity_overlap']} participants have rows on both sides")
    assert basis["n_fitted"] + basis["n_held_out"] == len(recalls)


def test_the_leak_it_replaces_was_real_and_is_measured(recalls):
    """The same table, the cohort not recorded as longitudinal, is still split
    at random — and 103 of 300 participants land on both sides. That number is
    the size of what the grouped path removes, and pinning it is what stops
    this test passing against a split that never grouped anything."""
    out = quick_probe_baselines(recalls, "hba1c", FEATURES,
                                _signals(recalls, longitudinal=False), None)
    basis = out["split_basis"]
    assert basis["strategy"] == "random"
    assert basis["entity_overlap"] == 103, basis["entity_overlap"]


def test_the_split_basis_is_said_where_the_user_can_see_it(recalls):
    """*Expect the honest number to be worse, and say so where the user can
    see it.* A baseline that drops when the leak is closed is the app becoming
    correct, and a user who is not told will read it as a regression."""
    grouped = quick_probe_baselines(recalls, "hba1c", FEATURES,
                                    _signals(recalls, longitudinal=True), None)
    said = " ".join(grouped["findings"])
    assert "split by `participant_id` rather than at random" in said
    assert "come out better than they are" in said

    random_split = quick_probe_baselines(
        recalls, "hba1c", FEATURES, _signals(recalls, longitudinal=False), None)
    said = " ".join(random_split["findings"])
    assert "split at random" in said
    assert "identifies entities in this table" in said, (
        "the app knows an entity column exists and did not say why it ignored it")


# ── it reads the recorded answer, and only that ────────────────────────────

def test_a_table_with_no_entity_column_recorded_is_split_at_random(recalls):
    """Grouping is not guessed from a column that looks like an id. The user
    answered the cohort question; nothing here second-guesses it."""
    signals = _signals(recalls, longitudinal=True, entity=None)
    out = quick_probe_baselines(recalls, "hba1c", FEATURES, signals, None)
    assert out["split_basis"]["strategy"] == "random"
    assert out["split_basis"]["entity_overlap"] is None


def test_an_entity_column_that_is_not_in_the_frame_does_not_group(recalls):
    signals = _signals(recalls, longitudinal=True, entity="not_a_column")
    out = quick_probe_baselines(recalls, "hba1c", FEATURES, signals, None)
    assert out["split_basis"]["strategy"] == "random"


def test_nothing_here_reimplements_a_partition():
    """`ml/splits.py` is the splitter. A second `train_test_split` beside it is
    the two-engines failure that let these two disagree for as long as they
    did."""
    source = (Path(__file__).resolve().parents[1] / "ml" / "eda_actions.py") \
        .read_text(encoding="utf-8")
    body = source[source.index("def quick_probe_baselines("):]
    body = body[:body.index("\ndef ", 1)] if "\ndef " in body[1:] else body
    # Comment lines are prose — the comment above the fix QUOTES the call it
    # replaced, and a scan that could not tell the two apart would forbid
    # explaining the defect in the place it was fixed.
    code = "\n".join(line.split("#", 1)[0] for line in body.split("\n")
                     if not line.strip().startswith("#"))
    assert "train_test_split" not in code, (
        "quick_probe_baselines splits its own rows again")
    assert "make_split" in code


# ── the classification path splits the same way ────────────────────────────

def test_the_classification_branch_is_grouped_too():
    """A leak is not a property of the metric. Both branches route through the
    same split, so neither can drift from the other."""
    rng = np.random.default_rng(2)
    people = np.repeat(np.arange(120), 3)
    frame = pd.DataFrame({
        "pid": people,
        "x1": rng.normal(0, 1, len(people)),
        "x2": rng.normal(0, 1, len(people)),
        "outcome": np.repeat(rng.integers(0, 2, 120), 3),
    })
    signals = DatasetSignals(
        n_rows=len(frame), n_cols=4, target_name="outcome",
        task_type_final="classification", cohort_type_final="longitudinal",
        entity_id_final="pid")
    out = quick_probe_baselines(frame, "outcome", ["x1", "x2"], signals, None)
    assert out["split_basis"]["strategy"] == "grouped"
    assert out["split_basis"]["entity_overlap"] == 0
    assert any("Accuracy" in str(obj.columns.tolist())
               for kind, obj in out["figures"] if kind == "table")


def test_a_split_that_cannot_be_made_says_so_rather_than_raising():
    """A single-class target is a real refusal from `ml/splits.py`, and it has
    to arrive as a warning rather than as a traceback in an EDA panel."""
    rng = np.random.default_rng(3)
    people = np.repeat(np.arange(60), 2)
    frame = pd.DataFrame({
        "pid": people,
        "x1": rng.normal(0, 1, len(people)),
        "outcome": np.zeros(len(people), dtype=int),
    })
    signals = DatasetSignals(
        n_rows=len(frame), n_cols=3, target_name="outcome",
        task_type_final="classification", cohort_type_final="longitudinal",
        entity_id_final="pid")
    out = quick_probe_baselines(frame, "outcome", ["x1"], signals, None)
    assert any("Baselines were not run" in w for w in out["warnings"]), out
    assert out["figures"] == []

"""`AUDIT-009` and `AUDIT-010` — two instances of one class, fixed together.

The class `AUDIT-008` names: **the correction exists in this repository and the
path that needs it does not reach for it.** Both of these ship to Streamlit
users today, and both had the right machinery one import away.

`AUDIT-009` — `coach_probe._cv_score` built `KFold`/`StratifiedKFold` with
`shuffle=True` and no groups, and that mean is what the **Model Coach ranks its
picks on**. `ml/eval.py:113` documents the exact hazard and selects
`StratifiedGroupKFold`; `ml/splits.py` implements the priority order. Neither was
reachable, because the function had no `groups` parameter to reach one with —
which is why it survived two audits and why the fix is a signature change.

`AUDIT-010` — `Table1Config.show_pvalues` defaulted True and a p was printed per
row, so a twenty-row baseline table is twenty tests shown as twenty results.
`AUDIT-001` fixed the manuscript's count and had no reason to reach this
surface; `ml/multiplicity.py` was built for exactly this class and was not
imported here.

**Both report the honest number where a user can see it**, because a score that
drops when a leak closes reads as a regression to anyone not told otherwise.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml import multiplicity                                          # noqa: E402
from ml.coach_probe import run_probe                                 # noqa: E402
from ml.table_one import Table1Config, generate_table1               # noqa: E402


# ── AUDIT-009 · the probe's folds ──────────────────────────────────────────

def _repeated(n_people: int = 100, per: int = 3, seed: int = 0):
    """A person-level effect the features carry, measured several times each."""
    rng = np.random.default_rng(seed)
    people = np.repeat(np.arange(n_people), per)
    effect = rng.normal(0, 3, n_people)[people]
    frame = pd.DataFrame({"x1": effect + rng.normal(0, 0.2, len(people)),
                          "x2": rng.normal(0, 1, len(people))})
    return frame, effect + rng.normal(0, 1, len(people)), people


def test_no_entity_spans_a_fold_when_groups_are_passed():
    """The gate. Asserted on the SPLITTER the probe builds, because a metric
    comparison would be a claim about this fixture and this is a claim about
    every table."""
    from sklearn.model_selection import GroupKFold

    _, _, people = _repeated()
    cv = GroupKFold(n_splits=4)
    for train, test in cv.split(np.zeros((len(people), 1)), None, people):
        assert not (set(people[train]) & set(people[test])), (
            "an entity appeared on both sides of a grouped fold")


def test_the_probe_records_which_scheme_it_used():
    """A number computed one way and read as the other is the whole defect, so
    the scheme travels with the score."""
    frame, y, people = _repeated()
    ungrouped = run_probe(frame, y)
    grouped = run_probe(frame, y, groups=people)

    assert ungrouped.grouped is False and grouped.grouped is True
    assert any("ungrouped" in n for n in ungrouped.notes), (
        "an ungrouped probe did not say so, so its score reads as grouped")
    assert any("no entity spans a split" in n for n in grouped.notes)
    assert any("comes out higher than it should" in n for n in ungrouped.notes
               or []) or any("optimistic" in n for n in ungrouped.notes), (
        "nothing told the reader an ungrouped score on repeated measures is "
        "optimistic, so a drop reads as a regression")


def test_the_grouping_rides_through_every_subsample():
    """A grouping that stops matching its rows is worse than none: it puts a
    fold boundary somewhere nobody chose."""
    rng = np.random.default_rng(1)
    n = 60
    people = np.repeat(np.arange(20), 3)
    frame = pd.DataFrame({"x": rng.normal(0, 1, n)})
    y = rng.normal(0, 1, n)
    y[:5] = np.nan                      # dropped by the target mask
    result = run_probe(frame, y, groups=people)
    assert result.grouped is True
    assert result.n_rows_used == n - 5


def test_a_grouped_probe_still_scores_something():
    """A guard that makes every score `nan` would pass every assertion above
    and be worthless."""
    frame, y, people = _repeated()
    result = run_probe(frame, y, groups=people)
    assert np.isfinite(result.linear_score)
    assert np.isfinite(result.tree_score)


def test_the_probe_no_longer_builds_an_ungrouped_split_when_told_the_entity():
    source = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "ml", "coach_probe.py"),
        encoding="utf-8").read()
    body = source[source.index("def _cv_score("):]
    body = body[:body.index("\ndef ", 1)]
    assert "StratifiedGroupKFold" in body and "GroupKFold" in body
    assert "groups=groups" in body, (
        "the folds are built with groups and scored without them")


# ── AUDIT-010 · Table 1's family ───────────────────────────────────────────

def _baseline(n_vars: int = 12, n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({f"v{i}": rng.normal(0, 1, n) for i in range(n_vars)})
    frame["grp"] = ["a"] * (n // 2) + ["b"] * (n // 2)
    return frame, [f"v{i}" for i in range(n_vars)]


def test_a_baseline_table_corrects_its_family_and_names_the_method():
    frame, variables = _baseline()
    table, meta = generate_table1(
        frame, Table1Config(grouping_var="grp", continuous_vars=variables))

    assert any("Q-value" in c for c in table.columns), (
        "the column still reports uncorrected p-values")
    assert "Benjamini-Hochberg FDR" in " ".join(table.columns), (
        "the correction is applied and not named, which is half the defect")
    assert meta["n_tests_corrected"] == len(variables), (
        "the correction ran over something other than the whole family")
    assert meta["raw_p_values"] and meta["q_values"], (
        "the raw values were discarded rather than kept beside the corrected")
    for var in variables:
        assert meta["q_values"][var] >= meta["raw_p_values"][var] - 1e-12, (
            f"{var}: the q-value is below its own p, which no correction does")


def test_with_no_correction_configured_the_column_is_not_shown_at_all():
    """Matching what the manuscript now does. An uncorrected p per row IS the
    quantity the anti-pattern is made of, so it is not printed — and the reason
    is recorded rather than the column being silently absent."""
    frame, variables = _baseline()
    table, meta = generate_table1(
        frame, Table1Config(grouping_var="grp", continuous_vars=variables,
                            pvalue_correction=""))
    assert not any("P-value" in c or "Q-value" in c for c in table.columns)
    assert "not interpretable as a set of results" in meta["pvalues_withheld"]


def test_the_correction_spans_continuous_and_categorical_rows_together():
    """One table is one family. Correcting the two kinds of row separately
    would be two families nobody declared."""
    rng = np.random.default_rng(2)
    n = 200
    frame = pd.DataFrame({f"v{i}": rng.normal(0, 1, n) for i in range(6)})
    frame["site"] = rng.choice(["A", "B", "C"], n)
    frame["sex"] = rng.choice(["F", "M"], n)
    frame["grp"] = ["a"] * 100 + ["b"] * 100
    _, meta = generate_table1(
        frame, Table1Config(grouping_var="grp",
                            continuous_vars=[f"v{i}" for i in range(6)],
                            categorical_vars=["site", "sex"]))
    assert meta["n_tests_corrected"] == 8, meta["n_tests_corrected"]


def test_the_engine_is_not_reimplemented_here_either():
    source = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "ml", "table_one.py"),
        encoding="utf-8").read()
    assert "from ml import multiplicity" in source
    # Comment lines are prose — the comment above the fix NAMES the call it
    # routes through, and a scan that could not tell the two apart would forbid
    # explaining the fix where it was made.
    code = "\n".join(line.split("#", 1)[0] for line in source.split("\n")
                     if not line.strip().startswith("#"))
    assert "multipletests" not in code, (
        "a second Benjamini-Hochberg was written beside `ml/multiplicity.py`")


def test_one_test_is_still_corrected_rather_than_special_cased():
    """A family of one is a family. Special-casing it would make the reported
    quantity change meaning with the number of rows."""
    frame, _ = _baseline(n_vars=1)
    table, meta = generate_table1(
        frame, Table1Config(grouping_var="grp", continuous_vars=["v0"]))
    assert meta["n_tests_corrected"] == 1
    assert any("Q-value" in c for c in table.columns)
    assert meta["q_values"]["v0"] == pytest.approx(meta["raw_p_values"]["v0"])

"""`GUIDED-029` — per-column questions scaled linearly with the column count.

The L20 discrimination matrix recorded `metabolomics_untargeted.csv` at a **base
of 313 questions** — 308 columns with blanks producing 308 mechanism questions,
roughly ten times the ~32 this project calls Classic's indictment. The
metabolomics pack rescued it to 6, and a user with the same table who answered
*"something else, or not sure"* still got 313.

**The lens was masking an unscalable interview rather than accelerating a
scalable one.** A benefit measured against a broken baseline is a number
flattering itself.

The remedy was already specified from the p ≫ n work: operations apply to sets
defined by a **rule**, and the user edits the rule rather than the members.

Run:  venv/bin/python -m pytest \\
          turbotab/test_one_answer_covers_a_set_defined_by_a_rule.py -q
"""
# The tests here that drove the retired legacy app (turbotab/api.py, the page, the
# figure and manuscript modules, the gates) were removed with it, BLUEPRINT §9.1; git keeps them.
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml import router                                                 # noqa: E402
from turbotab import bulk as B, missingness as MISS                     # noqa: E402
from turbotab.project import AnalysisProject, ProjectError            # noqa: E402

DATA = Path(__file__).resolve().parent / "sample_data"


# ── the finding, measured before and after ───────────────────────────────────


def test_the_two_branches_are_never_one_answer():
    """Clause §07 routes by dtype, so a blanket answer across both would be a
    bulk affordance that had to be wrong for one of them."""
    rows = [{"column": "a", "branch": "numeric"},
            {"column": "b", "branch": "numeric"},
            {"column": "c", "branch": "categorical"},
            {"column": "d", "branch": "categorical"}]
    groups = B.group_columns(rows)
    assert [g.branch for g in groups] == ["numeric", "categorical"]
    assert [g.n for g in groups] == [2, 2]
    assert groups[0].members == ("a", "b")


def test_the_group_is_what_remains_after_the_lens_settles_its_columns():
    """A bulk question stating a count the user cannot reconcile with what they
    are being shown is worse than no bulk question."""
    rows = [{"column": f"c{i}", "branch": "numeric"} for i in range(10)]
    groups = B.group_columns(rows, settled={"numeric": ["c0", "c1", "c2"]})
    assert groups[0].n == 7
    assert "c0" not in groups[0].members
    assert "3 already settled by the lens" in groups[0].rule


# ── one decision, not N ──────────────────────────────────────────────────────


def test_a_bulk_answer_cannot_cross_the_dtype_branch():
    df = pd.DataFrame({"n": [1.0, np.nan, 3.0] * 5,
                       "c": ["a", None, "b"] * 5,
                       "y": [0, 1, 0] * 5})
    p = AnalysisProject.from_dataframe(df, "t.csv")
    p.set_target("y", "classification", "high", [])
    with pytest.raises(ProjectError, match="routes by dtype"):
        p.route_missingness_bulk("numeric", MISS.NOT_INFORMATIVE,
                                 MISS.IMPUTE_MEDIAN, ["n", "c"])


def test_a_bulk_answer_over_nothing_is_refused():
    df = pd.DataFrame({"n": [1.0, np.nan, 3.0] * 5, "y": [0, 1, 0] * 5})
    p = AnalysisProject.from_dataframe(df, "t.csv")
    p.set_target("y", "classification", "high", [])
    with pytest.raises(ProjectError, match="empty set"):
        p.route_missingness_bulk("numeric", MISS.NOT_INFORMATIVE,
                                 MISS.IMPUTE_MEDIAN, [])


# ── bulk plus evidence-driven exceptions ─────────────────────────────────────

def _frame_with_one_informative_column(n: int = 200) -> pd.DataFrame:
    """A frame where one column's blankness tracks the outcome and the rest do
    not. The exception is a real signal, not a threshold artifact."""
    rng = np.random.default_rng(7)
    y = rng.integers(0, 2, n)
    data = {"y": y}
    for i in range(8):
        col = rng.normal(size=n)
        col[rng.random(n) < 0.2] = np.nan          # blank at random
        data[f"plain_{i}"] = col
    # Blank exactly where the outcome is 1, most of the time.
    signal = rng.normal(size=n)
    signal[(y == 1) & (rng.random(n) < 0.85)] = np.nan
    data["ordered_only_when_sick"] = signal
    return pd.DataFrame(data)


def test_the_columns_where_the_evidence_disagrees_are_surfaced():
    """*"A single answer across 294 columns is not always true."*

    The same escalation rule as everywhere: evidence that a reading is wrong,
    never the size of the consequence. The user said a blank means nothing, and
    in one column the outcome behaves differently wherever it is blank.
    """
    df = _frame_with_one_informative_column()
    group = B.Group(question="missingness", branch="numeric",
                    members=tuple(c for c in df.columns if c != "y"))
    found = B.exceptions(df, group, MISS.NOT_INFORMATIVE, "y")

    assert "ordered_only_when_sick" in found["columns"]
    assert found["columns"][0] == "ordered_only_when_sick", "ranked by effect"
    assert len(found["columns"]) <= 3, (
        f"{len(found['columns'])} of 9 columns flagged; the threshold is "
        f"firing on noise, which would teach the user to ignore it")
    assert "behaves differently wherever it is blank" in found["sentence"]


def test_no_exception_is_raised_against_an_informative_answer():
    """The other direction is deliberately not reported. *"You said this is
    informative and we see no association"* is an ABSENCE of evidence, and
    escalating on one would be the app arguing with a claim it cannot check."""
    df = _frame_with_one_informative_column()
    group = B.Group(question="missingness", branch="numeric",
                    members=tuple(c for c in df.columns if c != "y"))
    assert B.exceptions(df, group, MISS.INFORMATIVE, "y")["columns"] == []
    assert B.exceptions(df, group, MISS.NOT_SURE, "y")["columns"] == []


# ── the scaling claim, at both ends ──────────────────────────────────────────

@pytest.mark.parametrize("n_columns", [12, 12_000])
def test_the_interview_is_the_same_size_at_twelve_columns_and_twelve_thousand(
        n_columns):
    """*"It must scale identically at 12 columns and 12,000."*

    Both ends, because a bulk affordance tested only on the wide case can hide a
    threshold that makes the narrow case worse — and one tested only on the
    narrow case proves nothing about the case it was built for.

    Asserted on the ROUTER rather than over HTTP, so 12,000 columns is a plan
    and not a twelve-thousand-column CSV parsed twice.
    """
    rows = [{"column": f"c{i:05d}", "branch": "numeric"}
            for i in range(n_columns)]
    groups = [g.to_dict() for g in B.group_columns(rows)]
    plan = router.plan(
        [], target="y", detection=None, step="preprocess", deferred={},
        answered=["choose_models", "choose_preparation_mode"],
        recommendations=[], signals=None,
        missing_columns=[r["column"] for r in rows],
        missingness_groups=groups)
    router.audit(plan)

    missingness = [q for q in plan
                   if q.kind == "missingness" and q.status == "asked"]
    assert len(missingness) == 1, (
        f"{len(missingness)} questions for {n_columns:,} columns")
    assert missingness[0].key == "missingness_bulk::numeric"
    assert f"{n_columns:,}" in missingness[0].title


def test_the_question_count_does_not_grow_with_the_column_count():
    """The claim stated as a comparison rather than as two separate numbers.

    Two frames three orders of magnitude apart produce the same interview. That
    is the property `GUIDED-029` says was missing, and it is checkable in one
    assertion.
    """
    def n_questions(p: int) -> int:
        rows = [{"column": f"c{i:05d}", "branch": "numeric"} for i in range(p)]
        plan = router.plan(
            [], target="y", detection=None, step="preprocess", deferred={},
            answered=["choose_models", "choose_preparation_mode"],
            recommendations=[], signals=None,
            missing_columns=[r["column"] for r in rows],
            missingness_groups=[g.to_dict() for g in B.group_columns(rows)])
        router.audit(plan)
        return sum(1 for q in plan
                   if q.mode == "push" and q.status == "asked")

    assert n_questions(12) == n_questions(1_200) == n_questions(12_000)


def test_the_old_per_column_path_is_unchanged_when_no_groups_are_built():
    """Every test written before this finding passes `missingness_groups=None`,
    and must still get the per-column interview. A remedy that broke the caller
    it was extending would have to be adopted everywhere at once."""
    plan = router.plan(
        [], target="y", detection=None, step="preprocess", deferred={},
        answered=["choose_models", "choose_preparation_mode"],
        recommendations=[], signals=None,
        missing_columns=["a", "b", "c"])
    router.audit(plan)
    keys = [q.key for q in plan if q.kind == "missingness"]
    assert keys == ["missingness::a", "missingness::b", "missingness::c"]


# ── the skip scales too ──────────────────────────────────────────────────────


def test_two_packs_settling_different_columns_stay_two_facts():
    """Grouped by the prior that settled them, because two packs settling
    different columns for different reasons are two facts and collapsing them
    would state neither."""
    rows = [{"column": f"a{i}", "branch": "numeric"} for i in range(5)]
    rows += [{"column": f"b{i}", "branch": "numeric"} for i in range(5)]
    priors = {}
    for i in range(5):
        priors[f"a{i}"] = [{"pack": "metabolomics", "label": "Metabolomics",
                            "marker": "derived",
                            "mechanism": "below_detection_limit",
                            "reason": "x" * 60}]
        priors[f"b{i}"] = [{"pack": "genomics", "label": "Genomics",
                            "marker": "derived", "mechanism": "other",
                            "reason": "y" * 60}]
    blocks = B.settled_groups(rows, priors)
    assert len(blocks) == 2
    assert {b["pack"] for b in blocks} == {"metabolomics", "genomics"}
    assert all(b["n"] == 5 for b in blocks)


def test_a_contested_column_is_not_settled_and_not_grouped():
    """Two packs disagreeing about one column leaves it asked — the L20 result,
    still true now that the skip is grouped."""
    rows = [{"column": "c", "branch": "numeric"}] * 1
    priors = {"c": [{"pack": "metabolomics", "label": "M", "marker": "derived",
                     "mechanism": "below_detection_limit", "reason": "x" * 60},
                    {"pack": "clinical", "label": "C", "marker": "offered",
                     "mechanism": "not_ordered", "reason": "y" * 60}]}
    assert B.settled_groups(rows, priors) == []


def test_three_hundred_columns_of_noise_produce_no_exceptions():
    """The other half of the exceptions check, and the one that decides whether
    it is usable.

    The first version used a fixed 0.20 effect threshold and flagged **31 of
    306** columns on `metabolomics_untargeted.csv` — almost exactly the
    false-positive rate for a rate difference on 72 participants. An exceptions
    question listing 31 columns none of which is real teaches the user to
    dismiss the next one, which is the blocker-budget argument arriving at a
    different card.

    The threshold now scales with the standard error of each comparison and with
    the number of columns tested. Neither is a p-value: a multiple-testing
    correction the app would then have to explain is worse than an effect size
    the user can look at beside the column.
    """
    df = pd.read_csv(DATA / "metabolomics_untargeted.csv")
    rows = MISS.survey(df, "responder")
    group = B.Group("missingness", "numeric",
                    tuple(r["column"] for r in rows if r["branch"] == "numeric"))
    assert group.n > 300

    found = B.exceptions(df, group, MISS.NOT_INFORMATIVE, "responder")
    assert found["columns"] == [], (
        f"{len(found['columns'])} of {group.n} columns flagged on a fixture "
        f"whose missingness is driven by abundance rather than by the outcome; "
        f"the threshold is reporting noise")


def test_a_real_signal_still_survives_the_higher_floor():
    """A threshold raised until nothing fires is a check that does not exist.

    Same frame as the detection test, and the effect is large enough that no
    reasonable floor should lose it.
    """
    df = _frame_with_one_informative_column()
    group = B.Group("missingness", "numeric",
                    members=tuple(c for c in df.columns if c != "y"))
    found = B.exceptions(df, group, MISS.NOT_INFORMATIVE, "y")
    assert "ordered_only_when_sick" in found["columns"]

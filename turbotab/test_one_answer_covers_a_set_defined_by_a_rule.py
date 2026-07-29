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

Run:  turbotab/.venv/bin/python -m pytest \\
          turbotab/test_one_answer_covers_a_set_defined_by_a_rule.py -q
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml import router                                                 # noqa: E402
from turbotab import api, bulk as B, missingness as MISS              # noqa: E402
from turbotab.project import AnalysisProject, ProjectError            # noqa: E402

DATA = Path(__file__).resolve().parent / "sample_data"


@pytest.fixture
def client():
    return TestClient(api.app)


def _upload(client, path: Path) -> str:
    with open(path, "rb") as fh:
        return client.post(
            "/project", files={"file": (path.name, fh, "text/csv")}).json()["id"]


def _asked(client, pid: str, step: str = "preprocess"):
    iv = client.get(f"/project/{pid}/interview?step={step}").json()
    return [q for q in iv["questions"]
            if q["mode"] == "push" and q["status"] == "asked"]


def _prepared(client, path: Path, target: str, lens=("other",)) -> str:
    pid = _upload(client, path)
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_lens", "payload": {"lens": list(lens)}})
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_target", "payload": {"column": target}})
    return pid


# ── the finding, measured before and after ───────────────────────────────────

def test_a_wide_table_with_no_lens_no_longer_asks_one_question_per_column(client):
    """The number in the finding: 308 columns with blanks, 308 questions.

    This is the fixture and the answer that produced the 313 — the lens is
    `other`, so no pack settles anything and nothing is rescued.
    """
    df = pd.read_csv(DATA / "metabolomics_untargeted.csv")
    with_blanks = [c for c in df.columns
                   if df[c].isna().any() and c != "responder"]
    assert len(with_blanks) > 300, "the fixture no longer has the shape"

    pid = _prepared(client, DATA / "metabolomics_untargeted.csv", "responder")
    missingness = [q for q in _asked(client, pid)
                   if q["kind"] == "missingness"]

    assert len(missingness) <= 3, (
        f"{len(missingness)} missingness questions for {len(with_blanks)} "
        f"columns; the interview still scales with p")

    bulk = [q for q in missingness if q["key"].startswith("missingness_bulk::")]
    assert len(bulk) == 1
    assert bulk[0]["key"] == "missingness_bulk::numeric"
    assert "306 numeric columns have blanks" in bulk[0]["title"]
    # The user edits the RULE, and the rule is in the question.
    assert "every numeric column with blanks" in bulk[0]["why"]


def test_a_group_of_one_is_asked_rather_than_ruled(client):
    """*"A bulk affordance offered over one leftover column is worse than
    asking."* `sex` is the only categorical column with blanks on this fixture,
    so it gets the ordinary question and no rule is invented for it."""
    pid = _prepared(client, DATA / "metabolomics_untargeted.csv", "responder")
    keys = {q["key"] for q in _asked(client, pid)}
    assert "missingness::sex" in keys
    assert "missingness_bulk::categorical" not in keys


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

def test_one_answer_writes_one_decision_and_one_sentence(client):
    pid = _prepared(client, DATA / "metabolomics_untargeted.csv", "responder")
    before = client.get(f"/project/{pid}").json()
    bulk = next(q for q in _asked(client, pid)
                if q["key"] == "missingness_bulk::numeric")

    project = api.STORE.get(pid)
    columns = [r["column"] for r in project.missingness_survey()
               if r["branch"] == "numeric"]
    assert len(columns) > 300

    r = client.post(f"/project/{pid}/decision", json={
        "kind": "route_missingness_bulk",
        "payload": {"branch": "numeric", "mechanism": MISS.NOT_INFORMATIVE,
                    "strategy": MISS.IMPUTE_MEDIAN, "columns": columns}})
    assert r.status_code == 200, r.text
    after = r.json()

    added = [d for d in after["decisions"]
             if d not in before["decisions"]
             and d["kind"] == "route_missingness_bulk"]
    assert len(added) == 1, "one answer must be one decision, not N"
    assert added[0]["payload"]["n_columns"] == len(columns)

    # THE SENTENCE A READER WANTS.
    assert added[0]["text"].startswith(
        f"Missing values in {len(columns):,} numeric column(s) will be filled")
    assert "within each training fold" in added[0]["text"]

    # The plan is still per column, because everything downstream reads it —
    # what changed is that the user answered once.
    assert len(after["missingness"]) == len(columns)
    assert all(d["bulk"] == "numeric" for d in after["missingness"])

    # ONE cascade entry. A cascade that fires 306 times for one answer trains
    # the user to ignore it.
    grew = (len(after["stale_downstream"])
            - len(before["stale_downstream"]))
    assert grew == 1, f"the cascade fired {grew} times for one answer"

    # And the question retires.
    assert "missingness_bulk::numeric" not in {q["key"] for q in _asked(client, pid)}


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


# ── the user edits the rule, not the members ─────────────────────────────────

def test_pulling_a_column_out_narrows_the_rule_and_the_sentence_says_so(client):
    pid = _prepared(client, DATA / "metabolomics_untargeted.csv", "responder")
    before = next(q for q in _asked(client, pid)
                  if q["key"] == "missingness_bulk::numeric")
    n_before = int(before["title"].split()[0].replace(",", ""))

    r = client.post(f"/project/{pid}/decision", json={
        "kind": "except_from_bulk", "payload": {"column": "mz_0022"}})
    assert r.status_code == 200, r.text

    keys = {q["key"] for q in _asked(client, pid)}
    after = next(q for q in _asked(client, pid)
                 if q["key"] == "missingness_bulk::numeric")
    n_after = int(after["title"].split()[0].replace(",", ""))
    assert n_after == n_before - 1
    assert "1 you pulled out" in after["why"]
    # And it rejoins the individually-asked columns.
    assert "missingness::mz_0022" in keys


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


def test_the_exceptions_are_one_question_and_not_n(client, tmp_path):
    """Otherwise this reintroduces the defect it exists to remove: 500
    exceptions asked one at a time is the unbounded interview arriving through
    the back door."""
    path = tmp_path / "exceptions.csv"
    rng = np.random.default_rng(3)
    n = 200
    y = rng.integers(0, 2, n)
    data = {"y": y}
    for i in range(40):
        col = rng.normal(size=n)
        col[(y == 1) & (rng.random(n) < 0.9)] = np.nan     # ALL informative
        data[f"lab_{i:02d}"] = col
    pd.DataFrame(data).to_csv(path, index=False)

    pid = _prepared(client, path, "y")
    project = api.STORE.get(pid)
    columns = [r["column"] for r in project.missingness_survey()]
    client.post(f"/project/{pid}/decision", json={
        "kind": "route_missingness_bulk",
        "payload": {"branch": "numeric", "mechanism": MISS.NOT_INFORMATIVE,
                    "strategy": MISS.IMPUTE_MEDIAN, "columns": columns}})

    exceptions = [q for q in _asked(client, pid)
                  if q["key"].startswith("missingness_exceptions::")]
    assert len(exceptions) == 1, (
        f"{len(exceptions)} exception questions; they must be a group too")
    assert "of those columns look like exceptions" in exceptions[0]["title"]
    assert "40" in exceptions[0]["title"] or "39" in exceptions[0]["title"]


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

def test_a_pack_settling_three_hundred_columns_renders_one_skip_not_three_hundred(client):
    """A rendered skip is still a rendered thing.

    Wiring the priors layer turned 306 questions into 306 SKIPS, which is a real
    improvement in what is being asked and no improvement at all in what is
    being drawn. `DESIGN_LANGUAGE.md` §09 wants skips to group *"so their
    density reads as machine work at a glance"*, and 306 of them is not a
    glance.
    """
    pid = _prepared(client, DATA / "metabolomics_untargeted.csv", "responder",
                    lens=("metabolomics",))
    iv = client.get(f"/project/{pid}/interview?step=preprocess").json()
    missingness = [q for q in iv["questions"] if q["kind"] == "missingness"]

    assert len(missingness) <= 4, (
        f"{len(missingness)} missingness entries rendered; the SKIP is scaling "
        f"with p even though the question is not")

    settled = [q for q in missingness if q["status"] == "skipped"]
    assert len(settled) == 1
    assert "settled by the" in settled[0]["title"]
    assert settled[0]["skip_reason"], "audit() refuses a skip with no reason"
    # It names the count, and it names the pack that made the claim.
    assert "300" in settled[0]["title"] or "306" in settled[0]["title"]
    assert "metabolomics" in settled[0]["skip_reason"].lower()


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

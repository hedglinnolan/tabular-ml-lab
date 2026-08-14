"""`DRIVE-040`, the half `L61` did not carry — three surfaces, named.

`L61-D1` closed this row on the figure caption, and the caption is right:
*"945 observations with 829 events of True."* The row is about **the encoded
value reaching a reader**, and after that fix it still did, at three places run
5 read on screen:

* the **PCA group annotation** — `<NA> 15,552, 1 5,527, 0 770`
* **Table 1's column headers** — `0 (n=770)` / `1 (n=5527)`
* the **event noticing card**, which flips from `False`/`True` to `0.0`/`1.0`
  the moment the answer is recorded

## What had to be built first, because the row assumed otherwise

The row said *"the name is available… verify that before building"*. Measured:
it was **half** available. `chosen_level_text` spells the level a caller asks
about, and the recorded decision carried `event_level` — the EVENT's name. All
three surfaces above render **both** levels, and the comparison level's name
was on the record nowhere: after the repair the live finding's own `spellings`
are recomputed against the encoded column to `{'0': '0', '1': '1'}`, so the
original words survived only inside the decision's prose sentence.

So `comparison_level_text` is new, `engine.record_fix` records both, and
`training.outcome_level_names` is the one reader. That is the *bigger finding*
the row anticipated, and it is small only because the sentence that already
spells both levels was one function away.

## What this file will not let happen

**Silence, never a guess.** Every project sealed before `L62` has no
`comparison_level` on its record, and every surface below must then render what
it rendered before — the encoded value. A renderer that filled the gap by
sorting, or by assuming `0` is the level that is not the event, would be
putting a word in a user's column that they never typed.
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbotab import api, engine, figure_bundle, manuscript   # noqa: E402
from turbotab import training as T                            # noqa: E402
from turbotab.project import AnalysisProject                  # noqa: E402

#: The event is the MAJORITY, as run 5's was, so a surface that happened to
#: render the more common level first would not pass by accident.
EVENT, COMPARISON = "True", "False"


def _frame(n: int = 320) -> pd.DataFrame:
    rng = np.random.default_rng(19)
    return pd.DataFrame({
        "age": rng.normal(50, 12, n).round(1),
        "bmi": rng.normal(27, 5, n).round(1),
        "meds": rng.choice([True, False], n, p=[0.85, 0.15]),
    })


def _answered(*, answer: bool = True) -> AnalysisProject:
    project = AnalysisProject.from_dataframe(_frame(), "p.csv")
    project.set_target("meds", "classification", "high", [])
    if answer:
        engine.record_fix(project, "positive_class__meds", choice="true")
    project.set_grain("one_row_per_person")
    project.set_eligibility("everyone")
    return project


@pytest.fixture(scope="module")
def client():
    return TestClient(api.app)


# ── the record, which is what the three surfaces read ───────────────────────

def test_the_record_keeps_both_level_names(client):
    """The half that was missing. One is not enough for a two-column header."""
    project = _answered()
    names = T.outcome_level_names(project)
    assert names == {1: EVENT, 0: COMPARISON}, names


def test_an_unanswered_project_names_nothing(client):
    """No decision, no names — and no guess."""
    assert T.outcome_level_names(_answered(answer=False)) == {}


def test_a_record_without_the_comparison_names_only_the_event():
    """**Every project sealed before `L62` is this project.**

    `engine.record_fix` recorded `event_level` from `L61` and
    `comparison_level` only from `L62`, so an older record has one of the two.
    The mapping must carry what it has and omit what it does not, because a
    partial answer is still an answer and a filled-in gap is a fabrication.
    """
    project = _answered()
    decision = T.event_decision(project)
    decision.payload.pop("comparison_level")
    assert T.outcome_level_names(project) == {1: EVENT}


# ── surface 1 · Table 1's column headers ────────────────────────────────────

def test_table_one_headers_name_the_levels(client):
    project = _answered()
    table, _ = manuscript.table_one(project)
    headers = [str(c) for c in table.columns]
    strata = [h for h in headers if re.fullmatch(r".+ \(n=\d+\)", h)]
    assert strata, headers
    for header in strata:
        level = header.split(" (n=")[0]
        assert level in (EVENT, COMPARISON), (
            f"Table 1 still labels a stratum {level!r}; the user answered "
            f"{EVENT!r} against {COMPARISON!r}")


def test_table_one_headers_keep_the_encoded_value_when_the_record_cannot_say(client):
    """The silent branch, on the same surface."""
    project = _answered(answer=False)
    table, _ = manuscript.table_one(project)
    strata = [str(c) for c in table.columns
              if re.fullmatch(r".+ \(n=\d+\)", str(c))]
    assert strata, list(table.columns)
    assert all(h.split(" (n=")[0] in ("True", "False") for h in strata), strata


def test_the_overall_column_is_not_renamed(client):
    """**The anchor matters.** A loose substring rewrite would rename
    `Overall (N=320)` the moment somebody's level is called `Overall`; this
    matches the stratum header's exact shape, and `Overall` uses `N=`."""
    project = _answered()
    table, _ = manuscript.table_one(project)
    assert any(str(c).startswith("Overall (N=") for c in table.columns), (
        list(table.columns))


# ── surface 2 · the PCA group annotation ────────────────────────────────────

def _pca(project):
    return figure_bundle._pca_payload(project)


def test_the_pca_group_counts_name_the_levels(client):
    project = _answered()
    counts = _pca(project)["group_counts"]
    assert set(counts) <= {EVENT, COMPARISON, "ungrouped"}, (
        f"the PCA legend names its groups {sorted(counts)}, which is the "
        f"encoding rather than the levels the user chose")
    assert EVENT in counts and COMPARISON in counts, counts


def test_the_pca_group_counts_fall_back_to_the_value(client):
    project = _answered(answer=False)
    counts = _pca(project)["group_counts"]
    assert EVENT in counts or "True" in counts, counts
    assert not {"0", "1"} & set(counts), (
        f"an unanswered project rendered encoded group names: {counts}")


# ── surface 3 · the event noticing card ─────────────────────────────────────

def test_the_event_card_offers_the_words_the_user_typed(client):
    """Before AND after the encode, because the flip is the finding."""
    pid = client.post("/project", files={
        "file": ("p.csv", _frame().to_csv(index=False).encode(), "text/csv")}
    ).json()["id"]
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_target", "payload": {"column": "meds"}})

    before = client.get(
        f"/project/{pid}/finding/positive_class__meds/preview").json()
    assert set(before["choices"].values()) == {EVENT, COMPARISON}, before

    client.post(f"/project/{pid}/decision",
                json={"kind": "apply", "subject": "positive_class__meds",
                      "payload": {"choice": "true"}})
    after = client.get(
        f"/project/{pid}/finding/positive_class__meds/preview").json()
    assert set(after["choices"].values()) == {EVENT, COMPARISON}, (
        f"after the encode the card offers {after['choices']}, which is the "
        f"encoding rather than anything in the user's column")
    # THE KEYS ARE UNTOUCHED. Only what a reader sees changed; the tokens the
    # route accepts are still the tokens the finding produced, so a click still
    # posts something the apply branch understands.
    assert set(after["choices"]) == {"0", "1"}, after["choices"]


# ── the class, not only the instance ────────────────────────────────────────

def test_every_surface_that_renders_the_outcome_reads_one_mapping():
    """`MISC-019`'s shape, guarded rather than described.

    The row closed at `L61` on one surface of four. What makes that
    recurrable is three renderers each deciding for themselves how to spell a
    level. They read `training.outcome_level_names`, and this asserts the
    count — so a fourth surface added without it is a fourth surface this test
    does not cover, and the number here is the honest statement of coverage.
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parent
    readers = sorted(
        p.name for p in list(root.glob("*.py")) + list(root.parent.glob("ml/*.py"))
        if not p.name.startswith("test_")
        and "outcome_level_names" in p.read_text(encoding="utf-8"))
    assert readers == ["api.py", "figure_bundle.py", "manuscript.py",
                       "training.py"], (
        f"the surfaces reading the recorded level names are {readers}; this "
        f"test covers Table 1, the PCA annotation and the event card. A new "
        f"one is not wrong — but it is uncovered until it is named here.")

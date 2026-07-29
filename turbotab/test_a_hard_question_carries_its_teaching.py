"""Layer 3, and the check that a question cannot ship without its teaching.

Two rulings. **The app keeps asking hard questions and invests in making them
answerable** — deciding more on the user's behalf was considered and rejected,
because answering these questions is itself the educational moment. And **layer 3
is the preview mechanic pointed at interview questions**, not a text panel:
teaching means showing consequences, and consequences are computable.

So every assertion here is about a **number computed from the fixture**, never
about a sentence. The one exception is the refusal, which is content.

Run:  ./venv/bin/python -m pytest \\
          turbotab/test_a_hard_question_carries_its_teaching.py -q
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbotab import api, teaching as T                               # noqa: E402

DATA = Path(__file__).resolve().parent / "sample_data"


@pytest.fixture
def client():
    return TestClient(api.app)


def _driven(client, fixture: str, target: str) -> str:
    with open(DATA / f"{fixture}.csv", "rb") as fh:
        pid = client.post("/project", files={
            "file": (f"{fixture}.csv", fh, "text/csv")}).json()["id"]
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_target", "payload": {"column": target}})
    return pid


# ── the check that makes teaching non-optional ───────────────────────────────

@pytest.mark.parametrize("question", T.TAUGHT)
def test_every_hard_question_carries_its_teaching(question, client):
    """A question with no layer-3 content fails. A new interview question cannot
    ship without its teaching.

    All four sub-questions, because three of four is a panel that answers the
    easy ones — and the fourth, the refusal, is the one a teaching panel is most
    tempted to skip.
    """
    pid = _driven(client, "dietary_recalls", "hba1c")
    panel = client.get(f"/project/{pid}/teaching/{question}").json()

    assert panel["title"], f"{question} has no layer-3 title"
    assert panel["consequences"], (
        f"{question} states no consequence — layer 3 is the preview mechanic "
        f"pointed at a question, and a panel with no computed consequence is a "
        f"text panel")
    for entry in panel["consequences"]:
        assert entry["answer"] and entry["headline"] and entry["detail"]
    assert panel["worked_example"], f"{question} has no worked example"
    assert panel["comparability"], f"{question} does not say if repeats compare"
    assert panel["cannot_answer"] == T.CANNOT_ANSWER


def test_a_question_with_no_panel_is_refused_rather_than_answered_emptily(client):
    pid = _driven(client, "dietary_recalls", "hba1c")
    r = client.get(f"/project/{pid}/teaching/state_eligibility")
    assert r.status_code == 404
    assert "No teaching panel" in r.json()["detail"]


# ── 1 · what each answer does to my data ─────────────────────────────────────

def test_the_grain_panel_computes_both_splits(client):
    """The consequence of the grain answer IS the split, computed both ways so
    the difference is a number rather than a warning."""
    pid = _driven(client, "dietary_recalls", "hba1c")
    panel = client.get(f"/project/{pid}/teaching/state_grain").json()
    by_answer = {c["answer"]: c for c in panel["consequences"]}

    random_split = by_answer["one_row_per_person"]
    assert random_split["n_rows"] == 600
    assert random_split["n_held_out"] == 90
    assert random_split["leaks"] is True, (
        "600 rows over 300 people is two rows each; a random split leaks and the "
        "panel must say so with the numbers")

    grouped = by_answer["people_repeat"]
    assert grouped["n_groups"] == 300
    assert grouped["n_held_out_groups"] == 45
    assert grouped["leaks"] is False


def test_the_unit_panel_computes_the_real_row_change(client):
    """*"600 rows become 300, one per `participant_id`"* — not "aggregation
    reduces rows"."""
    pid = _driven(client, "dietary_recalls", "hba1c")
    panel = client.get(f"/project/{pid}/teaching/state_unit_of_analysis").json()
    by_answer = {c["answer"]: c for c in panel["consequences"]}

    assert by_answer["person"]["n_rows_before"] == 600
    assert by_answer["person"]["n_rows_after"] == 300
    assert "600 rows become 300" in by_answer["person"]["headline"]
    assert "participant_id" in by_answer["person"]["headline"]

    assert by_answer["record"]["n_rows_after"] == 600
    assert by_answer["record"]["loses"] == []
    # And it names what the person answer would cost, by column.
    assert by_answer["person"]["loses"], (
        "the person answer takes the first value of every varying non-numeric "
        "column and the panel must name them")


def test_the_aggregation_panel_computes_all_four_outcomes(client):
    """*"The mean"* and *"the last"* become two numbers the user compares, rather
    than two words."""
    pid = _driven(client, "clinical_longitudinal", "progressed")
    panel = client.get(f"/project/{pid}/teaching/state_aggregation").json()
    by_answer = {c["answer"]: c for c in panel["consequences"]}
    assert set(by_answer) == {"mean", "first", "last", "change_from_baseline"}

    # The four are consistent with each other, which is checkable arithmetic
    # rather than four independent claims.
    assert by_answer["change_from_baseline"]["value"] == pytest.approx(
        by_answer["last"]["value"] - by_answer["first"]["value"])
    for entry in by_answer.values():
        assert isinstance(entry["value"], (int, float))


# ── 2 · a worked example on their own rows ───────────────────────────────────

def test_the_worked_example_quotes_real_rows_and_real_values(client):
    """Not the concept in general. The subject, the row labels and the values are
    checked against the frame, so the example cannot drift from the table."""
    pid = _driven(client, "dietary_recalls", "hba1c")
    panel = client.get(f"/project/{pid}/teaching/state_unit_of_analysis").json()
    example = panel["worked_example"]

    df = pd.read_csv(DATA / "dietary_recalls.csv")
    block = df[df["participant_id"] == example["subject"]]
    assert len(block) == 2, "the example names a subject that does not repeat"
    assert list(block.index[:2]) == example["rows"][:2]

    shown = example["shown_column"]
    assert list(block[shown].round(4)) == pytest.approx(example["values"])
    assert example["combined"] == pytest.approx(float(block[shown].mean()), abs=1e-3)
    assert example["subject"] in example["sentence"]
    assert shown in example["sentence"]


def test_the_worked_example_never_uses_the_target(client):
    """Correctness, not taste. `change_from_baseline` deliberately does NOT
    difference the target, so an example showing the outcome being averaged
    teaches the one case the operation treats differently.

    **The frame is built so the target WOULD be picked**, which the probe forced.
    Asserting this on the fixtures came back green with the exclusion removed —
    their ranking happens to prefer another column anyway, so the test proved
    nothing about the guard. Here `outcome` has by far the most within-person
    variation relative to its own spread, so it wins the ranking outright and
    only the exclusion keeps it out.
    """
    import numpy as np

    rng = np.random.default_rng(4)
    n = 120
    frame = pd.DataFrame({
        "pid": [f"S{i // 3:03d}" for i in range(n)],
        # Nearly constant within a person: aggregation barely touches these.
        "age": np.repeat(rng.integers(30, 80, n // 3), 3),
        "height_cm": np.repeat(rng.normal(170, 8, n // 3), 3),
        # And the target, which flips every row.
        "outcome": [i % 2 for i in range(n)],
    })

    df = frame
    ranked = T._column_to_show(df, "pid", ["age", "height_cm", "outcome"], None)
    assert ranked == "outcome", (
        "the frame no longer makes the target the top-ranked column, so this "
        "test cannot prove the exclusion does anything")

    assert T._column_to_show(df, "pid", ["age", "height_cm", "outcome"],
                             "outcome") != "outcome"

    # And end to end, on the fixtures, where it must also hold.
    for fixture, target in (("clinical_longitudinal", "progressed"),
                            ("dietary_recalls", "hba1c")):
        pid = _driven(client, fixture, target)
        for question in ("state_unit_of_analysis", "state_aggregation"):
            panel = client.get(f"/project/{pid}/teaching/{question}").json()
            example = panel.get("worked_example") or {}
            assert example.get("shown_column") != target, (
                f"{question} on {fixture} shows the target being combined")


def test_the_worked_example_never_uses_a_bookkeeping_column(client):
    """`recall_number` and `visit` are replicate indexes: the one kind of column
    whose variation carries no information, so an example on one teaches about
    bookkeeping. `recall_date` and `visit_date` order the rows and are not
    measurements either."""
    for fixture, target, banned in (
            ("dietary_recalls", "hba1c", {"recall_number", "recall_date"}),
            ("clinical_longitudinal", "progressed", {"visit", "visit_date"})):
        pid = _driven(client, fixture, target)
        panel = client.get(f"/project/{pid}/teaching/state_aggregation").json()
        assert panel["worked_example"]["shown_column"] not in banned


# ── 3 · are my repeats comparable ────────────────────────────────────────────

def test_comparability_is_measured_and_not_reassured(client):
    """Spacing regularity and which columns actually differ. The dietary and
    clinical fixtures are the pair, and they must come out opposite."""
    dietary = client.get(
        f"/project/{_driven(client, 'dietary_recalls', 'hba1c')}"
        f"/teaching/state_grain").json()["comparability"]
    clinical = client.get(
        f"/project/{_driven(client, 'clinical_longitudinal', 'progressed')}"
        f"/teaching/state_grain").json()["comparability"]

    assert dietary["verdict"] == "repeats"
    assert clinical["verdict"] == "time_points"
    assert dietary["spacing"]["cv"] > 0.3
    assert clinical["spacing"]["cv"] < 0.15
    assert "schedule" in clinical["detail"]

    # And it names which columns differ within a person, because "comparable" is
    # a measurement about columns and not a reassurance about the study.
    assert dietary["n_varying"] > 0
    assert dietary["varying_columns"]
    assert "participant_id" not in dietary["varying_columns"]


def test_a_table_where_nothing_repeats_says_so(client):
    pid = _driven(client, "metabolomics_untargeted", "responder")
    panel = client.get(f"/project/{pid}/teaching/state_grain").json()
    assert panel["comparability"]["verdict"] == "nothing_repeats"
    assert panel["consequences"][0]["n_rows"] == 80


# ── 4 · which should I pick ──────────────────────────────────────────────────

def test_the_refusal_is_content_and_not_a_gap():
    """The fourth sub-question, and the one a teaching panel is most tempted to
    answer. Answering it would be the app deciding on the user's behalf, which
    is the thing ruling 1 rejected — so the refusal is stated plainly and it is
    the same sentence at every question, because the reason is the same one.
    """
    assert "the app will not guess at it" in T.CANNOT_ANSWER
    assert "a fact about the study rather than about the table" in T.CANNOT_ANSWER
    # It says WHY, not merely that it will not. A bare refusal is a shrug.
    assert "leak" in T.CANNOT_ANSWER

    df = pd.read_csv(DATA / "dietary_recalls.csv")
    for question in T.TAUGHT:
        panel = T.panel(question, df)
        assert panel["cannot_answer"] == T.CANNOT_ANSWER


# ── nothing here parses or generates language ────────────────────────────────

def test_layer_three_computes_and_never_generates_prose_about_a_concept():
    """There is no LLM and none is coming, so this is checkable: every sentence
    the panel emits is a template filled with values from the frame, and every
    consequence carries the numbers it is about.
    """
    df = pd.read_csv(DATA / "dietary_recalls.csv")
    for question in T.TAUGHT:
        panel = T.panel(question, df)
        for entry in panel["consequences"]:
            has_number = any(
                isinstance(v, (int, float)) and not isinstance(v, bool)
                for k, v in entry.items() if k not in ("answer",))
            assert has_number, (
                f"{question}/{entry['answer']} states a consequence with no "
                f"computed value in it, which is prose about a concept")
        example = panel.get("worked_example")
        if example:
            # The example names the frame's own objects, so it cannot be
            # generic: the subject and a column both appear in the sentence.
            assert example["subject"] in example["sentence"]
            assert example["column"] in df.columns


# ── the page offers exactly the teaching the server has ──────────────────────

def test_the_page_and_the_server_agree_on_which_questions_are_taught():
    """A page offering teaching the server does not have is a button that 404s;
    a server carrying teaching the page never opens is `DRIVE-001` again."""
    import re
    page = (Path(__file__).resolve().parents[1] / "turbotab" / "web"
            / "index.html").read_text()
    start = page.index("var TAUGHT = [")
    declared = re.findall(r'"([^"]+)"', page[start:page.index("];", start)])
    assert declared == list(T.TAUGHT), (
        f"the page teaches {declared} and the server teaches {list(T.TAUGHT)}")


def test_the_disclosure_toggles():
    """`DRIVE-004`. The driver clicked "Show me what this means" a second time
    and nothing happened, which teaches that the control is broken rather than
    that it is already open."""
    page = (Path(__file__).resolve().parents[1] / "turbotab" / "web"
            / "index.html").read_text()
    start = page.index('var teach = ev.target.closest("[data-teach]");')
    block = page[start:start + 1400]
    assert 'aria-expanded' in block, "no open/closed state to toggle against"
    assert "Hide that" in block, "the label never says the second press closes"
    assert "classList.add(\"is-hidden\")" in block, "nothing closes it"

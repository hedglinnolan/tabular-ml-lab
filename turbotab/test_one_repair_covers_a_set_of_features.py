"""`DRIVE-002` — nine features, one idea, nine show-me-then-apply cycles.

> Nine NHANES features are binary written as text. The engine found all nine and
> the driver had to open and apply each individually. Show what it means for
> one, then let the user select which features to run it on, then apply to the
> selected set.

This is `turbotab/bulk.py`'s rule-scope pointed at repairs rather than at
questions: **operations apply to sets defined by a rule.** `bulk.py` was built
for the missingness question, where 308 columns with blanks produced 308
questions. A repair is the same shape one object over.

## What this file asserts, and the one that matters most

The load-bearing assertion is **the frame**, not the record. A bulk apply that
wrote one satisfying sentence into the transcript and repaired one column would
satisfy every plausible test of the receipt — and that is the exact shape of the
critical this project closed last loop, where nine tests read a seal's record
and none read the draw. So the columns are checked in the dataframe the project
holds afterwards, and the ones left out are checked to be **unchanged**.

The second is that **the record and the interview agree about the declined
members**. The first implementation of this recorded *"1 other in the same group
was deliberately left as recorded"* and then re-asked that member on its own
card. Both statements were rendered, in the same session, to the same user. A
sentence in the transcript that the interview contradicts is worse than no
sentence, because it is the app asserting something the app itself does not
believe.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml import router                                                 # noqa: E402
from turbotab import engine, repairs as R                             # noqa: E402

DATA = Path(__file__).resolve().parent / "sample_data"


def _client():
    from fastapi.testclient import TestClient

    from turbotab import api
    return TestClient(api.app)


def _upload(client, name):
    with open(DATA / f"{name}.csv", "rb") as fh:
        return client.post("/project", files={
            "file": (f"{name}.csv", fh, "text/csv")}).json()["id"]


def _pushed(client, pid, step="data"):
    return [q for q in client.get(
        f"/project/{pid}/interview?step={step}").json()["questions"]
        if q["mode"] == "push"]


def _frame(pid):
    from turbotab.api import STORE
    return STORE.get(pid).df


# ── the grouping ─────────────────────────────────────────────────────────────

def test_nine_features_one_repair_becomes_one_question():
    """The defect, as the count it produced."""
    client = _client()
    pid = _upload(client, "metabolomics_untargeted")
    keys = [q["key"] for q in _pushed(client, pid)]
    binary = [k for k in keys if k.startswith("repair::binary_text__")]
    groups = [k for k in keys if k.startswith("repair_bulk::")]

    assert not binary, (
        f"{len(binary)} binary-text repairs are still asked one at a time: "
        f"{binary}")
    assert "repair_bulk::read_as_binary" in groups
    group = client.get(f"/project/{pid}/repair_group/read_as_binary").json()
    assert group["n"] >= 3, group["n"]
    assert set(group["columns"]) == {"batch", "sample_type", "sex"}


def test_a_group_of_one_is_not_a_group():
    """`bulk.MIN_GROUP`'s argument, and the same number for the same reason.

    A rule over one column is a column with extra words in front of it, and a
    bulk affordance offered over a single leftover is worse than asking.
    """
    findings = [{"id": "a", "fix_kind": "coerce_numeric", "fix_label": "x",
                 "affected_columns": ["one"]}]
    assert R.group(findings) == []
    assert len(R.group(findings + [dict(findings[0], id="b",
                                        affected_columns=["two"])])) == 1


@pytest.mark.parametrize("kind", sorted(R.NEVER_GROUPED))
def test_the_repairs_that_never_group_say_why(kind):
    """A name on the exclusion list is a claim that bulk would be WRONG, not
    merely awkward — so each carries its reason, and none is a bare entry."""
    findings = [{"id": f"{kind}__{i}", "fix_kind": kind, "fix_label": "x",
                 "affected_columns": [f"c{i}"]} for i in range(3)]
    assert R.group(findings) == [], f"{kind} grouped and must not"
    assert len(R.NEVER_GROUPED[kind]) > 60, (
        f"{kind}'s exclusion has no argument behind it")


def test_a_grouped_finding_is_not_also_asked_on_its_own():
    """Two controls applying one repair is worse than a duplicate card: the
    second runs against a table the first replaced."""
    client = _client()
    pid = _upload(client, "clinic_visits")
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_target", "payload": {"column": "outcome"}})
    plan = _pushed(client, pid)
    covered = {c for q in plan for c in (q.get("covers") or [])}
    assert covered, "no group covers anything on this fixture"
    alone = {q["key"][len("repair::"):] for q in plan
             if q["key"].startswith("repair::")}
    assert not (covered & alone), (
        f"asked twice: {sorted(covered & alone)}")


# ── the effect, read back off the frame ──────────────────────────────────────

def test_applying_to_a_selected_set_changes_those_columns_and_no_others():
    """**The load-bearing assertion.** Read off the dataframe, not the receipt.

    A bulk apply that wrote a satisfying sentence and repaired one column would
    pass any test of the record. So: the selected columns are numeric
    afterwards, and the one left out is still text.
    """
    client = _client()
    pid = _upload(client, "metabolomics_untargeted")
    group = client.get(f"/project/{pid}/repair_group/read_as_binary").json()
    members = group["members"]
    assert len(members) >= 3

    before = _frame(pid)
    chosen = members[:2]
    left_out = members[2]
    chosen_cols = [c for m in chosen for c in m["columns"]]
    left_cols = left_out["columns"]
    for col in chosen_cols + left_cols:
        assert not pd.api.types.is_numeric_dtype(before[col]), (
            f"{col} is already numeric, so this test cannot see the repair")

    r = client.post(f"/project/{pid}/decision", json={
        "kind": "apply_bulk", "subject": "read_as_binary",
        "payload": {"findings": [m["id"] for m in chosen]}})
    assert r.status_code == 200, r.text

    after = _frame(pid)
    for col in chosen_cols:
        assert pd.api.types.is_numeric_dtype(after[col]), (
            f"{col} was selected and was not repaired — the record may say it "
            f"was, which is worse than the repair simply not running")
    for col in left_cols:
        assert not pd.api.types.is_numeric_dtype(after[col]), (
            f"{col} was NOT selected and was repaired anyway; the selection "
            f"is decoration")
    assert len(after) == len(before), "a binary read changed the row count"


def test_it_is_one_decision_and_not_n():
    """*"One decision covering N features"* is the finding's own words.

    N `apply` rows beside one `apply_bulk` row would make the transcript say
    the work happened twice, and the methods section is the product.
    """
    client = _client()
    pid = _upload(client, "metabolomics_untargeted")
    group = client.get(f"/project/{pid}/repair_group/read_as_binary").json()
    client.post(f"/project/{pid}/decision", json={
        "kind": "apply_bulk", "subject": "read_as_binary",
        "payload": {"findings": [m["id"] for m in group["members"]]}})

    decisions = client.get(f"/project/{pid}").json()["decisions"]
    kinds = [d["kind"] for d in decisions]
    assert kinds.count("apply_bulk") == 1
    assert "apply" not in kinds, (
        "the members were recorded individually as well, so the transcript "
        "says the work happened twice")

    said = next(d for d in decisions if d["kind"] == "apply_bulk")
    for col in group["columns"]:
        assert f"`{col}`" in said["text"], (
            f"the sentence does not name {col}, so a reader of the methods "
            f"section cannot check it")


def test_the_record_and_the_interview_agree_about_what_was_left_alone():
    """The disagreement the first implementation shipped with.

    It recorded *"1 other was deliberately left as recorded"* and then re-asked
    that member on its own card. Both were rendered to the same user in the
    same session. A sentence the interview contradicts is worse than no
    sentence — it is the app asserting something the app does not believe.
    """
    client = _client()
    pid = _upload(client, "metabolomics_untargeted")
    group = client.get(f"/project/{pid}/repair_group/read_as_binary").json()
    left_out = group["members"][-1]
    client.post(f"/project/{pid}/decision", json={
        "kind": "apply_bulk", "subject": "read_as_binary",
        "payload": {"findings": [m["id"] for m in group["members"][:-1]]}})

    said = next(d for d in client.get(f"/project/{pid}").json()["decisions"]
                if d["kind"] == "apply_bulk")
    assert "deliberately left as recorded" in said["text"]
    for col in left_out["columns"]:
        assert f"`{col}`" in said["text"], (
            "the sentence counts what was left out without naming it, so the "
            "claim cannot be checked")

    keys = [q["key"] for q in _pushed(client, pid)]
    assert f"repair::{left_out['id']}" not in keys, (
        "the transcript says this was deliberately left alone and the "
        "interview is asking about it again")

    # And reopening it is still free — a decision, not a door closing.
    client.post(f"/project/{pid}/decision",
                json={"kind": "undismiss", "subject": left_out["id"],
                      "payload": {}})
    assert f"repair::{left_out['id']}" in [q["key"] for q in _pushed(client, pid)]


def test_declining_the_whole_group_is_an_answer_with_a_sentence():
    """§09's recorded-absence rule, on a repair.

    A group the user considered and declined must not read like a group nobody
    reached.
    """
    client = _client()
    pid = _upload(client, "metabolomics_untargeted")
    group = client.get(f"/project/{pid}/repair_group/read_as_binary").json()
    before = _frame(pid).copy()

    r = client.post(f"/project/{pid}/decision", json={
        "kind": "decline_bulk", "subject": "read_as_binary", "payload": {}})
    assert r.status_code == 200, r.text

    said = next(d for d in client.get(f"/project/{pid}").json()["decisions"]
                if d["kind"] == "decline_bulk")
    assert "deliberately left as recorded" in said["text"]
    for col in group["columns"]:
        assert f"`{col}`" in said["text"]
    assert "repair_bulk::read_as_binary" not in [
        q["key"] for q in _pushed(client, pid)]
    # Nothing moved.
    pd.testing.assert_frame_equal(_frame(pid), before)


def test_applying_to_nothing_is_refused_rather_than_recorded_as_a_repair():
    """A repair over zero columns is not a repair, and *"leave them all"* has
    its own kind so the record says which happened in its own words."""
    client = _client()
    pid = _upload(client, "metabolomics_untargeted")
    r = client.post(f"/project/{pid}/decision", json={
        "kind": "apply_bulk", "subject": "read_as_binary",
        "payload": {"findings": []}})
    assert r.status_code == 400
    assert "at least one" in r.text


def test_the_group_is_named_by_its_operation_and_not_by_one_columns_button():
    """The engine's `fix_label` is an imperative for one button on one column —
    *"Read 'batch' as binary (B2 = 1, B1 = 0)"*. As a card title over nine
    columns it is a sentence that is wrong about eight of them."""
    client = _client()
    pid = _upload(client, "metabolomics_untargeted")
    group = client.get(f"/project/{pid}/repair_group/read_as_binary").json()
    assert group["label"] == "read as binary"
    assert "batch" not in group["label"]
    # Each member keeps its own precise label.
    assert any("batch" in m["title"] for m in group["members"])


def test_the_worked_example_is_a_real_preview_of_a_real_column():
    """One worked example, because nine before/after tables are unreadable —
    and it is the FIRST member's, so the example on screen is the finding the
    card cites."""
    client = _client()
    pid = _upload(client, "metabolomics_untargeted")
    group = client.get(f"/project/{pid}/repair_group/read_as_binary").json()
    example = group["example"]
    assert example and example["applicable"] is True
    assert example["finding_id"] == group["members"][0]["id"]
    assert example["sample"]["rows"], "the worked example has no rows to show"
    assert group["example_error"] is None


def test_the_group_cites_a_finding_so_it_counts_as_findings_driven():
    """*"Push the notable"* is only true if the question says what it is
    pushing. A grouped question that cited nothing scored as a question that
    exists because a pipeline stage exists."""
    df = pd.read_csv(DATA / "clinic_visits.csv")
    findings = engine.rank_findings(
        engine.diagnose(df, target="outcome"),
        engine.profile(df, "outcome", None), lens=[], df=df)
    plan = router.plan(findings, target="outcome", step="data",
                       detection={"detected": "classification",
                                  "confidence": "high", "reasons": []})
    groups = [q for q in plan if q.key.startswith("repair_bulk::")]
    assert groups, "no groups on this fixture"
    for q in groups:
        assert q.is_findings_driven, f"{q.key} cites no finding"
        assert q.triggering_finding in q.covers


def test_a_deferred_group_comes_back_at_the_step_it_names():
    """Deferral is a first-class disposition only if it comes back.

    The first implementation set `status = "deferred"` whenever the key was in
    `deferred`, regardless of the step being planned — so a group deferred to
    Explore stayed deferred AT Explore and never resurfaced. Deferral would
    have been a discard with manners.
    """
    df = pd.read_csv(DATA / "clinic_visits.csv")
    findings = engine.rank_findings(
        engine.diagnose(df, target="outcome"),
        engine.profile(df, "outcome", None), lens=[], df=df)
    detection = {"detected": "classification", "confidence": "high",
                 "reasons": []}
    first = router.plan(findings, target="outcome", detection=detection,
                        step="data")
    key = next(q.key for q in first if q.key.startswith("repair_bulk::"))

    at_data = router.plan(findings, target="outcome", detection=detection,
                          step="data", deferred={key: "explore"})
    moved = next(q for q in at_data if q.key == key)
    assert moved.status == "deferred" and moved.defer_target == "explore"

    at_explore = router.plan(findings, target="outcome", detection=detection,
                             step="explore", deferred={key: "explore"},
                             answered=["choose_target"])
    back = next((q for q in at_explore if q.key == key), None)
    assert back is not None, "a deferred group never resurfaced"
    assert back.status == "asked", (
        "the group came back still deferred, so deferral is a discard with "
        "manners")
    assert back.deferred_from == "data"

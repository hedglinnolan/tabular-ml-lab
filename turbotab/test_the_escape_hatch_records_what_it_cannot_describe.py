"""`My design isn't described here` — the answer space is closed, the world is not.

A nested case-control with matched pairs has no correct answer among *no*, *yes
with this column* and *not sure*. Its rows are neither independent participants
nor repeated measures of one person; they are **matched sets**, and a split has
to keep a set together for a reason none of the three describes. A crossover
trial is the same shape from another direction.

Forcing one of the three produces exactly the confidently-wrong answer the
constitution forbids — and it produces it **in the record**, where a reader takes
it as a description of the study.

Same shape as *"I'm not sure"*, and for the same reason: **uncertainty must never
cost more than a wrong confident answer.**

Two independent hatches, because the two questions are independent: the design
may be undescribable while the unit is clear, or the reverse.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbotab import api, grain as G, repeats as R                     # noqa: E402

DATA = Path(__file__).resolve().parent / "sample_data"


@pytest.fixture
def client():
    return TestClient(api.app)


def _upload(client) -> str:
    with open(DATA / "clinical_longitudinal.csv", "rb") as fh:
        pid = client.post("/project", files={
            "file": ("clinical_longitudinal.csv", fh, "text/csv")}).json()["id"]
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_target", "payload": {"column": "progressed"}})
    return pid


def _decide(client, pid, kind, payload):
    return client.post(f"/project/{pid}/decision",
                       json={"kind": kind, "payload": payload})


# ── it is offered at both questions ──────────────────────────────────────────

def test_both_questions_offer_it_and_the_page_can_submit_it():
    from ml import router
    plan = router.plan([], target="y", detection=None, step="data", deferred={},
                       answered=["state_lens", "choose_target"],
                       recommendations=[], signals=None, missing_columns=[])
    grain = next(q.to_dict() for q in plan if q.key == "state_grain")
    assert "My design isn't described here" in grain["options"]
    assert G.DESIGN_NOT_DESCRIBED in grain["option_values"]

    plan = router.plan([], target="y", detection=None, step="data", deferred={},
                       answered=["state_lens", "choose_target", "state_grain",
                                 "state_repeat_kind"],
                       recommendations=[], signals=None, missing_columns=[],
                       repeats={"reading": R.REPEATS, "sentence": "…",
                                "confidence": "medium", "kind": R.REPEATS,
                                "unit": None, "menu": None})
    unit = next(q.to_dict() for q in plan if q.key == "state_unit_of_analysis")
    assert "My design isn't described here" in unit["options"]
    assert R.UNIT_NOT_DESCRIBED in unit["option_values"]


# ── what it records, and the treatment it routes to ──────────────────────────

def test_the_grain_hatch_routes_to_the_conservative_split(client):
    """Grouped where a column was named, undetermined otherwise — never
    `cross_sectional`, which would be a claim about independence the user
    explicitly declined to make."""
    assert G.seal_basis(G.DESIGN_NOT_DESCRIBED, "subject_id", 200) == "grouped"
    assert G.seal_basis(G.DESIGN_NOT_DESCRIBED, None) == "undetermined"
    assert G.seal_basis(G.DESIGN_NOT_DESCRIBED, "subject_id", 2) == \
        "repetition_found_grouping_abandoned"

    pid = _upload(client)
    _decide(client, pid, "set_grain",
            {"answer": G.DESIGN_NOT_DESCRIBED, "group_col": "subject_id"})
    _decide(client, pid, "set_eligibility", {"answer": "everyone"})
    assert _decide(client, pid, "seal",
                   {"fraction": 0.15, "seed": 42}).status_code == 200

    body = client.get(f"/project/{pid}").json()
    assert body["lockbox"]["seal_basis"] == "grouped"
    assert body["grain"]["design_not_described"] is True
    assert body["n_rows"] == 600, "nothing was combined"


def test_the_numbers_are_exploratory_even_though_the_basis_is_grouped(client):
    """The flag cannot be derived from the basis, and that is the subtle part.

    With a grouping column named, `grouped` is genuinely the most conservative
    basis available and is NOT one of the two bases `is_exploratory_basis`
    covers. The basis is honest; the app still cannot vouch that grouping is the
    right treatment for a design it was never told. So the flag rides on the
    ANSWER, where the reason lives.
    """
    assert G.is_exploratory_basis("grouped") is False

    pid = _upload(client)
    _decide(client, pid, "set_grain",
            {"answer": G.DESIGN_NOT_DESCRIBED, "group_col": "subject_id"})
    _decide(client, pid, "set_eligibility", {"answer": "everyone"})
    _decide(client, pid, "seal", {"fraction": 0.15, "seed": 42})

    disclosures = client.get(f"/project/{pid}").json()["disclosures"]
    assert disclosures["exploratory"] is True
    assert "most conservative treatment available" in disclosures["seal"]
    assert R.AUTHOR_REQUIRED in disclosures["seal"]


def test_the_chain_does_not_fire_under_the_grain_hatch(client):
    """There is nothing left to ask: the answer already routes to rows-survive
    and no-aggregation, so asking how to combine them would be asking about a
    treatment that was declined."""
    pid = _upload(client)
    _decide(client, pid, "set_grain",
            {"answer": G.DESIGN_NOT_DESCRIBED, "group_col": "subject_id"})
    for kind, payload in (("set_repeat_kind", {"kind": "time_points"}),
                          ("set_unit_of_analysis", {"unit": "person"}),
                          ("set_aggregation", {"method": "mean"})):
        assert _decide(client, pid, kind, payload).status_code == 400

    # And the seal is not blocked on a chain that does not apply.
    _decide(client, pid, "set_eligibility", {"answer": "everyone"})
    assert _decide(client, pid, "seal", {"fraction": 0.15,
                                         "seed": 42}).status_code == 200


def test_the_unit_hatch_leaves_the_rows_alone_and_refuses_aggregation(client):
    """The second hatch, for the case where the grain is clear and the UNIT is a
    matched set — which the app has no aggregation for."""
    pid = _upload(client)
    _decide(client, pid, "set_grain",
            {"answer": "people_repeat", "group_col": "subject_id"})
    _decide(client, pid, "set_repeat_kind", {"kind": "time_points"})
    assert _decide(client, pid, "set_unit_of_analysis",
                   {"unit": R.UNIT_NOT_DESCRIBED}).status_code == 200

    refused = _decide(client, pid, "set_aggregation", {"method": "mean"})
    assert refused.status_code == 400
    assert "no aggregation for a matched set" in refused.json()["detail"]

    body = client.get(f"/project/{pid}").json()
    assert body["n_rows"] == 600
    said = [d for d in body["decisions"]
            if d["kind"] == "set_unit_of_analysis"][-1]
    assert said["payload"]["design_not_described"] is True
    # NOT the `record` sentence. Saying "one row is one record" would be the app
    # describing a design it was just told it cannot describe.
    assert "one row is one record" not in said["text"].lower()
    assert "Neither one row per participant nor one row per record" in said["text"]


def test_the_conservative_split_is_actually_grouped(client):
    """The seal SAYS "chosen by subject rather than by row, so no subject appears
    in both halves." This checks the sentence against the draw.

    It is here because the first version was false. `set_grain` kept `group_col`
    only for `people_repeat`, so the hatch recorded `basis: grouped` with
    `group_col: None` — and `draw_holdout` requires BOTH, so it fell through to a
    ROW split while `seal_disclosure` read the basis and announced a grouped one.
    The count gave it away ("from 0 subjects"), but the count is a symptom: a
    subject genuinely could appear on both sides. The escape hatch's whole promise
    is the conservative treatment, and a promise the split did not keep is worse
    than the wrong confident answer the option exists to avoid.
    """
    import pandas as pd

    pid = _upload(client)
    _decide(client, pid, "set_grain",
            {"answer": G.DESIGN_NOT_DESCRIBED, "group_col": "subject_id"})
    _decide(client, pid, "set_eligibility", {"answer": "everyone"})
    _decide(client, pid, "seal", {"fraction": 0.15, "seed": 42})

    body = client.get(f"/project/{pid}").json()
    lb = body["lockbox"]
    assert lb["group_col"] == "subject_id", "the named column survived to the seal"
    assert lb["n_groups"] == 200
    assert body["grain"]["group_col"] == "subject_id"

    df = pd.read_csv(DATA / "clinical_longitudinal.csv")
    held = {int(x) for x in lb["labels"]}
    test_subjects = set(df.loc[sorted(held), "subject_id"])
    train_subjects = set(df.drop(index=sorted(held))["subject_id"])
    assert not (test_subjects & train_subjects), (
        f"{len(test_subjects & train_subjects)} subject(s) appear on both sides, "
        f"and the seal sentence says none do")
    assert test_subjects, "nothing was held out"
    assert f"from {len(test_subjects)} subjects" in body["disclosures"]["seal"], (
        "the count in the sentence is not the count that was drawn")


def test_a_grouping_column_that_does_not_exist_is_refused(client):
    """The same defect from the other end: the existence check ran only for
    `people_repeat`, so the hatch accepted a name that was not a column and then
    dropped it — a typo became a silent row split."""
    pid = _upload(client)
    r = _decide(client, pid, "set_grain",
                {"answer": G.DESIGN_NOT_DESCRIBED, "group_col": "subjekt_id"})
    assert r.status_code == 400
    assert "subjekt_id" in r.json()["detail"]


# ── what the manuscript says at that point ───────────────────────────────────

def test_the_gap_sits_where_the_app_cannot_describe(client):
    """Recording *"my design isn't described here"* and then writing a methods
    section that describes it anyway would be the governing rule broken by the
    mechanism built to honor it."""
    pid = _upload(client)
    _decide(client, pid, "set_grain",
            {"answer": G.DESIGN_NOT_DESCRIBED, "group_col": "subject_id"})
    gaps = client.get(f"/project/{pid}/gaps").json()

    assert gaps["n"] == 1
    gap = gaps["gaps"][0]
    assert gap["where"] == "study_design"
    assert gap["after"] == "participants", "the gap is placed, not appended"
    assert gap["question"] == "state_grain"
    assert gap["text"].startswith(R.AUTHOR_REQUIRED)
    # It names what it does not know rather than gesturing at it.
    for named in ("Matched sets", "crossover", "nested sampling"):
        assert named in gap["text"]


def test_both_gaps_appear_when_both_hatches_are_taken(client):
    pid = _upload(client)
    _decide(client, pid, "set_grain",
            {"answer": "people_repeat", "group_col": "subject_id"})
    _decide(client, pid, "set_repeat_kind", {"kind": "time_points"})
    _decide(client, pid, "set_unit_of_analysis",
            {"unit": R.UNIT_NOT_DESCRIBED})
    gaps = client.get(f"/project/{pid}/gaps").json()
    assert [g["where"] for g in gaps["gaps"]] == ["unit_of_analysis"]

    pid2 = _upload(client)
    _decide(client, pid2, "set_grain",
            {"answer": G.DESIGN_NOT_DESCRIBED, "group_col": "subject_id"})
    assert [g["where"] for g in client.get(f"/project/{pid2}/gaps").json()["gaps"]] \
        == ["study_design"]


def test_a_record_with_no_hatch_taken_carries_no_gap(client):
    """A gap that appeared unprompted would make the marker meaningless."""
    pid = _upload(client)
    _decide(client, pid, "set_grain",
            {"answer": "people_repeat", "group_col": "subject_id"})
    assert client.get(f"/project/{pid}/gaps").json()["n"] == 0


def test_uncertainty_costs_no_more_than_a_wrong_confident_answer(client):
    """The rule the hatch exists to satisfy, asserted as a comparison.

    Every step a confident answer reaches, this answer reaches too. It labels
    and it records; it does not block.
    """
    pid = _upload(client)
    _decide(client, pid, "set_grain",
            {"answer": G.DESIGN_NOT_DESCRIBED, "group_col": "subject_id"})
    _decide(client, pid, "set_eligibility", {"answer": "everyone"})
    assert _decide(client, pid, "seal", {"fraction": 0.15,
                                         "seed": 42}).status_code == 200
    body = client.get(f"/project/{pid}").json()
    assert body["barrier_raised"] is True
    assert body["lockbox"]["n_test"] > 0

"""The eligibility question — clause §04, and clause §01's position for it.

`GUIDED-020`: this question was asked in NEITHER door, and the first coverage
matcher marked clause 04 covered because a test for the *other* half of it
existed. Three obligations, and the two that were missing are the two here:

* asked in **scientific terms**, with the target's distribution **withheld**;
* applied **pre-seal**, **changing N**, with the numbers participant flow needs.

The third — a robustness trim never touches the sealed rows — is pinned by
`tests/test_the_trim_does_not_touch_the_sealed_rows.py` and is post-seal.

The withheld distribution is the part worth being careful about, because it is a
**subtraction** and subtractions are invisible. A test that only checked the
question's wording would pass on an implementation that showed a histogram
beside it. So the assertions here are on the KEYS of the evidence payload: what
is absent is the claim.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from turbotab import eligibility as E, engine, grain as G          # noqa: E402
from turbotab.project import AnalysisProject, ProjectError         # noqa: E402


def study(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "record_id": [f"R{i:04d}" for i in range(n)],
        "age": rng.integers(12, 90, n).astype(float),
        "site": rng.choice(["north", "south", "east"], n),
        "glucose": rng.normal(95, 15, n),
        "outcome": rng.integers(0, 2, n),
    })


def _asked(df: pd.DataFrame | None = None) -> AnalysisProject:
    """A project standing exactly where the eligibility question is asked."""
    p = AnalysisProject.from_dataframe(df if df is not None else study(), "t")
    p.set_target("outcome", "classification", "high", [])
    p.set_grain(G.ONE_ROW_PER_PERSON)
    return p


# ── the question, and what it may not show ───────────────────────────────────

def test_the_question_withholds_the_distribution_and_says_why():
    """Clause §04's rigor point, and it is a REFUSAL rather than an omission.

    An eligibility criterion comes from the research question, not from the
    histogram. A user who needs to see the shape to decide where to cut is doing
    data-driven cohort selection — its own publishable bias — and showing them
    the shape is the app causing it.

    Asserted on the KEYS of the payload, not on its prose. "Where should I cut?"
    is answerable from a median, a quartile, a histogram, or per-value counts,
    and any of those could be added in one line while every sentence in the
    question still read correctly.

    Clause: `lockbox-04`
    """
    p = _asked()
    ev = E.permitted_evidence(p.df, "glucose")

    # PERMITTED: everything here answers "is this data corrupted?"
    assert ev["observed_min"] is not None and ev["observed_max"] is not None
    assert ev["n_missing"] == 0
    assert "n_sentinel" in ev and "n_negative" in ev

    # WITHHELD: everything that answers "where should I cut?"
    banned = {"median", "mean", "std", "quantiles", "percentiles", "histogram",
              "bins", "counts", "distribution", "quartiles", "iqr", "hist",
              "value_counts", "deciles"}
    leaked = sorted(k for k in ev if k.lower() in banned)
    assert not leaked, (
        f"the eligibility evidence carries {leaked}, which answers 'where "
        "should I cut?'. Clause §04 permits only what answers 'is this data "
        "corrupted?' — observed min/max and impossible-value flags.")

    # And the refusal is stated to the user, because a subtraction nobody
    # mentions reads as an omission rather than as rigor.
    assert "not showing you the outcome's distribution" in E.WITHHELD_DISCLOSURE
    assert "belongs later" in E.WITHHELD_DISCLOSURE
    assert E.EVIDENCE_CAPTION in ev["caption"]


def test_a_categorical_column_offers_its_values_but_not_their_counts():
    """"Which sites exist" is what the question asks the user to choose among.
    "How many rows per site" is a distribution, and it is the thing that would
    let somebody pick the site with the best outcome rate.

    Clause: `lockbox-04`
    """
    ev = E.permitted_evidence(_asked().df, "site")
    assert set(ev["values"]) == {"north", "south", "east"}, (
        "the values a user chooses among are not offered")
    assert "counts" not in ev and "n_per_value" not in ev, (
        "per-value counts are a distribution: they are what would let somebody "
        "pick the site with the best outcome rate, which is the cut-point "
        "question clause §04 withholds")


# ── the criterion: pre-seal, changes N, carries its flow numbers ─────────────

def test_an_exclusion_records_its_participant_flow_numbers():
    """TRIPOD+AI names continuous-variable restrictions as an eligibility item
    reported in participant flow. So the numbers are the deliverable, not a
    side effect: how many before, how many excluded, how many after, and why.

    Clause: `lockbox-04`
    """
    p = _asked()
    before = len(p.df)
    p.set_eligibility(E.RESTRICTED, column="age", minimum=18,
                      reason="The study is about adults.")

    rec = p.eligibility
    assert rec["answer"] == E.RESTRICTED
    assert rec["n_before"] == before
    assert rec["n_after"] == len(p.df), "the recorded N and the table disagree"
    assert rec["n_excluded"] == before - len(p.df)
    assert rec["n_excluded"] > 0, "this fixture excludes nobody, so it proves nothing"
    assert rec["reason"] == "The study is about adults."
    assert str(rec["n_excluded"]) in rec["sentence"] and "adults" in rec["sentence"]

    # It CHANGED N — the rows are gone, not masked. A view would leave them
    # available to the seal, which is the distinction §04 draws against a trim.
    assert (p.df["age"] >= 18).all()
    assert "labels" not in rec, (
        "the row labels of excluded people are stored on the project; "
        "participant flow needs the COUNT, not a list of who was removed")


def test_an_exclusion_with_no_reason_is_refused():
    """Participant flow reports how many AND why. A criterion whose reason
    cannot be written down should not be applied.

    Clause: `lockbox-04`
    """
    p = _asked()
    with pytest.raises(ProjectError, match="needs its reason"):
        p.set_eligibility(E.RESTRICTED, column="age", minimum=18)


def test_everyone_is_a_recorded_answer_not_an_absence():
    """"The study is about everyone here" and "nobody was asked" must not look
    the same in the record — the second is a step that did not happen, and a
    reader of the methods section has to be able to tell.

    Clause: `lockbox-04`
    """
    p = _asked()
    assert p.eligibility is None, "not asked yet"
    p.set_eligibility(E.EVERYONE)

    assert p.eligibility["answer"] == E.EVERYONE
    assert p.eligibility["n_excluded"] == 0
    assert "set_eligibility" in [d.kind for d in p.decisions], (
        "answering 'everyone' left no trace, so it is indistinguishable from "
        "never having been asked")
    assert E.disclosure(p.eligibility) != E.disclosure(
        {"answer": E.RESTRICTED, "sentence": "x"}), (
        "an unrestricted study and a restricted one render alike")


# ── the position in the sequence: clause §01 ─────────────────────────────────

def test_the_seal_cannot_be_drawn_before_eligibility_is_settled():
    """Clause §01 fixes the order as grain → eligibility → SEAL, and this is
    that ordering as a PRECONDITION rather than a comment.

    Answering afterwards would mean the held-out rows were chosen from people
    the study is not about, and the test set would be obeying a criterion it was
    not drawn under.

    Clause: `lockbox-01`
    """
    p = _asked()
    drawn = engine.draw_holdout(p.df, "outcome", "classification", p.grain)
    with pytest.raises(ProjectError, match="eligibility question"):
        p.seal_lockbox(drawn["labels"], **drawn["disclosure"])

    p.set_eligibility(E.EVERYONE)
    drawn = engine.draw_holdout(p.df, "outcome", "classification", p.grain)
    p.seal_lockbox(drawn["labels"], **drawn["disclosure"])
    assert p.barrier_raised


def test_eligibility_cannot_be_answered_before_the_grain():
    """The other end of the same ordering. §01 is a sequence, not a set.

    Clause: `lockbox-01`
    """
    p = AnalysisProject.from_dataframe(study(), "t")
    p.set_target("outcome", "classification", "high", [])
    with pytest.raises(ProjectError, match="grain question comes before"):
        p.set_eligibility(E.EVERYONE)


def test_an_exclusion_after_the_seal_is_refused_and_says_where_to_go():
    """§04: *"Also trim the test set to match" is permanently off the menu.* A
    user who truly wants the narrower population is routed back to the pre-seal
    question, which requires a re-seal and is its own hard, logged decision.

    Clause: `lockbox-04`
    """
    p = _asked()
    p.set_eligibility(E.EVERYONE)
    drawn = engine.draw_holdout(p.df, "outcome", "classification", p.grain)
    p.seal_lockbox(drawn["labels"], **drawn["disclosure"])

    with pytest.raises(ProjectError) as exc:
        p.set_eligibility(E.RESTRICTED, column="age", minimum=18,
                          reason="adults only")
    said = str(exc.value)
    assert "already sealed" in said
    assert "re-seal" in said, (
        "the refusal does not say what to do instead, which makes it a dead end")


def test_a_criterion_that_empties_the_study_is_refused():
    """Nothing downstream can run on an empty study, and a seal drawn from zero
    rows would be a lock over nothing.

    Clause: `lockbox-04`
    """
    p = _asked()
    with pytest.raises(ProjectError, match="removes every row"):
        p.set_eligibility(E.RESTRICTED, column="age", minimum=500,
                          reason="impossible cohort")


# ── over HTTP: what a browser can actually do ────────────────────────────────

@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from turbotab.api import app
    return TestClient(app)


def _upload(client, df: pd.DataFrame) -> str:
    r = client.post("/project", files={
        "file": ("study.csv", df.to_csv(index=False).encode(), "text/csv")})
    assert r.status_code == 200, r.text
    return r.json()["id"]


def test_a_driver_excludes_a_cohort_and_reads_the_flow_numbers_back(client):
    """The whole obligation over HTTP: the question is asked in its place, the
    permitted evidence is servable, the exclusion changes N, and the disclosure
    that comes back says what happened.

    Clause: `lockbox-04`
    """
    pid = _upload(client, study())
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_target", "payload": {"column": "outcome"}})
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_grain",
                      "payload": {"answer": G.ONE_ROW_PER_PERSON}})

    iv = client.get(f"/project/{pid}/interview?step=data").json()
    q = next(q for q in iv["questions"] if q["key"] == "state_eligibility")
    assert q["status"] == "asked"
    assert q["consumer"], "a FACT must name what reads its answer"
    assert "not showing you the outcome's distribution" in q["why"], (
        "the question does not tell the user what it is withholding, so the "
        "refusal reads as an omission")

    ev = client.post(f"/project/{pid}/decision",
                     json={"kind": "eligibility_evidence",
                           "payload": {"column": "age"}}).json()
    assert "observed_min" in ev and "median" not in ev

    before = client.get(f"/project/{pid}").json()["n_rows"]
    r = client.post(f"/project/{pid}/decision",
                    json={"kind": "set_eligibility",
                          "payload": {"answer": "restricted", "column": "age",
                                      "minimum": 18,
                                      "reason": "The study is about adults."}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_rows"] < before, "the exclusion did not change N"
    assert body["eligibility"]["n_excluded"] == before - body["n_rows"]
    assert "excluded before the held-out set was drawn" in \
        body["disclosures"]["eligibility"]

    sealed = client.post(f"/project/{pid}/decision", json={"kind": "seal"})
    assert sealed.status_code == 200, sealed.text
    assert sealed.json()["lockbox"]["n_total"] == body["n_rows"], (
        "the seal was drawn against the wider population, so the held-out set "
        "includes people the study is not about")


def test_the_seal_endpoint_refuses_before_eligibility_is_answered(client):
    """§01's ordering, over HTTP, with the recorded answer named as the way out.

    Clause: `lockbox-01`
    """
    pid = _upload(client, study())
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_target", "payload": {"column": "outcome"}})
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_grain",
                      "payload": {"answer": G.ONE_ROW_PER_PERSON}})
    r = client.post(f"/project/{pid}/decision", json={"kind": "seal"})
    assert r.status_code == 400
    assert "eligibility question comes before the seal" in r.json()["detail"]
    assert "everyone here" in r.json()["detail"], (
        "the refusal does not name the answer that settles it")


def test_the_exclusion_survives_the_save_file():
    """A restored project that lost its eligibility answer would ask the
    question again AFTER its seal exists — which `set_eligibility` refuses — so
    it would be stuck with no way to say the step had happened.

    Clause: `lockbox-04`
    """
    from turbotab import archive
    p = _asked()
    p.set_eligibility(E.RESTRICTED, column="age", minimum=18,
                      reason="The study is about adults.")
    drawn = engine.draw_holdout(p.df, "outcome", "classification", p.grain)
    p.seal_lockbox(drawn["labels"], **drawn["disclosure"])

    back = archive.from_bytes(archive.to_bytes(p))
    assert back.eligibility["answer"] == E.RESTRICTED
    assert back.eligibility["n_excluded"] == p.eligibility["n_excluded"]
    assert back.eligibility["reason"] == "The study is about adults."

"""`GUIDED-048` and `GUIDED-049` — the question, and its first two consumers.

`DOMAIN_SCIENCE.md` §01.3 is the deepest of the seven convergences:

> The same dataset, the same target and the same lens can require **opposite**
> handling depending on whether the user wants prediction or inference.

Five places across four domains where the advice does not shade but **inverts**.
TurboTab assumed prediction throughout — reasonably, for a predictive-modeling
app — and never asked. A tool that gives the inference answer to somebody
building a bedside model is wrong, and so is the reverse.

## Why the wiring matters more than the question

A question with no consumer is a question that changes nothing, and this file
exists mostly to assert that two consumers really read it:

* **The missing-data route** (`GUIDED-048`). A was-it-missing indicator carries
  the clinician's decision to order a test. It is observable at deployment, so
  it is legitimate and often helpful for prediction — and a known source of bias
  in an association estimate. Same column, same data, opposite answer.
* **The class-imbalance advice** (`GUIDED-049`), which is the first instance of
  the anti-pattern audit and is a **defect in shipped code** rather than a
  feature request: the app recommended rebalancing and then asserted it in the
  generated manuscript.

## The shape of the refusal

Blocked with both exits, never hard-refused. §09's CONSEQUENCE is
resolve-or-attest, the user may have a reason, and a tool that blocks a correct
analysis is a tool people route around. And **unanswered blocks nothing** — the
app does not get to infer a purpose and then hold somebody to it.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml import imbalance_advice as IA                                 # noqa: E402
from ml import router                                                 # noqa: E402
from turbotab import purpose as PU                                    # noqa: E402

DATA = Path(__file__).resolve().parent / "sample_data"


def _client():
    from fastapi.testclient import TestClient

    from turbotab import api
    return TestClient(api.app)


def _project(client, target="outcome", purpose=None, name="clinic_visits"):
    with open(DATA / f"{name}.csv", "rb") as fh:
        pid = client.post("/project", files={
            "file": (f"{name}.csv", fh, "text/csv")}).json()["id"]
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_target", "payload": {"column": target}})
    if purpose:
        r = client.post(f"/project/{pid}/decision",
                        json={"kind": "set_purpose", "payload": {"answer": purpose}})
        assert r.status_code == 200, r.text
    return pid


# ── the question ────────────────────────────────────────────────────────────

def test_it_is_asked_immediately_after_the_target():
    client = _client()
    pid = _project(client)
    plan = client.get(f"/project/{pid}/interview?step=data").json()
    asked = {q["key"]: q for q in plan["questions"]
             if q["mode"] == "push" and q["status"] == "asked"}
    assert "state_purpose" in asked, sorted(asked)
    assert asked["state_purpose"]["seq"] == "2.5"
    assert asked["state_purpose"]["clause"] == "lockbox-01"


def test_it_is_not_asked_before_there_is_a_target():
    """It is about what the target is FOR, so it needs one."""
    client = _client()
    with open(DATA / "clinic_visits.csv", "rb") as fh:
        pid = client.post("/project", files={
            "file": ("c.csv", fh, "text/csv")}).json()["id"]
    keys = [q["key"] for q in
            client.get(f"/project/{pid}/interview?step=data").json()["questions"]]
    assert "state_purpose" not in keys


def test_it_is_a_choice_and_never_skippable_at_any_confidence():
    """Nothing in the data reveals it, so no confidence in the engine could
    make it moot — and a pre-selected purpose would be the app deciding what
    the user's paper is about."""
    assert "purpose" in router.CHOICE_KINDS
    assert "purpose" not in router.FACT_KINDS
    assert router._skip_is_permitted("high", "purpose") is False
    client = _client()
    pid = _project(client)
    q = next(q for q in
             client.get(f"/project/{pid}/interview?step=data").json()["questions"]
             if q["key"] == "state_purpose")
    assert q["status"] == "asked"
    # No option is marked as the recommendation.
    assert len(q["option_values"]) == 2
    assert set(q["option_values"]) == set(PU.PURPOSES)


def test_both_answers_are_recorded_with_a_methods_sentence():
    client = _client()
    for answer in PU.PURPOSES:
        pid = _project(client, purpose=answer)
        said = next(d for d in client.get(f"/project/{pid}").json()["decisions"]
                    if d["kind"] == "set_purpose")
        assert said["text"] == PU.methods_sentence(answer)
        assert len(said["text"]) > 80
        assert "TurboTab" not in said["text"] and "the app" not in said["text"], (
            "the sentence is about the software rather than about the study, "
            "so it cannot appear in a methods section")
    assert PU.methods_sentence(PU.PREDICTION) != PU.methods_sentence(PU.INFERENCE)


def test_there_is_no_third_answer_and_no_default():
    client = _client()
    pid = _project(client)
    r = client.post(f"/project/{pid}/decision",
                    json={"kind": "set_purpose", "payload": {"answer": "both"}})
    assert r.status_code == 400
    assert "no default" in r.text
    assert client.get(f"/project/{pid}").json()["purpose"] is None


# ── consumer 1 · the missing-data route ─────────────────────────────────────

def _route_indicator(client, pid, column):
    return client.post(f"/project/{pid}/decision", json={
        "kind": "route_missingness", "subject": column,
        "payload": {"column": column, "strategy": "indicator",
                    "mechanism": "not_sure"}})


def _first_missing_column(client, pid):
    cards = client.get(f"/project/{pid}/evidence/missingness").json()["cards"]
    assert cards, "no missingness card on this fixture"
    return cards[0]["column"]


def test_the_same_choice_is_accepted_for_prediction_and_blocked_for_inference():
    """**The claim, in one test.** Same fixture, same column, same strategy —
    and the app answers oppositely, because the purpose is what differs."""
    client = _client()

    predicting = _project(client, purpose=PU.PREDICTION)
    column = _first_missing_column(client, predicting)
    assert _route_indicator(client, predicting, column).status_code == 200

    inferring = _project(client, purpose=PU.INFERENCE)
    blocked = _route_indicator(client, inferring, column)
    assert blocked.status_code == 409, (
        "a was-it-missing indicator is a known source of bias in an "
        "association estimate and was accepted without a word")


def test_the_block_carries_both_exits_and_its_citation():
    """An interface cannot render the interruption without also rendering its
    way out — the same shape the lens contradiction uses."""
    client = _client()
    pid = _project(client, purpose=PU.INFERENCE)
    detail = _route_indicator(client, pid, _first_missing_column(client, pid)).json()["detail"]
    assert [e["id"] for e in detail["exits"]] == ["impute_median", "attest"]
    assert detail["acknowledgment_kind"] == "typed"
    assert detail["evidence_status"] == "SETTLED" and detail["source"]
    # It states the OTHER reading too, which is what makes it a fork rather
    # than a scolding.
    assert "prediction objective it is" in detail["message"]


def test_an_unanswered_purpose_blocks_nothing():
    """The app does not get to infer a purpose and then hold somebody to it."""
    client = _client()
    pid = _project(client)                                 # no purpose recorded
    assert client.get(f"/project/{pid}").json()["purpose"] is None
    assert _route_indicator(
        client, pid, _first_missing_column(client, pid)).status_code == 200, (
        "an unrecorded purpose blocked a choice, so the app inferred one and "
        "then held the user to it")


def test_the_attestation_exit_actually_lets_it_through():
    """A blocker with an exit that does not work is a hard refusal wearing a
    resolve-or-attest costume."""
    client = _client()
    pid = _project(client, purpose=PU.INFERENCE)
    column = _first_missing_column(client, pid)
    r = client.post(f"/project/{pid}/decision", json={
        "kind": "route_missingness", "subject": column,
        "payload": {"column": column, "strategy": "indicator",
                    "mechanism": "not_sure", "acknowledge_signal_loss": True}})
    assert r.status_code == 200, r.text
    assert any(d["kind"] == "route_missingness"
               for d in client.get(f"/project/{pid}").json()["decisions"])


def test_the_purpose_survives_the_save_file():
    """The downstream defaults are read from it, so a restored project that
    lost it would silently revert to the app's old assumption."""
    from turbotab import archive
    from turbotab.api import STORE

    client = _client()
    pid = _project(client, purpose=PU.INFERENCE)
    blob = archive.to_bytes(STORE.get(pid))
    restored = archive.from_bytes(blob)
    assert (restored.purpose or {}).get("answer") == PU.INFERENCE


# ── consumer 2 · the class-imbalance anti-pattern (`GUIDED-049`) ────────────

def test_the_app_no_longer_recommends_rebalancing_to_anybody():
    """The defect, as the property it broke.

    `recommended` is never True. The two purposes the app can distinguish are
    the two where rebalancing is contraindicated — for *different* reasons, and
    both are stated.
    """
    for purpose in (PU.PREDICTION, PU.INFERENCE, None):
        advice = IA.advice(purpose)
        assert advice["recommended"] is False
        assert IA.CITATION in advice["advisory"]
        assert advice["evidence_status"] == "SETTLED" and advice["source"]
        assert advice["instead"], "removing advice without replacing it"
    assert IA.advice(PU.INFERENCE)["advisory"] != IA.advice(PU.PREDICTION)["advisory"], (
        "prediction and inference land in the same place for different "
        "reasons, and saying so is the honest reading rather than a shortcut")
    assert "intercept" in IA.advice(PU.INFERENCE)["advisory"]


def test_the_capability_is_routed_rather_than_deleted():
    """A fixed-operating-point classifier is the one place it survives, and the
    app cannot currently tell one from a risk model — so it is offered with the
    citation, never recommended."""
    assert "fixed operating point" in IA.FIXED_POINT_NOTE
    for purpose in (PU.PREDICTION, PU.INFERENCE, None):
        assert IA.advice(purpose)["offered_note"] == IA.FIXED_POINT_NOTE


def test_the_engine_no_longer_advises_smote_or_class_weights():
    """Read off the shipped advisory, not off the source.

    `ml/dataset_profile.py:429` said "Use class weights in training" and
    "Consider SMOTE or other resampling"; `ml/eda_recommender.py:419` repeated
    it.
    """
    import numpy as np
    import pandas as pd

    from ml.dataset_profile import compute_dataset_profile, generate_warnings
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({
        "x1": rng.normal(size=n), "x2": rng.normal(size=n),
        "y": np.r_[np.ones(12), np.zeros(n - 12)].astype(int)})
    profile = compute_dataset_profile(df, target_col="y",
                                      task_type="classification")
    warnings = generate_warnings(profile)
    actions = [a for w in warnings for a in (getattr(w, "suggested_actions", None) or [])]
    assert actions, "no suggested actions at all; the fixture stopped triggering"
    joined = " ".join(actions).lower()
    assert "smote" not in joined, "SMOTE is still recommended"
    assert "class weight" not in joined, "class weights are still recommended"
    assert "precision-recall" in joined or "calibration" in joined, (
        "the advice was removed and nothing honest replaced it")


def test_the_manuscript_reports_what_was_done_and_no_longer_endorses_it():
    """The serious half. The app asserted this in the artifact that IS the
    product — unconditionally, and approvingly."""
    said = IA.manuscript_sentence(PU.PREDICTION)
    assert "class_weight='balanced') was applied" in said, (
        "the manuscript stopped reporting what was done; a reader has to know")
    assert "reported as a limitation" in said
    assert IA.CITATION in said
    assert "To address class imbalance" not in said, (
        "the endorsing framing survived")
    assert "intercept" in IA.manuscript_sentence(PU.INFERENCE)


def test_the_narrative_engine_uses_the_one_sentence():
    """Principle-locality: the citation and the qualification live in one
    place, and the manuscript reads it rather than holding a copy."""
    source = (Path(__file__).resolve().parents[1] / "ml" / "narrative_engine.py"
              ).read_text(encoding="utf-8")
    assert len(source) > 20_000                            # positive control
    assert "manuscript_sentence" in source
    # Asserted on the CODE, not on a grep of the file: the comment above the
    # fix quotes the old sentence to say why it went, and a whole-file search
    # cannot tell an explanation from a relapse. `GUIDED-045`, on the test
    # written to close `GUIDED-049`.
    code = "\n".join(l for l in source.split("\n")
                     if not l.strip().startswith("#"))
    assert "To address class imbalance" not in code, (
        "the old unconditional sentence is still in the manuscript chain")

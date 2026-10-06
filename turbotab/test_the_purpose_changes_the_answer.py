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
# The tests here that drove the retired legacy app (turbotab/api.py, the page, the
# figure and manuscript modules, the gates) were removed with it, BLUEPRINT §9.1; git keeps them.
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml import imbalance_advice as IA                                 # noqa: E402
from turbotab import purpose as PU                                    # noqa: E402

DATA = Path(__file__).resolve().parent / "sample_data"


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

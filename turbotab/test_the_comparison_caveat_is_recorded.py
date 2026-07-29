"""Per-model preparation is correct, and it costs something. The record says so.

If Lasso gets scaled features and a boosted ensemble gets raw ones and the
ensemble wins — did the model win, or did the pipeline? Under per-model
preparation that question has no answer from the results alone, and the honest
response is neither to forbid per-model preparation nor to ship the ambiguity
silently. It is to ask once, recommend the right answer, and **write the caveat
into the manuscript automatically.**

Asked ONCE, not per model: it is a property of the comparison rather than of any
model in it. Asked AFTER the models are chosen, because "should they all get the
same preparation" is not a question until there is a *they*.

**Automatically is the load-bearing word.** A caveat the user has to remember to
add is a caveat that appears in the methods sections of careful people and
nowhere else, and the careful people were never the risk. So the assertion here
is against the drafted manuscript — the object a reader would actually receive —
and not against the decision record, which would prove only that the app knew.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml import router                                                 # noqa: E402
from turbotab import draft, eligibility as E, engine, grain as G      # noqa: E402
from turbotab.api import app                                          # noqa: E402
from turbotab.project import AnalysisProject, ProjectError            # noqa: E402


def study(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    df = pd.DataFrame({
        "pid": [f"P{i:04d}" for i in range(n)],
        "age": rng.integers(20, 80, n).astype(float),
        "bmi": np.round(rng.normal(28, 5, n), 1),
        "hdl": np.round(rng.normal(52, 14, n), 1),
    })
    df["outcome"] = (df["age"] + rng.normal(0, 8, n) > 52).astype(int)
    return df


def sealed(models=("rf", "logreg")) -> AnalysisProject:
    p = AnalysisProject.from_dataframe(study(), "caveat")
    p.set_target("outcome", "classification", "high", [])
    p.set_grain(G.ONE_ROW_PER_PERSON)
    p.set_eligibility(E.EVERYONE)
    d = engine.draw_holdout(p.df, "outcome", "classification", p.grain)
    p.seal_lockbox(d["labels"], **d["disclosure"])
    p.select_models(list(models))
    return p


# ── the question: once, in the right place, and a CHOICE ─────────────────────

def test_the_question_is_not_asked_until_there_is_a_comparison_to_caveat():
    """"Should they all get the same preparation" needs a *they*.

    Asked before the models are chosen, it is a question about an empty set —
    and answering it would record a methodological commitment the user made
    before seeing what it applied to.
    """
    before = router.plan([], target="outcome", step="preprocess", answered=[])
    assert not any(q.key == "choose_preparation_mode" for q in before), (
        "the preparation question was asked before any model was chosen, so it "
        "is a question about an empty set — and answering it records a "
        "methodological commitment made before seeing what it applies to")
    assert any(q.key == "choose_models" for q in before), (
        "model selection is not asked at Preprocess, so the preparation "
        "question can never become reachable")

    after = router.plan([], target="outcome", step="preprocess",
                        answered=["choose_models"])
    assert any(q.key == "choose_preparation_mode" for q in after)


def test_the_question_is_asked_once_and_stays_answered():
    answered = ["choose_models", "choose_preparation_mode"]
    plan = router.plan([], target="outcome", step="preprocess", answered=answered)
    assert not any(q.key == "choose_preparation_mode" for q in plan), (
        "the preparation question comes back after being answered; asked per "
        "model, it stops being a property of the comparison")


def test_it_is_a_choice_and_therefore_never_skippable():
    """The engine has a recommendation, and a recommendation is not certainty.

    What comparison you want to make is not a property of the data, so no
    confidence could make this skippable. Asserted against the routing
    constitution's own sets rather than against the question's text, because
    the sets are what `_skip_is_permitted` reads.
    """
    q = next(x for x in router.plan([], target="outcome", step="preprocess",
                                    answered=["choose_models"])
             if x.key == "choose_preparation_mode")
    assert q.kind in router.CHOICE_KINDS
    assert q.kind not in router.FACT_KINDS
    assert not router._skip_is_permitted("high", q.kind), (
        "the preparation mode can be skipped at high confidence, which would "
        "commit the analysis to a comparison the user never chose")


def test_the_question_recommends_and_states_what_the_recommendation_costs():
    """Prose IS the deliverable, so the assertion is on the two claims.

    A recommendation with no cost attached is advice the reader cannot weigh;
    a cost with no recommendation is a question that pretends the two answers
    are equally good when they are not.
    """
    q = next(x for x in router.plan([], target="outcome", step="preprocess",
                                    answered=["choose_models"])
             if x.key == "choose_preparation_mode")
    assert "recommend" in q.why.lower(), "the question makes no recommendation"
    assert "not informative either" in q.why, (
        "the recommendation does not say why per-model is right — a model "
        "handicapped by preparation it does not suit is not informative either")
    assert "written into your methods section automatically" in q.why, (
        "the question does not tell the user the caveat will be recorded for "
        "them, so the honest choice looks like the costly one")
    assert any("recommended" in o.lower() for o in q.options), (
        "neither option is marked as the recommended one, so the "
        "recommendation lives only in the explanatory text")


# ── the caveat: automatic, and in the manuscript ─────────────────────────────

def test_choosing_per_model_writes_the_caveat_into_the_manuscript():
    """The claim, against the drafted document rather than the decision record.

    Asserting on `decision.payload["caveat"]` would prove the app knew. What
    matters is that a reader who receives the manuscript reads it, so this
    asserts against `draft.draft()` output — the object that actually travels.
    """
    p = sealed()
    p.set_preparation_mode("per_model")
    doc = draft.draft(p.to_dict())
    limitations = next(s for s in doc["sections"] if s["key"] == "limitations")
    text = " ".join(s["text"] for s in limitations["sentences"])

    assert "reflects the model and its preparation together" in text, (
        "the manuscript does not state that a difference between two models is "
        "confounded with their preparation, which is the whole content of the "
        "caveat")
    assert "cannot be separated from these results alone" in text, (
        "the caveat states the confound without stating that these results "
        "cannot resolve it, which is the part a reader needs")


def test_the_caveat_does_not_appear_when_there_is_nothing_to_caveat():
    """A caveat attached to every analysis is a caveat nobody reads.

    Under uniform preparation the confound does not exist, so recording it
    would be recording a limitation the study does not have — which costs the
    reader exactly as much credibility as omitting a real one.
    """
    p = sealed()
    p.set_preparation_mode("uniform")
    doc = draft.draft(p.to_dict())
    limitations = next(s for s in doc["sections"] if s["key"] == "limitations")
    text = " ".join(s["text"] for s in limitations["sentences"])
    assert "reflects the model and its preparation together" not in text, (
        "choosing uniform recorded the per-model caveat anyway, so the "
        "manuscript states a limitation this analysis does not have — which "
        "costs the reader exactly as much credibility as omitting a real one")
    assert "differences between the models rather than between their pipelines" \
        in text, (
        "choosing uniform recorded nothing at all; the choice to hold "
        "preparation constant is itself a methods sentence")


def test_the_caveat_is_the_full_paragraph_and_not_a_summary_of_it():
    """Structural, against the module's own constant.

    A caveat truncated to fit a card is the failure this project keeps finding
    in other places — the concern exists, and the version that reaches the
    reader is the one that fits.
    """
    from turbotab.project import _COMPARISON_CAVEAT
    p = sealed()
    d = p.set_preparation_mode("per_model")
    assert d.payload["caveat"] == _COMPARISON_CAVEAT
    doc = draft.draft(p.to_dict())
    limitations = next(s for s in doc["sections"] if s["key"] == "limitations")
    text = " ".join(s["text"] for s in limitations["sentences"])
    assert _COMPARISON_CAVEAT in text, (
        "the manuscript carries a paraphrase of the caveat rather than the "
        "caveat")


def test_an_unknown_mode_is_refused():
    p = sealed()
    with pytest.raises(ProjectError):
        p.set_preparation_mode("whichever")


# ── what the answer actually changes ─────────────────────────────────────────

def test_uniform_resolves_every_model_against_one_and_says_which():
    """Otherwise "the same preparation" is a promise nothing keeps.

    The substitution is recorded on each row rather than performed silently: a
    user reading a linear model's recipe under a tree's settings needs to know
    why it says what it says, or the interface looks broken at the exact moment
    it is doing what they asked.
    """
    p = sealed(models=["rf", "logreg"])
    per = p.resolved_recipes()          # no mode set yet: per-model
    p.set_preparation_mode("uniform")
    uni = p.resolved_recipes()

    def variant(rows, op):
        return next(r["variant"] for r in rows if r["operation"] == op)

    assert variant(per["rf"], "scale") != variant(per["logreg"], "scale"), (
        "the two models do not differ under per-model preparation, so this "
        "fixture cannot tell uniform from per-model")
    assert variant(uni["rf"], "scale") == variant(uni["logreg"], "scale")

    # The shelf's order decides which model the shared settings come from, so
    # the source is read rather than assumed — a test that hardcoded `rf` would
    # start failing the day the coach reordered the shelf, for no reason.
    source = p.selected_models[0]
    other = next(k for k in p.selected_models if k != source)
    assert not [r for r in uni[source] if r.get("uniform_source")], (
        "the model the settings came FROM is marked as borrowing them")
    borrowed = [r for r in uni[other] if r.get("uniform_source")]
    assert borrowed, "no row says whose settings it is showing"
    assert all(r["uniform_source"] == source for r in borrowed)
    assert "one shared preparation" in borrowed[0]["reason"], (
        "the row shows another model's setting without saying so, which reads "
        "as the table getting the model wrong")


def test_the_mode_survives_the_archive_with_the_models_it_describes():
    """A restored project that lost the mode would redraft without the caveat.

    The caveat is generated from the recorded decision, so losing the record
    does not produce a visible error — it produces a manuscript that is missing
    a limitation, which is the quietest possible failure.
    """
    from turbotab import archive
    p = sealed()
    p.set_preparation_mode("per_model")
    back = archive.from_bytes(archive.to_bytes(p))

    assert back.preparation_mode == "per_model", (
        "the preparation mode did not survive the archive, so a restored "
        "project redrafts without the caveat and nothing raises")
    assert back.selected_models == p.selected_models, (
        "the selected models did not survive, so the caveat would describe a "
        "comparison between models the record can no longer name")
    doc = draft.draft(back.to_dict())
    limitations = next(s for s in doc["sections"] if s["key"] == "limitations")
    text = " ".join(s["text"] for s in limitations["sentences"])
    assert "cannot be separated from these results alone" in text


# ── the same path over HTTP, because that is how a driver reaches it ─────────

@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_the_interview_stops_asking_once_the_decision_is_recorded(client):
    """The replay half: an answered question that keeps coming back is not answered.

    `api` maps `set_preparation_mode` back to the question key, and this is the
    only place that mapping is checked — a typo there would ask the user the
    same methodological question at every visit to the step.
    """
    raw = study().to_csv(index=False).encode()
    pid = client.post("/project",
                      files={"file": ("s.csv", raw, "text/csv")}).json()["id"]

    def decide(kind, **kw):
        r = client.post(f"/project/{pid}/decision", json={"kind": kind, **kw})
        assert r.status_code < 400, f"{kind} refused: {r.text[:300]}"

    decide("set_target", payload={"column": "outcome"})
    decide("set_grain", payload={"answer": G.ONE_ROW_PER_PERSON})
    decide("set_eligibility", payload={"answer": E.EVERYONE})
    decide("seal")
    decide("select_models", payload={"models": ["rf", "logreg"]})

    keys = [q["key"] for q in
            client.get(f"/project/{pid}/interview?step=preprocess").json()["questions"]]
    assert "choose_preparation_mode" in keys
    assert "choose_models" not in keys, (
        "the model question came back after being answered")

    decide("set_preparation_mode", payload={"mode": "per_model"})
    keys = [q["key"] for q in
            client.get(f"/project/{pid}/interview?step=preprocess").json()["questions"]]
    assert "choose_preparation_mode" not in keys

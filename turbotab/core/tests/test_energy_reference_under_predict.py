"""BEATS_PREDICT §9 ruling 5 in the methods reference and the review packets (W1-P11, repair
round): under Predict the Router asks neither the energy model nor the substitution, so the
generated reference and packets say the app does not ask them there, rather than print the energy
model's ranking as what the app offers first; under Estimate both read as before.

Every expectation is from construction: which question a goal asks is the ruling's, read off the
Router (``interview.route``) on a dietary table built here, never off the generator's own words;
the captures the packet test reads are built here too. Each guard is checked both ways: the
"not asked" words and the stale-capture note appear under Predict and stay out under Estimate.
"""
from __future__ import annotations

from turbotab.core import custom_sound
from turbotab.core.decisions import ProjectState
from turbotab.core.interview import route
from turbotab.core.methods.energy import RANKING
from turbotab.core.reference import defaults, journeys, methods, packet

ROLES = {"participant_id": "identifier", "age": "covariate", "energy_kcal": "energy",
         "protein_g": "exposure", "fat_g": "exposure", "carbohydrate_g": "exposure"}
FRESH = {s: {"status": "fresh"} for s in (
    "ingest", "oriented", "profile", "findings", "structure", "working", "target_info", "roles",
    "proposals", "cohort", "split", "shelf", "design", "fit", "substitution")}


def _router(purpose: str) -> dict:
    """The Router's steps on a dietary table with total energy beside three nutrients."""
    state = ProjectState(lens=["dietary"], target="hba1c", task="regression", purpose=purpose,
                         grain={"grain": "one_row_per_unit"}, roles=ROLES, exclusions=[],
                         missing="impute", split={"holdout": 0.2})
    return {s.key: s for s in route(state, FRESH)}


def _reason(key: str) -> str:
    step = _router("prediction")[key]
    assert step.status == "not_applicable", key  # the ruling: not asked under Predict
    return step.reason


def test_the_reference_reads_why_from_the_router_under_predict_and_nothing_under_estimate():
    for key in ("energy_adjustment", "substitution"):
        assert defaults.not_asked(key, "prediction") == _reason(key)
        # under Estimate the Router asks both on this table, and the reference names no reason
        assert _router("inference")[key].status in ("open", "waiting"), key
        assert defaults.not_asked(key, "inference") is None, key
    # a question the goal alone never removes has no such reason under either goal
    for purpose in defaults.PURPOSES:
        assert defaults.not_asked("missing", purpose) is None


def test_the_energy_models_entry_says_it_is_not_asked_under_predict_and_ranks_under_estimate():
    reason = _reason("energy_adjustment")
    text = "\n".join(methods.custom_sound_section("energy_adjustment", "the energy model"))
    assert "Under **prediction** it is not asked" in text and f"“{reason}”" in text
    assert "Under **prediction**, soundest first" not in text
    assert "- Under prediction" not in text  # no option offered first under Predict
    # under Estimate the ranking stands, soundest first, with the data's own first options
    first = custom_sound.energy("inference").options[0].key
    assert "Under **inference**, soundest first" in text and f"| 1 | `{first}`:" in text
    assert "- Under inference, a table with" in text
    assert methods.question_first("energy_adjustment", "prediction") is None
    assert methods.question_first("energy_adjustment", "inference")[0] == first
    assert defaults.cases("energy_adjustment", "prediction") is None
    assert {c.key for c in defaults.cases("energy_adjustment", "inference")} <= set(
        RANKING["inference"])


def test_an_intakes_transform_offers_no_energy_model_under_predict_and_its_first_under_estimate():
    reason = _reason("energy_adjustment")
    intake = [c for c in defaults.cases("exposure_transform", "prediction")
              if c.when.startswith("an intake")]
    assert len(intake) == 1
    assert intake[0].key == "" and reason in intake[0].first
    assert not any(f"`{m}`" in intake[0].first for m in RANKING["prediction"])
    estimate = [c for c in defaults.cases("exposure_transform", "inference")
                if c.when.startswith("an intake")]
    assert estimate and all(c.key in RANKING["inference"] for c in estimate)
    # the other inputs' transforms do not depend on the goal
    assert ([c for c in defaults.cases("exposure_transform", "prediction") if c not in intake]
            == [c for c in defaults.cases("exposure_transform", "inference")
                if c not in estimate])


def _capture(*kinds: str) -> dict:
    answers = [{"kind": "set_lens", "status": 200, "summary": '{"lenses": ["dietary"]}',
                "source": "the journey", "seq": 1}]
    answers += [{"kind": k, "status": 200, "summary": "{}", "source": "the journey",
                 "seq": i + 2} for i, k in enumerate(kinds)]
    return {"captured": {"commit": "0000000", "date": "2026-10-01", "seconds": 1, "workers": 2},
            "answers": answers, "readings": None, "export": {"status": 409}}


def test_a_packet_flags_a_capture_that_answered_a_question_its_goal_no_longer_asks():
    predict, estimate = (journeys.JOURNEYS["dietary-prediction"],
                         journeys.JOURNEYS["dietary-inference"])
    stale = _capture("set_split", "set_energy_adjustment", "set_substitution")
    assert packet.unasked_answers(predict, stale) == [(3, "energy_adjustment"),
                                                      (4, "substitution")]
    text = "\n".join(packet.journey_section(predict, stale))
    assert packet.STALE in text and "rows 3 and 4" in text
    for key in ("energy_adjustment", "substitution"):
        assert f"`{key}` (“{_reason(key)}”)" in text
    assert f"python -m turbotab.core.reference.journeys {predict.name}" in text
    # silent where the goal asks them, and where the capture never answered them
    assert packet.unasked_answers(estimate, stale) == []
    assert packet.STALE not in "\n".join(packet.journey_section(estimate, stale))
    assert packet.STALE not in "\n".join(packet.journey_section(predict, _capture("set_split")))


def test_the_dietary_packet_says_the_energy_model_is_not_asked_under_predict():
    text = packet.render("dietary", packet.captures_for("dietary"))
    labeled = text[text.index("### 3.2"):text.index("### 3.3")]
    row = next(line for line in labeled.splitlines() if "(`energy_adjustment`)" in line)
    assert row.split(" | ")[1] == "not asked under prediction"
    assert row.split(" | ")[2].startswith(packet.BY_DATA)
    ranked = text[text.index("### 3.4"):text.index("## 4 ")]
    assert "| The energy model (`energy_adjustment`) | prediction |" not in ranked
    assert "| The energy model (`energy_adjustment`) | inference |" in ranked

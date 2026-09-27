"""The Router (Tier A, graph semantics): order, the one open question, waiting, skipping."""
from __future__ import annotations

from datetime import datetime, timezone

from turbotab.core.decisions import DecisionRecord, ProjectState, parse_decision
from turbotab.core.interview import QUESTION_KEYS, route

ALL_FRESH = {s: {"status": "fresh"} for s in (
    "ingest", "profile", "target_info", "findings", "roles", "proposals", "cohort", "split",
    "shelf", "design", "fit", "substitution")}
DIET_ROLES = {"participant_id": "identifier", "age": "covariate", "energy_kcal": "energy",
              "protein_g": "exposure", "fat_g": "exposure", "carbohydrate_g": "exposure"}
GENE_ROLES = {"sample_id": "identifier", "age": "covariate", "gene_0001": "exposure", "gene_0002": "exposure"}


def _stages(**overrides):
    out = {k: dict(v) for k, v in ALL_FRESH.items()}
    for name, status in overrides.items():
        out[name] = {"status": status}
    return out


def _by_key(steps):
    return {s.key: s for s in steps}


def _record(seq, decision):
    return DecisionRecord(id=f"r{seq}", seq=seq, at=datetime.now(timezone.utc), decision=parse_decision(decision))


def test_questions_come_in_order_with_exactly_one_open():
    steps = route(ProjectState(), _stages())
    assert [s.key for s in steps] == list(QUESTION_KEYS)
    assert [s.status for s in steps].count("open") == 1 and steps[0].key == "lens" and steps[0].status == "open"
    assert all(s.status == "waiting" and s.waiting_on[0] == "lens" for s in steps[1:])


def test_answered_questions_name_their_record_and_the_next_one_opens():
    records = [_record(1, {"kind": "set_lens", "lenses": ["dietary"]}),
               _record(2, {"kind": "set_target", "column": "hba1c"})]
    state = ProjectState(lens=["dietary"], target="hba1c")
    info = {"column": "hba1c", "task": "regression", "confidence": "medium", "reason": "Continuous values."}
    steps = _by_key(route(state, _stages(), {"target_info": info}, records))
    assert steps["lens"].status == "answered" and steps["lens"].decision_id == "r1"
    assert steps["target"].decision_id == "r2"
    assert steps["task"].status == "open"  # medium confidence is asked
    assert steps["purpose"].waiting_on == ["task"]


def test_a_confident_detection_skips_the_task_with_its_reason():
    state = ProjectState(lens=["dietary"], target="hba1c")
    info = {"column": "hba1c", "task": "regression", "confidence": "high", "reason": "`hba1c` is continuous."}
    steps = _by_key(route(state, _stages(), {"target_info": info}))
    assert steps["task"].status == "skipped" and steps["task"].reason == "`hba1c` is continuous."
    assert steps["purpose"].status == "open"


def test_a_question_waits_on_a_stage_still_computing_but_not_on_one_that_failed():
    state = ProjectState(lens=["dietary"], target="hba1c", task="regression", purpose="prediction")
    running = _by_key(route(state, _stages(roles="running")))
    assert running["roles"].status == "waiting" and running["roles"].waiting_on == ["roles"]
    assert "open" not in {s.status for s in running.values()}
    # idle behind a running dependency counts as on its way
    idle = _by_key(route(state, _stages(profile="running", roles="idle")))
    assert idle["roles"].waiting_on == ["roles"]
    failed = _by_key(route(state, _stages(roles="error")))
    assert failed["roles"].status == "open"


def test_energy_adjustment_applies_under_the_dietary_lens_only():
    base = dict(target="hba1c", task="regression", purpose="prediction", exclusions=[], missing="impute",
                split={"holdout": 0.2})
    diet = _by_key(route(ProjectState(lens=["dietary"], roles=DIET_ROLES, **base), _stages()))
    assert diet["energy_adjustment"].status == "open"
    assert diet["substitution"].status == "waiting"
    genes = _by_key(route(ProjectState(lens=["genomics"], roles=GENE_ROLES, **base), _stages()))
    assert genes["energy_adjustment"].status == "not_applicable"
    assert "dietary lens" in genes["energy_adjustment"].reason
    assert genes["substitution"].status == "not_applicable"
    assert genes["models"].status == "open"  # the next applicable question opens


def test_energy_adjustment_says_which_ingredient_is_missing():
    base = dict(lens=["dietary"], target="y", task="regression", purpose="prediction")
    no_energy = {k: v for k, v in DIET_ROLES.items() if v != "energy"}
    step = _by_key(route(ProjectState(roles=no_energy, **base), _stages()))["energy_adjustment"]
    assert step.status == "not_applicable" and "energy role" in step.reason
    no_bearing = {"energy_kcal": "energy", "sodium_mg": "exposure", "age": "covariate"}
    step = _by_key(route(ProjectState(roles=no_bearing, **base), _stages()))["energy_adjustment"]
    assert step.status == "not_applicable" and "carries energy" in step.reason


def test_substitution_waits_for_a_fresh_fit_and_needs_two_energy_nutrients():
    state = ProjectState(lens=["dietary"], target="hba1c", task="regression", purpose="prediction",
                         roles=DIET_ROLES, exclusions=[], missing="impute", split={"holdout": 0.2},
                         energy_adjustment={"method": "none"}, models=["linear"])
    steps = _by_key(route(state, _stages(fit="stale")))
    assert steps["substitution"].status == "waiting" and steps["substitution"].waiting_on == ["fit"]
    assert _by_key(route(state, _stages()))["substitution"].status == "open"
    one = {"energy_kcal": "energy", "protein_g": "exposure", "age": "covariate"}
    step = _by_key(route(state.model_copy(update={"roles": one}), _stages()))["substitution"]
    assert step.status == "not_applicable" and "two" in step.reason


def test_changing_an_earlier_answer_never_reopens_a_later_one():
    later = ProjectState(lens=["dietary"], target="bmi", purpose="prediction", roles=DIET_ROLES,
                         exclusions=[], missing="impute")  # a new target: its task is unanswered
    steps = _by_key(route(later, _stages(target_info="running")))
    assert steps["task"].status == "waiting" and steps["task"].waiting_on == ["target_info"]
    assert all(steps[k].status == "answered" for k in ("purpose", "roles", "exclusions", "missing"))

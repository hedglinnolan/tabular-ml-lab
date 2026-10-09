"""The causal lane's leash, its Router question and its contracts (``turbotab/core/causal.py``).

Fast checks on the decision path: every refusal carries its reason and a way forward, the Router
asks the question only where MODELING_SEQUENCE §0 ruling 1 rung (d) puts it, and every method enters
through a §13 contract whose relations are named. The estimators themselves are checked against
their independent references in ``acceptance/test_causal_lane.py``.
"""
from __future__ import annotations

from typing import Any

import pytest

from turbotab.core import causal
from turbotab.core import decisions as d
from turbotab.core.decisions import ProjectState, Refusal
from turbotab.core.interview import route

ROLES = {"person_id": "identifier", "age": "covariate", "smoker": "covariate",
         "supplement": "exposure", "ldl": "covariate"}
ANSWERS = {"age": {"causes_exposure": "yes", "causes_outcome": "yes", "after_exposure": "no",
                   "exposure": "supplement"},
           "smoker": {"causes_exposure": "yes", "causes_outcome": "yes", "after_exposure": "no",
                      "exposure": "supplement"},
           "ldl": {"causes_exposure": "no", "causes_outcome": "yes", "after_exposure": "yes",
                   "exposure": "supplement"}}
ALL4 = list(causal.ASSUMPTIONS)
FRESH = {s: {"status": "fresh"} for s in (
    "ingest", "oriented", "profile", "findings", "structure", "working", "target_info", "roles",
    "proposals", "cohort", "split", "shelf", "design", "fit", "substitution", "causal_design")}


def _state(**change: Any) -> ProjectState:
    base = dict(lens=["clinical"], target="sbp", task="regression", purpose="inference",
                grain={"grain": "one_row_per_unit"}, roles=ROLES, exclusions=[],
                missing="complete_case", split={"holdout": 0.0},
                estimand={"exposure": "supplement", "measure": "mean_difference"},
                adjustment=ANSWERS, clusters={"column": None, "acknowledged": True})
    base.update(change)
    return ProjectState(**base)


def _design(**change: Any) -> dict[str, Any]:
    card = {"purpose": "inference", "offered": True, "exposure": "supplement",
            "exposure_kind": "binary", "level": "1", "task": "regression", "n": 1200,
            "n_limit": 1200, "candidates": ["age", "smoker"], "many": False,
            "stated": "the primary model estimates the declared effect.",
            "assumptions": [{"key": a, "label": causal.ASSUMPTION_WORDS[a].capitalize(),
                             "statement": f"{a} statement.", "diagnostic": f"{a} diagnostic.",
                             "status": "stated", "source": causal.HERNAN} for a in ALL4],
            "positivity": {"violated": False, "reason": None},
            "missing": {"n_rows": 1200, "n_complete": 1200, "n_incomplete": 0,
                        "answer": "complete_case", "blocked": False, "reason": None},
            "survey": {"population": False, "weight": None}}
    card.update(change)
    return card


def _ctx(state: ProjectState, design: dict[str, Any] | None = None, task: str = "regression"):
    return {"state": state, "task": task, "artifact": lambda stage: (
        design if stage == "causal_design" else None)}


def _refused(decision: Any, ctx: Any) -> Refusal:
    with pytest.raises(Refusal) as caught:
        d.validate(decision, ctx)
    assert caught.value.exits, f"{caught.value.code} has no way forward"
    return caught.value


def _taken(exit_: dict[str, Any]) -> Any:
    """An exit's decision, as the client would post it back."""
    return d.parse_decision(exit_["decision"])


def _set(**change: Any) -> d.SetCausal:
    body = dict(exposure="supplement", method="tmle", learner="linear", assumptions=ALL4)
    body.update(change)
    return d.SetCausal(**body)


# ── the leash ────────────────────────────────────────────────────────────────


def test_never_under_prediction_and_the_exit_is_the_purpose():
    refusal = _refused(_set(), _ctx(_state(purpose="prediction"), _design()))
    assert refusal.code == "not_inference"
    assert _taken(refusal.exits[0]) == d.SetPurpose(purpose="inference")


def test_only_after_the_exposure_its_effect_and_the_adjustment_set():
    assert _refused(_set(), _ctx(_state(estimand=None), _design())).code == "no_estimand"
    assert _refused(_set(), _ctx(_state(adjustment={}), _design())).code == "no_adjustment"
    other = _refused(_set(exposure="age"), _ctx(_state(), _design()))
    assert other.code == "other_exposure"
    assert _taken(other.exits[0]).exposure == "supplement"
    direct = _refused(_set(), _ctx(_state(estimand={"exposure": "supplement", "effect": "direct",
                                                   "measure": "mean_difference"}), _design()))
    assert direct.code == "outside_the_lane" and "mediation" in direct.message
    assert _taken(direct.exits[0]).method == "none"


def test_the_assumptions_are_declared_before_any_estimate_each_beside_its_diagnostic():
    refusal = _refused(_set(assumptions=["positivity"]), _ctx(_state(), _design()))
    assert refusal.code == "assumptions_first"
    for a in ("no_unmeasured_confounding", "consistency", "time_ordering"):
        assert f"{a} statement. {a} diagnostic." in refusal.message
    assert _taken(refusal.exits[0]).assumptions == ALL4
    assert _taken(refusal.exits[1]).method == "none"
    d.validate(_set(), _ctx(_state(), _design()))  # all four declared: recorded


def test_positivity_is_block_and_record_with_a_stated_trimming_exit():
    design = _design(positivity={"violated": True, "reason": "Rows lie beyond the bound."})
    refusal = _refused(_set(), _ctx(_state(), design))
    assert refusal.code == "positivity" and refusal.message == "Rows lie beyond the bound."
    trim, record = refusal.exits
    assert _taken(trim).trim == causal.TRIM_AT and "overlap population" in trim["label"]
    assert _taken(record).acknowledged is True
    for exit_ in (trim, record):
        d.validate(_taken(exit_), _ctx(_state(), design))  # each exit is taken
    numeric = _design(exposure_kind="continuous",
                      positivity={"violated": True, "reason": "The covariates explain 95%."})
    refusal = _refused(_set(method="dml_plr"), _ctx(_state(), numeric))
    assert [_taken(e).acknowledged for e in refusal.exits] == [True]  # no propensity to trim


def test_the_method_fits_the_declared_estimand():
    numeric = _design(exposure_kind="continuous")
    refusal = _refused(_set(method="tmle"), _ctx(_state(), numeric))
    assert refusal.code == "needs_binary_exposure"
    assert _taken(refusal.exits[0]).method == "dml_plr"
    att = _refused(_set(method="tmle", population="exposed"), _ctx(_state(), _design()))
    assert att.code == "att_needs_irm"
    trimmed = _refused(_set(method="dml_plr", trim=0.1), _ctx(_state(), _design()))
    assert trimmed.code == "trim_needs_propensity"
    binary = _state(task="binary", estimand={"exposure": "supplement", "measure": "odds_ratio"})
    conditional = _refused(_set(), _ctx(binary, _design(task="binary"), task="binary"))
    assert conditional.code == "measure_conditional"
    assert _taken(conditional.exits[0]).measure == "risk_difference"
    pds = _refused(_set(method="pds_lasso"), _ctx(
        _state(task="binary", estimand={"exposure": "supplement", "measure": "risk_difference"}),
        _design(task="binary"), task="binary"))
    assert pds.code == "pds_needs_numeric_outcome"
    spline = _state(exposure_forms={"supplement": {"form": "spline", "knots": 4}})
    refusal = _refused(_set(method="dml_plr"), _ctx(spline, _design(exposure_kind="continuous")))
    assert refusal.code == "form_is_several_terms"
    assert _taken(refusal.exits[0]) == d.SetExposureForm(column="supplement", form="linear")


def test_survey_weights_enter_or_the_method_is_block_and_record():
    population = _state(survey={"estimand": "population", "weight": "w", "strata": "s",
                                "psu": "p"})
    refusal = _refused(_set(method="pds_lasso"), _ctx(population, _design()))
    assert refusal.code == "survey_pds"
    weighted, sample = refusal.exits
    assert _taken(weighted).method == "dml_plr"  # takes the weights in every fit
    assert _taken(sample).sample_only is True
    d.validate(_taken(sample), _ctx(population, _design()))  # recorded as unweighted
    d.validate(_set(method="dml_plr"), _ctx(population, _design()))
    d.validate(_set(method="pds_lasso"), _ctx(_state(), _design()))  # no design: allowed


def test_incomplete_rows_under_imputation_are_block_and_record():
    blocked = _design(missing={"n_rows": 1200, "n_complete": 1000, "n_incomplete": 200,
                               "answer": "multiple_imputation", "blocked": True,
                               "reason": "200 rows miss a value."})
    refusal = _refused(_set(), _ctx(_state(), blocked))
    assert refusal.code == "incomplete_rows"
    assert _taken(refusal.exits[0]).complete_rows is True
    d.validate(_taken(refusal.exits[0]), _ctx(_state(), blocked))


def test_the_record_names_the_learner_the_estimate_uses():
    completed = d.validate(_set(learner=None, method="dml_irm"), _ctx(_state(), _design(n=1200)))
    assert completed.learner == "nuisance_forest"
    small = d.validate(_set(learner=None, method="dml_irm"), _ctx(_state(), _design(n=300)))
    assert small.learner == "lasso"


# ── the Router ───────────────────────────────────────────────────────────────


def _step(state: ProjectState, design: dict[str, Any] | None, records: list[Any] = ()) -> Any:
    artifacts = {"causal_design": design} if design is not None else {}
    return {s.key: s for s in route(state, FRESH, artifacts, records)}["causal"]


def test_the_router_states_the_lane_and_names_what_ranks_first():
    stated = _step(_state(), _design())
    assert stated.status == "skipped"
    assert stated.reason == "the primary model estimates the declared effect."
    many = causal.stated_reason("pds_lasso", True, 140, 1200)
    assert many == ("the primary model estimates the declared effect; with 140 candidate terms "
                    "for a limiting sample size of 1,200 (more than one per 10), post-double-"
                    "selection lasso ranks first and is one step away.")
    assert _step(_state(), _design(many=True, stated=many)).reason == many
    # While the card computes, the lane is stated too: it never holds the models question back.
    computing = _step(_state(), None)
    assert computing.status == "skipped" and computing.reason == causal.STATED
    # (FORM: the form question's card finds no continuous term here, so it does not apply)
    no_forms = {"purpose": "inference", "ready": True, "needs": []}
    models = {s.key: s for s in route(_state(), FRESH, {"forms": no_forms}, [])}["models"]
    assert models.status == "open"
    waiting = _step(_state(adjustment={}), None)  # the adjustment set comes first
    assert waiting.status == "waiting"
    prediction = _step(_state(purpose="prediction", estimand=None, adjustment=None), None)
    assert prediction.status == "not_applicable" and "prediction" in prediction.reason
    family = _step(_state(roles={**ROLES, "age": "exposure"},
                          estimand={"family": True, "measure": "mean_difference"}), _design())
    assert family.status == "not_applicable" and "family" in family.reason
    survival = _step(_state(task="time_to_event", estimand={"exposure": "supplement",
                                                            "measure": "hazard_ratio"}), _design())
    assert survival.status == "not_applicable"


def test_an_answer_for_another_exposure_is_re_asked_never_kept():
    spec = {"exposure": "supplement", "method": "tmle", "assumptions": ALL4}
    assert causal.current_causal(_state(causal=spec)) is not None
    assert _step(_state(causal=spec), _design(many=True)).status == "answered"
    moved = _state(roles={**ROLES, "age": "exposure"}, causal=spec,
                   estimand={"exposure": "age", "measure": "mean_difference"})
    assert causal.current_causal(moved) is None
    assert _step(moved, _design(exposure="age", many=True)).status != "answered"


def test_many_candidates_is_harrells_one_per_ten():
    assert causal.many_candidates(1000, 101) and not causal.many_candidates(1000, 100)
    ranked = causal.options("continuous", "regression", True, False)
    assert ranked[0].key == "pds_lasso" and ranked[0].sound.verdict == "sound"
    few = causal.options("continuous", "regression", False, False)
    assert few[0].key == "dml_plr" and [o.key for o in few][-1] == "none"
    weighted = causal.options("continuous", "regression", True, True)
    assert next(o for o in weighted if o.key == "pds_lasso").sound.verdict == "unsound"
    binary = causal.options("binary", "binary", False, False)
    assert [o.key for o in binary] == ["tmle", "dml_irm", "dml_plr", "none"]
    for options in (ranked, few, weighted, binary):
        for o in options:
            assert o.customary.source and len(o.sound.reason.split()) >= 4


# ── the contracts (BLUEPRINT §13) ────────────────────────────────────────────


def test_every_method_enters_through_a_contract_with_named_relations():
    from turbotab.core import contracts as C

    assert set(causal.CONTRACTS) == set(causal.METHODS)
    registry = C.contracts()  # the one registry (BLUEPRINT §13)
    used = set()
    for contract in causal.CONTRACTS.values():
        assert registry[contract.key] is contract and contract.package == "CAUSAL"
        assert contract.slot == "model" and contract.question == "causal"
        assert contract.scope and contract.needs and contract.storyboard
        assert contract.leash["prediction"] == "refused"
        names = {r.name for r in contract.relations}
        assert names <= set(causal.RELATIONS), contract.key
        used |= names
    assert used == set(causal.RELATIONS)  # every relation belongs to a method
    for relation in causal.RELATIONS.values():
        if relation.kind == "conflicts":
            assert relation.exits, relation.name  # a conflict names its way forward
            assert relation.rung in ("refused", "block_and_record"), relation.name

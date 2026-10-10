"""VAL · selection validators (WAVE_C6A_PLAN §7 ruling 13; INBOX "From C6a phase 2's integration").

* **Purposes.** A family is refused under a goal its declared ``purposes`` exclude
  (MODEL_FAMILY_CONTRACT C2; RECIPES §5: robust linear regression is not offered under inference in
  v2). The validator reads the declaration, never a key: a probe under a key of its own is refused
  the same way.
* **Availability.** A family whose library cannot load here says so through its ``unavailable``
  member, and is refused when chosen, not only when it fits; the shelf keeps it, marked, last.
* **Ranges.** A value set by hand outside its dimension's range is refused by ``tuning.make_plan``
  and by the ``set_tuning`` validator, naming the range (RECIPES §4.1's table: Huber's t in [1, 3]).

Every expectation is written out here by hand: the ranges from RECIPES §4.1, the purposes from
MODEL_FAMILY_CONTRACT §3.1–3.2's tables.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

import turbotab.core.models  # noqa: F401 - registers the families
from turbotab.core.decisions import ProjectState, Refusal, SelectModels, validate
from turbotab.core.models.base import (Situation, get_family, rank, register_family,
                                       unregister_family)
from turbotab.core.models.tuning import Dimension, TuningDecl, make_plan

INFERENCE = {"task": "regression", "purpose": "inference"}
PREDICTION = {"task": "regression", "purpose": "prediction"}


def _refused(models: list[str], ctx: object) -> Refusal:
    with pytest.raises(Refusal) as caught:
        validate(SelectModels(models=models), ctx)
    return caught.value


# ── purposes ─────────────────────────────────────────────────────────────────


def test_robust_linear_is_refused_under_inference_with_a_plain_reason_and_an_exit() -> None:
    refusal = _refused(["linear", "huber"], INFERENCE)
    assert refusal.code == "model_not_for_purpose"
    assert refusal.message == ("Robust linear regression is offered only to predict, not when the "
                               "goal is to estimate an effect.")
    assert refusal.exits[0] == {"label": "Keep the families that can",
                                "decision": {"kind": "select_models", "models": ["linear"]}}
    assert refusal.exits[-1] == {"label": "Choose from the shelf", "decision": None}
    validate(refusal.exits[0]["decision"], INFERENCE)  # the exit is accepted
    validate(SelectModels(models=["linear", "huber"]), PREDICTION)


def test_the_purpose_is_read_from_the_state_when_the_context_names_none() -> None:
    state = ProjectState(target="y", task="regression", purpose="inference")
    assert _refused(["huber"], {"state": state, "task": "regression"}).code == \
        "model_not_for_purpose"
    # alone, it leaves only the shelf
    assert [e["label"] for e in _refused(["huber"], INFERENCE).exits] == ["Choose from the shelf"]
    # no purpose known yet: nothing to check
    validate(SelectModels(models=["huber"]), {"task": "regression"})


def test_the_declaration_is_read_never_the_key() -> None:
    from turbotab.core.models.huber import Huber

    probe = type("Probe", (Huber,), {"key": "robust_probe", "label": "Robust probe"})()
    register_family(probe)
    try:
        refusal = _refused(["robust_probe"], INFERENCE)
        assert refusal.code == "model_not_for_purpose" and "Robust probe" in refusal.message
    finally:
        unregister_family("robust_probe")
    # a family that only tests is offered only to estimate an effect
    refusal = _refused(["featurewise", "linear"], PREDICTION)
    assert refusal.code == "model_not_for_purpose"
    assert refusal.message == ("Feature-wise regression is offered only to estimate an effect, "
                               "not when the goal is to predict.")
    validate(SelectModels(models=["featurewise"]), INFERENCE)


# ── availability ─────────────────────────────────────────────────────────────


@pytest.fixture
def xgboost_cannot_load(monkeypatch: pytest.MonkeyPatch) -> None:
    """``import xgboost`` raises ImportError, as it does when the library or libomp is missing."""
    from turbotab.core.models import xgboost_family

    monkeypatch.setitem(sys.modules, "xgboost", None)
    monkeypatch.delitem(xgboost_family._LOADED, "module", raising=False)


def test_a_family_that_loads_is_available() -> None:
    assert get_family("xgboost").unavailable() is None
    assert get_family("linear").unavailable is None  # declared as None: it adds nothing
    validate(SelectModels(models=["xgboost"]), {"task": "binary", "purpose": "prediction"})


def test_xgboost_that_cannot_load_is_refused_at_selection(xgboost_cannot_load: None) -> None:
    reason = get_family("xgboost").unavailable()
    assert reason is not None and reason.startswith("The xgboost library could not load here")
    ctx = {"task": "binary", "purpose": "prediction"}
    refusal = _refused(["xgboost", "linear"], ctx)
    assert refusal.code == "model_unavailable"
    assert refusal.message.startswith("XGBoost cannot be fit on this computer: The xgboost "
                                      "library could not load here")
    # the family it declares itself the same kind as takes its place
    assert refusal.exits[0] == {
        "label": "Use Boosted trees instead",
        "decision": {"kind": "select_models", "models": ["boosted_trees", "linear"]}}
    assert refusal.exits[1]["decision"]["models"] == ["linear"]
    for e in refusal.exits[:2]:
        validate(e["decision"], ctx)
    # with boosted trees already chosen there is nothing to swap in
    labels = [e["label"] for e in _refused(["boosted_trees", "xgboost"], ctx).exits]
    assert labels == ["Keep the families that can", "Choose from the shelf"]


def test_the_shelf_keeps_an_unavailable_family_marked_and_last(xgboost_cannot_load: None) -> None:
    s = Situation(task="binary", purpose="prediction", n_rows=2000, n_features=5, n_events=600)
    ranked = rank(s)
    keys = [f.key for f, _ in ranked]
    assert "xgboost" in keys and keys[-1] == "xgboost"  # never shortened
    judged = dict((f.key, a) for f, a in ranked)["xgboost"]
    assert judged.fit == "poor"
    assert judged.concerns[0].startswith("Cannot be fit on this computer: The xgboost library")


def test_any_family_declaring_itself_unavailable_is_refused() -> None:
    from turbotab.core.models.ridge import Ridge

    probe = type("Probe", (Ridge,), {"key": "unloadable_probe", "label": "Unloadable probe",
                                     "unavailable": lambda self: "its library is missing"})()
    register_family(probe)
    try:
        refusal = _refused(["unloadable_probe", "ridge"], PREDICTION)
        assert refusal.code == "model_unavailable"
        assert refusal.message == ("Unloadable probe cannot be fit on this computer: its library "
                                   "is missing.")
    finally:
        unregister_family("unloadable_probe")


# ── ranges of values set by hand ─────────────────────────────────────────────


def _dim(name, low, high, scale, **more):
    return Dimension(name, f"plain words for {name}", f"term for {name}", low, high, scale,
                     source="Friedman 2001", **more)


PROBE = TuningDecl(
    "search",
    dimensions=(_dim("max_depth", 2, 10, "int"),
                _dim("criterion", 0, 0, "choice", choices=("squared_error", "poisson")),
                _dim("min_samples_leaf", 0, 0.1, "share_of_units"),
                _dim("l2_regularization", 1e-3, 10, "log")),
    standard={"max_depth": 6, "criterion": "squared_error", "min_samples_leaf": "default",
              "l2_regularization": 0.0},
    space_version="probe_ranges/1")


def _plan(family, manual, mode="manual"):
    return make_plan(family, task="regression", loss="mse", n_plan=400, plan_rows=400,
                     unit="units", split_seed=0, mode=mode, manual=manual)


def test_huber_threshold_set_by_hand_outside_1_to_3_is_refused_by_the_plan() -> None:
    huber = get_family("huber")
    for t in (5.0, 0.5):
        with pytest.raises(ValueError, match=r"set by hand to [\d.]+; it runs from 1 to 3"):
            _plan(huber, {"t": t})
    for t in (1, 2.0, 3):  # the ends are inside
        assert dict(_plan(huber, {"t": t}).candidates[0].values) == {"t": t}


def test_every_scale_is_checked_and_the_standard_value_is_always_allowed() -> None:
    fam = SimpleNamespace(key="probe_ranges", label="Probe", tasks=("regression",), tuning=PROBE,
                          defaults_version="1", settings=None)
    bad = {"max_depth": 11, "criterion": "absolute_error", "min_samples_leaf": 0.2,
           "l2_regularization": 20.0}
    says = {"max_depth": "it runs from 2 to 10", "criterion": "it is one of squared_error, poisson",
            "min_samples_leaf": "it runs from just above 0 to 0.1",
            "l2_regularization": "it runs from 0.001 to 10"}
    for name, value in bad.items():
        with pytest.raises(ValueError, match=says[name]):
            _plan(fam, {name: value}, mode="automatic")  # held values are set by hand too
    for name, value in (("max_depth", 4.5), ("min_samples_leaf", 0.0), ("max_depth", "deep"),
                        ("max_depth", True)):
        with pytest.raises(ValueError, match="set by hand"):
            _plan(fam, {name: value})
    # the standard value is always allowed, even where it sits outside the searched range
    # (scikit-learn's l2_regularization = 0 below the search's 1e-3; a symbolic "default")
    held = _plan(fam, {"l2_regularization": 0.0, "min_samples_leaf": "default", "max_depth": 2,
                       "criterion": "poisson"})
    assert dict(held.candidates[0].values) == {"max_depth": 2, "criterion": "poisson",
                                               "min_samples_leaf": "default",
                                               "l2_regularization": 0.0}


def test_the_set_tuning_validator_refuses_a_value_outside_its_range() -> None:
    """``set_tuning`` (RT-6) is not registered yet; its range check is, and reads the decision's
    ``family``, ``mode`` and ``values`` as RECIPES §3.1 declares them."""
    from turbotab.core.decisions import _VALIDATORS, _tuning_values_in_range

    assert _tuning_values_in_range in _VALIDATORS["set_tuning"]
    decision = SimpleNamespace(kind="set_tuning", family="huber", mode="manual", values={"t": 5})
    with pytest.raises(Refusal) as caught:
        _tuning_values_in_range(decision, {"task": "regression"})
    assert caught.value.code == "tuning_out_of_range"
    assert caught.value.message == ("How far a row may sit before it counts less (Huber "
                                    "threshold) was set by hand to 5; it runs from 1 to 3.")
    assert [e["label"] for e in caught.value.exits] == ["Set a value from 1 to 3",
                                                        "Keep the standard setting"]
    # inside the range, an unknown family (refused elsewhere) and no task: nothing to refuse here
    _tuning_values_in_range(SimpleNamespace(family="huber", mode="manual", values={"t": 2}),
                            {"task": "regression"})
    _tuning_values_in_range(SimpleNamespace(family="nope", mode="manual", values={"t": 9}), None)
    with pytest.raises(Refusal):
        _tuning_values_in_range(decision, None)  # no task known: every declaration is read

"""Feature-wise regression with no study factor to test is refused in words, never a crash (the
swarm's C2: five of Sullivan 2021's walks ended ``stage_failed`` at the fit, and one at the effects
stage, on ``ValueError: No study factor to test``).

The walks' plans gave no column the role of what you study: the roles answer named `DBQ700`, the
estimand's study factor, a covariate, and recorded it as codes. The family tests the columns with
the role of what you study (``FeatureWise.build_for``), so it had nothing to test. Where the
engine knows that when the models are chosen (no column holds the role), the models answer is
refused there, its ways forward the roles question and the other families chosen; where it learns
it only at the fit (no column of the model matrix bears a study factor's name: a category enters
as its indicators), the fit and the effects stage refuse in the same words, coded. The guard stays
silent on a study factor the matrix holds, whose estimate is least squares' (NumPy's ``lstsq``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import turbotab.core.models  # noqa: F401 - registers the families
from turbotab.core import decisions as d
from turbotab.core.graph import StageRefusal
from turbotab.core.jobs import failure_of
from turbotab.core.models import featurewise as fw

WORDS = ("No study factor to test: feature-wise regression tests the columns with the role of what "
         "you study, one at a time, adjusted for the rest")
ROLES_WAY = "Give the columns you study the role of what you study (the roles question)"
ANOTHER = "Choose another model family"


def _matrix(n: int = 40, seed: int = 3) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"x": rng.normal(size=n), "age": rng.normal(50, 10, n),
                          "DBQ700_2": rng.integers(0, 2, n).astype(float),
                          "DBQ700_3": rng.integers(0, 2, n).astype(float)})
    y = 1.5 * frame["x"].to_numpy() + 0.1 * frame["age"].to_numpy() + rng.normal(size=n)
    return frame, y


# ── at the fit: the model matrix holds no study factor ───────────────────────


def test_no_column_with_the_role_is_a_typed_refusal_at_the_fit():
    frame, y = _matrix()
    with pytest.raises(StageRefusal) as refused:
        fw.featurewise_table(frame[["age"]], y, "regression", [])
    exc = refused.value
    assert isinstance(exc, ValueError) and exc.code == "featurewise_no_study_factor"
    assert str(exc) == f"{WORDS}; no column has that role."
    assert [e["label"] for e in exc.exits] == [ROLES_WAY, ANOTHER]
    shown, refusal = failure_of(exc)
    assert shown == f"ValueError: {WORDS}; no column has that role."
    assert refusal is not None and refusal["code"] == "featurewise_no_study_factor"


def test_a_study_factor_the_matrix_holds_no_column_of_is_refused_in_the_same_words():
    """`DBQ700` with the role, entered as its indicators: no column bears its name."""
    frame, y = _matrix()
    with pytest.raises(StageRefusal) as refused:
        fw.featurewise_table(frame[["age", "DBQ700_2", "DBQ700_3"]], y, "regression", ["DBQ700"])
    assert refused.value.code == "featurewise_no_study_factor"
    assert str(refused.value) == (f"{WORDS}; `DBQ700` has that role, but the model matrix holds no "
                                  f"column of that name.")
    assert [e["label"] for e in refused.value.exits] == [ANOTHER]


def test_the_model_step_refuses_the_same_way():
    frame, y = _matrix()
    with pytest.raises(StageRefusal) as refused:
        fw.FeatureWiseTests(task="regression", features=[]).fit(frame[["age"]], y)
    assert refused.value.code == "featurewise_no_study_factor"


def test_the_fit_stays_silent_on_a_study_factor_the_matrix_holds():
    """`x` tested, adjusted for `age`: its estimate is least squares' coefficient (NumPy)."""
    frame, y = _matrix()
    table = fw.featurewise_table(frame[["x", "age"]], y, "regression", ["x"])
    (row,) = [r for r in table.rows if r["feature"] == "x"]
    design = np.column_stack([np.ones(len(y)), frame["x"], frame["age"]])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    assert row["estimate"] == pytest.approx(beta[1], rel=1e-10)


# ── at the models answer: no column holds the role ───────────────────────────


def _state(roles: dict[str, str]) -> d.ProjectState:
    return d.ProjectState(target="glucose", task="regression", purpose="inference",
                          roles=dict(roles))


SULLIVAN = {"DBQ700": "covariate", "age": "covariate", "kcal": "energy", "SEQN": "identifier"}


def test_choosing_it_where_no_column_has_the_role_is_refused_with_the_roles_way():
    with pytest.raises(d.Refusal) as refused:
        d.validate(d.SelectModels(models=["featurewise"]), {"state": _state(SULLIVAN),
                                                            "task": "regression"})
    assert refused.value.code == "featurewise_no_study_factor"
    assert refused.value.message == f"{WORDS}; no column has that role."
    assert refused.value.exits == [{"label": ROLES_WAY, "decision": None},
                                   {"label": ANOTHER, "decision": None}]


def test_beside_other_families_its_way_forward_keeps_them():
    ctx = {"state": _state(SULLIVAN), "task": "regression"}
    with pytest.raises(d.Refusal) as refused:
        d.validate(d.SelectModels(models=["linear", "featurewise"]), ctx)
    keep = refused.value.exits[1]
    assert keep == {"label": "Fit the other family chosen, without feature-wise regression",
                    "decision": {"kind": "select_models", "models": ["linear"]}}
    d.validate(keep["decision"], ctx)  # the way forward is accepted


@pytest.mark.parametrize("roles", [
    {**SULLIVAN, "DBQ700": "exposure"},  # a study factor
    {**SULLIVAN, "mz_1": "exposure", "mz_2": "exposure"},
], ids=["one", "several"])
def test_choosing_it_where_a_column_has_the_role_is_not_refused_for_it(roles):
    d.validate(d.SelectModels(models=["featurewise"]), {"state": _state(roles),
                                                        "task": "regression"})


def test_with_no_roles_answered_yet_the_models_answer_is_not_refused_for_it():
    state = d.ProjectState(target="glucose", task="regression", purpose="inference")
    d.validate(d.SelectModels(models=["featurewise"]), {"state": state, "task": "regression"})

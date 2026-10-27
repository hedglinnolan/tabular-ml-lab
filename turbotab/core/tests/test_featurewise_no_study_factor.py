"""Feature-wise regression with no study factor to test is refused in words, never a crash (the
swarm's C2: five of Sullivan 2021's walks ended ``stage_failed`` at the fit, and one at the effects
stage, on ``ValueError: No study factor to test``).

The walks' plans gave no column the role of what you study: the roles answer named `DBQ700`, the
estimand's study factor, a covariate, and recorded it as codes. The family tests the columns with
the role of what you study (``FeatureWise.build_for``), so it had nothing to test. Where the
engine knows that when the models are chosen (no column holds the role), the models answer is
refused there, its ways forward the roles question and the other families chosen; where it learns
it only at the fit (no column of the model matrix bears a study factor's name: a category enters
as its indicators), feature-wise regression's table is refused in the same words at the fit and
the effects stage, and every family chosen beside it is fit (the orchestrator's ruling (b) on the
swarm's tail: the fit stage refused for them all; ``test_featurewise_table_refusals_at_the_stages``
runs the stages). The guard stays silent on a study factor the matrix holds, whose estimate is
least squares' (NumPy's ``lstsq``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import turbotab.core.models  # noqa: F401 - registers the families
from turbotab.core import decisions as d
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
# The table is refused, not the fit (the orchestrator's ruling (b) on the swarm's tail): no row,
# so no number; the words its reason, its caption's (with no rows it was estimated from) and its
# first concern; its ways forward; and the rows it was handed.


def _refused(table, words: str, labels: list[str], n_rows: int) -> None:
    assert table.rows == []
    assert table.info["refused"] == words
    assert table.info["caption"] == f"No tests: {words}"
    assert [e["label"] for e in table.info["exits"]] == labels
    assert table.info["n_rows"] == n_rows
    assert table.concerns[0] == words


def test_no_column_with_the_role_refuses_the_table_at_the_fit():
    frame, y = _matrix()
    table = fw.featurewise_table(frame[["age"]], y, "regression", [])
    _refused(table, f"{WORDS}; no column has that role.", [ROLES_WAY, ANOTHER], len(frame))


def test_a_study_factor_the_matrix_holds_no_column_of_is_refused_in_the_same_words():
    """`DBQ700` with the role, entered as its indicators: no column bears its name."""
    frame, y = _matrix()
    table = fw.featurewise_table(frame[["age", "DBQ700_2", "DBQ700_3"]], y, "regression",
                                 ["DBQ700"])
    _refused(table, (f"{WORDS}; `DBQ700` has that role, but the model matrix holds no column of "
                     f"that name."), [ANOTHER], len(frame))


def test_the_model_step_refuses_its_table_the_same_way_and_is_fit():
    frame, y = _matrix()
    step = fw.FeatureWiseTests(task="regression", features=[]).fit(frame[["age"]], y)
    _refused(step.table_, f"{WORDS}; no column has that role.", [ROLES_WAY, ANOTHER], len(frame))


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


# ── under an energy adjustment: the study factor enters under the energy step's name ─────────
# The energy step replaces an adjusted nutrient (the residual method's `sugar_adj`, a density's
# `sugar_per_kcal`, a partition's `kcal_from_protein_g`), and the family tests that column: it is
# the study factor as the model holds it. Each estimate is least squares' on columns NumPy builds
# from the raw ones, by each method's definition: the residual `sugar − b(kcal − mean kcal)`, b
# the slope of `sugar` on `kcal`; the density `sugar / kcal`; the partition `4 × protein_g` (the
# general Atwater factor for protein, kcal per gram) beside the rest of the energy.


def _intake(n: int = 200, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    kcal = rng.normal(2000, 300, n)
    frame = pd.DataFrame({"sugar": 0.03 * kcal + rng.normal(0, 10, n), "kcal": kcal,
                          "age": rng.normal(50, 10, n),
                          "group": rng.choice(["a", "b", "c"], n)})
    frame["protein_g"] = 0.02 * kcal + rng.normal(0, 10, n)
    y = (0.5 * frame["sugar"] + 0.3 * frame["protein_g"] + 0.1 * frame["age"]
         + rng.normal(0, 5, n)).to_numpy()
    return frame, y


def _fit(frame: pd.DataFrame, y: np.ndarray, roles: dict[str, str], energy: dict | None) -> object:
    from turbotab.core.models.pipeline import DesignSpec, build_pipeline

    predictors = list(roles)
    categorical = [c for c in predictors if frame[c].dtype == object]
    spec = DesignSpec(predictors=predictors, inputs=predictors, categorical=categorical,
                      numeric=[c for c in predictors if c not in categorical], energy=energy,
                      impute=False, roles=dict(roles))
    pipeline = build_pipeline(spec, fw.FEATUREWISE, "regression", "inference", len(frame),
                              len(predictors))
    pipeline.fit(frame[predictors], y)
    return fw.FEATUREWISE.inference(pipeline, frame[predictors], y, task="regression",
                                    clusters=None)


def _least_squares(y: np.ndarray, *columns: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(y)), *columns])
    return np.linalg.lstsq(design, y, rcond=None)[0]


SUGAR = {"sugar": "exposure", "kcal": "energy", "age": "covariate"}


def _residual(frame: pd.DataFrame) -> np.ndarray:
    kcal, sugar = frame["kcal"].to_numpy(), frame["sugar"].to_numpy()
    slope = np.polyfit(kcal, sugar, 1)[0]
    return sugar - slope * (kcal - kcal.mean())


@pytest.mark.parametrize("method,roles,tested,columns", [
    ("residual", SUGAR, "sugar_adj", lambda f: [_residual(f), f["kcal"], f["age"]]),
    ("residual_energy_dropped", SUGAR, "sugar_adj", lambda f: [_residual(f), f["age"]]),
    ("density", SUGAR, "sugar_per_kcal", lambda f: [f["sugar"] / f["kcal"], f["age"]]),
    ("density_multivariate", SUGAR, "sugar_per_kcal",
     lambda f: [f["sugar"] / f["kcal"], f["kcal"], f["age"]]),
    ("partition", {"protein_g": "exposure", "kcal": "energy", "age": "covariate"},
     "kcal_from_protein_g",
     lambda f: [4 * f["protein_g"], f["kcal"] - 4 * f["protein_g"], f["age"]]),
], ids=["residual", "residual_energy_dropped", "density", "density_multivariate", "partition"])
def test_an_energy_adjusted_study_factor_is_tested_under_the_energy_steps_name(method, roles,
                                                                                 tested, columns):
    """The tested column first among ``columns``, the adjustment columns after it."""
    frame, y = _intake()
    nutrient = next(c for c, r in roles.items() if r == "exposure")
    table = _fit(frame, y, roles, {"method": method, "energy_column": "kcal",
                                   "nutrients": [nutrient]})
    assert [r["feature"] for r in table.rows] == [tested]
    beta = _least_squares(y, *[np.asarray(c, dtype=float) for c in columns(frame)])
    assert table.rows[0]["estimate"] == pytest.approx(beta[1], rel=1e-9)


def test_the_residual_methods_estimate_is_the_nutrients_in_the_standard_model():
    """Frisch–Waugh: with `kcal` in the model, `sugar_adj` (sugar less a multiple of kcal) carries
    the coefficient `sugar` carries beside `kcal`, as the energy step's sources say."""
    frame, y = _intake()
    adjusted = _fit(frame, y, SUGAR, {"method": "residual", "energy_column": "kcal",
                                      "nutrients": ["sugar"]})
    standard = _fit(frame, y, SUGAR, {"method": "standard", "energy_column": "kcal",
                                      "nutrients": ["sugar"]})
    assert [r["feature"] for r in standard.rows] == ["sugar"]
    beta = _least_squares(y, frame["sugar"], frame["kcal"], frame["age"])
    assert standard.rows[0]["estimate"] == pytest.approx(beta[1], rel=1e-9)
    assert adjusted.rows[0]["estimate"] == pytest.approx(beta[1], rel=1e-9)


def test_a_coded_study_factor_beside_an_energy_adjustment_is_still_refused():
    """`group` has the role, enters as its indicators, and is no nutrient the energy step adjusts."""
    frame, y = _intake()
    table = _fit(frame, y, {"group": "exposure", "sugar": "covariate", "kcal": "energy",
                            "age": "covariate"},
                 {"method": "residual", "energy_column": "kcal", "nutrients": ["sugar"]})
    _refused(table, f"{WORDS}; `group` has that role, but the model matrix holds no column of that "
                    f"name.", [ANOTHER], len(frame))

"""Under Estimate, the declared models (``effects``) and the model further adjusted beside the
primary (``secondary``) are refit only for a family with a coefficient table. Where no chosen
family has one (elastic net, ridge, boosted trees, random forest, XGBoost: the crash sweep's
cases on the walking_bp table) the stage is refused, typed as a fit-time refusal (SWARM_PLAN §8)
with its code and ways forward, so the record classes it as refused, not a crash
(``jobs.failure_of``), and with the words a person reads unchanged.

Expected values come from the construction: the families registered for a numeric outcome whose
declarations give a coefficient table under inference are the linear model, the feature-wise
regression, the random-intercept mixed model and GEE (``models/linear.py``, ``featurewise.py``,
``repeated.py``), the linear model first; each way forward adds one of them to the chosen
families. With the first one taken, the stage fits the linear model.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import turbotab.core.models  # noqa: F401 - registers the families
from turbotab.core.decisions import parse_decision
from turbotab.core.graph import StageRefusal
from turbotab.core.jobs import failure_of
from turbotab.core.tests.acceptance import estimand_fixtures as ef

# Read off the family declarations by hand, in their registration order (the shelf's).
WITH_A_TABLE = ["linear", "featurewise", "mixed", "gee"]
LABELS = {"linear": "Linear model", "featurewise": "Feature-wise regression",
          "mixed": "Random-intercept mixed model", "gee": "GEE (exchangeable)"}
EFFECTS_WORDS = ("None of the chosen model families has a coefficient table, so there is no "
                 "effect to report; choose the linear model.")
SECONDARY_WORDS = ("None of the chosen model families has a coefficient table, so there is no "
                   "estimate to set beside the primary's; choose the linear model.")


def _run(tmp_path: Path, models: list[str], *, secondary: bool) -> dict:
    """``ef.cohort(300)`` under Estimate (the mean difference in `glucose` per unit of `fiber`,
    `bmi` of unknown timing, so a secondary model is declared beside the primary), the chosen
    ``models``: design, then the effects stage, or the secondary stage."""
    from turbotab.core.stages.effects import effects_stage
    from turbotab.core.stages.modeling import design_stage
    from turbotab.core.stages.secondary import secondary_stage
    from turbotab.core.tests import modeling_fixtures as mf

    frame = ef.cohort(300, seed=5)
    st = ef.state(target="glucose", task="regression", measure="mean_difference", models=models)
    paths = mf.ingest_frame(frame, tmp_path)
    split = mf.split_bundle(np.arange(len(frame)), holdout=0.0, seed=st.split.seed)
    info = mf.target_info(st.task, st.target)
    design = design_stage(mf.context(st, {"split": split, "target_info": info}, paths))
    stage = secondary_stage if secondary else effects_stage
    inputs = {"design": design, "split": split, "target_info": info}
    return stage(mf.context(st, inputs, paths)).data


def _check_refusal(exc: StageRefusal, words: str, chosen: list[str]) -> None:
    assert isinstance(exc, ValueError)
    shown, refusal = failure_of(exc)
    assert shown == f"ValueError: {words}"
    assert refusal is not None and refusal["code"] == "no_coefficient_table"
    assert [e["label"] for e in refusal["exits"]] == [
        f"Add {LABELS[k]}, which has a coefficient table" for k in WITH_A_TABLE]
    for key, way in zip(WITH_A_TABLE, refusal["exits"]):
        decision = parse_decision(way["decision"])
        assert decision.kind == "select_models" and decision.models == [*chosen, key]


@pytest.mark.parametrize("family", ["elastic_net", "random_forest"])
def test_the_declared_models_with_no_family_that_has_a_table_are_refused_typed(tmp_path, family):
    """Elastic net reports shrunk coefficients and no inference table; a random forest reports
    none: under Estimate the effects stage has nothing to refit, and refuses in words, typed, with
    a way forward for each family that has a table, where it raised a bare ``ValueError``."""
    with pytest.raises(StageRefusal) as refused:
        _run(tmp_path, [family], secondary=False)
    _check_refusal(refused.value, EFFECTS_WORDS, [family])


@pytest.mark.parametrize("family", ["elastic_net", "random_forest"])
def test_the_secondary_model_with_no_family_that_has_a_table_is_refused_typed(tmp_path, family):
    """The model further adjusted for `bmi` beside the primary: the same refusal, in its words."""
    with pytest.raises(StageRefusal) as refused:
        _run(tmp_path, [family], secondary=True)
    _check_refusal(refused.value, SECONDARY_WORDS, [family])


def test_with_the_first_way_forward_taken_the_linear_model_is_refit(tmp_path):
    """The first way forward adds the linear model to elastic net: the effects stage refits the
    linear model's declared models (Model 2 reports the exposure's estimate), and the secondary
    stage fits the linear model with and without `bmi`; elastic net takes no part in either."""
    effects = _run(tmp_path / "effects", ["elastic_net", "linear"], secondary=False)
    assert [f["family"] for f in effects["families"]] == ["linear"]
    model_2 = ef.sequence(effects)["model_2"]
    assert [r["feature"] for r in model_2["effects"]] == ["fiber"]
    secondary = _run(tmp_path / "secondary", ["elastic_net", "linear"], secondary=True)
    [linear] = secondary["families"]
    assert linear["family"] == "linear"
    assert [[r["feature"] for r in f["coefficients"]] for f in linear["fits"]] == [["fiber"],
                                                                                   ["fiber"]]

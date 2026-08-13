"""`MODELS-025` — a model the shelf offers must complete a training run.

## The absence this file is named for

Four of the twenty-two registry models — `glm`, `huber`, `rf` and `nn`, every
`BaseModelWrapper` subclass — **could not complete a training run at all**, and
had not since `scikit-learn` 1.6 made the tags API required for an estimator
used inside a `Pipeline`. `training.train(project, ["glm"])` returned a result
carrying::

    'GLMWrapper' object has no attribute '__sklearn_tags__'

`PRODUCT_VISION.md` says the shelf is never shortened: every model stays
selectable and the judgment is carried by order and a stated concern. So those
four sat on the shelf, were ranked, carried an evidence-bearing clause about a
table this shape — and errored the moment a user fitted one.

**The suite was green over it, and that is the part worth a file of its own.**
No test in `turbotab/` or `tests/` fitted `glm`, `huber` or `rf`, and none
asserted that a run came back with no errored results. 2,449 passing tests said
nothing about four unusable models. **A defect survives exactly as long as
nothing asks the question**, so this asks it — for every model the environment
can fit, not for the four that were broken, because the next one to break will
be a different one.

## Why it is a sweep and not four cases

Naming the four would encode today's breakage as the test's subject. The claim
is not *these four work*; it is **every model this registry offers can be
fitted**, which is what "the shelf is never shortened" means operationally. A
model that cannot be fitted is shortened from the shelf by the engine rather
than by the app, and nothing on screen says so.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.model_registry import get_registry                          # noqa: E402
from turbotab import eventfixture                                   # noqa: E402
from turbotab import training as T                                  # noqa: E402
from turbotab.project import AnalysisProject                        # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data")

#: Models this environment cannot fit, each with the reason. `nn` needs `torch`,
#: which `TEST-038` records as deliberately absent — a lean install runs the
#: whole app without it. **An entry here is an argument, not a keyword**, and the
#: set is ASSERTED below rather than trusted, so a model quietly joining it is a
#: failure rather than a shrug.
CANNOT_FIT_HERE = {
    "nn": "needs torch, which TEST-038 records as deliberately not installed",
}


def _sealed(fixture: str, target: str, task: str) -> AnalysisProject:
    df = pd.read_csv(os.path.join(DATA, fixture))
    df = df[df[target].notna()].copy()
    project = AnalysisProject.from_dataframe(df, fixture)
    project.target, project.task_type = target, task
    project.set_grain("not_sure")
    project.set_eligibility("everyone")
    rng = np.random.default_rng(42)
    idx = list(project.df.index)
    rng.shuffle(idx)
    project.seal_lockbox(idx[:int(round(len(idx) * 0.25))],
                         fraction=len(idx[:int(round(len(idx) * 0.25))]) / len(idx))
    # `DRIVE-041`. Every model on the shelf has to reach a fit, and after
    # `L60-A` a classification fit refuses until the event is on the record.
    eventfixture.choose_event(project, required=(task == "classification"))
    return project


#: `GUIDED-097`: two fixtures of different target shape. A classifier and a
#: regressor exercise different halves of the registry, and a model that fits
#: one and not the other would pass a single-shape sweep.
SHAPES = {
    "binary_classification": ("clinical_risk.csv", "readmit_30d", "classification"),
    "continuous_regression": ("clinical_risk.csv", "length_of_stay_days", "regression"),
}


def test_the_environment_can_fit_everything_except_the_named_exemption():
    """The exemption list is checked before it is relied on.

    Without this, a model becoming unfittable could be absorbed by widening
    `CANNOT_FIT_HERE` and the sweep below would go on passing over a smaller
    denominator wearing the same name.
    """
    registry = get_registry()
    assert set(CANNOT_FIT_HERE) < set(registry), (
        f"the exemption names a key that is not in the registry: "
        f"{sorted(set(CANNOT_FIT_HERE) - set(registry))}")
    thin = [k for k, why in CANNOT_FIT_HERE.items() if len(why) < 40]
    assert not thin, f"{thin} claim an exemption without giving a reason"
    assert len(CANNOT_FIT_HERE) == 1, (
        f"{len(CANNOT_FIT_HERE)} models are exempt from being fittable. Each "
        f"one is a model on a shelf PRODUCT_VISION says is never shortened, "
        f"and the list growing is the defect MODELS-025 is about.")


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_every_model_the_task_can_use_completes_a_training_run(shape):
    """`MODELS-025`. The question nothing was asking.

    Not *does it fit well* — that is the coach's job and the shelf renders it
    as order. This asks whether the run comes back **without an error**, which
    is the difference between a model a user may reasonably choose and a model
    that raises when they do.
    """
    fixture, target, task = SHAPES[shape]
    project = _sealed(fixture, target, task)
    registry = get_registry()
    supports = f"supports_{task}"
    candidates = sorted(k for k, spec in registry.items()
                        if getattr(spec.capabilities, supports, False)
                        and k not in CANNOT_FIT_HERE)

    # POSITIVE CONTROL — the registry actually offers models for this task, so
    # an empty failure list means "all of them ran" rather than "none were
    # tried".
    assert len(candidates) >= 5, (
        f"{shape}: only {len(candidates)} models can be fitted for this task; "
        f"the sweep has almost no subject")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = T.train(project, candidates)

    ran = {r.key for r in run.results}
    assert ran == set(candidates), (
        f"{shape}: asked for {sorted(candidates)} and the run reports "
        f"{sorted(ran)} — a model was dropped before it was even attempted")

    broken = {r.key: (r.error or "")[:160] for r in run.results if r.error}
    assert not broken, (
        f"{shape}: {len(broken)} of {len(candidates)} models on the shelf "
        f"cannot complete a training run:\n  "
        + "\n  ".join(f"{k}: {v}" for k, v in sorted(broken.items())) +
        "\n\nPRODUCT_VISION says the shelf is never shortened — a model that "
        "errors when fitted is shortened by the engine rather than by the app, "
        "and nothing on screen says so. MODELS-025.")

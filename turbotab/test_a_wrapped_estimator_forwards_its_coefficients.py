"""`GUIDED-234` — the two models named for coefficients now have some.

`GLMWrapper` and `HuberGLMWrapper` hold a `LinearRegression` /
`LogisticRegression` / `HuberRegressor` in `self.model` and forwarded none of
its parameters. Both are registered `interpretability_tier="high"`,
`supports_shap="linear"`, with notes reading *"Interpretable"* — and both were
invisible to the one check that decides whether §A4.7's coefficient forest plot
can be drawn, `hasattr(estimator, "coef_")` in
`turbotab.figure_bundle._coefficients_for`.

## Forwarded on the BASE class, and correct in both directions

The accessor is a property on `BaseModelWrapper` rather than two copies on two
wrappers. **A property that raises `AttributeError` is the mechanism, not an
oversight**: `hasattr` answers `False` when the access raises, so a wrapper
around a Random Forest keeps saying *no coefficients* — which is true — while
the linear ones start saying yes, and a wrapper added later is right without
anyone remembering to declare anything.

## The capability is RE-MEASURED, never edited

`ModelCapabilities.exposes_coefficients` means exactly what
`_coefficients_for` asks, and its whole value is that it is measured against a
real fit. `glm` and `huber` flipped to `True` because the measurement changed,
not because an opinion did — `turbotab/test_the_shelf_reads_the_recorded_design
.py::test_the_declared_capability_matches_a_real_fit` is the check that forces
that order, and it fails if a declaration is edited ahead of the behavior.

## What this file CANNOT show, and the reason is a different defect

**The forest plot still does not draw for these two models**, and not because of
anything here: `MODELS-025` — every `BaseModelWrapper` subclass fails to
complete a training run under `scikit-learn` 1.9, so `glm`, `huber`, `rf` and
`nn` come back with `result.error` set and `_coefficients_for` skips any errored
result before it ever asks about coefficients. So the consequence is
demonstrated **at the pipeline**, which is exactly where `_coefficients_for`
looks, and the end-to-end figure is named as blocked rather than claimed.
`GUIDED-234` stays `PARTIAL` for that reason: its `item` says these are two
models *the coefficient figure cannot draw*, and that sentence is still true.
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
from models.glm import GLMWrapper                                   # noqa: E402
from models.huber_glm import HuberGLMWrapper                        # noqa: E402
from models.rf import RFWrapper                                     # noqa: E402


def _table(n: int = 60):
    rng = np.random.default_rng(11)
    X = pd.DataFrame({f"x{i}": rng.normal(size=n) for i in range(4)})
    return (X,
            pd.Series(X["x0"] * 2 + rng.normal(size=n)),
            pd.Series((X["x0"] > 0).astype(int)))


def _fit(wrapper, X, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wrapper.fit(X.values, y.values)
    return wrapper


# ═══════════ 1 · the forwarding, both directions ════════════════════════════

@pytest.mark.parametrize("make,task", [
    (lambda: GLMWrapper(task_type="regression"), "regression"),
    (lambda: GLMWrapper(task_type="classification"), "classification"),
    (lambda: HuberGLMWrapper(), "regression"),
], ids=["glm-regression", "glm-classification", "huber"])
def test_a_linear_wrapper_forwards_the_estimators_own_coefficients(make, task):
    """Equality with the wrapped estimator, not merely presence.

    A property that returned something of the right shape and the wrong values
    would satisfy `hasattr` and put a wrong forest plot on screen, which is a
    worse outcome than the absence this row is about.
    """
    X, y_reg, y_clf = _table()
    wrapper = _fit(make(), X, y_reg if task == "regression" else y_clf)

    assert hasattr(wrapper, "coef_"), (
        "the wrapper exposes no `coef_`, so `figure_bundle._coefficients_for` "
        "skips it and §A4.7's figure cannot be drawn for a model the registry "
        "calls interpretable. GUIDED-234.")
    assert np.allclose(np.ravel(np.asarray(wrapper.coef_, dtype=float)),
                       np.ravel(np.asarray(wrapper.model.coef_, dtype=float))), (
        "the forwarded coefficients are not the wrapped estimator's")
    assert hasattr(wrapper, "intercept_"), (
        "`coef_` without `intercept_` is a plot of slopes with no statement of "
        "where the line crosses; `reporting_checklist` asks for both")
    assert np.allclose(np.ravel(np.asarray(wrapper.intercept_, dtype=float)),
                       np.ravel(np.asarray(wrapper.model.intercept_, dtype=float)))


def test_a_wrapper_around_a_model_with_no_coefficients_still_says_so():
    """The other direction, and it is what makes the accessor safe on the BASE.

    Putting the property on `BaseModelWrapper` gives it to every wrapper. If it
    answered for a Random Forest, the figure would be offered for a model that
    has nothing to put in it — which is the same false claim as the absence,
    with the sign flipped.
    """
    X, y_reg, _ = _table()
    forest = _fit(RFWrapper(n_estimators=10, task_type="regression"), X, y_reg)

    assert not hasattr(forest, "coef_"), (
        f"a Random Forest wrapper reports coefficients: "
        f"{getattr(forest, 'coef_', None)!r}")
    assert not hasattr(forest, "intercept_")


def test_an_unfitted_wrapper_has_no_coefficients_to_report():
    """There are none before a fit, and reporting some would invent them."""
    assert not hasattr(GLMWrapper(task_type="regression"), "coef_")
    assert not hasattr(HuberGLMWrapper(), "coef_")


# ═══════════ 2 · the registry declaration follows the measurement ═══════════

def test_the_two_flipped_declarations_match_a_real_fit():
    """`exposes_coefficients` is re-measured, never edited.

    Named specifically rather than left to the registry-wide check, because
    these two are the ones this loop changed and the direction matters: the
    declaration flipped BECAUSE the fit changed.
    """
    X, y_reg, y_clf = _table()
    registry = get_registry()
    for key, task in (("glm", "classification"), ("huber", "regression")):
        spec = registry[key]
        estimator = _fit(spec.factory(task, 0),
                         X, y_clf if task == "classification" else y_reg)
        measured = hasattr(estimator, "coef_")
        assert measured is True, (
            f"{key} does not expose coefficients when fitted, so its "
            f"declaration is now ahead of its behavior")
        assert spec.capabilities.exposes_coefficients is True, (
            f"{key} exposes coefficients and the registry still says it does "
            f"not")


# ═══════════ 3 · the consequence, at the place that reads it ════════════════

def test_the_fitted_pipeline_step_is_what_the_figure_asks_about():
    """`_coefficients_for` reads `pipe.named_steps["model"]` — so does this.

    **This is the furthest the consequence can be driven today**, and the
    reason is `MODELS-025`: every `BaseModelWrapper` subclass fails to complete
    a training run under scikit-learn 1.9, so a real project's `glm` result
    carries an error and `_coefficients_for` skips it before asking about
    coefficients at all. Asserting the figure draws would be asserting
    something false; asserting the predicate it turns on is asserting what
    changed.
    """
    from turbotab import figure_bundle as FB
    from turbotab import pipeline_plan as PP
    from turbotab.project import AnalysisProject
    from turbotab import training as T

    df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "sample_data", "clinical_risk.csv"))
    df = df[df["readmit_30d"].notna()].copy()
    project = AnalysisProject.from_dataframe(df, "clinical_risk.csv")
    project.target, project.task_type = "readmit_30d", "classification"
    project.set_grain("not_sure")
    project.set_eligibility("everyone")
    rng = np.random.default_rng(42)
    idx = list(project.df.index)
    rng.shuffle(idx)
    project.seal_lockbox(idx[:int(round(len(idx) * 0.25))], fraction=0.25)

    rows = project.training_rows
    rows = rows[rows["readmit_30d"].notna()]
    X = T.feature_frame(project, rows)
    spec = get_registry()["glm"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe = PP.compose(project, "glm", X, seed=42).build(
            spec.factory("classification", 42))
        pipe.fit(X, rows["readmit_30d"])

    estimator = pipe.named_steps.get("model")
    assert estimator is not None, (
        "the composed pipeline has no `model` step, so this test is not about "
        "the thing `_coefficients_for` reads")
    assert hasattr(estimator, "coef_"), (
        "the fitted pipeline's model step exposes no coefficients, which is "
        "the exact predicate `figure_bundle._coefficients_for` turns on")
    coefficients = np.ravel(np.asarray(estimator.coef_, dtype=float))
    assert coefficients.size, "the estimator exposes an empty coefficient vector"

    # AND THE BLOCKER IS ASSERTED RATHER THAN DESCRIBED, so this test goes red
    # the day `MODELS-025` is fixed and the end-to-end claim becomes available.
    # A file that says "blocked" in prose and checks nothing goes on saying it
    # after the block is gone.
    run = T.train(project, ["glm"])
    errored = [r for r in run.results if r.error]
    assert errored, (
        "a `glm` training run now completes without error, so MODELS-025 is "
        "fixed and this test should be replaced by the end-to-end claim: "
        "`figure_bundle._coefficients_for(project)` returning a non-empty list "
        "for a project whose only fitted model is `glm`.")
    assert FB._coefficients_for(project) == [], (
        "the coefficient figure now draws for a project whose only fitted "
        "model errored, which means `_coefficients_for` stopped skipping "
        "errored results")

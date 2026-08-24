"""Every custom transformer must survive `sklearn.base.clone`.

`UnitHarmonizer`'s docstring states the rule — "Store conversion_factors by
reference (no copy) so sklearn clone works" — and it was the only class in the
module that obeyed it. `PlausibilityGate` rebuilt both bound arrays in
`__init__`; `OutlierCapping` rewrote its own `params` with `params or {}` and
set the trailing-underscore fitted markers before anything was fitted. Both
failed `clone`'s identity check.

That mattered because nothing in the app clones defensively. `reconcile_pipeline_columns`
clones *unconditionally* before it knows whether any column drifted, so with
plausibility bounds configured every training run raised there; and
`make_cv_pipeline` clones the preprocessing to keep each fold's statistics
inside that fold. Both raises were caught by broad handlers that degraded to a
transient banner — the results table and the manuscript rendered complete,
without cross-validation, and the two failures presented as unrelated symptoms.

The rule is now a test rather than a comment on the one class that kept it, and
the test discovers its subjects: any transformer added to
`ml/preprocess_operators.py` is cloned here or the roster check fails.

Findings: STATE-068 (the invariant), STATE-002, TEST-003, MINE-006,
STATE-059, STATE-065, SWEEP-024.
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, clone
from sklearn.exceptions import NotFittedError

import ml.preprocess_operators as ops
from ml.eval import make_cv_pipeline
from ml.pipeline import build_preprocessing_pipeline, reconcile_pipeline_columns
from ml.preprocess_operators import OutlierCapping, PlausibilityGate, UnitHarmonizer


# ── the roster: discovered, not listed ───────────────────────────────────

def transformer_classes():
    """Every estimator class *defined in* ml/preprocess_operators.py."""
    return sorted(
        (name, obj) for name, obj in vars(ops).items()
        if inspect.isclass(obj)
        and issubclass(obj, BaseEstimator)
        and obj.__module__ == ops.__name__
    )


# Representative constructor arguments, one entry per interesting shape.
# `OutlierCapping` gets three because its bug only showed with a falsy
# `params` — which is exactly what ml/pipeline.py passes when no outlier
# parameters are configured (`numeric_outlier_params or {}`).
CONSTRUCTIONS = {
    "UnitHarmonizer": {
        "factors": dict(conversion_factors=[2.0, 0.5]),
    },
    "PlausibilityGate": {
        "one bound per column": dict(lower_bounds=[0.0, 1.0],
                                     upper_bounds=[10.0, 20.0]),
        "an ungated column": dict(lower_bounds=[0.0, None],
                                  upper_bounds=[10.0, None]),
    },
    "OutlierCapping": {
        "constructor defaults": dict(),
        "an empty params dict": dict(method="percentile", params={}),
        "percentile params": dict(method="percentile",
                                  params={"lower_q": 0.05, "upper_q": 0.95}),
        "mad params": dict(method="mad", params={"threshold": 3.0}),
    },
}

CASES = [
    pytest.param(cls, kwargs, id=f"{name}-{case}")
    for name, cls in transformer_classes()
    for case, kwargs in CONSTRUCTIONS.get(name, {}).items()
]


def test_every_transformer_in_the_module_is_covered_here():
    """A new transformer must not join the module unexamined."""
    missing = [name for name, _ in transformer_classes()
               if name not in CONSTRUCTIONS]
    assert not missing, (
        f"{missing} were added to ml/preprocess_operators.py without a clone "
        "case — add constructor arguments to CONSTRUCTIONS")
    assert CASES, "the roster is empty; discovery is broken"


# ── the invariant itself ─────────────────────────────────────────────────

@pytest.mark.parametrize("cls, kwargs", CASES)
def test_clone_does_not_raise(cls, kwargs):
    """STATE-068. `clone` reconstructs from get_params and asserts identity."""
    clone(cls(**kwargs))


@pytest.mark.parametrize("cls, kwargs", CASES)
def test_the_clone_carries_the_same_parameters(cls, kwargs):
    original = cls(**kwargs)
    copy = clone(original)
    assert copy.get_params() == original.get_params()


@pytest.mark.parametrize("cls, kwargs", CASES)
def test_a_freshly_constructed_transformer_is_not_marked_fitted(cls, kwargs):
    """MINE-006. Trailing-underscore attributes are sklearn's fitted marker.

    Setting them in `__init__` makes `check_is_fitted` see a fitted estimator
    that has learned nothing, so a pipeline rebuilt from stored config and
    transformed without fitting passes data through untouched while the badge
    says the step is on.
    """
    fresh = cls(**kwargs)
    fitted_markers = [a for a in vars(fresh)
                      if a.endswith("_") and not a.startswith("__")]
    assert not fitted_markers, (
        f"{cls.__name__}.__init__ sets {fitted_markers}, which marks an "
        "unfitted estimator as fitted")


@pytest.mark.parametrize("cls, kwargs", CASES)
def test_a_clone_still_fits_and_transforms(cls, kwargs):
    """Cloning must produce a working transformer, not merely a silent one."""
    X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0],
                  [4.0, 40.0], [5.0, 50.0]])
    out = clone(cls(**kwargs)).fit(X).transform(X)
    assert np.asarray(out).shape == X.shape


# ── the gate keeps gating, and says so when it has not been fitted ───────

def test_the_plausibility_gate_still_sets_out_of_range_values_to_missing():
    X = np.array([[-5.0, 15.0], [5.0, 15.0], [50.0, 15.0]])
    gate = PlausibilityGate(lower_bounds=[0.0, None], upper_bounds=[10.0, None])
    out = gate.fit(X).transform(X)

    assert np.isnan(out[0, 0]) and np.isnan(out[2, 0]), (
        "values outside the bounds were not set to missing")
    assert out[1, 0] == 5.0
    assert not np.isnan(out[:, 1]).any(), "the ungated column was touched"


def test_the_plausibility_gate_refuses_to_transform_before_fit():
    gate = PlausibilityGate(lower_bounds=[0.0], upper_bounds=[10.0])
    with pytest.raises(NotFittedError):
        gate.transform(np.array([[-5.0]]))


def test_outlier_capping_refuses_to_transform_before_fit():
    """MINE-006's other half: the silent no-op is the dangerous answer.

    Returning the input unchanged means a pipeline rebuilt from stored config
    and transformed without fitting emits uncapped data while the recorded
    configuration says capping is on.
    """
    capper = OutlierCapping(method="percentile", params={"lower_q": 0.1,
                                                         "upper_q": 0.9})
    with pytest.raises(NotFittedError):
        capper.transform(np.array([[1.0], [2.0], [900.0]]))


def test_outlier_capping_still_caps():
    X = np.array([[1.0], [2.0], [3.0], [4.0], [100.0]])
    out = OutlierCapping(method="percentile",
                         params={"lower_q": 0.1, "upper_q": 0.9}).fit_transform(X)
    assert out.max() < 100.0


def test_unit_harmonizer_still_converts():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = UnitHarmonizer(conversion_factors=[2.0, 0.5]).fit_transform(X)
    assert out[0, 0] == 2.0 and out[0, 1] == 1.0


# ── the pipeline the app actually builds ─────────────────────────────────

def frame(n=40):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "glucose": rng.normal(95, 12, n),
        "creatinine": rng.normal(1.0, 0.2, n),
        "sex": rng.choice(["Male", "Female"], n),
    })


def gated_pipeline():
    """What ml/pipeline.py composes when a project has plausibility bounds."""
    return build_preprocessing_pipeline(
        numeric_features=["glucose", "creatinine"],
        categorical_features=["sex"],
        numeric_outlier_treatment="percentile",
        numeric_outlier_params=None,  # -> `or {}` -> the falsy-params case
        unit_harmonization_factors=[1.0, 1.0],
        plausibility_bounds={"lower_bounds": [40.0, 0.2],
                             "upper_bounds": [400.0, 15.0]},
        plausibility_mode="clip",
    )


def test_a_pipeline_with_plausibility_bounds_can_be_cloned():
    """STATE-002. Every refit path in the app goes through this clone."""
    clone(gated_pipeline())


def test_reconcile_self_heals_a_gated_pipeline_instead_of_raising():
    """STATE-065 / SWEEP-024. Reconcile clones before it knows about drift.

    The existing drift test builds a plain scaler pipeline, which clones fine,
    so it could not see this: a pipeline holding one unclonable transformer
    raised on *every* training run rather than on drifted ones, and the raise
    was caught into a transient warning.
    """
    healed, dropped = reconcile_pipeline_columns(
        gated_pipeline(), ["glucose", "sex"])
    assert "creatinine" in dropped, "the drifted column was not reported"
    assert healed is not gated_pipeline()
    healed.fit(frame()[["glucose", "sex"]])


def test_reconcile_leaves_a_gated_pipeline_alone_when_nothing_drifted():
    pipe = gated_pipeline()
    same, dropped = reconcile_pipeline_columns(
        pipe, ["glucose", "creatinine", "sex"])
    assert dropped == []
    assert same is pipe


def test_cross_validation_runs_with_plausibility_bounds_configured():
    """STATE-059. The invariant was not violated, it was silently skipped.

    `make_cv_pipeline` clones the preprocessing so each fold re-fits its own
    transformers. With a gate in the pipeline that clone raised, pages/06
    caught it, and the run reported metrics with no cross-validation at all.
    Nothing imported `make_cv_pipeline` from a test before this one.
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score

    df = frame()
    y = df["glucose"] * 0.5 + np.random.default_rng(1).normal(0, 1, len(df))

    cv_pipe = make_cv_pipeline(gated_pipeline(), Ridge())
    scores = cross_val_score(cv_pipe, df, y, cv=3,
                             scoring="neg_mean_absolute_error")

    assert len(scores) == 3
    assert np.isfinite(scores).all(), (
        "cross-validation produced no usable scores")


def test_the_cv_pipeline_is_unfitted_so_every_fold_refits_the_preprocessing():
    """The reason the clone is there at all — stated as an assertion."""
    fitted = gated_pipeline().fit(frame())
    cv_pipe = make_cv_pipeline(fitted, None)

    prep = cv_pipe.named_steps["prep"]
    with pytest.raises(NotFittedError):
        prep.transform(frame())


def test_the_cv_composite_is_scored_on_raw_training_data():
    """STATE-059's second failure mode, which no runtime test can see.

    Composing the preprocessing into the CV estimator is only half the
    invariant; the other half is *what it is scored on*. Handing
    `perform_cross_validation` the already-transformed `X_train_model` would
    double-transform and, worse, leak every fold's held-out rows into the
    imputer/scaler statistics — and it would raise no exception and pass the
    whole suite, because the numbers stay plausible and merely improve.
    """
    import ast
    import os

    page = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "pages", "06_Train_and_Compare.py")
    with open(page, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=page)

    composites = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "make_cv_pipeline"
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert composites, "pages/06 no longer composes a CV pipeline"

    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "perform_cross_validation"
        and node.args and isinstance(node.args[0], ast.Name)
        and node.args[0].id in composites
    ]
    assert calls, "the CV composite is not the estimator being cross-validated"

    for call in calls:
        scored_on = call.args[1]
        assert isinstance(scored_on, ast.Name) and scored_on.id == "X_train", (
            f"cross-validation is scored on {ast.dump(scored_on)} rather than "
            "the raw X_train; a pre-transformed matrix leaks each fold's "
            "held-out rows into the transformer fits")

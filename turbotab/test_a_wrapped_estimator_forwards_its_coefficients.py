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

## What this file found on the way, and then unblocked

**The forest plot could not draw for these two models even once they had
coefficients**, and not because of anything here: `MODELS-025` — every
`BaseModelWrapper` subclass failed to complete a training run under
`scikit-learn` 1.9, so `glm`, `huber`, `rf` and `nn` came back with
`result.error` set and `_coefficients_for` skipped any errored result before it
ever asked about coefficients.

That was filed as a separate row and then **fixed in the same loop**, because it
is the plumbing the whole feature rests on. The last test in this file asserted
the blocker rather than describing it, so replacing it with the end-to-end claim
was obligatory once the block was gone — which is what happened, and is why the
final test now drives a real `glm` run and asserts the `forest` figure is
admitted.
"""
from __future__ import annotations

import importlib.util
import inspect
import os
import sys
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.model_registry import get_registry                          # noqa: E402
from models.base import BaseModelWrapper                            # noqa: E402
from models.glm import GLMWrapper                                   # noqa: E402
from models.huber_glm import HuberGLMWrapper                        # noqa: E402
from models.registry_wrappers import RegistryModelWrapper           # noqa: E402
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


# ═══════════ 1b · the rule, derived from the estimator ══════════════════════
#
# `L64-A5`. THIS FILE EXISTS BECAUSE A WRAPPER DROPPED AN ATTRIBUTE, AND IT
# CONTAINED ZERO OCCURRENCES OF `classes_` — the attribute the next dropped one
# turned out to be. `coef_` was named at L56, `intercept_` beside it, and
# `classes_` stayed missing for seven loops and cost a `glm`-only project its
# ROC, its calibration plot and its decision curve.
#
# So the attribute set is no longer written down here. It is read off the
# fitted estimator, which is the only source that cannot go stale.

#: Fitted by their own declaration, and skipped with a reason rather than
#: silently. `nn` is absent because `torch` is not installed on this machine —
#: `models/nn_whuber.py` cannot even be imported — and it is the one wrapper
#: that ASSIGNS `classes_` itself, so it is also the one this rule most needs
#: to cover. Reported by `test_the_skipped_wrapper_is_named` below.
_UNFITTABLE_HERE = {"NNWeightedHuberWrapper": "torch is not installed"}


def _concrete_wrappers():
    """Every concrete `BaseModelWrapper` subclass that SHIPS.

    Scoped to `models.*` on purpose. This file defines a stand-in below, and a
    sweep that swept its own scaffolding would be measuring itself — the shape
    `AGENT_ONBOARD.md` §07 #3 warns about, arriving through the enumeration
    rather than through a fixture.
    """
    import models.glm, models.huber_glm, models.rf                 # noqa: F401
    import models.registry_wrappers                                # noqa: F401

    found, seen = [], set()

    def walk(cls):
        for sub in cls.__subclasses__():
            if sub.__name__ in seen:
                continue
            seen.add(sub.__name__)
            if (not inspect.isabstract(sub)
                    and (sub.__module__ or "").startswith("models.")):
                found.append(sub)
            walk(sub)

    walk(BaseModelWrapper)
    return sorted(found, key=lambda c: c.__name__)


def _fitted_pairs():
    """`(label, wrapper)` fitted on a task each one declares it supports."""
    X, y_reg, y_clf = _table()
    out = []
    for cls in _concrete_wrappers():
        if cls.__name__ in _UNFITTABLE_HERE:
            continue
        if cls is RegistryModelWrapper:
            # It takes an estimator rather than a task string, and it is built
            # by no registry key — `pages/06` and `headless_train` construct it
            # directly. Covered on both shapes anyway.
            out.append(("registry-regression",
                        _fit(cls(LinearRegression(), "reg"), X, y_reg)))
            out.append(("registry-classification",
                        _fit(cls(LogisticRegression(max_iter=200), "clf"),
                             X, y_clf)))
            continue
        for task, y in (("regression", y_reg), ("classification", y_clf)):
            try:
                wrapper = cls(task_type=task)
            except TypeError:
                if task != "regression":
                    continue
                wrapper = cls()
            # WHAT IT DECLARES IT SUPPORTS, asked of the object rather than of
            # a table here: a regression-only wrapper fitted on labels would
            # fail for a reason that has nothing to do with forwarding.
            if task == "classification" and not wrapper.supports_proba():
                continue
            out.append((f"{cls.__name__}-{task}", _fit(wrapper, X, y)))
    return out


def _fitted_attributes(estimator):
    """sklearn's own convention for *something a fit produced*."""
    return sorted(
        name for name in dir(estimator)
        if name.endswith("_") and not name.startswith("_")
        and not name.endswith("__") and hasattr(estimator, name))


def test_a_wrapper_answers_for_every_attribute_its_estimator_learned(capsys):
    """The positive direction, over the whole set rather than three names."""
    pairs = _fitted_pairs()
    # THE CONTROL. An empty enumeration asserts nothing and looks identical to
    # a clean sweep — `AGENT_ONBOARD.md` §07 trap 5c, and this is the negative
    # shape it warns about.
    assert len(pairs) >= 4, (
        f"only {len(pairs)} wrapper(s) could be fitted, so this sweep's "
        f"silence says nothing: {[p[0] for p in pairs]}")

    reported = []
    for label, wrapper in pairs:
        learned = _fitted_attributes(wrapper.model)
        assert learned, f"{label}: the wrapped estimator learned nothing"
        missing = [a for a in learned if not hasattr(wrapper, a)]
        assert not missing, (
            f"{label} holds a fitted estimator with {missing} and the wrapper "
            f"answers for none of them. That is `GUIDED-234` and `GUIDED-245` "
            f"in one sentence: the wrapper is lying about what it knows, and "
            f"every consumer that asks it — the forest plot, `training.py`'s "
            f"event record — believes the lie.")
        reported.append(f"{label}:{len(learned)}")
    with capsys.disabled():
        print(f"\n  forwarded, per wrapper: {', '.join(reported)}")


def test_a_wrapper_does_not_answer_for_an_attribute_its_estimator_lacks():
    """The negative direction, and it is what keeps the rule honest.

    A wrapper that answered for everything would satisfy the test above and put
    a forest plot on a Random Forest.
    """
    X, y_reg, _ = _table()
    forest = _fit(RFWrapper(n_estimators=10, task_type="regression"), X, y_reg)
    learned = set(_fitted_attributes(forest.model))
    # `classes_` is the case this row is about: a REGRESSOR has none, and the
    # forwarding must not invent one.
    assert "classes_" not in learned
    for absent in ("coef_", "intercept_", "classes_", "a_made_up_attribute_"):
        assert absent not in learned
        assert not hasattr(forest, absent), (
            f"the wrapper answers for `{absent}`, which its estimator does not "
            f"have")


def test_an_unfitted_wrapper_answers_for_nothing_its_estimator_has_not_learned():
    """Before a fit there is nothing to forward, and a `clone()` is unfitted.

    `clone()` is in here because sklearn calls it inside every `Pipeline`, and
    a clone that inherited a fitted attribute would report a fit that never
    happened.
    """
    X, _y_reg, y_clf = _table()
    fitted = _fit(GLMWrapper(task_type="classification"), X, y_clf)
    assert hasattr(fitted, "classes_")

    for label, wrapper in (("unfitted", GLMWrapper(task_type="classification")),
                           ("clone-of-fitted", clone(fitted))):
        for attr in ("coef_", "intercept_", "classes_"):
            assert not hasattr(wrapper, attr), (
                f"{label} reports `{attr}`, so it is claiming a fit that has "
                f"not happened")


def test_the_skipped_wrapper_is_named():
    """A skip nobody counts is coverage nobody has. `AGENT_ONBOARD.md` §07 3d.

    `nn` is the wrapper that assigns `classes_` in `__init__` before any data
    exists, which is precisely the case a naive read-only property breaks on —
    so the one wrapper this rule most needs to cover is the one this machine
    cannot fit. That is stated here rather than left as an absence in a list.
    """
    assert _UNFITTABLE_HERE == {"NNWeightedHuberWrapper": "torch is not installed"}
    assert importlib.util.find_spec("torch") is None, (
        "`torch` is installed now, so `NNWeightedHuberWrapper` can be fitted "
        "and belongs in the sweep above rather than in the skip list")
    # And the property it exercises is asserted on a stand-in of the same shape,
    # so the mechanism is covered even while the wrapper is not.
    stand_in = _AssignsItsOwnClasses()
    assert stand_in.classes_ is None, (
        "a wrapper that assigns `None` reads it back as something else, which "
        "is the bug both naive versions of this forwarding have")
    stand_in.classes_ = np.array([0, 1])
    assert list(stand_in.classes_) == [0, 1]


class _AssignsItsOwnClasses(BaseModelWrapper):
    """`models/nn_whuber.py:320`'s shape, without torch.

    It assigns `classes_ = None` in `__init__` before any data exists and reads
    it back with `is not None` at `:636`. A plain read-only property fails at
    construction; a "defer to the instance attribute if it is not None" version
    falls through to `self.model` and raises.
    """

    def __init__(self) -> None:
        super().__init__("assigns-its-own")
        self.classes_ = None

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        self.classes_ = np.unique(y_train)
        self.is_fitted = True
        return {}

    def predict(self, X):                                  # pragma: no cover
        return X


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

    Two claims in one place, deliberately: the predicate `_coefficients_for`
    turns on, and then the figure it produces. The second half was blocked by
    `MODELS-025` when this file was written and is asserted now that the block
    is gone — the test named the blocker rather than describing it, which is
    what made replacing it obligatory rather than optional.
    """
    from turbotab import figure_bundle as FB
    from turbotab import pipeline_plan as PP
    from turbotab.project import AnalysisProject
    from turbotab import eventfixture
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
    eventfixture.choose_event(project, required=True)     # `DRIVE-041`

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
    coefficients_from_pipeline = np.ravel(np.asarray(estimator.coef_, dtype=float))
    assert coefficients_from_pipeline.size, (
        "the estimator exposes an empty coefficient vector")

    # THE END-TO-END CLAIM, WHICH THIS TEST ASKED FOR AND NOW MAKES.
    #
    # It used to assert that `MODELS-025` still blocked it — *"a glm run now
    # completes without error, so this test should be replaced by the
    # end-to-end claim"* — and that instruction was followed in the same loop
    # rather than left for a later one. The blocker being asserted rather than
    # described is what made the replacement obligatory instead of optional.
    project.training_run = T.train(project, ["glm"])
    errored = [r.error for r in project.training_run.results if r.error]
    assert not errored, (
        f"a `glm` training run errored: {errored}. MODELS-025 has regressed — "
        f"every BaseModelWrapper subclass fails under scikit-learn 1.9 without "
        f"`__sklearn_tags__` and `__sklearn_is_fitted__`.")

    coefficients = FB._coefficients_for(project)
    assert coefficients, (
        "no coefficients reached the figure layer for a project whose only "
        "fitted model is `glm` — §A4.7's forest plot cannot be drawn for the "
        "model the registry calls interpretable. GUIDED-234.")
    assert len(coefficients) == coefficients_from_pipeline.size, (
        f"the figure layer reports {len(coefficients)} coefficients and the "
        f"fitted pipeline has {coefficients_from_pipeline.size}")
    for row in coefficients:
        assert row["name"], "a coefficient reached the figure with no predictor name"

    drawn = {row["id"] for row in FB.render(project).get("admitted") or []}
    assert "forest" in drawn, (
        f"the coefficient figure is still not admitted for a GLM-only project; "
        f"admitted: {sorted(drawn)}")

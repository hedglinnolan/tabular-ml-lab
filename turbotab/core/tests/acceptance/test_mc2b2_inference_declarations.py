"""MC-2b-2 · the inference switches read the families' declarations (WAVE_C6A_PLAN §3, MC-2b-2;
MODEL_FAMILY_CONTRACT §3.3).

Each predicate that replaced a switch on a family's key or its estimator's class is pinned to select
exactly the families the switch named. The reference for each is the switch itself, copied below
from the code as it stood at ``turbotab-next`` 05325dca (its tables, its signature check, its class
check), never the new code's output. The methods words each table held are checked word for word.
"""
from __future__ import annotations

import inspect
import itertools
import math
import warnings
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from turbotab.core.models import get_family
from turbotab.core.models.base import TASKS, families, reports_coefficients

# ── the switches as they were (05325dca), the reference ──────────────────────

# models/survey.py
OLD_DESIGN_FAMILY = {"regression": "linear", "binary": "linear", "multiclass": "linear",
                     "ordinal": "proportional_odds", "time_to_event": "cox"}
OLD_DESIGN_LABEL = {
    ("linear", "regression"): "survey-weighted least squares",
    ("linear", "binary"): "survey-weighted logistic regression",
    ("linear", "multiclass"): "survey-weighted multinomial logistic regression",
    ("linear", "ordinal"): "survey-weighted multinomial logistic regression",
    ("proportional_odds", "ordinal"): "the survey-weighted proportional-odds model",
    ("cox", "time_to_event"): "survey-weighted Cox regression",
}
OLD_ESTIMATOR_WORDS = {
    ("linear", "regression"): "least squares",
    ("linear", "binary"): "logistic regression (pseudo-maximum likelihood)",
    ("linear", "multiclass"): "multinomial logistic regression (pseudo-maximum likelihood)",
    ("linear", "ordinal"): "multinomial logistic regression (pseudo-maximum likelihood)",
    ("proportional_odds", "ordinal"): "the proportional-odds model (pseudo-maximum likelihood)",
    ("cox", "time_to_event"): "Cox regression (Binder's pseudo-likelihood, Efron ties)",
}
OLD_BLOCKED_WORDS = {
    "mixed": "the random-intercept mixed model",
    "gee": "the GEE model",
    "featurewise": "feature-wise regression",
    "elastic_net": "the elastic net",
    "boosted_trees": "the gradient-boosted tree model",
}


def old_has_design_estimator(family: Any, task: str) -> bool:
    fn = family.inference
    if fn is None or task not in getattr(family, "tasks", ()):
        return False
    return "survey" in inspect.signature(fn).parameters


def _listing(items: list[str]) -> str:
    return "".join(items) if len(items) <= 1 else f"{', '.join(items[:-1])} and {items[-1]}"


def old_models_sentence(models: list[str], task: str, weight: str | None) -> str | None:
    based, stopped = [], []
    for key in models:
        family = get_family(key)
        if old_has_design_estimator(family, task):
            based.append(OLD_ESTIMATOR_WORDS.get((key, task)) or family.methods_label(task))
        else:
            stopped.append(OLD_BLOCKED_WORDS.get(key) or family.methods_label(task))
    parts = []
    if based:
        by = f" by `{weight}`" if weight else ""
        verb = "was" if len(based) == 1 else "were"
        parts.append(f"for the surveyed population, {_listing(based)} {verb} weighted{by}, with "
                     f"standard errors by Taylor linearization over the survey design")
    if stopped:
        verb = "has" if len(stopped) == 1 else "have"
        whose = "its" if len(stopped) == 1 else "their"
        parts.append(f"{_listing(stopped)} {verb} no design-based estimator, so {whose} estimates "
                     f"were blocked and not reported")
    if not parts:
        return None
    text = "; ".join(parts)
    return text[0].upper() + text[1:]


# methods/interaction.py
OLD_SUPPORTED = ("linear", "cox", "proportional_odds")


def old_measure(task: str, family: str, target: str) -> tuple[str, bool]:
    if family == "cox" or task == "time_to_event":
        return "hazard ratio", True
    if family == "proportional_odds" or task == "ordinal":
        return "cumulative odds ratio", True
    if task == "binary":
        return "odds ratio", True
    return f"difference in mean `{target}`", False


# scales.py:methods_sentence
OLD_SCALE_MODEL = {"proportional_odds": "proportional-odds", "linear": "logistic"}


# stages/class_substitution.py
def old_plain_multinomial(model: Any) -> bool:
    from sklearn.linear_model import LogisticRegression

    if not isinstance(model, LogisticRegression):
        return False
    params = model.get_params(deep=False)
    unpenalized = (params.get("penalty") in (None, "none")
                   or not math.isfinite(float(params.get("C", 1.0))))
    return (unpenalized and params.get("class_weight") is None
            and bool(params.get("fit_intercept", True)) and len(getattr(model, "classes_", ())) >= 3)


def _families() -> dict[str, Any]:
    from turbotab.core.contracts import contracts

    contracts()  # the omics chain registers the screened elastic net
    return {f.key: f for f in families()}


def _pairs() -> list[tuple[Any, str]]:
    return [(f, t) for f in _families().values() for t in TASKS]


# ── models/survey.py: design_based ───────────────────────────────────────────


def test_the_design_based_estimator_is_the_one_whose_inference_takes_a_survey():
    from turbotab.core.models.survey import has_design_estimator

    chosen = {(f.key, t) for f, t in _pairs() if has_design_estimator(f, t)}
    assert chosen == {(f.key, t) for f, t in _pairs() if old_has_design_estimator(f, t)}
    assert chosen == {("linear", "regression"), ("linear", "binary"), ("linear", "multiclass"),
                      ("linear", "ordinal"), ("proportional_odds", "ordinal"),
                      ("cox", "time_to_event")}


def test_a_family_that_declares_no_design_estimator_is_not_given_one():
    """The declaration decides, not the signature: linear's ``inference`` takes a survey."""
    from turbotab.core.models.linear import Linear
    from turbotab.core.models.survey import has_design_estimator

    declared = replace(Linear.inference_decl, design_based=False, default_for=())
    probe = type("Probe", (Linear,), {"key": "probe", "inference_decl": declared})()
    assert "survey" in inspect.signature(probe.inference).parameters
    assert not any(has_design_estimator(probe, t) for t in TASKS)


def test_the_family_offered_in_place_of_one_with_no_design_estimator_is_the_old_table():
    from turbotab.core.models.survey import design_family

    _families()
    assert {t: design_family(t) for t in TASKS} == OLD_DESIGN_FAMILY


def test_the_design_based_words_are_the_old_tables_word_for_word():
    from turbotab.core.models.survey import blocked_words, design_label, estimator_words

    for (key, task), label in OLD_DESIGN_LABEL.items():
        assert design_label(get_family(key), task) == label, (key, task)
    for f, t in _pairs():
        if t not in f.tasks:
            continue
        if old_has_design_estimator(f, t):
            assert estimator_words(f, t) == (OLD_ESTIMATOR_WORDS.get((f.key, t))
                                             or f.methods_label(t)), (f.key, t)
        else:
            assert blocked_words(f, t) == (OLD_BLOCKED_WORDS.get(f.key)
                                           or f.methods_label(t)), (f.key, t)


def test_the_exit_and_the_shelf_name_the_design_based_family_as_before():
    from turbotab.core.models.base import Assessment
    from turbotab.core.models.survey import no_design_estimator, population_shelf

    trees = get_family("boosted_trees")
    exits = no_design_estimator(trees, "ordinal", ["boosted_trees", "linear"]).info["exits"]
    assert exits[0] == {"label": "Use the survey-weighted proportional-odds model",
                        "decision": {"kind": "select_models",
                                     "models": ["proportional_odds", "linear"]}}
    shelf = population_shelf([(trees, Assessment(1.0, "good")),
                              (get_family("cox"), Assessment(0.5, "good"))], "time_to_event")
    assert [f.key for f, _ in shelf] == ["cox", "boosted_trees"]
    assert shelf[1][1].concerns[0] == (
        "Under the surveyed population it has no design-based estimator, so its estimates are "
        "blocked and recorded; survey-weighted Cox regression has one.")


@pytest.mark.parametrize("task", TASKS)
def test_the_models_sentence_is_unchanged_for_every_pair_of_families(task):
    from turbotab.core.models.survey import models_sentence

    keys = [f.key for f in _families().values() if task in f.tasks]
    state = SimpleNamespace(purpose="inference",
                            survey=SimpleNamespace(estimand="population", weight="wtmec2yr"))
    for chosen in itertools.chain(itertools.combinations(keys, 1),
                                  itertools.combinations(keys, 2)):
        assert models_sentence(state, list(chosen), task) == old_models_sentence(
            list(chosen), task, "wtmec2yr"), chosen


def test_the_models_sentence_reads_as_written():
    """One sentence written out by hand, so the reference above is itself checked."""
    from turbotab.core.models.survey import models_sentence

    _families()
    state = SimpleNamespace(purpose="inference",
                            survey=SimpleNamespace(estimand="population", weight=None))
    assert models_sentence(state, ["linear", "proportional_odds", "elastic_net"],
                           "ordinal") == (
        "For the surveyed population, multinomial logistic regression (pseudo-maximum likelihood) "
        "and the proportional-odds model (pseudo-maximum likelihood) were weighted, with standard "
        "errors by Taylor linearization over the survey design; the elastic net has no "
        "design-based estimator, so its estimates were blocked and not reported")


def test_the_marginal_measure_is_blocked_for_the_family_the_old_check_named():
    """``plan_previews.population_block`` blocked the marginal risk difference or ratio when the
    linear family was chosen (``"linear" in models``), as ``stages.effects`` standardizes it."""
    from turbotab.core.models.survey import standardizes_margin

    assert {f.key for f in _families().values() if standardizes_margin(f, "binary")} == {"linear"}


# ── methods/interaction.py: product_terms, raw_scale ─────────────────────────


def test_the_product_terms_are_tested_by_the_families_the_old_list_named():
    from turbotab.core.methods.interaction import tests_product_terms

    found = _families().values()
    assert {f.key for f in found if tests_product_terms(f)} == {
        f.key for f in found if reports_coefficients(f) and f.key in OLD_SUPPORTED}


def test_each_measure_is_the_old_one_for_every_family_and_task():
    from turbotab.core.methods.interaction import _measure

    for f, t in _pairs():
        if t in f.tasks:  # a family is fit only on a task of its own
            assert _measure(t, f, "kcal") == old_measure(t, f.key, "kcal"), (f.key, t)
    for t in TASKS:  # no family known: the task decides, as an empty key did
        assert _measure(t, None, "kcal") == old_measure(t, "", "kcal"), t


def test_the_survey_weights_enter_the_product_terms_for_the_family_the_old_check_named():
    from turbotab.core.methods.interaction import weighted_products

    assert {f.key for f in _families().values() if weighted_products(f)} == {"linear"}


# ── the calibration predicate, shared by three places ────────────────────────


def test_regression_calibration_corrects_exactly_the_linear_family():
    from turbotab.core.stages.calibration import corrects_by_calibration

    assert {f.key for f in _families().values() if corrects_by_calibration(f)} == {"linear"}


def test_the_predicate_does_not_lean_on_proportional_odds_declaring_values_nonlinear():
    """WAVE_C6A_PLAN §5 reading 8: proportional odds declares ``linear_in_values = False``, though
    its cumulative logit is a weighted sum of the values as given; were it declared True, the
    predicate would still not select it (its coefficient is not a least-squares or logistic one)."""
    from turbotab.core.models.ordinal import ProportionalOdds
    from turbotab.core.stages.calibration import corrects_by_calibration

    po = get_family("proportional_odds")
    assert po.linear_in_values is False
    probe = type("Probe", (ProportionalOdds,), {"key": "probe", "linear_in_values": True})()
    assert not corrects_by_calibration(probe)


@pytest.mark.parametrize("models", [
    None, [], ["linear"], ["boosted_trees", "linear"], ["proportional_odds"], ["elastic_net"],
    ["featurewise", "gee", "mixed", "cox"], ["no_such_family", "linear"]])
def test_the_three_places_choose_the_family_the_old_check_named(models):
    from turbotab.core.quest import _linear_family
    from turbotab.core.stages.calibration import calibrated_family

    _families()
    old = "linear" if "linear" in (models or []) else None
    assert calibrated_family(models) == old
    assert _linear_family(SimpleNamespace(models=models)) == (models is None or old is not None)


# ── stages/class_substitution.py: the refined multinomial logit ──────────────


def test_the_multinomial_refinement_applies_where_the_class_check_did():
    from turbotab.core.stages.class_substitution import _plain_multinomial

    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(size=(120, 3)), columns=["a", "b", "c"])
    y = np.array(["low", "mid", "high"])[rng.integers(0, 3, size=120)]
    seen = {}
    for f in _families().values():
        if "multiclass" not in f.tasks:
            continue
        model = f.build("multiclass", "inference", len(X), X.shape[1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X, y)
        seen[f.key] = _plain_multinomial(f, model)
        assert seen[f.key] == old_plain_multinomial(model), f.key
    assert seen["linear"] is True and seen["boosted_trees"] is False


def test_the_refinement_still_reads_the_fit_s_own_settings():
    """The declaration names the model; a penalized or weighted fit is still not refined."""
    from sklearn.linear_model import LogisticRegression

    from turbotab.core.stages.class_substitution import _plain_multinomial

    rng = np.random.default_rng(5)
    X = rng.normal(size=(90, 2))
    y = np.array([0, 1, 2])[rng.integers(0, 3, size=90)]
    linear = get_family("linear")
    for model in (LogisticRegression(C=1.0), LogisticRegression(C=np.inf, class_weight="balanced"),
                  LogisticRegression(C=np.inf, fit_intercept=False)):
        model.fit(X, y)
        assert old_plain_multinomial(model) is False
        assert _plain_multinomial(linear, model) is False


class _Wrapper:
    """A fitted classifier that states none of its settings, as a wrapper might."""

    classes_ = np.array([0, 1, 2])

    def get_params(self, deep: bool = False) -> dict[str, Any]:
        return {}


def test_a_fit_that_does_not_state_its_penalty_is_not_refined():
    """The class check stood guard over a fit with no ``penalty`` or ``C`` among its settings (a
    ridge-type classifier, a wrapper): the declaration alone must not read one as unpenalized.
    Reference: the old class check, which refused both."""
    from sklearn.linear_model import RidgeClassifier

    from turbotab.core.stages.class_substitution import _plain_multinomial

    rng = np.random.default_rng(7)
    X = rng.normal(size=(90, 2))
    y = np.array([0, 1, 2])[rng.integers(0, 3, size=90)]
    linear = get_family("linear")
    for model in (RidgeClassifier().fit(X, y), _Wrapper()):
        assert old_plain_multinomial(model) is False
        assert _plain_multinomial(linear, model) is False


def test_the_stage_s_refit_reaches_the_maximum_likelihood_estimate():
    """``class_predictor`` as the stage calls it on a refit (``class_family_entries`` and the
    pooled refit pass the family) gives the multinomial logit's maximum-likelihood probabilities.
    Reference: statsmodels' ``MNLogit`` on the same rows, converged to a 1e-12 step."""
    import statsmodels.api as sm
    from sklearn.pipeline import Pipeline

    from turbotab.core.stages.class_substitution import class_predictor

    rng = np.random.default_rng(11)
    n = 300
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    eta = np.column_stack([np.zeros(n), 0.8 * X["a"] - 0.4, -0.6 * X["b"] + 0.3 * X["a"]])
    p = np.exp(eta) / np.exp(eta).sum(axis=1, keepdims=True)
    y = np.array(["high", "low", "mid"])[[rng.choice(3, p=row) for row in p]]
    linear = get_family("linear")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline = Pipeline([("model", linear.build("multiclass", "inference", n, 2))]).fit(X, y)
    classes = ["low", "mid", "high"]
    predict = class_predictor(pipeline, X, y, classes, linear)

    codes = pd.Categorical(y, categories=classes).codes
    reference = sm.MNLogit(codes, sm.add_constant(X.to_numpy())).fit(
        method="newton", maxiter=200, tol=1e-12, disp=False)
    expected = reference.predict(sm.add_constant(X.to_numpy()))
    assert np.max(np.abs(predict(X) - expected)) < 1e-8
    with pytest.raises(ValueError, match="does not hold every class"):
        class_predictor(pipeline, X, y, ["low", "mid"], linear)


# ── models/survey.py: the words tables stay keyed by labels that exist ───────


def test_each_word_table_key_is_a_registered_family_s_methods_label():
    """``_DESIGN_WORDS`` and ``_NO_DESIGN_WORDS`` are keyed by ``methods_label``: a family that
    renames its estimator would fall back to its label in silence and change the methods
    sentence. Each key must be the label of a registered family that has (or lacks) a
    design-based estimator for a task of its own. Reference: the old tables' families, written
    in the test."""
    from turbotab.core.models.survey import _DESIGN_WORDS, _NO_DESIGN_WORDS, has_design_estimator

    design, other = {}, {}
    for f, t in _pairs():
        if t in f.tasks:
            (design if has_design_estimator(f, t) else other).setdefault(
                f.methods_label(t), set()).add(f.key)
    assert set(_DESIGN_WORDS) <= set(design), sorted(set(_DESIGN_WORDS) - set(design))
    assert set(_NO_DESIGN_WORDS) <= set(other), sorted(set(_NO_DESIGN_WORDS) - set(other))
    assert set().union(*(design[k] for k in _DESIGN_WORDS)) == {
        "linear", "proportional_odds", "cox"}
    assert set().union(*(other[k] for k in _NO_DESIGN_WORDS)) >= {
        "mixed", "gee", "featurewise", "elastic_net", "boosted_trees"}


# ── estimand.py: the feature-wise family by its declarations ─────────────────


def test_an_exposure_family_is_estimated_by_the_family_the_old_check_named():
    from turbotab.core.estimand import _estimates_each_in_turn

    assert {f.key for f in _families().values() if _estimates_each_in_turn(f.key)} == {
        "featurewise"}
    assert not _estimates_each_in_turn("no_such_family")


# ── scales.py: the correction's model in words ───────────────────────────────


def test_the_correction_s_model_is_named_as_the_old_table_named_it():
    from turbotab.core.models.base import inference_default
    from turbotab.core.scales import _correction_model

    _families()
    for task in ("binary", "ordinal"):  # the tasks whose correction is an odds ratio
        key = inference_default(task).key
        assert _correction_model(key) == OLD_SCALE_MODEL[key], task
    assert _correction_model("no_such_family") == ""
    assert _correction_model(None) == ""

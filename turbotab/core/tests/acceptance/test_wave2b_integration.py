"""Wave 2b's seams: where LEASH, RC, FORM, EXPLORE and MULTISUB meet each other and the repairs
that landed on turbotab-next while they were built (wave 2a's and wave 1b's).

Each package's own acceptance file tests its method against its references. This file tests the
places where two of them wrote the same rule, or where one package's rule must hold inside the
other's method:

* **One registry** (BLUEPRINT §13). Every wave 2b method enters the one registry, tagged with the
  package that declared it, its module among the declaring modules.
* **One SD for a difference's E-value** (MODELING_SEQUENCE §0 ruling 14; the wave 2a repairs and
  LEASH). Both wrote the population's SD: the repairs as ``models.effects.outcome_sd``, LEASH as
  ``estimand_sd``, which also says whose SD it is. They are one rule now (``outcome_sd`` is
  ``estimand_sd``'s value), and the methods text carries both packages' words: the repairs' "of the
  straight-line estimate beside the curve" and the partially linear model's robustness value, and
  LEASH's "standardized by the outcome's design-weighted standard deviation in the surveyed
  population".
* **One list of score fields under inference** (ruling 13; LEASH and EXPLORE). LEASH withholds an
  inference fit's scores while its plan is open; EXPLORE withholds every cross-validated score from
  an inference fit, answered or not. Both drop the same fields (``estimand.MODEL_SCORES``,
  ``FIT_SCORES`` and the baseline's value), so neither leaves a score the other removes.
* **A bootstrap replicate imputes as the analysis's copies do** (RC and the wave 1b MI repair).
  RC's whole-chain bootstrap re-imputes each replicate (Boot MI); the MI repair carries a column
  the user confirmed as one value per unit to its unit's blank rows and imputes it once per unit.
  The replicate's imputation takes the same columns.

The expected values come from an independent path (NumPy written out, the packages' own constants
for the rules) or from the data's own structure (a value constant within each unit).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import turbotab.core.models  # noqa: F401 - registers the families
from turbotab.core import decisions as d

AT = "2026-10-05T00:00:00Z"


def _fold(*decisions: d.BaseModel) -> d.ProjectState:
    return d.fold([d.DecisionRecord(id=f"r{i}", seq=i, at=AT, decision=x)
                   for i, x in enumerate(decisions, start=1)])


# ── one registry ─────────────────────────────────────────────────────────────

WAVE2B = {
    "LEASH": ({"adjustment_guesses", "censored_below_detection", "copies_not_repeats",
               "evalue_sd", "fit_statistics_withheld", "grouping_by_structure"},
              "turbotab.core.routing_leash"),
    "RC": ({"regression_calibration"}, "turbotab.core.methods.calibration"),
    "FORM": ({"effect_modification", "exposure_transform", "functional_form", "interaction"},
             "turbotab.core.methods.exposure_form"),
    "EXPLORE": ({"design_based_cv", "explore", "imbalance_correction", "inner_cv_form",
                 "intended_use", "spline_rule", "variable_selection", "variance_filter"},
                "turbotab.core.stages.explore"),
    "MULTISUB": ({"multiclass_substitution"}, "turbotab.core.methods.substitution"),
}


def test_every_wave2b_method_enters_through_the_one_registry():
    """BLUEPRINT §13: each method's slot, data scope, needs, routing (question, options labeled
    customary and sound), storyboard, sentence and relations, in the one registry, tagged with the
    package that declared it; each package's module is a declaring module."""
    from turbotab.core import contracts as C

    registry = C.contracts()
    for package, (keys, module) in WAVE2B.items():
        assert {k for k, c in registry.items() if c.package == package} == keys, package
        assert module in C.DECLARING_MODULES, package
        for key in keys:
            c = registry[key]
            assert c.slot in C.SLOTS and c.scope in C.SCOPES, key
            assert c.needs and c.question and c.storyboard and c.options and c.sentence, key
            for o in c.options:
                assert o.label and o.customary, (key, o.key)


# ── one SD for a difference's E-value (ruling 14) ────────────────────────────


def test_the_e_value_sd_is_one_rule_written_out_as_svyvar():
    """``outcome_sd`` (the wave 2a repairs) and ``estimand_sd`` (LEASH) give one value: R survey's
    ``svyvar`` written out, ``n / (n − 1) · Σ w (y − ȳ_w)² / Σ w`` over the rows with a positive
    weight, its square root; the rows' own ``sd`` without weights. ``estimand_sd`` says whose."""
    from turbotab.core.models.effects import estimand_sd, outcome_sd

    rng = np.random.default_rng(20261005)
    y = rng.normal(120, 15, 400)
    w = rng.uniform(0.2, 6.0, 400)
    keep = w > 0
    mean = float(np.sum(w * y) / np.sum(w))
    n = int(keep.sum())
    by_hand = math.sqrt(n / (n - 1) * float(np.sum(w * (y - mean) ** 2)) / float(np.sum(w)))
    assert by_hand != pytest.approx(float(np.std(y, ddof=1)), rel=1e-3)
    assert outcome_sd(y, w) == pytest.approx(by_hand, rel=1e-12)
    assert estimand_sd(y, w) == (pytest.approx(by_hand, rel=1e-12), "design_weighted")
    assert outcome_sd(y) == pytest.approx(float(np.std(y, ddof=1)), rel=1e-12)
    assert estimand_sd(y) == (pytest.approx(float(np.std(y, ddof=1)), rel=1e-12), "sample")


E_VALUE = "the E-value for the estimate and for the confidence limit nearer the null"
POPULATION = ("the difference standardized by the outcome's design-weighted standard deviation in "
              "the surveyed population")


def test_the_sensitivity_sentence_carries_both_packages_words():
    """The effects stage's clause names what it bounds when it is the straight line beside a curve
    (wave 2a repairs) and the population's SD under the population answer (LEASH), together; the
    causal lane's partially linear model leads with its final step's robustness value (wave 2a
    repairs) and names the population's SD after its E-value (LEASH). Verbatim."""
    from turbotab.core import causal
    from turbotab.core.stages.effects import STRAIGHT_LINE, sensitivity_clause

    assert sensitivity_clause(["e_value"], of=STRAIGHT_LINE, sd_basis="design_weighted") == (
        f"Sensitivity to unmeasured confounding of the straight-line estimate beside the curve is "
        f"reported by {E_VALUE}, {POPULATION}, never as a pass or a fail.")
    assert sensitivity_clause(["e_value"], sd_basis="sample") == (
        f"Sensitivity to unmeasured confounding is reported by {E_VALUE}, never as a pass or a "
        f"fail.")
    lane = causal.sensitivity_sentence({
        "computed": True, "method": "dml_plr", "methods": ["robustness_value", "e_value"],
        "e_value": {"point": 1.4, "limit": 1.1, "sd": 2.0, "sd_basis": "design_weighted"}})
    assert lane == (
        " Sensitivity to unmeasured confounding is reported by the Cinelli–Hazlett robustness "
        "value of the final least-squares step, the outcome's residual on the exposure's (the form "
        "the omitted-variable bound of Chernozhukov, Cinelli, Newey, Sharma & Syrgkanis 2022, NBER "
        "w30302 takes in the partially linear model), the median over the sample splits (its form "
        "for the 95% interval is not reported: it assumes a classical standard error, and the "
        f"interval reported is the estimating equation's) and by {E_VALUE}, {POPULATION}, never as "
        "a pass or a fail.")


# ── one list of score fields under inference (ruling 13) ─────────────────────


def _scored_fit() -> dict:
    from turbotab.core.estimand import FIT_SCORES, MODEL_SCORES

    model = {"family": "linear", "label": "Linear", "cv": {"r2": {"mean": 0.31}},
             "baseline": {"label": "the mean", "value": 0.0},
             "coefficients": [{"feature": "x", "estimate": 1.0}], "inference": {"n": 100},
             "exposure_tests": [], "concerns": ["R² 0.31 in cross-validation."],
             "score_concerns": ["R² 0.31 in cross-validation."]}
    model.update({k: {"value": 0.5} for k in MODEL_SCORES})
    fit = {"models": [model], "comparisons": [{"a": "linear"}], "chain": []}
    fit.update({k: {"value": 0.5} for k in FIT_SCORES})
    return fit


def test_inference_withholds_the_same_score_fields_open_or_answered():
    """LEASH's withheld fit (a plan question open) and EXPLORE's served fit (ruling 13: under
    inference, answered or not) drop every field of the one list: each model's cross-validated
    scores, the baseline's value and every ``MODEL_SCORES`` field, and the fit's ``FIT_SCORES``;
    EXPLORE's says why. Under prediction EXPLORE's leaves the fit as it is."""
    from turbotab.core import estimand
    from turbotab.core.stages.evaluation import NO_SCORE_UNDER_INFERENCE, withhold_scores

    state = _fold(d.SetPurpose(purpose="inference"))
    gated = estimand.withhold("fit", _scored_fit(), {"reason": "The exposure is not declared.",
                                                    "purpose": "inference"})
    served = withhold_scores(_scored_fit(), state)
    for fit in (gated, served):
        for m in fit["models"]:
            assert m["cv"] == {} and m["baseline"]["value"] is None
            assert all(m[k] is None for k in estimand.MODEL_SCORES)
        assert all(fit[k] is None for k in estimand.FIT_SCORES)
        assert fit["comparisons"] == []
    assert served["cv_definition"] == NO_SCORE_UNDER_INFERENCE
    assert served["models"][0]["concerns"] == []  # the concern that quoted a score left with it
    assert served["models"][0]["coefficients"]  # the estimates stay: they are the result
    fit = _scored_fit()
    assert withhold_scores(fit, state.model_copy(update={"purpose": "prediction"})) is fit


# ── a bootstrap replicate imputes as the analysis's copies do (RC × MI repair) ─


def test_a_calibration_replicate_imputes_a_confirmed_unit_column_once_per_unit():
    """RC's Boot MI re-imputes each bootstrap replicate (``stages.calibration._impute``) as the
    coefficient table's copies are drawn: a column the user confirmed as one value per unit
    (``time_invariant``, the analysis plan's ``unit_level``) is carried to its unit's blank rows and
    imputed once per unit, so it holds one value within every unit of every replicate's copies;
    without it the same column is imputed row by row and differs within some unit."""
    from turbotab.core.decisions import MissingSpec
    from turbotab.core.models.inference import Clusters
    from turbotab.core.models.pipeline import design_spec
    from turbotab.core.stages.calibration import _impute

    rng = np.random.default_rng(11)
    units = 80
    frame = pd.DataFrame({"g": np.repeat(np.arange(units), 3), "x": rng.normal(size=3 * units)})
    frame["w"] = np.repeat(rng.normal(size=units), 3)
    frame.loc[rng.random(3 * units) < 0.25, "w"] = np.nan
    frame.loc[[0, 1, 2], "w"] = np.nan  # a unit with no record: imputed once for the unit
    frame.loc[rng.random(3 * units) < 0.15, "x"] = np.nan
    y = 1 + frame["x"].fillna(0).to_numpy() + rng.normal(size=3 * units)
    state = _fold(d.ConfirmReading(reading="time_invariant", column="w", value="yes")).model_copy(
        update=dict(target="y", task="regression", purpose="inference",
                    roles={"x": "exposure", "w": "covariate"},
                    missing=MissingSpec(strategy="multiple_imputation")))
    X = frame[["x", "w"]]
    spec = design_spec(state, X, ["x", "w"])
    clusters = Clusters(column="g", codes=frame["g"].to_numpy(), n_clusters=units)

    def within(copies: list[pd.DataFrame]) -> int:
        return max(int(c.assign(g=frame["g"].to_numpy()).groupby("g")["w"].nunique().max())
                   for c in copies)

    once = _impute(spec, X, y, "regression", 3, 5, None, clusters, {}, {}, time_invariant=["w"])
    rows = _impute(spec, X, y, "regression", 3, 5, None, clusters, {}, {})
    assert all(not c["w"].isna().any() for c in once)
    assert within(once) == 1
    assert within(rows) > 1

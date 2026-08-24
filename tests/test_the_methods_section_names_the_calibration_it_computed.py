"""`AUDIT-017` — the Methods section named three things a regression run never computes.

`ml/publication.generate_methods_section` renders one sentence per selected
explainability method. The calibration one read, for every task type:

> *"Model calibration was assessed using reliability diagrams, Brier score, and
> expected calibration error (ECE)."*

All three are defined for **predicted probabilities**. On a regression run
`ml/calibration.calibration_regression` returns a calibration slope, intercept
and R² and leaves `brier_score`, `ece` and every bin field `None`; the
reliability diagram (`plot_calibration_curve`) is drawn only on the
classification branch of `pages/06_Train_and_Compare.py`. So the sentence named
three analyses that do not exist for a continuous outcome, in the artifact that
leaves the building.

## What the registry requires, and why the sentence was not simply deleted

`research/CLINICAL_SURVEY_PACK.md` §A5.3 [SETTLED] requires a report to carry
*"calibration intercept and slope with CIs, the flexible calibration curve, O:E
ratio, Brier score (and scaled Brier)"* — for the binary case it names. §A5.1
requires that a project either *"report the calibration curve honestly or apply
post-hoc recalibration"*. The correction is therefore **two sentences, not
zero**: what a regression run does assess (the observed outcome regressed on the
predicted value — slope, intercept, R²), and the disclosure that the three named
probability metrics were not computed for a continuous outcome. The shelf is not
shortened; the false clause is replaced by the true one plus its own absence.

## Which path this is on

`pages/10_Report_Export._build_methods_section_for_export` runs `NarrativeEngine`
first and falls back to `generate_methods_section` when workflow provenance is
empty or the engine raises. `NarrativeEngine` says nothing about calibration at
all, so **the fallback is the only Classic producer of a calibration methods
sentence** — and it is a real export path, not a dead one.

## `GUIDED-097` — three target shapes, and the third is a measurement

* **continuous** — the shape the row is about.
* **binary 0/1** — the shape the true claim survives on.
* **binary as strings** (`"yes"`/`"no"`) — driven because `float()` succeeding on
  a `0/1` target is exactly what `GUIDED-097` was filed about. What it found is
  recorded in `test_a_string_outcome_computes_no_calibration_at_all`: the shipped
  `calibration_classification` raises `ValueError` on string labels, so
  `pages/06`'s per-model `except` swallows it, `calibration_results` stays empty,
  and `pages/10`'s gate never offers `calibration` for mention. The app is
  **silent** rather than false, which is the governing rule's permitted branch —
  but the silence is total and nobody is told, which is filed separately.

The shape not covered is named at the bottom of this file.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.calibration import (                                          # noqa: E402
    calibration_classification, calibration_regression,
)
from ml.publication import generate_methods_section                   # noqa: E402

#: The false clause, verbatim from the row.
FALSE_CLAUSE = "assessed using reliability diagrams"

#: The three probability-scale quantities the clause named.
PROBABILITY_QUANTITIES = ("reliability diagram", "Brier score",
                          "expected calibration error")


def _methods(task_type: str, explainability):
    """`generate_methods_section` with the arguments `pages/10`'s fallback
    supplies, and nothing else — the point is which sentence it composes."""
    return generate_methods_section(
        data_config={"feature_cols": ["age", "bmi"], "target_col": "y"},
        preprocessing_config={},
        model_configs={"ridge": {}},
        split_config={"stratify": False},
        n_total=500, n_train=300, n_val=100, n_test=100,
        feature_names=["age", "bmi"],
        target_name="y",
        task_type=task_type,
        metrics_used=["RMSE"] if task_type == "regression" else ["AUROC"],
        explainability_methods=list(explainability),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1 · the numbers the sentence is a claim about
# ─────────────────────────────────────────────────────────────────────────────

def _continuous(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.normal(50, 10, n)
    return y, y * 0.8 + rng.normal(0, 3, n)


def _binary(n=400, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.05, 0.95, n)
    return (rng.uniform(size=n) < p).astype(int), p


def test_a_regression_run_computes_no_brier_no_ece_and_nothing_to_draw_a_diagram_from():
    """The premise the sentence has to match, observed on the shipped function
    rather than read off its docstring."""
    y, pred = _continuous()
    cal = calibration_regression(y, pred, model_name="ridge")

    assert cal.calibration_slope is not None
    assert cal.calibration_intercept is not None
    assert cal.calibration_r2 is not None

    assert cal.brier_score is None, "a regression run produced a Brier score"
    assert cal.ece is None, "a regression run produced an ECE"
    for field in ("bin_edges", "bin_true_freq", "bin_pred_mean", "bin_counts"):
        assert getattr(cal, field) is None, (
            f"a regression run produced {field}, which is what a reliability "
            f"diagram is drawn from")


def test_a_binary_run_does_compute_all_three_so_the_claim_is_true_there():
    y, proba = _binary()
    cal = calibration_classification(y, proba, model_name="ridge")
    assert cal.brier_score is not None
    assert cal.ece is not None
    assert cal.bin_edges is not None


# ─────────────────────────────────────────────────────────────────────────────
# 2 · the sentence
# ─────────────────────────────────────────────────────────────────────────────

def test_the_regression_methods_section_does_not_claim_the_three_it_never_computed():
    text = _methods("regression", ["calibration"])
    assert FALSE_CLAUSE not in text, (
        "the regression Methods section still says calibration was "
        f"{FALSE_CLAUSE}: {text[-900:]}")


def test_the_regression_methods_section_states_what_it_did_assess():
    """Not silence — the corrected claim. Slope, intercept and R², named with
    the values that mean 'perfect', which is what makes the numbers readable."""
    text = _methods("regression", ["calibration"])
    assert "regressing the observed" in text, text[-900:]
    for named in ("calibration slope", "calibration intercept", "R²"):
        assert named in text, f"{named!r} missing from: {text[-900:]}"


def test_the_regression_methods_section_discloses_the_three_it_did_not_compute():
    """§A5.3 names Brier score; a reader who knows the checklist will look for
    it. Saying nothing would leave them to assume it was computed and omitted."""
    text = _methods("regression", ["calibration"])
    assert "were not computed for this continuous outcome" in text, text[-900:]
    for quantity in PROBABILITY_QUANTITIES:
        assert quantity.lower() in text.lower(), (
            f"the disclosure does not name {quantity!r}: {text[-900:]}")


def test_the_classification_methods_section_keeps_the_claim_that_is_true_there():
    """The shelf is not shortened. `pages/06` computes all three on the binary
    branch, so the original sentence is correct for a classification run and
    stays."""
    text = _methods("classification", ["calibration"])
    assert FALSE_CLAUSE in text, text[-900:]
    assert "Brier score" in text and "expected calibration error" in text, text[-900:]


def test_no_other_selected_method_acquires_the_calibration_claim():
    """A guard on the branch itself: the sentence is chosen per method, so a
    regression run that mentions SHAP and permutation importance must carry no
    calibration language at all."""
    text = _methods("regression", ["shap", "permutation_importance"])
    for quantity in PROBABILITY_QUANTITIES:
        assert quantity.lower() not in text.lower(), text[-900:]
    assert "calibration slope" not in text, text[-900:]


# ─────────────────────────────────────────────────────────────────────────────
# 3 · `GUIDED-097`'s third shape, and what driving it found
# ─────────────────────────────────────────────────────────────────────────────

def test_a_string_outcome_computes_no_calibration_at_all():
    """A `"yes"`/`"no"` target, driven rather than assumed.

    `calibration_classification` casts `y_true` to float, so a string outcome
    raises. `pages/06_Train_and_Compare.py:2166` catches per model, the
    calibration dict stays empty, and `pages/10_Report_Export.py:1838` therefore
    never adds `calibration` to the methods multiselect's options — so the
    section says nothing about calibration, which is silence rather than a false
    claim. Both halves are asserted here because the second is what makes the
    first safe.
    """
    y01, proba = _binary()
    ystr = np.where(y01 == 1, "yes", "no")
    with pytest.raises(ValueError):
        calibration_classification(ystr, proba, model_name="ridge")

    text = _methods("classification", [])
    for quantity in PROBABILITY_QUANTITIES:
        assert quantity.lower() not in text.lower(), (
            f"a run that computed no calibration still names {quantity!r}: "
            f"{text[-900:]}")


#: NOT COVERED, said out loud — `GUIDED-097`'s second clause.
#:
#: MULTICLASS. A 3+ class target reaches `pages/06:2151-2157`, which declines to
#: compute anything (`"calibration curves are shown for binary classification
#: only"`) and `continue`s, so the calibration dict is empty for every model and
#: `pages/10`'s gate never offers the method. Checked by reading that branch
#: rather than driven, because reaching it needs a trained multiclass model with
#: `predict_proba` and a fitted preprocessing pipeline in Streamlit session
#: state. The claim is therefore *this file did not drive multiclass*, not *the
#: multiclass path is proven silent*.
#:
#: TIME-TO-EVENT. No calibration path exists; `GUIDED-105`/`GUIDED-118` are the
#: rows and the family is L53.

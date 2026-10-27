"""Feature-wise regression refuses its table where a study factor's declared form enters it as
other than one straight-line column (the verifier's B1 and B2 on the swarm's C2 repair).

Feature-wise regression tests each study factor as one column, adjusted for the rest, its
estimate "per unit of what you study" (``FeatureWise.describe``). A declared spline enters the
factor as its spline terms (``x``, ``x'``, ``x''``), so testing ``x`` beside its own nonlinear
terms reported the slope below the first knot as the factor's estimate (0.419 on the verifier's
construction, against 0.557 for the straight line, the data built linear); quintiles, categories,
a cut point or a mass at zero's two parts enter it as indicators, and it was refused as having no
column of its name, under an energy adjustment in words that gave the role to the energy step's
column (``sugar_adj``), which has none. Now its table is refused, no number in it, in words that
name the column the roles answer gave the role, under an energy adjustment with the name that step
gives it, and its form, with the straight line as the way forward. The table is refused, not the
fit, so every family chosen beside it is still fit. The refusal of a study factor the matrix holds
no column of names it the same way.

The guard stays silent on a study factor entered as a straight line, beside a covariate's spline:
its estimate and standard error are statsmodels' least squares on columns built here, the
covariate's restricted cubic spline from Harrell's truncated-power formula at the knots the form
step placed (any basis of the same knots spans the same columns, so the factor's coefficient is
the same).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

import turbotab.core.models  # noqa: F401 - registers the families
from turbotab.core.models import featurewise as fw
from turbotab.core.models.pipeline import DesignSpec, build_pipeline

LEAD = ("Feature-wise regression tests each study factor as one straight-line column, one at a "
        "time, adjusted for the rest")
NO_FACTOR = ("No study factor to test: feature-wise regression tests the columns with the role of "
             "what you study, one at a time, adjusted for the rest")
ANOTHER = {"label": "Choose another model family", "decision": None}
RESIDUAL = {"method": "residual", "energy_column": "kcal", "nutrients": ["sugar"]}
SUGAR = {"sugar": "exposure", "kcal": "energy", "age": "covariate"}


def _intake(n: int = 200, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """The verifier's construction: the outcome linear in `sugar` and `age`."""
    rng = np.random.default_rng(seed)
    kcal = rng.normal(2000, 300, n)
    frame = pd.DataFrame({"sugar": 0.03 * kcal + rng.normal(0, 10, n), "kcal": kcal,
                          "age": rng.normal(50, 10, n)})
    frame["drinks"] = np.where(rng.random(n) < 0.3, 0.0, rng.gamma(2.0, 3.0, n))
    y = (0.5 * frame["sugar"] + 0.1 * frame["age"] + rng.normal(0, 5, n)).to_numpy()
    return frame, y


def _pipeline(roles: dict[str, str], forms: dict, energy: dict | None) -> object:
    predictors = list(roles)
    spec = DesignSpec(predictors=predictors, inputs=predictors, categorical=[],
                      numeric=predictors, energy=energy, impute=False, roles=dict(roles),
                      exposure_forms=dict(forms))
    return build_pipeline(spec, fw.FEATUREWISE, "regression", "inference", 200, len(predictors))


def _table(frame: pd.DataFrame, y: np.ndarray, roles: dict[str, str], forms: dict,
           energy: dict | None = None) -> tuple[object, object]:
    pipeline = _pipeline(roles, forms, energy)
    pipeline.fit(frame[list(roles)], y)
    return pipeline, fw.FEATUREWISE.inference(pipeline, frame[list(roles)], y, task="regression",
                                              clusters=None)


def _line(column: str) -> dict:
    """The straight line for ``column``, as the decision model reads the answer."""
    from turbotab.core.decisions import SetExposureForm

    return {"label": f"Enter `{column}` as a straight line",
            "decision": SetExposureForm(column=column, form="linear").model_dump(mode="json")}


SPLINE = {"form": "spline", "knots": 4}
FORMS = {
    "spline": (SPLINE, "a restricted cubic spline", "its spline terms"),
    "quintiles": ({"form": "quintiles"}, "quintiles", "indicators of its quintiles"),
    "categories": ({"form": "categories", "cuts": [55.0, 65.0]},
                   "categories at declared cut points", "indicators of its categories"),
    "optimal": ({"form": "optimal"}, "a data-derived cut point",
                "an indicator of the values above the cut"),
}


def _refused(table: object, words: str, exits: list[dict]) -> None:
    """A refused table: no row, so no number; the words its reason, its caption's and its first
    concern; its ways forward."""
    assert table.rows == []
    assert table.info["refused"] == words
    assert table.info["caption"] == f"No tests: {words}"
    assert table.info["exits"] == exits
    assert table.concerns[0] == words


@pytest.mark.parametrize("form", list(FORMS))
def test_a_study_factor_in_a_declared_form_has_its_table_refused_naming_it_and_its_form(form):
    frame, y = _intake()
    spec, label, enters = FORMS[form]
    pipeline, table = _table(frame, y, SUGAR, {"sugar": spec})
    words = (f"{LEAD}; `sugar` has that role, and its declared form, {label}, enters it as "
             f"{enters}, not one straight-line column.")
    _refused(table, words, [_line("sugar"), ANOTHER])
    _refused(pipeline[-1].table_, words, [_line("sugar"), ANOTHER])  # the model step's own


@pytest.mark.parametrize("form", ["spline", "quintiles"])
def test_under_an_energy_adjustment_it_names_the_roles_column_and_the_energy_steps(form):
    """The verifier's probes: `sugar` has the role; `sugar_adj` is the column the residual method
    makes of it, and has none."""
    frame, y = _intake()
    spec, label, enters = FORMS[form]
    _, table = _table(frame, y, SUGAR, {"sugar": spec}, RESIDUAL)
    _refused(table, (f"{LEAD}; `sugar` (entered as `sugar_adj` by the energy adjustment) has that "
                     f"role, and its declared form, {label}, enters it as {enters}, not one "
                     f"straight-line column."), [_line("sugar"), ANOTHER])


def test_a_mass_at_zero_and_several_formed_study_factors_are_each_named():
    frame, y = _intake()
    roles = {"sugar": "exposure", "drinks": "exposure", "kcal": "energy", "age": "covariate"}
    _, table = _table(frame, y, roles, {"sugar": SPLINE,
                                        "drinks": {"form": "zero_spline", "knots": 3}})
    _refused(table, (f"{LEAD}; `sugar` and `drinks` have that role, and their declared forms enter "
                     f"each otherwise than as one straight-line column: `sugar`, a restricted "
                     f"cubic spline, as its spline terms; `drinks`, non-consumers apart beside a "
                     f"spline among consumers, as a consumer indicator and its spline terms."),
             [_line("sugar"), _line("drinks"), ANOTHER])


def test_several_formed_study_factors_under_an_energy_adjustment_name_its_columns():
    frame, y = _intake()
    roles = {"sugar": "exposure", "age": "exposure", "kcal": "energy"}
    _, table = _table(frame, y, roles, {"sugar": SPLINE, "age": {"form": "quintiles"}}, RESIDUAL)
    _refused(table, (f"{LEAD}; `sugar` and `age` have that role, and their declared forms enter "
                     f"each otherwise than as one straight-line column: `sugar` (entered as "
                     f"`sugar_adj` by the energy adjustment), a restricted cubic spline, as its "
                     f"spline terms; `age`, quintiles, as indicators of its quintiles."),
             [_line("sugar"), _line("age"), ANOTHER])


def test_the_way_forward_is_an_answer_the_decision_model_reads():
    from turbotab.core import decisions as d

    frame, y = _intake()
    _, table = _table(frame, y, SUGAR, {"sugar": SPLINE}, RESIDUAL)
    taken = d.parse_decision(table.info["exits"][0]["decision"])
    assert isinstance(taken, d.SetExposureForm) and (taken.column, taken.form) == ("sugar",
                                                                                   "linear")


# ── silent: a study factor entered as a straight line ────────────────────────


def _rcs(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Harrell's restricted cubic spline's nonlinear terms, from his truncated-power formula (RMS
    §2.4.5): for j = 1 … k − 2, (x − t_j)³₊ − (x − t_{k−1})³₊ (t_k − t_j)/(t_k − t_{k−1})
    + (x − t_k)³₊ (t_{k−1} − t_j)/(t_k − t_{k−1}); unscaled, which spans the same columns."""
    t = np.asarray(knots, dtype=float)
    k = len(t)
    plus = lambda v: np.maximum(v, 0.0) ** 3  # noqa: E731
    return np.column_stack([plus(x - t[j]) - plus(x - t[k - 2]) * (t[k - 1] - t[j]) / (t[k - 1] - t[k - 2])
                            + plus(x - t[k - 1]) * (t[k - 2] - t[j]) / (t[k - 1] - t[k - 2])
                            for j in range(k - 2)])


def _ols(y: np.ndarray, *columns: np.ndarray) -> tuple[float, float]:
    fit = sm.OLS(y, sm.add_constant(np.column_stack(columns))).fit()
    return float(fit.params[1]), float(fit.bse[1])


@pytest.mark.parametrize("declared", [{}, {"sugar": {"form": "linear"}}], ids=["no form", "linear"])
def test_a_straight_line_study_factor_is_estimated_as_least_squares(declared):
    frame, y = _intake()
    _, table = _table(frame, y, SUGAR, declared)
    (row,) = table.rows
    estimate, se = _ols(y, frame["sugar"], frame["kcal"], frame["age"])
    assert row["feature"] == "sugar"
    assert row["estimate"] == pytest.approx(estimate, rel=1e-9)
    assert row["se"] == pytest.approx(se, rel=1e-9)


def test_a_straight_line_study_factor_beside_a_covariates_spline_is_estimated_adjusted_for_it():
    frame, y = _intake()
    pipeline, table = _table(frame, y, SUGAR, {"age": SPLINE}, RESIDUAL)
    knots = pipeline.named_steps["form"].knots_["age"]
    (row,) = table.rows
    kcal, sugar = frame["kcal"].to_numpy(), frame["sugar"].to_numpy()
    residual = sugar - np.polyfit(kcal, sugar, 1)[0] * (kcal - kcal.mean())
    age = frame["age"].to_numpy()
    estimate, se = _ols(y, residual, kcal, age, _rcs(age, knots))
    assert row["feature"] == "sugar_adj"
    assert row["estimate"] == pytest.approx(estimate, rel=1e-8)
    assert row["se"] == pytest.approx(se, rel=1e-8)


# ── the refusal of no column of the study factor's name (B2) ─────────────────


def test_a_study_factor_the_matrix_holds_no_column_of_is_named_as_the_roles_answer_gave_it():
    """Under an energy adjustment the column sought is the energy step's (`sugar_adj`); the role is
    `sugar`'s, and the words say so."""
    frame, y = _intake()
    table = fw.featurewise_table(frame[["kcal", "age"]], y, "regression", ["sugar_adj"],
                                 named=["sugar"])
    _refused(table, (f"{NO_FACTOR}; `sugar` (entered as `sugar_adj` by the energy adjustment) has "
                     f"that role, but the model matrix holds no column of that name."), [ANOTHER])


def test_without_an_energy_adjustment_the_words_are_as_they_were():
    frame, y = _intake()
    table = fw.featurewise_table(frame[["kcal", "age"]], y, "regression", ["sugar"],
                                 named=["sugar"])
    _refused(table, (f"{NO_FACTOR}; `sugar` has that role, but the model matrix holds no column of "
                     f"that name."), [ANOTHER])

"""The multiple-imputation method's served label and sound sentence say only what the plan's fit
does (pp2-mi-words, MW1: the fe-missing-mi verifiers' V2 and V3).

* **The label states m's rule as the rule.** It sits beside the researcher's own m on the
  missing-values card, so it states no count of its own: "m at least 20, more when more rows are
  incomplete" was false for an m of 100 or more, where no share of the rows is larger and the fit
  draws the m asked. Its rule (the number asked, 20 by default, or the percentage of rows the
  imputation model fills, rounded up, if that is larger) is checked against White, Royston &
  Wood's rule written out here, over every m the contract accepts.
* **The label's count is the one each path draws.** The joint imputation counts the rows its model
  fills; the feature-wise tests count in whichever study factor's model fills the most rows
  (``featurewise_mi.imputation_count``), checked on blanks spread over two factors, where "the
  percentage of rows with an imputed value" gave twice the count drawn.
* **Total energy is named only where the imputation model holds a column declared it.** The
  imputation model holds the primary model's inputs (``missing.imputation_frame`` over
  ``pipeline.model_predictors``), a column declared total energy among them unless the adjustment
  answers leave it out; the sentence said "with the outcome and energy" on every inference
  project, one with no energy column included, and named a declared one the answers left out.
* **Interval uncertainty is claimed only for a family whose table pools the imputations.** A
  family with no coefficient table (elastic net, boosted trees) is fit on the single fill and draws
  no copy (``stages.modeling._missing_for_table``), where the sentence said "its intervals carry
  the imputation's uncertainty" whatever was chosen. The card is served before the families are
  chosen (the proposals stage does not read them), so the sentence states the condition.

Expected values come from the construction (the columns given a role, the rows left blank, the
adjustment answers and the role VanderWeele's criterion derives from them), the contract's own
bounds and default for m (``SetMissing``), and White's rule as its source states it.
"""
from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import turbotab.core.models  # noqa: F401 - registers the families
from turbotab.core import custom_sound
from turbotab.core.decisions import (CovariateAnswers, MissingSpec, ProjectState, SetAdjustment,
                                     SetEstimand, SetMissing, fold_onto, validate)
from turbotab.core.methods.imputation import m_rule
from turbotab.core.methods.missing import METHODS, MI_EXIT_LABEL, MI_LABEL, methods_for
from turbotab.core.models import families, get_family

MI = "multiple_imputation"
# The two halves of the sound sentence that do not depend on the plan: the families' clause, the
# same for every plan, since the card is served before any family is chosen.
POOLED = ("a model family with a coefficient table pools its estimates over the imputations, so "
          "its intervals carry the imputation's uncertainty; a family without one is fit on a "
          "single fill learned without the outcome.")


def _sound(held: str) -> str:
    return (f"Sound: with {held} in the imputation model it is unbiased under missing at random "
            f"(Moons et al. 2006), and {POOLED}")


# ── the label: m's rule as the rule ──────────────────────────────────────────


def _whites_rule(asked: int, n_incomplete: int, n_rows: int) -> int:
    """White, Royston & Wood (2011), as White, Pandis & Pham put it: "the number of imputations
    should be at least equal to the percentage of incomplete cases"; never fewer than the number
    asked. The fewest whole imputations that are both."""
    return max(asked, math.ceil(Fraction(100 * n_incomplete, n_rows)))


def test_the_label_states_ms_rule_true_for_every_m_the_card_accepts():
    """Every m the contract accepts, every share of rows with an imputed value on 1 to 300 rows:
    the fit's m is the label's rule. The words it replaces claimed more imputations when more rows
    are incomplete, true below an m of 100 (38 of 100 rows draw 38 beside 20) and false from 100
    on, where no share of the rows is larger than m and the fit draws the m asked."""
    bounds = SetMissing.model_json_schema()["properties"]["m"]
    low, high = int(bounds["minimum"]), int(bounds["maximum"])
    default = SetMissing(strategy=MI).m
    assert (low, high, default) == (20, 200, 20)
    for asked in range(low, high + 1):
        for n_rows in (1, 3, 7, 100, 300):
            for n_incomplete in range(n_rows + 1):
                assert m_rule(n_incomplete, n_rows, asked) == _whites_rule(asked, n_incomplete,
                                                                           n_rows)
    assert m_rule(38, 100, 20) == 38
    assert all(m_rule(k, 300, asked) == asked for asked in (100, 150, 200) for k in range(301))
    assert MI_LABEL == METHODS[MI].label == f"Multiple imputation ({_rule(default)})"
    assert MI_EXIT_LABEL.endswith(f" ({_rule(default)})")
    for label in (MI_LABEL, MI_EXIT_LABEL):
        assert "at least" not in label and "more when" not in label


def _rule(default: int) -> str:
    return (f"m = the number asked, {default} by default, or the percentage of rows the imputation "
            f"model fills, rounded up, if that is larger; for feature-wise tests, counted in "
            f"whichever study factor's model fills the most rows")


def test_the_label_counts_m_where_each_path_counts_it():
    """300 rows; study factor `A` blank on rows 0–89, `B` on rows 90–179, `age` complete; m 20
    asked. The joint imputation fills 180 rows, so it draws ⌈100 × 180 / 300⌉ = 60; the
    feature-wise tests impute each factor in its own model, each filling 90 rows, and draw
    ⌈100 × 90 / 300⌉ = 30, where "the percentage of rows with an imputed value" said 60. The label
    states both counts."""
    from turbotab.core.methods.featurewise_mi import imputation_count as featurewise_count
    from turbotab.core.models.pipeline import imputation_count as joint_count

    n = 300
    rng = np.random.default_rng(3)
    matrix = pd.DataFrame({"A": rng.normal(size=n), "B": rng.normal(size=n),
                           "age": rng.normal(size=n)})
    matrix.loc[0:89, "A"] = np.nan
    matrix.loc[90:179, "B"] = np.nan
    assert int(matrix[["A", "B"]].isna().any(axis=1).sum()) == 180
    assert _whites_rule(20, 180, n) == 60 and _whites_rule(20, 90, n) == 30
    assert joint_count(None, matrix, [], None, {"strategy": MI, "m": 20}) == 60
    assert featurewise_count(matrix, ["A", "B"], ["age"], 20) == 30
    # Blanks in an adjustment column are filled in every factor's model: `age` blank on rows
    # 170–199 adds 20 rows to B's model (rows 180–199) and 30 to A's, so B's fills 110 and A's 120.
    matrix.loc[170:199, "age"] = np.nan
    assert _whites_rule(20, 120, n) == 40
    assert featurewise_count(matrix, ["A", "B"], ["age"], 20) == 40
    assert MI_LABEL == f"Multiple imputation ({_rule(SetMissing(strategy=MI).m)})"
    assert "rows with an imputed value" not in MI_LABEL


# ── the sound sentence: total energy only where declared ─────────────────────


def _frame(n: int = 200) -> pd.DataFrame:
    """`x` blank on its first 40 rows; `kcal` a total-energy-like column, whatever its role."""
    rng = np.random.default_rng(11)
    frame = pd.DataFrame({"y": rng.normal(size=n), "x": rng.normal(size=n),
                          "age": rng.integers(20, 80, size=n).astype(float),
                          "kcal": rng.normal(2000, 400, size=n)})
    frame.loc[frame.index[:40], "x"] = np.nan
    return frame


def _state(roles: dict[str, str], lens: list[str] | None = None,
           purpose: str = "inference") -> ProjectState:
    return ProjectState(target="y", task="regression", purpose=purpose, roles=roles, lens=lens)


def _card(state: ProjectState) -> tuple[dict[str, Any], dict[str, Any]]:
    """The missing-values card's multiple imputation, as the proposals serve it on ``state``: the
    method (``missing.methods``) and its customary-and-sound option (``labels.missing``)."""
    from turbotab.core.stages.proposals import build_proposals

    frame = _frame()
    columns = [{"name": c, "dtype": "numeric", "n_unique": int(frame[c].nunique()),
                "n_missing": int(frame[c].isna().sum())} for c in frame.columns]
    out = build_proposals(frame, columns, lens=state.lens, target="y", roles=state.roles,
                          purpose=state.purpose, state=state)
    method = next(m for m in out["missing"]["methods"] if m["key"] == MI)
    option = next(o for o in out["labels"]["missing"]["options"] if o["key"] == MI)
    return method, option


def _imputation_columns(state: ProjectState) -> list[str]:
    """The columns the primary model's imputation model holds on ``state`` once multiple
    imputation is answered (``missing.imputation_frame`` over ``pipeline.model_predictors``), the
    outcome's terms aside."""
    from turbotab.core.methods.missing import imputation_frame
    from turbotab.core.models.pipeline import design_spec, model_predictors

    frame = _frame()
    st = state.model_copy(update={"missing": MissingSpec(strategy=MI)})
    spec = design_spec(st, frame.drop(columns="y"), model_predictors(st))
    data, *_ = imputation_frame(spec, frame[list(spec.inputs)], frame["y"], "regression")
    return [c for c in data.columns if not str(c).startswith("__")]


def test_the_sound_names_total_energy_where_a_column_is_declared_it():
    """`kcal` declared total energy: the imputation model holds it, and the card's sentence names
    it, on the dietary lens's card and on one without it."""
    roles = {"x": "exposure", "age": "covariate", "kcal": "energy"}
    want = _sound("the outcome and total energy (`kcal`)")
    for lens in (["dietary"], None):
        assert "kcal" in _imputation_columns(_state(roles, lens)), lens
        method, option = _card(_state(roles, lens))
        assert method["sound"] == want, lens
        assert option["sound"]["reason"] == want, lens
    assert methods_for("inference", energy=["kcal"])[0]["sound"] == want


# The modified disjunctive cause criterion's answers (VanderWeele 2019; ``estimand.derive``): a
# cause of neither the exposure nor the outcome is left out of the primary model, a cause of both
# is adjusted for, and one left out but "further adjusted for" is in the secondary model only.
NEITHER = {"causes_exposure": "no", "causes_outcome": "no", "after_exposure": "no"}
BOTH = {"causes_exposure": "yes", "causes_outcome": "yes", "after_exposure": "no"}


def _answered(kcal: dict[str, Any]) -> ProjectState:
    """`kcal` declared total energy without the dietary lens, where it is a covariate the
    adjustment question asks about; the estimand and the adjustment answers recorded as the
    contract takes them (``validate``, then ``fold_onto``)."""
    st = _state({"x": "exposure", "age": "covariate", "kcal": "energy"})
    for decision in (SetEstimand(exposure="x", effect="total", measure="mean_difference"),
                     SetAdjustment(exposure="x", answers={"age": CovariateAnswers(**BOTH),
                                                          "kcal": CovariateAnswers(**kcal)})):
        validate(decision, {"state": st})
        st = fold_onto(st, decision)
    return st


def test_the_sound_names_energy_only_where_the_adjustment_answers_keep_it():
    """Without the dietary lens a column declared total energy is a covariate the adjustment
    answers place. Answered a cause of neither, it is left out of every model: the imputation
    model holds `x` and `age` only, and the sentence names no energy (it named `kcal`). Answered a
    cause of both, it is adjusted for: the imputation model holds it and the sentence names it.
    Answered a cause of neither but "further adjusted for", it is in the secondary model only,
    whose own imputation model holds it (``stages.secondary``); the sentence speaks of the
    primary model's, which does not, and names none."""
    from turbotab.core.estimand import adjustment_left_out, secondary_columns

    left_out = _answered(NEITHER)
    assert adjustment_left_out(left_out) == ["kcal"] and secondary_columns(left_out) == []
    assert _imputation_columns(left_out) == ["x", "age"]
    kept = _answered(BOTH)
    assert adjustment_left_out(kept) == [] and secondary_columns(kept) == []
    assert _imputation_columns(kept) == ["x", "age", "kcal"]
    further = _answered({**NEITHER, "further": True})
    assert adjustment_left_out(further) == ["kcal"] and secondary_columns(further) == ["kcal"]
    assert _imputation_columns(further) == ["x", "age"]
    for state, held in ((left_out, "the outcome"),
                        (kept, "the outcome and total energy (`kcal`)"),
                        (further, "the outcome")):
        method, option = _card(state)
        assert method["sound"] == option["sound"]["reason"] == _sound(held), held


def test_the_sound_names_no_energy_where_no_column_is_declared_it():
    """`kcal` in the table with no role: the imputation model does not hold it, and the sentence
    names no energy (it said "with the outcome and energy" on every inference project)."""
    roles = {"x": "exposure", "age": "covariate"}
    assert "kcal" not in _imputation_columns(_state(roles))
    want = _sound("the outcome")
    method, option = _card(_state(roles))
    assert method["sound"] == option["sound"]["reason"] == want
    assert "energy" not in want
    assert methods_for("inference", energy=())[0]["sound"] == want
    # A caller that does not know the plan says where energy is held, and names none: where the
    # analysis model holds a declared column, not wherever one is declared (the adjustment
    # answers can leave it out; test_the_sound_names_energy_only_where_the_adjustment_answers_keep_it).
    unknown = METHODS[MI].sound["inference"]
    assert unknown == methods_for("inference")[0]["sound"] == _sound(
        "the outcome (and total energy, where the analysis model holds a column declared it)")
    reason = next(o for o in custom_sound.missing("inference").options if o.key == MI).sound.reason
    assert reason == unknown


def test_under_prediction_the_sentence_is_unchanged_whatever_the_energy():
    """Under prediction multiple imputation is refused with its own reason; the energy answer
    does not reach it."""
    words = ("Not here: it uses the outcome, which a new row does not have; impute in each fold "
             "without it (Sisk et al. 2023).")
    for energy in (None, (), ["kcal"]):
        assert next(m for m in methods_for("prediction", energy=energy)
                    if m["key"] == MI)["sound"] == words
    method, _ = _card(_state({"x": "exposure", "kcal": "energy"}, purpose="prediction"))
    assert method["sound"] == words


# ── the sound sentence: interval uncertainty only where the table pools ──────


def test_interval_uncertainty_is_claimed_only_for_a_family_whose_table_pools_the_copies(
        tmp_path: Path):
    """112 of 300 rows with `activity` blank (``ef.cohort``), multiple imputation asked with m 20.
    Elastic net alone, with no coefficient table, draws no copy: it is fit on the single fill.
    The linear model's table draws ⌈100 × 112 / 300⌉ = 38 copies and pools them. The sentence
    claims the pooled intervals for a family with a table and says one without is fit on a single
    fill; every family with a table pools its imputations, so the claim holds for each one."""
    from turbotab.core.tests.test_imputation_count_and_spline_refusal import _fit_under_mi

    (tmp_path / "alone").mkdir()
    (tmp_path / "table").mkdir()
    assert get_family("elastic_net").inference is None
    _, alone = _fit_under_mi(tmp_path / "alone", ["elastic_net"])
    assert (alone.objects or {}).get("imputations") is None
    assert get_family("linear").inference is not None
    _, pooled = _fit_under_mi(tmp_path / "table", ["linear"])
    assert len(pooled.objects["imputations"]["frames"]) == math.ceil(Fraction(100 * 112, 300))
    [linear] = pooled.data["models"]
    assert linear["inference"]["missing"]["m"] == 38
    for family in families():
        if family.inference is not None:
            assert getattr(family, "pools_imputations", True), family.key
    assert METHODS[MI].sound["inference"].endswith(POOLED)
    for energy in (None, (), ["kcal"]):
        sound = methods_for("inference", energy=energy)[0]["sound"]
        assert sound.endswith(POOLED)
        assert sound.count("intervals") == 1

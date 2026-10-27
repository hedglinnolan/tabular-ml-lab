"""A spline whose values are too heavily tied to place three distinct knots is a typed refusal in
words (the swarm's C1: 17 of the plasma panel's 40 walks ended ``fit_failed`` or a stage failed on
``ValueError: Fewer than 3 distinct knots can be placed``).

Where the walks met it: ``mz_0126`` (86 of the 100 analyzed rows blank) under a spline of 4 knots,
the median fill before the form step giving 74 of an 80-row fold's values one value (and the
explain stage's bootstrap refits the like). On every analyzed row its recorded values, and its
values after the fill, place the knots; so the engine knows the tie only when the form step is fit
on the rows a fit sees, and it refuses there, as it refuses a spline on too few values
(``spline_too_few_values``): coded, its words counting the values, the distinct ones and the most
that share one value, with the ways forward the refusal shape offers (another form).

The truth is ``Hmisc::rcspline.eval``'s default placement worked by hand (it stops with "fewer
than 3 unique knots"): 80 values, 2 below 400, 74 at 400 and 4 above, 4 knots. Type-7 quantiles at
0.05, 0.35, 0.65 and 0.95 sit at sorted positions 79p = 3.95, 27.65, 51.35 and 75.05: positions 2
to 75 hold 400, so the first three are 400 and the last is 400 + 0.05 × (410 − 400) = 400.5, two
distinct knots; the eight percentiles from 0.05 to 0.95 it tries next sit at 3.95 … 75.05, every
one 400 but the last, still two. A column of four values, ten rows each, has two interior values,
so two knots (``rcspline.eval``'s ``nk >= n.unique - 2`` branch). And it stays silent where three
distinct knots can be placed: 100 values, 380 to 386, 86 at 400 and 410 to 416, give knots at
positions 99p = 4.95 (384.95), 34.65 and 64.35 (400) and 94.05 (411.05), three distinct.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from turbotab.core.graph import StageRefusal
from turbotab.core.jobs import failure_of
from turbotab.core.methods import exposure_form as ef

FOLD = [380.0, 390.0] + [400.0] * 74 + [410.0, 420.0, 430.0, 440.0]  # 80 values, 7 distinct
FOUR = [1.0] * 10 + [2.0] * 10 + [3.0] * 10 + [4.0] * 10  # 40 values, 4 distinct
EVERY_ROW = [380.0 + i for i in range(7)] + [400.0] * 86 + [410.0 + i for i in range(7)]


def _form_fit(values, form="spline", column="mz_0126"):
    step = ef.ExposureForms({column: {"form": form, "knots": 4}})
    return step.fit(pd.DataFrame({column: np.asarray(values, dtype=float)}))


def test_the_hand_worked_tie_places_two_distinct_knots_so_hmisc_stops():
    """The construction's arithmetic, by NumPy's type-7 quantiles (R's default)."""
    x = np.asarray(FOLD)
    assert np.unique(np.quantile(x, [0.05, 0.35, 0.65, 0.95])).tolist() == [400.0, 400.5]
    assert np.unique(np.quantile(x, np.linspace(0.05, 0.95, 8))).tolist() == [400.0, 400.5]


def test_a_spline_on_a_folds_tied_values_is_a_typed_refusal_counting_them():
    """The walks' case: 80 rows, each with a value (a fill may have given it), 7 distinct, 74 one
    value. A ``ValueError`` still (every caller that handles one handles it as before), coded,
    naming the column and saying what it counted."""
    with pytest.raises(StageRefusal) as refused:
        _form_fit(FOLD)
    exc = refused.value
    assert isinstance(exc, ValueError) and exc.code == "spline_too_tied"
    assert str(exc) == ("A spline of `mz_0126` needs at least 3 distinct knots, and its values are "
                        "too heavily tied to place them: its 80 values, one from each row it is "
                        "fit on, take 7 distinct values, 74 of them the same value.")
    assert [e["label"] for e in exc.exits] == [
        "Enter `mz_0126` as a straight line, which places no knots",
        "Enter `mz_0126` in categories at cut points declared from outside these data"]
    assert all(e["decision"] is None for e in exc.exits)


def test_the_refusal_is_served_as_a_refusal_in_the_same_words():
    """What the stage's status serves (``jobs.failure_of``): the ``ValueError`` words, and the
    refusal apart, with its code and both ways forward."""
    with pytest.raises(StageRefusal) as refused:
        _form_fit(FOLD)
    shown, refusal = failure_of(refused.value)
    assert shown == f"ValueError: {refused.value}"
    assert refusal is not None and refusal["code"] == "spline_too_tied"
    assert len(refusal["exits"]) == 2


def test_rows_that_leave_it_blank_are_counted_apart_from_the_values():
    """20 blank rows beside the 80 values: the rows that record it are counted among the rows it is
    fit on."""
    with pytest.raises(StageRefusal) as refused:
        _form_fit(FOLD + [np.nan] * 20)
    assert str(refused.value) == ("A spline of `mz_0126` needs at least 3 distinct knots, and its "
                                  "values are too heavily tied to place them: the 80 of the 100 "
                                  "rows it is fit on that record it take 7 distinct values, 74 of "
                                  "them the same value.")


def test_four_distinct_values_give_two_interior_knots_and_are_refused_the_same_way():
    """``rcspline.eval``'s other stop: as many knots as interior values, here two."""
    with pytest.raises(StageRefusal) as refused:
        ef.rcs_knots(np.asarray(FOUR), 4)
    assert refused.value.code == "spline_too_tied"
    assert str(refused.value) == ("A spline needs at least 3 distinct knots, and its values are "
                                  "too heavily tied to place them: its 40 values take 4 distinct "
                                  "values, 10 of them the same value.")
    assert [e["label"] for e in refused.value.exits] == [
        "Enter the column as a straight line, which places no knots",
        "Enter the column in categories at cut points declared from outside these data"]


def test_the_mass_at_zero_spline_counts_its_values_above_zero():
    """Among consumers: 300 non-consumers beside the four values ten times each, the knots placed
    on the 40 values above zero."""
    with pytest.raises(StageRefusal) as refused:
        ef.place({"form": "zero_spline", "knots": 4}, np.concatenate([np.zeros(300), FOUR]),
                 column="alc")
    assert refused.value.code == "spline_too_tied"
    assert str(refused.value) == ("A spline of `alc` among its values above zero needs at least 3 "
                                  "distinct knots, and its values are too heavily tied to place "
                                  "them: its 40 values above zero take 4 distinct values, 10 of "
                                  "them the same value.")


@pytest.mark.parametrize("values, knots", [
    # The tie on every row: 384 + 0.95 × (385 − 384), 400 twice, 411 + 0.05 × (412 − 411)
    (EVERY_ROW, [384.95, 400.0, 411.05]),
    (list(np.arange(1.0, 81.0)), None),  # an ordinary column: Harrell's four
], ids=["heavily_tied_but_placeable", "ordinary"])
def test_the_refusal_stays_silent_where_three_distinct_knots_can_be_placed(values, knots):
    step = _form_fit(values)
    placed = step.knots_["mz_0126"]
    if knots is not None:
        assert np.allclose(placed, knots)
    else:
        assert len(placed) == 4

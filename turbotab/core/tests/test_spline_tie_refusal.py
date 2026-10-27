"""A spline whose values are too heavily tied to place three distinct knots is a typed refusal in
words (the swarm's C1: 17 of the plasma panel's 40 walks ended ``fit_failed`` or a stage failed on
``ValueError: Fewer than 3 distinct knots can be placed``).

Where the walks met it: ``mz_0126`` (86 of the 100 analyzed rows blank) under a spline of 4 knots,
the median fill before the form step giving 74 of an 80-row fold's values one value (and the
explain stage's bootstrap refits the like). On every analyzed row its recorded values, and its
values after the fill, place the knots; so there the engine knows the tie only when the form step
is fit on the rows a fit sees, and it refuses there, as it refuses a spline on too few values
(``spline_too_few_values``): coded, its words counting the values, the distinct ones and the most
that share one value, with the ways forward the refusal shape offers (another form).

Where the values are known when the spline is declared (the verifier's M1), it is refused then, in
the same words with the same ways forward, the straight line one tap away: under inference the
coefficient table is fit on every analyzed row, so a column every analyzed row records, entering
the form step as recorded, gives the form step exactly those values. Elsewhere the fit decides:
an energy model's residual can break the tie (the construction below places four distinct knots
on it), a fill over blank rows changes the values, and under prediction the form step is fit on
a fold's rows.

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


# ── when the spline is declared: the analyzed rows' tie (the verifier's M1) ──────────────────────
# 1,000 analyzed values, 5 below 50.5, 950 at it and 45 above: 51 distinct, so the form question
# counts the column continuous and offers a spline. The commonest interior value's share, 0.95, is
# above ``fractied`` (0.05), so no knot is set aside for a tie at an end; Harrell's percentiles for
# 3 to 5 knots, and the 2k percentiles spread over the same range, lie in [0.05, 0.95], at type-7
# sorted positions 999p from 49.95 to 949.05, and positions 5 to 954 all hold 50.5: one distinct
# knot. Decimals, so the column reads as an amount (whole numbers would ask codes or amounts).

WIDE = [1.5, 2.5, 3.5, 4.5, 5.5] + [50.5] * 950 + [100.25 + i for i in range(45)]
EVERY_ROW_AMOUNTS = [v + 0.5 for v in EVERY_ROW]  # 100 values: knots 385.45, 400.5, 411.55
ROLES = {"pid": "identifier", "x": "exposure", "z": "covariate", "kcal": "covariate"}


class _Store:
    """A store a decision context reads (the server's ``DataStore`` shape)."""

    def __init__(self, frame: pd.DataFrame):
        self.frame = frame
        self.columns = list(frame.columns)

    def materialize(self, columns, rows=None):
        out = self.frame[list(columns)]
        return out if rows is None else out.iloc[np.asarray(rows, dtype=int)]

    def whole_numbers(self, columns):
        out = {}
        for c in columns:
            v = pd.to_numeric(self.frame[c], errors="coerce").dropna().to_numpy(float)
            if len(v) and np.all(v == np.round(v)):
                out[c] = {"whole": True, "zero_one": bool(set(np.unique(v)) <= {0.0, 1.0}),
                          "n_values": int(len(np.unique(v))), "min": float(v.min()),
                          "max": float(v.max())}
        return out

    def text_numbers(self, columns):
        return {}


def _answer_ctx(x, *, purpose="inference", rows="all", **slots):
    from turbotab.core import decisions as d
    from turbotab.core.tests import modeling_fixtures as mf

    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(7)
    n = len(x)
    frame = pd.DataFrame({"pid": np.arange(n), "x": x, "z": rng.normal(size=n) + 0.5,
                          "kcal": rng.normal(2000, 300, n), "y": rng.normal(size=n)})
    state = mf.state(roles=dict(ROLES), role_confirmations=dict(ROLES), target="y",
                     task="regression", purpose=purpose,
                     estimand=d.EstimandSpec(exposure="x", measure="mean_difference"), **slots)
    analyzed = (lambda: np.arange(n)) if rows == "all" else (lambda: None)
    return {"state": state, "store": lambda: _Store(frame), "analyzed": analyzed,
            "task": "regression"}, frame


def _tied_words(values, column="x"):
    """The fit's words, their counts from NumPy: the values, the distinct ones, the commonest's."""
    v = np.asarray(values, dtype=float)
    _, counts = np.unique(v, return_counts=True)
    return (f"A spline of `{column}` needs at least 3 distinct knots, and its values are too "
            f"heavily tied to place them: its {len(v):,} values, one from each row it is fit on, "
            f"take {len(counts):,} distinct values, {counts.max():,} of them the same value.")


def test_the_wide_tie_places_one_distinct_knot_by_type_7_quantiles():
    """The construction's arithmetic, by NumPy (R's default quantiles), and the fit's refusal."""
    x = np.asarray(WIDE)
    assert len(np.unique(x)) == 51 and len(x) == 1000
    for p in ([0.10, 0.50, 0.90], [0.05, 0.35, 0.65, 0.95], np.linspace(0.05, 0.95, 10)):
        assert np.unique(np.quantile(x, p)).tolist() == [50.5]
    with pytest.raises(StageRefusal) as refused:
        _form_fit(WIDE, column="x")
    assert str(refused.value) == _tied_words(WIDE)


@pytest.mark.parametrize("knots", [3, 4, 5, None], ids=["k3", "k4", "k5", "k_by_the_rule"])
def test_a_spline_the_analyzed_rows_cannot_place_is_refused_when_it_is_declared(knots):
    from turbotab.core import decisions as d

    ctx, _ = _answer_ctx(WIDE)
    decision = {"kind": "set_exposure_form", "column": "x", "form": "spline"}
    if knots is not None:
        decision["knots"] = knots
    with pytest.raises(d.Refusal) as refused:
        d.validate(decision, ctx)
    assert refused.value.code == "spline_too_tied"
    assert refused.value.message == _tied_words(WIDE)
    exits = refused.value.exits
    assert [e["label"] for e in exits] == [
        "Enter `x` as a straight line, which places no knots",
        "Enter `x` in categories at cut points declared from outside these data"]
    line = exits[0]["decision"]
    assert (line["kind"], line["column"], line["form"]) == ("set_exposure_form", "x", "linear")
    assert exits[1]["decision"] is None
    assert d.validate(line, ctx).form == "linear"  # the way forward is accepted


def test_the_one_tap_answer_is_refused_with_the_other_columns_forms_kept():
    from turbotab.core import decisions as d

    ctx, _ = _answer_ctx(WIDE)
    forms = {"x": {"form": "spline", "knots": 4}, "z": {"form": "spline", "knots": 4}}
    with pytest.raises(d.Refusal) as refused:
        d.validate({"kind": "set_forms", "forms": forms}, ctx)
    assert refused.value.code == "spline_too_tied"
    assert refused.value.message == _tied_words(WIDE)
    taken = refused.value.exits[0]["decision"]
    assert taken["kind"] == "set_forms"
    assert taken["forms"]["x"]["form"] == "linear"
    assert (taken["forms"]["z"]["form"], taken["forms"]["z"]["knots"]) == ("spline", 4)
    d.validate(taken, ctx)


def test_a_mass_at_zero_spline_is_refused_on_its_values_above_zero():
    from turbotab.core import decisions as d

    values = [0.0] * 200 + WIDE
    ctx, _ = _answer_ctx(values)
    with pytest.raises(d.Refusal) as refused:
        d.validate({"kind": "set_exposure_form", "column": "x", "form": "zero_spline",
                    "knots": 4}, ctx)
    assert refused.value.code == "spline_too_tied"
    assert refused.value.message == (
        "A spline of `x` among its values above zero needs at least 3 distinct knots, and its "
        "values are too heavily tied to place them: its 1,000 values above zero take 51 distinct "
        "values, 950 of them the same value.")


def test_a_heavily_tied_column_whose_knots_can_be_placed_is_accepted_and_placed_as_before():
    """The silent side: 86 of 100 values one value, three distinct knots by hand (the walks'
    every-row case, half a unit up), declared and then placed at 385.45, 400.5 and 411.55."""
    from turbotab.core import decisions as d

    ctx, _ = _answer_ctx(EVERY_ROW_AMOUNTS)
    done = d.validate({"kind": "set_exposure_form", "column": "x", "form": "spline",
                       "knots": 4}, ctx)
    assert (done.form, done.knots) == ("spline", 4)
    placed = _form_fit(EVERY_ROW_AMOUNTS, column="x").knots_["x"]
    assert np.allclose(placed, [385.45, 400.5, 411.55])


def _residual(frame):
    kcal, x = frame["kcal"].to_numpy(), frame["x"].to_numpy()
    return x - np.polyfit(kcal, x, 1)[0] * (kcal - kcal.mean())


def test_where_an_energy_residual_breaks_the_tie_the_declaration_is_not_refused():
    """The residual on `kcal` is the column the form step receives, and NumPy's type-7 quantiles
    place Harrell's four knots on it apart: the fit can place them, so the answer is not refused."""
    from turbotab.core import decisions as d

    adjustment = d.EnergyAdjustment(method="residual", energy_column="kcal", nutrients=["x"])
    ctx, frame = _answer_ctx(WIDE, energy_adjustment=adjustment)
    assert len(np.unique(np.quantile(_residual(frame), [0.05, 0.35, 0.65, 0.95]))) == 4
    done = d.validate({"kind": "set_exposure_form", "column": "x", "form": "spline",
                       "knots": 4}, ctx)
    assert done.form == "spline" and done.scale != "raw"


@pytest.mark.parametrize("case", ["blank_rows", "prediction", "rows_not_read"])
def test_where_the_fit_alone_knows_the_values_the_declaration_is_left_to_it(case):
    """A blank row takes a fill's value before the form step; under prediction the form step is
    fit on a fold's rows; and rows not yet read are never the whole table's in their place."""
    from turbotab.core import decisions as d

    values = WIDE + [np.nan] * 10 if case == "blank_rows" else WIDE
    ctx, _ = _answer_ctx(values, purpose="prediction" if case == "prediction" else "inference",
                         rows="none" if case == "rows_not_read" else "all")
    done = d.validate({"kind": "set_exposure_form", "column": "x", "form": "spline",
                       "knots": 4}, ctx)
    assert (done.form, done.knots) == ("spline", 4)

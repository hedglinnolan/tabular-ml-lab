"""How wide a chained-equations imputation model may be, judged by its width and its rows, the same
at the answer as at Fit (INBOX: Sullivan 2021's Haiku runs 22 and 26 locked multiple imputation
that Fit then refused at a fixed 80 columns, whatever the rows, after the lock).

The rule removes no capability the base offered (the orchestrator's ruling, 2026-11-02): an
imputation model of up to 80 columns (categories as indicators) is drawn whatever its rows, as on
the base, and a wider one is drawn where each incomplete column's model has the rows per parameter
the engine's rule asks for on the rows that record it, and refused only where one has not. The
rule: about 2 rows per predictor parameter for least squares (Austin & Steyerberg 2015, J Clin
Epidemiol 68:627), 10 events (its rarer value's rows) per parameter for a model of two levels
(Peduzzi et al. 1996, the engine's rule for logistic coefficients), and 10 per parameter of the
effective sample size n − Σn³/n² for a multinomial model, whose parameters are a slope per
predictor for each level beyond the first. The chain fits a numeric code the user declared a
category as its indicators, as the width counts it. Every expected value here is from construction
or a derivation: the parameters counted from the columns built, the rows and events from pandas,
the effective size by its formula, the multinomial model's slopes from R's ``nnet::multinom``, and
the width by R's ``ncol(model.matrix(~ ., data)) - 1`` on the imputation frame, its strata a
factor. Each boundary is tested on both sides (80 columns and 81; each rule's rows); the brief's
scale (about 90 columns: 8,000 rows imputed, 150 refused) through the sampler, through the answer
and through the plan's lock, and through the fit (and over HTTP,
``server/tests/test_imputation_width_api.py``).
Where the width rule refuses a model, the answer says what the sampler refuses first (a logged
column's zeros, a column with fewer than two recorded values, then the width) with the fit's ways
forward (each where the record accepts it), the fit's table is given the answer's set of columns
an exit may leave out and offers to leave out what the answer offers, and both read one lead;
where the width rule does not refuse the model, the answer refuses what the sampler refuses before
it draws, with what the fit's table offers then: to leave out any column but the exposure and
those the energy model is computed with (``test_imputation_agreement``).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from turbotab.core.decisions import MissingSpec, ProjectState, Refusal, SetMissing, validate
from turbotab.core.methods import smcfcs  # importing it registers the check at the answer
from turbotab.core.methods.imputation import ImputationRefused
from turbotab.core.methods.smcfcs import Variable, impute
from turbotab.core.tests.acceptance.r_reference import needs_r, run_r
from turbotab.core.tests.stage_harness import Ingested

CAP = 80  # the base's widest imputation model, drawn whatever its rows (the ruling of 2026-11-02)
SPV = 2  # Austin & Steyerberg 2015: about 2 subjects per variable for least-squares coefficients
EPV = 10  # Peduzzi et al. 1996: events per parameter (a multinomial model: effective size)
LEAD = "Multiple imputation cannot run on these data: "
SOURCE = ("where Austin & Steyerberg 2015 found about 2 per parameter enough for least-squares "
          "coefficients, standard errors and intervals")


def opening(width: int) -> str:
    """A refusal's first sentence: the model's width, and both conditions it is refused on."""
    return (f"The imputation model would hold {width:,} columns (categories as indicators): up to "
            f"{CAP} it is drawn whatever its rows, and wider only where each incomplete column's "
            f"model has the rows per parameter its rule asks for. ")


def effective(counts: list[int]) -> float:
    """Whitehead's effective sample size, n − Σ n_k³ / n², by its formula."""
    n = float(sum(counts))
    return n - sum(float(c) ** 3 for c in counts) / n ** 2


def per(counted: float, parameters: int) -> str:
    return f"{counted / parameters:.1f}"


def short_of(rows: int, parameters: int) -> str:
    """A least-squares model after the first, in its own rows and its own rule."""
    return (f"{rows:,} rows on {parameters} parameters, {per(rows, parameters)} per parameter "
            f"where its rule asks for about 2")


def numbers(n: int, k: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    return {f"c{j:02d}": rng.normal(size=n) for j in range(1, k + 1)}


# ── the width is R's model.matrix ────────────────────────────────────────────


@needs_r
def test_the_width_is_r_model_matrix_columns_on_a_frame_with_categories_and_strata(tmp_path):
    """A frame of every kind the chain reads: numbers, text of five levels, a 0/1 column, a numeric
    code declared a category (four codes), the survey design's 12 strata and two PSUs as text and
    its weight, and the outcome's term; blanks in four of them. The width is R's columns of
    ``model.matrix(~ ., data)`` less the intercept, every text column and the declared code a
    factor over all rows."""
    rng = np.random.default_rng(1)
    n = 200
    frame = pd.DataFrame({
        "x": rng.normal(size=n), "grp": rng.choice(list("abcde"), n).astype(object),
        "flag": (rng.random(n) < 0.3).astype(float), "code": rng.choice([1, 2, 3, 7], n),
        "__stratum": [f"stratum {k}" for k in rng.integers(1, 13, n)],
        "__psu": [f"psu {k}" for k in rng.integers(1, 3, n)],
        "__weight": rng.uniform(1000, 9000, n), "__outcome": rng.normal(size=n)})
    for c, share in (("x", 0.2), ("grp", 0.1), ("flag", 0.15)):
        frame.loc[rng.random(n) < share, c] = np.nan
    from turbotab.core.methods.smcfcs import model_width

    variables = [Variable("x", "numeric"), Variable("grp", "categorical"),
                 Variable("flag", "binary")]
    ours = model_width(frame, variables, kinds={"code": "categorical"}).width
    r = run_r("d <- read.csv(frame_csv, stringsAsFactors = FALSE, na.strings = c('', 'NA'))\n"
              "for (c in names(d)) if (is.character(d[[c]])) d[[c]] <- factor(d[[c]])\n"
              "d$code <- factor(d$code)\n"
              "mf <- model.frame(~ ., d, na.action = na.pass)\n"
              "out(list(width = ncol(model.matrix(~ ., mf)) - 1))\n", {"frame": frame}, tmp_path)
    assert ours == r["width"] == 1 + 4 + 1 + 3 + 11 + 1 + 1 + 1


# ── up to 80 columns, whatever the rows; wider, each rule's boundary ─────────


def _least_squares(recorded: int, others: int = 85) -> pd.DataFrame:
    """`x` recorded on its first ``recorded`` of 200 rows beside ``others`` complete numbers and the
    outcome's term: ``others`` + 2 columns, and `x`'s model ``others`` + 1 predictor parameters, by
    construction."""
    rng = np.random.default_rng(2)
    frame = pd.DataFrame({"x": rng.normal(size=200), **numbers(200, others, rng),
                          "__outcome": rng.normal(size=200)})
    frame.loc[recorded:, "x"] = np.nan
    return frame


def test_at_80_columns_or_fewer_a_model_short_of_its_rows_is_drawn_as_on_the_base():
    """The rule refuses nothing the base drew: `x` recorded on 100 of 200 rows beside 78 complete
    numbers and the outcome's term is a model of 80 columns, `x`'s on 79 parameters (1.3 rows each,
    short of 2), and it is drawn, as the base drew every model of 80 columns or fewer. One more
    number makes 81 columns, and the same 100 rows (80 parameters, 1.2 each) are refused."""
    frame = _least_squares(100, others=78)
    assert frame.shape[1] == CAP and int(frame["x"].notna().sum()) == 100 < SPV * 79
    drawn = impute(frame, [Variable("x", "numeric")], m=2, iterations=1)
    assert drawn.m == 2 and drawn.imputed == {"x": 100}
    assert all(f["x"].notna().all() for f in drawn.frames)
    wider = _least_squares(100, others=79)
    assert wider.shape[1] == CAP + 1
    with pytest.raises(ImputationRefused) as refused:
        impute(wider, [Variable("x", "numeric")], m=2, iterations=1)
    assert str(refused.value) == (
        f"{opening(81)}The chained equations would impute `x` from a model of 80 predictor "
        f"parameters (categories as indicators) fit on the 100 rows that record it, "
        f"{per(100, 80)} per parameter, {SOURCE}.")


def test_least_squares_above_80_columns_is_drawn_at_two_rows_per_parameter_and_refused_below():
    """87 columns, `x`'s model 86 parameters: 172 recorded rows (2.0 each) are imputed; 171 are
    refused (1.99 each: one digit would round to the 2 asked), in words whose counts are the
    frame's own."""
    drawn = impute(_least_squares(172), [Variable("x", "numeric")], m=2, iterations=1)
    assert drawn.m == 2 and drawn.imputed == {"x": 28}
    frame = _least_squares(171)
    assert frame.shape[1] == 87 and int(frame["x"].notna().sum()) == 171 < SPV * 86
    assert f"{171 / 86:.1f}" == "2.0" and f"{171 / 86:.2f}" == "1.99"
    with pytest.raises(ImputationRefused) as refused:
        impute(frame, [Variable("x", "numeric")], m=2, iterations=1)
    assert str(refused.value) == (
        f"{opening(87)}The chained equations would impute `x` from a model of 86 predictor "
        f"parameters (categories as indicators) fit on the 171 rows that record it, 1.99 per "
        f"parameter, {SOURCE}.")
    assert getattr(refused.value, "columns", None) is None  # nothing the others already span


def _several_short(k: int) -> pd.DataFrame:
    """``k`` incomplete numbers `a1` … `ak` on 200 rows, `aj` recorded on its first 100 + j, beside
    80 complete numbers and the outcome's term: k + 81 columns, and each model k + 80 predictor
    parameters (the other k − 1, the 80 and the outcome's), so every one falls short of 2 per
    parameter (100 + j rows against 2k + 160), `a1` the furthest (its rows are the fewest)."""
    rng = np.random.default_rng(11)
    frame = pd.DataFrame({**{f"a{j}": rng.normal(size=200) for j in range(1, k + 1)},
                          **numbers(200, 80, rng), "__outcome": rng.normal(size=200)})
    for j in range(1, k + 1):
        frame.loc[100 + j:, f"a{j}"] = np.nan
    return frame


@pytest.mark.parametrize("k, tail", [
    (2, f"The model of `a2` falls short of its own rule too: {short_of(102, 82)}."),
    (3, f"The models of `a2` ({short_of(102, 83)}) and `a3` ({short_of(103, 83)}) fall short of "
        f"their own rules too."),
    (6, f"The models of `a2` ({short_of(102, 86)}), `a3` ({short_of(103, 86)}), `a4` "
        f"({short_of(104, 86)}) and 2 more fall short of their own rules too."),
], ids=["one", "two", "five"])
def test_each_model_short_after_the_first_is_named_with_its_own_count_and_rule(k, tail):
    """One model short after the first is named in the singular, with its own rows, parameters and
    rule; several in the plural, three so and the rest counted. The parameters and rows are the
    frame's own: `a1`'s 101 recorded rows against k + 80 parameters, below 2 each for every k."""
    frame = _several_short(k)
    p = k + 80
    assert [int(frame[f"a{j}"].notna().sum()) for j in range(1, k + 1)] == list(range(101, 101 + k))
    assert 100 + k < SPV * p  # the last column, the nearest the rule, is short too
    with pytest.raises(ImputationRefused) as refused:
        impute(frame, [Variable(f"a{j}", "numeric") for j in range(1, k + 1)], m=2, iterations=1)
    assert str(refused.value) == (
        f"{opening(k + 81)}The chained equations would impute `a1` from a model of {p} predictor "
        f"parameters (categories as indicators) fit on the 101 rows that record it, "
        f"{per(101, p)} per parameter, {SOURCE}. {tail}")


def test_each_model_after_the_first_is_counted_by_its_own_rule_not_the_first_s():
    """Three kinds short on one frame of 300 rows beside 80 complete numbers and the outcome's term
    (85 columns): a 0/1 `b` with 20 ones (its events) on 84 parameters, the furthest from its 10;
    text `g` of three levels recorded 30 times each, an effective size of 90 − 3·30³/90² = 80 on
    (3 − 1)·83 = 166 parameters, against 10; and a number `a` on 101 rows and 84 parameters, against
    about 2. After `b`'s words, `g` is said in its effective size and `a` in its rows, each against
    its own rule, neither in `b`'s events."""
    rng = np.random.default_rng(13)
    n = 300
    a = rng.normal(size=n)
    a[101:] = np.nan
    b = np.r_[np.ones(20), np.zeros(260), np.full(20, np.nan)]
    g = pd.Series(["p"] * 30 + ["q"] * 30 + ["r"] * 30 + [np.nan] * 210, dtype=object)
    frame = pd.DataFrame({"a": a, "b": b, "g": g, **numbers(n, 80, rng),
                          "__outcome": rng.normal(size=n)})
    assert effective([30, 30, 30]) == pytest.approx(80.0)
    # the furthest first: b at 20/84 of 10, g at 80/166 of 10, a at 101/84 of 2
    assert 20 / 84 / EPV < 80 / 166 / EPV < 101 / 84 / SPV < 1
    variables = [Variable("a", "numeric"), Variable("b", "binary"), Variable("g", "categorical")]
    with pytest.raises(ImputationRefused) as refused:
        impute(frame, variables, m=2, iterations=1)
    assert str(refused.value) == (
        f"{opening(85)}The chained equations would impute `b` from a model of 84 predictor "
        f"parameters (categories as indicators) fit on the 280 rows that record it, 20 of them its "
        f"rarer value, {per(20, 84)} events per parameter, where a common rule for logistic "
        f"coefficients asks for 10 (Peduzzi et al. 1996). The models of `g` (an effective sample "
        f"size of 80 on 166 parameters, {per(80, 166)} per parameter where its rule asks for 10) "
        f"and `a` ({short_of(101, 84)}) fall short of their own rules too.")


def test_a_count_of_one_is_said_in_the_singular_and_a_figure_below_0_05_in_two_digits():
    """0/1 columns `a` and `b` each record their rarer value on one row (of 200 and 250 recorded),
    and a number `x` is recorded on 3 rows, beside 87 complete numbers and the outcome's term: 91
    columns, each model 90 predictor parameters. `a` and `b` are equally far short (1/90 events per
    parameter against 10), so the chain's order names `a` first, in full; then `b` and `x` in their
    own counts, `b`'s one row in the singular. 1/90 = 0.0111… and 3/90 = 0.0333…: one decimal
    would read both as 0.0, so each is said to two significant digits. A variable imputed once per
    unit, recorded on the two rows of one unit, is fit on 1 unit: said so, and 1/86 = 0.0116… per
    parameter as 0.012."""
    rng = np.random.default_rng(17)
    n = 300
    a = np.r_[1.0, np.zeros(199), np.full(100, np.nan)]
    b = np.r_[np.zeros(100), 1.0, np.zeros(149), np.full(50, np.nan)]
    x = np.r_[rng.normal(size=3), np.full(n - 3, np.nan)]
    frame = pd.DataFrame({"x": x, **numbers(n, 87, rng), "a": a, "b": b,
                          "__outcome": rng.normal(size=n)})
    assert [int(frame[c].notna().sum()) for c in "abx"] == [200, 250, 3]
    assert [int(frame[c].value_counts().min()) for c in "ab"] == [1, 1]
    variables = [Variable("a", "binary"), Variable("b", "binary"), Variable("x", "numeric")]
    assert smcfcs.model_width(frame, variables).width == 87 + 3 + 1 == 91
    with pytest.raises(ImputationRefused) as refused:
        impute(frame, variables, m=2, iterations=1)
    assert str(refused.value) == (
        f"{opening(91)}The chained equations would impute `a` from a model of 90 predictor "
        f"parameters (categories as indicators) fit on the 200 rows that record it, 1 of them its "
        f"rarer value, 0.011 events per parameter, where a common rule for logistic coefficients "
        f"asks for 10 (Peduzzi et al. 1996). The models of `b` (1 row of its rarer value on 90 "
        f"parameters, 0.011 events per parameter where its rule asks for 10) and `x` (3 rows on 90 "
        f"parameters, 0.033 per parameter where its rule asks for about 2) fall short of their own "
        f"rules too.")
    # once per unit: `u` recorded on unit 0's two rows; 85 numbers and the outcome's term, row-level
    units = np.repeat(np.arange(n // 2), 2)
    u = np.r_[0.4, 0.4, np.full(n - 2, np.nan)]
    clustered = pd.DataFrame({"u": u, **numbers(n, 85, rng), "__outcome": rng.normal(size=n)})
    assert int(clustered["u"].notna().sum()) == 2 and len(set(units[:2])) == 1
    with pytest.raises(ImputationRefused) as refused:
        impute(clustered, [Variable("u", "numeric", unit_level=True)], units=units, m=2,
               iterations=1)
    assert str(refused.value) == (
        f"{opening(87)}The chained equations would impute `u` from a model of 86 predictor "
        f"parameters (categories as indicators) fit on the 1 unit that records it, 0.012 per "
        f"parameter, {SOURCE}.")


def _yes_no(ones: int, zeros: int) -> pd.DataFrame:
    """A 0/1 column `b` recorded as ``ones`` ones and ``zeros`` zeros, blank on five more rows,
    beside 80 complete numbers and the outcome's term: 82 columns, `b`'s model 81 predictor
    parameters."""
    rng = np.random.default_rng(3)
    n = ones + zeros + 5
    b = np.r_[np.ones(ones), np.zeros(zeros), np.full(5, np.nan)]
    return pd.DataFrame({"b": b, **numbers(n, 80, rng), "__outcome": rng.normal(size=n)})


def test_a_yes_no_model_is_held_to_ten_events_per_parameter_on_its_rarer_value():
    """81 parameters need 810 events, the rows of the rarer value (Peduzzi's count, which the
    engine's logistic rule counts): 810 ones beside 1,620 zeros are imputed, and 810 zeros beside
    1,620 ones; 809 ones beside 1,620 zeros are refused, and so are 809 zeros beside 1,620 ones, in
    words of events (9.99 each: one digit would round to the 10 asked). The effective sample size
    3ab/n would have drawn both (about 1,619 of the 810 asked)."""
    assert 810 == EPV * 81 and 809 < EPV * 81 <= effective([809, 1620])
    assert f"{809 / 81:.1f}" == "10.0" and f"{809 / 81:.2f}" == "9.99"
    impute(_yes_no(810, 1620), [Variable("b", "binary")], m=2, iterations=1)
    impute(_yes_no(1620, 810), [Variable("b", "binary")], m=2, iterations=1)
    for ones, zeros in ((809, 1620), (1620, 809)):
        with pytest.raises(ImputationRefused) as refused:
            impute(_yes_no(ones, zeros), [Variable("b", "binary")], m=2, iterations=1)
        assert str(refused.value) == (
            f"{opening(82)}The chained equations would impute `b` from a model of 81 predictor "
            f"parameters (categories as indicators) fit on the 2,429 rows that record it, 809 of "
            f"them its rarer value, 9.99 events per parameter, where a common rule for logistic "
            f"coefficients asks for 10 (Peduzzi et al. 1996).")


def _categories(counts: list[int], levels: str = "pqr") -> pd.DataFrame:
    """Text `g` recorded ``counts`` times at each of ``levels``, blank on four more rows, beside 80
    complete numbers and the outcome's term: `g`'s model 81 predictor parameters."""
    rng = np.random.default_rng(4)
    values = [lv for lv, k in zip(levels, counts) for _ in range(k)] + [np.nan] * 4
    n = len(values)
    return pd.DataFrame({"g": pd.Series(values, dtype=object), **numbers(n, 80, rng),
                         "__outcome": rng.normal(size=n)})


def test_a_category_of_two_levels_is_a_logistic_model_held_to_its_events():
    """Text of two levels is drawn by the same logistic model as a 0/1 column, so it is held to
    the same events: 810 `p` beside 1,620 `q` on 81 parameters are imputed, 809 are refused."""
    impute(_categories([810, 1620], "pq"), [Variable("g", "categorical")], m=2, iterations=1)
    with pytest.raises(ImputationRefused) as refused:
        impute(_categories([809, 1620], "pq"), [Variable("g", "categorical")], m=2, iterations=1)
    assert str(refused.value).startswith(opening(82))
    assert "fit on the 2,429 rows that record it, 809 of them its rarer value, " \
           "9.99 events per parameter" in str(refused.value)


@needs_r
def test_a_multinomial_model_s_parameters_are_nnet_multinom_s_slopes(tmp_path):
    """The multinomial model's parameters, (k − 1)·p, are the slopes R's ``nnet::multinom`` fits:
    its coefficient matrix on three levels and three predictors is (k − 1) × (p + 1), a row for
    each level beyond the first and an intercept column, so 2 × 3 = 6 slopes, as
    :attr:`ModelSize.parameters` counts them for that model."""
    from turbotab.core.methods.imputation import model_size

    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"g": np.repeat(list("pqr"), 30), **{c: rng.normal(size=90)
                                                              for c in ("a", "b", "c")}})
    r = run_r("d <- read.csv(frame_csv, stringsAsFactors = TRUE)\n"
              "m <- nnet::multinom(g ~ ., d, trace = FALSE)\n"
              "out(list(rows = nrow(coef(m)), cols = ncol(coef(m))))\n", {"frame": frame},
              tmp_path)
    assert (r["rows"], r["cols"]) == (3 - 1, 3 + 1)
    sized = model_size("g", "categorical", 3, frame["g"])
    assert sized.levels == 3 and sized.parameters == r["rows"] * (r["cols"] - 1) == 6


def test_a_multinomial_model_counts_a_slope_per_predictor_for_each_level_beyond_the_first():
    """Three levels on 81 predictor parameters (83 columns): a multinomial model fits a slope per
    predictor column for each level beyond the first (``nnet::multinom``'s (k − 1) × (p + 1)
    coefficients, above), so (3 − 1)·81 = 162 parameters and the rule asks for an effective size
    of 1,620. 607, 608 and 608 recorded give 1,620.4 and are imputed; 607, 607 and 608 give
    1,619.6 and are refused, in words that count the 162 parameters (9.997 each: fewer digits
    would round to the 10 asked). Counted as 81, both would have been drawn (810 asked)."""
    assert effective([607, 608, 608]) >= EPV * 162 > effective([607, 607, 608]) >= EPV * 81
    impute(_categories([607, 608, 608]), [Variable("g", "categorical")], m=2, iterations=1)
    with pytest.raises(ImputationRefused) as refused:
        impute(_categories([607, 607, 608]), [Variable("g", "categorical")], m=2, iterations=1)
    size = effective([607, 607, 608])
    assert f"{size:,.0f}" == "1,620" and f"{size / 162:.2f}" == "10.00"  # the fewest digits below
    assert str(refused.value) == (
        f"{opening(83)}The chained equations would impute `g` from a multinomial model of 162 "
        f"parameters (81 predictor parameters, categories as indicators, for each of its 2 levels "
        f"beyond the first) fit on the 1,822 rows that record it, an effective sample size of "
        f"{size:,.1f} (n − Σn³/n² over its 3 levels), {size / 162:.3f} per parameter, where a "
        f"common rule of thumb asks for 10.")


# ── a column whose recorded values hold one level ────────────────────────────


def _one_level(text: bool, second: bool) -> tuple[pd.DataFrame, Variable]:
    """`b` recorded on 100 of 120 rows, every recorded value one level (1, or the text "yes"), or
    with ``second`` the other level (0, or "no") on one row, blank on 20, beside 80 complete numbers
    and the outcome's term: `b`'s model 81 predictor parameters, on 100 rows."""
    rng = np.random.default_rng(14)
    one, other = ("yes", "no") if text else (1.0, 0.0)
    values = [one] * 99 + [other if second else one] + [np.nan] * 20
    b = pd.Series(values, dtype=object if text else float)
    frame = pd.DataFrame({"b": b, **numbers(120, 80, rng), "__outcome": rng.normal(size=120)})
    return frame, Variable("b", "categorical" if text else "binary")


@pytest.mark.parametrize("text", [False, True], ids=["zero-one", "text"])
def test_a_column_recorded_at_one_level_is_not_held_to_the_rule_and_one_of_two_levels_is(text):
    """The rule counts a model's events, or its effective size, over the levels its recorded values
    hold; one level has neither (``ModelSize.held``). A wide model (above 80 columns) whose `b` is
    recorded at one level on 100 rows is drawn, short of nothing, its blanks as its one level. The
    same `b` with the other level on one row has 1 event on 81 parameters, and is refused."""
    frame, variable = _one_level(text, second=False)
    width = 81 if text else 82  # text of one level adds no indicator; a 0/1 column adds one
    assert smcfcs.model_width(frame, [variable]).width == width > CAP
    drawn = impute(frame, [variable], m=3, iterations=1)
    assert drawn.imputed == {"b": 20}
    # no model: the one level recorded
    assert all(set(f["b"].iloc[100:]) == {"yes" if text else 1.0} for f in drawn.frames)
    frame, variable = _one_level(text, second=True)
    with pytest.raises(ImputationRefused) as refused:
        impute(frame, [variable], m=3, iterations=1)
    assert str(refused.value).startswith(
        f"{opening(82)}The chained equations would impute `b` from a model of 81 predictor "
        f"parameters (categories as indicators) fit on the 100 rows that record it, 1 of them its "
        f"rarer value, ")


# ── a declared code is a category in the chain's own models ─────────────────


def _coded(n: int = 600, seed: int = 11) -> pd.DataFrame:
    """`x` (blank on about one row in five) is 10 where the complete code `code` is 2 and 0 on its
    other codes (1, 3 and 7), plus noise of SD 0.3; beside text `grp` (five levels, blanks), a 0/1
    `flag` (blanks), a number `z` and the outcome's term."""
    rng = np.random.default_rng(seed)
    code = rng.choice([1, 2, 3, 7], n).astype(float)
    x = 10.0 * (code == 2) + rng.normal(scale=0.3, size=n)
    frame = pd.DataFrame({"x": x, "grp": rng.choice(list("abcde"), n).astype(object),
                          "flag": (rng.random(n) < 0.3).astype(float), "code": code,
                          "z": rng.normal(size=n), "__outcome": rng.normal(size=n)})
    for c, share in (("x", 0.2), ("grp", 0.1), ("flag", 0.15)):
        frame.loc[rng.random(n) < share, c] = np.nan
    return frame


CODED = [Variable("x", "numeric"), Variable("grp", "categorical"), Variable("flag", "binary")]


def _designs(frame: pd.DataFrame, variables: list[Variable], **kw: Any) -> dict[str, int]:
    """Each drawn column's design columns, the intercept not counted, as the chain builds them."""
    seen: dict[str, int] = {}
    rows, units = smcfcs._Chain.design_rows, smcfcs._Chain.design_units

    def spy(real: Any) -> Any:
        def built(self: Any, target: str, skip: Any) -> np.ndarray:
            X = real(self, target, skip)
            seen[target] = X.shape[1] - 1
            return X
        return built

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(smcfcs._Chain, "design_rows", spy(rows))
        mp.setattr(smcfcs._Chain, "design_units", spy(units))
        impute(frame, variables, m=1, iterations=1, **kw)
    return seen


def test_a_declared_numeric_code_enters_the_chain_as_the_indicators_the_width_counts():
    """`code` declared a category: each model the chain fits holds its three indicators, as the
    width counts them (`x`: grp's 4 + flag + code's 3 + z + the outcome's term = 10; `grp`: 7;
    `flag`: 10), and so does each model on clustered rows (1,200 rows in units of four, a
    unit-level `u`: `x` on the 11 other columns, the 10 row-level ones' unit means and its own =
    22; `u` on the 11 row-level columns' unit means). Undeclared, `code` is one number in both."""
    frame = _coded()
    kinds = {"code": "categorical"}
    sized = smcfcs.model_width(frame, CODED, kinds=kinds)
    widths = {m.column: m.predictors for m in sized.models}
    assert widths == {"x": 10, "grp": 7, "flag": 10}
    assert _designs(frame, CODED, kinds=kinds) == widths
    plain = {m.column: m.predictors for m in smcfcs.model_width(frame, CODED).models}
    assert plain == {"x": 8, "grp": 5, "flag": 8} == _designs(frame, CODED)
    rng = np.random.default_rng(12)
    rows = _coded(1200)
    units = np.repeat(np.arange(len(rows) // 4), 4)
    u = np.repeat(rng.normal(size=len(rows) // 4), 4)
    u[np.repeat(rng.random(len(rows) // 4) < 0.2, 4)] = np.nan
    clustered = rows.drop(columns=["__outcome"]).assign(u=u, __outcome=rows["__outcome"])
    variables = [*CODED, Variable("u", "numeric", unit_level=True)]
    sized = smcfcs.model_width(clustered, variables, units=units, kinds=kinds)
    widths = {m.column: m.predictors for m in sized.models}
    assert widths["x"] == 11 + 10 + 1 and widths["u"] == 11
    assert _designs(clustered, variables, units=units, kinds=kinds) == widths


def test_a_declared_numeric_code_is_imputed_from_its_levels_not_as_a_line():
    """`x` is 10 on code 2 and 0 on codes 1, 3 and 7 (`_coded`): drawn from `code`'s indicators,
    the blanks on code 2 average about 10 and the others about 0 (within 0.5, the noise's SD
    0.3 and the draws' own spread allowing); a line in the code's value cannot put them there."""
    frame = _coded()
    blank = frame["x"].isna().to_numpy()
    on_two = blank & (frame["code"].to_numpy() == 2)
    off_two = blank & ~(frame["code"].to_numpy() == 2)
    assert on_two.sum() > 20 and off_two.sum() > 60
    drawn = impute(frame, CODED, kinds={"code": "categorical"}, m=5, iterations=5)
    for copy in drawn.frames:
        x = copy["x"].to_numpy(dtype=float)
        assert abs(x[on_two].mean() - 10.0) < 0.5
        assert abs(x[off_two].mean()) < 0.5


# ── the brief's scale: about 90 columns ──────────────────────────────────────


def wide(n: int, seed: int = 5) -> pd.DataFrame:
    """About 90 model columns: the outcome `y`, `x` (what is studied, about 10% blank), 84
    complete numbers, `g` (text, four levels) and `smoke` (0/1, about 5% blank), and an identifier."""
    rng = np.random.default_rng(seed)
    cols = numbers(n, 84, rng)
    x = rng.normal(size=n)
    y = 0.5 * x + 0.05 * sum(cols[f"c{j:02d}"] for j in range(1, 11)) + rng.normal(size=n)
    smoke = (rng.random(n) < 0.4).astype(float)
    x[rng.random(n) < 0.1] = np.nan
    smoke[rng.random(n) < 0.05] = np.nan
    return pd.DataFrame({"id": np.arange(n), "y": y, "x": x, **cols,
                         "g": rng.choice(list("abcd"), n).astype(object), "smoke": smoke})


WIDE_ROLES = {"id": "identifier", "x": "exposure", **{f"c{j:02d}": "covariate" for j in range(1, 85)},
              "g": "covariate", "smoke": "covariate"}
WIDTH = 1 + 84 + 3 + 1 + 1  # x, the numbers, g's indicators, smoke and the outcome's term


def _wide_frame(table: pd.DataFrame) -> tuple[pd.DataFrame, list[Variable]]:
    frame = table.drop(columns=["id", "y"]).assign(__outcome=table["y"].to_numpy())
    return frame, [Variable("x", "numeric"), Variable("smoke", "binary")]


def _refusal_words(table: pd.DataFrame) -> str:
    """The refusal's words for :func:`wide`'s table, from pandas' counts: the width, then `smoke`'s
    model, the furthest short (its rarer value's rows, the events, against 10 per parameter), then
    `x`'s, in its own rows against its own rule."""
    smoke = table["smoke"].dropna()
    events = int(smoke.value_counts().min())
    x = int(table["x"].notna().sum())
    p = WIDTH - 1
    return (f"{opening(WIDTH)}The chained equations would impute `smoke` from a model of {p} "
            f"predictor parameters (categories as indicators) fit on the {len(smoke):,} rows that "
            f"record it, {events:,} of them its rarer value, {per(events, p)} events per parameter, "
            f"where a common rule for logistic coefficients asks for 10 (Peduzzi et al. 1996). The "
            f"model of `x` falls short of its own rule too: {short_of(x, p)}.")


def test_ninety_columns_are_imputed_on_8000_rows_and_refused_on_150_by_the_sampler():
    """The same 90 columns: on 8,000 rows every model has its rows (`x` about 80 per parameter);
    on 150 neither `x` (about 1.5 per parameter) nor `smoke` has them, and the refusal says so."""
    big, variables = _wide_frame(wide(8000))
    drawn = impute(big, variables, m=2, iterations=1)
    assert drawn.imputed == {"x": int(big["x"].isna().sum()), "smoke": int(big["smoke"].isna().sum())}
    from turbotab.core.methods.smcfcs import model_width

    assert model_width(big, variables).width == WIDTH
    small_table = wide(150)
    small, variables = _wide_frame(small_table)
    assert int(small["x"].notna().sum()) < SPV * (WIDTH - 1)
    with pytest.raises(ImputationRefused) as refused:
        impute(small, variables, m=2, iterations=1)
    assert str(refused.value) == _refusal_words(small_table)


# ── the ways forward the words name ──────────────────────────────────────────


def _surveyed(strata: int, rows_per: int = 2) -> pd.DataFrame:
    """`x` (blank on one row in ten) beside eight numbers, the outcome's term and a survey design
    of ``strata`` strata of ``rows_per`` rows, two PSUs in each, and a weight."""
    rng = np.random.default_rng(6)
    n = strata * rows_per
    x = rng.normal(size=n)
    x[::10] = np.nan
    return pd.DataFrame({
        "x": x, **numbers(n, 8, rng), "__outcome": rng.normal(size=n),
        "__stratum": [f"stratum {k // rows_per + 1}" for k in range(n)],
        "__psu": [f"psu {k % 2 + 1}" for k in range(n)], "__weight": rng.uniform(1, 9, n)})


def _more(frame: pd.DataFrame, k: int) -> pd.DataFrame:
    return frame.assign(**{f"d{j}": np.random.default_rng(j).normal(size=len(frame))
                           for j in range(k)})


def test_where_the_survey_design_is_most_of_the_width_the_words_say_so_and_offer_the_sample():
    """70 strata of two rows: the design's 69 + 1 + 1 terms are 71 of the model's 81 columns, and
    without them the model holds 10 (drawn whatever its rows; `x`'s, 9 parameters on its 126 rows,
    has them too), so the sample answer is offered. With 5 strata of 24 rows and 70 more numbers
    the design is 6 of 86 and goes unsaid. With 90 strata and 75 more numbers the design's 91
    terms are most of 176 columns, but without them the model holds 85 and `x`'s is still short
    (84 parameters on 162 rows, 1.9 each): the words count the design's terms, and offer no
    sample answer."""
    from turbotab.core.methods.missing import WITHOUT_THE_DESIGN, sample_way

    frame = _surveyed(70)
    assert int(frame["x"].notna().sum()) == 126
    with pytest.raises(ImputationRefused) as refused:
        impute(frame, [Variable("x", "numeric")], m=2, iterations=1)
    said = str(refused.value)
    assert said.startswith(f"{opening(81)}The chained equations would impute `x` from a model of "
                           f"80 predictor parameters (categories as indicators) fit on the 126 "
                           f"rows that record it, {per(126, 80)} per parameter")
    assert ("Of the imputation model's 81 columns, 71 are the survey design's: 69 indicators of "
            "its strata, one indicator of its PSUs within their strata and its weight. "
            f"{WITHOUT_THE_DESIGN}") in said
    assert refused.value.exits == [sample_way()]
    # silent where the design is not most of the width
    few = _surveyed(5, rows_per=24)
    impute(few, [Variable("x", "numeric")], m=2, iterations=1)
    with pytest.raises(ImputationRefused) as refused:
        impute(_more(few, 70), [Variable("x", "numeric")], m=2, iterations=1)
    assert str(refused.value).startswith(opening(86))
    assert "survey design" not in str(refused.value)
    assert getattr(refused.value, "exits", None) is None
    # most of the width, but without the design the model is still refused
    crowded = _more(_surveyed(90), 75)
    assert int(crowded["x"].notna().sum()) == 162 < SPV * 84
    with pytest.raises(ImputationRefused) as refused:
        impute(crowded, [Variable("x", "numeric")], m=2, iterations=1)
    assert "Of the imputation model's 176 columns, 91 are the survey design's" in str(refused.value)
    assert WITHOUT_THE_DESIGN not in str(refused.value)
    assert getattr(refused.value, "exits", None) is None


def _recoded(recode: bool, others: int = 80, recorded: int = 172) -> pd.DataFrame:
    """`x` recorded on ``recorded`` of 200 rows beside ``others`` numbers, a six-level code
    `race6`, `race3` (its recode into three groups, or three groups drawn apart from it) and the
    outcome's term: ``others`` + 9 columns, and `x`'s model ``others`` + 5 + 2 + 1 parameters, two
    fewer without `race3`."""
    rng = np.random.default_rng(7)
    n = 200
    race6 = np.array([f"r{k % 6 + 1}" for k in range(n)], dtype=object)
    groups = {"r1": "A", "r2": "A", "r3": "B", "r4": "B", "r5": "C", "r6": "C"}
    race3 = (np.array([groups[v] for v in race6], dtype=object) if recode
             else rng.choice(list("ABC"), n).astype(object))
    x = rng.normal(size=n)
    x[recorded:] = np.nan
    return pd.DataFrame({"x": x, **numbers(n, others, rng), "race6": race6, "race3": race3,
                         "__outcome": rng.normal(size=n)})


SPANNED = ("`race3` adds nothing the other columns do not hold on the {n} rows that record them all "
           "(as a recode adds nothing beside the column it recodes), and without it ")


def test_a_recode_beside_its_code_is_offered_to_leave_out_and_nothing_else_is():
    """`race3` is a function of `race6` by construction, so its indicators lie in the span of
    `race6`'s and the intercept: the refusal names it, of the columns it may leave out. On 89
    columns `x`'s model has 88 parameters on 172 rows (1.95 each), and without `race3` 86 (2.0
    each): the words say each model then has its rows. On 81 columns (72 numbers, 100 rows),
    without `race3` the model holds 79, drawn whatever its rows, and the words say that instead.
    Drawn apart from `race6`, no column is named."""
    variables = [Variable("x", "numeric")]
    leavable = ["race6", "race3"]
    with pytest.raises(ImputationRefused) as refused:
        impute(_recoded(True), variables, m=2, iterations=1, leavable=leavable)
    assert refused.value.columns == ["race3"]
    assert str(refused.value).endswith(
        SPANNED.format(n=172) + "each incomplete column's model has the rows per parameter its "
        "rule asks for.")
    impute(_recoded(True).drop(columns=["race3"]), variables, m=2, iterations=1)  # drawn
    narrow = _recoded(True, others=72, recorded=100)
    assert smcfcs.model_width(narrow, variables).width == 1 + 72 + 5 + 2 + 1 == CAP + 1
    with pytest.raises(ImputationRefused) as refused:
        impute(narrow, variables, m=2, iterations=1, leavable=leavable)
    assert refused.value.columns == ["race3"]
    assert str(refused.value).endswith(
        SPANNED.format(n=100) + f"the imputation model would hold 79 columns, no more than the "
        f"{CAP} drawn whatever the rows.")
    impute(narrow.drop(columns=["race3"]), variables, m=2, iterations=1)  # drawn
    with pytest.raises(ImputationRefused) as refused:
        impute(_recoded(False), variables, m=2, iterations=1, leavable=leavable)
    assert getattr(refused.value, "columns", None) is None
    from turbotab.core.methods.smcfcs import redundant_columns

    kinds = smcfcs.chain_kinds(_recoded(False), variables)
    assert redundant_columns(_recoded(False), kinds, ["race3", "race6"]) == ([], 172)


def test_the_sampler_names_a_column_to_leave_out_only_from_the_set_it_is_given():
    """`race3` recodes `race6`, and so does `race2` (two groups); both are spanned. On 176 recorded
    rows `x`'s model is short with both (89 parameters) and has its rows without either (88, or
    87). Given `race3` alone as a column it may leave out, the refusal names `race3` and never
    `race2`; given no set, it names no column and says nothing of leaving one out (the fit's table
    passes the answer's set, so it names `race3` as the answer does: below)."""
    frame = _recoded(True, recorded=176)
    frame.insert(frame.columns.get_loc("race3"), "race2",
                 np.where(np.isin(frame["race6"], ["r1", "r2", "r3"]), "u", "v").astype(object))
    variables = [Variable("x", "numeric")]
    with pytest.raises(ImputationRefused) as refused:
        impute(frame, variables, m=2, iterations=1, leavable=["race3"])
    assert refused.value.columns == ["race3"] and "`race2`" not in str(refused.value)
    with pytest.raises(ImputationRefused) as refused:
        impute(frame, variables, m=2, iterations=1)
    assert getattr(refused.value, "columns", None) is None
    assert "adds nothing" not in str(refused.value) and "`race3`" not in str(refused.value)


# ── at the answer, before any lock ───────────────────────────────────────────


def wide_state(**slots: Any) -> ProjectState:
    base: dict[str, Any] = dict(target="y", task="regression", purpose="inference",
                                grain={"grain": "one_row_per_unit"}, roles=dict(WIDE_ROLES),
                                exclusions=[], missing=MissingSpec(strategy="complete_case"))
    base.update(slots)
    return ProjectState(**base)


@pytest.fixture(scope="module")
def tables(tmp_path_factory) -> dict[int, tuple[pd.DataFrame, Ingested]]:
    out = {}
    for n in (150, 8000):
        folder = tmp_path_factory.mktemp(f"wide_{n}")
        table = wide(n)
        path = folder / "wide.csv"
        table.to_csv(path, index=False)
        out[n] = (table, Ingested(path, folder))
    return out


def ctx_of(state: ProjectState, ingested: Ingested) -> dict[str, Any]:
    """The validators' context as the server builds it, over the table the analysis reads."""
    return {"state": state, "store": ingested.store, "task": state.task}


def test_multiple_imputation_on_150_rows_is_refused_at_the_answer_with_the_fit_s_words(tables):
    """The answer itself is refused: the fit's words (the counts are pandas'), and complete cases
    as the way forward. The plan has no lock, nor can it get one: the lock Fit records is refused
    the same way. On 8,000 rows both the answer and the lock are accepted."""
    table, ingested = tables[150]
    now = wide_state()
    mi = SetMissing(strategy="multiple_imputation")
    with pytest.raises(Refusal) as refused:
        validate(mi, ctx_of(now, ingested))
    assert refused.value.code == "imputation_too_wide"
    assert refused.value.message == LEAD + _refusal_words(table)
    assert [e["label"] for e in refused.value.exits] == ["Complete cases, with their assumption stated"]
    assert refused.value.exits[0]["decision"]["strategy"] == "complete_case"
    locked = wide_state(missing=MissingSpec(strategy="multiple_imputation"))
    assert not locked.plan_locked
    with pytest.raises(Refusal) as refused:
        validate({"kind": "lock_plan"}, ctx_of(locked, ingested))
    assert refused.value.code == "imputation_too_wide"
    assert refused.value.message == LEAD + _refusal_words(table)
    _, big = tables[8000]
    validate(mi, ctx_of(wide_state(), big))
    validate({"kind": "lock_plan"}, ctx_of(wide_state(missing=MissingSpec(
        strategy="multiple_imputation")), big))


def test_a_roles_answer_that_widens_the_model_later_is_refused_and_one_within_the_rule_is_not(
        tables):
    """Multiple imputation recorded on 150 rows while 79 numbers and `smoke` are left out by their
    roles (`x`'s model: 9 parameters, about 15 rows each). The roles answer that brings them all
    back (90 columns) is refused at the answer; one that brings five back (15 columns) is
    accepted."""
    from turbotab.core.decisions import SetRoles

    _, ingested = tables[150]
    out = {**{f"c{j:02d}": "excluded" for j in range(6, 85)}, "smoke": "excluded"}
    narrow = wide_state(roles={**WIDE_ROLES, **out},
                        missing=MissingSpec(strategy="multiple_imputation"))
    with pytest.raises(Refusal) as refused:
        validate(SetRoles(roles=dict(WIDE_ROLES)), ctx_of(narrow, ingested))
    assert refused.value.code == "imputation_too_wide"
    back = {f"c{j:02d}" for j in range(6, 11)}
    validate(SetRoles(roles={**WIDE_ROLES, **{c: r for c, r in out.items() if c not in back}}),
             ctx_of(narrow, ingested))
    at = smcfcs.imputation_at(narrow, ctx_of(narrow, ingested))
    assert at.sized.width == 1 + 5 + 3 + 1 and at.sized.short() == []


def _recoded_project(folder: Any) -> tuple[ProjectState, Ingested]:
    """:func:`_recoded`'s table as a project: `x` the exposure, the numbers, `race6` and its recode
    `race3` covariates, under multiple imputation."""
    table = _recoded(True).drop(columns=["__outcome"]).assign(
        y=np.random.default_rng(9).normal(size=200))
    path = folder / "recoded.csv"
    table.to_csv(path, index=False)
    roles = {"x": "exposure", **{f"c{j:02d}": "covariate" for j in range(1, 81)},
             "race6": "covariate", "race3": "covariate"}
    state = ProjectState(target="y", task="regression", purpose="inference",
                         grain={"grain": "one_row_per_unit"}, roles=roles, exclusions=[],
                         missing=MissingSpec(strategy="multiple_imputation"))
    return state, Ingested(path, folder)


def test_at_the_answer_a_recode_beside_its_code_is_offered_to_leave_out(tmp_path):
    """:func:`_recoded`'s project, `race3` a covariate recoding `race6`: the lock Fit records is
    refused with complete cases and leaving `race3` out, which the missing-values answer records
    as a column left out; taken, the lock is accepted."""
    state, ingested = _recoded_project(tmp_path)
    with pytest.raises(Refusal) as refused:
        validate({"kind": "lock_plan"}, ctx_of(state, ingested))
    assert refused.value.message.endswith(
        SPANNED.format(n=172) + "each incomplete column's model has the rows per parameter its "
        "rule asks for.")
    exits = refused.value.exits
    assert [e["label"] for e in exits] == ["Complete cases, with their assumption stated",
                                           "Leave `race3` out"]
    assert exits[1]["decision"]["kind"] == "set_missing"
    assert exits[1]["decision"]["strategy"] == "multiple_imputation"
    assert exits[1]["decision"]["drop_columns"] == ["race3"]
    taken = state.model_copy(update={"missing": MissingSpec(strategy="multiple_imputation",
                                                            drop_columns=["race3"])})
    validate({"kind": "lock_plan"}, ctx_of(taken, ingested))


def test_given_the_answer_s_set_the_sampler_says_what_the_answer_says(tmp_path):
    """Where a redundant covariate exists (`race3`, :func:`_recoded_project`), the sampler run on
    the frame the answer built, with the set of columns the answer may leave out, is refused in
    the answer's very words; and so it is called as the fit's table now calls it
    (``missing.impute_for_inference`` on the design's inputs, given the answer's set), naming
    `race3` as the column to leave out. Given no set, the sampler names none
    (:func:`test_the_sampler_names_a_column_to_leave_out_only_from_the_set_it_is_given`)."""
    from turbotab.core.methods.missing import impute_for_inference

    state, ingested = _recoded_project(tmp_path)
    ctx = ctx_of(state, ingested)
    with pytest.raises(Refusal) as answered:
        validate({"kind": "lock_plan"}, ctx)
    found = smcfcs.imputation_at(state, ctx)
    assert "race3" in found.leavable and "x" not in found.leavable
    plan = found.plan
    with pytest.raises(ImputationRefused) as drawn:
        impute(plan.data, plan.variables, identity=plan.identity, units=plan.units,
               kinds=plan.kinds, leavable=found.leavable, m=2, iterations=1)
    assert LEAD + str(drawn.value) == answered.value.message
    assert drawn.value.columns == ["race3"]
    table = ingested.frame()
    with pytest.raises(ImputationRefused) as fit_s:
        impute_for_inference(found.spec, table[list(found.spec.inputs)], table["y"].to_numpy(),
                             "regression", leavable=found.leavable)
    sentence = (SPANNED.format(n=172) + "each incomplete column's model has the rows per "
                "parameter its rule asks for.")
    assert f"{LEAD}{fit_s.value}" == answered.value.message
    assert answered.value.message.endswith(sentence)
    assert fit_s.value.columns == ["race3"]


@needs_r
def test_the_frame_at_the_answer_is_the_survey_design_s_and_its_width_is_r_s(tmp_path):
    """A project with a survey design (70 strata of two rows, two PSUs each, a weight) under the
    population answer: the frame the check at the answer builds holds the design's terms, and
    its width is R's model.matrix columns on it (strata a factor); the refusal says the design is
    most of it and offers the sample answer."""
    from turbotab.core.decisions import SurveySpec
    from turbotab.core.models.survey import SAMPLE_EXIT

    rng = np.random.default_rng(8)
    n = 140
    x = rng.normal(size=n)
    x[::10] = np.nan
    table = pd.DataFrame({"y": rng.normal(size=n), "x": x, **numbers(n, 8, rng),
                          "strat": np.repeat(np.arange(101, 171), 2), "psu": np.tile([1, 2], 70),
                          "wt": rng.uniform(1000, 9000, n)})
    path = tmp_path / "surveyed.csv"
    table.to_csv(path, index=False)
    ingested = Ingested(path, tmp_path)
    roles = {"x": "exposure", **{f"c{j:02d}": "covariate" for j in range(1, 9)},
             "strat": "design", "psu": "design", "wt": "design"}
    state = ProjectState(target="y", task="regression", purpose="inference",
                         grain={"grain": "one_row_per_unit"}, roles=roles, exclusions=[],
                         missing=MissingSpec(strategy="multiple_imputation"),
                         survey=SurveySpec(estimand="population", weight="wt", strata="strat",
                                           psu="psu"))
    with pytest.raises(Refusal) as refused:
        validate({"kind": "lock_plan"}, ctx_of(state, ingested))
    assert "Of the imputation model's 81 columns, 71 are the survey design's" in refused.value.message
    assert [e["label"] for e in refused.value.exits] == [
        "Complete cases, with their assumption stated", SAMPLE_EXIT]
    found = smcfcs.imputation_at(state, ctx_of(state, ingested))
    data = found.plan.data
    assert {"__stratum", "__psu", "__weight"} <= set(data.columns)
    r = run_r("d <- read.csv(frame_csv, stringsAsFactors = FALSE, na.strings = c('', 'NA'))\n"
              "for (c in names(d)) if (is.character(d[[c]])) d[[c]] <- factor(d[[c]])\n"
              "mf <- model.frame(~ ., d, na.action = na.pass)\n"
              "out(list(width = ncol(model.matrix(~ ., mf)) - 1))\n", {"frame": data}, tmp_path)
    assert found.sized.width == r["width"] == 1 + 8 + 1 + 69 + 1 + 1


# ── the fit says the same ────────────────────────────────────────────────────


def _graph_fit(path: Any, folder: Any, state: ProjectState) -> tuple[Any, list[Any]]:
    """The fit's artifact for ``state`` run through the stage graph (as a plan recorded before
    this check, or by a log it could not read, reaches it), and each width the sampler measured."""
    from turbotab.core.tests.graph_runner import GraphRun

    seen: list[Any] = []
    real = getattr(smcfcs, "model_width", None)

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen.append(real(*args, **kwargs))
        return seen[-1]

    run = GraphRun(path, folder)
    try:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(smcfcs, "model_width", spy, raising=False)
            out = run.public(run.run(state, upto=["fit"]))
    finally:
        run.close()
    return out["fit"], seen


def test_the_fit_refuses_the_150_row_plan_with_the_answer_s_words(tmp_path):
    """The 150-row plan run through the stage graph: the fit's table is refused with the words the
    answer is refused with, and the width the fit's sampler measured is the one the check at the
    answer measured."""
    table = wide(150)
    path = tmp_path / "wide.csv"
    table.to_csv(path, index=False)
    state = wide_state(missing=MissingSpec(strategy="multiple_imputation"),
                       split={"holdout": 0.0, "seed": 0, "folds": 5}, models=["linear"],
                       lens=["clinical"])
    fit, seen = _graph_fit(path, tmp_path / "project", state)
    assert LEAD + _refusal_words(table) in repr(fit)
    ingested = Ingested(path, tmp_path / "answer")
    at = smcfcs.imputation_at(state, ctx_of(state, ingested))
    assert seen and seen[0].width == at.sized.width == WIDTH
    assert [(m.column, m.parameters, m.rows) for m in seen[0].models] == \
        [(m.column, m.parameters, m.rows) for m in at.sized.models]


def test_the_fit_never_names_the_exposure_to_leave_out_and_says_what_the_answer_says(tmp_path):
    """The exposure `race3` recodes the covariate `race6`, and the covariate `w`, recorded on 172 of
    200 rows, is short of its rows on 89 columns (88 parameters, 1.95 each); without `race3` it
    would have them (86, 2.0 each), but the exposure is what the analysis studies, never a column
    an exit may leave out. The answer names no column to leave out, and the fit, run through the
    stage graph, names none either: its words are the answer's."""
    rng = np.random.default_rng(7)
    n = 200
    race6 = np.array([f"r{k % 6 + 1}" for k in range(n)], dtype=object)
    groups = {"r1": "A", "r2": "A", "r3": "B", "r4": "B", "r5": "C", "r6": "C"}
    w = rng.normal(size=n)
    w[172:] = np.nan
    cols = numbers(n, 80, rng)
    table = pd.DataFrame({"y": rng.normal(size=n), "w": w, **cols, "race6": race6,
                          "race3": np.array([groups[v] for v in race6], dtype=object)})
    path = tmp_path / "exposure.csv"
    table.to_csv(path, index=False)
    roles = {"race3": "exposure", "w": "covariate", **{c: "covariate" for c in cols},
             "race6": "covariate"}
    state = ProjectState(target="y", task="regression", purpose="inference",
                         grain={"grain": "one_row_per_unit"}, roles=roles, exclusions=[],
                         missing=MissingSpec(strategy="multiple_imputation"),
                         split={"holdout": 0.0, "seed": 0, "folds": 5}, models=["linear"],
                         lens=["clinical"])
    with pytest.raises(Refusal) as answered:
        validate({"kind": "lock_plan"}, ctx_of(state, Ingested(path, tmp_path / "answer")))
    assert [e["label"] for e in answered.value.exits] == [
        "Complete cases, with their assumption stated"]
    assert answered.value.message == (
        f"{LEAD}{opening(89)}The chained equations would impute `w` from a model of 88 predictor "
        f"parameters (categories as indicators) fit on the 172 rows that record it, 1.95 per "
        f"parameter, {SOURCE}.")
    fit, _ = _graph_fit(path, tmp_path / "project", state)
    [said] = [m["inference"]["caption"] for m in fit["models"]]
    assert said == f"No coefficients: {answered.value.message}"
    assert "`race3`" not in said


# ── the answer says the sampler's first refusal, with the fit's ways forward ─


ONCE = "`q` has fewer than two observed values, so it cannot be imputed."
ZEROS = ("`fat` is analyzed on the log scale and is zero or negative on {k}, so it cannot be "
         "imputed on that scale; a zero a log would turn into a blank goes to the detection-limit "
         "question, never to a fill.")
ZEROS_WAY = "Say how the zeros of `fat` are read: values below a detection limit, or true zeros"
NO_LOG = "Adjust for energy without the log"
FIT = {"split": {"holdout": 0.0, "seed": 0, "folds": 5}, "models": ["linear"], "lens": ["clinical"]}


def _with_q(folder: Any, n: int, recorded: str) -> tuple[pd.DataFrame, Any, ProjectState]:
    """:func:`wide`'s table with a covariate `q` (91 model columns), recorded on one row
    (``"once"``) or on the last 200, which an eligibility rule leaves out of the analysis
    (``"excluded"``), under multiple imputation."""
    from turbotab.core.decisions import ExclusionRule

    table = wide(n)
    q = np.full(n, np.nan)
    if recorded == "once":
        q[0] = 1.5
    else:
        q[n - 200:] = np.random.default_rng(6).normal(size=200)  # amounts, not codes
    table["q"] = q
    path = folder / "wide_q.csv"
    table.to_csv(path, index=False)
    rules = ([ExclusionRule(column="id", high=float(n - 201), reason="the last 200 rows")]
             if recorded == "excluded" else [])
    state = wide_state(roles={**WIDE_ROLES, "q": "covariate"}, exclusions=rules,
                       missing=MissingSpec(strategy="multiple_imputation"), **FIT)
    return table, path, state


def _fit_table(path: Any, folder: Any, state: ProjectState) -> dict[str, Any]:
    """The linear family's coefficient table, as the fit run through the stage graph serves it."""
    fit, _ = _graph_fit(path, folder, state)
    [table] = [m["inference"] for m in fit["models"]]
    return table


@pytest.mark.parametrize("recorded", ["once", "excluded"])
def test_a_column_with_fewer_than_two_values_is_refused_alike_at_the_answer_and_the_fit(
        tmp_path, recorded):
    """On 8,000 rows :func:`wide`'s 90 columns are drawn; with `q` recorded on one analyzed row, or
    on none (its 200 values on rows the eligibility rule leaves out), its model falls short and
    the width rule refuses the 91. The sampler refuses first that `q` has fewer than two values,
    so the answer, its lock and the fit say that, and offer the one way the fit offers: leaving
    `q` out, which the record accepts (the 90 are drawn). Complete cases keep one row or none,
    which the record refuses, so neither offers them."""
    table, path, state = _with_q(tmp_path, 8000, recorded)
    analyzed = table if recorded == "once" else table[table["id"] <= 7799]
    assert int(analyzed["q"].notna().sum()) == (1 if recorded == "once" else 0)
    ingested = Ingested(path, tmp_path / "answer")
    ctx = ctx_of(state, ingested)
    assert smcfcs.imputation_at(state, ctx).sized.refused()  # the width rule refuses it too
    with pytest.raises(Refusal) as answered:
        validate({"kind": "lock_plan"}, ctx)
    assert answered.value.message == LEAD + ONCE
    exits = answered.value.exits
    assert [e["label"] for e in exits] == ["Leave `q` out"]
    assert exits[0]["decision"]["strategy"] == "multiple_imputation"
    assert exits[0]["decision"]["drop_columns"] == ["q"]
    validate(exits[0]["decision"], ctx)  # taken, it is accepted
    before = state.model_copy(update={"missing": None})  # not yet answered
    with pytest.raises(Refusal) as refused:
        validate(SetMissing(strategy="multiple_imputation"), ctx_of(before, ingested))
    assert refused.value.message == LEAD + ONCE
    assert [e["label"] for e in refused.value.exits] == ["Leave `q` out"]
    said = _fit_table(path, tmp_path / "project", state)
    assert said["caption"] == f"No coefficients: {LEAD}{ONCE}"
    assert [e["label"] for e in said["exits"]] == ["Leave `q` out"]
    assert said["exits"][0]["decision"]["drop_columns"] == ["q"]


def test_the_exposure_recorded_once_is_never_offered_to_leave_out(tmp_path):
    """`x`, what the analysis studies, recorded on one of :func:`wide`'s 8,000 rows: the answer and
    the fit say `x` has fewer than two values, and neither offers to leave it out, since an exit
    leaves out a settled covariate only; complete cases keep one row, which the record refuses."""
    table = wide(8000)
    table.loc[1:, "x"] = np.nan
    path = tmp_path / "wide_x.csv"
    table.to_csv(path, index=False)
    state = wide_state(missing=MissingSpec(strategy="multiple_imputation"), **FIT)
    assert int(table["x"].notna().sum()) == 1
    once = "`x` has fewer than two observed values, so it cannot be imputed."
    with pytest.raises(Refusal) as answered:
        validate({"kind": "lock_plan"}, ctx_of(state, Ingested(path, tmp_path / "answer")))
    assert answered.value.message == LEAD + once
    assert answered.value.exits == []
    said = _fit_table(path, tmp_path / "project", state)
    assert said["caption"] == f"No coefficients: {LEAD}{once}"
    assert said["exits"] == []


def test_where_leaving_the_column_out_leaves_a_model_short_neither_offers_it(tmp_path):
    """`q` recorded once on :func:`wide`'s 150 rows: the answer and the fit say `q` has fewer than
    two values, and neither offers to leave it out, since the 90 columns without it are refused on
    150 rows (above); complete cases keep one row, which the record refuses."""
    _, path, state = _with_q(tmp_path, 150, "once")
    with pytest.raises(Refusal) as answered:
        validate({"kind": "lock_plan"}, ctx_of(state, Ingested(path, tmp_path / "answer")))
    assert answered.value.message == LEAD + ONCE
    assert answered.value.exits == []
    said = _fit_table(path, tmp_path / "project", state)
    assert said["caption"] == f"No coefficients: {LEAD}{ONCE}"
    assert said["exits"] == []


KCAL_ONCE = "`kcal` has fewer than two observed values, so it cannot be imputed."


def _kcal_once(table: pd.DataFrame, path: Any) -> pd.DataFrame:
    """``table`` with total energy `kcal` recorded on its first row only, written to ``path``."""
    kcal = np.full(len(table), np.nan)
    kcal[0] = 2100.0
    table = table.assign(kcal=kcal)
    table.to_csv(path, index=False)
    return table


def test_above_the_width_neither_offers_to_leave_out_a_column_that_is_not_a_covariate(tmp_path):
    """`kcal`, total energy in the standard energy model (its role energy, not a covariate),
    recorded on one of :func:`wide`'s 8,000 rows: the width rule refuses the 91 columns, so the
    answer says `kcal` has fewer than two values and, an exit leaving out a settled covariate
    only, offers no way to leave it out (the energy model names it); complete cases keep one row,
    which the record refuses. The fit says the same with the same ways forward, none: it offers
    to leave out what the answer offers where the answer refused the plan in these words."""
    from turbotab.core.decisions import EnergyAdjustment

    path = tmp_path / "wide_kcal.csv"
    table = _kcal_once(wide(8000), path)
    assert int(table["kcal"].notna().sum()) == 1
    state = wide_state(roles={**WIDE_ROLES, "kcal": "energy"},
                       energy_adjustment=EnergyAdjustment(method="standard", energy_column="kcal",
                                                          nutrients=["x"]),
                       missing=MissingSpec(strategy="multiple_imputation"), **FIT)
    ctx = ctx_of(state, Ingested(path, tmp_path / "answer"))
    found = smcfcs.imputation_at(state, ctx)
    assert found.sized.width == WIDTH + 1 and found.sized.refused()  # `kcal`'s model: one row
    assert "kcal" not in found.leavable
    with pytest.raises(Refusal) as answered:
        validate({"kind": "lock_plan"}, ctx)
    assert answered.value.message == LEAD + KCAL_ONCE
    assert answered.value.exits == []
    said = _fit_table(path, tmp_path / "project", state)
    assert said["caption"] == f"No coefficients: {LEAD}{KCAL_ONCE}"
    assert said["exits"] == []


def test_within_the_width_the_fit_offers_to_leave_out_any_column_but_the_exposure(tmp_path):
    """`kcal`, of the energy role and named by no energy model, recorded on one of 300 rows of a
    model 8 columns wide (`x`, `kcal`, five covariates and the outcome's term): the width rule does
    not refuse it, but the sampler refuses that `kcal` has fewer than two values, so the answer is
    refused in those words, and the fit's table says the same; both offer what the fit offered
    before the check at the answer: to leave out any column but the exposure (and those an energy
    model is computed with, none here), `kcal` here. Taken, the record accepts it, and the fit
    pools the copies drawn without `kcal`: its imputation model holds `x`, the five covariates and
    the outcome."""
    from turbotab.core.methods.imputation import M_DEFAULT

    rng = np.random.default_rng(4)
    n = 300
    x = np.exp(rng.normal(3, 0.3, size=n))
    x[rng.random(n) < 0.1] = np.nan
    cols = {f"c{j}": rng.normal(size=n) for j in range(1, 6)}
    path = tmp_path / "kcal.csv"
    _kcal_once(pd.DataFrame({"id": np.arange(n), "y": rng.normal(size=n), "x": x, **cols}), path)
    roles = {"id": "identifier", "x": "exposure", **{c: "covariate" for c in cols},
             "kcal": "energy"}
    state = wide_state(roles=roles, missing=MissingSpec(strategy="multiple_imputation"), **FIT)
    ctx = ctx_of(state, Ingested(path, tmp_path / "answer"))
    assert smcfcs.imputation_at(state, ctx).sized.width == 1 + 1 + 5 + 1
    with pytest.raises(Refusal) as answered:  # the answer refuses what the sampler refuses
        validate({"kind": "lock_plan"}, ctx)
    assert answered.value.message == LEAD + KCAL_ONCE
    assert [e["label"] for e in answered.value.exits] == ["Leave `kcal` out"]
    said = _fit_table(path, tmp_path / "project", state)
    assert said["caption"] == f"No coefficients: {LEAD}{KCAL_ONCE}"
    assert [e["label"] for e in said["exits"]] == ["Leave `kcal` out"]
    left = said["exits"][0]["decision"]
    assert left["strategy"] == "multiple_imputation" and left["drop_columns"] == ["kcal"]
    validate(left, ctx)  # taken, it is accepted
    taken = state.model_copy(update={"missing": MissingSpec(strategy="multiple_imputation",
                                                            drop_columns=["kcal"])})
    drawn = _fit_table(path, tmp_path / "taken", taken)
    assert drawn["caption"].startswith(f"Multiple imputation, m = {M_DEFAULT}:")
    assert drawn["refused"] is None and drawn["missing"]["m"] == M_DEFAULT
    assert drawn["missing"]["variables"] == ["x", *cols, "the outcome"]


def _fat(folder: Any, method: str, zeros: int = 1) -> tuple[pd.DataFrame, Any, ProjectState]:
    """:func:`wide`'s 150 rows with its exposure `fat`, about 10% blank and zero on ``zeros`` rows,
    adjusted for total energy `kcal` (a density, or the log residual method): the analysis takes
    `fat`'s log, and the model's 90 columns are refused by the width rule."""
    from turbotab.core.decisions import EnergyAdjustment

    table = wide(150).rename(columns={"x": "fat"})
    rng = np.random.default_rng(3)
    fat = np.exp(rng.normal(3, 0.3, 150))
    fat[:zeros] = 0.0
    fat[zeros:zeros + 15] = np.nan
    table["fat"] = fat
    table["kcal"] = np.exp(rng.normal(7.5, 0.2, 150))
    path = folder / "wide_fat.csv"
    table.to_csv(path, index=False)
    roles = {**{c: r for c, r in WIDE_ROLES.items() if c != "x"}, "fat": "exposure",
             "kcal": "energy"}
    adjustment = EnergyAdjustment(method=method, energy_column="kcal", nutrients=["fat"],
                                  log_transform=method == "residual")
    state = wide_state(roles=roles, energy_adjustment=adjustment,
                       missing=MissingSpec(strategy="multiple_imputation"), **FIT)
    return table, path, state


@pytest.mark.parametrize("zeros, rows", [(1, "1 row"), (3, "3 rows")])
def test_a_logged_column_s_zeros_are_refused_alike_at_the_answer_and_the_fit(tmp_path, zeros,
                                                                             rows):
    """`fat` as a density is analyzed on the log scale, and is zero on ``zeros`` rows (pandas): the
    sampler refuses its zeros before the width, so the answer and the fit say that, a count of one
    in the singular, with the one way forward the fit offers (how the zeros are read)."""
    table, path, state = _fat(tmp_path, "density", zeros)
    assert int((table["fat"] <= 0).sum()) == zeros
    ctx = ctx_of(state, Ingested(path, tmp_path / "answer"))
    assert smcfcs.imputation_at(state, ctx).sized.refused()
    with pytest.raises(Refusal) as answered:
        validate({"kind": "lock_plan"}, ctx)
    assert answered.value.message == LEAD + ZEROS.format(k=rows)
    assert [(e["label"], e["decision"]) for e in answered.value.exits] == [(ZEROS_WAY, None)]
    said = _fit_table(path, tmp_path / "project", state)
    assert said["caption"] == f"No coefficients: {answered.value.message}"
    assert [(e["label"], e["decision"]) for e in said["exits"]] == [(ZEROS_WAY, None)]


def test_under_the_log_residual_the_answer_offers_no_way_the_record_refuses(tmp_path):
    """Under the log residual method, `fat`'s zero is refused at the answer in the words of the
    fit's table (``stages.modeling._missing_for_table``, called on the frame the answer built,
    since the design stage's own residual fit stops at the zero first). Without the log the
    imputation model holds the same 91 columns on the same 150 rows (the log enters no column's
    count), which the width rule refuses, so the record refuses the energy adjustment without the
    log, and the answer offers only how the zeros are read."""
    from turbotab.core.graph import StageContext
    from turbotab.core.stages.modeling import _missing_for_table

    table, path, state = _fat(tmp_path, "residual")
    ctx = ctx_of(state, Ingested(path, tmp_path / "answer"))
    with pytest.raises(Refusal) as answered:
        validate({"kind": "lock_plan"}, ctx)
    assert answered.value.message == LEAD + ZEROS.format(k="1 row")
    assert [(e["label"], e["decision"]) for e in answered.value.exits] == [(ZEROS_WAY, None)]
    without = {"kind": "set_energy_adjustment", **state.energy_adjustment.model_dump(),
               "log_transform": False}
    with pytest.raises(Refusal) as taken:
        validate(without, ctx)
    assert taken.value.code == smcfcs.WIDTH_REFUSAL
    assert taken.value.message.startswith(LEAD + opening(WIDTH + 1))  # `fat` for `x`, and `kcal`
    spec = smcfcs.imputation_at(state, ctx).spec
    stage = StageContext(project_id="fat", state=state, inputs={}, paths={}, settings={})
    held = _missing_for_table(stage, spec, table[list(spec.inputs)], table["y"].to_numpy(),
                              "regression", ["linear"])
    assert held.refusal == answered.value.message


def test_the_answer_offers_the_way_without_the_log_where_the_record_accepts_it(tmp_path,
                                                                               monkeypatch):
    """The guard above the other way. No state has the answer refuse a logged zero and the record
    accept the energy adjustment without the log (the model the width rule refused is as wide
    without it), so the record's check is made to accept every answer: the answer then offers
    the way without the log first, as the fit's table words it, then how the zeros are read."""
    from turbotab.core import row_floor

    _, path, state = _fat(tmp_path, "residual")
    monkeypatch.setattr(row_floor, "accepted_exit", lambda decision, ctx: dict(decision))
    with pytest.raises(Refusal) as answered:
        validate({"kind": "lock_plan"}, ctx_of(state, Ingested(path, tmp_path / "answer")))
    assert answered.value.message == LEAD + ZEROS.format(k="1 row")
    assert [e["label"] for e in answered.value.exits] == [NO_LOG, ZEROS_WAY]
    assert answered.value.exits[0]["decision"] == {
        "kind": "set_energy_adjustment", **state.energy_adjustment.model_dump(),
        "log_transform": False}


def test_the_fit_names_the_recode_to_leave_out_as_the_answer_does(tmp_path):
    """:func:`_recoded_project` (`race3` a covariate recoding `race6`) run through the stage
    graph: the fit's table is refused in the answer's words, the sentence naming `race3` among
    them, with the answer's ways forward: complete cases and leaving `race3` out."""
    state, _ = _recoded_project(tmp_path)
    state = ProjectState(**{**state.model_dump(), **FIT})
    path = tmp_path / "recoded.csv"
    with pytest.raises(Refusal) as answered:
        validate({"kind": "lock_plan"}, ctx_of(state, Ingested(path, tmp_path / "answer")))
    said = _fit_table(path, tmp_path / "project", state)
    assert said["caption"] == f"No coefficients: {answered.value.message}"
    assert said["caption"].endswith(
        SPANNED.format(n=172) + "each incomplete column's model has the rows per parameter its "
        "rule asks for.")
    assert [e["label"] for e in said["exits"]] == [e["label"] for e in answered.value.exits] == [
        "Complete cases, with their assumption stated", "Leave `race3` out"]
    assert said["exits"][1]["decision"]["drop_columns"] == ["race3"]


def test_the_answer_and_the_fit_read_one_refusal_lead(tmp_path, monkeypatch):
    """The lead both refusals begin with is one constant (``smcfcs.REFUSED_LEAD``): changed there,
    the answer's refusal and the fit's both begin with the change."""
    table = wide(150)
    path = tmp_path / "wide.csv"
    table.to_csv(path, index=False)
    state = wide_state(missing=MissingSpec(strategy="multiple_imputation"), **FIT)
    assert smcfcs.REFUSED_LEAD == LEAD
    monkeypatch.setattr(smcfcs, "REFUSED_LEAD", "A LEAD UNDER TEST: ")
    with pytest.raises(Refusal) as answered:
        validate({"kind": "lock_plan"}, ctx_of(state, Ingested(path, tmp_path / "answer")))
    assert answered.value.message == "A LEAD UNDER TEST: " + _refusal_words(table)
    said = _fit_table(path, tmp_path / "project", state)
    assert said["caption"] == f"No coefficients: {answered.value.message}"

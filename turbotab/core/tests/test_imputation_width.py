"""How wide a chained-equations imputation model may be, judged by its rows, the same at the answer
as at Fit (INBOX: Sullivan 2021's Haiku runs 22 and 26 locked multiple imputation that Fit then
refused at a fixed 80 columns, whatever the rows, after the lock).

Each incomplete column's model is held to the engine's rows-per-parameter rule on the rows that
record it: about 2 rows per predictor parameter for least squares (Austin & Steyerberg 2015, J Clin
Epidemiol 68:627), 10 events (its rarer value's rows) per parameter for a model of two levels
(Peduzzi et al. 1996, the engine's rule for logistic coefficients), and 10 per parameter of the
effective sample size n − Σn³/n² for a multinomial model, whose parameters are a slope per
predictor for each level beyond the first. The chain fits a numeric code the user declared a
category as its indicators, as the width counts it. Every expected value here is from
construction or a derivation: the parameters counted from the columns built, the rows and events
from pandas, the effective size by its formula, the multinomial model's slopes from
scikit-learn, and the width by R's ``ncol(model.matrix(~ ., data)) - 1`` on the imputation frame,
its strata a factor. Each boundary is tested on both sides; the brief's scale (about 90 columns:
8,000 rows imputed, 150 refused) through the sampler, through the answer and through the plan's
lock, and through the fit (and over HTTP, ``server/tests/test_imputation_width_api.py``).
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

SPV = 2  # Austin & Steyerberg 2015: about 2 subjects per variable for least-squares coefficients
EPV = 10  # Peduzzi et al. 1996: events per parameter (a multinomial model: effective size)
LEAD = "Multiple imputation cannot run on these data: "
SOURCE = ("where Austin & Steyerberg 2015 found about 2 per parameter enough for least-squares "
          "coefficients, standard errors and intervals")


def effective(counts: list[int]) -> float:
    """Whitehead's effective sample size, n − Σ n_k³ / n², by its formula."""
    n = float(sum(counts))
    return n - sum(float(c) ** 3 for c in counts) / n ** 2


def per(counted: float, parameters: int) -> str:
    return f"{counted / parameters:.1f}"


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


# ── the rule's boundaries, on both sides ─────────────────────────────────────


def _least_squares(recorded: int) -> pd.DataFrame:
    """`x` recorded on ``recorded`` of 30 rows beside nine complete numbers and the outcome's
    term: its model has 10 predictor parameters by construction."""
    rng = np.random.default_rng(2)
    frame = pd.DataFrame({"x": rng.normal(size=30), **numbers(30, 9, rng),
                          "__outcome": rng.normal(size=30)})
    frame.loc[recorded:, "x"] = np.nan
    return frame


def test_least_squares_is_drawn_at_two_rows_per_parameter_and_refused_below():
    """10 parameters: 20 recorded rows (2.0 each) are imputed; 19 (1.9 each) are refused, in words
    whose counts are the frame's own."""
    drawn = impute(_least_squares(20), [Variable("x", "numeric")], m=2, iterations=1)
    assert drawn.m == 2 and drawn.imputed == {"x": 10}
    frame = _least_squares(19)
    assert int(frame["x"].notna().sum()) == 19 < SPV * 10
    with pytest.raises(ImputationRefused) as refused:
        impute(frame, [Variable("x", "numeric")], m=2, iterations=1)
    assert str(refused.value) == (
        f"The chained equations would impute `x` from a model of 10 predictor parameters "
        f"(categories as indicators) fit on the 19 rows that record it, {per(19, 10)} per "
        f"parameter, {SOURCE}.")
    assert getattr(refused.value, "columns", None) is None  # nothing the others already span


def _several_short(k: int) -> pd.DataFrame:
    """``k`` incomplete numbers `a1` … `ak` on 30 rows, `aj` recorded on its first 11 + j, beside
    nine complete numbers and the outcome's term: each model has k + 9 predictor parameters (the
    other k − 1, the nine and the outcome's), so every one falls short of 2 per parameter (11 + j
    rows against 2k + 18), `a1` the furthest (its rows are the fewest)."""
    rng = np.random.default_rng(11)
    frame = pd.DataFrame({**{f"a{j}": rng.normal(size=30) for j in range(1, k + 1)},
                          **numbers(30, 9, rng), "__outcome": rng.normal(size=30)})
    for j in range(1, k + 1):
        frame.loc[11 + j:, f"a{j}"] = np.nan
    return frame


@pytest.mark.parametrize("k, tail", [
    (2, "The model of `a2` falls short of the rule too."),
    (3, "The models of `a2` and `a3` fall short of the rule too."),
    (6, "The models of `a2`, `a3`, `a4` and 2 more fall short of the rule too."),
])
def test_the_models_short_after_the_first_are_named_as_one_or_as_several(k, tail):
    """One model short after the first is named in the singular; several in the plural, three by
    name and the rest counted. The parameters and rows are the frame's own: `a1`'s 12 recorded
    rows against k + 9 parameters, below 2 each for every k here."""
    frame = _several_short(k)
    p = k + 9
    assert [int(frame[f"a{j}"].notna().sum()) for j in range(1, k + 1)] == list(range(12, 12 + k))
    assert 11 + k < SPV * p  # the last column, the nearest the rule, is short too
    with pytest.raises(ImputationRefused) as refused:
        impute(frame, [Variable(f"a{j}", "numeric") for j in range(1, k + 1)], m=2, iterations=1)
    assert str(refused.value) == (
        f"The chained equations would impute `a1` from a model of {p} predictor parameters "
        f"(categories as indicators) fit on the 12 rows that record it, {per(12, p)} per "
        f"parameter, {SOURCE}. {tail}")


def _yes_no(ones: int, zeros: int) -> pd.DataFrame:
    """A 0/1 column `b` recorded as ``ones`` ones and ``zeros`` zeros, blank on five more rows,
    beside two complete numbers and the outcome's term: 3 predictor parameters."""
    rng = np.random.default_rng(3)
    n = ones + zeros + 5
    b = np.r_[np.ones(ones), np.zeros(zeros), np.full(5, np.nan)]
    return pd.DataFrame({"b": b, **numbers(n, 2, rng), "__outcome": rng.normal(size=n)})


def test_a_yes_no_model_is_held_to_ten_events_per_parameter_on_its_rarer_value():
    """3 parameters need 30 events, the rows of the rarer value (Peduzzi's count, which the
    engine's logistic rule counts): 30 ones beside 60 zeros are imputed; 29 ones beside 60 zeros
    are refused, and so are 29 zeros beside 60 ones, in words of events. The effective sample size
    3ab/n would have drawn both (58.7 of the 30 asked)."""
    assert 30 == EPV * 3 and 29 < EPV * 3 <= effective([29, 60])
    impute(_yes_no(30, 60), [Variable("b", "binary")], m=2, iterations=1)
    impute(_yes_no(60, 30), [Variable("b", "binary")], m=2, iterations=1)
    for ones, zeros in ((29, 60), (60, 29)):
        with pytest.raises(ImputationRefused) as refused:
            impute(_yes_no(ones, zeros), [Variable("b", "binary")], m=2, iterations=1)
        assert str(refused.value) == (
            f"The chained equations would impute `b` from a model of 3 predictor parameters "
            f"(categories as indicators) fit on the 89 rows that record it, 29 of them its rarer "
            f"value, {per(29, 3)} events per parameter, where a common rule for logistic "
            f"coefficients asks for 10 (Peduzzi et al. 1996).")


def _categories(counts: list[int], levels: str = "pqr") -> pd.DataFrame:
    rng = np.random.default_rng(4)
    values = [lv for lv, k in zip(levels, counts) for _ in range(k)] + [np.nan] * 4
    n = len(values)
    return pd.DataFrame({"g": pd.Series(values, dtype=object), **numbers(n, 2, rng),
                         "__outcome": rng.normal(size=n)})


def test_a_category_of_two_levels_is_a_logistic_model_held_to_its_events():
    """Text of two levels is drawn by the same logistic model as a 0/1 column, so it is held to
    the same events: 30 `p` beside 60 `q` on 3 parameters are imputed, 29 are refused."""
    impute(_categories([30, 60], "pq"), [Variable("g", "categorical")], m=2, iterations=1)
    with pytest.raises(ImputationRefused) as refused:
        impute(_categories([29, 60], "pq"), [Variable("g", "categorical")], m=2, iterations=1)
    assert "fit on the 89 rows that record it, 29 of them its rarer value, " \
           f"{per(29, 3)} events per parameter" in str(refused.value)


def test_a_multinomial_model_counts_a_slope_per_predictor_for_each_level_beyond_the_first():
    """Three levels on 3 predictor parameters: scikit-learn's multinomial model fits
    (3 − 1)·3 = 6 identified slopes, so the rule asks for an effective size of 60. 22, 23 and 23
    recorded give 60.4 and are imputed; 22, 22 and 23 give 59.5 and are refused, in words that
    count the 6 parameters. Counted as 3, both would have been drawn (30 asked)."""
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    model = LogisticRegression(max_iter=500).fit(rng.normal(size=(90, 3)),
                                                 np.repeat(list("pqr"), 30))
    assert model.coef_.shape == (3, 3)  # one row per level: identified up to a common shift
    assert effective([22, 23, 23]) >= EPV * 6 > effective([22, 22, 23]) >= EPV * 3
    impute(_categories([22, 23, 23]), [Variable("g", "categorical")], m=2, iterations=1)
    with pytest.raises(ImputationRefused) as refused:
        impute(_categories([22, 22, 23]), [Variable("g", "categorical")], m=2, iterations=1)
    size = effective([22, 22, 23])
    assert str(refused.value) == (
        f"The chained equations would impute `g` from a multinomial model of 6 parameters (3 "
        f"predictor parameters, categories as indicators, for each of its 2 levels beyond the "
        f"first) fit on the 67 rows that record it, an effective sample size of {size:,.1f} "
        f"(n − Σn³/n² over its 3 levels), {per(size, 6)} per parameter, where a common rule of "
        f"thumb asks for 10.")


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
    """The refusal's words for :func:`wide`'s table, from pandas' counts: `smoke`'s model is the
    furthest short (its rarer value's rows, the events, against 10 per parameter), then `x`'s."""
    smoke = table["smoke"].dropna()
    events = int(smoke.value_counts().min())
    p = WIDTH - 1
    return (f"The chained equations would impute `smoke` from a model of {p} predictor parameters "
            f"(categories as indicators) fit on the {len(smoke):,} rows that record it, {events:,} "
            f"of them its rarer value, {per(events, p)} events per parameter, where a common rule "
            f"for logistic coefficients asks for 10 (Peduzzi et al. 1996). The model of `x` falls "
            f"short of the rule too.")


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


def test_where_the_survey_design_is_most_of_the_width_the_words_say_so_and_offer_the_sample():
    """60 strata of two rows: the design's 59 + 1 + 1 terms are 61 of the model's 71 columns, and
    without them `x`'s model has 9 parameters on its 108 rows, so the sample answer is offered.
    With 5 strata of 24 rows the design is 6 of 16 and goes unsaid (`x`, 108 rows on 15
    parameters, is drawn: 7.2 each)."""
    from turbotab.core.methods.missing import WITHOUT_THE_DESIGN, sample_way

    frame = _surveyed(60)
    assert int(frame["x"].notna().sum()) == 108
    with pytest.raises(ImputationRefused) as refused:
        impute(frame, [Variable("x", "numeric")], m=2, iterations=1)
    said = str(refused.value)
    assert said.startswith("The chained equations would impute `x` from a model of 70 predictor "
                           "parameters (categories as indicators) fit on the 108 rows that "
                           f"record it, {per(108, 70)} per parameter")
    assert ("Of the imputation model's 71 columns, 61 are the survey design's: 59 indicators of "
            "its strata, one indicator of its PSUs within their strata and its weight. "
            f"{WITHOUT_THE_DESIGN}") in said
    assert refused.value.exits == [sample_way()]
    # silent where the design is not most of the width
    few = _surveyed(5, rows_per=24)
    impute(few, [Variable("x", "numeric")], m=2, iterations=1)
    tight = few.assign(**{f"d{j}": np.random.default_rng(j).normal(size=len(few))
                          for j in range(40)})
    with pytest.raises(ImputationRefused) as refused:
        impute(tight, [Variable("x", "numeric")], m=2, iterations=1)
    assert "survey design" not in str(refused.value)
    assert getattr(refused.value, "exits", None) is None
    # most of the width, but without the design `x`'s model is still short (59 parameters on 108
    # rows, 1.8 each): the words count the design's terms, and offer no sample answer
    crowded = frame.assign(**{f"d{j}": np.random.default_rng(j).normal(size=len(frame))
                              for j in range(50)})
    with pytest.raises(ImputationRefused) as refused:
        impute(crowded, [Variable("x", "numeric")], m=2, iterations=1)
    assert "Of the imputation model's 121 columns, 61 are the survey design's" in str(refused.value)
    assert WITHOUT_THE_DESIGN not in str(refused.value)
    assert getattr(refused.value, "exits", None) is None


def _recoded(recode: bool) -> pd.DataFrame:
    """`x` recorded on 30 of 40 rows beside eight numbers, a six-level code `race6`, `race3` (its
    recode into three groups, or three groups drawn apart from it) and the outcome's term: `x`'s
    model has 8 + 5 + 2 + 1 = 16 parameters (1.9 each), 14 without `race3` (2.1 each)."""
    rng = np.random.default_rng(7)
    n = 40
    race6 = np.array([f"r{k % 6 + 1}" for k in range(n)], dtype=object)
    groups = {"r1": "A", "r2": "A", "r3": "B", "r4": "B", "r5": "C", "r6": "C"}
    race3 = (np.array([groups[v] for v in race6], dtype=object) if recode
             else rng.choice(list("ABC"), n).astype(object))
    x = rng.normal(size=n)
    x[30:] = np.nan
    return pd.DataFrame({"x": x, **numbers(n, 8, rng), "race6": race6, "race3": race3,
                         "__outcome": rng.normal(size=n)})


def test_a_recode_beside_its_code_is_offered_to_leave_out_and_nothing_else_is():
    """`race3` is a function of `race6` by construction, so its indicators lie in the span of
    `race6`'s and the intercept: the refusal names it, and without it `x`'s model has its rows
    (2.1 per parameter). Drawn apart from `race6`, no column is named."""
    variables = [Variable("x", "numeric")]
    with pytest.raises(ImputationRefused) as refused:
        impute(_recoded(True), variables, m=2, iterations=1)
    assert refused.value.columns == ["race3"]
    assert str(refused.value).endswith(
        "`race3` adds nothing the other columns do not hold on the 30 rows that record them all "
        "(as a recode adds nothing beside the column it recodes), and without it each incomplete "
        "column's model has the rows per parameter the rule asks for.")
    impute(_recoded(True).drop(columns=["race3"]), variables, m=2, iterations=1)  # drawn
    with pytest.raises(ImputationRefused) as refused:
        impute(_recoded(False), variables, m=2, iterations=1)
    assert getattr(refused.value, "columns", None) is None
    from turbotab.core.methods.smcfcs import redundant_columns

    kinds = smcfcs.chain_kinds(_recoded(False), variables)
    assert redundant_columns(_recoded(False), kinds, ["race3", "race6"]) == ([], 30)


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
    back is refused at the answer; one that brings five back (14 parameters, about 9.6 each) is
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


def test_at_the_answer_a_recode_beside_its_code_is_offered_to_leave_out(tmp_path):
    """:func:`_recoded`'s table as a project, `race3` a covariate recoding `race6`: the lock Fit
    records is refused with complete cases and leaving `race3` out, which the missing-values answer
    records as a column left out; taken, the lock is accepted."""
    table = _recoded(True).drop(columns=["__outcome"]).assign(
        y=np.random.default_rng(9).normal(size=40))
    path = tmp_path / "recoded.csv"
    table.to_csv(path, index=False)
    ingested = Ingested(path, tmp_path)
    roles = {"x": "exposure", **{f"c{j:02d}": "covariate" for j in range(1, 9)},
             "race6": "covariate", "race3": "covariate"}
    state = ProjectState(target="y", task="regression", purpose="inference",
                         grain={"grain": "one_row_per_unit"}, roles=roles, exclusions=[],
                         missing=MissingSpec(strategy="multiple_imputation"))
    with pytest.raises(Refusal) as refused:
        validate({"kind": "lock_plan"}, ctx_of(state, ingested))
    exits = refused.value.exits
    assert [e["label"] for e in exits] == ["Complete cases, with their assumption stated",
                                           "Leave `race3` out"]
    assert exits[1]["decision"]["kind"] == "set_missing"
    assert exits[1]["decision"]["strategy"] == "multiple_imputation"
    assert exits[1]["decision"]["drop_columns"] == ["race3"]
    taken = state.model_copy(update={"missing": MissingSpec(strategy="multiple_imputation",
                                                            drop_columns=["race3"])})
    validate({"kind": "lock_plan"}, ctx_of(taken, ingested))


@needs_r
def test_the_frame_at_the_answer_is_the_survey_design_s_and_its_width_is_r_s(tmp_path):
    """A project with a survey design (60 strata of two rows, two PSUs each, a weight) under the
    population answer: the frame the check at the answer builds holds the design's terms, and
    its width is R's model.matrix columns on it (strata a factor); the refusal says the design is
    most of it and offers the sample answer."""
    from turbotab.core.decisions import SurveySpec
    from turbotab.core.models.survey import SAMPLE_EXIT

    rng = np.random.default_rng(8)
    n = 120
    x = rng.normal(size=n)
    x[::10] = np.nan
    table = pd.DataFrame({"y": rng.normal(size=n), "x": x, **numbers(n, 8, rng),
                          "strat": np.repeat(np.arange(101, 161), 2), "psu": np.tile([1, 2], 60),
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
    assert "Of the imputation model's 71 columns, 61 are the survey design's" in refused.value.message
    assert [e["label"] for e in refused.value.exits] == [
        "Complete cases, with their assumption stated", SAMPLE_EXIT]
    found = smcfcs.imputation_at(state, ctx_of(state, ingested))
    data = found.plan.data
    assert {"__stratum", "__psu", "__weight"} <= set(data.columns)
    r = run_r("d <- read.csv(frame_csv, stringsAsFactors = FALSE, na.strings = c('', 'NA'))\n"
              "for (c in names(d)) if (is.character(d[[c]])) d[[c]] <- factor(d[[c]])\n"
              "mf <- model.frame(~ ., d, na.action = na.pass)\n"
              "out(list(width = ncol(model.matrix(~ ., mf)) - 1))\n", {"frame": data}, tmp_path)
    assert found.sized.width == r["width"] == 1 + 8 + 1 + 59 + 1 + 1


# ── the fit says the same ────────────────────────────────────────────────────


def test_the_fit_refuses_the_150_row_plan_with_the_answer_s_words(tmp_path):
    """The 150-row plan run through the stage graph (as recorded before this check, or by a log it
    could not read): the fit's table is refused with the words the answer is refused with, and
    the width the fit's sampler measured is the one the check at the answer measured."""
    from turbotab.core.tests.graph_runner import GraphRun

    table = wide(150)
    path = tmp_path / "wide.csv"
    table.to_csv(path, index=False)
    state = wide_state(missing=MissingSpec(strategy="multiple_imputation"),
                       split={"holdout": 0.0, "seed": 0, "folds": 5}, models=["linear"],
                       lens=["clinical"])
    seen: list[Any] = []
    real = getattr(smcfcs, "model_width", None)

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen.append(real(*args, **kwargs))
        return seen[-1]

    run = GraphRun(path, tmp_path / "project")
    try:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(smcfcs, "model_width", spy, raising=False)
            out = run.public(run.run(state, upto=["fit"]))
    finally:
        run.close()
    assert LEAD + _refusal_words(table) in repr(out["fit"])
    ingested = Ingested(path, tmp_path / "answer")
    at = smcfcs.imputation_at(state, ctx_of(state, ingested))
    assert seen and seen[0].width == at.sized.width == WIDTH
    assert [(m.column, m.parameters, m.rows) for m in seen[0].models] == \
        [(m.column, m.parameters, m.rows) for m in at.sized.models]


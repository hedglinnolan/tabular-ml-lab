"""The data area's claims (``turbotab/core/claims/data.py``; R0_PLAN.md R0b): the Silent tests
stage 1 left missing, and the regime predicates tested both ways.

A Silent test holds a guard quiet where its evidence says it must be: a UL that covers every
source, an NHANES table with no variable measured on a subsample alone, a yes/no outcome in groups
large enough for an intercept each. Each was watched failing on a copy of the base whose guard was
loosened. Its truth is the construction or an independent library, never the claim it tests: this
file names no claim and none of the constants the claims set (meta-test 2), and reads no
threshold, so every case sits far from the cut-off it would cross.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from turbotab.core.claims import data as regimes


# ── BX-040 · a UL that covers every source ───────────────────────────────────────────────────

# The IOM DRI summary tables' adult ULs (19-50 years) for nutrients whose footnotes limit neither
# the form nor the source, so each covers intake from food, water and supplements: calcium
# 2,500 mg, iron 45 mg, zinc 40 mg, selenium 400 µg, vitamin D 100 µg, phosphorus 4,000 mg.
TOTAL_INTAKE_ULS = {"calcium_mg": 2500, "DR1TCALC": 2500, "iron_mg": 45, "DR1TZINC": 40,
                    "DR1TSELE": 400, "vitamin_d_ug": 100, "phosphorus_mg": 4000}


def test_a_ul_that_covers_every_source_is_read_as_total_intake():
    """Claims backtest BX-040, the silent side: the reading of a UL that covers one form only
    (magnesium's supplements, folic acid's, niacin's and vitamin E's synthetic forms, vitamin A's
    retinol) stays off a nutrient whose UL covers food, water and supplements. Expected, from the
    IOM tables above: no form is named for it, and its share above the UL says the UL covers food,
    water and supplements and that a share from foods alone is a lower bound on the share at risk,
    never that one form alone is compared."""
    from turbotab.core import usual_intake as ui
    from turbotab.core.methods import usual_intake as nci

    for name, ul in TOTAL_INTAKE_ULS.items():
        form = ui.ul_form(name)
        assert form is None, (name, form)
        sentence = nci.share_phrase(ul, "UL", ul_form=form)
        assert "food, water and supplements" in sentence, sentence
        assert "lower bound" in sentence, sentence
        assert " only" not in sentence and "that form alone" not in sentence, sentence


# ── C-143 · the NHANES weight's silent sides ─────────────────────────────────────────────────


def _merge(seed: int = 143, n: int = 800) -> pd.DataFrame:
    """An August 2021-August 2023 merge as the CDC files build it (DEMO_L, BMX_L, TCHOL_L, GLU_L,
    DIQ_L): every participant interviewed (WTINT2YR, DIQ010), three in four examined (WTMEC2YR 0
    otherwise, BMXBMI on each examined row), blood drawn from four in five of the examined
    (WTPH2YR positive; 0 for one who gave none, blank off the file), and the morning fasting
    subsample, two in five of those drawn (WTSAF2YR positive)."""
    rng = np.random.default_rng(seed)
    examined = rng.random(n) < 0.75
    drawn = examined & (rng.random(n) < 0.8)
    morning = drawn & (rng.random(n) < 0.4)
    return pd.DataFrame({
        "SEQN": np.arange(130000, 130000 + n), "RIDAGEYR": rng.integers(18, 80, n),
        "DIQ010": rng.choice([1, 2], n, p=[0.12, 0.88]),
        "WTINT2YR": rng.uniform(3000, 90000, n).round(1),
        "WTMEC2YR": np.where(examined, rng.uniform(3000, 90000, n), 0.0).round(1),
        "WTPH2YR": np.where(drawn, rng.uniform(3000, 90000, n), np.where(examined, 0.0, np.nan)),
        "WTSAF2YR": np.where(morning, rng.uniform(5000, 150000, n),
                             np.where(examined, 0.0, np.nan)),
        "BMXBMI": np.where(examined, rng.normal(28, 6, n).round(1), np.nan)})


def test_no_variable_recorded_only_on_blood_drawn_rows_keeps_the_mec_weight():
    """Claims backtest C-143, the silent side of the phlebotomy weight: beside WTPH2YR, a table
    whose variables were measured on examined participants who gave no blood as well (BMI, by
    construction recorded on every examined row, a fifth of them outside the phlebotomy sample)
    is the examination sample's, so the rule names WTMEC2YR and leaves WTPH2YR as a question for
    the user, never settled."""
    from turbotab.core.recognizers import least_common_denominator

    frame = _merge()
    drawn, examined = frame["WTPH2YR"] > 0, frame["WTMEC2YR"] > 0
    bmi = frame["BMXBMI"].notna()
    assert (bmi == examined).all() and (bmi & ~drawn).sum() >= 0.15 * bmi.sum()
    lcd = least_common_denominator(["BMXBMI", "RIDAGEYR", "WTINT2YR", "WTMEC2YR", "WTPH2YR"],
                                   frame)
    assert lcd["use"] == "WTMEC2YR", lcd
    assert lcd["unconfirmed"] == ["WTPH2YR"] and not lcd["settled"], lcd


def test_a_variable_measured_outside_a_subsample_is_not_that_subsample_s():
    """Claims backtest C-143, the silent side of reading a subsample's variable by where it is
    recorded: hs-CRP, by construction measured on the whole phlebotomy sample (so on every fasting
    row and on as many drawn rows outside the fasting subsample), is no fasting variable, whatever
    the fasting rows hold: the rule does not name WTSAF2YR for it, and offers the fasting weight
    only as the user's question."""
    from turbotab.core.recognizers import least_common_denominator, subsample_variables

    frame = _merge()
    drawn, morning = frame["WTPH2YR"] > 0, frame["WTSAF2YR"] > 0
    rng = np.random.default_rng(1431)
    frame["LBXHSCRP"] = np.where(drawn, rng.lognormal(0.5, 1.0, len(frame)).round(2), np.nan)
    crp = frame["LBXHSCRP"].notna()
    assert (crp & morning).sum() == morning.sum() and (crp & ~morning).sum() >= morning.sum()
    assert "LBXHSCRP" not in subsample_variables(frame, "WTSAF2YR")
    lcd = least_common_denominator(["LBXHSCRP", "RIDAGEYR", "WTMEC2YR", "WTSAF2YR"], frame)
    assert lcd["use"] == "WTMEC2YR", lcd
    assert lcd["unconfirmed"] == ["WTSAF2YR"] and not lcd["settled"], lcd


def test_an_examination_variable_keeps_the_mec_weight_beside_interview_ones():
    """Claims backtest C-143, the silent side of the interview weight (the weighting tutorial's
    Example 2, "You would use the MEC exam weight for your analysis"): diabetes diagnosed (DIQ010)
    and age are recorded on every interviewed participant, but BMI only on the examined, so beside
    BMI the rule keeps WTMEC2YR; and a table of examined rows alone is the examination sample, so
    even interview variables keep WTMEC2YR there."""
    from turbotab.core.recognizers import least_common_denominator

    frame = _merge()
    examined = frame["WTMEC2YR"] > 0
    assert frame[["DIQ010", "RIDAGEYR"]].notna().all().all() and (~examined).sum() >= 100
    assert not (frame["BMXBMI"].notna() & ~examined).any()
    interview = ["DIQ010", "RIDAGEYR", "WTINT2YR", "WTMEC2YR"]
    assert least_common_denominator([*interview, "BMXBMI"], frame)["use"] == "WTMEC2YR"
    assert least_common_denominator(interview, frame[examined])["use"] == "WTMEC2YR"


# ── D-058 · groups large enough for an intercept each ────────────────────────────────────────


class _Store:
    """A store a decision context reads (the server's ``DataStore`` shape)."""

    def __init__(self, frame: pd.DataFrame):
        self.frame = frame
        self.columns = list(frame.columns)

    def materialize(self, columns, rows=None):
        out = self.frame[list(columns)]
        return out if rows is None else out.iloc[np.asarray(rows, dtype=int)]


def test_a_yes_no_outcome_in_groups_of_forty_keeps_an_intercept_per_group():
    """Claims backtest D-058, the silent side: the incidental-parameter refusal stays off groups
    large enough that an intercept each estimates the truth. Known truth: 100 households of 40,
    the exposure's log odds ratio 1 given the household. statsmodels' conditional logistic
    regression, which conditions the intercepts away, recovers it, and the unconditional fit with
    one indicator per household (statsmodels' ``Logit``) lands within 0.15 of it. Expected: "Adjust
    and cluster" is accepted, offered first beside clustering alone, and the design stage's
    backstop names no refusal."""
    import warnings

    import statsmodels.api as sm
    from statsmodels.discrete.conditional_models import ConditionalLogit

    from turbotab.core import decisions as d, estimand
    from turbotab.core.decisions import ProjectState, Refusal

    rng = np.random.default_rng(58)
    groups, size = 100, 40
    group = np.repeat(np.arange(groups), size)
    shared = rng.normal(0, 0.5, groups)[group]
    x = rng.normal(0, 1, len(group)) + 0.5 * shared
    y = rng.binomial(1, 1 / (1 + np.exp(-(shared + 1.0 * x))))
    per_group = pd.Series(y).groupby(group).mean()
    assert ((per_group > 0) & (per_group < 1)).all()  # no household's outcome is all one answer
    dummies = pd.get_dummies(group, drop_first=True, dtype=float).to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        conditional = ConditionalLogit(y, x[:, None], groups=group).fit(disp=0).params[0]
        unconditional = sm.Logit(y, sm.add_constant(np.column_stack([x, dummies]))).fit(
            disp=0, maxiter=200).params[1]
    assert abs(conditional - 1.0) < 0.2, conditional
    assert abs(unconditional - conditional) < 0.15, (unconditional, conditional)

    frame = pd.DataFrame({"pid": np.arange(len(y)), "household": group, "x": x, "ill": y})
    state = ProjectState(lens=["clinical"], target="ill", task="binary", purpose="inference",
                         grain={"grain": "one_row_per_unit"},
                         roles={"pid": "identifier", "household": "cluster", "x": "exposure"})
    ctx = {"state": state, "columns": list(frame.columns), "n_rows": len(frame),
           "column_info": {"household": {"dtype": "integer", "n_unique": groups, "n_missing": 0},
                           "ill": {"dtype": "numeric", "n_unique": 2, "n_missing": 0}},
           "store": lambda: _Store(frame), "analyzed": lambda: np.arange(len(frame))}
    adjust = {"kind": "set_clusters", "column": "household", "adjust": "fixed_effects"}
    assert d.validate(adjust, ctx).adjust == "fixed_effects"
    with pytest.raises(Refusal) as how:
        d.validate({"kind": "set_clusters", "column": "household"}, ctx)
    assert how.value.code == "how_adjusted"
    assert [(e["decision"] or {}).get("adjust") for e in how.value.exits] == [
        "fixed_effects", "cluster_only"]
    fixed = state.model_copy(update={"clusters": d.ClusterSpec(column="household",
                                                               adjust="fixed_effects")})
    assert estimand.intercepts_refusal(fixed, _Store(frame), np.arange(len(frame)),
                                       "binary") is None


# ── the regime predicates, both ways ─────────────────────────────────────────────────────────


def test_the_constant_column_regime_counts_a_blank_as_a_value():
    """CLAIMS_REGISTRY.md §10, C-123: a predicate that called ``constant_values`` would share its
    bug. This one counts a blank as a value of its own, so a checkbox ('yes' where ticked, blank
    where not; 1 and blank) is two values, though dropping the blanks first, as the test before
    stage 1 did, counts one."""
    for values in (["yes", None, None, "yes", None], [1.0, math.nan, 1.0]):
        assert pd.Series(values).dropna().nunique() == 1  # the count that read it as a constant
        assert not regimes.one_value_on_every_row({"values": pd.Series(values)}), values
    for values in ([3.0] * 40, ["yes"] * 5):
        assert regimes.one_value_on_every_row({"values": pd.Series(values)}), values
    assert not regimes.one_value_on_every_row({"values": pd.Series([], dtype=float)})
    assert not regimes.one_value_on_every_row({"values": pd.Series([1.0, 2.0])})


@pytest.mark.parametrize("predicate, inside, outside", [
    ("a_share_above_a_ul", {"cutoff": 350, "cutoff_kind": "UL"},
     {"cutoff": 350, "cutoff_kind": "EAR"}),
    ("a_share_above_a_ul", {"cutoff": 2500.0, "cutoff_kind": "UL"},
     {"cutoff": None, "cutoff_kind": "UL"}),
    ("names_an_nhanes_weight", {"columns": ["LBXTC", "WTPH2YR"]},
     {"columns": ["LBXTC", "sample_weight_mg"]}),
    ("names_an_nhanes_weight", {"columns": ["DR1TKCAL", "wtdrd1pp"]},
     {"columns": ["DR1TKCAL", "BMXWT"]}),
    ("under_an_omics_lens", {"lenses": ["clinical", "metabolomics"]},
     {"lenses": ["dietary"]}),
    ("names_a_grouping", {"grouping": "site"}, {"grouping": None}),
    ("a_least_squares_fit_with_a_cut_off", {"fit": "least_squares", "cut_off": 126},
     {"fit": "logistic", "cut_off": 126}),
])
def test_each_regime_predicate_holds_inside_its_regime_and_fails_outside(predicate, inside,
                                                                         outside):
    """Each predicate reads its context as the claim's evidence words it, written apart from the
    code the claim guards: a UL cut-off, a weight named as the CDC codebooks name it (a tissue
    mass and a body weight are no sampling weight), an omics lens, a grouping column, a
    least-squares fit on an outcome with a clinical cut-off."""
    held = getattr(regimes, predicate)
    assert held(inside) is True and held(outside) is False

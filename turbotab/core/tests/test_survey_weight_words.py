"""A pooled survey design's weight, as the answer records it, the fit takes it and its sentence says
it (PP2 follow-up SW1 and SW2; ``survey.offered``, ``survey._pools_by_the_cycle_read``,
``methods.survey.analysis_weights``, ``voice._weight_words``).

SW1: on a 2001–2004 pool (no 1999–2000) the survey question offered the four-year weight, and the
recorded sentence said the 1999–2002 rows took it, "as NCHS directs for 1999–2000", while the fit
weighed every row by the 2-year weight ÷ 2. SW2: a pool answered with no cycle was fit on the
2-year weights as they are, k times the population; a constant scale leaves the design-based
coefficients and their Taylor standard errors where they are, but not the causal lane's lasso
propensity (PP2 follow-up: −1.51320 against −1.51294 on a served 2001–2004 fit), so the answer
is completed with the cycle where the reading certainly reads it, and refused where it does not.

Repair round 1: under prediction the scores' weight is built on the training rows, so the
sentence reads the rows with the target recorded and says a branch only where every set of
training rows takes it (P1); a cycle the roles adjust for is pooled by, its weight no longer k
times the population (P3).

Every expected weight is NCHS's rule applied by hand to the constructed table's columns
(NHANES Analytic Guidelines 2011–2016): §3.1.3, "When combining two or more 2-year cycles from
2001–2002 onward, new multi-year sample weights can be computed by simply dividing the 2-year
sample weights by the number of 2-year cycles in the analysis"; §3.1.4 and Table F, "4 years
1999-2002 Provided on the Public-use Data Files" and "6 years 1999-2004 If sddsrvyr in (1,2) then
MEC6YR = 2/3 * WTMEC4YR; … If sddsrvyr=3 then MEC6YR = 1/3 * WTMEC2YR"; one cycle alone takes its
own 2-year weight.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from turbotab.core import decisions as d
from turbotab.core import voice
from turbotab.core.methods.survey import for_fit, proposal
from turbotab.core.stages.modeling import fit_designs

ROLES = {"SEQN": "identifier", "x": "exposure", "SDDSRVYR": "time", "WTMEC2YR": "design",
         "WTMEC4YR": "design", "SDMVSTRA": "design", "SDMVPSU": "design"}
DESIGN = {"kind": "set_survey", "estimand": "population", "weight": "WTMEC2YR",
          "strata": "SDMVSTRA", "psu": "SDMVPSU"}
FOUR_YEAR_WORDS = ("weighted by `WTMEC4YR` on the 1999–2002 rows (doubled, then divided by the "
                   "number of cycles pooled in `SDDSRVYR`) and by `WTMEC2YR` on the others "
                   "(divided by the same number), as NCHS directs for 1999–2000")
DIVIDED_WORDS = "weighted by `WTMEC2YR` divided by the number of cycles pooled in `SDDSRVYR`"
ALONE_WORDS = "weighted by `WTMEC4YR`, the four-year weight NCHS provides for 1999–2002"
# Under prediction the scores' weight is built on the training rows (``stages.evaluation``'s
# design-based cross-validation, ``stages.modeling.fit_designs``), so their cycles are counted.
SCORED_DIVIDED_WORDS = ("each score weighted by `WTMEC2YR` divided by the number of cycles the "
                        "training rows hold in `SDDSRVYR`, by design-based")
SCORED_RULE_WORDS = ("each score weighted by `WTMEC2YR` (or, on the 1999–2002 rows where "
                     "1999–2000 is pooled, `WTMEC4YR`), pooled by NCHS's rule over the cycles the "
                     "training rows hold in `SDDSRVYR`, by design-based")
SCORED_AS_IT_IS_WORDS = "each score weighted by `WTMEC2YR`, by design-based"


class _Store:
    """The DataStore surface the survey code reads: ``columns`` and ``materialize``."""

    def __init__(self, frame: pd.DataFrame):
        self.frame = frame
        self.columns = list(frame.columns)

    def materialize(self, columns, row_ids=None):
        rows = self.frame if row_ids is None else self.frame.loc[row_ids]
        return rows[list(columns)]


def _pool(cycles: tuple[int, ...], cycle: str = "SDDSRVYR", **extra: Any) -> pd.DataFrame:
    """NHANES-shaped rows of the given release codes: each cycle two strata of two PSUs of three
    rows, a 2-year exam weight on every row, and the 1999–2002 four-year weight on the 1999–2000
    and 2001–2002 rows only (blank in 2003–2004, as DEMO_C carries none). The weights differ row
    by row, so each rule gives its own numbers."""
    rows = []
    for c in cycles:
        for h in (1, 2):
            for p in (1, 2):
                for _ in range(3):
                    j = len(rows)
                    rows.append({"SEQN": j + 1, cycle: c, "SDMVSTRA": 10 * c + h, "SDMVPSU": p,
                                 "WTMEC2YR": 1000.0 * c + 10.0 * j,
                                 "WTMEC4YR": 400.0 + 10.0 * j if c in (1, 2) else np.nan,
                                 "x": float(j % 5), "y": float((3 * j) % 7)})
    frame = pd.DataFrame(rows)
    for name, values in extra.items():
        frame[name] = values(frame)
    frame.index = pd.Index(np.arange(len(frame)), name="row_id")
    return frame


def _by_hand(frame: pd.DataFrame) -> np.ndarray:
    """NCHS's weight for the cycles ``frame`` pools (module docstring), from its columns."""
    cycles = sorted(frame["SDDSRVYR"].unique())
    k = len(cycles)
    w2, w4 = frame["WTMEC2YR"].to_numpy(float), frame["WTMEC4YR"].to_numpy(float)
    if k == 1:
        return w2
    if 1 not in cycles:
        return w2 / k  # §3.1.3
    early = frame["SDDSRVYR"].isin([1, 2]).to_numpy()
    return np.where(early, 2 * w4 / k, w2 / k)  # §3.1.4, Table F (k = 2: the four-year weight)


def _state(frame: pd.DataFrame, roles: dict[str, str] | None = None,
           purpose: str = "inference") -> d.ProjectState:
    roles = {c: r for c, r in (roles or ROLES).items() if c in frame.columns}
    return d.ProjectState(lens=["clinical"], target="y", purpose=purpose, roles=roles)


def _ctx(frame: pd.DataFrame, state: d.ProjectState) -> dict[str, Any]:
    """What the server's ``DecisionContext`` tells the survey question's checks."""
    return {"state": state, "columns": list(frame.columns), "store": lambda: _Store(frame),
            "target": state.target}


def _recorded(frame: pd.DataFrame, decision: dict, state: d.ProjectState) -> Any:
    return d.validate(decision, _ctx(frame, state))


def _fit_weights(frame: pd.DataFrame, recorded: Any, state: d.ProjectState) -> np.ndarray:
    spec = d.SurveySpec(**{k: getattr(recorded, k) for k in (
        "estimand", "weight", "strata", "psu", "cycle", "four_year_weight", "acknowledged")})
    fitted = for_fit(state.model_copy(update={"survey": spec}), _Store(frame))
    assert fitted.refusal is None, fitted.refusal
    return np.asarray(fitted.design.weight, dtype=float)


def _score_weights(frame: pd.DataFrame, recorded: Any, state: d.ProjectState,
                   rows: Any) -> np.ndarray:
    """The weight every prediction stage gives ``rows`` (its training rows) under the answer."""
    spec = d.SurveySpec(**{k: getattr(recorded, k) for k in (
        "estimand", "weight", "strata", "psu", "cycle", "four_year_weight", "acknowledged")})
    designs = fit_designs(state.model_copy(update={"survey": spec}), _Store(frame),
                          np.asarray(rows))
    return designs["weight"].to_numpy(float)


def _said(frame: pd.DataFrame, recorded: Any, state: d.ProjectState) -> str:
    return voice.sentence_for(recorded, state, {"datastore": _Store(frame)})


def _offered(frame: pd.DataFrame, state: d.ProjectState) -> dict:
    [population] = [o["decision"] for o in proposal(state, _Store(frame))["options"]
                    if o["decision"].get("estimand") == "population"]
    return population


# ── SW1: the four-year weight is offered, and said, only where 1999–2000 is pooled ──────────


@pytest.mark.parametrize("cycles,four_year,words", [
    ((1, 2, 3), "WTMEC4YR", FOUR_YEAR_WORDS),
    ((1, 2), "WTMEC4YR", ALONE_WORDS),
    ((2, 3), None, DIVIDED_WORDS),
])
def test_the_offered_answer_names_the_four_year_weight_and_says_the_weight_the_fit_takes(
        cycles, four_year, words):
    """The survey question's population option on each pool, recorded: it names ``WTMEC4YR``
    where 1999–2000 is pooled and not on 2001–2004 (the base named it there); the fit weighs the
    rows by NCHS's weight for those cycles, by hand; the sentence says that weight (on 2001–2004
    the base said the 1999–2002 rows took the four-year weight, which no row does; on 1999–2002
    alone it said "the others" took ``WTMEC2YR``, and there are none)."""
    frame = _pool(cycles)
    state = _state(frame)
    option = _offered(frame, state)
    assert (option["cycle"], option["four_year_weight"]) == ("SDDSRVYR", four_year)
    recorded = _recorded(frame, option, state)
    np.testing.assert_allclose(_fit_weights(frame, recorded, state), _by_hand(frame), rtol=1e-12)
    said = _said(frame, recorded, state)
    assert f"rows were {words}, and standard errors" in said, said
    if 1 not in cycles:
        assert "WTMEC4YR" not in said and "1999" not in said


def test_a_2001_2004_answer_naming_the_four_year_weight_says_the_2_year_weight_it_is_fit_on():
    """The verifier's served case (pp3-survey-columns, round 2): 2001–2004 answered with
    ``SDDSRVYR`` and ``WTMEC4YR``. It is accepted, and fit on ``WTMEC2YR`` ÷ 2 (by hand); the
    sentence says so, under either goal (under prediction counting the training rows' cycles),
    never the four-year rule (the base's)."""
    frame = _pool((2, 3))
    for purpose, words in (("inference", DIVIDED_WORDS), ("prediction", SCORED_DIVIDED_WORDS)):
        state = _state(frame, purpose=purpose)
        recorded = _recorded(frame, {**DESIGN, "cycle": "SDDSRVYR",
                                     "four_year_weight": "WTMEC4YR"}, state)
        assert recorded.four_year_weight == "WTMEC4YR"
        np.testing.assert_allclose(_fit_weights(frame, recorded, state),
                                   frame["WTMEC2YR"].to_numpy(float) / 2, rtol=1e-12)
        said = _said(frame, recorded, state)
        assert words in said and "WTMEC4YR" not in said, said


def test_one_cycle_named_as_the_cycle_is_weighted_and_said_as_it_is():
    """One cycle (2003–2004) answered with its cycle column named: the fit takes ``WTMEC2YR`` as
    it is (by hand, one cycle's own weight), and the sentence says no division (the base said it
    was divided by the number of cycles, one)."""
    frame = _pool((3,))
    state = _state(frame)
    recorded = _recorded(frame, {**DESIGN, "cycle": "SDDSRVYR"}, state)
    np.testing.assert_allclose(_fit_weights(frame, recorded, state), _by_hand(frame), rtol=1e-12)
    said = _said(frame, recorded, state)
    assert "rows were weighted by `WTMEC2YR`, and standard errors" in said, said


def test_without_the_table_the_sentence_states_the_rule_not_a_branch():
    """With no table to read (a sentence authored with no context), a pooled answer naming the
    four-year weight is said by the rule, which holds whichever cycles are pooled, never as the
    four-year branch; with no cycle named, the weight as it is, which ``analysis_weights`` takes
    whatever the table holds."""
    pooled = d.parse_decision({**DESIGN, "cycle": "SDDSRVYR", "four_year_weight": "WTMEC4YR"})
    said = voice.sentence_for(pooled, _state(_pool((2, 3))))
    assert ("rows were weighted by `WTMEC2YR` (or, on the 1999–2002 rows where 1999–2000 is "
            "pooled, `WTMEC4YR`), pooled over the cycles in `SDDSRVYR` by NCHS's rule") in said
    assert "as NCHS directs for 1999–2000" not in said
    plain = voice.sentence_for(d.parse_decision(DESIGN), _state(_pool((2, 3))))
    assert "rows were weighted by `WTMEC2YR`, and standard errors" in plain, plain


# ── SW1 under prediction: the scores weigh the training rows, not the working table ──────────


def _outcome_blank_in(frame: pd.DataFrame, cycle: int) -> pd.DataFrame:
    """``frame`` with the outcome not measured on any row of release ``cycle``."""
    out = frame.copy()
    out.loc[out["SDDSRVYR"] == cycle, "y"] = np.nan
    return out


@pytest.mark.parametrize("cycles,words", [
    ((1, 2, 3), SCORED_DIVIDED_WORDS),
    ((1, 2), SCORED_AS_IT_IS_WORDS),
])
def test_under_prediction_a_cycle_without_the_outcome_is_not_said_to_weigh_the_scores(
        cycles, words):
    """The verifier's case (PP2 follow-up, repair round 1, P1): 1999–2000's outcome not
    measured, under prediction, answered with the survey question's own option (``SDDSRVYR``
    and ``WTMEC4YR``, 1999–2000 being in the table). The scores' rows are the training rows,
    among the rows with the outcome recorded (the cohort), so 1999–2000 is not among them: by
    hand, ``WTMEC2YR`` ÷ 2 on 2001–2004 (§3.1.3), and ``WTMEC2YR`` as it is on 2001–2002 alone.
    The sentence says that weight; the package's first build said the four-year weight, which no
    score row takes."""
    frame = _outcome_blank_in(_pool(cycles), 1)
    state = _state(frame, purpose="prediction")
    option = _offered(frame, state)
    assert (option["cycle"], option["four_year_weight"]) == ("SDDSRVYR", "WTMEC4YR")
    recorded = _recorded(frame, option, state)
    rows = frame.index[frame["y"].notna()]
    np.testing.assert_allclose(_score_weights(frame, recorded, state, rows),
                               _by_hand(frame.loc[rows]), rtol=1e-12)
    said = _said(frame, recorded, state)
    assert words in said and "WTMEC4YR" not in said and "1999" not in said, said


def test_under_prediction_a_pool_holding_1999_2000_is_said_by_the_rule_its_training_rows_take():
    """1999–2004 under prediction, the outcome recorded in every cycle: which branch of NCHS's
    rule the scores take is the training rows' (the split and exclusions, drawn after the survey
    question, keep or drop whole cycles), so the sentence states the rule, never one branch.
    Each set of training rows here takes its own weight, by hand: all three cycles Table F's
    6-year weight; 1999–2002 (a cut-point at 2003–2004) the four-year weight itself; 2001–2004
    (1999–2000 excluded) ``WTMEC2YR`` ÷ 2; 1999–2000 alone its own 2-year weight."""
    frame = _pool((1, 2, 3))
    state = _state(frame, purpose="prediction")
    recorded = _recorded(frame, _offered(frame, state), state)
    for kept in ((1, 2, 3), (1, 2), (2, 3), (1,)):
        rows = frame.index[frame["SDDSRVYR"].isin(kept)]
        np.testing.assert_allclose(_score_weights(frame, recorded, state, rows),
                                   _by_hand(frame.loc[rows]), rtol=1e-12)
    said = _said(frame, recorded, state)
    assert SCORED_RULE_WORDS in said, said
    assert "as NCHS directs for 1999–2000" not in said and "provides for 1999–2002" not in said


def test_under_inference_a_cycle_without_the_outcome_stays_in_the_design():
    """The same table under inference is kept as it is: the fit reads every row of the working
    table (a domain analysis, ``methods.survey.for_fit``), so 1999–2000 is pooled and the rows
    take Table F's 6-year weight over the three cycles (by hand), and the sentence says the
    four-year branch."""
    frame = _outcome_blank_in(_pool((1, 2, 3)), 1)
    state = _state(frame)
    recorded = _recorded(frame, _offered(frame, state), state)
    np.testing.assert_allclose(_fit_weights(frame, recorded, state), _by_hand(frame), rtol=1e-12)
    said = _said(frame, recorded, state)
    assert f"rows were {FOUR_YEAR_WORDS}, and standard errors" in said, said


# ── SW2: a pool answered with no cycle ──────────────────────────────────────────────────────


def test_a_pool_answered_with_no_cycle_is_pooled_by_the_cycle_nchs_names():
    """2001–2004 answered with ``WTMEC2YR`` and no cycle: ``SDDSRVYR`` is NCHS's own release
    code and the only column that pools, so the answer is recorded pooled by it, and the fit
    takes ``WTMEC2YR`` ÷ 2 (by hand; the base took ``WTMEC2YR`` as it is, twice the
    population), its sentence saying the division. Under prediction too, where the scores are
    weighted by the same weight."""
    frame = _pool((2, 3))
    for purpose, words in (("inference", DIVIDED_WORDS), ("prediction", SCORED_DIVIDED_WORDS)):
        state = _state(frame, purpose=purpose)
        recorded = _recorded(frame, DESIGN, state)
        assert recorded.cycle == "SDDSRVYR"
        np.testing.assert_allclose(_fit_weights(frame, recorded, state), _by_hand(frame),
                                   rtol=1e-12)
        assert words in _said(frame, recorded, state)


@pytest.mark.parametrize("case", ["one_cycle", "pooled_weight"])
def test_an_answer_with_no_cycle_is_kept_where_its_weight_is_right_as_it_is(case):
    """Where no division is due the answer is kept as given, the weight as it is, and said so:
    one cycle in the table (one cycle's own weight, by hand); a weight NCHS does not publish as
    one cycle's (``WTMEC6YR``, here a weight the researcher pooled already, ``WTMEC2YR`` ÷ 2 over
    the two cycles by construction: dividing it again would describe half the population)."""
    if case == "one_cycle":
        frame, weight, roles = _pool((3,)), "WTMEC2YR", ROLES
    else:
        frame = _pool((2, 3), WTMEC6YR=lambda f: f["WTMEC2YR"] / 2)
        weight, roles = "WTMEC6YR", {**ROLES, "WTMEC6YR": "design"}
    state = _state(frame, roles)
    recorded = _recorded(frame, {**DESIGN, "weight": weight}, state)
    assert recorded.cycle is None
    np.testing.assert_allclose(_fit_weights(frame, recorded, state), _by_hand_weight(frame, case),
                               rtol=1e-12)
    said = _said(frame, recorded, state)
    assert f"rows were weighted by `{weight}`, and standard errors" in said, said


def _by_hand_weight(frame: pd.DataFrame, case: str) -> np.ndarray:
    """One cycle's own 2-year weight, or the researcher's pooled weight as constructed."""
    if case == "one_cycle":
        return _by_hand(frame)
    return frame["WTMEC2YR"].to_numpy(float) / 2


@pytest.mark.parametrize("role", ["covariate", "exposure"])
def test_a_cycle_the_roles_make_a_predictor_still_pools_the_weight(role):
    """PP2 follow-up, repair round 1, P3: ``SDDSRVYR`` adjusted for (a covariate, or the exposure
    of a trend), its role settled, over 2001–2004. The package's first build kept the answer with
    no cycle and fit ``WTMEC2YR`` as it is, twice the population, and refused the survey
    question's own option naming the cycle (``design_is_predictor``), so no answer reached NCHS's
    weight. The cycle says only which weight each row takes, which a predictor's use of the same
    column does not contradict: the answer with no cycle is completed with it, and the option is
    accepted, each fit on ``WTMEC2YR`` ÷ 2 (by hand, §3.1.3) and said so. With 1999–2000 pooled
    the option's four-year weight is taken too (Table F, by hand)."""
    roles = {**ROLES, "SDDSRVYR": role}
    frame = _pool((2, 3))
    state = _state(frame, roles)
    recorded = _recorded(frame, DESIGN, state)
    assert recorded.cycle == "SDDSRVYR"
    for answer in (recorded, _recorded(frame, _offered(frame, state), state)):
        np.testing.assert_allclose(_fit_weights(frame, answer, state), _by_hand(frame),
                                   rtol=1e-12)
        assert f"rows were {DIVIDED_WORDS}, and standard errors" in _said(frame, answer, state)
    frame = _pool((1, 2, 3))
    state = _state(frame, roles)
    option = _offered(frame, state)
    assert (option["cycle"], option["four_year_weight"]) == ("SDDSRVYR", "WTMEC4YR")
    recorded = _recorded(frame, option, state)
    np.testing.assert_allclose(_fit_weights(frame, recorded, state), _by_hand(frame), rtol=1e-12)
    assert f"rows were {FOUR_YEAR_WORDS}, and standard errors" in _said(frame, recorded, state)


@pytest.mark.parametrize("slot,column,code", [
    ("weight", "WTMEC2YR", "design_is_predictor"),
    ("strata", "SDMVSTRA", "design_is_predictor"),
    ("psu", "SDMVPSU", "design_is_predictor"),
    ("cycle", "y", "design_is_outcome"),
])
def test_a_weight_strata_or_psu_that_is_a_predictor_and_a_cycle_that_is_the_outcome_are_refused(
        slot, column, code):
    """The guard the cycle leaves still holds where it describes the sampling: a weight, strata
    or PSU the roles make a predictor, and a cycle that is the outcome, are refused."""
    frame = _pool((2, 3))
    roles = {**ROLES, column: "covariate"} if code == "design_is_predictor" else ROLES
    state = _state(frame, roles)
    with pytest.raises(d.Refusal) as refused:
        _recorded(frame, {**DESIGN, "cycle": "SDDSRVYR", slot: column}, state)
    assert refused.value.code == code


def test_a_pool_whose_cycle_the_reading_does_not_certainly_read_is_refused_with_its_ways_on():
    """2001–2004 under a cycle column the cycle reader takes by its name but NCHS does not use
    (``survey_cycle``), answered with no cycle: refused, saying the 2 values and the division
    (interpolated from the column), with the ways on: pool by ``survey_cycle`` (which records and
    fits ``WTMEC2YR`` ÷ 2, by hand) or these participants. Two columns that each pool are refused
    too, each named."""
    frame = _pool((2, 3), cycle="survey_cycle")
    roles = {**{k: v for k, v in ROLES.items() if k != "SDDSRVYR"}, "survey_cycle": "time"}
    state = _state(frame, roles)
    with pytest.raises(d.Refusal) as refused:
        _recorded(frame, DESIGN, state)
    assert refused.value.code == "name_the_cycle"
    assert refused.value.message == (
        "`survey_cycle` reads as a survey cycle and holds 2 values, and `WTMEC2YR` is one cycle's "
        "two-year weight: pooled over 2 cycles it is divided by 2 (NHANES Analytic Guidelines "
        "2011–2016 §3.1.3). Name the cycle column, or estimate for these participants.")
    pool, sample = refused.value.exits
    assert pool["label"] == "Pool by `survey_cycle`"
    assert (sample["decision"]["kind"], sample["decision"]["estimand"]) == ("set_survey", "sample")
    recorded = _recorded(frame, pool["decision"], state)
    assert recorded.cycle == "survey_cycle"
    np.testing.assert_allclose(_fit_weights(frame, recorded, state),
                               frame["WTMEC2YR"].to_numpy(float) / 2, rtol=1e-12)

    both = _pool((2, 3), survey_cycle=lambda f: f["SDDSRVYR"])
    state = _state(both, {**ROLES, "survey_cycle": "time"})
    with pytest.raises(d.Refusal) as refused:
        _recorded(both, DESIGN, state)
    assert refused.value.code == "name_the_cycle"
    assert [e["label"] for e in refused.value.exits] == [
        "Pool by `SDDSRVYR`", "Pool by `survey_cycle`", "Estimate for these participants instead"]


def test_an_nchs_cycle_whose_role_rode_along_unconfirmed_is_named_by_the_user_not_filled_in():
    """BLUEPRINT §14 rule 2: ``SDDSRVYR``'s role carried by a bulk confirm below high and not
    confirmed since is not read to change a number, so the answer with no cycle is refused with
    the way on that names it (the user records it); recorded so, the fit takes ``WTMEC2YR`` ÷ 2
    (by hand)."""
    frame = _pool((2, 3))
    state = _state(frame).model_copy(update={"roles_unconfirmed": ["SDDSRVYR"]})
    with pytest.raises(d.Refusal) as refused:
        _recorded(frame, DESIGN, state)
    assert refused.value.code == "name_the_cycle"
    assert refused.value.message.startswith("`SDDSRVYR` reads as a survey cycle and holds 2 values")
    pool = refused.value.exits[0]
    assert pool["label"] == "Pool by `SDDSRVYR`"
    recorded = _recorded(frame, pool["decision"], state)
    np.testing.assert_allclose(_fit_weights(frame, recorded, state), _by_hand(frame), rtol=1e-12)

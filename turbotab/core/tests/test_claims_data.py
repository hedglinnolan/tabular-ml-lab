"""The data area's claims (``turbotab/core/claims/data.py``; R0_PLAN.md R0b): the Silent tests
stage 1 left missing, the share above a UL for a name read as no nutrient, the Subsample C
weight read by its name and by where its variable is recorded (and a word that begins like it
read as no weight), the NHANES subsample tolerances both ways, the omics reading's floor, span,
skewness, names, distinct-values and per-million tests both ways, the grouping entry's number
held to its guard, and the regime predicates tested both ways.

A Silent test holds a guard quiet where its evidence says it must be: a UL that covers every
source, a name read as calcium, an NHANES table with no variable measured on a subsample alone, a
block of two-level columns, sample totals near a million, a yes/no outcome in groups large enough
for an intercept each. Each was watched failing on a copy of the base whose guard was loosened.
Its truth is the construction or an independent library, never the claim it tests: this file
names no claim and none of the constants the claims set (meta-test 2), and reads no threshold of
TurboTab's own, so every case sits far from the cut-off it would cross. The per-million tests read
the genomics pack's tolerance, a copy the claim does not set, and sit a tenth of it to either
side; the one test that crosses a cut-off finds it from the guard's behavior, to hold a sentence
to it.
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


def _methods(label: str, cutoff: float) -> str:
    """The usual-intake methods sentence for a share above a UL of ``cutoff`` on the component
    ``label``, its UL's form read from the name as the stage reads it."""
    from turbotab.core import usual_intake as ui
    from turbotab.core.methods import usual_intake as nci

    return nci.methods_sentence(
        label=label, model="amount_only", names=["intercept"], lam=0.4, n_persons=500,
        recalls={1: 200, 2: 300}, n_repeat=300, zeros_replaced=0, population="whole",
        consumer_column=None, weight=None, replication=None, n_ok=0, cutoff=cutoff,
        cutoff_kind="UL", ul_form=ui.ul_form(label))


def test_a_ul_share_on_a_name_read_as_no_nutrient_is_a_share_of_the_distribution():
    """Claims backtest BX-040, review: the share above a UL was worded as a UL covering food, water
    and supplements wherever the name read as no nutrient whose UL covers one form. NHANES's
    supplement-recall magnesium total, DS1TMAGN, is read as no nutrient at all, and the IOM DRI
    tables' footnote says magnesium's UL ("intake from a pharmacological agent only and do not
    include intake from food and water") covers no food or water. Expected, for DS1TMAGN and a
    bare ``Mg`` against magnesium's 350 mg UL: the share is reported as a share of the
    distribution, and what a UL covers is said to depend on the nutrient, magnesium's covering
    supplements and medicines; nothing says the UL covers food, water and supplements, or that a
    share is a lower bound. The UL is stated."""
    from turbotab.core.methods import usual_intake as nci

    def unread(sentence: str) -> None:
        assert "share of the distribution" in sentence, sentence
        assert "depends on the nutrient" in sentence, sentence
        assert "supplements and medicines" in sentence, sentence
        assert "350" in sentence, sentence
        assert "food, water and supplements" not in sentence, sentence
        assert "lower bound" not in sentence and "at risk of excess where" not in sentence

    for label in ("DS1TMAGN", "Mg"):
        unread(_methods(label, 350))
    unread(nci.share_phrase(350, "UL", nutrient_read=False))


def _usual_intake_decision(cutoff) -> dict:
    return {"kind": "set_usual_intake", "nutrient": "DS1TMAGN", "model": "amount_only",
            "cutoff": cutoff, "cutoff_kind": "UL"}


@pytest.mark.parametrize("cutoff", [float("inf"), float("nan")], ids=["inf", "nan"])
def test_a_non_finite_cut_off_is_refused_at_the_decision(cutoff):
    """The verifier's V2 on claims backtest BX-040: a share above an infinite UL reached the
    unread-name claim's card with a cut-off outside its regime (a finite positive number) and
    raised there. A UL is a number of milligrams or micrograms, so a cut-off that is no finite
    number (infinity, or not a number) is refused where it is set, for DS1TMAGN, whether it comes
    as a value or as JSON's Infinity and NaN."""
    from pydantic import ValidationError

    from turbotab.core.decisions import SetUsualIntake, parse_decision

    with pytest.raises(ValidationError):
        parse_decision(_usual_intake_decision(cutoff))
    text = "Infinity" if cutoff == float("inf") else "NaN"
    with pytest.raises(ValidationError):
        SetUsualIntake.model_validate_json(
            '{"nutrient": "DS1TMAGN", "model": "amount_only", "cutoff": %s, "cutoff_kind": "UL"}'
            % text)


def test_a_finite_cut_off_is_accepted_at_the_decision():
    """The silent side: magnesium's 350 mg UL for DS1TMAGN is set as given."""
    from turbotab.core.decisions import parse_decision

    assert parse_decision(_usual_intake_decision(350.0)).cutoff == 350.0


def test_a_share_above_an_infinite_ul_never_reaches_the_card(monkeypatch):
    """share_phrase never hands a cut-off that is no finite number to card(): for DS1TMAGN, a
    name read as no nutrient, a share above an infinite UL is refused with a plain error before
    any card is rendered, in the methods sentence and in the share's own; magnesium's 350 mg UL
    renders the card once, with that cut-off."""
    from turbotab.core import claims as registry
    from turbotab.core.claims import ClaimError
    from turbotab.core.methods import usual_intake as nci

    seen = []
    card = registry.card

    def spied(id, ctx=None, strict=None):
        seen.append(dict(ctx or {}).get("cutoff"))
        return card(id, ctx, strict)

    monkeypatch.setattr(registry, "card", spied)
    for render in (lambda: _methods("DS1TMAGN", float("inf")),
                   lambda: nci.share_phrase(float("inf"), "UL", nutrient_read=False)):
        with pytest.raises(ValueError) as r:
            render()
        assert not isinstance(r.value, ClaimError), repr(r.value)
    assert seen == []
    assert "350" in _methods("DS1TMAGN", 350.0)
    assert seen == [350.0]


# The share's sentence for a name read as calcium against its 2,500 mg UL, as the base wrote it:
# the IOM DRI tables' calcium UL covers intake from food, water and supplements.
CALCIUM_SHARE = (" The share above the UL (2,500) is a share at risk of excess where the recalls "
                 "include supplements, the UL covering intake from food, water and supplements "
                 "(Institute of Medicine 2000a); from foods alone it is a lower bound on that "
                 "share.")


def test_a_ul_share_on_a_name_read_as_calcium_keeps_its_sentence():
    """Claims backtest BX-040, the silent side of the sentence for a name read as no nutrient: a
    name read as calcium (``calcium_mg``, NHANES's ``DR1TCALC``), whose UL covers every source and
    covers no one form, keeps the total-intake sentence byte for byte, in the methods sentence and
    in the share's own."""
    from turbotab.core.methods import usual_intake as nci

    for label in ("calcium_mg", "DR1TCALC"):
        assert _methods(label, 2500).endswith(CALCIUM_SHARE), _methods(label, 2500)
    assert nci.share_phrase(2500, "UL") == CALCIUM_SHARE


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


def test_a_variable_recorded_only_where_the_subsample_c_weight_is_positive_names_it():
    """Claims backtest C-143, the fires side for a subsample the names cannot place: NCHS's
    Subsample C weight, WTSC2YR (``survey.NHANES_WEIGHTS`` lists it), positive on a third of the
    examined, 0 on the other examined rows and blank off the examination, as a subsample file's
    weight is. A urinary analyte recorded by construction on exactly those rows is that
    subsample's whatever its name, so the rule names WTSC2YR for it, never the examination weight
    as settled; and beside BMI alone, which nothing restricts to its rows, the subsample weight is
    one whose analytes TurboTab cannot tell, so the choice is the user's."""
    from turbotab.core.recognizers import least_common_denominator

    frame = _merge()
    rng = np.random.default_rng(1432)
    examined = frame["WTMEC2YR"] > 0
    subsample = examined & (rng.random(len(frame)) < 1 / 3)
    frame["WTSC2YR"] = np.where(subsample, rng.uniform(5000, 150000, len(frame)),
                                np.where(examined, 0.0, np.nan))
    frame["urinary_analyte"] = np.where(subsample, rng.lognormal(1.0, 0.8, len(frame)), np.nan)
    assert (frame["urinary_analyte"].notna() == (frame["WTSC2YR"] > 0)).all()
    assert subsample.sum() >= 100 and (examined & ~subsample).sum() >= 100
    lcd = least_common_denominator(["urinary_analyte", "RIDAGEYR", "WTINT2YR", "WTMEC2YR",
                                    "WTSC2YR"], frame)
    assert lcd["use"] == "WTSC2YR" and lcd["because"] == ["urinary_analyte"], lcd
    alone = least_common_denominator(["BMXBMI", "RIDAGEYR", "WTMEC2YR", "WTSC2YR"], frame)
    assert alone["other_subsamples"] == ["WTSC2YR"] and not alone["settled"], alone


def _summaries(frame: pd.DataFrame) -> list[dict]:
    """The profile's column summaries, as the roles proposal reads them."""
    return [{"name": c, "dtype": "integer" if frame[c].dtype.kind == "i" else "numeric",
             "n_unique": int(frame[c].nunique()), "n_missing": int(frame[c].isna().sum()),
             "min": float(frame[c].min()), "median": float(frame[c].median())} for c in frame]


def test_a_word_that_begins_like_subsample_c_s_weight_is_no_sampling_weight():
    """Claims backtest C-143 and audit IN-10, the silent side of reading Subsample C's weight by
    its name: a clinical table with no survey design (no strata, no sampling units), whose columns
    begin with Subsample C's letters only as words do. By construction ``wtscale`` is a body
    weight read off a scale (45 to 120 kg), ``wtscore`` a score (0.1 to 10) and ``WTSCHOOL`` a
    child's weight at school (15 to 60 kg); none is a weight as NCHS names one, ``WTS``, the
    subsample's letters and its cycle (WTSC2YR). Expected: none reads as a sampling weight, the
    roles proposal proposes none as part of a survey design, and the survey question offers no
    weight."""
    from turbotab.core import survey
    from turbotab.core.recognizers import reads_as_survey_weight
    from turbotab.core.stages import rows

    rng = np.random.default_rng(1433)
    n = 200
    frame = pd.DataFrame({"id": np.arange(n), "age": rng.integers(20, 80, n),
                          "wtscale": rng.uniform(45, 120, n).round(1),
                          "wtscore": rng.uniform(0.1, 10, n).round(2),
                          "WTSCHOOL": rng.uniform(15, 60, n).round(1),
                          "sbp": rng.normal(120, 15, n).round(0)})
    words = ("wtscale", "wtscore", "WTSCHOOL")
    for name in words:
        assert not reads_as_survey_weight(name, median=float(frame[name].median())), name
    proposals = {p["column"]: p for p in rows.propose_roles(_summaries(frame), lens=["clinical"],
                                                            target="sbp", n_rows=n)}
    for name in words:
        assert proposals[name]["proposed"] != "design", proposals[name]
        assert "sampling weight" not in proposals[name]["reason"], proposals[name]
    assert survey.read_design(list(frame.columns), target="sbp").weights == []


def test_subsample_c_s_weight_beside_the_design_is_named_as_nchs_names_it():
    """Claims backtest C-143, the fires side of the roles proposal's name test: WTSC2YR, Subsample
    C's weight as NCHS names it (``survey.NHANES_WEIGHTS`` lists it), beside the survey's own
    strata and sampling units (SDMVSTRA, SDMVPSU, whole-number codes) and the examination weight,
    its values positive weights on a third of the examined rows. Expected: it is proposed as part
    of the design, its values read and its name read as a published design variable, so its reason
    never says its name is none."""
    from turbotab.core.stages import rows

    frame = _merge()
    rng = np.random.default_rng(1434)
    examined = frame["WTMEC2YR"] > 0
    frame["WTSC2YR"] = np.where(examined & (rng.random(len(frame)) < 1 / 3),
                                rng.uniform(5000, 150000, len(frame)), 0.0).round(1)
    frame["SDMVSTRA"] = rng.integers(173, 188, len(frame))
    frame["SDMVPSU"] = rng.integers(1, 3, len(frame))
    frame = frame[["SEQN", "RIDAGEYR", "BMXBMI", "WTMEC2YR", "WTSC2YR", "SDMVSTRA", "SDMVPSU",
                   "DIQ010"]]
    facts = {c: {"design_values": rows.design_values_fit(c, frame[c])}
             for c in ("WTMEC2YR", "WTSC2YR", "SDMVSTRA", "SDMVPSU")}
    assert all(f["design_values"] for f in facts.values()), facts
    proposals = {p["column"]: p for p in rows.propose_roles(
        _summaries(frame), lens=["clinical"], target="DIQ010", n_rows=len(frame), facts=facts)}
    weight = proposals["WTSC2YR"]
    assert weight["proposed"] == "design", weight
    assert "no published design variable" not in weight["reason"], weight
    assert "Confirm it is part of the design" in weight["reason"], weight


# ── C-143 · the subsample tolerances, both ways ──────────────────────────────────────────────


def test_a_variable_on_four_rows_inside_the_fasting_subsample_is_not_that_subsample_s():
    """Claims backtest C-143, the silent side of the subsample's row floor: a variable recorded on
    four rows only, all of them by construction inside the fasting subsample (WTSAF2YR positive),
    four of the some 190 fasting rows. Four rows cannot show where a variable was measured, so it
    is no fasting variable: the rule does not read it as the subsample's, and names WTMEC2YR."""
    from turbotab.core.recognizers import least_common_denominator, subsample_variables

    frame = _merge()
    morning = np.flatnonzero((frame["WTSAF2YR"] > 0).to_numpy())
    assert len(morning) >= 100
    frame["analyte_q"] = np.nan
    frame.loc[morning[:4], "analyte_q"] = [1.2, 3.4, 0.8, 2.2]
    assert frame["analyte_q"].notna().sum() == 4
    assert (frame["analyte_q"].notna() <= (frame["WTSAF2YR"] > 0)).all()
    assert "analyte_q" not in subsample_variables(frame, "WTSAF2YR")
    lcd = least_common_denominator(["analyte_q", "RIDAGEYR", "WTMEC2YR", "WTSAF2YR"], frame)
    assert lcd["use"] == "WTMEC2YR", lcd


def test_a_variable_on_nearly_every_row_is_no_subsample_s_where_the_weight_covers_nearly_all():
    """Claims backtest C-143, the silent side of the subsample's share of blanks: a table of
    examined participants alone, 600 rows, every one but six with blood drawn (WTPH2YR positive,
    0 on the six), and BMI, measured at the examination, blank on those same six rows by
    construction (they left before both). BMI is then recorded only where the phlebotomy weight
    is positive, but it is blank on 1% of the table, too few blanks to mark a subsample: it is no
    phlebotomy variable, and the rule keeps WTMEC2YR for it."""
    from turbotab.core.recognizers import least_common_denominator, subsample_variables

    rng = np.random.default_rng(1435)
    n = 600
    left = np.zeros(n, dtype=bool)
    left[rng.choice(n, 6, replace=False)] = True
    frame = pd.DataFrame({
        "SEQN": np.arange(130000, 130000 + n), "RIDAGEYR": rng.integers(18, 80, n),
        "WTMEC2YR": rng.uniform(3000, 90000, n).round(1),
        "WTPH2YR": np.where(left, 0.0, rng.uniform(3000, 90000, n)).round(1),
        "BMXBMI": np.where(left, np.nan, rng.normal(28, 6, n).round(1))})
    assert (frame["BMXBMI"].notna() == (frame["WTPH2YR"] > 0)).all()
    assert frame["BMXBMI"].isna().sum() == 6
    assert "BMXBMI" not in subsample_variables(frame, "WTPH2YR")
    lcd = least_common_denominator(["BMXBMI", "RIDAGEYR", "WTMEC2YR", "WTPH2YR"], frame)
    assert lcd["use"] == "WTMEC2YR", lcd


def test_a_subsample_variable_with_a_stray_row_outside_it_is_still_that_subsample_s():
    """Claims backtest C-143, the fires side of the subsample's slack: a fasting analyte recorded
    on every row of the fasting subsample (WTSAF2YR positive) and, by construction, on one row
    outside it, a merge's stray row, so one of its some 480 rows (about 0.2%) lies outside the
    subsample. It is the subsample's: the rule reads it as a fasting variable whatever its name,
    and names WTSAF2YR for it."""
    from turbotab.core.recognizers import least_common_denominator, subsample_variables

    frame = _merge(n=2000)
    rng = np.random.default_rng(1436)
    morning = (frame["WTSAF2YR"] > 0).to_numpy()
    frame["analyte_q"] = np.where(morning, rng.lognormal(1.0, 0.5, len(frame)), np.nan)
    stray = np.flatnonzero((frame["WTMEC2YR"] > 0).to_numpy() & ~morning)[0]
    frame.loc[stray, "analyte_q"] = 2.5
    recorded = frame["analyte_q"].notna()
    assert recorded.sum() >= 400 and (recorded & ~morning).sum() == 1
    assert "analyte_q" in subsample_variables(frame, "WTSAF2YR")
    lcd = least_common_denominator(["analyte_q", "RIDAGEYR", "WTMEC2YR", "WTSAF2YR"], frame)
    assert lcd["use"] == "WTSAF2YR" and lcd["because"] == ["analyte_q"], lcd


def test_a_table_with_five_rows_outside_the_examination_keeps_the_mec_weight():
    """Claims backtest C-143, the silent side of the interview weight's row floor: 300 examined
    participants and, by construction, five interviewed and not examined (WTMEC2YR 0), with
    diabetes diagnosed and age recorded on all 305. Five rows outside the examined sample do not
    make the table the interview sample, so the rule keeps WTMEC2YR; the whole merge, a quarter of
    its rows outside the examination, takes WTINT2YR for the same two variables."""
    from turbotab.core.recognizers import least_common_denominator

    frame = _merge()
    examined = (frame["WTMEC2YR"] > 0).to_numpy()
    keep = np.concatenate([np.flatnonzero(examined)[:300], np.flatnonzero(~examined)[:5]])
    small = frame.iloc[keep].reset_index(drop=True)
    assert (small["WTMEC2YR"] == 0).sum() == 5 and small[["DIQ010", "RIDAGEYR"]].notna().all().all()
    interview = ["DIQ010", "RIDAGEYR", "WTINT2YR", "WTMEC2YR"]
    assert least_common_denominator(interview, small)["use"] == "WTMEC2YR"
    assert least_common_denominator(interview, frame)["use"] == "WTINT2YR"


def test_the_weight_card_states_the_rows_at_which_the_interview_reading_turns():
    """The verifier's V3 on claims backtest C-143, the card against the guard: the card of each
    claim the interview reading enforces (``recognizers._interview_only`` among its enforcers)
    states how many interviewed participants who were not examined a table needs before its
    variables take the interview weight. The number is read from the rule's behavior, never
    from the claim: 300 examined participants and k not examined (WTMEC2YR 0), diabetes
    diagnosed and age recorded on all, for k from 1 to 40; the smallest k that takes WTINT2YR is
    the number, every smaller k keeps WTMEC2YR, and the card states it."""
    from turbotab.core import claims as registry
    from turbotab.core.recognizers import least_common_denominator

    frame = _merge()
    examined = np.flatnonzero((frame["WTMEC2YR"] > 0).to_numpy())[:300]
    outside = np.flatnonzero(~(frame["WTMEC2YR"] > 0).to_numpy())
    interview = ["DIQ010", "RIDAGEYR", "WTINT2YR", "WTMEC2YR"]
    used = {}
    for k in range(1, 41):
        small = frame.iloc[np.concatenate([examined, outside[:k]])].reset_index(drop=True)
        used[k] = least_common_denominator(interview, small)["use"]
    turns = [k for k, w in used.items() if w == "WTINT2YR"]
    assert turns and all(used[k] == "WTMEC2YR" for k in range(1, turns[0])), used
    enforced = [c.id for c in registry.registry()
                if "turbotab.core.recognizers:_interview_only" in c.enforced_by]
    assert enforced
    for id_ in enforced:
        card = registry.card(id_, {"columns": list(frame.columns)})
        assert f"at least {turns[0]} of them" in card, card


# ── C-193 · a column of two values is no assay reading ───────────────────────────────────────


def _levels(rng: np.random.Generator, values: list[int], n: int = 60) -> np.ndarray:
    """A column holding each of ``values`` at least once, the rest drawn from them."""
    return rng.permutation(np.concatenate([values, rng.choice(values, n - len(values))]))


def test_columns_of_three_values_or_more_are_read_as_counts_and_two_level_ones_are_left_out():
    """Claims backtest C-193, the fires side of the distinct-values test: under the genomics lens,
    whole numbers reaching the hundreds in columns of three values or more (0, 7 and 500) are raw
    counts, and the reading names those columns alone: beside 30 Poisson(200) gene counts, eight
    columns of 0 and 500 only (an indicator, however large its code) are outside the block it
    names."""
    from turbotab.core.methods import omics

    rng = np.random.default_rng(193)
    three = pd.DataFrame({f"g{i}": _levels(rng, [0, 7, 500]) for i in range(30)})
    reading = omics.scale_reading(three, ["genomics"])
    assert reading is not None and reading.kind == "counts", reading
    assert reading.columns == tuple(three.columns)
    genes = {f"g{i}": rng.poisson(200, 60) for i in range(30)}
    flags = {f"flag{i}": _levels(rng, [0, 500]) for i in range(8)}
    mixed = pd.DataFrame({**genes, **flags})
    reading = omics.scale_reading(mixed, ["genomics"])
    assert reading is not None and reading.kind == "counts", reading
    assert reading.columns == tuple(genes), reading.columns


def test_a_block_of_two_level_columns_is_never_read_as_raw_counts():
    """Claims backtest C-193, the silent side of the distinct-values test: 30 columns that each
    hold 0 and 500 only, under the genomics lens. Two values make an indicator, however large the
    code, so no column is an assay reading and nothing is read as raw counts."""
    from turbotab.core.methods import omics

    rng = np.random.default_rng(1931)
    frame = pd.DataFrame({f"g{i}": _levels(rng, [0, 500]) for i in range(30)})
    assert all(frame[c].nunique() == 2 for c in frame) and frame.to_numpy().max() == 500
    assert omics.scale_reading(frame, ["genomics"]) is None


def _panel(draw, k: int = 25, n: int = 60) -> pd.DataFrame:
    """``k`` assay columns of ``n`` samples, each drawn by ``draw(size)``."""
    return pd.DataFrame({f"m{j}": draw(n) for j in range(k)})


def test_an_identifier_and_an_age_beside_the_assay_columns_stay_out_of_the_block():
    """Claims backtest C-193, the fires side of the names the block leaves out: an NHANES
    participant number (SEQN, whole numbers from 130,000), an age in years (18 to 80) under its
    plain name and under NHANES's (RIDAGEYR), and a sample number (sample_id, from 5,001 in steps
    of 3, so no numbering of the rows) are numeric, non-negative and take many values, so only
    their names keep them out of the assay block. Beside 25 gene counts (Poisson(200), 60 samples)
    under the genomics lens, the reading names the 25 genes alone, and each sample's total is the
    sum of its gene counts (numpy). Beside 25 log2 intensities (normal, mean 20 and standard
    deviation 2) under the metabolomics lens, nothing is read as raw intensities, as the block
    alone reads: the participant numbers' size is no intensity's."""
    from turbotab.core.methods import omics

    rng = np.random.default_rng(1937)
    n = 60
    genes = pd.DataFrame({f"g{j}": rng.poisson(200, n) for j in range(25)})
    person = pd.DataFrame({"SEQN": np.arange(130000, 130000 + 7 * n, 7),
                           "age": rng.integers(18, 81, n), "RIDAGEYR": rng.integers(18, 81, n),
                           "sample_id": np.arange(5001, 5001 + 3 * n, 3)})
    reading = omics.scale_reading(pd.concat([person, genes], axis=1), ["genomics"])
    assert reading is not None and reading.kind == "counts", reading
    assert reading.columns == tuple(genes.columns), reading.columns
    assert np.array_equal(reading.totals, genes.to_numpy().sum(axis=1))
    logged = pd.DataFrame({f"m{j}": rng.normal(20.0, 2.0, n) for j in range(25)})
    assert omics.scale_reading(logged, ["metabolomics"]) is None
    beside = pd.concat([person[["SEQN"]], logged], axis=1)
    assert omics.scale_reading(beside, ["metabolomics"]) is None


# The names the omics reading leaves out of the block (the orchestrator's ruling on claims
# backtest C-193, round 3): an identifier, a person's characteristic or a time, as the whole name
# or one whole word of it, in any case.
NOT_ASSAY_NAMES = ("SEQN", "seqn", "id", "ID", "sample_id", "subject_id", "patient_id", "visit_id",
                   "Specimen_ID", "Age", "sex", "GENDER", "bmi", "RIDAGEYR", "riagendr", "RIDRETH1",
                   "BMXBMI", "INDFMPIR", "visit", "Time", "timepoint", "date", "day", "days",
                   "hour", "hours", "year", "years", "session", "age_at_draw", "Visit number",
                   "draw.date", "hours-fasted", "BMI (kg/m2)")


@pytest.mark.parametrize("name", NOT_ASSAY_NAMES)
def test_each_name_the_reading_leaves_out_leaves_the_block_and_its_totals(name):
    """Each name of the ruling's list, as the whole name or one whole word of it, in any case: a
    column so named, numeric with many values (whole numbers 18 to 1,000), stays out of the block
    beside 25 gene counts (Poisson(200), 60 samples) under the genomics lens, and each sample's
    total is the sum of its gene counts alone (numpy)."""
    from turbotab.core.methods import omics

    rng = np.random.default_rng(1939)
    n = 60
    genes = pd.DataFrame({f"g{j}": rng.poisson(200, n) for j in range(25)})
    frame = pd.concat([pd.DataFrame({name: rng.integers(18, 1001, n)}), genes], axis=1)
    reading = omics.scale_reading(frame, ["genomics"])
    assert reading is not None and reading.kind == "counts", reading
    assert reading.columns == tuple(genes.columns), reading.columns
    assert np.array_equal(reading.totals, genes.to_numpy().sum(axis=1))


# Assay features whose names the roles stage's word lists read as a covariate or an identifier:
# methionine and its sulfoxide as Biocrates panels name them (Met, Met-SO), and the genes MET (the
# hepatocyte growth factor receptor), IDS (iduronate 2-sulfatase) and PIR (pirin).
FEATURES_NAMED_LIKE_COVARIATES = ("Met", "Met-SO", "MET", "IDS", "PIR")


def test_methionine_and_the_genes_met_ids_and_pir_stay_in_the_block():
    """The silent side of the names the reading leaves out: Met, Met-SO, MET, IDS and PIR are
    assay features, none of them a whole name or a whole word of the ruling's list. Beside 20
    other assay columns, each of the 25 Poisson(200) counts by construction (60 samples) under
    the genomics lens and log-normal intensities in the thousands (median 5,000, log SD 1)
    under the metabolomics lens, every one stays in the block the reading names."""
    from turbotab.core.methods import omics

    rng = np.random.default_rng(1940)
    names = FEATURES_NAMED_LIKE_COVARIATES + tuple(f"f{j}" for j in range(20))
    counts = pd.DataFrame({name: rng.poisson(200, 60) for name in names})
    reading = omics.scale_reading(counts, ["genomics"])
    assert reading is not None and reading.kind == "counts", reading
    assert reading.columns == names, set(names) - set(reading.columns)
    intensities = pd.DataFrame({name: rng.lognormal(np.log(5000.0), 1.0, 60) for name in names})
    reading = omics.scale_reading(intensities, ["metabolomics"])
    assert reading is not None and reading.kind == "intensities", reading
    assert reading.columns == names, set(names) - set(reading.columns)


# Assay columns named as laboratories name them, among them gene symbols that begin with a
# covariate's or a time's letters but are whole other words: AGER (the advanced glycosylation
# end-product receptor), TIMELESS, HIPK2, BMI1, IDH1, PER1.
ASSAY_NAMES = ("AGER", "TIMELESS", "HIPK2", "BMI1", "IDH1", "PER1", "TP53", "GAPDH", "ACTB",
               "ENSG00000141510", "HMDB0000122", "LysoPC a C18:0", "mz_123.4_rt_5.6", "glucose",
               "lactate", "alanine", "C14:0", "feature_1", "gene_7", "X21", "SLC2A1", "HMDB0000190",
               "PC aa C36:2", "Cer d18:1/16:0", "citrulline", "M123T45")


def test_an_assay_column_whose_name_is_no_identifier_characteristic_or_time_stays_in():
    """Claims backtest C-193, the silent side of the names the block leaves out: 26 assay columns
    named as laboratories name them (gene symbols, Ensembl and HMDB accessions, lipid and m/z
    features, metabolites), several beginning with the letters of a covariate or a time as a
    word does not, each Poisson(200) counts by construction, 60 samples, under the genomics lens.
    Expected: every one stays in the block the reading names."""
    from turbotab.core.methods import omics

    rng = np.random.default_rng(1938)
    frame = pd.DataFrame({name: rng.poisson(200, 60) for name in ASSAY_NAMES})
    reading = omics.scale_reading(frame, ["genomics"])
    assert reading is not None and reading.kind == "counts", reading
    assert reading.columns == ASSAY_NAMES, set(ASSAY_NAMES) - set(reading.columns)


def _median_skewness(frame: pd.DataFrame) -> float:
    """The median over the columns of each one's sample skewness, as scipy computes it."""
    from scipy.stats import skew

    return float(np.median(skew(frame.to_numpy(dtype=float), axis=0)))


def _span(frame: pd.DataFrame) -> float:
    """The largest value over the smallest positive one."""
    v = frame.to_numpy(dtype=float)
    return float(v.max() / v[v > 0].min())


def test_a_right_skewed_panel_below_the_thousands_reads_as_raw_intensities():
    """Claims backtest C-193, the fires side of the skewness test: a targeted metabolomics panel in
    µM, 25 metabolites on 60 samples, each log-normal by construction (log-concentration normal,
    mean 2 and standard deviation 1), so unlogged. Its values stay in the hundreds, never reaching
    the thousands raw peak areas reach, but they span orders of magnitude and every feature is
    right-skewed (scipy's sample skewness, median well above 1), as an unlogged concentration is.
    Expected: under the metabolomics lens it reads as raw intensities, all 25 columns."""
    from turbotab.core.methods import omics

    rng = np.random.default_rng(1933)
    frame = _panel(lambda n: rng.lognormal(2.0, 1.0, n))
    assert frame.to_numpy().max() < 500 and _span(frame) > 300, (frame.to_numpy().max(),
                                                                  _span(frame))
    assert _median_skewness(frame) > 1.5, _median_skewness(frame)
    reading = omics.scale_reading(frame, ["metabolomics"])
    assert reading is not None and reading.kind == "intensities", reading
    assert reading.columns == tuple(frame.columns)


def test_a_symmetric_block_below_the_thousands_is_not_read_as_raw_intensities():
    """Claims backtest C-193, the silent side of the skewness test: 25 columns on 60 samples, each
    uniform from 1 to 400 by construction, so symmetric (scipy's median sample skewness near 0),
    with no right tail an unlogged concentration has. They span two orders of magnitude and stay
    below the thousands. Expected: nothing is read as raw intensities, since neither the
    thousands nor a right skew says the values are unlogged."""
    from turbotab.core.methods import omics

    rng = np.random.default_rng(1934)
    frame = _panel(lambda n: rng.uniform(1.0, 400.0, n))
    assert frame.to_numpy().max() < 400 and _span(frame) > 200, _span(frame)
    assert abs(_median_skewness(frame)) < 0.2, _median_skewness(frame)
    assert omics.scale_reading(frame, ["metabolomics"]) is None


def test_a_block_in_the_thousands_that_spans_little_is_not_read_as_raw_intensities():
    """Claims backtest C-193, the silent side of the span test: log2 intensities (normal, mean 20
    and standard deviation 1.5) stored as thousandths, so by construction logged values that reach
    the tens of thousands but span less than threefold, 25 features on 60 samples. Expected:
    nothing is read as raw intensities, since raw intensities reaching the thousands also span
    orders of magnitude."""
    from turbotab.core.methods import omics

    rng = np.random.default_rng(1935)
    frame = _panel(lambda n: (rng.normal(20.0, 1.5, n) * 1000).round(0))
    assert frame.to_numpy().max() > 10_000 and _span(frame) < 3, _span(frame)
    assert omics.scale_reading(frame, ["metabolomics"]) is None


def test_a_clinical_handful_of_laboratory_columns_is_no_feature_block():
    """Claims backtest C-193, the silent side of the block's floor: eight laboratory columns, a
    clinical table's handful, under either omics lens. By construction each would read as raw on
    its values: under genomics whole Poisson(200) counts, under metabolomics log-normal values
    (log mean 7, standard deviation 1) reaching the thousands and spanning orders of magnitude.
    Expected: neither eight-column table is read as a block, while the same draws over 25
    columns read as raw counts and raw intensities."""
    from turbotab.core.methods import omics

    rng = np.random.default_rng(1936)
    counts = _panel(lambda n: rng.poisson(200, n))
    peaks = _panel(lambda n: rng.lognormal(7.0, 1.0, n).round(1))
    assert peaks.to_numpy().max() > 1_000 and _span(peaks) > 300, _span(peaks)
    assert omics.scale_reading(counts, ["genomics"]).kind == "counts"
    assert omics.scale_reading(peaks, ["metabolomics"]).kind == "intensities"
    handful = [f"m{j}" for j in range(8)]
    assert omics.scale_reading(counts[handful], ["genomics"]) is None
    assert omics.scale_reading(peaks[handful], ["metabolomics"]) is None


def _estimated_block(share_of_a_million: float, seed: int = 1932) -> pd.DataFrame:
    """Fractional values as a transcript quantifier writes them, 30 genes on 40 samples, each
    gene with a zero and the first gene reaching the tens of thousands, every sample's total then
    scaled to ``share_of_a_million`` of a million."""
    rng = np.random.default_rng(seed)
    values = rng.gamma(0.8, 400.0, (40, 30)) + 0.37
    values[rng.integers(0, 40, 30), np.arange(30)] = 0.0
    values[:, 0] *= 40.0
    values *= share_of_a_million * 1e6 / values.sum(axis=1, keepdims=True)
    return pd.DataFrame(values, columns=[f"gene{j}" for j in range(30)])


def _per_million_tolerance() -> float:
    """The genomics pack's tolerance for a CPM's or TPM's sample totals around a million
    (GENOMICS_PACK §02 row 4, ``packs._LIBRARY_NEAR``), which its data-type card reads."""
    from turbotab import packs

    return float(packs._LIBRARY_NEAR)


def test_fractional_values_whose_totals_leave_the_packs_tolerance_of_a_million_are_asked_about():
    """Claims backtest C-193, the fires side of the per-million test: fractional values with zeros
    and the thousands in many columns, every sample's total a tenth beyond the genomics pack's
    tolerance of a million (above it and below it), are estimated counts or FPKM, which the
    values cannot tell apart, so the reading asks."""
    from turbotab.core.methods import omics

    near = _per_million_tolerance()
    for share in (1 + 1.1 * near, 1 - 1.1 * near):
        frame = _estimated_block(share)
        totals = frame.sum(axis=1).to_numpy()
        assert np.allclose(totals, share * 1e6) and (frame == 0).any().sum() >= 25
        assert (frame.to_numpy() % 1 != 0).any() and frame.to_numpy().max() >= 10_000
        reading = omics.scale_reading(frame, ["genomics"])
        assert reading is not None and reading.estimated, (share, reading)


def test_fractional_values_whose_totals_sit_within_the_packs_tolerance_of_a_million_are_normal():
    """Claims backtest C-193, the silent side of the per-million test: the same fractional block
    with every sample's total within nine tenths of the genomics pack's tolerance of a million
    (TMM-scaled CPM, whose totals are near a million but not exactly one) is normalized, so
    nothing is asked."""
    from turbotab.core.methods import omics

    near = _per_million_tolerance()
    for share in (1 + 0.9 * near, 1 - 0.9 * near, 1.0):
        frame = _estimated_block(share)
        assert np.allclose(frame.sum(axis=1).to_numpy(), share * 1e6)
        assert omics.scale_reading(frame, ["genomics"]) is None, share


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


def test_the_grouping_entry_states_the_rows_per_group_at_which_the_refusal_stops():
    """Claims backtest D-058, the sentences against the guard: the grouping entry's "why" and its
    "Adjust and cluster" option state the rows per group below which a yes/no outcome gets no
    intercept each, and the design stage's backstop keeps its own copy of that number. The number
    is read from the backstop's behavior, never from either copy: 30 households of m rows each,
    every household holding both answers, for m from 2 to 40; the smallest m it lets through is
    the number, every smaller m is refused, and both sentences state it."""
    from turbotab.core import decisions as d, estimand
    from turbotab.core.decisions import ProjectState
    from turbotab.core.teaching import content

    refused = {}
    for m in range(2, 41):
        group = np.repeat(np.arange(30), m)
        frame = pd.DataFrame({"pid": np.arange(len(group)), "household": group,
                              "x": np.tile(np.linspace(-1, 1, m), 30),
                              "ill": np.tile(np.arange(m) % 2, 30)})
        state = ProjectState(lens=["clinical"], target="ill", task="binary", purpose="inference",
                             grain={"grain": "one_row_per_unit"},
                             roles={"pid": "identifier", "household": "cluster", "x": "exposure"},
                             clusters=d.ClusterSpec(column="household", adjust="fixed_effects"))
        refused[m] = estimand.intercepts_refusal(state, _Store(frame), np.arange(len(frame)),
                                                 "binary") is not None
    passed = [m for m, r in refused.items() if not r]
    assert passed and refused[2], refused
    rows = passed[0]
    assert all(refused[m] for m in range(2, rows)) and all(not refused[m] for m in passed), refused
    why = content.CLUSTERS["why"]
    adjust = next(o for o in content.CLUSTERS["options"] if o["value"] == "fixed_effects")
    assert f"groups under {rows} rows" in why, (rows, why)
    assert f"need {rows} rows per group" in adjust["consequence"], (rows, adjust)


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
    ("a_nutrient_the_iom_sets_a_ul_for", {"nutrient": "magnesium"}, {"nutrient": None}),
    ("a_nutrient_the_iom_sets_a_ul_for", {"nutrient": "Vitamin D"}, {"nutrient": "vitamin K"}),
    ("a_nutrient_the_iom_sets_a_ul_for", {"nutrient": "calcium"}, {"nutrient": "potassium"}),
    ("a_name_read_as_no_nutrient", {"nutrient": None}, {"nutrient": "magnesium"}),
    ("a_name_read_as_no_nutrient", {"nutrient": None, "cutoff": 350}, {"cutoff": 350}),
    ("an_outcome_its_evidence_covers", {"task": "binary", "purpose": "inference"},
     {"task": "ordinal", "purpose": "inference"}),
    ("an_outcome_its_evidence_covers", {"task": "regression", "purpose": "inference"},
     {"task": "multiclass", "purpose": "inference"}),
    ("an_outcome_its_evidence_covers", {"task": "ordinal", "purpose": "prediction"},
     {"task": "time_to_event", "purpose": "inference"}),
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
    code the claim guards: a UL cut-off; a nutrient the IOM DRI tables set a UL for (vitamin K and
    potassium have none, and a name the reader does not read, NHANES's supplement-recall total
    DS1TMAGN among them, names no nutrient); a name read as no nutrient (a context that does not
    say how the name was read is outside); a weight named as the CDC codebooks name it (a tissue
    mass and a body weight are no sampling weight); an omics lens; a grouping column; under
    inference a numeric or a yes/no outcome, the two the incidental-parameter evidence covers
    (an ordinal, multiclass or time-to-event outcome is outside it), and under prediction any; a
    least-squares fit on an outcome with a clinical cut-off."""
    held = getattr(regimes, predicate)
    assert held(inside) is True and held(outside) is False


# Weights as NCHS names them: the subsamples' (Subsample A, B and C, the VOC subsample, each "WTS",
# the subsample's letters and the cycle, cut to the eight characters a SAS transport name holds),
# the dietary weights pooled as the weighting tutorial pools them (the sample's word and the
# years: DRD1 and DR2D over four years), and the other samples' with their cycles. Beside them,
# names that only begin like one: a word that starts with a sample's letters, a body weight, a
# tissue mass.
NHANES_WEIGHT_NAMES = ("WTSA2YR", "WTSB2YR", "WTSC2YR", "WTSHM2YR", "WTSVOC2Y", "WTSSBJ2Y",
                       "DRD14YR", "DR2D4YR", "MEC4YR", "WTMEC6YR", "WTINTPRP", "WTDR4YR",
                       "WTSOG2YR", "wtph2yr")
NOT_NHANES_WEIGHTS = ("WTMECHANIC", "WTDRUG", "WTINTERVAL", "WTSAFETY", "WTPHONE", "BMXWT",
                      "WTKG", "sample_weight_mg")


def test_the_nhanes_weight_regime_reads_the_names_nchs_gives_and_no_look_alike():
    """The C-143 predicate, at its edges: a table carries an NHANES sample weight when a column
    is named as NCHS names one, a subsample's whose sample the rule does not know included, and
    not when a column only begins with a sample's letters (a mechanic, a drug, an interval)."""
    for name in NHANES_WEIGHT_NAMES:
        assert regimes.names_an_nhanes_weight({"columns": ["LBXTC", name]}) is True, name
    for name in NOT_NHANES_WEIGHTS:
        assert regimes.names_an_nhanes_weight({"columns": ["LBXTC", name]}) is False, name

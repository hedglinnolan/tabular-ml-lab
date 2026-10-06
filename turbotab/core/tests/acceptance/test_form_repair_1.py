"""FORM · repair round 1: the independent verifier's open item and leash findings on wave 2b's
package FORM (MODELING_SEQUENCE §1 rows 5 and 7, §2, §4; BLUEPRINT §14–§14.3).

1. **k by the declared rule is read on the rows the fit sees** (Harrell, RMS §2.4.6, on the
   effective sample size of §4.4). Before: a spline declared while the form card recomputed got
   k = 4 with no n (the task was detected, not answered), and a record's k and its stated n were
   never re-derived when the analyzed rows changed (the complete cases, the consumers-only
   domain: "here 950" on a fit of 86 rows). Expected: the rule reads the detected task and only
   known rows, waits (``not_yet``) while they are being read, and a rule-set spline whose rows
   changed is asked again, its re-answer stating the rows' own n.
2. **The Router reads only a fresh form card.** Before: while the card recomputed, the form
   question stood answered on the card last computed, so the models were accepted before it
   reopened (``test_ledger_repair_3``'s race). Expected: it waits on the card; only a failed card
   stage lets the last card stand.
3. **Codes or amounts is the ledger's reading** (BLUEPRINT §14.1: only ``readings.py`` decides a
   registry kind). Before: ``band`` (whole numbers 1–12, its reading unsettled) was proposed a
   spline and recorded as one; confirmed as codes it entered as indicators while the methods text
   still said spline. Expected: the card waits for the reading and asks it, no form is recorded on
   it, and a form recorded before a later "codes" answer is corrected in the methods text.
4. **Effect modification on strata that cannot carry an effect** is refused or said plainly:
   never an overflow's text, never a sentence with an empty contrast or a difference of differences
   for an odds ratio; the family names the test it could not compute.
5. **Separation** (Firth's penalized likelihood): the spline's test of association and the
   modification's heterogeneity test are penalized likelihood-ratio tests, each effect a profile
   interval (against R: the penalized likelihood written out and maximized by ``optim`` with its
   exact gradient, cross-checked by ``brglm2``; the profile by ``uniroot``).
6. **A modifier withdrawn after the estimates were seen** still counts in the family.

Expected values come from an independent path: pandas and NumPy counts, Harrell's rule as his text
states it, R, or the fixture's declared truth; never from the code under test.
"""
from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from turbotab.core import decisions as d
from turbotab.core.methods import exposure_form as ef
from turbotab.core.methods import interaction as ix
from turbotab.core.tests import modeling_fixtures as mf
from turbotab.core.tests.acceptance import estimand_fixtures as est_f
from turbotab.core.tests.acceptance.r_reference import needs_r, run_r
from turbotab.core.tests.acceptance.server_drive import (
    Truth, answer_wp17, local_server, open_project,
)
from turbotab.core.tests.truths import answer_adjustment, asked
from turbotab.core.tests.truths import answers as truth_answers

RULE = ("Harrell's rule (3 knots below an effective sample size of 30, 5 from 100, else 4")


def harrell(n: float) -> int:
    """Harrell's rule as RMS §2.4.6 states it: "Small samples (< 30, say) may require the use of
    k = 3"; "When the sample size is large (e.g., n ≥ 100 …), k = 5 is a good choice"; else 4."""
    return 3 if n < 30 else (5 if n >= 100 else 4)


def _record(seq: int, decision: Any, sentence: str = "") -> d.DecisionRecord:
    return d.DecisionRecord(id=f"{seq:032x}", seq=seq, at=datetime.now(timezone.utc),
                            decision=decision, sentence=sentence)


class _Store:
    """A store a decision context reads (the server's ``DataStore`` shape)."""

    def __init__(self, frame: pd.DataFrame):
        self.frame = frame
        self.columns = list(frame.columns)

    def materialize(self, columns: list[str], rows: Any = None) -> pd.DataFrame:
        out = self.frame[list(columns)]
        return out if rows is None else out.iloc[np.asarray(rows, dtype=int)]

    def whole_numbers(self, columns: list[str]) -> dict[str, Any]:
        out = {}
        for c in columns:
            v = pd.to_numeric(self.frame[c], errors="coerce").dropna().to_numpy(float)
            if len(v) and np.all(v == np.round(v)):
                out[c] = {"whole": True, "zero_one": bool(set(np.unique(v)) <= {0.0, 1.0}),
                          "n_values": int(len(np.unique(v))), "min": float(v.min()),
                          "max": float(v.max())}
        return out

    def text_numbers(self, columns: list[str]) -> dict[str, Any]:
        return {}


# ═════════════════════════════════════════════════════════════════════════════
# 1 · k by the rule, on the rows the fit sees
# ═════════════════════════════════════════════════════════════════════════════

ROLES = {"pid": "identifier", "x": "exposure", "z": "covariate"}


def _rule_state(**slots: Any) -> d.ProjectState:
    base = dict(roles=dict(ROLES), role_confirmations=dict(ROLES), target="y", task=None,
                purpose="inference",
                estimand=d.EstimandSpec(exposure="x", measure="mean_difference"))
    return mf.state(**{**base, **slots})


def _rule_frame(n: int = 140, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = rng.normal(size=n)
    y[rng.random(n) < 0.15] = np.nan
    return pd.DataFrame({"pid": np.arange(n), "x": rng.gamma(3.0, 2.0, n),
                         "z": rng.normal(size=n), "y": y,
                         "b": np.where(rng.random(n) < 0.2, "yes", "no")})


@pytest.mark.parametrize("rows", [slice(0, 140), slice(0, 90), slice(0, 31)])
def test_r1_with_no_fresh_card_the_rule_reads_the_detected_task_on_the_known_rows(rows):
    """The verifier's item 2a (state.task None, the task detected): k by the rule on the analyzed
    rows' outcome, the number counted here with pandas, k from Harrell's text."""
    frame = _rule_frame()
    ids = np.arange(len(frame))[rows]
    ctx = {"state": _rule_state(), "store": lambda: _Store(frame), "analyzed": lambda: ids,
           "task": "regression"}
    n = int(frame["y"].iloc[ids].notna().sum())
    done = d.validate({"kind": "set_exposure_form", "column": "x", "form": "spline"}, ctx)
    assert (done.knots, done.knots_rule, done.n_effective) == (harrell(n), "harrell", n)
    from turbotab.core.voice import sentence_for

    assert (f"with `{harrell(n)}` knots, k = {harrell(n)} by {RULE}; here {n:,}, the number of "
            f"analyzed rows)") in sentence_for(done, ctx["state"], ctx)


def test_r1_a_yes_no_outcome_detected_reads_the_smaller_of_events_and_non_events():
    frame = _rule_frame()
    ids = np.arange(len(frame))
    st = _rule_state(target="b", event="yes")
    ctx = {"state": st, "store": lambda: _Store(frame), "analyzed": lambda: ids,
           "detected_task": "binary"}
    events = int((frame["b"] == "yes").sum())
    n = min(events, len(frame) - events)
    done = d.validate({"kind": "set_exposure_form", "column": "x", "form": "spline"}, ctx)
    assert (done.knots, done.n_effective) == (harrell(n), n)


def test_r1_while_the_rows_are_being_read_the_rule_waits_and_k_given_is_the_way_forward():
    """The server's context reads rows, and they are being read (the cohort recomputing for an
    answer just given): the rule is not applied to the whole table's rows in their place, nor
    given its default; the declaration is refused ``not_yet`` with the spline at a k given."""
    frame = _rule_frame()
    ctx = {"state": _rule_state(), "store": lambda: _Store(frame), "analyzed": lambda: None,
           "task": "regression"}
    with pytest.raises(d.Refusal) as refused:
        d.validate({"kind": "set_exposure_form", "column": "x", "form": "spline"}, ctx)
    assert refused.value.code == "not_yet"
    assert [e["decision"]["knots"] for e in refused.value.exits] == [3, 4, 5]
    taken = d.validate(refused.value.exits[1]["decision"], ctx)
    assert (taken.knots, taken.knots_rule, taken.n_effective) == (4, None, None)
    # the one-tap answer's way forward keeps the other columns' forms
    with pytest.raises(d.Refusal) as both:
        d.validate({"kind": "set_forms", "forms": {"x": {"form": "spline"},
                                                    "z": {"form": "linear"}}}, ctx)
    assert both.value.code == "not_yet"
    assert both.value.exits[0]["decision"] == {
        "kind": "set_forms", "forms": {
            "x": {**d.ExposureFormSpec(form="spline", knots=3).model_dump(mode="json")},
            "z": d.ExposureFormSpec(form="linear").model_dump(mode="json")}}


def test_r1_where_no_rows_are_read_the_sentence_never_claims_the_rule_was_applied():
    st = _rule_state()
    done = d.validate({"kind": "set_exposure_form", "column": "x", "form": "spline"}, {"state": st})
    assert (done.knots, done.knots_rule, done.n_effective) == (4, "harrell", None)
    from turbotab.core.voice import sentence_for

    said = sentence_for(done, st)
    assert (f"with `4` knots, k = 4, the default of {RULE}; no effective sample size was read)"
            ) in said
    assert "4 by Harrell's rule" not in said


def test_r1_a_rule_set_spline_on_other_rows_than_the_cards_is_asked_again():
    """The form answer stands only while the n its rule read is the fresh card's: a spline set by
    the rule at 950 rows is stale on a card read at 86 (the consumers-only domain), asked again,
    and answered once re-declared at 86; a k the user gave is not re-derived."""
    st = _rule_state(exposure_forms={"x": d.ExposureFormSpec(form="spline", knots=5,
                                                             knots_rule="harrell",
                                                             n_effective=950.0, scale="raw")})
    card = {"purpose": "inference", "n_effective": 86.0,
            "needs": [{"column": "x", "role": "exposure"}], "waiting": []}
    assert ef.form_answer(st, card) is None
    assert list(ef.stale_forms(st, card)) == ["x"] and ef.unanswered_forms(st, card) == ["x"]
    again = st.model_copy(update={"exposure_forms": {"x": d.ExposureFormSpec(
        form="spline", knots=4, knots_rule="harrell", n_effective=86.0, scale="raw")}})
    assert list(ef.form_answer(again, card)) == ["x"] and not ef.stale_forms(again, card)
    given = st.model_copy(update={"exposure_forms": {"x": d.ExposureFormSpec(
        form="spline", knots=5, scale="raw")}})
    assert list(ef.form_answer(given, card)) == ["x"]


# ═════════════════════════════════════════════════════════════════════════════
# 3 · codes or amounts is the ledger's reading
# ═════════════════════════════════════════════════════════════════════════════

BAND_ROLES = {"pid": "identifier", "x": "exposure", "band": "covariate"}


def _band_frame(n: int = 600, seed: int = 9) -> pd.DataFrame:
    """``band``: whole numbers 1–12 (twelve age bands, say, or twelve clinics), a cause of the
    exposure and the outcome; nothing in its values says which."""
    rng = np.random.default_rng(seed)
    band = rng.integers(1, 13, n)
    x = rng.normal(20, 4, n) + 0.3 * band
    y = 0.2 * x + np.where(band % 3 == 0, 1.5, -0.5) + rng.normal(0, 1, n)
    return pd.DataFrame({"pid": np.arange(n), "x": x.round(3), "band": band, "y": y.round(4)})


def _band_state(**slots: Any) -> d.ProjectState:
    base = dict(roles=dict(BAND_ROLES), role_confirmations=dict(BAND_ROLES), target="y",
                task="regression", purpose="inference",
                estimand=d.EstimandSpec(exposure="x", measure="mean_difference"),
                adjustment=est_f.answers_for("x", {"band": est_f.CONFOUNDER}))
    return mf.state(**{**base, **slots})


def _band_info(frame: pd.DataFrame) -> dict[str, Any]:
    return {c: {"dtype": "integer" if pd.api.types.is_integer_dtype(frame[c]) else "numeric",
                "n_unique": int(frame[c].nunique())} for c in frame.columns}


def test_r3_the_card_waits_for_an_unsettled_reading_and_proposes_no_form_on_a_guess():
    frame = _band_frame()
    st = _band_state()
    facts = _Store(frame).whole_numbers(["x", "band"])
    card = ef.forms_card(st, frame, _band_info(frame), frame["y"], "regression", facts)
    assert [n["column"] for n in card["needs"]] == ["x"]
    assert card["answer"] == {"kind": "set_forms", "forms": {
        "x": {"form": "spline", "knots": harrell(len(frame)), "knots_rule": "harrell"}}}
    (waiting,) = card["waiting"]
    assert (waiting["column"], waiting["role"]) == ("band", "confounder")
    k = int(frame["band"].nunique())
    assert f"`{k}` whole-number values from {frame['band'].min()} to {frame['band'].max()}" in \
        waiting["why"]
    assert ef.form_answer(st.model_copy(update={"exposure_forms": {
        "x": d.ExposureFormSpec(form="linear", scale="raw")}}), card) is None
    assert ef.form_gate(st, card) is None  # asked, not "not applicable"


@pytest.mark.parametrize("value", ["code", "amount"])
def test_r3_each_answer_of_the_reading_produces_its_behavior(value):
    """BLUEPRINT §14.3: confirming each alternative produces that alternative's behavior, through
    the fold: codes are stated (indicators, no form); an amount is asked its form."""
    frame = _band_frame()
    folded = d.fold([_record(1, d.ConfirmReading(reading="code_or_count", column="band",
                                                 value=value))])
    st = _band_state(shape_confirmations=folded.shape_confirmations)
    facts = _Store(frame).whole_numbers(["x", "band"])
    card = ef.forms_card(st, frame, _band_info(frame), frame["y"], "regression", facts)
    assert card["waiting"] == []
    if value == "code":
        assert {"column": "band", "why": "declared codes: it enters as indicators"} in \
            card["stated"]
        assert [n["column"] for n in card["needs"]] == ["x"]
    else:
        assert [n["column"] for n in card["needs"]] == ["x", "band"]
        band = card["needs"][1]
        assert band["proposal"] == {"form": "spline", "knots": harrell(len(frame)),
                                    "knots_rule": "harrell"}


def test_r3_a_form_on_an_unsettled_or_coded_column_is_refused_with_the_readings_answers():
    frame = _band_frame()
    st = _band_state()
    ctx = {"state": st, "store": lambda: _Store(frame), "analyzed": lambda: np.arange(len(frame)),
           "column_info": _band_info(frame)}
    with pytest.raises(d.Refusal) as refused:
        d.validate({"kind": "set_exposure_form", "column": "band", "form": "spline"}, ctx)
    assert refused.value.code == "reading_unsettled"
    assert ("code_or_count", "band") in asked(refused.value.exits)
    offered = {(e["decision"]["column"], e["decision"]["value"]) for e in refused.value.exits
               if (e.get("decision") or {}).get("kind") == "confirm_reading"}
    assert offered == {("band", "amount"), ("band", "code")}
    with pytest.raises(d.Refusal) as one_tap:
        d.validate({"kind": "set_forms", "forms": {"x": {"form": "linear"},
                                                    "band": {"form": "spline"}}}, ctx)
    assert one_tap.value.code == "reading_unsettled"
    coded = st.model_copy(update={"shape_confirmations": {"code_or_count:band": "code"}})
    with pytest.raises(d.Refusal) as codes:
        d.validate({"kind": "set_exposure_form", "column": "band", "form": "linear"},
                   {**ctx, "state": coded})
    assert codes.value.code == "codes_take_no_form"
    assert codes.value.exits[0]["decision"] == {"kind": "confirm_reading",
                                                "reading": "code_or_count", "column": "band",
                                                "value": "amount"}


def test_r3_a_form_recorded_before_a_later_codes_answer_is_not_applied_and_the_methods_say_so():
    """Declared on an amount, then confirmed as codes: the form no longer stands (no spline in the
    design, nothing re-asked: codes take no form), and the methods text, which says what the
    analysis is now, ends the declaration's sentence with the correction; the Record keeps it as
    said."""
    from turbotab.core.voice import restate, sentence_for

    st = _band_state(shape_confirmations={"code_or_count:band": "amount"})
    form = d.validate({"kind": "set_forms", "forms": {"x": {"form": "linear"},
                                                       "band": {"form": "spline", "knots": 4}}},
                      {"state": st})
    said = sentence_for(form, st)
    assert "`band` entered the models as a restricted cubic spline with `4` knots" in said
    now = st.model_copy(update={
        "exposure_forms": d.fold([_record(1, form)]).exposure_forms,
        "shape_confirmations": {"code_or_count:band": "code"}})
    assert "band" not in ef.current_forms(now) and "band" not in ef.stale_forms(now)
    assert "band" not in ef.design_forms(now, ["x", "band"], ["x", "band"])
    assert restate(form, said, st, now) == (
        f"{said} `band` was then recorded as codes for categories, so it entered the models as "
        f"one indicator per level, not in the form declared here.")
    assert restate(form, said, st, st) == said


# ═════════════════════════════════════════════════════════════════════════════
# 4 · a stratum that cannot carry the exposure's effect
# ═════════════════════════════════════════════════════════════════════════════


def _sparse_frame() -> pd.DataFrame:
    """ESTIMAND's cohort with ``region``: one participant in the west; and ``grp``: forty
    participants in group b, none of whom had the event."""
    frame = est_f.cohort(n=900, seed=31)
    frame["region"] = np.where(np.arange(len(frame)) % 2 == 0, "north", "south")
    frame.loc[frame.index[7], "region"] = "west"
    no = frame.index[frame["dm"] == "no"][:40]
    frame["grp"] = "a"
    frame.loc[no, "grp"] = "b"
    return frame


def _modification_state(modifier: str, **slots: Any) -> d.ProjectState:
    return est_f.state(target="dm", task="binary", event="yes", measure="odds_ratio",
                       modifications={modifier: d.ModificationSpec(kind="effect_modification",
                                                                   exposure="fiber")}, **slots)


def test_r4_the_strata_checks_are_the_counts():
    frame = _sparse_frame()
    events = (frame["dm"] == "yes").astype(float).to_numpy()
    west = int((frame["region"] == "west").sum())
    b = frame["grp"] == "b"
    assert (west, int(b.sum()), int(frame.loc[b, "dm"].eq("yes").sum())) == (1, 40, 0)
    assert ix.strata_problems("region", "fiber", frame["region"], frame["fiber"], events,
                              "binary", categorical=True) == [
        "`region` = `west` has 1 row, so `fiber`'s effect within it cannot be estimated"]
    assert ix.strata_problems("grp", "fiber", frame["grp"], frame["fiber"], events, "binary",
                              categorical=True) == [
        "`grp` = `b` has no event among its 40 rows, so the odds ratio within it cannot be "
        "estimated"]
    # a two-valued number is two strata; a constant exposure in one leaves no contrast there
    flag = (frame["fiber"] > frame["fiber"].median()).astype(float)
    flat = frame["fiber"].where(flag == 0, 20.0)
    n1 = int((flag == 1).sum())
    assert ix.strata_problems("flag", "fiber", flag, flat, None, "regression",
                              categorical=False) == [
        f"`flag` = `1` has {n1:,} rows, all at one value of `fiber`, so `fiber`'s effect within "
        f"it cannot be estimated"]
    # a numeric modifier with many values has no strata of rows; healthy strata pass
    assert ix.strata_problems("age", "fiber", frame["age"], frame["fiber"], events, "binary",
                              categorical=False) == []
    assert ix.strata_problems("sex", "fiber", frame["sex"], frame["fiber"], events, "binary",
                              categorical=True) == []


def test_r4_such_a_modifier_is_refused_at_declaration_with_its_way_forward():
    frame = _sparse_frame()
    st = est_f.state(target="dm", task="binary", event="yes", measure="odds_ratio")
    ctx = {"state": st, "store": lambda: _Store(frame), "analyzed": lambda: np.arange(len(frame)),
           "columns": list(frame.columns)}
    with pytest.raises(d.Refusal) as refused:
        d.validate({"kind": "set_modification", "modifier": "region"}, ctx)
    assert refused.value.code == "modifier_stratum_not_estimable"
    assert refused.value.message.startswith(
        "`region` = `west` has 1 row, so `fiber`'s effect within it cannot be estimated.")
    assert refused.value.exits == [{"label": "Choose another modifier (one whose every stratum "
                                             "holds the exposure's contrast and the outcome's "
                                             "both kinds)", "decision": None}]
    with pytest.raises(d.Refusal) as none:
        d.validate({"kind": "set_modification", "modifier": "grp"}, ctx)
    assert "`grp` = `b` has no event among its 40 rows" in none.value.message
    assert d.validate({"kind": "set_modification", "modifier": "sex"}, ctx).modifier == "sex"


@pytest.fixture(scope="module")
def sparse_run(tmp_path_factory) -> dict[str, Any]:
    """The stage when the rows changed after the declaration: no fit, said plainly."""
    from turbotab.core.tests.acceptance.test_form_7_modification import _run

    frame = _sparse_frame()
    st = _modification_state("region")
    return _run(frame, tmp_path_factory.mktemp("sparse"), st)


def test_r4_the_stage_says_it_in_plain_words_and_its_sentence_is_whole(sparse_run):
    (result,) = sparse_run["modifications"]
    assert result["families"] == [] and result["contrast"] == "" and result["strata"] == []
    reason = "`region` = `west` has 1 row, so `fiber`'s effect within it cannot be estimated"
    assert result["concerns"] == [f"Not estimated: {reason}. Choose another modifier, or one "
                                  f"whose every stratum holds rows of both kinds."]
    assert result["sentence"] == (
        f"Effect modification of `fiber`'s effect on `dm` by `region` was declared before the "
        f"estimates were seen; the odds ratio within each level of `region` was not estimated: "
        f"{reason}.")
    text = " ".join([result["sentence"], *result["concerns"], sparse_run["methods"]])
    for raw in ("math range error", "f(a) and f(b)", "` at ,", "at `,", "difference of differences"):
        assert raw not in text, raw
    count = sparse_run["family_count"]
    assert count["n_tests"] == 2 and count["not_computed"] == ["region"]
    assert count["statement"] == (
        "2 tests in the family: the exposure's effect and 1 declared heterogeneity test "
        "(`region`); the p-values are reported unadjusted with the number of tests stated; the "
        "heterogeneity test of `region` could not be computed on these rows (said with it) and "
        "is counted as declared")


@pytest.mark.parametrize("raised", [ValueError("math range error"),
                                    ValueError("f(a) and f(b) must have different signs"),
                                    np.linalg.LinAlgError("Singular matrix")])
def test_r4_a_refit_that_fails_otherwise_is_said_in_plain_words(raised, tmp_path, monkeypatch):
    """Strata that each hold the effect, and a refit that fails for a reason no check foresaw (the
    estimation raising, as an overflow or a root finder does): the concern and the sentence say
    what failed in plain words, never the exception's own text; the family count names the test
    it could not compute."""
    from turbotab.core.tests.acceptance.test_form_7_modification import _run

    def fails(*args: Any, **kwargs: Any) -> Any:
        raise raised

    monkeypatch.setattr(ix, "_family", fails)
    frame = est_f.cohort(n=600, seed=37)
    out = _run(frame, tmp_path, _modification_state("sex"))
    (result,) = out["modifications"]
    (fit,) = result["families"]
    plain = ("Linear model could not be refit with the product terms on these rows: its "
             "estimation did not reach finite estimates (a stratum with too few rows or events for "
             "the terms it adds, or nearly collinear terms).")
    assert (fit["measure"], fit["scale"], fit["effects"]) == ("odds ratio", "ratio", [])
    assert fit["concerns"] == [plain]
    assert result["sentence"] == (
        "Effect modification of `fiber`'s effect on `dm` by `sex` was declared before the "
        "estimates were seen; the odds ratio within each level of `sex` was not estimated: "
        f"l{plain[1:]}")
    text = " ".join([result["sentence"], *fit["concerns"], *result["concerns"], out["methods"]])
    assert str(raised) not in text
    assert out["family_count"]["not_computed"] == ["sex"]


def test_r4_a_family_named_for_a_person_keeps_its_capital_in_the_sentence():
    st = SimpleNamespace(target="t", task="time_to_event")
    cox = SimpleNamespace(key="cox", label="Cox proportional hazards")
    words = ix._failure_words(cox, OverflowError("math range error"))
    result = ix.ModificationResult(
        modifier="sex", kind="effect_modification", exposure="fiber", status="declared",
        contrast="", low=0.0, high=0.0, strata=[], reference="", adjusted_for=[],
        families=[ix.ModificationFit(family="cox", label=cox.label, measure="hazard ratio",
                                     scale="ratio", n_rows=10, concerns=[words])], sentence="")
    assert ix.sentence(st, result, task="time_to_event", not_estimated=[words]) == (
        "Effect modification of `fiber`'s effect on `t` by `sex` was declared before the estimates "
        "were seen; the hazard ratio within each level of `sex` was not estimated: Cox "
        "proportional hazards could not be refit with the product terms on these rows: its "
        "estimation did not reach finite estimates (a stratum with too few rows or events for the "
        "terms it adds, or nearly collinear terms).")


# ═════════════════════════════════════════════════════════════════════════════
# 6 · a modifier withdrawn after the estimates were seen still counts
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("locked", [True, False])
def test_r6_a_withdrawal_after_the_estimates_keeps_the_test_in_the_family(locked):
    from turbotab.core.voice import sentence_for

    base = est_f.state(target="dm", task="binary", event="yes", measure="odds_ratio")
    first = d.validate({"kind": "set_modification", "modifier": "sex"}, {"state": base})
    second = d.validate({"kind": "set_modification", "modifier": "smoking"}, {"state": base})
    records = [_record(1, first), _record(2, second)]
    declared = base.model_copy(update={"modifications": d.fold(records).modifications})
    assert ix.family_count(declared)["n_tests"] == 3  # the exposure's test and two declared
    at_withdrawal = declared.model_copy(update={"plan_locked": True if locked else None})
    gone = d.validate({"kind": "set_modification", "modifier": "smoking", "withdraw": True,
                       "post_hoc": not locked}, {"state": at_withdrawal})  # the client's flag
    assert gone.post_hoc is locked  # the server's: whether the estimates had been seen
    after = at_withdrawal.model_copy(update={
        "modifications": d.fold([*records, _record(3, gone)]).modifications})
    assert list(ix.declared(after)) == ["sex"]
    count = ix.family_count(after)
    if locked:
        assert (count["n_tests"], count["withdrawn"]) == (3, ["smoking"])
        assert count["statement"] == (
            "3 tests in the family: the exposure's effect, 1 declared heterogeneity test (`sex`) "
            "and 1 withdrawn after the estimates were seen (`smoking`); the p-values are reported "
            "unadjusted with the number of tests stated")
        assert sentence_for(gone, at_withdrawal) == (
            "The declared modifier `smoking` was withdrawn; it was declared when the estimates "
            "were seen, so its heterogeneity test still counts in the family of tests stated.")
    else:
        assert (count["n_tests"], count["withdrawn"]) == (2, [])
        assert sentence_for(gone, at_withdrawal) == "The declared modifier `smoking` was withdrawn."


# ═════════════════════════════════════════════════════════════════════════════
# 1, 2, 3 through the server: the rows change, the card recomputes, a reading is asked
# ═════════════════════════════════════════════════════════════════════════════


def _step(drive: Any, key: str) -> dict[str, Any]:
    return next(s for s in drive.view()["interview"] if s["key"] == key)


def _until(fn: Any, timeout: float = 240.0, what: str = "") -> Any:
    end = time.monotonic() + timeout
    while True:
        found = fn()
        if found:
            return found
        assert time.monotonic() < end, what
        time.sleep(0.05)


def _card(drive: Any, until: Any) -> dict[str, Any]:
    """The form card once it is fresh and says what the test waits for."""
    return _until(lambda: (lambda c: c if until(c) else None)(drive.artifact("forms")),
                  what="the form card never read the present rows")


def _open(drive: Any, roles: dict[str, str], missing: str) -> None:
    drive.decide({"kind": "set_lens", "lenses": ["clinical"]})
    drive.reach("target")
    drive.decide({"kind": "set_target", "column": "y"})
    drive.answer("task", {"kind": "set_task", "column": "y", "task": "regression"})
    drive.reach("purpose")
    drive.decide({"kind": "set_purpose", "purpose": "inference"})
    drive.answer("grain", {"kind": "set_grain", "grain": "one_row_per_unit", "id_column": "pid"})
    drive.reach("roles")
    drive.decide_roles(dict(roles))
    drive.answer("exclusions", {"kind": "set_exclusions", "rules": []})
    drive.answer("missing", {"kind": "set_missing", "strategy": missing})
    drive.answer("split", {"kind": "set_split", "holdout": 0.0, "seed": 0, "folds": 5})


ROWS_ROLES = {"pid": "identifier", "dose": "exposure", "age": "covariate", "z": "covariate"}
ROWS_TRUTH = {"adjust:age": "yes,yes,no", "adjust:z": "yes,yes,no",
              "code_or_count:pid": "amount", "code_or_count:dose": "amount"}


def _rows_frame(n: int = 950, seed: int = 61) -> pd.DataFrame:
    """A supplement few take (``dose``, zero for about nine in ten), a confounder ``z`` missing for
    a quarter of the participants, and ``age``."""
    rng = np.random.default_rng(seed)
    age = rng.normal(50, 9, n).round(2)
    z = rng.normal(0, 1, n)
    takes = rng.random(n) < 0.09 + 0.02 * (z > 0)
    dose = np.where(takes, rng.gamma(2.0, 40.0, n), 0.0).round(2)
    y = (100 + 0.05 * np.minimum(dose, 120) + 0.3 * (age - 50) + 2 * z
         + rng.normal(0, 5, n)).round(3)
    z = np.where(rng.random(n) < 0.25, np.nan, z.round(4))
    return pd.DataFrame({"pid": np.arange(1, n + 1), "dose": dose, "age": age, "z": z, "y": y})


@pytest.fixture(scope="module")
def rows_journey(tmp_path_factory) -> dict[str, Any]:
    folder = tmp_path_factory.mktemp("form_rows")
    frame = _rows_frame()
    csv = folder / "rows.csv"
    frame.to_csv(csv, index=False)
    seen: dict[str, Any] = {"frame": frame}
    with local_server(folder / "home") as client:
        drive = open_project(client, csv, Truth(ROWS_TRUTH, fixture="the FORM rows journey"))
        _open(drive, ROWS_ROLES, "multiple_imputation")
        drive.answer("estimand", {"kind": "set_estimand", "exposure": "dose", "effect": "total",
                                  "measure": "mean_difference"})
        drive.reach("adjustment")
        answer_adjustment(drive.post, drive.artifact("proposals")["adjustment"], drive.truth)
        step = _until(lambda: (lambda s: s if s["status"] == "open" else None)(
            _step(drive, "form")), what="the form never opened")
        # `dose`'s numbers (nine in ten at zero) do not settle "amount" by their values: the
        # question asks the reading first, answered from the fixture's truth
        seen["ask"] = step["ask"]
        for decision in truth_answers({"exits": step["ask"]["exits"]}, drive.truth):
            r = drive.post(decision)
            assert r.status_code == 200, r.text[:600]
        seen["card_all"] = _card(drive, lambda c: not c["waiting"])
        every = {"kind": "set_forms", "forms": {c: {"form": "spline"} for c in ("dose", "age", "z")}}
        r = drive.post(every)
        assert r.status_code == 200, r.text[:600]
        seen["record_all"] = drive.view()["decisions"][-1]
        # The verifier's repro of item 2a: the complete cases, then at once a spline of `age`.
        r = drive.post({"kind": "set_missing", "strategy": "complete_case"})
        assert r.status_code == 200, r.text[:600]
        spline = {"kind": "set_exposure_form", "column": "age", "form": "spline"}
        responses = []
        while True:
            r = drive.post(spline)
            responses.append((r.status_code, (r.json().get("error") or {}).get("code")))
            if r.status_code != 409 or responses[-1][1] != "not_yet":
                break
            time.sleep(0.05)
        assert r.status_code == 200, r.text[:600]
        seen["repro_responses"] = responses
        seen["record_repro"] = drive.view()["decisions"][-1]
        seen["card_cc"] = _card(drive, lambda c: c["n_effective"] is not None)
        seen["step_cc"] = _until(lambda: (lambda s: s if s["status"] == "open" else None)(
            _step(drive, "form")), what="the form question was not asked again")
        r = drive.post(every)
        assert r.status_code == 200, r.text[:600]
        seen["record_cc"] = drive.view()["decisions"][-1]
        _until(lambda: _step(drive, "form")["status"] == "answered", what="never answered")
        # The consumers-only domain: an estimand change, and fewer rows.
        r = drive.post({"kind": "set_exposure_form", "column": "dose", "form": "spline",
                        "domain": "consumers"})
        while r.status_code == 409 and r.json()["error"]["code"] == "not_yet":
            time.sleep(0.05)
            r = drive.post({"kind": "set_exposure_form", "column": "dose", "form": "spline",
                            "domain": "consumers"})
        assert r.status_code == 200, r.text[:600]
        seen["record_domain"] = drive.view()["decisions"][-1]
        n_cc = seen["card_cc"]["n_effective"]
        seen["card_domain"] = _card(drive, lambda c: c["n_effective"] not in (None, n_cc))
        seen["step_domain"] = _until(lambda: (lambda s: s if s["status"] == "open" else None)(
            _step(drive, "form")), what="the form question was not asked again")
        r = drive.post(seen["card_domain"]["answer"])
        assert r.status_code == 200, r.text[:600]
        seen["record_domain_again"] = drive.view()["decisions"][-1]
        drive.answer("models", {"kind": "select_models", "models": ["linear"]})
        seen["fit"] = drive.artifact("fit")
        seen["view"] = drive.view()
        seen["methods"] = client.get(f"/api/projects/{drive.pid}/methods").json()
        # the Router's inputs, for the counterfactual stage statuses (section 2)
        status = seen["view"]["stages"]
        seen["artifacts"] = {
            s: client.get(f"/api/projects/{drive.pid}/stages/{s}").json()["artifact"]
            for s in ("target_info", "oriented", "structure", "roles", "proposals",
                      "causal_design", "time_varying", "forms")
            if (status.get(s) or {}).get("status") == "fresh"}
    return seen


def _counts(frame: pd.DataFrame) -> tuple[int, int, int]:
    """pandas: every row with the outcome (multiple imputation keeps them), the complete cases
    over the model's columns, and the consumers among them."""
    every = int(frame["y"].notna().sum())
    complete = frame[["dose", "age", "z", "y"]].notna().all(axis=1)
    return every, int(complete.sum()), int((complete & (frame["dose"] > 0)).sum())


def _forms_of(record: dict[str, Any]) -> dict[str, Any]:
    dec = record["decision"]
    return dec["forms"] if dec["kind"] == "set_forms" else {dec["column"]: dec}


def test_r1_through_the_server_k_and_its_n_follow_the_rows_the_fit_sees(rows_journey):
    every, complete, consumers = _counts(rows_journey["frame"])
    assert consumers < 100 <= complete  # the domain moves k from 5 to 4
    # `dose`'s reading (nine in ten at zero, its values not settling "amount") was asked first
    assert ("code_or_count", "dose") in asked(rows_journey["ask"]["exits"])
    assert rows_journey["card_all"]["n_effective"] == every
    for column, spec in _forms_of(rows_journey["record_all"]).items():
        assert (spec["knots"], spec["n_effective"]) == (harrell(every), every), column
    # item 2a: the task detected, not answered (``state.task`` None, as in the verifier's repro);
    # right after the complete cases the rule waited or read the complete cases, never 4 with no n
    assert rows_journey["view"]["state"]["task"] is None
    assert all(code == "not_yet" for _, code in rows_journey["repro_responses"][:-1])
    repro = _forms_of(rows_journey["record_repro"])["age"]
    assert (repro["knots"], repro["n_effective"]) == (harrell(complete), complete)
    assert f"here {complete:,}, the number of analyzed rows" in rows_journey["record_repro"][
        "sentence"]
    # item 2b: the forms the rule set on every row were asked again on the complete cases
    assert rows_journey["card_cc"]["n_effective"] == complete
    assert rows_journey["step_cc"]["followup"] == "stale"
    for column, spec in _forms_of(rows_journey["record_cc"]).items():
        assert spec["n_effective"] == complete, column
    # and on the consumers among them, the domain kept by the one-tap answer
    assert rows_journey["card_domain"]["n_effective"] == consumers
    assert rows_journey["step_domain"]["followup"] == "stale"
    again = _forms_of(rows_journey["record_domain_again"])
    assert again["dose"]["domain"] == "consumers"
    for column, spec in again.items():
        assert (spec["knots"], spec["n_effective"]) == (harrell(consumers), consumers), column
    sentence = rows_journey["record_domain_again"]["sentence"]
    assert (f"`dose` entered the models as a restricted cubic spline with `{harrell(consumers)}` "
            f"knots, k = {harrell(consumers)} by {RULE}; here {consumers:,}, the number of "
            f"analyzed rows)") in sentence


def test_r1_the_fit_is_on_the_rows_and_knots_the_record_states(rows_journey):
    _, _, consumers = _counts(rows_journey["frame"])
    model = rows_journey["fit"]["models"][0]
    assert model["inference"]["n_rows"] == consumers
    tests = {(t["column"], t["test"]): t for t in model["exposure_tests"]}
    assert len(tests[("dose", "overall")]["knots"]) == harrell(consumers)
    in_force = [ln for ln in rows_journey["methods"]["lines"] if ln["in_force"]
                and ln["kind"] in ("set_forms", "set_exposure_form")]
    assert in_force and all(f"here {consumers:,}," in ln["sentence"] for ln in in_force
                            if "Harrell's rule" in ln["sentence"])


def test_r2_the_router_reads_only_a_fresh_form_card(rows_journey):
    """The race (``test_ledger_repair_3``): the Router, handed the card last computed while the
    form card recomputes, holds the form question and every later one; only a failed card stage
    lets the card last computed stand."""
    from turbotab.core.interview import route

    view = rows_journey["view"]
    state = d.ProjectState.model_validate(view["state"])
    records = [d.DecisionRecord.model_validate(r) for r in view["decisions"]]
    stages = dict(view["stages"])
    fresh = dict(rows_journey["artifacts"])
    card = fresh.pop("forms")

    def steps(status: str | None, artifacts: dict[str, Any]) -> dict[str, Any]:
        st = dict(stages)
        if status is not None:
            st["forms"] = {**stages["forms"], "status": status}
        return {s.key: s for s in route(state, st, artifacts, records)}

    from turbotab.core.estimand import served_gate

    now = steps(None, {**fresh, "forms": card})
    assert now["form"].status == "answered" and now["models"].status == "answered"
    assert served_gate(state, list(now.values())) is None
    # the race: the models not chosen yet, the card recomputing for an answer just given
    state = state.model_copy(update={"models": None})
    assert steps(None, {**fresh, "forms": card})["models"].status == "open"
    for status in ("running", "queued", "stale"):
        held = steps(status, {**fresh, "forms_shown": card})
        assert held["form"].status == "waiting" and "forms" in held["form"].waiting_on, status
        assert held["models"].status == "waiting", status
        assert served_gate(state, list(held.values()))["question"] == "form"
    failed = steps("error", {**fresh, "forms_shown": card})
    assert failed["form"].status == "answered" and failed["models"].status == "open"


BAND_TRUTH = {"code_or_count:band": "code", "adjust:band": "yes,yes,no",
              "code_or_count:pid": "amount"}


@pytest.fixture(scope="module")
def band_journey(tmp_path_factory) -> dict[str, Any]:
    folder = tmp_path_factory.mktemp("form_band")
    frame = _band_frame()
    csv = folder / "band.csv"
    frame.to_csv(csv, index=False)
    seen: dict[str, Any] = {"frame": frame}
    with local_server(folder / "home") as client:
        drive = open_project(client, csv, Truth(BAND_TRUTH, fixture="the FORM band journey"))
        _open(drive, BAND_ROLES, "complete_case")
        drive.answer("estimand", {"kind": "set_estimand", "exposure": "x", "effect": "total",
                                  "measure": "mean_difference"})
        drive.reach("adjustment")
        answer_adjustment(drive.post, drive.artifact("proposals")["adjustment"], drive.truth)
        seen["step"] = _until(lambda: (lambda s: s if s["status"] == "open" else None)(
            _step(drive, "form")), what="the form never opened")
        seen["card"] = drive.artifact("forms")
        seen["band_spline"] = drive.post({"kind": "set_exposure_form", "column": "band",
                                          "form": "spline"})
        r = drive.post(seen["card"]["answer"])
        assert r.status_code == 200, r.text[:600]
        seen["step_after_answer"] = _step(drive, "form")
        seen["models_early"] = drive.post({"kind": "select_models", "models": ["linear"]})
        r = drive.post({"kind": "confirm_reading", "reading": "code_or_count", "column": "band",
                        "value": "code"})
        assert r.status_code == 200, r.text[:600]
        seen["card_codes"] = _card(drive, lambda c: not c["waiting"])
        _until(lambda: _step(drive, "form")["status"] == "answered", what="never answered")
        drive.answer("models", {"kind": "select_models", "models": ["linear"]})
        seen["fit"] = drive.artifact("fit")
        seen["methods"] = client.get(f"/api/projects/{drive.pid}/methods").json()
    return seen


def test_r3_through_the_server_the_reading_is_asked_first_and_codes_enter_as_indicators(
        band_journey):
    from turbotab.core.tests.acceptance.server_drive import every_row

    frame = band_journey["frame"]
    card = band_journey["card"]
    assert [w["column"] for w in card["waiting"]] == ["band"]
    assert [n["column"] for n in card["needs"]] == ["x"] and list(card["answer"]["forms"]) == ["x"]
    ask = band_journey["step"]["ask"]
    assert ask["question"] == "form" and ("code_or_count", "band") in asked(ask["exits"])
    assert next(g for g in ask["groups"] if g["columns"] == ["band"])["guess"] == "amount"
    refused = band_journey["band_spline"]
    assert refused.status_code == 409 and refused.json()["error"]["code"] == "reading_unsettled"
    # the rest answered, the question still asks the reading, and nothing later is answered
    assert band_journey["step_after_answer"]["status"] == "open"
    early = band_journey["models_early"]
    assert early.status_code == 409 and early.json()["error"]["code"] == "not_yet"
    assert {"column": "band", "why": "declared codes: it enters as indicators"} in \
        band_journey["card_codes"]["stated"]
    model = band_journey["fit"]["models"][0]
    indicators = [r["feature"] for r in every_row(model) if str(r["feature"]).startswith("band")]
    assert len(indicators) == frame["band"].nunique() - 1
    text = band_journey["methods"]["text"]
    assert "`band` entered the models" not in text
    assert "`x` entered the models as a restricted cubic spline" in text


# ═════════════════════════════════════════════════════════════════════════════
# 5 · separation: Firth's penalized likelihood, against R written out
# ═════════════════════════════════════════════════════════════════════════════

# The penalized log-likelihood l(β) + ½ log |XᵀWX| (Firth 1993) and its exact gradient, the
# modified score Xᵀ(y − μ + h(½ − μ)) (Heinze & Schemper 2002), written out in R and maximized by
# BFGS with every coefficient free, or some held at given values (the restricted fits of a
# likelihood-ratio test and of a profile); ``brglm2``'s mean bias reduction (Kosmidis & Firth
# 2009, which for a logistic model is Firth's) checks the free maximum.
FIRTH_R = """
pl <- function(beta) {
  eta <- drop(X %*% beta); w <- plogis(eta) * plogis(-eta)
  sum(y * plogis(eta, log.p = TRUE) + (1 - y) * plogis(-eta, log.p = TRUE)) +
    0.5 * as.numeric(determinant(crossprod(X, X * w), logarithm = TRUE)$modulus)
}
score <- function(beta) {
  eta <- drop(X %*% beta); mu <- plogis(eta); w <- mu * (1 - mu)
  I <- crossprod(X, X * w); h <- w * rowSums((X %*% solve(I)) * X)
  drop(crossprod(X, y - mu + h * (0.5 - mu)))
}
fit <- function(fixed = integer(0), at = numeric(0), start = NULL) {
  free <- setdiff(seq_len(ncol(X)), fixed)
  full <- function(b) { beta <- numeric(ncol(X)); beta[fixed] <- at; beta[free] <- b; beta }
  s0 <- if (is.null(start)) numeric(length(free)) else start[free]
  o <- optim(s0, function(b) -pl(full(b)), function(b) -score(full(b))[free],
             method = "BFGS", control = list(reltol = 1e-15, maxit = 10000))
  o <- optim(o$par, function(b) -pl(full(b)), function(b) -score(full(b))[free],
             method = "BFGS", control = list(reltol = 1e-15, maxit = 10000))
  list(value = -o$value, beta = full(o$par), conv = o$convergence)
}
profile_ci <- function(j, top) {
  crit <- qchisq(0.95, 1)
  drop_at <- function(b) 2 * (top$value - fit(j, b, top$beta)$value) - crit
  se <- sqrt(diag(solve(crossprod(X, X * (plogis(drop(X %*% top$beta)) *
                                            plogis(-drop(X %*% top$beta)))))))[j]
  out <- c()
  for (side in c(-1, 1)) {
    reach <- 2 * se
    while (drop_at(top$beta[j] + side * reach) < 0) reach <- 2 * reach
    out <- c(out, uniroot(drop_at, sort(c(top$beta[j], top$beta[j] + side * reach)),
                          tol = 1e-11)$root)
  }
  c(out, pchisq(2 * (top$value - fit(j, 0, top$beta)$value), 1, lower.tail = FALSE))
}
"""


def _separated_cohort(n: int = 420, seed: int = 12) -> pd.DataFrame:
    """A yes/no outcome every participant with ``s`` = 1 has (quasi-complete separation: the
    maximum-likelihood log-odds of ``s`` is infinite), beside a curved ``x`` and a ``z``."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    x = rng.gamma(3.0, 2.0, n) + 0.4 * z
    s = (rng.random(n) < 0.08).astype(int)
    eta = -0.6 + 0.35 * np.log(x + 1) + 0.5 * z
    b = np.where(s == 1, 1, (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(int))
    return pd.DataFrame({"pid": np.arange(n), "x": x.round(6), "z": z.round(6), "s": s, "b": b})


def _separated_fit(frame: pd.DataFrame, folder: Path, k: int) -> Any:
    from turbotab.core.stages.modeling import design_stage, fit_stage

    roles = {"pid": "identifier", "x": "exposure", "z": "covariate", "s": "covariate"}
    st = mf.state(roles=roles, role_confirmations=dict(roles), target="b", task="binary",
                  event="1", purpose="inference", models=["linear"], lens=["clinical"],
                  split=d.SplitSpec(holdout=0.0, seed=0, folds=5),
                  shape_confirmations={"code_or_count:b": "amount", "code_or_count:s": "amount"},
                  exposure_forms={"x": d.ExposureFormSpec(form="spline", knots=k)})
    paths = mf.ingest_frame(frame, folder)
    split = mf.split_bundle(np.arange(len(frame)), holdout=0.0)
    info = mf.target_info("binary", "b")
    design = design_stage(mf.context(st, {"split": split, "target_info": info}, paths))
    return fit_stage(mf.context(st, {"design": design, "split": split, "target_info": info},
                                paths))


@needs_r
@pytest.mark.parametrize("k", [3, 4])
def test_r5_under_separation_the_test_of_association_is_a_penalized_likelihood_ratio_test(
        k, tmp_path):
    """Before: "The spline tests for `x` need a Wald covariance, which this table (profile) does
    not give", and no test of association. Expected: the overall and nonlinearity tests are
    2(l*(β̂) − l*(β̃)) on χ²(q), β̃ the fit with the spline's (or its nonlinear) terms held at
    zero, l* the full design's penalized log-likelihood (as the table's own p-values are), to
    1e-6 against R written out; the free fit agrees with brglm2's."""
    frame = _separated_cohort()
    model = _separated_fit(frame, tmp_path / "e", k).data["models"][0]
    assert model["inference"]["covariance"] == "profile" and "s" in model["inference"]["separated"]
    tests = {t["test"]: t for t in model["exposure_tests"] if t["column"] == "x"}
    rows = {r["feature"]: r for r in [*(model.get("coefficients") or []),
                                      *(model.get("adjustment_terms") or [])]}
    knots = tests["overall"]["knots"]
    assert len(knots) == k
    found = run_r(FIRTH_R + f"""
d <- read.csv(frame_csv)
B <- Hmisc::rcspline.eval(d$x, knots = c({", ".join(repr(float(v)) for v in knots)}),
                          inclx = TRUE)
X <- cbind(1, B, d$z, d$s); y <- d$b
colnames(X) <- paste0("c", seq_len(ncol(X)))
k <- {k}
top <- fit()
bg <- glm(y ~ X - 1, family = binomial, method = brglm2::brglmFit, type = "AS_mean",
          control = list(epsilon = 1e-12, maxit = 500))
out(list(overall = 2 * (top$value - fit(2:k, rep(0, k - 1), top$beta)$value),
         nonlinear = 2 * (top$value - fit(3:k, rep(0, k - 2), top$beta)$value),
         beta = top$beta, brglm = unname(coef(bg))))
""", {"frame": frame}, tmp_path / "r")
    assert np.allclose(found["beta"], found["brglm"], rtol=0, atol=1e-6)
    names = ["(intercept)", *ef.spline_names("x", k), "z", "s"]
    for name, b in zip(names, found["beta"]):
        assert rows[name]["estimate"] == pytest.approx(b, abs=1e-6), name
    from scipy import stats

    for kind, q in (("overall", k - 1), ("nonlinear", k - 2)):
        ours = tests[kind]
        assert (ours["distribution"], ours["df_num"], ours["df_den"]) == ("chi2", q, None)
        assert ours["statistic"] == pytest.approx(found[kind], rel=1e-6, abs=1e-8)
        assert ours["p"] == pytest.approx(stats.chi2.sf(found[kind], q), rel=1e-5, abs=1e-12)
    assert tests["overall"]["caption"].startswith(
        "The test of association: Penalized likelihood-ratio test that all ")
    assert not any("need a Wald covariance" in c for c in model.get("concerns") or [])


@pytest.fixture(scope="module")
def separated_modification(tmp_path_factory) -> dict[str, Any]:
    from turbotab.core.tests.acceptance.test_form_7_modification import _run

    frame = est_f.cohort(n=700, seed=23)
    rng = np.random.default_rng(230)
    frame["s"] = (rng.random(len(frame)) < 0.07).astype(float)
    frame.loc[frame["s"] == 1, "dm"] = "yes"
    roles = {**est_f.ROLES, "s": "covariate"}
    st = est_f.state(target="dm", task="binary", event="yes", measure="odds_ratio", roles=roles,
                     answers={**est_f.ANSWERS, "s": est_f.CONFOUNDER},
                     amounts={**est_f.AMOUNTS, "code_or_count:s": "amount"},
                     modifications={"sex": d.ModificationSpec(kind="effect_modification",
                                                              exposure="fiber")})
    folder = tmp_path_factory.mktemp("separated_em")
    return {"frame": frame, "state": st, "out": _run(frame, folder, st)}


@needs_r
def test_r5_under_separation_the_modification_reports_profile_intervals_and_a_lr_test(
        separated_modification, tmp_path):
    """Before: "It could not be fit: the model gave no covariance to test the product terms on",
    while the family counted the test. Expected: the effect within each sex (each a single
    coefficient once sex's reference is chosen: female, then male, in R) with its profile
    penalized-likelihood interval and p-value; the heterogeneity test the penalized
    likelihood-ratio test of the product term; the RERI's point estimate from R's coefficients,
    with no interval; the sentence says each."""
    frame = separated_modification["frame"]
    found = run_r(FIRTH_R + """
d <- read.csv(frame_csv)
y <- as.integer(d$dm == "yes")
res <- list()
for (ref in c("female", "male")) {
  d$sx <- relevel(factor(d$sex), ref = ref)
  X <- model.matrix(~ fiber * sx + age + smoking + activity + s, data = d)
  top <- fit()
  j <- which(colnames(X) == "fiber")
  inter <- grep("^fiber:sx", colnames(X))
  res[[ref]] <- list(ci = profile_ci(j, top), beta = setNames(as.list(top$beta), colnames(X)),
                     het = 2 * (top$value - fit(inter, 0, top$beta)$value))
}
out(res)
""", {"frame": frame}, tmp_path)
    (result,) = separated_modification["out"]["modifications"]
    (fit,) = result["families"]
    assert fit["penalized"] and fit["measure"] == "odds ratio"
    a0, a1 = result["low"], result["high"]
    assert (a0, a1) == tuple(pytest.approx(v, abs=1e-12) for v in np.quantile(
        frame["fiber"].to_numpy(float), [0.25, 0.75], method="linear"))
    delta = a1 - a0
    effects = {q["stratum"]: q for q in fit["effects"]}
    for stratum in ("female", "male"):
        low, high, p = found[stratum]["ci"]
        b = found[stratum]["beta"]["fiber"]
        q = effects[stratum]
        assert q["estimate"] == pytest.approx(math.exp(delta * b), rel=1e-5), stratum
        assert q["ci_low"] == pytest.approx(math.exp(delta * low), rel=1e-4), stratum
        assert q["ci_high"] == pytest.approx(math.exp(delta * high), rel=1e-4), stratum
        assert q["p"] == pytest.approx(p, rel=1e-3, abs=1e-10), stratum
    het = fit["heterogeneity"]
    assert (het["distribution"], het["df_num"]) == ("chi2", 1)
    assert het["statistic"] == pytest.approx(found["female"]["het"], rel=1e-5, abs=1e-8)
    assert het["statistic"] == pytest.approx(found["male"]["het"], rel=1e-5, abs=1e-8)
    b = found["female"]["beta"]
    L10 = delta * b["fiber"]
    L01 = b["sxmale"] + b["fiber:sxmale"] * a0
    L11 = L10 + b["sxmale"] + b["fiber:sxmale"] * a1
    (reri,) = fit["reri"]
    assert reri["estimate"] == pytest.approx(math.exp(L11) - math.exp(L10) - math.exp(L01) + 1,
                                             rel=1e-5)
    assert (reri["ci_low"], reri["ci_high"]) == (None, None)
    assert "the heterogeneity test is the penalized likelihood-ratio test that the 1 product " \
           "term is zero" in result["sentence"]
    assert "its point estimate only" in result["sentence"]
    assert "delta-method interval (Hosmer" not in result["sentence"]
    count = separated_modification["out"]["family_count"]
    assert count["n_tests"] == 2 and count["not_computed"] == []


@needs_r
def test_r5_under_multiple_imputation_the_penalized_tests_are_pooled_by_d2_as_mitml_pools_them(
        tmp_path):
    """Each imputed copy's penalized likelihood-ratio test has no covariance for D1, so the
    copies' χ² statistics are pooled by D2 (Li, Meng, Raghunathan & Rubin 1991), against
    ``mitml``'s own D2 (``mitml:::.D2``, which ``testModels(method = "D2")`` runs), and the MI
    pooling of the form tests takes it where D1 cannot run (never dropping the test)."""
    from scipy import stats

    from turbotab.core.stages.modeling import _pool_form_tests

    rng = np.random.default_rng(5)
    copies = [list(rng.chisquare(2, 20) + 3.0) for _ in range(3)]
    found = run_r("""
d <- read.csv(stats_csv)
res <- lapply(split(d$d, d$set), function(x) { r <- mitml:::.D2(x, 2)
  c(r$F, r$v, pf(r$F, 2, r$v, lower.tail = FALSE)) })
out(unname(res))
""", {"stats": pd.DataFrame({"set": np.repeat([0, 1, 2], 20), "d": np.concatenate(copies)})},
                  tmp_path)
    for statistics, (F, nu, p) in zip(copies, found):
        ours = ef.pooled_chi_square(statistics, 2)
        assert (ours["statistic"], ours["df_den"], ours["p"]) == (
            pytest.approx(F, rel=1e-10), pytest.approx(nu, rel=1e-10), pytest.approx(p, rel=1e-8))
    # through the pooling of the form tests: tables with no covariance (Firth's)
    x = rng.gamma(3.0, 2.0, 200)
    step = ef.ExposureForms({"x": {"form": "spline", "knots": 4}}).fit(pd.DataFrame({"x": x}))
    names = ["(intercept)", *ef.spline_names("x", 4)]
    tables = [SimpleNamespace(cov=None, info={"covariance": "profile"},
                              rows=[{"feature": n, "estimate": 0.1} for n in names])
              for _ in copies[0]]
    tests = [[{"column": "x", "form": "spline", "test": "overall", "statistic": s, "df_num": 2,
               "df_den": None, "distribution": "chi2", "p": float(stats.chi2.sf(s, 2)),
               "caption": "Penalized likelihood-ratio test."}] for s in copies[0]]
    fits = [SimpleNamespace(named_steps={"form": step}) for _ in copies[0]]
    (pooled,) = _pool_form_tests(tests, tables, fits, len(copies[0]))
    F, nu, p = found[0]
    assert (pooled["distribution"], pooled["df_num"]) == ("F", 2)
    assert (pooled["statistic"], pooled["df_den"], pooled["p"]) == (
        pytest.approx(F, rel=1e-10), pytest.approx(nu, rel=1e-10), pytest.approx(p, rel=1e-8))
    assert pooled["caption"].startswith("Pooled over 20 imputations by Li, Meng, Raghunathan & "
                                        "Rubin's D2")

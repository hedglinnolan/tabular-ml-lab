"""TRUST · the engine trust leaks the quest-log critique found (calm/QUEST_LOG_CRITIQUE_2026-10-10
§4 item 4), and four methods sentences that contradicted their own exhibits (WAVE_C6A_PLAN §7
ruling 13).

1. A question the fit waits on, stated as why another question does not apply ("Tell me about
   these columns: …" as the causal lane's reason), is a Decide at the readings ask card, never For
   the record.
2. The second swap (``set_substitution``) says it is a second comparison reported beside the
   primary estimate, and its curve is one of Results' exhibits under Estimate.
3. Years and codes are never printed with thousands separators ("2001 to 2017", not "2,001").
4. Explore's collinear finding carries the label the person sees: Decide only when what you study
   is inside the dependency.
5. (a) the band's refits say m of n rows, rescaled, where the table is larger than the band's
   bound; (b) the estimand's sentence says the swap's step in the exposure's own grams; (c) "no row
   is missing" names the values the data's provider imputed; (d) Results counts only the exhibits
   its goal serves.

Every expected sentence and value is written here by hand; the goals of each exhibit are read from
the crosswalk (``crosswalk.json``), which the engine does not read.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd
import pytest

from turbotab.core import decisions as d
from turbotab.core import materiality as M
from turbotab.core import quest, sweep, voice
from turbotab.core.decisions import ProjectState
from turbotab.core.interview import route
from turbotab.core.readings import amounts_by_values
from turbotab.core.tests import path_fuzzer
from turbotab.core.tests.test_c6a_phase1_follow_ups import _sugar_share
from turbotab.core.tests.test_collinear_adjustment import ADJUST, state as collinear_state, table

REPO = Path(__file__).resolve().parents[3]
CROSSWALK = REPO / "docs" / "turbotab-next" / "crosswalk" / "crosswalk.json"

NOT_APPLICABLE = "Each unit appears once, so no exposure changes over time."
# The fit's refusal when `sugar` is confirmed as codes while the energy answer computes with it as
# an amount (``readings.predictors_or_ask``'s clash): written out by hand, with no ledger ask
# ("Tell me about …") in it, so the rule cannot be reading one ask's wording.
CLASH = ("`sugar` is recorded as codes for categories, but the energy answer (residual) computes "
         "with it as amounts. Say which holds: the codes answer or the energy answer.")


def _inference(**update) -> ProjectState:
    base = dict(lens=["dietary"], target="glucose", purpose="inference",
                roles={"sugar": "exposure", "protein": "covariate", "age": "covariate"},
                estimand=d.EstimandSpec(exposure="sugar", measure="mean_difference",
                                        contrast="substitution"))
    return ProjectState(**{**base, **update})


# ── 1 · what something waits on is a Decide ──────────────────────────────────


def _clashing(stop: str = "open_seal"):
    """The NHANES plan answered through ``stop``; then the energy answer becomes the residual
    method with `sugar` among its nutrients, and `sugar` is confirmed as codes: the fit refuses."""
    state, _artifacts, records = path_fuzzer.answered_through(path_fuzzer.NHANES, "inference",
                                                               stop=stop)
    for decision in (d.SetEnergyAdjustment(method="residual", energy_column="kcal",
                                           nutrients=["sugar"]),
                     d.ConfirmReading(reading="code_or_count", column="sugar", value="code")):
        records.append(path_fuzzer._record(records, decision))
    state = d.fold(records)
    artifacts = path_fuzzer.artifacts_for(state, path_fuzzer.NHANES, records)
    # the causal lane's card, as ``stages.causal`` serves it when ``prepare`` meets the refusal
    artifacts["causal_design"] = {"purpose": "inference", "offered": False, "reason": CLASH,
                                  "exposure": "sugar"}
    steps = route(state, {"fit": {"status": "idle"}}, artifacts, records)
    return state, artifacts, records, steps


def _log(state, artifacts, records, steps, asked):
    return quest.quest_log(state, records, steps, {"fit": {"status": "idle"}},
                           findings=artifacts.get("findings"), columns=path_fuzzer.NHANES.columns,
                           artifacts=artifacts, asked=asked)


def test_the_fits_refusal_is_measured_whatever_its_wording():
    state, _artifacts, records, steps = _clashing()
    asked = quest.fit_waits(state)
    assert (asked.question, asked.message) == ("models", CLASH)
    assert "Tell me about" not in asked.message
    # the causal lane's gate reads the card's reason: not applicable, for the fit's refusal
    causal = next(s for s in steps if s.key == "causal")
    assert (causal.status, causal.reason) == ("not_applicable", CLASH)
    assert quest.waits_on_an_answer(causal, [asked])
    # a reason that merely reads like an ask is no refusal measured on this state
    assert not quest.waits_on_an_answer(
        causal.model_copy(update={"reason": "Tell me about this column: `x`: an amount?"}),
        [asked])
    # nothing waits once the plan is answered as the NHANES journey answers it
    settled, _a, _r = path_fuzzer.answered_through(path_fuzzer.NHANES, "inference")
    assert quest.fit_waits(settled) is None


def test_a_refusal_the_fit_waits_on_is_a_decide_in_models_never_for_the_record():
    state, artifacts, records, steps = _clashing()
    asked = quest.fit_waits(state)
    assert next(s for s in steps if s.key == "models").status == "answered"
    log = _log(state, artifacts, records, steps, [asked])
    [(stage, line)] = [(s.key, l) for s in log.stages for l in s.lines if l.key == "ask:models"]
    assert (stage, line.id, line.label, line.status, line.counted, line.name, line.reason) == (
        "models", "decision:readings-ask-card", "Decide", "open", True, CLASH, CLASH)
    # its ways forward are the refusal's own, under the reading it waits on
    [item] = line.items
    assert (item.kind, item.column, item.value) == ("code_or_count", "sugar", "code")
    assert [o.label for o in item.options] == [
        "`sugar` is an amount, as the energy answer reads it",
        "Change the energy answer (the energy question)"]
    assert item.options[0].decision == {"kind": "confirm_reading", "reading": "code_or_count",
                                        "column": "sugar", "value": "amount"}
    # and it sits just ahead of the families question it reopens
    models = next(s for s in log.stages if s.key == "models")
    keys = [l.key for l in models.lines]
    assert keys.index("ask:models") == keys.index("models") - 1
    # For the record leaves the refusal out; a question that does not apply stays, with its reason
    record = sweep.for_the_record(log, quest.record_steps(steps, [asked]), records)
    texts = [(s.key, r.kind, r.text) for s in record.stages for r in s.lines]
    assert not [t for t in texts if t[2] == CLASH]
    assert [t for t in texts if t[1] == "not_applicable" and t[2].startswith("Each unit appears")]
    # without the measured refusal it was there, as the critique found it
    bare = sweep.for_the_record(log, steps, records)
    assert ("models", "not_applicable", CLASH) in [
        (s.key, r.kind, r.text) for s in bare.stages for r in s.lines]


def test_the_decide_says_why_models_gained_a_line_and_your_data_keeps_its_count():
    state, artifacts, records, steps = _clashing()
    asked = quest.fit_waits(state)
    before = {s.key: s for s in _log(state, artifacts, records, steps, []).stages}
    after = {s.key: s for s in _log(state, artifacts, records, steps, [asked]).stages}
    # Your data is not touched (I11: no line gained there)
    assert after["data"].progress == before["data"].progress
    assert after["data"].reopened == before["data"].reopened
    # Models gains one open line, and names its cause: `sugar` confirmed as codes, in Your data
    # (r14), the last change to what the families question rests on after it was answered (r12)
    b, a = before["models"].progress, after["models"].progress
    assert (a.answered, a.required, a.complete) == (b.answered, b.required + 1, False)
    [line] = [l for l in after["models"].lines if l.key == "ask:models"]
    assert line.reopened_by == quest.ReopenedBy(decision_id="r14", kind="confirm_reading",
                                                stage="data")
    [reason] = [r for r in after["models"].reopened if r.decision_id == "r14"]
    assert reason.questions == ["decision:readings-ask-card"]
    assert reason.sentence == "Your change to Your data reopened 1 question in Models."


def test_while_the_families_question_is_open_the_refusal_is_a_decide_beside_it():
    # The clash holds settled readings, so no ask card rides on the open families question
    # (``ask.card`` asks only unsettled ones): the refusal is its own Decide, with no cause, since
    # nothing in Models was answered before it.
    state, artifacts, records, steps = _clashing(stop="models")
    models = next(s for s in steps if s.key == "models")
    assert models.status == "open" and models.ask is None
    log = _log(state, artifacts, records, steps, [quest.fit_waits(state)])
    lines = [l for s in log.stages for l in s.lines]
    [line] = [l for l in lines if l.key == "ask:models"]
    assert (line.label, line.status, line.reopened_by) == ("Decide", "open", None)
    assert [(l.label, l.status) for l in lines if l.key == "models"] == [("Decide", "open")]


def test_an_ask_card_on_the_open_families_question_is_the_one_decide():
    # A whole-number covariate whose codes-or-amounts reading nobody settled: the ask card on the
    # open families question asks it, so no second line repeats it.
    state, _artifacts, records = path_fuzzer.answered_through(path_fuzzer.NHANES, "inference",
                                                               stop="models")
    info = {"age": {"dtype": "integer", "n_unique": 68}}
    asked = quest.fit_waits(state, info)
    assert asked is not None and "`age`" in asked.message
    from turbotab.core.ask import AskContext

    artifacts = path_fuzzer.artifacts_for(state, path_fuzzer.NHANES, records)
    steps = route(state, {"fit": {"status": "idle"}}, artifacts, records,
                  ask=AskContext(state, artifacts, column_info=info))
    models = next(s for s in steps if s.key == "models")
    assert models.status == "open" and models.ask is not None
    log = _log(state, artifacts, records, steps, [asked])
    assert [l.key for s in log.stages for l in s.lines if l.key.startswith("ask:")] == []


# ── 2 · the second swap ──────────────────────────────────────────────────────


def test_the_second_swap_says_it_is_a_second_comparison_beside_the_primary():
    swap = d.SetSubstitution(donor="sugar", recipient="protein", step_kcal=100.0)
    assert voice.sentence_for(swap, _inference()) == (
        "Beside the primary estimate, a second comparison is reported: `sugar` replaced by "
        "`protein`, in steps of `100` kcal at the same total energy.")
    # under Predict there is no primary estimate: the swap is the comparison studied
    predict = _inference(purpose="prediction", estimand=None)
    assert voice.sentence_for(swap, predict) == (
        "The substitution studied is `sugar` replaced by `protein`, in steps of `100` kcal at the "
        "same total energy.")


def test_the_substitution_curve_is_an_exhibit_of_results_under_estimate():
    state, _artifacts, _records = path_fuzzer.answered_through(path_fuzzer.NHANES, "inference")
    swapped = state.model_copy(update={"substitution": d.SubstitutionSpec(
        donor="sugar", recipient="protein")})
    assert "substitution" in quest.served_exhibits(swapped)
    assert "substitution" not in quest.served_exhibits(state)  # no swap declared, no curve


# ── 3 · years and codes are never grouped ────────────────────────────────────


def test_years_and_codes_are_printed_as_the_data_hold_them():
    years = pd.Series([2001, 2003, 2005, 2007, 2009, 2011, 2013, 2015, 2017] * 20)
    assert amounts_by_values(years).evidence == "`9` whole-number values from 2001 to 2017"
    fips = pd.Series([6037, 17031, 36061, 48201] * 30)
    assert amounts_by_values(fips).evidence == "`4` whole-number values from 6037 to 48201"


# ── 4 · Explore's collinear finding carries its label ────────────────────────


def _explore(*findings):
    return {"findings": [{"id": f"explore::{k}", "kind": k, "summary": "", "columns": c,
                          "view": "table_focus"} for k, c in findings]}


def test_the_collinear_finding_is_served_for_the_record_when_what_you_study_is_outside_it():
    from turbotab.core.stages.explore import ExploreFinding
    from turbotab.server.service import explore_labeled

    assert ExploreFinding(id="x", kind="wide", summary="", columns=[], view="table_focus"
                          ).label is None
    f = table()
    assert _sugar_share(f, ADJUST) < 0.5  # by hand: sugar outside the near dependency
    s = collinear_state(ADJUST)
    served = explore_labeled(_explore(("collinear", ["fat", "protein"]), ("low_variance", ["x"])),
                             s, lambda: M.collinear_noticing(s, f))
    assert [x["label"] for x in served["findings"]] == ["For the record", "Confirm"]


def test_the_collinear_finding_is_served_as_a_decide_when_what_you_study_is_inside_it():
    from turbotab.server.service import explore_labeled

    f = table(sugar_inside=True)
    columns = [*ADJUST, "starch"]
    assert _sugar_share(f, columns) >= 0.5  # by hand: sugar inside the near dependency
    s = collinear_state(columns)
    served = explore_labeled(_explore(("collinear", ["carb", "sugar"])), s,
                             lambda: M.collinear_noticing(s, f))
    assert served["findings"][0]["label"] == "Decide"


def test_under_predict_the_pair_scan_is_a_decide_and_nothing_is_measured():
    from turbotab.server.service import explore_labeled

    def never():
        raise AssertionError("measured under Predict")

    served = explore_labeled(_explore(("collinear", ["fat", "protein"])),
                             collinear_state(ADJUST, purpose="prediction"), never)
    assert served["findings"][0]["label"] == "Decide"


# ── 5 · methods sentences that agree with their exhibits ─────────────────────

SWAP_HEAD = ("Beside the primary estimate, a second comparison is reported: `sugar` replaced by "
             "`protein`, in steps of `100` kcal at the same total energy; its band comes from "
             "`200` refits of each model on bootstrap resamples of ")
RESCALED = ("drawn as whole units where a unit has several rows, the band's spread {then}rescaled "
            "by √(m/n) to the full sample (an m-out-of-n bootstrap).")


def test_the_band_says_m_of_n_rescaled_where_the_analyzed_rows_exceed_its_bound():
    from turbotab.core.stages.modeling import BAND_ROWS

    assert voice.BAND_ROWS == BAND_ROWS == 10_000
    swap = d.SetSubstitution(donor="sugar", recipient="protein", step_kcal=100.0, n_boot=200)

    def said(ctx, state=None):
        return voice.sentence_for(swap, state or _inference(), ctx)

    # 21,849 analyzed rows: each resample draws about 10,000 of them, rescaled (the caption's m of n)
    assert said({"n_analyzed": 21_849}) == SWAP_HEAD + (
        "about `10,000` of the `21,849` analyzed rows each, " + RESCALED.format(then=""))
    # at most the bound: every analyzed row, whether the analyzed rows or the table say so
    every = SWAP_HEAD + "every analyzed row."
    assert said({"n_analyzed": 400}) == said({"n_rows": 400}) == said({"n_cohort": 400}) == every
    # not known (the record written with no counts), or only the table's 30,000 rows known: both
    # cases, each with its condition, never "every analyzed row" alone
    either = SWAP_HEAD + ("every analyzed row, or, above `10,000` analyzed rows, of about `10,000` "
                          "of them each, " + RESCALED.format(then="then "))
    assert said({}) == said({"n_rows": 30_000}) == either
    # under Predict the band reads the training rows, whose number the analyzed rows do not give
    predict = _inference(purpose="prediction", estimand=None)
    assert said({"n_analyzed": 21_849}, predict).endswith(
        "resamples of training rows, or, above `10,000` training rows, of about `10,000` of them "
        "each, " + RESCALED.format(then="then "))


def _swapped(column: str, unit: str | None):
    units = {column: d.ColumnUnitSpec(unit=unit)} if unit else None
    return _inference(roles={column: "exposure", "protein": "covariate"},
                      estimand=d.EstimandSpec(exposure=column, measure="mean_difference",
                                              contrast="substitution"),
                      substitution=d.SubstitutionSpec(donor=column, recipient="protein"),
                      column_units=units)


@pytest.mark.parametrize("column, unit, said", [
    # 100 kcal at Atwater's 4 kcal per g of sugar is 25 g; at 9 kcal per g of fat, 11.1 g;
    # at 4,000 kcal per kg, 0.025 kg
    ("sugar", "g", "; `100` kcal, the second comparison's step, is `25` g of `sugar` at `4` kcal "
                   "per g"),
    ("fat_total", "g", "; `100` kcal, the second comparison's step, is `11.1` g of `fat_total` at "
                       "`9` kcal per g"),
    ("sugar", "kg", "; `100` kcal, the second comparison's step, is `0.025` kg of `sugar` at "
                    "`4000` kcal per kg"),
])
def test_the_estimand_sentence_says_the_second_comparisons_step_in_the_settled_unit(column, unit,
                                                                                    said):
    estimand = d.SetEstimand(exposure=column, measure="mean_difference", contrast="substitution")
    text = voice.sentence_for(estimand, _swapped(column, unit))
    assert f"per unit of `{column}`{said}" in text


def test_with_no_settled_unit_or_no_swap_moving_it_no_step_is_said():
    estimand = d.SetEstimand(exposure="sugar", measure="mean_difference", contrast="substitution")
    assert "the second comparison's step" in voice.sentence_for(estimand, _swapped("sugar", "g"))
    # the name "sugar" settles no unit (``readings.kcal_per_unit``: never a name)
    assert "step" not in voice.sentence_for(estimand, _swapped("sugar", None))
    # no second comparison declared, or one that moves other nutrients
    state = _swapped("sugar", "g")
    assert "step" not in voice.sentence_for(estimand, state.model_copy(
        update={"substitution": None}))
    assert "step" not in voice.sentence_for(estimand, state.model_copy(update={
        "substitution": d.SubstitutionSpec(donor="fat_total", recipient="protein")}))


def _adjusted_with_a_mediator():
    """The NHANES plan, with `bp_sys` answered as changed by sugar (a mediator: no model reads it
    under a total effect) and `waist` of unknown timing (Model 3, the further-adjusted model)."""
    state, _artifacts, records = path_fuzzer.answered_through(path_fuzzer.NHANES, "inference")
    roles = next(r.decision for r in records if r.decision.kind == "set_roles")
    records.append(path_fuzzer._record(records, roles.model_copy(update={"roles": {
        **roles.roles, "bp_sys": "covariate", "waist": "covariate"}})))
    records.append(path_fuzzer._record(records, d.SetAdjustment(exposure="sugar", answers={
        "bp_sys": d.CovariateAnswers(causes_exposure="no", causes_outcome="yes",
                                     after_exposure="yes"),
        "waist": d.CovariateAnswers(causes_exposure="no", causes_outcome="yes",
                                    after_exposure="unknown")})))
    return d.fold(records)


def test_no_row_missing_names_the_imputed_values_of_the_columns_a_model_reads():
    state = _adjusted_with_a_mediator()
    frame = pd.DataFrame({"imputed_bp_sys": [True, False, False, False, False],
                          "imputed_waist": [False, True, False, False, False],
                          "imputed_bmi": [False] * 5})
    said = voice.sentence_for(d.SetMissing(strategy="complete_case"), state,
                              {"n_complete": 5, "n_before": 5, "frame": frame})
    # `bp_sys` (no model reads it) is not named; `waist` (Model 3) is; `bmi` holds no flagged value
    assert said == ("A complete-case analysis was applied: no row is missing any predictor, so "
                    "all `5` rows remain; `waist` holds values the data's provider imputed before "
                    "the file arrived (flagged in `imputed_waist`), analyzed as recorded.")
    # no value flagged: nothing to say
    clean = voice.sentence_for(d.SetMissing(strategy="complete_case"), state,
                               {"n_complete": 5, "n_before": 5,
                                "frame": frame.assign(imputed_waist=False)})
    assert "imputed" not in clean


def test_a_flag_only_on_rows_screened_out_names_nothing():
    state = _adjusted_with_a_mediator()
    # six rows in the table; the sixth, the only one with `waist` imputed, was screened out
    frame = pd.DataFrame({"imputed_waist": [False] * 5 + [True]}, index=range(6))
    remain = {"n_complete": 5, "n_before": 5, "frame": frame}
    said = voice.sentence_for(d.SetMissing(strategy="complete_case"), state,
                              {**remain, "row_ids": [0, 1, 2, 3, 4]})
    assert "imputed" not in said
    # with the rows that remain unknown, a table that holds more rows than remain says nothing
    assert "imputed" not in voice.sentence_for(d.SetMissing(strategy="complete_case"), state,
                                               remain)
    # the same flag on a row that remains is named
    named = voice.sentence_for(d.SetMissing(strategy="complete_case"), state,
                               {**remain, "row_ids": [1, 2, 3, 4, 5]})
    assert named.endswith("; `waist` holds values the data's provider imputed before the file "
                          "arrived (flagged in `imputed_waist`), analyzed as recorded.")


def _crosswalk_goals() -> dict[str, list[str]]:
    data = json.loads(CROSSWALK.read_text())
    items = data["items"] if isinstance(data, dict) and "items" in data else data
    return {i["id"]: (i["goals"] if isinstance(i["goals"], list) else ast.literal_eval(i["goals"]))
            for i in items}


def test_each_exhibit_serves_the_goals_its_crosswalk_card_names():
    goals = _crosswalk_goals()
    for name, (_stage, card) in quest.COMPUTE.items():
        if card is not None and card.startswith("exhibit:"):
            assert sorted(quest.EXHIBITS[name]) == sorted(goals[card]), name
    # the curve has no card of its own: its question's goals
    assert sorted(quest.EXHIBITS["substitution"]) == sorted(goals["decision:substitution-pair"])


def test_results_under_estimate_counts_only_the_estimate_exhibits():
    # NHANES under Estimate is served usual_intake (Describe), fit and evaluation (Predict),
    # secondary and effects (Estimate): only the last two are Results' objectives.
    state, _artifacts, _records = path_fuzzer.answered_through(path_fuzzer.NHANES, "inference")
    assert quest.goal_of(state) == "inference"
    assert sorted(quest.served_exhibits(state)) == ["effects", "secondary"]


def test_a_goal_with_nothing_of_its_own_to_serve_is_complete_at_0_of_0():
    # Describe (no estimand) on the NHANES plan without the dietary lens: usual intake is the only
    # Describe exhibit, and the engine computes it under any lens, so take the families away too:
    # Results then has no exhibit served and none that waits only on what the fit waits on.
    state, _artifacts, _records = path_fuzzer.answered_through(path_fuzzer.NHANES, "inference")
    describe = state.model_copy(update={"estimand": None})
    assert quest.goal_of(describe) == "describe"
    assert quest.served_exhibits(describe) == ("usual_intake",)
    assert quest.exhibits_waiting(describe) == ()
    # Under Estimate with the families withdrawn, Table 2 and Model 3 wait only on them
    withdrawn = state.model_copy(update={"models": None})
    assert quest.served_exhibits(withdrawn) == ()
    assert sorted(quest.exhibits_waiting(withdrawn)) == ["effects", "secondary"]

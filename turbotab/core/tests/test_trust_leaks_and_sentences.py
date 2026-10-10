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

# The causal lane's reason on the critique's screen (the m3 NHANES journey's ``/record``): the
# fit's refusal, carried as why the causal question does not apply.
WAITING = ("`age` and `cycle_begin_year` hold whole numbers, which may be codes for categories "
           "(one indicator per level) or amounts (one slope); the fit waits for the answer for "
           "each. Tell me about these columns: `age`: an amount? (`68` whole-number values from "
           "18 to 85); `cycle_begin_year`: codes for categories? (`9` whole-number values from "
           "2001 to 2017, few of them).")
ASK = ("Tell me about these columns: `age`: an amount? (`68` whole-number values from 18 to 85); "
       "`cycle_begin_year`: codes for categories? (`9` whole-number values from 2001 to 2017, few "
       "of them).")
NOT_APPLICABLE = "Each unit appears once, so no exposure changes over time."


def _inference(**update) -> ProjectState:
    base = dict(lens=["dietary"], target="glucose", purpose="inference",
                roles={"sugar": "exposure", "protein": "covariate", "age": "covariate"},
                estimand=d.EstimandSpec(exposure="sugar", measure="mean_difference",
                                        contrast="substitution"))
    return ProjectState(**{**base, **update})


# ── 1 · a question something waits on is a Decide ────────────────────────────


def _nhanes_with(reasons: dict[str, str]):
    state, artifacts, records = path_fuzzer.answered_through(path_fuzzer.NHANES, "inference")
    steps = route(state, {"fit": {"status": "idle"}}, artifacts, records)
    steps = [s.model_copy(update={"status": "not_applicable", "reason": reasons[s.key]})
             if s.key in reasons else s for s in steps]
    log = quest.quest_log(state, records, steps, {"fit": {"status": "idle"}},
                          findings=artifacts.get("findings"), columns=path_fuzzer.NHANES.columns,
                          artifacts=artifacts)
    return log, steps, records


def test_a_question_the_fit_waits_on_is_a_decide_at_the_ask_card_not_for_the_record():
    log, steps, records = _nhanes_with({"causal": WAITING, "time_varying": NOT_APPLICABLE})
    lines = [(s.key, l) for s in log.stages for l in s.lines]
    [(stage, line)] = [(k, l) for k, l in lines if l.name == ASK]
    assert (stage, line.id, line.label, line.status, line.counted) == (
        "data", "decision:readings-ask-card", "Decide", "open", True)
    assert line.reason == WAITING
    record = sweep.for_the_record(log, quest.record_steps(steps), records)
    texts = [(s.key, r.kind, r.text) for s in record.stages for r in s.lines]
    assert not [t for t in texts if "Tell me about" in t[2]]
    # a question that does not apply stays For the record, with its reason
    assert ("models", "not_applicable", NOT_APPLICABLE) in texts


def test_the_rule_reads_the_ask_wherever_it_sits_and_one_ask_is_one_line():
    log, _steps, _records = _nhanes_with({"causal": WAITING, "time_varying": WAITING})
    asked = [l for s in log.stages for l in s.lines if l.id == "decision:readings-ask-card"]
    assert [l.name for l in asked] == [ASK]
    assert quest.asks("Tell me about this column: `x`: an amount?") == (
        "Tell me about this column: `x`: an amount?")
    assert quest.asks(NOT_APPLICABLE) is None


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


def test_the_band_says_m_of_n_rescaled_where_the_table_is_larger_than_its_bound():
    from turbotab.core.stages.modeling import BAND_ROWS

    assert voice.BAND_ROWS == BAND_ROWS == 10_000
    swap = d.SetSubstitution(donor="sugar", recipient="protein", step_kcal=100.0, n_boot=200)
    big = voice.sentence_for(swap, _inference(), {"n_rows": 21_849})
    assert big.endswith(
        "; its band comes from `200` refits of each model on bootstrap resamples of the analyzed "
        "rows, each resample drawing `10,000` of them when there are more, with the spread "
        "rescaled to the full sample (an m-out-of-n bootstrap).")
    small = voice.sentence_for(swap, _inference(), {"n_rows": 400})
    assert small.endswith("; its band comes from `200` refits of each model on bootstrap "
                          "resamples of every analyzed row.")


@pytest.mark.parametrize("column, said", [
    # 100 kcal at 4 kcal/g is 25 g; at 9 kcal/g, 11.1 g
    ("sugar", "; the swap studied moves `100` kcal, which is `25` g of `sugar` at `4` kcal per g, "
              "if it is recorded in grams"),
    ("sugar_g", "; the swap studied moves `100` kcal, which is `25` g of `sugar_g` at `4` kcal "
                "per g"),
    ("fat_total", "; the swap studied moves `100` kcal, which is `11.1` g of `fat_total` at `9` "
                  "kcal per g, if it is recorded in grams"),
])
def test_the_estimand_sentence_says_the_swaps_step_in_the_exposures_grams(column, said):
    state = _inference(roles={column: "exposure", "protein": "covariate"},
                       estimand=d.EstimandSpec(exposure=column, measure="mean_difference",
                                               contrast="substitution"),
                       substitution=d.SubstitutionSpec(donor=column, recipient="protein"))
    text = voice.sentence_for(d.SetEstimand(exposure=column, measure="mean_difference",
                                            contrast="substitution"), state)
    assert f"per unit of `{column}`{said}" in text
    # with no swap declared, the step is not known: nothing is said of it
    bare = voice.sentence_for(d.SetEstimand(exposure=column, measure="mean_difference",
                                            contrast="substitution"),
                              state.model_copy(update={"substitution": None}))
    assert "swap studied moves" not in bare


def test_no_row_missing_names_the_values_the_provider_imputed():
    state = ProjectState(target="glucose", purpose="inference",
                         roles={"sugar": "exposure", "weight": "covariate", "waist": "covariate",
                                "imputed_weight": "flag", "imputed_waist": "flag"})
    frame = pd.DataFrame({"imputed_weight": [False, True, False, False],
                          "imputed_waist": [False, False, False, False]})
    said = voice.sentence_for(d.SetMissing(strategy="complete_case"), state,
                              {"n_complete": 4, "n_before": 4, "frame": frame})
    assert said.startswith(
        "A complete-case analysis was applied: no row is missing any predictor, so all `4` rows "
        "remain; `weight` holds values the data's provider imputed before the file arrived "
        "(flagged in `imputed_weight`), analyzed as recorded")
    # no value flagged: nothing to say
    clean = voice.sentence_for(d.SetMissing(strategy="complete_case"), state,
                               {"n_complete": 4, "n_before": 4,
                                "frame": frame.assign(imputed_weight=False)})
    assert "imputed" not in clean


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

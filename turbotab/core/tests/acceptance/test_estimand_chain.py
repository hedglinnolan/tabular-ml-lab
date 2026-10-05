"""ESTIMAND · the chain test (BLUEPRINT §13: "A chain test asserts that every implied consequence
appears in the participant flow, the lineage and the methods sentence"; MODELING_SEQUENCE §6).

Two journeys through the real server (``server_drive``), each answered from its fixture's declared
truth (``truths.Truth``), never a constant:

* **A · dietary, a common yes/no outcome.** Protein (an energy-bearing exposure) and diabetes
  (about 35% of rows), total energy in the model. The estimand card ranks the marginal measures
  first; the estimand is a substitution's marginal risk difference; the covariates are answered by
  the disjunctive cause criterion (BMI of unknown timing: Model 3); Model 1 (age, sex and energy)
  is declared beside the adjustment card, before any estimate; the energy model is the standard
  one; the fit and the effects stage show the exposure only, the adjustment terms apart; the plan
  locks when the first estimate is shown and exports byte-identically, also after the server is
  restarted on the same workspace.
* **B · clinical, a time to event.** A hazard ratio that changes over follow-up, a cause of the
  outcome only among the covariates. The card, the record and the caption say that adjusting for it
  changes the conditional estimand; the proportional-hazards check fails for the exposure, its exit
  is taken, and the response is recorded after the estimates were seen and shown beside them.

Every relation the package's method contracts declare (in the one registry,
``turbotab/core/contracts.py``) is mapped to the test that asserts it fires (:data:`RELATION_TESTS`),
and a contract whose relation no test exercises fails
:func:`test_every_relation_the_contracts_declare_is_asserted`.
"""
from __future__ import annotations

import importlib
import time
from typing import Any

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from turbotab.core import contracts, plan_lock
from turbotab.core.models.effects import APPENDIX_TITLE, BOOT, cook_threshold
from turbotab.core.tests.acceptance.server_drive import local_server, open_project
from turbotab.core.tests.truths import Truth, answer_adjustment

DIET_ROLES = {"pid": "identifier", "age": "covariate", "sex": "covariate", "smoking": "covariate",
              "activity": "covariate", "bmi": "covariate", "kcal": "energy",
              "protein_g": "exposure"}
DIET_TRUTH = {"code_or_count:smoking": "amount", "code_or_count:activity": "amount",
              "code_or_count:pid": "amount", "code_or_count:kcal": "amount", "day_count:kcal": "1",
              "unit:protein_g": "g", "unit:kcal": "kcal",
              "adjust:age": "yes,yes,no", "adjust:sex": "yes,yes,no",
              "adjust:smoking": "yes,yes,no", "adjust:activity": "no,yes,no",
              "adjust:bmi": "unknown,yes,unknown"}


def _diet(n: int = 900, seed: int = 21) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.normal(52, 9, n).round(1)
    sex = rng.choice(["female", "male"], n)
    smoking = rng.binomial(1, 0.3, n)
    activity = rng.integers(0, 8, n)
    kcal = rng.normal(2100, 400, n).round(0)
    protein_g = (0.035 * kcal + rng.normal(0, 10, n) + 2 * smoking).round(1)
    bmi = (27 - 0.03 * (protein_g - 75) + rng.normal(0, 3, n)).round(1)
    logit = (-0.85 + 0.012 * (protein_g - 75) - 0.0004 * (kcal - 2100) + 0.7 * smoking
             + 0.03 * (age - 52) - 0.2 * (activity - 3.5))
    dm = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "yes", "no")
    return pd.DataFrame({"pid": np.arange(1, n + 1), "age": age, "sex": sex, "smoking": smoking,
                         "activity": activity, "bmi": bmi, "kcal": kcal, "protein_g": protein_g,
                         "dm": dm})


def _open(drive: Any, lens: str, target: str, event: str) -> None:
    drive.decide({"kind": "set_lens", "lenses": [lens]})
    drive.reach("target")
    drive.decide({"kind": "set_target", "column": target})
    drive.answer("event", {"kind": "set_event", "column": target, "level": event})


def _seal(drive: Any, roles: dict[str, str]) -> None:
    drive.reach("purpose")
    drive.decide({"kind": "set_purpose", "purpose": "inference"})
    drive.answer("grain", {"kind": "set_grain", "grain": "one_row_per_unit", "id_column": "pid"})
    drive.reach("roles")
    drive.decide_roles(roles)
    drive.answer("exclusions", {"kind": "set_exclusions", "rules": []})
    drive.answer("missing", {"kind": "set_missing", "strategy": "complete_case"})
    drive.answer("split", {"kind": "set_split", "holdout": 0.0, "seed": 0, "folds": 5})


def _fresh(drive: Any, stage: str, until: Any, timeout: float = 240.0) -> dict[str, Any]:
    end = time.monotonic() + timeout
    while True:
        found = drive.artifact(stage)
        if until(found):
            return found
        assert time.monotonic() < end, f"{stage} never reached the state the test waits for"
        time.sleep(0.1)


@pytest.fixture(scope="module")
def diet(tmp_path_factory) -> dict[str, Any]:
    """Journey A, run once; what the server served at each step."""
    folder = tmp_path_factory.mktemp("diet")
    frame = _diet()
    csv = folder / "diet.csv"
    frame.to_csv(csv, index=False)
    truth = Truth(DIET_TRUTH, fixture="the dietary chain")
    seen: dict[str, Any] = {"frame": frame}
    home = folder / "home"
    with local_server(home) as client:
        drive = open_project(client, csv, truth)
        _open(drive, "dietary", "dm", "True")
        _seal(drive, DIET_ROLES)
        assert drive.reach("estimand")["status"] == "open"
        early = drive.post({"kind": "select_models", "models": ["linear"]})
        seen["models_before"] = (early.status_code, early.json()["error"]["code"])
        seen["estimand_card"] = drive.artifact("proposals")["estimand"]
        drive.decide({"kind": "set_estimand", "exposure": "protein_g", "effect": "total",
                      "contrast": "substitution", "measure": "risk_difference"})
        assert drive.reach("adjustment")["status"] == "open"
        card = drive.artifact("proposals")["adjustment"]
        answer_adjustment(drive.post, card, truth)
        proposals = _fresh(drive, "proposals", lambda p: p.get("model_sequence") is not None)
        seen["model_sequence_card"] = proposals["model_sequence"]
        drive.decide(proposals["model_sequence"]["decision"])
        seen["locked_before_estimates"] = bool(drive.view()["state"].get("plan_locked"))
        drive.answer("energy_adjustment", {"kind": "set_energy_adjustment", "method": "standard",
                                           "energy_column": "kcal", "nutrients": ["protein_g"]})
        drive.answer("models", {"kind": "select_models", "models": ["linear"]})
        seen["fit"] = drive.artifact("fit")
        seen["effects"] = drive.artifact("effects")
        view = drive.view()
        seen["view"] = view
        seen["plan"] = [client.get(f"/api/projects/{drive.pid}/plan").content for _ in range(2)]
        service = client.app.state.service
        seen["plan_records"] = plan_lock.plan_export(service.log(drive.pid).records())
        # a change after the lock: sex is answered a cause of neither, so it leaves the primary set
        # and Model 1 (age, sex, kcal) no longer holds: it is declared again, never kept silently
        drive.decide({"kind": "set_adjustment", "exposure": "protein_g", "answers": {"sex": {
            "causes_exposure": "no", "causes_outcome": "no", "after_exposure": "no"}}})
        seen["proposals_after"] = _fresh(
            drive, "proposals", lambda p: (p.get("model_sequence") or {}).get("declared") is None)
        seen["effects_after"] = _fresh(
            drive, "effects", lambda e: "model_1" not in {s["key"] for s in
                                                          e["families"][0]["sequence"]})
        seen["view_after"] = drive.view()
        seen["plan_final"] = client.get(f"/api/projects/{drive.pid}/plan").content
        pid = drive.pid
    with local_server(home) as again:  # the same workspace, a new server: the log replayed
        seen["plan_replayed"] = again.get(f"/api/projects/{pid}/plan").content
    return seen


def _cohort(n: int = 600, seed: int = 31) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.binomial(1, 0.4, n)
    grp = rng.choice(["a", "b", "c"], n)
    activity = rng.integers(0, 8, n)
    rest = 0.3 * x2 + 0.4 * (grp == "b") - 0.15 * (activity - 3.5)
    early, late = np.exp(1.0 * x1 + rest), np.exp(-0.3 * x1 + rest)
    E = rng.exponential(1.0, n)
    T = np.where(E < 0.5 * early, E / early, 0.5 + (E - 0.5 * early) / late)
    C = rng.exponential(2.0, n)
    return pd.DataFrame({"pid": np.arange(1, n + 1), "x1": x1.round(4), "x2": x2, "grp": grp,
                         "activity": activity, "followup_years": np.minimum(T, C).round(4),
                         "event": (T <= C).astype(int)})


@pytest.fixture(scope="module")
def cox(tmp_path_factory) -> dict[str, Any]:
    """Journey B, run once."""
    folder = tmp_path_factory.mktemp("cox")
    frame = _cohort()
    csv = folder / "cohort.csv"
    frame.to_csv(csv, index=False)
    truth = Truth({"code_or_count:x2": "amount", "code_or_count:activity": "amount",
                   "code_or_count:pid": "amount", "adjust:x2": "yes,yes,no",
                   "adjust:grp": "yes,yes,no", "adjust:activity": "no,yes,no"},
                  fixture="the time-to-event chain")
    roles = {"pid": "identifier", "x1": "exposure", "x2": "covariate", "grp": "covariate",
             "activity": "covariate", "followup_years": "time"}
    seen: dict[str, Any] = {"frame": frame}
    with local_server(folder / "home") as client:
        drive = open_project(client, csv, truth)
        _open(drive, "clinical", "event", "1")
        assert drive.reach("follow_up")["status"] == "open"
        drive.decide({"kind": "set_task", "column": "event", "task": "time_to_event"})
        drive.decide({"kind": "set_follow_up", "column": "event", "time_column": "followup_years"})
        _seal(drive, roles)
        drive.reach("estimand")
        seen["estimand_card"] = drive.artifact("proposals")["estimand"]
        drive.decide({"kind": "set_estimand", "exposure": "x1", "measure": "hazard_ratio"})
        drive.reach("adjustment")
        answer_adjustment(drive.post, drive.artifact("proposals")["adjustment"], truth)
        seen["adjustment_card"] = _fresh(
            drive, "proposals", lambda p: "activity" in (p.get("adjustment") or {}).get(
                "answered", {}))["adjustment"]
        drive.answer("models", {"kind": "select_models", "models": ["cox"]})
        seen["fit"] = drive.artifact("fit")
        seen["effects"] = drive.artifact("effects")
        exit_ = seen["effects"]["families"][0]["diagnostics"][0]["exits"][0]["decision"]
        seen["exit"] = exit_
        drive.decide(exit_)
        seen["effects_after"] = _fresh(
            drive, "effects", lambda e: e["families"][0]["diagnostics"][0]["response"] is not None)
        seen["view"] = drive.view()
    return seen


def _records(view: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    return [r for r in view["decisions"] if r["decision"]["kind"] == kind]


# ── journey A ────────────────────────────────────────────────────────────────


def test_a_the_estimand_card_labels_the_measures_and_ranks_the_marginal_ones_first(diet):
    card = diet["estimand_card"]
    share = float((diet["frame"]["dm"] == "yes").mean())
    assert card["prevalence"] == pytest.approx(share) and share > 0.10
    assert [(m["measure"], m["scale"], m["conditioning"], m["collapsible"], m["rank"])
            for m in card["measures"]] == [
        ("risk_difference", "difference", "marginal", True, 1),
        ("risk_ratio", "ratio", "marginal", True, 2),
        ("odds_ratio", "ratio", "conditional", False, 3)]
    assert card["measures"][2]["reason"].startswith("conditional and non-collapsible")


def test_a_no_estimate_before_the_estimand_and_the_display_after_it(diet):
    assert diet["models_before"] == (409, "not_yet")
    assert diet["locked_before_estimates"] is False  # Model 1 was declared before any estimate
    fit = diet["fit"]
    assert fit["withheld"] is None and fit["estimand"]["measure"] == "risk_difference"
    assert "as a marginal risk difference per unit of `protein_g`" in fit["estimand"]["caption"]
    assert fit["estimand"]["caption"].count("whose conditional odds ratio is shown beside it") == 1
    assert diet["effects"]["measure"] == "risk_difference"
    assert diet["view"]["state"]["plan_locked"] is True  # the first estimate shown locked the plan


def test_a_the_results_show_the_exposure_only_with_the_adjustment_terms_apart(diet):
    model = diet["fit"]["models"][0]
    assert [r["feature"] for r in model["coefficients"]] == ["protein_g"]
    assert model["coefficients"][0]["ratio"] is not None  # the conditional odds ratio, beside
    assert {r["feature"] for r in model["adjustment_terms"]} == {
        "(intercept)", "age", "sex_male", "smoking", "activity", "kcal"}
    assert diet["fit"]["estimand"]["appendix"] == APPENDIX_TITLE
    family = diet["effects"]["families"][0]
    seq = {s["key"]: s for s in family["sequence"]}
    assert list(seq) == ["crude", "model_1", "model_2", "model_3"]
    assert all([r["feature"] for r in s["effects"]] == ["protein_g"] for s in seq.values())
    assert seq["model_1"]["adjusted_for"] == ["age", "sex", "kcal"]
    assert seq["model_2"]["adjusted_for"] == ["age", "sex", "smoking", "activity", "kcal"]
    assert seq["model_3"]["adjusted_for"] == ["age", "sex", "smoking", "activity", "kcal", "bmi"]
    assert seq["model_3"]["note"] == "further adjusted for `bmi`, a possible mediator: not a total effect."
    # the primary agrees with statsmodels on the same rows (the model's conditional log-odds)
    frame = diet["frame"]
    X = pd.DataFrame({"const": 1.0, "protein_g": frame["protein_g"], "age": frame["age"],
                      "sex_male": (frame["sex"] == "male").astype(float),
                      "smoking": frame["smoking"], "activity": frame["activity"],
                      "kcal": frame["kcal"]})
    y = (frame["dm"] == "yes").astype(float)
    reference = sm.Logit(y, X).fit(disp=0, method="newton", tol=1e-12)
    assert seq["model_2"]["effects"][0]["estimate"] == pytest.approx(reference.params["protein_g"],
                                                                     rel=1e-6)


def test_a_the_marginal_risks_are_standardized_with_a_whole_chain_bootstrap(diet):
    marginal = diet["effects"]["families"][0]["marginal"]
    [c] = marginal["contrasts"]
    assert marginal["declared"] == "risk_difference" and marginal["refused"] is None
    assert c["n_boot"] == BOOT and c["n_failed"] == 0 and c["rd_low"] < c["rd"] < c["rd_high"]
    assert "refit the whole chain" in marginal["method"]


def test_a_sensitivity_is_offered_with_no_threshold(diet):
    [s] = diet["effects"]["families"][0]["sensitivity"]
    assert s["methods"] == ["e_value"] and s["e_value"]["measure"] == "RR"
    assert s["reading"].startswith("E-value for the marginal risk ratio:")
    assert "never as a pass or a fail" in s["reading"]


def test_a_the_methods_sentence_is_written_from_what_was_fitted(diet):
    """Asserted verbatim; its numbers come from outside the stage: the rows, the bootstrap's size,
    and Cook's reference and the count above it from statsmodels' GLM influence."""
    frame = diet["frame"]
    X = pd.DataFrame({"const": 1.0, "age": frame["age"],
                      "sex_male": (frame["sex"] == "male").astype(float),
                      "smoking": frame["smoking"], "activity": frame["activity"],
                      "kcal": frame["kcal"], "protein_g": frame["protein_g"]})
    y = (frame["dm"] == "yes").astype(float)
    glm = sm.GLM(y, X, family=sm.families.Binomial()).fit(tol=1e-12)
    cooks = glm.get_influence().cooks_distance[0]
    p, n = X.shape[1], len(frame)
    above = int(np.sum(cooks > cook_threshold(p, n)))
    expected = (
        f"The estimate of `protein_g` is reported across a declared sequence of models fit on all "
        f"{n:,} analyzed rows: unadjusted; Model 1, adjusted for `age`, `sex` and `kcal`; Model 2, "
        f"the primary, adjusted for `age`, `sex`, `smoking`, `activity` and `kcal`; Model 3, further "
        f"adjusted for `bmi`, a possible mediator, so not a total effect. Only the exposure's "
        f"estimates are shown as effects; every other coefficient is listed apart as an adjustment "
        f"term, not an effect estimate (Westreich & Greenland 2013, Am J Epidemiol 177:292). The "
        f"marginal risk difference and risk ratio (one unit higher than observed) were estimated by "
        f"standardization over the analyzed rows (g-computation) from the logistic model, with 95% "
        f"percentile intervals from {BOOT:,} bootstrap resamples refitting the whole chain. "
        f"Influence was checked by Cook's distance against the median of F({p}, {n - p:,}) (Cook "
        f"1977, Technometrics 19:15): {above:,} row{'s' if above != 1 else ''} above it. "
        f"Sensitivity to unmeasured confounding is reported by the E-value for the estimate and "
        f"for the confidence limit nearer the null, never as a pass or a fail.")
    assert diet["effects"]["methods"] == expected


def test_a_the_record_states_each_decision_in_a_sentence(diet):
    view = diet["view"]
    [estimand] = _records(view, "set_estimand")
    assert estimand["sentence"] == (
        "The analysis estimates the total effect of `protein_g` on `dm` (a substitution: in place "
        "of other energy sources at fixed total energy), as a marginal risk difference per unit of "
        "`protein_g`, by standardization over the analyzed rows (g-computation) from the logistic "
        "model, with a bootstrap of the whole chain for its interval, the model's conditional odds "
        "ratio reported beside it.")
    [sequence] = _records(view, "set_model_sequence")
    assert sequence["sentence"] == (
        "The estimate of `protein_g` is reported in a declared sequence of models: unadjusted; "
        "Model 1, adjusted for `age`, `sex` and `kcal`; Model 2, the primary, with the full "
        "adjustment set; and Model 3, further adjusted for any possible mediators the answers set "
        "beside the primary, which is not a total effect.")
    assert not sequence["after_estimates"]
    assert diet["model_sequence_card"]["guess"] == ["age", "sex", "kcal"]


def test_a_a_change_that_takes_a_model_1_column_out_of_the_primary_re_asks_model_1(diet):
    after = diet["proposals_after"]["model_sequence"]
    assert after["declared"] is None and "sex" not in after["allowed"]
    keys = [s["key"] for s in diet["effects_after"]["families"][0]["sequence"]]
    assert keys == ["crude", "model_2", "model_3"]
    [change] = [r for r in _records(diet["view_after"], "set_adjustment") if r["after_estimates"]]
    assert change["sentence"].startswith("After the estimates were seen, ")


def test_a_the_plan_exports_byte_identically_and_never_calls_itself_registered(diet):
    import json

    first, second = diet["plan"]
    assert first == second == diet["plan_records"]
    # replayed on a new server over the same workspace: the same bytes as the log's last state
    assert diet["plan_replayed"] == diet["plan_final"] != first
    later = json.loads(diet["plan_final"])
    assert later["sha256"] != json.loads(first)["sha256"]
    assert [r["kind"] for r in later["after_estimates"]] == ["set_adjustment"]
    assert later["plan"] == json.loads(first)["plan"]  # the plan as declared stays as declared
    doc = json.loads(first)
    assert doc["status"] == "locked" and doc["plan"]["model_sequence"]["model_1"] == [
        "age", "sex", "kcal"]
    assert doc["plan"]["estimand"]["measure"] == "risk_difference"
    text = first.decode("utf-8").lower()
    assert not any(w in text for w in ("prespecified", "pre-specified", "preregistered",
                                       "pre-registered"))


# ── journey B ────────────────────────────────────────────────────────────────


def test_b_a_cause_of_the_outcome_only_changes_the_conditional_hazard_ratio_and_the_app_says_so(cox):
    note = ("Under a conditional hazard ratio, adjusting for `activity`, a cause of the outcome only, "
            "changes the conditional estimand, not only its precision: the ratio is non-collapsible "
            "(Daniel, Zhang & Farewell 2021, Biom J 63:528).")
    card = cox["adjustment_card"]
    assert card["answered"]["activity"]["estimand_note"] == note == card["estimand_note"]
    sentences = [r["sentence"] for r in _records(cox["view"], "set_adjustment")]
    assert any(s.endswith(note) for s in sentences)
    assert ("a conditional ratio, which changes with the covariates even without confounding: "
            "adjusting for `activity`, a cause of the outcome only, changes the conditional "
            "estimand, not only its precision" in cox["fit"]["estimand"]["caption"])
    assert [m["measure"] for m in cox["estimand_card"]["measures"] if m["fitted"]] == ["hazard_ratio"]


def test_b_a_failed_check_leads_to_a_recorded_change_never_a_silent_one(cox):
    [before] = cox["effects"]["families"][0]["diagnostics"]
    assert before["check"] == "proportional_hazards" and before["status"] == "failed"
    assert before["response"] is None and before["change"] is None
    assert cox["exit"] == {"kind": "respond_diagnostic", "exposure": "x1",
                           "check": "proportional_hazards", "action": "period_hazard_ratios"}
    [after] = cox["effects_after"]["families"][0]["diagnostics"]
    assert after["response"] == "period_hazard_ratios" and len(after["change"]) == 2
    assert after["change_label"].startswith("The hazard ratio before and after ")
    [record] = _records(cox["view"], "respond_diagnostic")
    assert record["after_estimates"] is True
    assert record["sentence"] == (
        "After the estimates were seen, the proportional-hazards check (Schoenfeld residuals) of "
        "`x1`'s primary model failed; the response recorded is the exposure's hazard ratio before "
        "and after the median event time, beside the average over follow-up.")
    primary = lambda e: next(s for s in e["families"][0]["sequence"]  # noqa: E731
                             if s["key"] == "model_2")["effects"][0]["estimate"]
    assert primary(cox["effects_after"]) == primary(cox["effects"])  # nothing switched silently
    assert ("the check failed for the exposure, and the response recorded is the exposure's hazard "
            "ratio before and after the median event time" in cox["effects_after"]["methods"])
    assert "no response is recorded yet" in cox["effects"]["methods"]


def test_b_the_hazard_ratio_s_sensitivity_is_offered(cox):
    [s] = cox["effects"]["families"][0]["sensitivity"]
    share = float(cox["frame"]["event"].mean())
    assert s["e_value"]["measure"] == "HR" and s["e_value"]["rare"] is (share < 0.15)


# ── every relation is asserted ───────────────────────────────────────────────

HERE = "turbotab.core.tests.acceptance."
RELATION_TESTS = {
    "measure-labels": HERE + "test_estimand_chain::test_a_the_estimand_card_labels_the_measures_and_ranks_the_marginal_ones_first",
    "marginal-first": HERE + "test_estimand_2_g_computation::test_2_the_marginal_measures_rank_first_for_a_common_outcome",
    "precision-changes-estimand": HERE + "test_estimand_chain::test_b_a_cause_of_the_outcome_only_changes_the_conditional_hazard_ratio_and_the_app_says_so",
    "measure-shown": HERE + "test_estimand_chain::test_a_no_estimate_before_the_estimand_and_the_display_after_it",
    "gcomp-bootstrap-chain": HERE + "test_estimand_2_g_computation::test_2_rows_that_repeat_are_resampled_by_whole_unit",
    "gcomp-survey-blocked": HERE + "test_estimand_2_g_computation::test_2_a_surveyed_population_blocks_the_marginal_risks_and_records_the_way_out",
    "gcomp-or-beside": HERE + "test_estimand_chain::test_a_the_results_show_the_exposure_only_with_the_adjustment_terms_apart",
    "exposure-enables-display": HERE + "test_estimand_chain::test_a_no_estimate_before_the_estimand_and_the_display_after_it",
    "crude-always": HERE + "test_estimand_3_table2::test_3_the_unadjusted_estimate_is_shown_for_every_family",
    "table2-appendix": HERE + "test_estimand_chain::test_a_the_results_show_the_exposure_only_with_the_adjustment_terms_apart",
    "model3-labeled": HERE + "test_estimand_chain::test_a_the_results_show_the_exposure_only_with_the_adjustment_terms_apart",
    "sequence-invalidated": HERE + "test_estimand_chain::test_a_a_change_that_takes_a_model_1_column_out_of_the_primary_re_asks_model_1",
    "sequence-in-lock": HERE + "test_estimand_chain::test_a_the_plan_exports_byte_identically_and_never_calls_itself_registered",
    "failed-check-recorded": HERE + "test_estimand_chain::test_b_a_failed_check_leads_to_a_recorded_change_never_a_silent_one",
    "ph-period-ratios": HERE + "test_estimand_4_diagnostics::test_4_the_stage_reports_the_check_and_offers_a_recorded_response",
    "influence-refit": HERE + "test_estimand_4_diagnostics::test_4_an_influential_row_fails_the_check_and_its_response_refits_without_it",
    "sensitivity-offered": HERE + "test_estimand_5_sensitivity::test_5_every_family_offers_it_or_says_why",
    "rv-first-linear": HERE + "test_estimand_5_sensitivity::test_5_the_stage_offers_it_for_a_linear_outcome_robustness_value_first",
    "no-threshold": HERE + "test_estimand_chain::test_a_sensitivity_is_offered_with_no_threshold",
    "family-implies-multiplicity": HERE + "test_estimand_6_families::test_6_a_few_declared_nutrient_hypotheses_state_the_number_of_tests",
    "family-every-member": HERE + "test_estimand_6_families::test_6_every_member_is_shown_with_its_q_value",
    "omics-unadjusted-blocked": HERE + "test_estimand_6_families::test_6_bh_is_declared_unless_another_method_is_and_an_omics_family_without_it_is_recorded",
    "gcomp-unit-floor": HERE + "test_estimand_2_g_computation::test_2_below_the_unit_floor_the_marginal_risks_carry_no_interval",
    "model3-own-rows": HERE + "test_estimand_3_table2::test_3_model_2_is_the_fits_primary_and_model_3_alone_takes_fewer_rows",
    "sequence-design-based": HERE + "test_estimand_3_table2::test_3_under_the_surveyed_population_every_model_is_design_based",
    "rv-interval-classical": HERE + "test_estimand_5_sensitivity::test_5_the_stage_offers_it_for_a_linear_outcome_robustness_value_first",
    "curve-straight-line": HERE + "test_estimand_5_sensitivity::test_5_a_curve_is_bounded_through_its_straight_line_estimate",
    "export-replays": HERE + "test_estimand_chain::test_a_the_plan_exports_byte_identically_and_never_calls_itself_registered",
    "never-preregistered": HERE + "test_estimand_7_plan_export::test_7_the_text_says_what_was_declared_and_never_that_it_was_registered",
}


def test_every_relation_the_contracts_declare_is_asserted():
    """Each relation of the package's method contracts names the test that asserts it fires; a
    relation with no test, or a test that does not exist, fails here."""
    import turbotab.core.stages.effects  # noqa: F401 - registers the contracts

    mine = [c for c in contracts.contracts().values() if c.package == "ESTIMAND"]
    declared = {r.id for c in mine for r in c.relations}
    assert declared == set(RELATION_TESTS)
    for relation, where in RELATION_TESTS.items():
        module, name = where.split("::")
        assert callable(getattr(importlib.import_module(module), name, None)), (relation, where)
    for c in mine:
        assert c.slot in contracts.SLOTS and c.scope in contracts.SCOPES, c.key
        assert c.scope_note and c.storyboard and c.sentence and c.options, c.key
        assert c.leash.get("inference") and c.leash.get("prediction") == "not_offered", c.key
        for o in c.options:  # offered under inference only (BLUEPRINT north star 5's two labels)
            assert o.customary and o.sound["inference"] and o.rung["prediction"] == "not_offered"
        for r in c.relations:
            assert r.purposes == ("inference",) and r.condition, (c.key, r.id)
            if r.kind == "conflicts":
                assert r.rung == "block_and_record", (c.key, r.id)
        module, _, name = c.sentence.partition(":")
        assert callable(getattr(importlib.import_module(module), name)), c.key

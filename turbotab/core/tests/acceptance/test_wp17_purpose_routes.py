"""WP17 · the declared purpose routes the questions (AUDIT_REPORT §5; closes RO-03, RO-04, RO-06,
RO-07, RO-08). MODELING_SEQUENCE DRAFT 2 steps 2–3 govern over the audit's older wording.

The package's five acceptance tests, in its order:

1. **Customary and sound.** Every option of the energy, missing-values, split and exclusions
   questions carries "customary in <field>" with a source and "sound for <purpose>" with a reason;
   the order under inference differs from the order under prediction for each, and one line names
   the tension where the field's first choice is not the soundest.
2. **Censoring.** On the staggered-entry fixture a follow-up question fires; the models wait behind
   it, so the audit's p ≈ 10⁻¹³ log-odds is never served; "the same for everyone" over a follow-up
   that varies is blocked and recorded; the time-to-event refusal's own exit is accepted in the
   ``target_info`` window (it was refused "not yet"); the route ends at the Cox family.
3. **Inference questions** (MODELING_SEQUENCE §1 steps 2–3). On the NHANES export under inference no
   coefficient is shown until the exposure, its effect and every covariate's answers to the
   modified disjunctive cause criterion are given; the roles are derived from the answers; the
   covariates the pack guesses alike are confirmed in one tap per group; a mediator kept in a
   total-effect set is blocked and recorded, and "further adjusted for" the body-size measures of
   unknown timing is the declared secondary; the caption is worded from the estimand.
4. **Clusters.** On the audit's multi-site fixtures a cluster question fires, and under inference
   site fixed effects with site-clustered intervals reproduce the audit's estimate.
5. **Energy order under prediction.** The energy-keeping methods rank first, the residual method's
   energy-dropped form says it discards energy, and on data where the outcome follows energy the
   cross-validated R² bears the order out.

Every expected value comes from outside the engine: statsmodels and lifelines on the fixture's own
columns, the CR2 covariance written out by definition (``references.cr2_by_definition``), the
fixtures' own generators, and the primary sources quoted below.

**Source check (3).** VanderWeele TJ. Principles of confounder selection. *Eur J Epidemiol*
2019;34:211–219 (PMC6447501, read on 2026-10-03): "control for each covariate that is a cause of the
exposure, or of the outcome, or of both; exclude from this set any variable known to be an
instrumental variable; and include as a covariate any proxy for an unmeasured variable that is a
common cause of both the exposure and the outcome"; "Statistical analyses cannot in general
distinguish between confounders, which ought to be controlled for in the estimation of the total
effect, versus mediators, which ought not be controlled for in the estimation of the total effect";
and, of BMI measured at the same time as the exposure, "We cannot adequately distinguish in this
setting between confounding and mediation."
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

from turbotab.core import custom_sound, decisions as d, estimand, interview, voice
from turbotab.core.decisions import ProjectState, Refusal
from turbotab.core.tests import modeling_fixtures as mf
from turbotab.core.tests.acceptance import references as ref
from turbotab.core.tests.acceptance.server_drive import local_server, open_project
from turbotab.core.tests.stage_harness import NHANES, SAMPLES, Ingested
from turbotab.core.tests.truths import Truth, adjustment_truth, fixture_truth

VANDERWEELE_QUOTES = (
    "control for each covariate that is a cause of the exposure, or of the outcome, or of both; "
    "exclude from this set any variable known to be an instrumental variable; and include as a "
    "covariate any proxy for an unmeasured variable that is a common cause of both the exposure "
    "and the outcome",
    "Statistical analyses cannot in general distinguish between confounders, which ought to be "
    "controlled for in the estimation of the total effect, versus mediators, which ought not be "
    "controlled for",
)


def _flat(text: str) -> str:
    return " ".join(str(text).split())


# ── 1 · customary and sound on every option ──────────────────────────────────

DIET_ROLES = {"participant_id": "identifier", "age": "covariate", "sex": "covariate",
              "bmi": "covariate", "energy_kcal": "energy", "protein_g": "exposure",
              "fat_g": "exposure", "carbohydrate_g": "exposure", "fiber_g": "covariate"}


def _labels(tmp_path: Path, purpose: str) -> dict[str, Any]:
    """The four questions' labels as the server serves them: the proposals stage (energy,
    missing values, exclusions) and the seal plan (the split), on the dietary recalls."""
    from turbotab.core.seal import plan
    from turbotab.core.stages.proposals import build_proposals

    table = Ingested(SAMPLES / "dietary_recalls.csv", tmp_path / purpose)
    frame = table.frame()
    state = ProjectState(lens=["dietary"], target="hba1c", task="regression", purpose=purpose,
                         roles=DIET_ROLES, grain={"grain": "repeated",
                                                  "id_column": "participant_id"})
    proposals = build_proposals(frame, table.info["columns"], lens=["dietary"], target="hba1c",
                                roles=DIET_ROLES, purpose=purpose)
    with table.store() as store:
        sealed = plan(state, frame.index.to_numpy(), store, "regression")
    return {**proposals["labels"], "split": sealed["labels"], "_proposals": proposals,
            "_plan": sealed}


def test_1_every_option_is_labeled_customary_and_sound_and_ordered_by_purpose(tmp_path):
    """AUDIT_REPORT §5 WP17 test 1 (north star 5; RO-06). Reference: the audit's own table of the
    two labels (§3.1–§3.4), and each question's order as its own stage serves it (the energy
    ranking, the missing-data methods, the seal plan's options), never a list written here."""
    served = {p: _labels(tmp_path, p) for p in ("inference", "prediction")}
    for question in ("energy_adjustment", "missing", "split", "exclusions"):
        orders = {}
        for purpose, labels in served.items():
            q = labels[question]
            assert q is not None and q["purpose"] == purpose, (question, purpose)
            for o in q["options"]:
                c, s = o["customary"], o["sound"]
                assert c["field"] and c["text"] and c["source"], (question, o)
                assert s["purpose"] == purpose and s["verdict"] in ("sound", "conditional",
                                                                    "unsound")
                assert len(s["reason"].split()) >= 4, (question, o)
            keys = [o["key"] for o in q["options"]]
            orders[purpose] = keys
            # one line names the tension exactly where the field's first choice is not first
            assert (q["tension"] is not None) == (keys[0] != q["customary_first"]), (question, q)
            if q["tension"]:
                assert len(q["tension"].split()) <= 40
        assert orders["inference"] != orders["prediction"], question
        # at least one purpose finds the customary first choice is not the soundest: the tension
        assert any(served[p][question]["tension"] for p in served), question
    for purpose, labels in served.items():
        proposals, sealed = labels["_proposals"], labels["_plan"]
        # each question's labels are in the order its own card offers the options
        assert [o["key"] for o in labels["missing"]["options"]] == [
            m["key"] for m in proposals["missing"]["methods"]]
        assert [o["key"] for o in labels["energy_adjustment"]["options"]] == \
            proposals["energy"]["ranking"]["order"]
        offered = [o["validation"] for o in sealed["validation"]["options"]]
        holdout_first = not sealed["cv_first"]
        assert [o["key"] for o in labels["split"]["options"]] == (
            (["holdout"] if holdout_first else []) + offered + ([] if holdout_first else ["holdout"]))
        assert set(o["key"] for o in labels["exclusions"]["options"]) == {
            "keep_every_row", *[e["key"] for e in proposals["exclusions"]]}
    # the soundness verdicts that differ by purpose are the audit's (§3.3, §3.2, §3.4)
    sound = {p: {q: {o["key"]: o["sound"]["verdict"] for o in served[p][q]["options"]}
                 for q in ("missing", "exclusions", "split")} for p in served}
    assert sound["inference"]["missing"]["multiple_imputation"] == "sound"
    assert sound["prediction"]["missing"]["multiple_imputation"] == "unsound"
    assert sound["inference"]["missing"]["impute"] == "unsound"
    assert sound["prediction"]["exclusions"]["keep_every_row"] == "sound"
    assert sound["inference"]["split"]["holdout"] == "unsound"
    assert served["prediction"]["exclusions"]["options"][0]["key"] == "keep_every_row"
    assert served["inference"]["split"]["options"][0]["key"] == "kfold"


def test_1_the_customary_sources_say_what_the_labels_claim():
    """Source check: the sentences the customary labels rest on, quoted in the module that carries
    them (read on 2026-10-03): Sterne et al. 2009, Steyerberg 2018, Banna et al. 2017."""
    doc = _flat(custom_sound.__doc__)
    for quote in ("Researchers usually address missing data by including in the analysis only "
                  "complete cases",
                  "The standard requirement in major medical journals is nowadays that validity "
                  "outside the development sample needs to be shown",
                  "random data splitting should be abolished for validation of prediction models",
                  "a drawback of this crude approach is that it is not individualized and does not "
                  "capture all implausible reports"):
        assert quote in doc, quote


# ── 2 · censoring: the follow-up question and the Cox family ─────────────────


def _staggered() -> pd.DataFrame:
    from turbotab.core.tests.acceptance.test_wp12b_cox_mixed_gee import _staggered_entry_cohort

    return _staggered_entry_cohort()


def test_2_the_follow_up_question_fires_and_holds_the_models(tmp_path):
    """The Router on the staggered-entry fixture: the outcome reads as yes/no, `followup_years`
    reads as a follow-up time that varies (its own range: pandas), so the follow-up question is
    asked, and every later question waits behind it. The served-estimate gate names it."""
    frame = _staggered()
    table = Ingested(_csv(frame, tmp_path, "cohort_tte_null"), tmp_path)
    state = ProjectState(lens=["clinical"], target="cvd_event")
    from turbotab.core.stages.target import target_info_stage

    info = table.run(target_info_stage, state)
    [found] = info["follow_up"]
    assert found["column"] == "followup_years" and found["varies"] is True
    assert found["min"] == pytest.approx(frame["followup_years"].min())
    assert found["max"] == pytest.approx(frame["followup_years"].max())
    fresh = {s: {"status": "fresh"} for s in ("ingest", "oriented", "target_info", "structure")}
    steps = {s.key: s for s in interview.route(state, fresh, {"target_info": info})}
    assert steps["task"].status == "skipped"  # 0/1: read as yes/no at high confidence
    assert steps["event"].status == "open" and steps["follow_up"].status == "waiting"
    state = state.model_copy(update={"event": "1"})
    steps = {s.key: s for s in interview.route(state, fresh, {"target_info": info})}
    assert steps["follow_up"].status == "open"
    assert steps["models"].status == "waiting" and steps["models"].waiting_on[0] == "follow_up"
    gate = estimand.served_gate(state, list(steps.values()))
    assert gate is not None and gate["question"] == "follow_up"
    # The routing gate (NHANES linked mortality's `PERMTH_INT`, read by no name): under the
    # clinical lens a yes/no outcome is asked whatever its columns are named; under an assay lens
    # alone, with nothing that reads as follow-up, the skip is stated.
    plain = dict(info, follow_up=[])
    steps = {s.key: s for s in interview.route(state, fresh, {"target_info": plain})}
    assert steps["follow_up"].status == "open"
    assay = state.model_copy(update={"lens": ["metabolomics"]})
    steps = {s.key: s for s in interview.route(assay, fresh, {"target_info": plain})}
    assert steps["follow_up"].status == "skipped" and "follow-up time" in steps["follow_up"].reason


def _csv(frame: pd.DataFrame, folder: Path, name: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.csv"
    frame.to_csv(path, index=False)
    return path


def _window(service: Any, monkeypatch: Any) -> Any:
    """Hold the ``target_info`` window open: the decision context sees the stage still running, so
    the task is not known and the event question waits on it (test_methods_gate's device)."""
    real = service.decision_context

    def window(pid: str, stages: Any = None) -> Any:
        stages = dict(service.engine.status(pid))
        stages["target_info"] = stages["target_info"].model_copy(
            update={"status": "running", "key": None, "fresh": False})
        return real(pid, stages)

    monkeypatch.setattr(service, "decision_context", window)
    return real


def test_2_the_time_to_event_exit_is_taken_in_the_target_info_window(tmp_path, monkeypatch):
    """The follow-up's refusal names one exit, "analyze it as a time to event" (``set_task``). Posted
    while ``target_info`` was still computing it was refused "not yet": the event question waits on
    the task, and the order check held the task behind it. The event, the task and the follow-up
    describe one outcome and no longer hold each other back; the questions before them still do.
    Reference: the exit itself, and the Router's own order for a question outside the three."""
    path = _csv(_staggered(), tmp_path, "cohort_tte_null")
    with local_server(tmp_path / "home") as client:
        drive = open_project(client, path)
        drive.decide({"kind": "set_lens", "lenses": ["clinical"]})
        drive.reach("target")
        drive.decide({"kind": "set_target", "column": "cvd_event"})
        real = _window(client.app.state.service, monkeypatch)
        refused = drive.post({"kind": "set_follow_up", "column": "cvd_event",
                              "time_column": "followup_years"})
        assert refused.status_code == 409, refused.text
        error = refused.json()["error"]
        assert error["code"] == "not_time_to_event"
        exit_ = error["exits"][0]["decision"]
        assert exit_ == {"kind": "set_task", "column": "cvd_event", "task": "time_to_event"}
        later = drive.post({"kind": "set_purpose", "purpose": "inference"})
        assert later.status_code == 409 and later.json()["error"]["code"] == "not_yet"
        taken = drive.post(exit_)
        assert taken.status_code == 200, taken.text  # was 409 not_yet
        monkeypatch.setattr(client.app.state.service, "decision_context", real)
        assert drive.view()["state"]["task"] == "time_to_event"


def test_2_censoring_routes_to_the_cox_family_through_the_server(tmp_path):
    """Through the real HTTP API. The follow-up question is open; the models are refused "not yet"
    behind it, so no log-odds is ever fit on censored follow-up; "the same for everyone" is
    refused (block and record) with the time-to-event exit and the attestation; the exit is taken
    and the follow-up named; the shelf offers the Cox family alone and its hazard ratio is
    lifelines' on the same rows (reference: ``CoxPHFitter``, Efron ties)."""
    from turbotab.core.tests.acceptance.test_wp12b_cox_mixed_gee import _lifelines

    frame = _staggered()
    path = _csv(frame, tmp_path, "cohort_tte_null")
    truth = Truth({"code_or_count:age": "amount", "adjust:age": "yes,yes,no"},
                  fixture="the staggered-entry cohort")
    with local_server(tmp_path / "home") as client:
        drive = open_project(client, path, truth)
        drive.decide({"kind": "set_lens", "lenses": ["clinical"]})
        drive.reach("target")
        drive.decide({"kind": "set_target", "column": "cvd_event"})
        drive.answer("event", {"kind": "set_event", "column": "cvd_event", "level": "1"})
        step = drive.reach("follow_up")
        assert step["status"] == "open"
        held = drive.post({"kind": "select_models", "models": ["linear"]})
        assert held.status_code == 409 and held.json()["error"]["code"] == "not_yet"
        same = drive.post({"kind": "set_censoring", "column": "cvd_event"})
        assert same.status_code == 409, same.text
        error = same.json()["error"]
        assert error["code"] == "follow_up_varies" and "`followup_years`" in error["message"]
        assert error["exits"][0]["decision"] == {"kind": "set_task", "column": "cvd_event",
                                                 "task": "time_to_event"}
        assert error["exits"][-1]["decision"]["acknowledged"] is True  # the attestation exit
        refused = drive.post({"kind": "set_follow_up", "column": "cvd_event",
                              "time_column": "followup_years"})
        assert refused.status_code == 409
        exit_ = refused.json()["error"]["exits"][0]["decision"]
        assert exit_ == {"kind": "set_task", "column": "cvd_event", "task": "time_to_event"}

        drive.decide(exit_)
        assert drive.reach("follow_up")["status"] == "open"  # a time to event: name its follow-up
        drive.decide({"kind": "set_follow_up", "column": "cvd_event",
                      "time_column": "followup_years"})
        assert drive.reach("follow_up")["status"] == "answered"
        drive.decide({"kind": "set_purpose", "purpose": "inference"})
        drive.answer("grain", {"kind": "set_grain", "grain": "one_row_per_unit"})
        drive.reach("roles")
        drive.decide_roles({"participant_id": "identifier", "fiber_g": "exposure",
                            "age": "covariate", "followup_years": "time"})
        drive.answer("exclusions", {"kind": "set_exclusions", "rules": []})
        drive.answer("missing", {"kind": "set_missing", "strategy": "complete_case"})
        drive.answer("split", {"kind": "set_split", "holdout": 0.0, "seed": 0, "folds": 5})
        drive.answer_plan("fiber_g")  # after the split (MODELING_SEQUENCE §1 steps 2–3)
        drive.reach("models")
        assert [f["key"] for f in drive.artifact("shelf")["families"]] == ["cox"]
        drive.decide({"kind": "select_models", "models": ["cox"]})
        fit = drive.artifact("fit")
        view = drive.view()
    assert view["state"]["task"] == "time_to_event"
    model = fit["models"][0]
    assert fit["task"] == "time_to_event" and model["family"] == "cox"
    fiber = next(r for r in model["coefficients"] if r["feature"] == "fiber_g")
    reference = _lifelines(frame, "followup_years", "cvd_event", ["fiber_g", "age"])
    assert fiber["estimate"] == pytest.approx(reference.params_["fiber_g"], rel=1e-6)
    assert (round(math.exp(fiber["ci_low"]), 3), round(math.exp(fiber["ci_high"]), 3)) == (
        0.997, 1.027)
    assert "hazard ratio" in fit["estimand"]["caption"]


def test_2_the_same_follow_up_answer_is_recorded_when_attested():
    """Block and record, the other way out: the attested "same for everyone" is recorded and its
    sentence carries the limitation (reference: the answer itself)."""
    said = voice.sentence_for(d.SetCensoring(column="cvd_event", acknowledged=True),
                              ProjectState(target="cvd_event"))
    assert "a stated limitation" in said and "yes/no outcome" in said
    state = d.fold([d.DecisionRecord(id="a", seq=1, at="2026-10-03T00:00:00Z",
                                     decision=d.SetTarget(column="cvd_event")),
                    d.DecisionRecord(id="b", seq=2, at="2026-10-03T00:00:01Z",
                                     decision=d.SetCensoring(column="cvd_event",
                                                             acknowledged=True))])
    assert state.censoring == "same_attested"
    info = {"column": "cvd_event", "task": "binary", "confidence": "high", "follow_up": [
        {"column": "followup_years", "min": 0.1, "max": 14.9, "varies": True}]}
    steps = {s.key: s for s in interview.route(state, {}, {"target_info": info})}
    assert steps["follow_up"].status == "answered"


# ── 3 · inference: the exposure, its effect, and the adjustment set ──────────

needs_nhanes = pytest.mark.skipif(not NHANES.is_file(),
                                  reason="the NHANES export (_tt_tmp_nhanes.csv) is not on this machine")
NHANES_COVARIATES = ["age", "gender", "cycle_begin_year", "protein", "carb", "fat_total", "fat_sat",
                     "fat_mon", "fat_poly", "weight", "height", "bmi", "waist", "bp_sys", "bp_di",
                     "hdl", "triglycerides", "meds_hbp", "meds_chol"]
NUTRIENTS = ["protein", "carb", "fat_total", "fat_sat", "fat_mon", "fat_poly"]
NHANES_ROLES = {"SEQN": "identifier", "sugar": "exposure", "kcal": "energy",
                **{c: "exposure" for c in NUTRIENTS},
                **{c: "covariate" for c in NHANES_COVARIATES if c not in NUTRIENTS},
                **{c: "flag" for c in ("imputed_weight", "imputed_height", "imputed_bmi",
                                       "imputed_waist", "imputed_bp_sys", "imputed_bp_di")}}


def _reference(raw: pd.DataFrame, covariates: list[str]) -> Any:
    """statsmodels OLS of glucose on sugar, total energy and ``covariates`` over the complete
    cases, with HC3 standard errors on t(n − p): the table the app should report."""
    columns = ["glucose", "sugar", "kcal", *covariates]
    rows = raw.dropna(subset=columns).copy()
    terms = " + ".join(f"C({c})" if c in ("gender", "cycle_begin_year") else c
                       for c in ["sugar", "kcal", *covariates])
    return smf.ols(f"glucose ~ {terms}", rows).fit(cov_type="HC3", use_t=True), len(rows)


@needs_nhanes
def test_3_no_coefficient_before_the_plan_and_the_roles_come_from_the_answers(tmp_path):
    """AUDIT_REPORT §5 WP17 test 3, as MODELING_SEQUENCE §1 steps 2–3 govern it, on the NHANES
    export under inference (sugar → fasting glucose). References: statsmodels (HC3, t) on the
    complete cases of the primary set and of the declared secondary; the derived roles from the
    fixture's declared causal truth (``truths.py``); VanderWeele 2019 for the criterion."""
    truth = fixture_truth("_tt_tmp_nhanes.csv")
    raw = pd.read_csv(NHANES)
    with local_server(tmp_path / "home") as client:
        drive = open_project(client, NHANES, truth)
        drive.decide({"kind": "set_lens", "lenses": ["dietary"]})
        drive.reach("target")
        drive.decide({"kind": "set_target", "column": "glucose"})
        drive.reach("purpose")
        drive.decide({"kind": "set_purpose", "purpose": "inference"})
        drive.answer("grain", {"kind": "set_grain", "grain": "one_row_per_unit"})
        drive.reach("roles")
        drive.decide_roles(NHANES_ROLES)
        assert drive.reach("clusters")["status"] == "skipped"  # nothing reads as a site
        # The opening sequence ends at the seal; the exposure and the adjustment set are the
        # modeling sequence's steps 2–3, asked after it (MODELING_SEQUENCE §1).
        drive.answer("exclusions", {"kind": "set_exclusions", "rules": []})
        drive.answer("missing", {"kind": "set_missing", "strategy": "complete_case"})
        drive.answer("split", {"kind": "set_split", "holdout": 0.0, "seed": 0, "folds": 5})
        assert drive.reach("estimand")["status"] == "open"
        # No estimate before the plan: the models are held behind the exposure question.
        held = drive.post({"kind": "select_models", "models": ["linear"]})
        assert held.status_code == 409 and held.json()["error"]["code"] == "not_yet"
        card = drive.artifact("proposals")["estimand"]
        assert [m["measure"] for m in card["measures"] if m["fitted"]] == ["mean_difference"]
        sugar = next(e for e in card["exposures"] if e["column"] == "sugar")
        assert sugar["energy_contrast"] is True
        base = {"kind": "set_estimand", "exposure": "sugar", "effect": "total",
                "measure": "mean_difference"}
        asked = drive.post(base)
        assert asked.status_code == 409 and asked.json()["error"]["code"] == "which_contrast"
        wrong = drive.post({**base, "contrast": "substitution", "measure": "odds_ratio"})
        assert wrong.status_code == 409 and wrong.json()["error"]["code"] == "measure_mismatch"
        drive.decide({**base, "contrast": "substitution"})

        assert drive.reach("adjustment")["status"] == "open"
        card = drive.artifact("proposals")["adjustment"]
        while card is None or card["exposure"] != "sugar":
            card = drive.artifact("proposals")["adjustment"]
        groups = {g["key"]: g for g in card["groups"]}
        assert groups["demographic"]["columns"] == ["age", "gender"]
        assert groups["dietary"]["columns"] == NUTRIENTS  # other dietary components: confounders
        assert groups["dietary"]["derived"] == "confounder"
        # LEASH: body size, the clinical measurements and the medications share one guess under
        # sugar → fasting glucose (possible mediators, or measured after the exposure: declared
        # without them and, beside, with them), so they are one block.
        assert groups["body"]["columns"] == ["weight", "height", "bmi", "waist", "bp_sys", "bp_di",
                                             "hdl", "triglycerides", "meds_hbp", "meds_chol"]
        assert groups["body"]["derived"] == "timing_unknown"
        assert groups["unguessed"]["columns"] == ["cycle_begin_year"]
        assert groups["unguessed"]["guess"] is None  # the pack has no guess: asked plainly
        asked_about = {c for g in card["groups"] for c in g["columns"]}
        assert asked_about == set(NHANES_COVARIATES)  # not sugar, not total energy
        # A mediator kept in a total-effect set: blocked, with the secondary and the record as exits.
        hdl = {"causes_exposure": "no", "causes_outcome": "yes", "after_exposure": "yes"}
        kept = drive.post({"kind": "set_adjustment", "exposure": "sugar",
                           "answers": {"hdl": {**hdl, "keep": True}}})
        assert kept.status_code == 409, kept.text
        error = kept.json()["error"]
        assert error["code"] == "mediator_in_total_effect" and "ought not" in error["message"]
        assert error["exits"][0]["decision"]["answers"]["hdl"]["further"] is True
        assert error["exits"][1]["decision"]["answers"]["hdl"]["acknowledged"] is True
        # Still nothing to fit: the plan is unanswered.
        assert drive.post({"kind": "select_models", "models": ["linear"]}).status_code == 409
        posted = _answer_from_truth(drive, card)
        # One tap per group the pack guesses alike (demographics, other nutrients); the fixture's
        # author reads the clinical measurements and medications as mediators, not the block's
        # guess, so that block is answered by its two declared sets, and the unguessed survey
        # cycle by its own: 5 taps, 19 covariates.
        assert len(posted) == 5 and posted[0] == groups["demographic"]["decision"]
        # The estimand's contrast constrains the energy model (MODELING_SEQUENCE §2): a model that
        # lets total energy leave estimates no substitution (Tomova et al. 2022).
        crude = drive.post({"kind": "set_energy_adjustment", "method": "none",
                            "energy_column": "kcal", "nutrients": ["sugar", *NUTRIENTS]})
        assert crude.status_code == 409 and crude.json()["error"]["code"] == "contrast_mismatch"
        assert {e["decision"]["method"] for e in crude.json()["error"]["exits"][:3]} == {
            "all_components", "standard", "residual"}
        drive.decide({"kind": "set_energy_adjustment", "method": "standard",
                      "energy_column": "kcal", "nutrients": ["sugar", *NUTRIENTS]})
        drive.decide({"kind": "select_models", "models": ["linear"]})
        fit = drive.artifact("fit")
        secondary = drive.artifact("secondary")
        view = drive.view()
        # Taking an answer back re-opens the question and withholds every estimate again (the other
        # nutrients' tap: they stay in the model either way, so no reading is unsettled by it).
        tap = next(r for r in view["decisions"] if r["decision"]["kind"] == "set_adjustment"
                   and "protein" in r["decision"]["answers"])
        drive.decide({"kind": "revert", "decision_id": tap["id"]})
        assert drive.reach("adjustment")["status"] in ("open", "waiting")
        withheld = client.get(f"/api/projects/{drive.pid}/stages/fit").json()["artifact"]

    # the derived roles: the fixture's truth through the criterion (VanderWeele 2019)
    derived = {c: estimand.derive(adjustment_truth(truth, c)).role for c in NHANES_COVARIATES}
    primary = [c for c in NHANES_COVARIATES if derived[c] in ("confounder",)]
    assert set(primary) == {"age", "gender", "cycle_begin_year", *NUTRIENTS}
    assert {c for c, r in derived.items() if r == "mediator"} == {
        "bp_sys", "bp_di", "hdl", "triglycerides", "meds_hbp", "meds_chol"}
    model = fit["models"][0]
    features = {r["feature"] for r in model["coefficients"]}
    assert "sugar" in features and not features & {"bmi", "hdl", "bp_sys", "weight", "meds_hbp"}
    reference, n = _reference(raw, primary)
    row = next(r for r in model["coefficients"] if r["feature"] == "sugar")
    assert model["coefficients_n"] == n
    assert row["estimate"] == pytest.approx(reference.params["sugar"], rel=1e-6)
    assert row["se"] == pytest.approx(reference.bse["sugar"], rel=1e-6)
    # the caption is worded from the estimand
    caption = fit["estimand"]["caption"]
    assert caption.startswith("The total effect of `sugar` on `glucose` (in place of other energy "
                              "sources at fixed total energy), as a difference in the mean outcome")
    assert "left out as consequences of the exposure" in caption
    assert "further adjusted for `weight`, `height`, `bmi` and `waist`" in caption
    assert "VanderWeele 2019" in caption and fit["estimand"]["features"] == ["sugar"]
    # the declared "further adjusted for" model, on the same rows (statsmodels on its own)
    body = ["weight", "height", "bmi", "waist"]
    [family] = secondary["families"]
    first, further = family["fits"]
    assert first["label"] == "Primary" and further["label"].startswith("Further adjusted for")
    reference2, n2 = _reference(raw.dropna(subset=body), primary + body)
    reference1, _ = _reference(raw.dropna(subset=body), primary)
    assert further["n_rows"] == n2
    assert further["coefficients"][0]["estimate"] == pytest.approx(reference2.params["sugar"], rel=1e-6)
    assert first["coefficients"][0]["estimate"] == pytest.approx(reference1.params["sugar"], rel=1e-6)
    # the record states each derived role, in a methods sentence
    said = " ".join(r["sentence"] for r in view["decisions"]
                    if r["decision"]["kind"] in ("set_estimand", "set_adjustment"))
    assert "`hdl`" in said and "mediators, left out" in said
    assert "of unknown timing, left out of the primary model and adjusted for in a declared " \
           "secondary one" in said
    # withheld once an answer is taken back
    assert withheld["withheld"].startswith("No estimate is shown until the adjustment-set question")
    assert all(m["coefficients"] is None for m in withheld["models"])


def _answer_from_truth(drive: Any, card: dict[str, Any]) -> list[dict[str, Any]]:
    from turbotab.core.tests.truths import answer_adjustment

    return answer_adjustment(drive.post, card, drive.truth)


def test_3_the_criterion_derives_each_role_from_the_answers():
    """The derivation itself, answer by answer, against VanderWeele's (2019) criterion as quoted
    (reference: the quoted rule, one case per clause), and the source check."""
    doc = _flat(estimand.__doc__)
    for quote in VANDERWEELE_QUOTES:
        assert _flat(quote) in doc, quote

    def role(ce: str, co: str, after: str, **flags: Any) -> tuple[str, bool]:
        found = estimand.derive(d.CovariateAnswers(causes_exposure=ce, causes_outcome=co,
                                                   after_exposure=after, **flags))
        return found.role, found.adjusted

    # "control for each covariate that is a cause of the exposure, or of the outcome, or of both"
    assert role("yes", "yes", "no") == ("confounder", True)
    assert role("yes", "no", "no") == ("exposure_cause", True)
    assert role("no", "yes", "no") == ("precision", True)
    assert role("no", "no", "no") == ("not_a_cause", False)
    # "exclude from this set any variable known to be an instrumental variable"
    assert role("yes", "no", "no", instrument=True) == ("instrument", False)
    # "include as a covariate any proxy for an unmeasured ... common cause"
    assert role("no", "no", "no", proxy=True) == ("proxy", True)
    # mediators "ought not be controlled for in the estimation of the total effect"
    assert role("no", "yes", "yes") == ("mediator", False)
    assert estimand.derive(d.CovariateAnswers(causes_exposure="no", causes_outcome="yes",
                                              after_exposure="yes"), "direct").adjusted is True
    assert role("no", "no", "yes") == ("collider", False)
    # unknown timing: the declared with-and-without pair
    found = estimand.derive(d.CovariateAnswers(causes_exposure="unknown", causes_outcome="yes",
                                               after_exposure="unknown"))
    assert (found.role, found.adjusted, found.secondary) == ("timing_unknown", False, True)
    # kept, attested: in the primary set, recorded as not a total effect
    kept = estimand.derive(d.CovariateAnswers(causes_exposure="no", causes_outcome="yes",
                                              after_exposure="yes", keep=True, acknowledged=True))
    assert kept.adjusted is True
    said = voice.sentence_for(d.SetAdjustment(exposure="sugar", answers={"hdl": d.CovariateAnswers(
        causes_exposure="no", causes_outcome="yes", after_exposure="yes", keep=True,
        acknowledged=True)}), ProjectState(target="glucose"))
    assert "not a total effect" in said


def test_3_the_effect_measure_offers_only_what_the_engine_fits():
    """Ruling 9 (MODELING_SEQUENCE §0): the measure is part of the estimand, and only the
    engine's own measures are offered. Since the package ESTIMAND a yes/no outcome's marginal risk
    difference and ratio are fitted (g-computation, ``test_estimand_2_g_computation.py``); another
    task's request for them is named and refused with the reason."""
    assert [m["measure"] for m in estimand.measures_offered("binary")] == [
        "odds_ratio", "risk_difference", "risk_ratio"]
    assert [m["fitted"] for m in estimand.measures_offered("binary")] == [True, True, True]
    state = ProjectState(target="dm", task="binary", purpose="inference",
                         roles={"fiber_g": "exposure", "age": "covariate"},
                         role_confirmations={"fiber_g": "exposure", "age": "covariate"})
    d.validate({"kind": "set_estimand", "exposure": "fiber_g", "measure": "risk_ratio"},
               {"state": state, "task": "binary"})
    with pytest.raises(Refusal) as refused:
        d.validate({"kind": "set_estimand", "exposure": "fiber_g", "measure": "risk_ratio"},
                   {"state": state.model_copy(update={"task": "time_to_event"}),
                    "task": "time_to_event"})
    assert refused.value.code == "measure_not_fitted"
    assert refused.value.exits[0]["decision"]["measure"] == "hazard_ratio"
    with pytest.raises(Refusal) as refused:
        d.validate({"kind": "set_estimand", "exposure": "fiber_g", "measure": "odds_ratio"},
                   {"state": state.model_copy(update={"purpose": "prediction"}), "task": "binary"})
    assert refused.value.code == "not_inference"
    d.validate({"kind": "set_estimand", "exposure": "fiber_g", "measure": "odds_ratio"},
               {"state": state, "task": "binary"})


def test_3_an_exposure_family_is_reported_one_at_a_time_by_the_feature_wise_family():
    """MODELING_SEQUENCE §1 step 2: "one exposure or an exposure family (feature-wise, with its
    multiplicity method)". The family's measure is what the feature-wise family estimates
    (``models/featurewise.py``: a numeric outcome's difference per unit; a yes/no outcome's
    difference in each exposure's mean, the limma design); no other exposure is a covariate; a
    joint model, which adjusts each exposure for the others, is refused against it; one exposure
    is no family. And a covariate whose role rode along unconfirmed is asked with the rest, so
    confirming it later finds its answers given (the question does not reopen)."""
    genes = {f"g{i}": "exposure" for i in range(3)}
    state = ProjectState(target="case", task="binary", purpose="inference",
                         roles={**genes, "age": "covariate", "batch": "covariate"},
                         role_confirmations={**genes, "age": "covariate"},
                         roles_unconfirmed=["batch"])
    ctx = {"state": state, "task": "binary"}
    assert estimand.fitted_measures("binary", family=True) == ["exposure_mean_difference"]
    assert estimand.fitted_measures("regression", family=True) == ["mean_difference"]
    assert estimand.fitted_measures("ordinal", family=True) == []
    with pytest.raises(Refusal) as refused:
        d.validate({"kind": "set_estimand", "family": True, "measure": "odds_ratio"}, ctx)
    assert refused.value.code == "measure_mismatch"
    d.validate({"kind": "set_estimand", "family": True, "measure": "exposure_mean_difference"}, ctx)
    with pytest.raises(Refusal) as refused:
        d.validate({"kind": "set_estimand", "exposure": "g0", "measure": "exposure_mean_difference"},
                   ctx)
    assert refused.value.code == "measure_mismatch"
    one = state.model_copy(update={"roles": {"g0": "exposure", "age": "covariate"},
                                   "role_confirmations": {"g0": "exposure", "age": "covariate"},
                                   "roles_unconfirmed": None})
    with pytest.raises(Refusal) as refused:
        d.validate({"kind": "set_estimand", "family": True, "measure": "exposure_mean_difference"},
                   {"state": one, "task": "binary"})
    assert refused.value.code == "no_family"
    assert refused.value.exits[0]["decision"]["exposure"] == "g0"

    family = state.model_copy(update={"estimand": d.EstimandSpec(
        family=True, measure="exposure_mean_difference")})
    assert estimand.covariates(family) == ["age"]  # no gene is another's covariate
    assert estimand.asked_covariates(family) == ["age", "batch"]  # batch rode along: asked too
    answered = family.model_copy(update={"adjustment": {c: d.AdjustmentAnswer(
        exposure=d.EXPOSURE_FAMILY, causes_exposure="no", causes_outcome="yes",
        after_exposure="no") for c in ("age", "batch")}})
    assert estimand.adjustment_answer(answered) is not None
    confirmed = answered.model_copy(update={
        "role_confirmations": {**answered.role_confirmations, "batch": "covariate"}})
    assert estimand.adjustment_answer(confirmed) is not None  # confirming batch reopens nothing
    assert set(estimand.derived_roles(confirmed)) == {"age", "batch"}
    with pytest.raises(Refusal) as refused:
        d.validate({"kind": "select_models", "models": ["featurewise", "linear"]},
                   {"state": confirmed, "task": "binary"})
    assert refused.value.code == "family_needs_featurewise"
    d.validate({"kind": "select_models", "models": ["featurewise"]}, {"state": confirmed})
    caption = estimand.caption(confirmed)
    assert caption.startswith("The total effect of each of the 3 exposures (`g0`, `g1` and `g2`) "
                              "on `case`, one at a time, as the difference in each exposure's mean")


# ── 4 · clusters above the person ────────────────────────────────────────────


def _multisite(sites: int) -> pd.DataFrame:
    """The audit's two multi-site fixtures, regenerated from their generators (repro.tar.gz):
    I/make_fixtures.py's F4 (12 sites, 600 rows, seed 42, after its F2 and F3 draws) and
    I-skeptic/mkfx.py's I5 (10 sites, 700 rows, seed 2026, after its I1, I3 and I4 draws). Sodium
    has no effect within site; site confounds it."""
    if sites == 12:
        rng = np.random.default_rng(42)
        n = 600
        age = rng.normal(55, 8, n).round(0)  # F2
        fiber = rng.gamma(4, 5, n).round(1)
        rng.exponential(1 / (0.02 * np.exp(0.04 * (age - 55) - 0.03 * (fiber - 20))))
        rng.uniform(1, 15, n)
        rng.normal(0, 1, n)  # F3: the ordinal latent
        site = rng.integers(0, 12, n)
        effect = rng.normal(0, 1.0, 12)[site]
        x = rng.normal(0, 1, n) + 0.8 * effect
        y = 0.0 * x + 1.5 * effect + rng.normal(0, 1, n)
        ids = [f"P{i:04d}" for i in range(n)]
    else:
        rng = np.random.default_rng(2026)
        n = 4000
        rng.normal(50, 10, n); rng.gamma(4, 5, n); rng.normal(0, 5, n)  # I1
        n = 3000
        entry = rng.uniform(0, 14, n); rng.normal(0, 4, n); age = rng.normal(55, 8, n)  # I3
        rng.exponential(1 / (0.03 * np.exp(0.04 * (age - 55))))
        m = 2400
        rng.random(m); rng.integers(1, 16, m); rng.integers(1, 3, m)  # I4
        rng.normal(2100, 500, m); rng.gamma(4, 5, m); rng.normal(0, 1, m)
        n = 700
        site = rng.integers(0, 10, n)
        effect = rng.normal(0, 1, 10)[site]
        x = rng.normal(0, 1, n) + 0.8 * effect
        y = 0 * x + 1.5 * effect + rng.normal(0, 1, n)
        ids = [f"M{i:04d}" for i in range(n)]
    return pd.DataFrame({"participant_id": ids, "site": [f"S{k:02d}" for k in site],
                         "sodium_g": (x + 3).round(3), "sbp_change": y.round(3)})


# The audit's numbers for each fixture: site fixed effects with site-clustered errors, statsmodels'
# default (CR1 on z), as the audit computed them (RO-08: "−0.04 (−0.11, 0.03) to −0.08 (−0.15,
# −0.005)").
AUDIT = {12: (-0.04, -0.11, 0.03), 10: (-0.08, -0.15, -0.005)}


@pytest.mark.parametrize("sites", [12, 10])
def test_4_a_cluster_question_fires_and_site_fixed_effects_reproduce_the_audit(sites, tmp_path):
    """AUDIT_REPORT §5 WP17 test 4 (RO-08), through the real HTTP API. The audit's −0.04
    (−0.11, 0.03) is the 12-site fixture's and −0.08 (−0.15, −0.005) the 10-site one's; the
    package text names the 10-site fixture with the 12-site numbers, so both are run.

    References, each computed outside the engine: the fixture is the audit's (statsmodels' OLS with
    C(site) and its default cluster covariance, CR1 on z, reproduces the audit's three numbers to
    their printed digits); the app's estimate is that OLS's to 10⁻⁸; its interval is CR2 with
    Bell–McCaffrey df written out by definition (``references.cr2_by_definition``), which with
    G ≤ 12 is wider than the audit's z interval, as MA-06 requires."""
    frame = _multisite(sites)
    path = _csv(frame, tmp_path, f"multisite_{sites}")
    g = frame["site"].astype("category").cat.codes
    audit = smf.ols("sbp_change ~ sodium_g + C(site)", frame).fit(cov_type="cluster",
                                                                    cov_kwds={"groups": g})
    estimate, low, high = AUDIT[sites]
    ci = audit.conf_int().loc["sodium_g"]
    digits = 2 if sites == 12 else 3
    assert round(audit.params["sodium_g"], 2) == estimate
    assert (round(ci[0], 2), round(ci[1], digits)) == (low, high)

    truth = Truth({}, fixture=f"the {sites}-site cohort")
    with local_server(tmp_path / "home") as client:
        drive = open_project(client, path, truth)
        drive.decide({"kind": "set_lens", "lenses": ["clinical"]})
        drive.reach("target")
        drive.decide({"kind": "set_target", "column": "sbp_change"})
        drive.reach("purpose")
        drive.decide({"kind": "set_purpose", "purpose": "inference"})
        drive.answer("grain", {"kind": "set_grain", "grain": "one_row_per_unit"})
        drive.reach("roles")
        drive.decide_roles({"participant_id": "identifier", "site": "cluster",
                            "sodium_g": "exposure"})
        assert drive.reach("clusters")["status"] == "open"  # the question fires
        none = drive.post({"kind": "set_clusters", "column": None})
        assert none.status_code == 409 and none.json()["error"]["code"] == "grouping_reads"
        how = drive.post({"kind": "set_clusters", "column": "site"})
        assert how.status_code == 409 and how.json()["error"]["code"] == "how_adjusted"
        drive.decide(how.json()["error"]["exits"][0]["decision"])  # adjust and cluster
        drive.answer("exclusions", {"kind": "set_exclusions", "rules": []})
        drive.answer("missing", {"kind": "set_missing", "strategy": "complete_case"})
        drive.answer("split", {"kind": "set_split", "holdout": 0.0, "seed": 0, "folds": 5})
        drive.answer_plan("sodium_g")  # after the split (MODELING_SEQUENCE §1 steps 2–3)
        assert drive.reach("adjustment")["status"] == "not_applicable"  # site is the fixed effect
        drive.reach("models")
        drive.decide({"kind": "select_models", "models": ["linear"]})
        fit = drive.artifact("fit")
    model = fit["models"][0]
    info = model["inference"]
    assert info["covariance"] == "CR2" and info["grouped_by"] == "site"
    assert info["n_clusters"] == sites
    row = next(r for r in model["coefficients"] if r["feature"] == "sodium_g")
    assert row["estimate"] == pytest.approx(audit.params["sodium_g"], rel=1e-8)
    assert round(row["estimate"], 2) == estimate  # the audit's −0.04 / −0.08
    X = sm.add_constant(pd.concat([frame[["sodium_g"]],
                                   pd.get_dummies(frame["site"], drop_first=True, dtype=float)],
                                  axis=1)).to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(X, frame["sbp_change"].to_numpy(), rcond=None)
    V, df = ref.cr2_by_definition(X, frame["sbp_change"].to_numpy() - X @ beta, g.to_numpy())
    half = stats.t.ppf(0.975, df[1]) * math.sqrt(V[1, 1])
    assert row["ci_low"] == pytest.approx(beta[1] - half, rel=1e-6)
    assert row["ci_high"] == pytest.approx(beta[1] + half, rel=1e-6)
    assert row["ci_low"] < ci[0] and row["ci_high"] > ci[1]  # wider than z with few clusters
    assert "conditional on an intercept for each `site` (fixed effects)" in fit["estimand"]["caption"]


def test_4_under_prediction_the_grouping_names_internal_external_validation(tmp_path):
    """Under prediction the cluster answer carries no intercepts (refused, with the exit that
    records the grouping alone) and the split question's internal–external option folds by it."""
    from turbotab.core.seal import plan

    frame = _multisite(12)
    table = Ingested(_csv(frame, tmp_path, "multisite_12"), tmp_path)
    state = ProjectState(lens=["clinical"], target="sbp_change", task="regression",
                         purpose="prediction", grain={"grain": "one_row_per_unit"},
                         roles={"participant_id": "identifier", "site": "cluster",
                                "sodium_g": "exposure"})
    with pytest.raises(Refusal) as refused:
        d.validate({"kind": "set_clusters", "column": "site", "adjust": "fixed_effects"},
                   {"state": state})
    assert refused.value.code == "not_inference"
    grouped = state.model_copy(update={"clusters": d.ClusterSpec(column="site")})
    with table.store() as store:
        sealed = plan(grouped, table.frame().index.to_numpy(), store, "regression")
    ie = next(o for o in sealed["validation"]["options"] if o["validation"] == "internal_external")
    assert ie["cluster"] == "site"
    assert estimand.fixed_effects_column(grouped) is None  # no intercepts under prediction


# ── 5 · the energy order under prediction ────────────────────────────────────


def _energy_outcome(n: int = 1000, seed: int = 17) -> pd.DataFrame:
    """An outcome that follows total energy (the audit's RO-07 setting): energy drives it, and
    the nutrients carry energy as real intakes do."""
    rng = np.random.default_rng(seed)
    energy = rng.lognormal(np.log(2000), 0.3, n)
    fat = 0.35 * energy / 9 * rng.lognormal(0, 0.15, n)
    protein = 0.16 * energy / 4 * rng.lognormal(0, 0.15, n)
    carb = (energy - 9 * fat - 4 * protein).clip(50) / 4
    y = 0.004 * energy + 0.01 * fat + rng.normal(0, 2, n)
    return pd.DataFrame({"energy_kcal": energy.round(1), "fat_g": fat.round(2),
                         "protein_g": protein.round(2), "carbohydrate_g": carb.round(2),
                         "y": y.round(3)})


def test_5_under_prediction_the_methods_that_keep_energy_lead(tmp_path):
    """AUDIT_REPORT §5 WP17 test 5 (RO-07). The labels and the ranking under prediction put every
    method that keeps total energy before every one that drops it, and the line states the
    residual method's energy-dropped form discards energy's signal. Reference for the order: on
    data where the outcome follows energy, scikit-learn's cross-validated R² of the
    energy-keeping design (nutrient plus energy) is far above the energy-dropped residual's, as
    the audit measured (0.58–0.79 against 0.01–0.39)."""
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold, cross_val_score

    from turbotab.core.stages.proposals import build_proposals

    frame = _energy_outcome()
    table = Ingested(_csv(frame, tmp_path, "energy_outcome"), tmp_path)
    roles = {"energy_kcal": "energy", "fat_g": "exposure", "protein_g": "exposure",
             "carbohydrate_g": "exposure"}
    proposals = build_proposals(table.frame(), table.info["columns"], lens=["dietary"], target="y",
                                roles=roles, purpose="prediction")
    labels = proposals["labels"]["energy_adjustment"]
    keys = [o["key"] for o in labels["options"]]
    keeps = {"standard", "residual", "density_multivariate", "partition", "all_components"}
    drops = {"none", "residual_energy_dropped", "density"}
    assert max(keys.index(k) for k in keeps if k in keys) < min(keys.index(k) for k in drops)
    assert keys == proposals["energy"]["ranking"]["order"]
    assert "discards total energy's signal" in labels["tension"]
    dropped = next(o for o in labels["options"] if o["key"] == "residual_energy_dropped")
    assert dropped["sound"]["verdict"] == "unsound" and "discards" in dropped["sound"]["reason"]

    folds = KFold(5, shuffle=True, random_state=0)
    y = frame["y"].to_numpy()
    keep_X = frame[["fat_g", "energy_kcal"]].to_numpy()
    slope = np.polyfit(frame["energy_kcal"], frame["fat_g"], 1)[0]
    residual = (frame["fat_g"] - slope * (frame["energy_kcal"] - frame["energy_kcal"].mean()))
    keep = cross_val_score(LinearRegression(), keep_X, y, cv=folds, scoring="r2").mean()
    drop = cross_val_score(LinearRegression(), residual.to_numpy()[:, None], y, cv=folds,
                           scoring="r2").mean()
    print(f"\nCV R²: energy kept {keep:.3f}, energy dropped (residual) {drop:.3f}")
    assert keep > drop + 0.3


# ── the question order: the opening sequence, then the modeling sequence's steps 2–3 ──


def test_the_exposure_and_the_adjustment_set_are_asked_after_the_seal_and_before_the_energy_model():
    """MODELING_SEQUENCE §1 is "everything after the seal", in "the order in which the Router
    asks": step 2 the exposure and estimand, step 3 the adjustment set, step 4 the domain
    transforms (the energy model), step 9 the shelf; missing values are "asked before the seal".
    OPENING_SEQUENCE §01 ends the opening at eligibility, then the SEAL, and "Nothing may be
    resequenced." So under the integrated Router (WP16–WP18) the estimand and the adjustment come
    after the exclusions, the missing values and the split, and before the energy model and the
    models; the teaching cards follow the same order. Reference: the repository's own documents,
    quoted."""
    from turbotab.core import teaching
    from turbotab.core.interview import QUESTION_KEYS

    root = Path(__file__).resolve().parents[4] / "docs"
    modeling = (root / "turbotab-next/MODELING_SEQUENCE.md").read_text("utf-8")
    opening = (root / "turbotab-next/reference/OPENING_SEQUENCE.md").read_text("utf-8")
    assert "# The modeling sequence — everything after the seal" in modeling
    assert "This is the order in which the Router asks." in modeling
    assert "| 6 | **Missing data** (asked before the seal, executed here)" in modeling
    assert "| — | **SEAL** | | |" in opening and "Nothing may be resequenced." in opening

    at = QUESTION_KEYS.index
    assert at("exclusions") < at("missing") < at("split") < at("estimand") < at("adjustment") \
        < at("energy_adjustment") < at("models")
    assert [k for k in teaching.QUESTION_KEYS] == list(QUESTION_KEYS)

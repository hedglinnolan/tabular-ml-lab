"""The previews keep the leash (calm/FOUNDATION §5 rules 6–8; MODELING_SEQUENCE §4; BLUEPRINT §14):
the defects found before the pause, each held to the fit's own behavior or a hand computation.

1. **No outcome-model estimate before the lock.** Under inference nothing estimated from the
   outcome model appears before the analysis plan is locked. The energy question's
   energy-dropped residual quoted the outcome model's coefficient, and the diagnostic response
   counted the rows above the fitted model's Cook's distance. The walk below previews every
   registered decision kind on two tables that differ only in which row holds which outcome value
   (the same values, the same blanks): every number a preview may show (the outcome's own
   distribution, its counts, the predictors and their relations) is the same on both, and any
   number estimated from the outcome model is not. So under inference before the lock both tables
   preview identically, kind by kind, as the server serves them (its refusal, or its preview).
   After the lock the same walk tells them apart: the detector sees an estimate where one may
   appear.
2. **Never an empty canvas.** A preview that cannot draw says in one line what is missing and the
   question that settles it, and recording that answer makes it draw: the adjustment set and the
   model sequence before an exposure is declared, and the modeling previews whose fit asks first
   (an unsettled reading), which offer the fit's own ask.
3. **The survey design as the fit applies it.** Where a grouping's rows span PSUs, the model
   sequence previews the fit's refusal with the sample-only exit, as the survey and grouping
   previews already do.
4. **Population estimand without a design-based estimator: block and record.** A chosen family
   with no design-based estimator, a marginal measure, a scale's correction and a time-varying
   lane preview the block the fit, the effects, the scales and the time-varying stages record,
   with their exits and recording the answer as it is.

Every expected value is the stage's own artifact once the answer is recorded, or a hand
computation (NumPy least squares, Cook's distance and correlations written out here); the server's
refusal is its validators' on the same state.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from turbotab.core import consequences, decisions as d
from turbotab.core.tests import modeling_fixtures as mf
from turbotab.core.tests.acceptance import preview_fixtures as F
from turbotab.core.tests.acceptance.preview_harness import Project, view
from turbotab.core.tests.acceptance.test_previews_kinds import Planned, vocabulary

A = d.AdjustmentAnswer
CONFOUNDER = dict(causes_exposure="yes", causes_outcome="yes", after_exposure="no")
UPTO = ["design", "cohort", "split", "proposals", "findings", "fit", "effects", "shelf"]


# ── the walk's fixture: one table, and the same table with its outcome re-paired ─────────────


def walk_table(repaired: bool) -> pd.DataFrame:
    """MODELING_FIXTURES' NHANES-shaped table (glucose depends on the diet, age, BMI and sex) with
    24 recruiting sites from their own stream. ``repaired``: the same glucose values given to the
    rows in a fixed random order, so the outcome keeps its values and its blanks but no longer
    relates to anything."""
    frame = mf.nhanes_like(600, seed=5)
    rng = np.random.default_rng(55)
    frame["site"] = [f"S{k:02d}" for k in rng.integers(0, 24, len(frame))]
    if repaired:
        frame["glucose"] = np.random.default_rng(56).permutation(frame["glucose"].to_numpy())
    return frame


ROLES = {**{c: "excluded" for c in walk_table(False).columns},
         "SEQN": "identifier", "age": "covariate", "gender": "covariate", "bmi": "covariate",
         "kcal": "energy", "carb": "exposure", "protein": "exposure", "fat_total": "exposure",
         "site": "cluster"}
ANSWERS = {"age": CONFOUNDER, "gender": CONFOUNDER, "protein": CONFOUNDER, "fat_total": CONFOUNDER,
           "bmi": dict(causes_exposure="unknown", causes_outcome="yes", after_exposure="unknown")}
NUTRIENTS = dict(energy_column="kcal", nutrients=["carb", "protein", "fat_total"])


def walk_state(**update: Any) -> d.ProjectState:
    """The plan answered under inference, the estimates computed and not yet displayed: carb's
    substitution for other calories on glucose, its adjustment set, Model 1, the sites, the
    residual energy model and the linear family."""
    state = d.ProjectState(
        lens=["dietary"], target="glucose", task="regression", purpose="inference",
        roles=dict(ROLES), role_confirmations=dict(ROLES),
        grain=d.GrainSpec(grain="one_row_per_unit", id_column="SEQN"),
        shape_confirmations={"code_or_count:age": "amount"},
        exclusions=[], missing=d.MissingSpec(strategy="complete_case"),
        split=d.SplitSpec(holdout=0.0, seed=0, folds=5), models=["linear"],
        estimand=d.EstimandSpec(exposure="carb", contrast="substitution",
                                measure="mean_difference"),
        adjustment={c: A(exposure="carb", **a) for c, a in ANSWERS.items()},
        model_sequence=d.ModelSequenceSpec(exposure="carb", model_1=["age", "gender"]),
        clusters=d.ClusterSpec(column="site", adjust="cluster_only"),
        energy_adjustment=d.EnergyAdjustment(method="residual", **NUTRIENTS),
        column_units=mf.grams("carb", "protein", "fat_total"))
    return state.model_copy(update=update)


DROPPED = d.SetEnergyAdjustment(method="residual_energy_dropped", **NUTRIENTS)
AT = "2026-10-05T00:00:00Z"
# A log in which the energy-dropped residual was answered and then replaced: undoing the second
# answer previews the first.
EARLIER = d.DecisionRecord(id="a" * 32, seq=1, at=AT, decision=DROPPED)
LATER = d.DecisionRecord(id="b" * 32, seq=2, at=AT, decision=d.SetEnergyAdjustment(
    method="residual", **NUTRIENTS))

WALK: list[Any] = [
    *[d.SetEnergyAdjustment(method=m, **NUTRIENTS) for m in
      ("residual", "residual_energy_dropped", "standard", "density", "density_multivariate",
       "partition", "all_components")],
    d.SetEnergyAdjustment(method="none"),
    d.Revert(decision_id=LATER.id),
    d.SelectModels(models=["linear", "elastic_net"]),
    d.SetEstimand(exposure="protein", contrast="substitution", measure="mean_difference"),
    d.SetAdjustment(exposure="carb", answers={"bmi": d.CovariateAnswers(**CONFOUNDER)}),
    d.SetModelSequence(exposure="carb", model_1=["age"]),
    d.SetMultiplicity(method="bh"),
    d.SetExposureForm(column="carb", form="spline", knots=4),
    d.SetForms(forms={"carb": d.ExposureFormSpec(form="spline", knots=4)}),
    d.SetModification(modifier="gender"),
    d.SetClusters(column="site", adjust="fixed_effects"),
    d.SetSurvey(estimand="sample"),
    d.SetOutcomeScale(column="glucose", scale="log"),
    d.SetCategorical(columns=["gender"]),
    d.SetTask(column="glucose", task="regression"),
    d.SetSensitivity(analyses=[d.SensitivityAnalysis(label="800–4,000 kcal", rules=[
        d.ExclusionRule(column="kcal", low=800, high=4000, reason="implausible intakes")])]),
    d.RespondDiagnostic(exposure="carb", check="influence", action="without_influential"),
    d.SetExclusions(rules=[d.ExclusionRule(column="kcal", low=800, high=4000,
                                           reason="implausible intakes")]),
    d.SetMissing(strategy="complete_case"),
    d.SetSplit(holdout=0.2, seed=0, folds=5),
    d.SetRoles(roles={**ROLES, "waist": "covariate"}),
    d.ConfirmRole(column="bmi", role="covariate"),
    d.ConfirmReading(reading="code_or_count", column="age", value="amount"),
    d.ConfirmReadings(items=[d.ReadingItem(reading="code_or_count", column="age",
                                           value="amount")]),
    d.SetColumnUnit(column="kcal", unit="kcal", days=1),
    d.SetSubstitution(donor="fat_total", recipient="carb", step_kcal=100),
    d.SetExplain(curves="ale"),
    d.SetLevers(forms="rule"),
    d.SetSelection(method="elastic_net"),
    d.SetIntendedUse(use="risk_estimation"),
    d.SetUpdating(method="shrinkage"),
    d.OpenSeal(),
    d.SetLens(lenses=["dietary"]),
    d.SetTarget(column="glucose"),
    d.SetPurpose(purpose="inference"),
    d.SetCausal(exposure="carb", method="dml_plr"),
    d.SetMeasurementError(method="regression_calibration", exposures=["carb"]),
    d.SetBatch(column="site", method="covariate"),
    d.SetGrain(grain="one_row_per_unit", id_column="SEQN"),
    d.ApplyRepair(finding_id="pack::survey::sentinel_codes", option="to_missing",
                  params={"code": 9}),
    d.ImportCodebook(codebook="c0123456789", name="dictionary.csv", form="table",
                     items=[d.ReadingItem(reading="unit", column="weight", value="kg")],
                     n_entries=1, n_matched=1),
    d.JoinFiles(file="f0123456789", on="SEQN", name="labs.csv",
                counts=d.JoinCounts(relation="one-to-one", table_rows=600, file_rows=580,
                                    matched_keys=290, table_unmatched=20, file_unmatched=0,
                                    rows=600, added_columns=3)),
    d.SetAggregation(method="mean", outcome="mean"),
    d.SetEvent(column="glucose", level="1"),
    d.SetFeatureTable(label="metabolite", annotations=["mz"]),
    d.SetFollowUp(column="glucose", time_column="age"),
    d.SetOrientation(orientation="feature_major"),
    d.SetOutcomeOrder(column="gender", levels=["female", "male"]),
    d.SetRepeatKind(repeat_kind="repeats"),
    d.SetScales(scales=[d.ScaleSpec(name="body", items=["weight", "waist", "height"], low=1,
                                    high=5, kind="reflective")]),
    d.SetTemporal(temporal=True, time_column="cycle_begin_year"),
    d.SetTimeVarying(exposure="carb", method="msm_iptw", ordering="exposure_precedes_outcome"),
    d.SetUnit(unit="unit"),
    d.SetUsualIntake(nutrient="carb", model="amount_only", days=["carb", "sugar"]),
]


@pytest.fixture(scope="module")
def walk(tmp_path_factory):
    """The two tables run through the stage graph to the effects, as the server would have them
    computed and not yet displayed."""
    folder = tmp_path_factory.mktemp("previews_leash_walk")
    pair = (Project(walk_table(False), folder / "paired", walk_state(), upto=UPTO),
            Project(walk_table(True), folder / "repaired", walk_state(), upto=UPTO))
    yield pair
    for project in pair:
        project.close()


def served(project: Project, decision: Any, state: d.ProjectState) -> dict[str, Any]:
    """What ``POST /preview`` returns for ``decision`` on ``state``: its refusal (the server
    validates first, 409), else the planned preview, as JSON."""
    ctx = {"state": state, "columns": list(project.store.columns), "target": state.target,
           "artifact": lambda stage: project.artifacts.get(stage)}
    try:
        parsed = d.validate(decision, ctx)
    except d.Refusal as refused:
        return {"refused": refused.to_dict()}
    pctx = project.context(state=state)
    if parsed.kind == "revert":
        pctx.settings["records"] = [EARLIER, LATER]
    result, _ = project.preview(parsed, ctx=pctx)
    return result.model_dump(mode="json")


def differing(pair: tuple[Project, Project], state: d.ProjectState,
              decisions: list[Any]) -> list[str]:
    paired, repaired = pair
    out = []
    for decision in decisions:
        one, other = served(paired, decision, state), served(repaired, decision, state)
        if json.dumps(one, sort_keys=True) != json.dumps(other, sort_keys=True):
            out.append(f"{decision.kind} ({getattr(decision, 'method', '')}): "
                       f"{json.dumps(one, sort_keys=True)[:400]}")
    return out


def test_1_the_walk_covers_every_registered_kind():
    assert {x.kind for x in WALK} == consequences.registered_kinds()


def test_1_the_two_tables_differ_only_in_the_outcome_models_estimates(walk):
    """The fixture's premise, held by hand: the outcome has the same values and blanks on both
    tables and every other column is the same, while the outcome model is not: NumPy least squares
    of glucose on carb, kcal, age, BMI and sex explains 38% of its variance on one table and under
    1% on the other, and the fit's own coefficients differ."""
    one, other = walk_table(False), walk_table(True)
    assert sorted(one["glucose"]) == sorted(other["glucose"])
    assert one.drop(columns="glucose").equals(other.drop(columns="glucose"))
    r2 = []
    for t in (one, other):
        X = np.column_stack([np.ones(len(t)), t["carb"], t["kcal"], t["age"], t["bmi"],
                             t["gender"] == "male"]).astype(float)
        y = t["glucose"].to_numpy(float)
        resid = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        r2.append(1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum())
    assert r2[0] > 0.35 and r2[1] < 0.01
    fitted = [{c["feature"]: c["estimate"] for c in p.artifacts["fit"].data["models"][0]
               ["coefficients"]} for p in walk]
    assert fitted[0]["age"] == pytest.approx(0.25, abs=0.03)
    assert fitted[1]["age"] == pytest.approx(0.0, abs=0.03)


def test_1_before_the_lock_no_preview_shows_an_outcome_model_estimate(walk):
    assert differing(walk, walk_state(), WALK) == []


def test_1_after_the_lock_the_walk_sees_the_estimates_it_may_show(walk):
    """The positive control: once the plan is locked the energy-dropped residual quotes the
    outcome model's coefficient (audit ME-03), directly and through a revert, so the two tables'
    previews differ, and the walk's detector sees it."""
    found = differing(walk, walk_state(plan_locked=True), [DROPPED, d.Revert(decision_id=LATER.id)])
    assert [x.split(" ")[0] for x in found] == ["set_energy_adjustment", "revert"]
    assert all("coefficient" in x for x in found)


def test_1_the_energy_dropped_residual_says_what_it_does_without_the_coefficient(walk):
    """Before the lock the caption is the method's own picture: carb's correlation with kcal on
    every analyzed row (NumPy), none after the residual, and kcal leaving the outcome model; with
    the residual already on record, the same choice says what it adds to it. After the lock it
    quotes the coefficients, as audit ME-03 asked."""
    paired, _ = walk
    unanswered = walk_state(energy_adjustment=None)
    result = served(paired, DROPPED, unanswered)
    frame = walk_table(False)
    r = np.corrcoef(frame["carb"], frame["kcal"])[0, 1]
    assert result["views"][0]["caption"] == (
        f"`carb` correlates {r:.2f} with `kcal`; after residual adjustment, 0.00; `kcal` leaves "
        f"the outcome model.")
    assert "coefficient" not in json.dumps(result)
    recorded = served(paired, DROPPED, walk_state())
    assert recorded["views"][0]["caption"] == (
        "Recorded now: `carb_adj` correlates 0.00 with `kcal`; with this choice, `carb_adj` "
        "correlates 0.00; `kcal` leaves the outcome model.")
    locked = served(paired, DROPPED, unanswered.model_copy(update={"plan_locked": True}))
    assert locked["views"][0]["caption"].startswith(
        "`kcal` leaves the outcome model: `carb_adj` coefficient ")


# ── the diagnostic response: the fitted model's influence check waits for the estimates ──────

INFLUENCE = d.RespondDiagnostic(exposure="fiber", check="influence", action="without_influential")


@pytest.fixture(scope="module")
def influential(tmp_path_factory):
    """ESTIMAND's 300-row cohort with one row far out on the exposure and a wild outcome, the
    influence check's own fixture (``test_estimand_4_diagnostics``, where R flags that row)."""
    from turbotab.core.tests.acceptance import estimand_fixtures as ef

    frame = ef.cohort(300, seed=2)
    frame.loc[0, ["fiber", "glucose"]] = [70.0, 260.0]
    roles = {c: r for c, r in ef.ROLES.items() if c != "bmi"}
    answers = {c: a for c, a in ef.ANSWERS.items() if c != "bmi"}
    state = ef.state(target="glucose", task="regression", measure="mean_difference", roles=roles,
                     answers=answers)
    project = Project(frame, tmp_path_factory.mktemp("previews_leash_influence"), state,
                      upto=["design", "cohort", "split", "fit", "effects"])
    yield project, frame
    project.close()


def cooks_flagged(frame: pd.DataFrame) -> tuple[int, float]:
    """By hand: least squares of glucose on fiber, age, sex, smoking and activity; each row's
    Cook's distance e²h / (p s² (1 − h)²), against the median of F(p, n − p)."""
    from scipy import stats

    from turbotab.core.tests.acceptance import estimand_fixtures as ef

    X = ef.design_matrix(frame, ["fiber", "age", "sex", "smoking", "activity"]).to_numpy(float)
    y = frame["glucose"].to_numpy(float)
    n, p = X.shape
    h = np.einsum("ij,jk,ik->i", X, np.linalg.inv(X.T @ X), X)
    e = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    s2 = (e @ e) / (n - p)
    cooks = e ** 2 * h / (p * s2 * (1 - h) ** 2)
    threshold = float(stats.f.ppf(0.5, p, n - p))
    return int((cooks > threshold).sum()), threshold


def test_1_the_influence_check_is_not_previewed_before_the_estimates(influential):
    """Before the lock the check has not been shown, so the server refuses the response, and the
    preview itself counts no row from the fitted model; after the lock it counts the rows Cook's
    distance flags, by hand."""
    project, frame = influential
    flagged, threshold = cooks_flagged(frame)
    assert flagged == 1
    state = project.state
    refused = served(project, INFLUENCE, state)
    assert refused["refused"]["error"]["code"] == "check_not_shown"
    assert "Cook" not in json.dumps(refused)
    result, _ = project.preview(INFLUENCE, ctx=project.context(state=state))
    assert result.views == [] and result.caution is None
    assert result.note == ("The checks are read with the estimates, which are not shown yet; "
                           "nothing about them is drawn before.")
    locked = served(project, INFLUENCE, state.model_copy(update={"plan_locked": True}))
    [flow] = locked["views"]
    assert flow["after"][-1]["dropped"] == flagged
    assert flow["caption"] == (f"`{flagged}` rows above Cook's distance `{threshold:.3g}` leave "
                               f"the refit shown beside the estimate.")


# ── 2 · never an empty canvas: what is missing, and the question that settles it ─────────────

GENERIC = "Nothing about this choice can be shown on your data yet."
ADJUSTMENT = d.SetAdjustment(exposure="fiber", answers={
    c: d.CovariateAnswers(**a) for c, a in F.ESTIMAND_ANSWERS.items()})
SEQUENCE = d.SetModelSequence(exposure="fiber", model_1=["age", "sex"])
FIBER = d.SetEstimand(exposure="fiber", measure="mean_difference")


@pytest.fixture(scope="module")
def undeclared(tmp_path_factory):
    """ESTIMAND's cohort under inference with no exposure declared yet (so no adjustment answers
    and no model sequence)."""
    state = F.estimand_state(estimand=None, adjustment=None, model_sequence=None)
    planned = Planned(Project(F.estimand_table(), tmp_path_factory.mktemp("previews_leash_none"),
                              state, upto=["design", "cohort", "split"]))
    yield planned
    planned.project.close()


def refusal(decision: Any, state: d.ProjectState) -> d.Refusal:
    """The server's answer to recording ``decision`` on ``state``: its validators' refusal."""
    with pytest.raises(d.Refusal) as refused:
        d.validate(decision, {"state": state})
    return refused.value


def test_2_the_adjustment_set_before_an_exposure_names_the_question_that_settles_it(undeclared):
    """No exposure: the preview says what the server says of the answer (each covariate is asked
    about against the exposure, and the exposure question comes first). The question it names
    settles it: with the exposure declared, the same answers draw the set the served fit adjusts
    for once recorded."""
    from turbotab.core.estimand import annotate_fit
    from turbotab.core.voice import question_name

    state = undeclared.project.state
    result, _ = undeclared.preview("adjustment", ADJUSTMENT)
    assert result.views == [] and result.caution is None
    assert result.note == refusal(ADJUSTMENT, state).message != GENERIC
    assert question_name("estimand") in result.note
    declared = d.fold_onto(state, FIBER)
    drawn, _ = undeclared.preview("adjustment_declared", ADJUSTMENT, state=declared)
    vocabulary(drawn)
    after, out = undeclared.after("adjustment_declared", ADJUSTMENT, ["design", "fit"],
                                  state=declared)
    served_fit = annotate_fit(out["fit"].data, after)["estimand"]
    lineage = view(drawn, "lineage")
    assert {n.column for n in lineage.after.nodes if n.lane == "matrix"} - {"fiber"} == set(
        served_fit["adjusted"])


def test_2_the_model_sequence_before_the_plan_names_the_questions_that_settle_it(undeclared):
    """No exposure: the preview says what the server says (Model 1 is a part of the primary
    model's adjustment set), naming both questions; answered, the same declaration draws the
    effects stage's models frame by frame."""
    from turbotab.core.voice import question_name

    state = undeclared.project.state
    result, _ = undeclared.preview("sequence", SEQUENCE)
    assert result.views == [] and result.caution is None
    assert result.note == refusal(SEQUENCE, state).message != GENERIC
    assert question_name("adjustment") in result.note
    planned = d.fold_onto(d.fold_onto(state, FIBER), ADJUSTMENT.model_copy(
        update={"exposure": "fiber"}))
    drawn, _ = undeclared.preview("sequence_planned", SEQUENCE, state=planned)
    vocabulary(drawn)
    _, out = undeclared.after("sequence_planned", SEQUENCE, ["effects"], state=planned)
    sequence = {s["key"]: s["adjusted_for"] for s in out["effects"].data["families"][0]["sequence"]}
    lineage = view(drawn, "lineage")
    frames = [f.lineage for f in lineage.story] + [lineage.after]
    assert [sorted(n.column for n in f.nodes if n.lane == "matrix" and n.column != "fiber")
            for f in frames] == [sorted(sequence[k]) for k in sequence]


def test_2_a_modeling_preview_without_its_rows_says_which_answer_draws_them(undeclared):
    """No rows to draw on: before the held-out rows question is answered the preview names it, and
    while the rows are read again after an answer it says so; never the generic line."""
    from turbotab.core.voice import question_name

    planned = d.fold_onto(d.fold_onto(undeclared.project.state, FIBER), ADJUSTMENT)
    for split, said in ((None, question_name("split")), (planned.split, "being read again")):
        state = planned.model_copy(update={"split": split})
        for decision in (d.SetEstimand(exposure="supplement", measure="mean_difference"),
                         SEQUENCE):
            ctx = undeclared.project.context(state=state)
            ctx.training_row_ids = None
            result, _ = undeclared.project.preview(decision, ctx=ctx)
            assert result.views == [] and said in (result.note or ""), (decision.kind, result.note)


def design_asks(project: Project, decision: Any, state: d.ProjectState) -> Any:
    """The design stage's own ask once ``decision`` is recorded on ``state`` (``readings.
    Unsettled``): the fit computes nothing on an unsettled reading (BLUEPRINT §14)."""
    from turbotab.core.readings import Unsettled

    base = project.state
    project.state = state
    try:
        with pytest.raises(Unsettled) as asked:
            project.after(decision, upto=["design"])
    finally:
        project.state = base
    return asked.value


def the_ask(asked: Any) -> tuple[str, list[dict[str, Any]]]:
    return str(asked), [e["decision"] for e in asked.exits if e.get("decision")]


def test_2_the_modeling_previews_offer_the_fits_ask_on_an_unsettled_reading(walk):
    """``age``'s code-or-amount reading unsettled: the design stage asks before it builds any
    matrix. The families' preview offers that ask instead of matrices built on a guess; the energy
    preview keeps the method's own picture (it reads only carb and kcal) but draws no model matrix
    and offers the ask; the causal lane's preview offers it with each confirmation."""
    paired, _ = walk
    state = walk_state(shape_confirmations={})
    for decision in (d.SelectModels(models=["linear", "elastic_net"]),
                     d.SetEnergyAdjustment(method="density", **NUTRIENTS),
                     d.SetCausal(exposure="carb", method="dml_plr")):
        expected = the_ask(design_asks(paired, decision, state))
        assert sorted(x["value"] for x in expected[1]) == ["amount", "code"]
        result, _ = paired.preview(decision, ctx=paired.context(state=state))
        assert result.caution is not None, (decision.kind, result.note)
        assert (result.caution.text, [x.decision for x in result.caution.exits]) == expected
        assert result.note is None
        kinds = [v.kind for v in result.views]
        if decision.kind == "set_energy_adjustment":
            assert kinds == ["relationship", "distribution"]
        else:
            assert kinds == []


# ── 3 · the survey design as the fit applies it: the model sequence ──────────────────────────


@pytest.fixture(scope="module")
def spanning(tmp_path_factory):
    """REPAIR-PREVIEWS' spanning layout (ten sites' rows drawn into PSUs at random) with the
    grouping and the surveyed population recorded, so the fit refuses every coefficient."""
    from turbotab.core.tests.acceptance import test_previews_repair as R

    planned = R.project(tmp_path_factory.mktemp("previews_leash_spanning"), "spanning",
                        clusters=d.ClusterSpec(column="site", adjust="cluster_only"),
                        survey=d.SurveySpec(**R.POPULATION.model_dump(exclude={"kind"})))
    yield planned
    planned.project.close()


def test_3_the_model_sequence_previews_the_refusal_the_fit_and_effects_record(spanning):
    """The fit refuses every coefficient (a site's rows span PSUs) and the effects stage reports
    Model 2 alone, with nothing estimated and that reason: the model sequence's preview says so,
    with the sample-only exit, beside the declared models it draws."""
    decision = d.SetModelSequence(exposure="fiber", model_1=["age"])
    result, _ = spanning.preview("sequence", decision)
    fit = spanning.project.artifacts["fit"].data["models"][0]
    refused = fit["inference"]["refused"]
    assert fit["coefficients"] == [] and refused.startswith("10 `site` units have rows in more")
    _, out = spanning.after("sequence", decision, ["effects"])
    [model] = out["effects"].data["families"][0]["sequence"]
    assert (model["key"], model["effects"], model["concerns"]) == ("model_2", None, [refused])
    assert result.caution is not None and result.caution.text == refused
    assert [x.decision for x in result.caution.exits] == [
        {"kind": "set_survey", "estimand": "sample"}, decision.model_dump(mode="json")]
    vocabulary(result)


# ── 4 · population estimand without a design-based estimator: block and record ───────────────


@pytest.fixture(scope="module")
def svy(tmp_path_factory):
    planned = Planned(F.survey_project(tmp_path_factory.mktemp("previews_leash_survey")))
    yield planned
    planned.project.close()


TWO_FAMILIES = ["linear", "elastic_net"]
RECORD = "Record it as it is: what has no design-based estimator is blocked and recorded"
POPULATION = d.SetSurvey(estimand="population", weight="WTDRD1", strata="SDMVSTRA",
                         psu="SDMVPSU")


def blocked_family(out: dict[str, Any], key: str) -> dict[str, Any]:
    return next(m for m in out["fit"].data["models"] if m["family"] == key)["inference"]


def test_4_a_family_without_a_design_based_estimator_previews_its_block(svy):
    """Under the surveyed population the elastic net has no design-based estimator: recorded, the
    fit blocks its coefficients with the reason and the exits (the survey-weighted family in its
    place; the sample-only answer). The families' preview and the survey answer's preview say the
    same, with recording the answer as it is; the survey answer still draws the design the linear
    family is estimated over, its degrees of freedom the fit's."""
    models = d.SelectModels(models=TWO_FAMILIES)
    result, _ = svy.preview("families", models)
    _, out = svy.after("families", models, ["fit"])
    info = blocked_family(out, "elastic_net")
    assert info["refused"].startswith("Elastic net has no design-based estimator")
    assert blocked_family(out, "linear")["refused"] is None
    fit_exits = [e["decision"] for e in info["exits"]]
    assert result.caution is not None and result.caution.text == info["refused"]
    assert [x.decision for x in result.caution.exits] == [*fit_exits, models.model_dump(mode="json")]
    assert result.caution.exits[-1].label == RECORD
    vocabulary(result)

    unanswered = svy.project.state.model_copy(update={"survey": None, "models": TWO_FAMILIES})
    survey, ctx = svy.preview("population", POPULATION, state=unanswered)
    _, out = svy.after("population", POPULATION, ["fit"], state=unanswered)
    info = blocked_family(out, "elastic_net")
    assert survey.caution is not None and survey.caution.text == info["refused"]
    assert [x.decision for x in survey.caution.exits] == [
        *(e["decision"] for e in info["exits"]), POPULATION.model_dump(mode="json")]
    assert ctx.read["survey"]["df"] == blocked_family(out, "linear")["survey"]["df"] == 15


@pytest.fixture(scope="module")
def binary_svy(tmp_path_factory):
    """The NCI package's NHANES-shaped table under its survey design with a yes/no outcome from its
    own total cholesterol (200 mg/dL or more), the exposure's effect a conditional odds ratio."""
    from turbotab.core.tests.acceptance import test_nci_usual_intake as T

    frame = T.nhanes_table(seed=21, n=900)
    frame["high_tc"] = np.where(frame["LBXTC"] >= 200, "yes", "no")
    roles = {**T.NHANES_ROLES, "LBXTC": "excluded"}
    state = d.ProjectState(
        lens=["dietary"], target="high_tc", task="binary", event="yes", purpose="inference",
        roles=roles, role_confirmations=dict(roles),
        grain=d.GrainSpec(grain="one_row_per_unit", id_column="SEQN"),
        shape_confirmations=dict(T.NHANES_TRUTH), exclusions=[],
        survey=d.SurveySpec(**POPULATION.model_dump(exclude={"kind"})),
        estimand=d.EstimandSpec(exposure="DR1TPROT", measure="odds_ratio"),
        adjustment={"RIDAGEYR": A(exposure="DR1TPROT", **CONFOUNDER)},
        missing=d.MissingSpec(strategy="complete_case"),
        split=d.SplitSpec(holdout=0.0, seed=0, folds=5), models=["linear"])
    planned = Planned(Project(frame, tmp_path_factory.mktemp("previews_leash_binary"), state,
                              upto=["design", "cohort", "split"]))
    yield planned
    planned.project.close()


def test_4_a_marginal_measure_previews_the_block_the_effects_stage_records(binary_svy):
    """A marginal risk difference is standardized over these participants only: recorded under the
    surveyed population, the effects stage blocks it with its reason and exits (the sample-only
    answer; the conditional odds ratio), and the estimand's preview says the same."""
    decision = d.SetEstimand(exposure="DR1TPROT", measure="risk_difference")
    result, _ = binary_svy.preview("marginal", decision)
    _, out = binary_svy.after("marginal", decision, ["effects"])
    marginal = out["effects"].data["families"][0]["marginal"]
    assert marginal["refused"].startswith("The marginal risks are standardized over these")
    assert result.caution is not None and result.caution.text == marginal["refused"]
    assert [x.decision for x in result.caution.exits] == [
        *(e["decision"] for e in marginal["exits"]), decision.model_dump(mode="json")]
    assert result.caution.exits[-1].label == RECORD
    vocabulary(result)
    plain, _ = binary_svy.preview("conditional", d.SetEstimand(exposure="DR1TPROT",
                                                               measure="odds_ratio"))
    assert plain.caution is None


DESIGN = {"stratum": "design", "psu": "design", "wt": "design"}
SURVEYED = d.SurveySpec(estimand="population", weight="wt", strata="stratum", psu="psu")


def with_design(frame: pd.DataFrame, unit: str | None = None, seed: int = 71) -> pd.DataFrame:
    """``frame`` with 8 strata of 2 PSUs and a weight from their own stream, drawn per ``unit``
    (a person's rows in one PSU, as the design needs) or per row."""
    keys = frame[unit] if unit else pd.Series(np.arange(len(frame)), index=frame.index)
    levels = pd.Index(keys.unique())
    rng = np.random.default_rng(seed)
    cell = pd.Series(rng.integers(0, 16, len(levels)), index=levels)
    weight = pd.Series(np.round(rng.uniform(1000, 9000, len(levels)), 1), index=levels)
    out = frame.copy()
    out["stratum"] = 100 + keys.map(cell).to_numpy() // 2
    out["psu"] = 1 + keys.map(cell).to_numpy() % 2
    out["wt"] = keys.map(weight).to_numpy()
    return out


@pytest.fixture(scope="module")
def surveyed_scale(tmp_path_factory):
    """MS8's reflective scale (``preview_fixtures.scales_project``) under a survey design, its
    estimates declared for the surveyed population."""
    from turbotab.core.tests.acceptance import scales_fixtures as sf

    roles = {"pid": "identifier", "age": "covariate", "bmi": "covariate",
             **{c: "covariate" for c in sf.SAT}, "sat_ref": "excluded", **DESIGN}
    state = d.ProjectState(
        lens=["survey"], target="sbp", task="regression", purpose="inference",
        roles=roles, role_confirmations=dict(roles),
        grain=d.GrainSpec(grain="one_row_per_unit", id_column="pid"),
        missing=d.MissingSpec(strategy="complete_case"), exclusions=[],
        split=d.SplitSpec(holdout=0.0, seed=0, folds=5), models=["linear"], survey=SURVEYED,
        shape_confirmations={**{f"code_or_count:{c}": "amount" for c in sf.SAT},
                             "code_or_count:age": "amount"})
    planned = Planned(Project(with_design(sf.linear_scale_table(seed=11, n=600)),
                              tmp_path_factory.mktemp("previews_leash_scale"), state,
                              upto=["design", "cohort", "split"]))
    yield planned
    planned.project.close()


def test_4_a_scales_correction_under_the_population_previews_the_stages_block(surveyed_scale):
    """The correction has no design-based estimator: recorded, the scales stage leaves the score
    uncorrected with its reason and the sample-only exit, and the preview says the same beside the
    score it still draws (no calibrated column)."""
    from turbotab.core.tests.acceptance.test_previews_kinds import _scale

    decision = _scale()
    result, _ = surveyed_scale.preview("scale", decision)
    _, out = surveyed_scale.after("scale", decision, ["scales"])
    [scale] = out["scales"].data["scales"]
    assert scale["correction"] is None and scale["not_corrected"].startswith(
        "The correction and the uncorrected coefficient beside it have no design-based")
    assert result.caution is not None and result.caution.text == scale["not_corrected"]
    assert [x.decision for x in result.caution.exits] == [
        *(e["decision"] for e in scale["exits"]), decision.model_dump(mode="json")]
    table = view(result, "table_focus")
    assert "sat_score (calibrated)" not in table.columns_after
    vocabulary(result)


@pytest.fixture(scope="module")
def surveyed_lane(tmp_path_factory):
    """TIMEVARY's feedback cohort under a survey design drawn per person, its estimates declared
    for the surveyed population."""
    from turbotab.core.tests.acceptance import test_timevary as T
    from turbotab.core.tests.acceptance.timevary_fixtures import feedback_cohort

    base = T.feedback_state()
    roles = {**base.roles, **DESIGN}
    state = T.feedback_state(time_varying=T.MSM_LANE, models=["linear"], roles=roles,
                             survey=SURVEYED)
    planned = Planned(Project(with_design(feedback_cohort(n=800, visits=5), unit="pid"),
                              tmp_path_factory.mktemp("previews_leash_lane"), state,
                              upto=["time_varying", "cohort", "split"]))
    yield planned
    planned.project.close()


def test_4_a_time_varying_lane_under_the_population_previews_the_stages_block(surveyed_lane):
    """The g-methods have no design-based estimator: the declared lane, recorded, is blocked with
    the stage's reason and the sample-only exit; the preview draws the weights' diagnostics (they
    describe these rows) and says the same."""
    from turbotab.core.tests.acceptance import test_timevary as T

    key = surveyed_lane.project.artifacts["time_varying"].data["diagnosed"]["key"]
    decision = d.SetTimeVarying(**{**T.MSM_LANE.model_dump(), "truncation": "p1_p99",
                                   "diagnostics_seen": key})
    result, _ = surveyed_lane.preview("lane", decision)
    _, out = surveyed_lane.after("lane", decision, ["time_varying"])
    lane = out["time_varying"].data
    assert lane["reason"].startswith("The g-methods here have no design-based estimator")
    assert result.caution is not None and result.caution.text == lane["reason"]
    assert [x.decision for x in result.caution.exits] == [
        *(e["decision"] for e in lane["exits"]), decision.model_dump(mode="json")]
    assert [v.kind for v in result.views] == ["distribution"]

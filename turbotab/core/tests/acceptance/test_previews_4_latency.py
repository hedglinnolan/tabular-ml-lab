"""PREVIEWS (4) · every new preview answers in under a second at the 95th percentile on the NHANES
fixture (V2 definition of done §3, gate 5: "Previews take < 1 s at the 95th percentile").

The fixture is the real NHANES export (21,849 participants), with the columns its previews need
added from their own seeded stream where the export has none (``nhanes_previews``); the
repeated-measures previews read its participants seen up to three times, until an event, or
recalled twice.
Each preview is planned as the server plans it (a fresh context each time, as each hover is) twenty
times after one warm-up; the 95th percentile of those twenty must be under one second. A join and a
codebook import also count the server's own work before the preview: the completion that reads the
file or the codebook against every row (``decisions.validate``).

Where a preview reads a sample, the basis states the rule; these assertions hold the statement.
Skipped where the export is not on the machine (``stage_harness.NHANES``).
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import pytest

from turbotab.core import consequences, decisions as d
from turbotab.core.tests.acceptance import nhanes_previews as N
from turbotab.core.tests.acceptance.preview_harness import Project, p95

pytestmark = pytest.mark.skipif(N.source() is None, reason="the NHANES export is not on this machine")

REPEATS = 20
BUDGET = 1.0
ALL = ["no_unmeasured_confounding", "positivity", "consistency", "time_ordering"]
C = d.CovariateAnswers


def hover(project: Project, decision, validate: dict | None = None) -> list[float]:
    """Seconds per preview over :data:`REPEATS` plans after one warm-up. ``validate``: the server's
    completion is part of the request (a join's counts, a codebook's assessment)."""
    def once() -> float:
        ctx = project.context()
        started = time.perf_counter()
        planned = d.validate(decision, validate) if validate is not None else decision
        result = consequences.plan(planned, ctx, basis="")
        took = time.perf_counter() - started
        assert result.views, (decision.kind, result.note)
        return took

    once()
    return [once() for _ in range(REPEATS)]


@pytest.fixture(scope="module")
def export():
    return N.augment(pd.read_csv(N.source()))


@pytest.fixture(scope="module")
def inference(export, tmp_path_factory):
    roles = {**N.ROLES, **{c: "covariate" for c in N.SCALE}}
    project = Project(export, tmp_path_factory.mktemp("latency_inference"),
                      N.inference_state(roles=roles, role_confirmations=dict(roles)),
                      upto=["design", "cohort", "split", "target_info", "proposals",
                            "usual_intake"])
    yield project
    project.close()


@pytest.fixture(scope="module")
def prediction(export, tmp_path_factory):
    project = Project(export, tmp_path_factory.mktemp("latency_prediction"),
                      N.inference_state(purpose="prediction", estimand=None, adjustment=None,
                                        survey=None, split=d.SplitSpec(holdout=0.2, seed=0, folds=5)),
                      upto=["cohort", "split", "findings"])
    yield project
    project.close()


@pytest.fixture(scope="module")
def causal(export, tmp_path_factory):
    answers = {c: d.AdjustmentAnswer(exposure="supplement", causes_exposure="yes",
                                     causes_outcome="yes", after_exposure="no")
               for c in ("age", "gender", "bmi", "protein", "kcal")}
    project = Project(export, tmp_path_factory.mktemp("latency_causal"), N.inference_state(
        estimand=d.EstimandSpec(exposure="supplement", measure="mean_difference"),
        adjustment=answers), upto=["cohort", "split"])
    yield project
    project.close()


@pytest.fixture(scope="module")
def follow(export, tmp_path_factory):
    roles = {**N.ROLES, "followup_years": "time"}
    project = Project(export, tmp_path_factory.mktemp("latency_follow"), N.inference_state(
        target="cvd_event", task="time_to_event", event="1", roles=roles,
        role_confirmations=dict(roles), models=["cox"],
        follow_up=d.FollowUpSpec(time_column="followup_years")), upto=["cohort", "split"])
    yield project
    project.close()


@pytest.fixture(scope="module")
def visits(export, tmp_path_factory):
    raw = pd.read_csv(N.source())
    roles = {"SEQN": "identifier", "visit": "time", "supp": "exposure", "sbp": "covariate",
             "female": "covariate", "age": "covariate"}
    conf = dict(causes_exposure="yes", causes_outcome="yes")
    lane = d.TimeVaryingSpec(exposure="supp", method="msm_iptw", ordering="exposure_precedes_outcome",
                             confounders=["sbp"], baseline=["female", "age"])
    state = d.ProjectState(
        lens=["clinical"], target="event", task="binary", event="1", purpose="inference",
        grain=d.GrainSpec(grain="repeated", id_column="SEQN"),
        repeat_kind=d.RepeatSpec(repeat_kind="time_points", time_column="visit"), unit="row",
        roles=roles, role_confirmations=dict(roles), censoring="same",
        missing=d.MissingSpec(strategy="complete_case"), exclusions=[], models=["linear"],
        split=d.SplitSpec(holdout=0.0, seed=0, folds=5), temporal=d.TemporalSpec(temporal=False),
        estimand=d.EstimandSpec(exposure="supp", measure="odds_ratio"),
        adjustment={"sbp": d.AdjustmentAnswer(exposure="supp", after_exposure="yes", **conf),
                    "female": d.AdjustmentAnswer(exposure="supp", after_exposure="no", **conf),
                    "age": d.AdjustmentAnswer(exposure="supp", after_exposure="no", **conf)},
        shape_confirmations={"code_or_count:age": "amount", "code_or_count:female": "code",
                             "code_or_count:visit": "amount"}, time_varying=lane)
    project = Project(N.visits(raw), tmp_path_factory.mktemp("latency_visits"), state,
                      upto=["time_varying", "cohort", "split"])
    yield project
    project.close()


@pytest.fixture(scope="module")
def recalls(export, tmp_path_factory):
    raw = pd.read_csv(N.source())
    roles = {"SEQN": "identifier", "age": "covariate", "gender": "covariate", "kcal": "energy",
             "protein": "exposure", "recall": "time"}
    state = d.ProjectState(
        lens=["dietary"], target="glucose", task="regression", purpose="inference",
        roles=roles, role_confirmations=dict(roles),
        grain=d.GrainSpec(grain="repeated", id_column="SEQN"),
        repeat_kind=d.RepeatSpec(repeat_kind="repeats"), unit="unit",
        aggregation=d.AggregationSpec(method="mean"), temporal=d.TemporalSpec(temporal=False),
        shape_confirmations={"code_or_count:age": "amount", "code_or_count:kcal": "amount",
                             "code_or_count:glucose": "amount"},
        column_units={"kcal": d.ColumnUnitSpec(unit="kcal", days=1)}, exclusions=[],
        missing=d.MissingSpec(strategy="complete_case"),
        split=d.SplitSpec(holdout=0.0, seed=0, folds=5), models=["linear"],
        energy_adjustment=d.EnergyAdjustment(method="residual", energy_column="kcal",
                                             nutrients=["protein"]))
    project = Project(N.recalls(raw), tmp_path_factory.mktemp("latency_recalls"), state,
                      upto=["design", "cohort", "split"])
    yield project
    project.close()


@pytest.fixture(scope="module")
def datain(export, tmp_path_factory):
    """The export as the table, a labs file added on ``SEQN`` (its first 15,000 participants and
    2,000 the table does not hold), and a variable table documenting its columns."""
    from turbotab.core import codebook as cb
    from turbotab.core.datastore import ingest

    folder = tmp_path_factory.mktemp("latency_datain")
    project = Project(export, folder, d.ProjectState(lens=["dietary"]), upto=["working", "findings"])
    rng = np.random.default_rng(3)
    seqn = np.concatenate([export["SEQN"].to_numpy()[:15_000], 10_000_000 + np.arange(2_000)])
    labs = pd.DataFrame({"SEQN": seqn, "LBXTC": np.round(rng.normal(190, 35, len(seqn)))})
    where = project.run.folder / "files" / "f0000000000"
    where.mkdir(parents=True)
    labs.to_csv(folder / "labs.csv", index=False)
    info = ingest(folder / "labs.csv", where / "raw.parquet")
    (where / "file.json").write_text(json.dumps({"id": "f0000000000", "name": "labs.csv",
                                                 "n_rows": info.n_rows}))
    table = pd.DataFrame([
        {"variable": "SEQN", "label": "Respondent sequence number", "unit": "", "type": "identifier",
         "codes": ""},
        {"variable": "gender", "label": "Gender", "unit": "", "type": "categorical",
         "codes": "male=Male; female=Female"},
        {"variable": "age", "label": "Age in years", "unit": "years", "type": "continuous",
         "codes": ""},
        {"variable": "kcal", "label": "Energy (kcal)", "unit": "kcal", "type": "continuous",
         "codes": ""},
        {"variable": "protein", "label": "Protein (gm)", "unit": "g", "type": "continuous",
         "codes": ""},
        {"variable": "weight", "label": "Weight (kg)", "unit": "kg", "type": "continuous",
         "codes": ""}])
    table.to_csv(folder / "dictionary.csv", index=False)
    book = cb.read(folder / "dictionary.csv")
    cb.stage(book, project.run.folder, folder / "dictionary.csv")
    project.project_dir = project.run.folder
    project.codebook = book.id
    yield project
    project.close()


def _assert_fast(times: list[float], kind: str) -> None:
    print(f"\nMEASURE preview {kind} p95={p95(times):.3f}s max={max(times):.3f}s n={len(times)}")
    assert p95(times) < BUDGET, f"{kind}: p95 {p95(times):.3f}s over {len(times)} plans"


INFERENCE = [
    d.SetEstimand(exposure="carb", measure="mean_difference"),
    d.SetAdjustment(exposure="protein", answers={
        "bmi": C(causes_exposure="no", causes_outcome="yes", after_exposure="yes")}),
    d.SetModelSequence(exposure="protein", model_1=["age", "gender", "kcal"]),
    d.SetExposureForm(column="protein", form="spline", knots=4),
    d.SetClusters(column="site", adjust="fixed_effects"),
    d.SetSurvey(estimand="population", weight="WTDRD1", strata="SDMVSTRA", psu="SDMVPSU"),
    d.SetOutcomeScale(column="glucose", scale="log"),
    d.SetColumnUnit(column="kcal", unit="kj"),
    d.SetScales(scales=[d.ScaleSpec(name="sat_score", items=N.SCALE, low=1, high=5,
                                    kind="reflective", correction="regression_calibration")]),
    d.SetUsualIntake(nutrient="protein", model="amount_only", days=["protein", "protein_d2"]),
]


@pytest.mark.parametrize("decision", INFERENCE, ids=lambda x: x.kind)
def test_4_inference_previews_answer_within_a_second(inference, decision):
    _assert_fast(hover(inference, decision), decision.kind)


def test_4_the_samples_are_stated_on_the_preview(inference):
    """The 21,849 rows are read as a sample of 5,000 where a model's steps are fitted, and the
    basis says so; the usual-intake model is fitted on a stated random sample of participants."""
    from turbotab.server.service import preview_basis

    for decision, said in (
            (INFERENCE[3], "Values on a sample of 5,000 of the 21,849 analyzed rows"),
            (INFERENCE[-1], "The usual-intake model fitted on a random sample of 5,000 of the "
                            "21,849 eligible participants (seed 0); the analysis fits every one.")):
        ctx = inference.context()
        result = consequences.plan(decision, ctx, basis="")
        assert preview_basis(ctx, result, 21_849).startswith(said)


def test_4_multiplicity_answers_within_a_second(inference):
    family = inference.state.model_copy(update={
        "estimand": d.EstimandSpec(family=True, measure="mean_difference", multiplicity="fdr_bh"),
        "adjustment": {c: d.AdjustmentAnswer(exposure=d.EXPOSURE_FAMILY, causes_exposure="yes",
                                             causes_outcome="yes", after_exposure="no")
                       for c in ("age", "gender", "bmi")}})
    times = []
    for _ in range(REPEATS + 1):
        ctx = inference.context(state=family)
        started = time.perf_counter()
        result = consequences.plan(d.SetMultiplicity(method="bh"), ctx, basis="")
        times.append(time.perf_counter() - started)
        assert result.views
    _assert_fast(times[1:], "set_multiplicity")


def test_4_batch_answers_within_a_second(prediction):
    _assert_fast(hover(prediction, d.SetBatch(column="batch", method="reference_combat")),
                 "set_batch")


@pytest.mark.parametrize("decision", [
    d.SetCausal(exposure="supplement", method="tmle", learner="linear", trim=0.05,
                assumptions=ALL),
    d.SetCausal(exposure="supplement", method="dml_irm", learner="random_forest", trim=0.05,
                assumptions=ALL),
    d.SetCausal(exposure="supplement", method="dml_plr", learner="boosted_trees", assumptions=ALL),
], ids=lambda x: x.learner)
def test_4_causal_answers_within_a_second(causal, decision):
    _assert_fast(hover(causal, decision), "set_causal")


def test_4_follow_up_answers_within_a_second(follow):
    _assert_fast(hover(follow, d.SetFollowUp(column="cvd_event", time_column="followup_years",
                                             landmark=2.0, horizon=10.0)), "set_follow_up")


def test_4_time_varying_answers_within_a_second(visits):
    key = visits.artifacts["time_varying"].data["diagnosed"]["key"]
    for decision in (
            d.SetTimeVarying(exposure="supp", method="msm_iptw", ordering="exposure_precedes_outcome",
                             confounders=["sbp"], baseline=["female", "age"], truncation="p1_p99",
                             diagnostics_seen=key),
            d.SetTimeVarying(exposure="supp", method="gformula", ordering="exposure_precedes_outcome",
                             confounders=["sbp"], baseline=["female", "age"])):
        _assert_fast(hover(visits, decision), "set_time_varying")


def test_4_measurement_error_answers_within_a_second(recalls):
    _assert_fast(hover(recalls, d.SetMeasurementError(method="regression_calibration")),
                 "set_measurement_error")


def test_4_a_join_and_a_codebook_answer_within_a_second_with_the_servers_completion(datain):
    ctx = {"project_dir": str(datain.project_dir), "state": datain.state, "ingest_status": "fresh"}
    join = d.JoinFiles(file="f0000000000", on="SEQN")
    _assert_fast(hover(datain, join, validate=ctx), "join_files")
    _assert_fast(hover(datain, d.ImportCodebook(codebook=datain.codebook), validate=ctx),
                 "import_codebook")

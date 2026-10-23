"""The pilot's launcher (PILOT_READINESS §5; ``research/swarm/launcher.py``): the known answers read
from each dataset's own answers file and checked against the generator that planted them, the crash
sweep's report gating a launch both ways, the fit budget from its measured fits by hand, and the
start command line the launcher hands the driver: no label of its own, the run named by the
driver, the known answer written where no analyst's command reads it."""
from __future__ import annotations

import json
import math

import pytest

from research.swarm import driver, launcher as L

ENGINE = {"commit": "c0", "engine_tree": "t0", "engine_modified": False}


def report(**changes):
    out = {"dataset": "outpatient_admission_predict", "complete": True, "unvisited": [],
           "defects": [], "passed": True, "engine_tree": "t0", "engine_modified": False,
           "commit": "c0", "fit_seconds": {"fits": 4, "total": 20.0, "median": 4.0, "p90": 7.2,
                                           "max": 7.2}}
    out.update(changes)
    return out


# ── the known answers ────────────────────────────────────────────────────────


def test_the_estimate_fixtures_known_answer_is_its_planted_effect_on_the_engines_measure():
    from research.swarm.generators import ESTIMATE_MODEL
    from turbotab.core.estimand import MEASURE_OF_TASK

    known, source = L.known_answer("walking_bp_estimate")
    answers = json.loads((L.DATASETS / "walking_bp_estimate" / "answers.json").read_text())
    stated = next(t for t in answers["truths"]
                  if t["name"] == "the effect of walks_daily on sbp_mmhg, a difference in means")
    assert known == {"feature": "walks_daily", "measure": MEASURE_OF_TASK["regression"],
                     "value": stated["value"]}
    assert known["value"] == ESTIMATE_MODEL["effect"]  # the generator's own constant
    assert source.startswith("answers.json")


@pytest.mark.parametrize("dataset", ["outpatient_admission_predict", "survey_support_estimate"])
def test_a_null_decoy_and_a_null_effect_are_written_null(dataset):
    """The decoy's outcome reads no column (its answers file says so); the survey fixture states
    the effect of age as none."""
    known, _ = L.known_answer(dataset)
    assert known == {"null": True}


@pytest.mark.parametrize("dataset", ["clinical_readmission_predict", "bang2009", "sullivan2021",
                                     "nhanes_glucose_predict", "plasma_panel_estimate"])
def test_a_dataset_with_no_answer_of_that_form_has_none_and_says_why(dataset):
    known, why = L.known_answer(dataset)
    assert known is None and why


def test_the_checks_read_the_known_answer_as_the_launcher_writes_it():
    """The record's known-answer check compares a result on the known answer's measure with its
    value, and a null one with the scale's no-effect value (``checks._known_value``)."""
    from types import SimpleNamespace

    from research.swarm import checks

    walking, _ = L.known_answer("walking_bp_estimate")
    result = SimpleNamespace(measure="mean_difference", scale="difference")
    assert checks._known_value(result, walking) == (walking["value"], None)
    null, _ = L.known_answer("outpatient_admission_predict")
    assert checks._known_value(result, null)[0] == 0.0
    other = SimpleNamespace(measure="odds_ratio", scale="ratio")
    value, why = checks._known_value(other, walking)
    assert value is None and why


def test_the_records_dataset_holds_the_control_kind_from_the_answers_file():
    meta = L.dataset_meta("outpatient_admission_predict")
    assert meta["control"] == "null_decoy" and meta["known"] == {"null": True}
    assert L.dataset_meta("walking_bp_estimate")["control"] == "known_answer"
    assert L.dataset_meta("bang2009")["control"] is None


# ── the crash sweep gates the launch ─────────────────────────────────────────


def test_a_crash_sweep_that_passed_on_this_engine_lets_the_dataset_launch():
    assert L.crash_passed(report(), "outpatient_admission_predict", ENGINE) is None


@pytest.mark.parametrize("changes, engine, words", [
    ({"dataset": "bang2009"}, ENGINE, "not of"),
    ({"complete": False, "unvisited": ["question:missing=impute"]}, ENGINE, "did not visit"),
    ({"defects": [{"refit": "x"}], "passed": False}, ENGINE, "engine defect"),
    ({"passed": False}, ENGINE, "did not pass"),
    ({"engine_modified": True}, ENGINE, "when the crash sweep ran"),
    ({}, {**ENGINE, "engine_modified": True}, "now"),
    ({"engine_tree": "t1"}, ENGINE, "runs again on the frozen engine"),
], ids=["other_dataset", "incomplete", "defect", "not_passed", "modified_then", "modified_now",
        "other_engine"])
def test_a_crash_sweep_that_has_not_passed_on_this_engine_refuses_the_launch(changes, engine,
                                                                               words):
    why = L.crash_passed(report(**changes), "outpatient_admission_predict", engine)
    assert why is not None and words in why


def test_the_budget_is_its_fits_at_the_slowest_fit_measured_rounded_up():
    budget = L.budget_of(report(), 12)
    assert budget == {"fits": 12, "seconds": float(math.ceil(12 * 7.2)), "slowest_fit": 7.2}
    assert budget["seconds"] == 87.0  # 86.4 rounded up
    with pytest.raises(L.Refused, match="no fit's run time"):
        L.budget_of(report(fit_seconds={"fits": 0, "max": None}), 12)
    with pytest.raises(L.Refused, match="at least one fit"):
        L.budget_of(report(), 0)


# ── the start the launcher hands the driver ──────────────────────────────────


def test_a_launch_starts_the_run_the_driver_names_and_keeps_its_answer_apart(tmp_path,
                                                                             monkeypatch):
    crash = tmp_path / "crash.json"
    crash.write_text(json.dumps(report()))
    monkeypatch.setattr(L, "engine_now", lambda: dict(ENGINE))
    data = tmp_path / "outpatient_visits.csv"
    data.write_text("x,y\n1,0\n")
    runs, launches = tmp_path / "runs", tmp_path / "launches"
    given: list[list[str]] = []

    def start(argv):
        given.append(list(argv))
        driver.build_parser().parse_args(argv)  # a start command line the driver reads
        return 0, {"started": {"run": driver.new_run_id(runs)}}

    out = L.launch("outpatient_admission_predict", crash, budget_fits=10,
                   model="claude-haiku-5-5", persona="skeptical", prompt_version="pilot-1",
                   effort="high", runs=runs, launches=launches, data=data, start=start)
    (argv,) = given
    assert argv == ["start", "--data", str(data), "--budget-fits", "10", "--budget-seconds",
                    "72", "--model", "claude-haiku-5-5", "--prompt-version", "pilot-1",
                    "--persona", "skeptical", "--effort", "high"]
    label, secret = out["run"].rsplit("-", 1)
    assert driver.RUN_ID.fullmatch(out["run"]) and label == "run_01" and len(secret) == 32
    assert all(word not in out["run"] for word in ("decoy", "null", "outpatient"))
    kept = json.loads((launches / f"{out['run']}.json").read_text())
    assert kept["dataset"]["known"] == {"null": True}
    assert kept["dataset"]["control"] == "null_decoy"
    assert kept["budget"] == {"fits": 10, "seconds": 72.0, "slowest_fit": 7.2}
    assert not (runs / out["run"]).exists()  # nothing written into the run's own folder
    meta = L.record_meta(kept)
    assert meta["dataset"]["known"] == {"null": True} and "known_source" not in meta["dataset"]
    assert meta["persona"] == "skeptical" and meta["design"] == 2
    # Under the corpus, where no analyst's command opens a file (the driver's seal).
    assert driver.unread(L.LAUNCHES / "run_01-x.json")


def test_a_dataset_whose_crash_sweep_did_not_pass_is_not_started(tmp_path, monkeypatch):
    crash = tmp_path / "crash.json"
    crash.write_text(json.dumps(report(passed=False, defects=[{"refit": "x"}])))
    monkeypatch.setattr(L, "engine_now", lambda: dict(ENGINE))

    def start(argv):  # pragma: no cover - must not be reached
        raise AssertionError("started")

    with pytest.raises(L.Refused, match="is not launched"):
        L.launch("outpatient_admission_predict", crash, budget_fits=10, model="m",
                 persona=None, prompt_version="p", runs=tmp_path, launches=tmp_path,
                 data=tmp_path / "x.csv", start=start)

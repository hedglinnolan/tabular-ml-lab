"""The markers the engine reads (SWARM_PLAN §4; ``markers.py``), on a project view and routes built
here: every number is constructed (480 rows, 112 events among them, 9 candidate parameters, 384
rows analyzed after an exclusion, 12 blanks in one column), so each marker's value is known before
it is read; and each guard both ways: a stage that is not fresh at its key serves no marker, and
no estimate stage is ever read."""
from __future__ import annotations

import math
from typing import Any

import pytest

from research.swarm import markers as M
from research.swarm import vocabulary as V
from research.swarm.record import Marker

STATE = {
    "purpose": "prediction", "target": "y", "task": None, "event": "1", "lens": ["clinical"],
    "roles": {"id": "identifier", "y": "covariate", "age": "covariate", "x": "exposure",
              "z": "exposure", "plate_id": "excluded"},
    "missing": {"strategy": "complete_case"}, "grain": {"grain": "one_row_per_unit"},
    "repeat_kind": None, "aggregation": None, "temporal": None, "survey": None, "clusters": None,
    "unit": None, "design": None, "batch": None, "intended_use": {
        "use": "decision_support", "threshold_low": 0.05, "threshold_high": 0.3,
        "threshold": 0.1, "subgroups": ["sex"], "fairness": "subgroup_performance"},
    "estimand": None, "multiplicity": None, "column_units": {"age": {"unit": "years"}},
}
ARTIFACTS = {
    "ingest": {"n_rows": 480, "columns": [
        {"name": "id", "n_missing": 0}, {"name": "y", "n_missing": 0},
        {"name": "age", "n_missing": 12}, {"name": "x", "n_missing": 0},
        {"name": "plate_id", "n_missing": 0}, {"name": "injection_order", "n_missing": 0}]},
    "target_info": {"column": "y", "task": "binary",
                    "classes": [{"value": 0, "count": 368}, {"value": 1, "count": 112}]},
    "cohort": {"n_final": 384, "predictors": ["age", "x", "z"],
               "complete_case_loss": {"n_before": 480, "n_kept": 384, "n_dropped": 96}},
    "structure": {"time_column": None},
    "seal_plan": {"basis": {"state": "one_row_per_unit", "column": None, "n_units": None},
                  "time_ordered_folds": False},
    "shelf": {"sample_size": {"parameters": 9}},
    "findings": {"findings": [{"id": "below_detection__x", "title": "x below"}]},
}
READINGS = {"read_from_data": [
    {"kind": "unit", "column": "x", "value": "g", "evidence": "grams by the Atwater sum"},
    {"kind": "role", "column": "age", "value": "covariate", "evidence": "a characteristic"}]}
FITS = {"records": [{"primary_metric": "log_loss"}, {"primary_metric": "auc"}]}


def reader(artifacts: dict[str, Any] | None = None, stale: tuple[str, ...] = (),
           asked: list[str] | None = None):
    arts = dict(ARTIFACTS if artifacts is None else artifacts)

    def read(path: str) -> Any:
        if asked is not None:
            asked.append(path)
        if path == "readings":
            return READINGS
        if path == "fits":
            return FITS
        stage = path.split("/", 1)[1]
        if stage not in arts:
            return None
        return {"fresh": stage not in stale, "artifact": arts[stage]}

    return read


def by_name(markers: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {m["name"]: m for m in markers}


def view(**state: Any) -> dict[str, Any]:
    return {"state": {**STATE, **state}, "stages": {"fit": {"key": "k9"}}}


def test_every_live_marker_is_read_once_and_validates_as_the_record_holds_it():
    got = M.live(view(), reader())
    assert [m["name"] for m in got] == list(V.LIVE)
    for m in got:
        Marker.model_validate(m)
    assert M.fit_key(view()) == "k9"


def test_each_marker_is_its_constructed_number():
    m = {k: v["value"] for k, v in by_name(M.live(view(), reader())).items()}
    assert m["rows"] == 480 and m["rows_analyzed"] == 384
    assert m["effective_size"] == 112  # the smaller class: 112 events, 368 non-events
    assert m["events_per_parameter"] == pytest.approx(112 / 9)
    assert m["predictors_by_role"] == {"covariate": 1, "exposure": 2, "excluded": 1,
                                       "identifier": 1}  # the target is not a predictor
    assert m["predictors_per_row"] == pytest.approx(3 / 384)
    assert m["outcome_type"] == "binary" and m["event_share"] == pytest.approx(112 / 480)
    assert m["missingness"] == {"age": pytest.approx(12 / 480)}
    assert m["complete_case_share"] == pytest.approx(384 / 480)
    assert m["detection_limit_share"] == {"columns": ["x"], "shares": None}
    assert m["batch_columns"] == {"read": ["plate_id"], "answered": None}
    assert m["run_order"] == ["injection_order"]
    assert m["units"] == [{"column": "x", "kind": "unit", "value": "g",
                           "basis": "grams by the Atwater sum"},
                          {"column": "age", "kind": "unit", "value": "years", "basis": "recorded"}]
    assert m["primary_score"] == {"metric": "auc"}  # the newest fit record's
    assert m["thresholds"] == {"threshold_low": 0.05, "threshold_high": 0.3, "threshold": 0.1}
    assert m["purpose"] == "prediction" and m["imputed_copies"] is False


@pytest.mark.parametrize("stage,names", [
    ("cohort", ["rows_analyzed", "predictors_per_row", "complete_case_share"]),
    ("ingest", ["rows", "missingness", "batch_columns", "run_order"]),
    ("shelf", ["events_per_parameter"]),
    ("target_info", ["effective_size", "event_share", "outcome_type"]),
    ("findings", ["detection_limit_share"]),
])
def test_a_stage_not_fresh_at_its_key_serves_no_marker_and_says_why(stage, names):
    fresh = by_name(M.live(view(), reader()))
    stale = by_name(M.live(view(), reader(stale=(stage,))))
    for name in names:
        assert fresh[name]["value"] is not None, name
        assert stale[name]["value"] is None and stale[name]["why"], name


def test_markers_the_answers_do_not_give_say_why():
    m = by_name(M.live(view(purpose=None, intended_use=None, roles=None,
                            missing={"strategy": "impute"}), reader()))
    for name in ("purpose", "intended_use", "thresholds", "estimand", "multiplicity_family",
                 "predictors_by_role", "complete_case_share", "omics_scale"):
        assert m[name]["value"] is None and m[name]["why"], name


def test_a_numeric_outcomes_effective_size_is_its_rows_analyzed_and_it_has_no_event_share():
    m = by_name(M.live(view(task="regression"), reader()))
    assert m["effective_size"]["value"] == 384 and m["event_share"]["value"] is None


def test_under_estimate_the_primary_score_is_none_and_the_estimand_is_read():
    m = by_name(M.live(view(purpose="inference", estimand={"exposure": "x", "family": False,
                                                           "measure": "odds_ratio"},
                            multiplicity={"method": "bh"}), reader()))
    assert m["primary_score"]["value"] is None
    assert m["estimand"]["value"]["exposure"] == "x"
    assert m["multiplicity_family"]["value"] == {"family": False, "method": "bh"}


def test_no_estimate_stage_is_ever_read():
    from turbotab.core.plan_lock import ESTIMATE_STAGES

    asked: list[str] = []
    M.live(view(), reader(asked=asked))
    stages = {p.split("/", 1)[1] for p in asked if p.startswith("stages/")}
    assert stages and not stages & set(ESTIMATE_STAGES)


def test_a_marker_whose_reading_fails_says_so_and_the_rest_are_read():
    def broken(path: str) -> Any:
        if path == "readings":
            return {"read_from_data": 5}  # not a list of readings: iterating it raises
        return reader()(path)

    m = by_name(M.live(view(), broken))
    assert m["units"]["value"] is None and "could not read" in m["units"]["why"]
    assert m["rows"]["value"] == 480


def test_the_post_hoc_markers_under_predict_and_estimate():
    attempts = [{"id": "fit2", "scores": {"linear": {"auc": {"estimate": 0.74, "ci_low": 0.69,
                                                              "ci_high": 0.79}}}}]
    predict = {"purpose": "prediction", "attempt": "fit2", "family": "linear", "metric": "auc",
               "scores": {"linear": {"auc": 0.71}}, "refused": None}
    signal, result = M.post_hoc(predict, attempts)
    assert signal["value"] == {"family": "linear", "metric": "auc", "estimate": 0.74,
                               "ci_low": 0.69, "ci_high": 0.79}
    assert result["value"]["held_out"] == 0.71
    estimate = {"purpose": "inference", "scale": "ratio", "measure": "odds_ratio", "refused": None,
                "estimates": [{"family": "linear", "feature": "x", "estimate": 1.5,
                               "ci_low": 1.25, "ci_high": 1.75}]}
    signal, result = M.post_hoc(estimate, [])
    # null 1 on a ratio scale, whose interval is symmetric on the log scale (the verifier's P5)
    assert signal["value"] == pytest.approx(abs(math.log(1.5))
                                             / ((math.log(1.75) - math.log(1.25)) / 2))
    assert result["value"]["estimate"] == 1.5
    for m in M.post_hoc({"refused": "The held-out rows were never opened."}, []):
        assert m["value"] is None and "never opened" in m["why"]
        Marker.model_validate(m)


def _signal(scale: str, estimate: float, low: float, high: float):
    result = {"purpose": "inference", "scale": scale, "refused": None,
              "estimates": [{"family": "linear", "feature": "x", "estimate": estimate,
                             "ci_low": low, "ci_high": high}]}
    return M.post_hoc(result, [])[0]


@pytest.mark.parametrize("estimate, low, high, expected", [
    # Two odds ratios whose intervals just touch 1, mirror images on the log scale: one half-width.
    (2.0, 1.0, 4.0, 1.0),
    (0.5, 0.25, 1.0, 1.0),
    # An interval that excludes 1 lies more than one half-width away: log 3 / (log 4 / 2).
    (3.0, 1.5, 6.0, math.log(3.0) / (math.log(4.0) / 2)),
])
def test_a_ratios_signal_is_its_distance_from_one_in_half_widths_on_the_log_scale(
        estimate, low, high, expected):
    m = _signal("ratio", estimate, low, high)
    assert m["value"] == pytest.approx(expected, rel=1e-12)
    assert "log scale" in m["basis"]


def test_a_differences_signal_is_its_distance_from_zero_in_half_widths_on_its_own_scale():
    m = _signal("difference", 0.3, 0.1, 0.5)
    assert m["value"] == pytest.approx(0.3 / 0.2, rel=1e-12)
    assert "log scale" not in m["basis"]
    assert _signal("difference", -0.3, -0.5, -0.1)["value"] == pytest.approx(1.5, rel=1e-12)


def test_a_ratio_whose_interval_reaches_zero_has_no_signal_and_says_why():
    m = _signal("ratio", 0.5, 0.0, 1.2)
    assert m["value"] is None and "log" in m["why"]
    Marker.model_validate(m)

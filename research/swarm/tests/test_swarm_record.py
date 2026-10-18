"""The corpus record's schema (DRIVER_PLAN D3; SWARM_PLAN §4) and its writing: each guard fires on a
record built to break it and stays silent on the constructed record that keeps it; a record that
fails is written to ``failed/`` with its error, never to the corpus, and neither is written over."""
from __future__ import annotations

import copy
import json

import pytest
from pydantic import ValidationError

from research.swarm import record as R
from research.swarm.assemble import write
from research.swarm.tests.constructed import estimate_record, predict_record, revert_record


@pytest.mark.parametrize("make", [predict_record, estimate_record, revert_record])
def test_the_constructed_records_validate(make):
    rec = R.validate(make())
    assert rec.version == R.VERSION
    assert rec.decisions[0].seq == 1


@pytest.mark.parametrize("where", ["record", "decision", "result", "run"])
def test_no_part_of_a_record_holds_a_validity(where):
    """V is never set by a run (SWARM_PLAN §4): a validity anywhere fails the schema."""
    data = predict_record()
    target = {"record": data, "decision": data["decisions"][3], "result": data["result"],
              "run": data["run"]}[where]
    target["validity"] = "valid"
    with pytest.raises(ValidationError, match="validity"):
        R.validate(data)


def test_a_materiality_holds_the_engines_measure_or_why_not_both_and_not_neither():
    measured = {"source": "fits", "attempt": "fit2",
                "movement": [{"family": "linear", "movement": {"value": 0.4}}]}
    assert R.Materiality.model_validate(measured).movement
    assert R.Materiality.model_validate({"source": None, "why": "no fit"}).why == "no fit"
    for bad in ({"source": None}, {**measured, "why": "and a why"},
                {"source": None, "movement": measured["movement"]},
                {"source": "mattered", "departures": []}):
        with pytest.raises(ValidationError):
            R.Materiality.model_validate(bad)


def test_a_measure_holds_a_number_and_a_list_of_whys_is_not_one():
    """A fit's movement whose every family has only a why, or a curve whose every departure was
    not estimated, measured nothing: it is a why, not a measure."""
    whys = [{"family": "elastic_net", "movement": None, "why": "not fitted before"}]
    with pytest.raises(ValidationError, match="number"):
        R.Materiality.model_validate({"source": "fits", "attempt": "fit2", "movement": whys})
    R.Materiality.model_validate({"source": "fits", "attempt": "fit2", "movement": whys + [
        {"family": "linear", "movement": {"value": 0.0}, "why": None}]})
    unestimated = [{"choice": "exclusion:0", "spec": None, "not_estimated": "a reading waits"}]
    with pytest.raises(ValidationError, match="number"):
        R.Materiality.model_validate({"source": "mattered", "departures": unestimated})
    R.Materiality.model_validate({"source": "mattered", "departures": unestimated + [
        {"choice": "adjusted:age", "spec": {"estimate": 0.4}, "not_estimated": None}]})


def test_a_run_holds_its_spend_or_why_it_has_none():
    run = predict_record()["run"]
    R.Run.model_validate(run)
    R.Run.model_validate({**run, "spend": None, "spend_why": "no transcript"})
    for bad in ({**run, "spend_why": "and a why"}, {**run, "spend": None}):
        with pytest.raises(ValidationError, match="spend"):
            R.Run.model_validate(bad)


def test_a_result_holds_its_numbers_or_why_it_has_none():
    result = predict_record()["result"]
    R.Result.model_validate(result)
    refused = {**result, "scores": None, "refused": "never opened", "refused_kind": "not_reported"}
    R.Result.model_validate(refused)
    for bad in ({**result, "refused": "and refused"}, {**result, "scores": None},
                {**refused, "refused_kind": None}, {**result, "refused_kind": "not_known"}):
        with pytest.raises(ValidationError, match="result"):
            R.Result.model_validate(bad)


def test_an_export_holds_the_bundles_hash_or_the_refusal():
    export = predict_record()["export"]
    R.Export.model_validate(export)
    R.Export.model_validate({**export, "sha256": None, "refused": {"code": "not_ready"}})
    for bad in ({**export, "refused": {"code": "x"}}, {**export, "sha256": None}):
        with pytest.raises(ValidationError, match="export"):
            R.Export.model_validate(bad)


def test_a_fit_or_a_result_naming_what_the_record_does_not_hold_fails():
    R.validate(predict_record())
    stray = predict_record()
    stray["attempts"][1]["decisions"].append(99)
    with pytest.raises(ValidationError, match="99"):
        R.validate(stray)
    lost = predict_record()
    lost["result"]["attempt"] = "fit9"
    with pytest.raises(ValidationError, match="no attempt"):
        R.validate(lost)
    unread = predict_record()
    unread["result"]["decisions"].append(98)
    with pytest.raises(ValidationError, match="98"):
        R.validate(unread)
    unchanged = predict_record()
    unchanged["attempts"][1]["changed"].append(97)
    with pytest.raises(ValidationError, match="97"):
        R.validate(unchanged)
    for field, shift, stray in (("withdrawn", {"seq": 96, "by": 4}, 96),
                                ("reinstated", {"seq": 2, "by": 95}, 95)):
        moved = predict_record()
        moved["attempts"][1][field] = [shift]
        with pytest.raises(ValidationError, match=str(stray)):
            R.validate(moved)
    repeated = predict_record()
    repeated["decisions"][1]["seq"] = 1
    with pytest.raises(ValidationError, match="seqs"):
        R.validate(repeated)


def test_a_valid_record_is_written_to_the_corpus_once(tmp_path):
    path, ok = write(predict_record(), "run-1", tmp_path)
    assert ok and path == tmp_path / "run-1.json"
    kept = json.loads(path.read_text("utf-8"))
    assert R.validate(kept).run.id == "run-1"
    assert kept["claims"][0]["register"] == "card"  # the wire name, as the analyst log serves it
    assert not (tmp_path / "failed").exists()
    with pytest.raises(FileExistsError):
        write(predict_record(), "run-1", tmp_path)


def test_a_record_that_fails_its_schema_is_a_failed_run_kept_with_its_error(tmp_path):
    raw = predict_record()
    raw["decisions"][3]["materiality"] = {"source": None}
    path, ok = write(copy.deepcopy(raw), "run-1", tmp_path)
    assert not ok and path == tmp_path / "failed" / "run-1.json"
    assert not (tmp_path / "run-1.json").exists()
    body = json.loads(path.read_text("utf-8"))
    assert "materiality" in body["error"] and body["assembled"] == raw
    with pytest.raises(FileExistsError):
        write(predict_record(), "run-1", tmp_path)  # a failed run is not written over by a pass

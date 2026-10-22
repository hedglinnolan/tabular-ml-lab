"""The corpus record's schema (DRIVER_PLAN D3; SWARM_PLAN §4) and its writing: each guard fires on a
record built to break it and stays silent on the constructed record that keeps it; a record that
fails is written to ``failed/`` with its error, never to the corpus, and neither is written over."""
from __future__ import annotations

import copy
import json
import re

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


# ── version 2 (SWARM_PLAN §4's new fields), each guard both ways ────────────


def test_a_version_1_record_is_not_a_version_2_record():
    data = predict_record()
    assert R.validate(data).version == 2
    with pytest.raises(ValidationError, match="version"):
        R.validate({**data, "version": 1})


def test_the_redacted_view_holds_no_model_persona_effort_prompt_or_spend():
    data = predict_record()
    blind = R.redacted(data)
    run = blind["run"]
    assert run["redacted"] is True and run["persona"] is None and run["effort"] is None
    assert run["analyst_model"] == run["prompt_version"] == R.REDACTED
    assert run["spend"] is None and run["spend_why"] == R.REDACTED_SPEND
    assert R.validate(blind).run.redacted
    # Nothing else moves but the times, which keep their order only (the verifier's P2): the rest
    # of the record is the record.
    source = R.validate(data).model_dump(mode="json", by_alias=True)
    assert {k: v for k, v in _timeless(blind).items() if k != "run"} == {
        k: v for k, v in _timeless(source).items() if k != "run"}
    assert data["run"]["analyst_model"] == "claude-haiku-5-5"  # the record itself is untouched


STAMP = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?")


def _timeless(obj):
    """``obj`` with every time in its strings blanked, so what else it holds can be compared."""
    if isinstance(obj, dict):
        return {k: _timeless(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_timeless(v) for v in obj]
    return STAMP.sub("T", obj) if isinstance(obj, str) else obj


def _times(obj, path=()):
    """Every time in ``obj``'s strings, by where it sits and its place in the string."""
    if isinstance(obj, dict):
        return [t for k, v in obj.items() for t in _times(v, (*path, k))]
    if isinstance(obj, list):
        return [t for i, v in enumerate(obj) for t in _times(v, (*path, i))]
    if isinstance(obj, str):
        return [((*path, j), m.group()) for j, m in enumerate(STAMP.finditer(obj))]
    return []


def _instant(text):
    from datetime import datetime, timezone

    t = datetime.fromisoformat(text.replace(" ", "T"))
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def _dry_run_record():
    from pathlib import Path

    path = sorted((Path(R.__file__).parent / "dryrun").glob("run_02-*.json"))[0]
    data = json.loads(path.read_text("utf-8"))
    data["run"].update(analyst_model="claude-haiku-5-5", persona="skeptical",
                       prompt_version="analyst-v2-skeptical")
    return data


@pytest.mark.parametrize("make", [predict_record, estimate_record, _dry_run_record])
def test_the_redacted_view_keeps_no_time_of_the_record_only_their_order(make):
    """With an agent analyst, the gaps between its commands are its model's response time (the
    verifier's P2): no time the record holds survives in the view, each is its place in the
    record's order of times, and every comparison between two of them reads as it did."""
    data = make()
    source = R.validate(data).model_dump(mode="json", by_alias=True)
    blind = R.redacted(data)
    before, after = _times(source), _times(blind)
    assert before and [where for where, _ in before] == [where for where, _ in after]
    shown = json.dumps(blind)
    assert not [t for _, t in before if t in shown]  # no time of the record's survives
    old = [_instant(t) for _, t in before]
    new = [_instant(t) for _, t in after]
    for i in range(len(old)):
        for j in range(len(old)):
            assert (old[i] < old[j]) == (new[i] < new[j]) and (old[i] == old[j]) == (new[i] == new[j])
    # Each is its order only: the n-th distinct time of the record is n seconds after the epoch.
    ranks = {t: n for n, t in enumerate(sorted(set(old)), start=1)}
    assert [int(t.timestamp()) for t in new] == [ranks[t] for t in old]
    R.validate(blind)


def test_the_checks_read_the_redacted_view_as_they_read_the_record():
    from research.swarm import checks

    for make in (predict_record, estimate_record, _dry_run_record):
        data = make()
        said = [(o.check, o.outcome, o.problems, o.unchecked)
                for o in checks.run_checks([R.validate(data)])]
        blind = [(o.check, o.outcome, o.problems, o.unchecked)
                 for o in checks.run_checks([R.validate(R.redacted(data))])]
        assert said == blind, make.__name__


@pytest.mark.parametrize("field,value", [("analyst_model", "claude-haiku-5-5"),
                                         ("persona", "explorer"), ("effort", "high"),
                                         ("prompt_version", "v1")])
def test_a_redacted_run_that_still_names_its_model_or_persona_fails(field, value):
    blind = R.redacted(predict_record())
    blind["run"][field] = value
    with pytest.raises(ValidationError, match="redacted run"):
        R.validate(blind)


def test_a_run_that_is_not_redacted_names_its_model_and_its_persona_is_one_of_three():
    data = predict_record()
    data["run"]["analyst_model"] = R.REDACTED
    with pytest.raises(ValidationError, match="not redacted"):
        R.validate(data)
    for persona in ("conventional", "skeptical", "explorer", None):
        R.validate({**predict_record(), "run": {**predict_record()["run"], "persona": persona}})
    with pytest.raises(ValidationError, match="persona"):
        R.validate({**predict_record(), "run": {**predict_record()["run"], "persona": "bold"}})


def test_a_run_says_which_vocabulary_its_tags_were_taken_under():
    data = predict_record()
    del data["run"]["vocabulary"]
    with pytest.raises(ValidationError, match="vocabulary"):
        R.validate(data)


def _marker(name, value=None, **kw):
    from research.swarm.tests.constructed import marker

    return marker(name, value, **kw)


def test_a_marker_holds_its_value_with_its_basis_or_why_as_the_vocabulary_fixes_it():
    R.Marker.model_validate(_marker("rows", 480))
    R.Marker.model_validate(_marker("rows", why="not read"))
    R.Marker.model_validate(_marker("rows", 0))  # zero is a value
    for bad, match in (({**_marker("rows", 480), "why": "and a why"}, "not both"),
                       (_marker("rows"), "not both"),
                       ({**_marker("rows", 480), "basis": None}, "where the engine served"),
                       ({**_marker("rows", 480), "timing": "post_hoc"}, "vocabulary"),
                       ({**_marker("rows", 480), "axis": "purpose"}, "vocabulary"),
                       ({**_marker("rows", 480), "name": "vibes"}, "name"),
                       # a declared marker is the dataset's, never one the engine reads
                       ({**_marker("rows", 480), "name": "instrument", "timing": "declared",
                         "axis": "measurement"}, "timing")):
        with pytest.raises(ValidationError, match=match):
            R.Marker.model_validate(bad)


def test_the_regime_and_each_fit_list_a_marker_once():
    data = predict_record()
    data["regime"]["markers"].append(_marker("rows", 1))
    with pytest.raises(ValidationError, match="more than once"):
        R.validate(data)
    data = predict_record()
    data["attempts"][0]["markers"].append(_marker("rows", 1))
    with pytest.raises(ValidationError, match="more than once"):
        R.validate(data)


def test_a_fit_holds_its_live_markers_or_why_and_no_post_hoc_one():
    data = predict_record()
    data["attempts"][0].update(markers=None, markers_why="no command read them")
    R.validate(data)
    for markers, why, match in (([_marker("rows", 480)], "and a why", "not both"),
                                (None, None, "not both"),
                                ([_marker("signal", 0.7)], None, "post hoc")):
        bad = predict_record()
        bad["attempts"][0].update(markers=markers, markers_why=why)
        with pytest.raises(ValidationError, match=match):
            R.validate(bad)


def test_an_analysts_tag_names_a_marker_of_the_vocabulary_on_an_analysts_record_only():
    data = predict_record()
    data["decisions"][3]["tag"] = {"marker": "events_per_parameter",
                                   "rejected": {"id": "linear", "decision": {"kind": "select_models",
                                                                             "models": ["linear"]}}}
    assert R.validate(data).decisions[3].tag.marker == "events_per_parameter"
    unknown = predict_record()
    unknown["decisions"][3]["tag"] = {"marker": "hunch"}
    with pytest.raises(ValidationError, match="marker"):
        R.validate(unknown)
    app = estimate_record()
    app["decisions"][3]["tag"] = {"marker": "estimand"}  # the lock, TurboTab's own record
    with pytest.raises(ValidationError, match="TurboTab itself"):
        R.validate(app)


def test_a_want_names_one_of_the_four_kinds_and_a_marker():
    want = {"at": "2026-10-18T10:30:00+00:00", "kind": "option",
            "what": "drift correction by run order", "reason": "the QC samples drift",
            "marker": "run_order"}
    data = predict_record()
    data["wants"] = [want]
    assert R.validate(data).wants[0].kind == "option"
    for field, value in (("kind", "feature"), ("marker", "hunch"), ("reason", ""), ("what", "")):
        bad = predict_record()
        bad["wants"] = [{**want, field: value}]
        with pytest.raises(ValidationError):
            R.validate(bad)


def test_the_account_is_the_drivers_marked_after_the_results_or_not():
    account = {"at": "2026-10-18T10:58:00+00:00", "text": "I kept the linear model.",
               "after_results": True, "after_seal": True}
    data = predict_record()
    data["account"] = account
    assert R.validate(data).account.source == "driver"
    for bad in ({**account, "source": "analyst"}, {**account, "text": ""},
                {k: v for k, v in account.items() if k != "after_results"}):
        broken = predict_record()
        broken["account"] = bad
        with pytest.raises(ValidationError):
            R.validate(broken)


def test_a_replay_holds_its_numbers_over_its_draws_or_why():
    ok = {"scorer": "replay-1", "generator": "estimate_fixture", "draws": 3, "seeds": [1, 2, 3],
          "mean_bias": 0.01, "bias_sd": 0.05, "bias_se": 0.03, "coverage": 0.95}
    assert R.Replay.model_validate(ok).draws == 3
    R.Replay.model_validate({"scorer": "replay-1", "generator": "g", "draws": 1,
                             "why": "every draw was refused", "refused": 1})
    for bad, match in (({**ok, "seeds": [1, 2]}, "one seed per draw"),
                       ({**ok, "why": "and a why"}, "not both"),
                       ({"scorer": "s", "generator": "g", "draws": 2}, "not both"),
                       ({**ok, "refused": 4}, "at most every draw"),
                       ({**ok, "coverage": 1.2}, "less than or equal"),
                       ({**ok, "draws": 0, "seeds": []}, "greater than or equal")):
        with pytest.raises(ValidationError, match=match):
            R.Replay.model_validate(bad)


def test_a_failed_fit_is_refused_exactly_where_the_engine_served_a_refusal():
    refusal = {"code": "nothing_kept", "message": "The elastic net kept no predictor.", "exits": []}
    base = {"at": "2026-10-18T10:20:00+00:00", "command": "fit", "stage": "fit", "key": "k1",
            "error": "ValueError: The elastic net kept no predictor."}
    R.FailedFit.model_validate({**base, "kind": "refused", "refusal": refusal})
    R.FailedFit.model_validate({**base, "kind": "failed", "refusal": None})
    for bad in ({**base, "kind": "refused", "refusal": None},
                {**base, "kind": "failed", "refusal": refusal},
                {**base, "kind": "refused", "refusal": {"code": "nothing_kept"}}):
        with pytest.raises(ValidationError, match="refus"):
            R.FailedFit.model_validate(bad)


def test_a_signed_change_holds_its_number_or_why_and_names_a_fit_the_record_holds():
    change = {"family": "linear", "metric": "auc", "against": "fit1", "paired": True,
              "difference": 0.02, "better": True, "se": 0.01, "df": 4.0, "ci_low": -0.008,
              "ci_high": 0.048, "excludes_zero": False, "folds": [5, 5], "half_width": 0.05,
              "why": None}
    data = predict_record()
    data["attempts"][1]["change"] = [change]
    R.validate(data)
    with pytest.raises(ValidationError, match="not both"):
        R.SignedChange.model_validate({**change, "difference": None})
    stray = predict_record()
    stray["attempts"][1]["change"] = [{**change, "against": "fit9"}]
    with pytest.raises(ValidationError, match="fit9"):
        R.validate(stray)


def test_a_declared_marker_is_quoted_or_not_stated_and_is_a_declared_name():
    quoted = {"name": "assay_platform", "status": "declared", "value": "Roche Cobas 6000",
              "quote": "Plasma glucose was measured on a Roche Cobas 6000.",
              "source": "NCHS GLU_J documentation, Laboratory Method"}
    data = predict_record()
    data["dataset"]["declared"] = [quoted, {"name": "recall_method", "status": "not_stated"}]
    assert [m.status for m in R.validate(data).dataset.declared] == ["declared", "not_stated"]
    for bad in ({**quoted, "quote": None}, {"name": "recall_method", "status": "not_stated",
                                            "value": "24-hour recall"},
                {**quoted, "name": "rows"}):
        broken = predict_record()
        broken["dataset"]["declared"] = [bad]
        with pytest.raises(ValidationError):
            R.validate(broken)
    twice = predict_record()
    twice["dataset"]["declared"] = [quoted, quoted]
    with pytest.raises(ValidationError, match="once"):
        R.validate(twice)


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

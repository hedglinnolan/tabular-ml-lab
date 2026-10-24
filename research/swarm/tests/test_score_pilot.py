"""The pilot's scoring (SWARM_PLAN §7; ``research/swarm/score_pilot.py``): prereg.json's rules
applied to constructed records, replays and sweeps whose verdicts are known by construction, each
rule both ways.

* A replay is judged only as its record's plan, on the pre-registered draws and rows on the
  frozen engine: one of another dataset, bundle or family, another generator, recorded seed or
  evaluation rows, another engine tree or modified files, or with no record is invalid, two of one
  run leave the plan not judged, and with no crash fill nothing is judged; the well-formed one is.
  The replay command refuses a bundle that is not the record's export, and a replay names the
  engine it ran on. A record made on another engine tree, with tracked files modified or not
  known to be unmodified is refused with why and scored nowhere, one on the frozen engine is
  scored, and with no crash fill every record is refused.
* A replay on the Estimate fixture: a mean bias built just inside b0 passes and just outside fails;
  a coverage count at each end of the band passes and one beside it fails; a refused draw makes a
  refused plan, and draws other than the pre-registered ones are refused as invalid. A replay
  refused as a whole (its run reported no result) is a refused plan with its words, one whose
  reported number the bundle does not hold once unjudged; an exported record whose run reported
  no result is a refused plan in its model's refusal rate, given a replay or not.
* On the Predict fixture: a regret below r0 passes, above fails, and with no r0 the plan is not
  judged. On the measurement fixture: a discovery set built with one false and one missed feature
  gives its proportion and power, an effect error built at 0.1 away from zero gives 0.1.
* The unit: 6 of 9 within, 5 of 9 outside, refused plans counted as neither, one plan not judged.
* The sweeps: prereg.json's margins on a constructed refit's numbers agree with the sweep's verdict
  where it applied the same rule, and a verdict built to differ is listed; the measurement
  fixture's discovery-set rule at k and below it, and refused without q-values; on the shape
  sweep.py writes there (its estimates carry no q-value) the rule is applied to no refit, the
  dataset holds no material set, and the report counts no decision material or not; a decision
  none of whose refits is read is counted apart as not read, with why; a refit scored as the plan
  in every one of its folds is not material and read, one whose folds all moved by the same amount
  is not read.
* Reach over a union of three, none over two; the persona rule kept, not kept and deferred.
* The referees' share, the export rate with the blind reader's classes, the decision rule (the
  cheaper of two acceptable, none acceptable, and no analyst named while the reference is outside,
  not judged or without a run), early stopping (once the model has a record under every persona,
  a stuck run's included, and its unit is outside; not while its unit is not judged) and the
  projected spend.
* The report's sentences carry the numbers of the report they come from; the coverage band's
  false alarms are bounded over every plan judged on coverage, a count outside the band reported
  as that, never as a false alarm.
* The replay of a plan: the row a draw is read from is located on the recorded bundle (one row,
  never two), and (slow) on two fresh draws its estimate and interval are statsmodels' on the
  generator's own table at each seed.
"""
from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np
import pytest

from research.swarm import generators as G, prereg_fill as P, score_pilot as S
from research.swarm.record import validate
from research.swarm.tests import constructed as C

ENGINE = {"commit": "c0", "engine_tree": "t0", "engine_modified": False}


@pytest.fixture(scope="module")
def prereg():
    return P.load()


@pytest.fixture(scope="module")
def filled(prereg):
    """prereg.json with its crash fields filled from constructed reports (engine tree ``t0``,
    :data:`ENGINE`'s) and r0 measured on a constructed replay of the default plan."""
    from research.swarm.tests.test_prereg import measured

    return measured(prereg)


@pytest.fixture(scope="module")
def crashed(prereg):
    """prereg.json with its crash fields filled and r0 not yet measured."""
    from research.swarm.tests.test_prereg import reports

    return P.fill(prereg, reports(["boosted_trees", "elastic_net"]), engine=ENGINE)


def seeds(prereg) -> list[int]:
    first, last = prereg["replays"]["seeds"]
    return list(range(first, last + 1))


def record(kind: str = "estimate", run: str = "run_01", model: str = "claude-haiku-5-5",
           persona: str = "conventional", dataset: str | None = None, reached: bool = True,
           fresh: int = 1000, minute: int = 0, exported: bool = True,
           result_refused: str | None = None, commit: str = ENGINE["commit"],
           modified: bool | None = False) -> Any:
    """A constructed record, made on the frozen engine (:data:`ENGINE`'s commit, which the crash
    fill names, no file modified) unless ``commit`` and ``modified`` say otherwise; ``exported``
    False: the run stuck before its export, which the server refused after it (no bundle, so no
    plan); ``result_refused``: the result's kind of none (``record.RefusedKind``), the run having
    reported no estimate."""
    data = C.estimate_record() if kind == "estimate" else C.predict_record()
    data["run"].update(id=run, analyst_model=model, persona=persona,
                       started=f"2026-10-24T10:{minute:02d}:00+00:00", engine_commit=commit,
                       engine_modified=modified)
    data["run"]["spend"]["fresh"] = fresh
    data["dataset"]["id"] = dataset or {"estimate": P.ESTIMATE_FIXTURE,
                                        "predict": P.PREDICT_FIXTURE}[kind]
    data["export"]["reached"] = reached and exported
    if not exported:
        data["export"].update(sha256=None, methods=None, checklist=None,
                              refused={"code": "result_not_ready", "message": "no fit"})
    if result_refused:
        data["result"].update(estimates=None, measure=None, scale=None, conditioning=None,
                              source=None, refused=f"constructed: {result_refused}",
                              refused_kind=result_refused)
    for d in data["decisions"]:  # every preview made through the driver, with its reason
        for e in d["examined"]:
            if e["how"] == "preview":
                e["reason"] = "the preview's reason"
    return validate(data)


# ── the Estimate fixture ─────────────────────────────────────────────────────


def made(prereg, fixture: str, run: str, target: dict, **more: Any) -> dict:
    """A replay's top as ``make_replay`` writes it for :func:`record`'s plan: the fixture's
    generator and recorded seed, the record's bundle and reading, the crash sweeps' engine."""
    fx = prereg[fixture]
    return {"run": run, "dataset": fx["dataset"], "generator": fx["generator"],
            "recorded_seed": fx["recorded_seed"], "target": target,
            "bundle_sha256": C.surroundings()["export"]["sha256"], "engine": dict(ENGINE)} | more


def estimate_replay(prereg, *, bias: float, covered: int, run: str = "run_01") -> dict:
    """Draws whose estimates are the planted effect plus ``bias`` and ±0.5 in turn (so their mean
    bias is ``bias``), the first ``covered`` with a wide interval holding the effect and the rest
    a narrow one beside it."""
    effect = prereg["estimate_fixture"]["effect"]
    draws = []
    for i, s in enumerate(seeds(prereg)):
        est = effect + bias + (0.5 if i % 2 else -0.5)
        half = 10.0 if i < covered else 1e-6
        draws.append({"seed": s, "refused": None, "estimate": est, "ci_low": est - half,
                      "ci_high": est + half})
    return made(prereg, "estimate_fixture", run, S.target_of(record(run=run), prereg),
                draws=draws)


def test_a_mean_bias_just_inside_b0_passes_and_just_outside_fails(filled):
    prereg = filled
    b0 = prereg["estimate_fixture"]["b0"]
    inside = S.score_replay(estimate_replay(prereg, bias=b0 - 1e-6, covered=95), prereg)
    outside = S.score_replay(estimate_replay(prereg, bias=-(b0 + 1e-6), covered=95), prereg)
    assert inside["mean_bias"] == pytest.approx(b0 - 1e-6, abs=1e-9)
    assert (inside["verdict"], inside["bias_within"]) == ("within", True)
    assert (outside["verdict"], outside["bias_within"], outside["coverage_within"]) == (
        "outside", False, True)


def test_a_coverage_count_at_each_end_of_the_band_passes_and_beside_it_fails(filled):
    prereg = filled
    band = prereg["estimate_fixture"]["coverage"]
    for count, verdict in ((band["low"], "within"), (band["high"], "within"),
                           (band["low"] - 1, "outside"), (band["high"] + 1, "outside")):
        got = S.score_replay(estimate_replay(prereg, bias=0.0, covered=count), prereg)
        assert (got["covered"], got["verdict"]) == (count, verdict)


def test_a_refused_draw_makes_a_refused_plan_and_other_draws_are_invalid(filled):
    prereg = filled
    replay = estimate_replay(prereg, bias=0.0, covered=95)
    replay["draws"][3] = {"seed": replay["draws"][3]["seed"], "refused": "a spline's k changed"}
    got = S.score_replay(replay, prereg)
    assert got["verdict"] == "refused" and got["reasons"] == {"a spline's k changed": 1}
    short = estimate_replay(prereg, bias=0.0, covered=95)
    short["draws"] = short["draws"][:-1]
    assert S.score_replay(short, prereg)["verdict"] == "invalid"
    moved = estimate_replay(prereg, bias=0.0, covered=95)
    moved["draws"][0]["seed"] = 1
    assert S.score_replay(moved, prereg)["verdict"] == "invalid"


def test_a_replay_refused_as_a_whole_is_a_refused_plan_and_one_unread_is_unjudged(filled):
    """A replay with no draws carries why at its top: refused (the run reported no result) is a
    refused plan with those words, unjudged (the bundle holds no one row equal to the record's
    number) is unjudged with its words; neither is a replay of the wrong draws."""
    prereg = filled
    whole = estimate_replay(prereg, bias=0.0, covered=95, run="x") | {
        "draws": [], "refused": "The run reported no result (none reported): never locked"}
    got = S.score_replay(whole, prereg)
    assert (got["verdict"], got["why"]) == ("refused", whole["refused"])
    unread = {"run": "x", "dataset": P.ESTIMATE_FIXTURE, "draws": [], "unjudged": "no one row"}
    got = S.score_replay(unread, prereg)
    assert (got["verdict"], got["why"]) == ("unjudged", "no one row")
    got = S.score_replay({**whole, "refused": None}, prereg)
    assert got["verdict"] == "invalid" and "pre-registered ones" in got["why"]
    assert S.score_replay(estimate_replay(prereg, bias=0.0, covered=95), prereg)["verdict"] == \
        "within"


def _bundle(tmp_path, estimates: dict) -> Any:
    import json
    import zipfile

    path = tmp_path / "export.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("turbotab-export/provenance.json", json.dumps({"estimates": estimates}))
    return path


def test_the_replay_of_a_record_with_no_result_is_refused_and_of_an_unlocated_row_unjudged(
        prereg, tmp_path):
    """Before any draw: a record whose result the run reported none of (or the engine served none
    of) is replayed as refused with its words, one whose record cannot say what was reported as
    unjudged, as the record's own checks read them; a reported number the bundle does not hold
    once is unjudged."""
    import hashlib

    bundle = _bundle(tmp_path, {"table2|linear/model_2/x|estimate": 0.30})
    for kind, key in (("not_reported", "refused"), ("not_served", "refused"),
                      ("not_known", "unjudged")):
        rec = record(run="r", result_refused=kind)
        target = S.target_of(rec, prereg)
        assert target["result_refused"] == {"kind": kind, "words": f"constructed: {kind}"}
        got = S.make_replay(bundle, target, P.ESTIMATE_FIXTURE, "r", prereg, seeds=[],
                            engine=ENGINE)
        assert got["draws"] == [] and f"constructed: {kind}" in got[key], kind
        assert S.score_replay(got, prereg)["verdict"] == key
    target = S.target_of(record(run="r"), prereg)
    assert "result_refused" not in target and target["rows"][0]["estimate"] == 0.30
    got = S.make_replay(bundle, target, P.ESTIMATE_FIXTURE, "r", prereg, seeds=[],
                        engine=ENGINE)
    assert got.get("unjudged") is None and got.get("refused") is None
    assert got["bundle_sha256"] == hashlib.sha256(bundle.read_bytes()).hexdigest()
    other = _bundle(tmp_path, {"table2|linear/model_2/x|estimate": 0.31})
    got = S.make_replay(other, target, P.ESTIMATE_FIXTURE, "r", prereg, seeds=[],
                        engine=ENGINE)
    assert "no one row" in got["unjudged"] and S.score_replay(got, prereg)["verdict"] == "unjudged"


def test_a_fixture_record_whose_result_was_refused_is_a_refused_plan_in_the_refusal_rate(filled):
    """With no replay given, an exported record on a fixture whose run reported no result is a
    refused plan with its words, counted as neither in the unit and in the model's refusal rate;
    one that cannot say what was reported is unjudged; one with its numbers is still unscored. A
    replay given for a record with no result does not judge it."""
    prereg = filled
    b0 = prereg["estimate_fixture"]["b0"]
    recs = [record(run="a"), record(run="b"), record(run="none", result_refused="not_reported")]
    replays = [estimate_replay(prereg, bias=2 * b0, covered=95, run=r) for r in ("a", "b")]
    replays.append(estimate_replay(prereg, bias=0.0, covered=95, run="none"))
    report = S.score(recs, replays, [], prereg)
    plans = {p["run"]: p for p in report["plans"]}
    assert plans["none"]["verdict"] == "refused"
    assert "constructed: not_reported" in plans["none"]["why"]
    haiku = report["models"]["haiku"]
    assert haiku["units"]["estimate"]["verdict"] == "outside"  # 0 of 2 judged within
    assert (haiku["refusals"]["refused"], haiku["refusals"]["share"]) == (1, 1 / 3)
    recs = [record(run="unknown", result_refused="not_known"), record(run="whole")]
    plans = {p["run"]: p for p in S.score(recs, [], [], prereg)["plans"]}
    assert plans["unknown"]["verdict"] == "unjudged" and "not_known" in plans["unknown"]["why"]
    assert plans["whole"]["verdict"] == "unscored"
    stuck = record(run="stuck", exported=False, result_refused="not_reported")
    assert S.score([stuck], [], [], prereg)["plans"] == []  # no bundle, so no plan


# ── a replay judged only as its record's plan, on the pre-registered draws, rows and engine ──


@pytest.mark.parametrize("change,said", [
    (lambda r: r | {"dataset": P.MEASUREMENT_FIXTURE}, "dataset is 'plasma_panel_estimate'"),
    (lambda r: r | {"bundle_sha256": "0" * 64}, "bundle is '" + "0" * 64),
    (lambda r: r | {"target": r["target"] | {"family": "ridge"}}, "family `ridge`"),
    (lambda r: r | {"target": r["target"] | {"rows": []}}, "not what its record reported"),
    (lambda r: r | {"generator": "given: tests:small"}, "generator is 'given: tests:small'"),
    (lambda r: r | {"recorded_seed": 7}, "recorded seed is 7"),
    (lambda r: r | {"engine": {**ENGINE, "engine_tree": "t1"}},
     "engine tree t1 (no file modified)"),
    (lambda r: r | {"engine": {**ENGINE, "engine_modified": True}},
     "engine tree t0 (tracked files modified)"),
    (lambda r: {k: v for k, v in r.items() if k != "engine"},
     "engine tree None (not known to be unmodified)"),
])
def test_a_replay_not_its_records_plan_on_the_preregistered_draws_and_engine_is_invalid(
        filled, change, said):
    """The record's own replay is judged; changed in one way (another dataset, bundle or reading,
    another generator or recorded seed, another engine tree or modified files, or none named) it
    is the record's plan no longer, and invalid with why, in the plan's own kind."""
    rec = record(run="h1")
    good = estimate_replay(filled, bias=0.0, covered=95, run="h1")
    assert [p["verdict"] for p in S.score([rec], [good], [], filled)["plans"]] == ["within"]
    [plan] = S.score([rec], [change(copy.deepcopy(good))], [], filled)["plans"]
    assert (plan["run"], plan["kind"], plan["model"], plan["verdict"]) == (
        "h1", "estimate", "haiku", "invalid")
    assert said in plan["why"]
    unit = S.score([rec, record(run="h2")], [change(copy.deepcopy(good)),
                                            estimate_replay(filled, bias=0.0, covered=95,
                                                            run="h2")], [], filled)
    assert unit["models"]["haiku"]["units"]["estimate"]["verdict"] == "not_judged"


def test_a_predict_plan_is_judged_only_on_its_own_replay_and_the_preregistered_rows(filled):
    """On the Predict fixture the record's replay is judged against r0; one on other evaluation
    rows (a test's, named so) is invalid, and an Estimate replay given for the Predict record is
    not its plan: the Predict plan is listed invalid, not judged as an Estimate plan."""
    rec = record(kind="predict", run="p1")
    r0 = filled["predict_fixture"]["r0"]["value"]
    good = predict_replay(filled, r0 * 0.5, run="p1")
    [plan] = S.score([rec], [good], [], filled)["plans"]
    assert (plan["kind"], plan["verdict"]) == ("predict", "within")
    [plan] = S.score([rec], [good | {"evaluation": {"given": True, "rows": 10}}], [],
                     filled)["plans"]
    assert plan["verdict"] == "invalid" and "evaluation rows is {'given': True" in plan["why"]
    [plan] = S.score([rec], [estimate_replay(filled, bias=0.0, covered=95, run="p1")], [],
                     filled)["plans"]
    assert (plan["kind"], plan["verdict"]) == ("predict", "invalid")
    assert "dataset is 'walking_bp_estimate'" in plan["why"]


def test_two_replays_of_one_run_leave_its_plan_not_judged(filled):
    """One replay each: h1 within and h2 outside, the unit outside (1 of 2). A second replay of
    h1 makes one plan of it, not judged, and the unit with it."""
    b0 = filled["estimate_fixture"]["b0"]
    recs = [record(run="h1"), record(run="h2")]
    h1 = estimate_replay(filled, bias=0.0, covered=95, run="h1")
    h2 = estimate_replay(filled, bias=2 * b0, covered=95, run="h2")
    once = S.score(recs, [h1, h2], [], filled)
    assert once["models"]["haiku"]["units"]["estimate"]["verdict"] == "outside"
    twice = S.score(recs, [h1, h2, h1], [], filled)
    plans = {p["run"]: p for p in twice["plans"]}
    assert len(twice["plans"]) == 2 and plans["h2"]["verdict"] == "outside"
    assert plans["h1"]["verdict"] == "unjudged" and "2 replays" in plans["h1"]["why"]
    unit = twice["models"]["haiku"]["units"]["estimate"]
    assert (unit["plans"], unit["judged"], unit["verdict"]) == (2, 1, "not_judged")


def test_a_replay_with_no_record_is_no_plan(filled):
    replay = estimate_replay(filled, bias=0.0, covered=95, run="ghost")
    [plan] = S.score([], [replay], [], filled)["plans"]
    assert (plan["verdict"], plan["model"]) == ("invalid", None) and "No record" in plan["why"]
    [plan] = S.score([record(run="ghost")], [replay], [], filled)["plans"]
    assert plan["verdict"] == "within"


def test_with_no_crash_fill_the_frozen_engine_is_not_known_and_no_replay_is_judged(prereg,
                                                                                  filled):
    replay = estimate_replay(prereg, bias=0.0, covered=95)
    got = S.score_replay(replay, prereg)
    assert got["verdict"] == "unjudged" and "frozen engine" in got["why"]
    assert S.score_replay(replay, filled)["verdict"] == "within"


def test_a_record_not_made_on_the_frozen_engine_is_refused_with_why_and_scores_nothing(filled):
    """F10, both ways. A record made on the frozen engine (the crash fill's commit, no file
    modified) is scored; one made on another commit (whose turbotab/ tree git does not give as
    the frozen one), with tracked files modified, or not known to be unmodified is refused, with
    the reason, and counts nowhere: no plan, no model's runs, no reach; its replay is no plan's."""
    runs = {"h1": {}, "h2": {"commit": "not-the-frozen-commit"}, "h3": {"modified": True},
            "h4": {"modified": None}}
    recs = [record(run=r, minute=i, **more) for i, (r, more) in enumerate(runs.items())]
    replays = [estimate_replay(filled, bias=0.0, covered=95, run=r) for r in runs]
    report = S.score(recs, replays, [], filled)
    refused = {r["run"]: r for r in report["refused_records"]}
    assert set(refused) == {"h2", "h3", "h4"}
    assert "engine commit not-the-frozen-commit" in refused["h2"]["why"]
    assert "tree t0" in refused["h2"]["why"] and "not the frozen engine" in refused["h2"]["why"]
    assert "tracked files modified" in refused["h3"]["why"]
    assert "not known to be unmodified" in refused["h4"]["why"]
    plans = {p["run"]: p for p in report["plans"]}
    assert plans["h1"]["verdict"] == "within"
    for run in ("h2", "h3", "h4"):
        assert plans[run]["verdict"] == "invalid" and plans[run]["model"] is None
        assert plans[run]["why"] == f"The record of this run is refused: {refused[run]['why']}"
    assert report["models"]["haiku"]["runs"] == 1 and set(report["reach"]) == {"h1"}
    text = S.report_md(report, filled)
    assert "## Records refused" in text
    assert f"- h2: {refused['h2']['why']}" in text


def test_with_no_crash_fill_no_record_is_known_to_be_on_the_frozen_engine(prereg):
    """Without the crash fill the frozen engine is not known, so no record can be shown to be
    made on it: each is refused, with why, and nothing is scored."""
    report = S.score([record(run="h1")], [], [], prereg)
    (refused,) = report["refused_records"]
    assert refused["run"] == "h1" and "no crash fill" in refused["why"]
    assert report["models"] == {} and report["plans"] == []


def test_a_replay_names_the_engine_it_ran_on(prereg, tmp_path):
    from research.swarm import launcher

    bundle = _bundle(tmp_path, {"table2|linear/model_2/x|estimate": 0.30})
    target = S.target_of(record(run="r", result_refused="not_reported"), prereg)
    given = S.make_replay(bundle, target, P.ESTIMATE_FIXTURE, "r", prereg, seeds=[],
                          engine=ENGINE)
    assert given["engine"] == ENGINE
    now = S.make_replay(bundle, target, P.ESTIMATE_FIXTURE, "r", prereg, seeds=[])
    assert now["engine"] == launcher.engine_now()
    assert set(now["engine"]) == {"commit", "engine_tree", "engine_modified"}


def test_the_replay_command_refuses_a_bundle_that_is_not_the_records_export(filled, tmp_path,
                                                                           capsys):
    """The record's own bundle is replayed (a record whose run reported no result, so no draw is
    made); another bundle is refused before any draw, and nothing is written."""
    import hashlib
    import json

    path = P.write(filled, tmp_path / "prereg.json")
    bundle = _bundle(tmp_path, {"table2|linear/model_2/x|estimate": 0.30})
    data = record(run="r", result_refused="not_reported").model_dump(mode="json", by_alias=True)
    out = tmp_path / "replay.json"

    def replay(sha: str) -> int:
        data["export"]["sha256"] = sha
        (tmp_path / "record.json").write_text(json.dumps(data))
        return S.main(["--prereg", str(path), "replay", "--record", str(tmp_path / "record.json"),
                       "--bundle", str(bundle), "--out", str(out)])

    assert replay("0" * 64) == 1
    assert "not the record's export" in capsys.readouterr().out and not out.exists()
    assert replay(hashlib.sha256(bundle.read_bytes()).hexdigest()) == 0
    written = json.loads(out.read_text())
    assert "constructed: not_reported" in written["refused"] and written["draws"] == []


# ── the Predict and measurement fixtures ─────────────────────────────────────


def predict_replay(prereg, regret: float, run: str = "run_02") -> dict:
    return made(prereg, "predict_fixture", run, {"kind": "predict", "family": "linear"},
                evaluation=dict(prereg["replays"]["predict_evaluation"]),
                draws=[{"seed": s, "refused": None, "regret": regret} for s in seeds(prereg)])


def test_a_regret_below_r0_passes_above_fails_and_without_r0_is_not_judged(crashed, filled):
    r0 = filled["predict_fixture"]["r0"]["value"]
    assert S.score_replay(predict_replay(filled, r0 * 0.99), filled)["verdict"] == "within"
    assert S.score_replay(predict_replay(filled, r0 * 1.01), filled)["verdict"] == "outside"
    got = S.score_replay(predict_replay(crashed, 0.001), crashed)
    assert got["verdict"] == "unjudged" and "no r0" in got["why"]


def test_the_measurement_fixtures_numbers_are_those_built_into_its_draws(filled):
    prereg = filled
    truth = G.panel_truth()
    change = truth["log2_fold_change"]
    associated = sorted(truth["associated"])
    q = prereg["measurement_fixture"]["q"]
    draws = []
    for s in seeds(prereg):
        features = {f"mz_{j:04d}": {"estimate": 0.0, "q": 0.9} for j in range(1, 151)}
        for f in associated[1:]:  # one associated feature missed
            features[f] = {"estimate": change[f] + math.copysign(0.1, change[f]), "q": q / 2}
        false = next(f for f in features if f not in change)
        features[false] = {"estimate": 0.3, "q": q / 10}  # one false discovery
        draws.append({"seed": s, "refused": None, "scale": "log2", "features": features})
    top = made(prereg, "measurement_fixture", "run_03", {"kind": "measurement"})
    got = S.score_replay(top | {"draws": draws}, prereg)
    found = len(associated)  # 14 true and 1 false
    assert got["verdict"] == "reported"
    assert got["fdp"] == pytest.approx(1 / found) and got["power"] == pytest.approx(
        (len(associated) - 1) / len(associated))
    assert got["effect_error_away_from_zero"] == pytest.approx(0.1)
    signs = [math.copysign(1, change[f]) for f in associated[1:]]
    assert got["effect_error"] == pytest.approx(0.1 * np.mean(signs))
    assert got["power_share"] == pytest.approx(got["power"]
                                               / prereg["measurement_fixture"]["oracle"]["power"])
    for d in draws:
        d["scale"] = None
    plain = S.score_replay(top | {"draws": draws}, prereg)
    assert plain["effect_error"] is None and "log2" in plain["effect_why"]


# ── the unit ─────────────────────────────────────────────────────────────────


def plans(*verdicts: str) -> list[dict]:
    return [{"run": f"r{i}", "verdict": v, "why": None} for i, v in enumerate(verdicts)]


@pytest.mark.parametrize("verdicts,expected", [
    (["within"] * 6 + ["outside"] * 3, "within"),
    (["within"] * 5 + ["outside"] * 4, "outside"),
    (["within"] * 2 + ["outside"] + ["refused"] * 6, "within"),
    (["within", "outside", "outside"], "outside"),
    (["within", "refused", "refused"], "not_judged"),
    (["within", "within", "unscored"], "not_judged"),
])
def test_the_unit_is_two_thirds_of_the_non_refused_plans_with_at_least_two(prereg, verdicts,
                                                                           expected):
    assert S.unit(plans(*verdicts), prereg)["verdict"] == expected


# ── the sweeps ───────────────────────────────────────────────────────────────


def refit(point: str, option: str, change: list[dict], material: Any, **more: Any) -> dict:
    return {"id": f"{point}={option}", "point": point, "option": option, "kind": "scored",
            "change": change, "material": material, **more}


def test_prereg_margins_agree_with_a_sweep_that_applied_them_and_list_one_that_did_not(prereg):
    half = prereg["material"]["half"]
    sweep = {"dataset": P.ESTIMATE_FIXTURE, "plan": "default", "purpose": "inference",
             "fit_seconds": {"total": 7200.0},
             "refits": [refit("question:missing", "complete_case",
                              [{"shift": half * 2.0, "half_width": 2.0, "material": True}], True),
                        refit("question:adjustment", "z_no",
                              [{"shift": half * 2.0 - 1e-6, "half_width": 2.0}], False),
                        refit("question:exclusions", "age", [{"shift": 0.1, "half_width": 2.0}],
                              True),
                        {"id": "question:models=x", "point": "question:models", "option": "x",
                         "kind": "refused", "material": None}],
             "decisions": [{"point": "question:missing", "material": True},
                           {"point": "question:adjustment", "material": False},
                           {"point": "question:exclusions", "material": True}]}
    got = S.sweep_verdicts(sweep, prereg)
    assert [r["prereg"] for r in got["refits"]] == [True, False, False]
    assert [d["refit"] for d in got["disagreements"]] == ["question:exclusions=age"]
    assert [d["point"] for d in got["decision_disagreements"]] == ["question:exclusions"]
    assert got["material"] == ["question:missing"]


def test_under_predict_a_refit_is_material_only_where_its_interval_excludes_zero(prereg):
    half = prereg["material"]["half"]
    big = {"difference": -0.02, "half_width": 0.02, "excludes_zero": True}
    sweep = {"dataset": P.PREDICT_FIXTURE, "plan": "default", "purpose": "prediction",
             "refits": [refit("question:missing", "a", [big], True),
                        refit("question:missing", "b", [{**big, "excludes_zero": False}], False),
                        refit("question:split", "c", [{**big, "difference": -half * 0.02 + 1e-9}],
                              False),
                        refit("question:split", "d", [{"difference": None, "half_width": 0.02,
                                                       "excludes_zero": None}], None,
                              change_why="unpaired folds")],
             "decisions": []}
    got = S.sweep_verdicts(sweep, prereg)
    assert [r["prereg"] for r in got["refits"]] == [True, False, False, None]
    assert got["refits"][3]["why"] == "unpaired folds" and got["disagreements"] == []


def predict_refit(option: str, folds: list[float], *, paired: bool = True,
                  digest: str = "d0") -> dict:
    """A Predict refit as sweep.py writes it: its fit's scores (one family's log loss over
    ``folds``) and its change against the plan's as ``signed.change`` gives it."""
    from research.swarm import signed

    plan_folds = PLAN_FOLDS
    change = signed.change({"folds": folds, "repeats": 1, "estimator": "fold_mean"},
                           {"folds": plan_folds, "repeats": 1, "estimator": "fold_mean",
                            "ci_low": 0.60, "ci_high": 0.70},
                           same_folds=paired, higher_is_better=False, metric="log_loss",
                           after_split=FIVE_FOLDS, before_split=FIVE_FOLDS)
    return refit("question:missing", option,
                 [{"family": "linear", "metric": "log_loss", "against": "f0", **change}], None,
                 scores={"metric": "log_loss", "folds": digest,
                         "families": {"linear": {"log_loss": {"folds": folds}}}})


PLAN_FOLDS = [0.66, 0.64, 0.65, 0.67, 0.63]
FIVE_FOLDS = {"n_train": 100, "folds": 5, "repeats": 1, "fold_scheme": "random",
              "validation": "kfold", "grouped_by": None}


def test_a_refit_scored_as_the_plan_in_every_fold_is_not_material_and_is_read(prereg):
    """F15, both ways: a refit whose fold scores are the plan's on the same folds has no signed
    change (no spread) but a zero one by construction, so prereg.json reads it not material, and
    its decision is read, not counted apart; one whose every fold moved by the same 0.01 is not the
    plan again, and stays not read. The plan's scores on other folds are an unpaired change with
    a number (zero), read by the rule as any number is."""
    sweep = {"dataset": P.PREDICT_FIXTURE, "plan": "default", "purpose": "prediction",
             "baseline": {"scores": {"metric": "log_loss", "folds": "d0", "families": {
                 "linear": {"log_loss": {"folds": PLAN_FOLDS}}}}},
             "refits": [predict_refit("same", list(PLAN_FOLDS)),
                        predict_refit("shifted", [x + 0.01 for x in PLAN_FOLDS]),
                        predict_refit("elsewhere", list(PLAN_FOLDS), paired=False,
                                      digest="d1")],
             "decisions": [{"point": "question:missing", "material": None}]}
    changes = [r["change"][0] for r in sweep["refits"]]
    assert [c["difference"] is None for c in changes] == [True, True, False]
    got = S.sweep_verdicts(sweep, prereg)
    verdicts = {r["option"]: r["prereg"] for r in got["refits"]}
    assert verdicts["same"] is False and verdicts["shifted"] is None
    assert verdicts["elsewhere"] is False  # unpaired, a number of zero: not material by the rule
    (same,) = [r for r in got["refits"] if r["option"] == "same"]
    assert same["why"] is None
    (decision,) = got["decisions"]
    assert decision["prereg"] is False and got["not_read"] == []
    alone = {**sweep, "refits": sweep["refits"][1:2]}
    assert S.sweep_verdicts(alone, prereg)["not_read"][0]["point"] == "question:missing"


def test_the_measurement_fixtures_discovery_set_rule_at_k_below_it_and_without_q(prereg):
    k = prereg["material"]["measurement"]["k"]
    q = prereg["material"]["measurement"]["q"]
    base = [{"feature": f"mz_{j:04d}", "estimate": 1.0, "q": q / 2 if j <= 10 else 0.5}
            for j in range(1, 21)]

    def moved(n: int) -> list[dict]:
        rows = copy.deepcopy(base)
        for r in rows[10:10 + n]:
            r["q"] = q / 2
        return rows

    sweep = {"dataset": P.MEASUREMENT_FIXTURE, "plan": "default", "purpose": "inference",
             "baseline": {"estimates": base},
             "refits": [refit("question:batch", "k", [], None, estimates=moved(k)),
                        refit("question:batch", "below", [], None, estimates=moved(k - 1)),
                        refit("question:batch", "no_q", [], None,
                              estimates=[{k_: v for k_, v in r.items() if k_ != "q"}
                                         for r in base])],
             "decisions": []}
    got = S.sweep_verdicts(sweep, prereg)
    assert [r["prereg"] for r in got["refits"]] == [True, False, None]
    assert "q-value" in got["refits"][2]["why"]
    assert len(got["disagreements"]) == 2  # the sweep read no discovery set: None against both


def sweep_shaped(dataset: str, purpose: str, plan_rows: list[dict],
                 refits: list[tuple[str, str, list[dict]]]) -> dict:
    """A sweep as sweep.py writes one: its refits and the plan's as ``sweep.Refit``, each refit's
    change and material verdict ``sweep.materiality``'s, its decisions ``sweep.decisions_of``'."""
    from research.swarm import sweep as W

    plan = W.Refit(point=W.PLAN, option=None, decision=None, kind="scored", estimates=plan_rows)
    done = []
    for point, option, rows in refits:
        r = W.Refit(point=point, option=option, decision=None, kind="scored", estimates=rows)
        W.materiality(r, plan, purpose)
        done.append(r)
    points = [W.Point(*p.split(":", 1)) for p in dict.fromkeys(r[0] for r in refits)]
    return {"dataset": dataset, "plan": "default", "purpose": purpose, "baseline": plan.json(),
            "refits": [r.json() for r in done], "decisions": W.decisions_of(points, done),
            "fit_seconds": {"total": 60.0}}


def panel_rows(shift: float, q: float) -> list[dict]:
    """Six features' rows as the measurement fixture's fit serves them, each with its q-value,
    read into the sweep's estimates as ``assemble.fit_estimates`` reads them (sweep.py's
    ``estimate_of``); the first three shifted by ``shift``."""
    from research.swarm import assemble as A

    served = [{"feature": f"mz_{j:04d}", "estimate": 0.1 * j + (shift if j <= 3 else 0.0),
               "ci_low": 0.1 * j - 0.2, "ci_high": 0.1 * j + 0.2, "q": q / 2 if j <= 3 else 0.5}
              for j in range(1, 7)]
    return A.fit_estimates({"featurewise": served}, "difference")


def test_on_the_measurement_sweep_sweep_py_writes_the_k_rule_is_applied_to_no_refit(prereg):
    """sweep.py's estimates on the measurement fixture carry no q-value (assemble.fit_estimates
    keeps none), so prereg.json's discovery-set rule reads no refit: the sweep's own verdicts (the
    half-width rule) are listed beside 'not read', the dataset holds no material set, and the
    report says the rule was applied to no refit instead of counting 0 material decisions. A
    sweep whose estimates carry q-values is read (the test above)."""
    q = prereg["material"]["measurement"]["q"]
    sweep = sweep_shaped(P.MEASUREMENT_FIXTURE, "inference", panel_rows(0.0, q),
                         [("question:batch", "a", panel_rows(0.5, q)),
                          ("question:batch", "b", panel_rows(0.0, q))])
    assert all("q" not in e for r in sweep["refits"] for e in r["estimates"])
    assert [r["material"] for r in sweep["refits"]] == [True, False]  # sweep.py's own rule
    got = S.sweep_verdicts(sweep, prereg)
    assert [r["prereg"] for r in got["refits"]] == [None, None]
    assert (got["applied"], got["material"]) == (False, [])
    assert "no q-value" in got["not_applied"]
    assert got["not_read"] == [{"point": "question:batch", "why": got["not_applied"]}]
    report = S.score([], [], [sweep], prereg)
    assert report["material"] == {P.MEASUREMENT_FIXTURE: None}
    assert "no q-value" in report["material_unread"][P.MEASUREMENT_FIXTURE]
    text = S.report_md(report, prereg)
    k = prereg["material"]["measurement"]["k"]
    assert (f"- {P.MEASUREMENT_FIXTURE} around the default plan: prereg.json's discovery-set "
            f"rule (k = {k} features at q < {q:g}) was applied to no refit") in text
    assert "0 material" not in text
    assert "`question:batch=a` (sweep yes, prereg.json not read)" in text


def test_a_decision_none_of_whose_refits_is_read_is_counted_apart_as_not_read(prereg):
    """On an Estimate sweep one decision's refit shifts the estimate by a whole half-width
    (material), and another's refit reports no estimate the plan reports (no number): the report
    reads '1 material, 1 not read' with the reason, and prints 'not read' for that verdict."""
    def rows(shift: float, feature: str = "walks_daily") -> list[dict]:
        return [{"family": "linear", "feature": feature, "setting": None, "estimate": shift,
                 "ci_low": shift - 1.0, "ci_high": shift + 1.0, "why": None}]

    sweep = sweep_shaped(P.ESTIMATE_FIXTURE, "inference", rows(0.0),
                         [("question:missing", "complete_case", rows(1.0)),
                          ("question:exclusions", "age", rows(0.0, feature="other"))])
    got = S.sweep_verdicts(sweep, prereg)
    assert [r["prereg"] for r in got["refits"]] == [True, None]
    assert got["applied"] is True and got["material"] == ["question:missing"]
    [unread] = got["not_read"]
    assert unread == {"point": "question:exclusions",
                      "why": "The plan reports no estimate for this family, feature and setting."}
    text = S.report_md(S.score([], [], [sweep], prereg), prereg)
    assert (f"- {P.ESTIMATE_FIXTURE} around the default plan: 1 material, 1 not read "
            f"(`question:exclusions`: The plan reports no estimate for this family, feature and "
            f"setting); 0 refit(s) where the two verdicts differ.") in text


# ── reach and the personas ───────────────────────────────────────────────────


def examined_record(run: str, *, reason: str | None = "why not") -> Any:
    rec = record(run=run).model_dump(mode="json", by_alias=True)
    rec["decisions"][2]["examined"][0]["reason"] = reason
    return validate(rec)


def test_reach_is_the_share_of_a_union_of_three_and_is_not_reported_below(prereg):
    union = {"question:adjustment", "question:missing", "question:exclusions"}
    assert S.reach_of(examined_record("a"), union, prereg) == pytest.approx(1 / 3)
    assert S.reach_of(examined_record("a", reason=None), union, prereg) == 0.0
    assert S.reach_of(examined_record("a"), set(list(sorted(union))[:2]), prereg) is None


def test_the_persona_rule_keeps_not_keeps_and_defers(prereg):
    recs, reach = [], {}
    values = {"conventional": [0.2, 0.3, 0.25], "explorer": [0.6, 0.7, 0.65],
              "skeptical": [0.3, 0.2, 0.35]}
    for persona, vs in values.items():
        for i, v in enumerate(vs):
            run = f"{persona}-{i}"
            recs.append(record(run=run, persona=persona))
            reach[run] = v
    got = S.persona_rule(recs, reach, prereg)
    assert got["explorer"]["kept"] is True and got["skeptical"]["kept"] is False
    assert got["conventional"]["range"] == pytest.approx(0.1)
    sparse = {**reach, "explorer-1": None, "explorer-2": None}
    assert S.persona_rule(recs, sparse, prereg)["explorer"]["kept"] is None


# ── the referees, robustness and the decision ────────────────────────────────


def test_the_referees_share_meets_four_in_five_and_not_three(prereg):
    verdicts = [{"run": "a", "seq": i, "planted": False, "faithful": i < 4} for i in range(5)]
    verdicts += [{"run": "a", "seq": 9, "planted": True, "faithful": False},
                 {"run": "a", "seq": 10, "planted": True, "faithful": True}]
    got = S.faithfulness(verdicts, {"a"}, prereg)
    assert (got["faithful"], got["judged"], got["meets"], got["catch_rate"]) == (4, 5, True, 0.5)
    verdicts[3]["faithful"] = False
    assert S.faithfulness(verdicts, {"a"}, prereg)["meets"] is False


def test_the_export_rate_counts_only_analyst_failures_and_waits_for_every_class(prereg):
    recs = [record(run=f"r{i}") for i in range(9)] + [record(run="stuck", reached=False)]
    assert S.robustness(recs, {"stuck": "analyst_failure"}, prereg)["meets"] is True
    recs2 = recs[:8] + [record(run="s2", reached=False), recs[-1]]
    got = S.robustness(recs2, {"stuck": "analyst_failure", "s2": "analyst_failure"}, prereg)
    assert (got["rate"], got["meets"]) == (0.8, False)
    got = S.robustness(recs2, {"stuck": "engine_defect", "s2": "driver_defect"}, prereg)
    assert (got["rate"], got["meets"]) == (1.0, True)
    assert S.robustness(recs2, {"stuck": "analyst_failure"}, prereg)["meets"] is None


def stats(unit: str, faithful: bool | None, exported: bool | None, cost: float | None) -> dict:
    return {"units": {"estimate": {"verdict": unit}}, "faithfulness": {"meets": faithful},
            "robustness": {"meets": exported}, "cost": {"cost_per_useful": cost}}


def test_the_decision_names_the_cheaper_acceptable_model(prereg):
    models = {"haiku": stats("within", True, True, 10.0),
              "sonnet": stats("within", True, True, 30.0),
              "opus": stats("within", None, None, None)}
    assert S.decision(models, prereg)["production"] == "haiku"
    models["haiku"] = stats("within", False, True, 10.0)
    assert S.decision(models, prereg)["production"] == "sonnet"
    models["sonnet"] = stats("outside", True, True, 30.0)
    got = S.decision(models, prereg)
    assert got["production"] is None and "returns to Nolan" in got["why"]


def test_the_decision_waits_on_an_undecided_model(prereg):
    models = {"haiku": stats("within", True, True, 10.0),
              "sonnet": stats("not_judged", True, True, 5.0),
              "opus": stats("within", None, None, None)}
    got = S.decision(models, prereg)
    assert got["production"] is None and "sonnet" in got["why"]


@pytest.mark.parametrize("reference,said", [
    (stats("outside", None, None, None), "outside the margins"),
    ({"units": {"estimate": {"verdict": "not_judged",
                             "why": "1 non-refused plan(s), fewer than 2."}}},
     "not judged (1 non-refused plan(s), fewer than 2)"),
    (None, "has no run"),
])
def test_no_production_analyst_is_named_until_the_reference_is_within(prereg, reference, said):
    """The reference checks the fixture itself (§7), so the candidates' verdicts name no analyst
    until its own unit on the Estimate fixture is within: outside, not judged, or with no run,
    no analyst is named, and why names the reference's state."""
    models = {"haiku": stats("within", True, True, 10.0),
              "sonnet": stats("within", True, True, 30.0),
              "opus": stats("within", None, None, None)}
    assert S.decision(models, prereg)["production"] == "haiku"
    if reference is None:
        del models["opus"]
    else:
        models["opus"] = reference
    got = S.decision(models, prereg)
    assert got["production"] is None
    assert got["why"].startswith("The reference, opus, ") and said in got["why"]
    assert prereg["decision_rule"]["reference_rule"] in got["why"]
    assert got["models"]["haiku"]["acceptable"] is True


def test_early_stopping_stops_a_model_outside_once_it_has_a_record_under_every_persona(filled):
    """A persona has run once the model has a record on the Estimate fixture under it, a run
    stuck before its export (so with no plan) included, and the unit decides: haiku's
    conventional and skeptical plans outside (0 of 2 judged within) and its explorer stuck stop
    it; with no explorer record yet, or with the plans within, it goes on."""
    prereg = filled
    b0 = prereg["estimate_fixture"]["b0"]
    planned = [record(run="hc", persona="conventional"), record(run="hs", persona="skeptical")]
    stuck = record(run="he", persona="explorer", reached=False, exported=False)

    def stopped(recs: list, bias: float) -> tuple[list[str], str]:
        replays = [estimate_replay(prereg, bias=bias, covered=95, run=r.run.id) for r in planned]
        report = S.score(recs, replays, [], prereg)
        return report["early_stopping"]["stopped"], report["models"]["haiku"]["units"][
            "estimate"]["verdict"]

    assert stopped([*planned, stuck], 2 * b0) == (["haiku"], "outside")
    assert stopped(planned, 2 * b0) == ([], "outside")
    assert stopped([*planned, stuck], 0.0) == ([], "within")
    recs = [record(run=f"h{p}", persona=p, fresh=2000) for p in P.PERSONAS]
    got = S.early_stopping({"haiku": {"verdict": "outside"}}, recs, prereg, None)
    assert got["stopped"] == ["haiku"] and got["stop_waves"] is None
    got = S.early_stopping({"haiku": {"verdict": "outside"}}, recs[:2], prereg, None)
    assert got["stopped"] == []


def test_early_stopping_does_not_stop_a_model_whose_unit_is_not_judged(filled):
    """Haiku has a record under every persona on the Estimate fixture, but two of its three runs
    stuck before the export, so one plan is judged, fewer than two: its unit is not judged and
    it goes on. With the skeptical run's plan judged outside beside it, it stops."""
    b0 = filled["estimate_fixture"]["b0"]
    conventional = record(run="hc", persona="conventional")
    stuck = [record(run=f"h{p[0]}", persona=p, reached=False, exported=False)
             for p in ("skeptical", "explorer")]
    outside = estimate_replay(filled, bias=2 * b0, covered=95, run="hc")
    report = S.score([conventional, *stuck], [outside], [], filled)
    unit = report["models"]["haiku"]["units"]["estimate"]
    assert (unit["judged"], unit["verdict"]) == (1, "not_judged")
    assert report["early_stopping"]["stopped"] == []
    skeptical = record(run="hs", persona="skeptical")
    report = S.score([conventional, skeptical, stuck[1]],
                     [outside, estimate_replay(filled, bias=2 * b0, covered=95, run="hs")], [],
                     filled)
    assert report["early_stopping"]["stopped"] == ["haiku"]
    recs = [record(run=f"h{p}", persona=p) for p in P.PERSONAS]
    assert S.early_stopping({"haiku": {"verdict": "not_judged"}}, recs, filled,
                            None)["stopped"] == []


def test_the_projected_spend_stops_the_waves_past_the_budget_and_not_below(prereg):
    recs = [record(run="h", fresh=1000), record(run="s", model="claude-sonnet-5-5", fresh=3000),
            record(run="o", model="claude-opus-5-5", fresh=5000)]
    runs = {m: sum(len(s["personas"]) * len(s["datasets"]) * s["runs"]
                   for s in prereg["models"]["schedule"] if s["model"] == m)
            for m in prereg["models"]["keys"]}
    projected = runs["haiku"] * 1000 + runs["sonnet"] * 3000 + runs["opus"] * 5000
    got = S.early_stopping({}, recs, prereg, projected - 1)
    assert got["projected"] == projected and got["stop_waves"] is True
    assert S.early_stopping({}, recs, prereg, projected)["stop_waves"] is False
    assert S.early_stopping({}, recs[:2], prereg, projected)["projected"] is None


def test_regression_checks_name_a_decision_with_no_reason_and_pass_a_whole_record():
    rec = record()
    assert S.regression_checks(rec) == []
    data = rec.model_dump(mode="json", by_alias=True)
    data["decisions"][0]["reason"] = None
    assert S.regression_checks(validate(data)) == ["Record #1 (set_purpose) holds no reason."]
    data = rec.model_dump(mode="json", by_alias=True)
    data["decisions"][2]["examined"][0]["reason"] = None
    assert S.regression_checks(validate(data)) == [
        "A preview of adjust:z beside record #3 holds no reason."]


# ── the whole and its report ─────────────────────────────────────────────────


def test_the_score_and_its_report_carry_the_same_numbers(filled):
    prereg = filled
    recs = [record(run=f"h{p}", persona=p) for p in P.PERSONAS]
    recs.append(record(run="unreplayed", persona="conventional"))
    b0 = prereg["estimate_fixture"]["b0"]
    replays = [estimate_replay(prereg, bias=0.1, covered=95, run="hconventional"),
               estimate_replay(prereg, bias=b0 * 2, covered=95, run="hskeptical"),
               estimate_replay(prereg, bias=0.0, covered=80, run="hexplorer")]
    points = ("question:adjustment", "question:missing", "question:exclusions")
    sweep = {"dataset": P.ESTIMATE_FIXTURE, "plan": "default", "purpose": "inference",
             "fit_seconds": {"total": 7200.0},
             "refits": [refit(p, "x", [{"shift": 1.0, "half_width": 1.0}], True)
                        for p in points],
             "decisions": [{"point": p, "material": True} for p in points]}
    report = S.score(recs, replays, [sweep], prereg)
    # Three material decisions, every one the sweep's own, over two fit-hours; each run examined
    # one of them (the adjustment, previewed with a reason) on 1,000 fresh tokens.
    assert report["material"] == {P.ESTIMATE_FIXTURE: sorted(points)}
    [found] = report["finding_what_matters"]["sweeps"]
    assert (found["reach"], found["fit_hours"], found["reach_per_fit_hour"]) == (1.0, 2.0, 0.5)
    assert report["reach"]["hexplorer"] == pytest.approx(1 / 3)
    assert report["finding_what_matters"]["agents"]["haiku"] == pytest.approx((1 / 3) / 1e-3)
    # The band's miss rate is a false alarm's chance for a plan covering at exactly 95% only, so
    # over the three plans judged on coverage it bounds the false alarms expected; the explorer's
    # count outside the band is reported as outside it, not as a false alarm.
    alarms = report["false_alarms"]["coverage"]
    assert (alarms["plans"], alarms["outside_band"]) == (3, 1)
    assert alarms["expected_at_most"] == pytest.approx(
        3 * prereg["expected_false_alarms"]["coverage_band"]["per_plan"])
    assert "observed" not in alarms and "expected" not in alarms
    verdicts = {p["run"]: p["verdict"] for p in report["plans"]}
    assert verdicts == {"hconventional": "within", "hskeptical": "outside",
                        "hexplorer": "outside", "unreplayed": "unscored"}
    assert report["models"]["haiku"]["units"]["estimate"]["verdict"] == "not_judged"
    assert report["choosing_well"]["estimate"]["haiku"]["within"] == 1
    text = S.report_md(report, prereg)
    assert f"b0 = {S._n(b0)}" in text
    assert f"| hskeptical | haiku | estimate | outside | {S._n(b0 * 2)} |" in text
    assert report["decision"]["why"] in text
    band = prereg["estimate_fixture"]["coverage"]
    assert (f"3 Estimate plan(s) were judged on coverage. If each covered at exactly 95%, "
            f"{S._n(alarms['expected_at_most'])} of them would be expected to fall outside the "
            f"band {band['low']} to {band['high']} by chance (false alarms); 1 fell outside "
            f"it.") in text
    assert (f"- {P.ESTIMATE_FIXTURE} around the default plan: 3 material, 0 not read; 0 "
            f"refit(s) where the two verdicts differ.") in text


def test_the_plans_to_sweep_are_each_named_run_by_when_it_started_never_by_its_result(prereg):
    est = P.ESTIMATE_FIXTURE
    recs = [record(run="h-late", minute=30), record(run="h-early", minute=5),
            record(run="h-explorer", persona="explorer", minute=1),
            record(run="s-conv", model="claude-sonnet-5-5", minute=9),
            record(run="o-conv", model="claude-opus-5-5", minute=2),
            record(run="arm", model="mechanical-default", persona=None, minute=0),
            record(kind="predict", run="h-predict", minute=3)]
    named = S.named_plans(recs, prereg)
    assert named[est] == [{"model": "haiku", "persona": "conventional", "run": "h-early"},
                          {"model": "sonnet", "persona": "conventional", "run": "s-conv"},
                          {"model": "opus", "persona": "conventional", "run": "o-conv"},
                          {"model": "haiku", "persona": "explorer", "run": "h-explorer"},
                          {"model": "sonnet", "persona": "explorer", "run": None}]
    assert named[P.PREDICT_FIXTURE][0]["run"] == "h-predict"
    assert named["bang2009"][0]["run"] is None


def test_the_command_writes_the_score_and_its_report_from_the_folders_it_is_given(filled,
                                                                                 tmp_path):
    import json

    prereg = filled
    path = P.write(filled, tmp_path / "prereg.json")
    corpus, replays, out = tmp_path / "corpus", tmp_path / "replays", tmp_path / "out"
    corpus.mkdir()
    replays.mkdir()
    rec = record(run="run_07")
    (corpus / "run_07.json").write_text(json.dumps(rec.model_dump(mode="json", by_alias=True)))
    (replays / "run_07.json").write_text(json.dumps(estimate_replay(prereg, bias=0.0, covered=95,
                                                                    run="run_07")))
    assert S.main(["--prereg", str(path), "score", "--corpus", str(corpus), "--replays",
                   str(replays), "--out", str(out)]) == 0
    written = json.loads((out / "score.json").read_text())
    assert [(p["run"], p["verdict"]) for p in written["plans"]] == [("run_07", "within")]
    assert "| run_07 | haiku | estimate | within |" in (out / "score.md").read_text()


def test_the_fill_scripts_check_passes_and_ready_refuses_the_committed_file(capsys):
    assert P.main(["--check"]) == 0
    assert P.main(["ready"]) == 1
    assert "no fit budget" in capsys.readouterr().out


def test_a_decoy_runs_false_alarm_is_counted_against_its_expected_rate(filled):
    prereg = filled

    def decoy(run: str, low: float):
        data = C.predict_record(control="null_decoy")
        data["run"].update(id=run, engine_commit=ENGINE["commit"])  # on the frozen engine
        data["dataset"]["id"] = P.DECOY
        data["result"]["intervals"]["auc"]["ci_low"] = low
        return validate(data)

    quiet = S.score([decoy("d1", 0.45), decoy("d2", 0.45)], [], [], prereg)
    alarmed = S.score([decoy("d1", 0.45), decoy("d2", 0.55)], [], [], prereg)
    per_run = prereg["expected_false_alarms"]["decoy"]["per_run"]
    assert quiet["false_alarms"]["decoy"] == {"runs": 2, "expected": 2 * per_run, "observed": 0}
    assert alarmed["false_alarms"]["decoy"]["observed"] == 1
    assert per_run == pytest.approx((1 - 0.95) / 2)


# ── the replay of a plan ─────────────────────────────────────────────────────


def test_a_draws_row_is_located_on_the_bundle_once_and_never_among_two():
    estimates = {"table2|linear/crude/walks_daily|estimate": -6.0,
                 "table2|linear/model_2/walks_daily|estimate": -4.9,
                 "table2|linear/model_2/walks_daily|ci_low": -6.4,
                 "table2|linear/model_2/age|estimate": -4.9,
                 "table2_marginal|linear/marginal/one higher|rd": 0.1}
    assert S.locate(estimates, "linear", ["walks_daily"], -4.9) == (
        "table2", "linear/model_2/walks_daily", "estimate")
    assert S.locate(estimates, "linear", ["one higher"], 0.1) == (
        "table2_marginal", "linear/marginal/one higher", "rd")
    assert S.locate(estimates, "linear", ["walks_daily", "age"], -4.9) is None
    assert S.locate(estimates, "linear", ["walks_daily"], -5.0) is None
    assert S.locate(estimates, "ridge", ["walks_daily"], -4.9) is None


@pytest.mark.slow
def test_a_replayed_plans_estimate_on_each_fresh_draw_is_statsmodels_on_the_generators_table(
        prereg, tmp_path):
    """A short reference journey on 400 rows of the Estimate generator, its bundle's unadjusted
    row located and read on two fresh draws: each draw's estimate is statsmodels' least squares of
    the outcome on the exposure on the generator's table at that seed, and its interval the
    estimate ± scipy's t quantile at the row's own degrees of freedom times its standard error.
    That standard error is the engine's pooled over the imputed copies, which differs from
    statsmodels' complete-data one by under 1% although the row holds no imputed column (reported
    as an open issue, not settled here)."""
    import statsmodels.api as sm
    from scipy import stats

    from turbotab.core.reference import journeys as J
    from turbotab.core.tests.truths import FIXTURE_TRUTHS

    rows, recorded = 400, 102

    def small(seed: int):
        return G.blood_pressure(seed, rows)

    table = tmp_path / G.ESTIMATE_FILE
    small(recorded).to_csv(table, index=False)
    spec = J.Journey("p6b-estimate", "clinical", "inference", "?", str(table), "test",
                     lambda: table,
                     lambda: J.truth(dict(FIXTURE_TRUTHS[G.ESTIMATE_FILE]), fixture=table.name),
                     target="sbp_mmhg", lenses=("clinical",), exposure="walks_daily",
                     roles={"person_id": "identifier", "age": "covariate",
                            "chronic_pain": "covariate", "walks_daily": "exposure",
                            "height_cm": "covariate", "health_concern_score": "covariate"})
    capture = J.run_journey(spec, bundles=tmp_path, out_dir=tmp_path / "captures")
    assert capture["export"]["status"] == 200, capture["notes"]
    bundle = tmp_path / "p6b-estimate.zip"
    crude = S._bundle_estimates(bundle)["table2|linear/crude/walks_daily|estimate"]
    target = {"kind": "estimate", "family": "linear",
              "rows": [{"feature": "walks_daily", "setting": None, "estimate": crude}]}
    local = copy.deepcopy(prereg)
    local["estimate_fixture"]["recorded_seed"] = recorded
    got = S.make_replay(bundle, target, P.ESTIMATE_FIXTURE, "run_x", local, generator=small,
                        seeds=[recorded + 1, recorded + 2])
    assert [d["seed"] for d in got["draws"]] == [recorded + 1, recorded + 2]
    for d in got["draws"]:
        frame = small(d["seed"])
        ols = sm.OLS(frame["sbp_mmhg"], sm.add_constant(frame[["walks_daily"]].astype(float))).fit()
        assert d["refused"] is None
        assert d["estimate"] == pytest.approx(ols.params["walks_daily"], abs=1e-9)
        assert d["df"] == ols.df_resid
        half = stats.t.ppf(0.975, d["df"]) * d["se"]
        assert (d["ci_low"], d["ci_high"]) == (pytest.approx(d["estimate"] - half, abs=1e-9),
                                               pytest.approx(d["estimate"] + half, abs=1e-9))
        assert d["se"] == pytest.approx(ols.bse["walks_daily"], rel=0.01)


def _training_rows(home, pid: str) -> np.ndarray:
    """The rows the rebuilt plan's design trains on (its design stage, read in its home)."""
    import dataclasses

    from turbotab.core.config import Settings
    from turbotab.core.export.replay import _wait
    from turbotab.core.export.scorer import _bundle
    from turbotab.core.stages.modeling import row_ids_of
    from turbotab.server.service import ProjectService

    service = ProjectService(dataclasses.replace(Settings.from_env(), home=home, mode="local",
                                                 workers=2), replay=True)
    try:
        service.engine.on_decision(pid)
        _wait(service, pid, ("design",), 600)
        design = _bundle(service, pid, "design", service.engine.status(pid))
        return row_ids_of(design.frames["training"])
    finally:
        service.close()


@pytest.mark.slow
def test_a_replayed_plans_regret_on_a_fresh_draw_is_statsmodels_logit_on_its_training_rows(
        prereg, tmp_path):
    """A short reference journey (the straight-line family, a fifth of the rows held out) on 400
    rows of the hardened Predict generator, replayed cut before its opening on one fresh draw: its
    regret is statsmodels' Logit refit on that draw's training rows of the generator's own table
    (sex recoded as the repair says, `f` as 1), its predictions on 1,000 evaluation rows of the
    generator against their true risk."""
    import statsmodels.api as sm

    from turbotab.core.reference import journeys as J
    from turbotab.core.tests.truths import FIXTURE_TRUTHS

    rows, recorded = 400, 101

    def small(seed: int):
        return G.readmission(seed, rows)

    table = tmp_path / G.READMISSION_FILE
    small(recorded).to_csv(table, index=False)
    spec = J.Journey("p6b-predict", "clinical", "prediction", "?", str(table), "test",
                     lambda: table,
                     lambda: J.truth(dict(FIXTURE_TRUTHS[G.READMISSION_FILE]), fixture=table.name),
                     target=G.READMISSION_OUTCOME, lenses=("clinical",), event="1",
                     answers={"split": {"kind": "set_split", "holdout": 0.2, "seed": 0,
                                        "folds": 5},
                              "models": {"kind": "select_models", "models": ["linear"]}},
                     before={"target": J.apply_first_repair(".")})
    capture = J.run_journey(spec, bundles=tmp_path, out_dir=tmp_path / "captures")
    assert capture["export"]["status"] == 200, capture["notes"]
    evaluation = G.readmission(prereg["replays"]["predict_evaluation"]["seed"], 1000)
    local = copy.deepcopy(prereg)
    local["predict_fixture"]["recorded_seed"] = recorded
    seed = recorded + 7
    got = S.make_replay(tmp_path / "p6b-predict.zip", {"kind": "predict", "family": "linear"},
                        P.PREDICT_FIXTURE, "run_y", local, generator=small, seeds=[seed],
                        evaluation=evaluation, keep=tmp_path / "kept")
    [draw] = got["draws"]
    assert draw["refused"] is None and draw["seed"] == seed
    # The replay names what stood in for prereg.json's, so the r0 fill cannot read it as r0's.
    import hashlib

    assert got["evaluation"] == {"given": True, "rows": 1000}
    assert got["generator"].startswith("given: ") and got["generator"] != \
        prereg["predict_fixture"]["generator"]
    assert got["bundle_sha256"] == hashlib.sha256(
        (tmp_path / "p6b-predict.zip").read_bytes()).hexdigest()
    train = _training_rows(tmp_path / "kept" / str(seed) / "home", draw["project"])
    frame = small(seed)
    predictors = [c for c in frame.columns if c not in ("encounter_id", G.READMISSION_OUTCOME)]

    def design(f):
        X = f[predictors].copy()
        X["sex"] = (X["sex"] == "F").astype(float)
        return sm.add_constant(X.astype(float), has_constant="add")

    fit = sm.Logit(frame.loc[train, G.READMISSION_OUTCOME].to_numpy(),
                   design(frame.loc[train])).fit(disp=0, tol=1e-12, maxiter=200)
    expected = float(np.mean((np.asarray(fit.predict(design(evaluation)))
                              - G.readmission_risk(evaluation)) ** 2))
    assert len(train) == int(round(rows * 0.8))
    assert draw["regret"] == pytest.approx(expected, rel=1e-5)

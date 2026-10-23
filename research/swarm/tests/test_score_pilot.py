"""The pilot's scoring (SWARM_PLAN §7; ``research/swarm/score_pilot.py``): prereg.json's rules
applied to constructed records, replays and sweeps whose verdicts are known by construction, each
rule both ways.

* A replay on the Estimate fixture: a mean bias built just inside b0 passes and just outside fails;
  a coverage count at each end of the band passes and one beside it fails; a refused draw makes a
  refused plan, and draws other than the pre-registered ones are refused as invalid.
* On the Predict fixture: a regret below r0 passes, above fails, and with no r0 the plan is not
  judged. On the measurement fixture: a discovery set built with one false and one missed feature
  gives its proportion and power, an effect error built at 0.1 away from zero gives 0.1.
* The unit: 6 of 9 within, 5 of 9 outside, refused plans counted as neither, one plan not judged.
* The sweeps: prereg.json's margins on a constructed refit's numbers agree with the sweep's verdict
  where it applied the same rule, and a verdict built to differ is listed; the measurement
  fixture's discovery-set rule at k and below it, and refused without q-values.
* Reach over a union of three, none over two; the persona rule kept, not kept and deferred.
* The referees' share, the export rate with the blind reader's classes, the decision rule (the
  cheaper of two acceptable, the reference outside, none acceptable), early stopping and the
  projected spend.
* The report's sentences carry the numbers of the report they come from.
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
    """prereg.json with its crash fields filled from constructed reports, the default plan a
    straight-line one, so r0 is half the straight-line class's regret."""
    from research.swarm.tests.test_prereg import reports

    return P.fill(prereg, reports(["elastic_net", "boosted_trees"]), engine=ENGINE)


def seeds(prereg) -> list[int]:
    first, last = prereg["replays"]["seeds"]
    return list(range(first, last + 1))


def record(kind: str = "estimate", run: str = "run_01", model: str = "claude-haiku-5-5",
           persona: str = "conventional", dataset: str | None = None, reached: bool = True,
           fresh: int = 1000, minute: int = 0) -> Any:
    data = C.estimate_record() if kind == "estimate" else C.predict_record()
    data["run"].update(id=run, analyst_model=model, persona=persona,
                       started=f"2026-10-24T10:{minute:02d}:00+00:00")
    data["run"]["spend"]["fresh"] = fresh
    data["dataset"]["id"] = dataset or {"estimate": P.ESTIMATE_FIXTURE,
                                        "predict": P.PREDICT_FIXTURE}[kind]
    data["export"]["reached"] = reached
    for d in data["decisions"]:  # every preview made through the driver, with its reason
        for e in d["examined"]:
            if e["how"] == "preview":
                e["reason"] = "the preview's reason"
    return validate(data)


# ── the Estimate fixture ─────────────────────────────────────────────────────


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
    return {"run": run, "dataset": P.ESTIMATE_FIXTURE, "draws": draws}


def test_a_mean_bias_just_inside_b0_passes_and_just_outside_fails(prereg):
    b0 = prereg["estimate_fixture"]["b0"]
    inside = S.score_replay(estimate_replay(prereg, bias=b0 - 1e-6, covered=95), prereg)
    outside = S.score_replay(estimate_replay(prereg, bias=-(b0 + 1e-6), covered=95), prereg)
    assert inside["mean_bias"] == pytest.approx(b0 - 1e-6, abs=1e-9)
    assert (inside["verdict"], inside["bias_within"]) == ("within", True)
    assert (outside["verdict"], outside["bias_within"], outside["coverage_within"]) == (
        "outside", False, True)


def test_a_coverage_count_at_each_end_of_the_band_passes_and_beside_it_fails(prereg):
    band = prereg["estimate_fixture"]["coverage"]
    for count, verdict in ((band["low"], "within"), (band["high"], "within"),
                           (band["low"] - 1, "outside"), (band["high"] + 1, "outside")):
        got = S.score_replay(estimate_replay(prereg, bias=0.0, covered=count), prereg)
        assert (got["covered"], got["verdict"]) == (count, verdict)


def test_a_refused_draw_makes_a_refused_plan_and_other_draws_are_invalid(prereg):
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


# ── the Predict and measurement fixtures ─────────────────────────────────────


def predict_replay(prereg, regret: float, run: str = "run_02") -> dict:
    return {"run": run, "dataset": P.PREDICT_FIXTURE,
            "draws": [{"seed": s, "refused": None, "regret": regret} for s in seeds(prereg)]}


def test_a_regret_below_r0_passes_above_fails_and_without_r0_is_not_judged(prereg, filled):
    r0 = filled["predict_fixture"]["r0"]["value"]
    assert S.score_replay(predict_replay(filled, r0 * 0.99), filled)["verdict"] == "within"
    assert S.score_replay(predict_replay(filled, r0 * 1.01), filled)["verdict"] == "outside"
    got = S.score_replay(predict_replay(prereg, 0.001), prereg)
    assert got["verdict"] == "unjudged" and "no r0" in got["why"]


def test_the_measurement_fixtures_numbers_are_those_built_into_its_draws(prereg):
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
    got = S.score_replay({"run": "run_03", "dataset": P.MEASUREMENT_FIXTURE, "draws": draws},
                         prereg)
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
    plain = S.score_replay({"run": "run_03", "dataset": P.MEASUREMENT_FIXTURE, "draws": draws},
                           prereg)
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


def test_the_decision_waits_on_an_undecided_model_and_on_a_reference_outside(prereg):
    models = {"haiku": stats("within", True, True, 10.0),
              "sonnet": stats("not_judged", True, True, 5.0),
              "opus": stats("within", None, None, None)}
    got = S.decision(models, prereg)
    assert got["production"] is None and "sonnet" in got["why"]
    models["sonnet"] = stats("within", True, True, 5.0)
    models["opus"] = stats("outside", None, None, None)
    got = S.decision(models, prereg)
    assert got["production"] is None and got["why"] == prereg["decision_rule"]["reference_rule"]


def test_early_stopping_stops_a_model_outside_once_every_persona_is_scored(prereg):
    est = P.ESTIMATE_FIXTURE
    scored = [{"model": "haiku", "dataset": est, "persona": p, "verdict": "outside"}
              for p in P.PERSONAS]
    recs = [record(run=f"h{i}", fresh=2000) for i in range(3)]
    got = S.early_stopping({"haiku": {"verdict": "outside"}}, recs, scored, prereg, None)
    assert got["stopped"] == ["haiku"] and got["stop_waves"] is None
    got = S.early_stopping({"haiku": {"verdict": "outside"}}, recs, scored[:2], prereg, None)
    assert got["stopped"] == []


def test_the_projected_spend_stops_the_waves_past_the_budget_and_not_below(prereg):
    recs = [record(run="h", fresh=1000), record(run="s", model="claude-sonnet-5-5", fresh=3000),
            record(run="o", model="claude-opus-5-5", fresh=5000)]
    runs = {m: sum(len(s["personas"]) * len(s["datasets"]) * s["runs"]
                   for s in prereg["models"]["schedule"] if s["model"] == m)
            for m in prereg["models"]["keys"]}
    projected = runs["haiku"] * 1000 + runs["sonnet"] * 3000 + runs["opus"] * 5000
    got = S.early_stopping({}, recs, [], prereg, projected - 1)
    assert got["projected"] == projected and got["stop_waves"] is True
    assert S.early_stopping({}, recs, [], prereg, projected)["stop_waves"] is False
    assert S.early_stopping({}, recs[:2], [], prereg, projected)["projected"] is None


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


def test_the_score_and_its_report_carry_the_same_numbers(prereg):
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
    # The coverage band's misses are counted among the plans unbiased by the margin only: the
    # conventional and the explorer plans, the explorer's a miss.
    alarms = report["false_alarms"]["coverage"]
    assert (alarms["plans"], alarms["observed"]) == (2, 1)
    assert alarms["expected"] == pytest.approx(
        2 * prereg["expected_false_alarms"]["coverage_band"]["per_plan"])
    verdicts = {p["run"]: p["verdict"] for p in report["plans"]}
    assert verdicts == {"hconventional": "within", "hskeptical": "outside",
                        "hexplorer": "outside", "unreplayed": "unscored"}
    assert report["models"]["haiku"]["units"]["estimate"]["verdict"] == "not_judged"
    assert report["choosing_well"]["estimate"]["haiku"]["within"] == 1
    text = S.report_md(report, prereg)
    assert f"b0 = {S._n(b0)}" in text
    assert f"| hskeptical | haiku | estimate | outside | {S._n(b0 * 2)} |" in text
    assert report["decision"]["why"] in text


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


def test_the_command_writes_the_score_and_its_report_from_the_folders_it_is_given(prereg,
                                                                                 tmp_path):
    import json

    corpus, replays, out = tmp_path / "corpus", tmp_path / "replays", tmp_path / "out"
    corpus.mkdir()
    replays.mkdir()
    rec = record(run="run_07")
    (corpus / "run_07.json").write_text(json.dumps(rec.model_dump(mode="json", by_alias=True)))
    (replays / "run_07.json").write_text(json.dumps(estimate_replay(prereg, bias=0.0, covered=95,
                                                                    run="run_07")))
    assert S.main(["score", "--corpus", str(corpus), "--replays", str(replays), "--out",
                   str(out)]) == 0
    written = json.loads((out / "score.json").read_text())
    assert [(p["run"], p["verdict"]) for p in written["plans"]] == [("run_07", "within")]
    assert "| run_07 | haiku | estimate | within |" in (out / "score.md").read_text()


def test_the_fill_scripts_check_passes_and_ready_refuses_the_committed_file(capsys):
    assert P.main(["--check"]) == 0
    assert P.main(["ready"]) == 1
    assert "no fit budget" in capsys.readouterr().out


def test_a_decoy_runs_false_alarm_is_counted_against_its_expected_rate(prereg):
    def decoy(run: str, low: float):
        data = C.predict_record(control="null_decoy")
        data["run"]["id"] = run
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

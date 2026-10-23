"""The crash sweep on a real engine (SWARM_PLAN §6 step 1, §7; ``research/swarm/sweep.py``), on a
small table written here, never a pilot dataset: 240 rows, three numeric predictors drawn by
numpy, a yes/no outcome from a logistic model in two of them, no blanks.

Each claim from construction: the 5 × 5 scoring's folds read from the split the engine cached
(every training row in exactly one fold of each of five repeats, 48 rows a fold: 240 / 5), and,
for the refit that holds a fifth out, 192 training rows (240 − 48) in the same scheme; the plan's
own answer deduplicated by the plan's digest; a refusal the engine types at the answer
(multiple imputation that reads the outcome under prediction); an HTTP 500 induced in this process
(the server's preview of one option raises), classed failed and reported with its reproduction;
every option the driver offers listed and visited; the materiality of each paired refit
recomputed by numpy and scipy from the two fit records' served fold scores; a second run after
an interruption that refits nothing the first finished; and a stage error induced in the engine
under one option (the cohort failed with an error it did not type; the draw refused in words),
classed failed and reported as a defect with the stage's own words, and refused with the stage,
its code, words and way forward, while every other option is neither.

The options that take fields of their own, on tables written here: a value built from the table
(a range from a predictor's 1st to its 99th percentile, numpy's; an adult floor of 20 years, the
CDC 2000 growth charts' last age; the rarest level by construction) labeled built; the bang2009
card's rules, read from the card and checked against its sentences, each alone under each handling
and its whole population as one answer; each answer's rows, read from the flow the refit served,
equal to a pandas filter of the same codes; a 500 induced on one card rule's preview, a defect; a
resumed crash sweep on a report finished before, fitting only the new options, each fit's record
written by the server three seconds late and each fit still read as scored; and, under Estimate,
the adjustment set of one's own as the covariates the plan's roles name.

A refit's wait and the estimate stages after a fitted refit (the panel audit's F1 and N2), under
Estimate on the walking table, with stand-in stages run as jobs in the workers
(:func:`standin_job`): under one models answer each estimate stage sleeps past the refit's wait
on both workers, and that refit is cancelled, unreached in words that name what it waited on and
for how long, and its copy removed, while the next refit is fitted at once and scored and the
runner, read as the server closes, holds no job queued or running and every stand-in job
cancelled; the effects stage's bare error under complete cases, read after the fit, a failed
refit and an engine defect; its typed refusal under another answer, refused with its code; the
plan and the next refit, neither in error, scored; and a second run, which presses Fit no more."""
from __future__ import annotations

import json
import math
import re

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from research.swarm import arms, sweep as W

ONLY = ["question:missing", "question:split", "declaration:set_selection"]
QUESTION = arms.Question(target="y", purpose="prediction", meaning="an event (1 yes, 0 no)")
CRASHES_ON = "vip"  # the selection option whose preview the server fails on, here


def table(path):
    rng = np.random.default_rng(1)
    n = 240
    frame = pd.DataFrame({f"x{i}": rng.normal(size=n).round(3) for i in (1, 2, 3)})
    p = 1 / (1 + np.exp(-(0.8 * frame.x1 - 0.5 * frame.x2)))
    frame["y"] = (rng.random(n) < p).astype(int)
    frame.to_csv(path, index=False)
    return path


class Interrupted(BaseException):
    """An interruption, as a keyboard's: not an error the sweep catches."""


@pytest.fixture(scope="module")
def swept(tmp_path_factory):
    """The crash sweep run twice into one folder: the first stopped after its third refit, the
    second taking it up; with the presses of Fit each run made counted, and the preview of one
    option failing in the server."""
    from turbotab.server.service import ProjectService

    work = tmp_path_factory.mktemp("crash")
    data = table(work / "tiny.csv")
    out = work / "out"
    presses = {"first": 0, "second": 0}
    run = {"now": "first"}
    original_fit, original_preview = arms.Local.fit, ProjectService._preview

    def counted(self, wait=3600.0):
        presses[run["now"]] += 1
        return original_fit(self, wait)

    def failing(self, pid, decision):
        body = decision.model_dump() if hasattr(decision, "model_dump") else dict(decision)
        if body.get("kind") == "set_selection" and body.get("method") == CRASHES_ON:
            raise RuntimeError("the preview fails here, as a defect would")
        return original_preview(self, pid, decision)

    mp = pytest.MonkeyPatch()
    mp.setattr(arms.Local, "fit", counted)
    mp.setattr(ProjectService, "_preview", failing)
    refits: list[str] = []

    def stop_after_three(rid: str) -> None:
        refits.append(rid)
        if run["now"] == "first" and len(refits) == 4:
            raise Interrupted()

    try:
        with pytest.raises(Interrupted):
            W.crash("tiny", out, data=data, question=QUESTION, only=ONLY, log=stop_after_three)
        first = [json.loads(x) for x in (out / "progress.jsonl").read_text().splitlines()]
        run["now"] = "second"
        report = W.crash("tiny", out, data=data, question=QUESTION, only=ONLY,
                         log=stop_after_three)
    finally:
        mp.undo()
    return {"report": report, "out": out, "first": first, "presses": presses, "data": data}


@pytest.mark.slow
def test_the_plan_is_scored_on_five_by_five_random_folds_within_its_training_rows(swept):
    from turbotab.core.graph import read_artifact

    report, out = swept["report"], swept["out"]
    plan = report["baseline"]
    assert report["scoring"]["set"] is True and "why" not in report["scoring"]
    assert plan["split"] == {"n_train": 240, "folds": 5, "repeats": 5, "fold_scheme": "random",
                             "validation": "repeated_kfold", "grouped_by": None}
    for family, scores in plan["scores"]["families"].items():
        primary = scores[plan["scores"]["metric"]]
        assert len(primary["folds"]) == 25 and primary["repeats"] == 5, family
    # The folds themselves, from the split the engine cached for the template.
    template = report["template"]["project"]
    cache = out / "home" / "projects" / template / "cache"
    with W.arms.Local(out / "home") as session:
        session.use(template)
        key = session.view()["stages"]["split"]["key"]
    assignment = read_artifact(cache, "split", key).frames["assignment"]
    train = assignment[assignment["partition"] == "train"]
    assert len(train) == 240
    repeats = [c for c in assignment.columns if c == "fold" or c.startswith("fold_r")]
    assert len(repeats) == 5
    for c in repeats:
        assert sorted(train[c].value_counts().tolist()) == [48] * 5, c
    assert len({tuple(train[c].tolist()) for c in repeats}) == 5  # five draws, not one


@pytest.mark.slow
def test_a_refit_that_holds_rows_out_is_scored_on_the_same_scheme_within_its_own(swept):
    refits = {r["id"]: r for r in swept["report"]["refits"]}
    fifth = refits["question:split=0.2"]
    assert fifth["kind"] == "scored", fifth
    assert fifth["split"] == {"n_train": 192, "folds": 5, "repeats": 5, "fold_scheme": "random",
                              "validation": "repeated_kfold", "grouped_by": None}
    assert all(c["paired"] is False for c in fifth["change"])  # other training rows: unpaired


@pytest.mark.slow
def test_the_plans_own_answer_is_deduplicated_by_its_digest(swept):
    report = swept["report"]
    refits = {r["id"]: r for r in report["refits"]}
    plan = {p["id"]: p for p in report["points"]}
    own_missing = plan["question:missing"]["decision"]["strategy"]
    own_split = format(float(plan["question:split"]["decision"]["holdout"]), "g")
    for rid in (f"question:missing={own_missing}", f"question:split={own_split}"):
        assert refits[rid]["kind"] == "duplicate" and refits[rid]["duplicate_of"] == "plan", rid
        assert refits[rid]["digest"] == report["baseline"]["digest"]
    digests = [r["digest"] for r in report["refits"] if r["kind"] == "scored"]
    assert len(digests) == len(set(digests)) and report["baseline"]["digest"] not in digests


@pytest.mark.slow
def test_a_refusal_is_kept_with_its_code_its_words_and_the_option_refused(swept):
    refits = {r["id"]: r for r in swept["report"]["refits"]}
    mi = refits["question:missing=multiple_imputation"]
    assert mi["kind"] == "refused"
    assert mi["refused"]["at"] == "answer" and mi["refused"]["by"] == "server"
    assert mi["refused"]["code"] == "imputation_with_the_outcome" and mi["refused"]["message"]
    assert mi["refused"]["option"] == "multiple_imputation"
    # Every other refusal, whichever fires (a solver can tip one): typed, at the fit or the answer.
    for r in swept["report"]["refits"]:
        if r["kind"] == "refused":
            assert r["refused"]["code"] and r["refused"]["message"] and r["refused"]["option"]
    decision = next(d for d in swept["report"]["decisions"] if d["point"] == "question:missing")
    counted = sum(decision["classes"][k] for k in ("scored", "refused", "failed"))
    assert decision["refused_share"] == pytest.approx(decision["classes"]["refused"] / counted)
    assert decision["refused_codes"]["imputation_with_the_outcome"] == 1


@pytest.mark.slow
def test_an_http_500_is_a_failed_refit_reported_as_an_engine_defect_with_its_reproduction(swept):
    report = swept["report"]
    refits = {r["id"]: r for r in report["refits"]}
    crashed = refits[f"declaration:set_selection=set_selection:{CRASHES_ON}"]
    assert crashed["kind"] == "failed" and crashed["failed"]["code"] == "http_500"
    assert [r["id"] for r in report["refits"] if r["kind"] == "failed"] == [crashed["id"]]
    (defect,) = report["defects"]
    assert defect["refit"] == crashed["id"] and report["passed"] is False
    kept = json.loads(open(defect["path"]).read())
    assert kept["requests"][0]["status"] == 500 and kept["requests"][0]["path"].endswith(
        "/preview")
    assert json.loads(kept["requests"][0]["body"])["method"] == CRASHES_ON
    kinds = [e["decision"]["kind"] for e in kept["log"]]
    assert "set_validation" in kinds and "set_target" in kinds
    assert (swept["out"] / "home" / "projects" / crashed["project"]).is_dir()  # the copy kept
    others = [r for r in report["refits"] if r["kind"] != "failed" and r.get("project")]
    assert others == []  # every other copy was removed


@pytest.mark.slow
def test_every_option_offered_is_listed_and_visited(swept):
    """The options listed are what the engine serves for each decision, read here from the
    artifacts the driver composes them from (the missing-values card's labeled options; the seal
    plan's holdouts and a cut-point of one's own; the selection line's composition)."""
    from research.swarm import declarations as decl, driver

    report, out = swept["report"], swept["out"]
    offered = {o["point"]: [x["id"] for x in o["options"]] for o in report["offered"]}
    with W.arms.Local(out / "home") as session:
        session.use(report["template"]["project"])
        proposals = session.read("stages/proposals")["artifact"]
        plan = session.read("stages/seal_plan")["artifact"]
        composed = decl.compose("set_selection", session.view(),
                                driver.served(session.run, session.client))
    labeled = [o["key"] for o in proposals["labels"]["missing"]["options"]]
    assert offered["question:missing"] == labeled
    holdouts = [format(float(o["holdout"]), "g") for o in plan["options"]]
    assert offered["question:split"] == [*holdouts, "cut"]
    assert offered["declaration:set_selection"] == [o["id"] for o in composed["options"]]
    ids = {r["id"] for r in report["refits"]}
    for point, options in offered.items():
        assert {f"{point}={o}" for o in options} <= ids, point
    assert report["complete"] is True and report["unvisited"] == []
    cut = next(r for r in report["refits"] if r["id"] == "question:split=cut")
    assert cut["kind"] == "unreached" and "records nothing by itself" in cut["why"]
    # No card states its held-out rows, and no cut-point is built: unreached, with why, counted.
    assert "the card states no held-out cut-point" in cut["why"]
    assert {"refit": cut["id"], "why": cut["why"]} in report["unreached"]


@pytest.mark.slow
def test_a_paired_refits_materiality_is_recomputed_by_numpy_from_the_served_fold_scores(swept):
    report = swept["report"]
    plan = report["baseline"]
    metric = plan["scores"]["metric"]
    share = 48 / 192
    checked = 0
    for r in report["refits"]:
        for c in r["change"]:
            if not c.get("paired") or c.get("difference") is None:
                continue
            after = np.asarray(r["scores"]["families"][c["family"]][metric]["folds"], float)
            before = np.asarray(plan["scores"]["families"][c["family"]][metric]["folds"], float)
            d = after - before
            se = math.sqrt((1 / d.size + share) * d.var(ddof=1))
            q = stats.t.ppf(0.975, d.size - 1)
            assert (c["difference"], c["ci_low"], c["ci_high"]) == pytest.approx(
                (d.mean(), d.mean() - q * se, d.mean() + q * se), rel=1e-9, abs=1e-12)
            served = plan["scores"]["families"][c["family"]][metric]
            half = (served["ci_high"] - served["ci_low"]) / 2
            assert c["half_width"] == pytest.approx(half)
            assert c["material"] is bool((c["ci_low"] > 0 or c["ci_high"] < 0)
                                         and abs(c["difference"]) >= 0.5 * half)
            checked += 1
    assert checked >= 2


@pytest.mark.slow
def test_a_second_run_after_an_interruption_refits_nothing_the_first_finished(swept):
    first = [e["refit"]["id"] for e in swept["first"] if "refit" in e]
    report = swept["report"]
    assert "plan" in first and len(first) == 4  # the plan's own and three options
    entries = [json.loads(x) for x in (swept["out"] / "progress.jsonl").read_text().splitlines()]
    written = [e["refit"]["id"] for e in entries if "refit" in e]
    assert len(written) == len(set(written))  # each refit written once, in either run
    assert len([e for e in entries if "template" in e]) == 1  # the plan built once
    # A refit pressed Fit exactly where it kept the press's wait (``Refit.seconds``): not for an
    # answer refused, a duplicate, or an option that records nothing by itself.
    every = {r["id"]: r for r in [report["baseline"], *report["refits"]]}
    pressed_first = [rid for rid in first if every[rid]["seconds"] is not None]
    pressed_later = [r for rid, r in every.items() if rid not in first and r["seconds"] is not None]
    assert swept["presses"]["first"] == len(pressed_first) >= 2
    assert swept["presses"]["second"] == len(pressed_later)


@pytest.mark.slow
def test_each_fits_run_time_is_summed_into_the_report(swept):
    report = swept["report"]
    timed = [W.timed_of(W.Refit.of(r)) for r in [report["baseline"], *report["refits"]]]
    timed = [t for t in timed if t is not None]
    assert report["fit_seconds"] == W.seconds_of(timed) and report["fit_seconds"]["fits"] >= 5
    assert (swept["out"] / "crash.json").exists() and (swept["out"] / "crash.md").exists()


def dated_table(path):
    """The small table with a ``cycle`` column: 60 rows in each of four cycles, 2011 to 2017."""
    rng = np.random.default_rng(2)
    n = 240
    frame = pd.DataFrame({f"x{i}": rng.normal(size=n).round(3) for i in (1, 2)})
    frame["cycle"] = np.repeat([2011, 2013, 2015, 2017], 60)
    p = 1 / (1 + np.exp(-(0.8 * frame.x1)))
    frame["y"] = (rng.random(n) < p).astype(int)
    frame.to_csv(path, index=False)
    return frame


@pytest.mark.slow
def test_under_a_cut_point_the_folds_order_is_answered_random_and_scored_five_by_five(tmp_path):
    """SWARM_PLAN §10's watch note on package 3: a cut-point seal leaves the folds' order free, so
    the sweep's scoring answers it random; the plan's training rows are those before the cut
    (pandas: 180 rows before 2017), each in one fold of each of five repeats."""
    frame = dated_table(tmp_path / "dated.csv")
    policy = arms.Default(QUESTION)

    def build(session):
        pid = session.open(tmp_path / "dated.csv")
        arms.walk(session, policy)
        split = next(r for r in W.live(session.view()) if r.decision.kind == "set_split")
        cut = {**split.decision.model_dump(mode="json"), "holdout": 0.0,
               "cut_point": {"column": "cycle", "at": 2017}, "fold_order": "time",
               "validation": "kfold"}
        said = session.decide(cut, "A cut-point at 2017, its folds in time order.")
        assert said.ok, said.payload
        arms.walk(session, policy)
        return pid

    sweep = W.Sweep(tmp_path / "out", dataset="dated", question=QUESTION, crash=False,
                    plan="cut", only=[])
    with arms.Local(tmp_path / "out" / "home") as session:
        report = sweep.run(session, build)
    assert report["scoring"]["set"] is True and "why" not in report["scoring"]
    assert report["scoring"]["decision"]["fold_order"] == "random"
    assert int((frame["cycle"] < 2017).sum()) == 180
    assert report["baseline"]["split"] == {"n_train": 180, "folds": 5, "repeats": 5,
                                           "fold_scheme": "random",
                                           "validation": "repeated_kfold", "grouped_by": None}


@pytest.mark.slow
def test_a_named_plan_is_rebuilt_from_its_bundle_cut_before_the_opening(tmp_path):
    """A plan that opened its held-out rows, exported, is swept from its log cut before the
    opening (``replay.truncate``): the template holds every record before it and no opening, and
    its refits are scored as the default plan's are."""
    import time as _time

    data = table(tmp_path / "tiny.csv")
    policy = arms.Default(QUESTION)
    with arms.Local(tmp_path / "made") as s:
        s.open(data)
        arms.walk(s, policy)
        assert s.fit().ok
        q = arms.walk(s, policy)
        family = arms.families_of(q, s.view())[0]
        assert arms.seal_through(s, policy, family, []).ok
        recorded = [r["decision"]["kind"] for r in s.view()["decisions"]]
        end = _time.monotonic() + 300
        while True:
            r = s.client.get(f"/projects/{s.pid}/export")
            if r.status == 200 or _time.monotonic() > end:
                break
            _time.sleep(1.0)
        assert r.status == 200, r.body[:400]
    bundle = tmp_path / "named.zip"
    bundle.write_bytes(r.body)
    out = tmp_path / "out"
    report = W.sweep_plan("tiny", bundle, "named", out, data=data, question=QUESTION,
                          only=["question:missing"])
    opening = recorded.index("open_seal") + 1  # its seq
    cut = json.loads((out / W.REPLAYED).read_text())["cut"]
    assert cut["kind"] == "open_seal" and cut["seq"] == opening and cut["kept"] == opening - 1
    with W.arms.Local(out / "home") as session:
        session.use(report["template"]["project"])
        kinds = [d["decision"]["kind"] for d in session.view()["decisions"]]
    assert "open_seal" not in kinds and kinds[:opening - 1] == recorded[:opening - 1]
    assert kinds[opening - 1:] == ["set_validation"]  # the sweep's scoring, then nothing
    assert report["baseline"]["split"]["repeats"] == 5 and report["complete"] is True
    assert {r["id"].split("=")[0] for r in report["refits"]} == {"question:missing"}
    assert (out / "sweep.json").exists()


# ── a stage error induced in the engine after an option ──────────────────────

CRASHES_UNDER = "complete_case"  # the missing-values option under which the cohort fails, here
CRASHED = "RuntimeError: the cohort fails here, as a defect would"
REFUSED_AT = 0.3  # the held-out share whose draw the split refuses in words, here
REFUSED_WORDS = "The draw refuses this share here, in words."


@pytest.fixture(scope="module")
def induced(tmp_path_factory):
    """The crash sweep of the missing values and the held-out share, with the engine recording a
    stage in error where one option is in force: the cohort failed with an error it did not type
    under complete cases, and the split's draw refused in words (``graph.StageRefusal``, its code,
    words and a way forward) under a share of 0.3. Each is recorded where the engine records a
    stage it could not start (``Engine._start``), since the stages run in worker processes a test
    cannot patch."""
    from turbotab.core import graph
    from turbotab.core.decisions import missing_strategy
    from turbotab.core.jobs import failure_of

    work = tmp_path_factory.mktemp("induced")
    data = table(work / "tiny.csv")
    original = graph.Engine._start

    def start(self, p, stage, key):
        state = p.ctx.state
        if stage.name == "cohort" and missing_strategy(state) == CRASHES_UNDER:
            p.errors[(stage.name, key)] = CRASHED
            return None
        split = getattr(state, "split", None)
        if stage.name == "split" and split is not None and float(split.holdout) == REFUSED_AT:
            error, refusal = failure_of(graph.StageRefusal(
                "induced_refusal", REFUSED_WORDS,
                exits=[{"label": "Hold out a fifth", "decision": {"kind": "set_split",
                                                                  "holdout": 0.2}}]))
            p.errors[(stage.name, key)] = error
            p.refusals[(stage.name, key)] = {"stage": stage.name, **refusal}
            return None
        return original(self, p, stage, key)

    mp = pytest.MonkeyPatch()
    mp.setattr(graph.Engine, "_start", start)
    try:
        report = W.crash("tiny", work / "out", data=data, question=QUESTION,
                         only=["question:missing", "question:split"])
    finally:
        mp.undo()
    return {"report": report, "out": work / "out", "data": data}


@pytest.mark.slow
def test_a_stage_that_fails_untyped_after_an_option_is_a_failed_refit_and_an_engine_defect(
        induced):
    report = induced["report"]
    plan = {p["id"]: p for p in report["points"]}
    assert plan["question:missing"]["decision"]["strategy"] != CRASHES_UNDER  # an option, not
    refits = {r["id"]: r for r in report["refits"]}  # the plan's own answer
    crashed = refits[f"question:missing={CRASHES_UNDER}"]
    assert crashed["kind"] == "failed", crashed
    assert crashed["failed"]["option"] == CRASHES_UNDER, crashed["failed"]
    (defect,) = report["defects"]
    assert defect["refit"] == crashed["id"] and report["passed"] is False
    kept = json.loads(open(defect["path"]).read())
    cohort = next(s for s in kept["stages"] if s["stage"] == "cohort")
    assert cohort["kind"] == "failed" and cohort["error"] == CRASHED
    assert kept["data"] == str(induced["data"])
    assert {"kind": "set_missing", "strategy": CRASHES_UNDER}.items() <= next(
        e["decision"] for e in reversed(kept["log"])
        if e["decision"]["kind"] == "set_missing").items()
    assert (induced["out"] / "home" / "projects" / crashed["project"]).is_dir()


@pytest.mark.slow
def test_a_stage_that_refuses_in_words_after_an_option_is_a_refused_refit(induced):
    report = induced["report"]
    refits = {r["id"]: r for r in report["refits"]}
    refused = refits[f"question:split={REFUSED_AT:g}"]
    assert refused["kind"] == "refused", refused
    r = refused["refused"]
    assert (r["by"], r["stage"], r["code"], r["message"]) == (
        "engine", "split", "induced_refusal", REFUSED_WORDS)
    assert r["option"] == f"{REFUSED_AT:g}"
    assert [e["label"] for e in r["exits"]] == ["Hold out a fifth"]
    assert r["exits"][0]["decision"]["holdout"] == 0.2
    split = next(d for d in report["decisions"] if d["point"] == "question:split")
    assert split["refused_codes"] == {"induced_refusal": 1}


@pytest.mark.slow
def test_no_other_option_is_failed_or_refused_at_a_stage(induced):
    """Both ways: under every option no stage was made to fail under, the refit is not failed and
    not refused at a stage or the fit (a refusal the server gives at the answer, as multiple
    imputation's, is the answer's), and no other defect is said."""
    report = induced["report"]
    refits = {r["id"]: r for r in report["refits"]}
    induced_ids = {f"question:missing={CRASHES_UNDER}", f"question:split={REFUSED_AT:g}"}
    others = {rid: r for rid, r in refits.items() if rid not in induced_ids}
    assert len(others) >= 6, sorted(others)
    for rid, r in others.items():
        assert r["kind"] != "failed", rid
        if r["kind"] == "refused":
            assert r["refused"]["at"] == "answer" and r["refused"]["by"] == "server", rid
    assert others["question:missing=multiple_imputation"]["kind"] == "refused"
    assert others["question:split=cut"]["kind"] == "unreached"  # it takes a field of one's own
    assert [d["refit"] for d in report["defects"]] == [f"question:missing={CRASHES_UNDER}"]
    assert report["only"] == ["question:missing", "question:split"]


# ── a reading the models answer reads (the verifier's V2-1) ──────────────────

READING = "reading:code_or_count:stage"


def coded_table(path):
    """The verifier's case: 240 rows, three numeric predictors, ``weight_kg`` with twelve values
    of 999, ``stage`` an integer from 1 to 4 (the default path reads it as a code), and a yes/no
    outcome from a logistic model in two of the predictors."""
    rng = np.random.default_rng(3)
    n = 240
    frame = pd.DataFrame({f"x{i}": rng.normal(size=n).round(3) for i in (1, 2, 3)})
    weight = rng.normal(75, 12, n).round(1)
    weight[rng.choice(n, 12, replace=False)] = 999
    frame["weight_kg"] = weight
    frame["stage"] = rng.integers(1, 5, n)
    p = 1 / (1 + np.exp(-(0.8 * frame.x1 - 0.5 * frame.x2)))
    frame["y"] = (rng.random(n) < p).astype(int)
    frame.to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def coded(tmp_path_factory):
    """The crash sweep of the stage reading on the coded table, as the engine is, and again with
    the server's preview of the reading's other answer (an amount) failing as a defect would."""
    from turbotab.server.service import ProjectService

    work = tmp_path_factory.mktemp("coded")
    data = coded_table(work / "coded.csv")
    clean = W.crash("coded", work / "clean", data=data, question=QUESTION, only=[READING])
    original = ProjectService._preview

    def failing(self, pid, decision):
        body = decision.model_dump() if hasattr(decision, "model_dump") else dict(decision)
        if body.get("kind") == "confirm_reading" and body.get("column") == "stage" \
                and body.get("value") == "amount":
            raise RuntimeError("the preview fails here, as a defect would")
        return original(self, pid, decision)

    mp = pytest.MonkeyPatch()
    mp.setattr(ProjectService, "_preview", failing)
    try:
        induced = W.crash("coded", work / "induced", data=data, question=QUESTION,
                          only=[READING])
    finally:
        mp.undo()
    with W.arms.Local(work / "clean" / "home") as session:
        session.use(clean["template"]["project"])
        log = session.view()["decisions"]
    return {"clean": clean, "induced": induced, "log": log}


def by_value(report):
    """Each refit of the stage reading, by the answer it gives."""
    return {r["decision"]["value"]: r for r in report["refits"]
            if r["point"] == READING and r["decision"]}


@pytest.mark.slow
def test_a_reading_the_models_answer_reads_is_swept_and_the_crash_sweep_passes(coded):
    """The models answer withdrawn first, as the engine's refusal asks, each of the reading's
    two answers (its ways: a code, an amount) answered and the models answer given again as the
    plan had it; the reading visited, every option, and no defect: the sweep passes but for the one
    decision it was told to sweep."""
    from turbotab.core.readings import VALUES

    report = coded["clean"]
    models = [r for r in coded["log"] if r["decision"]["kind"] == "select_models"]
    reading = [r for r in coded["log"] if r["decision"]["kind"] == "confirm_reading"
               and r["decision"].get("column") == "stage"]
    assert len(models) == 1 and len(reading) == 1 and reading[0]["seq"] < models[0]["seq"]
    assert reading[0]["decision"]["value"] == "code"  # the default path's reading
    (offered,) = report["offered"]
    assert sorted(o["decision"]["value"] for o in offered["options"]) == sorted(VALUES[
        "code_or_count"])  # every answer the reading has (``readings.VALUES``)
    assert offered["withdrawn"] == [{"what": "models answer", "column": "stage",
                                     "seq": models[0]["seq"], "record": models[0]["id"]}]
    assert report["unlisted"] == [] and report["complete"] is True and report["defects"] == []
    assert report["only"] == [READING] and W.passed(report) is False  # some decisions only
    assert W.passed(report | {"only": None}) is True
    refits = by_value(report)
    for value, r in refits.items():
        assert [w["decision"] for w in r["withdrawn"]] == [models[0]["decision"]], value
        again = next(a for a in r["answered"] if a["answer"].startswith("the plan's models"))
        assert again["kind"] == "ok", (value, again)
    # The plan's own reading, answered again with the models answer after it: the plan itself.
    assert refits["code"]["kind"] == "duplicate" and refits["code"]["duplicate_of"] == "plan"
    # The other: fitted, or refused by the engine in words (a solver can tip it), never failed.
    assert refits["amount"]["kind"] in ("scored", "refused"), refits["amount"]
    if refits["amount"]["kind"] == "refused":
        assert refits["amount"]["refused"]["code"] and refits["amount"]["refused"]["message"]


@pytest.mark.slow
def test_an_induced_500_on_the_readings_other_answer_still_fails_the_crash_sweep(coded):
    report = coded["induced"]
    refits = by_value(report)
    amount = refits["amount"]
    assert amount["kind"] == "failed" and amount["failed"]["code"] == "http_500"
    assert [d["refit"] for d in report["defects"]] == [amount["id"]]
    assert report["complete"] is True and W.passed(report | {"only": None}) is False
    kept = json.loads(open(report["defects"][0]["path"]).read())
    assert json.loads(kept["requests"][0]["body"])["value"] == "amount"
    assert refits["code"]["kind"] == "duplicate"


# ── the options that take fields of their own ────────────────────────────────
# Each answered by the crash sweep with a value it built from the table or read from the card,
# previewed, answered and fitted on the engine; the rows each kept read from the participant flow
# the refit served and checked against a pandas filter of the same codes over the table written
# here, never against the engine; the card's codes checked against the card's own sentence.

def level(value):
    """A level in the one spelling the flow compares (``2``, ``2.0`` and ``"2"`` are one)."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def passes(frame, rule):
    """The rows ``rule`` keeps, by pandas over the table's codes as the rule models document them
    (``decisions.ExclusionRule``, ``LevelRule``): a range keeps a value from its low to its high; a
    level rule every level but those it excludes; a blank (for a level rule also a refused or a
    don't-know code) stays only where the rule keeps what it cannot confirm."""
    column = frame[rule["column"]]
    blank = column.isna()
    if rule["kind"] == "range":
        values = pd.to_numeric(column, errors="coerce")
        inside = ~blank
        if rule.get("low") is not None:
            inside &= (values >= rule["low"]).fillna(False)
        if rule.get("high") is not None:
            inside &= (values <= rule["high"]).fillna(False)
        return inside | blank if rule["missing"] == "keep" else inside
    spelled = column.map(lambda v: None if pd.isna(v) else level(v))
    named = lambda levels: spelled.isin([str(v) for v in levels])  # noqa: E731
    if rule["missing"] == "keep":
        return ~named(rule["exclude"])
    return ~named(rule["exclude"]) & ~blank & ~named(rule["refused"]) & ~named(rule["dont_know"])


def flow_by_pandas(frame, outcome, rules):
    """The rows with the outcome recorded, then those left after each rule in its order."""
    kept = frame[outcome].notna()
    out = [int(kept.sum())]
    for rule in rules:
        kept &= passes(frame, rule)
        out.append(int(kept.sum()))
    return out


def flow_served(refit):
    """The same counts from the participant flow the refit served (``Refit.flow``)."""
    steps = {s["key"]: s["n"] for s in refit["flow"]}
    rules = refit["decision"]["rules"]
    return [steps["outcome_measured"], *(steps[f"exclusion:{i}"] for i in range(len(rules)))]


def fielded_table(path):
    """240 rows: ``x1`` numeric with six blanks, ``x2`` numeric, ``grp`` a text predictor whose
    levels are a (120 rows), b (96) and c (24, the rarest), ``age`` in whole years from 12 to 85,
    and a yes/no outcome from a logistic model in x1 and x2."""
    rng = np.random.default_rng(4)
    n = 240
    frame = pd.DataFrame({"x1": rng.normal(size=n).round(3), "x2": rng.normal(size=n).round(3)})
    frame["grp"] = rng.permutation(["a"] * 120 + ["b"] * 96 + ["c"] * 24)
    frame["age"] = rng.integers(12, 86, n)
    p = 1 / (1 + np.exp(-(0.8 * frame.x1 - 0.5 * frame.x2)))
    frame["y"] = (rng.random(n) < p).astype(int)
    frame.loc[rng.choice(n, 6, replace=False), "x1"] = np.nan
    frame.to_csv(path, index=False)
    return frame


EXCLUSIONS = "question:exclusions"
BEFORE = "It records nothing by itself: it takes a field of its own."  # the sweep's words before
LATE = 3.0  # seconds the server's recorder waits before it writes a fit's record, here


def late_records(mp, seconds):
    """Each fit's record written ``seconds`` after its key turns fresh: the server writes it on
    its recorder's thread (``service._fit_event``), here that much later than it would."""
    import time

    from turbotab.server.service import ProjectService

    original = ProjectService._record_fit

    def late(self, pid, key, at, spent):
        time.sleep(seconds)
        return original(self, pid, key, at, spent)

    mp.setattr(ProjectService, "_record_fit", late)


@pytest.fixture(scope="module")
def fielded(tmp_path_factory):
    """The crash sweep of the exclusions on the constructed table, twice into one folder: first as
    it ran before it answered a field of its own (each such option unreached in its words of
    then), a finished report; then as it runs now, the presses of Fit each run made counted, and
    each fit's record written :data:`LATE` seconds after the fit turned fresh."""
    work = tmp_path_factory.mktemp("fielded")
    frame = fielded_table(work / "fielded.csv")
    out = work / "out"
    presses = {"before": 0, "now": 0}
    run = {"now": "before"}
    original_fit = arms.Local.fit

    def counted(self, wait=3600.0):
        presses[run["now"]] += 1
        return original_fit(self, wait)

    mp = pytest.MonkeyPatch()
    mp.setattr(arms.Local, "fit", counted)
    mp.setattr(W.Fields, "answers", lambda self, point, option: ([], BEFORE))
    mp.setattr(W.Fields, "population", lambda self, point, offered: [])
    try:
        before = W.crash("fielded", out, data=work / "fielded.csv", question=QUESTION,
                         only=[EXCLUSIONS])
        first = [json.loads(x) for x in (out / "progress.jsonl").read_text().splitlines()]
        mp.undo()
        mp.setattr(arms.Local, "fit", counted)
        late_records(mp, LATE)
        run["now"] = "now"
        report = W.crash("fielded", out, data=work / "fielded.csv", question=QUESTION,
                         only=[EXCLUSIONS])
    finally:
        mp.undo()
    return {"before": before, "first": first, "report": report, "out": out, "frame": frame,
            "presses": presses}


def built(report):
    return {r["option"].split("@")[0]: r for r in report["refits"]
            if (r.get("answered_with") or {}).get("source") == "built"}


@pytest.mark.slow
def test_a_value_built_from_the_table_is_labeled_built_and_keeps_the_rows_pandas_keeps(fielded):
    frame, refits = fielded["frame"], built(fielded["report"])
    assert set(refits) == {"custom", "eligibility", "levels"}
    low, high = np.percentile(frame["x1"].dropna(), [1, 99])
    expected = {  # by construction: x1 the first numeric predictor, c the rarest level of grp,
        # and an adult 20 or older (the CDC 2000 growth charts end at 20 years)
        "custom": {"kind": "range", "column": "x1", "low": low, "high": high},
        "eligibility": {"kind": "range", "column": "age", "low": 20.0, "high": None},
        "levels": {"kind": "levels", "column": "grp", "exclude": ["c"], "refused": [],
                   "dont_know": []}}
    for oid, r in refits.items():
        # Each fit's record came LATE seconds after its key turned fresh: a fit is still scored.
        assert r["kind"] in ("scored", "refused"), (oid, r.get("why") or r.get("refused")
                                                    or r.get("failed"))
        assert (r["fit_record"] is not None) == (r["kind"] == "scored"), oid
        (rule,) = r["decision"]["rules"]
        for k, v in expected[oid].items():
            assert rule[k] == (pytest.approx(v, abs=1e-12) if isinstance(v, float) else v), oid
        assert r["answered_with"]["source"] == "built" and "built" in r["answered_with"]["words"]
        assert r["flow"] is not None, oid
        assert flow_served(r) == flow_by_pandas(frame, "y", [rule]), oid
        assert r["previews"][0]["of"] == r["id"] and r["previews"][0]["kind"] == "ok"
    assert any(r["kind"] == "scored" for r in refits.values())
    assert fielded["report"]["defects"] == [] and fielded["report"]["unreached"] == []


@pytest.mark.slow
def test_a_resumed_crash_sweep_on_a_finished_report_fits_only_the_new_options(fielded):
    before, report = fielded["before"], fielded["report"]
    assert {u["refit"] for u in before["unreached"]} == {
        f"{EXCLUSIONS}={o}" for o in ("custom", "eligibility", "levels")}
    assert all(u["why"] == BEFORE for u in before["unreached"]) and before["complete"] is True
    assert f"; unreached: 3; swept only {EXCLUSIONS}." in W.table_md(before)
    # The second run fitted each new option and nothing the first had finished.
    new = [r for r in report["refits"] if r.get("answered_with")]
    assert len(new) == 3
    assert fielded["presses"]["now"] == sum(1 for r in new if r["seconds"] is not None) >= 1
    first = {e["refit"]["id"] for e in fielded["first"] if "refit" in e}
    entries = [json.loads(x) for x in (fielded["out"] / "progress.jsonl").read_text().splitlines()]
    later = [e["refit"]["id"] for e in entries[len(fielded["first"]):] if "refit" in e]
    assert sorted(later) == sorted(r["id"] for r in new) and not set(later) & first
    # The options unreached before are now the refits that answered them, and none is unreached.
    ids = {r["id"] for r in report["refits"]}
    assert not ids & {u["refit"] for u in before["unreached"]}
    assert report["unreached"] == [] and report["complete"] is True
    md = (fielded["out"] / "crash.md").read_text()
    assert f"; unreached: 0; swept only {EXCLUSIONS}." in md


def bang_table(path):
    """900 people aged 12 to 85, coded as NHANES codes them, with the skip patterns behind the
    blanks: RIDEXPRG asked of women aged 20 to 44 only, so blank for every man; DIQ070 asked only
    of a person told of diabetes or borderline (DIQ010 1 or 3); DIQ050 blank on about one row in
    twelve; ``diabetes_fpg`` (the card's outcome) from a logistic model in age and BMI, blank on
    about a third of the rows."""
    rng = np.random.default_rng(2009)
    n = 900
    age = rng.integers(12, 86, n)
    male = rng.random(n) < 0.49
    diq010 = rng.choice([1, 2, 3, 7, 9], n, p=[0.10, 0.76, 0.06, 0.04, 0.04]).astype(float)
    diq010[rng.random(n) < 0.03] = np.nan
    told = np.isin(diq010, [1, 3])
    asked = ~male & (age >= 20) & (age <= 44)
    frame = pd.DataFrame({
        "SEQN": np.arange(1, n + 1), "RIDAGEYR": age, "RIAGENDR": np.where(male, 1, 2),
        "RIDEXPRG": np.where(asked, rng.choice([1, 2, 3], n, p=[0.12, 0.80, 0.08]), np.nan),
        "DIQ010": diq010,
        "DIQ050": rng.choice([1, 2, 7, 9], n, p=[0.05, 0.89, 0.03, 0.03]).astype(float),
        "DIQ070": np.where(told, rng.choice([1, 2, 7, 9], n, p=[0.5, 0.4, 0.05, 0.05]), np.nan),
        "BMXBMI": np.round(rng.normal(28, 5, n), 1)})
    frame.loc[rng.random(n) < 0.08, "DIQ050"] = np.nan
    p = 1 / (1 + np.exp(-(-1.2 + 0.04 * (age - 50) + 0.12 * (frame["BMXBMI"] - 28))))
    frame["diabetes_fpg"] = (rng.random(n) < p).astype(float)
    frame.loc[rng.random(n) < 0.33, "diabetes_fpg"] = np.nan
    for c in ("RIDEXPRG", "DIQ010", "DIQ050", "DIQ070", "diabetes_fpg"):
        frame[c] = frame[c].astype("Int64")
    frame.to_csv(path, index=False)
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def bang(tmp_path_factory):
    """The crash sweep of the exclusions on a table coded as Bang's is, under the bang2009 card
    (its question and its eligibility rules, read from the card), with the server's preview of one
    card rule under one handling failing as a defect would: the card's second level rule, keeping
    what it cannot confirm."""
    from turbotab.server.service import ProjectService

    work = tmp_path_factory.mktemp("bang")
    frame = bang_table(work / "bang.csv")
    rules, _ = W.card_rules(arms.card_of("bang2009"))
    fails = [r for r in rules if r.kind == "levels"][1]
    original = ProjectService._preview

    def failing(self, pid, decision):
        body = decision.model_dump() if hasattr(decision, "model_dump") else dict(decision)
        posted = body.get("rules") or []
        if body.get("kind") == "set_exclusions" and len(posted) == 1 \
                and posted[0].get("column") == fails.column and posted[0].get("missing") == "keep":
            raise RuntimeError("the preview fails here, as a defect would")
        return original(self, pid, decision)

    mp = pytest.MonkeyPatch()
    mp.setattr(ProjectService, "_preview", failing)
    try:
        report = W.crash("bang2009", work / "out", data=work / "bang.csv", only=[EXCLUSIONS])
    finally:
        mp.undo()
    return {"report": report, "frame": frame, "rules": rules, "fails": fails,
            "card": arms.card_of("bang2009")}


def stated(card, rule):
    """Whether the card's own sentence for ``rule``'s column states the rule's codes, in the words
    the intake writes them (``intake.age_line``, ``intake.level_line``)."""
    said = next(s for s in card["eligibility"] if f"rule on {rule['column']} " in s)
    if rule["kind"] == "range":
        return f"a range rule on {rule['column']} from {rule['low']:g}." in said
    named = (f"naming {rule['refused'][0]} as its refused code and {rule['dont_know'][0]} as its "
             f"don't-know code" if rule["refused"] else "which has no refused or don't-know code")
    return f"a level rule on {rule['column']} excluding level {rule['exclude'][0]} (" in said \
        and named in said and len(rule["exclude"]) == 1


def card_refits(report):
    return [r for r in report["refits"] if (r.get("answered_with") or {}).get("source") == "card"]


@pytest.mark.slow
def test_each_card_rule_is_answered_and_fitted_and_keeps_the_rows_pandas_keeps(bang):
    """Each of the card's rules alone, the age range once and each level rule under each handling,
    is previewed, answered and fitted: refused only in the engine's words, and the rows it keeps,
    read from the flow, those a pandas filter of the card's codes keeps."""
    report, frame, card = bang["report"], bang["frame"], bang["card"]
    alone = [r for r in card_refits(report) if not r["option"].startswith("population@")]
    failed = [r for r in alone if r["kind"] == "failed"]
    assert len(alone) == 1 + 2 * 4 and len(failed) == 1
    for r in alone:
        if r["kind"] == "failed":
            continue
        assert r["kind"] in ("scored", "refused"), (r["id"], r.get("refused"))
        (rule,) = r["decision"]["rules"]
        assert stated(card, rule), r["id"]
        assert r["flow"] is not None, r["id"]
        assert flow_served(r) == flow_by_pandas(frame, "diabetes_fpg", [rule]), r["id"]
    assert sum(r["kind"] == "scored" for r in alone) >= 4
    assert {r["decision"]["rules"][0]["missing"] for r in alone if r["decision"]
            and r["decision"]["rules"][0]["kind"] == "levels"} == {"exclude", "keep"}


@pytest.mark.slow
def test_the_cards_whole_population_is_one_answer_of_its_rules_in_its_order(bang):
    report, frame, card, rules = bang["report"], bang["frame"], bang["card"], bang["rules"]
    whole = {r["option"]: r for r in card_refits(report) if r["option"].startswith("population@")}
    assert sorted(whole) == ["population@card:exclude", "population@card:keep"]
    for handling in ("exclude", "keep"):
        r = whole[f"population@card:{handling}"]
        assert r["kind"] in ("scored", "refused"), (r["id"], r.get("refused"))
        assert r["answered_with"]["ids"] == [
            "eligibility" if x.kind == "range" else "levels" for x in rules]
        posted = r["decision"]["rules"]
        assert [x["column"] for x in posted] == [x.column for x in rules]
        assert all(stated(card, x) for x in posted)
        assert {x["missing"] for x in posted if x["kind"] == "levels"} == {handling}
        if r["flow"] is not None:
            assert flow_served(r) == flow_by_pandas(frame, "diabetes_fpg", posted)
        else:  # the answer refused in the engine's words (the rules leave too few rows)
            assert r["kind"] == "refused" and r["refused"]["at"] == "answer", r["refused"]
            assert r["refused"]["code"] and r["refused"]["message"], r["refused"]
    assert whole["population@card:keep"]["flow"] is not None


@pytest.mark.slow
def test_an_induced_500_on_a_card_rule_is_a_failed_refit_and_an_engine_defect(bang):
    report, fails = bang["report"], bang["fails"]
    rid = f"{EXCLUSIONS}=levels@card:{fails.place}:{fails.column}:keep"
    refits = {r["id"]: r for r in report["refits"]}
    assert refits[rid]["kind"] == "failed" and refits[rid]["failed"]["code"] == "http_500"
    assert [d["refit"] for d in report["defects"]] == [rid] and report["passed"] is False
    assert [r["id"] for r in report["refits"] if r["kind"] == "failed"] == [rid]
    kept = json.loads(open(report["defects"][0]["path"]).read())
    (rule,) = json.loads(kept["requests"][0]["body"])["rules"]
    assert (rule["column"], rule["missing"]) == (fails.column, "keep")


@pytest.mark.slow
def test_on_bangs_table_the_range_of_ones_own_is_built_and_said_so(bang):
    report, frame = bang["report"], bang["frame"]
    (r,) = [r for r in report["refits"] if r["option"].startswith("custom@")]
    assert r["answered_with"]["source"] == "built" and r["option"] == "custom@built:RIDAGEYR"
    (rule,) = r["decision"]["rules"]
    low, high = np.percentile(frame["RIDAGEYR"].dropna(), [1, 99])
    assert (rule["low"], rule["high"]) == pytest.approx((low, high), abs=1e-12)
    assert r["kind"] in ("scored", "refused") and r["flow"] is not None
    assert flow_served(r) == flow_by_pandas(frame, "diabetes_fpg", [rule])
    md = W.table_md(report)
    assert "## Options answered with values the crash sweep supplied" in md
    assert f"; unreached: {len(report['unreached'])}; swept only {EXCLUSIONS}." in md


def walking_table(path):
    """300 people: age and height recorded at enrollment, chronic pain (1 yes, 0 no; blank on 12)
    and daily walking (1 yes, 0 no), a health-concern score and systolic blood pressure measured
    later, from a linear model in walking, age and height."""
    rng = np.random.default_rng(5)
    n = 300
    age = rng.integers(25, 80, n)
    height = rng.normal(170, 9, n).round(1)
    pain = (rng.random(n) < 0.3).astype(float)
    walks = (rng.random(n) < 1 / (1 + np.exp(-(1.0 - 0.03 * (age - 50) - 1.2 * pain)))).astype(int)
    frame = pd.DataFrame({"person_id": np.arange(1, n + 1), "age": age, "chronic_pain": pain,
                          "walks_daily": walks, "height_cm": height})
    frame["health_concern_score"] = (2 + 1.5 * pain + rng.normal(0, 1, n)).round(2)
    frame["sbp_mmhg"] = (110 + 0.5 * (age - 50) - 4 * walks - 0.1 * (height - 170)
                         + rng.normal(0, 8, n)).round(1)
    frame.loc[rng.choice(n, 12, replace=False), "chronic_pain"] = np.nan
    frame.to_csv(path, index=False)
    return frame


@pytest.mark.slow
def test_an_adjustment_set_of_ones_own_is_the_covariates_the_roles_name(tmp_path):
    """Under Estimate the adjustment's set of one's own is answered with the covariates the plan's
    roles name, each a confounder of the declared exposure, previewed, answered and fitted."""
    walking_table(tmp_path / "walking.csv")
    question = arms.Question(target="sbp_mmhg", purpose="inference", exposure="walks_daily",
                             meaning="systolic blood pressure (mmHg)")
    report = W.crash("walking", tmp_path / "out", data=tmp_path / "walking.csv",
                     question=question, only=["question:adjustment"])
    with W.arms.Local(tmp_path / "out" / "home") as session:
        session.use(report["template"]["project"])
        roles = session.view()["state"]["roles"]
    plan = next(p for p in report["points"] if p["id"] == "question:adjustment")
    covariates = [c for c in plan["decision"]["answers"] if roles.get(c) == "covariate"]
    assert covariates  # the roles name some
    (r,) = [r for r in report["refits"] if r["option"] == "answers@built:covariates"]
    confounder = {"causes_exposure": "yes", "causes_outcome": "yes", "after_exposure": "no"}
    neither = {"causes_exposure": "no", "causes_outcome": "no", "after_exposure": "no"}
    assert r["kind"] in ("scored", "refused"), (r.get("refused") or r.get("failed"))
    assert r["decision"]["exposure"] == "walks_daily"
    assert {c: {k: a[k] for k in confounder} for c, a in r["decision"]["answers"].items()} == {
        c: confounder if c in covariates else neither for c in plan["decision"]["answers"]}
    assert r["answered_with"]["source"] == "built" and report["defects"] == []
    # The set answered is the one fitted: the walk after it answered the adjustment no more.
    assert not [a for a in r["answered"][1:] if ":adjustment=" in a["answer"]], r["answered"]
    if r["kind"] == "scored":
        assert r["digest"] != report["baseline"]["digest"]


# ── a refit's wait, and the estimate stages after a fitted refit (the panel audit's F1, N2) ──

SLOW, NEXT, TYPED_UNDER = "mixed", "gee", "featurewise"  # models answers, in the options' order
BARE_UNDER = "complete_case"  # the missing values' answer the audit met the effects stage fail on
BARE_WORDS = "the effects stage fails here untyped, as the audit met it under complete cases"
TYPED_WORDS = "The effects stage refuses these models here, in words."
WAIT = 60.0  # the refit's wait, set for the test in place of the sweep's hour
SLEEPS = 3600.0  # how long a stand-in stage sleeps: past any wait of the test's


def standin_job(spec):
    """A stage's job as the engine runs it in a worker (``graph.run_stage_job``), but for the
    stand-ins: under the models answer SLOW each estimate stage sleeps past the refit's wait and
    never reads its cancel flag (only the runner's terminate stops it); the effects stage raises a
    bare ValueError under complete cases, and the typed refusal (``graph.StageRefusal``) under
    TYPED_UNDER. A top-level function of this module, which the worker imports to run it."""
    import time

    from turbotab.core import estimand, graph
    from turbotab.core.decisions import missing_strategy

    state = spec["state"]
    models = list(getattr(state, "models", None) or [])
    if spec["stage"] in estimand.ESTIMATE_STAGES and models == [SLOW]:
        time.sleep(SLEEPS)
    if spec["stage"] == "effects" and missing_strategy(state) == BARE_UNDER:
        raise ValueError(BARE_WORDS)
    if spec["stage"] == "effects" and models == [TYPED_UNDER]:
        raise graph.StageRefusal("induced_refusal", TYPED_WORDS, exits=[
            {"label": "Choose the linear model",
             "decision": {"kind": "select_models", "models": ["linear"]}}])
    return graph.run_stage_job(spec)


@pytest.fixture(scope="module")
def waited(tmp_path_factory):
    """The crash sweep of the missing values and the models on the walking table under Estimate,
    with the refit's wait WAIT and the stand-in stages (:func:`standin_job`, in the workers), run
    twice into one folder; the presses of Fit each run made; each job the server's runner was
    given (its project, its stage, and whether it is a stand-in that sleeps: an estimate stage
    under SLOW), and the state of each as the first run's server closed, read from the runner."""
    from turbotab.core import graph
    from turbotab.core.jobs import JobRunner

    work = tmp_path_factory.mktemp("waited")
    walking_table(work / "walking.csv")
    question = arms.Question(target="sbp_mmhg", purpose="inference", exposure="walks_daily",
                             meaning="systolic blood pressure (mmHg)")
    presses = {"first": 0, "second": 0}
    run = {"now": "first"}
    jobs: dict[str, dict] = {}
    left: dict[str, str] = {}
    original_fit, original_submit = arms.Local.fit, JobRunner.submit
    original_exit = arms.Local.__exit__

    def counted(self, wait=3600.0):
        presses[run["now"]] += 1
        return original_fit(self, wait)

    def submitted(self, fn, *args, **kw):
        from turbotab.core import estimand

        jid = original_submit(self, fn, *args, **kw)
        spec = args[0] if args and isinstance(args[0], dict) else {}
        models = list(getattr(spec.get("state"), "models", None) or [])
        jobs[jid] = {"project": spec.get("project_id"), "stage": kw.get("stage"),
                     "asleep": models == [SLOW] and kw.get("stage") in estimand.ESTIMATE_STAGES}
        return jid

    def closing(self, *exc):
        if run["now"] == "first":
            runner = self.client.c.app.state.service.runner
            for jid in jobs:
                try:
                    left[jid] = runner.get(jid).state
                except KeyError:  # a finished job the runner no longer remembers
                    left[jid] = "forgotten"
        return original_exit(self, *exc)

    mp = pytest.MonkeyPatch()
    mp.setattr(W, "REFIT_WAIT", WAIT)
    mp.setattr(graph, "run_stage_job", standin_job)
    mp.setattr(arms.Local, "fit", counted)
    mp.setattr(JobRunner, "submit", submitted)
    mp.setattr(arms.Local, "__exit__", closing)
    only = ["question:missing", "question:models"]
    try:
        report = W.crash("walking", work / "out", data=work / "walking.csv", question=question,
                         only=only)
        run["now"] = "second"
        again = W.crash("walking", work / "out", data=work / "walking.csv", question=question,
                        only=only)
    finally:
        mp.undo()
    return {"report": report, "again": again, "out": work / "out", "presses": presses,
            "jobs": jobs, "left": left}


def refits_of(report, point):
    return {r["option"]: r for r in report["refits"] if r["point"] == point}


@pytest.mark.slow
def test_a_refit_past_its_wait_is_cancelled_unreached_and_the_next_is_fitted_at_once(waited):
    """The refit under SLOW, its estimate stages asleep on both workers: its fit not done when the
    wait runs out, it is cancelled through the server's cancel, unreached with what it waited on
    and for how long, and its copy removed; the next refit is fitted at once and scored; no job is
    left queued or running, and each stand-in job the runner was given under SLOW was cancelled."""
    report = waited["report"]
    models = refits_of(report, "question:models")
    slow, after = models[SLOW], models[NEXT]
    assert slow["kind"] == "unreached", slow
    assert re.fullmatch(
        rf"The fit was (running|queued|stale) {WAIT:g} s after the sweep pressed Fit, as long as "
        rf"it waits for a refit \(still computing: [a-z_ ,]+\)\. The sweep cancelled the copy's "
        rf"fit(; its stages still computing [0-9.]+ s later, it cancelled their jobs too "
        rf"\([a-z_, ]+\))?; its jobs ended and the copy was removed\.", slow["why"]), slow["why"]
    assert {"stage": "fit", "status": "running"} in slow["cancelled"]["waiting_on"]
    assert slow["cancelled"]["after"] == WAIT and slow["cancelled"]["idle"] is True
    assert slow["project"] is None and slow["seconds"] >= WAIT
    assert {u["refit"] for u in report["unreached"]} >= {slow["id"]}
    assert after["kind"] == "scored" and after["seconds"] < WAIT, after
    assert after["estimates"], after
    # Every project left in the home is the template's, the plan's or a defect's.
    home = waited["out"] / "home" / "projects"
    kept = {report["template"]["project"], report["baseline"]["project"]} | {
        d["project"] for d in report["defects"]}
    assert {p.name for p in home.iterdir()} == kept
    # From the runner itself: no job queued or running when the server closed, and every stand-in
    # job (an estimate stage under SLOW) cancelled: it slept past the wait, so only a cancel ended
    # it.
    left = waited["left"]
    assert left and not [j for j, state in left.items() if state in ("queued", "running")]
    asleep = [j for j, job in waited["jobs"].items() if job["asleep"]]
    assert asleep and {left[j] for j in asleep} == {"cancelled"}, [
        (waited["jobs"][j], left[j]) for j in asleep]


@pytest.mark.slow
def test_an_untyped_error_in_a_stage_the_fit_does_not_wait_on_is_a_defect(waited):
    """Under complete cases the effects stage raises a bare ValueError in its worker after the fit
    is fresh: the refit is failed at the effects stage in its words and reported as an engine
    defect with its reproduction, its copy kept."""
    report = waited["report"]
    crashed = refits_of(report, "question:missing")[BARE_UNDER]
    assert crashed["kind"] == "failed", crashed
    f = crashed["failed"]
    assert (f["at"], f["stage"], f["code"]) == ("stage", "effects", "stage_failed")
    assert f["message"] == f"ValueError: {BARE_WORDS}" and f["option"] == BARE_UNDER
    (defect,) = [d for d in report["defects"] if d["refit"] == crashed["id"]]
    kept = json.loads(open(defect["path"]).read())
    effects = next(s for s in kept["stages"] if s["stage"] == "effects")
    assert effects["kind"] == "failed" and effects["error"] == f"ValueError: {BARE_WORDS}"
    assert "read the effects stage" in kept["reproduce"]
    assert (waited["out"] / "home" / "projects" / crashed["project"]).is_dir()
    assert report["passed"] is False


@pytest.mark.slow
def test_a_typed_refusal_in_a_stage_the_fit_does_not_wait_on_is_refused_with_its_code(waited):
    report = waited["report"]
    refused = refits_of(report, "question:models")[TYPED_UNDER]
    assert refused["kind"] == "refused", refused
    r = refused["refused"]
    assert (r["by"], r["at"], r["stage"], r["code"], r["message"]) == (
        "engine", "stage", "effects", "induced_refusal", TYPED_WORDS)
    assert r["option"] == TYPED_UNDER
    assert r["exits"] == [{"label": "Choose the linear model",
                           "decision": {"kind": "select_models", "models": ["linear"]}}]
    assert refused["id"] not in {d["refit"] for d in report["defects"]}
    models = next(d for d in report["decisions"] if d["point"] == "question:models")
    assert models["refused_codes"].get("induced_refusal") == 1


@pytest.mark.slow
def test_a_fitted_refit_with_no_estimate_stage_in_error_is_scored(waited):
    """Both ways: the plan and the next refit, no stand-in under either, are scored, neither
    refused nor failed nor cancelled, and neither is a defect."""
    report = waited["report"]
    plan, after = report["baseline"], refits_of(report, "question:models")[NEXT]
    for r in (plan, after):
        assert r["kind"] == "scored" and r["failed"] is None and r["refused"] is None, r
        assert r["cancelled"] is None
    assert not {d["refit"] for d in report["defects"]} & {plan["id"], after["id"]}


@pytest.mark.slow
def test_a_rerun_after_a_cancelled_refit_fits_nothing_the_first_run_finished(waited):
    first, again = waited["report"], waited["again"]
    assert waited["presses"]["first"] > 0 and waited["presses"]["second"] == 0
    assert [(r["id"], r["kind"], r["why"]) for r in again["refits"]] == [
        (r["id"], r["kind"], r["why"]) for r in first["refits"]]

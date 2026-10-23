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
its code, words and way forward, while every other option is neither."""
from __future__ import annotations

import json
import math

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

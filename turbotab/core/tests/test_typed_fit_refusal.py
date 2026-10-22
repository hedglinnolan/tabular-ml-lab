"""A typed fit-time refusal (SWARM_PLAN §8; the sixth review's §3 item 2): a stage that refuses in
words carries a code and its ways forward, as an answer-time refusal does, and the stage's status
serves them apart from a crash, so a script classes a refused fit without reading its prose.

Each guard both ways, on stages built here: a refusal in words is served with its code and exits
(a light stage, a heavy one in a job worker, and a stage waiting on a refused one); a crash, the
same ``ValueError`` text with no refusal, is served with none. The fit's own refusals are held to
their construction: an empty selection (columns of a Hadamard matrix, exactly uncorrelated with
the outcome, so nothing is kept) refuses with ``nothing_kept`` and the two selections the words
name; the design stage's refusals in words keep their words and gain a code. A selection's
fit-time ways forward are the recorded answer with only its method changed (its pre-selection
answer kept), as the answer-time refusal builds them, wherever the stage ran; with no answer
recorded, as raised."""
from __future__ import annotations

import pickle
import uuid

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import hadamard

from turbotab.core import decisions as d
from turbotab.core.graph import Engine, Graph, Stage, StageRefusal, StageStatus
from turbotab.core.tests.test_graph import Projects, RecordingBus, wait_for
from turbotab.core.tests.toy_stages import ToyState

GRAPH = "turbotab.core.tests.test_typed_fit_refusal:refusing_graph"
WORDS = "The toy refuses in words: choose no selection."
EXITS = [{"label": "No selection", "decision": {"kind": "set_selection", "method": "none"}},
         {"label": "Wait for the table", "decision": None}]


def stage_refuses(ctx):
    raise StageRefusal("toy_refused", WORDS, exits=EXITS)


def stage_crashes(ctx):
    raise ValueError(WORDS)  # the same words, untyped: a crash


def stage_after(ctx):
    return {"after": ctx.inputs}


def refusing_graph() -> Graph:
    """R refuses (light), RH refuses in a job worker, C and CH fail with the same words untyped,
    AR waits on RH, AC on CH."""
    return Graph([
        Stage("R", 1, (), (), stage_refuses),
        Stage("RH", 1, (), (), stage_refuses, heavy=True),
        Stage("C", 1, (), (), stage_crashes),
        Stage("CH", 1, (), (), stage_crashes, heavy=True),
        Stage("AR", 1, ("RH",), (), stage_after),
        Stage("AC", 1, ("CH",), (), stage_after),
    ])


def job_refuses() -> None:
    raise StageRefusal("toy_refused", WORDS, exits=EXITS)


def job_crashes() -> None:
    raise ValueError(WORDS)


def normalized(exits):
    """Each way forward as an answer-time refusal stores it: its decision parsed by the decision
    model and dumped as JSON (``decisions.Refusal``)."""
    return [{"label": e["label"], "decision": None if e["decision"] is None
             else d.parse_decision(e["decision"]).model_dump(mode="json")} for e in exits]


@pytest.fixture
def statuses(runner, tmp_path):
    projects, bus = Projects(tmp_path), RecordingBus()
    engine = Engine(graph_factory=GRAPH, runner=runner, bus=bus, project_ctx=projects)
    pid = uuid.uuid4().hex[:8]
    projects.states[pid] = ToyState()
    engine.on_decision(pid)
    try:
        yield wait_for(engine, pid, lambda s: all(x.status == "error" for x in s.values()),
                       timeout=30)
    finally:
        engine.shutdown()


@pytest.mark.parametrize("stage", ["R", "RH"])
def test_a_refusal_in_words_is_served_with_its_code_and_exits(statuses, stage):
    """Light or in a job worker: the error text is the one a ValueError with these words gives
    (what a person reads does not move), and the refusal is served beside it."""
    status = statuses[stage]
    assert status.error == f"ValueError: {WORDS}"
    assert status.refusal is not None
    assert status.refusal.model_dump(mode="json") == {
        "stage": stage, "code": "toy_refused", "message": WORDS, "exits": normalized(EXITS)}


@pytest.mark.parametrize("stage", ["C", "CH"])
def test_a_crash_with_the_same_words_is_served_with_no_refusal(statuses, stage):
    status = statuses[stage]
    assert status.error == f"ValueError: {WORDS}"
    assert status.refusal is None


def test_a_stage_waiting_on_a_refused_one_serves_that_refusal_and_on_a_crash_none(statuses):
    assert statuses["AR"].error == "Needs 'RH', which failed."
    assert statuses["AR"].refusal == statuses["RH"].refusal
    assert statuses["AR"].refusal.stage == "RH"
    assert statuses["AC"].error == "Needs 'CH', which failed."
    assert statuses["AC"].refusal is None


def test_the_status_without_a_refusal_serves_none():
    assert StageStatus(stage="s", status="fresh").model_dump(mode="json")["refusal"] is None


def test_a_job_that_refuses_reports_the_refusal_and_one_that_crashes_none(runner):
    """The job pipe carries the refusal whole (its code, words and exits); a crash carries none."""
    import time

    ids = {"refuses": runner.submit(job_refuses, label="refuses"),
           "crashes": runner.submit(job_crashes, label="crashes")}
    end = time.monotonic() + 30
    views = {}
    while time.monotonic() < end and len(views) < 2:
        for name, job in ids.items():
            view = runner.get(job)
            if view.state in ("done", "error", "cancelled"):
                views[name] = view
        time.sleep(0.05)
    assert views["refuses"].state == views["crashes"].state == "error"
    assert views["refuses"].error == views["crashes"].error == f"ValueError: {WORDS}"
    assert views["refuses"].refusal == {"code": "toy_refused", "message": WORDS,
                                        "exits": normalized(EXITS)}
    assert views["crashes"].refusal is None


def test_a_refusal_is_a_value_error_with_its_words_and_pickles_whole():
    refusal = StageRefusal("toy_refused", WORDS, exits=EXITS)
    assert isinstance(refusal, ValueError) and str(refusal) == WORDS
    again = pickle.loads(pickle.dumps(refusal))
    assert (again.code, again.message, again.exits) == ("toy_refused", WORDS, normalized(EXITS))


def test_a_way_forward_that_is_no_decision_is_refused_when_raised():
    with pytest.raises(Exception):
        StageRefusal("toy_refused", WORDS, exits=[{"label": "x", "decision": {"kind": "no_such"}}])


# ── the fit's own refusals ───────────────────────────────────────────────────

N = 64


def orthogonal(carried: bool) -> tuple[pd.DataFrame, pd.Series]:
    """Six candidates exactly uncorrelated with the outcome (columns of a Hadamard matrix); with
    ``carried``, the first carries the outcome."""
    H = hadamard(N).astype(float)
    y = pd.Series(H[:, 1], name="y")
    X = pd.DataFrame(H[:, 2:8], columns=[f"x{i}" for i in range(6)])
    if carried:
        X["x0"] = H[:, 1] + 0.5 * H[:, 9]
    return X, y


def selector(method: str, columns: list[str]):
    from turbotab.core.models.variable_selection import Selector

    return Selector(method, "regression", None, None, None, 50, 5, 0, None, columns)


@pytest.mark.parametrize("method", ["univariable", "elastic_net", "stepwise"])
def test_a_selection_that_keeps_nothing_refuses_with_nothing_kept_and_its_two_ways_forward(method):
    from turbotab.core.models.variable_selection import LABELS, nothing_kept

    X, y = orthogonal(carried=False)
    assert np.allclose(X.T.to_numpy() @ (y - y.mean()).to_numpy(), 0)  # the construction
    fitted = selector(method, list(X.columns)).fit(X, y)
    with pytest.raises(StageRefusal) as refused:
        fitted.transform(X)
    r = refused.value
    assert r.code == "nothing_kept" and str(r) == nothing_kept(method, N)  # the words as before
    # The words name No selection and in-fold screening; the exits are those two selections.
    assert r.exits == normalized([
        {"label": LABELS["none"], "decision": {"kind": "set_selection", "method": "none"}},
        {"label": LABELS["screening"], "decision": {"kind": "set_selection",
                                                    "method": "screening"}}])


@pytest.mark.parametrize("method", ["univariable", "elastic_net", "stepwise"])
def test_a_selection_that_keeps_a_predictor_refuses_nothing(method):
    X, y = orthogonal(carried=True)
    kept = selector(method, list(X.columns)).fit(X, y).transform(X)
    assert "x0" in kept.columns


def _design(tmp_path, st, rows=None, frame=None):
    from turbotab.core.stages.modeling import design_stage
    from turbotab.core.tests import modeling_fixtures as mf

    frame = frame if frame is not None else mf.nhanes_like(60, seed=1)
    paths = mf.ingest_frame(frame, tmp_path)
    ids = np.arange(len(frame)) if rows is None else rows
    return design_stage(mf.context(st, {"split": mf.split_bundle(ids),
                                        "target_info": mf.target_info("regression")}, paths))


def test_a_design_handed_no_rows_refuses_with_a_code_and_its_words(tmp_path):
    from turbotab.core.tests import modeling_fixtures as mf

    with pytest.raises(StageRefusal) as refused:
        _design(tmp_path, mf.state(models=["linear"]), rows=np.array([], dtype=np.int64))
    assert refused.value.code == "no_rows" and refused.value.exits == []
    assert str(refused.value).startswith("There are no training rows to build the models on")


def test_a_design_with_no_predictor_refuses_with_a_code(tmp_path):
    from turbotab.core.tests import modeling_fixtures as mf

    roles = {c: ("identifier" if c == "SEQN" else "excluded") for c in mf.NHANES_ROLES}
    with pytest.raises(StageRefusal) as refused:
        _design(tmp_path, mf.state(models=["linear"], roles=roles))
    assert refused.value.code == "no_predictors"
    assert "nothing to model the outcome with" in str(refused.value)


def test_a_design_whose_outcome_takes_one_value_refuses_with_a_code(tmp_path):
    from turbotab.core.tests import modeling_fixtures as mf

    frame = mf.nhanes_like(60, seed=1)
    frame["glucose"] = 100.0
    with pytest.raises(StageRefusal) as refused:
        _design(tmp_path, mf.state(models=["linear"]), frame=frame)
    assert refused.value.code == "one_outcome_value"
    assert "in every one of the" in str(refused.value)


def test_a_design_that_can_be_built_refuses_nothing(tmp_path):
    from turbotab.core.tests import modeling_fixtures as mf

    out = _design(tmp_path, mf.state(models=["linear"]))
    assert out is not None


def test_a_batch_confounded_with_the_outcome_is_typed_as_the_answer_refuses_it():
    """The design's backstop says the answer-time refusal's words (``confounding_refusal``); typed,
    it carries that refusal's code and exits, found by its words among the table's columns."""
    from turbotab.core.methods.batch import confounding_refusal
    from turbotab.core.stages.modeling import _confounding_typed

    answer = confounding_refusal("plate", "y")
    typed = _confounding_typed(answer.message, ["age", "plate", "y"], "y")
    assert (typed.code, typed.message, typed.exits) == (answer.code, answer.message, answer.exits)
    assert _confounding_typed(None, ["age", "plate", "y"], "y") is None


def test_a_grouping_too_small_for_intercepts_is_typed_with_the_cluster_only_way_forward():
    from turbotab.core.stages.modeling import _small_groups_typed

    st = d.ProjectState(purpose="inference", target="y",
                        clusters=d.ClusterSpec(column="site", adjust="fixed_effects"))
    typed = _small_groups_typed("too few rows per group", st)
    assert typed.code == "incidental_parameters"
    assert typed.exits == normalized([{
        "label": "Cluster the intervals by `site` only",
        "decision": {"kind": "set_clusters", "column": "site", "adjust": "cluster_only"}}])
    # Under prediction no grouping gets intercepts: the refusal names no way forward of that kind.
    plain = _small_groups_typed("too few rows per group",
                                st.model_copy(update={"purpose": "prediction"}))
    assert plain.code == "incidental_parameters" and plain.exits == []


# ── stepwise's in-fit backstop (the verifier's P8) ───────────────────────────


def wide(rows: int, columns: int) -> tuple[pd.DataFrame, pd.Series]:
    """``columns`` candidates on ``rows`` rows, drawn here: backward elimination starts from all of
    them, which an intercept and every column can be fit to only when columns < rows − 1."""
    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.normal(size=(rows, columns)), columns=[f"x{i}" for i in range(columns)])
    return X, pd.Series(X["x0"] + rng.normal(size=rows), name="y")


def answer_time_stepwise_refusal(monkeypatch, columns: int, rows: int):
    """The refusal the selection answer gives where the smallest training fold has ``rows`` rows
    for ``columns`` candidates (``_stepwise_has_rows``, its room read as given)."""
    from turbotab.core.models import variable_selection as vs

    monkeypatch.setattr(vs, "stepwise_room", lambda *a, **k: (columns, rows))
    with pytest.raises(d.Refusal) as refused:
        vs._stepwise_has_rows(d.SetSelection(method="stepwise"), None, None, "regression")
    return refused.value


def test_stepwise_with_more_columns_than_its_rows_allow_refuses_typed_as_the_answer_does(
        monkeypatch):
    """10 candidates on 8 training rows: backward elimination cannot start (10 ≥ 8 − 1). The fit
    refuses with the answer-time code and its three ways forward (elastic net, screening, no
    selection), and its words are the ones a person read before."""
    X, y = wide(8, 10)
    with pytest.raises(StageRefusal) as refused:
        selector("stepwise", list(X.columns)).fit(X, y)
    r = refused.value
    answer = answer_time_stepwise_refusal(monkeypatch, 10, 8)
    assert r.code == answer.code == "stepwise_needs_rows"
    assert [(e["label"], e["decision"]["kind"], e["decision"]["method"]) for e in r.exits] == [
        (e["label"], e["decision"]["kind"], e["decision"]["method"]) for e in answer.exits]
    assert [e["decision"]["method"] for e in r.exits] == ["elastic_net", "screening", "none"]
    assert str(r) == ("Backward elimination starts from all 10 candidate columns, and these "
                      "training rows number 8: a model with every column and an intercept needs "
                      "more rows than that. Choose the elastic net or in-fold screening in the "
                      "selection question.")
    from turbotab.core.jobs import failure_of

    text, refusal = failure_of(r)
    assert text == f"ValueError: {r}" and refusal["code"] == "stepwise_needs_rows"


def test_stepwise_with_room_for_every_column_refuses_nothing():
    """6 candidates on 64 rows (6 < 64 − 1): the backstop stays silent and the fit runs."""
    X, y = wide(64, 6)
    fitted = selector("stepwise", list(X.columns)).fit(X, y)
    assert "x0" in fitted.transform(X).columns


# ── a fit-time way forward keeps the recorded selection (the verifier's V3) ──

RECORDED = d.SelectionSpec(method="stepwise", pre_selected="no", keep=3)


def answer_time_exits(monkeypatch, recorded: d.SelectionSpec):
    """The ways forward the selection answer's own refusal gives for ``recorded`` (each the
    answer with only its method changed, ``decision.model_copy``), as a refusal stores them."""
    from turbotab.core.models import variable_selection as vs

    monkeypatch.setattr(vs, "stepwise_room", lambda *a, **k: (10, 8))
    answer = d.SetSelection(**recorded.model_dump())
    with pytest.raises(d.Refusal) as refused:
        vs._stepwise_has_rows(answer, None, None, "regression")
    return [{"label": e["label"], "decision": d.parse_decision(
        e["decision"].model_dump(mode="json") if hasattr(e["decision"], "model_dump")
        else e["decision"]).model_dump(mode="json")} for e in refused.value.exits]


def stepwise_backstop() -> StageRefusal:
    X, y = wide(8, 10)
    with pytest.raises(StageRefusal) as refused:
        selector("stepwise", list(X.columns)).fit(X, y)
    return refused.value


def test_the_stepwise_backstops_ways_forward_are_the_recorded_answers_as_the_answer_gives_them(
        monkeypatch):
    """The recorded selection answered pre_selected "no" (TRIPOD+AI 9a) and keep 3: each way
    forward on that state is that answer with only its method changed, labeled and recorded as
    the answer-time refusal's ways forward are."""
    r = stepwise_backstop().on_answers(d.ProjectState(purpose="prediction", selection=RECORDED))
    assert r.exits == answer_time_exits(monkeypatch, RECORDED)
    assert [(e["decision"]["method"], e["decision"]["pre_selected"], e["decision"]["keep"])
            for e in r.exits] == [("elastic_net", "no", 3), ("screening", "no", 3),
                                  ("none", "no", 3)]


def test_nothing_kepts_ways_forward_keep_the_recorded_answer_but_its_method():
    from turbotab.core.models.variable_selection import LABELS

    recorded = d.SelectionSpec(method="univariable", threshold=0.01, pre_selected="other_data")
    X, y = orthogonal(carried=False)
    fitted = selector("univariable", list(X.columns)).fit(X, y)
    with pytest.raises(StageRefusal) as refused:
        fitted.transform(X)
    r = refused.value.on_answers(d.ProjectState(purpose="prediction", selection=recorded))
    assert r.exits == normalized([
        {"label": LABELS[m], "decision": {**recorded.model_dump(mode="json"),
                                          "kind": "set_selection", "method": m}}
        for m in ("none", "screening")])


def test_with_no_selection_recorded_or_a_refusal_that_changes_none_the_ways_forward_stay():
    """Where the state records no selection the ways forward are as raised (the method alone, on
    the model's defaults); a refusal whose ways forward are not changes of an answer (the toy's)
    is the same on a state that records one."""
    raised = stepwise_backstop()
    before = [dict(e) for e in raised.exits]
    assert raised.on_answers(d.ProjectState(purpose="prediction")).exits == before
    assert raised.on_answers(None).exits == before
    toy = StageRefusal("toy_refused", WORDS, exits=EXITS)
    assert toy.on_answers(d.ProjectState(selection=RECORDED)).exits == normalized(EXITS)


def test_a_refusal_whose_ways_forward_were_made_the_recorded_answers_pickles_whole():
    r = stepwise_backstop().on_answers(d.ProjectState(purpose="prediction", selection=RECORDED))
    again = pickle.loads(pickle.dumps(r))
    assert (again.code, again.message, again.exits) == (r.code, r.message, r.exits)


SELECTING = "turbotab.core.tests.test_typed_fit_refusal:selecting_graph"


def stage_selects(ctx):
    X, y = wide(8, 10)
    selector("stepwise", list(X.columns)).fit(X, y)  # the backstop refuses
    return {}


def selecting_graph() -> Graph:
    """S runs the stepwise backstop on this engine's threads, SH in a job worker."""
    return Graph([Stage("S", 1, (), (), stage_selects),
                  Stage("SH", 1, (), (), stage_selects, heavy=True)])


@pytest.mark.parametrize("stage", ["S", "SH"])
def test_the_served_ways_forward_keep_the_selection_the_fit_ran_with(runner, tmp_path, stage):
    """Light or in a job worker, the status serves each way forward as the project's recorded
    selection with only its method changed; a project with none recorded is served them as
    raised."""
    projects, bus = Projects(tmp_path), RecordingBus()
    engine = Engine(graph_factory=SELECTING, runner=runner, bus=bus, project_ctx=projects)
    recorded, bare = uuid.uuid4().hex[:8], uuid.uuid4().hex[:8]
    projects.states[recorded] = ToyState(purpose="prediction", selection=RECORDED)
    projects.states[bare] = ToyState(purpose="prediction")
    try:
        served = {}
        for pid in (recorded, bare):
            engine.on_decision(pid)
            served[pid] = wait_for(engine, pid, lambda s: all(x.status == "error"
                                                              for x in s.values()), timeout=30)
    finally:
        engine.shutdown()
    kept = served[recorded][stage].refusal
    assert kept is not None and kept.code == "stepwise_needs_rows"
    assert [(e["decision"]["method"], e["decision"]["pre_selected"]) for e in kept.exits] == [
        ("elastic_net", "no"), ("screening", "no"), ("none", "no")]
    raised = served[bare][stage].refusal
    assert [e["decision"]["pre_selected"] for e in raised.exits] == [None, None, None]
    assert [e["label"] for e in kept.exits] == [e["label"] for e in raised.exits]
